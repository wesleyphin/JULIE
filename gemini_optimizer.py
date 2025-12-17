import pandas as pd
import requests
import json
import logging
import datetime
import numpy as np
from config import CONFIG

class GeminiSessionOptimizer:
    def __init__(self):
        self.config = CONFIG.get('GEMINI', {})
        self.api_key = self.config.get('api_key', '')
        self.model = "gemini-3-pro-preview"
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        self.headers = {"Content-Type": "application/json"}
        self.csv_path = 'es_2023_2024_2025.csv'

    def _load_historical_data(self):
        """Loads and cleans the historical CSV."""
        try:
            # FIX 1: Added low_memory=False to suppress DtypeWarning
            df = pd.read_csv(self.csv_path, thousands=',', low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]

            date_col = next((c for c in df.columns if 'date' in c), None)
            if not date_col: return None

            df['timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df.set_index('timestamp', inplace=True)

            # Numeric cleanup
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    # Convert to string first to handle mixed types safely
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logging.error(f"Error loading historical CSV: {e}")
            return None

    def _calculate_adx(self, df, period=14):
        """Calculates Average Directional Index to measure Trend Strength (0-100)."""
        if df.empty: return 0
        df = df.copy()

        # True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)

        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']

        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

        # Smoothed
        tr_smooth = df['tr'].rolling(period).sum()
        plus_di = 100 * (df['plus_dm'].rolling(period).sum() / tr_smooth)
        minus_di = 100 * (df['minus_dm'].rolling(period).sum() / tr_smooth)

        # Avoid division by zero
        sum_di = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / sum_di.replace(0, 1)

        adx = dx.rolling(period).mean().iloc[-1]

        return round(adx, 2)

    def _calculate_market_profile(self, df):
        """Identifies Value Area (70% of volume) and Point of Control."""
        if df.empty: return {}

        tick_size = 0.25
        min_price = df['low'].min()
        max_price = df['high'].max()

        # Create bins
        bins = np.arange(min_price, max_price + tick_size, tick_size)

        # FIX 2: Added observed=False to suppress FutureWarning
        volume_profile = df.groupby(pd.cut(df['close'], bins), observed=False)['volume'].sum()

        # Find POC
        poc_bin = volume_profile.idxmax()
        poc_price = poc_bin.mid

        # Calculate Value Area (70%)
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.70

        sorted_vol = volume_profile.sort_values(ascending=False)
        cumulative_vol = sorted_vol.cumsum()

        value_area_bins = sorted_vol[cumulative_vol <= target_volume].index

        vah = max([b.right for b in value_area_bins])
        val = min([b.left for b in value_area_bins])

        return {
            "POC": poc_price,
            "VAH": vah,
            "VAL": val,
            "current_price": df.iloc[-1]['close']
        }

    def optimize_new_session(self, master_df, session_name, events_data, base_sl, base_tp):
        """
        Main Optimization Routine with Advanced Metrics.
        """
        logging.info(f"🧠 Gemini 3.0: Analyzing Market Structure & Sentiment...")

        # --- 1. PROCESS CURRENT DATA ---
        if master_df.empty: return None

        # Last 13 Days Window
        last_time = master_df.index.max()
        start_time_13d = last_time - datetime.timedelta(days=13)
        current_window = master_df[master_df.index >= start_time_13d]

        if current_window.empty:
            logging.warning("Insufficient data for window analysis")
            return None

        # A. Basic Volatility
        avg_range = round((current_window['high'] - current_window['low']).mean(), 2)

        # B. Advanced Trend Strength (ADX)
        # Resample to 15min for cleaner ADX calculation
        df_15m = current_window.resample('15min').agg({'high':'max', 'low':'min', 'close':'last'}).dropna()
        adx_score = self._calculate_adx(df_15m)
        trend_status = "TRENDING" if adx_score > 25 else "CHOPPY/RANGING"

        # C. Market Profile (Value Area)
        profile = self._calculate_market_profile(current_window)
        curr_price = profile.get('current_price', 0)
        vah = profile.get('VAH', 0)
        val = profile.get('VAL', 0)

        profile_status = "INSIDE VALUE (Balance)"
        if curr_price > vah: profile_status = "ABOVE VALUE (Bullish Imbalance)"
        elif curr_price < val: profile_status = "BELOW VALUE (Bearish Imbalance)"

        # --- 2. HISTORICAL CONTEXT ---
        hist_df = self._load_historical_data()
        hist_context = "Historical Data Unavailable"
        if hist_df is not None:
            mask = (
                (hist_df.index.month == last_time.month) &
                (hist_df.index.day >= start_time_13d.day) &
                (hist_df.index.day <= last_time.day)
            )
            hist_window = hist_df[mask]
            if not hist_window.empty:
                hist_avg_range = round((hist_window['high'] - hist_window['low']).mean(), 2)
                hist_context = f"Historical Avg Range: {hist_avg_range} pts"

        # --- 3. CONSTRUCT ADVANCED PROMPT ---
        system_instruction = (
            f"You are a Quantitative Risk Manager optimizing for the {session_name} session. "
            "Use Market Structure (Value Area), Trend Strength (ADX), and News to adjust TP/SL.\n"
            "Rules:\n"
            "- High ADX (>30) + Imbalance: Widen TP significantly (Trend Following).\n"
            "- Low ADX (<20) + Inside Value: Tighten TP, Widen SL slightly (Mean Reversion).\n"
            "- High Impact Events: Maximize SL (Volatility Protection)."
        )

        user_prompt = (
            f"**SESSION: {session_name}**\n\n"
            f"=== MARKET STRUCTURE (Last 13 Days) ===\n"
            f"Trend Strength (ADX): {adx_score} ({trend_status})\n"
            f"Market Profile: {profile_status}\n"
            f"Price Location: {curr_price} (VAH: {vah} | VAL: {val})\n\n"

            f"=== VOLATILITY REGIME ===\n"
            f"Current Avg Range: {avg_range} pts\n"
            f"Historical Norm: {hist_context}\n\n"

            f"=== NEWS EVENTS ===\n"
            f"{events_data}\n\n"

            f"=== BASE PARAMETERS ===\n"
            f"SL: {base_sl} | TP: {base_tp}\n\n"

            "**OPTIMIZATION TASK:**\n"
            "Output STRICT JSON:\n"
            "{\n"
            '  "session": "' + session_name + '",\n'
            '  "regime": "TREND" or "RANGE" or "BREAKOUT",\n'
            '  "sl_multiplier": number (e.g. 1.1),\n'
            '  "tp_multiplier": number (e.g. 1.5),\n'
            '  "reasoning": "Explain using ADX and Value Area"\n'
            "}"
        )

        # --- 4. EXECUTE ---
        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=45)
            if response.status_code == 200:
                return json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])
            return None
        except Exception as e:
            logging.error(f"Gemini Optimization Error: {e}")
            return None
