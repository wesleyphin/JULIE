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

    def _slice_dataframe_by_session(self, df, session_name):
        """
        Returns a subset of the dataframe containing ONLY rows that fall
        within the specified session's trading hours.
        This enables "like-for-like" comparison of session behavior.
        """
        session_hours = CONFIG.get('SESSIONS', {}).get(session_name, {}).get('HOURS', [])

        if not session_hours or df.empty:
            return df  # Fallback to full df

        # Filter: Keep only rows where the hour is in the session definition
        # Ensure df index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logging.warning("DataFrame index is not DatetimeIndex, cannot slice by session")
            return df

        mask = df.index.hour.isin(session_hours)
        return df[mask]

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

    def optimize_new_session(self, master_df, session_name, events_data, base_sl, base_tp,
                             structure_context=""):
        """
        Main Optimization Routine with Session-Aligned Metrics.
        Uses "like-for-like" comparison: analyzes only the same session hours
        from the past 13 days for accurate context.
        """
        logging.info(f"🧠 Gemini 3.0: Analyzing Session-Aligned Context for {session_name}...")

        if master_df.empty:
            return None

        # --- STEP 1: TIME SLICING (Session-Aligned Analysis) ---
        session_aligned_df = self._slice_dataframe_by_session(master_df, session_name)

        if session_aligned_df.empty:
            logging.warning(f"No history found for session {session_name}")
            return None

        logging.info(f"📊 Session-aligned data: {len(session_aligned_df)} bars for {session_name}")

        # --- STEP 2: CALCULATE METRICS ---

        # Calculate Base Risk:Reward Ratio
        base_rr = base_tp / base_sl if base_sl > 0 else 0.0

        # A. Volatility
        daily_session_stats = session_aligned_df.resample('D').agg({
            'high': 'max', 'low': 'min', 'volume': 'sum'
        }).dropna()
        daily_session_stats['range'] = daily_session_stats['high'] - daily_session_stats['low']
        avg_session_range = round(daily_session_stats['range'].mean(), 2) if not daily_session_stats.empty else 0
        num_session_days = len(daily_session_stats)

        # B. Volume Profile
        avg_session_volume = int(daily_session_stats['volume'].mean()) if not daily_session_stats.empty else 0

        # C. ADX
        df_15m = session_aligned_df.resample('15min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        adx_score = self._calculate_adx(df_15m) if not df_15m.empty else 0
        trend_status = "TRENDING" if adx_score > 25 else "CHOPPY/RANGING"

        # D. Market Profile
        profile = self._calculate_market_profile(session_aligned_df)
        curr_price = profile.get('current_price', 0)
        vah = profile.get('VAH', 0)
        val = profile.get('VAL', 0)
        profile_status = "INSIDE VALUE (Balance)"
        if curr_price > vah: profile_status = "ABOVE VALUE (Bullish Imbalance)"
        elif curr_price < val: profile_status = "BELOW VALUE (Bearish Imbalance)"

        # --- STEP 3: HISTORICAL CONTEXT ---
        hist_df = self._load_historical_data()
        hist_session_range = "N/A"
        hist_context = "Historical Data Unavailable"

        if hist_df is not None:
            hist_session_df = self._slice_dataframe_by_session(hist_df, session_name)
            if not hist_session_df.empty:
                hist_daily_stats = hist_session_df.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
                hist_daily_stats['range'] = hist_daily_stats['high'] - hist_daily_stats['low']
                hist_session_range = round(hist_daily_stats['range'].mean(), 2)
                hist_context = f"{hist_session_range} pts (from {len(hist_daily_stats)} historical sessions)"

        # --- STEP 4: CONSTRUCT ADVANCED PROMPT ---
        system_instruction = (
            f"You are a Quantitative Risk Manager optimizing for the {session_name} session.\n"
            "Use Market Structure (S/R, FVGs), Trend Strength (ADX), and News to adjust TP/SL.\n\n"
            "CRITICAL RULES:\n"
            "- **CHECK BASE RR:** If Base RR is already High (>3.0), do NOT widen TP multiplier significantly. Prioritize hitting the target over greed.\n"
            "- **EXTREME RR (>5.0):** If RR is massive, keep TP Multiplier <= 1.0 unless ADX > 50 (Parabolic).\n"
            "- High ADX + Open Space: Widen TP significantly only if RR < 3.0.\n"
            "- Price sandwiched between S/R: Tighten TP, reduce SL multiplier.\n"
            "- High Impact Events: Maximize SL (Volatility Protection).\n"
        )

        user_prompt = (
            f"**OPTIMIZATION REQUEST FOR: {session_name}**\n\n"

            f"=== BASE PARAMETERS ===\n"
            f"SL: {base_sl} | TP: {base_tp}\n"
            f"BASE RR: {base_rr:.2f}R (Reward/Risk Ratio)\n\n"

            f"=== MARKET STRUCTURE ===\n{structure_context}\n\n"

            f"=== SESSION CONTEXT (Vs. Past {num_session_days} Sessions) ===\n"
            f"Avg Session Range: {avg_session_range} pts\n"
            f"Recent Trend: {adx_score} ({trend_status})\n"
            f"Market Profile: {profile_status}\n\n"

            f"=== HISTORICAL CONTEXT ===\n"
            f"Long-term Avg Range: {hist_context}\n\n"

            f"=== NEWS EVENTS ===\n{events_data}\n\n"

            "**TASK:**\n"
            "Provide TP/SL multipliers. "
            "If Base RR is high, be conservative with TP multiplier to ensure fill."
            "\nOutput STRICT JSON: {session, regime, sl_multiplier, tp_multiplier, reasoning}"
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
                # 1. Parse Response
                data = json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])

                raw_sl = float(data.get('sl_multiplier', 1.0))
                raw_tp = float(data.get('tp_multiplier', 1.0))

                # --- 2. DYNAMIC RR-BASED GUARDRAILS ---

                # If RR is already > 4.0, cap TP increase to 10% max (1.1x)
                # This prevents turning a 25pt target into an impossible 37.5pt target
                rr_tp_cap = 1.1 if base_rr > 4.0 else 1.5

                # Apply the limits
                # Floor of 0.5 to prevent stop-outs on entry
                data['sl_multiplier'] = max(0.5, min(raw_sl, 1.5))
                data['tp_multiplier'] = max(0.5, min(raw_tp, rr_tp_cap))

                if raw_tp > rr_tp_cap:
                    logging.warning(f"⚠️ High RR ({base_rr:.1f}R) detected. Capped TP Mult: {raw_tp} -> {data['tp_multiplier']}")

                return data
            return None
        except Exception as e:
            logging.error(f"Gemini Optimization Error: {e}")
            return None
