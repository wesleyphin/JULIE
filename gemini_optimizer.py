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
        """
        session_hours = CONFIG.get('SESSIONS', {}).get(session_name, {}).get('HOURS', [])

        if not session_hours or df.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            logging.warning("DataFrame index is not DatetimeIndex, cannot slice by session")
            return df

        mask = df.index.hour.isin(session_hours)
        return df[mask]

    def _load_historical_data(self):
        try:
            df = pd.read_csv(self.csv_path, thousands=',', low_memory=False)
            df.columns = [c.strip().lower() for c in df.columns]

            date_col = next((c for c in df.columns if 'date' in c), None)
            if not date_col: return None

            df['timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df.set_index('timestamp', inplace=True)

            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logging.error(f"Error loading historical CSV: {e}")
            return None

    def _calculate_adx(self, df, period=14):
        if df.empty: return 0
        df = df.copy()
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)

        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']

        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

        tr_smooth = df['tr'].rolling(period).sum()
        plus_di = 100 * (df['plus_dm'].rolling(period).sum() / tr_smooth)
        minus_di = 100 * (df['minus_dm'].rolling(period).sum() / tr_smooth)
        sum_di = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / sum_di.replace(0, 1)

        adx = dx.rolling(period).mean().iloc[-1]
        return round(adx, 2)

    def _calculate_market_profile(self, df):
        if df.empty: return {}
        tick_size = 0.25
        min_price = df['low'].min()
        max_price = df['high'].max()
        bins = np.arange(min_price, max_price + tick_size, tick_size)
        volume_profile = df.groupby(pd.cut(df['close'], bins), observed=False)['volume'].sum()

        poc_bin = volume_profile.idxmax()
        poc_price = poc_bin.mid
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.70
        sorted_vol = volume_profile.sort_values(ascending=False)
        cumulative_vol = sorted_vol.cumsum()
        value_area_bins = sorted_vol[cumulative_vol <= target_volume].index

        vah = max([b.right for b in value_area_bins])
        val = min([b.left for b in value_area_bins])

        return {"POC": poc_price, "VAH": vah, "VAL": val, "current_price": df.iloc[-1]['close']}

    def optimize_new_session(self, master_df, session_name, events_data, base_sl, base_tp, structure_context=""):
        logging.info(f"🧠 Gemini 3.0: Analyzing Session-Aligned Context for {session_name}...")

        if master_df.empty: return None

        # --- STEP 1: TIME SLICING ---
        session_aligned_df = self._slice_dataframe_by_session(master_df, session_name)
        if session_aligned_df.empty: return None

        # --- STEP 2: CALCULATE METRICS ---

        # Calculate Base RR
        base_rr = base_tp / base_sl if base_sl > 0 else 0.0

        daily_session_stats = session_aligned_df.resample('D').agg({'high': 'max', 'low': 'min', 'volume': 'sum'}).dropna()
        daily_session_stats['range'] = daily_session_stats['high'] - daily_session_stats['low']
        avg_session_range = round(daily_session_stats['range'].mean(), 2) if not daily_session_stats.empty else 0
        avg_session_volume = int(daily_session_stats['volume'].mean()) if not daily_session_stats.empty else 0
        num_session_days = len(daily_session_stats)

        df_15m = session_aligned_df.resample('15min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        adx_score = self._calculate_adx(df_15m) if not df_15m.empty else 0
        trend_status = "TRENDING" if adx_score > 25 else "CHOPPY/RANGING"

        profile = self._calculate_market_profile(session_aligned_df)
        curr_price = profile.get('current_price', 0)
        vah = profile.get('VAH', 0)
        val = profile.get('VAL', 0)
        profile_status = "INSIDE VALUE"
        if curr_price > vah: profile_status = "ABOVE VALUE"
        elif curr_price < val: profile_status = "BELOW VALUE"

        # --- STEP 3: HISTORICAL CONTEXT ---
        hist_df = self._load_historical_data()
        hist_context = "Unavailable"
        if hist_df is not None:
            hist_session_df = self._slice_dataframe_by_session(hist_df, session_name)
            if not hist_session_df.empty:
                hist_daily_stats = hist_session_df.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
                hist_daily_stats['range'] = hist_daily_stats['high'] - hist_daily_stats['low']
                hist_context = f"{round(hist_daily_stats['range'].mean(), 2)} pts"

        # --- STEP 4: PROMPT ---
        system_instruction = (
            f"You are a Risk Manager optimizing {session_name}. Adjust TP/SL & Trend Filters based on Context.\n"
            "CRITICAL RULES:\n"
            "1. **SL/TP:** If Base RR > 3.0, DO NOT increase TP. Reduce TP if Choppy (ADX<20).\n"
            "2. **TREND FILTER:** Prevents fading impulses (Counter-Trend Block).\n"
            "   - **TRENDING (ADX>30):** LOWER multipliers (e.g., 1.2x, 1.5x) to BLOCK fades aggressively.\n"
            "   - **CHOPPY (ADX<20):** HIGHER multipliers (e.g., 2.5x, 4.0x) to ALLOW mean reversion fades.\n"
            "   - **DEFAULT:** Vol 1.5x, Body 1.5x/2.0x/3.0x."
        )

        user_prompt = (
            f"**OPTIMIZATION FOR: {session_name}**\n\n"
            f"=== BASE PARAMS ===\n"
            f"SL: {base_sl} | TP: {base_tp} | RR: {base_rr:.2f}R\n\n"
            f"=== CONTEXT ===\n"
            f"Structure: {structure_context}\n"
            f"Avg Range: {avg_session_range} (Hist: {hist_context})\n"
            f"ADX: {adx_score} ({trend_status})\n"
            f"Profile: {profile_status}\n"
            f"News: {events_data}\n\n"
            "**TASK:** Return JSON:\n"
            "{\n"
            '  "sl_multiplier": float,\n'
            '  "tp_multiplier": float,\n'
            '  "trend_params": {\n'
            '      "t1_vol": float, "t1_body": float, "t2_body": float, "t3_body": float, "regime": "TRENDING/CHOPPY"\n'
            '  },\n'
            '  "reasoning": string\n'
            "}"
        )

        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=45)
            if response.status_code == 200:
                data = json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])

                raw_sl = float(data.get('sl_multiplier', 1.0))
                raw_tp = float(data.get('tp_multiplier', 1.0))

                # --- STRICT GUARDRAILS ---
                # 1. Determine Max TP Cap based on Base RR
                if base_rr >= 4.0:
                    tp_cap = 1.0   # FORBID expansion for 4R+ trades
                elif base_rr >= 2.5:
                    tp_cap = 1.15  # Very slight expansion allowed
                else:
                    tp_cap = 1.5   # Normal expansion allowed

                # 2. Apply Limits (Allow reduction down to 0.5, but cap upside)
                final_sl = max(0.5, min(raw_sl, 1.5))
                final_tp = max(0.5, min(raw_tp, tp_cap))

                data['sl_multiplier'] = final_sl
                data['tp_multiplier'] = final_tp

                # Log if cap was hit
                if raw_tp > tp_cap:
                    logging.warning(f"🛡️ High RR Protection ({base_rr:.1f}R): Capped TP {raw_tp} -> {final_tp}")

                # Ensure trend_params exists in response
                if 'trend_params' not in data:
                    data['trend_params'] = {}

                return data
            return None
        except Exception as e:
            logging.error(f"Gemini Optimization Error: {e}")
            return None
