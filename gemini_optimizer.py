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
        session_hours = CONFIG.get('SESSIONS', {}).get(session_name, {}).get('HOURS', [])
        if not session_hours or df.empty: return df
        if not isinstance(df.index, pd.DatetimeIndex): return df
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

    # =========================================================================
    # CORE OPTIMIZATION LOGIC
    # =========================================================================
    def optimize_new_session(self, master_df, session_name, events_data, base_sl, base_tp, structure_context=""):
        logging.info(f"🧠 Gemini 3.0: Analyzing Session-Aligned Context for {session_name}...")

        if master_df.empty: return None

        # --- STEP 1: CALCULATE SHARED CONTEXT (Done Once) ---
        session_aligned_df = self._slice_dataframe_by_session(master_df, session_name)
        if session_aligned_df.empty: return None

        base_rr = base_tp / base_sl if base_sl > 0 else 0.0
        daily_session_stats = session_aligned_df.resample('D').agg({'high': 'max', 'low': 'min', 'volume': 'sum'}).dropna()
        daily_session_stats['range'] = daily_session_stats['high'] - daily_session_stats['low']
        avg_session_range = round(daily_session_stats['range'].mean(), 2) if not daily_session_stats.empty else 0

        df_15m = session_aligned_df.resample('15min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        adx_score = self._calculate_adx(df_15m) if not df_15m.empty else 0
        trend_status = "TRENDING" if adx_score > 25 else "CHOPPY/RANGING"

        profile = self._calculate_market_profile(session_aligned_df)
        curr_price = profile.get('current_price', 0)
        vah, val = profile.get('VAH', 0), profile.get('VAL', 0)
        profile_status = "INSIDE VALUE"
        if curr_price > vah: profile_status = "ABOVE VALUE"
        elif curr_price < val: profile_status = "BELOW VALUE"

        hist_df = self._load_historical_data()
        hist_context = "Unavailable"
        if hist_df is not None:
            hist_session_df = self._slice_dataframe_by_session(hist_df, session_name)
            if not hist_session_df.empty:
                hist_daily_stats = hist_session_df.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
                hist_daily_stats['range'] = hist_daily_stats['high'] - hist_daily_stats['low']
                hist_context = f"{round(hist_daily_stats['range'].mean(), 2)} pts"

        # --- PREPARE CONTEXT STRING (Shared) ---
        context_str = (
            f"Structure: {structure_context}\n"
            f"Avg Range: {avg_session_range} (Hist: {hist_context})\n"
            f"ADX: {adx_score} ({trend_status})\n"
            f"Profile: {profile_status}\n"
            f"News: {events_data}\n"
        )

        # --- STEP 2: EXECUTE TRIPLE PROMPTS ---

        # Call 1: Risk Engine (SL/TP)
        sltp_data = self._fetch_sltp_optimization(session_name, context_str, base_sl, base_tp, base_rr)

        # Call 2: Trend Engine (Filter Params)
        trend_data = self._fetch_trend_optimization(session_name, context_str)

        # Call 3: Chop Engine (Thresholds) - NEW
        chop_data = self._fetch_chop_optimization(session_name, context_str)

        # --- STEP 3: MERGE & RETURN ---
        final_result = {
            "sl_multiplier": sltp_data.get('sl_multiplier', 1.0),
            "tp_multiplier": sltp_data.get('tp_multiplier', 1.0),
            "trend_params": trend_data.get('trend_params', {}),
            "chop_multiplier": chop_data.get('chop_multiplier', 1.0),
            "reasoning": (
                f"RISK: {sltp_data.get('reasoning', 'None')} | "
                f"TREND: {trend_data.get('reasoning', 'None')} | "
                f"CHOP: {chop_data.get('reasoning', 'None')}"
            )
        }

        return final_result

    # -------------------------------------------------------------------------
    # PROMPT 1: RISK ENGINE (SL/TP)
    # -------------------------------------------------------------------------
    def _fetch_sltp_optimization(self, session_name, context_str, base_sl, base_tp, base_rr):
        system_instruction = (
            f"You are a Risk Manager for {session_name}. Adjust SL/TP multipliers based on Volatility.\n"
            "CRITICAL RULES:\n"
            "- If Base RR > 3.0, DO NOT increase TP (Keep <= 1.0). Win rate is priority.\n"
            "- If Choppy (ADX < 20) or Inside Value, REDUCE TP (0.8x - 0.9x).\n"
            "- If Trending (ADX > 30), allow TP expansion (up to 1.5x) IF RR < 2.5."
        )

        user_prompt = (
            f"**OPTIMIZE SL/TP FOR: {session_name}**\n\n"
            f"=== BASE PARAMS ===\n"
            f"SL: {base_sl} | TP: {base_tp} | RR: {base_rr:.2f}R\n\n"
            f"=== CONTEXT ===\n"
            f"{context_str}\n"
            "**TASK:** Return JSON {sl_multiplier, tp_multiplier, reasoning}."
        )

        try:
            response = self._call_gemini(system_instruction, user_prompt)
            data = json.loads(response)

            # Apply Hard Guardrails immediately
            raw_tp = float(data.get('tp_multiplier', 1.0))
            if base_rr >= 4.0: tp_cap = 1.0
            elif base_rr >= 2.5: tp_cap = 1.15
            else: tp_cap = 1.5

            data['sl_multiplier'] = max(0.5, min(float(data.get('sl_multiplier', 1.0)), 1.5))
            data['tp_multiplier'] = max(0.5, min(raw_tp, tp_cap))
            return data
        except Exception as e:
            logging.error(f"Gemini SL/TP Error: {e}")
            return {'sl_multiplier': 1.0, 'tp_multiplier': 1.0, 'reasoning': 'Error'}

    # -------------------------------------------------------------------------
    # PROMPT 2: TREND ENGINE (FILTERS)
    # -------------------------------------------------------------------------
    def _fetch_trend_optimization(self, session_name, context_str):
        system_instruction = (
            f"You are an Execution Algorithm for {session_name}. Configure the Trend Filter to block bad counter-trend trades.\n"
            "LOGIC:\n"
            "- **STRONG TREND (ADX > 30):** Use LOWER multipliers (1.2x - 1.5x) to trigger the filter easily. We want to BLOCK fading attempts.\n"
            "- **CHOP/RANGE (ADX < 20):** Use HIGHER multipliers (2.5x - 4.0x) to make the filter loose. We want to ALLOW fading.\n"
            "- **DEFAULT:** Vol 1.5, Body 1.5 (T1), 2.0 (T2), 3.0 (T3)."
        )

        user_prompt = (
            f"**CONFIGURE TREND FILTER FOR: {session_name}**\n\n"
            f"=== CONTEXT ===\n"
            f"{context_str}\n"
            "**TASK:** Return JSON:\n"
            "{\n"
            '  "trend_params": {"t1_vol": float, "t1_body": float, "t2_body": float, "t3_body": float, "regime": "TRENDING/CHOPPY"},\n'
            '  "reasoning": string\n'
            "}"
        )

        try:
            response = self._call_gemini(system_instruction, user_prompt)
            return json.loads(response)
        except Exception as e:
            logging.error(f"Gemini Trend Error: {e}")
            return {'trend_params': {}, 'reasoning': 'Error'}

    # -------------------------------------------------------------------------
    # PROMPT 3: CHOP ENGINE (THRESHOLDS) - NEW
    # -------------------------------------------------------------------------
    def _fetch_chop_optimization(self, session_name, context_str):
        system_instruction = (
            f"You are a Volatility Manager for {session_name}. Adjust the 'Chop Threshold' multiplier.\n"
            "THEORY:\n"
            "- The 'Chop Threshold' is the ceiling for volatility. Below this = CHOP. Above this = ACTIVE.\n"
            "LOGIC:\n"
            "- **EXPECTING EXPLOSIVE MOVE (News/Breakout):** LOWER the multiplier (0.6x - 0.9x). "
            "We want the ceiling LOW so the market easily breaks out of 'Chop' status and allows trend trades.\n"
            "- **EXPECTING ROTATION/RANGE:** RAISE the multiplier (1.2x - 2.0x). "
            "We want the ceiling HIGH so the market stays 'In Chop', enabling Fade strategies (Buy Low/Sell High).\n"
            "- **DEFAULT:** 1.0"
        )

        user_prompt = (
            f"**OPTIMIZE CHOP THRESHOLD FOR: {session_name}**\n\n"
            f"=== CONTEXT ===\n"
            f"{context_str}\n"
            "**TASK:** Return JSON:\n"
            "{\n"
            ' "chop_multiplier": float,\n'
            ' "reasoning": string\n'
            "}"
        )

        try:
            response = self._call_gemini(system_instruction, user_prompt)
            data = json.loads(response)
            # Safety clamp (0.5x to 3.0x)
            data['chop_multiplier'] = max(0.5, min(float(data.get('chop_multiplier', 1.0)), 3.0))
            return data
        except Exception as e:
            logging.error(f"Gemini Chop Error: {e}")
            return {'chop_multiplier': 1.0, 'reasoning': 'Error'}

    def _call_gemini(self, sys_instr, user_prompt):
        payload = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "system_instruction": {"parts": [{"text": sys_instr}]},
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1}
        }
        response = requests.post(self.url, headers=self.headers, json=payload, timeout=45)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        raise Exception(f"API Status {response.status_code}")
