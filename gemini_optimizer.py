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
        self.model = self.config.get('model', 'gemini-3-pro-preview')
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

        # Calculate Initial Balance (first hour of trading - first 4 bars if 15min data)
        ib_high, ib_low = None, None
        if len(df) >= 4:
            first_hour = df.head(4)
            ib_high = round(first_hour['high'].max(), 2)
            ib_low = round(first_hour['low'].min(), 2)

        return {
            "POC": poc_price,
            "VAH": vah,
            "VAL": val,
            "current_price": df.iloc[-1]['close'],
            "IB_High": ib_high,
            "IB_Low": ib_low
        }

    # =========================================================================
    # CORE OPTIMIZATION LOGIC
    # =========================================================================
    def optimize_new_session(self, master_df, session_name, events_data, base_sl, base_tp, structure_context="", active_fvgs=None, holiday_context=""):
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
        # Process HTF FVGs into a readable summary
        fvg_summary = "None Active"
        if active_fvgs:
            fvg_summary = " | ".join([
                f"{f['tf']} {f['type'].upper()} FVG @ {f['bottom']:.2f}-{f['top']:.2f}"
                for f in active_fvgs
            ])

        # NEW: High-Resolution Context String
        context_str = (
            f"Structure: {structure_context}\n"
            f"Holiday Status: {holiday_context}\n"
            f"Levels: VAH({profile.get('VAH')}), VAL({profile.get('VAL')}), POC({profile.get('POC')})\n"
            f"IB Range: {profile.get('IB_Low')} - {profile.get('IB_High')}\n"
            f"HTF FVGs: {fvg_summary}\n"
            f"ADX: {adx_score} ({trend_status})\n"
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
            f"You are a Risk Manager for {session_name}. Your task is to adjust SL and TP multipliers by balancing mathematical volatility (ADX) against structural market barriers.\n\n"
            "CORE VOLATILITY RULES:\n"
            "- If the Base Risk/Reward (RR) is > 3.0, DO NOT increase the TP multiplier (keep it ≤ 1.0).\n"
            "- If the market is Choppy (ADX < 20) or price is 'Inside Value' (between VAH and VAL), you MUST reduce the TP multiplier to 0.8x - 0.9x.\n"
            "- If the market is Trending (ADX > 30), you may allow TP expansion up to 1.5x, but ONLY if the current Base RR is < 2.5.\n\n"
            "STRUCTURAL BOUNDARY RULES:\n"
            "- HTF FVG Hard Boundaries: If a trade's TP target lies beyond an overhead Bearish FVG, you MUST reduce the tp_multiplier to exit before that resistance.\n"
            "- SL Tightening: If the current price is supported by a Bullish HTF FVG, you may tighten the sl_multiplier to 0.7x - 0.9x.\n\n"
            "HOLIDAY PLAYBOOK:\n"
            "- If a Bank Holiday is TODAY or TOMORROW (within 24 hours), aggressively reduce TP multipliers (0.5x - 0.7x) to account for potential volume drop-off and erratic spreads.\n"
            "- If a Bank Holiday is within 2-3 days, apply moderate TP reduction (0.8x - 0.9x) as institutional participation begins to decline.\n"
            "- During holiday periods, prioritize capital preservation over profit maximization."
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
            f"You are an Execution Algorithm for {session_name}. Configure Trend Filter multipliers to distinguish between high-probability breakouts and traps.\n\n"
            "MERGED EXECUTION RULES:\n"
            "- HTF FVG ROADBLOCKS: Treat HTF FVGs as structural barriers. If price is approaching a Bearish FVG from below or a Bullish FVG from above, increase the t1_body multiplier (e.g., 2.5x+) to ensure only high-momentum impulses trigger entry.\n"
            "- VALUE AREA & IB LOGIC: If price is inside the Value Area (VAH/VAL) and below the IB High, maintain HIGHER multipliers (2.5x - 4.0x) to allow for rotational/mean-reversion trades.\n"
            "- TREND CONFIRMATION: If price breaks the IB High/Low with an ADX > 25, switch to aggressive 'Trending' mode with LOWER multipliers (1.2x - 1.5x)."
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
            f"You are a Volatility Manager for {session_name}. Adjust the 'Chop Threshold' multiplier.\n\n"
            "LOGIC & STRUCTURAL RULES:\n"
            "- EXPECTING EXPLOSIVE MOVE: LOWER the multiplier to 0.6x - 0.9x to allow for trend breakouts.\n"
            "- EXPECTING ROTATION/RANGE: RAISE the multiplier to 1.2x - 2.0x to prioritize Fade strategies.\n"
            "- POC & IB CONFLUENCE: If the current price is hovering at the POC and is inside the IB Range with no HTF FVGs nearby, RAISE the multiplier to 1.5x - 2.0x to signal a high-probability range-bound environment.\n\n"
            "HOLIDAY PLAYBOOK:\n"
            "- If a Bank Holiday is near (within 1-3 days), favor mean-reversion by RAISING the Chop Multiplier (1.5x - 2.0x+), as institutional trend-following volume is typically absent.\n"
            "- Holiday periods create range-bound, choppy conditions. Prioritize fade strategies over breakout plays during these times."
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
            "systemInstruction": {"parts": [{"text": sys_instr}]},
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 1.0,
                "thinkingConfig": {
                    "thinkingLevel": "high"
                }
            }
        }
        response = requests.post(self.url, headers=self.headers, json=payload, timeout=45)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        raise Exception(f"API Status {response.status_code}")
