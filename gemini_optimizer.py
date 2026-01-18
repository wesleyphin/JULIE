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
        self.csv_path = 'ml_mes_et.csv'

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
            date_col = None
            if 'ts_event' in df.columns:
                date_col = 'ts_event'
            elif 'timestamp' in df.columns:
                date_col = 'timestamp'
            else:
                date_col = next((c for c in df.columns if 'date' in c), None)
            if not date_col:
                return None
            df['timestamp'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('America/New_York')
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
        up_move = df['high'].diff()
        down_move = -df['low'].diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        alpha = 1 / period
        truerange = df['tr'].ewm(alpha=alpha, adjust=False).mean()
        plus = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / truerange
        minus = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / truerange

        sum_di = plus + minus
        dx = 100 * (plus - minus).abs() / sum_di.replace(0, 1)
        adx_series = dx.ewm(alpha=alpha, adjust=False).mean()
        adx = adx_series.iloc[-1]
        if pd.isna(adx):
            try:
                last_ts = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
                logging.warning(f"Gemini ADX NaN -> 0 | bars={len(df)} | last_ts={last_ts}")
            except Exception:
                logging.warning("Gemini ADX NaN -> 0 | bars=unknown")
        return round(float(adx) if pd.notna(adx) else 0, 2)

    def _calculate_choppiness_index(self, df, period=14):
        if df.empty or len(df) < period:
            return 0
        df = df.copy()
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)

        recent = df.iloc[-period:]
        sum_tr = recent['tr'].sum()
        price_range = recent['high'].max() - recent['low'].min()
        if price_range <= 0 or sum_tr <= 0:
            return 0

        chop = 100 * (np.log10(sum_tr / price_range) / np.log10(period))
        return round(float(chop), 2)

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
    def optimize_new_session(self, master_df, session_name, events_data, base_sl, base_tp, structure_context="", active_fvgs=None, holiday_context="", seasonal_context="NORMAL_SEASONAL", base_session_name=None):
        logging.info(f"🧠 Gemini 3.0: Analyzing Session-Aligned Context for {session_name}...")

        if master_df.empty: return None

        # --- STEP 1: CALCULATE SHARED CONTEXT (Done Once) ---
        # Use full 24h data for ADX/market profile to match chart session settings
        session_aligned_df = master_df
        if session_aligned_df.empty: return None
        base_rr = base_tp / base_sl if base_sl > 0 else 0.0
        daily_session_stats = session_aligned_df.resample('D').agg({'high': 'max', 'low': 'min', 'volume': 'sum'}).dropna()
        daily_session_stats['range'] = daily_session_stats['high'] - daily_session_stats['low']
        avg_session_range = round(daily_session_stats['range'].mean(), 2) if not daily_session_stats.empty else 0

        df_15m = session_aligned_df.resample('15min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        adx_score = self._calculate_adx(df_15m) if not df_15m.empty else 0
        chop_score = self._calculate_choppiness_index(df_15m) if not df_15m.empty else 0
        trend_status = "TRENDING" if adx_score > 25 else "CHOPPY/RANGING"
        chop_status = "CHOPPY" if chop_score >= 61.8 else ("TRENDING" if chop_score <= 38.2 else "NEUTRAL")

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

        # NEW: High-Resolution Context String (Including Seasonal & Micro-Session)
        context_str = (
            f"Session: {session_name}\n"
            f"Holiday Status: {holiday_context}\n"
            f"Seasonal Phase: {seasonal_context}\n"
            f"Structure: {structure_context}\n"
            f"Levels: VAH({profile.get('VAH')}), VAL({profile.get('VAL')}), POC({profile.get('POC')})\n"
            f"IB Range: {profile.get('IB_Low')} - {profile.get('IB_High')}\n"
            f"HTF FVGs: {fvg_summary}\n"
            f"ADX: {adx_score} ({trend_status})\n"
            f"CHOP_INDEX: {chop_score} ({chop_status})\n"
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
            f"You are a Risk Manager for {session_name}. Adjust SL and TP multipliers.\n\n"

            "*** MASTER SEASONAL RULES (PHASE 2 - DEAD ZONE) ***\n"
            "- 'PHASE_2_DEAD_ZONE' (Dec 24 - Dec 31): Market is choppy and thin. Stops get hunted.\n"
            "  >>> YOU MUST INCREASE SL MULTIPLIER (1.2x - 1.5x) to give trades breathing room.\n"
            "  >>> YOU MUST DECREASE TP MULTIPLIER (0.6x - 0.8x) to take profit early.\n"
            "  >>> DO NOT TIGHTEN STOPS in this phase. It causes instant stop-outs.\n\n"

            "*** MASTER HOLIDAY GAMEPLAN (OVERRIDES) ***\n"
            "THE VOLATILITY EXPLOSION:\n"
            "- 'POST_HOLIDAY_EXPLOSION' (Labor Day): Volatility is MASSIVE. Widen stops (1.5x). Target expansion (1.3x-1.5x).\n\n"

            "THE DEAD ZONES:\n"
            "- 'PRE_HOLIDAY': Liquidity drying up. REDUCE TP to 0.6x-0.7x. Widen SL slightly (1.1x) to survive noise.\n"
            "- 'HOLIDAY_TODAY': Dead market. Set tp_multiplier to 0.5x. Set sl_multiplier to 1.5x (survival mode).\n\n"

            "*** INTRADAY PROTOCOLS ***\n"
            "- 'NY_LUNCH' (10:30-12:30): Zombie Zone. REDUCE TP to 0.6x. MAINTAIN SL (1.0x) or WIDEN (1.1x). Do not tighten.\n"
            "- 'NY_CLOSE' (15:00-16:00): Reduce TP to 0.8x for scalps.\n\n"

            "*** ORIGINAL CORE RULES ***\n"
            "- If Base RR > 3.0, DO NOT increase TP multiplier.\n"
            "- If market is Choppy (ADX < 20), REDUCE TP multiplier to 0.8x.\n"
            "- If market is Trending (ADX > 30), allow TP expansion up to 1.5x.\n\n"
            "- If CHOP_INDEX >= 61.8, treat as CHOPPY and bias toward tighter TP.\n"
            "- If CHOP_INDEX <= 38.2, treat as TRENDING and allow TP expansion.\n\n"

            "*** GENERAL GUIDANCE ***\n"
            "- In low volume/chop, WIDER STOPS + TIGHTER TARGETS = Higher Win Rate."
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

            if isinstance(data, list):
                if len(data) > 0:
                    data = data[0]
                else:
                    return {'sl_multiplier': 1.0, 'tp_multiplier': 1.0, 'reasoning': 'Gemini returned empty list'}

            # --- HARD GUARDRAILS FOR DEAD ZONE ---
            # If we are in the Dead Zone phase, forcibly prevent tightening stops below 1.0x
            if "PHASE_2_DEAD_ZONE" in context_str:
                raw_sl = float(data.get('sl_multiplier', 1.0))
                data['sl_multiplier'] = max(1.0, raw_sl)  # Enforce 1.0x minimum
                logging.info(f"🛡️ GUARDRAIL: Enforcing Min SL 1.0x for Dead Zone (Gemini asked for {raw_sl})")

            # Standard clamping
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
            f"You are an Execution Algorithm for {session_name}. Configure Trend Filter multipliers.\n\n"

            "*** SEASONAL RULES (PHASE 2 - DEAD ZONE) ***\n"
            "- 'PHASE_2_DEAD_ZONE': False breakouts are everywhere. INCREASE filters (2.5x+) to avoid traps.\n\n"

            "*** MASTER HOLIDAY GAMEPLAN ***\n"
            "- 'POST_HOLIDAY_EXPLOSION': DISABLE Mean Reversion. LOWER filters (1.0x) to catch trends.\n"
            "- 'PRE_HOLIDAY': INCREASE filters (3.0x - 4.0x) to avoid low-volume traps.\n"
            "- 'HOLIDAY_TODAY': Max filters (4.0x).\n\n"

            "*** INTRADAY PROTOCOLS ***\n"
            "- 'NY_LUNCH': INCREASE filters to 3.0x+.\n"
            "- 'NY_CLOSE': Set 'regime' to 'CHOPPY'.\n\n"

            "*** ORIGINAL RULES ***\n"
            "- HTF FVG ROADBLOCKS: Increase t1_body (2.5x+) if approaching resistance.\n"
            "- VALUE AREA: If inside VA, maintain HIGHER multipliers.\n"
            "- TREND: If ADX > 25, switch to 'TRENDING' mode with LOWER multipliers.\n"
            "- CHOP: If CHOP_INDEX >= 61.8, switch to 'CHOPPY' mode with HIGHER multipliers.\n"
            "- TREND: If CHOP_INDEX <= 38.2, reinforce 'TRENDING' mode with LOWER multipliers."
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
            data = json.loads(response)

            # --- FIX START ---
            # If Gemini returned a list [{...}], grab the first item
            if isinstance(data, list):
                if len(data) > 0:
                    data = data[0]
                else:
                    return {'trend_params': {}, 'reasoning': 'Gemini returned empty list'}
            # --- FIX END ---

            return data
        except Exception as e:
            logging.error(f"Gemini Trend Error: {e}")
            return {'trend_params': {}, 'reasoning': 'Error'}

    # -------------------------------------------------------------------------
    # PROMPT 3: CHOP ENGINE (THRESHOLDS) - NEW
    # -------------------------------------------------------------------------
    def _fetch_chop_optimization(self, session_name, context_str):
        system_instruction = (
            f"You are a Volatility Manager for {session_name}. Adjust the 'Chop Threshold' multiplier.\n\n"

            "*** SEASONAL RULES (PHASE 2 - DEAD ZONE) ***\n"
            "- 'PHASE_2_DEAD_ZONE': RAISE chop_multiplier (2.0x). We want to FADE moves. Force fade logic.\n\n"

            "*** MASTER HOLIDAY GAMEPLAN ***\n"
            "- 'POST_HOLIDAY_EXPLOSION': LOWER chop_multiplier (0.6x). We want to enter trends.\n"
            "- 'PRE_HOLIDAY': RAISE chop_multiplier (2.0x - 2.5x). Force fading.\n"
            "- 'HOLIDAY_TODAY': Set chop_multiplier to 3.0x.\n\n"

            "*** INTRADAY PROTOCOLS ***\n"
            "- 'NY_LUNCH': Set chop_multiplier to 2.5x.\n"
            "- 'NY_CLOSE': Set chop_multiplier to 1.5x.\n\n"

            "*** ORIGINAL RULES ***\n"
            "- EXPECTING EXPLOSIVE MOVE: LOWER multiplier (0.6x).\n"
            "- EXPECTING ROTATION: RAISE multiplier (1.5x).\n"
            "- If CHOP_INDEX >= 61.8, bias toward ROTATION.\n"
            "- If CHOP_INDEX <= 38.2, bias toward EXPLOSIVE MOVE."
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

            # --- FIX START ---
            # If Gemini returned a list [{...}], grab the first item
            if isinstance(data, list):
                if len(data) > 0:
                    data = data[0]
                else:
                    return {'chop_multiplier': 1.0, 'reasoning': 'Gemini returned empty list'}
            # --- FIX END ---

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
        # CHANGE: Increased timeout from 45 to 300 to accommodate "thinking" time
        response = requests.post(self.url, headers=self.headers, json=payload, timeout=300)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        raise Exception(f"API Status {response.status_code}")
