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

        Args:
            structure_context: Textual summary of nearby S/R levels and FVGs
        """
        logging.info(f"🧠 Gemini 3.0: Analyzing Session-Aligned Context for {session_name}...")

        if master_df.empty:
            return None

        # --- STEP 1: TIME SLICING (Session-Aligned Analysis) ---
        # Instead of looking at continuous time, we only look at "Like-for-Like" sessions
        # from the last ~14 days (approx 20k bars).
        session_aligned_df = self._slice_dataframe_by_session(master_df, session_name)

        if session_aligned_df.empty:
            logging.warning(f"No history found for session {session_name}")
            return None

        logging.info(f"📊 Session-aligned data: {len(session_aligned_df)} bars for {session_name}")

        # --- STEP 2: CALCULATE METRICS ON SESSION-ALIGNED DATA ---

        # A. Volatility (Average range of THIS specific session over last 13 days)
        # Resample by day first to get daily session ranges
        daily_session_stats = session_aligned_df.resample('D').agg({
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }).dropna()

        # Calculate average range for this session type
        daily_session_stats['range'] = daily_session_stats['high'] - daily_session_stats['low']
        avg_session_range = round(daily_session_stats['range'].mean(), 2) if not daily_session_stats.empty else 0
        num_session_days = len(daily_session_stats)

        # B. Volume Profile (Relative Volume for this session)
        avg_session_volume = int(daily_session_stats['volume'].mean()) if not daily_session_stats.empty else 0

        # C. ADX on session-aligned data (reflects session-specific momentum)
        df_15m = session_aligned_df.resample('15min').agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()
        adx_score = self._calculate_adx(df_15m) if not df_15m.empty else 0
        trend_status = "TRENDING" if adx_score > 25 else "CHOPPY/RANGING"

        # D. Market Profile on session-aligned data
        profile = self._calculate_market_profile(session_aligned_df)
        curr_price = profile.get('current_price', 0)
        vah = profile.get('VAH', 0)
        val = profile.get('VAL', 0)

        profile_status = "INSIDE VALUE (Balance)"
        if curr_price > vah:
            profile_status = "ABOVE VALUE (Bullish Imbalance)"
        elif curr_price < val:
            profile_status = "BELOW VALUE (Bearish Imbalance)"

        # --- STEP 3: HISTORICAL CONTEXT (Same Session from CSV) ---
        hist_df = self._load_historical_data()
        hist_session_range = "N/A"
        hist_context = "Historical Data Unavailable"

        if hist_df is not None:
            # Slice historical data by same session hours
            hist_session_df = self._slice_dataframe_by_session(hist_df, session_name)

            if not hist_session_df.empty:
                # Calculate historical session stats
                hist_daily_stats = hist_session_df.resample('D').agg({
                    'high': 'max',
                    'low': 'min'
                }).dropna()
                hist_daily_stats['range'] = hist_daily_stats['high'] - hist_daily_stats['low']
                hist_session_range = round(hist_daily_stats['range'].mean(), 2)
                hist_context = f"{hist_session_range} pts (from {len(hist_daily_stats)} historical sessions)"

        # --- STEP 4: CONSTRUCT ADVANCED PROMPT ---
        system_instruction = (
            f"You are a Quantitative Risk Manager optimizing for the {session_name} session.\n"
            "Use Market Structure (S/R, FVGs), Trend Strength (ADX), and News to adjust TP/SL.\n"
            "Rules:\n"
            "- High ADX + Open Space (No S/R nearby): Widen TP significantly (Trend Following).\n"
            "- Price sandwiched between S/R or inside FVG: Tighten TP, reduce SL multiplier (Chop/Mean Reversion).\n"
            "- Approaching Major FVG: Conservative TP to front-run the reversal zone.\n"
            "- High ADX (>30) + Imbalance: Widen TP significantly (Trend Following).\n"
            "- Low ADX (<20) + Inside Value: Tighten TP, Widen SL slightly (Mean Reversion).\n"
            "- High Impact Events: Maximize SL (Volatility Protection).\n"
            "- Compare recent session range to historical norm to detect expansion/contraction."
        )

        user_prompt = (
            f"**OPTIMIZATION REQUEST FOR: {session_name}**\n\n"

            f"=== MARKET STRUCTURE (Walls & Magnets) ===\n"
            f"{structure_context}\n\n"

            f"=== SESSION CONTEXT (Vs. Past {num_session_days} {session_name} Sessions) ===\n"
            f"Avg Session Range: {avg_session_range} pts\n"
            f"Avg Session Volume: {avg_session_volume:,}\n"
            f"Recent Trend (ADX on Session Data): {adx_score} ({trend_status})\n"
            f"Current Market Profile: {profile_status}\n"
            f"Price Location: {curr_price} (VAH: {vah} | VAL: {val})\n\n"

            f"=== HISTORICAL CONTEXT (2023-2025 Data) ===\n"
            f"Long-term Avg {session_name} Range: {hist_context}\n\n"

            f"=== NEWS EVENTS ===\n"
            f"{events_data}\n\n"

            f"=== BASE PARAMETERS ===\n"
            f"SL: {base_sl} | TP: {base_tp}\n\n"

            "**TASK:**\n"
            "Compare recent session behavior to historical norms. "
            "Consider Market Structure (S/R walls, FVGs) when setting TP targets. "
            "If recent sessions are expanding (High Range/Vol) with open space, increase TP multiplier. "
            "If contracting (Chop) or sandwiched between structure, decrease TP multiplier.\n\n"

            "Output STRICT JSON:\n"
            "{\n"
            f'  "session": "{session_name}",\n'
            '  "regime": "TREND" or "RANGE" or "BREAKOUT",\n'
            '  "sl_multiplier": number (e.g. 1.1),\n'
            '  "tp_multiplier": number (e.g. 1.5),\n'
            '  "reasoning": "Explain using ADX, Value Area, S/R levels, FVGs, and Range comparison"\n'
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
                # 1. Parse the Raw JSON
                data = json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])

                # 2. APPLY SAFETY GUARDRAILS (Hard Cap at 1.5x, Floor at 0.5x)
                # We use .get() to avoid crashing if keys are missing, defaulting to 1.0
                raw_sl = float(data.get('sl_multiplier', 1.0))
                raw_tp = float(data.get('tp_multiplier', 1.0))

                # Apply the Cap: Min of (Value, 1.5) and Max of (Value, 0.5)
                # Lower bound (0.5) prevents stops from becoming too tight
                data['sl_multiplier'] = max(0.5, min(raw_sl, 1.5))
                data['tp_multiplier'] = max(0.5, min(raw_tp, 1.5))

                # Logging the intervention if it happened
                if raw_sl > 1.5 or raw_tp > 1.5:
                    logging.warning(f"⚠️ Gemini output capped! SL: {raw_sl}->{data['sl_multiplier']}, TP: {raw_tp}->{data['tp_multiplier']}")

                return data

            return None
        except Exception as e:
            logging.error(f"Gemini Optimization Error: {e}")
            return None
