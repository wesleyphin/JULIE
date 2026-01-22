import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

from event_logger import event_logger


class TrendFilter:
    """
    Prevents fading strong momentum moves ("Catching a falling knife").
    Now supports DYNAMIC SENSITIVITY updates from Gemini.

    UPDATED to 4-Tier System:

    TIER 1 (Volume): "Smart Money" Move
    - Logic: Body > 1.5x Avg AND Volume > 1.5x Avg
    - Danger: Highest statistical probability of continuation (46.2%).

    TIER 2 (Standard): "Breakout" Move
    - Logic: Body > 2.0x Avg.
    - Danger: Standard strong momentum.

    TIER 3 (Extreme): "Capitulation" Move
    - Logic: Body > 3.0x Avg.
    - Danger: Violent price shock.

    TIER 4 (Macro Trend): "The Trend is Your Friend"
    - Logic: 50 EMA vs 200 EMA alignment (Strong Bullish/Bearish).
    - Danger: Fading the macro trend.
    - Exception: "Smart Bypass" allows fading if Chop Analyzer detects a Range Fade setup.
    """
    def __init__(self, lookback: int = 200,
                 wick_ratio_threshold: float = 0.5,
                 # Tier 1-3 Parameters
                 tier1_vol_multiplier: float = 1.5,
                 tier1_body_multiplier: float = 1.5,
                 tier2_body_multiplier: float = 2.0,
                 tier3_body_multiplier: float = 3.0,
                 # Tier 4 (Trend) Parameters
                 fast_period: int = 50,
                 slow_period: int = 200):

        # Ensure we have enough data for the Slow EMA
        self.lookback = max(lookback, slow_period + 20)
        self.wick_ratio_threshold = wick_ratio_threshold

        # Dynamic Thresholds (can be updated by Gemini)
        self.t1_vol_mult = tier1_vol_multiplier
        self.t1_body_mult = tier1_body_multiplier
        self.t2_body_mult = tier2_body_multiplier
        self.t3_body_mult = tier3_body_multiplier
        self.fast_period = fast_period
        self.slow_period = slow_period

        # Track current regime for logging
        self.current_regime = "DEFAULT"

    def update_dynamic_params(self, params: dict):
        """
        Dynamically update filter thresholds from Gemini/Optimizer.
        Logic:
        - TRENDING Market -> Lower multipliers (Stricter, block more fades)
        - CHOPPY Market -> Higher multipliers (Looser, allow range fading)
        """
        if not params:
            return

        try:
            logging.info(f"🌊 UPDATING TREND FILTER ({params.get('regime', 'Custom')}): {params}")

            if 't1_vol' in params:
                self.t1_vol_mult = float(params['t1_vol'])
            if 't1_body' in params:
                self.t1_body_mult = float(params['t1_body'])
            if 't2_body' in params:
                self.t2_body_mult = float(params['t2_body'])
            if 't3_body' in params:
                self.t3_body_mult = float(params['t3_body'])

            self.current_regime = params.get('regime', 'DYNAMIC')

        except Exception as e:
            logging.error(f"❌ Error updating TrendFilter params: {e}")

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        if df.empty or len(df) < period:
            return 0.0
        if not {'high', 'low', 'close'}.issubset(df.columns):
            return 0.0

        local_df = df.copy()
        local_df['tr0'] = (local_df['high'] - local_df['low']).abs()
        local_df['tr1'] = (local_df['high'] - local_df['close'].shift(1)).abs()
        local_df['tr2'] = (local_df['low'] - local_df['close'].shift(1)).abs()
        local_df['tr'] = local_df[['tr0', 'tr1', 'tr2']].max(axis=1)

        up_move = local_df['high'].diff()
        down_move = -local_df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        alpha = 1 / period
        truerange = local_df['tr'].ewm(alpha=alpha, adjust=False).mean()
        plus = 100 * pd.Series(plus_dm, index=local_df.index).ewm(alpha=alpha, adjust=False).mean() / truerange
        minus = 100 * pd.Series(minus_dm, index=local_df.index).ewm(alpha=alpha, adjust=False).mean() / truerange

        sum_di = plus + minus
        dx = 100 * (plus - minus).abs() / sum_di.replace(0, 1)
        adx_series = dx.ewm(alpha=alpha, adjust=False).mean()
        adx = adx_series.iloc[-1]
        if pd.isna(adx):
            return 0.0
        return float(adx)

    def should_block_trade(self, df: pd.DataFrame, side: str, is_range_fade: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Block Reversals based on 4 Tiers of Impulse/Trend.
        """
        if len(df) < self.lookback:
            return False, None

        # --- 1. Calculate Basic Candle Stats ---
        short_window = 20
        opens = df['open'].iloc[-short_window:]
        closes = df['close'].iloc[-short_window:]
        bodies = np.abs(closes - opens)
        avg_body_size = bodies.mean()

        last_bar = df.iloc[-1]
        current_price = last_bar['close']
        last_candle_body = abs(last_bar['close'] - last_bar['open'])

        # --- 2. Calculate Wicks IMMEDIATELY (Move to Top) ---
        last_candle_high = last_bar['high']
        last_candle_low = last_bar['low']
        last_candle_open = last_bar['open']
        last_candle_close = last_bar['close']

        upper_wick = last_candle_high - max(last_candle_open, last_candle_close)
        lower_wick = min(last_candle_open, last_candle_close) - last_candle_low

        # Use existing wick_ratio_threshold (default 0.5 or 50% of body)
        wick_threshold = last_candle_body * self.wick_ratio_threshold

        # --- 3. Determine Macro Trend State ---
        closes_full = df['close']
        trend_state = "NEUTRAL"
        if len(closes_full) >= self.slow_period:
            ema_fast_val = closes_full.ewm(span=self.fast_period, adjust=False).mean().iloc[-1]
            ema_slow_val = closes_full.ewm(span=self.slow_period, adjust=False).mean().iloc[-1]

            if current_price > ema_fast_val > ema_slow_val:
                trend_state = "BULLISH"
            elif current_price < ema_fast_val < ema_slow_val:
                trend_state = "BEARISH"

        # =========================================================
        # TIER 4: MACRO TREND FILTER (Modified with Wick Override)
        # =========================================================
        adx_current = self._calculate_adx(df)
        tier4_bypass = adx_current > 25.0
        if tier4_bypass and trend_state != "NEUTRAL":
            logging.info(f"🔓 Tier 4 Bypassed: ADX {adx_current:.2f} > 25 ({trend_state} trend ignored)")

        if not is_range_fade and not tier4_bypass:
            # BLOCK SHORTS IN BULL TREND
            if side == "SHORT" and trend_state == "BULLISH":
                # EXCEPTION: If we have a massive Shooting Star wick, allow the counter-trend trade
                if upper_wick > wick_threshold:
                    logging.info(f"✅ TREND EXCEPTION: Bullish Trend overridden by Shooting Star Wick")
                    return False, "Allowed: Counter-trend Shooting Star"

                reason = f"Blocked (Tier 4): Strong Bullish Uptrend (Price > {self.fast_period} > {self.slow_period})"
                logging.info(f"🚫 TREND FILTER: {reason}")
                event_logger.log_filter_check("TrendFilter", side, False, reason)
                return True, reason

            # BLOCK LONGS IN BEAR TREND
            if side == "LONG" and trend_state == "BEARISH":
                # EXCEPTION: If we have a massive Hammer wick, allow the counter-trend trade
                if lower_wick > wick_threshold:
                    logging.info(f"✅ TREND EXCEPTION: Bearish Trend overridden by Hammer Wick")
                    return False, "Allowed: Counter-trend Hammer"

                reason = f"Blocked (Tier 4): Strong Bearish Downtrend (Price < {self.fast_period} < {self.slow_period})"
                logging.info(f"🚫 TREND FILTER: {reason}")
                event_logger.log_filter_check("TrendFilter", side, False, reason)
                return True, reason
        else:
            if trend_state != "NEUTRAL":
                logging.info(f"🔓 Tier 4 Bypassed: Range Fade Logic Active ({trend_state} Trend ignored)")

        # =========================================================
        # TIERS 1-3: CANDLE IMPULSE FILTERS (Existing Logic)
        # =========================================================
        avg_vol = 0.0
        if 'volume' in df.columns:
            volumes = df['volume'].iloc[-short_window:]
            avg_vol = volumes.mean()

        last_candle_vol = last_bar.get('volume', 0.0)
        last_candle_dir = 'GREEN' if last_bar['close'] > last_bar['open'] else 'RED'

        # Check for Tier 3 Nuke in last 3 bars
        recent_opens = df['open'].iloc[-3:]
        recent_closes = df['close'].iloc[-3:]
        recent_bodies = np.abs(recent_closes - recent_opens)
        tier3_threshold = avg_body_size * self.t3_body_mult

        if (recent_bodies > tier3_threshold).any():
             # If nuke detected, check if CURRENT bar has rejection
            if side == 'LONG' and lower_wick > wick_threshold:
                pass
            elif side == 'SHORT' and upper_wick > wick_threshold:
                pass
            else:
                return True, "Blocked: Recent Tier 3 'Nuke' detected (Shockwave protection)"

        # Identify Impulse Tiers (Single Candle Check)
        is_tier1 = False
        is_tier2 = False
        is_tier3 = False

        # Tier 3: Extreme Price
        if last_candle_body > (avg_body_size * self.t3_body_mult):
            is_tier3 = True

        # Tier 2: Standard Price
        elif last_candle_body > (avg_body_size * self.t2_body_mult):
            is_tier2 = True

        # Tier 1: Volume Supported
        if avg_vol > 0:
            if (last_candle_body > (avg_body_size * self.t1_body_mult)) and \
               (last_candle_vol > (avg_vol * self.t1_vol_mult)):
                is_tier1 = True

        # If no candle impulse detected, we are clear
        if not (is_tier1 or is_tier2 or is_tier3):
            return False, None

        # Construct Block Reason
        impulse_name = ""
        details = ""

        if is_tier3:
            impulse_name = "Tier 3 (Extreme Price)"
            details = f"Body {last_candle_body:.2f} > {self.t3_body_mult}x Avg"
        elif is_tier2:
            impulse_name = "Tier 2 (Standard Impulse)"
            details = f"Body {last_candle_body:.2f} > {self.t2_body_mult}x Avg"

        if is_tier1:
            if impulse_name:
                impulse_name += " + Tier 1 (High Vol)"
                details += f" & Vol {last_candle_vol:.0f} > {self.t1_vol_mult}x Avg"
            else:
                impulse_name = "Tier 1 (Volume Impulse)"
                details = f"Vol {last_candle_vol:.0f} > {self.t1_vol_mult}x Avg"

        # --- Wick Safety Check (Override for Tiers 1-3) ---
        # Logic for LONG Signal (Fading a RED Impulse)
        if side == 'LONG' and last_candle_dir == 'RED':
            if lower_wick > wick_threshold:
                reason = f"Allowed: {impulse_name} has Hammer Wick (Rejection)"
                logging.info(f"✅ TREND FILTER: {reason}")
                event_logger.log_filter_check("TrendFilter", side, True, reason)
                return False, reason

            reason = f"Blocked (Tier 1-3): Catching Falling Knife ({impulse_name}: {details})"
            logging.info(f"🚫 TREND FILTER: {reason}")
            event_logger.log_filter_check("TrendFilter", side, False, reason)
            return True, reason

        # Logic for SHORT Signal (Fading a GREEN Impulse)
        if side == 'SHORT' and last_candle_dir == 'GREEN':
            if upper_wick > wick_threshold:
                reason = f"Allowed: {impulse_name} has Shooting Star Wick (Rejection)"
                logging.info(f"✅ TREND FILTER: {reason}")
                event_logger.log_filter_check("TrendFilter", side, True, reason)
                return False, reason

            reason = f"Blocked (Tier 1-3): Fading Rocket Ship ({impulse_name}: {details})"
            logging.info(f"🚫 TREND FILTER: {reason}")
            event_logger.log_filter_check("TrendFilter", side, False, reason)
            return True, reason

        return False, None
