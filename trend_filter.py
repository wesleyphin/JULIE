import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

from event_logger import event_logger


class TrendFilter:
    """
    Prevents fading strong momentum moves ("Catching a falling knife").

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

        # Tier Params
        self.t1_vol_mult = tier1_vol_multiplier
        self.t1_body_mult = tier1_body_multiplier
        self.t2_body_mult = tier2_body_multiplier
        self.t3_body_mult = tier3_body_multiplier
        self.fast_period = fast_period
        self.slow_period = slow_period

    def should_block_trade(self, df: pd.DataFrame, side: str, is_range_fade: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Block Reversals based on 4 Tiers of Impulse/Trend.

        Args:
            df: DataFrame with OHLCV data
            side: 'LONG' or 'SHORT'
            is_range_fade: If True, bypasses Tier 4 (Trend) logic (Smart Bypass).

        Returns:
            Tuple of (should_block: bool, reason: Optional[str])
        """
        if len(df) < self.lookback:
            return False, None

        # --- Calculate Stats ---
        short_window = 20
        opens = df['open'].iloc[-short_window:]
        closes = df['close'].iloc[-short_window:]
        bodies = np.abs(closes - opens)
        avg_body_size = bodies.mean()

        avg_vol = 0.0
        if 'volume' in df.columns:
            volumes = df['volume'].iloc[-short_window:]
            avg_vol = volumes.mean()

        # Last Candle Data
        last_bar = df.iloc[-1]
        current_price = last_bar['close']
        last_candle_body = abs(last_bar['close'] - last_bar['open'])
        last_candle_vol = last_bar.get('volume', 0.0)
        last_candle_dir = 'GREEN' if last_bar['close'] > last_bar['open'] else 'RED'
        last_candle_high = last_bar['high']
        last_candle_low = last_bar['low']
        last_candle_open = last_bar['open']
        last_candle_close = last_bar['close']

        # --- Macro Trend Stats (Tier 4) ---
        closes_full = df['close']
        trend_state = "NEUTRAL"
        if len(closes_full) >= self.slow_period:
            ema_fast_val = closes_full.ewm(span=self.fast_period, adjust=False).mean().iloc[-1]
            ema_slow_val = closes_full.ewm(span=self.slow_period, adjust=False).mean().iloc[-1]

            # Strong Bullish: Price > Fast > Slow
            if current_price > ema_fast_val > ema_slow_val:
                trend_state = "BULLISH"
            # Strong Bearish: Price < Fast < Slow
            elif current_price < ema_fast_val < ema_slow_val:
                trend_state = "BEARISH"

        # =========================================================
        # TIER 4: MACRO TREND FILTER (The "Big Picture" Check)
        # =========================================================
        if not is_range_fade:
            if side == "SHORT" and trend_state == "BULLISH":
                reason = f"Blocked (Tier 4): Strong Bullish Uptrend (Price > {self.fast_period}EMA > {self.slow_period}EMA)"
                logging.info(f"🚫 TREND FILTER: {reason}")
                event_logger.log_filter_check("TrendFilter", side, False, reason)
                return True, reason

            if side == "LONG" and trend_state == "BEARISH":
                reason = f"Blocked (Tier 4): Strong Bearish Downtrend (Price < {self.fast_period}EMA < {self.slow_period}EMA)"
                logging.info(f"🚫 TREND FILTER: {reason}")
                event_logger.log_filter_check("TrendFilter", side, False, reason)
                return True, reason
        else:
            if trend_state != "NEUTRAL":
                logging.info(f"🔓 Tier 4 Bypassed: Range Fade Logic Active ({trend_state} Trend ignored)")

        # =========================================================
        # TIERS 1-3: CANDLE IMPULSE FILTERS (The "Immediate" Check)
        # =========================================================

        # Identify Impulse Tiers
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
        # NOTE: Tier 4 (Trend) is NOT overridden by wicks, only by 'is_range_fade'.
        upper_wick = last_candle_high - max(last_candle_open, last_candle_close)
        lower_wick = min(last_candle_open, last_candle_close) - last_candle_low
        wick_threshold = last_candle_body * self.wick_ratio_threshold

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
