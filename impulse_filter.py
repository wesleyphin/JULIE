import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

from event_logger import event_logger


class ImpulseFilter:
    """
    Prevents fading strong momentum moves ("Catching a falling knife").

    UPDATED to 4-Tier System (Merged with TrendFilter):

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
    def __init__(self, lookback: int = 200,  # Increased default for Tier 4 EMA
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

        # State
        self.avg_body_size = 0.0
        self.avg_vol = 0.0

        # Candle State
        self.last_candle_body = 0.0
        self.last_candle_vol = 0.0
        self.last_candle_dir: Optional[str] = None
        self.last_candle_high = 0.0
        self.last_candle_low = 0.0
        self.last_candle_open = 0.0
        self.last_candle_close = 0.0

        # Trend State (Tier 4)
        self.trend_state = "NEUTRAL" # 'BULLISH', 'BEARISH', 'NEUTRAL'
        self.current_price = 0.0
        self.ema_fast_val = 0.0
        self.ema_slow_val = 0.0

    def update(self, df: pd.DataFrame):
        """Update filter state with new candle data."""
        if len(df) < self.lookback:
            return

        # --- 1. Candle Impulse Stats (Tiers 1-3) ---
        # Use last 20 bars for body averages (consistent with original logic)
        short_window = 20
        opens = df['open'].iloc[-short_window:]
        closes = df['close'].iloc[-short_window:]
        bodies = np.abs(closes - opens)
        self.avg_body_size = bodies.mean()

        if 'volume' in df.columns:
            volumes = df['volume'].iloc[-short_window:]
            self.avg_vol = volumes.mean()
        else:
            self.avg_vol = 0.0

        # Store Last Candle Data
        last_bar = df.iloc[-1]
        self.current_price = last_bar['close']
        self.last_candle_body = abs(last_bar['close'] - last_bar['open'])
        self.last_candle_vol = last_bar.get('volume', 0.0)
        self.last_candle_dir = 'GREEN' if last_bar['close'] > last_bar['open'] else 'RED'

        self.last_candle_high = last_bar['high']
        self.last_candle_low = last_bar['low']
        self.last_candle_open = last_bar['open']
        self.last_candle_close = last_bar['close']

        # --- 2. Macro Trend Stats (Tier 4) ---
        # Calculate EMAs on the full lookback window
        closes_full = df['close']
        if len(closes_full) >= self.slow_period:
            self.ema_fast_val = closes_full.ewm(span=self.fast_period, adjust=False).mean().iloc[-1]
            self.ema_slow_val = closes_full.ewm(span=self.slow_period, adjust=False).mean().iloc[-1]

            # Determine Strong Trend State
            # Strong Bullish: Price > Fast > Slow
            if self.current_price > self.ema_fast_val > self.ema_slow_val:
                self.trend_state = "BULLISH"
            # Strong Bearish: Price < Fast < Slow
            elif self.current_price < self.ema_fast_val < self.ema_slow_val:
                self.trend_state = "BEARISH"
            else:
                self.trend_state = "NEUTRAL"

    def should_block_trade(self, signal_side: str, is_range_fade: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Block Reversals based on 4 Tiers of Impulse/Trend.

        Args:
            signal_side: 'LONG' or 'SHORT'
            is_range_fade: If True, bypasses Tier 4 (Trend) logic (Smart Bypass).
        """

        # =========================================================
        # TIER 4: MACRO TREND FILTER (The "Big Picture" Check)
        # =========================================================
        if not is_range_fade:
            if signal_side == "SHORT" and self.trend_state == "BULLISH":
                reason = f"Blocked (Tier 4): Strong Uptrend (Price > {self.fast_period} > {self.slow_period})"
                logging.info(f"🚫 IMPULSE FILTER: {reason}")
                event_logger.log_filter_check("ImpulseFilter", signal_side, False, reason)
                return True, reason

            if signal_side == "LONG" and self.trend_state == "BEARISH":
                reason = f"Blocked (Tier 4): Strong Downtrend (Price < {self.fast_period} < {self.slow_period})"
                logging.info(f"🚫 IMPULSE FILTER: {reason}")
                event_logger.log_filter_check("ImpulseFilter", signal_side, False, reason)
                return True, reason
        else:
            if self.trend_state != "NEUTRAL":
                logging.info(f"🔓 Tier 4 Bypassed: Range Fade Logic Active ({self.trend_state} Trend ignored)")

        # =========================================================
        # TIERS 1-3: CANDLE IMPULSE FILTERS (The "Immediate" Check)
        # =========================================================

        # Identify Impulse Tiers
        is_tier1 = False
        is_tier2 = False
        is_tier3 = False

        # Tier 3: Extreme Price
        if self.last_candle_body > (self.avg_body_size * self.t3_body_mult):
            is_tier3 = True

        # Tier 2: Standard Price
        elif self.last_candle_body > (self.avg_body_size * self.t2_body_mult):
            is_tier2 = True

        # Tier 1: Volume Supported
        if self.avg_vol > 0:
            if (self.last_candle_body > (self.avg_body_size * self.t1_body_mult)) and \
               (self.last_candle_vol > (self.avg_vol * self.t1_vol_mult)):
                is_tier1 = True

        # If no candle impulse detected, we are clear
        if not (is_tier1 or is_tier2 or is_tier3):
            return False, None

        # Construct Block Reason
        impulse_name = ""
        details = ""

        if is_tier3:
            impulse_name = "Tier 3 (Extreme Price)"
            details = f"Body {self.last_candle_body:.2f} > {self.t3_body_mult}x Avg"
        elif is_tier2:
            impulse_name = "Tier 2 (Standard Impulse)"
            details = f"Body {self.last_candle_body:.2f} > {self.t2_body_mult}x Avg"

        if is_tier1:
            if impulse_name:
                impulse_name += " + Tier 1 (High Vol)"
                details += f" & Vol {self.last_candle_vol:.0f} > {self.t1_vol_mult}x Avg"
            else:
                impulse_name = "Tier 1 (Volume Impulse)"
                details = f"Vol {self.last_candle_vol:.0f} > {self.t1_vol_mult}x Avg"

        # --- Wick Safety Check (Override for Tiers 1-3) ---
        # NOTE: Tier 4 (Trend) is NOT overridden by wicks, only by 'is_range_fade'.
        upper_wick = self.last_candle_high - max(self.last_candle_open, self.last_candle_close)
        lower_wick = min(self.last_candle_open, self.last_candle_close) - self.last_candle_low
        wick_threshold = self.last_candle_body * self.wick_ratio_threshold

        # Logic for LONG Signal (Fading a RED Impulse)
        if signal_side == 'LONG' and self.last_candle_dir == 'RED':
            if lower_wick > wick_threshold:
                reason = f"Allowed: {impulse_name} has Hammer Wick (Rejection)"
                logging.info(f"✅ IMPULSE FILTER: {reason}")
                event_logger.log_filter_check("ImpulseFilter", signal_side, True, reason)
                return False, reason

            reason = f"Blocked (Tier 1-3): Catching Falling Knife ({impulse_name}: {details})"
            logging.info(f"🚫 IMPULSE FILTER: {reason}")
            event_logger.log_filter_check("ImpulseFilter", signal_side, False, reason)
            return True, reason

        # Logic for SHORT Signal (Fading a GREEN Impulse)
        if signal_side == 'SHORT' and self.last_candle_dir == 'GREEN':
            if upper_wick > wick_threshold:
                reason = f"Allowed: {impulse_name} has Shooting Star Wick (Rejection)"
                logging.info(f"✅ IMPULSE FILTER: {reason}")
                event_logger.log_filter_check("ImpulseFilter", signal_side, True, reason)
                return False, reason

            reason = f"Blocked (Tier 1-3): Fading Rocket Ship ({impulse_name}: {details})"
            logging.info(f"🚫 IMPULSE FILTER: {reason}")
            event_logger.log_filter_check("ImpulseFilter", signal_side, False, reason)
            return True, reason

        return False, None
