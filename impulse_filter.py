"""
Impulse Filter - Prevents fading strong momentum moves ("Catching a falling knife").

Logic:
1. Calculate the 'Body Size' of recent candles.
2. If the last closed candle was unusually large (Impulse), BLOCK reversal trades.
3. Force the bot to wait for a 'stabilization' candle (smaller range) before entering.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

from event_logger import event_logger


class ImpulseFilter:
    """
    Prevents fading strong momentum moves ("Catching a falling knife").

    Logic:
    1. Calculate the 'Body Size' of recent candles.
    2. If the last closed candle was unusually large (Impulse), BLOCK reversal trades.
    3. Force the bot to wait for a 'stabilization' candle (smaller range) before entering.
    """
    def __init__(
        self,
        lookback: int = 20,
        impulse_multiplier: float = 2.5,
        wick_ratio_threshold: float = 0.5,
        atr_window: int = 14,
        atr_multiplier: float = 0.8,
        min_body_points: float = 0.75,
        body_ratio_threshold: float = 0.6,
    ):
        self.lookback = lookback
        self.impulse_multiplier = impulse_multiplier  # How much bigger than avg to be considered "Impulse"
        self.wick_ratio_threshold = wick_ratio_threshold  # Wick must be this fraction of body to override
        self.atr_window = atr_window
        self.atr_multiplier = atr_multiplier
        self.min_body_points = min_body_points
        self.body_ratio_threshold = body_ratio_threshold  # Body must be this fraction of total range
        self.avg_body_size = 0.0
        self.atr = 0.0
        self.last_candle_body = 0.0
        self.last_candle_dir: Optional[str] = None  # 'GREEN' or 'RED'
        # NEW: Store OHLC for wick calculation
        self.last_candle_high = 0.0
        self.last_candle_low = 0.0
        self.last_candle_open = 0.0
        self.last_candle_close = 0.0
        self.last_candle_range = 0.0

    def update(self, df: pd.DataFrame):
        """Update filter state with new candle data."""
        if len(df) < max(self.lookback, self.atr_window + 1):
            return

        # Calculate Candle Body Sizes (Abs(Close - Open))
        opens = df['open'].iloc[-self.lookback:]
        closes = df['close'].iloc[-self.lookback:]
        bodies = np.abs(closes - opens)

        # Current average body size (volatility baseline)
        self.avg_body_size = bodies.mean()

        # ATR-style volatility baseline (True Range)
        if self.atr_window > 0:
            highs = df['high']
            lows = df['low']
            prev_close = df['close'].shift(1)
            tr = np.maximum(
                highs - lows,
                np.maximum((highs - prev_close).abs(), (lows - prev_close).abs())
            )
            self.atr = tr.iloc[-self.atr_window:].mean()

        # Analyze the MOST RECENT closed bar
        last_bar = df.iloc[-1]
        self.last_candle_body = abs(last_bar['close'] - last_bar['open'])
        self.last_candle_dir = 'GREEN' if last_bar['close'] > last_bar['open'] else 'RED'

        # NEW: Store OHLC for wick calculation
        self.last_candle_high = last_bar['high']
        self.last_candle_low = last_bar['low']
        self.last_candle_open = last_bar['open']
        self.last_candle_close = last_bar['close']
        self.last_candle_range = self.last_candle_high - self.last_candle_low

    def get_impulse_stats(self) -> Tuple[bool, float, float]:
        """
        Returns:
            (is_impulse, impulse_threshold, body_ratio)
        """
        if self.avg_body_size <= 0:
            return False, 0.0, 0.0

        thresholds = [self.avg_body_size * self.impulse_multiplier]
        if self.atr > 0 and self.atr_multiplier > 0:
            thresholds.append(self.atr * self.atr_multiplier)
        if self.min_body_points > 0:
            thresholds.append(self.min_body_points)

        impulse_threshold = max(thresholds)

        body_ratio = 0.0
        if self.last_candle_range > 0:
            body_ratio = self.last_candle_body / self.last_candle_range

        is_impulse = self.last_candle_body > impulse_threshold
        if self.body_ratio_threshold > 0 and self.last_candle_range > 0:
            is_impulse = is_impulse and body_ratio >= self.body_ratio_threshold

        return is_impulse, impulse_threshold, body_ratio

    def should_block_trade(self, signal_side: str) -> Tuple[bool, Optional[str]]:
        """
        Block Reversals if the last candle was an Impulse Candle.

        NEW: Check for rejection wicks. A candle with a massive body but also
        a massive rejection wick (Hammer/Shooting Star) is often the BEST time
        to enter a reversal, not the worst.

        Args:
            signal_side: 'LONG' or 'SHORT'

        Returns:
            Tuple of (should_block: bool, reason: Optional[str])
        """
        if not signal_side:
            return False, None
        signal_side = signal_side.upper()

        is_impulse, impulse_threshold, body_ratio = self.get_impulse_stats()

        if not is_impulse:
            return False, None

        # Calculate wicks
        upper_wick = self.last_candle_high - max(self.last_candle_open, self.last_candle_close)
        lower_wick = min(self.last_candle_open, self.last_candle_close) - self.last_candle_low

        # Wick threshold: wick must be at least X% of body to indicate rejection
        wick_threshold = self.last_candle_body * self.wick_ratio_threshold

        # BLOCK LONGs if we just had a massive RED impulse
        if signal_side == 'LONG' and self.last_candle_dir == 'RED':
            # NEW: Check for Hammer pattern (massive lower wick on red candle)
            # A hammer shows rejection of lower prices - good time to go long!
            if lower_wick > wick_threshold:
                reason = (
                    f"Allowed: Red Impulse has Hammer wick "
                    f"(lower_wick: {lower_wick:.2f} > {wick_threshold:.2f})"
                )
                logging.info(f"✅ IMPULSE FILTER: {reason}")
                event_logger.log_filter_check(
                    "ImpulseFilter",
                    signal_side,
                    True,
                    reason
                )
                return False, reason

            reason = (
                "Blocked: Catching Falling Knife "
                f"(Red Impulse: {self.last_candle_body:.2f} > {impulse_threshold:.2f}, "
                f"body_ratio={body_ratio:.2f})"
            )
            logging.info(f"🚫 IMPULSE FILTER: {reason}")

            # Log to event logger
            event_logger.log_filter_check(
                "ImpulseFilter",
                signal_side,
                False,
                reason
            )
            return True, reason

        # BLOCK SHORTs if we just had a massive GREEN impulse
        if signal_side == 'SHORT' and self.last_candle_dir == 'GREEN':
            # NEW: Check for Shooting Star pattern (massive upper wick on green candle)
            # A shooting star shows rejection of higher prices - good time to go short!
            if upper_wick > wick_threshold:
                reason = (
                    f"Allowed: Green Impulse has Shooting Star wick "
                    f"(upper_wick: {upper_wick:.2f} > {wick_threshold:.2f})"
                )
                logging.info(f"✅ IMPULSE FILTER: {reason}")
                event_logger.log_filter_check(
                    "ImpulseFilter",
                    signal_side,
                    True,
                    reason
                )
                return False, reason

            reason = (
                "Blocked: Fading Rocket Ship "
                f"(Green Impulse: {self.last_candle_body:.2f} > {impulse_threshold:.2f}, "
                f"body_ratio={body_ratio:.2f})"
            )
            logging.info(f"🚫 IMPULSE FILTER: {reason}")

            # Log to event logger
            event_logger.log_filter_check(
                "ImpulseFilter",
                signal_side,
                False,
                reason
            )
            return True, reason

        return False, None
