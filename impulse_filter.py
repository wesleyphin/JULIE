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
    def __init__(self, lookback: int = 20, impulse_multiplier: float = 2.5):
        self.lookback = lookback
        self.impulse_multiplier = impulse_multiplier  # How much bigger than avg to be considered "Impulse"
        self.avg_body_size = 0.0
        self.last_candle_body = 0.0
        self.last_candle_dir: Optional[str] = None  # 'GREEN' or 'RED'

    def update(self, df: pd.DataFrame):
        """Update filter state with new candle data."""
        if len(df) < self.lookback:
            return

        # Calculate Candle Body Sizes (Abs(Close - Open))
        opens = df['open'].iloc[-self.lookback:]
        closes = df['close'].iloc[-self.lookback:]
        bodies = np.abs(closes - opens)

        # Current average body size (volatility baseline)
        self.avg_body_size = bodies.mean()

        # Analyze the MOST RECENT closed bar
        last_bar = df.iloc[-1]
        self.last_candle_body = abs(last_bar['close'] - last_bar['open'])
        self.last_candle_dir = 'GREEN' if last_bar['close'] > last_bar['open'] else 'RED'

    def should_block_trade(self, signal_side: str) -> Tuple[bool, Optional[str]]:
        """
        Block Reversals if the last candle was an Impulse Candle.

        Args:
            signal_side: 'LONG' or 'SHORT'

        Returns:
            Tuple of (should_block: bool, reason: Optional[str])
        """
        # Threshold: If last candle is 2.5x larger than average, it's an Impulse.
        impulse_threshold = self.avg_body_size * self.impulse_multiplier

        is_impulse = self.last_candle_body > impulse_threshold

        if not is_impulse:
            return False, None

        # BLOCK LONGs if we just had a massive RED impulse
        if signal_side == 'LONG' and self.last_candle_dir == 'RED':
            reason = f"Blocked: Catching Falling Knife (Red Impulse: {self.last_candle_body:.2f} > {impulse_threshold:.2f})"
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
            reason = f"Blocked: Fading Rocket Ship (Green Impulse: {self.last_candle_body:.2f} > {impulse_threshold:.2f})"
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
