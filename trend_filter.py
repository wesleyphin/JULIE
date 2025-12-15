import pandas as pd
from typing import Tuple


class TrendFilter:
    """
    Filters counter-trend trades when the trend is extremely strong.
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def should_block_trade(self, df: pd.DataFrame, side: str) -> Tuple[bool, str]:
        if len(df) < self.slow_period:
            return False, ""

        closes = df["close"]
        ema_fast = closes.ewm(span=self.fast_period, adjust=False).mean().iloc[-1]
        ema_slow = closes.ewm(span=self.slow_period, adjust=False).mean().iloc[-1]
        price = closes.iloc[-1]

        # Block Shorts in Strong Uptrend
        if price > ema_fast > ema_slow:
            if side == "SHORT":
                return True, f"Trend Filter: Price > {self.fast_period}EMA > {self.slow_period}EMA (Strong Bullish)"

        # Block Longs in Strong Downtrend
        if price < ema_fast < ema_slow:
            if side == "LONG":
                return True, f"Trend Filter: Price < {self.fast_period}EMA < {self.slow_period}EMA (Strong Bearish)"

        return False, ""
