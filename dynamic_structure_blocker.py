import numpy as np
import pandas as pd
from collections import deque


class DynamicStructureBlocker:
    """
    Blocks trades at "Weak" levels using data-mined regime buckets (2023-2025 data).

    SETTINGS VALIDATED ON MES DATA:
    - Lookback: 20 (identifies swing highs/lows every ~18 mins, filters noise)
    - Regimes: Quiet (<1.25), Normal (1.25-3.25), Volatile (>3.25)
    """

    def __init__(self, lookback: int = 20):
        self.swings_high = deque(maxlen=10)
        self.swings_low = deque(maxlen=10)
        self.lookback = lookback

        # --- DATA-DRIVEN SETTINGS ---
        self.BUCKETS = {
            "QUIET": {"max_range": 1.25, "tolerance": 0.75},
            "NORMAL": {"max_range": 3.25, "tolerance": 1.50},
            "VOLATILE": {"max_range": 999, "tolerance": 3.00},
        }

        self.current_regime = "NORMAL"
        self.current_tolerance = 1.50
        self.market_trend = "NEUTRAL"
        self.last_structure_high = -np.inf
        self.last_structure_low = np.inf

    def _update_regime(self, df: pd.DataFrame) -> None:
        if len(df) < 5:
            return
        avg_range = (df["high"] - df["low"]).tail(5).mean()

        if avg_range <= self.BUCKETS["QUIET"]["max_range"]:
            self.current_regime = "QUIET"
            self.current_tolerance = self.BUCKETS["QUIET"]["tolerance"]
        elif avg_range <= self.BUCKETS["NORMAL"]["max_range"]:
            self.current_regime = "NORMAL"
            self.current_tolerance = self.BUCKETS["NORMAL"]["tolerance"]
        else:
            self.current_regime = "VOLATILE"
            self.current_tolerance = self.BUCKETS["VOLATILE"]["tolerance"]

    def update(self, df: pd.DataFrame) -> None:
        # Need Lookback * 2 + buffer to confirm swing
        if len(df) < (self.lookback * 2) + 5:
            return

        self._update_regime(df)

        # Identify swings [Current - Lookback]
        curr_idx = len(df) - 1 - self.lookback
        curr_high = df["high"].iloc[curr_idx]
        curr_low = df["low"].iloc[curr_idx]

        # Check fractal high
        is_high = True
        for i in range(1, self.lookback + 1):
            if df["high"].iloc[curr_idx - i] >= curr_high or df["high"].iloc[curr_idx + i] >= curr_high:
                is_high = False
                break

        # Check fractal low
        is_low = True
        for i in range(1, self.lookback + 1):
            if df["low"].iloc[curr_idx - i] <= curr_low or df["low"].iloc[curr_idx + i] <= curr_low:
                is_low = False
                break

        # Update structure
        if is_high:
            is_strong = curr_high > self.last_structure_high
            if is_strong:
                self.last_structure_high = curr_high
                if self.market_trend != "BULLISH":
                    self.market_trend = "NEUTRAL"
            self.swings_high.append({"price": curr_high, "strong": is_strong})

            # Trend check: lower highs
            if len(self.swings_high) >= 2 and self.swings_high[-1]["price"] < self.swings_high[-2]["price"]:
                self.market_trend = "BEARISH"

        if is_low:
            is_strong = curr_low < self.last_structure_low
            if is_strong:
                self.last_structure_low = curr_low
                if self.market_trend != "BEARISH":
                    self.market_trend = "NEUTRAL"
            self.swings_low.append({"price": curr_low, "strong": is_strong})

            # Trend check: higher lows
            if len(self.swings_low) >= 2 and self.swings_low[-1]["price"] > self.swings_low[-2]["price"]:
                self.market_trend = "BULLISH"

    def should_block_trade(self, signal_side: str, current_price: float):
        tolerance = self.current_tolerance

        # Block shorts at weak EQH
        if signal_side == "SHORT":
            if self.market_trend == "BULLISH":
                tolerance *= 1.5
            for swing in self.swings_high:
                if abs(current_price - swing["price"]) < tolerance:
                    if not swing["strong"]:
                        return True, f"Blocked: Weak EQH ({swing['price']:.2f}) [{self.current_regime}]"

        # Block longs at weak EQL
        if signal_side == "LONG":
            if self.market_trend == "BEARISH":
                tolerance *= 1.5
            for swing in self.swings_low:
                if abs(current_price - swing["price"]) < tolerance:
                    if not swing["strong"]:
                        return True, f"Blocked: Weak EQL ({swing['price']:.2f}) [{self.current_regime}]"

        return False, None
