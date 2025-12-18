import pandas as pd
import numpy as np
import logging
from collections import deque
from config import CONFIG


# =============================================================================
# NEW IMPLEMENTATION: Penalty Box Blocker (Simple & Aggressive)
# =============================================================================
class DynamicStructureBlocker:
    """
    Simple structure blocker using a "penalty box" mechanism.

    - Looks back 50 candles to find ceiling/floor
    - Blocks trades for 3 bars when price enters the danger zone
    - Fixed 5.0 point tolerance
    """

    def __init__(self):
        # Configuration
        self.lookback = 50          # Look back 50 candles to find the "Ceiling" and "Floor"

        # --- CRITICAL UPDATE: WIDENED TOLERANCE ---
        # Your High was 6791.75, Entry was 6787.5 (Diff 4.25).
        # We set tolerance to 5.0 to ensure this 4.25 gap is detected as "Near EQH".
        self.tolerance = 5.0

        # --- PERSISTENT BLOCKING (The "Penalty Box") ---
        # If we touch the danger zone, we stay blocked for 3 bars.
        # This prevents the "19:20" trade where price dipped slightly but was still dangerous.
        self.block_duration = 3

        # State variables
        self.long_block_counter = 0
        self.short_block_counter = 0
        self.last_processed_time = None

    def update(self, df):
        """
        Call this on every new candle close to update the blocking state.
        """
        if df.empty:
            return

        # 1. State Management: Process each candle exactly once
        current_time = df.index[-1]
        if self.last_processed_time == current_time:
            return
        self.last_processed_time = current_time

        # 2. Decrement existing blocks (Count down the penalty timer)
        if self.long_block_counter > 0:
            self.long_block_counter -= 1
        if self.short_block_counter > 0:
            self.short_block_counter -= 1

        # 3. Detect Structure (Instant Scan)
        recent_data = df.tail(self.lookback)
        current_close = recent_data.iloc[-1]['close']

        # Find the absolute High/Low of the recent window (excluding current bar to avoid repainting)
        if len(recent_data) > 1:
            prev_high = recent_data.iloc[:-1]['high'].max()
            prev_low = recent_data.iloc[:-1]['low'].min()
        else:
            return

        # --- LOGIC A: Check EQH (Resistance) -> Blocks Longs ---
        # We only block if we are BELOW the high (approaching resistance).
        # If we are above it, it's a breakout (we don't block).
        if current_close < prev_high:
            dist_to_high = prev_high - current_close

            # If within 5.0 points of the high...
            if dist_to_high <= self.tolerance:
                # ...TRIGGER THE PENALTY BOX
                self.long_block_counter = self.block_duration
                logging.info(f"⛔ STRUCTURE: Price {current_close} is within {self.tolerance}pts of EQH {prev_high}. Blocking Longs for {self.block_duration} bars.")

        # --- LOGIC B: Check EQL (Support) -> Blocks Shorts ---
        # We only block if we are ABOVE the low (approaching support).
        if current_close > prev_low:
            dist_to_low = current_close - prev_low

            # If within 5.0 points of the low...
            if dist_to_low <= self.tolerance:
                # ...TRIGGER THE PENALTY BOX
                self.short_block_counter = self.block_duration
                logging.info(f"⛔ STRUCTURE: Price {current_close} is within {self.tolerance}pts of EQL {prev_low}. Blocking Shorts for {self.block_duration} bars.")

    def can_long(self):
        """Returns (Bool, Reason)"""
        if self.long_block_counter > 0:
            return False, f"Blocked by EQH (Wait {self.long_block_counter} bars)"
        return True, "OK"

    def can_short(self):
        """Returns (Bool, Reason)"""
        if self.short_block_counter > 0:
            return False, f"Blocked by EQL (Wait {self.short_block_counter} bars)"
        return True, "OK"


# =============================================================================
# LEGACY IMPLEMENTATION: Regime-Based Blocker (Adaptive & Nuanced)
# =============================================================================
class RegimeStructureBlocker:
    """
    Blocks trades at "Weak" levels using data-mined regime buckets (2023-2025 data).

    LOGIC:
    - Bidirectional checks: Treats Weak Highs/Lows as BOTH Magnets (liquidity) and Resistance/Support.
    - Prevents "Buying the Top" of a chop range (Longs at EQH).
    - Prevents "Selling the Bottom" of a chop range (Shorts at EQL).

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
        """
        Check BOTH sides of structure for every trade signal.
        1. Magnet Check: Don't fade a weak level (it will likely get swept).
        2. Barrier Check: Don't trade directly into a weak level (it is resistance/support).
        """
        tolerance = self.current_tolerance

        # =========================
        # SHORTS
        # =========================
        if signal_side == "SHORT":
            if self.market_trend == "BULLISH":
                tolerance *= 1.5

            # 1. Magnet Check: Don't Short Weak Highs (EQH)
            # Logic: Price will likely go UP to sweep this high before dropping.
            for swing in self.swings_high:
                if abs(current_price - swing["price"]) < tolerance:
                    if not swing["strong"]:
                        return True, f"Blocked: Weak EQH Magnet ({swing['price']:.2f}) [{self.current_regime}]"

            # 2. Barrier Check: Don't Short into Weak Support (EQL)
            # Logic: If we are at the bottom of a range, don't short the floor. Wait for breakdown.
            for swing in self.swings_low:
                if abs(current_price - swing["price"]) < tolerance:
                    # If it's weak, it might hold as range support first
                    return True, f"Blocked: Selling into Weak Support EQL ({swing['price']:.2f})"

        # =========================
        # LONGS
        # =========================
        if signal_side == "LONG":
            if self.market_trend == "BEARISH":
                tolerance *= 1.5

            # 1. Magnet Check: Don't Long Weak Lows (EQL)
            # Logic: Price will likely go DOWN to sweep this low before rallying.
            for swing in self.swings_low:
                if abs(current_price - swing["price"]) < tolerance:
                    if not swing["strong"]:
                        return True, f"Blocked: Weak EQL Magnet ({swing['price']:.2f}) [{self.current_regime}]"

            # 2. Barrier Check: Don't Long into Weak Resistance (EQH)
            # Logic: If we are at the top of a range, don't buy the ceiling. Wait for breakout.
            for swing in self.swings_high:
                if abs(current_price - swing["price"]) < tolerance:
                    # If it's weak, it acts as resistance until proven otherwise
                    return True, f"Blocked: Buying into Weak Resistance EQH ({swing['price']:.2f})"

        return False, None
