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
    def __init__(self, pivot_window=5, lookback=50):
        """
        :param pivot_window: Bars on left/right to confirm a fractal pivot.
        :param lookback: Legacy parameter for backwards compatibility.
        """
        self.pivot_window = pivot_window
        self.lookback = lookback
        self.df = None  # Stored dataframe from update()

    def update(self, df):
        """Store the latest dataframe for blocking checks."""
        if df is not None and not df.empty:
            self.df = df.copy()

    def _find_pivots(self, df):
        """Identify fractal swing points."""
        df = df.copy()
        # Vectorized pivot detection
        # Rolling max/min with center=True checks left and right neighbors
        df['is_pivot_high'] = df['high'].rolling(window=self.pivot_window*2+1, center=True).max() == df['high']
        df['is_pivot_low'] = df['low'].rolling(window=self.pivot_window*2+1, center=True).min() == df['low']

        return df[df['is_pivot_high']], df[df['is_pivot_low']]

    def get_structure_trend(self, df):
        """
        Returns 'UP' (HH+HL), 'DOWN' (LH+LL), or 'NEUTRAL'.
        """
        if len(df) < 50:
            return "NEUTRAL"
        highs, lows = self._find_pivots(df)

        if len(highs) < 2 or len(lows) < 2:
            return "NEUTRAL"

        # Compare last two confirmed pivots
        last_h, prev_h = highs.iloc[-1]['high'], highs.iloc[-2]['high']
        last_l, prev_l = lows.iloc[-1]['low'], lows.iloc[-2]['low']

        if last_h < prev_h and last_l < prev_l:
            return "DOWN"
        elif last_h > prev_h and last_l > prev_l:
            return "UP"

        return "NEUTRAL"

    def check_fade_setup(self, df, signal_type):
        """
        Detects Bottoms/Tops using PRICE ACTION + VOLUME (No RSI).

        Logic for LONG Fade (Bottom Catching):
        1. Volume Spike: Current Vol > 2.0x Average Vol (Capitulation).
        2. Rejection Wick: Long lower wick (Hammer pattern).
        3. Extension: Price is below the Lower Bollinger Band (Overextended).
        """
        if len(df) < 20:
            return False

        current = df.iloc[-1]
        avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
        vol_ratio = current['volume'] / avg_vol if avg_vol > 0 else 1.0

        # Calculate Candle Body & Wicks
        open_ = current['open']
        close = current['close']
        high = current['high']
        low = current['low']
        range_ = high - low

        if range_ == 0:
            return False

        # --- FADE A DOWNTREND (Buy the Bottom) ---
        if signal_type == "LONG":
            # 1. VOLUME FILTER: Is this a "Stopping Volume" event?
            # Data shows bottoms often have spikes > 2.0x normal volume
            is_vol_climax = vol_ratio > 2.0

            # 2. WICK FILTER: Is there a long lower wick?
            # Rejection: The candle dipped but buyers pushed it back up into the top 40%
            lower_wick = min(open_, close) - low
            is_hammer = (lower_wick / range_) > 0.40  # Wick is 40% of the candle

            # 3. ENGULFING FILTER: Did we totally reverse the previous Red candle?
            prev_close = df['close'].iloc[-2]
            prev_open = df['open'].iloc[-2]
            is_engulfing = (close > prev_open) and (open_ < prev_close) and (close > open_)

            # TRIGGER: Volume Spike + (Hammer OR Engulfing)
            if is_vol_climax and (is_hammer or is_engulfing):
                return True

        # --- FADE AN UPTREND (Short the Top) ---
        elif signal_type == "SHORT":
            is_vol_climax = vol_ratio > 2.0

            # Upper Wick Rejection (Shooting Star)
            upper_wick = high - max(open_, close)
            is_shooting_star = (upper_wick / range_) > 0.40

            # Bearish Engulfing
            prev_close = df['close'].iloc[-2]
            prev_open = df['open'].iloc[-2]
            is_engulfing = (close < prev_open) and (open_ > prev_close) and (close < open_)

            if is_vol_climax and (is_shooting_star or is_engulfing):
                return True

        return False

    def is_trade_allowed(self, df, signal_type):
        """
        Master check:
        1. Check Macro Structure (HH/LL).
        2. If Trend opposes Signal, allow ONLY if Fade Setup is valid.
        """
        trend = self.get_structure_trend(df)

        # Case 1: Trend says DOWN, Signal is LONG
        if trend == "DOWN" and signal_type == "LONG":
            if self.check_fade_setup(df, "LONG"):
                logging.info(f"✅ Trade ALLOWED (Fade): Vol Climax + Rejection Wick detected.")
                return True, "Fade setup valid: Vol Climax + Rejection Wick"
            else:
                logging.info(f"⛔ Trade BLOCKED: Trend is DOWN and no Volume/Wick Rejection.")
                return False, "Trend is DOWN and no Volume/Wick Rejection"

        # Case 2: Trend says UP, Signal is SHORT
        if trend == "UP" and signal_type == "SHORT":
            if self.check_fade_setup(df, "SHORT"):
                logging.info(f"✅ Trade ALLOWED (Fade): Vol Climax + Shooting Star detected.")
                return True, "Fade setup valid: Vol Climax + Shooting Star"
            else:
                logging.info(f"⛔ Trade BLOCKED: Trend is UP and no Volume/Wick Rejection.")
                return False, "Trend is UP and no Volume/Wick Rejection"

        return True, "OK"

    def should_block_trade(self, side, current_price):
        """
        Interface method for julie001.py integration.
        Returns (blocked: bool, reason: str)
        """
        if self.df is None or len(self.df) < 20:
            return False, "OK"

        # Convert side to signal_type
        signal_type = "LONG" if side.upper() == "LONG" else "SHORT"

        # Use is_trade_allowed and invert for blocking logic
        allowed, reason = self.is_trade_allowed(self.df, signal_type)

        if allowed:
            return False, reason  # Not blocked
        else:
            return True, reason   # Blocked

    # Legacy methods for backwards compatibility
    def can_long(self):
        """Returns (Bool, Reason)"""
        blocked, reason = self.should_block_trade("LONG", None)
        return not blocked, reason

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
        blocked, reason = self.should_block_trade("SHORT", None)
        return not blocked, reason
