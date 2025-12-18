import pandas as pd
import numpy as np
import logging
from collections import deque
from config import CONFIG


# =============================================================================
# PENALTY BOX BLOCKER: Fixed Tolerance + 3-Bar Decay (Simple & Aggressive)
# =============================================================================
class PenaltyBoxBlocker:
    """
    Simple structure blocker using a "penalty box" mechanism.

    LOGIC:
    - Looks back 50 candles to find ceiling (highest high) and floor (lowest low)
    - When price enters within 5.0 points of ceiling/floor, blocks trades for 3 bars
    - Fixed 5.0 point tolerance (no regime adjustment)
    - Time-based penalty box with 3-bar decay

    USE CASE:
    - Prevents buying at the top of a range
    - Prevents selling at the bottom of a range
    - Forces waiting after entering danger zones
    """

    def __init__(self, lookback: int = 50, tolerance: float = 5.0, penalty_bars: int = 3):
        """
        :param lookback: Number of candles to find ceiling/floor (default 50)
        :param tolerance: Fixed point tolerance for danger zone (default 5.0)
        :param penalty_bars: Bars to block after entering danger zone (default 3)
        """
        self.lookback = lookback
        self.tolerance = tolerance
        self.penalty_bars = penalty_bars

        # Penalty box counters (decrement each bar)
        self.long_block_counter = 0
        self.short_block_counter = 0

        # Tracked levels
        self.ceiling = None
        self.floor = None

    def update(self, df: pd.DataFrame) -> None:
        """
        Call each bar to:
        1. Update ceiling/floor levels
        2. Decay penalty box counters
        3. Check if price entered danger zones
        """
        if df is None or len(df) < self.lookback:
            return

        # Get last N bars for ceiling/floor
        lookback_df = df.tail(self.lookback)
        self.ceiling = lookback_df['high'].max()
        self.floor = lookback_df['low'].min()

        current_price = df['close'].iloc[-1]

        # Decay existing counters
        if self.long_block_counter > 0:
            self.long_block_counter -= 1
            if self.long_block_counter == 0:
                logging.info("🔓 LONG penalty box expired")

        if self.short_block_counter > 0:
            self.short_block_counter -= 1
            if self.short_block_counter == 0:
                logging.info("🔓 SHORT penalty box expired")

        # Check if price entered CEILING danger zone (blocks LONGS)
        if self.ceiling is not None:
            distance_to_ceiling = self.ceiling - current_price
            if 0 < distance_to_ceiling <= self.tolerance:
                if self.long_block_counter == 0:
                    logging.warning(f"⚠️ PENALTY BOX: Price near CEILING ({self.ceiling:.2f}), blocking LONGS for {self.penalty_bars} bars")
                self.long_block_counter = self.penalty_bars

        # Check if price entered FLOOR danger zone (blocks SHORTS)
        if self.floor is not None:
            distance_to_floor = current_price - self.floor
            if 0 < distance_to_floor <= self.tolerance:
                if self.short_block_counter == 0:
                    logging.warning(f"⚠️ PENALTY BOX: Price near FLOOR ({self.floor:.2f}), blocking SHORTS for {self.penalty_bars} bars")
                self.short_block_counter = self.penalty_bars

    def should_block_trade(self, side: str, current_price: float):
        """
        Interface method for julie001.py integration.
        Returns (blocked: bool, reason: str)
        """
        if side.upper() == "LONG":
            if self.long_block_counter > 0:
                return True, f"Penalty Box: LONG blocked ({self.long_block_counter} bars left) - Near ceiling {self.ceiling:.2f}"
            # Also instant-block if currently in danger zone
            if self.ceiling is not None and current_price is not None:
                distance = self.ceiling - current_price
                if 0 < distance <= self.tolerance:
                    return True, f"Danger Zone: Price within {distance:.2f} pts of ceiling ({self.ceiling:.2f})"

        elif side.upper() == "SHORT":
            if self.short_block_counter > 0:
                return True, f"Penalty Box: SHORT blocked ({self.short_block_counter} bars left) - Near floor {self.floor:.2f}"
            # Also instant-block if currently in danger zone
            if self.floor is not None and current_price is not None:
                distance = current_price - self.floor
                if 0 < distance <= self.tolerance:
                    return True, f"Danger Zone: Price within {distance:.2f} pts of floor ({self.floor:.2f})"

        return False, "OK"

    def can_long(self):
        """Legacy API: Returns (allowed: bool, reason: str)"""
        blocked, reason = self.should_block_trade("LONG", None)
        return not blocked, reason

    def can_short(self):
        """Legacy API: Returns (allowed: bool, reason: str)"""
        blocked, reason = self.should_block_trade("SHORT", None)
        return not blocked, reason

    def get_status(self) -> dict:
        """Get current blocker state for monitoring."""
        return {
            'ceiling': self.ceiling,
            'floor': self.floor,
            'long_block_counter': self.long_block_counter,
            'short_block_counter': self.short_block_counter,
            'tolerance': self.tolerance,
        }


# =============================================================================
# DYNAMIC STRUCTURE BLOCKER: Macro Trend + Fade Detection (Price Action)
# =============================================================================
class DynamicStructureBlocker:
    """
    Structure blocker using macro trend detection and fade setups.

    LOGIC:
    - Detects macro structure trend via fractal pivots (HH/HL = UP, LH/LL = DOWN)
    - Blocks counter-trend trades UNLESS valid fade setup detected
    - Fade setup requires: Volume Spike (2x avg) + Rejection Wick (Hammer/Shooting Star)

    USE CASE:
    - Prevents fighting strong trends without confirmation
    - Allows counter-trend trades only on high-probability reversals
    """

    def __init__(self, pivot_window: int = 5, lookback: int = 50):
        """
        :param pivot_window: Bars on left/right to confirm a fractal pivot.
        :param lookback: Minimum bars needed for trend detection.
        """
        self.pivot_window = pivot_window
        self.lookback = lookback
        self.df = None  # Stored dataframe from update()

    def update(self, df: pd.DataFrame) -> None:
        """Store the latest dataframe for blocking checks."""
        if df is not None and not df.empty:
            self.df = df.copy()

    def _find_pivots(self, df: pd.DataFrame):
        """Identify fractal swing points."""
        df = df.copy()
        # Vectorized pivot detection
        # Rolling max/min with center=True checks left and right neighbors
        df['is_pivot_high'] = df['high'].rolling(window=self.pivot_window*2+1, center=True).max() == df['high']
        df['is_pivot_low'] = df['low'].rolling(window=self.pivot_window*2+1, center=True).min() == df['low']

        return df[df['is_pivot_high']], df[df['is_pivot_low']]

    def get_structure_trend(self, df: pd.DataFrame) -> str:
        """
        Returns 'UP' (HH+HL), 'DOWN' (LH+LL), or 'NEUTRAL'.
        """
        if len(df) < self.lookback:
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

    def check_fade_setup(self, df: pd.DataFrame, signal_type: str) -> bool:
        """
        Detects Bottoms/Tops using PRICE ACTION + VOLUME (No RSI).

        Logic for LONG Fade (Bottom Catching):
        1. Volume Spike: Current Vol > 2.0x Average Vol (Capitulation).
        2. Rejection Wick: Long lower wick (Hammer pattern) OR Bullish Engulfing.

        Logic for SHORT Fade (Top Catching):
        1. Volume Spike: Current Vol > 2.0x Average Vol.
        2. Rejection Wick: Long upper wick (Shooting Star) OR Bearish Engulfing.
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
            is_vol_climax = vol_ratio > 2.0

            # 2. WICK FILTER: Is there a long lower wick?
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

    def is_trade_allowed(self, df: pd.DataFrame, signal_type: str):
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

    def should_block_trade(self, side: str, current_price: float):
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

    def can_long(self):
        """Legacy API: Returns (allowed: bool, reason: str)"""
        blocked, reason = self.should_block_trade("LONG", None)
        return not blocked, reason

    def can_short(self):
        """Legacy API: Returns (allowed: bool, reason: str)"""
        blocked, reason = self.should_block_trade("SHORT", None)
        return not blocked, reason


# =============================================================================
# REGIME STRUCTURE BLOCKER: Adaptive Tolerance by Volatility (Legacy)
# =============================================================================
class RegimeStructureBlocker:
    """
    Blocks trades at "Weak" levels using data-mined regime buckets (2023-2025 data).

    LOGIC:
    - Bidirectional checks: Treats Weak Highs/Lows as BOTH Magnets (liquidity) and Resistance/Support.
    - Prevents "Buying the Top" of a chop range (Longs at EQH).
    - Prevents "Selling the Bottom" of a chop range (Shorts at EQL).
    - Uses ADAPTIVE tolerance based on current volatility regime.

    SETTINGS VALIDATED ON MES DATA:
    - Lookback: 20 (identifies swing highs/lows every ~18 mins, filters noise)
    - Regimes: Quiet (<1.25 pt range), Normal (1.25-3.25), Volatile (>3.25)
    """

    def __init__(self, lookback: int = 20):
        self.swings_high = deque(maxlen=10)
        self.swings_low = deque(maxlen=10)
        self.lookback = lookback

        # --- DATA-DRIVEN REGIME BUCKETS ---
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
        """Update volatility regime based on recent candle ranges."""
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
        """Update swing levels and regime on each bar."""
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

        return False, "OK"

    def can_long(self):
        """Legacy API: Returns (allowed: bool, reason: str)"""
        # Note: Requires current_price for full check, returns OK if no price
        return True, "OK (use should_block_trade with price for full check)"

    def can_short(self):
        """Legacy API: Returns (allowed: bool, reason: str)"""
        # Note: Requires current_price for full check, returns OK if no price
        return True, "OK (use should_block_trade with price for full check)"
