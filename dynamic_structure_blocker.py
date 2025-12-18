import pandas as pd
import numpy as np
import logging

class DynamicStructureBlocker:
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
        blocked, reason = self.should_block_trade("SHORT", None)
        return not blocked, reason
