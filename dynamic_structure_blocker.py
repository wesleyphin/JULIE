import pandas as pd
import numpy as np
import logging

class DynamicStructureBlocker:
    def __init__(self, lookback=50, pivot_window=5):
        """
        :param lookback: Bars to look back for EQH/EQL detection.
        :param pivot_window: Bars on left/right to confirm a fractal pivot.
        """
        # === OLD LOGIC: EQH/EQL Proximity ===
        self.lookback = lookback
        self.tolerance = 5.0
        self.block_duration = 3
        self.long_block_counter = 0
        self.short_block_counter = 0
        self.last_processed_time = None

        # === NEW LOGIC: Structure Trend + Fade ===
        self.pivot_window = pivot_window
        self.df = None  # Stored dataframe for fade checks

    # =========================================================================
    # OLD LOGIC: EQH/EQL PROXIMITY BLOCKING (Penalty Box)
    # =========================================================================

    def update(self, df):
        """
        Call this on every new candle close to update the blocking state.
        """
        if df is None or df.empty:
            return

        # Store df for new logic
        self.df = df.copy()

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
        if current_close < prev_high:
            dist_to_high = prev_high - current_close
            if dist_to_high <= self.tolerance:
                self.long_block_counter = self.block_duration
                logging.info(f"⛔ STRUCTURE: Price {current_close} is within {self.tolerance}pts of EQH {prev_high}. Blocking Longs for {self.block_duration} bars.")

        # --- LOGIC B: Check EQL (Support) -> Blocks Shorts ---
        if current_close > prev_low:
            dist_to_low = current_close - prev_low
            if dist_to_low <= self.tolerance:
                self.short_block_counter = self.block_duration
                logging.info(f"⛔ STRUCTURE: Price {current_close} is within {self.tolerance}pts of EQL {prev_low}. Blocking Shorts for {self.block_duration} bars.")

    def can_long(self):
        """OLD LOGIC: Returns (Bool, Reason)"""
        if self.long_block_counter > 0:
            return False, f"Blocked by EQH (Wait {self.long_block_counter} bars)"
        return True, "OK"

    def can_short(self):
        """OLD LOGIC: Returns (Bool, Reason)"""
        if self.short_block_counter > 0:
            return False, f"Blocked by EQL (Wait {self.short_block_counter} bars)"
        return True, "OK"

    # =========================================================================
    # NEW LOGIC: STRUCTURE TREND + FADE DETECTION
    # =========================================================================

    def _find_pivots(self, df):
        """Identify fractal swing points."""
        df = df.copy()
        df['is_pivot_high'] = df['high'].rolling(window=self.pivot_window*2+1, center=True).max() == df['high']
        df['is_pivot_low'] = df['low'].rolling(window=self.pivot_window*2+1, center=True).min() == df['low']
        return df[df['is_pivot_high']], df[df['is_pivot_low']]

    def get_structure_trend(self, df):
        """Returns 'UP' (HH+HL), 'DOWN' (LH+LL), or 'NEUTRAL'."""
        if len(df) < 50:
            return "NEUTRAL"
        highs, lows = self._find_pivots(df)

        if len(highs) < 2 or len(lows) < 2:
            return "NEUTRAL"

        last_h, prev_h = highs.iloc[-1]['high'], highs.iloc[-2]['high']
        last_l, prev_l = lows.iloc[-1]['low'], lows.iloc[-2]['low']

        if last_h < prev_h and last_l < prev_l:
            return "DOWN"
        elif last_h > prev_h and last_l > prev_l:
            return "UP"

        return "NEUTRAL"

    def check_fade_setup(self, df, signal_type):
        """
        Detects Bottoms/Tops using PRICE ACTION + VOLUME.
        Volume Spike (>2x avg) + Rejection Wick or Engulfing pattern.
        """
        if len(df) < 20:
            return False

        current = df.iloc[-1]
        avg_vol = df['volume'].rolling(window=20).mean().iloc[-1]
        vol_ratio = current['volume'] / avg_vol if avg_vol > 0 else 1.0

        open_ = current['open']
        close = current['close']
        high = current['high']
        low = current['low']
        range_ = high - low

        if range_ == 0:
            return False

        # --- FADE A DOWNTREND (Buy the Bottom) ---
        if signal_type == "LONG":
            is_vol_climax = vol_ratio > 2.0
            lower_wick = min(open_, close) - low
            is_hammer = (lower_wick / range_) > 0.40

            prev_close = df['close'].iloc[-2]
            prev_open = df['open'].iloc[-2]
            is_engulfing = (close > prev_open) and (open_ < prev_close) and (close > open_)

            if is_vol_climax and (is_hammer or is_engulfing):
                return True

        # --- FADE AN UPTREND (Short the Top) ---
        elif signal_type == "SHORT":
            is_vol_climax = vol_ratio > 2.0
            upper_wick = high - max(open_, close)
            is_shooting_star = (upper_wick / range_) > 0.40

            prev_close = df['close'].iloc[-2]
            prev_open = df['open'].iloc[-2]
            is_engulfing = (close < prev_open) and (open_ > prev_close) and (close < open_)

            if is_vol_climax and (is_shooting_star or is_engulfing):
                return True

        return False

    def check_trend_block(self, signal_type):
        """
        NEW LOGIC: Check if trade is blocked by structure trend.
        Returns (blocked: bool, reason: str)
        """
        if self.df is None or len(self.df) < 50:
            return False, "OK"

        trend = self.get_structure_trend(self.df)

        # Trend is DOWN, Signal is LONG -> Block unless fade setup
        if trend == "DOWN" and signal_type == "LONG":
            if self.check_fade_setup(self.df, "LONG"):
                logging.info(f"✅ TREND: Fade ALLOWED - Vol Climax + Rejection Wick detected.")
                return False, "Fade setup valid: Vol Climax + Rejection Wick"
            else:
                logging.info(f"⛔ TREND: BLOCKED - Trend is DOWN and no Volume/Wick Rejection.")
                return True, "Trend is DOWN and no Volume/Wick Rejection"

        # Trend is UP, Signal is SHORT -> Block unless fade setup
        if trend == "UP" and signal_type == "SHORT":
            if self.check_fade_setup(self.df, "SHORT"):
                logging.info(f"✅ TREND: Fade ALLOWED - Vol Climax + Shooting Star detected.")
                return False, "Fade setup valid: Vol Climax + Shooting Star"
            else:
                logging.info(f"⛔ TREND: BLOCKED - Trend is UP and no Volume/Wick Rejection.")
                return True, "Trend is UP and no Volume/Wick Rejection"

        return False, "OK"

    # =========================================================================
    # MASTER CHECK: BOTH LOGICS COMBINED
    # =========================================================================

    def should_block_trade(self, side, current_price):
        """
        Master interface for julie001.py. Runs BOTH checks independently.
        Returns (blocked: bool, reason: str)
        """
        signal_type = "LONG" if side.upper() == "LONG" else "SHORT"

        # CHECK 1: Old Logic (EQH/EQL Proximity)
        if signal_type == "LONG":
            eqh_allowed, eqh_reason = self.can_long()
        else:
            eqh_allowed, eqh_reason = self.can_short()

        if not eqh_allowed:
            return True, eqh_reason  # Blocked by EQH/EQL

        # CHECK 2: New Logic (Structure Trend + Fade)
        trend_blocked, trend_reason = self.check_trend_block(signal_type)

        if trend_blocked:
            return True, trend_reason  # Blocked by Trend

        return False, "OK"
