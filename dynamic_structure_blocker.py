import pandas as pd
import numpy as np
import logging
from config import CONFIG

class DynamicStructureBlocker:
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
