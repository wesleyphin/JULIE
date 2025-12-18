import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class DynamicChopAnalyzer:
    """Analyze chop using dynamic thresholds with breakout and HTF fade logic."""

    def __init__(self, client):
        self.client = client
        self.thresholds: Dict[str, float] = {
            "1M": 2.0,  # Default fallback
            "5M": 4.25,
            "15M": 6.75,
            "60M": 12.50,
        }
        # Rolling window for calculating range (High-Low)
        self.LOOKBACK = 20

    def calibrate(self, days_lookback: int = 30):
        """
        Calibrates thresholds using the 20th percentile of recent data.
        Call this on bot startup and every 4-6 hours.
        """
        del days_lookback  # unused but kept for future tuning
        try:
            # 1. Fetch 60-Minute Data (Tier 1)
            # We need enough bars for statistical significance (~1 month = ~500 hourly bars)
            df_60 = self.client.fetch_custom_bars(lookback_bars=500, minutes_per_bar=60)
            if not df_60.empty:
                r_60 = (
                    df_60["high"].rolling(self.LOOKBACK).max()
                    - df_60["low"].rolling(self.LOOKBACK).min()
                ).dropna()
                self.thresholds["60M"] = float(np.percentile(r_60, 20))  # Bottom 20%
                logging.info(
                    "[DynamicChop] Calibrated 60M Threshold: %.2f",
                    self.thresholds["60M"],
                )

            # 2. Fetch 15-Minute Data (Tier 2)
            df_15 = self.client.fetch_custom_bars(lookback_bars=500, minutes_per_bar=15)
            if not df_15.empty:
                r_15 = (
                    df_15["high"].rolling(self.LOOKBACK).max()
                    - df_15["low"].rolling(self.LOOKBACK).min()
                ).dropna()
                self.thresholds["15M"] = float(np.percentile(r_15, 20))
                logging.info(
                    "[DynamicChop] Calibrated 15M Threshold: %.2f",
                    self.thresholds["15M"],
                )

            # 3. Fetch 1-Minute Data (Tier 3)
            # Standard get_market_data usually gets 1m bars
            df_1 = self.client.get_market_data(lookback_minutes=1000)
            if not df_1.empty:
                r_1 = (
                    df_1["high"].rolling(self.LOOKBACK).max()
                    - df_1["low"].rolling(self.LOOKBACK).min()
                ).dropna()
                self.thresholds["1M"] = float(np.percentile(r_1, 20))
                logging.info(
                    "[DynamicChop] Calibrated 1M Threshold: %.2f",
                    self.thresholds["1M"],
                )

        except Exception as e:  # pragma: no cover - defensive logging
            logging.error("[DynamicChop] Calibration Error: %s", e)

    def check_market_state(
        self, df_1m_current: pd.DataFrame, df_60m_current: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, str]:
        """
        Determines market state with HTF breakout and fade opportunities.
        Returns: (is_blocked, reason)
        """
        if df_1m_current.empty or len(df_1m_current) < 20:
            return False, "Insufficient Data"

        # --- STEP 1: CALCULATE CURRENT VOLATILITY (1-Min Leading Indicator) ---
        current_1m_high = df_1m_current["high"].iloc[-self.LOOKBACK :].max()
        current_1m_low = df_1m_current["low"].iloc[-self.LOOKBACK :].min()
        current_1m_vol = current_1m_high - current_1m_low

        # --- STEP 2: 60-Min Volatility (using provided DF if available) ---
        if df_60m_current is not None and not df_60m_current.empty and len(df_60m_current) >= self.LOOKBACK:
            current_60m_high = df_60m_current["high"].iloc[-self.LOOKBACK :].max()
            current_60m_low = df_60m_current["low"].iloc[-self.LOOKBACK :].min()
            current_60m_vol = current_60m_high - current_60m_low

            if current_1m_vol > current_60m_vol * 0.8:
                return (
                    False,
                    f"🟢 HTF BREAKOUT: 1m Vol ({current_1m_vol:.2f}) approaching 60m Range ({current_60m_vol:.2f})",
                )

            current_price = df_1m_current["close"].iloc[-1]
            if current_60m_vol > 0:
                position_in_range = (current_price - current_60m_low) / current_60m_vol

                if position_in_range <= 0.15:
                    return False, "ALLOW_LONG_ONLY: At Bottom of 60M Range (Fade Support)"

                if position_in_range >= 0.85:
                    return False, "ALLOW_SHORT_ONLY: At Top of 60M Range (Fade Resistance)"

        # --- STEP 3: CHECK BREAKOUT PROPAGATION ---
        if current_1m_vol > self.thresholds["60M"]:
            return (
                False,
                "🟢 VOLATILITY EXPANSION: 1m Vol "
                f"({current_1m_vol:.2f}) > 60m Chop ({self.thresholds['60M']:.2f})",
            )

        # --- STEP 4: TIERED CHECKS ---
        if current_1m_vol < self.thresholds["1M"]:
            return True, f"🔴 1M MICRO CHOP: Vol {current_1m_vol:.2f} < {self.thresholds['1M']:.2f}"

        if current_1m_vol < self.thresholds["15M"]:
            return True, f"🟠 15M MID CHOP: Vol {current_1m_vol:.2f} < {self.thresholds['15M']:.2f}"

        return False, "✅ MARKET ACTIVE"

    def check_target_feasibility(self, entry_price: float, side: str, tp_distance: float, df_1m: pd.DataFrame) -> Tuple[bool, str]:
        """
        NEW FEASIBILITY CHECK (Fix for 20:56-21:00 Issue):

        If we are in a CHOP regime (Low Volatility), ensure the TP target
        sits INSIDE the current range (Box).

        If the TP extends past the High/Low of the chop, it implies a breakout
        is needed to win, which is low probability in chop. We block it.
        """
        if df_1m.empty or len(df_1m) < self.LOOKBACK:
             return True, "Insufficient Data"

        # 1. Define the Chop Box (High/Low of last 20 bars)
        box_high = df_1m["high"].iloc[-self.LOOKBACK:].max()
        box_low = df_1m["low"].iloc[-self.LOOKBACK:].min()
        current_vol = box_high - box_low

        # 2. Determine Regime: Are we in Chop?
        # We use the 15M Threshold as the dividing line.
        # If Vol > 15M Threshold, we are "Active/Trending" -> Allow Breakouts (Return True)
        if current_vol > self.thresholds.get("15M", 6.75):
            return True, "Trend/Breakout Regime (Target Check Skipped)"

        # 3. We are in CHOP. Enforce the Box Constraint.
        if side.upper() == "LONG":
            target = entry_price + tp_distance
            # If target is ABOVE the box high (requires breakout), Block.
            if target > box_high:
                return False, f"⛔ CHOP BOUND: TP ({target:.2f}) extends past Range High ({box_high:.2f}). Unlikely to hit."

        elif side.upper() == "SHORT":
            target = entry_price - tp_distance
            # If target is BELOW the box low (requires breakout), Block.
            if target < box_low:
                return False, f"⛔ CHOP BOUND: TP ({target:.2f}) extends past Range Low ({box_low:.2f}). Unlikely to hit."

        return True, "Target Feasible in Range"

    def should_recalibrate(self, last_calibration: float, interval_seconds: int = 14400) -> bool:
        """Helper to check if recalibration is due."""
        return time.time() - last_calibration > interval_seconds
