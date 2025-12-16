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

        Args:
            df_1m_current: 1-minute OHLCV dataframe
            df_60m_current: Optional 60-minute dataframe for accurate HTF comparison
        """
        if df_1m_current.empty or len(df_1m_current) < 20:
            return False, "Insufficient Data"

        # --- STEP 1: CALCULATE CURRENT VOLATILITY (1-Min Leading Indicator) ---
        # We use the 1-minute chart as the "Tip of the Spear"
        current_1m_high = df_1m_current["high"].iloc[-self.LOOKBACK :].max()
        current_1m_low = df_1m_current["low"].iloc[-self.LOOKBACK :].min()
        current_1m_vol = current_1m_high - current_1m_low

        # --- STEP 2: 60-Min Volatility (using provided DF if available) ---
        # If we passed the 60m dataframe from the main loop, use it for accurate comparison
        if df_60m_current is not None and not df_60m_current.empty and len(df_60m_current) >= self.LOOKBACK:
            # Calculate actual 60m volatility from the provided dataframe
            current_60m_high = df_60m_current["high"].iloc[-self.LOOKBACK :].max()
            current_60m_low = df_60m_current["low"].iloc[-self.LOOKBACK :].min()
            current_60m_vol = current_60m_high - current_60m_low

            # Compare 1m volatility against actual 60m volatility for breakout detection
            if current_1m_vol > current_60m_vol * 0.8:
                return (
                    False,
                    f"🟢 HTF BREAKOUT: 1m Vol ({current_1m_vol:.2f}) approaching 60m Range ({current_60m_vol:.2f})",
                )

            # HTF Range Fade: allow directional trades at the range extremes
            current_price = df_1m_current["close"].iloc[-1]
            if current_60m_vol > 0:
                position_in_range = (current_price - current_60m_low) / current_60m_vol

                if position_in_range <= 0.15:
                    return False, "ALLOW_LONG_ONLY: At Bottom of 60M Range (Fade Support)"

                if position_in_range >= 0.85:
                    return False, "ALLOW_SHORT_ONLY: At Top of 60M Range (Fade Resistance)"

        # --- STEP 3: CHECK BREAKOUT PROPAGATION (The "Shift") ---
        # If current 1M volatility > 60M Chop Threshold, the breakout is REAL.
        # It has likely already broken the HTF chop structure.
        if current_1m_vol > self.thresholds["60M"]:
            return (
                False,
                "🟢 VOLATILITY EXPANSION: 1m Vol "
                f"({current_1m_vol:.2f}) > 60m Chop ({self.thresholds['60M']:.2f})",
            )

        # --- STEP 4: TIERED CHECKS ---

        # If 1m volatility is extremely low (Tier 3 Chop), definitely block.
        if current_1m_vol < self.thresholds["1M"]:
            return True, f"🔴 1M MICRO CHOP: Vol {current_1m_vol:.2f} < {self.thresholds['1M']:.2f}"

        # If 1m volatility is medium, we check if it is contained by the HTF thresholds.
        # This implies the market is moving, but just bouncing inside the hourly range.
        if current_1m_vol < self.thresholds["15M"]:
            return True, f"🟠 15M MID CHOP: Vol {current_1m_vol:.2f} < {self.thresholds['15M']:.2f}"

        # If we are here, 1M Vol is healthy (> 15M chop), but maybe not explosive.
        return False, "✅ MARKET ACTIVE"

    def should_recalibrate(self, last_calibration: float, interval_seconds: int = 14400) -> bool:
        """Helper to check if recalibration is due."""
        return time.time() - last_calibration > interval_seconds
