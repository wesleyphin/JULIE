import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


class DynamicChopAnalyzer:
    """
    Analyze chop using dynamic thresholds with breakout and HTF fade logic.
    Now supports Gemini AI optimization via multipliers.
    """

    def __init__(self, client):
        self.client = client
        # Base thresholds (calibrated via historical data)
        self.base_thresholds: Dict[str, float] = {
            "1M": 2.0,  # Default fallback
            "5M": 4.25,
            "15M": 6.75,
            "60M": 12.50,
        }
        self.LOOKBACK = 20
        # Multiplier from Gemini (Default 1.0 = No change)
        self.gemini_multiplier = 1.0

    def update_gemini_params(self, multiplier: float):
        """Called by the bot when Gemini returns new optimization."""
        self.gemini_multiplier = multiplier
        logging.info(f"[DynamicChop] Updated Gemini Multiplier: {self.gemini_multiplier}x")

    def get_active_threshold(self, timeframe: str) -> float:
        """Returns the threshold adjusted by Gemini."""
        base = self.base_thresholds.get(timeframe, 2.0)
        return base * self.gemini_multiplier

    def calibrate(self, days_lookback: int = 30):
        """
        Calibrates thresholds using the 20th percentile of recent data.
        HYBRID LOGIC:
        - 60M: Old Logic (High Ceiling)
        - 15M: Compromise Logic (1 Hour Window)
        - 1M: Standard Logic (20 Min Window)
        """
        del days_lookback
        try:
            # 1. Fetch 60-Minute Data (Tier 1)
            # OLD LOGIC: Use LOOKBACK (20) -> 20 hours of data.
            # Keeps the Macro Ceiling HIGH to allow fades.
            df_60 = self.client.fetch_custom_bars(lookback_bars=500, minutes_per_bar=60)
            if not df_60.empty:
                r_60 = (
                    df_60["high"].rolling(self.LOOKBACK).max()
                    - df_60["low"].rolling(self.LOOKBACK).min()
                ).dropna()
                self.base_thresholds["60M"] = float(np.percentile(r_60, 20))
                logging.info(
                    "[DynamicChop] Calibrated 60M Threshold: %.2f (Old Logic)",
                    self.base_thresholds["60M"],
                )

            # 2. Fetch 15-Minute Data (Tier 2) - THE COMPROMISE
            # -----------------------------------------------------------------
            # rolling(4) = 4 * 15m = 60 Minutes of Data.
            # - Old (rolling 20) = 300 Mins (Too High/Loose)
            # - New (rolling 2)  = 30 Mins (Too Low/Tight)
            # - This (rolling 4) = 60 Mins (Just Right)
            # -----------------------------------------------------------------
            df_15 = self.client.fetch_custom_bars(lookback_bars=500, minutes_per_bar=15)
            if not df_15.empty:
                r_15 = (
                    df_15["high"].rolling(4).max()
                    - df_15["low"].rolling(4).min()
                ).dropna()
                self.base_thresholds["15M"] = float(np.percentile(r_15, 20))
                logging.info(
                    "[DynamicChop] Calibrated 15M Threshold: %.2f (Hybrid Logic: 1H Window)",
                    self.base_thresholds["15M"],
                )

            # 3. Fetch 1-Minute Data (Tier 3)
            # STANDARD LOGIC: Use LOOKBACK (20) -> 20 minutes of data.
            df_1 = self.client.get_market_data(lookback_minutes=1000)
            if not df_1.empty:
                r_1 = (
                    df_1["high"].rolling(self.LOOKBACK).max()
                    - df_1["low"].rolling(self.LOOKBACK).min()
                ).dropna()
                self.base_thresholds["1M"] = float(np.percentile(r_1, 20))
                logging.info(
                    "[DynamicChop] Calibrated 1M Threshold: %.2f",
                    self.base_thresholds["1M"],
                )

        except Exception as e:
            logging.error("[DynamicChop] Calibration Error: %s", e)

    def check_market_state(
        self, df_1m_current: pd.DataFrame, df_60m_current: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, str]:
        """
        Determines market state with HTF breakout and fade logic.
        Uses GEMINI MULTIPLIER to adjust sensitivity.
        """
        if df_1m_current.empty or len(df_1m_current) < 20:
            return False, "Insufficient Data"

        # --- STEP 1: CALCULATE CURRENT VOLATILITY (1-Min Leading Indicator) ---
        current_1m_high = df_1m_current["high"].iloc[-self.LOOKBACK :].max()
        current_1m_low = df_1m_current["low"].iloc[-self.LOOKBACK :].min()
        current_1m_vol = current_1m_high - current_1m_low

        # --- STEP 2: GET ACTIVE THRESHOLDS (SCALED BY GEMINI) ---
        thresh_1m = self.get_active_threshold("1M")
        thresh_15m = self.get_active_threshold("15M")
        thresh_60m = self.get_active_threshold("60M")

        # --- STEP 3: 60-Min Volatility (using provided DF if available) ---
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

        # --- STEP 4: CHECK BREAKOUT PROPAGATION ---
        if current_1m_vol > thresh_60m:
            return (
                False,
                "🟢 VOLATILITY EXPANSION: 1m Vol "
                f"({current_1m_vol:.2f}) > 60m Chop ({thresh_60m:.2f})",
            )

        # --- STEP 5: TIERED CHECKS (Using Scaled Thresholds) ---
        if current_1m_vol < thresh_1m:
            return True, f"🔴 1M MICRO CHOP: Vol {current_1m_vol:.2f} < {thresh_1m:.2f} (Mult: {self.gemini_multiplier}x)"

        if current_1m_vol < thresh_15m:
            return True, f"🟠 15M MID CHOP: Vol {current_1m_vol:.2f} < {thresh_15m:.2f} (Mult: {self.gemini_multiplier}x)"

        return False, "✅ MARKET ACTIVE"

    def check_target_feasibility(self, entry_price: float, side: str, tp_distance: float, df_1m: pd.DataFrame) -> Tuple[bool, str]:
        """
        Ensures TP target sits INSIDE the current range (Box) if volatility is low.
        Also uses the GEMINI SCALED threshold for regime detection.
        """
        if df_1m.empty or len(df_1m) < self.LOOKBACK:
             return True, "Insufficient Data"

        # =========================================================
        # 1. REGIME CHECK (The Safety Switch)
        # =========================================================
        box_high = df_1m["high"].iloc[-self.LOOKBACK:].max()
        box_low = df_1m["low"].iloc[-self.LOOKBACK:].min()
        current_vol = box_high - box_low

        # Use Scaled Threshold for "Trending" determination
        # If Multiplier is LOW (Breakout Mode), this threshold drops, making it EASIER to be "Trending"
        thresh_15m = self.get_active_threshold("15M")

        if current_vol > thresh_15m:
            return True, "Trend/Breakout Regime (Target Check Skipped)"

        # =========================================================
        # 2. MICRO-COMPRESSION CHECK (The "First 2 Bars" Fix)
        # =========================================================
        # Only active because we passed the Regime Check (we are NOT trending)
        micro_high = df_1m["high"].iloc[-2:].max()
        micro_low = df_1m["low"].iloc[-2:].min()
        micro_vol = micro_high - micro_low

        if micro_vol <= 5.0:
            # We are in tight compression in a non-trending market.
            # Rule: DO NOT assume a breakout. The TP MUST fit inside these 2 bars.
            if side.upper() == "LONG":
                target = entry_price + tp_distance
                if target > micro_high:
                    return False, f"⛔ MICRO COMPRESSION: Range {micro_vol:.2f}pts. TP {target} > High {micro_high}. Waiting for Expansion."
            elif side.upper() == "SHORT":
                target = entry_price - tp_distance
                if target < micro_low:
                    return False, f"⛔ MICRO COMPRESSION: Range {micro_vol:.2f}pts. TP {target} < Low {micro_low}. Waiting for Expansion."

        # =========================================================
        # 3. MACRO CHOP CHECK (The Standard 20-Bar Fix)
        # =========================================================
        # If in Chop (which we are), enforce the 20-bar Box
        if side.upper() == "LONG":
            target = entry_price + tp_distance
            if target > box_high:
                return False, f"⛔ CHOP BOUND: TP ({target:.2f}) extends past Range High ({box_high:.2f}). Unlikely to hit."

        elif side.upper() == "SHORT":
            target = entry_price - tp_distance
            if target < box_low:
                return False, f"⛔ CHOP BOUND: TP ({target:.2f}) extends past Range Low ({box_low:.2f}). Unlikely to hit."

        return True, "Target Feasible in Range"

    def should_recalibrate(self, last_calibration: float, interval_seconds: int = 14400) -> bool:
        """Helper to check if recalibration is due."""
        return time.time() - last_calibration > interval_seconds
