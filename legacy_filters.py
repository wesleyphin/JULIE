"""
Legacy Filter System - December 17th, 2025 8:11am California Time Logic
========================================================================

This module contains the EXACT filter logic from December 17th, 2025 at 8:11am California time.
Extracted from commit 10ea576 (Merge PR #52 - 7:26am California time).

Used in conjunction with the upgraded filter system for dual-filter arbitration.

Decision Matrix:
- Both BLOCK → Trade is blocked
- Both ALLOW → Trade is allowed
- Legacy ALLOWS, Upgraded BLOCKS → FilterArbitrator analyzes and decides
"""

import pandas as pd
import numpy as np
import logging
import datetime
import requests
from typing import Tuple, Optional, Dict
from datetime import timedelta
from datetime import timezone as dt_timezone
from collections import deque
from zoneinfo import ZoneInfo


# ============================================================
# LEGACY TREND FILTER (December 17th, 2025)
# ============================================================
class LegacyTrendFilter:
    """
    December 17th Trend Filter - Simple 50/200 EMA logic.
    Filters counter-trend trades when the trend is extremely strong.
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def should_block_trade(self, df: pd.DataFrame, side: str) -> Tuple[bool, str]:
        if len(df) < self.slow_period:
            return False, ""

        closes = df["close"]
        ema_fast = closes.ewm(span=self.fast_period, adjust=False).mean().iloc[-1]
        ema_slow = closes.ewm(span=self.slow_period, adjust=False).mean().iloc[-1]
        price = closes.iloc[-1]

        # Block Shorts in Strong Uptrend
        if price > ema_fast > ema_slow:
            if side == "SHORT":
                return True, f"Legacy Trend: Price > {self.fast_period}EMA > {self.slow_period}EMA (Strong Bullish)"

        # Block Longs in Strong Downtrend
        if price < ema_fast < ema_slow:
            if side == "LONG":
                return True, f"Legacy Trend: Price < {self.fast_period}EMA < {self.slow_period}EMA (Strong Bearish)"

        return False, ""


# ============================================================
# LEGACY CHOP FILTER (December 17th, 2025)
# 320 Hierarchical Thresholds
# ============================================================

# 320 Hierarchical Chop Thresholds from December 17th
# Key: YearlyQ_MonthlyQ_DayOfWeek_Session
LEGACY_CHOP_THRESHOLDS = {
    "Q1_W1_MON_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.25},
    "Q1_W1_MON_LONDON": {"chop": 3.50, "median": 5.25, "breakout": 7.75},
    "Q1_W1_MON_NY_AM": {"chop": 5.75, "median": 8.62, "breakout": 15.50},
    "Q1_W1_MON_NY_PM": {"chop": 5.75, "median": 9.00, "breakout": 13.81},
    "Q1_W1_TUE_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.00},
    "Q1_W1_TUE_LONDON": {"chop": 3.50, "median": 5.00, "breakout": 8.50},
    "Q1_W1_TUE_NY_AM": {"chop": 6.50, "median": 9.25, "breakout": 15.25},
    "Q1_W1_TUE_NY_PM": {"chop": 6.50, "median": 9.75, "breakout": 16.00},
    "Q1_W1_WED_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.25},
    "Q1_W1_WED_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.25},
    "Q1_W1_WED_NY_AM": {"chop": 6.25, "median": 9.50, "breakout": 15.75},
    "Q1_W1_WED_NY_PM": {"chop": 5.75, "median": 8.25, "breakout": 16.06},
    "Q1_W1_THU_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.50},
    "Q1_W1_THU_LONDON": {"chop": 4.00, "median": 5.50, "breakout": 8.25},
    "Q1_W1_THU_NY_AM": {"chop": 7.50, "median": 10.50, "breakout": 15.25},
    "Q1_W1_THU_NY_PM": {"chop": 6.75, "median": 10.00, "breakout": 17.00},
    "Q1_W1_FRI_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.50},
    "Q1_W1_FRI_LONDON": {"chop": 3.50, "median": 4.75, "breakout": 6.25},
    "Q1_W1_FRI_NY_AM": {"chop": 8.25, "median": 13.00, "breakout": 21.50},
    "Q1_W1_FRI_NY_PM": {"chop": 5.75, "median": 8.25, "breakout": 12.50},
    "Q1_W2_MON_ASIA": {"chop": 2.25, "median": 3.75, "breakout": 9.50},
    "Q1_W2_MON_LONDON": {"chop": 3.75, "median": 6.25, "breakout": 16.25},
    "Q1_W2_MON_NY_AM": {"chop": 6.00, "median": 10.25, "breakout": 19.25},
    "Q1_W2_MON_NY_PM": {"chop": 5.75, "median": 8.75, "breakout": 15.25},
    "Q1_W2_TUE_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.50},
    "Q1_W2_TUE_LONDON": {"chop": 4.00, "median": 5.25, "breakout": 8.06},
    "Q1_W2_TUE_NY_AM": {"chop": 7.75, "median": 12.50, "breakout": 21.50},
    "Q1_W2_TUE_NY_PM": {"chop": 6.75, "median": 10.00, "breakout": 16.00},
    "Q1_W2_WED_ASIA": {"chop": 2.00, "median": 2.50, "breakout": 4.00},
    "Q1_W2_WED_LONDON": {"chop": 3.50, "median": 4.50, "breakout": 6.50},
    "Q1_W2_WED_NY_AM": {"chop": 6.25, "median": 10.25, "breakout": 16.75},
    "Q1_W2_WED_NY_PM": {"chop": 5.75, "median": 8.50, "breakout": 13.00},
    "Q1_W2_THU_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.00},
    "Q1_W2_THU_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.25},
    "Q1_W2_THU_NY_AM": {"chop": 6.75, "median": 11.75, "breakout": 18.19},
    "Q1_W2_THU_NY_PM": {"chop": 6.25, "median": 9.50, "breakout": 14.25},
    "Q1_W2_FRI_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.25},
    "Q1_W2_FRI_LONDON": {"chop": 4.00, "median": 5.75, "breakout": 8.25},
    "Q1_W2_FRI_NY_AM": {"chop": 7.25, "median": 11.75, "breakout": 19.00},
    "Q1_W2_FRI_NY_PM": {"chop": 5.50, "median": 8.75, "breakout": 13.75},
    "Q1_W3_MON_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 5.25},
    "Q1_W3_MON_LONDON": {"chop": 2.75, "median": 4.25, "breakout": 7.25},
    "Q1_W3_MON_NY_AM": {"chop": 2.50, "median": 4.00, "breakout": 11.75},
    "Q1_W3_MON_NY_PM": {"chop": 3.50, "median": 8.00, "breakout": 13.25},
    "Q1_W3_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.25},
    "Q1_W3_TUE_LONDON": {"chop": 4.25, "median": 5.50, "breakout": 8.00},
    "Q1_W3_TUE_NY_AM": {"chop": 6.25, "median": 9.00, "breakout": 12.25},
    "Q1_W3_TUE_NY_PM": {"chop": 5.25, "median": 7.25, "breakout": 10.00},
    "Q1_W3_WED_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.50},
    "Q1_W3_WED_LONDON": {"chop": 3.50, "median": 4.50, "breakout": 6.50},
    "Q1_W3_WED_NY_AM": {"chop": 6.00, "median": 9.50, "breakout": 13.75},
    "Q1_W3_WED_NY_PM": {"chop": 5.75, "median": 8.75, "breakout": 13.50},
    "Q1_W3_THU_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.00},
    "Q1_W3_THU_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 7.00},
    "Q1_W3_THU_NY_AM": {"chop": 7.25, "median": 10.25, "breakout": 15.00},
    "Q1_W3_THU_NY_PM": {"chop": 6.25, "median": 8.75, "breakout": 12.25},
    "Q1_W3_FRI_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 3.50},
    "Q1_W3_FRI_LONDON": {"chop": 3.25, "median": 5.00, "breakout": 7.00},
    "Q1_W3_FRI_NY_AM": {"chop": 7.75, "median": 11.12, "breakout": 16.00},
    "Q1_W3_FRI_NY_PM": {"chop": 6.00, "median": 9.00, "breakout": 12.00},
    "Q1_W4_MON_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.50},
    "Q1_W4_MON_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 7.75},
    "Q1_W4_MON_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 13.00},
    "Q1_W4_MON_NY_PM": {"chop": 5.25, "median": 7.75, "breakout": 11.25},
    "Q1_W4_TUE_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 3.50},
    "Q1_W4_TUE_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.00},
    "Q1_W4_TUE_NY_AM": {"chop": 5.25, "median": 7.25, "breakout": 11.50},
    "Q1_W4_TUE_NY_PM": {"chop": 4.50, "median": 6.25, "breakout": 9.00},
    "Q1_W4_WED_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 3.75},
    "Q1_W4_WED_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 5.50},
    "Q1_W4_WED_NY_AM": {"chop": 5.00, "median": 7.25, "breakout": 10.50},
    "Q1_W4_WED_NY_PM": {"chop": 5.75, "median": 8.50, "breakout": 14.31},
    "Q1_W4_THU_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.50},
    "Q1_W4_THU_LONDON": {"chop": 3.50, "median": 4.50, "breakout": 6.00},
    "Q1_W4_THU_NY_AM": {"chop": 6.25, "median": 9.00, "breakout": 14.25},
    "Q1_W4_THU_NY_PM": {"chop": 5.25, "median": 8.00, "breakout": 12.25},
    "Q1_W4_FRI_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.00},
    "Q1_W4_FRI_LONDON": {"chop": 3.25, "median": 4.00, "breakout": 5.50},
    "Q1_W4_FRI_NY_AM": {"chop": 6.00, "median": 9.00, "breakout": 14.25},
    "Q1_W4_FRI_NY_PM": {"chop": 5.75, "median": 8.25, "breakout": 12.50},
    "Q2_W1_MON_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.75},
    "Q2_W1_MON_LONDON": {"chop": 2.50, "median": 4.25, "breakout": 7.00},
    "Q2_W1_MON_NY_AM": {"chop": 5.00, "median": 8.50, "breakout": 13.25},
    "Q2_W1_MON_NY_PM": {"chop": 5.00, "median": 7.00, "breakout": 10.50},
    "Q2_W1_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.25},
    "Q2_W1_TUE_LONDON": {"chop": 3.50, "median": 5.00, "breakout": 7.25},
    "Q2_W1_TUE_NY_AM": {"chop": 5.50, "median": 8.75, "breakout": 13.00},
    "Q2_W1_TUE_NY_PM": {"chop": 5.50, "median": 7.50, "breakout": 11.50},
    "Q2_W1_WED_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 5.00},
    "Q2_W1_WED_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.25},
    "Q2_W1_WED_NY_AM": {"chop": 6.25, "median": 9.00, "breakout": 12.25},
    "Q2_W1_WED_NY_PM": {"chop": 5.50, "median": 8.00, "breakout": 15.50},
    "Q2_W1_THU_ASIA": {"chop": 1.75, "median": 3.00, "breakout": 5.75},
    "Q2_W1_THU_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 7.31},
    "Q2_W1_THU_NY_AM": {"chop": 6.25, "median": 10.50, "breakout": 17.50},
    "Q2_W1_THU_NY_PM": {"chop": 6.25, "median": 10.25, "breakout": 16.75},
    "Q2_W1_FRI_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 5.75},
    "Q2_W1_FRI_LONDON": {"chop": 2.75, "median": 4.25, "breakout": 6.25},
    "Q2_W1_FRI_NY_AM": {"chop": 8.25, "median": 14.00, "breakout": 21.00},
    "Q2_W1_FRI_NY_PM": {"chop": 6.00, "median": 8.25, "breakout": 12.31},
    "Q2_W2_MON_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 5.75},
    "Q2_W2_MON_LONDON": {"chop": 3.00, "median": 4.75, "breakout": 9.75},
    "Q2_W2_MON_NY_AM": {"chop": 4.94, "median": 7.50, "breakout": 12.50},
    "Q2_W2_MON_NY_PM": {"chop": 4.25, "median": 6.50, "breakout": 10.50},
    "Q2_W2_TUE_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 5.75},
    "Q2_W2_TUE_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 7.56},
    "Q2_W2_TUE_NY_AM": {"chop": 5.50, "median": 8.25, "breakout": 14.75},
    "Q2_W2_TUE_NY_PM": {"chop": 4.75, "median": 7.00, "breakout": 10.75},
    "Q2_W2_WED_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 5.50},
    "Q2_W2_WED_LONDON": {"chop": 2.75, "median": 3.75, "breakout": 6.25},
    "Q2_W2_WED_NY_AM": {"chop": 6.75, "median": 10.75, "breakout": 19.00},
    "Q2_W2_WED_NY_PM": {"chop": 6.25, "median": 10.25, "breakout": 17.75},
    "Q2_W2_THU_ASIA": {"chop": 2.00, "median": 4.00, "breakout": 14.75},
    "Q2_W2_THU_LONDON": {"chop": 3.75, "median": 5.00, "breakout": 7.25},
    "Q2_W2_THU_NY_AM": {"chop": 7.25, "median": 10.25, "breakout": 16.25},
    "Q2_W2_THU_NY_PM": {"chop": 5.00, "median": 7.25, "breakout": 11.25},
    "Q2_W2_FRI_ASIA": {"chop": 2.00, "median": 3.75, "breakout": 17.25},
    "Q2_W2_FRI_LONDON": {"chop": 3.50, "median": 5.50, "breakout": 10.00},
    "Q2_W2_FRI_NY_AM": {"chop": 6.75, "median": 10.75, "breakout": 16.25},
    "Q2_W2_FRI_NY_PM": {"chop": 6.00, "median": 9.00, "breakout": 13.25},
    "Q2_W3_MON_ASIA": {"chop": 2.00, "median": 3.75, "breakout": 7.75},
    "Q2_W3_MON_LONDON": {"chop": 3.00, "median": 5.00, "breakout": 10.25},
    "Q2_W3_MON_NY_AM": {"chop": 4.75, "median": 8.25, "breakout": 14.25},
    "Q2_W3_MON_NY_PM": {"chop": 5.00, "median": 8.25, "breakout": 13.00},
    "Q2_W3_TUE_ASIA": {"chop": 1.75, "median": 3.50, "breakout": 7.25},
    "Q2_W3_TUE_LONDON": {"chop": 3.25, "median": 5.00, "breakout": 8.75},
    "Q2_W3_TUE_NY_AM": {"chop": 5.50, "median": 8.50, "breakout": 12.50},
    "Q2_W3_TUE_NY_PM": {"chop": 5.00, "median": 7.25, "breakout": 12.25},
    "Q2_W3_WED_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 6.50},
    "Q2_W3_WED_LONDON": {"chop": 2.75, "median": 5.00, "breakout": 8.00},
    "Q2_W3_WED_NY_AM": {"chop": 5.50, "median": 8.75, "breakout": 14.25},
    "Q2_W3_WED_NY_PM": {"chop": 5.25, "median": 8.75, "breakout": 16.00},
    "Q2_W3_THU_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 5.00},
    "Q2_W3_THU_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 7.00},
    "Q2_W3_THU_NY_AM": {"chop": 6.50, "median": 9.25, "breakout": 13.00},
    "Q2_W3_THU_NY_PM": {"chop": 6.25, "median": 9.75, "breakout": 14.50},
    "Q2_W3_FRI_ASIA": {"chop": 1.75, "median": 3.00, "breakout": 5.19},
    "Q2_W3_FRI_LONDON": {"chop": 3.50, "median": 5.00, "breakout": 7.00},
    "Q2_W3_FRI_NY_AM": {"chop": 5.75, "median": 9.00, "breakout": 13.75},
    "Q2_W3_FRI_NY_PM": {"chop": 5.50, "median": 7.50, "breakout": 10.00},
    "Q2_W4_MON_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.25},
    "Q2_W4_MON_LONDON": {"chop": 2.75, "median": 4.25, "breakout": 5.75},
    "Q2_W4_MON_NY_AM": {"chop": 4.00, "median": 6.50, "breakout": 10.75},
    "Q2_W4_MON_NY_PM": {"chop": 4.50, "median": 7.00, "breakout": 11.25},
    "Q2_W4_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.50},
    "Q2_W4_TUE_LONDON": {"chop": 3.50, "median": 5.00, "breakout": 7.25},
    "Q2_W4_TUE_NY_AM": {"chop": 5.50, "median": 8.25, "breakout": 12.00},
    "Q2_W4_TUE_NY_PM": {"chop": 5.50, "median": 7.75, "breakout": 10.50},
    "Q2_W4_WED_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.50},
    "Q2_W4_WED_LONDON": {"chop": 3.50, "median": 4.50, "breakout": 6.50},
    "Q2_W4_WED_NY_AM": {"chop": 5.00, "median": 7.75, "breakout": 11.75},
    "Q2_W4_WED_NY_PM": {"chop": 6.00, "median": 8.75, "breakout": 14.25},
    "Q2_W4_THU_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q2_W4_THU_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.50},
    "Q2_W4_THU_NY_AM": {"chop": 6.75, "median": 10.25, "breakout": 14.75},
    "Q2_W4_THU_NY_PM": {"chop": 5.50, "median": 8.25, "breakout": 12.00},
    "Q2_W4_FRI_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.50},
    "Q2_W4_FRI_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 6.50},
    "Q2_W4_FRI_NY_AM": {"chop": 7.25, "median": 10.50, "breakout": 15.25},
    "Q2_W4_FRI_NY_PM": {"chop": 5.50, "median": 7.50, "breakout": 12.50},
    "Q3_W1_MON_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.25},
    "Q3_W1_MON_LONDON": {"chop": 2.75, "median": 4.25, "breakout": 6.25},
    "Q3_W1_MON_NY_AM": {"chop": 3.25, "median": 6.00, "breakout": 10.50},
    "Q3_W1_MON_NY_PM": {"chop": 4.25, "median": 6.75, "breakout": 12.75},
    "Q3_W1_TUE_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 5.50},
    "Q3_W1_TUE_LONDON": {"chop": 3.25, "median": 5.00, "breakout": 7.25},
    "Q3_W1_TUE_NY_AM": {"chop": 5.50, "median": 9.25, "breakout": 14.00},
    "Q3_W1_TUE_NY_PM": {"chop": 4.75, "median": 8.00, "breakout": 11.75},
    "Q3_W1_WED_ASIA": {"chop": 2.00, "median": 3.50, "breakout": 5.50},
    "Q3_W1_WED_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 7.25},
    "Q3_W1_WED_NY_AM": {"chop": 5.50, "median": 8.50, "breakout": 12.75},
    "Q3_W1_WED_NY_PM": {"chop": 4.75, "median": 7.00, "breakout": 10.75},
    "Q3_W1_THU_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.25},
    "Q3_W1_THU_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 6.75},
    "Q3_W1_THU_NY_AM": {"chop": 5.50, "median": 8.75, "breakout": 13.25},
    "Q3_W1_THU_NY_PM": {"chop": 5.25, "median": 7.75, "breakout": 14.00},
    "Q3_W1_FRI_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q3_W1_FRI_LONDON": {"chop": 3.00, "median": 4.75, "breakout": 7.75},
    "Q3_W1_FRI_NY_AM": {"chop": 6.50, "median": 11.75, "breakout": 22.50},
    "Q3_W1_FRI_NY_PM": {"chop": 5.25, "median": 9.00, "breakout": 13.00},
    "Q3_W2_MON_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.00},
    "Q3_W2_MON_LONDON": {"chop": 3.19, "median": 4.50, "breakout": 6.50},
    "Q3_W2_MON_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 12.50},
    "Q3_W2_MON_NY_PM": {"chop": 4.50, "median": 6.50, "breakout": 9.75},
    "Q3_W2_TUE_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.50},
    "Q3_W2_TUE_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.50},
    "Q3_W2_TUE_NY_AM": {"chop": 5.50, "median": 8.00, "breakout": 12.00},
    "Q3_W2_TUE_NY_PM": {"chop": 4.50, "median": 5.75, "breakout": 8.75},
    "Q3_W2_WED_ASIA": {"chop": 2.00, "median": 2.50, "breakout": 3.75},
    "Q3_W2_WED_LONDON": {"chop": 2.75, "median": 3.75, "breakout": 5.00},
    "Q3_W2_WED_NY_AM": {"chop": 5.50, "median": 8.50, "breakout": 13.75},
    "Q3_W2_WED_NY_PM": {"chop": 5.25, "median": 7.00, "breakout": 10.25},
    "Q3_W2_THU_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.25},
    "Q3_W2_THU_LONDON": {"chop": 2.50, "median": 3.50, "breakout": 5.00},
    "Q3_W2_THU_NY_AM": {"chop": 6.50, "median": 10.50, "breakout": 15.50},
    "Q3_W2_THU_NY_PM": {"chop": 4.75, "median": 7.00, "breakout": 11.25},
    "Q3_W2_FRI_ASIA": {"chop": 2.25, "median": 3.50, "breakout": 6.25},
    "Q3_W2_FRI_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 6.25},
    "Q3_W2_FRI_NY_AM": {"chop": 5.50, "median": 8.50, "breakout": 12.50},
    "Q3_W2_FRI_NY_PM": {"chop": 5.00, "median": 7.25, "breakout": 10.25},
    "Q3_W3_MON_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 5.50},
    "Q3_W3_MON_LONDON": {"chop": 3.00, "median": 4.00, "breakout": 8.56},
    "Q3_W3_MON_NY_AM": {"chop": 4.75, "median": 7.25, "breakout": 12.00},
    "Q3_W3_MON_NY_PM": {"chop": 4.00, "median": 6.00, "breakout": 9.50},
    "Q3_W3_TUE_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.50},
    "Q3_W3_TUE_LONDON": {"chop": 2.75, "median": 4.00, "breakout": 6.25},
    "Q3_W3_TUE_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 11.50},
    "Q3_W3_TUE_NY_PM": {"chop": 4.75, "median": 6.75, "breakout": 9.50},
    "Q3_W3_WED_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.50},
    "Q3_W3_WED_LONDON": {"chop": 3.25, "median": 4.00, "breakout": 5.75},
    "Q3_W3_WED_NY_AM": {"chop": 4.75, "median": 7.25, "breakout": 11.25},
    "Q3_W3_WED_NY_PM": {"chop": 5.75, "median": 8.75, "breakout": 13.25},
    "Q3_W3_THU_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 4.50},
    "Q3_W3_THU_LONDON": {"chop": 3.50, "median": 4.75, "breakout": 6.50},
    "Q3_W3_THU_NY_AM": {"chop": 6.50, "median": 9.25, "breakout": 13.25},
    "Q3_W3_THU_NY_PM": {"chop": 5.25, "median": 7.50, "breakout": 10.75},
    "Q3_W3_FRI_ASIA": {"chop": 2.00, "median": 2.50, "breakout": 3.50},
    "Q3_W3_FRI_LONDON": {"chop": 3.00, "median": 4.00, "breakout": 5.75},
    "Q3_W3_FRI_NY_AM": {"chop": 6.25, "median": 8.75, "breakout": 12.50},
    "Q3_W3_FRI_NY_PM": {"chop": 5.19, "median": 7.00, "breakout": 9.50},
    "Q3_W4_MON_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.00},
    "Q3_W4_MON_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 5.75},
    "Q3_W4_MON_NY_AM": {"chop": 5.00, "median": 7.50, "breakout": 10.75},
    "Q3_W4_MON_NY_PM": {"chop": 4.50, "median": 6.25, "breakout": 8.75},
    "Q3_W4_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 3.75},
    "Q3_W4_TUE_LONDON": {"chop": 3.25, "median": 4.00, "breakout": 6.00},
    "Q3_W4_TUE_NY_AM": {"chop": 4.94, "median": 7.50, "breakout": 11.00},
    "Q3_W4_TUE_NY_PM": {"chop": 5.00, "median": 7.00, "breakout": 9.75},
    "Q3_W4_WED_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.00},
    "Q3_W4_WED_LONDON": {"chop": 2.75, "median": 3.75, "breakout": 5.25},
    "Q3_W4_WED_NY_AM": {"chop": 5.00, "median": 7.25, "breakout": 10.00},
    "Q3_W4_WED_NY_PM": {"chop": 5.25, "median": 8.00, "breakout": 12.50},
    "Q3_W4_THU_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.25},
    "Q3_W4_THU_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.50},
    "Q3_W4_THU_NY_AM": {"chop": 6.50, "median": 9.75, "breakout": 14.50},
    "Q3_W4_THU_NY_PM": {"chop": 5.75, "median": 8.75, "breakout": 13.75},
    "Q3_W4_FRI_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q3_W4_FRI_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 5.75},
    "Q3_W4_FRI_NY_AM": {"chop": 5.50, "median": 8.75, "breakout": 14.50},
    "Q3_W4_FRI_NY_PM": {"chop": 5.50, "median": 8.25, "breakout": 11.75},
    "Q4_W1_MON_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q4_W1_MON_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 6.00},
    "Q4_W1_MON_NY_AM": {"chop": 5.50, "median": 7.75, "breakout": 11.50},
    "Q4_W1_MON_NY_PM": {"chop": 4.50, "median": 6.25, "breakout": 9.25},
    "Q4_W1_TUE_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 6.25},
    "Q4_W1_TUE_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 6.00},
    "Q4_W1_TUE_NY_AM": {"chop": 6.25, "median": 9.00, "breakout": 13.00},
    "Q4_W1_TUE_NY_PM": {"chop": 5.00, "median": 8.00, "breakout": 10.75},
    "Q4_W1_WED_ASIA": {"chop": 2.25, "median": 3.50, "breakout": 5.50},
    "Q4_W1_WED_LONDON": {"chop": 3.75, "median": 5.75, "breakout": 8.50},
    "Q4_W1_WED_NY_AM": {"chop": 6.50, "median": 10.50, "breakout": 14.75},
    "Q4_W1_WED_NY_PM": {"chop": 5.00, "median": 7.00, "breakout": 9.50},
    "Q4_W1_THU_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q4_W1_THU_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.00},
    "Q4_W1_THU_NY_AM": {"chop": 5.25, "median": 8.25, "breakout": 12.50},
    "Q4_W1_THU_NY_PM": {"chop": 4.75, "median": 6.50, "breakout": 10.00},
    "Q4_W1_FRI_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 4.00},
    "Q4_W1_FRI_LONDON": {"chop": 2.75, "median": 4.00, "breakout": 5.50},
    "Q4_W1_FRI_NY_AM": {"chop": 6.75, "median": 11.75, "breakout": 16.75},
    "Q4_W1_FRI_NY_PM": {"chop": 5.25, "median": 7.75, "breakout": 12.00},
    "Q4_W2_MON_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 5.50},
    "Q4_W2_MON_LONDON": {"chop": 3.19, "median": 4.25, "breakout": 8.50},
    "Q4_W2_MON_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 11.50},
    "Q4_W2_MON_NY_PM": {"chop": 4.00, "median": 6.00, "breakout": 8.50},
    "Q4_W2_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 5.50},
    "Q4_W2_TUE_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 7.25},
    "Q4_W2_TUE_NY_AM": {"chop": 5.25, "median": 7.50, "breakout": 11.25},
    "Q4_W2_TUE_NY_PM": {"chop": 4.75, "median": 6.50, "breakout": 9.50},
    "Q4_W2_WED_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 3.75},
    "Q4_W2_WED_LONDON": {"chop": 2.50, "median": 3.50, "breakout": 5.00},
    "Q4_W2_WED_NY_AM": {"chop": 4.75, "median": 6.75, "breakout": 9.75},
    "Q4_W2_WED_NY_PM": {"chop": 4.00, "median": 6.00, "breakout": 9.75},
    "Q4_W2_THU_ASIA": {"chop": 2.00, "median": 2.50, "breakout": 3.75},
    "Q4_W2_THU_LONDON": {"chop": 2.75, "median": 3.50, "breakout": 4.75},
    "Q4_W2_THU_NY_AM": {"chop": 5.75, "median": 8.75, "breakout": 12.25},
    "Q4_W2_THU_NY_PM": {"chop": 5.50, "median": 8.00, "breakout": 11.25},
    "Q4_W2_FRI_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.50},
    "Q4_W2_FRI_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 6.50},
    "Q4_W2_FRI_NY_AM": {"chop": 5.50, "median": 9.50, "breakout": 13.50},
    "Q4_W2_FRI_NY_PM": {"chop": 5.00, "median": 7.00, "breakout": 10.81},
    "Q4_W3_MON_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.75},
    "Q4_W3_MON_LONDON": {"chop": 3.75, "median": 5.25, "breakout": 7.75},
    "Q4_W3_MON_NY_AM": {"chop": 5.00, "median": 7.50, "breakout": 11.75},
    "Q4_W3_MON_NY_PM": {"chop": 4.00, "median": 5.75, "breakout": 8.50},
    "Q4_W3_TUE_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.25},
    "Q4_W3_TUE_LONDON": {"chop": 3.00, "median": 4.00, "breakout": 5.75},
    "Q4_W3_TUE_NY_AM": {"chop": 4.75, "median": 7.50, "breakout": 11.50},
    "Q4_W3_TUE_NY_PM": {"chop": 4.50, "median": 6.25, "breakout": 8.75},
    "Q4_W3_WED_ASIA": {"chop": 2.25, "median": 3.75, "breakout": 6.00},
    "Q4_W3_WED_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.00},
    "Q4_W3_WED_NY_AM": {"chop": 5.00, "median": 8.25, "breakout": 12.00},
    "Q4_W3_WED_NY_PM": {"chop": 5.50, "median": 9.00, "breakout": 16.00},
    "Q4_W3_THU_ASIA": {"chop": 2.25, "median": 3.50, "breakout": 6.00},
    "Q4_W3_THU_LONDON": {"chop": 3.50, "median": 5.25, "breakout": 8.00},
    "Q4_W3_THU_NY_AM": {"chop": 8.00, "median": 11.00, "breakout": 17.75},
    "Q4_W3_THU_NY_PM": {"chop": 6.25, "median": 9.50, "breakout": 16.75},
    "Q4_W3_FRI_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 5.25},
    "Q4_W3_FRI_LONDON": {"chop": 3.00, "median": 4.75, "breakout": 8.81},
    "Q4_W3_FRI_NY_AM": {"chop": 6.25, "median": 10.00, "breakout": 16.75},
    "Q4_W3_FRI_NY_PM": {"chop": 5.00, "median": 8.75, "breakout": 13.25},
    "Q4_W4_MON_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.25},
    "Q4_W4_MON_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.25},
    "Q4_W4_MON_NY_AM": {"chop": 6.00, "median": 9.25, "breakout": 14.50},
    "Q4_W4_MON_NY_PM": {"chop": 4.50, "median": 7.50, "breakout": 10.50},
    "Q4_W4_TUE_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 3.75},
    "Q4_W4_TUE_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 5.75},
    "Q4_W4_TUE_NY_AM": {"chop": 5.25, "median": 7.25, "breakout": 11.00},
    "Q4_W4_TUE_NY_PM": {"chop": 4.50, "median": 6.75, "breakout": 10.25},
    "Q4_W4_WED_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 4.25},
    "Q4_W4_WED_LONDON": {"chop": 2.75, "median": 4.00, "breakout": 5.50},
    "Q4_W4_WED_NY_AM": {"chop": 5.00, "median": 7.50, "breakout": 11.00},
    "Q4_W4_WED_NY_PM": {"chop": 4.75, "median": 7.75, "breakout": 12.00},
    "Q4_W4_THU_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 4.00},
    "Q4_W4_THU_LONDON": {"chop": 2.25, "median": 4.25, "breakout": 6.50},
    "Q4_W4_THU_NY_AM": {"chop": 3.75, "median": 8.00, "breakout": 13.00},
    "Q4_W4_THU_NY_PM": {"chop": 5.25, "median": 8.25, "breakout": 12.00},
    "Q4_W4_FRI_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 3.50},
    "Q4_W4_FRI_LONDON": {"chop": 2.50, "median": 3.75, "breakout": 5.25},
    "Q4_W4_FRI_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 13.75},
    "Q4_W4_FRI_NY_PM": {"chop": 5.25, "median": 8.25, "breakout": 12.75},
}

# Day of week mapping
LEGACY_DOW_NAMES = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']


class LegacyChopFilter:
    """
    December 17th Chop Filter - Dynamic consolidation detection with structure validation.
    320 hierarchical thresholds based on 2023-2025 MES futures data.
    """

    def __init__(self, lookback: int = 20, swing_lookback: int = 5,
                 max_bars_in_chop: int = 20):
        self.lookback = lookback
        self.swing_lookback = swing_lookback
        self.max_bars_in_chop = max_bars_in_chop

        # Price tracking
        self.highs = deque(maxlen=lookback)
        self.lows = deque(maxlen=lookback)
        self.closes = deque(maxlen=lookback)

        # Chop range tracking
        self.chop_high = None
        self.chop_low = None

        # Breakout level
        self.breakout_level = None
        self.breakout_direction = None

        # Swing tracking
        self.swing_highs = []
        self.swing_lows = []
        self.bar_count = 0

        # Time decay tracking
        self.bars_in_chop = 0

        # Volatility Scalar baseline
        self.baseline_atr = 4.50

        # State
        self.state = 'NORMAL'
        self.current_threshold = None

    def _get_session(self, dt: datetime.datetime) -> str:
        """Determine trading session from datetime."""
        hour = dt.hour
        if 18 <= hour <= 23 or 0 <= hour < 3:
            return 'ASIA'
        elif 3 <= hour < 8:
            return 'LONDON'
        elif 8 <= hour < 12:
            return 'NY_AM'
        elif 12 <= hour < 17:
            return 'NY_PM'
        return 'CLOSED'

    def _get_yearly_quarter(self, month: int) -> str:
        """Get yearly quarter from month."""
        if month in [1, 2, 3]:
            return 'Q1'
        elif month in [4, 5, 6]:
            return 'Q2'
        elif month in [7, 8, 9]:
            return 'Q3'
        return 'Q4'

    def _get_monthly_quarter(self, day: int) -> str:
        """Get monthly week from day."""
        if day <= 7:
            return 'W1'
        elif day <= 14:
            return 'W2'
        elif day <= 21:
            return 'W3'
        return 'W4'

    def _get_threshold_key(self, dt: datetime.datetime) -> str:
        """Build threshold key: YearlyQ_MonthlyQ_DayOfWeek_Session"""
        yearly_q = self._get_yearly_quarter(dt.month)
        monthly_q = self._get_monthly_quarter(dt.day)
        dow = LEGACY_DOW_NAMES[dt.weekday()]
        session = self._get_session(dt)
        return f"{yearly_q}_{monthly_q}_{dow}_{session}"

    def _get_thresholds(self, dt: datetime.datetime) -> dict:
        """Get chop thresholds for current time context."""
        key = self._get_threshold_key(dt)

        if key in LEGACY_CHOP_THRESHOLDS:
            return LEGACY_CHOP_THRESHOLDS[key]

        # Fallback defaults
        session = self._get_session(dt)
        fallback_defaults = {
            'ASIA': {"chop": 2.00, "median": 3.00, "breakout": 5.00},
            'LONDON': {"chop": 3.25, "median": 4.50, "breakout": 6.50},
            'NY_AM': {"chop": 5.50, "median": 8.50, "breakout": 13.00},
            'NY_PM': {"chop": 5.00, "median": 7.50, "breakout": 11.00},
        }
        return fallback_defaults.get(session, {"chop": 4.00, "median": 6.00, "breakout": 10.00})

    def check_range_state(self, current_range: float, dt: datetime.datetime) -> str:
        """Check range state using hierarchical thresholds."""
        thresholds = self._get_thresholds(dt)
        if current_range < thresholds['chop']:
            return 'CHOP'
        elif current_range > thresholds['breakout']:
            return 'BREAKOUT'
        return 'NORMAL'


# ============================================================
# LEGACY NEWS FILTER (December 17th, 2025)
# Tiered Event Handling
# ============================================================
class LegacyNewsFilter:
    """
    December 17th News Filter - Tiered event handling.
    TIER 1 (CPI/FOMC/NFP): 10min pre-buffer, 60min duration
    TIER 2 (GDP/PPI): 5min pre-buffer, 30min duration
    DEFAULT: 3min pre-buffer, 15min duration
    """

    def __init__(self):
        self.et = ZoneInfo("America/New_York")
        self.ff_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

        # Daily Recurrent Blackouts
        self.daily_blackouts = [
            (16, 55, 70),  # CME Close
        ]

        # Event Containers
        self.calendar_blackouts = []
        self.recent_events = []

        # Tiered Keywords (December 17th)
        self.TIER_1_KEYWORDS = ['CPI', 'Non-Farm', 'FOMC', 'Rate Decision', 'Powell', 'NFP', 'Nonfarm']
        self.TIER_1_DURATION = 60
        self.TIER_1_BUFFER = 10

        self.TIER_2_KEYWORDS = ['GDP', 'PPI', 'Retail Sales', 'Unemployment Claims', 'ISM', 'PMI']
        self.TIER_2_DURATION = 30
        self.TIER_2_BUFFER = 5

        self.DEFAULT_DURATION = 15
        self.DEFAULT_BUFFER = 3

        # Load calendar
        self.refresh_calendar()

    def refresh_calendar(self):
        """Fetches 'Red Folder' (High Impact) USD news from ForexFactory."""
        try:
            logging.info("📅 Legacy NewsFilter: Fetching news calendar from ForexFactory...")
            resp = requests.get(self.ff_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            count = 0
            current_time = datetime.datetime.now(self.et)

            self.calendar_blackouts = []
            self.recent_events = []

            for event in data:
                if event.get('country') == 'USD' and event.get('impact') == 'High':
                    event_dt_str = event.get('date')
                    try:
                        event_dt = datetime.datetime.fromisoformat(event_dt_str)
                        event_dt_et = event_dt.astimezone(self.et)

                        title = event.get('title', '')

                        # Determine Tier
                        duration = self.DEFAULT_DURATION
                        pre_buffer = self.DEFAULT_BUFFER
                        tier = 3

                        if any(k.lower() in title.lower() for k in self.TIER_1_KEYWORDS):
                            duration = self.TIER_1_DURATION
                            pre_buffer = self.TIER_1_BUFFER
                            tier = 1
                        elif any(k.lower() in title.lower() for k in self.TIER_2_KEYWORDS):
                            duration = self.TIER_2_DURATION
                            pre_buffer = self.TIER_2_BUFFER
                            tier = 2

                        event_obj = {
                            'title': title,
                            'time': event_dt_et,
                            'date_str': event_dt_et.strftime('%Y-%m-%d %H:%M'),
                            'impact': 'High',
                            'tier': tier
                        }

                        if event_dt_et.month == current_time.month:
                            self.recent_events.append(event_obj)

                        if event_dt_et.date() >= current_time.date():
                            self.calendar_blackouts.append({
                                'time': event_dt_et,
                                'title': title,
                                'duration': duration,
                                'pre_buffer': pre_buffer,
                                'tier': tier
                            })
                            count += 1

                    except Exception as parse_err:
                        logging.warning(f"Failed to parse event date: {event_dt_str} - {parse_err}")

            logging.info(f"✅ Legacy NewsFilter: Calendar updated with {count} events.")

        except Exception as e:
            logging.error(f"❌ Legacy NewsFilter: Failed to fetch news calendar: {e}")

    def get_event_buffers(self, event_title: str) -> Tuple[int, int]:
        """Get tiered event buffers based on event title."""
        if any(k.lower() in event_title.lower() for k in self.TIER_1_KEYWORDS):
            return self.TIER_1_BUFFER, self.TIER_1_DURATION
        elif any(k.lower() in event_title.lower() for k in self.TIER_2_KEYWORDS):
            return self.TIER_2_BUFFER, self.TIER_2_DURATION
        return self.DEFAULT_BUFFER, self.DEFAULT_DURATION

    def should_block_trade(self, current_time: datetime.datetime) -> Tuple[bool, str]:
        """Check if trade should be blocked due to news."""
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=dt_timezone.utc).astimezone(self.et)
        else:
            current_time = current_time.astimezone(self.et)

        # Check Daily Recurring Blackouts
        for hour, minute, duration in self.daily_blackouts:
            start = current_time.replace(hour=hour, minute=minute, second=0, microsecond=0)
            end = start + timedelta(minutes=duration)
            if start <= current_time <= end:
                return True, f"Legacy Daily Blackout: {start.strftime('%H:%M')} - {end.strftime('%H:%M')}"

        # Check Dynamic Calendar Events
        for event in self.calendar_blackouts:
            event_time = event['time']
            block_start = event_time - timedelta(minutes=event['pre_buffer'])
            block_end = event_time + timedelta(minutes=event['duration'])

            if block_start <= current_time <= block_end:
                return True, f"Legacy NEWS: {event['title']} ({event_time.strftime('%H:%M')})"

        return False, ""


# ============================================================
# LEGACY HTF FVG FILTER (December 17th, 2025)
# Memory-based with pierced wall logic
# ============================================================
class LegacyHTFFVGFilter:
    """
    December 17th HTF FVG Filter - Stateful memory system with pierced wall bypass.
    40% room requirement (relaxed from 50%).
    """

    def __init__(self, expiration_bars=141):
        self.memory = []
        self.expiration_bars = expiration_bars
        self.required_room_pct = 0.40  # December 17th: 40% room requirement

    def _normalize_cols(self, df):
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _scan_for_new_fvgs(self, df, timeframe_label):
        """Scan dataframe for valid FVGs."""
        if df is None or len(df) < 3:
            return []

        df = self._normalize_cols(df)
        highs = df['high'].values
        lows = df['low'].values
        times = df.index

        found_fvgs = []

        for i in range(len(df) - 2):
            c1_h = highs[i]
            c1_l = lows[i]
            c3_h = highs[i + 2]
            c3_l = lows[i + 2]
            timestamp = times[i + 2]

            fvg = None

            # BULLISH FVG
            if c3_l > c1_h:
                fvg = {
                    'id': f"{timeframe_label}_{timestamp}",
                    'type': 'bullish',
                    'tf': timeframe_label,
                    'top': c3_l,
                    'bottom': c1_h,
                    'created_at': timestamp,
                    'bar_index': i + 2
                }

            # BEARISH FVG
            elif c3_h < c1_l:
                fvg = {
                    'id': f"{timeframe_label}_{timestamp}",
                    'type': 'bearish',
                    'tf': timeframe_label,
                    'top': c1_l,
                    'bottom': c3_h,
                    'created_at': timestamp,
                    'bar_index': i + 2
                }

            if fvg:
                # Check if invalidated later
                is_broken = False
                for j in range(fvg['bar_index'] + 1, len(df)):
                    if fvg['type'] == 'bullish' and lows[j] < fvg['bottom']:
                        is_broken = True
                        break
                    elif fvg['type'] == 'bearish' and highs[j] > fvg['top']:
                        is_broken = True
                        break

                if not is_broken:
                    found_fvgs.append(fvg)

        return found_fvgs

    def _update_memory(self, new_fvgs):
        """Merge new scans into memory."""
        existing_ids = {f['id'] for f in self.memory}

        for f in new_fvgs:
            if f['id'] not in existing_ids:
                self.memory.append(f)

    def _clean_memory(self, current_price, current_time=None):
        """Remove broken or expired FVGs."""
        valid_fvgs = []

        for f in self.memory:
            # Price Invalidation
            if f['type'] == 'bullish' and current_price < f['bottom']:
                continue
            elif f['type'] == 'bearish' and current_price > f['top']:
                continue

            # Time Expiration
            if current_time and f.get('created_at'):
                age = current_time - f['created_at']
                if f['tf'] == '1H':
                    max_age = timedelta(hours=self.expiration_bars)
                else:
                    max_age = timedelta(hours=self.expiration_bars * 4)

                if age > max_age:
                    continue

            valid_fvgs.append(f)

        self.memory = valid_fvgs

    def check_signal_blocked(self, signal: str, current_price: float,
                             df_1h=None, df_4h=None, tp_dist: float = None) -> Tuple[bool, str]:
        """
        December 17th logic: Ignore FVGs we have already pierced (dist < 0).
        40% room requirement.
        """
        # Refresh Memory
        if df_1h is not None and not df_1h.empty:
            fvgs_1h = self._scan_for_new_fvgs(df_1h, '1H')
            self._update_memory(fvgs_1h)

        if df_4h is not None and not df_4h.empty:
            fvgs_4h = self._scan_for_new_fvgs(df_4h, '4H')
            self._update_memory(fvgs_4h)

        # Clean Memory
        current_time = datetime.datetime.now(df_1h.index.tz) if (df_1h is not None and not df_1h.empty) else datetime.datetime.now().astimezone()
        self._clean_memory(current_price, current_time)

        if not self.memory:
            return False, ""

        signal = signal.upper()
        min_room_needed = (tp_dist * self.required_room_pct) if tp_dist else 10.0

        if signal in ['BUY', 'LONG']:
            for f in self.memory:
                if f['type'] == 'bearish':
                    if current_price < f['top']:
                        dist = f['bottom'] - current_price

                        # [Dec 17 Logic] Ignore if already pierced
                        if dist < 0:
                            continue

                        if dist < min_room_needed:
                            return True, f"Legacy: Bearish {f['tf']} FVG overhead @ {f['bottom']:.2f} (Dist: {dist:.2f} < {min_room_needed:.2f})"

        elif signal in ['SELL', 'SHORT']:
            for f in self.memory:
                if f['type'] == 'bullish':
                    if current_price > f['bottom']:
                        dist = current_price - f['top']

                        # [Dec 17 Logic] Ignore if already pierced
                        if dist < 0:
                            continue

                        if dist < min_room_needed:
                            return True, f"Legacy: Bullish {f['tf']} FVG support @ {f['top']:.2f} (Dist: {dist:.2f} < {min_room_needed:.2f})"

        return False, ""


# ============================================================
# LEGACY REJECTION FILTER (December 17th, 2025)
# Full bias tracking with continuation logic
# ============================================================
class LegacyRejectionFilter:
    """
    December 17th Rejection Filter - Full bias tracking with continuation logic.
    1 candle CLOSE required to establish or flip bias.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_day_pm_high: Optional[float] = None
        self.prev_day_pm_low: Optional[float] = None
        self.current_date: Optional[datetime.date] = None
        self.current_pm_high: Optional[float] = None
        self.current_pm_low: Optional[float] = None

        self.prev_session_high: Optional[float] = None
        self.prev_session_low: Optional[float] = None
        self.curr_session_high: Optional[float] = None
        self.curr_session_low: Optional[float] = None
        self.last_session: Optional[str] = None

        self.midnight_orb_high: Optional[float] = None
        self.midnight_orb_low: Optional[float] = None
        self.midnight_orb_set: bool = False

        self.prev_day_pm_bias: Optional[str] = None
        self.prev_session_bias: Optional[str] = None
        self.midnight_orb_bias: Optional[str] = None

        self.current_quarter: int = 0
        self.current_session_name: Optional[str] = None

        self.last_rejection_level: Optional[str] = None
        self.last_rejection_source: Optional[str] = None

    def get_session(self, hour: int) -> str:
        """Determine session from hour (ET)."""
        if 18 <= hour <= 23 or 0 <= hour < 3:
            return 'ASIA'
        elif 3 <= hour < 8:
            return 'LONDON'
        elif 8 <= hour < 12:
            return 'NY_AM'
        elif 12 <= hour < 17:
            return 'NY_PM'
        return 'CLOSED'

    def check_rejection(self, high: float, low: float, close: float,
                        level_high: Optional[float], level_low: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
        """Check if candle CLOSED showing rejection of a level."""
        if level_high is None or level_low is None:
            return None, None

        # Bullish rejection: swept low, CLOSED back above
        if low < level_low and close > level_low:
            return 'LONG', 'LOW'

        # Bearish rejection: swept high, CLOSED back below
        if high > level_high and close < level_high:
            return 'SHORT', 'HIGH'

        return None, None

    def check_bias_valid(self, current_bias: str, close: float,
                         level_high: float, level_low: float) -> bool:
        """
        December 17th: Bias validation with continuation logic.
        Bias remains valid until opposite rejection occurs.
        """
        return True  # December 17th didn't have invalidation, just flips on new rejection

    def should_block_trade(self, direction: str) -> Tuple[bool, str]:
        """Check if trade should be blocked based on rejection biases."""
        blocks = []
        details = []

        def block_if_opposite(bias, label):
            if bias == 'LONG' and direction == 'SELL':
                blocks.append(True)
                details.append(f"{label} bias LONG blocks SELL")
            elif bias == 'SHORT' and direction == 'BUY':
                blocks.append(True)
                details.append(f"{label} bias SHORT blocks BUY")

        block_if_opposite(self.prev_day_pm_bias, 'Legacy Prev Day PM')
        block_if_opposite(self.prev_session_bias, 'Legacy Prev Session')
        block_if_opposite(self.midnight_orb_bias, 'Legacy Midnight ORB')

        if any(blocks):
            reason = "; ".join(details)
            return True, reason

        return False, ""


# ============================================================
# LEGACY VOLATILITY FILTER (December 17th, 2025)
# Fixed multipliers for SL/TP adjustments
# ============================================================
class LegacyVolatilityFilter:
    """
    December 17th Volatility Filter - Fixed multipliers without dynamic scaling.
    Simple adjustment logic based on volatility regime.
    """

    def __init__(self):
        self.low_vol_mult = 1.5
        self.ultra_low_mult = 2.0

    def get_adjustments(self, regime: str, base_sl: float, base_tp: float) -> Dict:
        """Get simple fixed adjustments based on regime."""
        if regime == 'ULTRA_LOW':
            return {
                'sl': base_sl * self.ultra_low_mult,
                'tp': base_tp * self.ultra_low_mult,
                'adjusted': True
            }
        elif regime == 'LOW':
            return {
                'sl': base_sl * self.low_vol_mult,
                'tp': base_tp * self.low_vol_mult,
                'adjusted': True
            }
        return {
            'sl': base_sl,
            'tp': base_tp,
            'adjusted': False
        }


# ============================================================
# LEGACY FILTER SYSTEM (December 17th, 2025)
# Unified interface for all legacy filters
# ============================================================
class LegacyFilterSystem:
    """
    Unified interface for all December 17th legacy filters.
    Exact reproduction of the filter logic from December 17th, 2025 at 8:11am California time.
    """

    def __init__(self):
        self.trend_filter = LegacyTrendFilter()
        self.chop_filter = LegacyChopFilter()
        self.volatility_filter = LegacyVolatilityFilter()
        self.news_filter = LegacyNewsFilter()
        self.htf_fvg_filter = LegacyHTFFVGFilter()
        self.rejection_filter = LegacyRejectionFilter()

        logging.info("Legacy Filter System (Dec 17th 8:11am) initialized")

    def check_trend(self, df: pd.DataFrame, side: str) -> Tuple[bool, str]:
        """Check legacy trend filter."""
        return self.trend_filter.should_block_trade(df, side)

    def get_news_buffers(self, event_title: str) -> Tuple[int, int]:
        """Get legacy news event buffers (tiered)."""
        return self.news_filter.get_event_buffers(event_title)

    def check_news(self, current_time: datetime.datetime) -> Tuple[bool, str]:
        """Check legacy news filter."""
        return self.news_filter.should_block_trade(current_time)

    def check_htf_fvg(self, signal: str, current_price: float,
                      df_1h=None, df_4h=None, tp_dist: float = None) -> Tuple[bool, str]:
        """Check legacy HTF FVG filter."""
        return self.htf_fvg_filter.check_signal_blocked(signal, current_price, df_1h, df_4h, tp_dist)

    def check_rejection(self, direction: str) -> Tuple[bool, str]:
        """Check legacy rejection filter."""
        return self.rejection_filter.should_block_trade(direction)

    def check_bias_valid(self, current_bias: str, close: float,
                         level_high: float, level_low: float) -> bool:
        """Check if bias is valid (legacy always returns True)."""
        return self.rejection_filter.check_bias_valid(current_bias, close, level_high, level_low)

    def get_vol_adjustments(self, regime: str, base_sl: float, base_tp: float) -> Dict:
        """Get legacy volatility adjustments."""
        return self.volatility_filter.get_adjustments(regime, base_sl, base_tp)

    def check_chop_state(self, current_range: float, dt: datetime.datetime) -> str:
        """Check legacy chop state using 320 hierarchical thresholds."""
        return self.chop_filter.check_range_state(current_range, dt)
