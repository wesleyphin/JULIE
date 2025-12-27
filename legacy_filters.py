"""
Legacy Filter System - December 17th, 2025 Logic

This module contains the simpler, more permissive filter logic from December 17th.
Used in conjunction with the upgraded filter system for dual-filter arbitration.

Decision Matrix:
- Both BLOCK → Trade is blocked
- Both ALLOW → Trade is allowed
- Legacy ALLOWS, Upgraded BLOCKS → FilterArbitrator analyzes and decides
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from datetime import datetime, timedelta


class LegacyTrendFilter:
    """
    December 17th Trend Filter - Simple 50/200 EMA logic only.
    No tiered impulse detection, no volume analysis.
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


class LegacyChopFilter:
    """
    December 17th Chop Filter - No volatility scalar adjustment.
    Uses fixed thresholds without ATR-based dynamic scaling.
    """

    def __init__(self, base_thresholds: Dict = None):
        # Default December 17th thresholds
        self.base_thresholds = base_thresholds or {
            'chop': 8.0,
            'breakout': 15.0
        }

    def check_range_state(self, current_range: float, dt: datetime) -> str:
        """Simple range state check without volatility scaling."""
        if current_range < self.base_thresholds['chop']:
            return 'CHOP'
        elif current_range > self.base_thresholds['breakout']:
            return 'BREAKOUT'
        return 'NORMAL'


class LegacyVolatilityFilter:
    """
    December 17th Volatility Filter - Fixed multipliers without dynamic scaling.
    No R:R guardrails, simpler adjustment logic.
    """

    def __init__(self):
        self.low_vol_mult = 1.5  # Fixed multiplier for low vol
        self.ultra_low_mult = 2.0  # Fixed multiplier for ultra-low vol

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


class LegacyNewsFilter:
    """
    December 17th News Filter - Fixed 5min/35min buffers for all events.
    No tiered event handling.
    """

    def __init__(self):
        self.pre_buffer = 5   # Fixed 5 minutes before
        self.duration = 35    # Fixed 35 minutes after

    def get_event_buffers(self, event_title: str) -> Tuple[int, int]:
        """All events get the same fixed buffer."""
        return self.pre_buffer, self.duration


class LegacyHTFFVGFilter:
    """
    December 17th HTF FVG Filter - Original pierced wall bypass logic.
    Allows trades if we've already pierced the FVG level.
    """

    def __init__(self):
        self.required_room_pct = 0.5  # Original 50% room requirement

    def check_signal_blocked(self, signal: str, current_price: float,
                            fvg_level: float, tp_dist: float) -> Tuple[bool, str]:
        """
        Original logic: Ignore FVGs we have already pierced (dist < 0).
        """
        if signal == 'LONG':
            dist = fvg_level - current_price  # Distance to resistance above
        else:
            dist = current_price - fvg_level  # Distance to support below

        # [Dec 17 Logic] Ignore if we are already inside/below the entry
        if dist < 0:
            return False, "Legacy: Already pierced FVG level"

        min_room = tp_dist * self.required_room_pct
        if dist < min_room:
            return True, f"Legacy: FVG blocks - dist {dist:.2f} < required {min_room:.2f}"

        return False, ""


class LegacyRejectionFilter:
    """
    December 17th Rejection Filter - No bias invalidation logic.
    Bias persists until naturally reversed, not invalidated by price closes.
    """

    def __init__(self):
        self.bias = None

    def check_bias_valid(self, current_bias: str, close: float,
                        level_high: float, level_low: float) -> bool:
        """
        December 17th: No invalidation check.
        Bias is always considered valid once set.
        """
        return True  # Never invalidate


class LegacyFilterSystem:
    """
    Unified interface for all December 17th legacy filters.
    """

    def __init__(self):
        self.trend_filter = LegacyTrendFilter()
        self.chop_filter = LegacyChopFilter()
        self.volatility_filter = LegacyVolatilityFilter()
        self.news_filter = LegacyNewsFilter()
        self.htf_fvg_filter = LegacyHTFFVGFilter()
        self.rejection_filter = LegacyRejectionFilter()

        logging.info("Legacy Filter System (Dec 17th) initialized")

    def check_trend(self, df: pd.DataFrame, side: str) -> Tuple[bool, str]:
        """Check legacy trend filter."""
        return self.trend_filter.should_block_trade(df, side)

    def get_news_buffers(self, event_title: str) -> Tuple[int, int]:
        """Get legacy news event buffers."""
        return self.news_filter.get_event_buffers(event_title)

    def check_htf_fvg(self, signal: str, current_price: float,
                     fvg_level: float, tp_dist: float) -> Tuple[bool, str]:
        """Check legacy HTF FVG filter."""
        return self.htf_fvg_filter.check_signal_blocked(signal, current_price, fvg_level, tp_dist)

    def check_bias_valid(self, current_bias: str, close: float,
                        level_high: float, level_low: float) -> bool:
        """Check if bias is valid (legacy always returns True)."""
        return self.rejection_filter.check_bias_valid(current_bias, close, level_high, level_low)

    def get_vol_adjustments(self, regime: str, base_sl: float, base_tp: float) -> Dict:
        """Get legacy volatility adjustments."""
        return self.volatility_filter.get_adjustments(regime, base_sl, base_tp)
