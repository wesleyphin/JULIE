import datetime
import logging
from typing import Dict, Optional

import pandas as pd
from zoneinfo import ZoneInfo

from dynamic_sltp_params import dynamic_sltp_engine
from strategy_base import Strategy


class ConfluenceStrategy(Strategy):
    def __init__(self):
        self.et = ZoneInfo('America/New_York')
        # Session tracking
        self.current_session = None
        self.prev_session_high = None
        self.prev_session_low = None
        self.session_high = None
        self.session_low = None
        self.bear_swept_this_session = False
        self.bull_swept_this_session = False
        # Hourly FVG tracking
        self.last_hour_processed = None
        self.bull_fvg_low = None
        self.bull_fvg_high = None
        self.bear_fvg_low = None
        self.bear_fvg_high = None
        # Fixed TP/SL based on backtest optimization
        self.TAKE_PROFIT = 5.0  # Backtest showed 64.5% WR holds for all TP values up to 5pt
        self.STOP_LOSS = 2.0

    def _get_session_date(self, ts) -> datetime.date:
        """Assign session date: if before 18:00, belongs to previous day's session"""
        hour = ts.hour
        dte = ts.date()
        if hour >= 18:
            return dte
        else:
            return dte - datetime.timedelta(days=1)

    def _near_bank_level(self, price: float) -> bool:
        """Check if price is near a $12.50 bank level (±0.25 tolerance)"""
        BANK_GRID = 12.5
        BANK_TOL = 0.25
        remainder = price % BANK_GRID
        distance = min(remainder, BANK_GRID - remainder)
        return distance <= BANK_TOL

    def _in_macro_window(self, minute: int) -> bool:
        return (minute <= 10) or (minute >= 50)

    def _update_hourly_fvg(self, df: pd.DataFrame) -> None:
        """Update hourly body-gap FVGs (STRICT: body-based, not wick-based)"""
        # Resample to hourly
        h1 = df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        if len(h1) < 2:
            return

        current_hour = df.index[-1].floor('h')
        if self.last_hour_processed == current_hour:
            return
        self.last_hour_processed = current_hour

        # Get last 2 complete hourly candles
        curr_h = h1.iloc[-1]
        prev_h = h1.iloc[-2]

        # STRICT: Body calculations (not wick-based)
        curr_body_hi = max(curr_h['open'], curr_h['close'])
        curr_body_lo = min(curr_h['open'], curr_h['close'])
        prev_body_hi = max(prev_h['open'], prev_h['close'])
        prev_body_lo = min(prev_h['open'], prev_h['close'])

        # Bullish body gap: current body low > previous body high (gap up)
        if curr_body_lo > prev_body_hi:
            self.bull_fvg_low = prev_body_hi
            self.bull_fvg_high = curr_body_lo
            logging.debug(f"Confluence: Bullish FVG detected {self.bull_fvg_low:.2f} - {self.bull_fvg_high:.2f}")

        # Bearish body gap: current body high < previous body low (gap down)
        if curr_body_hi < prev_body_lo:
            self.bear_fvg_low = curr_body_hi
            self.bear_fvg_high = prev_body_lo
            logging.debug(f"Confluence: Bearish FVG detected {self.bear_fvg_low:.2f} - {self.bear_fvg_high:.2f}")

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 120:
            return None

        ts = df.index[-1]
        curr = df.iloc[-1]
        price = curr['close']

        # ========== STRICT DAY/TIME FILTERS ==========

        # Filter 1: Remove Tuesday/Thursday
        weekday = ts.weekday()
        if weekday in [1, 3]:  # Tuesday=1, Thursday=3
            return None

        # Filter 2: Remove 08:50-08:59
        hour = ts.hour
        minute = ts.minute
        if hour == 8 and minute >= 50:
            return None

        # Filter 3: Remove 13:50-13:59
        if hour == 13 and minute >= 50:
            return None

        # Filter 4: Macro window (only minutes 0-10 or 50-59)
        if not self._in_macro_window(minute):
            return None

        # ========== BANK LEVEL CHECK ==========
        if not self._near_bank_level(price):
            return None

        # ========== SESSION TRACKING ==========
        session_date = self._get_session_date(ts)

        if self.current_session != session_date:
            # New session: save previous session H/L
            self.prev_session_high = self.session_high
            self.prev_session_low = self.session_low
            self.session_high = curr['high']
            self.session_low = curr['low']
            self.current_session = session_date
            self.bear_swept_this_session = False
            self.bull_swept_this_session = False
        else:
            # Update session H/L
            self.session_high = max(self.session_high, curr['high']) if self.session_high else curr['high']
            self.session_low = min(self.session_low, curr['low']) if self.session_low else curr['low']

        # Need previous session data
        if self.prev_session_high is None or self.prev_session_low is None:
            return None

        # ========== SWEEP DETECTION (wick-based with re-cross) ==========

        # Bear sweep: wick above previous session high
        if curr['high'] > self.prev_session_high:
            self.bear_swept_this_session = True

        # Bull sweep: wick below previous session low
        if curr['low'] < self.prev_session_low:
            self.bull_swept_this_session = True

        # Re-cross back inside after sweep
        back_inside_after_bear = self.bear_swept_this_session and (curr['close'] < self.prev_session_high)
        back_inside_after_bull = self.bull_swept_this_session and (curr['close'] > self.prev_session_low)

        # ========== HOURLY FVG TRACKING ==========
        self._update_hourly_fvg(df)

        # Check if price is inside FVG
        in_bull_fvg = False
        if self.bull_fvg_low is not None and self.bull_fvg_high is not None:
            in_bull_fvg = self.bull_fvg_low <= price <= self.bull_fvg_high

        in_bear_fvg = False
        if self.bear_fvg_low is not None and self.bear_fvg_high is not None:
            in_bear_fvg = self.bear_fvg_low <= price <= self.bear_fvg_high

        # ========== SIGNAL GENERATION ==========

        # Get dynamic SL/TP
        sltp = dynamic_sltp_engine.calculate_dynamic_sltp(df)

        # LONG: Bull sweep (took low) + re-crossed back inside + in bullish FVG + at bank level
        if back_inside_after_bull and in_bull_fvg:
            logging.info(f"Confluence: LONG signal - Bull sweep + FVG + Bank level @ {price:.2f}")
            dynamic_sltp_engine.log_params(sltp)
            return {
                "strategy": "Confluence",
                "side": "LONG",
                "tp_dist": sltp['tp_dist'],
                "sl_dist": sltp['sl_dist']
            }

        # SHORT: Bear sweep (took high) + re-crossed back inside + in bearish FVG + at bank level
        if back_inside_after_bear and in_bear_fvg:
            logging.info(f"Confluence: SHORT signal - Bear sweep + FVG + Bank level @ {price:.2f}")
            dynamic_sltp_engine.log_params(sltp)
            return {
                "strategy": "Confluence",
                "side": "SHORT",
                "tp_dist": sltp['tp_dist'],
                "sl_dist": sltp['sl_dist']
            }

        return None
