import logging
import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from dynamic_sltp_params import dynamic_sltp_engine
from strategy_base import Strategy


class ICTModelStrategy(Strategy):
    """ICT Model Strategy implementation."""
    def __init__(self):
        self.reset_daily()
        # 5m FVG state
        self.curr_bull_fvg_low = np.nan
        self.curr_bull_fvg_high = np.nan
        self.curr_bear_fvg_low = np.nan
        self.curr_bear_fvg_high = np.nan
        self.fvg_bias = None
        self.last_5m_bar_ts = None
        # 1m bearish FVG for inversion trigger
        self.bear_fvg_1m_active = False
        self.bear_fvg_1m_high = np.nan

    def reset_daily(self):
        """Reset daily state"""
        self.current_date = None
        self.pdl = None
        self.pdh = None
        self.open_10am = None
        # Manipulation/setup tracking
        self.pending_long = False
        self.long_stop = np.nan
        self.setup_bar_count = 0

    def _is_in_ny_am_session(self, ts) -> bool:
        """Check if timestamp is within NY AM session (9:30-11:00 ET)"""
        t = ts.time()
        return (
            (t.hour == 9 and t.minute >= 30) or
            (t.hour == 10) or
            (t.hour == 11 and t.minute == 0)
        )

    def _update_pdh_pdl(self, df: pd.DataFrame, current_date) -> None:
        """Calculate Previous Day High/Low from historical data"""
        df_reset = df.reset_index()
        df_reset['date'] = df_reset['ts'].dt.date

        dates = sorted(df_reset['date'].unique())
        if len(dates) >= 2:
            prev_dates = [d for d in dates if d < current_date]
            if prev_dates:
                prev_date = prev_dates[-1]
                prev_day_data = df_reset[df_reset['date'] == prev_date]
                if not prev_day_data.empty:
                    self.pdh = prev_day_data['high'].max()
                    self.pdl = prev_day_data['low'].min()

    def _update_10am_open(self, df: pd.DataFrame, current_date) -> None:
        """Get the 10AM candle open for the current day"""
        df_reset = df.reset_index()
        df_reset['date'] = df_reset['ts'].dt.date
        df_reset['hour'] = df_reset['ts'].dt.hour
        df_reset['minute'] = df_reset['ts'].dt.minute

        today_10am = df_reset[
            (df_reset['date'] == current_date) &
            (df_reset['hour'] == 10) &
            (df_reset['minute'] == 0)
        ]
        if not today_10am.empty:
            self.open_10am = today_10am.iloc[0]['open']

    def _detect_5m_fvgs(self, df: pd.DataFrame) -> None:
        """Detect 5-minute FVGs and track respect/disrespect for bias"""
        df_5m = df.resample('5min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        if len(df_5m) < 3:
            return

        latest_5m_ts = df_5m.index[-1]
        if self.last_5m_bar_ts == latest_5m_ts:
            return
        self.last_5m_bar_ts = latest_5m_ts

        prev_5m = df_5m.iloc[-2]
        curr_5m = df_5m.iloc[-1]

        # Bullish FVG detection
        if curr_5m['low'] > prev_5m['high']:
            self.curr_bull_fvg_low = prev_5m['high']
            self.curr_bull_fvg_high = curr_5m['low']
        # Bearish FVG detection
        if curr_5m['high'] < prev_5m['low']:
            self.curr_bear_fvg_low = curr_5m['high']
            self.curr_bear_fvg_high = prev_5m['low']

        # Respect / disrespect tracking for bias
        last_interaction = None

        if not np.isnan(self.curr_bull_fvg_low):
            if curr_5m['low'] >= self.curr_bull_fvg_low and curr_5m['low'] <= self.curr_bull_fvg_high:
                last_interaction = 'bull_respect'
            elif curr_5m['low'] < self.curr_bull_fvg_low:
                last_interaction = 'bull_disrespect'
                self.curr_bull_fvg_low = np.nan
                self.curr_bull_fvg_high = np.nan

        if not np.isnan(self.curr_bear_fvg_low):
            if curr_5m['high'] <= self.curr_bear_fvg_high and curr_5m['high'] >= self.curr_bear_fvg_low:
                last_interaction = 'bear_respect'
            elif curr_5m['high'] > self.curr_bear_fvg_high:
                last_interaction = 'bear_disrespect'
                self.curr_bear_fvg_low = np.nan
                self.curr_bear_fvg_high = np.nan

        if last_interaction in ['bull_respect', 'bear_disrespect']:
            self.fvg_bias = 'BULL'
        elif last_interaction in ['bear_respect', 'bull_disrespect']:
            self.fvg_bias = 'BEAR'

    def _detect_1m_bearish_fvg(self, df: pd.DataFrame) -> None:
        """Detect 1-minute bearish FVG for inversion trigger"""
        if len(df) < 3:
            return

        highs = df['high'].values
        lows = df['low'].values

        # Bearish FVG: current high < 2-bars-ago low
        if highs[-1] < lows[-3]:
            self.bear_fvg_1m_high = lows[-3]
            self.bear_fvg_1m_active = True

    def _check_ifvg_long_trigger(self, curr_close: float) -> bool:
        """Check for Inversion FVG trigger (bearish FVG closed above)"""
        if self.bear_fvg_1m_active and curr_close > self.bear_fvg_1m_high:
            self.bear_fvg_1m_active = False
            return True
        return False

    def _lrl_ok_for_long(self, df: pd.DataFrame, stop_price: float,
                         lookback: int = 20, tolerance: float = 0.75) -> bool:
        """LRL filter: Avoid longs if equal lows clustered near stop level"""
        if len(df) < lookback:
            return True
        lows = df['low'].values[-lookback:]
        near = np.abs(lows - stop_price) <= tolerance
        return near.sum() < 2

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 100:
            return None

        ts = df.index[-1]
        curr = df.iloc[-1]
        curr_date = ts.date()

        # Reset on new day
        if self.current_date != curr_date:
            self.reset_daily()
            self.current_date = curr_date
            self._update_pdh_pdl(df, curr_date)

        # Update 10AM open
        if ts.hour >= 10 and self.open_10am is None:
            self._update_10am_open(df, curr_date)

        # Detect 1m bearish FVG
        self._detect_1m_bearish_fvg(df)

        # Check for IFVG trigger
        ifvg_long = self._check_ifvg_long_trigger(curr['close'])

        # Update 5m FVG tracking
        self._detect_5m_fvgs(df)

        # Only process during NY AM session (9:30-11:00 ET)
        if not self._is_in_ny_am_session(ts):
            return None

        # Expire old setups (90 bars max)
        if self.pending_long:
            self.setup_bar_count += 1
            if self.setup_bar_count > 90:
                self.pending_long = False
                self.setup_bar_count = 0

        # Determine 4H candle bias (after 10AM)
        candle_bias = None
        if ts.hour >= 10 and self.open_10am is not None:
            if curr['close'] > self.open_10am:
                candle_bias = 'BULL'
            elif curr['close'] < self.open_10am:
                candle_bias = 'BEAR'

        # Combined bullish bias check
        is_bullish = False
        if candle_bias == 'BULL' and self.fvg_bias == 'BULL':
            is_bullish = True
        elif candle_bias == 'BULL' and self.fvg_bias is None:
            is_bullish = True
        elif self.fvg_bias == 'BULL' and candle_bias is None:
            is_bullish = True

        # MANIPULATION DETECTION (Key Level Touch)
        if is_bullish and not self.pending_long:
            key_level_touched = False

            # Check PDL sweep/touch
            if self.pdl is not None and curr['low'] <= self.pdl:
                key_level_touched = True
                self.long_stop = curr['low'] - 1

            # Check bullish 5m FVG touch
            if not np.isnan(self.curr_bull_fvg_low):
                if curr['low'] <= self.curr_bull_fvg_high and curr['low'] >= self.curr_bull_fvg_low:
                    key_level_touched = True
                    self.long_stop = self.curr_bull_fvg_low - 1

            if key_level_touched:
                self.pending_long = True
                self.setup_bar_count = 0
                logging.info(f"ICT: Manipulation detected. PDL={self.pdl}, Bull FVG Low={self.curr_bull_fvg_low}")

        # ENTRY on IFVG trigger + LRL filter
        if self.pending_long and ifvg_long and is_bullish:
            if self._lrl_ok_for_long(df, self.long_stop):
                self.pending_long = False
                self.setup_bar_count = 0
                sltp = dynamic_sltp_engine.calculate_dynamic_sltp(df)
                logging.info(f"ICT: LONG signal triggered at {curr['close']:.2f}")
                dynamic_sltp_engine.log_params(sltp)
                return {
                    "strategy": "ICT_Model",
                    "side": "LONG",
                    "tp_dist": sltp['tp_dist'],
                    "sl_dist": sltp['sl_dist']
                }

        return None
