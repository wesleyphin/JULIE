"""SMT Analyzer for JULIE trading bot.

This module provides the SMTAnalyzer class which scans synchronized MNQ and MES
OHLCV DataFrames for specific SMT divergence patterns and emits trading signals
according to predefined context filters and macro windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd


@dataclass
class SMTAnalyzer:
    """Detect SMT divergence patterns and generate actionable signals.

    Parameters
    ----------
    timezone: str
        The timezone used for time-based filters (default: "America/New_York").
    macro_windows: Iterable[Tuple[str, str]]
        Inclusive macro time windows in 24h format (EST). Signals are only
        allowed inside these windows.
    swing_lookback: int
        Lookback window for swing high/low detection (default: 1 -> 3-candle
        fractal).
    """

    timezone: str = "America/New_York"
    macro_windows: Iterable[Tuple[str, str]] = (
        ("09:50", "10:10"),
        ("10:50", "11:10"),
        ("11:50", "12:10"),
    )
    swing_lookback: int = 1

    def generate_signals(self, df_mnq: pd.DataFrame, df_mes: pd.DataFrame) -> pd.DataFrame:
        """Generate SMT-based trading signals.

        Parameters
        ----------
        df_mnq : pd.DataFrame
            OHLCV data for MNQ. Must include columns: Open, High, Low, Close.
        df_mes : pd.DataFrame
            OHLCV data for MES. Must include columns: Open, High, Low, Close.

        Returns
        -------
        pd.DataFrame
            Copy of ``df_mnq`` with an added ``signal`` column: 1 (long),
            -1 (short), or 0 (no trade).
        """

        df_mnq = df_mnq.copy()
        df_mes = df_mes.copy()

        # Align indices to ensure synchronized processing
        common_index = df_mnq.index.intersection(df_mes.index)
        df_mnq = df_mnq.loc[common_index]
        df_mes = df_mes.loc[common_index]

        # Pre-compute structures
        mnq_fvg = self._fvg_levels(df_mnq)
        mes_fvg = self._fvg_levels(df_mes)

        swing_mnq = self._swing_points(df_mnq)
        swing_mes = self._swing_points(df_mes)

        midnight_open = self._midnight_open_series(df_mnq)
        macro_mask = self._macro_window_mask(df_mnq.index)

        # Active FVG zones (forward filled)
        mnq_bull_low = mnq_fvg["bull_low"].ffill()
        mnq_bull_high = mnq_fvg["bull_high"].ffill()
        mnq_bear_low = mnq_fvg["bear_low"].ffill()
        mnq_bear_high = mnq_fvg["bear_high"].ffill()

        mes_bull_low = mes_fvg["bull_low"].ffill()
        mes_bull_high = mes_fvg["bull_high"].ffill()
        mes_bear_low = mes_fvg["bear_low"].ffill()
        mes_bear_high = mes_fvg["bear_high"].ffill()

        # SMT Fill (FVG divergence)
        mnq_bull_fill = self._in_zone(df_mnq, mnq_bull_low, mnq_bull_high)
        mes_stays_above_bull = mes_bull_high.notna() & (df_mes["Low"] > mes_bull_high)
        bull_smt_fill = mnq_bull_fill & mes_stays_above_bull

        mnq_bear_fill = self._in_zone(df_mnq, mnq_bear_low, mnq_bear_high)
        mes_stays_below_bear = mes_bear_low.notna() & (df_mes["High"] < mes_bear_low)
        bear_smt_fill = mnq_bear_fill & mes_stays_below_bear

        # Market Structure Shift (Break of Structure SMT)
        mnq_last_swing_low = swing_mnq["swing_low_price"].ffill().shift()
        mes_last_swing_low = swing_mes["swing_low_price"].ffill().shift()
        mnq_last_swing_high = swing_mnq["swing_high_price"].ffill().shift()
        mes_last_swing_high = swing_mes["swing_high_price"].ffill().shift()

        bull_mss = (
            mnq_last_swing_low.notna()
            & (df_mnq["Close"] < mnq_last_swing_low)
            & (df_mes["Close"] >= mes_last_swing_low)
        )

        bear_mss = (
            mnq_last_swing_high.notna()
            & (df_mnq["Close"] > mnq_last_swing_high)
            & (df_mes["Close"] <= mes_last_swing_high)
        )

        # Precision Swing Point (color flip at swing extreme)
        swing_low_both = swing_mnq["swing_low"] & swing_mes["swing_low"]
        swing_high_both = swing_mnq["swing_high"] & swing_mes["swing_high"]

        bull_psp = swing_low_both & (df_mnq["Close"] < df_mnq["Open"]) & (
            df_mes["Close"] > df_mes["Open"]
        )
        bear_psp = swing_high_both & (df_mnq["Close"] > df_mnq["Open"]) & (
            df_mes["Close"] < df_mes["Open"]
        )

        # Sequential SMT (time-based higher high / lower low)
        bull_seq = (df_mnq["Low"] < df_mnq["Low"].shift(1)) & (
            df_mes["Low"] > df_mes["Low"].shift(1)
        )
        bear_seq = (df_mnq["High"] > df_mnq["High"].shift(1)) & (
            df_mes["High"] < df_mes["High"].shift(1)
        )

        bull_pattern = bull_smt_fill | bull_mss | bull_seq | bull_psp
        bear_pattern = bear_smt_fill | bear_mss | bear_seq | bear_psp

        price_discount = df_mnq["Close"] < midnight_open
        price_premium = df_mnq["Close"] > midnight_open

        long_signals = bull_pattern & price_discount & macro_mask
        short_signals = bear_pattern & price_premium & macro_mask

        df_mnq["signal"] = 0
        df_mnq.loc[long_signals, "signal"] = 1
        df_mnq.loc[short_signals, "signal"] = -1

        return df_mnq

    def _fvg_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify bullish and bearish Fair Value Gaps (FVG).

        A bullish FVG occurs when the current low is greater than the high two
        periods ago, leaving a gap. A bearish FVG occurs when the current high is
        lower than the low two periods ago.
        """

        bull_gap = df["Low"] > df["High"].shift(2)
        bear_gap = df["High"] < df["Low"].shift(2)

        fvg = pd.DataFrame(index=df.index)
        fvg["bull_low"] = df["High"].shift(2).where(bull_gap)
        fvg["bull_high"] = df["Low"].where(bull_gap)
        fvg["bear_low"] = df["High"].where(bear_gap)
        fvg["bear_high"] = df["Low"].shift(2).where(bear_gap)
        return fvg

    def _swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect swing highs and lows using a fractal pattern."""

        lb = self.swing_lookback
        highs = df["High"]
        lows = df["Low"]

        swing_high = (
            highs == highs.rolling(window=lb * 2 + 1, center=True).max()
        ) & highs.notna()
        swing_low = (lows == lows.rolling(window=lb * 2 + 1, center=True).min()) & lows.notna()

        swings = pd.DataFrame(index=df.index)
        swings["swing_high"] = swing_high
        swings["swing_low"] = swing_low
        swings["swing_high_price"] = df["High"].where(swing_high)
        swings["swing_low_price"] = df["Low"].where(swing_low)
        return swings

    @staticmethod
    def _in_zone(df: pd.DataFrame, lower: pd.Series, upper: pd.Series) -> pd.Series:
        """Check whether price trades inside a zone defined by lower/upper bounds."""

        return (
            lower.notna()
            & upper.notna()
            & (df["Low"] <= upper)
            & (df["High"] >= lower)
        )

    def _midnight_open_series(self, df: pd.DataFrame) -> pd.Series:
        """Compute the NY midnight open price for each trading day."""

        df_tz = self._ensure_timezone(df)
        day_index = pd.Index(df_tz.index.date, name="session_date")
        midnight_open = df_tz["Open"].groupby(day_index).transform("first")

        # Align back to original index order
        midnight_open.index = df.index
        return midnight_open

    def _macro_window_mask(self, index: pd.Index) -> pd.Series:
        """Boolean mask for timestamps inside macro windows (EST)."""

        idx_tz = self._ensure_datetime_index(index)
        times = idx_tz.tz_convert(self.timezone).time

        mask = pd.Series(False, index=index)
        for start, end in self.macro_windows:
            start_time = pd.to_datetime(start).time()
            end_time = pd.to_datetime(end).time()
            mask |= (times >= start_time) & (times <= end_time)
        return mask

    def _ensure_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame index is timezone-aware in the configured timezone."""

        idx = self._ensure_datetime_index(df.index)
        if idx.tzinfo is None:
            idx = idx.tz_localize(self.timezone)
        else:
            idx = idx.tz_convert(self.timezone)
        df = df.copy()
        df.index = idx
        return df

    def _ensure_datetime_index(self, index: pd.Index) -> pd.DatetimeIndex:
        """Guarantee a DatetimeIndex."""

        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("Index must be a DatetimeIndex for SMT analysis.")
        return index
