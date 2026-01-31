import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from dynamic_sltp_params import dynamic_sltp_engine
from strategy_base import Strategy
from volatility_filter import volatility_filter


class ImpulseBreakoutStrategy(Strategy):
    """
    Impulse breakout strategy:
    - Detects expansion bars (range + volume) and breaks of recent high/low.
    """

    def __init__(self) -> None:
        cfg = CONFIG.get("IMPULSE_BREAKOUT", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.lookback = int(cfg.get("lookback", 20))
        self.range_mult = float(cfg.get("range_mult", 1.5))
        self.volume_mult = float(cfg.get("volume_mult", 1.2))
        self.breakout_buffer_atr = float(cfg.get("breakout_buffer_atr", 0.1))
        self.atr_window = int(cfg.get("atr_window", 20))
        self.min_range = float(cfg.get("min_range", 0.75))
        self.sessions = set(cfg.get("sessions", ["NY_AM", "NY_PM", "LONDON"]))

        logging.info(
            "ImpulseBreakoutStrategy initialized | lookback=%s range_mult=%.2f vol_mult=%.2f",
            self.lookback,
            self.range_mult,
            self.volume_mult,
        )

    def _calc_atr(self, df: pd.DataFrame) -> Optional[float]:
        if df is None or len(df) < max(self.atr_window, 2):
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = pd.concat(
            [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self.atr_window).mean().iloc[-1]
        return float(atr) if np.isfinite(atr) else None

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if not self.enabled:
            return None
        if df is None or len(df) < self.lookback + 2:
            return None

        ts = df.index[-1]
        session = volatility_filter.get_session(ts.hour)
        if self.sessions and session not in self.sessions:
            return None

        curr = df.iloc[-1]
        prev_window = df.iloc[-(self.lookback + 1) : -1]
        if prev_window.empty:
            return None

        prev_high = float(prev_window["high"].max())
        prev_low = float(prev_window["low"].min())

        bar_range = float(curr["high"] - curr["low"])
        range_series = prev_window["high"] - prev_window["low"]
        avg_range = float(range_series.mean()) if len(range_series) else 0.0

        if not np.isfinite(bar_range) or not np.isfinite(avg_range):
            return None

        range_ok = bar_range >= max(self.min_range, avg_range * self.range_mult)

        volume_ok = True
        if "volume" in df.columns:
            vol_series = prev_window["volume"]
            avg_vol = float(vol_series.mean()) if len(vol_series) else 0.0
            volume_ok = avg_vol > 0 and float(curr["volume"]) >= avg_vol * self.volume_mult

        if not range_ok or not volume_ok:
            return None

        atr = self._calc_atr(df)
        buffer = (atr or 0.0) * self.breakout_buffer_atr

        close = float(curr["close"])
        open_ = float(curr["open"])

        # LONG breakout
        if close > prev_high + buffer and close > open_:
            sltp = dynamic_sltp_engine.calculate_sltp("ImpulseBreakout_LONG", df)
            logging.info("ImpulseBreakout: LONG signal | close=%.2f > prev_high=%.2f", close, prev_high)
            dynamic_sltp_engine.log_params(sltp, "ImpulseBreakout_LONG")
            return {
                "strategy": "ImpulseBreakout",
                "side": "LONG",
                "tp_dist": sltp["tp_dist"],
                "sl_dist": sltp["sl_dist"],
            }

        # SHORT breakout
        if close < prev_low - buffer and close < open_:
            sltp = dynamic_sltp_engine.calculate_sltp("ImpulseBreakout_SHORT", df)
            logging.info("ImpulseBreakout: SHORT signal | close=%.2f < prev_low=%.2f", close, prev_low)
            dynamic_sltp_engine.log_params(sltp, "ImpulseBreakout_SHORT")
            return {
                "strategy": "ImpulseBreakout",
                "side": "SHORT",
                "tp_dist": sltp["tp_dist"],
                "sl_dist": sltp["sl_dist"],
            }

        return None
