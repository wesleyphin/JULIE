import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from dynamic_sltp_params import dynamic_sltp_engine
from strategy_base import Strategy
from volatility_filter import volatility_filter
from volume_profile import build_volume_profile


class AuctionReversionStrategy(Strategy):
    """
    Auction/range mean reversion using volume profile value area.
    - Fade VAL (long) or VAH (short) when structure is rotational (low ER).
    """

    def __init__(self) -> None:
        cfg = CONFIG.get("AUCTION_REVERSION", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.lookback = int(cfg.get("lookback", 120))
        self.value_area_pct = float(cfg.get("value_area_pct", 0.70))
        self.touch_buffer = float(cfg.get("touch_buffer", 0.25))
        self.er_window = int(cfg.get("er_window", 30))
        self.er_max = float(cfg.get("er_max", 0.20))
        self.min_range = float(cfg.get("min_range", 4.0))
        self.sessions = set(cfg.get("sessions", ["NY_AM", "NY_PM", "LONDON"]))
        self.skip_high_vol = bool(cfg.get("skip_high_vol", True))
        self.tick_size = float(cfg.get("tick_size", 0.25))

        logging.info(
            "AuctionReversionStrategy initialized | lookback=%s er_max=%.2f va=%.2f",
            self.lookback,
            self.er_max,
            self.value_area_pct,
        )

    def _efficiency_ratio(self, closes: pd.Series) -> Optional[float]:
        if closes is None or len(closes) < self.er_window + 1:
            return None
        start = closes.iloc[-self.er_window - 1]
        end = closes.iloc[-1]
        net = abs(float(end) - float(start))
        total = float((closes.diff().abs().iloc[-self.er_window:]).sum())
        if total <= 0:
            return None
        return net / total

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if not self.enabled:
            return None
        if df is None or len(df) < self.lookback + self.er_window + 2:
            return None

        ts = df.index[-1]
        session = volatility_filter.get_session(ts.hour)
        if self.sessions and session not in self.sessions:
            return None

        if self.skip_high_vol:
            vol_regime, _, _ = volatility_filter.get_regime(df, ts)
            if str(vol_regime).lower() == "high":
                return None

        curr = df.iloc[-1]
        close = float(curr["close"])

        range_window = df.iloc[-self.lookback :]
        if float(range_window["high"].max() - range_window["low"].min()) < self.min_range:
            return None

        er = self._efficiency_ratio(df["close"])
        if er is None or er > self.er_max:
            return None

        profile = build_volume_profile(
            df,
            lookback=self.lookback,
            tick_size=self.tick_size,
            value_area_pct=self.value_area_pct,
        )
        if not profile:
            return None

        vah = float(profile["vah"])
        val = float(profile["val"])

        if close >= vah - self.touch_buffer:
            sltp = dynamic_sltp_engine.calculate_sltp("AuctionReversion_SHORT", df)
            logging.info("AuctionReversion: SHORT at VAH %.2f (close %.2f)", vah, close)
            dynamic_sltp_engine.log_params(sltp, "AuctionReversion_SHORT")
            return {
                "strategy": "AuctionReversion",
                "side": "SHORT",
                "tp_dist": sltp["tp_dist"],
                "sl_dist": sltp["sl_dist"],
            }

        if close <= val + self.touch_buffer:
            sltp = dynamic_sltp_engine.calculate_sltp("AuctionReversion_LONG", df)
            logging.info("AuctionReversion: LONG at VAL %.2f (close %.2f)", val, close)
            dynamic_sltp_engine.log_params(sltp, "AuctionReversion_LONG")
            return {
                "strategy": "AuctionReversion",
                "side": "LONG",
                "tp_dist": sltp["tp_dist"],
                "sl_dist": sltp["sl_dist"],
            }

        return None
