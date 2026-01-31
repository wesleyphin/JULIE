import logging
from typing import Dict, Optional

import pandas as pd

from config import CONFIG
from dynamic_sltp_params import dynamic_sltp_engine
from strategy_base import Strategy
from volatility_filter import volatility_filter
from volume_profile import build_volume_profile


class ValueAreaBreakoutStrategy(Strategy):
    """
    Volume profile value-area breakout/acceptance.
    - Go long if price accepts above VAH for N bars.
    - Go short if price accepts below VAL for N bars.
    """

    def __init__(self) -> None:
        cfg = CONFIG.get("VALUE_AREA_BREAKOUT", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.lookback = int(cfg.get("lookback", 120))
        self.value_area_pct = float(cfg.get("value_area_pct", 0.70))
        self.accept_bars = int(cfg.get("accept_bars", 2))
        self.buffer = float(cfg.get("buffer", 0.10))
        self.sessions = set(cfg.get("sessions", ["NY_AM", "NY_PM", "LONDON"]))
        self.tick_size = float(cfg.get("tick_size", 0.25))

        logging.info(
            "ValueAreaBreakoutStrategy initialized | lookback=%s accept_bars=%s",
            self.lookback,
            self.accept_bars,
        )

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if not self.enabled:
            return None
        if df is None or len(df) < self.lookback + self.accept_bars + 2:
            return None

        ts = df.index[-1]
        session = volatility_filter.get_session(ts.hour)
        if self.sessions and session not in self.sessions:
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

        closes = df["close"].iloc[-self.accept_bars :]
        if (closes > vah + self.buffer).all():
            sltp = dynamic_sltp_engine.calculate_sltp("ValueAreaBreakout_LONG", df)
            logging.info("ValueAreaBreakout: LONG acceptance above VAH %.2f", vah)
            dynamic_sltp_engine.log_params(sltp, "ValueAreaBreakout_LONG")
            return {
                "strategy": "ValueAreaBreakout",
                "side": "LONG",
                "tp_dist": sltp["tp_dist"],
                "sl_dist": sltp["sl_dist"],
            }

        if (closes < val - self.buffer).all():
            sltp = dynamic_sltp_engine.calculate_sltp("ValueAreaBreakout_SHORT", df)
            logging.info("ValueAreaBreakout: SHORT acceptance below VAL %.2f", val)
            dynamic_sltp_engine.log_params(sltp, "ValueAreaBreakout_SHORT")
            return {
                "strategy": "ValueAreaBreakout",
                "side": "SHORT",
                "tp_dist": sltp["tp_dist"],
                "sl_dist": sltp["sl_dist"],
            }

        return None
