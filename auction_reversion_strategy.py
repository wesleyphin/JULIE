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
        self.touch_buffer_atr = float(cfg.get("touch_buffer_atr", 0.10))
        self.require_rejection = bool(cfg.get("require_rejection", True))
        self.rejection_close_buffer = float(cfg.get("rejection_close_buffer", 0.0))
        self.rejection_close_buffer_atr = float(cfg.get("rejection_close_buffer_atr", 0.0))
        self.er_window = int(cfg.get("er_window", 30))
        self.er_max = float(cfg.get("er_max", 0.20))
        self.min_range = float(cfg.get("min_range", 4.0))
        self.sessions = set(cfg.get("sessions", ["NY_AM", "NY_PM", "LONDON"]))
        self.skip_high_vol = bool(cfg.get("skip_high_vol", True))
        self.tick_size = float(cfg.get("tick_size", 0.25))
        self.atr_window = int(cfg.get("atr_window", 20))
        self.rejection_wick_min = float(cfg.get("rejection_wick_min", 0.0))
        self.rejection_wick_atr = float(cfg.get("rejection_wick_atr", 0.0))
        self.volume_mult = float(cfg.get("volume_mult", 1.0))
        self.trend_ema_period = int(cfg.get("trend_ema_period", 50) or 50)
        self.long_only_above_ema = bool(cfg.get("long_only_above_ema", False))
        self.short_only_below_ema = bool(cfg.get("short_only_below_ema", False))
        self.cooldown_bars = int(cfg.get("cooldown_bars", 0) or 0)
        self._last_signal_ts: Optional[pd.Timestamp] = None

        logging.info(
            "AuctionReversionStrategy initialized | lookback=%s er_max=%.2f va=%.2f",
            self.lookback,
            self.er_max,
            self.value_area_pct,
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

    def _ema_value(self, closes: pd.Series, period: int) -> Optional[float]:
        if closes is None or len(closes) < max(int(period), 2):
            return None
        ema = closes.ewm(span=int(period), adjust=False).mean().iloc[-1]
        return float(ema) if np.isfinite(ema) else None

    def _is_cooldown_active(self, ts: pd.Timestamp) -> bool:
        if self.cooldown_bars <= 0 or self._last_signal_ts is None:
            return False
        try:
            elapsed = (pd.Timestamp(ts) - pd.Timestamp(self._last_signal_ts)).total_seconds() / 60.0
            return float(elapsed) < float(self.cooldown_bars)
        except Exception:
            return False

    def _mark_signal(self, ts: pd.Timestamp) -> None:
        try:
            self._last_signal_ts = pd.Timestamp(ts)
        except Exception:
            self._last_signal_ts = None

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if not self.enabled:
            return None
        if df is None or len(df) < self.lookback + self.er_window + 2:
            return None

        ts = df.index[-1]
        session = volatility_filter.get_session(ts.hour)
        if self.sessions and session not in self.sessions:
            return None
        if self._is_cooldown_active(ts):
            return None

        if self.skip_high_vol:
            vol_regime, _, _ = volatility_filter.get_regime(df, ts)
            if str(vol_regime).lower() == "high":
                return None

        curr = df.iloc[-1]
        close = float(curr["close"])
        high = float(curr["high"])
        low = float(curr["low"])
        open_price = float(curr["open"]) if "open" in curr else close

        atr = self._calc_atr(df)
        trend_ema = self._ema_value(df["close"], self.trend_ema_period)
        buffer = max(self.touch_buffer, (atr or 0.0) * self.touch_buffer_atr)
        close_buffer = max(
            self.rejection_close_buffer,
            (atr or 0.0) * self.rejection_close_buffer_atr,
        )
        wick_min = max(self.rejection_wick_min, (atr or 0.0) * self.rejection_wick_atr)
        upper_wick = max(0.0, high - max(open_price, close))
        lower_wick = max(0.0, min(open_price, close) - low)

        if self.volume_mult > 1.0 and "volume" in df.columns:
            vol_series = df["volume"].iloc[-self.lookback :]
            avg_vol = float(vol_series.mean()) if len(vol_series) else 0.0
            current_vol = float(curr.get("volume", 0.0))
            if avg_vol > 0 and current_vol < avg_vol * self.volume_mult:
                return None

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

        if self.require_rejection:
            short_reject = (
                high >= vah + buffer
                and close <= vah - close_buffer
                and upper_wick >= wick_min
            )
        else:
            short_reject = close >= vah - buffer
        if short_reject and self.short_only_below_ema and trend_ema is not None and close > trend_ema:
            short_reject = False

        if short_reject:
            sltp = dynamic_sltp_engine.calculate_sltp("AuctionReversion_SHORT", df)
            logging.info("AuctionReversion: SHORT at VAH %.2f (close %.2f)", vah, close)
            dynamic_sltp_engine.log_params(sltp, "AuctionReversion_SHORT")
            self._mark_signal(ts)
            return {
                "strategy": "AuctionReversion",
                "side": "SHORT",
                "tp_dist": sltp["tp_dist"],
                "sl_dist": sltp["sl_dist"],
            }

        if self.require_rejection:
            long_reject = (
                low <= val - buffer
                and close >= val + close_buffer
                and lower_wick >= wick_min
            )
        else:
            long_reject = close <= val + buffer
        if long_reject and self.long_only_above_ema and trend_ema is not None and close < trend_ema:
            long_reject = False

        if long_reject:
            sltp = dynamic_sltp_engine.calculate_sltp("AuctionReversion_LONG", df)
            logging.info("AuctionReversion: LONG at VAL %.2f (close %.2f)", val, close)
            dynamic_sltp_engine.log_params(sltp, "AuctionReversion_LONG")
            self._mark_signal(ts)
            return {
                "strategy": "AuctionReversion",
                "side": "LONG",
                "tp_dist": sltp["tp_dist"],
                "sl_dist": sltp["sl_dist"],
            }

        return None
