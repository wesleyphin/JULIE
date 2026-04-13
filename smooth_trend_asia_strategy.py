import logging
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from config import CONFIG
from strategy_base import Strategy


NY_TZ = ZoneInfo("America/New_York")


def _get_session(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=NY_TZ)
    else:
        ts = ts.astimezone(NY_TZ)
    hour = ts.hour
    if hour >= 18 or hour < 3:
        return "ASIA"
    if 3 <= hour < 8:
        return "LONDON"
    if 8 <= hour < 12:
        return "NY_AM"
    if 12 <= hour < 17:
        return "NY_PM"
    return "OFF"


class SmoothTrendAsiaStrategy(Strategy):
    """
    Smooth Trend Asia strategy.

    Trigger A: reclaim EMA20, then next bar makes higher high (long) or lower low (short).
    Stop: structure-based with max stop cap.
    """

    def __init__(self):
        self.strategy_name = "SmoothTrendAsia"
        self.cfg = CONFIG.get("SMOOTH_TREND_ASIA") or CONFIG.get("BACKTEST_SMOOTH_TREND_ASIA", {}) or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.last_signal_bar = None

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr_parts = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        )
        return tr_parts.max(axis=1)

    def _round_up(self, value: float, tick_size: float) -> float:
        return math.ceil(value / tick_size) * tick_size

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if not self.enabled or df is None or df.empty:
            return None

        ts = df.index[-1]
        if _get_session(ts) != "ASIA":
            return None

        cfg = self.cfg
        ema_fast = int(cfg.get("ema_fast", 20) or 20)
        ema_slow = int(cfg.get("ema_slow", 50) or 50)
        ema_slope_bars = int(cfg.get("ema_slope_bars", 20) or 20)
        er_window = int(cfg.get("er_window", 60) or 60)
        persistence_window = int(cfg.get("persistence_window", 60) or 60)
        closes_side_window = int(cfg.get("closes_side_window", 60) or 60)
        atr_window = int(cfg.get("atr_window", 20) or 20)
        atr_long_window = int(cfg.get("atr_long_window", 120) or 120)
        pullback_lookback = int(cfg.get("pullback_lookback", 20) or 20)
        cooldown_bars = int(cfg.get("cooldown_bars", 12) or 0)

        required = max(
            ema_slow,
            ema_slope_bars + 1,
            er_window + 1,
            persistence_window + 1,
            closes_side_window + 1,
            atr_long_window + 1,
            pullback_lookback + 1,
            3,
        )
        if len(df) < required:
            return None

        bar_idx = len(df)
        if cooldown_bars > 0 and self.last_signal_bar is not None:
            if bar_idx - self.last_signal_bar <= cooldown_bars:
                return None

        close = df["close"]
        ema_fast_series = close.ewm(span=ema_fast, adjust=False).mean()
        ema_slow_series = close.ewm(span=ema_slow, adjust=False).mean()
        ema_fast_val = float(ema_fast_series.iloc[-1])
        ema_slow_val = float(ema_slow_series.iloc[-1])
        slope = ema_fast_val - float(ema_fast_series.iloc[-ema_slope_bars])
        min_sep = float(cfg.get("min_ema_separation", 0.1) or 0.0)

        bias = None
        if ema_fast_val > ema_slow_val and slope > 0 and (ema_fast_val - ema_slow_val) >= min_sep:
            bias = "LONG"
        elif ema_fast_val < ema_slow_val and slope < 0 and (ema_slow_val - ema_fast_val) >= min_sep:
            bias = "SHORT"
        else:
            return None

        tr = self._true_range(df)
        atr_short = tr.rolling(atr_window).mean().iloc[-1]
        atr_long = tr.rolling(atr_long_window).mean().iloc[-1]
        max_tr = tr.rolling(atr_window).max().iloc[-1]
        if np.isnan(atr_short) or np.isnan(atr_long) or np.isnan(max_tr):
            return None
        if atr_short <= 0 or atr_long <= 0:
            return None

        er_min = float(cfg.get("er_min", 0.55) or 0.55)
        direction = abs(close.iloc[-1] - close.iloc[-er_window])
        volatility = close.diff().abs().iloc[-er_window:].sum()
        er = direction / volatility if volatility > 0 else 0.0

        persistence_min = float(cfg.get("persistence_min", 0.65) or 0.65)
        diffs = close.diff().iloc[-persistence_window:]
        if bias == "LONG":
            persistence = float((diffs > 0).mean())
        else:
            persistence = float((diffs < 0).mean())

        closes_side_min = float(cfg.get("closes_side_min", 0.80) or 0.80)
        close_window = close.iloc[-closes_side_window:]
        ema_fast_window = ema_fast_series.iloc[-closes_side_window:]
        if bias == "LONG":
            closes_side = float((close_window > ema_fast_window).mean())
        else:
            closes_side = float((close_window < ema_fast_window).mean())

        atr_ratio_max = float(cfg.get("atr_ratio_max", 1.15) or 1.15)
        max_tr_mult = float(cfg.get("max_tr_mult", 2.2) or 2.2)
        atr_ratio = atr_short / atr_long if atr_long > 0 else float("inf")
        atr_ok = (atr_ratio <= atr_ratio_max) and (max_tr <= atr_short * max_tr_mult)

        regime_min_passes = int(cfg.get("regime_min_passes", 3) or 3)
        passes = sum(
            [
                er >= er_min,
                persistence >= persistence_min,
                closes_side >= closes_side_min,
                atr_ok,
            ]
        )
        if passes < regime_min_passes:
            return None

        pullback_window = df.iloc[-pullback_lookback:]
        pullback_low = float(pullback_window["low"].min())
        pullback_high = float(pullback_window["high"].max())
        pullback_range = pullback_high - pullback_low
        if pullback_range <= 0:
            return None

        touch_mult = float(cfg.get("pullback_touch_atr_mult", 0.2) or 0.2)
        max_dd_mult = float(cfg.get("pullback_max_drawdown_mult", 0.8) or 0.8)
        ema50_buffer_mult = float(cfg.get("pullback_ema50_buffer_mult", 0.2) or 0.2)

        if bias == "LONG":
            touch_ok = pullback_low <= ema_fast_val + touch_mult * atr_short
            drawdown_ok = pullback_range <= max_dd_mult * atr_short
            no_break = (pullback_window["close"] >= (ema_slow_val - ema50_buffer_mult * atr_short)).all()
        else:
            touch_ok = pullback_high >= ema_fast_val - touch_mult * atr_short
            drawdown_ok = pullback_range <= max_dd_mult * atr_short
            no_break = (pullback_window["close"] <= (ema_slow_val + ema50_buffer_mult * atr_short)).all()

        if not (touch_ok and drawdown_ok and no_break):
            return None

        reclaim_bar = df.iloc[-2]
        current_bar = df.iloc[-1]
        ema_fast_prev = float(ema_fast_series.iloc[-2])

        if bias == "LONG":
            reclaim_ok = reclaim_bar["close"] > ema_fast_prev
            confirm_ok = current_bar["high"] > reclaim_bar["high"]
        else:
            reclaim_ok = reclaim_bar["close"] < ema_fast_prev
            confirm_ok = current_bar["low"] < reclaim_bar["low"]

        if not (reclaim_ok and confirm_ok):
            return None

        entry_price = float(current_bar["close"])
        max_stop_points = float(cfg.get("max_stop_points", 2.5) or 2.5)
        stop_ema50_buffer_mult = float(cfg.get("stop_ema50_buffer_mult", 0.3) or 0.3)
        if bias == "LONG":
            stop_structure = min(pullback_low, ema_slow_val - stop_ema50_buffer_mult * atr_short)
            stop_cap = entry_price - max_stop_points
            stop_price = max(stop_structure, stop_cap)
            sl_dist = entry_price - stop_price
        else:
            stop_structure = max(pullback_high, ema_slow_val + stop_ema50_buffer_mult * atr_short)
            stop_cap = entry_price + max_stop_points
            stop_price = min(stop_structure, stop_cap)
            sl_dist = stop_price - entry_price

        if sl_dist <= 0:
            return None

        tick_size = float(cfg.get("tick_size", 0.25) or 0.25)
        sl_dist = min(sl_dist, max_stop_points)
        sl_dist = max(sl_dist, tick_size)
        sl_dist = self._round_up(sl_dist, tick_size)
        if sl_dist > max_stop_points:
            sl_dist = max_stop_points

        tp_mult = float(cfg.get("tp_mult", 1.5) or 1.5)
        min_tp_points = float(cfg.get("min_tp_points", 1.0) or 1.0)
        tp_dist = max(sl_dist * tp_mult, min_tp_points)
        tp_dist = self._round_up(tp_dist, tick_size)

        self.last_signal_bar = bar_idx
        logging.info(
            f"SmoothTrendAsia {bias} signal | ER={er:.2f} pers={persistence:.2f} "
            f"side_ok={closes_side:.2f} ATRr={atr_ratio:.2f} "
            f"SL={sl_dist:.2f} TP={tp_dist:.2f}"
        )
        return {
            "strategy": self.strategy_name,
            "sub_strategy": "TriggerA",
            "side": bias,
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
        }
