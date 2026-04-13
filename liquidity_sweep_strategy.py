import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from dynamic_sltp_params import dynamic_sltp_engine
from strategy_base import Strategy
from volatility_filter import volatility_filter


class LiquiditySweepStrategy(Strategy):
    """
    Liquidity sweep reversal:
    - Long when price sweeps below recent low and closes back above it.
    - Short when price sweeps above recent high and closes back below it.
    """

    def __init__(self) -> None:
        cfg = CONFIG.get("LIQUIDITY_SWEEP", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.lookback = int(cfg.get("lookback", 20))
        self.atr_window = int(cfg.get("atr_window", 20))
        self.sweep_buffer_atr = float(cfg.get("sweep_buffer_atr", 0.10))
        self.reclaim_buffer_atr = float(cfg.get("reclaim_buffer_atr", 0.05))
        self.min_wick_atr = float(cfg.get("min_wick_atr", 0.20))
        self.volume_mult = float(cfg.get("volume_mult", 1.1))
        self.sessions = set(cfg.get("sessions", ["NY_AM", "NY_PM", "LONDON", "ASIA"]))
        self.use_pivots = bool(cfg.get("use_pivots", True))
        self.pivot_window = int(cfg.get("pivot_window", 2))
        self.pivot_max_age = int(cfg.get("pivot_max_age", 80))
        self.pivot_fallback_to_lookback = bool(cfg.get("pivot_fallback_to_lookback", False))
        self.confirm_followthrough = bool(cfg.get("confirm_followthrough", True))
        self.confirm_bars = int(cfg.get("confirm_bars", 1))
        self.min_sweep_points = float(cfg.get("min_sweep_points", 0.5))
        self.min_reclaim_points = float(cfg.get("min_reclaim_points", 0.0))
        self.min_wick_points = float(cfg.get("min_wick_points", 0.25))
        self.max_bar_range_atr = float(cfg.get("max_bar_range_atr", 1.5))
        self.cooldown_bars = int(cfg.get("cooldown_bars", 5))
        self.require_new_pivot = bool(cfg.get("require_new_pivot", True))
        self.allowed_regimes = {str(r).lower() for r in cfg.get("allowed_regimes", ["low", "normal"])}
        self._cooldown = {"LONG": 0, "SHORT": 0}
        self._pending = None
        self._last_pivot_idx = {"LONG": None, "SHORT": None}

        logging.info(
            "LiquiditySweepStrategy initialized | lookback=%s sweep_buffer_atr=%.2f",
            self.lookback,
            self.sweep_buffer_atr,
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

    def _tick_cooldowns(self) -> None:
        for side in self._cooldown:
            if self._cooldown[side] > 0:
                self._cooldown[side] -= 1

    def _find_recent_pivot(self, df: pd.DataFrame, kind: str) -> Optional[tuple]:
        if not self.use_pivots or self.pivot_window <= 0:
            return None
        w = self.pivot_window
        if len(df) < (w * 2 + 2):
            return None
        last_idx = len(df) - 1 - w
        min_idx = max(w, last_idx - self.pivot_max_age + 1)
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        for idx in range(last_idx, min_idx - 1, -1):
            left = idx - w
            right = idx + w
            if kind == "high":
                if highs[idx] == np.max(highs[left : right + 1]):
                    return idx, float(highs[idx])
            else:
                if lows[idx] == np.min(lows[left : right + 1]):
                    return idx, float(lows[idx])
        return None

    def _build_signal(self, side: str, df: pd.DataFrame) -> Dict:
        strategy_key = "LiquiditySweep_LONG" if side == "LONG" else "LiquiditySweep_SHORT"
        sltp = dynamic_sltp_engine.calculate_sltp(strategy_key, df)
        dynamic_sltp_engine.log_params(sltp, strategy_key)
        return {
            "strategy": "LiquiditySweep",
            "side": side,
            "tp_dist": sltp["tp_dist"],
            "sl_dist": sltp["sl_dist"],
        }

    def _confirm_followthrough(self, side: str, level: float, reclaim_buffer: float,
                               curr: pd.Series, prev: pd.Series) -> bool:
        if side == "SHORT":
            return curr["close"] < level - reclaim_buffer and curr["close"] < prev["close"]
        return curr["close"] > level + reclaim_buffer and curr["close"] > prev["close"]

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if not self.enabled:
            return None
        if df is None or len(df) < self.lookback + 2:
            return None

        ts = df.index[-1]
        session = volatility_filter.get_session(ts.hour)
        if self.sessions and session not in self.sessions:
            return None
        self._tick_cooldowns()

        regime_key = None
        if self.allowed_regimes:
            regime, _, _ = volatility_filter.get_regime(df, ts)
            regime_key = str(regime).lower()
            if regime_key not in self.allowed_regimes:
                return None

        curr = df.iloc[-1]
        prev = df.iloc[-2]
        prev_window = df.iloc[-(self.lookback + 1) : -1]
        if prev_window.empty:
            return None

        atr = self._calc_atr(df)
        if atr is None or atr <= 0:
            return None

        sweep_buffer = max(atr * self.sweep_buffer_atr, self.min_sweep_points)
        reclaim_buffer = max(atr * self.reclaim_buffer_atr, self.min_reclaim_points)
        min_wick = max(atr * self.min_wick_atr, self.min_wick_points)

        volume_ok = True
        if "volume" in df.columns and self.volume_mult > 0:
            avg_vol = float(prev_window["volume"].mean())
            volume_ok = avg_vol > 0 and float(curr["volume"]) >= avg_vol * self.volume_mult

        high = float(curr["high"])
        low = float(curr["low"])
        close = float(curr["close"])
        open_ = float(curr["open"])
        if self.max_bar_range_atr > 0 and (high - low) > atr * self.max_bar_range_atr:
            return None

        upper_wick = high - max(close, open_)
        lower_wick = min(close, open_) - low

        curr_idx = len(df) - 1
        if self._pending:
            pending = self._pending
            if pending["session"] != session or (regime_key and pending["regime"] != regime_key):
                self._pending = None
            elif curr_idx - pending["created_idx"] > self.confirm_bars:
                self._pending = None
            elif self._confirm_followthrough(
                pending["side"], pending["level"], pending["reclaim_buffer"], curr, prev
            ):
                side = pending["side"]
                if self._cooldown[side] == 0:
                    logging.info("LiquiditySweep: %s confirmed at %.2f", side, pending["level"])
                    self._cooldown[side] = self.cooldown_bars
                    self._last_pivot_idx[side] = pending["pivot_idx"]
                    self._pending = None
                    return self._build_signal(side, df)
            else:
                return None

        pivot_high = self._find_recent_pivot(df, "high")
        pivot_low = self._find_recent_pivot(df, "low")
        prev_high = None
        prev_low = None
        pivot_high_idx = None
        pivot_low_idx = None
        if pivot_high:
            pivot_high_idx, prev_high = pivot_high
        if pivot_low:
            pivot_low_idx, prev_low = pivot_low
        if prev_high is None and self.pivot_fallback_to_lookback:
            prev_high = float(prev_window["high"].max())
        if prev_low is None and self.pivot_fallback_to_lookback:
            prev_low = float(prev_window["low"].min())
        if self.require_new_pivot:
            if prev_high is not None and self._last_pivot_idx["SHORT"] is not None:
                if pivot_high_idx is not None and pivot_high_idx <= self._last_pivot_idx["SHORT"]:
                    prev_high = None
            if prev_low is not None and self._last_pivot_idx["LONG"] is not None:
                if pivot_low_idx is not None and pivot_low_idx <= self._last_pivot_idx["LONG"]:
                    prev_low = None

        # Short: sweep above prior high, close back below prior high (rejection)
        if (
            prev_high is not None
            and self._cooldown["SHORT"] == 0
            and volume_ok
            and high >= prev_high + sweep_buffer
            and close < prev_high - reclaim_buffer
            and upper_wick >= min_wick
        ):
            logging.info("LiquiditySweep: SHORT sweep above %.2f (close %.2f)", prev_high, close)
            if self.confirm_followthrough:
                self._pending = {
                    "side": "SHORT",
                    "level": prev_high,
                    "pivot_idx": pivot_high_idx,
                    "created_idx": curr_idx,
                    "reclaim_buffer": reclaim_buffer,
                    "session": session,
                    "regime": regime_key,
                }
                return None
            self._cooldown["SHORT"] = self.cooldown_bars
            self._last_pivot_idx["SHORT"] = pivot_high_idx
            return self._build_signal("SHORT", df)

        # Long: sweep below prior low, close back above prior low (rejection)
        if (
            prev_low is not None
            and self._cooldown["LONG"] == 0
            and volume_ok
            and low <= prev_low - sweep_buffer
            and close > prev_low + reclaim_buffer
            and lower_wick >= min_wick
        ):
            logging.info("LiquiditySweep: LONG sweep below %.2f (close %.2f)", prev_low, close)
            if self.confirm_followthrough:
                self._pending = {
                    "side": "LONG",
                    "level": prev_low,
                    "pivot_idx": pivot_low_idx,
                    "created_idx": curr_idx,
                    "reclaim_buffer": reclaim_buffer,
                    "session": session,
                    "regime": regime_key,
                }
                return None
            self._cooldown["LONG"] = self.cooldown_bars
            self._last_pivot_idx["LONG"] = pivot_low_idx
            return self._build_signal("LONG", df)

        return None
