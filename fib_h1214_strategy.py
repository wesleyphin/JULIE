"""Fib h1214 strategy — production module.

Strategy: 7-level Fibonacci retracement with 8-bar lookback extremum filter.

Mechanics:
  AM swing window: 9:30-11:59 ET (computes am_high, am_low, swing_dir, swing_range)
  Entry window: 12:00-14:59 ET only
  Skip days where swing_range < 15 pts

  Per 1-min bar in entry window, for each of 7 fib levels:
    1. Cooldown: bars-since-last-fire >= 30 for this level
    2. Touch: bar low ≤ fib_p ≤ bar high
    3. Counter-bar: close > prev_close (UP day) / close < prev_close (DOWN day)
    4. 8-bar Fibonacci-lookback extremum:
       LONG: this bar's low is the lowest of the last 8 bars
       SHORT: this bar's high is the highest of the last 8 bars

  Signal: LONG (UP day) / SHORT (DOWN day) — continuation with the swing
  Brackets: TP = 5 ticks, SL = 5 ticks

Adaptive sizing (Kelly-flavored):
  Track rolling 20-day strategy PnL (per-strategy persistence)
  size = 0 if rolling_20d_pnl < -$500   (PAUSE — strategy is bleeding)
  size = 1 if -$500 <= rolling_20d_pnl < +$500  (NORMAL)
  size = 2 if  $500 <= rolling_20d_pnl < +$1500 (CONFIDENT)
  size = 3 if           rolling_20d_pnl >= +$1500 (FULL)

Backtest (T2 2025-01-20 → 2026-04-24, sizing applied):
  n=384, WR=60.9%, +$6,744, DD=-$875, 2.01/day, $5,370/year
  Edge over random: +$6,069 (verified, 6.5σ on un-sized base)
"""
from __future__ import annotations
import logging
from collections import deque
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategy_base import Strategy

logger = logging.getLogger(__name__)

# Fib retracement ratios (price levels)
FIB_RATIOS = (
    ("236", 0.236),
    ("382", 0.382),
    ("500", 0.500),
    ("618", 0.618),
    ("705", 0.705),
    ("786", 0.786),
    ("1000", 1.000),
)

# Strategy parameters (hard-coded — tuned via 9-year ES backtest)
TP_TICKS = 5             # take profit in ticks (5 ticks = 1.25 pts = $62.50 on ES)
SL_TICKS = 5             # stop loss in ticks
TP_POINTS = TP_TICKS * 0.25
SL_POINTS = SL_TICKS * 0.25
COOLDOWN_BARS = 30       # bars-since-last-fire per level
MIN_SWING_RANGE = 15.0   # skip narrow-range AM days
FIB_LOOKBACK = 8         # 8-bar Fibonacci lookback (the actual Fib number)
ENTRY_HOUR_START = 12    # ET
ENTRY_HOUR_END = 15      # ET (exclusive)
AM_HOUR_START = 9
AM_MINUTE_START = 30
AM_HOUR_END = 12         # exclusive

# Adaptive sizing thresholds (rolling 20-day strategy PnL in dollars)
SIZE_THRESHOLDS = [
    (-500.0, 0),    # PAUSE if rolling 20d < -$500
    (500.0,  1),    # NORMAL otherwise (default, also covers below pause)
    (1500.0, 2),    # CONFIDENT if rolling 20d >= +$500
    (float("inf"), 3),  # FULL if rolling 20d >= +$1500
]
ADAPTIVE_SIZING_LOOKBACK_DAYS = 20

# ET tz
try:
    import zoneinfo
    NY_TZ = zoneinfo.ZoneInfo("US/Eastern")
except ImportError:
    import pytz  # fallback
    NY_TZ = pytz.timezone("US/Eastern")


def _adaptive_size(rolling_20d_pnl: float) -> int:
    """Map rolling 20-day strategy PnL to position size (0/1/2/3)."""
    if rolling_20d_pnl < -500.0:
        return 0
    if rolling_20d_pnl < 500.0:
        return 1
    if rolling_20d_pnl < 1500.0:
        return 2
    return 3


class FibH1214Strategy(Strategy):
    """Hour 12-14 ET Fibonacci retracement continuation strategy."""

    def __init__(self):
        super().__init__() if hasattr(Strategy, "__init__") else None
        self._last_processed_time: Optional[pd.Timestamp] = None
        self._am_swing_cache: Dict = {}  # date → swing dict (high, low, dir, fib_levels)
        self._last_fire_idx: Dict = {}    # (date, level) → bar_idx
        # Daily PnL history for adaptive sizing — populated by julie001 trade-realization hook
        self._daily_pnl_history: deque = deque(maxlen=60)  # list of (date, pnl_dollars)
        self._enabled: bool = True

    # === Public API: julie001 calls this on each bar tick ===
    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if df is None or len(df) < (FIB_LOOKBACK + 5):
            return None
        if not self._enabled:
            return None

        current_time = df.index[-1]
        if self._last_processed_time == current_time:
            return None
        self._last_processed_time = current_time

        # Convert to ET
        try:
            ts_et = current_time.tz_convert(NY_TZ) if current_time.tzinfo else current_time.tz_localize("UTC").tz_convert(NY_TZ)
        except Exception:
            return None

        date_et = ts_et.date()
        hour_et = ts_et.hour
        minute_et = ts_et.minute

        # Gate: only fire in 12:00-14:59 ET
        if hour_et < ENTRY_HOUR_START or hour_et >= ENTRY_HOUR_END:
            return None

        # Build/get AM swing for today
        swing = self._am_swing_cache.get(date_et)
        if swing is None:
            swing = self._compute_am_swing(df, date_et, ts_et)
            if swing is None:
                return None
            self._am_swing_cache[date_et] = swing

        if swing.get("invalid"):
            return None

        # Read latest bar OHLC
        cur = df.iloc[-1]
        prev = df.iloc[-2]
        cur_low = float(cur["low"]); cur_high = float(cur["high"])
        cur_close = float(cur["close"]); prev_close = float(prev["close"])

        sw_dir = swing["dir"]
        if sw_dir == 0:
            return None

        # Counter-bar: bar must close in swing direction
        if sw_dir > 0 and not (cur_close > prev_close):
            return None
        if sw_dir < 0 and not (cur_close < prev_close):
            return None

        # 8-bar lookback extremum (the Fibonacci-number filter)
        recent = df.iloc[-(FIB_LOOKBACK + 1):]  # last 8+1 bars including current
        if sw_dir > 0:
            if cur_low > float(recent["low"].min()):
                return None
        else:
            if cur_high < float(recent["high"].max()):
                return None

        # Iterate fib levels (deepest first — typically more likely to be the true exhaustion)
        for label, _ratio in FIB_RATIOS:
            fib_p = swing[f"fib_{label}"]
            if fib_p is None or pd.isna(fib_p):
                continue

            # Touch test
            if not (cur_low <= fib_p <= cur_high):
                continue

            # Cooldown
            key = (date_et, label)
            last_idx = self._last_fire_idx.get(key, -1_000_000)
            cur_idx = len(df) - 1
            if cur_idx - last_idx < COOLDOWN_BARS:
                continue

            # Build signal
            side = "LONG" if sw_dir > 0 else "SHORT"
            adaptive_size = _adaptive_size(self._rolling_20d_pnl())
            if adaptive_size <= 0:
                # Strategy paused due to recent drawdown
                logger.info("[FibH1214] PAUSED — rolling 20d PnL=$%.0f below -$500 floor", self._rolling_20d_pnl())
                return None

            self._last_fire_idx[key] = cur_idx

            signal = {
                "strategy": f"FibH1214_fib_{label}",
                "side": side,
                "tp_dist": TP_POINTS,
                "sl_dist": SL_POINTS,
                "size": adaptive_size,
                "fib_level": label,
                "swing_dir": int(sw_dir),
                "swing_range": float(swing["range"]),
                "rolling_20d_pnl": float(self._rolling_20d_pnl()),
                "current_time": current_time.isoformat() if hasattr(current_time, "isoformat") else str(current_time),
            }
            logger.info(
                "[FibH1214] FIRE side=%s fib=%s size=%d swing_dir=%+d rolling_20d=$%.0f",
                side, label, adaptive_size, int(sw_dir), self._rolling_20d_pnl(),
            )
            return signal

        return None

    # === AM swing computation (9:30-11:59 ET) ===
    def _compute_am_swing(self, df: pd.DataFrame, date_et, ts_et) -> Optional[Dict]:
        """Compute AM swing from 9:30-11:59 ET bars for the current day."""
        try:
            df_et = df.tz_convert(NY_TZ) if df.index.tz else df.tz_localize("UTC").tz_convert(NY_TZ)
        except Exception:
            return None

        am_mask = (
            (df_et.index.date == date_et)
            & (
                ((df_et.index.hour == AM_HOUR_START) & (df_et.index.minute >= AM_MINUTE_START))
                | (df_et.index.hour > AM_HOUR_START)
            )
            & (df_et.index.hour < AM_HOUR_END)
        )
        am = df_et[am_mask]
        if len(am) < 30:
            return {"invalid": True}

        am_high = float(am["high"].max())
        am_low = float(am["low"].min())
        am_open = float(am["open"].iloc[0])
        am_close = float(am["close"].iloc[-1])
        swing_range = am_high - am_low

        if swing_range < MIN_SWING_RANGE:
            return {"invalid": True}

        sw_dir = float(np.sign(am_close - am_open))

        result = {
            "high": am_high,
            "low": am_low,
            "range": swing_range,
            "dir": sw_dir,
            "open": am_open,
            "close": am_close,
            "invalid": False,
        }
        # Build fib levels
        for label, ratio in FIB_RATIOS:
            if sw_dir >= 0:
                fib_p = am_high - ratio * swing_range
            else:
                fib_p = am_low + ratio * swing_range
            result[f"fib_{label}"] = fib_p
        return result

    # === Adaptive sizing helpers ===
    def _rolling_20d_pnl(self) -> float:
        """Return rolling 20-day strategy PnL from internal history."""
        if not self._daily_pnl_history:
            return 0.0
        # Keep only last 20 trading days
        recent = list(self._daily_pnl_history)[-ADAPTIVE_SIZING_LOOKBACK_DAYS:]
        return float(sum(p for _, p in recent))

    def record_trade_pnl(self, trade_date, pnl_dollars: float) -> None:
        """Called by julie001 when a Fib h1214 trade closes. Updates rolling history."""
        # Aggregate by date
        if self._daily_pnl_history and self._daily_pnl_history[-1][0] == trade_date:
            d, prev_pnl = self._daily_pnl_history[-1]
            self._daily_pnl_history[-1] = (d, prev_pnl + pnl_dollars)
        else:
            self._daily_pnl_history.append((trade_date, pnl_dollars))

    def set_enabled(self, enabled: bool) -> None:
        """Toggle strategy on/off (e.g., from CLI/admin command)."""
        self._enabled = bool(enabled)
        logger.info("[FibH1214] enabled=%s", self._enabled)

    def state_for_persist(self) -> Dict:
        """Return state dict for runtime persistence (call from julie001 persist_runtime_state)."""
        return {
            "enabled": self._enabled,
            "daily_pnl_history": [(d.isoformat() if hasattr(d, "isoformat") else str(d), p)
                                  for d, p in self._daily_pnl_history],
        }

    def restore_state(self, state: Dict) -> None:
        """Restore state from persisted dict."""
        if not state:
            return
        self._enabled = bool(state.get("enabled", True))
        history = state.get("daily_pnl_history", [])
        self._daily_pnl_history = deque(maxlen=60)
        for d, p in history:
            try:
                d_obj = pd.Timestamp(d).date() if not hasattr(d, "year") else d
                self._daily_pnl_history.append((d_obj, float(p)))
            except Exception:
                continue
        logger.info("[FibH1214] state restored: %d daily PnL entries, rolling 20d=$%.0f",
                    len(self._daily_pnl_history), self._rolling_20d_pnl())
