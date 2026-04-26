"""Pure-logic replay function for early-exit hyperparameter sweep.

Given a row from artifacts/v11_corpus_with_bar_paths.parquet (which has a
30-bar OHLC trajectory pre-computed), apply a parameterized early-exit policy
and return the resulting trade outcome. No I/O, no contract pinning — this is
called millions of times in the §8.30 sweep.

Mechanics implemented:
  - BE-arm with configurable threshold (MFE pts) AND offset
      (`be_offset` is in points above entry where SL is moved on arm).
      `be_threshold=None` disables BE-arm entirely.
  - Pivot Trail (julie001 5-bar swing pivot, bank-anchor compute) with
      configurable `pivot_lookback` and `pivot_confirm_bars` (additional
      hold-bars before the pivot ratchet fires; 0 = immediate, the live default).
      `pivot_lookback=None` disables pivot trail.
  - Close-on-reverse hypothetical: 'never', 'always', 'confirmed', 'mfe_gate'.
      'never'   = no early close on opposite-side candidate (matches live)
      'always'  = if any opposite-side candidate fires within horizon, close
                  at the candidate's entry price ts on that bar's close
      'confirmed' = same as 'always' but only if 1 full bar has passed since
                  entry
      'mfe_gate' = same as 'always' but only if MFE so far >= threshold
                  (used to lock real profit before reversing)

The reverse logic needs the chronological candidate stream (we look up the
next opposite-side candidate within the trade's horizon window). The replay
takes a `next_opposite_ts` argument computed once before the sweep.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

# Live julie001 constants
TICK_SIZE: float = 0.25
ES_TICK = TICK_SIZE
BANK_FILL_STEP: float = 12.5
PIVOT_TRAIL_BUFFER: float = 0.25
PIVOT_TRAIL_MIN_PROFIT_PTS: float = 12.5
US_SESSION_HOURS_ET: frozenset = frozenset({9, 10, 11, 12, 13, 14, 15})

POINT_VALUE: float = 5.0   # MES
HAIRCUT: float = 7.50


# ---- Pivot detection (port of julie001._detect_pivot_high/low) ----
def detect_pivot_high(highs: list[float], lookback: int) -> Optional[float]:
    half = lookback // 2
    if len(highs) < lookback:
        return None
    window = highs[-lookback:]
    ph = window[half]
    for i, v in enumerate(window):
        if i == half:
            continue
        if v > ph + 1e-9:
            return None
    return float(ph)


def detect_pivot_low(lows: list[float], lookback: int) -> Optional[float]:
    half = lookback // 2
    if len(lows) < lookback:
        return None
    window = lows[-lookback:]
    pl = window[half]
    for i, v in enumerate(window):
        if i == half:
            continue
        if v < pl - 1e-9:
            return None
    return float(pl)


def compute_pivot_trail_sl(side: str, pivot_price: float, entry_price: float,
                            current_sl: float,
                            min_profit_pts: float = PIVOT_TRAIL_MIN_PROFIT_PTS,
                            step: float = BANK_FILL_STEP,
                            buffer: float = PIVOT_TRAIL_BUFFER) -> Optional[float]:
    side_u = side.upper()
    if side_u == "LONG":
        if pivot_price - entry_price < min_profit_pts - 1e-9:
            return None
        anchor_c = math.floor(pivot_price / step) * step
        anchor_b = anchor_c - step
        candidate_b = round(anchor_b - buffer, 4)
        if candidate_b > entry_price + 1e-9:
            candidate = candidate_b
        else:
            candidate = round(anchor_c - buffer, 4)
        if candidate <= current_sl + 1e-9 or candidate <= entry_price + 1e-9:
            return None
        return float(candidate)
    if side_u == "SHORT":
        if entry_price - pivot_price < min_profit_pts - 1e-9:
            return None
        anchor_c = math.ceil(pivot_price / step) * step
        anchor_b = anchor_c + step
        candidate_b = round(anchor_b + buffer, 4)
        if candidate_b < entry_price - 1e-9:
            candidate = candidate_b
        else:
            candidate = round(anchor_c + buffer, 4)
        if candidate >= current_sl - 1e-9 or candidate >= entry_price - 1e-9:
            return None
        return float(candidate)
    return None


def _is_us_session_ts(ts: pd.Timestamp) -> bool:
    if ts.tzinfo is None:
        return False
    et = ts.tz_convert("US/Eastern") if str(ts.tzinfo) != "US/Eastern" else ts
    return et.hour in US_SESSION_HOURS_ET


@dataclass
class ReplayResult:
    exit_reason: str   # 'take' | 'stop' | 'stop_pessimistic' | 'horizon' | 'reverse_close' | 'no_data'
    exit_price: float
    exit_bar: int
    pnl_points: float
    raw_pnl: float
    net_pnl: float
    be_armed: bool
    pivot_armed: bool
    final_sl: float
    mfe_points: float
    mae_points: float


def replay_with_early_exit_config(
    bar_path: list[dict],
    side: str,
    entry_price: float,
    initial_sl_price: float,
    initial_tp_price: float,
    *,
    be_threshold: Optional[float],         # MFE pts; None disables BE-arm
    be_offset: float,                       # SL = entry + be_offset on arm (LONG)
    pivot_lookback: Optional[int],         # None disables pivot trail
    pivot_confirm_bars: int,                # 0 = immediate, N = require N more bars
    close_reverse_policy: str,              # 'never' | 'always' | 'confirmed' | 'mfe_gate'
    close_reverse_mfe_threshold: float = 0.0,
    next_opposite_bar_idx: Optional[int] = None,  # bar index in bar_path of next opposite-side fire
    pivot_us_session_only: bool = True,
    fee: float = HAIRCUT,
    point_value: float = POINT_VALUE,
    tick_size: float = TICK_SIZE,
) -> ReplayResult:
    """Replay one trade with given early-exit params on its bar_path."""
    if not bar_path:
        return ReplayResult("no_data", float(entry_price), -1, 0.0,
                              -fee, -fee, False, False, float(initial_sl_price),
                              0.0, 0.0)
    side_u = side.upper()
    sl = float(initial_sl_price)
    tp = float(initial_tp_price)

    mfe = 0.0
    mae = 0.0
    be_armed = False
    pivot_armed = False
    win_highs: list[float] = []
    win_lows: list[float] = []
    n = len(bar_path)
    pending_pivot_sl: Optional[tuple[int, float]] = None  # (bar_idx_when_armed, candidate_sl)

    for j, bar in enumerate(bar_path):
        h = bar["high"]
        l = bar["low"]
        c = bar["close"]
        bar_ts = pd.Timestamp(bar["ts"])
        if bar_ts.tzinfo is None:
            bar_ts = bar_ts.tz_localize("US/Eastern")

        # 1. Update MFE/MAE BEFORE exit checks
        if side_u == "LONG":
            mfe = max(mfe, h - entry_price)
            mae = max(mae, entry_price - l)
        else:
            mfe = max(mfe, entry_price - l)
            mae = max(mae, h - entry_price)

        # 2. BE-arm (if enabled and threshold crossed)
        if (be_threshold is not None) and (not be_armed) and (mfe >= be_threshold - 1e-9):
            if side_u == "LONG":
                new_sl = float(entry_price) + float(be_offset)
                if new_sl > sl + 1e-9:
                    sl = new_sl
            else:
                new_sl = float(entry_price) - float(be_offset)
                if new_sl < sl - 1e-9:
                    sl = new_sl
            be_armed = True

        # 3. Pivot Trail
        win_highs.append(h)
        win_lows.append(l)
        if pivot_lookback is not None and len(win_highs) > pivot_lookback:
            win_highs.pop(0)
            win_lows.pop(0)

        # Resolve any pending pivot SL whose confirm-bars window has elapsed.
        if pending_pivot_sl is not None and pivot_lookback is not None:
            armed_idx, candidate = pending_pivot_sl
            if j - armed_idx >= pivot_confirm_bars:
                # Recheck ratchet condition (SL must still be improved).
                better = (candidate > sl + 1e-9) if side_u == "LONG" else (candidate < sl - 1e-9)
                if better:
                    sl = candidate
                    pivot_armed = True
                pending_pivot_sl = None

        if pivot_lookback is not None and (not pivot_us_session_only or _is_us_session_ts(bar_ts)):
            if len(win_highs) == pivot_lookback:
                pivot_price = (
                    detect_pivot_high(win_highs, pivot_lookback)
                    if side_u == "LONG" else detect_pivot_low(win_lows, pivot_lookback)
                )
                if pivot_price is not None:
                    candidate = compute_pivot_trail_sl(side_u, float(pivot_price),
                                                       entry_price, sl)
                    if candidate is not None:
                        if pivot_confirm_bars <= 0:
                            sl = candidate
                            pivot_armed = True
                        else:
                            # Stage; resolved on bar j + pivot_confirm_bars
                            pending_pivot_sl = (j, candidate)

        # 4. Close-on-reverse check (BEFORE SL/TP exit checks). If the next
        #    opposite-side candidate fires this bar AND the policy permits,
        #    we exit at the bar's CLOSE.
        if (close_reverse_policy != "never"
                and next_opposite_bar_idx is not None
                and j == int(next_opposite_bar_idx)):
            allow = False
            if close_reverse_policy == "always":
                allow = True
            elif close_reverse_policy == "confirmed":
                allow = (j >= 1)
            elif close_reverse_policy == "mfe_gate":
                allow = (mfe >= close_reverse_mfe_threshold - 1e-9)
            if allow:
                pnl_pts = (c - entry_price) if side_u == "LONG" else (entry_price - c)
                raw = pnl_pts * point_value
                return ReplayResult("reverse_close", float(c), j, float(pnl_pts),
                                      float(raw), float(raw - fee),
                                      be_armed, pivot_armed, float(sl),
                                      float(mfe), float(mae))

        # 5. SL/TP exit checks (using possibly-updated SL).
        next_bar = bar_path[j + 1] if j + 1 < n else None
        if side_u == "LONG":
            sl_hit = math.isfinite(sl) and l <= sl
            tp_touch = math.isfinite(tp) and h >= (tp + tick_size)
            tp_confirm = False
            if tp_touch:
                close_held = c >= (tp - tick_size)
                next_through = next_bar is not None and float(next_bar["high"]) >= tp
                tp_confirm = close_held or next_through
            tp_hit = tp_touch and tp_confirm
        else:
            sl_hit = math.isfinite(sl) and h >= sl
            tp_touch = math.isfinite(tp) and l <= (tp - tick_size)
            tp_confirm = False
            if tp_touch:
                close_held = c <= (tp + tick_size)
                next_through = next_bar is not None and float(next_bar["low"]) <= tp
                tp_confirm = close_held or next_through
            tp_hit = tp_touch and tp_confirm

        if sl_hit and tp_hit:
            pnl_pts = (sl - entry_price) if side_u == "LONG" else (entry_price - sl)
            raw = pnl_pts * point_value
            return ReplayResult("stop_pessimistic", float(sl), j,
                                  float(pnl_pts), float(raw), float(raw - fee),
                                  be_armed, pivot_armed, float(sl),
                                  float(mfe), float(mae))
        if sl_hit:
            pnl_pts = (sl - entry_price) if side_u == "LONG" else (entry_price - sl)
            raw = pnl_pts * point_value
            return ReplayResult("stop", float(sl), j, float(pnl_pts),
                                  float(raw), float(raw - fee),
                                  be_armed, pivot_armed, float(sl),
                                  float(mfe), float(mae))
        if tp_hit:
            pnl_pts = (tp - entry_price) if side_u == "LONG" else (entry_price - tp)
            raw = pnl_pts * point_value
            return ReplayResult("take", float(tp), j, float(pnl_pts),
                                  float(raw), float(raw - fee),
                                  be_armed, pivot_armed, float(sl),
                                  float(mfe), float(mae))

    # Horizon exit
    last = bar_path[-1]
    exit_price = float(last["close"])
    pnl_pts = (exit_price - entry_price) if side_u == "LONG" else (entry_price - exit_price)
    raw = pnl_pts * point_value
    return ReplayResult("horizon", float(exit_price), n - 1, float(pnl_pts),
                          float(raw), float(raw - fee),
                          be_armed, pivot_armed, float(sl),
                          float(mfe), float(mae))


def parse_bar_path(s: str) -> list[dict]:
    """Decode the bar_path_json column. Lightweight wrapper."""
    if not s or pd.isna(s):
        return []
    return json.loads(s)
