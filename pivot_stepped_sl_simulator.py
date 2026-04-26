"""Stepped-SL Pivot Trail simulator.

This is the missing piece called out repeatedly in
docs/STRATEGY_ARCHITECTURE_JOURNAL.md sections 8.20.5, 8.26.1 (Pivot DEFERRED):
the entry-only simulator could not honestly attribute Pivot Trail because
Pivot's SL ratchet is **bar-stepwise** — it watches every bar after entry,
detects confirmed swing pivots, and ratchets the stop to a bank-level anchor.

This module ports the production rule from julie001.py and walks bars forward
applying it. It re-uses the contract pinning + trade-through TP rule from
``simulator_trade_through.py`` so the stepped-SL path is consistent with the
fixed entry-only path (no merge bug, no any-touch TP).

Production constants (julie001.py:189-207, 416-479):
    _BANK_FILL_STEP        = 12.5    # ES bank-level grid (points)
    _PIVOT_TRAIL_LOOKBACK  = 5       # bars in swing-detection window
    _PIVOT_TRAIL_BUFFER    = 0.25    # 1 tick clearance beyond anchor
    _PIVOT_TRAIL_MIN_PROFIT_PTS = 12.5  # unrealised PnL gate before trail arms
    US session: ET hours 9-15 inclusive (frozenset {9,10,11,12,13,14,15})
    Eligible strategies: DynamicEngine3, AetherFlow, RegimeAdaptive

Confirmed-pivot detection: a 5-bar window where the **middle** bar (index 2)
is the extreme high (LONG candidate anchor) or extreme low (SHORT). The middle
bar is at chronological position ``len(window) - 3`` (i.e. confirmed by
2 subsequent bars).

SL update rule (LONG; mirror for SHORT):
    profit_pts = pivot_price - entry_price
    if profit_pts < 12.5: skip
    anchor_C = floor(pivot_price / 12.5) * 12.5      # at-or-below pivot
    anchor_B = anchor_C - 12.5                        # one level back
    candidate_B = anchor_B - 0.25
    if candidate_B > entry: candidate = candidate_B  (Reading B)
    else:                   candidate = anchor_C - 0.25  (Reading C fallback)
    if candidate <= current_sl or candidate <= entry: skip
    sl = candidate (ratcheted)

Bar resolution: julie001 calls the trail with ``new_df`` which is the 1-minute
master DataFrame. So this simulator walks 1-minute bars (NOT 5-minute).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from simulator_trade_through import (
    ES_TICK,
    front_month_by_calendar,
    get_walk_forward_bars,
    pin_contract,
)

# ---- Production constants (mirror julie001.py) ----
BANK_FILL_STEP: float = 12.5
PIVOT_TRAIL_LOOKBACK: int = 5
PIVOT_TRAIL_BUFFER: float = 0.25
PIVOT_TRAIL_MIN_PROFIT_PTS: float = 12.5
US_SESSION_HOURS_ET: frozenset = frozenset({9, 10, 11, 12, 13, 14, 15})


# ---- Pivot detection (port of julie001._detect_pivot_high/low) ----
def detect_pivot_high(highs: list[float], lookback: int = PIVOT_TRAIL_LOOKBACK) -> Optional[float]:
    """Return the confirmed swing-high price from the middle of the last
    ``lookback`` bars, or None. Mirrors julie001._detect_pivot_high."""
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


def detect_pivot_low(lows: list[float], lookback: int = PIVOT_TRAIL_LOOKBACK) -> Optional[float]:
    """Return the confirmed swing-low price from the middle of the last
    ``lookback`` bars, or None. Mirrors julie001._detect_pivot_low."""
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


# ---- Pivot SL candidate (port of julie001._compute_pivot_trail_sl) ----
def compute_pivot_trail_sl(
    side: str,
    pivot_price: float,
    entry_price: float,
    current_sl: float,
    min_profit_pts: float = PIVOT_TRAIL_MIN_PROFIT_PTS,
    step: float = BANK_FILL_STEP,
    buffer: float = PIVOT_TRAIL_BUFFER,
) -> Optional[float]:
    """Compute trailing SL candidate. Mirrors julie001._compute_pivot_trail_sl.
    Returns the new SL price if it is a favorable ratchet locking real profit;
    otherwise None."""
    side_u = side.upper()
    if side_u == "LONG":
        profit_pts = pivot_price - entry_price
        if profit_pts < min_profit_pts - 1e-9:
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
        profit_pts = entry_price - pivot_price
        if profit_pts < min_profit_pts - 1e-9:
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


# ---- Stepped-SL outcome ----
@dataclass
class PivotSimOutcome:
    exit_reason: str       # 'take' | 'stop' | 'stop_pessimistic' | 'horizon' | 'no_data'
    exit_price: float
    exit_bar: int          # 0-based index into walk-forward bars; -1 if no_data
    entry_price: float
    pnl_points: float
    sl_path: list = field(default_factory=list)   # [(bar_idx, new_sl, pivot_price), ...]
    pivot_armed: bool = False
    armed_at_bar: int = -1
    n_bars_walked: int = 0
    final_sl: float = float("nan")
    max_mfe_pts: float = 0.0


def _is_us_session(ts: pd.Timestamp) -> bool:
    """Match julie001 gate: ET hour in {9..15}. ts must be tz-aware."""
    if ts.tzinfo is None:
        return False
    et = ts.tz_convert("US/Eastern") if str(ts.tzinfo) != "US/Eastern" else ts
    return et.hour in US_SESSION_HOURS_ET


def simulate_with_pivot_trail(
    bars: pd.DataFrame,
    side: str,
    entry_price: float,
    initial_sl: float,
    initial_tp: float,
    pivot_active: bool = True,
    lookback: int = PIVOT_TRAIL_LOOKBACK,
    tick_size: float = ES_TICK,
    us_session_only: bool = True,
) -> PivotSimOutcome:
    """Bar-stepwise walk-forward simulator with optional Pivot Trail SL ratchet.

    ``bars`` is the walk-forward 1-minute slice (already pinned to the right
    contract, strictly after entry). ``initial_sl`` and ``initial_tp`` are
    absolute price levels.

    For each bar:
      1) Update running MFE.
      2) (If pivot_active and US session) detect a confirmed swing in the most
         recent ``lookback`` bars and, when found, ratchet the SL via
         compute_pivot_trail_sl().
      3) Check SL hit (any-touch — wicks count). If yes, exit at SL.
      4) Check TP via trade-through rule (high >= TP+tick AND (close >= TP-tick
         OR next-bar high >= TP)). If yes, exit at TP.
      5) Tie-break (both in same bar): SL wins (pessimistic).

    On end of horizon, exit at last bar's close.
    """
    if bars is None or len(bars) == 0:
        return PivotSimOutcome("no_data", float(entry_price), -1,
                                float(entry_price), 0.0, [], False, -1, 0,
                                float(initial_sl), 0.0)

    side_u = side.upper()
    bars_idx = bars.reset_index(drop=False)
    n = len(bars_idx)

    sl = float(initial_sl)
    tp = float(initial_tp)
    pivot_armed = False
    armed_at_bar = -1
    sl_path = []
    max_mfe = 0.0

    # Rolling 5-bar window of bar OHLC for pivot detection.
    win_highs: list[float] = []
    win_lows: list[float] = []

    for j in range(n):
        bar = bars_idx.iloc[j]
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        ts_col = bar.get("timestamp", None)
        if ts_col is None:
            ts_col = bars_idx.iloc[j].iloc[0]   # first column is the reset index
        bar_ts = pd.Timestamp(ts_col)

        # Update MFE
        if side_u == "LONG":
            mfe = max(0.0, h - entry_price)
        else:
            mfe = max(0.0, entry_price - l)
        if mfe > max_mfe:
            max_mfe = mfe

        # Pivot Trail step — operate BEFORE checking SL/TP for this bar so that
        # the ratcheted SL applies on the next bar (julie001 mirrors this:
        # detect_pivot looks at last 5 bars, anchored at the middle bar which
        # is 2 bars old; it then sets SL for the *next* bar's risk check).
        win_highs.append(h)
        win_lows.append(l)
        if len(win_highs) > lookback:
            win_highs.pop(0)
            win_lows.pop(0)

        if pivot_active and (not us_session_only or _is_us_session(bar_ts)):
            if len(win_highs) == lookback:
                pivot_high = detect_pivot_high(win_highs, lookback) if side_u == "LONG" else None
                pivot_low = detect_pivot_low(win_lows, lookback) if side_u == "SHORT" else None
                pivot_price = pivot_high if side_u == "LONG" else pivot_low
                if pivot_price is not None:
                    candidate = compute_pivot_trail_sl(
                        side_u, float(pivot_price), entry_price, sl
                    )
                    if candidate is not None:
                        sl = candidate
                        if not pivot_armed:
                            pivot_armed = True
                            armed_at_bar = j
                        sl_path.append((j, sl, float(pivot_price)))

        # Now check SL / TP using the (possibly ratcheted) SL.
        next_bar = bars_idx.iloc[j + 1] if j + 1 < n else None
        if side_u == "LONG":
            sl_hit = math.isfinite(sl) and l <= sl
            tp_touch = math.isfinite(tp) and h >= (tp + tick_size)
            tp_confirm = False
            if tp_touch:
                close_held = c >= (tp - tick_size)
                next_through = (
                    next_bar is not None and float(next_bar["high"]) >= tp
                )
                tp_confirm = close_held or next_through
            tp_hit = tp_touch and tp_confirm
        else:  # SHORT
            sl_hit = math.isfinite(sl) and h >= sl
            tp_touch = math.isfinite(tp) and l <= (tp - tick_size)
            tp_confirm = False
            if tp_touch:
                close_held = c <= (tp + tick_size)
                next_through = (
                    next_bar is not None and float(next_bar["low"]) <= tp
                )
                tp_confirm = close_held or next_through
            tp_hit = tp_touch and tp_confirm

        if sl_hit and tp_hit:
            pnl = (sl - entry_price) if side_u == "LONG" else (entry_price - sl)
            return PivotSimOutcome(
                "stop_pessimistic", float(sl), j, float(entry_price), float(pnl),
                sl_path, pivot_armed, armed_at_bar, j + 1, float(sl), max_mfe,
            )
        if sl_hit:
            pnl = (sl - entry_price) if side_u == "LONG" else (entry_price - sl)
            return PivotSimOutcome(
                "stop", float(sl), j, float(entry_price), float(pnl),
                sl_path, pivot_armed, armed_at_bar, j + 1, float(sl), max_mfe,
            )
        if tp_hit:
            pnl = (tp - entry_price) if side_u == "LONG" else (entry_price - tp)
            return PivotSimOutcome(
                "take", float(tp), j, float(entry_price), float(pnl),
                sl_path, pivot_armed, armed_at_bar, j + 1, float(sl), max_mfe,
            )

    # Horizon exit
    last = bars_idx.iloc[n - 1]
    exit_price = float(last["close"])
    pnl = (exit_price - entry_price) if side_u == "LONG" else (entry_price - exit_price)
    return PivotSimOutcome(
        "horizon", float(exit_price), n - 1, float(entry_price), float(pnl),
        sl_path, pivot_armed, armed_at_bar, n, float(sl), max_mfe,
    )


def simulate_trade_with_pivot_trail(
    parquet_df: pd.DataFrame,
    entry_ts,
    side: str,
    entry_price: float,
    sl_distance: float,
    tp_distance: float,
    contract: Optional[str] = None,
    horizon_bars: int = 30,
    pivot_active: bool = True,
    multiplier: float = 5.0,
    fee: float = 7.50,
    us_session_only: bool = True,
) -> dict:
    """High-level wrapper: pin contract → pull walk-forward 1-min bars →
    run stepped-SL walk → return dict with PnL.

    sl_distance / tp_distance are point distances (positive). Initial SL
    price = entry - sl_distance (LONG) or entry + sl_distance (SHORT). TP
    price = entry + tp_distance (LONG) / entry - tp_distance (SHORT).

    PnL: raw_pnl_dollars = pnl_points * multiplier; net_pnl = raw - fee.
    multiplier=5 = MES; the bot trades MES per Phase 1.
    """
    ts = pd.Timestamp(entry_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    if contract is None:
        contract = pin_contract(parquet_df, ts, entry_price)
    bars = get_walk_forward_bars(parquet_df, ts, contract, horizon_bars=horizon_bars)
    side_u = side.upper()
    if side_u == "LONG":
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance

    outcome = simulate_with_pivot_trail(
        bars,
        side=side_u,
        entry_price=entry_price,
        initial_sl=sl_price,
        initial_tp=tp_price,
        pivot_active=pivot_active,
        us_session_only=us_session_only,
    )
    raw_pnl = outcome.pnl_points * multiplier
    net_pnl = raw_pnl - fee
    return {
        "contract": contract,
        "exit_reason": outcome.exit_reason,
        "exit_price": outcome.exit_price,
        "exit_bar": outcome.exit_bar,
        "entry_price": outcome.entry_price,
        "pnl_points": outcome.pnl_points,
        "raw_pnl": raw_pnl,
        "net_pnl_after_haircut": net_pnl,
        "pivot_armed": outcome.pivot_armed,
        "armed_at_bar": outcome.armed_at_bar,
        "final_sl": outcome.final_sl,
        "max_mfe_pts": outcome.max_mfe_pts,
        "sl_path": outcome.sl_path,
        "n_walk_bars": outcome.n_bars_walked,
    }
