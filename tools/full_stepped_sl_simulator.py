"""Full integrated stepped-SL simulator: BE-arm + Pivot Trail INSIDE the bar walk.

Phase 6 (docs/STRATEGY_ARCHITECTURE_JOURNAL.md §8.28). Earlier phases applied
BE-arm POST-HOC (at end of trade, look at MFE; if stop AND mfe>=10 -> raw_pnl=0).
That underestimates BE-arm because it can't model the case where BE-arm
triggers during the trade and SL=entry then HOLDS while price reverts to TP.

This module walks bars forward and applies, IN ORDER on each bar:
  1. Update MFE/MAE.
  2. BE-arm trigger: once mfe_points >= tp_dist * 0.40 (julie001 config:
     trade_management.break_even.trigger_pct=0.40 -> for DE3 TP=25pt,
     fires at MFE=10pt), set SL = entry_price (LONG/SHORT mirror). Mark armed.
  3. Pivot Trail (US session only, gated by julie001 hours 9..15 ET):
     detect a confirmed swing pivot in the last 5 bars and ratchet SL to
     floor(pivot/12.5)*12.5 - 12.5 - 0.25  (LONG; mirror for SHORT) when
     the candidate is favorable and beats current SL.
  4. SL touch (any-touch). Exit at SL price.
  5. TP via trade-through rule: high >= TP+tick AND (close >= TP-tick OR
     next-bar high >= TP). Exit at TP price.

If both SL and TP hit on the same bar -> SL wins (pessimistic tiebreak).

Production constants quoted from julie001.py:
  - line 3066: trigger_points = round_points_to_tick(max(TICK_SIZE, tp_dist * trigger_pct))
  - line 4678 / 6070: trigger_pct = 0.40  (DE3 break_even)
  - pivot_stepped_sl_simulator.py:57-61: BANK_FILL_STEP=12.5,
    PIVOT_TRAIL_LOOKBACK=5, PIVOT_TRAIL_BUFFER=0.25,
    PIVOT_TRAIL_MIN_PROFIT_PTS=12.5, US_SESSION_HOURS_ET={9..15}.

Both LONG and SHORT semantics handled symmetrically.
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
from pivot_stepped_sl_simulator import (
    BANK_FILL_STEP,
    PIVOT_TRAIL_BUFFER,
    PIVOT_TRAIL_LOOKBACK,
    PIVOT_TRAIL_MIN_PROFIT_PTS,
    US_SESSION_HOURS_ET,
    compute_pivot_trail_sl,
    detect_pivot_high,
    detect_pivot_low,
)

# BE-arm trigger fraction from julie001.py:4678 / config.py:6070.
BE_TRIGGER_PCT: float = 0.40


def _is_us_session(ts: pd.Timestamp) -> bool:
    """ET session gate: hour in {9..15}. Mirrors pivot_stepped_sl_simulator."""
    if ts.tzinfo is None:
        return False
    et = ts.tz_convert("US/Eastern") if str(ts.tzinfo) != "US/Eastern" else ts
    return et.hour in US_SESSION_HOURS_ET


@dataclass
class FullSteppedOutcome:
    """Outcome of the full stepped-SL walk.

    Same shape as TradeOutcome but extended:
      - be_armed:        BE-arm triggered during the trade
      - be_armed_at_bar: 0-based index of the bar where BE armed (-1 if not)
      - pivot_armed:     Pivot Trail ratchet fired at least once
      - pivot_armed_at_bar: 0-based index of FIRST pivot ratchet (-1 if not)
      - final_sl:        SL price at exit (after BE-arm + Pivot ratchets)
      - sl_path:         list of (bar_idx, new_sl, source) tuples — source in
                         {"be_arm","pivot_trail"}
    """
    exit_reason: str       # 'take' | 'stop' | 'stop_pessimistic' | 'horizon' | 'no_data'
    exit_price: float
    exit_bar: int          # 0-based; -1 if no_data
    entry_price: float
    pnl_points: float
    mfe_points: float = 0.0
    mae_points: float = 0.0
    be_armed: bool = False
    be_armed_at_bar: int = -1
    pivot_armed: bool = False
    pivot_armed_at_bar: int = -1
    final_sl: float = float("nan")
    sl_path: list = field(default_factory=list)


def simulate_full_stepped(
    bars: pd.DataFrame,
    side: str,
    entry_price: float,
    initial_sl: float,
    initial_tp: float,
    *,
    be_arm_active: bool = True,
    pivot_active: bool = True,
    be_trigger_pct: float = BE_TRIGGER_PCT,
    pivot_lookback: int = PIVOT_TRAIL_LOOKBACK,
    us_session_only: bool = True,
    tick_size: float = ES_TICK,
) -> FullSteppedOutcome:
    """Walk bars forward applying BE-arm + Pivot Trail INSIDE the loop.

    Returns FullSteppedOutcome. PnL is in POINTS (caller multiplies by $/pt).

    BE-arm logic:
      tp_dist     = abs(initial_tp - entry_price)
      be_trigger  = tp_dist * be_trigger_pct       (e.g. 25 * 0.40 = 10pt)
      Once mfe_points >= be_trigger and not yet armed:
        SL is moved to entry_price (LONG: SL up to entry; SHORT: SL down to
        entry). If new SL is worse than current SL (e.g. Pivot Trail already
        moved it past entry), no change.

    Pivot Trail logic (mirrors pivot_stepped_sl_simulator):
      In US session, after each bar (using a rolling 5-bar OHLC window):
        detect confirmed swing high/low at the middle bar (index 2). If
        present AND candidate SL via compute_pivot_trail_sl beats current SL,
        ratchet SL.

    Exit precedence on each bar:
      SL hit (any-touch) -> stop
      TP hit (trade-through) -> take
      Both -> stop_pessimistic
    """
    if bars is None or len(bars) == 0:
        return FullSteppedOutcome(
            "no_data", float(entry_price), -1, float(entry_price), 0.0, 0.0, 0.0,
            False, -1, False, -1, float(initial_sl), [],
        )

    side_u = str(side).upper()
    bars_idx = bars.reset_index(drop=False)
    n = len(bars_idx)

    sl = float(initial_sl)
    tp = float(initial_tp)
    tp_dist = abs(tp - entry_price)
    be_threshold = tp_dist * float(be_trigger_pct)

    mfe_points = 0.0
    mae_points = 0.0

    be_armed = False
    be_armed_at_bar = -1
    pivot_armed = False
    pivot_armed_at_bar = -1
    sl_path: list = []

    win_highs: list[float] = []
    win_lows: list[float] = []

    for j in range(n):
        bar = bars_idx.iloc[j]
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
        # find timestamp column (the index was reset, so first col is the index)
        ts_col = bar.get("timestamp", None)
        if ts_col is None:
            ts_col = bars_idx.iloc[j].iloc[0]
        bar_ts = pd.Timestamp(ts_col)

        # 1. Update MFE/MAE BEFORE exit checks so the exit bar's wick is
        #    captured (a bar that wicks to MFE>=BE then reverses to SL has
        #    its MFE seen — important for BE-arm rescue modeling).
        if side_u == "LONG":
            mfe_points = max(mfe_points, h - entry_price)
            mae_points = max(mae_points, entry_price - l)
        else:  # SHORT
            mfe_points = max(mfe_points, entry_price - l)
            mae_points = max(mae_points, h - entry_price)

        # 2. BE-arm: if MFE crosses threshold this bar (or already), snap SL
        #    to entry. Only on first arm; never moves SL backwards.
        if be_arm_active and not be_armed and mfe_points >= be_threshold - 1e-9:
            # Move SL to entry (only if it's a forward / favorable move).
            new_sl = float(entry_price)
            if side_u == "LONG":
                if new_sl > sl + 1e-9:
                    sl = new_sl
                    sl_path.append((j, sl, "be_arm"))
            else:  # SHORT
                if new_sl < sl - 1e-9:
                    sl = new_sl
                    sl_path.append((j, sl, "be_arm"))
            be_armed = True
            be_armed_at_bar = j

        # 3. Pivot Trail: detect swing pivot in last 5 bars and ratchet SL.
        win_highs.append(h)
        win_lows.append(l)
        if len(win_highs) > pivot_lookback:
            win_highs.pop(0)
            win_lows.pop(0)

        if pivot_active and (not us_session_only or _is_us_session(bar_ts)):
            if len(win_highs) == pivot_lookback:
                if side_u == "LONG":
                    pivot_price = detect_pivot_high(win_highs, pivot_lookback)
                else:
                    pivot_price = detect_pivot_low(win_lows, pivot_lookback)
                if pivot_price is not None:
                    candidate = compute_pivot_trail_sl(
                        side_u, float(pivot_price), entry_price, sl
                    )
                    if candidate is not None:
                        sl = candidate
                        if not pivot_armed:
                            pivot_armed = True
                            pivot_armed_at_bar = j
                        sl_path.append((j, sl, "pivot_trail"))

        # 4. SL/TP exit checks using the (possibly updated) SL.
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
            return FullSteppedOutcome(
                "stop_pessimistic", float(sl), j, float(entry_price), float(pnl),
                float(mfe_points), float(mae_points),
                be_armed, be_armed_at_bar, pivot_armed, pivot_armed_at_bar,
                float(sl), sl_path,
            )
        if sl_hit:
            pnl = (sl - entry_price) if side_u == "LONG" else (entry_price - sl)
            return FullSteppedOutcome(
                "stop", float(sl), j, float(entry_price), float(pnl),
                float(mfe_points), float(mae_points),
                be_armed, be_armed_at_bar, pivot_armed, pivot_armed_at_bar,
                float(sl), sl_path,
            )
        if tp_hit:
            pnl = (tp - entry_price) if side_u == "LONG" else (entry_price - tp)
            return FullSteppedOutcome(
                "take", float(tp), j, float(entry_price), float(pnl),
                float(mfe_points), float(mae_points),
                be_armed, be_armed_at_bar, pivot_armed, pivot_armed_at_bar,
                float(sl), sl_path,
            )

    # Horizon exit on last bar's close.
    last = bars_idx.iloc[n - 1]
    exit_price = float(last["close"])
    pnl = (exit_price - entry_price) if side_u == "LONG" else (entry_price - exit_price)
    return FullSteppedOutcome(
        "horizon", exit_price, n - 1, float(entry_price), float(pnl),
        float(mfe_points), float(mae_points),
        be_armed, be_armed_at_bar, pivot_armed, pivot_armed_at_bar,
        float(sl), sl_path,
    )


def simulate_trade_with_full_steps(
    parquet_df: pd.DataFrame,
    entry_ts,
    side: str,
    entry_price: float,
    sl_distance: float,
    tp_distance: float,
    *,
    contract: Optional[str] = None,
    horizon_bars: int = 30,
    be_arm_active: bool = True,
    pivot_active: bool = True,
    multiplier: float = 5.0,
    fee: float = 7.50,
    us_session_only: bool = True,
) -> dict:
    """High-level wrapper: pin contract -> walk-forward 1-min bars -> full
    stepped-SL walk -> dict.

    sl_distance / tp_distance are positive point distances. Initial SL price
    = entry - sl_distance (LONG) or entry + sl_distance (SHORT). TP price
    = entry + tp_distance (LONG) or entry - tp_distance (SHORT).

    multiplier=5 => MES ($/pt); fee=$7.50 round-trip haircut.
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

    out = simulate_full_stepped(
        bars, side=side_u, entry_price=entry_price,
        initial_sl=sl_price, initial_tp=tp_price,
        be_arm_active=be_arm_active, pivot_active=pivot_active,
        us_session_only=us_session_only,
    )
    raw_pnl = out.pnl_points * multiplier
    net_pnl = raw_pnl - fee
    return {
        "contract": contract,
        "exit_reason": out.exit_reason,
        "exit_price": out.exit_price,
        "exit_bar": out.exit_bar,
        "entry_price": out.entry_price,
        "pnl_points": out.pnl_points,
        "raw_pnl": raw_pnl,
        "net_pnl_after_haircut": net_pnl,
        "mfe_points": out.mfe_points,
        "mae_points": out.mae_points,
        "be_armed": out.be_armed,
        "be_armed_at_bar": out.be_armed_at_bar,
        "pivot_armed": out.pivot_armed,
        "pivot_armed_at_bar": out.pivot_armed_at_bar,
        "final_sl": out.final_sl,
        "sl_path": out.sl_path,
        "n_walk_bars": int(len(bars)),
    }
