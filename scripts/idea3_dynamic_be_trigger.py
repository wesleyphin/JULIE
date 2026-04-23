#!/usr/bin/env python3
"""IDEA 3 — dynamic break-even trigger scaled to realized session range.

Current behavior: DE3 arms BE at MFE ≈ 10pt (static; actually
tp_dist × trigger_pct ≈ 10 for most variants).

New behavior:
    BE_trigger = clamp(0.4 × realized_session_high_low, min=3, max=10)

On dead_tape days (low range) → trigger lower → BE arms earlier.
On trend days (high range) → trigger up to the existing 10pt cap.

OOS protocol
------------
  * Train: trades with ts < 2026-04-01 (used only to sanity-check the
    distribution of session ranges and dynamic triggers).
  * Test:  April 2026 trades.
  * For each test trade, simulate two scenarios via forward-walk
    through the live-prices parquet:
        A) baseline  — BE arms at MFE ≥ 10pt (or tp_dist × 0.40 capped
                        at 10, the T2/T3 de-facto behavior).
        B) dynamic   — BE arms at clamp(0.4 × session_range, 3, 10).
    Each simulation runs from entry_time until the trade hits TP, SL,
    or lookahead cap (60 bars). BE-armed SL sits at entry price.
  * Ship gate: dynamic PnL > baseline PnL AND WR non-regressive AND
    MaxDD not materially worse.

The simulation is honest: it uses tick-level bar data from the
parquet, tracks MFE bar-by-bar, arms BE when MFE crosses the
trigger, and fills at either the original SL, the BE SL (once
armed), or TP. Losers that retrace through BE become
breakeven; winners still hit TP unless the post-BE retrace sweeps
them first.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
LEDGER = ROOT / "ai_loop_data/triathlon/ledger.db"
PRICES = ROOT / "ai_loop_data/live_prices.parquet"
OUT = ROOT / "backtest_reports/idea3_dynamic_be_results.json"

SPLIT_DATE = "2026-04-01"

# Simulation constants
MES_POINT_VALUE = 5.0
LOOKAHEAD_BARS = 60
STATIC_BE_TRIGGER = 10.0
DYNAMIC_BE_FRAC   = 0.40
DYNAMIC_BE_MIN    = 3.0
DYNAMIC_BE_MAX    = 10.0


# ─── data ─────────────────────────────────────────────────────
def load_trades() -> tuple[list[dict], list[dict]]:
    conn = sqlite3.connect(str(LEDGER))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        """
        SELECT s.ts, s.strategy, s.sub_strategy, s.side, s.regime,
               s.time_bucket, s.size, s.entry_price, s.tp_dist, s.sl_dist,
               o.pnl_dollars, o.bars_held
        FROM signals s JOIN outcomes o ON o.signal_id = s.signal_id
        WHERE s.source_tag IN ('seed_2025','seed_2026')
          AND o.counterfactual = 0 AND s.status = 'fired'
          AND s.strategy LIKE '%DynamicEngine3%'
          AND s.entry_price > 0
        ORDER BY s.ts
        """
    ))
    conn.close()
    trades = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["ts"])
        except Exception:
            continue
        # Prefer the sub_strategy-encoded SL/TP if the row fields are null
        sub = str(r["sub_strategy"] or "")
        sl_dist = r["sl_dist"]
        tp_dist = r["tp_dist"]
        if sl_dist is None or tp_dist is None:
            m_sl = re.search(r"_SL([\d.]+)", sub)
            m_tp = re.search(r"_TP([\d.]+)", sub)
            if m_sl and sl_dist is None: sl_dist = float(m_sl.group(1))
            if m_tp and tp_dist is None: tp_dist = float(m_tp.group(1))
        if sl_dist is None or tp_dist is None:
            continue
        trades.append({
            "ts": ts, "strategy": r["strategy"], "side": r["side"],
            "regime": r["regime"], "time_bucket": r["time_bucket"],
            "entry_price": float(r["entry_price"]),
            "tp_dist": float(tp_dist), "sl_dist": float(sl_dist),
            "size": max(1, int(r["size"] or 1)),
            "pnl_dollars_realized": float(r["pnl_dollars"] or 0.0),
        })
    split = datetime.fromisoformat(SPLIT_DATE)
    if trades and trades[0]["ts"].tzinfo is not None and split.tzinfo is None:
        split = split.replace(tzinfo=trades[0]["ts"].tzinfo)
    return [t for t in trades if t["ts"] < split], [t for t in trades if t["ts"] >= split]


def load_prices() -> pd.DataFrame:
    df = pd.read_parquet(PRICES)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df.sort_index()


# ─── session range ────────────────────────────────────────────
def session_range_at(df: pd.DataFrame, ts: datetime) -> Optional[float]:
    """High-low of the trading day containing `ts`, measured BEFORE ts
    (no lookahead). Returns None if no bars available."""
    ts_ny = pd.Timestamp(ts)
    if ts_ny.tzinfo is None:
        ts_ny = ts_ny.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    else:
        ts_ny = ts_ny.tz_convert("America/New_York")
    # Trading day = calendar date in ET. This is a simple and reasonable
    # bucketing; overnight sessions that cross midnight aren't special-cased.
    day_start = pd.Timestamp(ts_ny.date(), tz="America/New_York")
    sub = df.loc[(df.index >= day_start) & (df.index <= ts_ny), "price"]
    if sub.empty:
        return None
    return float(sub.max() - sub.min())


def dynamic_be_trigger(session_range: Optional[float]) -> float:
    """clamp(0.4 × session_range, 3, 10)."""
    if session_range is None:
        return STATIC_BE_TRIGGER   # fallback to current behavior when no data
    return max(DYNAMIC_BE_MIN, min(DYNAMIC_BE_MAX, DYNAMIC_BE_FRAC * session_range))


# ─── per-trade simulation ─────────────────────────────────────
def simulate_trade(
    df: pd.DataFrame, trade: dict, be_trigger: float,
    lookahead: int = LOOKAHEAD_BARS,
) -> Optional[tuple[float, int, str, bool]]:
    """Forward-walk: track MFE; arm BE when MFE >= be_trigger;
    exit on TP, SL (original or BE), or lookahead cap.

    Returns (pnl_points, bars_held, exit_source, be_armed).
    """
    ts_ny = pd.Timestamp(trade["ts"])
    if ts_ny.tzinfo is None:
        ts_ny = ts_ny.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    sub = df.loc[df.index > ts_ny, "price"].head(lookahead)
    if sub.empty:
        return None
    entry = trade["entry_price"]
    side = trade["side"]
    tp_dist = trade["tp_dist"]
    sl_dist = trade["sl_dist"]
    tp_price = entry + tp_dist if side == "LONG" else entry - tp_dist
    sl_price = entry - sl_dist if side == "LONG" else entry + sl_dist

    mfe = 0.0
    be_armed = False
    effective_sl = sl_price
    for i, (_, px) in enumerate(sub.items(), start=1):
        if side == "LONG":
            fav = px - entry
        else:
            fav = entry - px
        if fav > mfe:
            mfe = fav
        # BE arming trigger
        if not be_armed and mfe >= be_trigger:
            be_armed = True
            effective_sl = entry   # SL moves to entry
        # Check for exit (SL check first — conservative)
        if side == "LONG":
            if px <= effective_sl:
                realized_pts = effective_sl - entry  # 0 if BE armed, -sl_dist otherwise
                return (realized_pts, i, "be_sl" if be_armed else "sl", be_armed)
            if px >= tp_price:
                return (tp_dist, i, "tp", be_armed)
        else:
            if px >= effective_sl:
                realized_pts = entry - effective_sl
                return (realized_pts, i, "be_sl" if be_armed else "sl", be_armed)
            if px <= tp_price:
                return (tp_dist, i, "tp", be_armed)
    # Lookahead expired — close at last bar
    last = float(sub.iloc[-1])
    pts = (last - entry) if side == "LONG" else (entry - last)
    return (pts, len(sub), "expired", be_armed)


# ─── aggregate + ship gate ────────────────────────────────────
def stats(pnls: list[float]) -> dict:
    if not pnls: return {"n": 0, "pnl": 0.0, "wr": 0.0, "max_dd": 0.0, "pf": None, "avg": 0}
    cum = 0.0; peak = 0.0; dd = 0.0
    for p in pnls:
        cum += p; peak = max(peak, cum); dd = max(dd, peak - cum)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    return {
        "n": len(pnls), "pnl": round(cum, 2),
        "wr": round(len(wins) / len(pnls) * 100, 2),
        "max_dd": round(dd, 2),
        "pf": round(pf, 3) if pf != float("inf") else None,
        "avg": round(cum / len(pnls), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Cap trades for speed (debug only)")
    args = ap.parse_args()

    print(f"[idea3] loading trades + price parquet...")
    pre, april = load_trades()
    df = load_prices()
    print(f"  pre-April: {len(pre):,}  April: {len(april):,}  parquet rows: {len(df):,}")

    # Quick exploratory: distribution of session ranges on pre-April
    ranges_pre = [r for r in (session_range_at(df, t["ts"]) for t in pre) if r is not None]
    ranges_april = [r for r in (session_range_at(df, t["ts"]) for t in april) if r is not None]
    if ranges_pre:
        print(f"\n  pre-April session-range distribution (points):")
        print(f"    min={min(ranges_pre):.1f}  median={statistics.median(ranges_pre):.1f}  "
              f"p90={sorted(ranges_pre)[int(len(ranges_pre)*0.9)]:.1f}  max={max(ranges_pre):.1f}")
    if ranges_april:
        print(f"  April session-range distribution (points):")
        print(f"    min={min(ranges_april):.1f}  median={statistics.median(ranges_april):.1f}  "
              f"p90={sorted(ranges_april)[int(len(ranges_april)*0.9)]:.1f}  max={max(ranges_april):.1f}")

    # Simulate baseline + dynamic on the April holdout
    target = april if args.limit is None else april[: args.limit]
    print(f"\n[idea3] simulating {len(target):,} April trades under two BE policies...")

    baseline_pnls = []
    dynamic_pnls = []
    be_diff_count = 0
    be_trigger_dist: dict = defaultdict(int)
    by_regime: dict = defaultdict(lambda: {"n": 0, "base_pnl": 0.0, "dyn_pnl": 0.0})

    skipped_no_data = 0
    for t in target:
        sr = session_range_at(df, t["ts"])
        dyn_trig = dynamic_be_trigger(sr)
        be_trigger_dist[round(dyn_trig, 1)] += 1

        base = simulate_trade(df, t, STATIC_BE_TRIGGER)
        dyn  = simulate_trade(df, t, dyn_trig)
        if base is None or dyn is None:
            skipped_no_data += 1
            continue
        base_pts, _, _, _ = base
        dyn_pts, _, _, _ = dyn
        base_usd = base_pts * MES_POINT_VALUE * t["size"]
        dyn_usd  = dyn_pts  * MES_POINT_VALUE * t["size"]
        baseline_pnls.append(base_usd)
        dynamic_pnls.append(dyn_usd)
        if abs(base_usd - dyn_usd) > 0.01:
            be_diff_count += 1
        by_regime[t["regime"]]["n"]       += 1
        by_regime[t["regime"]]["base_pnl"]+= base_usd
        by_regime[t["regime"]]["dyn_pnl"] += dyn_usd

    print(f"  simulated: {len(baseline_pnls):,}  skipped (no fwd bar data): {skipped_no_data:,}")
    print(f"  trades where dynamic BE produced a different outcome: {be_diff_count:,}")
    print(f"\n  dynamic trigger distribution (pts, rounded to 0.1):")
    for trig in sorted(be_trigger_dist):
        print(f"    {trig:>5.1f}pt: {be_trigger_dist[trig]:>4}")

    base_stats = stats(baseline_pnls)
    dyn_stats  = stats(dynamic_pnls)

    print(f"\n═══ OOS RESULT (April 2026 holdout, simulated under each BE policy) ═══")
    print(f"  baseline (BE trigger = 10pt static):")
    print(f"    n={base_stats['n']}  PnL=${base_stats['pnl']:+,.2f}  "
          f"WR={base_stats['wr']:.2f}%  MaxDD=${base_stats['max_dd']:,.0f}  "
          f"PF={base_stats['pf']}  avg/trade=${base_stats['avg']:+.2f}")
    print(f"  dynamic (BE trigger = clamp(0.4 × session_range, 3, 10)):")
    print(f"    n={dyn_stats['n']}  PnL=${dyn_stats['pnl']:+,.2f}  "
          f"WR={dyn_stats['wr']:.2f}%  MaxDD=${dyn_stats['max_dd']:,.0f}  "
          f"PF={dyn_stats['pf']}  avg/trade=${dyn_stats['avg']:+.2f}")
    print(f"  DELTA:")
    print(f"    PnL    ${dyn_stats['pnl'] - base_stats['pnl']:+,.2f}")
    print(f"    WR     {dyn_stats['wr'] - base_stats['wr']:+.2f}pp")
    print(f"    MaxDD  ${dyn_stats['max_dd'] - base_stats['max_dd']:+,.0f}")

    print(f"\n  per-regime breakdown (dynamic vs baseline PnL):")
    for regime, bucket in sorted(by_regime.items()):
        d = bucket["dyn_pnl"] - bucket["base_pnl"]
        print(f"    {regime:<12} n={bucket['n']:>3}  base=${bucket['base_pnl']:+,.2f}  "
              f"dyn=${bucket['dyn_pnl']:+,.2f}  Δ=${d:+,.2f}")

    pnl_up = dyn_stats["pnl"] > base_stats["pnl"]
    wr_ok  = (dyn_stats["wr"] - base_stats["wr"]) >= -0.5
    dd_ok  = (dyn_stats["max_dd"] - base_stats["max_dd"]) <= max(0, base_stats["max_dd"] * 0.10)
    ship = pnl_up and wr_ok and dd_ok
    print(f"\n  ship criteria:")
    print(f"    PnL up?       {pnl_up} (${dyn_stats['pnl'] - base_stats['pnl']:+.2f})")
    print(f"    WR non-reg?   {wr_ok} ({dyn_stats['wr'] - base_stats['wr']:+.2f}pp)")
    print(f"    MaxDD ≤ +10%? {dd_ok}")
    print(f"  VERDICT: {'SHIP' if ship else 'DO NOT SHIP'}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "baseline": base_stats,
        "dynamic":  dyn_stats,
        "delta_pnl":    round(dyn_stats["pnl"] - base_stats["pnl"], 2),
        "delta_wr":     round(dyn_stats["wr"] - base_stats["wr"], 2),
        "delta_dd":     round(dyn_stats["max_dd"] - base_stats["max_dd"], 2),
        "be_trigger_distribution": dict(be_trigger_dist),
        "per_regime": dict(by_regime),
        "ship": ship,
    }, indent=2, default=str))
    print(f"\n[write] {OUT}")

    return 0 if ship else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
