#!/usr/bin/env python3
"""Quantify the low-vol-day-filter hypothesis against the 2025+2026 tape.

Built after the AI-loop price-parquet updater surfaced that both 2025
(288d, 3552 trades) and 2026 (82d, 961 trades) show the SAME pattern:
best days have 2× the intraday range of worst days — i.e. the bot
makes money on wide tapes, loses on quiet ones.

This script answers: if we had skipped entire trading days where the
final intraday range was below X pts (a "low-vol skip filter"), what
would PnL, DD, profit factor have looked like?

We can't evaluate "intraday so-far" from closed_trades alone because
we don't know when during the day a signal fired. So we use the whole
day's final range as a proxy — in a live filter we'd use a trailing
30-min window; if the full-day backtest shows NO lift, the trailing
filter definitely won't help. If it DOES show lift, we have grounds
to build the live filter.

Usage:
    python3 scripts/backtest_low_vol_filter.py
    python3 scripts/backtest_low_vol_filter.py --thr 20 30 40 60 80

Writes `backtest_reports/low_vol_filter_sweep.json` with the full grid.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")


def load_closed_trades(sources: list[Path]) -> list[dict]:
    """Dedupe by (entry_time, strategy, side, entry_price) across sources."""
    combined = []
    seen = set()
    for p in sources:
        raw = json.loads(p.read_text())
        for t in raw:
            key = (t.get("entry_time"), t.get("strategy"),
                   t.get("side"), t.get("entry_price"))
            if key in seen: continue
            seen.add(key)
            combined.append(t)
    combined.sort(key=lambda t: t["entry_time"])
    return combined


def day_stats_from_parquet(df: pd.DataFrame, day: str) -> dict | None:
    start = pd.Timestamp(day, tz="America/New_York")
    end = start + pd.Timedelta(days=1)
    sub = df.loc[(df.index >= start) & (df.index < end)]
    if sub.empty:
        return None
    px = sub["price"].astype(float)
    diffs = px.diff().dropna().abs()
    return {
        "range_pts": float(px.max() - px.min()),
        "bar_vol_pts": float(diffs.std()) if len(diffs) > 1 else 0.0,
    }


def simulate_filter(
    trades: list[dict],
    day_stats: dict[str, dict],
    *,
    min_range: float | None = None,
    min_vol: float | None = None,
) -> dict:
    """If the day's range < min_range OR bar-vol < min_vol, SKIP every
    trade that would have fired that day. Returns pnl/dd/n summary."""
    by_day: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_day[t["entry_time"][:10]].append(t)

    kept_days = 0; skipped_days = 0
    kept_trades = 0; skipped_trades = 0
    total_pnl = 0.0
    cum = 0.0; peak = 0.0; dd = 0.0
    chron = []
    for day in sorted(by_day):
        ds = day_stats.get(day)
        if ds is None:
            # No parquet data for this day → keep baseline
            skip = False
        else:
            skip = False
            if min_range is not None and ds["range_pts"] < min_range:
                skip = True
            if min_vol is not None and ds["bar_vol_pts"] < min_vol:
                skip = True
        day_trades = by_day[day]
        if skip:
            skipped_days += 1
            skipped_trades += len(day_trades)
            continue
        kept_days += 1
        kept_trades += len(day_trades)
        for t in day_trades:
            chron.append(t)

    # Equity curve over kept trades in time order
    chron.sort(key=lambda t: t["entry_time"])
    for t in chron:
        pnl = float(t.get("pnl_dollars", 0) or 0)
        total_pnl += pnl
        cum += pnl
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    wins = [float(t.get("pnl_dollars", 0) or 0) for t in chron if (t.get("pnl_dollars") or 0) > 0]
    losses = [float(t.get("pnl_dollars", 0) or 0) for t in chron if (t.get("pnl_dollars") or 0) < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    return {
        "pnl": round(total_pnl, 2),
        "max_dd": round(dd, 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else None,
        "kept_days": kept_days, "skipped_days": skipped_days,
        "kept_trades": kept_trades, "skipped_trades": skipped_trades,
        "wins": len(wins), "losses": len(losses),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", nargs="+", type=float,
                    default=[20, 30, 40, 60, 80, 100, 120],
                    help="Range thresholds in pts to sweep.")
    ap.add_argument("--vol", nargs="+", type=float,
                    default=[0.8, 1.0, 1.3, 1.6, 2.0],
                    help="Bar-vol thresholds in pts to sweep.")
    args = ap.parse_args()

    # Load parquet
    px_path = ROOT / "ai_loop_data" / "live_prices.parquet"
    if not px_path.exists():
        raise SystemExit(
            "price parquet not found — run `python3 -m tools.ai_loop.price_parquet_updater` first"
        )
    df = pd.read_parquet(px_path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("America/New_York", ambiguous="NaT",
                                         nonexistent="shift_forward")
    else:
        df.index = df.index.tz_convert("America/New_York")

    # Sources — same set the consensus journals use
    trades_2026 = load_closed_trades([
        ROOT / "backtest_reports/full_live_replay/2026_jan_apr/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2026_04_ml_stacks/closed_trades.json",
        ROOT / "backtest_reports/replay_apr2026_p1/live_loop_MES_20260421_061829/closed_trades.json",
        ROOT / "backtest_reports/af_fast_replay/2026_01/closed_trades.json",
        ROOT / "backtest_reports/af_fast_replay/2026_02/closed_trades.json",
        ROOT / "backtest_reports/af_fast_replay/2026_03/closed_trades.json",
        ROOT / "backtest_reports/af_fast_replay/2026_04/closed_trades.json",
        ROOT / "backtest_reports/pivot_week_4_19_21_ml/live_loop_MES_20260422_013828/closed_trades.json",
    ])
    trades_2025 = load_closed_trades([
        ROOT / "backtest_reports/full_live_replay/2025_01/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_02/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_03/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/outrageous_apr/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_05/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_06/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_07/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_08/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_09/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_10/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_11/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2025_12/closed_trades.json",
    ])

    results = {}
    for name, trades in (("2026", trades_2026), ("2025", trades_2025)):
        # Build per-day stats cache
        days = sorted({t["entry_time"][:10] for t in trades})
        day_stats = {}
        for d in days:
            ds = day_stats_from_parquet(df, d)
            if ds: day_stats[d] = ds

        baseline = simulate_filter(trades, day_stats)
        print(f"\n═══ {name} ({len(trades)} trades across {len(days)} days, "
              f"{len(day_stats)} days with price data) ═══")
        print(f"  baseline              "
              f"pnl=${baseline['pnl']:>+9.2f}  DD=${baseline['max_dd']:>7.0f}  "
              f"PF={baseline['profit_factor']:.2f}  trades={baseline['kept_trades']}  "
              f"days={baseline['kept_days']}")
        print(f"  {'─'*85}")
        print(f"  {'RANGE filter':<20}  {'pnl':>10}  {'Δ vs base':>10}  {'DD':>7}  "
              f"{'PF':>5}  {'kept':>5}  {'skip':>5}")
        for thr in args.thr:
            r = simulate_filter(trades, day_stats, min_range=thr)
            delta = r["pnl"] - baseline["pnl"]
            pf = (f"{r['profit_factor']:.2f}" if r['profit_factor'] is not None else "∞")
            print(f"  range < {thr:>5.0f}pt      "
                  f"${r['pnl']:>+9.2f}  ${delta:>+9.2f}  ${r['max_dd']:>5.0f}  "
                  f"{pf:>5}  {r['kept_days']:>5}/{r['skipped_days']:<3} {r['kept_trades']:>5}")
        print(f"  {'─'*85}")
        print(f"  {'VOL filter':<20}  {'pnl':>10}  {'Δ vs base':>10}  {'DD':>7}  "
              f"{'PF':>5}  {'kept':>5}  {'skip':>5}")
        for v in args.vol:
            r = simulate_filter(trades, day_stats, min_vol=v)
            delta = r["pnl"] - baseline["pnl"]
            pf = (f"{r['profit_factor']:.2f}" if r['profit_factor'] is not None else "∞")
            print(f"  bar-vol < {v:>4.2f}pt    "
                  f"${r['pnl']:>+9.2f}  ${delta:>+9.2f}  ${r['max_dd']:>5.0f}  "
                  f"{pf:>5}  {r['kept_days']:>5}/{r['skipped_days']:<3} {r['kept_trades']:>5}")

        results[name] = {
            "baseline": baseline,
            "range_sweep": {
                str(thr): simulate_filter(trades, day_stats, min_range=thr)
                for thr in args.thr
            },
            "vol_sweep": {
                str(v): simulate_filter(trades, day_stats, min_vol=v)
                for v in args.vol
            },
        }

    out = ROOT / "backtest_reports" / "low_vol_filter_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
