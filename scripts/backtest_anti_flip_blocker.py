#!/usr/bin/env python3
"""Quantify the anti-flip blocker hypothesis.

Rule being tested:
  When a trade stops out (PnL < 0 via `source='stop'` or `source='stop_gap'`),
  record the stop-out timestamp, the stop-out exit price, and the side.
  For the next `window_minutes`, reject any OPPOSITE-side signal whose
  entry price is within `max_distance_pts` of the stop-out price.

  LONG stop at 7172.50 → block any SHORT entry within ±5pt for 15 min.
  SHORT stop at 7172.50 → block any LONG entry within ±5pt for 15 min.

Why: the 2026-04-23 session showed DE3 SHORT stopping out at 7172.50 and
then DE3 flipping to LONG at 7171.75 (0.75pt below the stop) one minute
later. That LONG rode straight into a 64pt dump and lost. The "flip at
the stop" pattern wastes two trades on the same noise.

Usage:
    python3 scripts/backtest_anti_flip_blocker.py
    python3 scripts/backtest_anti_flip_blocker.py --verbose

Writes `backtest_reports/anti_flip_blocker_sweep.json`.

No parquet price data needed — closed_trades.json alone carries enough:
entry_time, entry_price, exit_price, side, pnl_points, source.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path("/Users/wes/Downloads/JULIE001")


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def load_trades(sources: list[Path]) -> list[dict]:
    combined = []
    seen = set()
    for p in sources:
        if not p.exists(): continue
        raw = json.loads(p.read_text())
        for t in raw:
            key = (t.get("entry_time"), t.get("strategy"),
                   t.get("side"), t.get("entry_price"))
            if key in seen: continue
            seen.add(key)
            combined.append(t)
    combined.sort(key=lambda t: _parse_ts(t["entry_time"]))
    return combined


STOP_SOURCES = {"stop", "stop_gap"}


def simulate(trades: list[dict], *, window_min: int, max_dist_pts: float) -> dict:
    """Walk trades chronologically, apply the anti-flip rule."""
    kept = []
    blocked = []

    # Last stop-out per side
    last_stop_long_ts: datetime | None = None
    last_stop_long_px: float | None = None
    last_stop_short_ts: datetime | None = None
    last_stop_short_px: float | None = None

    for t in trades:
        side = str(t.get("side", "")).upper()
        entry_ts = _parse_ts(t["entry_time"])
        try:
            entry_px = float(t.get("entry_price", 0.0) or 0.0)
        except (ValueError, TypeError):
            entry_px = 0.0

        # Check anti-flip rule: opposite-side stop-out, within window, within distance
        block_reason = None
        if side == "LONG" and last_stop_short_ts is not None:
            elapsed_min = (entry_ts - last_stop_short_ts).total_seconds() / 60.0
            if 0 <= elapsed_min <= window_min:
                dist = abs(entry_px - last_stop_short_px)
                if dist <= max_dist_pts:
                    block_reason = (f"SHORT stop {elapsed_min:.1f}min ago @ "
                                     f"{last_stop_short_px:.2f}; LONG entry {entry_px:.2f} "
                                     f"Δ={dist:.2f}pt")
        elif side == "SHORT" and last_stop_long_ts is not None:
            elapsed_min = (entry_ts - last_stop_long_ts).total_seconds() / 60.0
            if 0 <= elapsed_min <= window_min:
                dist = abs(entry_px - last_stop_long_px)
                if dist <= max_dist_pts:
                    block_reason = (f"LONG stop {elapsed_min:.1f}min ago @ "
                                     f"{last_stop_long_px:.2f}; SHORT entry {entry_px:.2f} "
                                     f"Δ={dist:.2f}pt")

        if block_reason is not None:
            blocked.append({**t, "_block_reason": block_reason})
            # Blocked trade never happens, so no stop-out to record from it
            continue

        kept.append(t)

        # Record stop-outs AFTER the trade runs (so the anti-flip only applies
        # to FUTURE signals, not this trade itself)
        source = str(t.get("source", "")).lower()
        pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
        if source in STOP_SOURCES and pnl < 0:
            exit_ts = _parse_ts(t.get("exit_time") or t["entry_time"])
            try:
                exit_px = float(t.get("exit_price", entry_px) or entry_px)
            except (ValueError, TypeError):
                exit_px = entry_px
            if side == "LONG":
                last_stop_long_ts = exit_ts
                last_stop_long_px = exit_px
            elif side == "SHORT":
                last_stop_short_ts = exit_ts
                last_stop_short_px = exit_px

    # Aggregate PnL over kept trades
    cum = 0.0; peak = 0.0; dd = 0.0
    for t in kept:
        pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
        cum += pnl; peak = max(peak, cum); dd = max(dd, peak - cum)
    pnls = [float(t.get("pnl_dollars", 0.0) or 0.0) for t in kept]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    pf = (sum(p for p in pnls if p > 0) /
          abs(sum(p for p in pnls if p < 0))) if losses else float("inf")

    blocked_pnl = sum(float(t.get("pnl_dollars", 0.0) or 0.0) for t in blocked)
    blocked_wins = sum(1 for t in blocked if float(t.get("pnl_dollars", 0) or 0) > 0)
    blocked_losses = sum(1 for t in blocked if float(t.get("pnl_dollars", 0) or 0) < 0)

    return {
        "config": {"window_min": window_min, "max_dist_pts": max_dist_pts},
        "kept_trades": len(kept), "blocked_trades": len(blocked),
        "pnl": round(cum, 2),
        "max_dd": round(dd, 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else None,
        "wr": round(wins / max(1, len(kept)) * 100, 1),
        "wins": wins, "losses": losses,
        "blocked_pnl_counterfactual": round(blocked_pnl, 2),
        "blocked_wins": blocked_wins, "blocked_losses": blocked_losses,
    }


def baseline(trades: list[dict]) -> dict:
    cum = 0.0; peak = 0.0; dd = 0.0
    for t in trades:
        pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
        cum += pnl; peak = max(peak, cum); dd = max(dd, peak - cum)
    pnls = [float(t.get("pnl_dollars", 0.0) or 0.0) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    pf = (sum(p for p in pnls if p > 0) /
          abs(sum(p for p in pnls if p < 0))) if losses else float("inf")
    return {
        "pnl": round(cum, 2), "max_dd": round(dd, 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else None,
        "wr": round(wins / max(1, len(trades)) * 100, 1),
        "wins": wins, "losses": losses, "n_trades": len(trades),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Same sources the cascade-blocker backtest uses
    trades_2026 = load_trades([
        ROOT / "backtest_reports/full_live_replay/2026_jan_apr/closed_trades.json",
        ROOT / "backtest_reports/full_live_replay/2026_04_ml_stacks/closed_trades.json",
        ROOT / "backtest_reports/replay_apr2026_p1/live_loop_MES_20260421_061829/closed_trades.json",
        ROOT / "backtest_reports/af_fast_replay/2026_01/closed_trades.json",
        ROOT / "backtest_reports/af_fast_replay/2026_02/closed_trades.json",
        ROOT / "backtest_reports/af_fast_replay/2026_03/closed_trades.json",
        ROOT / "backtest_reports/af_fast_replay/2026_04/closed_trades.json",
        ROOT / "backtest_reports/pivot_week_4_19_21_ml/live_loop_MES_20260422_013828/closed_trades.json",
    ])
    trades_2025 = load_trades([
        ROOT / f"backtest_reports/full_live_replay/2025_{m:02d}/closed_trades.json"
        for m in range(1, 13)
    ] + [ROOT / "backtest_reports/full_live_replay/outrageous_apr/closed_trades.json"])

    results = {}
    for tape_name, trades in (("2026", trades_2026), ("2025", trades_2025)):
        base = baseline(trades)
        print(f"\n═══ {tape_name} — baseline: {base['n_trades']} trades · "
              f"${base['pnl']:+,.2f} · DD ${base['max_dd']:,.0f} · "
              f"PF {base['profit_factor']} · WR {base['wr']}% ═══")
        # Diagnostic: count stop-outs in this tape
        stopouts = sum(1 for t in trades
                       if str(t.get("source","")).lower() in STOP_SOURCES
                       and float(t.get("pnl_dollars",0) or 0) < 0)
        print(f"  stop-outs in tape: {stopouts}")

        print(f"  {'-'*102}")
        print(f"  {'window_min':<11} {'max_dist_pts':<13}  {'pnl':>10}  "
              f"{'Δ base':>10}  {'DD':>7}  {'ΔDD':>8}  {'PF':>5}  {'kept':>5}  "
              f"{'blk':>4}  {'blk$':>9}  {'blk W/L':>8}")
        print(f"  {'-'*102}")
        grid = []
        for window_min in (5, 10, 15, 30, 60):
            for max_dist in (2.0, 3.0, 5.0, 8.0, 12.0):
                r = simulate(trades, window_min=window_min, max_dist_pts=max_dist)
                delta_pnl = r["pnl"] - base["pnl"]
                delta_dd = r["max_dd"] - base["max_dd"]
                pf = (f"{r['profit_factor']:.2f}"
                      if r['profit_factor'] is not None else "∞")
                print(f"  {window_min:<11} {max_dist:<13}  "
                      f"${r['pnl']:>+9.2f}  ${delta_pnl:>+8.2f}  "
                      f"${r['max_dd']:>5.0f}  ${delta_dd:>+6.0f}  "
                      f"{pf:>5}  {r['kept_trades']:>5}  {r['blocked_trades']:>4}  "
                      f"${r['blocked_pnl_counterfactual']:>+7.2f}  "
                      f"{r['blocked_wins']:>2}/{r['blocked_losses']:<2}")
                grid.append(r)
        results[tape_name] = {"baseline": base, "grid": grid,
                              "stopouts_in_tape": stopouts}

    # Best configs per tape by Δ PnL
    print(f"\n═══ Best configs (by Δ PnL, across both tapes) ═══")
    for tape in ("2026", "2025"):
        grid = results[tape]["grid"]
        base_pnl = results[tape]["baseline"]["pnl"]
        top = sorted(grid, key=lambda r: -(r["pnl"] - base_pnl))[:5]
        print(f"\n  {tape}:")
        for r in top:
            c = r["config"]
            delta = r["pnl"] - base_pnl
            print(f"    window={c['window_min']:>2}min dist={c['max_dist_pts']:>4.1f}pt"
                  f"  →  Δ ${delta:+,.2f}"
                  f"  (blocked {r['blocked_trades']} trades worth ${r['blocked_pnl_counterfactual']:+.0f},"
                  f" {r['blocked_wins']}W/{r['blocked_losses']}L)")

    # Also: find configs that win on BOTH years
    print(f"\n═══ Configs winning on BOTH tapes ═══")
    base_2026 = results["2026"]["baseline"]["pnl"]
    base_2025 = results["2025"]["baseline"]["pnl"]
    pair = []
    for r26 in results["2026"]["grid"]:
        c = r26["config"]
        r25 = next(x for x in results["2025"]["grid"] if x["config"] == c)
        d26 = r26["pnl"] - base_2026
        d25 = r25["pnl"] - base_2025
        if d26 > 0 and d25 > 0:
            pair.append((c, d26, d25, r26, r25))
    pair.sort(key=lambda x: -(x[1] + x[2]))
    for c, d26, d25, r26, r25 in pair[:10]:
        total = d26 + d25
        print(f"    window={c['window_min']:>2}min dist={c['max_dist_pts']:>4.1f}pt"
              f"  →  2026 Δ ${d26:+,.0f}, 2025 Δ ${d25:+,.0f}, combined ${total:+,.0f}")

    out = ROOT / "backtest_reports" / "anti_flip_blocker_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
