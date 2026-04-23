#!/usr/bin/env python3
"""Quantify the consecutive-loss circuit-breaker hypothesis.

The rule being tested:
    If in the last `window_min` minutes there have been ≥ `count` LOSING
    trades on the same SIDE, skip every new same-side entry for
    `cooldown_min` minutes after the last loss.

This is a pure mechanical filter — no ML, no regime knowledge. It fires
on the pattern that blew up on 2026-04-22 (3 same-side LONG losses in 8
minutes during a 40-pt crash).

We back-test on the same tape the consensus journals use (2025 full year,
288 days, 3552 trades; 2026 Jan-Apr, 82 days, 961 trades).

Important simplification: each closed trade has entry_time + side +
pnl_dollars. If the rule would have blocked a trade, we DROP it from the
P&L series. We don't simulate replacement trades; this is a lower-bound
for lift (in reality, with one position at a time, skipping a trade often
frees the slot for a better one that fires 30s later — we can't model
that without full replay, so "lower bound" is the honest framing).

Sweep grid:
    count       : 2, 3
    window_min  : 10, 15, 30, 60
    cooldown_min: 10, 15, 30

Usage:
    python3 scripts/backtest_consec_loss_blocker.py
    python3 scripts/backtest_consec_loss_blocker.py --verbose

Writes `backtest_reports/consec_loss_blocker_sweep.json`.
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
        raw = json.loads(p.read_text())
        for t in raw:
            key = (t.get("entry_time"), t.get("strategy"),
                   t.get("side"), t.get("entry_price"))
            if key in seen: continue
            seen.add(key)
            combined.append(t)
    combined.sort(key=lambda t: _parse_ts(t["entry_time"]))
    return combined


def simulate(
    trades: list[dict],
    *,
    count: int,
    window_min: int,
    cooldown_min: int,
) -> dict:
    """Walk trades chronologically. For each candidate:
        - Look at the window [ts - window_min, ts] for trades of the SAME side
        - If ≥ count of them closed with PnL < 0, check cooldown
        - Cooldown ends `cooldown_min` minutes AFTER the last losing trade
        - If still in cooldown for this side, BLOCK this trade (skip it)
    Otherwise keep the trade.

    Returns per-config PnL / DD / trades kept / trades blocked.
    """
    kept: list[dict] = []
    blocked: list[dict] = []

    # Pre-split losses per side with timestamps
    side_losses: dict[str, list[datetime]] = defaultdict(list)

    for t in trades:
        side = str(t.get("side", "")).upper()
        ts = _parse_ts(t["entry_time"])

        # Count same-side losses in the window that have CLOSED by this ts
        # (so the blocker only fires on info the bot would actually have)
        # Use entry_time of the loss + pnl_dollars sign — no exit_time cost here
        # because closed_trades already has finalized PnL.
        recent_losses = [
            lt for lt in side_losses[side]
            if (ts - lt) <= timedelta(minutes=window_min)
            and (ts - lt) >= timedelta(seconds=0)
        ]
        if len(recent_losses) >= count:
            # Cooldown from the MOST RECENT same-side loss
            last_loss = max(recent_losses)
            if (ts - last_loss) <= timedelta(minutes=cooldown_min):
                blocked.append(t)
                # IMPORTANT: we don't add this trade's PnL to the loss
                # ledger even if it would have lost — the bot never took it.
                continue

        kept.append(t)
        # Only add to side-loss ledger if THIS trade was itself a loss
        if float(t.get("pnl_dollars", 0) or 0) < 0:
            side_losses[side].append(ts)

    # P&L / DD over kept trades
    cum = 0.0; peak = 0.0; dd = 0.0
    for t in kept:
        pnl = float(t.get("pnl_dollars", 0) or 0)
        cum += pnl; peak = max(peak, cum); dd = max(dd, peak - cum)
    pnls = [float(t.get("pnl_dollars", 0) or 0) for t in kept]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)

    # What would the BLOCKED trades have contributed?  Sum and sign split.
    blocked_pnls = [float(t.get("pnl_dollars", 0) or 0) for t in blocked]
    blocked_sum = sum(blocked_pnls)
    blocked_wins = sum(1 for p in blocked_pnls if p > 0)
    blocked_losses = sum(1 for p in blocked_pnls if p < 0)

    pf = (sum(p for p in pnls if p > 0) /
          abs(sum(p for p in pnls if p < 0))) if losses else float("inf")

    return {
        "config": {"count": count, "window_min": window_min,
                   "cooldown_min": cooldown_min},
        "kept_trades": len(kept), "blocked_trades": len(blocked),
        "pnl": round(cum, 2),
        "max_dd": round(dd, 2),
        "profit_factor": round(pf, 3) if pf != float("inf") else None,
        "wr": round(wins / max(1, len(kept)) * 100, 1),
        "wins": wins, "losses": losses,
        "blocked_pnl_counterfactual": round(blocked_sum, 2),
        "blocked_wins": blocked_wins, "blocked_losses": blocked_losses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

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
        if (ROOT / f"backtest_reports/full_live_replay/2025_{m:02d}/closed_trades.json").exists()
    ] + [ROOT / "backtest_reports/full_live_replay/outrageous_apr/closed_trades.json"])

    # Baseline = no blocker
    def baseline(trades):
        cum = 0.0; peak = 0.0; dd = 0.0
        for t in trades:
            pnl = float(t.get("pnl_dollars", 0) or 0)
            cum += pnl; peak = max(peak, cum); dd = max(dd, peak - cum)
        pnls = [float(t.get("pnl_dollars", 0) or 0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        pf = (sum(p for p in pnls if p > 0) /
              abs(sum(p for p in pnls if p < 0))) if losses else float("inf")
        return {
            "pnl": round(cum, 2), "max_dd": round(dd, 2),
            "profit_factor": round(pf, 3) if pf != float("inf") else None,
            "wr": round(wins / max(1, len(trades)) * 100, 1),
            "wins": wins, "losses": losses,
            "n_trades": len(trades),
        }

    results = {}
    for tape, trades in (("2026", trades_2026), ("2025", trades_2025)):
        base = baseline(trades)
        print(f"\n═══ {tape} — baseline: {base['n_trades']} trades · "
              f"${base['pnl']:+,.2f} · DD ${base['max_dd']:,.0f} · "
              f"PF {base['profit_factor']} · WR {base['wr']}% ═══")
        hdr = (f"  {'count':<5} {'win_min':<7} {'cool':<5}  "
               f"{'pnl':>10}  {'Δ base':>9}  {'DD':>7}  {'ΔDD':>8}  "
               f"{'PF':>5}  {'WR':>5}  {'kept':>5}  {'blk':>4}  "
               f"{'blk$':>9}  {'blk W/L':>8}")
        print(hdr); print("  " + "─" * (len(hdr) - 2))

        grid = []
        for count in (2, 3):
            for window_min in (10, 15, 30, 60):
                for cooldown_min in (10, 15, 30):
                    r = simulate(trades, count=count, window_min=window_min,
                                 cooldown_min=cooldown_min)
                    delta_pnl = r["pnl"] - base["pnl"]
                    delta_dd = r["max_dd"] - base["max_dd"]
                    pf = (f"{r['profit_factor']:.2f}"
                          if r['profit_factor'] is not None else "∞")
                    print(f"  {count:<5} {window_min:<7} {cooldown_min:<5}  "
                          f"${r['pnl']:>+9.2f}  ${delta_pnl:>+8.2f}  "
                          f"${r['max_dd']:>5.0f}  ${delta_dd:>+6.0f}  "
                          f"{pf:>5}  {r['wr']:>4.1f}%  "
                          f"{r['kept_trades']:>5}  {r['blocked_trades']:>4}  "
                          f"${r['blocked_pnl_counterfactual']:>+7.2f}  "
                          f"{r['blocked_wins']:>2}/{r['blocked_losses']:<2}")
                    grid.append(r)

        results[tape] = {"baseline": base, "grid": grid}

    out = ROOT / "backtest_reports" / "consec_loss_blocker_sweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[write] {out}")

    # Best single-config summary
    print("\n═══ Best configs (by Δ PnL) ═══")
    for tape in ("2026", "2025"):
        grid = results[tape]["grid"]
        base_pnl = results[tape]["baseline"]["pnl"]
        top = sorted(grid, key=lambda r: -(r["pnl"] - base_pnl))[:3]
        print(f"  {tape}:")
        for r in top:
            c = r["config"]
            delta = r["pnl"] - base_pnl
            print(f"    count={c['count']} window={c['window_min']}min cool={c['cooldown_min']}min"
                  f"  →  Δ ${delta:+,.2f}"
                  f"   (blocked {r['blocked_trades']} trades,"
                  f" would-have-been ${r['blocked_pnl_counterfactual']:+.0f})")


if __name__ == "__main__":
    main()
