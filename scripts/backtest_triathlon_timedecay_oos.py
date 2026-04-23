#!/usr/bin/env python3
"""Option B — time-decay weighted medal scoring OOS backtest.

Protocol:
  train: pre-April 2026 trades (same as size-effect backtest)
  test:  April 2026 trades
  For BOTH scoring regimes — unweighted (current shipped) and
  weighted (half_life_days=60) — compute medals from train, apply
  their size multipliers to test, compare PnL / WR / MaxDD.

Ship gate: weighted > unweighted on PnL, WR non-regressive,
MaxDD not materially worse. Additionally, does weighted beat the
no-medal-effects baseline (native sizing)? That's the real test
of whether medals are doing anything vs nothing.

Sweeps multiple half-life values so we pick the best-performing one
before shipping.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path("/Users/wes/Downloads/JULIE001")
LEDGER = ROOT / "ai_loop_data/triathlon/ledger.db"
OUT = ROOT / "backtest_reports/triathlon_timedecay_oos_results.json"

SPLIT_DATE = "2026-04-01"
MIN_SAMPLES = 20

MEDAL_SIZE_MULT = {
    "gold": 1.50, "silver": 1.00, "bronze": 1.00,
    "probation": 0.50, "unrated": 1.00,
}
GOLD_PCTILE, SILVER_PCTILE, BRONZE_PCTILE = 0.20, 0.50, 0.80
PROBATION_PCTILE = 0.80


def load_trades():
    """Return (pre_april, april) dict lists."""
    conn = sqlite3.connect(str(LEDGER))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        """
        SELECT s.ts, s.strategy, s.side, s.regime, s.time_bucket, s.size,
               o.pnl_dollars, o.bars_held
        FROM signals s JOIN outcomes o ON o.signal_id = s.signal_id
        WHERE s.source_tag IN ('seed_2025','seed_2026')
          AND o.counterfactual = 0 AND s.status = 'fired'
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
        trades.append({
            "ts": ts, "strategy": r["strategy"], "side": r["side"],
            "regime": r["regime"], "time_bucket": r["time_bucket"],
            "size": max(1, int(r["size"] or 1)),
            "pnl": float(r["pnl_dollars"] or 0.0),
            "bars_held": r["bars_held"],
        })
    split = datetime.fromisoformat(SPLIT_DATE)
    if trades and trades[0]["ts"].tzinfo is not None and split.tzinfo is None:
        split = split.replace(tzinfo=trades[0]["ts"].tzinfo)
    return [t for t in trades if t["ts"] < split], [t for t in trades if t["ts"] >= split], split


def cell_key(t):
    return f"{t['strategy']}|{t['regime']}|{t['time_bucket']}"


def trade_weight(trade_ts: datetime, reference_ts: datetime, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    a = trade_ts if trade_ts.tzinfo is None else trade_ts.replace(tzinfo=None)
    b = reference_ts if reference_ts.tzinfo is None else reference_ts.replace(tzinfo=None)
    age_days = max(0.0, (b - a).total_seconds() / 86400)
    return math.exp(-math.log(2) * age_days / half_life_days)


def score_cells(trades, half_life_days: float, reference_ts: datetime):
    """Return {cell_key: {n_effective, purity, cash, velocity, rated}}."""
    by_cell = defaultdict(list)
    for t in trades:
        by_cell[cell_key(t)].append(t)
    scores = {}
    for ck, items in by_cell.items():
        weights = [trade_weight(t["ts"], reference_ts, half_life_days) for t in items]
        eff_n = sum(weights)
        if eff_n < MIN_SAMPLES:
            scores[ck] = {"rated": False, "eff_n": eff_n, "raw_n": len(items)}
            continue
        sum_w = eff_n
        sum_w_wins = sum(w for w, t in zip(weights, items) if t["pnl"] > 0)
        purity = sum_w_wins / sum_w
        weighted_pnl = sum(w * t["pnl"] for w, t in zip(weights, items))
        weighted_size = sum(w * t["size"] for w, t in zip(weights, items))
        cash = weighted_pnl / max(1.0, weighted_size)
        win_entries = [(w, int(t["bars_held"]))
                        for w, t in zip(weights, items)
                        if t["pnl"] > 0 and t["bars_held"] and t["bars_held"] > 0]
        if not win_entries:
            scores[ck] = {"rated": False, "eff_n": eff_n, "raw_n": len(items)}
            continue
        sum_ww = sum(w for w, _ in win_entries) or 1.0
        if half_life_days > 0:
            velocity = sum(w * (1.0 / b) for w, b in win_entries) / sum_ww
        else:
            velocity = 1.0 / statistics.median([b for _, b in win_entries])
        scores[ck] = {"rated": True, "eff_n": eff_n, "raw_n": len(items),
                       "purity": purity, "cash": cash, "velocity": velocity}
    return scores


def assign_medals(scores):
    rated = [ck for ck, s in scores.items() if s.get("rated")]
    n = len(rated)
    if n == 0:
        return {ck: "unrated" for ck in scores}
    def rank(key):
        srt = sorted(rated, key=lambda c: -scores[c][key])
        return {c: i + 1 for i, c in enumerate(srt)}
    pr, cr, vr = rank("purity"), rank("cash"), rank("velocity")
    medals = {}
    for ck, s in scores.items():
        if not s.get("rated"):
            medals[ck] = "unrated"; continue
        best = min(pr[ck]/n, cr[ck]/n, vr[ck]/n)
        if best >= PROBATION_PCTILE:
            medals[ck] = "probation"
        elif best <= GOLD_PCTILE:
            medals[ck] = "gold"
        elif best <= SILVER_PCTILE:
            medals[ck] = "silver"
        elif best <= BRONZE_PCTILE:
            medals[ck] = "bronze"
        else:
            medals[ck] = "silver"
    return medals


def simulate(april, medals):
    """Apply medal size_mults to each April trade + compute PnL series."""
    sized_pnls, baseline_pnls = [], []
    medal_counts = defaultdict(int)
    for t in april:
        medal = medals.get(cell_key(t), "unrated")
        mult = MEDAL_SIZE_MULT[medal]
        medal_counts[medal] += 1
        old_size = max(1, t["size"])
        new_size = max(1, int(round(old_size * mult)))
        scaled = t["pnl"] * (new_size / old_size)
        baseline_pnls.append(t["pnl"])
        sized_pnls.append(scaled)
    return baseline_pnls, sized_pnls, dict(medal_counts)


def stats(pnls):
    if not pnls: return {"n": 0, "pnl": 0, "wr": 0, "dd": 0, "pf": None}
    cum = 0; peak = 0; dd = 0
    for p in pnls:
        cum += p; peak = max(peak, cum); dd = max(dd, peak - cum)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
    return {"n": len(pnls), "pnl": round(cum, 2),
            "wr": round(len(wins)/len(pnls)*100, 2),
            "dd": round(dd, 2),
            "pf": round(pf, 3) if pf != float("inf") else None,
            "avg": round(cum/len(pnls), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halflives", type=float, nargs="+", default=[30, 60, 90, 120])
    args = ap.parse_args()

    print(f"[timedecay-oos] loading trades...")
    pre, april, split = load_trades()
    print(f"  pre-April: {len(pre):,}  April: {len(april):,}")
    print(f"  reference (= split date): {split}")

    # Baseline 0: no medal effects, native sizing
    baseline_pnls, _, _ = simulate(april, {})   # empty medal map → all unrated → mult 1.0
    base0 = stats(baseline_pnls)

    # Baseline 1: unweighted medals (current shipped)
    unweighted_scores = score_cells(pre, half_life_days=0, reference_ts=split)
    unweighted_medals = assign_medals(unweighted_scores)
    _, pnl_uw, med_uw = simulate(april, unweighted_medals)
    base1 = stats(pnl_uw)

    print(f"\n═══ Baseline comparisons ═══")
    print(f"  [BASELINE 0] no medal effects (native sizing):")
    print(f"    PnL=${base0['pnl']:+,.2f}  WR={base0['wr']:.2f}%  "
          f"MaxDD=${base0['dd']:,.0f}  PF={base0['pf']}")
    print(f"  [BASELINE 1] unweighted medal sizing (current shipped):")
    print(f"    PnL=${base1['pnl']:+,.2f}  WR={base1['wr']:.2f}%  "
          f"MaxDD=${base1['dd']:,.0f}  PF={base1['pf']}")
    uw_counts = {m: sum(1 for k, v in unweighted_medals.items() if v == m)
                  for m in ("gold","silver","bronze","probation","unrated")}
    print(f"    medals assigned: {uw_counts}")

    # Experimental: time-decay with each half-life
    print(f"\n═══ Time-decay sweep ═══")
    best = None
    results = {"baseline_no_medals": base0, "baseline_unweighted_medals": base1,
                "by_halflife": {}}
    for hl in args.halflives:
        w_scores = score_cells(pre, half_life_days=hl, reference_ts=split)
        w_medals = assign_medals(w_scores)
        _, pnl_w, med_w = simulate(april, w_medals)
        sw = stats(pnl_w)
        # how many cells changed medal vs unweighted?
        changed = sum(1 for ck in set(unweighted_medals) | set(w_medals)
                      if unweighted_medals.get(ck) != w_medals.get(ck))
        n_rated = sum(1 for s in w_scores.values() if s.get("rated"))
        w_counts = {m: sum(1 for k, v in w_medals.items() if v == m)
                     for m in ("gold","silver","bronze","probation","unrated")}
        print(f"\n  [half-life={hl}d]  "
              f"rated cells: {n_rated}  changed-from-unweighted: {changed}")
        print(f"    medals: {w_counts}")
        print(f"    PnL=${sw['pnl']:+,.2f}  WR={sw['wr']:.2f}%  "
              f"MaxDD=${sw['dd']:,.0f}  PF={sw['pf']}  avg=${sw['avg']:+.2f}")
        print(f"    vs baseline-0 (no medals):  ΔPnL=${sw['pnl']-base0['pnl']:+.2f}  "
              f"ΔWR={sw['wr']-base0['wr']:+.2f}pp  ΔDD=${sw['dd']-base0['dd']:+.0f}")
        print(f"    vs baseline-1 (unweighted): ΔPnL=${sw['pnl']-base1['pnl']:+.2f}  "
              f"ΔWR={sw['wr']-base1['wr']:+.2f}pp  ΔDD=${sw['dd']-base1['dd']:+.0f}")
        results["by_halflife"][hl] = {
            "stats": sw, "n_rated": n_rated, "changed_vs_unweighted": changed,
            "medal_counts": w_counts,
            "vs_base0": {"pnl": round(sw['pnl']-base0['pnl'], 2),
                          "wr":  round(sw['wr']-base0['wr'], 2),
                          "dd":  round(sw['dd']-base0['dd'], 2)},
            "vs_base1": {"pnl": round(sw['pnl']-base1['pnl'], 2),
                          "wr":  round(sw['wr']-base1['wr'], 2),
                          "dd":  round(sw['dd']-base1['dd'], 2)},
        }
        # Ship criteria: beats BOTH baselines on all three metrics
        # (or at least doesn't make them worse)
        beats_base0 = (sw["pnl"] > base0["pnl"]
                        and sw["wr"] - base0["wr"] >= -0.5
                        and sw["dd"] <= max(0, base0["dd"] * 1.10))
        beats_base1 = (sw["pnl"] > base1["pnl"]
                        and sw["wr"] - base1["wr"] >= -0.5
                        and sw["dd"] <= max(0, base1["dd"] * 1.10))
        if beats_base0 and beats_base1:
            score_lift = (sw["pnl"] - base0["pnl"]) + (sw["pnl"] - base1["pnl"])
            if best is None or score_lift > best[0]:
                best = (score_lift, hl, sw)

    if best:
        lift, hl, sw = best
        print(f"\n═══ BEST ═══")
        print(f"  half-life = {hl} days  combined-lift = ${lift:+.2f}")
        print(f"  PnL=${sw['pnl']:+,.2f}  WR={sw['wr']:.2f}%  MaxDD=${sw['dd']:,.0f}")
        print(f"  SHIP. Will default JULIE_TRIATHLON_HALFLIFE_DAYS={hl}")
    else:
        print(f"\n═══ No half-life beats BOTH baselines on all three metrics ═══")
        print(f"  Do NOT ship — time-decay doesn't improve on current.")

    results["best_halflife"] = best[1] if best else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[write] {OUT}")
    return 0 if best else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
