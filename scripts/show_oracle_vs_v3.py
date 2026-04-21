#!/usr/bin/env python3
"""Compare Oracle (per-subcategory best) vs V3 to pinpoint where V3 leaks money."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/wes/Downloads/JULIE001")
DATA = ROOT / "backtest_reports" / "sim_subcategorize_days.json"

VARIANTS = ("V0", "V1", "V3", "V4", "V7", "V8", "VB", "VF")


def main():
    rows = json.loads(DATA.read_text(encoding="utf-8"))

    # Aggregate per (source, category)
    agg = defaultdict(lambda: defaultdict(lambda: {"pnl": 0.0, "dd_viol": 0, "n": 0}))
    per_day = defaultdict(list)  # (source, cat) -> list of day rows
    for r in rows:
        key = (r["source"], r["category"])
        per_day[key].append(r)
        for v in VARIANTS:
            agg[key][v]["pnl"] += r[f"{v}_pnl"]
            agg[key][v]["dd_viol"] += r[f"{v}_dd_violation"]
            agg[key][v]["n"] += 1

    # Print Oracle vs V3 per subcategory
    print("=" * 110)
    print(f"{'set':<11}{'category':<14}{'n':>4}  "
          f"{'Oracle $':>11}{'(variant)':<10}   "
          f"{'V3 $':>11}   {'gap':>11}   {'DD: oracle / V3':<20}")
    print("-" * 110)

    oracle_total = 0.0
    v3_total = 0.0
    oracle_dd = 0
    v3_dd = 0
    gaps = []

    for (src, cat), variants in sorted(agg.items()):
        best = max(VARIANTS, key=lambda v: variants[v]["pnl"])
        oracle_pnl = variants[best]["pnl"]
        oracle_dd_cat = variants[best]["dd_viol"]
        v3_pnl = variants["V3"]["pnl"]
        v3_dd_cat = variants["V3"]["dd_viol"]
        n = variants["V3"]["n"]
        gap = oracle_pnl - v3_pnl

        oracle_total += oracle_pnl
        v3_total += v3_pnl
        oracle_dd += oracle_dd_cat
        v3_dd += v3_dd_cat
        gaps.append({
            "subcategory": f"{src} {cat}",
            "n": n,
            "best_variant": best,
            "oracle_pnl": oracle_pnl,
            "v3_pnl": v3_pnl,
            "gap": gap,
            "oracle_dd": oracle_dd_cat,
            "v3_dd": v3_dd_cat,
        })
        print(
            f"{src:<11}{cat:<14}{n:>4}  "
            f"{oracle_pnl:>+11.2f}  ({best:<3})      "
            f"{v3_pnl:>+11.2f}   {gap:>+11.2f}   {oracle_dd_cat}/{v3_dd_cat}"
        )

    print("-" * 110)
    print(f"{'TOTALS':<11}{'':<14}{'':<4}  "
          f"{oracle_total:>+11.2f}  {'':<10} "
          f"{v3_total:>+11.2f}   {oracle_total - v3_total:>+11.2f}   {oracle_dd}/{v3_dd}")

    # Sort gaps by magnitude and show how V3 could close them
    print()
    print("=" * 110)
    print("Biggest gaps (where V3 leaves money on the table):")
    print("=" * 110)
    for g in sorted(gaps, key=lambda x: -x["gap"]):
        if g["gap"] < 0.01:
            continue
        print(f"  {g['subcategory']:<25} n={g['n']:>3}  "
              f"gap=${g['gap']:>+8.2f}  "
              f"V3 does {g['v3_pnl']:+.2f}, {g['best_variant']} does {g['oracle_pnl']:+.2f}")

    # Look at the biggest gap (outrageous large_trend) in detail
    print()
    print("=" * 110)
    print("DRILL-DOWN: outrageous large_trend — why does V3 lose so badly?")
    print("=" * 110)
    key = ("outrageous", "large_trend")
    days = sorted(per_day[key], key=lambda r: r["V3_pnl"] - r["V1_pnl"])
    print(f"{'day':<12}{'range%':>8}{'drift%':>8}{'eff':>7}{'trds':>5}   "
          f"{'V0':>9}{'V1':>9}{'V3':>9}   {'V3-V1':>9}   {'V3 DD':>8}")
    for r in days[:15]:  # 15 worst (V3 - V1)
        diff = r["V3_pnl"] - r["V1_pnl"]
        print(f"{r['day']:<12}{r['range_pct']:>8.2f}{r['drift_pct']:>+8.2f}"
              f"{r['net_eff']:>7.2f}{r['n_trades']:>5}   "
              f"{r['V0_pnl']:>+9.2f}{r['V1_pnl']:>+9.2f}{r['V3_pnl']:>+9.2f}   "
              f"{diff:>+9.2f}   {r['V3_dd']:>8.2f}")
    print("  ...")
    for r in days[-5:]:  # 5 best (V3 - V1)
        diff = r["V3_pnl"] - r["V1_pnl"]
        print(f"{r['day']:<12}{r['range_pct']:>8.2f}{r['drift_pct']:>+8.2f}"
              f"{r['net_eff']:>7.2f}{r['n_trades']:>5}   "
              f"{r['V0_pnl']:>+9.2f}{r['V1_pnl']:>+9.2f}{r['V3_pnl']:>+9.2f}   "
              f"{diff:>+9.2f}   {r['V3_dd']:>8.2f}")

    # Distinguishing features analysis — what correlates with V3 < V1 vs V3 > V1?
    print()
    print("=" * 110)
    print("FEATURE CORRELATION: within outrageous large_trend, does any feature predict V3 behavior?")
    print("=" * 110)
    v3_wins = [r for r in days if r["V3_pnl"] > r["V1_pnl"]]
    v3_loses = [r for r in days if r["V3_pnl"] < r["V1_pnl"]]
    v3_ties = [r for r in days if abs(r["V3_pnl"] - r["V1_pnl"]) < 0.01]

    def stats(rs, label):
        if not rs:
            print(f"  {label}: (none)")
            return
        r_avg = sum(r["range_pct"] for r in rs) / len(rs)
        d_avg = sum(abs(r["drift_pct"]) for r in rs) / len(rs)
        e_avg = sum(r["net_eff"] for r in rs) / len(rs)
        t_avg = sum(r["n_trades"] for r in rs) / len(rs)
        print(f"  {label:<22} n={len(rs):>3}   range%={r_avg:>5.2f}   "
              f"|drift%|={d_avg:>5.2f}   eff={e_avg:>5.2f}   trades={t_avg:>5.1f}")

    stats(v3_wins, "V3 wins vs V1")
    stats(v3_loses, "V3 loses vs V1")
    stats(v3_ties, "V3 == V1")


if __name__ == "__main__":
    main()
