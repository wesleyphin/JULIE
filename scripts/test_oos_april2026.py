#!/usr/bin/env python3
"""Out-of-sample test: apply V_DYNAMIC to April 2026 TopStep historical replay.

Expects replay output at backtest_reports/replay_apr2026_p1/live_loop_MES_*/
(closed_trades.json + topstep_live_bot.log).  Can also accept additional
folders via argv.

Compares V1 (shipped), V3, and DYN against the baseline closed_trades.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT / "scripts"))
from sim_dynamic_classifier import (  # noqa: E402
    load_bar_timeline, load_regime_timeline, run_day, parse_ts,
)
from zoneinfo import ZoneInfo
from collections import defaultdict, Counter

NY = ZoneInfo("America/New_York")


def resolve_folders(argv):
    candidates = []
    if len(argv) > 1:
        for arg in argv[1:]:
            p = Path(arg)
            if p.is_dir():
                candidates.append(p)
    else:
        base = ROOT / "backtest_reports" / "replay_apr2026_p1"
        if base.is_dir():
            inner = sorted(base.glob("live_loop_MES_*"))
            if inner:
                candidates.append(inner[-1])
    # Also check whether the Apr 20 warm replay data should be included
    apr20_warm = ROOT / "backtest_reports" / "replay_apr20" / "baseline_warm"
    if apr20_warm.is_dir():
        inner = sorted(apr20_warm.glob("live_loop_MES_*"))
        if inner:
            candidates.append(inner[-1])
    return candidates


def run(folder: Path):
    trades = json.loads((folder / "closed_trades.json").read_text(encoding="utf-8"))
    regime_events = load_regime_timeline(folder / "topstep_live_bot.log")
    bars_by_day = load_bar_timeline(folder / "topstep_live_bot.log")

    by_day = defaultdict(list)
    for t in trades:
        try:
            dt = parse_ts(t["entry_time"]).astimezone(NY).date().isoformat()
        except Exception:
            continue
        by_day[dt].append(t)

    rows = []
    for day, day_trades in sorted(by_day.items()):
        day_trades.sort(key=lambda t: parse_ts(t["entry_time"]))
        bars = bars_by_day.get(day, [])
        res = {"day": day, "folder": folder.name, "n_trades": len(day_trades), "n_bars": len(bars)}
        for strategy in ("V0", "V1", "V3", "DYNAMIC"):
            r = run_day(day_trades, regime_events, bars, variant_strategy=strategy)
            res[f"{strategy}_pnl"] = r["pnl"]
            res[f"{strategy}_dd"] = r["max_dd"]
            res[f"{strategy}_dd_violation"] = r["dd_violation"]
            if strategy == "DYNAMIC":
                res["dyn_category"] = next(iter(r["predicted_categories"]), "unknown")
                res["dyn_variant"] = next(iter(r["variant_usage"]), "V1")
        rows.append(res)
    return rows


def summarize(rows):
    print("=" * 110)
    print(f"APRIL 2026 out-of-sample  ({len(rows)} days)")
    print("=" * 110)
    print(f"{'day':<12}{'trds':>5}  "
          f"{'V0':>9}{'V1':>9}{'V3':>9}{'DYN':>9}   "
          f"{'predicted cat/var':<22}   "
          f"{'V1 DD':>7}{'V3 DD':>7}{'DYN DD':>8}")
    print("-" * 110)
    for r in sorted(rows, key=lambda r: r["day"]):
        cat = r.get("dyn_category", "?")
        var = r.get("dyn_variant", "?")
        print(f"{r['day']:<12}{r['n_trades']:>5}  "
              f"{r['V0_pnl']:>+9.2f}{r['V1_pnl']:>+9.2f}"
              f"{r['V3_pnl']:>+9.2f}{r['DYNAMIC_pnl']:>+9.2f}   "
              f"{cat[:12]+'/'+var:<22}   "
              f"{r['V1_dd']:>7.2f}{r['V3_dd']:>7.2f}{r['DYNAMIC_dd']:>8.2f}")
    print("-" * 110)
    v0 = sum(r["V0_pnl"] for r in rows)
    v1 = sum(r["V1_pnl"] for r in rows)
    v3 = sum(r["V3_pnl"] for r in rows)
    vd = sum(r["DYNAMIC_pnl"] for r in rows)
    v0_vio = sum(r["V0_dd_violation"] for r in rows)
    v1_vio = sum(r["V1_dd_violation"] for r in rows)
    v3_vio = sum(r["V3_dd_violation"] for r in rows)
    vd_vio = sum(r["DYNAMIC_dd_violation"] for r in rows)
    print(f"TOTAL   V0 ${v0:+.2f} ({v0_vio})   V1 ${v1:+.2f} ({v1_vio})   "
          f"V3 ${v3:+.2f} ({v3_vio})   DYN ${vd:+.2f} ({vd_vio})")
    print()
    print(f"DYN vs V1 (shipped):  ${vd - v1:+.2f}   DD delta: {vd_vio - v1_vio:+d}")
    print(f"DYN vs V3 (best static): ${vd - v3:+.2f}   DD delta: {vd_vio - v3_vio:+d}")

    # Category distribution
    cat_counts = Counter(r.get("dyn_category", "?") for r in rows)
    var_counts = Counter(r.get("dyn_variant", "?") for r in rows)
    print(f"\nDYN category distribution: {dict(cat_counts)}")
    print(f"DYN variant assignments:   {dict(var_counts)}")


if __name__ == "__main__":
    folders = resolve_folders(sys.argv)
    if not folders:
        print("No replay folders found.")
        print("  Expected: backtest_reports/replay_apr2026_p1/live_loop_MES_*/ or pass paths as args")
        sys.exit(1)
    all_rows = []
    for f in folders:
        print(f"[load] {f}")
        all_rows.extend(run(f))
    # Dedupe by day (prefer last folder's data for a given day)
    dedup: dict[str, dict] = {}
    for r in all_rows:
        dedup[r["day"]] = r
    all_rows = list(dedup.values())
    summarize(all_rows)
    out = ROOT / "backtest_reports" / "test_oos_april2026.json"
    out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")
