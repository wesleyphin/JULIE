#!/usr/bin/env python3
"""Apply context-aware classifier to April 2026 days and report OOS results.

Builds features for each April 2026 day using:
 - Intraday 10:30 ET snapshot from replay bar logs
 - Context features from the combined 2025+2026 OHLC dataset

Then uses the trained classifier (context_classifier.pkl) to pick a variant
per day, simulates that variant, and compares to V1/V3/Oracle.
"""
from __future__ import annotations

import json
import pickle
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT / "scripts"))
from sim_dynamic_classifier import load_bar_timeline, load_regime_timeline, parse_ts, run_day  # noqa: E402
from reconstruct_regime import reconstruct_from_log  # noqa: E402
from build_daily_features import (  # noqa: E402
    build_continuous_daily_ohlc, full_day_features, context_features,
    intraday_snapshot_at, classify_day_full, compute_oracle_label,
)
from train_context_classifier import FEATURE_NAMES, to_feature_vector  # noqa: E402

NY = ZoneInfo("America/New_York")

APRIL_2026_SOURCES = [
    ROOT / "backtest_reports" / "replay_apr2026_p1",
    ROOT / "backtest_reports" / "replay_apr20" / "baseline_warm",
]


def find_replay_folders(roots):
    out = []
    for root in roots:
        if not root.exists():
            continue
        loops = sorted(root.glob("live_loop_MES_*"))
        if loops:
            out.append(loops[-1])
    return out


def load_classifier():
    with (ROOT / "backtest_reports" / "context_classifier.pkl").open("rb") as fh:
        return pickle.load(fh)


def main():
    clf_data = load_classifier()
    model = clf_data["model"]
    print(f"[load] classifier max_depth={clf_data['best_depth']}, min_leaf={clf_data['best_leaf']}")

    folders = find_replay_folders(APRIL_2026_SOURCES)
    print(f"[load] {len(folders)} April 2026 replay folders: {[f.name for f in folders]}")

    # Rebuild ohlc + full_features for all 2025 days + April 2026
    print("[load] building OHLC (will include March 2026 if present)...")
    ohlc_by_day = build_continuous_daily_ohlc()
    # Also add April 2026 bar data
    for folder in folders:
        bars_by_day = load_bar_timeline(folder / "topstep_live_bot.log")
        for day, bars in bars_by_day.items():
            if day in ohlc_by_day:
                continue
            if not bars:
                continue
            o = bars[0][1]
            hi = max(p for _, p in bars)
            lo = min(p for _, p in bars)
            c = bars[-1][1]
            ohlc_by_day[day] = {
                "open": o, "high": hi, "low": lo, "close": c,
                "bar_count": len(bars), "bars": bars,
            }
    days_sorted = sorted(ohlc_by_day.keys())
    full_features_by_day = {}
    for day, o in ohlc_by_day.items():
        ff = full_day_features(o)
        ff["category"] = classify_day_full(ff["range_pct"], ff["drift_pct"])
        full_features_by_day[day] = ff
    print(f"  {len(days_sorted)} total days with OHLC through {days_sorted[-1]}")
    # Check Mar 2026 coverage
    mar_2026_days = [d for d in days_sorted if d.startswith("2026-03")]
    print(f"  Mar 2026 days available: {len(mar_2026_days)}")
    apr_2026_days = [d for d in days_sorted if d.startswith("2026-04")]
    print(f"  Apr 2026 days available: {len(apr_2026_days)}")

    # Per-day classification + simulation for each April 2026 day
    results_by_day = {}
    for folder in folders:
        trades = json.loads((folder / "closed_trades.json").read_text(encoding="utf-8"))
        regime_events = load_regime_timeline(folder / "topstep_live_bot.log")
        if not regime_events:
            regime_events = reconstruct_from_log(folder / "topstep_live_bot.log")
        bars_by_day = load_bar_timeline(folder / "topstep_live_bot.log")
        trades_by_day = defaultdict(list)
        for t in trades:
            try:
                d = parse_ts(t["entry_time"]).astimezone(NY).date().isoformat()
            except Exception:
                continue
            trades_by_day[d].append(t)
        for day, day_trades in sorted(trades_by_day.items()):
            if not day.startswith("2026-04"):
                continue
            if day in results_by_day:
                continue  # prefer first folder's version
            day_trades.sort(key=lambda t: parse_ts(t["entry_time"]))
            bars = bars_by_day.get(day, [])
            intraday = intraday_snapshot_at(bars, 60)
            if intraday is None or day not in full_features_by_day:
                continue
            td = datetime.fromisoformat(day).date()
            ctx = context_features(days_sorted, td, full_features_by_day)
            row = {
                "day": day,
                "intraday_range": round(intraday["range_pct"], 3),
                "intraday_drift": round(intraday["drift_pct"], 3),
                "intraday_eff": round(intraday["eff"], 3),
                **ctx,
            }
            # Classify
            feat_vec = np.array([to_feature_vector(row)])
            pred = str(model.predict(feat_vec)[0])
            row["predicted_variant"] = pred
            # Simulate the predicted variant + V1/V3/Oracle
            label_info = compute_oracle_label(day_trades, regime_events, bars)
            row["variant_pnls"] = label_info["variant_pnls"]
            row["oracle_variant"] = label_info["oracle_variant"]
            # Also run each variant for DD info
            for v in ("V0", "V1", "V3", "V7", "V8"):
                r = run_day(day_trades, regime_events, bars, variant_strategy=v)
                row[f"{v}_dd"] = r["max_dd"]
                row[f"{v}_dd_violation"] = r["dd_violation"]
            results_by_day[day] = row

    rows = sorted(results_by_day.values(), key=lambda r: r["day"])
    # Print table
    print("\n" + "=" * 120)
    print("APRIL 2026 OOS — context classifier vs V1/V3/Oracle")
    print("=" * 120)
    hdr = f"{'day':<12}{'n_trds':>6}  " + \
          f"{'V0 $':>9}{'V1 $':>9}{'V3 $':>9}{'pred':>5} {'pred $':>9}  " + \
          f"{'oracle':>6} {'oracle $':>10}  {'context':<40}"
    print(hdr)
    print("-" * 120)
    sum_v0 = sum_v1 = sum_v3 = sum_pred = sum_oracle = 0.0
    for r in rows:
        p = r["variant_pnls"]
        pred = r["predicted_variant"]
        pred_pnl = p[pred]
        sum_v0 += p["V0"]; sum_v1 += p["V1"]; sum_v3 += p["V3"]
        sum_pred += pred_pnl; sum_oracle += p[r["oracle_variant"]]
        ctx_brief = (f"pm_dom={r.get('pm_dominant','?')[:10]} "
                     f"prior5_rng={r.get('prior5_range',0):.2f} "
                     f"prior5_eff={r.get('prior5_eff',0):.2f}")
        print(
            f"{r['day']:<12}{'':<6}  "
            f"{p['V0']:>+9.2f}{p['V1']:>+9.2f}{p['V3']:>+9.2f}"
            f"{pred:>5} {pred_pnl:>+9.2f}  "
            f"{r['oracle_variant']:>6} {p[r['oracle_variant']]:>+10.2f}  "
            f"{ctx_brief:<40}"
        )
    print("-" * 120)
    print(f"{'TOTAL':<12}{'':<6}  "
          f"{sum_v0:>+9.2f}{sum_v1:>+9.2f}{sum_v3:>+9.2f}"
          f"{'PRED':>5} {sum_pred:>+9.2f}  "
          f"{'ORACLE':>6} {sum_oracle:>+10.2f}")
    print(f"\nClassifier vs V1: ${sum_pred - sum_v1:+.2f}")
    print(f"Classifier vs V3: ${sum_pred - sum_v3:+.2f}")
    print(f"Classifier vs Oracle: ${sum_pred - sum_oracle:+.2f}")
    # Distribution
    pred_counter = Counter(r["predicted_variant"] for r in rows)
    print(f"\nPredicted variant distribution: {dict(pred_counter)}")
    print(f"Oracle variant distribution:    {dict(Counter(r['oracle_variant'] for r in rows))}")

    # Save
    out = ROOT / "backtest_reports" / "oos_april2026_context.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
