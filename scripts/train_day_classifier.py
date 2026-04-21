#!/usr/bin/env python3
"""Train/tune the day classifier by checking accuracy on 2025 known days.

Approach:
  1. For every 2025 day, compute intraday snapshot at 10:30 ET:
       range_so_far_pct, drift_so_far_pct, eff_so_far
  2. Full-day category is the label (from load_daily_ohlc full-day OHLC).
  3. Test various threshold sets — find thresholds that maximize
     classifier accuracy (predicted_10:30 matches actual_full_day).
  4. Also report by-category precision/recall.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/Users/wes/Downloads/JULIE001")
REPORT_ROOT = ROOT / "backtest_reports" / "full_live_replay"
NY = ZoneInfo("America/New_York")

RGX_BAR = re.compile(
    r"Bar: (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<price>[\d.]+)"
)

SOURCES = [
    "2025_03_ny_iter11_deadtape",
    "2025_05_ny_iter11_deadtape",
    "2025_06_ny_iter11_deadtape",
    "outrageous_feb",
    "outrageous_jul",
    "outrageous_aug",
    "outrageous_oct",
    "outrageous_dec",
    "outrageous_apr",
]


def load_bars_by_day(log_path: Path):
    by_day: dict[str, list] = defaultdict(list)
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RGX_BAR.search(line)
            if not m:
                continue
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
            mins = ts.hour * 60 + ts.minute
            if not (9 * 60 + 30 <= mins <= 16 * 60):
                continue
            by_day[ts.date().isoformat()].append((ts, float(m.group("price"))))
    for d in by_day:
        by_day[d].sort(key=lambda x: x[0])
    return by_day


def compute_snapshot(bars, minutes_into_session: int):
    """Return (range_pct, drift_pct_signed, eff) using bars from open through
    open + minutes_into_session. If not enough bars, return None."""
    if not bars:
        return None
    session_start = bars[0][0]
    day_open = bars[0][1]
    cutoff = session_start.replace(minute=session_start.minute)
    from datetime import timedelta
    cutoff = session_start + timedelta(minutes=minutes_into_session)
    window = [(t, p) for t, p in bars if t <= cutoff]
    if len(window) < 5:  # need at least 5 bars
        return None
    hi = max(p for _, p in window)
    lo = min(p for _, p in window)
    cur = window[-1][1]
    rng = (hi - lo) / day_open * 100.0
    drift = (cur - day_open) / day_open * 100.0
    eff = abs(drift) / rng if rng > 0 else 0.0
    return rng, drift, eff


def full_day_category(bars):
    if not bars:
        return "unknown"
    o = bars[0][1]
    hi = max(p for _, p in bars)
    lo = min(p for _, p in bars)
    close = bars[-1][1]
    r = (hi - lo) / o * 100.0
    d = (close - o) / o * 100.0
    ad = abs(d)
    if r < 0.30:
        return "flat_calm"
    if ad >= 1.5:
        return "breakout"
    if r >= 1.0 and ad <= 0.4:
        return "chop"
    if ad >= 0.6:
        return "large_trend"
    return "moderate"


def classify_at_1030(rng, drift, eff, *, minutes_elapsed=60, thresholds=None):
    """Classifier under test — uses simple threshold rules."""
    th = thresholds or dict(
        BREAK_RNG=0.80, BREAK_DRIFT=0.60,
        CHOP_RNG=0.60, CHOP_DRIFT_MAX=0.20,
        FLAT_RNG=0.18,
        LARGE_DRIFT=0.30,
    )
    scale = 60.0 / max(minutes_elapsed, 1)
    rng *= scale
    drift_signed = drift * scale
    drift = abs(drift_signed)
    if drift >= th["BREAK_DRIFT"] and rng >= th["BREAK_RNG"]:
        return "breakout"
    if rng >= th["CHOP_RNG"] and drift <= th["CHOP_DRIFT_MAX"]:
        return "chop"
    if rng < th["FLAT_RNG"]:
        return "flat_calm"
    if drift >= th["LARGE_DRIFT"] and rng < th["BREAK_RNG"]:
        return "large_trend"
    return "moderate"


def build_dataset():
    rows = []
    for folder_name in SOURCES:
        folder = REPORT_ROOT / folder_name
        bars_by_day = load_bars_by_day(folder / "topstep_live_bot.log")
        for day, bars in sorted(bars_by_day.items()):
            snap = compute_snapshot(bars, 60)
            if snap is None:
                continue
            rng, drift, eff = snap
            actual = full_day_category(bars)
            rows.append({
                "day": day,
                "folder": folder_name,
                "rng_at_1030": round(rng, 3),
                "drift_at_1030": round(drift, 3),
                "eff_at_1030": round(eff, 3),
                "actual_category": actual,
            })
    return rows


def confusion_matrix(rows, thresholds=None):
    mat = defaultdict(lambda: defaultdict(int))
    for r in rows:
        pred = classify_at_1030(
            r["rng_at_1030"], r["drift_at_1030"], r["eff_at_1030"],
            minutes_elapsed=60, thresholds=thresholds,
        )
        mat[r["actual_category"]][pred] += 1
    return mat


def accuracy(rows, thresholds=None):
    hits = 0
    for r in rows:
        pred = classify_at_1030(
            r["rng_at_1030"], r["drift_at_1030"], r["eff_at_1030"],
            minutes_elapsed=60, thresholds=thresholds,
        )
        if pred == r["actual_category"]:
            hits += 1
    return hits / len(rows) if rows else 0.0


def variant_accuracy(rows, thresholds=None):
    """What matters: does predicted variant match the oracle variant?

    Oracle variant comes from per-category winner (see sim_subcategorize).
    """
    cat_to_variant = {
        "breakout": "V0",
        "chop": "V1",
        "flat_calm": "V1",
        "large_trend": "V1",
        "moderate": "V3",
    }
    hits = 0
    total = 0
    for r in rows:
        actual_var = cat_to_variant.get(r["actual_category"], "V1")
        pred_cat = classify_at_1030(
            r["rng_at_1030"], r["drift_at_1030"], r["eff_at_1030"],
            minutes_elapsed=60, thresholds=thresholds,
        )
        pred_var = cat_to_variant.get(pred_cat, "V1")
        if pred_var == actual_var:
            hits += 1
        total += 1
    return hits / total if total else 0.0


def grid_search(rows):
    """Simple grid search over threshold values."""
    best_acc = 0.0
    best_th = None
    best_var_acc = 0.0
    for br in (0.50, 0.65, 0.80, 1.00, 1.20):
        for bd in (0.30, 0.45, 0.60, 0.80):
            for cr in (0.40, 0.60, 0.80, 1.00):
                for cdm in (0.10, 0.20, 0.30):
                    for fr in (0.10, 0.18, 0.25):
                        for ld in (0.20, 0.30, 0.45):
                            th = dict(
                                BREAK_RNG=br, BREAK_DRIFT=bd,
                                CHOP_RNG=cr, CHOP_DRIFT_MAX=cdm,
                                FLAT_RNG=fr,
                                LARGE_DRIFT=ld,
                            )
                            acc = accuracy(rows, thresholds=th)
                            var_acc = variant_accuracy(rows, thresholds=th)
                            # score: weight variant accuracy higher (that's what matters for P&L)
                            score = 0.7 * var_acc + 0.3 * acc
                            if score > best_acc:
                                best_acc = score
                                best_th = th
                                best_var_acc = var_acc
    return best_th, best_acc, best_var_acc


if __name__ == "__main__":
    print("[load] building 2025 snapshots @ 10:30...")
    rows = build_dataset()
    print(f"  {len(rows)} days with usable 10:30 snapshots")

    # Baseline accuracy with default thresholds
    base_acc = accuracy(rows)
    base_var = variant_accuracy(rows)
    print(f"\nDefault thresholds: full-category accuracy={base_acc:.1%}  variant accuracy={base_var:.1%}")

    print("\n[grid-search] over 6x4x4x3x3x3 = 1296 threshold combos...")
    best_th, best_score, best_var = grid_search(rows)
    best_acc = accuracy(rows, thresholds=best_th)
    print(f"  best score=0.7*var+0.3*cat = {best_score:.1%}")
    print(f"  best variant-accuracy = {best_var:.1%}")
    print(f"  best full-category accuracy = {best_acc:.1%}")
    print(f"  thresholds: {best_th}")

    # Confusion on the best thresholds
    cm = confusion_matrix(rows, thresholds=best_th)
    all_cats = sorted({r["actual_category"] for r in rows})
    print(f"\nConfusion matrix (actual rows x predicted cols):")
    hdr = "actual\\pred".ljust(13) + "".join(f"{c[:10]:>12}" for c in all_cats) + "  total"
    print(hdr)
    for a in all_cats:
        row_total = sum(cm[a][p] for p in all_cats)
        line = a[:12].ljust(13) + "".join(f"{cm[a][p]:>12}" for p in all_cats) + f"  {row_total:>5}"
        print(line)

    # Save tuned thresholds
    out = ROOT / "backtest_reports" / "day_classifier_thresholds.json"
    out.write_text(json.dumps({
        "thresholds": best_th,
        "variant_accuracy": best_var,
        "full_cat_accuracy": best_acc,
    }, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")
