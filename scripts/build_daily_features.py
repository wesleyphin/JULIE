#!/usr/bin/env python3
"""Build per-day feature + label dataset for the context-aware day classifier.

Features (computed from bar data only):
  intraday at 10:30:  range_10_30, drift_10_30, eff_10_30
  prior 5 sessions:   avg_abs_drift_5d, avg_range_5d, avg_eff_5d, n_breakout_5d
  week-to-date:       wtd_avg_range, wtd_avg_drift
  month-to-date:      mtd_avg_range, mtd_avg_abs_drift, mtd_days_breakout, mtd_days_chop
  prior calendar month: pm_avg_range, pm_avg_abs_drift, pm_dominant_category

Labels (only for days with closed_trades + iter-11 consistent regime log):
  oracle_variant  ∈ {"V0","V1","V3","V7","V8"}  (winner on that day by P&L)
  oracle_category ∈ {"chop","breakout","flat_calm","large_trend","moderate"}

Writes backtest_reports/daily_features.json.
"""
from __future__ import annotations

import json
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT / "scripts"))
from sim_dynamic_classifier import (  # noqa: E402
    load_bar_timeline, load_regime_timeline, parse_ts, run_day,
)
from reconstruct_regime import reconstruct_from_log  # noqa: E402

NY = ZoneInfo("America/New_York")
REPORT_ROOT = ROOT / "backtest_reports" / "full_live_replay"

# BAR SOURCES — scan every folder that has Bar: lines in its log.
# Plain monthly folders (2025_01, 2025_02, etc.) have continuous coverage.
BAR_SOURCES = [
    "2025_01", "2025_02", "2025_03", "2025_04", "2025_05", "2025_06",
    "2025_07", "2025_08", "2025_09", "2025_10", "2025_11", "2025_12",
    "outrageous_feb", "outrageous_jul", "outrageous_aug", "outrageous_oct",
    "outrageous_dec", "outrageous_apr",
    "2025_03_ny_iter11_deadtape", "2025_05_ny_iter11_deadtape",
    "2025_06_ny_iter11_deadtape",
]


def discover_external_bar_sources() -> list[Path]:
    """Return absolute paths to bar log folders outside full_live_replay/
    (e.g. iter11_h2_2025 and replay_mar2026 / replay_apr2026_p1)."""
    extras = []
    # H2 2025
    h2_root = ROOT / "backtest_reports" / "iter11_h2_2025"
    if h2_root.exists():
        for month_dir in sorted(h2_root.glob("2025_*")):
            for loop in sorted(month_dir.glob("live_loop_MES_*")):
                if (loop / "topstep_live_bot.log").exists():
                    extras.append(loop)
                    break
    # March 2026 bars-only pull (for April OOS context)
    bars_mar = ROOT / "backtest_reports" / "bars_mar2026"
    if bars_mar.exists() and (bars_mar / "topstep_live_bot.log").exists():
        extras.append(bars_mar)
    # April 2026 parts
    for folder_name in ("replay_apr2026_p1", "replay_mar2026"):
        root = ROOT / "backtest_reports" / folder_name
        if root.exists():
            for loop in sorted(root.glob("live_loop_MES_*")):
                if (loop / "topstep_live_bot.log").exists():
                    extras.append(loop)
                    break
    # Apr 20 warm replay
    warm = ROOT / "backtest_reports" / "replay_apr20" / "baseline_warm"
    if warm.exists():
        for loop in sorted(warm.glob("live_loop_MES_*")):
            if (loop / "topstep_live_bot.log").exists():
                extras.append(loop)
                break
    return extras

# LABEL SOURCES — iter-11 consistent runs where we have closed_trades.
# Those are the only days we can score for oracle_variant.
# The iter11_h2_2025/* folders are the H2 2025 monthly replays kicked off
# by scripts/run_h2_2025.sh — each contains a single live_loop_MES_* subfolder.
LABEL_SOURCES = [
    ("2025_03_ny_iter11_deadtape", "normal"),
    ("2025_05_ny_iter11_deadtape", "normal"),
    ("2025_06_ny_iter11_deadtape", "normal"),
    ("outrageous_feb", "outrageous"),
    ("outrageous_jul", "outrageous"),
    ("outrageous_aug", "outrageous"),
    ("outrageous_oct", "outrageous"),
    ("outrageous_dec", "outrageous"),
    ("outrageous_apr", "outrageous"),
    # H2 2025 from plain monthly folders (different config but trades look
    # compatible — DE3/RegimeAdaptive with similar sub_strategy shapes).
    # Regime transitions reconstructed offline from bar data.
    ("2025_01", "h1_plain"),
    ("2025_02", "h1_plain"),
    ("2025_07", "h2_plain"),
    ("2025_08", "h2_plain"),
    ("2025_09", "h2_plain"),
    ("2025_10", "h2_plain"),
    ("2025_11", "h2_plain"),
    ("2025_12", "h2_plain"),
]


def discover_h2_2025_sources() -> list[tuple[str, str]]:
    """Return (path_relative_to_REPORT_ROOT, source_tag) for Jul-Dec 2025
    iter-11 replays that completed. Each month produces a single
    live_loop_MES_* child folder under backtest_reports/iter11_h2_2025/2025_MM/.
    """
    out = []
    h2_root = ROOT / "backtest_reports" / "iter11_h2_2025"
    if not h2_root.exists():
        return out
    for month_dir in sorted(h2_root.glob("2025_*")):
        loops = sorted(month_dir.glob("live_loop_MES_*"))
        if not loops:
            continue
        loop = loops[-1]
        # Relative to REPORT_ROOT — but these are outside REPORT_ROOT. Use absolute
        # path in the source tuple.
        rel = loop.relative_to(ROOT / "backtest_reports" / "full_live_replay") \
            if str(loop).startswith(str(REPORT_ROOT)) else None
        if rel is None:
            out.append((str(loop), "h2_2025"))
    return out

VARIANTS_FOR_ORACLE = ("V0", "V1", "V3", "V7", "V8")


def classify_day_full(range_pct: float, drift_pct: float) -> str:
    r = abs(range_pct)
    d = abs(drift_pct)
    if r < 0.30:
        return "flat_calm"
    if d >= 1.5:
        return "breakout"
    if r >= 1.0 and d <= 0.4:
        return "chop"
    if d >= 0.6:
        return "large_trend"
    return "moderate"


def extract_bars(folder: Path) -> dict[str, list]:
    """Return {date_iso: [(naive_et_dt, price), ...]} NY-session bars."""
    return load_bar_timeline(folder / "topstep_live_bot.log")


def build_continuous_daily_ohlc() -> dict[str, dict]:
    """Merge bars from all BAR_SOURCES + external sources into a single
    per-day OHLC map. Later folders overwrite earlier ones on the same day;
    this is fine as long as the bar timestamps are consistent (they come
    from ProjectX).
    """
    by_day: dict[str, dict] = {}
    folders = [REPORT_ROOT / s for s in BAR_SOURCES]
    folders.extend(discover_external_bar_sources())
    for folder in folders:
        log = folder / "topstep_live_bot.log"
        if not log.exists():
            continue
        bars_by_day = extract_bars(folder)
        for day, bars in bars_by_day.items():
            if not bars:
                continue
            if day not in by_day:
                o = bars[0][1]
                hi = max(p for _, p in bars)
                lo = min(p for _, p in bars)
                c = bars[-1][1]
                by_day[day] = {
                    "open": o, "high": hi, "low": lo, "close": c,
                    "bar_count": len(bars),
                    "bars": bars,
                }
    return by_day


def intraday_snapshot_at(bars, minutes_elapsed: int = 60):
    if not bars:
        return None
    day_open = bars[0][1]
    session_start = bars[0][0]
    cutoff = session_start + timedelta(minutes=minutes_elapsed)
    window = [(t, p) for t, p in bars if t <= cutoff]
    if len(window) < 5:
        return None
    hi = max(p for _, p in window)
    lo = min(p for _, p in window)
    cur = window[-1][1]
    if day_open <= 0:
        return None
    rng = (hi - lo) / day_open * 100.0
    drift = (cur - day_open) / day_open * 100.0
    eff = abs(drift) / rng if rng > 0 else 0.0
    return {"range_pct": rng, "drift_pct": drift, "eff": eff}


def full_day_features(ohlc: dict) -> dict:
    o = ohlc["open"]
    if o <= 0:
        return {"range_pct": 0.0, "drift_pct": 0.0, "abs_drift_pct": 0.0, "eff": 0.0}
    r = (ohlc["high"] - ohlc["low"]) / o * 100.0
    d = (ohlc["close"] - o) / o * 100.0
    return {
        "range_pct": r,
        "drift_pct": d,
        "abs_drift_pct": abs(d),
        "eff": abs(d) / r if r > 0 else 0.0,
    }


def context_features(days_list: list, target_date: date, full_features_by_day: dict) -> dict:
    """Compute context features for target_date using prior days' full-day stats.

    days_list:   sorted list of all iso-date strings that have ohlc data
    target_date: python date object
    full_features_by_day: iso_date -> full-day features dict with category
    """
    # Prior 5 trading sessions
    prior_days = []
    i = bisect_right([d for d in days_list], target_date.isoformat()) - 1
    for j in range(max(0, i - 5), i):  # up to 5 prior days
        prior_days.append(days_list[j])
    prior_days = prior_days[-5:]

    if prior_days:
        prior5_abs_drift = mean(full_features_by_day[d]["abs_drift_pct"] for d in prior_days)
        prior5_range = mean(full_features_by_day[d]["range_pct"] for d in prior_days)
        prior5_eff = mean(full_features_by_day[d]["eff"] for d in prior_days)
        cats = Counter(full_features_by_day[d]["category"] for d in prior_days)
        prior5_breakout = cats.get("breakout", 0) / len(prior_days)
        prior5_chop = cats.get("chop", 0) / len(prior_days)
    else:
        prior5_abs_drift = 0.0
        prior5_range = 0.0
        prior5_eff = 0.0
        prior5_breakout = 0.0
        prior5_chop = 0.0

    # Week-to-date (Monday through target_date, exclusive)
    iso_weekday = target_date.weekday()  # 0=Mon, 6=Sun
    week_start = target_date - timedelta(days=iso_weekday)
    wtd_days = [d for d in days_list
                if week_start.isoformat() <= d < target_date.isoformat()]
    if wtd_days:
        wtd_abs_drift = mean(full_features_by_day[d]["abs_drift_pct"] for d in wtd_days)
        wtd_range = mean(full_features_by_day[d]["range_pct"] for d in wtd_days)
    else:
        wtd_abs_drift = 0.0
        wtd_range = 0.0

    # Month-to-date (first of month through target_date, exclusive)
    month_start = target_date.replace(day=1)
    mtd_days = [d for d in days_list
                if month_start.isoformat() <= d < target_date.isoformat()]
    if mtd_days:
        mtd_abs_drift = mean(full_features_by_day[d]["abs_drift_pct"] for d in mtd_days)
        mtd_range = mean(full_features_by_day[d]["range_pct"] for d in mtd_days)
        mtd_cats = Counter(full_features_by_day[d]["category"] for d in mtd_days)
        mtd_breakout_frac = mtd_cats.get("breakout", 0) / len(mtd_days)
        mtd_chop_frac = mtd_cats.get("chop", 0) / len(mtd_days)
    else:
        mtd_abs_drift = 0.0
        mtd_range = 0.0
        mtd_breakout_frac = 0.0
        mtd_chop_frac = 0.0

    # Prior calendar month
    if target_date.month == 1:
        pm_start = target_date.replace(year=target_date.year - 1, month=12, day=1)
        pm_end = target_date.replace(year=target_date.year - 1, month=12, day=31)
    else:
        pm_start = target_date.replace(month=target_date.month - 1, day=1)
        # last day of previous month = first of current month - 1 day
        pm_end = month_start - timedelta(days=1)
    pm_days = [d for d in days_list
               if pm_start.isoformat() <= d <= pm_end.isoformat()]
    if pm_days:
        pm_abs_drift = mean(full_features_by_day[d]["abs_drift_pct"] for d in pm_days)
        pm_range = mean(full_features_by_day[d]["range_pct"] for d in pm_days)
        pm_cats = Counter(full_features_by_day[d]["category"] for d in pm_days)
        pm_dominant = pm_cats.most_common(1)[0][0] if pm_cats else "unknown"
        pm_breakout_frac = pm_cats.get("breakout", 0) / len(pm_days)
        pm_chop_frac = pm_cats.get("chop", 0) / len(pm_days)
    else:
        pm_abs_drift = 0.0
        pm_range = 0.0
        pm_dominant = "unknown"
        pm_breakout_frac = 0.0
        pm_chop_frac = 0.0

    return {
        "prior5_abs_drift": round(prior5_abs_drift, 3),
        "prior5_range": round(prior5_range, 3),
        "prior5_eff": round(prior5_eff, 3),
        "prior5_breakout_frac": round(prior5_breakout, 3),
        "prior5_chop_frac": round(prior5_chop, 3),
        "wtd_abs_drift": round(wtd_abs_drift, 3),
        "wtd_range": round(wtd_range, 3),
        "mtd_abs_drift": round(mtd_abs_drift, 3),
        "mtd_range": round(mtd_range, 3),
        "mtd_breakout_frac": round(mtd_breakout_frac, 3),
        "mtd_chop_frac": round(mtd_chop_frac, 3),
        "pm_abs_drift": round(pm_abs_drift, 3),
        "pm_range": round(pm_range, 3),
        "pm_breakout_frac": round(pm_breakout_frac, 3),
        "pm_chop_frac": round(pm_chop_frac, 3),
        "pm_dominant": pm_dominant,
    }


def compute_oracle_label(trades, regime_events, bars):
    """Run each variant on this day's trades and return winner + per-variant."""
    results = {}
    for v in VARIANTS_FOR_ORACLE:
        r = run_day(trades, regime_events, bars, variant_strategy=v)
        results[v] = r["pnl"]
    winner = max(results, key=results.get)
    return {
        "oracle_variant": winner,
        "variant_pnls": results,
    }


def main():
    print("[load] building continuous daily OHLC from all 2025 bar logs...")
    ohlc_by_day = build_continuous_daily_ohlc()
    days_sorted = sorted(ohlc_by_day.keys())
    print(f"  {len(days_sorted)} days of bar data from 2025")

    # Compute full-day features + category for every day
    full_features_by_day = {}
    for day, o in ohlc_by_day.items():
        feats = full_day_features(o)
        feats["category"] = classify_day_full(feats["range_pct"], feats["drift_pct"])
        full_features_by_day[day] = feats

    # Build rows: per-day feature + label (label only where we have iter-11 trades)
    labeled_rows = []
    all_sources = list(LABEL_SOURCES) + discover_h2_2025_sources()
    for folder_name, source_tag in all_sources:
        # folder_name may be either (a) relative to REPORT_ROOT or (b) absolute path
        if folder_name.startswith("/"):
            folder = Path(folder_name)
        else:
            folder = REPORT_ROOT / folder_name
        if not (folder / "closed_trades.json").exists():
            print(f"  [skip] {folder}")
            continue
        trades = json.loads((folder / "closed_trades.json").read_text(encoding="utf-8"))
        regime_events = load_regime_timeline(folder / "topstep_live_bot.log")
        if not regime_events:
            # Old-config folder — no regime transitions logged. Reconstruct
            # from bars using the current classifier's logic.
            regime_events = reconstruct_from_log(folder / "topstep_live_bot.log")
        bars_by_day = extract_bars(folder)
        by_day = defaultdict(list)
        for t in trades:
            try:
                dt = parse_ts(t["entry_time"]).astimezone(NY).date().isoformat()
            except Exception:
                continue
            by_day[dt].append(t)
        for day, day_trades in sorted(by_day.items()):
            if day not in ohlc_by_day:
                continue
            day_trades.sort(key=lambda t: parse_ts(t["entry_time"]))
            bars = bars_by_day.get(day) or ohlc_by_day[day].get("bars", [])
            intraday = intraday_snapshot_at(bars, 60)
            if intraday is None:
                continue
            full_feats = full_features_by_day[day]
            td = datetime.fromisoformat(day).date()
            ctx = context_features(days_sorted, td, full_features_by_day)
            label = compute_oracle_label(day_trades, regime_events, bars)
            row = {
                "day": day,
                "source": source_tag,
                "folder": folder_name,
                "intraday_range": round(intraday["range_pct"], 3),
                "intraday_drift": round(intraday["drift_pct"], 3),
                "intraday_eff": round(intraday["eff"], 3),
                "full_range": round(full_feats["range_pct"], 3),
                "full_drift": round(full_feats["drift_pct"], 3),
                "full_category": full_feats["category"],
                **ctx,
                "oracle_variant": label["oracle_variant"],
                "variant_pnls": {k: round(v, 2) for k, v in label["variant_pnls"].items()},
            }
            labeled_rows.append(row)
    print(f"  {len(labeled_rows)} labeled rows (days with iter-11 trades + OHLC)")

    out_path = ROOT / "backtest_reports" / "daily_features.json"
    out_path.write_text(json.dumps({
        "all_days_ohlc": {d: full_features_by_day[d] for d in days_sorted},
        "labeled_rows": labeled_rows,
    }, indent=2), encoding="utf-8")
    print(f"\n[write] {out_path}")


if __name__ == "__main__":
    main()
