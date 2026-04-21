#!/usr/bin/env python3
"""Simulate V_DYNAMIC: day-classifier predicts category -> applies best variant.

Extends sim_subcategorize_days.py: for every day, at each trade's entry time
we compute intraday range/drift-so-far and run classify_day_intraday, then
apply the variant that wins for the predicted category.

This tests whether a REAL-TIME-computable classifier can capture meaningful
upside beyond V3 (the best static variant) and close some of the gap to the
hindsight-oracle.
"""
from __future__ import annotations

import json
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT / "scripts"))
from day_classifier import classify_day_intraday, variant_for_category  # noqa: E402

REPORT_ROOT = ROOT / "backtest_reports" / "full_live_replay"
NY = ZoneInfo("America/New_York")

REGIME_CAPPED = {"whipsaw", "calm_trend"}
UNLOCK_THRESHOLD = 200.0
UNLOCK_SIZE = 3
CAP_SIZE = 1
DD_LIMIT = 350.0
EFF_SPLIT_CALM_TREND = 0.18

DEFAULT_SOURCES = [
    ("2025_03_ny_iter11_deadtape", "normal"),
    ("2025_05_ny_iter11_deadtape", "normal"),
    ("2025_06_ny_iter11_deadtape", "normal"),
    ("outrageous_feb", "outrageous"),
    ("outrageous_jul", "outrageous"),
    ("outrageous_aug", "outrageous"),
    ("outrageous_oct", "outrageous"),
    ("outrageous_dec", "outrageous"),
    ("outrageous_apr", "outrageous"),
]

RGX_TRANSITION_FULL = re.compile(
    r"Regime transition: \S+ -> (?P<regime>\S+) \| vol=(?P<vol>[\d.]+)bp eff=(?P<eff>[\d.]+) .* ts=(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^ \|]*)"
)
RGX_BAR = re.compile(
    r"Bar: (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<price>[\d.]+)"
)


def parse_ts(s):
    return datetime.fromisoformat(s.strip())


def load_regime_timeline(log_path: Path):
    events = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Regime transition" not in line:
                continue
            m = RGX_TRANSITION_FULL.search(line)
            if not m:
                continue
            regime = m.group("regime").strip()
            if regime == "dead_tape":
                regime = "neutral"
            try:
                ts = parse_ts(m.group("ts"))
                vol = float(m.group("vol"))
                eff = float(m.group("eff"))
            except Exception:
                continue
            events.append((ts, regime, vol, eff))
    events.sort(key=lambda x: x[0])
    return events


def load_bar_timeline(log_path: Path):
    """Return {date_iso: [(naive_et_time, price), ...]} NY-session bars only."""
    bars_by_day: dict[str, list] = defaultdict(list)
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RGX_BAR.search(line)
            if not m:
                continue
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
            mins = ts.hour * 60 + ts.minute
            if not (9 * 60 + 30 <= mins <= 16 * 60):
                continue
            bars_by_day[ts.date().isoformat()].append((ts, float(m.group("price"))))
    for d in bars_by_day:
        bars_by_day[d].sort(key=lambda x: x[0])
    return bars_by_day


def regime_at(ts, events):
    if not events:
        return "warmup", 0.0, 0.0
    keys = [e[0] for e in events]
    i = bisect_right(keys, ts) - 1
    if i < 0:
        return "warmup", 0.0, 0.0
    return events[i][1], events[i][2], events[i][3]


def intraday_features_at(bars_for_day, trade_et_naive):
    """Return (range_pct_so_far, drift_pct_so_far, eff_so_far, minutes_elapsed)
    using bars from 09:30 up to trade_et_naive (exclusive).

    trade_et_naive is a naive datetime in ET.  Strip tz before calling.
    """
    if not bars_for_day:
        return 0.0, 0.0, 0.0, 0
    day_open = bars_for_day[0][1]
    session_start = bars_for_day[0][0]
    so_far = [b for b in bars_for_day if b[0] <= trade_et_naive]
    if not so_far:
        return 0.0, 0.0, 0.0, 0
    hi = max(p for _, p in so_far)
    lo = min(p for _, p in so_far)
    cur = so_far[-1][1]
    if day_open <= 0:
        return 0.0, 0.0, 0.0, 0
    rng = (hi - lo) / day_open * 100.0
    drift = (cur - day_open) / day_open * 100.0
    eff = abs(drift) / rng if rng > 0 else 0.0
    minutes = max(1, int((so_far[-1][0] - session_start).total_seconds() / 60))
    return rng, drift, eff, minutes


def apply_variant(t, *, variant, regime, eff, cum, is_rev, size):
    """Return new_size for this trade under the given variant."""
    capped_regime = regime in REGIME_CAPPED
    should_cap = capped_regime and size > CAP_SIZE

    if variant == "V0":
        should_cap = False
    elif variant == "V1":
        pass
    elif variant == "V3":
        if should_cap and is_rev and regime == "calm_trend":
            should_cap = False
    elif variant == "V7":
        if should_cap and is_rev and regime == "calm_trend" and eff >= EFF_SPLIT_CALM_TREND:
            should_cap = False
    elif variant == "V8":
        if should_cap and is_rev and regime == "calm_trend" and cum > 0:
            should_cap = False
    # else fall through to V1

    if should_cap:
        effective_cap = UNLOCK_SIZE if cum >= UNLOCK_THRESHOLD else CAP_SIZE
        return min(size, effective_cap)
    return size


def run_day(trades, regime_events, bars_for_day, *, variant_strategy):
    """variant_strategy is either a fixed string ('V0','V1','V3', etc.)
    or the string 'DYNAMIC' which uses the classifier.

    DYNAMIC classifies the day ONCE at 10:30 ET (or earliest possible) and
    applies that variant uniformly across all trades that day — more stable
    than per-trade re-classification.
    """
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    variant_usage = Counter()
    predicted_categories = Counter()
    trade_log = []

    # Classify once at 10:30 ET using bars up to that time.
    locked_variant = None
    locked_category = None
    if variant_strategy == "DYNAMIC" and bars_for_day:
        from datetime import timedelta
        session_start = bars_for_day[0][0]
        cutoff = session_start + timedelta(minutes=60)  # bars through 10:30
        rng, drift, eff_so_far, mins = intraday_features_at(bars_for_day, cutoff)
        if mins >= 5:  # enough bars for a meaningful snapshot
            locked_category = classify_day_intraday(rng, drift, eff_so_far, mins)
            locked_variant = variant_for_category(locked_category)
            predicted_categories[locked_category] += 1
        else:
            locked_variant = "V1"  # fallback on sparse data

    for t in trades:
        size = int(t.get("size", 1) or 1)
        pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
        per_contract = pnl / size if size > 0 else pnl

        et_aware = parse_ts(t["entry_time"])
        et_naive = et_aware.astimezone(NY).replace(tzinfo=None)

        regime, vol_bp, eff_at_trade = regime_at(et_aware, regime_events)
        sub = str(t.get("sub_strategy", ""))
        is_rev = "_Rev_" in sub

        if variant_strategy == "DYNAMIC":
            variant = locked_variant or "V1"
        else:
            variant = variant_strategy

        variant_usage[variant] += 1
        new_size = apply_variant(
            t, variant=variant, regime=regime, eff=eff_at_trade, cum=cum,
            is_rev=is_rev, size=size,
        )
        trade_pnl = per_contract * new_size
        cum += trade_pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        trade_log.append({
            "time": et_naive.strftime("%H:%M"),
            "sub": sub[:30],
            "size_orig": size, "size_new": new_size,
            "pnl_orig": round(pnl, 2), "pnl_after": round(trade_pnl, 2),
            "variant": variant,
        })

    return {
        "pnl": round(cum, 2),
        "max_dd": round(max_dd, 2),
        "dd_violation": 1 if max_dd > DD_LIMIT else 0,
        "variant_usage": dict(variant_usage),
        "predicted_categories": dict(predicted_categories),
        "trade_log": trade_log,
    }


def run_set(sources):
    out_rows = []
    for folder_name, source_tag in sources:
        folder = REPORT_ROOT / folder_name
        if not folder.exists():
            print(f"[skip] {folder}")
            continue
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
        for day, day_trades in sorted(by_day.items()):
            day_trades.sort(key=lambda t: parse_ts(t["entry_time"]))
            bars = bars_by_day.get(day, [])
            res = {"day": day, "source": source_tag, "folder": folder_name,
                   "n_trades": len(day_trades), "n_bars": len(bars)}
            for strategy in ("V0", "V1", "V3", "DYNAMIC"):
                r = run_day(day_trades, regime_events, bars, variant_strategy=strategy)
                res[f"{strategy}_pnl"] = r["pnl"]
                res[f"{strategy}_dd"] = r["max_dd"]
                res[f"{strategy}_dd_violation"] = r["dd_violation"]
                if strategy == "DYNAMIC":
                    res["dyn_variant_usage"] = r["variant_usage"]
                    res["dyn_predicted_categories"] = r["predicted_categories"]
                    res["dyn_trade_log"] = r["trade_log"]
            out_rows.append(res)
    return out_rows


def summarize(rows, name):
    print("=" * 90)
    print(f"{name}  ({len(rows)} days)")
    print("=" * 90)
    print(f"{'day':<12}{'src':<12}{'trds':>5}  "
          f"{'V0':>9}{'V1':>9}{'V3':>9}{'DYN':>9}  "
          f"{'dyn categories':<30}")
    for r in sorted(rows, key=lambda r: r["day"]):
        cats = r.get("dyn_predicted_categories", {})
        cats_str = ",".join(f"{k}:{v}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
        print(f"{r['day']:<12}{r['source']:<12}{r['n_trades']:>5}  "
              f"{r['V0_pnl']:>+9.2f}{r['V1_pnl']:>+9.2f}{r['V3_pnl']:>+9.2f}"
              f"{r['DYNAMIC_pnl']:>+9.2f}  {cats_str[:28]}")
    print()
    v0 = sum(r["V0_pnl"] for r in rows)
    v1 = sum(r["V1_pnl"] for r in rows)
    v3 = sum(r["V3_pnl"] for r in rows)
    vd = sum(r["DYNAMIC_pnl"] for r in rows)
    v0_vio = sum(r["V0_dd_violation"] for r in rows)
    v1_vio = sum(r["V1_dd_violation"] for r in rows)
    v3_vio = sum(r["V3_dd_violation"] for r in rows)
    vd_vio = sum(r["DYNAMIC_dd_violation"] for r in rows)
    print(f"TOTAL   V0 ${v0:+.2f} ({v0_vio} viol)   V1 ${v1:+.2f} ({v1_vio} viol)   "
          f"V3 ${v3:+.2f} ({v3_vio} viol)   DYN ${vd:+.2f} ({vd_vio} viol)")
    print(f"        DYN vs V1: ${vd - v1:+.2f}    DYN vs V3: ${vd - v3:+.2f}")

    # Variant usage aggregated
    total_usage = Counter()
    for r in rows:
        for v, n in r.get("dyn_variant_usage", {}).items():
            total_usage[v] += n
    total_cats = Counter()
    for r in rows:
        for c, n in r.get("dyn_predicted_categories", {}).items():
            total_cats[c] += n
    print(f"DYN variant usage across all trades: {dict(total_usage)}")
    print(f"DYN predicted categories:            {dict(total_cats)}")


if __name__ == "__main__":
    rows = run_set(DEFAULT_SOURCES)
    summarize(rows, "2025 training set (in-sample)")
    out = ROOT / "backtest_reports" / "sim_dynamic_classifier.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")
