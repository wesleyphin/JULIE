#!/usr/bin/env python3
"""Subcategorize days by tape-shape features and evaluate D variants per subcategory.

For every trade-day in the normal + outrageous sets, extract OHLC from the
topstep_live_bot.log "Bar:" lines, then classify as:

  chop          — range >= 1% of price AND |drift| <= 0.4%  (big range, no net)
  breakout      — |drift| >= 1.5%                           (strong directional)
  large_trend   — |drift| in [0.6%, 1.5%)                   (moderate-strong drift)
  flat_calm     — range < 0.3%                              (no-action days)
  moderate      — everything else                            (ordinary tape)

Then run variants V0/V1/V3/V4 (from sim_rev_skip_variants.py) per subcategory
and print a matrix: which variant wins each subcategory by P&L?

Use the winning variant per subcategory to propose a real-time policy driven
by the regime classifier's regime label + context.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path("/Users/wes/Downloads/JULIE001")
REPORT_ROOT = ROOT / "backtest_reports" / "full_live_replay"
NY = ZoneInfo("America/New_York")

REGIME_CAPPED = {"whipsaw", "calm_trend"}
UNLOCK_THRESHOLD = 200.0
UNLOCK_SIZE = 3
CAP_SIZE = 1
DD_LIMIT = 350.0

SOURCES = [
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
RGX_TRANSITION = re.compile(
    r"Regime transition: \S+ -> (?P<regime>\S+).* ts=(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^ \|]*)"
)
RGX_BAR = re.compile(
    r"Bar: (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<price>[\d.]+)"
)

# V7: within calm_trend, eff >= EFF_SPLIT means "trend is real" (skip Rev cap),
# eff < EFF_SPLIT means "fake trend / large_trend" (cap Rev like V1).
EFF_SPLIT_CALM_TREND = 0.18


def parse_ts(s):
    return datetime.fromisoformat(s.strip())


def load_regime_timeline(log_path):
    """Return list of (ts, regime, vol_bp, eff) sorted by ts."""
    events = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Regime transition" not in line:
                continue
            m = RGX_TRANSITION_FULL.search(line)
            if m:
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
                continue
            # Fallback: line w/o vol/eff
            m = RGX_TRANSITION.search(line)
            if not m:
                continue
            regime = m.group("regime").strip()
            if regime == "dead_tape":
                regime = "neutral"
            try:
                ts = parse_ts(m.group("ts"))
            except Exception:
                continue
            events.append((ts, regime, 0.0, 0.0))
    events.sort(key=lambda x: x[0])
    return events


def load_daily_ohlc(log_path) -> dict[str, dict]:
    """Extract per-day OHLC + NY-session range/drift from Bar: log lines.
    Keyed by ET date iso string. Uses 09:30-16:00 ET bars (NY session).
    Also stores day_open price so simulator can compute intraday drift."""
    by_day: dict[str, dict] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RGX_BAR.search(line)
            if not m:
                continue
            bar_ts_naive = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
            # Logs tag bars with "ET" suffix — treat as ET naive
            t = bar_ts_naive
            # Only NY session bars
            if not (9 * 60 + 30 <= t.hour * 60 + t.minute <= 16 * 60):
                continue
            date = t.date().isoformat()
            price = float(m.group("price"))
            row = by_day.setdefault(date, {
                "open": price, "high": price, "low": price, "close": price,
                "bars": 0, "day_open": price,
            })
            row["high"] = max(row["high"], price)
            row["low"] = min(row["low"], price)
            row["close"] = price
            row["bars"] += 1
    for date, row in by_day.items():
        o = row["open"]
        if o > 0:
            row["range_pct"] = round((row["high"] - row["low"]) / o * 100.0, 3)
            row["drift_pct"] = round((row["close"] - o) / o * 100.0, 3)
            row["net_eff"] = round(abs(row["drift_pct"]) / row["range_pct"], 3) if row["range_pct"] > 0 else 0.0
        else:
            row["range_pct"] = 0.0
            row["drift_pct"] = 0.0
            row["net_eff"] = 0.0
    return by_day


def classify_day(ohlc: dict) -> str:
    r = ohlc["range_pct"]
    d = abs(ohlc["drift_pct"])
    if r < 0.30:
        return "flat_calm"
    if d >= 1.5:
        return "breakout"
    if r >= 1.0 and d <= 0.4:
        return "chop"
    if d >= 0.6:
        return "large_trend"
    return "moderate"


def regime_at(ts, events):
    """Return (regime, vol_bp, eff) at ts."""
    if not events:
        return "warmup", 0.0, 0.0
    keys = [e[0] for e in events]
    i = bisect_right(keys, ts) - 1
    if i < 0:
        return "warmup", 0.0, 0.0
    return events[i][1], events[i][2], events[i][3]


def run_day(trades, regime_events, *, variant, day_open: float = 0.0):
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        size = int(t.get("size", 1) or 1)
        pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
        per_contract = pnl / size if size > 0 else pnl
        et = parse_ts(t["entry_time"])
        regime, vol_bp, eff = regime_at(et, regime_events)
        capped_regime = regime in REGIME_CAPPED
        sub = str(t.get("sub_strategy", ""))
        is_rev = "_Rev_" in sub

        should_cap = capped_regime and size > CAP_SIZE
        if variant == "V0":
            should_cap = False
        elif variant == "V1":
            pass
        elif variant == "V3":
            if should_cap and is_rev and regime == "calm_trend":
                should_cap = False
        elif variant == "V4":
            pass
        elif variant == "V7":
            # Split calm_trend by eff: high-eff = real trend (cap all like V1),
            # low-eff = fake/large_trend-like (cap Rev like V1), but calm_trend
            # w/ moderate eff + Mom skips cap only when trade aligns with drift.
            # Simpler form: skip Rev cap only when calm_trend AND eff >= split.
            if should_cap and is_rev and regime == "calm_trend" and eff >= EFF_SPLIT_CALM_TREND:
                should_cap = False
        elif variant == "V8":
            # V3 but only skip Rev cap when cum_sim is already positive (trust
            # "proven winners" of the day). Protects fresh-loss days.
            if should_cap and is_rev and regime == "calm_trend" and cum > 0:
                should_cap = False
        elif variant == "VB":
            # V_BEST: hybrid of the winning per-subcategory rules.
            # Only intervene on calm_trend + Rev shape; whipsaw stays fully
            # capped, neutral is already no-cap since not capped_regime.
            if should_cap and is_rev and regime == "calm_trend":
                # Skip the cap only if ONE of:
                #   - eff is high (>=0.18): real trend, Rev rides the pullback
                #   - cum_pnl already positive: day has proved itself
                if eff >= 0.18 or cum > 0:
                    should_cap = False
        elif variant == "VF":
            # V_FINAL — intraday-drift-aware hybrid.
            # Uses drift_so_far = (entry_price - day_open) / day_open.
            # Skip Rev cap on calm_trend when:
            #   - |drift_so_far| >= 1.5% (real breakout confirmed)
            #   - OR cum_pnl > 0 (day proved itself)
            # Mom on calm_trend and everything on whipsaw stays capped.
            if should_cap and is_rev and regime == "calm_trend":
                try:
                    entry_px = float(t.get("entry_price", 0.0) or 0.0)
                except Exception:
                    entry_px = 0.0
                drift_pct = 0.0
                if day_open > 0 and entry_px > 0:
                    drift_pct = (entry_px - day_open) / day_open * 100.0
                if abs(drift_pct) >= 1.5 or cum > 0:
                    should_cap = False

        if should_cap:
            if variant == "V4" and is_rev:
                effective_cap = 2
            else:
                effective_cap = UNLOCK_SIZE if cum >= UNLOCK_THRESHOLD else CAP_SIZE
            new_size = min(size, effective_cap)
            trade_pnl = per_contract * new_size
        else:
            trade_pnl = pnl

        cum += trade_pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return round(cum, 2), round(max_dd, 2)


def main():
    # Per-day rows with classification + per-variant result
    rows = []
    for folder_name, superset in SOURCES:
        folder = REPORT_ROOT / folder_name
        trades = json.loads((folder / "closed_trades.json").read_text(encoding="utf-8"))
        regime_events = load_regime_timeline(folder / "topstep_live_bot.log")
        ohlc_by_day = load_daily_ohlc(folder / "topstep_live_bot.log")
        by_day = defaultdict(list)
        for t in trades:
            try:
                dt_str = parse_ts(t["entry_time"]).astimezone(NY).date().isoformat()
            except Exception:
                continue
            by_day[dt_str].append(t)
        for day, day_trades in sorted(by_day.items()):
            day_trades.sort(key=lambda t: parse_ts(t["entry_time"]))
            ohlc = ohlc_by_day.get(day, {"range_pct": 0, "drift_pct": 0, "net_eff": 0})
            category = classify_day(ohlc) if ohlc.get("bars", 0) > 0 else "unknown"
            row = {
                "day": day,
                "source": superset,
                "category": category,
                "range_pct": ohlc.get("range_pct", 0),
                "drift_pct": ohlc.get("drift_pct", 0),
                "net_eff": ohlc.get("net_eff", 0),
                "n_trades": len(day_trades),
            }
            day_open = ohlc.get("day_open", 0.0)
            for variant in ("V0", "V1", "V3", "V4", "V7", "V8", "VB", "VF"):
                pnl, dd = run_day(day_trades, regime_events, variant=variant, day_open=day_open)
                row[f"{variant}_pnl"] = pnl
                row[f"{variant}_dd"] = dd
                row[f"{variant}_dd_violation"] = 1 if dd > DD_LIMIT else 0
            rows.append(row)

    # Aggregate by category (cross superset + category) + per-superset subcategory
    print("=" * 100)
    print("Per-day classification (sample)")
    print("=" * 100)
    print(f"{'day':<12}{'src':<11}{'cat':<13}{'range%':>8}{'drift%':>8}{'eff':>7}{'V1 pnl':>10}{'V3 pnl':>10}")
    for r in rows[:10]:
        print(f"{r['day']:<12}{r['source']:<11}{r['category']:<13}"
              f"{r['range_pct']:>8.2f}{r['drift_pct']:>+8.2f}{r['net_eff']:>7.2f}"
              f"{r['V1_pnl']:>+10.2f}{r['V3_pnl']:>+10.2f}")
    print("...")
    print()

    # Distribution of categories per superset
    print("=" * 100)
    print("Day distribution by category")
    print("=" * 100)
    from collections import Counter
    dist = defaultdict(Counter)
    for r in rows:
        dist[r["source"]][r["category"]] += 1
    for src in ("normal", "outrageous"):
        total = sum(dist[src].values())
        print(f"  {src} (n={total}): " + ", ".join(f"{c}={n}" for c, n in dist[src].most_common()))
    print()

    # Aggregate P&L and DD violations per (superset, category)
    print("=" * 100)
    print("Per-subcategory: variant P&L and DD violations")
    print("=" * 100)
    VARIANTS = ("V0", "V1", "V3", "V4", "V7", "V8", "VB", "VF")
    header = f"{'set':<11}{'category':<13}{'n':>4}  " + \
             "".join(f"{v + ' $':>10}" for v in VARIANTS) + "  best"
    print(header)
    print("-" * len(header))
    agg = defaultdict(lambda: defaultdict(lambda: {"pnl": 0.0, "dd_viol": 0, "n": 0}))
    for r in rows:
        key = (r["source"], r["category"])
        for v in VARIANTS:
            agg[key][v]["pnl"] += r[f"{v}_pnl"]
            agg[key][v]["dd_viol"] += r[f"{v}_dd_violation"]
            agg[key][v]["n"] += 1
    for src in ("normal", "outrageous"):
        cats = sorted({r["category"] for r in rows if r["source"] == src})
        for cat in cats:
            key = (src, cat)
            a = agg[key]
            n = a["V0"]["n"]
            best_variant = max(VARIANTS, key=lambda v: a[v]["pnl"])
            line = f"{src:<11}{cat:<13}{n:>4}  "
            for v in VARIANTS:
                line += f"{a[v]['pnl']:>+10.2f}"
            line += f"  {best_variant}"
            print(line)

    # Build OPTIMAL policy: per (source, category) pick best V and apply to those days
    print()
    print("=" * 100)
    print("If we could magically pick the best variant per subcategory (oracle):")
    print("=" * 100)
    oracle_total = 0.0
    oracle_dd_viol = 0
    for key, v_map in agg.items():
        best = max(v_map, key=lambda v: v_map[v]["pnl"])
        oracle_total += v_map[best]["pnl"]
        oracle_dd_viol += v_map[best]["dd_viol"]
    print(f"Oracle (perfect per-subcategory): ${oracle_total:+.2f}   DD viol: {oracle_dd_viol}")
    print()
    print("Shipped / candidate totals across all 136 days:")
    for v in VARIANTS:
        total = sum(r[f"{v}_pnl"] for r in rows)
        dd_ct = sum(r[f"{v}_dd_violation"] for r in rows)
        print(f"  {v}: ${total:+.2f}   DD viol: {dd_ct}")

    # Save full rows
    out = ROOT / "backtest_reports" / "sim_subcategorize_days.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
