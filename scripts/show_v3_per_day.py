#!/usr/bin/env python3
"""Show V3 (skip Rev caps on calm_trend) results day-by-day with V1 comparison."""
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

NORMAL = [
    ("2025_03_ny_iter11_deadtape", "normal"),
    ("2025_05_ny_iter11_deadtape", "normal"),
    ("2025_06_ny_iter11_deadtape", "normal"),
]
OUTRAGEOUS = [
    ("outrageous_feb", "outrageous"),
    ("outrageous_jul", "outrageous"),
    ("outrageous_aug", "outrageous"),
    ("outrageous_oct", "outrageous"),
    ("outrageous_dec", "outrageous"),
    ("outrageous_apr", "outrageous"),
]

RGX_TRANSITION = re.compile(
    r"Regime transition: \S+ -> (?P<regime>\S+).* ts=(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^ \|]*)"
)


def parse_ts(s):
    return datetime.fromisoformat(s.strip())


def load_regime_timeline(log_path):
    events = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Regime transition" not in line:
                continue
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
            events.append((ts, regime))
    events.sort(key=lambda x: x[0])
    return events


def regime_at(ts, events):
    if not events:
        return "warmup"
    keys = [e[0] for e in events]
    i = bisect_right(keys, ts) - 1
    if i < 0:
        return "warmup"
    return events[i][1]


def run_day(trades, regime_events, *, variant):
    """Return (net_pnl, max_dd, per_trade_list) for one day."""
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    trade_log = []
    for t in trades:
        size = int(t.get("size", 1) or 1)
        pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
        per_contract = pnl / size if size > 0 else pnl
        et = parse_ts(t["entry_time"])
        regime = regime_at(et, regime_events)
        capped_regime = regime in REGIME_CAPPED
        sub = str(t.get("sub_strategy", ""))
        is_rev = "_Rev_" in sub

        should_cap = capped_regime and size > CAP_SIZE
        if variant == "V0":
            should_cap = False
        elif variant == "V1":
            pass  # cap everything in capped regime
        elif variant == "V3":
            # Skip Rev caps only on calm_trend (keep whipsaw-Rev caps)
            if should_cap and is_rev and regime == "calm_trend":
                should_cap = False

        if should_cap:
            effective_cap = UNLOCK_SIZE if cum >= UNLOCK_THRESHOLD else CAP_SIZE
            new_size = min(size, effective_cap)
            trade_pnl = per_contract * new_size
            capped_flag = True
        else:
            new_size = size
            trade_pnl = pnl
            capped_flag = False

        cum += trade_pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
        trade_log.append({
            "et": et.strftime("%H:%M"),
            "sub": sub,
            "side": t.get("side", ""),
            "shape": "Rev" if is_rev else ("Mom" if "_Mom_" in sub else "?"),
            "regime": regime,
            "size_orig": size,
            "size_after": new_size,
            "pnl_orig": round(pnl, 2),
            "pnl_after": round(trade_pnl, 2),
            "capped": capped_flag,
        })
    return round(cum, 2), round(max_dd, 2), trade_log


def run_source_set(source_list):
    days_out = []  # list of {day, source, v0, v1, v3, trades}
    for folder_name, source_label in source_list:
        folder = REPORT_ROOT / folder_name
        trades = json.loads((folder / "closed_trades.json").read_text(encoding="utf-8"))
        regime_events = load_regime_timeline(folder / "topstep_live_bot.log")
        by_day = defaultdict(list)
        for t in trades:
            try:
                dt = parse_ts(t["entry_time"]).astimezone(NY).date().isoformat()
            except Exception:
                continue
            by_day[dt].append(t)
        for day in sorted(by_day):
            dt_trades = sorted(by_day[day], key=lambda t: parse_ts(t["entry_time"]))
            v0_pnl, v0_dd, _ = run_day(dt_trades, regime_events, variant="V0")
            v1_pnl, v1_dd, _ = run_day(dt_trades, regime_events, variant="V1")
            v3_pnl, v3_dd, v3_trades = run_day(dt_trades, regime_events, variant="V3")
            days_out.append({
                "day": day,
                "source": source_label,
                "folder": folder_name,
                "n_trades": len(dt_trades),
                "v0_pnl": v0_pnl, "v0_dd": v0_dd,
                "v1_pnl": v1_pnl, "v1_dd": v1_dd,
                "v3_pnl": v3_pnl, "v3_dd": v3_dd,
                "v3_trades": v3_trades,
            })
    return days_out


def fmt_line(d):
    marker_v1 = "*" if d["v1_dd"] > 350 else " "
    marker_v3 = "*" if d["v3_dd"] > 350 else " "
    diff = d["v3_pnl"] - d["v1_pnl"]
    diff_flag = "→" if abs(diff) < 0.01 else ("↑" if diff > 0 else "↓")
    return (
        f"{d['day']:<12}{d['source'][:4]:<5}{d['n_trades']:>4}  "
        f"V0:{d['v0_pnl']:>+8.2f}/{d['v0_dd']:>6.2f}  "
        f"V1:{d['v1_pnl']:>+8.2f}/{d['v1_dd']:>6.2f}{marker_v1} "
        f"V3:{d['v3_pnl']:>+8.2f}/{d['v3_dd']:>6.2f}{marker_v3}  "
        f"{diff_flag}{diff:>+8.2f}"
    )


if __name__ == "__main__":
    normal = run_source_set(NORMAL)
    outrageous = run_source_set(OUTRAGEOUS)

    print("=" * 110)
    print("Column key: V0 = no D | V1 = D caps all (shipped) | V3 = D skips Rev on calm_trend")
    print("   each cell: P&L / max-DD  |  * marks DD>\$350 violation  |  Δ is V3-V1 per-day delta")
    print("=" * 110)

    print("\n### NORMAL (Mar/May/Jun 2025) ###")
    print(f"{'day':<12}{'src':<5}{'trds':>4}  {'V0 pnl/dd':<20}  {'V1 pnl/dd':<21}  {'V3 pnl/dd':<21}  {'V3-V1':>11}")
    print("-" * 110)
    for d in normal:
        print(fmt_line(d))

    print("\n### OUTRAGEOUS (Feb/Apr/Jul/Aug/Oct/Dec 2025) ###")
    print(f"{'day':<12}{'src':<5}{'trds':>4}  {'V0 pnl/dd':<20}  {'V1 pnl/dd':<21}  {'V3 pnl/dd':<21}  {'V3-V1':>11}")
    print("-" * 110)
    for d in outrageous:
        print(fmt_line(d))

    # Totals
    def tot(rows, key):
        return round(sum(r[key] for r in rows), 2)

    print("\n" + "=" * 110)
    print("TOTALS")
    print("=" * 110)
    for label, rows in [("NORMAL", normal), ("OUTRAGEOUS", outrageous)]:
        v1_viol = sum(1 for r in rows if r["v1_dd"] > 350)
        v3_viol = sum(1 for r in rows if r["v3_dd"] > 350)
        print(
            f"{label:<12} n={len(rows)}  "
            f"V0 ${tot(rows, 'v0_pnl'):>+9.2f}  "
            f"V1 ${tot(rows, 'v1_pnl'):>+9.2f} ({v1_viol} viol)  "
            f"V3 ${tot(rows, 'v3_pnl'):>+9.2f} ({v3_viol} viol)  "
            f"V3-V1 ${tot(rows, 'v3_pnl') - tot(rows, 'v1_pnl'):>+8.2f}"
        )

    # Save full details with per-trade V3 breakdown
    out = ROOT / "backtest_reports" / "show_v3_per_day.json"
    out.write_text(json.dumps({"normal": normal, "outrageous": outrageous}, indent=2), encoding="utf-8")
    print(f"\n[write] {out}  (full per-trade V3 detail)")
