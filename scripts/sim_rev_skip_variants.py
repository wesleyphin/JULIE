#!/usr/bin/env python3
"""Compare filter-D variants across outrageous and normal day sets.

Variants:
  V0 — No filter D (baseline iter-11 only)
  V1 — D caps all (current shipped behaviour)
  V2 — D caps but skips _Rev_ sub-strategies (proposed change)

Applies same E (green-day unlock) logic to V1 and V2.
Does NOT simulate filter C (requires trend_day tier state — scope deferred).

Normal-day sources:  backtest_reports/full_live_replay/2025_{03,05,06}_ny_iter11_deadtape
Outrageous sources:  backtest_reports/full_live_replay/outrageous_{feb,jul,aug,oct,dec,apr}

Usage:
    python3 scripts/sim_rev_skip_variants.py
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

NORMAL_SOURCES = [
    ("2025_03", "2025_03_ny_iter11_deadtape"),
    ("2025_05", "2025_05_ny_iter11_deadtape"),
    ("2025_06", "2025_06_ny_iter11_deadtape"),
]
OUTRAGEOUS_SOURCES = [
    ("feb",  "outrageous_feb"),
    ("jul",  "outrageous_jul"),
    ("aug",  "outrageous_aug"),
    ("oct",  "outrageous_oct"),
    ("dec",  "outrageous_dec"),
    ("apr",  "outrageous_apr"),
]

RGX_TRANSITION = re.compile(
    r"Regime transition: \S+ -> (?P<regime>\S+).* ts=(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^ \|]*)"
)


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def load_regime_timeline(log_path: Path) -> list[tuple[datetime, str]]:
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


def simulate(trades, regime_events, *, cap_enabled: bool, skip_rev: bool,
             skip_rev_only_calm: bool = False, rev_cap_size: int | None = None) -> dict:
    by_day = defaultdict(list)
    for t in trades:
        try:
            et = parse_ts(t["entry_time"]).astimezone(NY)
        except Exception:
            continue
        by_day[et.date().isoformat()].append(t)

    total_pnl = 0.0
    dd_violations = 0
    max_dds = []
    day_rows = []

    for day, day_trades in sorted(by_day.items()):
        day_trades.sort(key=lambda t: parse_ts(t["entry_time"]))
        cum = 0.0
        peak = 0.0
        max_dd = 0.0

        for t in day_trades:
            size = int(t.get("size", 1) or 1)
            pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
            per_contract = pnl / size if size > 0 else pnl

            et = parse_ts(t["entry_time"])
            regime = regime_at(et, regime_events)
            capped_regime = regime in REGIME_CAPPED
            sub = str(t.get("sub_strategy", ""))
            is_rev = "_Rev_" in sub

            # Decide whether D fires
            should_cap = cap_enabled and capped_regime and size > CAP_SIZE
            if should_cap and skip_rev and is_rev:
                should_cap = False
            if should_cap and skip_rev_only_calm and is_rev and regime == "calm_trend":
                should_cap = False

            if should_cap:
                if is_rev and rev_cap_size is not None:
                    # Rev gets a softer cap (e.g. 2 instead of 1)
                    effective_cap = rev_cap_size
                else:
                    effective_cap = UNLOCK_SIZE if cum >= UNLOCK_THRESHOLD else CAP_SIZE
                new_size = min(size, effective_cap)
                trade_pnl = per_contract * new_size
            else:
                trade_pnl = pnl

            cum += trade_pnl
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

        total_pnl += cum
        max_dds.append(max_dd)
        if max_dd > 350.0:
            dd_violations += 1
        day_rows.append({"day": day, "pnl": round(cum, 2), "max_dd": round(max_dd, 2)})

    return {
        "total_pnl": round(total_pnl, 2),
        "days": len(day_rows),
        "dd_violations_over_350": dd_violations,
        "avg_dd": round(sum(max_dds) / len(max_dds), 2) if max_dds else 0.0,
        "max_dd_worst": round(max(max_dds), 2) if max_dds else 0.0,
        "day_rows": day_rows,
    }


def run_set(source_list, label) -> dict:
    all_trades = []
    all_regime_events = []
    # concatenate per-month trade lists and regime timelines (each month is
    # an isolated replay — regimes don't overlap across months)
    for tag, folder_name in source_list:
        folder = REPORT_ROOT / folder_name
        trades = json.loads((folder / "closed_trades.json").read_text(encoding="utf-8"))
        # tag trades with source so day keys don't collide across months
        for t in trades:
            t["_source"] = tag
        all_trades.extend(trades)
        all_regime_events.extend(load_regime_timeline(folder / "topstep_live_bot.log"))
    all_regime_events.sort(key=lambda x: x[0])

    results = {
        "V0_no_D":           simulate(all_trades, all_regime_events, cap_enabled=False, skip_rev=False),
        "V1_D_caps_all":     simulate(all_trades, all_regime_events, cap_enabled=True,  skip_rev=False),
        "V2_D_skips_Rev":    simulate(all_trades, all_regime_events, cap_enabled=True,  skip_rev=True),
        "V3_Rev_skip_calm":  simulate(all_trades, all_regime_events, cap_enabled=True,  skip_rev=False, skip_rev_only_calm=True),
        "V4_Rev_cap_to_2":   simulate(all_trades, all_regime_events, cap_enabled=True,  skip_rev=False, rev_cap_size=2),
    }
    return {"label": label, "trades": len(all_trades), "results": results}


def format_compare(name, data) -> str:
    out = [f"\n{'=' * 78}"]
    out.append(f"{name}  ({data['trades']} trades, {data['results']['V0_no_D']['days']} days)")
    out.append("=" * 78)
    out.append(f"{'variant':<22}{'net P&L':>12}{'DD>$350':>10}{'avg DD':>10}{'worst DD':>12}")
    out.append("-" * 78)
    for variant, r in data["results"].items():
        out.append(
            f"{variant:<22}{r['total_pnl']:>+12.2f}{r['dd_violations_over_350']:>10}"
            f"{r['avg_dd']:>10.2f}{r['max_dd_worst']:>12.2f}"
        )
    v0 = data["results"]["V0_no_D"]["total_pnl"]
    v1 = data["results"]["V1_D_caps_all"]["total_pnl"]
    v2 = data["results"]["V2_D_skips_Rev"]["total_pnl"]
    out.append("-" * 78)
    out.append(f"D delta (V1 - V0): ${v1 - v0:+.2f}    D-Rev-skip delta (V2 - V0): ${v2 - v0:+.2f}")
    out.append(f"Rev-skip improvement (V2 - V1): ${v2 - v1:+.2f}")
    return "\n".join(out)


if __name__ == "__main__":
    normal = run_set(NORMAL_SOURCES, "NORMAL (Mar/May/Jun 2025)")
    outrageous = run_set(OUTRAGEOUS_SOURCES, "OUTRAGEOUS (6-month set)")
    print(format_compare("NORMAL set", normal))
    print(format_compare("OUTRAGEOUS set", outrageous))

    out_path = ROOT / "backtest_reports" / "sim_rev_skip_variants.json"
    out_path.write_text(json.dumps({
        "normal": normal,
        "outrageous": outrageous,
    }, indent=2), encoding="utf-8")
    print(f"\n[write] {out_path}")
