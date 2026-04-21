#!/usr/bin/env python3
"""Validate filter D + E don't hurt P&L on non-outrageous (normal) 2025 tape.

Reconstructs per-minute regime state from iter-11 backtest logs (where the
classifier already ran and emitted "Regime transition" lines), then applies
filter D (cap size to 1 on whipsaw/calm_trend) and filter E (unlock to 3
once cum daily P&L >= $200) to the closed_trades.json from those runs.

Does NOT simulate filter C (requires trend_day tier state which isn't logged).

Usage:
    python3 scripts/validate_filter_de_on_normal_days.py

Months validated: 2025-03, 2025-05, 2025-06 (no outrageous 2025 days overlap).
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
MONTHS = ["2025_03", "2025_05", "2025_06"]
REGIME_CAPPED = {"whipsaw", "calm_trend"}
UNLOCK_THRESHOLD = 200.0
UNLOCK_SIZE = 3
CAP_SIZE = 1

# Fragment line like:
# 2026-04-20 17:47:04,105 [INFO] Regime transition: warmup -> dead_tape | vol=0.00bp eff=0.000 | buf_bal=0.10 buf_fp=0.12 rev=3 | ts=2025-02-28 23:59:00-05:00
RGX_TRANSITION = re.compile(
    r"Regime transition: \S+ -> (?P<regime>\S+).* ts=(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^ \|]*)"
)
NY = ZoneInfo("America/New_York")


def parse_ts(s: str) -> datetime:
    # Accept both "2025-03-03 08:15:00-05:00" and ISO forms.
    s = s.strip()
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    # Fallback: strip trailing tz block "-HH:MM" already handled above.
    raise ValueError(f"cannot parse ts: {s!r}")


def load_regime_timeline(log_path: Path) -> list[tuple[datetime, str]]:
    """Return list of (ts, regime) sorted by ts. Map dead_tape->neutral so it
    doesn't trigger the D size cap (current classifier doesn't emit dead_tape).
    """
    events: list[tuple[datetime, str]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RGX_TRANSITION.search(line)
            if not m:
                continue
            regime = m.group("regime").strip()
            if regime == "dead_tape":  # not emitted by current classifier
                regime = "neutral"
            try:
                ts = parse_ts(m.group("ts"))
            except Exception:
                continue
            events.append((ts, regime))
    events.sort(key=lambda x: x[0])
    return events


def regime_at(ts: datetime, events: list[tuple[datetime, str]],
              _keys_cache: dict[int, list[datetime]] = {}) -> str:
    """Binary search the regime active at ts. Anything before first event
    returns 'warmup' (uncapped since D only fires on whipsaw/calm_trend)."""
    if not events:
        return "warmup"
    key = id(events)
    keys = _keys_cache.get(key)
    if keys is None:
        keys = [e[0] for e in events]
        _keys_cache[key] = keys
    i = bisect_right(keys, ts) - 1
    if i < 0:
        return "warmup"
    return events[i][1]


def simulate_month(month_tag: str) -> dict:
    folder = REPORT_ROOT / f"{month_tag}_ny_iter11_deadtape"
    trades_path = folder / "closed_trades.json"
    log_path = folder / "topstep_live_bot.log"
    if not trades_path.exists() or not log_path.exists():
        return {"month": month_tag, "error": "missing artifacts"}

    regime_events = load_regime_timeline(log_path)
    trades = json.loads(trades_path.read_text(encoding="utf-8"))

    # Group by ET date
    by_day: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        try:
            et = parse_ts(t["entry_time"]).astimezone(NY)
        except Exception:
            continue
        by_day[et.date().isoformat()].append(t)

    day_rows: list[dict] = []
    total_base = 0.0
    total_sim = 0.0
    cap_trigger_count = 0
    unlock_trigger_count = 0

    for day, day_trades in sorted(by_day.items()):
        # Chronological ordering inside the day
        day_trades.sort(key=lambda t: parse_ts(t["entry_time"]))
        base_pnl = 0.0
        sim_pnl = 0.0
        cum_sim = 0.0  # running sum of SIM pnl for filter E unlock check
        day_caps = 0
        day_unlocks = 0

        for t in day_trades:
            size = int(t.get("size", 1) or 1)
            pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
            base_pnl += pnl

            # Per-contract pnl (approximation: trade cost/profit scales with size)
            per_contract = pnl / size if size > 0 else pnl

            et = parse_ts(t["entry_time"])
            regime = regime_at(et, regime_events)
            capped = regime in REGIME_CAPPED

            if capped and size > CAP_SIZE:
                # Filter E: unlock kicks in once cum_sim >= threshold
                effective_cap = CAP_SIZE
                if cum_sim >= UNLOCK_THRESHOLD:
                    effective_cap = UNLOCK_SIZE
                    if size > CAP_SIZE:
                        day_unlocks += 1
                        unlock_trigger_count += 1
                new_size = min(size, effective_cap)
                if new_size < size:
                    day_caps += 1
                    cap_trigger_count += 1
                sim_trade_pnl = per_contract * new_size
            else:
                sim_trade_pnl = pnl

            sim_pnl += sim_trade_pnl
            cum_sim += sim_trade_pnl

        delta = sim_pnl - base_pnl
        day_rows.append({
            "day": day,
            "trades": len(day_trades),
            "base_pnl": round(base_pnl, 2),
            "sim_pnl": round(sim_pnl, 2),
            "delta": round(delta, 2),
            "caps": day_caps,
            "unlocks": day_unlocks,
        })
        total_base += base_pnl
        total_sim += sim_pnl

    return {
        "month": month_tag,
        "days": day_rows,
        "totals": {
            "base_pnl": round(total_base, 2),
            "sim_pnl": round(total_sim, 2),
            "delta": round(total_sim - total_base, 2),
            "cap_triggers": cap_trigger_count,
            "unlock_triggers": unlock_trigger_count,
            "days_count": len(day_rows),
        },
    }


def format_report(results: list[dict]) -> str:
    lines = []
    grand_base = 0.0
    grand_sim = 0.0
    losing_days_delta = []
    winning_days_delta = []
    hurt_days = []  # days where D+E made things strictly worse
    helped_days = []

    for r in results:
        if "error" in r:
            lines.append(f"[{r['month']}] SKIP: {r['error']}")
            continue
        t = r["totals"]
        lines.append(
            f"\n=== {r['month']} ===  days={t['days_count']}  base=${t['base_pnl']:>+9.2f}  "
            f"sim=${t['sim_pnl']:>+9.2f}  delta=${t['delta']:>+9.2f}  "
            f"caps={t['cap_triggers']} unlocks={t['unlock_triggers']}"
        )
        grand_base += t["base_pnl"]
        grand_sim += t["sim_pnl"]

        # Per-day lines only for days where delta != 0 (filter actually fired)
        for d in r["days"]:
            if d["caps"] == 0 and d["unlocks"] == 0:
                continue
            marker = ""
            if d["delta"] < -0.01:
                marker = " HURT"
                hurt_days.append(d)
            elif d["delta"] > 0.01:
                marker = " helped"
                helped_days.append(d)
            lines.append(
                f"  {d['day']}  trades={d['trades']:>2}  "
                f"base=${d['base_pnl']:>+8.2f}  sim=${d['sim_pnl']:>+8.2f}  "
                f"delta=${d['delta']:>+8.2f}  caps={d['caps']} unlocks={d['unlocks']}{marker}"
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"GRAND TOTAL  base=${grand_base:>+9.2f}  sim=${grand_sim:>+9.2f}  "
        f"delta=${grand_sim - grand_base:>+9.2f}"
    )
    lines.append(
        f"Days where D+E helped: {len(helped_days)}  |  hurt: {len(hurt_days)}"
    )
    if hurt_days:
        worst = sorted(hurt_days, key=lambda d: d["delta"])[:5]
        lines.append("  worst 5 hurt days:")
        for d in worst:
            lines.append(f"    {d['day']}  delta=${d['delta']:>+8.2f}  caps={d['caps']} unlocks={d['unlocks']}")
    if helped_days:
        best = sorted(helped_days, key=lambda d: -d["delta"])[:5]
        lines.append("  top 5 helped days:")
        for d in best:
            lines.append(f"    {d['day']}  delta=${d['delta']:>+8.2f}  caps={d['caps']} unlocks={d['unlocks']}")
    return "\n".join(lines)


if __name__ == "__main__":
    results = [simulate_month(m) for m in MONTHS]
    print(format_report(results))

    # Write JSON for follow-up analysis
    out = ROOT / "backtest_reports" / "validate_filter_de_normal_days.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDetailed results: {out}")
