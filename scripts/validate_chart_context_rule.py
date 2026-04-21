#!/usr/bin/env python3
"""Validate the chart-derived bounce/dip-fade veto rule on 2025 data.

Rule (strategy-agnostic, no sub_strategy matching):
  LONG  velocity(30min) < -VELOCITY_THRESH  AND  entry - low(30min)  > DIST_THRESH  → veto
  SHORT velocity(30min) > +VELOCITY_THRESH  AND  high(30min) - entry > DIST_THRESH  → veto

For every trade we score:
  baseline_pnl = trade's actual pnl_dollars (iter-11 baseline)
  rule_pnl     = 0 if rule fires, else pnl_dollars

Then we report per-day + grand totals. Also sweep thresholds to find the
best (vel, dist) combo.

Sources: the same 136-day 2025 iter-11-consistent set used earlier
(3 normal months + 6 outrageous folders).
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from bisect import bisect_right
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/Users/wes/Downloads/JULIE001")
REPORT_ROOT = ROOT / "backtest_reports" / "full_live_replay"
NY = ZoneInfo("America/New_York")

RGX_BAR = re.compile(
    r"Bar: (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<price>[\d.]+)"
)

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


def parse_ts(s):
    return datetime.fromisoformat(s.strip())


def load_bars(log_path):
    bars = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RGX_BAR.search(line)
            if not m:
                continue
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
            bars.append((ts.replace(tzinfo=NY), float(m.group("price"))))
    bars.sort(key=lambda x: x[0])
    return bars


def window_before(bars, ts_et, minutes_before):
    """Return bars in [ts - minutes_before, ts] (inclusive)."""
    if not bars:
        return []
    lo = ts_et - timedelta(minutes=minutes_before)
    keys = [b[0] for b in bars]
    hi_idx = bisect_right(keys, ts_et)
    lo_idx = bisect_right(keys, lo)
    return bars[lo_idx:hi_idx]


def rule_fires(side, entry_price, window_bars,
               velocity_thresh=0.20, dist_thresh=5.0, window_minutes=30):
    """Return (fires, reason_dict)."""
    if len(window_bars) < 10:
        return False, {"skip": "insufficient_bars", "n": len(window_bars)}
    prices = [b[1] for b in window_bars]
    times  = [b[0] for b in window_bars]
    minutes = max(1.0, (times[-1] - times[0]).total_seconds() / 60.0)
    velocity = (prices[-1] - prices[0]) / minutes  # pts/min
    low, high = min(prices), max(prices)
    reason = {
        "velocity": round(velocity, 3),
        "low": low, "high": high,
        "dist_low": round(entry_price - low, 2),
        "dist_high": round(high - entry_price, 2),
    }
    if side == "LONG" and velocity < -velocity_thresh and entry_price > (low + dist_thresh):
        return True, {**reason, "why": "bounce_fade"}
    if side == "SHORT" and velocity > +velocity_thresh and entry_price < (high - dist_thresh):
        return True, {**reason, "why": "dip_fade"}
    return False, reason


def evaluate(velocity_thresh=0.20, dist_thresh=5.0, window_minutes=30):
    total_base = 0.0
    total_rule = 0.0
    vetoed_count = 0
    vetoed_base_pnl = 0.0
    vetoed_winners = 0
    vetoed_losers = 0
    per_day = defaultdict(lambda: {"base": 0.0, "rule": 0.0, "vetoed": 0})
    skipped = 0
    total_trades = 0

    for folder_name, source_tag in SOURCES:
        folder = REPORT_ROOT / folder_name
        ct_path = folder / "closed_trades.json"
        log_path = folder / "topstep_live_bot.log"
        if not (ct_path.exists() and log_path.exists()):
            continue
        trades = json.loads(ct_path.read_text(encoding="utf-8"))
        bars = load_bars(log_path)

        for t in trades:
            try:
                et = parse_ts(t["entry_time"]).astimezone(NY)
            except Exception:
                continue
            entry_price = float(t.get("entry_price", 0.0) or 0.0)
            side = str(t.get("side", "")).upper()
            pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
            total_trades += 1
            total_base += pnl

            w = window_before(bars, et, window_minutes)
            fires, reason = rule_fires(side, entry_price, w,
                                       velocity_thresh=velocity_thresh,
                                       dist_thresh=dist_thresh,
                                       window_minutes=window_minutes)
            day_key = et.date().isoformat()
            per_day[day_key]["base"] += pnl

            if "skip" in reason:
                skipped += 1
            if fires:
                vetoed_count += 1
                vetoed_base_pnl += pnl
                if pnl > 0:
                    vetoed_winners += 1
                elif pnl < 0:
                    vetoed_losers += 1
                # rule_pnl = 0 for vetoed trades
                per_day[day_key]["vetoed"] += 1
            else:
                total_rule += pnl
                per_day[day_key]["rule"] += pnl

    return {
        "velocity_thresh": velocity_thresh,
        "dist_thresh": dist_thresh,
        "window_minutes": window_minutes,
        "total_trades": total_trades,
        "skipped_bars": skipped,
        "vetoed_count": vetoed_count,
        "vetoed_base_pnl": round(vetoed_base_pnl, 2),
        "vetoed_winners": vetoed_winners,
        "vetoed_losers": vetoed_losers,
        "baseline_pnl": round(total_base, 2),
        "rule_pnl": round(total_rule, 2),
        "delta": round(total_rule - total_base, 2),
        "per_day": {d: {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()} for d, row in per_day.items()},
    }


def sweep_thresholds():
    print("=" * 80)
    print("Threshold sweep: velocity_thresh x dist_thresh")
    print("=" * 80)
    print(f"{'vel_thr':>8}{'dist_thr':>10}  {'vetoed':>7}{'winners':>8}{'losers':>7}  "
          f"{'vetoed_$':>10}{'base_$':>10}{'rule_$':>10}{'delta':>10}")
    best = None
    for vt in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for dt in [3.0, 5.0, 7.0, 10.0, 15.0]:
            r = evaluate(velocity_thresh=vt, dist_thresh=dt)
            print(f"{vt:>8.2f}{dt:>10.1f}  {r['vetoed_count']:>7}{r['vetoed_winners']:>8}"
                  f"{r['vetoed_losers']:>7}  {r['vetoed_base_pnl']:>+10.2f}"
                  f"{r['baseline_pnl']:>+10.2f}{r['rule_pnl']:>+10.2f}{r['delta']:>+10.2f}")
            if best is None or r["rule_pnl"] > best["rule_pnl"]:
                best = r
    print()
    print("BEST:")
    print(f"  velocity_thresh={best['velocity_thresh']}  dist_thresh={best['dist_thresh']}")
    print(f"  baseline P&L     ${best['baseline_pnl']:+.2f}")
    print(f"  rule P&L         ${best['rule_pnl']:+.2f}")
    print(f"  improvement      ${best['delta']:+.2f}")
    print(f"  vetoed {best['vetoed_count']} of {best['total_trades']} trades "
          f"({best['vetoed_winners']}W / {best['vetoed_losers']}L, vetoed $ = {best['vetoed_base_pnl']:+.2f})")
    return best


if __name__ == "__main__":
    # Baseline (no rule)
    print("=" * 80)
    print("BASELINE (no chart rule)")
    print("=" * 80)
    r0 = evaluate(velocity_thresh=9999.0, dist_thresh=9999.0)  # never fires
    print(f"  Total P&L across 136 iter-11 days: ${r0['baseline_pnl']:+.2f}   trades={r0['total_trades']}")
    print()

    # Proposed default from today's tape analysis
    print("=" * 80)
    print("PROPOSED (velocity_thresh=0.20, dist_thresh=5.0) — from Apr 21 tape")
    print("=" * 80)
    r1 = evaluate(velocity_thresh=0.20, dist_thresh=5.0)
    print(f"  baseline P&L:  ${r1['baseline_pnl']:+.2f}")
    print(f"  rule P&L:      ${r1['rule_pnl']:+.2f}")
    print(f"  delta:         ${r1['delta']:+.2f}")
    print(f"  vetoed: {r1['vetoed_count']} / {r1['total_trades']} "
          f"({r1['vetoed_winners']}W / {r1['vetoed_losers']}L)")
    print(f"  vetoed trades' baseline PnL: ${r1['vetoed_base_pnl']:+.2f}")
    print(f"  bars-insufficient skip: {r1['skipped_bars']}")
    print()

    # Full sweep
    best = sweep_thresholds()

    # Save
    out = ROOT / "backtest_reports" / "validate_chart_context_rule.json"
    out.write_text(json.dumps({
        "proposed": r1,
        "best": best,
    }, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")
