"""Post-hoc Kalshi blocker for AF trades from fast_aetherflow_replay.py.

Live AF signals pass through `_apply_kalshi_trade_overlay_to_signal` which calls
`build_trade_plan` (kalshi_trade_overlay.py) with a live Kalshi curve. We don't
have intraday Kalshi snapshots; we only have daily aggregates per strike. So
we approximate the block decision per trade:

  1. Skip trades outside _KALSHI_GATING_HOURS_ET = {12,13,14,15,16} → AUTO-PASS
     (matches the live bot's gating-hour gate exactly)
  2. For trades inside the window, find the next settlement-hour event for that
     trade's day (e.g. trade at 14:23 → 15:00 settlement)
  3. From the daily snapshot, find the strike row nearest the entry price
  4. Use (high+low)/2 cents → probability ∈ [0,1] as proxy for the live YES-price
  5. Compute aligned probability: LONG = P, SHORT = 1-P
  6. Apply the live block rule: aligned_prob < entry_threshold (0.45 by default)
     → BLOCK; else PASS

Output: closed_trades_kalshi_passed.json (subset that would have made it past
Kalshi).  Plus a summary of block/pass counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
NY = ZoneInfo("America/New_York")
KALSHI_DAILY = ROOT / "data" / "kalshi" / "kxinxu_2025_daily"

# Mirrors julie001.py:175 — only these hours have Kalshi gating active.
KALSHI_GATING_HOURS_ET = {12, 13, 14, 15, 16}

# Production threshold from kalshi_trade_overlay.py:516 default
ENTRY_THRESHOLD = 0.45


def load_kalshi_for_date(date_str: str) -> pd.DataFrame | None:
    """Load the daily snapshot parquet for a given trade date."""
    path = KALSHI_DAILY / f"{date_str}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df


def estimate_kalshi_aligned_prob(
    daily_snapshot: pd.DataFrame,
    settlement_hour_et: int,
    entry_price: float,
    side: str,
    event_date: str,
) -> float | None:
    """Pull the strike row nearest entry_price for the matching settlement
    event, return the aligned (side-adjusted) probability in [0, 1]."""
    sub = daily_snapshot[
        (daily_snapshot["settlement_hour_et"] == settlement_hour_et)
        & (daily_snapshot["event_date"] == event_date)
    ]
    if sub.empty:
        return None
    # Find closest strike to entry price
    sub = sub.copy()
    sub["dist"] = (sub["strike"] - entry_price).abs()
    row = sub.nsmallest(1, "dist").iloc[0]
    # high/low are integer cents (0-99). Use midpoint as our probability proxy.
    high_cents = float(row["high"])
    low_cents = float(row["low"])
    mid_cents = (high_cents + low_cents) / 2.0
    yes_prob = mid_cents / 100.0  # P(price > strike at settlement)
    yes_prob = max(0.0, min(1.0, yes_prob))
    if side == "LONG":
        return yes_prob
    elif side == "SHORT":
        return 1.0 - yes_prob
    return None


def next_settlement_hour(et_hour: int) -> int | None:
    """Map an entry hour to the next settlement event hour.
    Settlements are at 10,11,12,13,14,15,16 ET (top of each hour)."""
    for h in (10, 11, 12, 13, 14, 15, 16):
        if h > et_hour:
            return h
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="backtest_reports/af_fast_replay/af_full_2025_2026/closed_trades.json")
    ap.add_argument("--out-dir", default="backtest_reports/af_fast_replay/af_full_2025_2026")
    ap.add_argument("--threshold", type=float, default=ENTRY_THRESHOLD)
    args = ap.parse_args()

    trades_path = (ROOT / args.trades) if not Path(args.trades).is_absolute() else Path(args.trades)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)

    trades = json.loads(trades_path.read_text())
    print(f"[load] {len(trades)} AF trades from {trades_path}")

    snapshot_cache: dict[str, pd.DataFrame | None] = {}

    passed = []
    blocked = []
    auto_passed_outside_hours = []
    no_data = []

    for t in trades:
        et = datetime.fromisoformat(t["entry_time"]).astimezone(NY)
        et_hour = et.hour
        date_str = et.date().isoformat()

        if et_hour not in KALSHI_GATING_HOURS_ET:
            t = {**t, "kalshi_decision": "auto_pass_outside_hours"}
            auto_passed_outside_hours.append(t)
            passed.append(t)
            continue

        if date_str not in snapshot_cache:
            snapshot_cache[date_str] = load_kalshi_for_date(date_str)
        snapshot = snapshot_cache[date_str]
        if snapshot is None or snapshot.empty:
            t = {**t, "kalshi_decision": "no_data_pass"}
            no_data.append(t)
            passed.append(t)  # default to PASS when no Kalshi data (live: outside_gating_hours behaves same)
            continue

        next_set = next_settlement_hour(et_hour)
        if next_set is None:
            t = {**t, "kalshi_decision": "no_next_settlement_pass"}
            passed.append(t)
            continue

        aligned_prob = estimate_kalshi_aligned_prob(
            snapshot, next_set, float(t["entry_price"]), t["side"], date_str
        )
        if aligned_prob is None:
            t = {**t, "kalshi_decision": "no_strike_data_pass"}
            no_data.append(t)
            passed.append(t)
            continue

        if aligned_prob < args.threshold:
            t = {**t, "kalshi_decision": "blocked", "kalshi_aligned_prob": aligned_prob,
                 "kalshi_settlement_hour": next_set}
            blocked.append(t)
        else:
            t = {**t, "kalshi_decision": "passed", "kalshi_aligned_prob": aligned_prob,
                 "kalshi_settlement_hour": next_set}
            passed.append(t)

    # ---- Stats ----
    n = len(trades)
    n_pass = len(passed)
    n_block = len(blocked)
    n_auto = len(auto_passed_outside_hours)
    n_nodata = len(no_data)

    pnl_all = sum(float(t["pnl_dollars"]) for t in trades)
    pnl_pass = sum(float(t["pnl_dollars"]) for t in passed)
    pnl_block = sum(float(t["pnl_dollars"]) for t in blocked)
    wins_pass = sum(1 for t in passed if float(t["pnl_dollars"]) > 0)
    wins_block = sum(1 for t in blocked if float(t["pnl_dollars"]) > 0)

    print(f"\n=== Kalshi block analysis (threshold={args.threshold}) ===")
    print(f"  Total trades:                   {n}")
    print(f"  Auto-passed (outside 12-16 ET): {n_auto}")
    print(f"  No Kalshi data (auto-passed):   {n_nodata}")
    print(f"  In-window scored: blocked={n_block} passed={n_pass - n_auto - n_nodata}")
    print(f"  ---")
    print(f"  Trades after Kalshi:            {n_pass}/{n} ({n_pass/n:.0%})")
    print(f"  Pass PnL:                       ${pnl_pass:+,.2f}  winrate={wins_pass/max(1,n_pass):.1%}")
    print(f"  Block PnL (avoided):            ${pnl_block:+,.2f}  winrate={wins_block/max(1,n_block):.1%}")
    print(f"  Baseline (all):                 ${pnl_all:+,.2f}")

    out_path = out_dir / "closed_trades_kalshi_passed.json"
    out_path.write_text(json.dumps(passed, indent=2))
    print(f"\n[write] {out_path} ({n_pass} trades)")

    summary = {
        "total_trades": n,
        "passed": n_pass,
        "blocked": n_block,
        "auto_passed_outside_hours": n_auto,
        "no_data_passed": n_nodata,
        "pnl_all": pnl_all,
        "pnl_pass": pnl_pass,
        "pnl_block": pnl_block,
        "winrate_pass": wins_pass / max(1, n_pass),
        "winrate_block": wins_block / max(1, n_block),
        "threshold": args.threshold,
    }
    summary_path = out_dir / "kalshi_block_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
