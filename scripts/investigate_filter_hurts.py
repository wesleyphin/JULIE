#!/usr/bin/env python3
"""Investigate which trades filter D (and E) hurt on non-outrageous 2025 tape.

Same input data as validate_filter_de_on_normal_days.py. Instead of aggregating
by day, this script classifies every D-touched trade into:

  HURT    — D capped a trade that was a winner (pnl > 0)
  HELPED  — D capped a trade that was a loser (pnl < 0)
  NEUTRAL — trade was breakeven

Then reports patterns across the HURT set so we can see if there's a refinement
(time-of-day, sub-strategy shape, vol regime, prior loss streak, etc.) that
would let us skip the harmful cap without losing the protection on losers.

Usage:
    python3 scripts/investigate_filter_hurts.py
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
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
NY = ZoneInfo("America/New_York")

RGX_TRANSITION = re.compile(
    r"Regime transition: \S+ -> (?P<regime>\S+).* ts=(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^ \|]*)"
)


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def load_regime_timeline(log_path: Path) -> list[tuple[datetime, str, float, float]]:
    """Return (ts, regime, vol_bp, eff) events sorted by ts."""
    events = []
    rgx_full = re.compile(
        r"Regime transition: \S+ -> (?P<regime>\S+) \| vol=(?P<vol>[\d.]+)bp eff=(?P<eff>[\d.]+) .* ts=(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^ \|]*)"
    )
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = rgx_full.search(line)
            if not m:
                continue
            regime = m.group("regime").strip()
            if regime == "dead_tape":
                regime = "neutral"
            try:
                ts = parse_ts(m.group("ts"))
            except Exception:
                continue
            events.append((ts, regime, float(m.group("vol")), float(m.group("eff"))))
    events.sort(key=lambda x: x[0])
    return events


def regime_at(ts, events):
    if not events:
        return "warmup", 0.0, 0.0
    keys = [e[0] for e in events]
    i = bisect_right(keys, ts) - 1
    if i < 0:
        return "warmup", 0.0, 0.0
    return events[i][1], events[i][2], events[i][3]


def hour_bucket(dt: datetime) -> str:
    h = dt.hour
    if h < 9:
        return "pre-mkt"
    if h < 10:
        return "09:xx"
    if h < 11:
        return "10:xx"
    if h < 12:
        return "11:xx"
    if h < 13:
        return "12:xx"
    if h < 14:
        return "13:xx"
    if h < 15:
        return "14:xx"
    if h < 16:
        return "15:xx"
    return "aft-mkt"


def shape_bucket(sub: str) -> str:
    # Pull the Mom/Rev token out of "5min_09-12_Short_Rev_T6_SL10_TP12.5"
    parts = sub.split("_")
    for tok in parts:
        if tok in ("Rev", "Mom"):
            return tok
    return "?"


def size_bucket(sub: str) -> str:
    # Pull the T# token (confidence tier)
    for tok in sub.split("_"):
        if tok.startswith("T") and tok[1:].isdigit():
            return tok
    return "?"


def investigate_month(month_tag: str) -> list[dict]:
    folder = REPORT_ROOT / f"{month_tag}_ny_iter11_deadtape"
    trades_path = folder / "closed_trades.json"
    log_path = folder / "topstep_live_bot.log"
    if not trades_path.exists():
        return []

    regime_events = load_regime_timeline(log_path)
    trades = json.loads(trades_path.read_text(encoding="utf-8"))

    by_day: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        try:
            et = parse_ts(t["entry_time"]).astimezone(NY)
        except Exception:
            continue
        by_day[et.date().isoformat()].append(t)

    records = []
    for day, day_trades in sorted(by_day.items()):
        day_trades.sort(key=lambda t: parse_ts(t["entry_time"]))
        cum_sim = 0.0
        cum_base = 0.0
        day_loss_streak = 0  # consecutive losing trades so far today
        for idx, t in enumerate(day_trades):
            size = int(t.get("size", 1) or 1)
            pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
            per_contract = pnl / size if size > 0 else pnl
            et = parse_ts(t["entry_time"])
            regime, vol, eff = regime_at(et, regime_events)
            capped = regime in REGIME_CAPPED

            action = "none"
            sim_pnl = pnl
            if capped and size > CAP_SIZE:
                effective_cap = CAP_SIZE
                unlocked = cum_sim >= UNLOCK_THRESHOLD
                if unlocked:
                    effective_cap = UNLOCK_SIZE
                    action = "unlock"
                else:
                    action = "cap"
                new_size = min(size, effective_cap)
                sim_pnl = per_contract * new_size

            delta = sim_pnl - pnl
            if action == "cap":
                label = "HURT" if delta < -0.01 and pnl > 0 else \
                        "HELPED" if delta > 0.01 and pnl < 0 else \
                        "NEUTRAL"
            elif action == "unlock":
                label = "HELPED" if delta > 0.01 else \
                        "HURT" if delta < -0.01 else \
                        "NEUTRAL"
            else:
                label = "NONE"

            records.append({
                "month": month_tag,
                "day": day,
                "et_time": et.strftime("%H:%M"),
                "hour_bucket": hour_bucket(et),
                "strategy": t.get("strategy", ""),
                "sub_strategy": t.get("sub_strategy", ""),
                "shape": shape_bucket(t.get("sub_strategy", "")),
                "tier": size_bucket(t.get("sub_strategy", "")),
                "side": t.get("side", ""),
                "size_orig": size,
                "size_sim": int(size if action == "none" else (
                    UNLOCK_SIZE if action == "unlock" else CAP_SIZE
                )),
                "pnl_orig": round(pnl, 2),
                "pnl_sim": round(sim_pnl, 2),
                "delta": round(delta, 2),
                "regime": regime,
                "vol_bp": round(vol, 2),
                "eff": round(eff, 3),
                "day_pnl_before_this": round(cum_sim, 2),
                "day_loss_streak": day_loss_streak,
                "trade_idx_in_day": idx,
                "action": action,
                "label": label,
            })
            cum_sim += sim_pnl
            cum_base += pnl
            day_loss_streak = (day_loss_streak + 1) if pnl < 0 else 0
    return records


def report(records: list[dict]) -> str:
    out = []
    caps = [r for r in records if r["action"] == "cap"]
    unlocks = [r for r in records if r["action"] == "unlock"]

    hurt = [r for r in caps if r["label"] == "HURT"]
    helped = [r for r in caps if r["label"] == "HELPED"]
    neutral = [r for r in caps if r["label"] == "NEUTRAL"]

    hurt_total = sum(r["delta"] for r in hurt)
    helped_total = sum(r["delta"] for r in helped)
    net = sum(r["delta"] for r in caps) + sum(r["delta"] for r in unlocks)

    out.append(f"Total D-cap events:   {len(caps):>4}")
    out.append(f"  HURT   (winner capped, -\$): {len(hurt):>4}   sum delta = \${hurt_total:+.2f}")
    out.append(f"  HELPED (loser capped,  +\$): {len(helped):>4}   sum delta = \${helped_total:+.2f}")
    out.append(f"  NEUTRAL:                      {len(neutral):>4}")
    out.append(f"Unlock events:         {len(unlocks):>4}   delta = \${sum(r['delta'] for r in unlocks):+.2f}")
    out.append(f"Net filter delta:                  \${net:+.2f}")
    out.append("")

    def bucket_report(records, key, title, top_n=10):
        c = Counter()
        pnl_sum = defaultdict(float)
        for r in records:
            c[r[key]] += 1
            pnl_sum[r[key]] += r["delta"]
        out.append(f"--- {title} ---")
        out.append(f"{'bucket':<25}{'count':>8}{'delta $':>12}")
        for bucket, cnt in c.most_common(top_n):
            out.append(f"{str(bucket):<25}{cnt:>8}{pnl_sum[bucket]:>+12.2f}")
        out.append("")

    out.append("=" * 60)
    out.append("PATTERNS IN 'HURT' CAP EVENTS (winners D capped)")
    out.append("=" * 60)
    bucket_report(hurt, "hour_bucket", "by hour-of-day")
    bucket_report(hurt, "shape", "by shape (Mom / Rev)")
    bucket_report(hurt, "tier", "by confidence tier")
    bucket_report(hurt, "regime", "by regime")
    bucket_report(hurt, "sub_strategy", "by full sub_strategy (top 10)")

    out.append("=" * 60)
    out.append("PATTERNS IN 'HELPED' CAP EVENTS (losers D capped)")
    out.append("=" * 60)
    bucket_report(helped, "hour_bucket", "by hour-of-day")
    bucket_report(helped, "shape", "by shape (Mom / Rev)")

    # Worst individual HURT trades
    worst = sorted(hurt, key=lambda r: r["delta"])[:15]
    out.append("=" * 60)
    out.append("TOP 15 individual HURT trades (biggest $ give-up)")
    out.append("=" * 60)
    out.append(f"{'day':<12}{'time':<7}{'sub_strategy':<40}{'side':<6}{'sz':>3}{'pnl_orig':>10}{'delta':>10}  vol/eff/regime")
    for r in worst:
        out.append(
            f"{r['day']:<12}{r['et_time']:<7}{r['sub_strategy'][:38]:<40}"
            f"{r['side']:<6}{r['size_orig']:>3}{r['pnl_orig']:>+10.2f}{r['delta']:>+10.2f}  "
            f"{r['vol_bp']}bp/{r['eff']}/{r['regime']}"
        )

    # Cross-tab: hurt/helped by shape+regime
    out.append("")
    out.append("=" * 60)
    out.append("CROSS-TAB: shape x regime — net $ delta")
    out.append("=" * 60)
    grid = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    for r in caps:
        grid[r["shape"]][r["regime"]] += r["delta"]
        counts[r["shape"]][r["regime"]] += 1
    for shape in sorted(grid):
        for regime in sorted(grid[shape]):
            out.append(f"  {shape:<5} x {regime:<12}  n={counts[shape][regime]:>3}  delta=${grid[shape][regime]:+.2f}")

    return "\n".join(out)


if __name__ == "__main__":
    records = []
    for m in MONTHS:
        records.extend(investigate_month(m))
    print(report(records))
    (ROOT / "backtest_reports" / "filter_hurts_investigation.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(f"\n[write] {ROOT}/backtest_reports/filter_hurts_investigation.json  ({len(records)} trade records)")
