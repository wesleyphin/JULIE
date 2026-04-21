#!/usr/bin/env python3
"""Re-run the chart-velocity rule validation, but stratify by regime at entry.

Hypothesis: the bounce-fade pattern that was crystal-clear on 2026-04-21 might
generalize only in specific regime states (e.g. whipsaw) and be noise in others.

For every trade:
  1. Compute regime at entry_time (from log "Regime transition" events +
     offline reconstruction fallback for folders without them)
  2. Compute 30-min velocity + dist-from-extreme
  3. Record (regime, fires, pnl)

Then aggregate: per-regime baseline vs rule P&L, threshold sweep within regime.
"""
from __future__ import annotations

import json
import re
import sys
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/Users/wes/Downloads/JULIE001")
REPORT_ROOT = ROOT / "backtest_reports" / "full_live_replay"
NY = ZoneInfo("America/New_York")

sys.path.insert(0, str(ROOT / "scripts"))
from reconstruct_regime import reconstruct_from_log  # noqa: E402

RGX_BAR = re.compile(
    r"Bar: (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<price>[\d.]+)"
)
RGX_REGIME = re.compile(
    r"Regime transition: \S+ -> (?P<regime>\S+).* ts=(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[^ \|]*)"
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
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=NY)
            bars.append((ts, float(m.group("price"))))
    bars.sort()
    return bars


def load_regimes(log_path):
    events = []
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Regime transition" not in line:
                continue
            m = RGX_REGIME.search(line)
            if not m:
                continue
            try:
                ts = parse_ts(m.group("ts"))
            except Exception:
                continue
            regime = m.group("regime").strip()
            if regime == "dead_tape":
                regime = "neutral"
            events.append((ts, regime))
    events.sort()
    return events


def regime_at(ts, events):
    if not events:
        return "warmup"
    keys = [e[0] for e in events]
    i = bisect_right(keys, ts) - 1
    return events[i][1] if i >= 0 else "warmup"


def window_before(bars, ts_et, minutes_before=30):
    lo = ts_et - timedelta(minutes=minutes_before)
    keys = [b[0] for b in bars]
    hi_idx = bisect_right(keys, ts_et)
    lo_idx = bisect_right(keys, lo)
    return bars[lo_idx:hi_idx]


def rule_fires(side, entry_price, window_bars,
               velocity_thresh=0.20, dist_thresh=5.0):
    if len(window_bars) < 10:
        return False
    prices = [b[1] for b in window_bars]
    times  = [b[0] for b in window_bars]
    minutes = max(1.0, (times[-1] - times[0]).total_seconds() / 60.0)
    velocity = (prices[-1] - prices[0]) / minutes
    low, high = min(prices), max(prices)
    if side == "LONG" and velocity < -velocity_thresh and entry_price > (low + dist_thresh):
        return True
    if side == "SHORT" and velocity > +velocity_thresh and entry_price < (high - dist_thresh):
        return True
    return False


def build_labeled_trades():
    """For every trade: (regime, side, pnl, fires_at_various_thresholds)."""
    rows = []
    for folder_name, source_tag in SOURCES:
        folder = REPORT_ROOT / folder_name
        ct = folder / "closed_trades.json"
        log = folder / "topstep_live_bot.log"
        if not (ct.exists() and log.exists()):
            continue
        trades = json.loads(ct.read_text(encoding="utf-8"))
        bars = load_bars(log)
        regimes = load_regimes(log)
        if not regimes:
            # Offline reconstruction for pre-classifier logs
            r2 = reconstruct_from_log(log)
            regimes = [(ts, reg) for (ts, reg, _, _) in r2]
        for t in trades:
            try:
                et = parse_ts(t["entry_time"]).astimezone(NY)
            except Exception:
                continue
            entry_price = float(t.get("entry_price", 0.0) or 0.0)
            side = str(t.get("side", "")).upper()
            pnl = float(t.get("pnl_dollars", 0.0) or 0.0)
            regime = regime_at(et, regimes)
            w = window_before(bars, et, 30)
            if len(w) < 10:
                continue
            rows.append({
                "regime": regime,
                "side": side,
                "pnl": pnl,
                "entry_price": entry_price,
                "window": w,
            })
    return rows


def sweep_by_regime(rows):
    # For each (regime, threshold combo), compute baseline vs rule P&L
    groups = defaultdict(list)
    for r in rows:
        groups[r["regime"]].append(r)

    combos = []
    for vt in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for dt in [3.0, 5.0, 7.0, 10.0]:
            combos.append((vt, dt))

    for regime in ["whipsaw", "calm_trend", "neutral", "warmup"]:
        rs = groups.get(regime, [])
        if not rs:
            continue
        print("=" * 90)
        print(f"REGIME: {regime}  (n={len(rs)} trades)")
        print("=" * 90)
        base_pnl = sum(r["pnl"] for r in rs)
        print(f"  baseline P&L: ${base_pnl:+.2f}")
        print()
        print(f"{'vel':>6}{'dist':>7}  {'vetoed':>7}{'wins':>6}{'loss':>6}  "
              f"{'vetoed_$':>10}{'rule_$':>10}{'delta':>10}")
        best_delta = -1e9
        best_combo = None
        for (vt, dt) in combos:
            vetoed_pnl = 0.0
            vetoed_wins = 0
            vetoed_losses = 0
            vetoed_n = 0
            for r in rs:
                if rule_fires(r["side"], r["entry_price"], r["window"], vt, dt):
                    vetoed_n += 1
                    vetoed_pnl += r["pnl"]
                    if r["pnl"] > 0: vetoed_wins += 1
                    elif r["pnl"] < 0: vetoed_losses += 1
            rule_pnl = base_pnl - vetoed_pnl
            delta = rule_pnl - base_pnl
            marker = ""
            if delta > best_delta:
                best_delta = delta
                best_combo = (vt, dt, vetoed_n, vetoed_wins, vetoed_losses, vetoed_pnl, rule_pnl, delta)
            if delta > 0:
                marker = "  **"
            print(f"{vt:>6.2f}{dt:>7.1f}  {vetoed_n:>7}{vetoed_wins:>6}"
                  f"{vetoed_losses:>6}  {vetoed_pnl:>+10.2f}{rule_pnl:>+10.2f}{delta:>+10.2f}{marker}")
        print()
        bvt, bdt, bn, bw, bl, bv, br, bd = best_combo
        print(f"  BEST for {regime}: vel={bvt}, dist={bdt}  → veto {bn} trades "
              f"({bw}W/{bl}L), delta ${bd:+.2f}")
        print()


if __name__ == "__main__":
    print("[build] labeling trades with regime + pre-entry bars...")
    rows = build_labeled_trades()
    print(f"[build] {len(rows)} trades labeled")
    from collections import Counter
    print(f"[build] regime distribution: {dict(Counter(r['regime'] for r in rows))}")
    print()
    sweep_by_regime(rows)
