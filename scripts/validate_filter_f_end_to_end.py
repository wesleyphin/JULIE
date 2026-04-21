#!/usr/bin/env python3
"""End-to-end validation of Filter F (chart bounce-fade veto) using the
actual loss_factor_guard module.

For each trading day:
  1. Initialize a fresh LFG + regime classifier
  2. Replay bars in chronological order (feeds both modules)
  3. At each trade's entry time, query LFG.should_veto_entry(signal)
  4. Track: pnl with rule vs without, per-regime breakdown

Datasets:
  - 2025 136-day iter-11 subset (in-sample training target)
  - April 2026 OOS (17 days, 195 trades)
"""
from __future__ import annotations

import json
import os
import re
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Set env BEFORE importing modules
os.environ["JULIE_LOSS_FACTOR_GUARD"] = "1"
os.environ["JULIE_REGIME_CLASSIFIER"] = "1"
os.environ["JULIE_LFG_CHART_VETO"] = "1"

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import loss_factor_guard as lfg_mod
import regime_classifier as rc_mod

NY = ZoneInfo("America/New_York")
REPORT_ROOT = ROOT / "backtest_reports" / "full_live_replay"

SOURCES_2025 = [
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

SOURCES_OOS = [
    # April 2026 out-of-sample (baseline replay)
    (str(ROOT / "backtest_reports" / "replay_apr2026_p1"), "apr2026"),
    (str(ROOT / "backtest_reports" / "replay_apr20" / "baseline_warm"), "apr2026"),
]

RGX_BAR = re.compile(
    r"Bar: (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<price>[\d.]+)"
)


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.strip())


def load_bars(log_path: Path):
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


def find_folder_with_data(root_or_folder: Path):
    """Handle either a direct iter-11 folder or a replay root with nested loops."""
    if (root_or_folder / "closed_trades.json").exists():
        return root_or_folder
    loops = sorted(root_or_folder.glob("live_loop_MES_*"))
    for loop in loops:
        if (loop / "closed_trades.json").exists():
            return loop
    return None


def simulate_day(folder: Path, trades_for_day, all_bars_for_day):
    """Replay bars through LFG + regime classifier, apply veto at trade entries.
    Returns (base_pnl, rule_pnl, vetoed_list, per_regime_tracker)."""
    # Fresh LFG + classifier for each day (avoids cross-day leaks)
    rc_mod._CLASSIFIER = None
    rc_mod.init_regime_classifier()
    lfg_mod._GUARD = None
    g = lfg_mod.init_guard()

    if not trades_for_day:
        return 0.0, 0.0, [], defaultdict(lambda: {"base": 0.0, "vetoed_n": 0, "vetoed_pnl": 0.0})

    # Build a merged event stream: bars + trade-entry events in chronological order.
    trade_events = []
    for t in trades_for_day:
        try:
            et = parse_ts(t["entry_time"]).astimezone(NY)
        except Exception:
            continue
        trade_events.append((et, t))
    trade_events.sort(key=lambda x: x[0])

    base_pnl = 0.0
    rule_pnl = 0.0
    vetoed = []
    per_regime = defaultdict(lambda: {"base": 0.0, "vetoed_n": 0, "vetoed_pnl": 0.0, "n": 0})

    bar_idx = 0
    for (et, trade) in trade_events:
        # Feed bars up to et into classifier + LFG
        while bar_idx < len(all_bars_for_day) and all_bars_for_day[bar_idx][0] <= et:
            bts, bpx = all_bars_for_day[bar_idx]
            rc_mod.update_regime_classifier(bts, bpx)
            lfg_mod.notify_bar(bts, bpx)
            bar_idx += 1
        # Query veto
        regime = rc_mod.current_regime()
        side = str(trade.get("side", "")).upper()
        entry_price = float(trade.get("entry_price", 0.0) or 0.0)
        pnl = float(trade.get("pnl_dollars", 0.0) or 0.0)
        signal = {
            "side": side,
            "entry_price": entry_price,
            "sub_strategy": trade.get("sub_strategy", ""),
        }
        veto, reason = g.should_veto_entry(signal, et)
        base_pnl += pnl
        per_regime[regime]["base"] += pnl
        per_regime[regime]["n"] += 1
        if veto and "chart_" in reason:  # only count chart rule vetoes
            vetoed.append({"et": et, "side": side, "pnl": pnl, "reason": reason})
            per_regime[regime]["vetoed_n"] += 1
            per_regime[regime]["vetoed_pnl"] += pnl
        else:
            rule_pnl += pnl

    return base_pnl, rule_pnl, vetoed, per_regime


def run_set(sources):
    # Load all trades + bars per folder
    grand_base = 0.0
    grand_rule = 0.0
    grand_vetoed = []
    grand_per_regime = defaultdict(lambda: {"base": 0.0, "vetoed_n": 0, "vetoed_pnl": 0.0, "n": 0})
    day_count = 0
    trade_count = 0

    for src, tag in sources:
        folder = find_folder_with_data(Path(src) if src.startswith("/") else REPORT_ROOT / src)
        if folder is None:
            print(f"  [skip] {src}")
            continue
        trades = json.loads((folder / "closed_trades.json").read_text(encoding="utf-8"))
        bars = load_bars(folder / "topstep_live_bot.log")
        # Group bars + trades by ET date
        trades_by_day = defaultdict(list)
        for t in trades:
            try:
                d = parse_ts(t["entry_time"]).astimezone(NY).date().isoformat()
            except Exception:
                continue
            trades_by_day[d].append(t)
        bars_by_day = defaultdict(list)
        for b in bars:
            bars_by_day[b[0].date().isoformat()].append(b)

        for day, day_trades in sorted(trades_by_day.items()):
            day_bars = bars_by_day.get(day, [])
            if len(day_bars) < 30:
                # Not enough bars to warm up classifier; still count base P&L
                grand_base += sum(float(t.get("pnl_dollars", 0.0) or 0.0) for t in day_trades)
                grand_rule += sum(float(t.get("pnl_dollars", 0.0) or 0.0) for t in day_trades)
                continue
            base, rule, vetoed, per_regime = simulate_day(folder, day_trades, day_bars)
            grand_base += base
            grand_rule += rule
            grand_vetoed.extend(vetoed)
            for k, v in per_regime.items():
                grand_per_regime[k]["base"] += v["base"]
                grand_per_regime[k]["vetoed_n"] += v["vetoed_n"]
                grand_per_regime[k]["vetoed_pnl"] += v["vetoed_pnl"]
                grand_per_regime[k]["n"] += v["n"]
            day_count += 1
            trade_count += len(day_trades)
    return {
        "days": day_count,
        "trades": trade_count,
        "baseline_pnl": round(grand_base, 2),
        "rule_pnl": round(grand_rule, 2),
        "delta": round(grand_rule - grand_base, 2),
        "vetoed_count": len(grand_vetoed),
        "vetoed_winners": sum(1 for v in grand_vetoed if v["pnl"] > 0),
        "vetoed_losers": sum(1 for v in grand_vetoed if v["pnl"] < 0),
        "vetoed_pnl_removed": round(sum(v["pnl"] for v in grand_vetoed), 2),
        "per_regime": {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in grand_per_regime.items()},
    }


def format_report(label, r):
    print(f"\n{'=' * 80}")
    print(f"{label} ({r['days']} days, {r['trades']} trades)")
    print("=" * 80)
    print(f"  baseline P&L (no rule): ${r['baseline_pnl']:+.2f}")
    print(f"  with Filter F:          ${r['rule_pnl']:+.2f}")
    print(f"  delta:                  ${r['delta']:+.2f}")
    print(f"  vetoed {r['vetoed_count']} trades ({r['vetoed_winners']}W / {r['vetoed_losers']}L)")
    print(f"  vetoed trades' baseline P&L: ${r['vetoed_pnl_removed']:+.2f}")
    print(f"\n  per regime:")
    for regime, data in sorted(r["per_regime"].items()):
        print(f"    {regime:<12}  n={data['n']:>4}  base=${data['base']:>+10.2f}  "
              f"vetoed={data['vetoed_n']:>3}  vetoed_$=${data['vetoed_pnl']:>+10.2f}")


if __name__ == "__main__":
    print("[run] In-sample 2025 (136 days)...")
    r25 = run_set(SOURCES_2025)
    format_report("2025 136-day set (IN-SAMPLE, training target)", r25)

    print("\n[run] OOS April 2026...")
    r26 = run_set(SOURCES_OOS)
    format_report("April 2026 (OUT-OF-SAMPLE)", r26)

    # Save
    out = ROOT / "backtest_reports" / "validate_filter_f_end_to_end.json"
    out.write_text(json.dumps({"in_sample_2025": r25, "oos_apr2026": r26}, indent=2), encoding="utf-8")
    print(f"\n[write] {out}")
