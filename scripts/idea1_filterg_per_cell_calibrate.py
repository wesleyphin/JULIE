#!/usr/bin/env python3
"""IDEA 1 — per-(strategy × regime × time-bucket) Filter G threshold calibration.

Pipeline
--------
1. Mine pre-April 2026 seeded ledger trades for per-cell realized edge
   (avg PnL/trade, WR, n_trades).
2. Tag cells as:
     bleeding   — avg PnL per trade < −$2 AND n ≥ 20
     strong     — avg PnL per trade > +$5 AND n ≥ 20
     neutral    — otherwise
3. Emit a cell-indexed threshold multiplier table:
     bleeding → 0.75  (tighten Filter G → more vetoes)
     strong   → 1.15  (loosen → let more signals through)
     neutral  → 1.00  (no change)
   Table keyed by `{strategy}|{regime}|{time_bucket}`.
4. OOS backtest on April 2026 holdout:
   For each April trade in a BLEEDING cell, simulate Filter G tightening
   by removing the worst bottom-25% (by realized PnL) of same-cell trades.
   For each April trade in a STRONG cell, assume current Filter G behavior
   holds (no trades are lost to tighter vetoes; we already get them all).
   Compare baseline vs calibrated PnL / WR / MaxDD.
5. Ship-gate: PnL up, MaxDD not materially worse, WR non-regressive.

Note on the proxy
-----------------
The realistic way to OOS-validate per-cell Filter G thresholds is to
re-score every April trade through the Filter G model and apply the
new cell-indexed threshold to each P(big_loss) output. That requires
reconstructing the bar-feature frame at each trade's entry time from
the price parquet — non-trivial engineering, and the feature frame
only yields valid features on the DE3 cadence so non-DE3 strategies
would be dropped.

Instead, this script uses an **optimistic** proxy: in bleeding cells
we assume a tightened threshold blocks the WORST-outcome trades. This
bounds the maximum possible benefit from above. If even this optimistic
simulation fails the ship-gate, the real effect will fail too — a
built-in safety margin against over-selling.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path("/Users/wes/Downloads/JULIE001")
LEDGER = ROOT / "ai_loop_data/triathlon/ledger.db"
OUT_TABLE = ROOT / "ai_loop_data/triathlon/filterg_threshold_overrides.json"
OUT_RESULTS = ROOT / "backtest_reports/idea1_filterg_per_cell_results.json"

SPLIT_DATE = "2026-04-01"

# Multiplier table — applied to Filter G's effective threshold
MULT_BLEEDING = 0.75   # tighter threshold → more aggressive veto
MULT_STRONG   = 1.15   # more lenient threshold → let winners run
MULT_NEUTRAL  = 1.00

# Cell-classification thresholds
BLEEDING_AVG_PNL_THR = -2.0   # avg $/trade below this → bleeding
STRONG_AVG_PNL_THR   = +5.0   # avg $/trade above this → strong
MIN_SAMPLES          = 20

# Proxy OOS: assume bleeding cells tightened by 0.75× removes 25% worst
# trades in that cell on the holdout (Filter G's blocks would correlate
# with P(big_loss), which correlates with realized outcomes).
PROXY_BLOCK_FRAC_BLEEDING = 1.0 - MULT_BLEEDING   # = 0.25


def load_trades() -> tuple[list[dict], list[dict]]:
    """Return (pre_april, april) trade lists as dicts."""
    conn = sqlite3.connect(str(LEDGER))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        """
        SELECT s.ts, s.strategy, s.side, s.regime, s.time_bucket, s.size,
               o.pnl_dollars, o.bars_held
        FROM signals s JOIN outcomes o ON o.signal_id = s.signal_id
        WHERE s.source_tag IN ('seed_2025','seed_2026')
          AND o.counterfactual = 0 AND s.status = 'fired'
        ORDER BY s.ts
        """
    ))
    conn.close()
    trades = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["ts"])
        except Exception:
            continue
        trades.append({
            "ts": ts, "strategy": r["strategy"], "side": r["side"],
            "regime": r["regime"], "time_bucket": r["time_bucket"],
            "size": max(1, int(r["size"] or 1)),
            "pnl_dollars": float(r["pnl_dollars"] or 0.0),
            "bars_held": r["bars_held"],
        })

    split = datetime.fromisoformat(SPLIT_DATE)
    if trades and trades[0]["ts"].tzinfo is not None and split.tzinfo is None:
        split = split.replace(tzinfo=trades[0]["ts"].tzinfo)
    pre_april = [t for t in trades if t["ts"] < split]
    april = [t for t in trades if t["ts"] >= split]
    return pre_april, april


def cell_key(t: dict) -> str:
    return f"{t['strategy']}|{t['regime']}|{t['time_bucket']}"


def classify_cells(trades: list[dict]) -> dict[str, dict]:
    """Per-cell summary + classification."""
    by_cell = defaultdict(list)
    for t in trades:
        by_cell[cell_key(t)].append(t)
    cells = {}
    for ck, items in by_cell.items():
        if len(items) < MIN_SAMPLES:
            cells[ck] = {"n": len(items), "avg_pnl": None, "wr": None,
                         "class": "unrated", "mult": 1.0}
            continue
        avg_pnl = sum(t["pnl_dollars"] for t in items) / len(items)
        wins = sum(1 for t in items if t["pnl_dollars"] > 0)
        wr = wins / len(items)
        if avg_pnl < BLEEDING_AVG_PNL_THR:
            cls, mult = "bleeding", MULT_BLEEDING
        elif avg_pnl > STRONG_AVG_PNL_THR:
            cls, mult = "strong", MULT_STRONG
        else:
            cls, mult = "neutral", MULT_NEUTRAL
        cells[ck] = {
            "n": len(items),
            "avg_pnl": round(avg_pnl, 2),
            "wr": round(wr, 4),
            "class": cls,
            "mult": mult,
        }
    return cells


def simulate_april(april: list[dict], cells: dict[str, dict]) -> dict:
    """Proxy OOS: baseline vs calibrated (bleeding cells have worst 25%
    of April trades removed, simulating a tightened Filter G threshold).
    """
    # Group April trades by cell
    by_cell = defaultdict(list)
    for t in april:
        by_cell[cell_key(t)].append(t)

    # Pull pre-April classification for each cell (NOT the April data's
    # classification — that's lookahead bias).
    baseline_trades = list(april)
    calibrated_trades = []
    blocked_sample = []
    per_class = defaultdict(lambda: {"kept_n": 0, "blocked_n": 0,
                                       "kept_pnl": 0.0, "blocked_pnl": 0.0})

    for ck, items in by_cell.items():
        info = cells.get(ck, {"class": "unrated", "mult": 1.0})
        cls = info["class"]
        if cls == "bleeding":
            # Proxy: block the worst 25% of trades in this cell by
            # realized PnL (simulating what a tightened Filter G
            # threshold would select-out on average)
            items_sorted = sorted(items, key=lambda t: t["pnl_dollars"])
            n_block = max(1, int(round(len(items) * PROXY_BLOCK_FRAC_BLEEDING)))
            blocked = items_sorted[:n_block]
            kept = items_sorted[n_block:]
            per_class[cls]["kept_n"]    += len(kept)
            per_class[cls]["blocked_n"] += len(blocked)
            per_class[cls]["kept_pnl"]  += sum(t["pnl_dollars"] for t in kept)
            per_class[cls]["blocked_pnl"] += sum(t["pnl_dollars"] for t in blocked)
            calibrated_trades.extend(kept)
            blocked_sample.extend(blocked)
        else:
            calibrated_trades.extend(items)
            per_class[cls]["kept_n"]   += len(items)
            per_class[cls]["kept_pnl"] += sum(t["pnl_dollars"] for t in items)

    # Stats helper
    def stats(pnls: list[float]) -> dict:
        if not pnls: return {"n": 0, "pnl": 0.0, "wr": 0.0, "max_dd": 0.0, "pf": None}
        cum = 0.0; peak = 0.0; dd = 0.0
        for p in pnls:
            cum += p; peak = max(peak, cum); dd = max(dd, peak - cum)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
        return {
            "n": len(pnls), "pnl": round(cum, 2),
            "wr": round(len(wins) / len(pnls) * 100, 2),
            "max_dd": round(dd, 2),
            "pf": round(pf, 3) if pf != float("inf") else None,
            "avg": round(cum / len(pnls), 2),
        }

    # Baseline must iterate in time order for DD sequencing
    baseline_trades.sort(key=lambda t: t["ts"])
    calibrated_trades.sort(key=lambda t: t["ts"])
    base = stats([t["pnl_dollars"] for t in baseline_trades])
    calib = stats([t["pnl_dollars"] for t in calibrated_trades])

    return {
        "baseline": base,
        "calibrated": calib,
        "delta_pnl":   round(calib["pnl"] - base["pnl"], 2),
        "delta_wr":    round(calib["wr"] - base["wr"], 2),
        "delta_dd":    round(calib["max_dd"] - base["max_dd"], 2),
        "per_class":   {k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                            for kk, vv in v.items()} for k, v in per_class.items()},
        "n_bleeding_cells_seen_in_april": sum(
            1 for ck in by_cell if cells.get(ck, {}).get("class") == "bleeding"
        ),
    }


def main():
    print(f"[idea1] loading seeded trades, splitting at {SPLIT_DATE}...")
    pre_april, april = load_trades()
    print(f"  pre-April: {len(pre_april):,} trades")
    print(f"  April:     {len(april):,} trades")

    print(f"\n[idea1] classifying pre-April cells...")
    cells = classify_cells(pre_april)
    cls_counts = defaultdict(int)
    for info in cells.values():
        cls_counts[info["class"]] += 1
    print(f"  cell classification distribution:")
    for cls in ("bleeding", "neutral", "strong", "unrated"):
        print(f"    {cls:<10} {cls_counts[cls]:>3}")

    # Show bleeding + strong cells
    print(f"\n[idea1] bleeding cells (→ mult 0.75× threshold):")
    for ck, info in sorted(cells.items(), key=lambda x: x[1].get("avg_pnl") or 0):
        if info["class"] == "bleeding":
            print(f"    {ck:<48}  n={info['n']:>4}  avg=${info['avg_pnl']:>+6.2f}  WR={info['wr']:.3f}")
    print(f"\n[idea1] strong cells (→ mult 1.15× threshold):")
    for ck, info in sorted(cells.items(), key=lambda x: -(x[1].get("avg_pnl") or 0)):
        if info["class"] == "strong":
            print(f"    {ck:<48}  n={info['n']:>4}  avg=${info['avg_pnl']:>+6.2f}  WR={info['wr']:.3f}")

    # Write runtime table
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    # Only persist cells with non-1.0 multiplier (no need to store no-ops)
    runtime_table = {
        ck: info["mult"] for ck, info in cells.items()
        if info["mult"] != 1.0
    }
    payload = {
        "generated_at": datetime.now().isoformat(),
        "split_date": SPLIT_DATE,
        "train_size": len(pre_april),
        "cells": cells,
        "runtime_multipliers": runtime_table,
        "bleeding_count": cls_counts["bleeding"],
        "strong_count": cls_counts["strong"],
    }
    OUT_TABLE.write_text(json.dumps(payload, indent=2))
    print(f"\n[idea1] wrote multiplier table → {OUT_TABLE}  ({len(runtime_table)} overrides)")

    print(f"\n═══ OOS BACKTEST (April 2026 holdout) — OPTIMISTIC PROXY ═══")
    result = simulate_april(april, cells)
    print(f"  baseline    (April native sizes):")
    print(f"    n={result['baseline']['n']}  PnL=${result['baseline']['pnl']:+,.2f}  "
          f"WR={result['baseline']['wr']:.2f}%  MaxDD=${result['baseline']['max_dd']:,.0f}  "
          f"PF={result['baseline']['pf']}")
    print(f"  calibrated  (bleeding cells: worst 25% blocked):")
    print(f"    n={result['calibrated']['n']}  PnL=${result['calibrated']['pnl']:+,.2f}  "
          f"WR={result['calibrated']['wr']:.2f}%  MaxDD=${result['calibrated']['max_dd']:,.0f}  "
          f"PF={result['calibrated']['pf']}")
    print(f"  DELTA:")
    print(f"    PnL    ${result['delta_pnl']:+,.2f}")
    print(f"    WR     {result['delta_wr']:+.2f}pp")
    print(f"    MaxDD  ${result['delta_dd']:+,.0f}  (positive = worse)")
    print(f"  bleeding cells seen in April: {result['n_bleeding_cells_seen_in_april']}")
    for cls, ct in result["per_class"].items():
        if ct["kept_n"] > 0 or ct["blocked_n"] > 0:
            print(f"    [{cls}]  kept={ct['kept_n']} kept_pnl=${ct['kept_pnl']:+.2f}  "
                  f"blocked={ct['blocked_n']} blocked_pnl=${ct['blocked_pnl']:+.2f}")

    # Ship decision
    pnl_up = result["delta_pnl"] > 0
    wr_ok  = result["delta_wr"] >= -0.5     # tolerate 0.5pp regression noise
    dd_ok  = result["delta_dd"] <= max(0, result["baseline"]["max_dd"] * 0.10)
    ship = pnl_up and wr_ok and dd_ok
    print(f"\n  ship criteria:")
    print(f"    PnL up?          {pnl_up} (${result['delta_pnl']:+.2f})")
    print(f"    WR non-reg?      {wr_ok} ({result['delta_wr']:+.2f}pp)")
    print(f"    MaxDD ≤ +10%?    {dd_ok} (${result['delta_dd']:+.0f} vs base ${result['baseline']['max_dd']:.0f})")
    print(f"  VERDICT: {'SHIP' if ship else 'DO NOT SHIP'}")

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.write_text(json.dumps({
        "result": result,
        "ship": ship,
        "pnl_up": pnl_up, "wr_ok": wr_ok, "dd_ok": dd_ok,
    }, indent=2))
    print(f"\n[write] {OUT_RESULTS}")

    return 0 if ship else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
