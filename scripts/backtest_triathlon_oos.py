#!/usr/bin/env python3
"""Out-of-sample validation of the Triathlon medal-sizing system.

Pipeline:
  1. Load every seeded trade from the ledger (5,237 rows across
     2025-full + 2026 Jan-Apr, with regime + time_bucket already
     classified).
  2. 75/25 chronological split.
  3. On TRAIN only: compute standings + medal per cell using the
     exact same logic `triathlon.medals.rescore_standings` uses.
     Freeze those medals.
  4. Replay HOLDOUT twice:
     A. baseline — use each trade's native size/PnL
     B. sized — apply the frozen medal's size multiplier to the trade,
        scale PnL proportionally (pnl_dollars × new_size / old_size).
  5. Report: total PnL / WR / MaxDD under each regime, plus per-medal
     realized-edge comparison on holdout.
  6. Forward-stability: for every rated train cell, compute its
     holdout-period realized edge; report Spearman correlation
     between train rank and holdout edge (per league).
  7. 5-fold rolling-window variant: 9-month train → 1-month test,
     roll forward by 1 month. Report mean ± std of PnL delta.

Output:
  backtest_reports/triathlon_oos_results.json
  Printed summary on stdout.

The script deliberately does NOT import triathlon.ledger or any live
state — it reconstructs the scoring from closed_trades.json so the
test is independent of any ledger mutations the live bot made.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path("/Users/wes/Downloads/JULIE001")

# Mirror the live Triathlon constants exactly
MIN_SAMPLES = 20
GOLD_PCTILE = 0.20
SILVER_PCTILE = 0.50
BRONZE_PCTILE = 0.80
PROBATION_PCTILE = 0.80   # bottom 20% in EVERY league

MEDAL_SIZE_MULT = {
    "gold": 1.50,
    "silver": 1.00,
    "bronze": 0.75,
    "probation": 0.50,
    "unrated": 1.00,
}

# ─── data loading ─────────────────────────────────────────────
def load_trades_from_ledger() -> list[dict]:
    """Join signals + outcomes from the triathlon ledger, filter to
    source_tag in {seed_2025, seed_2026}, return list of dicts."""
    import sqlite3
    db = ROOT / "ai_loop_data" / "triathlon" / "ledger.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute(
        """
        SELECT
            s.ts, s.strategy, s.side, s.regime, s.time_bucket,
            s.entry_price, s.size,
            o.pnl_dollars, o.bars_held
        FROM signals s
        JOIN outcomes o ON o.signal_id = s.signal_id
        WHERE s.source_tag IN ('seed_2025', 'seed_2026')
          AND o.counterfactual = 0
          AND s.status = 'fired'
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
            "ts": ts,
            "strategy": r["strategy"],
            "side": r["side"],
            "regime": r["regime"],
            "time_bucket": r["time_bucket"],
            "entry_price": r["entry_price"],
            "size": max(1, int(r["size"] or 1)),
            "pnl_dollars": float(r["pnl_dollars"] or 0.0),
            "bars_held": int(r["bars_held"]) if r["bars_held"] is not None else None,
        })
    return trades


def cell_key(strategy: str, regime: str, time_bucket: str) -> str:
    return f"{strategy}|{regime}|{time_bucket}"


# ─── scoring ──────────────────────────────────────────────────
@dataclass
class CellScore:
    cell_key: str
    n_fired: int
    purity: Optional[float]
    cash: Optional[float]
    velocity: Optional[float]

    def is_rated(self) -> bool:
        return all(v is not None for v in (self.purity, self.cash, self.velocity))


def score_cells(trades: list[dict]) -> dict[str, CellScore]:
    """Compute per-cell purity/cash/velocity from a given trade list.
    Mirrors triathlon.leagues.score_cell logic exactly."""
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_cell[cell_key(t["strategy"], t["regime"], t["time_bucket"])].append(t)
    scores: dict[str, CellScore] = {}
    for ck, items in by_cell.items():
        if len(items) < MIN_SAMPLES:
            scores[ck] = CellScore(ck, len(items), None, None, None)
            continue
        wins = [t for t in items if t["pnl_dollars"] > 0]
        purity = len(wins) / len(items)
        total_pnl = sum(t["pnl_dollars"] for t in items)
        total_size = sum(max(1, t["size"]) for t in items)
        cash = total_pnl / max(1, total_size)
        win_bars = [t["bars_held"] for t in wins if t["bars_held"] is not None and t["bars_held"] > 0]
        if not win_bars:
            velocity = None
        else:
            velocity = 1.0 / float(statistics.median(win_bars))
        scores[ck] = CellScore(
            ck, len(items),
            round(purity, 4), round(cash, 2),
            round(velocity, 4) if velocity is not None else None,
        )
    return scores


def assign_medals(scores: dict[str, CellScore]) -> dict[str, str]:
    """Compute per-cell medals. Same logic as triathlon.medals.assign_medal."""
    rated = [s for s in scores.values() if s.is_rated()]
    n_rated = len(rated)
    if n_rated == 0:
        return {ck: "unrated" for ck in scores}

    purity_sorted = sorted(rated, key=lambda s: (-(s.purity or 0), s.cell_key))
    cash_sorted = sorted(rated, key=lambda s: (-(s.cash or 0), s.cell_key))
    velocity_sorted = sorted(rated, key=lambda s: (-(s.velocity or 0), s.cell_key))

    p_rank = {s.cell_key: i+1 for i, s in enumerate(purity_sorted)}
    c_rank = {s.cell_key: i+1 for i, s in enumerate(cash_sorted)}
    v_rank = {s.cell_key: i+1 for i, s in enumerate(velocity_sorted)}

    medals: dict[str, str] = {}
    for ck, s in scores.items():
        if not s.is_rated():
            medals[ck] = "unrated"
            continue
        pp = p_rank[ck] / n_rated
        pc = c_rank[ck] / n_rated
        pv = v_rank[ck] / n_rated
        best = min(pp, pc, pv)
        if best >= PROBATION_PCTILE:
            medals[ck] = "probation"
        elif best <= GOLD_PCTILE:
            medals[ck] = "gold"
        elif best <= SILVER_PCTILE:
            medals[ck] = "silver"
        elif best <= BRONZE_PCTILE:
            medals[ck] = "bronze"
        else:
            medals[ck] = "silver"
    return medals, {"purity": p_rank, "cash": c_rank, "velocity": v_rank}


# ─── holdout replay ───────────────────────────────────────────
def replay_holdout(holdout: list[dict], medals: dict[str, str]) -> dict:
    """Walk holdout trades twice (baseline + sized), compute stats."""
    baseline_pnls = []
    sized_pnls = []
    sized_by_medal: dict[str, list[float]] = defaultdict(list)
    medals_applied = defaultdict(int)

    for t in holdout:
        ck = cell_key(t["strategy"], t["regime"], t["time_bucket"])
        medal = medals.get(ck, "unrated")
        mult = MEDAL_SIZE_MULT[medal]
        medals_applied[medal] += 1

        base_pnl = float(t["pnl_dollars"])
        old_size = max(1, t["size"])
        new_size = max(1, int(round(old_size * mult)))
        scaled_pnl = base_pnl * (new_size / old_size)

        baseline_pnls.append(base_pnl)
        sized_pnls.append(scaled_pnl)
        sized_by_medal[medal].append(scaled_pnl)

    def stats(pnls: list[float]) -> dict:
        if not pnls:
            return {"n": 0, "pnl": 0.0, "wr": 0.0, "max_dd": 0.0, "pf": None}
        cum = 0.0; peak = 0.0; dd = 0.0
        for p in pnls:
            cum += p; peak = max(peak, cum); dd = max(dd, peak - cum)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else float("inf")
        return {
            "n": len(pnls),
            "pnl": round(cum, 2),
            "wr": round(len(wins) / len(pnls) * 100, 2),
            "max_dd": round(dd, 2),
            "pf": round(pf, 3) if pf != float("inf") else None,
            "avg_trade": round(cum / len(pnls), 2),
            "wins": len(wins), "losses": len(losses),
        }

    base_stats = stats(baseline_pnls)
    sized_stats = stats(sized_pnls)

    per_medal = {}
    for medal in ("gold", "silver", "bronze", "probation", "unrated"):
        per_medal[medal] = {
            "n": medals_applied[medal],
            "total_pnl": round(sum(sized_by_medal[medal]), 2) if sized_by_medal[medal] else 0.0,
            "avg_pnl": round(statistics.mean(sized_by_medal[medal]), 2) if sized_by_medal[medal] else 0.0,
        }
    return {
        "baseline": base_stats,
        "sized": sized_stats,
        "delta_pnl": round(sized_stats["pnl"] - base_stats["pnl"], 2),
        "delta_wr": round(sized_stats["wr"] - base_stats["wr"], 2),
        "delta_dd": round(sized_stats["max_dd"] - base_stats["max_dd"], 2),
        "per_medal": per_medal,
    }


# ─── forward stability ────────────────────────────────────────
def spearman(xs: list[float], ys: list[float]) -> Optional[float]:
    """Spearman rank correlation. Returns None if either list has <3 items."""
    if len(xs) < 3 or len(ys) < 3 or len(xs) != len(ys):
        return None
    def ranks(vals):
        sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0] * len(vals)
        for rank, i in enumerate(sorted_idx):
            r[i] = rank + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - (6 * d2) / (n * (n**2 - 1))


def forward_stability(
    train: list[dict], holdout: list[dict],
    train_scores: dict[str, CellScore], train_ranks: dict[str, dict[str, int]],
) -> dict:
    """For each train-rated cell, compute holdout cash/purity/velocity
    and correlate against train rank. Spearman because ranks are ordinal."""
    holdout_by_cell: dict[str, list[dict]] = defaultdict(list)
    for t in holdout:
        holdout_by_cell[cell_key(t["strategy"], t["regime"], t["time_bucket"])].append(t)

    train_purity, holdout_purity = [], []
    train_cash, holdout_cash = [], []
    train_velocity, holdout_velocity = [], []
    cells_included = []

    for ck, ts in train_scores.items():
        if not ts.is_rated():
            continue
        ho = holdout_by_cell.get(ck, [])
        if len(ho) < 5:   # need at least 5 holdout trades to compute anything
            continue
        cells_included.append(ck)
        # Train side
        train_purity.append(ts.purity)
        train_cash.append(ts.cash)
        train_velocity.append(ts.velocity if ts.velocity is not None else 0.0)
        # Holdout side
        wins = [t for t in ho if t["pnl_dollars"] > 0]
        holdout_purity.append(len(wins) / len(ho))
        total_pnl = sum(t["pnl_dollars"] for t in ho)
        total_size = sum(max(1, t["size"]) for t in ho)
        holdout_cash.append(total_pnl / max(1, total_size))
        win_bars = [t["bars_held"] for t in wins if t["bars_held"] is not None and t["bars_held"] > 0]
        holdout_velocity.append(1.0 / statistics.median(win_bars) if win_bars else 0.0)

    return {
        "n_cells": len(cells_included),
        "purity_spearman":   spearman(train_purity,   holdout_purity),
        "cash_spearman":     spearman(train_cash,     holdout_cash),
        "velocity_spearman": spearman(train_velocity, holdout_velocity),
    }


# ─── single-split driver ──────────────────────────────────────
def run_single_split(trades: list[dict], train_frac: float = 0.75, *, verbose: bool = True) -> dict:
    n_train = int(len(trades) * train_frac)
    train = trades[:n_train]
    holdout = trades[n_train:]
    if verbose:
        print(f"[oos] train: {len(train)} trades  ({train[0]['ts'].date()} → {train[-1]['ts'].date()})")
        print(f"[oos] holdout: {len(holdout)} trades  ({holdout[0]['ts'].date()} → {holdout[-1]['ts'].date()})")

    train_scores = score_cells(train)
    medals, train_ranks = assign_medals(train_scores)
    if verbose:
        n_by_medal = defaultdict(int)
        for m in medals.values():
            n_by_medal[m] += 1
        print(f"[oos] train medals: {dict(n_by_medal)}")

    replay = replay_holdout(holdout, medals)
    stability = forward_stability(train, holdout, train_scores, train_ranks)
    return {
        "train_size": len(train), "holdout_size": len(holdout),
        "train_date_range": (train[0]["ts"].isoformat(), train[-1]["ts"].isoformat()),
        "holdout_date_range": (holdout[0]["ts"].isoformat(), holdout[-1]["ts"].isoformat()),
        "train_medal_counts": {m: sum(1 for v in medals.values() if v == m) for m in ("gold","silver","bronze","probation","unrated")},
        "replay": replay,
        "stability": stability,
    }


# ─── 5-fold rolling window ────────────────────────────────────
def run_rolling_folds(trades: list[dict], n_folds: int = 5) -> dict:
    """9-month train → 1-month test, roll forward 1 month each fold."""
    # Bucket trades by YYYY-MM
    by_month: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_month[t["ts"].strftime("%Y-%m")].append(t)
    months = sorted(by_month.keys())
    if len(months) < 10:
        return {"error": f"need at least 10 months, got {len(months)}"}

    folds = []
    for i in range(n_folds):
        train_months = months[i : i + 9]
        test_month = months[i + 9] if (i + 9) < len(months) else None
        if test_month is None:
            break
        train_trades = [t for m in train_months for t in by_month[m]]
        test_trades = by_month[test_month]
        if len(test_trades) < 20:
            continue

        train_scores = score_cells(train_trades)
        medals, _ = assign_medals(train_scores)
        replay = replay_holdout(test_trades, medals)
        folds.append({
            "fold": i + 1,
            "train_months": (train_months[0], train_months[-1]),
            "test_month": test_month,
            "n_train": len(train_trades),
            "n_test": len(test_trades),
            "baseline_pnl": replay["baseline"]["pnl"],
            "sized_pnl":    replay["sized"]["pnl"],
            "delta_pnl":    replay["delta_pnl"],
            "delta_wr":     replay["delta_wr"],
            "delta_dd":     replay["delta_dd"],
        })
    if not folds:
        return {"error": "no valid folds"}
    deltas = [f["delta_pnl"] for f in folds]
    return {
        "n_folds": len(folds),
        "folds": folds,
        "delta_pnl_mean": round(statistics.mean(deltas), 2),
        "delta_pnl_std":  round(statistics.pstdev(deltas), 2) if len(deltas) > 1 else 0.0,
        "delta_pnl_min":  min(deltas),
        "delta_pnl_max":  max(deltas),
        "n_positive":     sum(1 for d in deltas if d > 0),
        "n_negative":     sum(1 for d in deltas if d < 0),
    }


# ─── main ─────────────────────────────────────────────────────
def run_date_split(trades: list[dict], split_date: datetime, *, verbose: bool = True) -> dict:
    """Split trades by ts: train = ts < split_date, holdout = ts >= split_date.
    Same scoring/replay/stability logic as run_single_split but with a
    calendar-cut boundary instead of a fraction."""
    train = [t for t in trades if t["ts"] < split_date]
    holdout = [t for t in trades if t["ts"] >= split_date]
    if verbose:
        tr_r = (train[0]["ts"].date(), train[-1]["ts"].date()) if train else (None, None)
        ho_r = (holdout[0]["ts"].date(), holdout[-1]["ts"].date()) if holdout else (None, None)
        print(f"[oos] train:   {len(train)} trades  ({tr_r[0]} → {tr_r[1]})")
        print(f"[oos] holdout: {len(holdout)} trades  ({ho_r[0]} → {ho_r[1]})")

    train_scores = score_cells(train)
    medals, train_ranks = assign_medals(train_scores)
    if verbose:
        n_by_medal = defaultdict(int)
        for m in medals.values():
            n_by_medal[m] += 1
        print(f"[oos] train medals: {dict(n_by_medal)}")

    replay = replay_holdout(holdout, medals)
    stability = forward_stability(train, holdout, train_scores, train_ranks)
    return {
        "split_date": split_date.isoformat(),
        "train_size": len(train), "holdout_size": len(holdout),
        "train_date_range": [train[0]["ts"].isoformat(), train[-1]["ts"].isoformat()] if train else None,
        "holdout_date_range": [holdout[0]["ts"].isoformat(), holdout[-1]["ts"].isoformat()] if holdout else None,
        "train_medal_counts": {m: sum(1 for v in medals.values() if v == m) for m in ("gold","silver","bronze","probation","unrated")},
        "train_medals_by_cell": medals,
        "replay": replay,
        "stability": stability,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-date", default="2026-04-01",
                    help="ISO date: trades strictly before this go into train, on/after into holdout.")
    ap.add_argument("--save", default=str(ROOT / "backtest_reports" / "triathlon_oos_results.json"))
    args = ap.parse_args()

    # Localize split_date to match the trades' timezone (ledger stores
    # ET-localized ISO strings). Use the first trade's tzinfo as canonical.
    split_date = datetime.fromisoformat(args.split_date)
    print(f"[oos] loading seeded trades from ledger...")
    trades = load_trades_from_ledger()
    print(f"[oos] loaded {len(trades)} trades  ({trades[0]['ts'].date()} → {trades[-1]['ts'].date()})")
    # Normalize split_date to match the trades' tzinfo (ET). If trades
    # are tz-aware and split_date is naive, give it the same tzinfo.
    if trades and trades[0]["ts"].tzinfo is not None and split_date.tzinfo is None:
        split_date = split_date.replace(tzinfo=trades[0]["ts"].tzinfo)

    print(f"\n═══ CALENDAR SPLIT: pre-{args.split_date} TRAIN → {args.split_date}+ HOLDOUT ═══")
    result = run_date_split(trades, split_date)
    r = result["replay"]

    print(f"\n  baseline (April 2026, native sizes):")
    print(f"    PnL ${r['baseline']['pnl']:+,.2f}  WR {r['baseline']['wr']:.2f}%  "
          f"MaxDD ${r['baseline']['max_dd']:,.0f}  PF {r['baseline']['pf']}  "
          f"avg/trade ${r['baseline']['avg_trade']:+.2f}  (n={r['baseline']['n']})")
    print(f"  sized    (April 2026, Triathlon multipliers applied):")
    print(f"    PnL ${r['sized']['pnl']:+,.2f}  WR {r['sized']['wr']:.2f}%  "
          f"MaxDD ${r['sized']['max_dd']:,.0f}  PF {r['sized']['pf']}  "
          f"avg/trade ${r['sized']['avg_trade']:+.2f}")
    print(f"  DELTA:")
    print(f"    PnL    ${r['delta_pnl']:+,.2f}")
    print(f"    WR     {r['delta_wr']:+.2f}pp")
    print(f"    MaxDD  ${r['delta_dd']:+,.0f}  (positive = worse; negative = better)")

    print(f"\n  per-medal realized edge on April 2026 (sized PnL):")
    for medal in ("gold", "silver", "bronze", "probation", "unrated"):
        m = r["per_medal"][medal]
        print(f"    {medal:<10} n={m['n']:>4}  total=${m['total_pnl']:>+8.2f}  "
              f"avg/trade=${m['avg_pnl']:>+6.2f}")

    s = result["stability"]
    print(f"\n  forward-stability (pre-April rank → April value, Spearman):")
    print(f"    cells with ≥5 April trades: {s['n_cells']}")
    def fmt(v): return "None" if v is None else f"{v:+.3f}"
    print(f"    purity   spearman: {fmt(s['purity_spearman'])}")
    print(f"    cash     spearman: {fmt(s['cash_spearman'])}")
    print(f"    velocity spearman: {fmt(s['velocity_spearman'])}")

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"single": result}, indent=2, default=str))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
