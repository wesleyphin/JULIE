#!/usr/bin/env python3
"""H9 GapFade — overlay-modified bracket simulation.

The live bot rewrites every signal's brackets through two consecutive overlays:
  1. Dead-tape (regime_classifier.apply_scalp_brackets):
       TP = 3.0 pts, SL = 5.0 pts, size forced to 1, BE disabled.
       Triggers when the rule classifier flags 'dead_tape' regime.
  2. Kalshi trade overlay: further trims TP (typical −0.25pt).

For FibH1214 (designed 1.25pt TP) the rewrites are mostly irrelevant
because they roughly preserve the geometry. For H9 GapFade the designed
brackets are 7-10x bigger (21pt TP / 28pt SL at ES=7000) — the rewrites
WOULD dramatically clip the strategy.

This script simulates four scenarios on the same daily table:
  A. Designed brackets    (0.30% TP / 0.40% SL)        ← what we tested
  B. Dead-tape always     (3.0pt TP / 5.0pt SL, size=1)
  C. Dead-tape + Kalshi   (2.75pt TP / 5.0pt SL, size=1)
  D. Dead-tape on 40% of  trades (the typical live activation rate)

Method: re-walk the same OHLC bars per trade with the new TP/SL distances
applied as POINT distances (not percentages), and recompute outcomes.

Output: artifacts/h9_gapfade_ml/overlay_simulation.json
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "h9_gapfade_ml"
PARQUET = ROOT / "es_master_outrights-2.parquet"

# Match the live bot's overlay constants exactly.
DEAD_TP_PTS = 3.0
DEAD_SL_PTS = 5.0
KALSHI_TP_TRIM_PTS = 0.25  # typical post-Kalshi tightening
KALSHI_TP_PTS = DEAD_TP_PTS - KALSHI_TP_TRIM_PTS  # 2.75
KALSHI_SL_PTS = DEAD_SL_PTS  # Kalshi typically only trims TP

DEAD_TAPE_FORCED_SIZE = 1  # dead-tape forces size=1 (vs designed 10 MES)
ORIGINAL_SIZE = 10
DOLLAR_PER_PT_MES = 5.0
COMMISSION = 1.50
HORIZON_MIN = 30

# Calendar for path simulation
ROLL_CALENDAR = []  # populated lazily

# ---------------------------------------------------------------------------
# Reuse the front-month builder from the training pipeline.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(ROOT / "tools"))
from h9_gapfade_ml import build_front_month  # noqa: E402


def label_period(ts):
    if ts < pd.Timestamp("2017-01-20", tz=ts.tz):
        return "pre_t1"
    if ts < pd.Timestamp("2021-01-20", tz=ts.tz):
        return "trump1"
    if ts < pd.Timestamp("2025-01-20", tz=ts.tz):
        return "biden"
    return "trump2"


def relabel_with_brackets(front, daily, tp_pts, sl_pts):
    """For each fired-signal day, walk forward HORIZON_MIN bars from 09:30
    using FIXED-POINT brackets (tp_pts / sl_pts) and relabel outcomes."""
    f = front.copy()
    f["date"] = f.index.tz_convert("US/Eastern").date
    f["minute"] = f.index.minute
    f["hour"] = f.index.hour
    rth_bars = f[((f["hour"] == 9) & (f["minute"] >= 30)) | (f["hour"] == 10)].copy()
    rth_bars = rth_bars.sort_index()
    bars_by_date = dict(list(rth_bars.groupby("date")))

    new_long_lbl = np.zeros(len(daily), dtype=np.int8)
    new_short_lbl = np.zeros(len(daily), dtype=np.int8)
    new_long_pnl = np.zeros(len(daily), dtype=np.float64)
    new_short_pnl = np.zeros(len(daily), dtype=np.float64)

    for i, (date, row) in enumerate(daily.iterrows()):
        date_key = date.date()
        if date_key not in bars_by_date:
            continue
        bars = bars_by_date[date_key].iloc[:HORIZON_MIN]
        if bars.empty:
            continue
        entry = float(row["open_0930"])
        long_tp, long_sl = entry + tp_pts, entry - sl_pts
        short_tp, short_sl = entry - tp_pts, entry + sl_pts
        long_outcome = short_outcome = 0
        long_exit = short_exit = entry
        for _, b in bars.iterrows():
            hi, lo = float(b["high"]), float(b["low"])
            if long_outcome == 0:
                if lo <= long_sl:
                    long_outcome, long_exit = -1, long_sl
                elif hi >= long_tp:
                    long_outcome, long_exit = 1, long_tp
            if short_outcome == 0:
                if hi >= short_sl:
                    short_outcome, short_exit = -1, short_sl
                elif lo <= short_tp:
                    short_outcome, short_exit = 1, short_tp
            if long_outcome != 0 and short_outcome != 0:
                break
        if long_outcome == 0:
            long_exit = float(bars.iloc[-1]["close"])
        if short_outcome == 0:
            short_exit = float(bars.iloc[-1]["close"])
        new_long_lbl[i] = long_outcome
        new_short_lbl[i] = short_outcome
        new_long_pnl[i] = (long_exit - entry)
        new_short_pnl[i] = (entry - short_exit)
    return new_long_lbl, new_short_lbl, new_long_pnl, new_short_pnl


def backtest(daily, long_lbl, short_lbl, long_pnl, short_pnl, size, label):
    trades = []
    for i, (ts, row) in enumerate(daily.iterrows()):
        if row["fire_long"]:
            side, outcome = "LONG", int(long_lbl[i])
            pnl_pts = float(long_pnl[i])
        elif row["fire_short"]:
            side, outcome = "SHORT", int(short_lbl[i])
            pnl_pts = float(short_pnl[i])
        else:
            continue
        gross = pnl_pts * size * DOLLAR_PER_PT_MES
        net = gross - COMMISSION * size
        trades.append({
            "ts": ts, "side": side, "outcome": outcome,
            "pnl_pts": pnl_pts, "pnl_dollars_net": net,
            "period": label_period(ts), "size": size,
        })
    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, {"label": label, "n_trades": 0}
    cum = tr["pnl_dollars_net"].cumsum()
    dd = (cum - cum.cummax()).min()
    span_yrs = max(1, (pd.to_datetime(tr["ts"]).max() - pd.to_datetime(tr["ts"]).min()).days / 365.25)
    summary = {
        "label": label,
        "n_trades": int(len(tr)),
        "size_per_trade": int(size),
        "total_pnl": float(tr["pnl_dollars_net"].sum()),
        "annualized_pnl": float(tr["pnl_dollars_net"].sum() / span_yrs),
        "win_rate": float((tr["pnl_dollars_net"] > 0).mean() * 100),
        "avg_win": float(tr.loc[tr["pnl_dollars_net"] > 0, "pnl_dollars_net"].mean() if (tr["pnl_dollars_net"] > 0).any() else 0),
        "avg_loss": float(tr.loc[tr["pnl_dollars_net"] < 0, "pnl_dollars_net"].mean() if (tr["pnl_dollars_net"] < 0).any() else 0),
        "max_drawdown": float(abs(dd)),
        "by_side": tr.groupby("side")["pnl_dollars_net"].agg(["sum", "count"]).to_dict(),
        "by_period": tr.groupby("period")["pnl_dollars_net"].agg(["sum", "count"]).to_dict(),
    }
    return tr, summary


def main():
    print("Loading data ...")
    daily = pd.read_parquet(ART / "daily_table.parquet")
    front = build_front_month()
    print(f"daily: {len(daily)}  front: {len(front)}")

    results = {}

    # --- Scenario A: designed brackets (0.30% / 0.40%) — already labelled ---
    print("\n=== A: Designed brackets (0.30% TP / 0.40% SL, size=10) ===")
    _, sum_a = backtest(daily, daily["long_lbl"].values, daily["short_lbl"].values,
                          daily["long_pnl_pts"].values, daily["short_pnl_pts"].values,
                          ORIGINAL_SIZE, "A_designed")
    print_summary(sum_a)
    results["A_designed"] = sum_a

    # --- Scenario B: Dead-tape brackets always (3.0pt TP / 5.0pt SL, size=1) ---
    print("\n=== B: Dead-tape rewrite ALWAYS (3.0pt TP / 5.0pt SL, size=1) ===")
    bL, bS, bLp, bSp = relabel_with_brackets(front, daily, DEAD_TP_PTS, DEAD_SL_PTS)
    _, sum_b = backtest(daily, bL, bS, bLp, bSp, DEAD_TAPE_FORCED_SIZE, "B_dead_tape_always")
    print_summary(sum_b)
    results["B_dead_tape_always"] = sum_b

    # --- Scenario C: Dead-tape + Kalshi (2.75pt TP / 5.0pt SL, size=1) ---
    print(f"\n=== C: Dead-tape + Kalshi ({KALSHI_TP_PTS}pt TP / {KALSHI_SL_PTS}pt SL, size=1) ===")
    cL, cS, cLp, cSp = relabel_with_brackets(front, daily, KALSHI_TP_PTS, KALSHI_SL_PTS)
    _, sum_c = backtest(daily, cL, cS, cLp, cSp, DEAD_TAPE_FORCED_SIZE, "C_dead_tape_plus_kalshi")
    print_summary(sum_c)
    results["C_dead_tape_plus_kalshi"] = sum_c

    # --- Scenario D: Dead-tape on 40% of days (typical live rate) ---
    print("\n=== D: Dead-tape on 40% of days (rest get designed brackets) ===")
    np.random.seed(42)
    activation_mask = np.random.rand(len(daily)) < 0.40
    mixed_long = np.where(activation_mask, bL, daily["long_lbl"].values)
    mixed_short = np.where(activation_mask, bS, daily["short_lbl"].values)
    mixed_long_pnl = np.where(activation_mask, bLp, daily["long_pnl_pts"].values)
    mixed_short_pnl = np.where(activation_mask, bSp, daily["short_pnl_pts"].values)
    sizes = np.where(activation_mask, DEAD_TAPE_FORCED_SIZE, ORIGINAL_SIZE)

    trades = []
    for i, (ts, row) in enumerate(daily.iterrows()):
        if row["fire_long"]:
            side, outcome = "LONG", int(mixed_long[i])
            pnl_pts = float(mixed_long_pnl[i])
        elif row["fire_short"]:
            side, outcome = "SHORT", int(mixed_short[i])
            pnl_pts = float(mixed_short_pnl[i])
        else:
            continue
        size = int(sizes[i])
        gross = pnl_pts * size * DOLLAR_PER_PT_MES
        net = gross - COMMISSION * size
        trades.append({
            "ts": ts, "side": side, "outcome": outcome,
            "pnl_pts": pnl_pts, "pnl_dollars_net": net,
            "size": size, "dead_tape_active": bool(activation_mask[i]),
            "period": label_period(ts),
        })
    tr = pd.DataFrame(trades)
    cum = tr["pnl_dollars_net"].cumsum()
    dd = (cum - cum.cummax()).min()
    span_yrs = max(1, (pd.to_datetime(tr["ts"]).max() - pd.to_datetime(tr["ts"]).min()).days / 365.25)
    sum_d = {
        "label": "D_mixed_40pct_dead_tape",
        "n_trades": int(len(tr)),
        "n_dead_tape": int(tr["dead_tape_active"].sum()),
        "n_designed": int((~tr["dead_tape_active"]).sum()),
        "total_pnl": float(tr["pnl_dollars_net"].sum()),
        "annualized_pnl": float(tr["pnl_dollars_net"].sum() / span_yrs),
        "win_rate": float((tr["pnl_dollars_net"] > 0).mean() * 100),
        "avg_win": float(tr.loc[tr["pnl_dollars_net"] > 0, "pnl_dollars_net"].mean() if (tr["pnl_dollars_net"] > 0).any() else 0),
        "avg_loss": float(tr.loc[tr["pnl_dollars_net"] < 0, "pnl_dollars_net"].mean() if (tr["pnl_dollars_net"] < 0).any() else 0),
        "max_drawdown": float(abs(dd)),
        "pnl_designed_subset": float(tr.loc[~tr["dead_tape_active"], "pnl_dollars_net"].sum()),
        "pnl_dead_tape_subset": float(tr.loc[tr["dead_tape_active"], "pnl_dollars_net"].sum()),
    }
    print_summary(sum_d)
    results["D_mixed_40pct_dead_tape"] = sum_d

    out_path = ART / "overlay_simulation.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved → {out_path}")

    # Headline comparison
    print("\n" + "=" * 70)
    print("HEADLINE: H9 GapFade under live overlay scenarios")
    print("=" * 70)
    print(f"{'Scenario':<32} {'Trades':>7} {'Net PnL':>12} {'WR':>6} {'Max DD':>8}")
    print("-" * 70)
    for k, v in results.items():
        print(f"{k:<32} {v['n_trades']:>7} ${v['total_pnl']:>10,.0f} "
              f"{v['win_rate']:>5.1f}% ${v['max_drawdown']:>7,.0f}")


def print_summary(s):
    print(f"  trades:    {s.get('n_trades', 0)}")
    print(f"  net PnL:   ${s.get('total_pnl', 0):,.0f}")
    print(f"  annualized: ${s.get('annualized_pnl', 0):,.0f}/yr")
    print(f"  WR:        {s.get('win_rate', 0):.1f}%")
    print(f"  avg win:   ${s.get('avg_win', 0):,.2f}")
    print(f"  avg loss:  ${s.get('avg_loss', 0):,.2f}")
    print(f"  max DD:    ${s.get('max_drawdown', 0):,.0f}")


if __name__ == "__main__":
    main()
