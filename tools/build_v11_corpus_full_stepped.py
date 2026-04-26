"""Rebuild v11 corpus with FULL integrated stepped-SL (BE-arm + Pivot Trail).

Phase 6: walks every entry parameter from artifacts/v11_training_corpus_with_mfe.parquet
through tools/full_stepped_sl_simulator.simulate_full_stepped() with both BE-arm
and Pivot Trail active inside the bar walk (NOT post-hoc).

Adds new columns (preserves all original):
  - raw_pnl_full_stepped, net_pnl_full_stepped
  - exit_reason_full_stepped, exit_price_full_stepped
  - mfe_full_stepped, mae_full_stepped
  - be_armed, be_armed_at_bar
  - pivot_armed, pivot_armed_at_bar
  - final_sl_full_stepped

Output: artifacts/v11_corpus_full_stepped.parquet

Sanity checks emitted:
  - row count vs corpus
  - NaN counts in new columns
  - smoking-gun trade matches test result
  - aggregate vs prior baselines
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tools.full_stepped_sl_simulator import simulate_full_stepped  # noqa: E402

CORPUS_IN = ROOT / "artifacts/v11_training_corpus_with_mfe.parquet"
CORPUS_OUT = ROOT / "artifacts/v11_corpus_full_stepped.parquet"
BAR_PARQUET = ROOT / "es_master_outrights.parquet"

HORIZON = 30
POINT_VALUE = 5.0  # MES = $5/pt
HAIRCUT = 7.50


def _load_bars() -> pd.DataFrame:
    df = pd.read_parquet(BAR_PARQUET)
    if df.index.tz is None:
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    elif str(df.index.tz) != "US/Eastern":
        df.index = df.index.tz_convert("US/Eastern")
    return df


def _walk(bars_for_sym: pd.DataFrame, ts: pd.Timestamp, side: str,
          entry_price: float, sl_dist: float, tp_dist: float):
    """Run full-stepped simulator. Return dict of fields."""
    if str(side).upper() == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist
    fwd = bars_for_sym.loc[bars_for_sym.index > ts].head(HORIZON)
    if fwd.empty:
        return {
            "exit_reason": "no_data",
            "raw_pnl": 0.0,
            "exit_price": float(entry_price),
            "mfe": 0.0, "mae": 0.0,
            "be_armed": False, "be_armed_at_bar": -1,
            "pivot_armed": False, "pivot_armed_at_bar": -1,
            "final_sl": float(sl_price),
        }
    fwd_reset = fwd.reset_index()
    out = simulate_full_stepped(
        fwd_reset, side=side,
        entry_price=entry_price,
        initial_sl=sl_price, initial_tp=tp_price,
        be_arm_active=True, pivot_active=True,
        us_session_only=True,
    )
    raw_pnl = out.pnl_points * POINT_VALUE
    return {
        "exit_reason": out.exit_reason,
        "raw_pnl": float(raw_pnl),
        "exit_price": float(out.exit_price),
        "mfe": float(out.mfe_points),
        "mae": float(out.mae_points),
        "be_armed": bool(out.be_armed),
        "be_armed_at_bar": int(out.be_armed_at_bar),
        "pivot_armed": bool(out.pivot_armed),
        "pivot_armed_at_bar": int(out.pivot_armed_at_bar),
        "final_sl": float(out.final_sl),
    }


def main():
    t0 = time.time()
    print(f"[v11-stepped] reading corpus {CORPUS_IN}")
    df = pd.read_parquet(CORPUS_IN)
    print(f"[v11-stepped] corpus shape: {df.shape}")
    print(f"[v11-stepped] reading bars  {BAR_PARQUET}")
    bars = _load_bars()
    print(f"[v11-stepped] bars rows: {len(bars):,}")

    by_sym: dict[str, pd.DataFrame] = {
        sym: g for sym, g in bars.groupby("symbol")
    }
    print(f"[v11-stepped] symbols: {sorted(by_sym.keys())}")

    n = len(df)
    out_rows = []
    for i, row in df.iterrows():
        ts = pd.Timestamp(row["ts"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("US/Eastern")
        else:
            ts = ts.tz_convert("US/Eastern")
        side = str(row["side"])
        entry = float(row["entry_price"])
        sl_dist = float(row["sl"])
        tp_dist = float(row["tp"])
        contract = str(row["contract"])

        sym_bars = by_sym.get(contract)
        if sym_bars is None or sym_bars.empty:
            r = {
                "exit_reason": "no_data", "raw_pnl": 0.0,
                "exit_price": entry, "mfe": 0.0, "mae": 0.0,
                "be_armed": False, "be_armed_at_bar": -1,
                "pivot_armed": False, "pivot_armed_at_bar": -1,
                "final_sl": entry - sl_dist if side == "LONG" else entry + sl_dist,
            }
        else:
            r = _walk(sym_bars, ts, side, entry, sl_dist, tp_dist)

        out_rows.append({
            "exit_reason_full_stepped": r["exit_reason"],
            "raw_pnl_full_stepped": r["raw_pnl"],
            "net_pnl_full_stepped": r["raw_pnl"] - HAIRCUT,
            "exit_price_full_stepped": r["exit_price"],
            "mfe_full_stepped": r["mfe"],
            "mae_full_stepped": r["mae"],
            "be_armed": r["be_armed"],
            "be_armed_at_bar": r["be_armed_at_bar"],
            "pivot_armed": r["pivot_armed"],
            "pivot_armed_at_bar": r["pivot_armed_at_bar"],
            "final_sl_full_stepped": r["final_sl"],
        })

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n}] elapsed={time.time()-t0:.1f}s")

    add_df = pd.DataFrame(out_rows, index=df.index)
    out = pd.concat([df, add_df], axis=1)

    print(f"[v11-stepped] rows: {len(out)}")
    print(f"[v11-stepped] new column NaN counts:")
    for col in ["raw_pnl_full_stepped", "net_pnl_full_stepped",
                "exit_reason_full_stepped", "be_armed", "pivot_armed",
                "final_sl_full_stepped"]:
        print(f"  {col}: NaN={out[col].isna().sum()}")

    n_be = int(out["be_armed"].sum())
    n_pivot = int(out["pivot_armed"].sum())
    n_both = int(((out["be_armed"]) & (out["pivot_armed"])).sum())
    print(f"[v11-stepped] BE-arm fired: {n_be} ({n_be/len(out)*100:.1f}%)")
    print(f"[v11-stepped] Pivot Trail armed: {n_pivot} ({n_pivot/len(out)*100:.1f}%)")
    print(f"[v11-stepped] Both fired: {n_both} ({n_both/len(out)*100:.1f}%)")

    # Holdout aggregate (allowed_by_friend_rule, ts >= 2026-01-01)
    df_h = out.copy()
    df_h["ts"] = pd.to_datetime(df_h["ts"])
    if df_h["ts"].dt.tz is None:
        df_h["ts"] = df_h["ts"].dt.tz_localize("US/Eastern")
    else:
        df_h["ts"] = df_h["ts"].dt.tz_convert("US/Eastern")
    df_h["ts_naive"] = df_h["ts"].dt.tz_localize(None)
    holdout = df_h[
        (df_h["ts_naive"] >= "2026-01-01")
        & (df_h["allowed_by_friend_rule"] == True)
    ].copy().sort_values("ts").reset_index(drop=True)
    print(f"\n[v11-stepped] HOLDOUT (allowed_by_friend_rule, 2026+): n={len(holdout)}")

    # Aggregate gates: WR, PnL, DD using net_pnl_full_stepped
    s = holdout["net_pnl_full_stepped"].astype(float)
    equity = s.cumsum()
    dd = float((equity.cummax() - equity).max()) if len(s) else 0.0
    pnl_total = float(s.sum())
    wr = float((s > 0).mean()) if len(s) else 0.0
    print(f"[v11-stepped] holdout aggregate: PnL=${pnl_total:.2f}  DD=${dd:.2f}  WR={wr:.4%}")

    # Smoking gun
    sg = out[
        (out["ts"].astype(str).str.startswith("2026-03-05 08:05"))
        & (out["contract"] == "ESH6")
    ]
    print(f"[v11-stepped] smoking gun rows: {len(sg)}")
    if len(sg):
        for _, r in sg.iterrows():
            print(
                f"  side={r['side']} entry={r['entry_price']} sl={r['sl']} tp={r['tp']}"
                f"  exit_reason_full_stepped={r['exit_reason_full_stepped']}"
                f"  raw_pnl_full_stepped={r['raw_pnl_full_stepped']:.2f}"
                f"  be_armed={r['be_armed']} pivot_armed={r['pivot_armed']}"
                f"  mfe={r['mfe_full_stepped']:.2f}"
            )

    print(f"[v11-stepped] writing {CORPUS_OUT}")
    out.to_parquet(CORPUS_OUT)
    print(f"[v11-stepped] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
