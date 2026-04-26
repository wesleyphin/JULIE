"""Rebuild v11 corpus with MFE/MAE columns + SL=6pt re-walk variants.

Reads the existing v11 corpus (3,438 rows) for entry parameters (ts, side,
entry_price, sl, tp, contract) and re-walks the simulator with the patched
simulate_trade_through() that returns mfe_points / mae_points.

Adds columns:
  - mfe_points, mae_points        (re-walked outcomes)
  - exit_reason_resim, raw_pnl_resim (sanity: must match original)
  - sl6_exit_reason, sl6_raw_pnl, sl6_mfe_points, sl6_mae_points
        (re-walked with SL distance shrunk to 6pt; only for DE3 rows whose
         original |sl| was ~10pt — RA's 1.5/4pt brackets are left as-is.)

Output: artifacts/v11_training_corpus_with_mfe.parquet (separate file from
the original; do NOT overwrite).

Usage: python tools/build_v11_corpus_with_mfe.py
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

from simulator_trade_through import simulate_trade_through

CORPUS_IN = ROOT / "artifacts/v11_training_corpus.parquet"
CORPUS_OUT = ROOT / "artifacts/v11_training_corpus_with_mfe.parquet"
BAR_PARQUET = ROOT / "es_master_outrights.parquet"

HORIZON = 30
POINT_VALUE = 5.0  # MES
HAIRCUT = 7.50

# SL tightening variant: shrink DE3 SL from 10pt to 6pt and re-walk.
SL_TIGHT_PT = 6.0
DE3_NORMAL_SL = 10.0  # rows where |sl - this| <= 0.01 are eligible for tightening
DE3_REDUCED_SL = 5.0  # some rows already use 5pt (shutdown/scale variants)


def _load_bars() -> pd.DataFrame:
    df = pd.read_parquet(BAR_PARQUET)
    if df.index.tz is None:
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        elif df.index.name == "timestamp":
            pass
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    elif str(df.index.tz) != "US/Eastern":
        df.index = df.index.tz_convert("US/Eastern")
    return df


def _bars_for_contract(bars: pd.DataFrame, contract: str) -> pd.DataFrame:
    return bars[bars["symbol"] == contract]


def _walk(bars_for_sym: pd.DataFrame, ts: pd.Timestamp, side: str,
          entry_price: float, sl_dist: float, tp_dist: float):
    """Run simulator and return (exit_reason, raw_pnl_dollars, mfe_pts, mae_pts)."""
    if str(side).upper() == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist
    fwd = bars_for_sym.loc[bars_for_sym.index > ts].head(HORIZON)
    if fwd.empty:
        return "no_data", 0.0, 0.0, 0.0
    fwd_reset = fwd.reset_index()
    out = simulate_trade_through(
        fwd_reset, side=side, entry_price=entry_price,
        tp_price=tp_price, sl_price=sl_price,
    )
    raw_pnl = out.pnl_points * POINT_VALUE
    return out.exit_reason, raw_pnl, out.mfe_points, out.mae_points


def main():
    t0 = time.time()
    print(f"[v11-mfe] reading corpus {CORPUS_IN}")
    df = pd.read_parquet(CORPUS_IN)
    print(f"[v11-mfe] corpus shape: {df.shape}")
    print(f"[v11-mfe] reading bars  {BAR_PARQUET}")
    bars = _load_bars()
    print(f"[v11-mfe] bars rows: {len(bars):,}")

    # Group bars by symbol once
    by_sym: dict[str, pd.DataFrame] = {
        sym: g for sym, g in bars.groupby("symbol")
    }
    print(f"[v11-mfe] symbols: {sorted(by_sym.keys())}")

    n = len(df)
    out_rows = []
    sanity_mismatches = 0
    sl6_eligible = 0
    sl6_walked = 0
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
        family = str(row["family"])

        sym_bars = by_sym.get(contract)
        if sym_bars is None or sym_bars.empty:
            mfe = 0.0; mae = 0.0; resim_reason = "no_data"; resim_pnl = 0.0
        else:
            resim_reason, resim_pnl, mfe, mae = _walk(
                sym_bars, ts, side, entry, sl_dist, tp_dist
            )

        # Sanity check
        if resim_reason != row["exit_reason"]:
            sanity_mismatches += 1

        # SL=6pt variant: only for DE3 rows whose original SL was 10pt
        # (the 5pt and other RA brackets we leave alone).
        sl6_reason = None
        sl6_pnl = float("nan")
        sl6_mfe = float("nan")
        sl6_mae = float("nan")
        eligible_sl6 = (family != "regimeadaptive") and (abs(sl_dist - DE3_NORMAL_SL) < 0.01)
        if eligible_sl6:
            sl6_eligible += 1
            if sym_bars is not None and not sym_bars.empty:
                sl6_reason, sl6_pnl, sl6_mfe, sl6_mae = _walk(
                    sym_bars, ts, side, entry, SL_TIGHT_PT, tp_dist
                )
                sl6_walked += 1

        out_rows.append({
            "mfe_points": float(mfe),
            "mae_points": float(mae),
            "exit_reason_resim": resim_reason,
            "raw_pnl_resim": float(resim_pnl),
            "sl6_eligible": bool(eligible_sl6),
            "sl6_exit_reason": sl6_reason,
            "sl6_raw_pnl": float(sl6_pnl) if sl6_pnl == sl6_pnl else float("nan"),
            "sl6_mfe_points": float(sl6_mfe) if sl6_mfe == sl6_mfe else float("nan"),
            "sl6_mae_points": float(sl6_mae) if sl6_mae == sl6_mae else float("nan"),
        })

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{n}] elapsed={time.time()-t0:.1f}s mismatches={sanity_mismatches} sl6_walked={sl6_walked}")

    add_df = pd.DataFrame(out_rows, index=df.index)
    out = pd.concat([df, add_df], axis=1)

    print(f"[v11-mfe] sanity mismatches (resim vs original exit_reason): {sanity_mismatches}/{n}")
    print(f"[v11-mfe] SL=6pt eligible rows: {sl6_eligible}, walked: {sl6_walked}")

    # Sanity: MFE/MAE NaN counts
    print(f"[v11-mfe] MFE NaN: {out['mfe_points'].isna().sum()}, MAE NaN: {out['mae_points'].isna().sum()}")
    print(f"[v11-mfe] MFE describe: \n{out['mfe_points'].describe()}")
    print(f"[v11-mfe] MAE describe: \n{out['mae_points'].describe()}")

    # MFE >= BE_thr? compute per-row
    out["be_arm_threshold_pts"] = (out["tp"] * 0.40)
    out["mfe_crosses_be_arm"] = out["mfe_points"] >= out["be_arm_threshold_pts"]
    n_stop = (out["exit_reason"] == "stop").sum()
    n_stop_be = ((out["exit_reason"] == "stop") & out["mfe_crosses_be_arm"]).sum()
    print(f"[v11-mfe] stop rows: {n_stop}; of those MFE>=BE_thr: {n_stop_be} ({n_stop_be/max(1,n_stop)*100:.1f}%)")

    print(f"[v11-mfe] writing {CORPUS_OUT}")
    out.to_parquet(CORPUS_OUT)
    print(f"[v11-mfe] done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
