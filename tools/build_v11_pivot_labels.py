"""Generate Pivot Trail ground-truth labels for the v11 training corpus.

For each row in artifacts/v11_training_corpus.parquet that is allowed by the
friend-rule filter, we walk the trade forward TWICE:

  1) ``pnl_no_pivot``   = stepped walk with pivot_active=False
  2) ``pnl_with_pivot`` = stepped walk with pivot_active=True

We then derive the "did Pivot actually help?" ground truth:

  pivot_arm_helped = pnl_with_pivot > pnl_no_pivot   (strict)

Label semantics for the v11 head:

  pivot_label = pivot_arm_helped (bool, cast to int)

i.e. the head outputs P(arming will help). Threshold sweep at deployment
applies "ARM if proba >= thr" — high proba → arm. This matches the
production deployment direction (the existing pivot_proba in the corpus
is also "arm if proba is high"; see julie001 ml_overlay_shadow gate at
line 11431: ``if _mls_pv.is_pivot_trail_live_active() and not _pt_ml_ratchet``
where _pt_ml_ratchet is the "skip" decision — i.e. low score means skip).

This direction is also more interpretable: a positive class is "Pivot adds
value" which lines up with the rest of the v11 family.

Sanity: the existing simulator path matches the corpus's recorded
``net_pnl_after_haircut`` for pivot_active=False (small differences possible
from contract-pinning edge cases — we trust the simulator and keep it as
the no-pivot baseline rather than the corpus column, so the comparison is
apples-to-apples).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pivot_stepped_sl_simulator import simulate_trade_with_pivot_trail  # noqa: E402

CORPUS_PATH = ROOT / "artifacts" / "v11_training_corpus.parquet"
MASTER_PATH = ROOT / "es_master_outrights.parquet"
OUT_PATH = ROOT / "artifacts" / "v11_pivot_labels.parquet"

HORIZON = 30
MULTIPLIER = 5.0
FEE = 7.50


def main() -> None:
    print(f"[pivot_labels] loading corpus: {CORPUS_PATH}")
    corpus = pd.read_parquet(CORPUS_PATH)
    print(f"[pivot_labels] corpus rows: {len(corpus)}")
    allowed = corpus[corpus["allowed_by_friend_rule"] == True].copy().reset_index(drop=True)
    print(f"[pivot_labels] allowed rows: {len(allowed)}")

    print(f"[pivot_labels] loading master: {MASTER_PATH}")
    master = pd.read_parquet(MASTER_PATH)

    out_rows = []
    t0 = time.time()
    for i, row in allowed.iterrows():
        ts = pd.Timestamp(row["ts"])
        side = str(row["side"]).upper()
        entry = float(row["entry_price"])
        sl_d = float(row["sl"])
        tp_d = float(row["tp"])
        contract = str(row["contract"])

        out_off = simulate_trade_with_pivot_trail(
            master, ts, side, entry, sl_d, tp_d,
            contract=contract, horizon_bars=HORIZON,
            pivot_active=False, multiplier=MULTIPLIER, fee=FEE,
        )
        out_on = simulate_trade_with_pivot_trail(
            master, ts, side, entry, sl_d, tp_d,
            contract=contract, horizon_bars=HORIZON,
            pivot_active=True, multiplier=MULTIPLIER, fee=FEE,
        )
        pnl_off = float(out_off["net_pnl_after_haircut"])
        pnl_on = float(out_on["net_pnl_after_haircut"])
        helped = bool(pnl_on > pnl_off + 1e-9)
        out_rows.append({
            "ts": ts,
            "strategy": str(row["strategy"]),
            "family": str(row["family"]),
            "side": side,
            "contract": contract,
            "entry_price": entry,
            "sl": sl_d,
            "tp": tp_d,
            "pivot_arm_proba": float(row.get("pivot_proba", float("nan"))),
            "pnl_no_pivot": pnl_off,
            "pnl_with_pivot": pnl_on,
            "delta_pnl": pnl_on - pnl_off,
            "pivot_arm_helped": helped,
            "pivot_label": int(helped),
            "exit_no_pivot": str(out_off["exit_reason"]),
            "exit_with_pivot": str(out_on["exit_reason"]),
            "pivot_armed_in_sim": bool(out_on["pivot_armed"]),
            "armed_at_bar": int(out_on["armed_at_bar"]),
            "max_mfe_pts": float(out_on["max_mfe_pts"]),
        })
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(allowed) - (i + 1)) / rate if rate > 0 else 0
            print(f"[pivot_labels] {i+1}/{len(allowed)}  rate={rate:.1f}/s  eta={eta:.0f}s")

    df = pd.DataFrame(out_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"[pivot_labels] wrote {len(df)} rows -> {OUT_PATH}")
    print()
    print("=== Label distribution ===")
    print(f"  pivot_arm_helped True : {int(df['pivot_arm_helped'].sum())} "
          f"({100*df['pivot_arm_helped'].mean():.1f}%)")
    print(f"  pivot_arm_helped False: {int((~df['pivot_arm_helped']).sum())} "
          f"({100*(1-df['pivot_arm_helped'].mean()):.1f}%)")
    print(f"  pivot_armed_in_sim True : {int(df['pivot_armed_in_sim'].sum())} "
          f"({100*df['pivot_armed_in_sim'].mean():.1f}%)")
    print()
    print("=== ΔPnL stats ===")
    helped = df[df["pivot_arm_helped"]]
    not_h = df[~df["pivot_arm_helped"]]
    print(f"  mean Δ when helped : ${helped['delta_pnl'].mean():.2f}  "
          f"median ${helped['delta_pnl'].median():.2f}  n={len(helped)}")
    print(f"  mean Δ when not    : ${not_h['delta_pnl'].mean():.2f}  "
          f"median ${not_h['delta_pnl'].median():.2f}  n={len(not_h)}")
    print()
    print("=== Always-pivot vs no-pivot global PnL ===")
    print(f"  Σ pnl_no_pivot   = ${df['pnl_no_pivot'].sum():.2f}")
    print(f"  Σ pnl_with_pivot = ${df['pnl_with_pivot'].sum():.2f}")
    print(f"  marginal value of always-pivot = ${df['delta_pnl'].sum():.2f}")
    print()
    print("By strategy:")
    by_strat = df.groupby("strategy").agg(
        n=("pivot_label", "size"),
        helped_pct=("pivot_arm_helped", "mean"),
        sum_no=("pnl_no_pivot", "sum"),
        sum_pivot=("pnl_with_pivot", "sum"),
    )
    by_strat["delta"] = by_strat["sum_pivot"] - by_strat["sum_no"]
    print(by_strat.to_string())


if __name__ == "__main__":
    main()
