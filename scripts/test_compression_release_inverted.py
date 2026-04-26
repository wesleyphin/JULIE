#!/usr/bin/env python3
"""Test the 'contrarian compression_release' hypothesis.

Finding: on OOS, compression_release shows WR *decreasing* as model
confidence rises — 35% at thr=0.500, 0% at thr=0.545+. That anti-
correlation suggests the model's ranking is inverted on this family
in the current regime. If true, taking the OPPOSITE side at entry
should show WR *increasing* with confidence.

This is an honest test of whether 'just invert the side' would ship
compression_release at 60%+ WR. It would — IF the inversion is structural
and not just a small-sample coincidence.

Caveats the script prints:
  1. Sample size per bucket is small; CIs are wide.
  2. Inverted trades still pay commission + slippage on every fill.
  3. If the model self-corrects mid-deployment (e.g. after retrain),
     a contrarian deployment would flip from winning to losing.
  4. The bracket TP/SL is symmetric — inverted side has inverted
     SL/TP hits which the simulator already accounts for.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aetherflow_features import build_feature_frame
from aetherflow_model_bundle import predict_bundle_probabilities
from tools.ai_loop import price_context
from aetherflow_calibrator_fit_and_validate import (
    features_for, OOS_START, OOS_END,
)

# Same outcome walker as estimate_outcomes but accepts an explicit side
# override so we can flip and re-simulate.
MES_PT_VALUE = 5.0
TP_POINTS = 6.0
SL_POINTS = 4.0


def estimate_outcomes_with_side(bars, features, side_override):
    """side_override: +1 means take the model's side as-given,
                      -1 means invert every side (contrarian)."""
    if features.empty or bars.empty:
        return pd.Series(dtype=float)
    orig_side = pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0).values
    side = orig_side * side_override
    pnl = np.zeros(len(features), dtype=float)
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close_idx = close.index
    lookahead = 60
    for i, (ts, s) in enumerate(zip(features.index, side)):
        if s == 0:
            continue
        try:
            start_pos = close_idx.searchsorted(ts) + 1
        except Exception:
            continue
        end_pos = min(start_pos + lookahead, len(close))
        if start_pos >= end_pos:
            continue
        entry = float(close.iloc[start_pos - 1])
        if s > 0:  # LONG
            sl = entry - SL_POINTS
            tp = entry + TP_POINTS
            hit_sl = low.iloc[start_pos:end_pos].le(sl)
            hit_tp = high.iloc[start_pos:end_pos].ge(tp)
        else:      # SHORT
            sl = entry + SL_POINTS
            tp = entry - TP_POINTS
            hit_sl = high.iloc[start_pos:end_pos].ge(sl)
            hit_tp = low.iloc[start_pos:end_pos].le(tp)
        sl_i = hit_sl.values.argmax() if hit_sl.any() else 1 << 30
        tp_i = hit_tp.values.argmax() if hit_tp.any() else 1 << 30
        if sl_i == 1 << 30 and tp_i == 1 << 30:
            last = float(close.iloc[end_pos - 1])
            pts = (last - entry) if s > 0 else (entry - last)
            pnl[i] = pts * MES_PT_VALUE
        elif sl_i < tp_i:
            pnl[i] = -SL_POINTS * MES_PT_VALUE
        else:
            pnl[i] = TP_POINTS * MES_PT_VALUE
    return pd.Series(pnl, index=features.index)


with (ROOT / "model_aetherflow_deploy_2026oos.pkl").open("rb") as fh:
    bundle = pickle.load(fh)
prices = price_context.load_prices().sort_index()
oos_bars, oos_feat = features_for(prices, OOS_START, OOS_END)
_side = pd.to_numeric(oos_feat.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
oos_feat = oos_feat.loc[(_side != 0).values]
probs = predict_bundle_probabilities(bundle, oos_feat)
probs = probs[np.isfinite(probs)]
oos_feat = oos_feat.iloc[:len(probs)]

# Filter to compression_release only
cr_mask = (oos_feat["setup_family"].values == "compression_release")
cr_feat = oos_feat.loc[cr_mask]
cr_probs = probs[cr_mask]
cr_bars = oos_bars

pnl_asgiven = estimate_outcomes_with_side(cr_bars, cr_feat, side_override=+1).values
pnl_inverted = estimate_outcomes_with_side(cr_bars, cr_feat, side_override=-1).values

print(f"[compression_release inversion test on OOS {OOS_START} → {OOS_END}]")
print(f"  total candidate rows: {len(cr_feat)}")
print()
print(f"{'thr':>6}  {'n':>5}  {'AS-GIVEN':>25}  {'INVERTED':>25}")
print(f"{'':>6}  {'':>5}  {'WR':>8} {'avg':>8} {'PnL':>8}  {'WR':>8} {'avg':>8} {'PnL':>8}")
print("-" * 80)
for thr in [0.500, 0.510, 0.520, 0.525, 0.530, 0.535, 0.540, 0.545, 0.550, 0.560]:
    mask = cr_probs >= thr
    n = int(mask.sum())
    if n == 0:
        continue
    pa = pnl_asgiven[mask]
    pi = pnl_inverted[mask]
    wr_a = (pa > 0).sum() / n * 100
    wr_i = (pi > 0).sum() / n * 100
    avg_a = pa.mean() if n else 0
    avg_i = pi.mean() if n else 0
    sum_a = pa.sum()
    sum_i = pi.sum()
    marker = " ★" if wr_i >= 60 and n >= 20 else ""
    print(f"{thr:>6.3f}  {n:>5}  {wr_a:>7.1f}% ${avg_a:>+6.2f} ${sum_a:>+6.0f}   "
          f"{wr_i:>7.1f}% ${avg_i:>+6.2f} ${sum_i:>+6.0f}{marker}")

# Also compute this for aligned_flow + transition_burst — SANITY CHECK.
# A healthy family should show INVERTED has WR dropping with threshold
# (because we're inverting a correctly-ranked signal).
print()
print("=== sanity check: same test on aligned_flow (should NOT invert cleanly) ===")
af_mask = (oos_feat["setup_family"].values == "aligned_flow")
af_feat = oos_feat.loc[af_mask]
af_probs = probs[af_mask]
pnl_af_asgiven  = estimate_outcomes_with_side(oos_bars, af_feat, +1).values
pnl_af_inverted = estimate_outcomes_with_side(oos_bars, af_feat, -1).values
print(f"{'thr':>6}  {'n':>5}  {'WR-as-given':>14}  {'WR-inverted':>14}")
for thr in [0.500, 0.525, 0.550]:
    mask = af_probs >= thr
    n = int(mask.sum())
    if n == 0: continue
    wr_a = (pnl_af_asgiven[mask]  > 0).sum() / n * 100
    wr_i = (pnl_af_inverted[mask] > 0).sum() / n * 100
    print(f"{thr:>6.3f}  {n:>5}  {wr_a:>13.1f}%  {wr_i:>13.1f}%")
