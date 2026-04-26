#!/usr/bin/env python3
"""Quick diagnostic: does the deployed AetherFlow model have any edge on
the OOS slice at ANY threshold, or is the simulator just unfavorable?

If higher thresholds produce monotonically-improving avg PnL, the model
has edge and the calibrator is restoring trades in the noise-floor zone.
If higher thresholds ALSO show flat/negative avg PnL, either the model
has lost edge entirely OR the crude simulator's fixed 6pt/4pt brackets
mismatch the live bracket distribution badly enough to invalidate results.
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
from aetherflow_features import build_feature_frame
from aetherflow_model_bundle import predict_bundle_probabilities
from tools.ai_loop import price_context

sys.path.insert(0, str(ROOT / "scripts"))
from aetherflow_calibrator_fit_and_validate import (
    features_for, estimate_outcomes, trade_stats, OOS_START, OOS_END,
)

with (ROOT / "model_aetherflow_deploy_2026oos.pkl").open("rb") as fh:
    bundle = pickle.load(fh)
prices = price_context.load_prices().sort_index()

import pandas as pd
print(f"[diag] OOS slice {OOS_START} → {OOS_END}")
oos_bars, oos_feat_full = features_for(prices, OOS_START, OOS_END)
print(f"  all rows: {len(oos_feat_full):,}")
# Filter to candidate rows (candidate_side != 0) — the live bot only
# predicts on these
_side = pd.to_numeric(oos_feat_full.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
oos_feat = oos_feat_full.loc[(_side != 0).values]
print(f"  candidate rows: {len(oos_feat):,} ({100*len(oos_feat)/len(oos_feat_full):.1f}%)")
probs = predict_bundle_probabilities(bundle, oos_feat)
probs = probs[np.isfinite(probs)]
oos_feat = oos_feat.iloc[:len(probs)]
pnl = estimate_outcomes(oos_bars, oos_feat).values

print(f"\n  overall feature distribution: "
      f"mean={probs.mean():.4f} p50={np.median(probs):.4f} p90={np.percentile(probs, 90):.4f}")

# Threshold sweep
print(f"\n  threshold sweep on crude fixed-bracket simulator (TP=6, SL=4):")
print(f"  {'thr':>6}  {'n':>7}  {'PnL':>10}  {'WR':>7}  {'avg':>9}  {'DD':>8}")
for thr in (0.45, 0.48, 0.50, 0.52, 0.53, 0.54, 0.55, 0.555, 0.56, 0.57, 0.58, 0.60):
    mask = probs >= thr
    if mask.sum() == 0:
        continue
    s = trade_stats(pnl[mask])
    print(f"  {thr:>6.3f}  {s['n']:>7d}  ${s['pnl']:>+9.2f}  {s['wr']:>6.2f}%  "
          f"${s['avg']:>+7.3f}  ${s['dd']:>7.0f}")

# Also break down by setup_family at the current deployed threshold (0.55)
mask_55 = probs >= 0.55
fam = oos_feat.loc[:, "setup_family"].astype(str)
print(f"\n  at threshold 0.55 ({int(mask_55.sum())} rows), setup_family breakdown:")
for f, group in oos_feat.groupby("setup_family"):
    fam_mask = (oos_feat["setup_family"] == f).values
    fam_probs = probs[fam_mask]
    fam_pnl = pnl[fam_mask]
    at_thr = fam_probs >= 0.55
    if at_thr.sum() == 0:
        continue
    wr = (fam_pnl[at_thr] > 0).sum() / max(1, at_thr.sum()) * 100
    avg = fam_pnl[at_thr].mean() if at_thr.sum() else 0
    print(f"    {f:<22}  n@0.55={int(at_thr.sum()):>4}  WR={wr:.1f}%  avg=${avg:+.3f}")
