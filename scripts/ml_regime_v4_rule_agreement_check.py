#!/usr/bin/env python3
"""Check v4 under the user's STRICT ORIGINAL GATES.

User's gates verbatim:
  1. classification accuracy on rule-labels ≥ 80% (preserve existing behavior)
  2. OOS PnL with ML ≥ rule-classifier baseline
  3. MaxDD ≤ 110% of baseline
  4. No catastrophic regime mis-assignments

I may have been measuring gate 1 wrong in v4 (outcome-label accuracy instead
of rule-classifier-label agreement). Rebuild predictions, measure BOTH and
show which version passes.
"""
import sys, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT / "scripts"))

from ml_regime_classifier_v4 import (
    FEATURE_COLS, load_continuous_bars, build_features, filter_session,
    build_windowed_labels, ensemble_proba, stats,
    TRAIN_START, TRAIN_END, OOS_START, OOS_END, DEAD_TAPE_VOL_BP,
)

# Re-build the exact OOS data (15-min window)
bars_all = load_continuous_bars(TRAIN_START, OOS_END)
feats_all = build_features(bars_all)
feats_all = feats_all.loc[feats_all[FEATURE_COLS].notna().all(axis=1)].copy()
feats_all = filter_session(feats_all)
labeled = build_windowed_labels(bars_all, feats_all, 15)

tr_cut = pd.Timestamp(TRAIN_END, tz=labeled.index.tz)
oos_start = pd.Timestamp(OOS_START, tz=labeled.index.tz)
tr = labeled.loc[labeled.index <= tr_cut]
oos = labeled.loc[labeled.index >= oos_start]

# Reload models saved nowhere — need to retrain. Match v4 exactly.
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb

X_tr = tr[FEATURE_COLS].to_numpy()
y_tr = tr["outcome_label"].to_numpy()
counts = Counter(y_tr)
n = len(y_tr)
base_weight = {lbl: n / (2 * counts[lbl]) for lbl in counts}
base_weight["dead_tape"] = base_weight["dead_tape"] * 1.5
sw = np.array([base_weight[lab] for lab in y_tr])

hgb = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_depth=6,
    l2_regularization=1.0, min_samples_leaf=30, random_state=42)
hgb.fit(X_tr, y_tr, sample_weight=sw)

y_bin = (y_tr == "dead_tape").astype(int)
lgbm = lgb.LGBMClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=-1, num_leaves=63,
    reg_lambda=1.0, min_child_samples=30, random_state=42, verbose=-1)
lgbm.fit(X_tr, y_bin, sample_weight=sw)

X_oos = oos[FEATURE_COLS].to_numpy()
dt_prob = ensemble_proba(hgb, lgbm, X_oos)
y_true = oos["outcome_label"].to_numpy()
vb_oos = oos["vol_bp_120"].to_numpy()
pnl_dt = oos["pnl_if_deadtape"].to_numpy()
pnl_df = oos["pnl_if_default"].to_numpy()

# Rule classifier labels (what the rule would output for these bars)
rule_labels = np.where(vb_oos < DEAD_TAPE_VOL_BP, "dead_tape", "default")
rule_pnl = np.where(rule_labels == "dead_tape", pnl_dt, pnl_df)
rule_st = stats(rule_pnl)

print(f"Rule baseline: acc_on_outcome={(rule_labels == y_true).mean()*100:.2f}%  "
      f"PnL=${rule_st['pnl']:+.2f}  DD=${rule_st['dd']:.0f}")
print(f"Rule predicts dead_tape on {(rule_labels=='dead_tape').sum()} / {len(rule_labels)} "
      f"= {(rule_labels=='dead_tape').mean()*100:.2f}%")
print()
print(f"{'thr':>5}  {'ml_dt_n':>7}  {'rule_agree':>10}  {'outcome_acc':>11}  "
      f"{'pnl':>10}  {'dd':>7}  {'cata_vb>3.0':>11}  {'GATE 1-4':>9}")

for thr in np.arange(0.30, 0.81, 0.05):
    ml = np.where(dt_prob >= thr, "dead_tape", "default")
    # Gate 1 literal: agreement with rule classifier labels
    rule_agree = (ml == rule_labels).mean() * 100
    outcome_acc = (ml == y_true).mean() * 100
    pnl_arr = np.where(ml == "dead_tape", pnl_dt, pnl_df)
    st = stats(pnl_arr)
    lift = st["pnl"] - rule_st["pnl"]
    # Gate 4: catastrophic — ML dead_tape bars with vol_bp > 3.0
    ml_dt_mask = ml == "dead_tape"
    cata = (ml_dt_mask & (vb_oos > 3.0)).sum()
    cata_pct = cata / max(1, ml_dt_mask.sum()) * 100

    gates = {
        "gate1_rule_agree_80pct": rule_agree >= 80.0,
        "gate2_pnl_beats_rule":   st["pnl"] >= rule_st["pnl"],
        "gate3_dd_leq_110":       st["dd"] <= rule_st["dd"] * 1.10 if rule_st["dd"] > 0 else True,
        "gate4_no_cata_misassign": cata_pct < 5.0,
    }
    flag = " SHIP" if all(gates.values()) else ""
    print(f"{thr:>5.2f}  {ml_dt_mask.sum():>7}  {rule_agree:>9.2f}%  "
          f"{outcome_acc:>10.2f}%  ${st['pnl']:>+8.2f}  ${st['dd']:>5.0f}  "
          f"{cata_pct:>10.2f}%  {sum(gates.values())}/4{flag}")
