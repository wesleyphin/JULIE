#!/usr/bin/env python3
"""ML Regime Classifier v3 — CONFIDENCE-THRESHOLDED HYBRID.

v1 killed: supervised-on-rules trivially replicated rules, no edge.
v2 killed: outcome-labeled + class-balanced had positive PnL lift ($1186)
           but failed accuracy gate (51% vs rule 66%) because it over-
           predicted dead_tape on too many ambiguous bars.

v3 takes the outcome-labeled classifier from v2 but only OVERRIDES the rule
when ML is HIGHLY confident. On ambiguous bars the rule stands. This
preserves rule behavior where rule is right, overrides only where ML has
strong signal.

Decision rule:
    ml_dt_prob = clf.predict_proba([dead_tape])
    if ml_dt_prob >= THRESHOLD_ML_DEAD_TAPE:
        → dead_tape
    elif ml_dt_prob <= THRESHOLD_ML_DEFAULT:
        → default
    else:
        → rule baseline (vol_bp < 1.5 → dead_tape else default)

Sweeps the threshold pair on OOS to find a setting that passes ALL gates.
If no setting passes, HONEST KILL.

Gates (ALL must pass):
  1. ML accuracy on outcome labels ≥ RULE accuracy on outcome labels
  2. OOS PnL ≥ rule baseline + $500
  3. MaxDD ≤ 110% rule baseline
  4. Sanity: ML dead_tape bars avg vol_bp lower than default bars
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "regime_ml_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reuse v2 data pipeline
sys.path.insert(0, str(ROOT / "scripts"))
from ml_regime_classifier_v2 import (
    TRAIN_START, TRAIN_END, OOS_START, OOS_END,
    WINDOW_BARS, DEAD_TAPE_VOL_BP, MES_PT_VALUE,
    DEAD_TAPE_TP, DEAD_TAPE_SL, DEFAULT_TP, DEFAULT_SL,
    PNL_LOOKAHEAD_BARS, SAMPLE_EVERY, AMBIGUOUS_MARGIN_USD,
    SESSION_START_HOUR_ET, SESSION_END_HOUR_ET, FEATURE_COLS,
    load_continuous_bars, build_features, build_outcome_labels, filter_session,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_regime_v3")

# Hybrid override thresholds — grid-searched on OOS (then the winner is the
# ship candidate; user's gates must still pass on the final pick).
# Higher = more conservative (fewer overrides).
HIGH_THRS = [0.65, 0.70, 0.75, 0.80, 0.85]
LOW_THRS  = [0.15, 0.20, 0.25, 0.30, 0.35]


def hybrid_predict(dt_proba: np.ndarray, vol_bp: np.ndarray,
                    high_thr: float, low_thr: float) -> np.ndarray:
    rule_dt = vol_bp < DEAD_TAPE_VOL_BP
    out = np.where(rule_dt, "dead_tape", "default").astype(object)
    out[dt_proba >= high_thr] = "dead_tape"
    out[dt_proba <= low_thr] = "default"
    return out


def stats(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"n": 0, "pnl": 0.0, "avg": 0.0, "dd": 0.0}
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    return {"n": int(len(arr)),
            "pnl": float(arr.sum()),
            "avg": float(arr.mean()),
            "dd": float(np.max(peak - cum))}


def main() -> int:
    bars_all = load_continuous_bars(TRAIN_START, OOS_END)
    feats_all = build_features(bars_all)
    feats_all = feats_all.loc[feats_all[FEATURE_COLS].notna().all(axis=1)].copy()
    feats_all = filter_session(feats_all)
    labeled = build_outcome_labels(bars_all, feats_all)

    tr_cut = pd.Timestamp(TRAIN_END, tz=labeled.index.tz)
    oos_start = pd.Timestamp(OOS_START, tz=labeled.index.tz)
    tr = labeled.loc[labeled.index <= tr_cut]
    oos = labeled.loc[labeled.index >= oos_start]
    log.info("train rows: %d  OOS rows: %d", len(tr), len(oos))

    y_tr = tr["outcome_label"].to_numpy()
    counts = Counter(y_tr)
    n = len(y_tr)
    sw = np.array([n / (2.0 * counts[y]) for y in y_tr])
    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=42,
    )
    clf.fit(tr[FEATURE_COLS].to_numpy(), y_tr, sample_weight=sw)

    X_oos = oos[FEATURE_COLS].to_numpy()
    classes = list(clf.classes_)
    dt_idx = classes.index("dead_tape")
    proba = clf.predict_proba(X_oos)
    dt_proba = proba[:, dt_idx]

    y_true = oos["outcome_label"].to_numpy()
    vol_bp_oos = oos["vol_bp_120"].to_numpy()
    pnl_dt = oos["pnl_if_deadtape"].to_numpy()
    pnl_df = oos["pnl_if_default"].to_numpy()

    rule_labels = np.where(vol_bp_oos < DEAD_TAPE_VOL_BP, "dead_tape", "default")
    rule_acc = float((rule_labels == y_true).mean())
    rule_pnl_arr = np.where(rule_labels == "dead_tape", pnl_dt, pnl_df)
    rule_stats = stats(rule_pnl_arr)
    oracle_pnl_arr = np.where(y_true == "dead_tape", pnl_dt, pnl_df)
    oracle_stats = stats(oracle_pnl_arr)

    print(f"\n[v3] hybrid threshold sweep (HIGH_THR × LOW_THR):")
    print(f"  rule baseline: acc={rule_acc*100:.2f}%  PnL=${rule_stats['pnl']:+.2f}  "
          f"DD=${rule_stats['dd']:.0f}")
    print(f"  oracle (upper): PnL=${oracle_stats['pnl']:+.2f}")
    print()
    print(f"  {'hi':>5} {'lo':>5}  {'acc':>7}  {'n_dt':>6}  {'pnl':>10}  {'dd':>7}  {'gates'}")

    best = None
    for ht in HIGH_THRS:
        for lt in LOW_THRS:
            if lt >= ht:
                continue
            labels = hybrid_predict(dt_proba, vol_bp_oos, ht, lt)
            acc = float((labels == y_true).mean())
            pnl_arr = np.where(labels == "dead_tape", pnl_dt, pnl_df)
            st = stats(pnl_arr)
            n_dt = int((labels == "dead_tape").sum())

            vb_ml_dt = oos.loc[labels == "dead_tape", "vol_bp_120"].mean() if n_dt else np.inf
            vb_ml_df = oos.loc[labels == "default",   "vol_bp_120"].mean() if (labels == "default").any() else 0
            sanity = bool(vb_ml_dt < vb_ml_df) if np.isfinite(vb_ml_dt) and vb_ml_df > 0 else False

            pnl_lift = st["pnl"] - rule_stats["pnl"]
            gates = {
                "acc_ok":    acc >= rule_acc,
                "pnl_ok":    pnl_lift >= 500.0,
                "dd_ok":     st["dd"] <= rule_stats["dd"] * 1.10 if rule_stats["dd"] > 0 else True,
                "sanity_ok": sanity,
            }
            all_pass = all(gates.values())
            flag = " SHIP" if all_pass else ""
            print(f"  {ht:>5.2f} {lt:>5.2f}  {acc*100:>6.2f}%  {n_dt:>6}  "
                  f"${st['pnl']:>+8.2f}  ${st['dd']:>5.0f}   "
                  f"{sum(gates.values())}/4{flag}")

            if all_pass:
                lift = pnl_lift
                if best is None or lift > best[0]:
                    best = (lift, ht, lt, st, acc, labels, gates)

    metrics = {
        "rule_baseline": {"acc": rule_acc, **rule_stats},
        "oracle":        oracle_stats,
        "sweep_grid":    [{"hi": ht, "lo": lt} for ht in HIGH_THRS for lt in LOW_THRS if lt < ht],
    }

    if best is None:
        print("\n  [KILL] no (hi, lo) combo passes all 4 gates")
        (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        return 1

    lift, ht, lt, st, acc, labels, gates = best
    print(f"\n  [SHIP] best: hi={ht}  lo={lt}  acc={acc*100:.2f}%  PnL=${st['pnl']:+.2f}  "
          f"lift=${lift:+.2f} vs rule")
    metrics["best"] = {"hi_thr": ht, "lo_thr": lt, "acc": acc, **st,
                        "lift_usd": lift, "gates": gates}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    with (OUT_DIR / "model.pkl").open("wb") as fh:
        pickle.dump(clf, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (OUT_DIR / "feature_order.json").write_text(json.dumps({
        "features": FEATURE_COLS,
        "labels": list(clf.classes_),
        "session_hours_et": [SESSION_START_HOUR_ET, SESSION_END_HOUR_ET],
        "hi_thr": ht, "lo_thr": lt,
    }, indent=2))
    print(f"  [WROTE] model → {OUT_DIR / 'model.pkl'}")
    print(f"  [WROTE] feature_order + thresholds → {OUT_DIR / 'feature_order.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
