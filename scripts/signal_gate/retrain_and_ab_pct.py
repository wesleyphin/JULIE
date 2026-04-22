"""Retrain PCT overlay with encoder + cross-market features, and score
the result with a rolling-origin A/B against v1.

PCT is a 3-class classifier (LBL_BREAKOUT=1 / LBL_PIVOT=0 / LBL_NEUTRAL=2
per pct_level_overlay convention; class encoding comes from the training
data). The A/B reports:
  - one-vs-rest AUC for class=1 (LBL_BREAKOUT), which is what the live
    inference path consumes via `p_bo`
  - overall 3-class accuracy on the test chunk

Artifact written: artifacts/signal_gate_2025/model_pct_overlay_v2.joblib
Results appended to: artifacts/signal_gate_2025/rolling_origin_ab_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.signal_gate.retrain_with_encoder import (
    compute_encoder_embeddings, compute_cross_market_features,
)

NY = ZoneInfo("America/New_York")
ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"

# Match the live inference schema (score_pct_overlay in ml_overlay_shadow.py)
# Note: `signed_level` and `abs_level` are missing from the training parquet —
# we derive them from `pct_from_open` on the fly.
NUMERIC = [
    "pct_from_open", "signed_level", "abs_level", "level_distance_pct",
    "atr_pct_30bar", "range_pct_at_touch", "hour_edge",
    "minutes_since_open", "dist_to_running_hi_pct", "dist_to_running_lo_pct",
    "rule_confidence",
]
CATEGORICAL = ["tier", "atr_bucket", "range_bucket", "hour_bucket", "direction"]

CLF_KWARGS = dict(n_estimators=250, max_depth=4, learning_rate=0.05,
                  min_samples_leaf=50, random_state=42)


def _encode_cat(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    parts = []
    for c in cols:
        known = sorted(df[c].astype(str).unique().tolist())
        enc = pd.DataFrame(
            {f"{c}__{v}": (df[c].astype(str) == v).astype(int) for v in known},
            index=df.index,
        )
        parts.append(enc)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def _build_matrices(ds: pd.DataFrame, *, include_aug: bool):
    """Return X, y, emb (for later save metadata) given the training dataframe.
    include_aug controls whether encoder + cross-market columns are added."""
    # Derive missing schema columns
    if "signed_level" not in ds.columns:
        ds = ds.assign(signed_level=ds["pct_from_open"].astype(float))
    if "abs_level" not in ds.columns:
        ds = ds.assign(abs_level=ds["signed_level"].abs())
    # PCT parquet doesn't have these live-only columns; fill zeros.
    for c in ("minutes_since_open", "dist_to_running_hi_pct", "dist_to_running_lo_pct"):
        if c not in ds.columns:
            ds[c] = 0.0
    # Rule confidence is always set
    if "rule_confidence" not in ds.columns:
        ds["rule_confidence"] = 0.0

    ds = ds.dropna(subset=[c for c in NUMERIC if c in ds.columns] + CATEGORICAL + ["label"]).reset_index(drop=True)
    y = ds["label"].astype(int).values

    parts = [ds[NUMERIC].astype(np.float32)]
    parts.append(_encode_cat(ds, CATEGORICAL))

    emb = None
    cm_df = None
    if include_aug:
        print("[pct] computing encoder embeddings...")
        emb = compute_encoder_embeddings(ds, ts_col="ts")
        print("[pct] computing cross-market features...")
        cm_df = compute_cross_market_features(ds, ts_col="ts")
        parts.append(cm_df)
        parts.append(pd.DataFrame(emb, columns=[f"enc_{i:02d}" for i in range(emb.shape[1])],
                                  index=ds.index).astype(np.float32))

    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, y, ds, emb, cm_df


def _auc_breakout(y_true, proba_matrix, clf) -> float:
    """One-vs-rest AUC for class=1 (LBL_BREAKOUT = breakout)."""
    if 1 not in list(clf.classes_):
        return float("nan")
    idx = list(clf.classes_).index(1)
    p = proba_matrix[:, idx]
    y_bin = (np.asarray(y_true) == 1).astype(int)
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    return float(roc_auc_score(y_bin, p))


def rolling_ab():
    ds_path = ARTIFACTS / "pct_overlay_training_data.parquet"
    ds = pd.read_parquet(ds_path).sort_values("ts").reset_index(drop=True)
    print(f"[pct] {len(ds)} rows")

    # Build v1 + v2 feature matrices on the same row order
    X_v1, y, ds, _, _ = _build_matrices(ds.copy(), include_aug=False)
    X_v2, _, _, emb, cm_df = _build_matrices(ds.copy(), include_aug=True)
    print(f"[pct] v1 dim={X_v1.shape[1]}  v2 dim={X_v2.shape[1]}")

    rows = []
    n = len(ds)
    for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        tr_end = int(n * t)
        te_end = min(n, int(n * (t + 0.10)))
        if te_end - tr_end < 100: continue
        X1_tr, X1_te = X_v1.iloc[:tr_end], X_v1.iloc[tr_end:te_end]
        X2_tr, X2_te = X_v2.iloc[:tr_end], X_v2.iloc[tr_end:te_end]
        y_tr, y_te = y[:tr_end], y[tr_end:te_end]

        clf1 = GradientBoostingClassifier(**CLF_KWARGS).fit(X1_tr, y_tr)
        clf2 = GradientBoostingClassifier(**CLF_KWARGS).fit(X2_tr, y_tr)
        p1 = clf1.predict_proba(X1_te)
        p2 = clf2.predict_proba(X2_te)

        auc1 = _auc_breakout(y_te, p1, clf1)
        auc2 = _auc_breakout(y_te, p2, clf2)
        acc1 = accuracy_score(y_te, clf1.predict(X1_te))
        acc2 = accuracy_score(y_te, clf2.predict(X2_te))

        rows.append({
            "train_frac": t, "test_frac": t + 0.10,
            "test_start": str(ds["ts"].iloc[tr_end]),
            "n_train": tr_end, "n_test": te_end - tr_end,
            "auc_v1": auc1, "auc_v2": auc2, "delta": auc2 - auc1,
            "acc_v1": float(acc1), "acc_v2": float(acc2),
        })
        print(f"  {int(t*100):>3}% → {int((t+0.1)*100):>3}%  "
              f"v1_auc={auc1:.3f} v2_auc={auc2:.3f} Δ={auc2-auc1:+.3f}  "
              f"v1_acc={acc1:.3f} v2_acc={acc2:.3f}")
    return rows, ds, emb, cm_df, X_v2, y


def save_v2(clf, ds, emb, cm_df, X_v2_cols, auc_v2_mean: float):
    # Build the correct metadata the first time so we don't need fixup
    cat_maps = {col: sorted(ds[col].astype(str).unique().tolist()) for col in CATEGORICAL}
    enc_cols = [f"enc_{i:02d}" for i in range(emb.shape[1])]
    full_numeric = list(NUMERIC) + list(cm_df.columns) + enc_cols
    payload = {
        "model": clf,
        "model_kind": "GBT_d4_pct_overlay_3class_v2_with_encoder_crossmarket",
        "classes": [int(c) for c in clf.classes_],
        "feature_names": list(X_v2_cols),
        "numeric_features": full_numeric,
        "categorical_features": CATEGORICAL,
        "categorical_maps": cat_maps,
        "ordinal_features": [],
        "uses_bar_encoder": True,
        "uses_cross_market": True,
        "encoder_embed_dim": emb.shape[1],
        "cross_market_keys": list(cm_df.columns),
        "cv_auc_mean": auc_v2_mean,  # one-vs-rest breakout AUC
        "cv_auc_source": "rolling_origin_ab_results.json (6 chunks, v2)",
    }
    out = ARTIFACTS / "model_pct_overlay_v2.joblib"
    joblib.dump(payload, out)
    print(f"[write] {out}")


def main():
    rows, ds, emb, cm_df, X_v2, y = rolling_ab()
    import statistics as s
    mean_v1 = s.mean(r["auc_v1"] for r in rows if not np.isnan(r["auc_v1"]))
    mean_v2 = s.mean(r["auc_v2"] for r in rows if not np.isnan(r["auc_v2"]))
    wins = sum(1 for r in rows if r["delta"] > 0)
    print(f"\nMEAN AUC(breakout-OvR): v1={mean_v1:.3f}  v2={mean_v2:.3f}  Δ={mean_v2-mean_v1:+.3f}")
    print(f"v2 beats v1 in AUC: {wins}/{len(rows)} chunks")

    # Fit a final v2 model on ALL data (like retrain_with_encoder does)
    # so we have a deployable artifact
    clf_full = GradientBoostingClassifier(**CLF_KWARGS).fit(X_v2, y)
    save_v2(clf_full, ds, emb, cm_df, X_v2.columns, mean_v2)

    # Append results to shared rolling-origin JSON
    out = ARTIFACTS / "rolling_origin_ab_results.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    data["pct_overlay"] = [
        {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in r.items()}
        for r in rows
    ]
    out.write_text(json.dumps(data, indent=2, default=str))
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
