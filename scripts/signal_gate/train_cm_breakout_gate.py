"""Train the Kalshi cross-market breakout gate (ML replacement for the
hand-tuned `vix < 20 AND mnq_5m > 0.10% AND corr > 0.30` rule in
kalshi_trade_overlay.py::build_trade_plan).

Target: binary classifier `p_win` = P(trade hits its TP before SL)
given the full context at signal time — Kalshi state, bar state, regime,
signal side, and the 8 cross-market features.

Used at inference as a second-opinion override: when Kalshi's entry gate
says BLOCK (support_score < threshold - buffer), consult this model; if
p_win is high enough, lift the block.

Training data:
  artifacts/signal_gate_2025/kalshi_training_data.parquet  (1977 rows,
  label balanced ~52/48, covers 2025-01 to 2025-12).

Artifact:
  artifacts/signal_gate_2025/model_cm_breakout_gate.joblib
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.signal_gate.retrain_with_encoder import (
    compute_cross_market_features,
)

ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"

NUMERIC = [
    # Kalshi state — what Kalshi saw at signal time
    "entry_probability", "probe_probability", "momentum_delta",
    "momentum_retention", "support_score", "threshold", "probe_distance_pts",
    "et_hour_frac",
    # Signal sub-strategy tags
    "sub_tier", "sub_is_rev", "sub_is_5min",
    # Bar / regime state
    "atr14_pts", "range_30bar_pts", "trend_20bar_pct",
    "dist_to_20bar_hi_pct", "dist_to_20bar_lo_pct", "vel_5bar_pts_per_min",
    "dist_to_bank_pts", "regime_vol_bp", "regime_eff",
]
CATEGORICAL = ["side", "role", "regime"]


def encode_categorical(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    parts = []
    for c in cols:
        known = sorted(df[c].astype(str).unique().tolist())
        enc = pd.DataFrame(
            {f"{c}__{v}": (df[c].astype(str) == v).astype(int) for v in known},
            index=df.index,
        )
        parts.append(enc)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def build_matrix(ds: pd.DataFrame):
    ds = ds.dropna(subset=NUMERIC + CATEGORICAL + ["label"]).reset_index(drop=True)
    y = ds["label"].astype(int).values

    # Cross-market features joined by ts
    print("[cm] computing cross-market features...")
    cm_df = compute_cross_market_features(ds, ts_col="ts")

    parts = [ds[NUMERIC].astype(np.float32)]
    parts.append(encode_categorical(ds, CATEGORICAL))
    parts.append(cm_df)
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, y, ds, cm_df


def rolling_ab(X, y):
    """Rolling-origin A/B: 6 windows, train_frac starting 0.40, step 0.10."""
    n = len(X)
    rows = []
    for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        tr_end = int(n * t)
        te_end = min(n, int(n * (t + 0.10)))
        if te_end - tr_end < 50:
            continue
        X_tr, X_te = X.iloc[:tr_end], X.iloc[tr_end:te_end]
        y_tr, y_te = y[:tr_end], y[tr_end:te_end]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=30, random_state=42,
        ).fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
        auc = float(roc_auc_score(y_te, p))
        # Count hypothetical "override wins": signals this model rates >= 0.60
        # that actually won. That's the sub-population the live override
        # would unblock on.
        mask_override = p >= 0.60
        n_override = int(mask_override.sum())
        n_override_wins = int(y_te[mask_override].sum()) if n_override else 0
        rows.append({
            "train_frac": t, "test_frac": t + 0.10,
            "n_train": tr_end, "n_test": te_end - tr_end,
            "auc": auc,
            "p>=0.60_n": n_override,
            "p>=0.60_wins": n_override_wins,
            "p>=0.60_win_rate": (n_override_wins / n_override) if n_override else float("nan"),
        })
        print(f"  {int(t*100):>3}% → {int((t+0.1)*100):>3}%  AUC={auc:.3f}  "
              f"p≥0.60: {n_override} signals, {n_override_wins} winners "
              f"({(n_override_wins/n_override*100) if n_override else 0:.0f}%)")
    return rows


def main():
    parquet = ARTIFACTS / "kalshi_training_data.parquet"
    ds = pd.read_parquet(parquet).sort_values("ts").reset_index(drop=True)
    print(f"[cm-gate] {len(ds)} rows  ({ds['ts'].min()} → {ds['ts'].max()})")

    X, y, ds, cm_df = build_matrix(ds)
    print(f"[cm-gate] X shape: {X.shape}  (features={X.shape[1]})")
    print(f"[cm-gate] class balance: 1={y.sum()}  0={len(y)-y.sum()}  "
          f"win_rate={y.mean():.3f}")

    print("\n=== rolling-origin A/B ===")
    rows = rolling_ab(X, y)
    import statistics as s
    mean_auc = s.mean(r["auc"] for r in rows)
    print(f"\nMEAN AUC across {len(rows)} chunks: {mean_auc:.3f}")

    # Final model trained on everything
    print("\n[cm-gate] training final model on all data...")
    clf_full = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=30, random_state=42,
    ).fit(X, y)

    # Derive categorical_maps and feature layout so ml_overlay_shadow._build_row
    # (or an equivalent inference helper) can construct rows properly.
    cat_maps = {c: sorted(ds[c].astype(str).unique().tolist()) for c in CATEGORICAL}
    full_numeric = list(NUMERIC) + list(cm_df.columns)

    payload = {
        "model": clf_full,
        "model_kind": "GBT_d3_cm_breakout_gate_v1",
        "feature_names": list(X.columns),
        "numeric_features": full_numeric,
        "categorical_features": CATEGORICAL,
        "categorical_maps": cat_maps,
        "ordinal_features": [],
        "uses_cross_market": True,
        "uses_bar_encoder": False,  # doesn't use bar encoder — too few rows
        "cross_market_keys": list(cm_df.columns),
        "override_threshold": 0.60,  # p_win ≥ this triggers override
        "cv_auc_mean": mean_auc,
        "cv_auc_source": "rolling_origin_ab (6 chunks, kalshi_training_data)",
        "training_rows": len(ds),
        "win_rate_base": float(y.mean()),
    }
    out = ARTIFACTS / "model_cm_breakout_gate.joblib"
    joblib.dump(payload, out)
    print(f"[write] {out}")

    # Also persist rolling-origin results JSON for reference
    results_path = ARTIFACTS / "cm_breakout_gate_ab_results.json"
    results_path.write_text(json.dumps({"chunks": rows, "mean_auc": mean_auc}, indent=2))
    print(f"[write] {results_path}")


if __name__ == "__main__":
    main()
