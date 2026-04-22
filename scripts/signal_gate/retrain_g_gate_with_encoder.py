"""Retrain the per-strategy 'Machine Learning G' big-loss gate with
encoder + cross-market features.

Scope (2026-04-22): only the DE3 family has enough training rows in
the current v2-training-rows parquet (1680 DE3 vs 32 RegimeAdaptive).
AetherFlow's training rows live in a separate pipeline not built by
scripts/signal_gate/build_training_data_v2.py. This script retrains
DE3 only; AF / RA retrains are a follow-up once their data is built.

Target: big_loss (pnl_dollars ≤ -$100), binary classifier with
GradientBoosting. v2 adds:
  + 32-dim bar-sequence encoder embedding
  +  8-dim cross-market features (MNQ / VIX)

A/B harness: 6 rolling-origin chunks, both v1 and v2 trained from
scratch each chunk, AUC reported per chunk + mean + chunk-wins.
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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"
NY = ZoneInfo("America/New_York")

from scripts.signal_gate.retrain_with_encoder import (
    compute_encoder_embeddings, compute_cross_market_features,
)

# Feature schema — matches the existing v1 gate + same regime detection
NUMERIC = [
    "de3_entry_ret1_atr", "de3_entry_body_pos1", "de3_entry_body1_ratio",
    "de3_entry_lower_wick_ratio", "de3_entry_upper_wick_ratio",
    "de3_entry_upper1_ratio", "de3_entry_close_pos1",
    "de3_entry_flips5", "de3_entry_down3", "de3_entry_range10_atr",
    "de3_entry_dist_low5_atr", "de3_entry_dist_high5_atr",
    "de3_entry_vol1_rel20", "de3_entry_atr14",
    "de3_entry_velocity_30", "de3_entry_dist_low30_atr",
    "de3_entry_dist_high30_atr", "de3_entry_ret30_atr",
]
CATEGORICAL = ["side", "regime", "session"]
ORDINAL = ["et_hour"]

CLF_KWARGS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                  min_samples_leaf=30, random_state=42)


def _encode_cat(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    parts = []
    for c in cols:
        known = sorted(df[c].astype(str).unique().tolist())
        parts.append(pd.DataFrame(
            {f"{c}__{v}": (df[c].astype(str) == v).astype(int) for v in known},
            index=df.index,
        ))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def build_matrices(ds: pd.DataFrame, *, include_aug: bool):
    """Return (X, y, ds_clean, emb, cm_df) — v1 or v2 depending on include_aug."""
    ds = ds.dropna(subset=NUMERIC + CATEGORICAL + ORDINAL + ["big_loss"]).reset_index(drop=True)
    y = ds["big_loss"].astype(int).values
    parts = [ds[NUMERIC].astype(np.float32)]
    parts.append(_encode_cat(ds, CATEGORICAL))
    parts.append(ds[ORDINAL].astype(float))
    emb = None
    cm_df = None
    if include_aug:
        print("  computing encoder embeddings...")
        emb = compute_encoder_embeddings(ds, ts_col="entry_time")
        print("  computing cross-market features...")
        cm_df = compute_cross_market_features(ds, ts_col="entry_time")
        parts.append(cm_df)
        parts.append(pd.DataFrame(emb, columns=[f"enc_{i:02d}" for i in range(emb.shape[1])],
                                  index=ds.index).astype(np.float32))
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, y, ds, emb, cm_df


def rolling_ab(X_v1, X_v2, y):
    n = len(y)
    rows = []
    for t in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        tr_end = int(n * t)
        te_end = min(n, int(n * (t + 0.10)))
        if te_end - tr_end < 50: continue
        y_tr, y_te = y[:tr_end], y[tr_end:te_end]
        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            continue
        clf1 = GradientBoostingClassifier(**CLF_KWARGS).fit(X_v1.iloc[:tr_end], y_tr)
        clf2 = GradientBoostingClassifier(**CLF_KWARGS).fit(X_v2.iloc[:tr_end], y_tr)
        p1 = clf1.predict_proba(X_v1.iloc[tr_end:te_end])[:, list(clf1.classes_).index(1)]
        p2 = clf2.predict_proba(X_v2.iloc[tr_end:te_end])[:, list(clf2.classes_).index(1)]
        a1 = float(roc_auc_score(y_te, p1))
        a2 = float(roc_auc_score(y_te, p2))
        rows.append({"t": t, "n_tr": tr_end, "n_te": te_end - tr_end,
                     "auc_v1": a1, "auc_v2": a2, "delta": a2 - a1})
        print(f"  {int(t*100):>3}% → {int((t+0.1)*100):>3}%  "
              f"v1={a1:.3f} v2={a2:.3f} Δ={a2-a1:+.3f}")
    return rows


def main():
    parquet = ARTIFACTS / "training_rows_v2.parquet"
    ds_all = pd.read_parquet(parquet).sort_values("entry_time").reset_index(drop=True)
    ds = ds_all[ds_all["strategy"] == "DynamicEngine3"].reset_index(drop=True)
    print(f"[de3 gate] {len(ds)} DE3 rows  "
          f"({ds['entry_time'].min()} → {ds['entry_time'].max()})")
    print(f"[de3 gate] big_loss rate: {ds['big_loss'].mean():.3f}")

    print("\n[v1] building v1 feature matrix...")
    X_v1, y, ds, _, _ = build_matrices(ds.copy(), include_aug=False)
    print(f"[v1] shape: {X_v1.shape}")

    print("\n[v2] building v2 feature matrix with encoder + cross-market...")
    X_v2, _, _, emb, cm_df = build_matrices(ds.copy(), include_aug=True)
    print(f"[v2] shape: {X_v2.shape}")

    print("\n=== rolling-origin A/B ===")
    rows = rolling_ab(X_v1, X_v2, y)
    if not rows:
        print("insufficient per-chunk data for A/B")
        return
    import statistics as s
    mean_v1 = s.mean(r["auc_v1"] for r in rows)
    mean_v2 = s.mean(r["auc_v2"] for r in rows)
    wins = sum(1 for r in rows if r["delta"] > 0)
    print(f"\nMEAN AUC: v1={mean_v1:.3f}  v2={mean_v2:.3f}  Δ={mean_v2-mean_v1:+.3f}")
    print(f"v2 beats v1 in AUC: {wins}/{len(rows)} chunks")

    # Save v2 full-data fit with correct metadata layout (same
    # numeric_features / categorical_maps fix that bit us on LFO v2)
    clf_full = GradientBoostingClassifier(**CLF_KWARGS).fit(X_v2, y)
    cat_maps = {c: sorted(ds[c].astype(str).unique().tolist()) for c in CATEGORICAL}
    enc_cols = [f"enc_{i:02d}" for i in range(emb.shape[1])]
    full_numeric = list(NUMERIC) + list(cm_df.columns) + enc_cols
    out_path = ARTIFACTS / "model_de3_v2.joblib"
    joblib.dump({
        "model": clf_full,
        "model_kind": "GBT_d3_per_strategy_v2_with_encoder_crossmarket",
        "target": "big_loss",
        "veto_threshold": 0.35,
        "feature_names": list(X_v2.columns),
        "numeric_features": full_numeric,
        "categorical_features": CATEGORICAL,
        "categorical_maps": cat_maps,
        "ordinal_features": ORDINAL,
        "uses_bar_encoder": True,
        "uses_cross_market": True,
        "encoder_embed_dim": emb.shape[1],
        "cross_market_keys": list(cm_df.columns),
        "cv_auc_mean": mean_v2,
        "training_rows": len(ds),
        "training_date_utc": pd.Timestamp.utcnow().isoformat(),
    }, out_path)
    print(f"\n[write] {out_path}")

    # Append results to rolling-origin JSON
    results_path = ARTIFACTS / "rolling_origin_ab_results.json"
    data = json.loads(results_path.read_text()) if results_path.exists() else {}
    data["de3_gate"] = [
        {k: (str(v) if hasattr(v, "isoformat") else v) for k, v in r.items()}
        for r in rows
    ]
    results_path.write_text(json.dumps(data, indent=2, default=str))
    print(f"[write] {results_path} (merged de3_gate into {list(data.keys())})")

    # Promotion decision
    if wins >= 4 and mean_v2 > mean_v1 + 0.005:
        print(f"\n✅ Promotion criteria met (v2 wins {wins}/{len(rows)}, Δ={mean_v2-mean_v1:+.3f}).")
        print(f"   Run:  cp artifacts/signal_gate_2025/model_de3.joblib  artifacts/signal_gate_2025/model_de3_v1_pre_encoder.joblib")
        print(f"   Then: cp artifacts/signal_gate_2025/model_de3_v2.joblib  artifacts/signal_gate_2025/model_de3.joblib")
    else:
        print(f"\n❌ Keep v1 canonical — v2 wins {wins}/{len(rows)}, "
              f"Δ={mean_v2-mean_v1:+.3f} (need ≥4/6 wins AND Δ>+0.005).")


if __name__ == "__main__":
    main()
