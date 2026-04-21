#!/usr/bin/env python3
"""v2 signal-gate trainer — uses REAL OHLCV features from parquet + 2026 data.

Expanded feature set (now includes previously-degenerate wick/body/volume
features that REAL OHLCV unlocks):
  +body1_ratio, body_pos1, close_pos1, lower/upper_wick_ratio, vol1_rel20

Temporal holdout: last 15% by date for a rough OOS check. No fully-clean
OOS is possible now that 2026 data is in training.

Output: artifacts/signal_gate_2025/model_v2.joblib (new file, doesn't
replace the v1 artifact until validation passes).
"""
from __future__ import annotations

import joblib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = Path("/Users/wes/Downloads/JULIE001")
OUT_DIR = ROOT / "artifacts" / "signal_gate_2025"

# Expanded numeric features — now including wick/body/volume since we have
# real OHLCV bars from the parquet.
NUMERIC_FEATURES = [
    "de3_entry_ret1_atr",
    "de3_entry_body_pos1",          # NEW
    "de3_entry_body1_ratio",        # NEW
    "de3_entry_close_pos1",         # NEW
    "de3_entry_lower_wick_ratio",   # NEW
    "de3_entry_upper_wick_ratio",   # NEW
    "de3_entry_down3",
    "de3_entry_flips5",
    "de3_entry_range10_atr",
    "de3_entry_dist_low5_atr",
    "de3_entry_dist_high5_atr",
    "de3_entry_vol1_rel20",         # NEW
    "de3_entry_velocity_30",
    "de3_entry_dist_low30_atr",
    "de3_entry_dist_high30_atr",
    "de3_entry_ret30_atr",
]
CATEGORICAL_FEATURES = ["side", "regime", "session"]
ORDINAL_FEATURES = ["et_hour"]


def _encode_categorical(series: pd.Series, known_values=None):
    if known_values is None:
        known_values = sorted(series.dropna().unique().tolist())
    out = {}
    for val in known_values:
        out[f"{series.name}__{val}"] = (series == val).astype(int)
    return pd.DataFrame(out, index=series.index), known_values


def assemble_X(df: pd.DataFrame, categorical_maps: dict | None = None):
    updated = dict(categorical_maps or {})
    parts = [df[NUMERIC_FEATURES].copy()]
    for col in CATEGORICAL_FEATURES:
        known = updated.get(col)
        encoded, known = _encode_categorical(df[col], known_values=known)
        updated[col] = known
        parts.append(encoded)
    parts.append(df[ORDINAL_FEATURES].astype(float))
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, updated


def eval_thresholds(df, y_proba, target="big_loss"):
    base = df["pnl_dollars"].sum()
    out = []
    for t in np.arange(0.20, 0.70, 0.025):
        if target == "big_loss":
            veto = y_proba >= t
        else:
            veto = y_proba < t
        kept = df.loc[~veto, "pnl_dollars"].sum()
        vetoed_pnl = df.loc[veto, "pnl_dollars"].sum()
        out.append({
            "thresh": round(float(t), 3),
            "vetoed": int(veto.sum()),
            "vetoed_wins": int(((df["pnl_dollars"] > 0) & veto).sum()),
            "vetoed_losses": int(((df["pnl_dollars"] < 0) & veto).sum()),
            "vetoed_pnl": round(float(vetoed_pnl), 2),
            "kept_pnl": round(float(kept), 2),
            "delta": round(float(kept - base), 2),
            "base_pnl": round(float(base), 2),
        })
    return out


def main():
    df = pd.read_parquet(OUT_DIR / "training_rows_v2.parquet")
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    df = df.dropna(subset=[c for c in required if c in df.columns]).reset_index(drop=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    print(f"[load] {len(df)} rows")
    print(f"  source: 2025={df['source_tag'].eq('2025').sum()}  2026={df['source_tag'].eq('2026').sum()}")
    print(f"  base P&L: ${df['pnl_dollars'].sum():+.2f}")

    X, cat_maps = assemble_X(df)
    print(f"[features] {X.shape[1]} columns (numerics={len(NUMERIC_FEATURES)}, "
          f"cat one-hots={X.shape[1] - len(NUMERIC_FEATURES) - len(ORDINAL_FEATURES)})")

    y_win = (df["pnl_dollars"] > 0).astype(int).values
    y_big = df["big_loss"].astype(int).values

    # Temporal holdout — last 15% by date
    split = int(len(df) * 0.85)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    df_te = df.iloc[split:]
    y_tr_big, y_te_big = y_big[:split], y_big[split:]
    y_tr_win, y_te_win = y_win[:split], y_win[split:]
    print(f"\n[split] temporal 85/15  train={len(X_tr)}  test={len(X_te)}  "
          f"test P&L: ${df_te['pnl_dollars'].sum():+.2f}")

    print("\n" + "=" * 80)
    print("CV on full data (5-fold stratified, target=big_loss)")
    print("=" * 80)
    cv_aucs = []
    cv_deltas = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y_big):
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.03,
                                          min_samples_leaf=25, random_state=42)
        clf.fit(X.iloc[tr], y_big[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        auc = roc_auc_score(y_big[te], p)
        cv_aucs.append(auc)
        # veto top 25% risk
        thr = np.percentile(p, 75)
        veto = p > thr
        pnl_te = df["pnl_dollars"].iloc[te].values
        cv_deltas.append(-pnl_te[veto].sum())
    print(f"CV AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")
    print(f"CV delta @ top-25% veto: ${np.mean(cv_deltas):+.2f} ± {np.std(cv_deltas):.2f}")

    print("\n" + "=" * 80)
    print("Temporal tail holdout (train on first 85%, test on last 15%)")
    print("=" * 80)
    configs = [
        ("GBT_d3", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20, random_state=42)),
        ("GBT_d5", GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, min_samples_leaf=25, random_state=42)),
        ("GBT_d5_big", GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.02, min_samples_leaf=30, random_state=42)),
        ("RF_d6", RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=15, random_state=42, n_jobs=-1)),
    ]
    best_config = None
    best_delta = -1e9
    for name, clf in configs:
        clf.fit(X_tr, y_tr_big)
        p_te = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te_big, p_te)
        results = eval_thresholds(df_te, p_te, target="big_loss")
        positive = [r for r in results if r["vetoed"] / max(1, len(df_te)) < 0.50]
        best = max(positive, key=lambda r: r["delta"]) if positive else max(results, key=lambda r: r["delta"])
        print(f"\n--- {name} (target=big_loss) ---")
        print(f"  OOS AUC: {auc:.3f}")
        print(f"  best: thresh={best['thresh']}, vetoed={best['vetoed']} "
              f"({best['vetoed_wins']}W/{best['vetoed_losses']}L), "
              f"delta=${best['delta']:+.2f}")
        if best["delta"] > best_delta:
            best_delta = best["delta"]
            best_config = (name, clf, best["thresh"], best, auc)

    print(f"\n[winner] {best_config[0]}  threshold={best_config[2]}  "
          f"tail-OOS delta=${best_config[3]['delta']:+.2f}  AUC={best_config[4]:.3f}")

    # Retrain winner on ALL data and save
    print(f"\n[retrain] full dataset (n={len(df)}) using {best_config[0]} config")
    name, clf_spec, threshold, best_row, auc_holdout = best_config
    # Re-init with same config
    if "GBT_d5_big" in name:
        final = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.02, min_samples_leaf=30, random_state=42)
    elif "GBT_d5" in name:
        final = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, min_samples_leaf=25, random_state=42)
    elif "GBT_d3" in name:
        final = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20, random_state=42)
    else:
        final = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=15, random_state=42, n_jobs=-1)
    final.fit(X, y_big)

    # Feature importance
    if hasattr(final, "feature_importances_"):
        importances = sorted(zip(X.columns, final.feature_importances_), key=lambda t: -t[1])
        print("\n[importance] top 15:")
        for n, imp in importances[:15]:
            print(f"  {n:<35} {imp:.4f}")

    out = OUT_DIR / "model_v2.joblib"
    joblib.dump({
        "model": final,
        "model_kind": best_config[0],
        "target": "big_loss",
        "veto_threshold": float(threshold),
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "categorical_maps": cat_maps,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(df)),
        "training_date_range": [df["day"].min(), df["day"].max()],
        "rows_2025": int((df["source_tag"] == "2025").sum()),
        "rows_2026": int((df["source_tag"] == "2026").sum()),
        "cv_auc_mean": float(np.mean(cv_aucs)),
        "cv_auc_std": float(np.std(cv_aucs)),
        "temporal_tail_auc": float(auc_holdout),
        "temporal_tail_delta": float(best_row["delta"]),
        "notes": (
            "v2: trained on 2025 + April 2026 trades using REAL OHLCV bars "
            "from es_master_outrights parquet (wicks/body/volume features "
            "now meaningful). Feature set expanded from 20 to 25 columns. "
            "No pure OOS set — April 2026 is partly in training. Temporal "
            "tail holdout (last 15% by date) used for model selection."
        ),
    }, out)
    print(f"\n[write] {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
