"""Train a bracket-aware classifier that takes (chart features + tp_pts + sl_pts)
as inputs and predicts P(SL hit before TP).

This solves G v1's problem: G v1 was trained only on DE3 trades (25/10 brackets)
and mis-calibrated on AetherFlow's tight 3/5 brackets. v3 takes the actual TP/SL
distances as input, so it generalizes across strategies.

Output: artifacts/signal_gate_2025/model_v3.joblib
"""
from __future__ import annotations

import joblib, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/Users/wes/Downloads/JULIE001")
OUT_DIR = ROOT / "artifacts" / "signal_gate_2025"

# Same numeric features as v2 (real OHLCV-derived) + bracket-geometry features
NUMERIC_FEATURES = [
    "de3_entry_ret1_atr",
    "de3_entry_body_pos1",
    "de3_entry_body1_ratio",
    "de3_entry_close_pos1",
    "de3_entry_lower_wick_ratio",
    "de3_entry_upper_wick_ratio",
    "de3_entry_down3",
    "de3_entry_flips5",
    "de3_entry_range10_atr",
    "de3_entry_dist_low5_atr",
    "de3_entry_dist_high5_atr",
    "de3_entry_vol1_rel20",
    "de3_entry_atr14",
    # NEW: bracket geometry — the key fix
    "tp_pts",
    "sl_pts",
    "rr_ratio",  # tp/sl
]
CATEGORICAL_FEATURES = ["side"]
ORDINAL_FEATURES = ["et_hour"]


def assemble_X(df: pd.DataFrame, categorical_maps: dict | None = None):
    updated = dict(categorical_maps or {})
    parts = [df[NUMERIC_FEATURES].copy()]
    for col in CATEGORICAL_FEATURES:
        known = updated.get(col)
        if known is None:
            known = sorted(df[col].dropna().unique().tolist())
            updated[col] = known
        encoded = pd.DataFrame({f"{col}__{v}": (df[col] == v).astype(int) for v in known}, index=df.index)
        parts.append(encoded)
    parts.append(df[ORDINAL_FEATURES].astype(float))
    X = pd.concat(parts, axis=1).fillna(0.0)
    return X, updated


def main():
    df = pd.read_parquet(OUT_DIR / "bracket_training.parquet")
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES + ["is_loss"]
    df = df.dropna(subset=[c for c in required if c in df.columns]).reset_index(drop=True)
    print(f"[load] {len(df):,} simulated trades")
    print(f"  loss rate: {df['is_loss'].mean():.1%}")
    print(f"  per-bracket loss rate:")
    for b, sub in df.groupby("bracket_name"):
        print(f"    {b:<20} n={len(sub):>5}  loss_rate={sub['is_loss'].mean():.1%}")

    X, cat_maps = assemble_X(df)
    y = df["is_loss"].astype(int).values
    print(f"\n[features] {X.shape[1]} columns")

    # 5-fold CV
    print("\n[cv] 5-fold stratified")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            min_samples_leaf=50, random_state=42,
        )
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    print(f"  CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # Per-bracket AUC on a held-out tail
    print("\n[per-bracket] held-out tail AUC (per-bracket calibration)")
    df_sorted = df.sort_values("ts").reset_index(drop=True)
    X_sorted, _ = assemble_X(df_sorted, categorical_maps=cat_maps)
    y_sorted = df_sorted["is_loss"].astype(int).values
    split = int(0.85 * len(df_sorted))
    X_tr, X_te = X_sorted.iloc[:split], X_sorted.iloc[split:]
    y_tr, y_te = y_sorted[:split], y_sorted[split:]
    df_te = df_sorted.iloc[split:]
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    print(f"  pooled AUC: {roc_auc_score(y_te, p_te):.3f}")
    df_te = df_te.assign(p_loss=p_te)
    print(f"  {'bracket':<22} {'n':>5}  {'actual_loss':>12}  {'pred_loss(avg)':>15}  {'AUC':>6}")
    for b, sub in df_te.groupby("bracket_name"):
        if len(sub) < 50: continue
        if sub["is_loss"].nunique() < 2: continue
        auc = roc_auc_score(sub["is_loss"], sub["p_loss"])
        print(f"  {b:<22} {len(sub):>5}  {sub['is_loss'].mean():>12.1%}  "
              f"{sub['p_loss'].mean():>14.1%}   {auc:>6.3f}")

    # Train final on full data
    print("\n[train] full dataset for ship")
    final = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )
    final.fit(X, y)

    # Feature importance
    importances = sorted(zip(X.columns, final.feature_importances_), key=lambda t: -t[1])
    print("\n[importance] top 12:")
    for name, imp in importances[:12]:
        print(f"  {name:<32} {imp:.4f}")

    # Pick reasonable veto threshold per bracket via cross-bracket calibration
    # Strategy: for each bracket type, find threshold that maximizes EV
    # EV(threshold) = sum over all trades >= threshold of (-sl_pts*5 if loss else +tp_pts*5)
    print("\n[thresholds] per-bracket EV-maximizing thresholds (on held-out tail)")
    df_te = df_te.copy()
    df_te["pnl"] = np.where(df_te["is_loss"] == 1, -df_te["sl_pts"] * 5, df_te["tp_pts"] * 5)
    bracket_thresholds = {}
    for b, sub in df_te.groupby("bracket_name"):
        if len(sub) < 50: continue
        best_thr, best_ev = 0.5, -1e9
        for thr in np.arange(0.20, 0.90, 0.025):
            kept = sub[sub["p_loss"] < thr]
            ev = kept["pnl"].sum() if len(kept) else 0
            if ev > best_ev:
                best_ev = ev; best_thr = float(thr)
        bracket_thresholds[b] = best_thr
        baseline = sub["pnl"].sum()
        kept_at_best = sub[sub["p_loss"] < best_thr]
        print(f"  {b:<22} EV-best thresh={best_thr:.3f}  "
              f"baseline ${baseline:>+8.2f}  with-veto ${kept_at_best['pnl'].sum():>+8.2f}  "
              f"vetoed {len(sub) - len(kept_at_best)}/{len(sub)}")

    out = OUT_DIR / "model_v3.joblib"
    joblib.dump({
        "model": final,
        "model_kind": "GBT_d5_bracket_aware",
        "target": "is_loss",  # P(SL hits before TP)
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "categorical_maps": cat_maps,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(df)),
        "training_date_range": [df["ts"].min().isoformat(), df["ts"].max().isoformat()],
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "bracket_thresholds": bracket_thresholds,
        "default_threshold": 0.50,
        "notes": (
            "v3: bracket-aware model — takes tp_pts and sl_pts as input features. "
            "Trained on 55,696 simulated outcomes across 8 (side, tp, sl) configs "
            "covering DE3 (25/10, 12.5/10), AetherFlow (3/5), and RegimeAdaptive (8/5) "
            "geometry. Replaces v1 which was DE3-only and mis-calibrated for AF setups. "
            "Per-bracket EV-best thresholds saved in bracket_thresholds dict."
        ),
    }, out)
    print(f"\n[write] {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
