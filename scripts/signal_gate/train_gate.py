#!/usr/bin/env python3
"""Train a binary classifier to predict trade-level win probability, using
close-based chart features + regime + session as inputs.

Output: artifacts/signal_gate_2025/model.joblib
  {
    "model": GradientBoostingClassifier,
    "feature_names": [...],
    "categorical_maps": {"side": {...}, "regime": {...}, "session": {...}},
    "threshold_search": [{"thresh":0.40, "vetoed":..., "delta":...}, ...],
    "best_threshold": 0.45,
    "training_date": ...,
    "notes": "curve-fit to 2025-YTD for 2025-2028 regime bandaid",
  }
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path("/Users/wes/Downloads/JULIE001")
OUT_DIR = ROOT / "artifacts" / "signal_gate_2025"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Close-based features (immune to synthetic-bar body/wick degeneracy).
NUMERIC_FEATURES = [
    "de3_entry_ret1_atr",
    "de3_entry_down3",
    "de3_entry_flips5",
    "de3_entry_range10_atr",
    "de3_entry_dist_low5_atr",
    "de3_entry_dist_high5_atr",
    "de3_entry_velocity_30",
    "de3_entry_dist_low30_atr",
    "de3_entry_dist_high30_atr",
    "de3_entry_ret30_atr",
]
CATEGORICAL_FEATURES = ["side", "regime", "session"]
ORDINAL_FEATURES = ["et_hour"]


def _encode_categorical(series: pd.Series, known_values: list[str] | None = None):
    """One-hot encode a categorical column. Returns (df, value_order_used)."""
    if known_values is None:
        known_values = sorted(series.dropna().unique().tolist())
    out = {}
    for val in known_values:
        out[f"{series.name}__{val}"] = (series == val).astype(int)
    return pd.DataFrame(out, index=series.index), known_values


def assemble_X(df: pd.DataFrame, categorical_maps: dict | None = None):
    """Build feature matrix. Returns (X, updated_maps)."""
    updated = dict(categorical_maps or {})
    parts = [df[NUMERIC_FEATURES].copy()]
    for col in CATEGORICAL_FEATURES:
        known = updated.get(col)
        encoded, known = _encode_categorical(df[col], known_values=known)
        updated[col] = known
        parts.append(encoded)
    parts.append(df[ORDINAL_FEATURES].astype(float))
    X = pd.concat(parts, axis=1)
    X = X.fillna(0.0)
    return X, updated


def evaluate_thresholds(df: pd.DataFrame, y_proba, thresholds: list[float]) -> list[dict]:
    """For each threshold: veto if P(win) < threshold. Report:
    net P&L delta vs no-veto baseline, vetoed count, vetoed W/L, DD viol.
    """
    base_pnl = df["pnl_dollars"].sum()
    results = []
    for t in thresholds:
        vetoed = y_proba < t
        kept_pnl = df.loc[~vetoed, "pnl_dollars"].sum()
        vetoed_df = df.loc[vetoed]
        vetoed_wins = (vetoed_df["pnl_dollars"] > 0).sum()
        vetoed_losses = (vetoed_df["pnl_dollars"] < 0).sum()
        results.append({
            "threshold": round(float(t), 3),
            "vetoed": int(vetoed.sum()),
            "vetoed_wins": int(vetoed_wins),
            "vetoed_losses": int(vetoed_losses),
            "vetoed_pnl_removed": round(float(vetoed_df["pnl_dollars"].sum()), 2),
            "baseline_pnl": round(float(base_pnl), 2),
            "kept_pnl": round(float(kept_pnl), 2),
            "delta": round(float(kept_pnl - base_pnl), 2),
        })
    return results


def main():
    df_path = OUT_DIR / "training_rows.parquet"
    df = pd.read_parquet(df_path)
    print(f"[load] {len(df)} training rows")

    # Drop rows with missing key features
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES + ["win"]
    df = df.dropna(subset=[c for c in required if c in df.columns]).reset_index(drop=True)
    print(f"[clean] {len(df)} rows after dropping NaN feature rows")
    print(f"  win rate: {df['win'].mean():.1%}")
    print(f"  baseline P&L: ${df['pnl_dollars'].sum():+.2f}")

    X, cat_maps = assemble_X(df)
    y = df["win"].astype(int).values
    feature_names = list(X.columns)
    print(f"[features] {len(feature_names)} columns: {feature_names}")

    # Temporal split: train on all rows sorted by date, hold out last 20%
    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    X_sorted, _ = assemble_X(df_sorted, categorical_maps=cat_maps)
    y_sorted = df_sorted["win"].astype(int).values
    split_at = int(len(df_sorted) * 0.8)
    X_tr, X_te = X_sorted.iloc[:split_at], X_sorted.iloc[split_at:]
    y_tr, y_te = y_sorted[:split_at], y_sorted[split_at:]
    df_te = df_sorted.iloc[split_at:]

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)

    # Evaluate on held-out tail
    p_te = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_te)
    print(f"\n[eval] held-out tail (n={len(y_te)}) AUC: {auc:.3f}")

    print("\n[eval] Threshold sweep on held-out tail (veto if P(win) < thresh):")
    sweep = evaluate_thresholds(df_te, p_te, [0.30, 0.35, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52])
    print(f"{'thresh':>8}{'vetoed':>8}{'W/L':>10}{'vetoed_$':>11}{'base_$':>11}{'kept_$':>11}{'delta':>10}")
    for row in sweep:
        wl = f"{row['vetoed_wins']}/{row['vetoed_losses']}"
        print(f"{row['threshold']:>8.2f}{row['vetoed']:>8}{wl:>10}"
              f"{row['vetoed_pnl_removed']:>+11.2f}{row['baseline_pnl']:>+11.2f}"
              f"{row['kept_pnl']:>+11.2f}{row['delta']:>+10.2f}")

    # Pick the threshold that maximizes held-out delta, require veto <40% of trades
    good = [r for r in sweep if r["vetoed"] / max(1, len(df_te)) < 0.40]
    best = max(good, key=lambda r: r["delta"]) if good else max(sweep, key=lambda r: r["delta"])
    print(f"\n[best] threshold={best['threshold']}  delta=${best['delta']:+.2f}")

    # Retrain on FULL 2025 dataset for the saved artifact
    clf_full = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    clf_full.fit(X, y)

    # Feature importance
    importances = sorted(zip(feature_names, clf_full.feature_importances_), key=lambda t: -t[1])
    print("\n[importance] top 10 features:")
    for name, imp in importances[:10]:
        print(f"  {name:<35} {imp:.4f}")

    # Save
    out = OUT_DIR / "model.joblib"
    joblib.dump({
        "model": clf_full,
        "feature_names": feature_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "categorical_maps": cat_maps,
        "threshold_sweep": sweep,
        "best_threshold": best["threshold"],
        "held_out_auc": float(auc),
        "training_date": datetime.utcnow().isoformat() + "Z",
        "training_rows": int(len(df)),
        "training_date_range": [df["day"].min(), df["day"].max()],
        "notes": "2025 signal-gate curve-fit — bandaid for tariff-regime; close-only features (synthetic OHLC from replay logs); revert with JULIE_SIGNAL_GATE_2025=0",
    }, out)
    print(f"\n[write] {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
