#!/usr/bin/env python3
"""Train a context-aware decision-tree classifier predicting oracle variant.

Uses features from daily_features.json:
  - intraday at 10:30: range, drift, eff
  - prior 5 sessions: avg abs_drift, range, eff, breakout/chop fractions
  - WTD, MTD aggregates
  - prior month aggregates + dominant category (one-hot encoded)

Training:
  - 5-fold cross-validation on 2025 136 days
  - Report per-class precision/recall
  - Also report P&L-weighted score: if classifier predicts variant V, how much
    $ do we capture vs always-V1 baseline? Vs always-V3? Vs oracle?

Saves trained tree + decision rules for later use in sim.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

ROOT = Path("/Users/wes/Downloads/JULIE001")
FEATURES = ROOT / "backtest_reports" / "daily_features.json"

FEATURE_NAMES = [
    "intraday_range", "intraday_drift", "intraday_eff",
    "prior5_abs_drift", "prior5_range", "prior5_eff",
    "prior5_breakout_frac", "prior5_chop_frac",
    "wtd_abs_drift", "wtd_range",
    "mtd_abs_drift", "mtd_range", "mtd_breakout_frac", "mtd_chop_frac",
    "pm_abs_drift", "pm_range", "pm_breakout_frac", "pm_chop_frac",
    # one-hot for pm_dominant
    "pm_dom_breakout", "pm_dom_chop", "pm_dom_flat_calm",
    "pm_dom_large_trend", "pm_dom_moderate",
]


def to_feature_vector(row: dict) -> list[float]:
    v = []
    for name in FEATURE_NAMES:
        if name.startswith("pm_dom_"):
            target = name.replace("pm_dom_", "")
            v.append(1.0 if row.get("pm_dominant") == target else 0.0)
        else:
            v.append(float(row.get(name, 0.0)))
    # use abs of intraday_drift because sign shouldn't matter for classification
    # (idx of intraday_drift in FEATURE_NAMES is 1)
    v[1] = abs(v[1])
    return v


def load_data():
    data = json.loads(FEATURES.read_text(encoding="utf-8"))
    rows = data["labeled_rows"]
    X = np.array([to_feature_vector(r) for r in rows])
    y = np.array([r["oracle_variant"] for r in rows])
    return X, y, rows


def pnl_captured(rows_subset, predictions) -> dict:
    """Return total P&L under classifier predictions vs V1, V3, Oracle."""
    total_pred = 0.0
    total_v1 = 0.0
    total_v3 = 0.0
    total_oracle = 0.0
    dd_pred = 0; dd_v1 = 0; dd_v3 = 0
    for r, pred in zip(rows_subset, predictions):
        pnls = r["variant_pnls"]
        total_pred += pnls[pred]
        total_v1 += pnls["V1"]
        total_v3 += pnls["V3"]
        total_oracle += pnls[r["oracle_variant"]]
    return {
        "pred": round(total_pred, 2),
        "v1": round(total_v1, 2),
        "v3": round(total_v3, 2),
        "oracle": round(total_oracle, 2),
    }


def cross_validate(X, y, rows, n_splits=5, max_depth=4, min_samples_leaf=6):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    all_preds = np.empty(len(y), dtype=object)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            # no class_weight — unbalanced is fine because we score by P&L
        )
        clf.fit(X[tr], y[tr])
        preds = clf.predict(X[te])
        all_preds[te] = preds
        test_rows = [rows[i] for i in te]
        metrics = pnl_captured(test_rows, preds)
        fold_metrics.append({"fold": fold, **metrics})

    # Aggregate
    totals = pnl_captured(rows, all_preds)
    print(f"CV pooled results (max_depth={max_depth}, min_samples_leaf={min_samples_leaf}):")
    print(f"  Classifier   ${totals['pred']:+.2f}")
    print(f"  Always V1    ${totals['v1']:+.2f}")
    print(f"  Always V3    ${totals['v3']:+.2f}")
    print(f"  Oracle       ${totals['oracle']:+.2f}")
    print(f"  vs V1:       ${totals['pred'] - totals['v1']:+.2f}")
    print(f"  vs V3:       ${totals['pred'] - totals['v3']:+.2f}")
    print(f"  vs Oracle:   ${totals['pred'] - totals['oracle']:+.2f}")

    # Classification accuracy
    print(f"\n  Raw variant accuracy: {(all_preds == y).sum() / len(y):.1%}")
    print(f"\nClassification report (CV, pooled test sets):")
    print(classification_report(y, all_preds, zero_division=0))
    return totals, all_preds


def train_final(X, y, max_depth=4, min_samples_leaf=6):
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    clf.fit(X, y)
    return clf


def train_final_rf(X, y, n_estimators=50, max_depth=5, min_samples_leaf=4):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def cv_rf(X, y, rows, n_estimators=50, max_depth=5, min_samples_leaf=4, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X[tr], y[tr])
        all_preds[te] = clf.predict(X[te])
    totals = pnl_captured(rows, all_preds)
    return totals, all_preds


def grid_search(X, y, rows):
    """Try several depth/leaf combinations."""
    best_score = -1e9
    best_params = None
    best_totals = None
    print("Grid-search over tree hyperparameters (by CV P&L):")
    print(f"{'max_depth':>10}{'min_leaf':>10}  {'classifier':>12}{'vs V1':>10}{'accuracy':>10}")
    for max_depth in [3, 4, 5, 6, 8]:
        for min_leaf in [3, 5, 8, 12]:
            totals, preds = cross_validate_silent(X, y, rows, max_depth=max_depth, min_samples_leaf=min_leaf)
            acc = (preds == y).sum() / len(y)
            delta = totals["pred"] - totals["v1"]
            print(f"{max_depth:>10}{min_leaf:>10}  ${totals['pred']:>+10.2f}  ${delta:>+8.2f}  {acc:>9.1%}")
            if totals["pred"] > best_score:
                best_score = totals["pred"]
                best_params = (max_depth, min_leaf)
                best_totals = totals
    return best_params, best_totals


def cross_validate_silent(X, y, rows, n_splits=5, max_depth=4, min_samples_leaf=6):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_preds = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X, y):
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            # no class_weight — unbalanced is fine because we score by P&L
        )
        clf.fit(X[tr], y[tr])
        all_preds[te] = clf.predict(X[te])
    totals = pnl_captured(rows, all_preds)
    return totals, all_preds


if __name__ == "__main__":
    X, y, rows = load_data()
    print(f"[load] {len(rows)} labeled rows, {X.shape[1]} features")
    print(f"  Oracle variant distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    print("\n" + "=" * 70)
    print("GRID SEARCH")
    print("=" * 70)
    (best_depth, best_leaf), best_totals = grid_search(X, y, rows)
    print(f"\nBest params: max_depth={best_depth}, min_samples_leaf={best_leaf}")

    print("\n" + "=" * 70)
    print("DETAILED BEST-MODEL RESULTS")
    print("=" * 70)
    totals, preds = cross_validate(X, y, rows, max_depth=best_depth, min_samples_leaf=best_leaf)

    # Train final model on all data
    final = train_final(X, y, max_depth=best_depth, min_samples_leaf=best_leaf)
    print(f"\nFinal model tree structure:")
    print(export_text(final, feature_names=FEATURE_NAMES, max_depth=best_depth))

    # Also try a Random Forest
    print("\n" + "=" * 70)
    print("RANDOM FOREST (grid over n_estimators x max_depth)")
    print("=" * 70)
    best_rf_score = -1e9
    best_rf_params = None
    for n_est in [30, 50, 80, 120]:
        for md in [3, 4, 5, 6, 8]:
            for mls in [2, 4, 8]:
                totals, preds = cv_rf(X, y, rows, n_estimators=n_est, max_depth=md, min_samples_leaf=mls)
                acc = (preds == y).sum() / len(y)
                if totals["pred"] > best_rf_score:
                    best_rf_score = totals["pred"]
                    best_rf_params = (n_est, md, mls)
                print(f"  RF(n={n_est}, depth={md}, leaf={mls})  ${totals['pred']:+.2f}  "
                      f"vs V1=${totals['pred']-totals['v1']:+.2f}  acc={acc:.1%}")
    print(f"\nBest RF: n_estimators={best_rf_params[0]}, max_depth={best_rf_params[1]}, min_leaf={best_rf_params[2]}")
    print(f"  CV total: ${best_rf_score:+.2f}")

    # Save the better of (tree, RF)
    use_rf = best_rf_score > totals["pred"]
    if use_rf:
        final = train_final_rf(X, y, *best_rf_params)
        final_kind = "random_forest"
    else:
        final_kind = "decision_tree"
    print(f"\nChose: {final_kind}")

    # Save
    out = ROOT / "backtest_reports" / "context_classifier.pkl"
    with out.open("wb") as fh:
        pickle.dump({
            "model": final,
            "model_kind": final_kind,
            "feature_names": FEATURE_NAMES,
            "best_depth": best_depth,
            "best_leaf": best_leaf,
            "cv_totals": totals,
        }, fh)
    print(f"\n[write] {out}")
