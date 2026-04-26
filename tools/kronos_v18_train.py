"""V18 training — V15-architecture meta-learner + Kronos features.

Builds two models:
  M1: LogisticRegression on (6 V15 probas + Kronos features)
  M2: HistGradientBoosting on (6 V15 probas + Kronos features)

Compares to V15 baseline (LogReg on 6 probas, threshold 0.725).

Walk-forward CV (5 folds, time-ordered) + final holdout (last 20%).
Threshold sweep + Pareto compare.
"""
from __future__ import annotations

import os
import sys
import json
import math
import joblib
import argparse
import warnings

import numpy as np
import pandas as pd
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = "/Users/wes/Downloads/JULIE001"
CORPUS_PATH = os.path.join(ROOT, "artifacts", "v12_training_corpus.parquet")
KRONOS_PATH = os.path.join(ROOT, "artifacts", "v18_kronos_features.parquet")
K12_BUNDLE_PATH = os.path.join(ROOT, "artifacts", "regime_ml_kalshi_v12", "de3", "model.joblib")
V15_BUNDLE_PATH = os.path.join(ROOT, "artifacts", "regime_ml_meta_v15", "de3", "model.joblib")
OUT_DIR = os.path.join(ROOT, "artifacts", "regime_ml_meta_v18", "de3")
REPORT_PATH = "/tmp/kronos_v18_final_report.md"

V15_FEATS = ["fg_proba", "kalshi_proba", "kalshi_v12_proba", "lfo_proba", "pct_proba", "pivot_proba"]
KRONOS_FEATS = [
    "kronos_pred_atr_30bar",
    "kronos_dir_move",
    "kronos_max_high_above",
    "kronos_min_low_below",
    "kronos_close_vs_entry",
]


def compute_k12_probas(corpus, k12_bundle):
    feats = k12_bundle["features"]
    model = k12_bundle["model"]
    X = corpus[feats].copy()
    # Replace NaN with 0 like the live code
    X = X.fillna(0.0)
    p = model.predict_proba(X)[:, 1]
    return p


def metrics_at_threshold(y_true, y_proba, pnl, thr):
    keep = y_proba >= thr
    if keep.sum() == 0:
        return {"thr": thr, "n_kept": 0, "win_rate": 0.0, "net_pnl": 0.0, "max_dd": 0.0}
    w = (y_true[keep] == 1).mean()
    pnls = pnl[keep]
    cum = pnls.cumsum()
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() if len(cum) else 0.0
    return {
        "thr": float(thr),
        "n_kept": int(keep.sum()),
        "win_rate": float(w),
        "net_pnl": float(pnls.sum()),
        "max_dd": float(dd),
    }


def walk_forward_auc(X, y, model_factory, n_folds=5):
    """Time-ordered 5-fold walk-forward CV. Predicts each fold using all prior data."""
    aucs = []
    n = len(X)
    fold_size = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        train_end = fold_size * k
        val_end = fold_size * (k + 1)
        Xtr, ytr = X[:train_end], y[:train_end]
        Xva, yva = X[train_end:val_end], y[train_end:val_end]
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            continue
        m = model_factory()
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        aucs.append(roc_auc_score(yva, p))
    return aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load corpus
    corpus = pd.read_parquet(CORPUS_PATH)
    de3_fr = corpus[(corpus["family"] == "de3") & (corpus["allowed_by_friend_rule"] == True)].copy()
    de3_fr = de3_fr.sort_values("ts").reset_index(drop=True)
    de3_fr["row_idx_orig"] = de3_fr.index

    # Load K12 bundle and compute kalshi_v12_proba
    k12 = joblib.load(K12_BUNDLE_PATH)
    de3_fr["kalshi_v12_proba"] = compute_k12_probas(de3_fr, k12)

    # is_winner
    de3_fr["is_winner"] = (de3_fr["net_pnl_after_haircut"] > 0).astype(int)

    # Load Kronos features
    kr = pd.read_parquet(KRONOS_PATH)
    print(f"corpus: {len(de3_fr)} rows; kronos: {len(kr)} rows")

    # We must merge on something stable. Kronos extractor used reset_index of (family=de3, allowed_by_friend_rule==True).
    # Re-derive that index here for matching:
    de3_match = corpus[(corpus["family"] == "de3") & (corpus["allowed_by_friend_rule"] == True)].copy().reset_index(drop=True)
    de3_match["row_idx"] = de3_match.index
    # Merge kronos onto de3_match by row_idx
    de3_match = de3_match.merge(kr[["row_idx"] + KRONOS_FEATS], on="row_idx", how="left")
    # Now sort by ts
    de3_match["kalshi_v12_proba"] = compute_k12_probas(de3_match, k12)
    de3_match["is_winner"] = (de3_match["net_pnl_after_haircut"] > 0).astype(int)
    de3_match = de3_match.sort_values("ts").reset_index(drop=True)

    # Drop rows missing any required feature
    base_feats = V15_FEATS
    full_feats = V15_FEATS + KRONOS_FEATS
    has_kronos = de3_match[KRONOS_FEATS].notna().all(axis=1)
    print(f"  rows with kronos features: {has_kronos.sum()} / {len(de3_match)}")

    base_data = de3_match.dropna(subset=base_feats + ["is_winner", "net_pnl_after_haircut"]).copy()
    full_data = de3_match[has_kronos].dropna(subset=full_feats + ["is_winner", "net_pnl_after_haircut"]).copy()

    print(f"  base_data (V15 only): {len(base_data)}")
    print(f"  full_data (V18 = V15 + Kronos): {len(full_data)}")

    # Use the SAME row-set for fair comparison: only rows where we have kronos features
    # i.e. base_data == full_data restricted to rows with kronos features.
    common = full_data.copy()  # this has all features
    common = common.sort_values("ts").reset_index(drop=True)

    # Holdout split (time-ordered)
    n = len(common)
    holdout_n = int(round(n * args.holdout_frac))
    train = common.iloc[: n - holdout_n].copy()
    holdout = common.iloc[n - holdout_n :].copy()
    print(f"  train: {len(train)}  holdout: {len(holdout)}")
    print(f"  train ts range: {train['ts'].min()} .. {train['ts'].max()}")
    print(f"  holdout ts range: {holdout['ts'].min()} .. {holdout['ts'].max()}")

    # ------- Baseline V15: LogReg on 6 probas (refit on this same train-set) -------
    Xb_tr = train[base_feats].values
    yb_tr = train["is_winner"].values
    Xb_ho = holdout[base_feats].values
    yb_ho = holdout["is_winner"].values
    pnl_ho = holdout["net_pnl_after_haircut"].values

    m_v15 = LogisticRegression(max_iter=1000, random_state=42).fit(Xb_tr, yb_tr)
    p_v15 = m_v15.predict_proba(Xb_ho)[:, 1]
    auc_v15 = roc_auc_score(yb_ho, p_v15) if len(np.unique(yb_ho)) > 1 else float("nan")

    wf_auc_v15 = walk_forward_auc(Xb_tr, yb_tr, lambda: LogisticRegression(max_iter=1000, random_state=42))

    # ------- V18-LR: LogReg on 6 probas + 5 Kronos -------
    Xf_tr = train[full_feats].values
    Xf_ho = holdout[full_feats].values

    m_v18_lr = LogisticRegression(max_iter=1000, random_state=42).fit(Xf_tr, yb_tr)
    p_v18_lr = m_v18_lr.predict_proba(Xf_ho)[:, 1]
    auc_v18_lr = roc_auc_score(yb_ho, p_v18_lr) if len(np.unique(yb_ho)) > 1 else float("nan")
    wf_auc_v18_lr = walk_forward_auc(Xf_tr, yb_tr, lambda: LogisticRegression(max_iter=1000, random_state=42))

    # ------- V18-HGB: HistGradientBoosting on full -------
    m_v18_hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42).fit(Xf_tr, yb_tr)
    p_v18_hgb = m_v18_hgb.predict_proba(Xf_ho)[:, 1]
    auc_v18_hgb = roc_auc_score(yb_ho, p_v18_hgb) if len(np.unique(yb_ho)) > 1 else float("nan")
    wf_auc_v18_hgb = walk_forward_auc(Xf_tr, yb_tr, lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4, random_state=42))

    # ------- Threshold sweep on holdout -------
    thresholds = np.arange(0.30, 0.96, 0.05)

    def sweep(p, y_true, pnl):
        return [metrics_at_threshold(y_true, p, pnl, t) for t in thresholds]

    sweep_v15 = sweep(p_v15, yb_ho, pnl_ho)
    sweep_v18_lr = sweep(p_v18_lr, yb_ho, pnl_ho)
    sweep_v18_hgb = sweep(p_v18_hgb, yb_ho, pnl_ho)

    # V15 @ 0.65 from instructions: 91 holdout / 59.34% / +$1,500 / $340 DD
    v15_at_065 = metrics_at_threshold(yb_ho, p_v15, pnl_ho, 0.65)

    # Find best Pareto V18 vs V15@0.65
    def is_pareto_better(cand, base):
        better_pnl = cand["net_pnl"] > base["net_pnl"]
        not_worse_dd = cand["max_dd"] <= base["max_dd"] * 1.10  # 10% tolerance
        not_worse_wr = cand["win_rate"] >= base["win_rate"] - 0.02
        return better_pnl and not_worse_dd and not_worse_wr

    pareto_lr = [s for s in sweep_v18_lr if is_pareto_better(s, v15_at_065) and s["n_kept"] >= 20]
    pareto_hgb = [s for s in sweep_v18_hgb if is_pareto_better(s, v15_at_065) and s["n_kept"] >= 20]
    best_lr = max(pareto_lr, key=lambda s: s["net_pnl"], default=None)
    best_hgb = max(pareto_hgb, key=lambda s: s["net_pnl"], default=None)

    # Save best V18 model (LR) if any pareto config exists
    if best_lr or best_hgb:
        bundle = {
            "model_lr": m_v18_lr,
            "model_hgb": m_v18_hgb,
            "features": full_feats,
            "best_threshold_lr": best_lr["thr"] if best_lr else None,
            "best_threshold_hgb": best_hgb["thr"] if best_hgb else None,
            "label": "pnl > 0 (winner)",
            "description": "V18 — V15 stack augmented with 5 Kronos forecast features.",
        }
        joblib.dump(bundle, os.path.join(OUT_DIR, "model.joblib"))
        print(f"Saved V18 bundle to {OUT_DIR}/model.joblib")

    # Build report
    lines = []
    lines.append("# Kronos V18 Final Report\n")
    lines.append(f"- corpus rows: {len(de3_match)}")
    lines.append(f"- with kronos features: {has_kronos.sum()}")
    lines.append(f"- train: {len(train)}  holdout: {len(holdout)} (last {args.holdout_frac:.0%})")
    lines.append("")
    lines.append("## Holdout AUC")
    lines.append(f"- V15 (LR, 6 probas): {auc_v15:.4f}")
    lines.append(f"- V18-LR (6 probas + 5 kronos): {auc_v18_lr:.4f}")
    lines.append(f"- V18-HGB (6 probas + 5 kronos): {auc_v18_hgb:.4f}")
    lines.append(f"- V18 lift over V15 (LR): {auc_v18_lr - auc_v15:+.4f}")
    lines.append(f"- V18 lift over V15 (HGB): {auc_v18_hgb - auc_v15:+.4f}")
    lines.append("")
    lines.append("## Walk-forward CV mean AUC (5 folds)")
    if wf_auc_v15:
        lines.append(f"- V15 mean: {np.mean(wf_auc_v15):.4f}  folds={[f'{x:.3f}' for x in wf_auc_v15]}")
    if wf_auc_v18_lr:
        lines.append(f"- V18-LR mean: {np.mean(wf_auc_v18_lr):.4f}  folds={[f'{x:.3f}' for x in wf_auc_v18_lr]}")
    if wf_auc_v18_hgb:
        lines.append(f"- V18-HGB mean: {np.mean(wf_auc_v18_hgb):.4f}  folds={[f'{x:.3f}' for x in wf_auc_v18_hgb]}")
    lines.append("")
    lines.append("## V15 @ 0.65 (refit) on holdout")
    for k, v in v15_at_065.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## V18-LR threshold sweep")
    lines.append("```")
    lines.append(pd.DataFrame(sweep_v18_lr).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## V18-HGB threshold sweep")
    lines.append("```")
    lines.append(pd.DataFrame(sweep_v18_hgb).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Best Pareto vs V15 @ 0.65")
    lines.append(f"- best V18-LR: {best_lr}")
    lines.append(f"- best V18-HGB: {best_hgb}")
    lines.append("")
    lines.append("## Verdict")
    auc_lift_max = max(auc_v18_lr - auc_v15, auc_v18_hgb - auc_v15)
    if (best_lr or best_hgb) and auc_lift_max > 0.01:
        verdict = "A — INTEGRATE: V18 Pareto-improves on V15."
    elif auc_lift_max > 0.005:
        verdict = "C — MARGINAL: small AUC lift, no clear Pareto win."
    else:
        verdict = "B — SKIP: Kronos adds no meaningful orthogonal signal."
    lines.append(verdict)

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote report: {REPORT_PATH}")
    print("\n" + "=" * 60)
    print(f"V15 holdout AUC: {auc_v15:.4f}  V18-LR: {auc_v18_lr:.4f}  V18-HGB: {auc_v18_hgb:.4f}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
