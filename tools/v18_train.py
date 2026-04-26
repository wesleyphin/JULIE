"""V18 trainer — Kronos-augmented meta-learners for DE3, RA, AF.

Targets:
  --target de3       train V18-DE3  (V15 6 base probas + 5 Kronos features)
  --target ra        train V18-RA   (V17 3 RA features + 5 Kronos features)
  --target af        train V18-AF   (regime/sub_strategy/side OHE + hour + Kronos)
  --target final_report  produce unified comparison vs baselines

Each target is independent and re-runnable.

Outputs:
  artifacts/regime_ml_v18_de3/de3/model.joblib
  artifacts/regime_ml_v18_ra/ra/model.joblib
  artifacts/regime_ml_v18_af/af/model.joblib
  artifacts/regime_ml_v18_report.json   (final unified report)
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


ROOT = "/Users/wes/Downloads/JULIE001"
CORPUS_PATH = os.path.join(ROOT, "artifacts", "v12_training_corpus.parquet")
KRONOS_DE3 = os.path.join(ROOT, "artifacts", "v18_kronos_features.parquet")
KRONOS_RA = os.path.join(ROOT, "artifacts", "v18_kronos_features_ra.parquet")
KRONOS_AF = os.path.join(ROOT, "artifacts", "v18_kronos_features_af.parquet")

V15_DE3_BUNDLE = os.path.join(ROOT, "artifacts", "regime_ml_meta_v15", "de3", "model.joblib")
K12_DE3_BUNDLE = os.path.join(ROOT, "artifacts", "regime_ml_kalshi_v12", "de3", "model.joblib")
V17_RA_BUNDLE = os.path.join(ROOT, "artifacts", "regime_ml_ra_ny_rule_v17", "ra", "model.joblib")

OUT_DE3 = os.path.join(ROOT, "artifacts", "regime_ml_v18_de3", "de3", "model.joblib")
OUT_RA = os.path.join(ROOT, "artifacts", "regime_ml_v18_ra", "ra", "model.joblib")
OUT_AF = os.path.join(ROOT, "artifacts", "regime_ml_v18_af", "af", "model.joblib")
OUT_REPORT = os.path.join(ROOT, "artifacts", "regime_ml_v18_report.json")

OUT_DE3_METRICS = os.path.join(ROOT, "artifacts", "regime_ml_v18_de3", "de3", "metrics.json")
OUT_RA_METRICS = os.path.join(ROOT, "artifacts", "regime_ml_v18_ra", "ra", "metrics.json")
OUT_AF_METRICS = os.path.join(ROOT, "artifacts", "regime_ml_v18_af", "af", "metrics.json")
FINAL_REPORT_MD = "/tmp/v18_final_report.md"


def _write_metrics_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  wrote {path}")

KRONOS_FEATS = [
    "kronos_max_high_above",
    "kronos_min_low_below",
    "kronos_pred_atr_30bar",
    "kronos_dir_move",
    "kronos_close_vs_entry",
]

# V15 baseline DE3 metrics from the user spec @ threshold 0.65
V15_BASELINE_DE3 = {
    "threshold": 0.65,
    "n_keep": 91,
    "win_rate": 0.5934,
    "pnl_dollars": 1500.0,
    "max_drawdown_dollars": 340.0,
}
V17_BASELINE_RA_THRESH = 0.40

# Ship gates (V15-style)
SHIP_DD_MAX = 870.0
SHIP_N_MIN = 50
SHIP_WR_MIN = 0.55


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def equity_drawdown(pnl_series):
    if len(pnl_series) == 0:
        return 0.0
    eq = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(dd.max())


def threshold_sweep(y_true, y_proba, pnl, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.20, 0.95, 0.05), 2)
    out = []
    for t in thresholds:
        keep = y_proba >= t
        n = int(keep.sum())
        if n == 0:
            out.append({"threshold": float(t), "n": 0, "wr": 0.0, "pnl": 0.0, "dd": 0.0})
            continue
        kept_pnl = pnl[keep]
        kept_y = np.asarray(y_true)[keep]
        wr = float(kept_y.mean()) if len(kept_y) else 0.0
        out.append({
            "threshold": float(t),
            "n": n,
            "wr": wr,
            "pnl": float(kept_pnl.sum()),
            "dd": equity_drawdown(kept_pnl),
        })
    return out


def cv_fit_predict(model_factory, X, y, n_splits=5):
    """Walk-forward CV. Returns out-of-fold proba aligned to X order, plus
    per-fold AUC."""
    n = len(X)
    proba_oof = np.full(n, np.nan)
    fold_aucs = []
    if n < n_splits + 1:
        return proba_oof, fold_aucs
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, n // 30)))
    for fold, (tr, te) in enumerate(tscv.split(X)):
        try:
            m = model_factory()
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            proba_oof[te] = p
            fold_aucs.append(safe_auc(y[te], p))
        except Exception as e:
            print(f"  ! fold {fold} failed: {e}")
            fold_aucs.append(float("nan"))
    return proba_oof, fold_aucs


# -------------------------------------------------------------------- DE3 ---
def get_v15_base_probas(corpus_df):
    """Re-fire k12 model on corpus to get kalshi_v12_proba; the other 5 base
    proba columns already exist in the corpus. Returns a frame of 6 base probas
    aligned to corpus_df.
    """
    base_cols = ["fg_proba", "kalshi_proba", "lfo_proba", "pct_proba", "pivot_proba"]
    out = corpus_df[base_cols].copy()

    # Re-fire k12
    if not os.path.exists(K12_DE3_BUNDLE):
        print(f"  ! K12 bundle missing: {K12_DE3_BUNDLE}; using kalshi_proba as kalshi_v12_proba surrogate")
        out["kalshi_v12_proba"] = corpus_df["kalshi_proba"].values
    else:
        try:
            k12 = joblib.load(K12_DE3_BUNDLE)
            feats = k12["features"]
            miss = [f for f in feats if f not in corpus_df.columns]
            if miss:
                print(f"  ! K12 missing {len(miss)} features in corpus; surrogate=kalshi_proba")
                out["kalshi_v12_proba"] = corpus_df["kalshi_proba"].values
            else:
                X12 = corpus_df[feats].astype(float).fillna(0.0).values
                out["kalshi_v12_proba"] = k12["model"].predict_proba(X12)[:, 1]
        except Exception as e:
            print(f"  ! K12 re-fire failed: {e}; surrogate=kalshi_proba")
            out["kalshi_v12_proba"] = corpus_df["kalshi_proba"].values

    return out[["fg_proba", "kalshi_proba", "kalshi_v12_proba", "lfo_proba", "pct_proba", "pivot_proba"]]


def train_de3():
    print("[V18-DE3] loading corpus + kronos features")
    if not os.path.exists(KRONOS_DE3):
        print(f"  FATAL: {KRONOS_DE3} missing")
        return {"target": "de3", "error": "kronos features missing"}

    corpus = pd.read_parquet(CORPUS_PATH)
    de3 = corpus[(corpus["family"] == "de3") & (corpus["allowed_by_friend_rule"] == True)].copy()
    de3 = de3.reset_index(drop=True)
    de3["row_idx"] = de3.index
    print(f"  DE3+friend rows: {len(de3)}")

    kronos = pd.read_parquet(KRONOS_DE3)
    print(f"  Kronos rows: {len(kronos)}")

    merged = de3.merge(kronos[["row_idx"] + KRONOS_FEATS], on="row_idx", how="inner")
    merged = merged.sort_values("ts").reset_index(drop=True)
    print(f"  merged rows: {len(merged)}")

    base = get_v15_base_probas(merged)
    feats_all = ["fg_proba", "kalshi_proba", "kalshi_v12_proba", "lfo_proba", "pct_proba", "pivot_proba"] + KRONOS_FEATS
    X = pd.concat([base.reset_index(drop=True), merged[KRONOS_FEATS].reset_index(drop=True)], axis=1)
    X = X.astype(float).fillna(0.0).values
    y = (merged["net_pnl_after_haircut"] > 0).astype(int).values
    pnl = merged["net_pnl_after_haircut"].astype(float).values
    ts = pd.to_datetime(merged["ts"])

    # Train/holdout split: holdout = >= 2026-01-01
    cutoff = pd.Timestamp("2026-01-01", tz=ts.dt.tz) if ts.dt.tz is not None else pd.Timestamp("2026-01-01")
    holdout_mask = ts >= cutoff
    train_mask = ~holdout_mask
    print(f"  train rows: {train_mask.sum()}, holdout rows: {holdout_mask.sum()}")

    results = {"target": "de3", "n_total": int(len(merged)), "features": feats_all}

    # Walk-forward CV on train portion
    X_tr, y_tr, pnl_tr = X[train_mask.values], y[train_mask.values], pnl[train_mask.values]
    X_ho, y_ho, pnl_ho = X[holdout_mask.values], y[holdout_mask.values], pnl[holdout_mask.values]

    def lr_factory():
        return Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=500, C=1.0))])

    def hgb_factory():
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=42)

    print("  CV (LogReg)...")
    oof_lr, aucs_lr = cv_fit_predict(lr_factory, X_tr, y_tr)
    print(f"    fold AUCs: {aucs_lr}  mean={np.nanmean(aucs_lr):.4f}")

    print("  CV (HGB)...")
    oof_hgb, aucs_hgb = cv_fit_predict(hgb_factory, X_tr, y_tr)
    print(f"    fold AUCs: {aucs_hgb}  mean={np.nanmean(aucs_hgb):.4f}")

    # Pick the better of the two by mean CV AUC
    use_hgb = np.nanmean(aucs_hgb) > np.nanmean(aucs_lr)
    chosen = "HGB" if use_hgb else "LogReg"
    factory = hgb_factory if use_hgb else lr_factory
    print(f"  chosen: {chosen}")

    # Fit on full train, predict on holdout
    final = factory()
    final.fit(X_tr, y_tr)
    proba_ho = final.predict_proba(X_ho)[:, 1]
    proba_tr_oof = oof_hgb if use_hgb else oof_lr

    auc_ho = safe_auc(y_ho, proba_ho)
    print(f"  holdout AUC: {auc_ho:.4f}")

    sweep = threshold_sweep(y_ho, proba_ho, pnl_ho)
    best = max(sweep, key=lambda r: r["pnl"]) if sweep else None

    # Pick a "ship-eligible" threshold
    ship = None
    for r in sorted(sweep, key=lambda r: -r["pnl"]):
        if (r["dd"] <= SHIP_DD_MAX) and (r["n"] >= SHIP_N_MIN) and (r["wr"] >= SHIP_WR_MIN):
            ship = r
            break

    cmp_v15 = {
        "v15_baseline": V15_BASELINE_DE3,
        "v18_best_holdout": best,
        "v18_ship_threshold": ship,
        "v18_holdout_auc": auc_ho,
        "v18_train_oof_mean_auc_lr": float(np.nanmean(aucs_lr)),
        "v18_train_oof_mean_auc_hgb": float(np.nanmean(aucs_hgb)),
        "chosen_model": chosen,
    }

    bundle = {
        "model": final,
        "features": feats_all,
        "best_threshold": (best["threshold"] if best else 0.5),
        "ship_threshold": (ship["threshold"] if ship else None),
        "label": "pnl > 0 (winner)",
        "semantic": "keep if proba >= threshold",
        "description": f"V18-DE3 — V15 stack + Kronos. {chosen} on 6 base probas + 5 Kronos.",
        "comparison_vs_v15": cmp_v15,
        "holdout_sweep": sweep,
        "cv_fold_aucs_lr": [float(x) for x in aucs_lr],
        "cv_fold_aucs_hgb": [float(x) for x in aucs_hgb],
    }
    _ensure_dir(OUT_DE3)
    joblib.dump(bundle, OUT_DE3)
    print(f"  wrote {OUT_DE3}")

    results.update(cmp_v15)
    results["holdout_sweep_top3_pnl"] = sorted(sweep, key=lambda r: -r["pnl"])[:3]
    results["holdout_sweep"] = sweep
    _write_metrics_json(OUT_DE3_METRICS, results)
    return results


# --------------------------------------------------------------------- RA ---
def train_ra():
    print("[V18-RA] loading corpus + kronos RA features")
    if not os.path.exists(KRONOS_RA):
        print(f"  FATAL: {KRONOS_RA} missing")
        return {"target": "ra", "error": "kronos RA features missing"}

    corpus = pd.read_parquet(CORPUS_PATH)
    ra = corpus[(corpus["family"] == "regimeadaptive") & (corpus["allowed_by_friend_rule"] == True)].copy()
    ra = ra.reset_index(drop=True)
    ra["row_idx"] = ra.index
    print(f"  RA+friend rows: {len(ra)}")

    kronos = pd.read_parquet(KRONOS_RA)
    print(f"  Kronos RA rows: {len(kronos)}")

    merged = ra.merge(kronos[["row_idx"] + KRONOS_FEATS + ["kronos_pred_favorable"]], on="row_idx", how="inner")
    merged = merged.sort_values("ts").reset_index(drop=True)
    print(f"  merged rows: {len(merged)}")

    v17_feats = ["pct_dist_to_running_hi_pct", "k12_below_10", "pct_minutes_since_open"]
    miss = [f for f in v17_feats if f not in merged.columns]
    if miss:
        print(f"  ! V17 features missing in corpus: {miss}")
        return {"target": "ra", "error": f"missing v17 features {miss}"}

    feats_all = v17_feats + KRONOS_FEATS + ["kronos_pred_favorable"]
    X = merged[feats_all].astype(float).fillna(0.0).values
    y = (merged["net_pnl_after_haircut"] > 0).astype(int).values
    pnl = merged["net_pnl_after_haircut"].astype(float).values
    ts = pd.to_datetime(merged["ts"])

    cutoff = pd.Timestamp("2026-01-01", tz=ts.dt.tz) if ts.dt.tz is not None else pd.Timestamp("2026-01-01")
    holdout_mask = ts >= cutoff
    train_mask = ~holdout_mask
    print(f"  train rows: {train_mask.sum()}, holdout rows: {holdout_mask.sum()}")

    X_tr, y_tr, pnl_tr = X[train_mask.values], y[train_mask.values], pnl[train_mask.values]
    X_ho, y_ho, pnl_ho = X[holdout_mask.values], y[holdout_mask.values], pnl[holdout_mask.values]

    def hgb_factory():
        return HistGradientBoostingClassifier(max_depth=2, learning_rate=0.06, max_iter=200, random_state=42)

    print("  CV (HGB-d2)...")
    n_splits = max(2, min(5, max(2, X_tr.shape[0] // 25)))
    oof, aucs = cv_fit_predict(hgb_factory, X_tr, y_tr, n_splits=n_splits)
    print(f"    fold AUCs: {aucs}  mean={np.nanmean(aucs):.4f}")

    final = hgb_factory()
    final.fit(X_tr, y_tr) if len(X_tr) else final.fit(X, y)
    proba_ho = final.predict_proba(X_ho)[:, 1] if len(X_ho) else np.array([])
    auc_ho = safe_auc(y_ho, proba_ho) if len(X_ho) else float("nan")
    print(f"  holdout AUC: {auc_ho:.4f}")

    sweep = threshold_sweep(y_ho, proba_ho, pnl_ho) if len(X_ho) else []
    best = max(sweep, key=lambda r: r["pnl"]) if sweep else None
    ship = None
    for r in sorted(sweep, key=lambda r: -r["pnl"]):
        if (r["dd"] <= SHIP_DD_MAX) and (r["n"] >= max(20, SHIP_N_MIN // 3)) and (r["wr"] >= SHIP_WR_MIN):
            ship = r
            break

    cmp_v17 = {
        "v17_baseline_threshold": V17_BASELINE_RA_THRESH,
        "v17_baseline_cv_auc": 0.7344,
        "v18_cv_mean_auc": float(np.nanmean(aucs)),
        "v18_holdout_auc": auc_ho,
        "v18_best_holdout": best,
        "v18_ship_threshold": ship,
    }

    bundle = {
        "model": final,
        "features": feats_all,
        "best_threshold": (best["threshold"] if best else V17_BASELINE_RA_THRESH),
        "ship_threshold": (ship["threshold"] if ship else None),
        "label": "rule_keep AND winner",
        "description": "V18-RA — V17 RA features + Kronos features (HGB-d2).",
        "comparison_vs_v17": cmp_v17,
        "holdout_sweep": sweep,
        "cv_fold_aucs": [float(x) for x in aucs],
    }
    _ensure_dir(OUT_RA)
    joblib.dump(bundle, OUT_RA)
    print(f"  wrote {OUT_RA}")

    ra_results = {"target": "ra", "n_total": int(len(merged)), "features": feats_all,
                  "holdout_sweep": sweep, "cv_fold_aucs": [float(x) for x in aucs], **cmp_v17}
    _write_metrics_json(OUT_RA_METRICS, ra_results)
    return ra_results


# --------------------------------------------------------------------- AF ---
def train_af():
    print("[V18-AF] loading kronos AF features")
    if not os.path.exists(KRONOS_AF):
        print(f"  FATAL: {KRONOS_AF} missing")
        return {"target": "af", "error": "kronos AF features missing"}

    kronos = pd.read_parquet(KRONOS_AF)
    kronos = kronos.sort_values("ts").reset_index(drop=True)
    print(f"  AF NY rows: {len(kronos)}")

    # Build features: regime OHE, sub_strategy OHE, side, hour, kronos
    df = kronos.copy()
    df["hour_et"] = pd.to_datetime(df["ts"]).dt.tz_convert("US/Eastern").dt.hour if hasattr(df["ts"].iloc[0], "tz") else pd.to_datetime(df["ts"]).dt.hour
    df["side_long"] = (df["side"].astype(str).str.upper() == "LONG").astype(int)

    regimes = ["CHOP_SPIRAL", "DISPERSED", "TREND_GEODESIC"]
    for r in regimes:
        df[f"regime_{r}"] = (df["regime"].astype(str) == r).astype(int)

    subs = ["transition_burst", "aligned_flow", "compression_release"]
    for s in subs:
        df[f"sub_{s}"] = (df["sub_strategy"].astype(str) == s).astype(int)

    feats_all = (
        [f"regime_{r}" for r in regimes]
        + [f"sub_{s}" for s in subs]
        + ["side_long", "hour_et"]
        + KRONOS_FEATS
        + ["kronos_pred_favorable"]
    )
    feats_all = [f for f in feats_all if f in df.columns]
    print(f"  features ({len(feats_all)}): {feats_all}")

    X = df[feats_all].astype(float).fillna(0.0).values
    y = (df["pnl_dollars"] > 0).astype(int).values
    pnl = df["pnl_dollars"].astype(float).values
    ts = pd.to_datetime(df["ts"])

    cutoff = pd.Timestamp("2026-01-01", tz=ts.dt.tz) if ts.dt.tz is not None else pd.Timestamp("2026-01-01")
    holdout_mask = ts >= cutoff
    train_mask = ~holdout_mask
    print(f"  train rows: {train_mask.sum()}, holdout rows: {holdout_mask.sum()}")

    # Baseline: TREND_GEODESIC regime filter only -> always keep TG
    tg_keep = df["regime"].astype(str) == "TREND_GEODESIC"
    tg_pnl = pnl[tg_keep.values]
    tg_y = y[tg_keep.values]
    baseline = {
        "name": "TREND_GEODESIC regime filter",
        "n": int(tg_keep.sum()),
        "wr": float(tg_y.mean()) if len(tg_y) else 0.0,
        "pnl": float(tg_pnl.sum()) if len(tg_pnl) else 0.0,
        "dd": equity_drawdown(tg_pnl) if len(tg_pnl) else 0.0,
    }
    print(f"  baseline (TG regime filter): {baseline}")

    X_tr, y_tr = X[train_mask.values], y[train_mask.values]
    X_ho, y_ho, pnl_ho = X[holdout_mask.values], y[holdout_mask.values], pnl[holdout_mask.values]

    def hgb_factory():
        return HistGradientBoostingClassifier(max_depth=2, learning_rate=0.06, max_iter=200, random_state=42)

    n_splits = max(2, min(5, max(2, X_tr.shape[0] // 25)))
    print(f"  CV (HGB-d2) n_splits={n_splits}...")
    oof, aucs = cv_fit_predict(hgb_factory, X_tr, y_tr, n_splits=n_splits) if len(X_tr) >= n_splits + 1 else (np.array([]), [])
    print(f"    fold AUCs: {aucs}  mean={np.nanmean(aucs) if aucs else float('nan'):.4f}")

    final = hgb_factory()
    if len(X_tr):
        final.fit(X_tr, y_tr)
    else:
        final.fit(X, y)

    proba_ho = final.predict_proba(X_ho)[:, 1] if len(X_ho) else np.array([])
    auc_ho = safe_auc(y_ho, proba_ho) if len(X_ho) else float("nan")
    print(f"  holdout AUC: {auc_ho:.4f}")

    sweep = threshold_sweep(y_ho, proba_ho, pnl_ho) if len(X_ho) else []
    best = max(sweep, key=lambda r: r["pnl"]) if sweep else None

    cmp_af = {
        "baseline_tg_regime": baseline,
        "v18_cv_mean_auc": float(np.nanmean(aucs)) if aucs else float("nan"),
        "v18_holdout_auc": auc_ho,
        "v18_best_holdout": best,
    }

    bundle = {
        "model": final,
        "features": feats_all,
        "best_threshold": (best["threshold"] if best else 0.5),
        "label": "pnl_dollars > 0",
        "description": "V18-AF — small classifier on AF NY trades + Kronos features (HGB-d2).",
        "comparison_vs_baseline": cmp_af,
        "holdout_sweep": sweep,
        "cv_fold_aucs": [float(x) for x in aucs] if aucs else [],
    }
    _ensure_dir(OUT_AF)
    joblib.dump(bundle, OUT_AF)
    print(f"  wrote {OUT_AF}")

    af_results = {"target": "af", "n_total": int(len(df)), "features": feats_all,
                  "holdout_sweep": sweep,
                  "cv_fold_aucs": [float(x) for x in aucs] if aucs else [], **cmp_af}
    _write_metrics_json(OUT_AF_METRICS, af_results)
    return af_results


# ----------------------------------------------------------------- REPORT ---
def final_report():
    print("[V18 final report]")
    out = {"version": "v18", "comparisons": {}}
    for tgt, path in [("de3", OUT_DE3), ("ra", OUT_RA), ("af", OUT_AF)]:
        if not os.path.exists(path):
            out["comparisons"][tgt] = {"status": "missing", "path": path}
            continue
        try:
            b = joblib.load(path)
            entry = {
                "status": "ok",
                "features": b.get("features"),
                "best_threshold": b.get("best_threshold"),
                "ship_threshold": b.get("ship_threshold"),
                "description": b.get("description"),
            }
            for k in ("comparison_vs_v15", "comparison_vs_v17", "comparison_vs_baseline"):
                if k in b:
                    entry[k] = b[k]
            entry["holdout_sweep_top3_pnl"] = sorted(b.get("holdout_sweep", []), key=lambda r: -r["pnl"])[:3]
            out["comparisons"][tgt] = entry
        except Exception as e:
            out["comparisons"][tgt] = {"status": "error", "error": str(e)}

    _ensure_dir(OUT_REPORT)
    with open(OUT_REPORT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  wrote {OUT_REPORT}")

    # ---------- Markdown unified report ----------
    md = []
    md.append("# V18 Kronos Integration — Unified Comparison\n")
    md.append("Source: artifacts/regime_ml_v18_*/.../metrics.json + model.joblib bundles\n")

    # ---- DE3 ----
    md.append("\n## DE3 — V15 baseline vs V18 (V15 6-proba stack + 5 Kronos features)\n")
    de3 = out["comparisons"].get("de3", {})
    if de3.get("status") != "ok":
        md.append(f"_status: {de3.get('status')} — {de3.get('error', de3.get('path',''))}_")
    else:
        cmp15 = de3.get("comparison_vs_v15", {}) or {}
        v15b = cmp15.get("v15_baseline", {})
        ship = cmp15.get("v18_ship_threshold")
        best = cmp15.get("v18_best_holdout")
        md.append(f"- chosen model: **{cmp15.get('chosen_model')}**")
        md.append(f"- V15 baseline (spec): thr={v15b.get('threshold')} n={v15b.get('n_keep')} wr={v15b.get('win_rate'):.4f} pnl=${v15b.get('pnl_dollars')} dd=${v15b.get('max_drawdown_dollars')}")
        md.append(f"- V18 holdout AUC: {cmp15.get('v18_holdout_auc')}")
        md.append(f"- V18 train OOF AUC: LR={cmp15.get('v18_train_oof_mean_auc_lr')}  HGB={cmp15.get('v18_train_oof_mean_auc_hgb')}")
        if best:
            md.append(f"- V18 best holdout: thr={best.get('threshold')} n={best.get('n')} wr={best.get('wr'):.4f} pnl=${best.get('pnl'):.0f} dd=${best.get('dd'):.0f}")
        if ship:
            md.append(f"- V18 ship-eligible: thr={ship.get('threshold')} n={ship.get('n')} wr={ship.get('wr'):.4f} pnl=${ship.get('pnl'):.0f} dd=${ship.get('dd'):.0f}")
        else:
            md.append("- V18 ship-eligible: **none** (no threshold satisfies G1/G2/G3/G4)")

    # ---- RA ----
    md.append("\n## RA — V17 baseline vs V18 (V17 RA features + Kronos)\n")
    ra = out["comparisons"].get("ra", {})
    if ra.get("status") != "ok":
        md.append(f"_status: {ra.get('status')} — {ra.get('error', ra.get('path',''))}_")
    else:
        cmp17 = ra.get("comparison_vs_v17", {}) or {}
        ship = cmp17.get("v18_ship_threshold")
        best = cmp17.get("v18_best_holdout")
        md.append(f"- V17 published cv_auc: {cmp17.get('v17_baseline_cv_auc')}  baseline thr: {cmp17.get('v17_baseline_threshold')}")
        md.append(f"- V17 train ref (spec): n=50 wr=70% pnl=$372 dd=$55")
        md.append(f"- V18 CV mean AUC: {cmp17.get('v18_cv_mean_auc')}")
        md.append(f"- V18 holdout AUC: {cmp17.get('v18_holdout_auc')}")
        if best:
            md.append(f"- V18 best holdout: thr={best.get('threshold')} n={best.get('n')} wr={best.get('wr'):.4f} pnl=${best.get('pnl'):.0f} dd=${best.get('dd'):.0f}")
        if ship:
            md.append(f"- V18 ship-eligible: thr={ship.get('threshold')} n={ship.get('n')} wr={ship.get('wr'):.4f} pnl=${ship.get('pnl'):.0f} dd=${ship.get('dd'):.0f}")
        else:
            md.append("- V18 ship-eligible: **none**")

    # ---- AF ----
    md.append("\n## AF — TREND_GEODESIC regime filter vs V18 (HGB-d2 + Kronos)\n")
    af = out["comparisons"].get("af", {})
    if af.get("status") != "ok":
        md.append(f"_status: {af.get('status')} — {af.get('error', af.get('path',''))}_")
    else:
        cmpaf = af.get("comparison_vs_baseline", {}) or {}
        b = cmpaf.get("baseline_tg_regime", {})
        best = cmpaf.get("v18_best_holdout")
        md.append(f"- AF spec ref: n=14 wr=50% pnl=$820 (TREND_GEODESIC train)")
        md.append(f"- TREND_GEODESIC computed: n={b.get('n')} wr={b.get('wr'):.4f} pnl=${b.get('pnl'):.0f} dd=${b.get('dd'):.0f}")
        md.append(f"- V18 CV mean AUC: {cmpaf.get('v18_cv_mean_auc')}")
        md.append(f"- V18 holdout AUC: {cmpaf.get('v18_holdout_auc')}")
        if best:
            md.append(f"- V18 best holdout: thr={best.get('threshold')} n={best.get('n')} wr={best.get('wr'):.4f} pnl=${best.get('pnl'):.0f} dd=${best.get('dd'):.0f}")

    # ---- Combined-stack projection ----
    md.append("\n## Combined-stack projection (best ship-eligible per strategy)\n")
    total_pnl = 0.0
    total_n = 0
    total_dd = 0.0
    rows = []
    for tgt, comp in out["comparisons"].items():
        if comp.get("status") != "ok":
            rows.append(f"- {tgt}: not available")
            continue
        # try ship_threshold first, fall back to v18_best_holdout
        for k in ("comparison_vs_v15", "comparison_vs_v17", "comparison_vs_baseline"):
            c = comp.get(k) or {}
            if "v18_ship_threshold" in c or "v18_best_holdout" in c:
                pick = c.get("v18_ship_threshold") or c.get("v18_best_holdout")
                if pick:
                    rows.append(f"- {tgt}: thr={pick.get('threshold')} n={pick.get('n')} wr={pick.get('wr'):.4f} pnl=${pick.get('pnl'):.0f} dd=${pick.get('dd'):.0f}")
                    total_pnl += float(pick.get("pnl", 0.0))
                    total_n += int(pick.get("n", 0))
                    total_dd += float(pick.get("dd", 0.0))
                break
    md.extend(rows)
    md.append(f"- **Combined: n={total_n} pnl=${total_pnl:.0f} dd-sum=${total_dd:.0f}**\n")

    with open(FINAL_REPORT_MD, "w") as f:
        f.write("\n".join(md))
    print(f"  wrote {FINAL_REPORT_MD}")
    print(json.dumps(out, indent=2, default=str)[:4000])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["de3", "ra", "af", "final_report"])
    args = ap.parse_args()

    try:
        if args.target == "de3":
            r = train_de3()
        elif args.target == "ra":
            r = train_ra()
        elif args.target == "af":
            r = train_af()
        elif args.target == "final_report":
            r = final_report()
        else:
            r = {"error": "unknown target"}
        print("\nRESULT:", json.dumps(r, indent=2, default=str)[:2000])
        return 0
    except Exception as e:
        print(f"FATAL in target {args.target}: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
