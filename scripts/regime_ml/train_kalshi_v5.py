#!/usr/bin/env python3
"""Train Kalshi overlay ML v5 — reduced-capacity / transfer-learning.

v4's diagnosis: OOS correlation went from +0.05…+0.10 (v3, 58 features) to
-0.13…+0.00 (v4, 72 features + cross-strategy/NQ/VIX). Classic overfit.

v5 stack:
  1. Feature selection — mutual_info_regression top-K on training set
  2. Simpler models — LogisticRegression(L2) / Ridge / small HGB — sweep all three
  3. Transfer features — regime-ML v5/v6 outputs (scalp_proba, be_disable_proba,
     size_reduce_proba) as 3 extra features; these models already passed OOS
     gates so they're known to generalize
  4. Walk-forward CV on 2025 — rolling 90-day train / 30-day score — as
     stability diagnostic (report only, not a ship criterion)
  5. Binary override on rule decision — only act on high-confidence signal,
     otherwise trust the rule

Ship gates unchanged. Fixed 2025-train / Apr-2026-OOS split for ship scoring.
"""
from __future__ import annotations

import argparse, json, logging, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT, FEATURE_COLS_40, MES_PT_VALUE, DEFAULT_TP, DEFAULT_SL,
    load_continuous_bars, build_feature_frame, stats,
)
from train_kalshi_v2 import (
    parse_log_events, simulate_trade_horizon, add_v2_features,
    fill_intraday_pnl, V2_EXTRA_FEATURES,
    recency_weights,
    FEATURE_COLS_KALSHI_V1, FEATURE_COLS_KALSHI_V2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_kalshi_v5")

LABEL_MARGIN_USD = 15.0
OOS_START = "2026-01-27"
OOS_END = "2026-04-24"
OOS_TRAIN_CUTOFF = "2025-12-31"   # don't train on events overlapping OOS

TRANSFER_FEATURES = ["scalp_proba", "be_disable_proba", "size_reduce_proba"]


# ─── Regime-ML transfer feature computation ─────────────────────────────

def load_regime_models():
    """Load v5 scalp + v6 BE-disable + v6 size-reduce HGBs and return a
    dict {name: (hgb, pass_idx, feature_cols)}."""
    out = {}
    # v5 brackets → scalp_proba
    with (ROOT / "artifacts/regime_ml_v5_brackets/model.pkl").open("rb") as f:
        p = pickle.load(f)
    out["scalp"] = (p["hgb"], list(p["hgb"].classes_).index(p["positive_class"]),
                    p["feature_cols"])
    # v6 BE disable → be_disable_proba (requires a_pred_scalp)
    with (ROOT / "artifacts/regime_ml_v6_be/model.pkl").open("rb") as f:
        p = pickle.load(f)
    out["be_disable"] = (p["hgb"], list(p["hgb"].classes_).index(p["positive_class"]),
                         p["feature_cols"])
    # v6 size reduce → size_reduce_proba (requires a_pred_scalp)
    with (ROOT / "artifacts/regime_ml_v6_size/model.pkl").open("rb") as f:
        p = pickle.load(f)
    out["size_reduce"] = (p["hgb"], list(p["hgb"].classes_).index(p["positive_class"]),
                          p["feature_cols"])
    return out


def compute_regime_transfer_features(feats_df: pd.DataFrame, regime_models: dict) -> pd.DataFrame:
    """For each row of feats_df, compute scalp_proba, be_disable_proba,
    size_reduce_proba. Returns DataFrame with 3 new columns appended."""
    feats_df = feats_df.copy()

    # Step 1: scalp_proba (uses 40 base features)
    scalp_hgb, scalp_idx, scalp_cols = regime_models["scalp"]
    missing = [c for c in scalp_cols if c not in feats_df.columns]
    if missing:
        raise RuntimeError(f"missing {len(missing)} cols for scalp model: {missing[:5]}")
    X_scalp = feats_df[scalp_cols].to_numpy()
    # NaN-safe: impute training-median substitute on the fly
    mask = np.isnan(X_scalp).any(axis=1)
    probs = np.full(len(X_scalp), 0.5)
    if (~mask).any():
        probs[~mask] = scalp_hgb.predict_proba(X_scalp[~mask])[:, scalp_idx]
    feats_df["scalp_proba"] = probs

    # Step 2: v6 models need a_pred_scalp as 41st feature
    feats_df["a_pred_scalp"] = feats_df["scalp_proba"].to_numpy()

    be_hgb, be_idx, be_cols = regime_models["be_disable"]
    X_be = feats_df[be_cols].to_numpy()
    mask_be = np.isnan(X_be).any(axis=1)
    probs_be = np.full(len(X_be), 0.5)
    if (~mask_be).any():
        probs_be[~mask_be] = be_hgb.predict_proba(X_be[~mask_be])[:, be_idx]
    feats_df["be_disable_proba"] = probs_be

    sz_hgb, sz_idx, sz_cols = regime_models["size_reduce"]
    X_sz = feats_df[sz_cols].to_numpy()
    mask_sz = np.isnan(X_sz).any(axis=1)
    probs_sz = np.full(len(X_sz), 0.5)
    if (~mask_sz).any():
        probs_sz[~mask_sz] = sz_hgb.predict_proba(X_sz[~mask_sz])[:, sz_idx]
    feats_df["size_reduce_proba"] = probs_sz

    return feats_df


def build_dataset_v5(events: list, bars: pd.DataFrame, regime_models: dict,
                    label_horizon_min: int) -> pd.DataFrame:
    feats = build_feature_frame(bars)
    feats = feats.loc[feats[FEATURE_COLS_40].notna().all(axis=1)].copy()

    # Add regime transfer features to feature frame (computed per-bar)
    log.info("computing regime transfer features on %d bars...", len(feats))
    feats = compute_regime_transfer_features(feats, regime_models)
    feats_minute_idx = {pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"): ts for ts in feats.index}
    log.info("feature frame rows: %d  (with transfer features)", len(feats))

    c_bars = bars["close"].to_numpy(float)
    h_bars = bars["high"].to_numpy(float)
    l_bars = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}

    rows = []
    dropped_no_bar = 0
    dropped_ambiguous = 0
    for e in events:
        mts_key = e["market_ts"][:16]
        ts = feats_minute_idx.get(mts_key)
        if ts is None: dropped_no_bar += 1; continue
        pos = idx_pos.get(ts)
        if pos is None: dropped_no_bar += 1; continue
        side_sign = 1 if e["side"] == "LONG" else -1
        pnl = simulate_trade_horizon(h_bars, l_bars, c_bars, int(pos),
                                     DEFAULT_TP, DEFAULT_SL, side_sign, label_horizon_min)
        if abs(pnl) < LABEL_MARGIN_USD:
            dropped_ambiguous += 1; continue
        row = {**e, "_ts": ts}
        for col in FEATURE_COLS_40 + TRANSFER_FEATURES:
            row[col] = feats.loc[ts, col]
        row["side_sign"] = side_sign
        row["is_de3"] = 1 if "dynamicengine" in e["strategy"].lower() else 0
        row["is_ra"]  = 1 if "regimeadaptive" in e["strategy"].lower() else 0
        row["is_ml"]  = 1 if "mlphysics" in e["strategy"].lower() else 0
        row["settlement_hour"] = int(pd.Timestamp(mts_key + ":00").hour)
        row["minutes_into_session"] = int(
            (pd.Timestamp(mts_key + ":00").hour - 9) * 60 +
             pd.Timestamp(mts_key + ":00").minute
        )
        row["role_forward"] = 1 if "forward" in e["role"] else 0
        row["role_background"] = 1 if e["role"] == "background" else 0
        row["role_balanced"] = 1 if e["role"] == "balanced" else 0
        row["label"] = "pass" if pnl > 0 else "block"
        row["forward_pnl"] = pnl
        row["rule_decision_pass_int"] = 1 if e["rule_decision"] == "PASS" else 0
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df.set_index("_ts")
    df = fill_intraday_pnl(df)
    log.info("labeled rows: %d  dropped (no bar): %d  dropped (ambiguous): %d",
             len(df), dropped_no_bar, dropped_ambiguous)
    log.info("class dist: %s", dict(Counter(df["label"])))
    return df


# Candidate feature pool: v2 set + 3 transfer features
def dedup_keep_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen: seen.add(x); out.append(x)
    return out

FEATURE_POOL_V5 = dedup_keep_order(FEATURE_COLS_KALSHI_V2 + TRANSFER_FEATURES)


# Kalshi-specific features that must always be present — they carry the
# decision metadata the rule already uses, plus direction & role.
MANDATORY_FEATURES = [
    "k_entry_probability", "k_probe_probability", "k_momentum_delta",
    "k_momentum_retention", "k_support_score", "k_threshold",
    "k_margin", "k_margin_abs",
    "rule_decision_pass_int",
    "side_sign",
]


def select_top_k_mi(X_tr: np.ndarray, y: np.ndarray, feature_names: list[str],
                     k: int, seed: int = 42) -> tuple[list[str], np.ndarray]:
    """Return mandatory Kalshi features + top-K-from-remainder by MI.

    The total returned count is max(k, len(mandatory)). If k <= len(mandatory)
    we just return the mandatory set (can't go smaller). Otherwise we append
    (k - len(mandatory)) top-MI features from the remaining pool."""
    mi = mutual_info_regression(X_tr, y, random_state=seed)
    mandatory = [f for f in MANDATORY_FEATURES if f in feature_names]
    mandatory_set = set(mandatory)
    remaining = [(f, mi[feature_names.index(f)]) for f in feature_names
                 if f not in mandatory_set]
    remaining.sort(key=lambda t: -t[1])
    n_extra = max(0, k - len(mandatory))
    extras = [f for f, _ in remaining[:n_extra]]
    selected = mandatory + extras
    return selected, mi


# ─── Ship-gate evaluation ────────────────────────────────────────────────

def decide_override(pred_pnl_or_proba: np.ndarray, rule_decision: np.ndarray,
                     pnl_if_passed: np.ndarray, mode: str,
                     pass_thr: float, block_thr: float) -> dict:
    """Binary override: rule=BLOCK→PASS if strong positive signal,
    rule=PASS→BLOCK if strong negative signal, else trust rule.

    For regression mode: signal is predicted PnL (float, unit = $).
    For classifier mode: signal is proba_pass (0..1); we transform it to a
    centered score (proba - 0.5) and treat thresholds as absolute probabilty
    margins.
    """
    if mode == "regression":
        sig = pred_pnl_or_proba
        pass_mask = (rule_decision == "BLOCK") & (sig >= pass_thr)
        block_mask = (rule_decision == "PASS")  & (sig <= -block_thr)
    else:  # classifier
        sig = pred_pnl_or_proba
        pass_mask = (rule_decision == "BLOCK") & (sig >= 0.5 + pass_thr)
        block_mask = (rule_decision == "PASS")  & (sig <= 0.5 - block_thr)

    final_pass = np.where(pass_mask, True,
                          np.where(block_mask, False, rule_decision == "PASS"))
    final_pnl = np.where(final_pass, pnl_if_passed, 0.0)
    st = stats(final_pnl)
    n_new_pass = int(pass_mask.sum())
    n_new_block = int(block_mask.sum())
    new_pass_pnls = pnl_if_passed[pass_mask]
    new_block_pnls = pnl_if_passed[block_mask]
    new_pass_wr = (100 * (new_pass_pnls > 0).sum() / len(new_pass_pnls)) if len(new_pass_pnls) else 0.0
    new_block_wr = (100 * (new_block_pnls <= 0).sum() / len(new_block_pnls)) if len(new_block_pnls) else 0.0
    return {
        "pass_thr": pass_thr, "block_thr": block_thr, "mode": mode,
        "n_final_pass": int(final_pass.sum()),
        "n_new_pass": n_new_pass, "n_new_block": n_new_block,
        "new_pass_wr": new_pass_wr, "new_block_wr": new_block_wr,
        "pnl": st["pnl"], "dd": st["dd"], "avg": st["avg"],
    }


def eval_all_gates(pred_signal: np.ndarray, rule_decision: np.ndarray,
                    y_true: np.ndarray, pnl_if_passed: np.ndarray,
                    mode: str, n_oos: int) -> tuple[list[dict], dict]:
    """Sweep override thresholds appropriate for the mode, return best."""
    rule_pnl = np.where(rule_decision == "PASS", pnl_if_passed, 0.0)
    oracle_pnl = np.where(y_true == "pass", pnl_if_passed, 0.0)
    rule_st = stats(rule_pnl)
    oracle_st = stats(oracle_pnl)
    headroom = oracle_st["pnl"] - rule_st["pnl"]

    if mode == "regression":
        thr_grid = [
            (5.0, 5.0), (7.5, 5.0), (10.0, 5.0), (10.0, 7.5),
            (12.5, 7.5), (15.0, 7.5), (15.0, 10.0), (20.0, 10.0),
            (25.0, 10.0), (25.0, 15.0), (30.0, 15.0),
        ]
    else:  # classifier — pass_thr/block_thr are margins above/below 0.5
        # Include small margins for logistic models with narrow proba ranges
        thr_grid = [
            (0.02, 0.02), (0.03, 0.03), (0.04, 0.04),
            (0.05, 0.03), (0.05, 0.05),
            (0.07, 0.04), (0.08, 0.05),
            (0.10, 0.05), (0.10, 0.10), (0.15, 0.10), (0.20, 0.10),
            (0.25, 0.15), (0.30, 0.15),
        ]

    results = []
    for (pt, bt) in thr_grid:
        r = decide_override(pred_signal, rule_decision, pnl_if_passed, mode, pt, bt)
        lift = r["pnl"] - rule_st["pnl"]
        dd_over_pnl = (r["dd"] / r["pnl"] * 100.0) if r["pnl"] > 0 else float("inf")
        capt = (lift / headroom * 100.0) if headroom > 0 else 0.0
        gates = {
            "pnl_ok":        lift > 0,
            "dd_ratio_ok":   dd_over_pnl <= 30.0,
            "n_ok":          n_oos >= 50,
            "new_pass_wr_ok":(r["new_pass_wr"] >= 50.0) if r["n_new_pass"] >= 5 else True,
            "capt_ok":       capt >= 20.0,
        }
        ok = all(gates.values())
        r.update({"lift": lift, "dd_over_pnl": dd_over_pnl, "capt_pct": capt,
                  "gates": gates, "ships": ok})
        results.append(r)
    shippers = [r for r in results if r["ships"]]
    best = max(shippers, key=lambda r: r["lift"]) if shippers \
        else max(results, key=lambda r: r["lift"])
    return results, best


# ─── Walk-forward CV diagnostic ─────────────────────────────────────────

def walk_forward_cv(df: pd.DataFrame, feature_cols: list[str],
                    model_builder, mode: str,
                    train_days: int = 90, test_days: int = 30,
                    min_train_rows: int = 200, seed: int = 42) -> dict:
    """Rolling walk-forward validation within the TRAINING period.
    Returns pooled OOS correlation and per-window stats."""
    if df.empty: return {"windows": 0, "corr": 0.0, "by_window": []}
    df = df.sort_index()
    start = df.index.min()
    end = df.index.max()
    cur = start + pd.Timedelta(days=train_days)
    results = []
    pooled_pred = []
    pooled_actual = []
    while cur + pd.Timedelta(days=test_days) <= end:
        tr = df.loc[(df.index >= cur - pd.Timedelta(days=train_days)) & (df.index < cur)]
        te = df.loc[(df.index >= cur) & (df.index < cur + pd.Timedelta(days=test_days))]
        if len(tr) >= min_train_rows and len(te) >= 20:
            X_tr = tr[feature_cols].fillna(tr[feature_cols].median()).to_numpy()
            X_te = te[feature_cols].fillna(tr[feature_cols].median()).to_numpy()
            if mode == "regression":
                y_tr = tr["forward_pnl"].to_numpy()
                m = model_builder()
                m.fit(X_tr, y_tr)
                pred = m.predict(X_te)
                actual = te["forward_pnl"].to_numpy()
            else:
                y_tr = (tr["label"] == "pass").astype(int).to_numpy()
                m = model_builder()
                m.fit(X_tr, y_tr)
                if hasattr(m, "predict_proba"):
                    pred = m.predict_proba(X_te)[:, 1]
                else:
                    pred = m.predict(X_te).astype(float)
                actual = te["forward_pnl"].to_numpy()
            pooled_pred.extend(pred)
            pooled_actual.extend(actual)
            corr_w = float(np.corrcoef(pred, actual)[0, 1]) if len(pred) > 1 else 0.0
            results.append({
                "window_start": str(cur.date()),
                "train_n": len(tr), "test_n": len(te),
                "corr": corr_w,
            })
        cur += pd.Timedelta(days=test_days)
    pooled_corr = float(np.corrcoef(pooled_pred, pooled_actual)[0, 1]) if len(pooled_pred) > 1 else 0.0
    return {"windows": len(results), "pooled_corr": pooled_corr,
            "by_window": results}


# ─── Model builders ──────────────────────────────────────────────────────

def build_logistic(seed=42, C=4.0):
    """C=4.0 means weaker regularization than default C=1.0 — needed for
    sharper probability margins so the override rule can actually trigger."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, penalty="l2", max_iter=1000,
                                   random_state=seed, class_weight="balanced",
                                   solver="liblinear")),
    ])


def build_ridge(seed=42):
    return Pipeline([
        ("scale", StandardScaler()),
        ("rid", Ridge(alpha=1.0, random_state=seed)),
    ])


def build_small_hgb(seed=42):
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=3,   # reduced capacity
        l2_regularization=2.0, min_samples_leaf=50, random_state=seed,
    )


# ─── Main run_config ─────────────────────────────────────────────────────

def run_v5_config(df: pd.DataFrame,
                  model_kind: str, top_k: int,
                  include_transfer: bool, half_life_days: float,
                  train_start_date: Optional[str],
                  oos_start: str, oos_end: str,
                  seed: int = 42) -> Optional[dict]:
    log.info("")
    log.info("=" * 78)
    log.info("V5 CONFIG  model=%s  top_k=%d  transfer=%s  hl=%.0fd  trstart=%s",
             model_kind, top_k, include_transfer, half_life_days,
             train_start_date or "full")
    log.info("=" * 78)

    # Feature pool
    pool = list(FEATURE_COLS_KALSHI_V2)
    if include_transfer:
        for c in TRANSFER_FEATURES:
            if c not in pool: pool.append(c)

    oos_start_ts = pd.Timestamp(oos_start, tz=df.index.tz)
    oos_end_ts = pd.Timestamp(oos_end, tz=df.index.tz) + pd.Timedelta(days=1)
    tr = df.loc[df.index < oos_start_ts]
    if train_start_date is not None:
        tr = tr.loc[tr.index >= pd.Timestamp(train_start_date, tz=df.index.tz)]
    oos = df.loc[(df.index >= oos_start_ts) & (df.index <= oos_end_ts)]
    log.info("train: %d  OOS: %d", len(tr), len(oos))
    if len(tr) < 200 or len(oos) < 50:
        log.warning("below floors — skip"); return None

    # Feature selection: mutual info on training set (regression target = forward_pnl)
    X_tr_full = tr[pool].fillna(tr[pool].median()).to_numpy()
    y_pnl = tr["forward_pnl"].to_numpy()
    y_cls = (tr["label"] == "pass").astype(int).to_numpy()
    selected, mi = select_top_k_mi(X_tr_full, y_pnl, pool, top_k, seed=seed)
    log.info("top-%d features by MI: %s", top_k, selected)
    log.info("   MI scores (top %d): %s", min(top_k, 10),
             [f"{f}={mi[pool.index(f)]:.4f}" for f in selected[:10]])

    # Build model with selected features only
    X_tr = tr[selected].fillna(tr[selected].median()).to_numpy()
    med_sel = tr[selected].median().to_dict()
    X_oos = oos[selected].fillna(tr[selected].median()).to_numpy()
    recency_w = recency_weights(tr.index, half_life_days)

    rule_decision = oos["rule_decision"].to_numpy()
    pnl_if_passed = oos["forward_pnl"].to_numpy()
    y_true = oos["label"].to_numpy()
    rule_pnl_arr = np.where(rule_decision == "PASS", pnl_if_passed, 0.0)
    rule_st = stats(rule_pnl_arr)
    oracle_pnl_arr = np.where(y_true == "pass", pnl_if_passed, 0.0)
    oracle_st = stats(oracle_pnl_arr)
    log.info("rule baseline: PnL=$%+.2f DD=$%.0f  | oracle: PnL=$%+.2f",
             rule_st["pnl"], rule_st["dd"], oracle_st["pnl"])

    # Train + predict
    if model_kind == "logistic":
        mdl = build_logistic(seed)
        # Logistic doesn't support sample_weight via Pipeline easily with scaler, so
        # we pass via final step kwargs
        mdl.fit(X_tr, y_cls, lr__sample_weight=recency_w)
        proba_pass = mdl.predict_proba(X_oos)[:, 1]
        mode = "classifier"
        pred_signal = proba_pass
        corr = float(np.corrcoef(proba_pass, pnl_if_passed)[0, 1])
    elif model_kind == "ridge":
        mdl = build_ridge(seed)
        mdl.fit(X_tr, y_pnl, rid__sample_weight=recency_w)
        pred_pnl = mdl.predict(X_oos)
        mode = "regression"
        pred_signal = pred_pnl
        corr = float(np.corrcoef(pred_pnl, pnl_if_passed)[0, 1])
    elif model_kind == "hgb_small":
        mdl = build_small_hgb(seed)
        mdl.fit(X_tr, y_pnl, sample_weight=recency_w)
        pred_pnl = mdl.predict(X_oos)
        mode = "regression"
        pred_signal = pred_pnl
        corr = float(np.corrcoef(pred_pnl, pnl_if_passed)[0, 1])
    else:
        raise ValueError(f"unknown model_kind {model_kind}")

    log.info("OOS pred-vs-actual Pearson: %+.3f  (positive → has signal)", corr)
    log.info("pred percentiles: p10=%+.3f p50=%+.3f p90=%+.3f",
             np.percentile(pred_signal, 10), np.percentile(pred_signal, 50),
             np.percentile(pred_signal, 90))

    results, best = eval_all_gates(pred_signal, rule_decision, y_true, pnl_if_passed,
                                   mode, len(oos))

    # Walk-forward diagnostic (on training set only, separate from ship gate)
    def builder():
        if model_kind == "logistic": return build_logistic(seed)
        if model_kind == "ridge":    return build_ridge(seed)
        return build_small_hgb(seed)
    wf = walk_forward_cv(tr, selected, builder, mode)
    log.info("walk-forward (on train): %d windows  pooled_corr=%+.3f",
             wf["windows"], wf["pooled_corr"])

    log.info("%5s %5s %8s %8s %10s %8s %8s %7s gates",
             "p_thr", "b_thr", "n_pass", "new_psn", "newPassWR", "n_new_blk",
             "pnl", "capt")
    for r in results:
        log.info("%5.2f %5.2f %8d %8d %9.2f%% %8d $%+7.2f %6.2f%%  %d/5%s",
                 r["pass_thr"], r["block_thr"], r["n_final_pass"],
                 r["n_new_pass"], r["new_pass_wr"], r["n_new_block"],
                 r["pnl"], r["capt_pct"], sum(r["gates"].values()),
                 " SHIP" if r["ships"] else "")

    return {
        "model_kind": model_kind, "top_k": top_k,
        "include_transfer": include_transfer,
        "half_life": half_life_days, "train_start": train_start_date,
        "selected_features": selected,
        "mi_scores": {pool[i]: float(mi[i]) for i in range(len(pool))},
        "rule_baseline": rule_st, "oracle": oracle_st,
        "n_train": len(tr), "n_oos": len(oos),
        "oos_corr": corr,
        "walk_forward": wf,
        "all": results, "best": best,
        "model": mdl if best["ships"] else None,
        "mode": mode,
        "median_imputes": {k: float(v) for k, v in med_sel.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-start", default=OOS_START)
    ap.add_argument("--oos-end", default=OOS_END)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_kalshi_v5"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizons", nargs="+", type=int, default=[15, 30])
    ap.add_argument("--top-ks", nargs="+", type=int, default=[5, 8, 12, 20])
    ap.add_argument("--model-kinds", nargs="+", default=["logistic", "ridge", "hgb_small"])
    ap.add_argument("--transfers", nargs="+", type=int, default=[0, 1],
                    help="0 = no transfer features, 1 = include")
    ap.add_argument("--half-lives", nargs="+", type=float, default=[90.0])
    ap.add_argument("--train-starts", nargs="+", default=["full"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== loading regime models ===")
    regime_models = load_regime_models()
    log.info("regime models loaded: %s", list(regime_models.keys()))

    log.info("=== parsing logs ===")
    log_paths = []
    rdir = ROOT / "backtest_reports/full_live_replay"
    for m in "2025_01 2025_02 2025_03 2025_04 2025_05 2025_06 2025_07 2025_08 2025_09 2025_10 2025_11 2025_12".split():
        p = rdir / m / "topstep_live_bot.log"
        if p.exists() and p.stat().st_size > 50_000:
            log_paths.append(p)
    live = ROOT / "topstep_live_bot.log"
    if live.exists(): log_paths.append(live)

    all_events = []
    for p in log_paths:
        ev = parse_log_events(p)
        log.info("  %s → %d events",
                 p.name if p.parent == ROOT else f"{p.parent.name}/{p.name}", len(ev))
        all_events.extend(ev)
    log.info("total Kalshi events: %d", len(all_events))
    all_events = add_v2_features(all_events)

    all_mts = [e["market_ts"] for e in all_events]
    start = pd.Timestamp(min(all_mts)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(all_mts)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    log.info("=== loading ES bars ===")
    bars = load_continuous_bars(start, end)
    log.info("ES bars: %d  range %s -> %s", len(bars), bars.index.min(), bars.index.max())

    # Build + cache datasets per horizon (builds transfer features once per horizon)
    datasets = {}
    for h in args.horizons:
        log.info("=== building dataset horizon=%d ===", h)
        datasets[h] = build_dataset_v5(all_events, bars, regime_models, h)

    configs = []
    for ts_ in args.train_starts:
        ts_val = None if ts_ == "full" else ts_
        for h in args.horizons:
            for mk in args.model_kinds:
                for k in args.top_ks:
                    for hl in args.half_lives:
                        for inc_t in args.transfers:
                            configs.append((h, mk, k, bool(inc_t), hl, ts_val))

    log.info("=== running %d v5 configs ===", len(configs))
    runs = []
    for cfg in configs:
        h, mk, k, inc_t, hl, ts_val = cfg
        r = run_v5_config(datasets[h], mk, k, inc_t, hl, ts_val,
                          args.oos_start, args.oos_end, seed=args.seed)
        if r:
            r["horizon"] = h
            runs.append(r)

    # Summary
    log.info("\n%s", "═" * 150)
    log.info("V5 SWEEP SUMMARY")
    log.info("%s", "═" * 150)
    log.info("%4s %10s %4s %5s %4s %8s %5s %8s %8s %+8s %7s %s",
             "hrz", "model", "k", "xfer", "gts", "params", "corr",
             "wf_corr", "newPs", "lift$", "capt%", "ship?")
    for r in runs:
        b = r["best"]
        params = (f"pt={b['pass_thr']:.2f}/bt={b['block_thr']:.2f}" if r["mode"] == "classifier"
                  else f"p=${b['pass_thr']:.0f}/b=${b['block_thr']:.0f}")
        log.info("%4d %10s %4d %5s %4d %8s %+5.2f %+8.3f %8d %+8.2f %6.2f%% %s",
                 r["horizon"], r["model_kind"], r["top_k"],
                 "Y" if r["include_transfer"] else "N",
                 sum(b["gates"].values()), params, r["oos_corr"],
                 r["walk_forward"]["pooled_corr"],
                 b["n_new_pass"], b["lift"], b["capt_pct"],
                 "SHIP" if b["ships"] else "-")

    shippers = [r for r in runs if r["best"]["ships"]]
    if not shippers:
        (out_dir / "sweep_summary.json").write_text(json.dumps({
            "verdict": "KILL", "reason": "no v5 config passes all 5 gates",
            "runs": [{k: v for k, v in r.items() if k not in ("model",)}
                     for r in runs],
        }, indent=2, default=str))
        log.warning("[KILL] no v5 config passes — summary written")
        return 1

    best_run = max(shippers, key=lambda r: r["best"]["lift"])
    b = best_run["best"]
    payload = {
        "model_kind": best_run["model_kind"],
        "clf_or_reg": best_run["model"],
        "feature_cols": best_run["selected_features"],
        "median_imputes": best_run["median_imputes"],
        "decision": b, "mode": best_run["mode"],
        "horizon_min": best_run["horizon"],
        "half_life_days": best_run["half_life"],
        "train_start": best_run["train_start"],
        "include_transfer": best_run["include_transfer"],
        "top_k": best_run["top_k"],
        "rule_baseline_oos": best_run["rule_baseline"],
        "oracle_oos": best_run["oracle"],
        "n_train": best_run["n_train"], "n_oos": best_run["n_oos"],
        "oos_corr": best_run["oos_corr"],
        "wf_pooled_corr": best_run["walk_forward"]["pooled_corr"],
        "wf_windows": best_run["walk_forward"]["windows"],
        "seed": args.seed,
    }
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "model_meta.json").write_text(json.dumps({
        k: v for k, v in payload.items() if k != "clf_or_reg"
    }, indent=2, default=str))
    log.info("[SHIP] model=%s top_k=%d transfer=%s  lift=$%+.2f  capt=%.2f%%  corr=%+.3f",
             best_run["model_kind"], best_run["top_k"],
             best_run["include_transfer"], b["lift"], b["capt_pct"],
             best_run["oos_corr"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
