"""Rolling-origin A/B comparison: v1 features vs v1 + encoder + cross-market.

The v1-vs-v2 AUC comparison printed by retrain_with_encoder.py is
fundamentally biased because the v1 model on disk was trained on the
FULL historical dataset, including what's treated as v2's holdout.
v1 has seen the "test" data; v2 hasn't.

This script fixes that by running a proper rolling-origin A/B: at each
split boundary, it trains BOTH variants from scratch on the pre-split
data, evaluates both on the post-split test chunk, and reports deltas.

Splits: 6 rolling windows, starting at train_frac = 0.40 and stepping
+0.10 each time (matches the rolling-origin protocol used in the
other Kalshi trainers for consistency).

Usage:
  python3 scripts/signal_gate/rolling_origin_ab.py lfo
  python3 scripts/signal_gate/rolling_origin_ab.py kalshi_tp
  python3 scripts/signal_gate/rolling_origin_ab.py pivot
  python3 scripts/signal_gate/rolling_origin_ab.py all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.signal_gate.retrain_with_encoder import (
    _load_bar_cache, compute_encoder_embeddings, compute_cross_market_features,
)

NY = ZoneInfo("America/New_York")
ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"


def _encode_categorical(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    parts = []
    for col in columns:
        known = sorted(df[col].astype(str).unique().tolist())
        enc = pd.DataFrame(
            {f"{col}__{v}": (df[col].astype(str) == v).astype(int) for v in known},
            index=df.index,
        )
        parts.append(enc)
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def _fit_predict_auc(X_tr, y_tr, X_te, y_te, *, clf_kwargs: dict) -> float:
    if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
        return float("nan")
    clf = GradientBoostingClassifier(**clf_kwargs)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
    return float(roc_auc_score(y_te, p))


def _rolling_splits(n: int, min_train_frac: float = 0.40, step: float = 0.10,
                    min_test_rows: int = 100):
    t = min_train_frac
    while t + step <= 1.0 + 1e-9:
        tr_end = int(n * t)
        te_end = min(n, int(n * (t + step)))
        if te_end - tr_end >= min_test_rows:
            yield t, t + step, tr_end, te_end
        t += step


# ---------- per-layer A/B ----------

def ab_lfo():
    parquet = ARTIFACTS / "lfo_training_data.parquet"
    ds = pd.read_parquet(parquet)
    NUMERIC = [
        "de3_entry_ret1_atr", "de3_entry_body_pos1", "de3_entry_body1_ratio",
        "de3_entry_lower_wick_ratio", "de3_entry_upper_wick_ratio",
        "de3_entry_range10_atr", "de3_entry_vol1_rel20", "de3_entry_atr14",
        "dist_to_bank_below", "dist_to_bank_above", "dist_to_bank_in_dir",
        "bar_range_pts", "bar_close_pct_body", "sl_dist_pts", "tp_dist_pts",
        "atr_ratio_to_sl",
    ]
    CAT = ["side", "session", "mkt_regime"]
    ds = ds.dropna(subset=NUMERIC + CAT + ["label_wait_better"]).reset_index(drop=True)
    ds = ds.sort_values("entry_time").reset_index(drop=True)
    print(f"[lfo] {len(ds)} rows (sorted by entry_time)")

    # Compute augment features ONCE
    print("[lfo] computing encoder embeddings...")
    emb = compute_encoder_embeddings(ds, ts_col="entry_time")
    print("[lfo] computing cross-market features...")
    cm_df = compute_cross_market_features(ds, ts_col="entry_time")

    num_df = ds[NUMERIC].astype(np.float32).reset_index(drop=True)
    cat_df = _encode_categorical(ds[CAT], CAT).reset_index(drop=True)
    emb_df = pd.DataFrame(emb, columns=[f"enc_{i:02d}" for i in range(emb.shape[1])]).reset_index(drop=True)
    cm_df = cm_df.reset_index(drop=True)

    X_v1 = pd.concat([num_df, cat_df], axis=1).fillna(0.0)
    X_v2 = pd.concat([num_df, cat_df, cm_df, emb_df], axis=1).fillna(0.0)
    y = ds["label_wait_better"].astype(int).values

    print(f"[lfo] v1 feature dim: {X_v1.shape[1]}  v2 feature dim: {X_v2.shape[1]}")

    CLF_KWARGS = dict(n_estimators=250, max_depth=4, learning_rate=0.05,
                      min_samples_leaf=50, random_state=42)
    rows = []
    for t, t_end, tr_end, te_end in _rolling_splits(len(ds)):
        auc_v1 = _fit_predict_auc(X_v1.iloc[:tr_end], y[:tr_end],
                                  X_v1.iloc[tr_end:te_end], y[tr_end:te_end],
                                  clf_kwargs=CLF_KWARGS)
        auc_v2 = _fit_predict_auc(X_v2.iloc[:tr_end], y[:tr_end],
                                  X_v2.iloc[tr_end:te_end], y[tr_end:te_end],
                                  clf_kwargs=CLF_KWARGS)
        test_start = ds["entry_time"].iloc[tr_end]
        test_end_ts = ds["entry_time"].iloc[te_end - 1]
        rows.append({
            "train_frac": t, "test_frac": t_end,
            "test_start": test_start, "test_end": test_end_ts,
            "n_train": tr_end, "n_test": te_end - tr_end,
            "auc_v1": auc_v1, "auc_v2": auc_v2,
            "delta": auc_v2 - auc_v1,
        })
    return "lfo", rows


def ab_kalshi_tp():
    parquet = ARTIFACTS / "kalshi_tp_training_data.parquet"
    ds = pd.read_parquet(parquet)
    NUMERIC = [
        "tp_aligned_prob", "tp_dist_pts", "tp_prob_edge", "tp_vs_entry_prob_delta",
        "nearest_strike_dist", "nearest_strike_oi", "nearest_strike_volume",
        "ladder_slope_near_tp", "minutes_to_settlement", "entry_aligned_prob",
        "atr14_pts", "range_30bar_pts", "trend_20bar_pct", "vel_5bar_pts_per_min",
        "sub_tier", "sub_is_rev", "sub_is_5min",
    ]
    CAT = ["side", "regime"]
    ds = ds.dropna(subset=NUMERIC + CAT).reset_index(drop=True)
    ds = ds.sort_values("ts").reset_index(drop=True)
    print(f"[kalshi_tp] {len(ds)} rows (sorted by ts)")

    print("[kalshi_tp] computing encoder embeddings...")
    emb = compute_encoder_embeddings(ds, ts_col="ts")
    print("[kalshi_tp] computing cross-market features...")
    cm_df = compute_cross_market_features(ds, ts_col="ts")

    num_df = ds[NUMERIC].astype(np.float32).reset_index(drop=True)
    cat_df = _encode_categorical(ds[CAT], CAT).reset_index(drop=True)
    emb_df = pd.DataFrame(emb, columns=[f"enc_{i:02d}" for i in range(emb.shape[1])]).reset_index(drop=True)
    cm_df = cm_df.reset_index(drop=True)

    X_v1 = pd.concat([num_df, cat_df], axis=1).fillna(0.0)
    X_v2 = pd.concat([num_df, cat_df, cm_df, emb_df], axis=1).fillna(0.0)
    y_bin = ds["hit_tp"].astype(int).values
    y_pnl = ds["pnl_dollars"].astype(float).values

    print(f"[kalshi_tp] v1 feature dim: {X_v1.shape[1]}  v2 feature dim: {X_v2.shape[1]}")

    CLF_KWARGS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                      min_samples_leaf=25, random_state=42)
    REG_KWARGS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                      min_samples_leaf=25, random_state=42)
    rows = []
    for t, t_end, tr_end, te_end in _rolling_splits(len(ds)):
        auc_v1 = _fit_predict_auc(X_v1.iloc[:tr_end], y_bin[:tr_end],
                                  X_v1.iloc[tr_end:te_end], y_bin[tr_end:te_end],
                                  clf_kwargs=CLF_KWARGS)
        auc_v2 = _fit_predict_auc(X_v2.iloc[:tr_end], y_bin[:tr_end],
                                  X_v2.iloc[tr_end:te_end], y_bin[tr_end:te_end],
                                  clf_kwargs=CLF_KWARGS)
        # Regressor gate PnL delta
        y_tr_pnl = y_pnl[:tr_end]; y_te_pnl = y_pnl[tr_end:te_end]
        reg_v1 = GradientBoostingRegressor(**REG_KWARGS).fit(X_v1.iloc[:tr_end], y_tr_pnl)
        reg_v2 = GradientBoostingRegressor(**REG_KWARGS).fit(X_v2.iloc[:tr_end], y_tr_pnl)
        pred_v1 = reg_v1.predict(X_v1.iloc[tr_end:te_end])
        pred_v2 = reg_v2.predict(X_v2.iloc[tr_end:te_end])
        rule_pnl = float(y_te_pnl.sum())
        v1_kept_pnl = float(y_te_pnl[pred_v1 > 0].sum())
        v2_kept_pnl = float(y_te_pnl[pred_v2 > 0].sum())
        rows.append({
            "train_frac": t, "test_frac": t_end,
            "test_start": ds["ts"].iloc[tr_end],
            "n_train": tr_end, "n_test": te_end - tr_end,
            "auc_v1": auc_v1, "auc_v2": auc_v2, "delta_auc": auc_v2 - auc_v1,
            "rule_pnl": rule_pnl,
            "v1_pnl_delta": v1_kept_pnl - rule_pnl,
            "v2_pnl_delta": v2_kept_pnl - rule_pnl,
            "pnl_advantage_v2": (v2_kept_pnl - rule_pnl) - (v1_kept_pnl - rule_pnl),
        })
    return "kalshi_tp", rows


def ab_pivot():
    parquet = ARTIFACTS / "pivot_trail_training_data.parquet"
    ds = pd.read_parquet(parquet)
    NUMERIC = [
        "pivot_range_pts", "pivot_body_pts", "upper_wick_pct", "lower_wick_pct",
        "pivot_height_pts", "atr14_pts", "range_30bar_pts", "trend_20bar_pct",
        "dist_to_20bar_hi_pct", "dist_to_20bar_lo_pct", "dist_pivot_to_bank_pts",
        "anchor_distance_from_entry_pts", "vel_5bar_pts_per_min",
        "vel_20bar_pts_per_min", "reading_b_buffer_pts",
    ]
    CAT = ["pivot_type", "session", "tape"]
    ORD = ["et_hour"]
    ds = ds.dropna(subset=NUMERIC + CAT + ORD).reset_index(drop=True)
    # Downsample for tractable encoder pass over 822k rows
    MAX = 150_000
    if len(ds) > MAX:
        held = ds[ds["held"] == 1]
        broke = ds[ds["held"] == 0]
        if len(held) >= MAX // 2:
            broke_sample = broke.sample(n=MAX - len(held), random_state=42)
        else:
            broke_sample = broke.sample(n=MAX - len(held), random_state=42)
        ds = pd.concat([held, broke_sample]).reset_index(drop=True)
    ds = ds.sort_values("ts").reset_index(drop=True)
    print(f"[pivot] {len(ds)} rows (sorted by ts, stratified-sampled)")

    print("[pivot] computing encoder embeddings...")
    emb = compute_encoder_embeddings(ds, ts_col="ts")
    print("[pivot] computing cross-market features...")
    cm_df = compute_cross_market_features(ds, ts_col="ts")

    num_df = ds[NUMERIC + ORD].astype(np.float32).reset_index(drop=True)
    cat_df = _encode_categorical(ds[CAT], CAT).reset_index(drop=True)
    emb_df = pd.DataFrame(emb, columns=[f"enc_{i:02d}" for i in range(emb.shape[1])]).reset_index(drop=True)
    cm_df = cm_df.reset_index(drop=True)

    X_v1 = pd.concat([num_df, cat_df], axis=1).fillna(0.0)
    X_v2 = pd.concat([num_df, cat_df, cm_df, emb_df], axis=1).fillna(0.0)
    y = ds["held"].astype(int).values

    print(f"[pivot] v1 feature dim: {X_v1.shape[1]}  v2 feature dim: {X_v2.shape[1]}")

    CLF_KWARGS = dict(n_estimators=250, max_depth=4, learning_rate=0.05,
                      min_samples_leaf=50, random_state=42)
    rows = []
    for t, t_end, tr_end, te_end in _rolling_splits(len(ds)):
        auc_v1 = _fit_predict_auc(X_v1.iloc[:tr_end], y[:tr_end],
                                  X_v1.iloc[tr_end:te_end], y[tr_end:te_end],
                                  clf_kwargs=CLF_KWARGS)
        auc_v2 = _fit_predict_auc(X_v2.iloc[:tr_end], y[:tr_end],
                                  X_v2.iloc[tr_end:te_end], y[tr_end:te_end],
                                  clf_kwargs=CLF_KWARGS)
        rows.append({
            "train_frac": t, "test_frac": t_end,
            "test_start": ds["ts"].iloc[tr_end],
            "n_train": tr_end, "n_test": te_end - tr_end,
            "auc_v1": auc_v1, "auc_v2": auc_v2, "delta": auc_v2 - auc_v1,
        })
    return "pivot", rows


def print_summary(layer: str, rows: list):
    if not rows:
        print(f"{layer}: no rows"); return
    print()
    print(f"=== {layer} rolling-origin A/B ===")
    if layer == "kalshi_tp":
        hdr = f"{'train %':>8}{'test %':>8}  {'test_start':<22}{'n_test':>7}  " \
              f"{'v1_auc':>8}{'v2_auc':>8}{'Δauc':>8}  " \
              f"{'rule_pnl':>10}{'v1_ΔPnL':>10}{'v2_ΔPnL':>10}{'v2_over_v1':>12}"
        print(hdr); print("-" * len(hdr))
        for r in rows:
            print(f"{r['train_frac']*100:>7.0f}%{r['test_frac']*100:>7.0f}%  "
                  f"{str(r['test_start'])[:19]:<22}{r['n_test']:>7}  "
                  f"{r['auc_v1']:>8.3f}{r['auc_v2']:>8.3f}{r['delta_auc']:>+8.3f}  "
                  f"{r['rule_pnl']:>+10.0f}{r['v1_pnl_delta']:>+10.0f}{r['v2_pnl_delta']:>+10.0f}"
                  f"{r['pnl_advantage_v2']:>+12.0f}")
        print("-" * len(hdr))
        mean_auc_v1 = np.mean([r["auc_v1"] for r in rows])
        mean_auc_v2 = np.mean([r["auc_v2"] for r in rows])
        mean_v1_pnl = np.mean([r["v1_pnl_delta"] for r in rows])
        mean_v2_pnl = np.mean([r["v2_pnl_delta"] for r in rows])
        v2_chunk_wins = sum(1 for r in rows if r["pnl_advantage_v2"] > 0)
        print(f"MEAN   AUC: v1={mean_auc_v1:.3f}  v2={mean_auc_v2:.3f}  Δ={mean_auc_v2-mean_auc_v1:+.3f}")
        print(f"MEAN PnL Δ: v1=${mean_v1_pnl:+.0f}/chunk  v2=${mean_v2_pnl:+.0f}/chunk  "
              f"v2-v1=${mean_v2_pnl-mean_v1_pnl:+.0f}/chunk")
        print(f"v2 beats v1 in PnL: {v2_chunk_wins}/{len(rows)} chunks")
    else:
        hdr = f"{'train %':>8}{'test %':>8}  {'test_start':<22}{'n_test':>7}  " \
              f"{'v1_auc':>8}{'v2_auc':>8}{'Δ':>8}"
        print(hdr); print("-" * len(hdr))
        for r in rows:
            print(f"{r['train_frac']*100:>7.0f}%{r['test_frac']*100:>7.0f}%  "
                  f"{str(r['test_start'])[:19]:<22}{r['n_test']:>7}  "
                  f"{r['auc_v1']:>8.3f}{r['auc_v2']:>8.3f}{r['delta']:>+8.3f}")
        print("-" * len(hdr))
        mean_auc_v1 = np.mean([r["auc_v1"] for r in rows])
        mean_auc_v2 = np.mean([r["auc_v2"] for r in rows])
        v2_chunk_wins = sum(1 for r in rows if r["delta"] > 0)
        print(f"MEAN AUC: v1={mean_auc_v1:.3f}  v2={mean_auc_v2:.3f}  Δ={mean_auc_v2-mean_auc_v1:+.3f}")
        print(f"v2 beats v1 in AUC: {v2_chunk_wins}/{len(rows)} chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", choices=["lfo", "kalshi_tp", "pivot", "all"])
    args = ap.parse_args()

    all_results = {}
    if args.layer in ("lfo", "all"):
        name, rows = ab_lfo(); all_results[name] = rows; print_summary(name, rows)
    if args.layer in ("kalshi_tp", "all"):
        name, rows = ab_kalshi_tp(); all_results[name] = rows; print_summary(name, rows)
    if args.layer in ("pivot", "all"):
        name, rows = ab_pivot(); all_results[name] = rows; print_summary(name, rows)

    # Save results JSON
    import json
    out = ROOT / "artifacts" / "signal_gate_2025" / "rolling_origin_ab_results.json"
    # Convert Timestamps to strings for JSON serialization
    serializable = {}
    for k, rows in all_results.items():
        serializable[k] = [
            {kk: (str(vv) if hasattr(vv, "isoformat") else vv) for kk, vv in r.items()}
            for r in rows
        ]
    out.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
