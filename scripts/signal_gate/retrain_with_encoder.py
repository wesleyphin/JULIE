"""Retrain existing ML overlay models with bar-encoder embeddings +
cross-market features added to the feature set.

Philosophy: don't re-walk replays. Each model has a training_data.parquet
with all features + labels already persisted. This script:
  1. Loads the existing training-data parquet
  2. For each row, uses entry_time / ts to look up:
       - the 60-bar window from es_master_outrights.parquet and feed
         it through the bar encoder → 32 extra features
       - cross-market features via rl/cross_market.py → 8 extra features
  3. Retrains the same sklearn GBT(s) on the augmented feature set
  4. Writes a new joblib alongside the old one (e.g.
     model_lfo_v2.joblib) WITHOUT overwriting the production model
  5. Prints a side-by-side comparison on the same rolling-origin split

Usage:
  python3 scripts/signal_gate/retrain_with_encoder.py lfo
  python3 scripts/signal_gate/retrain_with_encoder.py kalshi_tp
  python3 scripts/signal_gate/retrain_with_encoder.py pivot

If the v2 validation AUC beats v1 on the held-out split, operator can
manually promote v2 to canonical by copying model_<layer>_v2.joblib →
model_<layer>.joblib.
"""
from __future__ import annotations

import argparse
import math
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

NY = ZoneInfo("America/New_York")
ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"
ES_PARQUET = ROOT / "es_master_outrights.parquet"


# ---------- Bar cache (loaded once) ----------
_BAR_CACHE_DF = None


def _load_bar_cache():
    global _BAR_CACHE_DF
    if _BAR_CACHE_DF is not None:
        return _BAR_CACHE_DF
    print(f"[bar cache] loading {ES_PARQUET}")
    df = pd.read_parquet(ES_PARQUET)
    df = df[df.index.year >= 2024].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(NY)
    else:
        df.index = df.index.tz_convert(NY)
    # Front-month per day (prevents cross-contract price jumps)
    if "symbol" in df.columns and "volume" in df.columns:
        df["_d"] = df.index.date
        dsv = df.groupby(["_d", "symbol"])["volume"].sum().reset_index()
        dsv = dsv.sort_values(["_d", "volume"], ascending=[True, False])
        front = dsv.drop_duplicates("_d", keep="first").set_index("_d")["symbol"]
        df["_f"] = df["_d"].map(front)
        df = df[df["symbol"] == df["_f"]].drop(columns=["_d", "_f"])
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    _BAR_CACHE_DF = df
    print(f"  {len(df):,} bars")
    return df


def compute_encoder_embeddings(ds: pd.DataFrame, ts_col: str = "entry_time") -> np.ndarray:
    """For each row in ds, compute the 32-dim encoder embedding based on
    bars ending at ds[ts_col]. Returns (n, 32) ndarray."""
    from rl.bar_encoder import BarEncoder, encode
    import torch
    ckpt_path = ARTIFACTS / "bar_encoder.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = BarEncoder(seq_len=int(ckpt.get("seq_len", 60)),
                       embed_dim=int(ckpt.get("embed_dim", 32)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    embed_dim = model.embed_dim

    bars_df = _load_bar_cache()
    ts_index = bars_df.index

    # Parse timestamps with UTC normalization first (handles mixed tz input),
    # then convert to NY
    ts_arr = pd.to_datetime(ds[ts_col], utc=True, format="mixed").dt.tz_convert(NY)

    out = np.zeros((len(ds), embed_dim), dtype=np.float32)
    n_failed = 0
    t0 = time.time()
    batch_size = 256
    # Gather the 60-bar windows in chunks for batch inference
    for start in range(0, len(ds), batch_size):
        end = min(len(ds), start + batch_size)
        chunk_ts = ts_arr.iloc[start:end]
        windows = []
        idx_map = []
        for i_rel, ts_i in enumerate(chunk_ts):
            pos = ts_index.searchsorted(ts_i, side="right") - 1
            if pos < model.seq_len - 1 or pos >= len(bars_df):
                n_failed += 1
                continue
            start_bar = pos - model.seq_len + 1
            window = bars_df.iloc[start_bar: pos + 1][
                ["open", "high", "low", "close", "volume"]
            ].to_numpy(dtype=np.float64)
            if len(window) != model.seq_len:
                n_failed += 1
                continue
            from rl.bar_encoder import _normalize_bar_window
            windows.append(_normalize_bar_window(window))
            idx_map.append(start + i_rel)
        if not windows:
            continue
        x = torch.from_numpy(np.stack(windows).astype(np.float32))
        with torch.no_grad():
            emb = model.encode(x).numpy()
        for k, row_idx in enumerate(idx_map):
            out[row_idx] = emb[k]
        if (start // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            done = end
            rate = done / max(elapsed, 0.001)
            eta = (len(ds) - done) / max(rate, 0.001)
            print(f"  [encoder] {done}/{len(ds)}  {rate:.0f} rows/sec  eta={eta:.0f}s", flush=True)
    print(f"  [encoder] computed {len(ds) - n_failed}/{len(ds)} "
          f"({n_failed} rows had insufficient bar history)")
    return out


def compute_cross_market_features(ds: pd.DataFrame, ts_col: str = "entry_time") -> pd.DataFrame:
    """For each row, fetch the 8 cross-market features. Returns a DataFrame
    whose columns are CROSS_MARKET_FEATURE_KEYS."""
    from rl.cross_market import CrossMarketFeatures, CROSS_MARKET_FEATURE_KEYS
    cm = CrossMarketFeatures()
    bars_df = _load_bar_cache()
    ts_arr = pd.to_datetime(ds[ts_col], utc=True, format="mixed").dt.tz_convert(NY)
    feats_per_row = []
    for ts_i in ts_arr:
        feats = cm.extract_at(ts_i, mes_bars=bars_df)
        feats_per_row.append([feats[k] for k in CROSS_MARKET_FEATURE_KEYS])
    return pd.DataFrame(feats_per_row, columns=list(CROSS_MARKET_FEATURE_KEYS),
                        index=ds.index).astype(np.float32)


# ---------- Per-layer retrain drivers ----------

def retrain_lfo():
    parquet = ARTIFACTS / "lfo_training_data.parquet"
    ds = pd.read_parquet(parquet)
    print(f"[lfo] {len(ds)} training rows")

    # Existing schema
    NUMERIC = [
        "de3_entry_ret1_atr", "de3_entry_body_pos1", "de3_entry_body1_ratio",
        "de3_entry_lower_wick_ratio", "de3_entry_upper_wick_ratio",
        "de3_entry_range10_atr", "de3_entry_vol1_rel20", "de3_entry_atr14",
        "dist_to_bank_below", "dist_to_bank_above", "dist_to_bank_in_dir",
        "bar_range_pts", "bar_close_pct_body", "sl_dist_pts", "tp_dist_pts",
        "atr_ratio_to_sl",
    ]
    CATEGORICAL = ["side", "session", "mkt_regime"]
    # Filter rows with complete features
    ds = ds.dropna(subset=NUMERIC + CATEGORICAL + ["label_wait_better"]).reset_index(drop=True)
    y = ds["label_wait_better"].astype(int).values

    # Compute encoder embeddings + cross-market features
    print("[lfo] computing encoder embeddings...")
    emb = compute_encoder_embeddings(ds, ts_col="entry_time")
    print("[lfo] computing cross-market features...")
    cm_df = compute_cross_market_features(ds, ts_col="entry_time")

    # Build feature matrix
    parts = [ds[NUMERIC].copy().astype(np.float32)]
    for col in CATEGORICAL:
        known = sorted(ds[col].astype(str).unique().tolist())
        enc = pd.DataFrame(
            {f"{col}__{v}": (ds[col].astype(str) == v).astype(int) for v in known},
            index=ds.index,
        )
        parts.append(enc)
    parts.append(cm_df)
    parts.append(pd.DataFrame(emb, columns=[f"enc_{i:02d}" for i in range(emb.shape[1])],
                              index=ds.index).astype(np.float32))
    X = pd.concat(parts, axis=1).fillna(0.0)
    print(f"[lfo] feature matrix: {X.shape}")

    # Temporal split 85/15
    ds_sorted_idx = ds["entry_time"].astype(str).argsort().values
    split = int(0.85 * len(ds))
    tr_idx = ds_sorted_idx[:split]; te_idx = ds_sorted_idx[split:]
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    clf = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
    auc_v2 = float(roc_auc_score(y_te, p_te))

    # Load v1 model and score on same split for comparison
    v1_path = ARTIFACTS / "model_lfo.joblib"
    v1 = joblib.load(v1_path)
    # v1's feature schema is a subset — rebuild X using only v1's features
    v1_features = v1["feature_names"]
    X_te_v1 = pd.DataFrame(0.0, index=X_te.index, columns=v1_features)
    for col in v1_features:
        if col in X_te.columns:
            X_te_v1[col] = X_te[col].values
    p_te_v1 = v1["model"].predict_proba(X_te_v1)[:, list(v1["model"].classes_).index(1)]
    auc_v1 = float(roc_auc_score(y_te, p_te_v1))

    print(f"\n[lfo] holdout AUC: v1={auc_v1:.4f}  v2={auc_v2:.4f}  Δ={auc_v2-auc_v1:+.4f}")

    # Save v2 alongside v1
    out_path = ARTIFACTS / "model_lfo_v2.joblib"
    joblib.dump({
        "model": clf,
        "model_kind": "GBT_d4_lfo_v2_with_encoder_crossmarket",
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "uses_bar_encoder": True,
        "uses_cross_market": True,
        "encoder_embed_dim": emb.shape[1],
        "cross_market_keys": list(cm_df.columns),
        "veto_threshold": 0.40,
        "auc_v1": auc_v1,
        "auc_v2": auc_v2,
        "auc_delta": auc_v2 - auc_v1,
    }, out_path)
    print(f"[write] {out_path}")
    return auc_v1, auc_v2


def retrain_kalshi_tp():
    parquet = ARTIFACTS / "kalshi_tp_training_data.parquet"
    ds = pd.read_parquet(parquet)
    print(f"[kalshi_tp] {len(ds)} training rows")

    NUMERIC = [
        "tp_aligned_prob", "tp_dist_pts", "tp_prob_edge", "tp_vs_entry_prob_delta",
        "nearest_strike_dist", "nearest_strike_oi", "nearest_strike_volume",
        "ladder_slope_near_tp", "minutes_to_settlement", "entry_aligned_prob",
        "atr14_pts", "range_30bar_pts", "trend_20bar_pct", "vel_5bar_pts_per_min",
        "sub_tier", "sub_is_rev", "sub_is_5min",
    ]
    CATEGORICAL = ["side", "regime"]
    ds = ds.dropna(subset=NUMERIC + CATEGORICAL).reset_index(drop=True)
    y_bin = ds["hit_tp"].astype(int).values
    y_pnl = ds["pnl_dollars"].astype(float).values

    print("[kalshi_tp] computing encoder embeddings...")
    emb = compute_encoder_embeddings(ds, ts_col="ts")
    print("[kalshi_tp] computing cross-market features...")
    cm_df = compute_cross_market_features(ds, ts_col="ts")

    parts = [ds[NUMERIC].copy().astype(np.float32)]
    for col in CATEGORICAL:
        known = sorted(ds[col].astype(str).unique().tolist())
        enc = pd.DataFrame(
            {f"{col}__{v}": (ds[col].astype(str) == v).astype(int) for v in known},
            index=ds.index,
        )
        parts.append(enc)
    parts.append(cm_df)
    parts.append(pd.DataFrame(emb, columns=[f"enc_{i:02d}" for i in range(emb.shape[1])],
                              index=ds.index).astype(np.float32))
    X = pd.concat(parts, axis=1).fillna(0.0)
    print(f"[kalshi_tp] feature matrix: {X.shape}")

    # Temporal split
    ds_sorted_idx = ds["ts"].astype(str).argsort().values
    split = int(0.85 * len(ds))
    tr_idx = ds_sorted_idx[:split]; te_idx = ds_sorted_idx[split:]
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr_bin, y_te_bin = y_bin[tr_idx], y_bin[te_idx]
    y_tr_pnl, y_te_pnl = y_pnl[tr_idx], y_pnl[te_idx]

    # Classifier
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=25, random_state=42,
    )
    clf.fit(X_tr, y_tr_bin)
    p_te = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
    auc_v2 = float(roc_auc_score(y_te_bin, p_te))

    # Regressor
    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        min_samples_leaf=25, random_state=42,
    )
    reg.fit(X_tr, y_tr_pnl)
    pred_pnl = reg.predict(X_te)
    # Evaluate regressor gate at > $0 vs v1
    rule_pnl = float(y_te_pnl.sum())
    v2_kept_pnl = float(y_te_pnl[pred_pnl > 0].sum())
    v2_delta = v2_kept_pnl - rule_pnl

    # v1 comparison
    v1_path = ARTIFACTS / "model_kalshi_tp_gate.joblib"
    v1 = joblib.load(v1_path)
    v1_features = v1["feature_names"]
    X_te_v1 = pd.DataFrame(0.0, index=X_te.index, columns=v1_features)
    for col in v1_features:
        if col in X_te.columns:
            X_te_v1[col] = X_te[col].values
    p_te_v1 = v1["classifier"].predict_proba(X_te_v1)[:, list(v1["classifier"].classes_).index(1)]
    auc_v1 = float(roc_auc_score(y_te_bin, p_te_v1))
    pred_v1_pnl = v1["regressor"].predict(X_te_v1)
    v1_kept_pnl = float(y_te_pnl[pred_v1_pnl > 0].sum())
    v1_delta = v1_kept_pnl - rule_pnl

    print(f"\n[kalshi_tp] holdout AUC: v1={auc_v1:.4f}  v2={auc_v2:.4f}  Δ={auc_v2-auc_v1:+.4f}")
    print(f"[kalshi_tp] regressor gate PnL delta vs rule: v1=${v1_delta:+.0f}  v2=${v2_delta:+.0f}")

    out_path = ARTIFACTS / "model_kalshi_tp_gate_v2.joblib"
    joblib.dump({
        "classifier": clf,
        "regressor": reg,
        "model_kind": "GBT_kalshi_tp_v2_with_encoder_crossmarket",
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "uses_bar_encoder": True,
        "uses_cross_market": True,
        "encoder_embed_dim": emb.shape[1],
        "cross_market_keys": list(cm_df.columns),
        "gate_mode": "regressor_pnl",
        "regressor_gate_threshold": 0.0,
        "classifier_threshold": 0.50,
        "auc_v1": auc_v1, "auc_v2": auc_v2,
        "v1_delta_pnl": v1_delta, "v2_delta_pnl": v2_delta,
    }, out_path)
    print(f"[write] {out_path}")
    return auc_v1, auc_v2, v1_delta, v2_delta


def retrain_pivot():
    parquet = ARTIFACTS / "pivot_trail_training_data.parquet"
    ds = pd.read_parquet(parquet)
    # Downsample to make encoder inference tractable (the trainer does stratified sampling)
    MAX = 200_000
    if len(ds) > MAX:
        # Stratified: keep the full held class + downsample the broke class
        held = ds[ds["held"] == 1]
        broke = ds[ds["held"] == 0].sample(n=MAX - len(held), random_state=42) if len(held) < MAX else ds.sample(n=MAX, random_state=42)
        ds = pd.concat([held, broke]).reset_index(drop=True)
    print(f"[pivot] {len(ds)} training rows after stratified downsample")

    NUMERIC = [
        "pivot_range_pts", "pivot_body_pts", "upper_wick_pct", "lower_wick_pct",
        "pivot_height_pts", "atr14_pts", "range_30bar_pts", "trend_20bar_pct",
        "dist_to_20bar_hi_pct", "dist_to_20bar_lo_pct", "dist_pivot_to_bank_pts",
        "anchor_distance_from_entry_pts", "vel_5bar_pts_per_min",
        "vel_20bar_pts_per_min", "reading_b_buffer_pts",
    ]
    CATEGORICAL = ["pivot_type", "session", "tape"]
    ORDINAL = ["et_hour"]
    ds = ds.dropna(subset=NUMERIC + CATEGORICAL + ORDINAL).reset_index(drop=True)
    y = ds["held"].astype(int).values

    print("[pivot] computing encoder embeddings...")
    emb = compute_encoder_embeddings(ds, ts_col="ts")
    print("[pivot] computing cross-market features...")
    cm_df = compute_cross_market_features(ds, ts_col="ts")

    parts = [ds[NUMERIC + ORDINAL].copy().astype(np.float32)]
    for col in CATEGORICAL:
        known = sorted(ds[col].astype(str).unique().tolist())
        enc = pd.DataFrame(
            {f"{col}__{v}": (ds[col].astype(str) == v).astype(int) for v in known},
            index=ds.index,
        )
        parts.append(enc)
    parts.append(cm_df)
    parts.append(pd.DataFrame(emb, columns=[f"enc_{i:02d}" for i in range(emb.shape[1])],
                              index=ds.index).astype(np.float32))
    X = pd.concat(parts, axis=1).fillna(0.0)
    print(f"[pivot] feature matrix: {X.shape}")

    # Temporal split
    ds_sorted_idx = ds["ts"].astype(str).argsort().values
    split = int(0.85 * len(ds))
    tr_idx = ds_sorted_idx[:split]; te_idx = ds_sorted_idx[split:]
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    clf = GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        min_samples_leaf=50, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, list(clf.classes_).index(1)]
    auc_v2 = float(roc_auc_score(y_te, p_te))

    # v1 comparison
    v1_path = ARTIFACTS / "model_pivot_trail.joblib"
    v1 = joblib.load(v1_path)
    v1_features = v1["feature_names"]
    X_te_v1 = pd.DataFrame(0.0, index=X_te.index, columns=v1_features)
    for col in v1_features:
        if col in X_te.columns:
            X_te_v1[col] = X_te[col].values
    p_te_v1 = v1["model"].predict_proba(X_te_v1)[:, list(v1["model"].classes_).index(1)]
    auc_v1 = float(roc_auc_score(y_te, p_te_v1))

    print(f"\n[pivot] holdout AUC: v1={auc_v1:.4f}  v2={auc_v2:.4f}  Δ={auc_v2-auc_v1:+.4f}")

    out_path = ARTIFACTS / "model_pivot_trail_v2.joblib"
    joblib.dump({
        "model": clf,
        "model_kind": "GBT_pivot_trail_v2_with_encoder_crossmarket",
        "feature_names": list(X.columns),
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "ordinal_features": ORDINAL,
        "uses_bar_encoder": True,
        "uses_cross_market": True,
        "encoder_embed_dim": emb.shape[1],
        "cross_market_keys": list(cm_df.columns),
        "hold_threshold": 0.55,
        "auc_v1": auc_v1, "auc_v2": auc_v2,
    }, out_path)
    print(f"[write] {out_path}")
    return auc_v1, auc_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", choices=["lfo", "kalshi_tp", "pivot", "all"])
    args = ap.parse_args()

    results = {}
    if args.layer in ("lfo", "all"):
        try:
            results["lfo"] = retrain_lfo()
        except Exception as exc:
            print(f"[lfo] FAILED: {exc}")
            import traceback; traceback.print_exc()
    if args.layer in ("kalshi_tp", "all"):
        try:
            results["kalshi_tp"] = retrain_kalshi_tp()
        except Exception as exc:
            print(f"[kalshi_tp] FAILED: {exc}")
            import traceback; traceback.print_exc()
    if args.layer in ("pivot", "all"):
        try:
            results["pivot"] = retrain_pivot()
        except Exception as exc:
            print(f"[pivot] FAILED: {exc}")
            import traceback; traceback.print_exc()

    print("\n=== Final summary ===")
    for k, v in results.items():
        if isinstance(v, tuple):
            print(f"  {k}: v1_auc={v[0]:.4f}  v2_auc={v[1]:.4f}"
                  + (f"  v1_pnl={v[2]:+.0f}  v2_pnl={v[3]:+.0f}" if len(v) == 4 else ""))


if __name__ == "__main__":
    main()
