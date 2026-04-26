#!/usr/bin/env python3
"""AetherFlow feature-skew audit — compare training-window vs live-window.

Goal: diagnose why live gate_prob distribution (mean 0.484 over
2026-04-15→23) is compressed vs training (which achieved coverage≈24% at
threshold 0.51, implying median ~0.52). Three candidate causes:

    a) Market-regime drift (feature distributions changed)
    b) Feature-build pipeline bug (live builder disagrees with training)
    c) Genuine sample noise (only 8 days of live data — too short to judge)

Method: run the same `build_feature_frame` pipeline against two slices
of `ai_loop_data/live_prices.parquet`:
    - TRAINING SLICE: mid-2025 (inside the deploy model's 2025-01→2026-01
      training window, stable liquidity, no Fed surprises)
    - LIVE SLICE: last 14 days (2026-04-09→04-23)
For each of the 84 model feature columns, compute mean/std/median/quartiles,
flag features whose distribution has shifted materially (|Δmean|/σ_train > 1.0
OR variance ratio > 2×). Also run `predict_bundle_probabilities` on both
slices and compare output distributions head-to-head.

Output: stdout report + JSON dump to backtest_reports/aetherflow_feature_skew_audit.json.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aetherflow_features import build_feature_frame        # noqa: E402
from aetherflow_model_bundle import predict_bundle_probabilities  # noqa: E402
from tools.ai_loop import price_context                    # noqa: E402

BUNDLE_PATH = ROOT / "model_aetherflow_deploy_2026oos.pkl"
OUT = ROOT / "backtest_reports" / "aetherflow_feature_skew_audit.json"

# Training window was 2025-01-01 → 2026-01-26 per deploy metadata.
# Pick a stable middle chunk — avoid Jan 2025 warmup, avoid year-end.
TRAIN_SLICE_START = "2025-07-01"
TRAIN_SLICE_END   = "2025-08-31"   # 2 months
# Live window — last 14 bar-days the parquet has
LIVE_SLICE_DAYS = 14

# Flag thresholds
Z_ABS_FLAG = 1.0           # |mean_live - mean_train| / std_train
VAR_RATIO_FLAG = 2.0       # max(var_ratio, 1/var_ratio)

logging.basicConfig(level=logging.WARNING)


def summarise(arr: np.ndarray) -> dict:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "p10": None,
                "p50": None, "p90": None}
    return {
        "n":    int(a.size),
        "mean": float(np.mean(a)),
        "std":  float(np.std(a)),
        "p10":  float(np.percentile(a, 10)),
        "p50":  float(np.median(a)),
        "p90":  float(np.percentile(a, 90)),
    }


def build_features_for_slice(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Build AetherFlow features for bars in [start, end]. Returns the
    feature DataFrame (one row per candidate evaluation)."""
    mask = (prices.index >= pd.Timestamp(start, tz=prices.index.tz)) & \
           (prices.index <= pd.Timestamp(end, tz=prices.index.tz))
    sub = prices.loc[mask]
    if sub.empty:
        return pd.DataFrame()
    # The feature builder wants OHLC-style bars; live_prices has a single
    # 'price' column (ticks). Resample to 1-min bars first.
    bars = pd.DataFrame({
        "open":  sub["price"].resample("1min").first(),
        "high":  sub["price"].resample("1min").max(),
        "low":   sub["price"].resample("1min").min(),
        "close": sub["price"].resample("1min").last(),
        "volume": 0.0,
    }).dropna()
    if len(bars) < 900:
        print(f"  [warn] only {len(bars)} bars in slice — feature builder needs ≥900", flush=True)
        return pd.DataFrame()
    features = build_feature_frame(bars)
    return features


def main() -> int:
    print("[audit] loading bundle + prices...", flush=True)
    with BUNDLE_PATH.open("rb") as fh:
        bundle = pickle.load(fh)
    expected_cols = list(bundle.get("feature_columns", []))
    print(f"  bundle expects {len(expected_cols)} features")

    prices = price_context.load_prices()
    if prices is None:
        print("ERROR: couldn't load live_prices.parquet", file=sys.stderr)
        return 2
    prices = prices.sort_index()
    parquet_end = prices.index.max()
    print(f"  parquet range: {prices.index.min()} → {parquet_end}")

    # TRAIN slice
    print(f"\n[audit] building features for TRAIN slice {TRAIN_SLICE_START} → {TRAIN_SLICE_END}...")
    train_feat = build_features_for_slice(prices, TRAIN_SLICE_START, TRAIN_SLICE_END)
    print(f"  train feature rows: {len(train_feat)}")

    # LIVE slice — last N days up to parquet_end
    live_start = (parquet_end - pd.Timedelta(days=LIVE_SLICE_DAYS)).strftime("%Y-%m-%d")
    live_end = parquet_end.strftime("%Y-%m-%d %H:%M")
    print(f"\n[audit] building features for LIVE slice {live_start} → {live_end}...")
    live_feat = build_features_for_slice(prices, live_start, live_end)
    print(f"  live feature rows: {len(live_feat)}")

    if train_feat.empty or live_feat.empty:
        print("ERROR: one of the slices is empty — cannot audit")
        return 2

    # Model output distribution comparison
    print("\n[audit] running predict_bundle_probabilities on both slices...")
    train_probs = predict_bundle_probabilities(bundle, train_feat)
    live_probs  = predict_bundle_probabilities(bundle, live_feat)
    train_probs = train_probs[np.isfinite(train_probs)]
    live_probs  = live_probs[np.isfinite(live_probs)]

    print("\n════════════ OUTPUT DISTRIBUTION ════════════")
    for label, probs in [("TRAIN  slice", train_probs), ("LIVE   slice", live_probs)]:
        s = summarise(probs)
        print(f"  {label}:  n={s['n']:<5}  mean={s['mean']:.4f}  "
              f"std={s['std']:.4f}  p10={s['p10']:.4f}  "
              f"p50={s['p50']:.4f}  p90={s['p90']:.4f}")
    for thr in (0.51, 0.53, 0.55, 0.555, 0.56, 0.58):
        cov_train = float((train_probs >= thr).mean()) * 100
        cov_live  = float((live_probs  >= thr).mean()) * 100
        gap = cov_train - cov_live
        flag = "  ⚠" if (cov_train > 1.0 and cov_live < cov_train / 2.0) else ""
        print(f"  coverage @ ≥{thr}:  train={cov_train:6.2f}%   live={cov_live:6.2f}%   Δ={gap:+.2f}pp{flag}")

    # Per-feature drift
    print("\n════════════ PER-FEATURE DRIFT (top 15 by |z|) ════════════")
    feat_report = []
    for col in expected_cols:
        if col not in train_feat.columns or col not in live_feat.columns:
            continue
        a = train_feat[col].to_numpy(dtype=float)
        b = live_feat[col].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 50 or b.size < 50:
            continue
        m_a, s_a = float(np.mean(a)), float(np.std(a))
        m_b, s_b = float(np.mean(b)), float(np.std(b))
        z = abs(m_b - m_a) / max(s_a, 1e-9)
        vratio = max(s_b / max(s_a, 1e-9), s_a / max(s_b, 1e-9))
        feat_report.append({
            "feature": col, "train_mean": m_a, "train_std": s_a,
            "live_mean": m_b, "live_std": s_b, "abs_z": z, "var_ratio": vratio,
        })

    feat_report.sort(key=lambda r: -r["abs_z"])
    flagged = []
    print(f"  {'feature':<36} {'train μ':>10} {'live μ':>10}  {'|z|':>6}  {'σ-ratio':>7}")
    print(f"  {'-'*36} {'-'*10} {'-'*10}  {'-'*6}  {'-'*7}")
    for r in feat_report[:15]:
        marker = ""
        if r["abs_z"] > Z_ABS_FLAG:
            marker += " Z!"
        if r["var_ratio"] > VAR_RATIO_FLAG:
            marker += " V!"
        if marker:
            flagged.append(r)
        print(f"  {r['feature']:<36} {r['train_mean']:+10.3f} {r['live_mean']:+10.3f}  "
              f"{r['abs_z']:6.2f}  {r['var_ratio']:7.2f}{marker}")

    n_strong_drift = sum(1 for r in feat_report if r["abs_z"] > Z_ABS_FLAG or r["var_ratio"] > VAR_RATIO_FLAG)
    n_total = len(feat_report)
    print(f"\n  → features with material drift: {n_strong_drift} / {n_total}")

    # Verdict
    print("\n════════════ VERDICT ════════════")
    live_median = float(np.median(live_probs)) if live_probs.size else None
    train_median = float(np.median(train_probs)) if train_probs.size else None
    if train_median is not None and live_median is not None:
        median_gap = train_median - live_median
    else:
        median_gap = None

    if median_gap is not None and median_gap > 0.02 and n_strong_drift >= 5:
        verdict = "REGIME_DRIFT — live features differ meaningfully from training window. Recommend post-hoc calibration (percentile-map) or model retrain on recent data."
    elif median_gap is not None and median_gap > 0.02 and n_strong_drift < 5:
        verdict = "OUTPUT_SHIFT_WITHOUT_FEATURE_DRIFT — features look similar but model output is lower in live. Suspect model is hitting a thin-support region; recommend investigating tail-feature interactions or shipping temperature calibration."
    elif median_gap is not None and abs(median_gap) <= 0.02:
        verdict = "NO_MEANINGFUL_SHIFT — live and train output distributions are within normal variation. The earlier under-firing was almost entirely Bug 1 (over-tight thresholds); Bug 2 may be sample noise."
    else:
        verdict = "INCONCLUSIVE — slices too small to judge."

    print(f"  {verdict}")

    # Write JSON
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "train_slice": {"start": TRAIN_SLICE_START, "end": TRAIN_SLICE_END,
                         "n_rows": int(len(train_feat))},
        "live_slice":  {"start": live_start, "end": live_end,
                         "n_rows": int(len(live_feat))},
        "output_distribution": {
            "train": summarise(train_probs),
            "live":  summarise(live_probs),
        },
        "coverage_at_thresholds": {
            str(thr): {
                "train_pct": float((train_probs >= thr).mean()) * 100,
                "live_pct":  float((live_probs  >= thr).mean()) * 100,
            } for thr in (0.51, 0.53, 0.55, 0.555, 0.56, 0.58)
        },
        "feature_drift": feat_report,
        "n_strong_drift": n_strong_drift,
        "verdict": verdict,
    }, indent=2, default=str))
    print(f"\n[write] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
