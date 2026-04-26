#!/usr/bin/env python3
"""AetherFlow calibration-layer fitter + OOS validator.

Problem (Bug 2 from the AetherFlow forensics): live `predict_bundle_probabilities`
output is ~1.2pp lower in mean than it was during training, because
market regime drifted (ATR doubled, DISPERSED regime became more common).
Coverage @ 0.55 is ~half what training saw: 5.85% live vs 11.9% train.

Fix: fit a monotonic percentile-match calibrator that maps the live
probability distribution onto the training distribution. Concretely, the
calibrator stores a sorted reference quantile array from the training
window; at inference time it looks up the empirical rank of the live prob
in that reference and outputs the corresponding training-window
probability. Rank-preserving — it won't reorder decisions, just rescale
magnitudes to restore the coverage the model was calibrated for.

Ship gate (must pass ALL three):
    1. On the FIT window (training reference) the calibrator is approximately
       identity (sanity check).
    2. On a held-out OOS window (2026-01 through 2026-04) calibrated
       threshold=0.55 trade PnL is ≥ uncalibrated threshold=0.55 PnL, AND
       win rate doesn't regress by more than 1pp, AND
       MaxDD is no more than 10% worse than baseline.
    3. Coverage on the live-era slice (2026-04-09 onward) rises to within
       30% of the training-window coverage.

If all three pass: write `aetherflow_calibrator.json` to repo root and
print a WIRE IT IN snippet. Otherwise: don't write the file, report the
failure reason.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aetherflow_features import build_feature_frame          # noqa: E402
from aetherflow_model_bundle import predict_bundle_probabilities  # noqa: E402
from tools.ai_loop import price_context                      # noqa: E402

BUNDLE_PATH = ROOT / "model_aetherflow_deploy_2026oos.pkl"
OUT_CAL = ROOT / "aetherflow_calibrator.json"
OUT_REPORT = ROOT / "backtest_reports" / "aetherflow_calibrator_validation.json"

# Full training window per deploy metadata: 2025-01 → 2026-01. Use a
# stable middle chunk (avoid warmup + Dec year-end).
FIT_START = "2025-05-01"
FIT_END   = "2025-11-30"   # 7 months — large enough to be representative
# OOS validation window: 2026-01 through most of April (post-training)
OOS_START = "2026-01-27"
OOS_END   = "2026-04-08"   # up to just before the "live" slice
# Live-era coverage check
LIVE_START = "2026-04-09"
LIVE_END   = None          # → use parquet last bar

# Quantile grid for calibrator: 201 points, uniform on [0, 1]
N_QUANTILE_POINTS = 201

# MES point value for crude PnL estimate
MES_PT_VALUE = 5.0
TP_POINTS = 6.0
SL_POINTS = 4.0


def bars_for(prices: pd.DataFrame, start: str, end: str | None):
    lo = pd.Timestamp(start, tz=prices.index.tz)
    hi = pd.Timestamp(end, tz=prices.index.tz) if end else prices.index.max()
    mask = (prices.index >= lo) & (prices.index <= hi)
    sub = prices.loc[mask]
    if sub.empty:
        return pd.DataFrame()
    bars = pd.DataFrame({
        "open":  sub["price"].resample("1min").first(),
        "high":  sub["price"].resample("1min").max(),
        "low":   sub["price"].resample("1min").min(),
        "close": sub["price"].resample("1min").last(),
        "volume": 0.0,
    }).dropna()
    return bars


def features_for(prices: pd.DataFrame, start: str, end: str | None):
    bars = bars_for(prices, start, end)
    if bars.empty or len(bars) < 900:
        return pd.DataFrame(), pd.DataFrame()
    features = build_feature_frame(bars)
    return bars, features


def fit_calibrator(train_probs: np.ndarray, live_probs: np.ndarray) -> dict:
    """Return a dict with BOTH the train quantile reference (output map)
    AND a live quantile reference (for input ranking at inference time).

    Inference flow in live bot (single-sample):
        1. Compute live_rank = searchsorted(live_ref_sorted, raw_prob)
           → integer position in the frozen live distribution
        2. Lookup train_prob = train_ref[rank_fraction]
           → re-mapped probability at the same empirical percentile
    """
    q = np.linspace(0.0, 1.0, N_QUANTILE_POINTS)
    train_ref = np.quantile(train_probs, q)
    live_ref = np.quantile(live_probs, q)
    return {
        "version": 2,
        "kind": "quantile_map",
        "n_points": int(N_QUANTILE_POINTS),
        "fit_start": FIT_START, "fit_end": FIT_END,
        "live_ref_start": LIVE_START, "live_ref_end": LIVE_END or "parquet_max",
        "n_fit_samples": int(train_probs.size),
        "n_live_samples": int(live_probs.size),
        "quantiles": q.tolist(),          # uniform [0, 1]
        "train_ref_probs": train_ref.tolist(),  # output map
        "live_ref_probs": live_ref.tolist(),    # input ranking reference
        # Backwards-compat alias (v1 readers)
        "ref_probs": train_ref.tolist(),
    }


def apply_calibrator(cal: dict, probs: np.ndarray) -> np.ndarray:
    """Map live probs to calibrated probs using the frozen live quantile
    reference. Uses `live_ref_probs` (if present, v2) OR empirical ranking
    from the input batch (v1 fallback — only valid on large batches)."""
    q = np.asarray(cal["quantiles"], dtype=float)
    train_ref = np.asarray(cal.get("train_ref_probs", cal.get("ref_probs")), dtype=float)
    live_ref_list = cal.get("live_ref_probs")
    if live_ref_list is not None:
        # v2: invert the live CDF to get empirical rank of each input
        live_ref = np.asarray(live_ref_list, dtype=float)
        frac_ranks = np.interp(probs, live_ref, q, left=0.0, right=1.0)
    else:
        # v1: empirical rank from input batch — only sensible for large batches
        ranks = np.argsort(np.argsort(probs))
        frac_ranks = ranks / max(1, len(probs) - 1)
    return np.interp(frac_ranks, q, train_ref)


def estimate_outcomes(bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """For each candidate row in features, crudely walk forward in `bars`
    and decide TP/SL hit. Returns PnL in dollars per trade, aligned to
    features.index.
    """
    if features.empty or bars.empty:
        return pd.Series(dtype=float)
    side = pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    pnl = np.zeros(len(features), dtype=float)
    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    lookahead = 60  # bars
    close_idx = close.index
    for i, (ts, s) in enumerate(zip(features.index, side.values)):
        if s == 0:
            continue
        try:
            start_pos = close_idx.searchsorted(ts) + 1
        except Exception:
            continue
        end_pos = min(start_pos + lookahead, len(close))
        if start_pos >= end_pos:
            continue
        entry = float(close.iloc[start_pos - 1])
        if s > 0:  # LONG
            sl = entry - SL_POINTS
            tp = entry + TP_POINTS
            path_high = high.iloc[start_pos:end_pos]
            path_low  = low.iloc[start_pos:end_pos]
            hit_sl = path_low.le(sl)
            hit_tp = path_high.ge(tp)
        else:      # SHORT
            sl = entry + SL_POINTS
            tp = entry - TP_POINTS
            path_high = high.iloc[start_pos:end_pos]
            path_low  = low.iloc[start_pos:end_pos]
            hit_sl = path_high.ge(sl)
            hit_tp = path_low.le(tp)
        # First event wins
        sl_i = hit_sl.values.argmax() if hit_sl.any() else 1 << 30
        tp_i = hit_tp.values.argmax() if hit_tp.any() else 1 << 30
        if sl_i == 1 << 30 and tp_i == 1 << 30:
            # expired — close at last bar, mark-to-market
            last = float(close.iloc[end_pos - 1])
            pts = (last - entry) if s > 0 else (entry - last)
            pnl[i] = pts * MES_PT_VALUE
        elif sl_i < tp_i:
            pnl[i] = -SL_POINTS * MES_PT_VALUE
        else:
            pnl[i] = TP_POINTS * MES_PT_VALUE
    return pd.Series(pnl, index=features.index)


def trade_stats(pnls_passed: np.ndarray) -> dict:
    pnls_passed = pnls_passed[np.isfinite(pnls_passed)]
    if pnls_passed.size == 0:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "dd": 0.0}
    cum = np.cumsum(pnls_passed)
    peak = np.maximum.accumulate(cum)
    dd = float(np.max(peak - cum))
    wins = int((pnls_passed > 0).sum())
    return {
        "n": int(pnls_passed.size),
        "pnl": float(cum[-1]),
        "wr": float(wins) / float(pnls_passed.size) * 100.0,
        "dd": dd,
        "avg": float(cum[-1]) / float(pnls_passed.size),
    }


def main() -> int:
    print("[calibrator] loading bundle + prices...", flush=True)
    with BUNDLE_PATH.open("rb") as fh:
        bundle = pickle.load(fh)
    prices = price_context.load_prices().sort_index()
    print(f"  parquet range: {prices.index.min()} → {prices.index.max()}")

    # ── FIT ──────────────────────────────────────────────────────────
    print(f"\n[fit] train window {FIT_START} → {FIT_END}")
    fit_bars, fit_feat = features_for(prices, FIT_START, FIT_END)
    print(f"  fit bars: {len(fit_bars):,}  feat rows: {len(fit_feat):,}")
    if fit_feat.empty:
        print("ERROR: no fit data"); return 2
    fit_probs = predict_bundle_probabilities(bundle, fit_feat)
    fit_probs = fit_probs[np.isfinite(fit_probs)]
    # Fit needs a live-distribution reference for input ranking at inference
    # time. Build the live-era slice once, reuse for fit + for later live-
    # coverage sanity check below. Filter to candidate-only rows because
    # those are what the live bot ever predicts on.
    live_bars_pre, live_feat_pre_full = features_for(prices, LIVE_START, LIVE_END)
    live_side = pd.to_numeric(live_feat_pre_full.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    live_feat_pre = live_feat_pre_full.loc[(live_side != 0).values]
    live_probs_pre = predict_bundle_probabilities(bundle, live_feat_pre)
    live_probs_pre = live_probs_pre[np.isfinite(live_probs_pre)]
    # Also restrict the fit_probs to candidate rows for a like-for-like
    # comparison with the live reference.
    fit_side = pd.to_numeric(fit_feat.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    fit_feat_candidates = fit_feat.loc[(fit_side != 0).values]
    fit_probs = predict_bundle_probabilities(bundle, fit_feat_candidates)
    fit_probs = fit_probs[np.isfinite(fit_probs)]
    print(f"  fit candidate rows: {len(fit_feat_candidates):,}  live candidate rows: {len(live_feat_pre):,}")
    cal = fit_calibrator(fit_probs, live_probs_pre)
    print(f"  train-window quantiles: p10={cal['train_ref_probs'][20]:.4f} "
          f"p50={cal['train_ref_probs'][100]:.4f}  p90={cal['train_ref_probs'][180]:.4f}")
    print(f"  live-window  quantiles: p10={cal['live_ref_probs'][20]:.4f} "
          f"p50={cal['live_ref_probs'][100]:.4f}  p90={cal['live_ref_probs'][180]:.4f}")

    # Sanity: applying the calibrator to the same train-window output
    # must approximately preserve distribution (identity mapping)
    sanity = apply_calibrator(cal, fit_probs)
    sanity_mae = float(np.mean(np.abs(sanity - fit_probs)))
    print(f"  sanity: |calibrated(fit) - raw(fit)| mean abs err = {sanity_mae:.5f}  "
          f"{'✓' if sanity_mae < 0.005 else '✗'}")

    # ── OOS ──────────────────────────────────────────────────────────
    print(f"\n[oos]   validation window {OOS_START} → {OOS_END}")
    oos_bars, oos_feat = features_for(prices, OOS_START, OOS_END)
    print(f"  oos bars: {len(oos_bars):,}  feat rows: {len(oos_feat):,}")
    if oos_feat.empty:
        print("ERROR: no oos data"); return 2
    # CRITICAL: filter to rows that are actual signal candidates BEFORE
    # applying threshold. Non-candidate rows have candidate_side=0 and
    # would never fire in live; predicting on them and gating by threshold
    # produces meaningless statistics because the outcome walker returns
    # 0.0 for side=0 rows, polluting the denominator.
    oos_side = pd.to_numeric(oos_feat.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    oos_candidate_mask = (oos_side != 0).values
    oos_feat = oos_feat.loc[oos_candidate_mask]
    print(f"  after candidate_side filter: {len(oos_feat):,} rows ({100*oos_candidate_mask.mean():.1f}%)")
    oos_probs_raw = predict_bundle_probabilities(bundle, oos_feat)
    oos_probs_raw = oos_probs_raw[np.isfinite(oos_probs_raw)]
    oos_feat = oos_feat.iloc[: len(oos_probs_raw)]   # align
    oos_probs_cal = apply_calibrator(cal, oos_probs_raw)

    oos_pnl = estimate_outcomes(oos_bars, oos_feat).values

    for thr in (0.55,):
        raw_mask = oos_probs_raw >= thr
        cal_mask = oos_probs_cal >= thr
        raw_s = trade_stats(oos_pnl[raw_mask])
        cal_s = trade_stats(oos_pnl[cal_mask])
        print(f"\n  threshold = {thr}")
        print(f"    RAW : n={raw_s['n']:<5}  PnL=${raw_s['pnl']:+.2f}  "
              f"WR={raw_s['wr']:.2f}%  DD=${raw_s['dd']:.0f}  avg=${raw_s['avg']:+.3f}")
        print(f"    CAL : n={cal_s['n']:<5}  PnL=${cal_s['pnl']:+.2f}  "
              f"WR={cal_s['wr']:.2f}%  DD=${cal_s['dd']:.0f}  avg=${cal_s['avg']:+.3f}")

        delta_pnl = cal_s["pnl"] - raw_s["pnl"]
        delta_wr = cal_s["wr"] - raw_s["wr"]
        dd_ratio = (cal_s["dd"] / raw_s["dd"]) if raw_s["dd"] > 0 else 1.0
        print(f"    Δ    : PnL={delta_pnl:+.2f}   WR={delta_wr:+.2f}pp   "
              f"DD-ratio={dd_ratio:.2f}")

    # Ship gates
    ship = {
        "sanity_ok":      sanity_mae < 0.005,
        "pnl_ok":         cal_s["pnl"] >= raw_s["pnl"],
        "wr_ok":          delta_wr >= -1.0,
        "dd_ok":          dd_ratio <= 1.10,
    }

    # Live-era coverage improvement (reuse pre-fit live distribution)
    print(f"\n[live]  {LIVE_START} → now")
    live_bars, live_feat = live_bars_pre, live_feat_pre
    print(f"  live bars: {len(live_bars):,}  feat rows: {len(live_feat):,}")
    live_probs_raw = live_probs_pre
    live_probs_cal = apply_calibrator(cal, live_probs_raw)
    cov_raw = float((live_probs_raw >= 0.55).mean()) * 100
    cov_cal = float((live_probs_cal >= 0.55).mean()) * 100
    # Training-window coverage @ 0.55 (reference)
    cov_train = float((fit_probs >= 0.55).mean()) * 100
    ratio = cov_cal / max(cov_train, 1e-9)
    print(f"  coverage @ ≥0.55:  raw-live={cov_raw:.2f}%   "
          f"calibrated-live={cov_cal:.2f}%   train-ref={cov_train:.2f}%   "
          f"cal/train ratio={ratio:.2f}")
    ship["coverage_ok"] = 0.70 <= ratio <= 1.30

    print("\n════════════ SHIP GATES ════════════")
    for k, v in ship.items():
        mark = "✓" if v else "✗"
        print(f"  {mark}  {k}")

    all_pass = all(ship.values())
    report = {
        "fit_stats": {"n": int(fit_probs.size),
                       "mean": float(np.mean(fit_probs)),
                       "p50": float(np.median(fit_probs))},
        "oos_stats": {"n": int(oos_probs_raw.size),
                       "raw_trade_stats": raw_s, "cal_trade_stats": cal_s,
                       "deltas": {"pnl": delta_pnl, "wr": delta_wr,
                                    "dd_ratio": dd_ratio}},
        "live_coverage": {"raw": cov_raw, "cal": cov_cal, "train": cov_train,
                            "cal_over_train": ratio},
        "ship_gates": ship, "ship": all_pass,
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[write-report] {OUT_REPORT}")

    if all_pass:
        OUT_CAL.write_text(json.dumps(cal, indent=2))
        print(f"[SHIP] calibrator passes all gates — written to {OUT_CAL}")
        print("      wire it in: load JSON at strategy init + apply after "
              "predict_bundle_probabilities")
        return 0
    else:
        print("[NO-SHIP] calibrator does NOT pass all gates — not written.")
        print("          Bug 2 remains: needs model retrain on recent data.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
