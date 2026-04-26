#!/usr/bin/env python3
"""Compare a retrained AetherFlow bundle against the currently-deployed one.

Runs both models through the same OOS slice (2026-01-27 → 2026-04-08,
post-deploy-training, pre-live-forensics) and the same live slice
(2026-04-09 → parquet_end). Reports PnL/WR/DD/coverage/avg-prob for both.

Ship gate for the retrained bundle (must pass ALL):
    - OOS PnL ≥ deployed OOS PnL (or deployed OOS PnL is <= 0 and retrained is >)
    - OOS WR regression ≤ 1pp
    - OOS MaxDD ≤ 110% of deployed MaxDD
    - Live coverage @ threshold is within 50–150% of the retrained model's
      TRAIN coverage (i.e. distribution match is restored, not a coverage collapse)

Usage:
    python3 scripts/aetherflow_compare_retrained_vs_deployed.py \
        --retrained model_aetherflow_deploy_2026full.pkl \
        --retrained-threshold aetherflow_thresholds_deploy_2026full.json
"""
from __future__ import annotations

import argparse
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

DEPLOY_PKL = ROOT / "model_aetherflow_deploy_2026oos.pkl"
DEPLOY_THR = ROOT / "aetherflow_thresholds_deploy_2026oos.json"
OUT = ROOT / "backtest_reports" / "aetherflow_retrained_vs_deployed.json"

# Same windows as calibrator-validation, for like-for-like comparison
TRAIN_SLICE_START = "2025-07-01"
TRAIN_SLICE_END   = "2025-08-31"
OOS_START = "2026-01-27"
OOS_END   = "2026-04-08"
LIVE_START = "2026-04-09"

TP_POINTS = 6.0
SL_POINTS = 4.0
MES_PT_VALUE = 5.0


def bars_for(prices: pd.DataFrame, start: str, end: str | None):
    lo = pd.Timestamp(start, tz=prices.index.tz)
    hi = pd.Timestamp(end, tz=prices.index.tz) if end else prices.index.max()
    sub = prices.loc[(prices.index >= lo) & (prices.index <= hi)]
    if sub.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "open":  sub["price"].resample("1min").first(),
        "high":  sub["price"].resample("1min").max(),
        "low":   sub["price"].resample("1min").min(),
        "close": sub["price"].resample("1min").last(),
        "volume": 0.0,
    }).dropna()


def features_for(prices, start, end):
    bars = bars_for(prices, start, end)
    if bars.empty or len(bars) < 900:
        return pd.DataFrame(), pd.DataFrame()
    return bars, build_feature_frame(bars)


def estimate_outcomes(bars, features):
    if features.empty or bars.empty:
        return pd.Series(dtype=float)
    side = pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0).values
    pnl = np.zeros(len(features), dtype=float)
    close = bars["close"].astype(float)
    high  = bars["high"].astype(float)
    low   = bars["low"].astype(float)
    close_idx = close.index
    lookahead = 60
    for i, (ts, s) in enumerate(zip(features.index, side)):
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
        if s > 0:
            sl = entry - SL_POINTS; tp = entry + TP_POINTS
            hit_sl = low.iloc[start_pos:end_pos].le(sl)
            hit_tp = high.iloc[start_pos:end_pos].ge(tp)
        else:
            sl = entry + SL_POINTS; tp = entry - TP_POINTS
            hit_sl = high.iloc[start_pos:end_pos].ge(sl)
            hit_tp = low.iloc[start_pos:end_pos].le(tp)
        sl_i = hit_sl.values.argmax() if hit_sl.any() else 1 << 30
        tp_i = hit_tp.values.argmax() if hit_tp.any() else 1 << 30
        if sl_i == 1 << 30 and tp_i == 1 << 30:
            last = float(close.iloc[end_pos - 1])
            pts = (last - entry) if s > 0 else (entry - last)
            pnl[i] = pts * MES_PT_VALUE
        elif sl_i < tp_i:
            pnl[i] = -SL_POINTS * MES_PT_VALUE
        else:
            pnl[i] = TP_POINTS * MES_PT_VALUE
    return pd.Series(pnl, index=features.index)


def trade_stats(pnl_arr):
    a = pnl_arr[np.isfinite(pnl_arr)]
    if a.size == 0:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "dd": 0.0, "avg": 0.0}
    cum = np.cumsum(a)
    peak = np.maximum.accumulate(cum)
    return {
        "n":   int(a.size),
        "pnl": float(cum[-1]),
        "wr":  float((a > 0).sum()) / a.size * 100.0,
        "dd":  float(np.max(peak - cum)),
        "avg": float(cum[-1]) / a.size,
    }


def summary(probs):
    a = probs[np.isfinite(probs)]
    return {
        "n":    int(a.size),
        "mean": float(np.mean(a)) if a.size else 0.0,
        "std":  float(np.std(a))  if a.size else 0.0,
        "p50":  float(np.median(a)) if a.size else 0.0,
        "p90":  float(np.percentile(a, 90)) if a.size else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrained", required=True)
    ap.add_argument("--retrained-threshold", required=True)
    args = ap.parse_args()

    with DEPLOY_PKL.open("rb") as fh:  deploy = pickle.load(fh)
    with open(args.retrained, "rb") as fh:  retr = pickle.load(fh)
    deploy_thr = float(json.load(DEPLOY_THR.open())["threshold"])
    retr_thr = float(json.load(open(args.retrained_threshold))["threshold"])

    print(f"[compare] deployed threshold={deploy_thr}  retrained threshold={retr_thr}")
    prices = price_context.load_prices().sort_index()

    # Train slice (for coverage ratio reference) — candidate-only filter
    print(f"\n[train slice] {TRAIN_SLICE_START} → {TRAIN_SLICE_END}")
    _, tr_feat = features_for(prices, TRAIN_SLICE_START, TRAIN_SLICE_END)
    _tr_side = pd.to_numeric(tr_feat.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    tr_feat = tr_feat.loc[(_tr_side != 0).values]
    dep_tr_probs = predict_bundle_probabilities(deploy, tr_feat)
    retr_tr_probs = predict_bundle_probabilities(retr, tr_feat)
    print(f"  candidate rows: {len(tr_feat):,}")
    print(f"  deployed  train dist: {summary(dep_tr_probs)}")
    print(f"  retrained train dist: {summary(retr_tr_probs)}")

    # OOS — filter to candidate rows (candidate_side != 0) because the
    # live bot only predicts on candidate-setup rows; predicting on every
    # bar with zero outcomes pollutes the stats.
    print(f"\n[OOS] {OOS_START} → {OOS_END}")
    oos_bars, oos_feat = features_for(prices, OOS_START, OOS_END)
    _side = pd.to_numeric(oos_feat.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    oos_feat = oos_feat.loc[(_side != 0).values]
    oos_pnl = estimate_outcomes(oos_bars, oos_feat).values
    dep_oos = predict_bundle_probabilities(deploy, oos_feat)
    retr_oos = predict_bundle_probabilities(retr, oos_feat)
    print(f"  candidate rows: {len(oos_feat):,}")
    print(f"  deployed  OOS dist: {summary(dep_oos)}")
    print(f"  retrained OOS dist: {summary(retr_oos)}")

    dep_mask = dep_oos >= deploy_thr
    retr_mask = retr_oos >= retr_thr
    dep_stats = trade_stats(oos_pnl[dep_mask])
    retr_stats = trade_stats(oos_pnl[retr_mask])
    print(f"\n  deployed  @ ≥{deploy_thr}: n={dep_stats['n']:<5}  PnL=${dep_stats['pnl']:+.2f}  "
          f"WR={dep_stats['wr']:.2f}%  DD=${dep_stats['dd']:.0f}  avg=${dep_stats['avg']:+.3f}")
    print(f"  retrained @ ≥{retr_thr}: n={retr_stats['n']:<5}  PnL=${retr_stats['pnl']:+.2f}  "
          f"WR={retr_stats['wr']:.2f}%  DD=${retr_stats['dd']:.0f}  avg=${retr_stats['avg']:+.3f}")

    # Live coverage — candidate-only
    print(f"\n[LIVE] {LIVE_START} → parquet_end")
    _, live_feat = features_for(prices, LIVE_START, None)
    _live_side = pd.to_numeric(live_feat.get("candidate_side", 0.0), errors="coerce").fillna(0.0)
    live_feat = live_feat.loc[(_live_side != 0).values]
    dep_live = predict_bundle_probabilities(deploy, live_feat)
    retr_live = predict_bundle_probabilities(retr, live_feat)
    dep_cov = float((dep_live >= deploy_thr).mean()) * 100
    retr_cov = float((retr_live >= retr_thr).mean()) * 100
    dep_train_cov = float((dep_tr_probs >= deploy_thr).mean()) * 100
    retr_train_cov = float((retr_tr_probs >= retr_thr).mean()) * 100
    dep_cov_ratio  = dep_cov / max(dep_train_cov, 1e-9)
    retr_cov_ratio = retr_cov / max(retr_train_cov, 1e-9)
    print(f"  rows: {len(live_feat):,}")
    print(f"  deployed  live coverage: {dep_cov:.2f}%  (train={dep_train_cov:.2f}% → ratio {dep_cov_ratio:.2f})")
    print(f"  retrained live coverage: {retr_cov:.2f}%  (train={retr_train_cov:.2f}% → ratio {retr_cov_ratio:.2f})")

    # Ship gates
    dep_pnl = dep_stats["pnl"]
    ship = {
        "oos_pnl_ok":  retr_stats["pnl"] >= dep_pnl or (dep_pnl <= 0 and retr_stats["pnl"] > dep_pnl),
        "oos_wr_ok":   (retr_stats["wr"] - dep_stats["wr"]) >= -1.0,
        "oos_dd_ok":   retr_stats["dd"] <= dep_stats["dd"] * 1.10 if dep_stats["dd"] > 0 else True,
        "live_cov_ok": 0.50 <= retr_cov_ratio <= 1.50,
    }
    print("\n════════════ SHIP GATES ════════════")
    for k, v in ship.items():
        print(f"  {'✓' if v else '✗'}  {k}")

    all_pass = all(ship.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "deploy_threshold": deploy_thr, "retrained_threshold": retr_thr,
        "train_dist": {"deployed": summary(dep_tr_probs), "retrained": summary(retr_tr_probs)},
        "oos": {
            "dist_deployed":  summary(dep_oos),
            "dist_retrained": summary(retr_oos),
            "stats_deployed":  dep_stats,
            "stats_retrained": retr_stats,
        },
        "live_coverage": {
            "deployed_pct":   dep_cov,  "deployed_ratio":   dep_cov_ratio,
            "retrained_pct":  retr_cov, "retrained_ratio":  retr_cov_ratio,
        },
        "ship_gates": ship,
        "ship": all_pass,
    }, indent=2, default=str))
    print(f"\n[write] {OUT}")
    if all_pass:
        print("\n[SHIP] retrained bundle passes all gates — ready to flip config.py")
    else:
        print("\n[NO-SHIP] retrained bundle does NOT pass all gates — investigate before flipping live")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
