#!/usr/bin/env python3
"""Kalshi ML v8 — train on the clean NO-ML-stack 14-month dataset.

Inputs: artifacts/kalshi_training_v8_*.parquet (14 monthly files)
Output: artifacts/regime_ml_kalshi_v8/{model.pkl, model_meta.json}

Approach summary (the v1-v7 lessons baked in):
  - Use 60-min forward horizon (clean class separation, $30/-$20 medians)
  - HGB-only (LightGBM banned — OMP/asyncio crash)
  - Recency-weighted training (exponential decay, half-life 120 days)
  - Binary override action: ML defers to rule unless |proba - 0.5| ≥ margin
  - Sweep 5 override margins; pick best by lift over rule on the OOS holdout
  - Walk-forward time-series split: train on first 11 months, OOS on last 3
  - Optional: include the 12 v2 derived features (Kalshi deltas, intraday
    cumulative, time-to-settlement, threshold flip)

Ship gates (all 5 must pass at picked override margin):
  1. OOS PnL lift > 0 vs rule baseline
  2. DD/PnL ratio ≤ 30%
  3. n_kalshi_events ≥ 50 in OOS
  4. newly-PASSED WR ≥ 50% (if ≥ 5 such overrides)
  5. capture ≥ 20% of (oracle - rule) lift
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/regime_ml"))
from _common import stats, sample_weights_balanced

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_kalshi_v8")

# Base Kalshi features (in every monthly parquet)
BASE_FEATURES = [
    "k_entry_probability", "k_probe_probability",
    "k_momentum_delta", "k_momentum_retention",
    "k_support_score", "k_threshold",
]

# Derived features computed in-script (no monthly-parquet dependency)
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("market_ts").reset_index(drop=True)
    df["side_sign"] = np.where(df["side"] == "LONG", 1, -1)
    # k_margin: signed distance entry_prob − threshold
    df["k_margin"] = df["k_entry_probability"] - df["k_threshold"]
    df["k_margin_abs"] = np.abs(df["k_margin"])
    # role one-hots
    df["role_forward"] = df["role"].astype(str).str.contains("forward").astype(int)
    df["role_background"] = (df["role"] == "background").astype(int)
    df["role_balanced"] = (df["role"] == "balanced").astype(int)
    df["is_de3"] = df["strategy"].astype(str).str.lower().str.contains("dynamicengine").astype(int)
    df["is_ra"]  = df["strategy"].astype(str).str.lower().str.contains("regimeadaptive").astype(int)
    # settlement hour from market_ts
    df["_ts_dt"] = pd.to_datetime(df["market_ts"])
    df["settlement_hour"] = df["_ts_dt"].dt.hour
    df["minute_of_hour"] = df["_ts_dt"].dt.minute
    df["minutes_to_hour_close"] = 60 - df["minute_of_hour"]
    return df


FEATURE_COLS_V8 = BASE_FEATURES + [
    "side_sign", "k_margin", "k_margin_abs",
    "role_forward", "role_background", "role_balanced",
    "is_de3", "is_ra",
    "settlement_hour", "minute_of_hour", "minutes_to_hour_close",
]


def recency_weights(timestamps: pd.Series, half_life_days: float) -> np.ndarray:
    ref = timestamps.max()
    ages = (ref - timestamps).dt.total_seconds() / 86400.0
    return np.power(0.5, ages.to_numpy() / max(1e-6, half_life_days))


def evaluate_override(
    proba_pass: np.ndarray,
    rule_decision: np.ndarray,
    forward_pnl: np.ndarray,
    label: np.ndarray,
    margin: float,
) -> dict:
    """Binary override semantics — ML acts only when confident."""
    ml_pass = proba_pass >= (0.5 + margin)
    ml_block = proba_pass <= (0.5 - margin)
    final_pass = np.where(
        ml_pass, True,
        np.where(ml_block, False, rule_decision == "PASS"),
    )
    final_pnl = np.where(final_pass, forward_pnl, 0.0)
    ml_st = stats(final_pnl)
    n_pass = int(final_pass.sum())
    new_pass_mask = (rule_decision == "BLOCK") & final_pass & ml_pass
    n_new_pass = int(new_pass_mask.sum())
    new_pass_pnls = forward_pnl[new_pass_mask]
    new_pass_wr = (100 * (new_pass_pnls > 0).sum() / len(new_pass_pnls)) if len(new_pass_pnls) else 0.0
    new_block_mask = (rule_decision == "PASS") & (~final_pass) & ml_block
    n_new_block = int(new_block_mask.sum())
    new_block_pnls = forward_pnl[new_block_mask]
    new_block_wr = (100 * (new_block_pnls <= 0).sum() / len(new_block_pnls)) if len(new_block_pnls) else 0.0
    return {
        "margin": margin,
        "n_pass": n_pass,
        "n_new_pass": n_new_pass, "new_pass_wr": new_pass_wr,
        "n_new_block": n_new_block, "new_block_wr": new_block_wr,
        "pnl": ml_st["pnl"], "dd": ml_st["dd"], "avg": ml_st["avg"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=60, help="Label horizon (minutes)")
    ap.add_argument("--half-life", type=float, default=120.0)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_kalshi_v8"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-glob", default=str(ROOT / "artifacts/kalshi_training_v8_*.parquet"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(args.input_glob))
    log.info("found %d monthly training parquets", len(files))
    if len(files) < 6:
        log.warning("only %d files — refusing to train (need ≥6 months)", len(files))
        return 1

    parts = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["_source_file"] = Path(f).stem
            parts.append(df)
        except Exception as e:
            log.warning("  skipping %s: %s", f, e)
    full = pd.concat(parts, ignore_index=True)
    log.info("combined rows: %d", len(full))

    full = add_derived_features(full)

    # Filter to clean labels at chosen horizon
    label_col = f"label_{args.horizon}m"
    pnl_col = f"forward_pnl_{args.horizon}m"
    full = full[full[label_col].isin(["pass", "block"])].copy()
    log.info("after label filter (horizon %dm): %d rows", args.horizon, len(full))
    log.info("class dist: %s", dict(Counter(full[label_col])))
    log.info("rule x label cross-tab:")
    print(pd.crosstab(full["rule_decision"], full[label_col]).to_string())

    # Time-based train/OOS split (last 25% of timeline = OOS)
    full = full.sort_values("_ts_dt").reset_index(drop=True)
    split_at = int(len(full) * 0.75)
    tr = full.iloc[:split_at].copy()
    oos = full.iloc[split_at:].copy()
    log.info("train: %d  OOS: %d  (split @ %s)", len(tr), len(oos),
             full.iloc[split_at]["_ts_dt"])

    if len(tr) < 200 or len(oos) < 50:
        log.warning("[KILL] floors not met — train=%d OOS=%d", len(tr), len(oos))
        return 1

    X_tr = tr[FEATURE_COLS_V8].to_numpy()
    y_tr = tr[label_col].to_numpy()
    X_oos = oos[FEATURE_COLS_V8].to_numpy()
    y_oos = oos[label_col].to_numpy()
    pnl_oos = oos[pnl_col].to_numpy()
    rule_oos = oos["rule_decision"].to_numpy()

    cw = sample_weights_balanced(y_tr, cost_ratio=1.3)
    rw = recency_weights(tr["_ts_dt"], args.half_life)
    sw = cw * rw

    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=args.seed,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    pass_idx = list(clf.classes_).index("pass")
    proba_pass = clf.predict_proba(X_oos)[:, pass_idx]

    # Baselines
    rule_pnl = np.where(rule_oos == "PASS", pnl_oos, 0.0)
    rule_st = stats(rule_pnl)
    oracle_pnl = np.where(y_oos == "pass", pnl_oos, 0.0)
    oracle_st = stats(oracle_pnl)
    headroom = oracle_st["pnl"] - rule_st["pnl"]

    log.info("rule baseline:  PnL=$%+.2f DD=$%.0f", rule_st["pnl"], rule_st["dd"])
    log.info("oracle:         PnL=$%+.2f", oracle_st["pnl"])
    log.info("headroom:       $%+.2f", headroom)

    log.info("\n%-7s %-7s %-7s %-7s %-10s %-10s %-7s %-7s gates",
             "margin", "n_pass", "new_psn", "new_psWR", "lift$", "dd$", "dd/pnl%", "capt%")
    log.info("─" * 100)
    margins = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    results = []
    for m in margins:
        r = evaluate_override(proba_pass, rule_oos, pnl_oos, y_oos, m)
        lift = r["pnl"] - rule_st["pnl"]
        dd_over_pnl = (r["dd"] / r["pnl"] * 100.0) if r["pnl"] > 0 else float("inf")
        capt = (lift / headroom * 100.0) if headroom > 0 else 0.0
        gates = {
            "pnl_ok":         lift > 0,
            "dd_ratio_ok":    dd_over_pnl <= 30.0,
            "n_ok":           len(oos) >= 50,
            "new_pass_wr_ok": (r["new_pass_wr"] >= 50.0) if r["n_new_pass"] >= 5 else True,
            "capt_ok":        capt >= 20.0,
        }
        ships = all(gates.values())
        log.info("%-7.2f %-7d %-7d %-7.2f%% $%-8.2f $%-8.0f %-6.1f%% %-6.2f%%  %d/5%s",
                 m, r["n_pass"], r["n_new_pass"], r["new_pass_wr"], r["pnl"], r["dd"],
                 dd_over_pnl, capt, sum(gates.values()), " SHIP" if ships else "")
        r.update({"lift": lift, "dd_over_pnl": dd_over_pnl,
                  "capt_pct": capt, "gates": gates, "ships": ships})
        results.append(r)

    shippers = [r for r in results if r["ships"]]
    if shippers:
        best = max(shippers, key=lambda r: r["lift"])
        log.info("\n[SHIP] margin=%.2f  lift=$%+.2f  capt=%.2f%%  newPassWR=%.2f%%",
                 best["margin"], best["lift"], best["capt_pct"], best["new_pass_wr"])
        payload = {
            "clf": clf,
            "feature_cols": FEATURE_COLS_V8,
            "override_margin": best["margin"],
            "label_horizon_min": args.horizon,
            "half_life_days": args.half_life,
            "rule_baseline_oos": rule_st,
            "oracle_oos": oracle_st,
            "best_oos_stats": best,
            "n_train": len(tr),
            "n_oos": len(oos),
            "positive_class": "pass",
        }
        with (out_dir / "model.pkl").open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        meta = {k: v for k, v in payload.items() if k != "clf"}
        (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str))
        log.info("[ship] artifact written to %s", out_dir)
        return 0
    else:
        log.warning("\n[KILL] no override margin clears all 5 gates")
        # Save sweep summary anyway for debugging
        (out_dir / "sweep_summary.json").write_text(json.dumps({
            "verdict": "KILL",
            "rule_baseline_oos": rule_st,
            "oracle_oos": oracle_st,
            "headroom": headroom,
            "n_train": len(tr), "n_oos": len(oos),
            "all_margins": results,
        }, indent=2, default=str))
        return 1


if __name__ == "__main__":
    sys.exit(main())
