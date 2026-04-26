"""Train the v11 Pivot Trail head and run a holdout-PnL threshold sweep.

Pipeline (per strategy):
  1) Load v11_pivot_labels.parquet → join back to v11_training_corpus.parquet
     for the 40 regime + 13 Kalshi tick + 4 upstream proba features.
  2) Time-split: train < 2026-01-01, holdout >= 2026-01-01 (matches the
     deployment-time guarantee that we never look at future data).
  3) Skip strategy if n_train < 100 OR n_test < 50 OR positive count == 0
     in either split.
  4) HGB(max_depth=3, max_iter=200), cost-sensitive sample_weight =
     clip(|delta_pnl| / 25.0, 0.5, 4.0).
  5) Threshold sweep 0.40..0.85 step 0.05 — at each threshold, simulate
     the head's decision row-by-row on holdout and sum net_pnl.
     "ARM if proba >= thr" (positive class = "Pivot helped").
  6) Apply ship gates: G1 DD<=$870, G2 PnL >= holdout-no-pivot baseline AND
     trades <= baseline trade count, G3 n_OOS >= 50, G4 WR >= 55%.
  7) If at least one threshold passes all 4 gates: SHIP (write model.joblib +
     thresholds.json with the gate-passing threshold whose PnL is highest).
     Otherwise KILL (write metrics.json with the verdict).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS_PATH = ROOT / "artifacts" / "v11_training_corpus.parquet"
LABELS_PATH = ROOT / "artifacts" / "v11_pivot_labels.parquet"
OUT_BASE = ROOT / "artifacts" / "regime_ml_pivot_v11"

THRESHOLDS = np.arange(0.40, 0.85 + 1e-9, 0.05)
N_TRAIN_MIN = 100
N_TEST_MIN = 50
DD_LIMIT = 870.0
WR_MIN = 0.55

# Feature lists ----------------------------------------------------------
BF_FEATURES = [
    "bf_de3_entry_ret1_atr", "bf_de3_entry_body_pos1", "bf_de3_entry_body1_ratio",
    "bf_de3_entry_lower_wick_ratio", "bf_de3_entry_upper_wick_ratio",
    "bf_de3_entry_upper1_ratio", "bf_de3_entry_close_pos1",
    "bf_de3_entry_flips5", "bf_de3_entry_down3", "bf_de3_entry_range10_atr",
    "bf_de3_entry_dist_high5_atr", "bf_de3_entry_dist_low5_atr",
    "bf_de3_entry_vol1_rel20", "bf_de3_entry_atr14",
    "bf_sl_dist_pts", "bf_tp_dist_pts", "bf_atr_ratio_to_sl",
    "bf_dist_to_bank_below", "bf_dist_to_bank_above",
    "bf_dist_to_bank_in_dir", "bf_dist_to_bank_pts",
    "bf_bar_range_pts", "bf_bar_close_pct_body",
    "bf_atr14_pts", "bf_range_30bar_pts", "bf_trend_20bar_pct",
    "bf_dist_to_20bar_hi_pct", "bf_dist_to_20bar_lo_pct",
    "bf_vel_5bar_pts_per_min", "bf_vel_20bar_pts_per_min",
    "bf_regime_vol_bp", "bf_regime_eff",
]
PCT_FEATURES = [
    "pct_pct_from_open", "pct_signed_level", "pct_abs_level",
    "pct_level_distance_pct", "pct_atr_pct_30bar", "pct_range_pct_at_touch",
    "pct_hour_edge", "pct_minutes_since_open",
    "pct_dist_to_running_hi_pct", "pct_dist_to_running_lo_pct",
    "pct_rule_confidence",
]
PROBA_FEATURES = ["fg_proba", "kalshi_proba", "lfo_proba", "pct_proba"]
NUMERIC_FEATURES = BF_FEATURES + PCT_FEATURES + PROBA_FEATURES   # 32 + 11 + 4 = 47


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp("2026-01-01", tz="US/Eastern")
    return df[df["ts"] < cutoff].reset_index(drop=True), df[df["ts"] >= cutoff].reset_index(drop=True)


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(dd.max())


def run_strategy(strategy: str, joined: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    strat_df = joined[joined["strategy"] == strategy].copy()
    n_total = len(strat_df)
    train, test = split_train_test(strat_df)
    n_train = len(train)
    n_test = len(test)
    n_train_pos = int(train["pivot_label"].sum()) if n_train else 0
    n_test_pos = int(test["pivot_label"].sum()) if n_test else 0
    info = {
        "strategy": strategy, "stage": "pivot_v11",
        "n_total": n_total, "n_train": n_train, "n_test": n_test,
        "n_train_pos": n_train_pos, "n_test_pos": n_test_pos,
    }

    # Skip-split gate
    if n_train < N_TRAIN_MIN or n_test < N_TEST_MIN or n_train_pos < 5 or n_test_pos < 1:
        info["status"] = "SKIP_split"
        info["reason"] = (
            f"n_train={n_train} (need >={N_TRAIN_MIN}), "
            f"n_test={n_test} (need >={N_TEST_MIN}), "
            f"n_train_pos={n_train_pos} (need >=5), "
            f"n_test_pos={n_test_pos} (need >=1)"
        )
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(info, f, indent=2, default=str)
        print(f"[{strategy}] {info['status']}: {info['reason']}")
        return info

    # Features ------------------------------------------------------------
    X_train = train[NUMERIC_FEATURES].astype(float).values
    X_test = test[NUMERIC_FEATURES].astype(float).values
    y_train = train["pivot_label"].astype(int).values
    y_test = test["pivot_label"].astype(int).values

    # Cost-sensitive weights — emphasise rows where the decision matters
    delta_train = train["delta_pnl"].astype(float).abs().values
    sw = np.clip(delta_train / 25.0, 0.5, 4.0)

    # HGB ------------------------------------------------------------------
    model = HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.05,
        random_state=42, early_stopping=False,
    )
    model.fit(X_train, y_train, sample_weight=sw)
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]
    try:
        train_auc = float(roc_auc_score(y_train, proba_train))
    except ValueError:
        train_auc = float("nan")
    try:
        test_auc = float(roc_auc_score(y_test, proba_test))
    except ValueError:
        test_auc = float("nan")

    info["train_auc"] = train_auc
    info["test_auc"] = test_auc

    # Holdout baselines (no-pivot, always-pivot, and "perfect" oracle) ----
    pnl_no = test["pnl_no_pivot"].astype(float).values
    pnl_pv = test["pnl_with_pivot"].astype(float).values
    baseline_no_pivot = {
        "trades": int(n_test),
        "wr": float((pnl_no > 0).mean()),
        "pnl": float(pnl_no.sum()),
        "dd": max_drawdown(pnl_no.cumsum()),
    }
    baseline_always_pivot = {
        "trades": int(n_test),
        "wr": float((pnl_pv > 0).mean()),
        "pnl": float(pnl_pv.sum()),
        "dd": max_drawdown(pnl_pv.cumsum()),
    }
    oracle_pnl = np.maximum(pnl_no, pnl_pv).sum()
    info["holdout_baselines"] = {
        "no_pivot": baseline_no_pivot,
        "always_pivot": baseline_always_pivot,
        "oracle_pnl": float(oracle_pnl),
    }

    # Threshold sweep -----------------------------------------------------
    sweep = []
    for thr in THRESHOLDS:
        # ARM if proba >= thr  (positive class = pivot helped)
        decisions = proba_test >= thr
        chosen_pnl = np.where(decisions, pnl_pv, pnl_no)
        wr = float((chosen_pnl > 0).mean())
        pnl = float(chosen_pnl.sum())
        dd = max_drawdown(chosen_pnl.cumsum())
        n_arm = int(decisions.sum())
        gates = {
            "G1_dd_le_870": bool(dd <= DD_LIMIT),
            # G2: PnL >= no-pivot baseline AND total trades <= baseline trades.
            # (Trade count is the same — head decides ARM vs NOT, doesn't kill
            # trades — so G2 simplifies to PnL >= baseline.)
            "G2_pnl_ge_baseline": bool(pnl >= baseline_no_pivot["pnl"] - 1e-9
                                       and n_test <= baseline_no_pivot["trades"]),
            "G3_n_oos_ge_50": bool(n_test >= 50),
            "G4_wr_ge_55": bool(wr >= WR_MIN),
        }
        gates["all_pass"] = bool(all(gates[k] for k in
                                     ["G1_dd_le_870", "G2_pnl_ge_baseline",
                                      "G3_n_oos_ge_50", "G4_wr_ge_55"]))
        sweep.append({
            "threshold": float(thr),
            "n_arm": n_arm,
            "trades": int(n_test),
            "wr": wr,
            "pnl": pnl,
            "dd": dd,
            **gates,
        })
    info["threshold_sweep"] = sweep

    # Pick best gate-passing threshold, else best PnL ---------------------
    passing = [s for s in sweep if s["all_pass"]]
    if passing:
        best = max(passing, key=lambda s: s["pnl"])
        status = "SHIP"
        binding = "all_passed"
    else:
        best = max(sweep, key=lambda s: s["pnl"])
        status = "KILL"
        # Find first failing gate at the best-PnL threshold
        for g in ["G1_dd_le_870", "G2_pnl_ge_baseline", "G3_n_oos_ge_50", "G4_wr_ge_55"]:
            if not best[g]:
                binding = g
                break
        else:
            binding = "n/a"
    info["best_threshold"] = best
    info["status"] = status
    info["binding_gate"] = binding

    # Persist -------------------------------------------------------------
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    if status == "SHIP":
        joblib.dump(model, out_dir / "model.joblib")
        with open(out_dir / "thresholds.json", "w") as f:
            json.dump({
                "threshold": best["threshold"],
                "decision": "ARM if proba >= threshold",
                "label_semantics": "positive class = pivot trail arming helped",
                "n_oos": n_test,
                "pnl_oos": best["pnl"],
                "wr_oos": best["wr"],
                "dd_oos": best["dd"],
            }, f, indent=2)

    print(f"[{strategy}] {status}  thr={best['threshold']:.2f} "
          f"n_arm={best['n_arm']}/{best['trades']} pnl=${best['pnl']:.2f} "
          f"wr={best['wr']*100:.1f}% dd=${best['dd']:.2f}  "
          f"binding={binding}  test_auc={test_auc:.3f}")
    return info


def main() -> None:
    print("[pivot_v11] loading corpus + labels")
    corpus = pd.read_parquet(CORPUS_PATH)
    labels = pd.read_parquet(LABELS_PATH)
    # Join on (ts, strategy, side, contract, entry_price)
    keys = ["ts", "strategy", "side", "contract", "entry_price"]
    joined = labels.merge(corpus, on=keys, how="left", suffixes=("", "_corpus"))
    print(f"[pivot_v11] joined rows: {len(joined)}  (labels={len(labels)})")
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    summary = {}
    for strategy, slug in [("DynamicEngine3", "de3"), ("RegimeAdaptive", "ra")]:
        out_dir = OUT_BASE / slug
        info = run_strategy(strategy, joined, out_dir)
        summary[slug] = info

    with open(OUT_BASE / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[pivot_v11] summary -> {OUT_BASE / 'summary.json'}")


if __name__ == "__main__":
    main()
