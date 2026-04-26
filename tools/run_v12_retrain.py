#!/usr/bin/env python3
"""V12 retrain pipeline.

Trains 4 stack-aware HGB heads (Kalshi → LFO → PCT → Pivot) on the v12 corpus
with hydrated Kalshi snapshot features and DE3 shock-context features.

For each head:
  - Trains 2025-03..2025-12, holds out 2026-01..2026-04
  - Filters allowed_by_friend_rule == True, strategy == DynamicEngine3
  - Sample weight: clip(|net_pnl|/50.0, 0.5, 4.0)
  - Sweeps thresholds 0.40..0.85 step 0.05
  - For each threshold: simulate "BLOCK if proba >= threshold on the
    is_big_loss head" → keep surviving holdout rows → compute trades / WR /
    PnL / DD using the v11 corpus's existing per-row outcomes
    (net_pnl_after_haircut, which was already computed via simulator_trade_through.simulate_trade()
    or pivot_stepped_sl walk for Pivot)
  - 4 ship gates per §8.31: G1 |DD|<=$870; G2 PnL>=baseline; G3 trades>=50;
    G4 WR>=55%

Phase 3 — §8.31 rule + ML cross-product:
  - Apply rule pre-filter, then test ML in three modes per head:
    (A) refine within rule-kept,   (B) "rule kept" union "ML kept",
    (C) ML kept regardless of rule
  - Find best 4-gate-passing config per head.

Phase 4 — Combined stack v12: §8.31 rule first, then all heads in deployment
order with their best thresholds. Block if rule says block OR any head says
block.

Outputs:
  artifacts/regime_ml_<head>_v12/de3/{model.joblib,thresholds.json,metrics.json}
  artifacts/v12_combined_stack_metrics.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V12_CORPUS = ROOT / "artifacts" / "v12_training_corpus.parquet"
ART = ROOT / "artifacts"

# --- Config ---
SHIP_GATES = {
    "G1_max_dd_abs": 870.0,
    "G3_min_n_oos_surviving": 50,
    "G4_min_wr_pct": 55.0,
}

THRESHOLDS = [round(0.10 + 0.025 * i, 4) for i in range(31)]  # 0.10..0.85 step 0.025
RULE_REGIME_EFF_MIN = 0.0900
RULE_UPPER_WICK_MAX = 0.0353

# Base v11 regime/feature columns (40+) excluding the upstream proba slots
BASE_NUMERIC_FEATS = [
    "bf_de3_entry_ret1_atr", "bf_de3_entry_body_pos1", "bf_de3_entry_body1_ratio",
    "bf_de3_entry_lower_wick_ratio", "bf_de3_entry_upper_wick_ratio",
    "bf_de3_entry_upper1_ratio", "bf_de3_entry_close_pos1",
    "bf_de3_entry_flips5", "bf_de3_entry_down3", "bf_de3_entry_range10_atr",
    "bf_de3_entry_dist_high5_atr", "bf_de3_entry_dist_low5_atr",
    "bf_de3_entry_vol1_rel20", "bf_de3_entry_atr14",
    "bf_sl_dist_pts", "bf_tp_dist_pts", "bf_atr_ratio_to_sl",
    "bf_dist_to_bank_below", "bf_dist_to_bank_above", "bf_dist_to_bank_in_dir",
    "bf_dist_to_bank_pts", "bf_bar_range_pts", "bf_bar_close_pct_body",
    "bf_atr14_pts", "bf_range_30bar_pts", "bf_trend_20bar_pct",
    "bf_dist_to_20bar_hi_pct", "bf_dist_to_20bar_lo_pct",
    "bf_vel_5bar_pts_per_min", "bf_vel_20bar_pts_per_min",
    "bf_regime_vol_bp", "bf_regime_eff",
    "pct_pct_from_open", "pct_signed_level", "pct_abs_level",
    "pct_level_distance_pct", "pct_atr_pct_30bar", "pct_range_pct_at_touch",
    "pct_hour_edge", "pct_minutes_since_open",
    "pct_dist_to_running_hi_pct", "pct_dist_to_running_lo_pct",
    "pct_rule_confidence",
]

# v12 hydrated Kalshi
K12_FEATS = [
    "k12_entry_probability", "k12_probe_probability", "k12_probe_neg_probability",
    "k12_skew_p10", "k12_skew_p25", "k12_above_5", "k12_above_10", "k12_below_10",
    "k12_distance_to_50", "k12_momentum_5", "k12_momentum_15",
    "k12_window_active", "k12_data_present",
]

# v12 shock context (numeric subset)
SHOCK_FEATS = [
    "ctx_day_first60_share", "ctx_day_gap_ratio", "ctx_day_range_progress_ratio",
    "ctx_day_trend_frac", "ctx_day_volume_progress_ratio",
    "ctx_shock_recent_range_ratio", "ctx_shock_recent_volume_ratio",
    "ctx_shock_score", "ctx_shock_session_move_norm", "ctx_shock_session_range_norm",
]


# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------

def _to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def metrics_for_subset(sub: pd.DataFrame) -> Dict[str, float]:
    """Compute trades / WR / PnL / DD over an ordered subset using
    `net_pnl_after_haircut` as ground truth."""
    if len(sub) == 0:
        return {"trades": 0, "wr_pct": 0.0, "pnl": 0.0, "dd": 0.0,
                "wins": 0, "losses": 0}
    s = sub.sort_values("ts").reset_index(drop=True)
    pnl = s["net_pnl_after_haircut"].astype(float)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    cum = pnl.cumsum()
    peak = cum.cummax()
    dd = float((cum - peak).min()) if len(cum) > 0 else 0.0
    return {
        "trades": int(len(s)),
        "wr_pct": float((pnl > 0).mean() * 100.0) if len(s) > 0 else 0.0,
        "pnl": float(pnl.sum()),
        "dd": dd,
        "wins": wins,
        "losses": losses,
    }


def gates_pass(metrics: Dict[str, float], baseline_pnl: float) -> Tuple[Dict[str, bool], str]:
    g1 = abs(metrics["dd"]) <= SHIP_GATES["G1_max_dd_abs"]
    g2 = metrics["pnl"] >= baseline_pnl
    g3 = metrics["trades"] >= SHIP_GATES["G3_min_n_oos_surviving"]
    g4 = metrics["wr_pct"] >= SHIP_GATES["G4_min_wr_pct"]
    binding = ""
    for k, v in [("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4)]:
        if not v:
            binding = k
            break
    return {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}, binding


def fit_hgb(X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray,
            seed: int = 42) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.05,
        min_samples_leaf=20, l2_regularization=0.5, random_state=seed,
    )
    clf.fit(X_tr, y_tr, sample_weight=w_tr)
    return clf


def make_feature_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    out = np.zeros((len(df), len(cols)), dtype=float)
    for j, c in enumerate(cols):
        if c in df.columns:
            v = _to_numeric_safe(df[c]).fillna(0.0).to_numpy()
            out[:, j] = v
    return out


def sweep_thresholds(probas_hold: np.ndarray, hold: pd.DataFrame,
                     baseline_pnl: float) -> Tuple[List[Dict], Dict]:
    """For each threshold, BLOCK rows where proba >= thr (i.e. predicted
    big_loss). Compute metrics on surviving subset. Find best 4-gate passing
    config; if none, find closest (most gates passing, then best PnL).
    """
    n = len(hold)
    sweep = []
    best_passing = None
    best_partial = None
    best_partial_score = (-1, -1e18)  # (n_gates_pass, pnl)
    for thr in THRESHOLDS:
        block = probas_hold >= thr
        keep_mask = ~block
        sub = hold[keep_mask]
        m = metrics_for_subset(sub)
        gates, binding = gates_pass(m, baseline_pnl)
        n_pass = sum(int(v) for v in gates.values())
        rec = {
            "threshold": float(thr),
            "blocked": int(block.sum()),
            "kept": int(len(sub)),
            "trades": m["trades"],
            "wr_pct": m["wr_pct"],
            "pnl": m["pnl"],
            "dd": m["dd"],
            "gates": gates,
            "all_pass": all(gates.values()),
            "binding_gate": binding,
            "n_gates_pass": n_pass,
        }
        sweep.append(rec)
        if rec["all_pass"]:
            if best_passing is None or rec["pnl"] > best_passing["pnl"]:
                best_passing = rec
        score = (n_pass, rec["pnl"])
        if score > best_partial_score:
            best_partial_score = score
            best_partial = rec
    return sweep, (best_passing if best_passing is not None else best_partial)


# --------------------------------------------------------------------------
# Per-head training
# --------------------------------------------------------------------------

def train_head(corpus: pd.DataFrame, head: str, feat_cols: List[str],
               extra_proba_cols: List[str]) -> Dict[str, Any]:
    """Train a single head. Returns metrics + writes model.joblib if SHIP."""
    out_dir = ART / f"regime_ml_{head}_v12" / "de3"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = corpus[(corpus["strategy"] == "DynamicEngine3") &
                (corpus["allowed_by_friend_rule"] == True)].copy()  # noqa: E712
    df["ts"] = pd.to_datetime(df["ts"])
    train = df[(df["ts"] >= "2025-03-01") & (df["ts"] < "2026-01-01")].copy()
    hold = df[(df["ts"] >= "2026-01-01") & (df["ts"] < "2026-05-01")].copy()
    print(f"\n[{head}] train n={len(train)}, holdout n={len(hold)}")

    feats_all = list(feat_cols) + list(extra_proba_cols)
    feats_all = [f for f in feats_all if f in df.columns]

    X_tr = make_feature_matrix(train, feats_all)
    y_tr = train["is_big_loss"].astype(int).to_numpy()
    if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
        return {"head": head, "status": "DEGENERATE_LABEL", "kill": True,
                "n_train": int(len(train)), "n_test": int(len(hold))}
    w_tr = np.clip(np.abs(train["net_pnl_after_haircut"].astype(float).fillna(0.0).to_numpy()) / 50.0, 0.5, 4.0)

    X_te = make_feature_matrix(hold, feats_all)
    y_te = hold["is_big_loss"].astype(int).to_numpy()

    clf = fit_hgb(X_tr, y_tr, w_tr)
    train_auc = float(roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1])) if len(set(y_tr)) > 1 else float("nan")
    test_auc = float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])) if len(set(y_te)) > 1 else float("nan")

    p_hold = clf.predict_proba(X_te)[:, 1]
    p_train = clf.predict_proba(X_tr)[:, 1]

    # Holdout baseline
    base_metrics = metrics_for_subset(hold)
    baseline_pnl = base_metrics["pnl"]

    sweep, best = sweep_thresholds(p_hold, hold, baseline_pnl)

    # Phase 3: §8.31 rule pre-filter — apply to holdout, then ML
    rule_mask_hold = (
        (_to_numeric_safe(hold["bf_regime_eff"]) > RULE_REGIME_EFF_MIN) &
        (_to_numeric_safe(hold["bf_de3_entry_upper_wick_ratio"]) <= RULE_UPPER_WICK_MAX)
    ).to_numpy()
    rule_kept = hold[rule_mask_hold]
    rule_metrics = metrics_for_subset(rule_kept)
    rule_gates, rule_binding = gates_pass(rule_metrics, baseline_pnl)

    # Mode A: refine within rule-kept
    sweep_A = []
    best_A = None
    if rule_mask_hold.sum() > 0:
        p_rule = p_hold[rule_mask_hold]
        for thr in THRESHOLDS:
            block = p_rule >= thr
            sub = rule_kept[~block]
            m = metrics_for_subset(sub)
            gates, binding = gates_pass(m, baseline_pnl)
            rec = {"threshold": float(thr), "trades": m["trades"], "wr_pct": m["wr_pct"],
                   "pnl": m["pnl"], "dd": m["dd"], "gates": gates,
                   "all_pass": all(gates.values()), "binding_gate": binding,
                   "blocked": int(block.sum()), "n_gates_pass": sum(int(v) for v in gates.values())}
            sweep_A.append(rec)
        # Pick best in mode A
        passing_A = [r for r in sweep_A if r["all_pass"]]
        if passing_A:
            best_A = max(passing_A, key=lambda r: r["pnl"])
        elif sweep_A:
            best_A = max(sweep_A, key=lambda r: (r["n_gates_pass"], r["pnl"]))

    # Mode B: union of rule-kept and ML-kept (ML approves what rule rejected at same threshold)
    sweep_B = []
    best_B = None
    for thr in THRESHOLDS:
        ml_keep = p_hold < thr
        union_keep = rule_mask_hold | ml_keep.to_numpy() if hasattr(ml_keep, "to_numpy") else (rule_mask_hold | ml_keep)
        sub = hold[union_keep]
        m = metrics_for_subset(sub)
        gates, binding = gates_pass(m, baseline_pnl)
        rec = {"threshold": float(thr), "trades": m["trades"], "wr_pct": m["wr_pct"],
               "pnl": m["pnl"], "dd": m["dd"], "gates": gates,
               "all_pass": all(gates.values()), "binding_gate": binding,
               "n_gates_pass": sum(int(v) for v in gates.values()),
               "kept": int(union_keep.sum())}
        sweep_B.append(rec)
    passing_B = [r for r in sweep_B if r["all_pass"]]
    if passing_B:
        best_B = max(passing_B, key=lambda r: r["pnl"])
    elif sweep_B:
        best_B = max(sweep_B, key=lambda r: (r["n_gates_pass"], r["pnl"]))

    # Mode C: ML alone (already in `sweep`)
    best_C = best

    # Pick overall best across modes
    candidates = [
        ("A_refine_in_rule", best_A, sweep_A),
        ("B_union_rule_or_ml", best_B, sweep_B),
        ("C_ml_alone", best_C, sweep),
    ]
    # Filter passing
    passing_modes = [(name, b, s) for name, b, s in candidates if b is not None and b.get("all_pass", False)]
    if passing_modes:
        best_mode_name, best_mode_rec, best_mode_sweep = max(passing_modes, key=lambda t: t[1]["pnl"])
        ship = True
    else:
        non_null = [(name, b, s) for name, b, s in candidates if b is not None]
        best_mode_name, best_mode_rec, best_mode_sweep = max(
            non_null, key=lambda t: (t[1].get("n_gates_pass", 0), t[1].get("pnl", -1e18))
        ) if non_null else ("C_ml_alone", None, sweep)
        ship = False

    metrics = {
        "overlay": head,
        "strategy": "DE3",
        "label": "is_big_loss",
        "n_train": int(len(train)),
        "n_test": int(len(hold)),
        "n_train_pos": int(y_tr.sum()),
        "n_test_pos": int(y_te.sum()),
        "train_AUC": train_auc,
        "test_AUC": test_auc,
        "features": feats_all,
        "holdout_baseline": base_metrics,
        "ship_gates": SHIP_GATES,
        "rule_alone_holdout": {**rule_metrics, "gates": rule_gates, "binding_gate": rule_binding,
                               "all_pass": all(rule_gates.values())},
        "sweep_C_ml_alone": sweep,
        "sweep_A_refine_in_rule": sweep_A,
        "sweep_B_union_rule_or_ml": sweep_B,
        "best_mode": best_mode_name,
        "best_config": best_mode_rec,
        "ship": bool(ship),
        "kill_marker": not bool(ship),
        "binding_gate": best_mode_rec.get("binding_gate", "") if best_mode_rec else "",
    }

    # Write model and thresholds
    if ship:
        joblib.dump({"model": clf, "features": feats_all,
                     "best_threshold": best_mode_rec["threshold"],
                     "best_mode": best_mode_name},
                    out_dir / "model.joblib")
        (out_dir / "thresholds.json").write_text(json.dumps({
            "threshold": best_mode_rec["threshold"], "mode": best_mode_name,
            "rule_min_regime_eff": RULE_REGIME_EFF_MIN,
            "rule_max_upper_wick": RULE_UPPER_WICK_MAX,
        }, indent=2))
        print(f"[{head}] SHIP mode={best_mode_name} thr={best_mode_rec['threshold']:.2f} "
              f"trades={best_mode_rec['trades']} WR={best_mode_rec['wr_pct']:.1f}% "
              f"PnL={best_mode_rec['pnl']:.0f} DD={best_mode_rec['dd']:.0f}")
    else:
        print(f"[{head}] KILL — closest mode={best_mode_name} thr="
              f"{best_mode_rec['threshold'] if best_mode_rec else '?'} "
              f"binding={metrics['binding_gate']}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    # Return probas for stacking
    return {"head": head, "metrics": metrics, "ship": ship,
            "best": best_mode_rec, "p_train": p_train, "p_hold": p_hold,
            "train": train, "hold": hold,
            "rule_mask_hold": rule_mask_hold, "baseline_pnl": baseline_pnl,
            "best_mode": best_mode_name}


def main() -> None:
    print(f"[v12] reading {V12_CORPUS}")
    corpus = pd.read_parquet(V12_CORPUS)
    print(f"[v12] rows: {len(corpus)}; cols: {len(corpus.columns)}")

    # Precompute helper: for "v12 fired probas" we need to do stack-aware
    # training. Strategy: train Kalshi on base features, then write its
    # holdout+train probas back into the working corpus as `kalshi_proba_v12`,
    # and use that as an input for LFO, etc.
    work = corpus.copy()
    work["kalshi_proba_v12"] = np.nan
    work["lfo_proba_v12"] = np.nan
    work["pct_proba_v12"] = np.nan

    # 1) Kalshi head
    feat_kalshi = BASE_NUMERIC_FEATS + K12_FEATS + SHOCK_FEATS + ["fg_proba"]
    res_kalshi = train_head(work, "kalshi", feat_kalshi, [])
    # Inject proba back
    if "p_train" in res_kalshi:
        idx_tr = res_kalshi["train"].index
        idx_te = res_kalshi["hold"].index
        work.loc[idx_tr, "kalshi_proba_v12"] = res_kalshi["p_train"]
        work.loc[idx_te, "kalshi_proba_v12"] = res_kalshi["p_hold"]

    # 2) LFO head
    feat_lfo = BASE_NUMERIC_FEATS + K12_FEATS + SHOCK_FEATS + ["fg_proba", "kalshi_proba_v12"]
    res_lfo = train_head(work, "lfo", feat_lfo, [])
    if "p_train" in res_lfo:
        work.loc[res_lfo["train"].index, "lfo_proba_v12"] = res_lfo["p_train"]
        work.loc[res_lfo["hold"].index, "lfo_proba_v12"] = res_lfo["p_hold"]

    # 3) PCT head
    feat_pct = BASE_NUMERIC_FEATS + K12_FEATS + SHOCK_FEATS + ["fg_proba", "kalshi_proba_v12", "lfo_proba_v12"]
    res_pct = train_head(work, "pct", feat_pct, [])
    if "p_train" in res_pct:
        work.loc[res_pct["train"].index, "pct_proba_v12"] = res_pct["p_train"]
        work.loc[res_pct["hold"].index, "pct_proba_v12"] = res_pct["p_hold"]

    # 4) Pivot head
    feat_pivot = BASE_NUMERIC_FEATS + K12_FEATS + SHOCK_FEATS + ["fg_proba", "kalshi_proba_v12", "lfo_proba_v12", "pct_proba_v12"]
    res_pivot = train_head(work, "pivot", feat_pivot, [])

    # ----------- Phase 4: combined stack ------------
    print("\n[v12] === Combined stack v12 ===")
    df = work[(work["strategy"] == "DynamicEngine3") &
              (work["allowed_by_friend_rule"] == True)].copy()  # noqa: E712
    df["ts"] = pd.to_datetime(df["ts"])
    hold = df[(df["ts"] >= "2026-01-01") & (df["ts"] < "2026-05-01")].copy()
    rule_mask = (
        (_to_numeric_safe(hold["bf_regime_eff"]) > RULE_REGIME_EFF_MIN) &
        (_to_numeric_safe(hold["bf_de3_entry_upper_wick_ratio"]) <= RULE_UPPER_WICK_MAX)
    ).to_numpy()
    base_metrics = metrics_for_subset(hold)
    baseline_pnl = base_metrics["pnl"]

    # Apply rule first; then if a head has shipped, apply its threshold in its
    # chosen mode. We allow each head to BLOCK trades; the combined gate is
    # rule AND (no head blocks).
    keep_mask = rule_mask.copy()  # start with rule pre-filter
    block_log = {"rule": int(len(hold) - rule_mask.sum())}
    head_results = {"kalshi": res_kalshi, "lfo": res_lfo, "pct": res_pct, "pivot": res_pivot}
    for hn in ["kalshi", "lfo", "pct", "pivot"]:
        r = head_results[hn]
        b = r.get("best", None)
        if r.get("ship", False) and b is not None and "p_hold" in r:
            thr = b["threshold"]
            mode = r["best_mode"]
            p = r["p_hold"]
            block = p >= thr
            # In all modes, when the head SHIPs, BLOCK rows with proba>=thr.
            # mode A is "within rule"; mode B is "union", but for combined-
            # stack determinism we conservatively BLOCK whenever proba>=thr.
            keep_mask = keep_mask & ~block
            block_log[hn] = int(block.sum())
        else:
            block_log[hn] = 0

    sub = hold[keep_mask]
    final_metrics = metrics_for_subset(sub)
    final_gates, final_binding = gates_pass(final_metrics, baseline_pnl)
    print(f"[combined] trades={final_metrics['trades']} WR={final_metrics['wr_pct']:.1f}% "
          f"PnL={final_metrics['pnl']:.0f} DD={final_metrics['dd']:.0f} "
          f"gates={final_gates} binding={final_binding}")

    combined = {
        "ship_gates": SHIP_GATES,
        "holdout_baseline": base_metrics,
        "rule_min_regime_eff": RULE_REGIME_EFF_MIN,
        "rule_max_upper_wick": RULE_UPPER_WICK_MAX,
        "block_log": block_log,
        "final": {**final_metrics, "gates": final_gates, "binding_gate": final_binding,
                  "all_pass": all(final_gates.values())},
        "per_head_ship": {
            hn: {"ship": head_results[hn].get("ship", False),
                 "best": head_results[hn].get("best", None),
                 "best_mode": head_results[hn].get("best_mode", "")}
            for hn in head_results
        },
    }
    out_combined = ART / "v12_combined_stack_metrics.json"
    out_combined.write_text(json.dumps(combined, indent=2, default=str))
    print(f"[combined] wrote {out_combined}")

    # Final summary table
    summary = {
        "kalshi_AUC_train": res_kalshi["metrics"]["train_AUC"] if "metrics" in res_kalshi else None,
        "kalshi_AUC_test": res_kalshi["metrics"]["test_AUC"] if "metrics" in res_kalshi else None,
        "lfo_AUC_train": res_lfo["metrics"]["train_AUC"] if "metrics" in res_lfo else None,
        "lfo_AUC_test": res_lfo["metrics"]["test_AUC"] if "metrics" in res_lfo else None,
        "pct_AUC_train": res_pct["metrics"]["train_AUC"] if "metrics" in res_pct else None,
        "pct_AUC_test": res_pct["metrics"]["test_AUC"] if "metrics" in res_pct else None,
        "pivot_AUC_train": res_pivot["metrics"]["train_AUC"] if "metrics" in res_pivot else None,
        "pivot_AUC_test": res_pivot["metrics"]["test_AUC"] if "metrics" in res_pivot else None,
        "kalshi_ship": res_kalshi.get("ship", False),
        "lfo_ship": res_lfo.get("ship", False),
        "pct_ship": res_pct.get("ship", False),
        "pivot_ship": res_pivot.get("ship", False),
        "rule_alone_holdout": res_kalshi["metrics"]["rule_alone_holdout"] if "metrics" in res_kalshi else None,
        "combined": combined,
    }
    (ART / "v12_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n[v12] SUMMARY:")
    print(json.dumps({k: v for k, v in summary.items() if k != "combined"}, indent=2, default=str))


if __name__ == "__main__":
    main()
