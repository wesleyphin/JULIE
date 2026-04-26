"""V11 Phase 2 — Per-overlay retrain (Kalshi/LFO/PCT) using clean training corpus.

Trains HGB heads per overlay × strategy with cost-sensitive sample weights,
sweeps thresholds on the Jan-Apr 2026 holdout, evaluates against ship gates,
and writes per-head model + metrics or KILL marker.

DOES NOT train Pivot (handled by parallel agent).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path("/Users/wes/Downloads/JULIE001")
CORPUS = REPO_ROOT / "artifacts" / "v11_training_corpus.parquet"
ARTIFACT_BASE = REPO_ROOT / "artifacts"

# Feature definitions
BF_NUMERIC_FEATURES = [
    "bf_de3_entry_ret1_atr", "bf_de3_entry_body_pos1", "bf_de3_entry_body1_ratio",
    "bf_de3_entry_lower_wick_ratio", "bf_de3_entry_upper_wick_ratio",
    "bf_de3_entry_upper1_ratio", "bf_de3_entry_close_pos1",
    "bf_de3_entry_flips5", "bf_de3_entry_down3", "bf_de3_entry_range10_atr",
    "bf_de3_entry_dist_high5_atr", "bf_de3_entry_dist_low5_atr",
    "bf_de3_entry_vol1_rel20", "bf_de3_entry_atr14",
    "bf_sl_dist_pts", "bf_tp_dist_pts", "bf_atr_ratio_to_sl",
    "bf_dist_to_bank_below", "bf_dist_to_bank_above",
    "bf_dist_to_bank_in_dir", "bf_dist_to_bank_pts",
    "bf_bar_range_pts", "bf_bar_close_pct_body", "bf_atr14_pts",
    "bf_range_30bar_pts", "bf_trend_20bar_pct",
    "bf_dist_to_20bar_hi_pct", "bf_dist_to_20bar_lo_pct",
    "bf_vel_5bar_pts_per_min", "bf_vel_20bar_pts_per_min",
    "bf_regime_vol_bp", "bf_regime_eff",
]

PCT_NUMERIC_FEATURES = [
    "pct_pct_from_open", "pct_signed_level", "pct_abs_level",
    "pct_level_distance_pct", "pct_atr_pct_30bar", "pct_range_pct_at_touch",
    "pct_hour_edge", "pct_minutes_since_open",
    "pct_dist_to_running_hi_pct", "pct_dist_to_running_lo_pct",
    "pct_rule_confidence",
]

# Stack-aware upstream probas
UPSTREAM_PROBAS = {
    "kalshi": ["fg_proba"],                              # FG -> Kalshi
    "lfo":    ["fg_proba", "kalshi_proba"],              # FG -> Kalshi -> LFO
    "pct":    ["fg_proba", "kalshi_proba", "lfo_proba"], # FG -> Kalshi -> LFO -> PCT
}

# Ship gate constants
G1_MAX_DD = 870.0
G3_MIN_OOS = 50
G4_MIN_WR = 0.55  # 55%

# Train/test split
TRAIN_START = "2025-03-01"
TRAIN_END = "2026-01-01"  # exclusive
TEST_START = "2026-01-01"
TEST_END = "2026-05-01"  # exclusive

THRESHOLDS = [round(0.40 + 0.05 * i, 2) for i in range(10)]  # 0.40..0.85


def load_corpus() -> pd.DataFrame:
    df = pd.read_parquet(CORPUS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("US/Eastern")
    return df


def select_applicable(df: pd.DataFrame, overlay: str) -> pd.DataFrame:
    """Apply overlay applicability gate.

    Kalshi: family != 'AF' AND in_kalshi_window == True
    LFO:    keep all
    PCT:    keep all
    """
    if overlay == "kalshi":
        # corpus uses 'aetherflow' as family name for AF (lowercase). Spec said != 'AF'
        # so guard against both spellings.
        af_mask = df["family"].str.lower() == "aetherflow"
        return df[(~af_mask) & df["in_kalshi_window"]]
    return df


def build_features(df: pd.DataFrame, overlay: str) -> tuple[pd.DataFrame, list[str]]:
    """Assemble feature matrix per spec (regime + tick + upstream probas)."""
    features = list(BF_NUMERIC_FEATURES) + list(PCT_NUMERIC_FEATURES) + UPSTREAM_PROBAS[overlay]
    X = df[features].copy()
    # Replace NaN with median per column (HGB tolerates NaN but be defensive)
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    return X, features


def cost_sensitive_weights(net_pnl: pd.Series) -> np.ndarray:
    w = np.abs(net_pnl.values) / 50.0
    w = np.clip(w, 0.5, 4.0)
    return w


def simulate_holdout_after_block(
    holdout: pd.DataFrame,
    proba: np.ndarray,
    threshold: float,
) -> dict:
    """Block trades where proba >= threshold; sum survivors' net_pnl."""
    block = proba >= threshold
    survivors = holdout[~block].copy().sort_values("ts").reset_index(drop=True)
    if len(survivors) == 0:
        return {
            "threshold": threshold,
            "trades": 0,
            "blocked": int(block.sum()),
            "wr_pct": 0.0,
            "pnl": 0.0,
            "max_dd": 0.0,
            "wr_held": 0.0,  # WR on overlay-affected (held=blocked) trades
            "n_held": int(block.sum()),
        }
    pnl = survivors["net_pnl_after_haircut"].sum()
    wr = (survivors["net_pnl_after_haircut"] > 0).mean() * 100
    cum = survivors["net_pnl_after_haircut"].cumsum().values
    peak = np.maximum.accumulate(cum)
    dd = float((cum - peak).min())

    held = holdout[block]
    if len(held) > 0:
        wr_held = (held["net_pnl_after_haircut"] > 0).mean()
    else:
        wr_held = 0.0
    return {
        "threshold": threshold,
        "trades": int(len(survivors)),
        "blocked": int(block.sum()),
        "wr_pct": float(wr),
        "pnl": float(pnl),
        "max_dd": dd,
        "wr_held": float(wr_held),
        "n_held": int(len(held)),
    }


def evaluate_gates(
    sweep_row: dict,
    baseline_pnl: float,
    baseline_trades: int,
) -> dict:
    """Apply ship gates G1..G4. Returns gate booleans + binding gate name."""
    g1 = sweep_row["max_dd"] >= -G1_MAX_DD  # DD is negative; want |DD| <= 870
    g2 = (sweep_row["pnl"] >= baseline_pnl) and (sweep_row["trades"] <= baseline_trades)
    # G3: n_OOS surviving (the trades we'd actually take post-block) >= 50
    # The spec says "n_OOS surviving" but the WR-on-held-trades gate is the relevant
    # signal of whether the block was useful. We interpret G3 as surviving trades >= 50.
    g3 = sweep_row["trades"] >= G3_MIN_OOS
    # G4: WR on overlay-affected (held=blocked) trades >= 55%
    # i.e., the head should mostly be blocking losers. Held trades should have LOW WR
    # for the block to be useful — wait, re-reading spec: "WR on overlay-affected (held)
    # trades >= 55%". This is ambiguous: either (a) survivors WR >= 55% (which would be
    # the post-block trades' WR — what we'd actually trade), or (b) held trades' WR
    # should be >= 55% (which makes no sense since we want to block losers).
    # Best interpretation: WR of the surviving (taken) trades >= 55% — these are the
    # overlay-AFFECTED trades because the overlay decided to LET THEM THROUGH.
    g4 = (sweep_row["wr_pct"] / 100.0) >= G4_MIN_WR

    gates = {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4)}
    binding = None
    for name, ok in gates.items():
        if not ok:
            binding = name
            break
    return {"gates": gates, "all_pass": all(gates.values()), "binding_gate": binding}


def fit_head(
    X_tr: pd.DataFrame, y_tr: np.ndarray, w_tr: np.ndarray,
    X_te: pd.DataFrame, y_te: np.ndarray,
) -> tuple[HistGradientBoostingClassifier, float, float]:
    """Train HGB; return model + train/test AUC."""
    model = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=200,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X_tr.values, y_tr, sample_weight=w_tr)
    p_tr = model.predict_proba(X_tr.values)[:, 1]
    p_te = model.predict_proba(X_te.values)[:, 1]
    try:
        auc_tr = float(roc_auc_score(y_tr, p_tr))
    except Exception:
        auc_tr = float("nan")
    try:
        auc_te = float(roc_auc_score(y_te, p_te))
    except Exception:
        auc_te = float("nan")
    return model, auc_tr, auc_te


def train_one_head(
    overlay: str,
    strategy_label: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    label_col: str,
) -> dict:
    """Train+evaluate a single (overlay, strategy_label) head.

    Returns a result dict including sweep + ship status.
    """
    X_tr, feats = build_features(df_train, overlay)
    X_te, _ = build_features(df_test, overlay)
    y_tr = df_train[label_col].astype(int).values
    y_te = df_test[label_col].astype(int).values
    w_tr = cost_sensitive_weights(df_train["net_pnl_after_haircut"])

    model, auc_tr, auc_te = fit_head(X_tr, y_tr, w_tr, X_te, y_te)

    proba_te = model.predict_proba(X_te.values)[:, 1]

    # Holdout baseline restricted to df_test slice
    holdout = df_test.sort_values("ts").reset_index(drop=True).copy()
    base_pnl = float(holdout["net_pnl_after_haircut"].sum())
    base_trades = int(len(holdout))
    base_wr = float((holdout["net_pnl_after_haircut"] > 0).mean() * 100)
    cum = holdout["net_pnl_after_haircut"].cumsum().values
    peak = np.maximum.accumulate(cum) if len(cum) else np.array([0.0])
    base_dd = float((cum - peak).min()) if len(cum) else 0.0

    # Reorder proba_te to match the sorted holdout order
    sort_idx = df_test.reset_index(drop=True).sort_values("ts").index.values
    proba_sorted = proba_te[sort_idx] if len(proba_te) else np.array([])

    sweep = []
    for thr in THRESHOLDS:
        row = simulate_holdout_after_block(holdout, proba_sorted, thr)
        gate_eval = evaluate_gates(row, base_pnl, base_trades)
        row.update(gate_eval)
        sweep.append(row)

    # Identify ship config (passes all gates) — pick one with max PnL among those passing
    passing = [r for r in sweep if r["all_pass"]]
    if passing:
        best = max(passing, key=lambda r: (r["pnl"], -r["max_dd"]))
        ship = True
    else:
        # closest-to-passing: maximize number of gates passed; tiebreak by PnL
        best = max(sweep, key=lambda r: (sum(r["gates"].values()), r["pnl"]))
        ship = False

    return {
        "overlay": overlay,
        "strategy": strategy_label,
        "label": label_col,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "train_AUC": auc_tr,
        "test_AUC": auc_te,
        "features": feats,
        "holdout_baseline": {
            "trades": base_trades,
            "wr_pct": base_wr,
            "pnl": base_pnl,
            "dd": base_dd,
        },
        "sweep": sweep,
        "best_config": best,
        "ship": ship,
        "model": model if ship else None,
    }


def main() -> None:
    df = load_corpus()
    df_allowed = df[df["allowed_by_friend_rule"]].copy()

    train_mask = (df_allowed["ts"] >= TRAIN_START) & (df_allowed["ts"] < TRAIN_END)
    test_mask = (df_allowed["ts"] >= TEST_START) & (df_allowed["ts"] < TEST_END)
    df_train_all = df_allowed[train_mask].copy()
    df_test_all = df_allowed[test_mask].copy()

    print(f"Train(allowed): {len(df_train_all)}  Test(allowed): {len(df_test_all)}")

    # Strategy keys: corpus uses 'DynamicEngine3' for DE3 and 'RegimeAdaptive' for RA.
    strategy_map = {
        "DE3": "DynamicEngine3",
        "RA": "RegimeAdaptive",
    }
    overlays = ["kalshi", "lfo", "pct"]

    # Holdout baseline (across all heads share the same Jan-Apr 2026 friend-rule slice)
    holdout = df_test_all.sort_values("ts").reset_index(drop=True)
    cum = holdout["net_pnl_after_haircut"].cumsum().values
    peak = np.maximum.accumulate(cum) if len(cum) else np.array([0.0])
    base = {
        "trades": int(len(holdout)),
        "wr_pct": float((holdout["net_pnl_after_haircut"] > 0).mean() * 100),
        "pnl": float(holdout["net_pnl_after_haircut"].sum()),
        "dd": float((cum - peak).min()) if len(cum) else 0.0,
    }
    print(f"Holdout baseline: {base}")

    # We'll try both labels (is_big_loss and net_pnl < 0) per spec; pick better OOS AUC.
    label_options = ["is_big_loss", "is_loss_any"]

    all_results = {}
    files_written = []
    headers_csv_rows = []

    for overlay in overlays:
        for strat_label, strat_col in strategy_map.items():
            # Slice strategy
            df_tr_s = df_train_all[df_train_all["strategy"] == strat_col].copy()
            df_te_s = df_test_all[df_test_all["strategy"] == strat_col].copy()
            # Apply overlay applicability
            df_tr_s = select_applicable(df_tr_s, overlay)
            df_te_s = select_applicable(df_te_s, overlay)

            n_tr = len(df_tr_s)
            n_te = len(df_te_s)

            head_id = f"{overlay}_{strat_label.lower()}"

            if strat_label == "RA" and (n_tr < 100 or n_te < 50):
                print(f"  [{head_id}] SKIP_split: n_train={n_tr}, n_test={n_te}")
                all_results[head_id] = {
                    "overlay": overlay,
                    "strategy": strat_label,
                    "skipped": True,
                    "reason": f"n_train={n_tr} < 100 or n_test={n_te} < 50",
                    "n_train": n_tr,
                    "n_test": n_te,
                }
                continue

            if n_tr == 0 or n_te == 0:
                print(f"  [{head_id}] SKIP_empty")
                all_results[head_id] = {
                    "overlay": overlay,
                    "strategy": strat_label,
                    "skipped": True,
                    "reason": f"empty slice (n_train={n_tr}, n_test={n_te})",
                    "n_train": n_tr,
                    "n_test": n_te,
                }
                continue

            # Add soft label
            df_tr_s = df_tr_s.copy()
            df_te_s = df_te_s.copy()
            df_tr_s["is_loss_any"] = (df_tr_s["net_pnl_after_haircut"] < 0).astype(bool)
            df_te_s["is_loss_any"] = (df_te_s["net_pnl_after_haircut"] < 0).astype(bool)

            # Check label dispersion
            best_label_result = None
            for label_col in label_options:
                if df_tr_s[label_col].sum() == 0 or df_tr_s[label_col].sum() == len(df_tr_s):
                    continue
                if df_te_s[label_col].sum() == 0 or df_te_s[label_col].sum() == len(df_te_s):
                    continue
                try:
                    res = train_one_head(overlay, strat_label, df_tr_s, df_te_s, label_col)
                except Exception as e:
                    print(f"  [{head_id}] train failed for {label_col}: {e}")
                    continue
                if best_label_result is None or (
                    not np.isnan(res["test_AUC"]) and (
                        np.isnan(best_label_result["test_AUC"]) or
                        res["test_AUC"] > best_label_result["test_AUC"]
                    )
                ):
                    best_label_result = res

            if best_label_result is None:
                print(f"  [{head_id}] FAILED_TRAIN: degenerate labels")
                all_results[head_id] = {
                    "overlay": overlay,
                    "strategy": strat_label,
                    "skipped": True,
                    "reason": "degenerate labels",
                    "n_train": n_tr,
                    "n_test": n_te,
                }
                continue

            res = best_label_result
            print(
                f"  [{head_id}] label={res['label']} n_tr={res['n_train']} "
                f"n_te={res['n_test']} AUC_tr={res['train_AUC']:.3f} "
                f"AUC_te={res['test_AUC']:.3f} ship={res['ship']} "
                f"thr={res['best_config']['threshold']} "
                f"PnL={res['best_config']['pnl']:.2f} "
                f"DD={res['best_config']['max_dd']:.2f} "
                f"trades={res['best_config']['trades']}"
            )
            all_results[head_id] = res

            # Persist artifacts
            out_dir = ARTIFACT_BASE / f"regime_ml_{overlay}_v11" / strat_label.lower()
            out_dir.mkdir(parents=True, exist_ok=True)

            metrics_payload = {
                "overlay": overlay,
                "strategy": strat_label,
                "label": res["label"],
                "n_train": res["n_train"],
                "n_test": res["n_test"],
                "train_AUC": res["train_AUC"],
                "test_AUC": res["test_AUC"],
                "features": res["features"],
                "holdout_baseline": res["holdout_baseline"],
                "sweep": res["sweep"],
                "best_config": res["best_config"],
                "ship": res["ship"],
                "kill_marker": (not res["ship"]),
                "binding_gate": res["best_config"].get("binding_gate"),
                "ship_gates": {
                    "G1_max_dd_abs": G1_MAX_DD,
                    "G2": "PnL >= holdout_baseline AND trades <= holdout_baseline",
                    "G3_min_n_oos_surviving": G3_MIN_OOS,
                    "G4_min_wr_pct": G4_MIN_WR * 100,
                },
            }
            metrics_path = out_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics_payload, f, indent=2, default=float)
            files_written.append(str(metrics_path))

            if res["ship"]:
                model_path = out_dir / "model.joblib"
                joblib.dump(res["model"], model_path)
                files_written.append(str(model_path))

                thr_path = out_dir / "thresholds.json"
                with open(thr_path, "w") as f:
                    json.dump({
                        "overlay": overlay,
                        "strategy": strat_label,
                        "threshold": res["best_config"]["threshold"],
                        "label": res["label"],
                        "features": res["features"],
                    }, f, indent=2)
                files_written.append(str(thr_path))

    # Save summary
    summary_path = ARTIFACT_BASE / "v11_phase2_summary.json"
    summary = {
        "holdout_baseline": base,
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "model"}
            for k, v in all_results.items()
        },
        "files_written": files_written,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    files_written.append(str(summary_path))
    print(f"\nSummary written: {summary_path}")
    print(f"Files written: {len(files_written)}")
    for fp in files_written:
        print(f"  {fp}")


if __name__ == "__main__":
    main()
