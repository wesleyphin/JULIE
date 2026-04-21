"""Backtest the new per-strategy G gate models.

For each strategy (AetherFlow, RegimeAdaptive):

  IN-SAMPLE:        score every trade with the model, apply veto threshold,
                    compute baseline vs G-gated PnL/wr/DD. Biased upward
                    (model saw these trades during training).

  TEMPORAL-OOS:     train on first 85% (sorted by entry_time), test on last
                    15%. Honest measure of "would G have helped on the most
                    recent trades?" Same split the trainer reports.

  WALK-FORWARD:     5 folds. Each fold: train on trades up to fold start,
                    score the next 1/5 of trades. Aggregate over all folds.
                    Most honest OOS estimate.

Usage:
  python scripts/signal_gate/backtest_g_models.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path("/Users/wes/Downloads/JULIE001")
PARQUET = Path("/Users/wes/Downloads/es_master_outrights-2.parquet")
ART_DIR = ROOT / "artifacts" / "signal_gate_2025"
NY = ZoneInfo("America/New_York")

sys.path.insert(0, str(ROOT / "scripts" / "signal_gate"))
sys.path.insert(0, str(ROOT / "tools"))
from train_per_strategy_models import (
    collect_strategy_trades,
    compute_features_for_trades,
    assemble_X,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ORDINAL_FEATURES,
)


def fmt_money(x: float) -> str:
    return f"${x:+,.2f}"


def compute_dd(trade_pnls: list[float]) -> float:
    cum = peak = dd = 0.0
    for p in trade_pnls:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd


def evaluate_gate(
    df: pd.DataFrame,
    p_big_loss: np.ndarray,
    threshold: float,
) -> dict:
    """Apply gate: veto trades where p_big_loss >= threshold.
    Return baseline vs gated stats."""
    veto = p_big_loss >= threshold
    pnls_all = df["pnl_dollars"].values.astype(float)
    pnls_kept = pnls_all[~veto]
    pnls_vetoed = pnls_all[veto]

    return {
        "n_total": int(len(df)),
        "n_vetoed": int(veto.sum()),
        "n_kept": int(len(df) - veto.sum()),
        "veto_rate": float(veto.mean()),
        "baseline_pnl": float(pnls_all.sum()),
        "gated_pnl": float(pnls_kept.sum()),
        "delta": float(pnls_kept.sum() - pnls_all.sum()),
        "baseline_wr": float((pnls_all > 0).mean()) if len(pnls_all) else 0.0,
        "gated_wr": float((pnls_kept > 0).mean()) if len(pnls_kept) else 0.0,
        "baseline_dd": compute_dd(list(pnls_all)),
        "gated_dd": compute_dd(list(pnls_kept)),
        "vetoed_pnl_avoided": float(pnls_vetoed.sum()),
        "vetoed_wr": float((pnls_vetoed > 0).mean()) if len(pnls_vetoed) else 0.0,
    }


def print_eval(label: str, ev: dict):
    print(f"\n  {label}:")
    print(f"    {'trades':<25} {ev['n_total']}")
    print(f"    {'vetoed':<25} {ev['n_vetoed']}  ({ev['veto_rate']:.1%})")
    print(f"    {'baseline pnl':<25} {fmt_money(ev['baseline_pnl'])}  wr={ev['baseline_wr']:.1%}  dd={fmt_money(ev['baseline_dd'])}")
    print(f"    {'gated pnl':<25} {fmt_money(ev['gated_pnl'])}  wr={ev['gated_wr']:.1%}  dd={fmt_money(ev['gated_dd'])}")
    print(f"    {'delta (gate help)':<25} {fmt_money(ev['delta'])}")
    print(f"    {'avoided in vetoed set':<25} {fmt_money(-ev['vetoed_pnl_avoided'])} (vetoed wr={ev['vetoed_wr']:.1%})")


def backtest_strategy(strategy: str, master_df: pd.DataFrame) -> dict:
    print(f"\n{'='*78}\nBACKTESTING {strategy} G GATE\n{'='*78}")
    # Match the trainer's filename mapping: DynamicEngine3 → de3
    _family_map = {
        "dynamicengine3": "de3", "aetherflow": "aetherflow",
        "regimeadaptive": "regimeadaptive", "mlphysics": "mlphysics",
    }
    fam = _family_map.get(strategy.lower(), strategy.lower())
    model_path = ART_DIR / f"model_{fam}.joblib"
    if not model_path.exists():
        print(f"  [SKIP] no model at {model_path}")
        return {}
    payload = joblib.load(model_path)
    threshold = float(payload["veto_threshold"])
    target = payload["target"]
    print(f"  model: {model_path.name}")
    print(f"  target={target}  veto_threshold={threshold}")

    trades = collect_strategy_trades(strategy)
    print(f"  trades collected: {len(trades)}")
    if len(trades) < 50:
        print(f"  [SKIP] too few trades")
        return {}

    df = compute_features_for_trades(trades, master_df)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES + [target])
    df = df.sort_values("entry_time").reset_index(drop=True)
    print(f"  feature rows: {len(df)}")

    cat_maps = payload.get("categorical_maps")
    X, _ = assemble_X(df, cat_maps=cat_maps)
    feat_names = payload["feature_names"]
    # Align column order to model
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feat_names]

    # ---- IN-SAMPLE ----
    p_in = payload["model"].predict_proba(X)[:, 1]
    in_sample = evaluate_gate(df, p_in, threshold)
    print_eval("IN-SAMPLE (biased — model trained on these)", in_sample)

    # ---- TEMPORAL-OOS (last 15%) ----
    split = max(50, int(0.85 * len(df)))
    if len(df) - split < 20:
        split = max(1, len(df) - 20)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr = df.iloc[:split][target].astype(int).values
    df_te = df.iloc[split:].reset_index(drop=True)
    if y_tr.sum() < 5 or (1-y_tr).sum() < 5:
        print("  [skip OOS — insufficient class balance in train portion]")
        oos = None
    else:
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_samples_leaf=20, random_state=42,
        )
        clf.fit(X_tr, y_tr)
        p_oos = clf.predict_proba(X_te)[:, 1]
        oos = evaluate_gate(df_te, p_oos, threshold)
        print_eval(f"TEMPORAL-OOS (last 15%, n={len(df_te)})", oos)

    # ---- WALK-FORWARD (5 folds) ----
    n_folds = 5
    fold_size = len(df) // n_folds
    if fold_size < 20:
        print("  [skip walk-forward — too few trades for 5 folds]")
        walk = None
    else:
        all_pnl_baseline = 0.0
        all_pnl_gated = 0.0
        all_n_kept = 0
        all_n_vetoed = 0
        all_kept_pnls = []
        for fold in range(1, n_folds):
            tr_end = fold * fold_size
            te_start = tr_end
            te_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(df)
            X_tr_f = X.iloc[:tr_end]
            y_tr_f = df.iloc[:tr_end][target].astype(int).values
            df_te_f = df.iloc[te_start:te_end].reset_index(drop=True)
            if y_tr_f.sum() < 5 or (1-y_tr_f).sum() < 5 or len(df_te_f) == 0:
                continue
            X_te_f = X.iloc[te_start:te_end]
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                min_samples_leaf=20, random_state=42,
            )
            clf.fit(X_tr_f, y_tr_f)
            p_f = clf.predict_proba(X_te_f)[:, 1]
            ev_f = evaluate_gate(df_te_f, p_f, threshold)
            all_pnl_baseline += ev_f["baseline_pnl"]
            all_pnl_gated += ev_f["gated_pnl"]
            all_n_kept += ev_f["n_kept"]
            all_n_vetoed += ev_f["n_vetoed"]
            all_kept_pnls.extend(df_te_f[~(p_f >= threshold)]["pnl_dollars"].values.tolist())
        walk = {
            "n_total": all_n_kept + all_n_vetoed,
            "n_vetoed": all_n_vetoed,
            "n_kept": all_n_kept,
            "veto_rate": all_n_vetoed / max(1, all_n_kept + all_n_vetoed),
            "baseline_pnl": all_pnl_baseline,
            "gated_pnl": all_pnl_gated,
            "delta": all_pnl_gated - all_pnl_baseline,
            "baseline_wr": float("nan"),
            "gated_wr": float(np.mean([1.0 if p > 0 else 0.0 for p in all_kept_pnls])) if all_kept_pnls else 0.0,
            "baseline_dd": float("nan"),
            "gated_dd": compute_dd(all_kept_pnls),
            "vetoed_pnl_avoided": all_pnl_baseline - all_pnl_gated,
            "vetoed_wr": float("nan"),
        }
        print(f"\n  WALK-FORWARD (5 folds, OOS prediction at each step):")
        print(f"    {'trades scored OOS':<25} {walk['n_total']}")
        print(f"    {'vetoed':<25} {walk['n_vetoed']}  ({walk['veto_rate']:.1%})")
        print(f"    {'baseline pnl':<25} {fmt_money(walk['baseline_pnl'])}")
        print(f"    {'gated pnl':<25} {fmt_money(walk['gated_pnl'])}  wr={walk['gated_wr']:.1%}  dd={fmt_money(walk['gated_dd'])}")
        print(f"    {'delta (gate help)':<25} {fmt_money(walk['delta'])}")

    return {"in_sample": in_sample, "temporal_oos": oos, "walk_forward": walk,
            "threshold": threshold, "n_trades": len(df)}


def main():
    print(f"[load] master parquet for feature compute")
    master = pd.read_parquet(PARQUET)
    master = master[master.index >= "2025-01-01"]
    print(f"  rows={len(master):,}")

    results = {}
    for strat in ("DynamicEngine3", "AetherFlow", "RegimeAdaptive"):
        results[strat] = backtest_strategy(strat, master)

    # ---- Final summary ----
    print(f"\n\n{'='*78}\nFINAL SUMMARY — G GATE EFFECT PER STRATEGY\n{'='*78}")
    print(f"\n{'strategy':<18} {'thr':>6} {'view':<14} {'n':>5} {'veto%':>7} {'baseline':>12} {'gated':>12} {'delta':>10}")
    print("-" * 90)
    for strat, r in results.items():
        thr = r.get("threshold", float("nan"))
        for label, ev in (("in-sample", r.get("in_sample")), ("temporal-oos", r.get("temporal_oos")), ("walk-forward", r.get("walk_forward"))):
            if not ev: continue
            print(f"{strat:<18} {thr:>6.3f} {label:<14} {ev['n_total']:>5} {ev['veto_rate']*100:>6.1f}% "
                  f"{fmt_money(ev['baseline_pnl']):>12} {fmt_money(ev['gated_pnl']):>12} "
                  f"{fmt_money(ev['delta']):>10}")
    print()

    out = ROOT / "backtest_reports" / "g_gate_backtest.json"
    out.write_text(json.dumps(results, default=str, indent=2))
    print(f"[write] {out}")


if __name__ == "__main__":
    main()
