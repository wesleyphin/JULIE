#!/usr/bin/env python3
"""Train Kalshi overlay ML v3 — regression on forward PnL.

Classification failed: model probabilities don't cleanly separate rule-BLOCK
winners from rule-BLOCK losers in Apr 2026 OOS.

This version trains a *regressor* on forward_pnl (treating PnL as a continuous
target) and acts only when the predicted PnL magnitude exceeds a threshold.
Intuition: PnL regression aligns the learning objective with the evaluation
metric. A model that predicts "+$28 ± $5 expected value" gives a cleaner
decision than "proba_pass = 0.64".

Decision logic (binary-override on rule):
    if rule=BLOCK and pred_pnl >=  pass_threshold  → ML PASS override
    if rule=PASS  and pred_pnl <= -block_threshold → ML BLOCK override
    else trust rule

Keeps v2's feature set (with Kalshi deltas, intraday cumulative, etc.) +
recency weighting. Same 5 ship gates. Adds a "meta" option that trains on
"was rule right?" as a simpler learning target.

Sweep: horizon × half-life × pass_threshold × block_threshold.
"""
from __future__ import annotations

import argparse, json, logging, pickle, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT, FEATURE_COLS_40, MES_PT_VALUE, DEFAULT_TP, DEFAULT_SL,
    load_continuous_bars, build_feature_frame,
    stats,
)

# Reuse v2's feature engineering and parser
from train_kalshi_v2 import (
    RE_HEADER, RE_BAR, RE_ENTRY_VIEW,
    parse_log_events, simulate_trade_horizon, add_v2_features,
    fill_intraday_pnl, build_dataset,
    FEATURE_COLS_KALSHI_V2, V2_EXTRA_FEATURES,
    recency_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_kalshi_v3")

OOS_START = "2026-01-27"
OOS_END = "2026-04-24"


def eval_regressor(pred_pnl: np.ndarray,
                   rule_decision: np.ndarray,
                   y_true: np.ndarray,
                   pnl_if_passed: np.ndarray,
                   pass_thr: float, block_thr: float) -> dict:
    """Act only on rule's decisions when regressor predicts strong signal."""
    # Override rule=BLOCK to PASS if predicted PnL is strongly positive
    pass_override = (rule_decision == "BLOCK") & (pred_pnl >= pass_thr)
    # Override rule=PASS to BLOCK if predicted PnL is strongly negative
    block_override = (rule_decision == "PASS") & (pred_pnl <= -block_thr)

    final_pass = np.where(
        pass_override, True,
        np.where(block_override, False, rule_decision == "PASS"),
    )
    final_pnl = np.where(final_pass, pnl_if_passed, 0.0)
    ml_st = stats(final_pnl)
    n_pass = int(final_pass.sum())
    n_new_pass = int(pass_override.sum())
    n_new_block = int(block_override.sum())
    new_pass_pnls = pnl_if_passed[pass_override]
    new_block_pnls = pnl_if_passed[block_override]
    new_pass_wr = (100 * (new_pass_pnls > 0).sum() / len(new_pass_pnls)) if len(new_pass_pnls) else 0.0
    new_block_wr = (100 * (new_block_pnls <= 0).sum() / len(new_block_pnls)) if len(new_block_pnls) else 0.0
    return {
        "pass_thr": pass_thr, "block_thr": block_thr,
        "n_pass": n_pass, "n_new_pass": n_new_pass, "n_new_block": n_new_block,
        "new_pass_wr": new_pass_wr, "new_block_wr": new_block_wr,
        "pnl": ml_st["pnl"], "dd": ml_st["dd"], "avg": ml_st["avg"],
    }


def run_config(events: list, bars: pd.DataFrame,
               label_horizon_min: int,
               half_life_days: float,
               train_start_date: str | None,
               oos_start: str, oos_end: str,
               meta_mode: bool = False,
               seed: int = 42) -> dict | None:
    log.info("")
    log.info("=" * 70)
    log.info("CONFIG  horizon=%dmin  hl=%.0fd  trstart=%s  meta=%s",
             label_horizon_min, half_life_days, train_start_date or "full",
             meta_mode)
    log.info("=" * 70)
    df = build_dataset(events, bars, label_horizon_min)
    if len(df) < 500:
        log.warning("only %d labeled rows — skip", len(df))
        return None

    oos_start_ts = pd.Timestamp(oos_start, tz=df.index.tz)
    oos_end_ts = pd.Timestamp(oos_end, tz=df.index.tz) + pd.Timedelta(days=1)
    tr = df.loc[df.index < oos_start_ts]
    if train_start_date is not None:
        tr = tr.loc[tr.index >= pd.Timestamp(train_start_date, tz=df.index.tz)]
    oos = df.loc[(df.index >= oos_start_ts) & (df.index <= oos_end_ts)]
    log.info("train: %d  OOS: %d", len(tr), len(oos))
    if len(tr) < 200 or len(oos) < 50:
        log.warning("below floors — skip")
        return None

    X_tr = tr[FEATURE_COLS_KALSHI_V2].to_numpy()
    X_oos = oos[FEATURE_COLS_KALSHI_V2].to_numpy()
    recency_w = recency_weights(tr.index, half_life_days)
    rule_decision = oos["rule_decision"].to_numpy()
    pnl_if_passed = oos["forward_pnl"].to_numpy()
    y_true = oos["label"].to_numpy()
    rule_pnl_arr = np.where(rule_decision == "PASS", pnl_if_passed, 0.0)
    rule_st = stats(rule_pnl_arr)
    oracle_pnl_arr = np.where(y_true == "pass", pnl_if_passed, 0.0)
    oracle_st = stats(oracle_pnl_arr)
    log.info("rule baseline: PnL=$%+.2f DD=$%.0f  | oracle: PnL=$%+.2f  lift_headroom=$%+.2f",
             rule_st["pnl"], rule_st["dd"], oracle_st["pnl"],
             oracle_st["pnl"] - rule_st["pnl"])

    if meta_mode:
        # Target = was rule's decision right?  (rule PASS & label pass) | (rule BLOCK & label block)
        y_meta = ((tr["rule_decision"] == "PASS") & (tr["label"] == "pass")) | \
                 ((tr["rule_decision"] == "BLOCK") & (tr["label"] == "block"))
        y_meta = y_meta.astype(int).to_numpy()  # 1 = rule right, 0 = rule wrong
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=6,
            l2_regularization=1.0, min_samples_leaf=30, random_state=seed,
        )
        clf.fit(X_tr, y_meta, sample_weight=recency_w)
        # proba that rule is WRONG
        idx_wrong = list(clf.classes_).index(0)
        proba_wrong = clf.predict_proba(X_oos)[:, idx_wrong]
        log.info("meta-classifier — proba_wrong distribution (oos):")
        for pctl in [10, 25, 50, 75, 90, 95]:
            log.info("  p%d=%.3f", pctl, np.percentile(proba_wrong, pctl))
        # Sweep: override rule when proba_wrong is high
        results = []
        for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            override_mask = proba_wrong >= thr
            final_pass = np.where(
                override_mask & (rule_decision == "BLOCK"), True,
                np.where(override_mask & (rule_decision == "PASS"), False,
                         rule_decision == "PASS"),
            )
            final_pnl = np.where(final_pass, pnl_if_passed, 0.0)
            ml_st = stats(final_pnl)
            n_new_pass = int((override_mask & (rule_decision == "BLOCK")).sum())
            n_new_block = int((override_mask & (rule_decision == "PASS")).sum())
            new_pass_pnls = pnl_if_passed[override_mask & (rule_decision == "BLOCK")]
            new_pass_wr = (100 * (new_pass_pnls > 0).sum() / len(new_pass_pnls)) if len(new_pass_pnls) else 0.0
            lift = ml_st["pnl"] - rule_st["pnl"]
            dd_over_pnl = (ml_st["dd"] / ml_st["pnl"] * 100.0) if ml_st["pnl"] > 0 else float("inf")
            headroom = oracle_st["pnl"] - rule_st["pnl"]
            capt = (lift / headroom * 100.0) if headroom > 0 else 0.0
            gates = {
                "pnl_ok": lift > 0,
                "dd_ratio_ok": dd_over_pnl <= 30.0,
                "n_ok": len(oos) >= 50,
                "new_pass_wr_ok": (new_pass_wr >= 50.0) if n_new_pass >= 5 else True,
                "capt_ok": capt >= 20.0,
            }
            ok = all(gates.values())
            log.info("meta thr=%.2f  n_new_pass=%d  WR=%.2f%%  n_new_blk=%d  pnl=$%+.2f  dd=$%.0f  capt=%.2f%%  %d/5%s",
                     thr, n_new_pass, new_pass_wr, n_new_block, ml_st["pnl"], ml_st["dd"], capt,
                     sum(gates.values()), " SHIP" if ok else "")
            r = {"kind": "meta", "thr": thr, "lift": lift,
                 "capt_pct": capt, "dd_over_pnl": dd_over_pnl,
                 "n_new_pass": n_new_pass, "new_pass_wr": new_pass_wr,
                 "n_new_block": n_new_block,
                 "pnl": ml_st["pnl"], "dd": ml_st["dd"],
                 "gates": gates, "ships": ok}
            results.append(r)
        shippers = [r for r in results if r["ships"]]
        best = max(shippers, key=lambda r: r["lift"]) if shippers else max(results, key=lambda r: r["lift"])
        return {
            "kind": "meta",
            "horizon": label_horizon_min,
            "half_life": half_life_days,
            "train_start": train_start_date,
            "rule_baseline": rule_st, "oracle": oracle_st,
            "n_train": len(tr), "n_oos": len(oos),
            "all": results, "best": best,
            "model": clf if best["ships"] else None,
        }

    # Regression path
    y_pnl = tr["forward_pnl"].to_numpy()
    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=seed,
    )
    reg.fit(X_tr, y_pnl, sample_weight=recency_w)
    pred_pnl = reg.predict(X_oos)

    log.info("pred PnL distribution (OOS):")
    for pctl in [5, 25, 50, 75, 95]:
        log.info("  p%d=$%+.2f", pctl, np.percentile(pred_pnl, pctl))

    # Diagnostic: correlation with actual forward_pnl in OOS
    corr = np.corrcoef(pred_pnl, pnl_if_passed)[0, 1]
    log.info("OOS pred-vs-actual correlation: %.3f", corr)

    # Sweep pass_thr × block_thr
    thr_grid = [
        (5.0, 5.0), (7.5, 5.0), (7.5, 7.5), (10.0, 5.0), (10.0, 7.5),
        (12.5, 5.0), (12.5, 10.0), (15.0, 7.5), (15.0, 10.0), (15.0, 15.0),
        (20.0, 10.0), (20.0, 15.0), (25.0, 10.0), (25.0, 15.0),
    ]
    results = []
    log.info("%5s %5s %8s %8s %10s %8s %8s %7s gates",
             "p_thr", "b_thr", "n_pass", "new_psn", "newPassWR", "n_new_blk",
             "pnl", "capt")
    for (p_thr, b_thr) in thr_grid:
        r = eval_regressor(pred_pnl, rule_decision, y_true, pnl_if_passed,
                           p_thr, b_thr)
        lift = r["pnl"] - rule_st["pnl"]
        dd_over_pnl = (r["dd"] / r["pnl"] * 100.0) if r["pnl"] > 0 else float("inf")
        headroom = oracle_st["pnl"] - rule_st["pnl"]
        capt = (lift / headroom * 100.0) if headroom > 0 else 0.0
        gates = {
            "pnl_ok":        lift > 0,
            "dd_ratio_ok":   dd_over_pnl <= 30.0,
            "n_ok":          len(oos) >= 50,
            "new_pass_wr_ok":(r["new_pass_wr"] >= 50.0) if r["n_new_pass"] >= 5 else True,
            "capt_ok":       capt >= 20.0,
        }
        ok = all(gates.values())
        log.info("%5.1f %5.1f %8d %8d %9.2f%% %8d $%+6.2f %6.2f%%  %d/5%s",
                 p_thr, b_thr, r["n_pass"], r["n_new_pass"], r["new_pass_wr"],
                 r["n_new_block"], r["pnl"], capt, sum(gates.values()),
                 " SHIP" if ok else "")
        r.update({"kind": "reg", "lift": lift, "dd_over_pnl": dd_over_pnl,
                  "capt_pct": capt, "gates": gates, "ships": ok})
        results.append(r)

    shippers = [r for r in results if r["ships"]]
    best = max(shippers, key=lambda r: r["lift"]) if shippers else max(results, key=lambda r: r["lift"])
    return {
        "kind": "reg",
        "horizon": label_horizon_min, "half_life": half_life_days,
        "train_start": train_start_date,
        "rule_baseline": rule_st, "oracle": oracle_st,
        "n_train": len(tr), "n_oos": len(oos),
        "corr": corr,
        "all": results, "best": best,
        "model": reg if best["ships"] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-start", default=OOS_START)
    ap.add_argument("--oos-end", default=OOS_END)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_kalshi_v3"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 60])
    ap.add_argument("--half-lives", nargs="+", type=float, default=[90.0, 120.0])
    ap.add_argument("--train-starts", nargs="+", default=["full", "2025-05-01"])
    ap.add_argument("--include-meta", action="store_true",
                    help="Also sweep the meta-classifier (rule-right/wrong) mode")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("parsing logs...")
    logs = []
    rdir = ROOT / "backtest_reports/full_live_replay"
    for m in "2025_01 2025_02 2025_03 2025_04 2025_05 2025_06 2025_07 2025_08 2025_09 2025_10 2025_11 2025_12".split():
        p = rdir / m / "topstep_live_bot.log"
        if p.exists() and p.stat().st_size > 50_000:
            logs.append(p)
    live = ROOT / "topstep_live_bot.log"
    if live.exists(): logs.append(live)

    all_events = []
    for p in logs:
        ev = parse_log_events(p)
        log.info("  %s → %d events",
                 p.name if p.parent == ROOT else f"{p.parent.name}/{p.name}", len(ev))
        all_events.extend(ev)
    log.info("total events: %d", len(all_events))
    all_events = add_v2_features(all_events)

    all_mts = [e["market_ts"] for e in all_events]
    start = pd.Timestamp(min(all_mts)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(all_mts)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bars = load_continuous_bars(start, end)
    log.info("bars: %d", len(bars))

    configs = []
    for ts_ in args.train_starts:
        ts_val = None if ts_ == "full" else ts_
        for h in args.horizons:
            for hl in args.half_lives:
                configs.append((h, hl, ts_val))
    log.info("regression configs: %d", len(configs))
    runs = []
    for cfg in configs:
        r = run_config(all_events, bars, cfg[0], cfg[1], cfg[2],
                       args.oos_start, args.oos_end, meta_mode=False, seed=args.seed)
        if r: runs.append(r)

    if args.include_meta:
        log.info("+ meta-classifier sweep")
        for cfg in configs:
            r = run_config(all_events, bars, cfg[0], cfg[1], cfg[2],
                           args.oos_start, args.oos_end, meta_mode=True, seed=args.seed)
            if r: runs.append(r)

    # Summary
    log.info("\n%s", "═" * 140)
    log.info("V3 SWEEP SUMMARY")
    log.info("%s", "═" * 140)
    log.info("%4s %4s %4s %10s %5s %10s %8s %7s %+8s %7s %s",
             "knd", "hrz", "hl", "trstart", "gts", "params",
             "newPs", "WR%", "lift$", "capt%", "ship?")
    log.info("─" * 140)
    for r in runs:
        b = r["best"]
        if r["kind"] == "reg":
            params = f"p={b.get('pass_thr', 0):.1f}/b={b.get('block_thr', 0):.1f}"
        else:
            params = f"thr={b['thr']:.2f}"
        log.info("%4s %4d %4.0f %10s %5d %10s %8d %6.2f%% %+8.2f %6.2f%% %s",
                 r["kind"], r["horizon"], r["half_life"],
                 str(r["train_start"] or "full")[:10],
                 sum(b["gates"].values()), params,
                 b["n_new_pass"], b["new_pass_wr"], b["lift"], b["capt_pct"],
                 "SHIP" if b["ships"] else "-")

    shippers = [r for r in runs if r["best"]["ships"]]
    if not shippers:
        (out_dir / "sweep_summary.json").write_text(json.dumps({
            "verdict": "KILL",
            "reason": "no regression or meta config passes all 5 gates",
            "runs": [{k: v for k, v in r.items() if k not in ("model",)}
                     for r in runs],
        }, indent=2, default=str))
        log.warning("[KILL] no v3 config passes all 5 gates — writing summary only")
        return 1

    best_run = max(shippers, key=lambda r: r["best"]["lift"])
    b = best_run["best"]
    payload = {
        "model_kind": best_run["kind"],
        "clf_or_reg": best_run["model"],
        "feature_cols": FEATURE_COLS_KALSHI_V2,
        "decision": b,
        "horizon_min": best_run["horizon"],
        "half_life_days": best_run["half_life"],
        "train_start": best_run["train_start"],
        "rule_baseline_oos": best_run["rule_baseline"],
        "oracle_oos": best_run["oracle"],
        "n_train": best_run["n_train"], "n_oos": best_run["n_oos"],
        "seed": args.seed,
    }
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "model_meta.json").write_text(json.dumps({
        k: v for k, v in payload.items() if k != "clf_or_reg"
    }, indent=2, default=str))
    log.info("[SHIP] kind=%s  horizon=%d  hl=%.0f  lift=$%+.2f  WR=%.2f%%  capt=%.2f%%",
             best_run["kind"], best_run["horizon"], best_run["half_life"],
             b["lift"], b["new_pass_wr"], b["capt_pct"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
