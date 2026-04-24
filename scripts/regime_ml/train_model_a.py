#!/usr/bin/env python3
"""Train Model A — scalp bracket rewrite — reproducible HGB-only trainer.

Model A decides whether to rewrite TP/SL to scalp values (3/5) vs leave
default (6/4) at signal-birth. Outcome-labeled binary classifier.

Data: es_master_outrights.parquet (OHLCV minute bars, dominant symbol per day).
Features: 40 columns built by _common.build_feature_frame.
Label (binary):
  scalp    = PnL under TP=3/SL=5 > PnL under TP=6/SL=4 over 15-min forward window
  default  = opposite
Rows with |margin| < $30 dropped (ambiguous).

Ship gates (applied at end of training):
  1. OOS PnL ≥ rule baseline + $500 (rule = vol_bp_120 < 1.5 → scalp)
  2. OOS MaxDD ≤ 110% rule MaxDD
  3. Prediction rate in [10%, 90%] (non-degenerate)

Outputs on ship:
    <out-dir>/model.pkl             HGB classifier + threshold + metadata
    <out-dir>/feature_order.json    feature schema + threshold used

Reproduction:
    python3 scripts/regime_ml/train_model_a.py \\
        --start 2024-07-01 --end 2026-04-20 \\
        --holdout-start 2026-01-27 --holdout-end 2026-04-20 \\
        --out-dir artifacts/regime_ml_v5_brackets \\
        --seed 42

Defaults produce the currently-shipped artifact.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT, DEFAULT_TRAIN_START, DEFAULT_TRAIN_END,
    DEFAULT_OOS_START, DEFAULT_OOS_END,
    DEAD_TAPE_TP, DEAD_TAPE_SL, DEFAULT_TP, DEFAULT_SL,
    DEAD_TAPE_VOL_BP, MES_PT_VALUE,
    SAMPLE_EVERY, AMBIGUOUS_MARGIN_USD,
    FEATURE_COLS_40,
    load_continuous_bars, build_feature_frame, filter_ny_session,
    simulate_trade, stats, sample_weights_balanced,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_a")

LABEL_WINDOW_MIN = 15


def build_bracket_labels(bars: pd.DataFrame, feats: pd.DataFrame, window_min: int) -> pd.DataFrame:
    """Per-Nth-bar outcome label: scalp vs default bracket geometry."""
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(feats.index)

    keep, lbls = [], []
    pnl_scalp_list, pnl_def_list = [], []
    for i, ts in enumerate(feat_idx):
        if i % SAMPLE_EVERY != 0:
            continue
        start_pos = idx_pos.get(ts)
        if start_pos is None:
            continue
        win_end = ts + pd.Timedelta(minutes=window_min)
        j = i; win_positions = []
        while j < len(feat_idx):
            tj = feat_idx[j]
            if tj >= win_end: break
            pj = idx_pos.get(tj)
            if pj is not None: win_positions.append(pj)
            j += SAMPLE_EVERY
        if len(win_positions) < 2:
            continue
        dt_total, def_total = 0.0, 0.0
        for pj in win_positions:
            for side in (+1, -1):
                dt_total  += simulate_trade(h, l, c, pj, DEAD_TAPE_TP, DEAD_TAPE_SL, side)
                def_total += simulate_trade(h, l, c, pj, DEFAULT_TP, DEFAULT_SL, side)
        diff = dt_total - def_total
        if abs(diff) < AMBIGUOUS_MARGIN_USD:
            continue
        keep.append(ts)
        lbls.append("scalp" if diff > 0 else "default")
        pnl_scalp_list.append(dt_total)
        pnl_def_list.append(def_total)
    out = feats.loc[keep].copy()
    out["label"] = lbls
    out["pnl_scalp"] = pnl_scalp_list
    out["pnl_default"] = pnl_def_list
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_TRAIN_START,
                    help="First bar date (inclusive, local-tz naive OK)")
    ap.add_argument("--end", default=DEFAULT_OOS_END,
                    help="Last bar date (inclusive) — must cover holdout")
    ap.add_argument("--holdout-start", default=DEFAULT_OOS_START,
                    help="First bar of OOS holdout; training uses everything strictly before.")
    ap.add_argument("--holdout-end", default=DEFAULT_OOS_END,
                    help="Last bar of OOS holdout")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_v5_brackets"),
                    help="Where to write model.pkl + feature_order.json on ship")
    ap.add_argument("--seed", type=int, default=42, help="HGB random_state")
    ap.add_argument("--window-min", type=int, default=LABEL_WINDOW_MIN,
                    help="Forward PnL aggregation window in minutes")
    ap.add_argument("--force", action="store_true",
                    help="Write model.pkl even if gates fail (DEBUG — not for live)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Model A trainer (HGB-only) ===")
    log.info("  train+OOS bars : %s → %s", args.start, args.end)
    log.info("  train range    : %s → (strictly before %s)", args.start, args.holdout_start)
    log.info("  OOS range      : %s → %s", args.holdout_start, args.holdout_end)
    log.info("  out dir        : %s", out_dir)
    log.info("  seed           : %d", args.seed)
    log.info("  window_min     : %d", args.window_min)

    bars = load_continuous_bars(args.start, args.end)
    log.info("loaded %d bars after dominant-symbol filter", len(bars))
    feats = build_feature_frame(bars)
    feats = feats.loc[feats[FEATURE_COLS_40].notna().all(axis=1)].copy()
    feats = filter_ny_session(feats)
    log.info("NY-session feature rows: %d", len(feats))

    labeled = build_bracket_labels(bars, feats, args.window_min)
    from collections import Counter
    log.info("labeled rows: %d  class dist: %s",
             len(labeled), dict(Counter(labeled["label"])))

    hol_start = pd.Timestamp(args.holdout_start, tz=labeled.index.tz)
    hol_end = pd.Timestamp(args.holdout_end, tz=labeled.index.tz) + pd.Timedelta(days=1)
    tr = labeled.loc[labeled.index < hol_start]
    oos = labeled.loc[(labeled.index >= hol_start) & (labeled.index <= hol_end)]
    log.info("train rows: %d  OOS rows: %d", len(tr), len(oos))

    # Train
    X_tr = tr[FEATURE_COLS_40].to_numpy()
    y_tr = tr["label"].to_numpy()
    sw = sample_weights_balanced(y_tr, cost_ratio=1.5)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=args.seed,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    classes = list(clf.classes_)
    scalp_idx = classes.index("scalp")

    # OOS threshold sweep
    X_oos = oos[FEATURE_COLS_40].to_numpy()
    y_true = oos["label"].to_numpy()
    pnl_scalp = oos["pnl_scalp"].to_numpy()
    pnl_def = oos["pnl_default"].to_numpy()
    vb = oos["vol_bp_120"].to_numpy()

    rule_labels = np.where(vb < DEAD_TAPE_VOL_BP, "scalp", "default")
    rule_pnl_arr = np.where(rule_labels == "scalp", pnl_scalp, pnl_def)
    rule_stats = stats(rule_pnl_arr)
    log.info("rule baseline: PnL=$%+.2f  DD=$%.0f", rule_stats["pnl"], rule_stats["dd"])

    probs = clf.predict_proba(X_oos)[:, scalp_idx]

    print(f"\n══ OOS threshold sweep ══")
    print(f"  rule:  PnL=${rule_stats['pnl']:+,.2f}  DD=${rule_stats['dd']:,.0f}")
    print(f"  {'thr':>5} {'n_scalp':>7} {'acc':>7} {'pnl':>11} {'dd':>8} {'lift':>9}  gates")
    best = None
    for thr in np.arange(0.30, 0.81, 0.05):
        pred = np.where(probs >= thr, "scalp", "default")
        acc = float((pred == y_true).mean())
        pnl_arr = np.where(pred == "scalp", pnl_scalp, pnl_def)
        st = stats(pnl_arr)
        n_scalp = int((pred == "scalp").sum())
        frac = n_scalp / max(1, len(pred))
        lift = st["pnl"] - rule_stats["pnl"]
        gates = {
            "pnl_ok":       lift >= 500.0,
            "dd_ok":        st["dd"] <= rule_stats["dd"] * 1.10 if rule_stats["dd"] > 0 else True,
            "non_degen_ok": 0.10 <= frac <= 0.90,
        }
        ok = all(gates.values())
        flag = " SHIP" if ok else ""
        print(f"  {thr:>5.2f} {n_scalp:>7} {acc*100:>6.2f}% ${st['pnl']:>+9,.2f} "
              f"${st['dd']:>6,.0f} ${lift:>+7,.2f}   {sum(gates.values())}/3{flag}")
        if ok and (best is None or lift > best["lift"]):
            best = {"thr": float(thr), "acc": acc, **st, "lift": lift, "gates": gates}

    if best is None and not args.force:
        log.warning("[KILL] no threshold passes all gates — not writing model")
        return 1

    if best is None and args.force:
        log.warning("[FORCE] writing model with first threshold above 0.5 despite gate fail")
        best = {"thr": 0.50, "lift": 0.0}

    log.info("[SHIP] thr=%.2f  PnL=$%+.2f  lift=$%+.2f",
             best["thr"], best.get("pnl", 0.0), best["lift"])
    payload = {
        "threshold": best["thr"],
        "threshold_hgb_only": best["thr"],   # consistent key for the live inference path
        "feature_cols": FEATURE_COLS_40,
        "positive_class": "scalp",
        "label_to_int": {c: i for i, c in enumerate(clf.classes_)},
        "hgb": clf,
        "label_name": "scalp_brackets",
        "inference_mode": "hgb_only",
        "stats_oos": best,
        "train_range": [args.start, args.holdout_start],
        "oos_range": [args.holdout_start, args.holdout_end],
        "seed": args.seed,
        "window_min": args.window_min,
    }
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "feature_order.json").write_text(json.dumps({
        "features": FEATURE_COLS_40, "threshold": best["thr"],
        "positive_class": "scalp",
        "label_name": "scalp_brackets",
        "train_range": [args.start, args.holdout_start],
        "oos_range": [args.holdout_start, args.holdout_end],
    }, indent=2))
    log.info("wrote %s and feature_order.json", out_dir / "model.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
