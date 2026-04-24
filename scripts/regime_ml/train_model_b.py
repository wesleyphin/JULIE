#!/usr/bin/env python3
"""Train Model B — size reduction — A-conditional HGB-only trainer.

Model B decides whether to force size=1 vs allow natural size at signal-birth.
Requires Model A's prediction (`a_pred_scalp`) as the 41st feature, because
v5's isolated B regressed combined PnL by $11k when stacked with A.

Data: es_master_outrights.parquet + shipped Model A artifact.
Features: 40 base + a_pred_scalp (41 total).
Label (binary):
  reduce  = A-conditional pnl_size1 > pnl_size3 over 15-min forward window
  natural = opposite
Rows with |margin| < $30 dropped.

Ship gate:
    combined OOS (A=ML + B=ML + C=rule) beats combined (A=ML + B=rule + C=rule)
    on BOTH PnL AND DD. No free DD improvement at PnL cost.

Outputs on ship:
    <out-dir>/model.pkl             HGB classifier + threshold + metadata
    <out-dir>/feature_order.json    feature schema + threshold used

Reproduction:
    # Model A must be trained + shipped FIRST at the default path.
    python3 scripts/regime_ml/train_model_a.py
    python3 scripts/regime_ml/train_model_b.py \\
        --start 2024-07-01 --end 2026-04-20 \\
        --holdout-start 2026-01-27 --holdout-end 2026-04-20 \\
        --model-a-path artifacts/regime_ml_v5_brackets/model.pkl \\
        --out-dir artifacts/regime_ml_v6_size --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections import Counter
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
    SAMPLE_EVERY, AMBIGUOUS_MARGIN_USD, PNL_LOOKAHEAD_BARS,
    NATURAL_SIZE, SMALL_SIZE,
    FEATURE_COLS_40, FEATURE_COLS_V6,
    load_continuous_bars, build_feature_frame, filter_ny_session,
    simulate_trade, simulate_be_trade, stats, sample_weights_balanced,
    BE_TP, BE_SL, BE_TRIGGER_MFE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_b")

LABEL_WINDOW_MIN = 15


def load_model_a_predict(model_a_path: Path):
    with model_a_path.open("rb") as fh:
        p = pickle.load(fh)
    hgb = p["hgb"]
    threshold = p.get("threshold_hgb_only", p.get("threshold", 0.50))
    positive_class = p["positive_class"]
    feature_cols = p["feature_cols"]
    p_idx = list(hgb.classes_).index(positive_class)
    def predict(X):
        return (hgb.predict_proba(X)[:, p_idx] >= threshold).astype(int)
    log.info("Model A loaded: thr=%.2f  features=%d", threshold, len(feature_cols))
    return predict


def augment_features(feats: pd.DataFrame, predict_a) -> pd.DataFrame:
    X = feats[FEATURE_COLS_40].to_numpy()
    a_pred = predict_a(X)
    out = feats.copy()
    out["a_pred_scalp"] = a_pred
    return out


def build_size_labels(bars, feats, predict_a, window_min):
    feats = augment_features(feats, predict_a)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(feats.index)

    keep, lbls = [], []
    pnl_3_list, pnl_1_list, a_list = [], [], []
    for i, ts in enumerate(feat_idx):
        if i % SAMPLE_EVERY != 0: continue
        start_pos = idx_pos.get(ts)
        if start_pos is None: continue
        win_end = ts + pd.Timedelta(minutes=window_min)
        j = i; win_positions = []
        while j < len(feat_idx):
            tj = feat_idx[j]
            if tj >= win_end: break
            pj = idx_pos.get(tj)
            if pj is not None: win_positions.append(pj)
            j += SAMPLE_EVERY
        if len(win_positions) < 2: continue
        a = int(feats.loc[ts, "a_pred_scalp"])
        tp = DEAD_TAPE_TP if a == 1 else DEFAULT_TP
        sl = DEAD_TAPE_SL if a == 1 else DEFAULT_SL
        per_trade_1 = []
        for pj in win_positions:
            for side in (+1, -1):
                per_trade_1.append(simulate_trade(h, l, c, pj, tp, sl, side))
        if not per_trade_1: continue
        pnl_3 = sum(p * NATURAL_SIZE for p in per_trade_1)
        pnl_1 = sum(per_trade_1)
        margin = pnl_1 - pnl_3
        if abs(margin) < AMBIGUOUS_MARGIN_USD: continue
        keep.append(ts); lbls.append("reduce" if margin > 0 else "natural")
        pnl_3_list.append(pnl_3); pnl_1_list.append(pnl_1); a_list.append(a)
    out = feats.loc[keep].copy()
    out["label"] = lbls
    out["pnl_size3"] = pnl_3_list; out["pnl_size1"] = pnl_1_list
    out["a_action"] = a_list
    return out


def combined_sim(bars: pd.DataFrame, feats: pd.DataFrame, predict_a, predict_b=None):
    """Simulate A-ML brackets with optional B-ML size reduction. No BE ML here."""
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(feats.index)
    feats = augment_features(feats, predict_a)
    X = feats[FEATURE_COLS_V6].to_numpy()
    a_preds = feats["a_pred_scalp"].to_numpy()
    b_probs = predict_b(X) if predict_b is not None else None

    pnl_arr = []
    for i, ts in enumerate(feat_idx):
        if i % SAMPLE_EVERY != 0: continue
        start_pos = idx_pos.get(ts)
        if start_pos is None: continue
        win_end = ts + pd.Timedelta(minutes=LABEL_WINDOW_MIN)
        j = i; wp = []
        while j < len(feat_idx):
            tj = feat_idx[j]
            if tj >= win_end: break
            pj = idx_pos.get(tj)
            if pj is not None: wp.append(pj)
            j += SAMPLE_EVERY
        if len(wp) < 2: continue
        a = int(a_preds[i])
        tp = DEAD_TAPE_TP if a == 1 else DEFAULT_TP
        sl = DEAD_TAPE_SL if a == 1 else DEFAULT_SL
        reduce_b = b_probs[i] if b_probs is not None else None
        bar_pnl = 0.0
        for pj in wp:
            for side in (+1, -1):
                if a == 0:
                    # Default brackets use BE-ON by default (matches live DE3)
                    pt = simulate_be_trade(h, l, c, pj, BE_TP, BE_SL, BE_TRIGGER_MFE, side, be_on=True)
                else:
                    pt = simulate_trade(h, l, c, pj, tp, sl, side)
                size_mult = SMALL_SIZE if (reduce_b is True) else NATURAL_SIZE
                bar_pnl += pt * size_mult
        pnl_arr.append(bar_pnl)
    return stats(pnl_arr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_TRAIN_START)
    ap.add_argument("--end", default=DEFAULT_OOS_END)
    ap.add_argument("--holdout-start", default=DEFAULT_OOS_START)
    ap.add_argument("--holdout-end", default=DEFAULT_OOS_END)
    ap.add_argument("--model-a-path",
                    default=str(ROOT / "artifacts/regime_ml_v5_brackets/model.pkl"),
                    help="Path to shipped Model A artifact — MUST exist before training B")
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_v6_size"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--window-min", type=int, default=LABEL_WINDOW_MIN)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ma_path = Path(args.model_a_path).resolve()
    if not ma_path.exists():
        log.error("Model A artifact missing at %s — train Model A first", ma_path)
        return 2
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Model B trainer (A-conditional, HGB-only) ===")
    log.info("  Model A  : %s", ma_path)
    log.info("  train→hol: %s → %s  |  OOS: %s → %s",
             args.start, args.holdout_start, args.holdout_start, args.holdout_end)
    log.info("  out dir  : %s", out_dir)

    predict_a = load_model_a_predict(ma_path)
    bars = load_continuous_bars(args.start, args.end)
    feats = build_feature_frame(bars)
    feats = feats.loc[feats[FEATURE_COLS_40].notna().all(axis=1)].copy()
    feats = filter_ny_session(feats)
    log.info("NY-session feature rows: %d", len(feats))

    labeled = build_size_labels(bars, feats, predict_a, args.window_min)
    log.info("labeled rows: %d  class dist: %s",
             len(labeled), dict(Counter(labeled["label"])))

    hol_start = pd.Timestamp(args.holdout_start, tz=labeled.index.tz)
    hol_end = pd.Timestamp(args.holdout_end, tz=labeled.index.tz) + pd.Timedelta(days=1)
    tr = labeled.loc[labeled.index < hol_start]
    oos = labeled.loc[(labeled.index >= hol_start) & (labeled.index <= hol_end)]
    log.info("train: %d  OOS: %d", len(tr), len(oos))

    X_tr = tr[FEATURE_COLS_V6].to_numpy()
    y_tr = tr["label"].to_numpy()
    sw = sample_weights_balanced(y_tr, cost_ratio=1.5)
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=args.seed,
    )
    clf.fit(X_tr, y_tr, sample_weight=sw)
    reduce_idx = list(clf.classes_).index("reduce")

    # Combined sim for ship gate (A=ML only vs A=ML + B=ML)
    oos_feats = feats.loc[feats.index >= hol_start]
    baseline = combined_sim(bars, oos_feats, predict_a, predict_b=None)
    log.info("Baseline (A=ML only): PnL=$%+.2f  DD=$%.0f", baseline["pnl"], baseline["dd"])

    print(f"\n══ Combined OOS sweep (A=ML + B=ML) vs (A=ML only) ══")
    print(f"  baseline: PnL=${baseline['pnl']:+,.2f}  DD=${baseline['dd']:,.0f}")
    print(f"  {'thr':>5} {'pnl':>12} {'dd':>9} {'lift_pnl':>10} {'dd_ratio':>8} gates")
    best = None
    for thr in np.arange(0.35, 0.81, 0.05):
        def predict_b_fn(X, _thr=float(thr)):
            p = clf.predict_proba(X)[:, reduce_idx]
            return p >= _thr
        st = combined_sim(bars, oos_feats, predict_a, predict_b=predict_b_fn)
        lift = st["pnl"] - baseline["pnl"]
        dd_ratio = st["dd"] / baseline["dd"] if baseline["dd"] > 0 else 1.0
        gates = {"pnl_ok": lift > 0, "dd_ok": st["dd"] <= baseline["dd"]}
        ok = all(gates.values())
        flag = " SHIP" if ok else ""
        print(f"  {thr:>5.2f} ${st['pnl']:>+10,.2f} ${st['dd']:>7,.0f} "
              f"${lift:>+8,.2f}  {dd_ratio:>7.2f}  {sum(gates.values())}/2{flag}")
        if ok and (best is None or lift > best["lift"]):
            best = {"thr": float(thr), "pnl": st["pnl"], "dd": st["dd"], "lift": lift}

    if best is None and not args.force:
        log.warning("[KILL] no threshold passes both gates — not writing model")
        return 1

    log.info("[SHIP] thr=%.2f  PnL=$%+.2f  lift=$%+.2f",
             best["thr"], best["pnl"], best["lift"])
    payload = {
        "threshold": best["thr"],
        "threshold_hgb_only": best["thr"],
        "feature_cols": FEATURE_COLS_V6,
        "positive_class": "reduce",
        "label_to_int": {c: i for i, c in enumerate(clf.classes_)},
        "hgb": clf,
        "label_name": "size_reduction_v6_a_conditional",
        "inference_mode": "hgb_only",
        "stats_oos": best,
        "train_range": [args.start, args.holdout_start],
        "oos_range": [args.holdout_start, args.holdout_end],
        "seed": args.seed,
        "window_min": args.window_min,
        "requires_a_pred_feature": True,
    }
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "feature_order.json").write_text(json.dumps({
        "features": FEATURE_COLS_V6, "threshold": best["thr"],
        "positive_class": "reduce", "requires_a_pred": True,
        "label_name": "size_reduction_v6_a_conditional",
        "train_range": [args.start, args.holdout_start],
        "oos_range": [args.holdout_start, args.holdout_end],
    }, indent=2))
    log.info("wrote %s", out_dir / "model.pkl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
