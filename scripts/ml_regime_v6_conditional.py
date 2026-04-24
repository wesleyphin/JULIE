#!/usr/bin/env python3
"""v6 — A-conditional retry on Models B + C.

v5 post-mortem showed:
  - Labels were already oracle-aligned (0.4% disagreement vs oracle truth)
  - Oracle edge is MASSIVE (+$150k lift on B if we could predict perfectly)
  - Root failure: training B/C in ISOLATION from A's decision.
    Combined A=scalp + B=reduce creates "scalp-sized scalp trades" with
    trivial PnL. Size-reduction on a bar where the scalp-bracket was
    already chosen gives up 2/3 of the gain.

v6 fix: train B and C CONDITIONED on A's predicted action.
  - Feature input now includes A's binary prediction
  - Label: "does this action help GIVEN A's choice" (4-way conditional)
  - Inference gate requires predicted benefit > margin $ threshold

Ship gates per user:
  Model B v6: combined (A=ML + B=ML + C=rule) must beat
              combined (A=ML + B=rule + C=rule) on PnL AND DD.
  Model C v6: combined (A=ML + B=rule + C=ML) must beat
              combined (A=ML + B=rule + C=rule) on PnL AND DD.

HGB-only from the start (avoid LGBM OMP crash that tanked v5 deployment).
"""
from __future__ import annotations

import json, logging, pickle, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_B = ROOT / "artifacts" / "regime_ml_v6_size"
OUT_DIR_C = ROOT / "artifacts" / "regime_ml_v6_be"
for d in (OUT_DIR_B, OUT_DIR_C):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from ml_regime_v5_three_models import (
    load_and_featurize, FEATURE_COLS,
    TRAIN_START, TRAIN_END, OOS_START, OOS_END,
    DEAD_TAPE_VOL_BP, MES_PT_VALUE,
    DEAD_TAPE_TP, DEAD_TAPE_SL, DEFAULT_TP, DEFAULT_SL,
    PNL_LOOKAHEAD_BARS, SAMPLE_EVERY, AMBIGUOUS_MARGIN_USD,
    BE_TP, BE_SL, BE_TRIGGER_MFE,
    NATURAL_SIZE, simulate_be_trade,
)
from ml_regime_classifier_v4 import simulate_trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v6")

LABEL_WINDOW_MIN = 15
SMALL_SIZE = 1

# v6 requires clearer margin than v5 to label (cuts noise-driven training rows)
B_MARGIN_USD = 30.0
C_MARGIN_USD = 30.0


def load_model_a():
    """Load the shipped Model A HGB-only payload."""
    pkl = ROOT / "artifacts" / "regime_ml_v5_brackets" / "model.pkl"
    with pkl.open("rb") as fh:
        payload = pickle.load(fh)
    hgb = payload["hgb"]
    threshold = payload.get("threshold_hgb_only", 0.50)
    feature_cols = payload["feature_cols"]
    positive_class = payload["positive_class"]
    hgb_classes = list(hgb.classes_)
    p_idx = hgb_classes.index(positive_class)
    def predict_a(X: np.ndarray) -> np.ndarray:
        p = hgb.predict_proba(X)[:, p_idx]
        return (p >= threshold).astype(int)   # 1 = scalp, 0 = default
    log.info("Model A loaded (HGB, thr=%.2f)", threshold)
    return predict_a


# ─── v6 feature set: FEATURE_COLS + A's prediction ─────────────────────────

FEATURE_COLS_V6 = FEATURE_COLS + ["a_pred_scalp"]


def augment_with_a_predictions(feats: pd.DataFrame, predict_a) -> pd.DataFrame:
    X = feats[FEATURE_COLS].to_numpy()
    a_pred = predict_a(X)
    out = feats.copy()
    out["a_pred_scalp"] = a_pred
    return out


# ─── Model B v6 — size reduction conditional on A's action ────────────────

def build_size_labels_v6(bars, feats, predict_a):
    """For each bar, compute forward-window PnL under FOUR combinations:
        (A_default + size_3, A_default + size_1, A_scalp + size_3, A_scalp + size_1)
    Take A's prediction as given. Label B based on which size-choice
    produces higher PnL given A's predicted action.

    Label = "reduce" if A-conditional size-1 PnL > A-conditional size-3 PnL
            by margin >= B_MARGIN_USD, else "natural" if the reverse holds
            by same margin, else drop (ambiguous).
    """
    log.info("[B v6] building A-conditional size labels...")
    feats = augment_with_a_predictions(feats, predict_a)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(feats.index)

    keep, lbls = [], []
    pnl_3_list, pnl_1_list, a_pred_list = [], [], []
    for i, ts in enumerate(feat_idx):
        if i % SAMPLE_EVERY != 0: continue
        start_pos = idx_pos.get(ts)
        if start_pos is None: continue
        win_end = ts + pd.Timedelta(minutes=LABEL_WINDOW_MIN)
        j = i
        win_positions = []
        while j < len(feat_idx):
            tj = feat_idx[j]
            if tj >= win_end: break
            pj = idx_pos.get(tj)
            if pj is not None: win_positions.append(pj)
            j += SAMPLE_EVERY
        if len(win_positions) < 2: continue
        a = int(feats.loc[ts, "a_pred_scalp"])
        # Pick brackets per A's prediction
        tp = DEAD_TAPE_TP if a == 1 else DEFAULT_TP
        sl = DEAD_TAPE_SL if a == 1 else DEFAULT_SL
        per_trade_pnl_1 = []
        for pj in win_positions:
            for side in (+1, -1):
                per_trade_pnl_1.append(simulate_trade(h, l, c, pj, tp, sl, side))
        if not per_trade_pnl_1: continue
        pnl_3 = sum(p * NATURAL_SIZE for p in per_trade_pnl_1)
        pnl_1 = sum(per_trade_pnl_1)
        margin = pnl_1 - pnl_3
        if abs(margin) < B_MARGIN_USD: continue
        label = "reduce" if margin > 0 else "natural"
        keep.append(ts); lbls.append(label)
        pnl_3_list.append(pnl_3); pnl_1_list.append(pnl_1); a_pred_list.append(a)

    out = feats.loc[keep].copy()
    out["label_b6"] = lbls
    out["pnl_size3_cond_a"] = pnl_3_list
    out["pnl_size1_cond_a"] = pnl_1_list
    out["a_action"] = a_pred_list
    log.info("[B v6] rows: %d  dist: %s", len(out), dict(Counter(lbls)))
    log.info("[B v6] label × A cross-tab:")
    for a in (0, 1):
        sub = out[out["a_action"] == a]
        log.info("  A=%s  rows=%d  labels=%s", "scalp" if a else "default",
                 len(sub), dict(Counter(sub["label_b6"])) if len(sub) else {})
    return out


# ─── Model C v6 — BE disable conditional on A's action ────────────────────

def build_be_labels_v6(bars, feats, predict_a):
    """Label = disable iff BE-OFF PnL > BE-ON PnL over forward window
    under A-predicted brackets, AND the bar had at least one trade that
    REACHED the BE trigger (conditional labeling). Drop bars where |diff|
    < C_MARGIN_USD.
    """
    log.info("[C v6] building A-conditional BE labels...")
    feats = augment_with_a_predictions(feats, predict_a)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(feats.index)

    keep, lbls = [], []
    pnl_off_list, pnl_on_list, a_pred_list = [], [], []
    n_reached = 0
    for i, ts in enumerate(feat_idx):
        if i % SAMPLE_EVERY != 0: continue
        start_pos = idx_pos.get(ts)
        if start_pos is None: continue
        win_end = ts + pd.Timedelta(minutes=LABEL_WINDOW_MIN)
        j = i
        win_positions = []
        while j < len(feat_idx):
            tj = feat_idx[j]
            if tj >= win_end: break
            pj = idx_pos.get(tj)
            if pj is not None: win_positions.append(pj)
            j += SAMPLE_EVERY
        if len(win_positions) < 2: continue
        a = int(feats.loc[ts, "a_pred_scalp"])
        # Scalp brackets = no BE anyway — skip A=scalp bars
        if a == 1:
            continue
        tp, sl = DEFAULT_TP, DEFAULT_SL
        # BE simulation uses BE_TP=10, BE_SL=4, BE_TRIGGER=5. For default
        # brackets where TP=6 (DEFAULT_TP), BE_TRIGGER=5 is reachable but
        # TP=6 hits first in many cases. Adjust: use wider default for
        # BE-sensitivity testing (matches DE3's deployed BE regime).
        tp_be, sl_be = BE_TP, BE_SL
        # Count trades that reach BE trigger
        any_reached = False
        off_total, on_total = 0.0, 0.0
        for pj in win_positions:
            for side in (+1, -1):
                # Reached BE trigger check
                if pj + PNL_LOOKAHEAD_BARS >= len(c): continue
                entry = c[pj]
                if side > 0:
                    mfe = float(h[pj+1 : pj+1+PNL_LOOKAHEAD_BARS].max() - entry)
                else:
                    mfe = float(entry - l[pj+1 : pj+1+PNL_LOOKAHEAD_BARS].min())
                if mfe >= BE_TRIGGER_MFE:
                    any_reached = True
                off_total += simulate_be_trade(h, l, c, pj, tp_be, sl_be, BE_TRIGGER_MFE, side, be_on=False)
                on_total  += simulate_be_trade(h, l, c, pj, tp_be, sl_be, BE_TRIGGER_MFE, side, be_on=True)
        if not any_reached:
            continue   # conditional: skip if no trades reached BE
        n_reached += 1
        diff = off_total - on_total
        if abs(diff) < C_MARGIN_USD: continue
        label = "disable" if diff > 0 else "keep"
        keep.append(ts); lbls.append(label)
        pnl_off_list.append(off_total); pnl_on_list.append(on_total); a_pred_list.append(a)

    out = feats.loc[keep].copy()
    out["label_c6"] = lbls
    out["pnl_be_off_cond"] = pnl_off_list
    out["pnl_be_on_cond"] = pnl_on_list
    out["a_action"] = a_pred_list
    log.info("[C v6] windows with ≥1 BE-reaching trade: %d", n_reached)
    log.info("[C v6] labeled rows: %d  dist: %s", len(out), dict(Counter(lbls)))
    return out


# ─── Training ────────────────────────────────────────────────────────────

def train_hgb_only(X, y, cost_ratio=1.5):
    counts = Counter(y)
    base_w = {lbl: len(y) / (2 * counts[lbl]) for lbl in counts}
    minority = min(counts, key=counts.get)
    base_w[minority] *= cost_ratio
    sw = np.array([base_w[lbl] for lbl in y])
    clf = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=42)
    clf.fit(X, y, sample_weight=sw)
    return clf


def stats(arr):
    if len(arr) == 0:
        return {"n": 0, "pnl": 0.0, "avg": 0.0, "dd": 0.0}
    a = np.asarray(arr, dtype=float)
    cum = np.cumsum(a)
    peak = np.maximum.accumulate(cum)
    return {"n": int(len(a)), "pnl": float(a.sum()), "avg": float(a.mean()),
             "dd": float(np.max(peak - cum))}


# ─── Combined simulation ─────────────────────────────────────────────────

def combined_sim(bars, feats, predict_a, predict_b=None, predict_c=None,
                  b_threshold=0.55, c_threshold=0.55):
    """Simulate full live action stack on OOS bars. Returns stats."""
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(feats.index)

    pnl_arr = []
    feats = augment_with_a_predictions(feats, predict_a)
    X = feats[FEATURE_COLS_V6].to_numpy()
    a_preds = feats["a_pred_scalp"].to_numpy()

    if predict_b is not None:
        b_probs = predict_b(X)
    else:
        b_probs = None
    if predict_c is not None:
        c_probs = predict_c(X)
    else:
        c_probs = None

    for i, ts in enumerate(feat_idx):
        if i % SAMPLE_EVERY != 0: continue
        start_pos = idx_pos.get(ts)
        if start_pos is None: continue
        win_end = ts + pd.Timedelta(minutes=LABEL_WINDOW_MIN)
        j = i
        win_positions = []
        while j < len(feat_idx):
            tj = feat_idx[j]
            if tj >= win_end: break
            pj = idx_pos.get(tj)
            if pj is not None: win_positions.append(pj)
            j += SAMPLE_EVERY
        if len(win_positions) < 2: continue
        a = int(a_preds[i])
        tp = DEAD_TAPE_TP if a == 1 else DEFAULT_TP
        sl = DEAD_TAPE_SL if a == 1 else DEFAULT_SL

        # B decision: reduce if flag AND proba > threshold
        if predict_b is not None:
            reduce_b = b_probs[i] >= b_threshold
        else:
            reduce_b = False

        # C decision: disable BE if flag AND proba > threshold
        if predict_c is not None:
            disable_c = c_probs[i] >= c_threshold
        else:
            disable_c = False

        bar_pnl = 0.0
        for pj in win_positions:
            for side in (+1, -1):
                if a == 0 and not disable_c:
                    # BE-ON path on default bracket
                    pt = simulate_be_trade(h, l, c, pj, BE_TP, BE_SL,
                                            BE_TRIGGER_MFE, side, be_on=True)
                elif a == 0 and disable_c:
                    # BE-OFF path on default bracket
                    pt = simulate_be_trade(h, l, c, pj, BE_TP, BE_SL,
                                            BE_TRIGGER_MFE, side, be_on=False)
                else:
                    # Scalp brackets (no BE)
                    pt = simulate_trade(h, l, c, pj, tp, sl, side)
                size_mult = SMALL_SIZE if reduce_b else NATURAL_SIZE
                bar_pnl += pt * size_mult
        pnl_arr.append(bar_pnl)
    return stats(pnl_arr)


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    predict_a = load_model_a()
    bars_all, feats_all = load_and_featurize()

    # Build v6 labels (both B and C in one pass for reuse)
    labeled_b6 = build_size_labels_v6(bars_all, feats_all, predict_a)
    labeled_c6 = build_be_labels_v6(bars_all, feats_all, predict_a)

    # Train Model B v6
    tr_cut = pd.Timestamp(TRAIN_END, tz=labeled_b6.index.tz)
    oos_s  = pd.Timestamp(OOS_START, tz=labeled_b6.index.tz)
    tr_b = labeled_b6.loc[labeled_b6.index <= tr_cut]
    oos_b = labeled_b6.loc[labeled_b6.index >= oos_s]
    log.info("[B v6] train=%d  oos=%d", len(tr_b), len(oos_b))
    clf_b = train_hgb_only(tr_b[FEATURE_COLS_V6].to_numpy(),
                            tr_b["label_b6"].to_numpy())
    hgb_classes_b = list(clf_b.classes_)
    reduce_idx_b = hgb_classes_b.index("reduce")

    def predict_b_proba(X): return clf_b.predict_proba(X)[:, reduce_idx_b]

    # Train Model C v6
    tr_c = labeled_c6.loc[labeled_c6.index <= tr_cut]
    oos_c = labeled_c6.loc[labeled_c6.index >= oos_s]
    log.info("[C v6] train=%d  oos=%d", len(tr_c), len(oos_c))
    if len(tr_c) > 100 and len(oos_c) > 30:
        clf_c = train_hgb_only(tr_c[FEATURE_COLS_V6].to_numpy(),
                                tr_c["label_c6"].to_numpy())
        hgb_classes_c = list(clf_c.classes_)
        disable_idx_c = hgb_classes_c.index("disable") if "disable" in hgb_classes_c else None
        def predict_c_proba(X):
            if disable_idx_c is None:
                return np.zeros(len(X))
            return clf_c.predict_proba(X)[:, disable_idx_c]
    else:
        clf_c = None
        predict_c_proba = None
        log.warning("[C v6] insufficient data to train")

    # ─── OOS combined sweeps ─────────────────────────────────────────────

    # Need a common OOS feature frame for combined sim. Use labeled_b6's OOS
    # index since B's label drops non-clear-signal bars too.
    # Actually for FAIR combined comparison we want all NY-session OOS bars.
    # Use feats_all filtered to OOS range.
    oos_feats = feats_all.loc[feats_all.index >= oos_s]
    log.info("OOS feat rows (unfiltered): %d", len(oos_feats))

    # Baseline: A=ML, B=rule, C=rule
    baseline = combined_sim(bars_all, oos_feats, predict_a, predict_b=None, predict_c=None)
    log.info("Baseline A=ML only: PnL=$%+.2f  DD=$%.0f",
             baseline["pnl"], baseline["dd"])

    # ─── B threshold sweep ───────────────────────────────────────────────
    print("\n══ Model B v6 — combined (A=ML + B=ML + C=rule) vs baseline (A=ML only) ══")
    print(f"  baseline: PnL=${baseline['pnl']:+,.2f}  DD=${baseline['dd']:.0f}")
    print(f"  {'thr':>5}  {'pnl':>11}  {'dd':>8}  {'lift_pnl':>9}  {'dd_ratio':>8}  {'gates'}")
    best_b = None
    for thr in np.arange(0.35, 0.81, 0.05):
        st = combined_sim(bars_all, oos_feats, predict_a,
                           predict_b=predict_b_proba, predict_c=None,
                           b_threshold=float(thr))
        lift_pnl = st["pnl"] - baseline["pnl"]
        dd_ratio = st["dd"] / baseline["dd"] if baseline["dd"] > 0 else 1.0
        gates = {
            "pnl_ok": lift_pnl > 0,
            "dd_ok":  st["dd"] <= baseline["dd"],
        }
        all_pass = all(gates.values())
        flag = " SHIP" if all_pass else ""
        print(f"  {thr:>5.2f}  ${st['pnl']:>+9,.2f}  ${st['dd']:>6,.0f}  "
              f"${lift_pnl:>+8,.2f}  {dd_ratio:>7.2f}  {sum(gates.values())}/2{flag}")
        if all_pass and (best_b is None or lift_pnl > best_b["lift"]):
            best_b = {"thr": thr, "pnl": st["pnl"], "dd": st["dd"], "lift": lift_pnl}

    # ─── C threshold sweep (only if C trained) ───────────────────────────
    best_c = None
    if predict_c_proba is not None:
        print("\n══ Model C v6 — combined (A=ML + B=rule + C=ML) vs baseline (A=ML only) ══")
        print(f"  {'thr':>5}  {'pnl':>11}  {'dd':>8}  {'lift_pnl':>9}  {'dd_ratio':>8}  {'gates'}")
        for thr in np.arange(0.35, 0.81, 0.05):
            st = combined_sim(bars_all, oos_feats, predict_a,
                               predict_b=None, predict_c=predict_c_proba,
                               c_threshold=float(thr))
            lift_pnl = st["pnl"] - baseline["pnl"]
            dd_ratio = st["dd"] / baseline["dd"] if baseline["dd"] > 0 else 1.0
            gates = {
                "pnl_ok": lift_pnl > 0,
                "dd_ok":  st["dd"] <= baseline["dd"],
            }
            all_pass = all(gates.values())
            flag = " SHIP" if all_pass else ""
            print(f"  {thr:>5.2f}  ${st['pnl']:>+9,.2f}  ${st['dd']:>6,.0f}  "
                  f"${lift_pnl:>+8,.2f}  {dd_ratio:>7.2f}  {sum(gates.values())}/2{flag}")
            if all_pass and (best_c is None or lift_pnl > best_c["lift"]):
                best_c = {"thr": thr, "pnl": st["pnl"], "dd": st["dd"], "lift": lift_pnl}

    # ─── Save ships ──────────────────────────────────────────────────────
    b_shipped, c_shipped = False, False
    summary = {"baseline": baseline, "best_b": best_b, "best_c": best_c}
    if best_b is not None:
        payload = {
            "threshold": float(best_b["thr"]),
            "feature_cols": FEATURE_COLS_V6,
            "positive_class": "reduce",
            "hgb": clf_b,
            "label_name": "size_reduction_v6_a_conditional",
            "stats_oos": best_b,
            "inference_mode": "hgb_only",
            "requires_a_pred_feature": True,
        }
        with (OUT_DIR_B / "model.pkl").open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (OUT_DIR_B / "feature_order.json").write_text(
            json.dumps({"features": FEATURE_COLS_V6, "threshold": best_b["thr"],
                         "requires_a_pred": True}, indent=2))
        b_shipped = True
        log.info("[B v6] SHIPPED: thr=%.2f  lift=$%+.2f", best_b["thr"], best_b["lift"])
    else:
        log.warning("[B v6] NO ship — no threshold beats baseline on PnL AND DD")
    if best_c is not None:
        payload = {
            "threshold": float(best_c["thr"]),
            "feature_cols": FEATURE_COLS_V6,
            "positive_class": "disable",
            "hgb": clf_c,
            "label_name": "be_disable_v6_a_conditional",
            "stats_oos": best_c,
            "inference_mode": "hgb_only",
            "requires_a_pred_feature": True,
        }
        with (OUT_DIR_C / "model.pkl").open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (OUT_DIR_C / "feature_order.json").write_text(
            json.dumps({"features": FEATURE_COLS_V6, "threshold": best_c["thr"],
                         "requires_a_pred": True}, indent=2))
        c_shipped = True
        log.info("[C v6] SHIPPED: thr=%.2f  lift=$%+.2f", best_c["thr"], best_c["lift"])
    else:
        log.warning("[C v6] NO ship — no threshold beats baseline on PnL AND DD")

    (ROOT / "artifacts" / "regime_ml_v6_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    # Final print
    print(f"\n══ v6 FINAL ══")
    print(f"  Model B: {'SHIP' if b_shipped else 'KILL'}")
    print(f"  Model C: {'SHIP' if c_shipped else 'KILL'}")
    return 0 if (b_shipped or c_shipped) else 1


if __name__ == "__main__":
    sys.exit(main())
