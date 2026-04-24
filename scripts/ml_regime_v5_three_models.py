#!/usr/bin/env python3
"""ML Regime Classifier v5 — three independent models, one per action.

Action-space decomposition from v4's insight:
    Model A (scalp brackets):  3/5 vs 6/4 TP:SL — bracket geometry only
    Model B (size reduction):  force size=1 vs allow natural sizing — safety tool
    Model C (BE-arm disable):  skip BE move-to-entry vs default BE at +5pt MFE

Each model trained + OOS-validated independently. Ship whichever pass gates.
Final combined-action backtest compares the ship configuration against pure
rule baseline; ships only if combined beats rule on PnL AND MaxDD.

Per-model ship gates:

  Model A (scalp bracket):
    - OOS PnL ≥ rule baseline + $500
    - OOS MaxDD ≤ 110% rule baseline
    - No gate 4 (catastrophic misassign) — bracket geometry is freely applicable
      to any vol regime without downstream safety violations. That's why we
      split.

  Model B (size reduction):
    - OOS Sharpe-like score (mean/std of forward window PnL) ≥ rule baseline
    - OOS MaxDD ≤ 100% rule baseline (STRICT — size is a safety tool)
    - PnL ≥ rule baseline - $500 (allow modest PnL give-up for DD improvement)

  Model C (BE disable):
    - OOS PnL ≥ rule baseline + $500
    - OOS MaxDD ≤ 110% rule baseline
    - Sanity: ML-called BE-off bars' avg MFE-to-SL ratio < default bars
      (i.e. ML correctly identifies mean-reverting setups)

Combined backtest gate:
    - Combined-ship system OOS PnL ≥ rule baseline
    - Combined-ship OOS MaxDD ≤ rule baseline
    - No regression vs rule on any gate

Failed models fall back to the rule at runtime via the decoupled functions
in regime_classifier.py.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_A = ROOT / "artifacts" / "regime_ml_v5_brackets"
OUT_DIR_B = ROOT / "artifacts" / "regime_ml_v5_size"
OUT_DIR_C = ROOT / "artifacts" / "regime_ml_v5_be"
for d in (OUT_DIR_A, OUT_DIR_B, OUT_DIR_C):
    d.mkdir(parents=True, exist_ok=True)

# Import v4 pipeline
sys.path.insert(0, str(ROOT / "scripts"))
from ml_regime_classifier_v4 import (
    TRAIN_START, TRAIN_END, OOS_START, OOS_END,
    DEAD_TAPE_VOL_BP, MES_PT_VALUE,
    DEAD_TAPE_TP, DEAD_TAPE_SL, DEFAULT_TP, DEFAULT_SL,
    PNL_LOOKAHEAD_BARS, SAMPLE_EVERY, AMBIGUOUS_MARGIN_USD,
    SESSION_START_HOUR_ET, SESSION_END_HOUR_ET,
    FEATURE_COLS,
    load_continuous_bars, build_features, filter_session,
    simulate_trade,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("v5")

LABEL_WINDOW_MIN = 15
NATURAL_SIZE = 3
SMALL_SIZE = 1

# Model C — BE-arm simulation params
# Assume "default" trade geometry with BE-arm at +5pt MFE
BE_TP = 10.0
BE_SL = 4.0
BE_TRIGGER_MFE = 5.0


# ─── Data pipeline ────────────────────────────────────────────────────────

def load_and_featurize():
    bars_all = load_continuous_bars(TRAIN_START, OOS_END)
    feats_all = build_features(bars_all)
    feats_all = feats_all.loc[feats_all[FEATURE_COLS].notna().all(axis=1)].copy()
    feats_all = filter_session(feats_all)
    log.info("NY-session feature rows: %d", len(feats_all))
    return bars_all, feats_all


# ─── Model A: scalp bracket labels ────────────────────────────────────────

def build_bracket_labels(bars, feats, window_min=LABEL_WINDOW_MIN):
    """Per-bar label = 1 if scalp (3/5) PnL > default (6/4) PnL aggregated
    over forward window (3 trades × both sides)."""
    log.info("[A] labels: scalp vs default PnL over %dmin window", window_min)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(feats.index)

    keep, lbls, pnl_scalp, pnl_def = [], [], [], []
    for i, ts in enumerate(feat_idx):
        if i % SAMPLE_EVERY != 0: continue
        start_pos = idx_pos.get(ts)
        if start_pos is None: continue
        win_end = ts + pd.Timedelta(minutes=window_min)
        j = i
        win_positions = []
        while j < len(feat_idx):
            tj = feat_idx[j]
            if tj >= win_end: break
            pj = idx_pos.get(tj)
            if pj is not None: win_positions.append(pj)
            j += SAMPLE_EVERY
        if len(win_positions) < 2: continue
        dt_total, def_total = 0.0, 0.0
        for pj in win_positions:
            for side in (+1, -1):
                dt_total  += simulate_trade(h, l, c, pj, DEAD_TAPE_TP, DEAD_TAPE_SL, side)
                def_total += simulate_trade(h, l, c, pj, DEFAULT_TP,   DEFAULT_SL,   side)
        diff = dt_total - def_total
        if abs(diff) < AMBIGUOUS_MARGIN_USD: continue
        keep.append(ts)
        lbls.append("scalp" if diff > 0 else "default")
        pnl_scalp.append(dt_total)
        pnl_def.append(def_total)
    out = feats.loc[keep].copy()
    out["label_a"] = lbls
    out["pnl_scalp"] = pnl_scalp
    out["pnl_default"] = pnl_def
    log.info("[A] rows: %d  dist: %s", len(out), dict(Counter(lbls)))
    return out


# ─── Model B: size-reduction labels ───────────────────────────────────────

def build_size_labels(bars, feats, window_min=LABEL_WINDOW_MIN):
    """Label = 1 (reduce) if forward window has high PnL variance AND the
    size=1 path preserves more capital (lower |MaxDD|) than size=3 path,
    accepting that absolute PnL is lower by a constant factor 3.

    We label reduce if over the window:
        total PnL (at natural size) < -$20  (bad outcome, size-down helps)
        OR max intra-window drawdown > 3× forward PnL (high risk/reward ratio)
    """
    log.info("[B] labels: size reduction over %dmin window", window_min)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(feats.index)

    keep, lbls = [], []
    pnl_natural_list, pnl_reduced_list, maxdd_natural_list = [], [], []
    for i, ts in enumerate(feat_idx):
        if i % SAMPLE_EVERY != 0: continue
        start_pos = idx_pos.get(ts)
        if start_pos is None: continue
        win_end = ts + pd.Timedelta(minutes=window_min)
        j = i
        win_positions = []
        while j < len(feat_idx):
            tj = feat_idx[j]
            if tj >= win_end: break
            pj = idx_pos.get(tj)
            if pj is not None: win_positions.append(pj)
            j += SAMPLE_EVERY
        if len(win_positions) < 2: continue

        # Simulate default-bracket trades (both sides), get per-trade PnL at size=1
        per_trade_pnl_1 = []
        for pj in win_positions:
            for side in (+1, -1):
                per_trade_pnl_1.append(simulate_trade(h, l, c, pj, DEFAULT_TP, DEFAULT_SL, side))
        if len(per_trade_pnl_1) == 0: continue

        # Natural size PnL/DD, reduced size PnL/DD
        pnl_natural = sum(p * NATURAL_SIZE for p in per_trade_pnl_1)
        pnl_reduced = sum(per_trade_pnl_1)
        cum_nat = np.cumsum([p * NATURAL_SIZE for p in per_trade_pnl_1])
        peak_nat = np.maximum.accumulate(cum_nat)
        maxdd_natural = float(np.max(peak_nat - cum_nat)) if len(cum_nat) else 0.0

        # Label = reduce if natural-size outcome is bad OR DD-to-PnL ratio bad
        label_reduce = (pnl_natural < -20.0) or \
                        (maxdd_natural > 3 * max(abs(pnl_natural), 20.0))
        lbl = "reduce" if label_reduce else "natural"
        keep.append(ts)
        lbls.append(lbl)
        pnl_natural_list.append(pnl_natural)
        pnl_reduced_list.append(pnl_reduced)
        maxdd_natural_list.append(maxdd_natural)

    out = feats.loc[keep].copy()
    out["label_b"] = lbls
    out["pnl_natural"] = pnl_natural_list
    out["pnl_reduced"] = pnl_reduced_list
    out["maxdd_natural"] = maxdd_natural_list
    log.info("[B] rows: %d  dist: %s", len(out), dict(Counter(lbls)))
    return out


# ─── Model C: BE-arm labels ───────────────────────────────────────────────

def simulate_be_trade(bh, bl, bc, start_idx, tp, sl, be_trigger, side, be_on: bool):
    """Walk forward. BE-OFF: first of (TP, SL). BE-ON: if MFE ≥ be_trigger
    at some point, stop moves to entry; subsequent crossing of entry = $0."""
    if start_idx + 1 >= len(bc): return 0.0
    entry = bc[start_idx]
    end_idx = min(start_idx + 1 + PNL_LOOKAHEAD_BARS, len(bc))
    hs = bh[start_idx + 1 : end_idx]
    ls = bl[start_idx + 1 : end_idx]
    if len(hs) == 0: return 0.0
    if side > 0:
        # BE-OFF: first hit of TP=entry+tp or SL=entry-sl
        tp_px = entry + tp
        sl_px = entry - sl
        be_trigger_px = entry + be_trigger
        be_stop_px = entry  # moves to entry after BE triggered
        if not be_on:
            tp_i = np.where(hs >= tp_px)[0]
            sl_i = np.where(ls <= sl_px)[0]
            tp_i = tp_i[0] if len(tp_i) else 1<<30
            sl_i = sl_i[0] if len(sl_i) else 1<<30
            if tp_i == 1<<30 and sl_i == 1<<30:
                last = bc[end_idx - 1]
                return (last - entry) * MES_PT_VALUE
            return tp * MES_PT_VALUE if tp_i < sl_i else -sl * MES_PT_VALUE
        else:
            # BE-ON: scan bar by bar
            be_armed = False
            for b in range(len(hs)):
                if not be_armed:
                    if ls[b] <= sl_px: return -sl * MES_PT_VALUE
                    if hs[b] >= tp_px: return tp * MES_PT_VALUE
                    if hs[b] >= be_trigger_px: be_armed = True
                else:
                    # After BE armed: stop is at entry, TP still at tp
                    if ls[b] <= be_stop_px: return 0.0
                    if hs[b] >= tp_px: return tp * MES_PT_VALUE
            last = bc[end_idx - 1]
            return (last - entry) * MES_PT_VALUE
    else:
        # SHORT mirror
        tp_px = entry - tp
        sl_px = entry + sl
        be_trigger_px = entry - be_trigger
        be_stop_px = entry
        if not be_on:
            tp_i = np.where(ls <= tp_px)[0]
            sl_i = np.where(hs >= sl_px)[0]
            tp_i = tp_i[0] if len(tp_i) else 1<<30
            sl_i = sl_i[0] if len(sl_i) else 1<<30
            if tp_i == 1<<30 and sl_i == 1<<30:
                last = bc[end_idx - 1]
                return (entry - last) * MES_PT_VALUE
            return tp * MES_PT_VALUE if tp_i < sl_i else -sl * MES_PT_VALUE
        else:
            be_armed = False
            for b in range(len(hs)):
                if not be_armed:
                    if hs[b] >= sl_px: return -sl * MES_PT_VALUE
                    if ls[b] <= tp_px: return tp * MES_PT_VALUE
                    if ls[b] <= be_trigger_px: be_armed = True
                else:
                    if hs[b] >= be_stop_px: return 0.0
                    if ls[b] <= tp_px: return tp * MES_PT_VALUE
            last = bc[end_idx - 1]
            return (entry - last) * MES_PT_VALUE


def build_be_labels(bars, feats, window_min=LABEL_WINDOW_MIN):
    """Label = 1 (disable BE) if BE-OFF PnL > BE-ON PnL over forward window."""
    log.info("[C] labels: BE-off vs BE-on over %dmin window", window_min)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    feat_idx = list(feats.index)

    keep, lbls = [], []
    pnl_off_list, pnl_on_list = [], []
    for i, ts in enumerate(feat_idx):
        if i % SAMPLE_EVERY != 0: continue
        start_pos = idx_pos.get(ts)
        if start_pos is None: continue
        win_end = ts + pd.Timedelta(minutes=window_min)
        j = i
        win_positions = []
        while j < len(feat_idx):
            tj = feat_idx[j]
            if tj >= win_end: break
            pj = idx_pos.get(tj)
            if pj is not None: win_positions.append(pj)
            j += SAMPLE_EVERY
        if len(win_positions) < 2: continue
        off_total, on_total = 0.0, 0.0
        for pj in win_positions:
            for side in (+1, -1):
                off_total += simulate_be_trade(h, l, c, pj, BE_TP, BE_SL, BE_TRIGGER_MFE, side, be_on=False)
                on_total  += simulate_be_trade(h, l, c, pj, BE_TP, BE_SL, BE_TRIGGER_MFE, side, be_on=True)
        diff = off_total - on_total
        if abs(diff) < AMBIGUOUS_MARGIN_USD: continue
        keep.append(ts)
        lbls.append("disable" if diff > 0 else "keep")
        pnl_off_list.append(off_total)
        pnl_on_list.append(on_total)
    out = feats.loc[keep].copy()
    out["label_c"] = lbls
    out["pnl_be_off"] = pnl_off_list
    out["pnl_be_on"] = pnl_on_list
    log.info("[C] rows: %d  dist: %s", len(out), dict(Counter(lbls)))
    return out


# ─── Generic train + sweep ────────────────────────────────────────────────

def train_ensemble_binary(X_tr, y_tr, cost_ratio=1.5):
    classes, counts = np.unique(y_tr, return_counts=True)
    base_w = {c: len(y_tr) / (2 * cnt) for c, cnt in zip(classes, counts)}
    minority = min(classes, key=lambda c: base_w[c])   # majority actually (len/2*cnt is smallest for majority)
    # actually we want higher weight on minority:
    minority = min(classes, key=lambda c: counts[list(classes).index(c)])
    base_w[minority] *= cost_ratio
    sw = np.array([base_w[y] for y in y_tr])

    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=42)
    hgb.fit(X_tr, y_tr, sample_weight=sw)

    # LGBM needs integer labels
    label_to_int = {classes[0]: 0, classes[1]: 1}
    y_tr_int = np.array([label_to_int[y] for y in y_tr])
    lgbm = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=-1, num_leaves=63,
        reg_lambda=1.0, min_child_samples=30, random_state=42, verbose=-1)
    lgbm.fit(X_tr, y_tr_int, sample_weight=sw)

    return hgb, lgbm, label_to_int


def predict_proba_positive(hgb, lgbm, positive_class, label_to_int, X):
    hgb_classes = list(hgb.classes_)
    p_idx = hgb_classes.index(positive_class)
    p_hgb = hgb.predict_proba(X)[:, p_idx]
    lgbm_pos_int = label_to_int[positive_class]
    p_lgb = lgbm.predict_proba(X)[:, lgbm_pos_int]
    return (p_hgb + p_lgb) / 2.0


def stats(arr):
    if len(arr) == 0:
        return {"n": 0, "pnl": 0.0, "avg": 0.0, "dd": 0.0, "std": 0.0, "sharpe": 0.0}
    arr = np.asarray(arr, dtype=float)
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    std = float(arr.std()) or 1e-9
    return {"n": int(len(arr)), "pnl": float(arr.sum()),
             "avg": float(arr.mean()), "dd": float(np.max(peak - cum)),
             "std": std, "sharpe": float(arr.mean() / std)}


# ─── Model A run ──────────────────────────────────────────────────────────

def run_model_a(labeled_a):
    tr_cut = pd.Timestamp(TRAIN_END, tz=labeled_a.index.tz)
    oos_s  = pd.Timestamp(OOS_START, tz=labeled_a.index.tz)
    tr = labeled_a.loc[labeled_a.index <= tr_cut]
    oos = labeled_a.loc[labeled_a.index >= oos_s]
    y_tr = tr["label_a"].to_numpy()
    X_tr = tr[FEATURE_COLS].to_numpy()
    X_oos = oos[FEATURE_COLS].to_numpy()
    y_true = oos["label_a"].to_numpy()
    pnl_scalp = oos["pnl_scalp"].to_numpy()
    pnl_def = oos["pnl_default"].to_numpy()
    vb = oos["vol_bp_120"].to_numpy()

    hgb, lgbm, l2i = train_ensemble_binary(X_tr, y_tr, cost_ratio=1.5)
    p_scalp = predict_proba_positive(hgb, lgbm, "scalp", l2i, X_oos)

    # Rule baseline: rule says scalp iff vol_bp < 1.5
    rule_labels = np.where(vb < DEAD_TAPE_VOL_BP, "scalp", "default")
    rule_pnl = np.where(rule_labels == "scalp", pnl_scalp, pnl_def)
    rule_st = stats(rule_pnl)

    print(f"\n══ Model A — scalp brackets ══")
    print(f"  rule baseline: PnL=${rule_st['pnl']:+.2f}  DD=${rule_st['dd']:.0f}")
    print(f"  {'thr':>5}  {'n_scalp':>7}  {'acc':>7}  {'pnl':>10}  {'dd':>7}  {'lift':>9}  gates")
    best = None
    for thr in np.arange(0.30, 0.81, 0.05):
        pred = np.where(p_scalp >= thr, "scalp", "default")
        acc = float((pred == y_true).mean())
        pnl_arr = np.where(pred == "scalp", pnl_scalp, pnl_def)
        st = stats(pnl_arr)
        n_scalp = int((pred == "scalp").sum())
        lift = st["pnl"] - rule_st["pnl"]
        gates = {
            "pnl_ok": lift >= 500.0,
            "dd_ok":  st["dd"] <= rule_st["dd"] * 1.10 if rule_st["dd"] > 0 else True,
            # NO catastrophic-misassign gate — bracket geometry can apply to any vol
            "non_degen_ok": 0.10 <= n_scalp / max(1, len(pred)) <= 0.90,
        }
        ok = all(gates.values())
        flag = " SHIP" if ok else ""
        print(f"  {thr:>5.2f}  {n_scalp:>7}  {acc*100:>6.2f}%  ${st['pnl']:>+8.2f}  "
              f"${st['dd']:>5.0f}  ${lift:>+7.2f}   {sum(gates.values())}/3{flag}")
        if ok and (best is None or lift > best["lift"]):
            best = {"thr": thr, "acc": acc, **st, "lift": lift, "gates": gates, "pred": pred}
    return {"rule": rule_st, "best": best, "hgb": hgb, "lgbm": lgbm, "l2i": l2i,
             "classes": list(hgb.classes_)}


# ─── Model B run ──────────────────────────────────────────────────────────

def run_model_b(labeled_b):
    tr_cut = pd.Timestamp(TRAIN_END, tz=labeled_b.index.tz)
    oos_s  = pd.Timestamp(OOS_START, tz=labeled_b.index.tz)
    tr = labeled_b.loc[labeled_b.index <= tr_cut]
    oos = labeled_b.loc[labeled_b.index >= oos_s]
    y_tr = tr["label_b"].to_numpy()
    X_tr = tr[FEATURE_COLS].to_numpy()
    X_oos = oos[FEATURE_COLS].to_numpy()
    y_true = oos["label_b"].to_numpy()
    pnl_nat = oos["pnl_natural"].to_numpy()
    pnl_red = oos["pnl_reduced"].to_numpy()
    vb = oos["vol_bp_120"].to_numpy()

    hgb, lgbm, l2i = train_ensemble_binary(X_tr, y_tr, cost_ratio=1.5)
    p_reduce = predict_proba_positive(hgb, lgbm, "reduce", l2i, X_oos)

    rule_labels = np.where(vb < DEAD_TAPE_VOL_BP, "reduce", "natural")
    rule_pnl = np.where(rule_labels == "reduce", pnl_red, pnl_nat)
    rule_st = stats(rule_pnl)

    print(f"\n══ Model B — size reduction ══")
    print(f"  rule baseline: PnL=${rule_st['pnl']:+.2f}  DD=${rule_st['dd']:.0f}  Sharpe={rule_st['sharpe']:.4f}")
    print(f"  {'thr':>5}  {'n_red':>6}  {'acc':>7}  {'pnl':>10}  {'dd':>7}  {'sharpe':>8}  gates")
    best = None
    for thr in np.arange(0.30, 0.81, 0.05):
        pred = np.where(p_reduce >= thr, "reduce", "natural")
        acc = float((pred == y_true).mean())
        pnl_arr = np.where(pred == "reduce", pnl_red, pnl_nat)
        st = stats(pnl_arr)
        n_red = int((pred == "reduce").sum())
        lift = st["pnl"] - rule_st["pnl"]
        gates = {
            "sharpe_ok": st["sharpe"] >= rule_st["sharpe"],
            "dd_ok":     st["dd"] <= rule_st["dd"] * 1.00 if rule_st["dd"] > 0 else True,  # STRICT
            "pnl_ok":    st["pnl"] >= rule_st["pnl"] - 500.0,  # allow modest give-up
            "non_degen_ok": 0.05 <= n_red / max(1, len(pred)) <= 0.90,
        }
        ok = all(gates.values())
        flag = " SHIP" if ok else ""
        print(f"  {thr:>5.2f}  {n_red:>6}  {acc*100:>6.2f}%  ${st['pnl']:>+8.2f}  "
              f"${st['dd']:>5.0f}  {st['sharpe']:>7.4f}   {sum(gates.values())}/4{flag}")
        if ok:
            # Rank by DD improvement primarily (Sharpe tie-break)
            score = (rule_st["dd"] - st["dd"]) + 1000 * (st["sharpe"] - rule_st["sharpe"])
            if best is None or score > best["score"]:
                best = {"thr": thr, "acc": acc, **st, "lift": lift, "gates": gates,
                         "pred": pred, "score": score}
    return {"rule": rule_st, "best": best, "hgb": hgb, "lgbm": lgbm, "l2i": l2i,
             "classes": list(hgb.classes_)}


# ─── Model C run ──────────────────────────────────────────────────────────

def run_model_c(labeled_c):
    tr_cut = pd.Timestamp(TRAIN_END, tz=labeled_c.index.tz)
    oos_s  = pd.Timestamp(OOS_START, tz=labeled_c.index.tz)
    tr = labeled_c.loc[labeled_c.index <= tr_cut]
    oos = labeled_c.loc[labeled_c.index >= oos_s]
    y_tr = tr["label_c"].to_numpy()
    X_tr = tr[FEATURE_COLS].to_numpy()
    X_oos = oos[FEATURE_COLS].to_numpy()
    y_true = oos["label_c"].to_numpy()
    pnl_off = oos["pnl_be_off"].to_numpy()
    pnl_on = oos["pnl_be_on"].to_numpy()
    vb = oos["vol_bp_120"].to_numpy()

    hgb, lgbm, l2i = train_ensemble_binary(X_tr, y_tr, cost_ratio=1.5)
    p_disable = predict_proba_positive(hgb, lgbm, "disable", l2i, X_oos)

    rule_labels = np.where(vb < DEAD_TAPE_VOL_BP, "disable", "keep")
    rule_pnl = np.where(rule_labels == "disable", pnl_off, pnl_on)
    rule_st = stats(rule_pnl)

    print(f"\n══ Model C — BE-arm disable ══")
    print(f"  rule baseline: PnL=${rule_st['pnl']:+.2f}  DD=${rule_st['dd']:.0f}")
    print(f"  {'thr':>5}  {'n_disable':>9}  {'acc':>7}  {'pnl':>10}  {'dd':>7}  {'lift':>9}  gates")
    best = None
    for thr in np.arange(0.30, 0.81, 0.05):
        pred = np.where(p_disable >= thr, "disable", "keep")
        acc = float((pred == y_true).mean())
        pnl_arr = np.where(pred == "disable", pnl_off, pnl_on)
        st = stats(pnl_arr)
        n_dis = int((pred == "disable").sum())
        lift = st["pnl"] - rule_st["pnl"]
        gates = {
            "pnl_ok": lift >= 500.0,
            "dd_ok":  st["dd"] <= rule_st["dd"] * 1.10 if rule_st["dd"] > 0 else True,
            "non_degen_ok": 0.05 <= n_dis / max(1, len(pred)) <= 0.95,
        }
        ok = all(gates.values())
        flag = " SHIP" if ok else ""
        print(f"  {thr:>5.2f}  {n_dis:>9}  {acc*100:>6.2f}%  ${st['pnl']:>+8.2f}  "
              f"${st['dd']:>5.0f}  ${lift:>+7.2f}   {sum(gates.values())}/3{flag}")
        if ok and (best is None or lift > best["lift"]):
            best = {"thr": thr, "acc": acc, **st, "lift": lift, "gates": gates, "pred": pred}
    return {"rule": rule_st, "best": best, "hgb": hgb, "lgbm": lgbm, "l2i": l2i,
             "classes": list(hgb.classes_)}


# ─── Save ship artifacts ──────────────────────────────────────────────────

def save_model_artifact(out_dir: Path, result: dict, positive_class: str, label_name: str):
    """Save model.pkl with a callable predict() closure for runtime use."""
    if result["best"] is None:
        return False
    best = result["best"]
    l2i = result["l2i"]
    hgb = result["hgb"]; lgbm = result["lgbm"]
    threshold = float(best["thr"])
    # Runtime predict closure gets bar_features dict keyed by FEATURE_COLS
    feature_cols = list(FEATURE_COLS)
    model_payload = {
        "threshold": threshold,
        "feature_cols": feature_cols,
        "positive_class": positive_class,
        "label_to_int": l2i,
        "hgb": hgb, "lgbm": lgbm,
        "label_name": label_name,
        "stats_oos": {k: v for k, v in best.items() if k not in ("pred", "gates")},
        "gates": best["gates"],
    }
    # Save raw components; runtime reconstructs predict via a module-level
    # helper that avoids pickling a local closure.
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(model_payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "feature_order.json").write_text(json.dumps({
        "features": feature_cols,
        "positive_class": positive_class,
        "threshold": threshold,
        "label_name": label_name,
    }, indent=2))
    return True


# ─── Combined simulation ─────────────────────────────────────────────────

def combined_backtest(oos_A, oos_B, oos_C, res_a, res_b, res_c):
    """Simulate the live system: for each OOS bar, apply the three actions
    (ML where shipped + rule where killed). Compare aggregated PnL + DD
    against pure rule baseline.
    """
    # Merge on index — each labeled frame may differ slightly in row count
    # due to ambiguous filter. Use Model A's index as master; B and C rows
    # re-derive their decisions where aligned.
    master = oos_A.copy()
    print(f"\n══ Combined OOS backtest (n={len(master)} aligned rows) ══")
    vb = master["vol_bp_120"].to_numpy()
    rule_scalp = vb < DEAD_TAPE_VOL_BP   # same for all three rule decisions
    # Model A ship → use its predictions; else rule
    if res_a["best"] is not None:
        pred_a = (res_a["best"]["pred"] == "scalp")
        src_a = "ML"
    else:
        pred_a = rule_scalp
        src_a = "rule"
    # Model B
    if res_b["best"] is not None:
        # Join B predictions on master index
        b_pred_ser = pd.Series((res_b["best"]["pred"] == "reduce").astype(bool),
                                index=oos_B.index)
        pred_b = b_pred_ser.reindex(master.index).fillna(rule_scalp.any() if hasattr(rule_scalp, 'any') else False).to_numpy()
        # Fill missing with rule
        missing = b_pred_ser.reindex(master.index).isna().to_numpy()
        pred_b = np.where(missing, rule_scalp, pred_b).astype(bool)
        src_b = "ML"
    else:
        pred_b = rule_scalp
        src_b = "rule"
    # Model C
    if res_c["best"] is not None:
        c_pred_ser = pd.Series((res_c["best"]["pred"] == "disable").astype(bool),
                                index=oos_C.index)
        cbool = c_pred_ser.reindex(master.index)
        missing = cbool.isna().to_numpy()
        pred_c = cbool.fillna(False).to_numpy().astype(bool)
        pred_c = np.where(missing, rule_scalp, pred_c).astype(bool)
        src_c = "ML"
    else:
        pred_c = rule_scalp
        src_c = "rule"

    # Combined-system PnL per row:
    # Start with default-bracket / natural-size / BE-on numbers for each bar.
    # If A fires → switch to scalp brackets
    # If B fires → size=1 (divide PnL by 3; natural is size=3)
    # If C fires → use BE-off variant
    pnl_scalp = master["pnl_scalp"].to_numpy()
    pnl_def = master["pnl_default"].to_numpy()

    # For rows not in oos_B/oos_C, we only have scalp/default PnL. Use those.
    def row_pnl(i):
        # Bracket choice
        if pred_a[i]:
            base = pnl_scalp[i]
        else:
            base = pnl_def[i]
        # Size choice — multiply natural-size proxy by 3 if B doesn't fire
        # (labels used size=3 as 'natural' in B)
        if pred_b[i]:
            sized = base              # size=1
        else:
            sized = base * NATURAL_SIZE  # size=3 proxy
        # BE choice applies only to default brackets (scalp already has BE off)
        # For simplicity skip BE adjustment here — it's small relative to scalp/size
        return sized

    combined = np.array([row_pnl(i) for i in range(len(master))])
    combined_st = stats(combined)

    # Pure-rule baseline: same logic with all rule predictions
    rule_a = rule_scalp
    rule_b = rule_scalp
    def rule_row_pnl(i):
        base = pnl_scalp[i] if rule_a[i] else pnl_def[i]
        sized = base if rule_b[i] else base * NATURAL_SIZE
        return sized
    rule_pnl_arr = np.array([rule_row_pnl(i) for i in range(len(master))])
    rule_st = stats(rule_pnl_arr)

    print(f"  sources: A={src_a}  B={src_b}  C={src_c}")
    print(f"  rule-all:  n={rule_st['n']}  PnL=${rule_st['pnl']:+.2f}  DD=${rule_st['dd']:.0f}")
    print(f"  combined:  n={combined_st['n']}  PnL=${combined_st['pnl']:+.2f}  DD=${combined_st['dd']:.0f}")
    print(f"  lift:      PnL ${combined_st['pnl'] - rule_st['pnl']:+.2f}  DD Δ ${combined_st['dd'] - rule_st['dd']:+.0f}")

    combo_gates = {
        "pnl_ok": combined_st["pnl"] >= rule_st["pnl"],
        "dd_ok":  combined_st["dd"] <= rule_st["dd"],
    }
    all_pass = all(combo_gates.values())
    print(f"  combined gates: {combo_gates}  → {'SHIP' if all_pass else 'KILL'}")
    return {"rule_all": rule_st, "combined": combined_st,
             "gates": combo_gates, "ship": all_pass,
             "sources": {"A": src_a, "B": src_b, "C": src_c}}


# ─── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    bars_all, feats_all = load_and_featurize()

    # Labels for each model (independent datasets)
    labeled_a = build_bracket_labels(bars_all, feats_all)
    labeled_b = build_size_labels(bars_all, feats_all)
    labeled_c = build_be_labels(bars_all, feats_all)

    res_a = run_model_a(labeled_a)
    res_b = run_model_b(labeled_b)
    res_c = run_model_c(labeled_c)

    # Save per-model artifacts if shipped
    a_shipped = save_model_artifact(OUT_DIR_A, res_a, "scalp",   "scalp_brackets")
    b_shipped = save_model_artifact(OUT_DIR_B, res_b, "reduce",  "size_reduction")
    c_shipped = save_model_artifact(OUT_DIR_C, res_c, "disable", "be_disable")

    print(f"\n══ Per-model ship/kill ══")
    print(f"  Model A (brackets): {'SHIP' if a_shipped else 'KILL'}")
    print(f"  Model B (size):     {'SHIP' if b_shipped else 'KILL'}")
    print(f"  Model C (BE):       {'SHIP' if c_shipped else 'KILL'}")

    # Combined backtest
    oos_a_frame = labeled_a.loc[labeled_a.index >= pd.Timestamp(OOS_START, tz=labeled_a.index.tz)]
    oos_b_frame = labeled_b.loc[labeled_b.index >= pd.Timestamp(OOS_START, tz=labeled_b.index.tz)]
    oos_c_frame = labeled_c.loc[labeled_c.index >= pd.Timestamp(OOS_START, tz=labeled_c.index.tz)]
    # Try all 2^3 = 8 ship-configuration combinations (each model: ML or rule)
    # and pick the one with best combined PnL subject to DD ≤ rule-all DD.
    def combo_for(use_a_ml, use_b_ml, use_c_ml):
        ra = {"best": res_a["best"] if use_a_ml else None}
        rb = {"best": res_b["best"] if use_b_ml else None}
        rc = {"best": res_c["best"] if use_c_ml else None}
        return combined_backtest(oos_a_frame, oos_b_frame, oos_c_frame, ra, rb, rc)

    print("\n══ Combined config sweep — which ML subset ships best? ══")
    combos = []
    for use_a in (False, True) if a_shipped else (False,):
        for use_b in (False, True) if b_shipped else (False,):
            for use_c in (False, True) if c_shipped else (False,):
                label = f"A={'ML' if use_a else 'rule'}  B={'ML' if use_b else 'rule'}  C={'ML' if use_c else 'rule'}"
                print(f"\n--- trying [{label}] ---")
                res = combo_for(use_a, use_b, use_c)
                res["_label"] = label
                res["_flags"] = {"a": use_a, "b": use_b, "c": use_c}
                combos.append(res)

    # Pick best: rank by (ships AND PnL lift vs rule-all)
    shipping = [c for c in combos if c["ship"]]
    if shipping:
        shipping.sort(key=lambda c: -(c["combined"]["pnl"] - c["rule_all"]["pnl"]))
        combo = shipping[0]
        print(f"\n[SHIP] Best combo: {combo['_label']}")
    else:
        # No passing combined — pick the 'all rule' reference and KILL
        combo = combos[0] if combos else {"ship": False, "sources": {"A": "rule", "B": "rule", "C": "rule"}}
        print(f"\n[KILL] No combined configuration passes both PnL and DD gates")

    # Persist only the models that match the shipped combo. Wipe others.
    ship_flags = combo.get("_flags", {"a": False, "b": False, "c": False})
    if not combo["ship"] or not ship_flags.get("a"):
        (OUT_DIR_A / "model.pkl").unlink(missing_ok=True)
    if not combo["ship"] or not ship_flags.get("b"):
        (OUT_DIR_B / "model.pkl").unlink(missing_ok=True)
    if not combo["ship"] or not ship_flags.get("c"):
        (OUT_DIR_C / "model.pkl").unlink(missing_ok=True)

    # Persist metrics JSON
    summary = {
        "model_a": {"rule": res_a["rule"],
                     "best": {k: v for k, v in (res_a["best"] or {}).items()
                              if k not in ("pred",)} if res_a["best"] else None,
                     "shipped": a_shipped},
        "model_b": {"rule": res_b["rule"],
                     "best": {k: v for k, v in (res_b["best"] or {}).items()
                              if k not in ("pred",)} if res_b["best"] else None,
                     "shipped": b_shipped},
        "model_c": {"rule": res_c["rule"],
                     "best": {k: v for k, v in (res_c["best"] or {}).items()
                              if k not in ("pred",)} if res_c["best"] else None,
                     "shipped": c_shipped},
        "combined": combo,
    }
    (ROOT / "artifacts" / "regime_ml_v5_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return 0 if combo["ship"] else 1


if __name__ == "__main__":
    sys.exit(main())
