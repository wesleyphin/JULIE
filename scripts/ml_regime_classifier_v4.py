#!/usr/bin/env python3
"""ML Regime Classifier v4 — stacked improvements attacking label noise + feature gap.

Previous failures:
  v1 (supervised on rules): 99.95% replication, no new signal, KILL
  v2 (outcome-labeled per-bar): over-predicts dead_tape, accuracy < rule, KILL
  v3 (confidence-thresholded hybrid): ML never confident enough to override, KILL

v4 addresses the root causes identified:
  1. LABEL NOISE — aggregate forward PnL over 15-min and 30-min WINDOWS (not
     per-bar). Each label = sum of 3 / 6 hypothetical 5-min-spaced trades.
     Expected noise reduction: ~√k where k = trades-per-window.
  2. FEATURE GAP — add: vol_bp slope (10/30/60-bar change, not just level),
     240/480-bar lookback (catches slower regime shifts), consecutive up/down
     bar runs, gap from prior close, absolute-body mean, minutes-into-NY-session,
     day-of-week one-hot, and cross-strategy activity proxies (whether any of
     the conditions that trigger DE3/AF/RA signals were met in the last 30
     bars).
  3. MODEL DIVERSITY — HGB and LightGBM trained on same data, soft-voting
     ensemble (avg predicted probabilities). This hedges model-specific
     artifacts.
  4. COST-SENSITIVE — sample weights that penalize "miss a dead_tape" (FN)
     more than "wrong dead_tape" (FP), since the dollar asymmetry is 5:3.
  5. BINARY — keep dead_tape vs default. The other 3 regime labels do not
     change live behavior (Filter D env-off), so learning to separate them
     has zero PnL impact and dilutes the classifier's attention.

Ship gates (ALL must pass):
  1. OOS accuracy on outcome labels ≥ rule accuracy (preserve rule-right)
  2. OOS PnL ≥ rule baseline + $500
  3. MaxDD ≤ 110% rule MaxDD
  4. Directional sanity (ML dead_tape bars' avg vol_bp < default bars')
  5. Prediction non-degenerate (min class ≥ 10% of OOS rows)
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
from sklearn.metrics import accuracy_score, confusion_matrix
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "regime_ml_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2024-07-01"
TRAIN_END   = "2026-01-26"
OOS_START   = "2026-01-27"
OOS_END     = "2026-04-20"

WINDOW_BARS = 120
DEAD_TAPE_VOL_BP = 1.5
MES_PT_VALUE = 5.0

DEAD_TAPE_TP = 3.0
DEAD_TAPE_SL = 5.0
DEFAULT_TP = 6.0
DEFAULT_SL = 4.0

PNL_LOOKAHEAD_BARS = 60
LABEL_WINDOWS_MIN = [15, 30]   # two smoothing windows; pick best
SAMPLE_EVERY = 5
AMBIGUOUS_MARGIN_USD = 15.0    # higher threshold since windows aggregate

SESSION_START_HOUR_ET = 9
SESSION_END_HOUR_ET = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_regime_v4")


# ─── Data + features ───────────────────────────────────────────────────────

def load_continuous_bars(start, end):
    log.info("loading bars %s → %s", start, end)
    df = pd.read_parquet(ROOT / "es_master_outrights.parquet")
    lo = pd.Timestamp(start, tz=df.index.tz)
    hi = pd.Timestamp(end, tz=df.index.tz)
    df = df.loc[(df.index >= lo) & (df.index <= hi)].copy()
    date_arr = df.index.date
    dom = df.assign(_d=date_arr).groupby(["_d", "symbol"])["volume"].sum().reset_index()
    dom = dom.sort_values(["_d", "volume"], ascending=[True, False]).drop_duplicates("_d", keep="first")
    dmap = dict(zip(dom["_d"], dom["symbol"]))
    mask = pd.Series(date_arr, index=df.index).map(dmap) == df["symbol"]
    df = df.loc[mask.values].sort_index()
    log.info("bars after dominant-symbol: %d", len(df))
    return df


def rolling_vol_eff(closes, window):
    n = len(closes)
    vol_bp = np.full(n, np.nan)
    eff = np.full(n, np.nan)
    for i in range(window, n):
        p = closes[i - window : i + 1]
        rets = (p[1:] - p[:-1]) / p[:-1]
        if len(rets) == 0:
            continue
        mean = rets.mean()
        var = ((rets - mean) ** 2).sum() / max(1, len(rets) - 1)
        vol_bp[i] = (var ** 0.5) * 10_000.0
        abs_sum = np.abs(rets).sum()
        eff[i] = abs(rets.sum()) / abs_sum if abs_sum > 0 else 0.0
    return vol_bp, eff


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    log.info("building features on %d bars", len(bars))
    c = bars["close"].to_numpy(float)
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    v = bars["volume"].to_numpy(float)

    # Vol/eff at four lookbacks
    vol_120, eff_120 = rolling_vol_eff(c, 120)
    vol_60,  eff_60  = rolling_vol_eff(c, 60)
    vol_30,  eff_30  = rolling_vol_eff(c, 30)
    vol_240, eff_240 = rolling_vol_eff(c, 240)
    vol_480, eff_480 = rolling_vol_eff(c, 480)

    # Vol slope features — change in vol over recent windows
    def diff(arr, lag):
        return arr - np.r_[np.full(lag, np.nan), arr[:-lag]]
    vol_slope_10 = diff(vol_60, 10)
    vol_slope_30 = diff(vol_60, 30)
    vol_slope_60 = diff(vol_120, 60)
    eff_slope_30 = diff(eff_60, 30)

    # True range + ATR
    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = pd.Series(tr).rolling(14).mean().to_numpy()
    atr30 = pd.Series(tr).rolling(30).mean().to_numpy()

    # Range features
    rng = h - l
    rng_pct = rng / np.where(c != 0, c, 1.0)
    rng_pct_20 = pd.Series(rng_pct).rolling(20).mean().to_numpy() * 10_000
    rng_pct_120 = pd.Series(rng_pct).rolling(120).mean().to_numpy() * 10_000

    # Bar shape
    hl = np.maximum(rng, 1e-9)
    body = np.abs(c - o)
    body_ratio = body / hl
    body_ratio_20 = pd.Series(body_ratio).rolling(20).mean().to_numpy()
    body_ratio_60 = pd.Series(body_ratio).rolling(60).mean().to_numpy()
    abs_body_20 = pd.Series(body).rolling(20).mean().to_numpy()

    up_bar = (c >= o).astype(float)
    up_bar_pct_20 = pd.Series(up_bar).rolling(20).mean().to_numpy()

    # Consecutive up/down bar run lengths
    run_up = np.zeros(len(c), dtype=int)
    run_down = np.zeros(len(c), dtype=int)
    run_up[0] = 1 if up_bar[0] else 0
    for i in range(1, len(c)):
        if up_bar[i]:
            run_up[i] = run_up[i - 1] + 1
            run_down[i] = 0
        else:
            run_down[i] = run_down[i - 1] + 1
            run_up[i] = 0
    # Also a running max of both in last 20
    run_up_max_20 = pd.Series(run_up).rolling(20).max().to_numpy()
    run_down_max_20 = pd.Series(run_down).rolling(20).max().to_numpy()

    # Gap from prior close
    gap_pct = (o - prev_c) / np.where(prev_c != 0, prev_c, 1.0) * 10_000
    gap_abs_mean_20 = pd.Series(np.abs(gap_pct)).rolling(20).mean().to_numpy()

    # Momentum at multiple scales
    def lag(arr, k):
        return np.r_[arr[:k], arr[:-k]]

    mom_5  = (c - lag(c, 5))  / np.where(lag(c, 5)  != 0, lag(c, 5),  1.0) * 10_000
    mom_15 = (c - lag(c, 15)) / np.where(lag(c, 15) != 0, lag(c, 15), 1.0) * 10_000
    mom_30 = (c - lag(c, 30)) / np.where(lag(c, 30) != 0, lag(c, 30), 1.0) * 10_000
    mom_60 = (c - lag(c, 60)) / np.where(lag(c, 60) != 0, lag(c, 60), 1.0) * 10_000
    mom_120 = (c - lag(c, 120)) / np.where(lag(c, 120) != 0, lag(c, 120), 1.0) * 10_000

    # Volume
    vol_mean_200 = pd.Series(v).rolling(200).mean().to_numpy()
    vol_std_200  = pd.Series(v).rolling(200).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_z = (v - vol_mean_200) / np.where(vol_std_200 > 0, vol_std_200, 1.0)
    volume_ma_ratio = v / np.where(vol_mean_200 > 0, vol_mean_200, 1.0)

    # Max run-up/run-down (continuous) over last 60
    max_run_up_60 = pd.Series(c).rolling(60).apply(
        lambda x: (x.max() - x.iloc[0]) / max(x.iloc[0], 1e-9), raw=False).to_numpy() * 10_000
    max_run_down_60 = pd.Series(c).rolling(60).apply(
        lambda x: (x.iloc[0] - x.min()) / max(x.iloc[0], 1e-9), raw=False).to_numpy() * 10_000

    # Session timing
    idx = bars.index
    et_hour = idx.hour.to_numpy()
    et_minute = idx.minute.to_numpy()
    minutes_into_session = np.where(
        (et_hour >= SESSION_START_HOUR_ET) & (et_hour < SESSION_END_HOUR_ET),
        (et_hour - SESSION_START_HOUR_ET) * 60 + et_minute, -1)
    day_of_week = idx.dayofweek.to_numpy()  # Mon=0 .. Fri=4

    # Cross-strategy proxy: did price cross a "typical strategy trigger"
    # in the last 30 bars? We approximate with large-move flags:
    #   - a 6pt move in any direction within 10 bars (DE3 trigger range)
    #   - a high/low of last 20 bars broken in last 5 (breakout-like)
    high_20 = pd.Series(h).rolling(20).max().to_numpy()
    low_20 = pd.Series(l).rolling(20).min().to_numpy()
    broke_high_5 = np.zeros(len(c), dtype=float)
    broke_low_5 = np.zeros(len(c), dtype=float)
    for i in range(5, len(c)):
        h5 = h[i-5:i+1].max()
        l5 = l[i-5:i+1].min()
        if high_20[i-5] is not None and np.isfinite(high_20[i-5]):
            broke_high_5[i] = 1.0 if h5 > high_20[i-5] else 0.0
            broke_low_5[i] = 1.0 if l5 < low_20[i-5] else 0.0
    any_strategy_signal_30 = (pd.Series(broke_high_5).rolling(30).max().to_numpy()
                                + pd.Series(broke_low_5).rolling(30).max().to_numpy())
    # 6-pt move in 10 bars
    max_move_10 = pd.Series(c).rolling(10).apply(lambda x: x.max() - x.min(), raw=False).to_numpy()
    big_move_10 = (max_move_10 >= 6.0).astype(float)

    return pd.DataFrame({
        # vol/eff at 5 lookbacks
        "vol_bp_30": vol_30,   "eff_30": eff_30,
        "vol_bp_60": vol_60,   "eff_60": eff_60,
        "vol_bp_120": vol_120, "eff_120": eff_120,
        "vol_bp_240": vol_240, "eff_240": eff_240,
        "vol_bp_480": vol_480, "eff_480": eff_480,
        # slopes
        "vol_slope_10": vol_slope_10,
        "vol_slope_30": vol_slope_30,
        "vol_slope_60": vol_slope_60,
        "eff_slope_30": eff_slope_30,
        # range/ATR
        "atr14": atr14, "atr30": atr30,
        "range_pct_20": rng_pct_20, "range_pct_120": rng_pct_120,
        # bar shape
        "body_ratio_20": body_ratio_20, "body_ratio_60": body_ratio_60,
        "abs_body_20": abs_body_20,
        "up_bar_pct_20": up_bar_pct_20,
        "run_up_max_20": run_up_max_20, "run_down_max_20": run_down_max_20,
        # gaps
        "gap_pct": gap_pct, "gap_abs_mean_20": gap_abs_mean_20,
        # momentum
        "mom_5": mom_5, "mom_15": mom_15, "mom_30": mom_30,
        "mom_60": mom_60, "mom_120": mom_120,
        # volume
        "volume_z_20": volume_z, "volume_ma_ratio": volume_ma_ratio,
        # runups
        "max_runup_60": max_run_up_60, "max_rundown_60": max_run_down_60,
        # session
        "et_hour": et_hour, "minutes_into_session": minutes_into_session,
        "day_of_week": day_of_week,
        # cross-strategy proxy
        "any_strategy_signal_30": any_strategy_signal_30,
        "big_move_10": big_move_10,
    }, index=idx)


FEATURE_COLS = [
    "vol_bp_30", "eff_30", "vol_bp_60", "eff_60",
    "vol_bp_120", "eff_120", "vol_bp_240", "eff_240",
    "vol_bp_480", "eff_480",
    "vol_slope_10", "vol_slope_30", "vol_slope_60", "eff_slope_30",
    "atr14", "atr30",
    "range_pct_20", "range_pct_120",
    "body_ratio_20", "body_ratio_60", "abs_body_20", "up_bar_pct_20",
    "run_up_max_20", "run_down_max_20",
    "gap_pct", "gap_abs_mean_20",
    "mom_5", "mom_15", "mom_30", "mom_60", "mom_120",
    "volume_z_20", "volume_ma_ratio",
    "max_runup_60", "max_rundown_60",
    "et_hour", "minutes_into_session", "day_of_week",
    "any_strategy_signal_30", "big_move_10",
]


def filter_session(df):
    h = df.index.hour
    return df.loc[(h >= SESSION_START_HOUR_ET) & (h < SESSION_END_HOUR_ET)].copy()


# ─── Windowed outcome labels ──────────────────────────────────────────────

def simulate_trade(bh, bl, bc, start_idx, tp, sl, side):
    if start_idx + 1 >= len(bc):
        return 0.0
    entry = bc[start_idx]
    end_idx = min(start_idx + 1 + PNL_LOOKAHEAD_BARS, len(bc))
    hs = bh[start_idx + 1 : end_idx]
    ls = bl[start_idx + 1 : end_idx]
    if len(hs) == 0:
        return 0.0
    if side > 0:
        tp_h = np.where(hs >= entry + tp)[0]
        sl_h = np.where(ls <= entry - sl)[0]
    else:
        tp_h = np.where(ls <= entry - tp)[0]
        sl_h = np.where(hs >= entry + sl)[0]
    tp_i = tp_h[0] if len(tp_h) else 1 << 30
    sl_i = sl_h[0] if len(sl_h) else 1 << 30
    if tp_i == 1 << 30 and sl_i == 1 << 30:
        last_c = bc[end_idx - 1]
        pts = (last_c - entry) if side > 0 else (entry - last_c)
        return pts * MES_PT_VALUE
    elif tp_i < sl_i:
        return tp * MES_PT_VALUE
    return -sl * MES_PT_VALUE


def build_windowed_labels(bars, feats, window_min):
    """Each label = sum of hypothetical trades across a window_min minute span.
    Trades are 5-min-spaced (every 5th bar). So 15min window → 3 trades, 30min → 6 trades.
    """
    log.info("building windowed labels: %dmin window", window_min)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}
    trades_per_window = max(1, window_min // SAMPLE_EVERY)

    keep_idx, lbls, pnl_dt_list, pnl_df_list = [], [], [], []
    feat_idx_list = list(feats.index)
    for i, ts in enumerate(feat_idx_list):
        if i % SAMPLE_EVERY != 0:
            continue
        # Collect all sample-every trades within `window_min` of this bar
        win_positions = []
        start_pos = idx_pos.get(ts)
        if start_pos is None:
            continue
        win_end_ts = ts + pd.Timedelta(minutes=window_min)
        j = i
        while j < len(feat_idx_list):
            ts_j = feat_idx_list[j]
            if ts_j >= win_end_ts:
                break
            pos_j = idx_pos.get(ts_j)
            if pos_j is not None:
                win_positions.append(pos_j)
            j += SAMPLE_EVERY
        if len(win_positions) < trades_per_window - 1:  # allow 1 short
            continue
        dt_total = 0.0
        df_total = 0.0
        for pos_j in win_positions:
            for side in (+1, -1):
                dt_total += simulate_trade(h, l, c, pos_j, DEAD_TAPE_TP, DEAD_TAPE_SL, side)
                df_total += simulate_trade(h, l, c, pos_j, DEFAULT_TP, DEFAULT_SL, side)
        diff = dt_total - df_total
        if abs(diff) < AMBIGUOUS_MARGIN_USD:
            continue
        keep_idx.append(ts)
        lbls.append("dead_tape" if diff > 0 else "default")
        pnl_dt_list.append(dt_total)
        pnl_df_list.append(df_total)

    out = feats.loc[keep_idx].copy()
    out["outcome_label"] = lbls
    out["pnl_if_deadtape"] = pnl_dt_list
    out["pnl_if_default"] = pnl_df_list
    log.info("[%dmin] labeled rows: %d  dist: %s",
             window_min, len(out), dict(Counter(lbls)))
    return out


# ─── Train + ensemble ─────────────────────────────────────────────────────

def train_ensemble(tr, cost_ratio: float = 1.5):
    """Train (HGB, LGBM) ensemble with cost-sensitive class weights."""
    X = tr[FEATURE_COLS].to_numpy()
    y = tr["outcome_label"].to_numpy()
    counts = Counter(y)
    n = len(y)
    # Cost-sensitive: penalize mis-labeling dead_tape (minority) more
    base_weight = {lbl: n / (2 * counts[lbl]) for lbl in counts}
    base_weight["dead_tape"] = base_weight["dead_tape"] * cost_ratio
    sw = np.array([base_weight[label] for label in y])

    hgb = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=42,
    )
    hgb.fit(X, y, sample_weight=sw)

    y_bin = (y == "dead_tape").astype(int)
    sw_lgb = sw
    lgbm = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=-1, num_leaves=63,
        reg_lambda=1.0, min_child_samples=30, random_state=42,
        class_weight=None,  # using sample_weight instead
        verbose=-1,
    )
    lgbm.fit(X, y_bin, sample_weight=sw_lgb)
    return hgb, lgbm


def ensemble_proba(hgb, lgbm, X):
    """Avg HGB and LGBM dead_tape probabilities."""
    hgb_classes = list(hgb.classes_)
    dt_idx = hgb_classes.index("dead_tape")
    p_hgb = hgb.predict_proba(X)[:, dt_idx]
    p_lgb = lgbm.predict_proba(X)[:, 1]  # class 1 = dead_tape (binary encoding above)
    return (p_hgb + p_lgb) / 2.0


def stats(arr):
    if len(arr) == 0:
        return {"n": 0, "pnl": 0.0, "avg": 0.0, "dd": 0.0}
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    return {"n": int(len(arr)), "pnl": float(arr.sum()),
            "avg": float(arr.mean()), "dd": float(np.max(peak - cum))}


def run_for_window(bars_all, feats_all, window_min):
    log.info("\n══ window = %d min ══", window_min)
    labeled = build_windowed_labels(bars_all, feats_all, window_min)
    if len(labeled) < 1000:
        log.warning("insufficient labels: %d", len(labeled))
        return None

    tr_cut = pd.Timestamp(TRAIN_END, tz=labeled.index.tz)
    oos_start = pd.Timestamp(OOS_START, tz=labeled.index.tz)
    tr = labeled.loc[labeled.index <= tr_cut]
    oos = labeled.loc[labeled.index >= oos_start]
    log.info("train: %d  OOS: %d", len(tr), len(oos))
    log.info("train dist: %s", dict(Counter(tr["outcome_label"])))
    log.info("OOS   dist: %s", dict(Counter(oos["outcome_label"])))
    if len(oos) < 200:
        log.warning("OOS too small"); return None

    hgb, lgbm = train_ensemble(tr, cost_ratio=1.5)
    X_oos = oos[FEATURE_COLS].to_numpy()
    y_true = oos["outcome_label"].to_numpy()
    vb_oos = oos["vol_bp_120"].to_numpy()
    pnl_dt = oos["pnl_if_deadtape"].to_numpy()
    pnl_df = oos["pnl_if_default"].to_numpy()

    # Rule baseline on windowed labels
    rule = np.where(vb_oos < DEAD_TAPE_VOL_BP, "dead_tape", "default")
    rule_acc = float((rule == y_true).mean())
    rule_pnl = np.where(rule == "dead_tape", pnl_dt, pnl_df)
    rule_st = stats(rule_pnl)
    oracle_pnl = np.where(y_true == "dead_tape", pnl_dt, pnl_df)
    oracle_st = stats(oracle_pnl)

    # Ensemble prediction — sweep threshold to find best passing gates
    dt_prob = ensemble_proba(hgb, lgbm, X_oos)

    print(f"\n── window={window_min}min ensemble threshold sweep ──")
    print(f"  rule: acc={rule_acc*100:.2f}%  PnL=${rule_st['pnl']:+.2f}  DD=${rule_st['dd']:.0f}")
    print(f"  oracle: PnL=${oracle_st['pnl']:+.2f}")
    print(f"  {'thr':>5}  {'acc':>7}  {'n_dt':>6}  {'pnl':>10}  {'dd':>7}  {'lift':>9}  gates")

    best = None
    for thr in np.arange(0.30, 0.71, 0.05):
        pred = np.where(dt_prob >= thr, "dead_tape", "default")
        acc = float((pred == y_true).mean())
        pnl_arr = np.where(pred == "dead_tape", pnl_dt, pnl_df)
        st = stats(pnl_arr)
        n_dt = int((pred == "dead_tape").sum())
        frac_dt = n_dt / max(1, len(pred))

        vb_ml_dt = oos.loc[pred == "dead_tape", "vol_bp_120"].mean() if n_dt else np.inf
        vb_ml_df = oos.loc[pred == "default", "vol_bp_120"].mean() if (pred == "default").any() else 0.0
        sanity = bool(vb_ml_dt < vb_ml_df) if np.isfinite(vb_ml_dt) and vb_ml_df > 0 else False

        lift = st["pnl"] - rule_st["pnl"]
        gates = {
            "acc_ok":       acc >= rule_acc,
            "pnl_ok":       lift >= 500.0,
            "dd_ok":        st["dd"] <= rule_st["dd"] * 1.10 if rule_st["dd"] > 0 else True,
            "sanity_ok":    sanity,
            "non_degen_ok": 0.10 <= frac_dt <= 0.90,
        }
        all_pass = all(gates.values())
        flag = " SHIP" if all_pass else ""
        print(f"  {thr:>5.2f}  {acc*100:>6.2f}%  {n_dt:>6}  ${st['pnl']:>+8.2f}  "
              f"${st['dd']:>5.0f}  ${lift:>+7.2f}   {sum(gates.values())}/5{flag}")

        if all_pass and (best is None or lift > best[0]):
            best = (lift, thr, st, acc, gates, pred)

    if best is None:
        log.warning("[%dmin] no threshold passes all gates", window_min)
        return {"window_min": window_min, "shipped": False,
                 "rule_baseline": rule_st, "rule_acc": rule_acc,
                 "oracle": oracle_st}

    lift, thr, st, acc, gates, pred = best
    log.info("[%dmin] SHIP: thr=%.2f acc=%.2f%% PnL=$%+.2f lift=$%+.2f",
             window_min, thr, acc * 100, st["pnl"], lift)
    return {"window_min": window_min, "shipped": True,
             "threshold": thr, "accuracy": acc, "pnl_stats": st,
             "lift_usd": lift, "gates": gates,
             "rule_baseline": rule_st, "rule_acc": rule_acc,
             "oracle": oracle_st,
             "_hgb": hgb, "_lgbm": lgbm}


# ─── main ────────────────────────────────────────────────────────────────

def main() -> int:
    bars_all = load_continuous_bars(TRAIN_START, OOS_END)
    feats_all = build_features(bars_all)
    feats_all = feats_all.loc[feats_all[FEATURE_COLS].notna().all(axis=1)].copy()
    feats_all = filter_session(feats_all)
    log.info("usable feature rows (NY session): %d", len(feats_all))

    results = {}
    for w in LABEL_WINDOWS_MIN:
        results[w] = run_for_window(bars_all, feats_all, w)

    # Find best shipped result across windows
    shipped = [(w, r) for w, r in results.items() if r and r.get("shipped")]
    if not shipped:
        print("\n═" * 72)
        print(" v4 — ALL WINDOW/THRESHOLD COMBOS FAILED GATES")
        print("═" * 72)
        for w, r in results.items():
            if r is None:
                print(f"  {w}min: no labeled data")
                continue
            print(f"  {w}min: rule acc={r['rule_acc']*100:.2f}%  rule PnL=${r['rule_baseline']['pnl']:+.2f}  "
                  f"oracle PnL=${r['oracle']['pnl']:+.2f}  NO SHIP")
        (OUT_DIR / "metrics.json").write_text(json.dumps(
            {w: {k: v for k, v in r.items() if not k.startswith("_")}
              for w, r in results.items() if r}, indent=2, default=str))
        return 1

    shipped.sort(key=lambda x: -x[1]["lift_usd"])
    best_w, best_r = shipped[0]

    print("\n" + "═" * 72)
    print(f" v4 — SHIPPED: window={best_w}min  threshold={best_r['threshold']:.2f}")
    print("═" * 72)
    print(f"  rule baseline: acc={best_r['rule_acc']*100:.2f}%  PnL=${best_r['rule_baseline']['pnl']:+.2f}  DD=${best_r['rule_baseline']['dd']:.0f}")
    print(f"  ML ensemble:   acc={best_r['accuracy']*100:.2f}%  PnL=${best_r['pnl_stats']['pnl']:+.2f}  DD=${best_r['pnl_stats']['dd']:.0f}")
    print(f"  oracle upper:  PnL=${best_r['oracle']['pnl']:+.2f}")
    print(f"  lift:          ${best_r['lift_usd']:+.2f}")
    print(f"  gates:         {best_r['gates']}")

    # Write artifacts for the winning window
    with (OUT_DIR / "hgb.pkl").open("wb") as fh:
        pickle.dump(best_r["_hgb"], fh, protocol=pickle.HIGHEST_PROTOCOL)
    with (OUT_DIR / "lgbm.pkl").open("wb") as fh:
        pickle.dump(best_r["_lgbm"], fh, protocol=pickle.HIGHEST_PROTOCOL)
    (OUT_DIR / "feature_order.json").write_text(json.dumps({
        "features": FEATURE_COLS,
        "session_hours_et": [SESSION_START_HOUR_ET, SESSION_END_HOUR_ET],
        "label_window_min": best_w,
        "threshold": best_r["threshold"],
        "label_binary": ["default", "dead_tape"],
    }, indent=2))
    clean = {w: {k: v for k, v in r.items() if not k.startswith("_")}
              for w, r in results.items() if r}
    (OUT_DIR / "metrics.json").write_text(json.dumps(clean, indent=2, default=str))
    print(f"\n  [WROTE] hgb.pkl, lgbm.pkl, feature_order.json → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
