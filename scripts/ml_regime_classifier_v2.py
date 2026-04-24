#!/usr/bin/env python3
"""ML Regime Classifier v2 — OUTCOME-LABELED training + OOS ship-or-kill.

v1 (supervised on rule labels) was KILLED: 99.95% rule accuracy means the
ML just replicated the rule. Any disagreement is noise on the rule's own
feature surface, and costs PnL.

v2 pivots to OUTCOME LABELING:
  For each bar, simulate BOTH bracket policies forward 60 bars:
     A. dead_tape policy: TP=3 / SL=5  (suits low-vol mean-reverting)
     B. default  policy: TP=6 / SL=4  (suits trending)
  Take average of LONG and SHORT hypothetical entries under each policy.
  Label = "dead_tape" if A's PnL > B's PnL (by margin), else "default".
  Ambiguous bars (|A-B| < $5) are dropped to prevent noise-fitting.

  Train binary classifier on these labels with a broader feature set than
  the rule has (vol_bp at multiple windows, eff at multiple windows, ATR,
  volume z, time-of-day, momentum, bar shape). The ML output maps to:
     predict "dead_tape" → apply dead_tape bracket rewrite
     predict "default"   → leave brackets alone (passthrough)

  Compared against the RULE BASELINE (which uses the same dead_tape → 3/5
  and default → 6/4 mapping, keyed off the rule classifier's dead_tape
  call). Rule baseline = rule_classifier says dead_tape iff vol_bp < 1.5.

Ship gates (ALL must pass):
  1. OOS binary accuracy vs outcome-derived label ≥ 55% (coin-flip is 50)
  2. OOS PnL under ML label-driven brackets ≥ rule label-driven brackets
  3. MaxDD under ML ≤ 110% of rule MaxDD
  4. Sanity: ML-called dead_tape bars should have vol_bp strictly lower on
     average than ML-called default bars (+directional: dead_tape is a
     low-vol regime; if ML inverts that, fail)

Session filter: the live classifier only ticks during RTH + electronic,
but es_master_outrights is 24h. Overnight / Asia bars have vol << 1.5
trivially, which was the dead_tape-dominance in v1. v2 filters to
09:00-16:30 ET (NY session) where the regime call actually matters.
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

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "regime_ml_v2"
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
SAMPLE_EVERY = 5           # label every 5th bar to keep run tractable
AMBIGUOUS_MARGIN_USD = 5.0  # drop labels where |A-B| < this

# NY session filter (ET). Regime call matters during active trading only.
SESSION_START_HOUR_ET = 9
SESSION_END_HOUR_ET = 16

FEATURE_COLS = [
    "vol_bp_120", "eff_120",
    "vol_bp_60",  "eff_60",
    "vol_bp_30",  "eff_30",
    "atr14",
    "range_pct_20", "body_ratio_20", "up_bar_pct_20",
    "mom_5", "mom_15", "mom_30",
    "volume_z_20",
    "et_hour", "et_minute_bucket",
    "range_pct_120",
    "max_runup_60", "max_rundown_60",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_regime_v2")


def load_continuous_bars(start: str, end: str) -> pd.DataFrame:
    log.info("loading es_master_outrights %s → %s", start, end)
    df = pd.read_parquet(ROOT / "es_master_outrights.parquet")
    lo = pd.Timestamp(start, tz=df.index.tz)
    hi = pd.Timestamp(end, tz=df.index.tz)
    df = df.loc[(df.index >= lo) & (df.index <= hi)].copy()
    date_arr = df.index.date
    dom = df.assign(_date=date_arr).groupby(["_date", "symbol"])["volume"].sum().reset_index()
    dom = dom.sort_values(["_date", "volume"], ascending=[True, False]).drop_duplicates("_date", keep="first")
    dominant_map = dict(zip(dom["_date"], dom["symbol"]))
    mask = pd.Series(date_arr, index=df.index).map(dominant_map) == df["symbol"]
    df = df.loc[mask.values].sort_index()
    log.info("after dominant-symbol filter: %d bars", len(df))
    return df


def rolling_vol_eff(closes: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
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
    log.info("features on %d bars", len(bars))
    c = bars["close"].to_numpy(float)
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    v = bars["volume"].to_numpy(float)

    vol120, eff120 = rolling_vol_eff(c, WINDOW_BARS)
    vol60,  eff60  = rolling_vol_eff(c, 60)
    vol30,  eff30  = rolling_vol_eff(c, 30)

    prev_c = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr14 = pd.Series(tr).rolling(14).mean().to_numpy()

    rng_pct = (h - l) / np.where(c != 0, c, 1.0)
    rng_pct_20 = pd.Series(rng_pct).rolling(20).mean().to_numpy()
    rng_pct_120 = pd.Series(rng_pct).rolling(120).mean().to_numpy()

    hl = np.maximum(h - l, 1e-9)
    body_ratio_20 = pd.Series(np.abs(c - o) / hl).rolling(20).mean().to_numpy()
    up_bar_pct_20 = pd.Series((c >= o).astype(float)).rolling(20).mean().to_numpy()

    def lag(arr, k):
        return np.r_[arr[:k], arr[:-k]]

    mom_5  = (c - lag(c, 5))  / np.where(lag(c, 5)  != 0, lag(c, 5),  1.0)
    mom_15 = (c - lag(c, 15)) / np.where(lag(c, 15) != 0, lag(c, 15), 1.0)
    mom_30 = (c - lag(c, 30)) / np.where(lag(c, 30) != 0, lag(c, 30), 1.0)

    vol_mean_200 = pd.Series(v).rolling(200).mean().to_numpy()
    vol_std_200  = pd.Series(v).rolling(200).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_z = (v - vol_mean_200) / np.where(vol_std_200 > 0, vol_std_200, 1.0)

    # max run-up / run-down in last 60 bars
    max_run_up_60 = pd.Series(c).rolling(60).apply(
        lambda x: (x.max() - x.iloc[0]) / max(x.iloc[0], 1e-9), raw=False
    ).to_numpy() * 10_000.0
    max_run_down_60 = pd.Series(c).rolling(60).apply(
        lambda x: (x.iloc[0] - x.min()) / max(x.iloc[0], 1e-9), raw=False
    ).to_numpy() * 10_000.0

    idx = bars.index
    et_hour = idx.hour.to_numpy()
    et_min_bucket = (idx.minute // 10).to_numpy()

    return pd.DataFrame({
        "vol_bp_120": vol120, "eff_120": eff120,
        "vol_bp_60":  vol60,  "eff_60":  eff60,
        "vol_bp_30":  vol30,  "eff_30":  eff30,
        "atr14": atr14,
        "range_pct_20": rng_pct_20 * 10_000.0,
        "body_ratio_20": body_ratio_20,
        "up_bar_pct_20": up_bar_pct_20,
        "mom_5":  mom_5  * 10_000.0,
        "mom_15": mom_15 * 10_000.0,
        "mom_30": mom_30 * 10_000.0,
        "volume_z_20": volume_z,
        "et_hour": et_hour, "et_minute_bucket": et_min_bucket,
        "range_pct_120": rng_pct_120 * 10_000.0,
        "max_runup_60": max_run_up_60,
        "max_rundown_60": max_run_down_60,
    }, index=idx)


def simulate_trade(bars_h: np.ndarray, bars_l: np.ndarray, bars_c: np.ndarray,
                    start_idx: int, tp: float, sl: float, side: int) -> float:
    """Return $ PnL at 1 MES for hypothetical entry at bars_c[start_idx]."""
    if start_idx + 1 >= len(bars_c):
        return 0.0
    entry = bars_c[start_idx]
    end_idx = min(start_idx + 1 + PNL_LOOKAHEAD_BARS, len(bars_c))
    hs = bars_h[start_idx + 1 : end_idx]
    ls = bars_l[start_idx + 1 : end_idx]
    if len(hs) == 0:
        return 0.0
    if side > 0:
        tp_hits = np.where(hs >= entry + tp)[0]
        sl_hits = np.where(ls <= entry - sl)[0]
    else:
        tp_hits = np.where(ls <= entry - tp)[0]
        sl_hits = np.where(hs >= entry + sl)[0]
    tp_i = tp_hits[0] if len(tp_hits) else 1 << 30
    sl_i = sl_hits[0] if len(sl_hits) else 1 << 30
    if tp_i == 1 << 30 and sl_i == 1 << 30:
        last_c = bars_c[end_idx - 1]
        pts = (last_c - entry) if side > 0 else (entry - last_c)
        return pts * MES_PT_VALUE
    elif tp_i < sl_i:
        return tp * MES_PT_VALUE
    else:
        return -sl * MES_PT_VALUE


def build_outcome_labels(bars: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """For each Nth bar, simulate both policies (long+short), label = whichever
    wins by margin. Returns feats subsetted to labeled rows."""
    log.info("building outcome labels (every %d bars)", SAMPLE_EVERY)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars.index)}

    labels = []
    pnl_deadtape = []
    pnl_default = []
    keep_idx = []
    for i, ts in enumerate(feats.index):
        if i % SAMPLE_EVERY != 0:
            continue
        pos = idx_pos.get(ts)
        if pos is None:
            continue
        dt_long = simulate_trade(h, l, c, pos, DEAD_TAPE_TP, DEAD_TAPE_SL, +1)
        dt_short = simulate_trade(h, l, c, pos, DEAD_TAPE_TP, DEAD_TAPE_SL, -1)
        df_long = simulate_trade(h, l, c, pos, DEFAULT_TP, DEFAULT_SL, +1)
        df_short = simulate_trade(h, l, c, pos, DEFAULT_TP, DEFAULT_SL, -1)
        dt_total = dt_long + dt_short
        df_total = df_long + df_short
        # Label: which policy extracted more edge? Drop ambiguous.
        diff = dt_total - df_total
        if abs(diff) < AMBIGUOUS_MARGIN_USD:
            continue
        label = "dead_tape" if diff > 0 else "default"
        labels.append(label)
        pnl_deadtape.append(dt_total)
        pnl_default.append(df_total)
        keep_idx.append(ts)

    out = feats.loc[keep_idx].copy()
    out["outcome_label"] = labels
    out["pnl_if_deadtape"] = pnl_deadtape
    out["pnl_if_default"] = pnl_default
    log.info("labeled rows: %d  dist: %s", len(out), dict(Counter(labels)))
    return out


def filter_session(df: pd.DataFrame) -> pd.DataFrame:
    h = df.index.hour
    return df.loc[(h >= SESSION_START_HOUR_ET) & (h < SESSION_END_HOUR_ET)].copy()


def validate(clf, labeled_oos: pd.DataFrame) -> dict:
    X = labeled_oos[FEATURE_COLS].to_numpy()
    y_true = labeled_oos["outcome_label"].to_numpy()
    y_ml = clf.predict(X)

    # Rule baseline: rule says dead_tape iff vol_bp_120 < 1.5 (matches live classifier)
    y_rule = np.where(labeled_oos["vol_bp_120"].to_numpy() < DEAD_TAPE_VOL_BP, "dead_tape", "default")

    acc_ml = accuracy_score(y_true, y_ml)
    acc_rule = accuracy_score(y_true, y_rule)

    # PnL under each labeler's choice (ml / rule / oracle)
    pnl_dt = labeled_oos["pnl_if_deadtape"].to_numpy()
    pnl_df = labeled_oos["pnl_if_default"].to_numpy()
    pnl_ml = np.where(y_ml == "dead_tape", pnl_dt, pnl_df)
    pnl_rule = np.where(y_rule == "dead_tape", pnl_dt, pnl_df)
    pnl_oracle = np.where(y_true == "dead_tape", pnl_dt, pnl_df)

    def stats(arr):
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        dd = float(np.max(peak - cum)) if len(arr) else 0.0
        return {"n": int(len(arr)), "pnl": float(arr.sum()),
                "avg": float(arr.mean()) if len(arr) else 0.0, "dd": dd}

    # Sanity: ML dead_tape bars should have lower avg vol_bp than ML default bars
    vb_ml_dt = labeled_oos.loc[y_ml == "dead_tape", "vol_bp_120"].mean()
    vb_ml_df = labeled_oos.loc[y_ml == "default",   "vol_bp_120"].mean()
    sanity_directional = bool(vb_ml_dt < vb_ml_df) if (y_ml == "dead_tape").any() and (y_ml == "default").any() else False

    cm = confusion_matrix(y_true, y_ml, labels=["dead_tape", "default"])

    return {
        "oos_n": int(len(labeled_oos)),
        "dist_true": dict(Counter(y_true)),
        "dist_ml":   dict(Counter(y_ml)),
        "dist_rule": dict(Counter(y_rule)),
        "accuracy_ml":   float(acc_ml),
        "accuracy_rule": float(acc_rule),
        "confusion_ml":  cm.tolist(),
        "pnl_ml":     stats(pnl_ml),
        "pnl_rule":   stats(pnl_rule),
        "pnl_oracle": stats(pnl_oracle),
        "sanity_vb_ml_dead_tape": float(vb_ml_dt) if not pd.isna(vb_ml_dt) else None,
        "sanity_vb_ml_default":   float(vb_ml_df) if not pd.isna(vb_ml_df) else None,
        "sanity_directional_ok":  sanity_directional,
    }


def gate(rep: dict) -> dict:
    # Tighter gates to catch degenerate (majority-class) classifiers.
    # Majority-class prior on OOS is ~68% default; a trivial "always predict
    # default" classifier hits 68% accuracy and technically ties the rule on
    # PnL. We reject it with two additional gates:
    #   - dead_tape RECALL ≥ 30% (classifier actually learned the minority class)
    #   - PnL LIFT ≥ $200 absolute AND ≥ 10% of (oracle - rule) gap
    cm = rep["confusion_ml"]  # [[TT, TF], [FT, FF]] for [dead_tape, default]
    true_dt = cm[0][0] + cm[0][1]
    recall_dt = (cm[0][0] / true_dt) if true_dt > 0 else 0.0
    rep["recall_dead_tape"] = recall_dt

    pnl_lift = rep["pnl_ml"]["pnl"] - rep["pnl_rule"]["pnl"]
    rule_to_oracle = rep["pnl_oracle"]["pnl"] - rep["pnl_rule"]["pnl"]
    min_capture = max(200.0, 0.10 * rule_to_oracle) if rule_to_oracle > 0 else 200.0
    rep["pnl_lift_usd"] = pnl_lift
    rep["pnl_lift_required_usd"] = min_capture

    g = {
        "accuracy_ok":   rep["accuracy_ml"] >= 0.55,
        "recall_ok":     recall_dt >= 0.30,
        "pnl_ok":        pnl_lift >= min_capture,
        "dd_ok":         rep["pnl_ml"]["dd"] <= rep["pnl_rule"]["dd"] * 1.10
                          if rep["pnl_rule"]["dd"] > 0 else True,
        "sanity_ok":     bool(rep["sanity_directional_ok"]),
    }
    g["SHIP"] = all(g.values())
    return g


def main() -> int:
    bars_all = load_continuous_bars(TRAIN_START, OOS_END)
    feats_all = build_features(bars_all)
    feats_all = feats_all.loc[feats_all[FEATURE_COLS].notna().all(axis=1)].copy()
    log.info("non-nan feature rows: %d", len(feats_all))
    feats_all = filter_session(feats_all)
    log.info("NY-session rows (09:00–16:00 ET): %d", len(feats_all))

    labeled = build_outcome_labels(bars_all, feats_all)
    tr_cut = pd.Timestamp(TRAIN_END, tz=labeled.index.tz)
    oos_start = pd.Timestamp(OOS_START, tz=labeled.index.tz)
    tr = labeled.loc[labeled.index <= tr_cut]
    oos = labeled.loc[labeled.index >= oos_start]
    log.info("train rows: %d  OOS rows: %d", len(tr), len(oos))
    log.info("train label dist: %s", dict(Counter(tr["outcome_label"])))
    log.info("OOS   label dist: %s", dict(Counter(oos["outcome_label"])))

    if len(tr) < 500 or len(oos) < 100:
        log.warning("insufficient samples; aborting")
        return 2

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=42,
    )
    # Class-balance sample_weight so the minority class (dead_tape) gets equal
    # pull in the objective. Without this the model collapses to predicting
    # the majority class and trivially hits ~68% accuracy from the prior.
    y_tr = tr["outcome_label"].to_numpy()
    counts = Counter(y_tr)
    n = len(y_tr)
    sw = np.array([n / (2.0 * counts[y]) for y in y_tr])
    clf.fit(tr[FEATURE_COLS].to_numpy(), y_tr, sample_weight=sw)

    rep = validate(clf, oos)
    gates = gate(rep)

    print("\n" + "═" * 72)
    print(" ML REGIME CLASSIFIER v2 (OUTCOME-LABELED) — OOS VALIDATION")
    print("═" * 72)
    print(f"  OOS rows: {rep['oos_n']}")
    print(f"  true dist:  {rep['dist_true']}")
    print(f"  ML dist:    {rep['dist_ml']}")
    print(f"  rule dist:  {rep['dist_rule']}")
    print(f"\n  Accuracy:  ML = {rep['accuracy_ml']*100:.2f}%    rule = {rep['accuracy_rule']*100:.2f}%    (gate ML ≥55%)")
    print(f"  Confusion ML (true rows / pred cols, labels=[dead_tape, default]):")
    for lab, row in zip(["dead_tape", "default"], rep["confusion_ml"]):
        print(f"    {lab:<12} {row}")
    print(f"\n  PnL (OOS, 1 MES, hypothetical L+S entries every {SAMPLE_EVERY} bars):")
    print(f"    rule-driven:   n={rep['pnl_rule']['n']:<6}  PnL=${rep['pnl_rule']['pnl']:+.2f}  "
          f"avg=${rep['pnl_rule']['avg']:+.3f}  DD=${rep['pnl_rule']['dd']:.0f}")
    print(f"    ML-driven:     n={rep['pnl_ml']['n']:<6}  PnL=${rep['pnl_ml']['pnl']:+.2f}  "
          f"avg=${rep['pnl_ml']['avg']:+.3f}  DD=${rep['pnl_ml']['dd']:.0f}")
    print(f"    oracle (upper):n={rep['pnl_oracle']['n']:<6}  PnL=${rep['pnl_oracle']['pnl']:+.2f}  "
          f"avg=${rep['pnl_oracle']['avg']:+.3f}  DD=${rep['pnl_oracle']['dd']:.0f}")
    print(f"    Δ(ml - rule) = ${rep['pnl_ml']['pnl'] - rep['pnl_rule']['pnl']:+.2f}")
    print(f"\n  Sanity: avg vol_bp on ML dead_tape = {rep['sanity_vb_ml_dead_tape']}  "
          f"vs default = {rep['sanity_vb_ml_default']}  directional_ok={rep['sanity_directional_ok']}")
    print(f"\n  dead_tape recall: {rep.get('recall_dead_tape', 0)*100:.2f}%  (gate ≥ 30%)")
    print(f"  PnL lift:         ${rep.get('pnl_lift_usd', 0):+.2f}  "
          f"(gate ≥ ${rep.get('pnl_lift_required_usd', 200):.2f})")
    print(f"\n  SHIP GATES:")
    for k, v in gates.items():
        print(f"    {'✓' if v else '✗'}  {k}")

    metrics = {"report": rep, "gates": gates, "feature_columns": FEATURE_COLS,
               "train_window": [TRAIN_START, TRAIN_END], "oos_window": [OOS_START, OOS_END],
               "ambiguous_margin_usd": AMBIGUOUS_MARGIN_USD, "sample_every": SAMPLE_EVERY,
               "pnl_lookahead_bars": PNL_LOOKAHEAD_BARS,
               "session_filter": [SESSION_START_HOUR_ET, SESSION_END_HOUR_ET]}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    if gates["SHIP"]:
        with (OUT_DIR / "model.pkl").open("wb") as fh:
            pickle.dump(clf, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (OUT_DIR / "feature_order.json").write_text(
            json.dumps({"features": FEATURE_COLS, "labels": list(clf.classes_),
                         "session_hours_et": [SESSION_START_HOUR_ET, SESSION_END_HOUR_ET]}, indent=2)
        )
        print("\n  [SHIP] model + feature_order written to artifacts/regime_ml_v2/")
        return 0
    else:
        print("\n  [KILL] gates failed — model not written")
        return 1


if __name__ == "__main__":
    sys.exit(main())
