#!/usr/bin/env python3
"""ML Regime Classifier — train + OOS validate + ship-or-kill.

Replaces the rule-based `regime_classifier._classify()` (2-feature threshold
rules on vol_bp + eff) with a supervised HGB classifier that uses a broader
feature set. Rule labels are the training target.

Approach rationale (pragmatic):
  Option 1 (supervised on rule labels) chosen because the ship gates
  quantify "≥80% rule accuracy + PnL ≥ rule baseline", which is exactly
  what this approach tests. The disagreement set between ML and rule
  is the interesting surface; on those bars we check whether ML's call
  produces equal-or-better forward trade PnL under each regime's bracket
  policy. If yes → ship (ML learned regime structure rules missed). If
  no → kill (rules are already right, no smoothing earned).

Ship gates (ALL must pass):
  1. Rule-label accuracy on OOS ≥ 80%
  2. OOS PnL under ML-regime ≥ OOS PnL under rule-regime
  3. OOS MaxDD under ML ≤ 110% of rule MaxDD
  4. Sanity: <5% of ML-called dead_tape bars have vol_bp > 3.0

Pipeline:
  build_dataset()  → features DataFrame + rule labels
  train()          → HGB classifier on pre-April
  validate()       → OOS accuracy, confusion, PnL sim, sanity
  ship_or_kill()   → write model + report, or report-only

Run:
    python3 scripts/ml_regime_classifier.py
Outputs on ship:
    artifacts/regime_ml_v1/model.pkl
    artifacts/regime_ml_v1/feature_order.json
    artifacts/regime_ml_v1/metrics.json
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
OUT_DIR = ROOT / "artifacts" / "regime_ml_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Windows (same convention as AetherFlow retrain)
TRAIN_START = "2024-07-01"
TRAIN_END   = "2026-01-26"
OOS_START   = "2026-01-27"
OOS_END     = "2026-04-20"

# Match regime_classifier.py exactly
WINDOW_BARS = 120
DEAD_TAPE_VOL_BP = 1.5
EFF_LOW = 0.05
EFF_HIGH = 0.12

# Bracket policy for PnL sim (matches apply_dead_tape_brackets + default AF/DE3)
DEAD_TAPE_TP = 3.0
DEAD_TAPE_SL = 5.0
DEFAULT_TP = 6.0
DEFAULT_SL = 4.0
MES_PT_VALUE = 5.0

PNL_LOOKAHEAD_BARS = 60
PNL_SAMPLE_EVERY = 15   # simulate an entry every 15 bars in OOS

FEATURE_COLS = [
    "vol_bp_120", "eff_120",
    "vol_bp_60", "eff_60",
    "atr14",
    "range_pct_20", "body_ratio_20", "up_bar_pct_20",
    "mom_5", "mom_15", "mom_30",
    "volume_z_20",
    "et_hour", "et_minute_bucket",
    "range_pct_120",
]
LABELS = ["dead_tape", "whipsaw", "calm_trend", "neutral"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_regime")


# ─── data load ────────────────────────────────────────────────────────────

def load_continuous_bars(start: str, end: str) -> pd.DataFrame:
    """Load outright bars, pick dominant symbol per day (by volume) to get a
    clean continuous series for feature calc."""
    log.info("loading es_master_outrights for %s → %s", start, end)
    df = pd.read_parquet(ROOT / "es_master_outrights.parquet")
    lo = pd.Timestamp(start, tz=df.index.tz)
    hi = pd.Timestamp(end, tz=df.index.tz)
    df = df.loc[(df.index >= lo) & (df.index <= hi)].copy()
    # Pick dominant symbol per date (max daily volume) — preserve DatetimeIndex
    date_arr = df.index.date
    df_with_date = df.assign(_date=date_arr)
    dom = df_with_date.groupby(["_date", "symbol"])["volume"].sum().reset_index()
    dom = dom.sort_values(["_date", "volume"], ascending=[True, False])
    dom = dom.drop_duplicates("_date", keep="first")
    dominant_map = dict(zip(dom["_date"], dom["symbol"]))
    mask = pd.Series(date_arr, index=df.index).map(dominant_map) == df["symbol"]
    df = df.loc[mask.values].sort_index()
    log.info("after dominant-symbol filter: %d bars", len(df))
    return df


# ─── feature + label builder ──────────────────────────────────────────────

def rolling_vol_eff(closes: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Match regime_classifier._compute_metrics semantics:
       rets = (p_i - p_{i-1}) / p_{i-1}, computed inside each rolling window.
       vol_bp = std(rets) * 10000
       eff = |sum(rets)| / sum(|rets|)
    Returns (vol_bp, eff) aligned to closes index.
    """
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


def build_features_and_labels(bars: pd.DataFrame) -> pd.DataFrame:
    log.info("building features on %d bars", len(bars))
    c = bars["close"].to_numpy(dtype=float)
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    v = bars["volume"].to_numpy(dtype=float)

    vol120, eff120 = rolling_vol_eff(c, WINDOW_BARS)
    vol60,  eff60  = rolling_vol_eff(c, 60)

    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]),
                                        np.abs(l - np.r_[c[0], c[:-1]])))
    atr14 = pd.Series(tr).rolling(14).mean().to_numpy()

    rng_pct = (h - l) / np.where(c != 0, c, 1.0)
    rng_pct_20 = pd.Series(rng_pct).rolling(20).mean().to_numpy()
    rng_pct_120 = pd.Series(rng_pct).rolling(120).mean().to_numpy()

    hl = np.maximum(h - l, 1e-9)
    body_ratio = np.abs(c - o) / hl
    body_ratio_20 = pd.Series(body_ratio).rolling(20).mean().to_numpy()

    up_bar = (c >= o).astype(float)
    up_bar_pct_20 = pd.Series(up_bar).rolling(20).mean().to_numpy()

    mom_5  = (c - np.r_[c[:5], c[:-5]]) / np.where(np.r_[c[:5], c[:-5]] != 0, np.r_[c[:5], c[:-5]], 1.0)
    mom_15 = (c - np.r_[c[:15], c[:-15]]) / np.where(np.r_[c[:15], c[:-15]] != 0, np.r_[c[:15], c[:-15]], 1.0)
    mom_30 = (c - np.r_[c[:30], c[:-30]]) / np.where(np.r_[c[:30], c[:-30]] != 0, np.r_[c[:30], c[:-30]], 1.0)

    vol_mean_200 = pd.Series(v).rolling(200).mean().to_numpy()
    vol_std_200 = pd.Series(v).rolling(200).std().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_z_20 = (v - vol_mean_200) / np.where(vol_std_200 > 0, vol_std_200, 1.0)

    idx = bars.index
    et_hour = idx.hour.to_numpy()
    et_min_bucket = (idx.minute // 10).to_numpy()

    feats = pd.DataFrame({
        "vol_bp_120": vol120, "eff_120": eff120,
        "vol_bp_60": vol60,   "eff_60": eff60,
        "atr14": atr14,
        "range_pct_20": rng_pct_20 * 10_000.0,     # scale to bp
        "body_ratio_20": body_ratio_20,
        "up_bar_pct_20": up_bar_pct_20,
        "mom_5": mom_5 * 10_000.0, "mom_15": mom_15 * 10_000.0, "mom_30": mom_30 * 10_000.0,
        "volume_z_20": volume_z_20,
        "et_hour": et_hour, "et_minute_bucket": et_min_bucket,
        "range_pct_120": rng_pct_120 * 10_000.0,
    }, index=idx)

    # Rule label via vol_bp_120 + eff_120 (exact match to _classify)
    rule = np.full(len(feats), "warmup", dtype=object)
    valid = ~(np.isnan(vol120) | np.isnan(eff120))
    vb = vol120; ef = eff120
    rule[valid & (vb < DEAD_TAPE_VOL_BP)] = "dead_tape"
    mask_whipsaw = valid & (vb > 3.5) & (ef < EFF_LOW) & (vb >= DEAD_TAPE_VOL_BP)
    rule[mask_whipsaw] = "whipsaw"
    mask_calm = valid & (ef > EFF_HIGH) & (vb >= DEAD_TAPE_VOL_BP) & ~mask_whipsaw
    rule[mask_calm] = "calm_trend"
    # Neutral is everyone else that passed warmup
    rule_arr = np.where(rule == "warmup", rule,
                        np.where((rule == "dead_tape") | (rule == "whipsaw") | (rule == "calm_trend"),
                                  rule, "neutral"))
    # Enforce: if feature row invalid, label = warmup
    rule_arr[~valid] = "warmup"

    feats["rule_label"] = rule_arr
    feats["close"] = c
    feats["high"] = h
    feats["low"] = l
    # Drop warmup + NaNs
    keep = (feats["rule_label"] != "warmup") & feats[FEATURE_COLS].notna().all(axis=1)
    feats = feats.loc[keep].copy()
    log.info("usable feature rows: %d  rule label dist: %s",
             len(feats), dict(Counter(feats["rule_label"])))
    return feats


# ─── training ─────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, train_end: str) -> HistGradientBoostingClassifier:
    tr_cutoff = pd.Timestamp(train_end, tz=df.index.tz)
    tr = df.loc[df.index <= tr_cutoff]
    X = tr[FEATURE_COLS].to_numpy()
    y = tr["rule_label"].to_numpy()
    log.info("training on %d rows  label dist: %s", len(X), dict(Counter(y)))
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=50, random_state=42,
    )
    clf.fit(X, y)
    return clf


# ─── OOS simulation ───────────────────────────────────────────────────────

def pnl_for_bar_set(bars_df: pd.DataFrame, labels: np.ndarray, feats: pd.DataFrame) -> dict:
    """For every Nth bar, simulate BOTH long and short hypothetical entries
    using the bracket policy implied by the regime label. Walk forward up to
    PNL_LOOKAHEAD_BARS on the continuous bar series `bars_df`.

    Per-trade PnL in $ at 1 MES. Returns {pnl, wr, dd, n}.
    """
    # We need the full continuous bar series for forward walk. `bars_df` is
    # sorted; `feats` is a subset.
    high = bars_df["high"].to_numpy()
    low = bars_df["low"].to_numpy()
    close = bars_df["close"].to_numpy()
    idx_pos = {ts: i for i, ts in enumerate(bars_df.index)}

    trades = []
    for i, (ts, label) in enumerate(zip(feats.index, labels)):
        if i % PNL_SAMPLE_EVERY != 0:
            continue
        start_pos = idx_pos.get(ts)
        if start_pos is None or start_pos + 1 >= len(close):
            continue
        entry_pos = start_pos + 1
        entry_price = close[start_pos]
        if label == "dead_tape":
            tp, sl = DEAD_TAPE_TP, DEAD_TAPE_SL
        else:
            tp, sl = DEFAULT_TP, DEFAULT_SL
        # Long + short both
        for side in (+1, -1):
            end_pos = min(entry_pos + PNL_LOOKAHEAD_BARS, len(close))
            pnl = 0.0
            if side > 0:
                tp_px = entry_price + tp
                sl_px = entry_price - sl
                hit_tp = np.where(high[entry_pos:end_pos] >= tp_px)[0]
                hit_sl = np.where(low[entry_pos:end_pos] <= sl_px)[0]
            else:
                tp_px = entry_price - tp
                sl_px = entry_price + sl
                hit_tp = np.where(low[entry_pos:end_pos] <= tp_px)[0]
                hit_sl = np.where(high[entry_pos:end_pos] >= sl_px)[0]
            tp_i = hit_tp[0] if len(hit_tp) else 1 << 30
            sl_i = hit_sl[0] if len(hit_sl) else 1 << 30
            if tp_i == 1 << 30 and sl_i == 1 << 30:
                last_px = close[end_pos - 1]
                pts = (last_px - entry_price) if side > 0 else (entry_price - last_px)
                pnl = pts * MES_PT_VALUE
            elif tp_i < sl_i:
                pnl = tp * MES_PT_VALUE
            else:
                pnl = -sl * MES_PT_VALUE
            trades.append(pnl)
    a = np.array(trades)
    if len(a) == 0:
        return {"pnl": 0.0, "wr": 0.0, "dd": 0.0, "n": 0}
    cum = np.cumsum(a)
    peak = np.maximum.accumulate(cum)
    return {
        "pnl":  float(a.sum()),
        "wr":   100.0 * float((a > 0).sum()) / len(a),
        "dd":   float(np.max(peak - cum)),
        "n":    int(len(a)),
    }


# ─── validate + ship ──────────────────────────────────────────────────────

def validate(clf, df: pd.DataFrame, bars: pd.DataFrame) -> dict:
    oos_start = pd.Timestamp(OOS_START, tz=df.index.tz)
    oos_end = pd.Timestamp(OOS_END, tz=df.index.tz) + pd.Timedelta(days=1)
    oos = df.loc[(df.index >= oos_start) & (df.index <= oos_end)]
    log.info("OOS rows: %d", len(oos))
    if len(oos) == 0:
        return {"error": "no OOS rows"}
    X = oos[FEATURE_COLS].to_numpy()
    y_rule = oos["rule_label"].to_numpy()
    y_ml = clf.predict(X)

    # Metric 1: rule-label accuracy
    acc = accuracy_score(y_rule, y_ml)
    labels_sorted = sorted(set(y_rule) | set(y_ml))
    cm = confusion_matrix(y_rule, y_ml, labels=labels_sorted)

    # Metric 2/3: OOS PnL under rule vs ML
    bars_oos = bars.loc[(bars.index >= oos_start) & (bars.index <= oos_end)]
    rule_sim = pnl_for_bar_set(bars_oos, y_rule, oos)
    ml_sim   = pnl_for_bar_set(bars_oos, y_ml,   oos)

    # Metric 4: sanity — ML-called dead_tape bars with vol_bp > 3.0
    ml_dt_mask = (y_ml == "dead_tape")
    n_ml_dt = int(ml_dt_mask.sum())
    high_vol_dt = int(np.sum(ml_dt_mask & (oos["vol_bp_120"].to_numpy() > 3.0)))
    catastrophic_pct = (100.0 * high_vol_dt / n_ml_dt) if n_ml_dt > 0 else 0.0

    # Disagreement breakdown
    disagree_mask = y_rule != y_ml
    disagree_n = int(disagree_mask.sum())
    disagree_by_pair = Counter(zip(y_rule[disagree_mask], y_ml[disagree_mask]))

    return {
        "oos_n": int(len(oos)),
        "rule_label_dist": dict(Counter(y_rule)),
        "ml_label_dist":   dict(Counter(y_ml)),
        "accuracy": float(acc),
        "labels_sorted": labels_sorted,
        "confusion": cm.tolist(),
        "rule_sim": rule_sim,
        "ml_sim": ml_sim,
        "sanity_catastrophic_pct": catastrophic_pct,
        "sanity_n_ml_dead_tape": n_ml_dt,
        "sanity_n_high_vol_dead_tape": high_vol_dt,
        "disagree_n": disagree_n,
        "disagree_by_pair": {f"{a}->{b}": c for (a, b), c in disagree_by_pair.most_common(20)},
    }


def gate(rep: dict) -> dict:
    gates = {
        "accuracy_ok":     rep["accuracy"] >= 0.80,
        "pnl_ok":          rep["ml_sim"]["pnl"] >= rep["rule_sim"]["pnl"],
        "dd_ok":           rep["ml_sim"]["dd"] <= rep["rule_sim"]["dd"] * 1.10
                            if rep["rule_sim"]["dd"] > 0 else True,
        "sanity_ok":       rep["sanity_catastrophic_pct"] < 5.0,
    }
    gates["SHIP"] = all(gates.values())
    return gates


def main() -> int:
    bars_all = load_continuous_bars(TRAIN_START, OOS_END)
    feats_all = build_features_and_labels(bars_all)

    clf = train(feats_all, TRAIN_END)
    rep = validate(clf, feats_all, bars_all)
    gates = gate(rep)

    # Print report
    print("\n" + "═" * 72)
    print(" ML REGIME CLASSIFIER — OOS VALIDATION")
    print("═" * 72)
    print(f"  OOS rows: {rep['oos_n']}")
    print(f"  rule dist: {rep['rule_label_dist']}")
    print(f"  ML dist  : {rep['ml_label_dist']}")
    print(f"\n  Accuracy vs rule: {rep['accuracy'] * 100:.2f}%  (gate ≥ 80%)")
    print(f"  Disagreements   : {rep['disagree_n']} ({100*rep['disagree_n']/rep['oos_n']:.2f}%)")
    print(f"  Top disagree pairs (rule→ML):")
    for k, v in list(rep["disagree_by_pair"].items())[:10]:
        print(f"    {k:<30}  {v}")
    print(f"\n  Confusion matrix ({rep['labels_sorted']}):")
    for row, lab in zip(rep["confusion"], rep["labels_sorted"]):
        print(f"    {lab:<14} {row}")
    print(f"\n  PnL simulation (hypothetical entries every {PNL_SAMPLE_EVERY} bars, L+S):")
    print(f"    RULE-driven regime: n={rep['rule_sim']['n']:<5}  PnL=${rep['rule_sim']['pnl']:+.2f}  "
          f"WR={rep['rule_sim']['wr']:.2f}%  DD=${rep['rule_sim']['dd']:.0f}")
    print(f"    ML-driven   regime: n={rep['ml_sim']['n']:<5}  PnL=${rep['ml_sim']['pnl']:+.2f}  "
          f"WR={rep['ml_sim']['wr']:.2f}%  DD=${rep['ml_sim']['dd']:.0f}")
    pnl_delta = rep["ml_sim"]["pnl"] - rep["rule_sim"]["pnl"]
    print(f"    Δ PnL: ${pnl_delta:+.2f}")
    print(f"\n  Sanity (ML dead_tape at vol_bp>3.0): {rep['sanity_n_high_vol_dead_tape']} / "
          f"{rep['sanity_n_ml_dead_tape']} = {rep['sanity_catastrophic_pct']:.2f}% (gate <5%)")
    print()
    print("  SHIP GATES:")
    for k, v in gates.items():
        print(f"    {'✓' if v else '✗'}  {k}")
    print()

    metrics = {"report": rep, "gates": gates,
               "feature_columns": FEATURE_COLS, "labels": list(clf.classes_),
               "train_window": [TRAIN_START, TRAIN_END],
               "oos_window":   [OOS_START, OOS_END]}
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log.info("wrote metrics → %s", OUT_DIR / "metrics.json")

    if gates["SHIP"]:
        with (OUT_DIR / "model.pkl").open("wb") as fh:
            pickle.dump(clf, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (OUT_DIR / "feature_order.json").write_text(
            json.dumps({"features": FEATURE_COLS, "labels": list(clf.classes_)}, indent=2)
        )
        log.info("[SHIP] model written to %s", OUT_DIR / "model.pkl")
        print("  [SHIP] writing model + feature_order + metrics")
        return 0
    else:
        log.warning("[KILL] gates failed — not writing model")
        print("  [KILL] gates failed — not shipping")
        return 1


if __name__ == "__main__":
    sys.exit(main())
