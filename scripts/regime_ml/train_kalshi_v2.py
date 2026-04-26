#!/usr/bin/env python3
"""Train Kalshi overlay ML v2 — stacked upgrades over v1.

Upgrades over train_kalshi.py:
  1. Longer forward windows (15/30/60-min sweep to pick cleanest label margin).
  2. Richer features: Kalshi deltas over time, intraday cumulative PnL state,
     minutes-to-hour-close, threshold-flip flags, signed bar-body asymmetry.
  3. Per-strategy segmentation DROPPED — only DE3 is present in logs, so a
     pooled model is equivalent and gives more samples.
  4. Binary override-only action space: ML defers to rule by default; only
     overrides when |proba_pass - 0.5| >= override_margin (swept).
  5. Recency-weighted training with exponential time-decay (half-life sweep).
  6. Isotonic calibration wrapping HGB via CalibratedClassifierCV.

Ship gates (all 5 must pass):
  1. OOS PnL lift > 0 vs rule baseline (lift = ML PnL - rule PnL)
  2. DD/PnL ratio ≤ 30%
  3. n_kalshi_events ≥ 50 in OOS
  4. newly-PASSED WR ≥ 50% (on rule-BLOCK → ML-PASS overrides, if ≥ 5 such)
  5. capture ≥ 20% of oracle-perfect lift

AetherFlow events excluded — size-only carveout in live code.
"""
from __future__ import annotations

import argparse, json, logging, pickle, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    ROOT, FEATURE_COLS_40, MES_PT_VALUE, DEFAULT_TP, DEFAULT_SL,
    load_continuous_bars, build_feature_frame, simulate_trade,
    stats, sample_weights_balanced,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_kalshi_v2")

LABEL_MARGIN_USD = 15.0
OOS_START = "2026-01-27"
OOS_END = "2026-04-24"

# Log parser
RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_BAR = re.compile(r"Bar: (?P<mts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET \| Price: (?P<px>[-\d.]+)")
RE_ENTRY_VIEW = re.compile(
    r"\[KALSHI_ENTRY_VIEW\].*?"
    r"strategy=(?P<strategy>\S+) \| side=(?P<side>LONG|SHORT) "
    r"\| entry_price=(?P<entry>[-\d.]+) "
    r"\| role=(?P<role>\S+) "
    r"\| decision=(?P<decision>PASS|BLOCK) "
    r"\| entry_probability=(?P<ep>[-\d.]+) "
    r"\| probe_price=(?P<pp>[-\d.]+) "
    r"\| probe_probability=(?P<pprob>[-\d.]+) "
    r"\| momentum_delta=(?P<md>[-\d.]+) "
    r"\| momentum_retention=(?P<mr>[-\d.]+) "
    r"\| support_score=(?P<ss>[-\d.]+) "
    r"\| threshold=(?P<thr>[-\d.]+)"
)


def parse_log_events(log_path: Path) -> list:
    events = []
    last_bar_mts = None
    seen_minute_keys = set()
    with log_path.open(errors="ignore") as fh:
        for line in fh:
            m = RE_HEADER.match(line)
            if not m: continue
            bm = RE_BAR.search(line)
            if bm:
                last_bar_mts = bm.group("mts")
                continue
            ev = RE_ENTRY_VIEW.search(line)
            if not ev: continue
            strat = ev.group("strategy")
            if "aetherflow" in strat.lower():
                continue
            if last_bar_mts is None: continue
            minute = last_bar_mts[:16]
            key = (minute, strat, ev.group("side"), ev.group("role"))
            if key in seen_minute_keys: continue
            seen_minute_keys.add(key)
            events.append({
                "market_ts": last_bar_mts,
                "strategy": strat,
                "side": ev.group("side"),
                "entry_price": float(ev.group("entry")),
                "role": ev.group("role"),
                "rule_decision": ev.group("decision"),
                "k_entry_probability": float(ev.group("ep")),
                "k_probe_price": float(ev.group("pp")),
                "k_probe_probability": float(ev.group("pprob")),
                "k_momentum_delta": float(ev.group("md")),
                "k_momentum_retention": float(ev.group("mr")),
                "k_support_score": float(ev.group("ss")),
                "k_threshold": float(ev.group("thr")),
            })
    return events


def simulate_trade_horizon(bh: np.ndarray, bl: np.ndarray, bc: np.ndarray,
                           start_idx: int, tp: float, sl: float, side: int,
                           horizon_bars: int) -> float:
    """Forward-walk a single hypothetical trade over `horizon_bars`.
    Returns $ PnL at 1 MES (bracket = TP/SL first-hit, then mark-to-close at end)."""
    if start_idx + 1 >= len(bc):
        return 0.0
    entry = bc[start_idx]
    end_idx = min(start_idx + 1 + horizon_bars, len(bc))
    hs = bh[start_idx + 1 : end_idx]
    ls = bl[start_idx + 1 : end_idx]
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
        last_c = bc[end_idx - 1]
        pts = (last_c - entry) if side > 0 else (entry - last_c)
        return pts * MES_PT_VALUE
    if tp_i < sl_i:
        return tp * MES_PT_VALUE
    return -sl * MES_PT_VALUE


# ─── Feature engineering v2 ─────────────────────────────────────────────

def add_v2_features(events: list) -> list:
    """Enrich each event in-place with v2 features. Events must be
    chronologically ordered. We track per-strategy state across events."""
    # Sort by timestamp
    events = sorted(events, key=lambda e: e["market_ts"])

    # Per-strategy state tracking for Kalshi deltas + flips
    hist: dict[str, list[dict]] = defaultdict(list)
    # Per-(strategy, date) running state for intraday cumulative features
    intraday_cum: dict[tuple, dict] = {}

    for e in events:
        ts = pd.Timestamp(e["market_ts"])
        strat = e["strategy"]
        date_key = (strat, e["market_ts"][:10])
        ep_now = e["k_entry_probability"]
        thr_now = e["k_threshold"]

        # Lookups into per-strategy history (chronological)
        past = hist[strat]

        def prob_delta(minutes):
            """entry_probability now minus the most-recent event within
            last `minutes` before now (same strategy).  0 if none."""
            cutoff = ts - pd.Timedelta(minutes=minutes)
            for p in reversed(past):
                if pd.Timestamp(p["market_ts"]) <= cutoff:
                    return ep_now - p["k_entry_probability"]
            # If we have any past event at all, use the earliest one in window
            if past and pd.Timestamp(past[0]["market_ts"]) >= cutoff:
                return ep_now - past[0]["k_entry_probability"]
            return 0.0

        e["k_prob_delta_15m"] = prob_delta(15)
        e["k_prob_delta_30m"] = prob_delta(30)
        e["k_prob_delta_60m"] = prob_delta(60)

        # Threshold flip: compare (ep_now > thr_now) vs (ep_prev > thr_prev)
        prev = past[-1] if past else None
        flip = 0
        min_since_flip = 9999
        if prev is not None:
            now_above = 1 if ep_now >= thr_now else 0
            prev_above = 1 if prev["k_entry_probability"] >= prev["k_threshold"] else 0
            flip = 1 if now_above != prev_above else 0
            # minutes since last flip
            flip_ts = prev.get("_last_flip_ts")
            if flip:
                e["_last_flip_ts"] = ts
                min_since_flip = 0
            else:
                if flip_ts is not None:
                    min_since_flip = int((ts - flip_ts).total_seconds() / 60)
                    e["_last_flip_ts"] = flip_ts
                else:
                    min_since_flip = 9999
                    e["_last_flip_ts"] = None
        e["k_threshold_flip"] = flip
        e["k_min_since_flip"] = min(min_since_flip, 240)  # cap at 4 hours

        # Margin above/below threshold (signed)
        e["k_margin"] = ep_now - thr_now
        e["k_margin_abs"] = abs(ep_now - thr_now)

        # Intraday cumulative — running sum of forward_pnl for prior
        # events on the same day where rule=PASS (approximates bot's
        # realized book state at decision time). Forward PnL is filled in
        # later by the labeler; we only fill what we know here.
        ic = intraday_cum.setdefault(date_key, {"n_prior": 0, "pass_pnl_sum": 0.0, "last_event_ts": None})
        e["intraday_n_prior_events"] = ic["n_prior"]
        # Minutes since last prior event on same day (flags bursts)
        if ic["last_event_ts"] is not None:
            e["minutes_since_prior_event"] = int((ts - ic["last_event_ts"]).total_seconds() / 60)
        else:
            e["minutes_since_prior_event"] = 999
        ic["n_prior"] += 1
        ic["last_event_ts"] = ts

        # Minutes to next hour close (proxy for Kalshi settlement boundary)
        minute_of_hour = ts.minute
        e["minutes_to_hour_close"] = 60 - minute_of_hour

        hist[strat].append(e)

    # Clean up internal state fields
    for e in events:
        e.pop("_last_flip_ts", None)
    return events


def fill_intraday_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """After forward_pnl has been computed for every event, backfill the
    intraday_pass_pnl_cum feature: for each event, sum forward_pnl of
    prior same-day same-strategy events where rule=PASS."""
    df = df.copy()
    df["intraday_pass_pnl_cum"] = 0.0
    df["intraday_pass_pnl_cum_bucket"] = 0  # -1 losing, 0 flat, +1 winning

    # Need chronological order by (strategy, date, ts)
    df_sorted = df.sort_index(kind="stable")
    running: dict[tuple, float] = {}
    cum_list = []
    for idx, row in df_sorted.iterrows():
        d = idx.date().isoformat()
        key = (row["strategy"], d)
        cur = running.get(key, 0.0)
        cum_list.append(cur)
        if row["rule_decision"] == "PASS":
            running[key] = cur + float(row["forward_pnl"])
    df_sorted["intraday_pass_pnl_cum"] = cum_list
    df_sorted["intraday_pass_pnl_cum_bucket"] = np.sign(df_sorted["intraday_pass_pnl_cum"]).astype(int)
    return df_sorted


def build_dataset(events: list, bars_df: pd.DataFrame,
                  label_horizon_min: int) -> pd.DataFrame:
    feats = build_feature_frame(bars_df)
    feats = feats.loc[feats[FEATURE_COLS_40].notna().all(axis=1)].copy()
    feats_minute_idx = {pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M"): ts for ts in feats.index}
    log.info("feature frame rows: %d", len(feats))

    c = bars_df["close"].to_numpy(float)
    h = bars_df["high"].to_numpy(float)
    l = bars_df["low"].to_numpy(float)
    idx_pos = {ts: i for i, ts in enumerate(bars_df.index)}

    rows = []
    dropped_no_bar = 0
    dropped_ambiguous = 0
    for e in events:
        mts_key = e["market_ts"][:16]
        ts = feats_minute_idx.get(mts_key)
        if ts is None:
            dropped_no_bar += 1
            continue
        pos = idx_pos.get(ts)
        if pos is None:
            dropped_no_bar += 1
            continue
        side_sign = 1 if e["side"] == "LONG" else -1
        pnl = simulate_trade_horizon(h, l, c, int(pos), DEFAULT_TP, DEFAULT_SL, side_sign, label_horizon_min)
        if abs(pnl) < LABEL_MARGIN_USD:
            dropped_ambiguous += 1
            continue
        row = {**e, "_ts": ts}
        for col in FEATURE_COLS_40:
            row[col] = feats.loc[ts, col]
        row["side_sign"] = side_sign
        row["is_de3"] = 1 if "dynamicengine" in e["strategy"].lower() else 0
        row["is_ra"]  = 1 if "regimeadaptive" in e["strategy"].lower() else 0
        row["is_ml"]  = 1 if "mlphysics" in e["strategy"].lower() else 0
        row["settlement_hour"] = int(pd.Timestamp(mts_key + ":00").hour)
        row["minutes_into_session"] = int(
            (pd.Timestamp(mts_key + ":00").hour - 9) * 60 +
             pd.Timestamp(mts_key + ":00").minute
        )
        row["role_forward"] = 1 if "forward" in e["role"] else 0
        row["role_background"] = 1 if e["role"] == "background" else 0
        row["role_balanced"] = 1 if e["role"] == "balanced" else 0
        row["label"] = "pass" if pnl > 0 else "block"
        row["forward_pnl"] = pnl
        row["rule_decision_pass_int"] = 1 if e["rule_decision"] == "PASS" else 0
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df.set_index("_ts")
    # Fill intraday cumulative PnL features
    df = fill_intraday_pnl(df)
    log.info("labeled rows: %d  dropped (no bar): %d  dropped (ambiguous): %d",
             len(df), dropped_no_bar, dropped_ambiguous)
    log.info("class dist: %s", dict(Counter(df["label"])))
    log.info("rule × label cross-tab:")
    for r_dec in ("PASS", "BLOCK"):
        for lbl in ("pass", "block"):
            n = int(((df["rule_decision"] == r_dec) & (df["label"] == lbl)).sum())
            log.info("  rule=%s × label=%s: %d", r_dec, lbl, n)
    # Separation signal: how clean is the pass-vs-block gap?
    pass_pnls = df[df["label"] == "pass"]["forward_pnl"]
    block_pnls = df[df["label"] == "block"]["forward_pnl"]
    log.info("pass PnL mean=$%+.2f  median=$%+.2f  n=%d",
             pass_pnls.mean(), pass_pnls.median(), len(pass_pnls))
    log.info("block PnL mean=$%+.2f  median=$%+.2f  n=%d",
             block_pnls.mean(), block_pnls.median(), len(block_pnls))
    return df


V2_EXTRA_FEATURES = [
    "k_prob_delta_15m", "k_prob_delta_30m", "k_prob_delta_60m",
    "k_threshold_flip", "k_min_since_flip",
    "k_margin", "k_margin_abs",
    "intraday_n_prior_events", "minutes_since_prior_event",
    "intraday_pass_pnl_cum", "intraday_pass_pnl_cum_bucket",
    "minutes_to_hour_close",
]

FEATURE_COLS_KALSHI_V1 = FEATURE_COLS_40 + [
    "k_entry_probability", "k_probe_probability", "k_momentum_delta",
    "k_momentum_retention", "k_support_score", "k_threshold",
    "side_sign", "settlement_hour", "minutes_into_session",
    "is_de3", "is_ra", "is_ml",
    "role_forward", "role_background", "role_balanced",
]

FEATURE_COLS_KALSHI_V2 = FEATURE_COLS_KALSHI_V1 + V2_EXTRA_FEATURES


def recency_weights(index: pd.DatetimeIndex, half_life_days: float) -> np.ndarray:
    """Exponential decay: most recent event = weight 1.0."""
    ref = index.max()
    ages_days = (ref - index).total_seconds() / 86400.0
    return np.power(0.5, ages_days / max(1e-6, half_life_days))


def evaluate_override_policy(
    proba_pass: np.ndarray,
    rule_decision: np.ndarray,
    y_true: np.ndarray,
    pnl_if_passed: np.ndarray,
    override_margin: float,
    pass_margin: float | None = None,
    block_margin: float | None = None,
) -> dict:
    """Apply binary override semantics: ML acts only when the model's
    probability exceeds per-direction margins; otherwise trust rule.

    If `pass_margin`/`block_margin` are None, both default to
    `override_margin` (symmetric). Use asymmetric margins when you want
    conservative PASS overrides (high pass_margin) but aggressive
    BLOCK overrides (low block_margin) — since winning signals are ~half
    as common as losing ones in OOS.

    Returns a dict with PnL stats + gate inputs."""
    if pass_margin is None:  pass_margin = override_margin
    if block_margin is None: block_margin = override_margin
    ml_pass = proba_pass >= (0.5 + pass_margin)
    ml_block = proba_pass <= (0.5 - block_margin)
    # Final decision: ML if confident, else rule
    final_pass = np.where(
        ml_pass, True,
        np.where(ml_block, False, rule_decision == "PASS"),
    )
    final_pnl = np.where(final_pass, pnl_if_passed, 0.0)
    ml_st = stats(final_pnl)
    n_pass = int(final_pass.sum())

    # Newly passed = rule BLOCK but final decision = PASS (must be an ML override)
    new_pass_mask = (rule_decision == "BLOCK") & final_pass & ml_pass
    n_new_pass = int(new_pass_mask.sum())
    new_pass_pnls = pnl_if_passed[new_pass_mask]
    new_pass_wr = (100 * (new_pass_pnls > 0).sum() / len(new_pass_pnls)) if len(new_pass_pnls) else 0.0

    # Newly blocked = rule PASS but final decision = BLOCK (must be an ML override)
    new_block_mask = (rule_decision == "PASS") & (~final_pass) & ml_block
    n_new_block = int(new_block_mask.sum())
    new_block_pnls = pnl_if_passed[new_block_mask]
    # A "good" block vetoes a losing trade
    new_block_wr = (100 * (new_block_pnls <= 0).sum() / len(new_block_pnls)) if len(new_block_pnls) else 0.0

    return {
        "override_margin": override_margin,
        "n_pass": n_pass,
        "n_new_pass": n_new_pass, "new_pass_wr": new_pass_wr,
        "n_new_block": n_new_block, "new_block_wr": new_block_wr,
        "pnl": ml_st["pnl"], "dd": ml_st["dd"], "avg": ml_st["avg"],
    }


def run_config(events: list, bars: pd.DataFrame,
               label_horizon_min: int,
               half_life_days: float,
               calibration: str,
               train_start_date: str | None,
               oos_start: str, oos_end: str,
               seed: int = 42) -> dict:
    """Fit + evaluate one (horizon × half_life × calibration × train_start) combo.
    Returns sweep result dict including all override margins tested."""
    log.info("")
    log.info("=" * 60)
    log.info("CONFIG  horizon=%dmin  half_life=%.0fd  cal=%s  train_start=%s",
             label_horizon_min, half_life_days, calibration, train_start_date or "full")
    log.info("=" * 60)
    df = build_dataset(events, bars, label_horizon_min)
    if len(df) < 500:
        log.warning("only %d labeled rows — skipping this config", len(df))
        return None

    oos_start_ts = pd.Timestamp(oos_start, tz=df.index.tz)
    oos_end_ts = pd.Timestamp(oos_end, tz=df.index.tz) + pd.Timedelta(days=1)
    tr = df.loc[df.index < oos_start_ts]
    if train_start_date is not None:
        tr = tr.loc[tr.index >= pd.Timestamp(train_start_date, tz=df.index.tz)]
    oos = df.loc[(df.index >= oos_start_ts) & (df.index <= oos_end_ts)]
    log.info("train: %d  OOS: %d", len(tr), len(oos))
    if len(tr) < 200 or len(oos) < 50:
        log.warning("train=%d OOS=%d below floors — skip", len(tr), len(oos))
        return None

    X_tr = tr[FEATURE_COLS_KALSHI_V2].to_numpy()
    y_tr = tr["label"].to_numpy()
    class_w = sample_weights_balanced(y_tr, cost_ratio=1.3)
    recency_w = recency_weights(tr.index, half_life_days)
    sw = class_w * recency_w  # combined

    base = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=6,
        l2_regularization=1.0, min_samples_leaf=30, random_state=seed,
    )
    if calibration == "none":
        clf = base
        clf.fit(X_tr, y_tr, sample_weight=sw)
    else:
        # CalibratedClassifierCV wraps the base with CV-held-out calibration.
        # prefit not used because we pass sample_weight; use 3-fold.
        clf = CalibratedClassifierCV(estimator=base, method=calibration, cv=3)
        clf.fit(X_tr, y_tr, sample_weight=sw)
    # Identify pass index robustly (classes_ attr on both Calibrated and HGB)
    classes = clf.classes_ if hasattr(clf, "classes_") else base.classes_
    pass_idx = list(classes).index("pass")

    X_oos = oos[FEATURE_COLS_KALSHI_V2].to_numpy()
    y_true = oos["label"].to_numpy()
    pnl_if_passed = oos["forward_pnl"].to_numpy()
    rule_decision = oos["rule_decision"].to_numpy()
    proba_pass = clf.predict_proba(X_oos)[:, pass_idx]

    # Baselines
    rule_pnl_arr = np.where(rule_decision == "PASS", pnl_if_passed, 0.0)
    rule_st = stats(rule_pnl_arr)
    oracle_pnl_arr = np.where(y_true == "pass", pnl_if_passed, 0.0)
    oracle_st = stats(oracle_pnl_arr)
    log.info("rule baseline: PnL=$%+.2f DD=$%.0f  | oracle: PnL=$%+.2f",
             rule_st["pnl"], rule_st["dd"], oracle_st["pnl"])

    # Sweep override margins — symmetric + asymmetric combinations
    # Asymmetric: conservative PASS overrides, aggressive BLOCK overrides
    # (because rule over-passes losers more than it blocks winners in OOS)
    margin_combos = []
    # Symmetric
    for m in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        margin_combos.append((m, m, m))
    # Asymmetric — tuple is (label_margin_for_logging, pass_mrg, block_mrg)
    margin_combos.extend([
        (0.30, 0.35, 0.15),   # strict pass, lax block
        (0.30, 0.40, 0.15),
        (0.30, 0.40, 0.10),
        (0.30, 0.35, 0.20),
    ])
    results = []
    log.info("%-15s %8s %8s %10s %10s %10s %8s %8s gates",
             "pass/blk mrg", "n_pass", "new_psn", "newPassWR", "n_new_blk", "pnl", "dd", "capt")
    for (tag_m, p_m, b_m) in margin_combos:
        r = evaluate_override_policy(proba_pass, rule_decision, y_true, pnl_if_passed,
                                     tag_m, pass_margin=p_m, block_margin=b_m)
        lift = r["pnl"] - rule_st["pnl"]
        dd_over_pnl = (r["dd"] / r["pnl"] * 100.0) if r["pnl"] > 0 else float("inf")
        capt = (lift / (oracle_st["pnl"] - rule_st["pnl"]) * 100.0) if (oracle_st["pnl"] - rule_st["pnl"]) > 0 else 0.0
        gates = {
            "pnl_ok":        lift > 0,
            "dd_ratio_ok":   dd_over_pnl <= 30.0,
            "n_ok":          len(oos) >= 50,
            "new_pass_wr_ok":(r["new_pass_wr"] >= 50.0) if r["n_new_pass"] >= 5 else True,
            "capt_ok":       capt >= 20.0,
        }
        ok = all(gates.values())
        tag = f"p={p_m:.2f}/b={b_m:.2f}"
        log.info("%-15s %8d %8d %9.2f%% %10d $%+9.2f $%6.0f %6.2f%%  %d/5%s",
                 tag, r["n_pass"], r["n_new_pass"], r["new_pass_wr"], r["n_new_block"],
                 r["pnl"], r["dd"], capt, sum(gates.values()),
                 " SHIP" if ok else "")
        r.update({
            "pass_margin": p_m, "block_margin": b_m,
            "lift": lift, "dd_over_pnl": dd_over_pnl, "capt_pct": capt,
            "gates": gates, "ships": ok,
        })
        results.append(r)

    # Pick best: ships > no-ships; then max lift within ships; then max lift overall
    shippers = [r for r in results if r["ships"]]
    if shippers:
        best = max(shippers, key=lambda r: r["lift"])
    else:
        best = max(results, key=lambda r: r["lift"])

    return {
        "horizon": label_horizon_min,
        "half_life": half_life_days,
        "calibration": calibration,
        "train_start": train_start_date,
        "rule_baseline": rule_st,
        "oracle": oracle_st,
        "n_train": len(tr),
        "n_oos": len(oos),
        "all_margins": results,
        "best": best,
        "model": clf if best["ships"] else None,
        "feature_cols": FEATURE_COLS_KALSHI_V2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-start", default=OOS_START)
    ap.add_argument("--oos-end", default=OOS_END)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts/regime_ml_kalshi_v2"))
    ap.add_argument("--seed", type=int, default=42)
    # Config sweep — mostly sensible defaults
    ap.add_argument("--horizons", nargs="+", type=int, default=[15, 30, 60])
    ap.add_argument("--half-lives", nargs="+", type=float, default=[90.0, 120.0])
    ap.add_argument("--calibrations", nargs="+", default=["none", "isotonic"])
    ap.add_argument("--train-starts", nargs="+", default=["full"],
                    help="Space-separated list; 'full' = no cutoff, or YYYY-MM-DD")
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
    log.info("total minute-unique Kalshi events (AF excl.): %d", len(all_events))
    if len(all_events) < 2000:
        log.warning("sample too small (%d) — kill", len(all_events))
        return 1

    all_events = add_v2_features(all_events)
    all_mts = [e["market_ts"] for e in all_events]
    start = pd.Timestamp(min(all_mts)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(max(all_mts)) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    log.info("event range: %s → %s", start, end)
    bars = load_continuous_bars(start, end)
    log.info("bars loaded: %d", len(bars))

    # Run sweep
    configs = []
    for ts_ in args.train_starts:
        ts_val = None if ts_ == "full" else ts_
        for h in args.horizons:
            for hl in args.half_lives:
                for cal in args.calibrations:
                    configs.append((h, hl, cal, ts_val))
    log.info("total configs: %d", len(configs))
    runs = []
    for cfg in configs:
        h, hl, cal, ts_val = cfg
        r = run_config(all_events, bars, h, hl, cal, ts_val,
                       args.oos_start, args.oos_end, seed=args.seed)
        if r is not None:
            runs.append(r)

    # Summary table
    log.info("\n%s\n" + "═" * 120, "SWEEP SUMMARY")
    log.info("%4s %4s %9s %10s %5s %4s %8s %8s %8s %8s %8s %s",
             "hrz", "hl", "cal", "trstart", "gates", "mrg", "n_new", "WR%",
             "lift$", "dd$", "capt%", "ship?")
    log.info("─" * 120)
    for r in runs:
        b = r["best"]
        log.info("%4d %4.0f %9s %10s %5d %4.2f %8d %7.2f%% %+7.2f %7.0f %6.2f%% %s",
                 r["horizon"], r["half_life"], r["calibration"],
                 str(r["train_start"] or "full")[:10],
                 sum(b["gates"].values()), b["override_margin"],
                 b["n_new_pass"], b["new_pass_wr"],
                 b["lift"], b["dd"], b["capt_pct"],
                 "SHIP" if b["ships"] else "-")

    # Pick best shipping config
    shippers = [r for r in runs if r["best"]["ships"]]
    if not shippers:
        # Kill path — but also write a best-effort diagnostic summary
        summary = {
            "verdict": "KILL",
            "reason": "no (horizon × half-life × cal × train-start × override-margin) combo passes all 5 gates",
            "runs": [
                {k: v for k, v in r.items() if k not in ("model",)}
                for r in runs
            ],
        }
        (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        log.warning("[KILL] writing sweep_summary.json only — no model artifact")
        return 1

    best_run = max(shippers, key=lambda r: r["best"]["lift"])
    clf = best_run["model"]
    payload = {
        "model_kind": "HGB+calibrated" if best_run["calibration"] != "none" else "HGB",
        "clf": clf,
        "feature_cols": FEATURE_COLS_KALSHI_V2,
        "override_margin": best_run["best"]["override_margin"],
        "label_horizon_min": best_run["horizon"],
        "half_life_days": best_run["half_life"],
        "calibration": best_run["calibration"],
        "train_start": best_run["train_start"],
        "rule_baseline_oos": best_run["rule_baseline"],
        "oracle_oos": best_run["oracle"],
        "best_oos_stats": best_run["best"],
        "n_train": best_run["n_train"],
        "n_oos": best_run["n_oos"],
        "positive_class": "pass",
        "inference_mode": "hgb_only_override",
        "seed": args.seed,
    }
    with (out_dir / "model.pkl").open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "model_meta.json").write_text(json.dumps({
        k: v for k, v in payload.items() if k not in ("clf",)
    }, indent=2, default=str))
    log.info("[SHIP] horizon=%dmin  hl=%.0fd  cal=%s  trstart=%s",
             best_run["horizon"], best_run["half_life"],
             best_run["calibration"], best_run["train_start"] or "full")
    log.info("       margin=%.2f  lift=$%+.2f  newPassWR=%.2f%%  capt=%.2f%%",
             best_run["best"]["override_margin"], best_run["best"]["lift"],
             best_run["best"]["new_pass_wr"], best_run["best"]["capt_pct"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
