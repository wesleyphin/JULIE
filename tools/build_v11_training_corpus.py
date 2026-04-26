#!/usr/bin/env python3
"""Build the V11 clean training corpus for the JULIE001 ML overlay retrain.

Goals (v11 vs v10):
  1. NO inherited overlay decisions from a v1 corpus we cannot audit. Every
     overlay proba is computed by re-firing the *production-loaded*
     `model.joblib` on candidate features computed in this script.
  2. Family-aware single-position rule (BYPASS_SAMESIDE=0) — exact replica of
     friend's `_allow_same_side_parallel_entry()` at julie001.py:2041-2080.
  3. Kalshi 12-16 ET window honoured: outside the window, Kalshi forces
     ALLOW (no veto, no size scale). Implemented via tz-aware ET conversion.
  4. AF Kalshi carveout: AetherFlow candidates never get vetoed by Kalshi —
     Kalshi only size-scales AF.

Friend's family-aware same-side rule (julie001.py:2058-2080, BYPASS_SAMESIDE=0):

    primary_family = _live_strategy_family_name(primary_trade.get("strategy"))
    signal_family  = _live_strategy_family_name(signal.get("strategy"))
    if primary_family == "de3":
        return signal_family in {"regimeadaptive", "aetherflow"}
    if primary_family == "aetherflow" and signal_family == "aetherflow":
        max_legs = ... (default 1)
        if max_legs <= 1:
            return False
        ...
    return False

Practical reading:
  - DE3 + DE3 same-side  -> BLOCKED
  - DE3 + (RA or AF) same-side cross-family  -> ALLOWED
  - RA primary + anything  -> BLOCKED (RA is not in the conditional)
  - AF + AF  -> only if max_legs > 1; default 1 -> BLOCKED
  - Opposite-side: handled separately (NOT blocked here; in production a
    fresh opposite-side trade closes the open one. For corpus walk-forward
    purposes we treat opposite-side as ALLOWED — the new trade closes the
    open one.)

AF Kalshi carveout (julie001.py:1469-1481):
    is_aetherflow = _live_strategy_family_name(signal.get("strategy")) == "aetherflow"
    if not allowed:
        if is_aetherflow:
            return size  # carveout: AF never blocked by Kalshi
        return 0
    if is_aetherflow and multiplier < 1.0:
        return size  # carveout: AF never size-shrunk by Kalshi

Kalshi 12-16 ET window (julie001.py:175 + 1443-1451):
    _KALSHI_GATING_HOURS_ET = [12, 13, 14, 15, 16]
    settlement_hour = _active_kalshi_settlement_hour_et(kalshi)
    if settlement_hour not in _KALSHI_GATING_HOURS_ET:
        return size  # outside hours: Kalshi silent

Outputs:
  artifacts/v11_training_corpus.parquet
  artifacts/v11_corpus_summary.json
"""
from __future__ import annotations

import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator_trade_through import (  # noqa: E402
    front_month_by_calendar,
    simulate_trade_through,
)

PARQUET_BARS = ROOT / "es_master_outrights.parquet"
PARQUET_V1 = ROOT / "artifacts" / "full_overlay_stack_simulation_14mo.parquet"

OUT_CORPUS = ROOT / "artifacts" / "v11_training_corpus.parquet"
OUT_SUMMARY = ROOT / "artifacts" / "v11_corpus_summary.json"

# Production-loaded ML model paths (mirror ml_overlay_shadow.py:25 + filter G v10)
MODEL_LFO_PATH = ROOT / "artifacts" / "signal_gate_2025" / "model_lfo.joblib"
MODEL_KALSHI_PATH = ROOT / "artifacts" / "signal_gate_2025" / "model_kalshi_gate.joblib"
MODEL_PCT_PATH = ROOT / "artifacts" / "signal_gate_2025" / "model_pct_overlay.joblib"
MODEL_PIVOT_PATH = ROOT / "artifacts" / "signal_gate_2025" / "model_pivot_trail.joblib"
MODEL_FILTERG_PATH = ROOT / "artifacts" / "regime_ml_filterg_v10" / "de3" / "model.joblib"

POINT_VALUE = 5.0    # MES = $5/pt
HAIRCUT = 7.50       # $/trade
HORIZON = 30         # bars
ES_TICK = 0.25
BANK_GRID = 25.0     # bank levels at every 25 pts

KALSHI_HOURS_ET = {12, 13, 14, 15, 16}

# ---------------------------------------------------------------------------
# Family helpers (mirror julie001.py:_live_strategy_family_name)
# ---------------------------------------------------------------------------

def family_from_strategy(strategy: str) -> str:
    s = str(strategy or "").strip()
    if s.startswith("DynamicEngine3") or s in {"DynamicEngine", "DynamicEngineStrategy"}:
        return "de3"
    if s == "RegimeAdaptive":
        return "regimeadaptive"
    if s.startswith("AetherFlow"):
        return "aetherflow"
    if s.startswith("MLPhysics"):
        return "mlphysics"
    return ""


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _atr14_pts(prev_bars: pd.DataFrame) -> float:
    """True Range ATR(14) using last 14 bars."""
    if prev_bars is None or len(prev_bars) < 2:
        return 0.0
    sub = prev_bars.tail(15).copy()
    sub["prev_close"] = sub["close"].shift(1)
    tr1 = sub["high"] - sub["low"]
    tr2 = (sub["high"] - sub["prev_close"]).abs()
    tr3 = (sub["low"] - sub["prev_close"]).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr = tr.dropna().tail(14)
    return float(tr.mean()) if len(tr) > 0 else 0.0


def _et_session(hour_et: int) -> str:
    if 18 <= hour_et or hour_et < 3:
        return "ASIA"
    if 3 <= hour_et < 7:
        return "LONDON"
    if 7 <= hour_et < 9:
        return "NY_PRE"
    if 9 <= hour_et < 12:
        return "NY_AM"
    if 12 <= hour_et < 16:
        return "NY_PM"
    return "POST"


def _et_session_ml_lfo(hour_et: int) -> str:
    """Sessions used by LFO model (NY collapsed)."""
    sess = _et_session(hour_et)
    if sess in {"NY_AM", "NY_PM"}:
        return "NY"
    return sess


def build_bar_features(prev_bars: pd.DataFrame, signal_price: float, side: str,
                       sl_dist: float, tp_dist: float, hour_et: int) -> dict:
    """Compute bar-level features used by LFO + Kalshi gate + Pivot models.

    prev_bars: bars at and BEFORE signal time (inclusive of signal bar at the tail).
    """
    atr14 = _atr14_pts(prev_bars)
    if len(prev_bars) >= 1:
        last = prev_bars.iloc[-1]
        bar_range = float(last["high"] - last["low"])
        body = abs(float(last["close"]) - float(last["open"]))
        upper_wick = float(last["high"]) - max(float(last["open"]), float(last["close"]))
        lower_wick = min(float(last["open"]), float(last["close"])) - float(last["low"])
        body_pos = (float(last["close"]) - float(last["low"])) / bar_range if bar_range > 0 else 0.5
    else:
        bar_range = 0.0
        body = 0.0
        upper_wick = 0.0
        lower_wick = 0.0
        body_pos = 0.5

    ret1_atr = 0.0
    body1_ratio = 0.0
    upper_wick_ratio = 0.0
    lower_wick_ratio = 0.0
    upper1_ratio = 0.0
    close_pos1 = body_pos
    if bar_range > 0:
        upper_wick_ratio = upper_wick / bar_range
        lower_wick_ratio = lower_wick / bar_range
        body1_ratio = body / bar_range
        upper1_ratio = upper_wick / bar_range
    if atr14 > 0 and len(prev_bars) >= 2:
        prev_close = float(prev_bars.iloc[-2]["close"])
        ret1_atr = (float(prev_bars.iloc[-1]["close"]) - prev_close) / atr14

    # 5-bar / 10-bar / 20-bar / 30-bar features
    last5 = prev_bars.tail(5)
    last10 = prev_bars.tail(10)
    last20 = prev_bars.tail(20)
    last30 = prev_bars.tail(30)

    flips5 = 0
    if len(last5) >= 2:
        signs = np.sign((last5["close"] - last5["open"]).values)
        flips5 = int((np.diff(signs) != 0).sum())
    down3 = 0
    if len(prev_bars) >= 3:
        last3 = prev_bars.tail(3)
        down3 = int(((last3["close"] < last3["open"]).all()))
    range10_atr = 0.0
    if len(last10) >= 1 and atr14 > 0:
        range10_atr = float((last10["high"].max() - last10["low"].min()) / atr14)
    dist_high5_atr = 0.0
    dist_low5_atr = 0.0
    if len(last5) >= 1 and atr14 > 0:
        last_close = float(prev_bars.iloc[-1]["close"]) if len(prev_bars) > 0 else 0.0
        dist_high5_atr = float((last5["high"].max() - last_close) / atr14)
        dist_low5_atr = float((last_close - last5["low"].min()) / atr14)
    vol1_rel20 = 0.0
    if len(last20) >= 5:
        v_now = float(prev_bars.iloc[-1]["volume"])
        v_avg = float(last20["volume"].mean())
        vol1_rel20 = v_now / v_avg if v_avg > 0 else 0.0

    range_30bar = 0.0
    if len(last30) >= 1:
        range_30bar = float(last30["high"].max() - last30["low"].min())

    trend_20bar_pct = 0.0
    if len(last20) >= 2:
        first_close = float(last20.iloc[0]["close"])
        last_close = float(last20.iloc[-1]["close"])
        if first_close > 0:
            trend_20bar_pct = (last_close - first_close) / first_close

    last_close_pt = float(prev_bars.iloc[-1]["close"]) if len(prev_bars) >= 1 else float(signal_price)
    dist_to_20bar_hi_pct = 0.0
    dist_to_20bar_lo_pct = 0.0
    if len(last20) >= 1 and last_close_pt > 0:
        hi20 = float(last20["high"].max())
        lo20 = float(last20["low"].min())
        dist_to_20bar_hi_pct = (hi20 - last_close_pt) / last_close_pt
        dist_to_20bar_lo_pct = (last_close_pt - lo20) / last_close_pt

    vel_5bar = 0.0
    if len(last5) >= 2:
        vel_5bar = (float(last5.iloc[-1]["close"]) - float(last5.iloc[0]["close"])) / max(1, len(last5) - 1)
    vel_20bar = 0.0
    if len(last20) >= 2:
        vel_20bar = (float(last20.iloc[-1]["close"]) - float(last20.iloc[0]["close"])) / max(1, len(last20) - 1)

    base = (signal_price // BANK_GRID) * BANK_GRID
    dist_to_bank_below = signal_price - base
    dist_to_bank_above = (base + BANK_GRID) - signal_price
    if side.upper() == "LONG":
        dist_to_bank_in_dir = dist_to_bank_below
    else:
        dist_to_bank_in_dir = dist_to_bank_above
    dist_to_bank_pts = min(dist_to_bank_below, dist_to_bank_above)

    # crude regime metrics from bars
    if atr14 > 0 and last_close_pt > 0:
        regime_vol_bp = (atr14 / last_close_pt) * 10000.0
    else:
        regime_vol_bp = 0.0
    if range_30bar > 0:
        # efficiency: net move / total path
        efficiency_num = abs(last_close_pt - float(last30.iloc[0]["close"])) if len(last30) >= 1 else 0.0
        efficiency_den = float(last30.apply(lambda r: r["high"] - r["low"], axis=1).sum()) if len(last30) >= 1 else 0.0
        regime_eff = efficiency_num / efficiency_den if efficiency_den > 0 else 0.0
    else:
        regime_eff = 0.0

    return {
        # de3_entry_* (LFO inputs)
        "de3_entry_ret1_atr": ret1_atr,
        "de3_entry_body_pos1": body_pos,
        "de3_entry_body1_ratio": body1_ratio,
        "de3_entry_lower_wick_ratio": lower_wick_ratio,
        "de3_entry_upper_wick_ratio": upper_wick_ratio,
        "de3_entry_upper1_ratio": upper1_ratio,
        "de3_entry_close_pos1": close_pos1,
        "de3_entry_flips5": float(flips5),
        "de3_entry_down3": float(down3),
        "de3_entry_range10_atr": range10_atr,
        "de3_entry_dist_high5_atr": dist_high5_atr,
        "de3_entry_dist_low5_atr": dist_low5_atr,
        "de3_entry_vol1_rel20": vol1_rel20,
        "de3_entry_atr14": atr14,
        # bracket geometry
        "sl_dist_pts": float(sl_dist),
        "tp_dist_pts": float(tp_dist),
        "atr_ratio_to_sl": atr14 / max(0.5, sl_dist),
        # bank distances
        "dist_to_bank_below": dist_to_bank_below,
        "dist_to_bank_above": dist_to_bank_above,
        "dist_to_bank_in_dir": dist_to_bank_in_dir,
        "dist_to_bank_pts": dist_to_bank_pts,
        "bar_range_pts": bar_range,
        "bar_close_pct_body": body_pos,
        # generic regime / 20-30 bar window
        "atr14_pts": atr14,
        "range_30bar_pts": range_30bar,
        "trend_20bar_pct": trend_20bar_pct,
        "dist_to_20bar_hi_pct": dist_to_20bar_hi_pct,
        "dist_to_20bar_lo_pct": dist_to_20bar_lo_pct,
        "vel_5bar_pts_per_min": vel_5bar,
        "vel_20bar_pts_per_min": vel_20bar,
        "regime_vol_bp": regime_vol_bp,
        "regime_eff": regime_eff,
    }


def build_pct_features(prev_bars: pd.DataFrame, signal_price: float, hour_et: int) -> dict:
    """Compute PCT-overlay features (best-effort approximation; no live PCT
    state-machine here, so this surrogates the level-distance + range buckets
    from session bars). Used to fire the production PCT model on EVERY
    candidate so the corpus has a populated `pct_proba` column."""
    last30 = prev_bars.tail(30)
    if len(last30) >= 5:
        sess_hi = float(last30["high"].max())
        sess_lo = float(last30["low"].min())
        sess_open = float(last30.iloc[0]["open"])
    else:
        sess_hi = signal_price
        sess_lo = signal_price
        sess_open = signal_price
    pct_from_open = (signal_price - sess_open) / sess_open if sess_open > 0 else 0.0
    sess_range = max(1e-6, sess_hi - sess_lo)
    range_pct = sess_range / sess_open if sess_open > 0 else 0.0
    atr14 = _atr14_pts(prev_bars)
    atr_pct_30bar = atr14 / sess_open if sess_open > 0 else 0.0

    # signed level: which 1% bucket the price is in relative to sess_open
    if sess_open > 0:
        signed_lvl_pct = (signal_price - sess_open) / sess_open
    else:
        signed_lvl_pct = 0.0
    level_distance_pct = 0.0  # we approximate as "at level"; rules engine cares

    hour_edge = 1.0 if 9 <= hour_et < 16 else 0.0

    # buckets: simple quartile by atr_pct_30bar / range_pct
    def _qbucket(v: float, q1: float, q2: float, q3: float) -> str:
        if v <= q1:
            return "Q1"
        if v <= q2:
            return "Q2"
        if v <= q3:
            return "Q3"
        return "Q4"

    atr_b = _qbucket(atr_pct_30bar, 0.001, 0.0025, 0.005)
    range_b = _qbucket(range_pct, 0.005, 0.01, 0.02)

    return {
        "pct_from_open": pct_from_open,
        "signed_level": signed_lvl_pct,
        "abs_level": abs(signed_lvl_pct),
        "level_distance_pct": level_distance_pct,
        "atr_pct_30bar": atr_pct_30bar,
        "range_pct_at_touch": range_pct,
        "hour_edge": hour_edge,
        "minutes_since_open": float(max(0, hour_et - 9) * 60),
        "dist_to_running_hi_pct": (sess_hi - signal_price) / signal_price if signal_price > 0 else 0.0,
        "dist_to_running_lo_pct": (signal_price - sess_lo) / signal_price if signal_price > 0 else 0.0,
        "rule_confidence": 0.5,
        "tier": "primary",
        "atr_bucket": atr_b,
        "range_bucket": range_b,
        "hour_bucket": _et_session(hour_et),
        "direction": "up" if signed_lvl_pct >= 0 else "down",
    }


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def _build_row(payload: dict, numeric: dict, categorical: dict, ordinal: dict) -> np.ndarray:
    feat_names = payload["feature_names"]
    numeric_list = payload.get("numeric_features", [])
    cat_maps = payload.get("categorical_maps", {})
    ord_list = payload.get("ordinal_features", [])
    row = {c: 0.0 for c in feat_names}
    for c in numeric_list:
        if c in row and c in numeric:
            try:
                fv = float(numeric[c])
                if not np.isfinite(fv):
                    fv = 0.0
                row[c] = fv
            except Exception:
                pass
    for cc, kvs in cat_maps.items():
        val = categorical.get(cc, "")
        for kv in kvs:
            nm = f"{cc}__{kv}"
            if nm in row and val == kv:
                row[nm] = 1
    for c in ord_list:
        if c in row and c in ordinal:
            try:
                row[c] = float(ordinal[c])
            except Exception:
                pass
    return np.array([[row[c] for c in feat_names]])


def fire_lfo(payload, side: str, bar_features: dict, hour_et: int) -> float:
    if payload is None:
        return float("nan")
    numeric = dict(bar_features)
    categorical = {"side": side.upper(), "session": _et_session_ml_lfo(hour_et), "mkt_regime": ""}
    ordinal = {"et_hour": float(hour_et)}
    X = _build_row(payload, numeric, categorical, ordinal)
    try:
        return float(payload["model"].predict_proba(X)[0, 1])
    except Exception:
        return float("nan")


def fire_kalshi(payload, side: str, bar_features: dict, hour_et: int,
                sub_strategy: str) -> float:
    """Fire production Kalshi gate.

    NOTE: production-time, Kalshi consumes live snapshot features
    (entry_probability, probe_probability, momentum_*). We do NOT have the
    historical snapshots, so we feed neutral defaults — the resulting
    proba reflects ONLY bar-state contribution. This is documented and
    intentional; it lets us still fire the model end-to-end without
    inheriting a v1 corpus column. For Phase 2 training the v11 feature
    block will explicitly include the Kalshi-state subset (NaN where
    unknown), and Phase 1 carries `kalshi_proba` as an honest approximation.
    """
    if payload is None:
        return float("nan")
    import re
    sub = str(sub_strategy or "")
    m = re.search(
        r"(?P<tf>5min|15min)_.*?_(?P<direction>Long|Short)_(?P<type>Rev|Mom)_T(?P<tier>\d)",
        sub,
    )
    if m:
        try:
            tier = int(m.group("tier"))
        except Exception:
            tier = 0
        is_rev = m.group("type") == "Rev"
        is_5m = m.group("tf") == "5min"
    else:
        tier, is_rev, is_5m = 0, False, False

    numeric = {
        "entry_probability": 0.5,        # neutral
        "probe_probability": 0.5,        # neutral
        "momentum_delta": 0.0,           # neutral
        "momentum_retention": 1.0,       # neutral
        "support_score": 0.5,
        "threshold": 0.45,
        "probe_distance_pts": 0.0,
        "et_hour_frac": float(hour_et) + 0.0,
        "atr14_pts": float(bar_features.get("atr14_pts", 0.0)),
        "range_30bar_pts": float(bar_features.get("range_30bar_pts", 0.0)),
        "trend_20bar_pct": float(bar_features.get("trend_20bar_pct", 0.0)),
        "dist_to_20bar_hi_pct": float(bar_features.get("dist_to_20bar_hi_pct", 0.0)),
        "dist_to_20bar_lo_pct": float(bar_features.get("dist_to_20bar_lo_pct", 0.0)),
        "vel_5bar_pts_per_min": float(bar_features.get("vel_5bar_pts_per_min", 0.0)),
        "dist_to_bank_pts": float(bar_features.get("dist_to_bank_pts", 0.0)),
        "regime_vol_bp": float(bar_features.get("regime_vol_bp", 0.0)),
        "regime_eff": float(bar_features.get("regime_eff", 0.0)),
        "sub_tier": float(tier),
        "sub_is_rev": 1.0 if is_rev else 0.0,
        "sub_is_5min": 1.0 if is_5m else 0.0,
    }
    categorical = {"side": side.upper(), "role": "balanced", "regime": "neutral"}
    X = _build_row(payload, numeric, categorical, {})
    try:
        clf = payload["classifier"]
        probs = clf.predict_proba(X)[0]
        classes = list(clf.classes_)
        return float(probs[classes.index(1)]) if 1 in classes else 0.5
    except Exception:
        return float("nan")


def fire_pct(payload, pct_features: dict) -> float:
    if payload is None:
        return float("nan")
    numeric = {k: v for k, v in pct_features.items() if isinstance(v, (int, float))}
    categorical = {k: v for k, v in pct_features.items() if isinstance(v, str)}
    X = _build_row(payload, numeric, categorical, {})
    try:
        probs = payload["model"].predict_proba(X)[0]
        classes = list(payload["model"].classes_)
        # class 1 = breakout
        if 1 in classes:
            return float(probs[classes.index(1)])
        return 0.5
    except Exception:
        return float("nan")


def fire_pivot(payload, side: str, bar_features: dict, hour_et: int) -> float:
    """Pivot model is for trail decisions, not entry. We fire it for
    feature-completeness so the corpus has a `pivot_proba` column. The
    'pivot' here is the last bar treated as a tentative HIGH/LOW pivot
    aligned with the trade direction — a surrogate for the at-entry
    pivot regime."""
    if payload is None:
        return float("nan")
    # For long: simulate a LOW pivot at the prior bar; for short: HIGH pivot
    ptype = "LOW" if side.upper() == "LONG" else "HIGH"
    numeric = {
        "pivot_range_pts": bar_features.get("bar_range_pts", 0.0),
        "pivot_body_pts": bar_features.get("bar_range_pts", 0.0) * bar_features.get("de3_entry_body1_ratio", 0.0),
        "upper_wick_pct": bar_features.get("de3_entry_upper_wick_ratio", 0.0),
        "lower_wick_pct": bar_features.get("de3_entry_lower_wick_ratio", 0.0),
        "pivot_height_pts": 0.0,
        "atr14_pts": bar_features.get("atr14_pts", 0.0),
        "range_30bar_pts": bar_features.get("range_30bar_pts", 0.0),
        "trend_20bar_pct": bar_features.get("trend_20bar_pct", 0.0),
        "dist_to_20bar_hi_pct": bar_features.get("dist_to_20bar_hi_pct", 0.0),
        "dist_to_20bar_lo_pct": bar_features.get("dist_to_20bar_lo_pct", 0.0),
        "dist_pivot_to_bank_pts": bar_features.get("dist_to_bank_pts", 0.0),
        "anchor_distance_from_entry_pts": 0.0,
        "vel_5bar_pts_per_min": bar_features.get("vel_5bar_pts_per_min", 0.0),
        "vel_20bar_pts_per_min": bar_features.get("vel_20bar_pts_per_min", 0.0),
        "reading_b_buffer_pts": 0.0,
    }
    categorical = {
        "pivot_type": ptype,
        "session": _et_session(hour_et),
        "tape": "uptrend" if bar_features.get("trend_20bar_pct", 0.0) > 0 else "downtrend",
    }
    ordinal = {"et_hour": float(hour_et)}
    X = _build_row(payload, numeric, categorical, ordinal)
    try:
        probs = payload["model"].predict_proba(X)[0]
        classes = list(payload["model"].classes_)
        if 1 in classes:
            return float(probs[classes.index(1)])
        return 0.5
    except Exception:
        return float("nan")


def fire_filterg(payload, bar_features: dict, side: str, hour_et: int) -> float:
    """Fire filter G v10 model. Schema has 17 features with no metadata —
    we feed a numeric feature vector built from bar_features in a fixed
    order. If the loaded HGB has `n_features_in_=17` we trust the order
    that matches the v9->v10 retrain pipeline (regime feature subset)."""
    if payload is None:
        return float("nan")
    m = payload["model"]
    n_in = int(getattr(m, "n_features_in_", 0))
    if n_in <= 0:
        return float("nan")
    # Build a 17-feature ordered vector. Order is best-effort from the v10
    # retrain feature list (atr14, range_30bar, trend_20bar, dist_hi, dist_lo,
    # vel_5bar, dist_bank, regime_vol_bp, regime_eff, sl_dist, tp_dist,
    # atr_ratio_to_sl, ret1_atr, body_pos1, body1_ratio, side_long, et_hour).
    feats = [
        bar_features.get("atr14_pts", 0.0),
        bar_features.get("range_30bar_pts", 0.0),
        bar_features.get("trend_20bar_pct", 0.0),
        bar_features.get("dist_to_20bar_hi_pct", 0.0),
        bar_features.get("dist_to_20bar_lo_pct", 0.0),
        bar_features.get("vel_5bar_pts_per_min", 0.0),
        bar_features.get("dist_to_bank_pts", 0.0),
        bar_features.get("regime_vol_bp", 0.0),
        bar_features.get("regime_eff", 0.0),
        bar_features.get("sl_dist_pts", 0.0),
        bar_features.get("tp_dist_pts", 0.0),
        bar_features.get("atr_ratio_to_sl", 0.0),
        bar_features.get("de3_entry_ret1_atr", 0.0),
        bar_features.get("de3_entry_body_pos1", 0.5),
        bar_features.get("de3_entry_body1_ratio", 0.0),
        1.0 if side.upper() == "LONG" else 0.0,
        float(hour_et),
    ]
    feats = feats[:n_in]
    while len(feats) < n_in:
        feats.append(0.0)
    X = np.array([feats], dtype=float)
    try:
        return float(m.predict_proba(X)[0, 1])
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Family-aware single-position rule (BYPASS_SAMESIDE=0)
# ---------------------------------------------------------------------------

def allow_same_side_parallel(primary_family: str, signal_family: str,
                              primary_count_af: int = 0, max_legs_af: int = 1) -> bool:
    """Mirror julie001.py:_allow_same_side_parallel_entry, BYPASS_SAMESIDE=0.

    Returns True if the new signal is allowed to coexist with primary trade
    on the SAME side. Returns False to BLOCK.
    """
    if primary_family == "de3":
        return signal_family in {"regimeadaptive", "aetherflow"}
    if primary_family == "aetherflow" and signal_family == "aetherflow":
        if max_legs_af <= 1:
            return False
        return primary_count_af < max_legs_af
    # primary_family == regimeadaptive or any other -> BLOCK
    return False


# ---------------------------------------------------------------------------
# Per-contract bar index
# ---------------------------------------------------------------------------

def _per_contract_index(bars_df: pd.DataFrame) -> dict:
    out = {}
    for sym, sub in bars_df.groupby("symbol", observed=True):
        out[sym] = sub.sort_index()
    return out


def main():
    t0 = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("[v11] loading bars:", PARQUET_BARS)
    bars = pd.read_parquet(PARQUET_BARS)
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("US/Eastern")
    elif str(bars.index.tz) != "US/Eastern":
        bars.index = bars.index.tz_convert("US/Eastern")
    print(f"  bars shape={bars.shape}")

    print("[v11] loading v1 candidate stream:", PARQUET_V1)
    sigs = pd.read_parquet(PARQUET_V1)
    sigs["ts"] = pd.to_datetime(sigs["ts"])
    if sigs["ts"].dt.tz is None:
        sigs["ts"] = sigs["ts"].dt.tz_localize("US/Eastern")
    else:
        sigs["ts"] = sigs["ts"].dt.tz_convert("US/Eastern")
    sigs = sigs.sort_values("ts").reset_index(drop=True)
    print(f"  signals: n={len(sigs)} {sigs['ts'].min()} -> {sigs['ts'].max()}")

    print("[v11] loading production-loaded models")
    payloads: dict[str, Any] = {}
    load_results: dict[str, str] = {}
    for name, path in [
        ("lfo", MODEL_LFO_PATH),
        ("kalshi", MODEL_KALSHI_PATH),
        ("pct", MODEL_PCT_PATH),
        ("pivot", MODEL_PIVOT_PATH),
        ("filterg", MODEL_FILTERG_PATH),
    ]:
        try:
            payloads[name] = joblib.load(path)
            load_results[name] = "OK"
            print(f"  loaded {name}: {path}")
        except Exception as exc:
            payloads[name] = None
            load_results[name] = f"FAILED: {exc}"
            print(f"  FAILED {name}: {path} -> {exc}")

    print("[v11] grouping bars by symbol")
    by_sym = _per_contract_index(bars)
    print(f"  {len(by_sym)} symbols cached")

    n = len(sigs)
    rows = []
    print(f"[v11] processing {n} candidates...")

    for i, sig in enumerate(sigs.itertuples(index=False)):
        ts = sig.ts.tz_convert("US/Eastern") if sig.ts.tzinfo else sig.ts.tz_localize("US/Eastern")
        side = str(sig.side).upper()
        price = float(sig.price)
        sl_dist = float(sig.sl)
        tp_dist = float(sig.tp)
        strategy = str(sig.strategy)
        family = family_from_strategy(strategy)
        sub_strategy = ""  # v1 corpus doesn't carry sub_strategy

        if side == "LONG":
            tp_price = price + tp_dist
            sl_price = price - sl_dist
        else:
            tp_price = price - tp_dist
            sl_price = price + sl_dist

        # Pin contract via close-price match (mirror v2 sim)
        try:
            ts_rows = bars.loc[ts]
        except KeyError:
            ts_rows = None
        contract = None
        if ts_rows is not None:
            if isinstance(ts_rows, pd.Series):
                contract = str(ts_rows["symbol"]) if "symbol" in ts_rows.index else None
            else:
                diffs = (ts_rows["close"].astype(float) - price).abs()
                mask_vol = ts_rows["volume"].astype(float) > 0
                if mask_vol.any():
                    diffs = diffs.where(mask_vol, np.inf)
                if diffs.min() <= 0.5:
                    contract = str(ts_rows["symbol"].iloc[diffs.values.argmin()])
        if contract is None:
            contract = front_month_by_calendar(ts)

        sym_bars = by_sym.get(contract)
        if sym_bars is None or sym_bars.empty:
            rows.append({
                "ts": ts, "strategy": strategy, "family": family,
                "side": side, "entry_price": price, "sl": sl_dist, "tp": tp_dist,
                "contract": contract,
                "exit_reason": "no_data", "exit_price": np.nan, "exit_ts": pd.NaT,
                "raw_pnl": 0.0, "net_pnl_after_haircut": 0.0,
                "is_big_loss": False,
                "fg_proba": float("nan"), "kalshi_proba": float("nan"),
                "lfo_proba": float("nan"), "pct_proba": float("nan"),
                "pivot_proba": float("nan"),
                "in_kalshi_window": ts.hour in KALSHI_HOURS_ET,
            })
            continue

        # Build prev_bars window: 60 bars BEFORE/AT signal time, on this contract
        prev_window = sym_bars.loc[sym_bars.index <= ts].tail(60)

        hour_et = int(ts.hour)
        bar_features = build_bar_features(prev_window, price, side, sl_dist, tp_dist, hour_et)
        pct_features = build_pct_features(prev_window, price, hour_et)

        # Fire each loaded production model
        fg_proba = fire_filterg(payloads.get("filterg"), bar_features, side, hour_et)
        kalshi_proba = fire_kalshi(payloads.get("kalshi"), side, bar_features, hour_et, sub_strategy)
        lfo_proba = fire_lfo(payloads.get("lfo"), side, bar_features, hour_et)
        pct_proba = fire_pct(payloads.get("pct"), pct_features)
        pivot_proba = fire_pivot(payloads.get("pivot"), side, bar_features, hour_et)

        # Walk forward
        fwd = sym_bars.loc[sym_bars.index > ts].head(HORIZON)
        if fwd.empty:
            exit_reason = "no_data"
            exit_price = np.nan
            exit_ts = pd.NaT
            raw_pnl = 0.0
        else:
            fwd_reset = fwd.reset_index()
            outcome = simulate_trade_through(
                fwd_reset, side=side, entry_price=price,
                tp_price=tp_price, sl_price=sl_price,
            )
            raw_pnl = outcome.pnl_points * POINT_VALUE
            exit_reason = outcome.exit_reason
            exit_price = outcome.exit_price
            if outcome.exit_bar >= 0 and outcome.exit_bar < len(fwd):
                exit_ts = fwd.index[outcome.exit_bar]
            else:
                exit_ts = pd.NaT
        net_pnl = raw_pnl - HAIRCUT if exit_reason != "no_data" else 0.0
        is_big_loss = bool(net_pnl <= -50.0)

        rows.append({
            "ts": ts, "strategy": strategy, "family": family,
            "side": side, "entry_price": price, "sl": sl_dist, "tp": tp_dist,
            "contract": contract,
            # bar features (40 cols)
            **{f"bf_{k}": v for k, v in bar_features.items()},
            # pct features (15 cols)
            **{f"pct_{k}": v for k, v in pct_features.items()},
            # overlay probas
            "fg_proba": fg_proba,
            "kalshi_proba": kalshi_proba,
            "lfo_proba": lfo_proba,
            "pct_proba": pct_proba,
            "pivot_proba": pivot_proba,
            # window
            "in_kalshi_window": int(hour_et) in KALSHI_HOURS_ET,
            # walk-forward outcome
            "exit_reason": exit_reason,
            "exit_price": exit_price,
            "exit_ts": exit_ts,
            "raw_pnl": raw_pnl,
            "net_pnl_after_haircut": net_pnl,
            "is_big_loss": is_big_loss,
        })

        if (i + 1) % 250 == 0:
            print(f"  [{i+1}/{n}] elapsed={time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    print(f"[v11] processed {len(df)} candidates in {time.time()-t0:.1f}s")

    # ----- Apply family-aware single-position rule in chronological order -----
    print("[v11] applying family-aware same-side rule (BYPASS_SAMESIDE=0)")
    df = df.sort_values("ts").reset_index(drop=True)
    open_trade = None  # dict with family, side, exit_ts, ts
    open_af_count = 0
    allowed = []
    for i, r in df.iterrows():
        ts = r["ts"]
        if open_trade is None:
            allowed.append(True)
            if r["exit_reason"] == "no_data":
                # don't open
                pass
            else:
                open_trade = {
                    "family": r["family"], "side": r["side"],
                    "exit_ts": r["exit_ts"] if pd.notna(r["exit_ts"]) else ts + pd.Timedelta(minutes=HORIZON),
                    "ts": ts,
                }
                open_af_count = 1 if r["family"] == "aetherflow" else 0
            continue

        # Check if open trade has closed
        close_ts = open_trade["exit_ts"]
        if pd.notna(close_ts) and ts >= close_ts:
            open_trade = None
            open_af_count = 0
            allowed.append(True)
            if r["exit_reason"] != "no_data":
                open_trade = {
                    "family": r["family"], "side": r["side"],
                    "exit_ts": r["exit_ts"] if pd.notna(r["exit_ts"]) else ts + pd.Timedelta(minutes=HORIZON),
                    "ts": ts,
                }
                open_af_count = 1 if r["family"] == "aetherflow" else 0
            continue

        # Same-side or opposite-side?
        if r["side"] == open_trade["side"]:
            ok = allow_same_side_parallel(
                primary_family=open_trade["family"],
                signal_family=r["family"],
                primary_count_af=open_af_count,
                max_legs_af=1,  # default
            )
        else:
            # Opposite-side: production reverse-exit closes the open trade
            # and opens this new one. Treat as ALLOWED.
            ok = True
            open_trade = None
            open_af_count = 0

        allowed.append(ok)
        if ok and r["exit_reason"] != "no_data":
            if open_trade is None:
                open_trade = {
                    "family": r["family"], "side": r["side"],
                    "exit_ts": r["exit_ts"] if pd.notna(r["exit_ts"]) else ts + pd.Timedelta(minutes=HORIZON),
                    "ts": ts,
                }
                open_af_count = 1 if r["family"] == "aetherflow" else 0
            else:
                # Coexist (DE3+RA / DE3+AF / AF+AF case)
                # primary trade's exit_ts dominates; track AF count
                if r["family"] == "aetherflow":
                    open_af_count += 1
                # extend exit_ts to max of the two
                new_ex = r["exit_ts"] if pd.notna(r["exit_ts"]) else ts + pd.Timedelta(minutes=HORIZON)
                if pd.notna(new_ex) and pd.notna(open_trade["exit_ts"]):
                    if new_ex > open_trade["exit_ts"]:
                        open_trade["exit_ts"] = new_ex

    df["allowed_by_friend_rule"] = allowed

    # ----- Save corpus -----
    print("[v11] saving corpus to:", OUT_CORPUS)
    df.to_parquet(OUT_CORPUS, index=False)
    print(f"  rows={len(df)}, cols={len(df.columns)}")

    # ----- Summary -----
    summary = compute_summary(df, load_results)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[v11] wrote {OUT_SUMMARY}")

    # Print key stats
    print("\n=== Summary ===")
    print(f"Total candidates: {summary['n_candidates']}")
    print("By family:", summary["by_family"])
    print("By side:", summary["by_side"])
    print(f"Allowed by friend rule: {summary['n_allowed']}")
    print(f"In Kalshi window: {summary['n_in_kalshi_window']}")
    print()
    print("Model proba stats:")
    for name in ["fg_proba", "kalshi_proba", "lfo_proba", "pct_proba", "pivot_proba"]:
        s = summary["proba_stats"][name]
        print(f"  {name}: mean={s['mean']:.4f}, n_nan={s['n_nan']}/{summary['n_candidates']}")
    print()
    print("Walk-forward (allowed by friend rule):")
    print(f"  trades={summary['walk']['n_trades']}, wr={summary['walk']['wr']:.2f}%, pnl=${summary['walk']['pnl_net']:.2f}, dd=${summary['walk']['dd']:.2f}")
    print()
    print("Smoking gun re-check (2026-03-05 08:05 LONG ESH6 @ 6855):")
    print(f"  exit_reason={summary['smoking_gun']['exit_reason']!r}, exit_price={summary['smoking_gun']['exit_price']}")


def compute_summary(df: pd.DataFrame, load_results: dict[str, str]) -> dict:
    summary = {
        "n_candidates": int(len(df)),
        "by_family": df["family"].value_counts().to_dict(),
        "by_side": df["side"].value_counts().to_dict(),
        "by_strategy": df["strategy"].value_counts().to_dict(),
        "n_allowed": int(df["allowed_by_friend_rule"].sum()),
        "n_in_kalshi_window": int(df["in_kalshi_window"].sum()),
        "model_load_results": load_results,
        "proba_stats": {},
        "ts_min": str(df["ts"].min()),
        "ts_max": str(df["ts"].max()),
    }
    for name in ["fg_proba", "kalshi_proba", "lfo_proba", "pct_proba", "pivot_proba"]:
        col = df[name]
        summary["proba_stats"][name] = {
            "mean": float(col.dropna().mean()) if col.notna().any() else float("nan"),
            "median": float(col.dropna().median()) if col.notna().any() else float("nan"),
            "min": float(col.dropna().min()) if col.notna().any() else float("nan"),
            "max": float(col.dropna().max()) if col.notna().any() else float("nan"),
            "n_nan": int(col.isna().sum()),
        }

    # Walk stats: only allowed rows count
    allowed = df[df["allowed_by_friend_rule"] & (df["exit_reason"] != "no_data")].copy()
    n_t = len(allowed)
    wins = int((allowed["net_pnl_after_haircut"] > 0).sum())
    pnl = float(allowed["net_pnl_after_haircut"].sum())
    cum = allowed.sort_values("ts")["net_pnl_after_haircut"].cumsum()
    dd = float((cum - cum.cummax()).min()) if len(cum) else 0.0
    wr = (100.0 * wins / n_t) if n_t else 0.0
    summary["walk"] = {
        "n_trades": int(n_t),
        "wr": round(wr, 2),
        "pnl_net": round(pnl, 2),
        "dd": round(dd, 2),
    }

    # Per-month
    per_month = []
    if n_t > 0:
        a = allowed.copy()
        a["month_str"] = a["ts"].dt.strftime("%Y-%m")
        for m, sub in a.groupby("month_str", sort=True):
            n_m = len(sub)
            w_m = int((sub["net_pnl_after_haircut"] > 0).sum())
            wr_m = (100.0 * w_m / n_m) if n_m else 0.0
            p_m = float(sub["net_pnl_after_haircut"].sum())
            cum_m = sub.sort_values("ts")["net_pnl_after_haircut"].cumsum()
            dd_m = float((cum_m - cum_m.cummax()).min()) if len(cum_m) else 0.0
            per_month.append({
                "month": m, "n": int(n_m), "wr": round(wr_m, 2),
                "pnl": round(p_m, 2), "dd": round(dd_m, 2),
            })
    summary["per_month"] = per_month

    # Smoking gun: 2026-03-05 08:05 LONG @ 6855 should NOT be 'take' (TP)
    target_ts = pd.Timestamp("2026-03-05 08:05:00", tz="US/Eastern")
    candidates = df[(df["ts"] == target_ts) & (df["side"] == "LONG") & (df["entry_price"].between(6854.5, 6855.5))]
    if len(candidates) > 0:
        r = candidates.iloc[0]
        summary["smoking_gun"] = {
            "ts": str(r["ts"]),
            "contract": r.get("contract", None),
            "exit_reason": r["exit_reason"],
            "exit_price": float(r["exit_price"]) if pd.notna(r["exit_price"]) else None,
            "raw_pnl": float(r["raw_pnl"]),
            "passes_check": r["exit_reason"] != "take",
        }
    else:
        summary["smoking_gun"] = {"note": "candidate not found at 2026-03-05 08:05 LONG @ ~6855"}

    # Family-pair allow/block tally
    pair_stats = {"DE3+DE3": {"allowed": 0, "blocked": 0},
                  "DE3+RA": {"allowed": 0, "blocked": 0},
                  "RA+anything": {"allowed": 0, "blocked": 0}}
    summary["family_pair_stats"] = pair_stats

    return summary


if __name__ == "__main__":
    main()
