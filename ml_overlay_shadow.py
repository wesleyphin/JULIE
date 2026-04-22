"""Shadow-mode ML scoring for LFO + PCT overlay.

Loads the two ML models (model_lfo.joblib, model_pct_overlay.joblib) once
at startup. Exposes:
  - score_lfo(signal, current_price, bank_bar_features) -> (p_wait_better, veto_thr)
  - score_pct_overlay(state) -> (p_breakout, rule_bias_match)

Shadow mode: callers log both rule-based and ML decisions side-by-side
without changing live behavior. Flip JULIE_ML_LFO_ACTIVE=1 or
JULIE_ML_PCT_ACTIVE=1 to let ML take over.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "signal_gate_2025"

# Loaded on demand
_LFO_PAYLOAD: Optional[Dict[str, Any]] = None
_PCT_PAYLOAD: Optional[Dict[str, Any]] = None
_PIVOT_PAYLOAD: Optional[Dict[str, Any]] = None
_KALSHI_PAYLOAD: Optional[Dict[str, Any]] = None     # entry-gate ML (dual clf+reg)
_KALSHI_TP_PAYLOAD: Optional[Dict[str, Any]] = None  # TP-aligned Kalshi ML


def _try_load(fname: str) -> Optional[Dict[str, Any]]:
    path = _ARTIFACT_DIR / fname
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:
        logging.warning("ml_overlay_shadow: failed loading %s: %s", fname, exc)
        return None


def init_ml_overlays() -> Tuple[bool, bool, bool, bool, bool]:
    """Eager-load all five ML overlay models.

    Returns (lfo_loaded, pct_loaded, pivot_loaded, kalshi_loaded,
             kalshi_tp_loaded).
    """
    global _LFO_PAYLOAD, _PCT_PAYLOAD, _PIVOT_PAYLOAD, _KALSHI_PAYLOAD, _KALSHI_TP_PAYLOAD
    _LFO_PAYLOAD = _try_load("model_lfo.joblib")
    _PCT_PAYLOAD = _try_load("model_pct_overlay.joblib")
    _PIVOT_PAYLOAD = _try_load("model_pivot_trail.joblib")
    _KALSHI_PAYLOAD = _try_load("model_kalshi_gate.joblib")
    _KALSHI_TP_PAYLOAD = _try_load("model_kalshi_tp_gate.joblib")
    if _LFO_PAYLOAD:
        logging.info(
            "ml_overlay_shadow: LFO model loaded — %d features, thr=%.3f, cv_auc=%.3f",
            len(_LFO_PAYLOAD["feature_names"]), _LFO_PAYLOAD.get("veto_threshold", 0.5),
            _LFO_PAYLOAD.get("cv_auc_mean", 0.0),
        )
    if _PCT_PAYLOAD:
        logging.info(
            "ml_overlay_shadow: PCT overlay model loaded — %d features, cv_auc=%.3f",
            len(_PCT_PAYLOAD["feature_names"]),
            _PCT_PAYLOAD.get("cv_auc_mean", 0.0),
        )
    if _PIVOT_PAYLOAD:
        logging.info(
            "ml_overlay_shadow: Pivot-trail model loaded — %d features, thr=%.3f, cv_auc=%.3f",
            len(_PIVOT_PAYLOAD["feature_names"]),
            _PIVOT_PAYLOAD.get("hold_threshold", 0.55),
            _PIVOT_PAYLOAD.get("cv_auc_mean", 0.0),
        )
    if _KALSHI_PAYLOAD:
        logging.info(
            "ml_overlay_shadow: Kalshi-gate model loaded — %d features, rolling_auc=%.3f, "
            "thr=%.2f (clf+reg dual)",
            len(_KALSHI_PAYLOAD["feature_names"]),
            _KALSHI_PAYLOAD.get("rolling_origin_mean_auc", 0.0) or 0.0,
            _KALSHI_PAYLOAD.get("pass_threshold", 0.50),
        )
    if _KALSHI_TP_PAYLOAD:
        logging.info(
            "ml_overlay_shadow: Kalshi-TP model loaded — %d features, rolling_auc=%.3f, "
            "thr=%.2f (TP-price aligned probability)",
            len(_KALSHI_TP_PAYLOAD["feature_names"]),
            _KALSHI_TP_PAYLOAD.get("rolling_origin_mean_auc", 0.0) or 0.0,
            _KALSHI_TP_PAYLOAD.get("pass_threshold", 0.50),
        )
    return (
        _LFO_PAYLOAD is not None,
        _PCT_PAYLOAD is not None,
        _PIVOT_PAYLOAD is not None,
        _KALSHI_PAYLOAD is not None,
        _KALSHI_TP_PAYLOAD is not None,
    )


def _build_row(payload: Dict[str, Any], numeric: Dict[str, float],
                categorical: Dict[str, str], ordinal: Dict[str, float]) -> np.ndarray:
    feat_names = payload["feature_names"]
    numeric_list = payload.get("numeric_features", [])
    cat_maps = payload.get("categorical_maps", {})
    ord_list = payload.get("ordinal_features", [])
    row = {c: 0.0 for c in feat_names}
    for c in numeric_list:
        if c in row and c in numeric:
            try:
                fv = float(numeric[c])
                if not np.isfinite(fv): fv = 0.0
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


def score_lfo(signal: Dict[str, Any], bar_features: Dict[str, float],
               dist_to_bank_below: float, dist_to_bank_above: float,
               bar_range_pts: float, bar_close_pct_body: float,
               sl_dist: float, tp_dist: float,
               session: str, mkt_regime: str, et_hour: int) -> Optional[Tuple[float, float]]:
    """Return (p_wait_better, veto_threshold) or None if model not loaded."""
    if _LFO_PAYLOAD is None:
        return None
    side = str(signal.get("side", "")).upper()
    is_long = side == "LONG"
    dist_in_dir = dist_to_bank_below if is_long else dist_to_bank_above
    atr14 = float(bar_features.get("de3_entry_atr14", 0) or 0)
    numeric = dict(bar_features)
    numeric.update({
        "dist_to_bank_below": dist_to_bank_below,
        "dist_to_bank_above": dist_to_bank_above,
        "dist_to_bank_in_dir": dist_in_dir,
        "bar_range_pts": bar_range_pts,
        "bar_close_pct_body": bar_close_pct_body,
        "sl_dist_pts": sl_dist,
        "tp_dist_pts": tp_dist,
        "atr_ratio_to_sl": atr14 / max(0.5, sl_dist),
    })
    categorical = {"side": side, "session": session, "mkt_regime": mkt_regime}
    ordinal = {"et_hour": float(et_hour)}
    X = _build_row(_LFO_PAYLOAD, numeric, categorical, ordinal)
    try:
        p = float(_LFO_PAYLOAD["model"].predict_proba(X)[0, 1])
    except Exception:
        return None
    return p, float(_LFO_PAYLOAD.get("veto_threshold", 0.5))


def score_pct_overlay(state: Any) -> Optional[Tuple[float, str]]:
    """Score a PctLevelOverlay state. Returns (p_breakout, ml_bias) or None."""
    if _PCT_PAYLOAD is None or state is None or not getattr(state, "at_level", False):
        return None
    if state.session_open is None or state.nearest_level is None:
        return None
    import pct_level_overlay as _p  # noqa — to match imports at shared layer
    signed_lvl = float(state.nearest_level)
    numeric = {
        "pct_from_open": float(state.pct_from_open or 0),
        "signed_level": signed_lvl,
        "abs_level": abs(signed_lvl),
        "level_distance_pct": float(state.level_distance_pct or 0),
        "atr_pct_30bar": float(state.atr_pct_30bar or 0),
        "range_pct_at_touch": float(state.range_pct_at_touch or 0),
        "hour_edge": float(state.hour_edge or 0),
        "minutes_since_open": 0.0,  # unknown here — caller can override
        "dist_to_running_hi_pct": 0.0,
        "dist_to_running_lo_pct": 0.0,
        "rule_confidence": float(state.confidence or 0),
    }
    # Hour bucket from state.ts if present
    hour_et = getattr(state.ts, "hour", 12) if state.ts is not None else 12
    if 18 <= hour_et or hour_et < 3: h_bucket = "ASIA"
    elif 3 <= hour_et < 7: h_bucket = "LONDON"
    elif 7 <= hour_et < 9: h_bucket = "NY_PRE"
    elif 9 <= hour_et < 12: h_bucket = "NY_AM"
    elif 12 <= hour_et < 16: h_bucket = "NY_PM"
    else: h_bucket = "POST"
    categorical = {
        "tier": state.tier or "",
        "atr_bucket": state.atr_bucket or "",
        "range_bucket": state.range_bucket or "",
        "hour_bucket": h_bucket,
        "direction": "up" if signed_lvl >= 0 else "down",
    }
    X = _build_row(_PCT_PAYLOAD, numeric, categorical, {})
    try:
        probs = _PCT_PAYLOAD["model"].predict_proba(X)[0]
        classes = list(_PCT_PAYLOAD["model"].classes_)
        # LBL_BREAKOUT = 1
        if 1 in classes:
            p_bo = float(probs[classes.index(1)])
        else:
            p_bo = 0.5
    except Exception:
        return None
    ml_bias = "breakout_lean" if p_bo >= 0.55 else ("pivot_lean" if p_bo <= 0.45 else "neutral")
    return p_bo, ml_bias


def score_pivot_trail(
    pivot_type: str,
    pivot_price: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    *,
    atr14_pts: float,
    range_30bar_pts: float,
    trend_20bar_pct: float,
    dist_to_20bar_hi_pct: float,
    dist_to_20bar_lo_pct: float,
    vel_5bar_pts_per_min: float,
    vel_20bar_pts_per_min: float,
    anchor_c: float,
    anchor_b: float,
    session: str,
    tape: str,
    et_hour: int,
    anchor_distance_from_entry_pts: float = 0.0,
) -> Optional[Tuple[float, bool]]:
    """Return (p_hold, should_ratchet) for a confirmed swing pivot, or None if
    model not loaded. ``should_ratchet`` = p_hold >= the payload's default
    threshold (0.55 by default, tunable via the joblib)."""
    if _PIVOT_PAYLOAD is None:
        return None
    ptype = str(pivot_type).upper()
    if ptype not in ("HIGH", "LOW"):
        return None
    pivot_range = max(1e-9, bar_high - bar_low)
    upper_wick = (bar_high - max(bar_open, bar_close)) / pivot_range
    lower_wick = (min(bar_open, bar_close) - bar_low) / pivot_range
    pivot_body = abs(bar_close - bar_open)
    pivot_height = abs(pivot_price - (bar_open + bar_close) / 2.0)
    dist_pivot_to_bank = abs(pivot_price - anchor_c)
    reading_b_buffer = abs(anchor_c - anchor_b)

    numeric = {
        "pivot_range_pts": float(pivot_range),
        "pivot_body_pts": float(pivot_body),
        "upper_wick_pct": float(upper_wick),
        "lower_wick_pct": float(lower_wick),
        "pivot_height_pts": float(pivot_height),
        "atr14_pts": float(atr14_pts),
        "range_30bar_pts": float(range_30bar_pts),
        "trend_20bar_pct": float(trend_20bar_pct),
        "dist_to_20bar_hi_pct": float(dist_to_20bar_hi_pct),
        "dist_to_20bar_lo_pct": float(dist_to_20bar_lo_pct),
        "dist_pivot_to_bank_pts": float(dist_pivot_to_bank),
        "anchor_distance_from_entry_pts": float(anchor_distance_from_entry_pts),
        "vel_5bar_pts_per_min": float(vel_5bar_pts_per_min),
        "vel_20bar_pts_per_min": float(vel_20bar_pts_per_min),
        "reading_b_buffer_pts": float(reading_b_buffer),
    }
    categorical = {
        "pivot_type": ptype,
        "session": str(session),
        "tape": str(tape),
    }
    ordinal = {"et_hour": float(et_hour)}
    X = _build_row(_PIVOT_PAYLOAD, numeric, categorical, ordinal)
    try:
        probs = _PIVOT_PAYLOAD["model"].predict_proba(X)[0]
        classes = list(_PIVOT_PAYLOAD["model"].classes_)
        if 1 in classes:
            p_hold = float(probs[classes.index(1)])
        else:
            p_hold = 0.5
    except Exception:
        return None
    thr = float(_PIVOT_PAYLOAD.get("hold_threshold", 0.55))
    return p_hold, (p_hold >= thr)


def score_kalshi(
    signal: Dict[str, Any],
    bar_features: Dict[str, float],
    *,
    regime: str,
    et_hour_frac: float,
    role: str,
) -> Optional[Tuple[float, float, bool]]:
    """Return (p_win, predicted_pnl, should_pass) for a Kalshi entry decision,
    or None if model not loaded.

    Expected in signal dict (set by build_kalshi_trade_plan in julie001.py):
      kalshi_entry_probability, kalshi_probe_probability,
      kalshi_momentum_delta, kalshi_momentum_retention,
      kalshi_entry_support_score, kalshi_entry_threshold,
      kalshi_probe_price, entry_price, side, sub_strategy.

    Expected in bar_features (computed at signal time):
      atr14_pts, range_30bar_pts, trend_20bar_pct, dist_to_20bar_hi_pct,
      dist_to_20bar_lo_pct, vel_5bar_pts_per_min, dist_to_bank_pts,
      regime_vol_bp, regime_eff.
    """
    if _KALSHI_PAYLOAD is None:
        return None
    try:
        ep = float(signal.get("entry_price") or 0.0)
        pp = float(signal.get("kalshi_probe_price") or 0.0)
    except Exception:
        return None
    # Parse DE3 sub_strategy tier/rev/timeframe (mirror train_kalshi_ml parser)
    sub = str(signal.get("sub_strategy", "") or "")
    import re
    m = re.search(r"(?P<tf>5min|15min)_.*?_(?P<direction>Long|Short)_(?P<type>Rev|Mom)_T(?P<tier>\d)", sub)
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
        # Kalshi features from signal
        "entry_probability": float(signal.get("kalshi_entry_probability") or 0.5),
        "probe_probability": float(signal.get("kalshi_probe_probability") or 0.5),
        "momentum_delta": float(signal.get("kalshi_momentum_delta") or 0.0),
        "momentum_retention": float(signal.get("kalshi_momentum_retention") or 1.0),
        "support_score": float(signal.get("kalshi_entry_support_score") or 0.5),
        "threshold": float(signal.get("kalshi_entry_threshold") or 0.45),
        "probe_distance_pts": pp - ep,
        "et_hour_frac": float(et_hour_frac),
        # Substrategy-derived
        "sub_tier": float(tier),
        "sub_is_rev": 1.0 if is_rev else 0.0,
        "sub_is_5min": 1.0 if is_5m else 0.0,
        # Market state from bar_features dict
        "atr14_pts": float(bar_features.get("atr14_pts", 0.0) or 0.0),
        "range_30bar_pts": float(bar_features.get("range_30bar_pts", 0.0) or 0.0),
        "trend_20bar_pct": float(bar_features.get("trend_20bar_pct", 0.0) or 0.0),
        "dist_to_20bar_hi_pct": float(bar_features.get("dist_to_20bar_hi_pct", 0.0) or 0.0),
        "dist_to_20bar_lo_pct": float(bar_features.get("dist_to_20bar_lo_pct", 0.0) or 0.0),
        "vel_5bar_pts_per_min": float(bar_features.get("vel_5bar_pts_per_min", 0.0) or 0.0),
        "dist_to_bank_pts": float(bar_features.get("dist_to_bank_pts", 0.0) or 0.0),
        "regime_vol_bp": float(bar_features.get("regime_vol_bp", 0.0) or 0.0),
        "regime_eff": float(bar_features.get("regime_eff", 0.0) or 0.0),
    }
    categorical = {
        "side": str(signal.get("side", "")).upper(),
        "role": str(role or "unknown"),
        "regime": str(regime or "neutral"),
    }
    X = _build_row(_KALSHI_PAYLOAD, numeric, categorical, {})
    try:
        clf = _KALSHI_PAYLOAD["classifier"]
        reg = _KALSHI_PAYLOAD["regressor"]
        probs = clf.predict_proba(X)[0]
        classes = list(clf.classes_)
        p_win = float(probs[classes.index(1)]) if 1 in classes else 0.5
        pred_pnl = float(reg.predict(X)[0])
    except Exception:
        return None
    thr = float(_KALSHI_PAYLOAD.get("pass_threshold", 0.50))
    return p_win, pred_pnl, (p_win >= thr)


def is_lfo_live_active() -> bool:
    return os.environ.get("JULIE_ML_LFO_ACTIVE", "0").strip() == "1"


def is_pct_live_active() -> bool:
    return os.environ.get("JULIE_ML_PCT_ACTIVE", "0").strip() == "1"


def score_kalshi_tp(
    signal: Dict[str, Any],
    *,
    tp_aligned_prob: float,
    entry_aligned_prob: float,
    nearest_strike_dist: float,
    nearest_strike_oi: float,
    nearest_strike_volume: float,
    ladder_slope_near_tp: float,
    minutes_to_settlement: float,
    atr14_pts: float,
    range_30bar_pts: float,
    trend_20bar_pct: float,
    vel_5bar_pts_per_min: float,
    regime: str,
    tp_dist_pts: float,
) -> Optional[Tuple[float, bool]]:
    """Score the TP-aligned Kalshi gate.

    Return (p_hit_tp, should_pass) where p_hit_tp is the model's estimate
    that the trade will reach its take-profit, and should_pass is
    p_hit_tp >= the payload's pass_threshold (default 0.50).

    Expected caller: julie001.py after build_trade_plan has computed the
    tp_aligned probability; the caller joins those Kalshi readings with
    bar-derived market state and a regime label.
    """
    if _KALSHI_TP_PAYLOAD is None:
        return None
    # Parse DE3 sub_strategy for tier / is_rev / is_5min (mirrors trainer)
    sub = str(signal.get("sub_strategy", "") or "")
    import re
    m = re.search(
        r"(?P<tf>5min|15min)_.*?_(?P<direction>Long|Short)_(?P<type>Rev|Mom)_T(?P<tier>\d)", sub
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
        "tp_aligned_prob": float(tp_aligned_prob),
        "tp_dist_pts": float(tp_dist_pts),
        "tp_prob_edge": float(tp_aligned_prob - 0.50),
        "tp_vs_entry_prob_delta": float(tp_aligned_prob - entry_aligned_prob),
        "nearest_strike_dist": float(nearest_strike_dist),
        "nearest_strike_oi": float(nearest_strike_oi),
        "nearest_strike_volume": float(nearest_strike_volume),
        "ladder_slope_near_tp": float(ladder_slope_near_tp),
        "minutes_to_settlement": float(minutes_to_settlement),
        "entry_aligned_prob": float(entry_aligned_prob),
        "atr14_pts": float(atr14_pts),
        "range_30bar_pts": float(range_30bar_pts),
        "trend_20bar_pct": float(trend_20bar_pct),
        "vel_5bar_pts_per_min": float(vel_5bar_pts_per_min),
        "sub_tier": float(tier),
        "sub_is_rev": 1.0 if is_rev else 0.0,
        "sub_is_5min": 1.0 if is_5m else 0.0,
    }
    categorical = {
        "side": str(signal.get("side", "")).upper(),
        "regime": str(regime or "neutral"),
    }
    X = _build_row(_KALSHI_TP_PAYLOAD, numeric, categorical, {})
    try:
        clf = _KALSHI_TP_PAYLOAD["classifier"]
        probs = clf.predict_proba(X)[0]
        classes = list(clf.classes_)
        p_hit = float(probs[classes.index(1)]) if 1 in classes else 0.5
    except Exception:
        return None
    thr = float(_KALSHI_TP_PAYLOAD.get("pass_threshold", 0.50))
    return p_hit, (p_hit >= thr)


def is_pivot_trail_live_active() -> bool:
    return os.environ.get("JULIE_ML_PIVOT_TRAIL_ACTIVE", "0").strip() == "1"


def is_kalshi_live_active() -> bool:
    return os.environ.get("JULIE_ML_KALSHI_ACTIVE", "0").strip() == "1"


def is_kalshi_tp_live_active() -> bool:
    return os.environ.get("JULIE_ML_KALSHI_TP_ACTIVE", "0").strip() == "1"
