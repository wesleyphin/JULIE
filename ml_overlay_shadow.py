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


def init_ml_overlays() -> Tuple[bool, bool]:
    """Eager-load both models. Returns (lfo_loaded, pct_loaded)."""
    global _LFO_PAYLOAD, _PCT_PAYLOAD
    _LFO_PAYLOAD = _try_load("model_lfo.joblib")
    _PCT_PAYLOAD = _try_load("model_pct_overlay.joblib")
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
    return _LFO_PAYLOAD is not None, _PCT_PAYLOAD is not None


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


def is_lfo_live_active() -> bool:
    return os.environ.get("JULIE_ML_LFO_ACTIVE", "0").strip() == "1"


def is_pct_live_active() -> bool:
    return os.environ.get("JULIE_ML_PCT_ACTIVE", "0").strip() == "1"
