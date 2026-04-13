import hashlib
import json
import logging
import os
import re
import time
from collections import Counter, deque
from pathlib import Path
from typing import Callable, Deque, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import CONFIG
from dynamic_sltp_params import dynamic_sltp_engine
import ml_physics_pipeline as mlp
from session_manager import SessionManager
from strategy_base import Strategy
from volatility_filter import volatility_filter, VolRegime
from incremental_ohlcv_resampler import IncrementalOHLCVResampler

try:
    # Preferred: package-level re-exports from dist_bracket_ml/__init__.py
    from dist_bracket_ml import (
        load_bundle as _dist_load_bundle,
        predict as _dist_predict,
        prepare_runtime_features as _dist_prepare_runtime_features,
        predict_from_feature_row as _dist_predict_from_feature_row,
    )
except Exception:
    try:
        # Fallback: direct import from nested module package layout.
        from dist_bracket_ml.dist_bracket_ml import (
            load_bundle as _dist_load_bundle,
            predict as _dist_predict,
            prepare_runtime_features as _dist_prepare_runtime_features,
            predict_from_feature_row as _dist_predict_from_feature_row,
        )
    except Exception:
        _dist_load_bundle = None
        _dist_predict = None
        _dist_prepare_runtime_features = None
        _dist_predict_from_feature_row = None

# ==========================================
# ML FEATURE PIPELINE (Synced with ml_train_v11.py)
# ==========================================
# These constants & helpers mirror the institutional ML training script.
# They are used to build the exact same feature vector at runtime.

RSI_PERIOD = 14      # Smoother for 1m noise
ADX_PERIOD = 20      # More stable trend strength
ATR_PERIOD = 20      # Robust volatility context
ZSCORE_WINDOW = 100  # Longer normalization window for 1m
RVOL_LOOKBACK_DAYS = 20
SLOPE_WINDOW = 30
VOLATILITY_WINDOW = 30
RSI_VEL_LAG = 5
ADX_VEL_LAG = 5
RETURN_FAST = 1
RETURN_MED = 10
RETURN_SLOW = 30
VWAP_CROSS_WINDOW = 30
VWAP_PERSIST_CAP = 300
TREND_EMA_FAST = 20
TREND_EMA_SLOW = 50
TREND_PERSIST_CAP = 300
ATR_SLOPE_LAG = 10
SESSION_OPEN_MINUTES = 60
_MACRO_REGIMES_CACHE = None
_DIST_CACHE_SESSION_CATEGORIES = ["UNKNOWN", "ASIA", "LONDON", "NY_AM", "NY_PM", "OTHER", "CLOSED"]
_DIST_CACHE_DECISION_CATEGORIES = ["unknown", "no_signal", "blocked", "signal_long", "signal_short"]
_DIST_CACHE_REGIME_CATEGORIES = ["unknown", "low", "normal", "high"]


def _load_macro_regimes():
    """Load macro regimes from regimes.json (optional)."""
    global _MACRO_REGIMES_CACHE
    if _MACRO_REGIMES_CACHE is not None:
        return _MACRO_REGIMES_CACHE
    path = CONFIG.get("ML_PHYSICS_REGIMES_FILE") or "regimes.json"
    try:
        import json
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        _MACRO_REGIMES_CACHE = []
        return _MACRO_REGIMES_CACHE
    regimes = data.get("regimes") if isinstance(data, dict) else None
    parsed = []
    try:
        for idx, reg in enumerate(regimes or []):
            name = str(reg.get("name", f"regime_{idx}"))
            start = pd.to_datetime(reg.get("start"))
            end = pd.to_datetime(reg.get("end"))
            if pd.isna(start) or pd.isna(end):
                continue
            parsed.append((name, start.normalize(), end.normalize(), idx + 1))
    except Exception:
        parsed = []
    _MACRO_REGIMES_CACHE = parsed
    return _MACRO_REGIMES_CACHE


def _map_macro_regime_id(dt: pd.Series) -> np.ndarray:
    if not CONFIG.get("ML_PHYSICS_USE_MACRO_REGIME", False):
        return np.zeros(len(dt), dtype=np.int8)
    regimes = _load_macro_regimes()
    if not regimes:
        return np.zeros(len(dt), dtype=np.int8)
    dates = pd.to_datetime(dt).dt.normalize()
    out = np.zeros(len(dates), dtype=np.int8)
    for _, start, end, rid in regimes:
        mask = (dates >= start) & (dates <= end)
        out[mask.to_numpy()] = np.int8(rid)
    return out


def _extract_thresholds(
    threshold_value,
    regime_key: Optional[str],
    fallback: float,
    context: Optional[dict] = None,
) -> Tuple[float, float, Optional[str]]:
    """Normalize threshold config into long/short thresholds with optional context + fallback."""
    policy = None
    short_threshold = None

    base_value = threshold_value
    if isinstance(base_value, dict) and any(k in base_value for k in ("low", "normal", "high")):
        key = regime_key or "low"
        base_value = base_value.get(
            key,
            base_value.get("low", base_value.get("high", fallback)),
        )

    # Hierarchical threshold fallback: session+regime+structure -> session+regime -> base session.
    if isinstance(base_value, dict):
        hierarchy = base_value.get("hierarchy") or {}
        if not hierarchy and isinstance(threshold_value, dict):
            hierarchy = threshold_value.get("hierarchy") or {}
        if context and isinstance(hierarchy, dict):
            regime_ctx = str(context.get("regime_key") or regime_key or "low")
            try:
                atr_ctx = int(context.get("atr_state"))
            except Exception:
                atr_ctx = None
            try:
                trend_ctx = int(context.get("trend_state"))
            except Exception:
                trend_ctx = None
            try:
                liq_ctx = int(context.get("liquidity_state"))
            except Exception:
                liq_ctx = None
            structure_key = None
            if atr_ctx is not None and trend_ctx is not None and liq_ctx is not None:
                structure_key = f"{regime_ctx}|a{atr_ctx}|t{trend_ctx}|l{liq_ctx}"

            bucket_full = hierarchy.get("session_regime_structure") or {}
            bucket_regime = hierarchy.get("session_regime") or {}
            if structure_key and structure_key in bucket_full:
                base_value = bucket_full.get(structure_key, base_value)
            elif regime_ctx in bucket_regime:
                base_value = bucket_regime.get(regime_ctx, base_value)

    fallback_value = None
    if isinstance(threshold_value, dict):
        fallback_value = threshold_value.get("fallback")

    # Context thresholds (e.g. atr_state)
    ctx_container = None
    if isinstance(threshold_value, dict):
        ctx_container = threshold_value.get("context") or threshold_value.get("context_thresholds")
    if isinstance(base_value, dict):
        ctx_container = base_value.get("context") or ctx_container

    if context and isinstance(ctx_container, dict):
        atr_map = ctx_container.get("atr_state") or ctx_container.get("ATR_State") or {}
        try:
            atr_key = str(int(context.get("atr_state")))
        except Exception:
            atr_key = None
        if atr_key is not None and atr_key in atr_map:
            base_value = atr_map.get(atr_key, base_value)

    if isinstance(base_value, dict):
        policy = base_value.get("policy")
        try:
            req = float(base_value.get("threshold", fallback))
        except Exception:
            req = float(fallback)
        short_threshold = base_value.get("short_threshold")
    else:
        req = float(base_value)

    if short_threshold is None:
        short_req = 1.0 - req
    else:
        try:
            short_req = float(short_threshold)
        except Exception:
            short_req = 1.0 - req
    return req, short_req, policy


def _extract_gate_threshold(
    threshold_value,
    regime_key: Optional[str],
    context: Optional[dict] = None,
) -> Optional[float]:
    base_value = threshold_value
    if isinstance(base_value, dict) and any(k in base_value for k in ("low", "normal", "high")):
        key = regime_key or "low"
        base_value = base_value.get(key, base_value.get("low", base_value.get("high")))

    if isinstance(base_value, dict):
        hierarchy = base_value.get("hierarchy") or {}
        if context and isinstance(hierarchy, dict):
            regime_ctx = str(context.get("regime_key") or regime_key or "low")
            try:
                atr_ctx = int(context.get("atr_state"))
                trend_ctx = int(context.get("trend_state"))
                liq_ctx = int(context.get("liquidity_state"))
                structure_key = f"{regime_ctx}|a{atr_ctx}|t{trend_ctx}|l{liq_ctx}"
            except Exception:
                structure_key = None
            bucket_full = hierarchy.get("session_regime_structure") or {}
            bucket_regime = hierarchy.get("session_regime") or {}
            chosen = None
            if structure_key and structure_key in bucket_full:
                chosen = bucket_full.get(structure_key)
            elif regime_ctx in bucket_regime:
                chosen = bucket_regime.get(regime_ctx)
            if isinstance(chosen, dict):
                gt = chosen.get("gate_threshold")
                if gt is not None:
                    try:
                        return float(gt)
                    except Exception:
                        pass
        gt = base_value.get("gate_threshold")
        if gt is None and isinstance(base_value.get("gate"), dict):
            gt = base_value.get("gate", {}).get("threshold")
        if gt is not None:
            try:
                return float(gt)
            except Exception:
                return None
    return None


def _extract_ev_runtime(
    threshold_value,
    regime_key: Optional[str],
    context: Optional[dict] = None,
) -> Dict:
    def _select_regime_bucket(value):
        selected = value
        if isinstance(selected, dict) and any(k in selected for k in ("low", "normal", "high")):
            key = regime_key or "low"
            selected = selected.get(key, selected.get("low", selected.get("high")))
        return selected

    selected_value = _select_regime_bucket(threshold_value)
    selected_hierarchy = None
    if isinstance(selected_value, dict):
        hierarchy = selected_value.get("hierarchy") or {}
        if context and isinstance(hierarchy, dict):
            regime_ctx = str(context.get("regime_key") or regime_key or "low")
            try:
                atr_ctx = int(context.get("atr_state"))
                trend_ctx = int(context.get("trend_state"))
                liq_ctx = int(context.get("liquidity_state"))
                structure_key = f"{regime_ctx}|a{atr_ctx}|t{trend_ctx}|l{liq_ctx}"
            except Exception:
                structure_key = None
            bucket_full = hierarchy.get("session_regime_structure") or {}
            bucket_regime = hierarchy.get("session_regime") or {}
            if structure_key and structure_key in bucket_full:
                selected_hierarchy = bucket_full.get(structure_key)
            elif regime_ctx in bucket_regime:
                selected_hierarchy = bucket_regime.get(regime_ctx)

    merged: Dict = {}

    def _merge_ev(candidate):
        if not isinstance(candidate, dict):
            return
        raw = (
            candidate.get("ev_runtime")
            or candidate.get("ev_decision")
            or candidate.get("ev")
        )
        if isinstance(raw, dict):
            merged.update(raw)

    _merge_ev(threshold_value)
    _merge_ev(selected_value)
    _merge_ev(selected_hierarchy)
    return merged


def _safe_prob_bound(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return min(max(out, 0.0), 0.999)


def _resolve_gate_threshold_bounds(
    session_name: Optional[str],
    regime_key: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float]]:
    cfg = CONFIG.get("ML_PHYSICS_GATE_HARD_LIMITS", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return None, None

    min_thr: Optional[float] = None
    max_thr: Optional[float] = None

    def _apply(src):
        nonlocal min_thr, max_thr
        if not isinstance(src, dict):
            return
        if "min" in src:
            min_thr = _safe_prob_bound(src.get("min"), min_thr)
        if "max" in src:
            max_thr = _safe_prob_bound(src.get("max"), max_thr)

    _apply(cfg.get("default"))

    sess_map = cfg.get("sessions", {}) or {}
    sess_cfg = None
    if isinstance(sess_map, dict) and session_name:
        s_up = str(session_name).upper()
        sess_cfg = sess_map.get(s_up)
        if sess_cfg is None:
            sess_cfg = sess_map.get(str(session_name))
    _apply(sess_cfg)

    if isinstance(sess_cfg, dict):
        reg_map = sess_cfg.get("regimes")
        if isinstance(reg_map, dict):
            r_l = str(regime_key or "").lower()
            reg_cfg = reg_map.get(r_l)
            if reg_cfg is None and regime_key is not None:
                reg_cfg = reg_map.get(str(regime_key))
            _apply(reg_cfg)

    if min_thr is not None and max_thr is not None and min_thr > max_thr:
        min_thr, max_thr = max_thr, min_thr
    return min_thr, max_thr


ML_FEATURE_COLUMNS = [
    'Close_ZScore',
    'High_ZScore',
    'Low_ZScore',
    'ATR_ZScore',
    'Volatility_ZScore',
    'Range_ZScore',
    'Slope_ZScore',
    'RVol_ZScore',
    'RSI_Vel',
    'Adx_Vel',
    'RSI_Norm',
    'ADX_Norm',
    'Return_1',
    'Return_5',
    'Return_15',
    'Hour_Sin',
    'Hour_Cos',
    'Minute_Sin',
    'Minute_Cos',
    'DOW_Sin',
    'DOW_Cos',
    'Is_Trending',
    'Trend_Direction',
    'High_Volatility',
    'VWAP_Dist_ATR',
    'VWAP_Crosses_30',
    'VWAP_Bars_Since_Cross',
    'VWAP_Behavior',
    'Range_Overlap_Ratio',
    'Gap_Size_ATR',
    'Gap_Filled',
    'ATR_Slope',
    'Trend_Persistence',
    'Session_Open_Range_ATR',
    'Session_Open_Drive',
    'Session_ID',
    'ATR_State',
    'Trend_State',
    'Liquidity_State',
    'Macro_Regime_ID',
]

OPTIONAL_FEATURE_COLUMNS = {
    'RVol_ZScore',
}


def ml_calculate_rsi(series, period=RSI_PERIOD):
    """RSI calculation (same as ml_train_v11.calculate_rsi)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def ml_calculate_atr(high, low, close, period=ATR_PERIOD):
    """ATR for volatility-adjusted features (same as ml_train_v11.calculate_atr)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def ml_calculate_adx(high, low, close, period=ADX_PERIOD):
    """ADX + DI lines for regime detection (same as ml_train_v11.calculate_adx)."""
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed values
    atr = tr.rolling(window=period).mean()
    plus_series = pd.Series(plus_dm, index=high.index)
    minus_series = pd.Series(minus_dm, index=high.index)
    plus_di = 100 * plus_series.rolling(window=period).mean() / atr
    minus_di = 100 * minus_series.rolling(window=period).mean() / atr

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di


def ml_calculate_slope(series, window=15):
    """Rolling linear regression slope (same as ml_train_v11.calculate_slope)."""
    def slope_calc(y):
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y))
        covariance = np.cov(x, y)[0, 1]
        variance = np.var(x)
        return float(covariance / variance) if variance != 0 else 0.0
    return series.rolling(window=window).apply(slope_calc, raw=False)


def ml_zscore_normalize(series, window=ZSCORE_WINDOW):
    """Z-score normalization (same as ml_train_v11.zscore_normalize)."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    zscore = (series - rolling_mean) / (rolling_std + 1e-10)
    return zscore


def ml_calculate_rvol(volume, datetime_index, lookback_days=RVOL_LOOKBACK_DAYS):
    """Relative Volume: current vol / avg vol at this time of day (ml_train_v11.calculate_rvol)."""
    df_temp = pd.DataFrame({'volume': volume, 'datetime': datetime_index})
    df_temp['hour'] = df_temp['datetime'].dt.hour
    df_temp['minute'] = df_temp['datetime'].dt.minute
    df_temp['time_bucket'] = df_temp['hour'] * 60 + df_temp['minute']

    # Each time_bucket has ~1 observation per trading day (per bar). The rolling window should
    # therefore be in "days", not multiplied by bars-per-day.
    window = max(int(lookback_days), 1)
    min_periods = min(10, window)
    avg_vol_by_time = df_temp.groupby('time_bucket')['volume'].transform(
        lambda x: x.rolling(window=window, min_periods=min_periods).mean()
    )
    rvol = volume / (avg_vol_by_time + 1)
    return rvol


def ml_encode_cyclical_time(datetime_series):
    """Cyclical time encoding using sin/cos (same as ml_train_v11.encode_cyclical_time)."""
    hour = datetime_series.dt.hour
    minute = datetime_series.dt.minute
    day_of_week = datetime_series.dt.dayofweek

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    dow_sin = np.sin(2 * np.pi * day_of_week / 5)
    dow_cos = np.cos(2 * np.pi * day_of_week / 5)

    return hour_sin, hour_cos, minute_sin, minute_cos, dow_sin, dow_cos


class MLPhysicsStrategy(Strategy):
    """
    Session-specialized ML strategy using four neural networks.
    Mirrors juliemlsession.py behavior with identical feature pipeline.
    """

    def __init__(self):
        self._legacy_models_detached = bool(CONFIG.get("ML_PHYSICS_REPLACE_WITH_DIST", True))
        self._warned_no_legacy_fallback = False
        if self._legacy_models_detached:
            self.sm = None
            logging.info("MLPhysics: legacy SessionManager models detached (dist-only mode)")
        else:
            self.sm = SessionManager()
        self.window_size = CONFIG.get("WINDOW_SIZE", 15)
        self.model_loaded = any(self.sm.brains.values()) if self.sm is not None else False
        self._dist_bundle = None
        self._dist_mode = False
        self._dist_mode_reason = ""
        try:
            self._dist_input_max_bars = int(
                CONFIG.get(
                    "ML_PHYSICS_DIST_MAX_BARS",
                    CONFIG.get("BACKTEST_ML_DIST_INPUT_BARS", 3000),
                )
                or 0
            )
        except Exception:
            self._dist_input_max_bars = 0
        if self._dist_input_max_bars < 0:
            self._dist_input_max_bars = 0
        self._dist_eval_total = 0
        self._dist_eval_signals = 0
        self._dist_eval_blocked = 0
        self._dist_reason_counts: Counter[str] = Counter()
        self._dist_session_counts: Counter[str] = Counter()
        self._dist_log_every = int(CONFIG.get("ML_PHYSICS_DIST_LOG_EVERY_EVALS", 10000) or 10000)
        if self._dist_log_every < 0:
            self._dist_log_every = 0
        self._last_feature_log_ts = 0.0
        self._feature_log_interval = 300.0
        self._logged_resample = False
        self._tf_resamplers: Dict[int, IncrementalOHLCVResampler] = {}
        self.last_eval = None
        self._opt_cfg = CONFIG.get("ML_PHYSICS_OPT", {}) or {}
        self._opt_enabled = bool(self._opt_cfg.get("enabled", False))
        self._opt_mode = str(self._opt_cfg.get("mode", "backtest") or "backtest").lower()
        self._dist_lazy_backtest = bool(CONFIG.get("ML_PHYSICS_DIST_LAZY_LOAD_BACKTEST", True))
        self._opt_precomputed_df: Optional[pd.DataFrame] = None
        self._opt_dist_precomputed_df: Optional[pd.DataFrame] = None
        self._opt_dist_precompute_mode = False
        self._opt_eval_ts: Optional[pd.Timestamp] = None
        self._opt_eval_row: Optional[object] = None
        self._opt_precomputed_index_ns = np.empty(0, dtype=np.int64)
        self._opt_precomputed_can_eval = np.empty(0, dtype=np.int8)
        self._opt_precomputed_feature_matrix = np.empty((0, len(ML_FEATURE_COLUMNS)), dtype=np.float32)
        self._opt_precomputed_session: list[str] = []
        self._opt_precomputed_prob_up = np.empty(0, dtype=np.float32)
        self._opt_precomputed_cursor_pos = 0
        self._opt_dist_index_ns = np.empty(0, dtype=np.int64)
        self._opt_dist_has_signal_arr = np.empty(0, dtype=np.int8)
        self._opt_dist_signal_side_arr = np.empty(0, dtype=np.int8)
        self._opt_dist_signal_tp_dist_arr = np.empty(0, dtype=np.float32)
        self._opt_dist_signal_sl_dist_arr = np.empty(0, dtype=np.float32)
        self._opt_dist_signal_confidence_arr = np.empty(0, dtype=np.float32)
        self._opt_dist_signal_confidence_raw_arr = np.empty(0, dtype=np.float32)
        self._opt_dist_signal_ev_pred_arr = np.empty(0, dtype=np.float32)
        self._opt_dist_signal_ev_min_req_arr = np.empty(0, dtype=np.float32)
        self._opt_dist_eval_decision_codes = np.empty(0, dtype=np.int8)
        self._opt_dist_eval_session_codes = np.empty(0, dtype=np.int8)
        self._opt_dist_eval_regime_codes = np.empty(0, dtype=np.int8)
        self._opt_dist_eval_decision_categories = list(_DIST_CACHE_DECISION_CATEGORIES)
        self._opt_dist_eval_session_categories = list(_DIST_CACHE_SESSION_CATEGORIES)
        self._opt_dist_eval_regime_categories = list(_DIST_CACHE_REGIME_CATEGORIES)
        self._opt_dist_eval_blocked_reason_codes = np.empty(0, dtype=np.int32)
        self._opt_dist_eval_blocked_reason_categories: list[str] = []
        self._opt_dist_eval_gate_prob_arr = np.empty(0, dtype=np.float32)
        self._opt_dist_eval_gate_threshold_arr = np.empty(0, dtype=np.float32)
        self._opt_dist_eval_gate_margin_min_arr = np.empty(0, dtype=np.float32)
        self._opt_dist_signal_cache: list[Optional[Dict]] = []
        self._opt_dist_eval_cache: list[Optional[Dict]] = []
        self._opt_dist_cursor_pos = 0
        self._trade_budget_cfg = CONFIG.get("ML_PHYSICS_TRADE_BUDGET_LIVE", {}) or {}
        self._trade_budget_state: Dict[str, Dict[str, Deque[int]]] = {}
        self._ev_cfg = CONFIG.get("ML_PHYSICS_EV_DECISION", {}) or {}
        self._ev_cfg_session_overrides = CONFIG.get("ML_PHYSICS_EV_DECISION_SESSION_OVERRIDES", {}) or {}
        hyst_cfg = CONFIG.get("ML_PHYSICS_HYSTERESIS", {}) or {}
        self._hyst_enabled = bool(hyst_cfg.get("enabled", True))
        self._hyst_entry_margin = float(hyst_cfg.get("entry_margin", 0.0) or 0.0)
        self._hyst_exit_margin = float(hyst_cfg.get("exit_margin", 0.02) or 0.02)
        self._hyst_flip_margin = float(hyst_cfg.get("flip_margin", 0.01) or 0.01)
        self._hyst_retrigger_delta = float(hyst_cfg.get("retrigger_delta", 0.0) or 0.0)
        for field in (
            "_hyst_entry_margin",
            "_hyst_exit_margin",
            "_hyst_flip_margin",
            "_hyst_retrigger_delta",
        ):
            val = getattr(self, field, 0.0)
            if not np.isfinite(val):
                val = 0.0
            setattr(self, field, max(0.0, float(val)))
        self._hyst_state: Dict[str, Dict[str, float | str | None]] = {}
        risk_cfg = CONFIG.get("RISK", {}) or {}
        point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
        fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.5) or 2.5)
        self._roundtrip_fees_pts = (fees_per_side * 2.0) / point_value if point_value > 0 else 0.0
        self._dist_run_dir: Optional[Path] = None
        self._dist_session_hours_cache: Optional[dict] = None
        self._dist_session_hour_lookup_cache: Optional[tuple[str, ...]] = None

        self._init_dist_replacement()
        if self._dist_mode:
            self.model_loaded = True

    def _opt_backtest_active(self) -> bool:
        return bool(
            self._opt_enabled
            and self._opt_mode == "backtest"
            and self._opt_precomputed_df is not None
            and not self._opt_precomputed_df.empty
        )

    def _opt_dist_backtest_active(self) -> bool:
        return bool(
            self._opt_enabled
            and self._opt_mode == "backtest"
            and self._dist_mode
            and self._opt_dist_precomputed_df is not None
            and not self._opt_dist_precomputed_df.empty
        )

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return str(obj)

    @classmethod
    def _json_dumps_safe(cls, payload: Optional[Dict]) -> str:
        if not payload:
            return ""
        try:
            return json.dumps(payload, ensure_ascii=True, default=cls._json_default)
        except Exception:
            try:
                return json.dumps({k: str(v) for k, v in dict(payload).items()}, ensure_ascii=True)
            except Exception:
                return ""

    @staticmethod
    def _json_load_safe(text: str, default):
        if not text:
            return default
        try:
            out = json.loads(text)
            return out if out is not None else default
        except Exception:
            return default

    @staticmethod
    def _sanitize_token(value: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
        return safe.strip("_") or "unknown"

    def _dist_bundle_run_signature(self) -> str:
        run_dir_obj = getattr(self._dist_bundle, "run_dir", None)
        run_dir = Path(run_dir_obj) if run_dir_obj is not None else self._dist_run_dir
        if run_dir is None:
            return "dist_unknown"
        try:
            run_dir = Path(run_dir).expanduser().resolve()
            artifact_index = run_dir / "artifact_index.json"
            if artifact_index.exists() and artifact_index.is_file():
                try:
                    data = artifact_index.read_bytes()
                    return hashlib.sha1(data).hexdigest()[:10]
                except Exception:
                    pass
            return hashlib.sha1(str(run_dir).encode("utf-8")).hexdigest()[:10]
        except Exception:
            return "dist_unknown"

    def _dist_runtime_cache_signature(self) -> str:
        keys = (
            "ML_PHYSICS_EV_DECISION",
            "ML_PHYSICS_EV_DECISION_SESSION_OVERRIDES",
            "ML_PHYSICS_GATE_HARD_LIMITS",
            "ML_PHYSICS_DIST_GATE_THRESHOLD_CLAMP",
            "ML_PHYSICS_DIST_MIN_GATE_MARGIN",
            "ML_PHYSICS_DIST_RUNTIME_MIN_RR",
            "ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER",
            "ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS",
            "ML_PHYSICS_DIST_WIDE_BRACKET_RUNNER",
            "ML_PHYSICS_DIST_NORMAL_PROFILE",
            "ML_PHYSICS_DIST_NORMAL_BRACKET_POLICY",
            "ML_PHYSICS_HYSTERESIS",
            "ML_PHYSICS_DISABLED_REGIMES",
        )
        payload = {key: CONFIG.get(key) for key in keys}
        try:
            text = json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=True,
                default=self._json_default,
            )
        except Exception:
            text = self._json_dumps_safe(payload) or "runtime_unknown"
        return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:10]

    @staticmethod
    def _df_attr_text(df: Optional[pd.DataFrame], key: str) -> str:
        if df is None:
            return ""
        try:
            attrs = getattr(df, "attrs", {}) or {}
        except Exception:
            return ""
        if not isinstance(attrs, dict):
            return ""
        value = attrs.get(key)
        text = str(value or "").strip()
        return text

    def _dist_symbol_hint_from_df(self, df: Optional[pd.DataFrame]) -> str:
        for key in ("mlphysics_dist_symbol_hint", "selected_symbol", "backtest_selected_symbol"):
            text = self._df_attr_text(df, key)
            if text:
                return text
        return ""

    def _dist_symbol_mode_hint_from_df(self, df: Optional[pd.DataFrame]) -> str:
        for key in ("mlphysics_dist_symbol_mode", "selected_symbol_mode", "backtest_selected_symbol_mode"):
            text = self._df_attr_text(df, key).lower()
            if text:
                return text
        return ""

    def _dist_source_key_from_df(self, df: Optional[pd.DataFrame]) -> str:
        for key in ("mlphysics_dist_source_key", "source_cache_key", "backtest_source_cache_key"):
            text = self._df_attr_text(df, key)
            if text:
                return self._sanitize_token(text)
        return ""

    def _dist_source_label_from_df(self, df: Optional[pd.DataFrame]) -> str:
        for key in ("mlphysics_dist_source_label", "source_label", "backtest_source_label"):
            text = self._df_attr_text(df, key)
            if text:
                return text
        return ""

    def _dist_cache_components(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Path, str, str, str, int, str, int, str, pd.DatetimeIndex]:
        cache_dir = Path(str(self._opt_cfg.get("feature_cache_dir", "cache/ml_physics/") or "cache/ml_physics/"))
        if not cache_dir.is_absolute():
            cache_dir = Path(__file__).resolve().parent / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("US/Eastern")
        else:
            idx = idx.tz_convert("US/Eastern")
        start = str(idx.min().date()) if len(idx) else "na"
        end = str(idx.max().date()) if len(idx) else "na"
        tf_minutes = int(CONFIG.get("ML_PHYSICS_TIMEFRAME_MINUTES", 1) or 1)
        align_mode = str(CONFIG.get("ML_PHYSICS_BAR_ALIGNMENT", "open") or "open").lower()
        symbol_tokens = self._dist_symbol_tokens(
            primary=self._dist_symbol_hint_from_df(df),
            symbol_mode=self._dist_symbol_mode_hint_from_df(df),
        )
        symbol = symbol_tokens[0] if symbol_tokens else self._sanitize_token("ES")
        max_bars = int(self._dist_input_max_bars)
        source_key = self._dist_source_key_from_df(df)
        return cache_dir, symbol, start, end, tf_minutes, align_mode, max_bars, source_key, idx

    def _dist_symbol_tokens(
        self,
        *,
        primary: Optional[str] = None,
        symbol_mode: Optional[str] = None,
    ) -> list[str]:
        tokens: list[str] = []

        def _add(raw: Optional[str]) -> None:
            if raw is None:
                return
            text = str(raw).strip()
            if not text:
                return
            token = self._sanitize_token(text)
            if token and token not in tokens:
                tokens.append(token)

        primary_raw = str(primary).strip() if primary is not None else ""
        mode_raw = str(symbol_mode or "").strip().lower()
        target_symbol = str(CONFIG.get("TARGET_SYMBOL") or "").strip()
        contract_root = str(CONFIG.get("CONTRACT_ROOT") or "").strip()
        auto_mode = mode_raw in {"auto", "auto_by_day", "roll"} or primary_raw.upper() == "AUTO_BY_DAY"

        if primary_raw:
            _add(primary_raw)
        if auto_mode:
            _add("AUTO_BY_DAY")
            return tokens or ["AUTO_BY_DAY"]
        if target_symbol:
            _add(target_symbol)
            # Also index by root token so caches survive contract roll token changes (e.g., MES.H26 -> MES).
            _add(target_symbol.split(".")[0])
        if contract_root:
            _add(contract_root)
        if not tokens:
            _add("ES")
        return tokens

    def _dist_explicit_cache_path(self) -> Optional[Path]:
        raw = str(self._opt_cfg.get("dist_precomputed_file", "") or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path

    def _dist_cache_candidates(self, df: pd.DataFrame, preferred: Path) -> list[Path]:
        cache_dir, symbol, start, end, tf_minutes, align_mode, max_bars, req_source_key, _ = self._dist_cache_components(df)
        req_cfg_sig = self._dist_runtime_cache_signature()
        symbol_tokens = self._dist_symbol_tokens(
            primary=self._dist_symbol_hint_from_df(df) or symbol,
            symbol_mode=self._dist_symbol_mode_hint_from_df(df),
        )
        symbol_tokens_set = set(symbol_tokens)
        root_tokens = {tok for tok in symbol_tokens if "." not in tok and tok}
        candidates: list[Path] = [preferred]
        fname_re = re.compile(
            r"^(?P<sym>.+?)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})"
            r"_dist_tf(?P<tf>\d+)_(?P<align>[^_]+)_max(?P<max>\d+)"
            r"(?:_src(?P<src>[A-Za-z0-9._-]+))?"
            r"(?:_cfg(?P<cfg>[A-Za-z0-9._-]+))?_.+_signals\.parquet$"
        )
        try:
            discovered: dict[Path, tuple[str, str, str, int, str, str, float]] = {}

            req_start = str(start)
            req_end = str(end)
            req_max = int(max_bars)
            req_tf = int(tf_minutes)
            req_align = str(self._sanitize_token(align_mode))

            for path in cache_dir.glob("*_signals.parquet"):
                if path == preferred:
                    continue
                m = fname_re.match(path.name)
                if not m:
                    continue
                cache_sym = str(m.group("sym"))
                sym_family_match = any(cache_sym.startswith(f"{root}.") for root in root_tokens)
                if cache_sym not in symbol_tokens_set and not sym_family_match:
                    continue
                try:
                    cache_tf = int(m.group("tf"))
                    cache_max = int(m.group("max"))
                except Exception:
                    continue
                cache_src = self._sanitize_token(str(m.group("src") or "").strip())
                cache_cfg = self._sanitize_token(str(m.group("cfg") or "").strip())
                cache_align = str(m.group("align"))
                if cache_tf != req_tf or cache_align != req_align:
                    continue
                if req_cfg_sig and cache_cfg != req_cfg_sig:
                    continue
                cache_start = str(m.group("start"))
                cache_end = str(m.group("end"))
                cache_exact = bool(cache_start == req_start and cache_end == req_end and cache_max == req_max)
                if req_source_key:
                    if cache_src:
                        if cache_src != req_source_key:
                            continue
                    elif not cache_exact:
                        # Legacy caches without a source key are only safe when the requested window matches exactly.
                        continue
                try:
                    mtime = float(path.stat().st_mtime)
                except Exception:
                    mtime = 0.0
                discovered[path] = (
                    cache_sym,
                    cache_start,
                    cache_end,
                    cache_max,
                    cache_src,
                    cache_cfg,
                    mtime,
                )

            def _rank(item: tuple[Path, tuple[str, str, str, int, str, str, float]]) -> tuple:
                # Higher tuple sorts first.
                # Priority: source match > exact > covering same-max > covering any-max > recency.
                path, (cache_sym, c_start, c_end, c_max, c_src, _c_cfg, mtime) = item
                exact = int(c_start == req_start and c_end == req_end and c_max == req_max)
                same_max = int(c_max == req_max)
                covers = int(c_start <= req_start and c_end >= req_end)
                start_match = int(c_start == req_start)
                symbol_match = int(cache_sym == symbol)
                source_match = int(bool(req_source_key) and c_src == req_source_key)
                legacy_exact = int(not c_src and exact)
                return (source_match, legacy_exact, exact, covers, same_max, start_match, symbol_match, mtime)

            others = [
                p
                for p, _ in sorted(discovered.items(), key=_rank, reverse=True)
            ]
            candidates.extend(others)
        except Exception:
            pass
        return candidates

    def _dist_cache_path(self, df: pd.DataFrame) -> Path:
        cache_dir, symbol, start, end, tf_minutes, align_mode, max_bars, source_key, idx = self._dist_cache_components(df)
        run_sig = self._dist_bundle_run_signature()
        runtime_cfg_sig = self._dist_runtime_cache_signature()
        sig_payload = (
            f"{len(idx)}|{idx.min()}|{idx.max()}|tf{tf_minutes}|{align_mode}|"
            f"max{self._dist_input_max_bars}|src{source_key}|run{run_sig}|cfg{runtime_cfg_sig}"
        )
        sig = hashlib.sha1(sig_payload.encode("utf-8")).hexdigest()[:10]
        source_part = f"_src{source_key}" if source_key else ""
        name = (
            f"{symbol}_{self._sanitize_token(start)}_{self._sanitize_token(end)}"
            f"_dist_tf{tf_minutes}_{self._sanitize_token(align_mode)}_max{max_bars}"
            f"{source_part}_cfg{runtime_cfg_sig}_{sig}_signals.parquet"
        )
        return cache_dir / name

    def _normalize_cache_df(self, precomputed_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if precomputed_df is None or not isinstance(precomputed_df, pd.DataFrame) or precomputed_df.empty:
            return None
        df = precomputed_df
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                return None
        if df.index.tz is None:
            try:
                df = df.copy()
                df.index = df.index.tz_localize("US/Eastern")
            except Exception:
                pass
        else:
            try:
                df = df.copy()
                df.index = df.index.tz_convert("US/Eastern")
            except Exception:
                pass
        return df

    @staticmethod
    def _finite_float(value, default: float = np.nan) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(out):
            return float(default)
        return float(out)

    @staticmethod
    def _dist_min_bracket_values(tick_size: float) -> tuple[float, float]:
        try:
            tick = float(tick_size)
        except Exception:
            tick = 0.25
        if not np.isfinite(tick) or tick <= 0.0:
            tick = 0.25
        min_cfg = CONFIG.get("SLTP_MIN", {}) or {}
        min_sl = MLPhysicsStrategy._finite_float(min_cfg.get("sl"), tick)
        min_tp = MLPhysicsStrategy._finite_float(min_cfg.get("tp"), tick)
        min_sl = max(min_sl, tick)
        min_tp = max(min_tp, tick)
        min_sl = float(np.ceil(min_sl / tick) * tick)
        min_tp = float(np.ceil(min_tp / tick) * tick)
        return float(min_tp), float(min_sl)

    @staticmethod
    def _dist_apply_min_brackets(tp_dist: float, sl_dist: float, tick_size: float) -> tuple[float, float]:
        min_tp, min_sl = MLPhysicsStrategy._dist_min_bracket_values(tick_size)
        return float(max(tp_dist, min_tp)), float(max(sl_dist, min_sl))

    @staticmethod
    def _dist_session_cfg_value(raw_cfg, session_name: str, *, default_key: str = "default"):
        if not isinstance(raw_cfg, dict):
            return raw_cfg
        sessions_cfg = raw_cfg.get("sessions", {}) or {}
        if isinstance(sessions_cfg, dict):
            for key in (session_name, str(session_name).upper(), str(session_name).lower()):
                if key in sessions_cfg:
                    return sessions_cfg.get(key)
        return raw_cfg.get(default_key)

    def _dist_current_gate_threshold(self, session_name: str, side: str, fallback: float) -> float:
        threshold = self._finite_float(fallback, np.nan)
        bundle = getattr(self, "_dist_bundle", None)
        if bundle is None:
            try:
                self._ensure_dist_bundle_loaded()
            except Exception:
                bundle = None
            else:
                bundle = getattr(self, "_dist_bundle", None)
        gate_root = getattr(bundle, "gate", None) or {}
        session_gate = None
        for key in (session_name, str(session_name).upper(), str(session_name).lower(), "GLOBAL"):
            payload = gate_root.get(key)
            if isinstance(payload, dict):
                session_gate = payload
                break
        if isinstance(session_gate, dict):
            side_payload = None
            for key in (side, str(side).upper(), str(side).lower()):
                payload = session_gate.get(key)
                if isinstance(payload, dict):
                    side_payload = payload
                    break
            if isinstance(side_payload, dict):
                threshold = self._finite_float(side_payload.get("threshold"), threshold)
        if not np.isfinite(threshold):
            threshold = self._finite_float(fallback, 0.0)
        return float(max(0.0, min(1.0, threshold)))

    @staticmethod
    def _dist_eastern_hour(ts_like) -> Optional[int]:
        try:
            ts = pd.Timestamp(ts_like)
        except Exception:
            return None
        if ts.tzinfo is None:
            try:
                ts = ts.tz_localize("US/Eastern")
            except Exception:
                return None
        else:
            try:
                ts = ts.tz_convert("US/Eastern")
            except Exception:
                return None
        try:
            hour = int(ts.hour)
        except Exception:
            return None
        if hour < 0 or hour > 23:
            return None
        return hour

    def _dist_wide_bracket_runner_allowed(
        self,
        *,
        session_name: str,
        runtime_regime: str,
        runtime_high_vol: bool,
        confidence_prob: float,
        ev_abs: float,
        current_time=None,
    ) -> bool:
        cfg = CONFIG.get("ML_PHYSICS_DIST_WIDE_BRACKET_RUNNER", {}) or {}
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return True

        allowed_sessions_raw = cfg.get("sessions", ())
        allowed_sessions = set()
        if isinstance(allowed_sessions_raw, (list, tuple, set)):
            allowed_sessions = {
                str(item).strip().upper()
                for item in allowed_sessions_raw
                if str(item).strip()
            }
        elif allowed_sessions_raw not in (None, ""):
            allowed_sessions = {str(allowed_sessions_raw).strip().upper()}
        if allowed_sessions and str(session_name or "").upper() not in allowed_sessions:
            return False

        if bool(cfg.get("require_high_vol", False)) and not bool(runtime_high_vol):
            return False

        allowed_regimes_raw = cfg.get("allowed_regimes", ())
        allowed_regimes = set()
        if isinstance(allowed_regimes_raw, (list, tuple, set)):
            allowed_regimes = {
                str(item).strip().lower()
                for item in allowed_regimes_raw
                if str(item).strip()
            }
        elif allowed_regimes_raw not in (None, ""):
            allowed_regimes = {str(allowed_regimes_raw).strip().lower()}
        if allowed_regimes and str(runtime_regime or "").lower() not in allowed_regimes:
            return False

        min_conf = self._finite_float(cfg.get("min_confidence"), np.nan)
        if np.isfinite(min_conf) and float(confidence_prob) < float(min_conf):
            return False

        min_ev_abs = max(0.0, self._finite_float(cfg.get("min_ev_abs"), 0.0))
        if min_ev_abs > 0.0 and float(ev_abs) < min_ev_abs:
            return False

        hours_cfg = cfg.get("hours", {}) or {}
        raw_hours = None
        if isinstance(hours_cfg, dict):
            raw_hours = (
                hours_cfg.get(session_name)
                or hours_cfg.get(str(session_name).upper())
                or hours_cfg.get(str(session_name).lower())
            )
        elif isinstance(hours_cfg, (list, tuple, set)):
            raw_hours = hours_cfg
        elif hours_cfg not in (None, ""):
            raw_hours = [hours_cfg]

        allowed_hours = set()
        if raw_hours is not None:
            hour_values = raw_hours if isinstance(raw_hours, (list, tuple, set)) else [raw_hours]
            for item in hour_values:
                try:
                    hour_value = int(item)
                except Exception:
                    continue
                if 0 <= hour_value <= 23:
                    allowed_hours.add(hour_value)
        if allowed_hours:
            hour = self._dist_eastern_hour(current_time)
            if hour is None or hour not in allowed_hours:
                return False
        return True

    def _dist_target_entry_session(self, current_time, fallback_session: str) -> str:
        try:
            bar_minutes = max(1, int(CONFIG.get("BAR_MINUTES", 1) or 1))
        except Exception:
            bar_minutes = 1
        if current_time is None:
            return str(fallback_session or "UNKNOWN").upper()
        try:
            entry_ts = pd.Timestamp(current_time) + pd.Timedelta(minutes=bar_minutes)
        except Exception:
            return str(fallback_session or "UNKNOWN").upper()
        inferred = self._infer_dist_session(entry_ts)
        if str(inferred or "").strip():
            return str(inferred).upper()
        return str(fallback_session or "UNKNOWN").upper()

    def _dist_is_wide_bracket(self, tp_dist: float, sl_dist: float) -> bool:
        cfg = CONFIG.get("ML_PHYSICS_DIST_WIDE_BRACKET_RUNNER", {}) or {}
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return False
        wide_tp_min = max(0.0, self._finite_float(cfg.get("wide_tp_min"), np.inf))
        wide_sl_min = max(0.0, self._finite_float(cfg.get("wide_sl_min"), np.inf))
        is_wide_tp = np.isfinite(wide_tp_min) and float(tp_dist) > (wide_tp_min + 1e-12)
        is_wide_sl = np.isfinite(wide_sl_min) and float(sl_dist) > (wide_sl_min + 1e-12)
        return bool(is_wide_tp or is_wide_sl)

    def _dist_normal_bracket_policy(
        self,
        *,
        session_name: str,
        runtime_regime: str,
        tick_size: float,
        tp_dist: float,
        sl_dist: float,
    ) -> tuple[float, float, bool]:
        if self._dist_is_wide_bracket(tp_dist, sl_dist):
            return float(tp_dist), float(sl_dist), False

        cfg = CONFIG.get("ML_PHYSICS_DIST_NORMAL_BRACKET_POLICY", {}) or {}
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return float(tp_dist), float(sl_dist), False

        session_cfg = self._dist_session_cfg_value(cfg, session_name)
        bracket_cfg = None
        if isinstance(session_cfg, dict):
            regimes_cfg = session_cfg.get("regimes", {}) or {}
            if isinstance(regimes_cfg, dict):
                regime_cfg = (
                    regimes_cfg.get(runtime_regime)
                    or regimes_cfg.get(str(runtime_regime).lower())
                    or regimes_cfg.get(str(runtime_regime).upper())
                )
                if isinstance(regime_cfg, dict):
                    bracket_cfg = regime_cfg
            if bracket_cfg is None:
                if isinstance(session_cfg.get("default"), dict):
                    bracket_cfg = session_cfg.get("default")
                elif "tp" in session_cfg or "sl" in session_cfg:
                    bracket_cfg = session_cfg
        if bracket_cfg is None and isinstance(cfg.get("default"), dict):
            bracket_cfg = cfg.get("default")
        if not isinstance(bracket_cfg, dict):
            return float(tp_dist), float(sl_dist), False

        policy_tp = self._finite_float(bracket_cfg.get("tp"), np.nan)
        policy_sl = self._finite_float(bracket_cfg.get("sl"), np.nan)
        if not np.isfinite(policy_tp) or not np.isfinite(policy_sl):
            return float(tp_dist), float(sl_dist), False
        if policy_tp <= 0.0 or policy_sl <= 0.0:
            return float(tp_dist), float(sl_dist), False

        try:
            tick = float(tick_size)
        except Exception:
            tick = 0.25
        if not np.isfinite(tick) or tick <= 0.0:
            tick = 0.25
        policy_tp = float(np.ceil(policy_tp / tick) * tick)
        policy_sl = float(np.ceil(policy_sl / tick) * tick)
        policy_tp, policy_sl = self._dist_apply_min_brackets(policy_tp, policy_sl, tick)
        return float(policy_tp), float(policy_sl), True

    def _dist_normal_profile_allowed(
        self,
        *,
        session_name: str,
        side: str,
        runtime_regime: str,
        current_time=None,
    ) -> bool:
        cfg = CONFIG.get("ML_PHYSICS_DIST_NORMAL_PROFILE", {}) or {}
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return True

        session_cfg_map = cfg.get("sessions", {}) or {}
        session_cfg = None
        if isinstance(session_cfg_map, dict):
            session_cfg = (
                session_cfg_map.get(session_name)
                or session_cfg_map.get(str(session_name).upper())
                or session_cfg_map.get(str(session_name).lower())
            )
        if not isinstance(session_cfg, dict):
            default_allowed = cfg.get("default_allowed_sides", None)
            session_cfg = {"allowed_sides": default_allowed}

        raw_allowed = session_cfg.get("allowed_sides", None)
        if raw_allowed is None:
            side_allowed = True
        else:
            allowed_sides = set()
            if isinstance(raw_allowed, (list, tuple, set)):
                allowed_sides = {
                    str(item).strip().upper()
                    for item in raw_allowed
                    if str(item).strip()
                }
            elif raw_allowed not in ("", None):
                allowed_sides = {str(raw_allowed).strip().upper()}
            side_allowed = str(side or "").upper() in allowed_sides

        raw_allowed_regimes = session_cfg.get("allowed_regimes", None)
        raw_blocked_regimes = session_cfg.get("blocked_regimes", None)
        regime_key = str(runtime_regime or "").strip().lower()

        allowed_regimes = None
        if isinstance(raw_allowed_regimes, (list, tuple, set)):
            allowed_regimes = {
                str(item).strip().lower()
                for item in raw_allowed_regimes
                if str(item).strip()
            }
        elif raw_allowed_regimes not in ("", None):
            allowed_regimes = {str(raw_allowed_regimes).strip().lower()}

        blocked_regimes = set()
        if isinstance(raw_blocked_regimes, (list, tuple, set)):
            blocked_regimes = {
                str(item).strip().lower()
                for item in raw_blocked_regimes
                if str(item).strip()
            }
        elif raw_blocked_regimes not in ("", None):
            blocked_regimes = {str(raw_blocked_regimes).strip().lower()}

        if not side_allowed:
            return False
        if allowed_regimes is not None and regime_key not in allowed_regimes:
            return False
        if regime_key and regime_key in blocked_regimes:
            return False

        raw_allowed_hours = session_cfg.get("allowed_hours", None)
        raw_blocked_hours = session_cfg.get("blocked_hours", None)
        if raw_allowed_hours is not None or raw_blocked_hours is not None:
            hour = self._dist_eastern_hour(current_time)
            if hour is None:
                return False
            allowed_hours = set()
            if raw_allowed_hours is not None:
                hour_values = raw_allowed_hours if isinstance(raw_allowed_hours, (list, tuple, set)) else [raw_allowed_hours]
                for item in hour_values:
                    try:
                        hour_value = int(item)
                    except Exception:
                        continue
                    if 0 <= hour_value <= 23:
                        allowed_hours.add(hour_value)
            if allowed_hours and hour not in allowed_hours:
                return False

            blocked_hours = set()
            if raw_blocked_hours is not None:
                hour_values = raw_blocked_hours if isinstance(raw_blocked_hours, (list, tuple, set)) else [raw_blocked_hours]
                for item in hour_values:
                    try:
                        hour_value = int(item)
                    except Exception:
                        continue
                    if 0 <= hour_value <= 23:
                        blocked_hours.add(hour_value)
            if blocked_hours and hour in blocked_hours:
                return False
        return True

    def _dist_runtime_signal_block_reason(
        self,
        *,
        side: str,
        session_name: str,
        tp_dist: float,
        sl_dist: float,
        confidence_prob: float,
        tick_size: float,
        ev_abs: float,
        runtime_regime: str,
        runtime_high_vol: bool,
        current_time=None,
    ) -> Optional[str]:
        ev_cfg = CONFIG.get("ML_PHYSICS_DIST_RUNTIME_MIN_EV_ABS", {}) or {}
        ev_threshold = 0.0
        ev_enabled = True
        if isinstance(ev_cfg, dict):
            ev_enabled = bool(ev_cfg.get("enabled", False))
            raw_ev = self._dist_session_cfg_value(ev_cfg, session_name)
        else:
            raw_ev = ev_cfg
        if ev_enabled:
            ev_threshold = max(0.0, self._finite_float(raw_ev, 0.0))
        if ev_enabled and ev_threshold > 0.0 and float(ev_abs) < ev_threshold:
            return "ev_filter"

        is_wide = self._dist_is_wide_bracket(tp_dist, sl_dist)
        if is_wide and not self._dist_wide_bracket_runner_allowed(
                session_name=session_name,
                runtime_regime=runtime_regime,
                runtime_high_vol=runtime_high_vol,
                confidence_prob=confidence_prob,
                ev_abs=ev_abs,
                current_time=current_time,
            ):
                return "wide_bracket_runner_only"
        if not is_wide and not self._dist_normal_profile_allowed(
            session_name=session_name,
            side=side,
            runtime_regime=runtime_regime,
            current_time=current_time,
        ):
            return "normal_profile_side_block"

        rr_cfg = CONFIG.get("ML_PHYSICS_DIST_RUNTIME_MIN_RR", {}) or {}
        rr_threshold = 0.0
        rr_enabled = True
        if isinstance(rr_cfg, dict):
            rr_enabled = bool(rr_cfg.get("enabled", False))
            raw_rr = self._dist_session_cfg_value(rr_cfg, session_name)
        else:
            raw_rr = rr_cfg
        if rr_enabled:
            rr_threshold = max(0.0, self._finite_float(raw_rr, 0.0))
        rr = float(tp_dist / max(sl_dist, 1e-9))
        if rr_enabled and rr_threshold > 0.0 and rr <= rr_threshold + 1e-12:
            return "rr_filter"

        floor_cfg = CONFIG.get("ML_PHYSICS_DIST_FLOOR_BRACKET_FILTER", {}) or {}
        floor_enabled = True
        if isinstance(floor_cfg, dict):
            floor_enabled = bool(floor_cfg.get("enabled", False))
            raw_floor_conf = self._dist_session_cfg_value(floor_cfg, session_name)
        else:
            raw_floor_conf = floor_cfg
        floor_min_conf = self._finite_float(raw_floor_conf, np.nan)
        if not floor_enabled or not np.isfinite(floor_min_conf):
            return None

        min_tp, min_sl = self._dist_min_bracket_values(tick_size)
        tol = max(1e-6, float(max(tick_size, 0.25)) * 0.1)
        has_floor_leg = abs(float(sl_dist) - float(min_sl)) <= tol or abs(float(tp_dist) - float(min_tp)) <= tol
        if has_floor_leg and float(confidence_prob) < float(floor_min_conf):
            return "floor_bracket_filter"
        return None

    def _dist_entry_policy_block_reason(
        self,
        *,
        side: str,
        session_name: str,
        ev_abs: float,
        gate_prob: float,
        runtime_regime: str,
        is_runner: bool,
        current_time=None,
    ) -> Optional[str]:
        cfg = CONFIG.get("ML_PHYSICS_DIST_ENTRY_POLICY", {}) or {}
        if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
            return None

        session_cfg_map = cfg.get("sessions", {}) or {}
        session_cfg = None
        if isinstance(session_cfg_map, dict):
            session_cfg = (
                session_cfg_map.get(session_name)
                or session_cfg_map.get(str(session_name).upper())
                or session_cfg_map.get(str(session_name).lower())
            )
        if not isinstance(session_cfg, dict):
            session_cfg = {}

        default_allowed = cfg.get("default_allowed_sides", None)
        raw_allowed = session_cfg.get("allowed_sides", default_allowed)
        if raw_allowed is not None:
            allowed_sides = set()
            if isinstance(raw_allowed, (list, tuple, set)):
                allowed_sides = {
                    str(item).strip().upper()
                    for item in raw_allowed
                    if str(item).strip()
                }
            elif raw_allowed not in ("", None):
                allowed_sides = {str(raw_allowed).strip().upper()}
            if str(side or "").upper() not in allowed_sides:
                return "entry_policy_side_block"

        raw_allowed_regimes = session_cfg.get("allowed_regimes", cfg.get("default_allowed_regimes", None))
        if raw_allowed_regimes is not None:
            allowed_regimes = set()
            if isinstance(raw_allowed_regimes, (list, tuple, set)):
                allowed_regimes = {
                    str(item).strip().lower()
                    for item in raw_allowed_regimes
                    if str(item).strip()
                }
            elif raw_allowed_regimes not in ("", None):
                allowed_regimes = {str(raw_allowed_regimes).strip().lower()}
            if allowed_regimes and str(runtime_regime or "").lower() not in allowed_regimes:
                return "entry_policy_regime_block"

        allow_runner = session_cfg.get("allow_runner", cfg.get("default_allow_runner", None))
        if allow_runner is not None and not bool(allow_runner) and bool(is_runner):
            return "entry_policy_runner_block"

        raw_allowed_weekdays = session_cfg.get("allowed_weekdays", cfg.get("default_allowed_weekdays", None))
        if raw_allowed_weekdays is not None:
            allowed_weekdays = set()
            weekday_values = (
                raw_allowed_weekdays
                if isinstance(raw_allowed_weekdays, (list, tuple, set))
                else [raw_allowed_weekdays]
            )
            for item in weekday_values:
                if isinstance(item, (int, np.integer)):
                    try:
                        allowed_weekdays.add(int(item))
                    except Exception:
                        continue
                else:
                    text = str(item or "").strip().lower()
                    if not text:
                        continue
                    allowed_weekdays.add(text)
            try:
                ts = pd.Timestamp(current_time) if current_time is not None else None
            except Exception:
                ts = None
            if ts is None:
                return "entry_policy_weekday_block"
            if ts.tzinfo is None:
                try:
                    ts = ts.tz_localize("US/Eastern")
                except Exception:
                    return "entry_policy_weekday_block"
            else:
                try:
                    ts = ts.tz_convert("US/Eastern")
                except Exception:
                    return "entry_policy_weekday_block"
            weekday_num = int(ts.weekday())
            weekday_name = str(ts.day_name()).strip().lower()
            if weekday_num not in allowed_weekdays and weekday_name not in allowed_weekdays:
                return "entry_policy_weekday_block"

        raw_ev = session_cfg.get("min_ev_abs", cfg.get("default_min_ev_abs", None))
        min_ev_abs = self._finite_float(raw_ev, np.nan)
        if np.isfinite(min_ev_abs) and max(0.0, float(ev_abs)) < max(0.0, float(min_ev_abs)):
            return "entry_policy_ev_filter"

        raw_min_gate = session_cfg.get("min_gate_prob", cfg.get("default_min_gate_prob", None))
        min_gate_prob = self._finite_float(raw_min_gate, np.nan)
        if np.isfinite(min_gate_prob) and np.isfinite(gate_prob) and float(gate_prob) < float(min_gate_prob):
            return "entry_policy_gate_prob"

        raw_max_gate = session_cfg.get("max_gate_prob", cfg.get("default_max_gate_prob", None))
        max_gate_prob = self._finite_float(raw_max_gate, np.nan)
        if np.isfinite(max_gate_prob) and np.isfinite(gate_prob) and float(gate_prob) > float(max_gate_prob):
            return "entry_policy_gate_prob"

        return None

    @staticmethod
    def _dist_signal_side_code(side: Optional[str]) -> np.int8:
        side_key = str(side or "").upper()
        if side_key == "LONG":
            return np.int8(1)
        if side_key == "SHORT":
            return np.int8(-1)
        return np.int8(0)

    @staticmethod
    def _dist_signal_side_text(code: int) -> str:
        if int(code) > 0:
            return "LONG"
        if int(code) < 0:
            return "SHORT"
        return "NONE"

    @staticmethod
    def _dist_categorical_series(
        values,
        *,
        default: str,
        categories: Optional[list[str]] = None,
        upper: bool = False,
        lower: bool = False,
    ) -> pd.Series:
        series = pd.Series(values, copy=False)
        series = series.where(series.notna(), default)
        series = series.astype(str)
        if upper:
            series = series.str.upper()
        elif lower:
            series = series.str.lower()
        series = series.replace({"": default, "nan": default, "none": default, "None": default, "NONE": default})
        if categories is None:
            return series.astype("category")
        ordered_categories = list(categories)
        extras = [item for item in pd.unique(series) if item not in ordered_categories]
        return pd.Series(
            pd.Categorical(series, categories=[*ordered_categories, *extras]),
            index=series.index,
        )

    def _dist_compact_cache_from_legacy(
        self,
        df: pd.DataFrame,
        *,
        session_labels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        work = df.copy()
        if "signal_obj" not in work.columns:
            if "signal_json" in work.columns:
                work["signal_obj"] = work["signal_json"].map(lambda s: self._json_load_safe(str(s or ""), None))
            else:
                work["signal_obj"] = None
        if "eval_obj" not in work.columns:
            if "eval_json" in work.columns:
                work["eval_obj"] = work["eval_json"].map(lambda s: self._json_load_safe(str(s or ""), {}))
            else:
                work["eval_obj"] = {}

        total = len(work)
        has_signal = (
            pd.to_numeric(work.get("has_signal", 0), errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.int8, copy=True)
        )
        signal_side = np.zeros(total, dtype=np.int8)
        signal_tp_dist = np.zeros(total, dtype=np.float32)
        signal_sl_dist = np.zeros(total, dtype=np.float32)
        signal_conf = np.zeros(total, dtype=np.float32)
        signal_conf_raw = np.zeros(total, dtype=np.float32)
        signal_ev_pred = np.zeros(total, dtype=np.float32)
        signal_ev_min_req = np.zeros(total, dtype=np.float32)
        eval_decision = np.full(total, "no_signal", dtype=object)
        eval_session = np.full(total, "UNKNOWN", dtype=object)
        eval_blocked_reason = np.full(total, "", dtype=object)
        eval_regime = np.full(total, "unknown", dtype=object)
        eval_gate_prob = np.full(total, np.nan, dtype=np.float32)
        eval_gate_threshold = np.full(total, np.nan, dtype=np.float32)
        eval_gate_margin_min = np.zeros(total, dtype=np.float32)
        default_sessions = (
            np.asarray(session_labels, dtype=object)
            if session_labels is not None and len(session_labels) == total
            else np.full(total, "UNKNOWN", dtype=object)
        )

        raw_signal_cache = work["signal_obj"].tolist()
        raw_eval_cache = work["eval_obj"].tolist()
        for i in range(total):
            default_session = str(default_sessions[i] or "UNKNOWN").upper()
            eval_payload = raw_eval_cache[i] if isinstance(raw_eval_cache[i], dict) else {}
            decision = str(eval_payload.get("decision") or "no_signal").lower()
            blocked_reason = str(eval_payload.get("blocked_reason") or "").strip()
            session_name = str(eval_payload.get("session") or default_session).upper()
            runtime_regime = str(eval_payload.get("runtime_regime") or "unknown").lower()
            gate_prob = self._finite_float(eval_payload.get("trade_gate_prob"), np.nan)
            gate_threshold = self._finite_float(eval_payload.get("trade_gate_threshold"), np.nan)
            gate_margin_min = self._finite_float(eval_payload.get("trade_gate_margin_min"), 0.0)

            signal_payload = raw_signal_cache[i] if isinstance(raw_signal_cache[i], dict) else None
            if isinstance(signal_payload, dict):
                signal_payload = self._normalize_dist_signal_payload(signal_payload)
                side = str(signal_payload.get("side") or "").upper()
                side_code = self._dist_signal_side_code(side)
                if int(side_code) != 0:
                    has_signal[i] = np.int8(1)
                    signal_side[i] = side_code
                    signal_tp_dist[i] = np.float32(self._finite_float(signal_payload.get("tp_dist"), 0.0))
                    signal_sl_dist[i] = np.float32(self._finite_float(signal_payload.get("sl_dist"), 0.0))
                    signal_conf[i] = np.float32(self._finite_float(signal_payload.get("ml_confidence"), 0.0))
                    signal_conf_raw[i] = np.float32(
                        self._finite_float(
                            signal_payload.get("ml_confidence_raw"),
                            self._finite_float(signal_payload.get("ml_confidence"), 0.0),
                        )
                    )
                    signal_ev_min_req[i] = np.float32(self._finite_float(signal_payload.get("ml_ev_min_req"), 0.0))
                    if side == "LONG":
                        signal_ev_pred[i] = np.float32(self._finite_float(signal_payload.get("ml_ev_long"), 0.0))
                    else:
                        signal_ev_pred[i] = np.float32(self._finite_float(signal_payload.get("ml_ev_short"), 0.0))
                    if decision not in {"signal_long", "signal_short"}:
                        decision = "signal_long" if side == "LONG" else "signal_short"
                    if session_name in {"", "UNKNOWN"}:
                        strategy_name = str(signal_payload.get("strategy") or "")
                        if strategy_name.startswith("MLPhysics_"):
                            session_name = str(strategy_name.split("MLPhysics_", 1)[1] or default_session).upper()
                        else:
                            session_name = default_session
                    if runtime_regime == "unknown":
                        runtime_regime = str(signal_payload.get("ml_regime") or "unknown").lower()
                    if not np.isfinite(gate_prob):
                        gate_prob = self._finite_float(signal_payload.get("ml_trade_gate_prob"), np.nan)
                    if not np.isfinite(gate_threshold):
                        gate_threshold = self._finite_float(signal_payload.get("ml_trade_gate_threshold"), np.nan)
                    gate_margin_min = max(
                        gate_margin_min,
                        self._finite_float(signal_payload.get("ml_margin_req"), gate_margin_min),
                    )

            eval_decision[i] = decision or "no_signal"
            eval_session[i] = session_name or default_session
            eval_blocked_reason[i] = blocked_reason
            eval_regime[i] = runtime_regime or "unknown"
            eval_gate_prob[i] = np.float32(gate_prob if np.isfinite(gate_prob) else np.nan)
            eval_gate_threshold[i] = np.float32(gate_threshold if np.isfinite(gate_threshold) else np.nan)
            eval_gate_margin_min[i] = np.float32(max(0.0, gate_margin_min))

        out = pd.DataFrame(index=work.index)
        out["has_signal"] = has_signal
        out["signal_side"] = signal_side
        out["signal_tp_dist"] = signal_tp_dist
        out["signal_sl_dist"] = signal_sl_dist
        out["signal_confidence"] = signal_conf
        out["signal_confidence_raw"] = signal_conf_raw
        out["signal_ev_pred"] = signal_ev_pred
        out["signal_ev_min_req"] = signal_ev_min_req
        out["eval_decision"] = eval_decision
        out["eval_session"] = eval_session
        out["eval_blocked_reason"] = eval_blocked_reason
        out["eval_runtime_regime"] = eval_regime
        out["eval_trade_gate_prob"] = eval_gate_prob
        out["eval_trade_gate_threshold"] = eval_gate_threshold
        out["eval_trade_gate_margin_min"] = eval_gate_margin_min
        return out

    def _prepare_dist_precomputed_cache_df(
        self,
        precomputed_df: Optional[pd.DataFrame],
        *,
        session_labels: Optional[np.ndarray] = None,
        partial_gap_reason: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        norm = self._normalize_cache_df(precomputed_df) if precomputed_df is not None else None
        if norm is None:
            return None
        df = norm
        if "signal_side" not in df.columns or "eval_decision" not in df.columns:
            df = self._dist_compact_cache_from_legacy(df, session_labels=session_labels)
        else:
            df = df.copy()

        total = len(df)
        default_sessions = (
            np.asarray(session_labels, dtype=object)
            if session_labels is not None and len(session_labels) == total
            else np.full(total, "UNKNOWN", dtype=object)
        )

        signal_side_raw = df["signal_side"] if "signal_side" in df.columns else pd.Series(0, index=df.index)
        signal_side = pd.to_numeric(signal_side_raw, errors="coerce").fillna(0).astype(np.int8, copy=False)
        has_signal_raw = df["has_signal"] if "has_signal" in df.columns else pd.Series(0, index=df.index)
        has_signal = pd.to_numeric(has_signal_raw, errors="coerce").fillna(0)
        has_signal = ((has_signal.to_numpy(dtype=np.int8, copy=False) > 0) | (signal_side.to_numpy(dtype=np.int8, copy=False) != 0)).astype(np.int8)
        df["has_signal"] = has_signal
        df["signal_side"] = np.where(has_signal > 0, signal_side.to_numpy(dtype=np.int8, copy=False), 0).astype(np.int8, copy=False)

        float_defaults = {
            "signal_tp_dist": 0.0,
            "signal_sl_dist": 0.0,
            "signal_confidence": 0.0,
            "signal_confidence_raw": np.nan,
            "signal_ev_pred": 0.0,
            "signal_ev_min_req": 0.0,
            "eval_trade_gate_prob": np.nan,
            "eval_trade_gate_threshold": np.nan,
            "eval_trade_gate_margin_min": 0.0,
        }
        for col, default_value in float_defaults.items():
            raw = df[col] if col in df.columns else pd.Series(default_value, index=df.index)
            series = pd.to_numeric(raw, errors="coerce")
            if np.isnan(default_value):
                df[col] = series.astype(np.float32, copy=False)
            else:
                df[col] = series.fillna(default_value).astype(np.float32, copy=False)

        conf_raw = df["signal_confidence_raw"].to_numpy(dtype=np.float32, copy=False)
        conf = df["signal_confidence"].to_numpy(dtype=np.float32, copy=False)
        if np.any(~np.isfinite(conf_raw)):
            repaired = np.where(np.isfinite(conf_raw), conf_raw, conf)
            df["signal_confidence_raw"] = repaired.astype(np.float32, copy=False)

        session_series = df.get("eval_session", default_sessions)
        if not isinstance(session_series, pd.Series):
            session_series = pd.Series(session_series, index=df.index)
        session_series = session_series.where(session_series.notna(), pd.Series(default_sessions, index=df.index))
        if len(default_sessions) == total:
            session_series = session_series.mask(
                session_series.astype(str).str.strip().isin(["", "UNKNOWN", "nan", "None"]),
                pd.Series(default_sessions, index=df.index),
            )
        df["eval_session"] = self._dist_categorical_series(
            session_series,
            default="UNKNOWN",
            categories=_DIST_CACHE_SESSION_CATEGORIES,
            upper=True,
        )
        df["eval_decision"] = self._dist_categorical_series(
            df["eval_decision"] if "eval_decision" in df.columns else pd.Series("no_signal", index=df.index),
            default="no_signal",
            categories=_DIST_CACHE_DECISION_CATEGORIES,
            lower=True,
        )
        df["eval_runtime_regime"] = self._dist_categorical_series(
            df["eval_runtime_regime"] if "eval_runtime_regime" in df.columns else pd.Series("unknown", index=df.index),
            default="unknown",
            categories=_DIST_CACHE_REGIME_CATEGORIES,
            lower=True,
        )
        blocked_reason_series = df.get("eval_blocked_reason", "")
        if not isinstance(blocked_reason_series, pd.Series):
            blocked_reason_series = pd.Series(blocked_reason_series, index=df.index)
        blocked_reason_series = blocked_reason_series.where(blocked_reason_series.notna(), "")
        blocked_reason_series = blocked_reason_series.astype(str)
        blocked_reason_series = blocked_reason_series.replace({"nan": "", "None": ""})
        if partial_gap_reason:
            missing_mask = (
                (df["has_signal"].to_numpy(dtype=np.int8, copy=False) <= 0)
                & (df["eval_decision"].astype(str).to_numpy() == "no_signal")
                & (blocked_reason_series.to_numpy(dtype=object, copy=False) == "")
            )
            if np.any(missing_mask):
                blocked_reason_series = blocked_reason_series.copy()
                blocked_reason_series.iloc[np.flatnonzero(missing_mask)] = str(partial_gap_reason)
        df["eval_blocked_reason"] = blocked_reason_series.astype("category")
        return df

    @staticmethod
    def _dist_category_value(code_arr: np.ndarray, pos: int, categories: list[str], default: str) -> str:
        if 0 <= pos < len(code_arr):
            code = int(code_arr[pos])
            if 0 <= code < len(categories):
                value = str(categories[code] or default)
                return value if value else default
        return default

    def _dist_blocked_reason_value(self, pos: int) -> Optional[str]:
        if not (0 <= pos < len(self._opt_dist_eval_blocked_reason_codes)):
            return None
        code = int(self._opt_dist_eval_blocked_reason_codes[pos])
        if 0 <= code < len(self._opt_dist_eval_blocked_reason_categories):
            value = str(self._opt_dist_eval_blocked_reason_categories[code] or "").strip()
            return value or None
        return None

    def _dist_eval_from_cache_row(self, pos: int, current_time=None) -> dict:
        decision = self._dist_category_value(
            self._opt_dist_eval_decision_codes,
            pos,
            self._opt_dist_eval_decision_categories,
            "unknown",
        )
        session_name = self._dist_category_value(
            self._opt_dist_eval_session_codes,
            pos,
            self._opt_dist_eval_session_categories,
            "UNKNOWN",
        )
        if session_name in {"", "UNKNOWN"} and current_time is not None:
            session_name = self._infer_dist_session(current_time)
        runtime_regime = self._dist_category_value(
            self._opt_dist_eval_regime_codes,
            pos,
            self._opt_dist_eval_regime_categories,
            "unknown",
        )
        gate_prob = (
            float(self._opt_dist_eval_gate_prob_arr[pos])
            if 0 <= pos < len(self._opt_dist_eval_gate_prob_arr)
            else np.nan
        )
        gate_threshold = (
            float(self._opt_dist_eval_gate_threshold_arr[pos])
            if 0 <= pos < len(self._opt_dist_eval_gate_threshold_arr)
            else np.nan
        )
        gate_margin_min = (
            float(self._opt_dist_eval_gate_margin_min_arr[pos])
            if 0 <= pos < len(self._opt_dist_eval_gate_margin_min_arr)
            else 0.0
        )
        payload = {
            "decision": decision,
            "blocked_reason": self._dist_blocked_reason_value(pos),
            "session": session_name,
            "ev_source": "dist_bracket_ml",
            "runtime_regime": runtime_regime,
            "trade_gate_margin_min": gate_margin_min,
        }
        if np.isfinite(gate_prob):
            payload["trade_gate_prob"] = gate_prob
        if np.isfinite(gate_threshold):
            payload["trade_gate_threshold"] = gate_threshold
        if np.isfinite(gate_prob) and np.isfinite(gate_threshold):
            payload["trade_gate_margin"] = float(gate_prob - gate_threshold)
        return payload

    def _dist_signal_from_cache_row(self, pos: int, eval_payload: Optional[dict], *, current_time=None) -> Optional[Dict]:
        if not (0 <= pos < len(self._opt_dist_signal_side_arr)):
            return None
        side_code = int(self._opt_dist_signal_side_arr[pos])
        side = self._dist_signal_side_text(side_code)
        if side not in {"LONG", "SHORT"}:
            return None
        eval_payload = eval_payload if isinstance(eval_payload, dict) else {}
        session_name = str(eval_payload.get("session") or "UNKNOWN").upper()
        session_name = self._dist_target_entry_session(current_time, session_name)
        runtime_regime = str(eval_payload.get("runtime_regime") or "unknown").lower()
        gate_prob = self._finite_float(eval_payload.get("trade_gate_prob"), 0.0)
        gate_threshold = self._finite_float(eval_payload.get("trade_gate_threshold"), np.nan)
        gate_margin_req = self._finite_float(eval_payload.get("trade_gate_margin_min"), 0.0)
        confidence = (
            float(self._opt_dist_signal_confidence_arr[pos])
            if 0 <= pos < len(self._opt_dist_signal_confidence_arr)
            else 0.0
        )
        confidence_raw = (
            float(self._opt_dist_signal_confidence_raw_arr[pos])
            if 0 <= pos < len(self._opt_dist_signal_confidence_raw_arr)
            else confidence
        )
        ev_pred = (
            float(self._opt_dist_signal_ev_pred_arr[pos])
            if 0 <= pos < len(self._opt_dist_signal_ev_pred_arr)
            else 0.0
        )
        ev_min_req = (
            float(self._opt_dist_signal_ev_min_req_arr[pos])
            if 0 <= pos < len(self._opt_dist_signal_ev_min_req_arr)
            else 0.0
        )
        if side == "LONG":
            prob_up = confidence
            prob_down = float(1.0 - confidence)
            ev_long = ev_pred
            ev_short = -ev_pred
        else:
            prob_down = confidence
            prob_up = float(1.0 - confidence)
            ev_short = ev_pred
            ev_long = -ev_pred
        if np.isfinite(gate_threshold):
            gate_threshold = float(max(0.0, min(1.0, gate_threshold)))
        else:
            gate_threshold = self._dist_current_gate_threshold(session_name, side, 0.0)
        gate_margin = float(gate_prob - gate_threshold) if np.isfinite(gate_prob) and np.isfinite(gate_threshold) else 0.0
        if isinstance(eval_payload, dict):
            eval_payload["session"] = session_name
            eval_payload["runtime_regime"] = runtime_regime
            eval_payload["trade_gate_prob"] = gate_prob
            eval_payload["trade_gate_threshold"] = gate_threshold
            eval_payload["trade_gate_margin"] = gate_margin
            eval_payload["trade_gate_margin_min"] = gate_margin_req
        bundle_cfg = getattr(getattr(self, "_dist_bundle", None), "cfg", None)
        tick_size = float(getattr(bundle_cfg, "tick_size", CONFIG.get("TICK_SIZE", 0.25)) or 0.25)
        tp_dist = float(self._opt_dist_signal_tp_dist_arr[pos]) if 0 <= pos < len(self._opt_dist_signal_tp_dist_arr) else 0.0
        sl_dist = float(self._opt_dist_signal_sl_dist_arr[pos]) if 0 <= pos < len(self._opt_dist_signal_sl_dist_arr) else 0.0
        tp_dist, sl_dist = self._dist_apply_min_brackets(tp_dist, sl_dist, tick_size)
        tp_dist, sl_dist, _ = self._dist_normal_bracket_policy(
            session_name=session_name,
            runtime_regime=runtime_regime,
            tick_size=tick_size,
            tp_dist=tp_dist,
            sl_dist=sl_dist,
        )
        is_runner = bool(self._dist_is_wide_bracket(tp_dist, sl_dist))
        ev_abs = abs(float(ev_pred))
        blocked_reason = self._dist_runtime_signal_block_reason(
            side=side,
            session_name=session_name,
            tp_dist=tp_dist,
            sl_dist=sl_dist,
            confidence_prob=confidence,
            tick_size=tick_size,
            ev_abs=ev_abs,
            runtime_regime=runtime_regime,
            runtime_high_vol=(runtime_regime == "high"),
            current_time=current_time,
        )
        if blocked_reason:
            if isinstance(eval_payload, dict):
                eval_payload["decision"] = "blocked"
                eval_payload["blocked_reason"] = blocked_reason
                eval_payload["session"] = session_name
                eval_payload["runtime_regime"] = runtime_regime
                eval_payload["trade_gate_prob"] = gate_prob
                eval_payload["trade_gate_threshold"] = gate_threshold
                eval_payload["trade_gate_margin"] = gate_margin
                eval_payload["trade_gate_margin_min"] = gate_margin_req
            return None
        entry_policy_blocked_reason = self._dist_entry_policy_block_reason(
            side=side,
            session_name=session_name,
            ev_abs=ev_abs,
            gate_prob=gate_prob,
            runtime_regime=runtime_regime,
            is_runner=is_runner,
            current_time=current_time,
        )
        if entry_policy_blocked_reason:
            if isinstance(eval_payload, dict):
                eval_payload["decision"] = "blocked"
                eval_payload["blocked_reason"] = entry_policy_blocked_reason
                eval_payload["session"] = session_name
                eval_payload["runtime_regime"] = runtime_regime
                eval_payload["trade_gate_prob"] = gate_prob
                eval_payload["trade_gate_threshold"] = gate_threshold
                eval_payload["trade_gate_margin"] = gate_margin
                eval_payload["trade_gate_margin_min"] = gate_margin_req
            return None
        return {
            "strategy": f"MLPhysics_{session_name}",
            "side": side,
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "ml_is_runner": is_runner,
            "ml_confidence": confidence,
            "ml_confidence_raw": confidence_raw,
            "ml_threshold": gate_threshold,
            "ml_prob_up": prob_up,
            "ml_prob_down": prob_down,
            "ml_short_threshold": gate_threshold,
            "ml_regime": runtime_regime,
            "ml_high_vol": runtime_regime == "high",
            "ml_candidate_side": side,
            "ml_margin": gate_margin,
            "ml_margin_req": gate_margin_req,
            "ml_gate_min_conf": gate_threshold,
            "ml_gate_block_sides": "",
            "ml_trade_gate_prob": gate_prob,
            "ml_trade_gate_threshold": gate_threshold,
            "ml_trade_gate_margin": gate_margin,
            "ml_trade_gate_soft_penalty": 0.0,
            "ml_trade_gate_policy": "dist_gate",
            "ml_trade_gate_required": True,
            "ml_budget_cov_recent": None,
            "ml_ev_long": ev_long,
            "ml_ev_short": ev_short,
            "ml_ev_long_effective": ev_long,
            "ml_ev_short_effective": ev_short,
            "ml_ev_source": "dist_bracket_ml",
            "ml_ev_min_req": ev_min_req,
            "ml_ev_min_req_effective": ev_min_req,
            "ml_ev_prob_edge": 0.0,
            "ml_ev_prob_edge_req": 0.0,
            "ml_decision": eval_payload.get("decision"),
            "ml_blocked_reason": eval_payload.get("blocked_reason"),
        }

    @staticmethod
    def _datetime_index_ns(index: pd.DatetimeIndex) -> np.ndarray:
        try:
            return index.asi8
        except Exception:
            return index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)

    @staticmethod
    def _timestamp_to_eastern_ns(ts_like) -> Optional[int]:
        try:
            ts = pd.Timestamp(ts_like)
        except Exception:
            return None
        if ts.tzinfo is None:
            try:
                ts = ts.tz_localize("US/Eastern")
            except Exception:
                return None
        try:
            return int(ts.value)
        except Exception:
            return None

    @staticmethod
    def _lookup_sorted_ns(index_ns: np.ndarray, ns: int, cursor_pos: int) -> Optional[int]:
        size = int(len(index_ns))
        if size <= 0:
            return None
        pos = min(max(int(cursor_pos or 0), 0), size - 1)
        if int(index_ns[pos]) == ns:
            return pos
        if pos + 1 < size and int(index_ns[pos + 1]) == ns:
            return pos + 1
        if pos > 0 and int(index_ns[pos - 1]) == ns:
            return pos - 1
        found = int(np.searchsorted(index_ns, ns, side="left"))
        if 0 <= found < size and int(index_ns[found]) == ns:
            return found
        return None

    def set_precomputed_dist_backtest_df(self, precomputed_df: Optional[pd.DataFrame]) -> None:
        df = self._prepare_dist_precomputed_cache_df(precomputed_df)
        if df is None:
            self._opt_dist_precomputed_df = None
            self._opt_dist_index_ns = np.empty(0, dtype=np.int64)
            self._opt_dist_has_signal_arr = np.empty(0, dtype=np.int8)
            self._opt_dist_signal_side_arr = np.empty(0, dtype=np.int8)
            self._opt_dist_signal_tp_dist_arr = np.empty(0, dtype=np.float32)
            self._opt_dist_signal_sl_dist_arr = np.empty(0, dtype=np.float32)
            self._opt_dist_signal_confidence_arr = np.empty(0, dtype=np.float32)
            self._opt_dist_signal_confidence_raw_arr = np.empty(0, dtype=np.float32)
            self._opt_dist_signal_ev_pred_arr = np.empty(0, dtype=np.float32)
            self._opt_dist_signal_ev_min_req_arr = np.empty(0, dtype=np.float32)
            self._opt_dist_eval_decision_codes = np.empty(0, dtype=np.int8)
            self._opt_dist_eval_session_codes = np.empty(0, dtype=np.int8)
            self._opt_dist_eval_regime_codes = np.empty(0, dtype=np.int8)
            self._opt_dist_eval_decision_categories = list(_DIST_CACHE_DECISION_CATEGORIES)
            self._opt_dist_eval_session_categories = list(_DIST_CACHE_SESSION_CATEGORIES)
            self._opt_dist_eval_regime_categories = list(_DIST_CACHE_REGIME_CATEGORIES)
            self._opt_dist_eval_blocked_reason_codes = np.empty(0, dtype=np.int32)
            self._opt_dist_eval_blocked_reason_categories = []
            self._opt_dist_eval_gate_prob_arr = np.empty(0, dtype=np.float32)
            self._opt_dist_eval_gate_threshold_arr = np.empty(0, dtype=np.float32)
            self._opt_dist_eval_gate_margin_min_arr = np.empty(0, dtype=np.float32)
            self._opt_dist_signal_cache = []
            self._opt_dist_eval_cache = []
            self._opt_dist_cursor_pos = 0
            return
        self._opt_dist_precomputed_df = df
        try:
            idx = pd.DatetimeIndex(df.index)
            self._opt_dist_index_ns = self._datetime_index_ns(idx).copy()
        except Exception:
            self._opt_dist_index_ns = np.empty(0, dtype=np.int64)
        try:
            self._opt_dist_has_signal_arr = (
                pd.to_numeric(df.get("has_signal", 0), errors="coerce")
                .fillna(0)
                .to_numpy(dtype=np.int8, copy=True)
            )
        except Exception:
            self._opt_dist_has_signal_arr = np.zeros(len(df), dtype=np.int8)
        try:
            self._opt_dist_signal_side_arr = pd.to_numeric(df.get("signal_side", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int8, copy=True)
            self._opt_dist_signal_tp_dist_arr = pd.to_numeric(df.get("signal_tp_dist", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
            self._opt_dist_signal_sl_dist_arr = pd.to_numeric(df.get("signal_sl_dist", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
            self._opt_dist_signal_confidence_arr = pd.to_numeric(df.get("signal_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
            self._opt_dist_signal_confidence_raw_arr = pd.to_numeric(df.get("signal_confidence_raw", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
            self._opt_dist_signal_ev_pred_arr = pd.to_numeric(df.get("signal_ev_pred", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
            self._opt_dist_signal_ev_min_req_arr = pd.to_numeric(df.get("signal_ev_min_req", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
        except Exception:
            total = len(df)
            self._opt_dist_signal_side_arr = np.zeros(total, dtype=np.int8)
            self._opt_dist_signal_tp_dist_arr = np.zeros(total, dtype=np.float32)
            self._opt_dist_signal_sl_dist_arr = np.zeros(total, dtype=np.float32)
            self._opt_dist_signal_confidence_arr = np.zeros(total, dtype=np.float32)
            self._opt_dist_signal_confidence_raw_arr = np.zeros(total, dtype=np.float32)
            self._opt_dist_signal_ev_pred_arr = np.zeros(total, dtype=np.float32)
            self._opt_dist_signal_ev_min_req_arr = np.zeros(total, dtype=np.float32)
        try:
            decision_cat = df["eval_decision"].astype("category")
            session_cat = df["eval_session"].astype("category")
            regime_cat = df["eval_runtime_regime"].astype("category")
            blocked_reason_cat = df["eval_blocked_reason"].astype("category")
            self._opt_dist_eval_decision_codes = decision_cat.cat.codes.to_numpy(dtype=np.int8, copy=True)
            self._opt_dist_eval_session_codes = session_cat.cat.codes.to_numpy(dtype=np.int8, copy=True)
            self._opt_dist_eval_regime_codes = regime_cat.cat.codes.to_numpy(dtype=np.int8, copy=True)
            self._opt_dist_eval_decision_categories = [str(val) for val in decision_cat.cat.categories.tolist()]
            self._opt_dist_eval_session_categories = [str(val) for val in session_cat.cat.categories.tolist()]
            self._opt_dist_eval_regime_categories = [str(val) for val in regime_cat.cat.categories.tolist()]
            self._opt_dist_eval_blocked_reason_codes = blocked_reason_cat.cat.codes.to_numpy(dtype=np.int32, copy=True)
            self._opt_dist_eval_blocked_reason_categories = [str(val) for val in blocked_reason_cat.cat.categories.tolist()]
        except Exception:
            total = len(df)
            self._opt_dist_eval_decision_codes = np.zeros(total, dtype=np.int8)
            self._opt_dist_eval_session_codes = np.zeros(total, dtype=np.int8)
            self._opt_dist_eval_regime_codes = np.zeros(total, dtype=np.int8)
            self._opt_dist_eval_decision_categories = list(_DIST_CACHE_DECISION_CATEGORIES)
            self._opt_dist_eval_session_categories = list(_DIST_CACHE_SESSION_CATEGORIES)
            self._opt_dist_eval_regime_categories = list(_DIST_CACHE_REGIME_CATEGORIES)
            self._opt_dist_eval_blocked_reason_codes = np.full(total, -1, dtype=np.int32)
            self._opt_dist_eval_blocked_reason_categories = []
        try:
            self._opt_dist_eval_gate_prob_arr = pd.to_numeric(df.get("eval_trade_gate_prob", np.nan), errors="coerce").to_numpy(dtype=np.float32, copy=True)
            self._opt_dist_eval_gate_threshold_arr = pd.to_numeric(df.get("eval_trade_gate_threshold", np.nan), errors="coerce").to_numpy(dtype=np.float32, copy=True)
            self._opt_dist_eval_gate_margin_min_arr = pd.to_numeric(df.get("eval_trade_gate_margin_min", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
        except Exception:
            total = len(df)
            self._opt_dist_eval_gate_prob_arr = np.full(total, np.nan, dtype=np.float32)
            self._opt_dist_eval_gate_threshold_arr = np.full(total, np.nan, dtype=np.float32)
            self._opt_dist_eval_gate_margin_min_arr = np.zeros(total, dtype=np.float32)
        self._opt_dist_signal_cache = []
        self._opt_dist_eval_cache = []
        self._opt_dist_cursor_pos = 0
        logging.info("MLPhysics OPT(dist): attached precomputed rows=%d", len(df))

    @staticmethod
    def _format_duration(seconds: Optional[float]) -> str:
        if seconds is None or not np.isfinite(seconds):
            return "unknown"
        total = int(max(0, round(float(seconds))))
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _vol_regime_tail_bars() -> int:
        scale_cfg = CONFIG.get("VOLATILITY_STD_WINDOW_SCALING", {}) or {}
        windows_cfg = CONFIG.get("VOLATILITY_STD_WINDOWS", {}) or {}
        try:
            lookback = int(scale_cfg.get("lookback", 200) or 200)
        except Exception:
            lookback = 200
        try:
            default_window = int(windows_cfg.get("default", 20) or 20)
        except Exception:
            default_window = 20
        session_windows = windows_cfg.get("sessions", {}) or {}
        max_session_window = default_window
        if isinstance(session_windows, dict):
            for value in session_windows.values():
                try:
                    w = int(value)
                except Exception:
                    continue
                if w > max_session_window:
                    max_session_window = w
        tail_bars = max(lookback + max_session_window + 5, 260)
        return int(max(100, tail_bars))

    def _reset_dist_session_cache(self) -> None:
        self._dist_session_hours_cache = None
        self._dist_session_hour_lookup_cache = None

    def _dist_session_hours_map(self) -> dict:
        if isinstance(self._dist_session_hours_cache, dict):
            return self._dist_session_hours_cache
        resolved: dict = {}
        try:
            bundle = self._dist_bundle
            if bundle is not None:
                cfg = getattr(bundle, "cfg", None)
                sess = getattr(cfg, "session_hours", None) if cfg is not None else None
                if isinstance(sess, dict):
                    resolved = sess
                    self._dist_session_hours_cache = resolved
                    return resolved
        except Exception:
            pass
        try:
            run_dir = self._dist_run_dir
            if run_dir is not None:
                cfg_path = Path(run_dir) / "config.json"
                if cfg_path.exists():
                    data = json.loads(cfg_path.read_text(encoding="utf-8"))
                    sess = data.get("session_hours")
                    if isinstance(sess, dict):
                        resolved = sess
                        self._dist_session_hours_cache = resolved
                        return resolved
        except Exception:
            pass
        self._dist_session_hours_cache = resolved
        return resolved

    def _dist_session_hour_lookup(self) -> tuple[str, ...]:
        cached = self._dist_session_hour_lookup_cache
        if isinstance(cached, tuple) and len(cached) == 24:
            return cached
        lookup = ["OTHER"] * 24
        session_hours = self._dist_session_hours_map()
        if isinstance(session_hours, dict) and session_hours:
            for name, hours in session_hours.items():
                label = str(name).upper()
                try:
                    for raw_hour in (hours or []):
                        hour = int(raw_hour) % 24
                        lookup[hour] = label
                except Exception:
                    continue
        else:
            for hour in (18, 19, 20, 21, 22, 23, 0, 1, 2):
                lookup[hour] = "ASIA"
            for hour in (3, 4, 5, 6, 7):
                lookup[hour] = "LONDON"
            for hour in (8, 9, 10, 11):
                lookup[hour] = "NY_AM"
            for hour in (12, 13, 14, 15, 16):
                lookup[hour] = "NY_PM"
        resolved = tuple(lookup)
        self._dist_session_hour_lookup_cache = resolved
        return resolved

    def _infer_dist_sessions_index(self, idx: pd.DatetimeIndex) -> np.ndarray:
        if idx is None or len(idx) == 0:
            return np.empty(0, dtype=object)
        try:
            if idx.tz is None:
                et_idx = idx.tz_localize("US/Eastern")
            else:
                et_idx = idx.tz_convert("US/Eastern")
        except Exception:
            et_idx = pd.DatetimeIndex(idx)
        hours = et_idx.hour.to_numpy(dtype=np.int16, copy=False)
        lut = np.asarray(self._dist_session_hour_lookup(), dtype=object)
        safe_hours = np.clip(hours, 0, 23)
        return lut[safe_hours]

    def _infer_dist_session(self, ts_like) -> str:
        try:
            ts = pd.Timestamp(ts_like)
        except Exception:
            return "UNKNOWN"
        if ts.tzinfo is None:
            try:
                ts = ts.tz_localize("US/Eastern")
            except Exception:
                pass
        else:
            try:
                ts = ts.tz_convert("US/Eastern")
            except Exception:
                pass
        try:
            hour = int(ts.hour)
        except Exception:
            return "UNKNOWN"
        if hour < 0 or hour > 23:
            return "UNKNOWN"
        return str(self._dist_session_hour_lookup()[hour])

    def _consume_dist_signal(
        self,
        signal,
        recent: Optional[pd.DataFrame],
        *,
        runtime_regime: Optional[str] = None,
        runtime_high_vol: Optional[bool] = None,
    ) -> Optional[Dict]:
        side = str(getattr(signal, "side", "NONE") or "NONE").upper()
        confidence_raw = float(getattr(signal, "confidence", 0.0) or 0.0)
        ev_pred = float(getattr(signal, "ev_pred", 0.0) or 0.0)
        reason_codes = list(getattr(signal, "reason_codes", []) or [])
        debug = dict(getattr(signal, "debug", {}) or {})
        session_name = str(getattr(signal, "session", "OTHER") or "OTHER").upper()
        session_name = self._dist_target_entry_session(
            recent.index[-1] if isinstance(recent, pd.DataFrame) and len(recent.index) > 0 else None,
            session_name,
        )
        hyst_state_key = f"DIST_{session_name}"
        precompute_mode = bool(self._opt_dist_precompute_mode)
        runtime_regime = str(runtime_regime).lower() if runtime_regime else "unknown"
        runtime_high_vol = bool(runtime_high_vol) if runtime_high_vol is not None else (runtime_regime == "high")
        gate_prob = float(debug.get("gate_p_take", np.nan)) if debug.get("gate_p_take") is not None else None
        gate_threshold = float(debug.get("gate_threshold", np.nan)) if debug.get("gate_threshold") is not None else None
        gate_margin = (
            float(gate_prob - gate_threshold)
            if gate_prob is not None and gate_threshold is not None
            else None
        )
        # Dist `signal.confidence` is an internal score (unbounded), not guaranteed probability.
        # Use gate probability when available, otherwise squash score into [0,1].
        if gate_prob is not None and np.isfinite(gate_prob):
            confidence_prob = float(np.clip(gate_prob, 0.0, 1.0))
        elif np.isfinite(confidence_raw):
            if 0.0 <= confidence_raw <= 1.0:
                confidence_prob = float(confidence_raw)
            else:
                confidence_prob = float(1.0 / (1.0 + np.exp(-np.clip(confidence_raw, -20.0, 20.0))))
        else:
            confidence_prob = 0.0
        disabled_regimes_cfg = CONFIG.get("ML_PHYSICS_DISABLED_REGIMES", {}) or {}
        disabled_regimes = set()
        if isinstance(disabled_regimes_cfg, dict):
            raw_disabled = (
                disabled_regimes_cfg.get(session_name)
                or disabled_regimes_cfg.get(str(session_name).upper())
                or disabled_regimes_cfg.get(str(session_name).lower())
            )
            if raw_disabled is not None:
                if isinstance(raw_disabled, (list, tuple, set)):
                    disabled_regimes = {str(x).strip().lower() for x in raw_disabled if x is not None}
                else:
                    disabled_regimes = {str(raw_disabled).strip().lower()}

        gate_margin_req = 0.0
        gate_margin_cfg = CONFIG.get("ML_PHYSICS_DIST_MIN_GATE_MARGIN", {}) or {}
        raw_margin = None
        if isinstance(gate_margin_cfg, dict):
            sessions_cfg = gate_margin_cfg.get("sessions", {}) or {}
            if isinstance(sessions_cfg, dict):
                raw_margin = (
                    sessions_cfg.get(session_name)
                    or sessions_cfg.get(str(session_name).upper())
                    or sessions_cfg.get(str(session_name).lower())
                )
            if raw_margin is None:
                raw_margin = gate_margin_cfg.get("default", 0.0)
        else:
            raw_margin = gate_margin_cfg
        try:
            gate_margin_req = max(0.0, float(raw_margin or 0.0))
        except Exception:
            gate_margin_req = 0.0

        if side not in {"LONG", "SHORT"}:
            self._hysteresis_gate(hyst_state_key, None, 0.0, 0.0)
            blocked = "gate_reject" if "GATE_REJECT" in reason_codes else (reason_codes[0].lower() if reason_codes else None)
            if precompute_mode:
                self.last_eval = {
                    "decision": "blocked" if blocked else "no_signal",
                    "blocked_reason": blocked,
                    "session": session_name,
                    "ev_source": "dist_bracket_ml",
                }
                return None
            self.last_eval = {
                "decision": "blocked" if blocked else "no_signal",
                "blocked_reason": blocked,
                "reason_codes": reason_codes,
                "session": session_name,
                "ev_source": "dist_bracket_ml",
                "confidence_raw": confidence_raw,
                "confidence_prob": confidence_prob,
                "trade_gate_prob": gate_prob,
                "trade_gate_threshold": gate_threshold,
                "trade_gate_margin": gate_margin,
                "trade_gate_margin_min": gate_margin_req,
                "runtime_regime": runtime_regime,
                "dist_run": str(getattr(self._dist_bundle, "run_dir", "")),
            }
            self._record_dist_eval(self.last_eval)
            return None

        hyst_req_conf = float(gate_threshold) if gate_threshold is not None else 0.0
        hyst_ok, hyst_reason = self._hysteresis_gate(
            hyst_state_key,
            side,
            confidence_prob,
            hyst_req_conf,
        )
        if not hyst_ok:
            blocked_codes = list(reason_codes) + [f"HYST_{str(hyst_reason).upper()}"]
            if precompute_mode:
                self.last_eval = {
                    "decision": "no_signal",
                    "blocked_reason": "hysteresis",
                    "session": session_name,
                    "ev_source": "dist_bracket_ml",
                }
                return None
            self.last_eval = {
                "decision": "no_signal",
                "blocked_reason": "hysteresis",
                "reason_codes": blocked_codes,
                "session": session_name,
                "ev_source": "dist_bracket_ml",
                "confidence_raw": confidence_raw,
                "confidence_prob": confidence_prob,
                "trade_gate_prob": gate_prob,
                "trade_gate_threshold": gate_threshold,
                "trade_gate_margin": gate_margin,
                "trade_gate_margin_min": gate_margin_req,
                "runtime_regime": runtime_regime,
                "dist_run": str(getattr(self._dist_bundle, "run_dir", "")),
            }
            self._record_dist_eval(self.last_eval)
            return None

        if runtime_regime not in {"high", "normal", "low"}:
            if precompute_mode and not disabled_regimes:
                runtime_regime = "unknown"
                runtime_high_vol = False
            else:
                try:
                    if recent is None or recent.empty:
                        raise ValueError("recent data unavailable")
                    vol_input = recent.tail(self._vol_regime_tail_bars()).set_index("ts")[["open", "high", "low", "close", "volume"]]
                    vol_regime, _, _ = volatility_filter.get_regime(vol_input)
                    if vol_regime == VolRegime.ULTRA_LOW:
                        vol_regime = VolRegime.LOW
                    if vol_regime == VolRegime.HIGH:
                        runtime_regime = "high"
                    elif vol_regime == VolRegime.NORMAL:
                        runtime_regime = "normal"
                    else:
                        runtime_regime = "low"
                    runtime_high_vol = runtime_regime == "high"
                except Exception:
                    runtime_regime = "unknown"
                    runtime_high_vol = False

        if runtime_regime in disabled_regimes:
            blocked_codes = list(reason_codes) + [f"REGIME_DISABLED_{runtime_regime.upper()}"]
            if precompute_mode:
                self.last_eval = {
                    "decision": "blocked",
                    "blocked_reason": "regime_disabled",
                    "session": session_name,
                    "ev_source": "dist_bracket_ml",
                }
                return None
            self.last_eval = {
                "decision": "blocked",
                "blocked_reason": "regime_disabled",
                "reason_codes": blocked_codes,
                "session": session_name,
                "ev_source": "dist_bracket_ml",
                "confidence_raw": confidence_raw,
                "confidence_prob": confidence_prob,
                "trade_gate_prob": gate_prob,
                "trade_gate_threshold": gate_threshold,
                "trade_gate_margin": gate_margin,
                "trade_gate_margin_min": gate_margin_req,
                "runtime_regime": runtime_regime,
                "dist_run": str(getattr(self._dist_bundle, "run_dir", "")),
            }
            self._record_dist_eval(self.last_eval)
            return None

        if gate_margin_req > 0.0 and gate_margin is not None and gate_margin < gate_margin_req:
            blocked_codes = list(reason_codes) + ["GATE_MARGIN_SHORTFALL"]
            if precompute_mode:
                self.last_eval = {
                    "decision": "blocked",
                    "blocked_reason": "gate_margin_shortfall",
                    "session": session_name,
                    "ev_source": "dist_bracket_ml",
                }
                return None
            self.last_eval = {
                "decision": "blocked",
                "blocked_reason": "gate_margin_shortfall",
                "reason_codes": blocked_codes,
                "session": session_name,
                "ev_source": "dist_bracket_ml",
                "confidence_raw": confidence_raw,
                "confidence_prob": confidence_prob,
                "trade_gate_prob": gate_prob,
                "trade_gate_threshold": gate_threshold,
                "trade_gate_margin": gate_margin,
                "trade_gate_margin_min": gate_margin_req,
                "runtime_regime": runtime_regime,
                "dist_run": str(getattr(self._dist_bundle, "run_dir", "")),
            }
            self._record_dist_eval(self.last_eval)
            return None

        tick_size = float(getattr(self._dist_bundle.cfg, "tick_size", CONFIG.get("TICK_SIZE", 0.25)) or 0.25)
        tp_ticks = int(getattr(signal, "tp_ticks", 0) or 0)
        sl_ticks = int(getattr(signal, "sl_ticks", 0) or 0)
        tp_dist = float(max(0.0, tp_ticks * tick_size))
        sl_dist = float(max(0.0, sl_ticks * tick_size))
        tp_dist, sl_dist = self._dist_apply_min_brackets(tp_dist, sl_dist, tick_size)
        tp_dist, sl_dist, _ = self._dist_normal_bracket_policy(
            session_name=session_name,
            runtime_regime=runtime_regime,
            tick_size=tick_size,
            tp_dist=tp_dist,
            sl_dist=sl_dist,
        )
        current_ts = None
        if isinstance(recent, pd.DataFrame) and len(recent.index) > 0:
            try:
                current_ts = pd.Timestamp(recent.index[-1])
            except Exception:
                current_ts = None
        runtime_blocked_reason = self._dist_runtime_signal_block_reason(
            side=side,
            session_name=session_name,
            tp_dist=tp_dist,
            sl_dist=sl_dist,
            confidence_prob=confidence_prob,
            tick_size=tick_size,
            ev_abs=abs(float(ev_pred)),
            runtime_regime=runtime_regime,
            runtime_high_vol=runtime_high_vol,
            current_time=current_ts,
        )
        if runtime_blocked_reason:
            blocked_codes = list(reason_codes) + [runtime_blocked_reason.upper()]
            self.last_eval = {
                "decision": "blocked",
                "blocked_reason": runtime_blocked_reason,
                "reason_codes": blocked_codes,
                "session": session_name,
                "ev_source": "dist_bracket_ml",
                "confidence_raw": confidence_raw,
                "confidence_prob": confidence_prob,
                "trade_gate_prob": gate_prob,
                "trade_gate_threshold": gate_threshold,
                "trade_gate_margin": gate_margin,
                "trade_gate_margin_min": gate_margin_req,
                "runtime_regime": runtime_regime,
                "dist_run": str(getattr(self._dist_bundle, "run_dir", "")),
            }
            if not precompute_mode:
                self._record_dist_eval(self.last_eval)
            return None

        is_runner = bool(self._dist_is_wide_bracket(tp_dist, sl_dist))
        entry_policy_blocked_reason = self._dist_entry_policy_block_reason(
            side=side,
            session_name=session_name,
            ev_abs=abs(float(ev_pred)),
            gate_prob=gate_prob,
            runtime_regime=runtime_regime,
            is_runner=is_runner,
            current_time=current_ts,
        )
        if entry_policy_blocked_reason:
            blocked_codes = list(reason_codes) + [entry_policy_blocked_reason.upper()]
            self.last_eval = {
                "decision": "blocked",
                "blocked_reason": entry_policy_blocked_reason,
                "reason_codes": blocked_codes,
                "session": session_name,
                "ev_source": "dist_bracket_ml",
                "confidence_raw": confidence_raw,
                "confidence_prob": confidence_prob,
                "trade_gate_prob": gate_prob,
                "trade_gate_threshold": gate_threshold,
                "trade_gate_margin": gate_margin,
                "trade_gate_margin_min": gate_margin_req,
                "runtime_regime": runtime_regime,
                "dist_run": str(getattr(self._dist_bundle, "run_dir", "")),
            }
            if not precompute_mode:
                self._record_dist_eval(self.last_eval)
            return None

        prob_proxy = float(np.clip(confidence_prob, 0.0, 1.0))
        prob_up = prob_proxy if side == "LONG" else (1.0 - prob_proxy)
        prob_down = prob_proxy if side == "SHORT" else (1.0 - prob_proxy)

        self.last_eval = {
            "decision": "signal_long" if side == "LONG" else "signal_short",
            "blocked_reason": None,
            "session": session_name,
            "reason_codes": reason_codes,
            "ev_source": "dist_bracket_ml",
            "confidence_raw": confidence_raw,
            "confidence_prob": confidence_prob,
            "ev_long": ev_pred if side == "LONG" else -ev_pred,
            "ev_short": ev_pred if side == "SHORT" else -ev_pred,
            "trade_gate_prob": gate_prob,
            "trade_gate_threshold": gate_threshold,
            "trade_gate_margin": gate_margin,
            "trade_gate_margin_min": gate_margin_req,
            "runtime_regime": runtime_regime,
            "dist_run": str(getattr(self._dist_bundle, "run_dir", "")),
            "hysteresis_reason": hyst_reason,
        }
        self._record_dist_eval(self.last_eval)

        return self._normalize_dist_signal_payload({
            "strategy": f"MLPhysics_{session_name}",
            "side": side,
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "ml_is_runner": is_runner,
            "ml_confidence": confidence_prob,
            "ml_confidence_raw": confidence_raw,
            "ml_threshold": (
                float(gate_threshold)
                if gate_threshold is not None and np.isfinite(gate_threshold)
                else 0.0
            ),
            "ml_prob_up": prob_up,
            "ml_prob_down": prob_down,
            "ml_short_threshold": (
                float(gate_threshold)
                if gate_threshold is not None and np.isfinite(gate_threshold)
                else 0.0
            ),
            "ml_regime": runtime_regime,
            "ml_high_vol": runtime_high_vol,
            "ml_candidate_side": side,
            "ml_margin": float(debug.get("gate_p_take", 0.0) - debug.get("gate_threshold", 0.0))
            if debug.get("gate_p_take") is not None and debug.get("gate_threshold") is not None
            else 0.0,
            "ml_margin_req": gate_margin_req,
            "ml_gate_min_conf": float(debug.get("gate_threshold", 0.0) or 0.0),
            "ml_gate_block_sides": "",
            "ml_trade_gate_prob": float(debug.get("gate_p_take", 0.0) or 0.0),
            "ml_trade_gate_threshold": float(debug.get("gate_threshold", 0.0) or 0.0),
            "ml_trade_gate_margin": float(debug.get("gate_p_take", 0.0) - debug.get("gate_threshold", 0.0))
            if debug.get("gate_p_take") is not None and debug.get("gate_threshold") is not None
            else 0.0,
            "ml_trade_gate_soft_penalty": 0.0,
            "ml_trade_gate_policy": "dist_gate",
            "ml_trade_gate_required": True,
            "ml_budget_cov_recent": None,
            "ml_ev_long": ev_pred if side == "LONG" else -ev_pred,
            "ml_ev_short": ev_pred if side == "SHORT" else -ev_pred,
            "ml_ev_long_effective": ev_pred if side == "LONG" else -ev_pred,
            "ml_ev_short_effective": ev_pred if side == "SHORT" else -ev_pred,
            "ml_ev_source": "dist_bracket_ml",
            "ml_ev_min_req": float(self._dist_bundle.cfg.decision.ev_min),
            "ml_ev_min_req_effective": float(self._dist_bundle.cfg.decision.ev_min),
            "ml_ev_prob_edge": 0.0,
            "ml_ev_prob_edge_req": 0.0,
            "ml_decision": self.last_eval.get("decision"),
            "ml_blocked_reason": self.last_eval.get("blocked_reason"),
        })

    def _normalize_dist_signal_payload(self, payload: Optional[Dict]) -> Optional[Dict]:
        if not isinstance(payload, dict):
            return payload
        out = dict(payload)

        def _to_float(val) -> Optional[float]:
            try:
                f = float(val)
            except Exception:
                return None
            if not np.isfinite(f):
                return None
            return float(f)

        side = str(out.get("side", "") or "").upper()
        conf_raw = _to_float(out.get("ml_confidence"))
        prob_up = _to_float(out.get("ml_prob_up"))
        prob_down = _to_float(out.get("ml_prob_down"))
        gate_prob = _to_float(out.get("ml_trade_gate_prob"))
        gate_thr = _to_float(out.get("ml_trade_gate_threshold"))

        conf_prob: Optional[float] = None
        if gate_prob is not None:
            conf_prob = float(np.clip(gate_prob, 0.0, 1.0))
        elif side == "LONG" and prob_up is not None and 0.0 <= prob_up <= 1.0:
            conf_prob = prob_up
        elif side == "SHORT" and prob_down is not None and 0.0 <= prob_down <= 1.0:
            conf_prob = prob_down
        elif conf_raw is not None:
            if 0.0 <= conf_raw <= 1.0:
                conf_prob = conf_raw
            else:
                conf_prob = float(1.0 / (1.0 + np.exp(-np.clip(conf_raw, -20.0, 20.0))))

        if conf_prob is not None:
            out["ml_confidence_raw"] = conf_raw if conf_raw is not None else conf_prob
            out["ml_confidence"] = float(np.clip(conf_prob, 0.0, 1.0))
            if side == "LONG":
                out["ml_prob_up"] = out["ml_confidence"]
                out["ml_prob_down"] = float(1.0 - out["ml_confidence"])
            elif side == "SHORT":
                out["ml_prob_down"] = out["ml_confidence"]
                out["ml_prob_up"] = float(1.0 - out["ml_confidence"])

        if gate_thr is not None:
            thr = float(np.clip(gate_thr, 0.0, 1.0))
            out["ml_threshold"] = thr
            out["ml_short_threshold"] = thr
        else:
            for k in ("ml_threshold", "ml_short_threshold"):
                cur = _to_float(out.get(k))
                if cur is not None:
                    out[k] = float(np.clip(cur, 0.0, 1.0))

        return out

    def precompute_dist_backtest_signals(
        self,
        full_df: pd.DataFrame,
        *,
        progress_every: int = 1000,
        progress_min_interval_sec: float = 5.0,
        status_cb: Optional[Callable[[dict], None]] = None,
        allow_build: bool = True,
    ) -> bool:
        if not (self._opt_enabled and self._opt_mode == "backtest" and self._dist_mode):
            return False
        if full_df is None or full_df.empty:
            return False

        def _emit_status(message: str, **extra) -> None:
            if not callable(status_cb):
                return
            payload = {"type": "status", "message": str(message)}
            if extra:
                payload.update(extra)
            try:
                status_cb(payload)
            except Exception:
                pass

        idx = pd.DatetimeIndex(full_df.index)
        if len(idx) == 0:
            return False
        if idx.tz is None:
            idx = idx.tz_localize("US/Eastern")
            full_df = full_df.copy()
            full_df.index = idx
        else:
            idx = idx.tz_convert("US/Eastern")
            if not idx.equals(full_df.index):
                full_df = full_df.copy()
                full_df.index = idx
        session_labels = self._infer_dist_sessions_index(idx)

        cache_path = self._dist_cache_path(full_df)
        overwrite = bool(self._opt_cfg.get("overwrite_cache", False))
        allow_cache = bool(self._opt_cfg.get("prediction_cache", True))
        explicit_cache_path = self._dist_explicit_cache_path()
        explicit_strict = bool(self._opt_cfg.get("dist_precomputed_strict", False))
        req_source_key = self._dist_source_key_from_df(full_df)
        req_source_label = self._dist_source_label_from_df(full_df)
        req_symbol_hint = self._dist_symbol_hint_from_df(full_df)
        req_symbol_mode = self._dist_symbol_mode_hint_from_df(full_df)
        _emit_status("MLPhysics(dist): checking precompute cache...")

        def _try_attach_cache(candidate: Path, *, is_explicit: bool = False) -> bool:
            if not candidate.exists():
                return False
            try:
                cached = pd.read_parquet(candidate)
                cache_source_key = ""
                cache_source_label = ""
                cache_symbol_hint = ""
                cache_symbol_mode = ""
                legacy_input = False
                legacy_rewrite_safe = False
                if isinstance(cached, pd.DataFrame) and "cache_source_key" in cached.columns:
                    try:
                        source_vals = cached["cache_source_key"].dropna().astype(str)
                        if not source_vals.empty:
                            cache_source_key = self._sanitize_token(source_vals.iloc[0])
                    except Exception:
                        cache_source_key = ""
                if isinstance(cached, pd.DataFrame):
                    legacy_input = "signal_side" not in cached.columns or "eval_decision" not in cached.columns
                    legacy_rewrite_safe = bool(legacy_input)
                    if "cache_source_label" in cached.columns:
                        try:
                            label_vals = cached["cache_source_label"].dropna().astype(str)
                            if not label_vals.empty:
                                cache_source_label = str(label_vals.iloc[0]).strip()
                        except Exception:
                            cache_source_label = ""
                    if "cache_symbol_hint" in cached.columns:
                        try:
                            symbol_vals = cached["cache_symbol_hint"].dropna().astype(str)
                            if not symbol_vals.empty:
                                cache_symbol_hint = str(symbol_vals.iloc[0]).strip()
                        except Exception:
                            cache_symbol_hint = ""
                    if "cache_symbol_mode" in cached.columns:
                        try:
                            mode_vals = cached["cache_symbol_mode"].dropna().astype(str)
                            if not mode_vals.empty:
                                cache_symbol_mode = str(mode_vals.iloc[0]).strip()
                        except Exception:
                            cache_symbol_mode = ""
                if req_source_key and cache_source_key and cache_source_key != req_source_key:
                    logging.info(
                        "MLPhysics OPT(dist): skipping cache %s due to source mismatch (cache=%s req=%s)",
                        candidate,
                        cache_source_key,
                        req_source_key,
                    )
                    _emit_status(
                        f"MLPhysics(dist): cache source mismatch for {candidate.name}; trying next...",
                        phase="ml_dist_cache_source_mismatch",
                    )
                    return False
                if isinstance(cached, pd.DataFrame):
                    cached = cached.drop(
                        columns=[
                            "cache_source_key",
                            "cache_source_label",
                            "cache_symbol_hint",
                            "cache_symbol_mode",
                        ],
                        errors="ignore",
                    )
                self.set_precomputed_dist_backtest_df(cached)
                if self._opt_dist_precomputed_df is None or len(self._opt_dist_precomputed_df) <= 0:
                    return False
                attached_idx = pd.DatetimeIndex(self._opt_dist_precomputed_df.index)
                if attached_idx.tz is None:
                    attached_idx = attached_idx.tz_localize("US/Eastern")
                else:
                    attached_idx = attached_idx.tz_convert("US/Eastern")
                if len(attached_idx) != len(idx) or not attached_idx.equals(idx):
                    legacy_rewrite_safe = False
                    missing = idx.difference(attached_idx)
                    extra = attached_idx.difference(idx)
                    if len(missing) > 0:
                        allow_partial = bool(self._opt_cfg.get("dist_precomputed_allow_partial", True))
                        try:
                            max_missing_abs = int(self._opt_cfg.get("dist_precomputed_allow_partial_max_missing", 2000) or 2000)
                        except Exception:
                            max_missing_abs = 2000
                        try:
                            max_missing_ratio = float(
                                self._opt_cfg.get("dist_precomputed_allow_partial_max_ratio", 0.01) or 0.01
                            )
                        except Exception:
                            max_missing_ratio = 0.01
                        try:
                            min_coverage_ratio = float(
                                self._opt_cfg.get("dist_precomputed_allow_partial_min_coverage", 0.98) or 0.98
                            )
                        except Exception:
                            min_coverage_ratio = 0.98
                        attached_min = attached_idx.min() if len(attached_idx) else None
                        attached_max = attached_idx.max() if len(attached_idx) else None
                        missing_leading_only = bool(attached_min is not None and len(missing) > 0 and missing.max() < attached_min)
                        missing_trailing_only = bool(attached_max is not None and len(missing) > 0 and missing.min() > attached_max)
                        missing_ratio = (float(len(missing)) / float(len(idx))) if len(idx) > 0 else 1.0
                        coverage_ratio = max(0.0, 1.0 - missing_ratio)
                        can_partial_align = (
                            allow_partial
                            and (
                                (
                                    (missing_leading_only or missing_trailing_only)
                                    and len(missing) <= max_missing_abs
                                    and missing_ratio <= max_missing_ratio
                                )
                                or coverage_ratio >= min_coverage_ratio
                            )
                        )
                        if can_partial_align:
                            try:
                                aligned = self._opt_dist_precomputed_df.reindex(idx)
                                if len(session_labels) == len(aligned):
                                    aligned_sessions = session_labels
                                else:
                                    aligned_sessions = self._infer_dist_sessions_index(pd.DatetimeIndex(aligned.index))
                                aligned = self._prepare_dist_precomputed_cache_df(
                                    aligned,
                                    session_labels=aligned_sessions,
                                    partial_gap_reason="partial_cache_gap",
                                )
                                if aligned is None:
                                    raise ValueError("failed to normalize aligned cache")
                                self._opt_dist_precomputed_df = aligned
                                logging.info(
                                    "MLPhysics OPT(dist): partial cache align for %s "
                                    "(missing=%d extra=%d miss_ratio=%.4f coverage=%.4f "
                                    "leading_only=%s trailing_only=%s)",
                                    candidate,
                                    len(missing),
                                    len(extra),
                                    missing_ratio,
                                    coverage_ratio,
                                    missing_leading_only,
                                    missing_trailing_only,
                                )
                            except Exception as exc:
                                logging.warning(
                                    "MLPhysics OPT(dist): failed partial alignment for %s (%s); skipping candidate",
                                    candidate,
                                    exc,
                                )
                                self._opt_dist_precomputed_df = None
                                return False
                        else:
                            logging.info(
                                "MLPhysics OPT(dist): cache index mismatch for %s "
                                "(missing=%d extra=%d cache=%d backtest=%d ratio=%.4f); skipping candidate",
                                candidate,
                                len(missing),
                                len(extra),
                                len(attached_idx),
                                len(idx),
                                missing_ratio,
                            )
                            _emit_status(
                                f"MLPhysics(dist): cache mismatch for {candidate.name} "
                                f"(missing={len(missing)} ratio={missing_ratio:.4f}); trying next...",
                                phase="ml_dist_cache_mismatch",
                                missing=int(len(missing)),
                                missing_ratio=float(missing_ratio),
                            )
                            self._opt_dist_precomputed_df = None
                            return False
                    else:
                        try:
                            aligned = self._opt_dist_precomputed_df.loc[idx]
                            self._opt_dist_precomputed_df = aligned
                            logging.info(
                                "MLPhysics OPT(dist): aligned superset cache %s to requested index (%d rows)",
                                candidate,
                                len(aligned),
                            )
                        except Exception as exc:
                            logging.warning(
                                "MLPhysics OPT(dist): failed aligning superset cache %s (%s); ignoring",
                                candidate,
                                exc,
                            )
                            self._opt_dist_precomputed_df = None
                            return False
                if legacy_rewrite_safe and isinstance(self._opt_dist_precomputed_df, pd.DataFrame):
                    try:
                        rewritten = self._opt_dist_precomputed_df.copy()
                        if cache_source_key:
                            rewritten["cache_source_key"] = cache_source_key
                        if cache_source_label:
                            rewritten["cache_source_label"] = cache_source_label
                        if cache_symbol_hint:
                            rewritten["cache_symbol_hint"] = cache_symbol_hint
                        if cache_symbol_mode:
                            rewritten["cache_symbol_mode"] = cache_symbol_mode
                        rewritten.to_parquet(candidate, engine="pyarrow", compression="zstd")
                        logging.info("MLPhysics OPT(dist): rewrote legacy cache %s to compact format", candidate)
                    except Exception as exc:
                        logging.warning(
                            "MLPhysics OPT(dist): failed rewriting legacy cache %s (%s)",
                            candidate,
                            exc,
                        )
                if is_explicit:
                    logging.info("MLPhysics OPT(dist): loaded explicit precompute cache %s", candidate)
                elif candidate != cache_path:
                    logging.info(
                        "MLPhysics OPT(dist): loaded compatible cache %s (preferred path %s)",
                        candidate,
                        cache_path,
                    )
                else:
                    logging.info("MLPhysics OPT(dist): loaded cache %s", cache_path)
                _emit_status(
                    f"MLPhysics(dist): cache hit ({candidate.name})",
                    phase="ml_dist_cache_hit",
                    done=int(len(idx)),
                    total=int(len(idx)),
                )
                return True
            except Exception as exc:
                logging.warning("MLPhysics OPT(dist): failed loading cache %s (%s)", candidate, exc)
                self._opt_dist_precomputed_df = None
                return False

        if allow_cache and not overwrite and explicit_cache_path is not None:
            _emit_status("MLPhysics(dist): checking explicit precompute cache...")
            if _try_attach_cache(explicit_cache_path, is_explicit=True):
                return True
            if explicit_strict:
                logging.warning(
                    "MLPhysics OPT(dist): explicit precompute cache is configured but unusable (%s); strict mode prevents fallback",
                    explicit_cache_path,
                )
                return False

        if allow_cache and not overwrite:
            for candidate in self._dist_cache_candidates(full_df, cache_path):
                if explicit_cache_path is not None and candidate == explicit_cache_path:
                    continue
                if _try_attach_cache(candidate, is_explicit=False):
                    return True

        if not bool(allow_build):
            try:
                self.set_precomputed_dist_backtest_df(None)
            except Exception:
                pass
            _emit_status(
                "MLPhysics(dist): cache miss; using per-bar inference.",
                phase="ml_dist_cache_miss",
            )
            return False

        if self._dist_bundle is None and not self._ensure_dist_bundle_loaded():
            logging.warning("MLPhysics OPT(dist): no cache hit and dist bundle is unavailable")
            _emit_status(
                "MLPhysics(dist): cache miss and bundle unavailable; falling back to per-bar dist inference.",
                phase="ml_dist_precompute_unavailable",
            )
            return False
        _emit_status("MLPhysics(dist): cache miss; building precompute table...")

        tf_minutes = int(CONFIG.get("ML_PHYSICS_TIMEFRAME_MINUTES", 1) or 1)
        align_mode = str(CONFIG.get("ML_PHYSICS_BAR_ALIGNMENT", "open") or "open").lower()
        if align_mode not in {"open", "close"}:
            align_mode = "open"
        max_bars = int(self._dist_input_max_bars or 0)

        total = len(idx)
        if total <= 0:
            return False
        progress_every = max(1, int(progress_every or 1))
        progress_min_interval_sec = max(0.0, float(progress_min_interval_sec or 0.0))

        started_at = time.perf_counter()
        last_progress_log_at = started_at
        next_progress_due = progress_every

        def _log_progress(done: int, *, force: bool = False) -> None:
            nonlocal last_progress_log_at, next_progress_due
            if not force and done < next_progress_due:
                return
            now = time.perf_counter()
            if not force and progress_min_interval_sec > 0 and (now - last_progress_log_at) < progress_min_interval_sec:
                return
            elapsed = max(0.0, now - started_at)
            rate = (done / elapsed) if elapsed > 1e-9 else 0.0
            remaining = max(0, total - done)
            eta = (remaining / rate) if rate > 1e-9 else None
            pct = (100.0 * done / total) if total > 0 else 100.0
            logging.info(
                "MLPhysics OPT(dist) precompute progress: %.1f%% (%d/%d) elapsed=%s eta=%s rate=%.1f bars/s",
                pct,
                done,
                total,
                self._format_duration(elapsed),
                self._format_duration(eta),
                rate,
            )
            _emit_status(
                f"MLPhysics(dist) precompute {pct:.1f}% ({done}/{total}) "
                f"elapsed={self._format_duration(elapsed)} eta={self._format_duration(eta)}",
                phase="ml_dist_precompute",
                done=int(done),
                total=int(total),
                pct=float(pct),
            )
            last_progress_log_at = now
            if not force:
                next_progress_due = (int(done // progress_every) + 1) * progress_every

        has_signal = np.zeros(total, dtype=np.int8)
        signal_side = np.zeros(total, dtype=np.int8)
        signal_tp_dist = np.zeros(total, dtype=np.float32)
        signal_sl_dist = np.zeros(total, dtype=np.float32)
        signal_confidence = np.zeros(total, dtype=np.float32)
        signal_confidence_raw = np.zeros(total, dtype=np.float32)
        signal_ev_pred = np.zeros(total, dtype=np.float32)
        signal_ev_min_req = np.zeros(total, dtype=np.float32)
        eval_decision = np.full(total, "no_signal", dtype=object)
        eval_session = np.full(total, "UNKNOWN", dtype=object)
        eval_blocked_reason = np.full(total, "", dtype=object)
        eval_runtime_regime = np.full(total, "unknown", dtype=object)
        eval_gate_prob = np.full(total, np.nan, dtype=np.float32)
        eval_gate_threshold = np.full(total, np.nan, dtype=np.float32)
        eval_gate_margin_min = np.zeros(total, dtype=np.float32)

        def _default_session(i: int) -> str:
            if 0 <= i < total and i < len(session_labels):
                return str(session_labels[i]).upper()
            return "UNKNOWN"

        def _set_eval_state(
            i: int,
            decision: str,
            blocked_reason: Optional[str] = None,
            *,
            session_name: Optional[str] = None,
            runtime_regime: Optional[str] = None,
            gate_prob: float = np.nan,
            gate_threshold: float = np.nan,
            gate_margin_min: float = 0.0,
        ) -> None:
            eval_decision[i] = str(decision or "no_signal").lower()
            eval_blocked_reason[i] = str(blocked_reason or "").strip()
            eval_session[i] = str(session_name or _default_session(i)).upper()
            eval_runtime_regime[i] = str(runtime_regime or "unknown").lower()
            eval_gate_prob[i] = np.float32(gate_prob if np.isfinite(gate_prob) else np.nan)
            eval_gate_threshold[i] = np.float32(gate_threshold if np.isfinite(gate_threshold) else np.nan)
            eval_gate_margin_min[i] = np.float32(max(0.0, self._finite_float(gate_margin_min, 0.0)))

        def _apply_eval_payload(i: int, payload: Optional[dict]) -> None:
            if not isinstance(payload, dict):
                return
            _set_eval_state(
                i,
                str(payload.get("decision") or "no_signal"),
                payload.get("blocked_reason"),
                session_name=str(payload.get("session") or _default_session(i)).upper(),
                runtime_regime=str(payload.get("runtime_regime") or "unknown").lower(),
                gate_prob=self._finite_float(payload.get("trade_gate_prob"), np.nan),
                gate_threshold=self._finite_float(payload.get("trade_gate_threshold"), np.nan),
                gate_margin_min=self._finite_float(payload.get("trade_gate_margin_min"), 0.0),
            )

        def _apply_signal_payload(i: int, payload: Optional[dict]) -> None:
            if not isinstance(payload, dict):
                return
            normalized = self._normalize_dist_signal_payload(payload)
            if not isinstance(normalized, dict):
                return
            side = str(normalized.get("side") or "").upper()
            side_code = self._dist_signal_side_code(side)
            if int(side_code) == 0:
                return
            has_signal[i] = np.int8(1)
            signal_side[i] = side_code
            signal_tp_dist[i] = np.float32(self._finite_float(normalized.get("tp_dist"), 0.0))
            signal_sl_dist[i] = np.float32(self._finite_float(normalized.get("sl_dist"), 0.0))
            signal_confidence[i] = np.float32(self._finite_float(normalized.get("ml_confidence"), 0.0))
            signal_confidence_raw[i] = np.float32(
                self._finite_float(
                    normalized.get("ml_confidence_raw"),
                    self._finite_float(normalized.get("ml_confidence"), 0.0),
                )
            )
            signal_ev_min_req[i] = np.float32(self._finite_float(normalized.get("ml_ev_min_req"), 0.0))
            if side == "LONG":
                signal_ev_pred[i] = np.float32(self._finite_float(normalized.get("ml_ev_long"), 0.0))
            else:
                signal_ev_pred[i] = np.float32(self._finite_float(normalized.get("ml_ev_short"), 0.0))
            if str(eval_decision[i]) not in {"signal_long", "signal_short"}:
                _set_eval_state(
                    i,
                    "signal_long" if side == "LONG" else "signal_short",
                    None,
                    session_name=str(eval_session[i] or _default_session(i)).upper(),
                    runtime_regime=str(eval_runtime_regime[i] or "unknown").lower(),
                    gate_prob=float(eval_gate_prob[i]),
                    gate_threshold=float(eval_gate_threshold[i]),
                    gate_margin_min=float(eval_gate_margin_min[i]),
                )

        prior_hyst = dict(self._hyst_state)
        prior_last_eval = self.last_eval
        self._hyst_state = {}
        self.last_eval = None
        self._opt_dist_precompute_mode = True

        infer_errors = 0
        try:
            _log_progress(0, force=True)
            can_fast = bool(callable(_dist_prepare_runtime_features) and callable(_dist_predict_from_feature_row))

            lower_cols = {str(col).strip().lower(): col for col in full_df.columns}
            open_col = lower_cols.get("open")
            high_col = lower_cols.get("high")
            low_col = lower_cols.get("low")
            close_col = lower_cols.get("close")
            volume_col = lower_cols.get("volume")
            ts_col = lower_cols.get("ts")

            if not all((open_col, high_col, low_col, close_col)):
                for i in range(total):
                    _set_eval_state(i, "no_signal", "missing_ohlc_columns")
                    _log_progress(i + 1)
            else:
                ts_series = full_df[ts_col] if ts_col is not None else full_df.index
                volume_series = full_df[volume_col] if volume_col is not None else pd.Series(0.0, index=full_df.index)
                recent_runtime = pd.DataFrame(
                    {
                        "ts": ts_series,
                        "open": full_df[open_col],
                        "high": full_df[high_col],
                        "low": full_df[low_col],
                        "close": full_df[close_col],
                        "volume": volume_series,
                    }
                ).reset_index(drop=True)

                feat_df = None
                if can_fast:
                    try:
                        feat_df = _dist_prepare_runtime_features(self._dist_bundle, recent_runtime)
                    except Exception as exc:
                        can_fast = False
                        logging.warning("MLPhysics OPT(dist): fast feature prep failed, falling back (%s)", exc)

                if tf_minutes > 1:
                    minute_arr = idx.minute.to_numpy(dtype=np.int16, copy=False)
                    if align_mode == "close":
                        aligned_mask = (minute_arr % tf_minutes) == (tf_minutes - 1)
                        row_indices = np.arange(total, dtype=np.int64)
                    else:
                        aligned_mask = (minute_arr % tf_minutes) == 0
                        row_indices = np.arange(total, dtype=np.int64) - 1
                else:
                    aligned_mask = np.ones(total, dtype=bool)
                    row_indices = np.arange(total, dtype=np.int64)

                if max_bars > 0:
                    starts = np.maximum(0, np.arange(total, dtype=np.int64) + 1 - max_bars)
                else:
                    starts = np.zeros(total, dtype=np.int64)

                min_hist = 220
                atr_col = str(getattr(self._dist_bundle.cfg.targets, "atr_col", "atr_14"))
                valid_mask = None
                feat_ts_pos_lookup = None
                disabled_regimes_cfg = CONFIG.get("ML_PHYSICS_DISABLED_REGIMES", {}) or {}
                need_signal_recent_slice = bool(disabled_regimes_cfg)
                vol_tail_bars = self._vol_regime_tail_bars() if need_signal_recent_slice else 0
                feat_runtime_df = None
                feat_ready = bool(can_fast and isinstance(feat_df, pd.DataFrame) and len(feat_df) > 0)
                if feat_ready:
                    gate_hint_cols = [
                        "session",
                        atr_col,
                        "atr_14",
                        "atr_28",
                        "dvwap",
                        "range_over_atr_21",
                        "inside_bar_count_13",
                        "dist_ema_20",
                        "dist_ema_50",
                        "dist_ema_200",
                        "ts",
                    ]
                    runtime_cols = list(dict.fromkeys(list(self._dist_bundle.feature_cols) + gate_hint_cols))
                    feat_runtime_df = feat_df.reindex(columns=runtime_cols)
                    try:
                        valid_mask = (
                            feat_runtime_df[self._dist_bundle.feature_cols + [atr_col]]
                            .notna()
                            .all(axis=1)
                            .to_numpy(dtype=bool, copy=False)
                        )
                    except Exception:
                        valid_mask = np.ones(len(feat_runtime_df), dtype=bool)
                    if len(feat_runtime_df) != len(recent_runtime):
                        try:
                            ts_vals = pd.to_datetime(feat_runtime_df["ts"], errors="coerce")
                            ts_idx = pd.DatetimeIndex(ts_vals)
                            if ts_idx.tz is None:
                                ts_idx = ts_idx.tz_localize("US/Eastern")
                            else:
                                ts_idx = ts_idx.tz_convert("US/Eastern")
                            feat_ts_pos_lookup = {}
                            for pos, ts_item in enumerate(ts_idx):
                                if pd.notna(ts_item):
                                    feat_ts_pos_lookup[pd.Timestamp(ts_item)] = int(pos)
                        except Exception:
                            feat_ts_pos_lookup = None

                if tf_minutes > 1 and not bool(np.all(aligned_mask)):
                    not_aligned_idx = np.flatnonzero(~aligned_mask)
                    for i in not_aligned_idx:
                        _set_eval_state(int(i), "no_signal", "tf_alignment")
                    aligned_indices = np.flatnonzero(aligned_mask)
                else:
                    aligned_indices = np.arange(total, dtype=np.int64)

                for i_val in aligned_indices:
                    i = int(i_val)
                    eval_payload = None
                    signal_payload = None
                    row_idx = int(row_indices[i])
                    start_idx = int(starts[i])
                    hist_len = row_idx - start_idx + 1
                    if row_idx < start_idx or row_idx < 0 or hist_len < min_hist:
                        _set_eval_state(i, "no_signal", "insufficient_history")
                    elif feat_ready and isinstance(feat_runtime_df, pd.DataFrame):
                        feat_row = None
                        if feat_ts_pos_lookup is not None:
                            ts_key = idx[row_idx]
                            pos = feat_ts_pos_lookup.get(ts_key)
                            if pos is not None and (valid_mask is None or bool(valid_mask[pos])):
                                feat_row = feat_runtime_df.iloc[int(pos)]
                        elif row_idx < len(feat_runtime_df):
                            if valid_mask is None or bool(valid_mask[row_idx]):
                                feat_row = feat_runtime_df.iloc[row_idx]

                        if feat_row is None:
                            _set_eval_state(i, "no_signal", "insufficient_history")
                        else:
                            try:
                                signal = _dist_predict_from_feature_row(self._dist_bundle, feat_row)
                                predicted_side = str(getattr(signal, "side", "NONE") or "NONE").upper()
                                recent_slice = None
                                if need_signal_recent_slice and predicted_side in {"LONG", "SHORT"}:
                                    vol_start = max(0, row_idx - vol_tail_bars + 1)
                                    recent_slice = recent_runtime.iloc[vol_start : row_idx + 1]
                                maybe_signal = self._consume_dist_signal(signal, recent_slice)
                                if isinstance(maybe_signal, dict):
                                    signal_payload = maybe_signal
                                if isinstance(self.last_eval, dict):
                                    eval_payload = self.last_eval
                            except Exception as exc:
                                infer_errors += 1
                                if infer_errors <= 3:
                                    logging.warning("MLPhysics OPT(dist): fast inference failure at %s (%s)", idx[i], exc)
                                _set_eval_state(i, "no_signal", "dist_inference_error")
                    else:
                        # Conservative fallback if fast runtime entrypoints are unavailable.
                        hist_df = full_df.iloc[start_idx : i + 1]
                        try:
                            maybe_signal = self._on_bar_dist(hist_df, idx[i])
                        except Exception:
                            maybe_signal = None
                        if isinstance(maybe_signal, dict):
                            signal_payload = maybe_signal
                        if isinstance(self.last_eval, dict):
                            eval_payload = self.last_eval
                        elif not eval_payload:
                            _set_eval_state(i, "no_signal", "dist_inference_error")

                    _apply_eval_payload(i, eval_payload)
                    if isinstance(signal_payload, dict):
                        _apply_signal_payload(i, signal_payload)
                    _log_progress(i + 1)
        finally:
            self._opt_dist_precompute_mode = False
            self._hyst_state = prior_hyst
            self.last_eval = prior_last_eval
            _log_progress(total, force=True)

        cache_df = pd.DataFrame(
            {
                "has_signal": has_signal.astype(np.int8, copy=False),
                "signal_side": signal_side.astype(np.int8, copy=False),
                "signal_tp_dist": signal_tp_dist.astype(np.float32, copy=False),
                "signal_sl_dist": signal_sl_dist.astype(np.float32, copy=False),
                "signal_confidence": signal_confidence.astype(np.float32, copy=False),
                "signal_confidence_raw": signal_confidence_raw.astype(np.float32, copy=False),
                "signal_ev_pred": signal_ev_pred.astype(np.float32, copy=False),
                "signal_ev_min_req": signal_ev_min_req.astype(np.float32, copy=False),
                "eval_decision": eval_decision,
                "eval_session": eval_session,
                "eval_blocked_reason": eval_blocked_reason,
                "eval_runtime_regime": eval_runtime_regime,
                "eval_trade_gate_prob": eval_gate_prob.astype(np.float32, copy=False),
                "eval_trade_gate_threshold": eval_gate_threshold.astype(np.float32, copy=False),
                "eval_trade_gate_margin_min": eval_gate_margin_min.astype(np.float32, copy=False),
            },
            index=idx,
        )
        cache_df = self._prepare_dist_precomputed_cache_df(cache_df, session_labels=session_labels)
        if cache_df is None:
            _emit_status("MLPhysics(dist): failed to normalize precompute cache.", phase="ml_dist_precompute_error")
            return False
        self.set_precomputed_dist_backtest_df(cache_df)

        if allow_cache:
            to_store = cache_df
            if any((req_source_key, req_source_label, req_symbol_hint, req_symbol_mode)):
                to_store = cache_df.copy()
            if req_source_key:
                to_store["cache_source_key"] = req_source_key
            if req_source_label:
                to_store["cache_source_label"] = req_source_label
            if req_symbol_hint:
                to_store["cache_symbol_hint"] = req_symbol_hint
            if req_symbol_mode:
                to_store["cache_symbol_mode"] = req_symbol_mode
            try:
                to_store.to_parquet(cache_path, engine="pyarrow", compression="zstd")
                logging.info("MLPhysics OPT(dist): saved cache %s", cache_path)
            except Exception as exc:
                logging.warning("MLPhysics OPT(dist): failed saving cache %s (%s)", cache_path, exc)
        _emit_status("MLPhysics(dist): precompute ready.", phase="ml_dist_precompute_ready")
        return True

    def set_precomputed_backtest_df(self, precomputed_df: Optional[pd.DataFrame]) -> None:
        if precomputed_df is None or not isinstance(precomputed_df, pd.DataFrame) or precomputed_df.empty:
            self._opt_precomputed_df = None
            self._opt_eval_row = None
            self._opt_precomputed_index_ns = np.empty(0, dtype=np.int64)
            self._opt_precomputed_can_eval = np.empty(0, dtype=np.int8)
            self._opt_precomputed_feature_matrix = np.empty((0, len(ML_FEATURE_COLUMNS)), dtype=np.float32)
            self._opt_precomputed_session = []
            self._opt_precomputed_prob_up = np.empty(0, dtype=np.float32)
            self._opt_precomputed_cursor_pos = 0
            return
        df = precomputed_df
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                self._opt_precomputed_df = None
                self._opt_eval_row = None
                self._opt_precomputed_index_ns = np.empty(0, dtype=np.int64)
                self._opt_precomputed_can_eval = np.empty(0, dtype=np.int8)
                self._opt_precomputed_feature_matrix = np.empty((0, len(ML_FEATURE_COLUMNS)), dtype=np.float32)
                self._opt_precomputed_session = []
                self._opt_precomputed_prob_up = np.empty(0, dtype=np.float32)
                self._opt_precomputed_cursor_pos = 0
                return
        if df.index.tz is None:
            try:
                df = df.copy()
                df.index = df.index.tz_localize("US/Eastern")
            except Exception:
                pass
        self._opt_precomputed_df = df
        self._opt_eval_row = None
        try:
            idx = pd.DatetimeIndex(df.index)
            self._opt_precomputed_index_ns = self._datetime_index_ns(idx).copy()
        except Exception:
            self._opt_precomputed_index_ns = np.empty(0, dtype=np.int64)
        try:
            can_eval = pd.to_numeric(df.get("ml_can_eval", 1), errors="coerce").fillna(1)
            self._opt_precomputed_can_eval = can_eval.to_numpy(dtype=np.int8, copy=True)
        except Exception:
            self._opt_precomputed_can_eval = np.ones(len(df), dtype=np.int8)
        try:
            feature_df = df.reindex(columns=ML_FEATURE_COLUMNS)
            for col in feature_df.columns:
                feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
            self._opt_precomputed_feature_matrix = feature_df.to_numpy(dtype=np.float32, copy=True)
        except Exception:
            self._opt_precomputed_feature_matrix = np.full((len(df), len(ML_FEATURE_COLUMNS)), np.nan, dtype=np.float32)
        try:
            self._opt_precomputed_session = df.get("ml_session", pd.Series("", index=df.index)).fillna("").astype(str).tolist()
        except Exception:
            self._opt_precomputed_session = [""] * len(df)
        try:
            prob_up = pd.to_numeric(df.get("ml_prob_up", np.nan), errors="coerce").fillna(np.nan)
            self._opt_precomputed_prob_up = prob_up.to_numpy(dtype=np.float32, copy=True)
        except Exception:
            self._opt_precomputed_prob_up = np.full(len(df), np.nan, dtype=np.float32)
        self._opt_precomputed_cursor_pos = 0
        logging.info("MLPhysics OPT: attached precomputed backtest rows=%d", len(df))

    def _log_dist_bundle_summary(self) -> None:
        if self._dist_bundle is None:
            return
        gate_preview = []
        try:
            for session in sorted((self._dist_bundle.gate or {}).keys()):
                per_side = self._dist_bundle.gate.get(session, {}) or {}
                for side in ("LONG", "SHORT"):
                    payload = per_side.get(side) or {}
                    thr = payload.get("threshold")
                    if thr is None:
                        continue
                    try:
                        gate_preview.append(f"{session}/{side}={float(thr):.3f}")
                    except Exception:
                        gate_preview.append(f"{session}/{side}={thr}")
        except Exception:
            gate_preview = []

        gate_preview_text = ", ".join(gate_preview[:8]) if gate_preview else "none"
        if len(gate_preview) > 8:
            gate_preview_text += f", +{len(gate_preview) - 8} more"

        model_sessions = sorted((self._dist_bundle.model_index or {}).keys())
        logging.info(
            "MLPhysics(dist) active: run=%s ev_min=%.3f features=%d sessions=%s gate_enabled=%s thresholds=%s",
            self._dist_bundle.run_dir,
            float(self._dist_bundle.cfg.decision.ev_min),
            len(self._dist_bundle.feature_cols),
            ",".join(model_sessions),
            bool(self._dist_bundle.gate_enabled),
            gate_preview_text,
        )

    def _configure_dist_xgb_runtime(self) -> None:
        if self._dist_bundle is None:
            return

        gpu_enabled = bool(CONFIG.get("ML_PHYSICS_DIST_XGB_GPU_ENABLED", True))
        device = str(CONFIG.get("ML_PHYSICS_DIST_XGB_DEVICE", "cuda") or "cuda")
        predictor = str(CONFIG.get("ML_PHYSICS_DIST_XGB_PREDICTOR", "gpu_predictor") or "gpu_predictor")
        try:
            target_frac = float(CONFIG.get("ML_PHYSICS_DIST_XGB_GPU_TARGET_FRACTION", 0.75) or 0.75)
        except Exception:
            target_frac = 0.75
        target_frac = min(1.0, max(0.10, target_frac))
        cpu_count = int(os.cpu_count() or 8)
        feed_threads = max(1, int(round(cpu_count * target_frac)))

        total = 0
        updated = 0
        skipped = 0
        errors = 0

        def _tune_model(model) -> None:
            nonlocal total, updated, skipped, errors
            total += 1
            if not (hasattr(model, "get_xgb_params") and hasattr(model, "set_params")):
                skipped += 1
                return
            try:
                params = {"n_jobs": feed_threads}
                if gpu_enabled:
                    params["device"] = device
                    params["predictor"] = predictor
                model.set_params(**params)
                updated += 1
            except Exception:
                errors += 1

        try:
            for per_side in (self._dist_bundle.models or {}).values():
                for model_pack in (per_side or {}).values():
                    for payload in (model_pack or {}).values():
                        if isinstance(payload, dict):
                            for model in payload.values():
                                _tune_model(model)
                        elif payload is not None:
                            _tune_model(payload)
            for per_side in (self._dist_bundle.gate or {}).values():
                for gate_pack in (per_side or {}).values():
                    for k in ("classifier", "regressor", "calibrator"):
                        model = (gate_pack or {}).get(k)
                        if model is not None:
                            _tune_model(model)
        except Exception as exc:
            logging.warning("MLPhysics(dist): xgb runtime tuning failed: %s", exc)
            return

        logging.info(
            "MLPhysics(dist) xgb runtime tuning: total=%d updated=%d skipped=%d errors=%d "
            "gpu=%s device=%s predictor=%s target_fraction=%.2f feed_threads=%d",
            total,
            updated,
            skipped,
            errors,
            gpu_enabled,
            device,
            predictor,
            target_frac,
            feed_threads,
        )

    def _apply_dist_gate_threshold_clamp(self) -> None:
        """Clamp dist gate thresholds at runtime to handle inference-time distribution shift."""
        if self._dist_bundle is None:
            return
        cfg = CONFIG.get("ML_PHYSICS_DIST_GATE_THRESHOLD_CLAMP", {}) or {}
        if not bool(cfg.get("enabled", False)):
            return

        default_cfg = cfg.get("default", {}) or {}
        default_min = _safe_prob_bound(
            default_cfg.get("min"),
            None,
        ) if isinstance(default_cfg, dict) else None
        default_max = _safe_prob_bound(
            default_cfg.get("max"),
            None,
        ) if isinstance(default_cfg, dict) else None

        session_cfg_map = cfg.get("sessions", {}) or {}
        if not isinstance(session_cfg_map, dict):
            session_cfg_map = {}

        changed = 0
        total = 0
        for session, per_side in (self._dist_bundle.gate or {}).items():
            session_cfg = session_cfg_map.get(str(session).upper())
            if session_cfg is None:
                session_cfg = session_cfg_map.get(str(session))
            if not isinstance(session_cfg, dict):
                session_cfg = {}

            session_min = _safe_prob_bound(session_cfg.get("min"), default_min)
            session_max = _safe_prob_bound(session_cfg.get("max"), default_max)

            for side, payload in (per_side or {}).items():
                total += 1
                if not isinstance(payload, dict):
                    continue
                try:
                    old_thr = float(payload.get("threshold"))
                except Exception:
                    continue

                side_cfg = session_cfg.get(str(side).upper())
                if side_cfg is None:
                    side_cfg = session_cfg.get(str(side))
                if not isinstance(side_cfg, dict):
                    side_cfg = {}

                side_min = _safe_prob_bound(side_cfg.get("min"), session_min)
                side_max = _safe_prob_bound(side_cfg.get("max"), session_max)
                if side_min is not None and side_max is not None and side_min > side_max:
                    side_min, side_max = side_max, side_min

                new_thr = old_thr
                if side_min is not None:
                    new_thr = max(new_thr, float(side_min))
                if side_max is not None:
                    new_thr = min(new_thr, float(side_max))
                if not np.isfinite(new_thr):
                    continue
                if abs(new_thr - old_thr) <= 1e-12:
                    continue

                payload["threshold"] = float(new_thr)
                changed += 1
                logging.info(
                    "MLPhysics(dist) gate threshold clamp: %s/%s %.3f -> %.3f",
                    str(session),
                    str(side),
                    old_thr,
                    float(new_thr),
                )

        logging.info(
            "MLPhysics(dist) gate threshold clamp summary: changed=%d/%d default_min=%s default_max=%s",
            changed,
            total,
            f"{default_min:.3f}" if default_min is not None else "none",
            f"{default_max:.3f}" if default_max is not None else "none",
        )

    def _record_dist_eval(self, payload: Optional[Dict]) -> None:
        if not payload:
            return
        if self._opt_dist_precompute_mode:
            return
        self._dist_eval_total += 1
        decision = str(payload.get("decision") or "unknown").lower()
        session = str(payload.get("session") or "UNKNOWN").upper()
        self._dist_session_counts[session] += 1
        if decision in {"signal_long", "signal_short"}:
            self._dist_eval_signals += 1
        elif decision == "blocked":
            self._dist_eval_blocked += 1

        blocked_reason = payload.get("blocked_reason")
        if blocked_reason:
            self._dist_reason_counts[str(blocked_reason)] += 1
        for code in payload.get("reason_codes") or []:
            self._dist_reason_counts[f"code:{code}"] += 1

        if self._dist_log_every <= 0 or self._dist_eval_total % self._dist_log_every != 0:
            return

        top_reasons = ", ".join(
            f"{name}={count}"
            for name, count in self._dist_reason_counts.most_common(5)
        ) or "none"
        top_sessions = ", ".join(
            f"{name}={count}"
            for name, count in self._dist_session_counts.most_common(4)
        ) or "none"
        signal_rate = (self._dist_eval_signals / self._dist_eval_total) if self._dist_eval_total else 0.0
        logging.info(
            "MLPhysics(dist) evals=%d signals=%d blocked=%d signal_rate=%.1f%% top_reasons=%s top_sessions=%s",
            self._dist_eval_total,
            self._dist_eval_signals,
            self._dist_eval_blocked,
            signal_rate * 100.0,
            top_reasons,
            top_sessions,
        )

    def _hyst_key(self, key: Optional[str]) -> str:
        norm = str(key or "GLOBAL").strip().upper()
        return norm or "GLOBAL"

    def _hysteresis_gate(
        self,
        state_key: str,
        side: Optional[str],
        confidence: float,
        required_confidence: float,
    ) -> Tuple[bool, str]:
        if not self._hyst_enabled:
            return bool(side), "disabled"

        key = self._hyst_key(state_key)
        side_norm = str(side or "").upper()
        if side_norm not in {"LONG", "SHORT"}:
            self._hyst_state.pop(key, None)
            return False, "neutral_disarm"

        try:
            conf = float(confidence)
        except Exception:
            conf = 0.0
        if not np.isfinite(conf):
            conf = 0.0

        try:
            req = float(required_confidence)
        except Exception:
            req = 0.0
        if not np.isfinite(req):
            req = 0.0

        entry_level = float(np.clip(req + self._hyst_entry_margin, 0.0, 1.0))
        exit_level = float(np.clip(req - self._hyst_exit_margin, 0.0, 1.0))
        flip_level = float(np.clip(req + self._hyst_flip_margin, 0.0, 1.0))

        state = self._hyst_state.get(key)
        active_side = str((state or {}).get("side") or "").upper()
        try:
            last_conf = float((state or {}).get("conf", 0.0) or 0.0)
        except Exception:
            last_conf = 0.0
        if not np.isfinite(last_conf):
            last_conf = 0.0

        if not active_side:
            if conf >= entry_level:
                self._hyst_state[key] = {"side": side_norm, "conf": conf}
                return True, "edge_entry"
            return False, "entry_not_met"

        if side_norm != active_side:
            if conf >= flip_level:
                self._hyst_state[key] = {"side": side_norm, "conf": conf}
                return True, "edge_flip"
            return False, "flip_not_met"

        if conf <= exit_level:
            self._hyst_state.pop(key, None)
            return False, "exit_disarm"

        if self._hyst_retrigger_delta > 0.0 and conf >= (last_conf + self._hyst_retrigger_delta):
            self._hyst_state[key] = {"side": side_norm, "conf": conf}
            return True, "retrigger"

        self._hyst_state[key] = {"side": side_norm, "conf": max(last_conf, conf)}
        return False, "held"

    def _log_feature_issue(self, cols, bar_count, optional=False):
        now = time.time()
        if now - self._last_feature_log_ts < self._feature_log_interval:
            return
        status = "optional" if optional else "critical"
        level = logging.info if optional else logging.warning
        level(f"MLPhysics: {status} feature NaNs: {', '.join(cols)} | bars={bar_count}")
        self._last_feature_log_ts = now

    def _resample_ohlcv(self, df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset({c.lower() for c in df.columns}):
            return df
        try:
            tf = int(minutes)
        except Exception:
            return df
        if tf <= 1:
            return df
        resampler = self._tf_resamplers.get(tf)
        if resampler is None:
            resampler = IncrementalOHLCVResampler(tf)
            self._tf_resamplers[tf] = resampler
        return resampler.update(df)

    def _resolve_dist_run_dir(self) -> Optional[Path]:
        def is_dist_run_dir(path: Path) -> bool:
            return (
                path.is_dir()
                and (path / "config.json").exists()
                and (path / "artifact_index.json").exists()
            )

        def collect_candidates(root: Path) -> list[Path]:
            candidates: list[Path] = []
            if is_dist_run_dir(root):
                candidates.append(root)
            try:
                level1 = [p for p in root.iterdir() if p.is_dir()]
            except Exception:
                return candidates
            for p in level1:
                if is_dist_run_dir(p):
                    candidates.append(p)
                try:
                    for c in p.iterdir():
                        if c.is_dir() and is_dist_run_dir(c):
                            candidates.append(c)
                except Exception:
                    continue
            return candidates

        configured = str(CONFIG.get("ML_PHYSICS_DIST_RUN_DIR", "") or "").strip()
        if configured:
            run_dir = Path(configured).expanduser().resolve()
            if is_dist_run_dir(run_dir):
                return run_dir
            if run_dir.is_dir():
                nested = collect_candidates(run_dir)
                if nested:
                    nested.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    chosen = nested[0]
                    logging.info(
                        "MLPhysics(dist): configured path is a parent; using latest nested run: %s",
                        chosen,
                    )
                    return chosen
            logging.warning("MLPhysics(dist): configured run dir invalid: %s", run_dir)
            return None

        base_dir = Path(
            str(CONFIG.get("ML_PHYSICS_DIST_RUN_BASE_DIR", "dist_bracket_ml_runs") or "dist_bracket_ml_runs")
        ).expanduser().resolve()
        if not base_dir.exists() or not base_dir.is_dir():
            return None

        candidates = collect_candidates(base_dir)
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _init_dist_replacement(self) -> None:
        replace_enabled = bool(CONFIG.get("ML_PHYSICS_REPLACE_WITH_DIST", True))
        if not replace_enabled:
            self._dist_mode_reason = "disabled_by_config"
            return
        if _dist_load_bundle is None or _dist_predict is None:
            self._dist_mode_reason = "dist_module_unavailable"
            return
        run_dir = self._resolve_dist_run_dir()
        if run_dir is None:
            self._dist_mode_reason = "dist_run_not_found"
            return
        self._dist_run_dir = Path(run_dir).expanduser().resolve()
        self._reset_dist_session_cache()

        # Backtest fast-start mode: if cache is present, we can defer loading heavy model artifacts.
        lazy_backtest = bool(
            self._dist_lazy_backtest
            and self._opt_enabled
            and self._opt_mode == "backtest"
            and bool(self._opt_cfg.get("prediction_cache", True))
            and not bool(self._opt_cfg.get("overwrite_cache", False))
        )
        if lazy_backtest:
            self._dist_mode = True
            self._dist_bundle = None
            self._dist_mode_reason = f"lazy_backtest:{self._dist_run_dir}"
            logging.info("MLPhysics(dist): lazy backtest mode enabled for run: %s", self._dist_run_dir)
            return

        if not self._ensure_dist_bundle_loaded():
            return

    def _ensure_dist_bundle_loaded(self) -> bool:
        if self._dist_bundle is not None:
            return True
        if _dist_load_bundle is None or _dist_predict is None:
            self._dist_mode = False
            self._dist_mode_reason = "dist_module_unavailable"
            return False

        run_dir = self._dist_run_dir or self._resolve_dist_run_dir()
        if run_dir is None:
            self._dist_mode = False
            self._dist_mode_reason = "dist_run_not_found"
            return False
        run_dir = Path(run_dir).expanduser().resolve()
        self._dist_run_dir = run_dir
        self._reset_dist_session_cache()
        try:
            self._dist_bundle = _dist_load_bundle(run_dir)
            self._apply_dist_gate_threshold_clamp()
            self._configure_dist_xgb_runtime()
            self._dist_mode = True
            self._dist_mode_reason = f"loaded:{run_dir}"
            logging.info("MLPhysics now using dist_bracket_ml replacement run: %s", run_dir)
            self._log_dist_bundle_summary()
            return True
        except Exception as exc:
            self._dist_mode = False
            self._dist_bundle = None
            self._dist_mode_reason = f"load_failed:{exc}"
            logging.warning("MLPhysics(dist): failed to load replacement run: %s", exc)
            return False

    def _on_bar_dist(self, df: pd.DataFrame, current_time=None) -> Optional[Dict]:
        if self._opt_dist_backtest_active() and not self._opt_dist_precompute_mode and current_time is not None:
            ts_ns = self._timestamp_to_eastern_ns(current_time)
            if ts_ns is not None and len(self._opt_dist_index_ns) > 0:
                pos = self._lookup_sorted_ns(self._opt_dist_index_ns, ts_ns, self._opt_dist_cursor_pos)
                if pos is not None:
                    self._opt_dist_cursor_pos = int(pos)
                    eval_payload = self._dist_eval_from_cache_row(int(pos), current_time=current_time)
                    has_signal = (
                        int(self._opt_dist_has_signal_arr[pos]) > 0
                        if 0 <= pos < len(self._opt_dist_has_signal_arr)
                        else False
                    )
                    if not has_signal:
                        self.last_eval = eval_payload
                        self._record_dist_eval(self.last_eval)
                        return None
                    signal_payload = self._dist_signal_from_cache_row(
                        int(pos),
                        eval_payload,
                        current_time=current_time,
                    )
                    self.last_eval = eval_payload
                    self._record_dist_eval(self.last_eval)
                    if isinstance(signal_payload, dict):
                        return signal_payload
                    return None

        if self._dist_bundle is None or _dist_predict is None:
            if not self._ensure_dist_bundle_loaded():
                return None
        ts_for_session = current_time
        if ts_for_session is None and isinstance(df, pd.DataFrame) and len(df.index) > 0:
            try:
                ts_for_session = pd.Timestamp(df.index[-1])
            except Exception:
                ts_for_session = None
        session_name = self._infer_dist_session(ts_for_session)
        if df is None or df.empty:
            self.last_eval = {
                "decision": "no_signal",
                "blocked_reason": "empty_history",
                "session": session_name,
                "ev_source": "dist_bracket_ml",
            }
            self._record_dist_eval(self.last_eval)
            return None

        source_df = df
        if self._dist_input_max_bars > 0 and len(source_df) > self._dist_input_max_bars:
            source_df = source_df.iloc[-self._dist_input_max_bars :]

        lower_cols = {str(col).strip().lower(): col for col in source_df.columns}
        open_col = lower_cols.get("open")
        high_col = lower_cols.get("high")
        low_col = lower_cols.get("low")
        close_col = lower_cols.get("close")
        volume_col = lower_cols.get("volume")
        ts_col = lower_cols.get("ts")

        if not all((open_col, high_col, low_col, close_col)):
            self.last_eval = {
                "decision": "no_signal",
                "blocked_reason": "missing_ohlc_columns",
                "session": session_name,
                "ev_source": "dist_bracket_ml",
            }
            self._record_dist_eval(self.last_eval)
            return None

        ts_series = source_df[ts_col] if ts_col is not None else source_df.index
        if volume_col is not None:
            volume_series = source_df[volume_col]
        else:
            volume_series = pd.Series(0.0, index=source_df.index)
        recent = pd.DataFrame(
            {
                "ts": ts_series,
                "open": source_df[open_col],
                "high": source_df[high_col],
                "low": source_df[low_col],
                "close": source_df[close_col],
                "volume": volume_series,
            }
        ).reset_index(drop=True)
        if len(recent) < 220:
            self.last_eval = {
                "decision": "no_signal",
                "blocked_reason": "insufficient_history",
                "session": session_name,
                "ev_source": "dist_bracket_ml",
            }
            self._record_dist_eval(self.last_eval)
            return None

        tf_minutes = int(CONFIG.get("ML_PHYSICS_TIMEFRAME_MINUTES", 1) or 1)
        if tf_minutes > 1:
            align_mode = str(CONFIG.get("ML_PHYSICS_BAR_ALIGNMENT", "open") or "open").lower()
            ts_now = current_time
            if ts_now is None and len(recent) > 0:
                try:
                    ts_now = pd.Timestamp(recent["ts"].iloc[-1])
                except Exception:
                    ts_now = None
            minute = None
            if ts_now is not None:
                try:
                    minute = int(pd.Timestamp(ts_now).minute)
                except Exception:
                    minute = None
            if minute is not None:
                if align_mode == "close":
                    if minute % int(tf_minutes) != int(tf_minutes) - 1:
                        self.last_eval = {
                            "decision": "no_signal",
                            "blocked_reason": "tf_alignment",
                            "session": session_name,
                            "ev_source": "dist_bracket_ml",
                        }
                        self._record_dist_eval(self.last_eval)
                        return None
                else:
                    if minute % int(tf_minutes) != 0:
                        self.last_eval = {
                            "decision": "no_signal",
                            "blocked_reason": "tf_alignment",
                            "session": session_name,
                            "ev_source": "dist_bracket_ml",
                        }
                        self._record_dist_eval(self.last_eval)
                        return None
                    # At tf bar open, use completed bars only (drop current 1m bar).
                    if len(recent) > 1:
                        recent = recent.iloc[:-1].reset_index(drop=True)
                        if len(recent) < 220:
                            self.last_eval = {
                                "decision": "no_signal",
                                "blocked_reason": "insufficient_history",
                                "session": session_name,
                                "ev_source": "dist_bracket_ml",
                            }
                            self._record_dist_eval(self.last_eval)
                            return None

        try:
            signal = _dist_predict(self._dist_bundle, recent)
        except Exception as exc:
            logging.warning("MLPhysics(dist): inference failure: %s", exc)
            self.last_eval = {
                "decision": "no_signal",
                "blocked_reason": "dist_inference_error",
                "session": session_name,
                "ev_source": "dist_bracket_ml",
            }
            self._record_dist_eval(self.last_eval)
            return None
        return self._consume_dist_signal(signal, recent)

    def _trend_context(self, hist_df: pd.DataFrame, X: pd.DataFrame, regime_key: Optional[str], high_vol: bool) -> str:
        """Build a concise trend context string for logging (NY focus, low overhead)."""
        parts = []
        try:
            is_trending = int(X.iloc[0].get("Is_Trending", 0))
            trend_dir_val = float(X.iloc[0].get("Trend_Direction", 0.0))
            if trend_dir_val > 0:
                trend_dir = "UP"
            elif trend_dir_val < 0:
                trend_dir = "DOWN"
            else:
                trend_dir = "FLAT"
            parts.append(f"is_trending={is_trending}")
            parts.append(f"trend_dir={trend_dir}")
        except Exception:
            pass

        try:
            close = hist_df["close"]
            if len(close) >= 50:
                ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
                ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
                if ema20 > ema50:
                    ema_bias = "UP"
                elif ema20 < ema50:
                    ema_bias = "DOWN"
                else:
                    ema_bias = "FLAT"
                parts.append(f"ema_bias={ema_bias}")
            if len(close) >= 60:
                ret60 = close.iloc[-1] - close.iloc[-60]
                parts.append(f"ret60={ret60:.2f}")
        except Exception:
            pass

        if regime_key is not None:
            parts.append(f"regime={regime_key}")
        parts.append(f"high_vol={int(bool(high_vol))}")
        return " | ".join(parts) if parts else ""

    def calculate_slope(self, values):
        """Exact copy from juliemlsession.py"""
        y = np.array(values)
        x = np.arange(len(y))
        if len(y) < 2:
            return 0
        cov = np.cov(x, y)[0, 1]
        var = np.var(x)
        return cov / var if var != 0 else 0

    def prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or len(df) < 200:
            return None

        # Backtest optimized path: direct lookup from precomputed table.
        if self._opt_backtest_active() and self._opt_eval_ts is not None:
            ts_ns = self._timestamp_to_eastern_ns(self._opt_eval_ts)
            if ts_ns is not None and len(self._opt_precomputed_index_ns) > 0:
                pos = self._lookup_sorted_ns(
                    self._opt_precomputed_index_ns,
                    ts_ns,
                    self._opt_precomputed_cursor_pos,
                )
                if pos is not None:
                    self._opt_precomputed_cursor_pos = int(pos)
                    can_eval = (
                        int(self._opt_precomputed_can_eval[pos])
                        if 0 <= pos < len(self._opt_precomputed_can_eval)
                        else 0
                    )
                    if can_eval <= 0:
                        return None
                    prob_up = (
                        float(self._opt_precomputed_prob_up[pos])
                        if 0 <= pos < len(self._opt_precomputed_prob_up)
                        else float("nan")
                    )
                    row_session = (
                        self._opt_precomputed_session[pos]
                        if 0 <= pos < len(self._opt_precomputed_session)
                        else ""
                    )
                    self._opt_eval_row = {
                        "ml_session": row_session,
                        "ml_prob_up": prob_up,
                    }
                    feature_vals = (
                        self._opt_precomputed_feature_matrix[pos]
                        if 0 <= pos < len(self._opt_precomputed_feature_matrix)
                        else None
                    )
                    if feature_vals is not None:
                        feature_vals = np.asarray(feature_vals, dtype=np.float32)
                        nan_mask = ~np.isfinite(feature_vals)
                        if not np.any(nan_mask):
                            return pd.DataFrame([feature_vals], columns=ML_FEATURE_COLUMNS)
                        critical_missing = []
                        optional_filled = []
                        feature_out = feature_vals.astype(np.float32, copy=True)
                        for idx_col, missing in enumerate(nan_mask.tolist()):
                            if not missing:
                                continue
                            col = ML_FEATURE_COLUMNS[idx_col]
                            if col in OPTIONAL_FEATURE_COLUMNS:
                                feature_out[idx_col] = 0.0
                                optional_filled.append(col)
                            else:
                                critical_missing.append(col)
                        if critical_missing:
                            self._log_feature_issue(sorted(critical_missing), len(df), optional=False)
                            return None
                        if optional_filled:
                            self._log_feature_issue(sorted(optional_filled), len(df), optional=True)
                        return pd.DataFrame([feature_out], columns=ML_FEATURE_COLUMNS)

        # Legacy / live path: compute vectorized features on the current history.
        try:
            feature_df = mlp.compute_features(df, cast_float32=False)
        except Exception as exc:
            logging.warning("MLPhysics: feature compute failure: %s", exc)
            return None
        if feature_df is None or feature_df.empty:
            return None
        last = feature_df.iloc[-1]
        feature_vals = []
        critical_missing = []
        optional_filled = []
        for col in ML_FEATURE_COLUMNS:
            val = last.get(col, np.nan)
            if pd.isna(val):
                if col in OPTIONAL_FEATURE_COLUMNS:
                    val = 0.0
                    optional_filled.append(col)
                else:
                    critical_missing.append(col)
            feature_vals.append(val)
        if critical_missing:
            self._log_feature_issue(sorted(critical_missing), len(feature_df), optional=False)
            return None
        if optional_filled:
            self._log_feature_issue(sorted(optional_filled), len(feature_df), optional=True)
        return pd.DataFrame([feature_vals], columns=ML_FEATURE_COLUMNS)

    @staticmethod
    def _efficiency_ratio(close: pd.Series, window: int) -> Optional[float]:
        if close is None or len(close) < window + 1:
            return None
        recent = close.iloc[-(window + 1):]
        net = float(abs(recent.iloc[-1] - recent.iloc[0]))
        denom = float(recent.diff().abs().sum())
        if denom <= 0:
            return 0.0
        return net / denom

    @staticmethod
    def _vwap_crosses(df: pd.DataFrame, window: int) -> Optional[int]:
        if df is None or len(df) < window:
            return None
        if "close" not in df.columns or "volume" not in df.columns:
            return None
        price = df["close"]
        volume = df["volume"]
        pv = (price * volume).rolling(window).sum()
        vv = volume.rolling(window).sum()
        vwap = pv / vv.replace(0, np.nan)
        delta = (price - vwap).iloc[-window:]
        sign = np.sign(delta.to_numpy())
        for i in range(1, len(sign)):
            if sign[i] == 0:
                sign[i] = sign[i - 1]
        if len(sign) < 2:
            return 0
        crosses = np.sum(sign[1:] * sign[:-1] < 0)
        return int(crosses)

    def _trade_budget_key(self, session_name: str, regime_key: Optional[str]) -> str:
        regime = str(regime_key or "low")
        return f"{session_name}|{regime}"

    def _get_trade_budget_state(self, key: str) -> Dict[str, Deque[int]]:
        state = self._trade_budget_state.get(key)
        if state is None:
            state = {
                "evals": deque(),
                "trades": deque(),
            }
            self._trade_budget_state[key] = state
        return state

    def _apply_trade_budget_nudge(
        self,
        session_name: str,
        regime_key: Optional[str],
        req: float,
        short_req: float,
    ) -> Tuple[float, float, Optional[float]]:
        cfg = self._trade_budget_cfg or {}
        if not bool(cfg.get("enabled", False)):
            return req, short_req, None

        sess_overrides = cfg.get("sessions", {}) or {}
        sess_cfg = sess_overrides.get(session_name, {}) if isinstance(sess_overrides, dict) else {}
        min_cov = sess_cfg.get("min_coverage", cfg.get("min_coverage"))
        max_cov = sess_cfg.get("max_coverage", cfg.get("max_coverage"))
        step = sess_cfg.get("nudge_step", cfg.get("nudge_step", 0.01))
        window = int(sess_cfg.get("window_evals", cfg.get("window_evals", 120)) or 120)
        if window <= 0:
            return req, short_req, None

        try:
            min_cov = None if min_cov is None else float(min_cov)
        except Exception:
            min_cov = None
        try:
            max_cov = None if max_cov is None else float(max_cov)
        except Exception:
            max_cov = None
        try:
            step = float(step or 0.0)
        except Exception:
            step = 0.0
        if step <= 0:
            return req, short_req, None

        key = self._trade_budget_key(session_name, regime_key)
        state = self._get_trade_budget_state(key)
        evals = state["evals"]
        trades = state["trades"]
        while len(evals) > window:
            evals.popleft()
        while len(trades) > window:
            trades.popleft()

        if len(evals) < max(20, int(window * 0.25)):
            return req, short_req, None
        realized_cov = float(sum(trades)) / float(sum(evals)) if sum(evals) > 0 else 0.0
        center = (float(req) + float(short_req)) / 2.0
        margin = max(float(req) - center, center - float(short_req))
        new_margin = margin
        if min_cov is not None and realized_cov < min_cov:
            new_margin = max(0.01, margin - step)
        elif max_cov is not None and realized_cov > max_cov:
            new_margin = min(0.49, margin + step)

        if new_margin == margin:
            return req, short_req, realized_cov
        new_req = min(0.99, center + new_margin)
        new_short = max(0.01, center - new_margin)
        return new_req, new_short, realized_cov

    def _record_trade_budget_eval(self, session_name: str, regime_key: Optional[str], took_trade: bool) -> None:
        cfg = self._trade_budget_cfg or {}
        if not bool(cfg.get("enabled", False)):
            return
        key = self._trade_budget_key(session_name, regime_key)
        state = self._get_trade_budget_state(key)
        state["evals"].append(1)
        state["trades"].append(1 if took_trade else 0)

    def _apply_ev_session_overrides(
        self,
        ev_cfg_runtime: Dict,
        session_name: Optional[str],
        regime_key: Optional[str],
    ) -> Dict:
        merged = dict(ev_cfg_runtime or {})
        overrides = self._ev_cfg_session_overrides or {}
        if not isinstance(overrides, dict):
            return merged
        if not session_name:
            return merged
        sess_cfg = overrides.get(session_name)
        if not isinstance(sess_cfg, dict):
            return merged
        regime_overrides = sess_cfg.get("regimes")
        for key, value in sess_cfg.items():
            if key == "regimes":
                continue
            merged[key] = value
        if isinstance(regime_overrides, dict):
            reg_key = str(regime_key or "").lower()
            reg_cfg = regime_overrides.get(reg_key) or regime_overrides.get(str(regime_key or ""))
            if isinstance(reg_cfg, dict):
                merged.update(reg_cfg)
        return merged

    def on_bar(self, df: Optional[pd.DataFrame], current_time=None) -> Optional[Dict]:
        """
        Adapted from juliemlsession.py on_bar method.
        Uses SessionManager to get the correct model and parameters.
        Uses the full dataframe passed in (which already has 500 bars of history).
        """
        self.last_eval = None
        self._opt_eval_row = None
        self._opt_eval_ts = None
        if self._opt_backtest_active() and current_time is not None:
            try:
                self._opt_eval_ts = pd.Timestamp(current_time)
            except Exception:
                self._opt_eval_ts = None
        # In lazy_backtest mode, _dist_bundle is intentionally deferred and we still
        # must route through _on_bar_dist so precomputed cache rows can be used.
        if self._dist_mode:
            return self._on_bar_dist(df, current_time=current_time)
        if self.sm is None:
            if not self._warned_no_legacy_fallback:
                logging.warning(
                    "MLPhysics: dist mode unavailable (%s) and legacy models are detached; returning no signal",
                    self._dist_mode_reason or "unknown_reason",
                )
                self._warned_no_legacy_fallback = True
            return None
        # 1. Get current session setup
        setup = self.sm.get_current_setup(current_time)

        if setup is None:
            logging.info("💤 Market Closed (No active session strategy)")
            return None

        if setup.get("disabled"):
            logging.info(f"⚠️ MLPhysics {setup['name']} disabled by guardrails")
            return None

        if setup['model'] is None:
            logging.info(f"⚠️ Session {setup['name']} active, but brain file is missing!")
            return None

        # 2. Convert df to the format expected by prepare_features
        # Need columns: datetime, Open, High, Low, Close, Volume
        hist_df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        use_opt_backtest = self._opt_backtest_active()
        if not use_opt_backtest:
            tf_minutes = setup.get("timeframe_minutes") or CONFIG.get("ML_PHYSICS_TIMEFRAME_MINUTES", 1)
            if isinstance(tf_minutes, (int, float)) and tf_minutes > 1:
                align_mode = str(CONFIG.get("ML_PHYSICS_BAR_ALIGNMENT", "open") or "open").lower()
                current_ts = current_time
                if current_ts is None and isinstance(hist_df.index, pd.DatetimeIndex) and len(hist_df.index) > 0:
                    current_ts = hist_df.index[-1]
                if current_ts is not None:
                    try:
                        ts = pd.Timestamp(current_ts)
                        if ts.tzinfo is None and isinstance(hist_df.index, pd.DatetimeIndex) and hist_df.index.tz is not None:
                            ts = ts.tz_localize(hist_df.index.tz)
                        minute = int(ts.minute)
                    except Exception:
                        minute = None
                    if minute is not None:
                        if align_mode == "close":
                            if minute % int(tf_minutes) != int(tf_minutes) - 1:
                                return None
                        else:
                            if minute % int(tf_minutes) != 0:
                                return None
                            # At tf bar open, drop the current 1m bar so we only use the completed tf bar.
                            if len(hist_df) > 1:
                                hist_df = hist_df.iloc[:-1]
                if hist_df is None or hist_df.empty:
                    return None
                hist_df = self._resample_ohlcv(hist_df, int(tf_minutes))
                if not self._logged_resample:
                    logging.info(f"MLPhysics: resampling to {int(tf_minutes)}min for feature build")
                    self._logged_resample = True
        if not use_opt_backtest:
            # Legacy path keeps local datetime materialization.
            hist_df = hist_df.copy()
            hist_df["datetime"] = hist_df.index

            # Normalize OHLCV names (no-op if already lowercase).
            col_map = {}
            for col in hist_df.columns:
                if col.lower() == "open":
                    col_map[col] = "open"
                elif col.lower() == "high":
                    col_map[col] = "high"
                elif col.lower() == "low":
                    col_map[col] = "low"
                elif col.lower() == "close":
                    col_map[col] = "close"
                elif col.lower() == "volume":
                    col_map[col] = "volume"
            hist_df = hist_df.rename(columns=col_map)

        # Ensure we have enough data
        if use_opt_backtest:
            if self._opt_eval_ts is None:
                return None
        elif len(hist_df) < 20:
            logging.info(f"📊 {setup['name']}: Building Physics Data ({len(hist_df)}/20)...")
            return None

        # 3. Run Prediction
        X = self.prepare_features(hist_df)

        if X is not None:
            model = setup.get("model")
            ev_models = setup.get("ev_models")
            threshold = setup.get("threshold")
            trade_gate_model = setup.get("gate_model")
            trade_gate_threshold = setup.get("gate_threshold")
            vol_split_cfg = CONFIG.get("ML_PHYSICS_VOL_SPLIT", {}) or {}
            vol_split_3way = CONFIG.get("ML_PHYSICS_VOL_SPLIT_3WAY", {}) or {}
            vol_guard = CONFIG.get("ML_PHYSICS_VOL_GUARD", {}) or {}
            gate_cfg = CONFIG.get("ML_PHYSICS_HIGH_VOL_DIRECTIONAL_GATE", {}) or {}
            bump_cfg = CONFIG.get("ML_PHYSICS_HIGH_VOL_THRESHOLD_BUMP", {}) or {}

            split_regimes = set(setup.get("split_regimes") or [])
            split_3way = bool(vol_split_3way.get("enabled")) and setup.get("name") in set(vol_split_3way.get("sessions", []))
            if "normal" in split_regimes:
                split_3way = True

            regime_key = None
            high_vol = False
            ev_pair = None
            if split_3way:
                try:
                    vol_regime, _, _ = volatility_filter.get_regime(hist_df)
                except Exception:
                    vol_regime = VolRegime.LOW
                if vol_regime == VolRegime.ULTRA_LOW:
                    vol_regime = VolRegime.LOW
                if vol_regime == VolRegime.HIGH:
                    regime_key = "high"
                elif vol_regime == VolRegime.NORMAL:
                    regime_key = "normal"
                else:
                    regime_key = "low"
                high_vol = regime_key == "high"
            else:
                feature_name = (
                    gate_cfg.get("feature")
                    or vol_guard.get("feature")
                    or vol_split_cfg.get("feature")
                    or "High_Volatility"
                )
                if feature_name in X.columns:
                    try:
                        high_vol = float(X.iloc[0].get(feature_name, 0.0)) >= 0.5
                    except Exception:
                        high_vol = False
                regime_key = "high" if high_vol else "low"

            # Volatility split: choose model/threshold by regime
            if setup.get("split") and isinstance(model, dict):
                regime = regime_key or ("high" if high_vol else "low")
                if regime in (setup.get("disabled_regimes") or set()):
                    logging.info(f"⚠️ MLPhysics {setup['name']} {regime} disabled by guardrails")
                    return None
                model_candidate = model.get(regime)
                ev_candidate = ev_models.get(regime) if isinstance(ev_models, dict) else None
                if model_candidate is None and regime == "normal":
                    model_candidate = model.get("low") or model.get("high")
                    if isinstance(ev_models, dict):
                        ev_candidate = ev_models.get("low") or ev_models.get("high")
                    if model_candidate is not None:
                        logging.info(
                            f"⚠️ MLPhysics {setup['name']} normal model missing; "
                            "falling back to low/high"
                        )
                model = model_candidate
                ev_pair = ev_candidate
                if model is None:
                    logging.info(f"⚠️ MLPhysics {setup['name']} missing {regime} model")
                    return None
            else:
                # Optional volatility guard (skip ML trades during high-vol regimes)
                if vol_guard.get("enabled"):
                    guard_sessions = set(vol_guard.get("sessions", []))
                    if setup.get("name") in guard_sessions and high_vol:
                        logging.info(f"⚠️ MLPhysics {setup['name']} blocked (high volatility)")
                        return None
                ev_pair = ev_models if isinstance(ev_models, dict) else None

            context = {}
            if "ATR_State" in X.columns:
                try:
                    context["atr_state"] = int(X.iloc[0].get("ATR_State", 0))
                except Exception:
                    context = {}
            if "Trend_State" in X.columns:
                try:
                    context["trend_state"] = int(X.iloc[0].get("Trend_State", 0))
                except Exception:
                    pass
            if "Liquidity_State" in X.columns:
                try:
                    context["liquidity_state"] = int(X.iloc[0].get("Liquidity_State", 1))
                except Exception:
                    pass
            context["regime_key"] = str(regime_key or ("high" if high_vol else "low"))
            req, short_req, policy = _extract_thresholds(
                threshold,
                regime_key or ("high" if high_vol else "low"),
                setup.get("threshold"),
                context=context or None,
            )
            if policy == "disabled":
                logging.info(f"⚠️ MLPhysics {setup['name']} disabled by threshold policy")
                self._record_trade_budget_eval(setup.get("name"), regime_key, False)
                return None
            dynamic_gate_threshold = _extract_gate_threshold(
                threshold,
                regime_key or ("high" if high_vol else "low"),
                context=context or None,
            )
            if dynamic_gate_threshold is not None:
                trade_gate_threshold = dynamic_gate_threshold
            ev_cfg_runtime = dict(self._ev_cfg or {})
            ev_runtime_override = _extract_ev_runtime(
                threshold,
                regime_key or ("high" if high_vol else "low"),
                context=context or None,
            )
            if isinstance(ev_runtime_override, dict) and ev_runtime_override:
                ev_cfg_runtime.update(ev_runtime_override)
            high_vol_overrides = ev_cfg_runtime.get("high_vol_overrides")
            if high_vol and isinstance(high_vol_overrides, dict):
                sess_hv = high_vol_overrides.get(setup.get("name"))
                if isinstance(sess_hv, dict):
                    ev_cfg_runtime.update(sess_hv)
            ev_cfg_runtime = self._apply_ev_session_overrides(
                ev_cfg_runtime,
                setup.get("name"),
                regime_key or ("high" if high_vol else "low"),
            )

            ny_normal_cfg = CONFIG.get("ML_PHYSICS_NY_NORMAL_FILTER", {}) or {}
            normal_margin = None
            if ny_normal_cfg.get("enabled") and setup.get("name") in ("NY_AM", "NY_PM") and regime_key == "normal":
                er_window = int(ny_normal_cfg.get("er_window", 30) or 30)
                vwap_window = int(ny_normal_cfg.get("vwap_cross_window", 60) or 60)
                er_min = float(ny_normal_cfg.get("er_min", 0.25) or 0.25)
                vwap_cross_max = int(ny_normal_cfg.get("vwap_cross_max", 3) or 3)
                er_value = self._efficiency_ratio(hist_df["close"], er_window)
                vwap_crosses = self._vwap_crosses(hist_df, vwap_window)
                if er_value is None or vwap_crosses is None:
                    logging.info(f"⚠️ MLPhysics {setup['name']} normal: insufficient data for ER/VWAP filter")
                    return None
                is_trend = (er_value >= er_min) and (vwap_crosses <= vwap_cross_max)
                if not is_trend and ny_normal_cfg.get("block_chop", True):
                    logging.info(
                        f"⚠️ MLPhysics {setup['name']} normal chop blocked "
                        f"(ER={er_value:.2f} VWAPx={vwap_crosses})"
                    )
                    return None
                try:
                    normal_margin = float(ny_normal_cfg.get("margin", 0.0) or 0.0)
                except Exception:
                    normal_margin = None

            # High-vol threshold bump
            if high_vol and bump_cfg.get("enabled"):
                bump_sessions = set(bump_cfg.get("sessions", []))
                if setup.get("name") in bump_sessions:
                    try:
                        bump = float(bump_cfg.get("bump", 0.0) or 0.0)
                    except Exception:
                        bump = 0.0
                    if bump > 0:
                        try:
                            max_thr = float(bump_cfg.get("max_threshold", 0.95))
                        except Exception:
                            max_thr = 0.95
                        center = (float(req) + float(short_req)) / 2.0
                        margin = max(float(req) - center, center - float(short_req))
                        new_margin = margin + bump
                        new_req = min(max_thr, center + new_margin)
                        new_short = max(1.0 - max_thr, center - new_margin)
                        if new_req != float(req) or new_short != float(short_req):
                            logging.info(
                                f"⚠️ MLPhysics {setup['name']} high-vol threshold bump: "
                                f"{req:.2f}/{short_req:.2f} -> {new_req:.2f}/{new_short:.2f}"
                            )
                        req = new_req
                        short_req = new_short

            req, short_req, realized_cov = self._apply_trade_budget_nudge(
                setup.get("name"),
                regime_key,
                float(req),
                float(short_req),
            )

            # High-vol directional gate config
            gate_block_sides = set()
            gate_min_conf = None
            if high_vol and gate_cfg.get("enabled"):
                overrides = gate_cfg.get("overrides", {}) or {}
                sess_cfg = overrides.get(setup.get("name"), {}) if isinstance(overrides, dict) else {}
                gate_block_sides = {str(s).upper() for s in sess_cfg.get("block", [])}
                if gate_block_sides:
                    delta = sess_cfg.get("min_conf_delta", gate_cfg.get("min_conf_delta", 0.0))
                    min_conf = sess_cfg.get("min_conf", gate_cfg.get("min_conf"))
                    if min_conf is None:
                        min_conf = float(req) + float(delta or 0.0)
                    else:
                        try:
                            min_conf = float(min_conf)
                        except Exception:
                            min_conf = float(req)
                        if delta:
                            min_conf = max(min_conf, float(req) + float(delta))
                    try:
                        max_conf = float(sess_cfg.get("max_conf", gate_cfg.get("max_conf", 0.95)))
                    except Exception:
                        max_conf = 0.95
                    gate_min_conf = min(max_conf, min_conf)

            # Align columns to what the specific brain expects
            if hasattr(model, "feature_names_in_"):
                X = X.reindex(columns=model.feature_names_in_, fill_value=0)

            gate_prob = None
            gate_margin = None
            gate_soft_penalty = 0.0
            gate_threshold_used = _safe_prob_bound(trade_gate_threshold, None)
            require_trade_gate = bool(ev_cfg_runtime.get("require_trade_gate", True))

            rework_cfg = CONFIG.get("ML_PHYSICS_DECISION_REWORK", {}) or {}
            ev_first = bool(ev_cfg_runtime.get("ev_first", rework_cfg.get("ev_first", True)))
            trade_gate_policy = str(
                ev_cfg_runtime.get("trade_gate_policy", rework_cfg.get("trade_gate_policy", "soft")) or "soft"
            ).lower()
            if trade_gate_policy not in {"hard", "soft", "off"}:
                trade_gate_policy = "soft"

            min_trade_gate_prob = _safe_prob_bound(
                ev_cfg_runtime.get("min_trade_gate_prob", None),
                None,
            )
            max_trade_gate_prob = _safe_prob_bound(
                ev_cfg_runtime.get("max_trade_gate_prob", None),
                None,
            )

            hard_min, hard_max = _resolve_gate_threshold_bounds(
                setup.get("name"),
                regime_key or ("high" if high_vol else "low"),
            )
            if hard_min is not None:
                min_trade_gate_prob = max(min_trade_gate_prob, hard_min) if min_trade_gate_prob is not None else hard_min
            if hard_max is not None:
                max_trade_gate_prob = min(max_trade_gate_prob, hard_max) if max_trade_gate_prob is not None else hard_max
            if (
                min_trade_gate_prob is not None
                and max_trade_gate_prob is not None
                and min_trade_gate_prob > max_trade_gate_prob
            ):
                min_trade_gate_prob, max_trade_gate_prob = max_trade_gate_prob, min_trade_gate_prob

            if gate_threshold_used is None:
                effective_gate_threshold = min_trade_gate_prob
            else:
                effective_gate_threshold = gate_threshold_used
                if min_trade_gate_prob is not None:
                    effective_gate_threshold = max(float(effective_gate_threshold), float(min_trade_gate_prob))
            if effective_gate_threshold is not None and max_trade_gate_prob is not None:
                effective_gate_threshold = min(float(effective_gate_threshold), float(max_trade_gate_prob))

            try:
                if trade_gate_model is not None and effective_gate_threshold is not None:
                    X_gate = X
                    if hasattr(trade_gate_model, "feature_names_in_"):
                        X_gate = X_gate.reindex(columns=trade_gate_model.feature_names_in_, fill_value=0)
                    gate_prob = float(trade_gate_model.predict_proba(X_gate)[0][1])
                    gate_margin = float(gate_prob) - float(effective_gate_threshold)
                    if require_trade_gate and trade_gate_policy == "hard" and gate_margin < 0.0:
                        logging.info(
                            f"⚠️ MLPhysics {setup['name']} tradeability gate block "
                            f"(p_trade={gate_prob:.2f} < {effective_gate_threshold:.2f})"
                        )
                        self.last_eval = {
                            "time": current_time,
                            "session": setup.get("name"),
                            "strategy": f"MLPhysics_{setup.get('name')}",
                            "regime": regime_key,
                            "decision": "blocked",
                            "blocked_reason": "tradeability_gate",
                            "trade_gate_prob": gate_prob,
                            "trade_gate_threshold": float(effective_gate_threshold),
                            "trade_gate_margin": float(gate_margin),
                            "trade_gate_policy": str(trade_gate_policy),
                            "trade_gate_required": bool(require_trade_gate),
                        }
                        self._record_trade_budget_eval(setup.get("name"), regime_key, False)
                        return None

                    if require_trade_gate and trade_gate_policy == "soft" and gate_margin < 0.0:
                        try:
                            gate_penalty_per = float(
                                ev_cfg_runtime.get(
                                    "trade_gate_penalty_points_per_prob",
                                    rework_cfg.get("trade_gate_penalty_points_per_prob", 2.0),
                                )
                                or 2.0
                            )
                        except Exception:
                            gate_penalty_per = 2.0
                        try:
                            gate_penalty_cap = float(
                                ev_cfg_runtime.get(
                                    "trade_gate_penalty_cap_points",
                                    rework_cfg.get("trade_gate_penalty_cap_points", 1.0),
                                )
                                or 1.0
                            )
                        except Exception:
                            gate_penalty_cap = 1.0
                        gate_soft_penalty = min(
                            max(gate_penalty_cap, 0.0),
                            max(0.0, -float(gate_margin)) * max(gate_penalty_per, 0.0),
                        )

                # Ask the Specialist Brain (use cached batch score in optimized backtest mode).
                prob_up = None
                if isinstance(self._opt_eval_row, (pd.Series, dict)):
                    try:
                        row_session = str(self._opt_eval_row.get("ml_session") or "")
                        row_prob_up = float(self._opt_eval_row.get("ml_prob_up"))
                        if (
                            row_session.upper() == str(setup.get("name", "")).upper()
                            and np.isfinite(row_prob_up)
                            and 0.0 <= row_prob_up <= 1.0
                        ):
                            prob_up = row_prob_up
                    except Exception:
                        prob_up = None
                if prob_up is None:
                    prob_up = model.predict_proba(X)[0][1]
                prob_down = 1.0 - prob_up

                status = "💚" if prob_up > 0.5 else "🔴"
                # Use precomputed asymmetric thresholds as exposure control.
                req = float(req)
                short_req = float(short_req)

                candidate_side_threshold = None
                if prob_up >= req:
                    candidate_side_threshold = "LONG"
                elif prob_up <= short_req:
                    candidate_side_threshold = "SHORT"

                # Compute expected edge from current dynamic bracket + calibrated probability.
                sltp_eval = dynamic_sltp_engine.calculate_dynamic_sltp(hist_df)
                tp_eval = float(sltp_eval.get("tp_dist", setup.get("tp", 0.0) or 0.0) or 0.0)
                sl_eval = float(sltp_eval.get("sl_dist", setup.get("sl", 0.0) or 0.0) or 0.0)
                ev_long_formula = (float(prob_up) * tp_eval) - (float(prob_down) * sl_eval) - float(self._roundtrip_fees_pts)
                ev_short_formula = (float(prob_down) * tp_eval) - (float(prob_up) * sl_eval) - float(self._roundtrip_fees_pts)

                def _predict_ev(model_obj) -> Optional[float]:
                    if model_obj is None:
                        return None
                    try:
                        X_ev = X
                        if hasattr(model_obj, "feature_names_in_"):
                            X_ev = X_ev.reindex(columns=model_obj.feature_names_in_, fill_value=0)
                        pred = float(model_obj.predict(X_ev)[0])
                        if not np.isfinite(pred):
                            return None
                        return pred
                    except Exception:
                        return None

                ev_source = "prob_bracket"
                ev_model_long = ev_pair.get("long") if isinstance(ev_pair, dict) else None
                ev_model_short = ev_pair.get("short") if isinstance(ev_pair, dict) else None
                use_model_ev = bool(ev_cfg_runtime.get("use_model_predictions", True))
                ev_long_model = _predict_ev(ev_model_long) if use_model_ev else None
                ev_short_model = _predict_ev(ev_model_short) if use_model_ev else None
                if ev_long_model is not None and ev_short_model is not None:
                    ev_long = float(ev_long_model)
                    ev_short = float(ev_short_model)
                    ev_source = "model"
                else:
                    ev_long = float(ev_long_formula)
                    ev_short = float(ev_short_formula)
                try:
                    max_ev_disagree = float(ev_cfg_runtime.get("max_ev_disagreement_points", 1.0e9) or 1.0e9)
                except Exception:
                    max_ev_disagree = 1.0e9
                require_ev_sign_agree = bool(ev_cfg_runtime.get("require_ev_sign_agreement", False))
                ev_gap_long = abs(float(ev_long) - float(ev_long_formula))
                ev_gap_short = abs(float(ev_short) - float(ev_short_formula))
                long_uncertain = False
                short_uncertain = False
                if ev_source == "model":
                    if require_ev_sign_agree and ((ev_long >= 0.0) != (ev_long_formula >= 0.0)):
                        long_uncertain = True
                    if require_ev_sign_agree and ((ev_short >= 0.0) != (ev_short_formula >= 0.0)):
                        short_uncertain = True
                    if ev_gap_long > max_ev_disagree:
                        long_uncertain = True
                    if ev_gap_short > max_ev_disagree:
                        short_uncertain = True
                prob_edge = abs(float(prob_up) - float(prob_down))

                ev_enabled = bool(ev_cfg_runtime.get("enabled", False))
                try:
                    min_ev_base = float(ev_cfg_runtime.get("min_ev_points", 0.0) or 0.0)
                except Exception:
                    min_ev_base = 0.0
                min_ev = float(min_ev_base)
                if high_vol:
                    try:
                        hv_min_ev = ev_cfg_runtime.get("high_vol_min_ev_points", None)
                        if hv_min_ev is not None:
                            hv_min_ev = float(hv_min_ev)
                            if np.isfinite(hv_min_ev):
                                min_ev = max(min_ev, hv_min_ev)
                    except Exception:
                        pass
                try:
                    min_prob_edge = float(ev_cfg_runtime.get("min_prob_edge", 0.0) or 0.0)
                except Exception:
                    min_prob_edge = 0.0
                require_threshold_gate = bool(ev_cfg_runtime.get("require_threshold_gate", True))
                try:
                    uncertainty_penalty = float(
                        ev_cfg_runtime.get(
                            "ev_uncertainty_penalty_points",
                            rework_cfg.get("ev_uncertainty_penalty_points", 0.35),
                        )
                        or 0.35
                    )
                except Exception:
                    uncertainty_penalty = 0.35
                uncertainty_penalty = max(0.0, uncertainty_penalty)

                long_ev_effective = float(ev_long) - (uncertainty_penalty if long_uncertain else 0.0)
                short_ev_effective = float(ev_short) - (uncertainty_penalty if short_uncertain else 0.0)
                min_ev_effective = float(min_ev + gate_soft_penalty)

                if ev_source == "model":
                    long_ev_ok = (long_ev_effective >= min_ev_effective)
                    short_ev_ok = (short_ev_effective >= min_ev_effective)
                else:
                    long_ev_ok = (long_ev_effective >= min_ev_effective) and (prob_edge >= min_prob_edge)
                    short_ev_ok = (short_ev_effective >= min_ev_effective) and (prob_edge >= min_prob_edge)
                if require_threshold_gate:
                    long_ev_ok = long_ev_ok and (candidate_side_threshold == "LONG")
                    short_ev_ok = short_ev_ok and (candidate_side_threshold == "SHORT")

                candidate_side = candidate_side_threshold
                if ev_enabled:
                    if long_ev_ok and short_ev_ok:
                        if ev_first:
                            candidate_side = "LONG" if long_ev_effective >= short_ev_effective else "SHORT"
                        else:
                            candidate_side = "LONG" if ev_long >= ev_short else "SHORT"
                    elif long_ev_ok:
                        candidate_side = "LONG"
                    elif short_ev_ok:
                        candidate_side = "SHORT"
                    else:
                        candidate_side = None

                eval_payload = {
                    "time": current_time,
                    "session": setup.get("name"),
                    "strategy": f"MLPhysics_{setup.get('name')}",
                    "regime": regime_key,
                    "high_vol": bool(high_vol),
                    "threshold": float(req),
                    "short_threshold": float(short_req),
                    "policy": policy,
                    "prob_up": float(prob_up),
                    "prob_down": float(prob_down),
                    "candidate_side": candidate_side,
                    "candidate_side_threshold": candidate_side_threshold,
                    "ev_enabled": bool(ev_enabled),
                    "ev_source": ev_source,
                    "ev_long": float(ev_long),
                    "ev_short": float(ev_short),
                    "ev_long_formula": float(ev_long_formula),
                    "ev_short_formula": float(ev_short_formula),
                    "ev_gap_long": float(ev_gap_long),
                    "ev_gap_short": float(ev_gap_short),
                    "ev_uncertain_long": bool(long_uncertain),
                    "ev_uncertain_short": bool(short_uncertain),
                    "ev_min_req": float(min_ev),
                    "ev_min_req_base": float(min_ev_base),
                    "ev_min_req_effective": float(min_ev_effective),
                    "ev_long_effective": float(long_ev_effective),
                    "ev_short_effective": float(short_ev_effective),
                    "ev_uncertainty_penalty_points": float(uncertainty_penalty),
                    "ev_first": bool(ev_first),
                    "ev_prob_edge": float(prob_edge),
                    "ev_prob_edge_req": float(min_prob_edge),
                    "tp_eval": float(tp_eval),
                    "sl_eval": float(sl_eval),
                    "fees_eval": float(self._roundtrip_fees_pts),
                    "margin_req": float(normal_margin) if normal_margin is not None else None,
                    "margin": None,
                    "gate_min_conf": float(gate_min_conf) if gate_min_conf is not None else None,
                    "gate_block_sides": sorted(gate_block_sides) if gate_block_sides else [],
                    "trade_gate_prob": float(gate_prob) if gate_prob is not None else None,
                    "trade_gate_threshold": float(effective_gate_threshold) if effective_gate_threshold is not None else None,
                    "trade_gate_margin": float(gate_margin) if gate_margin is not None else None,
                    "trade_gate_soft_penalty": float(gate_soft_penalty),
                    "trade_gate_policy": str(trade_gate_policy),
                    "trade_gate_hard_min": float(hard_min) if hard_min is not None else None,
                    "trade_gate_hard_max": float(hard_max) if hard_max is not None else None,
                    "trade_gate_required": bool(require_trade_gate),
                    "budget_cov_recent": float(realized_cov) if realized_cov is not None else None,
                    "decision": None,
                    "blocked_reason": None,
                }
                hyst_state_key = f"LEGACY_{setup.get('name')}"

                trend_ctx = ""
                if candidate_side:
                    trend_ctx = self._trend_context(hist_df, X, regime_key, high_vol)
                    if trend_ctx:
                        logging.info(
                            f"MLPhysics TrendCtx {setup['name']} {candidate_side} | {trend_ctx}"
                        )

                apply_prob_margin_with_model_ev = bool(
                    ev_cfg_runtime.get("apply_prob_margin_with_model_ev", False)
                )
                if (
                    regime_key == "normal"
                    and normal_margin is not None
                    and normal_margin > 0
                    and not (ev_enabled and ev_source == "model" and not apply_prob_margin_with_model_ev)
                ):
                    margin_val = abs(prob_up - 0.5)
                    eval_payload["margin"] = float(margin_val)
                    if margin_val < normal_margin:
                        logging.info(
                            f"⚠️ MLPhysics {setup['name']} normal margin block "
                            f"(margin {margin_val:.3f} < {normal_margin:.3f})"
                        )
                        eval_payload["decision"] = "blocked"
                        eval_payload["blocked_reason"] = "normal_margin"
                        self.last_eval = eval_payload
                        self._record_trade_budget_eval(setup.get("name"), regime_key, False)
                        return None

                logging.info(
                    f"{setup['name']} Analysis {status} | "
                    f"ConfUp: {prob_up:.1%} (Req>= {req:.1%}) | "
                    f"ConfDn: {prob_down:.1%} (Req>= {short_req:.1%}) | "
                    f"EV[{ev_source}](L/S): {ev_long:.2f}/{ev_short:.2f}"
                )

                # LONG signal: high probability of up move
                if candidate_side == "LONG":
                    apply_conf_gates_with_model_ev = bool(
                        ev_cfg_runtime.get("apply_confidence_gates_with_model_ev", False)
                    )
                    if (
                        high_vol
                        and "LONG" in gate_block_sides
                        and gate_min_conf is not None
                        and (ev_source != "model" or apply_conf_gates_with_model_ev)
                    ):
                        if prob_up < gate_min_conf:
                            logging.info(
                                f"⚠️ MLPhysics {setup['name']} LONG blocked in high vol "
                                f"(conf {prob_up:.1%} < gate {gate_min_conf:.1%})"
                            )
                            eval_payload["decision"] = "blocked"
                            eval_payload["blocked_reason"] = "high_vol_gate_long"
                            self.last_eval = eval_payload
                            self._record_trade_budget_eval(setup.get("name"), regime_key, False)
                            return None
                    hyst_ok, hyst_reason = self._hysteresis_gate(
                        hyst_state_key,
                        "LONG",
                        float(prob_up),
                        float(req),
                    )
                    if not hyst_ok:
                        eval_payload["decision"] = "no_signal"
                        eval_payload["blocked_reason"] = "hysteresis"
                        eval_payload["hysteresis_reason"] = hyst_reason
                        self.last_eval = eval_payload
                        self._record_trade_budget_eval(setup.get("name"), regime_key, False)
                        return None
                    logging.info(
                        f"🎯 {setup['name']} LONG SIGNAL CONFIRMED "
                        f"(ConfUp={prob_up:.1%}, EV={ev_long:.2f})"
                    )
                    dynamic_sltp_engine.log_params(sltp_eval)
                    eval_payload["decision"] = "signal_long"
                    self.last_eval = eval_payload
                    self._record_trade_budget_eval(setup.get("name"), regime_key, True)
                    return {
                        "strategy": f"MLPhysics_{setup['name']}",
                        "side": "LONG",
                        "tp_dist": float(sltp_eval.get('tp_dist', tp_eval)),
                        "sl_dist": float(sltp_eval.get('sl_dist', sl_eval)),
                        "ml_confidence": float(prob_up),
                        "ml_threshold": float(req),
                        "ml_prob_up": float(prob_up),
                        "ml_prob_down": float(prob_down),
                        "ml_short_threshold": float(short_req),
                        "ml_regime": regime_key,
                        "ml_high_vol": bool(high_vol),
                        "ml_candidate_side": candidate_side,
                        "ml_margin": eval_payload["margin"],
                        "ml_margin_req": eval_payload["margin_req"],
                        "ml_gate_min_conf": eval_payload["gate_min_conf"],
                        "ml_gate_block_sides": eval_payload["gate_block_sides"],
                        "ml_trade_gate_prob": eval_payload["trade_gate_prob"],
                        "ml_trade_gate_threshold": eval_payload["trade_gate_threshold"],
                        "ml_trade_gate_margin": eval_payload["trade_gate_margin"],
                        "ml_trade_gate_soft_penalty": eval_payload["trade_gate_soft_penalty"],
                        "ml_trade_gate_policy": eval_payload["trade_gate_policy"],
                        "ml_trade_gate_required": eval_payload["trade_gate_required"],
                        "ml_budget_cov_recent": eval_payload["budget_cov_recent"],
                        "ml_ev_long": eval_payload["ev_long"],
                        "ml_ev_short": eval_payload["ev_short"],
                        "ml_ev_long_effective": eval_payload["ev_long_effective"],
                        "ml_ev_short_effective": eval_payload["ev_short_effective"],
                        "ml_ev_source": eval_payload["ev_source"],
                        "ml_ev_min_req": eval_payload["ev_min_req"],
                        "ml_ev_min_req_effective": eval_payload["ev_min_req_effective"],
                        "ml_ev_prob_edge": eval_payload["ev_prob_edge"],
                        "ml_ev_prob_edge_req": eval_payload["ev_prob_edge_req"],
                        "ml_decision": eval_payload["decision"],
                        "ml_blocked_reason": eval_payload["blocked_reason"],
                        "ml_hysteresis_reason": eval_payload.get("hysteresis_reason"),
                    }

                # SHORT signal: high probability of down move (low prob_up)
                elif candidate_side == "SHORT":
                    apply_conf_gates_with_model_ev = bool(
                        ev_cfg_runtime.get("apply_confidence_gates_with_model_ev", False)
                    )
                    if (
                        high_vol
                        and "SHORT" in gate_block_sides
                        and gate_min_conf is not None
                        and (ev_source != "model" or apply_conf_gates_with_model_ev)
                    ):
                        if prob_down < gate_min_conf:
                            logging.info(
                                f"⚠️ MLPhysics {setup['name']} SHORT blocked in high vol "
                                f"(conf {prob_down:.1%} < gate {gate_min_conf:.1%})"
                            )
                            eval_payload["decision"] = "blocked"
                            eval_payload["blocked_reason"] = "high_vol_gate_short"
                            self.last_eval = eval_payload
                            self._record_trade_budget_eval(setup.get("name"), regime_key, False)
                            return None
                    hyst_ok, hyst_reason = self._hysteresis_gate(
                        hyst_state_key,
                        "SHORT",
                        float(prob_down),
                        float(short_req),
                    )
                    if not hyst_ok:
                        eval_payload["decision"] = "no_signal"
                        eval_payload["blocked_reason"] = "hysteresis"
                        eval_payload["hysteresis_reason"] = hyst_reason
                        self.last_eval = eval_payload
                        self._record_trade_budget_eval(setup.get("name"), regime_key, False)
                        return None
                    logging.info(
                        f"🎯 {setup['name']} SHORT SIGNAL CONFIRMED "
                        f"(ConfDn={prob_down:.1%}, EV={ev_short:.2f})"
                    )
                    dynamic_sltp_engine.log_params(sltp_eval)
                    eval_payload["decision"] = "signal_short"
                    self.last_eval = eval_payload
                    self._record_trade_budget_eval(setup.get("name"), regime_key, True)
                    return {
                        "strategy": f"MLPhysics_{setup['name']}",
                        "side": "SHORT",
                        "tp_dist": float(sltp_eval.get('tp_dist', tp_eval)),
                        "sl_dist": float(sltp_eval.get('sl_dist', sl_eval)),
                        "ml_confidence": float(prob_down),
                        "ml_threshold": float(req),
                        "ml_prob_up": float(prob_up),
                        "ml_prob_down": float(prob_down),
                        "ml_short_threshold": float(short_req),
                        "ml_regime": regime_key,
                        "ml_high_vol": bool(high_vol),
                        "ml_candidate_side": candidate_side,
                        "ml_margin": eval_payload["margin"],
                        "ml_margin_req": eval_payload["margin_req"],
                        "ml_gate_min_conf": eval_payload["gate_min_conf"],
                        "ml_gate_block_sides": eval_payload["gate_block_sides"],
                        "ml_trade_gate_prob": eval_payload["trade_gate_prob"],
                        "ml_trade_gate_threshold": eval_payload["trade_gate_threshold"],
                        "ml_trade_gate_margin": eval_payload["trade_gate_margin"],
                        "ml_trade_gate_soft_penalty": eval_payload["trade_gate_soft_penalty"],
                        "ml_trade_gate_policy": eval_payload["trade_gate_policy"],
                        "ml_trade_gate_required": eval_payload["trade_gate_required"],
                        "ml_budget_cov_recent": eval_payload["budget_cov_recent"],
                        "ml_ev_long": eval_payload["ev_long"],
                        "ml_ev_short": eval_payload["ev_short"],
                        "ml_ev_long_effective": eval_payload["ev_long_effective"],
                        "ml_ev_short_effective": eval_payload["ev_short_effective"],
                        "ml_ev_source": eval_payload["ev_source"],
                        "ml_ev_min_req": eval_payload["ev_min_req"],
                        "ml_ev_min_req_effective": eval_payload["ev_min_req_effective"],
                        "ml_ev_prob_edge": eval_payload["ev_prob_edge"],
                        "ml_ev_prob_edge_req": eval_payload["ev_prob_edge_req"],
                        "ml_decision": eval_payload["decision"],
                        "ml_blocked_reason": eval_payload["blocked_reason"],
                        "ml_hysteresis_reason": eval_payload.get("hysteresis_reason"),
                    }

                eval_payload["decision"] = "no_signal"
                if candidate_side is None:
                    self._hysteresis_gate(hyst_state_key, None, 0.0, 0.0)
                if ev_enabled and candidate_side is None:
                    if require_trade_gate and trade_gate_policy == "soft" and gate_soft_penalty > 0.0:
                        eval_payload["blocked_reason"] = "tradeability_gate_soft"
                    elif require_threshold_gate and candidate_side_threshold is None:
                        eval_payload["blocked_reason"] = "threshold_gate"
                    elif ev_source == "model" and (long_uncertain or short_uncertain):
                        eval_payload["blocked_reason"] = "ev_uncertainty"
                    else:
                        eval_payload["blocked_reason"] = "ev_filter"
                self.last_eval = eval_payload
                self._record_trade_budget_eval(setup.get("name"), regime_key, False)

            except Exception as e:
                logging.error(f"Prediction Error: {e}")
                self.last_eval = {
                    "time": current_time,
                    "session": setup.get("name") if setup else None,
                    "strategy": f"MLPhysics_{setup.get('name')}" if setup else "MLPhysics",
                    "decision": "error",
                    "error": str(e),
                }
                self._record_trade_budget_eval(setup.get("name"), regime_key, False)

        return None
