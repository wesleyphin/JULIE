import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from tools.regimeadaptive_filterless_runner import COMBO_SPACE, _combo_key_from_id


GATE_FEATURE_COLUMNS = [
    "final_side",
    "original_side",
    "reverted",
    "quarter",
    "week_in_month",
    "day_of_week",
    "session_code",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "rule_type_code",
    "sma_fast_period",
    "sma_slow_period",
    "cross_atr_mult",
    "pattern_lookback",
    "touch_atr_mult",
    "strength",
    "strength_atr",
    "close_fast_dist_atr",
    "close_slow_dist_atr",
    "fast_slow_spread_atr",
    "bar_range_atr",
    "body_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "atr_pct",
    "return_1",
    "return_5",
    "return_15",
    "vol_30",
    "vol_ratio",
    "range_vs_mean",
]


RULE_TYPE_CODE = {
    "pullback": 0,
    "continuation": 1,
    "breakout": 2,
}

_COMBO_CONTEXT_CACHE: Optional[dict[str, np.ndarray]] = None


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _safe_div(numerator, denominator, default: float = 0.0):
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.full(np.broadcast(num, den).shape, float(default), dtype=np.float64)
    valid = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    if np.any(valid):
        out[valid] = num[valid] / den[valid]
    return out


def _combo_context_cache() -> dict[str, np.ndarray]:
    global _COMBO_CONTEXT_CACHE
    if _COMBO_CONTEXT_CACHE is not None:
        return _COMBO_CONTEXT_CACHE
    quarter = np.zeros(COMBO_SPACE, dtype=np.int8)
    week = np.zeros(COMBO_SPACE, dtype=np.int8)
    day = np.zeros(COMBO_SPACE, dtype=np.int8)
    session = np.zeros(COMBO_SPACE, dtype=np.int8)
    for combo_id in range(COMBO_SPACE):
        combo_key = _combo_key_from_id(combo_id)
        parts = [str(part or "").strip().upper() for part in combo_key.split("_")]
        if len(parts) != 4:
            continue
        quarter[combo_id] = int(parts[0][1:]) if parts[0].startswith("Q") else 0
        week[combo_id] = int(parts[1][1:]) if parts[1].startswith("W") else 0
        day_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
        day[combo_id] = int(day_map.get(parts[2], 0))
        session_map = {"ASIA": 0, "LONDON": 1, "NY_AM": 2, "NY_PM": 3, "CLOSED": 4}
        session[combo_id] = int(session_map.get(parts[3], 4))
    _COMBO_CONTEXT_CACHE = {
        "quarter": quarter,
        "week": week,
        "day": day,
        "session": session,
    }
    return _COMBO_CONTEXT_CACHE


def resolve_gate_model_path(base_path: Path, path_text: str) -> Optional[Path]:
    raw = str(path_text or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_path.parent / path
    path = path.resolve()
    return path if path.is_file() else None


class RegimeAdaptiveGateModel:
    def __init__(self, model, feature_columns: list[str], threshold: float):
        self.model = model
        self.feature_columns = [str(col) for col in feature_columns] if feature_columns else list(GATE_FEATURE_COLUMNS)
        self.threshold = float(threshold)

    def predict_proba_frame(self, features: pd.DataFrame) -> np.ndarray:
        if features is None or features.empty:
            return np.empty(0, dtype=np.float64)
        x_df = (
            features.reindex(columns=self.feature_columns, fill_value=0.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        probs = self.model.predict_proba(x_df.to_numpy(dtype=np.float32, copy=False))
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return np.asarray(probs[:, 1], dtype=np.float64)
        return np.asarray(probs, dtype=np.float64).reshape(-1)

    def predict_proba_row(self, feature_row: dict) -> float:
        features = pd.DataFrame([feature_row], columns=self.feature_columns)
        probs = self.predict_proba_frame(features)
        return float(probs[0]) if len(probs) else 0.0


def load_regimeadaptive_gate_model(artifact_path: Path, gate_config: dict) -> Optional[RegimeAdaptiveGateModel]:
    if not isinstance(gate_config, dict) or not bool(gate_config.get("enabled", False)):
        return None
    model_path = resolve_gate_model_path(artifact_path, str(gate_config.get("model_path", "") or ""))
    if model_path is None:
        return None
    try:
        bundle = joblib.load(model_path)
    except Exception:
        return None
    model = bundle.get("model") if isinstance(bundle, dict) and "model" in bundle else bundle
    feature_columns = (
        [str(col) for col in bundle.get("feature_columns", [])]
        if isinstance(bundle, dict) and isinstance(bundle.get("feature_columns"), list)
        else list(gate_config.get("feature_columns", []) or GATE_FEATURE_COLUMNS)
    )
    configured_threshold = gate_config.get("threshold")
    if configured_threshold is not None:
        threshold = _safe_float(configured_threshold, 0.5)
    elif isinstance(bundle, dict):
        threshold = _safe_float(bundle.get("threshold"), 0.5)
    else:
        threshold = 0.5
    if model is None:
        return None
    return RegimeAdaptiveGateModel(model=model, feature_columns=feature_columns, threshold=threshold)


def build_gate_feature_frame_for_positions(
    index: pd.DatetimeIndex,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    combo_ids: np.ndarray,
    signal_side: np.ndarray,
    original_side: np.ndarray,
    selected_rule_index: np.ndarray,
    positions: np.ndarray,
    rule_order: list[str],
    rule_catalog: dict[str, dict],
    rolling_cache: dict[int, np.ndarray],
    atr_cache: dict[int, np.ndarray],
    long_strength_matrix: np.ndarray,
    short_strength_matrix: np.ndarray,
    vol_window: int = 30,
    vol_median_window: int = 120,
    range_window: int = 20,
) -> pd.DataFrame:
    if positions.size == 0:
        return pd.DataFrame(columns=GATE_FEATURE_COLUMNS)

    close_series = pd.Series(closes, copy=False)
    vol_arr = close_series.pct_change().rolling(int(vol_window)).std().to_numpy(dtype=np.float64)
    vol_median_arr = (
        pd.Series(vol_arr, copy=False).rolling(int(vol_median_window)).median().to_numpy(dtype=np.float64)
    )
    bar_range_arr = np.asarray(highs - lows, dtype=np.float64)
    range_mean_arr = pd.Series(bar_range_arr, copy=False).rolling(int(range_window)).mean().to_numpy(dtype=np.float64)
    ret_1_arr = close_series.pct_change(1).to_numpy(dtype=np.float64)
    ret_5_arr = close_series.pct_change(5).to_numpy(dtype=np.float64)
    ret_15_arr = close_series.pct_change(15).to_numpy(dtype=np.float64)
    body_arr = np.asarray(closes - opens, dtype=np.float64)
    upper_wick_arr = np.asarray(highs - np.maximum(opens, closes), dtype=np.float64)
    lower_wick_arr = np.asarray(np.minimum(opens, closes) - lows, dtype=np.float64)

    pos = np.asarray(positions, dtype=np.int32)
    rule_idx = np.asarray(selected_rule_index[pos], dtype=np.int16)
    if np.any(rule_idx < 0):
        raise ValueError("Selected rule index contains negative values for gated positions.")
    combo_context = _combo_context_cache()
    combo_pos = np.asarray(combo_ids[pos], dtype=np.int16)

    selected_atr = np.full(len(pos), np.nan, dtype=np.float64)
    selected_fast = np.full(len(pos), np.nan, dtype=np.float64)
    selected_slow = np.full(len(pos), np.nan, dtype=np.float64)
    rule_type_code = np.zeros(len(pos), dtype=np.int8)
    sma_fast_period = np.zeros(len(pos), dtype=np.int16)
    sma_slow_period = np.zeros(len(pos), dtype=np.int16)
    cross_atr_mult = np.zeros(len(pos), dtype=np.float64)
    pattern_lookback = np.zeros(len(pos), dtype=np.int16)
    touch_atr_mult = np.zeros(len(pos), dtype=np.float64)
    rule_id_text = np.empty(len(pos), dtype=object)

    for unique_rule_idx in np.unique(rule_idx):
        rule_mask = rule_idx == unique_rule_idx
        rule_id = str(rule_order[int(unique_rule_idx)])
        rule_payload = rule_catalog[rule_id]
        fast_period = int(rule_payload.get("sma_fast", 20) or 20)
        slow_period = int(rule_payload.get("sma_slow", 200) or 200)
        atr_period = int(rule_payload.get("atr_period", 20) or 20)
        selected_fast[rule_mask] = rolling_cache[fast_period][pos[rule_mask]]
        selected_slow[rule_mask] = rolling_cache[slow_period][pos[rule_mask]]
        selected_atr[rule_mask] = atr_cache[atr_period][pos[rule_mask]]
        rule_type_code[rule_mask] = int(RULE_TYPE_CODE.get(str(rule_payload.get("rule_type", "pullback") or "pullback").strip().lower(), 0))
        sma_fast_period[rule_mask] = fast_period
        sma_slow_period[rule_mask] = slow_period
        cross_atr_mult[rule_mask] = _safe_float(rule_payload.get("cross_atr_mult"), 0.0)
        pattern_lookback[rule_mask] = int(rule_payload.get("pattern_lookback", 0) or 0)
        touch_atr_mult[rule_mask] = _safe_float(rule_payload.get("touch_atr_mult"), 0.0)
        rule_id_text[rule_mask] = rule_id

    original_sel = np.asarray(original_side[pos], dtype=np.int8)
    final_sel = np.asarray(signal_side[pos], dtype=np.int8)
    long_strength = long_strength_matrix[rule_idx, pos]
    short_strength = short_strength_matrix[rule_idx, pos]
    selected_strength = np.where(original_sel > 0, long_strength, short_strength).astype(np.float64)

    ts = pd.DatetimeIndex(index[pos])
    hours = np.asarray(ts.hour, dtype=np.int16)
    minutes = np.asarray(ts.minute, dtype=np.int16)
    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    minute_sin = np.sin(2.0 * np.pi * minutes / 60.0)
    minute_cos = np.cos(2.0 * np.pi * minutes / 60.0)

    features = pd.DataFrame(
        {
            "final_side": final_sel.astype(np.float64),
            "original_side": original_sel.astype(np.float64),
            "reverted": (final_sel != original_sel).astype(np.float64),
            "quarter": combo_context["quarter"][combo_pos].astype(np.float64),
            "week_in_month": combo_context["week"][combo_pos].astype(np.float64),
            "day_of_week": combo_context["day"][combo_pos].astype(np.float64),
            "session_code": combo_context["session"][combo_pos].astype(np.float64),
            "hour_sin": hour_sin.astype(np.float64),
            "hour_cos": hour_cos.astype(np.float64),
            "minute_sin": minute_sin.astype(np.float64),
            "minute_cos": minute_cos.astype(np.float64),
            "rule_type_code": rule_type_code.astype(np.float64),
            "sma_fast_period": sma_fast_period.astype(np.float64),
            "sma_slow_period": sma_slow_period.astype(np.float64),
            "cross_atr_mult": cross_atr_mult.astype(np.float64),
            "pattern_lookback": pattern_lookback.astype(np.float64),
            "touch_atr_mult": touch_atr_mult.astype(np.float64),
            "strength": selected_strength.astype(np.float64),
            "strength_atr": _safe_div(selected_strength, selected_atr),
            "close_fast_dist_atr": _safe_div(closes[pos] - selected_fast, selected_atr),
            "close_slow_dist_atr": _safe_div(closes[pos] - selected_slow, selected_atr),
            "fast_slow_spread_atr": _safe_div(selected_fast - selected_slow, selected_atr),
            "bar_range_atr": _safe_div(bar_range_arr[pos], selected_atr),
            "body_atr": _safe_div(body_arr[pos], selected_atr),
            "upper_wick_atr": _safe_div(upper_wick_arr[pos], selected_atr),
            "lower_wick_atr": _safe_div(lower_wick_arr[pos], selected_atr),
            "atr_pct": _safe_div(selected_atr, closes[pos]),
            "return_1": np.nan_to_num(ret_1_arr[pos], nan=0.0, posinf=0.0, neginf=0.0),
            "return_5": np.nan_to_num(ret_5_arr[pos], nan=0.0, posinf=0.0, neginf=0.0),
            "return_15": np.nan_to_num(ret_15_arr[pos], nan=0.0, posinf=0.0, neginf=0.0),
            "vol_30": np.nan_to_num(vol_arr[pos], nan=0.0, posinf=0.0, neginf=0.0),
            "vol_ratio": np.nan_to_num(_safe_div(vol_arr[pos], vol_median_arr[pos]), nan=0.0, posinf=0.0, neginf=0.0),
            "range_vs_mean": np.nan_to_num(_safe_div(bar_range_arr[pos], range_mean_arr[pos]), nan=0.0, posinf=0.0, neginf=0.0),
        },
        index=ts,
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features.attrs["rule_ids"] = list(rule_id_text)
    return features.reindex(columns=GATE_FEATURE_COLUMNS, fill_value=0.0)


def build_runtime_gate_feature_row(
    ts: pd.Timestamp,
    combo_key: str,
    final_signal: str,
    original_signal: str,
    rule_payload: dict,
    strength: float,
    sma_fast_value: float,
    sma_slow_value: float,
    atr_value: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    ret_1: float,
    ret_5: float,
    ret_15: float,
    vol_value: float,
    vol_median_value: float,
    range_mean_value: float,
) -> dict:
    parts = [str(part or "").strip().upper() for part in str(combo_key or "").split("_")]
    quarter = int(parts[0][1:]) if len(parts) == 4 and parts[0].startswith("Q") else 0
    week = int(parts[1][1:]) if len(parts) == 4 and parts[1].startswith("W") else 0
    day_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
    session_map = {"ASIA": 0, "LONDON": 1, "NY_AM": 2, "NY_PM": 3, "CLOSED": 4}
    day_code = int(day_map.get(parts[2], 0)) if len(parts) == 4 else 0
    session_code = int(session_map.get(parts[3], 4)) if len(parts) == 4 else 4
    atr = _safe_float(atr_value, 0.0)
    bar_range = _safe_float(bar_high, 0.0) - _safe_float(bar_low, 0.0)
    body = _safe_float(bar_close, 0.0) - _safe_float(bar_open, 0.0)
    upper_wick = _safe_float(bar_high, 0.0) - max(_safe_float(bar_open, 0.0), _safe_float(bar_close, 0.0))
    lower_wick = min(_safe_float(bar_open, 0.0), _safe_float(bar_close, 0.0)) - _safe_float(bar_low, 0.0)
    hour = int(ts.hour)
    minute = int(ts.minute)
    final_side_num = 1.0 if str(final_signal).upper() == "LONG" else -1.0
    original_side_num = 1.0 if str(original_signal).upper() == "LONG" else -1.0
    return {
        "final_side": final_side_num,
        "original_side": original_side_num,
        "reverted": 1.0 if final_side_num != original_side_num else 0.0,
        "quarter": float(quarter),
        "week_in_month": float(week),
        "day_of_week": float(day_code),
        "session_code": float(session_code),
        "hour_sin": float(math.sin(2.0 * math.pi * hour / 24.0)),
        "hour_cos": float(math.cos(2.0 * math.pi * hour / 24.0)),
        "minute_sin": float(math.sin(2.0 * math.pi * minute / 60.0)),
        "minute_cos": float(math.cos(2.0 * math.pi * minute / 60.0)),
        "rule_type_code": float(RULE_TYPE_CODE.get(str(rule_payload.get("rule_type", "pullback") or "pullback").strip().lower(), 0)),
        "sma_fast_period": float(int(rule_payload.get("sma_fast", 20) or 20)),
        "sma_slow_period": float(int(rule_payload.get("sma_slow", 200) or 200)),
        "cross_atr_mult": _safe_float(rule_payload.get("cross_atr_mult"), 0.0),
        "pattern_lookback": float(int(rule_payload.get("pattern_lookback", 0) or 0)),
        "touch_atr_mult": _safe_float(rule_payload.get("touch_atr_mult"), 0.0),
        "strength": _safe_float(strength, 0.0),
        "strength_atr": float(_safe_div([strength], [atr])[0]),
        "close_fast_dist_atr": float(_safe_div([_safe_float(bar_close) - _safe_float(sma_fast_value)], [atr])[0]),
        "close_slow_dist_atr": float(_safe_div([_safe_float(bar_close) - _safe_float(sma_slow_value)], [atr])[0]),
        "fast_slow_spread_atr": float(_safe_div([_safe_float(sma_fast_value) - _safe_float(sma_slow_value)], [atr])[0]),
        "bar_range_atr": float(_safe_div([bar_range], [atr])[0]),
        "body_atr": float(_safe_div([body], [atr])[0]),
        "upper_wick_atr": float(_safe_div([upper_wick], [atr])[0]),
        "lower_wick_atr": float(_safe_div([lower_wick], [atr])[0]),
        "atr_pct": float(_safe_div([atr], [_safe_float(bar_close, 0.0)])[0]),
        "return_1": _safe_float(ret_1, 0.0),
        "return_5": _safe_float(ret_5, 0.0),
        "return_15": _safe_float(ret_15, 0.0),
        "vol_30": _safe_float(vol_value, 0.0),
        "vol_ratio": float(_safe_div([vol_value], [vol_median_value])[0]),
        "range_vs_mean": float(_safe_div([bar_range], [range_mean_value])[0]),
    }
