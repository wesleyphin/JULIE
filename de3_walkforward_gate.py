import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import joblib
import numpy as np
import pandas as pd


NUMERIC_FEATURE_COLUMNS = [
    "month",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "de3_final_score",
    "de3_selection_score",
    "de3_edge_points",
    "de3_edge_confidence",
    "de3_v4_route_confidence",
    "de3_v4_execution_quality_score",
    "de3_v4_lane_candidate_count",
    "de3_v4_selected_sl",
    "de3_v4_selected_tp",
    "de3_v4_entry_model_score",
    "de3_v4_entry_model_threshold",
    "de3_v4_entry_model_margin",
    "de3_entry_atr14",
    "de3_entry_body1_ratio",
    "de3_entry_body_pos1",
    "de3_entry_close_pos1",
    "de3_entry_dist_high5_atr",
    "de3_entry_dist_low5_atr",
    "de3_entry_down3",
    "de3_entry_flips5",
    "de3_entry_lower_wick_ratio",
    "de3_entry_range10_atr",
    "de3_entry_ret1_atr",
    "de3_entry_upper1_ratio",
    "de3_entry_upper_wick_ratio",
    "de3_entry_vol1_rel20",
    "sl_dist",
    "tp_dist",
    "de3_profit_gate_soft_pass",
    "de3_profit_gate_catastrophic_block",
    "de3_exec_soft_pass",
    "de3_exec_hard_limit_triggered",
    "lane_ctx_history_count",
    "lane_ctx_history_avg_pnl_per_contract",
    "lane_ctx_history_winrate",
    "lane_ctx_history_pnl_std",
    "variant_history_count",
    "variant_history_avg_pnl_per_contract",
    "variant_history_winrate",
    "variant_history_pnl_std",
]

CAT_FEATURE_COLUMNS = [
    "session",
    "side",
    "de3_timeframe",
    "de3_strategy_type",
    "de3_v4_selected_lane",
    "de3_v4_selected_variant_id",
    "de3_v4_execution_policy_tier",
    "de3_v4_entry_model_scope",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value or "").strip().lower()
    return 1.0 if text in {"1", "true", "yes", "y", "t"} else 0.0


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _parse_timestamp(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, dt.datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(text)
    except Exception:
        return None


def _cyclical(value: int, period: int) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(value % period) / float(period)
    return float(math.sin(angle)), float(math.cos(angle))


def _history_stats(values: Any) -> dict[str, float]:
    if values is None:
        return {
            "count": 0.0,
            "avg": 0.0,
            "winrate": 0.0,
            "std": 0.0,
        }
    arr = [float(v) for v in values if math.isfinite(_safe_float(v, float("nan")))]
    if not arr:
        return {
            "count": 0.0,
            "avg": 0.0,
            "winrate": 0.0,
            "std": 0.0,
        }
    avg = float(sum(arr) / len(arr))
    winrate = float(sum(1 for v in arr if v > 0.0) / len(arr))
    std = float(np.std(np.asarray(arr, dtype=np.float64), ddof=1)) if len(arr) > 1 else 0.0
    return {
        "count": float(len(arr)),
        "avg": avg,
        "winrate": winrate,
        "std": std,
    }


def de3_variant_key(payload: Mapping[str, Any]) -> str:
    return _safe_str(
        payload.get("de3_v4_selected_variant_id")
        or payload.get("sub_strategy")
        or ""
    )


def de3_lane_context_key(payload: Mapping[str, Any]) -> str:
    timeframe = _safe_str(payload.get("de3_timeframe") or payload.get("timeframe"))
    session = _safe_str(payload.get("session"))
    lane = _safe_str(payload.get("de3_v4_selected_lane"))
    parts = [part for part in (timeframe, session, lane) if part]
    return "|".join(parts)


def build_feature_row(
    payload: Mapping[str, Any],
    *,
    timestamp: Any = None,
    lane_ctx_history: Optional[Mapping[str, Any]] = None,
    variant_history: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    ts = _parse_timestamp(timestamp)
    if ts is None:
        ts = _parse_timestamp(payload.get("entry_time"))
    if ts is None:
        ts = _parse_timestamp(payload.get("signal_time"))
    month = int(ts.month) if ts is not None else 0
    day_of_week = int(ts.weekday()) if ts is not None else 0
    hour_sin, hour_cos = _cyclical(int(ts.hour) if ts is not None else 0, 24)
    minute_sin, minute_cos = _cyclical(int(ts.minute) if ts is not None else 0, 60)

    entry_model_score = _safe_float(payload.get("de3_v4_entry_model_score"), 0.0)
    entry_model_threshold = _safe_float(payload.get("de3_v4_entry_model_threshold"), 0.0)
    entry_model_margin = float(entry_model_score - entry_model_threshold)

    lane_key = de3_lane_context_key(payload)
    variant_key = de3_variant_key(payload)
    lane_stats = _history_stats(lane_ctx_history.get(lane_key) if lane_ctx_history and lane_key else None)
    variant_stats = _history_stats(variant_history.get(variant_key) if variant_history and variant_key else None)

    return {
        "month": float(month),
        "day_of_week": float(day_of_week),
        "hour_sin": float(hour_sin),
        "hour_cos": float(hour_cos),
        "minute_sin": float(minute_sin),
        "minute_cos": float(minute_cos),
        "session": _safe_str(payload.get("session")),
        "side": _safe_str(payload.get("side")),
        "de3_timeframe": _safe_str(payload.get("de3_timeframe") or payload.get("timeframe")),
        "de3_strategy_type": _safe_str(payload.get("de3_strategy_type") or payload.get("strategy_type")),
        "de3_v4_selected_lane": _safe_str(payload.get("de3_v4_selected_lane")),
        "de3_v4_selected_variant_id": variant_key,
        "de3_v4_execution_policy_tier": _safe_str(payload.get("de3_v4_execution_policy_tier")),
        "de3_v4_entry_model_scope": _safe_str(payload.get("de3_v4_entry_model_scope")),
        "de3_final_score": _safe_float(payload.get("de3_final_score"), 0.0),
        "de3_selection_score": _safe_float(payload.get("de3_selection_score"), 0.0),
        "de3_edge_points": _safe_float(payload.get("de3_edge_points"), 0.0),
        "de3_edge_confidence": _safe_float(payload.get("de3_edge_confidence"), 0.0),
        "de3_v4_route_confidence": _safe_float(payload.get("de3_v4_route_confidence"), 0.0),
        "de3_v4_execution_quality_score": _safe_float(payload.get("de3_v4_execution_quality_score"), 0.0),
        "de3_v4_lane_candidate_count": _safe_float(payload.get("de3_v4_lane_candidate_count"), 0.0),
        "de3_v4_selected_sl": _safe_float(payload.get("de3_v4_selected_sl"), 0.0),
        "de3_v4_selected_tp": _safe_float(payload.get("de3_v4_selected_tp"), 0.0),
        "de3_v4_entry_model_score": float(entry_model_score),
        "de3_v4_entry_model_threshold": float(entry_model_threshold),
        "de3_v4_entry_model_margin": float(entry_model_margin),
        "de3_entry_atr14": _safe_float(payload.get("de3_entry_atr14"), 0.0),
        "de3_entry_body1_ratio": _safe_float(payload.get("de3_entry_body1_ratio"), 0.0),
        "de3_entry_body_pos1": _safe_float(payload.get("de3_entry_body_pos1"), 0.0),
        "de3_entry_close_pos1": _safe_float(payload.get("de3_entry_close_pos1"), 0.0),
        "de3_entry_dist_high5_atr": _safe_float(payload.get("de3_entry_dist_high5_atr"), 0.0),
        "de3_entry_dist_low5_atr": _safe_float(payload.get("de3_entry_dist_low5_atr"), 0.0),
        "de3_entry_down3": _safe_float(payload.get("de3_entry_down3"), 0.0),
        "de3_entry_flips5": _safe_float(payload.get("de3_entry_flips5"), 0.0),
        "de3_entry_lower_wick_ratio": _safe_float(payload.get("de3_entry_lower_wick_ratio"), 0.0),
        "de3_entry_range10_atr": _safe_float(payload.get("de3_entry_range10_atr"), 0.0),
        "de3_entry_ret1_atr": _safe_float(payload.get("de3_entry_ret1_atr"), 0.0),
        "de3_entry_upper1_ratio": _safe_float(payload.get("de3_entry_upper1_ratio"), 0.0),
        "de3_entry_upper_wick_ratio": _safe_float(payload.get("de3_entry_upper_wick_ratio"), 0.0),
        "de3_entry_vol1_rel20": _safe_float(payload.get("de3_entry_vol1_rel20"), 0.0),
        "sl_dist": _safe_float(payload.get("sl_dist"), 0.0),
        "tp_dist": _safe_float(payload.get("tp_dist"), 0.0),
        "de3_profit_gate_soft_pass": _safe_bool(payload.get("de3_v4_profit_gate_soft_pass")),
        "de3_profit_gate_catastrophic_block": _safe_bool(payload.get("de3_v4_profit_gate_catastrophic_block")),
        "de3_exec_soft_pass": _safe_bool(payload.get("de3_v4_execution_policy_soft_pass")),
        "de3_exec_hard_limit_triggered": _safe_bool(payload.get("de3_v4_execution_policy_hard_limit_triggered")),
        "lane_ctx_history_count": float(lane_stats["count"]),
        "lane_ctx_history_avg_pnl_per_contract": float(lane_stats["avg"]),
        "lane_ctx_history_winrate": float(lane_stats["winrate"]),
        "lane_ctx_history_pnl_std": float(lane_stats["std"]),
        "variant_history_count": float(variant_stats["count"]),
        "variant_history_avg_pnl_per_contract": float(variant_stats["avg"]),
        "variant_history_winrate": float(variant_stats["winrate"]),
        "variant_history_pnl_std": float(variant_stats["std"]),
    }


def build_model_frame(
    rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    model_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        raw_df = rows.copy()
    else:
        raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return pd.DataFrame(columns=list(model_columns or []))

    for col in NUMERIC_FEATURE_COLUMNS:
        if col not in raw_df.columns:
            raw_df[col] = 0.0
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in CAT_FEATURE_COLUMNS:
        if col not in raw_df.columns:
            raw_df[col] = ""
        raw_df[col] = raw_df[col].fillna("").astype(str)

    x_df = pd.get_dummies(
        raw_df[NUMERIC_FEATURE_COLUMNS + CAT_FEATURE_COLUMNS],
        columns=CAT_FEATURE_COLUMNS,
        dummy_na=False,
    )
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if model_columns is None:
        return x_df
    return x_df.reindex(columns=list(model_columns), fill_value=0.0)


def build_model_vector(
    row: Mapping[str, Any],
    *,
    model_columns: list[str],
    model_column_index: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    col_index = (
        dict(model_column_index)
        if model_column_index is not None
        else {str(col): idx for idx, col in enumerate(model_columns)}
    )
    vec = np.zeros((1, len(model_columns)), dtype=np.float32)
    for col in NUMERIC_FEATURE_COLUMNS:
        idx = col_index.get(col)
        if idx is None:
            continue
        vec[0, int(idx)] = np.float32(_safe_float(row.get(col), 0.0))
    for col in CAT_FEATURE_COLUMNS:
        value = _safe_str(row.get(col))
        if not value:
            continue
        idx = col_index.get(f"{col}_{value}")
        if idx is not None:
            vec[0, int(idx)] = np.float32(1.0)
    return vec


def load_artifact(path_text: str | Path) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_model_bundle(path_text: str | Path) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    try:
        bundle = joblib.load(path)
    except Exception:
        return {}
    return bundle if isinstance(bundle, dict) else {"model": bundle}
