import argparse
import copy
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from de3_v4_decision_policy_trainer import (
    _bucket_balance,
    _bucket_body_ratio,
    _bucket_close_pos,
    _bucket_down3,
    _bucket_range_ratio,
    _bucket_vol_ratio,
    _directionless_strategy_style,
)


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"File not found: {path_arg}")


def _resolve_output_dir(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        raw = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(raw):
        return float(default)
    return float(raw)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _bucket_key(row: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = []
    for field in fields:
        key = str(field or "").strip()
        if not key:
            continue
        raw = row.get(key, "")
        text = str(raw).strip().lower()
        if not text or text == "nan":
            return ""
        parts.append(text)
    return "|".join(parts)


def _normalize_text(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    return "" if text in {"", "nan", "nat", "none"} else text


def _derive_session_substate(*, session_text: str, timestamp_value: Any) -> str:
    ts = pd.to_datetime(timestamp_value, errors="coerce", utc=True)
    if pd.isna(ts):
        return ""
    ts = ts.tz_convert("America/New_York")
    session = str(session_text or "").strip()
    start_hour = int(ts.hour)
    if "-" in session:
        try:
            start_hour = int(str(session).split("-", 1)[0])
        except Exception:
            start_hour = int(ts.hour)
    elapsed_minutes = ((int(ts.hour) - int(start_hour)) * 60) + int(ts.minute)
    if elapsed_minutes < 0:
        elapsed_minutes = int(ts.minute)
    if elapsed_minutes < 60:
        return "open"
    if elapsed_minutes < 120:
        return "mid"
    return "late"


def _prepare_frame(data: pd.DataFrame) -> pd.DataFrame:
    local = data.copy()
    if local.empty:
        return local
    if "timestamp" not in local.columns:
        local["timestamp"] = pd.NaT
    local["timestamp"] = pd.to_datetime(local["timestamp"], errors="coerce", utc=True)
    if "year" not in local.columns:
        local["year"] = pd.NA
    local["year"] = pd.to_numeric(local["year"], errors="coerce")
    missing_year = local["year"].isna() & local["timestamp"].notna()
    if bool(missing_year.any()):
        local.loc[missing_year, "year"] = local.loc[missing_year, "timestamp"].dt.tz_convert("America/New_York").dt.year
    for col in [
        "session",
        "ctx_session_substate",
        "ctx_volatility_regime",
        "side_pattern",
        "chosen_side",
        "best_action",
        "chosen_sub_strategy",
        "long_sub_strategy",
        "short_sub_strategy",
        "long_timeframe",
        "short_timeframe",
        "long_strategy_type",
        "short_strategy_type",
        "long_strategy_style",
        "short_strategy_style",
    ]:
        if col not in local.columns:
            local[col] = ""
        local[col] = local[col].apply(_normalize_text)
    if "ctx_hour_et" not in local.columns:
        local["ctx_hour_et"] = pd.NA
    local["ctx_hour_et"] = pd.to_numeric(local.get("ctx_hour_et"), errors="coerce")
    missing_hour = local["ctx_hour_et"].isna() & local["timestamp"].notna()
    if bool(missing_hour.any()):
        local.loc[missing_hour, "ctx_hour_et"] = (
            local.loc[missing_hour, "timestamp"].dt.tz_convert("America/New_York").dt.hour.astype(float)
        )
    missing_substate = local["ctx_session_substate"].astype(str).str.strip() == ""
    if bool(missing_substate.any()):
        local.loc[missing_substate, "ctx_session_substate"] = [
            _derive_session_substate(session_text=row["session"], timestamp_value=row["timestamp"])
            for _, row in local.loc[missing_substate, ["session", "timestamp"]].iterrows()
        ]
    local["ctx_hour_bucket"] = local["ctx_hour_et"].apply(
        lambda raw: str(int(round(float(raw)))) if math.isfinite(_safe_float(raw, float("nan"))) else ""
    )
    local["ctx_price_location"] = pd.to_numeric(local.get("ctx_price_location"), errors="coerce")
    local["ctx_price_loc_bucket"] = local["ctx_price_location"].apply(
        lambda raw: _bucket_balance(raw, strong_neg=-1.25, neg=-0.35, pos=0.35, strong_pos=1.25)
    )
    numeric_cols = [
        "de3_entry_close_pos1",
        "de3_entry_upper_wick_ratio",
        "de3_entry_lower_wick_ratio",
        "de3_entry_body1_ratio",
        "de3_entry_ret1_atr",
        "de3_entry_vol1_rel20",
        "de3_entry_range10_atr",
        "de3_entry_dist_low5_atr",
        "de3_entry_dist_high5_atr",
        "de3_entry_down3",
        "long_rank",
        "short_rank",
        "long_final_score",
        "short_final_score",
        "long_edge_points",
        "short_edge_points",
        "long_structural_score",
        "short_structural_score",
        "long_pnl_points",
        "short_pnl_points",
        "baseline_points",
        "year",
    ]
    for col in numeric_cols:
        local[col] = pd.to_numeric(local.get(col), errors="coerce")
    wick_bias = local["de3_entry_lower_wick_ratio"].fillna(0.0) - local["de3_entry_upper_wick_ratio"].fillna(0.0)
    location_bias = local["de3_entry_dist_high5_atr"].fillna(0.0) - local["de3_entry_dist_low5_atr"].fillna(0.0)
    local["st_close_bucket"] = local["de3_entry_close_pos1"].apply(_bucket_close_pos)
    local["st_wick_bias_bucket"] = wick_bias.apply(
        lambda raw: _bucket_balance(raw, strong_neg=-0.28, neg=-0.10, pos=0.10, strong_pos=0.28)
    )
    local["st_body_bucket"] = local["de3_entry_body1_ratio"].apply(_bucket_body_ratio)
    local["st_ret_bucket"] = local["de3_entry_ret1_atr"].apply(
        lambda raw: _bucket_balance(raw, strong_neg=-0.45, neg=-0.12, pos=0.12, strong_pos=0.45)
    )
    local["st_vol_bucket"] = local["de3_entry_vol1_rel20"].apply(_bucket_vol_ratio)
    local["st_range_bucket"] = local["de3_entry_range10_atr"].apply(_bucket_range_ratio)
    local["st_location_bucket"] = location_bias.apply(
        lambda raw: _bucket_balance(raw, strong_neg=-1.20, neg=-0.35, pos=0.35, strong_pos=1.20)
    )
    local["st_down3_bucket"] = local["de3_entry_down3"].apply(_bucket_down3)
    local["st_pressure_bucket"] = (
        local["st_ret_bucket"].astype(str).str.strip()
        + "|"
        + local["st_wick_bias_bucket"].astype(str).str.strip()
    )
    local["long_strategy_style"] = local["long_strategy_type"].apply(_directionless_strategy_style)
    local["short_strategy_style"] = local["short_strategy_type"].apply(_directionless_strategy_style)
    def _adv_bucket(value: Any, neg_hi: float, neg_lo: float, pos_lo: float, pos_hi: float) -> str:
        raw = _safe_float(value, float("nan"))
        if not math.isfinite(raw):
            return ""
        if raw <= neg_hi:
            return "strong_short"
        if raw <= neg_lo:
            return "short"
        if raw < pos_lo:
            return "flat"
        if raw < pos_hi:
            return "long"
        return "strong_long"
    local["rank_adv"] = local["short_rank"] - local["long_rank"]
    local["score_adv"] = local["long_final_score"] - local["short_final_score"]
    local["edge_adv"] = local["long_edge_points"] - local["short_edge_points"]
    local["struct_adv"] = local["long_structural_score"] - local["short_structural_score"]
    local["rank_adv_bucket"] = local["rank_adv"].apply(lambda raw: _adv_bucket(raw, -1.5, -0.5, 0.5, 1.5))
    local["score_adv_bucket"] = local["score_adv"].apply(lambda raw: _adv_bucket(raw, -1.0, -0.25, 0.25, 1.0))
    local["edge_adv_bucket"] = local["edge_adv"].apply(lambda raw: _adv_bucket(raw, -1.0, -0.25, 0.25, 1.0))
    local["struct_adv_bucket"] = local["struct_adv"].apply(lambda raw: _adv_bucket(raw, -1.0, -0.25, 0.25, 1.0))
    local["score_edge_combo"] = local["score_adv_bucket"].astype(str) + "|" + local["edge_adv_bucket"].astype(str)
    local["rank_score_combo"] = local["rank_adv_bucket"].astype(str) + "|" + local["score_adv_bucket"].astype(str)
    return local


def _default_profiles() -> Dict[str, Dict[str, Any]]:
    base_scopes = [
        {
            "name": "side_pattern_shape",
            "fields": ["side_pattern", "st_close_bucket", "st_wick_bias_bucket", "st_pressure_bucket"],
            "min_decisions": 80,
            "weight": 1.00,
        },
        {
            "name": "session_substate_shape",
            "fields": ["session", "ctx_session_substate", "side_pattern", "st_ret_bucket", "st_location_bucket"],
            "min_decisions": 72,
            "weight": 0.96,
        },
        {
            "name": "session_hour_styles",
            "fields": ["session", "ctx_hour_bucket", "side_pattern", "long_strategy_style", "short_strategy_style"],
            "min_decisions": 60,
            "weight": 0.90,
        },
        {
            "name": "timeframe_pressure_combo",
            "fields": ["side_pattern", "long_timeframe", "short_timeframe", "st_pressure_bucket", "st_down3_bucket"],
            "min_decisions": 54,
            "weight": 0.84,
        },
        {
            "name": "hour_advantage_combo",
            "fields": ["ctx_hour_bucket", "side_pattern", "score_edge_combo", "st_location_bucket"],
            "min_decisions": 48,
            "weight": 0.76,
        },
        {
            "name": "session_volatility_flow",
            "fields": ["session", "ctx_volatility_regime", "side_pattern", "st_vol_bucket", "st_range_bucket"],
            "min_decisions": 48,
            "weight": 0.72,
        },
    ]
    guarded_scopes = copy.deepcopy(base_scopes) + [
        {
            "name": "session_price_loc_guard",
            "fields": ["session", "side_pattern", "ctx_price_loc_bucket", "st_location_bucket"],
            "min_decisions": 44,
            "weight": 0.70,
        },
    ]
    opportunistic_scopes = copy.deepcopy(base_scopes) + [
        {
            "name": "hour_rank_score_combo",
            "fields": ["ctx_hour_bucket", "side_pattern", "rank_score_combo", "st_close_bucket"],
            "min_decisions": 42,
            "weight": 0.70,
        },
    ]
    both_relative_scopes = [
        {
            "name": "both_session_substate_shape",
            "fields": ["session", "ctx_session_substate", "st_ret_bucket", "st_location_bucket", "st_pressure_bucket"],
            "min_decisions": 28,
            "weight": 1.00,
        },
        {
            "name": "both_session_hour_styles",
            "fields": ["session", "ctx_hour_bucket", "long_strategy_style", "short_strategy_style"],
            "min_decisions": 24,
            "weight": 0.95,
        },
        {
            "name": "both_timeframe_pressure",
            "fields": ["long_timeframe", "short_timeframe", "st_pressure_bucket", "st_down3_bucket"],
            "min_decisions": 24,
            "weight": 0.88,
        },
        {
            "name": "both_hour_advantage",
            "fields": ["ctx_hour_bucket", "score_edge_combo", "st_location_bucket"],
            "min_decisions": 24,
            "weight": 0.82,
        },
        {
            "name": "both_hour_rank_score",
            "fields": ["ctx_hour_bucket", "rank_score_combo", "st_close_bucket"],
            "min_decisions": 22,
            "weight": 0.78,
        },
        {
            "name": "both_session_volatility_flow",
            "fields": ["session", "ctx_volatility_regime", "st_vol_bucket", "st_range_bucket"],
            "min_decisions": 22,
            "weight": 0.72,
        },
    ]
    both_baseline_scopes = [
        {
            "name": "both_baseline_session_shape",
            "fields": ["chosen_side", "session", "ctx_session_substate", "st_pressure_bucket", "st_location_bucket"],
            "min_decisions": 22,
            "weight": 1.00,
        },
        {
            "name": "both_baseline_hour_styles",
            "fields": ["chosen_side", "ctx_hour_bucket", "long_strategy_style", "short_strategy_style"],
            "min_decisions": 20,
            "weight": 0.92,
        },
        {
            "name": "both_baseline_timeframe_pressure",
            "fields": ["chosen_side", "long_timeframe", "short_timeframe", "st_pressure_bucket", "st_down3_bucket"],
            "min_decisions": 20,
            "weight": 0.86,
        },
        {
            "name": "both_baseline_advantage_combo",
            "fields": ["chosen_side", "ctx_hour_bucket", "rank_score_combo", "score_edge_combo"],
            "min_decisions": 18,
            "weight": 0.82,
        },
        {
            "name": "both_baseline_volatility_flow",
            "fields": ["chosen_side", "session", "ctx_volatility_regime", "st_vol_bucket", "st_range_bucket"],
            "min_decisions": 18,
            "weight": 0.74,
        },
    ]
    morning_shape_scopes = [
        {
            "name": "morning_close_location",
            "fields": ["chosen_side", "long_timeframe", "short_timeframe", "st_close_bucket", "st_location_bucket"],
            "min_decisions": 16,
            "weight": 1.00,
        },
        {
            "name": "morning_close_vol_down3",
            "fields": ["chosen_side", "st_close_bucket", "st_vol_bucket", "st_down3_bucket"],
            "min_decisions": 16,
            "weight": 0.94,
        },
        {
            "name": "morning_advantage_shape",
            "fields": ["chosen_side", "ctx_hour_bucket", "score_edge_combo", "st_close_bucket"],
            "min_decisions": 14,
            "weight": 0.86,
        },
        {
            "name": "morning_wick_location",
            "fields": ["chosen_side", "st_wick_bias_bucket", "st_location_bucket", "st_down3_bucket"],
            "min_decisions": 14,
            "weight": 0.76,
        },
    ]
    morning_relative_scopes = [
        {
            "name": "morning_rel_close_location",
            "fields": ["long_timeframe", "short_timeframe", "st_close_bucket", "st_location_bucket"],
            "min_decisions": 16,
            "weight": 1.00,
        },
        {
            "name": "morning_rel_close_vol_down3",
            "fields": ["st_close_bucket", "st_vol_bucket", "st_down3_bucket"],
            "min_decisions": 16,
            "weight": 0.92,
        },
        {
            "name": "morning_rel_advantage_shape",
            "fields": ["ctx_hour_bucket", "score_edge_combo", "st_close_bucket"],
            "min_decisions": 14,
            "weight": 0.84,
        },
        {
            "name": "morning_rel_wick_location",
            "fields": ["st_wick_bias_bucket", "st_location_bucket", "st_down3_bucket"],
            "min_decisions": 14,
            "weight": 0.74,
        },
    ]
    both_exact_pair_scopes = [
        {
            "name": "both_exact_session_hour_pair_shape",
            "fields": ["session", "ctx_hour_bucket", "long_sub_strategy", "short_sub_strategy", "st_pressure_bucket"],
            "min_decisions": 10,
            "weight": 1.00,
        },
        {
            "name": "both_exact_session_substate_pair_location",
            "fields": ["session", "ctx_session_substate", "long_sub_strategy", "short_sub_strategy", "st_location_bucket"],
            "min_decisions": 10,
            "weight": 0.92,
        },
        {
            "name": "both_exact_hour_pair_advantage",
            "fields": ["ctx_hour_bucket", "long_sub_strategy", "short_sub_strategy", "score_edge_combo"],
            "min_decisions": 8,
            "weight": 0.86,
        },
        {
            "name": "both_exact_pair_close_vol",
            "fields": ["long_sub_strategy", "short_sub_strategy", "st_close_bucket", "st_vol_bucket"],
            "min_decisions": 10,
            "weight": 0.78,
        },
    ]
    both_compare_exact_scopes = [
        {
            "name": "both_compare_exact_session_hour_pair",
            "fields": ["chosen_side", "session", "ctx_hour_bucket", "long_sub_strategy", "short_sub_strategy"],
            "min_decisions": 12,
            "weight": 1.00,
        },
        {
            "name": "both_compare_exact_substate_pair_shape",
            "fields": ["chosen_side", "session", "ctx_session_substate", "long_sub_strategy", "short_sub_strategy", "st_pressure_bucket"],
            "min_decisions": 12,
            "weight": 0.92,
        },
        {
            "name": "both_compare_exact_pair_location",
            "fields": ["chosen_side", "long_sub_strategy", "short_sub_strategy", "st_location_bucket", "st_close_bucket"],
            "min_decisions": 12,
            "weight": 0.84,
        },
        {
            "name": "both_compare_exact_pair_advantage",
            "fields": ["chosen_side", "long_sub_strategy", "short_sub_strategy", "score_edge_combo"],
            "min_decisions": 10,
            "weight": 0.78,
        },
    ]
    long_only_exact_scopes = [
        {
            "name": "long_sub_session_hour_shape",
            "fields": ["session", "ctx_hour_bucket", "long_sub_strategy", "st_pressure_bucket", "st_location_bucket"],
            "min_decisions": 28,
            "weight": 1.00,
        },
        {
            "name": "long_sub_session_hour_closevol",
            "fields": ["session", "ctx_hour_bucket", "long_sub_strategy", "st_close_bucket", "st_vol_bucket"],
            "min_decisions": 26,
            "weight": 0.94,
        },
        {
            "name": "long_sub_substate_wick_down3",
            "fields": ["session", "ctx_session_substate", "long_sub_strategy", "st_wick_bias_bucket", "st_down3_bucket"],
            "min_decisions": 24,
            "weight": 0.88,
        },
        {
            "name": "long_sub_hour_advantage",
            "fields": ["ctx_hour_bucket", "long_sub_strategy", "score_edge_combo", "st_location_bucket"],
            "min_decisions": 22,
            "weight": 0.80,
        },
        {
            "name": "long_sub_shape_fallback",
            "fields": ["long_sub_strategy", "st_pressure_bucket", "st_location_bucket", "st_close_bucket"],
            "min_decisions": 36,
            "weight": 0.72,
        },
    ]
    short_only_exact_scopes = [
        {
            "name": "short_sub_session_hour_shape",
            "fields": ["session", "ctx_hour_bucket", "short_sub_strategy", "st_pressure_bucket", "st_location_bucket"],
            "min_decisions": 14,
            "weight": 1.00,
        },
        {
            "name": "short_sub_session_hour_closevol",
            "fields": ["session", "ctx_hour_bucket", "short_sub_strategy", "st_close_bucket", "st_vol_bucket"],
            "min_decisions": 12,
            "weight": 0.94,
        },
        {
            "name": "short_sub_substate_wick_down3",
            "fields": ["session", "ctx_session_substate", "short_sub_strategy", "st_wick_bias_bucket", "st_down3_bucket"],
            "min_decisions": 10,
            "weight": 0.88,
        },
        {
            "name": "short_sub_hour_advantage",
            "fields": ["ctx_hour_bucket", "short_sub_strategy", "score_edge_combo", "st_location_bucket"],
            "min_decisions": 10,
            "weight": 0.80,
        },
        {
            "name": "short_sub_shape_fallback",
            "fields": ["short_sub_strategy", "st_pressure_bucket", "st_location_bucket", "st_close_bucket"],
            "min_decisions": 14,
            "weight": 0.72,
        },
    ]
    hybrid_sidepattern_scopes = [
        {
            "name": "hybrid_sidepattern_session_shape",
            "fields": ["side_pattern", "chosen_side", "session", "ctx_session_substate", "st_pressure_bucket", "st_location_bucket"],
            "min_decisions": 20,
            "weight": 1.00,
        },
        {
            "name": "hybrid_sidepattern_hour_styles",
            "fields": ["side_pattern", "chosen_side", "ctx_hour_bucket", "long_strategy_style", "short_strategy_style"],
            "min_decisions": 18,
            "weight": 0.94,
        },
        {
            "name": "hybrid_sidepattern_timeframe_pressure",
            "fields": ["side_pattern", "chosen_side", "long_timeframe", "short_timeframe", "st_pressure_bucket", "st_down3_bucket"],
            "min_decisions": 18,
            "weight": 0.88,
        },
        {
            "name": "hybrid_sidepattern_advantage_combo",
            "fields": ["side_pattern", "chosen_side", "ctx_hour_bucket", "rank_score_combo", "score_edge_combo"],
            "min_decisions": 18,
            "weight": 0.84,
        },
        {
            "name": "hybrid_sidepattern_volatility_flow",
            "fields": ["side_pattern", "chosen_side", "session", "ctx_volatility_regime", "st_vol_bucket", "st_range_bucket"],
            "min_decisions": 18,
            "weight": 0.76,
        },
        {
            "name": "hybrid_sidepattern_price_loc_guard",
            "fields": ["side_pattern", "chosen_side", "session", "ctx_price_loc_bucket", "st_location_bucket"],
            "min_decisions": 18,
            "weight": 0.72,
        },
    ]
    compare_baseline_scopes = [
        {
            "name": "compare_baseline_sub_shape",
            "fields": ["side_pattern", "chosen_side", "chosen_sub_strategy", "st_pressure_bucket", "st_location_bucket"],
            "min_decisions": 20,
            "weight": 1.00,
        },
        {
            "name": "compare_baseline_session_hour_sub",
            "fields": ["session", "ctx_hour_bucket", "chosen_sub_strategy", "st_close_bucket", "st_vol_bucket"],
            "min_decisions": 18,
            "weight": 0.94,
        },
        {
            "name": "compare_baseline_pair_advantage",
            "fields": ["side_pattern", "chosen_side", "score_edge_combo", "rank_score_combo", "st_location_bucket"],
            "min_decisions": 18,
            "weight": 0.88,
        },
        {
            "name": "compare_baseline_styles_shape",
            "fields": ["side_pattern", "chosen_side", "long_strategy_style", "short_strategy_style", "st_pressure_bucket"],
            "min_decisions": 18,
            "weight": 0.82,
        },
        {
            "name": "compare_baseline_timeframe_flow",
            "fields": ["side_pattern", "chosen_side", "long_timeframe", "short_timeframe", "st_down3_bucket", "st_range_bucket"],
            "min_decisions": 16,
            "weight": 0.76,
        },
        {
            "name": "compare_baseline_price_loc_guard",
            "fields": ["side_pattern", "chosen_side", "session", "ctx_price_loc_bucket", "st_location_bucket"],
            "min_decisions": 16,
            "weight": 0.72,
        },
    ]
    return {
        "decision_side_local_balanced": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.12,
            "prior_component_scale": 0.60,
            "prior_formula": "relative_minus_no_trade",
            "scopes": base_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 300.0,
            "year_coverage_full_years": 10.0,
            "mean_scale_points": 2.75,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.16,
            "best_rate_center": 0.33,
            "best_rate_scale": 0.18,
            "no_trade_rate_center": 0.38,
            "no_trade_rate_scale": 0.18,
            "trade_threshold_candidates": [0.04, 0.08, 0.12, 0.16],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.08, 0.12, 0.16],
            "no_trade_margin_candidates": [0.00, 0.04, 0.08],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 90,
            "min_trade_score_to_override": 0.0,
        },
        "decision_side_local_guarded": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.10,
            "prior_component_scale": 0.65,
            "prior_formula": "relative_minus_no_trade",
            "scopes": guarded_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 320.0,
            "year_coverage_full_years": 10.0,
            "mean_scale_points": 2.90,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.15,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.40,
            "no_trade_rate_scale": 0.16,
            "trade_threshold_candidates": [0.08, 0.12, 0.16, 0.20],
            "side_margin_candidates": [0.04, 0.08, 0.12],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.15,
            "min_override_count": 70,
            "min_trade_score_to_override": 0.02,
        },
        "decision_side_local_guarded_hard": {
            "application_mode": "hard_override",
            "prior_component_weight": 0.0,
            "prior_component_scale": 0.65,
            "prior_formula": "relative_minus_no_trade",
            "scopes": guarded_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 320.0,
            "year_coverage_full_years": 10.0,
            "mean_scale_points": 2.90,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.15,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.40,
            "no_trade_rate_scale": 0.16,
            "trade_threshold_candidates": [0.12, 0.16, 0.20, 0.24],
            "side_margin_candidates": [0.04, 0.08, 0.12],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.15,
            "min_override_count": 120,
            "min_trade_score_to_override": 0.04,
        },
        "decision_side_local_opportunistic": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.16,
            "prior_component_scale": 0.55,
            "prior_formula": "relative_minus_no_trade",
            "scopes": opportunistic_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 260.0,
            "year_coverage_full_years": 10.0,
            "mean_scale_points": 2.55,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.18,
            "best_rate_center": 0.33,
            "best_rate_scale": 0.20,
            "no_trade_rate_center": 0.36,
            "no_trade_rate_scale": 0.20,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.06, 0.10, 0.14],
            "no_trade_margin_candidates": [0.00, 0.03, 0.06],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 120,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_local_balanced_hard": {
            "application_mode": "hard_override",
            "prior_component_weight": 0.0,
            "prior_component_scale": 0.60,
            "prior_formula": "relative_minus_no_trade",
            "scopes": base_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 300.0,
            "year_coverage_full_years": 10.0,
            "mean_scale_points": 2.75,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.16,
            "best_rate_center": 0.33,
            "best_rate_scale": 0.18,
            "no_trade_rate_center": 0.38,
            "no_trade_rate_scale": 0.18,
            "trade_threshold_candidates": [0.12, 0.16, 0.20, 0.24],
            "side_margin_candidates": [0.04, 0.08, 0.12],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 140,
            "min_trade_score_to_override": 0.04,
        },
        "decision_side_long_only_guarded_hard": {
            "application_mode": "hard_override",
            "prior_component_weight": 0.0,
            "prior_component_scale": 0.65,
            "prior_formula": "relative_minus_no_trade",
            "apply_side_patterns": ["long_only"],
            "scopes": guarded_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 320.0,
            "year_coverage_full_years": 10.0,
            "mean_scale_points": 2.90,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.15,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.40,
            "no_trade_rate_scale": 0.16,
            "trade_threshold_candidates": [0.12, 0.16, 0.20, 0.24],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.15,
            "min_override_count": 120,
            "min_trade_score_to_override": 0.04,
        },
        "decision_side_long_only_guarded_hard_ex_18_21": {
            "application_mode": "hard_override",
            "prior_component_weight": 0.0,
            "prior_component_scale": 0.65,
            "prior_formula": "relative_minus_no_trade",
            "apply_side_patterns": ["long_only"],
            "exclude_sessions": ["18-21"],
            "scopes": guarded_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 320.0,
            "year_coverage_full_years": 10.0,
            "mean_scale_points": 2.90,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.15,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.40,
            "no_trade_rate_scale": 0.16,
            "trade_threshold_candidates": [0.12, 0.16, 0.20, 0.24],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.15,
            "min_override_count": 120,
            "min_trade_score_to_override": 0.04,
        },
        "decision_side_long_only_guarded_soft": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.08,
            "prior_component_scale": 0.65,
            "prior_formula": "relative_minus_no_trade",
            "apply_side_patterns": ["long_only"],
            "scopes": guarded_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 320.0,
            "year_coverage_full_years": 10.0,
            "mean_scale_points": 2.90,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.15,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.40,
            "no_trade_rate_scale": 0.16,
            "trade_threshold_candidates": [0.08, 0.12, 0.16, 0.20],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 80,
            "min_trade_score_to_override": 0.0,
        },
        "decision_side_long_only_guarded_soft_ex_18_21": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.08,
            "prior_component_scale": 0.65,
            "prior_formula": "relative_minus_no_trade",
            "apply_side_patterns": ["long_only"],
            "exclude_sessions": ["18-21"],
            "scopes": guarded_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 320.0,
            "year_coverage_full_years": 10.0,
            "mean_scale_points": 2.90,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.15,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.40,
            "no_trade_rate_scale": 0.16,
            "trade_threshold_candidates": [0.08, 0.12, 0.16, 0.20],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 70,
            "min_trade_score_to_override": 0.0,
        },
        "decision_side_short_only_guarded_hard": {
            "application_mode": "hard_override",
            "prior_component_weight": 0.0,
            "prior_component_scale": 0.62,
            "prior_formula": "relative_minus_no_trade",
            "apply_side_patterns": ["short_only"],
            "scopes": guarded_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 120.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.45,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.35,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.38,
            "no_trade_rate_scale": 0.18,
            "trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 20,
            "min_trade_score_to_override": 0.04,
        },
        "decision_side_short_only_exact_soft": {
            "application_mode": "soft_prior",
            "apply_prior_only_when_predicted": True,
            "prior_component_weight": 0.08,
            "prior_component_scale": 0.56,
            "prior_formula": "side_minus_no_trade",
            "apply_side_patterns": ["short_only"],
            "scopes": short_only_exact_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 84.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.25,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.36,
            "no_trade_rate_scale": 0.18,
            "trade_threshold_candidates": [0.08, 0.12, 0.16, 0.20],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 16,
            "min_trade_score_to_override": 0.02,
        },
        "decision_side_long_only_exact_soft_ex_18_21": {
            "application_mode": "soft_prior",
            "apply_prior_only_when_predicted": True,
            "prior_component_weight": 0.08,
            "prior_component_scale": 0.60,
            "prior_formula": "side_minus_no_trade",
            "apply_side_patterns": ["long_only"],
            "exclude_sessions": ["18-21"],
            "scopes": long_only_exact_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 180.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.55,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.15,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.38,
            "no_trade_rate_scale": 0.18,
            "trade_threshold_candidates": [0.08, 0.12, 0.16, 0.20],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 36,
            "min_trade_score_to_override": 0.02,
        },
        "decision_side_long_only_exact_hard_ex_18_21": {
            "application_mode": "hard_override",
            "prior_component_weight": 0.0,
            "prior_component_scale": 0.60,
            "prior_formula": "side_minus_no_trade",
            "apply_side_patterns": ["long_only"],
            "exclude_sessions": ["18-21"],
            "scopes": long_only_exact_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 180.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.55,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.15,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.38,
            "no_trade_rate_scale": 0.18,
            "trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 28,
            "min_trade_score_to_override": 0.04,
        },
        "decision_side_short_only_exact_hard": {
            "application_mode": "hard_override",
            "prior_component_weight": 0.0,
            "prior_component_scale": 0.56,
            "prior_formula": "side_minus_no_trade",
            "apply_side_patterns": ["short_only"],
            "scopes": short_only_exact_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 84.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.25,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.34,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.36,
            "no_trade_rate_scale": 0.18,
            "trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 16,
            "min_trade_score_to_override": 0.04,
        },
        "decision_side_both_longonly_hybrid": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.12,
            "prior_component_scale": 0.55,
            "prior_formula": "relative_minus_no_trade",
            "apply_side_patterns": ["both", "long_only"],
            "scopes": hybrid_sidepattern_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 140.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.34,
            "no_trade_rate_scale": 0.20,
            "trade_threshold_candidates": [0.04, 0.08, 0.12, 0.16],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.08, 0.12, 0.16],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 40,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_longonly_hybrid_ex_hour_12": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.12,
            "prior_component_scale": 0.55,
            "prior_formula": "relative_minus_no_trade",
            "apply_side_patterns": ["both", "long_only"],
            "exclude_hour_buckets": ["12"],
            "scopes": hybrid_sidepattern_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 140.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.34,
            "no_trade_rate_scale": 0.20,
            "trade_threshold_candidates": [0.04, 0.08, 0.12, 0.16],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.08, 0.12, 0.16],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 36,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_relative": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.18,
            "prior_component_scale": 0.55,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "scopes": both_relative_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 140.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.20,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.18,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.18,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 50,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_edge": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.20,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "relative_side_mode": True,
            "scopes": both_relative_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 140.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 50,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_baseline_edge": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.14,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "relative_side_mode": True,
            "scopes": both_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 110.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 40,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_baseline_edge_ex_midday": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.14,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_sessions": ["12-15"],
            "relative_side_mode": True,
            "scopes": both_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 110.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 24,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_baseline_edge_ex_hours_10_12": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.14,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_hour_buckets": ["10", "12"],
            "relative_side_mode": True,
            "scopes": both_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 110.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 24,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_baseline_edge_ex_hour_12": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.14,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_hour_buckets": ["12"],
            "relative_side_mode": True,
            "scopes": both_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 110.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 24,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_baseline_edge_ex_hour_10": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.14,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_hour_buckets": ["10"],
            "relative_side_mode": True,
            "scopes": both_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 110.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 24,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_exact_pair_ex_hour_12": {
            "application_mode": "soft_prior",
            "apply_prior_only_when_predicted": True,
            "prior_component_weight": 0.08,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_hour_buckets": ["12"],
            "relative_side_mode": True,
            "scopes": both_exact_pair_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 48.0,
            "year_coverage_full_years": 7.0,
            "mean_scale_points": 1.85,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.13,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.14,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.04, 0.08, 0.12, 0.16],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 16,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_exact_pair_consistent_ex_hour_12": {
            "application_mode": "soft_prior",
            "apply_prior_only_when_predicted": True,
            "prior_component_weight": 0.06,
            "prior_component_scale": 0.48,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_hour_buckets": ["12"],
            "relative_side_mode": True,
            "scopes": both_exact_pair_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 64.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 1.75,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.12,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.13,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.20,
            "year_consistency_min_years": 4,
            "year_consistency_min_rows_per_year": 4,
            "year_consistency_floor": 0.50,
            "year_consistency_power": 1.35,
            "trade_threshold_candidates": [0.08, 0.12, 0.16, 0.20],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 12,
            "min_trade_score_to_override": 0.02,
        },
        "decision_side_both_exact_pair_consistent_hard_ex_hour_12": {
            "application_mode": "hard_override",
            "prior_component_weight": 0.0,
            "prior_component_scale": 0.48,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_hour_buckets": ["12"],
            "relative_side_mode": True,
            "scopes": both_exact_pair_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 64.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 1.75,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.12,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.13,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.20,
            "year_consistency_min_years": 4,
            "year_consistency_min_rows_per_year": 4,
            "year_consistency_floor": 0.50,
            "year_consistency_power": 1.35,
            "trade_threshold_candidates": [0.12, 0.16, 0.20, 0.24],
            "side_margin_candidates": [0.00, 0.04, 0.08],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18, 0.22],
            "no_trade_margin_candidates": [0.04, 0.08, 0.12],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 12,
            "min_trade_score_to_override": 0.04,
        },
        "decision_side_both_exact_pair": {
            "application_mode": "soft_prior",
            "apply_prior_only_when_predicted": True,
            "prior_component_weight": 0.08,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "relative_side_mode": True,
            "scopes": both_exact_pair_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 48.0,
            "year_coverage_full_years": 7.0,
            "mean_scale_points": 1.85,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.13,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.14,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.04, 0.08, 0.12, 0.16],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 16,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_compare_exact_ex_hour_12": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.12,
            "prior_component_scale": 0.46,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_hour_buckets": ["12"],
            "relative_side_mode": True,
            "scopes": both_compare_exact_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 72.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 1.85,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.12,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.13,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.20,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 18,
            "min_trade_score_to_override": 0.0,
        },
        "decision_side_both_compare_exact_consistent_ex_hour_12": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.10,
            "prior_component_scale": 0.46,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_hour_buckets": ["12"],
            "relative_side_mode": True,
            "scopes": both_compare_exact_scopes,
            "max_scopes_per_row": 2,
            "support_full_decisions": 72.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 1.85,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.12,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.13,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.20,
            "year_consistency_min_years": 4,
            "year_consistency_min_rows_per_year": 4,
            "year_consistency_floor": 0.45,
            "year_consistency_power": 1.20,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.09,
            "min_override_count": 14,
            "min_trade_score_to_override": 0.0,
        },
        "decision_side_both_baseline_edge_ex_hours_8_10_12": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.14,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "exclude_hour_buckets": ["8", "10", "12"],
            "relative_side_mode": True,
            "scopes": both_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 110.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 20,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_baseline_edge_0912_only": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.14,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "apply_sessions": ["09-12"],
            "relative_side_mode": True,
            "scopes": both_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 110.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 12,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_baseline_edge_0912_consistent": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.10,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "apply_sessions": ["09-12"],
            "relative_side_mode": True,
            "scopes": both_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 110.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.15,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "year_consistency_min_years": 4,
            "year_consistency_min_rows_per_year": 4,
            "year_consistency_floor": 0.35,
            "year_consistency_power": 1.0,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 10,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_morning_shape": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.10,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "apply_sessions": ["09-12"],
            "relative_side_mode": True,
            "scopes": morning_shape_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 80.0,
            "year_coverage_full_years": 7.0,
            "mean_scale_points": 1.90,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.13,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.14,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 10,
            "min_trade_score_to_override": -0.02,
        },
        "decision_side_both_morning_relative": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.10,
            "prior_component_scale": 0.50,
            "prior_formula": "relative_side_only",
            "apply_side_patterns": ["both"],
            "apply_sessions": ["09-12"],
            "relative_side_mode": True,
            "scopes": morning_relative_scopes,
            "max_scopes_per_row": 3,
            "support_full_decisions": 80.0,
            "year_coverage_full_years": 7.0,
            "mean_scale_points": 1.90,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.13,
            "best_rate_center": 0.40,
            "best_rate_scale": 0.14,
            "no_trade_rate_center": 0.30,
            "no_trade_rate_scale": 0.22,
            "trade_threshold_candidates": [0.02, 0.06, 0.10, 0.14],
            "side_margin_candidates": [0.00, 0.03, 0.06],
            "no_trade_threshold_candidates": [0.10, 0.14, 0.18],
            "no_trade_margin_candidates": [0.02, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.08,
            "min_override_count": 10,
            "min_trade_score_to_override": -0.02,
        },
        "decision_action_compare_baseline_soft": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.08,
            "prior_component_scale": 0.55,
            "prior_formula": "side_minus_no_trade",
            "compare_to_baseline_mode": True,
            "scopes": compare_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 180.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "uplift_scale_points": 1.80,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "uplift_positive_center": 0.46,
            "uplift_positive_scale": 0.16,
            "best_rate_center": 0.38,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.36,
            "no_trade_rate_scale": 0.20,
            "trade_threshold_candidates": [0.04, 0.08, 0.12, 0.16, 0.22],
            "side_margin_candidates": [0.00, 0.03, 0.06, 0.10],
            "no_trade_threshold_candidates": [0.04, 0.08, 0.12, 0.16, 0.22],
            "no_trade_margin_candidates": [0.00, 0.03, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 30,
            "min_trade_score_to_override": -0.02,
        },
        "decision_action_compare_baseline_hard": {
            "application_mode": "hard_override",
            "prior_component_weight": 0.0,
            "prior_component_scale": 0.55,
            "prior_formula": "side_minus_no_trade",
            "compare_to_baseline_mode": True,
            "scopes": compare_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 180.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "uplift_scale_points": 1.80,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "uplift_positive_center": 0.46,
            "uplift_positive_scale": 0.16,
            "best_rate_center": 0.38,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.36,
            "no_trade_rate_scale": 0.20,
            "trade_threshold_candidates": [0.08, 0.12, 0.16, 0.22, 0.30],
            "side_margin_candidates": [0.00, 0.03, 0.06, 0.10],
            "no_trade_threshold_candidates": [0.06, 0.10, 0.14, 0.18, 0.24],
            "no_trade_margin_candidates": [0.00, 0.03, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 50,
            "min_trade_score_to_override": 0.0,
        },
        "decision_action_compare_baseline_soft_ex_hour_12": {
            "application_mode": "soft_prior",
            "prior_component_weight": 0.08,
            "prior_component_scale": 0.55,
            "prior_formula": "side_minus_no_trade",
            "compare_to_baseline_mode": True,
            "exclude_hour_buckets": ["12"],
            "scopes": compare_baseline_scopes,
            "max_scopes_per_row": 4,
            "support_full_decisions": 180.0,
            "year_coverage_full_years": 8.0,
            "mean_scale_points": 2.10,
            "uplift_scale_points": 1.80,
            "positive_rate_center": 0.50,
            "positive_rate_scale": 0.14,
            "uplift_positive_center": 0.46,
            "uplift_positive_scale": 0.16,
            "best_rate_center": 0.38,
            "best_rate_scale": 0.16,
            "no_trade_rate_center": 0.36,
            "no_trade_rate_scale": 0.20,
            "trade_threshold_candidates": [0.04, 0.08, 0.12, 0.16, 0.22],
            "side_margin_candidates": [0.00, 0.03, 0.06, 0.10],
            "no_trade_threshold_candidates": [0.04, 0.08, 0.12, 0.16, 0.22],
            "no_trade_margin_candidates": [0.00, 0.03, 0.06, 0.10],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 26,
            "min_trade_score_to_override": -0.02,
        },
    }


def _normalized_text_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip().lower() for v in values if str(v).strip()]


def _row_model_applicable(row: Dict[str, Any], model: Dict[str, Any]) -> bool:
    if not isinstance(model, dict):
        return True
    row_side_pattern = str(row.get("side_pattern", "") or "").strip().lower()
    allowed_side_patterns = _normalized_text_list(model.get("apply_side_patterns", []))
    if allowed_side_patterns and row_side_pattern not in allowed_side_patterns:
        return False
    row_chosen_side = str(row.get("chosen_side", "") or "").strip().lower()
    allowed_chosen_sides = _normalized_text_list(model.get("apply_chosen_sides", []))
    if allowed_chosen_sides and row_chosen_side not in allowed_chosen_sides:
        return False
    row_session = str(row.get("session", "") or "").strip().lower()
    allowed_sessions = _normalized_text_list(model.get("apply_sessions", []))
    if allowed_sessions and row_session not in allowed_sessions:
        return False
    blocked_sessions = _normalized_text_list(model.get("exclude_sessions", []))
    if blocked_sessions and row_session in blocked_sessions:
        return False
    row_hour_bucket = str(row.get("ctx_hour_bucket", "") or "").strip().lower()
    allowed_hour_buckets = _normalized_text_list(model.get("apply_hour_buckets", []))
    if allowed_hour_buckets and row_hour_bucket not in allowed_hour_buckets:
        return False
    blocked_hour_buckets = _normalized_text_list(model.get("exclude_hour_buckets", []))
    if blocked_hour_buckets and row_hour_bucket in blocked_hour_buckets:
        return False
    row_close_bucket = str(row.get("st_close_bucket", "") or "").strip().lower()
    allowed_close_buckets = _normalized_text_list(model.get("apply_st_close_buckets", []))
    if allowed_close_buckets and row_close_bucket not in allowed_close_buckets:
        return False
    blocked_close_buckets = _normalized_text_list(model.get("exclude_st_close_buckets", []))
    if blocked_close_buckets and row_close_bucket in blocked_close_buckets:
        return False
    return True


def _apply_cfg_metadata_to_model(model: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    model["application_mode"] = str(cfg.get("application_mode", "soft_prior") or "soft_prior")
    model["prior_component_weight"] = float(_safe_float(cfg.get("prior_component_weight", 0.12), 0.12))
    model["prior_component_scale"] = float(_safe_float(cfg.get("prior_component_scale", 0.60), 0.60))
    model["prior_formula"] = str(cfg.get("prior_formula", "relative_minus_no_trade") or "relative_minus_no_trade")
    for cfg_key in (
        "apply_side_patterns",
        "apply_chosen_sides",
        "apply_sessions",
        "exclude_sessions",
        "apply_hour_buckets",
        "exclude_hour_buckets",
        "apply_st_close_buckets",
        "exclude_st_close_buckets",
    ):
        values = _normalized_text_list(cfg.get(cfg_key, []))
        if values:
            model[cfg_key] = list(values)


def _build_model(train_df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    support_full = max(1.0, _safe_float(cfg.get("support_full_decisions", 300.0), 300.0))
    year_full = max(1.0, _safe_float(cfg.get("year_coverage_full_years", 10.0), 10.0))
    mean_scale = max(0.10, _safe_float(cfg.get("mean_scale_points", 2.75), 2.75))
    positive_center = _safe_float(cfg.get("positive_rate_center", 0.50), 0.50)
    positive_scale = max(0.05, _safe_float(cfg.get("positive_rate_scale", 0.16), 0.16))
    best_center = _safe_float(cfg.get("best_rate_center", 0.33), 0.33)
    best_scale = max(0.05, _safe_float(cfg.get("best_rate_scale", 0.18), 0.18))
    no_trade_center = _safe_float(cfg.get("no_trade_rate_center", 0.38), 0.38)
    no_trade_scale = max(0.05, _safe_float(cfg.get("no_trade_rate_scale", 0.18), 0.18))
    uplift_scale = max(0.10, _safe_float(cfg.get("uplift_scale_points", mean_scale), mean_scale))
    uplift_positive_center = _safe_float(cfg.get("uplift_positive_center", positive_center), positive_center)
    uplift_positive_scale = max(0.05, _safe_float(cfg.get("uplift_positive_scale", positive_scale), positive_scale))
    max_scopes_per_row = max(1, _safe_int(cfg.get("max_scopes_per_row", 3), 3))
    max_abs_score = 1.5
    relative_side_mode = bool(cfg.get("relative_side_mode", False))
    compare_to_baseline_mode = bool(cfg.get("compare_to_baseline_mode", False))
    consistency_min_years = max(0, _safe_int(cfg.get("year_consistency_min_years", 0), 0))
    consistency_min_rows = max(1, _safe_int(cfg.get("year_consistency_min_rows_per_year", 4), 4))
    consistency_floor = _clip(_safe_float(cfg.get("year_consistency_floor", 0.0), 0.0), 0.0, 1.0)
    consistency_power = max(0.0, _safe_float(cfg.get("year_consistency_power", 0.0), 0.0))
    scopes_out: List[Dict[str, Any]] = []
    bucket_total = 0
    for idx, raw_scope in enumerate(cfg.get("scopes", [])):
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
        if not fields:
            continue
        min_decisions = max(10, _safe_int(scope.get("min_decisions", 60), 60))
        weight = max(0.0, _safe_float(scope.get("weight", 1.0), 1.0))
        if weight <= 0.0:
            continue
        scope_name = str(scope.get("name", f"scope_{idx}") or f"scope_{idx}")
        scope_rows = train_df[
            fields
            + [
                "year",
                "best_action",
                "baseline_points",
                "long_available",
                "short_available",
                "long_pnl_points",
                "short_pnl_points",
            ]
        ].copy()
        scope_rows["_bucket_key"] = [_bucket_key(row, fields) for row in scope_rows[fields].to_dict("records")]
        scope_rows = scope_rows[scope_rows["_bucket_key"].astype(str).str.strip() != ""].copy()
        if scope_rows.empty:
            continue
        buckets: Dict[str, Dict[str, Any]] = {}
        for bucket_key, grp in scope_rows.groupby("_bucket_key", sort=False):
            n_decisions = int(len(grp))
            if n_decisions < min_decisions:
                continue
            year_coverage = int(pd.to_numeric(grp["year"], errors="coerce").dropna().nunique())
            support_ratio = _clip(float(n_decisions) / float(support_full), 0.0, 1.0)
            year_ratio = _clip(float(year_coverage) / float(year_full), 0.0, 1.0)
            weight_mult = math.sqrt(max(0.0, support_ratio)) * year_ratio
            long_avail = grp[pd.to_numeric(grp["long_available"], errors="coerce").fillna(0).astype(int) > 0].copy()
            short_avail = grp[pd.to_numeric(grp["short_available"], errors="coerce").fillna(0).astype(int) > 0].copy()
            long_mean = _safe_float(pd.to_numeric(long_avail["long_pnl_points"], errors="coerce").mean(), 0.0)
            short_mean = _safe_float(pd.to_numeric(short_avail["short_pnl_points"], errors="coerce").mean(), 0.0)
            long_positive_rate = float(
                (pd.to_numeric(long_avail["long_pnl_points"], errors="coerce").fillna(0.0) > 0.0).mean()
            ) if not long_avail.empty else 0.0
            short_positive_rate = float(
                (pd.to_numeric(short_avail["short_pnl_points"], errors="coerce").fillna(0.0) > 0.0).mean()
            ) if not short_avail.empty else 0.0
            best_counts = Counter(str(v or "").strip().lower() for v in grp["best_action"].tolist())
            long_best_rate = float(best_counts.get("long", 0) / max(1, n_decisions))
            short_best_rate = float(best_counts.get("short", 0) / max(1, n_decisions))
            no_trade_rate = float(best_counts.get("no_trade", 0) / max(1, n_decisions))
            max_side_mean = max(float(long_mean), float(short_mean))
            max_side_positive = max(float(long_positive_rate), float(short_positive_rate))
            baseline_points = pd.to_numeric(grp["baseline_points"], errors="coerce").fillna(0.0)
            long_uplift = pd.to_numeric(long_avail["long_pnl_points"], errors="coerce") - pd.to_numeric(
                long_avail["baseline_points"], errors="coerce"
            ).fillna(0.0)
            short_uplift = pd.to_numeric(short_avail["short_pnl_points"], errors="coerce") - pd.to_numeric(
                short_avail["baseline_points"], errors="coerce"
            ).fillna(0.0)
            no_trade_uplift = 0.0 - baseline_points
            long_uplift_mean = _safe_float(long_uplift.mean(), 0.0)
            short_uplift_mean = _safe_float(short_uplift.mean(), 0.0)
            no_trade_uplift_mean = _safe_float(no_trade_uplift.mean(), 0.0)
            long_uplift_positive_rate = float((long_uplift.fillna(0.0) > 0.0).mean()) if not long_avail.empty else 0.0
            short_uplift_positive_rate = float((short_uplift.fillna(0.0) > 0.0).mean()) if not short_avail.empty else 0.0
            no_trade_uplift_positive_rate = float((no_trade_uplift.fillna(0.0) > 0.0).mean()) if n_decisions > 0 else 0.0
            if relative_side_mode:
                year_sign_consistency = 1.0
                if consistency_min_years > 0 and consistency_power > 0.0:
                    relative_year_signs: List[int] = []
                    for _, year_grp in grp.groupby("year", sort=False):
                        if int(len(year_grp)) < consistency_min_rows:
                            continue
                        year_long_mean = _safe_float(
                            pd.to_numeric(year_grp["long_pnl_points"], errors="coerce").mean(),
                            0.0,
                        )
                        year_short_mean = _safe_float(
                            pd.to_numeric(year_grp["short_pnl_points"], errors="coerce").mean(),
                            0.0,
                        )
                        year_rel_mean = float(year_long_mean - year_short_mean)
                        if math.isfinite(year_rel_mean) and abs(year_rel_mean) > 1e-9:
                            relative_year_signs.append(1 if year_rel_mean > 0.0 else -1)
                    if len(relative_year_signs) >= consistency_min_years:
                        year_sign_consistency = abs(sum(relative_year_signs)) / float(len(relative_year_signs))
                        weight_mult *= max(consistency_floor, float(year_sign_consistency)) ** consistency_power
                relative_side_score = (
                    0.65 * math.tanh((long_mean - short_mean) / mean_scale)
                    + 0.20 * math.tanh((long_positive_rate - short_positive_rate) / positive_scale)
                    + 0.15 * math.tanh((long_best_rate - short_best_rate) / best_scale)
                )
                relative_side_score = _clip(relative_side_score * weight_mult, -max_abs_score, max_abs_score)
                long_score = float(relative_side_score)
                short_score = float(-relative_side_score)
                no_trade_score = 0.0
            elif compare_to_baseline_mode:
                long_score = (
                    0.62 * math.tanh(long_uplift_mean / uplift_scale)
                    + 0.23 * math.tanh((long_uplift_positive_rate - uplift_positive_center) / uplift_positive_scale)
                    + 0.15 * math.tanh((long_best_rate - best_center) / best_scale)
                )
                short_score = (
                    0.62 * math.tanh(short_uplift_mean / uplift_scale)
                    + 0.23 * math.tanh((short_uplift_positive_rate - uplift_positive_center) / uplift_positive_scale)
                    + 0.15 * math.tanh((short_best_rate - best_center) / best_scale)
                )
                no_trade_score = (
                    0.62 * math.tanh(no_trade_uplift_mean / uplift_scale)
                    + 0.23 * math.tanh((no_trade_uplift_positive_rate - uplift_positive_center) / uplift_positive_scale)
                    + 0.15 * math.tanh((no_trade_rate - no_trade_center) / no_trade_scale)
                )
                long_score = _clip(long_score * weight_mult, -max_abs_score, max_abs_score)
                short_score = _clip(short_score * weight_mult, -max_abs_score, max_abs_score)
                no_trade_score = _clip(no_trade_score * weight_mult, -max_abs_score, max_abs_score)
            else:
                long_score = (
                    0.50 * math.tanh(long_mean / mean_scale)
                    + 0.25 * math.tanh((long_positive_rate - positive_center) / positive_scale)
                    + 0.25 * math.tanh((long_best_rate - best_center) / best_scale)
                )
                short_score = (
                    0.50 * math.tanh(short_mean / mean_scale)
                    + 0.25 * math.tanh((short_positive_rate - positive_center) / positive_scale)
                    + 0.25 * math.tanh((short_best_rate - best_center) / best_scale)
                )
                no_trade_score = (
                    0.55 * math.tanh((no_trade_rate - no_trade_center) / no_trade_scale)
                    + 0.25 * math.tanh((-max_side_mean) / mean_scale)
                    + 0.20 * math.tanh((positive_center - max_side_positive) / positive_scale)
                )
                long_score = _clip(long_score * weight_mult, -max_abs_score, max_abs_score)
                short_score = _clip(short_score * weight_mult, -max_abs_score, max_abs_score)
                no_trade_score = _clip(no_trade_score * weight_mult, -max_abs_score, max_abs_score)
            if max(abs(long_score), abs(short_score), abs(no_trade_score)) < 0.02:
                continue
            buckets[str(bucket_key)] = {
                "long_score": float(long_score),
                "short_score": float(short_score),
                "no_trade_score": float(no_trade_score),
                "n_decisions": int(n_decisions),
                "year_coverage": int(year_coverage),
                "long_mean_pnl_points": float(long_mean),
                "short_mean_pnl_points": float(short_mean),
                "long_positive_rate": float(long_positive_rate),
                "short_positive_rate": float(short_positive_rate),
                "long_best_rate": float(long_best_rate),
                "short_best_rate": float(short_best_rate),
                "no_trade_rate": float(no_trade_rate),
                "long_uplift_mean_points": float(long_uplift_mean),
                "short_uplift_mean_points": float(short_uplift_mean),
                "no_trade_uplift_mean_points": float(no_trade_uplift_mean),
                "long_uplift_positive_rate": float(long_uplift_positive_rate),
                "short_uplift_positive_rate": float(short_uplift_positive_rate),
                "no_trade_uplift_positive_rate": float(no_trade_uplift_positive_rate),
                "year_sign_consistency": float(year_sign_consistency if relative_side_mode else 1.0),
            }
        if not buckets:
            continue
        bucket_total += len(buckets)
        scopes_out.append(
            {
                "name": str(scope_name),
                "fields": list(fields),
                "weight": float(weight),
                "min_decisions": int(min_decisions),
                "buckets": dict(buckets),
            }
        )
    model = {
        "enabled": bool(scopes_out),
        "schema_version": "de3_v4_decision_side_model_v1",
        "support_full_decisions": int(support_full),
        "year_coverage_full_years": float(year_full),
        "max_scopes_per_row": int(max_scopes_per_row),
        "max_abs_score": float(max_abs_score),
        "scopes": list(scopes_out),
    }
    summary = {
        "enabled": bool(scopes_out),
        "scopes_selected": int(len(scopes_out)),
        "bucket_count": int(bucket_total),
        "max_scopes_per_row": int(max_scopes_per_row),
    }
    return model, summary


def _evaluate_row(row: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(model, dict) or not bool(model.get("enabled", False)):
        return {"long_score": 0.0, "short_score": 0.0, "no_trade_score": 0.0, "match_count": 0, "matched_scopes": []}
    matches: List[Dict[str, Any]] = []
    max_scopes_per_row = max(1, _safe_int(model.get("max_scopes_per_row", 3), 3))
    for raw_scope in model.get("scopes", []):
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
        if not fields:
            continue
        bucket_key = _bucket_key(row, fields)
        if not bucket_key:
            continue
        buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets"), dict) else {}
        bucket = buckets.get(bucket_key, {})
        if not isinstance(bucket, dict) or not bucket:
            continue
        weight = max(0.0, _safe_float(scope.get("weight", 1.0), 1.0))
        if weight <= 0.0:
            continue
        matches.append(
            {
                "scope_name": str(scope.get("name", "") or ""),
                "weight": float(weight),
                "weighted_abs": float(
                    max(
                        abs(_safe_float(bucket.get("long_score"), 0.0)),
                        abs(_safe_float(bucket.get("short_score"), 0.0)),
                        abs(_safe_float(bucket.get("no_trade_score"), 0.0)),
                    )
                    * weight
                ),
                "long_score": float(_safe_float(bucket.get("long_score"), 0.0)),
                "short_score": float(_safe_float(bucket.get("short_score"), 0.0)),
                "no_trade_score": float(_safe_float(bucket.get("no_trade_score"), 0.0)),
            }
        )
    if not matches:
        return {"long_score": 0.0, "short_score": 0.0, "no_trade_score": 0.0, "match_count": 0, "matched_scopes": []}
    matches.sort(key=lambda item: float(item["weighted_abs"]), reverse=True)
    matches = matches[:max_scopes_per_row]
    total_weight = sum(max(0.0, float(item["weight"])) for item in matches)
    long_score = sum(float(item["long_score"]) * float(item["weight"]) for item in matches) / max(total_weight, 1e-9)
    short_score = sum(float(item["short_score"]) * float(item["weight"]) for item in matches) / max(total_weight, 1e-9)
    no_trade_score = sum(float(item["no_trade_score"]) * float(item["weight"]) for item in matches) / max(total_weight, 1e-9)
    return {
        "long_score": float(long_score),
        "short_score": float(short_score),
        "no_trade_score": float(no_trade_score),
        "match_count": int(len(matches)),
        "matched_scopes": [str(item["scope_name"]) for item in matches if str(item["scope_name"])],
    }


def _apply_model(row: Dict[str, Any], model: Dict[str, Any], params: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
    if not _row_model_applicable(row=row, model=model):
        return "", {"long_score": 0.0, "short_score": 0.0, "no_trade_score": 0.0, "match_count": 0, "matched_scopes": []}
    eval_row = _evaluate_row(row, model)
    long_available = bool(row.get("long_available", False))
    short_available = bool(row.get("short_available", False))
    side_scores: Dict[str, float] = {}
    if long_available:
        side_scores["long"] = float(eval_row.get("long_score", 0.0) or 0.0)
    if short_available:
        side_scores["short"] = float(eval_row.get("short_score", 0.0) or 0.0)
    if int(eval_row.get("match_count", 0) or 0) < max(1, int(params.get("min_match_count", 1) or 1)):
        return "", eval_row
    if not side_scores:
        return "no_trade", eval_row
    best_side = max(side_scores.items(), key=lambda item: float(item[1]))[0]
    best_score = float(side_scores[best_side])
    other_scores = [float(v) for k, v in side_scores.items() if k != best_side]
    second_score = max(other_scores) if other_scores else float("-inf")
    no_trade_score = float(eval_row.get("no_trade_score", 0.0) or 0.0)
    if no_trade_score >= float(params.get("no_trade_threshold", 0.0) or 0.0) and no_trade_score >= (
        best_score + float(params.get("no_trade_margin", 0.0) or 0.0)
    ):
        return "no_trade", eval_row
    if best_score >= float(params.get("trade_threshold", 0.0) or 0.0) and (
        not math.isfinite(second_score) or (best_score - second_score) >= float(params.get("side_margin", 0.0) or 0.0)
    ):
        return str(best_side), eval_row
    return "", eval_row


def _points_for_action(row: Dict[str, Any], action: str) -> float:
    action_norm = str(action or "").strip().lower()
    if action_norm == "long" and bool(row.get("long_available", False)):
        return float(_safe_float(row.get("long_pnl_points"), 0.0))
    if action_norm == "short" and bool(row.get("short_available", False)):
        return float(_safe_float(row.get("short_pnl_points"), 0.0))
    return 0.0


def _evaluate_thresholds(tune_df: pd.DataFrame, model: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    trials: List[Dict[str, Any]] = []
    min_override_count = max(1, _safe_int(cfg.get("min_override_count", 90), 90))
    dd_weight = float(_safe_float(cfg.get("objective_weight_max_drawdown", 0.12), 0.12))
    min_trade_score_to_override = float(_safe_float(cfg.get("min_trade_score_to_override", 0.0), 0.0))
    for trade_threshold in cfg.get("trade_threshold_candidates", [0.08]):
        for side_margin in cfg.get("side_margin_candidates", [0.04]):
            for no_trade_threshold in cfg.get("no_trade_threshold_candidates", [0.12]):
                for no_trade_margin in cfg.get("no_trade_margin_candidates", [0.04]):
                    params = {
                        "trade_threshold": float(trade_threshold),
                        "side_margin": float(side_margin),
                        "no_trade_threshold": float(no_trade_threshold),
                        "no_trade_margin": float(no_trade_margin),
                        "min_match_count": 1,
                    }
                    rows: List[Dict[str, Any]] = []
                    for row in tune_df.to_dict("records"):
                        predicted, eval_row = _apply_model(row, model, params)
                        baseline_action = str(row.get("chosen_side", "") or "").strip().lower()
                        if baseline_action not in {"long", "short"}:
                            baseline_action = "no_trade"
                        applied_action = predicted if predicted in {"long", "short", "no_trade"} else baseline_action
                        applied_points = _points_for_action(row, applied_action)
                        baseline_points = _points_for_action(row, baseline_action)
                        side_scores = []
                        if bool(row.get("long_available", False)):
                            side_scores.append(float(eval_row.get("long_score", 0.0) or 0.0))
                        if bool(row.get("short_available", False)):
                            side_scores.append(float(eval_row.get("short_score", 0.0) or 0.0))
                        rows.append(
                            {
                                "timestamp": row.get("timestamp"),
                                "baseline_points": float(baseline_points),
                                "applied_points": float(applied_points),
                                "predicted_action": str(predicted),
                                "baseline_action": str(baseline_action),
                                "applied_action": str(applied_action),
                                "override": bool(predicted in {"long", "short", "no_trade"} and predicted != baseline_action),
                                "best_side_score": max(side_scores) if side_scores else float("-inf"),
                            }
                        )
                    eval_df = pd.DataFrame(rows)
                    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"], errors="coerce", utc=True)
                    eval_df = eval_df.sort_values("timestamp", kind="mergesort")
                    equity = eval_df["applied_points"].cumsum()
                    max_dd = float((equity.cummax() - equity).max()) if not equity.empty else 0.0
                    total_points = float(eval_df["applied_points"].sum()) if not eval_df.empty else 0.0
                    baseline_points = float(eval_df["baseline_points"].sum()) if not eval_df.empty else 0.0
                    override_count = int(eval_df["override"].fillna(False).astype(bool).sum()) if not eval_df.empty else 0
                    score_mean = (
                        float(eval_df.loc[eval_df["override"], "best_side_score"].mean())
                        if override_count > 0
                        else 0.0
                    )
                    valid = bool(override_count >= min_override_count and score_mean >= min_trade_score_to_override)
                    objective = float(total_points - (dd_weight * max_dd))
                    trials.append(
                        {
                            "trade_threshold": float(trade_threshold),
                            "side_margin": float(side_margin),
                            "no_trade_threshold": float(no_trade_threshold),
                            "no_trade_margin": float(no_trade_margin),
                            "valid": bool(valid),
                            "override_count": int(override_count),
                            "override_rate": float(override_count / max(1, len(eval_df))),
                            "baseline_points": float(baseline_points),
                            "total_points": float(total_points),
                            "points_uplift": float(total_points - baseline_points),
                            "max_drawdown_points": float(max_dd),
                            "objective": float(objective),
                        }
                    )
    baseline_trial = {
        "trade_threshold": float("inf"),
        "side_margin": 0.0,
        "no_trade_threshold": float("inf"),
        "no_trade_margin": float("inf"),
        "valid": True,
        "override_count": 0,
        "override_rate": 0.0,
        "baseline_points": float(pd.to_numeric(tune_df["baseline_points"], errors="coerce").fillna(0.0).sum()),
        "total_points": float(pd.to_numeric(tune_df["baseline_points"], errors="coerce").fillna(0.0).sum()),
        "points_uplift": 0.0,
        "max_drawdown_points": 0.0,
        "objective": float(pd.to_numeric(tune_df["baseline_points"], errors="coerce").fillna(0.0).sum()),
    }
    valid_trials = [trial for trial in trials if bool(trial.get("valid", False))]
    candidate_trials = list(valid_trials) + [dict(baseline_trial)]
    if valid_trials:
        candidate_trials.sort(
            key=lambda item: (float(item.get("objective", float("-inf"))), float(item.get("points_uplift", float("-inf")))),
            reverse=True,
        )
        selected = dict(candidate_trials[0])
    else:
        selected = dict(baseline_trial)
    return {"selected": dict(selected), "baseline": dict(baseline_trial), "trials": list(trials)}


def train_candidates(
    *,
    dataset_path: Path,
    base_bundle_path: Path,
    output_dir: Path,
    side_patterns: List[str],
    profile_names: List[str],
) -> None:
    data = pd.read_csv(dataset_path)
    data = _prepare_frame(data)
    data = data[data["year"].notna()].copy()
    allowed_side_patterns = [str(v).strip().lower() for v in side_patterns if str(v).strip()]
    if allowed_side_patterns:
        data = data[data["side_pattern"].astype(str).str.strip().str.lower().isin(allowed_side_patterns)].copy()
    train_df = data[data["year"] <= 2023].copy()
    tune_df = data[data["year"] == 2024].copy()
    if train_df.empty or tune_df.empty:
        raise SystemExit("Decision-side dataset missing train/tune rows.")
    base_bundle = json.loads(base_bundle_path.read_text(encoding="utf-8"))
    profiles = _default_profiles()
    selected_profile_names = [str(v).strip() for v in profile_names if str(v).strip()]
    if selected_profile_names:
        profiles = {
            name: cfg
            for name, cfg in profiles.items()
            if name in set(selected_profile_names)
        }
        if not profiles:
            raise SystemExit("No matching decision-side profiles selected.")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "base_bundle_path": str(base_bundle_path),
        "side_patterns": list(allowed_side_patterns),
        "profile_names": list(selected_profile_names),
        "candidates": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in profiles.items():
        model, model_summary = _build_model(train_df=train_df, cfg=cfg)
        _apply_cfg_metadata_to_model(model=model, cfg=cfg)
        threshold_eval = _evaluate_thresholds(tune_df=tune_df, model=model, cfg=cfg)
        selected = dict(threshold_eval["selected"])
        model["selected_trade_threshold"] = float(selected.get("trade_threshold", float("inf")) or float("inf"))
        model["selected_side_margin"] = float(selected.get("side_margin", 0.0) or 0.0)
        model["selected_no_trade_threshold"] = float(selected.get("no_trade_threshold", float("inf")) or float("inf"))
        model["selected_no_trade_margin"] = float(selected.get("no_trade_margin", 0.0) or 0.0)
        model["selected_threshold_source"] = (
            "tune_2024_optimized" if bool(selected.get("override_count", 0)) else "baseline_no_override"
        )
        bundle = copy.deepcopy(base_bundle)
        bundle["decision_side_model"] = copy.deepcopy(model)
        report = {
            "status": "ok",
            "dataset_path": str(dataset_path),
            "train_rows": int(len(train_df)),
            "tune_rows": int(len(tune_df)),
            "side_patterns": list(allowed_side_patterns),
            "model_summary": dict(model_summary),
            "baseline_tune_metrics": dict(threshold_eval["baseline"]),
            "selected_tune_metrics": dict(selected),
            "threshold_trials": list(threshold_eval["trials"]),
        }
        bundle["decision_side_training_report"] = copy.deepcopy(report)
        meta = bundle.get("metadata", {}) if isinstance(bundle.get("metadata"), dict) else {}
        meta["decision_side_model_retrained_at_utc"] = datetime.now(timezone.utc).isoformat()
        meta["decision_side_model_dataset_path"] = str(dataset_path)
        meta["decision_side_candidate_name"] = str(name)
        if allowed_side_patterns:
            meta["decision_side_model_side_patterns"] = list(allowed_side_patterns)
        bundle["metadata"] = meta
        bundle_path = output_dir / f"dynamic_engine3_v4_bundle.{name}.json"
        report_path = output_dir / f"{name}.decision_side_training_report.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        summary["candidates"][name] = {
            "bundle_path": str(bundle_path),
            "report_path": str(report_path),
            "tune_objective_score": float(selected.get("objective", float("-inf"))),
            "tune_points_uplift": float(selected.get("points_uplift", 0.0)),
            "tune_override_count": int(selected.get("override_count", 0)),
            "tune_override_rate": float(selected.get("override_rate", 0.0)),
            "selected_trade_threshold": model.get("selected_trade_threshold"),
            "selected_side_margin": model.get("selected_side_margin"),
            "selected_no_trade_threshold": model.get("selected_no_trade_threshold"),
            "selected_no_trade_margin": model.get("selected_no_trade_margin"),
            "bucket_count": int(model_summary.get("bucket_count", 0)),
            "scopes_selected": int(model_summary.get("scopes_selected", 0)),
        }
        print(
            f"{name}: trade_thr={model.get('selected_trade_threshold')} "
            f"side_margin={model.get('selected_side_margin')} "
            f"no_trade_thr={model.get('selected_no_trade_threshold')} "
            f"uplift={selected.get('points_uplift')} "
            f"overrides={selected.get('override_count')}"
        )
    summary_path = output_dir / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"candidate_summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DE3 decision-level side-choice bundle candidates from a decision-side dataset."
    )
    parser.add_argument("--dataset", default="reports/de3_decision_side_dataset.csv")
    parser.add_argument("--base-bundle", default="artifacts/de3_v4_live/latest.json")
    parser.add_argument("--output-dir", default="artifacts/de3_decision_side_candidates")
    parser.add_argument(
        "--side-patterns",
        default="",
        help="Optional comma-separated filter on dataset side_pattern values, e.g. both or both,long_only.",
    )
    parser.add_argument(
        "--profiles",
        default="",
        help="Optional comma-separated profile names to train.",
    )
    args = parser.parse_args()
    side_patterns = [part.strip().lower() for part in str(args.side_patterns or "").split(",") if part.strip()]
    profile_names = [part.strip() for part in str(args.profiles or "").split(",") if part.strip()]
    train_candidates(
        dataset_path=_resolve_path(str(args.dataset)),
        base_bundle_path=_resolve_path(str(args.base_bundle)),
        output_dir=_resolve_output_dir(str(args.output_dir)),
        side_patterns=side_patterns,
        profile_names=profile_names,
    )


if __name__ == "__main__":
    main()
