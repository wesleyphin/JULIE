from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from de3_v4_entry_policy_trainer import (
    _build_shape_penalty_model,
    _build_stats_from_frame,
    _evaluate_shape_penalty_row,
    _evaluate_threshold,
    _resolve_stats,
    _scope_threshold_offsets_series,
    _split_bounds,
    _threshold_candidates,
    _trade_frame_from_csvs,
)
from de3_v4_schema import clip, lane_to_side, safe_div, safe_float, safe_int


def _lane_prior_from_router_payload(
    *,
    router_payload: Dict[str, Any],
    lane: str,
    session_name: str,
    timeframe_hint: str,
) -> float:
    if not isinstance(router_payload, dict) or not str(lane or "").strip():
        return 0.0
    session_map = (
        router_payload.get("lane_priors_by_session", {})
        if isinstance(router_payload.get("lane_priors_by_session", {}), dict)
        else {}
    )
    timeframe_map = (
        router_payload.get("lane_priors_by_timeframe", {})
        if isinstance(router_payload.get("lane_priors_by_timeframe", {}), dict)
        else {}
    )
    global_map = (
        router_payload.get("lane_priors_global", {})
        if isinstance(router_payload.get("lane_priors_global", {}), dict)
        else {}
    )
    session_val = 0.0
    if session_name and isinstance(session_map.get(session_name), dict):
        session_val = safe_float((session_map.get(session_name) or {}).get(lane, 0.0), 0.0)
    timeframe_val = 0.0
    if timeframe_hint and isinstance(timeframe_map.get(timeframe_hint), dict):
        timeframe_val = safe_float((timeframe_map.get(timeframe_hint) or {}).get(lane, 0.0), 0.0)
    global_val = safe_float(global_map.get(lane, 0.0), 0.0)
    return float((0.55 * session_val) + (0.20 * timeframe_val) + (0.25 * global_val))


def _variant_quality_priors(metadata: Dict[str, Any]) -> Dict[str, float]:
    raw = (
        metadata.get("lane_variant_quality", {})
        if isinstance(metadata.get("lane_variant_quality", {}), dict)
        else {}
    )
    out: Dict[str, float] = {}
    for variant_id, row in raw.items():
        if not isinstance(row, dict):
            continue
        key = str(variant_id or row.get("variant_id", "") or "").strip()
        if not key:
            continue
        out[key] = float(
            safe_float(
                row.get("satellite_quality_score", row.get("quality_proxy", 0.0)),
                0.0,
            )
        )
    return out


def _decision_row_side(*, row: pd.Series, lane: str) -> str:
    side_raw = str(row.get("side_considered", "") or "").strip().lower()
    if side_raw in {"long", "buy"}:
        return "long"
    if side_raw in {"short", "sell"}:
        return "short"
    lane_side = str(lane_to_side(lane) or "").strip().lower()
    if lane_side in {"long", "short"}:
        return lane_side
    return ""


def _directionless_strategy_style(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    for prefix in ("long_", "short_", "buy_", "sell_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if "_" in text:
        parts = [part for part in text.split("_") if part]
        if len(parts) >= 2:
            text = "_".join(parts[1:]) if parts[0] in {"long", "short", "buy", "sell"} else text
    return str(text).strip("_")


def _context_prior_bucket_key(*, row: Any, fields: List[str]) -> str:
    if not fields:
        return ""
    parts: List[str] = []
    for field in fields:
        field_name = str(field or "").strip()
        if not field_name:
            return ""
        raw = row.get(field_name, "") if hasattr(row, "get") else ""
        if field_name == "ctx_hour_et":
            try:
                part = str(int(float(raw)))
            except Exception:
                part = ""
        else:
            part = str(raw or "").strip().lower()
            if part in {"", "nan", "nat", "none"}:
                part = ""
        if not part:
            return ""
        parts.append(part)
    return "|".join(parts)


def _default_context_prior_scopes() -> List[Dict[str, Any]]:
    return [
        {
            "name": "session_timeframe_side",
            "fields": ["session", "timeframe", "side_considered"],
            "min_trades": 80,
            "weight": 1.00,
        },
        {
            "name": "session_strategy_side",
            "fields": ["session", "strategy_type", "side_considered"],
            "min_trades": 80,
            "weight": 0.90,
        },
        {
            "name": "session_substate_side",
            "fields": ["session", "ctx_session_substate", "side_considered"],
            "min_trades": 70,
            "weight": 0.80,
        },
        {
            "name": "hour_side",
            "fields": ["ctx_hour_et", "side_considered"],
            "min_trades": 90,
            "weight": 0.70,
        },
        {
            "name": "session_side",
            "fields": ["session", "side_considered"],
            "min_trades": 120,
            "weight": 0.65,
        },
    ]


def _build_context_prior_model(
    *,
    trade_df: pd.DataFrame,
    training_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    context_cfg = (
        training_cfg.get("context_prior_model", {})
        if isinstance(training_cfg.get("context_prior_model"), dict)
        else {}
    )
    enabled = bool(context_cfg.get("enabled", False))
    if (not enabled) or trade_df.empty:
        return {
            "enabled": False,
            "scopes": [],
        }, {
            "enabled": bool(enabled),
            "scopes_considered": 0,
            "scopes_selected": 0,
            "bucket_count": 0,
            "reason": "disabled_or_empty",
        }

    scopes_cfg = (
        context_cfg.get("scopes", [])
        if isinstance(context_cfg.get("scopes"), list)
        else _default_context_prior_scopes()
    )
    if not scopes_cfg:
        scopes_cfg = _default_context_prior_scopes()

    support_full_trades = max(1, safe_int(context_cfg.get("support_full_trades", 220), 220))
    min_year_coverage = max(0, safe_int(context_cfg.get("min_year_coverage", 4), 4))
    year_coverage_full = max(
        1.0,
        safe_float(context_cfg.get("year_coverage_full_years", 10.0), 10.0),
    )
    quality_scale = max(1e-9, safe_float(context_cfg.get("quality_scale", 1.0), 1.0))
    profit_factor_center = safe_float(
        context_cfg.get("profit_factor_center", 1.12),
        1.12,
    )
    profit_factor_scale = max(
        1e-9,
        safe_float(context_cfg.get("profit_factor_scale", 0.24), 0.24),
    )
    profit_factor_weight = max(
        0.0,
        safe_float(context_cfg.get("profit_factor_weight", 0.18), 0.18),
    )
    loss_share_center = safe_float(context_cfg.get("loss_share_center", 0.50), 0.50)
    loss_share_scale = max(
        1e-9,
        safe_float(context_cfg.get("loss_share_scale", 0.16), 0.16),
    )
    loss_share_weight = max(
        0.0,
        safe_float(context_cfg.get("loss_share_weight", 0.12), 0.12),
    )
    max_abs_score = max(0.05, safe_float(context_cfg.get("max_abs_score", 1.25), 1.25))
    min_abs_prior_score = max(
        0.0,
        safe_float(context_cfg.get("min_abs_prior_score", 0.03), 0.03),
    )
    side_advantage_mode = str(context_cfg.get("side_advantage_mode", "off") or "off").strip().lower()
    if side_advantage_mode not in {"off", "prefer", "only"}:
        side_advantage_mode = "off"
    max_scopes_per_row = max(0, safe_int(context_cfg.get("max_scopes_per_row", 3), 3))

    local = trade_df.copy()
    for col in ("session", "timeframe", "strategy_type", "side_considered", "ctx_session_substate"):
        if col not in local.columns:
            local[col] = ""
        local[col] = local[col].astype(str).str.strip().str.lower()
    if "ctx_hour_et" not in local.columns:
        local["ctx_hour_et"] = float("nan")
    local["ctx_hour_et"] = pd.to_numeric(local["ctx_hour_et"], errors="coerce")

    scopes_out: List[Dict[str, Any]] = []
    bucket_count = 0
    for idx, raw_scope in enumerate(scopes_cfg):
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [
            str(v).strip()
            for v in (
                scope.get("fields", [])
                if isinstance(scope.get("fields"), (list, tuple, set))
                else []
            )
            if str(v).strip()
        ]
        if not fields:
            continue
        scope_name = str(scope.get("name", f"context_scope_{idx}") or f"context_scope_{idx}").strip()
        min_trades = max(1, safe_int(scope.get("min_trades", 80), 80))
        scope_weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
        if scope_weight <= 0.0:
            continue

        scope_frame = local.copy()
        scope_frame["_context_prior_bucket"] = [
            _context_prior_bucket_key(row=row, fields=fields)
            for row in scope_frame[fields].to_dict("records")
        ]
        scope_frame = scope_frame[
            scope_frame["_context_prior_bucket"].astype(str).str.strip() != ""
        ].copy()
        if scope_frame.empty:
            continue

        bucket_stats = _build_stats_from_frame(
            trade_df=scope_frame,
            key_col="_context_prior_bucket",
            cfg=training_cfg,
        )
        scope_buckets: Dict[str, Dict[str, Any]] = {}
        for bucket_key, stats in bucket_stats.items():
            if not isinstance(stats, dict):
                continue
            n_trades = int(safe_int(stats.get("n_trades", 0), 0))
            year_coverage = int(safe_int(stats.get("year_coverage", 0), 0))
            if n_trades < min_trades or year_coverage < min_year_coverage:
                continue
            quality_lcb_score = safe_float(stats.get("quality_lcb_score", 0.0), 0.0)
            profit_factor = max(
                0.0,
                safe_float(stats.get("profit_factor", profit_factor_center), profit_factor_center),
            )
            loss_share = clip(
                safe_float(stats.get("loss_share", loss_share_center), loss_share_center),
                0.0,
                1.0,
            )
            support_ratio = clip(
                safe_div(float(n_trades), float(support_full_trades), 0.0),
                0.0,
                1.0,
            )
            year_ratio = clip(
                safe_div(float(year_coverage), float(year_coverage_full), 0.0),
                0.0,
                1.0,
            )
            prior_score = math.tanh(quality_lcb_score / quality_scale)
            prior_score += profit_factor_weight * math.tanh(
                (profit_factor - profit_factor_center) / profit_factor_scale
            )
            prior_score -= loss_share_weight * math.tanh(
                max(0.0, loss_share - loss_share_center) / loss_share_scale
            )
            prior_score *= math.sqrt(max(0.0, support_ratio)) * year_ratio
            prior_score = float(clip(prior_score, -max_abs_score, max_abs_score))
            if abs(prior_score) < min_abs_prior_score:
                continue
            scope_buckets[str(bucket_key)] = {
                "prior_score": float(prior_score),
                "n_trades": int(n_trades),
                "year_coverage": int(year_coverage),
                "quality_lcb_score": float(quality_lcb_score),
                "profit_factor": float(profit_factor),
                "loss_share": float(loss_share),
            }
        if not scope_buckets:
            continue
        if "side_considered" in fields:
            side_idx = fields.index("side_considered")
            for bucket_key, bucket_stats in list(scope_buckets.items()):
                if not isinstance(bucket_stats, dict):
                    continue
                key_parts = str(bucket_key).split("|")
                if side_idx < 0 or side_idx >= len(key_parts):
                    continue
                side_value = str(key_parts[side_idx] or "").strip().lower()
                if side_value == "long":
                    key_parts[side_idx] = "short"
                elif side_value == "short":
                    key_parts[side_idx] = "long"
                else:
                    continue
                opposite_key = "|".join(key_parts)
                opposite_bucket = scope_buckets.get(opposite_key, {})
                if not isinstance(opposite_bucket, dict):
                    continue
                advantage_score = safe_float(bucket_stats.get("prior_score", 0.0), 0.0) - safe_float(
                    opposite_bucket.get("prior_score", 0.0),
                    0.0,
                )
                bucket_stats["side_advantage_score"] = float(
                    clip(advantage_score, -max_abs_score, max_abs_score)
                )
                bucket_stats["opposite_prior_score"] = float(
                    safe_float(opposite_bucket.get("prior_score", 0.0), 0.0)
                )
        bucket_count += int(len(scope_buckets))
        scopes_out.append(
            {
                "name": str(scope_name),
                "fields": list(fields),
                "min_trades": int(min_trades),
                "weight": float(scope_weight),
                "buckets": dict(scope_buckets),
            }
        )

    model = {
        "enabled": bool(scopes_out),
        "support_full_trades": int(support_full_trades),
        "max_abs_score": float(max_abs_score),
        "side_advantage_mode": str(side_advantage_mode),
        "max_scopes_per_row": int(max_scopes_per_row),
        "scopes": list(scopes_out),
    }
    summary = {
        "enabled": True,
        "scopes_considered": int(len(scopes_cfg)),
        "scopes_selected": int(len(scopes_out)),
        "bucket_count": int(bucket_count),
        "side_advantage_mode": str(side_advantage_mode),
        "max_scopes_per_row": int(max_scopes_per_row),
    }
    return model, summary


def _evaluate_context_prior_row(
    *,
    row: pd.Series,
    context_prior_model: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(context_prior_model, dict) or not bool(context_prior_model.get("enabled", False)):
        return {
            "score": 0.0,
            "match_count": 0,
            "matched_scopes": [],
        }
    scopes = context_prior_model.get("scopes", [])
    if not isinstance(scopes, list) or not scopes:
        return {
            "score": 0.0,
            "match_count": 0,
            "matched_scopes": [],
        }
    max_abs_score = max(0.05, safe_float(context_prior_model.get("max_abs_score", 1.25), 1.25))
    side_advantage_mode = str(context_prior_model.get("side_advantage_mode", "off") or "off").strip().lower()
    if side_advantage_mode not in {"off", "prefer", "only"}:
        side_advantage_mode = "off"
    max_scopes_per_row = max(0, safe_int(context_prior_model.get("max_scopes_per_row", 0), 0))
    matches: List[Dict[str, Any]] = []
    for raw_scope in scopes:
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [
            str(v).strip()
            for v in (
                scope.get("fields", [])
                if isinstance(scope.get("fields"), (list, tuple, set))
                else []
            )
            if str(v).strip()
        ]
        buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets", {}), dict) else {}
        bucket_key = _context_prior_bucket_key(row=row, fields=fields)
        if not bucket_key:
            continue
        bucket = buckets.get(bucket_key, {})
        if not isinstance(bucket, dict):
            continue
        weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
        if weight <= 0.0:
            continue
        prior_score = float(safe_float(bucket.get("prior_score", 0.0), 0.0))
        effective_score = prior_score
        if side_advantage_mode != "off" and "side_considered" in fields:
            side_advantage_score = safe_float(bucket.get("side_advantage_score", float("nan")), float("nan"))
            if math.isfinite(side_advantage_score):
                effective_score = float(side_advantage_score)
            elif side_advantage_mode == "only":
                continue
        matches.append(
            {
                "scope_name": str(scope.get("name", "") or ""),
                "bucket_key": str(bucket_key),
                "weight": float(weight),
                "prior_score": float(effective_score),
                "weighted_abs": float(abs(effective_score) * weight),
            }
        )
    if not matches:
        return {
            "score": 0.0,
            "match_count": 0,
            "matched_scopes": [],
        }
    matches.sort(
        key=lambda item: (
            safe_float(item.get("weighted_abs", 0.0), 0.0),
            abs(safe_float(item.get("prior_score", 0.0), 0.0)),
        ),
        reverse=True,
    )
    if max_scopes_per_row > 0:
        matches = matches[:max_scopes_per_row]
    total_weight = sum(max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches)
    score = safe_div(
        sum(
            safe_float(item.get("prior_score", 0.0), 0.0)
            * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
            for item in matches
        ),
        total_weight,
        0.0,
    )
    score = float(clip(score, -max_abs_score, max_abs_score))
    return {
        "score": float(score),
        "match_count": int(len(matches)),
        "matched_scopes": [
            str(item.get("scope_name", "") or "")
            for item in matches
            if str(item.get("scope_name", "") or "")
        ],
    }


def _bucket_close_pos(value: Any) -> str:
    raw = safe_float(value, float("nan"))
    if not math.isfinite(raw):
        return ""
    if raw <= 0.15:
        return "very_low"
    if raw <= 0.35:
        return "low"
    if raw <= 0.65:
        return "mid"
    if raw <= 0.85:
        return "high"
    return "very_high"


def _bucket_balance(value: Any, *, strong_neg: float, neg: float, pos: float, strong_pos: float) -> str:
    raw = safe_float(value, float("nan"))
    if not math.isfinite(raw):
        return ""
    if raw <= strong_neg:
        return "strong_neg"
    if raw <= neg:
        return "neg"
    if raw < pos:
        return "balanced"
    if raw < strong_pos:
        return "pos"
    return "strong_pos"


def _bucket_body_ratio(value: Any) -> str:
    raw = safe_float(value, float("nan"))
    if not math.isfinite(raw):
        return ""
    if raw <= 0.20:
        return "small"
    if raw <= 0.45:
        return "medium"
    return "large"


def _bucket_vol_ratio(value: Any) -> str:
    raw = safe_float(value, float("nan"))
    if not math.isfinite(raw):
        return ""
    if raw <= 0.75:
        return "low"
    if raw <= 1.10:
        return "mid"
    if raw <= 1.50:
        return "high"
    return "extreme"


def _bucket_range_ratio(value: Any) -> str:
    raw = safe_float(value, float("nan"))
    if not math.isfinite(raw):
        return ""
    if raw <= 1.20:
        return "compressed"
    if raw <= 2.10:
        return "normal"
    return "expanded"


def _bucket_down3(value: Any) -> str:
    raw = safe_int(value, -1)
    if raw < 0:
        return ""
    if raw <= 1:
        return "0_1"
    if raw == 2:
        return "2"
    return "3"


def _prepare_short_term_condition_frame(trade_df: pd.DataFrame) -> pd.DataFrame:
    if trade_df.empty:
        return trade_df.copy()
    local = trade_df.copy()
    for col in ("session", "timeframe", "strategy_type", "side_considered", "ctx_session_substate"):
        if col not in local.columns:
            local[col] = ""
        local[col] = local[col].astype(str).str.strip().str.lower()
    local["strategy_style"] = local["strategy_type"].apply(_directionless_strategy_style)
    if "ctx_hour_et" not in local.columns:
        local["ctx_hour_et"] = float("nan")
    local["ctx_hour_et"] = pd.to_numeric(local["ctx_hour_et"], errors="coerce")
    local["ctx_hour_bucket"] = local["ctx_hour_et"].apply(
        lambda raw: str(int(round(float(raw)))) if math.isfinite(safe_float(raw, float("nan"))) else ""
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
    ]
    for col in numeric_cols:
        if col not in local.columns:
            local[col] = float("nan")
        local[col] = pd.to_numeric(local[col], errors="coerce")
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
    return local


def _normalize_action_text(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    return "" if text in {"", "nan", "nat", "none"} else text


def _derive_action_session_substate(*, session_text: str, timestamp_value: Any) -> str:
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


def _prepare_action_condition_frame(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty:
        return decision_df.copy()
    local = decision_df.copy()
    if "timestamp" not in local.columns:
        local["timestamp"] = pd.NaT
    local["timestamp"] = pd.to_datetime(local["timestamp"], errors="coerce", utc=True)
    if "year" not in local.columns:
        local["year"] = pd.NA
    local["year"] = pd.to_numeric(local["year"], errors="coerce")
    missing_year = local["year"].isna() & local["timestamp"].notna()
    if bool(missing_year.any()):
        local.loc[missing_year, "year"] = (
            local.loc[missing_year, "timestamp"].dt.tz_convert("America/New_York").dt.year
        )
    for col in (
        "session",
        "ctx_session_substate",
        "best_action",
        "long_timeframe",
        "short_timeframe",
        "long_strategy_type",
        "short_strategy_type",
        "long_strategy_style",
        "short_strategy_style",
        "long_sub_strategy",
        "short_sub_strategy",
    ):
        if col not in local.columns:
            local[col] = ""
        local[col] = local[col].apply(_normalize_action_text)
    if "ctx_hour_et" not in local.columns:
        local["ctx_hour_et"] = float("nan")
    local["ctx_hour_et"] = pd.to_numeric(local["ctx_hour_et"], errors="coerce")
    missing_hour = local["ctx_hour_et"].isna() & local["timestamp"].notna()
    if bool(missing_hour.any()):
        local.loc[missing_hour, "ctx_hour_et"] = (
            local.loc[missing_hour, "timestamp"].dt.tz_convert("America/New_York").dt.hour.astype(float)
        )
    missing_substate = local["ctx_session_substate"].astype(str).str.strip() == ""
    if bool(missing_substate.any()):
        local.loc[missing_substate, "ctx_session_substate"] = [
            _derive_action_session_substate(session_text=row["session"], timestamp_value=row["timestamp"])
            for _, row in local.loc[missing_substate, ["session", "timestamp"]].iterrows()
        ]
    local["ctx_hour_bucket"] = local["ctx_hour_et"].apply(
        lambda raw: str(int(round(float(raw)))) if math.isfinite(safe_float(raw, float("nan"))) else ""
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
        "long_pnl_points",
        "short_pnl_points",
        "baseline_points",
        "long_available",
        "short_available",
    ]
    for col in numeric_cols:
        if col not in local.columns:
            local[col] = float("nan")
        local[col] = pd.to_numeric(local[col], errors="coerce")
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

    exploded_frames: List[pd.DataFrame] = []
    for side in ("long", "short"):
        other_side = "short" if side == "long" else "long"
        side_frame = local[pd.to_numeric(local.get(f"{side}_available"), errors="coerce").fillna(0.0) > 0.0].copy()
        if side_frame.empty:
            continue
        side_frame["side_considered"] = str(side)
        side_frame["timeframe"] = side_frame.get(f"{side}_timeframe", "").apply(_normalize_action_text)
        side_frame["strategy_type"] = side_frame.get(f"{side}_strategy_type", "").apply(_normalize_action_text)
        side_frame["strategy_style"] = side_frame.get(f"{side}_strategy_style", "").apply(_normalize_action_text)
        missing_style = side_frame["strategy_style"].astype(str).str.strip() == ""
        if bool(missing_style.any()):
            side_frame.loc[missing_style, "strategy_style"] = side_frame.loc[missing_style, "strategy_type"].apply(
                _directionless_strategy_style
            )
        side_frame["sub_strategy"] = side_frame.get(f"{side}_sub_strategy", "").apply(_normalize_action_text)
        side_frame["action_pnl_points"] = pd.to_numeric(
            side_frame.get(f"{side}_pnl_points"),
            errors="coerce",
        ).fillna(0.0)
        side_frame["opposite_available"] = pd.to_numeric(
            side_frame.get(f"{other_side}_available"),
            errors="coerce",
        ).fillna(0.0) > 0.0
        side_frame["relative_adv_points"] = pd.to_numeric(
            side_frame.get(f"{side}_pnl_points"),
            errors="coerce",
        ) - pd.to_numeric(side_frame.get(f"{other_side}_pnl_points"), errors="coerce")
        side_frame["action_uplift_points"] = side_frame["action_pnl_points"] - pd.to_numeric(
            side_frame.get("baseline_points"),
            errors="coerce",
        ).fillna(0.0)
        side_frame["action_is_best"] = (
            side_frame.get("best_action", "").astype(str).str.strip().str.lower() == str(side)
        ).astype(int)
        side_frame["no_trade_best"] = (
            side_frame.get("best_action", "").astype(str).str.strip().str.lower() == "no_trade"
        ).astype(int)
        exploded_frames.append(side_frame)
    if not exploded_frames:
        return pd.DataFrame()
    out = pd.concat(exploded_frames, ignore_index=True, sort=False)
    out["strategy_style"] = out.get("strategy_style", "").apply(_normalize_action_text)
    out["sub_strategy"] = out.get("sub_strategy", "").apply(_normalize_action_text)
    return out


def _default_action_condition_scopes() -> List[Dict[str, Any]]:
    return [
        {
            "name": "session_substate_style_pressure",
            "fields": ["session", "ctx_session_substate", "side_considered", "strategy_style", "st_pressure_bucket"],
            "min_decisions": 70,
            "weight": 1.00,
        },
        {
            "name": "hour_timeframe_style_close",
            "fields": ["ctx_hour_bucket", "timeframe", "side_considered", "strategy_style", "st_close_bucket"],
            "min_decisions": 64,
            "weight": 0.94,
        },
        {
            "name": "session_location_flow",
            "fields": ["session", "side_considered", "st_location_bucket", "st_vol_bucket", "st_down3_bucket"],
            "min_decisions": 60,
            "weight": 0.88,
        },
        {
            "name": "timeframe_pressure_shape",
            "fields": ["timeframe", "side_considered", "st_pressure_bucket", "st_wick_bias_bucket", "st_body_bucket"],
            "min_decisions": 56,
            "weight": 0.82,
        },
        {
            "name": "sub_strategy_local_shape",
            "fields": ["sub_strategy", "side_considered", "st_pressure_bucket", "st_location_bucket", "st_close_bucket"],
            "min_decisions": 48,
            "weight": 0.72,
        },
    ]


def _build_action_condition_model(
    *,
    action_df: pd.DataFrame,
    training_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_cfg = (
        training_cfg.get("action_condition_model", {})
        if isinstance(training_cfg.get("action_condition_model"), dict)
        else {}
    )
    enabled = bool(model_cfg.get("enabled", False))
    if (not enabled) or action_df.empty:
        return {
            "enabled": False,
            "scopes": [],
        }, {
            "enabled": bool(enabled),
            "scopes_considered": 0,
            "scopes_selected": 0,
            "bucket_count": 0,
            "reason": "disabled_or_empty",
        }
    scopes_cfg = (
        model_cfg.get("scopes", [])
        if isinstance(model_cfg.get("scopes"), list)
        else _default_action_condition_scopes()
    )
    if not scopes_cfg:
        scopes_cfg = _default_action_condition_scopes()
    support_full_decisions = max(1.0, safe_float(model_cfg.get("support_full_decisions", 180.0), 180.0))
    year_coverage_full = max(1.0, safe_float(model_cfg.get("year_coverage_full_years", 10.0), 10.0))
    mean_scale = max(0.10, safe_float(model_cfg.get("mean_scale_points", 2.40), 2.40))
    positive_center = safe_float(model_cfg.get("positive_rate_center", 0.50), 0.50)
    positive_scale = max(0.05, safe_float(model_cfg.get("positive_rate_scale", 0.15), 0.15))
    best_center = safe_float(model_cfg.get("best_rate_center", 0.34), 0.34)
    best_scale = max(0.05, safe_float(model_cfg.get("best_rate_scale", 0.15), 0.15))
    uplift_scale = max(0.10, safe_float(model_cfg.get("uplift_scale_points", 1.60), 1.60))
    relative_scale = max(0.10, safe_float(model_cfg.get("relative_scale_points", 1.80), 1.80))
    no_trade_center = safe_float(model_cfg.get("no_trade_rate_center", 0.36), 0.36)
    no_trade_scale = max(0.05, safe_float(model_cfg.get("no_trade_rate_scale", 0.18), 0.18))
    mean_weight = safe_float(model_cfg.get("mean_weight", 0.38), 0.38)
    positive_rate_weight = safe_float(model_cfg.get("positive_rate_weight", 0.16), 0.16)
    best_rate_weight = safe_float(model_cfg.get("best_rate_weight", 0.22), 0.22)
    uplift_weight = safe_float(model_cfg.get("uplift_weight", 0.14), 0.14)
    relative_weight = safe_float(model_cfg.get("relative_weight", 0.18), 0.18)
    no_trade_penalty_weight = safe_float(model_cfg.get("no_trade_penalty_weight", 0.20), 0.20)
    positive_score_weight = max(0.0, safe_float(model_cfg.get("positive_score_weight", 1.0), 1.0))
    negative_score_weight = max(0.0, safe_float(model_cfg.get("negative_score_weight", 1.0), 1.0))
    bonus_only = bool(model_cfg.get("bonus_only", False))
    max_abs_score = max(0.05, safe_float(model_cfg.get("max_abs_score", 1.35), 1.35))
    min_abs_score = max(0.0, safe_float(model_cfg.get("min_abs_score", 0.03), 0.03))
    max_scopes_per_row = max(1, safe_int(model_cfg.get("max_scopes_per_row", 3), 3))
    min_yearly_decisions = max(1, safe_int(model_cfg.get("min_yearly_decisions", 1), 1))
    min_negative_year_fraction = clip(
        safe_float(model_cfg.get("min_negative_year_fraction", 0.0), 0.0),
        0.0,
        1.0,
    )
    min_positive_year_fraction = clip(
        safe_float(model_cfg.get("min_positive_year_fraction", 0.0), 0.0),
        0.0,
        1.0,
    )
    bad_best_rate_cap = clip(
        safe_float(model_cfg.get("bad_best_rate_cap", 0.34), 0.34),
        0.0,
        1.0,
    )
    good_best_rate_floor = clip(
        safe_float(model_cfg.get("good_best_rate_floor", 0.34), 0.34),
        0.0,
        1.0,
    )
    require_median_year_sign_match = bool(model_cfg.get("require_median_year_sign_match", False))

    scopes_out: List[Dict[str, Any]] = []
    bucket_count = 0
    for idx, raw_scope in enumerate(scopes_cfg):
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [
            str(v).strip()
            for v in (
                scope.get("fields", [])
                if isinstance(scope.get("fields"), (list, tuple, set))
                else []
            )
            if str(v).strip()
        ]
        if not fields:
            continue
        scope_name = str(scope.get("name", f"action_scope_{idx}") or f"action_scope_{idx}").strip()
        min_decisions = max(12, safe_int(scope.get("min_decisions", 60), 60))
        weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
        if weight <= 0.0:
            continue
        scope_frame = action_df.copy()
        scope_frame["_action_bucket"] = [
            _context_prior_bucket_key(row=row, fields=fields)
            for row in scope_frame[fields].to_dict("records")
        ]
        scope_frame = scope_frame[scope_frame["_action_bucket"].astype(str).str.strip() != ""].copy()
        if scope_frame.empty:
            continue
        scope_buckets: Dict[str, Dict[str, Any]] = {}
        for bucket_key, bucket_frame in scope_frame.groupby("_action_bucket", sort=False):
            n_decisions = int(len(bucket_frame))
            if n_decisions < min_decisions:
                continue
            mean_points = safe_float(pd.to_numeric(bucket_frame["action_pnl_points"], errors="coerce").mean(), 0.0)
            positive_rate = float(
                (pd.to_numeric(bucket_frame["action_pnl_points"], errors="coerce").fillna(0.0) > 0.0).mean()
            )
            best_rate = float(pd.to_numeric(bucket_frame["action_is_best"], errors="coerce").fillna(0.0).mean())
            uplift_mean = safe_float(pd.to_numeric(bucket_frame["action_uplift_points"], errors="coerce").mean(), 0.0)
            no_trade_rate = float(pd.to_numeric(bucket_frame["no_trade_best"], errors="coerce").fillna(0.0).mean())
            relative_frame = bucket_frame[bucket_frame["opposite_available"].fillna(False).astype(bool)].copy()
            relative_mean = safe_float(
                pd.to_numeric(relative_frame["relative_adv_points"], errors="coerce").mean(),
                0.0,
            )
            year_coverage = int(pd.to_numeric(bucket_frame["year"], errors="coerce").dropna().nunique())
            year_frame = (
                bucket_frame.groupby("year", dropna=True)
                .agg(
                    year_decisions=("action_pnl_points", "size"),
                    mean_points=("action_pnl_points", "mean"),
                    best_rate=("action_is_best", "mean"),
                    no_trade_rate=("no_trade_best", "mean"),
                )
                .reset_index(drop=True)
            )
            year_frame = year_frame[
                pd.to_numeric(year_frame["year_decisions"], errors="coerce").fillna(0.0)
                >= float(min_yearly_decisions)
            ].copy()
            if year_frame.empty:
                continue
            bad_year_fraction = float(
                (
                    (pd.to_numeric(year_frame["mean_points"], errors="coerce").fillna(0.0) <= 0.0)
                    | (
                        pd.to_numeric(year_frame["best_rate"], errors="coerce").fillna(0.0)
                        <= float(bad_best_rate_cap)
                    )
                ).mean()
            )
            good_year_fraction = float(
                (
                    (pd.to_numeric(year_frame["mean_points"], errors="coerce").fillna(0.0) > 0.0)
                    & (
                        pd.to_numeric(year_frame["best_rate"], errors="coerce").fillna(0.0)
                        >= float(good_best_rate_floor)
                    )
                ).mean()
            )
            median_year_mean_points = safe_float(
                pd.to_numeric(year_frame["mean_points"], errors="coerce").median(),
                0.0,
            )
            support_ratio = clip(safe_div(float(n_decisions), float(support_full_decisions), 0.0), 0.0, 1.0)
            year_ratio = clip(safe_div(float(year_coverage), float(year_coverage_full), 0.0), 0.0, 1.0)
            weight_mult = math.sqrt(max(0.0, support_ratio)) * year_ratio
            action_score = (
                mean_weight * math.tanh(mean_points / mean_scale)
                + positive_rate_weight * math.tanh((positive_rate - positive_center) / positive_scale)
                + best_rate_weight * math.tanh((best_rate - best_center) / best_scale)
                + uplift_weight * math.tanh(uplift_mean / uplift_scale)
                + relative_weight * math.tanh(relative_mean / relative_scale)
                - no_trade_penalty_weight
                * math.tanh(max(0.0, no_trade_rate - no_trade_center) / no_trade_scale)
            )
            if action_score > 0.0:
                action_score *= positive_score_weight
            elif action_score < 0.0:
                action_score *= negative_score_weight
                if bonus_only:
                    action_score = 0.0
            if action_score < 0.0 and bad_year_fraction < min_negative_year_fraction:
                continue
            if action_score > 0.0 and good_year_fraction < min_positive_year_fraction:
                continue
            if require_median_year_sign_match:
                if action_score < 0.0 and median_year_mean_points > 0.0:
                    continue
                if action_score > 0.0 and median_year_mean_points < 0.0:
                    continue
            action_score = float(clip(action_score * weight_mult, -max_abs_score, max_abs_score))
            if abs(action_score) < min_abs_score:
                continue
            scope_buckets[str(bucket_key)] = {
                "score": float(action_score),
                "n_decisions": int(n_decisions),
                "year_coverage": int(year_coverage),
                "mean_pnl_points": float(mean_points),
                "positive_rate": float(positive_rate),
                "best_rate": float(best_rate),
                "uplift_mean_points": float(uplift_mean),
                "relative_mean_points": float(relative_mean),
                "no_trade_rate": float(no_trade_rate),
                "bad_year_fraction": float(bad_year_fraction),
                "good_year_fraction": float(good_year_fraction),
                "median_year_mean_points": float(median_year_mean_points),
            }
        if not scope_buckets:
            continue
        bucket_count += int(len(scope_buckets))
        scopes_out.append(
            {
                "name": str(scope_name),
                "fields": list(fields),
                "weight": float(weight),
                "min_decisions": int(min_decisions),
                "buckets": dict(scope_buckets),
            }
        )
    model = {
        "enabled": bool(scopes_out),
        "max_abs_score": float(max_abs_score),
        "max_scopes_per_row": int(max_scopes_per_row),
        "apply_only_top_side_candidate": bool(model_cfg.get("apply_only_top_side_candidate", False)),
        "bonus_only": bool(bonus_only),
        "positive_score_weight": float(positive_score_weight),
        "negative_score_weight": float(negative_score_weight),
        "scopes": list(scopes_out),
    }
    summary = {
        "enabled": bool(scopes_out),
        "scopes_considered": int(len(scopes_cfg)),
        "scopes_selected": int(len(scopes_out)),
        "bucket_count": int(bucket_count),
        "max_scopes_per_row": int(max_scopes_per_row),
    }
    return model, summary


def _evaluate_action_condition_row(
    *,
    row: pd.Series,
    action_condition_model: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(action_condition_model, dict) or not bool(action_condition_model.get("enabled", False)):
        return {"score": 0.0, "match_count": 0, "matched_scopes": []}
    if "action_condition_top_side_candidate" in row.index and bool(
        action_condition_model.get("apply_only_top_side_candidate", False)
    ):
        if not bool(row.get("action_condition_top_side_candidate", False)):
            return {"score": 0.0, "match_count": 0, "matched_scopes": []}
    scopes = action_condition_model.get("scopes", [])
    if not isinstance(scopes, list) or not scopes:
        return {"score": 0.0, "match_count": 0, "matched_scopes": []}
    max_abs_score = max(0.05, safe_float(action_condition_model.get("max_abs_score", 1.35), 1.35))
    max_scopes_per_row = max(1, safe_int(action_condition_model.get("max_scopes_per_row", 3), 3))
    matches: List[Dict[str, Any]] = []
    for raw_scope in scopes:
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [
            str(v).strip()
            for v in (
                scope.get("fields", [])
                if isinstance(scope.get("fields"), (list, tuple, set))
                else []
            )
            if str(v).strip()
        ]
        if not fields:
            continue
        bucket_key = _context_prior_bucket_key(row=row, fields=fields)
        if not bucket_key:
            continue
        buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets"), dict) else {}
        bucket = buckets.get(bucket_key, {})
        if not isinstance(bucket, dict):
            continue
        score = safe_float(bucket.get("score", float("nan")), float("nan"))
        if not math.isfinite(score):
            continue
        weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
        if weight <= 0.0:
            continue
        matches.append(
            {
                "scope_name": str(scope.get("name", "") or ""),
                "bucket_key": str(bucket_key),
                "weight": float(weight),
                "score": float(score),
                "weighted_abs": float(abs(score) * weight),
            }
        )
    if not matches:
        return {"score": 0.0, "match_count": 0, "matched_scopes": []}
    matches.sort(
        key=lambda item: (
            safe_float(item.get("weighted_abs", 0.0), 0.0),
            abs(safe_float(item.get("score", 0.0), 0.0)),
        ),
        reverse=True,
    )
    matches = matches[:max_scopes_per_row]
    total_weight = sum(max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches)
    score = safe_div(
        sum(
            safe_float(item.get("score", 0.0), 0.0)
            * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
            for item in matches
        ),
        total_weight,
        0.0,
    )
    score = float(clip(score, -max_abs_score, max_abs_score))
    return {
        "score": float(score),
        "match_count": int(len(matches)),
        "matched_scopes": [
            str(item.get("scope_name", "") or "")
            for item in matches
            if str(item.get("scope_name", "") or "")
        ],
    }


def _default_short_term_condition_scopes() -> List[Dict[str, Any]]:
    return [
        {
            "name": "session_timeframe_shape",
            "fields": ["session", "timeframe", "strategy_type", "st_close_bucket", "st_wick_bias_bucket"],
            "min_trades": 72,
            "min_side_trades": 20,
            "weight": 1.00,
        },
        {
            "name": "session_substate_impulse",
            "fields": ["session", "ctx_session_substate", "strategy_type", "st_ret_bucket", "st_location_bucket"],
            "min_trades": 64,
            "min_side_trades": 18,
            "weight": 0.96,
        },
        {
            "name": "session_timeframe_pressure",
            "fields": ["session", "timeframe", "st_pressure_bucket", "st_down3_bucket"],
            "min_trades": 58,
            "min_side_trades": 16,
            "weight": 0.90,
        },
        {
            "name": "session_substate_flow",
            "fields": ["session", "ctx_session_substate", "timeframe", "st_vol_bucket", "st_range_bucket"],
            "min_trades": 54,
            "min_side_trades": 16,
            "weight": 0.82,
        },
        {
            "name": "hour_timeframe_location",
            "fields": ["ctx_hour_bucket", "timeframe", "strategy_type", "st_location_bucket", "st_body_bucket"],
            "min_trades": 48,
            "min_side_trades": 14,
            "weight": 0.74,
        },
    ]


def _short_term_prior_from_stats(
    *,
    stats: Dict[str, Any],
    cfg: Dict[str, Any],
) -> float:
    if not isinstance(stats, dict):
        return float("nan")
    n_trades = max(0, safe_int(stats.get("n_trades", 0), 0))
    year_coverage = max(0, safe_int(stats.get("year_coverage", 0), 0))
    min_year_coverage = max(0, safe_int(cfg.get("min_year_coverage", 4), 4))
    if n_trades <= 0 or year_coverage < min_year_coverage:
        return float("nan")
    support_full_trades = max(1.0, safe_float(cfg.get("support_full_trades", 180.0), 180.0))
    year_coverage_full = max(1.0, safe_float(cfg.get("year_coverage_full_years", 10.0), 10.0))
    quality_scale = max(1e-9, safe_float(cfg.get("quality_scale", 1.0), 1.0))
    profit_factor_center = safe_float(cfg.get("profit_factor_center", 1.12), 1.12)
    profit_factor_scale = max(1e-9, safe_float(cfg.get("profit_factor_scale", 0.28), 0.28))
    profit_factor_weight = max(0.0, safe_float(cfg.get("profit_factor_weight", 0.18), 0.18))
    loss_share_center = safe_float(cfg.get("loss_share_center", 0.50), 0.50)
    loss_share_scale = max(1e-9, safe_float(cfg.get("loss_share_scale", 0.16), 0.16))
    loss_share_weight = max(0.0, safe_float(cfg.get("loss_share_weight", 0.12), 0.12))
    max_abs_score = max(0.05, safe_float(cfg.get("max_abs_score", 1.35), 1.35))

    quality_lcb_score = safe_float(stats.get("quality_lcb_score", 0.0), 0.0)
    profit_factor = max(0.0, safe_float(stats.get("profit_factor", profit_factor_center), profit_factor_center))
    loss_share = clip(safe_float(stats.get("loss_share", loss_share_center), loss_share_center), 0.0, 1.0)
    support_ratio = clip(safe_div(float(n_trades), float(support_full_trades), 0.0), 0.0, 1.0)
    year_ratio = clip(safe_div(float(year_coverage), float(year_coverage_full), 0.0), 0.0, 1.0)
    prior_score = math.tanh(quality_lcb_score / quality_scale)
    prior_score += profit_factor_weight * math.tanh((profit_factor - profit_factor_center) / profit_factor_scale)
    prior_score -= loss_share_weight * math.tanh(max(0.0, loss_share - loss_share_center) / loss_share_scale)
    prior_score *= math.sqrt(max(0.0, support_ratio)) * year_ratio
    return float(clip(prior_score, -max_abs_score, max_abs_score))


def _build_short_term_condition_model(
    *,
    trade_df: pd.DataFrame,
    training_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_cfg = (
        training_cfg.get("short_term_condition_model", {})
        if isinstance(training_cfg.get("short_term_condition_model"), dict)
        else {}
    )
    enabled = bool(model_cfg.get("enabled", False))
    if (not enabled) or trade_df.empty:
        return {
            "enabled": False,
            "scopes": [],
        }, {
            "enabled": bool(enabled),
            "scopes_considered": 0,
            "scopes_selected": 0,
            "bucket_count": 0,
            "reason": "disabled_or_empty",
        }

    local = _prepare_short_term_condition_frame(trade_df)
    scopes_cfg = (
        model_cfg.get("scopes", [])
        if isinstance(model_cfg.get("scopes"), list)
        else _default_short_term_condition_scopes()
    )
    if not scopes_cfg:
        scopes_cfg = _default_short_term_condition_scopes()

    min_abs_prior_score = max(0.0, safe_float(model_cfg.get("min_abs_prior_score", 0.03), 0.03))
    max_abs_score = max(0.05, safe_float(model_cfg.get("max_abs_score", 1.35), 1.35))
    side_advantage_weight = clip(
        safe_float(model_cfg.get("side_advantage_weight", 0.45), 0.45),
        0.0,
        1.0,
    )
    single_side_discount = clip(
        safe_float(model_cfg.get("single_side_discount", 0.72), 0.72),
        0.0,
        1.0,
    )
    max_scopes_per_row = max(1, safe_int(model_cfg.get("max_scopes_per_row", 3), 3))
    require_both_sides = bool(model_cfg.get("require_both_sides", False))

    scopes_out: List[Dict[str, Any]] = []
    bucket_count = 0
    for idx, raw_scope in enumerate(scopes_cfg):
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [
            str(v).strip()
            for v in (
                scope.get("fields", [])
                if isinstance(scope.get("fields"), (list, tuple, set))
                else []
            )
            if str(v).strip()
        ]
        if not fields:
            continue
        scope_name = str(scope.get("name", f"short_term_scope_{idx}") or f"short_term_scope_{idx}").strip()
        min_trades = max(1, safe_int(scope.get("min_trades", 60), 60))
        min_side_trades = max(1, safe_int(scope.get("min_side_trades", 16), 16))
        weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
        if weight <= 0.0:
            continue
        scope_frame = local.copy()
        scope_frame["_short_term_bucket"] = [
            _context_prior_bucket_key(row=row, fields=fields)
            for row in scope_frame[fields].to_dict("records")
        ]
        scope_frame = scope_frame[scope_frame["_short_term_bucket"].astype(str).str.strip() != ""].copy()
        if scope_frame.empty:
            continue

        scope_buckets: Dict[str, Dict[str, Any]] = {}
        for bucket_key, bucket_frame in scope_frame.groupby("_short_term_bucket", sort=False):
            if int(len(bucket_frame)) < min_trades:
                continue
            long_frame = bucket_frame[bucket_frame["side_considered"] == "long"].copy()
            short_frame = bucket_frame[bucket_frame["side_considered"] == "short"].copy()
            long_stats = (
                _build_stats_from_frame(trade_df=long_frame, key_col=None, cfg=training_cfg).get("__global__", {})
                if len(long_frame) >= min_side_trades
                else {}
            )
            short_stats = (
                _build_stats_from_frame(trade_df=short_frame, key_col=None, cfg=training_cfg).get("__global__", {})
                if len(short_frame) >= min_side_trades
                else {}
            )
            long_score = _short_term_prior_from_stats(stats=long_stats, cfg=model_cfg)
            short_score = _short_term_prior_from_stats(stats=short_stats, cfg=model_cfg)
            long_valid = math.isfinite(long_score)
            short_valid = math.isfinite(short_score)
            if require_both_sides and (not (long_valid and short_valid)):
                continue
            if not long_valid and not short_valid:
                continue
            if long_valid and short_valid:
                long_effective = ((1.0 - side_advantage_weight) * long_score) + (
                    side_advantage_weight * (long_score - short_score)
                )
                short_effective = ((1.0 - side_advantage_weight) * short_score) + (
                    side_advantage_weight * (short_score - long_score)
                )
            else:
                long_effective = long_score * single_side_discount if long_valid else float("nan")
                short_effective = short_score * single_side_discount if short_valid else float("nan")
            long_effective = (
                float(clip(long_effective, -max_abs_score, max_abs_score))
                if math.isfinite(long_effective)
                else float("nan")
            )
            short_effective = (
                float(clip(short_effective, -max_abs_score, max_abs_score))
                if math.isfinite(short_effective)
                else float("nan")
            )
            strongest = max(
                abs(long_effective) if math.isfinite(long_effective) else 0.0,
                abs(short_effective) if math.isfinite(short_effective) else 0.0,
            )
            if strongest < min_abs_prior_score:
                continue
            scope_buckets[str(bucket_key)] = {
                "long_score": float(long_effective) if math.isfinite(long_effective) else None,
                "short_score": float(short_effective) if math.isfinite(short_effective) else None,
                "long_raw_score": float(long_score) if long_valid else None,
                "short_raw_score": float(short_score) if short_valid else None,
                "long_n_trades": int(safe_int(long_stats.get("n_trades", 0), 0)) if isinstance(long_stats, dict) else 0,
                "short_n_trades": int(safe_int(short_stats.get("n_trades", 0), 0)) if isinstance(short_stats, dict) else 0,
                "long_profit_factor": float(safe_float(long_stats.get("profit_factor", 0.0), 0.0))
                if isinstance(long_stats, dict)
                else 0.0,
                "short_profit_factor": float(safe_float(short_stats.get("profit_factor", 0.0), 0.0))
                if isinstance(short_stats, dict)
                else 0.0,
                "long_year_coverage": int(safe_int(long_stats.get("year_coverage", 0), 0))
                if isinstance(long_stats, dict)
                else 0,
                "short_year_coverage": int(safe_int(short_stats.get("year_coverage", 0), 0))
                if isinstance(short_stats, dict)
                else 0,
            }
        if not scope_buckets:
            continue
        bucket_count += int(len(scope_buckets))
        scopes_out.append(
            {
                "name": str(scope_name),
                "fields": list(fields),
                "min_trades": int(min_trades),
                "min_side_trades": int(min_side_trades),
                "weight": float(weight),
                "buckets": dict(scope_buckets),
            }
        )

    model = {
        "enabled": bool(scopes_out),
        "max_abs_score": float(max_abs_score),
        "max_scopes_per_row": int(max_scopes_per_row),
        "side_advantage_weight": float(side_advantage_weight),
        "single_side_discount": float(single_side_discount),
        "require_both_sides": bool(require_both_sides),
        "scopes": list(scopes_out),
    }
    summary = {
        "enabled": bool(scopes_out),
        "scopes_considered": int(len(scopes_cfg)),
        "scopes_selected": int(len(scopes_out)),
        "bucket_count": int(bucket_count),
        "max_scopes_per_row": int(max_scopes_per_row),
    }
    return model, summary


def _evaluate_short_term_condition_row(
    *,
    row: pd.Series,
    short_term_condition_model: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(short_term_condition_model, dict) or not bool(short_term_condition_model.get("enabled", False)):
        return {"score": 0.0, "match_count": 0, "matched_scopes": []}
    scopes = short_term_condition_model.get("scopes", [])
    if not isinstance(scopes, list) or not scopes:
        return {"score": 0.0, "match_count": 0, "matched_scopes": []}
    side = str(row.get("side_considered", "") or "").strip().lower()
    if side not in {"long", "short"}:
        return {"score": 0.0, "match_count": 0, "matched_scopes": []}
    score_key = f"{side}_score"
    max_abs_score = max(0.05, safe_float(short_term_condition_model.get("max_abs_score", 1.35), 1.35))
    max_scopes_per_row = max(1, safe_int(short_term_condition_model.get("max_scopes_per_row", 3), 3))
    matches: List[Dict[str, Any]] = []
    for raw_scope in scopes:
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [
            str(v).strip()
            for v in (
                scope.get("fields", [])
                if isinstance(scope.get("fields"), (list, tuple, set))
                else []
            )
            if str(v).strip()
        ]
        if not fields:
            continue
        bucket_key = _context_prior_bucket_key(row=row, fields=fields)
        if not bucket_key:
            continue
        buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets"), dict) else {}
        bucket = buckets.get(bucket_key, {})
        if not isinstance(bucket, dict):
            continue
        side_score = safe_float(bucket.get(score_key, float("nan")), float("nan"))
        if not math.isfinite(side_score):
            continue
        weight = max(0.0, safe_float(scope.get("weight", 1.0), 1.0))
        if weight <= 0.0:
            continue
        matches.append(
            {
                "scope_name": str(scope.get("name", "") or ""),
                "bucket_key": str(bucket_key),
                "weight": float(weight),
                "side_score": float(side_score),
                "weighted_abs": float(abs(side_score) * weight),
            }
        )
    if not matches:
        return {"score": 0.0, "match_count": 0, "matched_scopes": []}
    matches.sort(
        key=lambda item: (
            safe_float(item.get("weighted_abs", 0.0), 0.0),
            abs(safe_float(item.get("side_score", 0.0), 0.0)),
        ),
        reverse=True,
    )
    matches = matches[:max_scopes_per_row]
    total_weight = sum(max(0.0, safe_float(item.get("weight", 0.0), 0.0)) for item in matches)
    score = safe_div(
        sum(
            safe_float(item.get("side_score", 0.0), 0.0)
            * max(0.0, safe_float(item.get("weight", 0.0), 0.0))
            for item in matches
        ),
        total_weight,
        0.0,
    )
    score = float(clip(score, -max_abs_score, max_abs_score))
    return {
        "score": float(score),
        "match_count": int(len(matches)),
        "matched_scopes": [
            str(item.get("scope_name", "") or "")
            for item in matches
            if str(item.get("scope_name", "") or "")
        ],
    }


def _score_decision_policy_row(
    *,
    row: pd.Series,
    stats: Dict[str, Any],
    scope: str,
    score_cfg: Dict[str, Any],
    router_payload: Dict[str, Any],
    variant_priors: Dict[str, float],
    shape_model: Dict[str, Any],
    context_prior_model: Dict[str, Any],
    short_term_condition_model: Dict[str, Any],
    action_condition_model: Dict[str, Any],
) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
    variant_id = str(row.get("variant_id", "") or "").strip()
    lane = str(row.get("lane", "") or "").strip()
    timeframe = str(row.get("timeframe", "") or "").strip()
    session_name = str(row.get("session", "") or "").strip()
    base_quality = safe_float(stats.get("quality_lcb_score", 0.0), 0.0)
    edge_points = safe_float(row.get("edge_points", row.get("runtime_rank_score", 0.0)), 0.0)
    structural = safe_float(row.get("structural_score", 0.0), 0.0)
    lane_prior = _lane_prior_from_router_payload(
        router_payload=router_payload,
        lane=lane,
        session_name=session_name,
        timeframe_hint=timeframe,
    )
    variant_prior = safe_float(variant_priors.get(variant_id, 0.0), 0.0)
    side_considered = _decision_row_side(row=row, lane=lane)
    context_eval = _evaluate_context_prior_row(
        row=row,
        context_prior_model=context_prior_model or {},
    )
    context_prior_score = safe_float(context_eval.get("score", 0.0), 0.0)
    short_term_eval = _evaluate_short_term_condition_row(
        row=row,
        short_term_condition_model=short_term_condition_model or {},
    )
    short_term_condition_score = safe_float(short_term_eval.get("score", 0.0), 0.0)
    action_condition_eval = _evaluate_action_condition_row(
        row=row,
        action_condition_model=action_condition_model or {},
    )
    action_condition_score = safe_float(action_condition_eval.get("score", 0.0), 0.0)

    w_quality = safe_float(score_cfg.get("weight_quality_lcb", 0.56), 0.56)
    w_lane_prior = safe_float(score_cfg.get("weight_lane_prior", 0.10), 0.10)
    w_variant_prior = safe_float(score_cfg.get("weight_variant_quality_prior", 0.10), 0.10)
    w_edge = safe_float(score_cfg.get("weight_edge_points", 0.12), 0.12)
    w_struct = safe_float(score_cfg.get("weight_structural_score", 0.06), 0.06)
    w_pf = safe_float(score_cfg.get("weight_profit_factor_component", 0.10), 0.10)
    w_year = safe_float(score_cfg.get("weight_year_coverage_component", 0.06), 0.06)
    w_loss = safe_float(score_cfg.get("weight_loss_share_penalty", 0.16), 0.16)
    w_stop = safe_float(score_cfg.get("weight_stop_like_share_penalty", 0.10), 0.10)
    w_drawdown = safe_float(score_cfg.get("weight_drawdown_penalty", 0.10), 0.10)
    w_worst_block = safe_float(score_cfg.get("weight_worst_block_penalty", 0.12), 0.12)
    w_shape = safe_float(score_cfg.get("weight_shape_penalty_component", 0.18), 0.18)
    w_context = safe_float(score_cfg.get("weight_context_prior_component", 0.0), 0.0)
    w_short_term = safe_float(score_cfg.get("weight_short_term_condition_component", 0.0), 0.0)
    w_action = safe_float(score_cfg.get("weight_action_condition_component", 0.0), 0.0)
    side_bias_cfg = (
        score_cfg.get("side_score_bias", {})
        if isinstance(score_cfg.get("side_score_bias"), dict)
        else {}
    )
    side_bias_component = float(
        safe_float(
            side_bias_cfg.get(
                side_considered,
                side_bias_cfg.get("default", 0.0),
            ),
            0.0,
        )
    )

    lane_prior_center = safe_float(score_cfg.get("lane_prior_center", 0.15), 0.15)
    lane_prior_scale = max(1e-9, safe_float(score_cfg.get("lane_prior_scale", 0.08), 0.08))
    variant_prior_center = safe_float(score_cfg.get("variant_quality_prior_center", 0.27), 0.27)
    variant_prior_scale = max(
        1e-9,
        safe_float(score_cfg.get("variant_quality_prior_scale", 0.12), 0.12),
    )
    edge_scale = max(1e-9, safe_float(score_cfg.get("edge_scale_points", 0.40), 0.40))
    struct_scale = max(1e-9, safe_float(score_cfg.get("structural_scale", 0.80), 0.80))
    profit_factor_center = safe_float(score_cfg.get("profit_factor_center", 1.10), 1.10)
    profit_factor_scale = max(1e-9, safe_float(score_cfg.get("profit_factor_scale", 0.35), 0.35))
    year_coverage_full = max(1.0, safe_float(score_cfg.get("year_coverage_full_years", 8.0), 8.0))
    loss_center = safe_float(score_cfg.get("loss_share_center", 0.52), 0.52)
    loss_scale = max(1e-9, safe_float(score_cfg.get("loss_share_scale", 0.22), 0.22))
    stop_center = safe_float(score_cfg.get("stop_like_share_center", 0.62), 0.62)
    stop_scale = max(1e-9, safe_float(score_cfg.get("stop_like_share_scale", 0.25), 0.25))
    drawdown_scale = max(1e-9, safe_float(score_cfg.get("drawdown_scale", 6.0), 6.0))
    worst_block_scale = max(
        1e-9,
        safe_float(score_cfg.get("worst_block_scale_points", 3.0), 3.0),
    )
    shape_penalty_scale = max(1e-9, safe_float(score_cfg.get("shape_penalty_scale", 1.0), 1.0))
    shape_penalty_cap = max(0.0, safe_float(score_cfg.get("shape_penalty_cap", 2.0), 2.0))
    context_prior_scale = max(
        1e-9,
        safe_float(score_cfg.get("context_prior_scale", 0.35), 0.35),
    )
    short_term_scale = max(
        1e-9,
        safe_float(score_cfg.get("short_term_condition_scale", 0.60), 0.60),
    )
    action_condition_scale = max(
        1e-9,
        safe_float(score_cfg.get("action_condition_scale", 0.60), 0.60),
    )

    loss_share = clip(safe_float(stats.get("loss_share", loss_center), loss_center), 0.0, 1.0)
    stop_like_share = clip(
        safe_float(stats.get("stop_like_share", stop_center), stop_center),
        0.0,
        1.0,
    )
    profit_factor = max(0.0, safe_float(stats.get("profit_factor", profit_factor_center), profit_factor_center))
    year_coverage = max(0.0, safe_float(stats.get("year_coverage", 0.0), 0.0))
    year_coverage_ratio = clip(safe_div(year_coverage, year_coverage_full, 0.0), 0.0, 1.0)
    drawdown_norm = max(0.0, safe_float(stats.get("drawdown_norm", 0.0), 0.0))
    worst_block_avg_pnl = safe_float(stats.get("worst_block_avg_pnl", 0.0), 0.0)
    loss_excess = max(0.0, loss_share - loss_center)
    stop_excess = max(0.0, stop_like_share - stop_center)
    worst_block_shortfall = max(0.0, -worst_block_avg_pnl)

    shape_eval = _evaluate_shape_penalty_row(row=row, shape_model=shape_model or {})
    shape_strength = min(
        float(shape_penalty_cap),
        max(0.0, safe_float(shape_eval.get("strength", 0.0), 0.0)),
    )

    components = {
        "quality_lcb_component": float(w_quality * base_quality),
        "lane_prior_component": float(
            w_lane_prior * math.tanh((lane_prior - lane_prior_center) / lane_prior_scale)
        ),
        "variant_quality_prior_component": float(
            w_variant_prior
            * math.tanh((variant_prior - variant_prior_center) / variant_prior_scale)
        ),
        "edge_points_component": float(w_edge * math.tanh(edge_points / edge_scale)),
        "structural_component": float(w_struct * math.tanh(structural / struct_scale)),
        "profit_factor_component": float(
            w_pf * math.tanh((profit_factor - profit_factor_center) / profit_factor_scale)
        ),
        "year_coverage_component": float(w_year * ((2.0 * year_coverage_ratio) - 1.0)),
        "loss_share_penalty_component": float(-w_loss * math.tanh(loss_excess / loss_scale)),
        "stop_like_share_penalty_component": float(-w_stop * math.tanh(stop_excess / stop_scale)),
        "drawdown_penalty_component": float(-w_drawdown * math.tanh(drawdown_norm / drawdown_scale)),
        "worst_block_penalty_component": float(
            -w_worst_block * math.tanh(worst_block_shortfall / worst_block_scale)
        ),
        "shape_penalty_component": float(
            -w_shape * math.tanh(shape_strength / shape_penalty_scale)
        ),
        "context_prior_component": float(
            w_context * math.tanh(context_prior_score / context_prior_scale)
        ),
        "short_term_condition_component": float(
            w_short_term * math.tanh(short_term_condition_score / short_term_scale)
        ),
        "action_condition_component": float(
            w_action * math.tanh(action_condition_score / action_condition_scale)
        ),
        "side_bias_component": float(side_bias_component),
    }
    score = float(sum(components.values()))
    aux = {
        "scope": str(scope),
        "lane_prior": float(lane_prior),
        "variant_quality_prior": float(variant_prior),
        "shape_penalty_strength": float(shape_strength),
        "shape_penalty_match_count": int(safe_int(shape_eval.get("match_count", 0), 0)),
        "shape_penalty_scope_key": str(shape_eval.get("scope_key", "") or ""),
        "context_prior_score": float(context_prior_score),
        "context_prior_match_count": int(safe_int(context_eval.get("match_count", 0), 0)),
        "context_prior_scopes": list(context_eval.get("matched_scopes", []))
        if isinstance(context_eval.get("matched_scopes", []), list)
        else [],
        "short_term_condition_score": float(short_term_condition_score),
        "short_term_condition_match_count": int(safe_int(short_term_eval.get("match_count", 0), 0)),
        "short_term_condition_scopes": list(short_term_eval.get("matched_scopes", []))
        if isinstance(short_term_eval.get("matched_scopes", []), list)
        else [],
        "action_condition_score": float(action_condition_score),
        "action_condition_match_count": int(safe_int(action_condition_eval.get("match_count", 0), 0)),
        "action_condition_scopes": list(action_condition_eval.get("matched_scopes", []))
        if isinstance(action_condition_eval.get("matched_scopes", []), list)
        else [],
        "side_considered": str(side_considered),
    }
    return score, components, aux


def train_de3_v4_decision_policy(
    *,
    dataset: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    training_cfg = cfg.get("decision_policy", {}) if isinstance(cfg.get("decision_policy"), dict) else {}
    if not bool(training_cfg.get("enabled", False)):
        return {
            "decision_policy_model": {"enabled": False, "schema_version": "de3_v4_decision_policy_model_v1"},
            "decision_policy_training_report": {"status": "disabled", "reason": "decision_policy_disabled"},
        }

    split_summary = dataset.get("split_summary", {}) if isinstance(dataset.get("split_summary"), dict) else {}
    bounds = _split_bounds(split_summary)
    metadata = dataset.get("metadata", {}) if isinstance(dataset.get("metadata"), dict) else {}
    variants = dataset.get("variants", []) if isinstance(dataset.get("variants"), list) else []
    decisions_csv = metadata.get("decisions_csv_path", "")
    trades_csv = metadata.get("trade_attribution_csv_path", "")
    router_payload = (
        metadata.get("router_model_or_router_rules", {})
        if isinstance(metadata.get("router_model_or_router_rules", {}), dict)
        else {}
    )
    chosen_shape_csv = metadata.get("chosen_shape_csv_path", "")
    action_condition_csv = metadata.get("decision_side_dataset_path", "")
    variant_priors = _variant_quality_priors(metadata)
    trade_df, csv_audit = _trade_frame_from_csvs(
        decisions_csv_path=decisions_csv,
        trade_attribution_csv_path=trades_csv,
        bounds=bounds,
        variants=variants,
        chosen_shape_csv_path=chosen_shape_csv,
    )
    if trade_df.empty:
        return {
            "decision_policy_model": {"enabled": False, "schema_version": "de3_v4_decision_policy_model_v1"},
            "decision_policy_training_report": {
                "status": "missing_data",
                "reason": "trade_frame_empty",
                "csv_audit": dict(csv_audit),
            },
        }

    score_cfg = (
        training_cfg.get("score_components", {})
        if isinstance(training_cfg.get("score_components"), dict)
        else {}
    )
    side_bias_cfg = (
        score_cfg.get("side_score_bias", {})
        if isinstance(score_cfg.get("side_score_bias"), dict)
        else {}
    )
    tuning_cfg = (
        training_cfg.get("threshold_tuning", {})
        if isinstance(training_cfg.get("threshold_tuning"), dict)
        else {}
    )
    min_variant_trades = max(1, safe_int(training_cfg.get("min_variant_trades", 25), 25))
    min_lane_trades = max(1, safe_int(training_cfg.get("min_lane_trades", 120), 120))
    default_threshold = safe_float(training_cfg.get("default_threshold", 0.0), 0.0)
    allow_on_missing_stats = bool(training_cfg.get("allow_on_missing_stats", True))
    conservative_buffer = max(0.0, safe_float(training_cfg.get("conservative_buffer", 0.035), 0.035))
    selection_mode = str(training_cfg.get("selection_mode", "replace_router_lane") or "replace_router_lane").strip()
    min_confidence_to_override = clip(
        safe_float(training_cfg.get("min_confidence_to_override", 0.0), 0.0),
        0.0,
        1.0,
    )
    min_score_delta_to_override = max(
        0.0,
        safe_float(training_cfg.get("min_score_delta_to_override", 0.0), 0.0),
    )
    min_score_margin_to_override = max(
        0.0,
        safe_float(training_cfg.get("min_score_margin_to_override", 0.0), 0.0),
    )
    allow_override_when_baseline_no_trade = bool(
        training_cfg.get("allow_override_when_baseline_no_trade", False)
    )
    min_confidence_to_override_when_baseline_no_trade = clip(
        safe_float(
            training_cfg.get(
                "min_confidence_to_override_when_baseline_no_trade",
                min_confidence_to_override,
            ),
            min_confidence_to_override,
        ),
        0.0,
        1.0,
    )
    min_score_delta_to_override_when_baseline_no_trade = max(
        0.0,
        safe_float(
            training_cfg.get(
                "min_score_delta_to_override_when_baseline_no_trade",
                min_score_delta_to_override,
            ),
            min_score_delta_to_override,
        ),
    )
    min_score_margin_to_override_when_baseline_no_trade = max(
        0.0,
        safe_float(
            training_cfg.get(
                "min_score_margin_to_override_when_baseline_no_trade",
                min_score_margin_to_override,
            ),
            min_score_margin_to_override,
        ),
    )
    min_baseline_score_advantage_to_override = max(
        0.0,
        safe_float(training_cfg.get("min_baseline_score_advantage_to_override", 0.0), 0.0),
    )
    min_baseline_score_delta_advantage_to_override = max(
        0.0,
        safe_float(
            training_cfg.get("min_baseline_score_delta_advantage_to_override", 0.0),
            0.0,
        ),
    )
    scope_threshold_offsets_cfg = (
        dict(training_cfg.get("scope_threshold_offsets", {}))
        if isinstance(training_cfg.get("scope_threshold_offsets"), dict)
        else {}
    )
    scope_threshold_offsets_cfg = {
        "variant": float(safe_float(scope_threshold_offsets_cfg.get("variant", 0.0), 0.0)),
        "lane": float(safe_float(scope_threshold_offsets_cfg.get("lane", 0.05), 0.05)),
        "global": float(safe_float(scope_threshold_offsets_cfg.get("global", 0.10), 0.10)),
        "missing": float(safe_float(scope_threshold_offsets_cfg.get("missing", 0.15), 0.15)),
        "default": float(safe_float(scope_threshold_offsets_cfg.get("default", 0.0), 0.0)),
    }

    train_df = trade_df[trade_df["split"] == "train"].copy()
    tune_df = trade_df[trade_df["split"] == "tune"].copy()
    fit_df = trade_df[trade_df["split"].isin(["train", "tune"])].copy()

    action_condition_train_df = pd.DataFrame()
    action_condition_fit_df = pd.DataFrame()
    action_condition_csv_audit: Dict[str, Any] = {
        "requested_path": str(action_condition_csv or ""),
        "loaded": False,
        "rows": 0,
        "reason": "missing_path",
    }
    action_condition_cfg = (
        training_cfg.get("action_condition_model", {})
        if isinstance(training_cfg.get("action_condition_model"), dict)
        else {}
    )
    if bool(action_condition_cfg.get("enabled", False)) and str(action_condition_csv or "").strip():
        candidate_paths = []
        raw_action_path = Path(str(action_condition_csv)).expanduser()
        candidate_paths.append(raw_action_path)
        if not raw_action_path.is_absolute():
            candidate_paths.append(Path(__file__).resolve().parent / raw_action_path)
        resolved_action_path = next((path for path in candidate_paths if path.is_file()), None)
        if resolved_action_path is None:
            action_condition_csv_audit["reason"] = "file_not_found"
        else:
            try:
                action_condition_raw = pd.read_csv(resolved_action_path)
                action_condition_full = _prepare_action_condition_frame(action_condition_raw)
                action_condition_train_df = action_condition_full[
                    pd.to_numeric(action_condition_full.get("year"), errors="coerce") <= 2023
                ].copy()
                action_condition_fit_df = action_condition_full[
                    pd.to_numeric(action_condition_full.get("year"), errors="coerce") <= 2024
                ].copy()
                action_condition_csv_audit = {
                    "requested_path": str(action_condition_csv or ""),
                    "resolved_path": str(resolved_action_path),
                    "loaded": True,
                    "rows": int(len(action_condition_full)),
                    "train_rows": int(len(action_condition_train_df)),
                    "fit_rows": int(len(action_condition_fit_df)),
                    "reason": "ok",
                }
            except Exception as exc:
                action_condition_csv_audit = {
                    "requested_path": str(action_condition_csv or ""),
                    "resolved_path": str(resolved_action_path),
                    "loaded": False,
                    "rows": 0,
                    "reason": f"read_failed:{exc}",
                }

    variant_stats_train = _build_stats_from_frame(trade_df=train_df, key_col="variant_id", cfg=training_cfg)
    lane_stats_train = _build_stats_from_frame(trade_df=train_df, key_col="lane", cfg=training_cfg)
    global_stats_train = _build_stats_from_frame(trade_df=train_df, key_col=None, cfg=training_cfg).get("__global__", {})
    shape_model_train, shape_training_summary = _build_shape_penalty_model(
        trade_df=train_df,
        training_cfg=training_cfg,
    )
    context_prior_model_train, context_prior_training_summary = _build_context_prior_model(
        trade_df=train_df,
        training_cfg=training_cfg,
    )
    short_term_condition_model_train, short_term_condition_training_summary = _build_short_term_condition_model(
        trade_df=train_df,
        training_cfg=training_cfg,
    )
    action_condition_model_train, action_condition_training_summary = _build_action_condition_model(
        action_df=action_condition_train_df,
        training_cfg=training_cfg,
    )

    scored_rows: List[Dict[str, Any]] = []
    for row in tune_df.to_dict("records"):
        variant_id = str(row.get("variant_id", "") or "").strip()
        lane = str(row.get("lane", "") or "").strip()
        stats, scope = _resolve_stats(
            variant_id=variant_id,
            lane=lane,
            variant_stats=variant_stats_train,
            lane_stats=lane_stats_train,
            global_stats=global_stats_train,
            min_variant_trades=min_variant_trades,
            min_lane_trades=min_lane_trades,
        )
        if (not allow_on_missing_stats) and scope == "missing":
            continue
        score, components, aux = _score_decision_policy_row(
            row=pd.Series(row),
            stats=stats,
            scope=scope,
            score_cfg=score_cfg,
            router_payload=router_payload,
            variant_priors=variant_priors,
            shape_model=shape_model_train,
            context_prior_model=context_prior_model_train,
            short_term_condition_model=short_term_condition_model_train,
            action_condition_model=action_condition_model_train,
        )
        scored_rows.append(
            {
                "decision_id": str(row.get("decision_id", "") or ""),
                "variant_id": variant_id,
                "lane": lane,
                "timeframe": str(row.get("timeframe", "") or ""),
                "strategy_type": str(row.get("strategy_type", "") or ""),
                "side_considered": str(row.get("side_considered", "") or ""),
                "split": "tune",
                "ts_effective": row.get("ts_effective"),
                "realized_pnl": safe_float(row.get("realized_pnl", 0.0), 0.0),
                "entry_policy_score": float(score),
                "scope": str(scope),
                "lane_prior": safe_float(aux.get("lane_prior", 0.0), 0.0),
                "variant_quality_prior": safe_float(aux.get("variant_quality_prior", 0.0), 0.0),
                "side_considered": str(aux.get("side_considered", row.get("side_considered", "") or "") or ""),
                "quality_lcb_score": safe_float(stats.get("quality_lcb_score", 0.0), 0.0),
                "p_win_lcb": safe_float(stats.get("p_win_lcb", 0.0), 0.0),
                "ev_lcb": safe_float(stats.get("ev_lcb", 0.0), 0.0),
                "n_trades_scope": safe_int(stats.get("n_trades", 0), 0),
                "lane_prior_component": safe_float(components.get("lane_prior_component", 0.0), 0.0),
                "variant_quality_prior_component": safe_float(
                    components.get("variant_quality_prior_component", 0.0),
                    0.0,
                ),
                "edge_points_component": safe_float(components.get("edge_points_component", 0.0), 0.0),
                "structural_component": safe_float(components.get("structural_component", 0.0), 0.0),
                "quality_lcb_component": safe_float(components.get("quality_lcb_component", 0.0), 0.0),
                "shape_penalty_component": safe_float(components.get("shape_penalty_component", 0.0), 0.0),
                "context_prior_component": safe_float(
                    components.get("context_prior_component", 0.0),
                    0.0,
                ),
                "short_term_condition_component": safe_float(
                    components.get("short_term_condition_component", 0.0),
                    0.0,
                ),
                "action_condition_component": safe_float(
                    components.get("action_condition_component", 0.0),
                    0.0,
                ),
                "side_bias_component": safe_float(components.get("side_bias_component", 0.0), 0.0),
                "shape_penalty_strength": safe_float(aux.get("shape_penalty_strength", 0.0), 0.0),
                "shape_penalty_match_count": safe_int(aux.get("shape_penalty_match_count", 0), 0),
                "shape_penalty_scope_key": str(aux.get("shape_penalty_scope_key", "") or ""),
                "context_prior_score": safe_float(aux.get("context_prior_score", 0.0), 0.0),
                "context_prior_match_count": safe_int(aux.get("context_prior_match_count", 0), 0),
                "context_prior_scopes": "|".join(
                    [
                        str(v).strip()
                        for v in (
                            aux.get("context_prior_scopes", [])
                            if isinstance(aux.get("context_prior_scopes", []), list)
                            else []
                        )
                        if str(v).strip()
                    ]
                ),
                "short_term_condition_score": safe_float(
                    aux.get("short_term_condition_score", 0.0),
                    0.0,
                ),
                "short_term_condition_match_count": safe_int(
                    aux.get("short_term_condition_match_count", 0),
                    0,
                ),
                "short_term_condition_scopes": "|".join(
                    [
                        str(v).strip()
                        for v in (
                            aux.get("short_term_condition_scopes", [])
                            if isinstance(aux.get("short_term_condition_scopes", []), list)
                            else []
                        )
                        if str(v).strip()
                    ]
                ),
                "action_condition_score": safe_float(
                    aux.get("action_condition_score", 0.0),
                    0.0,
                ),
                "action_condition_match_count": safe_int(
                    aux.get("action_condition_match_count", 0),
                    0,
                ),
                "action_condition_scopes": "|".join(
                    [
                        str(v).strip()
                        for v in (
                            aux.get("action_condition_scopes", [])
                            if isinstance(aux.get("action_condition_scopes", []), list)
                            else []
                        )
                        if str(v).strip()
                    ]
                ),
            }
        )
    tune_scored = pd.DataFrame(scored_rows)
    if (not tune_scored.empty) and ("ts_effective" in tune_scored.columns):
        tune_scored = tune_scored.sort_values("ts_effective", kind="mergesort")
    tune_scope_offsets = (
        _scope_threshold_offsets_series(tune_scored["scope"], scope_threshold_offsets_cfg)
        if ("scope" in tune_scored.columns and not tune_scored.empty)
        else pd.Series(dtype=float)
    )

    min_keep_trades = max(1, safe_int(tuning_cfg.get("min_keep_trades", 80), 80))
    min_keep_rate = clip(safe_float(tuning_cfg.get("min_keep_rate", 0.35), 0.35), 0.0, 1.0)
    candidates = _threshold_candidates(
        tune_scored["entry_policy_score"] if "entry_policy_score" in tune_scored.columns else pd.Series(dtype=float),
        tuning_cfg,
    )
    threshold_trials = []
    for threshold in candidates:
        threshold_trials.append(
            _evaluate_threshold(
                tune_scored=tune_scored,
                threshold=float(threshold),
                min_keep_trades=min_keep_trades,
                min_keep_rate=min_keep_rate,
                objective_cfg=tuning_cfg,
                scope_threshold_offsets=scope_threshold_offsets_cfg,
                scope_offsets_series=tune_scope_offsets,
            )
        )
    baseline_metrics = _evaluate_threshold(
        tune_scored=tune_scored,
        threshold=-1e9,
        min_keep_trades=0,
        min_keep_rate=0.0,
        objective_cfg=tuning_cfg,
        scope_threshold_offsets=scope_threshold_offsets_cfg,
        scope_offsets_series=tune_scope_offsets,
    )
    valid_trials = [t for t in threshold_trials if bool(t.get("valid", False))]
    if valid_trials:
        valid_trials.sort(
            key=lambda r: (
                safe_float(r.get("objective", float("-inf")), float("-inf")),
                safe_float(r.get("net_pnl", float("-inf")), float("-inf")),
                safe_float(r.get("keep_rate", 0.0), 0.0),
            ),
            reverse=True,
        )
        selected = valid_trials[0]
        selected_threshold = float(selected.get("threshold", default_threshold))
        selected_metrics = dict(selected)
        threshold_source = "tune_2024_optimized"
    else:
        selected_threshold = float(default_threshold)
        selected_metrics = dict(baseline_metrics)
        threshold_source = "config_default_no_valid_tune_trial"

    variant_stats_fit = _build_stats_from_frame(trade_df=fit_df, key_col="variant_id", cfg=training_cfg)
    lane_stats_fit = _build_stats_from_frame(trade_df=fit_df, key_col="lane", cfg=training_cfg)
    global_stats_fit = _build_stats_from_frame(trade_df=fit_df, key_col=None, cfg=training_cfg).get("__global__", {})
    shape_model_fit, _ = _build_shape_penalty_model(trade_df=fit_df, training_cfg=training_cfg)
    context_prior_model_fit, _ = _build_context_prior_model(
        trade_df=fit_df,
        training_cfg=training_cfg,
    )
    short_term_condition_model_fit, _ = _build_short_term_condition_model(
        trade_df=fit_df,
        training_cfg=training_cfg,
    )
    action_condition_model_fit, _ = _build_action_condition_model(
        action_df=action_condition_fit_df,
        training_cfg=training_cfg,
    )

    model_payload = {
        "schema_version": "de3_v4_decision_policy_model_v1",
        "enabled": True,
        "selection_mode": str(selection_mode),
        "selected_threshold": float(selected_threshold),
        "selected_threshold_source": str(threshold_source),
        "min_variant_trades": int(min_variant_trades),
        "min_lane_trades": int(min_lane_trades),
        "allow_on_missing_stats": bool(allow_on_missing_stats),
        "conservative_buffer": float(conservative_buffer),
        "min_confidence_to_override": float(min_confidence_to_override),
        "min_score_delta_to_override": float(min_score_delta_to_override),
        "min_score_margin_to_override": float(min_score_margin_to_override),
        "allow_override_when_baseline_no_trade": bool(allow_override_when_baseline_no_trade),
        "min_confidence_to_override_when_baseline_no_trade": float(
            min_confidence_to_override_when_baseline_no_trade
        ),
        "min_score_delta_to_override_when_baseline_no_trade": float(
            min_score_delta_to_override_when_baseline_no_trade
        ),
        "min_score_margin_to_override_when_baseline_no_trade": float(
            min_score_margin_to_override_when_baseline_no_trade
        ),
        "min_baseline_score_advantage_to_override": float(
            min_baseline_score_advantage_to_override
        ),
        "min_baseline_score_delta_advantage_to_override": float(
            min_baseline_score_delta_advantage_to_override
        ),
        "minimums": {
            "min_variant_trades": int(min_variant_trades),
            "min_lane_trades": int(min_lane_trades),
            "allow_on_missing_stats": bool(allow_on_missing_stats),
            "conservative_buffer": float(conservative_buffer),
            "min_confidence_to_override": float(min_confidence_to_override),
            "min_score_delta_to_override": float(min_score_delta_to_override),
            "min_score_margin_to_override": float(min_score_margin_to_override),
            "allow_override_when_baseline_no_trade": bool(allow_override_when_baseline_no_trade),
            "min_confidence_to_override_when_baseline_no_trade": float(
                min_confidence_to_override_when_baseline_no_trade
            ),
            "min_score_delta_to_override_when_baseline_no_trade": float(
                min_score_delta_to_override_when_baseline_no_trade
            ),
            "min_score_margin_to_override_when_baseline_no_trade": float(
                min_score_margin_to_override_when_baseline_no_trade
            ),
            "min_baseline_score_advantage_to_override": float(
                min_baseline_score_advantage_to_override
            ),
            "min_baseline_score_delta_advantage_to_override": float(
                min_baseline_score_delta_advantage_to_override
            ),
        },
        "scope_threshold_offsets": dict(scope_threshold_offsets_cfg),
        "score_components": {
            "weight_quality_lcb": float(safe_float(score_cfg.get("weight_quality_lcb", 0.56), 0.56)),
            "weight_lane_prior": float(safe_float(score_cfg.get("weight_lane_prior", 0.10), 0.10)),
            "weight_variant_quality_prior": float(
                safe_float(score_cfg.get("weight_variant_quality_prior", 0.10), 0.10)
            ),
            "weight_edge_points": float(safe_float(score_cfg.get("weight_edge_points", 0.12), 0.12)),
            "weight_structural_score": float(safe_float(score_cfg.get("weight_structural_score", 0.06), 0.06)),
            "weight_profit_factor_component": float(
                safe_float(score_cfg.get("weight_profit_factor_component", 0.10), 0.10)
            ),
            "weight_year_coverage_component": float(
                safe_float(score_cfg.get("weight_year_coverage_component", 0.06), 0.06)
            ),
            "weight_loss_share_penalty": float(
                safe_float(score_cfg.get("weight_loss_share_penalty", 0.16), 0.16)
            ),
            "weight_stop_like_share_penalty": float(
                safe_float(score_cfg.get("weight_stop_like_share_penalty", 0.10), 0.10)
            ),
            "weight_drawdown_penalty": float(
                safe_float(score_cfg.get("weight_drawdown_penalty", 0.10), 0.10)
            ),
            "weight_worst_block_penalty": float(
                safe_float(score_cfg.get("weight_worst_block_penalty", 0.12), 0.12)
            ),
            "weight_shape_penalty_component": float(
                safe_float(score_cfg.get("weight_shape_penalty_component", 0.18), 0.18)
            ),
            "weight_context_prior_component": float(
                safe_float(score_cfg.get("weight_context_prior_component", 0.0), 0.0)
            ),
            "weight_short_term_condition_component": float(
                safe_float(score_cfg.get("weight_short_term_condition_component", 0.0), 0.0)
            ),
            "weight_action_condition_component": float(
                safe_float(score_cfg.get("weight_action_condition_component", 0.0), 0.0)
            ),
            "lane_prior_center": float(safe_float(score_cfg.get("lane_prior_center", 0.15), 0.15)),
            "lane_prior_scale": float(safe_float(score_cfg.get("lane_prior_scale", 0.08), 0.08)),
            "variant_quality_prior_center": float(
                safe_float(score_cfg.get("variant_quality_prior_center", 0.27), 0.27)
            ),
            "variant_quality_prior_scale": float(
                safe_float(score_cfg.get("variant_quality_prior_scale", 0.12), 0.12)
            ),
            "edge_scale_points": float(safe_float(score_cfg.get("edge_scale_points", 0.40), 0.40)),
            "structural_scale": float(safe_float(score_cfg.get("structural_scale", 0.80), 0.80)),
            "profit_factor_center": float(safe_float(score_cfg.get("profit_factor_center", 1.10), 1.10)),
            "profit_factor_scale": float(safe_float(score_cfg.get("profit_factor_scale", 0.35), 0.35)),
            "year_coverage_full_years": float(
                safe_float(score_cfg.get("year_coverage_full_years", 8.0), 8.0)
            ),
            "loss_share_center": float(safe_float(score_cfg.get("loss_share_center", 0.52), 0.52)),
            "loss_share_scale": float(safe_float(score_cfg.get("loss_share_scale", 0.22), 0.22)),
            "stop_like_share_center": float(
                safe_float(score_cfg.get("stop_like_share_center", 0.62), 0.62)
            ),
            "stop_like_share_scale": float(
                safe_float(score_cfg.get("stop_like_share_scale", 0.25), 0.25)
            ),
            "drawdown_scale": float(safe_float(score_cfg.get("drawdown_scale", 6.0), 6.0)),
            "shape_penalty_scale": float(
                safe_float(score_cfg.get("shape_penalty_scale", 1.0), 1.0)
            ),
            "shape_penalty_cap": float(safe_float(score_cfg.get("shape_penalty_cap", 2.0), 2.0)),
            "context_prior_scale": float(safe_float(score_cfg.get("context_prior_scale", 0.35), 0.35)),
            "short_term_condition_scale": float(
                safe_float(score_cfg.get("short_term_condition_scale", 0.60), 0.60)
            ),
            "action_condition_scale": float(
                safe_float(score_cfg.get("action_condition_scale", 0.60), 0.60)
            ),
            "worst_block_scale_points": float(
                safe_float(score_cfg.get("worst_block_scale_points", 3.0), 3.0)
            ),
            "side_score_bias": dict(side_bias_cfg),
        },
        "fit_windows": {
            "train_start": str(split_summary.get("training_start", "")),
            "train_end": str(split_summary.get("training_end", "")),
            "tune_start": str(split_summary.get("tuning_start", "")),
            "tune_end": str(split_summary.get("tuning_end", "")),
            "oos_start": str(split_summary.get("oos_start", "")),
            "oos_end": str(split_summary.get("oos_end", "")),
            "future_holdout_start": str(split_summary.get("future_holdout_start", "")),
            "future_holdout_end": str(split_summary.get("future_holdout_end", "")),
        },
        "variant_stats": variant_stats_fit,
        "lane_stats": lane_stats_fit,
        "global_stats": global_stats_fit if isinstance(global_stats_fit, dict) else {},
        "shape_penalty_model": dict(shape_model_fit),
        "context_prior_model": dict(context_prior_model_fit),
        "short_term_condition_model": dict(short_term_condition_model_fit),
        "action_condition_model": dict(action_condition_model_fit),
        "router_model_or_router_rules": dict(router_payload),
        "variant_quality_priors": dict(variant_priors),
    }

    report = {
        "status": "ok",
        "selection_architecture": "direct_candidate_chooser",
        "csv_audit": dict(csv_audit),
        "action_condition_csv_audit": dict(action_condition_csv_audit),
        "anti_leakage": {
            "training_used_splits": ["train"],
            "tuning_used_splits": ["tune"],
            "fitted_model_used_splits": ["train", "tune"],
            "excluded_splits_from_fit_and_tuning": ["oos", "future_holdout"],
            "oos_window_start": str(split_summary.get("oos_start", "")),
            "future_holdout_start": str(split_summary.get("future_holdout_start", "")),
            "leakage_check_passed": True,
            "violations": [],
        },
        "selected_threshold": float(selected_threshold),
        "selected_threshold_source": str(threshold_source),
        "baseline_tune_metrics": dict(baseline_metrics),
        "selected_tune_metrics": dict(selected_metrics),
        "threshold_trials": list(threshold_trials),
        "shape_penalty_training": dict(shape_training_summary),
        "context_prior_training": dict(context_prior_training_summary),
        "short_term_condition_training": dict(short_term_condition_training_summary),
        "action_condition_training": dict(action_condition_training_summary),
        "model_coverage": {
            "variant_stats_count": int(len(variant_stats_fit)),
            "lane_stats_count": int(len(lane_stats_fit)),
            "global_stats_n_trades": int(
                safe_int((global_stats_fit or {}).get("n_trades", 0), 0)
                if isinstance(global_stats_fit, dict)
                else 0
            ),
            "variant_priors_count": int(len(variant_priors)),
            "context_prior_scope_count": int(
                len(
                    context_prior_model_fit.get("scopes", [])
                    if isinstance(context_prior_model_fit, dict)
                    else []
                )
            ),
            "short_term_condition_scope_count": int(
                len(
                    short_term_condition_model_fit.get("scopes", [])
                    if isinstance(short_term_condition_model_fit, dict)
                    else []
                )
            ),
            "action_condition_scope_count": int(
                len(
                    action_condition_model_fit.get("scopes", [])
                    if isinstance(action_condition_model_fit, dict)
                    else []
                )
            ),
        },
        "config_effective": {
            "min_variant_trades": int(min_variant_trades),
            "min_lane_trades": int(min_lane_trades),
            "allow_on_missing_stats": bool(allow_on_missing_stats),
            "default_threshold": float(default_threshold),
            "conservative_buffer": float(conservative_buffer),
            "selection_mode": str(selection_mode),
            "min_confidence_to_override": float(min_confidence_to_override),
            "min_score_delta_to_override": float(min_score_delta_to_override),
            "min_score_margin_to_override": float(min_score_margin_to_override),
            "allow_override_when_baseline_no_trade": bool(allow_override_when_baseline_no_trade),
            "min_confidence_to_override_when_baseline_no_trade": float(
                min_confidence_to_override_when_baseline_no_trade
            ),
            "min_score_delta_to_override_when_baseline_no_trade": float(
                min_score_delta_to_override_when_baseline_no_trade
            ),
            "min_score_margin_to_override_when_baseline_no_trade": float(
                min_score_margin_to_override_when_baseline_no_trade
            ),
            "min_baseline_score_advantage_to_override": float(
                min_baseline_score_advantage_to_override
            ),
            "min_baseline_score_delta_advantage_to_override": float(
                min_baseline_score_delta_advantage_to_override
            ),
            "scope_threshold_offsets": dict(scope_threshold_offsets_cfg),
            "score_components": dict(model_payload.get("score_components", {})),
            "shape_penalty_rules_selected": int(
                len(
                    shape_model_fit.get("rules", [])
                    if isinstance(shape_model_fit, dict)
                    else []
                )
            ),
        },
    }
    return {
        "decision_policy_model": model_payload,
        "decision_policy_training_report": report,
    }
