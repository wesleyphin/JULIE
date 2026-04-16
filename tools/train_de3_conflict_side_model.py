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


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"File not found: {path_arg}")


def _normalize_value(raw: Any) -> str:
    if raw is None:
        return ""
    try:
        num = float(raw)
        if math.isfinite(num):
            rounded = round(num)
            if abs(num - rounded) <= 1e-9:
                return str(int(rounded))
            return f"{num:.4f}".rstrip("0").rstrip(".")
    except Exception:
        pass
    text = str(raw).strip().lower()
    return "" if text in {"", "nan", "nat", "none"} else text


def _bucket_key(row: Dict[str, Any], fields: List[str]) -> str:
    parts: List[str] = []
    for field in fields:
        value = _normalize_value(row.get(field))
        if not value:
            return ""
        parts.append(value)
    return "|".join(parts)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _derive_session_substate(*, session_text: str, timestamp_value: Any) -> str:
    ts = pd.to_datetime(timestamp_value, errors="coerce")
    if pd.isna(ts):
        return ""
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


def _direction_bucket(value: float, *, neg_hi: float, neg_lo: float, pos_lo: float, pos_hi: float) -> str:
    if not math.isfinite(value):
        return ""
    if value <= neg_hi:
        return "strong_short"
    if value <= neg_lo:
        return "short"
    if value < pos_lo:
        return "balanced"
    if value < pos_hi:
        return "long"
    return "strong_long"


def _prepare_conflict_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame[frame["timestamp"].notna()].copy()
    frame["timestamp"] = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["year"] = frame["timestamp"].dt.year
    frame["session"] = frame["session"].astype(str).str.strip()

    numeric_cols = [
        "ctx_hour_et",
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
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "ctx_hour_et" not in frame.columns:
        frame["ctx_hour_et"] = pd.NA
    frame["ctx_hour_et"] = frame["ctx_hour_et"].where(frame["ctx_hour_et"].notna(), frame["timestamp"].dt.hour)
    if "ctx_session_substate" not in frame.columns:
        frame["ctx_session_substate"] = ""
    frame["ctx_session_substate"] = frame["ctx_session_substate"].astype(str).replace({"nan": "", "None": ""})
    missing_substate = frame["ctx_session_substate"].str.strip() == ""
    frame.loc[missing_substate, "ctx_session_substate"] = [
        _derive_session_substate(session_text=row["session"], timestamp_value=row["timestamp"])
        for _, row in frame.loc[missing_substate, ["session", "timestamp"]].iterrows()
    ]
    frame["ctx_hour_bucket"] = frame["ctx_hour_et"].apply(
        lambda raw: str(int(round(float(raw)))) if pd.notna(raw) else ""
    )
    for col in ("long_strategy_type", "short_strategy_type", "long_timeframe", "short_timeframe"):
        frame[col] = frame[col].astype(str).str.strip()

    frame["rank_adv"] = frame["short_rank"].fillna(0.0) - frame["long_rank"].fillna(0.0)
    frame["score_adv"] = frame["long_final_score"].fillna(0.0) - frame["short_final_score"].fillna(0.0)
    frame["edge_adv"] = frame["long_edge_points"].fillna(0.0) - frame["short_edge_points"].fillna(0.0)
    frame["struct_adv"] = (
        frame["long_structural_score"].fillna(0.0) - frame["short_structural_score"].fillna(0.0)
    )
    frame["rank_adv_bucket"] = frame["rank_adv"].apply(
        lambda v: _direction_bucket(float(v), neg_hi=-1.5, neg_lo=-0.5, pos_lo=0.5, pos_hi=1.5)
    )
    frame["score_adv_bucket"] = frame["score_adv"].apply(
        lambda v: _direction_bucket(float(v), neg_hi=-1.0, neg_lo=-0.25, pos_lo=0.25, pos_hi=1.0)
    )
    frame["edge_adv_bucket"] = frame["edge_adv"].apply(
        lambda v: _direction_bucket(float(v), neg_hi=-1.0, neg_lo=-0.25, pos_lo=0.25, pos_hi=1.0)
    )
    frame["struct_adv_bucket"] = frame["struct_adv"].apply(
        lambda v: _direction_bucket(float(v), neg_hi=-1.0, neg_lo=-0.25, pos_lo=0.25, pos_hi=1.0)
    )
    frame["score_edge_combo"] = (
        frame["score_adv_bucket"].astype(str).str.strip() + "|" + frame["edge_adv_bucket"].astype(str).str.strip()
    )
    frame["rank_score_combo"] = (
        frame["rank_adv_bucket"].astype(str).str.strip() + "|" + frame["score_adv_bucket"].astype(str).str.strip()
    )
    return frame


def _conflict_profiles() -> Dict[str, Dict[str, Any]]:
    return {
        "conflict_side_time_context": {
            "support_full_decisions": 120,
            "year_coverage_full_years": 10.0,
            "min_year_coverage": 4,
            "advantage_scale_points": 8.0,
            "max_abs_score": 1.4,
            "min_abs_score": 0.04,
            "max_scopes_per_row": 3,
            "min_side_mean_points": 0.25,
            "min_advantage_gap_points": 0.20,
            "scope_threshold_candidates": [0.08, 0.12, 0.18, 0.24, 0.30, 0.38],
            "objective_weight_max_drawdown": 0.10,
            "min_override_count": 35,
            "scopes": [
                {
                    "name": "session_pair",
                    "fields": ["session", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 60,
                    "weight": 0.82,
                },
                {
                    "name": "session_substate_pair",
                    "fields": ["session", "ctx_session_substate", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 42,
                    "weight": 0.94,
                },
                {
                    "name": "session_hour_pair",
                    "fields": ["session", "ctx_hour_bucket", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 44,
                    "weight": 1.00,
                },
                {
                    "name": "session_hour_timeframe_pair",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_timeframe",
                        "short_timeframe",
                        "long_strategy_type",
                        "short_strategy_type",
                    ],
                    "min_decisions": 38,
                    "weight": 0.90,
                },
                {
                    "name": "session_hour_score_pair",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "score_adv_bucket",
                    ],
                    "min_decisions": 34,
                    "weight": 0.92,
                },
                {
                    "name": "session_hour_edge_pair",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "edge_adv_bucket",
                    ],
                    "min_decisions": 34,
                    "weight": 0.88,
                },
            ],
        },
        "conflict_side_time_context_guarded": {
            "support_full_decisions": 130,
            "year_coverage_full_years": 10.0,
            "min_year_coverage": 4,
            "advantage_scale_points": 8.0,
            "max_abs_score": 1.25,
            "min_abs_score": 0.05,
            "max_scopes_per_row": 2,
            "min_side_mean_points": 0.35,
            "min_advantage_gap_points": 0.45,
            "scope_threshold_candidates": [0.18, 0.24, 0.30, 0.36, 0.44, 0.52],
            "objective_weight_max_drawdown": 0.14,
            "min_override_count": 22,
            "scopes": [
                {
                    "name": "session_pair",
                    "fields": ["session", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 65,
                    "weight": 0.84,
                },
                {
                    "name": "session_substate_pair",
                    "fields": ["session", "ctx_session_substate", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 46,
                    "weight": 0.90,
                },
                {
                    "name": "session_hour_pair",
                    "fields": ["session", "ctx_hour_bucket", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 44,
                    "weight": 1.00,
                },
                {
                    "name": "session_hour_score_edge",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "score_edge_combo",
                    ],
                    "min_decisions": 28,
                    "weight": 0.96,
                },
                {
                    "name": "session_hour_rank_score",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "rank_score_combo",
                    ],
                    "min_decisions": 28,
                    "weight": 0.90,
                },
            ],
        },
        "conflict_side_time_context_notrade": {
            "support_full_decisions": 125,
            "year_coverage_full_years": 10.0,
            "min_year_coverage": 4,
            "advantage_scale_points": 8.0,
            "max_abs_score": 1.15,
            "min_abs_score": 0.05,
            "max_scopes_per_row": 2,
            "min_side_mean_points": 0.28,
            "min_advantage_gap_points": 0.30,
            "allow_no_trade": True,
            "no_trade_max_side_mean_points": 0.05,
            "no_trade_max_positive_rate": 0.47,
            "scope_threshold_candidates": [0.14, 0.20, 0.26, 0.32, 0.40],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 24,
            "scopes": [
                {
                    "name": "session_substate_pair",
                    "fields": ["session", "ctx_session_substate", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 40,
                    "weight": 0.88,
                },
                {
                    "name": "session_hour_pair",
                    "fields": ["session", "ctx_hour_bucket", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 40,
                    "weight": 0.96,
                },
                {
                    "name": "session_hour_score_edge",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "score_edge_combo",
                    ],
                    "min_decisions": 28,
                    "weight": 1.00,
                },
                {
                    "name": "session_hour_rank_score",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "rank_score_combo",
                    ],
                    "min_decisions": 28,
                    "weight": 0.92,
                },
            ],
        },
        "conflict_side_time_context_consensus": {
            "support_full_decisions": 130,
            "year_coverage_full_years": 10.0,
            "min_year_coverage": 4,
            "advantage_scale_points": 8.0,
            "max_abs_score": 1.2,
            "min_abs_score": 0.05,
            "max_scopes_per_row": 3,
            "min_side_mean_points": 0.30,
            "min_advantage_gap_points": 0.35,
            "min_match_count_to_override": 2,
            "min_consensus_ratio": 1.0,
            "scope_threshold_candidates": [0.12, 0.18, 0.22, 0.26, 0.32],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 18,
            "scopes": [
                {
                    "name": "session_hour_pair",
                    "fields": ["session", "ctx_hour_bucket", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 44,
                    "weight": 0.96,
                },
                {
                    "name": "session_hour_timeframe_pair",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_timeframe",
                        "short_timeframe",
                        "long_strategy_type",
                        "short_strategy_type",
                    ],
                    "min_decisions": 38,
                    "weight": 0.94,
                },
                {
                    "name": "session_hour_score_pair",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "score_adv_bucket",
                    ],
                    "min_decisions": 30,
                    "weight": 1.00,
                },
                {
                    "name": "session_hour_edge_pair",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "edge_adv_bucket",
                    ],
                    "min_decisions": 30,
                    "weight": 0.96,
                },
            ],
        },
        "conflict_side_time_context_consistent": {
            "support_full_decisions": 130,
            "year_coverage_full_years": 10.0,
            "min_year_coverage": 4,
            "min_year_direction_consistency": 0.62,
            "direction_consistency_power": 1.0,
            "advantage_scale_points": 8.0,
            "max_abs_score": 1.2,
            "min_abs_score": 0.05,
            "max_scopes_per_row": 3,
            "min_side_mean_points": 0.28,
            "min_advantage_gap_points": 0.30,
            "min_match_count_to_override": 2,
            "min_consensus_ratio": 1.0,
            "scope_threshold_candidates": [0.10, 0.14, 0.18, 0.22, 0.26],
            "objective_weight_max_drawdown": 0.12,
            "min_override_count": 15,
            "scopes": [
                {
                    "name": "session_hour_pair",
                    "fields": ["session", "ctx_hour_bucket", "long_strategy_type", "short_strategy_type"],
                    "min_decisions": 44,
                    "weight": 0.96,
                },
                {
                    "name": "session_hour_timeframe_pair",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_timeframe",
                        "short_timeframe",
                        "long_strategy_type",
                        "short_strategy_type",
                    ],
                    "min_decisions": 38,
                    "weight": 0.94,
                },
                {
                    "name": "session_hour_score_pair",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "score_adv_bucket",
                    ],
                    "min_decisions": 30,
                    "weight": 1.0,
                },
                {
                    "name": "session_hour_edge_pair",
                    "fields": [
                        "session",
                        "ctx_hour_bucket",
                        "long_strategy_type",
                        "short_strategy_type",
                        "edge_adv_bucket",
                    ],
                    "min_decisions": 30,
                    "weight": 0.96,
                },
            ],
        },
    }


def _build_model(
    *,
    train_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    support_full = max(1, int(cfg.get("support_full_decisions", 160) or 160))
    year_full = max(1.0, float(cfg.get("year_coverage_full_years", 10.0) or 10.0))
    min_years = max(1, int(cfg.get("min_year_coverage", 4) or 4))
    min_direction_consistency = float(cfg.get("min_year_direction_consistency", 0.0) or 0.0)
    direction_consistency_power = max(0.0, float(cfg.get("direction_consistency_power", 1.0) or 1.0))
    advantage_scale = max(1e-6, float(cfg.get("advantage_scale_points", 8.0) or 8.0))
    max_abs_score = max(0.10, float(cfg.get("max_abs_score", 1.4) or 1.4))
    min_abs_score = max(0.0, float(cfg.get("min_abs_score", 0.04) or 0.04))
    max_scopes_per_row = max(1, int(cfg.get("max_scopes_per_row", 3) or 3))
    scopes_cfg = list(cfg.get("scopes", [])) if isinstance(cfg.get("scopes", []), list) else []

    local = train_df.copy()
    local["year"] = pd.to_datetime(local["timestamp"], errors="coerce").dt.year
    scopes_out: List[Dict[str, Any]] = []
    bucket_total = 0
    for idx, raw_scope in enumerate(scopes_cfg):
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
        if not fields:
            continue
        scope_name = str(scope.get("name", f"scope_{idx}") or f"scope_{idx}")
        min_decisions = max(1, int(scope.get("min_decisions", 60) or 60))
        weight = max(0.0, float(scope.get("weight", 1.0) or 1.0))
        if weight <= 0.0:
            continue
        work = local.copy()
        work["_bucket_key"] = [_bucket_key(row, fields) for row in work[fields].to_dict("records")]
        work = work[work["_bucket_key"].astype(str).str.strip() != ""].copy()
        if work.empty:
            continue
        buckets: Dict[str, Dict[str, Any]] = {}
        for bucket_key, grp in work.groupby("_bucket_key", dropna=False):
            n_decisions = int(len(grp))
            year_coverage = int(pd.to_numeric(grp["year"], errors="coerce").dropna().astype(int).nunique())
            if n_decisions < min_decisions or year_coverage < min_years:
                continue
            long_mean = float(pd.to_numeric(grp["long_pnl_points"], errors="coerce").fillna(0.0).mean())
            short_mean = float(pd.to_numeric(grp["short_pnl_points"], errors="coerce").fillna(0.0).mean())
            advantage_mean = float(long_mean - short_mean)
            long_best_rate = float((grp["best_side"].astype(str) == "long").mean())
            short_best_rate = float((grp["best_side"].astype(str) == "short").mean())
            long_positive_rate = float(
                (pd.to_numeric(grp["long_pnl_points"], errors="coerce").fillna(0.0) > 0.0).mean()
            )
            short_positive_rate = float(
                (pd.to_numeric(grp["short_pnl_points"], errors="coerce").fillna(0.0) > 0.0).mean()
            )
            yearly_advantage = (
                grp.groupby("year", dropna=False)
                .apply(
                    lambda g: float(
                        pd.to_numeric(g["long_pnl_points"], errors="coerce").fillna(0.0).mean()
                        - pd.to_numeric(g["short_pnl_points"], errors="coerce").fillna(0.0).mean()
                    )
                )
                .tolist()
            )
            overall_sign = 1 if advantage_mean > 0.0 else (-1 if advantage_mean < 0.0 else 0)
            if overall_sign == 0:
                direction_consistency = 0.0
            else:
                direction_hits = 0
                direction_total = 0
                for raw_adv in yearly_advantage:
                    try:
                        year_adv = float(raw_adv)
                    except Exception:
                        continue
                    if not math.isfinite(year_adv):
                        continue
                    year_sign = 1 if year_adv > 0.0 else (-1 if year_adv < 0.0 else 0)
                    direction_total += 1
                    if year_sign == overall_sign:
                        direction_hits += 1
                direction_consistency = float(direction_hits / max(1, direction_total))
            if direction_consistency < min_direction_consistency:
                continue
            support_ratio = _clip(n_decisions / float(support_full), 0.0, 1.0)
            year_ratio = _clip(year_coverage / float(year_full), 0.0, 1.0)
            score = (
                (0.55 * math.tanh(advantage_mean / advantage_scale))
                + (0.30 * float(long_best_rate - short_best_rate))
                + (0.15 * float(long_positive_rate - short_positive_rate))
            )
            score *= math.sqrt(max(0.0, support_ratio)) * year_ratio * (
                float(direction_consistency) ** float(direction_consistency_power)
            )
            score = float(_clip(score, -max_abs_score, max_abs_score))
            if abs(score) < min_abs_score:
                continue
            buckets[str(bucket_key)] = {
                "score": float(score),
                "n_decisions": int(n_decisions),
                "year_coverage": int(year_coverage),
                "long_mean_pnl_points": float(long_mean),
                "short_mean_pnl_points": float(short_mean),
                "advantage_mean_points": float(advantage_mean),
                "long_best_rate": float(long_best_rate),
                "short_best_rate": float(short_best_rate),
                "long_positive_rate": float(long_positive_rate),
                "short_positive_rate": float(short_positive_rate),
                "direction_consistency": float(direction_consistency),
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
        "schema_version": "de3_v4_conflict_side_model_v1",
        "support_full_decisions": int(support_full),
        "year_coverage_full_years": float(year_full),
        "min_year_coverage": int(min_years),
        "advantage_scale_points": float(advantage_scale),
        "max_abs_score": float(max_abs_score),
        "max_scopes_per_row": int(max_scopes_per_row),
        "min_side_mean_points": float(cfg.get("min_side_mean_points", 0.25) or 0.25),
        "min_advantage_gap_points": float(cfg.get("min_advantage_gap_points", 0.0) or 0.0),
        "allow_no_trade": bool(cfg.get("allow_no_trade", False)),
        "no_trade_max_side_mean_points": float(cfg.get("no_trade_max_side_mean_points", 0.0) or 0.0),
        "no_trade_max_positive_rate": float(cfg.get("no_trade_max_positive_rate", 0.0) or 0.0),
        "min_match_count_to_override": int(cfg.get("min_match_count_to_override", 1) or 1),
        "min_consensus_ratio": float(cfg.get("min_consensus_ratio", 0.0) or 0.0),
        "scopes": list(scopes_out),
    }
    summary = {
        "enabled": bool(scopes_out),
        "scopes_selected": int(len(scopes_out)),
        "bucket_count": int(bucket_total),
        "max_scopes_per_row": int(max_scopes_per_row),
    }
    return model, summary


def _evaluate_row(
    *,
    row: Dict[str, Any],
    model: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(model, dict) or not bool(model.get("enabled", False)):
        return {
            "score": 0.0,
            "match_count": 0,
            "predicted_side": "",
            "long_mean": 0.0,
            "short_mean": 0.0,
            "long_best_rate": 0.0,
            "short_best_rate": 0.0,
            "long_positive_rate": 0.0,
            "short_positive_rate": 0.0,
            "positive_match_count": 0,
            "negative_match_count": 0,
            "consensus_ratio": 0.0,
        }
    max_abs_score = max(0.10, float(model.get("max_abs_score", 1.4) or 1.4))
    max_scopes_per_row = max(1, int(model.get("max_scopes_per_row", 3) or 3))
    matches: List[Dict[str, Any]] = []
    for raw_scope in model.get("scopes", []):
        scope = raw_scope if isinstance(raw_scope, dict) else {}
        fields = [str(v).strip() for v in scope.get("fields", []) if str(v).strip()]
        if not fields:
            continue
        bucket_key = _bucket_key(row, fields)
        if not bucket_key:
            continue
        buckets = scope.get("buckets", {}) if isinstance(scope.get("buckets", {}), dict) else {}
        bucket = buckets.get(bucket_key, {})
        if not isinstance(bucket, dict):
            continue
        weight = max(0.0, float(scope.get("weight", 1.0) or 1.0))
        if weight <= 0.0:
            continue
        score = float(bucket.get("score", 0.0) or 0.0)
        matches.append(
            {
                "scope_name": str(scope.get("name", "") or ""),
                "weight": float(weight),
                "score": float(score),
                "weighted_abs": float(abs(score) * weight),
                "long_mean": float(bucket.get("long_mean_pnl_points", 0.0) or 0.0),
                "short_mean": float(bucket.get("short_mean_pnl_points", 0.0) or 0.0),
                "long_best_rate": float(bucket.get("long_best_rate", 0.0) or 0.0),
                "short_best_rate": float(bucket.get("short_best_rate", 0.0) or 0.0),
                "long_positive_rate": float(bucket.get("long_positive_rate", 0.0) or 0.0),
                "short_positive_rate": float(bucket.get("short_positive_rate", 0.0) or 0.0),
            }
        )
    if not matches:
        return {
            "score": 0.0,
            "match_count": 0,
            "predicted_side": "",
            "long_mean": 0.0,
            "short_mean": 0.0,
            "long_best_rate": 0.0,
            "short_best_rate": 0.0,
            "long_positive_rate": 0.0,
            "short_positive_rate": 0.0,
            "positive_match_count": 0,
            "negative_match_count": 0,
            "consensus_ratio": 0.0,
        }
    matches.sort(key=lambda item: (item["weighted_abs"], abs(item["score"])), reverse=True)
    matches = matches[:max_scopes_per_row]
    total_weight = sum(max(0.0, float(item["weight"])) for item in matches)
    score = sum(float(item["score"]) * float(item["weight"]) for item in matches) / max(total_weight, 1e-9)
    long_mean = sum(float(item["long_mean"]) * float(item["weight"]) for item in matches) / max(total_weight, 1e-9)
    short_mean = sum(float(item["short_mean"]) * float(item["weight"]) for item in matches) / max(total_weight, 1e-9)
    long_best_rate = sum(float(item["long_best_rate"]) * float(item["weight"]) for item in matches) / max(
        total_weight, 1e-9
    )
    short_best_rate = sum(float(item["short_best_rate"]) * float(item["weight"]) for item in matches) / max(
        total_weight, 1e-9
    )
    long_positive_rate = sum(
        float(item["long_positive_rate"]) * float(item["weight"]) for item in matches
    ) / max(total_weight, 1e-9)
    short_positive_rate = sum(
        float(item["short_positive_rate"]) * float(item["weight"]) for item in matches
    ) / max(total_weight, 1e-9)
    positive_match_count = sum(1 for item in matches if float(item.get("score", 0.0) or 0.0) > 0.0)
    negative_match_count = sum(1 for item in matches if float(item.get("score", 0.0) or 0.0) < 0.0)
    consensus_ratio = (
        float(max(positive_match_count, negative_match_count) / max(1, len(matches)))
        if matches
        else 0.0
    )
    score = float(_clip(score, -max_abs_score, max_abs_score))
    return {
        "score": float(score),
        "match_count": int(len(matches)),
        "matched_scopes": [str(item["scope_name"]) for item in matches if str(item["scope_name"])],
        "long_mean": float(long_mean),
        "short_mean": float(short_mean),
        "long_best_rate": float(long_best_rate),
        "short_best_rate": float(short_best_rate),
        "long_positive_rate": float(long_positive_rate),
        "short_positive_rate": float(short_positive_rate),
        "positive_match_count": int(positive_match_count),
        "negative_match_count": int(negative_match_count),
        "consensus_ratio": float(consensus_ratio),
    }


def _baseline_points(row: Dict[str, Any]) -> float:
    chosen_side = str(row.get("chosen_side", "") or "").strip().lower()
    if chosen_side == "long":
        return float(row.get("long_pnl_points", 0.0) or 0.0)
    if chosen_side == "short":
        return float(row.get("short_pnl_points", 0.0) or 0.0)
    return 0.0


def _evaluate_threshold(
    *,
    tune_df: pd.DataFrame,
    model: Dict[str, Any],
    threshold: float,
) -> Dict[str, Any]:
    min_side_mean = float(model.get("min_side_mean_points", 0.25) or 0.25)
    min_advantage_gap = float(model.get("min_advantage_gap_points", 0.0) or 0.0)
    allow_no_trade = bool(model.get("allow_no_trade", False))
    no_trade_max_side_mean = float(model.get("no_trade_max_side_mean_points", 0.0) or 0.0)
    no_trade_max_positive_rate = float(model.get("no_trade_max_positive_rate", 0.0) or 0.0)
    min_match_count = max(1, int(model.get("min_match_count_to_override", 1) or 1))
    min_consensus_ratio = float(model.get("min_consensus_ratio", 0.0) or 0.0)
    rows = []
    for row in tune_df.to_dict("records"):
        eval_row = _evaluate_row(row=row, model=model)
        predicted_side = ""
        score = float(eval_row.get("score", 0.0) or 0.0)
        long_mean = float(eval_row.get("long_mean", 0.0) or 0.0)
        short_mean = float(eval_row.get("short_mean", 0.0) or 0.0)
        long_positive_rate = float(eval_row.get("long_positive_rate", 0.0) or 0.0)
        short_positive_rate = float(eval_row.get("short_positive_rate", 0.0) or 0.0)
        match_count = int(eval_row.get("match_count", 0) or 0)
        consensus_ratio = float(eval_row.get("consensus_ratio", 0.0) or 0.0)
        if (
            allow_no_trade
            and abs(score) >= threshold
            and max(long_mean, short_mean) <= no_trade_max_side_mean
            and max(long_positive_rate, short_positive_rate) <= no_trade_max_positive_rate
            and match_count >= min_match_count
            and consensus_ratio >= min_consensus_ratio
        ):
            predicted_side = "no_trade"
        elif (
            match_count >= min_match_count
            and consensus_ratio >= min_consensus_ratio
            and score >= threshold
            and long_mean >= min_side_mean
            and (long_mean - short_mean) >= min_advantage_gap
        ):
            predicted_side = "long"
        elif (
            match_count >= min_match_count
            and consensus_ratio >= min_consensus_ratio
            and score <= -threshold
            and short_mean >= min_side_mean
            and (short_mean - long_mean) >= min_advantage_gap
        ):
            predicted_side = "short"
        baseline_side = str(row.get("chosen_side", "") or "").strip().lower()
        applied_side = predicted_side if predicted_side in {"long", "short"} else baseline_side
        if applied_side == "long":
            applied_points = float(row.get("long_pnl_points", 0.0) or 0.0)
        elif applied_side == "short":
            applied_points = float(row.get("short_pnl_points", 0.0) or 0.0)
        else:
            applied_points = 0.0
        rows.append(
            {
                "timestamp": row.get("timestamp"),
                "baseline_points": _baseline_points(row),
                "applied_points": float(applied_points),
                "predicted_side": predicted_side,
                "applied_side": applied_side,
                "baseline_side": baseline_side,
                "override": bool(
                    (predicted_side in {"long", "short"} and predicted_side != baseline_side)
                    or predicted_side == "no_trade"
                ),
            }
        )
    eval_df = pd.DataFrame(rows)
    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"], errors="coerce")
    eval_df = eval_df.sort_values("timestamp", kind="mergesort")
    equity = eval_df["applied_points"].cumsum()
    max_dd = float((equity.cummax() - equity).max()) if not equity.empty else 0.0
    total_points = float(eval_df["applied_points"].sum()) if not eval_df.empty else 0.0
    baseline_points = float(eval_df["baseline_points"].sum()) if not eval_df.empty else 0.0
    override_count = int(eval_df["override"].fillna(False).astype(bool).sum()) if not eval_df.empty else 0
    objective = total_points - (float(model.get("_objective_weight_max_drawdown", 0.10) or 0.10) * max_dd)
    valid = override_count >= int(model.get("_min_override_count", 60) or 60)
    return {
        "threshold": float(threshold),
        "valid": bool(valid),
        "override_count": int(override_count),
        "override_rate": float(override_count / max(1, len(eval_df))),
        "baseline_points": float(baseline_points),
        "total_points": float(total_points),
        "points_uplift": float(total_points - baseline_points),
        "max_drawdown_points": float(max_dd),
        "objective": float(objective),
    }


def train_candidates(
    *,
    dataset_path: Path,
    base_bundle_path: Path,
    output_dir: Path,
) -> None:
    data = pd.read_csv(dataset_path)
    data = _prepare_conflict_frame(data)
    data = data[data["year"].notna()].copy()
    train_df = data[data["year"] <= 2023].copy()
    tune_df = data[data["year"] == 2024].copy()
    if train_df.empty or tune_df.empty:
        raise SystemExit("Conflict dataset missing train/tune rows.")

    base_bundle = json.loads(base_bundle_path.read_text(encoding="utf-8"))
    profiles = _conflict_profiles()
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "base_bundle_path": str(base_bundle_path),
        "candidates": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in profiles.items():
        model, model_summary = _build_model(train_df=train_df, cfg=cfg)
        model["_objective_weight_max_drawdown"] = float(cfg.get("objective_weight_max_drawdown", 0.10) or 0.10)
        model["_min_override_count"] = int(cfg.get("min_override_count", 60) or 60)
        trials = []
        for threshold in list(cfg.get("scope_threshold_candidates", [])):
            trials.append(_evaluate_threshold(tune_df=tune_df, model=model, threshold=float(threshold)))
        baseline_metrics = _evaluate_threshold(tune_df=tune_df, model=model, threshold=1e9)
        valid_trials = [trial for trial in trials if bool(trial.get("valid", False))]
        if valid_trials:
            valid_trials.sort(
                key=lambda item: (
                    float(item.get("objective", float("-inf"))),
                    float(item.get("points_uplift", float("-inf"))),
                ),
                reverse=True,
            )
            selected = valid_trials[0]
        else:
            selected = {
                "threshold": float(max(cfg.get("scope_threshold_candidates", [0.30]))),
                "valid": False,
                "override_count": 0,
                "override_rate": 0.0,
                "baseline_points": float(baseline_metrics.get("baseline_points", 0.0) or 0.0),
                "total_points": float(baseline_metrics.get("baseline_points", 0.0) or 0.0),
                "points_uplift": 0.0,
                "max_drawdown_points": float(baseline_metrics.get("max_drawdown_points", 0.0) or 0.0),
                "objective": float("-inf"),
            }
        model["selected_threshold"] = float(selected.get("threshold", 0.30) or 0.30)
        model["selected_threshold_source"] = "tune_2024_optimized" if valid_trials else "no_valid_tune_trial"
        model["min_side_mean_points"] = float(cfg.get("min_side_mean_points", 0.25) or 0.25)

        report = {
            "status": "ok",
            "dataset_path": str(dataset_path),
            "train_rows": int(len(train_df)),
            "tune_rows": int(len(tune_df)),
            "model_summary": dict(model_summary),
            "baseline_tune_metrics": dict(baseline_metrics),
            "selected_tune_metrics": dict(selected),
            "threshold_trials": list(trials),
            "selected_threshold": float(model["selected_threshold"]),
            "selected_threshold_source": str(model["selected_threshold_source"]),
            "config_effective": copy.deepcopy(cfg),
        }

        bundle = copy.deepcopy(base_bundle)
        bundle["conflict_side_model"] = copy.deepcopy(model)
        bundle["conflict_side_training_report"] = copy.deepcopy(report)
        meta = bundle.setdefault("metadata", {})
        if isinstance(meta, dict):
            meta["conflict_side_model_retrained_at_utc"] = datetime.now(timezone.utc).isoformat()
            meta["conflict_side_model_dataset_path"] = str(dataset_path)
            meta["conflict_side_candidate_name"] = str(name)

        bundle_path = output_dir / f"dynamic_engine3_v4_bundle.{name}.json"
        report_path = output_dir / f"{name}.conflict_training_report.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        summary["candidates"][name] = {
            "bundle_path": str(bundle_path),
            "report_path": str(report_path),
            "selected_threshold": float(model["selected_threshold"]),
            "selected_tune_metrics": dict(selected),
        }

    summary_path = output_dir / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"candidate_summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DE3 conflict-side override candidates.")
    parser.add_argument("--dataset", default="reports/de3_conflict_side_dataset_20260402.csv")
    parser.add_argument("--base-bundle", default="artifacts/de3_v4_live/latest.json")
    parser.add_argument("--output-dir", default="artifacts/de3_conflict_side_20260402")
    args = parser.parse_args()

    train_candidates(
        dataset_path=_resolve_path(str(args.dataset)),
        base_bundle_path=_resolve_path(str(args.base_bundle)),
        output_dir=(ROOT / str(args.output_dir)).resolve() if not Path(str(args.output_dir)).is_absolute() else Path(str(args.output_dir)).resolve(),
    )


if __name__ == "__main__":
    main()
