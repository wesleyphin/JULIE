import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from de3_v4_decision_policy_trainer import train_de3_v4_decision_policy


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


def _deep_merge(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _extract_split_summary(bundle: dict) -> dict:
    meta = bundle.get("metadata", {}) if isinstance(bundle.get("metadata"), dict) else {}
    training_split = meta.get("training_split", {}) if isinstance(meta.get("training_split"), dict) else {}
    if training_split:
        return dict(training_split)
    decision_model = bundle.get("decision_policy_model", {}) if isinstance(bundle.get("decision_policy_model"), dict) else {}
    fit_windows = decision_model.get("fit_windows", {}) if isinstance(decision_model.get("fit_windows"), dict) else {}
    if fit_windows:
        return {
            "training_start": fit_windows.get("train_start", "2011-01-01"),
            "training_end": fit_windows.get("train_end", "2023-12-31"),
            "tuning_start": fit_windows.get("tune_start", "2024-01-01"),
            "tuning_end": fit_windows.get("tune_end", "2024-12-31"),
            "oos_start": fit_windows.get("oos_start", "2025-01-01"),
            "oos_end": fit_windows.get("oos_end", "2025-12-31"),
            "future_holdout_start": fit_windows.get("future_holdout_start", "2026-01-01"),
            "future_holdout_end": fit_windows.get("future_holdout_end", ""),
        }
    raise SystemExit("Could not extract fit windows from the base bundle.")


def _extract_variants(bundle: dict) -> list[dict]:
    lane_variant_quality = (
        bundle.get("lane_variant_quality", {})
        if isinstance(bundle.get("lane_variant_quality"), dict)
        else {}
    )
    out = []
    seen = set()
    for row in lane_variant_quality.values():
        if not isinstance(row, dict):
            continue
        variant_id = str(row.get("variant_id", "") or "").strip()
        lane = str(row.get("lane", "") or "").strip()
        if not variant_id or not lane or variant_id in seen:
            continue
        out.append({"variant_id": variant_id, "lane": lane})
        seen.add(variant_id)
    if out:
        return out
    raise SystemExit("Could not extract variants from base bundle.")


def _base_training_cfg_from_bundle(base_bundle: dict) -> dict:
    cfg = copy.deepcopy(((CONFIG.get("DE3_V4") or {}).get("training") or {}))
    if not isinstance(cfg, dict):
        raise SystemExit("CONFIG['DE3_V4']['training'] missing.")
    decision_cfg = cfg.get("decision_policy", {})
    if not isinstance(decision_cfg, dict):
        raise SystemExit("CONFIG['DE3_V4']['training']['decision_policy'] missing.")
    decision_cfg["enabled"] = True
    bundle_model = (
        base_bundle.get("decision_policy_model", {})
        if isinstance(base_bundle.get("decision_policy_model"), dict)
        else {}
    )
    bundle_minimums = (
        bundle_model.get("minimums", {})
        if isinstance(bundle_model.get("minimums"), dict)
        else {}
    )
    if bundle_model:
        decision_cfg["selection_mode"] = str(
            bundle_model.get("selection_mode", decision_cfg.get("selection_mode", "replace_router_lane"))
        )
        for key in [
            "min_variant_trades",
            "min_lane_trades",
            "allow_on_missing_stats",
            "conservative_buffer",
            "min_confidence_to_override",
            "min_score_delta_to_override",
            "min_score_margin_to_override",
            "allow_override_when_baseline_no_trade",
            "min_confidence_to_override_when_baseline_no_trade",
            "min_score_delta_to_override_when_baseline_no_trade",
            "min_score_margin_to_override_when_baseline_no_trade",
            "min_baseline_score_advantage_to_override",
            "min_baseline_score_delta_advantage_to_override",
        ]:
            if key in bundle_model:
                decision_cfg[key] = copy.deepcopy(bundle_model[key])
            elif key in bundle_minimums:
                decision_cfg[key] = copy.deepcopy(bundle_minimums[key])
        if isinstance(bundle_model.get("scope_threshold_offsets"), dict):
            decision_cfg["scope_threshold_offsets"] = copy.deepcopy(bundle_model["scope_threshold_offsets"])
        if isinstance(bundle_model.get("score_components"), dict):
            decision_cfg["score_components"] = copy.deepcopy(bundle_model["score_components"])
        if isinstance(bundle_model.get("shape_penalty_model"), dict):
            decision_cfg["shape_penalty_model"] = copy.deepcopy(bundle_model["shape_penalty_model"])
        if isinstance(bundle_model.get("context_prior_model"), dict):
            decision_cfg["context_prior_model"] = copy.deepcopy(bundle_model["context_prior_model"])
    return cfg


def _candidate_profiles() -> dict[str, dict]:
    base_scopes = [
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
    style_scopes = [
        {
            "name": "session_timeframe_style_shape",
            "fields": ["session", "timeframe", "strategy_style", "st_close_bucket", "st_wick_bias_bucket"],
            "min_trades": 68,
            "min_side_trades": 16,
            "weight": 1.00,
        },
        {
            "name": "session_substate_style_impulse",
            "fields": ["session", "ctx_session_substate", "strategy_style", "st_ret_bucket", "st_location_bucket"],
            "min_trades": 60,
            "min_side_trades": 15,
            "weight": 0.96,
        },
        {
            "name": "session_timeframe_pressure",
            "fields": ["session", "timeframe", "st_pressure_bucket", "st_down3_bucket"],
            "min_trades": 56,
            "min_side_trades": 15,
            "weight": 0.90,
        },
        {
            "name": "hour_timeframe_style_location",
            "fields": ["ctx_hour_bucket", "timeframe", "strategy_style", "st_location_bucket", "st_body_bucket"],
            "min_trades": 44,
            "min_side_trades": 12,
            "weight": 0.84,
        },
        {
            "name": "session_style_flow",
            "fields": ["session", "strategy_style", "st_vol_bucket", "st_range_bucket"],
            "min_trades": 46,
            "min_side_trades": 12,
            "weight": 0.80,
        },
        {
            "name": "hour_pressure_location",
            "fields": ["ctx_hour_bucket", "timeframe", "st_pressure_bucket", "st_location_bucket"],
            "min_trades": 42,
            "min_side_trades": 12,
            "weight": 0.72,
        },
    ]
    hybrid_extra_scopes = [
        {
            "name": "hybrid_style_timeframe_shape",
            "fields": ["session", "timeframe", "strategy_style", "st_close_bucket", "st_wick_bias_bucket"],
            "min_trades": 60,
            "min_side_trades": 14,
            "weight": 0.92,
        },
        {
            "name": "hybrid_style_substate_impulse",
            "fields": ["session", "ctx_session_substate", "strategy_style", "st_ret_bucket", "st_location_bucket"],
            "min_trades": 56,
            "min_side_trades": 14,
            "weight": 0.88,
        },
        {
            "name": "hybrid_style_hour_location",
            "fields": ["ctx_hour_bucket", "timeframe", "strategy_style", "st_location_bucket", "st_body_bucket"],
            "min_trades": 42,
            "min_side_trades": 12,
            "weight": 0.76,
        },
    ]
    hybrid_scopes = copy.deepcopy(base_scopes) + copy.deepcopy(hybrid_extra_scopes)
    return {
        "short_term_local_balanced": {
            "decision_policy": {
                "score_components": {
                    "weight_lane_prior": 0.04,
                    "weight_variant_quality_prior": 0.04,
                    "weight_edge_points": 0.12,
                    "weight_structural_score": 0.10,
                    "weight_shape_penalty_component": 0.28,
                    "weight_short_term_condition_component": 0.18,
                    "short_term_condition_scale": 0.70,
                    "weight_context_prior_component": 0.02,
                    "side_score_bias": {"long": -0.12, "short": 0.04, "default": 0.0},
                },
                "threshold_tuning": {
                    "min_keep_trades": 90,
                    "min_keep_rate": 0.34,
                    "objective_weight_max_drawdown": 0.70,
                    "objective_weight_profit_factor": 175.0,
                    "objective_weight_keep_rate": 165.0,
                },
                "short_term_condition_model": {
                    "enabled": True,
                    "support_full_trades": 180,
                    "min_year_coverage": 4,
                    "year_coverage_full_years": 10.0,
                    "quality_scale": 1.0,
                    "profit_factor_center": 1.12,
                    "profit_factor_scale": 0.28,
                    "profit_factor_weight": 0.18,
                    "loss_share_center": 0.50,
                    "loss_share_scale": 0.16,
                    "loss_share_weight": 0.12,
                    "max_abs_score": 1.20,
                    "min_abs_prior_score": 0.03,
                    "side_advantage_weight": 0.35,
                    "single_side_discount": 0.62,
                    "require_both_sides": False,
                    "max_scopes_per_row": 3,
                    "scopes": copy.deepcopy(base_scopes),
                },
            }
        },
        "short_term_local_sideaware": {
            "decision_policy": {
                "score_components": {
                    "weight_lane_prior": 0.03,
                    "weight_variant_quality_prior": 0.03,
                    "weight_edge_points": 0.12,
                    "weight_structural_score": 0.10,
                    "weight_shape_penalty_component": 0.28,
                    "weight_short_term_condition_component": 0.24,
                    "short_term_condition_scale": 0.58,
                    "weight_context_prior_component": 0.03,
                    "side_score_bias": {"long": -0.08, "short": 0.03, "default": 0.0},
                },
                "threshold_tuning": {
                    "min_keep_trades": 95,
                    "min_keep_rate": 0.33,
                    "objective_weight_max_drawdown": 0.78,
                    "objective_weight_profit_factor": 185.0,
                    "objective_weight_keep_rate": 155.0,
                },
                "short_term_condition_model": {
                    "enabled": True,
                    "support_full_trades": 170,
                    "min_year_coverage": 4,
                    "year_coverage_full_years": 10.0,
                    "quality_scale": 0.95,
                    "profit_factor_center": 1.13,
                    "profit_factor_scale": 0.26,
                    "profit_factor_weight": 0.20,
                    "loss_share_center": 0.50,
                    "loss_share_scale": 0.15,
                    "loss_share_weight": 0.14,
                    "max_abs_score": 1.28,
                    "min_abs_prior_score": 0.035,
                    "side_advantage_weight": 0.52,
                    "single_side_discount": 0.56,
                    "require_both_sides": False,
                    "max_scopes_per_row": 3,
                    "scopes": copy.deepcopy(base_scopes),
                },
            }
        },
        "short_term_style_relative": {
            "decision_policy": {
                "score_components": {
                    "weight_lane_prior": 0.01,
                    "weight_variant_quality_prior": 0.01,
                    "weight_edge_points": 0.14,
                    "weight_structural_score": 0.12,
                    "weight_shape_penalty_component": 0.28,
                    "weight_short_term_condition_component": 0.32,
                    "short_term_condition_scale": 0.52,
                    "weight_context_prior_component": 0.02,
                    "side_score_bias": {"long": -0.10, "short": 0.05, "default": 0.0},
                },
                "threshold_tuning": {
                    "min_keep_trades": 95,
                    "min_keep_rate": 0.30,
                    "objective_weight_max_drawdown": 0.80,
                    "objective_weight_profit_factor": 195.0,
                    "objective_weight_keep_rate": 145.0,
                    "objective_weight_long_share_excess_penalty": 3000.0,
                    "objective_weight_negative_side_net_penalty": 1200.0,
                    "objective_weight_side_pf_shortfall_penalty": 900.0,
                    "objective_max_long_share": 0.72,
                    "objective_side_profit_factor_floor": 1.02,
                    "objective_side_profit_factor_scale": 0.22,
                    "objective_side_net_scale": 1200.0,
                    "max_long_share_valid": 0.78,
                    "min_short_trades": 180,
                },
                "short_term_condition_model": {
                    "enabled": True,
                    "support_full_trades": 160,
                    "min_year_coverage": 4,
                    "year_coverage_full_years": 10.0,
                    "quality_scale": 0.95,
                    "profit_factor_center": 1.12,
                    "profit_factor_scale": 0.24,
                    "profit_factor_weight": 0.22,
                    "loss_share_center": 0.50,
                    "loss_share_scale": 0.14,
                    "loss_share_weight": 0.14,
                    "max_abs_score": 1.30,
                    "min_abs_prior_score": 0.04,
                    "side_advantage_weight": 0.68,
                    "single_side_discount": 0.30,
                    "require_both_sides": True,
                    "max_scopes_per_row": 3,
                    "scopes": copy.deepcopy(style_scopes),
                },
            }
        },
        "short_term_style_flexible": {
            "decision_policy": {
                "score_components": {
                    "weight_lane_prior": 0.00,
                    "weight_variant_quality_prior": 0.00,
                    "weight_edge_points": 0.14,
                    "weight_structural_score": 0.12,
                    "weight_shape_penalty_component": 0.28,
                    "weight_short_term_condition_component": 0.30,
                    "short_term_condition_scale": 0.50,
                    "weight_context_prior_component": 0.01,
                    "side_score_bias": {"long": -0.14, "short": 0.08, "default": 0.0},
                },
                "threshold_tuning": {
                    "min_keep_trades": 90,
                    "min_keep_rate": 0.28,
                    "objective_weight_max_drawdown": 0.78,
                    "objective_weight_profit_factor": 192.0,
                    "objective_weight_keep_rate": 140.0,
                    "objective_weight_long_share_excess_penalty": 2600.0,
                    "objective_weight_negative_side_net_penalty": 1000.0,
                    "objective_weight_side_pf_shortfall_penalty": 800.0,
                    "objective_max_long_share": 0.70,
                    "objective_side_profit_factor_floor": 1.00,
                    "objective_side_profit_factor_scale": 0.24,
                    "objective_side_net_scale": 1200.0,
                    "max_long_share_valid": 0.80,
                    "min_short_trades": 170,
                },
                "short_term_condition_model": {
                    "enabled": True,
                    "support_full_trades": 150,
                    "min_year_coverage": 4,
                    "year_coverage_full_years": 10.0,
                    "quality_scale": 0.95,
                    "profit_factor_center": 1.12,
                    "profit_factor_scale": 0.24,
                    "profit_factor_weight": 0.22,
                    "loss_share_center": 0.50,
                    "loss_share_scale": 0.14,
                    "loss_share_weight": 0.16,
                    "max_abs_score": 1.35,
                    "min_abs_prior_score": 0.035,
                    "side_advantage_weight": 0.72,
                    "single_side_discount": 0.24,
                    "require_both_sides": False,
                    "max_scopes_per_row": 4,
                    "scopes": copy.deepcopy(style_scopes),
                },
            }
        },
        "short_term_style_guarded": {
            "decision_policy": {
                "score_components": {
                    "weight_lane_prior": 0.01,
                    "weight_variant_quality_prior": 0.01,
                    "weight_edge_points": 0.13,
                    "weight_structural_score": 0.11,
                    "weight_shape_penalty_component": 0.30,
                    "weight_short_term_condition_component": 0.24,
                    "short_term_condition_scale": 0.60,
                    "weight_context_prior_component": 0.02,
                    "side_score_bias": {"long": -0.10, "short": 0.06, "default": 0.0},
                },
                "threshold_tuning": {
                    "min_keep_trades": 100,
                    "min_keep_rate": 0.31,
                    "objective_weight_max_drawdown": 0.84,
                    "objective_weight_profit_factor": 188.0,
                    "objective_weight_keep_rate": 138.0,
                    "objective_weight_long_share_excess_penalty": 2200.0,
                    "objective_weight_negative_side_net_penalty": 900.0,
                    "objective_weight_side_pf_shortfall_penalty": 700.0,
                    "objective_max_long_share": 0.72,
                    "objective_side_profit_factor_floor": 1.00,
                    "objective_side_profit_factor_scale": 0.24,
                    "objective_side_net_scale": 1300.0,
                    "max_long_share_valid": 0.78,
                    "min_short_trades": 180,
                },
                "short_term_condition_model": {
                    "enabled": True,
                    "support_full_trades": 170,
                    "min_year_coverage": 4,
                    "year_coverage_full_years": 10.0,
                    "quality_scale": 1.0,
                    "profit_factor_center": 1.13,
                    "profit_factor_scale": 0.24,
                    "profit_factor_weight": 0.20,
                    "loss_share_center": 0.49,
                    "loss_share_scale": 0.14,
                    "loss_share_weight": 0.16,
                    "max_abs_score": 1.18,
                    "min_abs_prior_score": 0.04,
                    "side_advantage_weight": 0.62,
                    "single_side_discount": 0.22,
                    "require_both_sides": True,
                    "max_scopes_per_row": 2,
                    "scopes": copy.deepcopy(style_scopes),
                },
            }
        },
        "short_term_style_sideaware": {
            "decision_policy": {
                "score_components": {
                    "weight_lane_prior": 0.03,
                    "weight_variant_quality_prior": 0.03,
                    "weight_edge_points": 0.12,
                    "weight_structural_score": 0.10,
                    "weight_shape_penalty_component": 0.28,
                    "weight_short_term_condition_component": 0.24,
                    "short_term_condition_scale": 0.58,
                    "weight_context_prior_component": 0.03,
                    "side_score_bias": {"long": -0.08, "short": 0.03, "default": 0.0},
                },
                "threshold_tuning": {
                    "min_keep_trades": 95,
                    "min_keep_rate": 0.33,
                    "objective_weight_max_drawdown": 0.78,
                    "objective_weight_profit_factor": 185.0,
                    "objective_weight_keep_rate": 155.0,
                },
                "short_term_condition_model": {
                    "enabled": True,
                    "support_full_trades": 170,
                    "min_year_coverage": 4,
                    "year_coverage_full_years": 10.0,
                    "quality_scale": 0.95,
                    "profit_factor_center": 1.13,
                    "profit_factor_scale": 0.26,
                    "profit_factor_weight": 0.20,
                    "loss_share_center": 0.50,
                    "loss_share_scale": 0.15,
                    "loss_share_weight": 0.14,
                    "max_abs_score": 1.28,
                    "min_abs_prior_score": 0.035,
                    "side_advantage_weight": 0.52,
                    "single_side_discount": 0.56,
                    "require_both_sides": False,
                    "max_scopes_per_row": 3,
                    "scopes": copy.deepcopy(style_scopes),
                },
            }
        },
        "short_term_style_sideaware_balanced": {
            "decision_policy": {
                "score_components": {
                    "weight_lane_prior": 0.02,
                    "weight_variant_quality_prior": 0.02,
                    "weight_edge_points": 0.12,
                    "weight_structural_score": 0.10,
                    "weight_shape_penalty_component": 0.28,
                    "weight_short_term_condition_component": 0.26,
                    "short_term_condition_scale": 0.56,
                    "weight_context_prior_component": 0.02,
                    "side_score_bias": {"long": -0.10, "short": 0.05, "default": 0.0},
                },
                "threshold_tuning": {
                    "min_keep_trades": 95,
                    "min_keep_rate": 0.31,
                    "objective_weight_max_drawdown": 0.80,
                    "objective_weight_profit_factor": 188.0,
                    "objective_weight_keep_rate": 150.0,
                    "objective_weight_long_share_excess_penalty": 1200.0,
                    "objective_weight_negative_side_net_penalty": 700.0,
                    "objective_weight_side_pf_shortfall_penalty": 450.0,
                    "objective_max_long_share": 0.76,
                    "objective_side_profit_factor_floor": 1.00,
                    "objective_side_profit_factor_scale": 0.24,
                    "objective_side_net_scale": 1200.0,
                    "max_long_share_valid": 0.84,
                    "min_short_trades": 140,
                },
                "short_term_condition_model": {
                    "enabled": True,
                    "support_full_trades": 170,
                    "min_year_coverage": 4,
                    "year_coverage_full_years": 10.0,
                    "quality_scale": 0.95,
                    "profit_factor_center": 1.13,
                    "profit_factor_scale": 0.26,
                    "profit_factor_weight": 0.20,
                    "loss_share_center": 0.50,
                    "loss_share_scale": 0.15,
                    "loss_share_weight": 0.14,
                    "max_abs_score": 1.28,
                    "min_abs_prior_score": 0.035,
                    "side_advantage_weight": 0.58,
                    "single_side_discount": 0.45,
                    "require_both_sides": False,
                    "max_scopes_per_row": 3,
                    "scopes": copy.deepcopy(style_scopes),
                },
            }
        },
        "short_term_hybrid_sideaware": {
            "decision_policy": {
                "score_components": {
                    "weight_lane_prior": 0.03,
                    "weight_variant_quality_prior": 0.03,
                    "weight_edge_points": 0.12,
                    "weight_structural_score": 0.10,
                    "weight_shape_penalty_component": 0.28,
                    "weight_short_term_condition_component": 0.25,
                    "short_term_condition_scale": 0.56,
                    "weight_context_prior_component": 0.03,
                    "side_score_bias": {"long": -0.08, "short": 0.03, "default": 0.0},
                },
                "threshold_tuning": {
                    "min_keep_trades": 95,
                    "min_keep_rate": 0.33,
                    "objective_weight_max_drawdown": 0.78,
                    "objective_weight_profit_factor": 185.0,
                    "objective_weight_keep_rate": 155.0,
                },
                "short_term_condition_model": {
                    "enabled": True,
                    "support_full_trades": 175,
                    "min_year_coverage": 4,
                    "year_coverage_full_years": 10.0,
                    "quality_scale": 0.95,
                    "profit_factor_center": 1.13,
                    "profit_factor_scale": 0.26,
                    "profit_factor_weight": 0.20,
                    "loss_share_center": 0.50,
                    "loss_share_scale": 0.15,
                    "loss_share_weight": 0.14,
                    "max_abs_score": 1.28,
                    "min_abs_prior_score": 0.035,
                    "side_advantage_weight": 0.56,
                    "single_side_discount": 0.52,
                    "require_both_sides": False,
                    "max_scopes_per_row": 4,
                    "scopes": copy.deepcopy(hybrid_scopes),
                },
            }
        },
        "short_term_local_guarded": {
            "decision_policy": {
                "score_components": {
                    "weight_lane_prior": 0.05,
                    "weight_variant_quality_prior": 0.05,
                    "weight_edge_points": 0.12,
                    "weight_structural_score": 0.09,
                    "weight_shape_penalty_component": 0.30,
                    "weight_short_term_condition_component": 0.14,
                    "short_term_condition_scale": 0.80,
                    "weight_context_prior_component": 0.00,
                    "side_score_bias": {"long": -0.10, "short": 0.04, "default": 0.0},
                },
                "threshold_tuning": {
                    "min_keep_trades": 100,
                    "min_keep_rate": 0.36,
                    "objective_weight_max_drawdown": 0.82,
                    "objective_weight_profit_factor": 180.0,
                    "objective_weight_keep_rate": 150.0,
                },
                "short_term_condition_model": {
                    "enabled": True,
                    "support_full_trades": 200,
                    "min_year_coverage": 5,
                    "year_coverage_full_years": 10.0,
                    "quality_scale": 1.0,
                    "profit_factor_center": 1.14,
                    "profit_factor_scale": 0.24,
                    "profit_factor_weight": 0.20,
                    "loss_share_center": 0.49,
                    "loss_share_scale": 0.15,
                    "loss_share_weight": 0.16,
                    "max_abs_score": 1.10,
                    "min_abs_prior_score": 0.04,
                    "side_advantage_weight": 0.40,
                    "single_side_discount": 0.50,
                    "require_both_sides": True,
                    "max_scopes_per_row": 2,
                    "scopes": copy.deepcopy(base_scopes),
                },
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DE3 short-term condition decision-policy candidates.")
    parser.add_argument("--base-bundle", default="artifacts/de3_v4_live/latest.json")
    parser.add_argument("--decisions-csv", default="reports/de3_current_pool_2011_2024.csv")
    parser.add_argument("--trade-attribution-csv", default="reports/de3_current_pool_2011_2024_trade_attribution.csv")
    parser.add_argument("--chosen-shape-csv", default="reports/de3_current_pool_2011_2024_chosen_shape.csv")
    parser.add_argument("--output-dir", default="artifacts/de3_short_term_condition_candidates")
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    base_bundle_path = _resolve_path(str(args.base_bundle))
    decisions_csv_path = _resolve_path(str(args.decisions_csv))
    trade_csv_path = _resolve_path(str(args.trade_attribution_csv))
    chosen_shape_csv_path = _resolve_path(str(args.chosen_shape_csv))
    output_dir = _resolve_output_dir(str(args.output_dir))

    base_bundle = json.loads(base_bundle_path.read_text(encoding="utf-8"))
    split_summary = _extract_split_summary(base_bundle)
    variants = _extract_variants(base_bundle)
    base_training_cfg = _base_training_cfg_from_bundle(base_bundle)

    dataset = {
        "metadata": {
            "decisions_csv_path": str(decisions_csv_path),
            "trade_attribution_csv_path": str(trade_csv_path),
            "chosen_shape_csv_path": str(chosen_shape_csv_path),
            "router_model_or_router_rules": copy.deepcopy(
                base_bundle.get("router_model_or_router_rules", {})
            )
            if isinstance(base_bundle.get("router_model_or_router_rules"), dict)
            else {},
            "lane_variant_quality": copy.deepcopy(base_bundle.get("lane_variant_quality", {}))
            if isinstance(base_bundle.get("lane_variant_quality"), dict)
            else {},
        },
        "split_summary": dict(split_summary),
        "variants": list(variants),
    }

    profiles = _candidate_profiles()
    if str(args.only or "").strip():
        allowed = {item.strip() for item in str(args.only).split(",") if item.strip()}
        profiles = {name: value for name, value in profiles.items() if name in allowed}
        if not profiles:
            raise SystemExit("No candidate profiles matched --only.")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_bundle_path": str(base_bundle_path),
        "decisions_csv_path": str(decisions_csv_path),
        "trade_attribution_csv_path": str(trade_csv_path),
        "chosen_shape_csv_path": str(chosen_shape_csv_path),
        "split_summary": dict(split_summary),
        "variant_count": int(len(variants)),
        "candidates": {},
    }

    for name, overrides in profiles.items():
        candidate_cfg = copy.deepcopy(base_training_cfg)
        _deep_merge(candidate_cfg, copy.deepcopy(overrides))
        result = train_de3_v4_decision_policy(dataset=dataset, cfg=candidate_cfg)
        decision_model = result.get("decision_policy_model", {})
        decision_training_report = result.get("decision_policy_training_report", {})
        if not isinstance(decision_model, dict) or not isinstance(decision_training_report, dict):
            raise SystemExit(f"Decision-policy training failed for candidate {name}.")

        bundle_payload = copy.deepcopy(base_bundle)
        bundle_payload["decision_policy_model"] = decision_model
        bundle_payload["decision_policy_training_report"] = decision_training_report
        meta = bundle_payload.setdefault("metadata", {})
        if isinstance(meta, dict):
            meta["decision_policy_retrained_at_utc"] = datetime.now(timezone.utc).isoformat()
            meta["decision_policy_retrained_from_decisions_csv"] = str(decisions_csv_path)
            meta["decision_policy_retrained_from_trade_attribution_csv"] = str(trade_csv_path)
            meta["decision_policy_retrained_from_chosen_shape_csv"] = str(chosen_shape_csv_path)
            meta["decision_policy_retrained_base_bundle"] = str(base_bundle_path)
            meta["decision_policy_candidate_name"] = name

        bundle_path = output_dir / f"dynamic_engine3_v4_bundle.{name}.json"
        report_path = output_dir / f"{name}.decision_training_report.json"
        bundle_path.write_text(json.dumps(bundle_payload, indent=2, ensure_ascii=True), encoding="utf-8")
        report_path.write_text(
            json.dumps(decision_training_report, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        selected_metrics = (
            decision_training_report.get("selected_tune_metrics", {})
            if isinstance(decision_training_report.get("selected_tune_metrics"), dict)
            else {}
        )
        baseline_metrics = (
            decision_training_report.get("baseline_tune_metrics", {})
            if isinstance(decision_training_report.get("baseline_tune_metrics"), dict)
            else {}
        )
        summary["candidates"][name] = {
            "bundle_path": str(bundle_path),
            "decision_training_report_path": str(report_path),
            "selected_threshold": decision_model.get("selected_threshold"),
            "selected_threshold_source": decision_model.get("selected_threshold_source"),
            "selected_tune_metrics": selected_metrics,
            "baseline_tune_metrics": baseline_metrics,
            "decision_score_components": decision_model.get("score_components"),
            "short_term_condition_training": decision_training_report.get("short_term_condition_training"),
        }
        print(
            f"{name}: threshold={decision_model.get('selected_threshold')} "
            f"keep={selected_metrics.get('keep_trades')} "
            f"pf={selected_metrics.get('profit_factor')} "
            f"dd={selected_metrics.get('max_drawdown')} "
            f"net={selected_metrics.get('net_pnl')}"
        )

    summary_path = output_dir / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"candidate_summary={summary_path}")


if __name__ == "__main__":
    main()
