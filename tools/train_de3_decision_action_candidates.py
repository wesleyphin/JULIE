import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    raise SystemExit("Could not extract split summary from the base bundle.")


def _extract_variants(bundle: dict) -> list[dict]:
    lane_variant_quality = (
        bundle.get("lane_variant_quality", {})
        if isinstance(bundle.get("lane_variant_quality"), dict)
        else {}
    )
    variants = []
    seen = set()
    for row in lane_variant_quality.values():
        if not isinstance(row, dict):
            continue
        variant_id = str(row.get("variant_id", "") or "").strip()
        lane = str(row.get("lane", "") or "").strip()
        if not variant_id or not lane or variant_id in seen:
            continue
        variants.append({"variant_id": variant_id, "lane": lane})
        seen.add(variant_id)
    if not variants:
        raise SystemExit("Could not extract variants from the base bundle.")
    return variants


def _deep_merge(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _shape_training_cfg() -> dict:
    return {
        "enabled": True,
        "scope_mode": "lane_timeframe",
        "features": [
            "de3_entry_upper_wick_ratio",
            "de3_entry_lower_wick_ratio",
            "de3_entry_close_pos1",
            "de3_entry_body_pos1",
            "de3_entry_body1_ratio",
            "de3_entry_down3",
            "de3_entry_vol1_rel20",
            "de3_entry_ret1_atr",
            "de3_entry_dist_low5_atr",
            "de3_entry_dist_high5_atr",
        ],
        "low_quantiles": [0.15, 0.25, 0.35, 0.45],
        "high_quantiles": [0.55, 0.65, 0.75, 0.85],
        "min_scope_trades": 145,
        "min_rule_trades": 52,
        "min_complement_trades": 104,
        "max_rule_fraction": 0.40,
        "min_quality_gap": 0.07,
        "max_subset_profit_factor": 1.01,
        "max_subset_ev_mean": 0.20,
        "min_subset_loss_share": 0.52,
        "min_year_coverage": 4,
        "max_rules_per_scope": 4,
        "max_rules_per_row": 3,
    }


def _side_pattern_action_scopes() -> list[dict]:
    return [
        {
            "name": "side_pattern_session_style_pressure",
            "fields": ["side_pattern", "session", "side_considered", "strategy_style", "st_pressure_bucket"],
            "min_decisions": 84,
            "weight": 1.10,
        },
        {
            "name": "side_pattern_hour_timeframe_close",
            "fields": ["side_pattern", "ctx_hour_bucket", "timeframe", "side_considered", "st_close_bucket"],
            "min_decisions": 74,
            "weight": 1.02,
        },
        {
            "name": "side_pattern_sub_strategy_local",
            "fields": ["side_pattern", "sub_strategy", "side_considered", "st_location_bucket", "st_close_bucket"],
            "min_decisions": 58,
            "weight": 0.94,
        },
        {
            "name": "side_pattern_session_flow",
            "fields": ["side_pattern", "session", "side_considered", "st_vol_bucket", "st_down3_bucket"],
            "min_decisions": 72,
            "weight": 0.88,
        },
    ]


def _hybrid_action_scopes() -> list[dict]:
    return [
        {
            "name": "session_substate_style_pressure",
            "fields": ["session", "ctx_session_substate", "side_considered", "strategy_style", "st_pressure_bucket"],
            "min_decisions": 70,
            "weight": 1.00,
        },
        {
            "name": "side_pattern_session_style_pressure",
            "fields": ["side_pattern", "session", "side_considered", "strategy_style", "st_pressure_bucket"],
            "min_decisions": 84,
            "weight": 1.06,
        },
        {
            "name": "hour_timeframe_style_close",
            "fields": ["ctx_hour_bucket", "timeframe", "side_considered", "strategy_style", "st_close_bucket"],
            "min_decisions": 64,
            "weight": 0.92,
        },
        {
            "name": "side_pattern_sub_strategy_local",
            "fields": ["side_pattern", "sub_strategy", "side_considered", "st_location_bucket", "st_close_bucket"],
            "min_decisions": 58,
            "weight": 0.86,
        },
        {
            "name": "timeframe_pressure_shape",
            "fields": ["timeframe", "side_considered", "st_pressure_bucket", "st_wick_bias_bucket", "st_body_bucket"],
            "min_decisions": 56,
            "weight": 0.82,
        },
    ]


def _candidate_profiles(base_bundle: dict) -> dict[str, dict]:
    live_model = copy.deepcopy(
        base_bundle.get("decision_policy_model", {})
        if isinstance(base_bundle.get("decision_policy_model"), dict)
        else {}
    )
    score_base = copy.deepcopy(
        live_model.get("score_components", {})
        if isinstance(live_model.get("score_components"), dict)
        else {}
    )
    if not score_base:
        score_base = {}
    common_policy = copy.deepcopy(live_model)
    common_policy["enabled"] = True
    common_policy["default_threshold"] = float(live_model.get("selected_threshold", -0.98406) or -0.98406)
    common_policy["shape_penalty_model"] = _shape_training_cfg()
    common_policy["context_prior_model"] = copy.deepcopy(
        live_model.get("context_prior_model", {})
        if isinstance(live_model.get("context_prior_model"), dict)
        else {}
    )
    common_policy["short_term_condition_model"] = copy.deepcopy(
        live_model.get("short_term_condition_model", {})
        if isinstance(live_model.get("short_term_condition_model"), dict)
        else {}
    )
    common_policy["action_condition_model"] = {
        "enabled": True,
        "apply_only_top_side_candidate": True,
        "support_full_decisions": 180.0,
        "year_coverage_full_years": 10.0,
        "mean_scale_points": 2.40,
        "positive_rate_center": 0.50,
        "positive_rate_scale": 0.15,
        "best_rate_center": 0.34,
        "best_rate_scale": 0.15,
        "uplift_scale_points": 1.60,
        "relative_scale_points": 1.80,
        "no_trade_rate_center": 0.36,
        "no_trade_rate_scale": 0.18,
        "mean_weight": 0.34,
        "positive_rate_weight": 0.14,
        "best_rate_weight": 0.20,
        "uplift_weight": 0.12,
        "relative_weight": 0.16,
        "no_trade_penalty_weight": 0.12,
        "positive_score_weight": 1.0,
        "negative_score_weight": 0.35,
        "bonus_only": False,
        "max_abs_score": 1.10,
        "min_abs_score": 0.02,
        "max_scopes_per_row": 3,
    }
    common_policy["score_components"] = {
        **score_base,
        "weight_action_condition_component": 0.08,
        "action_condition_scale": 0.80,
    }
    common_policy["threshold_tuning"] = {
        "min_keep_rate": 0.72,
        "objective_weight_max_drawdown": 0.65,
        "objective_weight_profit_factor": 170.0,
        "objective_weight_keep_rate": 260.0,
        "objective_weight_daily_sharpe": 30.0,
        "objective_weight_daily_sortino": 12.0,
        "objective_weight_trade_sqn": 6.0,
        "max_long_share_valid": 0.95,
        "min_short_trades": 0,
    }
    return {
        "decision_action_blend_mild_v2": {
            "decision_policy": copy.deepcopy(common_policy),
        },
        "decision_action_bonus_only_v2": {
            "decision_policy": {
                **copy.deepcopy(common_policy),
                "action_condition_model": {
                    **copy.deepcopy(common_policy["action_condition_model"]),
                    "bonus_only": True,
                    "negative_score_weight": 0.0,
                    "no_trade_penalty_weight": 0.08,
                    "max_abs_score": 1.00,
                },
                "score_components": {
                    **copy.deepcopy(common_policy["score_components"]),
                    "weight_action_condition_component": 0.10,
                    "action_condition_scale": 0.74,
                },
                "threshold_tuning": {
                    **copy.deepcopy(common_policy["threshold_tuning"]),
                    "min_keep_rate": 0.76,
                    "objective_weight_keep_rate": 300.0,
                    "objective_weight_profit_factor": 155.0,
                },
            }
        },
        "decision_action_asym_keep_v2": {
            "decision_policy": {
                **copy.deepcopy(common_policy),
                "action_condition_model": {
                    **copy.deepcopy(common_policy["action_condition_model"]),
                    "positive_score_weight": 1.10,
                    "negative_score_weight": 0.20,
                    "no_trade_penalty_weight": 0.10,
                    "max_abs_score": 1.20,
                },
                "score_components": {
                    **copy.deepcopy(common_policy["score_components"]),
                    "weight_action_condition_component": 0.11,
                    "action_condition_scale": 0.70,
                },
                "threshold_tuning": {
                    **copy.deepcopy(common_policy["threshold_tuning"]),
                    "min_keep_rate": 0.68,
                    "objective_weight_keep_rate": 240.0,
                    "objective_weight_profit_factor": 185.0,
                },
            }
        },
        "decision_action_relative_side_v2": {
            "decision_policy": {
                **copy.deepcopy(common_policy),
                "action_condition_model": {
                    **copy.deepcopy(common_policy["action_condition_model"]),
                    "mean_weight": 0.06,
                    "positive_rate_weight": 0.04,
                    "best_rate_weight": 0.18,
                    "uplift_weight": 0.02,
                    "relative_weight": 0.52,
                    "no_trade_penalty_weight": 0.00,
                    "positive_score_weight": 1.0,
                    "negative_score_weight": 0.30,
                    "max_abs_score": 1.00,
                },
                "score_components": {
                    **copy.deepcopy(common_policy["score_components"]),
                    "weight_action_condition_component": 0.12,
                    "action_condition_scale": 0.68,
                },
                "threshold_tuning": {
                    **copy.deepcopy(common_policy["threshold_tuning"]),
                    "min_keep_rate": 0.80,
                    "objective_weight_keep_rate": 320.0,
                    "objective_weight_profit_factor": 150.0,
                },
            }
        },
        "decision_action_relative_bonus_v2": {
            "decision_policy": {
                **copy.deepcopy(common_policy),
                "action_condition_model": {
                    **copy.deepcopy(common_policy["action_condition_model"]),
                    "mean_weight": 0.04,
                    "positive_rate_weight": 0.02,
                    "best_rate_weight": 0.16,
                    "uplift_weight": 0.00,
                    "relative_weight": 0.50,
                    "no_trade_penalty_weight": 0.00,
                    "bonus_only": True,
                    "negative_score_weight": 0.0,
                    "max_abs_score": 0.95,
                },
                "score_components": {
                    **copy.deepcopy(common_policy["score_components"]),
                    "weight_action_condition_component": 0.10,
                    "action_condition_scale": 0.66,
                },
                "threshold_tuning": {
                    **copy.deepcopy(common_policy["threshold_tuning"]),
                    "min_keep_rate": 0.82,
                    "objective_weight_keep_rate": 340.0,
                    "objective_weight_profit_factor": 145.0,
                },
            }
        },
        "decision_action_sidepattern_guard_v1": {
            "decision_policy": {
                **copy.deepcopy(common_policy),
                "action_condition_model": {
                    **copy.deepcopy(common_policy["action_condition_model"]),
                    "scopes": _side_pattern_action_scopes(),
                    "mean_weight": 0.16,
                    "positive_rate_weight": 0.06,
                    "best_rate_weight": 0.16,
                    "uplift_weight": 0.02,
                    "relative_weight": 0.06,
                    "no_trade_penalty_weight": 0.36,
                    "positive_score_weight": 0.70,
                    "negative_score_weight": 1.55,
                    "max_abs_score": 1.35,
                    "max_scopes_per_row": 3,
                },
                "score_components": {
                    **copy.deepcopy(common_policy["score_components"]),
                    "weight_action_condition_component": 0.16,
                    "action_condition_scale": 0.56,
                },
                "threshold_tuning": {
                    **copy.deepcopy(common_policy["threshold_tuning"]),
                    "min_keep_rate": 0.60,
                    "objective_weight_keep_rate": 210.0,
                    "objective_weight_profit_factor": 205.0,
                    "objective_weight_max_drawdown": 0.78,
                    "objective_weight_daily_sharpe": 34.0,
                    "objective_weight_daily_sortino": 14.0,
                    "objective_weight_trade_sqn": 7.5,
                },
            }
        },
        "decision_action_sidepattern_blend_v1": {
            "decision_policy": {
                **copy.deepcopy(common_policy),
                "action_condition_model": {
                    **copy.deepcopy(common_policy["action_condition_model"]),
                    "scopes": _hybrid_action_scopes(),
                    "mean_weight": 0.22,
                    "positive_rate_weight": 0.09,
                    "best_rate_weight": 0.18,
                    "uplift_weight": 0.08,
                    "relative_weight": 0.12,
                    "no_trade_penalty_weight": 0.24,
                    "positive_score_weight": 0.82,
                    "negative_score_weight": 1.22,
                    "max_abs_score": 1.22,
                    "max_scopes_per_row": 3,
                },
                "score_components": {
                    **copy.deepcopy(common_policy["score_components"]),
                    "weight_action_condition_component": 0.14,
                    "action_condition_scale": 0.60,
                },
                "threshold_tuning": {
                    **copy.deepcopy(common_policy["threshold_tuning"]),
                    "min_keep_rate": 0.66,
                    "objective_weight_keep_rate": 235.0,
                    "objective_weight_profit_factor": 190.0,
                    "objective_weight_max_drawdown": 0.72,
                    "objective_weight_daily_sharpe": 32.0,
                    "objective_weight_daily_sortino": 13.0,
                    "objective_weight_trade_sqn": 6.8,
                },
            }
        },
        "decision_action_sidepattern_guard_consistent_v2": {
            "decision_policy": {
                **copy.deepcopy(common_policy),
                "action_condition_model": {
                    **copy.deepcopy(common_policy["action_condition_model"]),
                    "scopes": _side_pattern_action_scopes(),
                    "mean_weight": 0.14,
                    "positive_rate_weight": 0.06,
                    "best_rate_weight": 0.14,
                    "uplift_weight": 0.02,
                    "relative_weight": 0.04,
                    "no_trade_penalty_weight": 0.34,
                    "positive_score_weight": 0.65,
                    "negative_score_weight": 1.45,
                    "max_abs_score": 1.10,
                    "min_yearly_decisions": 8,
                    "min_negative_year_fraction": 0.62,
                    "min_positive_year_fraction": 0.58,
                    "bad_best_rate_cap": 0.30,
                    "good_best_rate_floor": 0.38,
                    "require_median_year_sign_match": True,
                    "max_scopes_per_row": 2,
                },
                "score_components": {
                    **copy.deepcopy(common_policy["score_components"]),
                    "weight_action_condition_component": 0.11,
                    "action_condition_scale": 0.62,
                },
                "threshold_tuning": {
                    **copy.deepcopy(common_policy["threshold_tuning"]),
                    "min_keep_rate": 0.68,
                    "objective_weight_keep_rate": 255.0,
                    "objective_weight_profit_factor": 195.0,
                    "objective_weight_max_drawdown": 0.76,
                    "objective_weight_daily_sharpe": 33.0,
                    "objective_weight_daily_sortino": 14.0,
                    "objective_weight_trade_sqn": 7.0,
                },
            }
        },
        "decision_action_sidepattern_blend_consistent_v2": {
            "decision_policy": {
                **copy.deepcopy(common_policy),
                "action_condition_model": {
                    **copy.deepcopy(common_policy["action_condition_model"]),
                    "scopes": _hybrid_action_scopes(),
                    "mean_weight": 0.18,
                    "positive_rate_weight": 0.08,
                    "best_rate_weight": 0.16,
                    "uplift_weight": 0.06,
                    "relative_weight": 0.10,
                    "no_trade_penalty_weight": 0.22,
                    "positive_score_weight": 0.78,
                    "negative_score_weight": 1.15,
                    "max_abs_score": 1.00,
                    "min_yearly_decisions": 8,
                    "min_negative_year_fraction": 0.58,
                    "min_positive_year_fraction": 0.56,
                    "bad_best_rate_cap": 0.30,
                    "good_best_rate_floor": 0.37,
                    "require_median_year_sign_match": True,
                    "max_scopes_per_row": 2,
                },
                "score_components": {
                    **copy.deepcopy(common_policy["score_components"]),
                    "weight_action_condition_component": 0.10,
                    "action_condition_scale": 0.66,
                },
                "threshold_tuning": {
                    **copy.deepcopy(common_policy["threshold_tuning"]),
                    "min_keep_rate": 0.72,
                    "objective_weight_keep_rate": 275.0,
                    "objective_weight_profit_factor": 180.0,
                    "objective_weight_max_drawdown": 0.72,
                    "objective_weight_daily_sharpe": 31.0,
                    "objective_weight_daily_sortino": 13.0,
                    "objective_weight_trade_sqn": 6.5,
                },
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DE3 direct-decision candidates with a local action-condition component."
    )
    parser.add_argument("--base-bundle", default="artifacts/de3_v4_live/latest.json")
    parser.add_argument("--decisions-csv", default="reports/de3_current_pool_2011_2024.csv")
    parser.add_argument("--trade-attribution-csv", default="reports/de3_current_pool_2011_2024_trade_attribution.csv")
    parser.add_argument("--decision-side-dataset", default="reports/de3_decision_side_dataset_fresh_current_live_2011_2024.csv")
    parser.add_argument("--output-dir", default="artifacts/de3_decision_action_candidates")
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    base_bundle_path = _resolve_path(str(args.base_bundle))
    decisions_csv_path = _resolve_path(str(args.decisions_csv))
    trade_csv_path = _resolve_path(str(args.trade_attribution_csv))
    decision_side_dataset_path = _resolve_path(str(args.decision_side_dataset))
    output_dir = _resolve_output_dir(str(args.output_dir))

    base_bundle = json.loads(base_bundle_path.read_text(encoding="utf-8"))
    split_summary = _extract_split_summary(base_bundle)
    variants = _extract_variants(base_bundle)

    dataset = {
        "metadata": {
            "decisions_csv_path": str(decisions_csv_path),
            "trade_attribution_csv_path": str(trade_csv_path),
            "decision_side_dataset_path": str(decision_side_dataset_path),
            "router_model_or_router_rules": copy.deepcopy(
                base_bundle.get("router_model_or_router_rules", {})
            )
            if isinstance(base_bundle.get("router_model_or_router_rules", {}), dict)
            else {},
            "lane_variant_quality": copy.deepcopy(base_bundle.get("lane_variant_quality", {}))
            if isinstance(base_bundle.get("lane_variant_quality", {}), dict)
            else {},
        },
        "split_summary": dict(split_summary),
        "variants": list(variants),
    }

    profiles = _candidate_profiles(base_bundle)
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
        "decision_side_dataset_path": str(decision_side_dataset_path),
        "split_summary": dict(split_summary),
        "variant_count": int(len(variants)),
        "candidates": {},
    }

    for name, overrides in profiles.items():
        candidate_cfg = {"decision_policy": {}}
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
            meta["decision_policy_retrained_from_decision_side_dataset"] = str(decision_side_dataset_path)
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
            "decision_selected_threshold": decision_model.get("selected_threshold"),
            "decision_selected_threshold_source": decision_model.get("selected_threshold_source"),
            "decision_selected_tune_metrics": selected_metrics,
            "decision_baseline_tune_metrics": baseline_metrics,
            "decision_score_components": decision_model.get("score_components"),
            "action_condition_training": decision_training_report.get("action_condition_training"),
        }
        print(
            f"{name}: threshold={decision_model.get('selected_threshold')} "
            f"pf={selected_metrics.get('profit_factor')} "
            f"net={selected_metrics.get('net_pnl')} "
            f"dd={selected_metrics.get('max_drawdown')}"
        )

    summary_path = output_dir / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"candidate_summary={summary_path}")


if __name__ == "__main__":
    main()
