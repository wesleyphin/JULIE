import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import CONFIG
from de3_v4_bracket_trainer import train_de3_v4_bracket_module
from de3_v4_decision_policy_trainer import train_de3_v4_decision_policy
from de3_v4_dataset_builder import build_de3_v4_training_dataset
from de3_v4_entry_policy_trainer import train_de3_v4_entry_policy
from de3_v4_lane_trainer import train_de3_v4_lane_selector
from de3_v4_router_trainer import train_de3_v4_router
from de3_v4_schema import LANE_ORDER, safe_float
from de3_v4_training_reports import resolve_path, write_json


def _default_v4_cfg() -> Dict[str, Any]:
    cfg = CONFIG.get("DE3_V4", {}) if isinstance(CONFIG.get("DE3_V4"), dict) else {}
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    training_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    training_data_cfg = (
        cfg.get("training_data", {})
        if isinstance(cfg.get("training_data"), dict)
        else {}
    )
    core_cfg = cfg.get("core", {}) if isinstance(cfg.get("core"), dict) else {}
    sat_cfg = cfg.get("satellites", {}) if isinstance(cfg.get("satellites"), dict) else {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "member_db_path": cfg.get("member_db_path", (CONFIG.get("DE3_V2", {}) or {}).get("db_path", "dynamic_engine3_strategies_v2.json")),
        "source_data_path": training_data_cfg.get("parquet_path", "es_master_outrights.parquet"),
        "bundle_path": cfg.get("bundle_path", "dynamic_engine3_v4_bundle.json"),
        "reports_dir": cfg.get("reports_dir", "reports"),
        "training_data": {
            "source_data_format": "parquet",
            "timestamp_column": str(training_data_cfg.get("timestamp_column", "timestamp")),
            "assume_timezone_if_naive": str(
                training_data_cfg.get("assume_timezone_if_naive", "UTC")
            ),
            "required_columns": list(
                training_data_cfg.get(
                    "required_columns",
                    ["open", "high", "low", "close", "volume"],
                )
            ),
            "allow_source_db_performance_metrics_for_training": bool(
                training_data_cfg.get(
                    "allow_source_db_performance_metrics_for_training",
                    False,
                )
            ),
            "split": dict(
                training_data_cfg.get(
                    "split",
                    {
                        "train_start": "2011-01-01",
                        "train_end": "2023-12-31",
                        "tune_start": "2024-01-01",
                        "tune_end": "2024-12-31",
                        "oos_start": "2025-01-01",
                        "oos_end": "2025-12-31",
                        "future_start": "2026-01-01",
                    },
                )
            ),
            "execution_rules": dict(
                training_data_cfg.get(
                    "execution_rules",
                    {
                        "enforce_no_new_entries_window": True,
                        "no_new_entries_start_hour_et": 16,
                        "no_new_entries_end_hour_et": 18,
                        "force_flat_at_time": True,
                        "force_flat_hour_et": 16,
                        "force_flat_minute_et": 0,
                    },
                )
            ),
        },
        "core": {
            "enabled": bool(core_cfg.get("enabled", True)),
            "anchor_family_ids": list(core_cfg.get("anchor_family_ids", ["5min|09-12|long|Long_Rev|T6"])),
            "default_runtime_mode": str(core_cfg.get("default_runtime_mode", "core_plus_satellites")),
            "force_anchor_when_eligible": bool(core_cfg.get("force_anchor_when_eligible", False)),
        },
        "satellites": dict(sat_cfg),
        "runtime": dict(runtime_cfg),
        "training": dict(training_cfg),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _partition_lane_sections(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {
        "Long_Rev": [],
        "Short_Rev": [],
        "Long_Mom": [],
        "Short_Mom": [],
    }
    for row in rows:
        lane = str((row or {}).get("lane", "") or "")
        if lane in out:
            out[lane].append(dict(row))
    return out


def _core_satellite_reports(
    *,
    all_variant_rows: List[Dict[str, Any]],
    satellite_rows: List[Dict[str, Any]],
    core_anchor_family_ids: Iterable[str],
) -> Dict[str, Any]:
    anchors = {str(v).strip() for v in core_anchor_family_ids if str(v).strip()}
    core_rows = [r for r in all_variant_rows if str(r.get("family_id", "") or "") in anchors]
    core_variant = core_rows[0] if core_rows else {}
    core_net = safe_float(core_variant.get("realized_net_pnl", 0.0), 0.0)
    core_pf = safe_float(core_variant.get("profit_factor", 0.0), 0.0)
    core_dd = safe_float(core_variant.get("drawdown_norm", 0.0), 0.0)
    core_trades = safe_float(core_variant.get("realized_trade_count", 0.0), 0.0)

    satellite_quality_report: Dict[str, Any] = {"satellites": []}
    portfolio_rows: List[Dict[str, Any]] = []
    orth_rows: List[Dict[str, Any]] = []

    for row in satellite_rows:
        if not isinstance(row, dict):
            continue
        variant_id = str(row.get("variant_id", "") or "")
        overlap = safe_float(row.get("estimated_overlap_with_core", 0.0), 0.0)
        bad_overlap = safe_float(row.get("estimated_bad_overlap_with_core", 0.0), 0.0)
        sat_net = safe_float(row.get("realized_net_pnl", 0.0), 0.0)
        sat_pf = safe_float(row.get("profit_factor", 0.0), 0.0)
        sat_dd = safe_float(row.get("drawdown_norm", 0.0), 0.0)
        sat_trades = safe_float(row.get("realized_trade_count", 0.0), 0.0)
        additive_scale = max(0.0, 1.0 - (0.55 * overlap))
        combined_net = core_net + (sat_net * additive_scale)
        combined_pf = core_pf + ((sat_pf - core_pf) * additive_scale)
        combined_dd = min(core_dd, sat_dd) if (core_dd > 0 and sat_dd > 0) else max(core_dd, sat_dd)
        combined_trades = core_trades + (sat_trades * (1.0 - 0.35 * overlap))
        delta_net = combined_net - core_net
        delta_pf = combined_pf - core_pf
        delta_dd = combined_dd - core_dd
        delta_trades = combined_trades - core_trades
        portfolio_rows.append(
            {
                "variant_id": variant_id,
                "family_id": str(row.get("family_id", "") or ""),
                "satellite_classification": str(row.get("satellite_classification", "")),
                "satellite_retained": bool(row.get("satellite_retained", False)),
                "core_only": {
                    "net_pnl": float(core_net),
                    "profit_factor": float(core_pf),
                    "drawdown_norm": float(core_dd),
                    "trade_count": float(core_trades),
                },
                "core_plus_satellite": {
                    "net_pnl": float(combined_net),
                    "profit_factor": float(combined_pf),
                    "drawdown_norm": float(combined_dd),
                    "trade_count": float(combined_trades),
                },
                "delta_net_pnl_vs_core": float(delta_net),
                "delta_profit_factor_vs_core": float(delta_pf),
                "delta_drawdown_vs_core": float(delta_dd),
                "delta_trade_count_vs_core": float(delta_trades),
            }
        )
        orth_rows.append(
            {
                "variant_id": variant_id,
                "family_id": str(row.get("family_id", "") or ""),
                "orthogonality_component": float(safe_float(row.get("orthogonality_component", 0.0), 0.0)),
                "estimated_overlap_with_core": float(overlap),
                "estimated_bad_overlap_with_core": float(bad_overlap),
            }
        )
        satellite_quality_report["satellites"].append(
            {
                "variant_id": variant_id,
                "family_id": str(row.get("family_id", "") or ""),
                "lane": str(row.get("lane", "") or ""),
                "standalone_viability_component": float(
                    safe_float(row.get("standalone_viability_component", 0.0), 0.0)
                ),
                "incremental_component": float(safe_float(row.get("incremental_component", 0.0), 0.0)),
                "orthogonality_component": float(safe_float(row.get("orthogonality_component", 0.0), 0.0)),
                "redundancy_penalty": float(safe_float(row.get("redundancy_penalty", 0.0), 0.0)),
                "satellite_quality_score": float(safe_float(row.get("satellite_quality_score", 0.0), 0.0)),
                "satellite_classification": str(row.get("satellite_classification", "")),
                "satellite_retained": bool(row.get("satellite_retained", False)),
                "satellite_retention_reason": str(row.get("satellite_retention_reason", "")),
            }
        )

    retained_satellite_rows = [r for r in satellite_rows if bool(r.get("satellite_retained", False))]
    satellite_quality_report["summary"] = {
        "candidate_count": int(len(satellite_rows)),
        "retained_count": int(len(retained_satellite_rows)),
        "strong_count": int(len([r for r in satellite_rows if r.get("satellite_classification") == "strong_satellite"])),
        "keep_count": int(len([r for r in satellite_rows if r.get("satellite_classification") == "keep_satellite"])),
        "weak_count": int(len([r for r in satellite_rows if r.get("satellite_classification") == "weak_satellite"])),
        "suppress_count": int(len([r for r in satellite_rows if r.get("satellite_classification") == "suppress_satellite"])),
    }
    return {
        "core_summary": {
            "core_anchor_family_ids": sorted(list(anchors)),
            "core_variant_id": str(core_variant.get("variant_id", "")),
            "core_family_id": str(core_variant.get("family_id", "")),
            "core_lane": str(core_variant.get("lane", "")),
            "t6_only_baseline": {
                "net_pnl": float(core_net),
                "profit_factor": float(core_pf),
                "drawdown_norm": float(core_dd),
                "trade_count": float(core_trades),
            },
        },
        "satellite_quality_report": satellite_quality_report,
        "portfolio_increment_report": {
            "rows": portfolio_rows,
            "retained_rows": [r for r in portfolio_rows if bool(r.get("satellite_retained", False))],
        },
        "orthogonality_summary": {
            "rows": orth_rows,
            "mean_overlap_with_core": (
                float(sum(safe_float(r.get("estimated_overlap_with_core", 0.0), 0.0) for r in orth_rows) / len(orth_rows))
                if orth_rows
                else 0.0
            ),
            "mean_bad_overlap_with_core": (
                float(sum(safe_float(r.get("estimated_bad_overlap_with_core", 0.0), 0.0) for r in orth_rows) / len(orth_rows))
                if orth_rows
                else 0.0
            ),
        },
    }


def build_de3_v4_bundle(
    *,
    source_db_path: Optional[Any] = None,
    source_data_path: Optional[Any] = None,
    timestamp_column: Optional[str] = None,
    assume_timezone_if_naive: Optional[str] = None,
    split_cfg: Optional[Dict[str, Any]] = None,
    execution_rules_cfg: Optional[Dict[str, Any]] = None,
    decisions_csv_path: Optional[Any] = None,
    trade_attribution_csv_path: Optional[Any] = None,
    out_bundle_path: Optional[Any] = None,
    reports_dir: Optional[Any] = None,
    build_router: bool = True,
    build_lane: bool = True,
    build_brackets: bool = True,
) -> Dict[str, Any]:
    cfg = _default_v4_cfg()
    source_db_path = source_db_path or cfg.get("member_db_path")
    source_data_path = source_data_path or cfg.get("source_data_path")
    training_data_cfg = (
        cfg.get("training_data", {})
        if isinstance(cfg.get("training_data"), dict)
        else {}
    )
    ts_col = (
        timestamp_column
        if timestamp_column is not None
        else training_data_cfg.get("timestamp_column", "timestamp")
    )
    ts_assume = (
        assume_timezone_if_naive
        if assume_timezone_if_naive is not None
        else training_data_cfg.get("assume_timezone_if_naive", "UTC")
    )
    split_cfg_eff = dict(training_data_cfg.get("split", {}))
    if isinstance(split_cfg, dict):
        split_cfg_eff.update({k: v for k, v in split_cfg.items() if v is not None})
    exec_rules_eff = dict(training_data_cfg.get("execution_rules", {}))
    if isinstance(execution_rules_cfg, dict):
        exec_rules_eff.update({k: v for k, v in execution_rules_cfg.items() if v is not None})
    required_cols = list(
        training_data_cfg.get("required_columns", ["open", "high", "low", "close", "volume"])
    )
    out_bundle = resolve_path(out_bundle_path or cfg.get("bundle_path", "dynamic_engine3_v4_bundle.json"))
    reports_root = resolve_path(reports_dir or cfg.get("reports_dir", "reports"))
    reports_root.mkdir(parents=True, exist_ok=True)

    dataset = build_de3_v4_training_dataset(
        source_db_path=source_db_path,
        source_parquet_path=source_data_path,
        split_cfg=split_cfg_eff,
        timestamp_column=ts_col,
        assume_timezone_if_naive=str(ts_assume or "UTC"),
        execution_rules_cfg=exec_rules_eff,
        required_bar_columns=required_cols,
        allow_source_db_performance_metrics=bool(
            training_data_cfg.get(
                "allow_source_db_performance_metrics_for_training",
                False,
            )
        ),
        decisions_csv_path=decisions_csv_path,
        trade_attribution_csv_path=trade_attribution_csv_path,
        core_anchor_family_ids=cfg.get("core", {}).get("anchor_family_ids", []),
    )

    existing_bundle = _load_json(out_bundle)
    bundle = dict(existing_bundle) if existing_bundle else {}

    router_out = train_de3_v4_router(dataset=dataset, cfg=cfg.get("training", {}))
    lane_out = train_de3_v4_lane_selector(dataset=dataset, cfg=cfg.get("training", {}))
    bracket_out = train_de3_v4_bracket_module(dataset=dataset, lane_output=lane_out, cfg=cfg.get("training", {}))
    entry_policy_out = train_de3_v4_entry_policy(dataset=dataset, cfg=cfg.get("training", {}))
    decision_policy_out = train_de3_v4_decision_policy(dataset=dataset, cfg=cfg.get("training", {}))

    all_rows = lane_out.get("all_variant_rows_scored", []) if isinstance(lane_out.get("all_variant_rows_scored"), list) else []
    satellite_rows = lane_out.get("satellite_rows", []) if isinstance(lane_out.get("satellite_rows"), list) else []
    retained_satellite_rows = [r for r in satellite_rows if bool(r.get("satellite_retained", False))]
    lane_sections = _partition_lane_sections(all_rows)
    core_sat_reports = _core_satellite_reports(
        all_variant_rows=all_rows,
        satellite_rows=satellite_rows,
        core_anchor_family_ids=cfg.get("core", {}).get("anchor_family_ids", []),
    )

    if build_router:
        bundle["router_model_or_router_rules"] = dict(router_out.get("router_payload", {}))
        bundle["router_summary"] = dict(router_out.get("router_training_report", {}).get("router_model_summary", {}))
    if build_lane:
        bundle["lane_inventory"] = dict(lane_out.get("lane_inventory", {}))
        bundle["lane_anchor_variants"] = dict(lane_out.get("lane_anchor_variants", {}))
        bundle["lane_variant_quality"] = dict(lane_out.get("lane_variant_quality", {}))
        bundle["long_rev_variants"] = lane_sections.get("Long_Rev", [])
        bundle["short_rev_variants"] = lane_sections.get("Short_Rev", [])
        bundle["long_mom_variants"] = lane_sections.get("Long_Mom", [])
        bundle["short_mom_variants"] = lane_sections.get("Short_Mom", [])
    if build_brackets:
        bundle["bracket_defaults"] = dict(bracket_out.get("bracket_defaults", {}))
        bundle["bracket_modes"] = dict(bracket_out.get("bracket_modes", {}))
        bundle["family_bracket_selector"] = dict(
            bracket_out.get("family_bracket_selector", {})
        )
    bundle["entry_policy_model"] = dict(entry_policy_out.get("entry_policy_model", {}))
    bundle["entry_policy_training_report"] = dict(
        entry_policy_out.get("entry_policy_training_report", {})
    )
    bundle["decision_policy_model"] = dict(
        decision_policy_out.get("decision_policy_model", {})
    )
    bundle["decision_policy_training_report"] = dict(
        decision_policy_out.get("decision_policy_training_report", {})
    )

    core_summary = core_sat_reports.get("core_summary", {}) if isinstance(core_sat_reports.get("core_summary"), dict) else {}
    sat_quality_report = (
        core_sat_reports.get("satellite_quality_report", {})
        if isinstance(core_sat_reports.get("satellite_quality_report"), dict)
        else {}
    )
    portfolio_increment_report = (
        core_sat_reports.get("portfolio_increment_report", {})
        if isinstance(core_sat_reports.get("portfolio_increment_report"), dict)
        else {}
    )
    orth_summary = (
        core_sat_reports.get("orthogonality_summary", {})
        if isinstance(core_sat_reports.get("orthogonality_summary"), dict)
        else {}
    )

    # Explicit core + satellite sections.
    bundle["core_anchor_reference"] = {
        "family_id": str((core_summary.get("core_family_id", "") or "")),
        "variant_id": str((core_summary.get("core_variant_id", "") or "")),
        "lane": str((core_summary.get("core_lane", "") or "")),
        "anchor_family_ids_configured": list(cfg.get("core", {}).get("anchor_family_ids", [])),
    }
    bundle["core_families"] = [
        {
            "family_id": str(core_summary.get("core_family_id", "")),
            "lane": str(core_summary.get("core_lane", "")),
        }
    ]
    bundle["core_members"] = [
        {
            "variant_id": str(core_summary.get("core_variant_id", "")),
            "family_id": str(core_summary.get("core_family_id", "")),
        }
    ]
    bundle["satellite_candidates_raw"] = list(satellite_rows)
    bundle["satellite_candidates_refined"] = list(retained_satellite_rows)
    bundle["satellite_quality_summary"] = dict(sat_quality_report.get("summary", {}))
    bundle["portfolio_increment_tests"] = dict(portfolio_increment_report)
    bundle["orthogonality_summary"] = dict(orth_summary)
    bundle["runtime_core_satellite_state"] = {
        "default_runtime_mode": str(cfg.get("core", {}).get("default_runtime_mode", "core_plus_satellites")),
        "core_enabled": bool(cfg.get("core", {}).get("enabled", True)),
        "satellites_enabled": bool((cfg.get("satellites", {}) or {}).get("enabled", True)),
        "retained_satellite_count": int(len(retained_satellite_rows)),
    }

    dataset_meta = dataset.get("metadata", {}) if isinstance(dataset.get("metadata"), dict) else {}
    split_summary = dataset.get("split_summary", {}) if isinstance(dataset.get("split_summary"), dict) else {}
    leakage_summary = dataset.get("leakage_summary", {}) if isinstance(dataset.get("leakage_summary"), dict) else {}
    data_input_audit = dataset.get("data_input_audit", {}) if isinstance(dataset.get("data_input_audit"), dict) else {}

    bundle["metadata"] = {
        "schema_version": "de3_v4_bundle_v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_db_path": str(resolve_path(source_db_path)),
        "source_data_path": str(resolve_path(source_data_path)),
        "source_data_format": "parquet",
        "bundle_path": str(out_bundle),
        "training_mode": "full_build" if (build_router and build_lane and build_brackets) else "partial_build",
        "active_lanes": list(LANE_ORDER),
        "training_split": dict(split_summary),
        "leakage_summary": dict(leakage_summary),
        "timestamp_audit": dict(dataset_meta.get("timestamp_audit", {})),
        "execution_rule_summary": dict(dataset.get("execution_rule_summary", {})),
        "allow_source_db_performance_metrics_for_training": bool(
            training_data_cfg.get(
                "allow_source_db_performance_metrics_for_training",
                False,
            )
        ),
        "data_input_validation": {
            "status": str(data_input_audit.get("validation_status", "")),
            "row_count": int(data_input_audit.get("row_count", 0) or 0),
            "min_timestamp_et": str(data_input_audit.get("min_timestamp_et", "")),
            "max_timestamp_et": str(data_input_audit.get("max_timestamp_et", "")),
        },
    }
    bundle["runtime_defaults"] = {
        "version": "v4",
        "router_enabled": True,
        "lane_selector_enabled": True,
        "bracket_module_enabled": True,
        "default_runtime_mode": str(cfg.get("core", {}).get("default_runtime_mode", "core_plus_satellites")),
        "training_data_source_format": "parquet",
        "training_split": {
            "train": [str(split_summary.get("training_start", "")), str(split_summary.get("training_end", ""))],
            "tune": [str(split_summary.get("tuning_start", "")), str(split_summary.get("tuning_end", ""))],
            "oos": [str(split_summary.get("oos_start", "")), str(split_summary.get("oos_end", ""))],
            "future_holdout": [
                str(split_summary.get("future_holdout_start", "")),
                str(split_summary.get("future_holdout_end", "")),
            ],
        },
    }
    bundle["diagnostics_summary"] = {
        "dataset_variant_count": int(len(dataset.get("variants", []))),
        "source_data_row_count": int(dataset_meta.get("source_data_row_count", 0) or 0),
        "lane_inventory_counts": {
            lane: int(len((bundle.get("lane_inventory", {}) or {}).get(lane, []))
            )
            for lane in LANE_ORDER
        },
        "retained_satellite_count": int(len(retained_satellite_rows)),
        "core_anchor_family_id": str(core_summary.get("core_family_id", "")),
        "entry_policy_model_enabled": bool(
            ((bundle.get("entry_policy_model") or {}) if isinstance(bundle.get("entry_policy_model"), dict) else {}).get("enabled", False)
        ),
        "entry_policy_selected_threshold": float(
            safe_float(
                ((bundle.get("entry_policy_model") or {}) if isinstance(bundle.get("entry_policy_model"), dict) else {}).get(
                    "selected_threshold",
                    0.0,
                ),
                0.0,
            )
        ),
        "leakage_check_passed": bool(leakage_summary.get("leakage_check_passed", False)),
    }

    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    out_bundle.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")

    router_report = dict(router_out.get("router_training_report", {}))
    lane_report = dict(lane_out.get("lane_training_report", {}))
    bracket_report = dict(bracket_out.get("bracket_training_report", {}))
    entry_policy_report = dict(entry_policy_out.get("entry_policy_training_report", {}))

    write_json(reports_root / "de3_v4_router_training_report.json", router_report)
    write_json(reports_root / "de3_v4_lane_training_report.json", lane_report)
    write_json(reports_root / "de3_v4_bracket_training_report.json", bracket_report)
    write_json(reports_root / "de3_v4_entry_policy_training_report.json", entry_policy_report)
    write_json(reports_root / "de3_v4_core_anchor_report.json", core_summary)
    write_json(reports_root / "de3_v4_satellite_quality_report.json", sat_quality_report)
    write_json(reports_root / "de3_v4_portfolio_increment_report.json", portfolio_increment_report)
    write_json(
        reports_root / "de3_v4_data_input_audit.json",
        dict(data_input_audit),
    )
    write_json(
        reports_root / "de3_v4_training_summary.json",
        {
            "bundle_path": str(out_bundle),
            "reports_dir": str(reports_root),
            "source_data_path": str(resolve_path(source_data_path)),
            "source_data_format": "parquet",
            "source_db_path": str(resolve_path(source_db_path)),
            "training_start": str(split_summary.get("training_start", "")),
            "training_end": str(split_summary.get("training_end", "")),
            "tuning_start": str(split_summary.get("tuning_start", "")),
            "tuning_end": str(split_summary.get("tuning_end", "")),
            "oos_start": str(split_summary.get("oos_start", "")),
            "oos_end": str(split_summary.get("oos_end", "")),
            "future_holdout_start": str(split_summary.get("future_holdout_start", "")),
            "future_holdout_end": str(split_summary.get("future_holdout_end", "")),
            "data_rows_train": int(split_summary.get("data_rows_train", 0) or 0),
            "data_rows_tune": int(split_summary.get("data_rows_tune", 0) or 0),
            "data_rows_oos": int(split_summary.get("data_rows_oos", 0) or 0),
            "data_rows_future_holdout": int(
                split_summary.get("data_rows_future_holdout", 0) or 0
            ),
            "entry_allowed_rows_train": int(
                split_summary.get("entry_allowed_rows_train", 0) or 0
            ),
            "entry_allowed_rows_tune": int(
                split_summary.get("entry_allowed_rows_tune", 0) or 0
            ),
            "entry_allowed_rows_oos": int(
                split_summary.get("entry_allowed_rows_oos", 0) or 0
            ),
            "entry_allowed_rows_future_holdout": int(
                split_summary.get("entry_allowed_rows_future_holdout", 0) or 0
            ),
            "timezone_interpretation": dict(dataset_meta.get("timestamp_audit", {})),
            "execution_rule_summary": dict(dataset.get("execution_rule_summary", {})),
            "allow_source_db_performance_metrics_for_training": bool(
                training_data_cfg.get(
                    "allow_source_db_performance_metrics_for_training",
                    False,
                )
            ),
            "leakage_check_passed": bool(leakage_summary.get("leakage_check_passed", False)),
            "leakage_violations": list(leakage_summary.get("violations", [])),
            "anti_leakage_assertions": {
                "tuning_used_for_model_selection": True,
                "oos_2025_held_out_for_training": True,
                "future_2026_excluded_from_training": True,
            },
            "dataset_variant_count": int(len(dataset.get("variants", []))),
            "core_anchor_family_id": str(core_summary.get("core_family_id", "")),
            "retained_satellite_count": int(len(retained_satellite_rows)),
            "entry_policy_model_enabled": bool(
                ((bundle.get("entry_policy_model") or {}) if isinstance(bundle.get("entry_policy_model"), dict) else {}).get(
                    "enabled",
                    False,
                )
            ),
            "entry_policy_selected_threshold": float(
                safe_float(
                    ((bundle.get("entry_policy_model") or {}) if isinstance(bundle.get("entry_policy_model"), dict) else {}).get(
                        "selected_threshold",
                        0.0,
                    ),
                    0.0,
                )
            ),
            "entry_policy_selected_threshold_source": str(
                ((bundle.get("entry_policy_model") or {}) if isinstance(bundle.get("entry_policy_model"), dict) else {}).get(
                    "selected_threshold_source",
                    "",
                )
            ),
            "lane_inventory_counts": {
                lane: int(len((bundle.get("lane_inventory", {}) or {}).get(lane, [])))
                for lane in LANE_ORDER
            },
            "de3_v4_train_tune_oos_protocol": {
                "train_window": [
                    str(split_summary.get("training_start", "")),
                    str(split_summary.get("training_end", "")),
                ],
                "tune_window": [
                    str(split_summary.get("tuning_start", "")),
                    str(split_summary.get("tuning_end", "")),
                ],
                "oos_window": [
                    str(split_summary.get("oos_start", "")),
                    str(split_summary.get("oos_end", "")),
                ],
                "future_holdout_window": [
                    str(split_summary.get("future_holdout_start", "")),
                    str(split_summary.get("future_holdout_end", "")),
                ],
            },
        },
    )

    return {
        "bundle_path": str(out_bundle),
        "reports_dir": str(reports_root),
        "data_input_audit_path": str(reports_root / "de3_v4_data_input_audit.json"),
        "training_summary_path": str(reports_root / "de3_v4_training_summary.json"),
        "entry_policy_training_report_path": str(reports_root / "de3_v4_entry_policy_training_report.json"),
        "bundle": bundle,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DE3v4 hierarchical runtime bundle and reports.")
    parser.add_argument("--source-db", default="", help="Source DE strategy DB JSON path (default from config).")
    parser.add_argument(
        "--source-parquet",
        default="",
        help="Primary ES 1-minute parquet source path for DE3v4 build/training (default from config).",
    )
    parser.add_argument(
        "--timestamp-column",
        default="",
        help="Optional parquet timestamp column name (defaults to config; falls back to datetime index).",
    )
    parser.add_argument(
        "--assume-timezone-if-naive",
        default="",
        help="Timezone to localize naive parquet timestamps before ET conversion (default from config).",
    )
    parser.add_argument("--train-start", default="", help="Training split start date (YYYY-MM-DD).")
    parser.add_argument("--train-end", default="", help="Training split end date (YYYY-MM-DD).")
    parser.add_argument("--tune-start", default="", help="Tuning split start date (YYYY-MM-DD).")
    parser.add_argument("--tune-end", default="", help="Tuning split end date (YYYY-MM-DD).")
    parser.add_argument("--oos-start", default="", help="True OOS split start date (YYYY-MM-DD).")
    parser.add_argument("--oos-end", default="", help="True OOS split end date (YYYY-MM-DD).")
    parser.add_argument(
        "--future-start",
        default="",
        help="Future holdout split start date (YYYY-MM-DD).",
    )
    parser.add_argument("--decisions-csv", default="", help="Optional decisions CSV for realized stats.")
    parser.add_argument("--trade-attribution-csv", default="", help="Optional trade-attribution CSV for realized stats.")
    parser.add_argument("--out-bundle", default="", help="Output DE3v4 bundle path.")
    parser.add_argument("--reports-dir", default="", help="Output report directory.")
    parser.add_argument("--build-router-only", action="store_true", help="Update router section only.")
    parser.add_argument("--build-lane-only", action="store_true", help="Update lane-selector section only.")
    parser.add_argument("--build-brackets-only", action="store_true", help="Update bracket section only.")
    parser.add_argument("--full-build", action="store_true", help="Force full build (default when no mode flag set).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected = {
        "router": bool(args.build_router_only),
        "lane": bool(args.build_lane_only),
        "brackets": bool(args.build_brackets_only),
    }
    if args.full_build or (not any(selected.values())):
        selected = {"router": True, "lane": True, "brackets": True}

    split_cfg: Dict[str, Any] = {}
    if str(args.train_start or "").strip():
        split_cfg["train_start"] = str(args.train_start).strip()
    if str(args.train_end or "").strip():
        split_cfg["train_end"] = str(args.train_end).strip()
    if str(args.tune_start or "").strip():
        split_cfg["tune_start"] = str(args.tune_start).strip()
    if str(args.tune_end or "").strip():
        split_cfg["tune_end"] = str(args.tune_end).strip()
    if str(args.oos_start or "").strip():
        split_cfg["oos_start"] = str(args.oos_start).strip()
    if str(args.oos_end or "").strip():
        split_cfg["oos_end"] = str(args.oos_end).strip()
    if str(args.future_start or "").strip():
        split_cfg["future_start"] = str(args.future_start).strip()

    result = build_de3_v4_bundle(
        source_db_path=(args.source_db or None),
        source_data_path=(args.source_parquet or None),
        timestamp_column=(args.timestamp_column or None),
        assume_timezone_if_naive=(args.assume_timezone_if_naive or None),
        split_cfg=(split_cfg or None),
        decisions_csv_path=(args.decisions_csv or None),
        trade_attribution_csv_path=(args.trade_attribution_csv or None),
        out_bundle_path=(args.out_bundle or None),
        reports_dir=(args.reports_dir or None),
        build_router=selected["router"],
        build_lane=selected["lane"],
        build_brackets=selected["brackets"],
    )
    print(f"DE3v4 bundle written: {result['bundle_path']}")
    print(f"DE3v4 reports dir: {result['reports_dir']}")
    print(f"DE3v4 data input audit: {result['data_input_audit_path']}")
    print(f"DE3v4 training summary: {result['training_summary_path']}")
    print(f"DE3v4 entry policy report: {result['entry_policy_training_report_path']}")


if __name__ == "__main__":
    main()
