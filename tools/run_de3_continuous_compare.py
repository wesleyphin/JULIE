from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
import tools.evaluate_de3_backtest_admission as eva


def _load_entry_model_report(path_text: str | None) -> Dict[str, Any]:
    raw = str(path_text or "").strip()
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_mode(
    symbol_df,
    *,
    start_raw: str,
    end_raw: str,
    mode_label: str,
    admission_enabled: bool | None,
    admission_key_granularity: str | None,
    admission_history_window_trades: int | None,
    admission_warmup_trades: int | None,
    admission_cold_avg_net_per_contract_usd: float | None,
    admission_cold_max_winrate: float | None,
    admission_defensive_size_multiplier: float | None,
    admission_require_signal_weakness: bool | None,
    admission_max_execution_quality_score: float | None,
    admission_max_entry_model_margin: float | None,
    admission_max_route_confidence: float | None,
    admission_max_edge_points: float | None,
    context_policy_enabled: bool | None,
    context_veto_enabled: bool | None,
    context_policy_mode: str | None,
    context_policy_model_path: str | None,
    context_policy_risk_min_mult: float | None,
    context_policy_risk_max_mult: float | None,
    context_policy_apply_to_size: bool | None,
    confidence_tier_ignore_policy: bool | None,
    use_policy_edge_in_ranking: bool | None,
    context_veto_model_path: str | None,
    context_veto_threshold: float | None,
    context_veto_uncertainty_z: float | None,
    context_veto_min_bucket_samples: int | None,
    context_veto_block_all_on_top: bool | None,
    policy_overlay_enabled: bool | None,
    policy_overlay_require_signal_weakness: bool | None,
    policy_overlay_min_policy_confidence: float | None,
    policy_overlay_min_policy_bucket_samples: int | None,
    policy_overlay_max_execution_quality_score: float | None,
    policy_overlay_max_entry_model_margin: float | None,
    policy_overlay_max_route_confidence: float | None,
    policy_overlay_max_edge_points: float | None,
    entry_model_report_path: str | None,
    entry_model_selected_threshold: float | None,
    entry_model_enforce_veto: bool | None,
    entry_margin_controller_enabled: bool | None,
    entry_margin_min_contracts: int | None,
    entry_margin_max_contracts: int | None,
    entry_margin_reduce_only: bool | None,
    entry_margin_defensive_max_margin: float | None,
    entry_margin_defensive_size_multiplier: float | None,
    entry_margin_lane_scope_size_multiplier: float | None,
    entry_margin_conservative_tier_size_multiplier: float | None,
    entry_margin_aggressive_min_margin: float | None,
    entry_margin_aggressive_size_multiplier: float | None,
    entry_margin_aggressive_variant_only: bool | None,
    enabled_prune_rules: List[str],
    disabled_prune_rules: List[str],
    signal_size_enabled: bool | None,
    enabled_signal_size_rules: List[str],
    disabled_signal_size_rules: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    cfg_backup = copy.deepcopy(bt.CONFIG)
    try:
        print(f"[{mode_label}] configuring runtime", flush=True)
        bt.CONFIG["BACKTEST_CONSOLE_PROGRESS"] = False
        bt.CONFIG["BACKTEST_CONSOLE_STATUS"] = False
        bt.CONFIG["BACKTEST_LIVE_REPORT_ENABLED"] = False
        eva._configure_break_even_only()
        if context_policy_enabled is not None or context_veto_enabled is not None:
            de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
            runtime_cfg = (
                copy.deepcopy(de3_cfg.get("runtime", {}) or {})
                if isinstance(de3_cfg.get("runtime", {}), dict)
                else {}
            )
            if context_policy_enabled is not None:
                runtime_cfg["disable_context_policy_gate"] = not bool(context_policy_enabled)
            if context_veto_enabled is not None:
                runtime_cfg["disable_context_veto_gate"] = not bool(context_veto_enabled)
            de3_cfg["runtime"] = runtime_cfg
            bt.CONFIG["DE3_V4"] = de3_cfg
            if context_policy_enabled is not None:
                policy_cfg = copy.deepcopy(bt.CONFIG.get("DE3_ADAPTIVE_POLICY", {}) or {})
                if isinstance(policy_cfg, dict):
                    policy_cfg["enabled"] = bool(context_policy_enabled)
                    if context_policy_mode:
                        policy_cfg["mode"] = str(context_policy_mode)
                    if context_policy_model_path:
                        policy_cfg["model_path"] = str(context_policy_model_path)
                    risk_cfg = copy.deepcopy(policy_cfg.get("risk", {}) or {})
                    if isinstance(risk_cfg, dict):
                        if context_policy_risk_min_mult is not None:
                            risk_cfg["min_mult"] = float(context_policy_risk_min_mult)
                        if context_policy_risk_max_mult is not None:
                            risk_cfg["max_mult"] = float(context_policy_risk_max_mult)
                        if context_policy_apply_to_size is not None:
                            risk_cfg["apply_to_size"] = bool(context_policy_apply_to_size)
                        policy_cfg["risk"] = risk_cfg
                    bt.CONFIG["DE3_ADAPTIVE_POLICY"] = policy_cfg
            if confidence_tier_ignore_policy is not None:
                runtime_cfg = (
                    copy.deepcopy(de3_cfg.get("runtime", {}) or {})
                    if isinstance(de3_cfg.get("runtime", {}), dict)
                    else {}
                )
                sizing_cfg = (
                    copy.deepcopy(runtime_cfg.get("confidence_tier_sizing", {}) or {})
                    if isinstance(runtime_cfg.get("confidence_tier_sizing", {}), dict)
                    else {}
                )
                if bool(confidence_tier_ignore_policy):
                    raw_fields = sizing_cfg.get("confidence_field_priority", []) or []
                    filtered_fields = [
                        str(field).strip()
                        for field in raw_fields
                        if str(field).strip() and not str(field).strip().startswith("de3_policy_")
                    ]
                    if not filtered_fields:
                        filtered_fields = ["de3_edge_confidence", "de3_v4_route_confidence"]
                    sizing_cfg["confidence_field_priority"] = filtered_fields
                    quality_cfg = (
                        copy.deepcopy(sizing_cfg.get("quality_adjustment", {}) or {})
                        if isinstance(sizing_cfg.get("quality_adjustment", {}), dict)
                        else {}
                    )
                    quality_cfg["enabled"] = False
                    sizing_cfg["quality_adjustment"] = quality_cfg
                runtime_cfg["confidence_tier_sizing"] = sizing_cfg
                de3_cfg["runtime"] = runtime_cfg
                bt.CONFIG["DE3_V4"] = de3_cfg
            if use_policy_edge_in_ranking is not None:
                selection_cfg = copy.deepcopy(bt.CONFIG.get("DE3_CANDIDATE_SELECTION", {}) or {})
                if isinstance(selection_cfg, dict):
                    selection_cfg["use_policy_edge_in_ranking"] = bool(use_policy_edge_in_ranking)
                    bt.CONFIG["DE3_CANDIDATE_SELECTION"] = selection_cfg
            if context_veto_enabled is not None:
                veto_cfg = copy.deepcopy(bt.CONFIG.get("DE3_CONTEXT_VETO", {}) or {})
                if isinstance(veto_cfg, dict):
                    veto_cfg["enabled"] = bool(context_veto_enabled)
                    if context_veto_model_path:
                        veto_cfg["model_path"] = str(context_veto_model_path)
                    if context_veto_threshold is not None:
                        veto_cfg["threshold"] = float(context_veto_threshold)
                    if context_veto_uncertainty_z is not None:
                        veto_cfg["uncertainty_z"] = float(context_veto_uncertainty_z)
                    if context_veto_min_bucket_samples is not None:
                        veto_cfg["min_bucket_samples"] = int(context_veto_min_bucket_samples)
                    if context_veto_block_all_on_top is not None:
                        veto_cfg["block_all_on_top_veto"] = bool(context_veto_block_all_on_top)
                    bt.CONFIG["DE3_CONTEXT_VETO"] = veto_cfg
        if (
            policy_overlay_enabled is not None
            or policy_overlay_require_signal_weakness is not None
            or policy_overlay_min_policy_confidence is not None
            or policy_overlay_min_policy_bucket_samples is not None
            or policy_overlay_max_execution_quality_score is not None
            or policy_overlay_max_entry_model_margin is not None
            or policy_overlay_max_route_confidence is not None
            or policy_overlay_max_edge_points is not None
        ):
            de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
            runtime_cfg = (
                copy.deepcopy(de3_cfg.get("runtime", {}) or {})
                if isinstance(de3_cfg.get("runtime", {}), dict)
                else {}
            )
            overlay_cfg = (
                copy.deepcopy(runtime_cfg.get("backtest_policy_context_overlay", {}) or {})
                if isinstance(runtime_cfg.get("backtest_policy_context_overlay", {}), dict)
                else {}
            )
            if policy_overlay_enabled is not None:
                overlay_cfg["enabled"] = bool(policy_overlay_enabled)
            if policy_overlay_require_signal_weakness is not None:
                overlay_cfg["require_signal_weakness"] = bool(policy_overlay_require_signal_weakness)
            if policy_overlay_min_policy_confidence is not None:
                overlay_cfg["min_policy_confidence"] = float(policy_overlay_min_policy_confidence)
            if policy_overlay_min_policy_bucket_samples is not None:
                overlay_cfg["min_policy_bucket_samples"] = int(policy_overlay_min_policy_bucket_samples)
            if policy_overlay_max_execution_quality_score is not None:
                overlay_cfg["max_execution_quality_score"] = float(policy_overlay_max_execution_quality_score)
            if policy_overlay_max_entry_model_margin is not None:
                overlay_cfg["max_entry_model_margin"] = float(policy_overlay_max_entry_model_margin)
            if policy_overlay_max_route_confidence is not None:
                overlay_cfg["max_route_confidence"] = float(policy_overlay_max_route_confidence)
            if policy_overlay_max_edge_points is not None:
                overlay_cfg["max_edge_points"] = float(policy_overlay_max_edge_points)
            runtime_cfg["backtest_policy_context_overlay"] = overlay_cfg
            de3_cfg["runtime"] = runtime_cfg
            bt.CONFIG["DE3_V4"] = de3_cfg
        if (
            entry_model_report_path
            or entry_model_selected_threshold is not None
            or entry_model_enforce_veto is not None
        ):
            de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
            runtime_cfg = (
                copy.deepcopy(de3_cfg.get("runtime", {}) or {})
                if isinstance(de3_cfg.get("runtime", {}), dict)
                else {}
            )
            execution_policy_cfg = (
                copy.deepcopy(runtime_cfg.get("execution_policy", {}) or {})
                if isinstance(runtime_cfg.get("execution_policy", {}), dict)
                else {}
            )
            entry_model_cfg = (
                copy.deepcopy(execution_policy_cfg.get("calibrated_entry_model", {}) or {})
                if isinstance(execution_policy_cfg.get("calibrated_entry_model", {}), dict)
                else {}
            )
            if entry_model_report_path:
                report_payload = _load_entry_model_report(entry_model_report_path)
                config_effective = (
                    report_payload.get("config_effective", {})
                    if isinstance(report_payload.get("config_effective", {}), dict)
                    else {}
                )
                score_components = (
                    config_effective.get("score_components", {})
                    if isinstance(config_effective.get("score_components", {}), dict)
                    else {}
                )
                if score_components:
                    for key, value in score_components.items():
                        entry_model_cfg[str(key)] = value
                scope_offsets = (
                    config_effective.get("scope_threshold_offsets", {})
                    if isinstance(config_effective.get("scope_threshold_offsets", {}), dict)
                    else {}
                )
                if scope_offsets:
                    entry_model_cfg["scope_threshold_offsets"] = copy.deepcopy(scope_offsets)
                if "min_variant_trades" in config_effective:
                    entry_model_cfg["min_variant_trades"] = int(config_effective.get("min_variant_trades"))
                if "min_lane_trades" in config_effective:
                    entry_model_cfg["min_lane_trades"] = int(config_effective.get("min_lane_trades"))
                if "allow_on_missing_stats" in config_effective:
                    entry_model_cfg["allow_on_missing_stats"] = bool(config_effective.get("allow_on_missing_stats"))
                selected_threshold_report = report_payload.get("selected_threshold")
                if selected_threshold_report not in (None, ""):
                    entry_model_cfg["selected_threshold"] = float(selected_threshold_report)
            if entry_model_selected_threshold is not None:
                entry_model_cfg["selected_threshold"] = float(entry_model_selected_threshold)
            if entry_model_enforce_veto is not None:
                entry_model_cfg["enforce_veto"] = bool(entry_model_enforce_veto)
            execution_policy_cfg["calibrated_entry_model"] = entry_model_cfg
            runtime_cfg["execution_policy"] = execution_policy_cfg
            de3_cfg["runtime"] = runtime_cfg
            bt.CONFIG["DE3_V4"] = de3_cfg
        if (
            entry_margin_controller_enabled is not None
            or entry_margin_min_contracts is not None
            or entry_margin_max_contracts is not None
            or entry_margin_reduce_only is not None
            or entry_margin_defensive_max_margin is not None
            or entry_margin_defensive_size_multiplier is not None
            or entry_margin_lane_scope_size_multiplier is not None
            or entry_margin_conservative_tier_size_multiplier is not None
            or entry_margin_aggressive_min_margin is not None
            or entry_margin_aggressive_size_multiplier is not None
            or entry_margin_aggressive_variant_only is not None
        ):
            de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
            runtime_cfg = (
                copy.deepcopy(de3_cfg.get("runtime", {}) or {})
                if isinstance(de3_cfg.get("runtime", {}), dict)
                else {}
            )
            entry_margin_cfg = (
                copy.deepcopy(runtime_cfg.get("backtest_entry_model_margin_controller", {}) or {})
                if isinstance(runtime_cfg.get("backtest_entry_model_margin_controller", {}), dict)
                else {}
            )
            if entry_margin_controller_enabled is not None:
                entry_margin_cfg["enabled"] = bool(entry_margin_controller_enabled)
            if entry_margin_min_contracts is not None:
                entry_margin_cfg["min_contracts"] = int(entry_margin_min_contracts)
            if entry_margin_max_contracts is not None:
                entry_margin_cfg["max_contracts"] = int(entry_margin_max_contracts)
            if entry_margin_reduce_only is not None:
                entry_margin_cfg["reduce_only"] = bool(entry_margin_reduce_only)
            if entry_margin_defensive_max_margin is not None:
                entry_margin_cfg["defensive_max_margin"] = float(entry_margin_defensive_max_margin)
            if entry_margin_defensive_size_multiplier is not None:
                entry_margin_cfg["defensive_size_multiplier"] = float(entry_margin_defensive_size_multiplier)
            if entry_margin_lane_scope_size_multiplier is not None:
                entry_margin_cfg["lane_scope_size_multiplier"] = float(entry_margin_lane_scope_size_multiplier)
            if entry_margin_conservative_tier_size_multiplier is not None:
                entry_margin_cfg["conservative_tier_size_multiplier"] = float(
                    entry_margin_conservative_tier_size_multiplier
                )
            if entry_margin_aggressive_min_margin is not None:
                entry_margin_cfg["aggressive_min_margin"] = float(entry_margin_aggressive_min_margin)
            if entry_margin_aggressive_size_multiplier is not None:
                entry_margin_cfg["aggressive_size_multiplier"] = float(entry_margin_aggressive_size_multiplier)
            if entry_margin_aggressive_variant_only is not None:
                entry_margin_cfg["aggressive_variant_only"] = bool(entry_margin_aggressive_variant_only)
            runtime_cfg["backtest_entry_model_margin_controller"] = entry_margin_cfg
            de3_cfg["runtime"] = runtime_cfg
            bt.CONFIG["DE3_V4"] = de3_cfg
        if enabled_prune_rules or disabled_prune_rules:
            de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
            runtime_cfg = (
                copy.deepcopy(de3_cfg.get("runtime", {}) or {})
                if isinstance(de3_cfg.get("runtime", {}), dict)
                else {}
            )
            prune_cfg = (
                copy.deepcopy(runtime_cfg.get("prune_rules", {}) or {})
                if isinstance(runtime_cfg.get("prune_rules", {}), dict)
                else {}
            )
            rules = [
                dict(rule)
                for rule in (
                    prune_cfg.get("rules", [])
                    if isinstance(prune_cfg.get("rules", []), (list, tuple))
                    else []
                )
                if isinstance(rule, dict)
            ]
            enable_set = {str(name).strip() for name in enabled_prune_rules if str(name).strip()}
            disable_set = {str(name).strip() for name in disabled_prune_rules if str(name).strip()}
            for rule in rules:
                rule_name = str(rule.get("name", "") or "").strip()
                if not rule_name:
                    continue
                if rule_name in enable_set:
                    rule["enabled"] = True
                if rule_name in disable_set:
                    rule["enabled"] = False
            prune_cfg["rules"] = rules
            runtime_cfg["prune_rules"] = prune_cfg
            de3_cfg["runtime"] = runtime_cfg
            bt.CONFIG["DE3_V4"] = de3_cfg
        if signal_size_enabled is not None or enabled_signal_size_rules or disabled_signal_size_rules:
            de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
            runtime_cfg = (
                copy.deepcopy(de3_cfg.get("runtime", {}) or {})
                if isinstance(de3_cfg.get("runtime", {}), dict)
                else {}
            )
            signal_size_cfg = (
                copy.deepcopy(runtime_cfg.get("backtest_signal_size_rules", {}) or {})
                if isinstance(runtime_cfg.get("backtest_signal_size_rules", {}), dict)
                else {}
            )
            if signal_size_enabled is not None:
                signal_size_cfg["enabled"] = bool(signal_size_enabled)
            rules = [
                dict(rule)
                for rule in (
                    signal_size_cfg.get("rules", [])
                    if isinstance(signal_size_cfg.get("rules", []), (list, tuple))
                    else []
                )
                if isinstance(rule, dict)
            ]
            enable_set = {
                str(name).strip()
                for name in enabled_signal_size_rules
                if str(name).strip()
            }
            disable_set = {
                str(name).strip()
                for name in disabled_signal_size_rules
                if str(name).strip()
            }
            for rule in rules:
                rule_name = str(rule.get("name", "") or "").strip()
                if not rule_name:
                    continue
                if rule_name in enable_set:
                    rule["enabled"] = True
                if rule_name in disable_set:
                    rule["enabled"] = False
            signal_size_cfg["rules"] = rules
            runtime_cfg["backtest_signal_size_rules"] = signal_size_cfg
            de3_cfg["runtime"] = runtime_cfg
            bt.CONFIG["DE3_V4"] = de3_cfg
        if admission_enabled is not None:
            runtime_cfg = (
                bt.CONFIG.get("DE3_V4", {}).get("runtime", {})
                if isinstance(bt.CONFIG.get("DE3_V4", {}).get("runtime", {}), dict)
                else {}
            )
            admission_cfg = (
                runtime_cfg.get("backtest_admission_controller", {})
                if isinstance(runtime_cfg.get("backtest_admission_controller", {}), dict)
                else {}
            )
            eva._configure_admission_controller(
                enabled=bool(admission_enabled),
                key_granularity=str(
                    admission_key_granularity
                    if admission_key_granularity is not None
                    else admission_cfg.get("key_granularity", "lane_context")
                ),
                history_window_trades=int(
                    admission_history_window_trades
                    if admission_history_window_trades is not None
                    else admission_cfg.get("history_window_trades", 20)
                ),
                warmup_trades=int(
                    admission_warmup_trades
                    if admission_warmup_trades is not None
                    else admission_cfg.get("warmup_trades", 10)
                ),
                cold_avg_net_per_contract_usd=float(
                    admission_cold_avg_net_per_contract_usd
                    if admission_cold_avg_net_per_contract_usd is not None
                    else admission_cfg.get("cold_avg_net_per_contract_usd", -10.0)
                ),
                cold_max_winrate=float(
                    admission_cold_max_winrate
                    if admission_cold_max_winrate is not None
                    else admission_cfg.get("cold_max_winrate", 0.38)
                ),
                defensive_size_multiplier=float(
                    admission_defensive_size_multiplier
                    if admission_defensive_size_multiplier is not None
                    else admission_cfg.get("defensive_size_multiplier", 0.60)
                ),
                block_avg_net_per_contract_usd=admission_cfg.get("block_avg_net_per_contract_usd"),
                block_max_winrate=admission_cfg.get("block_max_winrate"),
                min_contracts=int(admission_cfg.get("min_contracts", 1)),
                reduce_only=bool(admission_cfg.get("reduce_only", True)),
                require_signal_weakness=bool(
                    admission_require_signal_weakness
                    if admission_require_signal_weakness is not None
                    else admission_cfg.get("require_signal_weakness", False)
                ),
                max_execution_quality_score=(
                    admission_max_execution_quality_score
                    if admission_max_execution_quality_score is not None
                    else admission_cfg.get("max_execution_quality_score")
                ),
                max_entry_model_margin=(
                    admission_max_entry_model_margin
                    if admission_max_entry_model_margin is not None
                    else admission_cfg.get("max_entry_model_margin")
                ),
                max_route_confidence=(
                    admission_max_route_confidence
                    if admission_max_route_confidence is not None
                    else admission_cfg.get("max_route_confidence")
                ),
                max_edge_points=(
                    admission_max_edge_points
                    if admission_max_edge_points is not None
                    else admission_cfg.get("max_edge_points")
                ),
            )
        start_time = bt.parse_user_datetime(start_raw, bt.NY_TZ, is_end=False)
        end_time = bt.parse_user_datetime(end_raw, bt.NY_TZ, is_end=True)
        print(
            f"[{mode_label}] running backtest {start_time.isoformat()} -> {end_time.isoformat()}",
            flush=True,
        )
        stats = bt.run_backtest(
            symbol_df,
            start_time,
            end_time,
            enabled_strategies={"DynamicEngine3Strategy"},
            enabled_filters=set(),
        )
        print(f"[{mode_label}] backtest complete", flush=True)
        trade_log = stats.get("trade_log", []) or []
        risk = bt._compute_backtest_risk_metrics(trade_log)
        print(f"[{mode_label}] writing converted csv", flush=True)
        timestamp = datetime.now(bt.NY_TZ).strftime("%Y%m%d_%H%M%S")
        start_tag = start_time.strftime("%Y%m%d_%H%M")
        end_tag = end_time.strftime("%Y%m%d_%H%M")
        stem = f"backtest_AUTO_BY_DAY_{start_tag}_{end_tag}_{timestamp}_{mode_label}"
        report_path = out_dir / f"{stem}.json"
        csv_path = out_dir / f"converted_{stem}.csv"
        eva._write_converted_csv(trade_log, csv_path)
        print(f"[{mode_label}] converted csv written", flush=True)
        try:
            print(f"[{mode_label}] building monte carlo summary", flush=True)
            monte_carlo = (
                bt._build_monte_carlo_summary(
                    trade_log,
                    stats,
                    simulations=bt.BACKTEST_MONTE_CARLO_SIMULATIONS,
                    seed=bt.BACKTEST_MONTE_CARLO_SEED,
                    starting_balance=bt.BACKTEST_MONTE_CARLO_START_BALANCE,
                )
                if bt.BACKTEST_MONTE_CARLO_ENABLED
                else {"enabled": False, "status": "disabled"}
            )
        except Exception as exc:
            monte_carlo = {
                "enabled": bool(bt.BACKTEST_MONTE_CARLO_ENABLED),
                "status": "error",
                "error": str(exc),
            }
        print(f"[{mode_label}] writing lightweight report", flush=True)
        payload = {
            "created_at": datetime.now(bt.NY_TZ).isoformat(),
            "symbol": "AUTO_BY_DAY",
            "range_start": start_time.isoformat(),
            "range_end": end_time.isoformat(),
            "mode": mode_label,
            "summary": {
                "equity": float(stats.get("equity", 0.0) or 0.0),
                "trades": int(stats.get("trades", 0) or 0),
                "wins": int(stats.get("wins", 0) or 0),
                "losses": int(stats.get("losses", 0) or 0),
                "winrate": float(stats.get("winrate", 0.0) or 0.0),
                "max_drawdown": float(stats.get("max_drawdown", 0.0) or 0.0),
                "gross_profit": float(stats.get("gross_profit", 0.0) or 0.0),
                "gross_loss": float(stats.get("gross_loss", 0.0) or 0.0),
                "profit_factor": float(stats.get("profit_factor", 0.0) or 0.0),
                "avg_trade_net": float(stats.get("avg_trade_net", 0.0) or 0.0),
                "trade_sqn": float(stats.get("trade_sqn", 0.0) or 0.0),
                "trade_sharpe_like": float(stats.get("trade_sharpe_like", 0.0) or 0.0),
                "daily_sharpe": float(stats.get("daily_sharpe", 0.0) or 0.0),
                "daily_sortino": float(stats.get("daily_sortino", 0.0) or 0.0),
                "ui_sharpe": float(eva._ui_style_sharpe(trade_log)),
                "trading_days": int(stats.get("trading_days", 0) or 0),
            },
            "risk_metrics": risk,
            "monte_carlo": monte_carlo,
            "de3_backtest_admission_summary": copy.deepcopy(
                stats.get("de3_backtest_admission_summary", {}) or {}
            ),
            "de3_entry_model_margin_summary": copy.deepcopy(
                stats.get("de3_entry_model_margin_summary", {}) or {}
            ),
            "de3_policy_overlay_summary": copy.deepcopy(
                stats.get("de3_policy_overlay_summary", {}) or {}
            ),
            "csv_path": str(csv_path),
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[{mode_label}] report written: {report_path}", flush=True)
        return {
            "mode": mode_label,
            "report_path": str(report_path),
            "csv_path": str(csv_path),
            "summary": payload["summary"],
            "monte_carlo": monte_carlo,
            "de3_backtest_admission_summary": payload["de3_backtest_admission_summary"],
            "de3_entry_model_margin_summary": payload["de3_entry_model_margin_summary"],
            "de3_policy_overlay_summary": payload["de3_policy_overlay_summary"],
        }
    finally:
        bt.CONFIG.clear()
        bt.CONFIG.update(cfg_backup)


def _build_delta(baseline: Dict[str, Any], adapted: Dict[str, Any]) -> Dict[str, float]:
    base_summary = baseline.get("summary", {}) or {}
    adapted_summary = adapted.get("summary", {}) or {}
    base_mc = baseline.get("monte_carlo", {}) or {}
    adapted_mc = adapted.get("monte_carlo", {}) or {}
    return {
        "equity": float(adapted_summary.get("equity", 0.0) or 0.0)
        - float(base_summary.get("equity", 0.0) or 0.0),
        "trades": int(adapted_summary.get("trades", 0) or 0)
        - int(base_summary.get("trades", 0) or 0),
        "winrate": float(adapted_summary.get("winrate", 0.0) or 0.0)
        - float(base_summary.get("winrate", 0.0) or 0.0),
        "max_drawdown": float(adapted_summary.get("max_drawdown", 0.0) or 0.0)
        - float(base_summary.get("max_drawdown", 0.0) or 0.0),
        "daily_sharpe": float(adapted_summary.get("daily_sharpe", 0.0) or 0.0)
        - float(base_summary.get("daily_sharpe", 0.0) or 0.0),
        "profit_factor": float(adapted_summary.get("profit_factor", 0.0) or 0.0)
        - float(base_summary.get("profit_factor", 0.0) or 0.0),
        "mc_net_pnl_mean": float(adapted_mc.get("net_pnl_mean", 0.0) or 0.0)
        - float(base_mc.get("net_pnl_mean", 0.0) or 0.0),
        "mc_net_pnl_p05": float(adapted_mc.get("net_pnl_p05", 0.0) or 0.0)
        - float(base_mc.get("net_pnl_p05", 0.0) or 0.0),
        "mc_max_drawdown_mean": float(adapted_mc.get("max_drawdown_mean", 0.0) or 0.0)
        - float(base_mc.get("max_drawdown_mean", 0.0) or 0.0),
        "mc_prob_above_start": float(
            adapted_mc.get("probability_final_balance_above_start", 0.0) or 0.0
        )
        - float(base_mc.get("probability_final_balance_above_start", 0.0) or 0.0),
    }


def main() -> None:
    try:
        parser = argparse.ArgumentParser(
            description="Run continuous baseline/adapted DE3v4 backtests with lightweight JSON + converted CSV outputs."
        )
        parser.add_argument("--start", required=True, help="Start date/time, e.g. 2024-01-01")
        parser.add_argument("--end", required=True, help="End date/time, e.g. 2025-12-31 23:59")
        parser.add_argument(
            "--mode",
            choices=["baseline", "adapted", "both"],
            default="both",
            help="Which controller mode(s) to run.",
        )
        parser.add_argument(
            "--output-dir",
            default="backtest_reports/de3_continuous_compare",
            help="Directory to store lightweight reports and converted CSVs.",
        )
        parser.add_argument(
            "--admission",
            choices=["default", "on", "off"],
            default="default",
            help="Override the backtest admission controller state.",
        )
        parser.add_argument(
            "--admission-key-granularity",
            choices=["variant", "lane", "lane_context", "lane_margin_context"],
            default=None,
            help="Optional backtest admission controller key granularity override.",
        )
        parser.add_argument(
            "--admission-history-window-trades",
            type=int,
            default=None,
            help="Optional backtest admission controller history window override.",
        )
        parser.add_argument(
            "--admission-warmup-trades",
            type=int,
            default=None,
            help="Optional backtest admission controller warmup trade count override.",
        )
        parser.add_argument(
            "--admission-cold-avg-net-per-contract-usd",
            type=float,
            default=None,
            help="Optional backtest admission controller cold average net-per-contract threshold override.",
        )
        parser.add_argument(
            "--admission-cold-max-winrate",
            type=float,
            default=None,
            help="Optional backtest admission controller cold win-rate threshold override.",
        )
        parser.add_argument(
            "--admission-defensive-size-multiplier",
            type=float,
            default=None,
            help="Optional backtest admission controller defensive size multiplier override.",
        )
        parser.add_argument(
            "--admission-require-signal-weakness",
            choices=["default", "on", "off"],
            default="default",
            help="Require chosen DE3v4 signals to be weak before the admission controller can defendively size them.",
        )
        parser.add_argument(
            "--admission-max-execution-quality-score",
            type=float,
            default=None,
            help="Optional maximum DE3 execution quality score for admission-controller weakness.",
        )
        parser.add_argument(
            "--admission-max-entry-model-margin",
            type=float,
            default=None,
            help="Optional maximum DE3 entry-model margin for admission-controller weakness.",
        )
        parser.add_argument(
            "--admission-max-route-confidence",
            type=float,
            default=None,
            help="Optional maximum DE3 route confidence for admission-controller weakness.",
        )
        parser.add_argument(
            "--admission-max-edge-points",
            type=float,
            default=None,
            help="Optional maximum DE3 edge points for admission-controller weakness.",
        )
        parser.add_argument(
            "--context-policy",
            choices=["default", "on", "off"],
            default="default",
            help="Override the DE3 adaptive context-policy gate for this run.",
        )
        parser.add_argument(
            "--context-veto",
            choices=["default", "on", "off"],
            default="default",
            help="Override the DE3 legacy context-veto gate for this run.",
        )
        parser.add_argument(
            "--context-veto-model-path",
            default="",
            help="Optional DE3 context-veto model JSON path override.",
        )
        parser.add_argument(
            "--context-policy-mode",
            choices=["default", "block", "shadow"],
            default="default",
            help="Optional DE3 adaptive-policy mode override.",
        )
        parser.add_argument(
            "--context-policy-model-path",
            default="",
            help="Optional DE3 adaptive-policy model JSON path override.",
        )
        parser.add_argument(
            "--context-policy-risk-min-mult",
            type=float,
            default=None,
            help="Optional DE3 adaptive-policy minimum risk multiplier override.",
        )
        parser.add_argument(
            "--context-policy-risk-max-mult",
            type=float,
            default=None,
            help="Optional DE3 adaptive-policy maximum risk multiplier override.",
        )
        parser.add_argument(
            "--context-policy-apply-to-size",
            choices=["default", "on", "off"],
            default="default",
            help="Whether DE3 adaptive-policy should directly resize positions inside the strategy.",
        )
        parser.add_argument(
            "--confidence-tier-ignore-policy",
            choices=["default", "on", "off"],
            default="default",
            help="Keep DE3 confidence-tier sizing on baseline edge/route fields even when adaptive policy is enabled.",
        )
        parser.add_argument(
            "--use-policy-edge-in-ranking",
            choices=["default", "on", "off"],
            default="default",
            help="Whether adaptive-policy EV should influence candidate ranking.",
        )
        parser.add_argument(
            "--context-veto-threshold",
            type=float,
            default=None,
            help="Optional DE3 context-veto threshold override.",
        )
        parser.add_argument(
            "--context-veto-uncertainty-z",
            type=float,
            default=None,
            help="Optional DE3 context-veto uncertainty-z override.",
        )
        parser.add_argument(
            "--context-veto-min-bucket-samples",
            type=int,
            default=None,
            help="Optional DE3 context-veto minimum bucket sample override.",
        )
        parser.add_argument(
            "--context-veto-block-all-on-top",
            choices=["default", "on", "off"],
            default="default",
            help="Override DE3 context-veto top-candidate kill-switch behavior.",
        )
        parser.add_argument(
            "--policy-overlay",
            choices=["default", "on", "off"],
            default="default",
            help="Override the backtest-only DE3 policy weak-signal overlay.",
        )
        parser.add_argument(
            "--policy-overlay-require-signal-weakness",
            choices=["default", "on", "off"],
            default="default",
            help="Require chosen DE3v4 signals to be weak before the backtest policy overlay can downsize them.",
        )
        parser.add_argument(
            "--policy-overlay-min-policy-confidence",
            type=float,
            default=None,
            help="Optional minimum DE3 policy confidence required for the backtest policy overlay.",
        )
        parser.add_argument(
            "--policy-overlay-min-policy-bucket-samples",
            type=int,
            default=None,
            help="Optional minimum DE3 policy bucket samples required for the backtest policy overlay.",
        )
        parser.add_argument(
            "--policy-overlay-max-execution-quality-score",
            type=float,
            default=None,
            help="Optional maximum DE3 execution quality score for a signal to be considered weak by the overlay.",
        )
        parser.add_argument(
            "--policy-overlay-max-entry-model-margin",
            type=float,
            default=None,
            help="Optional maximum DE3 entry-model margin for a signal to be considered weak by the overlay.",
        )
        parser.add_argument(
            "--policy-overlay-max-route-confidence",
            type=float,
            default=None,
            help="Optional maximum DE3 route confidence for a signal to be considered weak by the overlay.",
        )
        parser.add_argument(
            "--policy-overlay-max-edge-points",
            type=float,
            default=None,
            help="Optional maximum DE3 edge points for a signal to be considered weak by the overlay.",
        )
        parser.add_argument(
            "--entry-model-report-path",
            default="",
            help="Optional DE3 entry-policy training report JSON to apply as the calibrated entry-model config.",
        )
        parser.add_argument(
            "--entry-model-selected-threshold",
            type=float,
            default=None,
            help="Optional override for DE3 calibrated entry-model selected threshold.",
        )
        parser.add_argument(
            "--entry-model-enforce-veto",
            choices=["default", "on", "off"],
            default="default",
            help="Override whether the DE3 calibrated entry model hard-vetoes below-threshold trades.",
        )
        parser.add_argument(
            "--entry-margin-controller",
            choices=["default", "on", "off"],
            default="default",
            help="Override the backtest-only DE3 entry-model margin sizing controller.",
        )
        parser.add_argument(
            "--entry-margin-min-contracts",
            type=int,
            default=None,
            help="Optional minimum contracts for the DE3 entry-model margin controller.",
        )
        parser.add_argument(
            "--entry-margin-max-contracts",
            type=int,
            default=None,
            help="Optional maximum contracts for the DE3 entry-model margin controller.",
        )
        parser.add_argument(
            "--entry-margin-reduce-only",
            choices=["default", "on", "off"],
            default="default",
            help="Prevent the DE3 entry-model margin controller from increasing size.",
        )
        parser.add_argument(
            "--entry-margin-defensive-max-margin",
            type=float,
            default=None,
            help="Optional maximum entry-model margin that still counts as a weak pass.",
        )
        parser.add_argument(
            "--entry-margin-defensive-size-multiplier",
            type=float,
            default=None,
            help="Optional size multiplier for weak-pass DE3 entry-model margins.",
        )
        parser.add_argument(
            "--entry-margin-lane-scope-size-multiplier",
            type=float,
            default=None,
            help="Optional size multiplier for DE3 lane/global fallback scopes.",
        )
        parser.add_argument(
            "--entry-margin-conservative-tier-size-multiplier",
            type=float,
            default=None,
            help="Optional size multiplier for conservative DE3 entry-model passes.",
        )
        parser.add_argument(
            "--entry-margin-aggressive-min-margin",
            type=float,
            default=None,
            help="Optional minimum entry-model margin required for aggressive DE3 sizing.",
        )
        parser.add_argument(
            "--entry-margin-aggressive-size-multiplier",
            type=float,
            default=None,
            help="Optional size multiplier for strong DE3 entry-model margins.",
        )
        parser.add_argument(
            "--entry-margin-aggressive-variant-only",
            choices=["default", "on", "off"],
            default="default",
            help="Restrict aggressive DE3 entry-model sizing to variant-scope passes.",
        )
        parser.add_argument(
            "--enable-prune-rule",
            action="append",
            default=[],
            help="Enable a named DE3v4 prune rule for this run. Can be repeated.",
        )
        parser.add_argument(
            "--disable-prune-rule",
            action="append",
            default=[],
            help="Disable a named DE3v4 prune rule for this run. Can be repeated.",
        )
        parser.add_argument(
            "--signal-size",
            choices=["default", "on", "off"],
            default="default",
            help="Override the backtest signal-size rules state.",
        )
        parser.add_argument(
            "--enable-signal-size-rule",
            action="append",
            default=[],
            help="Enable a named backtest signal-size rule for this run. Can be repeated.",
        )
        parser.add_argument(
            "--disable-signal-size-rule",
            action="append",
            default=[],
            help="Disable a named backtest signal-size rule for this run. Can be repeated.",
        )
        args = parser.parse_args()

        out_dir = (ROOT / args.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _, symbol_df = eva._load_symbol_df()

        admission_enabled: bool | None = None
        if args.admission == "on":
            admission_enabled = True
        elif args.admission == "off":
            admission_enabled = False
        admission_require_signal_weakness: bool | None = None
        if args.admission_require_signal_weakness == "on":
            admission_require_signal_weakness = True
        elif args.admission_require_signal_weakness == "off":
            admission_require_signal_weakness = False
        context_policy_enabled: bool | None = None
        if args.context_policy == "on":
            context_policy_enabled = True
        elif args.context_policy == "off":
            context_policy_enabled = False
        context_veto_enabled: bool | None = None
        if args.context_veto == "on":
            context_veto_enabled = True
        elif args.context_veto == "off":
            context_veto_enabled = False
        context_policy_apply_to_size: bool | None = None
        if args.context_policy_apply_to_size == "on":
            context_policy_apply_to_size = True
        elif args.context_policy_apply_to_size == "off":
            context_policy_apply_to_size = False
        confidence_tier_ignore_policy: bool | None = None
        if args.confidence_tier_ignore_policy == "on":
            confidence_tier_ignore_policy = True
        elif args.confidence_tier_ignore_policy == "off":
            confidence_tier_ignore_policy = False
        use_policy_edge_in_ranking: bool | None = None
        if args.use_policy_edge_in_ranking == "on":
            use_policy_edge_in_ranking = True
        elif args.use_policy_edge_in_ranking == "off":
            use_policy_edge_in_ranking = False
        context_policy_mode: str | None = None
        if args.context_policy_mode != "default":
            context_policy_mode = str(args.context_policy_mode)
        context_veto_block_all_on_top: bool | None = None
        if args.context_veto_block_all_on_top == "on":
            context_veto_block_all_on_top = True
        elif args.context_veto_block_all_on_top == "off":
            context_veto_block_all_on_top = False
        policy_overlay_enabled: bool | None = None
        if args.policy_overlay == "on":
            policy_overlay_enabled = True
        elif args.policy_overlay == "off":
            policy_overlay_enabled = False
        policy_overlay_require_signal_weakness: bool | None = None
        if args.policy_overlay_require_signal_weakness == "on":
            policy_overlay_require_signal_weakness = True
        elif args.policy_overlay_require_signal_weakness == "off":
            policy_overlay_require_signal_weakness = False
        entry_model_enforce_veto: bool | None = None
        if args.entry_model_enforce_veto == "on":
            entry_model_enforce_veto = True
        elif args.entry_model_enforce_veto == "off":
            entry_model_enforce_veto = False
        entry_margin_controller_enabled: bool | None = None
        if args.entry_margin_controller == "on":
            entry_margin_controller_enabled = True
        elif args.entry_margin_controller == "off":
            entry_margin_controller_enabled = False
        entry_margin_reduce_only: bool | None = None
        if args.entry_margin_reduce_only == "on":
            entry_margin_reduce_only = True
        elif args.entry_margin_reduce_only == "off":
            entry_margin_reduce_only = False
        entry_margin_aggressive_variant_only: bool | None = None
        if args.entry_margin_aggressive_variant_only == "on":
            entry_margin_aggressive_variant_only = True
        elif args.entry_margin_aggressive_variant_only == "off":
            entry_margin_aggressive_variant_only = False
        signal_size_enabled: bool | None = None
        if args.signal_size == "on":
            signal_size_enabled = True
        elif args.signal_size == "off":
            signal_size_enabled = False

        runs: List[Dict[str, Any]] = []
        if args.mode in {"baseline", "both"}:
            print("\n=== baseline ===", flush=True)
            baseline_compare_mode = args.mode == "both"
            runs.append(
                _run_mode(
                    symbol_df,
                    start_raw=args.start,
                    end_raw=args.end,
                    mode_label="baseline",
                    admission_enabled=(False if args.admission == "default" else admission_enabled),
                    admission_key_granularity=(
                        None if baseline_compare_mode else args.admission_key_granularity
                    ),
                    admission_history_window_trades=(
                        None if baseline_compare_mode else args.admission_history_window_trades
                    ),
                    admission_warmup_trades=(
                        None if baseline_compare_mode else args.admission_warmup_trades
                    ),
                    admission_cold_avg_net_per_contract_usd=(
                        None if baseline_compare_mode else args.admission_cold_avg_net_per_contract_usd
                    ),
                    admission_cold_max_winrate=(
                        None if baseline_compare_mode else args.admission_cold_max_winrate
                    ),
                    admission_defensive_size_multiplier=(
                        None if baseline_compare_mode else args.admission_defensive_size_multiplier
                    ),
                    admission_require_signal_weakness=(
                        None if baseline_compare_mode else admission_require_signal_weakness
                    ),
                    admission_max_execution_quality_score=(
                        None if baseline_compare_mode else args.admission_max_execution_quality_score
                    ),
                    admission_max_entry_model_margin=(
                        None if baseline_compare_mode else args.admission_max_entry_model_margin
                    ),
                    admission_max_route_confidence=(
                        None if baseline_compare_mode else args.admission_max_route_confidence
                    ),
                    admission_max_edge_points=(
                        None if baseline_compare_mode else args.admission_max_edge_points
                    ),
                    context_policy_enabled=(
                        None if baseline_compare_mode else context_policy_enabled
                    ),
                    context_veto_enabled=(
                        None if baseline_compare_mode else context_veto_enabled
                    ),
                    context_policy_mode=(
                        None if baseline_compare_mode else context_policy_mode
                    ),
                    context_policy_model_path=(
                        None if baseline_compare_mode else (str(args.context_policy_model_path or "").strip() or None)
                    ),
                    context_policy_risk_min_mult=(
                        None if baseline_compare_mode else args.context_policy_risk_min_mult
                    ),
                    context_policy_risk_max_mult=(
                        None if baseline_compare_mode else args.context_policy_risk_max_mult
                    ),
                    context_policy_apply_to_size=(
                        None if baseline_compare_mode else context_policy_apply_to_size
                    ),
                    confidence_tier_ignore_policy=(
                        None if baseline_compare_mode else confidence_tier_ignore_policy
                    ),
                    use_policy_edge_in_ranking=(
                        None if baseline_compare_mode else use_policy_edge_in_ranking
                    ),
                    context_veto_model_path=(
                        None if baseline_compare_mode else (str(args.context_veto_model_path or "").strip() or None)
                    ),
                    context_veto_threshold=(
                        None if baseline_compare_mode else args.context_veto_threshold
                    ),
                    context_veto_uncertainty_z=(
                        None if baseline_compare_mode else args.context_veto_uncertainty_z
                    ),
                    context_veto_min_bucket_samples=(
                        None if baseline_compare_mode else args.context_veto_min_bucket_samples
                    ),
                    context_veto_block_all_on_top=(
                        None if baseline_compare_mode else context_veto_block_all_on_top
                    ),
                    policy_overlay_enabled=(
                        None if baseline_compare_mode else policy_overlay_enabled
                    ),
                    policy_overlay_require_signal_weakness=(
                        None if baseline_compare_mode else policy_overlay_require_signal_weakness
                    ),
                    policy_overlay_min_policy_confidence=(
                        None if baseline_compare_mode else args.policy_overlay_min_policy_confidence
                    ),
                    policy_overlay_min_policy_bucket_samples=(
                        None if baseline_compare_mode else args.policy_overlay_min_policy_bucket_samples
                    ),
                    policy_overlay_max_execution_quality_score=(
                        None if baseline_compare_mode else args.policy_overlay_max_execution_quality_score
                    ),
                    policy_overlay_max_entry_model_margin=(
                        None if baseline_compare_mode else args.policy_overlay_max_entry_model_margin
                    ),
                    policy_overlay_max_route_confidence=(
                        None if baseline_compare_mode else args.policy_overlay_max_route_confidence
                    ),
                    policy_overlay_max_edge_points=(
                        None if baseline_compare_mode else args.policy_overlay_max_edge_points
                    ),
                    entry_model_report_path=(
                        None
                        if baseline_compare_mode
                        else (str(args.entry_model_report_path or "").strip() or None)
                    ),
                    entry_model_selected_threshold=(
                        None if baseline_compare_mode else args.entry_model_selected_threshold
                    ),
                    entry_model_enforce_veto=(
                        None if baseline_compare_mode else entry_model_enforce_veto
                    ),
                    entry_margin_controller_enabled=(
                        False
                        if baseline_compare_mode
                        else (
                            None
                            if args.entry_margin_controller == "default"
                            else entry_margin_controller_enabled
                        )
                    ),
                    entry_margin_min_contracts=(
                        None if baseline_compare_mode else args.entry_margin_min_contracts
                    ),
                    entry_margin_max_contracts=(
                        None if baseline_compare_mode else args.entry_margin_max_contracts
                    ),
                    entry_margin_reduce_only=(
                        None if baseline_compare_mode else entry_margin_reduce_only
                    ),
                    entry_margin_defensive_max_margin=(
                        None if baseline_compare_mode else args.entry_margin_defensive_max_margin
                    ),
                    entry_margin_defensive_size_multiplier=(
                        None
                        if baseline_compare_mode
                        else args.entry_margin_defensive_size_multiplier
                    ),
                    entry_margin_lane_scope_size_multiplier=(
                        None
                        if baseline_compare_mode
                        else args.entry_margin_lane_scope_size_multiplier
                    ),
                    entry_margin_conservative_tier_size_multiplier=(
                        None
                        if baseline_compare_mode
                        else args.entry_margin_conservative_tier_size_multiplier
                    ),
                    entry_margin_aggressive_min_margin=(
                        None if baseline_compare_mode else args.entry_margin_aggressive_min_margin
                    ),
                    entry_margin_aggressive_size_multiplier=(
                        None
                        if baseline_compare_mode
                        else args.entry_margin_aggressive_size_multiplier
                    ),
                    entry_margin_aggressive_variant_only=(
                        None
                        if baseline_compare_mode
                        else entry_margin_aggressive_variant_only
                    ),
                    enabled_prune_rules=([] if baseline_compare_mode else list(args.enable_prune_rule or [])),
                    disabled_prune_rules=([] if baseline_compare_mode else list(args.disable_prune_rule or [])),
                    signal_size_enabled=(
                        False
                        if baseline_compare_mode
                        else (False if args.signal_size == "default" else signal_size_enabled)
                    ),
                    enabled_signal_size_rules=(
                        [] if baseline_compare_mode else list(args.enable_signal_size_rule or [])
                    ),
                    disabled_signal_size_rules=(
                        [] if baseline_compare_mode else list(args.disable_signal_size_rule or [])
                    ),
                    out_dir=out_dir,
                )
            )
        if args.mode in {"adapted", "both"}:
            print("\n=== adapted ===", flush=True)
            runs.append(
                _run_mode(
                    symbol_df,
                    start_raw=args.start,
                    end_raw=args.end,
                    mode_label="adapted",
                    admission_enabled=(True if args.admission == "default" else admission_enabled),
                    admission_key_granularity=args.admission_key_granularity,
                    admission_history_window_trades=args.admission_history_window_trades,
                    admission_warmup_trades=args.admission_warmup_trades,
                    admission_cold_avg_net_per_contract_usd=args.admission_cold_avg_net_per_contract_usd,
                    admission_cold_max_winrate=args.admission_cold_max_winrate,
                    admission_defensive_size_multiplier=args.admission_defensive_size_multiplier,
                    admission_require_signal_weakness=admission_require_signal_weakness,
                    admission_max_execution_quality_score=args.admission_max_execution_quality_score,
                    admission_max_entry_model_margin=args.admission_max_entry_model_margin,
                    admission_max_route_confidence=args.admission_max_route_confidence,
                    admission_max_edge_points=args.admission_max_edge_points,
                    context_policy_enabled=context_policy_enabled,
                    context_veto_enabled=context_veto_enabled,
                    context_policy_mode=context_policy_mode,
                    context_policy_model_path=(str(args.context_policy_model_path or "").strip() or None),
                    context_policy_risk_min_mult=args.context_policy_risk_min_mult,
                    context_policy_risk_max_mult=args.context_policy_risk_max_mult,
                    context_policy_apply_to_size=context_policy_apply_to_size,
                    confidence_tier_ignore_policy=confidence_tier_ignore_policy,
                    use_policy_edge_in_ranking=use_policy_edge_in_ranking,
                    context_veto_model_path=(str(args.context_veto_model_path or "").strip() or None),
                    context_veto_threshold=args.context_veto_threshold,
                    context_veto_uncertainty_z=args.context_veto_uncertainty_z,
                    context_veto_min_bucket_samples=args.context_veto_min_bucket_samples,
                    context_veto_block_all_on_top=context_veto_block_all_on_top,
                    policy_overlay_enabled=policy_overlay_enabled,
                    policy_overlay_require_signal_weakness=policy_overlay_require_signal_weakness,
                    policy_overlay_min_policy_confidence=args.policy_overlay_min_policy_confidence,
                    policy_overlay_min_policy_bucket_samples=args.policy_overlay_min_policy_bucket_samples,
                    policy_overlay_max_execution_quality_score=args.policy_overlay_max_execution_quality_score,
                    policy_overlay_max_entry_model_margin=args.policy_overlay_max_entry_model_margin,
                    policy_overlay_max_route_confidence=args.policy_overlay_max_route_confidence,
                    policy_overlay_max_edge_points=args.policy_overlay_max_edge_points,
                    entry_model_report_path=(str(args.entry_model_report_path or "").strip() or None),
                    entry_model_selected_threshold=args.entry_model_selected_threshold,
                    entry_model_enforce_veto=entry_model_enforce_veto,
                    entry_margin_controller_enabled=(
                        None
                        if args.entry_margin_controller == "default"
                        else entry_margin_controller_enabled
                    ),
                    entry_margin_min_contracts=args.entry_margin_min_contracts,
                    entry_margin_max_contracts=args.entry_margin_max_contracts,
                    entry_margin_reduce_only=entry_margin_reduce_only,
                    entry_margin_defensive_max_margin=args.entry_margin_defensive_max_margin,
                    entry_margin_defensive_size_multiplier=args.entry_margin_defensive_size_multiplier,
                    entry_margin_lane_scope_size_multiplier=args.entry_margin_lane_scope_size_multiplier,
                    entry_margin_conservative_tier_size_multiplier=args.entry_margin_conservative_tier_size_multiplier,
                    entry_margin_aggressive_min_margin=args.entry_margin_aggressive_min_margin,
                    entry_margin_aggressive_size_multiplier=args.entry_margin_aggressive_size_multiplier,
                    entry_margin_aggressive_variant_only=entry_margin_aggressive_variant_only,
                    enabled_prune_rules=list(args.enable_prune_rule or []),
                    disabled_prune_rules=list(args.disable_prune_rule or []),
                    signal_size_enabled=(False if args.signal_size == "default" else signal_size_enabled),
                    enabled_signal_size_rules=list(args.enable_signal_size_rule or []),
                    disabled_signal_size_rules=list(args.disable_signal_size_rule or []),
                    out_dir=out_dir,
                )
            )

        payload: Dict[str, Any] = {
            "created_at": datetime.now(bt.NY_TZ).isoformat(),
            "start": args.start,
            "end": args.end,
            "mode": args.mode,
            "admission": args.admission,
            "admission_key_granularity": args.admission_key_granularity,
            "admission_history_window_trades": args.admission_history_window_trades,
            "admission_warmup_trades": args.admission_warmup_trades,
            "admission_cold_avg_net_per_contract_usd": args.admission_cold_avg_net_per_contract_usd,
            "admission_cold_max_winrate": args.admission_cold_max_winrate,
            "admission_defensive_size_multiplier": args.admission_defensive_size_multiplier,
            "admission_require_signal_weakness": args.admission_require_signal_weakness,
            "admission_max_execution_quality_score": args.admission_max_execution_quality_score,
            "admission_max_entry_model_margin": args.admission_max_entry_model_margin,
            "admission_max_route_confidence": args.admission_max_route_confidence,
            "admission_max_edge_points": args.admission_max_edge_points,
            "context_policy": args.context_policy,
            "context_policy_mode": args.context_policy_mode,
            "context_policy_model_path": str(args.context_policy_model_path or "").strip() or None,
            "context_policy_risk_min_mult": args.context_policy_risk_min_mult,
            "context_policy_risk_max_mult": args.context_policy_risk_max_mult,
            "context_policy_apply_to_size": args.context_policy_apply_to_size,
            "confidence_tier_ignore_policy": args.confidence_tier_ignore_policy,
            "use_policy_edge_in_ranking": args.use_policy_edge_in_ranking,
            "context_veto": args.context_veto,
            "context_veto_model_path": str(args.context_veto_model_path or "").strip() or None,
            "context_veto_threshold": args.context_veto_threshold,
            "context_veto_uncertainty_z": args.context_veto_uncertainty_z,
            "context_veto_min_bucket_samples": args.context_veto_min_bucket_samples,
            "context_veto_block_all_on_top": args.context_veto_block_all_on_top,
            "policy_overlay": args.policy_overlay,
            "policy_overlay_require_signal_weakness": args.policy_overlay_require_signal_weakness,
            "policy_overlay_min_policy_confidence": args.policy_overlay_min_policy_confidence,
            "policy_overlay_min_policy_bucket_samples": args.policy_overlay_min_policy_bucket_samples,
            "policy_overlay_max_execution_quality_score": args.policy_overlay_max_execution_quality_score,
            "policy_overlay_max_entry_model_margin": args.policy_overlay_max_entry_model_margin,
            "policy_overlay_max_route_confidence": args.policy_overlay_max_route_confidence,
            "policy_overlay_max_edge_points": args.policy_overlay_max_edge_points,
            "entry_model_report_path": str(args.entry_model_report_path or "").strip() or None,
            "entry_model_selected_threshold": args.entry_model_selected_threshold,
            "entry_model_enforce_veto": args.entry_model_enforce_veto,
            "entry_margin_controller": args.entry_margin_controller,
            "entry_margin_min_contracts": args.entry_margin_min_contracts,
            "entry_margin_max_contracts": args.entry_margin_max_contracts,
            "entry_margin_reduce_only": args.entry_margin_reduce_only,
            "entry_margin_defensive_max_margin": args.entry_margin_defensive_max_margin,
            "entry_margin_defensive_size_multiplier": args.entry_margin_defensive_size_multiplier,
            "entry_margin_lane_scope_size_multiplier": args.entry_margin_lane_scope_size_multiplier,
            "entry_margin_conservative_tier_size_multiplier": args.entry_margin_conservative_tier_size_multiplier,
            "entry_margin_aggressive_min_margin": args.entry_margin_aggressive_min_margin,
            "entry_margin_aggressive_size_multiplier": args.entry_margin_aggressive_size_multiplier,
            "entry_margin_aggressive_variant_only": args.entry_margin_aggressive_variant_only,
            "enabled_prune_rules": list(args.enable_prune_rule or []),
            "disabled_prune_rules": list(args.disable_prune_rule or []),
            "signal_size": args.signal_size,
            "enabled_signal_size_rules": list(args.enable_signal_size_rule or []),
            "disabled_signal_size_rules": list(args.disable_signal_size_rule or []),
            "runs": runs,
        }
        if args.mode == "both" and len(runs) == 2:
            payload["delta"] = _build_delta(runs[0], runs[1])
        summary_path = out_dir / (
            f"de3_continuous_compare_{args.start.replace(':', '').replace(' ', '_')}_{args.end.replace(':', '').replace(' ', '_')}.json"
        )
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"summary_path": str(summary_path), **payload}, indent=2), flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
