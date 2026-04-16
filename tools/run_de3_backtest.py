import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from config import CONFIG


def _resolve_source(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Data file not found: {path_arg}")


def _prepare_symbol_df(
    source_path: Path,
    start_time,
    end_time,
    symbol_mode: str,
    symbol_method: str,
    *,
    pre_roll_days: int = 0,
):
    df = bt.load_csv_cached(source_path, cache_dir=ROOT / "cache", use_cache=True)
    if df.empty:
        raise SystemExit("No rows found in the source file.")

    # Keep symbol selection scoped to the requested window, but allow a
    # configurable pre-roll buffer so strategy warmup has prior history.
    selection_start = start_time - pd.Timedelta(days=max(0, int(pre_roll_days)))
    selection_start = selection_start.replace(hour=0, minute=0, second=0, microsecond=0)
    source_df = df[(df.index >= selection_start) & (df.index <= end_time)]
    if source_df.empty:
        raise SystemExit("No rows found inside the requested symbol-selection window.")

    symbol = None
    symbol_distribution = {}
    symbol_df = source_df
    if "symbol" in source_df.columns:
        if symbol_mode != "single":
            symbol_df, auto_label, _ = bt.apply_symbol_mode(source_df, symbol_mode, symbol_method)
            if symbol_df.empty:
                raise SystemExit("No rows found after auto symbol selection.")
            selected_range_df = symbol_df[(symbol_df.index >= start_time) & (symbol_df.index <= end_time)]
            if selected_range_df.empty:
                raise SystemExit("No rows found in the selected range after auto symbol selection.")
            symbol_distribution = selected_range_df["symbol"].value_counts().to_dict()
            symbol = auto_label
        else:
            preferred_symbol = bt.CONFIG.get("TARGET_SYMBOL")
            symbol = bt.choose_symbol(source_df, preferred_symbol)
            symbol_df = source_df[source_df["symbol"] == symbol]
            if symbol_df.empty:
                raise SystemExit("No rows found for the selected symbol.")
            selected_range_df = symbol_df[(symbol_df.index >= start_time) & (symbol_df.index <= end_time)]
            if selected_range_df.empty:
                raise SystemExit("No rows found in the selected range for the selected symbol.")
            symbol_distribution = selected_range_df["symbol"].value_counts().to_dict()

        symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    else:
        selected_range_df = source_df[(source_df.index >= start_time) & (source_df.index <= end_time)]
        if selected_range_df.empty:
            raise SystemExit("No rows found in the selected range.")
        symbol = "AUTO"

    source_attrs = getattr(source_df, "attrs", {}) or {}
    symbol_df = bt.attach_backtest_symbol_context(
        symbol_df,
        symbol,
        symbol_mode,
        source_key=source_attrs.get("source_cache_key"),
        source_label=source_attrs.get("source_label"),
        source_path=source_attrs.get("source_path"),
    )
    return symbol_df, symbol, symbol_distribution


def _resolve_optional_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"File not found: {path_arg}")


def _sync_calibrated_entry_model_from_bundle(bundle_path: Path) -> None:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8-sig"))
    entry_model = bundle.get("entry_policy_model", {})
    if not isinstance(entry_model, dict):
        raise SystemExit(f"Bundle has no entry_policy_model: {bundle_path}")

    runtime_cfg = CONFIG.setdefault("DE3_V4", {}).setdefault("runtime", {})
    execution_policy = runtime_cfg.setdefault("execution_policy", {})
    cal_model = execution_policy.setdefault("calibrated_entry_model", {})
    if not isinstance(cal_model, dict):
        raise SystemExit("DE3_V4.runtime.execution_policy.calibrated_entry_model is not a dict.")

    minimums = entry_model.get("minimums", {}) if isinstance(entry_model.get("minimums"), dict) else {}
    score_components = (
        entry_model.get("score_components", {})
        if isinstance(entry_model.get("score_components"), dict)
        else {}
    )
    scope_offsets = (
        entry_model.get("scope_threshold_offsets", {})
        if isinstance(entry_model.get("scope_threshold_offsets"), dict)
        else {}
    )

    cal_model["enabled"] = bool(entry_model.get("enabled", cal_model.get("enabled", True)))
    cal_model["selected_threshold"] = float(entry_model.get("selected_threshold", cal_model.get("selected_threshold", 0.0)) or 0.0)
    cal_model["min_variant_trades"] = int(minimums.get("min_variant_trades", cal_model.get("min_variant_trades", 25)) or 25)
    cal_model["min_lane_trades"] = int(minimums.get("min_lane_trades", cal_model.get("min_lane_trades", 120)) or 120)
    cal_model["allow_on_missing_stats"] = bool(minimums.get("allow_on_missing_stats", cal_model.get("allow_on_missing_stats", True)))
    cal_model["conservative_buffer"] = float(minimums.get("conservative_buffer", cal_model.get("conservative_buffer", 0.035)) or 0.035)
    if scope_offsets:
        cal_model["scope_threshold_offsets"] = dict(scope_offsets)
    for key, value in score_components.items():
        cal_model[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DE3-only backtests with no filters and optional decision export."
    )
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME, help="Parquet/CSV source path.")
    parser.add_argument("--start", required=True, help="Start datetime: YYYY-MM-DD or YYYY-MM-DD HH:MM")
    parser.add_argument("--end", required=True, help="End datetime: YYYY-MM-DD or YYYY-MM-DD HH:MM")
    parser.add_argument(
        "--symbol-mode",
        default=str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single"),
        help="single, auto_by_day, or another supported backtest symbol mode.",
    )
    parser.add_argument(
        "--symbol-method",
        default=str(bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume"),
        help="Auto symbol selection method.",
    )
    parser.add_argument(
        "--pre-roll-days",
        type=int,
        default=0,
        help="Optional warmup history to include before start_time during symbol preparation.",
    )
    parser.add_argument("--report-dir", default="backtest_reports", help="Output directory for reports.")
    parser.add_argument(
        "--summary-out",
        default="",
        help="Optional JSON path for a compact run summary.",
    )
    parser.add_argument("--export-de3-decisions", action="store_true", help="Write DE3 decision CSV.")
    parser.add_argument("--de3-decisions-top-k", type=int, default=5, help="Decision export Top-K.")
    parser.add_argument(
        "--enable-post-run-gemini",
        action="store_true",
        help="Allow Gemini recommendation sidecar generation after the DE3 run.",
    )
    parser.add_argument(
        "--de3-decisions-out",
        default="reports/de3_decisions.csv",
        help="Decision export path. If left at default name, the engine will stamp it per run.",
    )
    parser.add_argument(
        "--bundle-path",
        default="",
        help="Override DE3_V4 bundle path for this run only.",
    )
    parser.add_argument(
        "--member-db-path",
        default="",
        help="Override DE3_V4 member DB path for this run only.",
    )
    parser.add_argument(
        "--sync-entry-model-from-bundle",
        action="store_true",
        help="Copy selected threshold and score-component weights from the bundle entry-policy model into runtime config for this run.",
    )
    parser.add_argument(
        "--entry-model-enabled",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override calibrated entry model enablement for this run.",
    )
    parser.add_argument(
        "--debug-de3-exceptions",
        action="store_true",
        help="Log DynamicEngine3Strategy exceptions instead of silently swallowing them in the DE3-only fast path.",
    )
    parser.add_argument(
        "--entry-model-threshold",
        type=float,
        default=None,
        help="Optional calibrated entry-model threshold override for this run.",
    )
    parser.add_argument(
        "--decision-side-apply-side-pattern",
        action="append",
        default=[],
        help="Override BACKTEST_DE3_V4_DECISION_SIDE_MODEL.apply_side_patterns for this run. Repeatable.",
    )
    parser.add_argument(
        "--decision-side-block-one-sided-on-no-prediction",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally block one-sided DE3 sessions when the decision-side model matches context but does not endorse the available side.",
    )
    parser.add_argument(
        "--exclude-variant-pattern",
        action="append",
        default=[],
        help="Add an extra DE3_V4.runtime.excluded_variant_patterns entry for this run. Repeatable.",
    )
    parser.add_argument(
        "--set-variant-size-multiplier",
        action="append",
        default=[],
        help="Override DE3_V4.runtime.confidence_tier_sizing.variant_size_multipliers for this run with VARIANT=VALUE. Repeatable.",
    )
    parser.add_argument(
        "--prune-rules-enabled",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override DE3_V4.runtime.prune_rules.enabled for this run.",
    )
    parser.add_argument(
        "--only-prune-rule",
        action="append",
        default=[],
        help="Enable only the named DE3 prune rule(s) for this run. Repeatable.",
    )
    parser.add_argument(
        "--signal-size-rules-enabled",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override DE3_V4.runtime.backtest_signal_size_rules.enabled for this run.",
    )
    parser.add_argument(
        "--only-signal-size-rule",
        action="append",
        default=[],
        help="Enable only the named DE3 backtest signal-size rule(s) for this run. Repeatable.",
    )
    parser.add_argument(
        "--core-enabled",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override DE3_V4.core.enabled for this run.",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=["default", "core_only", "core_plus_satellites", "satellites_only"],
        default="default",
        help="Optionally override DE3_V4.core.default_runtime_mode for this run.",
    )
    parser.add_argument(
        "--force-anchor-when-eligible",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override DE3_V4.core.force_anchor_when_eligible for this run.",
    )
    parser.add_argument(
        "--disable-context-policy-gate",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override DE3_V4.runtime.disable_context_policy_gate for this run.",
    )
    parser.add_argument(
        "--disable-context-veto-gate",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override DE3_V4.runtime.disable_context_veto_gate for this run.",
    )
    parser.add_argument(
        "--adaptive-policy-mode",
        choices=["default", "block", "shadow"],
        default="default",
        help="Optionally override DE3_ADAPTIVE_POLICY.mode for this run.",
    )
    parser.add_argument(
        "--adaptive-policy-model-path",
        default="",
        help="Optionally override DE3_ADAPTIVE_POLICY.model_path for this run.",
    )
    parser.add_argument(
        "--prefer-policy-ev-lcb",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override DE3_CANDIDATE_SELECTION.prefer_policy_ev_lcb for this run.",
    )
    parser.add_argument(
        "--regime-manifold-enabled",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override BACKTEST_REGIME_MANIFOLD.enabled for this run.",
    )
    parser.add_argument(
        "--regime-manifold-mode",
        choices=["default", "enforce", "shadow"],
        default="default",
        help="Optionally override BACKTEST_REGIME_MANIFOLD.mode for this run.",
    )
    parser.add_argument(
        "--intraday-regime-enabled",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override DE3_V4.runtime.backtest_intraday_regime_controller.enabled for this run.",
    )
    parser.add_argument(
        "--intraday-regime-mode",
        choices=["default", "block", "defensive", "block_defensive"],
        default="default",
        help="Optionally override DE3_V4.runtime.backtest_intraday_regime_controller.mode for this run.",
    )
    parser.add_argument(
        "--intraday-regime-session",
        action="append",
        default=[],
        help="Override DE3_V4.runtime.backtest_intraday_regime_controller.apply_sessions for this run. Repeatable.",
    )
    parser.add_argument(
        "--intraday-regime-apply-lane",
        action="append",
        default=[],
        help="Override DE3_V4.runtime.backtest_intraday_regime_controller.apply_lanes for this run. Repeatable.",
    )
    parser.add_argument(
        "--intraday-regime-enable-bullish-mirror",
        choices=["default", "true", "false"],
        default="default",
        help="Optionally override DE3_V4.runtime.backtest_intraday_regime_controller.enable_bullish_mirror for this run.",
    )
    parser.add_argument(
        "--intraday-regime-defensive-size-multiplier",
        type=float,
        default=None,
        help="Optional defensive size multiplier override for the DE3 intraday regime controller.",
    )
    parser.add_argument(
        "--intraday-regime-defensive-score-threshold",
        type=float,
        default=None,
        help="Optional defensive score threshold override for the DE3 intraday regime controller.",
    )
    parser.add_argument(
        "--intraday-regime-block-score-threshold",
        type=float,
        default=None,
        help="Optional block score threshold override for the DE3 intraday regime controller.",
    )
    parser.add_argument(
        "--intraday-regime-dominance-threshold",
        type=float,
        default=None,
        help="Optional dominance threshold override for the DE3 intraday regime controller.",
    )
    parser.add_argument(
        "--intraday-regime-block-dominance-threshold",
        type=float,
        default=None,
        help="Optional block-dominance threshold override for the DE3 intraday regime controller.",
    )
    args = parser.parse_args()

    source_path = _resolve_source(args.source)
    start_time = bt.parse_user_datetime(str(args.start), bt.NY_TZ, is_end=False)
    end_time = bt.parse_user_datetime(str(args.end), bt.NY_TZ, is_end=True)
    if start_time > end_time:
        raise SystemExit("Start must be before end.")

    symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        str(args.symbol_mode or "single").strip().lower(),
        str(args.symbol_method or "volume").strip().lower(),
        pre_roll_days=max(0, int(args.pre_roll_days or 0)),
    )

    cfg_backup = copy.deepcopy(CONFIG)
    bundle_path_text = ""
    member_db_path_text = ""
    try:
        if bool(args.enable_post_run_gemini):
            bt.BACKTEST_POST_RUN_CFG["enable_gemini_recommendation"] = True
            bt.BACKTEST_GEMINI_RECOMMENDER_ENABLED = True
        else:
            bt.BACKTEST_POST_RUN_CFG["enable_gemini_recommendation"] = False
            bt.BACKTEST_POST_RUN_CFG["allow_when_gemini_disabled"] = False
            bt.BACKTEST_GEMINI_RECOMMENDER_ENABLED = False

        if str(args.bundle_path or "").strip():
            bundle_path = _resolve_optional_path(str(args.bundle_path))
            bundle_path_text = str(bundle_path)
            CONFIG.setdefault("DE3_V4", {})["bundle_path"] = str(bundle_path)
            if bool(args.sync_entry_model_from_bundle):
                _sync_calibrated_entry_model_from_bundle(bundle_path)
        if str(args.member_db_path or "").strip():
            member_db_path = _resolve_optional_path(str(args.member_db_path))
            member_db_path_text = str(member_db_path)
            CONFIG.setdefault("DE3_V4", {})["member_db_path"] = str(member_db_path)

        if str(args.entry_model_enabled).strip().lower() != "default":
            cal_model = (
                (((CONFIG.get("DE3_V4") or {}).get("runtime") or {}).get("execution_policy") or {})
            ).get("calibrated_entry_model")
            if not isinstance(cal_model, dict):
                raise SystemExit("DE3_V4.runtime.execution_policy.calibrated_entry_model is missing.")
            cal_model["enabled"] = str(args.entry_model_enabled).strip().lower() == "true"
        if args.entry_model_threshold is not None:
            cal_model = (
                (((CONFIG.get("DE3_V4") or {}).get("runtime") or {}).get("execution_policy") or {})
            ).get("calibrated_entry_model")
            if not isinstance(cal_model, dict):
                raise SystemExit("DE3_V4.runtime.execution_policy.calibrated_entry_model is missing.")
            # An explicit CLI threshold should override any bundle-synced threshold.
            # Disable bundle-threshold sourcing for this run so the runtime reads
            # the value we are injecting below.
            cal_model["use_bundle_model"] = False
            cal_model["selected_threshold"] = float(args.entry_model_threshold)
        extra_excluded_variant_patterns = [
            str(v or "").strip()
            for v in (args.exclude_variant_pattern or [])
            if str(v or "").strip()
        ]
        if extra_excluded_variant_patterns:
            runtime_cfg = CONFIG.setdefault("DE3_V4", {}).setdefault("runtime", {})
            existing_patterns = runtime_cfg.get("excluded_variant_patterns", [])
            if not isinstance(existing_patterns, list):
                existing_patterns = list(existing_patterns) if existing_patterns else []
            merged_patterns = list(existing_patterns)
            for pattern in extra_excluded_variant_patterns:
                if pattern not in merged_patterns:
                    merged_patterns.append(pattern)
            runtime_cfg["excluded_variant_patterns"] = merged_patterns
        variant_multiplier_overrides = {}
        for raw_item in (args.set_variant_size_multiplier or []):
            text = str(raw_item or "").strip()
            if not text:
                continue
            variant_id, sep, mult_text = text.partition("=")
            variant_id = variant_id.strip()
            mult_text = mult_text.strip()
            if not sep or not variant_id or not mult_text:
                raise SystemExit(
                    f"Invalid --set-variant-size-multiplier value '{text}'. Use VARIANT=VALUE."
                )
            try:
                variant_multiplier_overrides[variant_id] = float(mult_text)
            except Exception as exc:
                raise SystemExit(
                    f"Invalid multiplier '{mult_text}' for variant '{variant_id}': {exc}"
                ) from exc
        if variant_multiplier_overrides:
            runtime_cfg = CONFIG.setdefault("DE3_V4", {}).setdefault("runtime", {})
            sizing_cfg = runtime_cfg.setdefault("confidence_tier_sizing", {})
            existing_variant_mults = sizing_cfg.get("variant_size_multipliers", {})
            if not isinstance(existing_variant_mults, dict):
                existing_variant_mults = {}
            merged_variant_mults = dict(existing_variant_mults)
            merged_variant_mults.update(variant_multiplier_overrides)
            sizing_cfg["variant_size_multipliers"] = merged_variant_mults
        decision_side_bt_cfg = CONFIG.setdefault("BACKTEST_DE3_V4_DECISION_SIDE_MODEL", {})
        if not isinstance(decision_side_bt_cfg, dict):
            raise SystemExit("BACKTEST_DE3_V4_DECISION_SIDE_MODEL is not a dict.")
        decision_side_patterns = [
            str(v or "").strip().lower()
            for v in (args.decision_side_apply_side_pattern or [])
            if str(v or "").strip()
        ]
        if decision_side_patterns:
            decision_side_bt_cfg["apply_side_patterns"] = decision_side_patterns
        if str(args.decision_side_block_one_sided_on_no_prediction).strip().lower() != "default":
            decision_side_bt_cfg["block_one_sided_on_no_prediction"] = (
                str(args.decision_side_block_one_sided_on_no_prediction).strip().lower() == "true"
            )
        if str(args.prune_rules_enabled).strip().lower() != "default":
            prune_cfg = (((CONFIG.get("DE3_V4") or {}).get("runtime") or {}).get("prune_rules"))
            if not isinstance(prune_cfg, dict):
                raise SystemExit("DE3_V4.runtime.prune_rules is missing.")
            prune_cfg["enabled"] = str(args.prune_rules_enabled).strip().lower() == "true"
        only_prune_rules = {
            str(v or "").strip()
            for v in (args.only_prune_rule or [])
            if str(v or "").strip()
        }
        if only_prune_rules:
            prune_cfg = (((CONFIG.get("DE3_V4") or {}).get("runtime") or {}).get("prune_rules"))
            if not isinstance(prune_cfg, dict):
                raise SystemExit("DE3_V4.runtime.prune_rules is missing.")
            rules = prune_cfg.get("rules", [])
            if not isinstance(rules, list):
                raise SystemExit("DE3_V4.runtime.prune_rules.rules is not a list.")
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                rule_name = str(rule.get("name", "") or "").strip()
                rule["enabled"] = rule_name in only_prune_rules
        if str(args.signal_size_rules_enabled).strip().lower() != "default":
            signal_size_cfg = (((CONFIG.get("DE3_V4") or {}).get("runtime") or {}).get("backtest_signal_size_rules"))
            if not isinstance(signal_size_cfg, dict):
                raise SystemExit("DE3_V4.runtime.backtest_signal_size_rules is missing.")
            signal_size_cfg["enabled"] = str(args.signal_size_rules_enabled).strip().lower() == "true"
        only_signal_size_rules = {
            str(v or "").strip()
            for v in (args.only_signal_size_rule or [])
            if str(v or "").strip()
        }
        if only_signal_size_rules:
            signal_size_cfg = (((CONFIG.get("DE3_V4") or {}).get("runtime") or {}).get("backtest_signal_size_rules"))
            if not isinstance(signal_size_cfg, dict):
                raise SystemExit("DE3_V4.runtime.backtest_signal_size_rules is missing.")
            rules = signal_size_cfg.get("rules", [])
            if not isinstance(rules, list):
                raise SystemExit("DE3_V4.runtime.backtest_signal_size_rules.rules is not a list.")
            for rule in rules:
                if not isinstance(rule, dict):
                    continue
                rule_name = str(rule.get("name", "") or "").strip()
                rule["enabled"] = rule_name in only_signal_size_rules
        core_cfg = CONFIG.setdefault("DE3_V4", {}).setdefault("core", {})
        if str(args.core_enabled).strip().lower() != "default":
            core_cfg["enabled"] = str(args.core_enabled).strip().lower() == "true"
        if str(args.runtime_mode).strip().lower() != "default":
            core_cfg["default_runtime_mode"] = str(args.runtime_mode).strip().lower()
        if str(args.force_anchor_when_eligible).strip().lower() != "default":
            core_cfg["force_anchor_when_eligible"] = (
                str(args.force_anchor_when_eligible).strip().lower() == "true"
            )
        runtime_cfg = CONFIG.setdefault("DE3_V4", {}).setdefault("runtime", {})
        if str(args.disable_context_policy_gate).strip().lower() != "default":
            runtime_cfg["disable_context_policy_gate"] = (
                str(args.disable_context_policy_gate).strip().lower() == "true"
            )
        if str(args.disable_context_veto_gate).strip().lower() != "default":
            runtime_cfg["disable_context_veto_gate"] = (
                str(args.disable_context_veto_gate).strip().lower() == "true"
            )
        intraday_regime_cfg = runtime_cfg.setdefault("backtest_intraday_regime_controller", {})
        if not isinstance(intraday_regime_cfg, dict):
            raise SystemExit("DE3_V4.runtime.backtest_intraday_regime_controller is missing.")
        if str(args.intraday_regime_enabled).strip().lower() != "default":
            intraday_regime_cfg["enabled"] = (
                str(args.intraday_regime_enabled).strip().lower() == "true"
            )
        if str(args.intraday_regime_mode).strip().lower() != "default":
            intraday_regime_cfg["mode"] = str(args.intraday_regime_mode).strip().lower()
        intraday_regime_sessions = [
            str(v or "").strip().upper()
            for v in (args.intraday_regime_session or [])
            if str(v or "").strip()
        ]
        if intraday_regime_sessions:
            intraday_regime_cfg["apply_sessions"] = intraday_regime_sessions
        intraday_regime_lanes = [
            str(v or "").strip()
            for v in (args.intraday_regime_apply_lane or [])
            if str(v or "").strip()
        ]
        if intraday_regime_lanes:
            intraday_regime_cfg["apply_lanes"] = intraday_regime_lanes
        if str(args.intraday_regime_enable_bullish_mirror).strip().lower() != "default":
            intraday_regime_cfg["enable_bullish_mirror"] = (
                str(args.intraday_regime_enable_bullish_mirror).strip().lower() == "true"
            )
        if args.intraday_regime_defensive_size_multiplier is not None:
            intraday_regime_cfg["defensive_size_multiplier"] = float(
                args.intraday_regime_defensive_size_multiplier
            )
        if args.intraday_regime_defensive_score_threshold is not None:
            intraday_regime_cfg["defensive_score_threshold"] = float(
                args.intraday_regime_defensive_score_threshold
            )
        if args.intraday_regime_block_score_threshold is not None:
            intraday_regime_cfg["block_score_threshold"] = float(
                args.intraday_regime_block_score_threshold
            )
        if args.intraday_regime_dominance_threshold is not None:
            intraday_regime_cfg["dominance_threshold"] = float(
                args.intraday_regime_dominance_threshold
            )
        if args.intraday_regime_block_dominance_threshold is not None:
            intraday_regime_cfg["block_dominance_threshold"] = float(
                args.intraday_regime_block_dominance_threshold
            )
        adaptive_policy_cfg = CONFIG.setdefault("DE3_ADAPTIVE_POLICY", {})
        if str(args.adaptive_policy_mode).strip().lower() != "default":
            adaptive_policy_cfg["mode"] = str(args.adaptive_policy_mode).strip().lower()
        if str(args.adaptive_policy_model_path or "").strip():
            adaptive_policy_cfg["model_path"] = str(
                _resolve_optional_path(str(args.adaptive_policy_model_path))
            )
        candidate_selection_cfg = CONFIG.setdefault("DE3_CANDIDATE_SELECTION", {})
        if str(args.prefer_policy_ev_lcb).strip().lower() != "default":
            candidate_selection_cfg["prefer_policy_ev_lcb"] = (
                str(args.prefer_policy_ev_lcb).strip().lower() == "true"
            )
        bt_manifold_cfg = CONFIG.setdefault("BACKTEST_REGIME_MANIFOLD", {})
        if str(args.regime_manifold_enabled).strip().lower() != "default":
            bt_manifold_cfg["enabled"] = (
                str(args.regime_manifold_enabled).strip().lower() == "true"
            )
        if str(args.regime_manifold_mode).strip().lower() != "default":
            bt_manifold_cfg["mode"] = str(args.regime_manifold_mode).strip().lower()
        if bool(args.debug_de3_exceptions):
            CONFIG["BACKTEST_DE3_DEBUG_EXCEPTIONS"] = True

        selected_filters = set()
        if bool((CONFIG.get("BACKTEST_REGIME_MANIFOLD", {}) or {}).get("enabled", False)):
            selected_filters.add("RegimeManifold")

        stats = bt.run_backtest(
            symbol_df,
            start_time,
            end_time,
            enabled_strategies={"DynamicEngine3Strategy"},
            enabled_filters=selected_filters,
            export_de3_decisions=bool(args.export_de3_decisions),
            de3_decisions_top_k=max(1, int(args.de3_decisions_top_k or 1)),
            de3_decisions_out=str(args.de3_decisions_out),
        )
    finally:
        CONFIG.clear()
        CONFIG.update(cfg_backup)
    stats["symbol_mode"] = str(args.symbol_mode or "").strip().lower()
    if symbol_distribution:
        stats["symbol_distribution"] = symbol_distribution
    if bundle_path_text:
        stats["de3_bundle_path"] = bundle_path_text
    if member_db_path_text:
        stats["de3_member_db_path"] = member_db_path_text

    report_dir = Path(args.report_dir).expanduser()
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_path = bt.save_backtest_report(stats, symbol, start_time, end_time, output_dir=report_dir)
    monte_carlo_path = report_path.with_name(f"{report_path.stem}_monte_carlo.json")
    monte_carlo_summary = {}
    if monte_carlo_path.exists():
        try:
            monte_carlo_payload = json.loads(monte_carlo_path.read_text(encoding="utf-8"))
            if isinstance(monte_carlo_payload.get("summary", {}), dict):
                monte_carlo_summary = dict(monte_carlo_payload.get("summary", {}))
        except Exception:
            monte_carlo_summary = {}

    summary_payload = {
        "report_path": str(report_path),
        "monte_carlo_path": str(monte_carlo_path) if monte_carlo_path.exists() else None,
        "symbol": str(symbol),
        "bundle_path": str(bundle_path_text) if bundle_path_text else None,
        "member_db_path": str(member_db_path_text) if member_db_path_text else None,
        "symbol_mode": str(args.symbol_mode or "").strip().lower(),
        "pre_roll_days": int(max(0, int(args.pre_roll_days or 0))),
        "selected_strategies": ["DynamicEngine3Strategy"],
        "selected_filters": stats.get("selection", {}).get("filters", []),
        "summary": dict(stats.get("summary", {})) if isinstance(stats.get("summary", {}), dict) else {},
        "monte_carlo": monte_carlo_summary,
    }
    summary_path = None
    if str(args.summary_out or "").strip():
        summary_path = Path(str(args.summary_out)).expanduser()
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"report={report_path}")
    if summary_path is not None:
        print(f"summary_out={summary_path}")
    print(f"symbol={symbol}")
    print("selected_strategies=['DynamicEngine3Strategy']")
    print(f"selected_filters={stats.get('selection', {}).get('filters', [])}")
    if bundle_path_text:
        print(f"de3_bundle_path={bundle_path_text}")
    if member_db_path_text:
        print(f"de3_member_db_path={member_db_path_text}")
    print(f"entry_model_override={args.entry_model_enabled}")
    print(f"entry_model_threshold_override={args.entry_model_threshold}")
    print(f"decision_side_apply_side_patterns={args.decision_side_apply_side_pattern}")
    print(
        "decision_side_block_one_sided_on_no_prediction_override="
        f"{args.decision_side_block_one_sided_on_no_prediction}"
    )
    print(f"extra_excluded_variant_patterns={args.exclude_variant_pattern}")
    print(f"variant_size_multiplier_overrides={args.set_variant_size_multiplier}")
    print(f"prune_rules_enabled_override={args.prune_rules_enabled}")
    print(f"only_prune_rule={args.only_prune_rule}")
    print(f"signal_size_rules_enabled_override={args.signal_size_rules_enabled}")
    print(f"only_signal_size_rule={args.only_signal_size_rule}")
    print(f"core_enabled_override={args.core_enabled}")
    print(f"runtime_mode_override={args.runtime_mode}")
    print(f"force_anchor_when_eligible_override={args.force_anchor_when_eligible}")
    print(f"disable_context_policy_gate_override={args.disable_context_policy_gate}")
    print(f"disable_context_veto_gate_override={args.disable_context_veto_gate}")
    print(f"intraday_regime_enabled_override={args.intraday_regime_enabled}")
    print(f"intraday_regime_mode_override={args.intraday_regime_mode}")
    print(f"intraday_regime_sessions_override={args.intraday_regime_session}")
    print(f"intraday_regime_apply_lanes_override={args.intraday_regime_apply_lane}")
    print(f"intraday_regime_enable_bullish_mirror_override={args.intraday_regime_enable_bullish_mirror}")
    print(
        "intraday_regime_threshold_overrides="
        f"{{'defensive_size_multiplier': {args.intraday_regime_defensive_size_multiplier}, "
        f"'defensive_score_threshold': {args.intraday_regime_defensive_score_threshold}, "
        f"'block_score_threshold': {args.intraday_regime_block_score_threshold}, "
        f"'dominance_threshold': {args.intraday_regime_dominance_threshold}, "
        f"'block_dominance_threshold': {args.intraday_regime_block_dominance_threshold}}}"
    )
    print(f"adaptive_policy_mode_override={args.adaptive_policy_mode}")
    print(f"adaptive_policy_model_path_override={args.adaptive_policy_model_path}")
    print(f"prefer_policy_ev_lcb_override={args.prefer_policy_ev_lcb}")
    print(f"regime_manifold_enabled_override={args.regime_manifold_enabled}")
    print(f"regime_manifold_mode_override={args.regime_manifold_mode}")
    print(f"pre_roll_days={int(max(0, int(args.pre_roll_days or 0)))}")
    print(f"sync_entry_model_from_bundle={bool(args.sync_entry_model_from_bundle)}")
    print(f"debug_de3_exceptions={bool(args.debug_de3_exceptions)}")
    print(f"post_run_gemini_enabled={bool(args.enable_post_run_gemini)}")
    print(f"equity={stats.get('equity')}")
    print(f"trades={stats.get('trades')}")
    print(f"winrate={stats.get('winrate')}")
    print(f"max_drawdown={stats.get('max_drawdown')}")
    decisions_meta = stats.get("de3_decisions_export", {}) or {}
    export_path = decisions_meta.get("path")
    if export_path:
        print(f"de3_decisions={export_path}")


if __name__ == "__main__":
    main()
