import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_base_cache import DEFAULT_FULL_MANIFOLD_BASE_FEATURES
import backtest_mes_et as bt
from aetherflow_features import build_feature_frame
from aetherflow_model_bundle import predict_bundle_probabilities
from aetherflow_strategy import REGIME_ID_TO_NAME
from tools.backtest_aetherflow_direct import (
    _load_base_features,
    _load_model_bundle,
    _prepare_symbol_df,
    _resolve_source,
    _simulate,
)
from tools.run_aetherflow_viability_suite import (
    _aggregate_variant_results,
    _bootstrap_metric_summary,
    _json_safe,
    _load_walkforward_report,
    _rule_match_mask,
    _resolve_fold_artifact_path,
    _resolve_path,
    _safe_float,
    _test_bounds,
    _trade_counts,
    _variant_summary,
)


def _normalize_int_list(value) -> list[int] | None:
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple, set)) else [value]
    out: list[int] = []
    for item in items:
        try:
            out.append(int(float(item)))
        except Exception:
            continue
    deduped = sorted({item for item in out})
    return deduped or None


def _normalize_str_list(value) -> list[str] | None:
    if value is None:
        return None
    items = value if isinstance(value, (list, tuple, set)) else [value]
    out = []
    for item in items:
        text = str(item or "").strip().upper()
        if text:
            out.append(text)
    deduped = sorted({item for item in out})
    return deduped or None


def _normalize_break_even_cfg(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("break_even must be an object when provided.")
    return {
        "enabled": bool(value.get("enabled", False)),
        "trigger_pct": max(0.0, _safe_float(value.get("trigger_pct"), 0.0)),
        "buffer_ticks": max(0, int(round(_safe_float(value.get("buffer_ticks"), 0.0)))),
        "trail_pct": max(0.0, _safe_float(value.get("trail_pct"), 0.0)),
        "activate_on_next_bar": bool(value.get("activate_on_next_bar", True)),
    }


def _normalize_early_exit_cfg(value) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RuntimeError("early_exit must be an object when provided.")
    return {
        "enabled": bool(value.get("enabled", False)),
        "exit_if_not_green_by": max(0, int(round(_safe_float(value.get("exit_if_not_green_by"), 0.0)))),
        "max_profit_crosses": max(0, int(round(_safe_float(value.get("max_profit_crosses"), 0.0)))),
    }


def _normalize_af_policy_mapping(raw, *, allow_match_fields: bool, allow_rules: bool) -> dict:
    if not isinstance(raw, dict):
        raise RuntimeError("aligned_flow policy must be an object.")
    out: dict[str, Any] = {}

    if "threshold" in raw:
        out["threshold"] = (_safe_float(raw.get("threshold"), float("nan")) if raw.get("threshold") is not None else None)
    if "allowed_session_ids" in raw:
        out["allowed_session_ids"] = _normalize_int_list(raw.get("allowed_session_ids"))
    if "allowed_regimes" in raw:
        out["allowed_regimes"] = _normalize_str_list(raw.get("allowed_regimes"))
    if "blocked_regimes" in raw:
        out["blocked_regimes"] = _normalize_str_list(raw.get("blocked_regimes"))
    if "max_abs_vwap_dist_atr" in raw:
        out["max_abs_vwap_dist_atr"] = (
            None if raw.get("max_abs_vwap_dist_atr") is None else max(0.0, _safe_float(raw.get("max_abs_vwap_dist_atr"), 0.0))
        )
    if "max_directional_vwap_dist_atr" in raw:
        out["max_directional_vwap_dist_atr"] = (
            None if raw.get("max_directional_vwap_dist_atr") is None else _safe_float(raw.get("max_directional_vwap_dist_atr"), 0.0)
        )
    if "min_d_alignment_3" in raw:
        out["min_d_alignment_3"] = (
            None if raw.get("min_d_alignment_3") is None else _safe_float(raw.get("min_d_alignment_3"), 0.0)
        )
    if "min_d_coherence_3" in raw:
        out["min_d_coherence_3"] = (
            None if raw.get("min_d_coherence_3") is None else _safe_float(raw.get("min_d_coherence_3"), 0.0)
        )
    if "min_setup_strength" in raw:
        out["min_setup_strength"] = (
            None if raw.get("min_setup_strength") is None else _safe_float(raw.get("min_setup_strength"), 0.0)
        )
    if "min_alignment_pct" in raw:
        out["min_alignment_pct"] = (
            None if raw.get("min_alignment_pct") is None else _safe_float(raw.get("min_alignment_pct"), 0.0)
        )
    if "min_smoothness_pct" in raw:
        out["min_smoothness_pct"] = (
            None if raw.get("min_smoothness_pct") is None else _safe_float(raw.get("min_smoothness_pct"), 0.0)
        )
    if "max_stress_pct" in raw:
        out["max_stress_pct"] = (
            None if raw.get("max_stress_pct") is None else _safe_float(raw.get("max_stress_pct"), 1.0)
        )
    if "max_flow_mag_slow" in raw:
        out["max_flow_mag_slow"] = (
            None if raw.get("max_flow_mag_slow") is None else _safe_float(raw.get("max_flow_mag_slow"), 1.0)
        )
    if "entry_mode" in raw:
        out["entry_mode"] = str(raw.get("entry_mode", "market_next_bar") or "market_next_bar").strip().lower()
    if "entry_pullback_atr" in raw:
        out["entry_pullback_atr"] = max(0.0, _safe_float(raw.get("entry_pullback_atr"), 0.0))
    if "entry_window_bars" in raw:
        out["entry_window_bars"] = max(1, int(round(_safe_float(raw.get("entry_window_bars"), 2.0))))
    if "sl_mult_override" in raw:
        out["sl_mult_override"] = (
            None if raw.get("sl_mult_override") is None else max(0.1, _safe_float(raw.get("sl_mult_override"), 1.0))
        )
    if "tp_mult_override" in raw:
        out["tp_mult_override"] = (
            None if raw.get("tp_mult_override") is None else max(0.1, _safe_float(raw.get("tp_mult_override"), 1.0))
        )
    if "horizon_bars_override" in raw:
        out["horizon_bars_override"] = (
            None if raw.get("horizon_bars_override") is None else max(1, int(round(_safe_float(raw.get("horizon_bars_override"), 18.0))))
        )
    if "use_horizon_time_stop" in raw:
        out["use_horizon_time_stop"] = bool(raw.get("use_horizon_time_stop", False))
    if "break_even" in raw:
        out["break_even"] = _normalize_break_even_cfg(raw.get("break_even")) or {}
    if "early_exit" in raw:
        out["early_exit"] = _normalize_early_exit_cfg(raw.get("early_exit")) or {}

    if allow_match_fields:
        if "name" in raw and str(raw.get("name", "") or "").strip():
            out["name"] = str(raw.get("name", "") or "").strip()
        if "match_session_ids" in raw:
            out["match_session_ids"] = _normalize_int_list(raw.get("match_session_ids"))
        if "match_regimes" in raw:
            out["match_regimes"] = _normalize_str_list(raw.get("match_regimes"))

    if allow_rules:
        raw_rules = raw.get("policy_rules", raw.get("rules"))
        if raw_rules is not None:
            if not isinstance(raw_rules, list):
                raise RuntimeError("aligned_flow policy_rules must be a list when provided.")
            out["policy_rules"] = [
                _normalize_af_policy_mapping(item, allow_match_fields=True, allow_rules=False)
                for item in raw_rules
                if isinstance(item, dict)
            ]

    return out


def _merge_af_policy_layers(*layers: dict) -> dict:
    merged: dict[str, Any] = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for key, value in layer.items():
            if key in {"policy_rules", "rules", "match_session_ids", "match_regimes", "name"}:
                continue
            if isinstance(value, dict):
                merged[key] = dict(value)
            elif isinstance(value, list):
                merged[key] = list(value)
            else:
                merged[key] = value
    return merged


def _apply_aligned_flow_policy_frame(
    features: pd.DataFrame,
    *,
    policy: dict,
    default_threshold: float,
    rule_name: str | None,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    scoped = features.copy()

    allowed_session_ids = policy.get("allowed_session_ids")
    if allowed_session_ids:
        session_series = pd.to_numeric(scoped.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
        scoped = scoped.loc[session_series.isin(allowed_session_ids)]
    if scoped.empty:
        return pd.DataFrame()

    allowed_regimes = policy.get("allowed_regimes")
    if allowed_regimes:
        scoped = scoped.loc[scoped["manifold_regime_name"].isin(allowed_regimes)]
    blocked_regimes = policy.get("blocked_regimes")
    if blocked_regimes:
        scoped = scoped.loc[~scoped["manifold_regime_name"].isin(blocked_regimes)]
    if scoped.empty:
        return pd.DataFrame()

    threshold = float(policy.get("threshold") if policy.get("threshold") is not None else default_threshold)
    scoped["policy_threshold"] = float(threshold)
    scoped = scoped.loc[scoped["aetherflow_confidence"] >= float(threshold)]
    if scoped.empty:
        return pd.DataFrame()

    max_abs_vwap_dist_atr = policy.get("max_abs_vwap_dist_atr")
    if max_abs_vwap_dist_atr is not None:
        vwap_dist = pd.to_numeric(scoped.get("vwap_dist_atr"), errors="coerce").fillna(0.0).abs()
        scoped = scoped.loc[vwap_dist <= float(max_abs_vwap_dist_atr)]
    if scoped.empty:
        return pd.DataFrame()

    max_directional_vwap_dist_atr = policy.get("max_directional_vwap_dist_atr")
    if max_directional_vwap_dist_atr is not None:
        dir_vwap = (
            pd.to_numeric(scoped.get("candidate_side"), errors="coerce").fillna(0.0)
            * pd.to_numeric(scoped.get("vwap_dist_atr"), errors="coerce").fillna(0.0)
        )
        scoped = scoped.loc[dir_vwap <= float(max_directional_vwap_dist_atr)]
    if scoped.empty:
        return pd.DataFrame()

    min_d_alignment_3 = policy.get("min_d_alignment_3")
    if min_d_alignment_3 is not None:
        d_alignment_3 = pd.to_numeric(scoped.get("d_alignment_3"), errors="coerce").fillna(0.0)
        scoped = scoped.loc[d_alignment_3 >= float(min_d_alignment_3)]
    if scoped.empty:
        return pd.DataFrame()

    min_d_coherence_3 = policy.get("min_d_coherence_3")
    if min_d_coherence_3 is not None:
        d_coherence_3 = pd.to_numeric(scoped.get("d_coherence_3"), errors="coerce").fillna(0.0)
        scoped = scoped.loc[d_coherence_3 >= float(min_d_coherence_3)]
    if scoped.empty:
        return pd.DataFrame()

    min_setup_strength = policy.get("min_setup_strength")
    if min_setup_strength is not None:
        setup_strength = pd.to_numeric(scoped.get("setup_strength"), errors="coerce").fillna(0.0)
        scoped = scoped.loc[setup_strength >= float(min_setup_strength)]
    if scoped.empty:
        return pd.DataFrame()

    min_alignment_pct = policy.get("min_alignment_pct")
    if min_alignment_pct is not None:
        alignment_pct = pd.to_numeric(scoped.get("manifold_alignment_pct"), errors="coerce").fillna(0.0)
        scoped = scoped.loc[alignment_pct >= float(min_alignment_pct)]
    if scoped.empty:
        return pd.DataFrame()

    min_smoothness_pct = policy.get("min_smoothness_pct")
    if min_smoothness_pct is not None:
        smoothness_pct = pd.to_numeric(scoped.get("manifold_smoothness_pct"), errors="coerce").fillna(0.0)
        scoped = scoped.loc[smoothness_pct >= float(min_smoothness_pct)]
    if scoped.empty:
        return pd.DataFrame()

    max_stress_pct = policy.get("max_stress_pct")
    if max_stress_pct is not None:
        stress_pct = pd.to_numeric(scoped.get("manifold_stress_pct"), errors="coerce").fillna(1.0)
        scoped = scoped.loc[stress_pct <= float(max_stress_pct)]
    if scoped.empty:
        return pd.DataFrame()

    max_flow_mag_slow = policy.get("max_flow_mag_slow")
    if max_flow_mag_slow is not None:
        flow_mag_slow = pd.to_numeric(scoped.get("flow_mag_slow"), errors="coerce").fillna(999.0)
        scoped = scoped.loc[flow_mag_slow <= float(max_flow_mag_slow)]
    if scoped.empty:
        return pd.DataFrame()

    scoped["entry_mode"] = str(policy.get("entry_mode", "market_next_bar") or "market_next_bar")
    scoped["entry_pullback_atr"] = float(policy.get("entry_pullback_atr", 0.0) or 0.0)
    scoped["entry_window_bars"] = int(policy.get("entry_window_bars", 2) or 2)
    scoped["use_horizon_time_stop"] = bool(policy.get("use_horizon_time_stop", False))

    if policy.get("sl_mult_override") is not None:
        scoped["sl_mult_override"] = float(policy["sl_mult_override"])
    if policy.get("tp_mult_override") is not None:
        scoped["tp_mult_override"] = float(policy["tp_mult_override"])
    if policy.get("horizon_bars_override") is not None:
        scoped["horizon_bars_override"] = int(policy["horizon_bars_override"])

    break_even_cfg = policy.get("break_even") or {}
    if break_even_cfg:
        scoped["break_even_enabled"] = bool(break_even_cfg.get("enabled", False))
        scoped["break_even_trigger_pct"] = float(break_even_cfg.get("trigger_pct", 0.0) or 0.0)
        scoped["break_even_buffer_ticks"] = int(break_even_cfg.get("buffer_ticks", 0) or 0)
        scoped["break_even_trail_pct"] = float(break_even_cfg.get("trail_pct", 0.0) or 0.0)
        scoped["break_even_activate_on_next_bar"] = bool(break_even_cfg.get("activate_on_next_bar", True))

    early_exit_cfg = policy.get("early_exit") or {}
    if early_exit_cfg:
        scoped["early_exit_enabled"] = bool(early_exit_cfg.get("enabled", False))
        scoped["early_exit_exit_if_not_green_by"] = int(early_exit_cfg.get("exit_if_not_green_by", 0) or 0)
        scoped["early_exit_max_profit_crosses"] = int(early_exit_cfg.get("max_profit_crosses", 0) or 0)

    if rule_name:
        scoped["policy_rule_name"] = str(rule_name)
    return scoped


def _load_variants(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_variants = payload.get("variants", payload if isinstance(payload, list) else []) or []
    if not isinstance(raw_variants, list) or not raw_variants:
        raise RuntimeError(f"No variants found in {path}")
    variants: list[dict] = []
    for raw in raw_variants:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "") or "").strip()
        if not name:
            continue
        variant = {
            "name": name,
            "description": str(raw.get("description", "") or ""),
        }
        variant.update(_normalize_af_policy_mapping(raw, allow_match_fields=False, allow_rules=True))
        variants.append(variant)
    if not variants:
        raise RuntimeError(f"No usable variants found in {path}")
    return variants


def _prepare_aligned_flow_signals(
    *,
    variant: dict,
    base_features: pd.DataFrame,
    model,
    feature_columns: list[str],
    default_threshold: float,
) -> pd.DataFrame:
    features = build_feature_frame(
        base_features=base_features,
        preferred_setup_families={"aligned_flow"},
    )
    if features.empty:
        return pd.DataFrame()
    features = features.loc[
        (features["setup_family"].astype(str) == "aligned_flow")
        & (pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0)
    ].copy()
    if features.empty:
        return pd.DataFrame()

    features["aetherflow_confidence"] = predict_bundle_probabilities(model, features)
    regime_names = pd.to_numeric(features.get("manifold_regime_id"), errors="coerce").fillna(-1).round().astype(int)
    features["manifold_regime_name"] = regime_names.map(REGIME_ID_TO_NAME).fillna("").astype(str).str.upper()
    base_policy = {
        key: value
        for key, value in dict(variant or {}).items()
        if key not in {"name", "description", "policy_rules", "rules"}
    }
    policy_rules = list(dict(variant or {}).get("policy_rules", []) or [])

    scoped_frames: list[pd.DataFrame] = []
    remaining = features.copy()
    for rule in policy_rules:
        match_mask = _rule_match_mask(remaining, rule)
        if not bool(match_mask.any()):
            continue
        matched = remaining.loc[match_mask].copy()
        remaining = remaining.loc[~match_mask].copy()
        effective_policy = _merge_af_policy_layers(base_policy, rule)
        scoped = _apply_aligned_flow_policy_frame(
            matched,
            policy=effective_policy,
            default_threshold=float(default_threshold),
            rule_name=str(rule.get("name", "") or "").strip() or None,
        )
        if not scoped.empty:
            scoped_frames.append(scoped)

    base_scoped = _apply_aligned_flow_policy_frame(
        remaining,
        policy=base_policy,
        default_threshold=float(default_threshold),
        rule_name=None,
    )
    if not base_scoped.empty:
        scoped_frames.append(base_scoped)
    if not scoped_frames:
        return pd.DataFrame()

    features = pd.concat(scoped_frames, axis=0).sort_index()
    features = features.sort_values(
        by=["aetherflow_confidence", "setup_strength"],
        ascending=[False, False],
        kind="mergesort",
    )
    features = features.loc[~features.index.duplicated(keep="first")]
    return features.sort_index()


def _run_fold_variant(
    *,
    fold: dict,
    variant: dict,
    source_path: Path,
    base_features_path: Path,
    symbol_mode: str,
    symbol_method: str,
    history_buffer_days: int,
    output_dir: Path,
    mc_simulations: int,
    mc_seed: int,
    skip_report_export: bool,
) -> dict:
    artifacts = fold.get("artifacts") or {}
    model_path = _resolve_fold_artifact_path(str(artifacts.get("model_file", "") or ""))
    thresholds_path = _resolve_fold_artifact_path(str(artifacts.get("thresholds_file", "") or ""))
    if not model_path.exists() or not thresholds_path.exists():
        raise RuntimeError(f"Missing fold artifacts for {fold.get('fold')}: {model_path} / {thresholds_path}")

    test_years = [int(y) for y in (fold.get("test_years") or [])]
    start_text, end_text = _test_bounds(test_years)
    start_time = bt.parse_user_datetime(start_text, bt.NY_TZ, is_end=False)
    end_time = bt.parse_user_datetime(end_text, bt.NY_TZ, is_end=True)
    symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        str(symbol_mode or "single").strip().lower(),
        str(symbol_method or "volume").strip().lower(),
        int(history_buffer_days),
    )
    model, feature_columns, default_threshold, _ = _load_model_bundle(model_path, thresholds_path)
    base_features = _load_base_features(base_features_path, pd.Timestamp(symbol_df.index.min()), pd.Timestamp(symbol_df.index.max()))

    signals = _prepare_aligned_flow_signals(
        variant=variant,
        base_features=base_features,
        model=model,
        feature_columns=list(feature_columns),
        default_threshold=float(default_threshold),
    )
    stats = _simulate(
        df=symbol_df,
        signals=signals,
        start_time=start_time,
        end_time=end_time,
        use_horizon_time_stop=bool(variant.get("use_horizon_time_stop", False)),
    )
    if symbol_distribution:
        stats["symbol_distribution"] = symbol_distribution

    fold_dir = output_dir / str(variant["name"]) / str(fold["fold"])
    fold_dir.mkdir(parents=True, exist_ok=True)
    report_path: Path | None = None
    if not bool(skip_report_export):
        report_path = bt.save_backtest_report(
            stats,
            symbol,
            start_time,
            end_time,
            output_dir=fold_dir,
        )
    trade_log = stats.get("trade_log", []) or []
    mc_trade_order = bt._build_monte_carlo_summary(
        trade_log,
        stats,
        simulations=max(1, int(mc_simulations)),
        seed=int(mc_seed),
        starting_balance=float(bt.BACKTEST_MONTE_CARLO_START_BALANCE),
    )
    mc_day_bootstrap = _bootstrap_metric_summary(
        trade_log,
        simulations=max(1, int(mc_simulations)),
        seed=int(mc_seed),
    )
    result = {
        "fold": str(fold.get("fold", "")),
        "test_years": test_years,
        "model_file": str(model_path),
        "thresholds_file": str(thresholds_path),
        "symbol": symbol,
        "symbol_distribution": symbol_distribution,
        "signals": int(len(signals)),
        "summary": _variant_summary(stats),
        "trade_entry_modes": _trade_counts(trade_log, "aetherflow_entry_mode"),
        "trade_regimes": _trade_counts(trade_log, "aetherflow_regime"),
        "trade_sessions": _trade_counts(trade_log, "session"),
        "monte_carlo_trade_order": mc_trade_order,
        "monte_carlo_trade_day_bootstrap": mc_day_bootstrap,
        "report_path": str(report_path) if report_path is not None else "",
        "break_even_armed_trades": int(stats.get("break_even_armed_trades", 0) or 0),
        "early_exit_closes": int(stats.get("early_exit_closes", 0) or 0),
    }
    lightweight_path = fold_dir / "summary.json"
    lightweight_path.write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    result["summary_path"] = str(lightweight_path)
    print(
        f"[af-method] {variant['name']} {fold.get('fold')} trades={result['summary']['trades']} "
        f"equity={result['summary']['equity']:.2f} pf={result['summary']['profit_factor']:.3f} "
        f"mc_day_p>0={_safe_float(((mc_day_bootstrap.get('net_pnl', {}) or {}).get('probability_above_threshold')), 0.0):.3f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run focused aligned-flow method-search OOS backtests with Monte Carlo summaries."
    )
    parser.add_argument(
        "--walkforward-report",
        default="artifacts/aetherflow_walkforward_fullrange_v2_2024plus_seedfix/walkforward_report.json",
    )
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME)
    parser.add_argument("--base-features", default=DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    parser.add_argument(
        "--variants-file",
        default="configs/aetherflow_aligned_flow_method_candidates.json",
    )
    parser.add_argument("--variants", default="all")
    parser.add_argument("--folds", default="all")
    parser.add_argument(
        "--symbol-mode",
        default=str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single"),
    )
    parser.add_argument(
        "--symbol-method",
        default=str(bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume"),
    )
    parser.add_argument("--history-buffer-days", type=int, default=14)
    parser.add_argument(
        "--output-dir",
        default="backtest_reports/aetherflow_aligned_flow_method_search",
    )
    parser.add_argument("--mc-simulations", type=int, default=500)
    parser.add_argument("--mc-seed", type=int, default=1337)
    parser.add_argument("--skip-report-export", action="store_true")
    args = parser.parse_args()

    source_path = _resolve_source(str(args.source))
    base_features_path = _resolve_path(str(args.base_features), DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    walkforward_report = _resolve_path(str(args.walkforward_report), "artifacts/aetherflow_walkforward_fullrange_v2_2024plus_seedfix/walkforward_report.json")
    variants_file = _resolve_path(str(args.variants_file), "configs/aetherflow_aligned_flow_method_candidates.json")
    output_dir = _resolve_path(str(args.output_dir), "backtest_reports/aetherflow_aligned_flow_method_search")
    output_dir.mkdir(parents=True, exist_ok=True)

    walkforward_payload = _load_walkforward_report(walkforward_report)
    folds = list(walkforward_payload.get("folds", []) or [])
    if str(args.folds).strip().lower() != "all":
        wanted_folds = {item.strip() for item in str(args.folds).split(",") if item.strip()}
        folds = [fold for fold in folds if str(fold.get("fold", "")) in wanted_folds]
    if not folds:
        raise RuntimeError("No matching folds selected.")

    variants = _load_variants(variants_file)
    if str(args.variants).strip().lower() != "all":
        wanted_variants = {item.strip() for item in str(args.variants).split(",") if item.strip()}
        variants = [variant for variant in variants if variant["name"] in wanted_variants]
    if not variants:
        raise RuntimeError("No matching variants selected.")

    created_at = datetime.now(bt.NY_TZ).strftime("%Y%m%d_%H%M%S")
    suite_results: list[dict] = []
    for variant in variants:
        fold_results = []
        for fold in folds:
            fold_results.append(
                _run_fold_variant(
                    fold=fold,
                    variant=variant,
                    source_path=source_path,
                    base_features_path=base_features_path,
                    symbol_mode=str(args.symbol_mode),
                    symbol_method=str(args.symbol_method),
                    history_buffer_days=int(args.history_buffer_days),
                    output_dir=output_dir,
                    mc_simulations=int(args.mc_simulations),
                    mc_seed=int(args.mc_seed),
                    skip_report_export=bool(args.skip_report_export),
                )
            )
        suite_results.append(
            {
                "name": str(variant["name"]),
                "description": str(variant.get("description", "") or ""),
                "settings": _json_safe(dict(variant)),
                "aggregate": _aggregate_variant_results(fold_results),
                "fold_results": fold_results,
            }
        )

    suite_payload = {
        "created_at": datetime.now(bt.NY_TZ).isoformat(),
        "walkforward_report": str(walkforward_report),
        "source": str(source_path),
        "base_features": str(base_features_path),
        "variants_file": str(variants_file),
        "folds": [str(fold.get("fold", "")) for fold in folds],
        "variants": suite_results,
    }
    suite_path = output_dir / f"aetherflow_aligned_flow_method_search_{created_at}.json"
    suite_path.write_text(json.dumps(_json_safe(suite_payload), indent=2), encoding="utf-8")
    print(f"suite_report={suite_path}", flush=True)


if __name__ == "__main__":
    main()
