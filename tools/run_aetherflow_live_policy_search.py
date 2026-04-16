import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from aetherflow_features import build_feature_frame
from aetherflow_strategy import (
    AetherFlowStrategy,
    _regime_name_from_row,
    _selection_score,
    _coerce_session_allowlist,
    _coerce_string_allowlist,
    _coerce_upper_string_allowlist,
    _normalize_family_policies,
)
from tools.backtest_aetherflow_direct import _load_base_features, _prepare_symbol_df, _resolve_source, _simulate
from tools.run_aetherflow_deploy_policy_search import _load_eval_windows, _window_bounds
from tools.run_aetherflow_viability_suite import _bootstrap_metric_summary, _json_safe, _trade_counts, _variant_summary


def _resolve_path(path_text: str, default_relative: str = "") -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


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
        family_policies = _normalize_family_policies(raw.get("family_policies", {}))
        variants.append(
            {
                "name": name,
                "description": str(raw.get("description", "") or ""),
                "threshold_override": raw.get("threshold_override"),
                "min_confidence": raw.get("min_confidence"),
                "allowed_session_ids": _coerce_session_allowlist(raw.get("allowed_session_ids")),
                "allowed_setup_families": _coerce_string_allowlist(raw.get("allowed_setup_families")),
                "hazard_block_regimes": _coerce_upper_string_allowlist(raw.get("hazard_block_regimes")),
                "family_policies": family_policies,
            }
        )
    if not variants:
        raise RuntimeError(f"No usable variants found in {path}")
    return variants


def _json_safe_sets(value: Any):
    if isinstance(value, set):
        return sorted(_json_safe_sets(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _json_safe_sets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_sets(item) for item in value]
    return value


def _configure_strategy(
    *,
    model_file: Path,
    thresholds_file: Path,
    metrics_file: Path | None,
    variant: dict,
) -> AetherFlowStrategy:
    strategy = AetherFlowStrategy()
    strategy.model_path = Path(model_file)
    strategy.thresholds_path = Path(thresholds_file)
    strategy.metrics_path = Path(metrics_file) if metrics_file is not None else Path(strategy.metrics_path)
    strategy.model = None
    strategy.model_bundle = None
    strategy.model_loaded = False
    strategy.threshold_override = None
    strategy.threshold = 0.58
    strategy.family_policies = {}
    strategy.allowed_session_ids = None
    strategy.allowed_setup_families = None
    strategy.hazard_block_regimes = set()
    strategy._load_artifacts()
    if not strategy.model_loaded or strategy.model_bundle is None:
        raise RuntimeError(f"AetherFlow model failed to load: {model_file}")

    threshold_override = variant.get("threshold_override")
    if threshold_override is not None:
        strategy.threshold_override = float(threshold_override)
        strategy.threshold = float(threshold_override)

    min_confidence = variant.get("min_confidence")
    if min_confidence is not None:
        strategy.min_confidence = float(min_confidence)

    family_policies = dict(variant.get("family_policies", {}) or {})
    strategy.family_policies = family_policies
    allowed_setup_families = variant.get("allowed_setup_families")
    if allowed_setup_families:
        strategy.allowed_setup_families = set(allowed_setup_families)
    elif family_policies:
        strategy.allowed_setup_families = set(str(name) for name in family_policies.keys())
    else:
        strategy.allowed_setup_families = None

    allowed_session_ids = variant.get("allowed_session_ids")
    strategy.allowed_session_ids = set(allowed_session_ids) if allowed_session_ids else None

    hazard_block_regimes = variant.get("hazard_block_regimes")
    strategy.hazard_block_regimes = set(hazard_block_regimes) if hazard_block_regimes else set()
    strategy.log_evals = False
    return strategy


def _build_variant_signals(strategy: AetherFlowStrategy, base_features: pd.DataFrame) -> pd.DataFrame:
    candidate_rows: list[dict[str, Any]] = []
    for family_name in strategy._candidate_family_names():
        frame = build_feature_frame(
            base_features=base_features,
            preferred_setup_families={family_name},
        )
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        frame = frame.loc[
            (frame["setup_family"].astype(str) == str(family_name))
            & (pd.to_numeric(frame.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0)
        ].copy()
        if frame.empty:
            continue
        frame["aetherflow_confidence"] = strategy._compute_probabilities(frame)
        for ts, row in zip(pd.DatetimeIndex(frame.index), frame.to_dict("records")):
            row["manifold_regime_name"] = _regime_name_from_row(row)
            policy = strategy._policy_for_family(family_name, row) or {}
            row["selection_score"] = _selection_score(float(row.get("aetherflow_confidence", 0.0) or 0.0), policy)
            row["entry_mode"] = str(policy.get("entry_mode", "market_next_bar") or "market_next_bar")
            row["use_horizon_time_stop"] = bool(policy.get("use_horizon_time_stop", False))
            if policy.get("sl_mult_override") is not None:
                row["sl_mult_override"] = float(policy.get("sl_mult_override"))
            if policy.get("tp_mult_override") is not None:
                row["tp_mult_override"] = float(policy.get("tp_mult_override"))
            if policy.get("horizon_bars_override") is not None:
                row["horizon_bars_override"] = int(policy.get("horizon_bars_override"))
            early_exit_cfg = dict(policy.get("early_exit", {}) or {})
            if early_exit_cfg:
                row["early_exit_enabled"] = bool(early_exit_cfg.get("enabled", False))
                row["early_exit_exit_if_not_green_by"] = int(early_exit_cfg.get("exit_if_not_green_by", 0) or 0)
                row["early_exit_max_profit_crosses"] = int(early_exit_cfg.get("max_profit_crosses", 0) or 0)
            break_even_cfg = dict(policy.get("break_even", {}) or {})
            if break_even_cfg:
                row["break_even_enabled"] = bool(break_even_cfg.get("enabled", False))
                row["break_even_trigger_pct"] = float(break_even_cfg.get("trigger_pct", 0.0) or 0.0)
                row["break_even_buffer_ticks"] = int(break_even_cfg.get("buffer_ticks", 0) or 0)
                row["break_even_trail_pct"] = float(break_even_cfg.get("trail_pct", 0.0) or 0.0)
                row["break_even_activate_on_next_bar"] = bool(break_even_cfg.get("activate_on_next_bar", True))
            if strategy._row_block_reason(row) == "":
                row["_timestamp"] = ts
                candidate_rows.append(row)
    if not candidate_rows:
        return pd.DataFrame()
    merged = pd.DataFrame(candidate_rows)
    merged = merged.sort_values(
        by=["selection_score", "aetherflow_confidence", "setup_strength"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    merged = merged.drop_duplicates(subset=["_timestamp"], keep="first")
    index = pd.DatetimeIndex(pd.to_datetime(merged.pop("_timestamp")))
    merged.index = index
    return merged.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate live-equivalent AetherFlow family policies on fixed windows.")
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--base-features", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--thresholds-file", required=True)
    parser.add_argument("--metrics-file", default=None)
    parser.add_argument("--variants-file", required=True)
    parser.add_argument("--output-dir", default="backtest_reports/aetherflow_live_policy_search")
    parser.add_argument("--years", default="2024,2025,2026")
    parser.add_argument("--windows-file", default=None)
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--history-buffer-days", type=int, default=14)
    parser.add_argument("--mc-simulations", type=int, default=1000)
    parser.add_argument("--mc-seed", type=int, default=1337)
    parser.add_argument("--allow-same-side-add-ons", action="store_true")
    parser.add_argument("--max-same-side-legs", type=int, default=1)
    args = parser.parse_args()

    source_path = _resolve_source(str(args.source))
    base_features_path = _resolve_path(str(args.base_features))
    model_path = _resolve_path(str(args.model_file))
    thresholds_path = _resolve_path(str(args.thresholds_file))
    metrics_path = _resolve_path(str(args.metrics_file)) if args.metrics_file else None
    variants_path = _resolve_path(str(args.variants_file))
    output_dir = _resolve_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_windows = _load_eval_windows(years_text=str(args.years), windows_file=args.windows_file)
    variants = _load_variants(variants_path)

    window_cache: dict[str, dict[str, Any]] = {}
    for window in eval_windows:
        label = str(window.get("label", "") or "")
        start_time, end_time = _window_bounds(str(window["start"]), str(window["end"]))
        symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
            source_path,
            start_time,
            end_time,
            str(args.symbol_mode or "auto_by_day").strip().lower(),
            str(args.symbol_method or "volume").strip().lower(),
            int(args.history_buffer_days),
        )
        base_features = _load_base_features(
            base_features_path,
            pd.Timestamp(symbol_df.index.min()),
            pd.Timestamp(symbol_df.index.max()),
        )
        window_cache[label] = {
            "start_time": start_time,
            "end_time": end_time,
            "symbol_df": symbol_df,
            "symbol": symbol,
            "symbol_distribution": symbol_distribution,
            "base_features": base_features,
            "window": window,
        }

    results = []
    for variant in variants:
        strategy = _configure_strategy(
            model_file=model_path,
            thresholds_file=thresholds_path,
            metrics_file=metrics_path,
            variant=variant,
        )
        evaluation_results = []
        total_equity = 0.0
        total_trades = 0
        worst_max_drawdown = 0.0
        positive_windows = 0
        all_trade_log = []

        for window in eval_windows:
            label = str(window.get("label", "") or "")
            cached = window_cache[label]
            signals = _build_variant_signals(strategy, cached["base_features"])
            stats = _simulate(
                df=cached["symbol_df"],
                signals=signals,
                start_time=cached["start_time"],
                end_time=cached["end_time"],
                use_horizon_time_stop=False,
                allow_same_side_add_ons=bool(args.allow_same_side_add_ons),
                max_same_side_legs=int(args.max_same_side_legs),
            )
            trade_log = stats.get("trade_log", []) or []
            summary = _variant_summary(stats)
            evaluation = {
                "window_label": label,
                "start": str(window.get("start", "") or ""),
                "end": str(window.get("end", "") or ""),
                "symbol": cached["symbol"],
                "symbol_distribution": cached["symbol_distribution"],
                "signals": int(len(signals)),
                "summary": summary,
                "trade_families": _trade_counts(trade_log, "aetherflow_setup_family"),
                "trade_regimes": _trade_counts(trade_log, "aetherflow_regime"),
                "trade_sessions": _trade_counts(trade_log, "session"),
            }
            if "year" in window:
                evaluation["year"] = int(window["year"])
            evaluation_results.append(evaluation)
            total_equity += float(summary.get("equity", 0.0) or 0.0)
            total_trades += int(summary.get("trades", 0) or 0)
            worst_max_drawdown = max(worst_max_drawdown, float(summary.get("max_drawdown", 0.0) or 0.0))
            positive_windows += 1 if float(summary.get("equity", 0.0) or 0.0) > 0.0 else 0
            all_trade_log.extend(trade_log)

            print(
                f"[live-policy] {variant['name']} {label} "
                f"equity={summary['equity']:.2f} trades={summary['trades']} "
                f"pf={summary['profit_factor']:.3f}",
                flush=True,
            )

        bootstrap = _bootstrap_metric_summary(
            all_trade_log,
            simulations=max(1, int(args.mc_simulations)),
            seed=int(args.mc_seed),
        )
        aggregate = {
            "window_count": int(len(evaluation_results)),
            "positive_windows": int(positive_windows),
            "negative_windows": int(len(evaluation_results) - positive_windows),
            "year_count": int(len(evaluation_results)),
            "positive_years": int(positive_windows),
            "negative_years": int(len(evaluation_results) - positive_windows),
            "total_equity": float(round(total_equity, 2)),
            "total_trades": int(total_trades),
            "worst_max_drawdown": float(round(worst_max_drawdown, 2)),
            "mean_mc_day_bootstrap_prob_above_zero": float(
                bootstrap.get("net_pnl", {}).get("probability_above_threshold", 0.0) or 0.0
            ),
            "mean_mc_profit_factor": float(bootstrap.get("profit_factor", {}).get("mean", 0.0) or 0.0),
            "p05_mc_profit_factor": float(bootstrap.get("profit_factor", {}).get("p05", 0.0) or 0.0),
            "mean_mc_daily_sharpe": float(bootstrap.get("daily_sharpe", {}).get("mean", 0.0) or 0.0),
            "p95_mc_max_drawdown": float(bootstrap.get("max_drawdown", {}).get("p95", 0.0) or 0.0),
        }
        results.append(
            {
                "name": str(variant.get("name", "")),
                "description": str(variant.get("description", "")),
                "min_confidence": float(strategy.min_confidence),
                "threshold": float(strategy.threshold),
                "allowed_session_ids": sorted(int(item) for item in strategy.allowed_session_ids) if strategy.allowed_session_ids else [],
                "allowed_setup_families": sorted(str(item) for item in strategy.allowed_setup_families) if strategy.allowed_setup_families else [],
                "hazard_block_regimes": sorted(str(item) for item in strategy.hazard_block_regimes) if strategy.hazard_block_regimes else [],
                "family_policies": _json_safe_sets(_json_safe(dict(strategy.family_policies or {}))),
                "aggregate": aggregate,
                "evaluation_results": evaluation_results,
                "yearly_results": evaluation_results,
                "monte_carlo_trade_day_bootstrap": bootstrap,
            }
        )

    payload = {
        "created_at": datetime.now(bt.NY_TZ).isoformat(),
        "source": str(source_path),
        "base_features": str(base_features_path),
        "model_file": str(model_path),
        "thresholds_file": str(thresholds_path),
        "metrics_file": str(metrics_path) if metrics_path is not None else "",
        "variants_file": str(variants_path),
        "years": [item.get("year") for item in eval_windows if "year" in item],
        "evaluation_windows": eval_windows,
        "allow_same_side_add_ons": bool(args.allow_same_side_add_ons),
        "max_same_side_legs": int(args.max_same_side_legs),
        "variants": results,
    }
    out_path = output_dir / f"aetherflow_live_policy_search_{datetime.now(bt.NY_TZ).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"live_policy_report={out_path}")


if __name__ == "__main__":
    main()
