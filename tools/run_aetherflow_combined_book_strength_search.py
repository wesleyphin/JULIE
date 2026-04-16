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

from aetherflow_base_cache import DEFAULT_FULL_MANIFOLD_BASE_FEATURES
import backtest_mes_et as bt
from tools.backtest_aetherflow_direct import (
    _load_base_features,
    _load_model_bundle,
    _prepare_symbol_df,
    _resolve_source,
    _simulate,
)
from tools.run_aetherflow_aligned_flow_method_search import (
    _normalize_af_policy_mapping,
    _normalize_break_even_cfg,
    _normalize_early_exit_cfg,
    _normalize_int_list,
    _normalize_str_list,
    _prepare_aligned_flow_signals,
)
from tools.run_aetherflow_viability_suite import (
    _aggregate_variant_results,
    _bootstrap_metric_summary,
    _json_safe,
    _load_walkforward_report,
    _normalize_family_policies,
    _prepare_family_signals,
    _resolve_fold_artifact_path,
    _resolve_path,
    _safe_float,
    _test_bounds,
    _trade_counts,
    _variant_summary,
)


DEFAULT_LIVE_PAIR_POLICIES = {
    "compression_release": {
        "threshold": 0.55,
        "allowed_session_ids": [1, 2, 3],
        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
    },
    "transition_burst": {
        "threshold": 0.55,
        "allowed_session_ids": [1, 2, 3],
        "blocked_regimes": ["ROTATIONAL_TURBULENCE"],
    },
}


def _normalize_live_pair_policies(value) -> dict[str, dict]:
    normalized = _normalize_family_policies(
        DEFAULT_LIVE_PAIR_POLICIES if value is None else value
    )
    out = {
        str(name): dict(policy or {})
        for name, policy in normalized.items()
        if str(name) in {"compression_release", "transition_burst"}
    }
    if not out:
        raise RuntimeError("Combined-book live_pair_policies must include compression_release and/or transition_burst.")
    return out


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
        include_live_pair = bool(raw.get("include_live_pair", True))
        include_aligned_flow = bool(raw.get("include_aligned_flow", False))
        variant = {
            "name": name,
            "description": str(raw.get("description", "") or ""),
            "include_live_pair": include_live_pair,
            "include_aligned_flow": include_aligned_flow,
            "score_bias": _safe_float(raw.get("score_bias"), 0.0),
            "score_scale": _safe_float(raw.get("score_scale"), 1.0),
        }
        if include_live_pair:
            variant["live_pair_policies"] = _normalize_live_pair_policies(raw.get("live_pair_policies"))
        if include_aligned_flow:
            variant.update(_normalize_af_policy_mapping(raw, allow_match_fields=False, allow_rules=True))
        if not include_live_pair and not include_aligned_flow:
            raise RuntimeError(f"Variant {name} must include at least one of live pair or aligned flow.")
        variants.append(variant)
    if not variants:
        raise RuntimeError(f"No usable variants found in {path}")
    return variants


def _merge_variant_signals(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not usable:
        return pd.DataFrame()
    merged = pd.concat(usable, axis=0).sort_index()
    if "selection_score" not in merged.columns:
        merged["selection_score"] = pd.to_numeric(merged.get("aetherflow_confidence"), errors="coerce").fillna(0.0)
    merged["aetherflow_confidence"] = pd.to_numeric(merged.get("aetherflow_confidence"), errors="coerce").fillna(0.0)
    merged["setup_strength"] = pd.to_numeric(merged.get("setup_strength"), errors="coerce").fillna(0.0)
    merged = merged.sort_values(
        by=["selection_score", "aetherflow_confidence", "setup_strength"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    merged = merged.loc[~merged.index.duplicated(keep="first")]
    return merged.sort_index()


def _annotate_selection_score(frame: pd.DataFrame, *, score_bias: float, score_scale: float) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    conf = pd.to_numeric(out.get("aetherflow_confidence"), errors="coerce").fillna(0.0)
    out["selection_score"] = (conf * float(score_scale)) + float(score_bias)
    return out


def _run_fold_variant(
    *,
    fold: dict,
    variant: dict,
    source_path: Path,
    base_features_path: Path,
    symbol_mode: str,
    symbol_method: str,
    history_buffer_days: int,
    mc_simulations: int,
    mc_seed: int,
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

    signal_frames: list[pd.DataFrame] = []
    live_pair_signal_count = 0
    if bool(variant.get("include_live_pair", True)):
        for family_name, policy in (variant.get("live_pair_policies") or DEFAULT_LIVE_PAIR_POLICIES).items():
            live_frame = _prepare_family_signals(
                family_name=str(family_name),
                policy=dict(policy or {}),
                base_features=base_features,
                model=model,
                feature_columns=list(feature_columns),
                default_threshold=float(default_threshold),
            )
            if not live_frame.empty:
                live_frame = _annotate_selection_score(
                    live_frame,
                    score_bias=0.0,
                    score_scale=1.0,
                )
                live_pair_signal_count += int(len(live_frame))
                signal_frames.append(live_frame)

    aligned_flow_signal_count = 0
    if bool(variant.get("include_aligned_flow", False)):
        af_frame = _prepare_aligned_flow_signals(
            variant=dict(variant),
            base_features=base_features,
            model=model,
            feature_columns=list(feature_columns),
            default_threshold=float(default_threshold),
        )
        if not af_frame.empty:
            af_frame = _annotate_selection_score(
                af_frame,
                score_bias=float(variant.get("score_bias", 0.0) or 0.0),
                score_scale=float(variant.get("score_scale", 1.0) or 1.0),
            )
            aligned_flow_signal_count = int(len(af_frame))
            signal_frames.append(af_frame)

    signals = _merge_variant_signals(signal_frames)
    stats = _simulate(
        df=symbol_df,
        signals=signals,
        start_time=start_time,
        end_time=end_time,
        use_horizon_time_stop=False,
    )
    if symbol_distribution:
        stats["symbol_distribution"] = symbol_distribution

    trade_log = stats.get("trade_log", []) or []
    af_trade_log = [
        trade
        for trade in trade_log
        if str(trade.get("aetherflow_setup_family", "") or "") == "aligned_flow"
    ]
    af_trade_pnl = float(sum(_safe_float(trade.get("pnl_net"), 0.0) for trade in af_trade_log))
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
        "symbol": symbol,
        "symbol_distribution": symbol_distribution,
        "signals": int(len(signals)),
        "live_pair_signals": int(live_pair_signal_count),
        "aligned_flow_signals": int(aligned_flow_signal_count),
        "summary": _variant_summary(stats),
        "trade_families": _trade_counts(trade_log, "aetherflow_setup_family"),
        "trade_regimes": _trade_counts(trade_log, "aetherflow_regime"),
        "trade_sessions": _trade_counts(trade_log, "session"),
        "aligned_flow_trade_count": int(len(af_trade_log)),
        "aligned_flow_trade_pnl": float(round(af_trade_pnl, 2)),
        "monte_carlo_trade_order": mc_trade_order,
        "monte_carlo_trade_day_bootstrap": mc_day_bootstrap,
    }
    print(
        f"[combined] {variant['name']} {fold.get('fold')} "
        f"equity={result['summary']['equity']:.2f} "
        f"trades={result['summary']['trades']} "
        f"af_trades={result['aligned_flow_trade_count']} "
        f"af_pnl={result['aligned_flow_trade_pnl']:.2f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact AetherFlow combined-book OOS search for live pair plus aligned-flow variants."
    )
    parser.add_argument(
        "--walkforward-report",
        default="artifacts/aetherflow_walkforward_fullrange_v2_2024plus_seedfix/walkforward_report.json",
    )
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME)
    parser.add_argument("--base-features", default=DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    parser.add_argument(
        "--variants-file",
        default="configs/aetherflow_combined_book_strength_candidates.json",
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
    parser.add_argument("--mc-simulations", type=int, default=1000)
    parser.add_argument("--mc-seed", type=int, default=1337)
    args = parser.parse_args()

    source_path = _resolve_source(str(args.source))
    base_features_path = _resolve_path(str(args.base_features), DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    walkforward_report = _resolve_path(
        str(args.walkforward_report),
        "artifacts/aetherflow_walkforward_fullrange_v2_2024plus_seedfix/walkforward_report.json",
    )
    variants_file = _resolve_path(
        str(args.variants_file),
        "configs/aetherflow_combined_book_strength_candidates.json",
    )
    output_dir = _resolve_path(
        str(args.output_dir),
        "backtest_reports/aetherflow_aligned_flow_method_search",
    )
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
                    mc_simulations=int(args.mc_simulations),
                    mc_seed=int(args.mc_seed),
                )
            )
        suite_results.append(
            {
                "name": str(variant["name"]),
                "description": str(variant.get("description", "") or ""),
                "include_live_pair": bool(variant.get("include_live_pair", True)),
                "include_aligned_flow": bool(variant.get("include_aligned_flow", False)),
                "live_pair_policies": (
                    {}
                    if not bool(variant.get("include_live_pair", True))
                    else _json_safe(dict(variant.get("live_pair_policies", {}) or {}))
                ),
                "aligned_flow_policy": (
                    {}
                    if not bool(variant.get("include_aligned_flow", False))
                    else {
                        key: _json_safe(val)
                        for key, val in variant.items()
                        if key
                        not in {
                            "name",
                            "description",
                            "include_live_pair",
                            "include_aligned_flow",
                            "live_pair_policies",
                        }
                    }
                ),
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
    suite_path = output_dir / f"af_combined_book_strength_search_{created_at}.json"
    suite_path.write_text(json.dumps(_json_safe(suite_payload), indent=2), encoding="utf-8")
    print(f"combined_report={suite_path}", flush=True)


if __name__ == "__main__":
    main()
