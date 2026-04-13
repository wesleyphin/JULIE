import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from tools.backtest_aetherflow_direct import (
    _load_base_features,
    _load_model_bundle,
    _prepare_symbol_df,
    _resolve_source,
    _simulate,
)
from tools.run_aetherflow_aligned_flow_method_search import _prepare_aligned_flow_signals
from tools.run_aetherflow_combined_book_strength_search import (
    _annotate_selection_score,
    _load_variants,
    _merge_variant_signals,
)
from tools.run_aetherflow_viability_suite import (
    _bootstrap_metric_summary,
    _json_safe,
    _prepare_family_signals,
    _trade_counts,
    _variant_summary,
)


def _resolve_path(path_text: str, default_relative: str = "") -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_time = bt.parse_user_datetime(f"{int(year):04d}-01-01", bt.NY_TZ, is_end=False)
    end_label = "2026-01-26" if int(year) == 2026 else f"{int(year):04d}-12-31"
    end_time = bt.parse_user_datetime(end_label, bt.NY_TZ, is_end=True)
    return start_time, end_time


def _window_bounds(start_text: str, end_text: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    return (
        bt.parse_user_datetime(str(start_text), bt.NY_TZ, is_end=False),
        bt.parse_user_datetime(str(end_text), bt.NY_TZ, is_end=True),
    )


def _load_eval_windows(*, years_text: str, windows_file: str | None) -> list[dict]:
    if windows_file:
        path = _resolve_path(str(windows_file))
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_windows = payload.get("windows", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_windows, list):
            raise RuntimeError(f"windows_file must contain a list or a dict with 'windows': {path}")
        windows: list[dict] = []
        for idx, raw in enumerate(raw_windows):
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("label", "") or "").strip() or f"window_{idx + 1}"
            start = str(raw.get("start", "") or "").strip()
            end = str(raw.get("end", "") or "").strip()
            if not start or not end:
                raise RuntimeError(f"Window '{label}' in {path} is missing start/end.")
            windows.append({"label": label, "start": start, "end": end})
        if not windows:
            raise RuntimeError(f"No usable windows found in {path}")
        return windows

    years = [int(item.strip()) for item in str(years_text).split(",") if str(item).strip()]
    if not years:
        raise RuntimeError("No evaluation windows selected.")
    windows = []
    for year in years:
        end_label = "2026-01-26" if int(year) == 2026 else f"{int(year):04d}-12-31"
        windows.append(
            {
                "label": str(int(year)),
                "start": f"{int(year):04d}-01-01",
                "end": end_label,
                "year": int(year),
            }
        )
    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Search corrected AetherFlow deploy-model policies on full-bar outrights data.")
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--base-features", required=True)
    parser.add_argument("--model-file", default="model_aetherflow_deploy_2026oos.pkl")
    parser.add_argument("--thresholds-file", default="aetherflow_thresholds_deploy_2026oos.json")
    parser.add_argument("--variants-file", required=True)
    parser.add_argument("--output-dir", default="backtest_reports/aetherflow_deploy_policy_search")
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
    variants_path = _resolve_path(str(args.variants_file))
    output_dir = _resolve_path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_windows = _load_eval_windows(years_text=str(args.years), windows_file=args.windows_file)

    model, feature_columns, default_threshold, _ = _load_model_bundle(model_path, thresholds_path)
    variants = _load_variants(variants_path)
    results = []

    for variant in variants:
        evaluation_results = []
        total_equity = 0.0
        total_trades = 0
        worst_max_drawdown = 0.0
        positive_windows = 0
        all_trade_log = []

        for window in eval_windows:
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

            signal_frames = []
            live_pair_signal_count = 0
            if bool(variant.get("include_live_pair", True)):
                for family_name, policy in (variant.get("live_pair_policies") or {}).items():
                    frame = _prepare_family_signals(
                        family_name=str(family_name),
                        policy=dict(policy or {}),
                        base_features=base_features,
                        model=model,
                        feature_columns=list(feature_columns),
                        default_threshold=float(default_threshold),
                    )
                    if not frame.empty:
                        frame = _annotate_selection_score(frame, score_bias=0.0, score_scale=1.0)
                        live_pair_signal_count += int(len(frame))
                        signal_frames.append(frame)

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
                allow_same_side_add_ons=bool(args.allow_same_side_add_ons),
                max_same_side_legs=int(args.max_same_side_legs),
            )
            trade_log = stats.get("trade_log", []) or []
            af_trade_log = [
                trade
                for trade in trade_log
                if str(trade.get("aetherflow_setup_family", "") or "") == "aligned_flow"
            ]
            af_trade_pnl = float(sum(float(trade.get("pnl_net", 0.0) or 0.0) for trade in af_trade_log))
            summary = _variant_summary(stats)

            evaluation = {
                "window_label": str(window.get("label", "") or ""),
                "start": str(window.get("start", "") or ""),
                "end": str(window.get("end", "") or ""),
                "symbol": symbol,
                "symbol_distribution": symbol_distribution,
                "signals": int(len(signals)),
                "live_pair_signals": int(live_pair_signal_count),
                "aligned_flow_signals": int(aligned_flow_signal_count),
                "summary": summary,
                "trade_families": _trade_counts(trade_log, "aetherflow_setup_family"),
                "trade_regimes": _trade_counts(trade_log, "aetherflow_regime"),
                "trade_sessions": _trade_counts(trade_log, "session"),
                "aligned_flow_trade_count": int(len(af_trade_log)),
                "aligned_flow_trade_pnl": float(round(af_trade_pnl, 2)),
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
                f"[deploy] {variant['name']} {evaluation['window_label']} "
                f"equity={summary['equity']:.2f} trades={summary['trades']} "
                f"af_trades={evaluation['aligned_flow_trade_count']} af_pnl={evaluation['aligned_flow_trade_pnl']:.2f}"
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
        "variants_file": str(variants_path),
        "years": [item.get("year") for item in eval_windows if "year" in item],
        "evaluation_windows": eval_windows,
        "allow_same_side_add_ons": bool(args.allow_same_side_add_ons),
        "max_same_side_legs": int(args.max_same_side_legs),
        "variants": results,
    }
    out_path = output_dir / f"aetherflow_deploy_policy_search_{datetime.now(bt.NY_TZ).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"deploy_report={out_path}")


if __name__ == "__main__":
    main()
