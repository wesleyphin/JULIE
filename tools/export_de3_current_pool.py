import argparse
import copy
import json
import sys
from pathlib import Path

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


def _prepare_symbol_df(source_path: Path, start_time, end_time, symbol_mode: str, symbol_method: str):
    df = bt.load_csv_cached(source_path, cache_dir=ROOT / "cache", use_cache=True)
    if df.empty:
        raise SystemExit("No rows found in the source file.")

    source_df = df[df.index <= end_time]
    if source_df.empty:
        raise SystemExit("No rows found before the requested end time.")

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
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
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
    cal_model["selected_threshold"] = float(
        entry_model.get("selected_threshold", cal_model.get("selected_threshold", 0.0)) or 0.0
    )
    cal_model["min_variant_trades"] = int(
        minimums.get("min_variant_trades", cal_model.get("min_variant_trades", 25)) or 25
    )
    cal_model["min_lane_trades"] = int(
        minimums.get("min_lane_trades", cal_model.get("min_lane_trades", 120)) or 120
    )
    cal_model["allow_on_missing_stats"] = bool(
        minimums.get("allow_on_missing_stats", cal_model.get("allow_on_missing_stats", True))
    )
    cal_model["conservative_buffer"] = float(
        minimums.get("conservative_buffer", cal_model.get("conservative_buffer", 0.035)) or 0.035
    )
    if scope_offsets:
        cal_model["scope_threshold_offsets"] = dict(scope_offsets)
    for key, value in score_components.items():
        cal_model[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the current DE3 candidate pool by backtesting with the calibrated entry model disabled."
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
    parser.add_argument("--report-dir", default="backtest_reports", help="Output directory for reports.")
    parser.add_argument(
        "--bundle-path",
        default="",
        help="Override DE3_V4 bundle path for this export run only.",
    )
    parser.add_argument(
        "--sync-entry-model-from-bundle",
        action="store_true",
        help="Copy threshold and score-component weights from the bundle entry-policy model into runtime config before disabling it for export.",
    )
    parser.add_argument(
        "--decisions-out",
        default="reports/de3_current_pool.csv",
        help="Decision export path. If left at default name, the engine will stamp it per run.",
    )
    parser.add_argument(
        "--de3-decisions-top-k",
        type=int,
        default=5,
        help="How many ranked DE3 candidates to keep per decision in the export. Higher values are useful for side-choice research.",
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
    )

    cfg_backup = copy.deepcopy(CONFIG)
    try:
        if str(args.bundle_path or "").strip():
            bundle_path = _resolve_optional_path(str(args.bundle_path))
            CONFIG.setdefault("DE3_V4", {})["bundle_path"] = str(bundle_path)
            if bool(args.sync_entry_model_from_bundle):
                _sync_calibrated_entry_model_from_bundle(bundle_path)

        cal_model = (
            (((CONFIG.get("DE3_V4") or {}).get("runtime") or {}).get("execution_policy") or {})
        ).get("calibrated_entry_model")
        if not isinstance(cal_model, dict):
            raise SystemExit("DE3_V4.runtime.execution_policy.calibrated_entry_model is missing.")
        cal_model["enabled"] = False

        stats = bt.run_backtest(
            symbol_df,
            start_time,
            end_time,
            enabled_strategies={"DynamicEngine3Strategy"},
            enabled_filters=set(),
            export_de3_decisions=True,
            de3_decisions_top_k=max(1, int(args.de3_decisions_top_k or 1)),
            de3_decisions_out=str(args.decisions_out),
        )
    finally:
        CONFIG.clear()
        CONFIG.update(cfg_backup)

    stats["symbol_mode"] = str(args.symbol_mode or "").strip().lower()
    if symbol_distribution:
        stats["symbol_distribution"] = symbol_distribution

    report_dir = Path(args.report_dir).expanduser()
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_path = bt.save_backtest_report(stats, symbol, start_time, end_time, output_dir=report_dir)

    print(f"report={report_path}")
    print(f"symbol={symbol}")
    print("selected_strategies=['DynamicEngine3Strategy']")
    print("selected_filters=[]")
    print("entry_model=disabled_for_export")
    if str(args.bundle_path or "").strip():
        print(f"de3_bundle_path={str(args.bundle_path).strip()}")
    print(f"equity={stats.get('equity')}")
    print(f"trades={stats.get('trades')}")
    print(f"winrate={stats.get('winrate')}")
    print(f"max_drawdown={stats.get('max_drawdown')}")
    print(f"de3_decisions_top_k={max(1, int(args.de3_decisions_top_k or 1))}")
    decisions_meta = stats.get("de3_decisions_export", {}) or {}
    export_path = decisions_meta.get("path")
    if export_path:
        print(f"de3_decisions={export_path}")
    trade_export_path = decisions_meta.get("trade_attribution_path")
    if trade_export_path:
        print(f"de3_trade_attribution={trade_export_path}")


if __name__ == "__main__":
    main()
