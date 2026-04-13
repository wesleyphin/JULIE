import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt


def _resolve_source(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Data file not found: {path_arg}")


def _prepare_symbol_df(source_path: Path, start_time, end_time, symbol_mode: str, symbol_method: str):
    base_dir = ROOT
    df = bt.load_csv_cached(source_path, cache_dir=base_dir / "cache", use_cache=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AetherFlow-only backtests with configurable artifacts.")
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
    parser.add_argument("--model-file", default="model_aetherflow_fullrange_v2.pkl")
    parser.add_argument("--thresholds-file", default="aetherflow_thresholds_fullrange_v2.json")
    parser.add_argument("--metrics-file", default="aetherflow_metrics_fullrange_v2.json")
    parser.add_argument("--threshold-override", type=float, default=None)
    parser.add_argument("--min-confidence-override", type=float, default=None)
    parser.add_argument("--allow-regime", action="append", default=None, help="Optional regime allowlist. Repeats allowed.")
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

    af_cfg = bt.CONFIG.setdefault("AETHERFLOW_STRATEGY", {})
    af_cfg["enabled_backtest"] = True
    af_cfg["model_file"] = str(args.model_file)
    af_cfg["thresholds_file"] = str(args.thresholds_file)
    af_cfg["metrics_file"] = str(args.metrics_file)
    af_cfg["log_evals"] = False
    if args.threshold_override is not None:
        af_cfg["min_confidence"] = float(args.threshold_override)
    elif args.min_confidence_override is not None:
        af_cfg["min_confidence"] = float(args.min_confidence_override)
    if args.allow_regime:
        allow = {str(x).strip().upper() for x in args.allow_regime if str(x).strip()}
        blocked = {"TREND_GEODESIC", "CHOP_SPIRAL", "DISPERSED", "ROTATIONAL_TURBULENCE"} - allow
        af_cfg["hazard_block_regimes"] = sorted(blocked)

    stats = bt.run_backtest(
        symbol_df,
        start_time,
        end_time,
        enabled_strategies={"AetherFlowStrategy"},
        enabled_filters=set(),
    )
    stats["symbol_mode"] = str(args.symbol_mode or "").strip().lower()
    if symbol_distribution:
        stats["symbol_distribution"] = symbol_distribution

    report_dir = Path(args.report_dir).expanduser()
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_path = bt.save_backtest_report(stats, symbol, start_time, end_time, output_dir=report_dir)

    print(f"report={report_path}")
    print(f"symbol={symbol}")
    print("selected_strategies=['AetherFlowStrategy']")
    print("selected_filters=[]")
    print(f"equity={stats.get('equity')}")
    print(f"trades={stats.get('trades')}")
    print(f"winrate={stats.get('winrate')}")
    print(f"max_drawdown={stats.get('max_drawdown')}")


if __name__ == "__main__":
    main()
