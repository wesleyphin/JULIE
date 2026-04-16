import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from ml_physics_legacy_experiment_strategy import MLPhysicsLegacyExperimentStrategy


def _resolve_source(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Data file not found: {path_arg}")


def _append_suffix(path_value: str, suffix: str) -> str:
    raw = str(path_value or "").strip()
    tag = str(suffix or "").strip()
    if not raw or not tag:
        return raw
    p = Path(raw)
    name = p.name
    if p.suffix:
        stem = p.stem
        if stem.endswith(tag):
            return raw
        new_name = f"{stem}{tag}{p.suffix}"
    else:
        if name.endswith(tag):
            return raw
        new_name = f"{name}{tag}"
    return str(p.with_name(new_name))


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


def _configure_legacy_experiment(
    *,
    artifact_dir: str,
    model_suffix: str,
    thresholds_file: str,
    metrics_file: str,
) -> None:
    experiment_cfg = dict(bt.CONFIG.get("ML_PHYSICS_LEGACY_EXPERIMENT", {}) or {})
    artifact_dir = str(artifact_dir or experiment_cfg.get("artifact_dir", "") or "").strip()
    thresholds_file = str(thresholds_file or experiment_cfg.get("thresholds_file", "") or "").strip()
    metrics_file = str(metrics_file or experiment_cfg.get("metrics_file", "") or "").strip()

    if model_suffix:
        thresholds_file = _append_suffix(thresholds_file, model_suffix)
        metrics_file = _append_suffix(metrics_file, model_suffix)

    experiment_cfg["artifact_dir"] = artifact_dir
    experiment_cfg["thresholds_file"] = thresholds_file
    experiment_cfg["metrics_file"] = metrics_file
    bt.CONFIG["ML_PHYSICS_LEGACY_EXPERIMENT"] = experiment_cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the legacy pre-dist MLPhysics joblib stack as an isolated experiment."
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
        "--report-dir",
        default="backtest_reports/legacy_mlphysics_experiment",
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--model-suffix",
        default="",
        help="Optional suffix appended to legacy threshold/metrics artifact names.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(
            ((bt.CONFIG.get("ML_PHYSICS_LEGACY_EXPERIMENT", {}) or {}).get("artifact_dir", "artifacts/ml_physics_legacy_experiment"))
            or "artifacts/ml_physics_legacy_experiment"
        ),
        help="Legacy experiment artifact directory.",
    )
    parser.add_argument(
        "--thresholds-file",
        default=str(
            ((bt.CONFIG.get("ML_PHYSICS_LEGACY_EXPERIMENT", {}) or {}).get("thresholds_file", "ml_physics_thresholds.json"))
            or "ml_physics_thresholds.json"
        ),
        help="Legacy experiment thresholds JSON.",
    )
    parser.add_argument(
        "--metrics-file",
        default=str(
            ((bt.CONFIG.get("ML_PHYSICS_LEGACY_EXPERIMENT", {}) or {}).get("metrics_file", "ml_physics_metrics.json"))
            or "ml_physics_metrics.json"
        ),
        help="Legacy experiment metrics JSON.",
    )
    args = parser.parse_args()

    _configure_legacy_experiment(
        artifact_dir=str(args.artifact_dir or "").strip(),
        model_suffix=str(args.model_suffix or "").strip(),
        thresholds_file=str(args.thresholds_file or "").strip(),
        metrics_file=str(args.metrics_file or "").strip(),
    )

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

    precompute_strategy = MLPhysicsLegacyExperimentStrategy()
    precomputed_df = precompute_strategy.build_precomputed_backtest_df(symbol_df)
    MLPhysicsLegacyExperimentStrategy.set_global_precomputed_backtest_df(precomputed_df)
    try:
        stats = bt.run_backtest(
            symbol_df,
            start_time,
            end_time,
            enabled_strategies={"MLPhysicsLegacyExperimentStrategy"},
            enabled_filters=set(),
        )
    finally:
        MLPhysicsLegacyExperimentStrategy.clear_global_precomputed_backtest_df()
    stats["symbol_mode"] = str(args.symbol_mode or "").strip().lower()
    stats["ml_physics_runtime_mode"] = "legacy_experiment"
    stats["ml_physics_legacy_experiment_artifact_dir"] = str(
        ((bt.CONFIG.get("ML_PHYSICS_LEGACY_EXPERIMENT", {}) or {}).get("artifact_dir", "")) or ""
    )
    stats["ml_physics_legacy_precomputed_rows"] = int(len(precomputed_df)) if hasattr(precomputed_df, "__len__") else 0
    stats["ml_physics_model_suffix"] = str(args.model_suffix or "").strip()
    stats["ml_physics_thresholds_file"] = str(
        ((bt.CONFIG.get("ML_PHYSICS_LEGACY_EXPERIMENT", {}) or {}).get("thresholds_file", "")) or ""
    )
    stats["ml_physics_metrics_file"] = str(
        ((bt.CONFIG.get("ML_PHYSICS_LEGACY_EXPERIMENT", {}) or {}).get("metrics_file", "")) or ""
    )
    if symbol_distribution:
        stats["symbol_distribution"] = symbol_distribution

    report_dir = Path(args.report_dir).expanduser()
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_path = bt.save_backtest_report(stats, symbol, start_time, end_time, output_dir=report_dir)

    print(f"report={report_path}")
    print(f"symbol={symbol}")
    print("selected_strategies=['MLPhysicsLegacyExperimentStrategy']")
    print("selected_filters=[]")
    print("ml_physics_runtime_mode=legacy_experiment")
    print(f"ml_physics_legacy_artifact_dir={((bt.CONFIG.get('ML_PHYSICS_LEGACY_EXPERIMENT', {}) or {}).get('artifact_dir', ''))}")
    print(f"ml_physics_legacy_precomputed_rows={stats.get('ml_physics_legacy_precomputed_rows')}")
    print(f"ml_physics_thresholds_file={((bt.CONFIG.get('ML_PHYSICS_LEGACY_EXPERIMENT', {}) or {}).get('thresholds_file', ''))}")
    print(f"ml_physics_metrics_file={((bt.CONFIG.get('ML_PHYSICS_LEGACY_EXPERIMENT', {}) or {}).get('metrics_file', ''))}")
    print(f"equity={stats.get('equity')}")
    print(f"trades={stats.get('trades')}")
    print(f"winrate={stats.get('winrate')}")
    print(f"max_drawdown={stats.get('max_drawdown')}")


if __name__ == "__main__":
    main()
