import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_base_cache import DEFAULT_FULL_MANIFOLD_BASE_FEATURES
import backtest_mes_et as bt
from tools.backtest_aetherflow_direct import (
    _build_signal_frame,
    _prepare_symbol_df,
    _resolve_source,
    _simulate,
)


def _resolve_path(path_text: str, default_relative: str) -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _test_bounds(test_years: list[int]) -> tuple[str, str]:
    years = sorted(int(y) for y in test_years)
    return f"{years[0]:04d}-01-01", f"{years[-1]:04d}-12-31"


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact direct AetherFlow backtests for walkforward folds.")
    parser.add_argument("--walkforward-report", required=True)
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME)
    parser.add_argument("--base-features", default=DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    parser.add_argument("--symbol-mode", default=str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single"))
    parser.add_argument("--symbol-method", default=str(bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume"))
    parser.add_argument("--history-buffer-days", type=int, default=14)
    parser.add_argument("--output-dir", default="backtest_reports/aetherflow_walkforward_direct")
    parser.add_argument("--use-horizon-time-stop", action="store_true")
    args = parser.parse_args()

    report_path = _resolve_path(str(args.walkforward_report), "")
    payload = json.loads(report_path.read_text())
    folds = payload.get("folds", []) or []
    if not folds:
        raise RuntimeError(f"No folds found in {report_path}")

    source_path = _resolve_source(str(args.source))
    base_features_path = _resolve_path(str(args.base_features), DEFAULT_FULL_MANIFOLD_BASE_FEATURES)
    output_dir = _resolve_path(str(args.output_dir), "backtest_reports/aetherflow_walkforward_direct")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for fold in folds:
        artifacts = fold.get("artifacts") or {}
        model_path = Path(str(artifacts.get("model_file", "") or ""))
        thresholds_path = Path(str(artifacts.get("thresholds_file", "") or ""))
        if not model_path.is_absolute():
            model_path = (ROOT / model_path).resolve()
        if not thresholds_path.is_absolute():
            thresholds_path = (ROOT / thresholds_path).resolve()
        if not model_path.exists() or not thresholds_path.exists():
            continue

        fold_name = str(fold.get("fold", "") or "fold")
        test_years = [int(y) for y in (fold.get("test_years") or [])]
        if not test_years:
            continue
        start_text, end_text = _test_bounds(test_years)
        start_time = bt.parse_user_datetime(start_text, bt.NY_TZ, is_end=False)
        end_time = bt.parse_user_datetime(end_text, bt.NY_TZ, is_end=True)
        symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
            source_path,
            start_time,
            end_time,
            str(args.symbol_mode or "single").strip().lower(),
            str(args.symbol_method or "volume").strip().lower(),
            int(args.history_buffer_days),
        )
        signals = _build_signal_frame(
            symbol_df=symbol_df,
            base_features_path=base_features_path,
            model_path=model_path,
            thresholds_path=thresholds_path,
            threshold_override=None,
            min_confidence_override=None,
            allow_setups=None,
            block_regimes={"ROTATIONAL_TURBULENCE"},
            allowed_session_ids={1, 2, 3},
        )
        result = _simulate(
            df=symbol_df,
            signals=signals,
            start_time=start_time,
            end_time=end_time,
            use_horizon_time_stop=bool(args.use_horizon_time_stop),
        )
        summary = result if isinstance(result, dict) else {}
        out = {
            "fold": fold_name,
            "test_years": test_years,
            "symbol": symbol,
            "symbol_distribution": symbol_distribution,
            "model_file": str(model_path),
            "thresholds_file": str(thresholds_path),
            "equity": float(summary.get("equity", 0.0) or 0.0),
            "trades": int(summary.get("trades", 0) or 0),
            "winrate": float(summary.get("winrate", 0.0) or 0.0),
            "max_drawdown": float(summary.get("max_drawdown", 0.0) or 0.0),
            "profit_factor": (
                None
                if summary.get("profit_factor") is None or not math.isfinite(float(summary.get("profit_factor", 0.0) or 0.0))
                else float(summary.get("profit_factor", 0.0) or 0.0)
            ),
            "gross_profit": float(summary.get("gross_profit", 0.0) or 0.0),
            "gross_loss": float(summary.get("gross_loss", 0.0) or 0.0),
        }
        rows.append(out)

    if not rows:
        raise RuntimeError("No fold artifacts were available to backtest.")

    summary_df = pd.DataFrame(rows)
    csv_path = output_dir / "walkforward_direct_summary.csv"
    json_path = output_dir / "walkforward_direct_summary.json"
    summary_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(_json_safe({"rows": rows}), indent=2))
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    print(f"folds={len(rows)}")
    print(f"total_equity={float(summary_df['equity'].sum())}")
    print(f"positive_folds={int((summary_df['equity'] > 0.0).sum())}")


if __name__ == "__main__":
    main()
