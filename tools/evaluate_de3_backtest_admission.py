from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from backtest_symbol_context import attach_backtest_symbol_context


EVAL_WINDOWS = [
    ("2022", "2022-01-01", "2022-12-31 23:59"),
    ("2023", "2023-01-01", "2023-12-31 23:59"),
    ("2024", "2024-01-01", "2024-12-31 23:59"),
    ("2025", "2025-01-01", "2025-12-31 23:59"),
]
WINDOW_BY_LABEL = {label: (label, start_raw, end_raw) for label, start_raw, end_raw in EVAL_WINDOWS}


def _load_symbol_df() -> Tuple[Any, Any]:
    source_path = ROOT / bt.DEFAULT_CSV_NAME
    print(f"Loading source: {source_path}", flush=True)
    df = bt.load_csv_cached(source_path, cache_dir=ROOT / "cache", use_cache=True)
    symbol_mode = str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single").lower()
    symbol_method = str(bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume").lower()
    if symbol_mode != "auto_by_day":
        raise RuntimeError(f"Expected auto_by_day symbol mode, got {symbol_mode!r}")
    symbol_df, auto_label, _ = bt.apply_symbol_mode(df, symbol_mode, symbol_method)
    if "symbol" in symbol_df.columns:
        symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    source_attrs = getattr(df, "attrs", {}) or {}
    symbol_df = attach_backtest_symbol_context(
        symbol_df,
        auto_label,
        symbol_mode,
        source_key=source_attrs.get("source_cache_key"),
        source_label=source_attrs.get("source_label"),
        source_path=source_attrs.get("source_path"),
    )
    return df, symbol_df


def _configure_break_even_only() -> None:
    de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
    runtime_cfg = (
        copy.deepcopy(de3_cfg.get("runtime", {}) or {})
        if isinstance(de3_cfg.get("runtime", {}), dict)
        else {}
    )
    trade_mgmt_cfg = (
        copy.deepcopy(runtime_cfg.get("trade_management", {}) or {})
        if isinstance(runtime_cfg.get("trade_management", {}), dict)
        else {}
    )
    break_even_cfg = (
        copy.deepcopy(trade_mgmt_cfg.get("break_even", {}) or {})
        if isinstance(trade_mgmt_cfg.get("break_even", {}), dict)
        else {}
    )
    early_exit_cfg = (
        copy.deepcopy(trade_mgmt_cfg.get("early_exit", {}) or {})
        if isinstance(trade_mgmt_cfg.get("early_exit", {}), dict)
        else {}
    )
    trade_mgmt_cfg["enabled"] = True
    break_even_cfg["enabled"] = True
    early_exit_cfg["enabled"] = False
    trade_mgmt_cfg["break_even"] = break_even_cfg
    trade_mgmt_cfg["early_exit"] = early_exit_cfg
    runtime_cfg["trade_management"] = trade_mgmt_cfg
    de3_cfg["runtime"] = runtime_cfg
    bt.CONFIG["DE3_V4"] = de3_cfg


def _configure_admission_controller(
    *,
    enabled: bool,
    key_granularity: str,
    history_window_trades: int,
    warmup_trades: int,
    cold_avg_net_per_contract_usd: float,
    cold_max_winrate: float,
    defensive_size_multiplier: float,
    block_avg_net_per_contract_usd: float | None,
    block_max_winrate: float | None,
    min_contracts: int,
    reduce_only: bool,
    require_signal_weakness: bool,
    max_execution_quality_score: float | None,
    max_entry_model_margin: float | None,
    max_route_confidence: float | None,
    max_edge_points: float | None,
) -> None:
    de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
    runtime_cfg = (
        copy.deepcopy(de3_cfg.get("runtime", {}) or {})
        if isinstance(de3_cfg.get("runtime", {}), dict)
        else {}
    )
    admission_cfg = (
        copy.deepcopy(runtime_cfg.get("backtest_admission_controller", {}) or {})
        if isinstance(runtime_cfg.get("backtest_admission_controller", {}), dict)
        else {}
    )
    admission_cfg["enabled"] = bool(enabled)
    admission_cfg["key_granularity"] = str(key_granularity or "lane_context")
    admission_cfg["history_window_trades"] = int(max(1, history_window_trades))
    admission_cfg["warmup_trades"] = int(max(0, warmup_trades))
    admission_cfg["cold_avg_net_per_contract_usd"] = float(cold_avg_net_per_contract_usd)
    admission_cfg["cold_max_winrate"] = float(cold_max_winrate)
    admission_cfg["defensive_size_multiplier"] = float(defensive_size_multiplier)
    admission_cfg["block_avg_net_per_contract_usd"] = (
        None if block_avg_net_per_contract_usd is None else float(block_avg_net_per_contract_usd)
    )
    admission_cfg["block_max_winrate"] = (
        None if block_max_winrate is None else float(block_max_winrate)
    )
    admission_cfg["min_contracts"] = int(max(1, min_contracts))
    admission_cfg["reduce_only"] = bool(reduce_only)
    admission_cfg["require_signal_weakness"] = bool(require_signal_weakness)
    admission_cfg["max_execution_quality_score"] = (
        None if max_execution_quality_score is None else float(max_execution_quality_score)
    )
    admission_cfg["max_entry_model_margin"] = (
        None if max_entry_model_margin is None else float(max_entry_model_margin)
    )
    admission_cfg["max_route_confidence"] = (
        None if max_route_confidence is None else float(max_route_confidence)
    )
    admission_cfg["max_edge_points"] = (
        None if max_edge_points is None else float(max_edge_points)
    )
    runtime_cfg["backtest_admission_controller"] = admission_cfg
    de3_cfg["runtime"] = runtime_cfg
    bt.CONFIG["DE3_V4"] = de3_cfg


def _write_converted_csv(trade_log: List[Dict[str, Any]], out_path: Path) -> None:
    header = [
        "Trade #",
        "Type",
        "Date and time",
        "Signal",
        "Price USD",
        "Position size (qty)",
        "Position size (value)",
        "Net P&L USD",
        "MFE points",
        "MAE points",
        "Cumulative P&L USD",
    ]
    cumulative = 0.0
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for idx, trade in enumerate(trade_log, start=1):
            entry_price = float(trade.get("entry_price", 0.0) or 0.0)
            qty = float(trade.get("size", 0.0) or 0.0)
            pnl_net = float(trade.get("pnl_net", 0.0) or 0.0)
            cumulative += pnl_net
            writer.writerow(
                [
                    idx,
                    "Trade",
                    str(trade.get("entry_time", "") or ""),
                    str(trade.get("side", "") or ""),
                    entry_price,
                    qty,
                    round(entry_price * qty, 4),
                    round(pnl_net, 4),
                    float(trade.get("mfe_points", 0.0) or 0.0),
                    float(trade.get("mae_points", 0.0) or 0.0),
                    round(cumulative, 4),
                ]
            )


def _ui_style_sharpe(trade_log: List[Dict[str, Any]]) -> float:
    if not trade_log:
        return 0.0
    pnls: List[float] = []
    timestamps: List[float] = []
    for trade in trade_log:
        pnl = bt.safe_float(trade.get("pnl_net"), float("nan"))
        if not math.isfinite(pnl):
            pnl = bt.safe_float(trade.get("pnl_dollars"), float("nan"))
        if not math.isfinite(pnl):
            continue
        pnls.append(float(pnl))
        raw_time = trade.get("entry_time")
        try:
            ts = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00")).timestamp()
        except Exception:
            continue
        timestamps.append(ts)
    if not pnls:
        return 0.0
    avg_pnl = sum(pnls) / len(pnls)
    std_dev = math.sqrt(sum((p - avg_pnl) ** 2 for p in pnls) / len(pnls))
    if std_dev <= 1e-12:
        return 0.0
    duration_years = 1.0
    if len(timestamps) > 1:
        timestamps.sort()
        duration_years = max((timestamps[-1] - timestamps[0]) / (60 * 60 * 24 * 365.25), 0.0027)
    trades_per_year = len(pnls) / duration_years
    return float((avg_pnl / std_dev) * math.sqrt(trades_per_year))


def _run_window(
    symbol_df,
    *,
    label: str,
    start_raw: str,
    end_raw: str,
    admission_enabled: bool,
    admission_kwargs: Dict[str, Any],
    out_dir: Path,
    write_artifacts: bool,
) -> Dict[str, Any]:
    cfg_backup = copy.deepcopy(bt.CONFIG)
    try:
        _configure_break_even_only()
        _configure_admission_controller(enabled=admission_enabled, **admission_kwargs)
        bt.CONFIG["BACKTEST_CONSOLE_PROGRESS"] = False
        bt.CONFIG["BACKTEST_CONSOLE_STATUS"] = False
        bt.CONFIG["BACKTEST_LIVE_REPORT_ENABLED"] = False
        start_time = bt.parse_user_datetime(start_raw, bt.NY_TZ, is_end=False)
        end_time = bt.parse_user_datetime(end_raw, bt.NY_TZ, is_end=True)
        stats = bt.run_backtest(
            symbol_df,
            start_time,
            end_time,
            enabled_strategies={"DynamicEngine3Strategy"},
            enabled_filters=set(),
        )
        trade_log = stats.get("trade_log", []) or []
        risk = bt._compute_backtest_risk_metrics(trade_log)
        report_path = None
        csv_path = None
        if write_artifacts:
            report_path = bt.save_backtest_report(
                stats,
                "AUTO_BY_DAY",
                start_time,
                end_time,
                output_dir=out_dir,
            )
            csv_path = report_path.with_name(f"converted_{report_path.stem}.csv")
            _write_converted_csv(trade_log, csv_path)
        return {
            "label": label,
            "range_start": start_time.isoformat(),
            "range_end": end_time.isoformat(),
            "report_path": str(report_path) if report_path is not None else None,
            "csv_path": str(csv_path) if csv_path is not None else None,
            "summary": {
                "equity": float(stats.get("equity", 0.0) or 0.0),
                "trades": int(stats.get("trades", 0) or 0),
                "wins": int(stats.get("wins", 0) or 0),
                "losses": int(stats.get("losses", 0) or 0),
                "winrate": float(stats.get("winrate", 0.0) or 0.0),
                "max_drawdown": float(stats.get("max_drawdown", 0.0) or 0.0),
                "ui_sharpe": _ui_style_sharpe(trade_log),
                "daily_sharpe": float(risk.get("daily_sharpe", 0.0) or 0.0),
                "profit_factor": float(risk.get("profit_factor", 0.0) or 0.0),
            },
            "de3_backtest_admission_summary": copy.deepcopy(
                stats.get("de3_backtest_admission_summary", {}) or {}
            ),
        }
    finally:
        bt.CONFIG.clear()
        bt.CONFIG.update(cfg_backup)


def _delta_from_runs(baseline: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, float]:
    base_summary = baseline["summary"]
    exp_summary = experiment["summary"]
    return {
        "equity": exp_summary["equity"] - base_summary["equity"],
        "trades": exp_summary["trades"] - base_summary["trades"],
        "winrate": exp_summary["winrate"] - base_summary["winrate"],
        "max_drawdown": exp_summary["max_drawdown"] - base_summary["max_drawdown"],
        "ui_sharpe": exp_summary["ui_sharpe"] - base_summary["ui_sharpe"],
        "daily_sharpe": exp_summary["daily_sharpe"] - base_summary["daily_sharpe"],
        "profit_factor": exp_summary["profit_factor"] - base_summary["profit_factor"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a backtest-only DE3v4 admission controller against the current break-even-only baseline."
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["2022", "2023", "2024", "2025"],
        help="Window labels to evaluate. Choices: 2022 2023 2024 2025",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_reports/de3_backtest_admission_eval",
        help="Directory to store reports and summary JSON.",
    )
    parser.add_argument("--key-granularity", default="lane_context")
    parser.add_argument("--history-window-trades", type=int, default=20)
    parser.add_argument("--warmup-trades", type=int, default=8)
    parser.add_argument("--cold-avg-net-per-contract-usd", type=float, default=-10.0)
    parser.add_argument("--cold-max-winrate", type=float, default=0.40)
    parser.add_argument("--defensive-size-multiplier", type=float, default=0.50)
    parser.add_argument("--block-avg-net-per-contract-usd", type=float, default=None)
    parser.add_argument("--block-max-winrate", type=float, default=None)
    parser.add_argument("--min-contracts", type=int, default=1)
    parser.add_argument("--allow-upsizing", action="store_true")
    parser.add_argument("--require-signal-weakness", action="store_true")
    parser.add_argument("--max-execution-quality-score", type=float, default=None)
    parser.add_argument("--max-entry-model-margin", type=float, default=None)
    parser.add_argument("--max-route-confidence", type=float, default=None)
    parser.add_argument("--max-edge-points", type=float, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-adapted", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    args = parser.parse_args()

    selected_windows = []
    for label in args.windows:
        if label not in WINDOW_BY_LABEL:
            raise SystemExit(f"Unknown window label: {label}")
        selected_windows.append(WINDOW_BY_LABEL[label])

    _, symbol_df = _load_symbol_df()
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    admission_kwargs = {
        "key_granularity": str(args.key_granularity),
        "history_window_trades": int(args.history_window_trades),
        "warmup_trades": int(args.warmup_trades),
        "cold_avg_net_per_contract_usd": float(args.cold_avg_net_per_contract_usd),
        "cold_max_winrate": float(args.cold_max_winrate),
        "defensive_size_multiplier": float(args.defensive_size_multiplier),
        "block_avg_net_per_contract_usd": (
            None
            if args.block_avg_net_per_contract_usd is None
            else float(args.block_avg_net_per_contract_usd)
        ),
        "block_max_winrate": (
            None
            if args.block_max_winrate is None
            else float(args.block_max_winrate)
        ),
        "min_contracts": int(args.min_contracts),
        "reduce_only": not bool(args.allow_upsizing),
        "require_signal_weakness": bool(args.require_signal_weakness),
        "max_execution_quality_score": (
            None
            if args.max_execution_quality_score is None
            else float(args.max_execution_quality_score)
        ),
        "max_entry_model_margin": (
            None
            if args.max_entry_model_margin is None
            else float(args.max_entry_model_margin)
        ),
        "max_route_confidence": (
            None
            if args.max_route_confidence is None
            else float(args.max_route_confidence)
        ),
        "max_edge_points": (
            None
            if args.max_edge_points is None
            else float(args.max_edge_points)
        ),
    }

    summary_rows: List[Dict[str, Any]] = []
    write_artifacts = not bool(args.no_artifacts)
    for label, start_raw, end_raw in selected_windows:
        print(f"\n=== {label} baseline ===", flush=True)
        baseline_result = None
        if not args.skip_baseline:
            baseline_result = _run_window(
                symbol_df,
                label=f"{label}_baseline",
                start_raw=start_raw,
                end_raw=end_raw,
                admission_enabled=False,
                admission_kwargs=admission_kwargs,
                out_dir=out_dir,
                write_artifacts=write_artifacts,
            )
        print(f"\n=== {label} adapted ===", flush=True)
        adapted_result = None
        if not args.skip_adapted:
            adapted_result = _run_window(
                symbol_df,
                label=f"{label}_adapted",
                start_raw=start_raw,
                end_raw=end_raw,
                admission_enabled=True,
                admission_kwargs=admission_kwargs,
                out_dir=out_dir,
                write_artifacts=write_artifacts,
            )

        row: Dict[str, Any] = {
            "label": label,
            "baseline": baseline_result,
            "adapted": adapted_result,
            "delta": None,
        }
        if baseline_result is not None and adapted_result is not None:
            row["delta"] = _delta_from_runs(baseline_result, adapted_result)
        summary_rows.append(row)

    summary_payload = {
        "generated_at": datetime.now(bt.NY_TZ).isoformat(),
        "windows": args.windows,
        "admission_controller": admission_kwargs,
        "rows": summary_rows,
    }
    summary_path = out_dir / "de3_backtest_admission_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
