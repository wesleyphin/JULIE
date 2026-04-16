from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

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


def _configure_manifold_adaptation(
    enabled: bool,
    blocked_regimes: Iterable[str],
    mode: str = "block",
    block_no_trade: bool = False,
    require_allow_style: bool = False,
) -> None:
    de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
    runtime_cfg = (
        copy.deepcopy(de3_cfg.get("runtime", {}) or {})
        if isinstance(de3_cfg.get("runtime", {}), dict)
        else {}
    )
    manifold_cfg = (
        copy.deepcopy(runtime_cfg.get("manifold_adaptation", {}) or {})
        if isinstance(runtime_cfg.get("manifold_adaptation", {}), dict)
        else {}
    )
    manifold_cfg["enabled"] = bool(enabled)
    manifold_cfg["mode"] = str(mode or "block").strip().lower() or "block"
    manifold_cfg["blocked_regimes"] = [
        str(item).strip().upper()
        for item in blocked_regimes
        if str(item).strip()
    ]
    manifold_cfg["block_no_trade"] = bool(block_no_trade)
    manifold_cfg["require_allow_style"] = bool(require_allow_style)
    runtime_cfg["manifold_adaptation"] = manifold_cfg
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
    label: str,
    start_raw: str,
    end_raw: str,
    adaptation_enabled: bool,
    blocked_regimes: Iterable[str],
    out_dir: Path,
    write_artifacts: bool,
) -> Dict[str, Any]:
    cfg_backup = copy.deepcopy(bt.CONFIG)
    try:
        _configure_break_even_only()
        _configure_manifold_adaptation(
            enabled=adaptation_enabled,
            blocked_regimes=blocked_regimes,
            mode="block",
            block_no_trade=False,
            require_allow_style=False,
        )
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
            "de3_manifold_adaptation_summary": copy.deepcopy(
                stats.get("de3_manifold_adaptation_summary", {}) or {}
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
        description="Evaluate backtest-only DE3v4 manifold adaptation against the current break-even-only baseline."
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["2022", "2023", "2024", "2025"],
        help="Window labels to evaluate. Choices: 2022 2023 2024 2025",
    )
    parser.add_argument(
        "--blocked-regime",
        action="append",
        default=["ROTATIONAL_TURBULENCE"],
        help="Regime label to block when manifold adaptation is enabled. Repeatable.",
    )
    parser.add_argument(
        "--out-dir",
        default="backtest_reports/de3_manifold_adaptation_eval",
        help="Directory for reports, CSVs, and the summary JSON.",
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help="Save backtest reports and converted CSVs for baseline and adapted runs.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Run only the adapted side.",
    )
    parser.add_argument(
        "--skip-adapted",
        action="store_true",
        help="Run only the baseline side.",
    )
    args = parser.parse_args()

    if bool(args.skip_baseline) and bool(args.skip_adapted):
        raise SystemExit("Cannot use --skip-baseline and --skip-adapted together.")

    selected_windows: List[Tuple[str, str, str]] = []
    for raw_label in args.windows:
        label = str(raw_label or "").strip()
        if label not in WINDOW_BY_LABEL:
            raise SystemExit(f"Unknown window label: {label}")
        selected_windows.append(WINDOW_BY_LABEL[label])

    blocked_regimes = [
        str(item).strip().upper()
        for item in (args.blocked_regime or [])
        if str(item).strip()
    ]
    if not blocked_regimes:
        raise SystemExit("At least one --blocked-regime is required.")

    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _, symbol_df = _load_symbol_df()
    summary_rows: List[Dict[str, Any]] = []

    for label, start_raw, end_raw in selected_windows:
        baseline = None
        experiment = None
        delta = None
        if not bool(args.skip_baseline):
            print(f"[{label}] baseline...", flush=True)
            baseline_dir = out_dir / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            baseline = _run_window(
                symbol_df,
                label,
                start_raw,
                end_raw,
                adaptation_enabled=False,
                blocked_regimes=blocked_regimes,
                out_dir=baseline_dir,
                write_artifacts=bool(args.write_artifacts),
            )
        if not bool(args.skip_adapted):
            print(f"[{label}] adapted...", flush=True)
            adapted_dir = out_dir / "adapted"
            adapted_dir.mkdir(parents=True, exist_ok=True)
            experiment = _run_window(
                symbol_df,
                label,
                start_raw,
                end_raw,
                adaptation_enabled=True,
                blocked_regimes=blocked_regimes,
                out_dir=adapted_dir,
                write_artifacts=bool(args.write_artifacts),
            )
        if baseline is not None and experiment is not None:
            delta = _delta_from_runs(baseline, experiment)
        row = {
            "label": label,
            "blocked_regimes": blocked_regimes,
            "baseline": baseline,
            "adapted": experiment,
            "delta": delta,
        }
        summary_rows.append(row)
        if delta is not None:
            print(
                f"[{label}] delta equity={delta['equity']:.2f} trades={delta['trades']} "
                f"winrate={delta['winrate']:.2f} max_dd={delta['max_drawdown']:.2f} "
                f"ui_sharpe={delta['ui_sharpe']:.3f}",
                flush=True,
            )

    payload = {
        "blocked_regimes": blocked_regimes,
        "windows": summary_rows,
    }
    summary_path = out_dir / "de3_manifold_adaptation_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
