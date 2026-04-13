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


EXPERIMENT_RULE_NAMES = [
    "hist_06_09_long_rev_t2_upper_wick_floor",
    "hist_09_12_long_mom_t3_upper_wick_cap",
    "hist_12_15_long_rev_t2_range10_cap",
    "hist_15_18_long_mom_t6_final_score_floor",
    "hist_09_12_short_rev_t6_upper_wick_floor",
    "hist_18_21_long_rev_t2_upper_wick_cap",
]

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


def _enable_selected_prune_rules(extra_rule_names: Iterable[str]) -> List[str]:
    runtime_cfg = bt.CONFIG.setdefault("DE3_V4", {}).setdefault("runtime", {})
    prune_cfg = runtime_cfg.setdefault("prune_rules", {})
    rules = prune_cfg.get("rules", [])
    if not isinstance(rules, list):
        raise RuntimeError("DE3_V4.runtime.prune_rules.rules is not a list")
    default_enabled = {
        str(rule.get("name", "") or "").strip()
        for rule in rules
        if isinstance(rule, dict) and bool(rule.get("enabled", False))
    }
    keep_enabled = default_enabled | {str(name).strip() for name in extra_rule_names if str(name).strip()}
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_name = str(rule.get("name", "") or "").strip()
        rule["enabled"] = rule_name in keep_enabled
    prune_cfg["enabled"] = True
    return sorted(keep_enabled)


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
    extra_rule_names: Iterable[str],
    out_dir: Path,
    write_artifacts: bool,
) -> Dict[str, Any]:
    cfg_backup = copy.deepcopy(bt.CONFIG)
    try:
        _configure_break_even_only()
        enabled_rules = _enable_selected_prune_rules(extra_rule_names)
        start_time = bt.parse_user_datetime(start_raw, bt.NY_TZ, is_end=False)
        end_time = bt.parse_user_datetime(end_raw, bt.NY_TZ, is_end=True)
        stats = bt.run_backtest(
            symbol_df,
            start_time,
            end_time,
            enabled_strategies={"DynamicEngine3Strategy"},
            enabled_filters=set(),
        )
        risk = bt._compute_backtest_risk_metrics(stats.get("trade_log", []) or [])
        trade_log = stats.get("trade_log", []) or []
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
            "enabled_rules": enabled_rules,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate backtest-only DE3 historical prune rules on selected yearly windows."
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=[label for label, _, _ in EVAL_WINDOWS],
        help="Year window labels to evaluate. Defaults to 2022 2023 2024 2025.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "baseline", "experiment"],
        default="both",
        help="Whether to run baseline, experiment, or both.",
    )
    parser.add_argument(
        "--out-dir",
        default="backtest_reports/de3_historical_weakness_analysis",
        help="Directory for summary JSON and any saved report artifacts.",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip saving report JSON and converted trade CSV files for each run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    requested_windows: List[Tuple[str, str, str]] = []
    for raw_year in args.years:
        label = str(raw_year or "").strip()
        if not label:
            continue
        if label not in WINDOW_BY_LABEL:
            raise SystemExit(f"Unknown year label '{label}'. Valid values: {', '.join(sorted(WINDOW_BY_LABEL))}")
        requested_windows.append(WINDOW_BY_LABEL[label])
    if not requested_windows:
        raise SystemExit("No valid years requested.")

    out_dir = ROOT / str(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, symbol_df = _load_symbol_df()
    results: Dict[str, Any] = {
        "experiment_rules": list(EXPERIMENT_RULE_NAMES),
        "last_mode": str(args.mode),
        "years": [label for label, _, _ in requested_windows],
        "windows": {},
    }
    for label, start_raw, end_raw in requested_windows:
        window_result: Dict[str, Any] = {}
        if args.mode in {"both", "baseline"}:
            print(f"Running baseline {label}...", flush=True)
            baseline = _run_window(
                symbol_df,
                f"{label}_baseline",
                start_raw,
                end_raw,
                [],
                out_dir=out_dir,
                write_artifacts=not bool(args.no_artifacts),
            )
            window_result["baseline"] = baseline
        if args.mode in {"both", "experiment"}:
            print(f"Running experiment {label}...", flush=True)
            experiment = _run_window(
                symbol_df,
                f"{label}_experiment",
                start_raw,
                end_raw,
                EXPERIMENT_RULE_NAMES,
                out_dir=out_dir,
                write_artifacts=not bool(args.no_artifacts),
            )
            window_result["experiment"] = experiment
        if "baseline" in window_result and "experiment" in window_result:
            window_result["delta"] = _delta_from_runs(
                window_result["baseline"],
                window_result["experiment"],
            )
        results["windows"][label] = window_result

    out_path = out_dir / "prune_experiment_break_even_only_summary.json"
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        if isinstance(existing, dict):
            merged_windows = existing.get("windows", {})
            if not isinstance(merged_windows, dict):
                merged_windows = {}
            for label, window_result in results["windows"].items():
                prior = merged_windows.get(label, {})
                if not isinstance(prior, dict):
                    prior = {}
                merged_window = dict(prior)
                merged_window.update(window_result)
                if "baseline" in merged_window and "experiment" in merged_window:
                    merged_window["delta"] = _delta_from_runs(
                        merged_window["baseline"],
                        merged_window["experiment"],
                    )
                merged_windows[label] = merged_window
            merged_years = []
            seen_years = set()
            for year in list(existing.get("years", []) or []) + list(results.get("years", []) or []):
                year_text = str(year)
                if year_text in seen_years:
                    continue
                seen_years.add(year_text)
                merged_years.append(year_text)
            results = dict(existing)
            results["experiment_rules"] = list(EXPERIMENT_RULE_NAMES)
            results["last_mode"] = str(args.mode)
            results["years"] = merged_years
            results["windows"] = merged_windows
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Summary: {out_path}", flush=True)
    print(json.dumps(results["windows"], indent=2), flush=True)


if __name__ == "__main__":
    main()
