import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from config import CONFIG
from tools.run_filterless_flip_policy_compare import _prepare_symbol_df, _resolve_source
from tools.run_live_strategy_day_compare import _extract_run_summary, _group_backtest_trades


DEFAULT_STRATEGIES = [
    "DynamicEngine3Strategy",
    "RegimeAdaptiveStrategy",
]

STRATEGY_ALIASES = {
    "de3": "DynamicEngine3Strategy",
    "dynamicengine3": "DynamicEngine3Strategy",
    "dynamicengine3strategy": "DynamicEngine3Strategy",
    "regimeadaptive": "RegimeAdaptiveStrategy",
    "regimeadaptivestrategy": "RegimeAdaptiveStrategy",
    "aetherflow": "AetherFlowStrategy",
    "aetherflowstrategy": "AetherFlowStrategy",
}


def _normalize_selected_strategies(values: list[str]) -> list[str]:
    if not values:
        return list(DEFAULT_STRATEGIES)
    resolved: list[str] = []
    for raw_value in values:
        key = str(raw_value or "").strip()
        if not key:
            continue
        strategy_name = STRATEGY_ALIASES.get(key.casefold(), key)
        resolved.append(strategy_name)
    unique = sorted(set(resolved))
    if not unique:
        raise SystemExit("No valid strategies selected.")
    return unique


def _run_backtest(
    *,
    symbol_df: pd.DataFrame,
    start_time,
    end_time,
    selected_strategies: list[str],
    workers: int,
    experimental_enabled: bool,
    live_report_dir: Path | None,
) -> dict:
    cfg_backup = copy.deepcopy(CONFIG)
    try:
        CONFIG.setdefault("GEMINI", {})["enabled"] = False
        CONFIG["BACKTEST_CONSOLE_PROGRESS"] = False
        CONFIG["BACKTEST_CONSOLE_STATUS"] = False
        CONFIG["BACKTEST_WORKERS"] = max(1, int(workers))
        CONFIG.setdefault("AETHERFLOW_STRATEGY", {})["enabled_backtest"] = True

        experimental_cfg = copy.deepcopy(
            CONFIG.get("BACKTEST_EXPERIMENTAL_MODIFIERS", {}) or {}
        )
        experimental_cfg["enabled"] = bool(experimental_enabled)
        CONFIG["BACKTEST_EXPERIMENTAL_MODIFIERS"] = experimental_cfg

        live_report_cfg = copy.deepcopy(CONFIG.get("BACKTEST_LIVE_REPORT", {}) or {})
        live_report_cfg["enabled"] = True
        live_report_cfg["include_trade_log"] = False
        if live_report_dir is not None:
            live_report_cfg["output_dir"] = str(live_report_dir)
        CONFIG["BACKTEST_LIVE_REPORT"] = live_report_cfg

        return bt.run_backtest(
            symbol_df.copy(),
            start_time,
            end_time,
            enabled_strategies=set(selected_strategies),
            enabled_filters=set(),
        )
    finally:
        CONFIG.clear()
        CONFIG.update(cfg_backup)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a full-range backtest for the main filterless strategies, "
            "optionally enabling structural-level and level-fill modifiers."
        )
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--start", default="2011-01-02")
    parser.add_argument("--end", default="2026-04-17")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--pre-roll-days", type=int, default=180)
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Repeatable strategy selection: de3, regimeadaptive, aetherflow.",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--experimental-enabled",
        choices=["true", "false"],
        default="true",
        help="Enable or disable experimental structural-level and level-fill modifiers.",
    )
    parser.add_argument(
        "--report-out",
        default="reports/fullrange_main_strategies_experimental.json",
    )
    parser.add_argument(
        "--live-report-dir",
        default="backtest_reports/fullrange_main_strategies_experimental_live",
    )
    args = parser.parse_args()

    selected_strategies = _normalize_selected_strategies(list(args.strategy or []))
    source_path = _resolve_source(str(args.source))
    start_time = bt.parse_user_datetime(str(args.start), bt.NY_TZ, is_end=False)
    end_time = bt.parse_user_datetime(str(args.end), bt.NY_TZ, is_end=True)

    symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        str(args.symbol_mode),
        str(args.symbol_method),
        pre_roll_days=max(0, int(args.pre_roll_days or 0)),
    )

    live_report_dir = Path(str(args.live_report_dir)).expanduser()
    if not live_report_dir.is_absolute():
        live_report_dir = (ROOT / live_report_dir).resolve()
    live_report_dir.mkdir(parents=True, exist_ok=True)

    stats = _run_backtest(
        symbol_df=symbol_df,
        start_time=start_time,
        end_time=end_time,
        selected_strategies=selected_strategies,
        workers=int(args.workers),
        experimental_enabled=str(args.experimental_enabled).strip().lower() == "true",
        live_report_dir=live_report_dir,
    )

    grouped = _group_backtest_trades(list(stats.get("trade_log", []) or []))
    payload = {
        "created_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "source_path": str(source_path),
        "window": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        },
        "selected_strategies": list(selected_strategies),
        "symbol": str(symbol),
        "symbol_distribution": symbol_distribution,
        "backtest_workers": int(args.workers),
        "experimental_enabled": str(args.experimental_enabled).strip().lower() == "true",
        "experimental": {
            "summary": _extract_run_summary(stats),
            "experimental_modifiers": dict(stats.get("experimental_modifiers", {}) or {}),
            "by_strategy": {
                name: dict((grouped.get(name) or {}).get("summary", {}) or {})
                for name in selected_strategies
            },
        },
        "live_report_dir": str(live_report_dir),
    }

    report_out = Path(str(args.report_out)).expanduser()
    if not report_out.is_absolute():
        report_out = (ROOT / report_out).resolve()
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"report={report_out}")
    print(f"live_report_dir={live_report_dir}")
    print(json.dumps(payload["experimental"]["summary"], indent=2))


if __name__ == "__main__":
    main()
