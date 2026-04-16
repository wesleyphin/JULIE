from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
import tools.evaluate_de3_backtest_admission as eva


WINDOWS = {
    "2022": ("2022-01-01", "2022-12-31 23:59"),
    "2023": ("2023-01-01", "2023-12-31 23:59"),
    "2024": ("2024-01-01", "2024-12-31 23:59"),
    "2025": ("2025-01-01", "2025-12-31 23:59"),
    "2024_2025": ("2024-01-01", "2025-12-31 23:59"),
    "2022_2025": ("2022-01-01", "2025-12-31 23:59"),
    "2022_01": ("2022-01-01", "2022-01-31 23:59"),
    "2023_01": ("2023-01-01", "2023-01-31 23:59"),
    "2024_01": ("2024-01-01", "2024-01-31 23:59"),
    "2025_01": ("2025-01-01", "2025-01-31 23:59"),
}


def _configure_intraday_regime_controller(
    *,
    enabled: bool,
    mode: str | None = None,
    sessions: List[str] | None = None,
    apply_lanes: List[str] | None = None,
    enable_bullish_mirror: bool | None = None,
    defensive_size_multiplier: float | None = None,
    defensive_score_threshold: float | None = None,
    block_score_threshold: float | None = None,
    dominance_threshold: float | None = None,
    block_dominance_threshold: float | None = None,
) -> None:
    de3_cfg = copy.deepcopy(bt.CONFIG.get("DE3_V4", {}) or {})
    runtime_cfg = (
        copy.deepcopy(de3_cfg.get("runtime", {}) or {})
        if isinstance(de3_cfg.get("runtime", {}), dict)
        else {}
    )
    regime_cfg = (
        copy.deepcopy(runtime_cfg.get("backtest_intraday_regime_controller", {}) or {})
        if isinstance(runtime_cfg.get("backtest_intraday_regime_controller", {}), dict)
        else {}
    )
    regime_cfg["enabled"] = bool(enabled)
    if mode:
        regime_cfg["mode"] = str(mode)
    if sessions is not None:
        regime_cfg["apply_sessions"] = [str(item).strip().upper() for item in sessions if str(item).strip()]
    if apply_lanes is not None:
        regime_cfg["apply_lanes"] = [str(item).strip() for item in apply_lanes if str(item).strip()]
    if enable_bullish_mirror is not None:
        regime_cfg["enable_bullish_mirror"] = bool(enable_bullish_mirror)
    if defensive_size_multiplier is not None:
        regime_cfg["defensive_size_multiplier"] = float(defensive_size_multiplier)
    if defensive_score_threshold is not None:
        regime_cfg["defensive_score_threshold"] = float(defensive_score_threshold)
    if block_score_threshold is not None:
        regime_cfg["block_score_threshold"] = float(block_score_threshold)
    if dominance_threshold is not None:
        regime_cfg["dominance_threshold"] = float(dominance_threshold)
    if block_dominance_threshold is not None:
        regime_cfg["block_dominance_threshold"] = float(block_dominance_threshold)
    runtime_cfg["backtest_intraday_regime_controller"] = regime_cfg
    de3_cfg["runtime"] = runtime_cfg
    bt.CONFIG["DE3_V4"] = de3_cfg


def _summary_from_stats(
    label: str,
    stats: Dict[str, Any],
    *,
    report_path: Path | None,
    csv_path: Path | None,
) -> Dict[str, Any]:
    trade_log = stats.get("trade_log", []) or []
    wins = int(stats.get("wins", 0) or 0)
    trades = int(stats.get("trades", len(trade_log)) or len(trade_log))
    win_rate = float(wins / trades) if trades > 0 else 0.0
    out = {
        "label": str(label),
        "net_pnl": float(stats.get("equity", 0.0) or 0.0),
        "trades": int(trades),
        "wins": int(wins),
        "win_rate": float(win_rate),
        "max_dd": float(stats.get("max_dd", 0.0) or 0.0),
        "ui_style_sharpe": float(eva._ui_style_sharpe(trade_log)),
        "intraday_regime_summary": copy.deepcopy(stats.get("de3_intraday_regime_summary", {}) or {}),
        "report_path": str(report_path) if report_path is not None else None,
        "converted_csv_path": str(csv_path) if csv_path is not None else None,
    }
    mc_stats = stats.get("monte_carlo_stats", {}) or {}
    if isinstance(mc_stats, dict) and mc_stats:
        out["monte_carlo_stats"] = copy.deepcopy(mc_stats)
    return out


def _run_window(
    symbol_df,
    *,
    label: str,
    start_raw: str,
    end_raw: str,
    intraday_enabled: bool,
    intraday_kwargs: Dict[str, Any],
    out_dir: Path,
    mode_name: str,
) -> Dict[str, Any]:
    cfg_backup = copy.deepcopy(bt.CONFIG)
    try:
        eva._configure_break_even_only()
        _configure_intraday_regime_controller(enabled=intraday_enabled, **intraday_kwargs)
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
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"de3_intraday_regime_{label}_{mode_name}_{stamp}"
        report_path = out_dir / f"{base_name}.json"
        report_payload = bt._serialize_json_value(stats)
        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=True), encoding="utf-8")
        csv_path = out_dir / f"converted_{base_name}.csv"
        eva._write_converted_csv(stats.get("trade_log", []) or [], csv_path)
        return _summary_from_stats(label, stats, report_path=report_path, csv_path=csv_path)
    finally:
        bt.CONFIG.clear()
        bt.CONFIG.update(cfg_backup)


def _delta_summary(baseline: Dict[str, Any], adapted: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "net_pnl": float(adapted.get("net_pnl", 0.0) - baseline.get("net_pnl", 0.0)),
        "trades": int(adapted.get("trades", 0) - baseline.get("trades", 0)),
        "win_rate": float(adapted.get("win_rate", 0.0) - baseline.get("win_rate", 0.0)),
        "max_dd": float(adapted.get("max_dd", 0.0) - baseline.get("max_dd", 0.0)),
        "ui_style_sharpe": float(
            adapted.get("ui_style_sharpe", 0.0) - baseline.get("ui_style_sharpe", 0.0)
        ),
    }
    base_mc = baseline.get("monte_carlo_stats", {}) or {}
    adapted_mc = adapted.get("monte_carlo_stats", {}) or {}
    if isinstance(base_mc, dict) and isinstance(adapted_mc, dict) and base_mc and adapted_mc:
        mc_delta = {}
        for key in (
            "net_pnl_mean",
            "net_pnl_p05",
            "net_pnl_p95",
            "max_drawdown_mean",
            "max_drawdown_p95",
            "probability_final_balance_above_start",
        ):
            base_val = base_mc.get(key)
            adapted_val = adapted_mc.get(key)
            if isinstance(base_val, (int, float)) and isinstance(adapted_val, (int, float)):
                mc_delta[key] = float(adapted_val - base_val)
        if mc_delta:
            out["monte_carlo_delta"] = mc_delta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DE3 backtest-only intraday regime controller.")
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["2022", "2024", "2025"],
        help=f"Window labels to run. Available: {', '.join(sorted(WINDOWS))}",
    )
    parser.add_argument(
        "--output-dir",
        default="backtest_reports/de3_intraday_regime_eval",
        help="Directory for JSON summaries and converted CSVs.",
    )
    parser.add_argument("--mode", default="block_defensive", choices=["block", "defensive", "block_defensive"])
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--apply-lanes", nargs="*", default=None)
    parser.add_argument("--enable-bullish-mirror", choices=["on", "off"], default=None)
    parser.add_argument("--defensive-size-multiplier", type=float, default=None)
    parser.add_argument("--defensive-score-threshold", type=float, default=None)
    parser.add_argument("--block-score-threshold", type=float, default=None)
    parser.add_argument("--dominance-threshold", type=float, default=None)
    parser.add_argument("--block-dominance-threshold", type=float, default=None)
    args = parser.parse_args()

    invalid = [label for label in args.windows if label not in WINDOWS]
    if invalid:
        raise SystemExit(f"Unknown windows: {', '.join(invalid)}")

    raw_df, symbol_df = eva._load_symbol_df()
    _ = raw_df
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    intraday_kwargs = {
        "mode": args.mode,
        "sessions": args.sessions,
        "apply_lanes": args.apply_lanes,
        "enable_bullish_mirror": (
            True if args.enable_bullish_mirror == "on" else False if args.enable_bullish_mirror == "off" else None
        ),
        "defensive_size_multiplier": args.defensive_size_multiplier,
        "defensive_score_threshold": args.defensive_score_threshold,
        "block_score_threshold": args.block_score_threshold,
        "dominance_threshold": args.dominance_threshold,
        "block_dominance_threshold": args.block_dominance_threshold,
    }

    summary = {
        "windows": {},
        "config": {
            "mode": args.mode,
            "sessions": args.sessions,
            "apply_lanes": args.apply_lanes,
            "enable_bullish_mirror": args.enable_bullish_mirror,
            "defensive_size_multiplier": args.defensive_size_multiplier,
            "defensive_score_threshold": args.defensive_score_threshold,
            "block_score_threshold": args.block_score_threshold,
            "dominance_threshold": args.dominance_threshold,
            "block_dominance_threshold": args.block_dominance_threshold,
        },
    }

    for label in args.windows:
        start_raw, end_raw = WINDOWS[label]
        print(f"[{label}] baseline", flush=True)
        baseline = _run_window(
            symbol_df,
            label=label,
            start_raw=start_raw,
            end_raw=end_raw,
            intraday_enabled=False,
            intraday_kwargs=intraday_kwargs,
            out_dir=out_dir,
            mode_name="baseline",
        )
        print(f"[{label}] adapted", flush=True)
        adapted = _run_window(
            symbol_df,
            label=label,
            start_raw=start_raw,
            end_raw=end_raw,
            intraday_enabled=True,
            intraday_kwargs=intraday_kwargs,
            out_dir=out_dir,
            mode_name="adapted",
        )
        summary["windows"][label] = {
            "baseline": baseline,
            "adapted": adapted,
            "delta": _delta_summary(baseline, adapted),
        }

    summary_path = out_dir / "de3_intraday_regime_summary.json"
    summary_path.write_text(json.dumps(bt._serialize_json_value(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
