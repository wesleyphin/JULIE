import argparse
import concurrent.futures as cf
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


FILTERLESS_STRATEGIES = {
    "DynamicEngine3Strategy",
    "RegimeAdaptiveStrategy",
    "AetherFlowStrategy",
}

STRATEGY_ALIASES = {
    "de3": "DynamicEngine3Strategy",
    "dynamicengine3": "DynamicEngine3Strategy",
    "dynamicengine3strategy": "DynamicEngine3Strategy",
    "regimeadaptive": "RegimeAdaptiveStrategy",
    "regimeadaptivestrategy": "RegimeAdaptiveStrategy",
    "aetherflow": "AetherFlowStrategy",
    "aetherflowstrategy": "AetherFlowStrategy",
}


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _base_live_opposite_reversal_cfg() -> dict:
    cfg = copy.deepcopy(CONFIG.get("LIVE_OPPOSITE_REVERSAL", {}) or {})
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "required_confirmations": max(1, _safe_int(cfg.get("required_confirmations"), 3)),
        "window_bars": max(1, _safe_int(cfg.get("window_bars"), 3)),
        "require_same_strategy_family": bool(
            cfg.get("require_same_strategy_family", False)
        ),
        "require_same_active_trade_family": bool(
            cfg.get("require_same_active_trade_family", False)
        ),
        "require_same_sub_strategy": bool(cfg.get("require_same_sub_strategy", False)),
        "allowed_vol_regimes": list(
            cfg.get("allowed_vol_regimes") or cfg.get("apply_vol_regimes") or []
        ),
        "block_countertrend_in_trend_day": bool(
            cfg.get("block_countertrend_in_trend_day", False)
        ),
    }


def _normalize_selected_strategies(values: list[str]) -> list[str]:
    if not values:
        return sorted(FILTERLESS_STRATEGIES)
    normalized = []
    for raw_value in values:
        key = str(raw_value or "").strip()
        if not key:
            continue
        resolved = STRATEGY_ALIASES.get(key.casefold(), key)
        if resolved not in FILTERLESS_STRATEGIES:
            raise SystemExit(f"Unknown strategy selection: {raw_value}")
        normalized.append(resolved)
    unique = sorted(set(normalized))
    if not unique:
        raise SystemExit("No valid strategies selected.")
    return unique


def _variant_cfg(**overrides) -> dict:
    cfg = _base_live_opposite_reversal_cfg()
    cfg.update(overrides)
    return cfg


DEFAULT_VARIANTS = {
    "live_current": _variant_cfg(),
    "no_flips": _variant_cfg(enabled=False),
    "cross_family_3": _variant_cfg(
        enabled=True,
        required_confirmations=3,
        window_bars=3,
        require_same_strategy_family=False,
        require_same_active_trade_family=False,
        require_same_sub_strategy=False,
        allowed_vol_regimes=[],
        block_countertrend_in_trend_day=False,
    ),
    "same_family_only_2": _variant_cfg(
        enabled=True,
        required_confirmations=2,
        window_bars=3,
        require_same_strategy_family=True,
        require_same_active_trade_family=True,
        require_same_sub_strategy=False,
        allowed_vol_regimes=[],
        block_countertrend_in_trend_day=False,
    ),
    "same_family_no_high_3": _variant_cfg(
        enabled=True,
        required_confirmations=3,
        window_bars=3,
        require_same_strategy_family=True,
        require_same_active_trade_family=True,
        require_same_sub_strategy=False,
        allowed_vol_regimes=["low", "normal"],
        block_countertrend_in_trend_day=False,
    ),
    "same_family_high_only_3": _variant_cfg(
        enabled=True,
        required_confirmations=3,
        window_bars=3,
        require_same_strategy_family=True,
        require_same_active_trade_family=True,
        require_same_sub_strategy=False,
        allowed_vol_regimes=["high"],
        block_countertrend_in_trend_day=False,
    ),
    "same_family_trend_aligned_3": _variant_cfg(
        enabled=True,
        required_confirmations=3,
        window_bars=3,
        require_same_strategy_family=True,
        require_same_active_trade_family=True,
        require_same_sub_strategy=False,
        allowed_vol_regimes=[],
        block_countertrend_in_trend_day=True,
    ),
}


def _resolve_source(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = (ROOT / path).resolve()
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Data file not found: {path_arg}")


def _prepare_symbol_df(
    source_path: Path,
    start_time,
    end_time,
    symbol_mode: str,
    symbol_method: str,
    *,
    pre_roll_days: int,
):
    df = bt.load_csv_cached(source_path, cache_dir=ROOT / "cache", use_cache=True)
    if df.empty:
        raise RuntimeError("No rows found in the source file.")

    selection_start = start_time - pd.Timedelta(days=max(0, int(pre_roll_days)))
    selection_start = selection_start.replace(hour=0, minute=0, second=0, microsecond=0)
    source_df = df[(df.index >= selection_start) & (df.index <= end_time)]
    if source_df.empty:
        raise RuntimeError("No rows found inside the requested symbol-selection window.")

    symbol = None
    symbol_distribution = {}
    symbol_df = source_df
    if "symbol" in source_df.columns:
        if symbol_mode != "single":
            symbol_df, auto_label, _ = bt.apply_symbol_mode(
                source_df,
                symbol_mode,
                symbol_method,
            )
            if symbol_df.empty:
                raise RuntimeError("No rows found after auto symbol selection.")
            selected_range_df = symbol_df[
                (symbol_df.index >= start_time) & (symbol_df.index <= end_time)
            ]
            if selected_range_df.empty:
                raise RuntimeError(
                    "No rows found in selected range after auto symbol selection."
                )
            symbol_distribution = selected_range_df["symbol"].value_counts().to_dict()
            symbol = auto_label
        else:
            preferred_symbol = bt.CONFIG.get("TARGET_SYMBOL")
            symbol = bt.choose_symbol(source_df, preferred_symbol)
            symbol_df = source_df[source_df["symbol"] == symbol]
            if symbol_df.empty:
                raise RuntimeError("No rows found for selected symbol.")
            selected_range_df = symbol_df[
                (symbol_df.index >= start_time) & (symbol_df.index <= end_time)
            ]
            if selected_range_df.empty:
                raise RuntimeError(
                    "No rows found in selected range for selected symbol."
                )
            symbol_distribution = selected_range_df["symbol"].value_counts().to_dict()

        symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    else:
        selected_range_df = source_df[
            (source_df.index >= start_time) & (source_df.index <= end_time)
        ]
        if selected_range_df.empty:
            raise RuntimeError("No rows found in the selected range.")
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


def _summarize_trade_rows(trades: list[dict]) -> dict:
    rows = [trade for trade in trades if isinstance(trade, dict)]
    count = len(rows)
    pnl_net = sum(_safe_float(trade.get("pnl_net"), 0.0) for trade in rows)
    wins = sum(1 for trade in rows if _safe_float(trade.get("pnl_net"), 0.0) >= 0.0)
    losses = max(0, count - wins)
    avg_net = (pnl_net / float(count)) if count else 0.0
    return {
        "trades": int(count),
        "wins": int(wins),
        "losses": int(losses),
        "winrate": (float(wins) / float(count) * 100.0) if count else 0.0,
        "pnl_net": float(pnl_net),
        "avg_trade_net": float(avg_net),
    }


def _group_trade_rows(trades: list[dict], field_name: str) -> dict:
    grouped: dict[str, list[dict]] = {}
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        key = str(trade.get(field_name) or "UNKNOWN")
        grouped.setdefault(key, []).append(trade)
    return {
        name: _summarize_trade_rows(grouped[name])
        for name in sorted(grouped.keys())
    }


def _reverse_trade_summary(trades: list[dict]) -> dict:
    reverse_trades = [
        trade
        for trade in trades
        if isinstance(trade, dict) and str(trade.get("exit_reason") or "") == "reverse"
    ]
    summary = _summarize_trade_rows(reverse_trades)
    summary["share_of_all_trades"] = (
        float(summary["trades"]) / float(len(trades)) if trades else 0.0
    )
    summary["by_strategy"] = _group_trade_rows(reverse_trades, "strategy")
    summary["by_session"] = _group_trade_rows(reverse_trades, "session")
    summary["by_vol_regime"] = _group_trade_rows(reverse_trades, "vol_regime")
    summary["by_trend_day_dir"] = _group_trade_rows(reverse_trades, "trend_day_dir")
    return summary


def _exit_reason_counts(trades: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for trade in trades:
        if not isinstance(trade, dict):
            continue
        name = str(trade.get("exit_reason") or "UNKNOWN")
        counts[name] = int(counts.get(name, 0) + 1)
    return {name: counts[name] for name in sorted(counts.keys())}


def _extract_variant_result(
    *,
    variant_name: str,
    variant_cfg: dict,
    start_time,
    end_time,
    symbol,
    symbol_distribution: dict,
    stats: dict,
) -> dict:
    trade_log = list(stats.get("trade_log", []) or [])
    return {
        "variant": variant_name,
        "config": dict(variant_cfg),
        "window": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        },
        "symbol": symbol,
        "symbol_distribution": symbol_distribution,
        "equity": stats.get("equity"),
        "trades": stats.get("trades"),
        "wins": stats.get("wins"),
        "losses": stats.get("losses"),
        "winrate": stats.get("winrate"),
        "profit_factor": stats.get("profit_factor"),
        "max_drawdown": stats.get("max_drawdown"),
        "avg_trade_net": stats.get("avg_trade_net"),
        "selection": stats.get("selection", {}),
        "trade_log_count": len(trade_log),
        "strategy_stats": _group_trade_rows(trade_log, "strategy"),
        "session_stats": _group_trade_rows(trade_log, "session"),
        "vol_regime_stats": _group_trade_rows(trade_log, "vol_regime"),
        "exit_reason_counts": _exit_reason_counts(trade_log),
        "reverse_stats": _reverse_trade_summary(trade_log),
        "opposite_reversal_summary": stats.get("opposite_reversal_summary", {}),
    }


def _run_variant(
    *,
    source_path: str,
    start_raw: str,
    end_raw: str,
    symbol_mode: str,
    symbol_method: str,
    pre_roll_days: int,
    selected_strategies: list[str],
    variant_name: str,
    variant_cfg: dict,
    worker_count: int,
) -> dict:
    orig_live_opp = copy.deepcopy(CONFIG.get("LIVE_OPPOSITE_REVERSAL", {}) or {})
    orig_gemini = copy.deepcopy(CONFIG.get("GEMINI", {}) or {})
    orig_live_report = copy.deepcopy(CONFIG.get("BACKTEST_LIVE_REPORT", {}) or {})
    orig_console_progress = CONFIG.get("BACKTEST_CONSOLE_PROGRESS", True)
    orig_workers = CONFIG.get("BACKTEST_WORKERS", 6)

    try:
        CONFIG["LIVE_OPPOSITE_REVERSAL"] = dict(variant_cfg)
        CONFIG.setdefault("GEMINI", {})["enabled"] = False
        CONFIG.setdefault("BACKTEST_LIVE_REPORT", {})["enabled"] = False
        CONFIG["BACKTEST_CONSOLE_PROGRESS"] = False
        CONFIG["BACKTEST_WORKERS"] = int(max(1, worker_count))

        start_time = bt.parse_user_datetime(start_raw, bt.NY_TZ, is_end=False)
        end_time = bt.parse_user_datetime(end_raw, bt.NY_TZ, is_end=True)
        symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
            Path(source_path),
            start_time,
            end_time,
            symbol_mode,
            symbol_method,
            pre_roll_days=pre_roll_days,
        )

        stats = bt.run_backtest(
            symbol_df,
            start_time,
            end_time,
            enabled_strategies=set(selected_strategies),
            enabled_filters=set(),
        )
        return _extract_variant_result(
            variant_name=variant_name,
            variant_cfg=variant_cfg,
            start_time=start_time,
            end_time=end_time,
            symbol=symbol,
            symbol_distribution=symbol_distribution,
            stats=stats,
        )
    finally:
        CONFIG["LIVE_OPPOSITE_REVERSAL"] = orig_live_opp
        CONFIG["GEMINI"] = orig_gemini
        CONFIG["BACKTEST_LIVE_REPORT"] = orig_live_report
        CONFIG["BACKTEST_CONSOLE_PROGRESS"] = orig_console_progress
        CONFIG["BACKTEST_WORKERS"] = orig_workers


def _attach_deltas(summary: dict) -> None:
    variants = summary.get("variants", {}) or {}
    baseline = variants.get("live_current")
    if not isinstance(baseline, dict) or baseline.get("error"):
        return
    baseline_equity = _safe_float(baseline.get("equity"), 0.0)
    baseline_dd = _safe_float(baseline.get("max_drawdown"), 0.0)
    baseline_trades = _safe_int(baseline.get("trades"), 0)
    baseline_reverse = (
        baseline.get("reverse_stats", {}) if isinstance(baseline.get("reverse_stats"), dict) else {}
    )
    baseline_reverse_pnl = _safe_float(baseline_reverse.get("pnl_net"), 0.0)

    ranking_rows = []
    for name, payload in variants.items():
        if not isinstance(payload, dict) or payload.get("error"):
            continue
        payload["delta_vs_live_current"] = {
            "equity": float(_safe_float(payload.get("equity"), 0.0) - baseline_equity),
            "max_drawdown": float(
                _safe_float(payload.get("max_drawdown"), 0.0) - baseline_dd
            ),
            "trades": int(_safe_int(payload.get("trades"), 0) - baseline_trades),
            "reverse_pnl_net": float(
                _safe_float(
                    (payload.get("reverse_stats", {}) or {}).get("pnl_net"),
                    0.0,
                )
                - baseline_reverse_pnl
            ),
        }
        ranking_rows.append(
            {
                "variant": name,
                "equity": _safe_float(payload.get("equity"), 0.0),
                "max_drawdown": _safe_float(payload.get("max_drawdown"), 0.0),
                "profit_factor": _safe_float(payload.get("profit_factor"), 0.0),
                "trades": _safe_int(payload.get("trades"), 0),
            }
        )

    summary["ranking_by_equity"] = sorted(
        ranking_rows,
        key=lambda row: (
            float(row.get("equity", 0.0)),
            -float(row.get("max_drawdown", 0.0)),
            float(row.get("profit_factor", 0.0)),
        ),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare shared flip/reversal policies on the current filterless "
            "DE3 + RegimeAdaptive + AetherFlow stack using one process per variant."
        )
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--pre-roll-days", type=int, default=180)
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        help="Repeatable strategy selection: de3, regimeadaptive, aetherflow.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Repeatable variant name. Defaults to all built-in variants.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Maximum number of variant processes to run at once.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help="BACKTEST_WORKERS to give each variant process.",
    )
    parser.add_argument(
        "--out",
        default="reports/combined_flip_policy_parallel/summary.json",
    )
    args = parser.parse_args()

    source_path = _resolve_source(str(args.source))
    selected_strategies = _normalize_selected_strategies(list(args.strategy or []))
    selected_variant_names = list(args.variant or [])
    if not selected_variant_names:
        selected_variant_names = list(DEFAULT_VARIANTS.keys())

    variants = {
        name: dict(DEFAULT_VARIANTS[name])
        for name in selected_variant_names
        if name in DEFAULT_VARIANTS
    }
    if not variants:
        raise SystemExit("No valid variants selected.")

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_parallel = max(1, min(int(args.max_parallel), len(variants)))
    summary = {
        "created_at": pd.Timestamp.now(tz="America/New_York").isoformat(),
        "source_path": str(source_path),
        "selected_strategies": list(selected_strategies),
        "selected_filters": [],
        "window": {"start": str(args.start), "end": str(args.end)},
        "symbol_mode": str(args.symbol_mode),
        "symbol_method": str(args.symbol_method),
        "pre_roll_days": int(args.pre_roll_days),
        "max_parallel": int(max_parallel),
        "worker_count_per_variant": int(max(1, int(args.worker_count))),
        "variants": {},
    }

    def write_summary() -> None:
        _attach_deltas(summary)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_summary()

    future_map = {}
    with cf.ProcessPoolExecutor(max_workers=max_parallel) as executor:
        for variant_name, variant_cfg in variants.items():
            future = executor.submit(
                _run_variant,
                source_path=str(source_path),
                start_raw=str(args.start),
                end_raw=str(args.end),
                symbol_mode=str(args.symbol_mode),
                symbol_method=str(args.symbol_method),
                pre_roll_days=int(args.pre_roll_days),
                selected_strategies=list(selected_strategies),
                variant_name=str(variant_name),
                variant_cfg=dict(variant_cfg),
                worker_count=int(args.worker_count),
            )
            future_map[future] = variant_name

        for future in cf.as_completed(future_map):
            variant_name = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                summary["variants"][variant_name] = {"error": str(exc)}
            else:
                summary["variants"][variant_name] = result
                reverse_stats = result.get("reverse_stats", {}) or {}
                print(
                    f"{variant_name}: equity={result.get('equity')} "
                    f"trades={result.get('trades')} dd={result.get('max_drawdown')} "
                    f"pf={result.get('profit_factor')} reverse_pnl={reverse_stats.get('pnl_net')}",
                    flush=True,
                )
            write_summary()

    print(f"Wrote summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
