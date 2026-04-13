from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from backtest_symbol_context import attach_backtest_symbol_context


def _build_symbol_df(base_dir: Path, start_time, end_time):
    source_path = base_dir / bt.DEFAULT_CSV_NAME
    print(f"Loading source: {source_path}", flush=True)
    df = bt.load_csv_cached(source_path, cache_dir=base_dir / "cache", use_cache=True)
    if end_time is None:
        source_df = df
    else:
        source_df = df[df.index <= end_time]
    symbol_mode = str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single").lower()
    symbol_method = bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume")
    if symbol_mode != "auto_by_day":
        raise RuntimeError(f"Expected auto_by_day symbol mode, got {symbol_mode!r}")
    symbol_df, auto_label, _ = bt.apply_symbol_mode(source_df, symbol_mode, symbol_method)
    if "symbol" in symbol_df.columns:
        symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    source_attrs = getattr(source_df, "attrs", {}) or {}
    symbol_df = attach_backtest_symbol_context(
        symbol_df,
        auto_label,
        symbol_mode,
        source_key=source_attrs.get("source_cache_key"),
        source_label=source_attrs.get("source_label"),
        source_path=source_attrs.get("source_path"),
    )
    return df, symbol_df


def _configure_break_even_only():
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


def _write_converted_csv(trade_log: list[dict], out_path: Path) -> None:
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


def main() -> None:
    base_dir = ROOT
    out_dir = base_dir / "backtest_reports" / "de3_v4_trade_mgmt_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_df, symbol_df = _build_symbol_df(base_dir, None, None)
    start_time = source_df.index.min()
    end_time = source_df.index.max()
    print(f"Full dataset range: {start_time} -> {end_time}", flush=True)

    _configure_break_even_only()

    print("Running full-history DE3 v4 filterless backtest with break-even only...", flush=True)
    stats = bt.run_backtest(
        symbol_df,
        start_time,
        end_time,
        enabled_strategies={"DynamicEngine3Strategy"},
        enabled_filters=set(),
    )
    trade_log = stats.get("trade_log", []) or []

    csv_path = out_dir / "converted_break_even_only_full_history_de3_v4_trades.csv"
    _write_converted_csv(trade_log, csv_path)

    exit_reason_counts: dict[str, int] = {}
    for trade in trade_log:
        reason = str(trade.get("exit_reason", "unknown"))
        exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
    summary = {
        "variant": "break_even_only_full_history",
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "csv_path": str(csv_path),
        "summary": {
            "equity": stats.get("equity"),
            "trades": stats.get("trades"),
            "wins": stats.get("wins"),
            "losses": stats.get("losses"),
            "winrate": stats.get("winrate"),
            "max_drawdown": stats.get("max_drawdown"),
        },
        "trade_management": stats.get("de3_trade_management"),
        "exit_reason_counts": exit_reason_counts,
        "break_even_applied_trades": sum(
            1 for trade in trade_log if trade.get("de3_break_even_applied")
        ),
        "break_even_stop_hits": sum(
            1 for trade in trade_log if trade.get("de3_break_even_stop_hit")
        ),
    }
    summary_path = out_dir / "break_even_only_full_history_de3_v4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"CSV: {csv_path}", flush=True)
    print(f"SUMMARY: {summary_path}", flush=True)
    print(json.dumps(summary["summary"]), flush=True)


if __name__ == "__main__":
    main()
