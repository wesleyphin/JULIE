from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from config import CONFIG
from tools.run_de3_backtest import _prepare_symbol_df


VARIANTS: dict[str, dict] = {
    "baseline": {},
    "trail40": {
        "break_even": {
            "post_activation_trail_pct": 0.40,
        },
    },
    "trail30": {
        "break_even": {
            "post_activation_trail_pct": 0.30,
        },
    },
    "trail35": {
        "break_even": {
            "post_activation_trail_pct": 0.35,
        },
    },
    "trail50": {
        "break_even": {
            "post_activation_trail_pct": 0.50,
        },
    },
    "trail35_trigger35": {
        "break_even": {
            "trigger_pct": 0.35,
            "post_activation_trail_pct": 0.35,
        },
    },
    "trail40_trigger35": {
        "break_even": {
            "trigger_pct": 0.35,
            "post_activation_trail_pct": 0.40,
        },
    },
    "tiered50_trail40": {
        "break_even": {
            "post_activation_trail_pct": 0.40,
            "post_partial_trail_pct": 0.45,
        },
        "tiered_take_profit": {
            "enabled": True,
            "trigger_pct": 0.50,
            "close_fraction": 0.50,
            "min_entry_contracts": 2,
            "min_remaining_contracts": 1,
            "arm_break_even_after_fill": True,
        },
    },
    "tiered60_trail40": {
        "break_even": {
            "post_activation_trail_pct": 0.40,
            "post_partial_trail_pct": 0.45,
        },
        "tiered_take_profit": {
            "enabled": True,
            "trigger_pct": 0.60,
            "close_fraction": 0.50,
            "min_entry_contracts": 2,
            "min_remaining_contracts": 1,
            "arm_break_even_after_fill": True,
        },
    },
    "tiered60_trail35": {
        "break_even": {
            "post_activation_trail_pct": 0.35,
            "post_partial_trail_pct": 0.40,
        },
        "tiered_take_profit": {
            "enabled": True,
            "trigger_pct": 0.60,
            "close_fraction": 0.50,
            "min_entry_contracts": 2,
            "min_remaining_contracts": 1,
            "arm_break_even_after_fill": True,
        },
    },
    "tiered50_3lot_only": {
        "break_even": {
            "post_activation_trail_pct": 0.35,
            "post_partial_trail_pct": 0.40,
        },
        "tiered_take_profit": {
            "enabled": True,
            "trigger_pct": 0.50,
            "close_fraction": 0.34,
            "min_entry_contracts": 3,
            "min_remaining_contracts": 2,
            "arm_break_even_after_fill": True,
        },
    },
    "tiered60_3lot_only": {
        "break_even": {
            "post_activation_trail_pct": 0.35,
            "post_partial_trail_pct": 0.40,
        },
        "tiered_take_profit": {
            "enabled": True,
            "trigger_pct": 0.60,
            "close_fraction": 0.34,
            "min_entry_contracts": 3,
            "min_remaining_contracts": 2,
            "arm_break_even_after_fill": True,
        },
    },
}


def _resolve_source(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Data file not found: {path_arg}")


def _apply_trade_management_patch(base_cfg: dict, patch: dict) -> None:
    CONFIG.clear()
    CONFIG.update(copy.deepcopy(base_cfg))
    trade_mgmt = CONFIG.setdefault("DE3_V4", {}).setdefault("runtime", {}).setdefault(
        "trade_management",
        {},
    )
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(trade_mgmt.get(key), dict):
            merged = copy.deepcopy(trade_mgmt.get(key) or {})
            merged.update(value)
            trade_mgmt[key] = merged
        else:
            trade_mgmt[key] = copy.deepcopy(value)


def _run_variant(
    *,
    source_path: Path,
    start_time,
    end_time,
    symbol_mode: str,
    symbol_method: str,
    pre_roll_days: int,
    variant_name: str,
    patch: dict,
    report_dir: Path,
) -> dict:
    symbol_df, symbol, symbol_distribution = _prepare_symbol_df(
        source_path,
        start_time,
        end_time,
        symbol_mode,
        symbol_method,
        pre_roll_days=pre_roll_days,
    )
    base_cfg = copy.deepcopy(CONFIG)
    try:
        _apply_trade_management_patch(base_cfg, patch)
        stats = bt.run_backtest(
            symbol_df,
            start_time,
            end_time,
            enabled_strategies={"DynamicEngine3Strategy"},
            enabled_filters=set(),
        )
    finally:
        CONFIG.clear()
        CONFIG.update(base_cfg)

    trade_mgmt = stats.get("de3_trade_management", {}) or {}
    summary = {
        "variant": variant_name,
        "patch": copy.deepcopy(patch),
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "symbol_mode": symbol_mode,
        "symbol_method": symbol_method,
        "symbol_label": symbol,
        "symbol_distribution": symbol_distribution,
        "pf": float(stats.get("profit_factor", 0.0) or 0.0),
        "equity": float(stats.get("equity", 0.0) or 0.0),
        "max_drawdown": float(stats.get("max_drawdown", 0.0) or 0.0),
        "trades": int(stats.get("trades", 0) or 0),
        "wins": int(stats.get("wins", 0) or 0),
        "losses": int(stats.get("losses", 0) or 0),
        "winrate": float(stats.get("winrate", 0.0) or 0.0),
        "de3_trade_management": trade_mgmt,
    }
    out_path = report_dir / f"{variant_name}.summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DE3 trade-management variants.")
    parser.add_argument("--source", default=bt.DEFAULT_CSV_NAME)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--symbol-mode", default=str(bt.CONFIG.get("BACKTEST_SYMBOL_MODE", "auto_by_day") or "auto_by_day"))
    parser.add_argument("--symbol-method", default=str(bt.CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume"))
    parser.add_argument("--pre-roll-days", type=int, default=0)
    parser.add_argument("--variant", action="append", default=[], help="Variant name to run. Repeatable. Defaults to all.")
    parser.add_argument("--report-dir", default="backtest_reports/de3_trade_mgmt_eval_20260408")
    args = parser.parse_args()

    source_path = _resolve_source(str(args.source))
    start_time = bt.parse_user_datetime(str(args.start), bt.NY_TZ, is_end=False)
    end_time = bt.parse_user_datetime(str(args.end), bt.NY_TZ, is_end=True)
    if start_time > end_time:
        raise SystemExit("Start must be before end.")

    requested = [str(v or "").strip() for v in (args.variant or []) if str(v or "").strip()]
    if requested:
        unknown = [name for name in requested if name not in VARIANTS]
        if unknown:
            raise SystemExit(f"Unknown variant(s): {', '.join(unknown)}")
        variant_names = requested
    else:
        variant_names = list(VARIANTS.keys())

    report_dir = ROOT / str(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    leaderboard: list[dict] = []
    for variant_name in variant_names:
        summary = _run_variant(
            source_path=source_path,
            start_time=start_time,
            end_time=end_time,
            symbol_mode=str(args.symbol_mode or "auto_by_day").strip().lower(),
            symbol_method=str(args.symbol_method or "volume").strip().lower(),
            pre_roll_days=max(0, int(args.pre_roll_days or 0)),
            variant_name=variant_name,
            patch=VARIANTS[variant_name],
            report_dir=report_dir,
        )
        leaderboard.append(summary)
        print(
            json.dumps(
                {
                    "variant": variant_name,
                    "pf": summary["pf"],
                    "equity": summary["equity"],
                    "max_drawdown": summary["max_drawdown"],
                    "trades": summary["trades"],
                    "tiered_take_fill_count": int(
                        (summary.get("de3_trade_management", {}) or {}).get("tiered_take_fill_count", 0)
                    ),
                }
            ),
            flush=True,
        )

    leaderboard.sort(
        key=lambda item: (
            float(item.get("pf", 0.0) or 0.0),
            float(item.get("equity", 0.0) or 0.0),
            -float(item.get("max_drawdown", 0.0) or 0.0),
        ),
        reverse=True,
    )
    leaderboard_path = report_dir / "leaderboard.json"
    leaderboard_path.write_text(json.dumps(leaderboard, indent=2, default=str), encoding="utf-8")
    print(f"leaderboard={leaderboard_path}", flush=True)


if __name__ == "__main__":
    main()
