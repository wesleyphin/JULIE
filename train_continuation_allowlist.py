import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from collections import defaultdict

import backtest_mes_et as bt
import data_cache
from config import (
    CONFIG,
    append_artifact_suffix,
    get_experimental_training_window,
    resolve_artifact_suffix,
)


def _load_csv(csv_path: Path, cache_dir: Optional[Path], use_cache: bool) -> pd.DataFrame:
    return data_cache.load_bars(csv_path, cache_dir=cache_dir, use_cache=use_cache)


def _parse_date(value: Optional[str], *, is_end: bool = False) -> Optional[pd.Timestamp]:
    if not value:
        return None
    raw = str(value).strip()
    has_time = ("T" in raw) or (":" in raw)
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value}")
    if ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    else:
        ts = ts.tz_convert("US/Eastern")
    if is_end and not has_time:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _filter_range(df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if df.empty:
        return df
    if start is not None:
        df = df.loc[df.index >= start]
    if end is not None:
        df = df.loc[df.index <= end]
    return df


def _parse_trade_time(value) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    else:
        ts = ts.tz_convert("US/Eastern")
    return ts


def _session_from_hour(hour: int) -> str:
    if hour >= 18 or hour < 3:
        return "Asia"
    if 3 <= hour < 8:
        return "London"
    if 8 <= hour < 17:
        return "NY"
    return "Other"


def _key_from_time(ts: pd.Timestamp, key_mode: str) -> Optional[str]:
    if ts is None:
        return None
    mode = str(key_mode or "full").lower()
    session = _session_from_hour(ts.hour)
    if session == "Other":
        return None
    day = ts.weekday() + 1
    if mode in ("session", "sess"):
        return session.upper()
    if mode in ("session_day", "day_session", "dow_session"):
        return f"D{day}_{session.upper()}"
    if mode in ("full", "qwd", "qwd_session"):
        quarter = (ts.month - 1) // 3 + 1
        week = int(ts.isocalendar().week)
        return f"Q{quarter}_W{week}_D{day}_{session}"
    return None


def _resolve_trade_regime(
    trade: dict,
    df: pd.DataFrame,
    entry_time: pd.Timestamp,
    allowed_regimes: set,
) -> Optional[str]:
    if not allowed_regimes:
        return None
    regime_raw = trade.get("vol_regime")
    if regime_raw:
        regime_key = str(regime_raw).lower()
        if regime_key not in ("unknown", "none", "nan"):
            return regime_key
    try:
        history_df = df.loc[:entry_time]
        regime_val, _, _ = bt.volatility_filter.get_regime(history_df, ts=entry_time)
        return str(regime_val).lower() if regime_val else None
    except Exception:
        return None


def _filter_trades_by_time(
    trades: list[dict],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> list[dict]:
    if not start and not end:
        return trades
    filtered = []
    for trade in trades:
        entry_time = _parse_trade_time(trade.get("entry_time"))
        if entry_time is None:
            continue
        if start is not None and entry_time < start:
            continue
        if end is not None and entry_time > end:
            continue
        filtered.append(trade)
    return filtered


def _build_allowlist_from_trades(
    trades: list[dict],
    df: pd.DataFrame,
    allow_cfg: dict,
    allowed_regimes: set,
) -> tuple[set, dict]:
    key_mode = str(allow_cfg.get("key_granularity", "full") or "full").lower()
    fast_cfg = allow_cfg.get("fast", {}) or {}
    folds = int(fast_cfg.get("folds", 4) or 4)
    min_win_rate = float(fast_cfg.get("min_win_rate", 0.0) or 0.0)

    min_total_trades = int(allow_cfg.get("min_total_trades", 0) or 0)
    min_fold_trades = int(allow_cfg.get("min_fold_trades", 1) or 1)
    min_avg_pnl_points = float(allow_cfg.get("min_avg_pnl_points", 0.0) or 0.0)
    min_fold_expectancy_points = float(allow_cfg.get("min_fold_expectancy_points", 0.0) or 0.0)
    min_folds = int(allow_cfg.get("min_folds", 1) or 1)
    min_positive_fold_ratio = float(allow_cfg.get("min_positive_fold_ratio", 0.0) or 0.0)

    base_ts = df.index[0].value
    span_ts = max(1, df.index[-1].value - base_ts)

    aggregate = defaultdict(
        lambda: {
            "total_trades": 0,
            "total_pnl_points": 0.0,
            "wins": 0,
            "fold_stats": defaultdict(lambda: {"trades": 0, "pnl_points": 0.0}),
        }
    )

    for trade in trades:
        strategy = str(trade.get("strategy") or "")
        if not strategy.startswith("Continuation_") and strategy != "Continuation_Structure":
            continue
        if str(trade.get("entry_mode") or "") != "rescued":
            continue

        entry_time = _parse_trade_time(trade.get("entry_time"))
        if entry_time is None:
            continue

        raw_key = bt.parse_continuation_key(strategy)
        key = bt.continuation_allowlist_key(raw_key, key_mode)
        if not key:
            key = _key_from_time(entry_time, key_mode)
        if not key:
            continue
        if key_mode == "full" and key not in bt.STRATEGY_CONFIGS:
            continue

        pnl_points = trade.get("pnl_points")
        if pnl_points is None:
            pnl_net = float(trade.get("pnl_net", 0.0) or 0.0)
            denom = bt.POINT_VALUE * bt.CONTRACTS
            pnl_points = pnl_net / denom if denom else 0.0
        else:
            pnl_points = float(pnl_points)

        if allowed_regimes:
            regime_key = _resolve_trade_regime(trade, df, entry_time, allowed_regimes)
            if not regime_key or regime_key not in allowed_regimes:
                continue

        fold_idx = 0
        if folds > 1:
            fold_idx = int(((entry_time.value - base_ts) / span_ts) * max(1, folds - 1))

        agg = aggregate[key]
        agg["total_trades"] += 1
        agg["total_pnl_points"] += pnl_points
        if pnl_points > 0:
            agg["wins"] += 1
        fold_stats = agg["fold_stats"][fold_idx]
        fold_stats["trades"] += 1
        fold_stats["pnl_points"] += pnl_points

    allowlist = set()
    stats_out = {}
    for key, agg in aggregate.items():
        total_trades = agg["total_trades"]
        total_pnl_points = agg["total_pnl_points"]
        wins = agg["wins"]
        avg_pnl = total_pnl_points / total_trades if total_trades else 0.0
        win_rate = wins / total_trades if total_trades else 0.0

        folds_used = 0
        positive_folds = 0
        for stats in agg["fold_stats"].values():
            trades = stats["trades"]
            pnl_points = stats["pnl_points"]
            if trades >= min_fold_trades:
                folds_used += 1
                expectancy = pnl_points / trades if trades else 0.0
                if expectancy >= min_fold_expectancy_points:
                    positive_folds += 1
        positive_ratio = (positive_folds / folds_used) if folds_used else 0.0

        allowed = (
            total_trades >= min_total_trades
            and avg_pnl >= min_avg_pnl_points
            and win_rate >= min_win_rate
            and folds_used >= min_folds
            and positive_ratio >= min_positive_fold_ratio
        )

        stats_out[key] = {
            "total_trades": total_trades,
            "avg_pnl_points": avg_pnl,
            "win_rate": win_rate,
            "folds": folds_used,
            "positive_ratio": positive_ratio,
            "allowed": allowed,
        }
        if allowed:
            allowlist.add(key)

    payload = {
        "generated_at": dt.datetime.now(bt.NY_TZ).isoformat(),
        "mode": "rescue_replay",
        "summary": {
            "keys_seen": len(stats_out),
            "keys_allowed": len(allowlist),
            "trades_seen": sum(v["total_trades"] for v in stats_out.values()),
        },
        "criteria": {
            "min_total_trades": min_total_trades,
            "min_fold_trades": min_fold_trades,
            "min_avg_pnl_points": min_avg_pnl_points,
            "min_fold_expectancy_points": min_fold_expectancy_points,
            "min_folds": min_folds,
            "min_positive_fold_ratio": min_positive_fold_ratio,
            "min_win_rate": min_win_rate,
            "folds": folds,
            "key_granularity": key_mode,
            "allowed_regimes": sorted(allowed_regimes) if allowed_regimes else [],
        },
        "allowlist": sorted(allowlist),
        "stats": stats_out,
    }
    return allowlist, payload


def _run_allowlist(
    df: pd.DataFrame,
    allow_cfg: dict,
    allowed_regimes: set,
    confirm_cfg: dict,
) -> tuple[set, dict]:
    cfg = dict(allow_cfg)
    cfg["cache_file"] = None
    allowlist, payload = bt.build_continuation_allowlist_from_df(
        df,
        trend_context=None,
        cfg=cfg,
        allowed_regimes=allowed_regimes,
        confirm_cfg=confirm_cfg,
    )
    allowlist = allowlist or set()
    return allowlist, payload or {}


def _parse_regime_list(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _run_allowlist_loro(
    df: pd.DataFrame,
    allow_cfg: dict,
    confirm_cfg: dict,
    regimes: list[str],
) -> tuple[set, dict]:
    combined = None
    per_regime = {}
    for regime in regimes:
        allowlist, payload = _run_allowlist(df, allow_cfg, {regime}, confirm_cfg)
        per_regime[regime] = {
            "summary": payload.get("summary", {}),
            "allowlist": sorted(allowlist),
            "stats": payload.get("stats", {}),
        }
        if combined is None:
            combined = set(allowlist)
        else:
            combined &= set(allowlist)
    if combined is None:
        combined = set()
    payload = {
        "mode": "csv_fast_loro",
        "regimes": regimes,
        "per_regime": per_regime,
    }
    return combined, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train continuation allowlist from CSV history.")
    parser.add_argument("--csv", default="es_master.csv", help="Path to CSV history file.")
    parser.add_argument("--out", default=None, help="Output JSON path (defaults to config cache_file).")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--experimental-window",
        action="store_true",
        help="Train only on configured experimental window (2011-01-01 .. 2017-12-31).",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Suffix appended to output artifacts (e.g. _exp2011_2017).",
    )
    parser.add_argument(
        "--recent-start",
        default="2023-01-01",
        help="Recent window start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--recent-end",
        default="2025-12-31",
        help="Recent window end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--regime-loro",
        action="store_true",
        dest="regime_loro",
        help="Intersect allowlists across regimes (default: disabled).",
    )
    parser.add_argument(
        "--no-regime-loro",
        action="store_false",
        dest="regime_loro",
        help="Disable LORO intersection.",
    )
    parser.add_argument(
        "--no-recent",
        action="store_true",
        help="Disable recency window.",
    )
    parser.add_argument(
        "--loro-regimes",
        default="low,normal,high",
        help="Comma-separated regimes for LORO (default: low,normal,high).",
    )
    parser.add_argument(
        "--regimes",
        default=None,
        help="Comma-separated regimes to include when not using LORO.",
    )
    parser.add_argument(
        "--recent-mode",
        default="intersect",
        choices=("intersect", "union", "recent_only"),
        help="How to combine full vs recent allowlists.",
    )
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Use fast proxy signals instead of rescue replay.",
    )
    parser.add_argument("--cache-dir", default="cache", help="Cache directory for parquet.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache read/write.")
    parser.set_defaults(regime_loro=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    exp_enabled = bool(args.experimental_window)
    train_start_raw = args.start
    train_end_raw = args.end
    if exp_enabled:
        exp_start, exp_end = get_experimental_training_window()
        train_start_raw = exp_start
        train_end_raw = exp_end
        logging.info("Experimental window enabled: %s -> %s", train_start_raw, train_end_raw)
    artifact_suffix = resolve_artifact_suffix(args.artifact_suffix, exp_enabled)
    if exp_enabled and not args.no_recent:
        args.recent_start = train_start_raw
        args.recent_end = train_end_raw
        logging.info("Experimental mode: recent window aligned to %s -> %s", args.recent_start, args.recent_end)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    df = _load_csv(csv_path, cache_dir, not args.no_cache)
    start = _parse_date(train_start_raw, is_end=False)
    end = _parse_date(train_end_raw, is_end=True)
    df = _filter_range(df, start, end)
    if df.empty:
        raise ValueError("No rows available for training after applying the date range.")

    symbol_mode = str(CONFIG.get("BACKTEST_SYMBOL_MODE", "auto_by_day") or "auto_by_day").lower()
    symbol_method = CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume")
    if "symbol" in df.columns and df["symbol"].nunique(dropna=True) > 1:
        if symbol_mode != "single":
            df, _, _ = bt.apply_symbol_mode(df, symbol_mode, symbol_method)
        else:
            preferred_symbol = CONFIG.get("TARGET_SYMBOL")
            symbol = bt.choose_symbol(df, preferred_symbol)
            df = df[df["symbol"] == symbol]
        if df.empty:
            raise ValueError("No rows available after applying symbol selection.")
        if symbol_mode != "single":
            df = df.drop(columns=["symbol"], errors="ignore")

    allow_cfg = CONFIG.get("BACKTEST_CONTINUATION_ALLOWLIST", {}) or {}
    allowed_regimes_cfg = {
        str(item).lower()
        for item in (CONFIG.get("BACKTEST_CONTINUATION_ALLOWED_REGIMES") or [])
        if item is not None
    }
    confirm_cfg = CONFIG.get("BACKTEST_CONTINUATION_CONFIRM", {}) or {}

    if args.proxy:
        if args.regime_loro:
            loro_regimes = _parse_regime_list(args.loro_regimes)
            if not loro_regimes:
                loro_regimes = ["low", "normal", "high"]
            full_allow, full_payload = _run_allowlist_loro(df, allow_cfg, confirm_cfg, loro_regimes)
        else:
            override_regimes = _parse_regime_list(args.regimes)
            allowed_regimes = set(override_regimes) if override_regimes else allowed_regimes_cfg
            full_allow, full_payload = _run_allowlist(df, allow_cfg, allowed_regimes, confirm_cfg)
        trades_full = None
    else:
        override_regimes = _parse_regime_list(args.regimes)
        allowed_regimes = set(override_regimes) if override_regimes else allowed_regimes_cfg
        start_time = start or df.index.min()
        end_time = end or df.index.max()
        prev_allow_cfg = CONFIG.get("BACKTEST_CONTINUATION_ALLOWLIST")
        allow_cfg_override = dict(allow_cfg)
        allow_cfg_override["enabled"] = False
        CONFIG["BACKTEST_CONTINUATION_ALLOWLIST"] = allow_cfg_override
        try:
            stats = bt.run_backtest(df, start_time, end_time)
        finally:
            CONFIG["BACKTEST_CONTINUATION_ALLOWLIST"] = prev_allow_cfg
        trades_full = stats.get("trade_log") or []
        full_allow, full_payload = _build_allowlist_from_trades(
            trades_full,
            df,
            allow_cfg,
            allowed_regimes,
        )

    recent_allow = None
    recent_payload = None
    if not args.no_recent:
        recent_start = _parse_date(args.recent_start, is_end=False)
        recent_end = _parse_date(args.recent_end, is_end=True)
    else:
        recent_start = None
        recent_end = None
    if recent_start is not None or recent_end is not None:
        if args.proxy:
            recent_df = _filter_range(df, recent_start, recent_end)
            if args.regime_loro:
                recent_allow, recent_payload = _run_allowlist_loro(recent_df, allow_cfg, confirm_cfg, loro_regimes)
            else:
                recent_allow, recent_payload = _run_allowlist(recent_df, allow_cfg, allowed_regimes, confirm_cfg)
        else:
            recent_trades = _filter_trades_by_time(trades_full or [], recent_start, recent_end)
            recent_allow, recent_payload = _build_allowlist_from_trades(
                recent_trades,
                df,
                allow_cfg,
                allowed_regimes,
            )
        if args.recent_mode == "union":
            final_allow = full_allow | recent_allow
        elif args.recent_mode == "recent_only":
            final_allow = set(recent_allow)
        else:
            final_allow = full_allow & recent_allow
    else:
        final_allow = set(full_allow)

    out_name = args.out or allow_cfg.get("cache_file") or "backtest_reports/continuation_allowlist.json"
    if artifact_suffix:
        out_name = append_artifact_suffix(str(out_name), artifact_suffix)
    out_path = out_name
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": dt.datetime.now(bt.NY_TZ).isoformat(),
        "mode": full_payload.get("mode", "csv_fast"),
        "summary": {
            "full_keys_allowed": len(full_allow),
            "final_keys_allowed": len(final_allow),
        },
        "criteria": full_payload.get("criteria", allow_cfg),
        "allowlist": sorted(final_allow),
        "stats": full_payload.get("stats", {}),
    }
    if full_payload.get("per_regime"):
        payload["per_regime"] = full_payload.get("per_regime")
        payload["regimes"] = full_payload.get("regimes")
    if recent_payload is not None:
        payload["recent"] = {
            "mode": args.recent_mode,
            "summary": recent_payload.get("summary", {}),
            "allowlist": sorted(recent_allow or []),
            "stats": recent_payload.get("stats", {}),
        }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    logging.info("Wrote continuation allowlist: %s", out_path)
    logging.info("Allowlist size: %s", len(final_allow))


if __name__ == "__main__":
    main()
