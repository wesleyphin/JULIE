import argparse
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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


def _compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def _feature_cache_key(
    source_key: str,
    df: pd.DataFrame,
    atr_window: int,
    trend_needed: bool,
) -> str:
    start = df.index.min().isoformat() if not df.empty else ""
    end = df.index.max().isoformat() if not df.empty else ""
    rows = len(df)
    token = f"{source_key}|{start}|{end}|{rows}|{atr_window}|{int(trend_needed)}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _load_feature_cache(cache_dir: Path, key: str, use_cache: bool) -> dict:
    if not use_cache:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = cache_dir / f"cont_sltp_{key}"
    paths = {
        "atr": base.with_suffix(".atr.parquet"),
        "trend": base.with_suffix(".trend.parquet"),
    }
    payload: dict = {"_paths": paths}
    if paths["atr"].exists():
        try:
            payload["atr_df"] = pd.read_parquet(paths["atr"])
        except Exception as exc:
            logging.warning("Continuation SLTP cache read failed (atr): %s", exc)
    if paths["trend"].exists():
        try:
            payload["trend_df"] = pd.read_parquet(paths["trend"])
        except Exception as exc:
            logging.warning("Continuation SLTP cache read failed (trend): %s", exc)
    return payload


def _save_feature_cache(payload: dict, use_cache: bool) -> None:
    if not use_cache or not payload:
        return
    paths = payload.get("_paths") or {}
    atr_series = payload.get("atr_series")
    trend_context = payload.get("trend_context")
    if atr_series is not None and paths.get("atr"):
        try:
            pd.DataFrame({"atr": atr_series}).to_parquet(paths["atr"], index=True)
        except Exception as exc:
            logging.warning("Continuation SLTP cache write failed (atr): %s", exc)
    if trend_context is not None and paths.get("trend"):
        try:
            trend_df = {
                key: series
                for key, series in trend_context.items()
                if isinstance(series, pd.Series)
            }
            pd.DataFrame(trend_df).to_parquet(paths["trend"], index=True)
        except Exception as exc:
            logging.warning("Continuation SLTP cache write failed (trend): %s", exc)


def _round_to_tick(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    return round(value / tick) * tick


def _build_env_key(
    key_fields: list[str],
    session_name: str,
    regime: Optional[str],
    trend_tier: Optional[int],
) -> str:
    parts = []
    for field in key_fields:
        if field == "session":
            parts.append(session_name or "UNKNOWN")
        elif field == "regime":
            parts.append(regime or "UNKNOWN")
        elif field == "trend_tier":
            parts.append(str(trend_tier) if trend_tier is not None else "NA")
        else:
            parts.append("UNKNOWN")
    return "|".join(parts)


def _score_brackets(stats: dict, cfg: dict, fold_mode: str, loro_regimes: list[str]) -> dict:
    scored = {}
    min_total = int(cfg.get("min_total_trades", 0) or 0)
    min_fold_trades = int(cfg.get("min_fold_trades", 1) or 1)
    min_avg = float(cfg.get("min_avg_pnl_points", 0.0) or 0.0)
    min_win = float(cfg.get("min_win_rate", 0.0) or 0.0)
    min_fold_exp = float(cfg.get("min_fold_expectancy_points", 0.0) or 0.0)
    min_folds = int(cfg.get("min_folds", 1) or 1)
    min_pos_ratio = float(cfg.get("min_positive_fold_ratio", 0.0) or 0.0)

    for bracket, agg in stats.items():
        total_trades = agg["total_trades"]
        total_pnl = agg["total_pnl_points"]
        wins = agg["wins"]
        avg_pnl = total_pnl / total_trades if total_trades else 0.0
        win_rate = wins / total_trades if total_trades else 0.0

        folds_regime = 0
        positive_regime = 0
        folds_time = 0
        positive_time = 0
        if fold_mode in ("regime", "regime_time"):
            for regime in loro_regimes:
                r_stats = agg["per_regime"].get(regime)
                if not r_stats:
                    continue
                trades = r_stats["trades"]
                pnl_points = r_stats["pnl_points"]
                if trades >= min_fold_trades:
                    folds_regime += 1
                    expectancy = pnl_points / trades if trades else 0.0
                    if expectancy >= min_fold_exp:
                        positive_regime += 1
        if fold_mode in ("time", "regime_time"):
            for r_stats in agg["fold_stats"].values():
                trades = r_stats["trades"]
                pnl_points = r_stats["pnl_points"]
                if trades >= min_fold_trades:
                    folds_time += 1
                    expectancy = pnl_points / trades if trades else 0.0
                    if expectancy >= min_fold_exp:
                        positive_time += 1
        regime_ratio = (positive_regime / folds_regime) if folds_regime else 0.0
        time_ratio = (positive_time / folds_time) if folds_time else 0.0

        if fold_mode == "regime":
            folds_used = folds_regime
            pos_ratio = regime_ratio
            fold_ok = folds_used >= min_folds and pos_ratio >= min_pos_ratio
        elif fold_mode == "time":
            folds_used = folds_time
            pos_ratio = time_ratio
            fold_ok = folds_used >= min_folds and pos_ratio >= min_pos_ratio
        elif fold_mode == "regime_time":
            folds_used = min(folds_regime, folds_time)
            pos_ratio = min(regime_ratio, time_ratio)
            fold_ok = (
                folds_regime >= min_folds
                and regime_ratio >= min_pos_ratio
                and folds_time >= min_folds
                and time_ratio >= min_pos_ratio
            )
        else:
            folds_used = 1
            pos_ratio = 1 if avg_pnl >= min_fold_exp else 0.0
            fold_ok = True

        allowed = (
            total_trades >= min_total
            and avg_pnl >= min_avg
            and win_rate >= min_win
            and fold_ok
        )
        scored[bracket] = {
            "total_trades": total_trades,
            "avg_pnl_points": avg_pnl,
            "win_rate": win_rate,
            "folds": folds_used,
            "positive_ratio": pos_ratio,
            "folds_regime": folds_regime,
            "positive_ratio_regime": regime_ratio,
            "folds_time": folds_time,
            "positive_ratio_time": time_ratio,
            "allowed": allowed,
        }
    return scored


def _pick_best(scored: dict, prefer_allowed: bool = True) -> Optional[tuple]:
    if not scored:
        return None
    candidates = [(k, v) for k, v in scored.items() if v.get("allowed")]
    if not candidates and not prefer_allowed:
        candidates = list(scored.items())
    if not candidates:
        candidates = list(scored.items())

    def sort_key(item):
        stats = item[1]
        return (
            stats.get("avg_pnl_points", 0.0),
            stats.get("win_rate", 0.0),
            stats.get("total_trades", 0),
        )

    return max(candidates, key=sort_key)


def _train_sltp_from_df(
    df: pd.DataFrame,
    cfg: dict,
    confirm_cfg: dict,
    signal_mode: str,
    cache_dir: Optional[Path] = None,
    source_key: Optional[str] = None,
    use_cache: bool = True,
) -> dict:
    if df.empty:
        return {}

    key_fields = list(cfg.get("key_fields") or ["session"])
    fold_mode = str(cfg.get("fold_mode", "regime") or "regime").lower()
    loro_regimes = [str(r).lower() for r in (cfg.get("loro_regimes") or ["low", "normal", "high"])]
    if "regime" in key_fields and fold_mode in ("regime", "regime_time"):
        fold_mode = "time"

    sl_mults = cfg.get("sl_mults") or [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
    tp_mults = cfg.get("tp_mults") or [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    atr_window = int(cfg.get("atr_window", 14) or 14)
    max_horizon = int(cfg.get("max_horizon_bars", 120) or 120)
    exit_at_horizon = str(cfg.get("exit_at_horizon", "close") or "close").lower()
    assume_sl_first = bool(cfg.get("assume_sl_first", True))
    tick_size = float(cfg.get("tick_size", 0.25) or 0.25)
    min_sl = float(cfg.get("min_sl", 0.0) or 0.0)
    min_tp = float(cfg.get("min_tp", 0.0) or 0.0)

    trend_needed = signal_mode == "structure" or (confirm_cfg and confirm_cfg.get("enabled", True))
    feature_cache = {}
    if cache_dir and source_key:
        feature_key = _feature_cache_key(source_key, df, atr_window, trend_needed)
        feature_cache = _load_feature_cache(cache_dir, feature_key, use_cache)

    atr_series = None
    atr_df = feature_cache.get("atr_df")
    if atr_df is not None and "atr" in atr_df.columns and atr_df.index.equals(df.index):
        atr_series = atr_df["atr"]
    if atr_series is None:
        atr_series = _compute_atr(df, atr_window)
        feature_cache["atr_series"] = atr_series
    close = df["close"]
    prev_close = close.shift(1)

    trend_context = None
    if trend_needed:
        trend_df = feature_cache.get("trend_df")
        if trend_df is not None and trend_df.index.equals(df.index):
            trend_context = {col: trend_df[col] for col in trend_df.columns}
        if trend_context is None:
            trend_raw = bt.compute_trend_day_series(df)
            trend_context = bt.align_trend_day_series(trend_raw, df.index)
            feature_cache["trend_context"] = trend_context

    if feature_cache and cache_dir and use_cache:
        _save_feature_cache(feature_cache, use_cache=True)

    prior_high = None
    prior_low = None
    if trend_context is not None:
        prior_high = trend_context.get("prior_session_high")
        prior_low = trend_context.get("prior_session_low")

    structure_up = None
    structure_down = None
    if signal_mode == "structure" and prior_high is not None and prior_low is not None:
        structure_up = (close > prior_high) & (prev_close <= prior_high)
        structure_down = (close < prior_low) & (prev_close >= prior_low)

    stats = {}
    atr_values = {}
    base_ts = df.index[0].value
    span_ts = max(1, df.index[-1].value - base_ts)

    def fold_index(ts_val: int, folds: int) -> int:
        if folds <= 1:
            return 0
        return int(((ts_val - base_ts) / span_ts) * max(1, folds - 1))

    fold_count = int(cfg.get("folds", 4) or 4)

    if signal_mode == "structure" and structure_up is not None and structure_down is not None:
        candidate_positions = np.where((structure_up | structure_down).to_numpy())[0]
    else:
        session_series = pd.Series(df.index).map(bt.get_session_name)
        session_next = session_series.shift(-1)
        is_session_end = session_series != session_next
        candidate_positions = np.where(is_session_end.to_numpy())[0]

    candidate_positions = [pos for pos in candidate_positions if 0 < pos < len(df) - 1]

    for pos in candidate_positions:
        current_time = df.index[pos]
        session_name = bt.get_session_name(current_time)
        if session_name == "OFF":
            continue

        if signal_mode == "structure":
            is_up = bool(structure_up.iloc[pos]) if structure_up is not None else False
            is_down = bool(structure_down.iloc[pos]) if structure_down is not None else False
            if not (is_up or is_down):
                continue
            if is_up:
                side = "LONG"
            else:
                side = "SHORT"
            if confirm_cfg and not bt.continuation_market_confirmed(
                side, current_time, float(close.iloc[pos]), trend_context, confirm_cfg
            ):
                continue
        else:
            # Calendar mode: evaluate only on session close bars (like allowlist)
            if session_name not in ("ASIA", "LONDON", "NY_AM", "NY_PM"):
                continue
            # Use last bar of the session window
            next_time = df.index[pos + 1]
            if bt.get_session_name(next_time) == session_name:
                continue
            if confirm_cfg and not bt.continuation_market_confirmed(
                "LONG", current_time, float(close.iloc[pos]), trend_context, confirm_cfg
            ) and not bt.continuation_market_confirmed(
                "SHORT", current_time, float(close.iloc[pos]), trend_context, confirm_cfg
            ):
                continue
            long_ok = bt.continuation_market_confirmed(
                "LONG", current_time, float(close.iloc[pos]), trend_context, confirm_cfg
            )
            short_ok = bt.continuation_market_confirmed(
                "SHORT", current_time, float(close.iloc[pos]), trend_context, confirm_cfg
            )
            if long_ok and not short_ok:
                side = "LONG"
            elif short_ok and not long_ok:
                side = "SHORT"
            else:
                continue

        atr = float(atr_series.iloc[pos]) if not np.isnan(atr_series.iloc[pos]) else None
        if atr is None or atr <= 0:
            continue

        if "regime" in key_fields or fold_mode == "regime":
            history_df = df.iloc[: pos + 1]
            try:
                regime_val, _, _ = bt.volatility_filter.get_regime(history_df, ts=current_time)
            except Exception:
                regime_val = None
            regime = str(regime_val).lower() if regime_val else "unknown"
        else:
            regime = None

        env_key = _build_env_key(key_fields, session_name, regime, None)
        atr_values.setdefault(env_key, []).append(atr)

        stats.setdefault(env_key, {})

        entry_pos = pos + 1
        entry_price = float(df.iloc[entry_pos]["open"])
        ts_val = current_time.value
        fold_idx = fold_index(ts_val, fold_count)

        for sl_mult in sl_mults:
            for tp_mult in tp_mults:
                sl_dist = _round_to_tick(atr * float(sl_mult), tick_size)
                tp_dist = _round_to_tick(atr * float(tp_mult), tick_size)
                if sl_dist <= 0 or tp_dist <= 0:
                    continue
                if sl_dist < min_sl or tp_dist < min_tp:
                    continue

                pnl_points = bt.simulate_trade_points(
                    df,
                    entry_pos,
                    side,
                    entry_price,
                    sl_dist,
                    tp_dist,
                    max_horizon,
                    assume_sl_first,
                    exit_at_horizon,
                )

                bracket = (float(sl_mult), float(tp_mult))
                agg = stats[env_key].setdefault(
                    bracket,
                    {
                        "total_trades": 0,
                        "total_pnl_points": 0.0,
                        "wins": 0,
                        "per_regime": {},
                        "fold_stats": {},
                    },
                )
                agg["total_trades"] += 1
                agg["total_pnl_points"] += pnl_points
                if pnl_points > 0:
                    agg["wins"] += 1
                if regime is not None:
                    reg_stats = agg["per_regime"].setdefault(
                        regime, {"trades": 0, "pnl_points": 0.0, "wins": 0}
                    )
                    reg_stats["trades"] += 1
                    reg_stats["pnl_points"] += pnl_points
                    if pnl_points > 0:
                        reg_stats["wins"] += 1
                fold_stats = agg["fold_stats"].setdefault(
                    fold_idx, {"trades": 0, "pnl_points": 0.0}
                )
                fold_stats["trades"] += 1
                fold_stats["pnl_points"] += pnl_points

    results = {}
    for env_key, env_stats in stats.items():
        scored = _score_brackets(env_stats, cfg, fold_mode, loro_regimes)
        best = _pick_best(scored, prefer_allowed=True)
        if best is None:
            continue
        (sl_mult, tp_mult), best_stats = best
        atr_vals = atr_values.get(env_key) or []
        atr_med = float(np.median(atr_vals)) if atr_vals else 0.0
        results[env_key] = {
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
            "atr_med": round(atr_med, 4),
            "stats": best_stats,
        }

    payload = {
        "generated_at": dt.datetime.now(bt.NY_TZ).isoformat(),
        "signal_mode": signal_mode,
        "key_fields": key_fields,
        "criteria": {
            "min_total_trades": cfg.get("min_total_trades", 0),
            "min_fold_trades": cfg.get("min_fold_trades", 1),
            "min_avg_pnl_points": cfg.get("min_avg_pnl_points", 0.0),
            "min_win_rate": cfg.get("min_win_rate", 0.0),
            "min_fold_expectancy_points": cfg.get("min_fold_expectancy_points", 0.0),
            "min_folds": cfg.get("min_folds", 1),
            "min_positive_fold_ratio": cfg.get("min_positive_fold_ratio", 0.0),
            "fold_mode": fold_mode,
            "folds": fold_count,
            "loro_regimes": loro_regimes,
        },
        "grid": {"sl_mults": sl_mults, "tp_mults": tp_mults},
        "results": results,
    }
    if key_fields == ["session"]:
        payload["session_defaults"] = {k: v for k, v in results.items()}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train continuation SL/TP ATR brackets from CSV history.")
    parser.add_argument("--csv", default="es_master.csv", help="Path to CSV history file.")
    parser.add_argument("--out", default=None, help="Output JSON path (defaults to config file).")
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
    parser.add_argument("--recent-start", default="2023-01-01", help="Recent window start date.")
    parser.add_argument("--recent-end", default="2025-12-31", help="Recent window end date.")
    parser.add_argument(
        "--recent-mode",
        default="intersect",
        choices=("intersect", "union", "recent_only"),
        help="How to combine full vs recent bracket pools.",
    )
    parser.add_argument("--no-recent", action="store_true", help="Disable recency window.")
    parser.add_argument(
        "--signal-mode",
        default=None,
        choices=("structure", "calendar"),
        help="Continuation signal mode (defaults to BACKTEST_CONTINUATION_SIGNAL_MODE).",
    )
    parser.add_argument("--cache-dir", default="cache", help="Cache directory for parquet.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache read/write.")
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
        csv_path = Path(__file__).resolve().parent / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    use_cache = not args.no_cache
    df = _load_csv(csv_path, cache_dir, use_cache)
    start = _parse_date(train_start_raw, is_end=False)
    end = _parse_date(train_end_raw, is_end=True)
    df = _filter_range(df, start, end)

    train_cfg = CONFIG.get("CONTINUATION_SLTP_TRAIN", {}) or {}
    confirm_cfg = CONFIG.get("BACKTEST_CONTINUATION_CONFIRM", {}) or {}
    signal_mode = args.signal_mode or str(
        CONFIG.get("BACKTEST_CONTINUATION_SIGNAL_MODE", "structure") or "structure"
    ).lower()
    source_key = data_cache.cache_key_for_source(csv_path) if cache_dir else None

    full_payload = _train_sltp_from_df(
        df,
        train_cfg,
        confirm_cfg,
        signal_mode,
        cache_dir=cache_dir,
        source_key=source_key,
        use_cache=use_cache,
    )
    full_results = full_payload.get("results") or {}

    final_results = dict(full_results)
    recent_payload = None
    if not args.no_recent:
        recent_start = _parse_date(args.recent_start, is_end=False)
        recent_end = _parse_date(args.recent_end, is_end=True)
        if recent_start is not None or recent_end is not None:
            recent_df = _filter_range(df, recent_start, recent_end)
            recent_payload = _train_sltp_from_df(
                recent_df,
                train_cfg,
                confirm_cfg,
                signal_mode,
                cache_dir=cache_dir,
                source_key=source_key,
                use_cache=use_cache,
            )
            recent_results = recent_payload.get("results") or {}
            combined = {}
            for key in set(full_results) | set(recent_results):
                full_entry = full_results.get(key)
                recent_entry = recent_results.get(key)
                if args.recent_mode == "recent_only":
                    combined[key] = recent_entry or full_entry
                elif args.recent_mode == "union":
                    combined[key] = recent_entry or full_entry
                else:
                    if full_entry and recent_entry:
                        combined[key] = recent_entry
                    else:
                        combined[key] = full_entry or recent_entry
            final_results = combined

    payload = dict(full_payload)
    payload["results"] = final_results
    if payload.get("key_fields") == ["session"]:
        payload["session_defaults"] = {k: v for k, v in final_results.items()}
    if recent_payload is not None:
        payload["recent"] = {
            "mode": args.recent_mode,
            "results": recent_payload.get("results", {}),
        }

    out_name = args.out or CONFIG.get("CONTINUATION_SLTP_FILE") or "backtest_reports/continuation_sltp.json"
    if artifact_suffix:
        out_name = append_artifact_suffix(str(out_name), artifact_suffix)
    out_path = out_name
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    logging.info("Wrote continuation SL/TP params: %s", out_path)
    logging.info("Environments: %s", len(final_results))


if __name__ == "__main__":
    main()
