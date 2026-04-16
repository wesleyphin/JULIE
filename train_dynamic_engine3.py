import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import logging
import math
import os
import time
from collections import defaultdict
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


TICK_SIZE = 0.25
NY_TZ = bt.NY_TZ


def _load_csv(csv_path: Path) -> pd.DataFrame:
    with csv_path.open("r", errors="ignore") as f:
        first = f.readline()
        second = f.readline()
        needs_skip = "Time Series" in first and "Date" in second

    df = pd.read_csv(csv_path, skiprows=1 if needs_skip else 0)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = None
    if "ts_event" in df.columns:
        date_col = "ts_event"
    elif "timestamp" in df.columns:
        date_col = "timestamp"
    elif "date" in df.columns:
        date_col = "date"
    if date_col is None:
        raise ValueError("CSV missing timestamp column")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', "").str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df.dropna(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    keep_cols = [c for c in ["open", "high", "low", "close", "volume", "symbol"] if c in df.columns]
    df = df[keep_cols]
    return df


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


def _session_bucket(ts: pd.Timestamp) -> Optional[str]:
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    hour = ts.hour
    session_start = (hour // 3) * 3
    session_end = session_start + 3
    if session_end > 24:
        return None
    return f"{session_start:02d}-{session_end:02d}"


def _next_session_boundary(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    hour = ts.hour
    session_end_hour = ((hour // 3) + 1) * 3
    day_offset = 0
    if session_end_hour >= 24:
        session_end_hour = 0
        day_offset = 1
    midnight = ts.normalize()
    return midnight + pd.Timedelta(days=day_offset, hours=session_end_hour)


def _round_to_tick(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    return round(value / tick) * tick


def _feature_cache_key(
    source_key: str,
    df: pd.DataFrame,
    symbol_mode: str,
    symbol_method: str,
) -> str:
    start = df.index.min().isoformat() if not df.empty else ""
    end = df.index.max().isoformat() if not df.empty else ""
    rows = len(df)
    token = f"{source_key}|{start}|{end}|{rows}|{symbol_mode}|{symbol_method}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _load_feature_cache(
    cache_dir: Path,
    key: str,
    folds: int,
    use_cache: bool,
) -> dict:
    if not use_cache:
        return {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    base = cache_dir / f"dyn3_{key}"
    paths = {
        "df_5m": base.with_suffix(".5m.parquet"),
        "df_15m": base.with_suffix(".15m.parquet"),
        "regime": base.with_suffix(".regime.npy"),
        "folds": base.with_suffix(f".folds{folds}.npy"),
    }
    if paths["df_5m"].exists() and paths["df_15m"].exists():
        try:
            payload["df_5m"] = pd.read_parquet(paths["df_5m"])
            payload["df_15m"] = pd.read_parquet(paths["df_15m"])
        except Exception as exc:
            logging.warning("DynamicEngine3 cache read failed: %s", exc)
    if paths["regime"].exists():
        try:
            payload["regime_arr"] = np.load(paths["regime"], allow_pickle=True)
        except Exception as exc:
            logging.warning("DynamicEngine3 regime cache read failed: %s", exc)
    if paths["folds"].exists():
        try:
            payload["fold_edges"] = np.load(paths["folds"], allow_pickle=True)
        except Exception as exc:
            logging.warning("DynamicEngine3 fold cache read failed: %s", exc)
    payload["_paths"] = paths
    return payload


def _save_feature_cache(payload: dict, use_cache: bool) -> None:
    if not use_cache or not payload:
        return
    paths = payload.get("_paths") or {}
    df_5m = payload.get("df_5m")
    df_15m = payload.get("df_15m")
    regime_arr = payload.get("regime_arr")
    fold_edges = payload.get("fold_edges")
    if df_5m is not None and paths.get("df_5m"):
        try:
            df_5m.to_parquet(paths["df_5m"], index=True)
        except Exception as exc:
            logging.warning("DynamicEngine3 cache write failed (5m): %s", exc)
    if df_15m is not None and paths.get("df_15m"):
        try:
            df_15m.to_parquet(paths["df_15m"], index=True)
        except Exception as exc:
            logging.warning("DynamicEngine3 cache write failed (15m): %s", exc)
    if regime_arr is not None and paths.get("regime"):
        try:
            np.save(paths["regime"], regime_arr)
        except Exception as exc:
            logging.warning("DynamicEngine3 regime cache write failed: %s", exc)
    if fold_edges is not None and paths.get("folds"):
        try:
            np.save(paths["folds"], fold_edges)
        except Exception as exc:
            logging.warning("DynamicEngine3 fold cache write failed: %s", exc)


def _resample_df(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty:
        return df
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"
    return df.resample(f"{minutes}min", closed="left", label="left").agg(agg).dropna()


def _simulate_trade_points_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_pos: int,
    side: str,
    entry_price: float,
    sl_dist: float,
    tp_dist: float,
    last_pos: int,
    assume_sl_first: bool,
    exit_at_horizon: str,
) -> float:
    last_pos = min(last_pos, len(high) - 1)
    for pos in range(entry_pos, last_pos + 1):
        h = float(high[pos])
        l = float(low[pos])
        if side == "LONG":
            hit_tp = h >= entry_price + tp_dist
            hit_sl = l <= entry_price - sl_dist
            if hit_tp and hit_sl:
                return -sl_dist if assume_sl_first else tp_dist
            if hit_tp:
                return tp_dist
            if hit_sl:
                return -sl_dist
        else:
            hit_tp = l <= entry_price - tp_dist
            hit_sl = h >= entry_price + sl_dist
            if hit_tp and hit_sl:
                return -sl_dist if assume_sl_first else tp_dist
            if hit_tp:
                return tp_dist
            if hit_sl:
                return -sl_dist

    if exit_at_horizon == "close":
        exit_price = float(close[last_pos])
        return (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
    return 0.0


def _compute_regimes(df_1m: pd.DataFrame) -> pd.Series:
    if df_1m.empty:
        return pd.Series(dtype=str)
    close = df_1m["close"].astype(float)
    vol = close.pct_change().rolling(20).std()
    vol = vol.fillna(method="bfill")
    q_low = vol.quantile(0.33)
    q_high = vol.quantile(0.66)
    regimes = pd.Series(index=vol.index, dtype=str)
    regimes[vol <= q_low] = "low"
    regimes[(vol > q_low) & (vol < q_high)] = "normal"
    regimes[vol >= q_high] = "high"
    return regimes


def _summarize_pnls(pnls: list[float]) -> dict:
    trades = len(pnls)
    if trades <= 0:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
        }
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / trades if trades else 0.0
    avg_pnl = sum(pnls) / trades if trades else 0.0
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    if gross_loss <= 0:
        profit_factor = float("inf") if gross_win > 0 else 0.0
    else:
        profit_factor = gross_win / gross_loss
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "profit_factor": profit_factor,
    }


def _summarize_from_stats(
    trades: int,
    wins: int,
    sum_pnl: float,
    gross_win: float,
    gross_loss: float,
) -> dict:
    if trades <= 0:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
        }
    win_rate = wins / trades if trades else 0.0
    avg_pnl = sum_pnl / trades if trades else 0.0
    if gross_loss <= 0:
        profit_factor = float("inf") if gross_win > 0 else 0.0
    else:
        profit_factor = gross_win / gross_loss
    return {
        "trades": trades,
        "wins": wins,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "profit_factor": profit_factor,
    }


def _score_candidate(summary: dict, min_trades: int) -> float:
    trades = summary.get("trades", 0)
    win_rate = summary.get("win_rate", 0.0)
    avg_pnl = summary.get("avg_pnl", 0.0)
    profit_factor = summary.get("profit_factor", 0.0)
    trade_weight = min(1.0, math.sqrt(trades / max(1, min_trades)))
    pf_adj = 0.0
    if math.isfinite(profit_factor):
        pf_adj = min(profit_factor, 3.0) / 3.0
    score = (avg_pnl * 0.7) + ((win_rate - 0.5) * 5.0) + (pf_adj * 0.3)
    return score * trade_weight


def _build_trades_legacy(
    df_tf: pd.DataFrame,
    df_1m_index: np.ndarray,
    thresholds: list[float],
    label: str = "",
) -> dict:
    trades = defaultdict(list)
    if df_tf.empty:
        return trades
    times = df_tf.index
    open_arr = df_tf["open"].values
    close_arr = df_tf["close"].values
    min_thresh = min(thresholds)
    thresholds_sorted = sorted(thresholds)

    for i in range(1, len(df_tf)):
        prev_open = float(open_arr[i - 1])
        prev_close = float(close_arr[i - 1])
        body = prev_close - prev_open
        abs_body = abs(body)
        if abs_body <= min_thresh:
            continue
        is_green = body > 0
        is_red = body < 0
        if not (is_green or is_red):
            continue
        curr_time = times[i]
        session = _session_bucket(curr_time)
        if not session:
            continue
        entry_time_ns = int(curr_time.value)
        entry_pos_1m = int(np.searchsorted(df_1m_index, entry_time_ns, side="left"))
        if entry_pos_1m >= len(df_1m_index):
            continue
        entry_price = float(open_arr[i])

        for thresh in thresholds_sorted:
            if abs_body <= thresh:
                break
            if is_red:
                trades[(session, "Long_Rev", thresh)].append(
                    (entry_pos_1m, i, entry_price, entry_time_ns)
                )
                trades[(session, "Short_Mom", thresh)].append(
                    (entry_pos_1m, i, entry_price, entry_time_ns)
                )
            else:
                trades[(session, "Short_Rev", thresh)].append(
                    (entry_pos_1m, i, entry_price, entry_time_ns)
                )
                trades[(session, "Long_Mom", thresh)].append(
                    (entry_pos_1m, i, entry_price, entry_time_ns)
                )
    return trades


def _build_trades(
    df_tf: pd.DataFrame,
    df_1m_index: np.ndarray,
    thresholds: list[float],
    label: str = "",
) -> dict:
    trades = defaultdict(list)
    if df_tf.empty:
        return trades

    t0 = time.time()
    times = df_tf.index
    times_ny = times.tz_convert(NY_TZ) if times.tz is not None else times.tz_localize(NY_TZ)
    times_ns = times.values.astype("datetime64[ns]").astype("int64")
    entry_pos_1m_all = np.searchsorted(df_1m_index, times_ns, side="left")

    open_arr = df_tf["open"].to_numpy()
    close_arr = df_tf["close"].to_numpy()
    min_thresh = min(thresholds)
    thresholds_sorted = np.array(sorted(thresholds), dtype=float)
    threshold_prefix = [tuple() for _ in range(len(thresholds_sorted) + 1)]
    running: list[float] = []
    for k, v in enumerate(thresholds_sorted, start=1):
        running.append(float(v))
        threshold_prefix[k] = tuple(running)

    hours = times_ny.hour
    session_idx = (hours // 3).astype(int)
    session_labels = np.array(
        ["00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"],
        dtype=object,
    )
    session_bucket = session_labels[session_idx]

    body = close_arr[:-1] - open_arr[:-1]
    abs_body = np.abs(body)
    valid_mask = abs_body > min_thresh
    if not np.any(valid_mask):
        logging.info("%s trade list built: 0 candidates (%.2fs)", label, time.time() - t0)
        return trades

    idxs = np.nonzero(valid_mask)[0]
    n = len(df_tf)
    total_candidates = len(idxs)
    log_every = 200_000
    last_log = t0
    for count, prev_i in enumerate(idxs, start=1):
        i = prev_i + 1
        abs_body_i = abs_body[prev_i]
        body_i = body[prev_i]
        if body_i == 0:
            continue

        entry_pos_1m = int(entry_pos_1m_all[i])
        if entry_pos_1m >= len(df_1m_index):
            continue
        entry_time_ns = int(times_ns[i])
        entry_price = float(open_arr[i])
        session = session_bucket[i]
        entry_tuple = (entry_pos_1m, i, entry_price, entry_time_ns)

        thresh_count = int(np.searchsorted(thresholds_sorted, abs_body_i, side="left"))
        if thresh_count <= 0:
            continue
        active_thresholds = threshold_prefix[thresh_count]
        if body_i < 0:
            for thresh in active_thresholds:
                trades[(session, "Long_Rev", thresh)].append(entry_tuple)
                trades[(session, "Short_Mom", thresh)].append(entry_tuple)
        else:
            for thresh in active_thresholds:
                trades[(session, "Short_Rev", thresh)].append(entry_tuple)
                trades[(session, "Long_Mom", thresh)].append(entry_tuple)

        now = time.time()
        if (count % log_every) == 0 or (now - last_log) >= 30:
            elapsed = max(1e-9, now - t0)
            rate = float(count) / elapsed
            eta = ((total_candidates - count) / rate) if rate > 0 else 0.0
            logging.info(
                "%s trade list: processed %s/%s candidates | elapsed=%s | ETA=%s | %.0f cand/sec",
                label,
                count,
                total_candidates,
                _format_duration(elapsed),
                _format_duration(eta),
                rate,
            )
            last_log = now

    elapsed = time.time() - t0
    rows_per_sec = n / elapsed if elapsed > 0 else 0.0
    logging.info(
        "%s trade list built: %s bars in %.2fs (%.0f rows/sec)",
        label,
        n,
        elapsed,
        rows_per_sec,
    )
    return trades


def _slice_last_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(NY_TZ)
    else:
        idx = idx.tz_convert(NY_TZ)
    unique_days = pd.Index(idx.date).drop_duplicates()
    if len(unique_days) <= days:
        return df
    keep_days = set(unique_days[-days:])
    mask = pd.Index(idx.date).isin(keep_days)
    return df.loc[mask]


def _compare_trade_dicts(a: dict, b: dict, label: str) -> None:
    if set(a.keys()) != set(b.keys()):
        missing_a = set(b.keys()) - set(a.keys())
        missing_b = set(a.keys()) - set(b.keys())
        raise ValueError(
            f"{label}: key mismatch. missing in new={sorted(missing_a)} missing in old={sorted(missing_b)}"
        )
    for key in sorted(a.keys()):
        list_a = a[key]
        list_b = b[key]
        if len(list_a) != len(list_b):
            raise ValueError(
                f"{label}: trade count mismatch for {key}: new={len(list_a)} old={len(list_b)}"
            )
        for idx, (ta, tb) in enumerate(zip(list_a, list_b)):
            if ta != tb:
                raise ValueError(
                    f"{label}: first mismatch at {key} index {idx}: new={ta} old={tb}"
                )


def _prepare_trade_metadata(
    trades: dict,
    df_1m_index: np.ndarray,
    df_tf_index: np.ndarray,
    max_horizon_1m: int,
    max_horizon_tf: int,
    limit_to_session: bool,
    label: str = "",
) -> dict:
    prepared = {}
    total_items = sum(len(items) for items in trades.values())
    processed = 0
    t0 = time.time()
    last_log = t0
    log_every = 200_000
    for key, items in trades.items():
        prepared_items = []
        for entry_pos_1m, entry_pos_tf, entry_price, entry_time_ns in items:
            entry_time = pd.Timestamp(entry_time_ns, tz="UTC").tz_convert(NY_TZ)
            if limit_to_session:
                end_time = _next_session_boundary(entry_time)
                end_time_ns = int(end_time.value)
                session_end_1m = int(np.searchsorted(df_1m_index, end_time_ns, side="left") - 1)
                session_end_tf = int(np.searchsorted(df_tf_index, end_time_ns, side="left") - 1)
                end_pos_1m = min(entry_pos_1m + max_horizon_1m, session_end_1m)
                end_pos_tf = min(entry_pos_tf + max_horizon_tf, session_end_tf)
            else:
                end_pos_1m = entry_pos_1m + max_horizon_1m
                end_pos_tf = entry_pos_tf + max_horizon_tf
            end_pos_1m = max(entry_pos_1m, end_pos_1m)
            end_pos_tf = max(entry_pos_tf, end_pos_tf)
            prepared_items.append(
                (
                    entry_pos_1m,
                    entry_pos_tf,
                    entry_price,
                    entry_time_ns,
                    end_pos_1m,
                    end_pos_tf,
                )
            )
            processed += 1
            if processed % log_every == 0 or (time.time() - last_log) >= 30:
                elapsed = time.time() - t0
                logging.info(
                    "Trade metadata %s: %s/%s (%.1fs)",
                    label or "build",
                    processed,
                    total_items,
                    elapsed,
                )
                last_log = time.time()
        prepared[key] = prepared_items
    elapsed = time.time() - t0
    if total_items:
        logging.info(
            "Trade metadata %s complete: %s items in %.2fs",
            label or "build",
            total_items,
            elapsed,
        )
    return prepared


def _evaluate_bucket(
    tf_label: str,
    session: str,
    stype: str,
    thresh: float,
    items: list,
    ctx: dict,
) -> list[dict]:
    results: list[dict] = []
    if not items:
        return results
    t0 = time.time()
    logging.info(
        "Eval start: %s %s %s thresh=%s trades=%s",
        tf_label,
        session,
        stype,
        thresh,
        len(items),
    )

    trade_resolution = ctx["trade_resolution"]
    if trade_resolution == "1m":
        high = ctx["high_1m"]
        low = ctx["low_1m"]
        close = ctx["close_1m"]
    else:
        if tf_label == "5min":
            high = ctx["high_5m"]
            low = ctx["low_5m"]
            close = ctx["close_5m"]
        else:
            high = ctx["high_15m"]
            low = ctx["low_15m"]
            close = ctx["close_15m"]

    tp_list = ctx["tp_list"]
    fold_edges = ctx["fold_edges"]
    folds = ctx["folds"]
    recent_start_ns = ctx["recent_start_ns"]
    recent_end_ns = ctx["recent_end_ns"]
    recent_active = ctx["recent_active"]
    min_trades = ctx["min_trades"]
    min_win_rate = ctx["min_win_rate"]
    min_avg_pnl = ctx["min_avg_pnl"]
    min_fold_trades = ctx["min_fold_trades"]
    min_fold_win_rate = ctx["min_fold_win_rate"]
    min_fold_avg_pnl = ctx["min_fold_avg_pnl"]
    min_positive_folds = ctx["min_positive_folds"]
    min_positive_ratio = ctx["min_positive_ratio"]
    assume_sl_first = ctx["assume_sl_first"]
    exit_at_horizon = ctx["exit_at_horizon"]
    recent_mode = ctx["recent_mode"]
    no_recent = ctx["no_recent"]
    loro_enabled = ctx["loro_enabled"]
    min_regime_trades = ctx["min_regime_trades"]
    min_regime_avg_pnl = ctx["min_regime_avg_pnl"]
    min_positive_regimes = ctx["min_positive_regimes"]
    regime_arr = ctx.get("regime_arr")

    side = "LONG" if stype.startswith("Long") else "SHORT"

    combo_count = len(tp_list)
    if combo_count == 0:
        return results

    sl_arr = np.array([sl for sl, _ in tp_list], dtype=float)
    tp_arr = np.array([tp for _, tp in tp_list], dtype=float)

    num_trades = len(items)
    entry_pos_1m = np.fromiter((t[0] for t in items), dtype=np.int64, count=num_trades)
    entry_pos_tf = np.fromiter((t[1] for t in items), dtype=np.int64, count=num_trades)
    entry_price_arr = np.fromiter((t[2] for t in items), dtype=float, count=num_trades)
    entry_time_ns = np.fromiter((t[3] for t in items), dtype=np.int64, count=num_trades)
    end_pos_1m = np.fromiter((t[4] for t in items), dtype=np.int64, count=num_trades)
    end_pos_tf = np.fromiter((t[5] for t in items), dtype=np.int64, count=num_trades)

    if trade_resolution == "1m":
        entry_pos_arr = entry_pos_1m
        end_pos_arr = end_pos_1m
    else:
        entry_pos_arr = entry_pos_tf
        end_pos_arr = end_pos_tf

    max_pos = len(high) - 1
    entry_pos_arr = np.minimum(entry_pos_arr, max_pos)
    end_pos_arr = np.minimum(end_pos_arr, max_pos)
    end_pos_arr = np.maximum(end_pos_arr, entry_pos_arr)

    fold_idx_arr = np.searchsorted(fold_edges, entry_time_ns, side="right") - 1
    if recent_active:
        recent_mask = np.ones(num_trades, dtype=bool)
        if recent_start_ns is not None:
            recent_mask &= entry_time_ns >= recent_start_ns
        if recent_end_ns is not None:
            recent_mask &= entry_time_ns <= recent_end_ns
    else:
        recent_mask = None

    sum_pnl = np.zeros(combo_count, dtype=float)
    wins = np.zeros(combo_count, dtype=np.int64)
    gross_win = np.zeros(combo_count, dtype=float)
    gross_loss = np.zeros(combo_count, dtype=float)

    fold_trades = np.zeros((combo_count, folds), dtype=np.int32)
    fold_wins = np.zeros((combo_count, folds), dtype=np.int32)
    fold_pnl = np.zeros((combo_count, folds), dtype=float)

    if recent_active:
        recent_trades = np.zeros(combo_count, dtype=np.int32)
        recent_wins = np.zeros(combo_count, dtype=np.int32)
        recent_sum_pnl = np.zeros(combo_count, dtype=float)
        recent_gross_win = np.zeros(combo_count, dtype=float)
        recent_gross_loss = np.zeros(combo_count, dtype=float)
    else:
        recent_trades = None
        recent_wins = None
        recent_sum_pnl = None
        recent_gross_win = None
        recent_gross_loss = None

    regime_trades = {}
    regime_pnl = {}

    for idx in range(num_trades):
        entry_pos = int(entry_pos_arr[idx])
        end_pos = int(end_pos_arr[idx])
        if entry_pos < 0 or entry_pos > max_pos:
            continue
        if end_pos < entry_pos:
            end_pos = entry_pos

        high_slice = high[entry_pos : end_pos + 1]
        low_slice = low[entry_pos : end_pos + 1]
        if high_slice.size == 0:
            continue
        cummax = np.maximum.accumulate(high_slice)
        cummin = np.minimum.accumulate(low_slice)
        window_len = len(cummax)

        entry_price = float(entry_price_arr[idx])
        exit_price = float(close[end_pos])
        if exit_at_horizon == "close":
            pnl_no_hit = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        else:
            pnl_no_hit = 0.0

        if side == "LONG":
            tp_levels = entry_price + tp_arr
            sl_levels = entry_price - sl_arr
            tp_idx = np.searchsorted(cummax, tp_levels, side="left")
            sl_idx = np.searchsorted(-cummin, -sl_levels, side="left")
        else:
            tp_levels = entry_price - tp_arr
            sl_levels = entry_price + sl_arr
            tp_idx = np.searchsorted(-cummin, -tp_levels, side="left")
            sl_idx = np.searchsorted(cummax, sl_levels, side="left")

        hit_tp = tp_idx < window_len
        hit_sl = sl_idx < window_len
        pnl = np.full(combo_count, pnl_no_hit, dtype=float)

        tp_before = hit_tp & (~hit_sl | (tp_idx < sl_idx))
        sl_before = hit_sl & (~hit_tp | (sl_idx < tp_idx))
        pnl[tp_before] = tp_arr[tp_before]
        pnl[sl_before] = -sl_arr[sl_before]

        equal = hit_tp & hit_sl & (tp_idx == sl_idx)
        if np.any(equal):
            # Tie-break rule matches _simulate_trade_points_np behavior.
            pnl[equal] = -sl_arr[equal] if assume_sl_first else tp_arr[equal]

        sum_pnl += pnl
        pos_mask = pnl > 0
        neg_mask = pnl < 0
        wins += pos_mask
        if np.any(pos_mask):
            gross_win += np.where(pos_mask, pnl, 0.0)
        if np.any(neg_mask):
            gross_loss += np.where(neg_mask, -pnl, 0.0)

        fold_idx = int(fold_idx_arr[idx])
        if 0 <= fold_idx < folds:
            fold_trades[:, fold_idx] += 1
            fold_pnl[:, fold_idx] += pnl
            fold_wins[:, fold_idx] += pos_mask

        if recent_active and recent_mask is not None and recent_mask[idx]:
            recent_trades += 1
            recent_sum_pnl += pnl
            recent_wins += pos_mask
            if np.any(pos_mask):
                recent_gross_win += np.where(pos_mask, pnl, 0.0)
            if np.any(neg_mask):
                recent_gross_loss += np.where(neg_mask, -pnl, 0.0)

        if loro_enabled and regime_arr is not None:
            entry_pos_1m_val = int(entry_pos_1m[idx])
            if 0 <= entry_pos_1m_val < len(regime_arr):
                regime_val = regime_arr[entry_pos_1m_val]
            else:
                regime_val = None
            if regime_val:
                regime_key = str(regime_val)
                if regime_key not in regime_trades:
                    regime_trades[regime_key] = np.zeros(combo_count, dtype=np.int32)
                    regime_pnl[regime_key] = np.zeros(combo_count, dtype=float)
                regime_trades[regime_key] += 1
                regime_pnl[regime_key] += pnl

    if num_trades <= 0:
        return results

    for j in range(combo_count):
        summary = _summarize_from_stats(
            num_trades,
            int(wins[j]),
            float(sum_pnl[j]),
            float(gross_win[j]),
            float(gross_loss[j]),
        )

        if recent_active and recent_trades is not None and recent_trades[j] > 0:
            recent_summary = _summarize_from_stats(
                int(recent_trades[j]),
                int(recent_wins[j]),
                float(recent_sum_pnl[j]),
                float(recent_gross_win[j]),
                float(recent_gross_loss[j]),
            )
        else:
            recent_summary = None

        full_ok = (
            summary["trades"] >= min_trades
            and summary["win_rate"] >= min_win_rate
            and summary["avg_pnl"] >= min_avg_pnl
        )
        recent_ok = True
        if recent_active:
            if recent_summary is None:
                recent_ok = False
            else:
                recent_ok = (
                    recent_summary["trades"] >= max(5, int(min_trades / 2))
                    and recent_summary["win_rate"] >= min_win_rate
                    and recent_summary["avg_pnl"] >= min_avg_pnl
                )

        if no_recent:
            keep_candidate = full_ok
        elif recent_mode == "recent_only":
            keep_candidate = bool(recent_ok)
        elif recent_mode == "union":
            keep_candidate = bool(full_ok or recent_ok)
        else:
            keep_candidate = bool(full_ok and recent_ok)

        if not keep_candidate:
            continue

        require_folds = True
        if not no_recent and recent_mode == "recent_only":
            require_folds = False
        elif not no_recent and recent_mode == "union" and not full_ok and recent_ok:
            require_folds = False

        if require_folds:
            positive_folds = 0
            for f_idx in range(folds):
                trades_fold = int(fold_trades[j, f_idx])
                if trades_fold < min_fold_trades:
                    continue
                win_rate_fold = fold_wins[j, f_idx] / trades_fold if trades_fold else 0.0
                avg_pnl_fold = fold_pnl[j, f_idx] / trades_fold if trades_fold else 0.0
                if win_rate_fold < min_fold_win_rate:
                    continue
                if avg_pnl_fold < min_fold_avg_pnl:
                    continue
                positive_folds += 1

            positive_ratio = positive_folds / folds if folds else 0.0
            if positive_folds < min_positive_folds:
                continue
            if positive_ratio < min_positive_ratio:
                continue

            if loro_enabled:
                positive_regimes = 0
                regimes_with_trades = 0
                for regime_key, trades_arr in regime_trades.items():
                    reg_trades = int(trades_arr[j])
                    if reg_trades < min_regime_trades:
                        continue
                    regimes_with_trades += 1
                    reg_avg_pnl = regime_pnl[regime_key][j] / reg_trades if reg_trades else 0.0
                    if reg_avg_pnl >= min_regime_avg_pnl:
                        positive_regimes += 1
                if regimes_with_trades > 0 and positive_regimes < min_positive_regimes:
                    continue

        score_base = summary
        if recent_mode == "recent_only" and recent_summary is not None:
            score_base = recent_summary
        elif recent_mode == "union" and not full_ok and recent_summary is not None:
            score_base = recent_summary
        score = _score_candidate(score_base, min_trades)

        results.append(
            {
                "TF": tf_label,
                "Session": session,
                "Type": stype,
                "Thresh": float(thresh),
                "Best_SL": float(sl_arr[j]),
                "Best_TP": float(tp_arr[j]),
                "Opt_WR": float(summary["win_rate"]),
                "Trades": int(summary["trades"]),
                "Avg_PnL": float(summary["avg_pnl"]),
                "Score": float(score),
                "Recent": recent_summary,
            }
        )

    elapsed = time.time() - t0
    logging.info(
        "Eval done: %s %s %s thresh=%s combos=%s in %.2fs",
        tf_label,
        session,
        stype,
        thresh,
        combo_count,
        elapsed,
    )
    return results


_WORKER_CTX: dict = {}


def _init_worker(ctx: dict) -> None:
    global _WORKER_CTX
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    _WORKER_CTX = ctx


def _unpack_task(task: tuple) -> tuple:
    if len(task) == 6:
        tf_label, session, stype, thresh, items, last_ns = task
    else:
        tf_label, session, stype, thresh, items = task
        last_ns = None
    return tf_label, session, stype, thresh, items, last_ns


def _eval_bucket_worker(task: tuple) -> list[dict]:
    tf_label, session, stype, thresh, items, _last_ns = _unpack_task(task)
    return _evaluate_bucket(tf_label, session, stype, thresh, items, _WORKER_CTX)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dynamic Engine 3 strategy database.")
    parser.add_argument("--csv", default="ml_mes_et.csv", help="Path to CSV history file.")
    parser.add_argument("--out", default="dynamic_engine3_strategies.json", help="Output JSON path.")
    parser.add_argument("--start", "--train-start", dest="start", default=None, help="Train start date (YYYY-MM-DD).")
    parser.add_argument("--end", "--train-end", dest="end", default=None, help="Train end date (YYYY-MM-DD).")
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
    parser.add_argument("--recent-end", default=None, help="Recent window end date.")
    parser.add_argument(
        "--recent-mode",
        default="intersect",
        choices=("intersect", "union", "recent_only"),
        help="How to combine full vs recent strategy sets.",
    )
    parser.add_argument("--no-recent", action="store_true", help="Disable recency window.")
    parser.add_argument("--thresholds", default="2,3,4,5,6,7,8,9,10,12,15")
    parser.add_argument("--sl-list", default="3,4,5,6,8,10,12,15")
    parser.add_argument("--rr-list", default="1.0,1.25,1.5,2.0,2.5,3.0")
    parser.add_argument("--min-tp", type=float, default=4.0)
    parser.add_argument("--max-tp", type=float, default=30.0)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--min-win-rate", type=float, default=0.45)
    parser.add_argument("--min-avg-pnl", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-fold-trades", type=int, default=10)
    parser.add_argument("--min-fold-win-rate", type=float, default=0.45)
    parser.add_argument("--min-fold-avg-pnl", type=float, default=0.0)
    parser.add_argument("--min-positive-folds", type=int, default=2)
    parser.add_argument("--min-positive-ratio", type=float, default=0.5)
    parser.add_argument("--loro", action="store_true", help="Enable volatility regime robustness check.")
    parser.add_argument("--min-regime-trades", type=int, default=10)
    parser.add_argument("--min-regime-avg-pnl", type=float, default=0.0)
    parser.add_argument("--min-positive-regimes", type=int, default=2)
    parser.add_argument("--max-per-bucket", type=int, default=6)
    parser.add_argument("--max-horizon", type=int, default=180, help="Max holding time in minutes.")
    parser.add_argument("--no-session-limit", action="store_true", help="Allow holding past session end.")
    parser.add_argument("--exit-at-horizon", default="close", choices=("close", "flat"))
    parser.add_argument("--assume-sl-first", action="store_true", help="If SL and TP hit in same bar, pick SL.")
    parser.add_argument("--assume-tp-first", action="store_true", help="If SL and TP hit in same bar, pick TP.")
    parser.add_argument(
        "--trade-resolution",
        default="1m",
        choices=("1m", "tf"),
        help="Resolution used for SL/TP simulation.",
    )
    parser.add_argument("--symbol-mode", default="auto_by_day", choices=("single", "auto", "auto_by_day", "roll"))
    parser.add_argument("--symbol-method", default="volume", choices=("volume", "rows"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for evaluation.")
    parser.add_argument("--chunk-size", type=int, default=8, help="Tasks per worker chunk.")
    parser.add_argument("--cache-dir", default="cache", help="Cache directory for parquet/features.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache read/write.")
    parser.add_argument("--cache-only", action="store_true", help="Require cached parquet and skip CSV.")
    parser.add_argument(
        "--verify-identical",
        action="store_true",
        help="Verify optimized trade builder matches legacy on last 30 days.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    exp_enabled = bool(args.experimental_window)
    train_start_raw = args.start
    train_end_raw = args.end
    if exp_enabled:
        exp_start, exp_end = get_experimental_training_window()
        train_start_raw = exp_start
        train_end_raw = exp_end
        logging.info("Experimental window enabled: %s -> %s", train_start_raw, train_end_raw)
    if not train_start_raw or not train_end_raw:
        logging.warning(
            "Training window is not fully bounded (start=%s end=%s). "
            "For strict OOS workflows, pass both --train-start and --train-end.",
            train_start_raw,
            train_end_raw,
        )
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
    logging.info("Stage: load bars")
    if args.cache_only:
        source_is_cached_format = csv_path.suffix.lower() in (".parquet", ".pq", ".feather", ".ft")
        if source_is_cached_format:
            logging.info("Cache-only with cached source format; loading input directly: %s", csv_path)
            df = data_cache.load_bars(csv_path, cache_dir=None, use_cache=False)
        else:
            if not cache_dir:
                raise SystemExit("--cache-only requires --cache-dir.")
            source_key = data_cache.cache_key_for_source(csv_path)
            cache_path = cache_dir / f"{source_key}.parquet"
            if not cache_path.exists():
                raise FileNotFoundError(f"Cache parquet not found: {cache_path}")
            df = data_cache.load_bars(cache_path, cache_dir=None, use_cache=False)
    else:
        df = data_cache.load_bars(csv_path, cache_dir=cache_dir, use_cache=not args.no_cache)
    logging.info("Loaded bars: %s rows", len(df))
    start = _parse_date(train_start_raw, is_end=False)
    end = _parse_date(train_end_raw, is_end=True)
    df = _filter_range(df, start, end)
    logging.info("Filtered range: %s -> %s (%s rows)", df.index.min(), df.index.max(), len(df))

    symbol_method = args.symbol_method
    logging.info("Stage: apply symbol mode")
    logging.info("Applying symbol mode: %s (%s)", args.symbol_mode, symbol_method)
    df, symbol_mode, symbol_map = bt.apply_symbol_mode(df, args.symbol_mode, symbol_method)
    logging.info("Symbol mode applied: %s (map size: %s)", symbol_mode, len(symbol_map) if symbol_map else 0)
    if df.empty:
        raise SystemExit("No data after filtering.")

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    sl_list = [float(x.strip()) for x in args.sl_list.split(",") if x.strip()]
    rr_list = [float(x.strip()) for x in args.rr_list.split(",") if x.strip()]

    tp_list = []
    for sl in sl_list:
        for rr in rr_list:
            tp = _round_to_tick(sl * rr, TICK_SIZE)
            if tp < args.min_tp or tp > args.max_tp:
                continue
            tp_list.append((sl, tp))
    tp_list = sorted(set(tp_list))

    logging.info("Loaded %s rows. Range: %s -> %s", len(df), df.index.min(), df.index.max())
    logging.info("Thresholds: %s", thresholds)
    logging.info("SL/TP combos: %s", len(tp_list))
    logging.info("Symbol mode: %s (%s)", symbol_mode, len(symbol_map) if symbol_map else "single")

    source_key = data_cache.cache_key_for_source(csv_path)
    feature_key = _feature_cache_key(source_key, df, symbol_mode, symbol_method)
    logging.info("Stage: feature cache + resample")
    logging.info("Preparing feature cache...")
    feature_cache = _load_feature_cache(cache_dir, feature_key, args.folds, not args.no_cache) if cache_dir else {}

    df_5m = feature_cache.get("df_5m")
    df_15m = feature_cache.get("df_15m")
    if df_5m is None:
        logging.info("Resampling 5m...")
        t_resample = time.time()
        df_5m = _resample_df(df, 5)
        elapsed = time.time() - t_resample
        rows_per_sec = len(df) / elapsed if elapsed > 0 else 0.0
        logging.info("Resample 5m done in %.2fs (%.0f rows/sec)", elapsed, rows_per_sec)
    if df_15m is None:
        logging.info("Resampling 15m...")
        t_resample = time.time()
        df_15m = _resample_df(df, 15)
        elapsed = time.time() - t_resample
        rows_per_sec = len(df) / elapsed if elapsed > 0 else 0.0
        logging.info("Resample 15m done in %.2fs (%.0f rows/sec)", elapsed, rows_per_sec)
    if df_5m.empty or df_15m.empty:
        raise SystemExit("Resampled data is empty.")

    index_1m = df.index.values.astype("datetime64[ns]")
    regimes = None
    regime_arr = feature_cache.get("regime_arr")
    if args.loro and regime_arr is None:
        logging.info("Computing regimes for LORO...")
        regimes = _compute_regimes(df)
        if not regimes.empty:
            regime_arr = regimes.reindex(df.index).fillna("").values

    if args.verify_identical:
        logging.info("Verify mode: comparing optimized vs legacy trade builder on last 30 days...")
        verify_df = _slice_last_days(df, 30)
        verify_5m = _resample_df(verify_df, 5)
        verify_15m = _resample_df(verify_df, 15)
        verify_index_1m = verify_df.index.values.astype("datetime64[ns]")
        new_5m = _build_trades(verify_5m, verify_index_1m, thresholds, label="verify-5m")
        old_5m = _build_trades_legacy(verify_5m, verify_index_1m, thresholds, label="verify-5m")
        _compare_trade_dicts(new_5m, old_5m, "verify-5m")
        new_15m = _build_trades(verify_15m, verify_index_1m, thresholds, label="verify-15m")
        old_15m = _build_trades_legacy(verify_15m, verify_index_1m, thresholds, label="verify-15m")
        _compare_trade_dicts(new_15m, old_15m, "verify-15m")
        logging.info("Verify mode: trade builder outputs match legacy.")

    logging.info("Stage: build trade lists")
    logging.info("Building trade lists (5m/15m)...")
    t_build = time.time()
    trades_5m = _build_trades(df_5m, index_1m, thresholds, label="5m")
    trades_15m = _build_trades(df_15m, index_1m, thresholds, label="15m")
    logging.info("Trade list build total: %.2fs", time.time() - t_build)

    max_horizon_1m = max(1, int(args.max_horizon))
    max_horizon_5m = max(1, int(math.ceil(args.max_horizon / 5)))
    max_horizon_15m = max(1, int(math.ceil(args.max_horizon / 15)))

    logging.info("Preparing trade metadata (5m)...")
    prepared_5m = _prepare_trade_metadata(
        trades_5m,
        index_1m,
        df_5m.index.values.astype("datetime64[ns]"),
        max_horizon_1m,
        max_horizon_5m,
        not args.no_session_limit,
        label="5m",
    )
    logging.info("Preparing trade metadata (15m)...")
    prepared_15m = _prepare_trade_metadata(
        trades_15m,
        index_1m,
        df_15m.index.values.astype("datetime64[ns]"),
        max_horizon_1m,
        max_horizon_15m,
        not args.no_session_limit,
        label="15m",
    )

    df_index_int = df.index.values.astype("datetime64[ns]").astype("int64")
    fold_edges = feature_cache.get("fold_edges")
    if fold_edges is None:
        logging.info("Computing fold edges...")
        fold_edges = np.linspace(
            int(df_index_int[0]),
            int(df_index_int[-1]),
            args.folds + 1,
        )

    recent_start = _parse_date(args.recent_start, is_end=False) if not args.no_recent else None
    recent_end = _parse_date(args.recent_end, is_end=True) if (args.recent_end and not args.no_recent) else None
    if recent_end is None and not args.no_recent:
        recent_end = df.index.max()
    recent_start_ns = int(recent_start.value) if recent_start is not None else None
    recent_end_ns = int(recent_end.value) if recent_end is not None else None

    if args.assume_sl_first and args.assume_tp_first:
        raise SystemExit("Choose only one of --assume-sl-first or --assume-tp-first.")
    assume_sl_first = not args.assume_tp_first

    ctx = {
        "trade_resolution": args.trade_resolution,
        "tp_list": tp_list,
        "fold_edges": fold_edges,
        "folds": args.folds,
        "recent_start_ns": recent_start_ns,
        "recent_end_ns": recent_end_ns,
        "recent_active": not args.no_recent and (recent_start_ns is not None or recent_end_ns is not None),
        "recent_mode": args.recent_mode,
        "no_recent": args.no_recent,
        "min_trades": args.min_trades,
        "min_win_rate": args.min_win_rate,
        "min_avg_pnl": args.min_avg_pnl,
        "min_fold_trades": args.min_fold_trades,
        "min_fold_win_rate": args.min_fold_win_rate,
        "min_fold_avg_pnl": args.min_fold_avg_pnl,
        "min_positive_folds": args.min_positive_folds,
        "min_positive_ratio": args.min_positive_ratio,
        "assume_sl_first": assume_sl_first,
        "exit_at_horizon": args.exit_at_horizon,
        "loro_enabled": bool(args.loro),
        "min_regime_trades": args.min_regime_trades,
        "min_regime_avg_pnl": args.min_regime_avg_pnl,
        "min_positive_regimes": args.min_positive_regimes,
        "regime_arr": regime_arr,
        "high_1m": df["high"].values,
        "low_1m": df["low"].values,
        "close_1m": df["close"].values,
        "high_5m": df_5m["high"].values,
        "low_5m": df_5m["low"].values,
        "close_5m": df_5m["close"].values,
        "high_15m": df_15m["high"].values,
        "low_15m": df_15m["low"].values,
        "close_15m": df_15m["close"].values,
    }

    if cache_dir and not args.no_cache:
        logging.info("Saving feature cache...")
        feature_cache["df_5m"] = df_5m
        feature_cache["df_15m"] = df_15m
        feature_cache["regime_arr"] = regime_arr
        feature_cache["fold_edges"] = fold_edges
        _save_feature_cache(feature_cache, use_cache=True)

    tasks: list[tuple] = []
    for (session, stype, thresh), items in prepared_5m.items():
        if items:
            last_ns = items[-1][3] if items else None
            tasks.append(("5min", session, stype, thresh, items, last_ns))
    for (session, stype, thresh), items in prepared_15m.items():
        if items:
            last_ns = items[-1][3] if items else None
            tasks.append(("15min", session, stype, thresh, items, last_ns))
    total_task_trades = sum(len(task[4]) for task in tasks) if tasks else 0
    logging.info("Tasks built: %s (total trades: %s)", len(tasks), total_task_trades)

    ny_index = df.index
    if ny_index.tz is None:
        ny_index = ny_index.tz_localize(NY_TZ)
    else:
        ny_index = ny_index.tz_convert(NY_TZ)
    unique_days = pd.Index(ny_index.date).drop_duplicates()
    total_days = int(len(unique_days))
    day_to_idx = {day: idx + 1 for idx, day in enumerate(unique_days)}

    results: list[dict] = []
    total_tasks = len(tasks)
    progress_state = {"next_pct": 1, "done": 0, "last_day_idx": 0}

    def log_progress(force: bool = False, last_ns: Optional[int] = None) -> None:
        if total_tasks <= 0:
            return
        done = progress_state["done"]
        pct = (done / total_tasks) * 100 if total_tasks else 0.0
        if force or pct >= progress_state["next_pct"]:
            logging.info("Progress: %s/%s (%.2f%%)", done, total_tasks, pct)
            while progress_state["next_pct"] <= pct:
                progress_state["next_pct"] += 1
        if last_ns and total_days:
            try:
                day_val = pd.Timestamp(int(last_ns), tz="UTC").tz_convert(NY_TZ).date()
            except Exception:
                day_val = None
            if day_val is not None:
                day_idx = day_to_idx.get(day_val)
                if day_idx and day_idx > progress_state["last_day_idx"]:
                    progress_state["last_day_idx"] = day_idx
                    day_pct = (day_idx / total_days) * 100 if total_days else 0.0
                    logging.info(
                        "Day progress: %s/%s (%.2f%%) | up to %s",
                        day_idx,
                        total_days,
                        day_pct,
                        day_val.isoformat(),
                    )

    logging.info("Stage: evaluate strategies")
    t_eval = time.time()
    workers = max(1, int(args.workers or 1))
    if workers > 1 and len(tasks) > 1:
        logging.info("Evaluating with %s workers (tasks: %s)", workers, len(tasks))
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(ctx,),
        ) as executor:
            for idx, chunk in enumerate(
                executor.map(_eval_bucket_worker, tasks, chunksize=max(1, int(args.chunk_size)))
            ):
                if chunk:
                    results.extend(chunk)
                progress_state["done"] += 1
                last_ns = tasks[idx][5] if len(tasks[idx]) > 5 else None
                log_progress(last_ns=last_ns)
    else:
        logging.info("Evaluating tasks in single process (tasks: %s)", len(tasks))
        for task in tasks:
            tf_label, session, stype, thresh, items, last_ns = _unpack_task(task)
            results.extend(_evaluate_bucket(tf_label, session, stype, thresh, items, ctx))
            progress_state["done"] += 1
            log_progress(last_ns=last_ns)

    log_progress(force=True)
    eval_elapsed = time.time() - t_eval
    tasks_per_sec = total_tasks / eval_elapsed if eval_elapsed > 0 else 0.0
    logging.info("Evaluation done in %.2fs (%.2f tasks/sec)", eval_elapsed, tasks_per_sec)

    if not results:
        logging.warning("No strategies met criteria.")

    grouped = defaultdict(list)
    for item in results:
        grouped[(item["TF"], item["Session"], item["Type"])].append(item)

    final_strategies = []
    for key, items in grouped.items():
        items.sort(key=lambda x: x.get("Score", 0.0), reverse=True)
        final_strategies.extend(items[: max(1, args.max_per_bucket)])

    logging.info("Stage: finalize + write output")
    logging.info("Selecting final strategies...")
    out_name = str(args.out)
    if artifact_suffix:
        out_name = append_artifact_suffix(out_name, artifact_suffix)
    out_path = Path(out_name)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": dt.datetime.now(NY_TZ).isoformat(),
        "source_csv": str(csv_path),
        "symbol_mode": symbol_mode,
        "symbol_map_size": len(symbol_map) if symbol_map else 0,
        "date_range": {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat(),
        },
        "settings": {
            "thresholds": thresholds,
            "sl_list": sl_list,
            "rr_list": rr_list,
            "min_tp": args.min_tp,
            "max_tp": args.max_tp,
            "min_trades": args.min_trades,
            "min_win_rate": args.min_win_rate,
            "min_avg_pnl": args.min_avg_pnl,
            "recent_start": args.recent_start if not args.no_recent else None,
            "recent_end": args.recent_end if not args.no_recent else None,
            "recent_mode": None if args.no_recent else args.recent_mode,
            "folds": args.folds,
            "min_fold_trades": args.min_fold_trades,
            "min_fold_win_rate": args.min_fold_win_rate,
            "min_fold_avg_pnl": args.min_fold_avg_pnl,
            "min_positive_folds": args.min_positive_folds,
            "min_positive_ratio": args.min_positive_ratio,
            "loro": bool(args.loro),
            "min_regime_trades": args.min_regime_trades,
            "min_regime_avg_pnl": args.min_regime_avg_pnl,
            "min_positive_regimes": args.min_positive_regimes,
            "max_per_bucket": args.max_per_bucket,
            "max_horizon": args.max_horizon,
            "limit_to_session": not args.no_session_limit,
            "exit_at_horizon": args.exit_at_horizon,
            "assume_sl_first": bool(args.assume_sl_first),
            "trade_resolution": args.trade_resolution,
            "requested_train_window": {
                "start": str(train_start_raw) if train_start_raw else None,
                "end": str(train_end_raw) if train_end_raw else None,
            },
        },
        "strategies": final_strategies,
        "summary": {
            "total_candidates": len(results),
            "total_strategies": len(final_strategies),
        },
    }

    logging.info("Writing output JSON...")
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    logging.info("Wrote DynamicEngine3 strategies: %s", out_path)
    logging.info("Strategies: %s", len(final_strategies))


if __name__ == "__main__":
    main()
