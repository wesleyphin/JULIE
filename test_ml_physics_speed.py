import argparse
import copy
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import CONFIG
import ml_physics_pipeline as mlp
from ml_physics_strategy import MLPhysicsStrategy


NY_TZ = "US/Eastern"
MIN_HISTORY_BARS = 200


def _load_price_df(source: Path, start: str, end: str, warmup_bars: int) -> pd.DataFrame:
    if source.suffix.lower() == ".parquet":
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source)

    if not isinstance(df.index, pd.DatetimeIndex):
        dt_col = None
        for candidate in ("datetime", "timestamp", "date", "time", "ts"):
            if candidate in df.columns:
                dt_col = candidate
                break
        if dt_col is None:
            raise ValueError("Could not find datetime column in source dataset.")
        idx = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.drop(columns=[dt_col])
        df.index = idx

    df = df[~df.index.isna()]
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    df = df.sort_index()

    rename = {}
    for col in df.columns:
        low = str(col).strip().lower()
        if low in {"open", "high", "low", "close", "volume"}:
            rename[col] = low
    df = df.rename(columns=rename)

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Missing OHLC columns. Found={sorted(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(NY_TZ)
    else:
        start_ts = start_ts.tz_convert(NY_TZ)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(NY_TZ)
    else:
        end_ts = end_ts.tz_convert(NY_TZ)

    in_range = df[(df.index >= start_ts) & (df.index <= end_ts)]
    if in_range.empty:
        raise ValueError(f"No rows in requested window {start} -> {end}")
    warmup = df[df.index < start_ts].tail(max(int(warmup_bars), 0))
    out = pd.concat([warmup, in_range])
    return out[["open", "high", "low", "close", "volume"]]


def _signal_tuple(ts: pd.Timestamp, signal: dict) -> Tuple:
    return (
        ts.isoformat(),
        str(signal.get("strategy", "")),
        str(signal.get("side", "")),
        round(float(signal.get("sl_dist", 0.0) or 0.0), 6),
        round(float(signal.get("tp_dist", 0.0) or 0.0), 6),
        round(float(signal.get("ml_confidence", 0.0) or 0.0), 6),
    )


def _run_strategy(df: pd.DataFrame, *, optimized: bool) -> Tuple[List[Tuple], float]:
    opt_cfg = copy.deepcopy(CONFIG.get("ML_PHYSICS_OPT", {}) or {})
    opt_cfg["enabled"] = bool(optimized)
    opt_cfg["mode"] = "backtest"
    CONFIG["ML_PHYSICS_OPT"] = opt_cfg

    strategy = MLPhysicsStrategy()
    if not strategy.model_loaded:
        raise RuntimeError("MLPhysicsStrategy model not loaded; cannot benchmark.")

    dist_mode = bool(getattr(strategy, "_dist_mode", False))
    if optimized and not dist_mode:
        precomputed = mlp.prepare_full_dataset(df, session_manager=getattr(strategy, "sm", None))
        strategy.set_precomputed_backtest_df(precomputed)
    elif optimized and dist_mode:
        ok = bool(strategy.precompute_dist_backtest_signals(df))
        if not ok:
            print("warning: dist mode precompute unavailable; falling back to per-bar dist inference")

    signals: List[Tuple] = []
    t0 = time.perf_counter()
    start_i = max(MIN_HISTORY_BARS, 1)
    for i in range(start_i, len(df)):
        hist_df = df.iloc[: i + 1]
        ts = df.index[i]
        sig = strategy.on_bar(hist_df, ts)
        if sig:
            signals.append(_signal_tuple(ts, sig))
    elapsed = time.perf_counter() - t0
    return signals, elapsed


def _assert_exact_match(baseline: List[Tuple], optimized: List[Tuple]) -> None:
    if baseline == optimized:
        return
    max_len = max(len(baseline), len(optimized))
    mismatch_idx = None
    for i in range(max_len):
        left = baseline[i] if i < len(baseline) else None
        right = optimized[i] if i < len(optimized) else None
        if left != right:
            mismatch_idx = i
            break
    raise AssertionError(
        f"Signal mismatch at index={mismatch_idx}; "
        f"baseline={baseline[mismatch_idx] if mismatch_idx is not None and mismatch_idx < len(baseline) else None}; "
        f"optimized={optimized[mismatch_idx] if mismatch_idx is not None and mismatch_idx < len(optimized) else None}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MLPhysics old/new speed benchmark + signal parity check.")
    parser.add_argument("--source", default="es_master.parquet", help="Path to source data (.parquet or .csv).")
    parser.add_argument("--speed-start", default="2025-01-01", help="Benchmark window start (inclusive).")
    parser.add_argument("--speed-end", default="2025-01-31", help="Benchmark window end (inclusive).")
    parser.add_argument("--reg-start", default="2025-01-06", help="Regression window start (inclusive).")
    parser.add_argument("--reg-end", default="2025-01-10", help="Regression window end (inclusive).")
    parser.add_argument("--warmup-bars", type=int, default=20000, help="Warmup bars prepended before each window.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    cfg_backup = copy.deepcopy(CONFIG)
    try:
        reg_df = _load_price_df(source, args.reg_start, args.reg_end, args.warmup_bars)
        old_signals, old_reg_s = _run_strategy(reg_df, optimized=False)
        new_signals, new_reg_s = _run_strategy(reg_df, optimized=True)
        _assert_exact_match(old_signals, new_signals)
        print(
            "Regression parity: PASS | "
            f"window={args.reg_start}..{args.reg_end} | "
            f"signals={len(old_signals)} | old={old_reg_s:.2f}s | new={new_reg_s:.2f}s"
        )

        speed_df = _load_price_df(source, args.speed_start, args.speed_end, args.warmup_bars)
        _, old_speed_s = _run_strategy(speed_df, optimized=False)
        _, new_speed_s = _run_strategy(speed_df, optimized=True)
        speedup = (old_speed_s / new_speed_s) if new_speed_s > 0 else np.inf
        print(
            "Speed benchmark: "
            f"window={args.speed_start}..{args.speed_end} | "
            f"old={old_speed_s:.2f}s | new={new_speed_s:.2f}s | speedup={speedup:.2f}x"
        )
    finally:
        CONFIG.clear()
        CONFIG.update(cfg_backup)


if __name__ == "__main__":
    main()
