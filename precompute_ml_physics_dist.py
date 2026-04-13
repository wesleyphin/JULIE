import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest_symbol_context import apply_symbol_mode, attach_backtest_symbol_context, choose_symbol
from config import CONFIG
from data_cache import cache_key_for_source
from ml_physics_strategy import MLPhysicsStrategy


NY_TZ = "US/Eastern"


def _parse_ts(raw: str, *, is_end: bool = False) -> pd.Timestamp:
    ts = pd.to_datetime(str(raw).strip(), errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid datetime: {raw}")
    has_time = ("T" in str(raw)) or (":" in str(raw))
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    if is_end and not has_time:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _load_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = None
        wanted_lower = {
            "open",
            "high",
            "low",
            "close",
            "volume",
            "symbol",
            "datetime",
            "timestamp",
            "ts_event",
            "date",
            "time",
            "ts",
        }
        try:
            import pyarrow.parquet as pq  # type: ignore

            pq_file = pq.ParquetFile(path)
            schema = getattr(pq_file, "schema", None)
            schema_cols = list(getattr(schema, "names", []) or [])
            selected_cols = [c for c in schema_cols if str(c).strip().lower() in wanted_lower]
            if selected_cols:
                df = pd.read_parquet(path, columns=selected_cols)
        except Exception:
            df = None
        if df is None:
            df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        dt_col = None
        for c in ("datetime", "timestamp", "ts_event", "date", "time", "ts"):
            if c in df.columns:
                dt_col = c
                break
        if dt_col is None:
            raise ValueError("Could not find datetime column in source dataset")
        dt = pd.to_datetime(df[dt_col], errors="coerce", utc=False)
        df = df.drop(columns=[dt_col])
        df.index = dt

    df = df[~df.index.isna()]
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    df = df.sort_index()

    rename = {}
    for c in df.columns:
        low = str(c).strip().lower()
        if low in {"open", "high", "low", "close", "volume"}:
            rename[c] = low
    df = df.rename(columns=rename)

    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    keep_cols = ["open", "high", "low", "close", "volume"]
    if "symbol" in df.columns:
        keep_cols.append("symbol")
    out = df[keep_cols]
    try:
        out.attrs["source_cache_key"] = cache_key_for_source(path)
        out.attrs["source_label"] = path.name
        out.attrs["source_path"] = str(path.resolve())
    except Exception:
        pass
    return out


def _slice_for_backtest(
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    warmup_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    if test_df.empty:
        raise ValueError("No rows in requested date range")
    warmup_df = df[df.index < start_ts].tail(max(0, int(warmup_bars)))
    full_df = pd.concat([warmup_df, test_df])
    return full_df, warmup_df, test_df


def _resolve_source_path(raw: Optional[str]) -> Path:
    if raw:
        return Path(raw)
    default_parquet = Path("es_master_outrights.parquet")
    if default_parquet.exists():
        return default_parquet
    return Path("es_master.csv")


def _resolve_explicit_cache_path(raw: Optional[str]) -> Optional[Path]:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _apply_symbol_selection(
    df: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    symbol_mode: str,
    symbol_method: str,
    preferred_symbol: Optional[str],
) -> tuple[pd.DataFrame, str, dict]:
    source_df = df[df.index <= end_ts]
    if source_df.empty:
        raise ValueError("No rows on or before requested end timestamp.")
    if "symbol" not in source_df.columns:
        return source_df, "NO_SYMBOL", {}

    range_df = source_df[(source_df.index >= start_ts) & (source_df.index <= end_ts)]
    if range_df.empty:
        raise ValueError("No rows in requested date range.")
    symbols = sorted(range_df["symbol"].dropna().astype(str).unique().tolist())
    if not symbols:
        raise ValueError("No symbols found in requested date range.")

    selected = str(preferred_symbol or "").strip()
    mode_key = str(symbol_mode or "single").lower()
    method_key = str(symbol_method or "volume").lower()

    if mode_key != "single" and len(symbols) > 1:
        symbol_df, selected_label, symbol_map = apply_symbol_mode(source_df, mode_key, method_key)
        if symbol_df.empty:
            raise ValueError("No rows after auto symbol selection.")
        selected_range = symbol_df[(symbol_df.index >= start_ts) & (symbol_df.index <= end_ts)]
        if selected_range.empty:
            raise ValueError("No rows in selected range after auto symbol selection.")
        return symbol_df, str(selected_label), dict(symbol_map or {})

    if not selected or selected not in symbols:
        selected = choose_symbol(range_df, selected or None)
    symbol_df = source_df[source_df["symbol"].astype(str) == str(selected)]
    if symbol_df.empty:
        raise ValueError(f"No rows found for selected symbol: {selected}")
    selected_range = symbol_df[(symbol_df.index >= start_ts) & (symbol_df.index <= end_ts)]
    if selected_range.empty:
        raise ValueError(f"No rows in selected range for symbol: {selected}")
    return symbol_df, str(selected), {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute dist-mode MLPhysics backtest cache with ETA logging."
    )
    parser.add_argument("--source", help="Path to source OHLCV dataset (parquet or csv).")
    parser.add_argument("--start", required=True, help="Start datetime/date in ET.")
    parser.add_argument("--end", required=True, help="End datetime/date in ET.")
    parser.add_argument(
        "--warmup-bars",
        type=int,
        default=3000,
        help="Requested warmup bars before start (hard-capped to 3000).",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Overwrite existing cached parquet instead of loading it.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Log progress every N bars.",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=5.0,
        help="Minimum seconds between progress logs.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU inference tuning (default uses CUDA on GPU hosts).",
    )
    parser.add_argument(
        "--gpu-target-fraction",
        type=float,
        default=float(CONFIG.get("ML_PHYSICS_DIST_XGB_GPU_TARGET_FRACTION", 0.75) or 0.75),
        help="Target host feed fraction for dist XGBoost GPU inference tuning.",
    )
    parser.add_argument(
        "--symbol-mode",
        default=str(CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single"),
        help="Symbol mode: single | auto | auto_by_day | roll",
    )
    parser.add_argument(
        "--symbol-method",
        default=str(CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume") or "volume"),
        help="Auto symbol method: volume | rows",
    )
    parser.add_argument(
        "--symbol",
        default=str(CONFIG.get("TARGET_SYMBOL") or ""),
        help="Preferred symbol for single mode (optional).",
    )
    parser.add_argument(
        "--explicit-cache-file",
        default=str((CONFIG.get("ML_PHYSICS_OPT", {}) or {}).get("dist_precomputed_file", "") or ""),
        help="Canonical cache path to copy result to for deterministic backtest lookup.",
    )
    parser.add_argument(
        "--no-register-explicit",
        action="store_true",
        help="Skip copying output to --explicit-cache-file.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    source_path = _resolve_source_path(args.source)
    start_ts = _parse_ts(args.start, is_end=False)
    end_ts = _parse_ts(args.end, is_end=True)
    if end_ts < start_ts:
        raise ValueError("end must be >= start")

    dist_max_bars = int(CONFIG.get("ML_PHYSICS_DIST_MAX_BARS", 3000) or 3000)
    warmup_cap = min(3000, max(1, dist_max_bars))
    warmup_bars = min(max(0, int(args.warmup_bars or 0)), warmup_cap)

    opt_cfg = CONFIG.get("ML_PHYSICS_OPT", {}) or {}
    opt_cfg["enabled"] = True
    opt_cfg["mode"] = "backtest"
    opt_cfg["prediction_cache"] = True
    opt_cfg["overwrite_cache"] = bool(args.overwrite_cache)
    if str(args.explicit_cache_file or "").strip():
        opt_cfg["dist_precomputed_file"] = str(args.explicit_cache_file).strip()
    CONFIG["ML_PHYSICS_OPT"] = opt_cfg
    gpu_enabled = not bool(args.cpu_only)
    target_frac = float(args.gpu_target_fraction)
    if target_frac < 0.10:
        target_frac = 0.10
    if target_frac > 1.0:
        target_frac = 1.0
    CONFIG["ML_PHYSICS_DIST_XGB_GPU_ENABLED"] = gpu_enabled
    CONFIG["ML_PHYSICS_DIST_XGB_DEVICE"] = "cuda"
    CONFIG["ML_PHYSICS_DIST_XGB_PREDICTOR"] = "gpu_predictor"
    CONFIG["ML_PHYSICS_DIST_XGB_GPU_TARGET_FRACTION"] = target_frac

    logging.info(
        "Dist precompute config: source=%s start=%s end=%s warmup=%d (cap=%d) dist_max=%d "
        "overwrite_cache=%s gpu_enabled=%s gpu_target_fraction=%.2f symbol_mode=%s symbol_method=%s "
        "symbol_pref=%s",
        source_path,
        start_ts,
        end_ts,
        warmup_bars,
        warmup_cap,
        dist_max_bars,
        bool(args.overwrite_cache),
        gpu_enabled,
        target_frac,
        str(args.symbol_mode),
        str(args.symbol_method),
        str(args.symbol or ""),
    )

    df = _load_source(source_path)
    symbol_df, selected_symbol, symbol_map = _apply_symbol_selection(
        df,
        start_ts=start_ts,
        end_ts=end_ts,
        symbol_mode=str(args.symbol_mode),
        symbol_method=str(args.symbol_method),
        preferred_symbol=str(args.symbol or "").strip() or None,
    )
    CONFIG["TARGET_SYMBOL"] = selected_symbol
    symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    full_df, warmup_df, test_df = _slice_for_backtest(symbol_df, start_ts, end_ts, warmup_bars)
    source_attrs = getattr(df, "attrs", {}) or {}
    full_df = attach_backtest_symbol_context(
        full_df,
        selected_symbol,
        str(args.symbol_mode),
        source_key=source_attrs.get("source_cache_key"),
        source_label=source_attrs.get("source_label"),
        source_path=source_attrs.get("source_path"),
    )

    strategy = MLPhysicsStrategy()
    if not strategy.model_loaded:
        raise RuntimeError("MLPhysicsStrategy model is not loaded.")
    if not bool(getattr(strategy, "_dist_mode", False)):
        raise RuntimeError(
            "Dist mode is not active. Set CONFIG['ML_PHYSICS_REPLACE_WITH_DIST']=True before running."
        )

    cache_path = strategy._dist_cache_path(full_df)
    logging.info(
        "Preparing cache: bars(full=%d warmup=%d test=%d) selected_symbol=%s symbol_map_days=%d cache_path=%s",
        len(full_df),
        len(warmup_df),
        len(test_df),
        selected_symbol,
        len(symbol_map),
        cache_path,
    )

    t0 = time.perf_counter()
    ok = strategy.precompute_dist_backtest_signals(
        full_df,
        progress_every=max(1, int(args.progress_every or 1)),
        progress_min_interval_sec=max(0.0, float(args.progress_interval_sec or 0.0)),
    )
    elapsed = time.perf_counter() - t0
    if not ok:
        raise RuntimeError("Dist precompute returned False.")

    exists = cache_path.exists()
    size_bytes = cache_path.stat().st_size if exists else 0
    logging.info(
        "Dist precompute complete: ok=%s elapsed=%.2fs cache_exists=%s cache_size_bytes=%d",
        ok,
        elapsed,
        exists,
        size_bytes,
    )

    if exists and not bool(args.no_register_explicit):
        explicit_cache_path = _resolve_explicit_cache_path(args.explicit_cache_file)
        if explicit_cache_path is not None:
            explicit_cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.resolve() != explicit_cache_path.resolve():
                shutil.copy2(cache_path, explicit_cache_path)
                logging.info(
                    "Registered canonical dist cache: %s -> %s",
                    cache_path,
                    explicit_cache_path,
                )
            else:
                logging.info("Canonical dist cache already at %s", explicit_cache_path)


if __name__ == "__main__":
    main()
