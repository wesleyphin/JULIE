import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd


def cache_key_for_source(path: Path) -> str:
    path = path.resolve()
    stat = path.stat()
    token = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = None
        if "ts_event" in df.columns:
            date_col = "ts_event"
        elif "timestamp" in df.columns:
            date_col = "timestamp"
        elif "date" in df.columns:
            date_col = "date"
        if date_col is None:
            raise ValueError("Data missing timestamp column")
        df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df.dropna(subset=["timestamp"], inplace=True)
        df.set_index("timestamp", inplace=True)

    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('"', "")
                .str.replace(",", "")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    keep_cols = [c for c in ["open", "high", "low", "close", "volume", "symbol"] if c in df.columns]
    return df[keep_cols]


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read parquet: {exc}") from exc


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to write parquet: {exc}") from exc


def load_bars(
    path: Path,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = _read_parquet(path)
        return _normalize_bars(df)
    if suffix in (".feather", ".ft"):
        try:
            df = pd.read_feather(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read feather: {exc}") from exc
        return _normalize_bars(df)

    cache_dir = Path(cache_dir) if cache_dir else None
    cache_path = None
    if cache_dir and use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = cache_key_for_source(path)
        cache_path = cache_dir / f"{key}.parquet"
        if cache_path.exists():
            try:
                cached = _read_parquet(cache_path)
                return _normalize_bars(cached)
            except Exception as exc:
                logging.warning("Cache read failed (%s), falling back to CSV.", exc)

    with path.open("r", errors="ignore") as f:
        first = f.readline()
        second = f.readline()
        needs_skip = "Time Series" in first and "Date" in second

    df = pd.read_csv(path, skiprows=1 if needs_skip else 0)
    df = _normalize_bars(df)

    if cache_path and use_cache:
        try:
            _write_parquet(df, cache_path)
        except Exception as exc:
            logging.warning("Cache write failed (%s).", exc)

    return df
