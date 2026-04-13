import argparse
import concurrent.futures as cf
import datetime as dt
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from zoneinfo import ZoneInfo

import backtest_mes_et as bt
from dynamic_signal_engine3 import get_signal_engine
from volatility_filter import volatility_filter

NY_TZ = ZoneInfo("America/New_York")


STRATEGY_ID_RE = re.compile(
    r"^(?P<tf>\d+min)_(?P<session>\d{2}-\d{2})_"
    r"(?P<stype>Long_Rev|Short_Rev|Long_Mom|Short_Mom)_T"
)


class _TimedProgressLogger:
    def __init__(self, label: str, total: int, interval_seconds: float = 60.0):
        self.label = str(label)
        self.total = int(max(0, total))
        self.interval_seconds = float(max(1.0, interval_seconds))
        self._last_value = -1
        self._start_ts = time.monotonic()
        self._last_log_ts = 0.0
        logging.info(
            "%s start: total=%d log_interval=%ss",
            self.label,
            self.total,
            int(self.interval_seconds),
        )
        self._flush_handlers()
        self._log_progress(0, force=True)

    @staticmethod
    def _flush_handlers() -> None:
        root = logging.getLogger()
        for handler in root.handlers:
            try:
                handler.flush()
            except Exception:
                continue

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        sec = int(max(0.0, seconds))
        mins, rem = divmod(sec, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{rem:02d}"
        return f"{mins:02d}:{rem:02d}"

    def _log_progress(self, current: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and current < self.total and (now - self._last_log_ts) < self.interval_seconds:
            return
        if self.total <= 0:
            pct = 100.0
        else:
            pct = (100.0 * float(current)) / float(self.total)
        elapsed = max(0.0, now - self._start_ts)
        rate = float(current) / elapsed if elapsed > 0 else 0.0
        eta = (float(self.total - current) / rate) if rate > 0 and current < self.total else 0.0
        logging.info(
            "%s progress: %.1f%% (%d/%d) elapsed=%s eta=%s",
            self.label,
            pct,
            int(current),
            int(self.total),
            self._fmt_duration(elapsed),
            self._fmt_duration(eta),
        )
        self._flush_handlers()
        self._last_log_ts = now

    def update(self, current: int) -> None:
        if self.total <= 0:
            self._log_progress(0, force=True)
            return
        cur = int(min(max(0, current), self.total))
        if cur <= self._last_value and cur < self.total:
            return
        self._last_value = cur
        self._log_progress(cur, force=(cur == 0 or cur >= self.total))


def _parse_strategy_id(strategy_id: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    if not strategy_id:
        return None, None, None
    match = STRATEGY_ID_RE.match(str(strategy_id))
    if not match:
        return None, None, None
    return match.group("tf"), match.group("session"), match.group("stype")


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, closed="left", label="left").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"])
    if "volume" in out.columns:
        out["volume"] = out["volume"].fillna(0.0)
    return out


def _compute_atr(df_5m: pd.DataFrame, period: int, median_window: int) -> pd.DataFrame:
    high = df_5m["high"]
    low = df_5m["low"]
    close = df_5m["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / float(period), adjust=False).mean()
    atr_median = atr.rolling(median_window, min_periods=median_window).median()
    df_5m = df_5m.copy()
    df_5m["atr_5m"] = atr
    df_5m["atr_5m_median"] = atr_median
    return df_5m


def _compute_price_location(df_5m: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    high_roll = df_5m["high"].rolling(window, min_periods=window).max()
    low_roll = df_5m["low"].rolling(window, min_periods=window).min()
    denom = (high_roll - low_roll).replace(0, np.nan)
    price_location = (df_5m["close"] - low_roll) / denom
    df_5m = df_5m.copy()
    df_5m["price_location"] = price_location.clip(lower=0.0, upper=1.0)
    return df_5m


def _compute_vwap(df_1m: pd.DataFrame) -> pd.Series:
    idx = df_1m.index
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ)
    day_index = idx.date
    typical_price = (df_1m["high"] + df_1m["low"] + df_1m["close"]) / 3.0
    volume = df_1m.get("volume")
    if volume is None:
        volume = pd.Series(1.0, index=df_1m.index)
    volume = volume.fillna(0.0)
    cum_pv = (typical_price * volume).groupby(day_index).cumsum()
    cum_v = volume.groupby(day_index).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)
    return vwap


def _session_bucket(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    hour = ts.hour
    start = (hour // 3) * 3
    end = start + 3
    return f"{start:02d}-{end:02d}"


def _normalize_vol_regime(value: Optional[str]) -> str:
    regime = str(value or "").strip().lower()
    if regime in {"ultra_low", "low", "normal", "high"}:
        return regime
    return "unknown"


def _thresh_bucket(value) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    # DE3 thresholds are configured as small integer bands; store a compact bucket key.
    iv = int(round(v))
    return f"T{iv}"


def _coerce_ts(val) -> Optional[pd.Timestamp]:
    if val is None:
        return None
    if isinstance(val, pd.Timestamp):
        ts = val
    elif isinstance(val, dt.datetime):
        ts = pd.Timestamp(val)
    else:
        try:
            ts = pd.to_datetime(val)
        except Exception:
            return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    return ts


def _is_bar_close(ts: pd.Timestamp, minutes: int) -> bool:
    return ts.minute % minutes == minutes - 1 and ts.second == 0


def _simulate_trade_outcome(
    side: str,
    entry_idx: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    sl_dist: float,
    tp_dist: float,
) -> int:
    entry_price = float(open_arr[entry_idx])
    if side == "LONG":
        stop_price = entry_price - sl_dist
        take_price = entry_price + tp_dist
    else:
        stop_price = entry_price + sl_dist
        take_price = entry_price - tp_dist

    for j in range(entry_idx, len(open_arr)):
        bar_open = float(open_arr[j])
        bar_high = float(high_arr[j])
        bar_low = float(low_arr[j])
        bar_close = float(close_arr[j])

        if bt.BACKTEST_GAP_FILLS:
            if side == "LONG":
                if bar_open <= stop_price:
                    return 0
                if bar_open >= take_price:
                    return 1
            else:
                if bar_open >= stop_price:
                    return 0
                if bar_open <= take_price:
                    return 1

        hit_stop = bar_low <= stop_price if side == "LONG" else bar_high >= stop_price
        hit_take = bar_high >= take_price if side == "LONG" else bar_low <= take_price
        if hit_stop and hit_take:
            _, reason = bt._resolve_sl_tp_conflict(side, bar_open, bar_close, stop_price, take_price)
            return 1 if reason.startswith("take") else 0
        if hit_stop:
            return 0
        if hit_take:
            return 1

    return 0


def build_context_dataset(
    csv_path: Path,
    out_dir: Path,
    start: Optional[str],
    end: Optional[str],
    atr_period: int,
    atr_median_window: int,
    overwrite: bool,
    workers: int = 1,
) -> Path:
    out_dir = Path(out_dir)
    if overwrite and out_dir.exists():
        for item in out_dir.glob("**/*"):
            if item.is_file():
                item.unlink()
        for item in sorted(out_dir.glob("**/*"), reverse=True):
            if item.is_dir():
                try:
                    item.rmdir()
                except OSError:
                    pass
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading bars: %s", csv_path)
    df = bt.load_csv_cached(Path(csv_path), cache_dir=Path("cache"))
    if start:
        start_ts = pd.to_datetime(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(NY_TZ)
        df = df[df.index >= start_ts]
    if end:
        end_ts = pd.to_datetime(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(NY_TZ)
        df = df[df.index < end_ts]
    if df.empty:
        raise ValueError("No bars in the requested date range.")

    logging.info("Preparing DE3 context dataset (pre-guard candidates)...")
    engine = get_signal_engine()

    logging.info("Preparing 5m/15m resampled bars...")
    df_5m = _resample_ohlcv(df, "5min")
    df_15m = _resample_ohlcv(df, "15min")
    df_5m = _compute_atr(df_5m, atr_period, atr_median_window)
    df_5m = _compute_price_location(df_5m, window=20)

    logging.info("Preparing VWAP series...")
    vwap_series = _compute_vwap(df)

    open_arr = df["open"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    close_arr = df["close"].to_numpy()
    idx = df.index
    indexer = pd.Index(idx)
    df_5m_index = df_5m.index
    df_15m_index = df_15m.index
    atr_arr = df_5m["atr_5m"].to_numpy()
    atr_med_arr = df_5m["atr_5m_median"].to_numpy()
    price_loc_arr = df_5m["price_location"].to_numpy()
    workers = max(1, int(workers or 1))

    records = []
    min_cfg = bt.CONFIG.get("SLTP_MIN", {}) or {}
    min_sl = float(min_cfg.get("sl", 1.25))

    # Precompute timestamp/index lookups once to avoid repeated index scans.
    current_times = df_5m_index + pd.Timedelta(minutes=4)
    entry_times = current_times + pd.Timedelta(minutes=1)
    current_pos = indexer.get_indexer(current_times)
    entry_pos = indexer.get_indexer(entry_times)
    valid_current = current_pos >= 0
    valid_entry = entry_pos >= 0

    # Precompute matching 15m bar location for each 5m decision bar.
    floor_15 = current_times.floor("15min")
    is_15_close = (current_times.minute % 15 == 14) & (current_times.second == 0)
    last_complete_15m = floor_15.where(is_15_close, floor_15 - pd.Timedelta(minutes=15))
    pos_15m = np.asarray(df_15m_index.searchsorted(last_complete_15m, side="right"), dtype=int) - 1
    pos_15m[last_complete_15m < df_15m_index[0]] = -1

    # Cache volatility regime by (session, threshold-key). This preserves behavior while
    # eliminating expensive repeated rolling-vol computations per bar.
    vol_regime_by_bar = np.full(len(df_5m_index), "unknown", dtype=object)
    vol_cache: dict[tuple[str, str], str] = {}
    vol_progress = _TimedProgressLogger("DE3 volatility regime cache", len(df_5m_index))
    for i, current_time in enumerate(current_times):
        vol_progress.update(i + 1)
        if not valid_current[i]:
            continue
        try:
            session = volatility_filter.get_session(int(current_time.hour))
            _, vol_key = volatility_filter.get_thresholds(current_time)
            cache_key = (str(session), str(vol_key))
            if cache_key not in vol_cache:
                vol_regime, _, _ = volatility_filter.get_regime(df, current_time)
                vol_cache[cache_key] = _normalize_vol_regime(vol_regime)
            vol_regime_by_bar[i] = vol_cache[cache_key]
        except Exception:
            vol_regime_by_bar[i] = "unknown"

    try:
        vwap_at_entry = vwap_series.reindex(entry_times, method="ffill")
    except Exception:
        vwap_at_entry = pd.Series(index=entry_times, dtype=float)

    def _process_bar(i: int) -> list[dict]:
        if i < 1:
            return []
        if not valid_current[i] or not valid_entry[i]:
            return []

        current_time = current_times[i]
        p15 = int(pos_15m[i])
        if p15 < 1:
            return []

        # Minimal rolling slices expected by DE3 signal engine.
        df_5m_slice = df_5m.iloc[max(0, i - 20) : i + 1]
        df_15m_slice = df_15m.iloc[max(0, p15 - 1) : p15 + 1]
        candidates = engine.check_signals(current_time, df_5m_slice, df_15m_slice, emit_logs=False)
        if not candidates:
            return []

        entry_idx = int(entry_pos[i])
        if entry_idx < 0:
            return []
        entry_time = entry_times[i]

        atr_5m = float(atr_arr[i])
        atr_med = float(atr_med_arr[i])
        price_loc = float(price_loc_arr[i]) if np.isfinite(price_loc_arr[i]) else 0.5
        if not np.isfinite(atr_5m) or atr_5m <= 0:
            return []
        if not np.isfinite(atr_med) or atr_med <= 0:
            return []
        if not np.isfinite(price_loc):
            price_loc = 0.5

        entry_price = float(open_arr[entry_idx])
        vwap_val = vwap_at_entry.iloc[i] if i < len(vwap_at_entry) else np.nan
        if pd.isna(vwap_val):
            vwap_dist_atr = 0.0
        else:
            vwap_dist_atr = float(abs(entry_price - float(vwap_val)) / atr_5m)

        vol_regime = str(vol_regime_by_bar[i] or "unknown")
        trade_date = entry_time.date()
        session_bucket = _session_bucket(entry_time)

        out: list[dict] = []
        for cand in candidates:
            sub_strategy = cand.get("strategy_id")
            tf, sess, stype = _parse_strategy_id(sub_strategy)
            if tf is None or stype is None:
                continue

            thresh_bucket = _thresh_bucket(cand.get("thresh"))
            sl_points = max(min_sl, float(cand.get("sl", min_sl)))
            tp_points = float(cand.get("tp", 0.0))
            if sl_points <= 0 or tp_points <= 0:
                continue

            outcome = _simulate_trade_outcome(
                side=str(cand.get("signal", "")).upper(),
                entry_idx=entry_idx,
                open_arr=open_arr,
                high_arr=high_arr,
                low_arr=low_arr,
                close_arr=close_arr,
                sl_dist=sl_points,
                tp_dist=tp_points,
            )

            out.append(
                {
                    "entry_time": entry_time.to_pydatetime(),
                    "trade_date": str(trade_date),
                    "session_bucket": session_bucket,
                    "timeframe": tf,
                    "strategy_type": stype,
                    "vol_regime": vol_regime,
                    "thresh_bucket": thresh_bucket,
                    "atr_5m": atr_5m,
                    "atr_5m_median": atr_med,
                    "price_location": price_loc,
                    "vwap_dist_atr": vwap_dist_atr,
                    "sl_points": float(sl_points),
                    "outcome": int(outcome),
                    "year": entry_time.year,
                    "month": entry_time.month,
                }
            )
        return out

    total_bars = len(df_5m_index)
    progress = _TimedProgressLogger("DE3 context build", total_bars)
    if workers <= 1:
        for i in range(total_bars):
            progress.update(i + 1)
            recs = _process_bar(i)
            if recs:
                records.extend(recs)
    else:
        logging.info("DE3 context build using thread workers=%d", workers)
        processed = 0
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for recs in ex.map(_process_bar, range(total_bars)):
                processed += 1
                progress.update(processed)
                if recs:
                    records.extend(recs)

    if not records:
        raise ValueError("No usable DE3 trades after feature extraction.")

    out_df = pd.DataFrame.from_records(records)
    logging.info("Final dataset rows: %s", len(out_df))

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except Exception as exc:
        raise RuntimeError("pyarrow is required to write the dataset.") from exc

    table = pa.Table.from_pandas(out_df, preserve_index=False)
    ds.write_dataset(
        table,
        out_dir,
        format="parquet",
        partitioning=["year", "month"],
        existing_data_behavior="overwrite_or_ignore" if overwrite else "overwrite_or_ignore",
    )
    logging.info("Wrote dataset to %s", out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DE3 context dataset from backtest.")
    parser.add_argument("--csv", default="es_master.csv", help="Path to ES/MES source (.csv or .parquet).")
    parser.add_argument("--out-dir", default="cache/de3_context_dataset", help="Output dataset directory.")
    parser.add_argument("--start", "--train-start", dest="start", default=None, help="Train start date (YYYY-MM-DD).")
    parser.add_argument("--end", "--train-end", dest="end", default=None, help="Train end date (YYYY-MM-DD).")
    parser.add_argument("--atr-period", type=int, default=20, help="ATR period for 5m bars.")
    parser.add_argument(
        "--atr-median-window",
        type=int,
        default=390,
        help="ATR median lookback window (5m bars).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Thread workers for bar processing (1 = single-thread).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output dataset directory.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )
    if not args.start or not args.end:
        logging.warning(
            "Dataset window is not fully bounded (start=%s end=%s). "
            "For strict OOS workflows, pass both --train-start and --train-end.",
            args.start,
            args.end,
        )
    build_context_dataset(
        csv_path=Path(args.csv),
        out_dir=Path(args.out_dir),
        start=args.start,
        end=args.end,
        atr_period=args.atr_period,
        atr_median_window=args.atr_median_window,
        overwrite=args.overwrite,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
