import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for name in candidates:
        if not name:
            continue
        if name in df.columns:
            return name
    return None


def _parse_timestamp(series: pd.Series, assume_timezone: str) -> pd.Series:
    ts = pd.to_datetime(series, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(assume_timezone, ambiguous="NaT", nonexistent="shift_forward")
    return ts.dt.tz_convert("UTC")


def _build_from_csv(
    source: Path,
    assume_timezone: str,
    timestamp_column: str,
    sl_column: str,
    tp_column: str,
    chop_column: str,
    default_sl: float,
    default_tp: float,
    default_chop: float,
) -> pd.DataFrame:
    if source.suffix.lower() == ".parquet":
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source, low_memory=False)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    ts_col = _pick_column(
        df,
        [
            timestamp_column,
            "timestamp",
            "ts",
            "ts_event",
            "datetime",
            "time",
        ],
    )
    sl_col = _pick_column(
        df,
        [
            sl_column,
            "sl_multiplier",
            "sl_mult",
            "dynamic_sl_multiplier",
        ],
    )
    tp_col = _pick_column(
        df,
        [
            tp_column,
            "tp_multiplier",
            "tp_mult",
            "dynamic_tp_multiplier",
        ],
    )
    chop_col = _pick_column(
        df,
        [
            chop_column,
            "chop_multiplier",
            "chop_mult",
            "dynamic_chop_multiplier",
        ],
    )
    if ts_col is None or sl_col is None or tp_col is None:
        raise ValueError(
            "CSV/parquet must contain timestamp + SL/TP multiplier columns. "
            f"Resolved columns: ts={ts_col}, sl={sl_col}, tp={tp_col}"
        )

    out = pd.DataFrame(
        {
            "timestamp": _parse_timestamp(df[ts_col], assume_timezone),
            "sl_multiplier": pd.to_numeric(df[sl_col], errors="coerce").fillna(default_sl),
            "tp_multiplier": pd.to_numeric(df[tp_col], errors="coerce").fillna(default_tp),
            "chop_multiplier": (
                pd.to_numeric(df[chop_col], errors="coerce").fillna(default_chop)
                if chop_col is not None
                else default_chop
            ),
        }
    )
    out = out.reset_index(drop=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out


def _build_from_log(
    source: Path,
    assume_timezone: str,
    default_chop: float,
) -> pd.DataFrame:
    line_re = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}).*NEW MULTIPLIERS \| "
        r"SL:\s*(?P<sl>[0-9]*\.?[0-9]+)x \| TP:\s*(?P<tp>[0-9]*\.?[0-9]+)x(?: \| CHOP:\s*(?P<chop>[0-9]*\.?[0-9]+)x)?"
    )
    rows = []
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = line_re.search(line)
            if not match:
                continue
            rows.append(
                {
                    "timestamp": match.group("ts"),
                    "sl_multiplier": float(match.group("sl")),
                    "tp_multiplier": float(match.group("tp")),
                    "chop_multiplier": float(match.group("chop") or default_chop),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["timestamp"] = pd.to_datetime(
        out["timestamp"], format="%Y-%m-%d %H:%M:%S,%f", errors="coerce"
    )
    out["timestamp"] = out["timestamp"].dt.tz_localize(
        assume_timezone, ambiguous="NaT", nonexistent="shift_forward"
    )
    out["timestamp"] = out["timestamp"].dt.tz_convert("UTC")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = (
            series.astype(str)
            .str.replace('"', "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _load_ohlcv_source(source: Path, assume_timezone: str) -> pd.DataFrame:
    if source.suffix.lower() == ".parquet":
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source, low_memory=False)
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    lookup = {str(col).strip().lower(): str(col).strip() for col in df.columns}

    if isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(df.index, errors="coerce")
    else:
        ts_col = _pick_column(
            df,
            [
                lookup.get("timestamp", ""),
                lookup.get("ts", ""),
                lookup.get("ts_event", ""),
                lookup.get("datetime", ""),
                lookup.get("time", ""),
                lookup.get("date", ""),
            ],
        )
        if ts_col is None:
            raise ValueError(
                "Could not find timestamp column in source. "
                "Expected one of: timestamp, ts, ts_event, datetime, time, date."
            )
        ts = pd.to_datetime(df[ts_col], errors="coerce")

    ts_index = pd.DatetimeIndex(ts)
    if ts_index.tz is None:
        ts_index = ts_index.tz_localize(
            assume_timezone,
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    ts_index = ts_index.tz_convert("America/New_York")

    open_col = _pick_column(df, [lookup.get("open", "")])
    high_col = _pick_column(df, [lookup.get("high", "")])
    low_col = _pick_column(df, [lookup.get("low", "")])
    close_col = _pick_column(df, [lookup.get("close", "")])
    if not all([open_col, high_col, low_col, close_col]):
        raise ValueError(
            "Source must contain OHLC columns: open/high/low/close. "
            f"Resolved: open={open_col} high={high_col} low={low_col} close={close_col}"
        )

    out = pd.DataFrame(
        {
            "open": _coerce_numeric(df[open_col]),
            "high": _coerce_numeric(df[high_col]),
            "low": _coerce_numeric(df[low_col]),
            "close": _coerce_numeric(df[close_col]),
        },
        index=ts_index,
    )
    out = out[~out.index.isna()]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _session_labels(index: pd.DatetimeIndex) -> pd.Series:
    hours = index.hour.to_numpy()
    minutes = index.minute.to_numpy()
    n = len(index)

    base = np.full(n, "POST_MARKET", dtype=object)
    base[(hours >= 18) | (hours < 3)] = "ASIA"
    base[(hours >= 3) & (hours < 8)] = "LONDON"
    base[(hours >= 8) & (hours < 12)] = "NY_AM"
    base[(hours >= 12) & (hours < 17)] = "NY_PM"

    session = base.copy()
    lunch_mask = (
        ((base == "NY_AM") & (((hours == 10) & (minutes >= 30)) | (hours == 11)))
        | ((base == "NY_PM") & ((hours == 12) & (minutes < 30)))
    )
    session[lunch_mask] = "NY_LUNCH"
    close_mask = (base == "NY_PM") & (hours >= 15)
    session[close_mask] = "NY_CLOSE"

    return pd.Series(session, index=index, dtype="object")


def _compute_regime_features(df_1m: pd.DataFrame) -> pd.DataFrame:
    bars_15m = df_1m.resample("15min").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if bars_15m.empty:
        return pd.DataFrame(index=df_1m.index, data={"adx_15m": 0.0, "chop_15m": 50.0})

    high = bars_15m["high"]
    low = bars_15m["low"]
    close = bars_15m["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    alpha = 1.0 / 14.0
    tr_ema = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = (
        100.0
        * pd.Series(plus_dm, index=bars_15m.index).ewm(alpha=alpha, adjust=False).mean()
        / tr_ema.replace(0.0, np.nan)
    )
    minus_di = (
        100.0
        * pd.Series(minus_dm, index=bars_15m.index).ewm(alpha=alpha, adjust=False).mean()
        / tr_ema.replace(0.0, np.nan)
    )
    sum_di = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / sum_di
    adx = dx.ewm(alpha=alpha, adjust=False).mean().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    period = 14.0
    sum_tr = tr.rolling(int(period), min_periods=int(period)).sum()
    period_range = (
        high.rolling(int(period), min_periods=int(period)).max()
        - low.rolling(int(period), min_periods=int(period)).min()
    )
    ratio = sum_tr / period_range.replace(0.0, np.nan)
    chop = 100.0 * np.log10(ratio) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50.0)

    features_15m = pd.DataFrame({"adx_15m": adx, "chop_15m": chop}, index=bars_15m.index)
    features_1m = features_15m.reindex(df_1m.index, method="ffill")
    features_1m["adx_15m"] = features_1m["adx_15m"].fillna(0.0)
    features_1m["chop_15m"] = features_1m["chop_15m"].fillna(50.0)
    return features_1m


def _apply_proxy_policy(
    session_label: pd.Series,
    adx_15m: pd.Series,
    chop_15m: pd.Series,
) -> pd.DataFrame:
    n = len(session_label)
    sl = np.full(n, 1.0, dtype=float)
    tp = np.full(n, 1.0, dtype=float)
    chop_mult = np.full(n, 1.0, dtype=float)

    session_arr = session_label.to_numpy()
    adx_arr = adx_15m.to_numpy(dtype=float)
    chop_arr = chop_15m.to_numpy(dtype=float)

    lunch = session_arr == "NY_LUNCH"
    close = session_arr == "NY_CLOSE"
    fixed = lunch | close

    sl[lunch] = 1.10
    tp[lunch] = 0.60
    chop_mult[lunch] = 2.50

    sl[close] = 1.05
    tp[close] = 0.90
    chop_mult[close] = 1.50

    choppy = (~fixed) & ((chop_arr >= 61.8) | (adx_arr < 20.0))
    trending = (~fixed) & (~choppy) & (adx_arr > 25.0) & (chop_arr <= 55.0)

    sl[choppy] = 1.10
    tp[choppy] = 0.80
    chop_mult[choppy] = 1.50

    sl[trending] = 1.00
    tp[trending] = 1.30
    chop_mult[trending] = 0.60

    trend_status = np.where(adx_arr > 25.0, "TRENDING", "CHOPPY_RANGING")
    chop_status = np.where(chop_arr >= 61.8, "CHOPPY", np.where(chop_arr <= 38.2, "TRENDING", "NEUTRAL"))
    regime_key = (
        pd.Series(session_arr, dtype="string")
        + "|"
        + pd.Series(trend_status, dtype="string")
        + "|"
        + pd.Series(chop_status, dtype="string")
    )

    return pd.DataFrame(
        {
            "sl_multiplier": sl,
            "tp_multiplier": tp,
            "chop_multiplier": chop_mult,
            "session_label": session_arr,
            "trend_status": trend_status,
            "chop_status": chop_status,
            "regime_key": regime_key.to_numpy(dtype=object),
        },
        index=session_label.index,
    )


def _build_from_es_master(
    source: Path,
    assume_timezone: str,
    event_mode: str,
    min_interval_minutes: float,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    df_1m = _load_ohlcv_source(source, assume_timezone)
    if df_1m.empty:
        return pd.DataFrame()

    if start:
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("America/New_York")
        else:
            start_ts = start_ts.tz_convert("America/New_York")
        df_1m = df_1m.loc[df_1m.index >= start_ts]
    if end:
        end_ts = pd.Timestamp(end)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("America/New_York")
        else:
            end_ts = end_ts.tz_convert("America/New_York")
        df_1m = df_1m.loc[df_1m.index <= end_ts]
    if df_1m.empty:
        return pd.DataFrame()

    session = _session_labels(df_1m.index)
    feats = _compute_regime_features(df_1m)
    policy = _apply_proxy_policy(session, feats["adx_15m"], feats["chop_15m"])

    base = pd.DataFrame(
        {
            "timestamp": pd.Series(df_1m.index, index=df_1m.index).dt.tz_convert("UTC"),
            "sl_multiplier": policy["sl_multiplier"].to_numpy(dtype=float),
            "tp_multiplier": policy["tp_multiplier"].to_numpy(dtype=float),
            "chop_multiplier": policy["chop_multiplier"].to_numpy(dtype=float),
            "session_label": policy["session_label"].to_numpy(dtype=object),
            "trend_status": policy["trend_status"].to_numpy(dtype=object),
            "chop_status": policy["chop_status"].to_numpy(dtype=object),
            "regime_key": policy["regime_key"].to_numpy(dtype=object),
        }
    )

    mode = (event_mode or "regime_change").strip().lower()
    if mode == "per_bar":
        out = base
    elif mode == "regime_change":
        change = base["regime_key"] != base["regime_key"].shift(1)
        candidate_idx = np.flatnonzero(change.to_numpy())
        if len(candidate_idx) == 0:
            return pd.DataFrame()
        selected = []
        min_gap = pd.Timedelta(minutes=float(max(0.0, min_interval_minutes)))
        last_ts = None
        timestamps = base["timestamp"].to_numpy()
        for idx_val in candidate_idx:
            ts = pd.Timestamp(timestamps[idx_val])
            if last_ts is None or (ts - last_ts) >= min_gap:
                selected.append(idx_val)
                last_ts = ts
        out = base.iloc[selected]
    else:
        raise ValueError(f"Unsupported event mode: {event_mode}")

    out = out.reset_index(drop=True)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out


def _build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    source = Path(args.source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source not found: {source}")

    mode = (args.mode or "auto").strip().lower()
    if mode == "auto":
        if source.suffix.lower() in {".log", ".txt"}:
            mode = "log"
        elif "es_master" in source.name.lower():
            mode = "es_master"
        else:
            mode = "csv"

    if mode == "log":
        out = _build_from_log(
            source=source,
            assume_timezone=args.assume_timezone,
            default_chop=args.default_chop_multiplier,
        )
    elif mode == "csv":
        out = _build_from_csv(
            source=source,
            assume_timezone=args.assume_timezone,
            timestamp_column=args.timestamp_column,
            sl_column=args.sl_column,
            tp_column=args.tp_column,
            chop_column=args.chop_column,
            default_sl=args.default_sl_multiplier,
            default_tp=args.default_tp_multiplier,
            default_chop=args.default_chop_multiplier,
        )
    elif mode == "es_master":
        out = _build_from_es_master(
            source=source,
            assume_timezone=args.assume_timezone,
            event_mode=args.event_mode,
            min_interval_minutes=args.min_interval_minutes,
            start=args.start,
            end=args.end,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if out.empty:
        raise ValueError("No multiplier rows were extracted from the source.")
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a backtest multiplier dataset (timestamp + SL/TP/CHOP multipliers)."
    )
    parser.add_argument("--source", required=True, help="Input path (CSV/parquet or log file).")
    parser.add_argument(
        "--output",
        default="backtest_gemini_multipliers.csv",
        help="Output path (.csv or .parquet).",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "csv", "log", "es_master"],
        help="Source parser mode.",
    )
    parser.add_argument(
        "--assume-timezone",
        default="America/New_York",
        help="Timezone used when source timestamps are naive.",
    )
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--sl-column", default="sl_multiplier")
    parser.add_argument("--tp-column", default="tp_multiplier")
    parser.add_argument("--chop-column", default="chop_multiplier")
    parser.add_argument("--default-sl-multiplier", type=float, default=1.0)
    parser.add_argument("--default-tp-multiplier", type=float, default=1.0)
    parser.add_argument("--default-chop-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--event-mode",
        default="regime_change",
        choices=["regime_change", "per_bar"],
        help="Output density for es_master mode.",
    )
    parser.add_argument(
        "--min-interval-minutes",
        type=float,
        default=30.0,
        help="Minimum gap between updates in regime_change mode.",
    )
    parser.add_argument(
        "--start",
        default="",
        help="Optional start timestamp filter (e.g. 2025-01-01 or 2025-01-01 09:30).",
    )
    parser.add_argument(
        "--end",
        default="",
        help="Optional end timestamp filter (e.g. 2025-12-31 or 2025-12-31 16:00).",
    )
    args = parser.parse_args()

    out = _build_dataset(args)
    dst = Path(args.output).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix.lower() == ".parquet":
        out.to_parquet(dst, index=False)
    else:
        saved = out.copy()
        saved["timestamp"] = saved["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        saved.to_csv(dst, index=False)

    print(
        "Built multiplier dataset:",
        f"rows={len(out)}",
        f"start={out['timestamp'].iloc[0].isoformat()}",
        f"end={out['timestamp'].iloc[-1].isoformat()}",
        f"output={dst}",
    )
    if (args.mode or "auto").strip().lower() in {"es_master", "auto"} and "es_master" in Path(args.source).name.lower():
        print(
            "Note: es_master mode uses a deterministic proxy policy (offline), "
            "not live Gemini API calls."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
