#!/usr/bin/env python3
"""Synthesize Kalshi historical cache JSONs for 2025 from ES bars + KXINXU parquet.

Uses a binary-option approximation to model intra-hour YES probability per strike:

    prob(S_T > K) ≈ Φ(ln(ES/K) / (σ · √(T-t)))

where σ is EWMA realized vol from the prior 60 ES 1-min log returns. Output JSONs
mirror the shape of the real cache (as fetched by
`tools/run_full_live_replay.py::_fetch_kalshi_historical_window`) so the replay
driver can transparently load synthetic data from disk cache.

Caveats (model — NOT true ticks):
- Uses ES front-month as SPX proxy (ignores ~5-10 point basis)
- Constant drift=0 inside the hour (reasonable — risk-free negligible at 1h)
- No bid/ask spread (the overlay reads a single probability per bar)
- Clamps to [low_observed, high_observed] cents envelope from the KXINXU
  parquet when available to prevent extreme vol misestimates
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "backtest_reports" / "kalshi_historical_cache"
ES_PARQUET = Path("/Users/wes/Downloads/es_master_outrights.parquet")
KXINXU_PARQUET = ROOT / "data" / "kalshi" / "kxinxu_hourly_2025.parquet"
NY_TZ = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

# Match existing HistoricalKalshiProvider conventions
SETTLEMENT_HOURS_ET = (10, 11, 12, 13, 14, 15, 16)
STRIKE_BAND_SPX = 100.0  # keep strikes within ±100 of ES at open
MIN_VOL_FLOOR = 1e-4  # per-minute sigma floor


def _event_ticker(event_date: dt.date, hour_et: int) -> str:
    return f"KXINXU-{event_date.strftime('%y%b%d').upper()}H{hour_et * 100:04d}"


def _load_es_2025() -> pd.DataFrame:
    """Load ES 2025 1-min bars, front-month stitched + forward-filled.

    The master_outrights parquet interleaves ESH5/ESM5/ESU5/ESZ5/ESH6 by
    timestamp. Without picking a single symbol per calendar day we get a
    50pt zigzag at roll edges that corrupts EWMA vol estimates.
    """
    df = pd.read_parquet(ES_PARQUET)
    start = pd.Timestamp("2024-12-31 22:00", tz="US/Eastern")
    end = pd.Timestamp("2026-01-01 00:00", tz="US/Eastern")
    df = df.loc[(df.index >= start) & (df.index < end)].copy().sort_index()
    if "symbol" in df.columns and df["symbol"].nunique() > 1:
        tmp = df.copy()
        tmp["day"] = tmp.index.tz_convert("US/Eastern").date
        daily_vol = tmp.groupby(["day", "symbol"])["volume"].sum().reset_index()
        idx = daily_vol.groupby("day")["volume"].idxmax()
        best = daily_vol.loc[idx, ["day", "symbol"]]
        day_to_sym = dict(zip(best["day"], best["symbol"]))
        tmp["front"] = tmp["day"].map(day_to_sym)
        df = tmp.loc[tmp["symbol"] == tmp["front"], ["open", "high", "low", "close", "volume"]]
    if df.index.has_duplicates:
        df = df.loc[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    # Forward-fill to continuous 1-min grid
    if len(df) >= 2:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz=df.index.tz)
        was_missing = ~full_idx.isin(df.index)
        df = df.reindex(full_idx)
        df["volume"] = df["volume"].fillna(0.0)
        df["close"] = df["close"].ffill()
        df["open"] = df["open"].where(~was_missing, df["close"])
        df["high"] = df["high"].where(~was_missing, df["close"])
        df["low"] = df["low"].where(~was_missing, df["close"])
        df = df.dropna(subset=["close"])
    return df


def _load_kxinxu_events() -> pd.DataFrame:
    """Return one row per (event_date, settlement_hour_et) with strike list."""
    df = pd.read_parquet(KXINXU_PARQUET)
    # One row per market (drop snapshot-day duplicates)
    df = df.drop_duplicates(subset=["market_ticker"], keep="last")
    # Parse event_date into a date
    df["event_date_parsed"] = pd.to_datetime(df["event_date"]).dt.date
    grouped = (
        df.groupby(["event_date_parsed", "settlement_hour_et"])
          .agg(
              strikes=("strike", lambda s: sorted(set(float(x) for x in s))),
              event_ticker=("event_ticker", "first"),
              # Observed high/low cents envelope (max over snapshots, min over snapshots)
              cents_high=("high", "max"),
              cents_low=("low", "min"),
          )
          .reset_index()
    )
    return grouped


def _ewma_sigma_per_min(log_returns: np.ndarray, halflife: int = 60) -> float:
    if len(log_returns) < 3:
        return 1e-3
    alpha = 2.0 / (halflife + 1)
    ewma_var = float(np.var(log_returns[: min(5, len(log_returns))]))
    for r in log_returns[5:]:
        ewma_var = alpha * r * r + (1.0 - alpha) * ewma_var
    return max(math.sqrt(ewma_var), MIN_VOL_FLOOR)


def _build_event(
    event_date: dt.date,
    hour_et: int,
    strikes: List[float],
    es_df: pd.DataFrame,
    cents_envelope: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, Any]]:
    # UTC bounds
    open_et = dt.datetime.combine(event_date, dt.time(max(0, hour_et - 1), 0), tzinfo=NY_TZ)
    close_et = dt.datetime.combine(event_date, dt.time(hour_et, 0), tzinfo=NY_TZ)
    open_utc_ts = int(open_et.astimezone(UTC).timestamp())
    close_utc_ts = int(close_et.astimezone(UTC).timestamp())

    # ES bars during the event window (inclusive)
    event_mask = (es_df.index >= open_et) & (es_df.index <= close_et)
    event_bars = es_df.loc[event_mask, ["close"]]
    if len(event_bars) < 5:
        return None
    bar_ts = np.array(
        [int(t.tz_convert("UTC").timestamp()) for t in event_bars.index], dtype=np.int64
    )
    es_price = event_bars["close"].to_numpy(dtype=np.float64)

    # Vol from prior 60-min window
    vol_start_et = open_et - dt.timedelta(minutes=60)
    vol_mask = (es_df.index >= vol_start_et) & (es_df.index < open_et)
    vol_closes = es_df.loc[vol_mask, "close"].to_numpy(dtype=np.float64)
    if len(vol_closes) >= 3:
        log_rets = np.diff(np.log(vol_closes))
    else:
        log_rets = np.array([0.0])
    sigma = _ewma_sigma_per_min(log_rets)

    # Minutes-to-close (from each bar)
    seconds_to_close = (close_utc_ts - bar_ts).astype(np.float64)
    minutes_to_close = np.maximum(seconds_to_close / 60.0, 0.01)
    sigma_tau = sigma * np.sqrt(minutes_to_close)

    es_open = float(es_price[0])
    relevant_strikes = [k for k in strikes if abs(k - es_open) <= STRIKE_BAND_SPX]
    if not relevant_strikes:
        return None

    # Clamp bounds from observed cents envelope
    if cents_envelope is not None:
        lo_c, hi_c = cents_envelope
        # cents -> probability
        lo_p = max(0.005, float(lo_c) / 100.0) if lo_c is not None else 0.005
        hi_p = min(0.995, float(hi_c) / 100.0) if hi_c is not None else 0.995
    else:
        lo_p, hi_p = 0.005, 0.995

    strikes_out: Dict[str, List[List[Any]]] = {}
    for K in relevant_strikes:
        d = np.log(es_price / K) / sigma_tau
        prob = ndtr(d)
        prob = np.clip(prob, lo_p, hi_p)
        # Enforce terminal certainty at the final bar (t → 0 makes d → ±∞)
        if sigma_tau[-1] < 0.02:
            prob[-1] = 0.995 if es_price[-1] > K else 0.005
        series = [[int(bar_ts[i]), round(float(prob[i]), 4)] for i in range(len(bar_ts))]
        strikes_out[f"{K:.4f}"] = series

    return {
        "event_ticker": _event_ticker(event_date, hour_et),
        "open_utc_ts": open_utc_ts,
        "close_utc_ts": close_utc_ts,
        "settlement_hour_et": hour_et,
        "strikes": strikes_out,
    }


def _worker(
    payload: Tuple[dt.date, int, List[float], Optional[float], Optional[float], str]
) -> Tuple[str, bool, Optional[str]]:
    event_date, hour_et, strikes, cents_high, cents_low, es_path = payload
    try:
        # Each worker loads ES once per process; cached in worker state
        if not hasattr(_worker, "_es_df"):
            df = pd.read_parquet(es_path)
            start = pd.Timestamp("2024-12-31 22:00", tz="US/Eastern")
            end = pd.Timestamp("2026-01-01 00:00", tz="US/Eastern")
            df = df.loc[(df.index >= start) & (df.index < end)].copy().sort_index()
            # Front-month stitch by daily volume (avoids ESM5/ESU5 zigzag at rolls)
            if "symbol" in df.columns and df["symbol"].nunique() > 1:
                tmp = df.copy()
                tmp["day"] = tmp.index.tz_convert("US/Eastern").date
                daily_vol = tmp.groupby(["day", "symbol"])["volume"].sum().reset_index()
                idx = daily_vol.groupby("day")["volume"].idxmax()
                best = daily_vol.loc[idx, ["day", "symbol"]]
                day_to_sym = dict(zip(best["day"], best["symbol"]))
                tmp["front"] = tmp["day"].map(day_to_sym)
                df = tmp.loc[tmp["symbol"] == tmp["front"], ["open", "high", "low", "close", "volume"]]
            if df.index.has_duplicates:
                df = df.loc[~df.index.duplicated(keep="first")]
            df = df.sort_index()
            # Forward-fill to continuous 1-min grid
            if len(df) >= 2:
                full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz=df.index.tz)
                was_missing = ~full_idx.isin(df.index)
                df = df.reindex(full_idx)
                df["volume"] = df["volume"].fillna(0.0)
                df["close"] = df["close"].ffill()
                df["open"] = df["open"].where(~was_missing, df["close"])
                df["high"] = df["high"].where(~was_missing, df["close"])
                df["low"] = df["low"].where(~was_missing, df["close"])
                df = df.dropna(subset=["close"])
            _worker._es_df = df
        es_df: pd.DataFrame = _worker._es_df  # type: ignore[attr-defined]
        envelope = None
        if cents_high is not None or cents_low is not None:
            envelope = (cents_low, cents_high)
        evt = _build_event(event_date, hour_et, strikes, es_df, envelope)
        if evt is None:
            return _event_ticker(event_date, hour_et), False, "no_bars_or_strikes"
        out_path = CACHE_DIR / f"{evt['event_ticker']}.json"
        tmp_path = CACHE_DIR / f".{evt['event_ticker']}.json.tmp"
        tmp_path.write_text(json.dumps(evt, indent=2))
        os.replace(tmp_path, out_path)
        return evt["event_ticker"], True, None
    except Exception as exc:  # pragma: no cover
        return _event_ticker(event_date, hour_et), False, str(exc)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading KXINXU 2025 parquet…", flush=True)
    events_df = _load_kxinxu_events()
    print(f"  {len(events_df):,} events across {events_df['event_date_parsed'].nunique()} dates", flush=True)

    # Skip events whose cache file already exists unless FORCE=1
    force = os.environ.get("FORCE", "").strip() not in ("", "0", "false", "False")
    payloads: List[Tuple[dt.date, int, List[float], Optional[float], Optional[float], str]] = []
    for _, row in events_df.iterrows():
        ticker = _event_ticker(row["event_date_parsed"], int(row["settlement_hour_et"]))
        out_path = CACHE_DIR / f"{ticker}.json"
        if out_path.exists() and not force:
            continue
        payloads.append(
            (
                row["event_date_parsed"],
                int(row["settlement_hour_et"]),
                list(row["strikes"]),
                float(row["cents_high"]) if row["cents_high"] is not None else None,
                float(row["cents_low"]) if row["cents_low"] is not None else None,
                str(ES_PARQUET),
            )
        )
    print(f"  {len(payloads):,} events need synthesis (force={force})", flush=True)
    if not payloads:
        print("Nothing to do.")
        return

    workers = int(os.environ.get("SYNTH_WORKERS", "6"))
    done = 0
    failed = 0
    skipped_noise: List[str] = []
    t0 = pd.Timestamp.utcnow()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_worker, p): p[0] for p in payloads}
        for fut in as_completed(futures):
            ticker, ok, err = fut.result()
            done += 1
            if not ok:
                failed += 1
                if len(skipped_noise) < 8:
                    skipped_noise.append(f"{ticker}: {err}")
            if done % 100 == 0 or done == len(payloads):
                elapsed = (pd.Timestamp.utcnow() - t0).total_seconds()
                pct = 100.0 * done / len(payloads)
                print(
                    f"  [{done}/{len(payloads)}] {pct:4.1f}% failed={failed} elapsed={elapsed:.0f}s",
                    flush=True,
                )

    if skipped_noise:
        print("First skipped events:")
        for line in skipped_noise:
            print(f"  {line}")
    print(f"Done. {done - failed} synthesized, {failed} skipped.")


if __name__ == "__main__":
    main()
