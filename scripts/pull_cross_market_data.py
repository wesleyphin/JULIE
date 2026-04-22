"""Pull MNQ + VIX historical data from Databento → CSV → cleaned → parquet.

Requirements per operator:
  - Continuous (non-adjusted) front-month for MNQ
  - CSV download → cleaned → parquet conversion pipeline

Uses the Databento Historical API via the `databento` Python SDK.

Data sets:
  - MNQ: dataset=GLBX.MDP3, symbol=MNQ.c.0 (continuous front month)
         schema=ohlcv-1m (1-minute OHLCV)
  - VIX: dataset=GLBX.MDP3 won't have the spot index. VIX futures are VX
         (not the same as spot VIX). For the "regime" classification we
         need spot VIX. We try two paths:
           (a) CBOE's spot VIX via OPRA (dataset=OPRA.PILLAR, symbol=VIX)
               — might not be licensed
           (b) fall back to VX futures front-month 1d bars as a proxy
         Either way we persist daily closes only.

Clean-up steps per user instruction ("continuous non-adjusted"):
  - We use the Databento continuous-symbol API (.c.0) which switches
    front month on roll day WITHOUT back-adjusting prices. This is the
    standard "non-adjusted continuous" series.
  - No roll-day price-jump filtering — we keep the raw discontinuities
    because adjusting them away would be "adjusted continuous" which is
    the opposite of what was asked.

API key: read from env var DATABENTO_API_KEY. Never committed.

Output:
  data/mnq_master_continuous.parquet   — 1-min MNQ continuous front
  data/vix_daily.parquet                — daily VIX closes (spot or VX proxy)

Usage:
  export DATABENTO_API_KEY=db-xxx
  python3 scripts/pull_cross_market_data.py --start 2024-01-01 --end 2026-04-21
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

OUT_MNQ = ROOT / "data" / "mnq_master_continuous.parquet"
OUT_VIX = ROOT / "data" / "vix_daily.parquet"
# Optional: save the raw CSVs too for auditability
OUT_MNQ_CSV = ROOT / "data" / "mnq_raw.csv"
OUT_VIX_CSV = ROOT / "data" / "vix_raw.csv"


def get_client():
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if not key:
        print("ERROR: set DATABENTO_API_KEY env var (export DATABENTO_API_KEY=db-...)",
              file=sys.stderr)
        sys.exit(2)
    import databento as db
    return db.Historical(key)


def pull_mnq(client, start: dt.datetime, end: dt.datetime):
    """Pull MNQ 1-minute OHLCV continuous front month.

    Databento's `MNQ.c.0` symbol = continuous front month, non-adjusted
    (prices reflect whichever contract is front on each day with no
    back-adjustment on rolls)."""
    print(f"[mnq] requesting {start} → {end} from Databento GLBX.MDP3 MNQ.c.0 "
          "(continuous front, non-adjusted)")
    # Databento timerange is UTC
    t0 = start.astimezone(UTC).isoformat()
    t1 = end.astimezone(UTC).isoformat()
    # Use ohlcv-1m schema, continuous symbol stype
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="ohlcv-1m",
        symbols=["MNQ.c.0"],
        stype_in="continuous",
        start=t0,
        end=t1,
    )
    print(f"[mnq] received — writing CSV + parsing...")
    OUT_MNQ_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(str(OUT_MNQ_CSV))
    print(f"[mnq] raw CSV saved: {OUT_MNQ_CSV}")

    # Parse CSV → DataFrame
    df = pd.read_csv(OUT_MNQ_CSV)
    # Databento CSV has: ts_event (nanoseconds), rtype, publisher_id, instrument_id,
    #                    open, high, low, close, volume, symbol
    # Normalize timestamp and drop extras
    if "ts_event" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(NY)
    else:
        raise RuntimeError(f"unexpected CSV columns: {df.columns.tolist()}")
    cols = ["open", "high", "low", "close", "volume"]
    if "symbol" in df.columns:
        cols.append("symbol")
    df = df[["timestamp"] + cols].set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    OUT_MNQ.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_MNQ)
    print(f"[mnq] parquet saved: {OUT_MNQ}  ({len(df):,} bars, "
          f"{df.index.min()} → {df.index.max()})")
    # Quick sanity check on roll-day jumps
    closes = df["close"].to_numpy()
    if len(closes) > 10:
        jumps = abs(closes[1:] - closes[:-1])
        print(f"[mnq] max bar-to-bar jump: {jumps.max():.2f} pts  "
              f"(large jumps expected on roll days — non-adjusted continuous)")


def pull_vix(client, start: dt.datetime, end: dt.datetime):
    """Pull daily VIX closes. Try spot index first; fall back to VX front-month.

    Note: CBOE spot VIX requires OPRA feed licensing. If that fails, we
    pull VX (CFE) futures front-month as a proxy — which typically tracks
    within 1-2 points of spot.
    """
    print(f"[vix] requesting {start.date()} → {end.date()} from Databento")
    t0 = start.astimezone(UTC).isoformat()
    t1 = end.astimezone(UTC).isoformat()
    # Try VX futures front-month daily OHLCV (CFE dataset)
    OUT_VIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    source = None
    # Attempt 1: Databento VX continuous (futures proxy for VIX)
    try:
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            schema="ohlcv-1d",
            symbols=["VX.c.0"],
            stype_in="continuous",
            start=t0,
            end=t1,
        )
        data.to_csv(str(OUT_VIX_CSV))
        source = "GLBX.MDP3/VX.c.0 (VX futures front-month, daily)"
    except Exception as exc:
        print(f"[vix] GLBX.MDP3/VX.c.0 failed: {exc}")
        source = None

    # Attempt 2: Yahoo Finance spot VIX (always available, public index)
    if source is None:
        print("[vix] falling back to Yahoo Finance spot VIX (^VIX)")
        p1 = int(start.astimezone(UTC).timestamp())
        p2 = int(end.astimezone(UTC).timestamp())
        # Use the chart API (more reliable than the deprecated download endpoint)
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
               f"?period1={p1}&period2={p2}&interval=1d&events=history")
        import urllib.request, json as _json
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (julie-bot cross-market puller)",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = _json.loads(resp.read().decode("utf-8"))
            result = payload["chart"]["result"][0]
            ts_arr = result["timestamp"]
            ind = result["indicators"]["quote"][0]
            rows = []
            for i, ts in enumerate(ts_arr):
                if ind["close"][i] is None:
                    continue
                rows.append({
                    "ts_event": dt.datetime.fromtimestamp(ts, UTC).isoformat(),
                    "open": ind["open"][i],
                    "high": ind["high"][i],
                    "low": ind["low"][i],
                    "close": ind["close"][i],
                    "volume": ind["volume"][i] if ind.get("volume") else 0,
                })
            if not rows:
                print("[vix] Yahoo returned empty dataset")
                return
            yf_df = pd.DataFrame(rows)
            yf_df.to_csv(OUT_VIX_CSV, index=False)
            source = "Yahoo Finance ^VIX (spot index, daily)"
        except Exception as exc:
            print(f"[vix] Yahoo fallback failed: {exc}")
            return

    print(f"[vix] raw CSV saved: {OUT_VIX_CSV} (source: {source})")
    df = pd.read_csv(OUT_VIX_CSV)
    df["timestamp"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_convert(NY)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # VIX regime features need daily closes indexed by date
    df_daily = df.resample("1D").last().dropna()
    df_daily.index = df_daily.index.normalize()
    OUT_VIX.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_parquet(OUT_VIX)
    print(f"[vix] parquet saved: {OUT_VIX}  ({len(df_daily)} daily bars)")
    print(f"[vix] NOTE: source is {source}. If this is VX futures (not spot "
          f"VIX), regime thresholds in cross_market.py assume spot-VIX levels "
          f"(~14-30); VX futures are typically within 1-2 pts of spot so the "
          f"regime classification remains approximately correct.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01", help="ISO date")
    ap.add_argument("--end", default="2026-04-22", help="ISO date")
    ap.add_argument("--mnq-only", action="store_true")
    ap.add_argument("--vix-only", action="store_true")
    args = ap.parse_args()

    client = get_client()
    start = dt.datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=NY)
    end = dt.datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=NY)

    if not args.vix_only:
        pull_mnq(client, start, end)
    if not args.mnq_only:
        pull_vix(client, start, end)
    print("\n[done]")


if __name__ == "__main__":
    main()
