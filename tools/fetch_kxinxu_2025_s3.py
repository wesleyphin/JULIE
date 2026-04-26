#!/usr/bin/env python3
"""Fetch Kalshi KXINXU hourly S&P 500 all-strikes data for all of 2025.

Source: Kalshi public S3 archive at
  https://kalshi-public-docs.s3.amazonaws.com/reporting/market_data_YYYY-MM-DD.json

Each daily JSON contains one record per (market, date) across all Kalshi
series. We stream through each day, filter to KXINXU hourly strike markets
(ticker like `KXINXU-25JUN02H1000-T5895.9999`), and write one parquet
partial per snapshot day. Final concat lives at
  data/kalshi/kxinxu_hourly_2025.parquet

Resume-safe: already-written daily partials are skipped on rerun.

Schema written per row:
    snapshot_date          the date the record was published (may differ
                           from event_date when the market is still open)
    event_ticker           e.g. KXINXU-25JUN02H1000
    event_date             the expiration calendar date, parsed from ticker
    settlement_hour_et     hour of day the contract settles (10..16)
    market_ticker          full ticker including strike
    strike                 float strike (e.g. 5895.9999)
    high, low              cents 0-100 across the snapshot day
    open_interest
    daily_volume
    block_volume
    status                 finalized / settled / active / ...
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import ijson
import pandas as pd


WORKERS = int(os.environ.get("KXINXU_WORKERS", "6"))


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "kalshi"
PARTIAL_DIR = OUT_DIR / "kxinxu_2025_daily"
OUTPUT_PATH = OUT_DIR / "kxinxu_hourly_2025.parquet"

BUCKET_URL = "https://kalshi-public-docs.s3.amazonaws.com/"
PREFIX = "reporting/market_data_2025-"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# KXINXU hourly strike ticker: KXINXU-YYMMMDDHHHHH-T<strike>
HOURLY_RX = re.compile(
    r"^KXINXU-(\d{2})([A-Z]{3})(\d{2})H(\d{4})-T([\d.]+)$"
)


def list_2025_keys() -> List[str]:
    """Page through the bucket's 2025 reporting keys."""
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        params = {"list-type": "2", "prefix": PREFIX, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        url = BUCKET_URL + "?" + urlencode(params)
        with urlopen(url, timeout=60) as resp:
            body = resp.read()
        root = ET.fromstring(body)
        for c in root.findall("s3:Contents", S3_NS):
            key_el = c.find("s3:Key", S3_NS)
            if key_el is not None and key_el.text:
                keys.append(key_el.text)
        truncated = root.findtext("s3:IsTruncated", "false", S3_NS) == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", None, S3_NS)
        if not token:
            break
    return sorted(keys)


def download_day(key: str) -> List[Dict[str, Any]]:
    """Stream-parse the daily JSON array, filter to KXINXU hourly rows.

    Late-2025 files grow to 1-2 GB which breaks whole-file json.loads, so we
    stream records through ijson and keep only what we need.
    """
    url = BUCKET_URL + key
    snapshot_date = day_from_key(key)
    req = Request(url, headers={"Accept-Encoding": "identity"})
    rows: List[Dict[str, Any]] = []
    with urlopen(req, timeout=600) as resp:
        for rec in ijson.items(resp, "item"):
            ticker = rec.get("ticker_name") if isinstance(rec, dict) else None
            if not isinstance(ticker, str) or not ticker.startswith("KXINXU-"):
                continue
            parsed = parse_row(snapshot_date, rec)
            if parsed is not None:
                rows.append(parsed)
    return rows


def parse_row(snapshot_date: str, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ticker = str(rec.get("ticker_name", "") or "")
    m = HOURLY_RX.match(ticker)
    if not m:
        return None
    yy, mon, dd, hhmm, strike_str = m.groups()
    month = MONTH_MAP.get(mon)
    if month is None:
        return None
    try:
        event_date = f"20{yy}-{month:02d}-{int(dd):02d}"
        hour_et = int(hhmm) // 100
        strike = float(strike_str)
    except (ValueError, TypeError):
        return None
    event_ticker = ticker.split("-T", 1)[0]
    return {
        "snapshot_date": snapshot_date,
        "event_ticker": event_ticker,
        "event_date": event_date,
        "settlement_hour_et": hour_et,
        "market_ticker": ticker,
        "strike": strike,
        "high": _as_int(rec.get("high")),
        "low": _as_int(rec.get("low")),
        "open_interest": _as_int(rec.get("open_interest")),
        "daily_volume": _as_int(rec.get("daily_volume")),
        "block_volume": _as_int(rec.get("block_volume")),
        "status": str(rec.get("status", "") or ""),
    }


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def day_from_key(key: str) -> str:
    # reporting/market_data_2025-06-02.json -> 2025-06-02
    name = Path(key).stem  # market_data_2025-06-02
    return name.replace("market_data_", "")


def filter_day(records: Iterable[Dict[str, Any]], snapshot_date: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        parsed = parse_row(snapshot_date, rec)
        if parsed is not None:
            rows.append(parsed)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Listing 2025 keys from Kalshi public archive…")
    keys = list_2025_keys()
    print(f"Found {len(keys)} daily files in 2025")

    existing = {p.stem for p in PARTIAL_DIR.glob("*.parquet")}
    pending = [k for k in keys if day_from_key(k) not in existing]
    print(f"{len(existing)} days already on disk, {len(pending)} pending")

    def _process(key: str) -> tuple[str, int, Optional[str]]:
        day = day_from_key(key)
        try:
            rows = download_day(key)
        except Exception as exc:
            return day, 0, str(exc)
        if rows:
            df = pd.DataFrame(rows)
            # Write to tmp then rename for atomicity under concurrent execution
            out = PARTIAL_DIR / f"{day}.parquet"
            tmp = PARTIAL_DIR / f".{day}.parquet.tmp"
            df.to_parquet(tmp)
            os.replace(tmp, out)
        return day, len(rows), None

    start = time.time()
    total_rows = 0
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_process, k): k for k in pending}
        for fut in as_completed(futures):
            day, nrows, err = fut.result()
            done += 1
            elapsed = time.time() - start
            pct = 100.0 * done / max(len(pending), 1)
            if err is not None:
                print(f"  [{done}/{len(pending)}] {day}: FAILED ({err})", flush=True)
            else:
                total_rows += nrows
                print(f"  [{done}/{len(pending)}] {day}: +{nrows:,} rows  "
                      f"({pct:4.1f}%  {elapsed:.0f}s)", flush=True)

    print("Concatenating daily partials…")
    parts = sorted(PARTIAL_DIR.glob("*.parquet"))
    if not parts:
        print("No partials written; nothing to concat.")
        return
    frames = [pd.read_parquet(p) for p in parts]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(OUTPUT_PATH)
    events = combined["event_ticker"].nunique()
    strikes_min = combined["strike"].min()
    strikes_max = combined["strike"].max()
    snap_min = combined["snapshot_date"].min()
    snap_max = combined["snapshot_date"].max()
    print(
        f"Wrote {OUTPUT_PATH} — {len(combined):,} rows, "
        f"{events:,} events, strikes {strikes_min:.0f}–{strikes_max:.0f}, "
        f"snapshots {snap_min}..{snap_max}"
    )


if __name__ == "__main__":
    main()
