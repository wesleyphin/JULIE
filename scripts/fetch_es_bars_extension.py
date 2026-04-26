#!/usr/bin/env python3
"""Pull ES (full-size) 1-min OHLCV bars from ProjectX for a date range.

Writes a parquet file matching the schema of es_master_outrights.parquet:
  index: timestamp (datetime64[ns, US/Eastern])
  columns: open (f64), high (f64), low (f64), close (f64), volume (i64), symbol (str)

Independent of ProjectXClient so it doesn't conflict with a running bot session.
Only makes REST calls (no websocket subscription).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import requests
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_secrets import SECRETS  # noqa

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
BASE_URL = "https://api.topstepx.com"


def login() -> str:
    url = f"{BASE_URL}/api/Auth/loginKey"
    resp = requests.post(url, json={
        "userName": SECRETS["USERNAME"],
        "apiKey": SECRETS["API_KEY"],
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errorCode") not in (0, None):
        raise RuntimeError(f"login error: {data}")
    token = data.get("token")
    if not token:
        raise RuntimeError(f"no token in login response: {data}")
    return token


def search_contract(token: str, search_text: str, target_suffix: str) -> str:
    """Find contract id matching .{target_suffix} (e.g. 'ES.M26')."""
    url = f"{BASE_URL}/api/Contract/search"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json={"live": False, "searchText": search_text},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    contracts = data.get("contracts", [])
    print(f"[search] searchText={search_text}  found {len(contracts)} contracts")
    for c in contracts:
        cid = c.get("id", "")
        name = c.get("name", "")
        print(f"  {cid}  ({name})")
        if cid.endswith(f".{target_suffix}"):
            return cid
    # Fallback: first with matching suffix
    for c in contracts:
        cid = c.get("id", "")
        if target_suffix.split(".")[-1] in cid:
            return cid
    raise RuntimeError(f"no contract matching .{target_suffix}")


def fetch_window(token: str, contract_id: str,
                 start_utc: dt.datetime, end_utc: dt.datetime) -> list[dict]:
    url = f"{BASE_URL}/api/History/retrieveBars"
    payload = {
        "contractId": contract_id,
        "live": False,
        "limit": 20000,
        "startTime": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unit": 2,   # minute
        "unitNumber": 1,
        "includePartialBar": False,
    }
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errorCode") not in (0, None):
        raise RuntimeError(f"retrieveBars error: {data}")
    return data.get("bars", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="ET start (YYYY-MM-DD HH:MM)")
    ap.add_argument("--end",   required=True, help="ET end   (YYYY-MM-DD HH:MM)")
    ap.add_argument("--contract-root", default="ES",
                    help="Search text sent to /Contract/search (e.g. ES, MES)")
    ap.add_argument("--contract-suffix", default="ES.M26",
                    help="Exact suffix to match on contract id (e.g. ES.M26, MES.M26)")
    ap.add_argument("--symbol-label", default="ESM6",
                    help="Value written to the 'symbol' column of the parquet")
    ap.add_argument("--out", required=True, help="Output parquet path")
    args = ap.parse_args()

    start_ny = dt.datetime.strptime(args.start, "%Y-%m-%d %H:%M").replace(tzinfo=NY)
    end_ny   = dt.datetime.strptime(args.end,   "%Y-%m-%d %H:%M").replace(tzinfo=NY)
    start_utc = start_ny.astimezone(UTC)
    end_utc   = end_ny.astimezone(UTC)
    print(f"[fetch] window ET {start_ny} -> {end_ny}")
    print(f"[fetch] window UTC {start_utc} -> {end_utc}")

    token = login()
    print("[fetch] authenticated")

    contract_id = search_contract(token, args.contract_root, args.contract_suffix)
    print(f"[fetch] contract_id = {contract_id}")

    # Pull in ≤14-day windows because API caps at 20k bars
    all_bars: list[dict] = []
    cursor = start_utc
    while cursor < end_utc:
        window_end = min(cursor + dt.timedelta(days=7), end_utc)
        print(f"[fetch]   window {cursor} -> {window_end}")
        bars = fetch_window(token, contract_id, cursor, window_end)
        print(f"[fetch]   got {len(bars)} bars")
        all_bars.extend(bars)
        if len(bars) < 20_000:
            cursor = window_end
        else:
            last_t = dt.datetime.fromisoformat(bars[0]["t"].replace("Z", "+00:00"))
            # API returns newest-first — last chronologically is bars[-1]
            last_t = dt.datetime.fromisoformat(bars[-1]["t"].replace("Z", "+00:00"))
            cursor = last_t + dt.timedelta(minutes=1)

    # Dedupe + sort by timestamp ascending
    seen: set[str] = set()
    uniq: list[dict] = []
    for b in all_bars:
        k = b["t"]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(b)
    uniq.sort(key=lambda x: x["t"])
    print(f"[fetch] total raw bars: {len(all_bars)}  unique: {len(uniq)}")

    if not uniq:
        print("[fetch] WARN: no bars returned — exiting with empty output")
        return 1

    rows = []
    for b in uniq:
        t_utc = dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        t_ny = t_utc.astimezone(NY)
        rows.append({
            "timestamp": t_ny,
            "open":   float(b["o"]),
            "high":   float(b["h"]),
            "low":    float(b["l"]),
            "close":  float(b["c"]),
            "volume": int(b.get("v", 0) or 0),
            "symbol": args.symbol_label,
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    # Match master parquet dtypes exactly
    df["open"]   = df["open"].astype("float64")
    df["high"]   = df["high"].astype("float64")
    df["low"]    = df["low"].astype("float64")
    df["close"]  = df["close"].astype("float64")
    df["volume"] = df["volume"].astype("int64")
    df["symbol"] = df["symbol"].astype("string")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, engine="pyarrow", compression="snappy")
    print(f"[write] {out}  ({out.stat().st_size:,} bytes)  rows={len(df):,}")
    print(f"[write] range: {df.index.min()} -> {df.index.max()}")
    print(f"[write] symbol: {df['symbol'].unique().tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
