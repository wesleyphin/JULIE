"""Pull Kalshi KXINXU all-strikes snapshots for a date range via API.

Generalizes fetch_kalshi_today.py: accepts --start and --end (ISO dates)
and writes one parquet per calendar day using the same schema the post-
hoc blocker expects. Idempotent: skips days that already exist on disk
unless --overwrite.

Usage:
    python fetch_kalshi_range.py --start 2026-04-06 --end 2026-04-17
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from config import CONFIG
from services.kalshi_provider import KalshiProvider
from fetch_kxinxu_2025 import parse_market_row, extract_markets, HOURLY_TICKER_RE

SERIES = "KXINXU"
OUT_DIR = ROOT / "data" / "kalshi" / "kxinxu_2025_daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_provider() -> KalshiProvider:
    cfg = dict(CONFIG.get("KALSHI", {}) or {})
    p = KalshiProvider(cfg)
    if not p.enabled:
        raise SystemExit("Kalshi provider disabled — missing credentials")
    return p


def enumerate_events_for_date(provider: KalshiProvider, date_str: str,
                              statuses=("active", "settled")) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for status in statuses:
        cursor: Optional[str] = None
        while True:
            params: Dict[str, Any] = {"series_ticker": SERIES, "limit": 200, "status": status}
            if cursor:
                params["cursor"] = cursor
            resp = provider._get("/events", params)
            if not isinstance(resp, dict):
                break
            batch = resp.get("events", []) or []
            if not batch:
                break
            for evt in batch:
                ticker = str(evt.get("event_ticker", "") or "")
                sd = str(evt.get("strike_date", "") or "")
                if not HOURLY_TICKER_RE.match(ticker):
                    continue
                if not sd.startswith(date_str):
                    continue
                events.append(evt)
            cursor = resp.get("cursor") or None
            if not cursor:
                break
    # Dedupe (active/settled overlap)
    seen = set()
    out = []
    for e in events:
        t = e.get("event_ticker")
        if t in seen:
            continue
        seen.add(t)
        out.append(e)
    out.sort(key=lambda e: str(e.get("strike_date", "") or ""))
    return out


def fetch_event_rows(provider: KalshiProvider, evt: Dict[str, Any]) -> List[Dict[str, Any]]:
    ticker = str(evt.get("event_ticker", ""))
    m = HOURLY_TICKER_RE.match(ticker)
    hour_et = int(int(m.group(1)) / 100) if m else 0
    strike_date = str(evt.get("strike_date", "") or "")
    event_date = strike_date[:10] if strike_date else ""
    resp = provider._get(f"/events/{ticker}", {"with_nested_markets": "true"})
    markets = extract_markets(resp)
    rows: List[Dict[str, Any]] = []
    for mk in markets:
        parsed = parse_market_row(ticker, event_date, hour_et, mk)
        if parsed is not None:
            rows.append(parsed)
    return rows


def to_cents(v):
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if f != f:
        return None
    return int(round(max(0.0, min(1.0, f)) * 100))


def pick_snapshot_cents(row):
    status = str(row.get("status", "") or "").lower()
    if status in {"finalized", "settled", "closed"}:
        c = to_cents(row.get("last_price"))
        if c is not None:
            return c
    c = to_cents(row.get("yes_mid"))
    if c is not None:
        return c
    yb, ya = to_cents(row.get("yes_bid")), to_cents(row.get("yes_ask"))
    if yb is not None and ya is not None:
        if (yb == 0 and ya == 100):
            return None
        return int(round((yb + ya) / 2))
    return yb if yb is not None else (ya if ya is not None else None)


def pull_day(p: KalshiProvider, target_date: str, overwrite: bool = False) -> int:
    out_path = OUT_DIR / f"{target_date}.parquet"
    if out_path.exists() and not overwrite:
        existing = pd.read_parquet(out_path)
        return len(existing)

    events = enumerate_events_for_date(p, target_date)
    if not events:
        return 0

    all_rows: List[Dict[str, Any]] = []
    for evt in events:
        all_rows.extend(fetch_event_rows(p, evt))
    if not all_rows:
        return 0

    df = pd.DataFrame(all_rows)
    df["snapshot_cents"] = df.apply(pick_snapshot_cents, axis=1)
    df["high"] = df["snapshot_cents"]
    df["low"] = df["snapshot_cents"]
    df["snapshot_date"] = target_date
    df["daily_volume"] = df.get("volume_24h").fillna(0).astype("Int64") if "volume_24h" in df else 0
    df["block_volume"] = 0
    df = df[["snapshot_date","event_ticker","event_date","settlement_hour_et",
             "market_ticker","strike","high","low","open_interest","daily_volume",
             "block_volume","status"]]
    df.to_parquet(out_path)
    return len(df)


def daterange(start: str, end: str):
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    while d0 <= d1:
        yield d0.isoformat()
        d0 += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    p = build_provider()
    t0 = time.time()
    days_done = 0
    rows_total = 0
    for d in daterange(args.start, args.end):
        # skip weekends (no KXINXU events)
        wd = date.fromisoformat(d).weekday()
        if wd >= 5:
            continue
        n = pull_day(p, d, overwrite=args.overwrite)
        status = "wrote" if n else "no data"
        print(f"  {d}  {status:<10}  {n:>5} rows")
        days_done += 1
        rows_total += n
    print(f"\nDone: {days_done} days, {rows_total} rows, {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
