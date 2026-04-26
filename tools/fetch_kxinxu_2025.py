#!/usr/bin/env python3
"""Fetch Kalshi KXINXU (hourly S&P 500) all-strikes snapshots for 2025.

Enumerates real settled events via /events (paginated) rather than guessing
ticker strings, filters to 2025 hourly tickers (ending in H\\d{4}), then
fetches nested markets per event and writes one row per strike.

Output:
    data/kalshi/kxinxu_2025_daily/<YYYY-MM-DD>.parquet (per calendar day)
    data/kalshi/kxinxu_hourly_2025.parquet             (final concatenated)
Resume-safe via data/kalshi/kxinxu_hourly_2025.checkpoint.json.
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from services.kalshi_provider import KalshiProvider  # noqa: E402


SERIES = "KXINXU"
OUT_DIR = ROOT / "data" / "kalshi"
PARTIAL_DIR = OUT_DIR / "kxinxu_2025_daily"
OUTPUT_PATH = OUT_DIR / "kxinxu_hourly_2025.parquet"
CHECKPOINT_PATH = OUT_DIR / "kxinxu_hourly_2025.checkpoint.json"

HOURLY_TICKER_RE = re.compile(r"^KXINXU-\d{2}[A-Z]{3}\d{2}H(\d{4})$")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_market_row(evt_ticker: str, event_date: str, hour_et: int, market: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ticker = str(market.get("ticker", "") or "")
    if "-T" not in ticker:
        return None
    try:
        strike = float(ticker.split("-T")[-1])
    except ValueError:
        return None

    yes_bid = _as_float(market.get("yes_bid_dollars"))
    if yes_bid is None:
        yes_bid = _as_float(market.get("yes_bid"))
    yes_ask = _as_float(market.get("yes_ask_dollars"))
    if yes_ask is None:
        yes_ask = _as_float(market.get("yes_ask"))
    yes_mid = (yes_bid + yes_ask) / 2.0 if (yes_bid is not None and yes_ask is not None) else None

    last_price = _as_float(market.get("last_price_dollars"))
    if last_price is None:
        last_price = _as_float(market.get("last_price"))

    return {
        "event_ticker": evt_ticker,
        "event_date": event_date,
        "settlement_hour_et": int(hour_et),
        "market_ticker": ticker,
        "strike": strike,
        "status": str(market.get("status", "") or ""),
        "result": str(market.get("result", "") or ""),
        "open_time": str(market.get("open_time", "") or ""),
        "close_time": str(market.get("close_time", "") or ""),
        "expiration_time": str(market.get("expiration_time", "") or ""),
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_mid": yes_mid,
        "last_price": last_price,
        "volume": _as_int(market.get("volume")),
        "volume_24h": _as_int(market.get("volume_24h")),
        "open_interest": _as_int(market.get("open_interest")),
        "previous_yes_bid": _as_float(market.get("previous_yes_bid")),
        "previous_yes_ask": _as_float(market.get("previous_yes_ask")),
        "previous_price": _as_float(market.get("previous_price")),
        "liquidity": _as_float(market.get("liquidity")),
    }


def extract_markets(response: Any) -> List[Dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    evt = response.get("event")
    if isinstance(evt, dict):
        markets = evt.get("markets") or []
        if markets:
            return list(markets)
    return list(response.get("markets") or [])


def load_checkpoint() -> Dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {"completed_events": []}


def save_checkpoint(state: Dict[str, Any]) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(state, indent=2))


def build_provider() -> KalshiProvider:
    cfg = dict(CONFIG.get("KALSHI", {}) or {})
    provider = KalshiProvider(cfg)
    if not provider.enabled:
        print("ERROR: Kalshi provider is disabled — check KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH.", file=sys.stderr)
        sys.exit(1)
    return provider


def enumerate_2025_hourly_events(provider: KalshiProvider) -> List[Dict[str, Any]]:
    """Page through settled KXINXU events, keep hourly tickers in 2025."""
    events: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    page = 0
    while True:
        params: Dict[str, Any] = {"series_ticker": SERIES, "limit": 200, "status": "settled"}
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
            strike_date = str(evt.get("strike_date", "") or "")
            if not HOURLY_TICKER_RE.match(ticker):
                continue
            if not strike_date.startswith("2025-"):
                continue
            events.append(evt)
        cursor = resp.get("cursor") or None
        page += 1
        if not cursor:
            break
    # Sort chronologically
    events.sort(key=lambda e: str(e.get("strike_date", "") or ""))
    return events


def fetch_event_rows(provider: KalshiProvider, evt: Dict[str, Any]) -> List[Dict[str, Any]]:
    ticker = str(evt.get("event_ticker", ""))
    match = HOURLY_TICKER_RE.match(ticker)
    hour_et = int(int(match.group(1)) / 100) if match else 0
    strike_date = str(evt.get("strike_date", "") or "")
    event_date = strike_date[:10] if strike_date else ""
    resp = provider._get(f"/events/{ticker}", {"with_nested_markets": "true"})
    markets = extract_markets(resp)
    rows: List[Dict[str, Any]] = []
    for m in markets:
        parsed = parse_market_row(ticker, event_date, hour_et, m)
        if parsed is not None:
            rows.append(parsed)
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

    provider = build_provider()
    state = load_checkpoint()
    completed = set(state.get("completed_events") or [])

    print("Enumerating 2025 hourly KXINXU events…")
    events = enumerate_2025_hourly_events(provider)
    print(f"Found {len(events)} 2025 hourly events")

    # Group events by calendar date so we can write one parquet per day
    by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in events:
        sd = str(e.get("strike_date", "") or "")
        by_date[sd[:10]].append(e)

    start_wall = time.time()
    total_rows_new = 0
    events_done = 0
    total = len(events)

    for day, day_events in sorted(by_date.items()):
        day_rows: List[Dict[str, Any]] = []
        for evt in day_events:
            tkr = str(evt.get("event_ticker", ""))
            if tkr in completed:
                continue
            rows = fetch_event_rows(provider, evt)
            if rows:
                day_rows.extend(rows)
            completed.add(tkr)
            events_done += 1
            # Periodic checkpoint save so interrupts don't lose much work
            if events_done % 50 == 0:
                state["completed_events"] = sorted(completed)
                save_checkpoint(state)
                elapsed = time.time() - start_wall
                pct = 100.0 * events_done / max(total, 1)
                print(f"  progress: {events_done}/{total} events ({pct:.1f}%) in {elapsed:.0f}s")

        if day_rows:
            # Append to existing day-partial if already present
            path = PARTIAL_DIR / f"{day}.parquet"
            if path.exists():
                prev = pd.read_parquet(path)
                combined = pd.concat([prev, pd.DataFrame(day_rows)], ignore_index=True)
                combined = combined.drop_duplicates(subset=["market_ticker"], keep="last")
                combined.to_parquet(path)
            else:
                pd.DataFrame(day_rows).to_parquet(path)
            total_rows_new += len(day_rows)
            print(f"{day}: +{len(day_rows)} rows ({len(day_events)} events today)")

    state["completed_events"] = sorted(completed)
    save_checkpoint(state)

    elapsed = time.time() - start_wall
    print(f"\nFetch complete: {events_done} events newly fetched, {total_rows_new} rows in {elapsed:.0f}s")

    parts = sorted(PARTIAL_DIR.glob("*.parquet"))
    if not parts:
        print("No partial parquet files; nothing to concatenate.")
        return
    frames = [pd.read_parquet(p) for p in parts]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} — {len(combined):,} rows, {combined['event_ticker'].nunique()} events, strikes {combined['strike'].min():.0f}-{combined['strike'].max():.0f}")


if __name__ == "__main__":
    main()
