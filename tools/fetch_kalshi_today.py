"""Pull today's Kalshi KXINXU all-strikes snapshots for 2026-04-21.

Saves in the same format as fetch_kxinxu_2025.py →
  data/kalshi/kxinxu_2025_daily/2026-04-21.parquet
so the existing post-hoc blocker picks it up unchanged.

Trade off: live events today are still "active" (not "settled" until
their settlement hour passes), so we query both states and keep the
latest snapshot per market ticker.
"""
from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from config import CONFIG
from services.kalshi_provider import KalshiProvider
# Re-use parser from the 2025 fetcher to stay in sync
from fetch_kxinxu_2025 import parse_market_row, extract_markets, HOURLY_TICKER_RE

SERIES = "KXINXU"
OUT_DIR = ROOT / "data" / "kalshi" / "kxinxu_2025_daily"  # keep existing path
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DATE = "2026-04-21"


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
    events.sort(key=lambda e: str(e.get("strike_date", "") or ""))
    # Dedupe on ticker (active/settled may overlap)
    seen = set()
    dedup = []
    for e in events:
        t = e.get("event_ticker")
        if t in seen:
            continue
        seen.add(t)
        dedup.append(e)
    return dedup


def fetch_event_rows(provider: KalshiProvider, evt: Dict[str, Any]) -> List[Dict[str, Any]]:
    ticker = str(evt.get("event_ticker", ""))
    m = HOURLY_TICKER_RE.match(ticker)
    hour_et = int(int(m.group(1)) / 100) if m else 0
    strike_date = str(evt.get("strike_date", "") or "")
    event_date = strike_date[:10] if strike_date else ""
    resp = provider._get(f"/events/{ticker}", {"with_nested_markets": "true"})
    markets = extract_markets(resp)
    rows: List[Dict[str, Any]] = []
    for m_ in markets:
        parsed = parse_market_row(ticker, event_date, hour_et, m_)
        if parsed is not None:
            rows.append(parsed)
    return rows


def main():
    p = build_provider()
    print(f"[kalshi] pulling events for {TARGET_DATE}...")
    events = enumerate_events_for_date(p, TARGET_DATE)
    print(f"[kalshi] {len(events)} {SERIES} hourly events for {TARGET_DATE}")
    if not events:
        print("No events — nothing to write.")
        return

    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, evt in enumerate(events, 1):
        rows = fetch_event_rows(p, evt)
        all_rows.extend(rows)
        if i % 5 == 0 or i == len(events):
            print(f"  {i}/{len(events)}  ticker={evt.get('event_ticker')}  rows_so_far={len(all_rows)}  elapsed={time.time()-t0:.1f}s")

    # We also need HIGH and LOW columns to match the existing schema the
    # post-hoc blocker expects. For today's live data, we don't have daily
    # H/L yet — use last_price (or yes_mid) as proxy for both (yes that's
    # degenerate, but it reflects "current snapshot" which is the best we
    # have for a still-active day).
    df = pd.DataFrame(all_rows)
    # Back-fill high/low with cents equivalents from mid / last / bid-ask
    def to_cents(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            f = float(v)
        except Exception:
            return None
        # yes_mid/last_price are in dollars (0-1). Convert to cents int 0-99
        return int(round(max(0.0, min(1.0, f)) * 100))

    def pick_snapshot_cents(row):
        # For SETTLED markets, yes_bid=0 yes_ask=1 (wide nothing), so yes_mid
        # degenerates to 0.50 which is meaningless. Use last_price (which is
        # 0.0 if NO won, 0.99 if YES won) as the primary settlement signal.
        # Only fall back to yes_mid for markets that are actually active.
        status = str(row.get("status", "") or "").lower()
        if status in {"finalized", "settled", "closed"}:
            c = to_cents(row.get("last_price"))
            if c is not None:
                return c
        # Active market: use yes_mid (real bid-ask midpoint)
        c = to_cents(row.get("yes_mid"))
        if c is not None:
            return c
        # Fallback: yes_bid / yes_ask
        yb, ya = to_cents(row.get("yes_bid")), to_cents(row.get("yes_ask"))
        if yb is not None and ya is not None:
            # Avoid degenerate 0.5 when the spread is the full book (0 to 1)
            if (yb == 0 and ya == 100):
                return None
            return int(round((yb + ya) / 2))
        return yb if yb is not None else (ya if ya is not None else None)

    if not df.empty:
        df["snapshot_cents"] = df.apply(pick_snapshot_cents, axis=1)
        # Use snapshot as high/low (degenerate; marks "current state")
        df["high"] = df["snapshot_cents"]
        df["low"] = df["snapshot_cents"]
        # Add snapshot_date and align schema with the daily files
        df["snapshot_date"] = TARGET_DATE
        df["daily_volume"] = df.get("volume_24h").fillna(0).astype("Int64")
        df["block_volume"] = 0
        df = df[["snapshot_date","event_ticker","event_date","settlement_hour_et",
                 "market_ticker","strike","high","low","open_interest","daily_volume",
                 "block_volume","status"]]

    out_path = OUT_DIR / f"{TARGET_DATE}.parquet"
    df.to_parquet(out_path)
    print(f"\n[write] {out_path}  ({len(df)} rows, {df['event_ticker'].nunique()} events)")
    if not df.empty:
        print(f"  settlement hours: {sorted(df['settlement_hour_et'].unique().tolist())}")
        print(f"  strike range: {df['strike'].min():.1f} – {df['strike'].max():.1f}")
        print(f"  snapshot cents range: {df['high'].min()}-{df['high'].max()} (median {df['high'].median()})")


if __name__ == "__main__":
    main()
