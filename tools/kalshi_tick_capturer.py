#!/usr/bin/env python3
"""Kalshi KXINXU tick capturer — going-forward minute-resolution capture.

Polls /series/KXINXU/markets/{ticker}/candlesticks once per minute for
strikes near the current ES price for each active settlement-hour event.
Appends to a daily parquet file under data/kalshi/kxinxu_minute_ticks/.

Why this exists
---------------
The Kalshi REST API only retains ~1 day of historical tick / candlestick
data — confirmed via empirical probe of /markets/trades and
/series/.../candlesticks across 4+ historical dates. Phase 2 of the
Kalshi ML build was BLOCKED on this constraint: we cannot retroactively
fetch 2025–early-2026 tick history to train on.

This capturer fixes that going forward. Every minute it writes the
minute-bar OHLCV + open-interest snapshot for each near-the-money strike
across all active KXINXU settlement-hour events. After ~30–60 days of
accumulation, there will be enough data to train a Kalshi ML model with
real intra-hour probability dynamics as features (prob velocity,
volatility, OI change, etc.) — features that were structurally
unavailable in v2–v6 attempts.

Schema (per row)
----------------
    capture_ts            (UTC ISO8601, when this row was captured)
    bar_end_ts            (epoch seconds, end of the 1-min bar)
    event_ticker          (e.g. KXINXU-26APR24H1500)
    market_ticker         (e.g. KXINXU-26APR24H1500-T6124.9999)
    settlement_hour_et    (10..16)
    strike                (float, e.g. 6125.0)
    es_at_capture         (float — ES futures price at capture, optional)
    open_dollars          (float)
    high_dollars          (float)
    low_dollars           (float)
    close_dollars         (float)
    mean_dollars          (float)
    open_interest_fp      (float, OI at end of bar)
    yes_bid_dollars       (float, market snapshot)
    yes_ask_dollars       (float, market snapshot)

Operational
-----------
- Polls every 60 sec while at least one settlement hour is active
- Idle outside 09:55–16:30 ET
- Daily parquet rotation: data/kalshi/kxinxu_minute_ticks/YYYY-MM-DD.parquet
- One row per (event_ticker, market_ticker, bar_end_ts)
- Dedupe on (event_ticker, market_ticker, bar_end_ts) keep last
- Crash-resilient: writes append-only to parquet via row buffer + flush

Usage
-----
    python3 tools/kalshi_tick_capturer.py
        # default: pulls strikes within ±20 ES points around current price,
        # for every active settlement hour, every 60 sec.

    python3 tools/kalshi_tick_capturer.py --strike-window 30
        # pull strikes within ±30 points

    python3 tools/kalshi_tick_capturer.py --debug
        # verbose logging

Run as launchd service or under tmux/nohup. Safe to run alongside the
live bot — it uses its own API session and respects rate limits.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from zoneinfo import ZoneInfo

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))

from config import CONFIG
from services.kalshi_provider import KalshiProvider

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
SETTLEMENT_HOURS_ET = [10, 11, 12, 13, 14, 15, 16]
SERIES = "KXINXU"
OUT_DIR = ROOT / "data/kalshi/kxinxu_minute_ticks"


def setup_logging(debug: bool) -> logging.Logger:
    log = logging.getLogger("kalshi_tick_capturer")
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(h)
    return log


def now_ny() -> datetime:
    return datetime.now(NY)


def es_current_price() -> Optional[float]:
    """Read latest ES price from live_prices.parquet if available."""
    p = ROOT / "ai_loop_data/live_prices.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        if df.empty: return None
        last = df["price"].dropna().iloc[-1]
        return float(last)
    except Exception:
        return None


def event_ticker_for_hour(now: datetime, hour_et: int) -> str:
    et = now.astimezone(NY)
    return f"{SERIES}-{et.strftime('%y%b%d').upper()}H{hour_et*100:04d}"


def active_hours(now: datetime) -> list[int]:
    """Hours whose settlement window has not yet closed."""
    et = now.astimezone(NY)
    return [h for h in SETTLEMENT_HOURS_ET if h > et.hour or
            (h == et.hour and et.minute < 5)]


def fetch_event_markets(provider: KalshiProvider,
                         event_ticker: str) -> list[dict]:
    """Get all 400 strikes for an event."""
    detail = provider._get(f"/events/{event_ticker}",
                            {"with_nested_markets": "true"})
    if not isinstance(detail, dict):
        return []
    return detail.get("event", {}).get("markets", []) or []


def select_atm_strikes(markets: list[dict], es_price: Optional[float],
                       window: float) -> list[dict]:
    """Select markets within ±window points of ES price.
    If ES price unknown, take 5 strikes around midpoint of available."""
    rows = []
    for m in markets:
        try:
            strike = float(m.get("floor_strike") or 0.0)
            if strike <= 0: continue
            rows.append((strike, m))
        except Exception:
            continue
    if not rows: return []
    rows.sort(key=lambda x: x[0])
    if es_price is None:
        # Take middle 5
        mid = len(rows) // 2
        return [m for _, m in rows[max(0, mid-2):mid+3]]
    return [m for s, m in rows if abs(s - es_price) <= window]


def fetch_recent_candles(provider: KalshiProvider, market_ticker: str,
                          end_ts_utc: int) -> list[dict]:
    """Pull last 5 minutes of 1-min candles for a market."""
    start_ts = end_ts_utc - 300
    r = provider._get(f"/series/{SERIES}/markets/{market_ticker}/candlesticks",
                       {"start_ts": start_ts, "end_ts": end_ts_utc + 60,
                        "period_interval": 1})
    if not isinstance(r, dict): return []
    return r.get("candlesticks", []) or []


def parquet_path_for(now: datetime) -> Path:
    et = now.astimezone(NY)
    return OUT_DIR / f"{et.strftime('%Y-%m-%d')}.parquet"


def append_rows(rows: list[dict], path: Path, log: logging.Logger):
    if not rows: return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists():
        try:
            old = pd.read_parquet(path)
            combined = pd.concat([old, new_df], ignore_index=True)
        except Exception as e:
            log.warning("parquet read failed (%s) — overwriting", e)
            combined = new_df
    else:
        combined = new_df
    # Dedupe: keep last per (event_ticker, market_ticker, bar_end_ts)
    combined = combined.drop_duplicates(
        subset=["event_ticker", "market_ticker", "bar_end_ts"], keep="last"
    )
    tmp = path.with_suffix(".parquet.tmp")
    combined.to_parquet(tmp, engine="pyarrow", compression="snappy")
    tmp.replace(path)
    log.info("flushed %d new rows -> %s (%d total today)",
             len(rows), path.name, len(combined))


def capture_iteration(provider: KalshiProvider, args, log: logging.Logger) -> int:
    n = now_ny()
    hours = active_hours(n)
    if not hours:
        log.debug("no active hours — idle")
        return 0
    es_px = es_current_price()
    log.debug("active hours: %s  ES≈%s", hours, es_px)

    end_ts_utc = int(n.astimezone(UTC).timestamp())
    rows = []
    for h in hours:
        ticker = event_ticker_for_hour(n, h)
        markets = fetch_event_markets(provider, ticker)
        if not markets:
            log.debug("  %s: no markets", ticker)
            continue
        picks = select_atm_strikes(markets, es_px, args.strike_window)
        log.debug("  %s: %d markets total, picking %d ATM (±%.0f)",
                  ticker, len(markets), len(picks), args.strike_window)
        for m in picks:
            mt = m.get("ticker") or m.get("market_ticker")
            if not mt: continue
            candles = fetch_recent_candles(provider, mt, end_ts_utc)
            for c in candles:
                price = c.get("price", {}) or {}
                rows.append({
                    "capture_ts": n.astimezone(UTC).isoformat(),
                    "bar_end_ts": int(c.get("end_period_ts", 0)),
                    "event_ticker": ticker,
                    "market_ticker": mt,
                    "settlement_hour_et": h,
                    "strike": float(m.get("floor_strike") or 0.0),
                    "es_at_capture": es_px,
                    "open_dollars": _f(price.get("open_dollars")),
                    "high_dollars": _f(price.get("high_dollars")),
                    "low_dollars":  _f(price.get("low_dollars")),
                    "close_dollars": _f(price.get("close_dollars")),
                    "mean_dollars": _f(price.get("mean_dollars")),
                    "open_interest_fp": _f(c.get("open_interest_fp")),
                    "yes_bid_dollars": _f(m.get("yes_bid_dollars")),
                    "yes_ask_dollars": _f(m.get("yes_ask_dollars")),
                })
    if rows:
        append_rows(rows, parquet_path_for(n), log)
    return len(rows)


def _f(x):
    try: return float(x) if x is not None else None
    except Exception: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strike-window", type=float, default=20.0,
                    help="±points around ES price for strike selection")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="Seconds between polls (default 60)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="Run a single capture iteration and exit")
    args = ap.parse_args()

    log = setup_logging(args.debug)
    log.info("kalshi_tick_capturer starting  out_dir=%s  strike_window=±%.0f  poll=%ds",
             OUT_DIR, args.strike_window, args.poll_interval)

    provider = KalshiProvider(dict(CONFIG.get("KALSHI", {}) or {}))
    if not provider.enabled:
        log.error("Kalshi provider disabled (missing creds?) — exiting")
        return 1

    if args.once:
        n = capture_iteration(provider, args, log)
        log.info("[once] captured %d rows", n)
        return 0

    while True:
        try:
            n = capture_iteration(provider, args, log)
            if n: log.info("captured %d rows", n)
        except KeyboardInterrupt:
            log.info("interrupted — exiting")
            return 0
        except Exception as exc:
            log.exception("capture iteration error: %s", exc)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
