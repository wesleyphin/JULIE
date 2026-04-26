#!/usr/bin/env python3
"""Pull Kalshi /historical/trades for all (event_ticker × ATM-strike) pairs
that align with KALSHI_ENTRY_VIEW events in our live + replay logs.

This is the data unlock that v2-v6 attempts didn't have. The endpoint
returns full tick-level history per market (back to 2023+) but only when
queried with the EXACT market ticker (event_ticker filter is silently
ignored — confirmed empirically).

Strategy
--------
1. Parse all KALSHI_ENTRY_VIEW events from logs to get
   (market_ts, entry_price) → derive (event_ticker, ATM strikes).
2. Group by event_ticker; for each, pull /historical/markets to see which
   strikes have any trades (volume_fp > 0). Cache.
3. For each market with volume > 0 within ±25 of any of our event
   entry_prices, pull /historical/trades paginated.
4. Write per-trade rows to a parquet file for downstream feature
   construction.

Output
------
data/kalshi/kxinxu_historical_trades.parquet
    columns: event_ticker, market_ticker, settlement_hour_et, strike,
             trade_id, created_time_utc, yes_price, no_price, count_fp,
             taker_side
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path("/Users/wes/Downloads/JULIE001")
sys.path.insert(0, str(ROOT))
from config import CONFIG
from services.kalshi_provider import KalshiProvider

OUT = ROOT / "data" / "kalshi" / "kxinxu_historical_trades.parquet"
EVENT_CACHE = ROOT / "data" / "kalshi" / "kxinxu_historical_markets_cache.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kalshi_historical_pull")


RE_HEADER = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_BAR = re.compile(r"Bar: (?P<mts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ET")
RE_ENTRY_VIEW = re.compile(
    r"\[KALSHI_ENTRY_VIEW\].*?strategy=(?P<strategy>\S+) \| side=(?P<side>LONG|SHORT) "
    r"\| entry_price=(?P<entry>[-\d.]+)"
)


def parse_kalshi_targets(log_paths: Iterable[Path]) -> list[dict]:
    """Yield dicts of {market_ts, entry_price} per minute-deduped Kalshi event."""
    rows = []
    for p in log_paths:
        last_bar_mts = None
        seen = set()
        with p.open(errors="ignore") as fh:
            for line in fh:
                m = RE_HEADER.match(line)
                if not m: continue
                bm = RE_BAR.search(line)
                if bm:
                    last_bar_mts = bm.group("mts")
                    continue
                ev = RE_ENTRY_VIEW.search(line)
                if not ev or last_bar_mts is None: continue
                strat = ev.group("strategy")
                if "aetherflow" in strat.lower(): continue
                key = (last_bar_mts[:16], strat, ev.group("side"))
                if key in seen: continue
                seen.add(key)
                rows.append({
                    "market_ts": last_bar_mts,
                    "entry_price": float(ev.group("entry")),
                })
    return rows


def event_ticker_for(market_ts: str) -> tuple[str, int]:
    """Map market_ts (string '2025-04-30 14:35:00') to its KXINXU event ticker
    (the next-hour settlement event in ET)."""
    et_dt = pd.Timestamp(market_ts).tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    if pd.isna(et_dt):
        return ("", 0)
    # Settlement hour = next hour (rolls to next hour at minute 5)
    if et_dt.minute < 5:
        settle = et_dt.replace(minute=0, second=0)
    else:
        settle = (et_dt + pd.Timedelta(hours=1)).replace(minute=0, second=0)
    h = int(settle.hour)
    if h < 10 or h > 16:
        return ("", 0)
    yymmdd = settle.strftime("%y%b%d").upper()
    return (f"KXINXU-{yymmdd}H{h*100:04d}", h)


def round_strike(price: float, step: int = 5) -> int:
    """Round ES price to nearest 5-point strike."""
    return int(round(price / step) * step)


def expected_market_ticker(event_ticker: str, strike: int) -> str:
    """Build a market ticker. Strikes are 'T<strike-1>.9999' i.e. above-this."""
    return f"{event_ticker}-T{strike - 1}.9999"


def get_event_tickers_and_strikes(targets: list[dict],
                                    strike_window: int = 25) -> dict[str, set[int]]:
    """Map event_ticker -> set of ATM strikes (ES integer floors) we want."""
    out: dict[str, set[int]] = {}
    skipped = 0
    for t in targets:
        et, hr = event_ticker_for(t["market_ts"])
        if not et:
            skipped += 1
            continue
        # ATM ± window in 5-point grid
        center = round_strike(t["entry_price"], 5)
        for offset in range(-strike_window, strike_window + 1, 5):
            out.setdefault(et, set()).add(center + offset)
    log.info("targets: %d events, skipped (off-hour): %d", len(targets), skipped)
    return out


def list_markets_for_event(provider: KalshiProvider, event_ticker: str) -> list[dict]:
    """Use /historical/markets paginated to find all markets that have
    historical data for this event."""
    out, cursor = [], None
    for _ in range(20):  # 20 pages × default = enough
        params = {"event_ticker": event_ticker, "limit": 1000}
        if cursor: params["cursor"] = cursor
        r = provider._get("/historical/markets", params)
        if not isinstance(r, dict): break
        ms = r.get("markets", [])
        if not ms: break
        # CRITICAL: /historical/markets lacks proper event_ticker filter too —
        # double-check by filtering returned markets ourselves
        ms_kept = [m for m in ms if str(m.get("event_ticker", "")) == event_ticker]
        out.extend(ms_kept)
        cursor = r.get("cursor")
        if not cursor: break
    return out


def fetch_trades_for_market(provider: KalshiProvider, market_ticker: str,
                              page_limit: int = 5) -> list[dict]:
    """Pull /historical/trades?ticker=<market> with pagination. Returns a list
    of trade dicts (each guaranteed to have ticker == market_ticker)."""
    out, cursor = [], None
    for _ in range(page_limit):
        params = {"ticker": market_ticker, "limit": 1000}
        if cursor: params["cursor"] = cursor
        r = provider._get("/historical/trades", params)
        if not isinstance(r, dict): break
        trades = r.get("trades", [])
        # Defensive filter — only keep trades that actually match our market
        trades = [t for t in trades if t.get("ticker") == market_ticker]
        out.extend(trades)
        cursor = r.get("cursor")
        if not cursor: break
        if len(trades) == 0: break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strike-window", type=int, default=25,
                    help="ES points around entry_price for ATM selection")
    ap.add_argument("--limit-events", type=int, default=0,
                    help="Cap event count for testing (0 = no cap)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=== loading log targets ===")
    log_paths = []
    rdir = ROOT / "backtest_reports/full_live_replay"
    for m in "2025_01 2025_02 2025_03 2025_04 2025_05 2025_06 2025_07 2025_08 2025_09 2025_10 2025_11 2025_12".split():
        p = rdir / m / "topstep_live_bot.log"
        if p.exists() and p.stat().st_size > 50_000:
            log_paths.append(p)
    live = ROOT / "topstep_live_bot.log"
    if live.exists(): log_paths.append(live)
    targets = parse_kalshi_targets(log_paths)
    log.info("Kalshi event targets: %d (AF excluded)", len(targets))

    et_strikes = get_event_tickers_and_strikes(targets, args.strike_window)
    log.info("unique event_tickers: %d  total (event,strike) pairs: %d",
             len(et_strikes),
             sum(len(s) for s in et_strikes.values()))

    # Drop pre-2025-02 events — hourly KXINXU contracts didn't exist before
    pre_feb_skipped = 0
    et_strikes_filtered = {}
    for et, strikes in et_strikes.items():
        # KXINXU-25FEB... or KXINXU-25MAR... onwards. KXINXU-25JAN... = no hourly.
        m = re.match(r"KXINXU-(\d{2})([A-Z]{3})(\d{2})H\d{4}$", et)
        if not m:
            pre_feb_skipped += 1; continue
        yy, mon, _ = m.groups()
        # 2024 and earlier — skip
        if yy < "25":
            pre_feb_skipped += 1; continue
        if yy == "25" and mon == "JAN":
            pre_feb_skipped += 1; continue
        et_strikes_filtered[et] = strikes
    log.info("pre-2025-02 skipped (hourly KXINXU didn't exist): %d", pre_feb_skipped)
    et_strikes = et_strikes_filtered

    if args.limit_events > 0:
        et_strikes = dict(list(sorted(et_strikes.items()))[:args.limit_events])
        log.info("limited to first %d events for testing", len(et_strikes))

    provider = KalshiProvider(dict(CONFIG.get("KALSHI", {}) or {}))

    # Phase A: For each event_ticker, list its actual markets via
    # /historical/markets (paginated). Filter to markets with volume_fp > 0
    # AND strike within our ATM window. This avoids querying nonexistent
    # ticker formats and skips zero-volume strikes.

    all_trades = []
    pulled_markets = 0
    skipped_no_event = 0
    skipped_no_volume = 0
    skipped_out_of_window = 0
    n_events = len(et_strikes)

    for i, (et, wanted_strikes) in enumerate(sorted(et_strikes.items())):
        if i % 20 == 0:
            log.info("[%d/%d] %s — wanted_strikes=%d  trades_so_far=%d",
                     i, n_events, et, len(wanted_strikes), len(all_trades))
        markets = list_markets_for_event(provider, et)
        if not markets:
            skipped_no_event += 1
            continue
        for m in markets:
            try:
                strike = int(m.get("floor_strike") or 0) + 1   # 5674.9999 → 5675
            except Exception:
                continue
            vol = float(m.get("volume_fp", 0) or 0)
            mt = m.get("ticker")
            if not mt: continue
            if vol <= 0:
                skipped_no_volume += 1
                continue
            # Only keep strikes near any of our entry_prices (already in
            # wanted_strikes set by 5-pt grid)
            if not any(abs(strike - ws) <= args.strike_window for ws in wanted_strikes):
                skipped_out_of_window += 1
                continue
            trades = fetch_trades_for_market(provider, mt, page_limit=5)
            if not trades:
                continue
            pulled_markets += 1
            for t in trades:
                all_trades.append({
                    "event_ticker": et,
                    "market_ticker": mt,
                    "strike": float(strike),
                    "trade_id": t.get("trade_id"),
                    "created_time_utc": t.get("created_time"),
                    "yes_price": float(t.get("yes_price_dollars", 0) or 0),
                    "no_price": float(t.get("no_price_dollars", 0) or 0),
                    "count_fp": float(t.get("count_fp", 0) or 0),
                    "taker_side": t.get("taker_side"),
                })
    log.info("phase A skips: no_event=%d no_volume=%d out_of_window=%d",
             skipped_no_event, skipped_no_volume, skipped_out_of_window)
    skipped_markets = skipped_no_event + skipped_no_volume + skipped_out_of_window  # back-compat

    log.info("done. pulled %d markets, skipped %d (no trades). total trades: %d",
             pulled_markets, skipped_markets, len(all_trades))
    if not all_trades:
        log.warning("no trades — exiting empty")
        return 1

    df = pd.DataFrame(all_trades)
    df["created_time_utc"] = pd.to_datetime(df["created_time_utc"])
    # Derive settlement hour from event_ticker
    df["settlement_hour_et"] = df["event_ticker"].str.extract(r"H(\d{4})$")[0].astype(int) // 100
    df = df.sort_values(["event_ticker", "market_ticker", "created_time_utc"])
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    log.info("[write] %s  rows=%d  size=%d bytes",
             out_path, len(df), out_path.stat().st_size)
    log.info("date range: %s -> %s",
             df["created_time_utc"].min(), df["created_time_utc"].max())
    log.info("unique markets pulled: %d  events covered: %d",
             df["market_ticker"].nunique(), df["event_ticker"].nunique())
    return 0


if __name__ == "__main__":
    sys.exit(main())
