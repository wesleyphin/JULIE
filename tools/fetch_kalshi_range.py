"""Fetch Kalshi KXINXU hourly ladders over a date range into local daily parquet files.

This is a local, repo-safe replacement for the older upstream helper that was
hard-wired to Wesley's filesystem. It pulls one parquet per trade date using
the same compact schema the replay-time overlay evaluator expects.

Usage:
    .venv\\Scripts\\python.exe tools\\fetch_kalshi_range.py --start 2025-01-01 --end 2026-04-21
"""
from __future__ import annotations

import argparse
import calendar
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from services.kalshi_provider import KalshiProvider  # noqa: E402

SERIES = "KXINXU"
SETTLEMENT_HOURS_ET = [10, 11, 12, 13, 14, 15, 16]
OUT_DIR = ROOT / "data" / "kalshi" / "kxinxu_2025_daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EVENT_TICKER_RE = re.compile(rf"^{SERIES}-(?P<yy>\d{{2}})(?P<mon>[A-Z]{{3}})(?P<dd>\d{{2}})H(?P<hhmm>\d{{4}})$")

def build_provider() -> KalshiProvider:
    provider = KalshiProvider(dict(CONFIG.get("KALSHI", {}) or {}))
    if not provider.enabled:
        raise SystemExit("Kalshi provider disabled - missing or invalid credentials")
    return provider


def event_ticker_for(trade_date: date, settlement_hour_et: int) -> str:
    return f"{SERIES}-{trade_date.strftime('%y%b%d').upper()}H{int(settlement_hour_et) * 100}"


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return float(number)


def _to_cents(probability: Any) -> Optional[int]:
    value = _coerce_float(probability)
    if value is None:
        return None
    return int(round(max(0.0, min(1.0, value)) * 100))


def _pick_snapshot_cents(row: pd.Series) -> Optional[int]:
    status = str(row.get("status", "") or "").lower()
    if status in {"finalized", "settled", "closed"}:
        cents = _to_cents(row.get("last_price"))
        if cents is not None:
            return cents
    yes_mid = row.get("yes_mid")
    cents = _to_cents(yes_mid)
    if cents is not None:
        return cents
    yes_bid = _to_cents(row.get("yes_bid"))
    yes_ask = _to_cents(row.get("yes_ask"))
    if yes_bid is not None and yes_ask is not None:
        if yes_bid == 0 and yes_ask == 100:
            return None
        return int(round((yes_bid + yes_ask) / 2))
    return yes_bid if yes_bid is not None else yes_ask


def _format_strike(strike: float) -> str:
    text = f"{float(strike):.4f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _parse_event_ticker(event_ticker: str) -> Optional[tuple[date, int]]:
    match = EVENT_TICKER_RE.match(str(event_ticker or "").strip().upper())
    if not match:
        return None
    month_lookup = {calendar.month_abbr[i].upper(): i for i in range(1, 13)}
    month = month_lookup.get(match.group("mon"))
    if not month:
        return None
    trade_date = date(int("20" + match.group("yy")), int(month), int(match.group("dd")))
    hour = int(match.group("hhmm")) // 100
    return trade_date, int(hour)


def fetch_event_rows(
    provider: KalshiProvider,
    trade_date: date,
    settlement_hour_et: int,
) -> List[Dict[str, Any]]:
    ticker = event_ticker_for(trade_date, settlement_hour_et)
    parsed: List[Dict[str, Any]] = []
    for market in provider._fetch_single_event(ticker) or []:  # noqa: SLF001 - provider already owns the fallback logic
        strike = _coerce_float(market.get("strike"))
        probability = _coerce_float(market.get("probability"))
        if strike is None or probability is None:
            continue
        parsed.append(
            {
                "snapshot_date": trade_date.isoformat(),
                "event_ticker": ticker,
                "event_date": trade_date.isoformat(),
                "settlement_hour_et": int(settlement_hour_et),
                "market_ticker": f"{ticker}-T{_format_strike(strike)}",
                "strike": float(strike),
                "yes_bid": probability,
                "yes_ask": probability,
                "yes_mid": probability,
                "last_price": probability,
                "open_interest": None,
                "volume_24h": _coerce_float(market.get("volume")),
                "status": str(market.get("status", "") or ""),
            }
        )
    return parsed


def pull_day(provider: KalshiProvider, trade_date: date, overwrite: bool = False) -> int:
    out_path = OUT_DIR / f"{trade_date.isoformat()}.parquet"
    if out_path.exists() and not overwrite:
        existing = pd.read_parquet(out_path)
        return int(len(existing))

    all_rows: List[Dict[str, Any]] = []
    for hour in SETTLEMENT_HOURS_ET:
        all_rows.extend(fetch_event_rows(provider, trade_date, hour))
    if not all_rows:
        return 0

    df = pd.DataFrame(all_rows)
    df["snapshot_cents"] = df.apply(_pick_snapshot_cents, axis=1)
    df["high"] = df["snapshot_cents"]
    df["low"] = df["snapshot_cents"]
    df["daily_volume"] = pd.to_numeric(df.get("volume_24h"), errors="coerce").fillna(0).astype("Int64")
    df["block_volume"] = 0
    df = df[
        [
            "snapshot_date",
            "event_ticker",
            "event_date",
            "settlement_hour_et",
            "market_ticker",
            "strike",
            "high",
            "low",
            "open_interest",
            "daily_volume",
            "block_volume",
            "status",
        ]
    ].copy()
    df.to_parquet(out_path)
    return int(len(df))


def iter_dates(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _write_rows_for_day(date_str: str, rows: List[Dict[str, Any]], overwrite: bool = False) -> int:
    out_path = OUT_DIR / f"{date_str}.parquet"
    if out_path.exists() and not overwrite:
        existing = pd.read_parquet(out_path)
        return int(len(existing))
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    df["snapshot_cents"] = df.apply(_pick_snapshot_cents, axis=1)
    df["high"] = df["snapshot_cents"]
    df["low"] = df["snapshot_cents"]
    df["daily_volume"] = pd.to_numeric(df.get("volume_24h"), errors="coerce").fillna(0).astype("Int64")
    df["block_volume"] = 0
    df = df[
        [
            "snapshot_date",
            "event_ticker",
            "event_date",
            "settlement_hour_et",
            "market_ticker",
            "strike",
            "high",
            "low",
            "open_interest",
            "daily_volume",
            "block_volume",
            "status",
        ]
    ].copy()
    df = df.drop_duplicates(subset=["market_ticker"], keep="first").sort_values(
        ["event_ticker", "strike"], kind="stable"
    )
    df.to_parquet(out_path)
    return int(len(df))


def pull_range_bulk_settled(
    provider: KalshiProvider,
    start_date: date,
    end_date: date,
    overwrite: bool = False,
    limit: int = 1000,
) -> tuple[int, int]:
    rows_by_day: Dict[str, List[Dict[str, Any]]] = {}
    cutoff_resp = provider._get("/historical/cutoff", {})
    cutoff_raw = str((cutoff_resp or {}).get("market_settled_ts", "") or "")
    cutoff_date = date.fromisoformat(cutoff_raw[:10]) if cutoff_raw else None

    def scan_endpoint(endpoint: str, base_params: Dict[str, Any], segment_start: date, segment_end: date, label: str) -> None:
        cursor: Optional[str] = None
        pages = 0
        while True:
            params = dict(base_params)
            params["limit"] = int(limit)
            if cursor:
                params["cursor"] = cursor
            response = provider._get(endpoint, params)
            markets = response.get("markets", []) if isinstance(response, dict) else []
            if not markets:
                break
            pages += 1
            page_dates: List[date] = []
            for market in markets:
                if not isinstance(market, dict):
                    continue
                event_ticker = str(market.get("event_ticker", "") or "")
                parsed = _parse_event_ticker(event_ticker)
                if parsed is None:
                    continue
                trade_date, settlement_hour_et = parsed
                page_dates.append(trade_date)
                if trade_date < segment_start or trade_date > segment_end:
                    continue
                strike = _coerce_float(market.get("floor_strike"))
                probability = _coerce_float(market.get("last_price_dollars", market.get("last_price")))
                if strike is None or probability is None:
                    continue
                date_str = trade_date.isoformat()
                rows_by_day.setdefault(date_str, []).append(
                    {
                        "snapshot_date": date_str,
                        "event_ticker": event_ticker,
                        "event_date": date_str,
                        "settlement_hour_et": int(settlement_hour_et),
                        "market_ticker": str(market.get("ticker", f"{event_ticker}-T{_format_strike(strike)}")),
                        "strike": float(strike),
                        "yes_bid": probability,
                        "yes_ask": probability,
                        "yes_mid": probability,
                        "last_price": probability,
                        "open_interest": _coerce_float(market.get("open_interest_fp", market.get("open_interest"))),
                        "volume_24h": _coerce_float(market.get("volume_24h_fp", market.get("volume_24h", market.get("volume_fp")))),
                        "status": str(market.get("status", "") or ""),
                    }
                )
            if page_dates and max(page_dates) < segment_start:
                break
            cursor = response.get("cursor") if isinstance(response, dict) else None
            if not cursor:
                break
            if pages % 25 == 0:
                newest = max(page_dates).isoformat() if page_dates else "?"
                oldest = min(page_dates).isoformat() if page_dates else "?"
                print(f"[{label}] page {pages}  newest={newest}  oldest={oldest}  covered_days={len(rows_by_day)}")

    if cutoff_date is None:
        scan_endpoint("/historical/markets", {"series_ticker": SERIES}, start_date, end_date, "historical")
        scan_endpoint("/markets", {"series_ticker": SERIES, "status": "settled"}, start_date, end_date, "live")
    else:
        historical_end = min(end_date, cutoff_date - timedelta(days=1))
        live_start = max(start_date, cutoff_date)
        if start_date <= historical_end:
            scan_endpoint("/historical/markets", {"series_ticker": SERIES}, start_date, historical_end, "historical")
        if live_start <= end_date:
            scan_endpoint("/markets", {"series_ticker": SERIES, "status": "settled"}, live_start, end_date, "live")

    day_count = 0
    row_count = 0
    for trade_date in iter_dates(start_date, end_date):
        if trade_date.weekday() >= 5:
            continue
        date_str = trade_date.isoformat()
        rows = rows_by_day.get(date_str, [])
        written = _write_rows_for_day(date_str, rows, overwrite=overwrite)
        status = "wrote" if written else "no data"
        print(f"{date_str}  {status:<8}  {written:>5} rows")
        day_count += 1
        row_count += written
    return day_count, row_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--bulk-settled",
        action="store_true",
        help="Use paginated /markets settled archive feed instead of per-event requests",
    )
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    provider = build_provider()

    day_count = 0
    row_count = 0
    t0 = time.time()
    if bool(args.bulk_settled):
        day_count, row_count = pull_range_bulk_settled(
            provider,
            start_date,
            end_date,
            overwrite=bool(args.overwrite),
        )
    else:
        for trade_date in iter_dates(start_date, end_date):
            if trade_date.weekday() >= 5:
                continue
            rows = pull_day(provider, trade_date, overwrite=bool(args.overwrite))
            status = "wrote" if rows else "no data"
            print(f"{trade_date.isoformat()}  {status:<8}  {rows:>5} rows")
            day_count += 1
            row_count += rows

    print(f"\nDone: {day_count} weekdays checked, {row_count} rows, {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
