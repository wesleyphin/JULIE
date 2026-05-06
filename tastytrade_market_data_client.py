"""TastyTrade market data client — real-time VIX (and other indices) via REST.

Drop-in replacement for YahooVIXClient. Same interface, no 15-min delay.

Why: Yahoo's free `^VIX` chart API serves with a ~15-min delay because Yahoo
is an aggregator buying delayed CBOE feeds for redistribution. TastyTrade is
a registered broker-dealer with direct CBOE entitlements; non-pro retail
customers get real-time index quotes via their REST market-data endpoint.

Caveats:
  - REST endpoint gives quote SNAPSHOTS, not minute-bar OHLCV. For 1-min
    bar history we synthesize bars from snapshots polled at ~60s cadence.
    Tracks last/bid/ask/timestamp per call.
  - First N seconds after startup we have only one bar; the bot's
    cross-market features tolerate this (NaN-fill).
  - For higher-frequency tick streaming (sub-second), TastyTrade exposes
    DXLink websocket via dxFeed. Out of scope for this drop-in client; the
    bot polls VIX once per main-loop tick (~60s) so REST snapshots match
    the existing cadence.

Drop-in compatibility with YahooVIXClient:
  - login() / fetch_contracts() / validate_session() — no-ops, mirror API
  - get_market_data(lookback_minutes=300, force_fetch=False) → DataFrame
    with columns ['open','high','low','close','volume'], DatetimeIndex (ET)
  - async_get_market_data(...) — async wrapper
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from config_secrets import SECRETS

API_BASE = SECRETS["TASTYTRADE_API_BASE"]


class TastyTradeIndexClient:
    """Real-time index quote client (VIX by default)."""

    _dependency_warned = False
    _bar_history: list[dict]  # [{ts, open, high, low, close, volume}, ...]

    def __init__(self, contract_root: str = "VIX", target_symbol: str = "VIX"):
        # TastyTrade index symbol convention: indexes are queried by their
        # ticker without the ^ prefix (e.g. VIX, SPX, NDX, RUT).
        self.contract_root = self._normalize_symbol(contract_root)
        self.target_symbol = self._normalize_symbol(target_symbol)
        self.account_id = "VIRTUAL_TT_ACC"
        self.contract_id = f"VIRTUAL_{self.target_symbol}_ID"
        self._bar_history = []
        self._current_bar_minute: Optional[pd.Timestamp] = None
        self._current_bar: Optional[dict] = None

    @staticmethod
    def _normalize_symbol(value) -> str:
        text = str(value or "").strip().lstrip("^").lstrip("$").upper()
        return text or "VIX"

    @classmethod
    def _warn_dependency_once(cls) -> None:
        if cls._dependency_warned:
            return
        cls._dependency_warned = True
        logging.warning(
            "TastyTradeIndexClient unavailable: requests/oauth not configured."
        )

    # ─── ProjectXClient/YahooVIXClient interface compatibility ───────────────

    def login(self) -> bool:
        try:
            from tastytrade_oauth import get_access_token
            tok = get_access_token()
            if tok:
                logging.info(
                    "TastyTradeIndexClient: OAuth token acquired (target=%s).",
                    self.target_symbol,
                )
                return True
        except Exception as exc:
            self._warn_dependency_once()
            logging.warning(
                "TastyTradeIndexClient: login failed (%s); continuing without live data.",
                exc,
            )
        return False

    def fetch_contracts(self):
        logging.info(
            f"TastyTradeIndexClient: contract '{self.target_symbol}' selected."
        )
        return self.contract_id

    def validate_session(self):
        # Token cache + lazy refresh in tastytrade_oauth handles this
        pass

    # ─── live quote → 1-min bar synthesis ────────────────────────────────────

    def _fetch_quote_snapshot(self) -> Optional[dict]:
        """REST call to TastyTrade /market-data for the index symbol.
        Returns dict with 'last', 'bid', 'ask', 'updated_at' or None on failure."""
        try:
            from tastytrade_oauth import auth_headers
            url = f"{API_BASE}/market-data/by-type"
            params = {"index": self.target_symbol}
            resp = requests.get(url, headers=auth_headers(), params=params, timeout=10)
            if resp.status_code != 200:
                logging.debug(
                    "TastyTrade quote fetch %s: HTTP %s %s",
                    self.target_symbol, resp.status_code, resp.text[:200],
                )
                return None
            data = resp.json()
            items = (data.get("data", {}).get("items")
                     or data.get("items")
                     or [])
            if not items:
                return None
            row = items[0] if isinstance(items, list) else items
            last = row.get("last") or row.get("close") or row.get("mark")
            bid = row.get("bid")
            ask = row.get("ask")
            updated = row.get("updated-at") or row.get("updated_at") or row.get("trade-time")
            try:
                last_f = float(last) if last is not None else None
            except (TypeError, ValueError):
                last_f = None
            if last_f is None or last_f <= 0:
                return None
            return {
                "last": last_f,
                "bid": float(bid) if bid not in (None, "") else last_f,
                "ask": float(ask) if ask not in (None, "") else last_f,
                "updated_at": updated,
            }
        except Exception as exc:
            logging.debug("TastyTrade quote fetch raised: %s", exc)
            return None

    def _ingest_quote_into_bars(self, snapshot: dict) -> None:
        """Bin the latest snapshot into a synthetic 1-min OHLC bar.
        On minute boundary, flush current bar to history and start new one."""
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        now_pd = pd.Timestamp(now)
        last = snapshot["last"]

        if self._current_bar_minute is None or now_pd != self._current_bar_minute:
            # Flush previous bar
            if self._current_bar is not None:
                self._bar_history.append(self._current_bar)
                # Cap history at 50k bars
                if len(self._bar_history) > 50_000:
                    self._bar_history = self._bar_history[-50_000:]
            self._current_bar_minute = now_pd
            self._current_bar = {
                "ts": now_pd,
                "open": last,
                "high": last,
                "low": last,
                "close": last,
                "volume": 0.0,
            }
        else:
            cur = self._current_bar
            cur["high"] = max(cur["high"], last)
            cur["low"] = min(cur["low"], last)
            cur["close"] = last

    def get_market_data(self, lookback_minutes: int = 300, force_fetch: bool = False) -> pd.DataFrame:
        """Fetch a fresh quote snapshot, fold into bar history, return tail of
        history matching YahooVIXClient's DataFrame shape."""
        snapshot = self._fetch_quote_snapshot()
        if snapshot is not None:
            self._ingest_quote_into_bars(snapshot)

        # Build DataFrame from bar history + current in-flight bar
        bars = list(self._bar_history)
        if self._current_bar is not None:
            bars.append(self._current_bar)
        if not bars:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(bars)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
        # Convert to ET to match YahooVIXClient's tz convention
        df.index = df.index.tz_convert("America/New_York")
        df = df[["open", "high", "low", "close", "volume"]]

        # Trim to lookback window
        if lookback_minutes > 0:
            cutoff = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(minutes=lookback_minutes)
            df = df[df.index >= cutoff]
        return df

    async def async_get_market_data(self, lookback_minutes: int = 300,
                                     force_fetch: bool = False) -> pd.DataFrame:
        """Async wrapper — wraps the sync REST call in a thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.get_market_data(lookback_minutes, force_fetch)
        )


# Convenience alias matching the YahooVIXClient name in julie001.py imports
TastyTradeVIXClient = TastyTradeIndexClient
