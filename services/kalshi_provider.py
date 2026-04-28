import base64
import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytz
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


logger = logging.getLogger("kalshi_provider")


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on", "y"}:
            return True
        if raw in {"0", "false", "no", "off", "n"}:
            return False
    return default


class KalshiProvider:
    """
    Kalshi API client for KXINXU (S&P 500 above/below hourly contracts).

    Public read-only mode uses Kalshi's unauthenticated REST market-data endpoints
    and keeps the same probability-curve interface used by the live overlay.
    """

    def __init__(self, config: Dict):
        config = config or {}
        self.key_id = str(config.get("key_id", "") or "")
        self.base_url = str(config.get("base_url", "") or "").rstrip("/")
        self.series = str(config.get("series", "KXINXU") or "KXINXU")
        self.cache_ttl = int(config.get("cache_ttl", 120) or 120)
        self.rate_delay = float(config.get("rate_limit_delay", 0.4) or 0.4)
        self.timeout = int(config.get("request_timeout", 15) or 15)
        self.max_retries = int(config.get("max_retries", 3) or 3)
        self.basis_offset = float(config.get("basis_offset", 0.0) or 0.0)
        self.enabled = bool(config.get("enabled", True))
        self.public_read_only = _coerce_bool(config.get("public_read_only"), False)
        self.market_data_proxy_url = str(
            config.get("market_data_proxy_url")
            or os.environ.get("KALSHI_MARKET_DATA_PROXY_URL")
            or os.environ.get("MARKET_DATA_PROXY_URL")
            or ""
        ).strip()
        self.trust_env = _coerce_bool(config.get("trust_env"), False)
        self.bridge_cache_path = str(config.get("bridge_cache_path", "") or "").strip()
        self.bridge_cache_only = _coerce_bool(config.get("bridge_cache_only"), False)
        self.bridge_cache_max_age_seconds = float(
            config.get("bridge_cache_max_age_seconds", 30) or 30
        )

        configured_session = config.get("session")
        self.session = configured_session if configured_session is not None else requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = self.trust_env
        if self.market_data_proxy_url and hasattr(self.session, "proxies"):
            self.session.proxies.update(
                {"http": self.market_data_proxy_url, "https": self.market_data_proxy_url}
            )

        self.private_key = None
        private_key_path = str(config.get("private_key_path", "") or "")
        if private_key_path and not self.public_read_only:
            try:
                with open(private_key_path, "rb") as key_file:
                    self.private_key = serialization.load_pem_private_key(key_file.read(), password=None)
            except (FileNotFoundError, OSError, ValueError) as exc:
                logger.warning("Unable to load Kalshi private key from %s: %s", private_key_path, exc)

        if not self.public_read_only and (not self.key_id or self.private_key is None):
            if self.enabled:
                logger.warning("Kalshi credentials missing or invalid; provider disabled")
            self.enabled = False

        self._cache: Dict[str, Dict] = {}
        self._cache_lock = threading.Lock()
        self._last_request_ts = 0.0
        self._sentiment_history: List[tuple[float, float]] = []

        self.consecutive_failures = 0
        self.last_success: Optional[float] = None
        self.is_healthy = True
        self.last_resolved_ticker: Optional[str] = None

    def refresh(self) -> List[Dict]:
        """Refresh the active event cache from a background task."""
        if not self.enabled:
            return []
        return self._fetch_event_markets()

    async def async_refresh(self) -> List[Dict]:
        return await asyncio.to_thread(self.refresh)

    def es_to_spx(self, es_price: float) -> float:
        return float(es_price) - float(self.basis_offset)

    def spx_to_es(self, spx_price: float) -> float:
        return float(spx_price) + float(self.basis_offset)

    def _sign(self, method: str, path: str, timestamp_str: str) -> str:
        if self.private_key is None:
            raise RuntimeError("Kalshi private key is not loaded")
        message = f"{timestamp_str}{method}{path}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        elapsed = time.time() - self._last_request_ts
        if elapsed < self.rate_delay:
            time.sleep(self.rate_delay - elapsed)

        if not self.enabled:
            return {}

        request_kwargs: Dict[str, Any] = {"params": params, "timeout": self.timeout}
        if not self.public_read_only:
            if not self.key_id or self.private_key is None:
                logger.warning("Kalshi credentials missing; disabling provider calls")
                return {}

            ts_str = str(int(time.time()))
            sig = self._sign("GET", f"/trade-api/v2{path}", ts_str)
            request_kwargs["headers"] = {
                "KALSHI-ACCESS-KEY": self.key_id,
                "KALSHI-ACCESS-SIGNATURE": sig,
                "KALSHI-ACCESS-TIMESTAMP": ts_str,
            }
        url = f"{self.base_url}{path}"

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, **request_kwargs)
                self._last_request_ts = time.time()

                if response.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning("Kalshi rate-limited; retrying in %ss", wait)
                    time.sleep(wait)
                    continue
                if response.status_code in (400, 404):
                    return {}

                response.raise_for_status()
                self.consecutive_failures = 0
                self.is_healthy = True
                self.last_success = time.time()
                return response.json()
            except requests.RequestException as exc:
                self.consecutive_failures += 1
                if self.consecutive_failures >= 5:
                    self.is_healthy = False
                if attempt == self.max_retries - 1:
                    logger.error("Kalshi request failed (%s): %s", path, exc)
                    return {}
                time.sleep(2)
        return {}

    def _load_bridge_markets(self, event_ticker: Optional[str] = None) -> Optional[List[Dict]]:
        if not self.bridge_cache_path:
            return None
        try:
            with open(self.bridge_cache_path, "r", encoding="utf-8") as fh:
                snapshot = json.load(fh)
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Unable to read Kalshi bridge cache %s: %s", self.bridge_cache_path, exc)
            if self.bridge_cache_only:
                self.is_healthy = False
            return None
        if not isinstance(snapshot, dict):
            if self.bridge_cache_only:
                self.is_healthy = False
            return None

        snapshot_ts = float(snapshot.get("ts", 0.0) or 0.0)
        if self.bridge_cache_max_age_seconds > 0:
            age = time.time() - snapshot_ts
            if snapshot_ts <= 0 or age > self.bridge_cache_max_age_seconds:
                logger.warning("Kalshi bridge cache is stale: age=%.1fs", age)
                if self.bridge_cache_only:
                    self.is_healthy = False
                return None

        snapshot_ticker = str(snapshot.get("event_ticker", "") or "").strip()
        requested_ticker = str(event_ticker or "").strip()
        if requested_ticker and snapshot_ticker and requested_ticker != snapshot_ticker:
            return None

        parsed: List[Dict] = []
        for row in snapshot.get("markets", []) or []:
            if not isinstance(row, dict):
                continue
            try:
                strike = float(row.get("strike"))
                probability = float(row.get("probability"))
            except (TypeError, ValueError):
                continue
            try:
                volume = float(row.get("volume", 0.0) or 0.0)
            except (TypeError, ValueError):
                volume = 0.0
            parsed.append(
                {
                    "strike": strike,
                    "probability": probability,
                    "volume": volume,
                    "status": str(row.get("status", "") or ""),
                    "result": str(row.get("result", "") or ""),
                }
            )

        if not parsed:
            if self.bridge_cache_only:
                self.is_healthy = False
            return None

        if self.bridge_cache_only and not self._markets_are_informative(parsed):
            logger.warning("Kalshi bridge cache is not informative; ignoring snapshot")
            self.is_healthy = False
            return None

        parsed.sort(key=lambda row: row["strike"])
        resolved_ticker = snapshot_ticker or requested_ticker or self._current_event_ticker()
        self.last_resolved_ticker = resolved_ticker
        self.last_success = snapshot_ts or time.time()
        self.is_healthy = True
        self.consecutive_failures = 0
        with self._cache_lock:
            self._cache[resolved_ticker] = {"data": parsed, "ts": snapshot_ts or time.time()}
        return parsed

    @staticmethod
    def _markets_are_informative(markets: List[Dict]) -> bool:
        probabilities: List[float] = []
        for row in markets:
            try:
                probabilities.append(float(row.get("probability")))
            except (AttributeError, TypeError, ValueError):
                continue
        if len(probabilities) < 8:
            return False
        if (max(probabilities) - min(probabilities)) < 0.08:
            return False
        if len({round(probability, 4) for probability in probabilities}) < 4:
            return False
        return True

    # KXINXU settlement hours are Eastern Time (per CFTC filing: "Time will
    # be measured in Eastern Time (ET)" covering "traditional market hours
    # 9:30 AM - 4 PM ET").  Hourly contracts settle at 10-16 ET.
    _SETTLEMENT_HOURS_ET = [10, 11, 12, 13, 14, 15, 16]

    def active_settlement_hour_et(self, ref_time: Optional[datetime] = None, rollover_minute: int = 5) -> Optional[int]:
        et = pytz.timezone("US/Eastern")
        if ref_time is None:
            now = datetime.now(et)
        elif ref_time.tzinfo is None:
            now = et.localize(ref_time)
        else:
            now = ref_time.astimezone(et)

        for hour in self._SETTLEMENT_HOURS_ET:
            if hour > now.hour or (hour == now.hour and now.minute < int(rollover_minute)):
                return hour
        return None

    def _current_event_ticker(self) -> str:
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
        next_hour = self.active_settlement_hour_et(now, rollover_minute=5)
        if next_hour is None:
            next_hour = self._SETTLEMENT_HOURS_ET[-1]
        return f"{self.series}-{now.strftime('%y%b%d').upper()}H{next_hour * 100}"

    def event_ticker_for_hour(self, et_hour: int, ref_date: Optional[datetime] = None) -> str:
        """Build an event ticker for a specific ET settlement hour."""
        et = pytz.timezone("US/Eastern")
        if ref_date is None:
            ref_date = datetime.now(et)
        return f"{self.series}-{ref_date.strftime('%y%b%d').upper()}H{et_hour * 100}"

    def fetch_daily_contracts(self) -> List[Dict]:
        """Fetch all of today's hourly contracts for historical backfill.

        Contracts settle at each hour 10 AM - 4 PM ET.  On startup, this
        pre-populates the cache so the ML pipeline has data even for
        contracts that already settled before the bot started.
        """
        if not self.enabled:
            return []
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
        results = []
        for hour in self._SETTLEMENT_HOURS_ET:
            ticker = self.event_ticker_for_hour(hour, now)
            markets = self._fetch_event_markets(event_ticker=ticker)
            results.append({
                "et_hour": hour,
                "event_ticker": ticker,
                "strikes": markets,
                "settled": now.hour >= hour,
                "strike_count": len(markets),
            })
        return results

    def _event_sort_ts(self, event: Dict) -> float:
        # strike_date is on events; close_time is on markets (actual settlement time)
        for key in ("strike_date", "close_time", "expected_expiration_time", "settlement_time", "expiration_time", "event_time", "open_time"):
            raw = event.get(key)
            if not raw:
                continue
            if isinstance(raw, (int, float)):
                return float(raw)
            if isinstance(raw, str):
                ts = raw.strip()
                try:
                    if ts.endswith("Z"):
                        ts = ts.replace("Z", "+00:00")
                    return datetime.fromisoformat(ts).timestamp()
                except ValueError:
                    continue
        return 0.0

    def _resolve_event_from_series(self) -> Optional[Dict]:
        """Find the next upcoming event for this series.

        The Kalshi /events API returns objects with these fields:
          event_ticker, series_ticker, strike_date, title, sub_title, category
        Note: no 'status' or 'close_time' on events — those are on markets.
        """
        data = self._get(
            "/events",
            {
                "series_ticker": self.series,
                "limit": 50,
            },
        )
        events = data.get("events", []) if isinstance(data, dict) else []
        if not events:
            return None

        et = pytz.timezone("US/Eastern")
        now_ts = datetime.now(et).timestamp()
        candidates: List[tuple[float, Dict]] = []

        for event in events:
            if not isinstance(event, dict):
                continue
            # Events use 'event_ticker', not 'ticker'
            ticker = str(event.get("event_ticker", "") or event.get("ticker", "") or "")
            if not ticker.startswith(f"{self.series}-"):
                continue
            ts = self._event_sort_ts(event)
            if ts > 0 and ts >= now_ts:
                candidates.append((ts, event))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]

        # No future events — return the most recent one
        all_valid = [
            (self._event_sort_ts(evt), evt)
            for evt in events
            if isinstance(evt, dict) and self._event_sort_ts(evt) > 0
        ]
        if all_valid:
            all_valid.sort(key=lambda item: item[0], reverse=True)
            return all_valid[0][1]
        return None

    def _parse_markets(self, markets: List[Dict]) -> List[Dict]:
        parsed = []
        for market in markets:
            ticker = str(market.get("ticker", "") or "")
            if "-T" not in ticker:
                continue
            try:
                strike = float(ticker.split("-T")[-1])
            except ValueError:
                continue

            # Probability: use yes_bid midpoint (matches Kalshi website "Chance"),
            # NOT last_price_dollars which is often 0 when no trades have occurred.
            prob = None
            yes_bid = market.get("yes_bid_dollars") or market.get("yes_bid")
            yes_ask = market.get("yes_ask_dollars") or market.get("yes_ask")
            if yes_bid is not None and yes_ask is not None:
                try:
                    bid_f = float(yes_bid)
                    ask_f = float(yes_ask)
                    prob = (bid_f + ask_f) / 2.0
                except (TypeError, ValueError):
                    pass
            # Fall back to last_price if bid/ask unavailable
            if prob is None or prob == 0:
                for field in ("last_price_dollars", "last_price"):
                    val = market.get(field)
                    if val is not None:
                        try:
                            fv = float(val)
                            if fv > 0:
                                prob = fv
                                break
                        except (TypeError, ValueError):
                            pass
            if prob is None:
                continue

            # Normalize cents (0-100) to probability (0-1)
            if prob > 1.0:
                prob = prob / 100.0

            volume = 0.0
            for vol_field in ("volume_fp", "volume", "volume_24h_fp"):
                vf = market.get(vol_field)
                if vf is not None:
                    try:
                        volume = float(vf)
                        if volume > 0:
                            break
                    except (TypeError, ValueError):
                        pass

            parsed.append(
                {
                    "strike": strike,
                    "probability": prob,
                    "volume": volume,
                    "status": str(market.get("status", "") or ""),
                    "result": str(market.get("result", "") or ""),
                }
            )
        return sorted(parsed, key=lambda row: row["strike"])

    def _all_finalized(self, parsed: List[Dict]) -> bool:
        """Check if all parsed markets are settled/finalized."""
        if not parsed:
            return False
        settled_statuses = {"finalized", "settled", "closed"}
        return all(
            str(m.get("status", "") or "").lower() in settled_statuses
            for m in parsed
        )

    def _fetch_single_event(self, event_ticker: str, resolved_event: Optional[Dict] = None) -> List[Dict]:
        """Fetch and parse markets for a single event ticker."""
        with self._cache_lock:
            cached = self._cache.get(event_ticker)
            if cached and (time.time() - float(cached.get("ts", 0))) < self.cache_ttl:
                return list(cached.get("data", []))

        markets = []
        if isinstance(resolved_event, dict):
            markets = resolved_event.get("markets", []) or []
        if not markets:
            data = self._get(f"/events/{event_ticker}", {"with_nested_markets": "true"})
            markets = (data.get("event") or {}).get("markets", []) if isinstance(data, dict) else []
        if not markets:
            data = self._get("/markets", {"event_ticker": event_ticker, "limit": 200})
            markets = data.get("markets", []) if isinstance(data, dict) else []

        parsed = self._parse_markets(markets)
        if not parsed:
            return []
        # Don't cache finalized events for the full TTL — allow fast rotation
        if self._all_finalized(parsed):
            cache_ts = time.time() - self.cache_ttl + 10  # expires in 10s
        else:
            cache_ts = time.time()
        with self._cache_lock:
            self._cache[event_ticker] = {"data": parsed, "ts": cache_ts}
        return parsed

    def _fetch_event_markets(self, event_ticker: Optional[str] = None) -> List[Dict]:
        bridge_markets = self._load_bridge_markets(event_ticker)
        if bridge_markets is not None:
            return bridge_markets
        if self.bridge_cache_only:
            return []

        # If a specific ticker was requested, return it directly
        if event_ticker is not None:
            self.last_resolved_ticker = event_ticker
            result = self._fetch_single_event(event_ticker)
            if not result:
                logger.warning("No Kalshi markets parsed for %s", event_ticker)
            return result

        # Try series resolution first, then current ticker
        resolved_event = self._resolve_event_from_series()
        event_ticker = (
            str((resolved_event or {}).get("event_ticker", "") or
                (resolved_event or {}).get("ticker", "") or "").strip()
            or self._current_event_ticker()
        )
        self.last_resolved_ticker = event_ticker

        parsed = self._fetch_single_event(event_ticker, resolved_event)

        # If all markets are finalized, rotate to the next settlement hour
        if self._all_finalized(parsed):
            et = pytz.timezone("US/Eastern")
            now = datetime.now(et)
            for hour in self._SETTLEMENT_HOURS_ET:
                if hour > now.hour or (hour == now.hour and now.minute < 5):
                    next_ticker = self.event_ticker_for_hour(hour, now)
                    if next_ticker != event_ticker:
                        logger.info("Event %s is settled, rotating to %s", event_ticker, next_ticker)
                        next_parsed = self._fetch_single_event(next_ticker)
                        if next_parsed and not self._all_finalized(next_parsed):
                            self.last_resolved_ticker = next_ticker
                            return next_parsed
            logger.info("All settlement hours finalized for today")

        if not parsed:
            logger.warning("No Kalshi markets parsed for %s", event_ticker)
        return parsed

    def get_probability(self, strike_price: float) -> Optional[float]:
        if not self.enabled or not self.is_healthy:
            return None
        markets = self._fetch_event_markets()
        if not markets:
            return None
        for market in markets:
            if abs(market["strike"] - strike_price) < 0.01:
                return market["probability"]
        below = [m for m in markets if m["strike"] <= strike_price]
        above = [m for m in markets if m["strike"] > strike_price]
        if below and above:
            lo = below[-1]
            hi = above[0]
            frac = (strike_price - lo["strike"]) / (hi["strike"] - lo["strike"])
            return lo["probability"] * (1.0 - frac) + hi["probability"] * frac
        if not below:
            return markets[0]["probability"]
        if not above:
            return markets[-1]["probability"]
        return None

    def get_cached_event_markets(
        self,
        event_ticker: Optional[str] = None,
        *,
        max_age_seconds: Optional[float] = None,
    ) -> List[Dict]:
        if not self.enabled or not self.is_healthy:
            return []
        ticker = str(event_ticker or self.last_resolved_ticker or self._current_event_ticker() or "").strip()
        if not ticker:
            return []
        with self._cache_lock:
            cached = self._cache.get(ticker)
            if not cached:
                return []
            age = time.time() - float(cached.get("ts", 0.0) or 0.0)
            if max_age_seconds is not None and age > float(max_age_seconds):
                return []
            return list(cached.get("data", []) or [])

    @staticmethod
    def _interpolated_probability_from_markets(markets: List[Dict], strike_price: float) -> Optional[float]:
        if not markets:
            return None
        for market in markets:
            if abs(float(market.get("strike", 0.0)) - float(strike_price)) < 0.01:
                return float(market.get("probability"))
        below = [m for m in markets if float(m.get("strike", 0.0)) <= float(strike_price)]
        above = [m for m in markets if float(m.get("strike", 0.0)) > float(strike_price)]
        if below and above:
            lo = below[-1]
            hi = above[0]
            span = float(hi["strike"]) - float(lo["strike"])
            if span <= 0:
                return float(lo["probability"])
            frac = (float(strike_price) - float(lo["strike"])) / span
            return float(lo["probability"]) * (1.0 - frac) + float(hi["probability"]) * frac
        if not below:
            return float(markets[0]["probability"])
        if not above:
            return float(markets[-1]["probability"])
        return None

    @staticmethod
    def _implied_level_from_markets(markets: List[Dict]) -> Optional[float]:
        if len(markets) < 2:
            return None
        for idx in range(len(markets) - 1):
            p1 = markets[idx]["probability"]
            p2 = markets[idx + 1]["probability"]
            if p1 >= 0.5 > p2:
                s1 = markets[idx]["strike"]
                s2 = markets[idx + 1]["strike"]
                frac = (0.5 - p1) / (p2 - p1)
                return s1 + frac * (s2 - s1)
        return None

    def get_cached_probability(self, strike_price: float, *, max_age_seconds: Optional[float] = None) -> Optional[float]:
        markets = self.get_cached_event_markets(max_age_seconds=max_age_seconds)
        return self._interpolated_probability_from_markets(markets, strike_price)

    def get_probability_curve(self) -> Dict[float, float]:
        if not self.enabled or not self.is_healthy:
            return {}
        return {m["strike"]: m["probability"] for m in self._fetch_event_markets()}

    def get_nearest_market(self, strike_price: float) -> Optional[Dict]:
        if not self.enabled or not self.is_healthy:
            return None
        markets = self._fetch_event_markets()
        if not markets:
            return None
        return min(markets, key=lambda market: abs(float(market["strike"]) - float(strike_price)))

    def get_nearest_market_for_es_price(self, es_price: float) -> Optional[Dict]:
        if not self.enabled or not self.is_healthy:
            return None
        spx_price = self.es_to_spx(es_price)
        market = self.get_nearest_market(spx_price)
        if market is None:
            return None
        strike_spx = float(market["strike"])
        strike_es = self.spx_to_es(strike_spx)
        return {
            **market,
            "strike_spx": round(strike_spx, 2),
            "strike_es": round(strike_es, 2),
            "reference_spx": round(float(spx_price), 2),
            "reference_es": round(float(es_price), 2),
            "distance_spx": round(strike_spx - float(spx_price), 2),
            "distance_es": round(strike_es - float(es_price), 2),
        }

    def get_relative_markets_for_ui(self, es_prices: Optional[List[float]] = None, window_size: int = 30) -> List[Dict]:
        if not self.enabled or not self.is_healthy:
            return []
        markets = self._fetch_event_markets()
        return self._relative_markets_for_ui_from_markets(markets, es_prices=es_prices, window_size=window_size)

    def get_cached_relative_markets_for_ui(
        self,
        es_prices: Optional[List[float]] = None,
        window_size: int = 30,
        *,
        max_age_seconds: Optional[float] = None,
    ) -> List[Dict]:
        markets = self.get_cached_event_markets(max_age_seconds=max_age_seconds)
        return self._relative_markets_for_ui_from_markets(markets, es_prices=es_prices, window_size=window_size)

    def _relative_markets_for_ui_from_markets(
        self,
        markets: List[Dict],
        *,
        es_prices: Optional[List[float]] = None,
        window_size: int = 30,
    ) -> List[Dict]:
        if not markets:
            return []
        if len(markets) <= int(window_size):
            return markets

        reference_spx_prices: List[float] = []
        for es_price in es_prices or []:
            if es_price is None:
                continue
            try:
                reference_spx_prices.append(self.es_to_spx(float(es_price)))
            except (TypeError, ValueError):
                continue

        if not reference_spx_prices:
            implied_level = self._implied_level_from_markets(markets)
            if implied_level is not None:
                reference_spx_prices.append(float(implied_level))

        window_size = max(1, int(window_size))
        if not reference_spx_prices:
            midpoint = len(markets) // 2
            start = max(0, midpoint - (window_size // 2))
            end = min(len(markets), start + window_size)
            return markets[max(0, end - window_size):end]

        nearest_indices = [
            min(
                range(len(markets)),
                key=lambda idx: abs(float(markets[idx]["strike"]) - reference_spx_price),
            )
            for reference_spx_price in reference_spx_prices
        ]
        low_idx = min(nearest_indices)
        high_idx = max(nearest_indices)
        span = (high_idx - low_idx) + 1

        if span >= window_size:
            midpoint = (low_idx + high_idx) // 2
            start = max(0, midpoint - (window_size // 2))
            end = min(len(markets), start + window_size)
            return markets[max(0, end - window_size):end]

        padding = window_size - span
        start = max(0, low_idx - (padding // 2))
        end = min(len(markets), high_idx + 1 + (padding - (padding // 2)))
        if (end - start) < window_size:
            if start == 0:
                end = min(len(markets), window_size)
            else:
                start = max(0, end - window_size)
        return markets[start:end]

    def get_target_probability(self, es_price: float, side: Optional[str] = None) -> Dict:
        payload = {
            "probability": None,
            "market_probability": None,
            "outcome_side": None,
            "strike_spx": None,
            "strike_es": None,
            "reference_spx": None,
            "reference_es": None,
            "distance_spx": None,
            "distance_es": None,
            "status": None,
            "result": None,
        }
        market = self.get_nearest_market_for_es_price(es_price)
        if market is None:
            return payload

        market_probability = market.get("probability")
        if market_probability is None:
            return payload
        raw_probability = float(market_probability)
        normalized_side = str(side or "").strip().upper()
        outcome_side = "below" if normalized_side == "SHORT" else "above"
        probability = 1.0 - raw_probability if outcome_side == "below" else raw_probability
        payload.update(
            {
                "probability": round(float(probability), 4),
                "market_probability": round(raw_probability, 4),
                "outcome_side": outcome_side,
                "strike_spx": market.get("strike_spx"),
                "strike_es": market.get("strike_es"),
                "reference_spx": market.get("reference_spx"),
                "reference_es": market.get("reference_es"),
                "distance_spx": market.get("distance_spx"),
                "distance_es": market.get("distance_es"),
                "status": market.get("status"),
                "result": market.get("result"),
            }
        )
        return payload

    def get_implied_level(self) -> Optional[float]:
        if not self.enabled or not self.is_healthy:
            return None
        markets = self._fetch_event_markets()
        return self._implied_level_from_markets(markets)

    def get_sentiment(self, es_price: float) -> Dict:
        return self._sentiment_from_markets(es_price, self._fetch_event_markets())

    def get_cached_sentiment(self, es_price: float, *, max_age_seconds: Optional[float] = None) -> Dict:
        return self._sentiment_from_markets(
            es_price,
            self.get_cached_event_markets(max_age_seconds=max_age_seconds),
        )

    def _sentiment_from_markets(self, es_price: float, markets: List[Dict]) -> Dict:
        payload = {
            "probability": None,
            "classification": "unavailable",
            "implied_level": None,
            "distance": None,
            "implied_level_es": None,
            "distance_es": None,
            "implied_level_spx": None,
            "distance_spx": None,
            "healthy": self.is_healthy,
        }
        if not self.enabled or not self.is_healthy:
            return payload

        es_price = float(es_price)
        spx_price = self.es_to_spx(es_price)
        probability = self._interpolated_probability_from_markets(markets, spx_price)
        implied_level_spx = self._implied_level_from_markets(markets)
        if probability is None:
            return payload

        if probability >= 0.70:
            classification = "strong_bull"
        elif probability >= 0.55:
            classification = "bull"
        elif probability >= 0.45:
            classification = "neutral"
        elif probability >= 0.30:
            classification = "bear"
        else:
            classification = "strong_bear"

        implied_level_es = self.spx_to_es(implied_level_spx) if implied_level_spx is not None else None
        distance_spx = implied_level_spx - spx_price if implied_level_spx is not None else None
        distance_es = implied_level_es - es_price if implied_level_es is not None else None
        payload.update(
            {
                "probability": round(float(probability), 4),
                "classification": classification,
                # Expose ES-space values by default so callers can compare
                # Kalshi context directly against MES/ES live prices.
                "implied_level": round(float(implied_level_es), 2) if implied_level_es is not None else None,
                "distance": round(float(distance_es), 2) if distance_es is not None else None,
                "implied_level_es": round(float(implied_level_es), 2) if implied_level_es is not None else None,
                "distance_es": round(float(distance_es), 2) if distance_es is not None else None,
                "implied_level_spx": round(float(implied_level_spx), 2) if implied_level_spx is not None else None,
                "distance_spx": round(float(distance_spx), 2) if distance_spx is not None else None,
                "healthy": True,
            }
        )
        return payload

    def get_probability_gradient(self, es_price: float) -> Optional[float]:
        if not self.enabled or not self.is_healthy:
            return None
        spx_price = self.es_to_spx(es_price)
        markets = self._fetch_event_markets()
        if len(markets) < 3:
            return None
        below = [m for m in markets if m["strike"] <= spx_price]
        above = [m for m in markets if m["strike"] > spx_price]
        if not below or not above:
            return None
        lo = below[-1]
        hi = above[0]
        gradient = (hi["probability"] - lo["probability"]) / (hi["strike"] - lo["strike"])
        return round(float(gradient), 6)

    def get_implied_distribution(self, es_price: float) -> Optional[Dict]:
        _ = es_price
        if not self.enabled or not self.is_healthy:
            return None
        markets = self._fetch_event_markets()
        if len(markets) < 10:
            return None
        densities: List[float] = []
        midpoints: List[float] = []
        for idx in range(len(markets) - 1):
            ds = markets[idx + 1]["strike"] - markets[idx]["strike"]
            if ds <= 0:
                continue
            dp = markets[idx]["probability"] - markets[idx + 1]["probability"]
            density = max(0.0, dp / ds)
            densities.append(density)
            midpoints.append((markets[idx]["strike"] + markets[idx + 1]["strike"]) / 2.0)
        total = sum(densities)
        if total <= 0:
            return None
        norm = [value / total for value in densities]
        mean = sum(mid * den for mid, den in zip(midpoints, norm))
        variance = sum(den * (mid - mean) ** 2 for mid, den in zip(midpoints, norm))
        std = variance**0.5
        skew = sum(den * (((mid - mean) / std) ** 3) for mid, den in zip(midpoints, norm)) if std > 0 else 0.0
        return {
            "implied_mean": round(float(mean), 2),
            "implied_std": round(float(std), 2),
            "implied_skew": round(float(skew), 4),
        }

    def get_sentiment_momentum(self, es_price: float, lookback: int = 3) -> Optional[float]:
        probability = self.get_probability(self.es_to_spx(es_price))
        return self._sentiment_momentum_from_probability(probability, lookback=lookback)

    def get_cached_sentiment_momentum(
        self,
        es_price: float,
        lookback: int = 3,
        *,
        max_age_seconds: Optional[float] = None,
    ) -> Optional[float]:
        probability = self.get_cached_probability(self.es_to_spx(es_price), max_age_seconds=max_age_seconds)
        return self._sentiment_momentum_from_probability(probability, lookback=lookback)

    def _sentiment_momentum_from_probability(self, probability: Optional[float], lookback: int = 3) -> Optional[float]:
        if probability is None:
            return None
        self._sentiment_history.append((time.time(), probability))
        self._sentiment_history = self._sentiment_history[-20:]
        if len(self._sentiment_history) < int(lookback) + 1:
            return None
        prior = self._sentiment_history[-(int(lookback) + 1)][1]
        return round(float(probability - prior), 4)

    def update_basis(self, es_price: float, spx_price: float) -> None:
        self.basis_offset = round(float(es_price) - float(spx_price), 2)
        logger.info("Kalshi basis updated: ES-SPX=%s", self.basis_offset)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
