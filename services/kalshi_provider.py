import base64
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

import pytz
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


logger = logging.getLogger("kalshi_provider")


class KalshiProvider:
    """
    Authenticated Kalshi API client for KXINXU (S&P 500 above/below hourly contracts).
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

        self.private_key = None
        private_key_path = str(config.get("private_key_path", "") or "")
        if private_key_path:
            try:
                with open(private_key_path, "rb") as key_file:
                    self.private_key = serialization.load_pem_private_key(key_file.read(), password=None)
            except (FileNotFoundError, OSError, ValueError) as exc:
                logger.warning("Unable to load Kalshi private key from %s: %s", private_key_path, exc)

        if not self.key_id or self.private_key is None:
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
        if not self.key_id or self.private_key is None:
            logger.warning("Kalshi credentials missing; disabling provider calls")
            return {}

        ts_str = str(int(time.time()))
        sig = self._sign("GET", f"/trade-api/v2{path}", ts_str)
        headers = {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts_str,
        }
        url = f"{self.base_url}{path}"

        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
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

    # Kalshi KXINXU contracts predict Pacific Time hours
    _PREDICTION_HOURS_PT = [10, 11, 12, 13, 14, 15, 16]

    def _current_event_ticker(self) -> str:
        pt = pytz.timezone("US/Pacific")
        now = datetime.now(pt)

        next_hour = None
        for hour in self._PREDICTION_HOURS_PT:
            if hour > now.hour or (hour == now.hour and now.minute < 5):
                next_hour = hour
                break
        if next_hour is None:
            next_hour = self._PREDICTION_HOURS_PT[-1]
        return f"{self.series}-{now.strftime('%y%b%d').upper()}H{next_hour * 100}"

    def event_ticker_for_hour(self, pt_hour: int, ref_date: Optional[datetime] = None) -> str:
        """Build an event ticker for a specific PT prediction hour."""
        pt = pytz.timezone("US/Pacific")
        if ref_date is None:
            ref_date = datetime.now(pt)
        return f"{self.series}-{ref_date.strftime('%y%b%d').upper()}H{pt_hour * 100}"

    def fetch_daily_contracts(self) -> List[Dict]:
        """Fetch all of today's hourly contracts for historical backfill.

        Each contract becomes tradable 4 hours before its PT prediction hour
        and closes 1 hour later (3 hours before the prediction hour).
        """
        if not self.enabled:
            return []
        pt = pytz.timezone("US/Pacific")
        now = datetime.now(pt)
        results = []
        for hour in self._PREDICTION_HOURS_PT:
            ticker = self.event_ticker_for_hour(hour, now)
            markets = self._fetch_event_markets(event_ticker=ticker)
            close_hour = hour - 3  # Trading closes 3 hours before prediction
            results.append({
                "pt_hour": hour,
                "event_ticker": ticker,
                "strikes": markets,
                "settled": now.hour >= close_hour,
                "strike_count": len(markets),
            })
        return results

    def _event_sort_ts(self, event: Dict) -> float:
        for key in ("expiration_time", "close_time", "settlement_time", "event_time", "open_time"):
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
        data = self._get(
            "/events",
            {
                "series_ticker": self.series,
                "with_nested_markets": "true",
                "limit": 200,
            },
        )
        events = data.get("events", []) if isinstance(data, dict) else []
        if not events:
            return None

        et = pytz.timezone("US/Eastern")
        now_ts = datetime.now(et).timestamp()
        preferred_statuses = {"open", "active", "initialized", "unsettled"}
        candidates: List[tuple[float, Dict]] = []
        fallback: List[tuple[float, Dict]] = []

        for event in events:
            if not isinstance(event, dict):
                continue
            ticker = str(event.get("ticker", "") or "")
            if not ticker.startswith(f"{self.series}-"):
                continue
            ts = self._event_sort_ts(event)
            status = str(event.get("status", "") or "").lower()
            if ts >= now_ts and status in preferred_statuses:
                candidates.append((ts, event))
            elif status in preferred_statuses:
                fallback.append((ts, event))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
        if fallback:
            fallback.sort(key=lambda item: abs(item[0] - now_ts))
            return fallback[0][1]
        events = [evt for evt in events if isinstance(evt, dict)]
        if not events:
            return None
        events.sort(key=self._event_sort_ts)
        return events[-1]

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
            last_price = market.get("last_price_dollars")
            if last_price is None:
                continue
            parsed.append(
                {
                    "strike": strike,
                    "probability": float(last_price),
                    "volume": float(market.get("volume_fp", 0) or 0.0),
                    "status": str(market.get("status", "") or ""),
                    "result": str(market.get("result", "") or ""),
                }
            )
        return sorted(parsed, key=lambda row: row["strike"])

    def _fetch_event_markets(self, event_ticker: Optional[str] = None) -> List[Dict]:
        resolved_event = None
        if event_ticker is None:
            resolved_event = self._resolve_event_from_series()
            event_ticker = (
                str((resolved_event or {}).get("ticker", "") or "").strip()
                or self._current_event_ticker()
            )
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
            logger.warning("No Kalshi markets parsed for %s", event_ticker)
            return []
        with self._cache_lock:
            self._cache[event_ticker] = {"data": parsed, "ts": time.time()}
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

    def get_probability_curve(self) -> Dict[float, float]:
        if not self.enabled or not self.is_healthy:
            return {}
        return {m["strike"]: m["probability"] for m in self._fetch_event_markets()}

    def get_implied_level(self) -> Optional[float]:
        if not self.enabled or not self.is_healthy:
            return None
        markets = self._fetch_event_markets()
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

    def get_sentiment(self, es_price: float) -> Dict:
        payload = {
            "probability": None,
            "classification": "unavailable",
            "implied_level": None,
            "distance": None,
            "healthy": self.is_healthy,
        }
        if not self.enabled or not self.is_healthy:
            return payload

        spx_price = float(es_price) - self.basis_offset
        probability = self.get_probability(spx_price)
        implied_level = self.get_implied_level()
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

        distance = implied_level - spx_price if implied_level is not None else None
        payload.update(
            {
                "probability": round(float(probability), 4),
                "classification": classification,
                "implied_level": round(float(implied_level), 2) if implied_level is not None else None,
                "distance": round(float(distance), 2) if distance is not None else None,
                "healthy": True,
            }
        )
        return payload

    def get_probability_gradient(self, es_price: float) -> Optional[float]:
        if not self.enabled or not self.is_healthy:
            return None
        spx_price = float(es_price) - self.basis_offset
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
        probability = self.get_probability(float(es_price) - self.basis_offset)
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
