from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.market_data_proxy_probe import (  # noqa: E402
    DEFAULT_POLYMARKET_US_BASE_URL,
    MarketDataProxyConfig,
)


DEFAULT_DISCOVERY_QUERIES = ("SPX", "S&P 500", "SPY", "ES", "S&P")
DEFAULT_LIMIT = 10
LEVEL_RE = re.compile(r"(?<!\d)\$?([1-9]\d{0,2}(?:,\d{3})+|[1-9]\d{3,5})(?:\.\d+)?")
INDEX_TOKENS = (
    "spx",
    "s&p 500",
    "s&p500",
    "standard & poor",
    "standard and poor",
    "spy",
    "e-mini",
    "e mini",
    "es futures",
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _amount_value(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _first_present(*pairs: Tuple[Dict[str, Any], str]) -> Any:
    for source, key in pairs:
        if isinstance(source, dict) and key in source and source.get(key) is not None:
            return source.get(key)
    return None


def extract_level_candidates(text: str) -> List[float]:
    levels: List[float] = []
    for match in LEVEL_RE.finditer(text):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if 1000 <= value <= 10000:
            levels.append(value)
    return sorted(set(levels))


def classify_candidate(text: str) -> str:
    lowered = text.lower()
    levels = extract_level_candidates(text)
    if "up or down" in lowered or "opens up or down" in lowered:
        return "directional_up_down"
    if "__" in text or "hit __" in lowered:
        return "multi_strike_parent"
    if levels and any(token in lowered for token in ("above", "below", "hit", "close", "reach")):
        return "strike_or_level"
    if levels:
        return "level_related"
    return "related"


def is_index_related(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in INDEX_TOKENS)


def _iter_search_events(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    events = payload.get("events")
    if isinstance(events, list):
        yield from (event for event in events if isinstance(event, dict))
        return

    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("events"), list):
        yield from (event for event in data["events"] if isinstance(event, dict))
        return

    if isinstance(data, list):
        yield from (event for event in data if isinstance(event, dict))


def extract_market_candidates(payload: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for event in _iter_search_events(payload):
        event_title = _clean_text(
            _first_present((event, "title"), (event, "question"), (event, "name"))
        )
        event_slug = _clean_text(_first_present((event, "slug"), (event, "eventSlug")))
        event_category = _clean_text(
            _first_present((event, "category"), (event, "subcategory"))
        )
        markets = event.get("markets")
        if not isinstance(markets, list):
            markets = [event]

        for market in markets:
            if not isinstance(market, dict):
                continue
            title = _clean_text(
                _first_present(
                    (market, "question"),
                    (market, "title"),
                    (market, "name"),
                    (market, "description"),
                )
            )
            slug = _clean_text(
                _first_present((market, "slug"), (market, "marketSlug"), (market, "id"))
            )
            combined_text = _clean_text(f"{event_title} {title}")
            candidates.append(
                {
                    "query": query,
                    "event_slug": event_slug,
                    "event_title": event_title,
                    "market_slug": slug,
                    "market_title": title,
                    "category": _clean_text(
                        _first_present(
                            (market, "category"),
                            (market, "subcategory"),
                            (event, "category"),
                        )
                        or event_category
                    ),
                    "active": _bool_or_none(_first_present((market, "active"), (event, "active"))),
                    "closed": _bool_or_none(_first_present((market, "closed"), (event, "closed"))),
                    "end_date": _clean_text(
                        _first_present(
                            (market, "endDate"),
                            (market, "endDateIso"),
                            (market, "end_date"),
                            (event, "endDate"),
                        )
                    )
                    or None,
                    "kind": classify_candidate(combined_text),
                    "index_related": is_index_related(combined_text),
                    "levels": extract_level_candidates(combined_text),
                    "best_bid": _amount_value(_first_present((market, "bestBid"), (market, "bid"))),
                    "best_ask": _amount_value(_first_present((market, "bestAsk"), (market, "ask"))),
                    "current_px": _amount_value(
                        _first_present(
                            (market, "currentPx"),
                            (market, "lastTradePrice"),
                            (market, "lastTradePx"),
                        )
                    ),
                    "volume": _amount_value(
                        _first_present((market, "volumeNum"), (market, "volume"))
                    ),
                    "liquidity": _amount_value(
                        _first_present((market, "liquidityNum"), (market, "liquidity"))
                    ),
                }
            )
    return candidates


def dedupe_candidates(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            str(candidate.get("event_slug") or ""),
            str(candidate.get("market_slug") or candidate.get("market_title") or ""),
        )
        if key not in deduped:
            deduped[key] = candidate
            continue
        existing_queries = set(str(deduped[key].get("query") or "").split(", "))
        existing_queries.add(str(candidate.get("query") or ""))
        deduped[key]["query"] = ", ".join(sorted(q for q in existing_queries if q))
    return list(deduped.values())


class PolymarketDiscoveryClient:
    def __init__(
        self,
        base_url: str = DEFAULT_POLYMARKET_US_BASE_URL,
        session: Optional[requests.Session] = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = str(base_url or DEFAULT_POLYMARKET_US_BASE_URL).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = float(timeout)

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object from {url}")
        return payload

    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
        return self._get("/v1/search", {"query": query, "limit": int(limit)})

    def discover(
        self,
        queries: Sequence[str] = DEFAULT_DISCOVERY_QUERIES,
        limit: int = DEFAULT_LIMIT,
    ) -> Dict[str, Any]:
        all_candidates: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        for query in queries:
            try:
                payload = self.search(query, limit=limit)
            except (requests.RequestException, ValueError) as exc:
                errors.append({"query": query, "error": str(exc)})
                continue
            all_candidates.extend(extract_market_candidates(payload, query))

        raw_candidates = dedupe_candidates(all_candidates)
        candidates = [
            candidate for candidate in raw_candidates if bool(candidate.get("index_related"))
        ]
        kind_counts = Counter(str(candidate.get("kind") or "unknown") for candidate in candidates)
        raw_kind_counts = Counter(
            str(candidate.get("kind") or "unknown") for candidate in raw_candidates
        )
        return {
            "base_url": self.base_url,
            "queries": list(queries),
            "raw_candidate_count": len(raw_candidates),
            "candidate_count": len(candidates),
            "kind_counts": dict(sorted(kind_counts.items())),
            "raw_kind_counts": dict(sorted(raw_kind_counts.items())),
            "errors": errors,
            "candidates": candidates,
        }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Polymarket US for SPX/ES-like market candidates without live bot wiring."
    )
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = MarketDataProxyConfig.from_env()
    errors = config.validate()
    result: Dict[str, Any] = {
        "config": config.describe(),
        "validation_errors": errors,
    }
    if errors:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    client = PolymarketDiscoveryClient(
        base_url=config.polymarket_us_base_url,
        session=config.build_requests_session(),
        timeout=args.timeout,
    )
    queries = tuple(args.queries or DEFAULT_DISCOVERY_QUERIES)
    result["discovery"] = client.discover(queries=queries, limit=args.limit)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["discovery"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
