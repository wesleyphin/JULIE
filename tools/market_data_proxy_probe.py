from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests


SUPPORTED_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h"}
DEFAULT_KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_POLYMARKET_US_BASE_URL = "https://gateway.polymarket.us"


def _clean_base_url(value: str, default: str) -> str:
    raw = str(value or "").strip() or default
    return raw.rstrip("/")


def sanitized_proxy_url(proxy_url: Optional[str]) -> Optional[str]:
    if not proxy_url:
        return None
    parsed = urlparse(proxy_url)
    if parsed.username or parsed.password:
        host = parsed.hostname or ""
        netloc = "***:***@" + host
        if parsed.port is not None:
            netloc = f"{netloc}:{parsed.port}"
        parsed = parsed._replace(netloc=netloc)
    return urlunparse(parsed)


@dataclass(frozen=True)
class MarketDataProxyConfig:
    proxy_url: Optional[str]
    kalshi_base_url: str = DEFAULT_KALSHI_BASE_URL
    polymarket_us_base_url: str = DEFAULT_POLYMARKET_US_BASE_URL

    @classmethod
    def from_env(cls, environ: Optional[Dict[str, str]] = None) -> "MarketDataProxyConfig":
        env = environ if environ is not None else os.environ
        proxy_url = str(env.get("MARKET_DATA_PROXY_URL", "") or "").strip() or None
        return cls(
            proxy_url=proxy_url,
            kalshi_base_url=_clean_base_url(
                str(env.get("KALSHI_BASE_URL", "") or ""),
                DEFAULT_KALSHI_BASE_URL,
            ),
            polymarket_us_base_url=_clean_base_url(
                str(env.get("POLYMARKET_US_BASE_URL", "") or ""),
                DEFAULT_POLYMARKET_US_BASE_URL,
            ),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if self.proxy_url:
            scheme = urlparse(self.proxy_url).scheme.lower()
            if scheme not in SUPPORTED_PROXY_SCHEMES:
                errors.append(
                    "MARKET_DATA_PROXY_URL must use one of: "
                    + ", ".join(sorted(SUPPORTED_PROXY_SCHEMES))
                )
            if scheme in {"socks5", "socks5h"} and importlib.util.find_spec("socks") is None:
                errors.append(
                    "SOCKS proxy URLs require PySocks. Install with `pip install requests[socks]`."
                )
        for label, url in {
            "KALSHI_BASE_URL": self.kalshi_base_url,
            "POLYMARKET_US_BASE_URL": self.polymarket_us_base_url,
        }.items():
            scheme = urlparse(url).scheme.lower()
            if scheme != "https":
                errors.append(f"{label} must be an https URL.")
        return errors

    def build_requests_session(self) -> requests.Session:
        session = requests.Session()
        session.trust_env = False
        if self.proxy_url:
            session.proxies.update({"http": self.proxy_url, "https": self.proxy_url})
        return session

    def describe(self) -> Dict[str, Any]:
        return {
            "proxy_enabled": bool(self.proxy_url),
            "proxy_url": sanitized_proxy_url(self.proxy_url),
            "kalshi_base_url": self.kalshi_base_url,
            "polymarket_us_base_url": self.polymarket_us_base_url,
            "requests_trust_env": False,
            "socks_support_installed": importlib.util.find_spec("socks") is not None,
        }


def _probe_get(session: requests.Session, url: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    response = session.get(url, params=params, timeout=timeout)
    payload: Dict[str, Any] = {
        "url": url,
        "status_code": response.status_code,
        "ok": bool(response.ok),
    }
    try:
        data = response.json()
    except ValueError:
        payload["json_keys"] = []
    else:
        payload["json_keys"] = sorted(data.keys()) if isinstance(data, dict) else []
    return payload


def run_probe(config: MarketDataProxyConfig, probe: str, timeout: float) -> Dict[str, Any]:
    session = config.build_requests_session()
    if probe == "kalshi-public":
        return _probe_get(
            session,
            urljoin(config.kalshi_base_url + "/", "markets"),
            {"series_ticker": "KXINXU", "limit": 1},
            timeout,
        )
    if probe == "polymarket-public":
        return _probe_get(
            session,
            urljoin(config.polymarket_us_base_url + "/", "v1/markets"),
            {"limit": 1},
            timeout,
        )
    return {"skipped": True}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run scoped market-data proxy settings without touching the live bot."
    )
    parser.add_argument(
        "--probe",
        choices=["none", "kalshi-public", "polymarket-public"],
        default="none",
        help="Optional unauthenticated network probe. Default only validates configuration.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = MarketDataProxyConfig.from_env()
    errors = config.validate()
    result: Dict[str, Any] = {"config": config.describe(), "errors": errors}
    if errors:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    if args.probe != "none":
        try:
            result["probe"] = run_probe(config, args.probe, args.timeout)
        except requests.RequestException as exc:
            result["probe"] = {"ok": False, "error": str(exc)}
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
