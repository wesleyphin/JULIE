import json
import time
import unittest
from unittest.mock import mock_open, patch

from services.kalshi_provider import KalshiProvider
from tools.kalshi_public_bridge import summarize_curve


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload=None):
        self.payload = payload or {"markets": []}
        self.calls = []
        self.proxies = {}
        self.trust_env = True

    def get(self, url, **kwargs):
        self.calls.append({"url": url, "kwargs": kwargs})
        return FakeResponse(self.payload)


class KalshiProviderPublicModeTests(unittest.TestCase):
    def test_public_read_only_stays_enabled_without_credentials(self):
        session = FakeSession({"markets": []})
        provider = KalshiProvider(
            {
                "enabled": True,
                "base_url": "https://kalshi.example.test/trade-api/v2",
                "public_read_only": True,
                "key_id": "",
                "private_key_path": "",
                "session": session,
            }
        )

        self.assertTrue(provider.enabled)
        self.assertTrue(provider.public_read_only)

    def test_public_read_only_does_not_send_auth_headers(self):
        session = FakeSession({"markets": []})
        provider = KalshiProvider(
            {
                "enabled": True,
                "base_url": "https://kalshi.example.test/trade-api/v2",
                "public_read_only": True,
                "session": session,
                "rate_limit_delay": 0,
            }
        )

        provider._get("/markets", {"series_ticker": "KXINXU", "limit": 1})

        self.assertEqual(1, len(session.calls))
        self.assertNotIn("headers", session.calls[0]["kwargs"])

    def test_provider_uses_scoped_proxy_and_ignores_environment_by_default(self):
        session = FakeSession({"markets": []})
        provider = KalshiProvider(
            {
                "enabled": True,
                "base_url": "https://kalshi.example.test/trade-api/v2",
                "public_read_only": True,
                "market_data_proxy_url": "http://127.0.0.1:8080",
                "trust_env": False,
                "session": session,
            }
        )

        self.assertIs(provider.session, session)
        self.assertFalse(session.trust_env)
        self.assertEqual("http://127.0.0.1:8080", session.proxies["http"])
        self.assertEqual("http://127.0.0.1:8080", session.proxies["https"])

    def test_bridge_cache_only_loads_snapshot_without_network(self):
        snapshot = {
            "ts": time.time(),
            "event_ticker": "KXINXU-26APR28H1600",
            "markets": [
                {"strike": 7000, "probability": 0.81, "volume": "bad", "status": "active"},
                {"strike": 7005, "probability": 0.72, "volume": 7, "status": "active"},
                {"strike": 7010, "probability": 0.63, "volume": 7, "status": "active"},
                {"strike": 7015, "probability": 0.55, "volume": 7, "status": "active"},
                {"strike": 7020, "probability": 0.44, "volume": 7, "status": "active"},
                {"strike": 7025, "probability": 0.33, "volume": 7, "status": "active"},
                {"strike": 7030, "probability": 0.22, "volume": 7, "status": "active"},
                {"strike": 7035, "probability": 0.11, "volume": 7, "status": "active"},
            ],
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(snapshot))):
            session = FakeSession({"events": []})
            provider = KalshiProvider(
                {
                    "enabled": True,
                    "base_url": "https://kalshi.example.test/trade-api/v2",
                    "public_read_only": True,
                    "bridge_cache_path": "kalshi_bridge_public_mode.json",
                    "bridge_cache_only": True,
                    "bridge_cache_max_age_seconds": 30,
                    "session": session,
                }
            )

            markets = provider.refresh()

        self.assertEqual(8, len(markets))
        self.assertEqual(0.0, markets[0]["volume"])
        self.assertEqual("KXINXU-26APR28H1600", provider.last_resolved_ticker)
        self.assertEqual([], session.calls)

    def test_bridge_cache_only_marks_stale_snapshot_unhealthy(self):
        snapshot = {
            "ts": time.time() - 120,
            "event_ticker": "KXINXU-26APR28H1600",
            "markets": [{"strike": 7000, "probability": 0.61}],
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(snapshot))):
            provider = KalshiProvider(
                {
                    "enabled": True,
                    "base_url": "https://kalshi.example.test/trade-api/v2",
                    "public_read_only": True,
                    "bridge_cache_path": "kalshi_bridge_public_mode.json",
                    "bridge_cache_only": True,
                    "bridge_cache_max_age_seconds": 30,
                    "session": FakeSession({"events": []}),
                }
            )

            markets = provider.refresh()

        self.assertEqual([], markets)
        self.assertFalse(provider.is_healthy)

    def test_bridge_cache_only_rejects_flat_snapshot_as_unhealthy(self):
        snapshot = {
            "ts": time.time(),
            "event_ticker": "KXINXU-26APR28H1600",
            "markets": [
                {"strike": 7000 + (idx * 5), "probability": 0.0}
                for idx in range(8)
            ],
        }
        with patch("builtins.open", mock_open(read_data=json.dumps(snapshot))):
            provider = KalshiProvider(
                {
                    "enabled": True,
                    "base_url": "https://kalshi.example.test/trade-api/v2",
                    "public_read_only": True,
                    "bridge_cache_path": "kalshi_bridge_public_mode.json",
                    "bridge_cache_only": True,
                    "bridge_cache_max_age_seconds": 30,
                    "session": FakeSession({"events": []}),
                }
            )

            markets = provider.refresh()

        self.assertEqual([], markets)
        self.assertFalse(provider.is_healthy)

    def test_bridge_summary_flags_flat_curve_as_not_informative(self):
        summary = summarize_curve(
            [
                {"probability": 0.0},
                {"probability": 0.0},
                {"probability": 0.0},
                {"probability": 0.0},
                {"probability": 0.0},
                {"probability": 0.0},
                {"probability": 0.0},
                {"probability": 0.0},
            ]
        )

        self.assertFalse(summary["informative_curve"])
        self.assertEqual(0, summary["nonzero_probability_count"])

    def test_bridge_summary_flags_varied_curve_as_informative(self):
        summary = summarize_curve(
            [
                {"probability": 0.81},
                {"probability": 0.72},
                {"probability": 0.63},
                {"probability": 0.55},
                {"probability": 0.44},
                {"probability": 0.33},
                {"probability": 0.22},
                {"probability": 0.11},
            ]
        )

        self.assertTrue(summary["informative_curve"])
        self.assertEqual(8, summary["nonzero_probability_count"])


if __name__ == "__main__":
    unittest.main()
