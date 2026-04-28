import importlib.util
import unittest

from tools.market_data_proxy_probe import (
    DEFAULT_KALSHI_BASE_URL,
    DEFAULT_POLYMARKET_US_BASE_URL,
    MarketDataProxyConfig,
    sanitized_proxy_url,
)


class MarketDataProxyProbeTests(unittest.TestCase):
    def test_defaults_are_direct_and_scoped(self):
        cfg = MarketDataProxyConfig.from_env({})
        session = cfg.build_requests_session()

        self.assertIsNone(cfg.proxy_url)
        self.assertEqual(DEFAULT_KALSHI_BASE_URL, cfg.kalshi_base_url)
        self.assertEqual(DEFAULT_POLYMARKET_US_BASE_URL, cfg.polymarket_us_base_url)
        self.assertFalse(session.trust_env)
        self.assertEqual({}, session.proxies)
        self.assertEqual([], cfg.validate())

    def test_proxy_url_is_applied_only_to_the_probe_session(self):
        cfg = MarketDataProxyConfig.from_env(
            {
                "MARKET_DATA_PROXY_URL": "http://proxy.local:8080",
                "KALSHI_BASE_URL": "https://kalshi.example.test/base/",
                "POLYMARKET_US_BASE_URL": "https://poly.example.test/",
            }
        )
        session = cfg.build_requests_session()

        self.assertFalse(session.trust_env)
        self.assertEqual("http://proxy.local:8080", session.proxies["http"])
        self.assertEqual("http://proxy.local:8080", session.proxies["https"])
        self.assertEqual("https://kalshi.example.test/base", cfg.kalshi_base_url)
        self.assertEqual("https://poly.example.test", cfg.polymarket_us_base_url)

    def test_sanitized_proxy_url_redacts_credentials(self):
        self.assertEqual(
            "socks5h://***:***@127.0.0.1:1080",
            sanitized_proxy_url("socks5h://user:secret@127.0.0.1:1080"),
        )

    def test_socks_validation_matches_installed_support(self):
        cfg = MarketDataProxyConfig.from_env(
            {"MARKET_DATA_PROXY_URL": "socks5h://127.0.0.1:1080"}
        )
        errors = cfg.validate()
        socks_installed = importlib.util.find_spec("socks") is not None

        if socks_installed:
            self.assertEqual([], errors)
        else:
            self.assertTrue(any("PySocks" in err for err in errors))


if __name__ == "__main__":
    unittest.main()
