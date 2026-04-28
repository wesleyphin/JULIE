import unittest

from tools.polymarket_discovery_probe import (
    PolymarketDiscoveryClient,
    classify_candidate,
    extract_level_candidates,
    extract_market_candidates,
    is_index_related,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payload)


SAMPLE_SEARCH_PAYLOAD = {
    "events": [
        {
            "slug": "spx-apr-27",
            "title": "S&P 500 (SPX) Up or Down on April 27?",
            "category": "finance",
            "active": True,
            "markets": [
                {
                    "slug": "spx-up-or-down-apr-27",
                    "question": "S&P 500 (SPX) Up or Down on April 27?",
                    "bestBid": {"value": "0.54", "currency": "USD"},
                    "bestAsk": {"value": "0.56", "currency": "USD"},
                    "volumeNum": "4800",
                    "liquidityNum": "26200",
                }
            ],
        },
        {
            "slug": "spx-june-levels",
            "title": "What will S&P 500 (SPX) hit by end of June?",
            "active": True,
            "markets": [
                {
                    "slug": "spx-june-7300",
                    "question": "Will S&P 500 (SPX) hit $7,300 by end of June?",
                    "bestBid": "0.62",
                    "bestAsk": "0.66",
                }
            ],
        },
    ]
}


class PolymarketDiscoveryProbeTests(unittest.TestCase):
    def test_classifies_directional_and_strike_candidates(self):
        self.assertEqual(
            "directional_up_down",
            classify_candidate("S&P 500 (SPX) Up or Down on April 27?"),
        )
        self.assertEqual(
            "strike_or_level",
            classify_candidate("Will S&P 500 (SPX) hit $7,300 by end of June?"),
        )
        self.assertEqual([7300.0], extract_level_candidates("SPX hit $7,300"))
        self.assertTrue(is_index_related("S&P 500 (SPX) Up or Down?"))
        self.assertFalse(is_index_related("Top Esports vs Weibo Gaming"))

    def test_extracts_candidates_from_search_payload(self):
        candidates = extract_market_candidates(SAMPLE_SEARCH_PAYLOAD, "SPX")

        self.assertEqual(2, len(candidates))
        self.assertEqual("spx-up-or-down-apr-27", candidates[0]["market_slug"])
        self.assertEqual("directional_up_down", candidates[0]["kind"])
        self.assertTrue(candidates[0]["index_related"])
        self.assertEqual(0.54, candidates[0]["best_bid"])
        self.assertEqual("strike_or_level", candidates[1]["kind"])
        self.assertEqual([7300.0], candidates[1]["levels"])

    def test_client_uses_public_search_endpoint(self):
        session = FakeSession(SAMPLE_SEARCH_PAYLOAD)
        client = PolymarketDiscoveryClient(
            base_url="https://gateway.polymarket.us",
            session=session,
            timeout=3.0,
        )

        result = client.discover(queries=("SPX",), limit=5)

        self.assertEqual(2, result["candidate_count"])
        self.assertEqual(2, result["raw_candidate_count"])
        self.assertEqual([], result["errors"])
        self.assertEqual("https://gateway.polymarket.us/v1/search", session.calls[0]["url"])
        self.assertEqual({"query": "SPX", "limit": 5}, session.calls[0]["params"])
        self.assertEqual(3.0, session.calls[0]["timeout"])


if __name__ == "__main__":
    unittest.main()
