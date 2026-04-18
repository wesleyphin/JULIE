import datetime
import unittest

import pandas as pd

import kalshi_trade_overlay


class _KalshiProviderStub:
    def __init__(self, rows, *, distance_es=8.0):
        self.enabled = True
        self.is_healthy = True
        self._rows = list(rows)
        self._distance_es = float(distance_es)

    def get_relative_markets_for_ui(self, es_prices=None, window_size=120):
        _ = es_prices, window_size
        return list(self._rows)

    def get_sentiment(self, es_price):
        _ = es_price
        return {"distance_es": self._distance_es}

    def get_sentiment_momentum(self, es_price, lookback=3):
        _ = es_price, lookback
        return 0.02


class TestKalshiTradeOverlay(unittest.TestCase):
    def test_build_trade_plan_uses_probe_probability_and_fade_strike(self):
        provider = _KalshiProviderStub(
            [
                {"strike_es": 100.0, "probability": 0.78},
                {"strike_es": 105.0, "probability": 0.72},
                {"strike_es": 110.0, "probability": 0.49},
                {"strike_es": 115.0, "probability": 0.28},
                {"strike_es": 120.0, "probability": 0.18},
                {"strike_es": 125.0, "probability": 0.11},
                {"strike_es": 130.0, "probability": 0.07},
                {"strike_es": 135.0, "probability": 0.04},
            ]
        )
        plan = kalshi_trade_overlay.build_trade_plan(
            {"side": "LONG", "tp_dist": 20.0, "entry_price": 100.0},
            100.0,
            provider,
            price_action_profile={"role": "forward_primary", "mode": "outrageous", "forward_weight": 0.78},
            tick_size=0.25,
        )
        self.assertTrue(plan["applied"])
        self.assertAlmostEqual(plan["entry_probability"], 0.78, places=4)
        self.assertAlmostEqual(plan["probe_price"], 105.0, places=2)
        self.assertAlmostEqual(plan["probe_probability"], 0.72, places=4)
        self.assertAlmostEqual(plan["momentum_delta"], -0.06, places=4)
        self.assertAlmostEqual(plan["momentum_retention"], 0.9231, places=4)
        self.assertEqual(plan["fade_reason"], "adjacent_drop")
        self.assertAlmostEqual(plan["anchor_price"], 105.0, places=2)
        self.assertAlmostEqual(plan["trail_trigger_price"], 105.0, places=2)

    def test_build_trade_plan_blocks_weak_forward_primary_entry(self):
        provider = _KalshiProviderStub(
            [
                {"strike_es": 100.0, "probability": 0.58},
                {"strike_es": 105.0, "probability": 0.38},
                {"strike_es": 110.0, "probability": 0.18},
                {"strike_es": 115.0, "probability": 0.08},
                {"strike_es": 120.0, "probability": 0.05},
                {"strike_es": 125.0, "probability": 0.03},
                {"strike_es": 130.0, "probability": 0.02},
                {"strike_es": 135.0, "probability": 0.01},
            ],
            distance_es=-6.0,
        )
        plan = kalshi_trade_overlay.build_trade_plan(
            {"side": "LONG", "tp_dist": 12.0, "entry_price": 100.0},
            100.0,
            provider,
            price_action_profile={"role": "forward_primary", "mode": "outrageous", "forward_weight": 0.78},
            tick_size=0.25,
        )
        self.assertTrue(plan["applied"])
        self.assertTrue(plan["entry_blocked"])
        self.assertLess(plan["momentum_retention"], 0.80)

    def test_compute_tp_trail_stop_locks_breached_fade_strike(self):
        trail = kalshi_trade_overlay.compute_tp_trail_stop(
            {
                "side": "LONG",
                "kalshi_tp_trail_enabled": True,
                "kalshi_tp_trigger_price": 105.0,
                "kalshi_tp_anchor_price": 105.0,
                "kalshi_tp_trail_buffer_ticks": 4,
                "current_stop_price": 100.0,
            },
            market_price=106.0,
            bar_high=106.25,
            bar_low=104.75,
            tick_size=0.25,
        )
        self.assertTrue(trail["triggered"])
        self.assertTrue(trail["should_update"])
        self.assertAlmostEqual(trail["stop_price"], 104.0)

    def test_recent_price_action_marks_outrageous_when_tape_is_unstable(self):
        start = pd.Timestamp("2026-04-01 18:00:00", tz="US/Eastern")
        rows = []
        price = 100.0
        for day in range(10):
            base = start + pd.Timedelta(days=day)
            for minute in range(60):
                ts = base + pd.Timedelta(minutes=minute)
                swing = 3.0 if minute % 2 == 0 else -3.5
                open_price = price
                close_price = price + swing
                high_price = max(open_price, close_price) + 70.0
                low_price = min(open_price, close_price) - 70.0
                rows.append(
                    {
                        "ts": ts,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                    }
                )
                price = close_price
        df = pd.DataFrame(rows).set_index("ts")
        profile = kalshi_trade_overlay.analyze_recent_price_action(df)
        self.assertEqual(profile["role"], "forward_primary")
        self.assertIn(profile["mode"], {"outrageous", "chop_outrageous"})


if __name__ == "__main__":
    unittest.main()
