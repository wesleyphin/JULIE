import datetime
import unittest

from level_fill_optimizer import FILL_IMMEDIATE, LevelFillOptimizer
import julie001


class _StructuralTrackerStub:
    def __init__(self, levels):
        self._levels = list(levels)

    def get_active_levels(self):
        return list(self._levels)


class _StopUpdateClientStub:
    def __init__(self):
        self.calls = []
        self._active_stop_order_id = 777001

    def modify_stop_to_breakeven(
        self,
        *,
        stop_price,
        side,
        known_size,
        stop_order_id=None,
        current_stop_price=None,
    ):
        self.calls.append(
            {
                "stop_price": float(stop_price),
                "side": str(side),
                "known_size": int(known_size),
                "stop_order_id": stop_order_id,
                "current_stop_price": float(current_stop_price),
            }
        )
        return True


class TestLiveFillAndTrailing(unittest.TestCase):
    def test_level_fill_rejects_wait_when_level_is_beyond_protected_drift(self):
        optimizer = LevelFillOptimizer()
        tracker = _StructuralTrackerStub(
            [
                {
                    "price": 100.5,
                    "name": "TestSupport",
                    "type": "structural",
                    "priority": 4,
                }
            ]
        )
        decision = optimizer.evaluate(
            {"side": "LONG", "sl_dist": 1.0, "strategy": "DynamicEngine3"},
            current_price=102.5,
            structural_tracker=tracker,
        )
        self.assertEqual(decision["mode"], FILL_IMMEDIATE)
        self.assertIn("beyond protected drift", decision["reason"])

    def test_level_fill_fires_only_when_touch_closes_near_target(self):
        optimizer = LevelFillOptimizer()
        optimizer.add_pending(
            "lf-near",
            {"side": "LONG", "sl_dist": 4.0, "strategy": "DynamicEngine3"},
            {
                "target_price": 100.0,
                "target_name": "Q1L_NY",
                "dist": 1.0,
                "max_bars": 3,
            },
            current_price=101.0,
        )
        result = optimizer.check_pending(
            "lf-near",
            {"high": 101.25, "low": 100.20, "close": 100.50},
        )
        self.assertTrue(result["fire"])
        self.assertFalse(result["abort"])

    def test_level_fill_touched_intrabar_but_closed_away_stays_pending(self):
        optimizer = LevelFillOptimizer()
        optimizer.add_pending(
            "lf-far",
            {"side": "LONG", "sl_dist": 4.0, "strategy": "DynamicEngine3"},
            {
                "target_price": 100.0,
                "target_name": "Q1L_NY",
                "dist": 1.0,
                "max_bars": 3,
            },
            current_price=101.0,
        )
        result = optimizer.check_pending(
            "lf-far",
            {"high": 101.50, "low": 100.10, "close": 101.20},
        )
        self.assertFalse(result["fire"])
        self.assertFalse(result["abort"])
        self.assertIn("touched intrabar", result["reason"])

    def test_live_level_fill_market_guard_blocks_drifted_execution(self):
        allowed, market_distance = julie001._level_fill_live_execution_allowed(
            {"target_price": 100.0},
            current_price=101.0,
        )
        self.assertFalse(allowed)
        self.assertAlmostEqual(market_distance, 1.0)

    def test_live_pivot_trail_candidate_requires_confirmed_post_entry_pivot(self):
        trade = {
            "side": "LONG",
            "entry_price": 100.0,
            "current_stop_price": 95.0,
            "entry_bar": 10,
        }
        self.assertIsNone(
            julie001._live_pivot_trail_candidate(
                trade,
                pivot_high=130.0,
                pivot_low=None,
                pivot_bar_index=10,
            )
        )
        candidate = julie001._live_pivot_trail_candidate(
            trade,
            pivot_high=130.0,
            pivot_low=None,
            pivot_bar_index=11,
        )
        self.assertAlmostEqual(candidate, 112.25)

    def test_apply_pivot_trail_sl_updates_live_stop(self):
        trade = {
            "strategy": "DynamicEngine3",
            "side": "LONG",
            "size": 1,
            "entry_price": 100.0,
            "current_stop_price": 95.0,
            "current_target_price": 140.0,
            "sl_dist": 5.0,
            "tp_dist": 40.0,
            "stop_order_id": 321,
            "de3_break_even_applied": False,
            "de3_break_even_move_count": 0,
        }
        client = _StopUpdateClientStub()
        result = julie001._apply_pivot_trail_sl(
            client,
            trade,
            112.25,
            current_time=datetime.datetime(2026, 4, 17, 12, 0, tzinfo=datetime.timezone.utc),
            market_price=120.0,
            bar_high=121.0,
            bar_low=119.0,
            bar_index=42,
        )
        self.assertEqual(result["status"], "updated")
        self.assertEqual(len(client.calls), 1)
        self.assertAlmostEqual(client.calls[0]["stop_price"], 112.25)
        self.assertEqual(client.calls[0]["stop_order_id"], 321)
        self.assertAlmostEqual(trade["current_stop_price"], 112.25)
        self.assertTrue(trade["de3_break_even_applied"])
        self.assertEqual(trade["stop_order_id"], 777001)


if __name__ == "__main__":
    unittest.main()
