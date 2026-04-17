import datetime
import unittest

from client import ProjectXClient
from julie001 import (
    _force_close_live_trade_for_crossed_stop,
    _has_live_direction_conflict,
    _should_apply_shared_consensus_bracket,
)


class LiveDirectionalConflictTests(unittest.TestCase):
    def test_mixed_candidate_directions_are_detected(self) -> None:
        has_conflict, counts = _has_live_direction_conflict(
            [
                (1, None, {"strategy": "DynamicEngine3", "side": "LONG"}, "DynamicEngine3"),
                (2, None, {"strategy": "RegimeAdaptive", "side": "SHORT"}, "RegimeAdaptive"),
            ]
        )

        self.assertTrue(has_conflict)
        self.assertEqual(counts["LONG"], 1)
        self.assertEqual(counts["SHORT"], 1)

    def test_shared_consensus_bracket_is_skipped_for_parallel_candidates(self) -> None:
        consensus_tp_signal = {"tp_dist": 10.0, "sl_dist": 5.0}

        self.assertFalse(
            _should_apply_shared_consensus_bracket(
                consensus_tp_signal,
                [
                    (1, None, {"side": "LONG"}, "DynamicEngine3"),
                    (2, None, {"side": "LONG"}, "RegimeAdaptive"),
                ],
            )
        )
        self.assertTrue(
            _should_apply_shared_consensus_bracket(
                consensus_tp_signal,
                [(1, None, {"side": "LONG"}, "DynamicEngine3")],
            )
        )


class BreakEvenBracketIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = ProjectXClient()
        self.client.account_id = 1
        self.client.contract_id = "TEST-CONTRACT"
        self.client._check_general_rate_limit = lambda: True

    def test_direct_stop_modify_does_not_run_global_stop_cleanup(self) -> None:
        self.client.modify_order = lambda order_id, stop_price=None, limit_price=None, size=None: True
        self.client._cleanup_duplicate_stop_orders = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate cleanup should not run for direct stop modify")
        )
        self.client._cancel_open_stop_orders = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global stop cleanup should not run for direct stop modify")
        )

        ok = self.client.modify_stop_to_breakeven(
            stop_price=5000.0,
            side="LONG",
            known_size=3,
            stop_order_id=12345,
            current_stop_price=4990.0,
        )

        self.assertTrue(ok)
        self.assertEqual(self.client._active_stop_order_id, 12345)

    def test_cancel_replace_only_cancels_the_matched_stop(self) -> None:
        order_state = {
            "orders": [
                {"orderId": 111, "type": 4, "side": 1, "size": 3, "stopPrice": 4990.0, "status": 1},
                {"orderId": 222, "type": 4, "side": 1, "size": 3, "stopPrice": 4980.0, "status": 1},
            ]
        }
        cancelled_order_ids: list[int] = []

        def get_cached_orders(*_args, **_kwargs):
            return list(order_state["orders"])

        def cancel_order(order_id: int) -> bool:
            cancelled_order_ids.append(int(order_id))
            order_state["orders"] = [
                order for order in order_state["orders"]
                if order.get("orderId") != int(order_id)
            ]
            return True

        def place_breakeven_stop(_be_price: float, _side: str, _size: int) -> bool:
            self.client._active_stop_order_id = 333
            return True

        self.client.modify_order = lambda order_id, stop_price=None, limit_price=None, size=None: False
        self.client._get_cached_orders = get_cached_orders
        self.client.cancel_order = cancel_order
        self.client._place_breakeven_stop = place_breakeven_stop
        self.client._cleanup_duplicate_stop_orders = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("duplicate cleanup should not run for cancel/replace fallback")
        )
        self.client._cancel_open_stop_orders = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("global stop cleanup should not run for cancel/replace fallback")
        )

        ok = self.client.modify_stop_to_breakeven(
            stop_price=5000.0,
            side="LONG",
            known_size=3,
            stop_order_id=None,
            current_stop_price=4990.0,
        )

        self.assertTrue(ok)
        self.assertEqual(cancelled_order_ids, [111])
        self.assertEqual(self.client._active_stop_order_id, 333)
        self.assertEqual(
            {order["orderId"] for order in order_state["orders"]},
            {222},
        )


class OrderSubmissionLockoutCooldownTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = ProjectXClient()
        self.client.account_id = 1
        self.client.contract_id = "TEST-CONTRACT"
        self.client._check_general_rate_limit = lambda: True

    def test_locked_out_rejection_arms_submission_cooldown(self) -> None:
        class FakeResponse:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {
                    "success": False,
                    "errorMessage": (
                        "Your account is locked out: Personal. "
                        "Please wait until your lockout expires."
                    ),
                }

        post_calls = []

        def fake_post(_url, json=None):
            post_calls.append(dict(json or {}))
            return FakeResponse()

        self.client.session.post = fake_post
        self.client._order_submission_lockout_cooldown_sec = 120.0

        signal = {
            "strategy": "DynamicEngine3",
            "side": "LONG",
            "size": 2,
            "sl_dist": 5.0,
            "tp_dist": 10.0,
        }

        first = self.client.place_order(signal, 5000.0)
        second = self.client.place_order(signal, 5000.0)

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(post_calls), 1)
        self.assertGreater(self.client._order_submission_lockout_until_ts, 0.0)
        self.assertIn("locked out", self.client._order_submission_lockout_reason.lower())


class MultiTradeManagementIsolationTests(unittest.TestCase):
    def test_crossed_stop_fallback_closes_only_the_target_leg_when_broker_is_still_open(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.close_trade_leg_calls = []
                self.close_position_calls = []
                self.cancel_open_exit_orders_calls = []
                self._last_close_order_details = {
                    "order_id": 456,
                    "exit_price": 4998.0,
                    "method": "partial_market_order",
                }

            def get_position(self):
                return {"side": "LONG", "size": 5, "avg_price": 5000.0, "stale": False}

            def close_trade_leg(self, trade):
                self.close_trade_leg_calls.append(dict(trade))
                return True

            def close_position(self, position):
                self.close_position_calls.append(dict(position))
                return True

            def cancel_open_exit_orders(self, side=None, reason=""):
                self.cancel_open_exit_orders_calls.append((side, reason))
                return 0

            def reconcile_trade_close(
                self,
                active_trade,
                *,
                exit_time=None,
                fallback_exit_price=None,
                close_order_id=None,
                point_value=5.0,
            ):
                return {
                    "source": "partial_market_order",
                    "entry_price": float(active_trade["entry_price"]),
                    "exit_price": float(fallback_exit_price),
                    "pnl_points": float(fallback_exit_price - active_trade["entry_price"]),
                    "pnl_dollars": float(
                        (fallback_exit_price - active_trade["entry_price"])
                        * point_value
                        * active_trade["size"]
                    ),
                    "order_id": close_order_id,
                    "exit_time": exit_time,
                }

        trade = {
            "strategy": "DynamicEngine3",
            "side": "LONG",
            "size": 2,
            "entry_price": 5000.0,
            "entry_time": datetime.datetime(2026, 4, 16, 9, 30, tzinfo=datetime.timezone.utc),
        }
        fake_client = FakeClient()

        result = _force_close_live_trade_for_crossed_stop(
            fake_client,
            trade,
            datetime.datetime(2026, 4, 16, 10, 0, tzinfo=datetime.timezone.utc),
            market_price=4998.0,
            target_stop_price=4999.0,
            failure_reason="unit_test_crossed_stop",
        )

        self.assertEqual(result["status"], "closed")
        self.assertEqual(len(fake_client.close_trade_leg_calls), 1)
        self.assertEqual(fake_client.close_trade_leg_calls[0]["strategy"], "DynamicEngine3")
        self.assertEqual(fake_client.close_position_calls, [])
        self.assertEqual(fake_client.cancel_open_exit_orders_calls, [])


if __name__ == "__main__":
    unittest.main()
