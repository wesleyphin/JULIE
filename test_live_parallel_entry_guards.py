import unittest

from julie001 import _allow_same_side_parallel_entry, _find_live_trade_for_strategy


class LiveParallelEntryGuardTests(unittest.TestCase):
    def test_de3_can_still_coexist_with_single_regimeadaptive_trade(self) -> None:
        de3_trade = {"strategy": "DynamicEngine3", "side": "LONG"}
        signal = {"strategy": "RegimeAdaptive", "side": "LONG"}

        allowed = _allow_same_side_parallel_entry(
            de3_trade,
            signal,
            tracked_live_trades=[de3_trade],
        )

        self.assertTrue(allowed)

    def test_same_strategy_cannot_open_second_parallel_trade(self) -> None:
        de3_trade = {"strategy": "DynamicEngine3", "side": "LONG"}
        regime_trade = {"strategy": "RegimeAdaptive", "side": "LONG"}
        signal = {"strategy": "RegimeAdaptive", "side": "LONG"}

        allowed = _allow_same_side_parallel_entry(
            de3_trade,
            signal,
            tracked_live_trades=[de3_trade, regime_trade],
        )

        self.assertFalse(allowed)

    def test_aetherflow_same_side_add_is_blocked_once_one_leg_is_active(self) -> None:
        aetherflow_trade = {"strategy": "AetherFlowStrategy", "side": "LONG"}
        signal = {"strategy": "AetherFlowStrategy", "side": "LONG"}

        allowed = _allow_same_side_parallel_entry(
            aetherflow_trade,
            signal,
            tracked_live_trades=[aetherflow_trade],
        )

        self.assertFalse(allowed)

    def test_same_strategy_opposite_side_signal_is_not_blocked_by_same_side_duplicate_guard(self) -> None:
        regime_trade = {"strategy": "RegimeAdaptive", "side": "LONG"}
        signal = {"strategy": "RegimeAdaptive", "side": "SHORT"}

        any_side_match = _find_live_trade_for_strategy(
            [regime_trade],
            signal,
        )
        same_side_match = _find_live_trade_for_strategy(
            [regime_trade],
            signal,
            require_same_side=True,
        )

        self.assertIs(any_side_match, regime_trade)
        self.assertIsNone(same_side_match)


if __name__ == "__main__":
    unittest.main()
