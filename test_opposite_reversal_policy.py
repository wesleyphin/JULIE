import unittest

from opposite_reversal_policy import (
    opposite_reversal_gate_reason,
    reversal_confirmation_state_is_confirmed,
    update_multi_family_reversal_consensus_state,
)


class OppositeReversalPolicyTests(unittest.TestCase):
    def test_disabled_policy_blocks_all_opposite_reversals(self) -> None:
        reason = opposite_reversal_gate_reason(
            {"side": "SHORT", "vol_regime": "high"},
            [{"side": "LONG", "trend_day_dir": "up"}],
            cfg={"enabled": False},
        )

        self.assertEqual(reason, "Opposite reversal disabled by config")

    def test_allowed_vol_regime_gate_blocks_non_matching_regime(self) -> None:
        reason = opposite_reversal_gate_reason(
            {"side": "SHORT", "vol_regime": "high"},
            [{"side": "LONG"}],
            cfg={"enabled": True, "allowed_vol_regimes": ["low", "normal"]},
        )

        self.assertEqual(reason, "Opposite reversal disabled in high vol")

    def test_countertrend_trend_day_gate_blocks_short_on_up_day(self) -> None:
        reason = opposite_reversal_gate_reason(
            {"side": "SHORT", "vol_regime": "normal", "trend_day_dir": "up"},
            [{"side": "LONG"}],
            cfg={"enabled": True, "block_countertrend_in_trend_day": True},
        )

        self.assertEqual(
            reason,
            "Opposite reversal blocked counter-trend on up trend day",
        )

    def test_policy_allows_signal_when_no_gate_matches(self) -> None:
        reason = opposite_reversal_gate_reason(
            {"side": "LONG", "vol_regime": "normal", "trend_day_dir": "up"},
            [{"side": "SHORT"}],
            cfg={
                "enabled": True,
                "allowed_vol_regimes": ["normal", "high"],
                "block_countertrend_in_trend_day": True,
            },
        )

        self.assertIsNone(reason)

    def test_prior_confirmation_state_respects_window_and_side(self) -> None:
        confirmed = reversal_confirmation_state_is_confirmed(
            {
                "count": 3,
                "side": "SHORT",
                "bar_index": 10,
            },
            signal_side="SHORT",
            current_bar_index=12,
            required_confirmations=3,
            window_bars=3,
        )
        expired = reversal_confirmation_state_is_confirmed(
            {
                "count": 3,
                "side": "SHORT",
                "bar_index": 10,
            },
            signal_side="SHORT",
            current_bar_index=15,
            required_confirmations=3,
            window_bars=3,
        )

        self.assertTrue(confirmed)
        self.assertFalse(expired)

    def test_multi_family_consensus_requires_all_active_families(self) -> None:
        state = {}
        confirmed, state, missing = update_multi_family_reversal_consensus_state(
            state,
            signal_side="SHORT",
            signal_family="de3",
            active_families=["de3", "regimeadaptive"],
            current_bar_index=20,
            window_bars=3,
        )
        self.assertFalse(confirmed)
        self.assertEqual(missing, ["regimeadaptive"])

        confirmed, state, missing = update_multi_family_reversal_consensus_state(
            state,
            signal_side="SHORT",
            signal_family="regimeadaptive",
            active_families=["de3", "regimeadaptive"],
            current_bar_index=21,
            window_bars=3,
        )
        self.assertTrue(confirmed)
        self.assertEqual(missing, [])

    def test_multi_family_consensus_resets_when_side_changes(self) -> None:
        _, state, _ = update_multi_family_reversal_consensus_state(
            {},
            signal_side="SHORT",
            signal_family="de3",
            active_families=["de3", "regimeadaptive"],
            current_bar_index=20,
            window_bars=3,
        )
        confirmed, state, missing = update_multi_family_reversal_consensus_state(
            state,
            signal_side="LONG",
            signal_family="de3",
            active_families=["de3", "regimeadaptive"],
            current_bar_index=21,
            window_bars=3,
        )

        self.assertFalse(confirmed)
        self.assertEqual(missing, ["regimeadaptive"])
        self.assertEqual(state.get("side"), "LONG")


if __name__ == "__main__":
    unittest.main()
