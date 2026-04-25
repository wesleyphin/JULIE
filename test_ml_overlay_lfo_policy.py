import os
import unittest

import ml_overlay_shadow as mls


class MlOverlayLfoPolicyTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            "JULIE_ML_LFO_ACTIVE": os.environ.get("JULIE_ML_LFO_ACTIVE"),
            "JULIE_LFO_POLICY_DE3": os.environ.get("JULIE_LFO_POLICY_DE3"),
            "JULIE_LFO_POLICY_REGIMEADAPTIVE": os.environ.get("JULIE_LFO_POLICY_REGIMEADAPTIVE"),
            "JULIE_LFO_POLICY_AETHERFLOW": os.environ.get("JULIE_LFO_POLICY_AETHERFLOW"),
            "JULIE_LFO_POLICY_MLPHYSICS": os.environ.get("JULIE_LFO_POLICY_MLPHYSICS"),
        }
        for key in self._saved:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_default_live_policy_matches_promoted_strategy_mix(self):
        self.assertEqual(mls.get_lfo_live_policy("DynamicEngine3"), "ml")
        self.assertEqual(mls.get_lfo_live_policy("RegimeAdaptive"), "rule")
        self.assertEqual(mls.get_lfo_live_policy("AetherFlowStrategy"), "off")
        self.assertEqual(mls.get_lfo_live_policy("MLPhysics_US"), "off")

    def test_family_override_takes_priority(self):
        os.environ["JULIE_LFO_POLICY_AETHERFLOW"] = "ml"
        self.assertEqual(mls.get_lfo_live_policy("AetherFlowStrategy"), "ml")

    def test_legacy_global_toggle_preserves_hybrid_behavior(self):
        os.environ["JULIE_ML_LFO_ACTIVE"] = "1"
        self.assertEqual(mls.get_lfo_live_policy("DynamicEngine3"), "hybrid")
        self.assertEqual(mls.get_lfo_live_policy("AetherFlowStrategy"), "hybrid")

    def test_build_ml_wait_decision_uses_next_bank_level_in_trade_direction(self):
        long_decision = mls.build_ml_wait_decision({"side": "LONG"}, 5000.0, p_wait=0.8, threshold=0.6)
        short_decision = mls.build_ml_wait_decision({"side": "SHORT"}, 5000.0)

        self.assertIsNotNone(long_decision)
        self.assertIsNotNone(short_decision)
        self.assertEqual(long_decision["mode"], "WAIT")
        self.assertEqual(short_decision["mode"], "WAIT")
        self.assertAlmostEqual(long_decision["target_price"], 4987.5)
        self.assertAlmostEqual(short_decision["target_price"], 5012.5)
        self.assertIn("p_wait=0.800>=0.600", long_decision["reason"])


if __name__ == "__main__":
    unittest.main()
