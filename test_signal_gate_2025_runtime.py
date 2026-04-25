import unittest

import numpy as np

import signal_gate_2025 as sg


class _CaptureModel:
    def __init__(self, p_big_loss=0.6):
        self.p_big_loss = float(p_big_loss)
        self.last_X = None

    def predict_proba(self, X):
        self.last_X = X.copy()
        n = len(X)
        p1 = np.full(n, self.p_big_loss, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


class SignalGate2025RuntimeTests(unittest.TestCase):
    def test_score_with_gate_encodes_local_and_market_regime_features(self):
        model = _CaptureModel(0.42)
        payload = {
            "model": model,
            "feature_names": [
                "side__LONG",
                "side__SHORT",
                "regime__TREND_GEODESIC",
                "mkt_regime__whipsaw",
                "session__NY",
                "et_hour",
            ],
            "numeric_features": [],
            "categorical_maps": {
                "side": ["LONG", "SHORT"],
                "regime": ["TREND_GEODESIC"],
                "mkt_regime": ["whipsaw"],
                "session": ["NY"],
            },
        }

        p = sg._score_with_gate(
            payload,
            side="long",
            regime="trend_geodesic",
            mkt_regime="WHIPSAW",
            et_hour=10,
            bar_features={},
        )

        self.assertAlmostEqual(p, 0.42)
        self.assertIsNotNone(model.last_X)
        row = model.last_X.iloc[0].to_dict()
        self.assertEqual(row["side__LONG"], 1)
        self.assertEqual(row["side__SHORT"], 0)
        self.assertEqual(row["regime__TREND_GEODESIC"], 1)
        self.assertEqual(row["mkt_regime__whipsaw"], 1)
        self.assertEqual(row["session__NY"], 1)
        self.assertEqual(row["et_hour"], 10.0)

    def test_should_veto_signal_uses_market_regime_feature_without_dynamic_threshold(self):
        prev_gates = dict(sg._GATES)
        prev_dynamic = sg._DYNAMIC_THRESHOLD_ENABLED
        try:
            model = _CaptureModel(0.55)
            payload = {
                "model": model,
                "feature_names": [
                    "side__LONG",
                    "mkt_regime__calm_trend",
                    "session__NY",
                    "et_hour",
                ],
                "numeric_features": [],
                "categorical_maps": {
                    "side": ["LONG"],
                    "mkt_regime": ["calm_trend"],
                    "session": ["NY"],
                },
                "veto_threshold": 0.50,
            }
            sg._GATES = {"de3": payload}
            sg._DYNAMIC_THRESHOLD_ENABLED = False

            veto, reason = sg.should_veto_signal(
                side="LONG",
                regime="",
                et_hour=11,
                bar_features={},
                strategy="DynamicEngine3",
                mkt_regime="calm_trend",
                cum_day_pnl=500.0,
            )

            self.assertTrue(veto)
            self.assertIn("0.500", reason)
            row = model.last_X.iloc[0].to_dict()
            self.assertEqual(row["mkt_regime__calm_trend"], 1)
        finally:
            sg._GATES = prev_gates
            sg._DYNAMIC_THRESHOLD_ENABLED = prev_dynamic

    def test_effective_threshold_respects_feature_only_default(self):
        prev_dynamic = sg._DYNAMIC_THRESHOLD_ENABLED
        try:
            sg._DYNAMIC_THRESHOLD_ENABLED = False
            eff, mult = sg._effective_threshold(0.65, "whipsaw", -500.0)
            self.assertEqual(eff, 0.65)
            self.assertEqual(mult, 1.0)
        finally:
            sg._DYNAMIC_THRESHOLD_ENABLED = prev_dynamic


if __name__ == "__main__":
    unittest.main()
