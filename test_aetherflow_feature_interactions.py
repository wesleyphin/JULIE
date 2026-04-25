import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from aetherflow_features import BASE_FEATURE_COLUMNS, build_feature_frame, ensure_feature_columns
from aetherflow_strategy import AetherFlowStrategy, _context_block_reason, _normalize_family_policies


EXPECTED_AF_MANIFOLD = (
    "artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet"
)


class AetherFlowFeatureInteractionTests(unittest.TestCase):
    @staticmethod
    def _aligned_flow_base(rows: int = 80) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01 09:30:00", periods=rows, freq="1min", tz="US/Eastern")
        base = pd.DataFrame(0.0, index=idx, columns=BASE_FEATURE_COLUMNS)
        base["manifold_R"] = 1.0
        base["manifold_alignment"] = 1.0
        base["manifold_smoothness"] = 1.0
        base["manifold_stress"] = 0.1
        base["manifold_dispersion"] = 0.2
        base["manifold_risk_mult"] = 1.0
        base["manifold_regime_id"] = 2.0
        base["manifold_R_pct"] = 0.8
        base["manifold_alignment_pct"] = 0.8
        base["manifold_smoothness_pct"] = 0.8
        base["manifold_stress_pct"] = 0.1
        base["manifold_dispersion_pct"] = 0.2
        base["ret_1"] = 0.001
        base["ret_5"] = 0.0015
        base["ret_15"] = 0.0015
        base["ema_slope_20"] = 0.0015
        base["ema_spread"] = 0.0015
        base["atr14"] = 1.0
        base["atr14_z"] = -1.0
        base["range_z"] = -1.0
        base["vwap_dist_atr"] = 1.25
        base["hour_cos"] = 1.0
        base["session_id"] = 2.0
        return base

    def test_build_feature_frame_assigns_setup_flags_before_interactions(self) -> None:
        features = build_feature_frame(
            base_features=self._aligned_flow_base(),
            preferred_setup_families={"aligned_flow"},
        )
        self.assertFalse(features.empty)
        candidates = features.loc[features["setup_family"].astype(str) == "aligned_flow"]
        self.assertFalse(candidates.empty)
        self.assertGreater(float(candidates["directional_vwap_dist"].abs().max()), 0.0)
        self.assertGreater(float(candidates["aligned_flow_edge"].abs().max()), 0.0)
        self.assertGreater(float(candidates["aligned_flow_ny_dispersed_edge"].abs().max()), 0.0)

        ensured = ensure_feature_columns(features)
        for col in [
            "directional_vwap_dist",
            "aligned_flow_edge",
            "aligned_flow_ny_dispersed_edge",
            "aligned_flow_nypm_trend_edge",
        ]:
            diff = (
                pd.to_numeric(features[col], errors="coerce").fillna(0.0)
                - pd.to_numeric(ensured[col], errors="coerce").fillna(0.0)
            ).abs()
            self.assertLessEqual(float(diff.max()), 1e-6, col)

    def test_strategy_exposes_canonical_base_paths(self) -> None:
        expected = (Path(__file__).resolve().parent / EXPECTED_AF_MANIFOLD).resolve()
        with mock.patch.object(AetherFlowStrategy, "_load_artifacts", lambda self: None):
            strategy = AetherFlowStrategy()
        self.assertEqual(strategy.backtest_base_features_path, expected)
        self.assertEqual(strategy.live_base_features_path, expected)

    def test_strategy_builds_signals_from_cached_base_features(self) -> None:
        with mock.patch.object(AetherFlowStrategy, "_load_artifacts", lambda self: None):
            strategy = AetherFlowStrategy()
        strategy.model_loaded = True
        strategy.model_bundle = object()
        strategy.threshold = 0.1
        strategy.min_confidence = 0.0
        strategy.size = 1
        strategy.allowed_session_ids = {2}
        strategy.hazard_block_regimes = set()
        strategy.family_policies = _normalize_family_policies(
            {
                "aligned_flow": {
                    "threshold": 0.1,
                    "allowed_session_ids": [2],
                    "allowed_regimes": ["DISPERSED"],
                }
            }
        )
        strategy._compute_probabilities = lambda features: np.full(len(features), 0.99)

        base = self._aligned_flow_base()
        signals = strategy.build_backtest_df_from_base_features(
            base,
            start_time=base.index[10],
            end_time=base.index[-1],
        )

        self.assertFalse(signals.empty)
        self.assertTrue((signals["side"].astype(str) == "LONG").any())

    def test_family_policy_rules_match_and_block_by_side(self) -> None:
        with mock.patch.object(AetherFlowStrategy, "_load_artifacts", lambda self: None):
            strategy = AetherFlowStrategy()
        strategy.threshold = 0.2
        strategy.family_policies = _normalize_family_policies(
            {
                "aligned_flow": {
                    "threshold": 0.2,
                    "policy_rules": [
                        {
                            "name": "short_only",
                            "match_sides": ["SHORT"],
                            "threshold": 0.9,
                            "allowed_sides": ["SHORT"],
                        }
                    ],
                }
            }
        )

        long_policy = strategy._policy_for_family("aligned_flow", {"candidate_side": 1.0})
        short_policy = strategy._policy_for_family("aligned_flow", {"candidate_side": -1.0})

        self.assertEqual(float(long_policy["threshold"]), 0.2)
        self.assertEqual(float(short_policy["threshold"]), 0.9)
        self.assertEqual(_context_block_reason({"candidate_side": 1.0}, short_policy), "side_not_allowed")


if __name__ == "__main__":
    unittest.main()
