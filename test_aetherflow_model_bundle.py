import unittest

import numpy as np
import pandas as pd

from aetherflow_model_bundle import (
    bundle_conditional_models_metadata,
    bundle_feature_columns,
    make_routed_ensemble_bundle,
    normalize_model_bundle,
    predict_bundle_probabilities,
)


class ConstantProbaModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, frame):
        p = np.full(len(frame), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


class SequenceProbaModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, frame):
        p = np.resize(self.probabilities, len(frame)).astype(float)
        return np.column_stack([1.0 - p, p])


class ConstantRouterModel:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, frame):
        p = np.full(len(frame), self.probability, dtype=float)
        return np.column_stack([1.0 - p, p])


class AetherFlowModelBundleTests(unittest.TestCase):
    def test_conditional_models_respect_match_sides(self) -> None:
        features = pd.DataFrame(
            {
                "setup_family": ["transition_burst", "transition_burst"],
                "candidate_side": [1.0, -1.0],
                "session_id": [2.0, 2.0],
                "manifold_regime_id": [1.0, 1.0],
            },
            index=pd.date_range("2026-01-01 09:30:00", periods=2, freq="1min"),
        )
        bundle = {
            "model": ConstantProbaModel(0.1),
            "feature_columns": ["candidate_side"],
            "conditional_models": [
                {
                    "family_name": "transition_burst",
                    "model": ConstantProbaModel(0.8),
                    "feature_columns": ["candidate_side"],
                    "match_sides": ["LONG"],
                },
                {
                    "family_name": "transition_burst",
                    "model": ConstantProbaModel(0.3),
                    "feature_columns": ["candidate_side"],
                    "match_sides": ["SHORT"],
                },
            ],
        }

        probs = predict_bundle_probabilities(bundle, features)
        np.testing.assert_allclose(probs, np.asarray([0.8, 0.3]), atol=1e-12)

        metadata = bundle_conditional_models_metadata(normalize_model_bundle(bundle))
        self.assertEqual(metadata[0]["match_sides"], ["LONG"])
        self.assertEqual(metadata[1]["match_sides"], ["SHORT"])

    def test_bundle_feature_columns_preserve_phase_extras(self) -> None:
        bundle = {
            "model": ConstantProbaModel(0.5),
            "feature_columns": ["flow_fast", "phase_regime_run_bars"],
        }
        self.assertIn("phase_regime_run_bars", bundle_feature_columns(bundle))

    def test_routed_ensemble_soft_blends_experts(self) -> None:
        features = pd.DataFrame(
            {
                "setup_family": ["aligned_flow", "aligned_flow"],
                "candidate_side": [1.0, -1.0],
                "session_id": [2.0, 3.0],
                "manifold_regime_id": [2.0, 2.0],
            },
            index=pd.date_range("2026-01-01 09:30:00", periods=2, freq="1min"),
        )
        bundle = make_routed_ensemble_bundle(
            experts=[
                {"name": "stable", "bundle": {"model": ConstantProbaModel(0.2), "feature_columns": ["candidate_side"]}},
                {"name": "adaptive", "bundle": {"model": ConstantProbaModel(0.8), "feature_columns": ["candidate_side"]}},
            ],
            router_model=ConstantRouterModel(0.75),
            router_feature_columns=["expert_0_prob", "expert_1_prob"],
            threshold=0.55,
            router_mode="soft_blend",
            router_weight_floor=0.0,
            router_weight_ceiling=1.0,
        )

        probs = predict_bundle_probabilities(bundle, features)
        np.testing.assert_allclose(probs, np.asarray([0.65, 0.65]), atol=1e-12)

    def test_routed_ensemble_override_min_prob_falls_back_to_reference(self) -> None:
        features = pd.DataFrame(
            {
                "setup_family": ["aligned_flow", "aligned_flow"],
                "candidate_side": [1.0, 1.0],
                "session_id": [2.0, 2.0],
                "manifold_regime_id": [0.0, 0.0],
            },
            index=pd.date_range("2026-01-01 09:30:00", periods=2, freq="1min"),
        )
        bundle = make_routed_ensemble_bundle(
            experts=[
                {"name": "stable", "bundle": {"model": ConstantProbaModel(0.2), "feature_columns": ["candidate_side"]}},
                {
                    "name": "adaptive",
                    "bundle": {"model": SequenceProbaModel([0.5, 0.7]), "feature_columns": ["candidate_side"]},
                    "override": {"enabled": True, "reference_expert": "stable", "min_prob": 0.6},
                },
            ],
            router_model=ConstantRouterModel(0.9),
            router_feature_columns=["expert_0_prob", "expert_1_prob"],
            threshold=0.55,
            router_mode="hard_route",
            router_fallback_expert="stable",
        )

        probs = predict_bundle_probabilities(bundle, features)
        np.testing.assert_allclose(probs, np.asarray([0.2, 0.7]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
