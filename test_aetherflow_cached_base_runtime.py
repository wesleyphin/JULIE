import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from aetherflow_features import BASE_FEATURE_COLUMNS
from aetherflow_strategy import AetherFlowStrategy
from config import CONFIG
from manifold_strategy_features import build_training_feature_frame


class AetherFlowCachedBaseRuntimeTests(unittest.TestCase):
    @staticmethod
    def _market_frame(rows: int = 720) -> pd.DataFrame:
        idx = pd.date_range("2026-04-20 00:00:00", periods=rows, freq="1min", tz="US/Eastern")
        t = np.arange(rows, dtype=float)
        close = 5000.0 + np.cumsum(0.18 + (0.35 * np.sin(t / 27.0)) + (0.08 * np.cos(t / 11.0)))
        open_ = np.concatenate([[close[0] - 0.12], close[:-1]])
        high = np.maximum(open_, close) + 0.35
        low = np.minimum(open_, close) - 0.35
        volume = 100 + ((t.astype(int) * 7) % 40)
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )

    def test_seeded_live_base_window_extends_cached_base_tail(self) -> None:
        raw = self._market_frame()
        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg["enabled"] = True
        full_base = build_training_feature_frame(raw, manifold_cfg=manifold_cfg, log_every=0)
        self.assertFalse(full_base.empty)

        seed_cutoff = full_base.index[-180]
        seed_base = full_base.loc[full_base.index <= seed_cutoff]
        self.assertFalse(seed_base.empty)

        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_path = Path(tmp_dir) / "seed_base.parquet"
            seed_base.to_parquet(seed_path, index=True)

            with mock.patch.object(AetherFlowStrategy, "_load_artifacts", lambda self: None):
                strategy = AetherFlowStrategy()

            strategy._live_base_features_path = seed_path
            strategy._live_base_features_validated = False
            strategy.live_base_history_bars = 320
            strategy.live_base_overlap_bars = 240
            strategy._live_base_window = None

            window = strategy._seeded_live_base_window(raw, end_time=raw.index[-1])
            self.assertFalse(window.empty)
            self.assertEqual(window.index.max(), full_base.index.max())
            self.assertLessEqual(len(window), strategy.live_base_history_bars)
            self.assertTrue(window.index.is_monotonic_increasing)
            self.assertFalse(window.index.duplicated().any())

            cached_prefix = window.loc[window.index <= seed_cutoff].reindex(columns=BASE_FEATURE_COLUMNS)
            expected_prefix = seed_base.reindex(columns=BASE_FEATURE_COLUMNS).tail(len(cached_prefix))
            self.assertFalse(cached_prefix.empty)
            pd.testing.assert_frame_equal(cached_prefix, expected_prefix, check_exact=False, atol=1e-9, rtol=1e-9)

            rebuilt_tail = window.loc[window.index > seed_cutoff]
            self.assertFalse(rebuilt_tail.empty)
            self.assertGreaterEqual(len(rebuilt_tail), 1)


if __name__ == "__main__":
    unittest.main()
