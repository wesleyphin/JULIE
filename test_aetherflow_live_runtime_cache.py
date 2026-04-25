import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from aetherflow_base_cache import metadata_path_for_parquet
from aetherflow_features import BASE_FEATURE_COLUMNS
from aetherflow_live_runtime_cache import AetherFlowRuntimeCacheUpdater
from aetherflow_strategy import AetherFlowStrategy
from config import CONFIG
from manifold_strategy_features import build_training_feature_frame, build_training_feature_frame_with_state


class AetherFlowLiveRuntimeCacheTests(unittest.TestCase):
    @staticmethod
    def _market_frame(rows: int = 720) -> pd.DataFrame:
        idx = pd.date_range("2026-04-22 00:00:00", periods=rows, freq="1min", tz="US/Eastern")
        t = np.arange(rows, dtype=float)
        close = 5025.0 + np.cumsum(0.22 + (0.28 * np.sin(t / 19.0)) + (0.06 * np.cos(t / 7.0)))
        open_ = np.concatenate([[close[0] - 0.15], close[:-1]])
        high = np.maximum(open_, close) + 0.30
        low = np.minimum(open_, close) - 0.30
        volume = 120 + ((t.astype(int) * 11) % 55)
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

    def test_runtime_cache_updater_writes_source_and_manifold_overlays(self) -> None:
        raw = self._market_frame()
        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg["enabled"] = True

        base_raw = raw.iloc[:-90].copy()
        base_features, base_state, lookback_bars = build_training_feature_frame_with_state(
            base_raw,
            manifold_cfg=manifold_cfg,
            log_every=0,
        )
        self.assertFalse(base_features.empty)
        full_features = build_training_feature_frame(raw, manifold_cfg=manifold_cfg, log_every=0)
        expected_tail = (
            full_features.loc[full_features.index > base_features.index.max()]
            .reindex(columns=sorted(set(BASE_FEATURE_COLUMNS)))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        self.assertFalse(expected_tail.empty)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            source_path = tmp_root / "es_master_outrights.parquet"
            source_frame = base_raw.assign(symbol="ESM6")
            source_frame.to_parquet(source_path, index=True)

            base_path = tmp_root / "manifold_base.parquet"
            base_features.to_parquet(base_path, index=True)
            metadata_path_for_parquet(base_path).write_text(
                json.dumps(
                    {
                        "continuation_mode": "engine_state_exact",
                        "engine_state_end": pd.Timestamp(base_features.index.max()).isoformat(),
                        "engine_lookback_bars": int(lookback_bars),
                        "engine_state": base_state,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            source_overlay_path = tmp_root / "runtime_source_tail.parquet"
            manifold_overlay_path = tmp_root / "runtime_manifold_tail.parquet"

            updater = AetherFlowRuntimeCacheUpdater(
                source_path=source_path,
                source_symbol="ESM6",
                base_features_path=base_path,
                source_overlay_path=source_overlay_path,
                manifold_overlay_path=manifold_overlay_path,
                flush_seconds=5,
                source_compact_seconds=0,
                manifold_min_new_bars=1,
                manifold_overlap_bars=180,
                history_tail_bars=720,
            )

            updater.enqueue(
                recent_bars=raw.iloc[-20:].copy(),
                history_tail=raw.copy(),
            )
            updater.flush_now()

            self.assertTrue(source_overlay_path.exists())
            source_overlay = pd.read_parquet(source_overlay_path)
            self.assertFalse(source_overlay.empty)
            self.assertEqual(source_overlay.index.max(), raw.index.max())
            self.assertIn("symbol", source_overlay.columns)
            self.assertEqual(set(source_overlay["symbol"].astype(str)), {"ESM6"})

            self.assertTrue(manifold_overlay_path.exists())
            manifold_overlay = pd.read_parquet(manifold_overlay_path)
            self.assertFalse(manifold_overlay.empty)
            self.assertEqual(manifold_overlay.index.max(), raw.index.max())
            self.assertTrue((manifold_overlay.index > base_features.index.max()).all())
            self.assertTrue(set(BASE_FEATURE_COLUMNS).issubset(set(manifold_overlay.columns)))
            manifold_overlay = (
                manifold_overlay.reindex(columns=sorted(set(BASE_FEATURE_COLUMNS)))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            pd.testing.assert_frame_equal(manifold_overlay, expected_tail, check_exact=False, atol=1e-9, rtol=1e-9)

            overlay_meta = json.loads(metadata_path_for_parquet(manifold_overlay_path).read_text(encoding="utf-8"))
            self.assertEqual(overlay_meta.get("continuation_mode"), "engine_state_exact")

    def test_strategy_live_seed_slice_combines_base_and_runtime_overlay(self) -> None:
        raw = self._market_frame(rows=540)
        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg["enabled"] = True
        full_base = build_training_feature_frame(raw, manifold_cfg=manifold_cfg, log_every=0)
        self.assertFalse(full_base.empty)

        split_idx = -60
        base = full_base.iloc[:split_idx].copy()
        overlay = full_base.iloc[split_idx:].copy()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "manifold_base.parquet"
            overlay_path = tmp_root / "runtime_manifold_tail.parquet"
            base.to_parquet(base_path, index=True)
            overlay.to_parquet(overlay_path, index=True)

            with mock.patch.object(AetherFlowStrategy, "_load_artifacts", lambda self: None):
                strategy = AetherFlowStrategy()

            strategy._live_base_features_path = base_path
            strategy._live_base_overlay_path = overlay_path
            strategy._live_base_features_validated = False
            strategy._live_base_overlay_validated = False

            start_time = full_base.index[-120]
            end_time = full_base.index[-1]
            combined = strategy._load_live_base_seed_slice(start_time=start_time, end_time=end_time)
            expected = (
                full_base.loc[(full_base.index >= start_time) & (full_base.index <= end_time)]
                .reindex(columns=sorted(set(BASE_FEATURE_COLUMNS)))
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            self.assertFalse(combined.empty)
            self.assertEqual(combined.index.max(), expected.index.max())
            pd.testing.assert_frame_equal(combined, expected, check_exact=False, atol=1e-9, rtol=1e-9)

    def test_strategy_live_seed_slice_preserves_original_base_when_overlay_overlaps(self) -> None:
        raw = self._market_frame(rows=540)
        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg["enabled"] = True
        full_base = build_training_feature_frame(raw, manifold_cfg=manifold_cfg, log_every=0)
        self.assertFalse(full_base.empty)

        split_idx = -80
        base = full_base.iloc[:split_idx].copy()
        overlay = full_base.iloc[split_idx:].copy()
        overlap_ts = base.index[-1]
        overlay.loc[overlap_ts] = 999.0
        overlay = overlay.sort_index()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            base_path = tmp_root / "manifold_base.parquet"
            overlay_path = tmp_root / "runtime_manifold_tail.parquet"
            base.to_parquet(base_path, index=True)
            overlay.to_parquet(overlay_path, index=True)

            with mock.patch.object(AetherFlowStrategy, "_load_artifacts", lambda self: None):
                strategy = AetherFlowStrategy()

            strategy._live_base_features_path = base_path
            strategy._live_base_overlay_path = overlay_path
            strategy._live_base_features_validated = False
            strategy._live_base_overlay_validated = False

            combined = strategy._load_live_base_seed_slice(
                start_time=base.index[-120],
                end_time=full_base.index[-1],
            )

            self.assertAlmostEqual(
                float(combined.loc[overlap_ts, "manifold_R"]),
                float(base.loc[overlap_ts, "manifold_R"]),
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
