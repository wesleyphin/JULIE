from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from aetherflow_base_cache import (
    load_manifold_engine_state_snapshot,
    metadata_path_for_parquet,
)
from config import CONFIG
from manifold_strategy_features import (
    EXPORT_COLUMNS,
    build_training_feature_frame_with_state,
)


class AetherFlowRuntimeCacheUpdater:
    """Maintain live source/manifold overlay parquet files for AetherFlow.

    The canonical base cache remains immutable during live trading. This updater
    writes only the runtime tail that arrives after that base cache so consumers
    can combine ``base + overlay`` without rebuilding the full manifold.
    """

    def __init__(
        self,
        *,
        source_path: str | Path,
        source_symbol: str,
        base_features_path: str | Path,
        source_overlay_path: str | Path,
        manifold_overlay_path: str | Path,
        flush_seconds: float = 5.0,
        source_compact_seconds: float = 0.0,
        manifold_min_new_bars: int = 1,
        manifold_overlap_bars: int = 720,
        history_tail_bars: int = 1200,
    ) -> None:
        self.source_path = Path(source_path)
        self.source_symbol = str(source_symbol or "").strip()
        self.base_features_path = Path(base_features_path)
        self.source_overlay_path = Path(source_overlay_path)
        self.manifold_overlay_path = Path(manifold_overlay_path)
        self.flush_seconds = float(flush_seconds or 0.0)
        self.source_compact_seconds = float(source_compact_seconds or 0.0)
        self.manifold_min_new_bars = max(1, int(manifold_min_new_bars or 1))
        self.manifold_overlap_bars = max(1, int(manifold_overlap_bars or 1))
        self.history_tail_bars = max(self.manifold_overlap_bars, int(history_tail_bars or self.manifold_overlap_bars))
        self._pending_recent_bars: Optional[pd.DataFrame] = None
        self._pending_history_tail: Optional[pd.DataFrame] = None

    @staticmethod
    def _prepare_frame(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        out = frame.copy()
        out.index = pd.DatetimeIndex(out.index)
        out = out.sort_index()
        return out.loc[~out.index.duplicated(keep="last")]

    def enqueue(self, *, recent_bars: pd.DataFrame, history_tail: pd.DataFrame) -> None:
        self._pending_recent_bars = self._prepare_frame(recent_bars)
        history = self._prepare_frame(history_tail)
        if not history.empty:
            history = history.tail(self.history_tail_bars)
        self._pending_history_tail = history

    def flush_now(self) -> None:
        recent = self._prepare_frame(self._pending_recent_bars)
        history = self._prepare_frame(self._pending_history_tail)
        if not recent.empty:
            self._write_source_overlay(recent)
        if not history.empty:
            self._write_manifold_overlay(history)

    def _write_source_overlay(self, recent: pd.DataFrame) -> None:
        out = recent.copy()
        if self.source_symbol:
            out["symbol"] = self.source_symbol
        self.source_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(self.source_overlay_path, index=True)

    def _base_cache_end_time(self) -> Optional[pd.Timestamp]:
        snapshot = load_manifold_engine_state_snapshot(self.base_features_path)
        end_time = snapshot.get("end_time")
        if end_time is not None:
            return pd.Timestamp(end_time)
        if not self.base_features_path.exists():
            return None
        try:
            base = pd.read_parquet(self.base_features_path, columns=["manifold_R"])
        except Exception:
            base = pd.read_parquet(self.base_features_path)
        if base.empty:
            return None
        base.index = pd.DatetimeIndex(base.index)
        return pd.Timestamp(base.index.max())

    def _write_manifold_overlay(self, history: pd.DataFrame) -> None:
        base_end = self._base_cache_end_time()
        if base_end is None:
            return
        new_bars = history.loc[pd.DatetimeIndex(history.index) > base_end]
        if len(new_bars) < self.manifold_min_new_bars:
            return

        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg["enabled"] = True
        snapshot = load_manifold_engine_state_snapshot(self.base_features_path)
        initial_state = snapshot.get("state")
        continuation_mode = "overlap_rebuild_fallback"

        if (
            snapshot.get("continuation_mode") == "engine_state_exact"
            and isinstance(initial_state, dict)
            and snapshot.get("end_time") is not None
            and pd.Timestamp(snapshot.get("end_time")) == base_end
        ):
            overlay, final_state, lookback_bars = build_training_feature_frame_with_state(
                history,
                manifold_cfg=manifold_cfg,
                log_every=0,
                initial_state=initial_state,
                start_after=base_end,
            )
            continuation_mode = "engine_state_exact"
        else:
            overlay, final_state, lookback_bars = build_training_feature_frame_with_state(
                history,
                manifold_cfg=manifold_cfg,
                log_every=0,
            )
            overlay = overlay.loc[pd.DatetimeIndex(overlay.index) > base_end]

        if overlay.empty:
            return
        overlay = (
            overlay.reindex(columns=EXPORT_COLUMNS)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .sort_index()
        )
        overlay = overlay.loc[~overlay.index.duplicated(keep="last")]

        self.manifold_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.to_parquet(self.manifold_overlay_path, index=True)
        metadata = {
            "feature_kind": "manifold_runtime_overlay",
            "base_features_path": str(self.base_features_path),
            "source_path": str(self.source_path),
            "source_overlay_path": str(self.source_overlay_path),
            "output_path": str(self.manifold_overlay_path),
            "continuation_mode": continuation_mode,
            "base_end": base_end.isoformat(),
            "engine_state_end": pd.Timestamp(overlay.index.max()).isoformat(),
            "engine_lookback_bars": int(lookback_bars),
            "engine_state": final_state,
            "rows": int(len(overlay)),
            "columns": list(overlay.columns),
            "range": {
                "start": pd.Timestamp(overlay.index.min()).isoformat(),
                "end": pd.Timestamp(overlay.index.max()).isoformat(),
            },
            "built_at": pd.Timestamp.now("UTC").isoformat(),
        }
        metadata_path_for_parquet(self.manifold_overlay_path).write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
