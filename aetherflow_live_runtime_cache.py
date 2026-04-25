from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from aetherflow_base_cache import (
    load_manifold_engine_state_snapshot,
    metadata_path_for_parquet,
    validate_full_manifold_base_features_path,
)
from aetherflow_features import BASE_FEATURE_COLUMNS
from config import CONFIG
from manifold_strategy_features import (
    build_training_feature_frame as build_manifold_base_frame,
    build_training_feature_frame_with_state,
)


ET = "US/Eastern"
SOURCE_COLUMNS = ["open", "high", "low", "close", "volume", "symbol"]


def _resolve_path(path_like: Optional[str | os.PathLike]) -> Optional[Path]:
    if path_like is None:
        return None
    raw = str(path_like).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path.resolve()


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{int(time.time() * 1000)}.tmp"
    )
    df.to_parquet(tmp_path, index=True)
    tmp_path.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.{int(time.time() * 1000)}.tmp"
    )
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _normalize_source_frame(df: pd.DataFrame, *, default_symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=SOURCE_COLUMNS)
    work = df.copy()
    work.columns = [str(col).lower() for col in work.columns]
    for col in ("open", "high", "low", "close"):
        if col not in work.columns:
            raise ValueError(f"Live source frame missing required column: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "volume" not in work.columns:
        work["volume"] = 0
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0).astype("int64")
    if "symbol" not in work.columns:
        work["symbol"] = str(default_symbol or "ES")
    else:
        work["symbol"] = work["symbol"].astype(str).replace("", str(default_symbol or "ES"))
    work = work.dropna(subset=["open", "high", "low", "close"])
    work.index = pd.DatetimeIndex(work.index)
    if work.index.tz is None:
        work.index = work.index.tz_localize(ET)
    else:
        work.index = work.index.tz_convert(ET)
    work.index.name = "timestamp"
    work = work.loc[:, SOURCE_COLUMNS]
    work = (
        work.reset_index()
        .sort_values(["timestamp", "symbol"], kind="stable")
        .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
        .set_index("timestamp")
        .sort_index(kind="stable")
    )
    return work


def _normalize_history_tail(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    work = df.copy()
    work.columns = [str(col).lower() for col in work.columns]
    for col in ("open", "high", "low", "close"):
        if col not in work.columns:
            raise ValueError(f"Live history tail missing required column: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "volume" not in work.columns:
        work["volume"] = 0
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0).astype("int64")
    work = work.dropna(subset=["open", "high", "low", "close"])
    work.index = pd.DatetimeIndex(work.index)
    if work.index.tz is None:
        work.index = work.index.tz_localize(ET)
    else:
        work.index = work.index.tz_convert(ET)
    work.index.name = "timestamp"
    work = work.loc[:, ["open", "high", "low", "close", "volume"]]
    work = work.loc[~work.index.duplicated(keep="last")].sort_index(kind="stable")
    return work


def _combine_source_frames(existing_df: pd.DataFrame, incoming_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df is None or existing_df.empty:
        return incoming_df.copy()
    if incoming_df is None or incoming_df.empty:
        return existing_df.copy()
    existing = existing_df.reset_index()
    incoming = incoming_df.reset_index()
    existing["_source_rank"] = 0
    incoming["_source_rank"] = 1
    combined = pd.concat([existing, incoming], axis=0, ignore_index=True)
    combined = combined.sort_values(["timestamp", "symbol", "_source_rank"], kind="stable")
    combined = combined.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    combined = combined.drop(columns=["_source_rank"], errors="ignore")
    combined = combined.set_index("timestamp").sort_index(kind="stable")
    combined.index = pd.DatetimeIndex(combined.index)
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize(ET)
    else:
        combined.index = combined.index.tz_convert(ET)
    combined.index.name = "timestamp"
    return combined.loc[:, SOURCE_COLUMNS]


def _empty_base_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=sorted(set(BASE_FEATURE_COLUMNS)))


class AetherFlowRuntimeCacheUpdater:
    """
    Persist live bars and a small manifold tail overlay off the hot path.

    The live strategy keeps trading from in-memory data. This worker simply
    mirrors the latest completed bars to disk and extends a tail-only manifold
    overlay so restarts can seed from base + overlay without a full rebuild.
    """

    def __init__(
        self,
        *,
        source_path: Optional[str | os.PathLike],
        source_symbol: str,
        base_features_path: Optional[str | os.PathLike],
        source_overlay_path: Optional[str | os.PathLike],
        manifold_overlay_path: Optional[str | os.PathLike],
        flush_seconds: float = 30.0,
        source_compact_seconds: float = 900.0,
        manifold_min_new_bars: int = 1,
        manifold_overlap_bars: int = 1440,
        history_tail_bars: int = 10080,
    ) -> None:
        self.source_path = _resolve_path(source_path)
        self.source_symbol = str(source_symbol or "ES")
        self.base_features_path = _resolve_path(base_features_path)
        self.source_overlay_path = _resolve_path(source_overlay_path)
        self.manifold_overlay_path = _resolve_path(manifold_overlay_path)
        self.flush_seconds = max(5.0, float(flush_seconds or 30.0))
        self.source_compact_seconds = max(0.0, float(source_compact_seconds or 0.0))
        self.manifold_min_new_bars = max(1, int(manifold_min_new_bars or 1))
        self.manifold_overlap_bars = max(256, int(manifold_overlap_bars or 1440))
        self.history_tail_bars = max(self.manifold_overlap_bars + 64, int(history_tail_bars or 10080))

        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending_source_frames: list[pd.DataFrame] = []
        self._pending_history_tail: Optional[pd.DataFrame] = None

        self._base_features_end: Optional[pd.Timestamp] = None
        self._manifold_overlay_validated = False
        self._last_source_compact_monotonic = time.monotonic()
        self._last_logged_insufficient_tail: Optional[pd.Timestamp] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="AetherFlowRuntimeCacheUpdater",
            daemon=True,
        )
        self._thread.start()
        logging.info(
            "AetherFlow runtime cache updater started: source_overlay=%s manifold_overlay=%s flush_seconds=%.1f",
            self.source_overlay_path,
            self.manifold_overlay_path,
            self.flush_seconds,
        )

    def stop(self, timeout: float = 15.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, float(timeout or 15.0)))
            self._thread = None

    def enqueue(
        self,
        *,
        recent_bars: Optional[pd.DataFrame],
        history_tail: Optional[pd.DataFrame],
    ) -> None:
        source_frame = _normalize_source_frame(recent_bars, default_symbol=self.source_symbol)
        tail_frame = _normalize_history_tail(history_tail)
        if source_frame.empty and tail_frame.empty:
            return
        with self._lock:
            if not source_frame.empty:
                self._pending_source_frames.append(source_frame)
            if not tail_frame.empty:
                self._pending_history_tail = tail_frame
        self._wake_event.set()

    def flush_now(self) -> None:
        source_batch, history_tail = self._drain_pending()
        self._flush_batches(source_batch=source_batch, history_tail=history_tail)

    def _run(self) -> None:
        while True:
            self._wake_event.wait(timeout=self.flush_seconds)
            self._wake_event.clear()
            self.flush_now()
            if self._stop_event.is_set():
                break

    def _drain_pending(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        with self._lock:
            source_frames = self._pending_source_frames
            history_tail = self._pending_history_tail
            self._pending_source_frames = []
            self._pending_history_tail = None
        source_batch = (
            _combine_source_frames(pd.DataFrame(columns=SOURCE_COLUMNS), pd.concat(source_frames, axis=0))
            if source_frames
            else pd.DataFrame(columns=SOURCE_COLUMNS)
        )
        history_tail = history_tail if isinstance(history_tail, pd.DataFrame) else pd.DataFrame()
        return source_batch, history_tail

    def _flush_batches(self, *, source_batch: pd.DataFrame, history_tail: pd.DataFrame) -> None:
        if not source_batch.empty:
            self._persist_source_overlay(source_batch)
        if not history_tail.empty:
            self._persist_manifold_overlay(history_tail)
        if self.source_compact_seconds > 0.0:
            self._maybe_compact_source_overlay()

    def _read_source_overlay(self) -> pd.DataFrame:
        path = self.source_overlay_path
        if path is None or not path.exists():
            return pd.DataFrame(columns=SOURCE_COLUMNS)
        try:
            overlay = pd.read_parquet(path)
        except Exception as exc:
            logging.warning("AetherFlow runtime source overlay read failed: %s", exc)
            return pd.DataFrame(columns=SOURCE_COLUMNS)
        if overlay.empty:
            return pd.DataFrame(columns=SOURCE_COLUMNS)
        return _normalize_source_frame(overlay, default_symbol=self.source_symbol)

    def _persist_source_overlay(self, source_batch: pd.DataFrame) -> None:
        path = self.source_overlay_path
        if path is None or source_batch.empty:
            return
        try:
            overlay = self._read_source_overlay()
            combined = _combine_source_frames(overlay, source_batch)
            _atomic_write_parquet(combined, path)
            logging.debug(
                "AetherFlow runtime source overlay updated: rows=%s range=%s->%s",
                int(len(combined)),
                combined.index.min(),
                combined.index.max(),
            )
        except Exception as exc:
            logging.warning("AetherFlow runtime source overlay write failed: %s", exc)

    def _load_base_features_end(self) -> Optional[pd.Timestamp]:
        if self._base_features_end is not None:
            return self._base_features_end
        path = self.base_features_path
        if path is None or not path.exists():
            return None
        try:
            validate_full_manifold_base_features_path(path, BASE_FEATURE_COLUMNS)
            base_index = pd.read_parquet(path, columns=[])
            if len(base_index.index) <= 0:
                return None
            self._base_features_end = pd.Timestamp(base_index.index.max())
            return self._base_features_end
        except Exception as exc:
            logging.warning("AetherFlow runtime base manifold inspection failed: %s", exc)
            return None

    def _read_manifold_overlay(self) -> pd.DataFrame:
        path = self.manifold_overlay_path
        if path is None or not path.exists():
            return _empty_base_frame()
        try:
            if not self._manifold_overlay_validated:
                validate_full_manifold_base_features_path(path, BASE_FEATURE_COLUMNS)
                self._manifold_overlay_validated = True
            overlay = pd.read_parquet(path)
        except Exception as exc:
            logging.warning("AetherFlow runtime manifold overlay read failed: %s", exc)
            return _empty_base_frame()
        if overlay.empty:
            return _empty_base_frame()
        overlay.index = pd.DatetimeIndex(overlay.index)
        return (
            overlay.sort_index(kind="stable")
            .reindex(columns=sorted(set(BASE_FEATURE_COLUMNS)))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    def _load_exact_snapshot(self, path: Optional[Path]) -> dict:
        if path is None or not path.exists():
            return {}
        snapshot = load_manifold_engine_state_snapshot(path)
        if str(snapshot.get("continuation_mode", "") or "") != "engine_state_exact":
            return {}
        if not isinstance(snapshot.get("state"), dict) or not snapshot.get("state"):
            return {}
        if snapshot.get("end_time") is None:
            return {}
        return snapshot

    def _write_manifold_overlay_metadata(
        self,
        *,
        path: Path,
        combined: pd.DataFrame,
        base_end: pd.Timestamp,
        continuation_mode: str,
        lookback_bars: Optional[int] = None,
        final_state: Optional[dict] = None,
        continuation_anchor_time: Optional[pd.Timestamp] = None,
    ) -> None:
        payload = {
            "kind": "aetherflow_live_runtime_manifold_overlay",
            "base_features_path": str(self.base_features_path) if self.base_features_path is not None else None,
            "base_features_end": pd.Timestamp(base_end).isoformat(),
            "overlay_path": str(path),
            "overlay_rows": int(len(combined)),
            "overlay_range": {
                "start": pd.Timestamp(combined.index.min()).isoformat(),
                "end": pd.Timestamp(combined.index.max()).isoformat(),
            },
            "mode": "base_plus_overlay_tail_only",
            "continuation_mode": str(continuation_mode or ""),
            "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        if continuation_anchor_time is not None:
            payload["continuation_anchor_time"] = pd.Timestamp(continuation_anchor_time).isoformat()
        if continuation_mode == "engine_state_exact" and isinstance(final_state, dict) and final_state:
            payload["engine_state_end"] = pd.Timestamp(combined.index.max()).isoformat()
            payload["engine_lookback_bars"] = int(lookback_bars or 0)
            payload["engine_state"] = final_state
        _atomic_write_json(payload, metadata_path_for_parquet(path))

    def _persist_manifold_overlay(self, history_tail: pd.DataFrame) -> None:
        base_end = self._load_base_features_end()
        path = self.manifold_overlay_path
        if base_end is None or path is None or history_tail.empty:
            return

        history_tail = history_tail.tail(self.history_tail_bars)
        if history_tail.index.max() <= base_end:
            return

        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg["enabled"] = True
        existing_overlay = self._read_manifold_overlay()
        exact_overlay_snapshot = self._load_exact_snapshot(path)
        exact_base_snapshot = self._load_exact_snapshot(self.base_features_path)

        exact_anchor_snapshot = {}
        exact_anchor_ts: Optional[pd.Timestamp] = None
        exact_prefix = _empty_base_frame()
        if exact_overlay_snapshot:
            exact_anchor_ts = pd.Timestamp(exact_overlay_snapshot["end_time"])
            if exact_anchor_ts >= base_end:
                exact_anchor_snapshot = exact_overlay_snapshot
                exact_prefix = existing_overlay.copy()
        if not exact_anchor_snapshot and exact_base_snapshot:
            exact_anchor_ts = pd.Timestamp(exact_base_snapshot["end_time"])
            if exact_anchor_ts == base_end:
                exact_anchor_snapshot = exact_base_snapshot
                exact_prefix = _empty_base_frame()

        if exact_anchor_snapshot:
            anchor_ts = pd.Timestamp(exact_anchor_snapshot["end_time"])
            if history_tail.index.max() > anchor_ts:
                exact_new_bar_count = int((history_tail.index > anchor_ts).sum())
                if exact_new_bar_count < self.manifold_min_new_bars:
                    return
                anchor_pos = int(history_tail.index.searchsorted(anchor_ts, side="left"))
                required_lookback = max(
                    1,
                    int(exact_anchor_snapshot.get("lookback_bars", 0) or 0),
                    self.manifold_overlap_bars,
                )
                has_anchor_row = (
                    anchor_pos < len(history_tail.index)
                    and pd.Timestamp(history_tail.index[anchor_pos]) == anchor_ts
                )
                has_required_history = bool(has_anchor_row) and (anchor_pos + 1) >= required_lookback
                if has_required_history:
                    try:
                        exact_tail, final_state, lookback_bars = build_training_feature_frame_with_state(
                            history_tail,
                            manifold_cfg=manifold_cfg,
                            log_every=0,
                            initial_state=exact_anchor_snapshot["state"],
                            start_after=anchor_ts,
                        )
                    except Exception as exc:
                        logging.warning("AetherFlow runtime exact manifold continuation failed: %s", exc)
                    else:
                        exact_tail = (
                            exact_tail.sort_index(kind="stable")
                            .reindex(columns=sorted(set(BASE_FEATURE_COLUMNS)))
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0.0)
                        )
                        if not exact_tail.empty:
                            combined = pd.concat([exact_prefix, exact_tail], axis=0).sort_index(kind="stable")
                            combined = combined.loc[combined.index > base_end]
                            combined = combined.loc[~combined.index.duplicated(keep="last")]
                            combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                            try:
                                _atomic_write_parquet(combined, path)
                                self._write_manifold_overlay_metadata(
                                    path=path,
                                    combined=combined,
                                    base_end=base_end,
                                    continuation_mode="engine_state_exact",
                                    lookback_bars=lookback_bars,
                                    final_state=final_state,
                                    continuation_anchor_time=anchor_ts,
                                )
                                self._manifold_overlay_validated = False
                                logging.info(
                                    "AetherFlow runtime manifold overlay updated (exact): rows=%s range=%s->%s",
                                    int(len(combined)),
                                    combined.index.min(),
                                    combined.index.max(),
                                )
                                return
                            except Exception as exc:
                                logging.warning("AetherFlow runtime manifold overlay exact write failed: %s", exc)
                else:
                    logging.info(
                        "AetherFlow runtime exact overlay waiting for anchor/lookback coverage: anchor=%s tail_start=%s tail_end=%s required_lookback=%s",
                        anchor_ts,
                        history_tail.index.min(),
                        history_tail.index.max(),
                        required_lookback,
                    )

        overlay_end = pd.Timestamp(existing_overlay.index.max()) if not existing_overlay.empty else None
        anchor_ts = overlay_end if overlay_end is not None else base_end
        if history_tail.index.max() <= anchor_ts:
            return
        if history_tail.index.min() > anchor_ts:
            if self._last_logged_insufficient_tail != history_tail.index.max():
                logging.info(
                    "AetherFlow runtime manifold overlay waiting for deeper tail history: anchor=%s tail_start=%s tail_end=%s",
                    anchor_ts,
                    history_tail.index.min(),
                    history_tail.index.max(),
                )
                self._last_logged_insufficient_tail = history_tail.index.max()
            return

        new_bar_count = int((history_tail.index > anchor_ts).sum())
        if new_bar_count < self.manifold_min_new_bars:
            return

        anchor_pos = int(history_tail.index.searchsorted(anchor_ts, side="left"))
        build_start_pos = max(0, anchor_pos - self.manifold_overlap_bars)
        build_source = history_tail.iloc[build_start_pos:]
        if build_source.empty:
            return

        try:
            rebuilt = build_manifold_base_frame(build_source, manifold_cfg=manifold_cfg, log_every=0)
        except Exception as exc:
            logging.warning("AetherFlow runtime manifold overlay rebuild failed: %s", exc)
            return
        if rebuilt.empty:
            return

        rebuilt = (
            rebuilt.sort_index(kind="stable")
            .reindex(columns=sorted(set(BASE_FEATURE_COLUMNS)))
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        rebuilt_tail = rebuilt.loc[rebuilt.index > base_end]
        if rebuilt_tail.empty:
            return

        prefix = existing_overlay.loc[existing_overlay.index < rebuilt_tail.index.min()] if not existing_overlay.empty else _empty_base_frame()
        combined = pd.concat([prefix, rebuilt_tail], axis=0).sort_index(kind="stable")
        combined = combined.loc[~combined.index.duplicated(keep="last")]
        combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        try:
            _atomic_write_parquet(combined, path)
            self._write_manifold_overlay_metadata(
                path=path,
                combined=combined,
                base_end=base_end,
                continuation_mode="overlap_rebuild_fallback",
                continuation_anchor_time=anchor_ts,
            )
            self._manifold_overlay_validated = False
            logging.info(
                "AetherFlow runtime manifold overlay updated (fallback): rows=%s range=%s->%s",
                int(len(combined)),
                combined.index.min(),
                combined.index.max(),
            )
        except Exception as exc:
            logging.warning("AetherFlow runtime manifold overlay write failed: %s", exc)

    def _maybe_compact_source_overlay(self) -> None:
        path = self.source_overlay_path
        source_path = self.source_path
        if path is None or source_path is None or not path.exists() or not source_path.exists():
            return
        now_mono = time.monotonic()
        if (now_mono - self._last_source_compact_monotonic) < self.source_compact_seconds:
            return
        self._last_source_compact_monotonic = now_mono
        try:
            overlay = self._read_source_overlay()
            if overlay.empty:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
                return
            existing = pd.read_parquet(source_path)
            existing = _normalize_source_frame(existing, default_symbol=self.source_symbol)
            combined = _combine_source_frames(existing, overlay)
            _atomic_write_parquet(combined, source_path)
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            logging.info(
                "AetherFlow runtime source compaction complete: path=%s rows=%s range=%s->%s",
                source_path,
                int(len(combined)),
                combined.index.min(),
                combined.index.max(),
            )
        except Exception as exc:
            logging.warning("AetherFlow runtime source compaction failed: %s", exc)
