from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from config import CONFIG
from ml_physics_legacy_experiment_common import (
    ML_FEATURE_COLUMNS,
    build_feature_frame,
    build_sltp_arrays,
    extract_latest_feature_row,
    latest_sltp,
    normalize_ohlcv_frame,
    resample_ohlcv,
    resolve_artifact_path,
    session_from_hour,
)
from strategy_base import Strategy


class MLPhysicsLegacyExperimentStrategy(Strategy):
    """Isolated pre-dist MLPhysics experiment using the older joblib model family."""

    _GLOBAL_PRECOMPUTED_BACKTEST_DF: Optional[pd.DataFrame] = None

    def __init__(self):
        cfg = CONFIG.get("ML_PHYSICS_LEGACY_EXPERIMENT", {}) or {}
        artifact_dir_raw = str(cfg.get("artifact_dir", "artifacts/ml_physics_legacy_experiment") or "").strip()
        self.artifact_dir = Path(artifact_dir_raw).expanduser()
        if not self.artifact_dir.is_absolute():
            self.artifact_dir = Path(__file__).resolve().parent / self.artifact_dir
        self.thresholds_path = resolve_artifact_path(
            self.artifact_dir,
            str(cfg.get("thresholds_file", "ml_physics_thresholds.json") or "ml_physics_thresholds.json"),
        )
        self.metrics_path = resolve_artifact_path(
            self.artifact_dir,
            str(cfg.get("metrics_file", "ml_physics_metrics.json") or "ml_physics_metrics.json"),
        )
        self.default_timeframe_minutes = int(cfg.get("timeframe_minutes", 5) or 5)
        self.min_history_bars = int(cfg.get("min_history_bars", 200) or 200)
        self.strategy_name = str(cfg.get("sltp_strategy_name", "Generic") or "Generic")
        self.bar_alignment = str(CONFIG.get("ML_PHYSICS_BAR_ALIGNMENT", "open") or "open").strip().lower()
        self._last_feature_log_ts = 0.0
        self._feature_log_interval = 300.0
        self.last_eval = None
        self._precomputed_backtest_df = None
        self._precomputed_lookup: dict[int, dict] = {}

        self.thresholds = self._load_json(self.thresholds_path)
        self.metrics = self._load_json(self.metrics_path)
        self.models = self._load_models()
        self.model_loaded = any(model is not None for model in self.models.values())
        if isinstance(self.__class__._GLOBAL_PRECOMPUTED_BACKTEST_DF, pd.DataFrame):
            self.set_precomputed_backtest_df(self.__class__._GLOBAL_PRECOMPUTED_BACKTEST_DF)

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logging.warning("MLPhysicsLegacyExperiment: failed loading %s (%s)", path, exc)
            return {}

    def _model_path_for_session(self, session_name: str) -> Optional[Path]:
        entry = self.thresholds.get(session_name)
        if isinstance(entry, dict):
            model_file = str(entry.get("model_file", "") or entry.get("saved_model", "") or "").strip()
            if model_file:
                return resolve_artifact_path(self.artifact_dir, model_file)
        session_cfg = (CONFIG.get("SESSIONS", {}) or {}).get(session_name, {}) or {}
        model_file = str(session_cfg.get("MODEL_FILE", "") or "").strip()
        if model_file:
            return resolve_artifact_path(self.artifact_dir, model_file)
        return None

    def _load_models(self) -> dict[str, object]:
        models: dict[str, object] = {}
        for session_name in ("ASIA", "LONDON", "NY_AM", "NY_PM"):
            model_path = self._model_path_for_session(session_name)
            if model_path is None or not model_path.exists():
                models[session_name] = None
                continue
            try:
                models[session_name] = joblib.load(model_path)
                logging.info("MLPhysicsLegacyExperiment: loaded %s model from %s", session_name, model_path)
            except Exception as exc:
                logging.warning(
                    "MLPhysicsLegacyExperiment: failed loading %s model %s (%s)",
                    session_name,
                    model_path,
                    exc,
                )
                models[session_name] = None
        return models

    def _session_threshold(self, session_name: str) -> float:
        entry = self.thresholds.get(session_name)
        if isinstance(entry, dict):
            try:
                return float(entry.get("threshold", entry.get("selected_threshold", 0.55)) or 0.55)
            except Exception:
                return 0.55
        if entry is not None:
            try:
                return float(entry)
            except Exception:
                return 0.55
        session_cfg = (CONFIG.get("SESSIONS", {}) or {}).get(session_name, {}) or {}
        try:
            return float(session_cfg.get("THRESHOLD", 0.55) or 0.55)
        except Exception:
            return 0.55

    def _session_timeframe(self, session_name: str) -> int:
        entry = self.thresholds.get(session_name)
        if isinstance(entry, dict):
            try:
                tf = int(entry.get("timeframe_minutes", self.default_timeframe_minutes) or self.default_timeframe_minutes)
                if tf > 0:
                    return tf
            except Exception:
                pass
        return self.default_timeframe_minutes

    @classmethod
    def set_global_precomputed_backtest_df(cls, df: Optional[pd.DataFrame]) -> None:
        cls._GLOBAL_PRECOMPUTED_BACKTEST_DF = None if df is None else df.copy()

    @classmethod
    def clear_global_precomputed_backtest_df(cls) -> None:
        cls._GLOBAL_PRECOMPUTED_BACKTEST_DF = None

    def set_precomputed_backtest_df(self, df: Optional[pd.DataFrame]) -> None:
        self._precomputed_backtest_df = None if df is None else df.copy()
        self._precomputed_lookup = {}
        if not isinstance(self._precomputed_backtest_df, pd.DataFrame) or self._precomputed_backtest_df.empty:
            return
        rows = self._precomputed_backtest_df.to_dict("records")
        for ts, row in zip(pd.DatetimeIndex(self._precomputed_backtest_df.index), rows):
            self._precomputed_lookup[int(ts.value)] = row

    def _alignment_offset(self, timeframe_minutes: int) -> pd.Timedelta:
        tf = max(1, int(timeframe_minutes or 1))
        if self.bar_alignment == "close":
            return pd.Timedelta(minutes=max(tf - 1, 0))
        return pd.Timedelta(minutes=tf)

    def build_precomputed_backtest_df(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = normalize_ohlcv_frame(df)
        if normalized.empty or not self.model_loaded:
            return pd.DataFrame()

        resampled_cache: dict[int, pd.DataFrame] = {}
        features_cache: dict[int, pd.DataFrame] = {}
        sltp_cache: dict[int, tuple[pd.Series, pd.Series]] = {}
        signal_frames: list[pd.DataFrame] = []

        for session_name in ("ASIA", "LONDON", "NY_AM", "NY_PM"):
            model = self.models.get(session_name)
            if model is None:
                continue
            timeframe_minutes = self._session_timeframe(session_name)
            if timeframe_minutes not in resampled_cache:
                resampled = resample_ohlcv(normalized, timeframe_minutes)
                resampled_cache[timeframe_minutes] = resampled
                features_cache[timeframe_minutes] = build_feature_frame(resampled)
                sl_dist, tp_dist = None, None
                if not resampled.empty:
                    sl_arr, tp_arr = build_sltp_arrays(resampled, strategy_name=self.strategy_name)
                    sl_dist = pd.Series(sl_arr, index=resampled.index, dtype=float)
                    tp_dist = pd.Series(tp_arr, index=resampled.index, dtype=float)
                sltp_cache[timeframe_minutes] = (
                    pd.Series(dtype=float) if sl_dist is None else sl_dist,
                    pd.Series(dtype=float) if tp_dist is None else tp_dist,
                )

            features = features_cache.get(timeframe_minutes)
            if not isinstance(features, pd.DataFrame) or features.empty:
                continue
            sl_series, tp_series = sltp_cache.get(timeframe_minutes, (pd.Series(dtype=float), pd.Series(dtype=float)))
            sl_series = sl_series.reindex(features.index)
            tp_series = tp_series.reindex(features.index)
            eval_index = features.index + self._alignment_offset(timeframe_minutes)
            session_mask = np.array([session_from_hour(int(ts.hour)) == session_name for ts in eval_index], dtype=bool)
            if not np.any(session_mask):
                continue
            eval_index = eval_index[session_mask]
            session_features = features.loc[features.index[session_mask]]
            session_sl = sl_series.loc[session_features.index]
            session_tp = tp_series.loc[session_features.index]
            if hasattr(model, "feature_names_in_"):
                session_features = session_features.reindex(columns=model.feature_names_in_, fill_value=0.0)
            else:
                session_features = session_features.reindex(columns=ML_FEATURE_COLUMNS, fill_value=0.0)
            if session_features.empty:
                continue
            try:
                proba = model.predict_proba(session_features)[:, 1]
            except Exception as exc:
                logging.warning("MLPhysicsLegacyExperiment: precompute predict_proba failure for %s (%s)", session_name, exc)
                continue
            threshold = self._session_threshold(session_name)
            long_mask = proba >= threshold
            short_mask = proba <= (1.0 - threshold)
            take_mask = long_mask | short_mask
            if not np.any(take_mask):
                continue
            signal_df = pd.DataFrame(
                {
                    "strategy": f"MLPhysicsLegacy_{session_name}",
                    "side": np.where(long_mask, "LONG", "SHORT"),
                    "tp_dist": session_tp.to_numpy(dtype=float),
                    "sl_dist": session_sl.to_numpy(dtype=float),
                    "ml_confidence": np.where(long_mask, proba, 1.0 - proba),
                    "ml_threshold": float(threshold),
                    "ml_prob_up": proba,
                    "ml_prob_down": 1.0 - proba,
                    "ml_runtime_mode": "legacy_experiment",
                },
                index=pd.DatetimeIndex(eval_index),
            )
            signal_df = signal_df.loc[take_mask]
            signal_df = signal_df[signal_df.index.isin(normalized.index)]
            if not signal_df.empty:
                signal_frames.append(signal_df)

        if not signal_frames:
            return pd.DataFrame()
        out = pd.concat(signal_frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out

    def on_bar(self, df: pd.DataFrame, current_time=None) -> Optional[Dict]:
        self.last_eval = None
        if current_time is not None and self._precomputed_lookup:
            ts = pd.Timestamp(current_time)
            row = self._precomputed_lookup.get(int(ts.value))
            if row is None:
                self.last_eval = {
                    "timestamp": ts.isoformat(),
                    "decision": "no_signal",
                    "ml_runtime_mode": "legacy_experiment",
                }
                return None
            self.last_eval = {
                "timestamp": ts.isoformat(),
                "session": str(row.get("strategy", "")).split("_")[-1],
                "decision": f"signal_{str(row.get('side', '')).lower()}",
                "ml_runtime_mode": "legacy_experiment",
                "ml_confidence": float(row.get("ml_confidence", 0.0) or 0.0),
                "ml_threshold": float(row.get("ml_threshold", 0.0) or 0.0),
                "ml_prob_up": float(row.get("ml_prob_up", 0.0) or 0.0),
                "ml_prob_down": float(row.get("ml_prob_down", 0.0) or 0.0),
                "side": str(row.get("side", "") or ""),
            }
            return dict(row)
        hist_df = normalize_ohlcv_frame(df)
        if hist_df.empty:
            return None
        eval_time = current_time if current_time is not None else hist_df.index[-1]
        eval_time = pd.Timestamp(eval_time)
        if eval_time.tzinfo is None:
            eval_time = eval_time.tz_localize(hist_df.index.tz)
        else:
            eval_time = eval_time.tz_convert(hist_df.index.tz)
        session_name = session_from_hour(int(eval_time.hour))
        if session_name == "CLOSED":
            return None

        model = self.models.get(session_name)
        if model is None:
            return None

        timeframe_minutes = self._session_timeframe(session_name)
        if timeframe_minutes > 1:
            minute_of_day = int(eval_time.hour * 60 + eval_time.minute)
            if self.bar_alignment == "close":
                aligned = ((minute_of_day + 1) % timeframe_minutes) == 0
            else:
                aligned = (minute_of_day % timeframe_minutes) == 0
            if not aligned:
                return None
        if timeframe_minutes > 1:
            hist_df = resample_ohlcv(hist_df, timeframe_minutes)
            if self.bar_alignment == "open" and len(hist_df) > 1:
                hist_df = hist_df.iloc[:-1]
        if len(hist_df) < self.min_history_bars:
            return None

        features = extract_latest_feature_row(hist_df)
        if features is None or features.empty:
            return None

        if hasattr(model, "feature_names_in_"):
            features = features.reindex(columns=model.feature_names_in_, fill_value=0.0)
        else:
            features = features.reindex(columns=ML_FEATURE_COLUMNS, fill_value=0.0)

        try:
            proba = float(model.predict_proba(features)[0][1])
        except Exception as exc:
            logging.warning("MLPhysicsLegacyExperiment: predict_proba failure for %s (%s)", session_name, exc)
            return None

        threshold = self._session_threshold(session_name)
        sltp = latest_sltp(hist_df, strategy_name=self.strategy_name)
        if not np.isfinite(sltp.get("sl_dist", np.nan)) or not np.isfinite(sltp.get("tp_dist", np.nan)):
            return None

        self.last_eval = {
            "timestamp": eval_time.isoformat(),
            "session": session_name,
            "decision": "no_signal",
            "ml_runtime_mode": "legacy_experiment",
            "ml_confidence": float(proba),
            "ml_threshold": float(threshold),
            "ml_prob_up": float(proba),
            "ml_prob_down": float(1.0 - proba),
        }

        if proba >= threshold:
            self.last_eval.update({"decision": "signal_long", "side": "LONG"})
            return {
                "strategy": f"MLPhysicsLegacy_{session_name}",
                "side": "LONG",
                "tp_dist": float(sltp["tp_dist"]),
                "sl_dist": float(sltp["sl_dist"]),
                "ml_confidence": float(proba),
                "ml_threshold": float(threshold),
                "ml_prob_up": float(proba),
                "ml_prob_down": float(1.0 - proba),
                "ml_runtime_mode": "legacy_experiment",
            }

        if proba <= (1.0 - threshold):
            self.last_eval.update({"decision": "signal_short", "side": "SHORT"})
            return {
                "strategy": f"MLPhysicsLegacy_{session_name}",
                "side": "SHORT",
                "tp_dist": float(sltp["tp_dist"]),
                "sl_dist": float(sltp["sl_dist"]),
                "ml_confidence": float(1.0 - proba),
                "ml_threshold": float(threshold),
                "ml_prob_up": float(proba),
                "ml_prob_down": float(1.0 - proba),
                "ml_runtime_mode": "legacy_experiment",
            }

        return None
