import json
import logging
import math
import warnings
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from config import CONFIG
from regime_manifold_engine import RegimeManifoldEngine
from strategy_base import Strategy
from manifold_confluence import apply_confluence_formula
from manifold_strategy_features import (
    AUX_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    REGIME_TO_ID,
    build_training_feature_frame,
    get_session_name,
    meta_to_feature_dict,
)

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


def _coerce_session_allowlist(value) -> Optional[set[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: set[int] = set()
    for item in items:
        try:
            out.add(int(float(item)))
        except Exception:
            continue
    return out if out else None


def _session_name_from_id(session_id: int) -> str:
    mapping = {
        -1: "OFF",
        0: "ASIA",
        1: "LONDON",
        2: "NY_AM",
        3: "NY_PM",
    }
    return mapping.get(int(session_id), f"SESSION_{int(session_id)}")


class _RollingWindowStats:
    """Rolling mean/std helper with O(1) push/update."""

    def __init__(self, maxlen: int):
        self.maxlen = max(2, int(maxlen))
        self.buf: Deque[float] = deque()
        self.total = 0.0
        self.total_sq = 0.0

    def push(self, value: float) -> None:
        x = float(value)
        if len(self.buf) >= self.maxlen:
            old = self.buf.popleft()
            self.total -= old
            self.total_sq -= old * old
        self.buf.append(x)
        self.total += x
        self.total_sq += x * x

    def zscore_last(self, min_periods: int = 40) -> float:
        n = len(self.buf)
        if n < max(2, int(min_periods)):
            return 0.0
        mean = self.total / float(n)
        var_num = self.total_sq - ((self.total * self.total) / float(n))
        var = var_num / float(max(1, n - 1))
        if not np.isfinite(var) or var <= 0.0:
            return 0.0
        std = math.sqrt(var)
        if std <= 0.0 or not np.isfinite(std):
            return 0.0
        z = (self.buf[-1] - mean) / std
        if not np.isfinite(z):
            return 0.0
        return float(z)


class _IncrementalAuxCache:
    """
    Append-only cache for live manifold aux features.
    Keeps per-bar updates O(1) instead of rebuilding feature history each call.
    """

    def __init__(self, max_rows: int = 2500, z_window: int = 200):
        self.max_rows = max(400, int(max_rows))
        self.z_window = max(50, int(z_window))
        self.min_z_periods = max(10, self.z_window // 5)
        self.alpha_atr = 1.0 / 14.0
        self.alpha_ema20 = 2.0 / (20.0 + 1.0)
        self.alpha_ema50 = 2.0 / (50.0 + 1.0)

        self.raw = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.aux = pd.DataFrame(columns=AUX_FEATURE_COLUMNS)
        self.last_ts: Optional[pd.Timestamp] = None

        self.prev_close: Optional[float] = None
        self.atr14: Optional[float] = None
        self.ema20: Optional[float] = None
        self.ema50: Optional[float] = None
        self.day_key: Optional[pd.Timestamp] = None
        self.day_cum_pv = 0.0
        self.day_cum_vol = 0.0

        self.close_hist: Deque[float] = deque(maxlen=16)
        self.ema20_hist: Deque[float] = deque(maxlen=6)
        self.atr_stats = _RollingWindowStats(self.z_window)
        self.range_stats = _RollingWindowStats(self.z_window)
        self.vol_stats = _RollingWindowStats(self.z_window)

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            out = float(value)
        except Exception:
            return float(default)
        if not np.isfinite(out):
            return float(default)
        return out

    @staticmethod
    def _session_id(ts: pd.Timestamp) -> float:
        session = get_session_name(ts)
        if session == "ASIA":
            return 0.0
        if session == "LONDON":
            return 1.0
        if session == "NY_AM":
            return 2.0
        if session == "NY_PM":
            return 3.0
        return -1.0

    @staticmethod
    def _pct_change(values: Deque[float], periods: int) -> float:
        p = int(periods)
        if p <= 0 or len(values) <= p:
            return 0.0
        curr = float(values[-1])
        prev = float(values[-(p + 1)])
        if prev == 0.0 or (not np.isfinite(curr)) or (not np.isfinite(prev)):
            return 0.0
        out = (curr / prev) - 1.0
        if not np.isfinite(out):
            return 0.0
        return float(out)

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        work = df.copy()
        work.columns = [str(c).lower() for c in work.columns]
        for col in ("open", "high", "low", "close"):
            if col not in work.columns:
                raise ValueError(f"Missing required column: {col}")
            work[col] = pd.to_numeric(work[col], errors="coerce")
        if "volume" not in work.columns:
            work["volume"] = 0.0
        work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
        work = work.dropna(subset=["open", "high", "low", "close"])
        work = work.sort_index()
        return work[["open", "high", "low", "close", "volume"]]

    def _append_state(
        self,
        ts: pd.Timestamp,
        open_px: float,
        high_px: float,
        low_px: float,
        close_px: float,
        volume: float,
    ) -> Dict[str, float]:
        ts = pd.Timestamp(ts)
        o = self._safe_float(open_px)
        h = self._safe_float(high_px, default=o)
        l = self._safe_float(low_px, default=o)
        c = self._safe_float(close_px, default=o)
        v = self._safe_float(volume, default=0.0)

        tr = abs(h - l)
        if self.prev_close is not None:
            tr = max(tr, abs(h - self.prev_close), abs(l - self.prev_close))
        if self.atr14 is None:
            self.atr14 = tr
        else:
            self.atr14 = ((1.0 - self.alpha_atr) * self.atr14) + (self.alpha_atr * tr)
        atr14 = self._safe_float(self.atr14)

        bar_range = abs(h - l)
        self.atr_stats.push(atr14)
        self.range_stats.push(bar_range)
        self.vol_stats.push(v)
        atr14_z = self.atr_stats.zscore_last(self.min_z_periods)
        range_z = self.range_stats.zscore_last(self.min_z_periods)
        vol_z = self.vol_stats.zscore_last(self.min_z_periods)

        day_key = (ts - pd.Timedelta(hours=18)).normalize()
        if self.day_key is None or day_key != self.day_key:
            self.day_key = day_key
            self.day_cum_pv = 0.0
            self.day_cum_vol = 0.0
        self.day_cum_pv += (c * v)
        self.day_cum_vol += v
        vwap = self.day_cum_pv / self.day_cum_vol if self.day_cum_vol > 0.0 else np.nan

        if self.ema20 is None:
            self.ema20 = c
        else:
            self.ema20 = ((1.0 - self.alpha_ema20) * self.ema20) + (self.alpha_ema20 * c)
        if self.ema50 is None:
            self.ema50 = c
        else:
            self.ema50 = ((1.0 - self.alpha_ema50) * self.ema50) + (self.alpha_ema50 * c)

        self.close_hist.append(c)
        ret_1 = self._pct_change(self.close_hist, 1)
        ret_5 = self._pct_change(self.close_hist, 5)
        ret_15 = self._pct_change(self.close_hist, 15)

        self.ema20_hist.append(float(self.ema20))
        ema_slope_20 = self._pct_change(self.ema20_hist, 5)
        ema_spread = ((self.ema20 - self.ema50) / c) if c != 0.0 else 0.0

        vwap_dist_atr = 0.0
        if atr14 > 0.0 and np.isfinite(vwap):
            vwap_dist_atr = (c - float(vwap)) / atr14
            if not np.isfinite(vwap_dist_atr):
                vwap_dist_atr = 0.0

        minutes = (float(ts.hour) * 60.0) + float(ts.minute)
        angle = (2.0 * math.pi * minutes) / 1440.0
        hour_sin = float(np.sin(angle))
        hour_cos = float(np.cos(angle))
        session_id = self._session_id(ts)

        self.prev_close = c

        return {
            "ret_1": float(ret_1),
            "ret_5": float(ret_5),
            "ret_15": float(ret_15),
            "atr14": float(atr14),
            "atr14_z": float(atr14_z),
            "range_z": float(range_z),
            "vol_z": float(vol_z),
            "vwap_dist_atr": float(vwap_dist_atr),
            "ema_spread": float(ema_spread if np.isfinite(ema_spread) else 0.0),
            "ema_slope_20": float(ema_slope_20),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "session_id": float(session_id),
        }

    def _trim_frames(self) -> None:
        if len(self.raw) > self.max_rows:
            self.raw = self.raw.iloc[-self.max_rows :].copy()
        if len(self.aux) > self.max_rows:
            self.aux = self.aux.iloc[-self.max_rows :].copy()

    def bootstrap(self, df: pd.DataFrame) -> None:
        work = self._normalize_ohlcv(df)
        if len(work) > self.max_rows:
            work = work.iloc[-self.max_rows :].copy()

        self.raw = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self.aux = pd.DataFrame(columns=AUX_FEATURE_COLUMNS)
        self.last_ts = None
        self.prev_close = None
        self.atr14 = None
        self.ema20 = None
        self.ema50 = None
        self.day_key = None
        self.day_cum_pv = 0.0
        self.day_cum_vol = 0.0
        self.close_hist.clear()
        self.ema20_hist.clear()
        self.atr_stats = _RollingWindowStats(self.z_window)
        self.range_stats = _RollingWindowStats(self.z_window)
        self.vol_stats = _RollingWindowStats(self.z_window)

        if work.empty:
            return

        aux_rows = []
        for row in work.itertuples(index=True):
            ts = pd.Timestamp(row.Index)
            feat = self._append_state(
                ts=ts,
                open_px=row.open,
                high_px=row.high,
                low_px=row.low,
                close_px=row.close,
                volume=row.volume,
            )
            aux_rows.append(feat)
            self.last_ts = ts

        self.raw = work.copy()
        self.aux = (
            pd.DataFrame(aux_rows, index=work.index)
            .reindex(columns=AUX_FEATURE_COLUMNS)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        self._trim_frames()

    def append_last(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        ts = pd.Timestamp(df.index[-1])
        if self.last_ts is not None and ts <= self.last_ts:
            return False

        cols = {str(c).lower(): c for c in df.columns}
        if not {"open", "high", "low", "close"}.issubset(cols):
            return False

        o = self._safe_float(df.iloc[-1][cols["open"]], default=0.0)
        h = self._safe_float(df.iloc[-1][cols["high"]], default=o)
        l = self._safe_float(df.iloc[-1][cols["low"]], default=o)
        c = self._safe_float(df.iloc[-1][cols["close"]], default=o)
        if "volume" in cols:
            v = self._safe_float(df.iloc[-1][cols["volume"]], default=0.0)
        else:
            v = 0.0

        aux_row = self._append_state(
            ts=ts,
            open_px=o,
            high_px=h,
            low_px=l,
            close_px=c,
            volume=v,
        )

        raw_row = pd.DataFrame(
            [{"open": o, "high": h, "low": l, "close": c, "volume": v}],
            index=[ts],
        )
        aux_df_row = pd.DataFrame([aux_row], index=[ts])

        self.raw = pd.concat([self.raw, raw_row], axis=0)
        self.aux = pd.concat([self.aux, aux_df_row], axis=0)
        self.last_ts = ts
        self._trim_frames()
        return True


class ManifoldStrategy(Strategy):
    """
    ML strategy driven by manifold-context + market micro features.
    This strategy does not use manifold as a hard gate; it uses manifold as predictive context.
    """

    def __init__(self):
        self.cfg = dict(CONFIG.get("MANIFOLD_STRATEGY", {}) or {})
        self.model_path = Path(self.cfg.get("model_file", "model_manifold_strategy.joblib"))
        self.thresholds_path = Path(
            self.cfg.get("thresholds_file", "manifold_strategy_thresholds.json")
        )
        self.confluence_path = Path(
            self.cfg.get("confluence_file", "manifold_strategy_confluence.json")
        )
        self.min_bars = int(self.cfg.get("min_bars", 250) or 250)
        self.min_confidence = float(self.cfg.get("min_confidence", 0.0) or 0.0)
        self.size = int(self.cfg.get("size", 5) or 5)
        self.sl_points = float(self.cfg.get("sl_points", 4.0) or 4.0)
        self.tp_points = float(self.cfg.get("tp_points", 8.0) or 8.0)
        self.sl_atr_mult = float(self.cfg.get("sl_atr_mult", 1.25) or 1.25)
        self.tp_atr_mult = float(self.cfg.get("tp_atr_mult", 2.0) or 2.0)
        self.respect_no_trade = bool(self.cfg.get("respect_no_trade", False))
        self.log_evals = bool(self.cfg.get("log_evals", False))
        default_no_trade_policy = "hard" if self.respect_no_trade else "off"
        self.no_trade_policy = str(
            self.cfg.get("no_trade_policy", default_no_trade_policy) or default_no_trade_policy
        ).strip().lower()
        if self.no_trade_policy not in {"hard", "soft", "off"}:
            self.no_trade_policy = default_no_trade_policy
        self.no_trade_conf_boost = float(self.cfg.get("no_trade_conf_boost", 0.0) or 0.0)
        if not np.isfinite(self.no_trade_conf_boost):
            self.no_trade_conf_boost = 0.0
        self.no_trade_conf_boost = max(0.0, self.no_trade_conf_boost)
        self.no_trade_size_mult = float(self.cfg.get("no_trade_size_mult", 0.5) or 0.5)
        if not np.isfinite(self.no_trade_size_mult):
            self.no_trade_size_mult = 0.5
        self.no_trade_size_mult = float(np.clip(self.no_trade_size_mult, 0.10, 1.0))
        self.no_trade_stress_hard = float(self.cfg.get("no_trade_stress_hard", 0.92) or 0.92)
        if not np.isfinite(self.no_trade_stress_hard):
            self.no_trade_stress_hard = 0.92
        self.no_trade_stress_hard = float(np.clip(self.no_trade_stress_hard, 0.0, 1.0))
        block_regimes_cfg = self.cfg.get("no_trade_block_regimes", []) or []
        self.no_trade_block_regimes = {
            str(item).strip().upper() for item in block_regimes_cfg if str(item).strip()
        }
        hyst_cfg = self.cfg.get("hysteresis", {}) or {}
        self.hyst_enabled = bool(hyst_cfg.get("enabled", True))
        self.hyst_entry_margin = float(hyst_cfg.get("entry_margin", 0.0) or 0.0)
        self.hyst_exit_margin = float(hyst_cfg.get("exit_margin", 0.02) or 0.02)
        self.hyst_flip_margin = float(hyst_cfg.get("flip_margin", 0.01) or 0.01)
        self.hyst_retrigger_delta = float(hyst_cfg.get("retrigger_delta", 0.0) or 0.0)
        for field in (
            "hyst_entry_margin",
            "hyst_exit_margin",
            "hyst_flip_margin",
            "hyst_retrigger_delta",
        ):
            val = getattr(self, field, 0.0)
            if not np.isfinite(val):
                val = 0.0
            setattr(self, field, max(0.0, float(val)))
        self._hyst_active_side: Optional[str] = None
        self._hyst_last_conf: float = 0.0

        self.model = None
        self.confluence_params: Dict = {}
        self.confluence_enabled = bool(self.cfg.get("confluence_enabled", True))
        self.size_scale_min = float(self.cfg.get("size_scale_min", 0.5) or 0.5)
        self.size_scale_max = float(self.cfg.get("size_scale_max", 2.0) or 2.0)
        self.use_confluence_for_size = bool(self.cfg.get("use_confluence_for_size", True))
        self.feature_columns = list(FEATURE_COLUMNS)
        self.long_threshold = 0.55
        self.short_threshold = 0.45
        self.allowed_session_ids: Optional[set[int]] = _coerce_session_allowlist(
            self.cfg.get("allowed_session_ids")
        )
        self.model_loaded = False
        self.last_eval: Optional[Dict] = None
        self._precomputed_backtest_df: Optional[pd.DataFrame] = None
        self._precomputed_lookup: dict[int, dict] = {}
        cache_rows_cfg = int(self.cfg.get("feature_cache_rows", 2500) or 2500)
        cache_rows = max(self.min_bars + 64, cache_rows_cfg)
        self._aux_cache = _IncrementalAuxCache(max_rows=cache_rows, z_window=200)

        self._load_artifacts()

        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg.update(self.cfg.get("manifold_params", {}) or {})
        manifold_cfg["enabled"] = True
        self.engine = RegimeManifoldEngine(manifold_cfg)

    def set_precomputed_backtest_df(self, df: Optional[pd.DataFrame]) -> None:
        self._precomputed_backtest_df = None if df is None else df.copy()
        self._precomputed_lookup = {}
        if not isinstance(self._precomputed_backtest_df, pd.DataFrame) or self._precomputed_backtest_df.empty:
            return
        rows = self._precomputed_backtest_df.to_dict("records")
        for ts, row in zip(pd.DatetimeIndex(self._precomputed_backtest_df.index), rows):
            self._precomputed_lookup[int(ts.value)] = row

    def _feature_regime_name(self, feature_row: pd.Series) -> str:
        try:
            regime_id = int(round(float(feature_row.get("manifold_regime_id", -1) or -1)))
        except Exception:
            regime_id = -1
        inverse = {int(v): str(k) for k, v in REGIME_TO_ID.items()}
        return inverse.get(regime_id, "UNKNOWN")

    def build_precomputed_backtest_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model_loaded or self.model is None or df is None or df.empty:
            return pd.DataFrame()

        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg.update(self.cfg.get("manifold_params", {}) or {})
        manifold_cfg["enabled"] = True
        features = build_training_feature_frame(df, manifold_cfg=manifold_cfg, log_every=0)
        if features.empty:
            return pd.DataFrame()

        if self.allowed_session_ids:
            sess = pd.to_numeric(features.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
            features = features.loc[sess.isin(sorted(self.allowed_session_ids))]
        if features.empty:
            return pd.DataFrame()

        if hasattr(self.model, "feature_names_in_"):
            model_cols = [str(c) for c in self.model.feature_names_in_]
            x_all = features.reindex(columns=model_cols, fill_value=0.0)
        else:
            x_all = features.reindex(columns=self.feature_columns, fill_value=0.0)
        x_all = x_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if hasattr(self.model, "n_jobs"):
            try:
                bt_workers = max(1, int(CONFIG.get("BACKTEST_WORKERS", 1) or 1))
                self.model.n_jobs = max(bt_workers, int(getattr(self.model, "n_jobs", 1) or 1))
            except Exception:
                pass

        prob_up_raw = np.asarray(self.model.predict_proba(x_all)[:, 1], dtype=float)
        prob_up = prob_up_raw.copy()
        alpha_scale = np.ones(len(features), dtype=float)
        directional_alignment = np.zeros(len(features), dtype=float)

        if self.confluence_enabled:
            try:
                conf_out = apply_confluence_formula(
                    x_all.reset_index(drop=True),
                    prob_up_raw,
                    self.confluence_params,
                )
                prob_up = np.asarray(conf_out["prob_up_adj"], dtype=float)
                alpha_scale = np.asarray(conf_out["alpha_scale"], dtype=float)
                directional_alignment = np.asarray(
                    conf_out["directional_alignment"],
                    dtype=float,
                )
            except Exception as exc:
                logging.debug("ManifoldStrategy backtest confluence apply failed: %s", exc)

        active_side: Optional[str] = None
        active_conf = 0.0
        signal_rows: list[dict] = []
        signal_index: list[pd.Timestamp] = []

        for pos, (ts, feature_row) in enumerate(features.iterrows()):
            prob_up_i = float(prob_up[pos])
            prob_up_raw_i = float(prob_up_raw[pos])
            prob_down_i = float(1.0 - prob_up_i)
            alpha_scale_i = float(alpha_scale[pos]) if pos < len(alpha_scale) else 1.0
            directional_alignment_i = (
                float(directional_alignment[pos]) if pos < len(directional_alignment) else 0.0
            )

            side = None
            conf = 0.0
            if prob_up_i >= self.long_threshold:
                side = "LONG"
                conf = prob_up_i
            elif prob_up_i <= self.short_threshold:
                side = "SHORT"
                conf = prob_down_i

            if side is None:
                active_side = None
                active_conf = 0.0
                continue

            meta_no_trade = bool(float(feature_row.get("manifold_no_trade", 0.0) or 0.0) >= 0.5)
            meta_stress = float(feature_row.get("manifold_stress", 0.0) or 0.0)
            meta_r = float(feature_row.get("manifold_R", 0.0) or 0.0)
            meta_regime = self._feature_regime_name(feature_row)
            meta_regime_key = str(meta_regime).upper()
            required_conf = float(self.min_confidence)
            soft_no_trade_active = False
            size_mult = 1.0

            if self.no_trade_policy == "hard" and meta_no_trade:
                continue

            if self.no_trade_policy == "soft" and meta_no_trade:
                hard_regime = bool(
                    self.no_trade_block_regimes and meta_regime_key in self.no_trade_block_regimes
                )
                hard_stress = bool(meta_stress >= self.no_trade_stress_hard)
                if hard_regime or hard_stress:
                    continue
                soft_no_trade_active = True
                required_conf = float(self.min_confidence + self.no_trade_conf_boost)
                size_mult = float(self.no_trade_size_mult)

            if conf < required_conf:
                continue

            if self.hyst_enabled:
                entry_level = float(np.clip(required_conf + self.hyst_entry_margin, 0.0, 1.0))
                exit_level = float(np.clip(required_conf - self.hyst_exit_margin, 0.0, 1.0))
                flip_level = float(np.clip(required_conf + self.hyst_flip_margin, 0.0, 1.0))

                hyst_ok = False
                if active_side is None:
                    if conf >= entry_level:
                        active_side = side
                        active_conf = conf
                        hyst_ok = True
                elif side != active_side:
                    if conf >= flip_level:
                        active_side = side
                        active_conf = conf
                        hyst_ok = True
                elif conf <= exit_level:
                    active_side = None
                    active_conf = 0.0
                    hyst_ok = False
                elif self.hyst_retrigger_delta > 0.0 and conf >= (active_conf + self.hyst_retrigger_delta):
                    active_conf = conf
                    hyst_ok = True
                else:
                    active_conf = max(active_conf, conf)
                    hyst_ok = False

                if not hyst_ok:
                    continue

            atr_pts = float(feature_row.get("atr14", 0.0) or 0.0)
            sl_dist, tp_dist = self._resolve_brackets_from_atr(atr_pts)
            size = int(self.size)
            if self.use_confluence_for_size:
                scale = float(np.clip(alpha_scale_i, self.size_scale_min, self.size_scale_max))
                size = int(max(1, round(float(self.size) * scale)))
            if soft_no_trade_active and size_mult < 1.0:
                size = int(max(1, round(float(size) * size_mult)))

            signal_rows.append(
                {
                    "strategy": "ManifoldStrategy",
                    "side": side,
                    "tp_dist": float(tp_dist),
                    "sl_dist": float(sl_dist),
                    "size": int(size),
                    "confidence": float(conf),
                    "manifold_confidence": float(conf),
                    "manifold_prob_up": float(prob_up_i),
                    "manifold_prob_up_raw": float(prob_up_raw_i),
                    "manifold_prob_down": float(prob_down_i),
                    "manifold_alpha_scale": float(alpha_scale_i),
                    "manifold_directional_alignment": float(directional_alignment_i),
                    "manifold_threshold": float(self.long_threshold),
                    "manifold_short_threshold": float(self.short_threshold),
                    "manifold_regime": meta_regime,
                    "manifold_R": float(meta_r),
                    "manifold_meta_stress": float(meta_stress),
                    "manifold_no_trade": bool(meta_no_trade),
                    "manifold_no_trade_policy": self.no_trade_policy,
                    "manifold_required_confidence": float(required_conf),
                    "manifold_soft_no_trade": bool(soft_no_trade_active),
                }
            )
            signal_index.append(pd.Timestamp(ts))

        if not signal_rows:
            return pd.DataFrame()

        out = pd.DataFrame(signal_rows, index=pd.DatetimeIndex(signal_index))
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    def _load_artifacts(self) -> None:
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                self.model_loaded = True
                logging.info("ManifoldStrategy model loaded: %s", self.model_path)
            else:
                logging.warning("ManifoldStrategy model missing: %s", self.model_path)
        except Exception as exc:
            logging.error("ManifoldStrategy model load failed (%s): %s", self.model_path, exc)
            self.model = None
            self.model_loaded = False

        try:
            if self.thresholds_path.exists():
                payload = json.loads(self.thresholds_path.read_text())
                self.long_threshold = float(payload.get("threshold", self.long_threshold))
                self.short_threshold = float(
                    payload.get("short_threshold", self.short_threshold)
                )
                cols = payload.get("feature_columns")
                if isinstance(cols, list) and cols:
                    self.feature_columns = [str(c) for c in cols]
                allow_sessions = _coerce_session_allowlist(payload.get("allowed_session_ids"))
                if allow_sessions:
                    self.allowed_session_ids = allow_sessions
                logging.info(
                    "ManifoldStrategy thresholds loaded: long=%.3f short=%.3f",
                    self.long_threshold,
                    self.short_threshold,
                )
                if self.allowed_session_ids:
                    names = [_session_name_from_id(s) for s in sorted(self.allowed_session_ids)]
                    logging.info(
                        "ManifoldStrategy session allowlist: ids=%s names=%s",
                        sorted(self.allowed_session_ids),
                        names,
                    )
            else:
                logging.warning(
                    "ManifoldStrategy thresholds file missing: %s", self.thresholds_path
                )
        except Exception as exc:
            logging.error(
                "ManifoldStrategy threshold load failed (%s): %s",
                self.thresholds_path,
                exc,
            )

        try:
            if self.confluence_enabled and self.confluence_path.exists():
                payload = json.loads(self.confluence_path.read_text())
                if isinstance(payload, dict) and bool(payload.get("enabled", False)):
                    params = payload.get("params")
                    if isinstance(params, dict):
                        self.confluence_params = dict(params)
                    self.long_threshold = float(payload.get("validation", {}).get("threshold", self.long_threshold))
                    self.short_threshold = float(
                        payload.get("validation", {}).get("short_threshold", self.short_threshold)
                    )
                    logging.info(
                        "ManifoldStrategy confluence loaded: %s (long=%.3f short=%.3f)",
                        self.confluence_path,
                        self.long_threshold,
                        self.short_threshold,
                    )
                else:
                    self.confluence_enabled = False
            elif self.confluence_enabled:
                logging.info("ManifoldStrategy confluence file missing: %s", self.confluence_path)
        except Exception as exc:
            self.confluence_enabled = False
            logging.warning(
                "ManifoldStrategy confluence load failed (%s): %s",
                self.confluence_path,
                exc,
            )

    @staticmethod
    def _atr_points(df: pd.DataFrame, span: int = 14) -> float:
        work = df.copy()
        work.columns = [str(c).lower() for c in work.columns]
        if not {"high", "low", "close"}.issubset(work.columns):
            return 0.0
        high = pd.to_numeric(work["high"], errors="coerce")
        low = pd.to_numeric(work["low"], errors="coerce")
        close = pd.to_numeric(work["close"], errors="coerce")
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / float(max(2, span)), adjust=False).mean().iloc[-1]
        try:
            val = float(atr)
        except Exception:
            val = 0.0
        if not np.isfinite(val):
            return 0.0
        return max(0.0, val)

    def _resolve_brackets(self, df: pd.DataFrame) -> tuple[float, float]:
        atr_pts = self._atr_points(df, span=14)
        sl = max(self.sl_points, atr_pts * self.sl_atr_mult)
        tp = max(self.tp_points, atr_pts * self.tp_atr_mult)
        return float(round(sl, 2)), float(round(tp, 2))

    def _resolve_brackets_from_atr(self, atr_pts: float) -> tuple[float, float]:
        try:
            atr_val = float(atr_pts)
        except Exception:
            atr_val = 0.0
        if not np.isfinite(atr_val):
            atr_val = 0.0
        atr_val = max(0.0, atr_val)
        sl = max(self.sl_points, atr_val * self.sl_atr_mult)
        tp = max(self.tp_points, atr_val * self.tp_atr_mult)
        return float(round(sl, 2)), float(round(tp, 2))

    def _hysteresis_gate(
        self,
        side: Optional[str],
        confidence: float,
        required_confidence: float,
    ) -> tuple[bool, str]:
        if not self.hyst_enabled:
            return bool(side), "disabled"

        side_norm = str(side or "").upper()
        if side_norm not in {"LONG", "SHORT"}:
            self._hyst_active_side = None
            self._hyst_last_conf = 0.0
            return False, "neutral_disarm"

        try:
            conf = float(confidence)
        except Exception:
            conf = 0.0
        if not np.isfinite(conf):
            conf = 0.0

        try:
            req = float(required_confidence)
        except Exception:
            req = 0.0
        if not np.isfinite(req):
            req = 0.0

        entry_level = float(np.clip(req + self.hyst_entry_margin, 0.0, 1.0))
        exit_level = float(np.clip(req - self.hyst_exit_margin, 0.0, 1.0))
        flip_level = float(np.clip(req + self.hyst_flip_margin, 0.0, 1.0))

        if self._hyst_active_side is None:
            if conf >= entry_level:
                self._hyst_active_side = side_norm
                self._hyst_last_conf = conf
                return True, "edge_entry"
            return False, "entry_not_met"

        if side_norm != self._hyst_active_side:
            if conf >= flip_level:
                self._hyst_active_side = side_norm
                self._hyst_last_conf = conf
                return True, "edge_flip"
            return False, "flip_not_met"

        if conf <= exit_level:
            self._hyst_active_side = None
            self._hyst_last_conf = 0.0
            return False, "exit_disarm"

        if self.hyst_retrigger_delta > 0.0 and conf >= (self._hyst_last_conf + self.hyst_retrigger_delta):
            self._hyst_last_conf = conf
            return True, "retrigger"

        self._hyst_last_conf = max(self._hyst_last_conf, conf)
        return False, "held"

    def _sync_live_cache(self, df: pd.DataFrame, ts: Optional[pd.Timestamp]) -> None:
        if df is None or df.empty:
            self._aux_cache.bootstrap(pd.DataFrame())
            return
        # Cache tracks the latest bar in df. Using df index avoids needless
        # rebuilds if external current_time has different tz/precision.
        ts = pd.Timestamp(df.index[-1])

        if self._aux_cache.last_ts is None or self._aux_cache.raw.empty:
            self._aux_cache.bootstrap(df)
            return

        if ts == self._aux_cache.last_ts:
            return

        can_append = False
        if len(df) >= 2:
            try:
                prev_ts = pd.Timestamp(df.index[-2])
                can_append = prev_ts == self._aux_cache.last_ts and ts > self._aux_cache.last_ts
            except Exception:
                can_append = False

        if can_append:
            appended = self._aux_cache.append_last(df)
            if appended:
                return

        # Fallback to full cache rebuild when timeline diverges.
        self._aux_cache.bootstrap(df)

    def on_bar(self, df: pd.DataFrame, current_time=None) -> Optional[Dict]:
        self.last_eval = None
        if current_time is not None and self._precomputed_lookup:
            ts = pd.Timestamp(current_time)
            row = self._precomputed_lookup.get(int(ts.value))
            if row is None:
                self.last_eval = {
                    "decision": "no_signal",
                    "reason": "precomputed_lookup_miss",
                    "ts": ts.isoformat(),
                }
                return None
            self.last_eval = {
                "decision": "signal",
                "side": row.get("side"),
                "confidence": float(row.get("manifold_confidence", row.get("confidence", 0.0)) or 0.0),
                "prob_up": float(row.get("manifold_prob_up", 0.0) or 0.0),
                "prob_up_raw": float(row.get("manifold_prob_up_raw", row.get("manifold_prob_up", 0.0)) or 0.0),
                "prob_down": float(row.get("manifold_prob_down", 0.0) or 0.0),
                "long_threshold": float(row.get("manifold_threshold", self.long_threshold) or self.long_threshold),
                "short_threshold": float(
                    row.get("manifold_short_threshold", self.short_threshold) or self.short_threshold
                ),
                "meta_regime": row.get("manifold_regime"),
                "meta_r": float(row.get("manifold_R", 0.0) or 0.0),
                "meta_stress": float(row.get("manifold_meta_stress", 0.0) or 0.0),
                "meta_no_trade": bool(row.get("manifold_no_trade", False)),
                "required_confidence": float(
                    row.get("manifold_required_confidence", self.min_confidence) or self.min_confidence
                ),
                "soft_no_trade_active": bool(row.get("manifold_soft_no_trade", False)),
                "no_trade_policy": row.get("manifold_no_trade_policy", self.no_trade_policy),
                "alpha_scale": float(row.get("manifold_alpha_scale", 1.0) or 1.0),
                "directional_alignment": float(row.get("manifold_directional_alignment", 0.0) or 0.0),
                "runtime_mode": "precomputed_backtest",
            }
            return dict(row)
        if not self.model_loaded or self.model is None:
            return None
        if df is None or df.empty or len(df) < self.min_bars:
            return None

        ts = current_time
        if ts is None:
            try:
                ts = pd.Timestamp(df.index[-1])
            except Exception:
                ts = None

        try:
            ts_eff = pd.Timestamp(df.index[-1]) if ts is None else pd.Timestamp(ts)
            if self.allowed_session_ids:
                session_id = int(round(float(_IncrementalAuxCache._session_id(ts_eff) or -999.0)))
                if session_id not in self.allowed_session_ids:
                    self.last_eval = {
                        "decision": "blocked",
                        "reason": "session_not_allowed",
                        "session_id": int(session_id),
                        "session_name": _session_name_from_id(session_id),
                        "allowed_session_ids": sorted(int(s) for s in self.allowed_session_ids),
                    }
                    return None

            self._sync_live_cache(df, ts)
            if self._aux_cache.raw.empty or self._aux_cache.aux.empty:
                return None

            work = self._aux_cache.raw
            ts_eff = pd.Timestamp(work.index[-1]) if ts is None else pd.Timestamp(ts)
            meta = self.engine.update(work, ts=ts_eff, session=get_session_name(ts_eff))
            meta_payload = meta if isinstance(meta, dict) else {}
            meta_no_trade = bool(meta_payload.get("no_trade", False))
            meta_regime = str(meta_payload.get("regime", "") or "")
            meta_regime_key = meta_regime.upper()
            try:
                meta_stress = float(meta_payload.get("stress", 0.0) or 0.0)
            except Exception:
                meta_stress = 0.0
            if not np.isfinite(meta_stress):
                meta_stress = 0.0
            try:
                meta_r = float(meta_payload.get("R", 0.0) or 0.0)
            except Exception:
                meta_r = 0.0
            if not np.isfinite(meta_r):
                meta_r = 0.0

            if self.no_trade_policy == "hard" and meta_no_trade:
                self.last_eval = {
                    "decision": "blocked",
                    "reason": "meta_no_trade",
                    "ts": str(ts_eff) if ts_eff is not None else None,
                    "meta_regime": meta_regime or None,
                    "meta_stress": float(meta_stress),
                    "no_trade_policy": self.no_trade_policy,
                }
                return None

            row = meta_to_feature_dict(meta_payload)
            aux_last = self._aux_cache.aux.iloc[-1]
            for col in AUX_FEATURE_COLUMNS:
                row[col] = float(aux_last.get(col, 0.0) or 0.0)
            x_row = (
                pd.DataFrame([row], index=[work.index[-1]])
                .reindex(columns=self.feature_columns)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

            if hasattr(self.model, "feature_names_in_"):
                cols = [str(c) for c in self.model.feature_names_in_]
                x_row = x_row.reindex(columns=cols, fill_value=0.0)
            else:
                x_row = x_row.reindex(columns=self.feature_columns, fill_value=0.0)

            prob_up_raw = float(self.model.predict_proba(x_row)[0][1])
            prob_up = prob_up_raw
            alpha_scale = 1.0
            directional_alignment = 0.0
            if self.confluence_enabled:
                try:
                    conf_out = apply_confluence_formula(
                        x_row.reset_index(drop=True),
                        np.asarray([prob_up_raw], dtype=float),
                        self.confluence_params,
                    )
                    prob_up = float(conf_out["prob_up_adj"][0])
                    alpha_scale = float(conf_out["alpha_scale"][0])
                    directional_alignment = float(conf_out["directional_alignment"][0])
                except Exception as exc:
                    logging.debug("ManifoldStrategy confluence apply failed: %s", exc)
            prob_down = float(1.0 - prob_up)

            side = None
            conf = 0.0
            if prob_up >= self.long_threshold:
                side = "LONG"
                conf = prob_up
            elif prob_up <= self.short_threshold:
                side = "SHORT"
                conf = prob_down

            self.last_eval = {
                "decision": "signal" if side else "no_signal",
                "side": side,
                "confidence": conf,
                "alpha_scale": alpha_scale,
                "directional_alignment": directional_alignment,
                "prob_up": prob_up,
                "prob_up_raw": prob_up_raw,
                "prob_down": prob_down,
                "long_threshold": self.long_threshold,
                "short_threshold": self.short_threshold,
                "meta_regime": meta_regime or None,
                "meta_r": float(meta_r),
                "meta_stress": float(meta_stress),
                "meta_no_trade": bool(meta_no_trade),
                "no_trade_policy": self.no_trade_policy,
            }

            if side is None:
                self._hysteresis_gate(None, 0.0, 0.0)
                return None

            required_conf = float(self.min_confidence)
            soft_no_trade_active = False
            size_mult = 1.0
            if self.no_trade_policy == "soft" and meta_no_trade:
                hard_regime = bool(
                    self.no_trade_block_regimes and meta_regime_key in self.no_trade_block_regimes
                )
                hard_stress = bool(meta_stress >= self.no_trade_stress_hard)
                if hard_regime or hard_stress:
                    self.last_eval.update(
                        {
                            "decision": "blocked",
                            "reason": "meta_no_trade_hard_safety",
                            "hard_regime": hard_regime,
                            "hard_stress": hard_stress,
                            "required_confidence": float(required_conf),
                        }
                    )
                    return None
                soft_no_trade_active = True
                required_conf = float(self.min_confidence + self.no_trade_conf_boost)
                size_mult = float(self.no_trade_size_mult)

            if conf < required_conf:
                self.last_eval.update(
                    {
                        "decision": "no_signal",
                        "reason": (
                            "below_confidence_soft_no_trade"
                            if soft_no_trade_active
                            else "below_confidence"
                        ),
                        "required_confidence": float(required_conf),
                    }
                )
                return None

            hyst_ok, hyst_reason = self._hysteresis_gate(side, conf, required_conf)
            if not hyst_ok:
                self.last_eval.update(
                    {
                        "decision": "no_signal",
                        "reason": "hysteresis",
                        "hysteresis_reason": hyst_reason,
                        "hysteresis_active_side": self._hyst_active_side,
                    }
                )
                return None

            atr_pts = float(aux_last.get("atr14", 0.0) or 0.0)
            sl_dist, tp_dist = self._resolve_brackets_from_atr(atr_pts)
            size = int(self.size)
            if self.use_confluence_for_size:
                scale = float(np.clip(alpha_scale, self.size_scale_min, self.size_scale_max))
                size = int(max(1, round(float(self.size) * scale)))
            if soft_no_trade_active and size_mult < 1.0:
                size = int(max(1, round(float(size) * size_mult)))
            self.last_eval.update(
                {
                    "required_confidence": float(required_conf),
                    "soft_no_trade_active": bool(soft_no_trade_active),
                    "size_mult": float(size_mult),
                    "hysteresis_reason": hyst_reason,
                    "hysteresis_active_side": self._hyst_active_side,
                }
            )
            if self.log_evals:
                logging.info(
                    "ManifoldStrategy signal: %s conf=%.3f req=%.3f p_up=%.3f p_raw=%.3f "
                    "alpha=%.3f thr=%.3f/%.3f regime=%s no_trade=%s policy=%s",
                    side,
                    conf,
                    required_conf,
                    prob_up,
                    prob_up_raw,
                    alpha_scale,
                    self.long_threshold,
                    self.short_threshold,
                    self.last_eval.get("meta_regime"),
                    meta_no_trade,
                    self.no_trade_policy,
                )

            return {
                "strategy": "ManifoldStrategy",
                "side": side,
                "tp_dist": tp_dist,
                "sl_dist": sl_dist,
                "size": int(size),
                "confidence": float(conf),
                "manifold_confidence": float(conf),
                "manifold_prob_up": float(prob_up),
                "manifold_prob_up_raw": float(prob_up_raw),
                "manifold_prob_down": float(prob_down),
                "manifold_alpha_scale": float(alpha_scale),
                "manifold_directional_alignment": float(directional_alignment),
                "manifold_threshold": float(self.long_threshold),
                "manifold_short_threshold": float(self.short_threshold),
                "manifold_regime": self.last_eval.get("meta_regime"),
                "manifold_R": self.last_eval.get("meta_r"),
                "manifold_meta_stress": float(meta_stress),
                "manifold_no_trade": bool(meta_no_trade),
                "manifold_no_trade_policy": self.no_trade_policy,
                "manifold_required_confidence": float(required_conf),
                "manifold_soft_no_trade": bool(soft_no_trade_active),
            }
        except Exception as exc:
            logging.error("ManifoldStrategy prediction error: %s", exc)
            self.last_eval = {
                "decision": "error",
                "error": str(exc),
                "ts": str(ts) if ts is not None else None,
            }
            return None
