import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dynamic_signal_engine2 import get_signal_engine as get_signal_engine2
from strategy_base import Strategy
from config import CONFIG
from dynamic_sltp_params import get_sltp
from fixed_sltp_framework import apply_fixed_sltp
from incremental_ohlcv_resampler import IncrementalOHLCVResampler
from volatility_filter import volatility_filter

class DynamicEngine2Strategy(Strategy):
    """
    Wrapper for DynamicSignalEngine2 to run 167 hardcoded Price Action strategies
    alongside Julie's existing filters.

    UPDATED: Now overrides static SL/TPs with Dynamic RegimeAdaptive parameters
    from dynamic_sltp_params.py to tighten risk management.
    """

    def __init__(self):
        self.engine = get_signal_engine2()
        self.last_processed_time = None
        self._resampler_5m = IncrementalOHLCVResampler(5)
        self._resampler_15m = IncrementalOHLCVResampler(15)
        self._policy_cfg = CONFIG.get("DYNAMIC_ENGINE2_POLICY", {}) or {}
        self._quality_cfg = self._policy_cfg.get("quality_filters", {}) or {}
        self._regime_cfg = self._policy_cfg.get("regime_allow", {}) or {}
        self._ev_cfg = self._policy_cfg.get("ev_ranking", {}) or {}
        self._context_profiles = self._policy_cfg.get("context_profiles", {}) or {}
        self._regime_mode = str(self._regime_cfg.get("mode", "soft") or "soft").strip().lower()
        if self._regime_mode not in {"hard", "soft"}:
            self._regime_mode = "soft"
        self._log_decisions = bool(self._policy_cfg.get("log_decisions", False))
        stability_cfg = self._policy_cfg.get("stability", {}) or {}
        self._stability_enabled = bool(stability_cfg.get("enabled", True))
        self._stability_window_min = max(
            1,
            int(self._safe_float(stability_cfg.get("window_minutes", 90), 90.0)),
        )
        self._stability_context_soft_cap = max(
            0,
            int(self._safe_float(stability_cfg.get("context_soft_cap", 12), 12.0)),
        )
        self._stability_strategy_soft_cap = max(
            0,
            int(self._safe_float(stability_cfg.get("strategy_soft_cap", 4), 4.0)),
        )
        self._stability_context_penalty = self._safe_float(
            stability_cfg.get("context_penalty_per_extra", 0.02),
            0.02,
        )
        self._stability_strategy_penalty = self._safe_float(
            stability_cfg.get("strategy_penalty_per_extra", 0.06),
            0.06,
        )
        self._stability_min_edge_prob = self._safe_float(
            stability_cfg.get("min_edge_prob", 0.01),
            0.01,
        )
        self._stability_session_mult = stability_cfg.get("session_penalty_mult", {}) or {}
        self._stability_regime_mult = stability_cfg.get("regime_penalty_mult", {}) or {}
        self._stability_timeframe_mult = stability_cfg.get("timeframe_penalty_mult", {}) or {}
        self._stability_timeframe_strategy_cap = (
            stability_cfg.get("timeframe_strategy_soft_cap", {}) or {}
        )
        self._signal_history: List[Dict] = []
        drift_cfg = CONFIG.get("DYNAMIC_ENGINE2_DRIFT", {}) or {}
        self._drift_enabled = bool(drift_cfg.get("enabled", True))
        self._drift_max_atr = self._safe_float(drift_cfg.get("max_atr", 1.0), 1.0)
        self._drift_atr_period = int(self._safe_float(drift_cfg.get("atr_period", 14), 14.0))
        self._drift_fallback_points = self._safe_float(drift_cfg.get("fallback_points", 0.0), 0.0)
        self._drift_anchors: Dict[Tuple[str, str, pd.Timestamp], float] = {}

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            out = float(value)
            if np.isfinite(out):
                return out
            return default
        except Exception:
            return default

    @staticmethod
    def _compute_atr_simple(df: pd.DataFrame, period: int) -> Optional[float]:
        if df is None or df.empty:
            return None
        try:
            period = int(period)
        except Exception:
            period = 14
        if period <= 1 or len(df) < period + 1:
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        try:
            value = float(atr.iloc[-1])
        except Exception:
            return None
        if not np.isfinite(value) or value <= 0:
            return None
        return value

    @staticmethod
    def _signal_bucket(ts, strategy_id: str) -> pd.Timestamp:
        ts_val = pd.Timestamp(ts)
        sid = str(strategy_id or "").lower()
        if "_15m_" in sid:
            return ts_val.floor("15min")
        if "_5m_" in sid:
            return ts_val.floor("5min")
        return ts_val.floor("1min")

    def _prune_drift_anchors(self, current_time: pd.Timestamp) -> None:
        if not self._drift_anchors:
            return
        cutoff = pd.Timestamp(current_time) - pd.Timedelta(hours=8)
        self._drift_anchors = {k: v for k, v in self._drift_anchors.items() if pd.Timestamp(k[2]) >= cutoff}

    def _prune_signal_history(self, current_time: pd.Timestamp) -> None:
        if not self._signal_history:
            return
        cutoff = pd.Timestamp(current_time) - pd.Timedelta(minutes=self._stability_window_min)
        self._signal_history = [
            row for row in self._signal_history if pd.Timestamp(row.get("ts")) >= cutoff
        ]

    def _stability_penalty(
        self,
        *,
        cand: Dict,
        regime_norm: str,
        current_time: pd.Timestamp,
        ctx_profile: Optional[Dict] = None,
    ) -> Tuple[float, Dict]:
        if not self._stability_enabled:
            return 0.0, {}
        self._prune_signal_history(current_time)
        profile = ctx_profile if isinstance(ctx_profile, dict) else {}

        session_key = str(cand.get("session", "") or "").upper()
        regime_key = str(regime_norm or "").lower()
        tf_key = str(cand.get("timeframe", "") or "").lower()
        side = str(cand.get("signal", "") or "").upper()
        strategy_id = str(cand.get("strategy_id", "") or "")

        context_cap = max(
            0,
            int(
                self._safe_float(
                    profile.get("context_soft_cap", self._stability_context_soft_cap),
                    float(self._stability_context_soft_cap),
                )
            ),
        )
        strategy_cap = max(
            0,
            int(
                self._safe_float(
                    profile.get("strategy_soft_cap", self._stability_strategy_soft_cap),
                    float(self._stability_strategy_soft_cap),
                )
            ),
        )
        tf_strategy_cap = self._safe_float(
            self._stability_timeframe_strategy_cap.get(tf_key),
            float(strategy_cap),
        )
        strategy_cap = max(0, int(min(float(strategy_cap), float(tf_strategy_cap))))

        context_count = 0
        strategy_count = 0
        for row in self._signal_history:
            if (
                str(row.get("session", "")).upper() == session_key
                and str(row.get("regime", "")).lower() == regime_key
                and str(row.get("timeframe", "")).lower() == tf_key
            ):
                context_count += 1
            if (
                str(row.get("strategy_id", "")) == strategy_id
                and str(row.get("side", "")).upper() == side
            ):
                strategy_count += 1

        extra_context = max(0, context_count - context_cap)
        extra_strategy = max(0, strategy_count - strategy_cap)

        penalty = (
            (float(extra_context) * float(self._stability_context_penalty))
            + (float(extra_strategy) * float(self._stability_strategy_penalty))
        )

        session_mult = self._map_lookup(self._stability_session_mult, session_key, 1.0)
        regime_mult = self._map_lookup(self._stability_regime_mult, regime_key, 1.0)
        tf_mult = self._map_lookup(self._stability_timeframe_mult, tf_key, 1.0)
        profile_mult = self._safe_float(profile.get("density_penalty_mult", 1.0), 1.0)
        penalty *= float(session_mult) * float(regime_mult) * float(tf_mult) * float(profile_mult)
        return float(max(0.0, penalty)), {
            "de2_density_context_count": float(context_count),
            "de2_density_strategy_count": float(strategy_count),
            "de2_density_context_cap": float(context_cap),
            "de2_density_strategy_cap": float(strategy_cap),
            "de2_density_penalty": float(max(0.0, penalty)),
        }

    def _record_signal(
        self,
        *,
        cand: Dict,
        regime_norm: str,
        current_time: pd.Timestamp,
    ) -> None:
        if not self._stability_enabled:
            return
        self._prune_signal_history(current_time)
        self._signal_history.append(
            {
                "ts": pd.Timestamp(current_time),
                "session": str(cand.get("session", "") or "").upper(),
                "regime": str(regime_norm or "").lower(),
                "timeframe": str(cand.get("timeframe", "") or "").lower(),
                "strategy_id": str(cand.get("strategy_id", "") or ""),
                "side": str(cand.get("signal", "") or "").upper(),
            }
        )

    def _passes_drift_gate(
        self,
        *,
        strategy_id: str,
        side: str,
        current_time: pd.Timestamp,
        current_price: float,
        df_1m: pd.DataFrame,
        df_tf: Optional[pd.DataFrame],
        max_atr_override: Optional[float] = None,
    ) -> Tuple[bool, Dict]:
        max_atr = self._drift_max_atr
        if max_atr_override is not None:
            max_atr = self._safe_float(max_atr_override, max_atr)
        if not self._drift_enabled or max_atr <= 0:
            return True, {}
        self._prune_drift_anchors(current_time)

        bucket = self._signal_bucket(current_time, strategy_id)
        key = (str(strategy_id or ""), str(side or "").upper(), bucket)
        if key not in self._drift_anchors:
            self._drift_anchors[key] = float(current_price)

        anchor = float(self._drift_anchors[key])
        dist_points = abs(float(current_price) - anchor)

        atr_value = None
        df_base = None
        if df_tf is not None and not df_tf.empty:
            df_base = df_tf.iloc[:-1] if len(df_tf) > 1 else df_tf
        atr_value = self._compute_atr_simple(df_base, self._drift_atr_period) if df_base is not None else None
        if atr_value is None:
            atr_value = self._compute_atr_simple(df_1m, self._drift_atr_period)
        if atr_value is None or not np.isfinite(atr_value) or atr_value <= 0:
            atr_value = self._drift_fallback_points if self._drift_fallback_points > 0 else None
        if atr_value is None or atr_value <= 0:
            return True, {
                "de2_drift_anchor": float(anchor),
                "de2_drift_dist_points": float(dist_points),
                "de2_drift_dist_atr": None,
                "de2_drift_limit_atr": float(max_atr),
            }

        dist_atr = float(dist_points / float(atr_value))
        ok = dist_atr <= float(max_atr)
        return ok, {
            "de2_drift_anchor": float(anchor),
            "de2_drift_dist_points": float(dist_points),
            "de2_drift_dist_atr": float(dist_atr),
            "de2_drift_limit_atr": float(max_atr),
            "de2_drift_atr": float(atr_value),
        }

    @staticmethod
    def _coerce_str_set(value) -> set:
        if value is None:
            return set()
        if isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [value]
        out = set()
        for item in items:
            text = str(item).strip()
            if text:
                out.add(text)
        return out

    @staticmethod
    def _normalize_regime(regime_val) -> str:
        text = str(regime_val or "").lower()
        if "high" in text:
            return "high"
        if "norm" in text:
            return "normal"
        if "low" in text:
            return "low"
        return "unknown"

    @staticmethod
    def _map_lookup(mapping: Dict, key, fallback: float) -> float:
        if not isinstance(mapping, dict):
            return fallback
        candidates = [key, str(key), str(key).lower(), str(key).upper()]
        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if cand in mapping:
                try:
                    out = float(mapping.get(cand))
                    if np.isfinite(out):
                        return out
                except Exception:
                    continue
        return fallback

    def _is_regime_allowed(self, cand: Dict, regime_norm: str) -> Tuple[bool, str]:
        cfg = self._regime_cfg or {}
        if not bool(cfg.get("enabled", True)):
            return True, ""

        tf_raw = str(cand.get("timeframe", "") or "").strip().lower()
        stype_raw = str(cand.get("strategy_type", "") or "").strip()
        session_raw = str(cand.get("session", "") or "").strip()
        regime_key = str(regime_norm or "unknown").lower()

        blocked_regimes = {str(x).strip().lower() for x in self._coerce_str_set(cfg.get("blocked_regimes", []))}
        if blocked_regimes and regime_key in blocked_regimes:
            return False, f"regime {regime_key} blocked"

        session_map = cfg.get("sessions", {}) or {}
        allow_sess_raw = session_map.get(regime_key, session_map.get("default"))
        if allow_sess_raw is not None:
            allow_sessions = {str(x).strip().upper() for x in self._coerce_str_set(allow_sess_raw)}
            if allow_sessions and session_raw.upper() not in allow_sessions:
                return False, f"session {session_raw} blocked in regime {regime_key}"

        tf_map = cfg.get("timeframes", {}) or {}
        allow_tf_raw = tf_map.get(regime_key, tf_map.get("default"))
        if allow_tf_raw is not None:
            allow_tfs = {str(x).strip().lower() for x in self._coerce_str_set(allow_tf_raw)}
            if allow_tfs and tf_raw not in allow_tfs:
                return False, f"regime {regime_key} blocks timeframe {tf_raw}"

        block_map = cfg.get("block_strategies", {}) or {}
        blocked_raw = block_map.get(regime_key, block_map.get("default"))
        if blocked_raw is not None:
            blocked = {str(x).strip().lower() for x in self._coerce_str_set(blocked_raw)}
            if stype_raw.lower() in blocked:
                return False, f"regime {regime_key} blocks strategy {stype_raw}"

        allow_map = cfg.get("allow_strategies", {}) or {}
        allowed_raw = allow_map.get(regime_key)
        if allowed_raw is not None:
            allowed = {str(x).strip().lower() for x in self._coerce_str_set(allowed_raw)}
            if allowed and stype_raw.lower() not in allowed:
                return False, f"regime {regime_key} not in allowlist for {stype_raw}"

        return True, ""

    def _resolve_context_profile(self, cand: Dict, regime_norm: str) -> Dict:
        """
        Merge context profile overrides in this order:
        default -> regime -> session -> timeframe -> strategy -> side
        """
        def _merge_payload(target: Dict, payload: Dict) -> None:
            if not isinstance(payload, dict):
                return
            for key, value in payload.items():
                if key == "quality_overrides" and isinstance(value, dict):
                    current = target.get("quality_overrides")
                    if isinstance(current, dict):
                        merged_q = dict(current)
                        merged_q.update(value)
                        target["quality_overrides"] = merged_q
                    else:
                        target["quality_overrides"] = dict(value)
                else:
                    target[key] = value

        profiles = self._context_profiles or {}
        if not isinstance(profiles, dict):
            return {}
        merged: Dict = {}
        default_cfg = profiles.get("default")
        if isinstance(default_cfg, dict):
            _merge_payload(merged, default_cfg)

        regime_map = profiles.get("regime", {}) or {}
        session_map = profiles.get("session", {}) or {}
        tf_map = profiles.get("timeframe", {}) or {}
        strategy_map = profiles.get("strategy", {}) or {}
        side_map = profiles.get("side", {}) or {}

        regime_key = str(regime_norm or "unknown").lower()
        session_key = str(cand.get("session", "") or "").upper()
        tf_key = str(cand.get("timeframe", "") or "")
        stype_key = str(cand.get("strategy_type", "") or "")
        side_key = str(cand.get("signal", "") or "").upper()

        for section, key in (
            (regime_map, regime_key),
            (session_map, session_key),
            (tf_map, tf_key),
            (strategy_map, stype_key),
            (side_map, side_key),
        ):
            if not isinstance(section, dict):
                continue
            payload = section.get(key)
            if payload is None:
                payload = section.get(str(key).lower())
            if payload is None:
                payload = section.get(str(key).upper())
            if isinstance(payload, dict):
                _merge_payload(merged, payload)

        return merged

    def _trigger_stats(self, df_tf: pd.DataFrame, atr_period: int) -> Dict:
        if df_tf is None or len(df_tf) < 4:
            return {}
        trigger = df_tf.iloc[-2]
        prev = df_tf.iloc[-3]
        try:
            o = float(trigger["open"])
            h = float(trigger["high"])
            l = float(trigger["low"])
            c = float(trigger["close"])
            prev_close = float(prev["close"])
        except Exception:
            return {}

        bar_range = max(0.0, h - l)
        body = abs(c - o)
        close_pos = (c - l) / bar_range if bar_range > 0 else 0.5
        atr = self._compute_atr_simple(df_tf.iloc[:-1], atr_period)
        range_atr = (bar_range / atr) if atr and atr > 0 else np.nan
        body_atr = (body / atr) if atr and atr > 0 else np.nan
        gap_atr = (abs(o - prev_close) / atr) if atr and atr > 0 else np.nan
        return {
            "body": float(body),
            "range": float(bar_range),
            "close_pos": float(close_pos),
            "range_atr": float(range_atr) if np.isfinite(range_atr) else np.nan,
            "body_atr": float(body_atr) if np.isfinite(body_atr) else np.nan,
            "gap_atr": float(gap_atr) if np.isfinite(gap_atr) else np.nan,
            "atr": float(atr) if atr and np.isfinite(atr) else np.nan,
        }

    def _passes_quality_filters(
        self,
        cand: Dict,
        df_tf: pd.DataFrame,
        quality_overrides: Optional[Dict] = None,
    ) -> Tuple[bool, str, Dict]:
        cfg = self._quality_cfg or {}
        if not bool(cfg.get("enabled", True)):
            return True, "", {}

        stype = str(cand.get("strategy_type", "") or "")
        side = str(cand.get("signal", "") or "").upper()
        merged = dict(cfg)
        if isinstance(quality_overrides, dict) and quality_overrides:
            merged.update(quality_overrides)
        per_strategy = cfg.get("per_strategy", {}) or {}
        stype_cfg = per_strategy.get(stype)
        if isinstance(stype_cfg, dict):
            merged.update(stype_cfg)

        atr_period = int(self._safe_float(merged.get("atr_period", 14), 14.0))
        stats = self._trigger_stats(df_tf, atr_period)
        if not stats:
            return False, "insufficient_trigger_stats", {}

        body_atr = self._safe_float(stats.get("body_atr", np.nan), np.nan)
        range_atr = self._safe_float(stats.get("range_atr", np.nan), np.nan)
        gap_atr = self._safe_float(stats.get("gap_atr", np.nan), np.nan)
        close_pos = self._safe_float(stats.get("close_pos", 0.5), 0.5)

        min_body_atr = self._safe_float(merged.get("min_body_atr", 0.20), 0.20)
        if min_body_atr > 0 and np.isfinite(body_atr) and body_atr < min_body_atr:
            return False, f"body_atr {body_atr:.2f} < {min_body_atr:.2f}", {}

        max_range_atr = self._safe_float(merged.get("max_range_atr", 2.80), 2.80)
        if max_range_atr > 0 and np.isfinite(range_atr) and range_atr > max_range_atr:
            return False, f"range_atr {range_atr:.2f} > {max_range_atr:.2f}", {}

        max_gap_atr = self._safe_float(merged.get("max_gap_atr", 0.90), 0.90)
        if max_gap_atr > 0 and np.isfinite(gap_atr) and gap_atr > max_gap_atr:
            return False, f"gap_atr {gap_atr:.2f} > {max_gap_atr:.2f}", {}

        max_shock_range_atr = self._safe_float(merged.get("max_shock_range_atr", 0.0), 0.0)
        if max_shock_range_atr > 0 and np.isfinite(range_atr) and range_atr > max_shock_range_atr:
            return False, f"shock_range_atr {range_atr:.2f} > {max_shock_range_atr:.2f}", {}

        apply_close_pos_for = self._coerce_str_set(
            merged.get("apply_close_pos_for", ["Follow_Color", "Inside_Break"])
        )
        if stype in apply_close_pos_for:
            long_min_close_pos = self._safe_float(merged.get("long_min_close_pos", 0.55), 0.55)
            short_max_close_pos = self._safe_float(merged.get("short_max_close_pos", 0.45), 0.45)
            if side == "LONG" and close_pos < long_min_close_pos:
                return False, f"close_pos {close_pos:.2f} < {long_min_close_pos:.2f}", {}
            if side == "SHORT" and close_pos > short_max_close_pos:
                return False, f"close_pos {close_pos:.2f} > {short_max_close_pos:.2f}", {}

        return True, "", {
            "de2_trigger_body_atr": float(body_atr) if np.isfinite(body_atr) else None,
            "de2_trigger_range_atr": float(range_atr) if np.isfinite(range_atr) else None,
            "de2_trigger_gap_atr": float(gap_atr) if np.isfinite(gap_atr) else None,
            "de2_trigger_close_pos": float(close_pos),
            "de2_trigger_atr": float(stats.get("atr")) if np.isfinite(self._safe_float(stats.get("atr"), np.nan)) else None,
        }

    def _winrate_prior(self, cand: Dict, regime_norm: str) -> float:
        cfg = self._ev_cfg or {}
        p = self._safe_float(cfg.get("assumed_win_rate", 0.56), 0.56)
        regime_map = cfg.get("regime_win_rate", {}) or {}
        quarter_map = cfg.get("quarter_win_rate", {}) or {}
        day_map = cfg.get("day_win_rate", {}) or {}
        tf_map = cfg.get("timeframe_win_rate", {}) or {}
        strat_map = cfg.get("strategy_win_rate", {}) or {}
        sess_map = cfg.get("session_win_rate", {}) or {}
        day_sess_map = cfg.get("day_session_win_rate", {}) or {}
        day_tf_map = cfg.get("day_timeframe_win_rate", {}) or {}
        sid_map = cfg.get("strategy_id_win_rate", {}) or {}

        regime_key = str(regime_norm or "").lower()
        quarter_key = str(cand.get("quarter", "") or "")
        day_key = str(cand.get("day", "") or "")
        tf_key = str(cand.get("timeframe", "") or "")
        stype_key = str(cand.get("strategy_type", "") or "")
        sess_key = str(cand.get("session", "") or "")
        sid_key = str(cand.get("strategy_id", "") or "")
        day_sess_key = f"{day_key}|{sess_key}"
        day_tf_key = f"{day_key}|{tf_key}"

        p = self._map_lookup(regime_map, regime_key, p)
        p = self._map_lookup(quarter_map, quarter_key, p)
        p = self._map_lookup(day_map, day_key, p)
        p = self._map_lookup(tf_map, tf_key, p)
        p = self._map_lookup(strat_map, stype_key, p)
        p = self._map_lookup(sess_map, sess_key, p)
        p = self._map_lookup(day_tf_map, day_tf_key, p)
        p = self._map_lookup(day_sess_map, day_sess_key, p)
        p = self._map_lookup(sid_map, sid_key, p)

        return float(np.clip(p, 0.05, 0.95))

    @staticmethod
    def _ev_points(
        p_win: float,
        tp_points: float,
        sl_points: float,
        fees_per_side: float,
        point_value: float,
    ) -> float:
        fee_points = 0.0
        if point_value > 0:
            fee_points = (2.0 * fees_per_side) / point_value
        return float((p_win * tp_points) - ((1.0 - p_win) * sl_points) - fee_points)

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """Resamples 1m data to 5m/15m and queries the engine."""
        if df is None or len(df) < 60:
            return None

        current_time = df.index[-1]
        if self.last_processed_time == current_time:
            return None

        df_5m = self._resampler_5m.update(df)

        df_15m = self._resampler_15m.update(df)
        self.last_processed_time = current_time

        candidates: List[Dict] = []
        if hasattr(self.engine, "check_signals"):
            try:
                candidates = self.engine.check_signals(current_time, df_5m, df_15m) or []
            except Exception:
                candidates = []
        if not candidates:
            signal_data = self.engine.check_signal(current_time, df_5m, df_15m)
            candidates = [signal_data] if signal_data else []
        if not candidates:
            return None

        regime_norm = "unknown"
        try:
            regime_val, _, _ = volatility_filter.get_regime(df, current_time)
            regime_norm = self._normalize_regime(regime_val)
        except Exception:
            regime_norm = "unknown"

        risk_cfg = CONFIG.get("RISK", {}) or {}
        point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
        fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.50) or 2.50)
        min_net_profit = float(risk_cfg.get("MIN_NET_PROFIT", 10.0) or 10.0)
        enforce_min_net = bool(risk_cfg.get("ENFORCE_MIN_NET_PROFIT", True))
        num_contracts = int(risk_cfg.get("CONTRACTS", 1) or 1)
        if num_contracts < 1:
            num_contracts = 1

        min_cfg = CONFIG.get("SLTP_MIN", {}) or {}
        min_sl = float(min_cfg.get("sl", 1.25) or 1.25)
        min_tp = float(min_cfg.get("tp", 1.5) or 1.5)

        min_rr = self._safe_float(self._policy_cfg.get("min_rr", 0.0), 0.0)
        min_ev_points = self._safe_float(self._ev_cfg.get("min_ev_points", 0.0), 0.0)
        use_ev_per_risk = bool(self._ev_cfg.get("use_ev_per_risk", True))
        log_rerank = bool(self._ev_cfg.get("log_rerank", True))
        fee_points = (2.0 * fees_per_side / point_value) if point_value > 0 else 0.0
        soft_regime_penalty_ev = self._safe_float(
            self._regime_cfg.get("soft_penalty_ev_points", 0.10),
            0.10,
        )

        top_candidate_id = str(candidates[0].get("strategy_id", "") or "") if candidates else ""
        dynamic_params_cache: Dict[str, Dict] = {}
        feasible: List[Dict] = []

        for cand in candidates:
            cand_id = str(cand.get("strategy_id", "") or "")
            side = str(cand.get("signal", "") or "").upper()
            cand_tf = str(cand.get("timeframe", "") or "")
            if not cand_tf:
                cand_tf = "15min" if "_15m_" in cand_id.lower() else "5min"
            source_df = df_15m if str(cand_tf).lower().startswith("15") else df_5m
            if source_df is None or source_df.empty:
                continue

            regime_ok, regime_reason = self._is_regime_allowed(cand, regime_norm)
            ctx_profile = self._resolve_context_profile(cand, regime_norm)
            ctx_quality_overrides = ctx_profile.get("quality_overrides")
            if not regime_ok:
                if self._regime_mode == "hard":
                    if self._log_decisions:
                        logging.info("DE2 block %s: %s", cand_id, regime_reason)
                    continue

            quality_ok, quality_reason, quality_ctx = self._passes_quality_filters(
                cand,
                source_df,
                quality_overrides=ctx_quality_overrides if isinstance(ctx_quality_overrides, dict) else None,
            )
            if not quality_ok:
                if self._log_decisions:
                    logging.info("DE2 block %s: %s", cand_id, quality_reason)
                continue

            if side not in dynamic_params_cache:
                strat_key = "RegimeAdaptive_LONG" if side == "LONG" else "RegimeAdaptive_SHORT"
                dynamic_params_cache[side] = get_sltp(strat_key, df)
            dynamic_params = dynamic_params_cache[side]

            fixed_ok, fixed_details = apply_fixed_sltp(
                {
                    "side": side,
                    "strategy": "DynamicEngine2",
                    "sl_dist": max(float(cand.get("sl", min_sl) or min_sl), min_sl),
                    "tp_dist": max(float(cand.get("tp", min_tp) or min_tp), min_tp),
                },
                df,
                float(df["close"].iloc[-1]),
                ts=current_time,
            )
            if not fixed_ok:
                if self._log_decisions:
                    reason = fixed_details.get("reason", "FixedSLTP blocked") if isinstance(fixed_details, dict) else "FixedSLTP blocked"
                    logging.info("DE2 block %s: %s", cand_id, reason)
                continue

            if fixed_details:
                final_sl = float(fixed_details["sl_dist"])
                final_tp = float(fixed_details["tp_dist"])
            else:
                final_sl = max(float(dynamic_params.get("sl_dist", min_sl) or min_sl), min_sl)
                final_tp = max(float(dynamic_params.get("tp_dist", min_tp) or min_tp), min_tp)

            # Align runtime brackets with the originating DE2 edge profile so fees do not
            # collapse expectancy when fixed/dynamic brackets become too symmetric.
            align_cfg = self._policy_cfg.get("edge_bracket_alignment", {}) or {}
            if bool(align_cfg.get("enabled", False)):
                edge_rr = self._safe_float(cand.get("rr", 0.0), 0.0)
                edge_rr_blend = float(np.clip(self._safe_float(align_cfg.get("edge_rr_blend", 0.70), 0.70), 0.0, 1.0))
                min_edge_rr = self._safe_float(align_cfg.get("min_edge_rr", 1.25), 1.25)
                min_target_rr = self._safe_float(align_cfg.get("min_target_rr", 1.45), 1.45)
                rr_cap = self._safe_float(align_cfg.get("target_rr_cap", 3.00), 3.00)
                tp_raise_only = bool(align_cfg.get("tp_raise_only", True))
                allow_sl_tighten = bool(align_cfg.get("allow_sl_tighten", True))
                max_tp_points = self._safe_float(align_cfg.get("max_tp_points", 0.0), 0.0)
                curr_rr = (final_tp / final_sl) if final_sl > 0 else 0.0
                target_rr = curr_rr
                if edge_rr >= min_edge_rr:
                    blended_rr = ((1.0 - edge_rr_blend) * curr_rr) + (edge_rr_blend * edge_rr)
                    target_rr = max(curr_rr, blended_rr, min_target_rr)
                    if rr_cap > 0:
                        target_rr = min(target_rr, rr_cap)
                if final_sl > 0 and target_rr > curr_rr:
                    tp_from_target = final_sl * target_rr
                    if max_tp_points > 0:
                        tp_from_target = min(tp_from_target, max_tp_points)
                    if tp_raise_only:
                        final_tp = max(final_tp, tp_from_target)
                    else:
                        final_tp = tp_from_target
                    if allow_sl_tighten and final_tp > 0:
                        sl_from_target = max(min_sl, final_tp / target_rr)
                        final_sl = min(final_sl, sl_from_target)

            rr = (final_tp / final_sl) if final_sl > 0 else 0.0
            min_rr_eff = self._safe_float(ctx_profile.get("min_rr", min_rr), min_rr)
            if min_rr_eff > 0 and rr < min_rr_eff:
                if self._log_decisions:
                    logging.info("DE2 block %s: rr %.2f < %.2f", cand_id, rr, min_rr_eff)
                continue

            gross_profit = final_tp * point_value * num_contracts
            total_fees = fees_per_side * 2 * num_contracts
            net_profit = gross_profit - total_fees
            if enforce_min_net and net_profit < min_net_profit:
                if self._log_decisions:
                    logging.info(
                        "DE2 block %s: net %.2f < %.2f",
                        cand_id,
                        net_profit,
                        min_net_profit,
                    )
                continue

            p_win = self._winrate_prior(cand, regime_norm)
            p_win_bias = self._safe_float(ctx_profile.get("ev_winrate_bias", 0.0), 0.0)
            if p_win_bias != 0.0:
                p_win = float(np.clip(p_win + p_win_bias, 0.05, 0.95))
            den = max(final_tp + final_sl, 1e-9)
            breakeven_prob = float((final_sl + fee_points) / den)
            edge_prob = float(p_win - breakeven_prob)
            min_edge_prob_eff = self._safe_float(
                ctx_profile.get("min_edge_prob", self._stability_min_edge_prob),
                self._stability_min_edge_prob,
            )
            if edge_prob < min_edge_prob_eff:
                if self._log_decisions:
                    logging.info(
                        "DE2 block %s: edge_prob %.3f < %.3f",
                        cand_id,
                        edge_prob,
                        min_edge_prob_eff,
                    )
                continue

            ev_points_raw = self._ev_points(p_win, final_tp, final_sl, fees_per_side, point_value)
            density_penalty_ev, density_ctx = self._stability_penalty(
                cand=cand,
                regime_norm=regime_norm,
                current_time=current_time,
                ctx_profile=ctx_profile,
            )
            ev_points = float(ev_points_raw - density_penalty_ev)
            min_ev_points_eff = self._safe_float(ctx_profile.get("min_ev_points", min_ev_points), min_ev_points)
            if not regime_ok and self._regime_mode == "soft":
                ev_points -= soft_regime_penalty_ev
            if ev_points < min_ev_points_eff:
                if self._log_decisions:
                    logging.info("DE2 block %s: ev %.2f < %.2f", cand_id, ev_points, min_ev_points_eff)
                continue

            use_ev_per_risk_eff = bool(ctx_profile.get("use_ev_per_risk", use_ev_per_risk))
            rank_score = (ev_points / max(final_sl, 1e-6)) if use_ev_per_risk_eff else ev_points
            rank_boost = self._safe_float(ctx_profile.get("rank_boost", 0.0), 0.0)
            rank_score += rank_boost
            feasible.append(
                {
                    "cand": cand,
                    "source_df": source_df,
                    "final_sl": float(final_sl),
                    "final_tp": float(final_tp),
                    "rr": float(rr),
                    "ev_points": float(ev_points),
                    "ev_points_raw": float(ev_points_raw),
                    "rank_score": float(rank_score),
                    "p_win": float(p_win),
                    "breakeven_prob": float(breakeven_prob),
                    "edge_prob": float(edge_prob),
                    "net_profit": float(net_profit),
                    "quality_ctx": quality_ctx,
                    "density_ctx": density_ctx,
                    "dynamic_source": str(dynamic_params.get("hierarchy_key", "")),
                    "ctx_profile": ctx_profile,
                }
            )

        if not feasible:
            return None

        feasible.sort(
            key=lambda item: (
                float(item.get("rank_score", 0.0) or 0.0),
                float(item.get("ev_points", 0.0) or 0.0),
                float(item.get("rr", 0.0) or 0.0),
            ),
            reverse=True,
        )

        chosen = None
        drift_ctx = {}
        for entry in feasible:
            cand = entry["cand"]
            cand_id = str(cand.get("strategy_id", "") or "")
            side = str(cand.get("signal", "") or "").upper()
            drift_ok, dctx = self._passes_drift_gate(
                strategy_id=cand_id,
                side=side,
                current_time=current_time,
                current_price=float(df["close"].iloc[-1]),
                df_1m=df,
                df_tf=entry["source_df"],
                max_atr_override=self._safe_float(
                    (entry.get("ctx_profile") or {}).get("drift_max_atr", self._drift_max_atr),
                    self._drift_max_atr,
                ),
            )
            if drift_ok:
                chosen = entry
                drift_ctx = dctx
                break
            if self._log_decisions:
                logging.info(
                    "DE2 block %s: drift dist_atr %.2f > %.2f",
                    cand_id,
                    float(dctx.get("de2_drift_dist_atr", 0.0) or 0.0),
                    float(dctx.get("de2_drift_limit_atr", self._drift_max_atr) or self._drift_max_atr),
                )

        if chosen is None:
            return None

        cand = chosen["cand"]
        cand_id = str(cand.get("strategy_id", "") or "")
        side = str(cand.get("signal", "") or "").upper()
        if log_rerank and top_candidate_id and cand_id != top_candidate_id:
            logging.info(
                "DE2 re-rank: top=%s -> selected=%s (rank=%.3f ev=%.3f rr=%.2f)",
                top_candidate_id,
                cand_id,
                float(chosen.get("rank_score", 0.0) or 0.0),
                float(chosen.get("ev_points", 0.0) or 0.0),
                float(chosen.get("rr", 0.0) or 0.0),
            )

        logging.info("🚀 DynamicEngine2 selected: %s %s", side, cand_id)
        logging.info(
            "DE2 metrics: ev=%.2f raw_ev=%.2f rank=%.3f p=%.3f edge=%.3f rr=%.2f regime=%s",
            float(chosen["ev_points"]),
            float(chosen.get("ev_points_raw", chosen["ev_points"])),
            float(chosen["rank_score"]),
            float(chosen["p_win"]),
            float(chosen.get("edge_prob", 0.0)),
            float(chosen["rr"]),
            regime_norm,
        )
        logging.info(
            "📉 DynamicEngine2 SLTP: static %.2f/%.2f -> %.2f/%.2f (source=%s)",
            float(cand.get("sl", 0.0) or 0.0),
            float(cand.get("tp", 0.0) or 0.0),
            float(chosen["final_sl"]),
            float(chosen["final_tp"]),
            chosen.get("dynamic_source", ""),
        )

        signal = {
            "strategy": "DynamicEngine2",
            "sub_strategy": cand_id,
            "side": side,
            "tp_dist": float(chosen["final_tp"]),
            "sl_dist": float(chosen["final_sl"]),
            "de2_ev_points": float(chosen["ev_points"]),
            "de2_ev_points_raw": float(chosen.get("ev_points_raw", chosen["ev_points"])),
            "de2_rank_score": float(chosen["rank_score"]),
            "de2_winrate_prior": float(chosen["p_win"]),
            "de2_breakeven_prob": float(chosen.get("breakeven_prob", 0.0)),
            "de2_edge_prob": float(chosen.get("edge_prob", 0.0)),
            "de2_rr": float(chosen["rr"]),
            "de2_regime": regime_norm,
        }
        if isinstance(chosen.get("quality_ctx"), dict):
            signal.update(chosen["quality_ctx"])
        if isinstance(chosen.get("density_ctx"), dict):
            signal.update(chosen["density_ctx"])
        if drift_ctx:
            signal.update(drift_ctx)
        self._record_signal(cand=cand, regime_norm=regime_norm, current_time=current_time)
        return signal

        return None
