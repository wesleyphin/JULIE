import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from dynamic_signal_engine import get_signal_engine
from fixed_sltp_framework import apply_fixed_sltp
from volatility_filter import volatility_filter
from strategy_base import Strategy
from config import CONFIG
from incremental_ohlcv_resampler import IncrementalOHLCVResampler


class DynamicEngineStrategy(Strategy):
    """
    Wrapper for DynamicSignalEngine to run 235 hardcoded strategies
    alongside Julie's existing filters.
    """

    def __init__(self):
        self.strategy_name = "DynamicEngine"
        self.engine = get_signal_engine()
        self.last_processed_time = None
        self._last_5m_close = None
        self._last_15m_close = None
        self._cached_5m = None
        self._cached_15m = None
        self._resampler_5m = IncrementalOHLCVResampler(5)
        self._resampler_15m = IncrementalOHLCVResampler(15)
        self._policy_cfg = CONFIG.get("DYNAMIC_ENGINE_DE1_POLICY", {}) or {}
        self._last_signal_ts: Optional[pd.Timestamp] = None
        drift_cfg = CONFIG.get("DYNAMIC_ENGINE_DE1_DRIFT", {}) or {}
        self._drift_enabled = bool(drift_cfg.get("enabled", True))
        self._drift_max_atr = self._safe_float(drift_cfg.get("max_atr", 1.0), 1.0)
        self._drift_atr_period = int(self._safe_float(drift_cfg.get("atr_period", 14), 14.0))
        self._drift_fallback_points = self._safe_float(drift_cfg.get("fallback_points", 0.0), 0.0)
        self._drift_anchors: Dict[Tuple[str, str, pd.Timestamp], float] = {}
        num_strategies = len(self.engine.strategies) if hasattr(self.engine, 'strategies') else 235
        logging.info(f"DynamicEngineStrategy initialized | {num_strategies} sub-strategies loaded")

    @staticmethod
    def _is_bar_close(ts: pd.Timestamp, minutes: int) -> bool:
        return ts.minute % minutes == minutes - 1 and ts.second == 0

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
    def _tf_minutes(tf: str) -> int:
        tf_norm = str(tf or "").lower()
        if tf_norm.startswith("15"):
            return 15
        return 5

    @staticmethod
    def _signal_bucket(ts, tf: str, strategy_id: str = "") -> pd.Timestamp:
        ts_val = pd.Timestamp(ts)
        tf_norm = str(tf or "").lower()
        sid = str(strategy_id or "").lower()
        if "15" in tf_norm or "_15" in sid:
            return ts_val.floor("15min")
        if "5" in tf_norm or "_5" in sid:
            return ts_val.floor("5min")
        return ts_val.floor("1min")

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

    @classmethod
    def _trigger_bar_stats(cls, df_tf: pd.DataFrame, atr_period: int) -> Dict[str, float]:
        if df_tf is None or len(df_tf) < 3:
            return {}
        trigger = df_tf.iloc[-2]
        try:
            bar_open = float(trigger["open"])
            bar_high = float(trigger["high"])
            bar_low = float(trigger["low"])
            bar_close = float(trigger["close"])
        except Exception:
            return {}

        bar_range = max(0.0, bar_high - bar_low)
        body = abs(bar_close - bar_open)
        close_pos = (bar_close - bar_low) / bar_range if bar_range > 0 else 0.5
        body_to_range = body / bar_range if bar_range > 0 else 0.0

        # ATR should be computed up to the trigger candle (exclude in-progress bar).
        atr = cls._compute_atr_simple(df_tf.iloc[:-1], atr_period)
        range_atr = (bar_range / atr) if atr and atr > 0 else np.nan
        body_atr = (body / atr) if atr and atr > 0 else np.nan

        return {
            "body": float(body),
            "range": float(bar_range),
            "close_pos": float(close_pos),
            "body_to_range": float(body_to_range),
            "range_atr": float(range_atr) if np.isfinite(range_atr) else np.nan,
            "body_atr": float(body_atr) if np.isfinite(body_atr) else np.nan,
        }

    def _is_cooldown_active(self, ts: pd.Timestamp, tf_minutes: int) -> bool:
        bars = int(self._safe_float(self._policy_cfg.get("cooldown_bars", 0), 0.0))
        if bars <= 0 or self._last_signal_ts is None:
            return False
        try:
            elapsed = (pd.Timestamp(ts) - pd.Timestamp(self._last_signal_ts)).total_seconds() / 60.0
        except Exception:
            return False
        cooldown_minutes = float(max(1, tf_minutes) * bars)
        return elapsed < cooldown_minutes

    def _mark_signal(self, ts: pd.Timestamp) -> None:
        try:
            self._last_signal_ts = pd.Timestamp(ts)
        except Exception:
            self._last_signal_ts = None

    def _prune_drift_anchors(self, current_time: pd.Timestamp) -> None:
        if not self._drift_anchors:
            return
        cutoff = pd.Timestamp(current_time) - pd.Timedelta(hours=8)
        self._drift_anchors = {k: v for k, v in self._drift_anchors.items() if pd.Timestamp(k[2]) >= cutoff}

    def _passes_drift_gate(
        self,
        *,
        strategy_id: str,
        side: str,
        signal_tf: str,
        current_time: pd.Timestamp,
        current_price: float,
        df_1m: pd.DataFrame,
        df_tf: Optional[pd.DataFrame],
    ) -> Tuple[bool, Dict]:
        if not self._drift_enabled or self._drift_max_atr <= 0:
            return True, {}
        self._prune_drift_anchors(current_time)

        bucket = self._signal_bucket(current_time, signal_tf, strategy_id)
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
                "de1_drift_anchor": float(anchor),
                "de1_drift_dist_points": float(dist_points),
                "de1_drift_dist_atr": None,
                "de1_drift_limit_atr": float(self._drift_max_atr),
            }

        dist_atr = float(dist_points / float(atr_value))
        ok = dist_atr <= float(self._drift_max_atr)
        return ok, {
            "de1_drift_anchor": float(anchor),
            "de1_drift_dist_points": float(dist_points),
            "de1_drift_dist_atr": float(dist_atr),
            "de1_drift_limit_atr": float(self._drift_max_atr),
            "de1_drift_atr": float(atr_value),
        }

    def _apply_de1_policy(
        self,
        signal_data: Dict,
        df_1m: pd.DataFrame,
        df_tf: pd.DataFrame,
        current_time: pd.Timestamp,
    ) -> Tuple[bool, str, Dict]:
        cfg = self._policy_cfg or {}
        if not bool(cfg.get("enabled", True)):
            return True, "", {}

        tf = str(signal_data.get("timeframe", "5min") or "5min")
        stype = str(signal_data.get("strategy_type", "") or "").lower()
        side = str(signal_data.get("signal", "") or "").upper()
        tf_minutes = self._tf_minutes(tf)

        if self._is_cooldown_active(current_time, tf_minutes):
            return False, f"cooldown active ({int(cfg.get('cooldown_bars', 0) or 0)} bars)", {}

        regime_norm = "unknown"
        try:
            regime, _, _ = volatility_filter.get_regime(df_1m, current_time)
            regime_norm = str(regime or "").lower()
        except Exception:
            regime_norm = "unknown"

        opt_wr = self._safe_float(signal_data.get("opt_wr", 0.0), 0.0)
        final_score = self._safe_float(signal_data.get("final_score", 0.0), 0.0)
        body = self._safe_float(signal_data.get("body", 0.0), 0.0)
        thresh = max(self._safe_float(signal_data.get("thresh", 0.0), 0.0), 1e-6)
        body_thresh_ratio = body / thresh

        min_opt_wr = self._safe_float(cfg.get("min_opt_wr", 0.0), 0.0)
        if opt_wr < min_opt_wr:
            return False, f"opt_wr {opt_wr:.3f} < {min_opt_wr:.3f}", {}

        min_final_score = cfg.get("min_final_score")
        if min_final_score is not None:
            min_final = self._safe_float(min_final_score, np.nan)
            if np.isfinite(min_final) and final_score < min_final:
                return False, f"final_score {final_score:.2f} < {min_final:.2f}", {}

        min_body_thresh_ratio = self._safe_float(cfg.get("min_body_thresh_ratio", 1.0), 1.0)
        if body_thresh_ratio < min_body_thresh_ratio:
            return False, f"body/thresh {body_thresh_ratio:.2f} < {min_body_thresh_ratio:.2f}", {}

        is_momentum = stype.endswith("mom")
        is_reversion = stype.endswith("rev")
        if is_reversion:
            max_rev_ratio = self._safe_float(cfg.get("max_reversion_body_thresh_ratio", 0.0), 0.0)
            if max_rev_ratio > 0 and body_thresh_ratio > max_rev_ratio:
                return False, f"reversion body/thresh {body_thresh_ratio:.2f} > {max_rev_ratio:.2f}", {}

        mom_allowed = {str(x).lower() for x in (cfg.get("momentum_allowed_regimes", []) or [])}
        rev_allowed = {str(x).lower() for x in (cfg.get("reversion_allowed_regimes", []) or [])}
        if is_momentum and mom_allowed and regime_norm not in mom_allowed:
            return False, f"momentum blocked in regime={regime_norm}", {}
        if is_reversion and rev_allowed and regime_norm not in rev_allowed:
            return False, f"reversion blocked in regime={regime_norm}", {}

        atr_period = int(self._safe_float(cfg.get("atr_period", 14), 14.0))
        stats = self._trigger_bar_stats(df_tf, atr_period)
        if stats:
            close_pos = self._safe_float(stats.get("close_pos", 0.5), 0.5)
            body_to_range = self._safe_float(stats.get("body_to_range", 0.0), 0.0)
            range_atr = self._safe_float(stats.get("range_atr", np.nan), np.nan)
            body_atr = self._safe_float(stats.get("body_atr", np.nan), np.nan)

            if is_momentum:
                mom_min_body_range = self._safe_float(cfg.get("momentum_min_body_range_ratio", 0.50), 0.50)
                if body_to_range < mom_min_body_range:
                    return False, f"momentum body/range {body_to_range:.2f} < {mom_min_body_range:.2f}", {}
                long_min = self._safe_float(cfg.get("momentum_long_min_close_pos", 0.65), 0.65)
                short_max = self._safe_float(cfg.get("momentum_short_max_close_pos", 0.35), 0.35)
                if side == "LONG" and close_pos < long_min:
                    return False, f"momentum close_pos {close_pos:.2f} < {long_min:.2f}", {}
                if side == "SHORT" and close_pos > short_max:
                    return False, f"momentum close_pos {close_pos:.2f} > {short_max:.2f}", {}
            elif is_reversion:
                long_max = self._safe_float(cfg.get("reversion_long_max_close_pos", 0.35), 0.35)
                short_min = self._safe_float(cfg.get("reversion_short_min_close_pos", 0.65), 0.65)
                if side == "LONG" and close_pos > long_max:
                    return False, f"reversion close_pos {close_pos:.2f} > {long_max:.2f}", {}
                if side == "SHORT" and close_pos < short_min:
                    return False, f"reversion close_pos {close_pos:.2f} < {short_min:.2f}", {}

            max_range_atr = self._safe_float(cfg.get("max_trigger_range_atr", 0.0), 0.0)
            if max_range_atr > 0 and np.isfinite(range_atr) and range_atr > max_range_atr:
                return False, f"trigger range/ATR {range_atr:.2f} > {max_range_atr:.2f}", {}

            min_mom_body_atr = self._safe_float(cfg.get("momentum_min_body_atr", 0.0), 0.0)
            if is_momentum and min_mom_body_atr > 0 and np.isfinite(body_atr) and body_atr < min_mom_body_atr:
                return False, f"momentum body/ATR {body_atr:.2f} < {min_mom_body_atr:.2f}", {}

        context = {
            "de1_policy": "sharpe_identity_v1",
            "de1_regime": regime_norm,
            "de1_body_thresh_ratio": float(body_thresh_ratio),
            "de1_opt_wr": float(opt_wr),
            "de1_final_score": float(final_score),
        }
        for key, val in stats.items():
            context[f"de1_trigger_{key}"] = val
        return True, "", context

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """Resamples 1m data to 5m/15m and queries the engine."""
        if df is None or len(df) < 60:
            return None

        current_time = df.index[-1]

        if self.last_processed_time == current_time:
            return None

        df_5m = self._resampler_5m.update(df)
        if df_5m.empty:
            return None
        self._cached_5m = df_5m
        self._last_5m_close = df_5m.index[-1]

        df_15m = self._resampler_15m.update(df)
        if df_15m.empty:
            return None
        self._cached_15m = df_15m
        self._last_15m_close = df_15m.index[-1]

        if df_5m is None or df_15m is None:
            return None

        self.last_processed_time = current_time
        signal_data = self.engine.check_signal(current_time, df_5m, df_15m)

        if signal_data:
            signal_tf = str(signal_data.get("timeframe", "5min") or "5min").lower()
            signal_id = str(signal_data.get("strategy_id", "") or "")
            signal_side = str(signal_data.get("signal", "") or "").upper()

            engine_session = None
            try:
                engine_session = self.engine.get_session_from_time(current_time)
            except Exception:
                engine_session = None

            ny_conf = CONFIG.get("DYNAMIC_ENGINE_NY_CONF", {}) or {}
            if ny_conf.get("enabled"):
                if engine_session and engine_session in set(ny_conf.get("sessions", [])):
                    try:
                        min_opt_wr = float(ny_conf.get("min_opt_wr", 0.0))
                    except Exception:
                        min_opt_wr = 0.0
                    min_final_score = ny_conf.get("min_final_score")
                    try:
                        opt_wr = float(signal_data.get("opt_wr", 0.0))
                    except Exception:
                        opt_wr = 0.0
                    try:
                        final_score = float(signal_data.get("final_score", 0.0))
                    except Exception:
                        final_score = 0.0

                    if opt_wr < min_opt_wr:
                        logging.warning(
                            f"🚫 DynamicEngine NY gate: {signal_data.get('strategy_id')} "
                            f"opt_wr {opt_wr:.3f} < {min_opt_wr:.3f} (session {engine_session})"
                        )
                        return None
                    if min_final_score is not None:
                        try:
                            min_final_score = float(min_final_score)
                        except Exception:
                            min_final_score = None
                        if min_final_score is not None and final_score < min_final_score:
                            logging.warning(
                                f"🚫 DynamicEngine NY gate: {signal_data.get('strategy_id')} "
                                f"score {final_score:.2f} < {min_final_score:.2f} (session {engine_session})"
                            )
                            return None

            source_df = df_15m if signal_tf.startswith("15") else df_5m
            policy_ok, policy_reason, policy_ctx = self._apply_de1_policy(
                signal_data=signal_data,
                df_1m=df,
                df_tf=source_df,
                current_time=current_time,
            )
            if not policy_ok:
                logging.warning(
                    "🚫 DynamicEngine DE1 policy: %s | %s",
                    signal_data.get("strategy_id"),
                    policy_reason,
                )
                return None

            # --- FIXED SL/TP FRAMEWORK (if enabled) ---
            sltp_session = volatility_filter.get_session(current_time.hour)
            fixed_ok, fixed_details = apply_fixed_sltp(
                {"side": signal_data["signal"], "strategy": "DynamicEngine"},
                df,
                float(df["close"].iloc[-1]),
                ts=current_time,
                session=sltp_session,
            )
            if not fixed_ok:
                reason = fixed_details.get("reason", "FixedSLTP blocked")
                logging.warning(f"🚫 DynamicEngine FixedSLTP: {reason}")
                return None

            if fixed_details:
                final_sl = float(fixed_details["sl_dist"])
                final_tp = float(fixed_details["tp_dist"])
            else:
                min_cfg = CONFIG.get("SLTP_MIN", {}) or {}
                min_sl = float(min_cfg.get("sl", 1.25))
                min_tp = float(min_cfg.get("tp", 1.5))
                final_sl = max(float(signal_data["sl"]), min_sl)
                final_tp = max(float(signal_data["tp"]), min_tp)

            tp_dist = final_tp
            rr = (final_tp / final_sl) if final_sl > 0 else 0.0
            min_rr = self._safe_float(self._policy_cfg.get("min_rr", 0.0), 0.0)
            if min_rr > 0 and rr < min_rr:
                logging.warning(
                    "🚫 DynamicEngine DE1 policy: %s | RR %.2f < %.2f",
                    signal_data.get("strategy_id"),
                    rr,
                    min_rr,
                )
                return None

            # Load risk settings from config or use defaults
            risk_cfg = CONFIG.get("RISK", {})
            point_value = risk_cfg.get("POINT_VALUE", 5.0)
            fees_per_side = risk_cfg.get("FEES_PER_SIDE", 2.50)
            min_net_profit = risk_cfg.get("MIN_NET_PROFIT", 10.0)
            enforce_min_net = bool(risk_cfg.get("ENFORCE_MIN_NET_PROFIT", True))
            num_contracts = risk_cfg.get("CONTRACTS", 1)

            gross_profit = tp_dist * point_value * num_contracts
            total_fees = fees_per_side * 2 * num_contracts  # Round trip
            net_profit = gross_profit - total_fees

            if enforce_min_net and net_profit < min_net_profit:
                logging.warning(
                    f"🚫 BLOCKED (Fees): {signal_data['strategy_id']} | "
                    f"Gross: ${gross_profit:.2f} - Fees: ${total_fees:.2f} = Net: ${net_profit:.2f} "
                    f"(< Min ${min_net_profit:.2f})"
                )
                return None

            current_price = float(df["close"].iloc[-1])
            drift_ok, drift_ctx = self._passes_drift_gate(
                strategy_id=signal_id,
                side=signal_side,
                signal_tf=signal_tf,
                current_time=current_time,
                current_price=current_price,
                df_1m=df,
                df_tf=source_df,
            )
            if not drift_ok:
                logging.info(
                    "DynamicEngine drift block: %s %s dist_atr=%.2f > %.2f",
                    signal_id,
                    signal_side,
                    float(drift_ctx.get("de1_drift_dist_atr", 0.0) or 0.0),
                    float(self._drift_max_atr),
                )
                return None

            logging.info(f"DynamicEngine: {signal_data['signal']} signal from {signal_data['strategy_id']}")
            logging.info(f"   TP: {final_tp:.2f} | SL: {final_sl:.2f}")
            logging.info(f"   Net profit: ${net_profit:.2f} (gross ${gross_profit:.2f} - fees ${total_fees:.2f})")
            if policy_ctx:
                logging.info(
                    "   DE1 policy: regime=%s body/thresh=%.2f score=%.2f wr=%.3f rr=%.2f",
                    policy_ctx.get("de1_regime"),
                    float(policy_ctx.get("de1_body_thresh_ratio", 0.0) or 0.0),
                    float(policy_ctx.get("de1_final_score", 0.0) or 0.0),
                    float(policy_ctx.get("de1_opt_wr", 0.0) or 0.0),
                    rr,
                )

            self._mark_signal(current_time)

            signal = {
                "strategy": "DynamicEngine",
                "sub_strategy": signal_data['strategy_id'],
                "side": signal_data['signal'],
                "tp_dist": final_tp,
                "sl_dist": final_sl
            }
            if policy_ctx:
                signal.update(policy_ctx)
            if drift_ctx:
                signal.update(drift_ctx)
            return signal

        return None
