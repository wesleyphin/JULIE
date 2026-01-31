import logging
from typing import Dict, Optional

import pandas as pd

from dynamic_signal_engine import get_signal_engine
from fixed_sltp_framework import apply_fixed_sltp
from volatility_filter import volatility_filter
from strategy_base import Strategy
from config import CONFIG


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
        num_strategies = len(self.engine.strategies) if hasattr(self.engine, 'strategies') else 235
        logging.info(f"DynamicEngineStrategy initialized | {num_strategies} sub-strategies loaded")

    @staticmethod
    def _is_bar_close(ts: pd.Timestamp, minutes: int) -> bool:
        return ts.minute % minutes == minutes - 1 and ts.second == 0

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """Resamples 1m data to 5m/15m and queries the engine."""
        if df is None or len(df) < 60:
            return None

        current_time = df.index[-1]

        if self.last_processed_time == current_time:
            return None

        if not self._is_bar_close(current_time, 5):
            return None

        update_5m = self._cached_5m is None or self._last_5m_close != current_time
        update_15m = self._cached_15m is None or (
            self._is_bar_close(current_time, 15) and self._last_15m_close != current_time
        )

        if update_5m:
            df_5m = df.resample('5min', closed='left', label='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            if df_5m.empty:
                return None
            self._cached_5m = df_5m
            self._last_5m_close = current_time
        else:
            df_5m = self._cached_5m

        if update_15m:
            df_15m = df.resample('15min', closed='left', label='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            if not self._is_bar_close(current_time, 15) and len(df_15m) > 0:
                df_15m = df_15m.iloc[:-1]
            if df_15m.empty:
                return None
            self._cached_15m = df_15m
            self._last_15m_close = df_15m.index[-1]
        else:
            df_15m = self._cached_15m

        if df_5m is None or df_15m is None:
            return None

        self.last_processed_time = current_time
        signal_data = self.engine.check_signal(current_time, df_5m, df_15m)

        if signal_data:
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

            logging.info(f"DynamicEngine: {signal_data['signal']} signal from {signal_data['strategy_id']}")
            logging.info(f"   TP: {final_tp:.2f} | SL: {final_sl:.2f}")
            logging.info(f"   Net profit: ${net_profit:.2f} (gross ${gross_profit:.2f} - fees ${total_fees:.2f})")

            return {
                "strategy": "DynamicEngine",
                "sub_strategy": signal_data['strategy_id'],
                "side": signal_data['signal'],
                "tp_dist": final_tp,
                "sl_dist": final_sl
            }

        return None
