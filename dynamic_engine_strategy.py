import logging
from typing import Dict, Optional

import pandas as pd

from dynamic_signal_engine import get_signal_engine
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
        num_strategies = len(self.engine.strategies) if hasattr(self.engine, 'strategies') else 235
        logging.info(f"DynamicEngineStrategy initialized | {num_strategies} sub-strategies loaded")

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """Resamples 1m data to 5m/15m and queries the engine."""
        if df is None or len(df) < 60:
            return None

        current_time = df.index[-1]

        df_5m = df.resample('5min', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        df_15m = df.resample('15min', closed='left', label='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()

        signal_data = self.engine.check_signal(current_time, df_5m, df_15m)

        if signal_data:
            # --- FEE & PROFITABILITY CHECK ---
            tp_dist = signal_data['tp']

            # Load risk settings from config or use defaults
            risk_cfg = CONFIG.get("RISK", {})
            point_value = risk_cfg.get("POINT_VALUE", 5.0)
            fees_per_side = risk_cfg.get("FEES_PER_SIDE", 2.50)
            min_net_profit = risk_cfg.get("MIN_NET_PROFIT", 10.0)
            num_contracts = risk_cfg.get("CONTRACTS", 1)

            gross_profit = tp_dist * point_value * num_contracts
            total_fees = fees_per_side * 2 * num_contracts  # Round trip
            net_profit = gross_profit - total_fees

            if net_profit < min_net_profit:
                logging.warning(
                    f"🚫 BLOCKED (Fees): {signal_data['strategy_id']} | "
                    f"Gross: ${gross_profit:.2f} - Fees: ${total_fees:.2f} = Net: ${net_profit:.2f} "
                    f"(< Min ${min_net_profit:.2f})"
                )
                return None

            logging.info(f"DynamicEngine: {signal_data['signal']} signal from {signal_data['strategy_id']}")
            logging.info(f"   TP: {signal_data['tp']:.2f} | SL: {signal_data['sl']:.2f}")
            logging.info(f"   Net profit: ${net_profit:.2f} (gross ${gross_profit:.2f} - fees ${total_fees:.2f})")

            # Enforce minimum SL/TP for positive RR
            MIN_SL = 4.0  # 16 ticks minimum
            MIN_TP = 6.0  # 24 ticks minimum (1.5:1 RR)

            final_sl = max(signal_data['sl'], MIN_SL)
            final_tp = max(signal_data['tp'], MIN_TP)

            return {
                "strategy": "DynamicEngine",
                "sub_strategy": signal_data['strategy_id'],
                "side": signal_data['signal'],
                "tp_dist": final_tp,
                "sl_dist": final_sl
            }

        return None
