import logging
from typing import Dict, Optional

import pandas as pd

from dynamic_signal_engine import get_signal_engine
from strategy_base import Strategy


class DynamicEngineStrategy(Strategy):
    """
    Wrapper for DynamicSignalEngine to run 235 hardcoded strategies
    alongside Julie's existing filters.
    """

    def __init__(self):
        self.engine = get_signal_engine()
        self.last_processed_time = None

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
            logging.info(f"🚀 DYNAMIC ENGINE TRIGGER: {signal_data['strategy_id']}")

            return {
                "strategy": "DynamicEngine",
                "sub_strategy": signal_data['strategy_id'],
                "side": signal_data['signal'],
                "tp_dist": signal_data['tp'],
                "sl_dist": signal_data['sl']
            }

        return None
