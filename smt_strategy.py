import logging
from typing import Dict, Optional

import pandas as pd

from dynamic_sltp_params import dynamic_sltp_engine
from smt_analyzer import SMTAnalyzer
from strategy_base import Strategy


class SMTStrategy(Strategy):
    """Strategy wrapper that converts SMTAnalyzer signals into bot actions."""

    def __init__(self, mnq_client, lookback_minutes: int = 1500):
        self.mnq_client = mnq_client
        self.lookback_minutes = lookback_minutes
        self.analyzer = SMTAnalyzer()

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

    def on_bar(self, df_mes: pd.DataFrame) -> Optional[Dict]:
        if df_mes.empty:
            return None

        # Fetch synchronized MNQ data using its dedicated client
        df_mnq = self.mnq_client.get_market_data(
            lookback_minutes=self.lookback_minutes
        )
        if df_mnq.empty:
            logging.debug("SMTStrategy: MNQ data unavailable")
            return None

        df_mnq_prepared = self._prepare_df(df_mnq)
        df_mes_prepared = self._prepare_df(df_mes)

        if df_mnq_prepared.empty or df_mes_prepared.empty:
            return None

        signals = self.analyzer.generate_signals(df_mnq_prepared, df_mes_prepared)
        latest = signals["signal"].iloc[-1]

        if latest == 0:
            return None

        side = "LONG" if latest == 1 else "SHORT"
        sltp = dynamic_sltp_engine.calculate_dynamic_sltp(df_mes)

        return {
            "strategy": "SMTAnalyzer",
            "side": side,
            "tp_dist": sltp["tp_dist"],
            "sl_dist": sltp["sl_dist"],
        }
