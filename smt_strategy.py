from typing import Dict, Optional

import pandas as pd

from dynamic_sltp_params import dynamic_sltp_engine
from smt_analyzer import SMTAnalyzer
from strategy_base import Strategy


class SMTStrategy(Strategy):
    """
    Passive SMT Strategy.

    This strategy no longer fetches its own MNQ data. The main bot supplies
    both MES and MNQ dataframes so we avoid duplicate API requests.
    """

    def __init__(self, lookback_minutes: int = 1500):
        self.lookback_minutes = lookback_minutes
        self.analyzer = SMTAnalyzer()

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Normalize column names to Capitalized (Open, High...) for Analyzer
        cols = {c: c.capitalize() for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns}
        return df.rename(columns=cols)

    def on_bar(self, df_mes: pd.DataFrame, df_mnq: pd.DataFrame = None) -> Optional[Dict]:
        if df_mes.empty or df_mnq is None or df_mnq.empty:
            return None

        df_mnq_prepared = self._prepare_df(df_mnq)
        df_mes_prepared = self._prepare_df(df_mes)

        # Sync Check: Ensure MNQ data isn't stale compared to MES
        # Still helpful even when fetched sequentially in the main loop
        if df_mnq_prepared.index[-1] < df_mes_prepared.index[-1] - pd.Timedelta(minutes=5):
            return None

        signals = self.analyzer.generate_signals(df_mnq_prepared, df_mes_prepared)

        if signals.empty:
            return None

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
