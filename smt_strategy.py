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

        # Internal storage for MNQ history
        self.mnq_history = pd.DataFrame()
        self.is_initialized = False

    def _update_mnq_history(self) -> pd.DataFrame:
        """Handles the incremental fetching logic for MNQ."""
        try:
            if not self.is_initialized:
                # FIRST RUN: Fetch the massive 20k bar chunk
                # We use 20000 to match the main bot, ensuring deep history for alignment
                logging.info(f"SMTStrategy: Fetching initial MNQ history (Max)...")
                new_data = self.mnq_client.get_market_data(lookback_minutes=20000, force_fetch=True)

                if not new_data.empty:
                    self.mnq_history = new_data
                    self.is_initialized = True
            else:
                # SUBSEQUENT RUNS: Fetch only the last 15 minutes
                new_data = self.mnq_client.get_market_data(lookback_minutes=15, force_fetch=True)

                if not new_data.empty:
                    # Combine and Deduplicate
                    self.mnq_history = pd.concat([self.mnq_history, new_data])
                    self.mnq_history = self.mnq_history[~self.mnq_history.index.duplicated(keep='last')]

                    # Trim to keep memory usage reasonable (keep slightly more than lookback)
                    # We keep 5000 bars to be safe for SMT calculations
                    if len(self.mnq_history) > 5000:
                        self.mnq_history = self.mnq_history.iloc[-5000:]

            return self.mnq_history

        except Exception as e:
            logging.error(f"SMT Data Error: {e}")
            return self.mnq_history

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # Normalize column names to Capitalized (Open, High...) for Analyzer
        cols = {c: c.capitalize() for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns}
        return df.rename(columns=cols)

    def on_bar(self, df_mes: pd.DataFrame) -> Optional[Dict]:
        if df_mes.empty:
            return None

        # 1. Get MNQ Data using our new incremental method
        df_mnq = self._update_mnq_history()

        if df_mnq.empty:
            # logging.debug("SMTStrategy: MNQ data unavailable") # Reduce noise
            return None

        df_mnq_prepared = self._prepare_df(df_mnq)
        df_mes_prepared = self._prepare_df(df_mes)

        # 2. Sync Check: Ensure MNQ data isn't stale compared to MES
        if df_mnq_prepared.index[-1] < df_mes_prepared.index[-1] - pd.Timedelta(minutes=5):
            logging.warning("SMTStrategy: MNQ data is lagging MES data. Skipping signal.")
            return None

        # 3. Run Analysis
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
