import datetime
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from dynamic_sltp_params import dynamic_sltp_engine
from strategy_base import Strategy


class IntradayDipStrategy(Strategy):
    """
    IntradayDip Strategy - Now uses DynamicSLTPEngine
    - LONG: Price down >= 1.0% from session open, z-score < -0.5, volatility spike
    - SHORT: Price up >= 1.25% from session open, z-score > 1.0, volatility spike
    """
    def __init__(self):
        self.session_open = None
        self.current_date = None

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 20:
            return None

        ts = df.index[-1]
        curr = df.iloc[-1]

        # Reset on new day
        if self.current_date != ts.date():
            self.session_open = None
            self.current_date = ts.date()

        # Capture session open at 9:30 ET
        if ts.time() == datetime.time(9, 30):
            self.session_open = curr['open']

        if self.session_open is None:
            return None

        # Calculate intraday % change from session open
        pct_change = (curr['close'] - self.session_open) / self.session_open * 100

        # Z-score calculation
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        std20 = df['close'].rolling(20).std().iloc[-1]
        if std20 == 0:
            return None
        z_score = (curr['close'] - sma20) / std20

        # Volatility spike detection
        range_series = df['high'] - df['low']
        range_sma = range_series.rolling(20).mean().iloc[-1]
        curr_range = curr['high'] - curr['low']
        is_vol_spike = curr_range > range_sma

        # LONG: Down 1%+, oversold (z < -0.5), volatility spike
        if (pct_change <= -1.0) and (z_score < -0.5) and is_vol_spike:
            # Get dynamic SL/TP with strategy name for proper lookup
            sltp = dynamic_sltp_engine.calculate_sltp("IntradayDip_LONG", df)
            logging.info(f"IntradayDip: LONG signal - Down {pct_change:.2f}%, Z={z_score:.2f}")
            dynamic_sltp_engine.log_params(sltp, "IntradayDip_LONG")
            return {"strategy": "IntradayDip", "side": "LONG",
                    "tp_dist": sltp['tp_dist'], "sl_dist": sltp['sl_dist']}

        # SHORT: Up 1.25%+, overbought (z > 1.0), volatility spike
        if (pct_change >= 1.25) and (z_score > 1.0) and is_vol_spike:
            # Get dynamic SL/TP with strategy name for proper lookup
            sltp = dynamic_sltp_engine.calculate_sltp("IntradayDip_SHORT", df)
            logging.info(f"IntradayDip: SHORT signal - Up {pct_change:.2f}%, Z={z_score:.2f}")
            dynamic_sltp_engine.log_params(sltp, "IntradayDip_SHORT")
            return {"strategy": "IntradayDip", "side": "SHORT",
                    "tp_dist": sltp['tp_dist'], "sl_dist": sltp['sl_dist']}

        return None
