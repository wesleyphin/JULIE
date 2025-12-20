import logging
from typing import Dict, Optional

import pandas as pd

from dynamic_signal_engine2 import get_signal_engine as get_signal_engine2
from strategy_base import Strategy
# Import the dynamic parameter engine to access tight, data-driven SL/TPs
from dynamic_sltp_params import get_sltp

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

        # Check for pattern triggers from the hardcoded engine
        signal_data = self.engine.check_signal(current_time, df_5m, df_15m)

        if signal_data:
            # Map the signal direction to the corresponding RegimeAdaptive profile
            # 'RegimeAdaptive' offers robust, volatility-adjusted tight stops suitable for PA
            strat_key = "RegimeAdaptive_LONG" if signal_data['signal'] == "LONG" else "RegimeAdaptive_SHORT"

            # Calculate dynamic SL/TP using the original 1m dataframe for accurate real-time ATR
            dynamic_params = get_sltp(strat_key, df)

            # --- LOGIC UPDATE: ENFORCE MINIMUM SL & POSITIVE RR ---
            MIN_SL_FLOOR = 4.0      # Minimum SL in points
            TARGET_RR = 1.5         # Minimum Reward-to-Risk Ratio

            final_sl = dynamic_params['sl_dist']
            final_tp = dynamic_params['tp_dist']

            # 1. Enforce SL Floor
            if final_sl < MIN_SL_FLOOR:
                final_sl = MIN_SL_FLOOR

            # 2. Enforce Positive RR based on NEW SL
            required_tp = final_sl * TARGET_RR
            if final_tp < required_tp:
                final_tp = required_tp

            logging.info(f"🚀 DYNAMIC ENGINE 2 TRIGGER: {signal_data['strategy_id']}")
            logging.info(f"📉 Tightening Risk: Overriding Static {signal_data['sl']}/{signal_data['tp']} "
                         f"with Dynamic {final_sl}/{final_tp} "
                         f"(Source: {dynamic_params['hierarchy_key']} + Floor/RR)")

            return {
                "strategy": "DynamicEngine2",
                "sub_strategy": signal_data['strategy_id'],
                "side": signal_data['signal'],
                # Use the tighter, calculated distributions instead of the engine's hardcoded ones
                "tp_dist": final_tp,
                "sl_dist": final_sl
            }

        return None
