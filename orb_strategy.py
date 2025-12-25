import datetime
import logging
from typing import Dict, Optional

import pandas as pd

from dynamic_sltp_params import dynamic_sltp_engine
from strategy_base import Strategy


class OrbStrategy(Strategy):
    """ORB Strategy - Now uses DynamicSLTPEngine for SL/TP"""
    def __init__(self):
        self.strategy_name = "ORB"
        self.reset_daily()
        logging.info("ORBStrategy initialized | 9:30-9:45 range | Breakout after midpoint retest")

    def reset_daily(self):
        self.orb_high = None
        self.orb_low = None
        self.orb_mid = None
        self.orb_range = None
        self.mid_retested = False
        self.orb_complete = False
        self.daily_open = None
        self.current_date = None
        self.trade_taken_today = False

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if len(df) < 2:
            return None

        ts = df.index[-1]
        curr = df.iloc[-1]
        curr_date = ts.date()

        if self.current_date != curr_date:
            self.reset_daily()
            self.current_date = curr_date

        t = ts.time()

        if self.daily_open is None and t >= datetime.time(9, 30):
            self.daily_open = curr['open']

        if datetime.time(9, 30) <= t < datetime.time(9, 45):
            if self.orb_high is None:
                self.orb_high = curr['high']
                self.orb_low = curr['low']
            else:
                self.orb_high = max(self.orb_high, curr['high'])
                self.orb_low = min(self.orb_low, curr['low'])

        if t >= datetime.time(9, 45) and not self.orb_complete and self.orb_high is not None:
            self.orb_range = self.orb_high - self.orb_low
            self.orb_mid = (self.orb_high + self.orb_low) / 2.0
            self.orb_complete = True
            logging.info(f"ORB: Range complete | H: {self.orb_high:.2f} L: {self.orb_low:.2f} Mid: {self.orb_mid:.2f} Range: {self.orb_range:.2f}")

            if self.orb_range >= 15.0:
                self.orb_complete = False
                logging.info(f"ORB: Range {self.orb_range:.2f} >= 15.0pt - Strategy disabled for day")

        if not self.orb_complete:
            return None
        if not (datetime.time(9, 45) <= t <= datetime.time(11, 30)):
            return None
        if self.trade_taken_today:
            return None
        if self.daily_open is None:
            return None

        if not self.mid_retested:
            if curr['low'] <= self.orb_mid <= curr['high']:
                self.mid_retested = True
                logging.info(f"ORB: Midpoint {self.orb_mid:.2f} retested")

        if self.mid_retested and len(df) >= 2:
            prev_close = df.iloc[-2]['close']
            curr_close = curr['close']

            if (prev_close <= self.orb_high) and (curr_close > self.orb_high):
                if curr_close > self.daily_open:
                    self.trade_taken_today = True

                    sltp = dynamic_sltp_engine.calculate_dynamic_sltp(df)
                    logging.info(f"ORB: LONG signal generated - breakout above ORB high")
                    logging.info(f"   ORB H/L/Mid: {self.orb_high:.2f}/{self.orb_low:.2f}/{self.orb_mid:.2f} | Range: {self.orb_range:.2f}")
                    logging.info(f"   Break: {prev_close:.2f} -> {curr_close:.2f} | Daily open: {self.daily_open:.2f}")
                    dynamic_sltp_engine.log_params(sltp)

                    return {
                        "strategy": "ORB", "side": "LONG",
                        "tp_dist": sltp['tp_dist'], "sl_dist": sltp['sl_dist']
                    }

        return None
