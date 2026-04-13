import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

class DynamicSignalEngine2:
    """
    Contains 167 hardcoded Price Action edges optimized for Positive Risk:Reward.
    No external CSV required. Matches strategy to current Quarter/Day/Session.
    """
    def __init__(self):
        # ==============================================================================
        # HARDCODED EDGES DATABASE (Optimized on 2023-2025 Data)
        # Criteria: Win Rate > 55% AND Reward >= Risk
        # Format: (Timeframe, Quarter, Day, Session): {'Strategy': ..., 'TP': ..., 'SL': ...}
        # ==============================================================================
        self.edges_db = {
            # --- 5 Minute Timeframe Edges ---
            ('5min', 'Q1', 'Mon', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Mon', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Mon', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Mon', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Tue', 'Asia'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Tue', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Tue', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Tue', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Wed', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Wed', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Wed', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Wed', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q1', 'Thu', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q1', 'Thu', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Thu', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Thu', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Fri', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Fri', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('5min', 'Q1', 'Fri', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q1', 'Sun', 'NY_AM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Mon', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Mon', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Mon', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Mon', 'NY_PM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q2', 'Tue', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Tue', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Tue', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q2', 'Tue', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Wed', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Wed', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Wed', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Wed', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Thu', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q2', 'Thu', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Thu', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Thu', 'NY_PM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Fri', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q2', 'Fri', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 25, 'RR': 1.0},
            ('5min', 'Q2', 'Fri', 'NY_AM'): {'Strategy': 'Inside_Break', 'TP': 30, 'SL': 25, 'RR': 1.2},
            ('5min', 'Q2', 'Fri', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q2', 'Sun', 'NY_AM'): {'Strategy': 'Inside_Break', 'TP': 15, 'SL': 15, 'RR': 1.0},
            ('5min', 'Q3', 'Mon', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Mon', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Mon', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q3', 'Mon', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 10, 'RR': 2.5},
            ('5min', 'Q3', 'Tue', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q3', 'Tue', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Tue', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Tue', 'NY_PM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('5min', 'Q3', 'Wed', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q3', 'Wed', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Wed', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Wed', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Thu', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Thu', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Thu', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Thu', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Fri', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Fri', 'London'): {'Strategy': 'Wick_Rejection', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Fri', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Sun', 'London'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q3', 'Sun', 'NY_AM'): {'Strategy': 'Wick_Rejection', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Mon', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Mon', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Mon', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q4', 'Mon', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Tue', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Tue', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Tue', 'NY_AM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q4', 'Tue', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Wed', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Wed', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Wed', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('5min', 'Q4', 'Wed', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Thu', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Thu', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Thu', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Thu', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Fri', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Fri', 'London'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Fri', 'NY_AM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 25, 'RR': 1.2},
            ('5min', 'Q4', 'Fri', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('5min', 'Q4', 'Sun', 'NY_AM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},

            # --- 15 Minute Timeframe Edges ---
            ('15min', 'Q1', 'Mon', 'Asia'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q1', 'Mon', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q1', 'Mon', 'NY_AM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q1', 'Mon', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q1', 'Tue', 'Asia'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q1', 'Tue', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q1', 'Tue', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 15, 'RR': 1.67},
            ('15min', 'Q1', 'Tue', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q1', 'Wed', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q1', 'Wed', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q1', 'Wed', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q1', 'Wed', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q1', 'Thu', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 30, 'RR': 1.0},
            ('15min', 'Q1', 'Thu', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q1', 'Thu', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q1', 'Thu', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q1', 'Fri', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q1', 'Fri', 'London'): {'Strategy': 'Wick_Rejection', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q1', 'Fri', 'NY_AM'): {'Strategy': 'Follow_Color', 'TP': 15, 'SL': 15, 'RR': 1.0},
            ('15min', 'Q1', 'Fri', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q1', 'Sun', 'London'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q2', 'Mon', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q2', 'Mon', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q2', 'Mon', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 10, 'RR': 2.5},
            ('15min', 'Q2', 'Mon', 'NY_PM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q2', 'Tue', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q2', 'Tue', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q2', 'Tue', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q2', 'Tue', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q2', 'Wed', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 25, 'RR': 1.2},
            ('15min', 'Q2', 'Wed', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q2', 'Wed', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q2', 'Wed', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q2', 'Thu', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 30, 'RR': 1.0},
            ('15min', 'Q2', 'Thu', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 30, 'RR': 1.0},
            ('15min', 'Q2', 'Thu', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q2', 'Thu', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q2', 'Fri', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q2', 'Fri', 'London'): {'Strategy': 'Engulfing', 'TP': 25, 'SL': 25, 'RR': 1.0},
            ('15min', 'Q2', 'Fri', 'NY_AM'): {'Strategy': 'Wick_Rejection', 'TP': 30, 'SL': 30, 'RR': 1.0},
            ('15min', 'Q2', 'Fri', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q2', 'Sun', 'NY_AM'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Mon', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q3', 'Mon', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Mon', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 15, 'SL': 10, 'RR': 1.5},
            ('15min', 'Q3', 'Mon', 'NY_PM'): {'Strategy': 'Wick_Rejection', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q3', 'Tue', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q3', 'Tue', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Tue', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q3', 'Tue', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Wed', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 20, 'RR': 1.25},
            ('15min', 'Q3', 'Wed', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Wed', 'NY_AM'): {'Strategy': 'Engulfing', 'TP': 25, 'SL': 10, 'RR': 2.5},
            ('15min', 'Q3', 'Wed', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q3', 'Thu', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Thu', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Thu', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q3', 'Thu', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 10, 'RR': 2.5},
            ('15min', 'Q3', 'Fri', 'Asia'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Fri', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 25, 'RR': 1.0},
            ('15min', 'Q3', 'Fri', 'NY_AM'): {'Strategy': 'Inside_Break', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q3', 'Fri', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Sun', 'London'): {'Strategy': 'Wick_Rejection', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q3', 'Sun', 'NY_AM'): {'Strategy': 'Wick_Rejection', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q4', 'Mon', 'Asia'): {'Strategy': 'Engulfing', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q4', 'Mon', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q4', 'Mon', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q4', 'Mon', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 15, 'RR': 1.67},
            ('15min', 'Q4', 'Tue', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q4', 'Tue', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q4', 'Tue', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 15, 'SL': 10, 'RR': 1.5},
            ('15min', 'Q4', 'Tue', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 10, 'RR': 2.5},
            ('15min', 'Q4', 'Wed', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q4', 'Wed', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q4', 'Wed', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q4', 'Wed', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q4', 'Thu', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 10, 'RR': 3.0},
            ('15min', 'Q4', 'Thu', 'London'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q4', 'Thu', 'NY_AM'): {'Strategy': 'Gap_Reversal', 'TP': 25, 'SL': 15, 'RR': 1.67},
            ('15min', 'Q4', 'Thu', 'NY_PM'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 20, 'RR': 1.5},
            ('15min', 'Q4', 'Fri', 'Asia'): {'Strategy': 'Gap_Reversal', 'TP': 30, 'SL': 15, 'RR': 2.0},
            ('15min', 'Q4', 'Fri', 'London'): {'Strategy': 'Engulfing', 'TP': 15, 'SL': 15, 'RR': 1.0},
            ('15min', 'Q4', 'Fri', 'NY_AM'): {'Strategy': 'Wick_Rejection', 'TP': 20, 'SL': 20, 'RR': 1.0},
            ('15min', 'Q4', 'Fri', 'NY_PM'): {'Strategy': 'Wick_Rejection', 'TP': 30, 'SL': 30, 'RR': 1.0},
        }

        # Strategy Map
        self.strategies = {
            'Gap_Reversal': self._check_gap_reversal,
            'Follow_Color': self._check_follow_color,
            'Fade_Color': self._check_fade_color,
            'Engulfing': self._check_engulfing,
            'Wick_Rejection': self._check_wick_rejection,
            'Inside_Break': self._check_inside_break
        }

    def _get_segment_info(self, timestamp):
        """Determines the current Q/Day/Session segment."""
        # 1. Quarter
        quarter = f"Q{(timestamp.month - 1) // 3 + 1}"
        
        # 2. Day
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day = day_names[timestamp.dayofweek]
        
        # 3. Session
        h = timestamp.hour + timestamp.minute / 60.0
        if (h >= 18.0) or (h < 3.0):
            session = 'Asia'
        elif (h >= 3.0) and (h < 9.5):
            session = 'London'
        elif (h >= 9.5) and (h < 12.0):
            session = 'NY_AM'
        elif (h >= 12.0) and (h < 17.0):
            session = 'NY_PM'
        else:
            session = 'Other'
            
        return quarter, day, session

    # ==========================================
    # PATTERN RECOGNITION LOGIC
    # ==========================================
    def _check_gap_reversal(self, df):
        # Open < Prev Close -> Buy (1)
        # Open > Prev Close -> Sell (-1)
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        if curr['open'] < prev['close']: return 1
        if curr['open'] > prev['close']: return -1
        return 0

    def _check_follow_color(self, df):
        # Prev Green -> Buy (1)
        # Prev Red -> Sell (-1)
        prev = df.iloc[-2]
        if prev['close'] > prev['open']: return 1
        if prev['close'] < prev['open']: return -1
        return 0

    def _check_fade_color(self, df):
        # Prev Green -> Sell (-1)
        # Prev Red -> Buy (1)
        prev = df.iloc[-2]
        if prev['close'] > prev['open']: return -1
        if prev['close'] < prev['open']: return 1
        return 0

    def _check_engulfing(self, df):
        if len(df) < 3: return 0
        prev = df.iloc[-2]
        pprev = df.iloc[-3]
        
        # Bullish Engulfing
        pprev_red = pprev['close'] < pprev['open']
        prev_green = prev['close'] > prev['open']
        if pprev_red and prev_green:
            if prev['close'] > pprev['open'] and prev['open'] < pprev['close']:
                return 1

        # Bearish Engulfing
        pprev_green = pprev['close'] > pprev['open']
        prev_red = prev['close'] < prev['open']
        if pprev_green and prev_red:
            if prev['close'] < pprev['open'] and prev['open'] > pprev['close']:
                return -1
        return 0

    def _check_wick_rejection(self, df):
        # Hammer (Buy) / Shooting Star (Sell)
        prev = df.iloc[-2]
        body = abs(prev['close'] - prev['open'])
        upper_wick = prev['high'] - max(prev['close'], prev['open'])
        lower_wick = min(prev['close'], prev['open']) - prev['low']
        
        if lower_wick > (2 * body) and upper_wick < body: return 1
        if upper_wick > (2 * body) and lower_wick < body: return -1
        return 0

    def _check_inside_break(self, df):
        if len(df) < 3: return 0
        prev = df.iloc[-2]
        pprev = df.iloc[-3]
        is_inside = (prev['high'] < pprev['high']) and (prev['low'] > pprev['low'])
        if is_inside:
            if pprev['close'] > pprev['open']: return 1
            else: return -1
        return 0

    # ==========================================
    # MAIN CHECK FUNCTION
    # ==========================================
    def check_signals(self, current_time, df_5m, df_15m, *, emit_logs: bool = False) -> List[Dict]:
        """
        Return all currently valid DE2 candidates for the active Q/Day/Session segment.
        """
        q, d, s = self._get_segment_info(current_time)
        candidates: List[Dict] = []

        for timeframe_str, tf_tag, df_tf in (
            ("15min", "15m", df_15m),
            ("5min", "5m", df_5m),
        ):
            edge = self.edges_db.get((timeframe_str, q, d, s))
            if not edge or df_tf is None or len(df_tf) < 5:
                continue

            check_func = self.strategies.get(edge["Strategy"])
            if check_func is None:
                continue

            direction = check_func(df_tf)
            if direction == 0:
                continue

            signal_side = "LONG" if direction == 1 else "SHORT"
            strategy_name = str(edge["Strategy"])
            candidate = {
                "strategy_id": f"Dynamic_{strategy_name}_{tf_tag}_{q}_{d}_{s}",
                "signal": signal_side,
                "tp": float(edge["TP"]),
                "sl": float(edge["SL"]),
                "rr": float(edge["RR"]),
                "timeframe": timeframe_str,
                "strategy_type": strategy_name,
                "quarter": q,
                "day": d,
                "session": s,
            }
            candidates.append(candidate)
            if emit_logs:
                logging.info(
                    "DE2 candidate: %s %s tf=%s rr=%.2f",
                    strategy_name,
                    signal_side,
                    timeframe_str,
                    float(edge["RR"]),
                )

        return candidates

    def check_signal(self, current_time, df_5m, df_15m) -> Optional[Dict]:
        """
        Backward-compatible single-candidate API.
        """
        candidates = self.check_signals(current_time, df_5m, df_15m, emit_logs=False)
        if not candidates:
            return None
        candidates.sort(
            key=lambda c: (float(c.get("rr", 0.0) or 0.0), float(c.get("tp", 0.0) or 0.0)),
            reverse=True,
        )
        return candidates[0]

def get_signal_engine():
    return DynamicSignalEngine2()
