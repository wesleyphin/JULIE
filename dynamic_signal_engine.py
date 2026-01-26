"""
Dynamic Signal Engine - HARDCODED STRATEGIES
ALL 235 optimized strategies embedded directly in code.
No CSV file dependency - completely self-contained.

Author: Senior Python Quant Developer
Date: December 2025
"""

import logging
from typing import Optional, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd  # Ensure pandas is imported for rolling calculation


# ============================================================================
# HARDCODED STRATEGY DATABASE - ALL 235 STRATEGIES FROM CSV
# ============================================================================
STRATEGY_DATABASE = [
    {'TF': '15min', 'Session': '21-24', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.088, 'Year': 3, 'Qtr': 8},
    {'TF': '15min', 'Session': '21-24', 'Type': 'Long_Rev', 'Thresh': 5, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.116, 'Year': 3, 'Qtr': 6},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 12, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.286, 'Year': 3, 'Qtr': 6},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 7, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.209, 'Year': 3, 'Qtr': 6},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.103, 'Year': 3, 'Qtr': 5},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Short_Mom', 'Thresh': 5, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.323, 'Year': 3, 'Qtr': 5},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.179, 'Year': 3, 'Qtr': 5},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 15, 'Best_TP': 15, 'Opt_WR': 0.345, 'Year': 3, 'Qtr': 5},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 5, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.145, 'Year': 3, 'Qtr': 5},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 8, 'Best_TP': 20, 'Opt_WR': 0.19, 'Year': 3, 'Qtr': 5},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.111, 'Year': 3, 'Qtr': 5},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 6, 'Best_SL': 12, 'Best_TP': 20, 'Opt_WR': 0.189, 'Year': 3, 'Qtr': 5},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.242, 'Year': 3, 'Qtr': 5},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Short_Rev', 'Thresh': 6, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.331, 'Year': 3, 'Qtr': 5},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.176, 'Year': 3, 'Qtr': 5},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.242, 'Year': 3, 'Qtr': 5},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.079, 'Year': 3, 'Qtr': 5},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.207, 'Year': 3, 'Qtr': 5},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.339, 'Year': 3, 'Qtr': 4},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Short_Mom', 'Thresh': 6, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.369, 'Year': 3, 'Qtr': 4},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Rev', 'Thresh': 15, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.299, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '21-24', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.135, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 8, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.244, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 9, 'Best_SL': 8, 'Best_TP': 20, 'Opt_WR': 0.201, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Short_Mom', 'Thresh': 6, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.298, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 7, 'Best_SL': 8, 'Best_TP': 20, 'Opt_WR': 0.169, 'Year': 3, 'Qtr': 4},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Rev', 'Thresh': 10, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.277, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.111, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 3, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.131, 'Year': 3, 'Qtr': 4},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.146, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 9, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.198, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 10, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.203, 'Year': 3, 'Qtr': 4},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 6, 'Best_TP': 15, 'Opt_WR': 0.231, 'Year': 3, 'Qtr': 4},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 7, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.073, 'Year': 3, 'Qtr': 4},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Mom', 'Thresh': 8, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.083, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.131, 'Year': 3, 'Qtr': 4},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 9, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.284, 'Year': 3, 'Qtr': 4},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 10, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.249, 'Year': 3, 'Qtr': 4},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.254, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 12, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.375, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 6, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.258, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Short_Rev', 'Thresh': 9, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.268, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 7, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.296, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 10, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.355, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 8, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.323, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Long_Rev', 'Thresh': 3, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.044, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Short_Rev', 'Thresh': 12, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.378, 'Year': 3, 'Qtr': 3},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Short_Mom', 'Thresh': 7, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.32, 'Year': 3, 'Qtr': 3},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 15, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.177, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Short_Mom', 'Thresh': 4, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.256, 'Year': 3, 'Qtr': 3},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 5, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.08, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Rev', 'Thresh': 12, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.291, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 5, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.168, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Short_Mom', 'Thresh': 15, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.308, 'Year': 3, 'Qtr': 3},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 9, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.086, 'Year': 3, 'Qtr': 3},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 12, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.151, 'Year': 3, 'Qtr': 3},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 8, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.145, 'Year': 3, 'Qtr': 3},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 10, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.047, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.211, 'Year': 3, 'Qtr': 3},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Long_Mom', 'Thresh': 12, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.192, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Short_Mom', 'Thresh': 10, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.255, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 12, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.268, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 15, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.267, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 7, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.227, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 8, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.242, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.247, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 6, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.202, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.077, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 9, 'Best_SL': 3, 'Best_TP': 15, 'Opt_WR': 0.141, 'Year': 3, 'Qtr': 3},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Short_Rev', 'Thresh': 6, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.25, 'Year': 2, 'Qtr': 4},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 10, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.207, 'Year': 2, 'Qtr': 4},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Short_Mom', 'Thresh': 8, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.335, 'Year': 2, 'Qtr': 4},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Rev', 'Thresh': 7, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.204, 'Year': 2, 'Qtr': 4},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 10, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.248, 'Year': 2, 'Qtr': 4},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.186, 'Year': 2, 'Qtr': 4},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.202, 'Year': 2, 'Qtr': 4},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 2, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.07, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 4, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.121, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 10, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.114, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Short_Mom', 'Thresh': 3, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.184, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 6, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.201, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '21-24', 'Type': 'Short_Mom', 'Thresh': 4, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.167, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Rev', 'Thresh': 9, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.276, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Long_Mom', 'Thresh': 7, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.131, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 5, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.134, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '21-24', 'Type': 'Long_Rev', 'Thresh': 7, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.153, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 10, 'Best_SL': 12, 'Best_TP': 20, 'Opt_WR': 0.299, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Short_Rev', 'Thresh': 6, 'Best_SL': 15, 'Best_TP': 15, 'Opt_WR': 0.377, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 3, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.083, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.188, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.25, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 5, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.108, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Rev', 'Thresh': 5, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.156, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Short_Mom', 'Thresh': 12, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.31, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.081, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.181, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 7, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.23, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 15, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.313, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 7, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.168, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Short_Rev', 'Thresh': 7, 'Best_SL': 12, 'Best_TP': 12, 'Opt_WR': 0.458, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 6, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.183, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.1, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.088, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Short_Mom', 'Thresh': 6, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.241, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Long_Mom', 'Thresh': 6, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.205, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 8, 'Best_SL': 15, 'Best_TP': 20, 'Opt_WR': 0.237, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.073, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 7, 'Best_SL': 15, 'Best_TP': 15, 'Opt_WR': 0.328, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 7, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.151, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Mom', 'Thresh': 8, 'Best_SL': 3, 'Best_TP': 20, 'Opt_WR': 0.122, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Short_Rev', 'Thresh': 8, 'Best_SL': 15, 'Best_TP': 15, 'Opt_WR': 0.437, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.161, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Short_Mom', 'Thresh': 12, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.113, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Short_Mom', 'Thresh': 4, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.298, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 6, 'Best_TP': 25, 'Opt_WR': 0.169, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 3, 'Best_SL': 6, 'Best_TP': 25, 'Opt_WR': 0.109, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.14, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 12, 'Best_SL': 12, 'Best_TP': 12, 'Opt_WR': 0.472, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Short_Rev', 'Thresh': 8, 'Best_SL': 6, 'Best_TP': 20, 'Opt_WR': 0.214, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Short_Rev', 'Thresh': 5, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.226, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 10, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.108, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 12, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.285, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 12, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.304, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Mom', 'Thresh': 8, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.154, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 5, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.101, 'Year': 2, 'Qtr': 3},
    {'TF': '15min', 'Session': '18-21', 'Type': 'Short_Rev', 'Thresh': 2, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.063, 'Year': 2, 'Qtr': 3},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 9, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.182, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 12, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.199, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.167, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Short_Mom', 'Thresh': 9, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.445, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 3, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.15, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 15, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.093, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 4, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.198, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Short_Rev', 'Thresh': 8, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.258, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Mom', 'Thresh': 15, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.377, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 9, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.362, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 10, 'Best_SL': 6, 'Best_TP': 25, 'Opt_WR': 0.389, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 12, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.382, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 15, 'Best_SL': 6, 'Best_TP': 25, 'Opt_WR': 0.465, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '18-21', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.084, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 9, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.234, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '21-24', 'Type': 'Short_Mom', 'Thresh': 5, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.192, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 9, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.275, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '18-21', 'Type': 'Short_Mom', 'Thresh': 3, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.123, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Short_Rev', 'Thresh': 10, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.35, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 6, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.26, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 5, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.167, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 3, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.083, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 3, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.106, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 6, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.135, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 10, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.287, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 9, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.265, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Short_Rev', 'Thresh': 5, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.206, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 5, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.137, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Long_Mom', 'Thresh': 6, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.096, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.121, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '18-21', 'Type': 'Short_Mom', 'Thresh': 4, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.151, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.166, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 4, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.109, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Mom', 'Thresh': 4, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.096, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Long_Mom', 'Thresh': 8, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.15, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Short_Rev', 'Thresh': 8, 'Best_SL': 4, 'Best_TP': 20, 'Opt_WR': 0.167, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.269, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.16, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Rev', 'Thresh': 7, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.138, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 7, 'Best_SL': 6, 'Best_TP': 25, 'Opt_WR': 0.268, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Short_Rev', 'Thresh': 5, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.182, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.229, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 2, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.065, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Short_Rev', 'Thresh': 6, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.241, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 8, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.339, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 12, 'Best_TP': 20, 'Opt_WR': 0.283, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 7, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.211, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.249, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 8, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.225, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Mom', 'Thresh': 7, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.191, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Rev', 'Thresh': 12, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.333, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 10, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.268, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 12, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.317, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 15, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.323, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '18-21', 'Type': 'Short_Mom', 'Thresh': 7, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.23, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Mom', 'Thresh': 8, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.293, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Short_Rev', 'Thresh': 9, 'Best_SL': 4, 'Best_TP': 20, 'Opt_WR': 0.177, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Short_Rev', 'Thresh': 7, 'Best_SL': 4, 'Best_TP': 20, 'Opt_WR': 0.161, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 12, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.218, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 7, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.252, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Short_Mom', 'Thresh': 12, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.319, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Mom', 'Thresh': 10, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.294, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.098, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Short_Rev', 'Thresh': 3, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.12, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Mom', 'Thresh': 9, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.248, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Long_Mom', 'Thresh': 10, 'Best_SL': 8, 'Best_TP': 25, 'Opt_WR': 0.203, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Rev', 'Thresh': 7, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.102, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Short_Rev', 'Thresh': 5, 'Best_SL': 6, 'Best_TP': 25, 'Opt_WR': 0.204, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.073, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Short_Rev', 'Thresh': 10, 'Best_SL': 4, 'Best_TP': 20, 'Opt_WR': 0.178, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Short_Mom', 'Thresh': 15, 'Best_SL': 15, 'Best_TP': 15, 'Opt_WR': 0.506, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '21-24', 'Type': 'Long_Rev', 'Thresh': 8, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.182, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.07, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Short_Mom', 'Thresh': 6, 'Best_SL': 15, 'Best_TP': 20, 'Opt_WR': 0.27, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '21-24', 'Type': 'Short_Rev', 'Thresh': 3, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.069, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '12-15', 'Type': 'Short_Mom', 'Thresh': 9, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.225, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 4, 'Best_SL': 6, 'Best_TP': 25, 'Opt_WR': 0.145, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.201, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '15-18', 'Type': 'Long_Mom', 'Thresh': 10, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.09, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Mom', 'Thresh': 4, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.066, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Long_Mom', 'Thresh': 7, 'Best_SL': 15, 'Best_TP': 20, 'Opt_WR': 0.309, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Short_Mom', 'Thresh': 7, 'Best_SL': 15, 'Best_TP': 20, 'Opt_WR': 0.292, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Short_Mom', 'Thresh': 7, 'Best_SL': 3, 'Best_TP': 20, 'Opt_WR': 0.114, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Mom', 'Thresh': 6, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.205, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Mom', 'Thresh': 6, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.141, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 5, 'Best_SL': 5, 'Best_TP': 25, 'Opt_WR': 0.115, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '21-24', 'Type': 'Long_Rev', 'Thresh': 3, 'Best_SL': 6, 'Best_TP': 25, 'Opt_WR': 0.12, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 10, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.258, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 9, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.254, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 10, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.272, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Short_Mom', 'Thresh': 5, 'Best_SL': 10, 'Best_TP': 25, 'Opt_WR': 0.289, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '09-12', 'Type': 'Short_Mom', 'Thresh': 8, 'Best_SL': 15, 'Best_TP': 20, 'Opt_WR': 0.313, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '00-03', 'Type': 'Long_Mom', 'Thresh': 4, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.133, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Long_Rev', 'Thresh': 3, 'Best_SL': 12, 'Best_TP': 20, 'Opt_WR': 0.269, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '06-09', 'Type': 'Long_Rev', 'Thresh': 4, 'Best_SL': 2, 'Best_TP': 12, 'Opt_WR': 0.124, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 6, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.146, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 12, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.294, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Mom', 'Thresh': 10, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.149, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '06-09', 'Type': 'Long_Rev', 'Thresh': 5, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.076, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 15, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.182, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Long_Rev', 'Thresh': 15, 'Best_SL': 4, 'Best_TP': 25, 'Opt_WR': 0.289, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '09-12', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 15, 'Best_TP': 25, 'Opt_WR': 0.212, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '12-15', 'Type': 'Short_Mom', 'Thresh': 12, 'Best_SL': 15, 'Best_TP': 20, 'Opt_WR': 0.391, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '15-18', 'Type': 'Long_Rev', 'Thresh': 12, 'Best_SL': 5, 'Best_TP': 20, 'Opt_WR': 0.153, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '03-06', 'Type': 'Long_Mom', 'Thresh': 5, 'Best_SL': 12, 'Best_TP': 25, 'Opt_WR': 0.12, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '18-21', 'Type': 'Short_Rev', 'Thresh': 4, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.104, 'Year': 2, 'Qtr': 2},
    {'TF': '5min', 'Session': '03-06', 'Type': 'Short_Rev', 'Thresh': 4, 'Best_SL': 7, 'Best_TP': 25, 'Opt_WR': 0.135, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '18-21', 'Type': 'Short_Rev', 'Thresh': 3, 'Best_SL': 3, 'Best_TP': 25, 'Opt_WR': 0.081, 'Year': 2, 'Qtr': 2},
    {'TF': '15min', 'Session': '18-21', 'Type': 'Long_Mom', 'Thresh': 4, 'Best_SL': 2, 'Best_TP': 25, 'Opt_WR': 0.045, 'Year': 2, 'Qtr': 2},
]


class DynamicSignalEngine:
    """
    Signal engine with all 235 strategies hardcoded.
    Checks for signals based on:
    - Current session (3-hour blocks in US/Eastern time)
    - Previous candle body vs threshold
    - Strategy type (Long_Rev, Short_Rev, Long_Mom, Short_Mom)
    """

    def __init__(self):
        """Initialize the signal engine."""
        self.et_tz = ZoneInfo('America/New_York')
        self.strategies = STRATEGY_DATABASE

    def _log_initialization(self):
        """Log strategy loading statistics."""
        logging.info("=" * 70)
        logging.info("🚀 DYNAMIC SIGNAL ENGINE - ALL STRATEGIES HARDCODED")
        logging.info("=" * 70)
        logging.info(f"✅ Loaded {len(self.strategies)} strategies")

        # Count by timeframe
        tf_5min = len([s for s in self.strategies if s['TF'] == '5min'])
        tf_15min = len([s for s in self.strategies if s['TF'] == '15min'])
        logging.info(f"   5min strategies: {tf_5min}")
        logging.info(f"   15min strategies: {tf_15min}")

        # Count by session
        sessions = {}
        for s in self.strategies:
            sessions[s['Session']] = sessions.get(s['Session'], 0) + 1
        logging.info(f"   Sessions covered: {sorted(sessions.keys())}")
        for session, count in sorted(sessions.items()):
            logging.info(f"      {session}: {count} strategies")

        # Count by type
        types = {}
        for s in self.strategies:
            types[s['Type']] = types.get(s['Type'], 0) + 1
        logging.info(f"   Strategy types:")
        for stype, count in sorted(types.items()):
            logging.info(f"      {stype}: {count} strategies")

        # Win rate statistics
        win_rates = [s['Opt_WR'] for s in self.strategies]
        logging.info(f"📈 Win Rate Statistics:")
        logging.info(f"   Min WR: {min(win_rates):.1%}")
        logging.info(f"   Max WR: {max(win_rates):.1%}")
        logging.info(f"   Mean WR: {sum(win_rates)/len(win_rates):.1%}")
        logging.info("=" * 70)

    def get_session_from_time(self, dt_et: datetime) -> str:
        """
        Convert US/Eastern datetime to 3-hour session string.
        Sessions: 00-03, 03-06, 06-09, 09-12, 12-15, 15-18, 18-21, 21-24
        """
        hour = dt_et.hour
        session_start = (hour // 3) * 3
        session_end = session_start + 3
        return f"{session_start:02d}-{session_end:02d}"

    def _calculate_body(self, open_price: float, close_price: float) -> float:
        """Calculate candle body size (close - open)."""
        return close_price - open_price

    def _is_green_candle(self, open_price: float, close_price: float) -> bool:
        """Check if candle is green (bullish)."""
        return close_price > open_price

    def _is_red_candle(self, open_price: float, close_price: float) -> bool:
        """Check if candle is red (bearish)."""
        return close_price < open_price

    def check_signal(self, current_time: datetime, df_5m, df_15m) -> Optional[Dict]:
        """
        Check for trading signals based on hardcoded strategies.

        NEW TIE-BREAKER HIERARCHY:
        1. Direction Count (Most signals wins)
        2. Price Location (Longs prefer Lows, Shorts prefer Highs)
        3. Win Rate (Best historic performance wins)
        """
        # Convert current time to ET if needed
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=self.et_tz)
        else:
            current_time = current_time.astimezone(self.et_tz)

        # Get current session
        session = self.get_session_from_time(current_time)
        logging.debug(f"🔍 Signal Check: Session={session}")

        # Filter strategies for current session
        matching_strategies = [s for s in self.strategies if s['Session'] == session]
        if not matching_strategies:
            return None

        # --- 1. CALCULATE PRICE LOCATION (0.0 to 1.0) ---
        # 0.0 = At Recent Low (Favors Longs)
        # 1.0 = At Recent High (Favors Shorts)
        price_location = 0.5  # Default neutral
        try:
            if df_5m is not None and len(df_5m) > 20:
                last_20 = df_5m.iloc[-20:]
                recent_high = last_20['high'].max()
                recent_low = last_20['low'].min()
                current_close = df_5m.iloc[-1]['close']

                if recent_high > recent_low:
                    price_location = (current_close - recent_low) / (recent_high - recent_low)
                    price_location = max(0.0, min(1.0, price_location)) # Clamp
        except Exception as e:
            logging.error(f"Error calculating price location: {e}")

        # --- 2. COLLECT TRIGGERS ---
        triggered_signals = []

        for timeframe_str, df in [('5min', df_5m), ('15min', df_15m)]:
            if df is None or len(df) < 2:
                continue

            prev_candle = df.iloc[-2]
            col_map = {col.lower(): col for col in df.columns}
            open_col = col_map.get('open')
            close_col = col_map.get('close')

            if not open_col or not close_col:
                continue

            prev_open = float(prev_candle[open_col])
            prev_close = float(prev_candle[close_col])
            body = self._calculate_body(prev_open, prev_close)
            abs_body = abs(body)
            is_green = self._is_green_candle(prev_open, prev_close)
            is_red = self._is_red_candle(prev_open, prev_close)

            tf_strategies = [s for s in matching_strategies if s['TF'] == timeframe_str]

            for strategy in tf_strategies:
                strategy_type = strategy['Type']
                thresh = float(strategy['Thresh'])
                signal = None

                if strategy_type == 'Long_Rev' and is_red and abs_body > thresh:
                    signal = 'LONG'
                elif strategy_type == 'Short_Rev' and is_green and abs_body > thresh:
                    signal = 'SHORT'
                elif strategy_type == 'Long_Mom' and is_green and abs_body > thresh:
                    signal = 'LONG'
                elif strategy_type == 'Short_Mom' and is_red and abs_body > thresh:
                    signal = 'SHORT'

                if signal:
                    sl_value = max(4.0, float(strategy['Best_SL']))
                    triggered_signals.append({
                        'signal': signal,
                        'sl': sl_value,
                        'tp': float(strategy['Best_TP']),
                        'opt_wr': float(strategy['Opt_WR']),
                        'timeframe': timeframe_str,
                        'strategy_type': strategy_type,
                        'thresh': thresh,
                        'body': abs_body,
                        'strategy_id': f"{timeframe_str}_{session}_{strategy_type}_T{int(thresh)}_Y{int(strategy['Year'])}Q{int(strategy['Qtr'])}"
                    })
                    logging.info(f"✅ TRIGGER: {strategy_type} on {timeframe_str} | Body={abs_body:.2f} > Thresh={thresh:.2f}")

        if not triggered_signals:
            return None

        # --- 3. APPLY HIERARCHICAL SCORING ---
        long_count = sum(1 for s in triggered_signals if s['signal'] == 'LONG')
        short_count = sum(1 for s in triggered_signals if s['signal'] == 'SHORT')

        # Bucketed, quality-weighted count (reduces correlation bias)
        bucket_scores = {"LONG": {}, "SHORT": {}}
        for sig in triggered_signals:
            side = sig['signal']
            bucket_key = (sig['timeframe'], sig['strategy_type'])
            current_best = bucket_scores[side].get(bucket_key, 0.0)
            if sig['opt_wr'] > current_best:
                bucket_scores[side][bucket_key] = sig['opt_wr']

        bucket_score_raw = {
            side: sum(bucket_scores[side].values()) for side in bucket_scores
        }
        bucket_score_cap = 1.0
        bucket_score_norm = {
            side: min(bucket_score_raw[side], bucket_score_cap) / bucket_score_cap
            for side in bucket_scores
        }

        for sig in triggered_signals:
            # Priority 1: Bucketed Count Score (Magnitude 10.0, capped/normalized)
            # One bucket per (TF, Type), weighted by best Opt_WR in that bucket.
            count_score = bucket_score_norm['LONG'] if sig['signal'] == 'LONG' else bucket_score_norm['SHORT']

            # Priority 2: Price Location (Magnitude 2.0)
            # Longs want Low price (1 - loc). Shorts want High price (loc).
            # Max score 2.0, enough to break ties in count, but not overcome count difference.
            if sig['signal'] == 'LONG':
                loc_score = 1.0 - price_location
            else:
                loc_score = price_location

            # Priority 3: Win Rate (Magnitude 1.0)
            # Standard decimal WR (e.g., 0.35) adds final tie-break.
            wr_score = sig['opt_wr']

            # Final Formula
            sig['final_score'] = (count_score * 10.0) + (loc_score * 2.0) + wr_score
            sig['debug_info'] = f"BucketScore:{count_score:.2f} Loc:{loc_score:.2f} WR:{wr_score:.3f}"

        # Sort by Final Score Descending
        triggered_signals.sort(key=lambda x: x['final_score'], reverse=True)

        best_signal = triggered_signals[0]

        logging.info(
            f"🎯 TIE-BREAK: {len(triggered_signals)} signals. "
            f"Counts: LONG={long_count}, SHORT={short_count} | "
            f"Buckets: LONG={len(bucket_scores['LONG'])}, SHORT={len(bucket_scores['SHORT'])}"
        )
        logging.info(
            f"   BucketScore raw: LONG={bucket_score_raw['LONG']:.3f} "
            f"SHORT={bucket_score_raw['SHORT']:.3f} (cap {bucket_score_cap:.2f})"
        )
        logging.info(f"   Price Location: {price_location:.2f} (0=Low, 1=High)")
        logging.info(f"   Winner: {best_signal['strategy_id']} (Score: {best_signal['final_score']:.3f} | {best_signal['debug_info']})")
        if len(triggered_signals) > 1:
            logging.info(f"   Runner-up: {triggered_signals[1]['strategy_id']} (Score: {triggered_signals[1]['final_score']:.3f})")

        logging.info("=" * 70)
        logging.info(f"🚀 FINAL SIGNAL SELECTED")
        logging.info(f"   Direction: {best_signal['signal']}")
        logging.info(f"   Timeframe: {best_signal['timeframe']}")
        logging.info(f"   Strategy: {best_signal['strategy_type']}")
        logging.info(f"   Stop Loss: {best_signal['sl']:.1f} points")
        logging.info(f"   Take Profit: {best_signal['tp']:.1f} points")
        logging.info(f"   ID: {best_signal['strategy_id']}")
        logging.info("=" * 70)

        return best_signal


# Singleton instance for global access
_engine_instance = None

def get_signal_engine() -> DynamicSignalEngine:
    """Get or create singleton instance of DynamicSignalEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DynamicSignalEngine()
    return _engine_instance


if __name__ == "__main__":
    # Test the engine
    logging.basicConfig(level=logging.INFO)
    engine = get_signal_engine()
    print(f"\n✅ Engine ready with {len(engine.strategies)} strategies!")
