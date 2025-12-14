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
import pytz


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
        self.et_tz = pytz.timezone('US/Eastern')
        self.strategies = STRATEGY_DATABASE
        self._log_initialization()

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

        Args:
            current_time: Current datetime in US/Eastern timezone
            df_5m: 5-minute OHLC dataframe (must have: open, high, low, close)
            df_15m: 15-minute OHLC dataframe (must have: open, high, low, close)

        Returns:
            Dict with signal details or None:
            {
                'signal': 'LONG' or 'SHORT',
                'sl': stop loss distance (points),
                'tp': take profit distance (points),
                'strategy_id': string description,
                'timeframe': '5min' or '15min',
                'opt_wr': optimized win rate,
                'thresh': threshold that triggered,
                'body': actual candle body size
            }
        """
        # Convert current time to ET if needed
        if current_time.tzinfo is None:
            current_time = self.et_tz.localize(current_time)
        else:
            current_time = current_time.astimezone(self.et_tz)

        # Get current session
        session = self.get_session_from_time(current_time)

        logging.debug(f"🔍 Signal Check: Session={session}")

        # Filter strategies for current session
        matching_strategies = [s for s in self.strategies if s['Session'] == session]

        if not matching_strategies:
            logging.debug(f"⚪ No strategies for session {session}")
            return None

        logging.debug(f"📋 Checking {len(matching_strategies)} strategies for session {session}")

        # Check both timeframes
        triggered_signals = []

        for timeframe_str, df in [('5min', df_5m), ('15min', df_15m)]:
            if df is None or len(df) < 2:
                continue

            # Get previous candle (fully closed)
            prev_candle = df.iloc[-2]

            # Ensure we have required columns (case-insensitive)
            col_map = {col.lower(): col for col in df.columns}
            open_col = col_map.get('open')
            close_col = col_map.get('close')

            if not open_col or not close_col:
                logging.warning(f"⚠️ Missing OHLC columns in {timeframe_str} dataframe")
                continue

            prev_open = float(prev_candle[open_col])
            prev_close = float(prev_candle[close_col])
            body = self._calculate_body(prev_open, prev_close)
            abs_body = abs(body)

            is_green = self._is_green_candle(prev_open, prev_close)
            is_red = self._is_red_candle(prev_open, prev_close)

            logging.debug(
                f"   {timeframe_str} prev candle: "
                f"O={prev_open:.2f} C={prev_close:.2f} Body={body:.2f} "
                f"({'GREEN' if is_green else 'RED' if is_red else 'DOJI'})"
            )

            # Filter for current timeframe
            tf_strategies = [s for s in matching_strategies if s['TF'] == timeframe_str]

            logging.debug(f"   Found {len(tf_strategies)} {timeframe_str} strategies")

            for strategy in tf_strategies:
                strategy_type = strategy['Type']
                thresh = float(strategy['Thresh'])

                signal = None

                # Check strategy logic
                if strategy_type == 'Long_Rev':
                    # LONG if prev candle RED and abs(body) > threshold
                    if is_red and abs_body > thresh:
                        signal = 'LONG'

                elif strategy_type == 'Short_Rev':
                    # SHORT if prev candle GREEN and abs(body) > threshold
                    if is_green and abs_body > thresh:
                        signal = 'SHORT'

                elif strategy_type == 'Long_Mom':
                    # LONG if prev candle GREEN and abs(body) > threshold
                    if is_green and abs_body > thresh:
                        signal = 'LONG'

                elif strategy_type == 'Short_Mom':
                    # SHORT if prev candle RED and abs(body) > threshold
                    if is_red and abs_body > thresh:
                        signal = 'SHORT'

                if signal:
                    signal_data = {
                        'signal': signal,
                        'sl': float(strategy['Best_SL']),
                        'tp': float(strategy['Best_TP']),
                        'opt_wr': float(strategy['Opt_WR']),
                        'timeframe': timeframe_str,
                        'strategy_type': strategy_type,
                        'thresh': thresh,
                        'body': abs_body,
                        'year': int(strategy['Year']),
                        'qtr': int(strategy['Qtr']),
                        'strategy_id': (
                            f"{timeframe_str}_{session}_{strategy_type}_T{int(thresh)}_"
                            f"Y{int(strategy['Year'])}Q{int(strategy['Qtr'])}"
                        )
                    }
                    triggered_signals.append(signal_data)

                    logging.info(
                        f"✅ TRIGGER: {strategy_type} on {timeframe_str} | "
                        f"Body={abs_body:.2f} > Thresh={thresh:.2f} | "
                        f"WR={strategy['Opt_WR']:.1%} | "
                        f"SL={strategy['Best_SL']:.1f} TP={strategy['Best_TP']:.1f}"
                    )

        # Tie-breaking: pick highest Opt_WR if multiple signals
        if not triggered_signals:
            logging.debug(f"⚪ No triggers for session {session}")
            return None

        if len(triggered_signals) > 1:
            # Sort by Opt_WR descending
            triggered_signals.sort(key=lambda x: x['opt_wr'], reverse=True)
            logging.info(
                f"🎯 TIE-BREAK: {len(triggered_signals)} signals detected, "
                f"selecting highest WR"
            )
            logging.info(f"   Winner: {triggered_signals[0]['strategy_id']} "
                        f"(WR={triggered_signals[0]['opt_wr']:.1%})")
            for i, sig in enumerate(triggered_signals[1:], 1):
                logging.info(f"   #{i+1}: {sig['strategy_id']} (WR={sig['opt_wr']:.1%})")

        best_signal = triggered_signals[0]

        logging.info("=" * 70)
        logging.info(f"🚀 FINAL SIGNAL SELECTED")
        logging.info(f"   Direction: {best_signal['signal']}")
        logging.info(f"   Timeframe: {best_signal['timeframe']}")
        logging.info(f"   Strategy: {best_signal['strategy_type']}")
        logging.info(f"   Stop Loss: {best_signal['sl']:.1f} points")
        logging.info(f"   Take Profit: {best_signal['tp']:.1f} points")
        logging.info(f"   Win Rate: {best_signal['opt_wr']:.1%}")
        logging.info(f"   Trigger: Body {best_signal['body']:.2f} > Thresh {best_signal['thresh']:.2f}")
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
