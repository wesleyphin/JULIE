"""
Extension Filter Module for MES Futures Trading
===============================================
Detects when price has extended too far beyond normal daily/session range,
blocking continuation trades in the direction of the extension.

Logic:
- Tracks session high/low and daily high/low in real-time
- Compares current range to historical percentiles (90th = extended, 95th = extreme)
- When price is extended UP (large range with price near session high), blocks LONGs
- When price is extended DOWN (large range with price near session low), blocks SHORTs

320 hierarchical thresholds: YearlyQ_MonthlyQ_DayOfWeek_Session
Based on MES 2023-2025 historical data.
"""

from datetime import datetime
from typing import Tuple, Optional, Dict
import logging
import pandas as pd

# ============================================================
# EXTENSION THRESHOLDS - 320 Combinations
# YearlyQ_MonthlyQ_DayOfWeek_Session
# session_extended: 90th percentile of session range
# session_extreme: 95th percentile of session range  
# daily_extended: 90th percentile of daily range
# daily_extreme: 95th percentile of daily range
# ============================================================
EXTENSION_THRESHOLDS = {
    "Q1_W1_MON_ASIA": {"session_extended": 124.00, "session_extreme": 126.50, "daily_extended": 138.95, "daily_extreme": 158.85},
    "Q1_W1_MON_LONDON": {"session_extended": 44.00, "session_extreme": 44.38, "daily_extended": 138.95, "daily_extreme": 158.85},
    "Q1_W1_MON_NY_AM": {"session_extended": 75.20, "session_extreme": 81.72, "daily_extended": 138.95, "daily_extreme": 158.85},
    "Q1_W1_MON_NY_PM": {"session_extended": 94.40, "session_extreme": 119.07, "daily_extended": 138.95, "daily_extreme": 158.85},
    "Q1_W1_TUE_ASIA": {"session_extended": 76.60, "session_extreme": 77.30, "daily_extended": 116.40, "daily_extreme": 128.20},
    "Q1_W1_TUE_LONDON": {"session_extended": 51.30, "session_extreme": 54.40, "daily_extended": 116.40, "daily_extreme": 128.20},
    "Q1_W1_TUE_NY_AM": {"session_extended": 81.15, "session_extreme": 86.45, "daily_extended": 116.40, "daily_extreme": 128.20},
    "Q1_W1_TUE_NY_PM": {"session_extended": 93.50, "session_extreme": 100.50, "daily_extended": 116.40, "daily_extreme": 128.20},
    "Q1_W1_WED_ASIA": {"session_extended": 73.85, "session_extreme": 75.05, "daily_extended": 115.55, "daily_extreme": 117.15},
    "Q1_W1_WED_LONDON": {"session_extended": 30.50, "session_extreme": 39.25, "daily_extended": 115.55, "daily_extreme": 117.15},
    "Q1_W1_WED_NY_AM": {"session_extended": 61.70, "session_extreme": 63.97, "daily_extended": 115.55, "daily_extreme": 117.15},
    "Q1_W1_WED_NY_PM": {"session_extended": 98.47, "session_extreme": 106.61, "daily_extended": 115.55, "daily_extreme": 117.15},
    "Q1_W1_THU_ASIA": {"session_extended": 102.15, "session_extreme": 118.95, "daily_extended": 132.40, "daily_extreme": 137.20},
    "Q1_W1_THU_LONDON": {"session_extended": 44.65, "session_extreme": 49.95, "daily_extended": 132.40, "daily_extreme": 137.20},
    "Q1_W1_THU_NY_AM": {"session_extended": 64.80, "session_extreme": 66.40, "daily_extended": 132.40, "daily_extreme": 137.20},
    "Q1_W1_THU_NY_PM": {"session_extended": 73.10, "session_extreme": 78.80, "daily_extended": 132.40, "daily_extreme": 137.20},
    "Q1_W1_FRI_ASIA": {"session_extended": 67.05, "session_extreme": 73.65, "daily_extended": 112.00, "daily_extreme": 133.55},
    "Q1_W1_FRI_LONDON": {"session_extended": 43.60, "session_extreme": 46.55, "daily_extended": 112.00, "daily_extreme": 133.55},
    "Q1_W1_FRI_NY_AM": {"session_extended": 79.40, "session_extreme": 83.95, "daily_extended": 112.00, "daily_extreme": 133.55},
    "Q1_W1_FRI_NY_PM": {"session_extended": 57.45, "session_extreme": 60.15, "daily_extended": 112.00, "daily_extreme": 133.55},
    "Q1_W2_MON_ASIA": {"session_extended": 86.10, "session_extreme": 88.80, "daily_extended": 98.05, "daily_extreme": 114.25},
    "Q1_W2_MON_LONDON": {"session_extended": 39.50, "session_extreme": 42.50, "daily_extended": 98.05, "daily_extreme": 114.25},
    "Q1_W2_MON_NY_AM": {"session_extended": 57.65, "session_extreme": 62.55, "daily_extended": 98.05, "daily_extreme": 114.25},
    "Q1_W2_MON_NY_PM": {"session_extended": 59.80, "session_extreme": 64.90, "daily_extended": 98.05, "daily_extreme": 114.25},
    "Q1_W2_TUE_ASIA": {"session_extended": 85.10, "session_extreme": 114.30, "daily_extended": 113.55, "daily_extreme": 123.90},
    "Q1_W2_TUE_LONDON": {"session_extended": 45.55, "session_extreme": 53.90, "daily_extended": 113.55, "daily_extreme": 123.90},
    "Q1_W2_TUE_NY_AM": {"session_extended": 73.10, "session_extreme": 79.50, "daily_extended": 113.55, "daily_extreme": 123.90},
    "Q1_W2_TUE_NY_PM": {"session_extended": 61.85, "session_extreme": 69.05, "daily_extended": 113.55, "daily_extreme": 123.90},
    "Q1_W2_WED_ASIA": {"session_extended": 92.60, "session_extreme": 95.30, "daily_extended": 107.65, "daily_extreme": 130.07},
    "Q1_W2_WED_LONDON": {"session_extended": 45.25, "session_extreme": 51.00, "daily_extended": 107.65, "daily_extreme": 130.07},
    "Q1_W2_WED_NY_AM": {"session_extended": 60.70, "session_extreme": 63.35, "daily_extended": 107.65, "daily_extreme": 130.07},
    "Q1_W2_WED_NY_PM": {"session_extended": 66.45, "session_extreme": 74.10, "daily_extended": 107.65, "daily_extreme": 130.07},
    "Q1_W2_THU_ASIA": {"session_extended": 85.90, "session_extreme": 99.95, "daily_extended": 113.85, "daily_extreme": 124.05},
    "Q1_W2_THU_LONDON": {"session_extended": 40.35, "session_extreme": 52.55, "daily_extended": 113.85, "daily_extreme": 124.05},
    "Q1_W2_THU_NY_AM": {"session_extended": 63.70, "session_extreme": 68.85, "daily_extended": 113.85, "daily_extreme": 124.05},
    "Q1_W2_THU_NY_PM": {"session_extended": 63.95, "session_extreme": 70.10, "daily_extended": 113.85, "daily_extreme": 124.05},
    "Q1_W2_FRI_ASIA": {"session_extended": 78.50, "session_extreme": 97.50, "daily_extended": 99.95, "daily_extreme": 108.15},
    "Q1_W2_FRI_LONDON": {"session_extended": 44.00, "session_extreme": 47.60, "daily_extended": 99.95, "daily_extreme": 108.15},
    "Q1_W2_FRI_NY_AM": {"session_extended": 59.70, "session_extreme": 62.85, "daily_extended": 99.95, "daily_extreme": 108.15},
    "Q1_W2_FRI_NY_PM": {"session_extended": 56.70, "session_extreme": 72.10, "daily_extended": 99.95, "daily_extreme": 108.15},
    "Q1_W3_MON_ASIA": {"session_extended": 82.80, "session_extreme": 106.65, "daily_extended": 120.60, "daily_extreme": 147.65},
    "Q1_W3_MON_LONDON": {"session_extended": 40.65, "session_extreme": 47.20, "daily_extended": 120.60, "daily_extreme": 147.65},
    "Q1_W3_MON_NY_AM": {"session_extended": 62.95, "session_extreme": 73.35, "daily_extended": 120.60, "daily_extreme": 147.65},
    "Q1_W3_MON_NY_PM": {"session_extended": 70.60, "session_extreme": 92.70, "daily_extended": 120.60, "daily_extreme": 147.65},
    "Q1_W3_TUE_ASIA": {"session_extended": 97.35, "session_extreme": 112.25, "daily_extended": 128.70, "daily_extreme": 185.95},
    "Q1_W3_TUE_LONDON": {"session_extended": 45.15, "session_extreme": 53.00, "daily_extended": 128.70, "daily_extreme": 185.95},
    "Q1_W3_TUE_NY_AM": {"session_extended": 70.55, "session_extreme": 77.85, "daily_extended": 128.70, "daily_extreme": 185.95},
    "Q1_W3_TUE_NY_PM": {"session_extended": 73.65, "session_extreme": 86.70, "daily_extended": 128.70, "daily_extreme": 185.95},
    "Q1_W3_WED_ASIA": {"session_extended": 104.05, "session_extreme": 154.40, "daily_extended": 137.70, "daily_extreme": 166.00},
    "Q1_W3_WED_LONDON": {"session_extended": 44.25, "session_extreme": 54.70, "daily_extended": 137.70, "daily_extreme": 166.00},
    "Q1_W3_WED_NY_AM": {"session_extended": 77.30, "session_extreme": 88.65, "daily_extended": 137.70, "daily_extreme": 166.00},
    "Q1_W3_WED_NY_PM": {"session_extended": 79.05, "session_extreme": 87.55, "daily_extended": 137.70, "daily_extreme": 166.00},
    "Q1_W3_THU_ASIA": {"session_extended": 85.30, "session_extreme": 102.00, "daily_extended": 113.55, "daily_extreme": 130.50},
    "Q1_W3_THU_LONDON": {"session_extended": 38.45, "session_extreme": 42.20, "daily_extended": 113.55, "daily_extreme": 130.50},
    "Q1_W3_THU_NY_AM": {"session_extended": 62.65, "session_extreme": 72.30, "daily_extended": 113.55, "daily_extreme": 130.50},
    "Q1_W3_THU_NY_PM": {"session_extended": 63.10, "session_extreme": 68.45, "daily_extended": 113.55, "daily_extreme": 130.50},
    "Q1_W3_FRI_ASIA": {"session_extended": 100.70, "session_extreme": 106.10, "daily_extended": 117.40, "daily_extreme": 129.30},
    "Q1_W3_FRI_LONDON": {"session_extended": 51.95, "session_extreme": 60.85, "daily_extended": 117.40, "daily_extreme": 129.30},
    "Q1_W3_FRI_NY_AM": {"session_extended": 66.90, "session_extreme": 72.45, "daily_extended": 117.40, "daily_extreme": 129.30},
    "Q1_W3_FRI_NY_PM": {"session_extended": 67.15, "session_extreme": 78.85, "daily_extended": 117.40, "daily_extreme": 129.30},
    "Q1_W4_MON_ASIA": {"session_extended": 75.85, "session_extreme": 100.55, "daily_extended": 93.60, "daily_extreme": 109.90},
    "Q1_W4_MON_LONDON": {"session_extended": 34.85, "session_extreme": 47.20, "daily_extended": 93.60, "daily_extreme": 109.90},
    "Q1_W4_MON_NY_AM": {"session_extended": 62.60, "session_extreme": 67.35, "daily_extended": 93.60, "daily_extreme": 109.90},
    "Q1_W4_MON_NY_PM": {"session_extended": 57.30, "session_extreme": 72.15, "daily_extended": 93.60, "daily_extreme": 109.90},
    "Q1_W4_TUE_ASIA": {"session_extended": 79.50, "session_extreme": 87.20, "daily_extended": 108.70, "daily_extreme": 124.85},
    "Q1_W4_TUE_LONDON": {"session_extended": 42.95, "session_extreme": 51.85, "daily_extended": 108.70, "daily_extreme": 124.85},
    "Q1_W4_TUE_NY_AM": {"session_extended": 63.65, "session_extreme": 72.95, "daily_extended": 108.70, "daily_extreme": 124.85},
    "Q1_W4_TUE_NY_PM": {"session_extended": 68.20, "session_extreme": 77.40, "daily_extended": 108.70, "daily_extreme": 124.85},
    "Q1_W4_WED_ASIA": {"session_extended": 85.20, "session_extreme": 102.50, "daily_extended": 107.90, "daily_extreme": 123.30},
    "Q1_W4_WED_LONDON": {"session_extended": 37.85, "session_extreme": 47.45, "daily_extended": 107.90, "daily_extreme": 123.30},
    "Q1_W4_WED_NY_AM": {"session_extended": 59.75, "session_extreme": 66.50, "daily_extended": 107.90, "daily_extreme": 123.30},
    "Q1_W4_WED_NY_PM": {"session_extended": 63.35, "session_extreme": 73.10, "daily_extended": 107.90, "daily_extreme": 123.30},
    "Q1_W4_THU_ASIA": {"session_extended": 77.35, "session_extreme": 91.10, "daily_extended": 116.45, "daily_extreme": 129.65},
    "Q1_W4_THU_LONDON": {"session_extended": 38.35, "session_extreme": 45.20, "daily_extended": 116.45, "daily_extreme": 129.65},
    "Q1_W4_THU_NY_AM": {"session_extended": 71.90, "session_extreme": 82.95, "daily_extended": 116.45, "daily_extreme": 129.65},
    "Q1_W4_THU_NY_PM": {"session_extended": 65.65, "session_extreme": 73.65, "daily_extended": 116.45, "daily_extreme": 129.65},
    "Q1_W4_FRI_ASIA": {"session_extended": 90.00, "session_extreme": 105.95, "daily_extended": 117.65, "daily_extreme": 135.05},
    "Q1_W4_FRI_LONDON": {"session_extended": 46.05, "session_extreme": 50.35, "daily_extended": 117.65, "daily_extreme": 135.05},
    "Q1_W4_FRI_NY_AM": {"session_extended": 66.15, "session_extreme": 73.85, "daily_extended": 117.65, "daily_extreme": 135.05},
    "Q1_W4_FRI_NY_PM": {"session_extended": 58.85, "session_extreme": 63.50, "daily_extended": 117.65, "daily_extreme": 135.05},
    "Q2_W1_MON_ASIA": {"session_extended": 80.80, "session_extreme": 97.35, "daily_extended": 102.95, "daily_extreme": 116.95},
    "Q2_W1_MON_LONDON": {"session_extended": 32.30, "session_extreme": 35.25, "daily_extended": 102.95, "daily_extreme": 116.95},
    "Q2_W1_MON_NY_AM": {"session_extended": 52.65, "session_extreme": 57.60, "daily_extended": 102.95, "daily_extreme": 116.95},
    "Q2_W1_MON_NY_PM": {"session_extended": 57.65, "session_extreme": 66.90, "daily_extended": 102.95, "daily_extreme": 116.95},
    "Q2_W1_TUE_ASIA": {"session_extended": 80.90, "session_extreme": 93.85, "daily_extended": 98.95, "daily_extreme": 106.55},
    "Q2_W1_TUE_LONDON": {"session_extended": 31.30, "session_extreme": 35.30, "daily_extended": 98.95, "daily_extreme": 106.55},
    "Q2_W1_TUE_NY_AM": {"session_extended": 57.20, "session_extreme": 60.65, "daily_extended": 98.95, "daily_extreme": 106.55},
    "Q2_W1_TUE_NY_PM": {"session_extended": 56.55, "session_extreme": 61.65, "daily_extended": 98.95, "daily_extreme": 106.55},
    "Q2_W1_WED_ASIA": {"session_extended": 83.80, "session_extreme": 100.95, "daily_extended": 97.35, "daily_extreme": 102.15},
    "Q2_W1_WED_LONDON": {"session_extended": 29.90, "session_extreme": 33.15, "daily_extended": 97.35, "daily_extreme": 102.15},
    "Q2_W1_WED_NY_AM": {"session_extended": 57.45, "session_extreme": 64.40, "daily_extended": 97.35, "daily_extreme": 102.15},
    "Q2_W1_WED_NY_PM": {"session_extended": 49.30, "session_extreme": 54.15, "daily_extended": 97.35, "daily_extreme": 102.15},
    "Q2_W1_THU_ASIA": {"session_extended": 93.45, "session_extreme": 117.75, "daily_extended": 116.80, "daily_extreme": 125.40},
    "Q2_W1_THU_LONDON": {"session_extended": 36.05, "session_extreme": 42.75, "daily_extended": 116.80, "daily_extreme": 125.40},
    "Q2_W1_THU_NY_AM": {"session_extended": 66.95, "session_extreme": 74.70, "daily_extended": 116.80, "daily_extreme": 125.40},
    "Q2_W1_THU_NY_PM": {"session_extended": 64.25, "session_extreme": 82.70, "daily_extended": 116.80, "daily_extreme": 125.40},
    "Q2_W1_FRI_ASIA": {"session_extended": 73.55, "session_extreme": 80.25, "daily_extended": 95.50, "daily_extreme": 100.55},
    "Q2_W1_FRI_LONDON": {"session_extended": 33.20, "session_extreme": 38.95, "daily_extended": 95.50, "daily_extreme": 100.55},
    "Q2_W1_FRI_NY_AM": {"session_extended": 57.65, "session_extreme": 67.30, "daily_extended": 95.50, "daily_extreme": 100.55},
    "Q2_W1_FRI_NY_PM": {"session_extended": 56.95, "session_extreme": 61.70, "daily_extended": 95.50, "daily_extreme": 100.55},
    "Q2_W2_MON_ASIA": {"session_extended": 75.15, "session_extreme": 81.55, "daily_extended": 93.95, "daily_extreme": 103.05},
    "Q2_W2_MON_LONDON": {"session_extended": 30.80, "session_extreme": 34.10, "daily_extended": 93.95, "daily_extreme": 103.05},
    "Q2_W2_MON_NY_AM": {"session_extended": 50.90, "session_extreme": 54.45, "daily_extended": 93.95, "daily_extreme": 103.05},
    "Q2_W2_MON_NY_PM": {"session_extended": 61.35, "session_extreme": 70.30, "daily_extended": 93.95, "daily_extreme": 103.05},
    "Q2_W2_TUE_ASIA": {"session_extended": 75.45, "session_extreme": 84.20, "daily_extended": 94.45, "daily_extreme": 107.40},
    "Q2_W2_TUE_LONDON": {"session_extended": 35.00, "session_extreme": 40.00, "daily_extended": 94.45, "daily_extreme": 107.40},
    "Q2_W2_TUE_NY_AM": {"session_extended": 53.95, "session_extreme": 57.40, "daily_extended": 94.45, "daily_extreme": 107.40},
    "Q2_W2_TUE_NY_PM": {"session_extended": 51.55, "session_extreme": 58.30, "daily_extended": 94.45, "daily_extreme": 107.40},
    "Q2_W2_WED_ASIA": {"session_extended": 85.55, "session_extreme": 99.30, "daily_extended": 115.50, "daily_extreme": 131.40},
    "Q2_W2_WED_LONDON": {"session_extended": 34.95, "session_extreme": 42.10, "daily_extended": 115.50, "daily_extreme": 131.40},
    "Q2_W2_WED_NY_AM": {"session_extended": 57.50, "session_extreme": 65.75, "daily_extended": 115.50, "daily_extreme": 131.40},
    "Q2_W2_WED_NY_PM": {"session_extended": 68.15, "session_extreme": 79.55, "daily_extended": 115.50, "daily_extreme": 131.40},
    "Q2_W2_THU_ASIA": {"session_extended": 72.35, "session_extreme": 87.80, "daily_extended": 97.20, "daily_extreme": 105.80},
    "Q2_W2_THU_LONDON": {"session_extended": 31.30, "session_extreme": 35.45, "daily_extended": 97.20, "daily_extreme": 105.80},
    "Q2_W2_THU_NY_AM": {"session_extended": 60.10, "session_extreme": 70.80, "daily_extended": 97.20, "daily_extreme": 105.80},
    "Q2_W2_THU_NY_PM": {"session_extended": 51.00, "session_extreme": 56.35, "daily_extended": 97.20, "daily_extreme": 105.80},
    "Q2_W2_FRI_ASIA": {"session_extended": 74.60, "session_extreme": 79.65, "daily_extended": 97.25, "daily_extreme": 104.75},
    "Q2_W2_FRI_LONDON": {"session_extended": 30.70, "session_extreme": 35.35, "daily_extended": 97.25, "daily_extreme": 104.75},
    "Q2_W2_FRI_NY_AM": {"session_extended": 55.65, "session_extreme": 64.10, "daily_extended": 97.25, "daily_extreme": 104.75},
    "Q2_W2_FRI_NY_PM": {"session_extended": 59.90, "session_extreme": 70.00, "daily_extended": 97.25, "daily_extreme": 104.75},
    "Q2_W3_MON_ASIA": {"session_extended": 80.50, "session_extreme": 96.15, "daily_extended": 103.60, "daily_extreme": 113.75},
    "Q2_W3_MON_LONDON": {"session_extended": 34.20, "session_extreme": 37.80, "daily_extended": 103.60, "daily_extreme": 113.75},
    "Q2_W3_MON_NY_AM": {"session_extended": 55.30, "session_extreme": 64.05, "daily_extended": 103.60, "daily_extreme": 113.75},
    "Q2_W3_MON_NY_PM": {"session_extended": 57.50, "session_extreme": 62.20, "daily_extended": 103.60, "daily_extreme": 113.75},
    "Q2_W3_TUE_ASIA": {"session_extended": 90.65, "session_extreme": 101.05, "daily_extended": 107.25, "daily_extreme": 118.55},
    "Q2_W3_TUE_LONDON": {"session_extended": 36.30, "session_extreme": 43.50, "daily_extended": 107.25, "daily_extreme": 118.55},
    "Q2_W3_TUE_NY_AM": {"session_extended": 57.25, "session_extreme": 62.95, "daily_extended": 107.25, "daily_extreme": 118.55},
    "Q2_W3_TUE_NY_PM": {"session_extended": 56.50, "session_extreme": 66.55, "daily_extended": 107.25, "daily_extreme": 118.55},
    "Q2_W3_WED_ASIA": {"session_extended": 78.25, "session_extreme": 96.90, "daily_extended": 99.95, "daily_extreme": 114.25},
    "Q2_W3_WED_LONDON": {"session_extended": 34.40, "session_extreme": 41.10, "daily_extended": 99.95, "daily_extreme": 114.25},
    "Q2_W3_WED_NY_AM": {"session_extended": 58.90, "session_extreme": 68.55, "daily_extended": 99.95, "daily_extreme": 114.25},
    "Q2_W3_WED_NY_PM": {"session_extended": 59.95, "session_extreme": 72.20, "daily_extended": 99.95, "daily_extreme": 114.25},
    "Q2_W3_THU_ASIA": {"session_extended": 85.10, "session_extreme": 106.85, "daily_extended": 109.40, "daily_extreme": 122.40},
    "Q2_W3_THU_LONDON": {"session_extended": 34.10, "session_extreme": 39.20, "daily_extended": 109.40, "daily_extreme": 122.40},
    "Q2_W3_THU_NY_AM": {"session_extended": 59.10, "session_extreme": 65.80, "daily_extended": 109.40, "daily_extreme": 122.40},
    "Q2_W3_THU_NY_PM": {"session_extended": 59.65, "session_extreme": 66.25, "daily_extended": 109.40, "daily_extreme": 122.40},
    "Q2_W3_FRI_ASIA": {"session_extended": 89.05, "session_extreme": 111.75, "daily_extended": 104.65, "daily_extreme": 114.90},
    "Q2_W3_FRI_LONDON": {"session_extended": 33.95, "session_extreme": 39.65, "daily_extended": 104.65, "daily_extreme": 114.90},
    "Q2_W3_FRI_NY_AM": {"session_extended": 61.20, "session_extreme": 70.70, "daily_extended": 104.65, "daily_extreme": 114.90},
    "Q2_W3_FRI_NY_PM": {"session_extended": 58.95, "session_extreme": 69.90, "daily_extended": 104.65, "daily_extreme": 114.90},
    "Q2_W4_MON_ASIA": {"session_extended": 75.10, "session_extreme": 89.40, "daily_extended": 91.85, "daily_extreme": 102.50},
    "Q2_W4_MON_LONDON": {"session_extended": 30.85, "session_extreme": 34.20, "daily_extended": 91.85, "daily_extreme": 102.50},
    "Q2_W4_MON_NY_AM": {"session_extended": 51.75, "session_extreme": 59.75, "daily_extended": 91.85, "daily_extreme": 102.50},
    "Q2_W4_MON_NY_PM": {"session_extended": 59.90, "session_extreme": 67.10, "daily_extended": 91.85, "daily_extreme": 102.50},
    "Q2_W4_TUE_ASIA": {"session_extended": 73.75, "session_extreme": 84.40, "daily_extended": 98.10, "daily_extreme": 109.40},
    "Q2_W4_TUE_LONDON": {"session_extended": 33.25, "session_extreme": 38.75, "daily_extended": 98.10, "daily_extreme": 109.40},
    "Q2_W4_TUE_NY_AM": {"session_extended": 58.75, "session_extreme": 66.80, "daily_extended": 98.10, "daily_extreme": 109.40},
    "Q2_W4_TUE_NY_PM": {"session_extended": 52.25, "session_extreme": 60.30, "daily_extended": 98.10, "daily_extreme": 109.40},
    "Q2_W4_WED_ASIA": {"session_extended": 68.10, "session_extreme": 72.90, "daily_extended": 89.70, "daily_extreme": 99.50},
    "Q2_W4_WED_LONDON": {"session_extended": 30.45, "session_extreme": 35.40, "daily_extended": 89.70, "daily_extreme": 99.50},
    "Q2_W4_WED_NY_AM": {"session_extended": 52.60, "session_extreme": 61.75, "daily_extended": 89.70, "daily_extreme": 99.50},
    "Q2_W4_WED_NY_PM": {"session_extended": 57.20, "session_extreme": 64.80, "daily_extended": 89.70, "daily_extreme": 99.50},
    "Q2_W4_THU_ASIA": {"session_extended": 79.80, "session_extreme": 91.70, "daily_extended": 100.45, "daily_extreme": 113.60},
    "Q2_W4_THU_LONDON": {"session_extended": 33.70, "session_extreme": 39.65, "daily_extended": 100.45, "daily_extreme": 113.60},
    "Q2_W4_THU_NY_AM": {"session_extended": 56.80, "session_extreme": 63.30, "daily_extended": 100.45, "daily_extreme": 113.60},
    "Q2_W4_THU_NY_PM": {"session_extended": 55.95, "session_extreme": 62.75, "daily_extended": 100.45, "daily_extreme": 113.60},
    "Q2_W4_FRI_ASIA": {"session_extended": 68.40, "session_extreme": 77.55, "daily_extended": 90.60, "daily_extreme": 95.00},
    "Q2_W4_FRI_LONDON": {"session_extended": 31.60, "session_extreme": 34.70, "daily_extended": 90.60, "daily_extreme": 95.00},
    "Q2_W4_FRI_NY_AM": {"session_extended": 53.45, "session_extreme": 58.85, "daily_extended": 90.60, "daily_extreme": 95.00},
    "Q2_W4_FRI_NY_PM": {"session_extended": 53.65, "session_extreme": 60.00, "daily_extended": 90.60, "daily_extreme": 95.00},
    "Q3_W1_MON_ASIA": {"session_extended": 72.30, "session_extreme": 82.90, "daily_extended": 93.35, "daily_extreme": 101.40},
    "Q3_W1_MON_LONDON": {"session_extended": 32.45, "session_extreme": 38.00, "daily_extended": 93.35, "daily_extreme": 101.40},
    "Q3_W1_MON_NY_AM": {"session_extended": 52.35, "session_extreme": 56.40, "daily_extended": 93.35, "daily_extreme": 101.40},
    "Q3_W1_MON_NY_PM": {"session_extended": 50.60, "session_extreme": 57.25, "daily_extended": 93.35, "daily_extreme": 101.40},
    "Q3_W1_TUE_ASIA": {"session_extended": 62.95, "session_extreme": 70.35, "daily_extended": 85.15, "daily_extreme": 95.35},
    "Q3_W1_TUE_LONDON": {"session_extended": 28.80, "session_extreme": 32.60, "daily_extended": 85.15, "daily_extreme": 95.35},
    "Q3_W1_TUE_NY_AM": {"session_extended": 50.45, "session_extreme": 54.70, "daily_extended": 85.15, "daily_extreme": 95.35},
    "Q3_W1_TUE_NY_PM": {"session_extended": 48.95, "session_extreme": 53.85, "daily_extended": 85.15, "daily_extreme": 95.35},
    "Q3_W1_WED_ASIA": {"session_extended": 71.55, "session_extreme": 80.35, "daily_extended": 96.45, "daily_extreme": 110.80},
    "Q3_W1_WED_LONDON": {"session_extended": 30.15, "session_extreme": 35.25, "daily_extended": 96.45, "daily_extreme": 110.80},
    "Q3_W1_WED_NY_AM": {"session_extended": 50.35, "session_extreme": 56.00, "daily_extended": 96.45, "daily_extreme": 110.80},
    "Q3_W1_WED_NY_PM": {"session_extended": 52.00, "session_extreme": 59.35, "daily_extended": 96.45, "daily_extreme": 110.80},
    "Q3_W1_THU_ASIA": {"session_extended": 68.25, "session_extreme": 75.85, "daily_extended": 89.70, "daily_extreme": 102.50},
    "Q3_W1_THU_LONDON": {"session_extended": 27.95, "session_extreme": 32.05, "daily_extended": 89.70, "daily_extreme": 102.50},
    "Q3_W1_THU_NY_AM": {"session_extended": 53.05, "session_extreme": 59.65, "daily_extended": 89.70, "daily_extreme": 102.50},
    "Q3_W1_THU_NY_PM": {"session_extended": 50.55, "session_extreme": 58.00, "daily_extended": 89.70, "daily_extreme": 102.50},
    "Q3_W1_FRI_ASIA": {"session_extended": 63.70, "session_extreme": 71.90, "daily_extended": 91.85, "daily_extreme": 102.90},
    "Q3_W1_FRI_LONDON": {"session_extended": 28.50, "session_extreme": 32.20, "daily_extended": 91.85, "daily_extreme": 102.90},
    "Q3_W1_FRI_NY_AM": {"session_extended": 56.15, "session_extreme": 64.35, "daily_extended": 91.85, "daily_extreme": 102.90},
    "Q3_W1_FRI_NY_PM": {"session_extended": 57.50, "session_extreme": 68.60, "daily_extended": 91.85, "daily_extreme": 102.90},
    "Q3_W2_MON_ASIA": {"session_extended": 58.25, "session_extreme": 64.80, "daily_extended": 82.55, "daily_extreme": 94.85},
    "Q3_W2_MON_LONDON": {"session_extended": 27.45, "session_extreme": 31.00, "daily_extended": 82.55, "daily_extreme": 94.85},
    "Q3_W2_MON_NY_AM": {"session_extended": 48.95, "session_extreme": 53.50, "daily_extended": 82.55, "daily_extreme": 94.85},
    "Q3_W2_MON_NY_PM": {"session_extended": 46.50, "session_extreme": 51.40, "daily_extended": 82.55, "daily_extreme": 94.85},
    "Q3_W2_TUE_ASIA": {"session_extended": 64.95, "session_extreme": 72.25, "daily_extended": 90.45, "daily_extreme": 101.00},
    "Q3_W2_TUE_LONDON": {"session_extended": 28.30, "session_extreme": 32.30, "daily_extended": 90.45, "daily_extreme": 101.00},
    "Q3_W2_TUE_NY_AM": {"session_extended": 53.55, "session_extreme": 59.85, "daily_extended": 90.45, "daily_extreme": 101.00},
    "Q3_W2_TUE_NY_PM": {"session_extended": 53.30, "session_extreme": 59.70, "daily_extended": 90.45, "daily_extreme": 101.00},
    "Q3_W2_WED_ASIA": {"session_extended": 63.85, "session_extreme": 71.00, "daily_extended": 88.95, "daily_extreme": 100.50},
    "Q3_W2_WED_LONDON": {"session_extended": 28.55, "session_extreme": 34.35, "daily_extended": 88.95, "daily_extreme": 100.50},
    "Q3_W2_WED_NY_AM": {"session_extended": 54.70, "session_extreme": 62.10, "daily_extended": 88.95, "daily_extreme": 100.50},
    "Q3_W2_WED_NY_PM": {"session_extended": 55.00, "session_extreme": 62.35, "daily_extended": 88.95, "daily_extreme": 100.50},
    "Q3_W2_THU_ASIA": {"session_extended": 61.10, "session_extreme": 69.80, "daily_extended": 85.25, "daily_extreme": 93.60},
    "Q3_W2_THU_LONDON": {"session_extended": 26.85, "session_extreme": 29.70, "daily_extended": 85.25, "daily_extreme": 93.60},
    "Q3_W2_THU_NY_AM": {"session_extended": 54.10, "session_extreme": 60.50, "daily_extended": 85.25, "daily_extreme": 93.60},
    "Q3_W2_THU_NY_PM": {"session_extended": 51.65, "session_extreme": 57.20, "daily_extended": 85.25, "daily_extreme": 93.60},
    "Q3_W2_FRI_ASIA": {"session_extended": 62.65, "session_extreme": 69.80, "daily_extended": 84.85, "daily_extreme": 95.35},
    "Q3_W2_FRI_LONDON": {"session_extended": 26.45, "session_extreme": 29.65, "daily_extended": 84.85, "daily_extreme": 95.35},
    "Q3_W2_FRI_NY_AM": {"session_extended": 53.20, "session_extreme": 59.35, "daily_extended": 84.85, "daily_extreme": 95.35},
    "Q3_W2_FRI_NY_PM": {"session_extended": 52.80, "session_extreme": 58.85, "daily_extended": 84.85, "daily_extreme": 95.35},
    "Q3_W3_MON_ASIA": {"session_extended": 64.50, "session_extreme": 75.75, "daily_extended": 86.95, "daily_extreme": 96.65},
    "Q3_W3_MON_LONDON": {"session_extended": 26.60, "session_extreme": 30.40, "daily_extended": 86.95, "daily_extreme": 96.65},
    "Q3_W3_MON_NY_AM": {"session_extended": 50.10, "session_extreme": 56.20, "daily_extended": 86.95, "daily_extreme": 96.65},
    "Q3_W3_MON_NY_PM": {"session_extended": 52.10, "session_extreme": 59.70, "daily_extended": 86.95, "daily_extreme": 96.65},
    "Q3_W3_TUE_ASIA": {"session_extended": 70.95, "session_extreme": 82.30, "daily_extended": 91.10, "daily_extreme": 101.80},
    "Q3_W3_TUE_LONDON": {"session_extended": 29.85, "session_extreme": 33.70, "daily_extended": 91.10, "daily_extreme": 101.80},
    "Q3_W3_TUE_NY_AM": {"session_extended": 53.00, "session_extreme": 58.85, "daily_extended": 91.10, "daily_extreme": 101.80},
    "Q3_W3_TUE_NY_PM": {"session_extended": 51.25, "session_extreme": 57.50, "daily_extended": 91.10, "daily_extreme": 101.80},
    "Q3_W3_WED_ASIA": {"session_extended": 65.65, "session_extreme": 72.40, "daily_extended": 93.25, "daily_extreme": 109.80},
    "Q3_W3_WED_LONDON": {"session_extended": 28.05, "session_extreme": 31.85, "daily_extended": 93.25, "daily_extreme": 109.80},
    "Q3_W3_WED_NY_AM": {"session_extended": 55.15, "session_extreme": 60.65, "daily_extended": 93.25, "daily_extreme": 109.80},
    "Q3_W3_WED_NY_PM": {"session_extended": 54.20, "session_extreme": 62.70, "daily_extended": 93.25, "daily_extreme": 109.80},
    "Q3_W3_THU_ASIA": {"session_extended": 69.80, "session_extreme": 78.25, "daily_extended": 91.30, "daily_extreme": 99.90},
    "Q3_W3_THU_LONDON": {"session_extended": 29.55, "session_extreme": 33.90, "daily_extended": 91.30, "daily_extreme": 99.90},
    "Q3_W3_THU_NY_AM": {"session_extended": 52.80, "session_extreme": 59.30, "daily_extended": 91.30, "daily_extreme": 99.90},
    "Q3_W3_THU_NY_PM": {"session_extended": 54.95, "session_extreme": 60.95, "daily_extended": 91.30, "daily_extreme": 99.90},
    "Q3_W3_FRI_ASIA": {"session_extended": 64.75, "session_extreme": 72.65, "daily_extended": 82.30, "daily_extreme": 90.10},
    "Q3_W3_FRI_LONDON": {"session_extended": 25.80, "session_extreme": 29.50, "daily_extended": 82.30, "daily_extreme": 90.10},
    "Q3_W3_FRI_NY_AM": {"session_extended": 50.95, "session_extreme": 56.15, "daily_extended": 82.30, "daily_extreme": 90.10},
    "Q3_W3_FRI_NY_PM": {"session_extended": 51.75, "session_extreme": 56.15, "daily_extended": 82.30, "daily_extreme": 90.10},
    "Q3_W4_MON_ASIA": {"session_extended": 65.90, "session_extreme": 71.70, "daily_extended": 84.25, "daily_extreme": 96.85},
    "Q3_W4_MON_LONDON": {"session_extended": 28.65, "session_extreme": 32.40, "daily_extended": 84.25, "daily_extreme": 96.85},
    "Q3_W4_MON_NY_AM": {"session_extended": 51.25, "session_extreme": 55.85, "daily_extended": 84.25, "daily_extreme": 96.85},
    "Q3_W4_MON_NY_PM": {"session_extended": 51.65, "session_extreme": 57.70, "daily_extended": 84.25, "daily_extreme": 96.85},
    "Q3_W4_TUE_ASIA": {"session_extended": 66.20, "session_extreme": 75.30, "daily_extended": 92.85, "daily_extreme": 106.15},
    "Q3_W4_TUE_LONDON": {"session_extended": 31.30, "session_extreme": 36.15, "daily_extended": 92.85, "daily_extreme": 106.15},
    "Q3_W4_TUE_NY_AM": {"session_extended": 57.05, "session_extreme": 64.60, "daily_extended": 92.85, "daily_extreme": 106.15},
    "Q3_W4_TUE_NY_PM": {"session_extended": 57.00, "session_extreme": 63.60, "daily_extended": 92.85, "daily_extreme": 106.15},
    "Q3_W4_WED_ASIA": {"session_extended": 67.05, "session_extreme": 74.85, "daily_extended": 85.80, "daily_extreme": 94.00},
    "Q3_W4_WED_LONDON": {"session_extended": 27.55, "session_extreme": 31.10, "daily_extended": 85.80, "daily_extreme": 94.00},
    "Q3_W4_WED_NY_AM": {"session_extended": 51.50, "session_extreme": 56.35, "daily_extended": 85.80, "daily_extreme": 94.00},
    "Q3_W4_WED_NY_PM": {"session_extended": 51.10, "session_extreme": 56.10, "daily_extended": 85.80, "daily_extreme": 94.00},
    "Q3_W4_THU_ASIA": {"session_extended": 65.45, "session_extreme": 73.25, "daily_extended": 86.10, "daily_extreme": 96.40},
    "Q3_W4_THU_LONDON": {"session_extended": 27.45, "session_extreme": 30.85, "daily_extended": 86.10, "daily_extreme": 96.40},
    "Q3_W4_THU_NY_AM": {"session_extended": 52.25, "session_extreme": 57.55, "daily_extended": 86.10, "daily_extreme": 96.40},
    "Q3_W4_THU_NY_PM": {"session_extended": 52.70, "session_extreme": 58.10, "daily_extended": 86.10, "daily_extreme": 96.40},
    "Q3_W4_FRI_ASIA": {"session_extended": 64.10, "session_extreme": 70.00, "daily_extended": 81.00, "daily_extreme": 88.20},
    "Q3_W4_FRI_LONDON": {"session_extended": 27.10, "session_extreme": 30.65, "daily_extended": 81.00, "daily_extreme": 88.20},
    "Q3_W4_FRI_NY_AM": {"session_extended": 50.20, "session_extreme": 55.60, "daily_extended": 81.00, "daily_extreme": 88.20},
    "Q3_W4_FRI_NY_PM": {"session_extended": 52.55, "session_extreme": 57.65, "daily_extended": 81.00, "daily_extreme": 88.20},
    "Q4_W1_MON_ASIA": {"session_extended": 78.35, "session_extreme": 90.55, "daily_extended": 95.70, "daily_extreme": 105.00},
    "Q4_W1_MON_LONDON": {"session_extended": 33.40, "session_extreme": 39.35, "daily_extended": 95.70, "daily_extreme": 105.00},
    "Q4_W1_MON_NY_AM": {"session_extended": 54.45, "session_extreme": 60.60, "daily_extended": 95.70, "daily_extreme": 105.00},
    "Q4_W1_MON_NY_PM": {"session_extended": 53.85, "session_extreme": 60.60, "daily_extended": 95.70, "daily_extreme": 105.00},
    "Q4_W1_TUE_ASIA": {"session_extended": 76.70, "session_extreme": 87.15, "daily_extended": 97.55, "daily_extreme": 109.35},
    "Q4_W1_TUE_LONDON": {"session_extended": 32.15, "session_extreme": 38.15, "daily_extended": 97.55, "daily_extreme": 109.35},
    "Q4_W1_TUE_NY_AM": {"session_extended": 54.75, "session_extreme": 60.85, "daily_extended": 97.55, "daily_extreme": 109.35},
    "Q4_W1_TUE_NY_PM": {"session_extended": 57.25, "session_extreme": 63.80, "daily_extended": 97.55, "daily_extreme": 109.35},
    "Q4_W1_WED_ASIA": {"session_extended": 82.40, "session_extreme": 96.00, "daily_extended": 106.50, "daily_extreme": 120.90},
    "Q4_W1_WED_LONDON": {"session_extended": 35.25, "session_extreme": 41.35, "daily_extended": 106.50, "daily_extreme": 120.90},
    "Q4_W1_WED_NY_AM": {"session_extended": 59.50, "session_extreme": 66.70, "daily_extended": 106.50, "daily_extreme": 120.90},
    "Q4_W1_WED_NY_PM": {"session_extended": 59.05, "session_extreme": 65.50, "daily_extended": 106.50, "daily_extreme": 120.90},
    "Q4_W1_THU_ASIA": {"session_extended": 78.65, "session_extreme": 91.70, "daily_extended": 98.65, "daily_extreme": 111.35},
    "Q4_W1_THU_LONDON": {"session_extended": 32.50, "session_extreme": 38.20, "daily_extended": 98.65, "daily_extreme": 111.35},
    "Q4_W1_THU_NY_AM": {"session_extended": 57.70, "session_extreme": 64.75, "daily_extended": 98.65, "daily_extreme": 111.35},
    "Q4_W1_THU_NY_PM": {"session_extended": 55.25, "session_extreme": 61.65, "daily_extended": 98.65, "daily_extreme": 111.35},
    "Q4_W1_FRI_ASIA": {"session_extended": 73.00, "session_extreme": 82.55, "daily_extended": 91.35, "daily_extreme": 99.55},
    "Q4_W1_FRI_LONDON": {"session_extended": 29.80, "session_extreme": 33.85, "daily_extended": 91.35, "daily_extreme": 99.55},
    "Q4_W1_FRI_NY_AM": {"session_extended": 55.55, "session_extreme": 62.35, "daily_extended": 91.35, "daily_extreme": 99.55},
    "Q4_W1_FRI_NY_PM": {"session_extended": 55.25, "session_extreme": 62.00, "daily_extended": 91.35, "daily_extreme": 99.55},
    "Q4_W2_MON_ASIA": {"session_extended": 76.50, "session_extreme": 87.15, "daily_extended": 97.30, "daily_extreme": 108.80},
    "Q4_W2_MON_LONDON": {"session_extended": 33.50, "session_extreme": 39.00, "daily_extended": 97.30, "daily_extreme": 108.80},
    "Q4_W2_MON_NY_AM": {"session_extended": 55.60, "session_extreme": 61.80, "daily_extended": 97.30, "daily_extreme": 108.80},
    "Q4_W2_MON_NY_PM": {"session_extended": 55.90, "session_extreme": 62.70, "daily_extended": 97.30, "daily_extreme": 108.80},
    "Q4_W2_TUE_ASIA": {"session_extended": 74.35, "session_extreme": 84.35, "daily_extended": 96.80, "daily_extreme": 108.90},
    "Q4_W2_TUE_LONDON": {"session_extended": 31.35, "session_extreme": 36.35, "daily_extended": 96.80, "daily_extreme": 108.90},
    "Q4_W2_TUE_NY_AM": {"session_extended": 57.85, "session_extreme": 65.90, "daily_extended": 96.80, "daily_extreme": 108.90},
    "Q4_W2_TUE_NY_PM": {"session_extended": 57.00, "session_extreme": 64.10, "daily_extended": 96.80, "daily_extreme": 108.90},
    "Q4_W2_WED_ASIA": {"session_extended": 79.30, "session_extreme": 91.65, "daily_extended": 117.05, "daily_extreme": 124.77},
    "Q4_W2_WED_LONDON": {"session_extended": 37.95, "session_extreme": 45.40, "daily_extended": 117.05, "daily_extreme": 124.77},
    "Q4_W2_WED_NY_AM": {"session_extended": 64.25, "session_extreme": 72.90, "daily_extended": 117.05, "daily_extreme": 124.77},
    "Q4_W2_WED_NY_PM": {"session_extended": 71.70, "session_extreme": 93.60, "daily_extended": 117.05, "daily_extreme": 124.77},
    "Q4_W2_THU_ASIA": {"session_extended": 78.65, "session_extreme": 90.40, "daily_extended": 101.20, "daily_extreme": 112.60},
    "Q4_W2_THU_LONDON": {"session_extended": 34.90, "session_extreme": 41.10, "daily_extended": 101.20, "daily_extreme": 112.60},
    "Q4_W2_THU_NY_AM": {"session_extended": 59.95, "session_extreme": 67.45, "daily_extended": 101.20, "daily_extreme": 112.60},
    "Q4_W2_THU_NY_PM": {"session_extended": 56.25, "session_extreme": 63.00, "daily_extended": 101.20, "daily_extreme": 112.60},
    "Q4_W2_FRI_ASIA": {"session_extended": 72.20, "session_extreme": 81.20, "daily_extended": 93.65, "daily_extreme": 103.35},
    "Q4_W2_FRI_LONDON": {"session_extended": 32.50, "session_extreme": 37.55, "daily_extended": 93.65, "daily_extreme": 103.35},
    "Q4_W2_FRI_NY_AM": {"session_extended": 57.00, "session_extreme": 64.25, "daily_extended": 93.65, "daily_extreme": 103.35},
    "Q4_W2_FRI_NY_PM": {"session_extended": 57.45, "session_extreme": 64.95, "daily_extended": 93.65, "daily_extreme": 103.35},
    "Q4_W3_MON_ASIA": {"session_extended": 76.55, "session_extreme": 86.50, "daily_extended": 98.25, "daily_extreme": 108.80},
    "Q4_W3_MON_LONDON": {"session_extended": 34.15, "session_extreme": 40.00, "daily_extended": 98.25, "daily_extreme": 108.80},
    "Q4_W3_MON_NY_AM": {"session_extended": 57.05, "session_extreme": 63.50, "daily_extended": 98.25, "daily_extreme": 108.80},
    "Q4_W3_MON_NY_PM": {"session_extended": 57.65, "session_extreme": 64.75, "daily_extended": 98.25, "daily_extreme": 108.80},
    "Q4_W3_TUE_ASIA": {"session_extended": 79.45, "session_extreme": 90.50, "daily_extended": 106.45, "daily_extreme": 120.70},
    "Q4_W3_TUE_LONDON": {"session_extended": 35.95, "session_extreme": 42.60, "daily_extended": 106.45, "daily_extreme": 120.70},
    "Q4_W3_TUE_NY_AM": {"session_extended": 62.35, "session_extreme": 70.20, "daily_extended": 106.45, "daily_extreme": 120.70},
    "Q4_W3_TUE_NY_PM": {"session_extended": 62.85, "session_extreme": 71.10, "daily_extended": 106.45, "daily_extreme": 120.70},
    "Q4_W3_WED_ASIA": {"session_extended": 87.65, "session_extreme": 102.05, "daily_extended": 112.15, "daily_extreme": 126.80},
    "Q4_W3_WED_LONDON": {"session_extended": 39.55, "session_extreme": 47.45, "daily_extended": 112.15, "daily_extreme": 126.80},
    "Q4_W3_WED_NY_AM": {"session_extended": 67.95, "session_extreme": 77.05, "daily_extended": 112.15, "daily_extreme": 126.80},
    "Q4_W3_WED_NY_PM": {"session_extended": 67.85, "session_extreme": 77.20, "daily_extended": 112.15, "daily_extreme": 126.80},
    "Q4_W3_THU_ASIA": {"session_extended": 79.00, "session_extreme": 90.35, "daily_extended": 100.95, "daily_extreme": 111.95},
    "Q4_W3_THU_LONDON": {"session_extended": 34.60, "session_extreme": 40.55, "daily_extended": 100.95, "daily_extreme": 111.95},
    "Q4_W3_THU_NY_AM": {"session_extended": 60.65, "session_extreme": 68.10, "daily_extended": 100.95, "daily_extreme": 111.95},
    "Q4_W3_THU_NY_PM": {"session_extended": 59.50, "session_extreme": 66.50, "daily_extended": 100.95, "daily_extreme": 111.95},
    "Q4_W3_FRI_ASIA": {"session_extended": 73.35, "session_extreme": 82.75, "daily_extended": 96.00, "daily_extreme": 106.55},
    "Q4_W3_FRI_LONDON": {"session_extended": 33.90, "session_extreme": 39.60, "daily_extended": 96.00, "daily_extreme": 106.55},
    "Q4_W3_FRI_NY_AM": {"session_extended": 59.35, "session_extreme": 66.70, "daily_extended": 96.00, "daily_extreme": 106.55},
    "Q4_W3_FRI_NY_PM": {"session_extended": 58.55, "session_extreme": 65.40, "daily_extended": 96.00, "daily_extreme": 106.55},
    "Q4_W4_MON_ASIA": {"session_extended": 74.40, "session_extreme": 84.15, "daily_extended": 92.95, "daily_extreme": 102.45},
    "Q4_W4_MON_LONDON": {"session_extended": 32.55, "session_extreme": 37.85, "daily_extended": 92.95, "daily_extreme": 102.45},
    "Q4_W4_MON_NY_AM": {"session_extended": 53.80, "session_extreme": 59.50, "daily_extended": 92.95, "daily_extreme": 102.45},
    "Q4_W4_MON_NY_PM": {"session_extended": 53.40, "session_extreme": 59.35, "daily_extended": 92.95, "daily_extreme": 102.45},
    "Q4_W4_TUE_ASIA": {"session_extended": 73.35, "session_extreme": 82.95, "daily_extended": 92.85, "daily_extreme": 103.15},
    "Q4_W4_TUE_LONDON": {"session_extended": 32.25, "session_extreme": 37.45, "daily_extended": 92.85, "daily_extreme": 103.15},
    "Q4_W4_TUE_NY_AM": {"session_extended": 54.50, "session_extreme": 60.55, "daily_extended": 92.85, "daily_extreme": 103.15},
    "Q4_W4_TUE_NY_PM": {"session_extended": 52.80, "session_extreme": 58.65, "daily_extended": 92.85, "daily_extreme": 103.15},
    "Q4_W4_WED_ASIA": {"session_extended": 75.20, "session_extreme": 85.20, "daily_extended": 94.70, "daily_extreme": 105.15},
    "Q4_W4_WED_LONDON": {"session_extended": 32.95, "session_extreme": 38.30, "daily_extended": 94.70, "daily_extreme": 105.15},
    "Q4_W4_WED_NY_AM": {"session_extended": 54.85, "session_extreme": 60.80, "daily_extended": 94.70, "daily_extreme": 105.15},
    "Q4_W4_WED_NY_PM": {"session_extended": 54.45, "session_extreme": 60.35, "daily_extended": 94.70, "daily_extreme": 105.15},
    "Q4_W4_THU_ASIA": {"session_extended": 71.70, "session_extreme": 80.85, "daily_extended": 89.35, "daily_extreme": 98.30},
    "Q4_W4_THU_LONDON": {"session_extended": 30.90, "session_extreme": 35.70, "daily_extended": 89.35, "daily_extreme": 98.30},
    "Q4_W4_THU_NY_AM": {"session_extended": 52.35, "session_extreme": 57.90, "daily_extended": 89.35, "daily_extreme": 98.30},
    "Q4_W4_THU_NY_PM": {"session_extended": 51.90, "session_extreme": 57.50, "daily_extended": 89.35, "daily_extreme": 98.30},
    "Q4_W4_FRI_ASIA": {"session_extended": 66.75, "session_extreme": 74.90, "daily_extended": 84.00, "daily_extreme": 92.40},
    "Q4_W4_FRI_LONDON": {"session_extended": 29.80, "session_extreme": 34.25, "daily_extended": 84.00, "daily_extreme": 92.40},
    "Q4_W4_FRI_NY_AM": {"session_extended": 50.40, "session_extreme": 55.80, "daily_extended": 84.00, "daily_extreme": 92.40},
    "Q4_W4_FRI_NY_PM": {"session_extended": 49.90, "session_extreme": 55.20, "daily_extended": 84.00, "daily_extreme": 92.40},
}

# Default thresholds as fallback
DEFAULT_THRESHOLDS = {
    "session_extended": 60.0,
    "session_extreme": 80.0,
    "daily_extended": 100.0,
    "daily_extreme": 130.0
}


class ExtensionFilter:
    """
    Detects when price has moved too far beyond normal range,
    blocking continuation trades in the extended direction.
    
    Tracks both session range and daily range in real-time.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracking state."""
        self.daily_high: Optional[float] = None
        self.daily_low: Optional[float] = None
        self.current_date = None
        
        self.session_high: Optional[float] = None
        self.session_low: Optional[float] = None
        self.current_session: Optional[str] = None
        
        self.state = 'NORMAL'
        self.extension_direction: Optional[str] = None
        self.current_thresholds: Dict = DEFAULT_THRESHOLDS.copy()
    
    def _get_session(self, hour: int) -> str:
        """Determine session from hour (ET)."""
        if 18 <= hour <= 23 or 0 <= hour < 3:
            return 'ASIA'
        elif 3 <= hour < 8:
            return 'LONDON'
        elif 8 <= hour < 12:
            return 'NY_AM'
        elif 12 <= hour < 17:
            return 'NY_PM'
        return 'CLOSED'
    
    def _get_yearly_quarter(self, month: int) -> str:
        if month <= 3: return 'Q1'
        elif month <= 6: return 'Q2'
        elif month <= 9: return 'Q3'
        return 'Q4'
    
    def _get_monthly_quarter(self, day: int) -> str:
        if day <= 7: return 'W1'
        elif day <= 14: return 'W2'
        elif day <= 21: return 'W3'
        return 'W4'
    
    def _get_day_of_week(self, dow: int) -> str:
        DOW_MAP = {0: 'MON', 1: 'TUE', 2: 'WED', 3: 'THU', 4: 'FRI', 5: 'SAT', 6: 'SUN'}
        return DOW_MAP.get(dow, 'MON')
    
    def _get_thresholds(self, dt: datetime) -> Dict:
        """Get thresholds for current time context."""
        yearly_q = self._get_yearly_quarter(dt.month)
        monthly_q = self._get_monthly_quarter(dt.day)
        dow = self._get_day_of_week(dt.weekday())
        session = self._get_session(dt.hour)
        
        key = f"{yearly_q}_{monthly_q}_{dow}_{session}"
        return EXTENSION_THRESHOLDS.get(key, DEFAULT_THRESHOLDS)
    
    def _calculate_extension_state(self, current_price: float) -> str:
        """
        Determine extension state based on range size and how it was created.
        
        Key insight: If market extended UP (made big rally), block LONGs even if 
        price pulls back - the upside move is exhausted. Same for DOWN extensions.
        
        Logic:
        - If session/daily range is extended AND we're still in upper half -> EXTENDED_UP (block longs)
        - If session/daily range is extended AND we're still in lower half -> EXTENDED_DOWN (block shorts)
        - The 50% midpoint determines which direction was the extension
        """
        if self.session_high is None or self.session_low is None:
            return 'NORMAL'
        
        session_range = self.session_high - self.session_low
        daily_range = (self.daily_high - self.daily_low) if self.daily_high and self.daily_low else 0
        
        sess_extended = self.current_thresholds['session_extended']
        sess_extreme = self.current_thresholds['session_extreme']
        daily_extended = self.current_thresholds['daily_extended']
        daily_extreme = self.current_thresholds['daily_extreme']
        
        # Calculate midpoint and which half price is in
        session_mid = (self.session_high + self.session_low) / 2
        daily_mid = (self.daily_high + self.daily_low) / 2 if self.daily_high and self.daily_low else session_mid
        
        # Use daily midpoint for direction since it's the bigger picture
        in_upper_half = current_price >= daily_mid
        
        # Check for extreme daily extension
        if daily_range >= daily_extreme:
            if in_upper_half:
                return 'EXTREME_UP'  # Big range, price in upper half = extended UP, block longs
            else:
                return 'EXTREME_DOWN'  # Big range, price in lower half = extended DOWN, block shorts
        
        if daily_range >= daily_extended:
            if in_upper_half:
                return 'EXTENDED_UP'
            else:
                return 'EXTENDED_DOWN'
        
        # Check session extension
        session_in_upper = current_price >= session_mid
        
        if session_range >= sess_extreme:
            if session_in_upper:
                return 'EXTREME_UP'
            else:
                return 'EXTREME_DOWN'
        
        if session_range >= sess_extended:
            if session_in_upper:
                return 'EXTENDED_UP'
            else:
                return 'EXTENDED_DOWN'
        
        return 'NORMAL'
    
    def update(self, high: float, low: float, close: float, dt: datetime) -> str:
        """Update filter with new bar data."""
        current_date = dt.date()
        hour = dt.hour
        session = self._get_session(hour)
        
        if session == 'CLOSED':
            return self.state
        
        self.current_thresholds = self._get_thresholds(dt)
        
        # New day at 6pm ET
        if hour >= 18 and (self.current_date is None or self.current_date != current_date):
            self.daily_high = high
            self.daily_low = low
            self.current_date = current_date
            logging.info(f"📏 ExtFilter: NEW DAY - Reset daily range")
        
        # New session
        if session != self.current_session:
            self.session_high = high
            self.session_low = low
            self.current_session = session
            logging.info(f"📏 ExtFilter: NEW SESSION {session}")
        
        # Update ranges
        if self.daily_high is not None:
            self.daily_high = max(self.daily_high, high)
            self.daily_low = min(self.daily_low, low)
        else:
            self.daily_high = high
            self.daily_low = low
        
        if self.session_high is not None:
            self.session_high = max(self.session_high, high)
            self.session_low = min(self.session_low, low)
        else:
            self.session_high = high
            self.session_low = low
        
        new_state = self._calculate_extension_state(close)
        
        if new_state != self.state:
            session_range = self.session_high - self.session_low if self.session_high and self.session_low else 0
            daily_range = self.daily_high - self.daily_low if self.daily_high and self.daily_low else 0
            logging.info(
                f"📏 ExtFilter: {self.state} -> {new_state} | "
                f"Session: {session_range:.2f} (thresh: {self.current_thresholds['session_extended']:.2f}) | "
                f"Daily: {daily_range:.2f} (thresh: {self.current_thresholds['daily_extended']:.2f})"
            )
        
        self.state = new_state
        
        if new_state in ['EXTENDED_UP', 'EXTREME_UP']:
            self.extension_direction = 'UP'
        elif new_state in ['EXTENDED_DOWN', 'EXTREME_DOWN']:
            self.extension_direction = 'DOWN'
        else:
            self.extension_direction = None
        
        return self.state
    
    def should_block_trade(self, direction: str) -> Tuple[bool, Optional[str]]:
        """Check if trade should be blocked based on extension state."""
        direction = direction.upper()
        
        if self.state in ['EXTENDED_UP', 'EXTREME_UP'] and direction == 'LONG':
            session_range = self.session_high - self.session_low if self.session_high and self.session_low else 0
            daily_range = self.daily_high - self.daily_low if self.daily_high and self.daily_low else 0
            severity = "EXTREME" if "EXTREME" in self.state else "Extended"
            return True, f"{severity} UP: Session {session_range:.1f}pt, Daily {daily_range:.1f}pt - blocking LONG"
        
        if self.state in ['EXTENDED_DOWN', 'EXTREME_DOWN'] and direction == 'SHORT':
            session_range = self.session_high - self.session_low if self.session_high and self.session_low else 0
            daily_range = self.daily_high - self.daily_low if self.daily_high and self.daily_low else 0
            severity = "EXTREME" if "EXTREME" in self.state else "Extended"
            return True, f"{severity} DOWN: Session {session_range:.1f}pt, Daily {daily_range:.1f}pt - blocking SHORT"
        
        return False, None
    
    def get_status(self) -> Dict:
        """Get current filter status for logging."""
        session_range = self.session_high - self.session_low if self.session_high and self.session_low else 0
        daily_range = self.daily_high - self.daily_low if self.daily_high and self.daily_low else 0

        return {
            'state': self.state,
            'session': self.current_session,
            'session_high': self.session_high,
            'session_low': self.session_low,
            'session_range': session_range,
            'daily_high': self.daily_high,
            'daily_low': self.daily_low,
            'daily_range': daily_range,
            'thresholds': self.current_thresholds,
            'extension_direction': self.extension_direction
        }

    def backfill(self, df: pd.DataFrame):
        """Pre-load daily high/low from historical data on startup."""
        if df.empty:
            return

        # Filter for today's data (assuming df index is localized or handled elsewhere)
        # Simplified logic: just grab the max/min of the provided dataframe
        # Ideally, filter for 'current session' if df contains multiple days
        self.daily_high = df['high'].max()
        self.daily_low = df['low'].min()

        logging.info(f"✅ ExtensionFilter Backfilled: Daily Range {self.daily_low} - {self.daily_high}")
