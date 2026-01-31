"""
ChopFilter Module - Dynamic Consolidation Detection with Structure Validation
==============================================================================

320 hierarchical thresholds based on 2023-2025 MES futures data.
Key format: YearlyQ_MonthlyQ_DayOfWeek_Session

Logic:
1. Chop Detection: If 20-bar range < chop_threshold, market is consolidating
2. Breakout Detection: When price breaks above/below chop range
3. Structure Validation: After breakout, must see HH/HL for longs, LH/LL for shorts
   - If price breaks above important level but makes LH (not HH), block long continuations
   - If price breaks below important level but makes HL (not LL), block short continuations
"""

from datetime import datetime
from collections import deque

# 320 Hierarchical Chop Thresholds
# Key: YearlyQ_MonthlyQ_DayOfWeek_Session
CHOP_THRESHOLDS = {
    "Q1_W1_MON_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.25},
    "Q1_W1_MON_LONDON": {"chop": 3.50, "median": 5.25, "breakout": 7.75},
    "Q1_W1_MON_NY_AM": {"chop": 5.75, "median": 8.62, "breakout": 15.50},
    "Q1_W1_MON_NY_PM": {"chop": 5.75, "median": 9.00, "breakout": 13.81},
    "Q1_W1_TUE_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.00},
    "Q1_W1_TUE_LONDON": {"chop": 3.50, "median": 5.00, "breakout": 8.50},
    "Q1_W1_TUE_NY_AM": {"chop": 6.50, "median": 9.25, "breakout": 15.25},
    "Q1_W1_TUE_NY_PM": {"chop": 6.50, "median": 9.75, "breakout": 16.00},
    "Q1_W1_WED_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.25},
    "Q1_W1_WED_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.25},
    "Q1_W1_WED_NY_AM": {"chop": 6.25, "median": 9.50, "breakout": 15.75},
    "Q1_W1_WED_NY_PM": {"chop": 5.75, "median": 8.25, "breakout": 16.06},
    "Q1_W1_THU_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.50},
    "Q1_W1_THU_LONDON": {"chop": 4.00, "median": 5.50, "breakout": 8.25},
    "Q1_W1_THU_NY_AM": {"chop": 7.50, "median": 10.50, "breakout": 15.25},
    "Q1_W1_THU_NY_PM": {"chop": 6.75, "median": 10.00, "breakout": 17.00},
    "Q1_W1_FRI_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.50},
    "Q1_W1_FRI_LONDON": {"chop": 3.50, "median": 4.75, "breakout": 6.25},
    "Q1_W1_FRI_NY_AM": {"chop": 8.25, "median": 13.00, "breakout": 21.50},
    "Q1_W1_FRI_NY_PM": {"chop": 5.75, "median": 8.25, "breakout": 12.50},
    "Q1_W2_MON_ASIA": {"chop": 2.25, "median": 3.75, "breakout": 9.50},
    "Q1_W2_MON_LONDON": {"chop": 3.75, "median": 6.25, "breakout": 16.25},
    "Q1_W2_MON_NY_AM": {"chop": 6.00, "median": 10.25, "breakout": 19.25},
    "Q1_W2_MON_NY_PM": {"chop": 5.75, "median": 8.75, "breakout": 15.25},
    "Q1_W2_TUE_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.50},
    "Q1_W2_TUE_LONDON": {"chop": 4.00, "median": 5.25, "breakout": 8.06},
    "Q1_W2_TUE_NY_AM": {"chop": 7.75, "median": 12.50, "breakout": 21.50},
    "Q1_W2_TUE_NY_PM": {"chop": 6.75, "median": 10.00, "breakout": 16.00},
    "Q1_W2_WED_ASIA": {"chop": 2.00, "median": 2.50, "breakout": 4.00},
    "Q1_W2_WED_LONDON": {"chop": 3.50, "median": 4.50, "breakout": 6.50},
    "Q1_W2_WED_NY_AM": {"chop": 6.25, "median": 10.25, "breakout": 16.75},
    "Q1_W2_WED_NY_PM": {"chop": 5.75, "median": 8.50, "breakout": 13.00},
    "Q1_W2_THU_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.00},
    "Q1_W2_THU_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.25},
    "Q1_W2_THU_NY_AM": {"chop": 6.75, "median": 11.75, "breakout": 18.19},
    "Q1_W2_THU_NY_PM": {"chop": 6.25, "median": 9.50, "breakout": 14.25},
    "Q1_W2_FRI_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.25},
    "Q1_W2_FRI_LONDON": {"chop": 4.00, "median": 5.75, "breakout": 8.25},
    "Q1_W2_FRI_NY_AM": {"chop": 7.25, "median": 11.75, "breakout": 19.00},
    "Q1_W2_FRI_NY_PM": {"chop": 5.50, "median": 8.75, "breakout": 13.75},
    "Q1_W3_MON_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 5.25},
    "Q1_W3_MON_LONDON": {"chop": 2.75, "median": 4.25, "breakout": 7.25},
    "Q1_W3_MON_NY_AM": {"chop": 2.50, "median": 4.00, "breakout": 11.75},
    "Q1_W3_MON_NY_PM": {"chop": 3.50, "median": 8.00, "breakout": 13.25},
    "Q1_W3_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.25},
    "Q1_W3_TUE_LONDON": {"chop": 4.25, "median": 5.50, "breakout": 8.00},
    "Q1_W3_TUE_NY_AM": {"chop": 6.25, "median": 9.00, "breakout": 12.25},
    "Q1_W3_TUE_NY_PM": {"chop": 5.25, "median": 7.25, "breakout": 10.00},
    "Q1_W3_WED_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.50},
    "Q1_W3_WED_LONDON": {"chop": 3.50, "median": 4.50, "breakout": 6.50},
    "Q1_W3_WED_NY_AM": {"chop": 6.00, "median": 9.50, "breakout": 13.75},
    "Q1_W3_WED_NY_PM": {"chop": 5.75, "median": 8.75, "breakout": 13.50},
    "Q1_W3_THU_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.00},
    "Q1_W3_THU_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 7.00},
    "Q1_W3_THU_NY_AM": {"chop": 7.25, "median": 10.25, "breakout": 15.00},
    "Q1_W3_THU_NY_PM": {"chop": 6.25, "median": 8.75, "breakout": 12.25},
    "Q1_W3_FRI_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 3.50},
    "Q1_W3_FRI_LONDON": {"chop": 3.25, "median": 5.00, "breakout": 7.00},
    "Q1_W3_FRI_NY_AM": {"chop": 7.75, "median": 11.12, "breakout": 16.00},
    "Q1_W3_FRI_NY_PM": {"chop": 6.00, "median": 9.00, "breakout": 12.00},
    "Q1_W4_MON_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.50},
    "Q1_W4_MON_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 7.75},
    "Q1_W4_MON_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 13.00},
    "Q1_W4_MON_NY_PM": {"chop": 5.25, "median": 7.75, "breakout": 11.25},
    "Q1_W4_TUE_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 3.50},
    "Q1_W4_TUE_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.00},
    "Q1_W4_TUE_NY_AM": {"chop": 5.25, "median": 7.25, "breakout": 11.50},
    "Q1_W4_TUE_NY_PM": {"chop": 4.50, "median": 6.25, "breakout": 9.00},
    "Q1_W4_WED_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 3.75},
    "Q1_W4_WED_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 5.50},
    "Q1_W4_WED_NY_AM": {"chop": 5.00, "median": 7.25, "breakout": 10.50},
    "Q1_W4_WED_NY_PM": {"chop": 5.75, "median": 8.50, "breakout": 14.31},
    "Q1_W4_THU_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.50},
    "Q1_W4_THU_LONDON": {"chop": 3.50, "median": 4.50, "breakout": 6.00},
    "Q1_W4_THU_NY_AM": {"chop": 6.25, "median": 9.00, "breakout": 14.25},
    "Q1_W4_THU_NY_PM": {"chop": 5.25, "median": 8.00, "breakout": 12.25},
    "Q1_W4_FRI_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.00},
    "Q1_W4_FRI_LONDON": {"chop": 3.25, "median": 4.00, "breakout": 5.50},
    "Q1_W4_FRI_NY_AM": {"chop": 6.00, "median": 9.00, "breakout": 14.25},
    "Q1_W4_FRI_NY_PM": {"chop": 5.75, "median": 8.25, "breakout": 12.50},
    "Q2_W1_MON_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.75},
    "Q2_W1_MON_LONDON": {"chop": 2.50, "median": 4.25, "breakout": 7.00},
    "Q2_W1_MON_NY_AM": {"chop": 5.00, "median": 8.50, "breakout": 13.25},
    "Q2_W1_MON_NY_PM": {"chop": 5.00, "median": 7.00, "breakout": 10.50},
    "Q2_W1_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.25},
    "Q2_W1_TUE_LONDON": {"chop": 3.50, "median": 5.00, "breakout": 7.25},
    "Q2_W1_TUE_NY_AM": {"chop": 5.50, "median": 8.75, "breakout": 13.00},
    "Q2_W1_TUE_NY_PM": {"chop": 5.50, "median": 7.50, "breakout": 11.50},
    "Q2_W1_WED_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 5.00},
    "Q2_W1_WED_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.25},
    "Q2_W1_WED_NY_AM": {"chop": 6.25, "median": 9.00, "breakout": 12.25},
    "Q2_W1_WED_NY_PM": {"chop": 5.50, "median": 8.00, "breakout": 15.50},
    "Q2_W1_THU_ASIA": {"chop": 1.75, "median": 3.00, "breakout": 5.75},
    "Q2_W1_THU_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 7.31},
    "Q2_W1_THU_NY_AM": {"chop": 6.25, "median": 10.50, "breakout": 17.50},
    "Q2_W1_THU_NY_PM": {"chop": 6.25, "median": 10.25, "breakout": 16.75},
    "Q2_W1_FRI_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 5.75},
    "Q2_W1_FRI_LONDON": {"chop": 2.75, "median": 4.25, "breakout": 6.25},
    "Q2_W1_FRI_NY_AM": {"chop": 8.25, "median": 14.00, "breakout": 21.00},
    "Q2_W1_FRI_NY_PM": {"chop": 6.00, "median": 8.25, "breakout": 12.31},
    "Q2_W2_MON_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 5.75},
    "Q2_W2_MON_LONDON": {"chop": 3.00, "median": 4.75, "breakout": 9.75},
    "Q2_W2_MON_NY_AM": {"chop": 4.94, "median": 7.50, "breakout": 12.50},
    "Q2_W2_MON_NY_PM": {"chop": 4.25, "median": 6.50, "breakout": 10.50},
    "Q2_W2_TUE_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 5.75},
    "Q2_W2_TUE_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 7.56},
    "Q2_W2_TUE_NY_AM": {"chop": 5.50, "median": 8.25, "breakout": 14.75},
    "Q2_W2_TUE_NY_PM": {"chop": 4.75, "median": 7.00, "breakout": 10.75},
    "Q2_W2_WED_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 5.50},
    "Q2_W2_WED_LONDON": {"chop": 2.75, "median": 3.75, "breakout": 6.25},
    "Q2_W2_WED_NY_AM": {"chop": 6.75, "median": 10.75, "breakout": 19.00},
    "Q2_W2_WED_NY_PM": {"chop": 6.25, "median": 10.25, "breakout": 17.75},
    "Q2_W2_THU_ASIA": {"chop": 2.00, "median": 4.00, "breakout": 14.75},
    "Q2_W2_THU_LONDON": {"chop": 3.75, "median": 5.00, "breakout": 7.25},
    "Q2_W2_THU_NY_AM": {"chop": 7.25, "median": 10.25, "breakout": 16.25},
    "Q2_W2_THU_NY_PM": {"chop": 5.00, "median": 7.25, "breakout": 11.25},
    "Q2_W2_FRI_ASIA": {"chop": 2.00, "median": 3.75, "breakout": 17.25},
    "Q2_W2_FRI_LONDON": {"chop": 3.50, "median": 5.50, "breakout": 10.00},
    "Q2_W2_FRI_NY_AM": {"chop": 6.75, "median": 10.75, "breakout": 16.25},
    "Q2_W2_FRI_NY_PM": {"chop": 6.00, "median": 9.00, "breakout": 13.25},
    "Q2_W3_MON_ASIA": {"chop": 2.00, "median": 3.75, "breakout": 7.75},
    "Q2_W3_MON_LONDON": {"chop": 3.00, "median": 5.00, "breakout": 10.25},
    "Q2_W3_MON_NY_AM": {"chop": 4.75, "median": 8.25, "breakout": 14.25},
    "Q2_W3_MON_NY_PM": {"chop": 5.00, "median": 8.25, "breakout": 13.00},
    "Q2_W3_TUE_ASIA": {"chop": 1.75, "median": 3.50, "breakout": 7.25},
    "Q2_W3_TUE_LONDON": {"chop": 3.25, "median": 5.00, "breakout": 8.75},
    "Q2_W3_TUE_NY_AM": {"chop": 5.50, "median": 8.50, "breakout": 12.50},
    "Q2_W3_TUE_NY_PM": {"chop": 5.00, "median": 7.25, "breakout": 12.25},
    "Q2_W3_WED_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 6.50},
    "Q2_W3_WED_LONDON": {"chop": 2.75, "median": 5.00, "breakout": 8.00},
    "Q2_W3_WED_NY_AM": {"chop": 5.50, "median": 8.75, "breakout": 14.25},
    "Q2_W3_WED_NY_PM": {"chop": 5.25, "median": 8.75, "breakout": 16.00},
    "Q2_W3_THU_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 5.00},
    "Q2_W3_THU_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 7.00},
    "Q2_W3_THU_NY_AM": {"chop": 6.50, "median": 9.25, "breakout": 13.00},
    "Q2_W3_THU_NY_PM": {"chop": 6.25, "median": 9.75, "breakout": 14.50},
    "Q2_W3_FRI_ASIA": {"chop": 1.75, "median": 3.00, "breakout": 5.19},
    "Q2_W3_FRI_LONDON": {"chop": 3.50, "median": 5.00, "breakout": 7.00},
    "Q2_W3_FRI_NY_AM": {"chop": 5.75, "median": 9.00, "breakout": 13.75},
    "Q2_W3_FRI_NY_PM": {"chop": 5.50, "median": 7.50, "breakout": 10.00},
    "Q2_W4_MON_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.25},
    "Q2_W4_MON_LONDON": {"chop": 2.75, "median": 4.25, "breakout": 5.75},
    "Q2_W4_MON_NY_AM": {"chop": 4.00, "median": 6.50, "breakout": 10.75},
    "Q2_W4_MON_NY_PM": {"chop": 4.50, "median": 7.00, "breakout": 11.25},
    "Q2_W4_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.50},
    "Q2_W4_TUE_LONDON": {"chop": 3.50, "median": 5.00, "breakout": 7.25},
    "Q2_W4_TUE_NY_AM": {"chop": 5.50, "median": 8.25, "breakout": 12.00},
    "Q2_W4_TUE_NY_PM": {"chop": 5.50, "median": 7.75, "breakout": 10.50},
    "Q2_W4_WED_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.50},
    "Q2_W4_WED_LONDON": {"chop": 3.50, "median": 4.50, "breakout": 6.50},
    "Q2_W4_WED_NY_AM": {"chop": 5.00, "median": 7.75, "breakout": 11.75},
    "Q2_W4_WED_NY_PM": {"chop": 6.00, "median": 8.75, "breakout": 14.25},
    "Q2_W4_THU_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q2_W4_THU_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.50},
    "Q2_W4_THU_NY_AM": {"chop": 6.75, "median": 10.25, "breakout": 14.75},
    "Q2_W4_THU_NY_PM": {"chop": 5.50, "median": 8.25, "breakout": 12.00},
    "Q2_W4_FRI_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.50},
    "Q2_W4_FRI_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 6.50},
    "Q2_W4_FRI_NY_AM": {"chop": 7.25, "median": 10.50, "breakout": 15.25},
    "Q2_W4_FRI_NY_PM": {"chop": 5.50, "median": 7.50, "breakout": 12.50},
    "Q3_W1_MON_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.25},
    "Q3_W1_MON_LONDON": {"chop": 2.75, "median": 4.25, "breakout": 6.25},
    "Q3_W1_MON_NY_AM": {"chop": 3.25, "median": 6.00, "breakout": 10.50},
    "Q3_W1_MON_NY_PM": {"chop": 4.25, "median": 6.75, "breakout": 12.75},
    "Q3_W1_TUE_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 5.50},
    "Q3_W1_TUE_LONDON": {"chop": 3.25, "median": 5.00, "breakout": 7.25},
    "Q3_W1_TUE_NY_AM": {"chop": 5.50, "median": 9.25, "breakout": 14.00},
    "Q3_W1_TUE_NY_PM": {"chop": 4.75, "median": 8.00, "breakout": 11.75},
    "Q3_W1_WED_ASIA": {"chop": 2.00, "median": 3.50, "breakout": 5.50},
    "Q3_W1_WED_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 7.25},
    "Q3_W1_WED_NY_AM": {"chop": 5.50, "median": 8.50, "breakout": 12.75},
    "Q3_W1_WED_NY_PM": {"chop": 4.75, "median": 7.00, "breakout": 10.75},
    "Q3_W1_THU_ASIA": {"chop": 2.00, "median": 3.25, "breakout": 5.25},
    "Q3_W1_THU_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 6.75},
    "Q3_W1_THU_NY_AM": {"chop": 5.50, "median": 8.75, "breakout": 13.25},
    "Q3_W1_THU_NY_PM": {"chop": 5.25, "median": 7.75, "breakout": 14.00},
    "Q3_W1_FRI_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q3_W1_FRI_LONDON": {"chop": 3.00, "median": 4.75, "breakout": 7.75},
    "Q3_W1_FRI_NY_AM": {"chop": 6.50, "median": 11.75, "breakout": 22.50},
    "Q3_W1_FRI_NY_PM": {"chop": 5.25, "median": 9.00, "breakout": 13.00},
    "Q3_W2_MON_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 5.00},
    "Q3_W2_MON_LONDON": {"chop": 3.19, "median": 4.50, "breakout": 6.50},
    "Q3_W2_MON_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 12.50},
    "Q3_W2_MON_NY_PM": {"chop": 4.50, "median": 6.50, "breakout": 9.75},
    "Q3_W2_TUE_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.50},
    "Q3_W2_TUE_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.50},
    "Q3_W2_TUE_NY_AM": {"chop": 5.50, "median": 8.00, "breakout": 12.00},
    "Q3_W2_TUE_NY_PM": {"chop": 4.50, "median": 5.75, "breakout": 8.75},
    "Q3_W2_WED_ASIA": {"chop": 2.00, "median": 2.50, "breakout": 3.75},
    "Q3_W2_WED_LONDON": {"chop": 2.75, "median": 3.75, "breakout": 5.00},
    "Q3_W2_WED_NY_AM": {"chop": 5.50, "median": 8.50, "breakout": 13.75},
    "Q3_W2_WED_NY_PM": {"chop": 5.25, "median": 7.00, "breakout": 10.25},
    "Q3_W2_THU_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.25},
    "Q3_W2_THU_LONDON": {"chop": 2.50, "median": 3.50, "breakout": 5.00},
    "Q3_W2_THU_NY_AM": {"chop": 6.50, "median": 10.50, "breakout": 15.50},
    "Q3_W2_THU_NY_PM": {"chop": 4.75, "median": 7.00, "breakout": 11.25},
    "Q3_W2_FRI_ASIA": {"chop": 2.25, "median": 3.50, "breakout": 6.25},
    "Q3_W2_FRI_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 6.25},
    "Q3_W2_FRI_NY_AM": {"chop": 5.50, "median": 8.50, "breakout": 12.50},
    "Q3_W2_FRI_NY_PM": {"chop": 5.00, "median": 7.25, "breakout": 10.25},
    "Q3_W3_MON_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 5.50},
    "Q3_W3_MON_LONDON": {"chop": 3.00, "median": 4.00, "breakout": 8.56},
    "Q3_W3_MON_NY_AM": {"chop": 4.75, "median": 7.25, "breakout": 12.00},
    "Q3_W3_MON_NY_PM": {"chop": 4.00, "median": 6.00, "breakout": 9.50},
    "Q3_W3_TUE_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.50},
    "Q3_W3_TUE_LONDON": {"chop": 2.75, "median": 4.00, "breakout": 6.25},
    "Q3_W3_TUE_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 11.50},
    "Q3_W3_TUE_NY_PM": {"chop": 4.75, "median": 6.75, "breakout": 9.50},
    "Q3_W3_WED_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.50},
    "Q3_W3_WED_LONDON": {"chop": 3.25, "median": 4.00, "breakout": 5.75},
    "Q3_W3_WED_NY_AM": {"chop": 4.75, "median": 7.25, "breakout": 11.25},
    "Q3_W3_WED_NY_PM": {"chop": 5.75, "median": 8.75, "breakout": 13.25},
    "Q3_W3_THU_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 4.50},
    "Q3_W3_THU_LONDON": {"chop": 3.50, "median": 4.75, "breakout": 6.50},
    "Q3_W3_THU_NY_AM": {"chop": 6.50, "median": 9.25, "breakout": 13.25},
    "Q3_W3_THU_NY_PM": {"chop": 5.25, "median": 7.50, "breakout": 10.75},
    "Q3_W3_FRI_ASIA": {"chop": 2.00, "median": 2.50, "breakout": 3.50},
    "Q3_W3_FRI_LONDON": {"chop": 3.00, "median": 4.00, "breakout": 5.75},
    "Q3_W3_FRI_NY_AM": {"chop": 6.25, "median": 8.75, "breakout": 12.50},
    "Q3_W3_FRI_NY_PM": {"chop": 5.19, "median": 7.00, "breakout": 9.50},
    "Q3_W4_MON_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.00},
    "Q3_W4_MON_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 5.75},
    "Q3_W4_MON_NY_AM": {"chop": 5.00, "median": 7.50, "breakout": 10.75},
    "Q3_W4_MON_NY_PM": {"chop": 4.50, "median": 6.25, "breakout": 8.75},
    "Q3_W4_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 3.75},
    "Q3_W4_TUE_LONDON": {"chop": 3.25, "median": 4.00, "breakout": 6.00},
    "Q3_W4_TUE_NY_AM": {"chop": 4.94, "median": 7.50, "breakout": 11.00},
    "Q3_W4_TUE_NY_PM": {"chop": 5.00, "median": 7.00, "breakout": 9.75},
    "Q3_W4_WED_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.00},
    "Q3_W4_WED_LONDON": {"chop": 2.75, "median": 3.75, "breakout": 5.25},
    "Q3_W4_WED_NY_AM": {"chop": 5.00, "median": 7.25, "breakout": 10.00},
    "Q3_W4_WED_NY_PM": {"chop": 5.25, "median": 8.00, "breakout": 12.50},
    "Q3_W4_THU_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 4.25},
    "Q3_W4_THU_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.50},
    "Q3_W4_THU_NY_AM": {"chop": 6.50, "median": 9.75, "breakout": 14.50},
    "Q3_W4_THU_NY_PM": {"chop": 5.75, "median": 8.75, "breakout": 13.75},
    "Q3_W4_FRI_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q3_W4_FRI_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 5.75},
    "Q3_W4_FRI_NY_AM": {"chop": 5.50, "median": 8.75, "breakout": 14.50},
    "Q3_W4_FRI_NY_PM": {"chop": 5.50, "median": 8.25, "breakout": 11.75},
    "Q4_W1_MON_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q4_W1_MON_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 6.00},
    "Q4_W1_MON_NY_AM": {"chop": 5.50, "median": 7.75, "breakout": 11.50},
    "Q4_W1_MON_NY_PM": {"chop": 4.50, "median": 6.25, "breakout": 9.25},
    "Q4_W1_TUE_ASIA": {"chop": 2.25, "median": 3.25, "breakout": 6.25},
    "Q4_W1_TUE_LONDON": {"chop": 3.25, "median": 4.25, "breakout": 6.00},
    "Q4_W1_TUE_NY_AM": {"chop": 6.25, "median": 9.00, "breakout": 13.00},
    "Q4_W1_TUE_NY_PM": {"chop": 5.00, "median": 8.00, "breakout": 10.75},
    "Q4_W1_WED_ASIA": {"chop": 2.25, "median": 3.50, "breakout": 5.50},
    "Q4_W1_WED_LONDON": {"chop": 3.75, "median": 5.75, "breakout": 8.50},
    "Q4_W1_WED_NY_AM": {"chop": 6.50, "median": 10.50, "breakout": 14.75},
    "Q4_W1_WED_NY_PM": {"chop": 5.00, "median": 7.00, "breakout": 9.50},
    "Q4_W1_THU_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.75},
    "Q4_W1_THU_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.00},
    "Q4_W1_THU_NY_AM": {"chop": 5.25, "median": 8.25, "breakout": 12.50},
    "Q4_W1_THU_NY_PM": {"chop": 4.75, "median": 6.50, "breakout": 10.00},
    "Q4_W1_FRI_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 4.00},
    "Q4_W1_FRI_LONDON": {"chop": 2.75, "median": 4.00, "breakout": 5.50},
    "Q4_W1_FRI_NY_AM": {"chop": 6.75, "median": 11.75, "breakout": 16.75},
    "Q4_W1_FRI_NY_PM": {"chop": 5.25, "median": 7.75, "breakout": 12.00},
    "Q4_W2_MON_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 5.50},
    "Q4_W2_MON_LONDON": {"chop": 3.19, "median": 4.25, "breakout": 8.50},
    "Q4_W2_MON_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 11.50},
    "Q4_W2_MON_NY_PM": {"chop": 4.00, "median": 6.00, "breakout": 8.50},
    "Q4_W2_TUE_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 5.50},
    "Q4_W2_TUE_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 7.25},
    "Q4_W2_TUE_NY_AM": {"chop": 5.25, "median": 7.50, "breakout": 11.25},
    "Q4_W2_TUE_NY_PM": {"chop": 4.75, "median": 6.50, "breakout": 9.50},
    "Q4_W2_WED_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 3.75},
    "Q4_W2_WED_LONDON": {"chop": 2.50, "median": 3.50, "breakout": 5.00},
    "Q4_W2_WED_NY_AM": {"chop": 4.75, "median": 6.75, "breakout": 9.75},
    "Q4_W2_WED_NY_PM": {"chop": 4.00, "median": 6.00, "breakout": 9.75},
    "Q4_W2_THU_ASIA": {"chop": 2.00, "median": 2.50, "breakout": 3.75},
    "Q4_W2_THU_LONDON": {"chop": 2.75, "median": 3.50, "breakout": 4.75},
    "Q4_W2_THU_NY_AM": {"chop": 5.75, "median": 8.75, "breakout": 12.25},
    "Q4_W2_THU_NY_PM": {"chop": 5.50, "median": 8.00, "breakout": 11.25},
    "Q4_W2_FRI_ASIA": {"chop": 2.00, "median": 2.75, "breakout": 4.50},
    "Q4_W2_FRI_LONDON": {"chop": 3.25, "median": 4.75, "breakout": 6.50},
    "Q4_W2_FRI_NY_AM": {"chop": 5.50, "median": 9.50, "breakout": 13.50},
    "Q4_W2_FRI_NY_PM": {"chop": 5.00, "median": 7.00, "breakout": 10.81},
    "Q4_W3_MON_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.75},
    "Q4_W3_MON_LONDON": {"chop": 3.75, "median": 5.25, "breakout": 7.75},
    "Q4_W3_MON_NY_AM": {"chop": 5.00, "median": 7.50, "breakout": 11.75},
    "Q4_W3_MON_NY_PM": {"chop": 4.00, "median": 5.75, "breakout": 8.50},
    "Q4_W3_TUE_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 4.25},
    "Q4_W3_TUE_LONDON": {"chop": 3.00, "median": 4.00, "breakout": 5.75},
    "Q4_W3_TUE_NY_AM": {"chop": 4.75, "median": 7.50, "breakout": 11.50},
    "Q4_W3_TUE_NY_PM": {"chop": 4.50, "median": 6.25, "breakout": 8.75},
    "Q4_W3_WED_ASIA": {"chop": 2.25, "median": 3.75, "breakout": 6.00},
    "Q4_W3_WED_LONDON": {"chop": 3.25, "median": 4.50, "breakout": 6.00},
    "Q4_W3_WED_NY_AM": {"chop": 5.00, "median": 8.25, "breakout": 12.00},
    "Q4_W3_WED_NY_PM": {"chop": 5.50, "median": 9.00, "breakout": 16.00},
    "Q4_W3_THU_ASIA": {"chop": 2.25, "median": 3.50, "breakout": 6.00},
    "Q4_W3_THU_LONDON": {"chop": 3.50, "median": 5.25, "breakout": 8.00},
    "Q4_W3_THU_NY_AM": {"chop": 8.00, "median": 11.00, "breakout": 17.75},
    "Q4_W3_THU_NY_PM": {"chop": 6.25, "median": 9.50, "breakout": 16.75},
    "Q4_W3_FRI_ASIA": {"chop": 2.25, "median": 3.00, "breakout": 5.25},
    "Q4_W3_FRI_LONDON": {"chop": 3.00, "median": 4.75, "breakout": 8.81},
    "Q4_W3_FRI_NY_AM": {"chop": 6.25, "median": 10.00, "breakout": 16.75},
    "Q4_W3_FRI_NY_PM": {"chop": 5.00, "median": 8.75, "breakout": 13.25},
    "Q4_W4_MON_ASIA": {"chop": 2.00, "median": 3.00, "breakout": 4.25},
    "Q4_W4_MON_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 6.25},
    "Q4_W4_MON_NY_AM": {"chop": 6.00, "median": 9.25, "breakout": 14.50},
    "Q4_W4_MON_NY_PM": {"chop": 4.50, "median": 7.50, "breakout": 10.50},
    "Q4_W4_TUE_ASIA": {"chop": 1.75, "median": 2.75, "breakout": 3.75},
    "Q4_W4_TUE_LONDON": {"chop": 3.00, "median": 4.25, "breakout": 5.75},
    "Q4_W4_TUE_NY_AM": {"chop": 5.25, "median": 7.25, "breakout": 11.00},
    "Q4_W4_TUE_NY_PM": {"chop": 4.50, "median": 6.75, "breakout": 10.25},
    "Q4_W4_WED_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 4.25},
    "Q4_W4_WED_LONDON": {"chop": 2.75, "median": 4.00, "breakout": 5.50},
    "Q4_W4_WED_NY_AM": {"chop": 5.00, "median": 7.50, "breakout": 11.00},
    "Q4_W4_WED_NY_PM": {"chop": 4.75, "median": 7.75, "breakout": 12.00},
    "Q4_W4_THU_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 4.00},
    "Q4_W4_THU_LONDON": {"chop": 2.25, "median": 4.25, "breakout": 6.50},
    "Q4_W4_THU_NY_AM": {"chop": 3.75, "median": 8.00, "breakout": 13.00},
    "Q4_W4_THU_NY_PM": {"chop": 5.25, "median": 8.25, "breakout": 12.00},
    "Q4_W4_FRI_ASIA": {"chop": 1.75, "median": 2.50, "breakout": 3.50},
    "Q4_W4_FRI_LONDON": {"chop": 2.50, "median": 3.75, "breakout": 5.25},
    "Q4_W4_FRI_NY_AM": {"chop": 5.25, "median": 8.00, "breakout": 13.75},
    "Q4_W4_FRI_NY_PM": {"chop": 5.25, "median": 8.25, "breakout": 12.75},
}

# Day of week mapping
DOW_NAMES = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']


class ChopFilter:
    """
    Dynamic chop detection with breakout bias tracking and HH/HL structure validation.
    
    States:
    - IN_CHOP: Range < chop_threshold OR Bunched Swings detected
    - BREAKOUT_LONG: Broke above chop range, waiting for structure confirmation
    - BREAKOUT_SHORT: Broke below chop range, waiting for structure confirmation
    - CONFIRMED_LONG: Breakout + HH structure confirmed, block shorts
    - CONFIRMED_SHORT: Breakout + LL structure confirmed, block longs
    - FAILED_LONG: Broke above but made LH (failed structure), block long continuations
    - FAILED_SHORT: Broke below but made HL (failed structure), block short continuations
    - NORMAL: Range > breakout_threshold, no directional bias
    """
    
    def __init__(self, lookback: int = 20, swing_lookback: int = 5,
                 max_bars_in_chop: int = 20):
        self.lookback = lookback
        self.swing_lookback = swing_lookback  # Bars to confirm swing high/low
        self.max_bars_in_chop = max_bars_in_chop  # ~1 hour on 5m chart; disable fading after this

        # Price tracking
        self.highs = deque(maxlen=lookback)
        self.lows = deque(maxlen=lookback)
        self.closes = deque(maxlen=lookback)

        # Chop range tracking
        self.chop_high = None
        self.chop_low = None

        # Breakout level (the important level that was broken)
        self.breakout_level = None
        self.breakout_direction = None  # 'LONG' or 'SHORT'

        # Swing tracking for structure validation
        self.swing_highs = []  # List of (price, bar_index) tuples
        self.swing_lows = []
        self.bar_count = 0

        # NEW: Time decay tracking - "The longer the base, the higher in space"
        self.bars_in_chop = 0  # How many bars we've been consolidating
        self.last_dt = None  # Prevent double-counting on same bar

        # NEW: Volatility Scalar - Baseline ATR for threshold adjustment
        # Average 5m ATR for MES (2023-2025 Reference Value)
        self.baseline_atr = 4.50

        # State
        self.state = 'NORMAL'
        self.current_threshold = None
        
    def _get_session(self, dt: datetime) -> str:
        """Determine trading session from datetime."""
        hour = dt.hour
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
        """Get yearly quarter from month."""
        if month in [1, 2, 3]:
            return 'Q1'
        elif month in [4, 5, 6]:
            return 'Q2'
        elif month in [7, 8, 9]:
            return 'Q3'
        return 'Q4'
    
    def _get_monthly_quarter(self, day: int) -> str:
        """Get monthly week from day."""
        if day <= 7:
            return 'W1'
        elif day <= 14:
            return 'W2'
        elif day <= 21:
            return 'W3'
        return 'W4'
    
    def _get_threshold_key(self, dt: datetime) -> str:
        """Build threshold key: YearlyQ_MonthlyQ_DayOfWeek_Session"""
        yearly_q = self._get_yearly_quarter(dt.month)
        monthly_q = self._get_monthly_quarter(dt.day)
        dow = DOW_NAMES[dt.weekday()]
        session = self._get_session(dt)
        return f"{yearly_q}_{monthly_q}_{dow}_{session}"
    
    def _get_thresholds(self, dt: datetime) -> dict:
        """Get chop thresholds for current time context."""
        key = self._get_threshold_key(dt)
        
        # Note: Assuming CHOP_THRESHOLDS is defined globally in the file
        if key in CHOP_THRESHOLDS:
            return CHOP_THRESHOLDS[key]
        
        # Fallback: try session-only key with default values
        session = self._get_session(dt)
        fallback_defaults = {
            'ASIA': {"chop": 2.00, "median": 3.00, "breakout": 5.00},
            'LONDON': {"chop": 3.25, "median": 4.50, "breakout": 6.50},
            'NY_AM': {"chop": 5.50, "median": 8.50, "breakout": 13.00},
            'NY_PM': {"chop": 5.00, "median": 7.50, "breakout": 11.00},
        }
        return fallback_defaults.get(session, {"chop": 4.00, "median": 6.00, "breakout": 10.00})
    
    def _detect_swing_high(self) -> float:
        """Detect most recent swing high (higher than neighbors)."""
        if len(self.highs) < self.swing_lookback * 2 + 1:
            return None
        
        highs_list = list(self.highs)
        for i in range(len(highs_list) - self.swing_lookback - 1, self.swing_lookback - 1, -1):
            is_swing = True
            for j in range(1, self.swing_lookback + 1):
                if highs_list[i] <= highs_list[i - j] or highs_list[i] <= highs_list[i + j]:
                    is_swing = False
                    break
            if is_swing:
                return highs_list[i]
        return None
    
    def _detect_swing_low(self) -> float:
        """Detect most recent swing low (lower than neighbors)."""
        if len(self.lows) < self.swing_lookback * 2 + 1:
            return None
        
        lows_list = list(self.lows)
        for i in range(len(lows_list) - self.swing_lookback - 1, self.swing_lookback - 1, -1):
            is_swing = True
            for j in range(1, self.swing_lookback + 1):
                if lows_list[i] >= lows_list[i - j] or lows_list[i] >= lows_list[i + j]:
                    is_swing = False
                    break
            if is_swing:
                return lows_list[i]
        return None
    
    def _update_swings(self, high: float, low: float):
        """Track swing highs and lows for structure validation."""
        self.bar_count += 1
        
        # Detect new swing high
        swing_h = self._detect_swing_high()
        if swing_h and (not self.swing_highs or swing_h != self.swing_highs[-1][0]):
            self.swing_highs.append((swing_h, self.bar_count))
            # Keep only last 5 swings
            if len(self.swing_highs) > 5:
                self.swing_highs.pop(0)
        
        # Detect new swing low
        swing_l = self._detect_swing_low()
        if swing_l and (not self.swing_lows or swing_l != self.swing_lows[-1][0]):
            self.swing_lows.append((swing_l, self.bar_count))
            if len(self.swing_lows) > 5:
                self.swing_lows.pop(0)

    # ---------------------------------------------------------
    # NEW METHOD: Check for Bunched Swings
    # ---------------------------------------------------------
    def _check_bunched_swings(self, threshold: float) -> bool:
        """
        Check if combos of last 3 highs/2 lows or 2 highs/3 lows 
        are bunched together within the chop threshold.
        """
        # Need at least 5 swing points total in history to check this efficiently,
        # but specifically need 3 highs and 2 lows OR 2 highs and 3 lows.
        
        has_3h_2l = len(self.swing_highs) >= 3 and len(self.swing_lows) >= 2
        has_2h_3l = len(self.swing_highs) >= 2 and len(self.swing_lows) >= 3
        
        if not (has_3h_2l or has_2h_3l):
            return False
            
        bunched = False
        
        # Case 1: Last 3 Highs + Last 2 Lows
        if has_3h_2l:
            pts = [x[0] for x in self.swing_highs[-3:]] + [x[0] for x in self.swing_lows[-2:]]
            rng = max(pts) - min(pts)
            if rng <= threshold:
                bunched = True
                
        # Case 2: Last 2 Highs + Last 3 Lows
        if has_2h_3l and not bunched:
            pts = [x[0] for x in self.swing_highs[-2:]] + [x[0] for x in self.swing_lows[-3:]]
            rng = max(pts) - min(pts)
            if rng <= threshold:
                bunched = True
                
        return bunched
    
    def _check_structure_for_long(self, current_close: float) -> str:
        """
        After breakout above important level, check structure.
        """
        if len(self.swing_highs) < 2:
            return 'PENDING'
        
        # Get last two swing highs
        prev_sh = self.swing_highs[-2][0]
        curr_sh = self.swing_highs[-1][0]
        
        # Check if we're still above the breakout level
        if current_close < self.breakout_level:
            # Fell back below - reset
            return 'RESET'
        
        # HH = bullish structure confirmed
        if curr_sh > prev_sh:
            return 'CONFIRMED'
        
        # LH = failed structure (price above level but making lower highs)
        if curr_sh < prev_sh:
            return 'FAILED'
        
        return 'PENDING'
    
    def _check_structure_for_short(self, current_close: float) -> str:
        """
        After breakout below important level, check structure.
        """
        if len(self.swing_lows) < 2:
            return 'PENDING'
        
        # Get last two swing lows
        prev_sl = self.swing_lows[-2][0]
        curr_sl = self.swing_lows[-1][0]
        
        # Check if we're still below the breakout level
        if current_close > self.breakout_level:
            # Rose back above - reset
            return 'RESET'
        
        # LL = bearish structure confirmed
        if curr_sl < prev_sl:
            return 'CONFIRMED'
        
        # HL = failed structure (price below level but making higher lows)
        if curr_sl > prev_sl:
            return 'FAILED'
        
        return 'PENDING'
    
    def update(self, high: float, low: float, close: float, dt: datetime, current_atr: float = None) -> str:
        """
        Update filter with new bar data.
        Returns current state.

        Args:
            high: Bar high price
            low: Bar low price
            close: Bar close price
            dt: Bar datetime
            current_atr: Current ATR value for volatility scaling (optional)
        """
        # Only process once per new bar timestamp
        if self.last_dt is not None and dt <= self.last_dt:
            return self.state
        self.last_dt = dt

        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self._update_swings(high, low)

        if len(self.highs) < self.lookback:
            return 'NORMAL'

        # Calculate current range
        range_high = max(self.highs)
        range_low = min(self.lows)
        current_range = range_high - range_low

        # ==========================================
        # VOLATILITY SCALAR ("Accordion Effect")
        # ==========================================
        # If ATR is missing, default to 1.0 (no change)
        vol_scalar = 1.0
        if current_atr and current_atr > 0:
            # We use sqrt to damp the scalar so it doesn't swing too wildly
            # e.g., ATR doubles (4.5 -> 9.0) => scalar = sqrt(2) ≈ 1.41
            vol_scalar = (current_atr / self.baseline_atr) ** 0.5
            # Clamp to reasonable limits (0.8x to 2.0x)
            vol_scalar = max(0.8, min(vol_scalar, 2.0))

        # Get base thresholds
        base_thresholds = self._get_thresholds(dt)

        # Apply scalar - If volatility is high, we require a wider range to confirm breakout
        chop_thresh = base_thresholds['chop'] * vol_scalar
        breakout_thresh = base_thresholds['breakout'] * vol_scalar

        # Update current_threshold dict for logging/debugging
        self.current_threshold = {
            'chop': chop_thresh,
            'breakout': breakout_thresh,
            'median': base_thresholds.get('median', 0) * vol_scalar,
            'scalar': vol_scalar,
            'current_atr': current_atr
        }
        
        # State machine
        if self.state == 'NORMAL':
            # 1. Original Chop Check (Range based)
            if current_range < chop_thresh:
                self.state = 'IN_CHOP'
                self.chop_high = range_high
                self.chop_low = range_low
                self.bars_in_chop = 1  # NEW: Start counting

            # 2. NEW LOGIC: Bunched Swings Check
            elif self._check_bunched_swings(chop_thresh):
                self.state = 'IN_CHOP'
                self.chop_high = range_high
                self.chop_low = range_low
                self.bars_in_chop = 1  # NEW: Start counting

        elif self.state == 'IN_CHOP':
            # NEW: Increment time-in-chop counter
            self.bars_in_chop += 1

            # Update chop range
            self.chop_high = max(self.chop_high, range_high)
            self.chop_low = min(self.chop_low, range_low)

            # Check for breakout
            if close > self.chop_high:
                self.state = 'BREAKOUT_LONG'
                self.breakout_level = self.chop_high
                self.breakout_direction = 'LONG'
                self.swing_highs.clear()  # Reset for fresh structure tracking
                self.swing_lows.clear()
                self.bars_in_chop = 0  # NEW: Reset counter on breakout
            elif close < self.chop_low:
                self.state = 'BREAKOUT_SHORT'
                self.breakout_level = self.chop_low
                self.breakout_direction = 'SHORT'
                self.swing_highs.clear()
                self.swing_lows.clear()
                self.bars_in_chop = 0  # NEW: Reset counter on breakout
            elif current_range > breakout_thresh:
                # Range expanded without clear breakout
                self.state = 'NORMAL'
                self.chop_high = None
                self.chop_low = None
                self.bars_in_chop = 0  # NEW: Reset counter
                
        elif self.state == 'BREAKOUT_LONG':
            structure = self._check_structure_for_long(close)
            if structure == 'CONFIRMED':
                self.state = 'CONFIRMED_LONG'
            elif structure == 'FAILED':
                self.state = 'FAILED_LONG'
            elif structure == 'RESET':
                self.state = 'IN_CHOP'
            # Check if range normalized
            if current_range > breakout_thresh:
                self.state = 'NORMAL'
                self.breakout_level = None
                
        elif self.state == 'BREAKOUT_SHORT':
            structure = self._check_structure_for_short(close)
            if structure == 'CONFIRMED':
                self.state = 'CONFIRMED_SHORT'
            elif structure == 'FAILED':
                self.state = 'FAILED_SHORT'
            elif structure == 'RESET':
                self.state = 'IN_CHOP'
            if current_range > breakout_thresh:
                self.state = 'NORMAL'
                self.breakout_level = None
                
        elif self.state in ['CONFIRMED_LONG', 'FAILED_LONG']:
            # Check if price fell back through level
            buffer = 1.0
            if close < self.breakout_level - buffer:
                self.state = 'NORMAL'
                self.breakout_level = None
            elif current_range > breakout_thresh:
                self.state = 'NORMAL'
                self.breakout_level = None
                
        elif self.state in ['CONFIRMED_SHORT', 'FAILED_SHORT']:
            buffer = 1.0
            if close > self.breakout_level + buffer:
                self.state = 'NORMAL'
                self.breakout_level = None
            elif current_range > breakout_thresh:
                self.state = 'NORMAL'
                self.breakout_level = None
        
        return self.state
    
    def should_block_trade(
        self,
        direction: str,
        daily_bias: str = None,
        current_price: float = None,
        trend_state: str = "NEUTRAL",
        vol_regime: str = "normal",
    ) -> tuple:
        """
        Check if trade should be blocked based on chop state.

        UPDATED LOGIC: "Fade the Range"
        - Context-aware buy/sell zones that expand/contract with trend & volatility.
        - If IN_CHOP:
          - LONG zone starts at 25%, expands to 50% when strongly bullish, and widens in low vol.
          - SHORT zone starts at 75%, drifts to 50% when strongly bearish, and widens in low vol.
          - High vol tightens zones (accordion effect) to demand more extreme fades.
        """
        direction = direction.upper()

        # Use last known close if current_price is not provided by the strategy
        if current_price is None and len(self.closes) > 0:
            current_price = self.closes[-1]

        # 1. Handle Chop State (Range Fading Logic)
        if self.state == 'IN_CHOP':
            # Safety check for missing data
            if current_price is None or self.chop_high is None or self.chop_low is None:
                return True, "IN_CHOP: Market consolidating"

            # NEW: Time Decay - "The longer the base, the higher in space"
            # If consolidation has lasted too long, a breakout is more likely.
            # DISABLE fading logic and block ALL trades (wait for breakout confirmation)
            if self.bars_in_chop > self.max_bars_in_chop:
                return True, (f"IN_CHOP (STALE): Consolidation lasted {self.bars_in_chop} bars "
                              f"(>{self.max_bars_in_chop}). Fading disabled - wait for breakout")

            # Calculate range height
            chop_range = self.chop_high - self.chop_low

            # If range is extremely tight (< 1 point), don't try to fade it (too risky)
            min_chop_range = 1.0
            if chop_range < min_chop_range:
                return True, f"IN_CHOP: Range too tight to fade ({chop_range:.2f} pts)"

            # Calculate where we are in the range (0.0 = Low, 1.0 = High)
            # We clip values > 1.0 or < 0.0 in case price is slightly piercing edges
            position_in_range = (current_price - self.chop_low) / chop_range

            # --- DYNAMIC ZONES BASED ON TREND ---
            long_zone_limit = 0.25
            short_zone_limit = 0.75

            trend_state = trend_state or "NEUTRAL"
            vol_regime = vol_regime or "normal"
            if "Strong Bullish" in trend_state:
                long_zone_limit = 0.50
            if "Strong Bearish" in trend_state:
                short_zone_limit = 0.50

            # --- VOLATILITY-SCALED ZONES (Accordion Effect) ---
            if vol_regime in ("low", "ultra_low"):
                long_zone_limit += 0.10
                short_zone_limit -= 0.10
            elif vol_regime == "high":
                long_zone_limit -= 0.10
                short_zone_limit += 0.10

            # Clamp to sensible bounds
            long_zone_limit = max(0.0, min(long_zone_limit, 1.0))
            short_zone_limit = max(0.0, min(short_zone_limit, 1.0))

            # --- LONG LOGIC: Buy Support ---
            if direction == 'LONG':
                # Allow buying only in the bottom 25% of the chop box
                if position_in_range <= long_zone_limit:
                    return False, None  # ALLOW: Buying the bottom of the range
                else:
                    return True, ("IN_CHOP: Blocked Long at "
                                  f"{position_in_range:.0%} of range (Buy Zone < {long_zone_limit:.0%})")

            # --- SHORT LOGIC: Sell Resistance ---
            if direction == 'SHORT':
                # Allow selling only in the top 25% of the chop box
                if position_in_range >= short_zone_limit:
                    return False, None  # ALLOW: Selling the top of the range
                else:
                    return True, ("IN_CHOP: Blocked Short at "
                                  f"{position_in_range:.0%} of range (Sell Zone > {short_zone_limit:.0%})")

            return True, "IN_CHOP: Market consolidating"

        # 2. Handle Breakout States (Existing Logic)
        if self.state == 'BREAKOUT_LONG' and direction == 'SHORT':
            return True, "BREAKOUT_LONG: Pending bullish confirmation"
        if self.state == 'BREAKOUT_SHORT' and direction == 'LONG':
            return True, "BREAKOUT_SHORT: Pending bearish confirmation"

        if self.state == 'CONFIRMED_LONG' and direction == 'SHORT':
            return True, "CONFIRMED_LONG: HH structure, blocking shorts"
        if self.state == 'CONFIRMED_SHORT' and direction == 'LONG':
            return True, "CONFIRMED_SHORT: LL structure, blocking longs"

        # 3. Handle Failed Breakouts (Existing Logic)
        if self.state == 'FAILED_LONG':
            if direction == 'SHORT':
                return False, "OPPORTUNITY: Fading the Failed Long Breakout"
            if direction == 'LONG':
                if daily_bias and daily_bias.upper() == 'LONG':
                    return True, "FAILED_LONG: LH after breakout, blocking continuation"
                return True, "FAILED_LONG: Made LH after breakout"

        if self.state == 'FAILED_SHORT':
            if direction == 'LONG':
                return False, "OPPORTUNITY: Fading the Failed Short Breakout"
            if direction == 'SHORT':
                if daily_bias and daily_bias.upper() == 'SHORT':
                    return True, "FAILED_SHORT: HL after breakout, blocking continuation"
                return True, "FAILED_SHORT: Made HL after breakout"

        return False, None
    
    def get_status(self) -> dict:
        """Get current filter status for logging."""
        # Extract scalar from threshold dict if available
        vol_scalar = self.current_threshold.get('scalar', 1.0) if self.current_threshold else 1.0
        current_atr = self.current_threshold.get('current_atr') if self.current_threshold else None

        return {
            'state': self.state,
            'chop_high': self.chop_high,
            'chop_low': self.chop_low,
            'breakout_level': self.breakout_level,
            'breakout_direction': self.breakout_direction,
            'threshold': self.current_threshold,
            'swing_highs': [s[0] for s in self.swing_highs[-3:]] if self.swing_highs else [],
            'swing_lows': [s[0] for s in self.swing_lows[-3:]] if self.swing_lows else [],
            'bars_in_chop': self.bars_in_chop,
            'max_bars_in_chop': self.max_bars_in_chop,
            'vol_scalar': vol_scalar,  # NEW: Volatility scalar for debugging
            'current_atr': current_atr,
            'baseline_atr': self.baseline_atr,
        }

    def get_state(self) -> dict:
        return {
            "state": self.state,
            "bars_in_chop": self.bars_in_chop,
            "bar_count": self.bar_count,
            "chop_high": self.chop_high,
            "chop_low": self.chop_low,
            "breakout_level": self.breakout_level,
            "breakout_direction": self.breakout_direction,
            "swing_highs": self.swing_highs,
            "swing_lows": self.swing_lows,
            "highs": list(self.highs),
            "lows": list(self.lows),
            "closes": list(self.closes),
            "last_dt": self.last_dt.isoformat() if self.last_dt else None,
            "current_threshold": self.current_threshold,
        }

    def load_state(self, state: dict) -> None:
        if not state:
            return
        self.state = state.get("state", self.state)
        self.bars_in_chop = int(state.get("bars_in_chop", self.bars_in_chop))
        self.bar_count = int(state.get("bar_count", self.bar_count))
        self.chop_high = state.get("chop_high", self.chop_high)
        self.chop_low = state.get("chop_low", self.chop_low)
        self.breakout_level = state.get("breakout_level", self.breakout_level)
        self.breakout_direction = state.get("breakout_direction", self.breakout_direction)
        self.swing_highs = state.get("swing_highs", self.swing_highs) or []
        self.swing_lows = state.get("swing_lows", self.swing_lows) or []
        self.highs = deque(state.get("highs", []), maxlen=self.lookback)
        self.lows = deque(state.get("lows", []), maxlen=self.lookback)
        self.closes = deque(state.get("closes", []), maxlen=self.lookback)
        last_dt = state.get("last_dt")
        if last_dt:
            try:
                self.last_dt = datetime.fromisoformat(last_dt)
            except Exception:
                pass
        self.current_threshold = state.get("current_threshold", self.current_threshold)
    
    def reset(self):
        """Reset filter state (e.g., at session change)."""
        self.highs.clear()
        self.lows.clear()
        self.closes.clear()
        self.swing_highs.clear()
        self.swing_lows.clear()
        self.chop_high = None
        self.chop_low = None
        self.breakout_level = None
        self.breakout_direction = None
        self.state = 'NORMAL'
        self.bar_count = 0
        self.bars_in_chop = 0  # NEW: Reset time decay counter
        self.last_dt = None
