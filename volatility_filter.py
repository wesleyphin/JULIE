"""
JULIE001 Hierarchical Volatility Filter
========================================
Dynamic volatility filtering with 320 time-hierarchy combinations:
- Yearly Quarter: Q1-Q4
- Monthly Quarter: W1-W4
- Day of Week: MON-FRI
- Session: ASIA, LONDON, NY_AM, NY_PM

Thresholds derived from 2023-2025 MES futures data (958,390 active bars).
"""

import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, Tuple, Optional
from zoneinfo import ZoneInfo
from config import CONFIG

# ============================================================
# HIERARCHICAL VOLATILITY THRESHOLDS (320 combinations)
# ============================================================
VOLATILITY_HIERARCHY = {
    "Q1_W1_MON_ASIA": {"p10": 5.54e-05, "p25": 7.14e-05, "median": 9.78e-05, "p75": 0.0001426},
    "Q1_W1_MON_LONDON": {"p10": 7.93e-05, "p25": 0.0001068, "median": 0.0001514, "p75": 0.0002117},
    "Q1_W1_MON_NY_AM": {"p10": 0.0001298, "p25": 0.0001764, "median": 0.0002485, "p75": 0.0004271},
    "Q1_W1_MON_NY_PM": {"p10": 0.0001034, "p25": 0.0001806, "median": 0.0002583, "p75": 0.0003599},
    "Q1_W1_TUE_ASIA": {"p10": 5.49e-05, "p25": 6.92e-05, "median": 0.0001007, "p75": 0.0001473},
    "Q1_W1_TUE_LONDON": {"p10": 9.52e-05, "p25": 0.000113, "median": 0.0001501, "p75": 0.0002349},
    "Q1_W1_TUE_NY_AM": {"p10": 0.0001342, "p25": 0.000194, "median": 0.0002956, "p75": 0.0004527},
    "Q1_W1_TUE_NY_PM": {"p10": 0.0001371, "p25": 0.000187, "median": 0.0002667, "p75": 0.0004474},
    "Q1_W1_WED_ASIA": {"p10": 5.48e-05, "p25": 7.41e-05, "median": 0.0001062, "p75": 0.0001482},
    "Q1_W1_WED_LONDON": {"p10": 8.37e-05, "p25": 0.0001061, "median": 0.0001429, "p75": 0.0002006},
    "Q1_W1_WED_NY_AM": {"p10": 0.0001419, "p25": 0.0001964, "median": 0.0002786, "p75": 0.0004439},
    "Q1_W1_WED_NY_PM": {"p10": 0.000125, "p25": 0.0001806, "median": 0.0002645, "p75": 0.0004835},
    "Q1_W1_THU_ASIA": {"p10": 5.5e-05, "p25": 7.24e-05, "median": 0.0001013, "p75": 0.0001406},
    "Q1_W1_THU_LONDON": {"p10": 0.0001016, "p25": 0.0001225, "median": 0.0001719, "p75": 0.0002334},
    "Q1_W1_THU_NY_AM": {"p10": 0.0001742, "p25": 0.0002315, "median": 0.0003241, "p75": 0.0004581},
    "Q1_W1_THU_NY_PM": {"p10": 0.0001579, "p25": 0.0002045, "median": 0.0002933, "p75": 0.0004556},
    "Q1_W1_FRI_ASIA": {"p10": 5.93e-05, "p25": 7.46e-05, "median": 9.81e-05, "p75": 0.0001257},
    "Q1_W1_FRI_LONDON": {"p10": 8.72e-05, "p25": 0.0001048, "median": 0.0001424, "p75": 0.0001883},
    "Q1_W1_FRI_NY_AM": {"p10": 0.0001558, "p25": 0.0002318, "median": 0.0003777, "p75": 0.0005778},
    "Q1_W1_FRI_NY_PM": {"p10": 0.0001107, "p25": 0.0001775, "median": 0.0002504, "p75": 0.0003556},
    "Q1_W2_MON_ASIA": {"p10": 5.37e-05, "p25": 7.54e-05, "median": 0.0001158, "p75": 0.0002859},
    "Q1_W2_MON_LONDON": {"p10": 7.91e-05, "p25": 0.0001322, "median": 0.0001929, "p75": 0.0003703},
    "Q1_W2_MON_NY_AM": {"p10": 0.000118, "p25": 0.0001783, "median": 0.0002869, "p75": 0.0004926},
    "Q1_W2_MON_NY_PM": {"p10": 0.0001192, "p25": 0.0001696, "median": 0.0002549, "p75": 0.0004298},
    "Q1_W2_TUE_ASIA": {"p10": 5.48e-05, "p25": 6.79e-05, "median": 0.0001, "p75": 0.0001572},
    "Q1_W2_TUE_LONDON": {"p10": 9.97e-05, "p25": 0.0001193, "median": 0.0001594, "p75": 0.0002564},
    "Q1_W2_TUE_NY_AM": {"p10": 0.0001488, "p25": 0.0002358, "median": 0.0003717, "p75": 0.0005806},
    "Q1_W2_TUE_NY_PM": {"p10": 0.0001273, "p25": 0.0001983, "median": 0.000281, "p75": 0.0004652},
    "Q1_W2_WED_ASIA": {"p10": 5.23e-05, "p25": 6.57e-05, "median": 8.92e-05, "p75": 0.0001192},
    "Q1_W2_WED_LONDON": {"p10": 8.4e-05, "p25": 0.0001056, "median": 0.0001386, "p75": 0.000198},
    "Q1_W2_WED_NY_AM": {"p10": 0.0001481, "p25": 0.0002022, "median": 0.0003036, "p75": 0.0004223},
    "Q1_W2_WED_NY_PM": {"p10": 0.0001109, "p25": 0.0001726, "median": 0.0002477, "p75": 0.0003555},
    "Q1_W2_THU_ASIA": {"p10": 5.36e-05, "p25": 7.18e-05, "median": 0.0001015, "p75": 0.0001613},
    "Q1_W2_THU_LONDON": {"p10": 7.87e-05, "p25": 9.87e-05, "median": 0.0001374, "p75": 0.0001786},
    "Q1_W2_THU_NY_AM": {"p10": 0.0001306, "p25": 0.0002021, "median": 0.0003473, "p75": 0.0004745},
    "Q1_W2_THU_NY_PM": {"p10": 0.000119, "p25": 0.000174, "median": 0.000291, "p75": 0.0004076},
    "Q1_W2_FRI_ASIA": {"p10": 4.65e-05, "p25": 7.39e-05, "median": 0.0001029, "p75": 0.0001584},
    "Q1_W2_FRI_LONDON": {"p10": 8.91e-05, "p25": 0.000115, "median": 0.0001708, "p75": 0.0002315},
    "Q1_W2_FRI_NY_AM": {"p10": 0.0001636, "p25": 0.0002204, "median": 0.0003604, "p75": 0.0005456},
    "Q1_W2_FRI_NY_PM": {"p10": 0.0001109, "p25": 0.0001516, "median": 0.0002759, "p75": 0.0003817},
    "Q1_W3_MON_ASIA": {"p10": 5.26e-05, "p25": 7.11e-05, "median": 0.0001018, "p75": 0.0001494},
    "Q1_W3_MON_LONDON": {"p10": 6.51e-05, "p25": 9.08e-05, "median": 0.0001293, "p75": 0.0001874},
    "Q1_W3_MON_NY_AM": {"p10": 5.84e-05, "p25": 7.97e-05, "median": 0.0001211, "p75": 0.0002989},
    "Q1_W3_MON_NY_PM": {"p10": 5.66e-05, "p25": 0.0001221, "median": 0.0002224, "p75": 0.0003881},
    "Q1_W3_TUE_ASIA": {"p10": 5.01e-05, "p25": 6.68e-05, "median": 9.25e-05, "p75": 0.0001303},
    "Q1_W3_TUE_LONDON": {"p10": 9.65e-05, "p25": 0.00013, "median": 0.0001696, "p75": 0.0002256},
    "Q1_W3_TUE_NY_AM": {"p10": 0.0001454, "p25": 0.0001887, "median": 0.0002582, "p75": 0.0003643},
    "Q1_W3_TUE_NY_PM": {"p10": 9.26e-05, "p25": 0.0001599, "median": 0.0002298, "p75": 0.000291},
    "Q1_W3_WED_ASIA": {"p10": 5.08e-05, "p25": 6.55e-05, "median": 9.16e-05, "p75": 0.0001354},
    "Q1_W3_WED_LONDON": {"p10": 7.75e-05, "p25": 0.0001041, "median": 0.0001392, "p75": 0.0001908},
    "Q1_W3_WED_NY_AM": {"p10": 0.0001283, "p25": 0.0001796, "median": 0.0002598, "p75": 0.0004502},
    "Q1_W3_WED_NY_PM": {"p10": 0.0001075, "p25": 0.0001618, "median": 0.0002428, "p75": 0.000366},
    "Q1_W3_THU_ASIA": {"p10": 5.27e-05, "p25": 6.92e-05, "median": 9.11e-05, "p75": 0.0001219},
    "Q1_W3_THU_LONDON": {"p10": 7.89e-05, "p25": 0.0001055, "median": 0.000138, "p75": 0.0002073},
    "Q1_W3_THU_NY_AM": {"p10": 0.0001188, "p25": 0.000206, "median": 0.0003095, "p75": 0.000431},
    "Q1_W3_THU_NY_PM": {"p10": 0.0001184, "p25": 0.0001732, "median": 0.0002427, "p75": 0.0003425},
    "Q1_W3_FRI_ASIA": {"p10": 5.3e-05, "p25": 6.41e-05, "median": 9.29e-05, "p75": 0.0001179},
    "Q1_W3_FRI_LONDON": {"p10": 7.16e-05, "p25": 9.43e-05, "median": 0.000141, "p75": 0.0002041},
    "Q1_W3_FRI_NY_AM": {"p10": 0.0001481, "p25": 0.0002285, "median": 0.0003049, "p75": 0.000415},
    "Q1_W3_FRI_NY_PM": {"p10": 0.0001177, "p25": 0.0001838, "median": 0.0002696, "p75": 0.00036},
    "Q1_W4_MON_ASIA": {"p10": 5.49e-05, "p25": 6.8e-05, "median": 9.18e-05, "p75": 0.0001345},
    "Q1_W4_MON_LONDON": {"p10": 8.29e-05, "p25": 0.0001045, "median": 0.0001474, "p75": 0.0002299},
    "Q1_W4_MON_NY_AM": {"p10": 0.0001129, "p25": 0.000167, "median": 0.0002424, "p75": 0.0003909},
    "Q1_W4_MON_NY_PM": {"p10": 0.0001013, "p25": 0.0001512, "median": 0.0002349, "p75": 0.0003358},
    "Q1_W4_TUE_ASIA": {"p10": 4.91e-05, "p25": 6.01e-05, "median": 7.9e-05, "p75": 0.000111},
    "Q1_W4_TUE_LONDON": {"p10": 7.97e-05, "p25": 9.36e-05, "median": 0.00014, "p75": 0.0001885},
    "Q1_W4_TUE_NY_AM": {"p10": 0.000113, "p25": 0.0001518, "median": 0.0002108, "p75": 0.000344},
    "Q1_W4_TUE_NY_PM": {"p10": 9.74e-05, "p25": 0.0001389, "median": 0.000194, "p75": 0.0002663},
    "Q1_W4_WED_ASIA": {"p10": 5.2e-05, "p25": 6.5e-05, "median": 8.3e-05, "p75": 0.0001114},
    "Q1_W4_WED_LONDON": {"p10": 7.98e-05, "p25": 9.25e-05, "median": 0.0001185, "p75": 0.0001564},
    "Q1_W4_WED_NY_AM": {"p10": 0.0001152, "p25": 0.0001564, "median": 0.0002144, "p75": 0.0002959},
    "Q1_W4_WED_NY_PM": {"p10": 0.0001123, "p25": 0.0001634, "median": 0.0002587, "p75": 0.0003829},
    "Q1_W4_THU_ASIA": {"p10": 5.45e-05, "p25": 7.02e-05, "median": 9.12e-05, "p75": 0.0001308},
    "Q1_W4_THU_LONDON": {"p10": 8.7e-05, "p25": 0.0001075, "median": 0.0001338, "p75": 0.0001778},
    "Q1_W4_THU_NY_AM": {"p10": 0.0001373, "p25": 0.0001765, "median": 0.0002752, "p75": 0.0004088},
    "Q1_W4_THU_NY_PM": {"p10": 0.0001102, "p25": 0.0001463, "median": 0.0002388, "p75": 0.0003491},
    "Q1_W4_FRI_ASIA": {"p10": 5.14e-05, "p25": 6.41e-05, "median": 8.64e-05, "p75": 0.0001157},
    "Q1_W4_FRI_LONDON": {"p10": 8.36e-05, "p25": 9.92e-05, "median": 0.0001264, "p75": 0.0001682},
    "Q1_W4_FRI_NY_AM": {"p10": 0.0001322, "p25": 0.0001804, "median": 0.0002584, "p75": 0.0004415},
    "Q1_W4_FRI_NY_PM": {"p10": 0.0001124, "p25": 0.0001675, "median": 0.000245, "p75": 0.00037},
    "Q2_W1_MON_ASIA": {"p10": 5.46e-05, "p25": 6.78e-05, "median": 8.76e-05, "p75": 0.000134},
    "Q2_W1_MON_LONDON": {"p10": 6.38e-05, "p25": 8.5e-05, "median": 0.0001289, "p75": 0.0001933},
    "Q2_W1_MON_NY_AM": {"p10": 0.0001216, "p25": 0.0001574, "median": 0.000243, "p75": 0.0003711},
    "Q2_W1_MON_NY_PM": {"p10": 0.0001118, "p25": 0.00015, "median": 0.0002186, "p75": 0.0002914},
    "Q2_W1_TUE_ASIA": {"p10": 5.34e-05, "p25": 6.5e-05, "median": 8.57e-05, "p75": 0.0001163},
    "Q2_W1_TUE_LONDON": {"p10": 9.6e-05, "p25": 0.0001168, "median": 0.0001442, "p75": 0.000188},
    "Q2_W1_TUE_NY_AM": {"p10": 0.0001193, "p25": 0.0001491, "median": 0.0002351, "p75": 0.0003514},
    "Q2_W1_TUE_NY_PM": {"p10": 0.0001101, "p25": 0.0001711, "median": 0.000236, "p75": 0.0003221},
    "Q2_W1_WED_ASIA": {"p10": 5.76e-05, "p25": 7.09e-05, "median": 9.39e-05, "p75": 0.0001371},
    "Q2_W1_WED_LONDON": {"p10": 9.47e-05, "p25": 0.0001117, "median": 0.0001356, "p75": 0.0001706},
    "Q2_W1_WED_NY_AM": {"p10": 0.0001318, "p25": 0.0001829, "median": 0.0002625, "p75": 0.0003541},
    "Q2_W1_WED_NY_PM": {"p10": 0.0001168, "p25": 0.0001639, "median": 0.0002288, "p75": 0.0004076},
    "Q2_W1_THU_ASIA": {"p10": 5.29e-05, "p25": 6.72e-05, "median": 9.79e-05, "p75": 0.0001594},
    "Q2_W1_THU_LONDON": {"p10": 8.02e-05, "p25": 0.000108, "median": 0.0001471, "p75": 0.0002093},
    "Q2_W1_THU_NY_AM": {"p10": 0.0001472, "p25": 0.0001904, "median": 0.000273, "p75": 0.0004296},
    "Q2_W1_THU_NY_PM": {"p10": 0.0001443, "p25": 0.0001994, "median": 0.0002804, "p75": 0.0004073},
    "Q2_W1_FRI_ASIA": {"p10": 5.14e-05, "p25": 6.55e-05, "median": 8.84e-05, "p75": 0.000133},
    "Q2_W1_FRI_LONDON": {"p10": 7.43e-05, "p25": 9.55e-05, "median": 0.0001273, "p75": 0.0001849},
    "Q2_W1_FRI_NY_AM": {"p10": 0.0001611, "p25": 0.0002594, "median": 0.0003421, "p75": 0.0005319},
    "Q2_W1_FRI_NY_PM": {"p10": 0.0001257, "p25": 0.0001866, "median": 0.000237, "p75": 0.0003223},
    "Q2_W2_MON_ASIA": {"p10": 4.89e-05, "p25": 5.85e-05, "median": 8.02e-05, "p75": 0.000139},
    "Q2_W2_MON_LONDON": {"p10": 6.65e-05, "p25": 8.45e-05, "median": 0.0001379, "p75": 0.000259},
    "Q2_W2_MON_NY_AM": {"p10": 0.0001171, "p25": 0.0001471, "median": 0.0002101, "p75": 0.0003278},
    "Q2_W2_MON_NY_PM": {"p10": 9.39e-05, "p25": 0.0001329, "median": 0.0001833, "p75": 0.0002641},
    "Q2_W2_TUE_ASIA": {"p10": 4.79e-05, "p25": 5.71e-05, "median": 7.99e-05, "p75": 0.0001448},
    "Q2_W2_TUE_LONDON": {"p10": 7.4e-05, "p25": 0.0001018, "median": 0.0001289, "p75": 0.0001863},
    "Q2_W2_TUE_NY_AM": {"p10": 0.0001192, "p25": 0.0001652, "median": 0.0002349, "p75": 0.000352},
    "Q2_W2_TUE_NY_PM": {"p10": 0.0001141, "p25": 0.0001485, "median": 0.0002051, "p75": 0.0002731},
    "Q2_W2_WED_ASIA": {"p10": 5.36e-05, "p25": 6.72e-05, "median": 9.15e-05, "p75": 0.0001559},
    "Q2_W2_WED_LONDON": {"p10": 7.52e-05, "p25": 9.47e-05, "median": 0.0001168, "p75": 0.0001572},
    "Q2_W2_WED_NY_AM": {"p10": 0.0001468, "p25": 0.0002024, "median": 0.0002793, "p75": 0.0005272},
    "Q2_W2_WED_NY_PM": {"p10": 0.0001208, "p25": 0.0001799, "median": 0.000286, "p75": 0.0004596},
    "Q2_W2_THU_ASIA": {"p10": 5.25e-05, "p25": 6.77e-05, "median": 0.00011, "p75": 0.0003522},
    "Q2_W2_THU_LONDON": {"p10": 9.12e-05, "p25": 0.0001088, "median": 0.0001401, "p75": 0.0001949},
    "Q2_W2_THU_NY_AM": {"p10": 0.0001569, "p25": 0.0002061, "median": 0.0002928, "p75": 0.0004394},
    "Q2_W2_THU_NY_PM": {"p10": 0.0001038, "p25": 0.0001566, "median": 0.0002105, "p75": 0.0002877},
    "Q2_W2_FRI_ASIA": {"p10": 4.74e-05, "p25": 6.42e-05, "median": 0.0001166, "p75": 0.0003883},
    "Q2_W2_FRI_LONDON": {"p10": 8.74e-05, "p25": 0.0001113, "median": 0.0001534, "p75": 0.0002391},
    "Q2_W2_FRI_NY_AM": {"p10": 0.00014, "p25": 0.0002004, "median": 0.0002992, "p75": 0.0004165},
    "Q2_W2_FRI_NY_PM": {"p10": 0.0001162, "p25": 0.000191, "median": 0.0002635, "p75": 0.0003595},
    "Q2_W3_MON_ASIA": {"p10": 5.31e-05, "p25": 7.28e-05, "median": 0.0001091, "p75": 0.0001962},
    "Q2_W3_MON_LONDON": {"p10": 7.45e-05, "p25": 0.000103, "median": 0.0001557, "p75": 0.00025},
    "Q2_W3_MON_NY_AM": {"p10": 9.35e-05, "p25": 0.0001492, "median": 0.0002312, "p75": 0.0003854},
    "Q2_W3_MON_NY_PM": {"p10": 0.000103, "p25": 0.0001556, "median": 0.0002133, "p75": 0.0003494},
    "Q2_W3_TUE_ASIA": {"p10": 4.76e-05, "p25": 6.62e-05, "median": 0.0001014, "p75": 0.0001828},
    "Q2_W3_TUE_LONDON": {"p10": 7.18e-05, "p25": 0.0001008, "median": 0.0001464, "p75": 0.0002346},
    "Q2_W3_TUE_NY_AM": {"p10": 0.0001119, "p25": 0.0001464, "median": 0.0002408, "p75": 0.0003492},
    "Q2_W3_TUE_NY_PM": {"p10": 0.0001079, "p25": 0.0001545, "median": 0.000214, "p75": 0.0003094},
    "Q2_W3_WED_ASIA": {"p10": 5.19e-05, "p25": 6.81e-05, "median": 9.7e-05, "p75": 0.000156},
    "Q2_W3_WED_LONDON": {"p10": 7.12e-05, "p25": 9.67e-05, "median": 0.0001476, "p75": 0.0002037},
    "Q2_W3_WED_NY_AM": {"p10": 0.0001044, "p25": 0.0001646, "median": 0.0002556, "p75": 0.0003534},
    "Q2_W3_WED_NY_PM": {"p10": 0.0001175, "p25": 0.0001715, "median": 0.0002541, "p75": 0.000444},
    "Q2_W3_THU_ASIA": {"p10": 5.36e-05, "p25": 6.72e-05, "median": 8.89e-05, "p75": 0.0001277},
    "Q2_W3_THU_LONDON": {"p10": 7.73e-05, "p25": 0.0001007, "median": 0.0001437, "p75": 0.0001886},
    "Q2_W3_THU_NY_AM": {"p10": 0.0001338, "p25": 0.000178, "median": 0.000251, "p75": 0.0003506},
    "Q2_W3_THU_NY_PM": {"p10": 0.0001289, "p25": 0.0001874, "median": 0.0002681, "p75": 0.0003794},
    "Q2_W3_FRI_ASIA": {"p10": 4.69e-05, "p25": 6.35e-05, "median": 8.19e-05, "p75": 0.0001221},
    "Q2_W3_FRI_LONDON": {"p10": 8.99e-05, "p25": 0.0001148, "median": 0.0001373, "p75": 0.0001725},
    "Q2_W3_FRI_NY_AM": {"p10": 0.0001261, "p25": 0.0001684, "median": 0.0002722, "p75": 0.0003739},
    "Q2_W3_FRI_NY_PM": {"p10": 0.0001172, "p25": 0.0001646, "median": 0.0002146, "p75": 0.0002794},
    "Q2_W4_MON_ASIA": {"p10": 5.27e-05, "p25": 6.45e-05, "median": 8.33e-05, "p75": 0.0001165},
    "Q2_W4_MON_LONDON": {"p10": 7.44e-05, "p25": 9.35e-05, "median": 0.0001216, "p75": 0.0001628},
    "Q2_W4_MON_NY_AM": {"p10": 7.6e-05, "p25": 0.0001206, "median": 0.0001746, "p75": 0.0002705},
    "Q2_W4_MON_NY_PM": {"p10": 8.76e-05, "p25": 0.0001478, "median": 0.0002042, "p75": 0.0002814},
    "Q2_W4_TUE_ASIA": {"p10": 5.28e-05, "p25": 6.93e-05, "median": 9.01e-05, "p75": 0.0001242},
    "Q2_W4_TUE_LONDON": {"p10": 9.45e-05, "p25": 0.0001134, "median": 0.000142, "p75": 0.0001864},
    "Q2_W4_TUE_NY_AM": {"p10": 0.0001357, "p25": 0.0001643, "median": 0.0002343, "p75": 0.0003291},
    "Q2_W4_TUE_NY_PM": {"p10": 0.0001241, "p25": 0.0001674, "median": 0.0002268, "p75": 0.0003127},
    "Q2_W4_WED_ASIA": {"p10": 5.11e-05, "p25": 6.72e-05, "median": 9.7e-05, "p75": 0.0001474},
    "Q2_W4_WED_LONDON": {"p10": 8.22e-05, "p25": 0.0001047, "median": 0.0001379, "p75": 0.0001838},
    "Q2_W4_WED_NY_AM": {"p10": 0.0001111, "p25": 0.0001556, "median": 0.0002356, "p75": 0.0003447},
    "Q2_W4_WED_NY_PM": {"p10": 0.0001272, "p25": 0.0001766, "median": 0.0002564, "p75": 0.0003817},
    "Q2_W4_THU_ASIA": {"p10": 5.58e-05, "p25": 6.95e-05, "median": 9.37e-05, "p75": 0.0001333},
    "Q2_W4_THU_LONDON": {"p10": 8.05e-05, "p25": 9.79e-05, "median": 0.0001302, "p75": 0.0001861},
    "Q2_W4_THU_NY_AM": {"p10": 0.0001312, "p25": 0.0001958, "median": 0.000291, "p75": 0.0003927},
    "Q2_W4_THU_NY_PM": {"p10": 0.0001181, "p25": 0.000161, "median": 0.0002295, "p75": 0.0003182},
    "Q2_W4_FRI_ASIA": {"p10": 5.32e-05, "p25": 6.45e-05, "median": 9.13e-05, "p75": 0.000124},
    "Q2_W4_FRI_LONDON": {"p10": 7.55e-05, "p25": 9.64e-05, "median": 0.0001296, "p75": 0.0001738},
    "Q2_W4_FRI_NY_AM": {"p10": 0.0001596, "p25": 0.0002129, "median": 0.0003082, "p75": 0.0004063},
    "Q2_W4_FRI_NY_PM": {"p10": 0.0001268, "p25": 0.000172, "median": 0.0002297, "p75": 0.0003037},
    "Q3_W1_MON_ASIA": {"p10": 5.34e-05, "p25": 6.64e-05, "median": 8.66e-05, "p75": 0.0001233},
    "Q3_W1_MON_LONDON": {"p10": 6.78e-05, "p25": 8.49e-05, "median": 0.0001133, "p75": 0.0001588},
    "Q3_W1_MON_NY_AM": {"p10": 7.01e-05, "p25": 9.66e-05, "median": 0.0001516, "p75": 0.0002841},
    "Q3_W1_MON_NY_PM": {"p10": 6.72e-05, "p25": 0.0001279, "median": 0.0001976, "p75": 0.0002725},
    "Q3_W1_TUE_ASIA": {"p10": 4.94e-05, "p25": 6.16e-05, "median": 8.39e-05, "p75": 0.0001265},
    "Q3_W1_TUE_LONDON": {"p10": 7.14e-05, "p25": 9.56e-05, "median": 0.0001289, "p75": 0.0001767},
    "Q3_W1_TUE_NY_AM": {"p10": 7.06e-05, "p25": 0.0001533, "median": 0.0002407, "p75": 0.0003697},
    "Q3_W1_TUE_NY_PM": {"p10": 0.0001084, "p25": 0.0001499, "median": 0.0002011, "p75": 0.0002717},
    "Q3_W1_WED_ASIA": {"p10": 5.09e-05, "p25": 6.52e-05, "median": 9.06e-05, "p75": 0.0001324},
    "Q3_W1_WED_LONDON": {"p10": 7.69e-05, "p25": 0.0001012, "median": 0.0001315, "p75": 0.0001777},
    "Q3_W1_WED_NY_AM": {"p10": 0.0001193, "p25": 0.0001599, "median": 0.0002228, "p75": 0.0003116},
    "Q3_W1_WED_NY_PM": {"p10": 9.26e-05, "p25": 0.0001297, "median": 0.0002122, "p75": 0.0002854},
    "Q3_W1_THU_ASIA": {"p10": 5.02e-05, "p25": 6.4e-05, "median": 9.01e-05, "p75": 0.0001383},
    "Q3_W1_THU_LONDON": {"p10": 6.63e-05, "p25": 9.98e-05, "median": 0.0001322, "p75": 0.000173},
    "Q3_W1_THU_NY_AM": {"p10": 8.2e-05, "p25": 0.0001532, "median": 0.0002468, "p75": 0.0003567},
    "Q3_W1_THU_NY_PM": {"p10": 0.0001033, "p25": 0.0001526, "median": 0.0002214, "p75": 0.0003299},
    "Q3_W1_FRI_ASIA": {"p10": 4.39e-05, "p25": 6.2e-05, "median": 8.83e-05, "p75": 0.0001281},
    "Q3_W1_FRI_LONDON": {"p10": 7.62e-05, "p25": 0.0001007, "median": 0.0001387, "p75": 0.0001962},
    "Q3_W1_FRI_NY_AM": {"p10": 0.000111, "p25": 0.0001816, "median": 0.0003378, "p75": 0.0005369},
    "Q3_W1_FRI_NY_PM": {"p10": 9.69e-05, "p25": 0.0001512, "median": 0.0002408, "p75": 0.0003682},
    "Q3_W2_MON_ASIA": {"p10": 5.27e-05, "p25": 6.56e-05, "median": 8.87e-05, "p75": 0.0001339},
    "Q3_W2_MON_LONDON": {"p10": 6.66e-05, "p25": 8.63e-05, "median": 0.0001309, "p75": 0.0001719},
    "Q3_W2_MON_NY_AM": {"p10": 0.0001077, "p25": 0.0001403, "median": 0.0002127, "p75": 0.0003189},
    "Q3_W2_MON_NY_PM": {"p10": 7.98e-05, "p25": 0.0001314, "median": 0.0001918, "p75": 0.0002574},
    "Q3_W2_TUE_ASIA": {"p10": 4.68e-05, "p25": 5.96e-05, "median": 7.8e-05, "p75": 0.0001084},
    "Q3_W2_TUE_LONDON": {"p10": 6.76e-05, "p25": 9.15e-05, "median": 0.0001237, "p75": 0.0001659},
    "Q3_W2_TUE_NY_AM": {"p10": 0.0001044, "p25": 0.0001454, "median": 0.0002232, "p75": 0.0002958},
    "Q3_W2_TUE_NY_PM": {"p10": 8.58e-05, "p25": 0.0001224, "median": 0.0001722, "p75": 0.00023},
    "Q3_W2_WED_ASIA": {"p10": 4.83e-05, "p25": 5.94e-05, "median": 7.43e-05, "p75": 9.66e-05},
    "Q3_W2_WED_LONDON": {"p10": 6.25e-05, "p25": 8.21e-05, "median": 0.0001056, "p75": 0.0001352},
    "Q3_W2_WED_NY_AM": {"p10": 9.49e-05, "p25": 0.0001427, "median": 0.0002317, "p75": 0.0003325},
    "Q3_W2_WED_NY_PM": {"p10": 8.83e-05, "p25": 0.0001396, "median": 0.0001891, "p75": 0.0002559},
    "Q3_W2_THU_ASIA": {"p10": 4.96e-05, "p25": 6.12e-05, "median": 7.88e-05, "p75": 0.0001088},
    "Q3_W2_THU_LONDON": {"p10": 5.38e-05, "p25": 6.89e-05, "median": 0.0001037, "p75": 0.000143},
    "Q3_W2_THU_NY_AM": {"p10": 0.0001226, "p25": 0.0001803, "median": 0.0002703, "p75": 0.0004038},
    "Q3_W2_THU_NY_PM": {"p10": 9.38e-05, "p25": 0.00013, "median": 0.0001877, "p75": 0.0003056},
    "Q3_W2_FRI_ASIA": {"p10": 5.26e-05, "p25": 6.57e-05, "median": 9.61e-05, "p75": 0.0001401},
    "Q3_W2_FRI_LONDON": {"p10": 6.95e-05, "p25": 8.85e-05, "median": 0.0001269, "p75": 0.0001803},
    "Q3_W2_FRI_NY_AM": {"p10": 0.0001162, "p25": 0.0001545, "median": 0.0002215, "p75": 0.000319},
    "Q3_W2_FRI_NY_PM": {"p10": 8.62e-05, "p25": 0.0001345, "median": 0.0001934, "p75": 0.0002655},
    "Q3_W3_MON_ASIA": {"p10": 4.47e-05, "p25": 5.58e-05, "median": 7.42e-05, "p75": 0.0001351},
    "Q3_W3_MON_LONDON": {"p10": 6.89e-05, "p25": 8.82e-05, "median": 0.0001185, "p75": 0.0001925},
    "Q3_W3_MON_NY_AM": {"p10": 0.0001014, "p25": 0.0001326, "median": 0.0002058, "p75": 0.00032},
    "Q3_W3_MON_NY_PM": {"p10": 8.23e-05, "p25": 0.0001205, "median": 0.000159, "p75": 0.0002429},
    "Q3_W3_TUE_ASIA": {"p10": 4.79e-05, "p25": 5.99e-05, "median": 7.82e-05, "p75": 0.0001083},
    "Q3_W3_TUE_LONDON": {"p10": 6.56e-05, "p25": 8.16e-05, "median": 0.0001055, "p75": 0.0001646},
    "Q3_W3_TUE_NY_AM": {"p10": 0.0001011, "p25": 0.0001499, "median": 0.000215, "p75": 0.0002874},
    "Q3_W3_TUE_NY_PM": {"p10": 8.54e-05, "p25": 0.0001405, "median": 0.0001909, "p75": 0.0002415},
    "Q3_W3_WED_ASIA": {"p10": 5.4e-05, "p25": 6.58e-05, "median": 8.27e-05, "p75": 0.0001162},
    "Q3_W3_WED_LONDON": {"p10": 7.39e-05, "p25": 8.71e-05, "median": 0.0001132, "p75": 0.0001533},
    "Q3_W3_WED_NY_AM": {"p10": 9.66e-05, "p25": 0.0001319, "median": 0.0002166, "p75": 0.0003124},
    "Q3_W3_WED_NY_PM": {"p10": 0.0001039, "p25": 0.0001568, "median": 0.0002303, "p75": 0.0003343},
    "Q3_W3_THU_ASIA": {"p10": 5.46e-05, "p25": 6.62e-05, "median": 8.66e-05, "p75": 0.0001107},
    "Q3_W3_THU_LONDON": {"p10": 8.07e-05, "p25": 0.0001031, "median": 0.0001332, "p75": 0.0001673},
    "Q3_W3_THU_NY_AM": {"p10": 0.0001221, "p25": 0.0001595, "median": 0.0002379, "p75": 0.0003324},
    "Q3_W3_THU_NY_PM": {"p10": 0.0001061, "p25": 0.0001452, "median": 0.0002139, "p75": 0.0002793},
    "Q3_W3_FRI_ASIA": {"p10": 4.71e-05, "p25": 5.88e-05, "median": 7.34e-05, "p75": 9.72e-05},
    "Q3_W3_FRI_LONDON": {"p10": 7.69e-05, "p25": 9.08e-05, "median": 0.000112, "p75": 0.0001541},
    "Q3_W3_FRI_NY_AM": {"p10": 0.0001193, "p25": 0.0001743, "median": 0.0002439, "p75": 0.0003244},
    "Q3_W3_FRI_NY_PM": {"p10": 0.0001001, "p25": 0.000134, "median": 0.0001968, "p75": 0.0002588},
    "Q3_W4_MON_ASIA": {"p10": 4.64e-05, "p25": 5.7e-05, "median": 7.5e-05, "p75": 0.0001017},
    "Q3_W4_MON_LONDON": {"p10": 7.17e-05, "p25": 8.74e-05, "median": 0.0001125, "p75": 0.0001503},
    "Q3_W4_MON_NY_AM": {"p10": 0.0001012, "p25": 0.000126, "median": 0.000199, "p75": 0.000297},
    "Q3_W4_MON_NY_PM": {"p10": 8.41e-05, "p25": 0.0001171, "median": 0.0001716, "p75": 0.0002434},
    "Q3_W4_TUE_ASIA": {"p10": 4.63e-05, "p25": 5.9e-05, "median": 7.5e-05, "p75": 0.0001002},
    "Q3_W4_TUE_LONDON": {"p10": 6.81e-05, "p25": 8.44e-05, "median": 0.0001122, "p75": 0.0001487},
    "Q3_W4_TUE_NY_AM": {"p10": 9.28e-05, "p25": 0.0001298, "median": 0.0001956, "p75": 0.0002904},
    "Q3_W4_TUE_NY_PM": {"p10": 9.69e-05, "p25": 0.0001346, "median": 0.0001865, "p75": 0.0002559},
    "Q3_W4_WED_ASIA": {"p10": 4.71e-05, "p25": 5.99e-05, "median": 7.96e-05, "p75": 0.0001106},
    "Q3_W4_WED_LONDON": {"p10": 6.22e-05, "p25": 7.96e-05, "median": 0.0001081, "p75": 0.0001379},
    "Q3_W4_WED_NY_AM": {"p10": 0.0001047, "p25": 0.0001408, "median": 0.0001972, "p75": 0.0002753},
    "Q3_W4_WED_NY_PM": {"p10": 0.0001024, "p25": 0.0001451, "median": 0.0002169, "p75": 0.000328},
    "Q3_W4_THU_ASIA": {"p10": 5.28e-05, "p25": 6.56e-05, "median": 8.42e-05, "p75": 0.0001093},
    "Q3_W4_THU_LONDON": {"p10": 7.48e-05, "p25": 9.38e-05, "median": 0.0001232, "p75": 0.0001687},
    "Q3_W4_THU_NY_AM": {"p10": 0.0001259, "p25": 0.0001802, "median": 0.0002706, "p75": 0.0003769},
    "Q3_W4_THU_NY_PM": {"p10": 0.0001016, "p25": 0.0001559, "median": 0.0002502, "p75": 0.0003537},
    "Q3_W4_FRI_ASIA": {"p10": 4.92e-05, "p25": 6.15e-05, "median": 8.33e-05, "p75": 0.0001233},
    "Q3_W4_FRI_LONDON": {"p10": 7.78e-05, "p25": 9.74e-05, "median": 0.0001186, "p75": 0.0001514},
    "Q3_W4_FRI_NY_AM": {"p10": 0.0001076, "p25": 0.0001525, "median": 0.0002617, "p75": 0.0003925},
    "Q3_W4_FRI_NY_PM": {"p10": 8.37e-05, "p25": 0.0001445, "median": 0.0002178, "p75": 0.0003183},
    "Q4_W1_MON_ASIA": {"p10": 5.41e-05, "p25": 6.73e-05, "median": 8.81e-05, "p75": 0.000121},
    "Q4_W1_MON_LONDON": {"p10": 7.65e-05, "p25": 9.84e-05, "median": 0.0001192, "p75": 0.0001561},
    "Q4_W1_MON_NY_AM": {"p10": 0.0001056, "p25": 0.0001378, "median": 0.0002112, "p75": 0.0003027},
    "Q4_W1_MON_NY_PM": {"p10": 8.68e-05, "p25": 0.0001196, "median": 0.0001703, "p75": 0.0002284},
    "Q4_W1_TUE_ASIA": {"p10": 4.85e-05, "p25": 6.27e-05, "median": 8.86e-05, "p75": 0.0001629},
    "Q4_W1_TUE_LONDON": {"p10": 7.22e-05, "p25": 9.1e-05, "median": 0.0001317, "p75": 0.0001728},
    "Q4_W1_TUE_NY_AM": {"p10": 9.93e-05, "p25": 0.0001647, "median": 0.000241, "p75": 0.0003459},
    "Q4_W1_TUE_NY_PM": {"p10": 9.18e-05, "p25": 0.0001425, "median": 0.0002215, "p75": 0.0002974},
    "Q4_W1_WED_ASIA": {"p10": 5.44e-05, "p25": 6.87e-05, "median": 9.42e-05, "p75": 0.0001482},
    "Q4_W1_WED_LONDON": {"p10": 8.14e-05, "p25": 0.0001153, "median": 0.0001649, "p75": 0.0002143},
    "Q4_W1_WED_NY_AM": {"p10": 0.0001336, "p25": 0.0001826, "median": 0.0002703, "p75": 0.0003865},
    "Q4_W1_WED_NY_PM": {"p10": 9.22e-05, "p25": 0.0001255, "median": 0.0001703, "p75": 0.0002529},
    "Q4_W1_THU_ASIA": {"p10": 5.13e-05, "p25": 6.47e-05, "median": 8.65e-05, "p75": 0.0001197},
    "Q4_W1_THU_LONDON": {"p10": 6.25e-05, "p25": 9.2e-05, "median": 0.000127, "p75": 0.0001599},
    "Q4_W1_THU_NY_AM": {"p10": 9.22e-05, "p25": 0.0001472, "median": 0.0002296, "p75": 0.0003598},
    "Q4_W1_THU_NY_PM": {"p10": 0.0001044, "p25": 0.0001411, "median": 0.0002013, "p75": 0.0002849},
    "Q4_W1_FRI_ASIA": {"p10": 4.29e-05, "p25": 5.61e-05, "median": 7.78e-05, "p75": 0.0001051},
    "Q4_W1_FRI_LONDON": {"p10": 6.32e-05, "p25": 8.65e-05, "median": 0.0001183, "p75": 0.0001641},
    "Q4_W1_FRI_NY_AM": {"p10": 0.0001289, "p25": 0.0001752, "median": 0.0002873, "p75": 0.0004416},
    "Q4_W1_FRI_NY_PM": {"p10": 8.85e-05, "p25": 0.0001526, "median": 0.0002335, "p75": 0.0003011},
    "Q4_W2_MON_ASIA": {"p10": 4.49e-05, "p25": 5.93e-05, "median": 7.99e-05, "p75": 0.0001279},
    "Q4_W2_MON_LONDON": {"p10": 6.35e-05, "p25": 8.64e-05, "median": 0.0001235, "p75": 0.0002138},
    "Q4_W2_MON_NY_AM": {"p10": 9.21e-05, "p25": 0.000158, "median": 0.0002319, "p75": 0.0003033},
    "Q4_W2_MON_NY_PM": {"p10": 8.4e-05, "p25": 0.0001189, "median": 0.0001576, "p75": 0.0002335},
    "Q4_W2_TUE_ASIA": {"p10": 4.82e-05, "p25": 6.02e-05, "median": 8.05e-05, "p75": 0.0001332},
    "Q4_W2_TUE_LONDON": {"p10": 7e-05, "p25": 9.74e-05, "median": 0.0001307, "p75": 0.0002014},
    "Q4_W2_TUE_NY_AM": {"p10": 0.0001064, "p25": 0.0001606, "median": 0.0002236, "p75": 0.0003091},
    "Q4_W2_TUE_NY_PM": {"p10": 8.35e-05, "p25": 0.0001278, "median": 0.0001758, "p75": 0.0002477},
    "Q4_W2_WED_ASIA": {"p10": 4.71e-05, "p25": 5.87e-05, "median": 7.15e-05, "p75": 9.34e-05},
    "Q4_W2_WED_LONDON": {"p10": 6.32e-05, "p25": 7.78e-05, "median": 9.84e-05, "p75": 0.0001337},
    "Q4_W2_WED_NY_AM": {"p10": 9.97e-05, "p25": 0.0001276, "median": 0.0001779, "p75": 0.0002637},
    "Q4_W2_WED_NY_PM": {"p10": 8.14e-05, "p25": 0.0001024, "median": 0.00016, "p75": 0.0002462},
    "Q4_W2_THU_ASIA": {"p10": 4.79e-05, "p25": 5.96e-05, "median": 7.27e-05, "p75": 9.66e-05},
    "Q4_W2_THU_LONDON": {"p10": 6.68e-05, "p25": 8.48e-05, "median": 0.0001044, "p75": 0.0001307},
    "Q4_W2_THU_NY_AM": {"p10": 0.0001136, "p25": 0.0001708, "median": 0.0002331, "p75": 0.0003319},
    "Q4_W2_THU_NY_PM": {"p10": 0.0001142, "p25": 0.0001581, "median": 0.0002053, "p75": 0.0003111},
    "Q4_W2_FRI_ASIA": {"p10": 4.7e-05, "p25": 5.67e-05, "median": 7.55e-05, "p75": 0.0001118},
    "Q4_W2_FRI_LONDON": {"p10": 6.88e-05, "p25": 9.5e-05, "median": 0.0001298, "p75": 0.0001825},
    "Q4_W2_FRI_NY_AM": {"p10": 0.0001051, "p25": 0.0001498, "median": 0.0002581, "p75": 0.0003651},
    "Q4_W2_FRI_NY_PM": {"p10": 0.000102, "p25": 0.0001339, "median": 0.000181, "p75": 0.0002937},
    "Q4_W3_MON_ASIA": {"p10": 4.83e-05, "p25": 5.91e-05, "median": 7.69e-05, "p75": 0.0001098},
    "Q4_W3_MON_LONDON": {"p10": 8.7e-05, "p25": 0.0001045, "median": 0.0001385, "p75": 0.0002063},
    "Q4_W3_MON_NY_AM": {"p10": 0.0001127, "p25": 0.0001487, "median": 0.0001912, "p75": 0.0002773},
    "Q4_W3_MON_NY_PM": {"p10": 8.14e-05, "p25": 0.0001109, "median": 0.0001487, "p75": 0.0002362},
    "Q4_W3_TUE_ASIA": {"p10": 4.63e-05, "p25": 5.83e-05, "median": 7.45e-05, "p75": 9.95e-05},
    "Q4_W3_TUE_LONDON": {"p10": 7.49e-05, "p25": 9.16e-05, "median": 0.0001132, "p75": 0.0001663},
    "Q4_W3_TUE_NY_AM": {"p10": 0.0001046, "p25": 0.0001413, "median": 0.0002008, "p75": 0.0002919},
    "Q4_W3_TUE_NY_PM": {"p10": 9.97e-05, "p25": 0.0001275, "median": 0.0001637, "p75": 0.0002448},
    "Q4_W3_WED_ASIA": {"p10": 5.68e-05, "p25": 7.3e-05, "median": 0.0001021, "p75": 0.0001442},
    "Q4_W3_WED_LONDON": {"p10": 8.06e-05, "p25": 0.0001039, "median": 0.0001332, "p75": 0.0001686},
    "Q4_W3_WED_NY_AM": {"p10": 0.0001061, "p25": 0.000135, "median": 0.0002056, "p75": 0.0003183},
    "Q4_W3_WED_NY_PM": {"p10": 0.000117, "p25": 0.0001485, "median": 0.0002289, "p75": 0.000404},
    "Q4_W3_THU_ASIA": {"p10": 5.74e-05, "p25": 7.07e-05, "median": 0.0001033, "p75": 0.0001524},
    "Q4_W3_THU_LONDON": {"p10": 8.9e-05, "p25": 0.0001084, "median": 0.0001441, "p75": 0.0002065},
    "Q4_W3_THU_NY_AM": {"p10": 0.0001578, "p25": 0.0002167, "median": 0.0003327, "p75": 0.0004638},
    "Q4_W3_THU_NY_PM": {"p10": 0.0001162, "p25": 0.0001899, "median": 0.0002632, "p75": 0.0004502},
    "Q4_W3_FRI_ASIA": {"p10": 5.66e-05, "p25": 6.68e-05, "median": 8.89e-05, "p75": 0.0001317},
    "Q4_W3_FRI_LONDON": {"p10": 7.44e-05, "p25": 9.19e-05, "median": 0.0001395, "p75": 0.0002374},
    "Q4_W3_FRI_NY_AM": {"p10": 0.0001336, "p25": 0.000197, "median": 0.0002954, "p75": 0.0004184},
    "Q4_W3_FRI_NY_PM": {"p10": 0.0001077, "p25": 0.0001539, "median": 0.0002321, "p75": 0.0003646},
    "Q4_W4_MON_ASIA": {"p10": 5.3e-05, "p25": 6.64e-05, "median": 8.54e-05, "p75": 0.0001074},
    "Q4_W4_MON_LONDON": {"p10": 6.93e-05, "p25": 8.76e-05, "median": 0.0001192, "p75": 0.0001721},
    "Q4_W4_MON_NY_AM": {"p10": 0.0001102, "p25": 0.0001691, "median": 0.000256, "p75": 0.0003946},
    "Q4_W4_MON_NY_PM": {"p10": 7.97e-05, "p25": 0.0001268, "median": 0.0002016, "p75": 0.0003},
    "Q4_W4_TUE_ASIA": {"p10": 4.85e-05, "p25": 5.89e-05, "median": 7.38e-05, "p75": 9.97e-05},
    "Q4_W4_TUE_LONDON": {"p10": 6.41e-05, "p25": 8.9e-05, "median": 0.0001214, "p75": 0.000154},
    "Q4_W4_TUE_NY_AM": {"p10": 0.0001094, "p25": 0.0001444, "median": 0.0002003, "p75": 0.0003182},
    "Q4_W4_TUE_NY_PM": {"p10": 9.2e-05, "p25": 0.0001213, "median": 0.0001764, "p75": 0.0002845},
    "Q4_W4_WED_ASIA": {"p10": 4.42e-05, "p25": 5.56e-05, "median": 7.32e-05, "p75": 0.0001055},
    "Q4_W4_WED_LONDON": {"p10": 7.29e-05, "p25": 8.58e-05, "median": 0.0001033, "p75": 0.000132},
    "Q4_W4_WED_NY_AM": {"p10": 0.0001009, "p25": 0.0001393, "median": 0.0002015, "p75": 0.0002753},
    "Q4_W4_WED_NY_PM": {"p10": 0.0001078, "p25": 0.0001474, "median": 0.0002056, "p75": 0.0003054},
    "Q4_W4_THU_ASIA": {"p10": 4.51e-05, "p25": 5.54e-05, "median": 7.16e-05, "p75": 0.0001027},
    "Q4_W4_THU_LONDON": {"p10": 6.08e-05, "p25": 8.02e-05, "median": 0.0001154, "p75": 0.0001584},
    "Q4_W4_THU_NY_AM": {"p10": 5.61e-05, "p25": 0.0001031, "median": 0.0002132, "p75": 0.0003135},
    "Q4_W4_THU_NY_PM": {"p10": 8.58e-05, "p25": 0.0001408, "median": 0.000212, "p75": 0.0002743},
    "Q4_W4_FRI_ASIA": {"p10": 4.52e-05, "p25": 5.43e-05, "median": 6.9e-05, "p75": 8.94e-05},
    "Q4_W4_FRI_LONDON": {"p10": 6.76e-05, "p25": 7.96e-05, "median": 9.62e-05, "p75": 0.0001273},
    "Q4_W4_FRI_NY_AM": {"p10": 9.67e-05, "p25": 0.0001298, "median": 0.0002078, "p75": 0.000341},
    "Q4_W4_FRI_NY_PM": {"p10": 9.48e-05, "p25": 0.0001427, "median": 0.0002145, "p75": 0.0003247},
}


def _build_coarse_map(include_quarter: bool) -> Dict[str, Dict[str, float]]:
    buckets = defaultdict(lambda: {"p10": [], "p25": [], "median": [], "p75": []})
    for key, values in VOLATILITY_HIERARCHY.items():
        parts = key.split("_")
        if len(parts) < 4:
            continue
        quarter, _, dow, session = parts[0], parts[1], parts[2], parts[3]
        if include_quarter:
            coarse_key = f"{quarter}_{dow}_{session}"
        else:
            coarse_key = f"{dow}_{session}"
        bucket = buckets[coarse_key]
        for metric in ("p10", "p25", "median", "p75"):
            bucket[metric].append(float(values.get(metric, 0.0)))

    coarse = {}
    for key, metrics in buckets.items():
        if not metrics["p10"]:
            continue
        coarse[key] = {
            "p10": float(np.median(metrics["p10"])),
            "p25": float(np.median(metrics["p25"])),
            "median": float(np.median(metrics["median"])),
            "p75": float(np.median(metrics["p75"])),
        }
    return coarse


COARSE_VOLATILITY_HIERARCHY = _build_coarse_map(include_quarter=True)
COARSE_VOLATILITY_HIERARCHY_NO_Q = _build_coarse_map(include_quarter=False)

# Session-level fallbacks
SESSION_FALLBACKS = {
    "ASIA": {"p10": 0.0000506, "p25": 0.0000640, "median": 0.0000867, "p75": 0.0001263},
    "LONDON": {"p10": 0.0000747, "p25": 0.0000959, "median": 0.0001299, "p75": 0.0001807},
    "NY_AM": {"p10": 0.0001121, "p25": 0.0001619, "median": 0.0002458, "p75": 0.0003690},
    "NY_PM": {"p10": 0.0001009, "p25": 0.0001487, "median": 0.0002186, "p75": 0.0003197},
}


class VolRegime:
    ULTRA_LOW = "ultra_low"   # Below p10 - SKIP trading
    LOW = "low"               # p10-p25 - Widen stops, reduce size
    NORMAL = "normal"         # p25-p75 - Normal trading
    HIGH = "high"             # Above p75 - Tighten stops


class HierarchicalVolatilityFilter:
    """
    Dynamic volatility filter using hierarchical time-based thresholds.
    
    Looks up thresholds for specific: YearlyQ_MonthlyQ_DayOfWeek_Session
    Falls back to session defaults if specific combo not found.
    """
    
    def __init__(self, 
                 skip_ultra_low: bool = True,
                 adjust_low_vol: bool = True,
                 low_vol_stop_mult: float = 1.5,
                 low_vol_size_mult: float = 0.67,
                 std_window: int = 20):
        self.skip_ultra_low = skip_ultra_low
        self.adjust_low_vol = adjust_low_vol
        self.low_vol_stop_mult = low_vol_stop_mult
        self.low_vol_size_mult = low_vol_size_mult
        self.std_window = std_window
        self.et = ZoneInfo('America/New_York')
        
        # Cache
        self._last_std = None
        self._last_regime = None
        self._last_key = None
        self._calibrated_thresholds = {}
        self._calibrated_sessions = set()
    
    @staticmethod
    def get_yearly_quarter(month: int) -> str:
        return ['Q1','Q1','Q1','Q2','Q2','Q2','Q3','Q3','Q3','Q4','Q4','Q4'][month-1]
    
    @staticmethod
    def get_monthly_quarter(day: int) -> str:
        if day <= 7: return 'W1'
        elif day <= 14: return 'W2'
        elif day <= 21: return 'W3'
        return 'W4'
    
    @staticmethod
    def get_dow_name(dow: int) -> str:
        return ['MON','TUE','WED','THU','FRI','SAT','SUN'][dow]
    
    @staticmethod
    def get_session(hour: int) -> str:
        if 18 <= hour or hour < 3: return 'ASIA'
        elif 3 <= hour < 8: return 'LONDON'
        elif 8 <= hour < 12: return 'NY_AM'
        elif 12 <= hour < 17: return 'NY_PM'
        return 'CLOSED'

    @staticmethod
    def _hierarchy_mode() -> Tuple[str, bool]:
        cfg = CONFIG.get("VOLATILITY_HIERARCHY_MODE", {}) or {}
        mode = str(cfg.get("mode", "full") or "full").lower()
        include_quarter = bool(cfg.get("include_quarter", True))
        return mode, include_quarter

    def _std_window_for_session(self, session: str) -> int:
        cfg = CONFIG.get("VOLATILITY_STD_WINDOWS", {}) or {}
        default = int(cfg.get("default", self.std_window) or self.std_window)
        sessions = cfg.get("sessions", {}) or {}
        try:
            window = int(sessions.get(session, default))
        except Exception:
            window = default
        return window if window > 0 else default

    def _std_scale(self, df: pd.DataFrame, session: str) -> float:
        cfg = CONFIG.get("VOLATILITY_STD_WINDOW_SCALING", {}) or {}
        if not cfg.get("enabled", True):
            return 1.0
        default_window = int(CONFIG.get("VOLATILITY_STD_WINDOWS", {}).get("default", self.std_window) or self.std_window)
        session_window = self._std_window_for_session(session)
        if session_window == default_window:
            return 1.0
        lookback = int(cfg.get("lookback", 200) or 200)
        try:
            min_scale = float(cfg.get("min", 0.5))
            max_scale = float(cfg.get("max", 2.0))
        except Exception:
            min_scale, max_scale = 0.5, 2.0
        if len(df) < max(default_window, session_window) + 2:
            return 1.0
        returns = df["close"].pct_change()
        std_default = returns.rolling(default_window).std()
        std_session = returns.rolling(session_window).std()
        ratio = (std_session / std_default).replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.empty:
            return 1.0
        recent = ratio.iloc[-lookback:] if len(ratio) > lookback else ratio
        try:
            scale = float(np.median(recent.to_numpy()))
        except Exception:
            scale = 1.0
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        scale = max(min_scale, min(max_scale, scale))
        return scale
    
    def get_hierarchy_key(self, ts) -> str:
        yq = self.get_yearly_quarter(ts.month)
        mq = self.get_monthly_quarter(ts.day)
        dow = self.get_dow_name(ts.weekday())
        sess = self.get_session(ts.hour)
        mode, include_quarter = self._hierarchy_mode()
        if mode == "coarse":
            if include_quarter:
                return f"{yq}_{dow}_{sess}"
            return f"{dow}_{sess}"
        return f"{yq}_{mq}_{dow}_{sess}"
    
    def get_thresholds(self, ts) -> Tuple[Dict, str]:
        """Get volatility thresholds for current time hierarchy."""
        key = self.get_hierarchy_key(ts)
        session = self.get_session(ts.hour)
        mode, include_quarter = self._hierarchy_mode()

        if key in self._calibrated_thresholds:
            return self._calibrated_thresholds[key], f"CAL_{key}"

        # Try hierarchy first
        if mode == "coarse":
            hierarchy = COARSE_VOLATILITY_HIERARCHY if include_quarter else COARSE_VOLATILITY_HIERARCHY_NO_Q
            if key in hierarchy:
                return hierarchy[key], key
        else:
            if key in VOLATILITY_HIERARCHY:
                return VOLATILITY_HIERARCHY[key], key
        
        # Fallback to session
        if session in SESSION_FALLBACKS:
            return SESSION_FALLBACKS[session], f"FALLBACK_{session}"
        
        # Ultimate fallback
        return {"p10": 0.00008, "p25": 0.00012, "median": 0.00018, "p75": 0.00028}, "FALLBACK"
    
    def calculate_volatility(self, df: pd.DataFrame, ts=None, session: Optional[str] = None) -> float:
        """Calculate current 20-period standard deviation of returns."""
        if session is None:
            if ts is None:
                ts = df.index[-1]
            session = self.get_session(ts.hour)
        window = self._std_window_for_session(session)
        if len(df) < window + 1:
            return 0.0
        returns = df['close'].pct_change()
        std = returns.rolling(window).std().iloc[-1]
        return std if not np.isnan(std) else 0.0
    
    def get_regime(self, df: pd.DataFrame, ts=None) -> Tuple[str, float, str]:
        """
        Determine current volatility regime using hierarchical thresholds.
        
        Returns: (regime, current_std, hierarchy_key)
        """
        if ts is None:
            ts = df.index[-1]
        session = self.get_session(ts.hour)
        thresholds, key = self.get_thresholds(ts)
        scale = 1.0
        if not str(key).startswith("CAL_"):
            scale = self._std_scale(df, session)
        if scale != 1.0:
            thresholds = {
                "p10": thresholds["p10"] * scale,
                "p25": thresholds["p25"] * scale,
                "median": thresholds["median"] * scale,
                "p75": thresholds["p75"] * scale,
            }
        current_std = self.calculate_volatility(df, ts=ts, session=session)
        
        # Cache
        self._last_std = current_std
        self._last_key = key
        
        # Determine regime
        if current_std < thresholds['p10']:
            regime = VolRegime.ULTRA_LOW
        elif current_std < thresholds['p25']:
            regime = VolRegime.LOW
        elif current_std > thresholds['p75']:
            regime = VolRegime.HIGH
        else:
            regime = VolRegime.NORMAL
        
        self._last_regime = regime
        return regime, current_std, key
    
    def should_skip_trade(self, df: pd.DataFrame, ts=None) -> Tuple[bool, str]:
        """Check if trade should be skipped due to ultra-low volatility."""
        regime, current_std, key = self.get_regime(df, ts)
        
        if self.skip_ultra_low and regime == VolRegime.ULTRA_LOW:
            thresholds, _ = self.get_thresholds(ts if ts else df.index[-1])
            reason = f"Ultra-low vol [{key}] ({current_std:.7f} < p10={thresholds['p10']:.7f})"
            return True, reason
        
        return False, ""
    
    def get_adjustments(self, df: pd.DataFrame,
                       base_sl: float,
                       base_tp: float,
                       base_size: int = 1,
                       ts=None) -> Dict:
        """Get adjusted SL/TP/size based on volatility regime with SAFETY GUARDRAILS."""
        if ts is None:
            ts = df.index[-1]

        regime, current_std, key = self.get_regime(df, ts)
        thresholds, _ = self.get_thresholds(ts)

        # Calculate Base Risk:Reward
        base_rr = base_tp / base_sl if base_sl > 0 else 0.0

        adj_sl = base_sl
        adj_tp = base_tp
        adj_size = base_size
        adjustment_applied = False

        # --- DYNAMIC MULTIPLIER CALCULATION ---
        # Instead of fixed numbers, we normalize to the 'Median' (Normal) volatility.
        # This allows expansion > 2x if volatility is extremely compressed.
        # Guard against divide-by-zero with a small epsilon.
        safe_std = max(current_std, 1e-9)
        dynamic_mult = thresholds['median'] / safe_std

        # Cap the multiplier to prevent insanity (e.g. 100x) on bad data
        dynamic_mult = min(dynamic_mult, 10.0)

        if regime == VolRegime.ULTRA_LOW:
            if self.adjust_low_vol:
                # Rule: Expand SL dynamically to survive noise
                # If current_std is 1/5th of median, SL becomes 5x (Normalizing risk)
                adj_sl = base_sl * dynamic_mult

                # GUARDRAIL: Do not expand TP if RR is already High (> 3.0)
                if base_rr > 3.0:
                    adj_tp = base_tp  # Keep original
                else:
                    # FIX: Low/Med RR gets FULL DYNAMIC expansion (Uncapped)
                    # Can exceed 2.0x if volatility is tiny
                    adj_tp = base_tp * dynamic_mult

                adj_size = max(1, int(base_size * 0.5))
                adjustment_applied = True

        elif regime == VolRegime.LOW:
            if self.adjust_low_vol:
                # Rule: Widen SL dynamically (Usually 1.3x - 1.8x)
                # We use the calculated dynamic_mult instead of fixed 1.5
                adj_sl = base_sl * dynamic_mult

                # GUARDRAIL: Only cap High RR (> 3.0)
                if base_rr >= 3.0:
                    tp_mult = 1.0
                else:
                    # FIX: Low/Med RR gets FULL DYNAMIC multiplier
                    tp_mult = dynamic_mult

                adj_tp = base_tp * tp_mult
                adj_size = max(1, int(base_size * self.low_vol_size_mult))
                adjustment_applied = True
        elif regime == VolRegime.NORMAL:
            normal_cfg = CONFIG.get("VOLATILITY_NORMAL_ADJUSTMENTS", {}) or {}
            if normal_cfg.get("enabled"):
                session = self.get_session(ts.hour) if ts is not None else self.get_session(df.index[-1].hour)
                default_cfg = normal_cfg.get("default", {}) or {}
                session_cfg = (normal_cfg.get("sessions", {}) or {}).get(session, {}) or {}
                try:
                    sl_mult = float(session_cfg.get("sl_mult", default_cfg.get("sl_mult", 1.0)))
                except Exception:
                    sl_mult = 1.0
                try:
                    tp_mult = float(session_cfg.get("tp_mult", default_cfg.get("tp_mult", 1.0)))
                except Exception:
                    tp_mult = 1.0
                if sl_mult != 1.0 or tp_mult != 1.0:
                    adj_sl = base_sl * sl_mult
                    adj_tp = base_tp * tp_mult
                    adjustment_applied = True

        elif regime == VolRegime.HIGH:
            # High Vol: Tighten everything
            adj_sl = base_sl * 0.85
            adj_tp = base_tp * 0.85
            adjustment_applied = True

        # === CRITICAL: ROUNDING TO NEAREST TICK (0.25) ===
        # This ensures inputs from Gemini (e.g. 4.52) become valid (4.50)
        adj_sl = round(adj_sl * 4) / 4
        adj_tp = round(adj_tp * 4) / 4

        # Minimum constraints (4.0 SL / 6.0 TP for positive RR)
        adj_sl = max(adj_sl, 4.0)  # 16 ticks minimum
        adj_tp = max(adj_tp, 6.0)  # 24 ticks minimum (1.5:1 RR)

        return {
            'sl_dist': adj_sl,
            'tp_dist': adj_tp,
            'size': adj_size,
            'regime': regime,
            'current_std': current_std,
            'hierarchy_key': key,
            'thresholds': thresholds,
            'adjustment_applied': adjustment_applied,
            'base_sl': base_sl,
            'base_tp': base_tp,
            'base_size': base_size,
            'dynamic_mult': dynamic_mult  # Log this for visibility
        }
    
    def log_status(self, adjustments: Dict):
        """Log volatility filter status."""
        regime = adjustments['regime']
        emoji = {
            VolRegime.ULTRA_LOW: "🔴",
            VolRegime.LOW: "🟡", 
            VolRegime.NORMAL: "🟢",
            VolRegime.HIGH: "🔵"
        }.get(regime, "⚪")
        
        t = adjustments['thresholds']
        msg = (f"{emoji} VolFilter [{adjustments['hierarchy_key']}|{regime}]: "
               f"std={adjustments['current_std']:.7f} "
               f"(p10={t['p10']:.6f}, p25={t['p25']:.6f}, p75={t['p75']:.6f})")

        if adjustments['adjustment_applied']:
            msg += (f" | SL:{adjustments['base_sl']:.2f}→{adjustments['sl_dist']:.2f}, "
                   f"TP:{adjustments['base_tp']:.2f}→{adjustments['tp_dist']:.2f}")
        else:
            msg += " | No SL/TP change (volatility within normal band)"

        logging.info(msg)

    def calibrate(self, df: pd.DataFrame, min_samples: int = 30):
        """
        10/10 UPGRADE: Rolling Window Calibration Engine.
        Recalculates p10/p25/p75 thresholds based on recent live data.

        Args:
            df: DataFrame containing recent history (e.g., last 5,000-20,000 bars)
            min_samples: Minimum bars required in a specific time-bucket to trigger an update
        """
        if len(df) < 500:
            logging.warning("⚠️ Volatility Calibration Skipped: Insufficient history (<500 bars)")
            return

        logging.info(f"⚙️ STARTUP CALIBRATION: Analyzing {len(df)} bars to adjust volatility map...")

        # 1. Prepare Data with session-specific std windows
        df_cal = df.copy()
        df_cal['returns'] = df_cal['close'].pct_change()
        sessions = []
        for ts in df_cal.index:
            sessions.append(self.get_session(ts.hour))
        df_cal['session'] = sessions

        windows = {self._std_window_for_session(sess) for sess in set(sessions)}
        std_map = {}
        for window in windows:
            std_map[window] = df_cal['returns'].rolling(window).std()

        std_series = []
        for sess, ts in zip(sessions, df_cal.index):
            window = self._std_window_for_session(sess)
            std_series.append(std_map[window].loc[ts])
        df_cal['std'] = pd.Series(std_series, index=df_cal.index)
        df_cal = df_cal.dropna(subset=['std'])

        # 2. Tag Data with Hierarchy Keys
        hierarchy_keys = []
        for ts in df_cal.index:
            hierarchy_keys.append(self.get_hierarchy_key(ts))

        df_cal['hierarchy_key'] = hierarchy_keys

        # 3. Group & Calculate Distributions
        grouped = df_cal.groupby('hierarchy_key')['std']

        updates = 0
        skipped = 0

        calibrated = {}
        for key, group in grouped:
            # Only update if we have statistically significant recent data for this specific slot
            if len(group) < min_samples:
                skipped += 1
                continue

            # Calculate Live Percentiles
            new_p10 = float(group.quantile(0.10))
            new_p25 = float(group.quantile(0.25))
            new_med = float(group.median())
            new_p75 = float(group.quantile(0.75))

            # Log significant drifts vs static map (if present)
            if key in VOLATILITY_HIERARCHY:
                old_p75 = VOLATILITY_HIERARCHY[key]['p75']
                if old_p75:
                    drift = abs(new_p75 - old_p75) / old_p75
                    if drift > 0.20:
                        logging.info(
                            f"   🌊 DRIFT DETECTED [{key}]: p75 shifted {old_p75:.6f} -> {new_p75:.6f} ({drift:.1%})"
                        )

            calibrated[key] = {
                "p10": new_p10,
                "p25": new_p25,
                "median": new_med,
                "p75": new_p75
            }
            updates += 1
        self._calibrated_thresholds = calibrated
        self._calibrated_sessions = set(df_cal['session'].unique())
        logging.info(f"✅ CALIBRATION COMPLETE: Updated {updates} regime buckets based on recent market conditions.")


# Global instance
volatility_filter = HierarchicalVolatilityFilter(
    skip_ultra_low=True,
    adjust_low_vol=True,
    low_vol_stop_mult=1.5,
    low_vol_size_mult=0.67
)


def check_volatility(df: pd.DataFrame, 
                     base_sl: float, 
                     base_tp: float,
                     base_size: int = 1,
                     ts=None) -> Tuple[bool, Dict]:
    """
    Convenience function to check volatility and get adjustments.
    
    Returns: (should_trade: bool, adjustments: Dict)
    """
    should_skip, reason = volatility_filter.should_skip_trade(df, ts)
    
    if should_skip:
        logging.info(f"🔴 VOL SKIP: {reason}")
        return False, {'skip_reason': reason}
    
    adjustments = volatility_filter.get_adjustments(df, base_sl, base_tp, base_size, ts)
    volatility_filter.log_status(adjustments)
    
    return True, adjustments
