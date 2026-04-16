import builtins
import concurrent.futures as cf
import glob
import datetime as dt
import argparse
import csv
import json
import hashlib
import logging
import math
import re
import statistics
import time
import sys
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from zoneinfo import ZoneInfo

# Provide fallbacks for modules that assume these globals without imports.
builtins.logging = logging
builtins.datetime = dt

from config import CONFIG, refresh_target_symbol
from data_cache import cache_key_for_source
import param_scaler
from dynamic_chop import DynamicChopAnalyzer
from rejection_filter import RejectionFilter
from chop_filter import ChopFilter
from extension_filter import ExtensionFilter
from trend_filter import TrendFilter
from dynamic_structure_blocker import (
    DynamicStructureBlocker,
    RegimeStructureBlocker,
    PenaltyBoxBlocker,
)
from bank_level_quarter_filter import BankLevelQuarterFilter
from memory_sr_filter import MemorySRFilter
from ml_physics_strategy import MLPhysicsStrategy
import ml_physics_pipeline as ml_physics_pipeline
from volatility_filter import volatility_filter, check_volatility
from fixed_sltp_framework import apply_fixed_sltp, log_fixed_sltp, asia_viability_gate
from volume_profile import build_volume_profile
from dynamic_sltp_params import dynamic_sltp_engine
from directional_loss_blocker import DirectionalLossBlocker
from impulse_filter import ImpulseFilter
from legacy_filters import LegacyFilterSystem
from filter_arbitrator import FilterArbitrator
from continuation_strategy import FractalSweepStrategy, STRATEGY_CONFIGS
from htf_fvg_filter import HTFFVGFilter
from news_filter import NewsFilter
from regime_manifold_engine import RegimeManifoldEngine, apply_meta_policy
from strategy_gate_policy import evaluate_pre_signal_gate
from de3_v3_family_schema import canonical_context_usage_snapshot
from de3_walkforward_gate import (
    build_feature_row as build_de3_walkforward_feature_row,
    build_model_frame as build_de3_walkforward_model_frame,
    build_model_vector as build_de3_walkforward_model_vector,
    de3_lane_context_key as de3_walkforward_lane_context_key,
    de3_variant_key as de3_walkforward_variant_key,
    load_artifact as load_de3_walkforward_artifact,
    load_model_bundle as load_de3_walkforward_model_bundle,
)
from event_logger import event_logger


def safe_float(value, default=0.0):
    try:
        out = float(value)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _file_fingerprint(path: Path) -> dict:
    out = {
        "path": str(path),
        "exists": bool(path.exists()),
        "sha256": "",
        "size_bytes": 0,
        "mtime": None,
    }
    if not path.exists():
        return out
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        st = path.stat()
        out["sha256"] = hasher.hexdigest()
        out["size_bytes"] = int(getattr(st, "st_size", 0) or 0)
        out["mtime"] = dt.datetime.fromtimestamp(float(getattr(st, "st_mtime", 0.0)), NY_TZ).isoformat()
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _flatten_dict_for_audit(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            key_text = str(key)
            next_prefix = f"{prefix}.{key_text}" if prefix else key_text
            out.update(_flatten_dict_for_audit(sub, next_prefix))
    else:
        out[str(prefix)] = value
    return out


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_CSV_NAME = str(
    (
        (CONFIG.get("DE3_V4", {}) or {}).get("training_data", {}) or {}
    ).get("parquet_path", "es_master_outrights.parquet")
    or "es_master_outrights.parquet"
)
CONTRACTS = 5
POINT_VALUE = 5.0
# Round-trip fees for 20 contracts. If your total (open+close) fees are $3.70 for 5 contracts,
# that's $0.74/contract round-trip => $14.80 per 20 contracts.
FEES_PER_20_CONTRACTS = 14.80
FEE_PER_CONTRACT_RT = FEES_PER_20_CONTRACTS / 20.0
FEE_PER_TRADE = FEE_PER_CONTRACT_RT * CONTRACTS
BACKTEST_MONTE_CARLO_CFG = (
    CONFIG.get("BACKTEST_MONTE_CARLO", {})
    if isinstance(CONFIG.get("BACKTEST_MONTE_CARLO", {}), dict)
    else {}
)
BACKTEST_MONTE_CARLO_ENABLED = bool(BACKTEST_MONTE_CARLO_CFG.get("enabled", True))
BACKTEST_MONTE_CARLO_SIMULATIONS = max(
    1,
    int(BACKTEST_MONTE_CARLO_CFG.get("simulations", 1000) or 1000),
)
BACKTEST_MONTE_CARLO_SEED = int(BACKTEST_MONTE_CARLO_CFG.get("seed", 1337) or 1337)
BACKTEST_MONTE_CARLO_START_BALANCE = float(
    BACKTEST_MONTE_CARLO_CFG.get("starting_balance", 50000.0) or 50000.0
)
BACKTEST_POST_RUN_CFG = (
    CONFIG.get("BACKTEST_POST_RUN_RECOMMENDER", {})
    if isinstance(CONFIG.get("BACKTEST_POST_RUN_RECOMMENDER", {}), dict)
    else {}
)
BACKTEST_BASELINE_COMPARISON_ENABLED = bool(
    BACKTEST_POST_RUN_CFG.get("enable_baseline_comparison", True)
)
BACKTEST_BASELINE_LOOKBACK_RUNS = max(
    1, int(BACKTEST_POST_RUN_CFG.get("baseline_auto_lookback_runs", 20) or 20)
)
BACKTEST_BASELINE_REPORT_PATH = str(
    BACKTEST_POST_RUN_CFG.get("baseline_report_path", "") or ""
).strip()
BACKTEST_GEMINI_RECOMMENDER_ENABLED = bool(
    BACKTEST_POST_RUN_CFG.get("enable_gemini_recommendation", True)
)
BACKTEST_GEMINI_RECOMMENDER_TIMEOUT_SEC = max(
    10, int(BACKTEST_POST_RUN_CFG.get("gemini_timeout_sec", 60) or 60)
)
BACKTEST_GEMINI_RECOMMENDER_MAX_CODE_CHARS = max(
    400, int(BACKTEST_POST_RUN_CFG.get("max_code_context_chars", 1500) or 1500)
)
BACKTEST_GEMINI_RECOMMENDER_DE3_FILES = [
    str(v).strip()
    for v in (
        BACKTEST_POST_RUN_CFG.get(
            "de3_context_files",
            [
                "dynamic_engine3_strategy.py",
                "de3_v4_runtime.py",
                "de3_v4_router.py",
                "de3_v4_lane_selector.py",
                "de3_v4_bracket_module.py",
                "config.py",
            ],
        )
        or []
    )
    if str(v).strip()
]
TICK_SIZE = 0.25
WARMUP_BARS = 20000
OPPOSITE_SIGNAL_THRESHOLD = 3
MIN_SL = 0.0
MIN_TP = 0.0
ENABLE_CONSENSUS_BYPASS = True
DISABLE_CONTINUATION_NY = False
ENABLE_DYNAMIC_ENGINE_1 = False
ENABLE_DYNAMIC_ENGINE_2 = False
ENABLE_DYNAMIC_ENGINE_3 = True
ALLOW_DYNAMIC_ENGINE_SOLO = False
ENABLE_HOSTILE_DAY_GUARD = True
HOSTILE_DAY_MAX_TRADES = 3
HOSTILE_DAY_MIN_TRADES = 2
HOSTILE_DAY_LOSS_THRESHOLD = 2
TREND_DAY_ENABLED = True
ATR_BASELINE_WINDOW = 390
ATR_EXP_T1 = 1.5
ATR_EXP_T2 = 1.6
VWAP_SIGMA_T1 = 1.75
VWAP_NO_RECLAIM_BARS_T1 = 20
VWAP_NO_RECLAIM_BARS_T2 = 20
VWAP_RECLAIM_SIGMA = 0.5
VWAP_RECLAIM_CONSECUTIVE_BARS = 15
TREND_DAY_STICKY_RECLAIM_BARS = 10
# Trend-day "fade/rotation" deactivation (visual match)
TREND_DAY_DEACTIVATE_RECLAIM_WINDOW = 15
TREND_DAY_DEACTIVATE_RECLAIM_COUNT = 12
TREND_DAY_DEACTIVATE_SIGMA_THRESHOLD = 0.8
TREND_DAY_DEACTIVATE_SIGMA_BARS = 10
TREND_DAY_DEACTIVATE_SIGMA_DECAY_THRESHOLD = 0.9
TREND_DAY_DEACTIVATE_SIGMA_DECAY_BARS = 20
TREND_DAY_DEACTIVATE_SIGMA_GUARD_MAX = 2.0
TREND_DAY_DEACTIVATE_SIGMA_GUARD_CURRENT = 1.1
TREND_UP_EMA_SLOPE_BARS = 20
TREND_UP_ATR_EXP = 1.4
TREND_UP_ABOVE_EMA50_WINDOW = 10
TREND_UP_ABOVE_EMA50_COUNT = 8
TREND_UP_HL_SEGMENT = 5
TREND_DOWN_EMA_SLOPE_BARS = 20
TREND_DOWN_ATR_EXP = 1.4
TREND_DOWN_BELOW_EMA50_WINDOW = 10
TREND_DOWN_BELOW_EMA50_COUNT = 8
TREND_DOWN_LH_SEGMENT = 5
ADX_PERIOD = 14
ADX_FLIP_THRESHOLD = 25.0
ADX_FLIP_BARS = 50
SIGMA_WINDOW = 30
IMPULSE_MIN_BARS = 30
IMPULSE_MAX_RETRACE = 0.25
TREND_DAY_T1_REQUIRE_CONFIRMATION = True
TREND_DAY_TIMEFRAME_MINUTES = 1
ALT_PRE_TIER1_VWAP_SIGMA = 2.25
TREND_DAY_T1_REQUIRE_STRUCTURAL_BIAS = True
TREND_DAY_ASIA_STRUCTURE_ONLY = True
TREND_DAY_ASIA_DISABLE_TIER2 = True
BACKTEST_TRENDDAY_VERBOSE = bool(CONFIG.get("BACKTEST_TRENDDAY_VERBOSE", False))

_backtest_exec = CONFIG.get("BACKTEST_EXECUTION", {}) or {}
BACKTEST_SL_TP_CONFLICT = str(_backtest_exec.get("sl_tp_conflict", "ohlc") or "ohlc").lower()
# Hard enforce gap fills at stop/limit price in backtests.
# This preserves OCO semantics during open gaps (no open-price slippage fills).
BACKTEST_GAP_FILLS = True
try:
    _sl_cap_raw = _backtest_exec.get("max_stoploss_points", None)
    BACKTEST_MAX_STOPLOSS_POINTS = float(_sl_cap_raw) if _sl_cap_raw is not None else None
except Exception:
    BACKTEST_MAX_STOPLOSS_POINTS = None
if BACKTEST_MAX_STOPLOSS_POINTS is not None and (
    not np.isfinite(BACKTEST_MAX_STOPLOSS_POINTS) or BACKTEST_MAX_STOPLOSS_POINTS <= 0.0
):
    BACKTEST_MAX_STOPLOSS_POINTS = None
BACKTEST_DISABLE_MAX_STOPLOSS_FOR_MLPHYSICS = bool(
    _backtest_exec.get("disable_max_stoploss_for_mlphysics", False)
)
BACKTEST_DISABLE_MAX_STOPLOSS_FOR_DE3_V2 = bool(
    _backtest_exec.get("disable_max_stoploss_for_de3_v2", True)
)
BACKTEST_ENFORCE_NO_NEW_ENTRIES_WINDOW = bool(
    _backtest_exec.get("enforce_no_new_entries_window", True)
)
try:
    BACKTEST_NO_NEW_ENTRIES_START_HOUR_ET = int(
        _backtest_exec.get("no_new_entries_start_hour_et", 16)
    )
except Exception:
    BACKTEST_NO_NEW_ENTRIES_START_HOUR_ET = 16
try:
    BACKTEST_NO_NEW_ENTRIES_END_HOUR_ET = int(
        _backtest_exec.get("no_new_entries_end_hour_et", 18)
    )
except Exception:
    BACKTEST_NO_NEW_ENTRIES_END_HOUR_ET = 18
if not 0 <= BACKTEST_NO_NEW_ENTRIES_START_HOUR_ET <= 23:
    BACKTEST_NO_NEW_ENTRIES_START_HOUR_ET = 16
if not 0 <= BACKTEST_NO_NEW_ENTRIES_END_HOUR_ET <= 23:
    BACKTEST_NO_NEW_ENTRIES_END_HOUR_ET = 18
BACKTEST_FORCE_FLAT_AT_TIME = bool(_backtest_exec.get("force_flat_at_time", True))
try:
    BACKTEST_FORCE_FLAT_HOUR_ET = int(_backtest_exec.get("force_flat_hour_et", 16))
except Exception:
    BACKTEST_FORCE_FLAT_HOUR_ET = 16
try:
    BACKTEST_FORCE_FLAT_MINUTE_ET = int(_backtest_exec.get("force_flat_minute_et", 0))
except Exception:
    BACKTEST_FORCE_FLAT_MINUTE_ET = 0
if not 0 <= BACKTEST_FORCE_FLAT_HOUR_ET <= 23:
    BACKTEST_FORCE_FLAT_HOUR_ET = 16
if not 0 <= BACKTEST_FORCE_FLAT_MINUTE_ET <= 59:
    BACKTEST_FORCE_FLAT_MINUTE_ET = 0
BACKTEST_DRAWDOWN_SIZE_SCALING_ENABLED = bool(
    _backtest_exec.get("drawdown_size_scaling_enabled", False)
)
try:
    BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD = float(
        _backtest_exec.get("drawdown_size_scaling_start_usd", 0.0)
    )
except Exception:
    BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD = 0.0
try:
    BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD = float(
        _backtest_exec.get("drawdown_size_scaling_max_usd", 2000.0)
    )
except Exception:
    BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD = 2000.0
try:
    BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS = int(
        _backtest_exec.get("drawdown_size_scaling_base_contracts", CONTRACTS)
    )
except Exception:
    BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS = int(CONTRACTS)
try:
    BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS = int(
        _backtest_exec.get("drawdown_size_scaling_min_contracts", 1)
    )
except Exception:
    BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS = 1
if not np.isfinite(BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD):
    BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD = 0.0
if not np.isfinite(BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD):
    BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD = 2000.0
BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD = max(
    0.0, float(BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD)
)
BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD = max(
    BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD,
    float(BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD),
)
BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS = max(
    1, int(BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS)
)
BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS = max(
    1, int(BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS)
)
if BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS > BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS:
    BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS = BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS
if (
    BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD <= BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD
    or BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS <= BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS
):
    BACKTEST_DRAWDOWN_SIZE_SCALING_ENABLED = False
BACKTEST_DRAWDOWN_SIZE_SCALING_CONTRACT_RANGE = max(
    0,
    int(BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS)
    - int(BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS),
)
BACKTEST_DRAWDOWN_SIZE_SCALING_SPAN_USD = max(
    0.0,
    float(BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD)
    - float(BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD),
)
if (
    BACKTEST_DRAWDOWN_SIZE_SCALING_ENABLED
    and BACKTEST_DRAWDOWN_SIZE_SCALING_CONTRACT_RANGE > 0
    and BACKTEST_DRAWDOWN_SIZE_SCALING_SPAN_USD > 0.0
):
    BACKTEST_DRAWDOWN_SIZE_SCALING_STEP_USD = (
        BACKTEST_DRAWDOWN_SIZE_SCALING_SPAN_USD
        / float(BACKTEST_DRAWDOWN_SIZE_SCALING_CONTRACT_RANGE)
    )
    BACKTEST_DRAWDOWN_SIZE_SCALING_SPAN_INV = 1.0 / BACKTEST_DRAWDOWN_SIZE_SCALING_SPAN_USD
else:
    BACKTEST_DRAWDOWN_SIZE_SCALING_STEP_USD = 0.0
    BACKTEST_DRAWDOWN_SIZE_SCALING_SPAN_INV = 0.0
    BACKTEST_DRAWDOWN_SIZE_SCALING_ENABLED = False
BACKTEST_ENFORCE_US_HOLIDAY_CLOSURE = bool(
    _backtest_exec.get("enforce_us_holiday_closure", True)
)
_holiday_sessions_raw = _backtest_exec.get("holiday_closure_sessions_et", ["NY_AM", "NY_PM"])
if isinstance(_holiday_sessions_raw, str):
    _holiday_sessions_raw = [_holiday_sessions_raw]
if not isinstance(_holiday_sessions_raw, (list, tuple, set)):
    _holiday_sessions_raw = []
BACKTEST_HOLIDAY_CLOSURE_SESSIONS_ET = tuple(
    str(v).strip().upper() for v in _holiday_sessions_raw if str(v).strip()
)
BACKTEST_HOLIDAY_CLOSURE_SESSION_SET_ET = set(BACKTEST_HOLIDAY_CLOSURE_SESSIONS_ET)
_extra_closed_dates_raw = _backtest_exec.get("extra_closed_dates_et", [])
if not isinstance(_extra_closed_dates_raw, (list, tuple, set)):
    _extra_closed_dates_raw = []
BACKTEST_EXTRA_CLOSED_DATES_ET = tuple(str(v).strip() for v in _extra_closed_dates_raw if str(v).strip())
BACKTEST_EARLY_EXIT_ENABLED = False

# Backtest-only: disable FixedSLTP room-to-target viability check
_fixed_sltp = CONFIG.get("FIXED_SLTP_FRAMEWORK", {}) or {}
_viab = _fixed_sltp.get("viability", {}) or {}
_viab["disable_room_to_target"] = False
_fixed_sltp["viability"] = _viab
CONFIG["FIXED_SLTP_FRAMEWORK"] = _fixed_sltp

SL_BUCKETS = [4.0, 6.0, 8.0, 10.0, 15.0]
TP_BUCKETS = [6.0, 8.0, 10.0, 15.0, 20.0, 30.0]
RR_BUCKETS = [1.0, 1.5, 2.0, 3.0]
CONSENSUS_BYPASSED_FILTERS = [
    "RejectionFilter",
    "ImpulseFilter",
    "HTF_FVG",
    "StructureBlocker",
    "BankLevelQuarterFilter",
    "LegacyTrend",
    "FilterArbitrator",
    "ExtensionFilter",
]

BACKTEST_SELECTABLE_STRATEGIES = [
    "RegimeAdaptiveStrategy",
    "VIXReversionStrategy",
    "ImpulseBreakoutStrategy",
    "DynamicEngineStrategy",
    "DynamicEngine3Strategy",
    "IntradayDipStrategy",
    "AuctionReversionStrategy",
    "LiquiditySweepStrategy",
    "ValueAreaBreakoutStrategy",
    "ConfluenceStrategy",
    "SMTStrategy",
    "SmoothTrendAsiaStrategy",
    "MLPhysicsStrategy",
    "MLPhysicsLegacyExperimentStrategy",
    "ManifoldStrategy",
    "AetherFlowStrategy",
    "OrbStrategy",
    "ICTModelStrategy",
]

BACKTEST_SELECTABLE_FILTERS = [
    "NewsFilter",
    "PreCandidateGate",
    "RegimeManifold",
    "FixedSLTP",
    "TrendDayTier",
    "TargetFeasibility",
    "RejectionFilter",
    "DirectionalLossBlocker",
    "ImpulseFilter",
    "HTF_FVG",
    "StructureBlocker",
    "RegimeBlocker",
    "PenaltyBoxBlocker",
    "MemorySRFilter",
    "BankLevelQuarterFilter",
    "LegacyTrend",
    "TrendFilter",
    "ChopFilter",
    "ExtensionFilter",
    "VolatilityGuardrail",
    "MLVolRegimeGuard",
    "FilterArbitrator",
]


def _hour_in_window(hour: int, start_hour: int, end_hour: int) -> bool:
    """Half-open [start_hour, end_hour) with wrap support across midnight."""
    if start_hour == end_hour:
        return False
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def _parse_et_date(value) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except Exception:
        return None


def _build_closed_holiday_dates_et(index: pd.DatetimeIndex) -> set[dt.date]:
    if (not BACKTEST_ENFORCE_US_HOLIDAY_CLOSURE) or index is None or len(index) == 0:
        return set()
    try:
        idx = pd.DatetimeIndex(index)
        if idx.tz is None:
            idx_et = idx.tz_localize(NY_TZ)
        else:
            idx_et = idx.tz_convert(NY_TZ)
        start_day = pd.Timestamp(idx_et.min().date())
        end_day = pd.Timestamp(idx_et.max().date())
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=start_day, end=end_day)
        closed_dates = {ts.date() for ts in holidays}
    except Exception:
        closed_dates = set()

    for raw in BACKTEST_EXTRA_CLOSED_DATES_ET:
        extra_day = _parse_et_date(raw)
        if extra_day is not None:
            closed_dates.add(extra_day)
    return closed_dates


def _is_holiday_blocked_for_session(session_name: str) -> bool:
    # Empty list or explicit ALL means block all sessions on holiday dates.
    if not BACKTEST_HOLIDAY_CLOSURE_SESSION_SET_ET:
        return True
    if "ALL" in BACKTEST_HOLIDAY_CLOSURE_SESSION_SET_ET:
        return True
    return str(session_name or "").upper() in BACKTEST_HOLIDAY_CLOSURE_SESSION_SET_ET


def get_session_name(ts: dt.datetime) -> str:
    hour = ts.hour
    if hour >= 18 or hour < 3:
        return "ASIA"
    if 3 <= hour < 8:
        return "LONDON"
    if 8 <= hour < 12:
        return "NY_AM"
    if 12 <= hour < 17:
        return "NY_PM"
    return "OFF"


def asia_trend_bias(history_df: pd.DataFrame, cfg: dict) -> Optional[str]:
    if history_df.empty:
        return None
    close = history_df["close"]
    ema_fast = int(cfg.get("ema_fast", 20) or 20)
    ema_slow = int(cfg.get("ema_slow", 50) or 50)
    ema_slope_bars = int(cfg.get("ema_slope_bars", 20) or 20)
    if len(close) < max(ema_slow, ema_slope_bars + 1) + 1:
        return None
    ema_fast_series = close.ewm(span=ema_fast, adjust=False).mean()
    ema_slow_series = close.ewm(span=ema_slow, adjust=False).mean()
    fast_val = float(ema_fast_series.iloc[-1])
    slow_val = float(ema_slow_series.iloc[-1])
    slope = fast_val - float(ema_fast_series.iloc[-ema_slope_bars])
    min_sep = float(cfg.get("min_ema_separation", 0.1) or 0.0)
    if fast_val > slow_val and slope > 0 and (fast_val - slow_val) >= min_sep:
        return "LONG"
    if fast_val < slow_val and slope < 0 and (slow_val - fast_val) >= min_sep:
        return "SHORT"
    return None


def asia_target_feasibility_override(
    history_df: pd.DataFrame,
    side: str,
    tp_distance: Optional[float],
    trend_bias: Optional[str],
    cfg: dict,
    lookback: int,
) -> bool:
    if not cfg or not cfg.get("enabled", True):
        return False
    if not history_df.empty and trend_bias:
        if str(side).upper() != str(trend_bias).upper():
            return False
    else:
        return False
    try:
        tp_val = float(tp_distance)
    except Exception:
        return False
    if tp_val <= 0:
        return False
    lookback = int(cfg.get("lookback", lookback) or lookback)
    if len(history_df) < lookback:
        return False
    window = history_df.iloc[-lookback:]
    box_range = float(window["high"].max() - window["low"].min())
    min_box = float(cfg.get("min_box_range", 0.0) or 0.0)
    if box_range <= 0 or box_range < min_box:
        return False
    max_mult = float(cfg.get("max_tp_box_mult", 1.5) or 1.5)
    if tp_val <= box_range * max_mult:
        return True
    return bool(cfg.get("allow_trend_override", False))


def asia_chop_override(
    chop_reason: Optional[str],
    side: str,
    trend_bias: Optional[str],
    cfg: dict,
) -> bool:
    if not cfg or not cfg.get("enabled", True):
        return False
    if not trend_bias or str(side).upper() != str(trend_bias).upper():
        return False
    reason_lc = str(chop_reason or "").lower()
    if "wait for breakout" in reason_lc or "range too tight" in reason_lc:
        return False
    return bool(cfg.get("allow_trend_override", True))


def bucket_label(value: float, edges: list[float]) -> str:
    for edge in edges:
        if value <= edge:
            return f"<= {edge:.2f}"
    return f"> {edges[-1]:.2f}"


def _coerce_float(value, fallback: float) -> float:
    try:
        num = float(value)
    except Exception:
        return fallback
    if not math.isfinite(num):
        return fallback
    return num


def _coerce_int(value, fallback: int) -> int:
    try:
        num = int(value)
    except Exception:
        return fallback
    if num <= 0:
        return fallback
    return num


def _signal_base_size(signal: Optional[dict], fallback: int) -> int:
    if not isinstance(signal, dict):
        return int(fallback)
    try:
        size = int(signal.get("size", fallback) or fallback)
    except Exception:
        size = int(fallback)
    return max(1, size)


def _resolve_sl_tp_conflict(
    side: str,
    bar_open: float,
    bar_close: float,
    stop_price: float,
    take_price: float,
) -> tuple[float, str]:
    mode = BACKTEST_SL_TP_CONFLICT
    if mode == "take":
        return take_price, "take"
    if mode == "stop":
        return stop_price, "stop"
    if mode == "best":
        return (take_price, "take") if side == "LONG" else (take_price, "take")
    if mode == "worst":
        return (stop_price, "stop") if side == "LONG" else (stop_price, "stop")

    # OHLC heuristic: green bar assumes O-L-H-C, red bar assumes O-H-L-C.
    is_green = bar_close >= bar_open
    if side == "LONG":
        stop_first = is_green
    else:
        stop_first = not is_green
    return (stop_price, "stop") if stop_first else (take_price, "take")


def format_strategy_label(signal: dict, fallback: str) -> str:
    label = signal.get("strategy", fallback)
    sub = signal.get("sub_strategy")
    if sub:
        label = f"{label}:{sub}"
    return label


def format_rows(title: str, rows: list[tuple], headers: list[str], max_rows: int = 10) -> str:
    lines = [title]
    if not rows:
        lines.append("  (none)")
        return "\n".join(lines)
    lines.append("  " + " | ".join(headers))
    for row in rows[:max_rows]:
        lines.append("  " + " | ".join(str(item) for item in row))
    return "\n".join(lines)


def parse_continuation_key(strategy_name: Optional[str]) -> Optional[str]:
    if not strategy_name:
        return None
    name = str(strategy_name)
    if name.startswith("Continuation_"):
        return name.split("Continuation_", 1)[1]
    return None


def continuation_allowlist_key(raw_key: Optional[str], key_mode: Optional[str] = None) -> Optional[str]:
    if not raw_key:
        return None
    if key_mode is None:
        allow_cfg = CONFIG.get("BACKTEST_CONTINUATION_ALLOWLIST", {}) or {}
        key_mode = allow_cfg.get("key_granularity", "full")
    mode = str(key_mode or "full").lower()
    key_str = str(raw_key)
    if mode in ("full", "qwd", "qwd_session"):
        return key_str
    parts = key_str.split("_")
    session = parts[-1] if parts else key_str
    session_norm = str(session).upper()
    day = None
    for part in parts:
        if part.startswith("D") and part[1:].isdigit():
            day = part[1:]
            break
    if mode in ("session", "sess"):
        return session_norm
    if mode in ("session_day", "day_session", "dow_session"):
        if day:
            return f"D{day}_{session_norm}"
        return session_norm
    return key_str


def load_continuation_allowlist_file(path_value: Optional[str]) -> Optional[set]:
    if not path_value:
        return None
    try:
        path = Path(path_value)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            logging.warning(f"⚠️ Continuation allowlist file missing: {path}")
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowlist = payload.get("allowlist")
        if not isinstance(allowlist, list):
            return None
        return set(allowlist)
    except Exception as exc:
        logging.warning(f"⚠️ Continuation allowlist load failed: {exc}")
        return None


def build_continuation_allowlist(cfg: Optional[dict], base_dir: Path) -> tuple[Optional[set], dict]:
    if not cfg or not cfg.get("enabled", True):
        return None, {}
    pattern = cfg.get("reports_glob", "backtest_reports/backtest_*.json")
    report_paths = [Path(p) for p in glob.glob(str(base_dir / pattern))]
    key_mode = str(cfg.get("key_granularity", "full") or "full").lower()

    min_total_trades = int(cfg.get("min_total_trades", 0) or 0)
    min_fold_trades = int(cfg.get("min_fold_trades", 1) or 1)
    min_avg_pnl_points = float(cfg.get("min_avg_pnl_points", 0.0) or 0.0)
    min_fold_expectancy_points = float(cfg.get("min_fold_expectancy_points", 0.0) or 0.0)
    min_folds = int(cfg.get("min_folds", 1) or 1)
    min_positive_fold_ratio = float(cfg.get("min_positive_fold_ratio", 0.0) or 0.0)

    aggregate = defaultdict(
        lambda: {
            "total_trades": 0,
            "total_pnl_points": 0.0,
            "folds": 0,
            "positive_folds": 0,
            "fold_expectancies": [],
        }
    )
    reports_used = 0

    for report_path in report_paths:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summary = payload.get("summary", {}) or {}
        if summary.get("cancelled"):
            continue
        trade_log = payload.get("trade_log", []) or []
        if not trade_log:
            continue

        assumptions = payload.get("assumptions", {}) or {}
        point_value = float(assumptions.get("point_value", POINT_VALUE) or POINT_VALUE)
        contracts = float(assumptions.get("contracts", CONTRACTS) or CONTRACTS)
        denom = point_value * contracts if point_value and contracts else POINT_VALUE * CONTRACTS

        fold_stats = defaultdict(lambda: {"trades": 0, "pnl_points": 0.0})
        for trade in trade_log:
            raw_key = parse_continuation_key(trade.get("strategy"))
            key = continuation_allowlist_key(raw_key, key_mode)
            if not key:
                continue
            pnl_points = trade.get("pnl_points")
            if pnl_points is None:
                pnl_net = float(trade.get("pnl_net", 0.0) or 0.0)
                pnl_points = pnl_net / denom if denom else 0.0
            else:
                pnl_points = float(pnl_points)
            fold_stats[key]["trades"] += 1
            fold_stats[key]["pnl_points"] += pnl_points

        if not fold_stats:
            continue

        reports_used += 1
        for key, stats in fold_stats.items():
            trades = stats["trades"]
            pnl_points = stats["pnl_points"]
            agg = aggregate[key]
            agg["total_trades"] += trades
            agg["total_pnl_points"] += pnl_points
            if trades >= min_fold_trades:
                agg["folds"] += 1
                expectancy = pnl_points / trades if trades else 0.0
                agg["fold_expectancies"].append(expectancy)
                if expectancy >= min_fold_expectancy_points:
                    agg["positive_folds"] += 1

    allowlist = set()
    stats_out = {}
    for key, agg in aggregate.items():
        total_trades = agg["total_trades"]
        total_pnl_points = agg["total_pnl_points"]
        avg_pnl = total_pnl_points / total_trades if total_trades else 0.0
        folds = agg["folds"]
        positive_ratio = (agg["positive_folds"] / folds) if folds else 0.0
        allowed = (
            total_trades >= min_total_trades
            and avg_pnl >= min_avg_pnl_points
            and folds >= min_folds
            and positive_ratio >= min_positive_fold_ratio
        )
        stats_out[key] = {
            "total_trades": total_trades,
            "avg_pnl_points": avg_pnl,
            "folds": folds,
            "positive_ratio": positive_ratio,
            "allowed": allowed,
        }
        if allowed and key in STRATEGY_CONFIGS:
            if key_mode != "full" or key in STRATEGY_CONFIGS:
                allowlist.add(key)

    payload = {
        "generated_at": dt.datetime.now(NY_TZ).isoformat(),
        "summary": {
            "reports_seen": len(report_paths),
            "reports_used": reports_used,
            "keys_seen": len(stats_out),
            "keys_allowed": len(allowlist),
        },
        "criteria": {
            "min_total_trades": min_total_trades,
            "min_fold_trades": min_fold_trades,
            "min_avg_pnl_points": min_avg_pnl_points,
            "min_fold_expectancy_points": min_fold_expectancy_points,
            "min_folds": min_folds,
            "min_positive_fold_ratio": min_positive_fold_ratio,
            "key_granularity": key_mode,
        },
        "allowlist": sorted(allowlist),
        "stats": stats_out,
    }
    cache_file = cfg.get("cache_file")
    if cache_file:
        cache_path = Path(cache_file)
        if not cache_path.is_absolute():
            cache_path = base_dir / cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return allowlist, payload


def simulate_trade_points(
    df: pd.DataFrame,
    entry_pos: int,
    side: str,
    entry_price: float,
    sl_dist: float,
    tp_dist: float,
    max_horizon: int,
    assume_sl_first: bool,
    exit_at_horizon: str,
) -> float:
    last_pos = min(len(df) - 1, entry_pos + max_horizon)
    for pos in range(entry_pos, last_pos + 1):
        high = float(df.iloc[pos]["high"])
        low = float(df.iloc[pos]["low"])
        if side == "LONG":
            hit_tp = high >= entry_price + tp_dist
            hit_sl = low <= entry_price - sl_dist
            if hit_tp and hit_sl:
                return -sl_dist if assume_sl_first else tp_dist
            if hit_tp:
                return tp_dist
            if hit_sl:
                return -sl_dist
        else:
            hit_tp = low <= entry_price - tp_dist
            hit_sl = high >= entry_price + sl_dist
            if hit_tp and hit_sl:
                return -sl_dist if assume_sl_first else tp_dist
            if hit_tp:
                return tp_dist
            if hit_sl:
                return -sl_dist

    if exit_at_horizon == "close":
        exit_price = float(df.iloc[last_pos]["close"])
        return compute_pnl_points(side, entry_price, exit_price)
    return 0.0


def build_continuation_allowlist_from_df(
    df: pd.DataFrame,
    trend_context: dict,
    cfg: Optional[dict],
    allowed_regimes: set,
    confirm_cfg: Optional[dict],
) -> tuple[Optional[set], dict]:
    if not cfg or not cfg.get("enabled", True):
        return None, {}
    fast_cfg = cfg.get("fast", {}) or {}
    folds = int(fast_cfg.get("folds", 4) or 1)
    max_horizon = int(fast_cfg.get("max_horizon_bars", 120) or 120)
    exit_at_horizon = str(fast_cfg.get("exit_at_horizon", "close") or "close").lower()
    assume_sl_first = bool(fast_cfg.get("assume_sl_first", True))
    use_dynamic_sltp = bool(fast_cfg.get("use_dynamic_sltp", True))
    default_tp = float(fast_cfg.get("default_tp", MIN_TP) or MIN_TP)
    default_sl = float(fast_cfg.get("default_sl", MIN_SL) or MIN_SL)
    min_win_rate = float(fast_cfg.get("min_win_rate", 0.0) or 0.0)

    min_total_trades = int(cfg.get("min_total_trades", 0) or 0)
    min_fold_trades = int(cfg.get("min_fold_trades", 1) or 1)
    min_avg_pnl_points = float(cfg.get("min_avg_pnl_points", 0.0) or 0.0)
    min_fold_expectancy_points = float(cfg.get("min_fold_expectancy_points", 0.0) or 0.0)
    min_folds = int(cfg.get("min_folds", 1) or 1)
    min_positive_fold_ratio = float(cfg.get("min_positive_fold_ratio", 0.0) or 0.0)
    key_mode = str(cfg.get("key_granularity", "full") or "full").lower()

    if df.empty:
        return set(), {}

    def normalize_filter_values(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if item is not None]
        return [str(value)]

    symbol_prefixes = normalize_filter_values(
        fast_cfg.get("symbol_prefixes") or fast_cfg.get("symbol_prefix")
    )
    symbol_contains = normalize_filter_values(
        fast_cfg.get("symbol_contains") or fast_cfg.get("symbol_filter_contains")
    )

    def symbol_allowed(symbol: str) -> bool:
        if symbol_prefixes and not any(symbol.startswith(prefix) for prefix in symbol_prefixes):
            return False
        if symbol_contains and not any(token in symbol for token in symbol_contains):
            return False
        return True

    multi_symbol = "symbol" in df.columns and df["symbol"].nunique(dropna=True) > 1

    def iter_symbol_frames(frame: pd.DataFrame):
        if "symbol" not in frame.columns:
            yield None, frame
            return
        for symbol, symbol_df in frame.groupby("symbol"):
            symbol_name = str(symbol)
            if symbol_prefixes or symbol_contains:
                if not symbol_allowed(symbol_name):
                    continue
            yield symbol_name, symbol_df

    if allowed_regimes:
        try:
            calib_df = df
            if "symbol" in df.columns and (symbol_prefixes or symbol_contains):
                symbol_series = df["symbol"].astype(str)
                symbol_mask = symbol_series.apply(symbol_allowed)
                calib_df = df.loc[symbol_mask]
            if len(calib_df) >= 500:
                volatility_filter.calibrate(calib_df)
        except Exception:
            pass

    aggregate = defaultdict(
        lambda: {
            "total_trades": 0,
            "total_pnl_points": 0.0,
            "wins": 0,
            "folds": 0,
            "positive_folds": 0,
            "fold_expectancies": [],
        }
    )

    for symbol, symbol_df in iter_symbol_frames(df):
        if symbol_df.empty:
            continue
        symbol_df = normalize_index(symbol_df, NY_TZ)
        idx = symbol_df.index

        if trend_context is None or multi_symbol:
            trend_source = symbol_df
            if TREND_DAY_TIMEFRAME_MINUTES > 1:
                trend_source = resample_dataframe(symbol_df, TREND_DAY_TIMEFRAME_MINUTES)
            trend_series_raw = compute_trend_day_series(trend_source)
            local_trend_context = align_trend_day_series(trend_series_raw, symbol_df.index)
        else:
            local_trend_context = trend_context

        day_index = pd.Series(idx.date, index=idx)
        day_last = pd.Series(idx, index=idx).groupby(day_index).transform("max")
        # Use ndarray comparison to avoid tz-aware mismatch that yields all-False
        last_mask = idx.to_numpy() == day_last.to_numpy()

        hours = idx.hour
        session = np.where(
            (hours >= 18) | (hours < 3),
            "Asia",
            np.where(hours < 8, "London", np.where(hours < 17, "NY", "Other")),
        )

        quarters = idx.quarter
        weeks = idx.isocalendar().week.to_numpy()
        days = idx.weekday + 1

        base_ts = idx[0].value
        span_ts = max(1, idx[-1].value - base_ts)

        for pos in np.where(last_mask)[0]:
            if session[pos] == "Other":
                continue
            session_tag = session[pos]
            session_key = str(session_tag).upper()
            if key_mode in ("session", "sess"):
                key = session_key
            elif key_mode in ("session_day", "day_session", "dow_session"):
                key = f"D{days[pos]}_{session_key}"
            else:
                key = f"Q{quarters[pos]}_W{weeks[pos]}_D{days[pos]}_{session_tag}"
            if key not in STRATEGY_CONFIGS:
                if key_mode == "full":
                    continue
            current_time = idx[pos]
            bar_close = float(symbol_df.iloc[pos]["close"])

            if not confirm_cfg or not confirm_cfg.get("enabled", True):
                vwap_sigma = local_trend_context.get("vwap_sigma_dist")
                if isinstance(vwap_sigma, pd.Series):
                    try:
                        vwap_sigma = vwap_sigma.get(current_time, 0.0)
                    except Exception:
                        vwap_sigma = 0.0
                try:
                    vwap_sigma = float(vwap_sigma)
                except Exception:
                    vwap_sigma = 0.0
                if vwap_sigma > 0:
                    side = "LONG"
                elif vwap_sigma < 0:
                    side = "SHORT"
                else:
                    continue
            else:
                long_ok = continuation_market_confirmed(
                    "LONG", current_time, bar_close, local_trend_context, confirm_cfg
                )
                short_ok = continuation_market_confirmed(
                    "SHORT", current_time, bar_close, local_trend_context, confirm_cfg
                )
                if long_ok and not short_ok:
                    side = "LONG"
                elif short_ok and not long_ok:
                    side = "SHORT"
                else:
                    continue

            if allowed_regimes:
                try:
                    history_df = symbol_df.loc[:current_time]
                    regime, _, _ = volatility_filter.get_regime(history_df)
                except Exception:
                    regime = None
                if not regime or str(regime).lower() not in allowed_regimes:
                    continue

            entry_pos = pos + 1
            if entry_pos >= len(symbol_df):
                continue
            entry_price = float(symbol_df.iloc[entry_pos]["open"])

            tp_dist = default_tp
            sl_dist = default_sl
            if use_dynamic_sltp:
                try:
                    sltp = dynamic_sltp_engine.calculate_sltp(
                        "Continuation", symbol_df, ts=current_time
                    )
                    tp_dist = float(sltp.get("tp_dist", tp_dist))
                    sl_dist = float(sltp.get("sl_dist", sl_dist))
                except Exception:
                    pass
            tp_dist = max(tp_dist, MIN_TP)
            sl_dist = max(sl_dist, MIN_SL)

            pnl_points = simulate_trade_points(
                symbol_df,
                entry_pos,
                side,
                entry_price,
                sl_dist,
                tp_dist,
                max_horizon,
                assume_sl_first,
                exit_at_horizon,
            )

            fold_idx = int(((current_time.value - base_ts) / span_ts) * max(1, folds - 1))
            agg = aggregate[key]
            agg["total_trades"] += 1
            agg["total_pnl_points"] += pnl_points
            if pnl_points > 0:
                agg["wins"] += 1
            agg.setdefault("fold_stats", defaultdict(lambda: {"trades": 0, "pnl_points": 0.0}))
            agg["fold_stats"][fold_idx]["trades"] += 1
            agg["fold_stats"][fold_idx]["pnl_points"] += pnl_points

    allowlist = set()
    stats_out = {}
    for key, agg in aggregate.items():
        total_trades = agg["total_trades"]
        total_pnl_points = agg["total_pnl_points"]
        wins = agg["wins"]
        avg_pnl = total_pnl_points / total_trades if total_trades else 0.0
        win_rate = wins / total_trades if total_trades else 0.0

        fold_stats = agg.get("fold_stats", {})
        folds_used = 0
        positive_folds = 0
        for _, stats in fold_stats.items():
            trades = stats["trades"]
            pnl_points = stats["pnl_points"]
            if trades >= min_fold_trades:
                folds_used += 1
                expectancy = pnl_points / trades if trades else 0.0
                agg["fold_expectancies"].append(expectancy)
                if expectancy >= min_fold_expectancy_points:
                    positive_folds += 1
        positive_ratio = (positive_folds / folds_used) if folds_used else 0.0

        allowed = (
            total_trades >= min_total_trades
            and avg_pnl >= min_avg_pnl_points
            and win_rate >= min_win_rate
            and folds_used >= min_folds
            and positive_ratio >= min_positive_fold_ratio
        )

        stats_out[key] = {
            "total_trades": total_trades,
            "avg_pnl_points": avg_pnl,
            "win_rate": win_rate,
            "folds": folds_used,
            "positive_ratio": positive_ratio,
            "allowed": allowed,
        }
        if allowed:
            allowlist.add(key)

    payload = {
        "generated_at": dt.datetime.now(NY_TZ).isoformat(),
        "mode": "csv_fast",
        "summary": {
            "keys_seen": len(stats_out),
            "keys_allowed": len(allowlist),
        },
        "criteria": {
            "min_total_trades": min_total_trades,
            "min_fold_trades": min_fold_trades,
            "min_avg_pnl_points": min_avg_pnl_points,
            "min_fold_expectancy_points": min_fold_expectancy_points,
            "min_folds": min_folds,
            "min_positive_fold_ratio": min_positive_fold_ratio,
            "min_win_rate": min_win_rate,
            "folds": folds,
            "max_horizon_bars": max_horizon,
            "exit_at_horizon": exit_at_horizon,
            "assume_sl_first": assume_sl_first,
            "use_dynamic_sltp": use_dynamic_sltp,
            "key_granularity": key_mode,
        },
        "allowlist": sorted(allowlist),
        "stats": stats_out,
    }
    cache_file = cfg.get("cache_file")
    if cache_file:
        cache_path = Path(cache_file)
        if not cache_path.is_absolute():
            cache_path = Path(__file__).resolve().parent / cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return allowlist, payload


def build_flip_confidence_from_df(
    df: pd.DataFrame,
    cfg: Optional[dict] = None,
) -> dict:
    if df is None or df.empty:
        return {}
    local_cfg = dict(cfg or {})
    local_cfg["enabled"] = True
    local_cfg["cache_file"] = None
    prev_cfg = CONFIG.get("BACKTEST_FLIP_CONFIDENCE")
    CONFIG["BACKTEST_FLIP_CONFIDENCE"] = local_cfg
    try:
        start_time = df.index.min()
        end_time = df.index.max()
        stats = run_backtest(df, start_time, end_time)
        return stats.get("flip_confidence") or {}
    finally:
        CONFIG["BACKTEST_FLIP_CONFIDENCE"] = prev_cfg


class AttributionTracker:
    def __init__(self, recent_limit: int = 25):
        self.trades = []
        self.trade_count = 0
        self.win_trade_count = 0
        self.loss_trade_count = 0
        self.ml_trades = []
        self.recent_trades = deque(maxlen=recent_limit)
        self.filter_blocks = Counter()
        self.filter_rescues = Counter()
        self.filter_bypasses = Counter()
        self.strategy_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0})
        self.sub_strategy_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0})
        self.session_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        self.hour_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        self.exit_reason_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        self.entry_mode_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        self.vol_regime_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        self.sl_bucket_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        self.tp_bucket_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        self.rr_bucket_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0})
        self.mfe_sum = 0.0
        self.mae_sum = 0.0
        self.mfe_win_sum = 0.0
        self.mae_win_sum = 0.0
        self.mfe_loss_sum = 0.0
        self.mae_loss_sum = 0.0
        self.loss_streak_len = 0
        self.loss_streak_pnl = 0.0
        self.loss_streak_count = 0
        self.loss_streak_len_total = 0
        self.loss_streak_pnl_total = 0.0
        self.loss_streak_max_len = 0
        self.loss_streak_max_pnl = 0.0
        self.ml_diagnostics = []

    def record_filter(self, name: str, kind: str = "block") -> None:
        if kind == "rescue":
            self.filter_rescues[name] += 1
        elif kind == "bypass":
            self.filter_bypasses[name] += 1
        else:
            self.filter_blocks[name] += 1

    def record_ml_eval(self, payload: dict, max_records: int = 0) -> None:
        if max_records and len(self.ml_diagnostics) >= max_records:
            return
        self.ml_diagnostics.append(payload)

    @staticmethod
    def _to_float(value) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return out

    @staticmethod
    def _ml_fix_hint(reason: str) -> Optional[str]:
        text = str(reason or "").lower()
        if "gate_reject" in text or "tradeability_gate" in text or "threshold_gate" in text:
            return "Gate is the main blocker: relax gate threshold floors/caps or retrain gate with higher coverage target."
        if "ev_below_min" in text or "ev_filter" in text:
            return "EV filter is too strict: lower min EV requirement or improve bracket calibration for that session."
        if "rr_below_min" in text:
            return "RR constraint is blocking sides: widen TP ATR cap/candidates or reduce rr_min slightly."
        if "ev_uncertainty" in text:
            return "Model disagreement is high: relax disagreement caps or improve EV model stability for this session."
        if "insufficient_history" in text:
            return "Warmup is too short for current feature windows: increase warmup bars or start later."
        if "missing_models" in text:
            return "Artifact mismatch: verify run artifacts include both LONG/SHORT models for all active sessions."
        if "dist_inference_error" in text:
            return "Runtime inference errors detected: inspect traceback and model feature alignment."
        return None

    def summarize_ml_diagnostics(self, max_rows: int = 10) -> dict:
        total = len(self.ml_diagnostics)
        out = {
            "total_evals": total,
            "signal_count": 0,
            "blocked_count": 0,
            "no_signal_count": 0,
            "signal_rate": 0.0,
            "decision_counts": {},
            "blocked_reason_counts": {},
            "no_signal_reason_counts": {},
            "top_reason_codes": [],
            "session_rows": [],
            "gate_shortfall_negative_count": 0,
            "gate_shortfall_negative_avg": 0.0,
            "gate_shortfall_negative_max": 0.0,
            "unknown_session_count": 0,
            "unknown_session_ratio": 0.0,
            "recommendations": [],
        }
        if total == 0:
            return out

        decision_counts = Counter()
        blocked_reason_counts = Counter()
        no_signal_reason_counts = Counter()
        reason_code_counts = Counter()
        session_counts = Counter()
        session_signal_counts = Counter()
        session_block_reasons = defaultdict(Counter)
        session_no_signal_reasons = defaultdict(Counter)
        gate_shortfalls = []

        for row in self.ml_diagnostics:
            if not isinstance(row, dict):
                continue
            decision = str(row.get("decision") or "unknown").lower()
            session = str(row.get("session") or "UNKNOWN").upper()
            decision_counts[decision] += 1
            session_counts[session] += 1

            if decision in {"signal_long", "signal_short"}:
                session_signal_counts[session] += 1
            if decision == "blocked":
                out["blocked_count"] += 1
            elif decision == "no_signal":
                out["no_signal_count"] += 1

            blocked_reason = row.get("blocked_reason")
            if blocked_reason:
                reason = str(blocked_reason)
                if decision == "blocked":
                    blocked_reason_counts[reason] += 1
                    session_block_reasons[session][reason] += 1
                elif decision == "no_signal":
                    no_signal_reason_counts[reason] += 1
                    session_no_signal_reasons[session][reason] += 1

            reason_codes = row.get("reason_codes") or []
            if isinstance(reason_codes, (list, tuple)):
                for code in reason_codes:
                    reason_code_counts[str(code)] += 1

            gate_margin = self._to_float(row.get("trade_gate_margin"))
            if gate_margin is None:
                gate_prob = self._to_float(row.get("trade_gate_prob"))
                gate_threshold = self._to_float(row.get("trade_gate_threshold"))
                if gate_prob is not None and gate_threshold is not None:
                    gate_margin = gate_prob - gate_threshold
            if gate_margin is not None and gate_margin < 0.0:
                gate_shortfalls.append(abs(gate_margin))

        out["signal_count"] = int(
            decision_counts.get("signal_long", 0) + decision_counts.get("signal_short", 0)
        )
        out["signal_rate"] = (
            float(out["signal_count"]) / float(total)
            if total
            else 0.0
        )
        out["decision_counts"] = {k: int(v) for k, v in decision_counts.items()}
        out["blocked_reason_counts"] = {k: int(v) for k, v in blocked_reason_counts.items()}
        out["no_signal_reason_counts"] = {k: int(v) for k, v in no_signal_reason_counts.items()}
        out["top_reason_codes"] = [
            {"reason_code": key, "count": int(val)}
            for key, val in reason_code_counts.most_common(max_rows)
        ]
        out["unknown_session_count"] = int(session_counts.get("UNKNOWN", 0))
        out["unknown_session_ratio"] = (
            float(out["unknown_session_count"]) / float(total)
            if total
            else 0.0
        )

        session_rows = []
        for session in sorted(session_counts.keys()):
            evals = int(session_counts[session])
            signals = int(session_signal_counts.get(session, 0))
            top_block = session_block_reasons.get(session, Counter()).most_common(1)
            top_block_reason = top_block[0][0] if top_block else ""
            top_nosig = session_no_signal_reasons.get(session, Counter()).most_common(1)
            top_no_signal_reason = top_nosig[0][0] if top_nosig else ""
            session_rows.append(
                {
                    "session": session,
                    "evals": evals,
                    "signals": signals,
                    "signal_rate": (signals / evals) if evals else 0.0,
                    "top_block_reason": top_block_reason,
                    "top_no_signal_reason": top_no_signal_reason,
                }
            )
        out["session_rows"] = session_rows

        if gate_shortfalls:
            out["gate_shortfall_negative_count"] = int(len(gate_shortfalls))
            out["gate_shortfall_negative_avg"] = float(sum(gate_shortfalls) / len(gate_shortfalls))
            out["gate_shortfall_negative_max"] = float(max(gate_shortfalls))

        recommendations = []
        for reason, _count in blocked_reason_counts.most_common(max_rows):
            hint = self._ml_fix_hint(reason)
            if hint and hint not in recommendations:
                recommendations.append(hint)
        if not recommendations and reason_code_counts:
            for reason, _count in reason_code_counts.most_common(max_rows):
                hint = self._ml_fix_hint(reason)
                if hint and hint not in recommendations:
                    recommendations.append(hint)
        out["recommendations"] = recommendations[:max_rows]
        return out

    def _update_group(self, group: dict, pnl: float) -> None:
        group["pnl"] += pnl
        group["trades"] += 1

    def _update_win_loss(self, group: dict, pnl: float) -> None:
        group["pnl"] += pnl
        group["trades"] += 1
        if pnl >= 0:
            group["wins"] += 1
        else:
            group["losses"] += 1

    def _update_streak(self, pnl: float) -> None:
        if pnl < 0:
            self.loss_streak_len += 1
            self.loss_streak_pnl += pnl
            return
        if self.loss_streak_len > 0:
            self.loss_streak_count += 1
            self.loss_streak_len_total += self.loss_streak_len
            self.loss_streak_pnl_total += self.loss_streak_pnl
            if self.loss_streak_len > self.loss_streak_max_len:
                self.loss_streak_max_len = self.loss_streak_len
            if self.loss_streak_pnl < self.loss_streak_max_pnl:
                self.loss_streak_max_pnl = self.loss_streak_pnl
            self.loss_streak_len = 0
            self.loss_streak_pnl = 0.0

    def finalize_streaks(self) -> None:
        if self.loss_streak_len > 0:
            self.loss_streak_count += 1
            self.loss_streak_len_total += self.loss_streak_len
            self.loss_streak_pnl_total += self.loss_streak_pnl
            if self.loss_streak_len > self.loss_streak_max_len:
                self.loss_streak_max_len = self.loss_streak_len
            if self.loss_streak_pnl < self.loss_streak_max_pnl:
                self.loss_streak_max_pnl = self.loss_streak_pnl
            self.loss_streak_len = 0
            self.loss_streak_pnl = 0.0

    def record_trade(self, trade: dict) -> None:
        pnl = trade["pnl_net"]
        strategy = trade.get("strategy", "Unknown")
        sub_strategy = trade.get("sub_strategy")
        session = trade.get("session", "OFF")
        hour = trade.get("entry_time").hour if trade.get("entry_time") else -1
        exit_reason = trade.get("exit_reason", "unknown")
        entry_mode = trade.get("entry_mode", "standard")
        vol_regime = trade.get("vol_regime", "UNKNOWN")
        sl_dist = trade.get("sl_dist", MIN_SL)
        tp_dist = trade.get("tp_dist", MIN_TP)
        rr = tp_dist / sl_dist if sl_dist else 0.0
        mfe = trade.get("mfe_points", 0.0)
        mae = trade.get("mae_points", 0.0)

        self.trades.append(trade)
        self.trade_count += 1
        self.recent_trades.append(trade)
        if pnl >= 0:
            self.win_trade_count += 1
        else:
            self.loss_trade_count += 1
        if str(strategy).startswith("MLPhysics"):
            self.ml_trades.append(trade)

        self._update_win_loss(self.strategy_stats[strategy], pnl)
        if sub_strategy:
            key = f"{strategy}:{sub_strategy}"
            self._update_win_loss(self.sub_strategy_stats[key], pnl)
        self._update_group(self.session_stats[session], pnl)
        self._update_group(self.hour_stats[hour], pnl)
        self._update_group(self.exit_reason_stats[exit_reason], pnl)
        self._update_group(self.entry_mode_stats[entry_mode], pnl)
        self._update_group(self.vol_regime_stats[vol_regime], pnl)
        self._update_group(self.sl_bucket_stats[bucket_label(sl_dist, SL_BUCKETS)], pnl)
        self._update_group(self.tp_bucket_stats[bucket_label(tp_dist, TP_BUCKETS)], pnl)
        self._update_group(self.rr_bucket_stats[bucket_label(rr, RR_BUCKETS)], pnl)

        self.mfe_sum += mfe
        self.mae_sum += mae
        if pnl >= 0:
            self.mfe_win_sum += mfe
            self.mae_win_sum += mae
        else:
            self.mfe_loss_sum += mfe
            self.mae_loss_sum += mae

        self._update_streak(pnl)

    def build_report(self, max_rows: int = 10) -> str:
        def winrate(stats: dict) -> float:
            trades = stats.get("trades", 0)
            wins = stats.get("wins", 0)
            return (wins / trades * 100.0) if trades else 0.0

        def sort_rows(data: dict, key: str = "pnl", reverse: bool = False):
            rows = []
            for name, stats in data.items():
                rows.append((name, stats["pnl"], stats.get("trades", 0), winrate(stats)))
            return sorted(rows, key=lambda r: r[1], reverse=reverse)

        worst_strategies = sort_rows(self.strategy_stats)
        best_strategies = sort_rows(self.strategy_stats, reverse=True)
        sessions = sort_rows(self.session_stats)
        hours = sort_rows(self.hour_stats)
        exit_reasons = sort_rows(self.exit_reason_stats)
        entry_modes = sort_rows(self.entry_mode_stats)
        vol_regimes = sort_rows(self.vol_regime_stats)
        sl_buckets = sort_rows(self.sl_bucket_stats)
        tp_buckets = sort_rows(self.tp_bucket_stats)
        rr_buckets = sort_rows(self.rr_bucket_stats)

        trade_count = int(self.trade_count)
        win_count = int(self.win_trade_count)
        loss_count = int(self.loss_trade_count)

        avg_mfe = self.mfe_sum / trade_count if trade_count else 0.0
        avg_mae = self.mae_sum / trade_count if trade_count else 0.0
        avg_mfe_win = self.mfe_win_sum / max(1, win_count)
        avg_mae_win = self.mae_win_sum / max(1, win_count)
        avg_mfe_loss = self.mfe_loss_sum / max(1, loss_count)
        avg_mae_loss = self.mae_loss_sum / max(1, loss_count)

        loss_avg_len = (self.loss_streak_len_total / self.loss_streak_count) if self.loss_streak_count else 0.0
        loss_avg_pnl = (self.loss_streak_pnl_total / self.loss_streak_count) if self.loss_streak_count else 0.0

        ml_trades = self.ml_trades

        invalid_ml_conf_count = 0

        def _safe_prob(value) -> Optional[float]:
            if value is None:
                return None
            try:
                val = float(value)
            except Exception:
                return None
            if not math.isfinite(val):
                return None
            if 0.0 <= val <= 1.0:
                return float(val)
            return None

        def _ml_session_rows(trades: list[dict]) -> list[tuple]:
            nonlocal invalid_ml_conf_count
            session_stats = defaultdict(
                lambda: {"pnl": 0.0, "trades": 0, "wins": 0, "conf_sum": 0.0, "conf_n": 0}
            )
            for trade in trades:
                session = trade.get("session", "UNKNOWN")
                pnl = float(trade.get("pnl_net", 0.0))
                conf = _safe_prob(trade.get("ml_confidence"))
                if conf is None:
                    conf = _safe_prob(trade.get("ml_prob_up"))
                if conf is None and trade.get("ml_confidence") is not None:
                    invalid_ml_conf_count += 1
                row = session_stats[session]
                row["pnl"] += pnl
                row["trades"] += 1
                if pnl >= 0:
                    row["wins"] += 1
                if conf is not None:
                    row["conf_sum"] += float(conf)
                    row["conf_n"] += 1
            rows = []
            for session, stats in sorted(session_stats.items()):
                trades_count = stats["trades"]
                wins = stats["wins"]
                wr = (wins / trades_count * 100.0) if trades_count else 0.0
                avg_conf = stats["conf_sum"] / stats["conf_n"] if stats["conf_n"] else 0.0
                rows.append(
                    (
                        session,
                        f"{stats['pnl']:.2f}",
                        trades_count,
                        f"{wr:.1f}%",
                        f"{avg_conf:.3f}",
                    )
                )
            return rows

        def _ml_conf_buckets(trades: list[dict]) -> list[tuple]:
            nonlocal invalid_ml_conf_count
            bins = [(0.0, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.01)]
            bucket_stats = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0})
            for trade in trades:
                conf = _safe_prob(trade.get("ml_confidence"))
                if conf is None:
                    conf = _safe_prob(trade.get("ml_prob_up"))
                if conf is None:
                    if trade.get("ml_confidence") is not None:
                        invalid_ml_conf_count += 1
                    continue
                bucket = None
                for lo, hi in bins:
                    if lo <= conf < hi:
                        bucket = f"{lo:.2f}-{hi:.2f}"
                        break
                if bucket is None:
                    bucket = "unknown"
                pnl = float(trade.get("pnl_net", 0.0))
                row = bucket_stats[bucket]
                row["pnl"] += pnl
                row["trades"] += 1
                if pnl >= 0:
                    row["wins"] += 1
            rows = []
            for bucket, stats in bucket_stats.items():
                trades_count = stats["trades"]
                wr = (stats["wins"] / trades_count * 100.0) if trades_count else 0.0
                rows.append((bucket, f"{stats['pnl']:.2f}", trades_count, f"{wr:.1f}%"))
            return rows

        lines = []
        lines.append("Loss Driver Report")
        lines.append("")
        lines.append(f"Avg MFE: {avg_mfe:.2f} | Avg MAE: {avg_mae:.2f}")
        lines.append(f"Avg MFE (wins): {avg_mfe_win:.2f} | Avg MAE (wins): {avg_mae_win:.2f}")
        lines.append(f"Avg MFE (losses): {avg_mfe_loss:.2f} | Avg MAE (losses): {avg_mae_loss:.2f}")
        lines.append(
            "Loss streaks: max_len={} max_pnl={:.2f} avg_len={:.2f} avg_pnl={:.2f} current_len={}".format(
                self.loss_streak_max_len,
                self.loss_streak_max_pnl,
                loss_avg_len,
                loss_avg_pnl,
                self.loss_streak_len,
            )
        )
        lines.append("")
        lines.append(format_rows("Worst Strategies", worst_strategies, ["Strategy", "PnL", "Trades", "Win%"], max_rows))
        lines.append("")
        lines.append(format_rows("Best Strategies", best_strategies, ["Strategy", "PnL", "Trades", "Win%"], max_rows))
        lines.append("")
        lines.append(format_rows("Exit Reasons", exit_reasons, ["Reason", "PnL", "Trades", "Win%"], max_rows))
        lines.append("")
        lines.append(format_rows("Entry Modes", entry_modes, ["Mode", "PnL", "Trades", "Win%"], max_rows))
        lines.append("")
        lines.append(format_rows("Sessions", sessions, ["Session", "PnL", "Trades", "Win%"], max_rows))
        lines.append("")
        lines.append(format_rows("Hours (ET)", hours, ["Hour", "PnL", "Trades", "Win%"], max_rows))
        lines.append("")
        lines.append(format_rows("Volatility Regimes", vol_regimes, ["Regime", "PnL", "Trades", "Win%"], max_rows))
        lines.append("")
        lines.append(format_rows("SL Buckets", sl_buckets, ["SL", "PnL", "Trades", "Win%"], max_rows))
        lines.append("")
        lines.append(format_rows("TP Buckets", tp_buckets, ["TP", "PnL", "Trades", "Win%"], max_rows))
        lines.append("")
        lines.append(format_rows("RR Buckets", rr_buckets, ["RR", "PnL", "Trades", "Win%"], max_rows))
        if ml_trades:
            lines.append("")
            lines.append(
                format_rows(
                    "MLPhysics by Session",
                    _ml_session_rows(ml_trades),
                    ["Session", "PnL", "Trades", "Win%", "AvgConf"],
                    max_rows,
                )
            )
            lines.append("")
            lines.append(
                format_rows(
                    "MLPhysics Confidence Buckets (All Sessions)",
                    _ml_conf_buckets(ml_trades),
                    ["Conf", "PnL", "Trades", "Win%"],
                    max_rows,
                )
            )
            if invalid_ml_conf_count > 0:
                lines.append(f"ML confidence validity: ignored {invalid_ml_conf_count} non-probability values.")
            ny_pm_trades = [trade for trade in ml_trades if trade.get("session") == "NY_PM"]
            if ny_pm_trades:
                lines.append("")
                lines.append(
                    format_rows(
                        "MLPhysics Confidence Buckets (NY_PM)",
                        _ml_conf_buckets(ny_pm_trades),
                        ["Conf", "PnL", "Trades", "Win%"],
                        max_rows,
                    )
                )
        ml_diag_summary = self.summarize_ml_diagnostics(max_rows=max_rows)
        if ml_diag_summary.get("total_evals", 0):
            total_evals = int(ml_diag_summary.get("total_evals", 0))
            signal_count = int(ml_diag_summary.get("signal_count", 0))
            blocked_count = int(ml_diag_summary.get("blocked_count", 0))
            no_signal_count = int(ml_diag_summary.get("no_signal_count", 0))
            signal_rate = float(ml_diag_summary.get("signal_rate", 0.0) or 0.0)

            lines.append("")
            lines.append("MLPhysics Eval Diagnostics")
            lines.append(
                f"Evaluations: {total_evals} | Signals: {signal_count} ({signal_rate:.1%}) "
                f"| Blocked: {blocked_count} | NoSignal: {no_signal_count}"
            )

            neg_count = int(ml_diag_summary.get("gate_shortfall_negative_count", 0))
            if neg_count > 0:
                lines.append(
                    "Gate shortfall (p<threshold): "
                    f"count={neg_count} avg={float(ml_diag_summary.get('gate_shortfall_negative_avg', 0.0)):.3f} "
                    f"max={float(ml_diag_summary.get('gate_shortfall_negative_max', 0.0)):.3f}"
                )

            decision_counts = ml_diag_summary.get("decision_counts", {}) or {}
            decision_rows = []
            for decision, count in sorted(decision_counts.items(), key=lambda kv: kv[1], reverse=True):
                share = (float(count) / float(total_evals)) if total_evals else 0.0
                decision_rows.append((decision, int(count), f"{share:.1%}"))
            lines.append("")
            lines.append(
                format_rows(
                    "ML Eval Decisions",
                    decision_rows,
                    ["Decision", "Count", "Share"],
                    max_rows,
                )
            )

            blocked_reason_counts = ml_diag_summary.get("blocked_reason_counts", {}) or {}
            blocked_rows = []
            denom = max(1, blocked_count)
            for reason, count in sorted(blocked_reason_counts.items(), key=lambda kv: kv[1], reverse=True):
                blocked_rows.append((reason, int(count), f"{(float(count) / float(denom)):.1%}"))
            lines.append("")
            lines.append(
                format_rows(
                    "ML Block Reasons",
                    blocked_rows,
                    ["Reason", "Count", "ShareOfBlocked"],
                    max_rows,
                )
            )

            no_signal_reason_counts = ml_diag_summary.get("no_signal_reason_counts", {}) or {}
            no_signal_rows = []
            no_signal_denom = max(1, int(ml_diag_summary.get("no_signal_count", 0) or 0))
            for reason, count in sorted(no_signal_reason_counts.items(), key=lambda kv: kv[1], reverse=True):
                no_signal_rows.append((reason, int(count), f"{(float(count) / float(no_signal_denom)):.1%}"))
            lines.append("")
            lines.append(
                format_rows(
                    "ML No-Signal Reasons",
                    no_signal_rows,
                    ["Reason", "Count", "ShareOfNoSignal"],
                    max_rows,
                )
            )

            reason_code_rows = []
            for item in (ml_diag_summary.get("top_reason_codes", []) or [])[:max_rows]:
                reason_code_rows.append((item.get("reason_code", ""), int(item.get("count", 0))))
            lines.append("")
            lines.append(
                format_rows(
                    "ML Top Reason Codes",
                    reason_code_rows,
                    ["ReasonCode", "Count"],
                    max_rows,
                )
            )

            session_rows = []
            for row in (ml_diag_summary.get("session_rows", []) or [])[:max_rows]:
                session_rows.append(
                    (
                        row.get("session", ""),
                        int(row.get("evals", 0)),
                        int(row.get("signals", 0)),
                        f"{float(row.get('signal_rate', 0.0) or 0.0):.1%}",
                        row.get("top_block_reason", ""),
                        row.get("top_no_signal_reason", ""),
                    )
                )
            lines.append("")
            lines.append(
                format_rows(
                    "ML Session Coverage",
                    session_rows,
                    ["Session", "Evals", "Signals", "SignalRate", "TopBlockReason", "TopNoSignalReason"],
                    max_rows,
                )
            )
            unknown_count = int(ml_diag_summary.get("unknown_session_count", 0) or 0)
            unknown_ratio = float(ml_diag_summary.get("unknown_session_ratio", 0.0) or 0.0)
            lines.append(f"UNKNOWN session share: {unknown_count}/{total_evals} ({unknown_ratio:.1%})")

            recommendations = list(ml_diag_summary.get("recommendations", []) or [])
            if recommendations:
                lines.append("")
                lines.append("ML Fix Hints")
                for tip in recommendations[:max_rows]:
                    lines.append(f"  - {tip}")
        lines.append("")
        lines.append(format_rows("Filter Blocks", self.filter_blocks.most_common(max_rows), ["Filter", "Count"], max_rows))
        lines.append("")
        lines.append(format_rows("Rescue Triggers", self.filter_rescues.most_common(max_rows), ["Filter", "Count"], max_rows))
        lines.append("")
        lines.append(format_rows("Rescue Bypasses", self.filter_bypasses.most_common(max_rows), ["Filter", "Count"], max_rows))
        return "\n".join(lines)


def configure_risk() -> None:
    risk_cfg = CONFIG.setdefault("RISK", {})
    risk_cfg["POINT_VALUE"] = POINT_VALUE
    risk_cfg["CONTRACTS"] = CONTRACTS
    risk_cfg["FEES_PER_SIDE"] = FEES_PER_20_CONTRACTS / 20.0 / 2.0


def resample_dataframe(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    tf_code = f"{timeframe_minutes}min"
    return df.resample(tf_code).agg(agg_dict).dropna()


class ResampleCache:
    def __init__(self, df: pd.DataFrame, timeframe_minutes: int):
        self.df = df
        self.timeframe_minutes = timeframe_minutes
        self.freq = f"{timeframe_minutes}min"
        self.resampled_full = resample_dataframe(df, timeframe_minutes)
        self.index_to_pos = {ts: i for i, ts in enumerate(self.resampled_full.index)}
        self._cursor_pos = 0
        self._last_processed_ts = None
        self._current_bin_start = None
        self._current_open = None
        self._current_high = None
        self._current_low = None
        self._current_close = None
        self._current_volume = None

    def _reset_partial_state(self) -> None:
        self._cursor_pos = 0
        self._last_processed_ts = None
        self._current_bin_start = None
        self._current_open = None
        self._current_high = None
        self._current_low = None
        self._current_close = None
        self._current_volume = None

    def _advance_to(self, current_time: pd.Timestamp) -> bool:
        if self.df.empty:
            return False

        if self._last_processed_ts is not None and current_time < self._last_processed_ts:
            # Non-monotonic query: rebuild incremental state safely.
            self._reset_partial_state()

        end_pos = int(self.df.index.searchsorted(current_time, side="right")) - 1
        if end_pos < 0:
            return False

        if self._cursor_pos > end_pos:
            return self._current_bin_start is not None

        new_rows = self.df.iloc[self._cursor_pos : end_pos + 1]
        has_volume = "volume" in self.df.columns
        for row in new_rows.itertuples():
            ts = row.Index
            bin_start = pd.Timestamp(ts).floor(self.freq)
            o = float(row.open)
            h = float(row.high)
            l = float(row.low)
            c = float(row.close)
            v = float(getattr(row, "volume", 0.0)) if has_volume else 0.0

            if self._current_bin_start is None or bin_start != self._current_bin_start:
                self._current_bin_start = bin_start
                self._current_open = o
                self._current_high = h
                self._current_low = l
                self._current_close = c
                self._current_volume = v
            else:
                self._current_high = max(float(self._current_high), h)
                self._current_low = min(float(self._current_low), l)
                self._current_close = c
                self._current_volume = float(self._current_volume) + v

        self._cursor_pos = end_pos + 1
        self._last_processed_ts = self.df.index[end_pos]
        return self._current_bin_start is not None

    def _partial_bar(self, current_time: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
        bin_start = pd.Timestamp(current_time).floor(self.freq)
        if not self._advance_to(current_time):
            return pd.DataFrame(), bin_start
        partial_row = pd.DataFrame(
            {
                "open": [float(self._current_open)],
                "high": [float(self._current_high)],
                "low": [float(self._current_low)],
                "close": [float(self._current_close)],
                "volume": [float(self._current_volume)],
            },
            index=[self._current_bin_start],
        )
        return partial_row, self._current_bin_start

    def get_recent(self, current_time: pd.Timestamp, lookback: int) -> pd.DataFrame:
        partial_row, bin_start = self._partial_bar(current_time)
        if partial_row.empty:
            return partial_row
        if lookback <= 1:
            return partial_row
        pos = self.index_to_pos.get(bin_start)
        if pos is None:
            prev = self.resampled_full[self.resampled_full.index < bin_start].tail(lookback - 1)
        else:
            start = max(0, pos - (lookback - 1))
            prev = self.resampled_full.iloc[start:pos]
        if prev.empty:
            return partial_row
        return pd.concat([prev, partial_row])

    def get_full(self, current_time: pd.Timestamp) -> pd.DataFrame:
        partial_row, bin_start = self._partial_bar(current_time)
        if partial_row.empty:
            return partial_row
        pos = self.index_to_pos.get(bin_start)
        if pos is None:
            prev = self.resampled_full[self.resampled_full.index < bin_start]
        else:
            prev = self.resampled_full.iloc[:pos]
        if prev.empty:
            return partial_row
        return pd.concat([prev, partial_row])


class FlipConfidenceTracker:
    def __init__(self, full_df: pd.DataFrame, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.full_df = full_df
        self.allowed_filters = {
            str(item)
            for item in (self.cfg.get("allowed_filters") or [])
            if item is not None
        }
        self.key_fields = list(self.cfg.get("key_fields") or ["filter", "session", "side"])
        self.max_horizon = _coerce_int(self.cfg.get("max_horizon_bars", 120), 120)
        self.exit_at_horizon = str(self.cfg.get("exit_at_horizon", "close") or "close").lower()
        self.assume_sl_first = bool(self.cfg.get("assume_sl_first", True))
        self.min_total_trades = _coerce_int(self.cfg.get("min_total_trades", 0), 0)
        self.min_fold_trades = _coerce_int(self.cfg.get("min_fold_trades", 1), 1)
        self.min_avg_pnl_points = _coerce_float(self.cfg.get("min_avg_pnl_points", 0.0), 0.0)
        self.min_win_rate = _coerce_float(self.cfg.get("min_win_rate", 0.0), 0.0)
        self.min_fold_expectancy_points = _coerce_float(
            self.cfg.get("min_fold_expectancy_points", 0.0), 0.0
        )
        self.min_folds = _coerce_int(self.cfg.get("min_folds", 1), 1)
        self.min_positive_fold_ratio = _coerce_float(
            self.cfg.get("min_positive_fold_ratio", 0.0), 0.0
        )
        self.fold_mode = str(self.cfg.get("fold_mode", "regime") or "regime").lower()
        self.fold_count = _coerce_int(self.cfg.get("folds", 4), 4)
        self.loro_regimes = [
            str(item).lower()
            for item in (self.cfg.get("loro_regimes") or ["low", "normal", "high"])
            if item is not None
        ]
        self.base_ts = 0
        self.span_ts = 1
        if not full_df.empty:
            self.base_ts = full_df.index[0].value
            self.span_ts = max(1, full_df.index[-1].value - self.base_ts)
        self.stats = defaultdict(
            lambda: {
                "total_trades": 0,
                "total_pnl_points": 0.0,
                "wins": 0,
                "per_regime": defaultdict(lambda: {"trades": 0, "pnl_points": 0.0, "wins": 0}),
                "fold_stats": defaultdict(lambda: {"trades": 0, "pnl_points": 0.0}),
            }
        )
        self.skipped = Counter()
        self._data_len = int(len(full_df))
        self._open_arr = full_df["open"].to_numpy(dtype=float, copy=False) if "open" in full_df.columns else np.array([])
        self._high_arr = full_df["high"].to_numpy(dtype=float, copy=False) if "high" in full_df.columns else np.array([])
        self._low_arr = full_df["low"].to_numpy(dtype=float, copy=False) if "low" in full_df.columns else np.array([])
        self._close_arr = full_df["close"].to_numpy(dtype=float, copy=False) if "close" in full_df.columns else np.array([])
        self._sim_cache_max = max(1000, _coerce_int(self.cfg.get("sim_cache_size", 50000), 50000))
        self._sim_cache: OrderedDict[tuple, float] = OrderedDict()

    def _simulate_trade_points_fast(
        self,
        entry_pos: int,
        side: str,
        entry_price: float,
        sl_dist: float,
        tp_dist: float,
    ) -> float:
        if self._data_len <= 0:
            return 0.0
        last_pos = min(self._data_len - 1, entry_pos + self.max_horizon)
        highs = self._high_arr[entry_pos : last_pos + 1]
        lows = self._low_arr[entry_pos : last_pos + 1]

        if side == "LONG":
            tp_level = entry_price + tp_dist
            sl_level = entry_price - sl_dist
            tp_hits = np.flatnonzero(highs >= tp_level)
            sl_hits = np.flatnonzero(lows <= sl_level)
        else:
            tp_level = entry_price - tp_dist
            sl_level = entry_price + sl_dist
            tp_hits = np.flatnonzero(lows <= tp_level)
            sl_hits = np.flatnonzero(highs >= sl_level)

        first_tp = int(tp_hits[0]) if tp_hits.size else None
        first_sl = int(sl_hits[0]) if sl_hits.size else None

        if first_tp is not None and first_sl is not None:
            if first_tp == first_sl:
                return -sl_dist if self.assume_sl_first else tp_dist
            if first_tp < first_sl:
                return tp_dist
            return -sl_dist
        if first_tp is not None:
            return tp_dist
        if first_sl is not None:
            return -sl_dist

        if self.exit_at_horizon == "close":
            exit_price = float(self._close_arr[last_pos])
            return compute_pnl_points(side, entry_price, exit_price)
        return 0.0

    def _simulate_trade_points_cached(
        self,
        entry_pos: int,
        side: str,
        entry_price: float,
        sl_dist: float,
        tp_dist: float,
    ) -> float:
        key = (
            int(entry_pos),
            str(side),
            round(float(sl_dist), 6),
            round(float(tp_dist), 6),
        )
        cached = self._sim_cache.get(key)
        if cached is not None:
            self._sim_cache.move_to_end(key)
            return float(cached)
        pnl = self._simulate_trade_points_fast(entry_pos, side, entry_price, sl_dist, tp_dist)
        self._sim_cache[key] = float(pnl)
        if len(self._sim_cache) > self._sim_cache_max:
            self._sim_cache.popitem(last=False)
        return float(pnl)

    def _field_value(
        self,
        field: str,
        filter_name: str,
        session_name: Optional[str],
        regime: Optional[str],
        side: str,
        signal: dict,
    ) -> str:
        if field == "filter":
            return filter_name
        if field == "session":
            return session_name or "UNKNOWN"
        if field == "regime":
            return regime or "UNKNOWN"
        if field == "side":
            return side
        if field == "strategy":
            return str(signal.get("strategy") or "Unknown")
        if field == "sub_strategy":
            return str(signal.get("sub_strategy") or "None")
        return str(signal.get(field) or "UNKNOWN")

    def _build_key(
        self,
        filter_name: str,
        session_name: Optional[str],
        regime: Optional[str],
        side: str,
        signal: dict,
    ) -> str:
        return "|".join(
            self._field_value(field, filter_name, session_name, regime, side, signal)
            for field in self.key_fields
        )

    def _fold_index(self, current_time: dt.datetime) -> int:
        if self.fold_count <= 1:
            return 0
        ts_val = current_time.value if current_time is not None else self.base_ts
        return int(((ts_val - self.base_ts) / self.span_ts) * max(1, self.fold_count - 1))

    def record_block(
        self,
        filter_name: str,
        signal: Optional[dict],
        bar_index: int,
        current_time: dt.datetime,
        history_df: pd.DataFrame,
        session_name: Optional[str],
        vol_regime: Optional[str] = None,
    ) -> None:
        if not self.enabled:
            return
        if not filter_name:
            return
        if self.allowed_filters and filter_name not in self.allowed_filters:
            return
        if not signal:
            self.skipped["missing_signal"] += 1
            return
        side = str(signal.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            self.skipped["invalid_side"] += 1
            return
        sl_dist = _coerce_float(signal.get("sl_dist", 0.0), 0.0)
        tp_dist = _coerce_float(signal.get("tp_dist", 0.0), 0.0)
        if sl_dist <= 0 or tp_dist <= 0:
            self.skipped["invalid_sltp"] += 1
            return
        entry_pos = bar_index + 1
        if entry_pos >= self._data_len:
            self.skipped["entry_oob"] += 1
            return
        try:
            entry_price = float(self._open_arr[entry_pos])
        except Exception:
            self.skipped["entry_price"] += 1
            return
        if vol_regime is None:
            try:
                vol_regime, _, _ = volatility_filter.get_regime(history_df)
            except Exception:
                vol_regime = None
        regime_key = str(vol_regime).lower() if vol_regime else "unknown"
        flip_side = "SHORT" if side == "LONG" else "LONG"
        pnl_points = self._simulate_trade_points_cached(
            entry_pos=entry_pos,
            side=flip_side,
            entry_price=entry_price,
            sl_dist=sl_dist,
            tp_dist=tp_dist,
        )
        key = self._build_key(filter_name, session_name, regime_key, side, signal)
        agg = self.stats[key]
        agg["total_trades"] += 1
        agg["total_pnl_points"] += pnl_points
        if pnl_points > 0:
            agg["wins"] += 1
        reg_stats = agg["per_regime"][regime_key]
        reg_stats["trades"] += 1
        reg_stats["pnl_points"] += pnl_points
        if pnl_points > 0:
            reg_stats["wins"] += 1
        fold_idx = self._fold_index(current_time)
        fold_stats = agg["fold_stats"][fold_idx]
        fold_stats["trades"] += 1
        fold_stats["pnl_points"] += pnl_points

    def finalize(self) -> dict:
        if not self.enabled:
            return {}
        allowlist = set()
        stats_out = {}
        for key, agg in self.stats.items():
            total_trades = agg["total_trades"]
            total_pnl_points = agg["total_pnl_points"]
            wins = agg["wins"]
            avg_pnl = total_pnl_points / total_trades if total_trades else 0.0
            win_rate = wins / total_trades if total_trades else 0.0

            folds_regime = 0
            positive_regime = 0
            folds_time = 0
            positive_time = 0
            if self.fold_mode in ("regime", "regime_time"):
                regimes = self.loro_regimes or list(agg["per_regime"].keys())
                for regime in regimes:
                    stats = agg["per_regime"].get(regime)
                    if not stats:
                        continue
                    trades = stats["trades"]
                    pnl_points = stats["pnl_points"]
                    if trades >= self.min_fold_trades:
                        folds_regime += 1
                        expectancy = pnl_points / trades if trades else 0.0
                        if expectancy >= self.min_fold_expectancy_points:
                            positive_regime += 1
            if self.fold_mode in ("time", "regime_time"):
                for stats in agg["fold_stats"].values():
                    trades = stats["trades"]
                    pnl_points = stats["pnl_points"]
                    if trades >= self.min_fold_trades:
                        folds_time += 1
                        expectancy = pnl_points / trades if trades else 0.0
                        if expectancy >= self.min_fold_expectancy_points:
                            positive_time += 1
            regime_ratio = (positive_regime / folds_regime) if folds_regime else 0.0
            time_ratio = (positive_time / folds_time) if folds_time else 0.0

            if self.fold_mode == "regime":
                folds_used = folds_regime
                positive_ratio = regime_ratio
                fold_ok = folds_used >= self.min_folds and positive_ratio >= self.min_positive_fold_ratio
            elif self.fold_mode == "time":
                folds_used = folds_time
                positive_ratio = time_ratio
                fold_ok = folds_used >= self.min_folds and positive_ratio >= self.min_positive_fold_ratio
            elif self.fold_mode == "regime_time":
                folds_used = min(folds_regime, folds_time)
                positive_ratio = min(regime_ratio, time_ratio)
                fold_ok = (
                    folds_regime >= self.min_folds
                    and regime_ratio >= self.min_positive_fold_ratio
                    and folds_time >= self.min_folds
                    and time_ratio >= self.min_positive_fold_ratio
                )
            else:
                folds_used = 1
                positive_ratio = 1 if avg_pnl >= self.min_fold_expectancy_points else 0
                fold_ok = True

            allowed = (
                total_trades >= self.min_total_trades
                and avg_pnl >= self.min_avg_pnl_points
                and win_rate >= self.min_win_rate
                and fold_ok
            )

            stats_out[key] = {
                "total_trades": total_trades,
                "avg_pnl_points": avg_pnl,
                "win_rate": win_rate,
                "folds": folds_used,
                "positive_ratio": positive_ratio,
                "folds_regime": folds_regime,
                "positive_ratio_regime": regime_ratio,
                "folds_time": folds_time,
                "positive_ratio_time": time_ratio,
                "allowed": allowed,
                "per_regime": {k: dict(v) for k, v in agg["per_regime"].items()},
            }
            if allowed:
                allowlist.add(key)

        payload = {
            "generated_at": dt.datetime.now(NY_TZ).isoformat(),
            "summary": {
                "keys_seen": len(stats_out),
                "keys_allowed": len(allowlist),
                "candidates": sum(v["total_trades"] for v in stats_out.values()),
                "skipped": dict(self.skipped),
            },
            "criteria": {
                "min_total_trades": self.min_total_trades,
                "min_fold_trades": self.min_fold_trades,
                "min_avg_pnl_points": self.min_avg_pnl_points,
                "min_win_rate": self.min_win_rate,
                "min_fold_expectancy_points": self.min_fold_expectancy_points,
                "min_folds": self.min_folds,
                "min_positive_fold_ratio": self.min_positive_fold_ratio,
                "fold_mode": self.fold_mode,
                "folds": self.fold_count,
                "loro_regimes": self.loro_regimes,
            },
            "key_fields": self.key_fields,
            "allowed_filters": sorted(self.allowed_filters) if self.allowed_filters else [],
            "allowlist": sorted(allowlist),
            "stats": stats_out,
        }
        cache_file = self.cfg.get("cache_file")
        if cache_file:
            cache_path = Path(cache_file)
            if not cache_path.is_absolute():
                cache_path = Path(__file__).resolve().parent / cache_path
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return payload


def parse_user_datetime(value: str, tz: ZoneInfo, is_end: bool = False) -> dt.datetime:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if " " in raw and "T" not in raw:
        raw = raw.replace(" ", "T")
    if len(raw) == 10:
        parsed = dt.datetime.fromisoformat(raw)
        if is_end:
            parsed = parsed + dt.timedelta(days=1) - dt.timedelta(microseconds=1)
    else:
        parsed = dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz)
    return parsed.astimezone(tz)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["ts_event", "open", "high", "low", "close", "volume", "symbol"],
    )
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_event"])
    df = df.rename(columns={"ts_event": "ts"})
    df["ts"] = df["ts"].dt.tz_convert(NY_TZ)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _normalize_cached_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        ts_col = None
        if "ts" in df.columns:
            ts_col = "ts"
        elif "ts_event" in df.columns:
            ts_col = "ts_event"
        if ts_col is None:
            raise ValueError("Cached data missing timestamp column.")
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col])
        df = df.set_index(ts_col)
    if df.index.tz is None:
        df.index = df.index.tz_localize(NY_TZ)
    else:
        df.index = df.index.tz_convert(NY_TZ)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def load_csv_cached(
    path: Path,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    source_key = cache_key_for_source(path)
    source_label = path.name
    source_path = str(path.resolve())
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        df = _normalize_cached_bars(df)
        return attach_backtest_symbol_context(
            df,
            None,
            None,
            source_key=source_key,
            source_label=source_label,
            source_path=source_path,
        )

    cache_path = None
    if cache_dir and use_cache:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = cache_key_for_source(path)
        cache_path = cache_dir / f"backtest_{key}.parquet"
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                logging.info("Backtest cache hit: %s", cache_path.name)
                df = _normalize_cached_bars(df)
                return attach_backtest_symbol_context(
                    df,
                    None,
                    None,
                    source_key=source_key,
                    source_label=source_label,
                    source_path=source_path,
                )
            except Exception as exc:
                logging.warning("Backtest cache read failed (%s). Falling back to CSV.", exc)

    df = load_csv(path)
    if cache_path and use_cache:
        try:
            df.to_parquet(cache_path, index=True)
            logging.info("Backtest cache write: %s", cache_path.name)
        except Exception as exc:
            logging.warning("Backtest cache write failed (%s).", exc)
    return attach_backtest_symbol_context(
        df,
        None,
        None,
        source_key=source_key,
        source_label=source_label,
        source_path=source_path,
    )


def infer_bar_minutes(index: pd.Index, sample: int = 5000) -> Optional[int]:
    if index is None or len(index) < 2:
        return None
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    idx = index[:sample]
    diffs = np.diff(idx.values.astype("datetime64[ns]").astype("int64"))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    median_sec = float(np.median(diffs)) / 1e9
    minutes = max(1, int(round(median_sec / 60.0)))
    return minutes


def normalize_index(df: pd.DataFrame, tz: ZoneInfo) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)
    return df.sort_index()


def slice_df_upto(df: pd.DataFrame, current_time: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    pos = df.index.searchsorted(current_time, side="right")
    if pos <= 0:
        return df.iloc[:0]
    return df.iloc[:pos]


def choose_symbol(df: pd.DataFrame, preferred: Optional[str]) -> str:
    if "symbol" not in df.columns:
        raise KeyError("choose_symbol requires a 'symbol' column")
    work = _filter_backtest_outright_symbols(df)
    if preferred:
        symbols = set(work["symbol"].dropna().unique())
        if preferred in symbols:
            return preferred
    counts = work["symbol"].value_counts()
    return counts.index[0]


def _filter_backtest_outright_symbols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "symbol" not in df.columns:
        return df
    symbol_text = df["symbol"].astype(str)
    # Keep outright futures symbols and ignore calendar spreads like ESM4-ESU4.
    outright_mask = ~symbol_text.str.contains(r"[-+/]", regex=True, na=False)
    if bool(outright_mask.any()):
        filtered = df.loc[outright_mask]
        if not filtered.empty:
            return filtered
    return df


def _auto_select_symbol_by_day(
    df: pd.DataFrame, method: str = "volume"
) -> tuple[pd.DataFrame, dict]:
    if df.empty or "symbol" not in df.columns:
        return df, {}
    filtered_df = _filter_backtest_outright_symbols(df)
    work = filtered_df[["symbol", "volume"]].copy()
    work["date"] = work.index.date
    method_key = str(method or "volume").lower()
    if method_key == "rows" or work["volume"].isna().all():
        stats = work.groupby(["date", "symbol"]).size().rename("score").reset_index()
    else:
        stats = (
            work.groupby(["date", "symbol"])["volume"]
            .sum(min_count=1)
            .fillna(0.0)
            .rename("score")
            .reset_index()
        )
    stats = stats.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
    best = stats.drop_duplicates("date")
    day_to_symbol = dict(zip(best["date"], best["symbol"]))
    date_series = pd.Series(filtered_df.index.date, index=filtered_df.index)
    chosen = date_series.map(day_to_symbol)
    mask = filtered_df["symbol"].astype(str) == chosen.astype(str)
    return filtered_df.loc[mask], day_to_symbol


def apply_symbol_mode(
    df: pd.DataFrame,
    mode: str,
    method: str,
) -> tuple[pd.DataFrame, str, dict]:
    if df.empty or "symbol" not in df.columns:
        return df, "AUTO", {}
    unique_symbols = df["symbol"].nunique(dropna=True)
    if unique_symbols <= 1:
        symbol = str(df["symbol"].dropna().iloc[0]) if unique_symbols else "AUTO"
        return df, symbol, {}
    mode_key = str(mode or "single").lower()
    if mode_key in ("auto", "auto_by_day", "roll"):
        filtered, mapping = _auto_select_symbol_by_day(df, method=method)
        return filtered, "AUTO_BY_DAY", mapping
    return df, "AUTO", {}


def attach_backtest_symbol_context(
    df: pd.DataFrame,
    selected_symbol: Optional[str],
    symbol_mode: Optional[str],
    *,
    source_key: Optional[str] = None,
    source_label: Optional[str] = None,
    source_path: Optional[str] = None,
) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df
    try:
        if selected_symbol:
            df.attrs["selected_symbol"] = str(selected_symbol).strip()
        else:
            df.attrs.pop("selected_symbol", None)
        if symbol_mode:
            df.attrs["selected_symbol_mode"] = str(symbol_mode).strip().lower()
        else:
            df.attrs.pop("selected_symbol_mode", None)
        if source_key is not None:
            source_key_text = str(source_key).strip()
            if source_key_text:
                df.attrs["source_cache_key"] = source_key_text
            else:
                df.attrs.pop("source_cache_key", None)
        if source_label is not None:
            source_label_text = str(source_label).strip()
            if source_label_text:
                df.attrs["source_label"] = source_label_text
            else:
                df.attrs.pop("source_label", None)
        if source_path is not None:
            source_path_text = str(source_path).strip()
            if source_path_text:
                df.attrs["source_path"] = source_path_text
            else:
                df.attrs.pop("source_path", None)
    except Exception:
        pass
    return df


class BacktestClient:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_market_data(self, lookback_minutes: int = 1000, force_fetch: bool = False) -> pd.DataFrame:
        del force_fetch
        if self.df.empty:
            return self.df
        return self.df.tail(lookback_minutes)

    def fetch_custom_bars(self, lookback_bars: int, minutes_per_bar: int) -> pd.DataFrame:
        df = resample_dataframe(self.df, minutes_per_bar)
        if df.empty:
            return df
        return df.tail(lookback_bars)


class BacktestHTFFVGFilter(HTFFVGFilter):
    def check_signal_blocked(
        self,
        signal,
        current_price,
        df_1h=None,
        df_4h=None,
        tp_dist=None,
        current_time=None,
    ):
        if df_1h is not None and not df_1h.empty:
            fvgs_1h = self._scan_for_new_fvgs(df_1h, "1H")
            self._update_memory(fvgs_1h)

        if df_4h is not None and not df_4h.empty:
            fvgs_4h = self._scan_for_new_fvgs(df_4h, "4H")
            self._update_memory(fvgs_4h)

        if current_time is None:
            current_time = dt.datetime.now(NY_TZ)

        self._clean_memory(current_price, current_time)

        if not self.memory:
            return False, None

        signal = signal.upper()
        min_room_needed = (tp_dist * 0.40) if tp_dist else 10.0

        if signal in ["BUY", "LONG"]:
            for fvg in self.memory:
                if fvg["type"] == "bearish" and current_price < fvg["top"]:
                    dist = fvg["bottom"] - current_price
                    if dist < min_room_needed:
                        return (
                            True,
                            (
                                "Blocked LONG: Bearish "
                                f"{fvg['tf']} FVG overhead @ {fvg['bottom']:.2f} "
                                f"(Dist: {dist:.2f} < {min_room_needed:.2f})"
                            ),
                        )

        if signal in ["SELL", "SHORT"]:
            for fvg in self.memory:
                if fvg["type"] == "bullish" and current_price > fvg["bottom"]:
                    dist = current_price - fvg["top"]
                    if dist < min_room_needed:
                        return (
                            True,
                            (
                                "Blocked SHORT: Bullish "
                                f"{fvg['tf']} FVG below @ {fvg['top']:.2f} "
                                f"(Dist: {dist:.2f} < {min_room_needed:.2f})"
                            ),
                        )

        return False, None


class BacktestNewsFilter(NewsFilter):
    def refresh_calendar(self):
        self.calendar_blackouts = []
        self.recent_events = []


class ContinuationRescueManager:
    def __init__(self):
        self.configs = STRATEGY_CONFIGS
        self.strategy_instances = {}
        self.ny_tz = ZoneInfo("America/New_York")

    def _structure_break_signal(
        self,
        df: pd.DataFrame,
        current_time,
        required_side: str,
        current_price: Optional[float],
        trend_day_series: Optional[dict],
    ) -> Optional[dict]:
        if df.empty or trend_day_series is None:
            return None

        if current_price is None:
            try:
                current_price = float(df.iloc[-1]["close"])
            except Exception:
                return None

        prev_close = None
        if len(df) > 1:
            try:
                prev_close = float(df.iloc[-2]["close"])
            except Exception:
                prev_close = None

        def series_value(key: str, default):
            series = trend_day_series.get(key)
            if isinstance(series, pd.Series):
                try:
                    return series.iloc[-1]
                except Exception:
                    return default
            return series if series is not None else default

        prior_high = series_value("prior_session_high", None)
        prior_low = series_value("prior_session_low", None)

        structure_up = False
        structure_down = False
        if prior_high is not None and not pd.isna(prior_high):
            structure_up = current_price > float(prior_high)
            if prev_close is not None:
                structure_up = structure_up and prev_close <= float(prior_high)
        if prior_low is not None and not pd.isna(prior_low):
            structure_down = current_price < float(prior_low)
            if prev_close is not None:
                structure_down = structure_down and prev_close >= float(prior_low)

        if required_side == "LONG" and not structure_up:
            return None
        if required_side == "SHORT" and not structure_down:
            return None

        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=dt.timezone.utc)
        ny_time = current_time.astimezone(self.ny_tz)

        tp_dist = 6.0
        sl_dist = 4.0
        try:
            sltp = dynamic_sltp_engine.calculate_sltp("Continuation", df, ts=ny_time)
            tp_dist = float(sltp.get("tp_dist", tp_dist))
            sl_dist = float(sltp.get("sl_dist", sl_dist))
        except Exception:
            pass
        return {
            "strategy": "Continuation_Structure",
            "side": required_side,
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "size": CONTRACTS,
            "rescued": True,
        }

    def get_active_continuation_signal(
        self,
        df: pd.DataFrame,
        current_time,
        required_side: str,
        current_price: Optional[float] = None,
        trend_day_series: Optional[dict] = None,
        signal_mode: Optional[str] = None,
    ):
        if not CONFIG.get("CONTINUATION_ENABLED", True):
            return None
        if df.empty:
            return None

        mode = str(
            signal_mode or CONFIG.get("BACKTEST_CONTINUATION_SIGNAL_MODE", "calendar") or "calendar"
        ).lower()
        if mode == "structure":
            return self._structure_break_signal(
                df, current_time, required_side, current_price, trend_day_series
            )

        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=dt.timezone.utc)
        ny_time = current_time.astimezone(self.ny_tz)

        quarter = (ny_time.month - 1) // 3 + 1
        week = ny_time.isocalendar().week
        day = ny_time.weekday() + 1
        hour = ny_time.hour

        if 18 <= hour or hour < 3:
            session = "Asia"
        elif 3 <= hour < 8:
            session = "London"
        elif 8 <= hour < 17:
            session = "NY"
        else:
            session = "Other"

        candidate_key = f"Q{quarter}_W{week}_D{day}_{session}"
        if candidate_key not in self.configs:
            return None

        if candidate_key not in self.strategy_instances:
            try:
                self.strategy_instances[candidate_key] = FractalSweepStrategy(candidate_key)
            except ValueError:
                return None

        strat = self.strategy_instances[candidate_key]
        try:
            signals_df = strat.generate_signals(df)
            if signals_df.empty:
                return None

            last_sig_time = signals_df.index[-1]
            if last_sig_time.tzinfo is None:
                last_sig_time = last_sig_time.replace(tzinfo=dt.timezone.utc)
            else:
                last_sig_time = last_sig_time.astimezone(dt.timezone.utc)

            check_time = current_time.astimezone(dt.timezone.utc)
            if last_sig_time == check_time:
                tp_dist = strat.target if hasattr(strat, "target") else 6.0
                sl_dist = strat.stop if hasattr(strat, "stop") else 4.0
                try:
                    sltp = dynamic_sltp_engine.calculate_sltp("Continuation", df, ts=ny_time)
                    tp_dist = float(sltp.get("tp_dist", tp_dist))
                    sl_dist = float(sltp.get("sl_dist", sl_dist))
                except Exception:
                    pass
                return {
                    "strategy": f"Continuation_{candidate_key}",
                    "side": required_side,
                    "tp_dist": tp_dist,
                    "sl_dist": sl_dist,
                    "size": CONTRACTS,
                    "rescued": True,
                }
        except Exception:
            return None

        return None


def _multiplier_strategy_excluded(signal: dict, strategy_hint: Optional[str] = None) -> bool:
    cfg = CONFIG.get("BACKTEST_GEMINI_MULTIPLIERS", {}) or {}
    raw_prefixes = cfg.get("disabled_strategy_prefixes", [])
    if not isinstance(raw_prefixes, (list, tuple, set)):
        return False
    prefixes = [
        str(p).strip().lower()
        for p in raw_prefixes
        if str(p).strip()
    ]
    if not prefixes:
        return False

    names: list[str] = []
    if strategy_hint is not None:
        hint = str(strategy_hint).strip()
        if hint:
            names.append(hint)
    sig_name = str(signal.get("strategy", "") or "").strip()
    if sig_name:
        names.append(sig_name)

    for name in names:
        name_lc = name.lower()
        for prefix in prefixes:
            if name_lc.startswith(prefix):
                return True
    return False


def apply_multipliers(signal: dict, strategy_hint: Optional[str] = None) -> None:
    if _multiplier_strategy_excluded(signal, strategy_hint=strategy_hint):
        return
    sl_mult = CONFIG.get("DYNAMIC_SL_MULTIPLIER", 1.0)
    tp_mult = CONFIG.get("DYNAMIC_TP_MULTIPLIER", 1.0)
    old_sl = float(signal.get("sl_dist", MIN_SL))
    old_tp = float(signal.get("tp_dist", MIN_TP))
    signal["sl_dist"] = max(old_sl * sl_mult, MIN_SL)
    signal["tp_dist"] = max(old_tp * tp_mult, MIN_TP)


def _is_de3_v4_signal_for_sizing(signal: dict) -> bool:
    if not isinstance(signal, dict):
        return False
    de3_ver = str(signal.get("de3_version", "") or "").strip().lower()
    if de3_ver == "v4":
        return True
    if signal.get("de3_v4_selected_variant_id") or signal.get("de3_v4_selected_lane"):
        return True
    return False


def _de3_variant_id_from_payload(payload: dict) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(
        payload.get("de3_v4_selected_variant_id")
        or payload.get("sub_strategy")
        or ""
    ).strip()


def _de3_trade_day_key(
    timestamp,
    *,
    roll_hour_et: int = 18,
) -> Optional[dt.date]:
    if timestamp is None:
        return None
    try:
        ts = pd.Timestamp(timestamp)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    roll_hour = max(0, min(23, int(roll_hour_et)))
    shifted = ts - pd.Timedelta(hours=roll_hour)
    return shifted.date()


def _match_de3_rule_values(raw_values, value: str, *, lower: bool = False) -> bool:
    if not isinstance(raw_values, (list, tuple, set)):
        return True
    normalized = {
        (str(item).strip().lower() if lower else str(item).strip())
        for item in raw_values
        if str(item).strip()
    }
    if not normalized:
        return True
    key = str(value or "").strip().lower() if lower else str(value or "").strip()
    return key in normalized


def _apply_de3_v4_confidence_tier_size(signal: dict, base_size: int) -> int:
    runtime_cfg = (
        (CONFIG.get("DE3_V4", {}) or {}).get("runtime", {})
        if isinstance((CONFIG.get("DE3_V4", {}) or {}).get("runtime", {}), dict)
        else {}
    )
    sizing_cfg = (
        runtime_cfg.get("confidence_tier_sizing", {})
        if isinstance(runtime_cfg.get("confidence_tier_sizing", {}), dict)
        else {}
    )
    if not bool(sizing_cfg.get("enabled", False)):
        return int(base_size)
    if not _is_de3_v4_signal_for_sizing(signal):
        return int(base_size)

    raw_fields = sizing_cfg.get(
        "confidence_field_priority",
        ["de3_policy_confidence", "de3_edge_confidence", "de3_v4_route_confidence"],
    )
    fields = [str(v).strip() for v in raw_fields if str(v).strip()] if isinstance(raw_fields, (list, tuple, set)) else []
    if not fields:
        fields = ["de3_policy_confidence", "de3_edge_confidence", "de3_v4_route_confidence"]

    conf_val = float("nan")
    conf_field = ""
    for field_name in fields:
        raw = signal.get(field_name, None)
        if raw is None:
            continue
        val = _coerce_float(raw, float("nan"))
        if math.isfinite(val):
            conf_val = _coerce_float(min(1.0, max(0.0, val)), val)
            conf_field = field_name
            break

    if not math.isfinite(conf_val):
        signal["de3_v4_confidence_sizing_enabled"] = True
        signal["de3_v4_confidence_sizing_applied"] = False
        signal["de3_v4_confidence_sizing_reason"] = "missing_confidence"
        return int(base_size)

    high_thr = _coerce_float(sizing_cfg.get("high_threshold", 0.86), 0.86)
    mid_thr = _coerce_float(sizing_cfg.get("mid_threshold", 0.78), 0.78)
    high_mult = _coerce_float(sizing_cfg.get("high_multiplier", 1.0), 1.0)
    mid_mult = _coerce_float(sizing_cfg.get("mid_multiplier", 0.8), 0.8)
    low_mult = _coerce_float(sizing_cfg.get("low_multiplier", 0.6), 0.6)

    tier = "low"
    mult = low_mult
    if conf_val >= high_thr:
        tier = "high"
        mult = high_mult
    elif conf_val >= mid_thr:
        tier = "mid"
        mult = mid_mult

    quality_cfg = (
        sizing_cfg.get("quality_adjustment", {})
        if isinstance(sizing_cfg.get("quality_adjustment", {}), dict)
        else {}
    )
    quality_enabled = bool(quality_cfg.get("enabled", False))
    quality_multiplier = 1.0
    quality_reason = "disabled"
    quality_ev_field = str(quality_cfg.get("ev_lcb_field", "de3_policy_ev_lcb_points") or "").strip()
    quality_loss_field = str(quality_cfg.get("p_loss_std_field", "de3_policy_p_loss_std") or "").strip()
    quality_ev_value = _coerce_float(signal.get(quality_ev_field), float("nan")) if quality_ev_field else float("nan")
    quality_loss_value = _coerce_float(signal.get(quality_loss_field), float("nan")) if quality_loss_field else float("nan")
    if quality_enabled:
        min_q = _coerce_float(quality_cfg.get("min_quality_multiplier", 0.65), 0.65)
        max_q = _coerce_float(quality_cfg.get("max_quality_multiplier", 1.0), 1.0)
        if max_q < min_q:
            min_q, max_q = max_q, min_q
        ev_center = _coerce_float(quality_cfg.get("ev_lcb_center", 3.2), 3.2)
        ev_scale = max(1e-6, _coerce_float(quality_cfg.get("ev_lcb_scale", 1.2), 1.2))
        loss_ref = max(1e-6, _coerce_float(quality_cfg.get("p_loss_std_ref", 0.01), 0.01))

        ev_score = float("nan")
        if math.isfinite(quality_ev_value):
            ev_score = 0.5 + 0.5 * math.tanh((quality_ev_value - ev_center) / ev_scale)
            ev_score = min(1.0, max(0.0, ev_score))

        loss_score = float("nan")
        if math.isfinite(quality_loss_value):
            loss_score = 0.5 + 0.5 * math.tanh((loss_ref - quality_loss_value) / loss_ref)
            loss_score = min(1.0, max(0.0, loss_score))

        if math.isfinite(ev_score) and math.isfinite(loss_score):
            combined_score = 0.60 * ev_score + 0.40 * loss_score
            quality_reason = "ev_and_loss_std"
        elif math.isfinite(ev_score):
            combined_score = ev_score
            quality_reason = "ev_only"
        elif math.isfinite(loss_score):
            combined_score = loss_score
            quality_reason = "loss_std_only"
        else:
            combined_score = 1.0
            quality_reason = "missing_inputs"
        quality_multiplier = min_q + (max_q - min_q) * min(1.0, max(0.0, combined_score))

    variant_multiplier = 1.0
    variant_multiplier_reason = "default"
    variant_mult_cfg = (
        sizing_cfg.get("variant_size_multipliers", {})
        if isinstance(sizing_cfg.get("variant_size_multipliers", {}), dict)
        else {}
    )
    if variant_mult_cfg:
        variant_id = str(
            signal.get("de3_v4_selected_variant_id")
            or signal.get("sub_strategy")
            or ""
        ).strip()
        if variant_id:
            raw_variant_mult = variant_mult_cfg.get(variant_id)
            if raw_variant_mult is None:
                raw_variant_mult = variant_mult_cfg.get(variant_id.lower())
            variant_mult_val = _coerce_float(raw_variant_mult, 1.0)
            if math.isfinite(variant_mult_val):
                variant_multiplier = max(0.0, float(variant_mult_val))
                if variant_multiplier != 1.0:
                    variant_multiplier_reason = "variant_override"

    tier_multiplier = max(0.0, float(mult))
    quality_multiplier = max(0.0, float(quality_multiplier))
    variant_multiplier = max(0.0, float(variant_multiplier))
    combined_multiplier = tier_multiplier * quality_multiplier * variant_multiplier

    min_contracts = max(1, _coerce_int(sizing_cfg.get("min_contracts", 1), 1))
    max_contracts = max(min_contracts, _coerce_int(sizing_cfg.get("max_contracts", base_size), base_size))
    scaled = int(round(max(0.0, float(base_size) * combined_multiplier)))
    scaled = max(min_contracts, min(max_contracts, scaled))

    signal["de3_v4_confidence_sizing_enabled"] = True
    signal["de3_v4_confidence_sizing_applied"] = bool(scaled != int(base_size))
    signal["de3_v4_confidence_field"] = conf_field
    signal["de3_v4_confidence_value"] = float(conf_val)
    signal["de3_v4_confidence_tier"] = str(tier)
    signal["de3_v4_confidence_tier_multiplier"] = float(tier_multiplier)
    signal["de3_v4_confidence_quality_adjustment_enabled"] = bool(quality_enabled)
    signal["de3_v4_confidence_quality_multiplier"] = float(quality_multiplier)
    signal["de3_v4_confidence_quality_reason"] = str(quality_reason)
    signal["de3_v4_confidence_quality_ev_field"] = quality_ev_field
    signal["de3_v4_confidence_quality_ev_value"] = (
        float(quality_ev_value) if math.isfinite(quality_ev_value) else None
    )
    signal["de3_v4_confidence_quality_loss_std_field"] = quality_loss_field
    signal["de3_v4_confidence_quality_loss_std_value"] = (
        float(quality_loss_value) if math.isfinite(quality_loss_value) else None
    )
    signal["de3_v4_confidence_variant_multiplier"] = float(variant_multiplier)
    signal["de3_v4_confidence_variant_multiplier_reason"] = str(variant_multiplier_reason)
    signal["de3_v4_confidence_size_multiplier"] = float(combined_multiplier)
    signal["de3_v4_confidence_size_base"] = int(base_size)
    signal["de3_v4_confidence_size_adjusted"] = int(scaled)
    signal["de3_v4_confidence_sizing_reason"] = (
        "tier_quality_scaled" if quality_enabled else "tier_scaled"
    )
    return int(scaled)


def _load_backtest_multiplier_schedule() -> tuple[Optional[dict], dict]:
    cfg = CONFIG.get("BACKTEST_GEMINI_MULTIPLIERS", {}) or {}
    enabled = bool(cfg.get("enabled", False))
    default_sl = _coerce_float(cfg.get("default_sl_multiplier", 1.0), 1.0)
    default_tp = _coerce_float(cfg.get("default_tp_multiplier", 1.0), 1.0)
    default_chop = _coerce_float(cfg.get("default_chop_multiplier", 1.0), 1.0)
    assume_tz = str(cfg.get("assume_timezone", "America/New_York") or "America/New_York")
    path_value = str(cfg.get("path", "") or "").strip()
    if path_value:
        path = Path(path_value)
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent / path).resolve()
    else:
        path = None

    info = {
        "enabled": enabled,
        "path": str(path) if path is not None else "",
        "loaded": False,
        "rows": 0,
        "bars_with_multiplier": 0,
        "default_sl_multiplier": float(default_sl),
        "default_tp_multiplier": float(default_tp),
        "default_chop_multiplier": float(default_chop),
        "timestamp_column": "",
        "sl_column": "",
        "tp_column": "",
        "chop_column": "",
        "first_timestamp": None,
        "last_timestamp": None,
        "error": "",
    }
    if not enabled:
        return None, info
    if path is None:
        info["error"] = "empty_path"
        logging.warning("Backtest multipliers dataset enabled but no path configured.")
        return None, info
    if not path.is_file():
        info["error"] = "missing_file"
        logging.warning("Backtest multipliers dataset missing: %s", path)
        return None, info

    try:
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        info["error"] = f"load_failed:{exc}"
        logging.warning("Backtest multipliers dataset load failed (%s): %s", path, exc)
        return None, info

    if frame is None or frame.empty:
        info["error"] = "empty_dataset"
        logging.warning("Backtest multipliers dataset is empty: %s", path)
        return None, info

    frame = frame.copy()
    frame.columns = [str(col).strip() for col in frame.columns]

    def _pick_column(candidates: list[str]) -> Optional[str]:
        for candidate in candidates:
            if not candidate:
                continue
            if candidate in frame.columns:
                return candidate
        return None

    ts_col = _pick_column(
        [
            str(cfg.get("timestamp_column", "") or "").strip(),
            "timestamp",
            "ts",
            "ts_event",
            "datetime",
            "time",
        ]
    )
    if ts_col is None:
        info["error"] = "missing_timestamp_column"
        logging.warning("Backtest multipliers dataset missing timestamp column: %s", path)
        return None, info

    try:
        ts_series = pd.to_datetime(frame[ts_col], errors="coerce")
    except Exception as exc:
        info["error"] = f"timestamp_parse_failed:{exc}"
        logging.warning("Backtest multipliers timestamp parse failed (%s): %s", path, exc)
        return None, info

    try:
        if getattr(ts_series.dt, "tz", None) is None:
            ts_series = ts_series.dt.tz_localize(assume_tz, ambiguous="NaT", nonexistent="shift_forward")
        ts_series = ts_series.dt.tz_convert("UTC")
    except Exception as exc:
        info["error"] = f"timestamp_tz_failed:{exc}"
        logging.warning("Backtest multipliers timezone normalization failed (%s): %s", path, exc)
        return None, info

    sl_col = _pick_column(
        [
            str(cfg.get("sl_column", "") or "").strip(),
            "sl_multiplier",
            "sl_mult",
            "dynamic_sl_multiplier",
        ]
    )
    tp_col = _pick_column(
        [
            str(cfg.get("tp_column", "") or "").strip(),
            "tp_multiplier",
            "tp_mult",
            "dynamic_tp_multiplier",
        ]
    )
    chop_col = _pick_column(
        [
            str(cfg.get("chop_column", "") or "").strip(),
            "chop_multiplier",
            "chop_mult",
            "dynamic_chop_multiplier",
        ]
    )

    if sl_col is None or tp_col is None:
        info["error"] = "missing_sl_tp_columns"
        logging.warning(
            "Backtest multipliers dataset missing required SL/TP columns (%s): sl=%s tp=%s",
            path,
            sl_col,
            tp_col,
        )
        return None, info

    sl_values = pd.to_numeric(frame[sl_col], errors="coerce").fillna(default_sl)
    tp_values = pd.to_numeric(frame[tp_col], errors="coerce").fillna(default_tp)
    if chop_col is not None:
        chop_values = pd.to_numeric(frame[chop_col], errors="coerce").fillna(default_chop)
    else:
        chop_values = pd.Series(default_chop, index=frame.index, dtype=float)

    sched = pd.DataFrame(
        {
            "timestamp_utc": ts_series,
            "sl_multiplier": sl_values,
            "tp_multiplier": tp_values,
            "chop_multiplier": chop_values,
        }
    )
    sched = sched.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
    if sched.empty:
        info["error"] = "no_valid_rows"
        logging.warning("Backtest multipliers dataset has no valid rows after parsing: %s", path)
        return None, info
    sched = sched.drop_duplicates(subset=["timestamp_utc"], keep="last")

    try:
        index_ns = (
            sched["timestamp_utc"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .to_numpy(dtype="datetime64[ns]")
        )
        sl_arr = sched["sl_multiplier"].to_numpy(dtype=float)
        tp_arr = sched["tp_multiplier"].to_numpy(dtype=float)
        chop_arr = sched["chop_multiplier"].to_numpy(dtype=float)
    except Exception as exc:
        info["error"] = f"array_build_failed:{exc}"
        logging.warning("Backtest multipliers dataset array conversion failed (%s): %s", path, exc)
        return None, info

    info.update(
        {
            "loaded": True,
            "rows": int(len(sched)),
            "timestamp_column": ts_col,
            "sl_column": str(sl_col),
            "tp_column": str(tp_col),
            "chop_column": str(chop_col or ""),
            "first_timestamp": sched["timestamp_utc"].iloc[0].isoformat(),
            "last_timestamp": sched["timestamp_utc"].iloc[-1].isoformat(),
            "error": "",
        }
    )
    logging.info(
        "Backtest multipliers dataset loaded: rows=%s range=%s..%s path=%s",
        info["rows"],
        info["first_timestamp"],
        info["last_timestamp"],
        path,
    )
    return {
        "index_ns": index_ns,
        "sl": sl_arr,
        "tp": tp_arr,
        "chop": chop_arr,
    }, info


def trend_state_from_reason(reason: Optional[str]) -> str:
    if reason and "Bearish" in str(reason):
        return "Strong Bearish"
    if reason and "Bullish" in str(reason):
        return "Strong Bullish"
    return "NEUTRAL"


def continuation_market_confirmed(
    side: str,
    current_time: pd.Timestamp,
    bar_close: float,
    trend_context: Optional[dict],
    cfg: Optional[dict],
) -> bool:
    if not cfg or not cfg.get("enabled", True):
        return True
    if not isinstance(trend_context, dict) or not trend_context:
        # When trend-day context is intentionally disabled, skip this confirmation gate.
        return True

    def series_value(key: str, default):
        series = trend_context.get(key)
        if isinstance(series, pd.Series):
            try:
                return series.get(current_time, default)
            except Exception:
                return default
        return series if series is not None else default

    use_adx = cfg.get("use_adx", True)
    use_trend_alt = cfg.get("use_trend_alt", True)
    use_vwap = cfg.get("use_vwap", True)
    use_structure = cfg.get("use_structure_break", True)
    vwap_sigma_min = float(cfg.get("vwap_sigma_min", 0.0) or 0.0)
    require_any = cfg.get("require_any", True)

    adx_up = bool(series_value("adx_strong_up", False))
    adx_down = bool(series_value("adx_strong_down", False))
    trend_up = bool(series_value("trend_up_alt", False))
    trend_down = bool(series_value("trend_down_alt", False))
    vwap_sigma = series_value("vwap_sigma_dist", 0.0)
    try:
        vwap_sigma = float(vwap_sigma)
    except Exception:
        vwap_sigma = 0.0

    prior_high = series_value("prior_session_high", None)
    prior_low = series_value("prior_session_low", None)
    structure_up = False
    structure_down = False
    if prior_high is not None and not pd.isna(prior_high):
        structure_up = bar_close > float(prior_high)
    if prior_low is not None and not pd.isna(prior_low):
        structure_down = bar_close < float(prior_low)

    if side == "LONG":
        checks = []
        if use_adx:
            checks.append(adx_up)
        if use_trend_alt:
            checks.append(trend_up)
        if use_vwap:
            checks.append(vwap_sigma >= vwap_sigma_min)
        if use_structure:
            checks.append(structure_up)
    else:
        checks = []
        if use_adx:
            checks.append(adx_down)
        if use_trend_alt:
            checks.append(trend_down)
        if use_vwap:
            checks.append(vwap_sigma <= -vwap_sigma_min)
        if use_structure:
            checks.append(structure_down)

    if not checks:
        return True
    return any(checks) if require_any else all(checks)


def compute_pnl_points(side: str, entry_price: float, exit_price: float) -> float:
    return exit_price - entry_price if side == "LONG" else entry_price - exit_price


def round_points_to_tick(points: float) -> float:
    ticks = max(1, int(math.ceil(abs(points) / TICK_SIZE)))
    return ticks * TICK_SIZE


def consensus_ml_ok(signal: Optional[dict]) -> bool:
    """Backtest-only: require stronger ML confidence to support consensus."""
    if not signal:
        return False
    strat = str(signal.get("strategy", ""))
    if not strat.startswith("MLPhysics"):
        return True
    if bool(CONFIG.get("BACKTEST_ML_PHYSICS_GLOBAL_FILTERS_ONLY", False)):
        return True
    conf = signal.get("ml_confidence")
    threshold = signal.get("ml_threshold")
    if conf is None or threshold is None:
        return False
    try:
        conf_val = float(conf)
        thr_val = float(threshold)
    except Exception:
        return False
    min_conf = CONFIG.get("BACKTEST_CONSENSUS_ML_MIN_CONF")
    extra = CONFIG.get("BACKTEST_CONSENSUS_ML_EXTRA_MARGIN", 0.0)
    required = thr_val + float(extra or 0.0)
    if min_conf is not None:
        required = max(required, float(min_conf))
    return conf_val >= required


def ml_vol_regime_ok(
    signal: Optional[dict],
    session_name: Optional[str],
    vol_regime: Optional[str],
    asia_viable: Optional[bool] = None,
) -> bool:
    """Require stronger ML confidence by volatility regime."""
    if bool(CONFIG.get("BACKTEST_ML_PHYSICS_GLOBAL_FILTERS_ONLY", False)):
        return True
    cfg = CONFIG.get("ML_PHYSICS_VOL_REGIME_GUARD", {}) or {}
    if not cfg.get("enabled", True):
        return True
    if not signal:
        return False
    strat = str(signal.get("strategy", ""))
    if not strat.startswith("MLPhysics"):
        return True
    conf = signal.get("ml_confidence")
    threshold = signal.get("ml_threshold")
    if conf is None or threshold is None:
        return False
    try:
        conf_val = float(conf)
        thr_val = float(threshold)
    except Exception:
        return False

    regime_key = str(vol_regime or signal.get("vol_regime") or "normal").lower()
    if regime_key in ("unknown", "none", "nan"):
        regime_key = "normal"

    base_cfg = cfg.get("default", {}) or {}
    session_cfg = {}
    if session_name:
        sess_map = cfg.get("sessions", {}) or {}
        session_key = str(session_name).upper()
        if session_key in sess_map:
            session_cfg = sess_map.get(session_key) or {}
        else:
            session_cfg = sess_map.get(session_name) or {}

    reg_cfg: dict = {}

    def merge(src):
        if isinstance(src, dict):
            reg_cfg.update(src)

    merge(base_cfg.get("all"))
    merge(base_cfg.get(regime_key))
    merge(session_cfg.get("all"))
    merge(session_cfg.get(regime_key))

    if reg_cfg.get("block"):
        return False
    side = str(signal.get("side", "")).upper()
    block_sides = reg_cfg.get("block_sides") or reg_cfg.get("block_side")
    if block_sides:
        try:
            block_set = {str(item).upper() for item in block_sides}
        except Exception:
            block_set = set()
        if side in block_set:
            return False

    relax_low_vol_penalty = (
        bool(asia_viable)
        and session_name
        and str(session_name).upper() == "ASIA"
        and regime_key == "low"
    )
    delta = reg_cfg.get("min_conf_delta", 0.0)
    if isinstance(delta, dict):
        delta = delta.get(side, delta.get("default", 0.0))
    try:
        delta_val = 0.0 if relax_low_vol_penalty else float(delta or 0.0)
    except Exception:
        delta_val = 0.0
    required = thr_val + delta_val

    side_extra = reg_cfg.get("side_extra_delta")
    if not relax_low_vol_penalty and isinstance(side_extra, dict) and side in side_extra:
        try:
            required += float(side_extra[side])
        except Exception:
            pass

    min_conf = reg_cfg.get("min_conf")
    if min_conf is not None:
        try:
            required = max(required, float(min_conf))
        except Exception:
            pass
    max_conf = reg_cfg.get("max_conf")
    if max_conf is not None:
        try:
            required = min(required, float(max_conf))
        except Exception:
            pass
    return conf_val >= required


def add_bypass_filters_from_trigger(bypass_list: list[str], trigger: Optional[str]) -> None:
    if not trigger:
        return
    if trigger.startswith("FilterStack:"):
        raw = trigger.split(":", 1)[1]
        for name in raw.split("+"):
            if name and name not in bypass_list:
                bypass_list.append(name)
        return
    if trigger not in bypass_list:
        bypass_list.append(trigger)


def align_trend_day_series(series_dict: dict, target_index: pd.Index) -> dict:
    bool_keys = {
        "reclaim_down",
        "reclaim_up",
        "no_reclaim_down_t1",
        "no_reclaim_up_t1",
        "no_reclaim_down_t2",
        "no_reclaim_up_t2",
        "trend_up_alt",
        "trend_down_alt",
        "adx_strong_up",
        "adx_strong_down",
    }
    aligned: dict = {}
    for key, series in series_dict.items():
        if key == "day_index":
            idx = target_index
            if idx.tz is not None:
                idx = idx.tz_convert(NY_TZ)
            aligned[key] = pd.Series(idx.date, index=target_index)
            continue
        if not isinstance(series, pd.Series):
            aligned[key] = series
            continue
        s = series.reindex(target_index, method="ffill")
        if key in bool_keys:
            s = s.fillna(False).astype(bool)
        aligned[key] = s
    return aligned


def compute_trend_day_series(df: pd.DataFrame) -> dict:
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=50).mean()

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr20 = tr.ewm(alpha=1 / 20, adjust=False).mean()
    atr_baseline = atr20.rolling(ATR_BASELINE_WINDOW, min_periods=ATR_BASELINE_WINDOW).median()
    atr_expansion = atr20 / atr_baseline.replace(0, np.nan)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    tr_smooth = tr.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    plus_di = 100 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / tr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / ADX_PERIOD, adjust=False).mean()
    adx_strong_up = (adx >= ADX_FLIP_THRESHOLD) & (plus_di > minus_di)
    adx_strong_down = (adx >= ADX_FLIP_THRESHOLD) & (minus_di > plus_di)

    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ)
    day_index = idx.date

    typical_price = (high + low + close) / 3
    volume = df["volume"].fillna(0)
    cum_pv = (typical_price * volume).groupby(day_index).cumsum()
    cum_v = volume.groupby(day_index).cumsum()
    vwap = cum_pv / cum_v.replace(0, np.nan)

    ret = close.diff()
    sigma = ret.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW).std()
    sigma = sigma.ffill().clip(lower=TICK_SIZE)
    vwap_sigma_dist = (close - vwap) / sigma

    close_ge = close > vwap
    close_le = close < vwap
    reclaim_down = (
        close_ge.rolling(
            VWAP_RECLAIM_CONSECUTIVE_BARS, min_periods=VWAP_RECLAIM_CONSECUTIVE_BARS
        ).sum()
        == VWAP_RECLAIM_CONSECUTIVE_BARS
    )
    reclaim_up = (
        close_le.rolling(
            VWAP_RECLAIM_CONSECUTIVE_BARS, min_periods=VWAP_RECLAIM_CONSECUTIVE_BARS
        ).sum()
        == VWAP_RECLAIM_CONSECUTIVE_BARS
    )

    no_reclaim_down_t1 = (
        reclaim_down.rolling(VWAP_NO_RECLAIM_BARS_T1, min_periods=VWAP_NO_RECLAIM_BARS_T1).sum() == 0
    )
    no_reclaim_up_t1 = (
        reclaim_up.rolling(VWAP_NO_RECLAIM_BARS_T1, min_periods=VWAP_NO_RECLAIM_BARS_T1).sum() == 0
    )
    no_reclaim_down_t2 = (
        reclaim_down.rolling(VWAP_NO_RECLAIM_BARS_T2, min_periods=VWAP_NO_RECLAIM_BARS_T2).sum() == 0
    )
    no_reclaim_up_t2 = (
        reclaim_up.rolling(VWAP_NO_RECLAIM_BARS_T2, min_periods=VWAP_NO_RECLAIM_BARS_T2).sum() == 0
    )

    session_open = open_.groupby(day_index).transform("first")
    daily_low = low.groupby(day_index).min()
    daily_high = high.groupby(day_index).max()
    prior_session_low = pd.Series(day_index, index=df.index).map(daily_low.shift(1))
    prior_session_high = pd.Series(day_index, index=df.index).map(daily_high.shift(1))

    sma50_slope_up = sma50 - sma50.shift(TREND_UP_EMA_SLOPE_BARS)
    sma50_slope_up = sma50_slope_up > 0
    above_ema50 = close > ema50
    above_ema50_count = above_ema50.rolling(
        TREND_UP_ABOVE_EMA50_WINDOW, min_periods=TREND_UP_ABOVE_EMA50_WINDOW
    ).sum()
    above_ema50_ok = above_ema50_count >= TREND_UP_ABOVE_EMA50_COUNT
    seg = TREND_UP_HL_SEGMENT
    low_seg1 = low.rolling(seg, min_periods=seg).min()
    low_seg2 = low.shift(seg).rolling(seg, min_periods=seg).min()
    low_seg3 = low.shift(seg * 2).rolling(seg, min_periods=seg).min()
    higher_lows = (low_seg1 > low_seg2) & (low_seg2 > low_seg3)
    trend_up_alt = sma50_slope_up & above_ema50_ok & higher_lows & (atr_expansion >= TREND_UP_ATR_EXP)

    sma50_slope_down = sma50 - sma50.shift(TREND_DOWN_EMA_SLOPE_BARS)
    sma50_slope_down = sma50_slope_down < 0
    below_ema50 = close < ema50
    below_ema50_count = below_ema50.rolling(
        TREND_DOWN_BELOW_EMA50_WINDOW, min_periods=TREND_DOWN_BELOW_EMA50_WINDOW
    ).sum()
    below_ema50_ok = below_ema50_count >= TREND_DOWN_BELOW_EMA50_COUNT
    seg_down = TREND_DOWN_LH_SEGMENT
    high_seg1 = high.rolling(seg_down, min_periods=seg_down).max()
    high_seg2 = high.shift(seg_down).rolling(seg_down, min_periods=seg_down).max()
    high_seg3 = high.shift(seg_down * 2).rolling(seg_down, min_periods=seg_down).max()
    lower_highs = (high_seg1 < high_seg2) & (high_seg2 < high_seg3)
    trend_down_alt = sma50_slope_down & below_ema50_ok & lower_highs & (atr_expansion >= TREND_DOWN_ATR_EXP)

    return {
        "ema50": ema50,
        "ema200": ema200,
        "sma50": sma50,
        "atr20": atr20,
        "atr_expansion": atr_expansion,
        "vwap": vwap,
        "sigma": sigma,
        "vwap_sigma_dist": vwap_sigma_dist,
        "reclaim_down": reclaim_down.fillna(False),
        "reclaim_up": reclaim_up.fillna(False),
        "no_reclaim_down_t1": no_reclaim_down_t1.fillna(False),
        "no_reclaim_up_t1": no_reclaim_up_t1.fillna(False),
        "no_reclaim_down_t2": no_reclaim_down_t2.fillna(False),
        "no_reclaim_up_t2": no_reclaim_up_t2.fillna(False),
        "session_open": session_open,
        "prior_session_low": prior_session_low,
        "prior_session_high": prior_session_high,
        "trend_up_alt": trend_up_alt.fillna(False),
        "trend_down_alt": trend_down_alt.fillna(False),
        "adx_strong_up": adx_strong_up.fillna(False),
        "adx_strong_down": adx_strong_down.fillna(False),
        "day_index": pd.Series(day_index, index=df.index),
    }


def format_recent_trade(trade: dict) -> str:
    entry_time = trade.get("entry_time")
    exit_time = trade.get("exit_time")
    entry_str = entry_time.strftime("%Y-%m-%d %H:%M") if entry_time else "-"
    if exit_time:
        # Show full exit datetime when trade spans days to avoid ambiguous report lines.
        if entry_time and hasattr(entry_time, "date") and hasattr(exit_time, "date"):
            if exit_time.date() != entry_time.date():
                exit_str = exit_time.strftime("%Y-%m-%d %H:%M")
            else:
                exit_str = exit_time.strftime("%H:%M")
        else:
            exit_str = exit_time.strftime("%Y-%m-%d %H:%M")
    else:
        exit_str = "-"
    strategy = trade.get("strategy", "Unknown")
    sub_strategy = trade.get("sub_strategy")
    if sub_strategy:
        strategy = f"{strategy}:{sub_strategy}"
    side = trade.get("side", "")
    mode = trade.get("entry_mode", "standard")
    pnl = trade.get("pnl_net", 0.0)
    reason = trade.get("exit_reason", "unknown")
    rescue_from = trade.get("rescue_from_strategy")
    rescue_sub = trade.get("rescue_from_sub_strategy")
    rescue_trigger = trade.get("rescue_trigger")
    rescue_label = None
    if rescue_from:
        rescue_label = rescue_from
        if rescue_sub:
            rescue_label = f"{rescue_label}:{rescue_sub}"
        if rescue_trigger:
            rescue_label = f"{rescue_label} via {rescue_trigger}"
    suffix = f" rescue_from={rescue_label}" if rescue_label else ""
    consensus_contrib = trade.get("consensus_contributors")
    if consensus_contrib:
        suffix += f" consensus_from={','.join(consensus_contrib)}"
    bypassed = trade.get("bypassed_filters")
    if bypassed:
        suffix += f" bypassed={'+'.join(bypassed)}"
    hold_suffix = ""
    if entry_time and exit_time:
        try:
            hold_mins = int(round((exit_time - entry_time).total_seconds() / 60.0))
            if hold_mins >= 0:
                hold_suffix = f" hold={hold_mins}m"
        except Exception:
            hold_suffix = ""
    bracket_suffix = ""
    try:
        exec_sl = float(trade.get("sl_dist"))
        exec_tp = float(trade.get("tp_dist"))
        if math.isfinite(exec_sl) and math.isfinite(exec_tp) and exec_sl > 0 and exec_tp > 0:
            bracket_suffix += f" exec_sl={exec_sl:.2f} exec_tp={exec_tp:.2f}"
    except Exception:
        pass
    try:
        v4_sl = float(trade.get("de3_v4_selected_sl"))
        v4_tp = float(trade.get("de3_v4_selected_tp"))
        if math.isfinite(v4_sl) and math.isfinite(v4_tp) and v4_sl > 0 and v4_tp > 0:
            bracket_suffix += f" v4_sl={v4_sl:.2f} v4_tp={v4_tp:.2f}"
    except Exception:
        pass
    return (
        f"{entry_str} {exit_str} {side} {strategy} {mode} pnl={pnl:.2f} exit={reason}"
        f"{hold_suffix}{bracket_suffix}{suffix}"
    )


def _serialize_json_value(value):
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _serialize_json_value(value.item())
    if isinstance(value, dict):
        return {str(k): _serialize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def serialize_trade(trade: dict) -> dict:
    return _serialize_json_value(dict(trade))


def _extract_trade_pnl_array(trade_log: list[dict]) -> np.ndarray:
    pnl_values: list[float] = []
    for trade in trade_log:
        if not isinstance(trade, dict):
            continue
        pnl = safe_float(trade.get("pnl_net"), float("nan"))
        if not math.isfinite(pnl):
            pnl = safe_float(trade.get("pnl_dollars"), float("nan"))
        if not math.isfinite(pnl):
            pnl = safe_float(trade.get("pnl"), float("nan"))
        if math.isfinite(pnl):
            pnl_values.append(float(pnl))
    if not pnl_values:
        return np.asarray([], dtype=float)
    return np.asarray(pnl_values, dtype=float)


def _trade_timestamp_to_ny_day(value) -> Optional[str]:
    if value is None:
        return None
    ts = None
    if isinstance(value, pd.Timestamp):
        ts = value.to_pydatetime()
    elif isinstance(value, dt.datetime):
        ts = value
    else:
        try:
            ts = dt.datetime.fromisoformat(str(value))
        except Exception:
            return None
    try:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=NY_TZ)
        else:
            ts = ts.astimezone(NY_TZ)
    except Exception:
        return None
    return ts.date().isoformat()


def _compute_backtest_risk_metrics(trade_log: list[dict]) -> dict:
    pnl_array = _extract_trade_pnl_array(trade_log)
    if pnl_array.size == 0:
        return {
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "avg_trade_net": 0.0,
            "trade_sqn": 0.0,
            "trade_sharpe_like": 0.0,
            "daily_sharpe": 0.0,
            "daily_sortino": 0.0,
            "trading_days": 0,
        }

    gross_profit = float(np.sum(pnl_array[pnl_array > 0.0]))
    gross_loss = float(-np.sum(pnl_array[pnl_array < 0.0]))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-12 else (
        999.0 if gross_profit > 0.0 else 0.0
    )
    avg_trade_net = float(np.mean(pnl_array))

    trade_sqn = 0.0
    if pnl_array.size > 1:
        pnl_std = float(np.std(pnl_array, ddof=1))
        if pnl_std > 1e-12:
            trade_sqn = float((avg_trade_net / pnl_std) * math.sqrt(float(pnl_array.size)))

    daily_pnl: dict[str, float] = defaultdict(float)
    for trade in trade_log:
        if not isinstance(trade, dict):
            continue
        day_key = _trade_timestamp_to_ny_day(trade.get("exit_time") or trade.get("entry_time"))
        if not day_key:
            continue
        pnl = safe_float(trade.get("pnl_net"), float("nan"))
        if not math.isfinite(pnl):
            pnl = safe_float(trade.get("pnl_dollars"), float("nan"))
        if not math.isfinite(pnl):
            pnl = safe_float(trade.get("pnl"), 0.0)
        daily_pnl[day_key] += float(pnl)

    daily_values = np.asarray(list(daily_pnl.values()), dtype=float)
    daily_sharpe = 0.0
    daily_sortino = 0.0
    if daily_values.size > 1:
        daily_mean = float(np.mean(daily_values))
        daily_std = float(np.std(daily_values, ddof=1))
        if daily_std > 1e-12:
            daily_sharpe = float((daily_mean / daily_std) * math.sqrt(252.0))
        downside = np.minimum(daily_values, 0.0)
        downside_rms = float(math.sqrt(float(np.mean(np.square(downside)))))
        if downside_rms > 1e-12:
            daily_sortino = float((daily_mean / downside_rms) * math.sqrt(252.0))

    return {
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_trade_net": avg_trade_net,
        "trade_sqn": trade_sqn,
        # Keep an explicit alias so downstream consumers know this is trade-count scaled.
        "trade_sharpe_like": trade_sqn,
        "daily_sharpe": daily_sharpe,
        "daily_sortino": daily_sortino,
        "trading_days": int(daily_values.size),
    }


def _build_monte_carlo_summary(
    trade_log: list[dict],
    stats: dict,
    *,
    simulations: int,
    seed: int,
    starting_balance: float,
) -> dict:
    pnl_array = _extract_trade_pnl_array(trade_log)
    if pnl_array.size == 0:
        return {
            "enabled": True,
            "status": "skipped_no_trades",
            "simulations": int(simulations),
            "seed": int(seed),
            "starting_balance": float(starting_balance),
            "trade_count": 0,
        }
    rng = np.random.default_rng(int(seed))
    sample_size = int(pnl_array.size)
    final_balances = np.empty(int(simulations), dtype=float)
    max_drawdowns = np.empty(int(simulations), dtype=float)
    for idx in range(int(simulations)):
        sampled = rng.choice(pnl_array, size=sample_size, replace=True)
        equity_curve = float(starting_balance) + np.cumsum(sampled)
        equity_with_start = np.concatenate(
            ([float(starting_balance)], equity_curve.astype(float, copy=False))
        )
        running_peak = np.maximum.accumulate(equity_with_start)
        drawdowns = running_peak[1:] - equity_curve
        final_balances[idx] = (
            float(equity_curve[-1]) if equity_curve.size else float(starting_balance)
        )
        max_drawdowns[idx] = float(np.max(drawdowns)) if drawdowns.size else 0.0
    final_pnls = final_balances - float(starting_balance)
    realized_net = safe_float(stats.get("equity"), 0.0)
    realized_max_dd = safe_float(stats.get("max_drawdown"), 0.0)
    return {
        "enabled": True,
        "status": "ok",
        "simulations": int(simulations),
        "seed": int(seed),
        "starting_balance": float(starting_balance),
        "trade_count": int(sample_size),
        "realized_net_pnl": float(realized_net),
        "realized_final_balance": float(starting_balance + realized_net),
        "realized_max_drawdown": float(realized_max_dd),
        "final_balance_mean": float(np.mean(final_balances)),
        "final_balance_median": float(np.median(final_balances)),
        "final_balance_p05": float(np.percentile(final_balances, 5)),
        "final_balance_p95": float(np.percentile(final_balances, 95)),
        "net_pnl_mean": float(np.mean(final_pnls)),
        "net_pnl_median": float(np.median(final_pnls)),
        "net_pnl_p05": float(np.percentile(final_pnls, 5)),
        "net_pnl_p95": float(np.percentile(final_pnls, 95)),
        "max_drawdown_mean": float(np.mean(max_drawdowns)),
        "max_drawdown_median": float(np.median(max_drawdowns)),
        "max_drawdown_p95": float(np.percentile(max_drawdowns, 95)),
        "max_drawdown_p99": float(np.percentile(max_drawdowns, 99)),
        "probability_final_balance_above_start": float(
            np.mean(final_balances > float(starting_balance))
        ),
        "probability_final_balance_above_realized": float(
            np.mean(final_balances > float(starting_balance + realized_net))
        ),
        "probability_drawdown_worse_than_realized": float(
            np.mean(max_drawdowns > float(realized_max_dd))
        ),
    }


def _extract_summary_from_backtest_report(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        # Summary sits near the top in our report format; keep this fast for large files.
        head_text = path.open("r", encoding="utf-8", errors="ignore").read(32768)
        patterns = {
            "equity": r'"equity"\s*:\s*([-+]?\d+(?:\.\d+)?)',
            "trades": r'"trades"\s*:\s*(\d+)',
            "wins": r'"wins"\s*:\s*(\d+)',
            "losses": r'"losses"\s*:\s*(\d+)',
            "winrate": r'"winrate"\s*:\s*([-+]?\d+(?:\.\d+)?)',
            "max_drawdown": r'"max_drawdown"\s*:\s*([-+]?\d+(?:\.\d+)?)',
        }
        out = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, head_text)
            if not match:
                return None
            raw = match.group(1)
            out[key] = int(raw) if key in {"trades", "wins", "losses"} else float(raw)
        out["path"] = str(path)
        out["source"] = "fast_head_parse"
        return out
    except Exception:
        return None


def _select_baseline_report_path(
    report_dir: Path,
    symbol: str,
    start_tag: str,
    end_tag: str,
    current_report_path: Path,
) -> Optional[Path]:
    override = Path(BACKTEST_BASELINE_REPORT_PATH).expanduser() if BACKTEST_BASELINE_REPORT_PATH else None
    if override is not None:
        if not override.is_absolute():
            override = (Path(__file__).resolve().parent / override).resolve()
        if override.exists():
            return override
    safe_symbol = sanitize_filename(symbol)
    pattern = f"backtest_{safe_symbol}_{start_tag}_{end_tag}_*.json"
    candidates = sorted(
        [p for p in report_dir.glob(pattern) if p.resolve() != current_report_path.resolve()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    scanned = candidates[:BACKTEST_BASELINE_LOOKBACK_RUNS]
    best_path = None
    best_equity = -float("inf")
    for path in scanned:
        summary = _extract_summary_from_backtest_report(path)
        if not isinstance(summary, dict):
            continue
        eq = safe_float(summary.get("equity"), -float("inf"))
        if eq > best_equity:
            best_equity = eq
            best_path = path
    return best_path


def _build_baseline_comparison(current_summary: dict, baseline_summary: dict) -> dict:
    cur_equity = safe_float(current_summary.get("equity"), 0.0)
    cur_dd = safe_float(current_summary.get("max_drawdown"), 0.0)
    cur_wr = safe_float(current_summary.get("winrate"), 0.0)
    cur_trades = int(safe_float(current_summary.get("trades"), 0.0))
    base_equity = safe_float(baseline_summary.get("equity"), 0.0)
    base_dd = safe_float(baseline_summary.get("max_drawdown"), 0.0)
    base_wr = safe_float(baseline_summary.get("winrate"), 0.0)
    base_trades = int(safe_float(baseline_summary.get("trades"), 0.0))
    return {
        "current": {
            "equity": float(cur_equity),
            "max_drawdown": float(cur_dd),
            "winrate": float(cur_wr),
            "trades": int(cur_trades),
            "path": str(current_summary.get("path", "")),
        },
        "baseline": {
            "equity": float(base_equity),
            "max_drawdown": float(base_dd),
            "winrate": float(base_wr),
            "trades": int(base_trades),
            "path": str(baseline_summary.get("path", "")),
        },
        "delta": {
            "equity": float(cur_equity - base_equity),
            "max_drawdown": float(cur_dd - base_dd),
            "winrate": float(cur_wr - base_wr),
            "trades": int(cur_trades - base_trades),
        },
        "ratio_vs_baseline": {
            "equity": float(cur_equity / base_equity) if base_equity else None,
            "max_drawdown": float(cur_dd / base_dd) if base_dd else None,
            "trades": float(cur_trades / base_trades) if base_trades else None,
        },
    }


def _read_code_context_payload(base_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for rel in BACKTEST_GEMINI_RECOMMENDER_DE3_FILES:
        path = (base_dir / rel).resolve()
        if not path.exists() or not path.is_file():
            rows.append({"path": str(path), "exists": False, "excerpt": ""})
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            rows.append({"path": str(path), "exists": True, "error": str(exc), "excerpt": ""})
            continue
        excerpt = text[:BACKTEST_GEMINI_RECOMMENDER_MAX_CODE_CHARS]
        rows.append(
            {
                "path": str(path),
                "exists": True,
                "chars_total": int(len(text)),
                "chars_excerpt": int(len(excerpt)),
                "excerpt": excerpt,
            }
        )
    return rows


def _build_backtest_quant_report_prompt(
    backtest_summary: dict,
    monte_carlo_summary: dict,
    baseline_comparison: Optional[dict],
    market_conditions_summary: Optional[dict],
    code_context_rows: list[dict],
) -> str:
    code_blocks: list[str] = []
    for row in code_context_rows:
        if not bool(row.get("exists")):
            continue
        excerpt = str(row.get("excerpt", "") or "").strip()
        if not excerpt:
            continue
        path_text = str(row.get("path", "") or "")
        filename = Path(path_text).name if path_text else "unknown_file"
        code_blocks.append(
            "\n".join(
                [
                    f"--- STRATEGY FILE: {filename} ---",
                    "```python",
                    excerpt,
                    "```",
                ]
            )
        )
    code_context_text = (
        "\n\n".join(code_blocks)
        if code_blocks
        else "No strategy code excerpts were available."
    )
    baseline_payload = (
        baseline_comparison if isinstance(baseline_comparison, dict) else {"status": "not_available"}
    )
    market_payload = (
        market_conditions_summary
        if isinstance(market_conditions_summary, dict)
        else {"status": "not_available"}
    )
    return f"""
You are a Senior Quantitative Researcher at a top-tier proprietary trading firm.
Analyze this run and provide optimization recommendations grounded in risk control and robustness.

### 1. Historical Backtest Stats
{json.dumps(backtest_summary, indent=2, ensure_ascii=True)}

### 2. Monte Carlo Projections
{json.dumps(monte_carlo_summary, indent=2, ensure_ascii=True)}

### 3. Baseline Comparison
{json.dumps(baseline_payload, indent=2, ensure_ascii=True)}

### 4. Market-Condition Distribution
{json.dumps(market_payload, indent=2, ensure_ascii=True)}

### 5. Strategy Code Context
{code_context_text}

---

### Output Instructions
Return concise markdown with exactly these sections:
1) **Grade** (A+ to F) and one-line summary.
2) **Risk Profile**:
- Drawdown realism (historical vs Monte Carlo tails)
- Likely failure modes
- Prop-style survivability notes
3) **Baseline Delta Diagnosis**:
- Why this run is better/worse than baseline
- Which metric regressions matter most
4) **Code-Aware Recommendations**:
- Up to 5 concrete changes, each with:
  - target file(s)
  - expected impact (PnL/DD/WR/trade count)
  - overfitting risk level
5) **Next A/B Plan**:
- 3 experiments only, ranked by robustness (most robust first)
""".strip()


def _call_backtest_gemini_recommender(prompt: str) -> dict:
    gem_cfg = CONFIG.get("GEMINI", {}) if isinstance(CONFIG.get("GEMINI", {}), dict) else {}
    allow_when_disabled = bool(BACKTEST_POST_RUN_CFG.get("allow_when_gemini_disabled", True))
    if (not bool(gem_cfg.get("enabled", False))) and (not allow_when_disabled):
        return {"status": "disabled", "reason": "gemini_disabled"}
    api_key = str(gem_cfg.get("api_key", "") or "").strip()
    if not api_key:
        return {"status": "disabled", "reason": "missing_api_key"}
    model = str(gem_cfg.get("model", "gemini-3-pro-preview") or "gemini-3-pro-preview").strip()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "You are a skeptical, mathematically rigorous quant researcher. "
                        "Focus on downside risk, robustness, and practical A/B experiments."
                    )
                }
            ]
        },
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": 2048,
        },
    }
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=float(BACKTEST_GEMINI_RECOMMENDER_TIMEOUT_SEC),
        )
        if not resp.ok:
            return {
                "status": "error",
                "model": model,
                "http_status": int(resp.status_code),
                "response_text": resp.text[:1000],
            }
        data = resp.json()
        parts = (
            ((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")
            or []
        )
        text_chunks = [str((part or {}).get("text", "")) for part in parts if isinstance(part, dict)]
        text = "\n".join(chunk for chunk in text_chunks if chunk).strip()
        return {
            "status": "ok",
            "model": model,
            "text": text,
        }
    except Exception as exc:
        return {"status": "error", "model": model, "error": str(exc)}


def _sorted_counter_dict(counter: Counter, limit: int = 0) -> dict:
    items = sorted(counter.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    if limit and limit > 0:
        items = items[: int(limit)]
    return {str(k): int(v) for k, v in items}


def summarize_market_conditions(trades: list[dict], top_n: int = 50) -> dict:
    if not trades:
        return {
            "trades": 0,
            "sessions": {},
            "vol_regimes": {},
            "trend_day_tiers": {},
            "trend_day_dirs": {},
            "chop_state": {},
            "regime_manifold": {},
            "strategies": {},
            "entry_modes": {},
            "sides": {},
            "asia_viability": {},
            "top_context_tags": {},
        }

    session_counts = Counter()
    vol_regime_counts = Counter()
    trend_tier_counts = Counter()
    trend_dir_counts = Counter()
    chop_counts = Counter()
    manifold_regime_counts = Counter()
    strategy_counts = Counter()
    entry_mode_counts = Counter()
    side_counts = Counter()
    asia_viability_counts = Counter()
    context_tag_counts = Counter()

    for trade in trades:
        strategy = str(trade.get("strategy") or "Unknown")
        side = str(trade.get("side") or "UNKNOWN")
        entry_mode = str(trade.get("entry_mode") or "unknown")
        strategy_counts[strategy] += 1
        side_counts[side] += 1
        entry_mode_counts[entry_mode] += 1

        payload = trade.get("market_conditions")
        if not isinstance(payload, dict) or not payload:
            payload = trade.get("decision_market_conditions")
        if not isinstance(payload, dict):
            payload = {}

        session_name = payload.get("session") or trade.get("session") or "UNKNOWN"
        session_counts[str(session_name)] += 1

        vol_data = payload.get("volatility") if isinstance(payload.get("volatility"), dict) else {}
        vol_regime = (
            vol_data.get("runtime_vol_regime")
            or vol_data.get("signal_vol_regime")
            or trade.get("vol_regime")
            or "UNKNOWN"
        )
        vol_regime_counts[str(vol_regime)] += 1

        trend_data = payload.get("trend_day") if isinstance(payload.get("trend_day"), dict) else {}
        tier = trend_data.get("tier", trade.get("trend_day_tier"))
        if tier is None:
            tier_label = "UNKNOWN"
        else:
            tier_label = str(tier)
        trend_tier_counts[tier_label] += 1
        trend_dir = trend_data.get("dir", trade.get("trend_day_dir")) or "none"
        trend_dir_counts[str(trend_dir)] += 1

        chop_data = payload.get("chop") if isinstance(payload.get("chop"), dict) else {}
        chop_state = chop_data.get("is_choppy")
        if chop_state is True:
            chop_label = "CHOPPY"
        elif chop_state is False:
            chop_label = "TRENDING"
        else:
            chop_label = "UNKNOWN"
        chop_counts[chop_label] += 1

        manifold_data = (
            payload.get("regime_manifold") if isinstance(payload.get("regime_manifold"), dict) else {}
        )
        manifold_regime = manifold_data.get("regime") or "UNKNOWN"
        manifold_regime_counts[str(manifold_regime)] += 1

        asia_data = payload.get("asia") if isinstance(payload.get("asia"), dict) else {}
        asia_viable = asia_data.get("viable")
        if asia_viable is True:
            asia_label = "viable"
        elif asia_viable is False:
            asia_label = "not_viable"
        else:
            asia_label = "unknown"
        asia_viability_counts[asia_label] += 1

        context_tag = payload.get("context_tag")
        if context_tag:
            context_tag_counts[str(context_tag)] += 1

    return {
        "trades": int(len(trades)),
        "sessions": _sorted_counter_dict(session_counts),
        "vol_regimes": _sorted_counter_dict(vol_regime_counts),
        "trend_day_tiers": _sorted_counter_dict(trend_tier_counts),
        "trend_day_dirs": _sorted_counter_dict(trend_dir_counts),
        "chop_state": _sorted_counter_dict(chop_counts),
        "regime_manifold": _sorted_counter_dict(manifold_regime_counts),
        "strategies": _sorted_counter_dict(strategy_counts),
        "entry_modes": _sorted_counter_dict(entry_mode_counts),
        "sides": _sorted_counter_dict(side_counts),
        "asia_viability": _sorted_counter_dict(asia_viability_counts),
        "top_context_tags": _sorted_counter_dict(context_tag_counts, top_n),
    }


def log_mfe_tp_anomalies(trades: list[dict], max_examples: int = 5) -> None:
    if not trades:
        return
    epsilon = 1e-9
    anomalies: list[dict] = []
    for trade in trades:
        tp = trade.get("tp_dist")
        mfe = trade.get("mfe_points")
        if tp is None or mfe is None:
            continue
        exit_reason = trade.get("exit_reason", "")
        if mfe + epsilon >= tp and exit_reason not in ("take", "take_gap"):
            anomalies.append(trade)
    if not anomalies:
        return
    logging.warning(
        "Backtest validation: %s trades have MFE>=TP but exit_reason is not take.",
        len(anomalies),
    )
    for trade in anomalies[:max_examples]:
        entry_time = trade.get("entry_time")
        entry_str = entry_time.isoformat() if isinstance(entry_time, dt.datetime) else entry_time
        logging.warning(
            "MFE>=TP anomaly: %s %s %s tp=%.2f mfe=%.2f exit=%s",
            entry_str,
            trade.get("strategy", "Unknown"),
            trade.get("side", ""),
            float(trade.get("tp_dist", 0.0)),
            float(trade.get("mfe_points", 0.0)),
            trade.get("exit_reason", "unknown"),
        )


def sanitize_filename(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe) or "symbol"


def save_backtest_report(
    stats: dict,
    symbol: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    output_dir: Optional[Path] = None,
) -> Path:
    report_dir = output_dir or (Path(__file__).resolve().parent / "backtest_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(NY_TZ).strftime("%Y%m%d_%H%M%S")
    start_tag = start_time.strftime("%Y%m%d_%H%M")
    end_tag = end_time.strftime("%Y%m%d_%H%M")
    safe_symbol = sanitize_filename(symbol)
    filename = f"backtest_{safe_symbol}_{start_tag}_{end_tag}_{timestamp}.json"
    report_path = report_dir / filename
    monte_carlo_path = report_path.with_name(f"{report_path.stem}_monte_carlo.json")
    baseline_comparison_path = report_path.with_name(
        f"{report_path.stem}_baseline_comparison.json"
    )
    gemini_recommendation_path = report_path.with_name(
        f"{report_path.stem}_gemini_recommendation.json"
    )
    monte_carlo_summary: dict = {
        "enabled": bool(BACKTEST_MONTE_CARLO_ENABLED),
        "status": "disabled",
        "report_path": None,
    }
    baseline_comparison_summary: dict = {
        "enabled": bool(BACKTEST_BASELINE_COMPARISON_ENABLED),
        "status": "disabled",
        "report_path": None,
    }
    gemini_recommendation_summary: dict = {
        "enabled": bool(BACKTEST_GEMINI_RECOMMENDER_ENABLED),
        "status": "disabled",
        "report_path": None,
    }
    if BACKTEST_MONTE_CARLO_ENABLED:
        try:
            monte_carlo_summary = _build_monte_carlo_summary(
                stats.get("trade_log", []) or [],
                stats,
                simulations=BACKTEST_MONTE_CARLO_SIMULATIONS,
                seed=BACKTEST_MONTE_CARLO_SEED,
                starting_balance=BACKTEST_MONTE_CARLO_START_BALANCE,
            )
            monte_carlo_payload = {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "symbol": symbol,
                "range_start": start_time.isoformat(),
                "range_end": end_time.isoformat(),
                "report_path": str(report_path),
                "summary": monte_carlo_summary,
            }
            _write_json_report(monte_carlo_path, monte_carlo_payload)
            monte_carlo_summary = {
                **monte_carlo_summary,
                "report_path": str(monte_carlo_path),
            }
        except Exception as exc:
            logging.warning("Backtest Monte Carlo export failed (%s): %s", monte_carlo_path, exc)
            monte_carlo_summary = {
                "enabled": True,
                "status": "error",
                "error": str(exc),
                "report_path": None,
            }

    current_summary = {
        "equity": safe_float(stats.get("equity"), 0.0),
        "trades": int(safe_float(stats.get("trades"), 0.0)),
        "wins": int(safe_float(stats.get("wins"), 0.0)),
        "losses": int(safe_float(stats.get("losses"), 0.0)),
        "winrate": safe_float(stats.get("winrate"), 0.0),
        "max_drawdown": safe_float(stats.get("max_drawdown"), 0.0),
        "gross_profit": safe_float(stats.get("gross_profit"), 0.0),
        "gross_loss": safe_float(stats.get("gross_loss"), 0.0),
        "profit_factor": safe_float(stats.get("profit_factor"), 0.0),
        "avg_trade_net": safe_float(stats.get("avg_trade_net"), 0.0),
        "trade_sqn": safe_float(stats.get("trade_sqn"), 0.0),
        "trade_sharpe_like": safe_float(stats.get("trade_sharpe_like"), 0.0),
        "daily_sharpe": safe_float(stats.get("daily_sharpe"), 0.0),
        "daily_sortino": safe_float(stats.get("daily_sortino"), 0.0),
        "trading_days": int(safe_float(stats.get("trading_days"), 0.0)),
        "path": str(report_path),
        "source": "current_run_summary",
    }

    baseline_comparison_payload_for_prompt: Optional[dict] = None
    if BACKTEST_BASELINE_COMPARISON_ENABLED:
        try:
            baseline_path = _select_baseline_report_path(
                report_dir=report_dir,
                symbol=symbol,
                start_tag=start_tag,
                end_tag=end_tag,
                current_report_path=report_path,
            )
            if baseline_path is None:
                baseline_comparison_summary = {
                    "enabled": True,
                    "status": "no_baseline_found",
                    "report_path": None,
                }
            else:
                baseline_summary = _extract_summary_from_backtest_report(baseline_path)
                if not isinstance(baseline_summary, dict):
                    baseline_comparison_summary = {
                        "enabled": True,
                        "status": "baseline_parse_failed",
                        "report_path": None,
                        "baseline_path": str(baseline_path),
                    }
                else:
                    comparison = _build_baseline_comparison(current_summary, baseline_summary)
                    baseline_payload = {
                        "created_at": dt.datetime.now(NY_TZ).isoformat(),
                        "symbol": symbol,
                        "range_start": start_time.isoformat(),
                        "range_end": end_time.isoformat(),
                        "current_report_path": str(report_path),
                        "baseline_report_path": str(baseline_path),
                        "comparison": comparison,
                    }
                    _write_json_report(baseline_comparison_path, baseline_payload)
                    baseline_comparison_summary = {
                        "enabled": True,
                        "status": "ok",
                        "report_path": str(baseline_comparison_path),
                        "baseline_path": str(baseline_path),
                        "delta": comparison.get("delta", {}),
                    }
                    baseline_comparison_payload_for_prompt = comparison
        except Exception as exc:
            logging.warning(
                "Backtest baseline comparison export failed (%s): %s",
                baseline_comparison_path,
                exc,
            )
            baseline_comparison_summary = {
                "enabled": True,
                "status": "error",
                "error": str(exc),
                "report_path": None,
            }

    if BACKTEST_GEMINI_RECOMMENDER_ENABLED:
        try:
            base_dir = Path(__file__).resolve().parent
            code_context_rows = _read_code_context_payload(base_dir)
            prompt = _build_backtest_quant_report_prompt(
                backtest_summary=current_summary,
                monte_carlo_summary=monte_carlo_summary,
                baseline_comparison=baseline_comparison_payload_for_prompt,
                market_conditions_summary=stats.get("market_conditions_summary", {}),
                code_context_rows=code_context_rows,
            )
            result = _call_backtest_gemini_recommender(prompt)
            code_context_meta = []
            for row in code_context_rows:
                code_context_meta.append(
                    {
                        "path": str(row.get("path", "")),
                        "exists": bool(row.get("exists", False)),
                        "chars_total": int(safe_float(row.get("chars_total"), 0.0)),
                        "chars_excerpt": int(safe_float(row.get("chars_excerpt"), 0.0)),
                    }
                )
            recommendation_payload = {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "symbol": symbol,
                "range_start": start_time.isoformat(),
                "range_end": end_time.isoformat(),
                "report_path": str(report_path),
                "inputs": {
                    "backtest_summary": current_summary,
                    "monte_carlo_summary": monte_carlo_summary,
                    "baseline_comparison": baseline_comparison_payload_for_prompt,
                    "market_conditions_summary": stats.get("market_conditions_summary", {}),
                    "code_context_meta": code_context_meta,
                    "prompt_preview": prompt[:4000],
                },
                "recommendation": result,
            }
            _write_json_report(gemini_recommendation_path, recommendation_payload)
            gemini_recommendation_summary = {
                "enabled": True,
                "status": str(result.get("status", "unknown")),
                "report_path": str(gemini_recommendation_path),
                "model": result.get("model"),
                "reason": result.get("reason"),
            }
        except Exception as exc:
            logging.warning(
                "Backtest Gemini recommendation export failed (%s): %s",
                gemini_recommendation_path,
                exc,
            )
            gemini_recommendation_summary = {
                "enabled": True,
                "status": "error",
                "error": str(exc),
                "report_path": None,
            }

    payload = {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "symbol": symbol,
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "summary": {
            "equity": stats.get("equity"),
            "trades": stats.get("trades"),
            "wins": stats.get("wins"),
            "losses": stats.get("losses"),
            "winrate": stats.get("winrate"),
            "max_drawdown": stats.get("max_drawdown"),
            "gross_profit": stats.get("gross_profit"),
            "gross_loss": stats.get("gross_loss"),
            "profit_factor": stats.get("profit_factor"),
            "avg_trade_net": stats.get("avg_trade_net"),
            "trade_sqn": stats.get("trade_sqn"),
            "trade_sharpe_like": stats.get("trade_sharpe_like"),
            "daily_sharpe": stats.get("daily_sharpe"),
            "daily_sortino": stats.get("daily_sortino"),
            "trading_days": stats.get("trading_days"),
            "cancelled": stats.get("cancelled"),
        },
        "assumptions": {
            "contracts": CONTRACTS,
            "point_value": POINT_VALUE,
            "fees_per_20_contracts": FEES_PER_20_CONTRACTS,
            "bar_signal": "close",
            "entry": "next_open",
            "bar_minutes": stats.get("bar_minutes"),
            "max_stoploss_cap_points": stats.get("max_stoploss_cap_points"),
            "max_stoploss_cap_disabled_for_mlphysics": stats.get(
                "max_stoploss_cap_disabled_for_mlphysics"
            ),
            "max_stoploss_cap_disabled_for_de3_v2": stats.get(
                "max_stoploss_cap_disabled_for_de3_v2"
            ),
            "drawdown_size_scaling_enabled": stats.get("drawdown_size_scaling_enabled"),
            "drawdown_size_scaling_start_usd": stats.get("drawdown_size_scaling_start_usd"),
            "drawdown_size_scaling_max_usd": stats.get("drawdown_size_scaling_max_usd"),
            "drawdown_size_scaling_base_contracts": stats.get(
                "drawdown_size_scaling_base_contracts"
            ),
            "drawdown_size_scaling_min_contracts": stats.get(
                "drawdown_size_scaling_min_contracts"
            ),
            "entry_window_block_enabled": stats.get("entry_window_block_enabled"),
            "entry_window_block_hours_et": stats.get("entry_window_block_hours_et"),
            "force_flat_enabled": stats.get("force_flat_enabled"),
            "force_flat_time_et": stats.get("force_flat_time_et"),
            "fast_mode": stats.get("fast_mode"),
            "symbol_mode": stats.get("symbol_mode"),
            "symbol_distribution": stats.get("symbol_distribution"),
            "selected_strategies": ((stats.get("selection") or {}).get("strategies")),
            "selected_filters": ((stats.get("selection") or {}).get("filters")),
        },
        "report": stats.get("report", ""),
        "trade_log": stats.get("trade_log", []),
        "market_conditions_summary": stats.get("market_conditions_summary", {}),
        "ml_diagnostics": stats.get("ml_diagnostics", []),
        "ml_diagnostics_summary": stats.get("ml_diagnostics_summary", {}),
        "flip_confidence": stats.get("flip_confidence", {}),
        "de3_veto_summary": stats.get("de3_veto_summary"),
        "de3_veto_counterfactual": stats.get("de3_veto_counterfactual"),
        "de3_meta_summary": stats.get("de3_meta_summary"),
        "de3_manifold_adaptation_summary": stats.get("de3_manifold_adaptation_summary"),
        "de3_backtest_admission_summary": stats.get("de3_backtest_admission_summary"),
        "de3_variant_adaptation_summary": stats.get("de3_variant_adaptation_summary"),
        "de3_meta_counterfactual": stats.get("de3_meta_counterfactual"),
        "de3_decisions_export": stats.get("de3_decisions_export"),
        "monte_carlo": monte_carlo_summary,
        "baseline_comparison": baseline_comparison_summary,
        "gemini_recommendation": gemini_recommendation_summary,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return report_path


def _eval_fast_strategy_task(task):
    strat, strat_name, history_df, vix_slice = task
    signal = None
    try:
        if strat_name == "VIXReversionStrategy":
            signal = strat.on_bar(history_df, vix_slice)
        else:
            signal = strat.on_bar(history_df)
    except Exception:
        signal = None
    return strat, strat_name, signal


def _eval_standard_strategy_task(task):
    strat, strat_name, history_df, mnq_slice = task
    signal = None
    try:
        if strat_name == "SMTStrategy":
            signal = strat.on_bar(history_df, mnq_slice)
        else:
            signal = strat.on_bar(history_df)
    except Exception:
        signal = None
    return strat, strat_name, signal


def run_backtest(
    df: pd.DataFrame,
    start_time: dt.datetime,
    end_time: dt.datetime,
    mnq_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
    cancel_event: Optional["threading.Event"] = None,
    progress_every: int = 100,
    enabled_strategies: Optional[set[str]] = None,
    enabled_filters: Optional[set[str]] = None,
    export_de3_decisions: bool = False,
    de3_decisions_top_k: int = 5,
    de3_decisions_out: Optional[str] = None,
    de3_bucket_bracket_overrides: Optional[dict] = None,
) -> dict:
    configure_risk()
    # Backtest-only strategy hooks can key off this flag without affecting live.
    CONFIG["_BACKTEST_ACTIVE"] = True
    CONFIG.setdefault("GEMINI", {})["enabled"] = False
    CONFIG["DYNAMIC_SL_MULTIPLIER"] = 1.0
    CONFIG["DYNAMIC_TP_MULTIPLIER"] = 1.0
    strategy_selection = None
    if enabled_strategies is not None:
        strategy_selection = {
            str(name).strip()
            for name in enabled_strategies
            if str(name).strip()
        }
    multiplier_schedule, multiplier_dataset_info = _load_backtest_multiplier_schedule()
    if multiplier_schedule is not None and strategy_selection:
        excluded_names = [
            name
            for name in strategy_selection
            if _multiplier_strategy_excluded({"strategy": name}, strategy_hint=name)
        ]
        if len(excluded_names) == len(strategy_selection):
            multiplier_schedule = None
            multiplier_dataset_info = dict(multiplier_dataset_info or {})
            multiplier_dataset_info["enabled"] = False
            multiplier_dataset_info["bars_with_multiplier"] = 0
            multiplier_dataset_info["disabled_for_selected_strategies"] = True
            multiplier_dataset_info["disabled_reason"] = "all_selected_strategies_excluded"
            logging.info(
                "Backtest Gemini multiplier replay disabled: all selected strategies are excluded prefixes (%s).",
                ",".join(sorted(strategy_selection)),
            )
    filter_selection = None
    if enabled_filters is not None:
        filter_selection = {
            str(name).strip()
            for name in enabled_filters
            if str(name).strip()
        }
    selected_strategy_names = (
        sorted(strategy_selection)
        if strategy_selection is not None
        else list(BACKTEST_SELECTABLE_STRATEGIES)
    )
    selected_filter_names = (
        sorted(filter_selection)
        if filter_selection is not None
        else list(BACKTEST_SELECTABLE_FILTERS)
    )
    de3_decision_export_enabled = bool(export_de3_decisions)
    try:
        de3_decision_export_top_k = max(1, int(de3_decisions_top_k))
    except Exception:
        de3_decision_export_top_k = 5
    de3_decision_rows: list[dict] = []
    de3_decision_rows_written = 0
    de3_decisions_out_path: Optional[Path] = None
    de3_decisions_writer = None
    de3_decisions_file_handle = None
    de3_decisions_buffer: list[dict] = []
    de3_decisions_buffer_limit = 2048
    de3_decision_fieldnames = [
        "decision_id",
        "timestamp",
        "de3_version",
        "family_mode",
        "session",
        "side_considered",
        "chosen",
        "abstained",
        "abstain_reason",
        "rank",
        "family_rank",
        "family_id",
        "chosen_family_id",
        "family_score",
        "family_context_ev",
        "family_confidence",
        "family_prior",
        "family_profile",
        "family_member_count",
        "feasible_family_count",
        "feasible_family_ids",
        "family_context_inputs",
        "family_artifact",
        "family_role",
        "family_candidate_count",
        "family_candidate_set_gt_1",
        "family_candidate_source",
        "candidate_rank_before_adjustments",
        "choice_path_mode",
        "score_path_inconsistency_flag",
        "any_post_processing_adjustment_applied",
        "runner_up_available",
        "chosen_vs_runner_up_score_delta",
        "chosen_prior_component",
        "chosen_trusted_context_component",
        "chosen_evidence_adjustment",
        "chosen_adaptive_component",
        "chosen_competition_diversity_adjustment",
        "chosen_family_compatibility_component",
        "chosen_pre_adjustment_score",
        "chosen_final_family_score",
        "chosen_context_trusted",
        "chosen_support_tier",
        "chosen_compatibility_tier",
        "chosen_session_compatibility_tier",
        "chosen_timeframe_compatibility_tier",
        "chosen_strategy_type_compatibility_tier",
        "chosen_exploration_bonus_applied",
        "chosen_dominance_penalty_applied",
        "chosen_context_advantage_capped",
        "runner_up_family_id",
        "runner_up_prior_component",
        "runner_up_trusted_context_component",
        "runner_up_evidence_adjustment",
        "runner_up_adaptive_component",
        "runner_up_competition_diversity_adjustment",
        "runner_up_family_compatibility_component",
        "runner_up_pre_adjustment_score",
        "runner_up_final_family_score",
        "runner_up_context_trusted",
        "runner_up_support_tier",
        "runner_up_compatibility_tier",
        "runner_up_session_compatibility_tier",
        "runner_up_timeframe_compatibility_tier",
        "runner_up_strategy_type_compatibility_tier",
        "runner_up_exploration_bonus_applied",
        "runner_up_dominance_penalty_applied",
        "runner_up_context_advantage_capped",
        "canonical_member_id",
        "member_local_score",
        "family_context_support_ratio",
        "family_context_support_tier",
        "family_local_support_tier",
        "family_context_sample_count",
        "family_context_weight",
        "family_context_trusted",
        "family_context_fallback_priors",
        "family_active_context_buckets",
        "family_profile_used",
        "family_profile_fallback",
        "family_usability_state",
        "family_usability_component",
        "family_evidence_support_tier",
        "family_competition_status",
        "family_usability_adjustment",
        "family_suppression_reason",
        "family_compatibility_component",
        "family_compatibility_tier",
        "family_session_compatibility_tier",
        "family_timeframe_compatibility_tier",
        "family_strategy_type_compatibility_tier",
        "family_entered_via_compatible_band",
        "family_exact_match_eligible",
        "family_compatible_band_eligible",
        "family_incompatible_excluded",
        "family_excluded_by_candidate_cap",
        "family_eligibility_tier",
        "family_preliminary_family_score",
        "family_preliminary_compatibility_penalty_component",
        "family_entered_pre_cap_pool",
        "family_survived_cap",
        "family_cap_drop_reason",
        "family_cap_tier_slot_used",
        "family_final_competition_pool_flag",
        "pre_cap_candidate_count",
        "post_cap_candidate_count",
        "exact_match_survived_count",
        "compatible_band_survived_count",
        "compatible_band_dropped_by_cap_count",
        "base_family_score",
        "diversity_adjustment",
        "competition_diversity_adjustment",
        "final_family_score",
        "recent_chosen_share",
        "exploration_bonus",
        "dominance_penalty",
        "exploration_bonus_applied",
        "dominance_penalty_applied",
        "competition_margin_qualified",
        "context_advantage_capped",
        "context_advantage_cap_delta",
        "close_competition_decision",
        "bootstrap_competition_used_decision",
        "family_monopoly_active",
        "family_monopoly_top_share",
        "family_monopoly_top_family_id",
        "family_monopoly_unique_count",
        "monopoly_canonical_force_applied",
        "family_chosen_flag",
        "local_bracket_adaptation_mode",
        "local_bracket_adaptation_enabled",
        "local_bracket_override_applied",
        "local_member_count_within_family",
        "local_edge_component",
        "local_structural_component",
        "local_bracket_suitability_component",
        "local_confidence_component",
        "local_payoff_component",
        "local_final_member_score",
        "canonical_fallback_used",
        "why_non_anchor_beat_anchor",
        "why_anchor_forced",
        "no_local_alternative",
        "family_prior_eligible",
        "family_prior_eligibility_reason",
        "family_competition_eligible",
        "family_competition_eligibility_reason",
        "family_bootstrap_competition_included",
        "family_bootstrap_included",
        "family_catastrophic_prior",
        # Backward-compatible aliases.
        "family_eligible",
        "family_eligibility_reason",
        "ctx_volatility_regime",
        "ctx_chop_trend_regime",
        "ctx_compression_expansion_regime",
        "ctx_confidence_band",
        "ctx_rvol_liquidity_state",
        "ctx_session_substate",
        "ctx_atr_ratio",
        "ctx_vwap_dist_atr",
        "ctx_price_location",
        "ctx_rvol_ratio",
        "ctx_hour_et",
        "sub_strategy",
        "timeframe",
        "strategy_type",
        "thresh",
        "sl",
        "tp",
        "edge_points",
        "edge_gap_points",
        "runtime_rank_score",
        "structural_score",
        "final_score",
        "bucket_score",
        "stop_like_share",
        "loss_share",
        "profitable_block_ratio",
        "worst_block_avg_pnl",
        "worst_block_pf",
    ]
    # Keep DE3v4 runtime fields in the canonical decision CSV schema so
    # lane-aware trainers can consume them directly.
    de3_v4_decision_fieldnames = [
        "de3_v4_route_decision",
        "de3_v4_route_confidence",
        "de3_v4_selected_lane",
        "de3_v4_selected_variant_id",
        "de3_v4_lane_candidate_count",
        "de3_v4_lane_selection_reason",
        "de3_v4_bracket_mode",
        "de3_v4_selected_sl",
        "de3_v4_selected_tp",
        "de3_v4_canonical_default_used",
        "de3_v4_runtime_mode",
        "de3_v4_route_scores",
        "de3_v4_execution_policy_tier",
        "de3_v4_execution_quality_score",
        "de3_v4_execution_policy_reason",
        "de3_v4_execution_policy_source",
        "de3_v4_execution_policy_enforce_veto",
        "de3_v4_execution_policy_soft_pass",
        "de3_v4_execution_policy_hard_limit_triggered",
        "de3_v4_execution_policy_hard_limit_reason",
        "de3_v4_execution_policy_components",
        "de3_v4_entry_model_enabled",
        "de3_v4_entry_model_allow",
        "de3_v4_entry_model_tier",
        "de3_v4_entry_model_reason",
        "de3_v4_entry_model_score",
        "de3_v4_entry_model_threshold",
        "de3_v4_entry_model_threshold_base",
        "de3_v4_entry_model_threshold_scope_offset",
        "de3_v4_entry_model_scope",
        "de3_v4_entry_model_stats",
        "de3_v4_entry_model_components",
        "de3_v4_profit_gate_lane",
        "de3_v4_profit_gate_session",
        "de3_v4_profit_gate_min_samples_eff",
        "de3_v4_profit_gate_max_p_loss_std_eff",
        "de3_v4_profit_gate_min_ev_lcb_points_eff",
        "de3_v4_profit_gate_min_ev_mean_points_eff",
        "de3_v4_profit_gate_soft_pass",
        "de3_v4_profit_gate_catastrophic_block",
        "decision_side_model_enabled",
        "decision_side_model_name",
        "decision_side_model_application_mode",
        "decision_side_model_predicted_action",
        "decision_side_model_side_pattern",
        "decision_side_model_baseline_side_guess",
        "decision_side_model_match_count",
        "decision_side_model_long_score",
        "decision_side_model_short_score",
        "decision_side_model_no_trade_score",
        "decision_side_model_prior_score",
        "decision_side_model_prior_component_total",
    ]
    for _field in de3_v4_decision_fieldnames:
        if _field not in de3_decision_fieldnames:
            de3_decision_fieldnames.append(_field)
    de3_entry_decision_fieldnames = [
        "de3_entry_ret1_atr",
        "de3_entry_body_pos1",
        "de3_entry_lower_wick_ratio",
        "de3_entry_upper_wick_ratio",
        "de3_entry_upper1_ratio",
        "de3_entry_body1_ratio",
        "de3_entry_close_pos1",
        "de3_entry_flips5",
        "de3_entry_down3",
        "de3_entry_range10_atr",
        "de3_entry_dist_low5_atr",
        "de3_entry_dist_high5_atr",
        "de3_entry_vol1_rel20",
        "de3_entry_atr14",
        "de3_entry_filter_hit",
        "de3_entry_filter_reason",
    ]
    for _field in de3_entry_decision_fieldnames:
        if _field not in de3_decision_fieldnames:
            de3_decision_fieldnames.append(_field)
    if de3_decision_export_enabled:
        export_target = str(de3_decisions_out or "./reports/de3_decisions.csv").strip()
        if not export_target:
            export_target = "./reports/de3_decisions.csv"
        if export_target.lower() == "none":
            export_target = "./reports/de3_decisions.csv"
        de3_decisions_out_path = Path(export_target).expanduser()
        if not de3_decisions_out_path.is_absolute():
            de3_decisions_out_path = Path(__file__).resolve().parent / de3_decisions_out_path
        if str(de3_decisions_out_path.suffix).lower() != ".csv":
            de3_decisions_out_path = de3_decisions_out_path.with_suffix(".csv")
        # Prevent silent overwrite across runs when using the default filename.
        # If output target is the canonical default name, append a run stamp.
        if str(de3_decisions_out_path.name).lower() == "de3_decisions.csv":
            start_stamp = pd.Timestamp(start_time).strftime("%Y%m%d_%H%M")
            end_stamp = pd.Timestamp(end_time).strftime("%Y%m%d_%H%M")
            run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            de3_decisions_out_path = de3_decisions_out_path.with_name(
                f"de3_decisions_{start_stamp}_{end_stamp}_{run_stamp}.csv"
            )
        de3_decisions_out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            de3_decisions_file_handle = de3_decisions_out_path.open("w", newline="", encoding="utf-8")
            de3_decisions_writer = csv.DictWriter(
                de3_decisions_file_handle,
                fieldnames=de3_decision_fieldnames,
                extrasaction="ignore",
            )
            de3_decisions_writer.writeheader()
        except Exception as exc:
            logging.warning("DE3 decision export stream init failed (%s): %s", de3_decisions_out_path, exc)
            de3_decisions_writer = None
            if de3_decisions_file_handle is not None:
                try:
                    de3_decisions_file_handle.close()
                except Exception:
                    pass
                de3_decisions_file_handle = None

    de3_bucket_bracket_override_map: dict[str, dict[str, float]] = {}
    if isinstance(de3_bucket_bracket_overrides, dict):
        for raw_key, raw_payload in de3_bucket_bracket_overrides.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            payload = raw_payload if isinstance(raw_payload, dict) else {}
            try:
                sl_val = float(payload.get("sl", np.nan))
                tp_val = float(payload.get("tp", np.nan))
            except Exception:
                continue
            if not (math.isfinite(sl_val) and math.isfinite(tp_val)):
                continue
            if sl_val <= 0.0 or tp_val <= 0.0:
                continue
            de3_bucket_bracket_override_map[key] = {
                "sl": float(sl_val),
                "tp": float(tp_val),
            }

    def _flush_de3_decision_buffer() -> None:
        nonlocal de3_decision_rows_written
        if de3_decisions_writer is None or not de3_decisions_buffer:
            return
        de3_decisions_writer.writerows(de3_decisions_buffer)
        de3_decision_rows_written += int(len(de3_decisions_buffer))
        de3_decisions_buffer.clear()

    def _record_de3_decision_row(row: dict) -> None:
        if not de3_decision_export_enabled:
            return
        if de3_decisions_writer is not None:
            de3_decisions_buffer.append(row)
            if len(de3_decisions_buffer) >= de3_decisions_buffer_limit:
                _flush_de3_decision_buffer()
            return
        de3_decision_rows.append(row)

    def strategy_enabled(strategy_name: str) -> bool:
        if strategy_selection is None:
            return True
        return str(strategy_name).strip() in strategy_selection

    ml_only_strategy_names = {"MLPhysicsStrategy", "MLPhysicsLegacyExperimentStrategy"}
    ml_only_requested = (
        isinstance(strategy_selection, set)
        and len(strategy_selection) == 1
        and bool(strategy_selection & ml_only_strategy_names)
    )
    ml_only_diag_cfg = CONFIG.get("BACKTEST_ML_ONLY_DIAGNOSTIC_PROFILE", {}) or {}
    ml_only_diag_active = False

    init_started_at = time.perf_counter()
    console_status_enabled = bool(progress_cb is None and CONFIG.get("BACKTEST_CONSOLE_STATUS", True))
    console_progress_enabled = bool(progress_cb is None and CONFIG.get("BACKTEST_CONSOLE_PROGRESS", True))
    try:
        console_progress_every_sec = max(
            1.0,
            float(CONFIG.get("BACKTEST_CONSOLE_PROGRESS_EVERY_SEC", 15.0) or 15.0),
        )
    except Exception:
        console_progress_every_sec = 15.0

    def emit_init_status(message: str, **extra) -> None:
        elapsed = max(0.0, time.perf_counter() - init_started_at)
        if progress_cb is not None:
            payload = {
                "type": "status",
                "message": f"{message} (t+{elapsed:.1f}s)",
            }
            if extra:
                payload.update(extra)
            try:
                progress_cb(payload)
            except Exception:
                pass
            return
        if console_status_enabled:
            print(f"[backtest:init] {message} (t+{elapsed:.1f}s)", flush=True)

    def _format_hms(seconds: float) -> str:
        if not math.isfinite(seconds) or seconds < 0:
            return "?"
        total = int(seconds + 0.5)
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def filter_enabled(filter_name: str) -> bool:
        if filter_selection is None:
            return True
        return str(filter_name).strip() in filter_selection

    live_report_cfg = CONFIG.get("BACKTEST_LIVE_REPORT", {}) or {}
    live_report_enabled = bool(live_report_cfg.get("enabled", False))
    try:
        live_report_every_sec = max(1.0, float(live_report_cfg.get("write_every_sec", 15) or 15))
    except Exception:
        live_report_every_sec = 15.0
    live_report_include_trade_log = bool(live_report_cfg.get("include_trade_log", False))
    live_report_path: Optional[Path] = None
    live_report_write_error = False
    live_report_last_write = 0.0
    # Progress callback report generation can dominate runtime on long runs.
    # Cache the heavy textual report and refresh on a time/trade-count cadence.
    try:
        progress_report_every_sec = max(1.0, float(CONFIG.get("BACKTEST_PROGRESS_REPORT_EVERY_SEC", 8.0) or 8.0))
    except Exception:
        progress_report_every_sec = 8.0
    progress_report_refresh_on_trade = bool(CONFIG.get("BACKTEST_PROGRESS_REFRESH_ON_TRADE", False))
    progress_report_cache = ""
    progress_report_last_build = 0.0
    progress_report_last_trade_count = -1
    progress_recent_trades_cache: list[str] = []
    progress_recent_trades_last_trade_count = -1
    live_trade_log_cache: list[dict] = []
    live_trade_log_last_count = 0
    if live_report_enabled:
        try:
            custom_output_dir = live_report_cfg.get("output_dir")
            if custom_output_dir:
                report_dir = Path(str(custom_output_dir)).expanduser()
            else:
                report_dir = Path(__file__).resolve().parent / "backtest_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = dt.datetime.now(NY_TZ).strftime("%Y%m%d_%H%M%S")
            start_tag = start_time.strftime("%Y%m%d_%H%M")
            end_tag = end_time.strftime("%Y%m%d_%H%M")
            live_report_path = report_dir / f"backtest_live_{start_tag}_{end_tag}_{timestamp}.json"
            emit_init_status(f"Live report enabled: {live_report_path.name}")
        except Exception as exc:
            live_report_enabled = False
            logging.warning("Backtest live report disabled (init failure): %s", exc)

    if ml_only_requested and bool(ml_only_diag_cfg.get("enabled", False)):
        keep_filters_default = [
            "FixedSLTP",
            "TargetFeasibility",
            "VolatilityGuardrail",
            "MLVolRegimeGuard",
        ]
        keep_filters_raw = ml_only_diag_cfg.get("filters", keep_filters_default) or keep_filters_default
        keep_filters = {
            str(name).strip()
            for name in keep_filters_raw
            if str(name).strip() in BACKTEST_SELECTABLE_FILTERS
        }
        # Respect UI/user-selected filters by default.
        # Set BACKTEST_ML_ONLY_DIAGNOSTIC_PROFILE.force_filter_subset=true to restore old behavior.
        force_filter_subset = bool(ml_only_diag_cfg.get("force_filter_subset", False))
        if force_filter_subset and keep_filters:
            if filter_selection is None:
                filter_selection = set(keep_filters)
            else:
                filter_selection = {name for name in filter_selection if name in keep_filters}
        ml_only_diag_active = True
        logging.info(
            "ML-only diagnostic profile active%s: filters=%s",
            " (forced subset)" if force_filter_subset else "",
            ",".join(sorted(filter_selection)) if filter_selection is not None else "ALL",
        )
    # Backtest-only DE3 version override. Keeps live defaults untouched.
    de3_bt_override = str(CONFIG.get("BACKTEST_DE3_VERSION_OVERRIDE", "") or "").strip().lower()
    if de3_bt_override in {"v1", "v2", "v3", "v4"}:
        CONFIG["DE3_VERSION"] = de3_bt_override
        if de3_bt_override == "v2":
            de3_v2_cfg = dict(CONFIG.get("DE3_V2", {}) or {})
            de3_v2_cfg["enabled"] = True
            CONFIG["DE3_V2"] = de3_v2_cfg
        elif de3_bt_override == "v3":
            de3_v3_cfg = dict(CONFIG.get("DE3_V3", {}) or {})
            de3_v3_cfg["enabled"] = True
            CONFIG["DE3_V3"] = de3_v3_cfg
        elif de3_bt_override == "v4":
            de3_v4_cfg = dict(CONFIG.get("DE3_V4", {}) or {})
            de3_v4_cfg["enabled"] = True
            CONFIG["DE3_V4"] = de3_v4_cfg
        logging.info("Backtest DE3 override active: DE3_VERSION=%s", de3_bt_override)
    elif de3_bt_override:
        logging.warning(
            "Ignoring invalid BACKTEST_DE3_VERSION_OVERRIDE=%s (expected v1, v2, v3, or v4).",
            de3_bt_override,
        )
    de3_v4_side_bt_cfg = CONFIG.get("BACKTEST_DE3_V4_DECISION_SIDE_MODEL", {}) or {}
    if (
        str(CONFIG.get("DE3_VERSION", "") or "").strip().lower() == "v4"
        and isinstance(de3_v4_side_bt_cfg, dict)
        and bool(de3_v4_side_bt_cfg.get("enabled", False))
    ):
        de3_v4_cfg = dict(CONFIG.get("DE3_V4", {}) or {})
        de3_v4_cfg["enabled"] = True
        de3_v4_runtime_cfg = (
            dict(de3_v4_cfg.get("runtime", {}))
            if isinstance(de3_v4_cfg.get("runtime", {}), dict)
            else {}
        )
        merged_decision_side_cfg = (
            dict(de3_v4_runtime_cfg.get("decision_side_model", {}))
            if isinstance(de3_v4_runtime_cfg.get("decision_side_model", {}), dict)
            else {}
        )
        for key, value in de3_v4_side_bt_cfg.items():
            merged_decision_side_cfg[str(key)] = value
        de3_v4_runtime_cfg["decision_side_model"] = merged_decision_side_cfg
        de3_v4_cfg["runtime"] = de3_v4_runtime_cfg
        CONFIG["DE3_V4"] = de3_v4_cfg
        logging.info(
            "Backtest DE3v4 decision-side override active: mode=%s side_patterns=%s",
            merged_decision_side_cfg.get("application_mode", ""),
            ",".join(
                str(v)
                for v in (
                    merged_decision_side_cfg.get("apply_side_patterns", [])
                    if isinstance(merged_decision_side_cfg.get("apply_side_patterns", []), (list, tuple, set))
                    else []
                )
            )
            or "ALL",
        )
    # Backtest-only: DE3 warning logs are high-volume; keep live defaults unchanged.
    CONFIG["DE3_VERBOSE_WARNINGS"] = bool(CONFIG.get("BACKTEST_DE3_VERBOSE_WARNINGS", False))
    # Backtest-only: PenaltyBox warning logs are high-volume; keep live defaults unchanged.
    CONFIG["PENALTY_BOX_VERBOSE_WARNINGS"] = bool(
        CONFIG.get("BACKTEST_PENALTY_BOX_VERBOSE_WARNINGS", False)
    )
    speed_cfg = CONFIG.get("BACKTEST_SPEED_PROFILE", {}) or {}
    speed_enabled = bool(speed_cfg.get("enabled", False))
    if ml_only_diag_active and bool(ml_only_diag_cfg.get("disable_speed_profile", True)):
        speed_enabled = False
    root_logger = logging.getLogger()
    prev_root_level = root_logger.level
    if speed_enabled and bool(speed_cfg.get("suppress_warnings", True)):
        root_logger.setLevel(logging.ERROR)
    fast_cfg = CONFIG.get("BACKTEST_FAST_MODE", {}) or {}
    fast_enabled = bool(fast_cfg.get("enabled"))
    bar_stride = int(fast_cfg.get("bar_stride", 1) or 1)
    if bar_stride < 1:
        bar_stride = 1
    skip_mfe_mae = bool(fast_cfg.get("skip_mfe_mae", False))
    ml_diag_cfg = CONFIG.get("BACKTEST_ML_DIAGNOSTICS", {}) or {}
    ml_diag_enabled = bool(ml_diag_cfg.get("enabled", True))
    ml_diag_include_no_signal = bool(ml_diag_cfg.get("include_no_signal", False))
    ml_diag_max_records = int(ml_diag_cfg.get("max_records", 0) or 0)
    if speed_enabled and bool(speed_cfg.get("disable_ml_diagnostics", True)):
        ml_diag_enabled = False
    if ml_only_diag_active and bool(ml_only_diag_cfg.get("force_ml_diagnostics", True)):
        ml_diag_enabled = True
        ml_diag_include_no_signal = bool(ml_only_diag_cfg.get("include_no_signal", True))
        try:
            ml_diag_cap = int(ml_only_diag_cfg.get("max_diag_records", ml_diag_max_records) or 0)
        except Exception:
            ml_diag_cap = ml_diag_max_records
        ml_diag_max_records = max(0, ml_diag_cap)
    market_snap_cfg = CONFIG.get("BACKTEST_MARKET_SNAPSHOTS", {}) or {}
    market_snapshots_enabled = bool(market_snap_cfg.get("enabled", True))
    market_summary_enabled = bool(market_snap_cfg.get("summary_enabled", True))
    if speed_enabled and bool(speed_cfg.get("disable_market_snapshots", True)):
        market_snapshots_enabled = False
    if speed_enabled and bool(speed_cfg.get("disable_market_condition_summary", True)):
        market_summary_enabled = False
    speed_disable_flip_confidence = bool(
        speed_enabled and bool(speed_cfg.get("disable_flip_confidence", True))
    )
    try:
        ml_eval_stride = int(speed_cfg.get("ml_eval_stride", 1) or 1) if speed_enabled else 1
    except Exception:
        ml_eval_stride = 1
    if ml_eval_stride < 1:
        ml_eval_stride = 1
    oos_force_full_ml_eval = bool(CONFIG.get("BACKTEST_OOS_FORCE_FULL_ML_EVAL", False))
    if ml_only_diag_active and bool(ml_only_diag_cfg.get("force_full_ml_eval", True)):
        oos_force_full_ml_eval = True
    if oos_force_full_ml_eval:
        ml_eval_stride = 1
    de3_debug_exceptions = bool(CONFIG.get("BACKTEST_DE3_DEBUG_EXCEPTIONS", False))

    asia_calib_cfg = CONFIG.get("BACKTEST_ASIA_CALIBRATIONS", {}) or {}
    asia_calib_enabled = bool(asia_calib_cfg.get("enabled", False))
    smooth_trend_cfg = CONFIG.get("BACKTEST_SMOOTH_TREND_ASIA", {}) or {}
    smooth_trend_enabled = bool(smooth_trend_cfg.get("enabled", False))

    # Backtest-scoped dist ML runtime tuning (performance + GPU feed targeting).
    try:
        backtest_dist_bars = int(CONFIG.get("BACKTEST_ML_DIST_INPUT_BARS", 3000) or 0)
    except Exception:
        backtest_dist_bars = 0
    if speed_enabled:
        try:
            speed_dist_bars = int(speed_cfg.get("dist_input_bars", 0) or 0)
        except Exception:
            speed_dist_bars = 0
        if speed_dist_bars > 0:
            backtest_dist_bars = speed_dist_bars
    if backtest_dist_bars > 0:
        CONFIG["ML_PHYSICS_DIST_MAX_BARS"] = int(backtest_dist_bars)
    try:
        backtest_gpu_target = float(CONFIG.get("BACKTEST_GPU_TARGET_FRACTION", 0.75) or 0.75)
    except Exception:
        backtest_gpu_target = 0.75
    backtest_gpu_target = min(1.0, max(0.10, backtest_gpu_target))
    CONFIG["ML_PHYSICS_DIST_XGB_GPU_TARGET_FRACTION"] = float(backtest_gpu_target)
    CONFIG["ML_PHYSICS_DIST_XGB_GPU_ENABLED"] = bool(CONFIG.get("ML_PHYSICS_DIST_XGB_GPU_ENABLED", True))
    if speed_enabled:
        logging.info(
            "Backtest speed profile enabled: dist_input_bars=%s ml_eval_stride=%s "
            "oos_full_ml_eval=%s ml_diag=%s flip_conf=%s market_summary=%s",
            backtest_dist_bars,
            ml_eval_stride,
            oos_force_full_ml_eval,
            ml_diag_enabled,
            not speed_disable_flip_confidence,
            market_summary_enabled,
        )

    htf_fvg_cfg = CONFIG.get("HTF_FVG_FILTER", {}) or {}
    htf_fvg_enabled_backtest = bool(htf_fvg_cfg.get("enabled_backtest", True))
    ict_cfg = CONFIG.get("ICT_MODEL", {}) or {}
    ict_enabled_backtest = bool(ict_cfg.get("enabled_backtest", False))

    # Backtest-only: enable ML vol-split sessions without changing live defaults
    backtest_split_sessions = CONFIG.get("ML_PHYSICS_VOL_SPLIT_BACKTEST_SESSIONS", [])
    if backtest_split_sessions:
        vol_split = CONFIG.setdefault("ML_PHYSICS_VOL_SPLIT", {})
        sessions = set(vol_split.get("sessions", []))
        sessions.update(backtest_split_sessions)
        vol_split["sessions"] = sorted(sessions)
        vol_split["enabled"] = True

    # Backtest-only: disable ML vol-split for specific sessions
    backtest_unsplit_sessions = set(CONFIG.get("ML_PHYSICS_VOL_UNSPLIT_BACKTEST_SESSIONS", []))
    if backtest_unsplit_sessions:
        vol_split = CONFIG.setdefault("ML_PHYSICS_VOL_SPLIT", {})
        sessions = set(vol_split.get("sessions", []))
        sessions.difference_update(backtest_unsplit_sessions)
        vol_split["sessions"] = sorted(sessions)
        if not sessions:
            vol_split["enabled"] = False

    param_scaler.apply_scaling()
    refresh_target_symbol()
    emit_init_status("Building strategy stack...")

    fast_strategies = []
    if strategy_enabled("RegimeAdaptiveStrategy"):
        from regime_strategy import RegimeAdaptiveStrategy

        fast_strategies.append(RegimeAdaptiveStrategy())
    if strategy_enabled("VIXReversionStrategy"):
        from vixmeanreversion import VIXReversionStrategy

        fast_strategies.append(VIXReversionStrategy())
    if strategy_enabled("ImpulseBreakoutStrategy"):
        from impulse_breakout_strategy import ImpulseBreakoutStrategy

        fast_strategies.append(ImpulseBreakoutStrategy())
    if ENABLE_DYNAMIC_ENGINE_1 and strategy_enabled("DynamicEngineStrategy"):
        from dynamic_engine_strategy import DynamicEngineStrategy

        dynamic_engine_strat = DynamicEngineStrategy()
        fast_strategies.append(dynamic_engine_strat)
    if ENABLE_DYNAMIC_ENGINE_2 and strategy_enabled("DynamicEngine2Strategy"):
        from dynamic_engine2_strategy import DynamicEngine2Strategy

        dynamic_engine2_strat = DynamicEngine2Strategy()
        fast_strategies.append(dynamic_engine2_strat)
    dynamic_engine3_strat = None
    if ENABLE_DYNAMIC_ENGINE_3 and strategy_enabled("DynamicEngine3Strategy"):
        from dynamic_signal_engine3 import reset_signal_engine
        from dynamic_engine3_strategy import DynamicEngine3Strategy

        # UI keeps process state between runs; force fresh DE3 DB/config load per backtest.
        reset_signal_engine()
        dynamic_engine3_strat = DynamicEngine3Strategy()
        fast_strategies.append(dynamic_engine3_strat)
    de3_runtime_db_version = "none"
    de3_runtime_is_v2 = False
    de3_runtime_is_v3 = False
    de3_runtime_is_v4 = False
    de3_runtime_family_mode_enabled = False
    de3_runtime_family_artifact = None
    de3_runtime_family_artifact_loaded = False
    de3_runtime_context_profiles_loaded = False
    de3_runtime_enriched_export_required = False
    de3_runtime_context_profile_build = {}
    de3_runtime_active_context_dimensions = []
    de3_runtime_context_trust = {}
    de3_runtime_local_bracket_freeze = {}
    de3_runtime_state_loaded = False
    de3_runtime_state_build = {}
    de3_runtime_export_raw_context_fields = False
    de3_v3_activation_audit = {}
    de3_v3_runtime_path_counters = {}
    de3_v3_bundle_usage_audit = {}
    de3_v3_config_usage_audit = {}
    de3_v3_score_path_audit = {}
    de3_v3_choice_path_audit = {}
    de3_v3_family_score_trace = {}
    de3_v3_member_resolution_audit = {}
    de3_v3_family_eligibility_trace = {}
    de3_v3_family_reachability_summary = {}
    de3_v3_family_compatibility_audit = {}
    de3_v3_pre_cap_candidate_audit = {}
    de3_v3_family_score_component_summary = {}
    de3_v3_family_score_delta_ladder = {}
    de3_v3_runtime_mode_summary = {}
    de3_v3_core_summary = {}
    de3_v3_t6_anchor_report = {}
    de3_v3_satellite_quality_report = {}
    de3_v3_portfolio_increment_report = {}
    de3_v4_activation_audit = {}
    de3_v4_runtime_path_counters = {}
    de3_v4_router_summary = {}
    de3_v4_lane_selection_summary = {}
    de3_v4_bracket_summary = {}
    de3_v4_runtime_mode_summary = {}
    de3_v4_execution_policy_summary = {}
    de3_v4_decision_side_summary = {}
    de3_audit_reports_dir = Path(__file__).resolve().parent / "reports"
    de3_v3_run_id = f"de3v3_{dt.datetime.now(NY_TZ).strftime('%Y%m%d_%H%M%S')}"
    de3_v3_activation_audit_path = de3_audit_reports_dir / "de3_v3_activation_audit.json"
    de3_v3_runtime_path_counters_path = de3_audit_reports_dir / "de3_v3_runtime_path_counters.json"
    de3_v3_bundle_usage_audit_path = de3_audit_reports_dir / "de3_v3_bundle_usage_audit.json"
    de3_v3_config_usage_audit_path = de3_audit_reports_dir / "de3_v3_config_usage_audit.json"
    de3_v3_score_path_audit_path = de3_audit_reports_dir / "de3_v3_score_path_audit.json"
    de3_v3_choice_path_audit_path = de3_audit_reports_dir / "de3_v3_choice_path_audit.json"
    de3_v3_refined_vs_raw_audit_path = de3_audit_reports_dir / "de3_v3_refined_vs_raw_audit.json"
    de3_v3_family_score_trace_path = de3_audit_reports_dir / "de3_v3_family_score_trace.csv"
    de3_v3_family_competition_health_path = (
        de3_audit_reports_dir / "de3_v3_family_competition_health.json"
    )
    de3_v3_member_resolution_audit_path = (
        de3_audit_reports_dir / "de3_v3_member_resolution_audit.json"
    )
    de3_v3_family_eligibility_trace_path = (
        de3_audit_reports_dir / "de3_v3_family_eligibility_trace.csv"
    )
    de3_v3_family_reachability_summary_path = (
        de3_audit_reports_dir / "de3_v3_family_reachability_summary.json"
    )
    de3_v3_family_compatibility_audit_path = (
        de3_audit_reports_dir / "de3_v3_family_compatibility_audit.json"
    )
    de3_v3_pre_cap_candidate_audit_path = (
        de3_audit_reports_dir / "de3_v3_pre_cap_candidate_audit.json"
    )
    de3_v3_family_score_component_summary_path = (
        de3_audit_reports_dir / "de3_v3_family_score_component_summary.json"
    )
    de3_v3_family_score_delta_ladder_path = (
        de3_audit_reports_dir / "de3_v3_family_score_delta_ladder.json"
    )
    de3_v3_runtime_mode_summary_path = (
        de3_audit_reports_dir / "de3_v3_runtime_mode_summary.json"
    )
    de3_v3_core_summary_path = (
        de3_audit_reports_dir / "de3_v3_core_summary.json"
    )
    de3_v3_t6_anchor_report_path = (
        de3_audit_reports_dir / "de3_v3_t6_anchor_report.json"
    )
    de3_v3_satellite_quality_report_path = (
        de3_audit_reports_dir / "de3_v3_satellite_quality_report.json"
    )
    de3_v3_portfolio_increment_report_path = (
        de3_audit_reports_dir / "de3_v3_portfolio_increment_report.json"
    )
    de3_v3_diff_readiness_path = de3_audit_reports_dir / "de3_v3_diff_readiness.json"
    de3_v3_inert_change_summary_path = de3_audit_reports_dir / "de3_v3_inert_change_summary.json"
    de3_v4_activation_audit_path = de3_audit_reports_dir / "de3_v4_activation_audit.json"
    de3_v4_router_summary_path = de3_audit_reports_dir / "de3_v4_router_summary.json"
    de3_v4_lane_selection_summary_path = de3_audit_reports_dir / "de3_v4_lane_selection_summary.json"
    de3_v4_bracket_summary_path = de3_audit_reports_dir / "de3_v4_bracket_summary.json"
    de3_v4_runtime_mode_summary_path = de3_audit_reports_dir / "de3_v4_runtime_mode_summary.json"
    de3_v4_runtime_path_counters_path = de3_audit_reports_dir / "de3_v4_runtime_path_counters.json"
    de3_v4_execution_policy_summary_path = de3_audit_reports_dir / "de3_v4_execution_policy_summary.json"
    de3_v4_decision_side_summary_path = de3_audit_reports_dir / "de3_v4_decision_side_summary.json"
    if dynamic_engine3_strat is not None:
        try:
            de3_runtime_db_version = str(dynamic_engine3_strat.get_db_version() or "unknown")
        except Exception:
            try:
                de3_runtime_db_version = str(
                    getattr(getattr(dynamic_engine3_strat, "engine", None), "db_version", "unknown")
                )
            except Exception:
                de3_runtime_db_version = "unknown"
        de3_runtime_is_v2 = de3_runtime_db_version.strip().lower().startswith("v2")
        de3_runtime_is_v3 = de3_runtime_db_version.strip().lower().startswith("v3")
        de3_runtime_is_v4 = de3_runtime_db_version.strip().lower().startswith("v4")
        runtime_meta = {}
        try:
            runtime_meta = (
                dynamic_engine3_strat.get_runtime_metadata()
                if hasattr(dynamic_engine3_strat, "get_runtime_metadata")
                else {}
            )
        except Exception:
            runtime_meta = {}
        if isinstance(runtime_meta, dict):
            de3_runtime_family_mode_enabled = bool(runtime_meta.get("family_mode_enabled", False))
            artifact_val = runtime_meta.get("family_artifact_path")
            de3_runtime_family_artifact = str(artifact_val) if artifact_val else None
            de3_runtime_family_artifact_loaded = bool(runtime_meta.get("family_artifact_loaded", False))
            de3_runtime_context_profiles_loaded = bool(runtime_meta.get("context_profiles_loaded", False))
            de3_runtime_enriched_export_required = bool(runtime_meta.get("enriched_export_required", False))
            de3_runtime_export_raw_context_fields = bool(
                runtime_meta.get("export_raw_context_fields_in_decision_journal", False)
            )
            cp_build = runtime_meta.get("context_profile_build")
            if isinstance(cp_build, dict):
                de3_runtime_context_profile_build = dict(cp_build)
            active_dims = runtime_meta.get("active_context_dimensions")
            if isinstance(active_dims, (list, tuple)):
                de3_runtime_active_context_dimensions = [str(x) for x in active_dims]
            trust_meta = runtime_meta.get("context_trust")
            if isinstance(trust_meta, dict):
                de3_runtime_context_trust = dict(trust_meta)
            freeze_meta = runtime_meta.get("local_bracket_freeze")
            if isinstance(freeze_meta, dict):
                de3_runtime_local_bracket_freeze = dict(freeze_meta)
            de3_runtime_state_loaded = bool(runtime_meta.get("runtime_state_loaded", False))
            state_build_meta = runtime_meta.get("runtime_state_build")
            if isinstance(state_build_meta, dict):
                de3_runtime_state_build = dict(state_build_meta)
            runtime_counters = runtime_meta.get("runtime_path_counters")
            if isinstance(runtime_counters, dict):
                de3_v3_runtime_path_counters = dict(runtime_counters)
            bundle_usage = runtime_meta.get("bundle_usage_audit")
            if isinstance(bundle_usage, dict):
                de3_v3_bundle_usage_audit = dict(bundle_usage)
            config_usage = runtime_meta.get("config_usage_audit")
            if isinstance(config_usage, dict):
                de3_v3_config_usage_audit = dict(config_usage)
            score_path_audit = runtime_meta.get("score_path_audit")
            if isinstance(score_path_audit, dict):
                de3_v3_score_path_audit = dict(score_path_audit)
            choice_path_audit = runtime_meta.get("choice_path_audit")
            if isinstance(choice_path_audit, dict):
                de3_v3_choice_path_audit = dict(choice_path_audit)
            family_score_trace = runtime_meta.get("family_score_trace")
            if isinstance(family_score_trace, dict):
                de3_v3_family_score_trace = dict(family_score_trace)
            member_resolution_audit = runtime_meta.get("member_resolution_audit")
            if isinstance(member_resolution_audit, dict):
                de3_v3_member_resolution_audit = dict(member_resolution_audit)
            compatibility_audit = runtime_meta.get("family_compatibility_audit")
            if isinstance(compatibility_audit, dict):
                de3_v3_family_compatibility_audit = dict(compatibility_audit)
            pre_cap_audit = runtime_meta.get("pre_cap_candidate_audit")
            if isinstance(pre_cap_audit, dict):
                de3_v3_pre_cap_candidate_audit = dict(pre_cap_audit)
            component_summary = runtime_meta.get("family_score_component_summary")
            if isinstance(component_summary, dict):
                de3_v3_family_score_component_summary = dict(component_summary)
            score_delta_ladder = runtime_meta.get("family_score_delta_ladder")
            if isinstance(score_delta_ladder, dict):
                de3_v3_family_score_delta_ladder = dict(score_delta_ladder)
            runtime_mode_summary = runtime_meta.get("runtime_mode_summary")
            if isinstance(runtime_mode_summary, dict):
                de3_v3_runtime_mode_summary = dict(runtime_mode_summary)
            core_summary = runtime_meta.get("core_summary")
            if isinstance(core_summary, dict):
                de3_v3_core_summary = dict(core_summary)
            t6_anchor_report = runtime_meta.get("t6_anchor_report")
            if isinstance(t6_anchor_report, dict):
                de3_v3_t6_anchor_report = dict(t6_anchor_report)
            satellite_quality_report = runtime_meta.get("satellite_quality_report")
            if isinstance(satellite_quality_report, dict):
                de3_v3_satellite_quality_report = dict(satellite_quality_report)
            portfolio_increment_report = runtime_meta.get("portfolio_increment_report")
            if isinstance(portfolio_increment_report, dict):
                de3_v3_portfolio_increment_report = dict(portfolio_increment_report)
            activation_audit = runtime_meta.get("activation_audit")
            if isinstance(activation_audit, dict):
                de3_v3_activation_audit = dict(activation_audit)
            v4_activation = runtime_meta.get("v4_activation_audit")
            if isinstance(v4_activation, dict):
                de3_v4_activation_audit = dict(v4_activation)
            v4_counters = runtime_meta.get("v4_runtime_path_counters")
            if isinstance(v4_counters, dict):
                de3_v4_runtime_path_counters = dict(v4_counters)
            v4_router = runtime_meta.get("v4_router_summary")
            if isinstance(v4_router, dict):
                de3_v4_router_summary = dict(v4_router)
            v4_lane_summary = runtime_meta.get("v4_lane_selection_summary")
            if isinstance(v4_lane_summary, dict):
                de3_v4_lane_selection_summary = dict(v4_lane_summary)
            v4_bracket = runtime_meta.get("v4_bracket_summary")
            if isinstance(v4_bracket, dict):
                de3_v4_bracket_summary = dict(v4_bracket)
            v4_runtime_mode = runtime_meta.get("v4_runtime_mode_summary")
            if isinstance(v4_runtime_mode, dict):
                de3_v4_runtime_mode_summary = dict(v4_runtime_mode)
            v4_execution_policy = runtime_meta.get("v4_execution_policy_summary")
            if isinstance(v4_execution_policy, dict):
                de3_v4_execution_policy_summary = dict(v4_execution_policy)
        de3_v3_cfg_live = CONFIG.get("DE3_V3", {}) if isinstance(CONFIG.get("DE3_V3", {}), dict) else {}
        de3_v3_family_scoring = (
            de3_v3_cfg_live.get("family_scoring", {})
            if isinstance(de3_v3_cfg_live.get("family_scoring", {}), dict)
            else {}
        )
        de3_v3_prior_eligibility = (
            de3_v3_cfg_live.get("prior_eligibility", {})
            if isinstance(de3_v3_cfg_live.get("prior_eligibility", {}), dict)
            else (
                de3_v3_cfg_live.get("family_eligibility", {})
                if isinstance(de3_v3_cfg_live.get("family_eligibility", {}), dict)
                else {}
            )
        )
        de3_v3_family_competition = (
            de3_v3_cfg_live.get("family_competition", {})
            if isinstance(de3_v3_cfg_live.get("family_competition", {}), dict)
            else {}
        )
        de3_v3_competition_balance = (
            de3_v3_family_competition.get("family_competition_balance", {})
            if isinstance(de3_v3_family_competition.get("family_competition_balance", {}), dict)
            else {}
        )
        de3_v3_family_candidate_cap = (
            de3_v3_family_competition.get("family_candidate_cap", {})
            if isinstance(de3_v3_family_competition.get("family_candidate_cap", {}), dict)
            else {}
        )
        de3_v3_core_cfg = (
            de3_v3_cfg_live.get("de3v3_core", {})
            if isinstance(de3_v3_cfg_live.get("de3v3_core", {}), dict)
            else {}
        )
        de3_v3_sat_cfg = (
            de3_v3_cfg_live.get("de3v3_satellites", {})
            if isinstance(de3_v3_cfg_live.get("de3v3_satellites", {}), dict)
            else {}
        )
        de3_v3_bloat_cfg = (
            de3_v3_cfg_live.get("bloat_control", {})
            if isinstance(de3_v3_cfg_live.get("bloat_control", {}), dict)
            else {}
        )
        logging.info(
            (
                "Backtest DE3 runtime DB: version=%s (v2=%s v3=%s family_mode=%s family_artifact=%s loaded=%s "
                "v4=%s "
                "artifact_kind=%s bundle_loaded=%s context_profiles_loaded=%s enriched_export_required=%s runtime_state_loaded=%s "
                "runtime_use_refined=%s loaded_universe=%s raw_families=%s retained_families=%s raw_members=%s retained_members=%s "
                "fallback_to_priors=%s prior_eligibility_enabled=%s competition_floor=%s/%s export_raw_context_fields=%s)"
            ),
            de3_runtime_db_version,
            de3_runtime_is_v2,
            de3_runtime_is_v3,
            de3_runtime_is_v4,
            de3_runtime_family_mode_enabled,
            de3_runtime_family_artifact,
            de3_runtime_family_artifact_loaded,
            runtime_meta.get("artifact_kind") if isinstance(runtime_meta, dict) else None,
            runtime_meta.get("bundle_loaded") if isinstance(runtime_meta, dict) else None,
            de3_runtime_context_profiles_loaded,
            de3_runtime_enriched_export_required,
            de3_runtime_state_loaded,
            runtime_meta.get("runtime_use_refined") if isinstance(runtime_meta, dict) else None,
            runtime_meta.get("loaded_universe") if isinstance(runtime_meta, dict) else None,
            runtime_meta.get("raw_family_count") if isinstance(runtime_meta, dict) else None,
            runtime_meta.get("retained_family_count") if isinstance(runtime_meta, dict) else None,
            runtime_meta.get("raw_member_count") if isinstance(runtime_meta, dict) else None,
            runtime_meta.get("retained_member_count") if isinstance(runtime_meta, dict) else None,
            bool(de3_v3_family_scoring.get("fallback_to_priors_when_profile_weak", True)),
            bool(de3_v3_prior_eligibility.get("enabled", True)),
            bool(de3_v3_family_competition.get("use_bootstrap_family_competition_floor", True)),
            int(de3_v3_family_competition.get("bootstrap_min_competing_families", 3) or 3),
            de3_runtime_export_raw_context_fields,
        )
        if de3_runtime_is_v4:
            de3_v4_cfg_live = CONFIG.get("DE3_V4", {}) if isinstance(CONFIG.get("DE3_V4", {}), dict) else {}
            v4_core_cfg_live = (
                de3_v4_cfg_live.get("core", {})
                if isinstance(de3_v4_cfg_live.get("core", {}), dict)
                else {}
            )
            v4_runtime_cfg_live = (
                de3_v4_cfg_live.get("runtime", {})
                if isinstance(de3_v4_cfg_live.get("runtime", {}), dict)
                else {}
            )
            logging.info(
                "Backtest DE3v4 runtime: bundle=%s loaded=%s mode=%s core_enabled=%s anchors=%s",
                (runtime_meta.get("v4_status", {}) if isinstance(runtime_meta.get("v4_status", {}), dict) else {}).get("bundle_path"),
                (runtime_meta.get("v4_status", {}) if isinstance(runtime_meta.get("v4_status", {}), dict) else {}).get("bundle_loaded"),
                (de3_v4_runtime_mode_summary or {}).get("mode", v4_core_cfg_live.get("default_runtime_mode")),
                bool(v4_core_cfg_live.get("enabled", True)),
                list(v4_core_cfg_live.get("anchor_family_ids", [])),
            )
            logging.info(
                "Backtest DE3v4 modules: router=%s lane_selector=%s bracket_module=%s",
                bool(v4_runtime_cfg_live.get("router", {})),
                bool(v4_runtime_cfg_live.get("lane_selector", {})),
                bool(v4_runtime_cfg_live.get("bracket_module", {})),
            )
        if de3_runtime_is_v3:
            de3_v3_context_cfg = (
                de3_v3_cfg_live.get("context_profiles", {})
                if isinstance(de3_v3_cfg_live.get("context_profiles", {}), dict)
                else {}
            )
            logging.info(
                "Backtest DE3v3 context settings: require_enriched=%s min_bucket_samples=%s strong_bucket_samples=%s",
                bool(de3_v3_context_cfg.get("require_enriched_export_for_runtime", False)),
                de3_v3_context_cfg.get("min_bucket_samples"),
                de3_v3_context_cfg.get("strong_bucket_samples"),
            )
            logging.info(
                "Backtest DE3v3 active context dimensions: %s | trust=%s",
                de3_runtime_active_context_dimensions,
                de3_runtime_context_trust,
            )
            logging.info(
                "Backtest DE3v3 local bracket freeze settings: %s",
                de3_runtime_local_bracket_freeze,
            )
            logging.info(
                "Backtest DE3v3 runtime mode: mode=%s core_enabled=%s satellites_enabled=%s core_families=%s retained_satellites=%s",
                (de3_v3_runtime_mode_summary or {}).get("mode"),
                bool(de3_v3_core_cfg.get("enabled", True)),
                bool(de3_v3_sat_cfg.get("enabled", True)),
                (de3_v3_runtime_mode_summary or {}).get("core_family_ids_loaded", []),
                (de3_v3_runtime_mode_summary or {}).get("retained_satellite_family_ids", []),
            )
            logging.info(
                "Backtest DE3v3 bloat-control defaults: balancing=%s exploration=%s dominance=%s monopoly_force=%s compat_slot_pressure=%s",
                bool(de3_v3_bloat_cfg.get("enable_family_competition_balancing", False)),
                bool(de3_v3_bloat_cfg.get("enable_exploration_bonus", False)),
                bool(de3_v3_bloat_cfg.get("enable_dominance_penalty", False)),
                bool(de3_v3_bloat_cfg.get("enable_monopoly_canonical_force", False)),
                bool(de3_v3_bloat_cfg.get("enable_compatibility_tier_slot_pressure", False)),
            )
            logging.info(
                "Backtest DE3v3 prior eligibility settings: min_support=%s min_best_pf=%s min_best_pbr=%s "
                "min_best_worst_pf=%s min_best_worst_avg=%s max_median_dd=%s max_median_loss_share=%s",
                de3_v3_prior_eligibility.get("min_total_support_trades"),
                de3_v3_prior_eligibility.get("min_best_member_profit_factor"),
                de3_v3_prior_eligibility.get("min_best_member_profitable_block_ratio"),
                de3_v3_prior_eligibility.get("min_best_member_worst_block_pf"),
                de3_v3_prior_eligibility.get("min_best_member_worst_block_avg_pnl"),
                de3_v3_prior_eligibility.get("max_median_drawdown_norm"),
                de3_v3_prior_eligibility.get("max_median_loss_share"),
            )
            logging.info(
                "Backtest DE3v3 family competition floor: enabled=%s min_competing_families=%s",
                bool(de3_v3_family_competition.get("use_bootstrap_family_competition_floor", True)),
                int(de3_v3_family_competition.get("bootstrap_min_competing_families", 3) or 3),
            )
            de3_v3_compatibility_bands = (
                de3_v3_family_competition.get("compatibility_bands", {})
                if isinstance(de3_v3_family_competition.get("compatibility_bands", {}), dict)
                else {}
            )
            logging.info(
                (
                    "Backtest DE3v3 compatibility bands: include_exact_and_compatible_only=%s "
                    "max_family_candidates=%s compatible_family_max_count=%s compatible_penalty=%.4f "
                    "session_nearby_hours=%.2f timeframe_delta=%s timeframe_ratio=%.2f strategy_type_allow_related=%s"
                ),
                bool(de3_v3_family_competition.get("include_exact_and_compatible_only", True)),
                int(de3_v3_family_competition.get("max_family_candidates_per_decision", 6) or 6),
                int(de3_v3_family_competition.get("compatible_family_max_count", 4) or 4),
                float(de3_v3_family_competition.get("compatible_family_penalty", -0.06) or -0.06),
                float(de3_v3_compatibility_bands.get("session_nearby_max_hour_distance", 6.0) or 6.0),
                int(de3_v3_compatibility_bands.get("timeframe_nearby_max_minutes_delta", 10) or 10),
                float(de3_v3_compatibility_bands.get("timeframe_nearby_max_ratio", 3.0) or 3.0),
                bool(de3_v3_compatibility_bands.get("strategy_type_allow_related", True)),
            )
            logging.info(
                (
                    "Backtest DE3v3 candidate cap: enabled=%s max_total=%s "
                    "min_exact=%s min_compatible=%s max_exact=%s max_compatible=%s "
                    "use_preliminary_score=%s compat_penalty_exact=%.4f compat_penalty_compatible=%.4f "
                    "log_pre_cap_post_cap=%s"
                ),
                bool(de3_v3_family_candidate_cap.get("enabled", True)),
                int(
                    de3_v3_family_candidate_cap.get(
                        "max_total_candidates",
                        de3_v3_family_competition.get("max_family_candidates_per_decision", 6),
                    )
                    or 6
                ),
                int(de3_v3_family_candidate_cap.get("min_exact_match_candidates", 2) or 2),
                int(de3_v3_family_candidate_cap.get("min_compatible_band_candidates", 2) or 2),
                int(
                    de3_v3_family_candidate_cap.get(
                        "max_exact_match_candidates",
                        de3_v3_family_candidate_cap.get(
                            "max_total_candidates",
                            de3_v3_family_competition.get("max_family_candidates_per_decision", 6),
                        ),
                    )
                    or 6
                ),
                int(
                    de3_v3_family_candidate_cap.get(
                        "max_compatible_band_candidates",
                        de3_v3_family_competition.get("compatible_family_max_count", 4),
                    )
                    or 4
                ),
                bool(de3_v3_family_candidate_cap.get("use_preliminary_score_for_cap", True)),
                float(de3_v3_family_candidate_cap.get("compatibility_penalty_exact", 0.0) or 0.0),
                float(
                    de3_v3_family_candidate_cap.get(
                        "compatibility_penalty_compatible",
                        de3_v3_family_competition.get("compatible_family_penalty", -0.06),
                    )
                    or -0.06
                ),
                bool(de3_v3_family_candidate_cap.get("log_pre_cap_post_cap", True)),
            )
            logging.info(
                (
                    "Backtest DE3v3 competition balance: enabled=%s dominance_window=%s "
                    "start_share=%.3f dominance_penalty_max=%.3f low_support_bonus=%.3f "
                    "margin=%.3f cap_context_close=%s max_context_cap=%.3f"
                ),
                bool(de3_v3_competition_balance.get("enabled", True)),
                int(de3_v3_competition_balance.get("dominance_window_size", 160) or 160),
                float(de3_v3_competition_balance.get("dominance_penalty_start_share", 0.55) or 0.55),
                float(
                    de3_v3_competition_balance.get(
                        "dominance_penalty_max",
                        de3_v3_competition_balance.get("max_dominance_penalty", 0.12),
                    )
                    or 0.12
                ),
                float(
                    de3_v3_competition_balance.get(
                        "low_support_exploration_bonus",
                        de3_v3_competition_balance.get("max_exploration_bonus", 0.08),
                    )
                    or 0.08
                ),
                float(de3_v3_competition_balance.get("competition_margin_points", 0.22) or 0.22),
                bool(de3_v3_competition_balance.get("cap_context_advantage_in_close_competition", True)),
                float(de3_v3_competition_balance.get("max_context_advantage_cap", 0.12) or 0.12),
            )
            de3_v3_usable_cfg = (
                de3_v3_cfg_live.get("usable_family_universe", {})
                if isinstance(de3_v3_cfg_live.get("usable_family_universe", {}), dict)
                else {}
            )
            de3_v3_evidence_support_cfg = (
                de3_v3_usable_cfg.get("evidence_support", {})
                if isinstance(de3_v3_usable_cfg.get("evidence_support", {}), dict)
                else {}
            )
            de3_v3_evidence_adjust_cfg = (
                de3_v3_usable_cfg.get("evidence_adjustment", {})
                if isinstance(de3_v3_usable_cfg.get("evidence_adjustment", {}), dict)
                else {}
            )
            logging.info(
                "Backtest DE3v3 evidence model: min_mid_samples=%s strong_samples=%s adjustment=%s",
                de3_v3_evidence_support_cfg.get("min_mid_samples"),
                de3_v3_evidence_support_cfg.get("strong_samples"),
                de3_v3_evidence_adjust_cfg,
            )
            activation_cfg_snapshot = (
                dict(de3_v3_activation_audit.get("config_snapshot_relevant", {}))
                if isinstance(de3_v3_activation_audit.get("config_snapshot_relevant"), dict)
                else {
                    "DE3_V3": dict(de3_v3_cfg_live),
                }
            )
            artifact_path = Path(str(de3_runtime_family_artifact or de3_v3_cfg_live.get("family_db_path", "") or "")).expanduser()
            if not artifact_path.is_absolute():
                artifact_path = Path(__file__).resolve().parent / artifact_path
            artifact_fingerprint = _file_fingerprint(artifact_path)
            activation_payload = dict(de3_v3_activation_audit) if isinstance(de3_v3_activation_audit, dict) else {}
            activation_payload.update({
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "run_id": str(de3_v3_run_id),
                "run_timestamp": dt.datetime.now(NY_TZ).isoformat(),
                "command_line_args": list(sys.argv),
                "active_de3_version": str(de3_runtime_db_version),
                "active_bundle_path": str(artifact_path),
                "bundle_fingerprint": artifact_fingerprint,
                "raw_family_artifact_path": str(
                    de3_v3_cfg_live.get("family_inventory_legacy_path", "") or ""
                ),
                "refined_universe_mode_enabled": bool((de3_v3_cfg_live.get("refined_universe", {}) or {}).get("enabled", True)),
                "runtime_use_refined": bool(runtime_meta.get("runtime_use_refined", False)) if isinstance(runtime_meta, dict) else False,
                "loaded_universe": runtime_meta.get("loaded_universe") if isinstance(runtime_meta, dict) else None,
                "family_candidate_construction_mode": "family_first",
                "family_first_feasibility_mode_active": True,
                "family_scoring_weights_used": dict(de3_v3_family_scoring.get("weights", {}) if isinstance(de3_v3_family_scoring.get("weights", {}), dict) else {}),
                "local_member_scoring_weights_used": dict(
                    ((de3_v3_cfg_live.get("local_member_selection", {}) if isinstance(de3_v3_cfg_live.get("local_member_selection", {}), dict) else {}).get("weights", {}))
                ),
                "active_context_dimensions_used": list(de3_runtime_active_context_dimensions),
                "context_trust_thresholds_used": dict(de3_runtime_context_trust),
                "local_bracket_adaptation_settings_used": dict(de3_runtime_local_bracket_freeze),
                "family_competition_balancing_settings_used": (
                    dict((runtime_meta.get("family_status", {}) or {}).get("competition_balance", {}))
                    if isinstance(runtime_meta, dict)
                    else {}
                ),
                "raw_universe_override_allowed": bool((de3_v3_cfg_live.get("refined_universe", {}) or {}).get("allow_runtime_raw_universe_override", True)),
                "raw_universe_override_active": bool(
                    bool(runtime_meta.get("runtime_use_refined", False))
                    and str(runtime_meta.get("loaded_universe", "raw")) in {"raw", "legacy"}
                ) if isinstance(runtime_meta, dict) else False,
                "fallback_path_active": {
                    "fallback_to_priors_when_profile_weak": bool(de3_v3_family_scoring.get("fallback_to_priors_when_profile_weak", True)),
                    "enriched_export_required": bool(de3_runtime_enriched_export_required),
                    "context_profiles_loaded": bool(de3_runtime_context_profiles_loaded),
                },
                "force_canonical_under_monopoly_active": bool(
                    de3_runtime_local_bracket_freeze.get("force_canonical_when_family_monopoly", False)
                ),
                "debug_audit_modes_enabled": dict(
                    (
                        (de3_v3_cfg_live.get("observability", {}))
                        if isinstance(de3_v3_cfg_live.get("observability", {}), dict)
                        else {}
                    )
                ),
                "exact_relevant_config_snapshot": activation_cfg_snapshot,
                "exact_relevant_config_snapshot_flat": _flatten_dict_for_audit(activation_cfg_snapshot, "DE3_V3_RUNTIME"),
            })
            de3_v3_activation_audit = dict(activation_payload)
            try:
                _write_json_report(de3_v3_activation_audit_path, activation_payload)
            except Exception as exc:
                logging.warning("DE3v3 activation audit export failed (%s): %s", de3_v3_activation_audit_path, exc)
        dynamic_engine3_strat.configure_decision_journal_export(
            enabled=bool(de3_decision_export_enabled),
            top_k=int(de3_decision_export_top_k),
            sink=_record_de3_decision_row,
        )
        dynamic_engine3_strat.configure_de3_v2_bucket_bracket_overrides(
            de3_bucket_bracket_override_map
        )

    ml_runtime_name = None
    ml_strategy_selected = strategy_enabled("MLPhysicsStrategy") or strategy_enabled("MLPhysicsLegacyExperimentStrategy")
    if strategy_enabled("MLPhysicsStrategy") and strategy_enabled("MLPhysicsLegacyExperimentStrategy"):
        logging.warning(
            "Both MLPhysicsStrategy and MLPhysicsLegacyExperimentStrategy were selected; using MLPhysicsStrategy."
        )
    if strategy_enabled("MLPhysicsStrategy"):
        ml_strategy = MLPhysicsStrategy()
        ml_runtime_name = "MLPhysicsStrategy"
    elif strategy_enabled("MLPhysicsLegacyExperimentStrategy"):
        from ml_physics_legacy_experiment_strategy import MLPhysicsLegacyExperimentStrategy

        ml_strategy = MLPhysicsLegacyExperimentStrategy()
        ml_runtime_name = "MLPhysicsLegacyExperimentStrategy"
    else:
        class _DisabledMLStrategy:
            model_loaded = False
            last_eval = None

        ml_strategy = _DisabledMLStrategy()

    standard_strategies = []
    if strategy_enabled("IntradayDipStrategy"):
        from intraday_dip_strategy import IntradayDipStrategy

        standard_strategies.append(IntradayDipStrategy())
    if strategy_enabled("AuctionReversionStrategy"):
        from auction_reversion_strategy import AuctionReversionStrategy

        standard_strategies.append(AuctionReversionStrategy())
    if strategy_enabled("LiquiditySweepStrategy"):
        from liquidity_sweep_strategy import LiquiditySweepStrategy

        standard_strategies.append(LiquiditySweepStrategy())
    if strategy_enabled("ValueAreaBreakoutStrategy"):
        from value_area_breakout_strategy import ValueAreaBreakoutStrategy

        standard_strategies.append(ValueAreaBreakoutStrategy())
    if strategy_enabled("ConfluenceStrategy"):
        from confluence_strategy import ConfluenceStrategy

        standard_strategies.append(ConfluenceStrategy())
    if strategy_enabled("SMTStrategy"):
        from smt_strategy import SMTStrategy

        standard_strategies.append(SMTStrategy())
    if smooth_trend_enabled and strategy_enabled("SmoothTrendAsiaStrategy"):
        from smooth_trend_asia_strategy import SmoothTrendAsiaStrategy

        standard_strategies.append(SmoothTrendAsiaStrategy())
    if ml_strategy.model_loaded and ml_runtime_name in {"MLPhysicsStrategy", "MLPhysicsLegacyExperimentStrategy"}:
        standard_strategies.append(ml_strategy)
    manifold_strategy = None
    aetherflow_strategy = None
    manifold_cfg = CONFIG.get("MANIFOLD_STRATEGY", {}) or {}
    if bool(manifold_cfg.get("enabled_backtest", False)) and strategy_enabled("ManifoldStrategy"):
        from manifold_strategy import ManifoldStrategy

        manifold_strategy = ManifoldStrategy()
        if getattr(manifold_strategy, "model_loaded", False):
            standard_strategies.append(manifold_strategy)
        else:
            logging.warning("ManifoldStrategy enabled_backtest but model artifact is missing.")
    aetherflow_cfg = CONFIG.get("AETHERFLOW_STRATEGY", {}) or {}
    if bool(aetherflow_cfg.get("enabled_backtest", False)) and strategy_enabled("AetherFlowStrategy"):
        from aetherflow_strategy import AetherFlowStrategy

        aetherflow_strategy = AetherFlowStrategy()
        if getattr(aetherflow_strategy, "model_loaded", False):
            standard_strategies.append(aetherflow_strategy)
        else:
            logging.warning("AetherFlowStrategy enabled_backtest but model artifact is missing.")

    loose_strategies = []
    if strategy_enabled("OrbStrategy"):
        from orb_strategy import OrbStrategy

        loose_strategies.append(OrbStrategy())
    if ict_enabled_backtest and strategy_enabled("ICTModelStrategy"):
        from ict_model_strategy import ICTModelStrategy

        loose_strategies.append(ICTModelStrategy())
    elif ict_enabled_backtest and not strategy_enabled("ICTModelStrategy"):
        logging.info("Backtest ICTModelStrategy disabled by strategy selection")
    else:
        logging.info("Backtest ICTModelStrategy disabled via config")

    backtest_workers = _coerce_int(CONFIG.get("BACKTEST_WORKERS", 6), 6)
    parallel_strategy_eval = bool(CONFIG.get("BACKTEST_PARALLEL_STRATEGY_EVAL", False))
    fast_strategy_defs = [(strat, strat.__class__.__name__) for strat in fast_strategies]
    standard_strategy_defs = [(strat, strat.__class__.__name__) for strat in standard_strategies]
    standard_ml_strategy_names = {"MLPhysicsStrategy", "MLPhysicsLegacyExperimentStrategy"}
    standard_ml_defs = [item for item in standard_strategy_defs if item[1] in standard_ml_strategy_names]
    standard_non_ml_defs = [item for item in standard_strategy_defs if item[1] not in standard_ml_strategy_names]
    need_vix_slice = any(name == "VIXReversionStrategy" for _, name in fast_strategy_defs)
    need_mnq_slice = any(name == "SMTStrategy" for _, name in standard_non_ml_defs)
    try:
        vix_slice_tail_bars = int(CONFIG.get("BACKTEST_VIX_SLICE_TAIL_BARS", 128) or 128)
    except Exception:
        vix_slice_tail_bars = 128
    if vix_slice_tail_bars < 32:
        vix_slice_tail_bars = 32
    try:
        mnq_slice_tail_bars = int(CONFIG.get("BACKTEST_MNQ_SLICE_TAIL_BARS", 1800) or 1800)
    except Exception:
        mnq_slice_tail_bars = 1800
    if mnq_slice_tail_bars < 128:
        mnq_slice_tail_bars = 128
    for strat, strat_name in standard_non_ml_defs:
        if strat_name == "SMTStrategy":
            try:
                smt_lb = int(getattr(strat, "lookback_minutes", 0) or 0)
            except Exception:
                smt_lb = 0
            if smt_lb > 0:
                mnq_slice_tail_bars = max(mnq_slice_tail_bars, smt_lb + 10)
    de3_only_fast_mode = bool(
        dynamic_engine3_strat is not None
        and strategy_selection == {"DynamicEngine3Strategy"}
        and isinstance(filter_selection, set)
        and len(filter_selection) == 0
        and not standard_non_ml_defs
        and not standard_ml_defs
        and not loose_strategies
    )
    de3_ml_fast_mode = bool(
        dynamic_engine3_strat is not None
        and strategy_selection == {"DynamicEngine3Strategy", "MLPhysicsStrategy"}
        and isinstance(filter_selection, set)
        and len(filter_selection) == 0
        and not standard_non_ml_defs
        and len(standard_ml_defs) == 1
        and not loose_strategies
    )
    manifold_fast_strategy = None
    manifold_only_fast_mode = bool(
        strategy_selection == {"ManifoldStrategy"}
        and not fast_strategy_defs
        and len(standard_non_ml_defs) == 1
        and standard_non_ml_defs[0][1] == "ManifoldStrategy"
        and not standard_ml_defs
        and not loose_strategies
        and bool(manifold_cfg.get("backtest_hard_filters_only", False))
    )
    aetherflow_only_fast_mode = bool(
        strategy_selection == {"AetherFlowStrategy"}
        and not fast_strategy_defs
        and len(standard_non_ml_defs) == 1
        and standard_non_ml_defs[0][1] == "AetherFlowStrategy"
        and not standard_ml_defs
        and not loose_strategies
        and bool(aetherflow_cfg.get("backtest_hard_filters_only", False))
    )
    ml_only_fast_mode = bool(
        ml_only_requested
        and not fast_strategy_defs
        and not standard_non_ml_defs
        and not loose_strategies
    )
    manifold_fast_history_bars = 0
    aetherflow_fast_history_bars = 0
    if manifold_only_fast_mode:
        manifold_fast_strategy = standard_non_ml_defs[0][0]
        default_manifold_history_bars = getattr(
            getattr(manifold_fast_strategy, "_aux_cache", None),
            "max_rows",
            2500,
        )
        try:
            manifold_fast_history_bars = int(
                manifold_cfg.get("backtest_history_bars", default_manifold_history_bars)
                or default_manifold_history_bars
            )
        except Exception:
            manifold_fast_history_bars = int(default_manifold_history_bars)
        manifold_fast_history_bars = max(
            120,
            int(getattr(manifold_fast_strategy, "min_bars", 250) or 250) + 64,
            int(manifold_fast_history_bars),
        )
    if aetherflow_only_fast_mode and aetherflow_strategy is not None:
        default_aetherflow_history_bars = int(getattr(aetherflow_strategy, "max_feature_bars", 900) or 900)
        try:
            aetherflow_fast_history_bars = int(
                aetherflow_cfg.get("backtest_history_bars", default_aetherflow_history_bars)
                or default_aetherflow_history_bars
            )
        except Exception:
            aetherflow_fast_history_bars = int(default_aetherflow_history_bars)
        aetherflow_fast_history_bars = max(
            120,
            int(getattr(aetherflow_strategy, "min_bars", 320) or 320) + 64,
            int(aetherflow_fast_history_bars),
        )
    try:
        de3_fast_history_bars = int(CONFIG.get("BACKTEST_DE3_FAST_HISTORY_BARS", 3000) or 3000)
    except Exception:
        de3_fast_history_bars = 3000
    if de3_fast_history_bars < 120:
        de3_fast_history_bars = 120
    try:
        backtest_max_history_bars = int(CONFIG.get("BACKTEST_MAX_HISTORY_BARS", 0) or 0)
    except Exception:
        backtest_max_history_bars = 0
    if backtest_max_history_bars < 0:
        backtest_max_history_bars = 0
    try:
        de3_history_cap_bars = int(CONFIG.get("BACKTEST_DE3_HISTORY_MAX_BARS", 20000) or 20000)
    except Exception:
        de3_history_cap_bars = 20000
    if de3_history_cap_bars < 0:
        de3_history_cap_bars = 0
    if de3_only_fast_mode:
        logging.info(
            "Backtest fast path enabled: DE3-only with no enabled filters (history_bars=%s).",
            de3_fast_history_bars,
        )
    if de3_ml_fast_mode:
        logging.info(
            "Backtest fast path enabled: DE3+MLPhysics with no enabled filters (history_bars=%s).",
            de3_fast_history_bars,
        )
    if manifold_only_fast_mode:
        logging.info(
            "Backtest fast path enabled: ManifoldStrategy-only hard-filter mode (history_bars=%s).",
            manifold_fast_history_bars,
        )
    if aetherflow_only_fast_mode:
        logging.info(
            "Backtest fast path enabled: AetherFlowStrategy-only hard-filter mode (history_bars=%s).",
            aetherflow_fast_history_bars,
        )
    strategy_executor = None
    strategy_eval_workers = 1
    internal_parallel_strategies = {
        name
        for _, name in standard_non_ml_defs
        if name in {"ManifoldStrategy"}
    }
    if parallel_strategy_eval and internal_parallel_strategies:
        parallel_strategy_eval = False
        logging.info(
            "Backtest strategy thread pool disabled (internal parallelism detected): %s",
            ",".join(sorted(internal_parallel_strategies)),
        )
    if parallel_strategy_eval and backtest_workers > 1:
        eval_tasks = len(fast_strategy_defs) + len(standard_non_ml_defs)
        if eval_tasks > 1:
            strategy_eval_workers = max(1, min(int(backtest_workers), int(eval_tasks)))
            strategy_executor = cf.ThreadPoolExecutor(
                max_workers=strategy_eval_workers,
                thread_name_prefix="bt-strat",
            )
            logging.info("Backtest strategy eval workers=%s", strategy_eval_workers)

    exec_disabled_raw = CONFIG.get("STRATEGY_EXECUTION_DISABLED", []) or []
    exec_disabled = {str(name).strip().lower() for name in exec_disabled_raw if name}
    exec_disabled_by_session_cfg = CONFIG.get("STRATEGY_EXECUTION_DISABLED_BY_SESSION", {}) or {}
    exec_disabled_by_session: dict[str, set[str]] = {}
    if isinstance(exec_disabled_by_session_cfg, dict):
        for key, sessions in exec_disabled_by_session_cfg.items():
            if not key:
                continue
            if not isinstance(sessions, (list, tuple, set)):
                sessions = [sessions]
            sess_set = {str(s).upper() for s in sessions if s}
            if sess_set:
                exec_disabled_by_session[str(key).strip().lower()] = sess_set

    def execution_disabled_filter(strategy_label: str, session_name: str) -> Optional[str]:
        """Return a filter label when execution is disabled for this strategy/session."""
        if not strategy_label:
            return None
        strat_lower = str(strategy_label).strip().lower()
        if strat_lower in exec_disabled:
            return f"StrategyDisabled:{strategy_label}"
        sess = str(session_name).upper() if session_name else ""
        for key, sessions in exec_disabled_by_session.items():
            if not key:
                continue
            if strat_lower == key or strat_lower.startswith(key) or key in strat_lower:
                if sess in sessions:
                    return f"StrategyDisabled:{strategy_label}@{sess}"
        return None

    enabled_filter_rejection = bool(filter_enabled("RejectionFilter"))
    enabled_filter_bank = bool(filter_enabled("BankLevelQuarterFilter"))
    enabled_filter_chop = bool(filter_enabled("ChopFilter"))
    enabled_filter_extension = bool(filter_enabled("ExtensionFilter"))
    enabled_filter_trend = bool(filter_enabled("TrendFilter"))
    enabled_filter_htf_fvg = bool(filter_enabled("HTF_FVG"))
    enabled_filter_structure = bool(filter_enabled("StructureBlocker"))
    enabled_filter_regime_blocker = bool(filter_enabled("RegimeBlocker"))
    enabled_filter_penalty = bool(filter_enabled("PenaltyBoxBlocker"))
    enabled_filter_memory_sr = bool(filter_enabled("MemorySRFilter"))
    enabled_filter_directional_loss = bool(filter_enabled("DirectionalLossBlocker"))
    enabled_filter_impulse = bool(filter_enabled("ImpulseFilter"))
    enabled_filter_news = bool(filter_enabled("NewsFilter"))
    enabled_filter_legacy_trend = bool(filter_enabled("LegacyTrend"))
    enabled_filter_target_feasibility = bool(filter_enabled("TargetFeasibility"))
    enabled_filter_vol_guard = bool(filter_enabled("VolatilityGuardrail"))
    enabled_filter_fixed_sltp = bool(filter_enabled("FixedSLTP"))
    enabled_filter_regime_manifold = bool(filter_enabled("RegimeManifold"))
    enabled_filter_pre_candidate = bool(filter_enabled("PreCandidateGate"))
    enabled_filter_ml_vol_regime = bool(filter_enabled("MLVolRegimeGuard"))
    enabled_filter_filter_arbitrator = bool(filter_enabled("FilterArbitrator"))
    enabled_filter_trend_day_tier = bool(filter_enabled("TrendDayTier"))
    if manifold_only_fast_mode or aetherflow_only_fast_mode:
        enabled_filter_trend_day_tier = False
    # Backtest-only: filter-check event logs are extremely verbose and can
    # dominate runtime on long simulations when left on.
    event_logger.set_filter_check_logging(
        bool(CONFIG.get("BACKTEST_EVENT_LOG_FILTER_CHECKS", False))
    )
    rejection_filter = RejectionFilter()
    bank_filter = BankLevelQuarterFilter()
    chop_filter = ChopFilter(lookback=20)
    extension_filter = ExtensionFilter()
    trend_filter = TrendFilter()
    htf_fvg_filter = (
        BacktestHTFFVGFilter()
        if (htf_fvg_enabled_backtest and enabled_filter_htf_fvg)
        else None
    )
    structure_blocker = DynamicStructureBlocker(
        lookback=50,
        rejection_memory_bars=1,
    )
    regime_blocker = RegimeStructureBlocker(lookback=20)
    penalty_cfg = CONFIG.get("PENALTY_BOX", {}) or {}
    penalty_blocker = None
    if penalty_cfg.get("enabled", True):
        penalty_blocker = PenaltyBoxBlocker(
            lookback=int(penalty_cfg.get("lookback", 50) or 50),
            tolerance=float(penalty_cfg.get("tolerance", 5.0) or 5.0),
            penalty_bars=int(penalty_cfg.get("penalty_bars", 3) or 3),
        )
    penalty_blocker_asia = None
    if asia_calib_enabled:
        penalty_cfg = asia_calib_cfg.get("penalty_box", {}) or {}
        if penalty_cfg.get("enabled", penalty_blocker is not None):
            penalty_blocker_asia = PenaltyBoxBlocker(
                lookback=int(penalty_cfg.get("lookback", 50) or 50),
                tolerance=float(penalty_cfg.get("tolerance", 5.0) or 5.0),
                penalty_bars=int(penalty_cfg.get("penalty_bars", 3) or 3),
            )
    memory_sr = MemorySRFilter(lookback_bars=300, zone_width=2.0, touch_threshold=2)
    directional_loss_blocker = DirectionalLossBlocker(consecutive_loss_limit=3, block_minutes=15)
    impulse_filter = ImpulseFilter(lookback=20, impulse_multiplier=2.5)
    legacy_load_news = bool(CONFIG.get("BACKTEST_LEGACY_LOAD_NEWS", False))
    legacy_filters = LegacyFilterSystem(load_news=legacy_load_news)
    filter_arbitrator = FilterArbitrator(confidence_threshold=0.6)

    # Keep trend-filter inputs bounded so per-bar runtime stays flat as history grows.
    # These filters only need recent structure around slow EMA windows.
    trend_filter_lookback_const = max(
        int(getattr(trend_filter, "lookback", 220) or 220),
        int(getattr(trend_filter, "slow_period", 200) or 200) + 20,
        256,
    )
    legacy_trend_filter_obj = getattr(legacy_filters, "trend_filter", None)
    legacy_trend_lookback_const = max(
        int(getattr(legacy_trend_filter_obj, "slow_period", 200) or 200) + 20,
        256,
    )

    news_filter = BacktestNewsFilter()

    continuation_manager = ContinuationRescueManager()
    emit_init_status("Preparing continuation allowlist...")
    continuation_allow_cfg = CONFIG.get("BACKTEST_CONTINUATION_ALLOWLIST", {})
    allowlist_mode = str(continuation_allow_cfg.get("mode", "reports") or "reports").lower()
    runtime_train = bool(continuation_allow_cfg.get("runtime_train", False))
    continuation_allowlist = None
    continuation_allow_stats: dict = {}
    if allowlist_mode != "csv_fast":
        continuation_allowlist, continuation_allow_stats = build_continuation_allowlist(
            continuation_allow_cfg, Path(__file__).resolve().parent
        )
    continuation_allowed_regimes = {
        str(item).lower()
        for item in (CONFIG.get("BACKTEST_CONTINUATION_ALLOWED_REGIMES") or [])
        if item is not None
    }
    continuation_confirm_cfg = CONFIG.get("BACKTEST_CONTINUATION_CONFIRM", {})
    continuation_no_bypass = bool(CONFIG.get("BACKTEST_CONTINUATION_NO_BYPASS", False))
    continuation_signal_mode = str(
        CONFIG.get("BACKTEST_CONTINUATION_SIGNAL_MODE", "calendar") or "calendar"
    ).lower()
    emit_init_status("Normalizing backtest data...")

    if mnq_df is None:
        mnq_df = pd.DataFrame()
    if vix_df is None:
        vix_df = pd.DataFrame()

    df = normalize_index(df, NY_TZ)
    mnq_df = normalize_index(mnq_df, NY_TZ)
    vix_df = normalize_index(vix_df, NY_TZ)
    bar_minutes = infer_bar_minutes(df.index)
    require_1m = bool(CONFIG.get("BACKTEST_REQUIRE_1M", True))
    if require_1m and bar_minutes not in (None, 1):
        raise ValueError(
            f"Backtest requires 1-minute bars. Detected ~{bar_minutes}m. "
            "Use a 1-minute CSV (e.g., es_master.csv) or set BACKTEST_REQUIRE_1M=False."
        )
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=NY_TZ)
    else:
        start_time = start_time.astimezone(NY_TZ)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=NY_TZ)
    else:
        end_time = end_time.astimezone(NY_TZ)

    warmup_bars = int(WARMUP_BARS)
    try:
        de3_fast_warmup_target = int(CONFIG.get("BACKTEST_DE3_FAST_HISTORY_BARS", 3000) or 3000)
    except Exception:
        de3_fast_warmup_target = 3000
    de3_fast_warmup_target = max(120, de3_fast_warmup_target)
    if ml_only_requested:
        try:
            ml_warmup_target = int(
                CONFIG.get(
                    "BACKTEST_ML_DIST_INPUT_BARS",
                    CONFIG.get("ML_PHYSICS_DIST_MAX_BARS", 3000),
                )
                or 3000
            )
        except Exception:
            ml_warmup_target = 3000
        ml_warmup_target = max(1, ml_warmup_target)
        warmup_bars = max(1, min(int(WARMUP_BARS), ml_warmup_target))
        logging.info(
            "ML-only backtest detected: warmup bars reduced from %s to %s for faster cached startup.",
            int(WARMUP_BARS),
            int(warmup_bars),
        )
    elif de3_only_fast_mode or de3_ml_fast_mode:
        warmup_bars = max(1, min(int(WARMUP_BARS), int(de3_fast_warmup_target)))
        logging.info(
            "DE3 fast backtest detected: warmup bars reduced from %s to %s.",
            int(WARMUP_BARS),
            int(warmup_bars),
        )

    warmup_df = df[df.index < start_time].tail(warmup_bars)
    test_df = df[(df.index >= start_time) & (df.index <= end_time)]
    if test_df.empty:
        raise ValueError("No bars in range to backtest.")
    if fast_enabled and bar_stride > 1:
        warmup_df = warmup_df.iloc[::bar_stride]
        test_df = test_df.iloc[::bar_stride]
    total_bars = len(test_df)

    full_df = pd.concat([warmup_df, test_df])
    try:
        df_attrs = getattr(df, "attrs", {}) or {}
        full_df = attach_backtest_symbol_context(
            full_df,
            df_attrs.get("selected_symbol"),
            df_attrs.get("selected_symbol_mode"),
            source_key=df_attrs.get("source_cache_key"),
            source_label=df_attrs.get("source_label"),
            source_path=df_attrs.get("source_path"),
        )
    except Exception:
        pass
    full_index = full_df.index
    if manifold_only_fast_mode and manifold_fast_strategy is not None:
        try:
            emit_init_status("ManifoldStrategy: precomputing backtest signals...")
            manifold_precompute_t0 = time.perf_counter()
            manifold_precomputed_df = manifold_fast_strategy.build_precomputed_backtest_df(full_df)
            if isinstance(manifold_precomputed_df, pd.DataFrame) and not manifold_precomputed_df.empty:
                manifold_precomputed_df = manifold_precomputed_df.loc[
                    (manifold_precomputed_df.index >= start_time)
                    & (manifold_precomputed_df.index <= end_time)
                ]
            manifold_fast_strategy.set_precomputed_backtest_df(manifold_precomputed_df)
            manifold_rows = (
                int(len(manifold_precomputed_df))
                if isinstance(manifold_precomputed_df, pd.DataFrame)
                else 0
            )
            logging.info(
                "ManifoldStrategy backtest precompute complete: rows=%s elapsed=%.2fs",
                manifold_rows,
                time.perf_counter() - manifold_precompute_t0,
            )
        except Exception as exc:
            logging.warning("ManifoldStrategy backtest precompute failed: %s", exc)
            try:
                manifold_fast_strategy.set_precomputed_backtest_df(None)
            except Exception:
                pass
    if aetherflow_strategy is not None and getattr(aetherflow_strategy, "model_loaded", False):
        try:
            emit_init_status("AetherFlowStrategy: precomputing backtest signals...")
            aetherflow_precompute_t0 = time.perf_counter()
            aetherflow_precomputed_df = aetherflow_strategy.build_precomputed_backtest_df(full_df)
            if isinstance(aetherflow_precomputed_df, pd.DataFrame) and not aetherflow_precomputed_df.empty:
                aetherflow_precomputed_df = aetherflow_precomputed_df.loc[
                    (aetherflow_precomputed_df.index >= start_time)
                    & (aetherflow_precomputed_df.index <= end_time)
                ]
            aetherflow_strategy.set_precomputed_backtest_df(aetherflow_precomputed_df)
            aetherflow_rows = (
                int(len(aetherflow_precomputed_df))
                if isinstance(aetherflow_precomputed_df, pd.DataFrame)
                else 0
            )
            logging.info(
                "AetherFlowStrategy backtest precompute complete: rows=%s elapsed=%.2fs",
                aetherflow_rows,
                time.perf_counter() - aetherflow_precompute_t0,
            )
        except Exception as exc:
            logging.warning("AetherFlowStrategy backtest precompute failed: %s", exc)
            try:
                aetherflow_strategy.set_precomputed_backtest_df(None)
            except Exception:
                pass
    ml_precomputed_df = None
    ml_opt_cfg = CONFIG.get("ML_PHYSICS_OPT", {}) or {}
    ml_opt_enabled = bool(ml_opt_cfg.get("enabled", False))
    ml_opt_mode = str(ml_opt_cfg.get("mode", "backtest") or "backtest").lower()
    # Keep MLPhysics optimized precompute/cache attach as an ML-only backtest path.
    # In mixed runs (for example MLPhysics + DE3v4), forcing the dist precompute
    # branch changes startup behavior and makes the UI act unlike the selected
    # strategy mix. Mixed runs should use normal per-bar ML evaluation instead.
    ml_opt_backtest_active = (
        ml_only_requested
        and isinstance(ml_strategy, MLPhysicsStrategy)
        and bool(getattr(ml_strategy, "model_loaded", False))
        and ml_opt_enabled
        and ml_opt_mode == "backtest"
    )
    if ml_opt_backtest_active:
        try:
            ml_opt_t0 = time.perf_counter()
            if bool(getattr(ml_strategy, "_dist_mode", False)):
                emit_init_status("MLPhysics(dist): preparing precompute/cache...")
                ok = bool(
                    ml_strategy.precompute_dist_backtest_signals(
                        full_df,
                        progress_every=int(CONFIG.get("BACKTEST_ML_DIST_PROGRESS_EVERY", 2000) or 2000),
                        progress_min_interval_sec=float(
                            CONFIG.get("BACKTEST_ML_DIST_PROGRESS_SEC", 4.0) or 4.0
                        ),
                        status_cb=lambda payload: emit_init_status(
                            str(payload.get("message", "") or "").strip() or "MLPhysics(dist): working...",
                            **{k: v for k, v in payload.items() if k not in {"type", "message"}},
                        ),
                    )
                )
                if ok:
                    logging.info(
                        "MLPhysics OPT(dist): precomputed/cached rows=%d in %.2fs",
                        len(full_df),
                        time.perf_counter() - ml_opt_t0,
                    )
                    emit_init_status("MLPhysics(dist): precompute/cache ready.")
                else:
                    logging.warning("MLPhysics OPT(dist): precompute unavailable; falling back to per-bar dist inference")
                    emit_init_status("MLPhysics(dist): precompute unavailable, using per-bar inference.")
            else:
                emit_init_status("MLPhysics: preparing vectorized feature/prediction cache...")
                ml_precomputed_df = ml_physics_pipeline.prepare_full_dataset(
                    full_df,
                    session_manager=getattr(ml_strategy, "sm", None),
                )
                if isinstance(ml_precomputed_df, pd.DataFrame) and not ml_precomputed_df.empty:
                    ml_strategy.set_precomputed_backtest_df(ml_precomputed_df)
                    logging.info(
                        "MLPhysics OPT: precomputed rows=%d in %.2fs",
                        len(ml_precomputed_df),
                        time.perf_counter() - ml_opt_t0,
                    )
                    emit_init_status("MLPhysics: feature/prediction cache ready.")
                else:
                    ml_precomputed_df = None
                    logging.warning("MLPhysics OPT: precompute returned empty table; falling back to legacy path")
                    emit_init_status("MLPhysics: precompute returned empty table, using legacy path.")
        except Exception as exc:
            ml_precomputed_df = None
            try:
                ml_strategy.set_precomputed_backtest_df(None)
            except Exception:
                pass
            try:
                ml_strategy.set_precomputed_dist_backtest_df(None)
            except Exception:
                pass
            logging.warning("MLPhysics OPT: precompute failed; falling back to legacy path (%s)", exc)
            emit_init_status("MLPhysics: precompute failed, using legacy path.")
    elif (
        isinstance(ml_strategy, MLPhysicsStrategy)
        and bool(getattr(ml_strategy, "model_loaded", False))
        and ml_opt_enabled
        and ml_opt_mode == "backtest"
        and not ml_only_requested
    ):
        if bool(getattr(ml_strategy, "_dist_mode", False)):
            try:
                ml_opt_t0 = time.perf_counter()
                emit_init_status("MLPhysics(dist): checking cache for mixed strategy run...")
                ok = bool(
                    ml_strategy.precompute_dist_backtest_signals(
                        full_df,
                        progress_every=int(CONFIG.get("BACKTEST_ML_DIST_PROGRESS_EVERY", 2000) or 2000),
                        progress_min_interval_sec=float(
                            CONFIG.get("BACKTEST_ML_DIST_PROGRESS_SEC", 4.0) or 4.0
                        ),
                        status_cb=lambda payload: emit_init_status(
                            str(payload.get("message", "") or "").strip()
                            or "MLPhysics(dist): checking cache...",
                            **{k: v for k, v in payload.items() if k not in {"type", "message"}},
                        ),
                        allow_build=False,
                    )
                )
                if ok:
                    logging.info(
                        "MLPhysics OPT(dist): attached cached rows=%d for mixed strategy backtest in %.2fs",
                        len(full_df),
                        time.perf_counter() - ml_opt_t0,
                    )
                    emit_init_status("MLPhysics(dist): cache ready for mixed strategy run.")
                else:
                    logging.info(
                        "MLPhysics OPT(dist): no compatible cache for mixed strategy backtest; using per-bar inference"
                    )
            except Exception as exc:
                try:
                    ml_strategy.set_precomputed_dist_backtest_df(None)
                except Exception:
                    pass
                logging.warning(
                    "MLPhysics OPT(dist): mixed-run cache attach failed; using per-bar inference (%s)",
                    exc,
                )
        else:
            logging.info(
                "MLPhysics OPT: skipping precompute/cache attach for mixed strategy backtest selection"
            )
    ml_strategy_history_free_eval = False
    if isinstance(ml_strategy, MLPhysicsStrategy):
        try:
            ml_strategy_history_free_eval = bool(
                ml_strategy._opt_dist_backtest_active() or ml_strategy._opt_backtest_active()
            )
        except Exception:
            ml_strategy_history_free_eval = False
    if ml_only_requested:
        try:
            ml_only_history_cap_bars = int(
                CONFIG.get(
                    "BACKTEST_ML_ONLY_HISTORY_MAX_BARS",
                    CONFIG.get(
                        "BACKTEST_ML_DIST_INPUT_BARS",
                        CONFIG.get("ML_PHYSICS_DIST_MAX_BARS", 3000),
                    ),
                )
                or 0
            )
        except Exception:
            ml_only_history_cap_bars = 0
        if ml_only_history_cap_bars > 0:
            if backtest_max_history_bars <= 0 or backtest_max_history_bars > ml_only_history_cap_bars:
                logging.info(
                    "ML-only history cap active: %s bars (previous backtest cap=%s).",
                    int(ml_only_history_cap_bars),
                    int(backtest_max_history_bars),
                )
                backtest_max_history_bars = int(ml_only_history_cap_bars)
    full_index_ns = full_index.values.astype("datetime64[ns]")
    bar_sl_multiplier_arr = np.full(
        len(full_df),
        float(multiplier_dataset_info.get("default_sl_multiplier", 1.0)),
        dtype=float,
    )
    bar_tp_multiplier_arr = np.full(
        len(full_df),
        float(multiplier_dataset_info.get("default_tp_multiplier", 1.0)),
        dtype=float,
    )
    bar_chop_multiplier_arr = np.full(
        len(full_df),
        float(multiplier_dataset_info.get("default_chop_multiplier", 1.0)),
        dtype=float,
    )
    if multiplier_schedule is not None:
        sched_index_ns = multiplier_schedule.get("index_ns")
        sched_sl = multiplier_schedule.get("sl")
        sched_tp = multiplier_schedule.get("tp")
        sched_chop = multiplier_schedule.get("chop")
        if (
            isinstance(sched_index_ns, np.ndarray)
            and isinstance(sched_sl, np.ndarray)
            and isinstance(sched_tp, np.ndarray)
            and isinstance(sched_chop, np.ndarray)
            and len(sched_index_ns) > 0
        ):
            idx_map = np.searchsorted(sched_index_ns, full_index_ns, side="right") - 1
            valid = idx_map >= 0
            if np.any(valid):
                bar_sl_multiplier_arr[valid] = sched_sl[idx_map[valid]]
                bar_tp_multiplier_arr[valid] = sched_tp[idx_map[valid]]
                bar_chop_multiplier_arr[valid] = sched_chop[idx_map[valid]]
                multiplier_dataset_info["bars_with_multiplier"] = int(np.sum(valid))
    holiday_closed_dates_et = _build_closed_holiday_dates_et(full_index)
    if holiday_closed_dates_et:
        full_date_arr = np.array(full_index.date, dtype=object)
        holiday_dates_arr = np.array(sorted(holiday_closed_dates_et), dtype=object)
        holiday_closed_arr = np.isin(full_date_arr, holiday_dates_arr)
    else:
        holiday_closed_arr = np.zeros(len(full_df), dtype=bool)
    if BACKTEST_ENFORCE_US_HOLIDAY_CLOSURE:
        logging.info(
            "Backtest holiday closure active: %d closed dates in range (%d extra) | sessions=%s",
            len(holiday_closed_dates_et),
            len(BACKTEST_EXTRA_CLOSED_DATES_ET),
            (
                "ALL"
                if not BACKTEST_HOLIDAY_CLOSURE_SESSIONS_ET
                else ",".join(BACKTEST_HOLIDAY_CLOSURE_SESSIONS_ET)
            ),
        )
    full_session_arr = np.array([get_session_name(ts) for ts in full_index], dtype=object)
    full_trend_session_arr = np.where(
        np.isin(full_session_arr, np.array(["NY_AM", "NY_PM"], dtype=object)),
        "NY",
        full_session_arr,
    )
    open_arr = full_df["open"].to_numpy()
    high_arr = full_df["high"].to_numpy()
    low_arr = full_df["low"].to_numpy()
    close_arr = full_df["close"].to_numpy()
    # Precompute ATR20 exactly with the same formula as _calc_atr20 to avoid repeated O(n) scans.
    _high_series = full_df["high"]
    _low_series = full_df["low"]
    _close_series = full_df["close"]
    _prev_close_series = _close_series.shift(1)
    _tr_series = pd.concat(
        [
            (_high_series - _low_series).abs(),
            (_high_series - _prev_close_series).abs(),
            (_low_series - _prev_close_series).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr20_arr = _tr_series.rolling(20).mean().to_numpy(dtype=float)
    swing_high_30_arr = full_df["high"].rolling(30, min_periods=30).max().to_numpy(dtype=float)
    swing_low_30_arr = full_df["low"].rolling(30, min_periods=30).min().to_numpy(dtype=float)

    # Precompute session-aware context arrays used in NY gates / DE3 meta.
    if "volume" in full_df.columns:
        volume_arr = pd.to_numeric(full_df["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        volume_arr = np.ones(len(full_df), dtype=float)
    typical_price_arr = (high_arr + low_arr + close_arr) / 3.0
    ny_hour_arr = full_index.hour.to_numpy(dtype=int)
    ny_minute_arr = full_index.minute.to_numpy(dtype=int)
    ny_date_arr = np.array(full_index.date, dtype=object)
    ny_minute_of_day_arr = (ny_hour_arr * 60) + ny_minute_arr
    if BACKTEST_FORCE_FLAT_AT_TIME:
        force_flat_arr = (
            (ny_hour_arr == BACKTEST_FORCE_FLAT_HOUR_ET)
            & (ny_minute_arr == BACKTEST_FORCE_FLAT_MINUTE_ET)
        )
    else:
        force_flat_arr = np.zeros(len(full_df), dtype=bool)
    if BACKTEST_ENFORCE_US_HOLIDAY_CLOSURE and holiday_closed_dates_et:
        if (
            (not BACKTEST_HOLIDAY_CLOSURE_SESSION_SET_ET)
            or ("ALL" in BACKTEST_HOLIDAY_CLOSURE_SESSION_SET_ET)
        ):
            holiday_session_block_arr = np.ones(len(full_df), dtype=bool)
        else:
            holiday_session_values = np.array(
                sorted(BACKTEST_HOLIDAY_CLOSURE_SESSION_SET_ET),
                dtype=object,
            )
            holiday_session_block_arr = np.isin(
                np.char.upper(full_session_arr.astype(str)),
                holiday_session_values,
            )
        holiday_flat_arr = holiday_closed_arr & holiday_session_block_arr
    else:
        holiday_flat_arr = np.zeros(len(full_df), dtype=bool)
    shadow_forced_close_arr = force_flat_arr | holiday_flat_arr
    next_shadow_forced_close_idx = np.full(len(full_df), -1, dtype=int)
    _next_forced_idx = -1
    for _j in range(len(full_df) - 1, -1, -1):
        if shadow_forced_close_arr[_j]:
            _next_forced_idx = _j
        next_shadow_forced_close_idx[_j] = _next_forced_idx

    de3_session_key_arr = np.empty(len(full_df), dtype=object)
    de3_session_open_arr = np.full(len(full_df), np.nan, dtype=float)
    de3_session_high_arr = np.full(len(full_df), np.nan, dtype=float)
    de3_session_low_arr = np.full(len(full_df), np.nan, dtype=float)
    de3_prior_session_high_arr = np.full(len(full_df), np.nan, dtype=float)
    de3_prior_session_low_arr = np.full(len(full_df), np.nan, dtype=float)
    de3_prior_session_close_arr = np.full(len(full_df), np.nan, dtype=float)
    de3_session_vwap_arr = np.full(len(full_df), np.nan, dtype=float)
    ny_orh_arr = np.full(len(full_df), np.nan, dtype=float)
    ny_orl_arr = np.full(len(full_df), np.nan, dtype=float)
    ny_am_high_arr = np.full(len(full_df), np.nan, dtype=float)
    ny_am_low_arr = np.full(len(full_df), np.nan, dtype=float)
    ny_pm_high_arr = np.full(len(full_df), np.nan, dtype=float)
    ny_pm_low_arr = np.full(len(full_df), np.nan, dtype=float)

    _day_key = None
    _orh = np.nan
    _orl = np.nan
    _am_high = np.nan
    _am_low = np.nan
    _pm_high = np.nan
    _pm_low = np.nan
    _de3_key = None
    _de3_open = np.nan
    _de3_high = np.nan
    _de3_low = np.nan
    _de3_prev_high = np.nan
    _de3_prev_low = np.nan
    _de3_prev_close = np.nan
    _de3_num = 0.0
    _de3_den = 0.0
    for j in range(len(full_df)):
        day = ny_date_arr[j]
        minute_of_day = int(ny_minute_of_day_arr[j])
        hour = int(ny_hour_arr[j])
        high_val = float(high_arr[j])
        low_val = float(low_arr[j])

        if _day_key != day:
            _day_key = day
            _orh = np.nan
            _orl = np.nan
            _am_high = np.nan
            _am_low = np.nan
            _pm_high = np.nan
            _pm_low = np.nan

        if 570 <= minute_of_day < 585:
            _orh = high_val if not math.isfinite(_orh) else max(_orh, high_val)
            _orl = low_val if not math.isfinite(_orl) else min(_orl, low_val)
        ny_orh_arr[j] = _orh if math.isfinite(_orh) else np.nan
        ny_orl_arr[j] = _orl if math.isfinite(_orl) else np.nan

        if 480 <= minute_of_day < 720:
            _am_high = high_val if not math.isfinite(_am_high) else max(_am_high, high_val)
            _am_low = low_val if not math.isfinite(_am_low) else min(_am_low, low_val)
            ny_am_high_arr[j] = _am_high
            ny_am_low_arr[j] = _am_low
        if 720 <= minute_of_day < 1020:
            _pm_high = high_val if not math.isfinite(_pm_high) else max(_pm_high, high_val)
            _pm_low = low_val if not math.isfinite(_pm_low) else min(_pm_low, low_val)
            ny_pm_high_arr[j] = _pm_high
            ny_pm_low_arr[j] = _pm_low

        de3_key = day if hour >= 18 else (day - dt.timedelta(days=1))
        if _de3_key != de3_key:
            if _de3_key is not None:
                _de3_prev_high = _de3_high
                _de3_prev_low = _de3_low
                _de3_prev_close = float(close_arr[j - 1]) if j > 0 else np.nan
            _de3_key = de3_key
            _de3_open = float(open_arr[j])
            _de3_high = high_val
            _de3_low = low_val
            _de3_num = 0.0
            _de3_den = 0.0
        else:
            _de3_high = max(_de3_high, high_val) if math.isfinite(_de3_high) else high_val
            _de3_low = min(_de3_low, low_val) if math.isfinite(_de3_low) else low_val
        de3_session_key_arr[j] = de3_key
        de3_session_open_arr[j] = _de3_open if math.isfinite(_de3_open) else np.nan
        de3_session_high_arr[j] = _de3_high if math.isfinite(_de3_high) else np.nan
        de3_session_low_arr[j] = _de3_low if math.isfinite(_de3_low) else np.nan
        de3_prior_session_high_arr[j] = _de3_prev_high if math.isfinite(_de3_prev_high) else np.nan
        de3_prior_session_low_arr[j] = _de3_prev_low if math.isfinite(_de3_prev_low) else np.nan
        de3_prior_session_close_arr[j] = _de3_prev_close if math.isfinite(_de3_prev_close) else np.nan
        w = volume_arr[j]
        if not math.isfinite(w):
            w = 0.0
        _de3_num += float(typical_price_arr[j]) * float(w)
        _de3_den += float(w)
        if _de3_den > 0.0:
            de3_session_vwap_arr[j] = _de3_num / _de3_den
    de3_close_series = pd.Series(close_arr, index=full_df.index, dtype=float)
    de3_ret1_series = de3_close_series.diff().fillna(0.0)
    de3_up_pressure_arr = (
        de3_ret1_series.clip(lower=0.0).rolling(12, min_periods=1).sum().to_numpy(dtype=float)
    )
    de3_down_pressure_arr = (
        (-de3_ret1_series.clip(upper=0.0)).rolling(12, min_periods=1).sum().to_numpy(dtype=float)
    )
    de3_net_return_12_arr = (
        (de3_close_series - de3_close_series.shift(12)).fillna(0.0).to_numpy(dtype=float)
    )
    de3_session_vwap_slope_arr = np.full(len(full_df), np.nan, dtype=float)
    for j in range(len(full_df)):
        lookback = j - 8
        if lookback < 0:
            continue
        if de3_session_key_arr[j] != de3_session_key_arr[lookback]:
            continue
        vwap_now = de3_session_vwap_arr[j]
        vwap_then = de3_session_vwap_arr[lookback]
        if math.isfinite(vwap_now) and math.isfinite(vwap_then):
            de3_session_vwap_slope_arr[j] = float(vwap_now - vwap_then)
    vol_base = warmup_df if not warmup_df.empty else test_df
    if not vol_base.empty:
        try:
            volatility_filter.calibrate(vol_base)
        except Exception:
            pass

    trend_day_context_required = bool(TREND_DAY_ENABLED and enabled_filter_trend_day_tier)
    if trend_day_context_required:
        emit_init_status("Computing trend-day context...")
        trend_day_source_df = full_df
        if TREND_DAY_TIMEFRAME_MINUTES > 1:
            trend_day_source_df = resample_dataframe(full_df, TREND_DAY_TIMEFRAME_MINUTES)
        trend_day_series_raw = compute_trend_day_series(trend_day_source_df)
        trend_day_series = align_trend_day_series(trend_day_series_raw, full_df.index)
        td_ema50 = trend_day_series["ema50"]
        td_ema200 = trend_day_series["ema200"]
        td_atr_exp = trend_day_series["atr_expansion"]
        td_vwap = trend_day_series["vwap"]
        td_vwap_sigma = trend_day_series["vwap_sigma_dist"]
        td_reclaim_down = trend_day_series["reclaim_down"]
        td_reclaim_up = trend_day_series["reclaim_up"]
        td_no_reclaim_down_t1 = trend_day_series["no_reclaim_down_t1"]
        td_no_reclaim_up_t1 = trend_day_series["no_reclaim_up_t1"]
        td_no_reclaim_down_t2 = trend_day_series["no_reclaim_down_t2"]
        td_no_reclaim_up_t2 = trend_day_series["no_reclaim_up_t2"]
        td_session_open = trend_day_series["session_open"]
        td_prior_session_low = trend_day_series["prior_session_low"]
        td_prior_session_high = trend_day_series["prior_session_high"]
        td_trend_up_alt = trend_day_series["trend_up_alt"]
        td_trend_down_alt = trend_day_series["trend_down_alt"]
        td_adx_strong_up = trend_day_series["adx_strong_up"]
        td_adx_strong_down = trend_day_series["adx_strong_down"]
        td_day_index = trend_day_series["day_index"]
        td_ema50_arr = td_ema50.to_numpy()
        td_ema200_arr = td_ema200.to_numpy()
        td_atr_exp_arr = td_atr_exp.to_numpy()
        td_vwap_arr = td_vwap.to_numpy()
        td_vwap_sigma_arr = td_vwap_sigma.to_numpy()
        td_reclaim_down_arr = td_reclaim_down.to_numpy()
        td_reclaim_up_arr = td_reclaim_up.to_numpy()
        td_no_reclaim_down_t1_arr = td_no_reclaim_down_t1.to_numpy()
        td_no_reclaim_up_t1_arr = td_no_reclaim_up_t1.to_numpy()
        td_no_reclaim_down_t2_arr = td_no_reclaim_down_t2.to_numpy()
        td_no_reclaim_up_t2_arr = td_no_reclaim_up_t2.to_numpy()
        td_session_open_arr = td_session_open.to_numpy()
        td_prior_session_low_arr = td_prior_session_low.to_numpy()
        td_prior_session_high_arr = td_prior_session_high.to_numpy()
        td_trend_up_alt_arr = td_trend_up_alt.to_numpy()
        td_trend_down_alt_arr = td_trend_down_alt.to_numpy()
        td_adx_strong_up_arr = td_adx_strong_up.to_numpy()
        td_adx_strong_down_arr = td_adx_strong_down.to_numpy()
        td_day_index_arr = td_day_index.to_numpy()
    else:
        emit_init_status("Skipping trend-day context (TrendDayTier filter disabled)...")
        trend_day_series = None
        td_ema50_arr = np.full(len(full_df), np.nan, dtype=float)
        td_ema200_arr = np.full(len(full_df), np.nan, dtype=float)
        td_atr_exp_arr = np.full(len(full_df), np.nan, dtype=float)
        td_vwap_arr = np.full(len(full_df), np.nan, dtype=float)
        td_vwap_sigma_arr = np.full(len(full_df), np.nan, dtype=float)
        td_reclaim_down_arr = np.zeros(len(full_df), dtype=bool)
        td_reclaim_up_arr = np.zeros(len(full_df), dtype=bool)
        td_no_reclaim_down_t1_arr = np.zeros(len(full_df), dtype=bool)
        td_no_reclaim_up_t1_arr = np.zeros(len(full_df), dtype=bool)
        td_no_reclaim_down_t2_arr = np.zeros(len(full_df), dtype=bool)
        td_no_reclaim_up_t2_arr = np.zeros(len(full_df), dtype=bool)
        td_session_open_arr = np.full(len(full_df), np.nan, dtype=float)
        td_prior_session_low_arr = np.full(len(full_df), np.nan, dtype=float)
        td_prior_session_high_arr = np.full(len(full_df), np.nan, dtype=float)
        td_trend_up_alt_arr = np.zeros(len(full_df), dtype=bool)
        td_trend_down_alt_arr = np.zeros(len(full_df), dtype=bool)
        td_adx_strong_up_arr = np.zeros(len(full_df), dtype=bool)
        td_adx_strong_down_arr = np.zeros(len(full_df), dtype=bool)
        td_day_index_arr = np.array(full_index.date, dtype=object)

    if allowlist_mode == "csv_fast":
        if runtime_train:
            continuation_allowlist, continuation_allow_stats = build_continuation_allowlist_from_df(
                full_df,
                trend_day_series,
                continuation_allow_cfg,
                continuation_allowed_regimes,
                continuation_confirm_cfg,
            )
        else:
            continuation_allowlist = load_continuation_allowlist_file(
                continuation_allow_cfg.get("cache_file")
            )
            continuation_allow_stats = {
                "summary": {
                    "cache_only": True,
                    "keys_loaded": len(continuation_allowlist or []),
                }
            }

    if continuation_allow_cfg.get("enabled", True):
        summary = (continuation_allow_stats or {}).get("summary", {})
        logging.info(
            "Continuation allowlist (%s): %s keys (reports used: %s)",
            allowlist_mode,
            len(continuation_allowlist or []),
            summary.get("reports_used", 0),
        )

    need_htf_fvg_cache = bool(enabled_filter_htf_fvg and htf_fvg_enabled_backtest and htf_fvg_filter is not None)
    need_chop_analyzer = bool(
        not de3_only_fast_mode
        and not de3_ml_fast_mode
        and not ml_only_fast_mode
        and (
            enabled_filter_chop
            or enabled_filter_target_feasibility
            or enabled_filter_pre_candidate
            or bool(loose_strategies)
            or bool(standard_non_ml_defs)
            or bool(standard_ml_defs)
            or bool(asia_calib_enabled)
        )
    )
    need_resample_cache_60 = bool(need_htf_fvg_cache or need_chop_analyzer)
    resample_cache_60 = ResampleCache(full_df, 60) if need_resample_cache_60 else None
    resample_cache_240 = ResampleCache(full_df, 240) if need_htf_fvg_cache else None

    if need_chop_analyzer:
        emit_init_status("Calibrating dynamic chop analyzer...")
        chop_client = BacktestClient(full_df)
        chop_analyzer = DynamicChopAnalyzer(chop_client)
        chop_analyzer.calibrate()
        try:
            chop_mult = float(CONFIG.get("BACKTEST_DYNAMIC_CHOP_MULTIPLIER", 1.0))
            chop_analyzer.update_gemini_params(chop_mult)
        except Exception:
            chop_mult = 1.0
    else:
        chop_analyzer = None
        chop_mult = float(CONFIG.get("BACKTEST_DYNAMIC_CHOP_MULTIPLIER", 1.0) or 1.0)
    active_sl_multiplier = float(CONFIG.get("DYNAMIC_SL_MULTIPLIER", 1.0))
    active_tp_multiplier = float(CONFIG.get("DYNAMIC_TP_MULTIPLIER", 1.0))
    active_chop_multiplier = float(chop_mult)
    multiplier_updates = 0
    structure_lookback_const = max(_coerce_int(getattr(structure_blocker, "lookback", 50), 50), 50)
    regime_lookback_const = _coerce_int(getattr(regime_blocker, "lookback", 20), 20)
    regime_update_lookback_const = (regime_lookback_const * 2) + 5
    penalty_lookback_const = (
        _coerce_int(getattr(penalty_blocker, "lookback", 50), 50)
        if penalty_blocker is not None
        else None
    )
    penalty_asia_lookback_const = (
        _coerce_int(getattr(penalty_blocker_asia, "lookback", 50), 50)
        if penalty_blocker_asia is not None
        else None
    )
    memory_lookback_const = max(_coerce_int(getattr(memory_sr, "lookback", 300), 300), 50)
    impulse_lookback_const = _coerce_int(getattr(impulse_filter, "lookback", 20), 20)
    impulse_atr_window_const = _coerce_int(getattr(impulse_filter, "atr_window", 14), 14)
    impulse_update_lookback_const = max(impulse_lookback_const, impulse_atr_window_const + 1)
    chop_lookback_const = _coerce_int(getattr(chop_analyzer, "LOOKBACK", 20), 20) if chop_analyzer is not None else 20

    asia_trend_cfg = (asia_calib_cfg.get("trend_bias", {}) or {}) if asia_calib_enabled else {}
    asia_tf_cfg = (asia_calib_cfg.get("target_feasibility", {}) or {}) if asia_calib_enabled else {}
    asia_chop_cfg = (asia_calib_cfg.get("chop_filter", {}) or {}) if asia_calib_enabled else {}

    asia_trend_ema_fast = max(_coerce_int(asia_trend_cfg.get("ema_fast", 20), 20), 1)
    asia_trend_ema_slow = max(_coerce_int(asia_trend_cfg.get("ema_slow", 50), 50), 1)
    asia_trend_ema_slope_bars = max(
        _coerce_int(asia_trend_cfg.get("ema_slope_bars", 20), 20),
        1,
    )
    asia_trend_required_len = max(asia_trend_ema_slow, asia_trend_ema_slope_bars + 1) + 1
    asia_trend_min_sep = _coerce_float(asia_trend_cfg.get("min_ema_separation", 0.1), 0.1)
    asia_trend_alpha_fast = 2.0 / (asia_trend_ema_fast + 1.0)
    asia_trend_alpha_slow = 2.0 / (asia_trend_ema_slow + 1.0)

    asia_tf_enabled = bool(asia_tf_cfg) and bool(asia_tf_cfg.get("enabled", True))
    asia_tf_lookback = max(
        _coerce_int(asia_tf_cfg.get("lookback", chop_lookback_const), chop_lookback_const),
        1,
    )
    asia_tf_min_box = _coerce_float(asia_tf_cfg.get("min_box_range", 0.0), 0.0)
    asia_tf_max_mult = _coerce_float(asia_tf_cfg.get("max_tp_box_mult", 1.5), 1.5)
    asia_tf_allow_trend_override = bool(asia_tf_cfg.get("allow_trend_override", False))

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    trades = 0
    wins = 0
    losses = 0
    tracker = AttributionTracker()
    flip_cfg = CONFIG.get("BACKTEST_FLIP_CONFIDENCE", {}) or {}
    if speed_disable_flip_confidence and isinstance(flip_cfg, dict):
        flip_cfg = dict(flip_cfg)
        flip_cfg["enabled"] = False
    flip_tracker = FlipConfidenceTracker(full_df, flip_cfg)
    flip_context = None
    ny_gate_candidates = 0
    ny_gate_balance_blocked = 0
    ny_gate_acceptance_blocked = 0
    ny_gate_liquidity_blocked = 0
    ny_gate_vp_cache: dict[int, Optional[dict]] = {}
    ny_gate_blocked_by_strategy = defaultdict(int)
    ny_gate_blocked_by_session = defaultdict(int)
    de3_meta_cfg = CONFIG.get("DE3_META_POLICY", {}) or {}
    de3_meta_enabled = bool(de3_meta_cfg.get("enabled", False))
    if de3_meta_enabled and (de3_runtime_is_v2 or de3_runtime_is_v3 or de3_runtime_is_v4):
        logging.info("Backtest DE3 meta policy disabled for DE3 v2/v3/v4 runtime (v1-only filter).")
        de3_meta_enabled = False
    de3_meta_mode = str(de3_meta_cfg.get("mode", "shadow") or "shadow").lower()
    if de3_meta_mode not in {"shadow", "block"}:
        de3_meta_mode = "shadow"
    de3_meta_log_decisions = bool(de3_meta_cfg.get("log_decisions", True))
    de3_meta_blocked_types_cfg = de3_meta_cfg.get("blocked_types", []) or []
    de3_meta_blocked_types = {
        str(item).strip().lower() for item in de3_meta_blocked_types_cfg if item is not None
    }
    try:
        de3_meta_er_lookback = int(de3_meta_cfg.get("er_lookback", 10) or 10)
    except Exception:
        de3_meta_er_lookback = 10
    try:
        de3_meta_min_score = float(de3_meta_cfg.get("min_score", 60.0))
    except Exception:
        de3_meta_min_score = 60.0
    try:
        de3_meta_shock_gap_mult = float(de3_meta_cfg.get("shock_gap_atr_mult", 0.9))
    except Exception:
        de3_meta_shock_gap_mult = 0.9
    try:
        de3_meta_shock_range_mult = float(de3_meta_cfg.get("shock_range_atr_mult", 2.4))
    except Exception:
        de3_meta_shock_range_mult = 2.4
    try:
        de3_meta_mom_min_er = float(de3_meta_cfg.get("mom_min_er", 0.20))
    except Exception:
        de3_meta_mom_min_er = 0.20
    try:
        de3_meta_rev_max_er = float(de3_meta_cfg.get("rev_max_er", 0.55))
    except Exception:
        de3_meta_rev_max_er = 0.55
    try:
        de3_meta_hv_max_vwap = float(de3_meta_cfg.get("high_vol_long_mom_max_vwap_atr", 1.4))
    except Exception:
        de3_meta_hv_max_vwap = 1.4
    try:
        de3_meta_long_mom_max_close_pos = float(de3_meta_cfg.get("long_mom_max_close_pos", 0.92))
    except Exception:
        de3_meta_long_mom_max_close_pos = 0.92
    try:
        de3_meta_short_mom_min_close_pos = float(de3_meta_cfg.get("short_mom_min_close_pos", 0.08))
    except Exception:
        de3_meta_short_mom_min_close_pos = 0.08
    de3_meta_checked = 0
    de3_meta_would_block = 0
    de3_meta_blocked = 0
    de3_meta_shadow = 0
    de3_meta_reasons = Counter()
    de3_runtime_cfg = (((CONFIG.get("DE3_V4") or {}).get("runtime")) or {})
    if not isinstance(de3_runtime_cfg, dict):
        de3_runtime_cfg = {}
    de3_manifold_adapt_cfg = de3_runtime_cfg.get("manifold_adaptation", {}) or {}
    if not isinstance(de3_manifold_adapt_cfg, dict):
        de3_manifold_adapt_cfg = {}
    de3_manifold_adapt_enabled = bool(de3_manifold_adapt_cfg.get("enabled", False))
    de3_manifold_adapt_mode = str(
        de3_manifold_adapt_cfg.get("mode", "shadow") or "shadow"
    ).lower()
    if de3_manifold_adapt_mode not in {"shadow", "block"}:
        de3_manifold_adapt_mode = "shadow"
    de3_manifold_adapt_log_decisions = bool(
        de3_manifold_adapt_cfg.get("log_decisions", False)
    )
    de3_manifold_blocked_regimes = {
        str(item).strip().upper()
        for item in (de3_manifold_adapt_cfg.get("blocked_regimes") or [])
        if str(item).strip()
    }
    de3_manifold_block_no_trade = bool(
        de3_manifold_adapt_cfg.get("block_no_trade", False)
    )
    de3_manifold_require_allow_style = bool(
        de3_manifold_adapt_cfg.get("require_allow_style", False)
    )
    de3_manifold_adapt_checked = 0
    de3_manifold_adapt_would_block = 0
    de3_manifold_adapt_blocked = 0
    de3_manifold_adapt_shadow = 0
    de3_manifold_adapt_reasons = Counter()
    de3_variant_adapt_cfg = de3_runtime_cfg.get("backtest_variant_adaptation", {}) or {}
    if not isinstance(de3_variant_adapt_cfg, dict):
        de3_variant_adapt_cfg = {}
    de3_variant_adapt_enabled = bool(de3_variant_adapt_cfg.get("enabled", False))
    de3_variant_adapt_history_window = max(
        1,
        _coerce_int(de3_variant_adapt_cfg.get("history_window_trades", 16), 16),
    )
    de3_variant_adapt_warmup_trades = max(
        0,
        min(
            de3_variant_adapt_history_window,
            _coerce_int(de3_variant_adapt_cfg.get("warmup_trades", 6), 6),
        ),
    )
    de3_variant_adapt_cold_avg = _coerce_float(
        de3_variant_adapt_cfg.get("cold_avg_net_per_contract_usd", -10.0),
        -10.0,
    )
    de3_variant_adapt_cold_max_winrate = min(
        1.0,
        max(
            0.0,
            _coerce_float(de3_variant_adapt_cfg.get("cold_max_winrate", 0.40), 0.40),
        ),
    )
    de3_variant_adapt_cold_mult = max(
        0.0,
        _coerce_float(de3_variant_adapt_cfg.get("cold_size_multiplier", 0.50), 0.50),
    )
    raw_de3_variant_adapt_max_lifetime_avg = de3_variant_adapt_cfg.get(
        "max_lifetime_avg_net_per_contract_usd"
    )
    if raw_de3_variant_adapt_max_lifetime_avg in (None, ""):
        de3_variant_adapt_max_lifetime_avg = float("nan")
    else:
        de3_variant_adapt_max_lifetime_avg = _coerce_float(
            raw_de3_variant_adapt_max_lifetime_avg,
            float("nan"),
        )
    raw_deep_cold_avg = de3_variant_adapt_cfg.get("deep_cold_avg_net_per_contract_usd")
    if raw_deep_cold_avg in (None, ""):
        de3_variant_adapt_deep_cold_avg = float("nan")
    else:
        de3_variant_adapt_deep_cold_avg = _coerce_float(raw_deep_cold_avg, float("nan"))
    de3_variant_adapt_deep_cold_mult = max(
        0.0,
        _coerce_float(de3_variant_adapt_cfg.get("deep_cold_size_multiplier", 0.34), 0.34),
    )
    de3_variant_adapt_min_contracts = max(
        1,
        _coerce_int(de3_variant_adapt_cfg.get("min_contracts", 1), 1),
    )
    de3_variant_adapt_reduce_only = bool(de3_variant_adapt_cfg.get("reduce_only", True))
    de3_variant_adapt_recent_pnl_history: defaultdict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=de3_variant_adapt_history_window)
    )
    de3_variant_adapt_lifetime_pnl_sum = defaultdict(float)
    de3_variant_adapt_lifetime_trade_count = defaultdict(int)
    de3_variant_adapt_checked = 0
    de3_variant_adapt_applied = 0
    de3_variant_adapt_state_counts = Counter()
    de3_variant_adapt_variant_reductions = Counter()
    de3_admission_cfg = de3_runtime_cfg.get("backtest_admission_controller", {}) or {}
    if not isinstance(de3_admission_cfg, dict):
        de3_admission_cfg = {}
    de3_admission_enabled = bool(de3_admission_cfg.get("enabled", False))
    de3_admission_key_granularity = str(
        de3_admission_cfg.get("key_granularity", "lane_context") or "lane_context"
    ).strip().lower()
    if de3_admission_key_granularity not in {"variant", "lane", "lane_context", "lane_margin_context"}:
        de3_admission_key_granularity = "lane_context"
    de3_admission_history_window = max(
        1,
        _coerce_int(de3_admission_cfg.get("history_window_trades", 20), 20),
    )
    de3_admission_warmup_trades = max(
        0,
        min(
            de3_admission_history_window,
            _coerce_int(de3_admission_cfg.get("warmup_trades", 8), 8),
        ),
    )
    de3_admission_cold_avg = _coerce_float(
        de3_admission_cfg.get("cold_avg_net_per_contract_usd", -10.0),
        -10.0,
    )
    de3_admission_cold_max_winrate = min(
        1.0,
        max(
            0.0,
            _coerce_float(de3_admission_cfg.get("cold_max_winrate", 0.40), 0.40),
        ),
    )
    de3_admission_defensive_mult = max(
        0.0,
        _coerce_float(de3_admission_cfg.get("defensive_size_multiplier", 0.50), 0.50),
    )
    raw_de3_admission_block_avg = de3_admission_cfg.get("block_avg_net_per_contract_usd")
    if raw_de3_admission_block_avg in (None, ""):
        de3_admission_block_avg = float("nan")
    else:
        de3_admission_block_avg = _coerce_float(raw_de3_admission_block_avg, float("nan"))
    raw_de3_admission_block_wr = de3_admission_cfg.get("block_max_winrate")
    if raw_de3_admission_block_wr in (None, ""):
        de3_admission_block_max_winrate = float("nan")
    else:
        de3_admission_block_max_winrate = _coerce_float(raw_de3_admission_block_wr, float("nan"))
    de3_admission_min_contracts = max(
        1,
        _coerce_int(de3_admission_cfg.get("min_contracts", 1), 1),
    )
    de3_admission_reduce_only = bool(de3_admission_cfg.get("reduce_only", True))
    de3_admission_require_signal_weakness = bool(
        de3_admission_cfg.get("require_signal_weakness", False)
    )
    raw_de3_admission_max_quality = de3_admission_cfg.get("max_execution_quality_score")
    de3_admission_max_execution_quality = (
        float("nan")
        if raw_de3_admission_max_quality in (None, "")
        else _coerce_float(raw_de3_admission_max_quality, float("nan"))
    )
    raw_de3_admission_max_margin = de3_admission_cfg.get("max_entry_model_margin")
    de3_admission_max_entry_margin = (
        float("nan")
        if raw_de3_admission_max_margin in (None, "")
        else _coerce_float(raw_de3_admission_max_margin, float("nan"))
    )
    raw_de3_admission_max_route = de3_admission_cfg.get("max_route_confidence")
    de3_admission_max_route_confidence = (
        float("nan")
        if raw_de3_admission_max_route in (None, "")
        else _coerce_float(raw_de3_admission_max_route, float("nan"))
    )
    raw_de3_admission_max_edge = de3_admission_cfg.get("max_edge_points")
    de3_admission_max_edge_points = (
        float("nan")
        if raw_de3_admission_max_edge in (None, "")
        else _coerce_float(raw_de3_admission_max_edge, float("nan"))
    )
    de3_admission_recent_pnl_history: defaultdict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=de3_admission_history_window)
    )
    de3_admission_checked = 0
    de3_admission_applied = 0
    de3_admission_blocked = 0
    de3_admission_state_counts = Counter()
    de3_admission_key_actions = Counter()
    de3_intraday_regime_cfg = (
        de3_runtime_cfg.get("backtest_intraday_regime_controller", {}) or {}
    )
    if not isinstance(de3_intraday_regime_cfg, dict):
        de3_intraday_regime_cfg = {}
    de3_intraday_regime_enabled = bool(de3_intraday_regime_cfg.get("enabled", False))
    de3_intraday_regime_mode = str(
        de3_intraday_regime_cfg.get("mode", "block_defensive") or "block_defensive"
    ).strip().lower()
    if de3_intraday_regime_mode not in {"block", "defensive", "block_defensive"}:
        de3_intraday_regime_mode = "block_defensive"
    de3_intraday_regime_apply_sessions = {
        str(item).strip().upper()
        for item in (de3_intraday_regime_cfg.get("apply_sessions", []) or [])
        if str(item).strip()
    }
    de3_intraday_regime_apply_lanes = {
        str(item).strip()
        for item in (de3_intraday_regime_cfg.get("apply_lanes", []) or [])
        if str(item).strip()
    }
    de3_intraday_regime_enable_bullish_mirror = bool(
        de3_intraday_regime_cfg.get("enable_bullish_mirror", True)
    )
    de3_intraday_regime_defensive_mult = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("defensive_size_multiplier", 0.50), 0.50),
    )
    de3_intraday_regime_min_contracts = max(
        1,
        _coerce_int(de3_intraday_regime_cfg.get("min_contracts", 1), 1),
    )
    de3_intraday_regime_reduce_only = bool(
        de3_intraday_regime_cfg.get("reduce_only", True)
    )
    de3_intraday_regime_defensive_score = _coerce_float(
        de3_intraday_regime_cfg.get("defensive_score_threshold", 2.80),
        2.80,
    )
    de3_intraday_regime_block_score = _coerce_float(
        de3_intraday_regime_cfg.get("block_score_threshold", 4.10),
        4.10,
    )
    de3_intraday_regime_dominance = _coerce_float(
        de3_intraday_regime_cfg.get("dominance_threshold", 0.70),
        0.70,
    )
    de3_intraday_regime_block_dominance = _coerce_float(
        de3_intraday_regime_cfg.get("block_dominance_threshold", 1.10),
        1.10,
    )
    de3_intraday_regime_require_weak_block = bool(
        de3_intraday_regime_cfg.get("require_signal_weakness_for_block", True)
    )
    de3_intraday_regime_require_weak_defensive = bool(
        de3_intraday_regime_cfg.get("require_signal_weakness_for_defensive", False)
    )
    raw_intraday_regime_max_quality = de3_intraday_regime_cfg.get("max_execution_quality_score")
    de3_intraday_regime_max_execution_quality = (
        float("nan")
        if raw_intraday_regime_max_quality in (None, "")
        else _coerce_float(raw_intraday_regime_max_quality, float("nan"))
    )
    raw_intraday_regime_max_margin = de3_intraday_regime_cfg.get("max_entry_model_margin")
    de3_intraday_regime_max_entry_margin = (
        float("nan")
        if raw_intraday_regime_max_margin in (None, "")
        else _coerce_float(raw_intraday_regime_max_margin, float("nan"))
    )
    raw_intraday_regime_max_route = de3_intraday_regime_cfg.get("max_route_confidence")
    de3_intraday_regime_max_route_confidence = (
        float("nan")
        if raw_intraday_regime_max_route in (None, "")
        else _coerce_float(raw_intraday_regime_max_route, float("nan"))
    )
    raw_intraday_regime_max_edge = de3_intraday_regime_cfg.get("max_edge_points")
    de3_intraday_regime_max_edge_points = (
        float("nan")
        if raw_intraday_regime_max_edge in (None, "")
        else _coerce_float(raw_intraday_regime_max_edge, float("nan"))
    )
    de3_intraday_regime_strong_quality = _coerce_float(
        de3_intraday_regime_cfg.get("strong_execution_quality_score", 0.86),
        0.86,
    )
    de3_intraday_regime_strong_margin = _coerce_float(
        de3_intraday_regime_cfg.get("strong_entry_model_margin", 0.22),
        0.22,
    )
    de3_intraday_regime_strong_route = _coerce_float(
        de3_intraday_regime_cfg.get("strong_route_confidence", 0.22),
        0.22,
    )
    de3_intraday_regime_strong_relief = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("strong_signal_relief", 0.65), 0.65),
    )
    de3_intraday_regime_pressure_lookback = max(
        2,
        _coerce_int(de3_intraday_regime_cfg.get("pressure_lookback_bars", 12), 12),
    )
    de3_intraday_regime_pressure_balance_min = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("pressure_balance_min", 0.16), 0.16),
    )
    de3_intraday_regime_pressure_weight = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("pressure_balance_weight", 0.90), 0.90),
    )
    de3_intraday_regime_net_return_min_atr = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("net_return_min_atr", 0.45), 0.45),
    )
    de3_intraday_regime_net_return_weight = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("net_return_weight", 0.85), 0.85),
    )
    de3_intraday_regime_session_move_min_atr = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("session_move_min_atr", 0.55), 0.55),
    )
    de3_intraday_regime_session_move_weight = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("session_move_weight", 0.95), 0.95),
    )
    de3_intraday_regime_vwap_dist_min_atr = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("vwap_dist_min_atr", 0.18), 0.18),
    )
    de3_intraday_regime_vwap_dist_scale_atr = max(
        1e-6,
        _coerce_float(de3_intraday_regime_cfg.get("vwap_dist_scale_atr", 0.65), 0.65),
    )
    de3_intraday_regime_vwap_dist_weight = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("vwap_dist_weight", 0.90), 0.90),
    )
    de3_intraday_regime_vwap_slope_lookback = max(
        2,
        _coerce_int(de3_intraday_regime_cfg.get("vwap_slope_lookback_bars", 8), 8),
    )
    de3_intraday_regime_vwap_slope_min_atr = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("vwap_slope_min_atr", 0.05), 0.05),
    )
    de3_intraday_regime_vwap_slope_scale_atr = max(
        1e-6,
        _coerce_float(de3_intraday_regime_cfg.get("vwap_slope_scale_atr", 0.16), 0.16),
    )
    de3_intraday_regime_vwap_slope_weight = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("vwap_slope_weight", 0.75), 0.75),
    )
    de3_intraday_regime_gap_location_weight = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("gap_location_weight", 0.60), 0.60),
    )
    de3_intraday_regime_gap_location_low = min(
        1.0,
        max(0.0, _coerce_float(de3_intraday_regime_cfg.get("gap_location_low", 0.35), 0.35)),
    )
    de3_intraday_regime_gap_location_high = min(
        1.0,
        max(0.0, _coerce_float(de3_intraday_regime_cfg.get("gap_location_high", 0.65), 0.65)),
    )
    de3_intraday_regime_gap_outside_scale_atr = max(
        1e-6,
        _coerce_float(de3_intraday_regime_cfg.get("gap_outside_scale_atr", 0.70), 0.70),
    )
    de3_intraday_regime_gap_outside_weight = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("gap_outside_weight", 0.70), 0.70),
    )
    de3_intraday_regime_route_bias_min = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("route_bias_min", 0.12), 0.12),
    )
    de3_intraday_regime_route_bias_weight = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("route_bias_weight", 0.55), 0.55),
    )
    de3_intraday_regime_or_minutes = max(
        1,
        _coerce_int(de3_intraday_regime_cfg.get("opening_range_minutes", 15), 15),
    )
    de3_intraday_regime_or_scale_atr = max(
        1e-6,
        _coerce_float(de3_intraday_regime_cfg.get("opening_range_break_scale_atr", 0.70), 0.70),
    )
    de3_intraday_regime_or_weight = max(
        0.0,
        _coerce_float(de3_intraday_regime_cfg.get("opening_range_weight", 1.10), 1.10),
    )
    de3_intraday_regime_checked = 0
    de3_intraday_regime_applied = 0
    de3_intraday_regime_blocked = 0
    de3_intraday_regime_state_counts = Counter()
    de3_intraday_regime_action_counts = Counter()
    de3_walkforward_gate_cfg = (
        de3_runtime_cfg.get("backtest_walkforward_gate", {}) or {}
    )
    if not isinstance(de3_walkforward_gate_cfg, dict):
        de3_walkforward_gate_cfg = {}
    de3_walkforward_gate_enabled = bool(de3_walkforward_gate_cfg.get("enabled", False))
    de3_walkforward_gate_mode = str(
        de3_walkforward_gate_cfg.get("mode", "block_defensive") or "block_defensive"
    ).strip().lower()
    if de3_walkforward_gate_mode not in {"block", "defensive", "block_defensive"}:
        de3_walkforward_gate_mode = "block_defensive"
    de3_walkforward_gate_defensive_mult = max(
        0.0,
        _coerce_float(de3_walkforward_gate_cfg.get("defensive_size_multiplier", 0.50), 0.50),
    )
    de3_walkforward_gate_min_contracts = max(
        1,
        _coerce_int(de3_walkforward_gate_cfg.get("min_contracts", 1), 1),
    )
    de3_walkforward_gate_reduce_only = bool(
        de3_walkforward_gate_cfg.get("reduce_only", True)
    )
    de3_walkforward_gate_artifact_raw = str(
        de3_walkforward_gate_cfg.get("artifact_path", "") or ""
    ).strip()
    de3_walkforward_gate_artifact_path: Optional[Path] = None
    de3_walkforward_gate_artifact: dict = {}
    if de3_walkforward_gate_artifact_raw:
        de3_walkforward_gate_artifact_path = Path(de3_walkforward_gate_artifact_raw).expanduser()
        if not de3_walkforward_gate_artifact_path.is_absolute():
            de3_walkforward_gate_artifact_path = (
                Path(__file__).resolve().parent / de3_walkforward_gate_artifact_path
            ).resolve()
        if de3_walkforward_gate_artifact_path.is_file():
            de3_walkforward_gate_artifact = load_de3_walkforward_artifact(
                de3_walkforward_gate_artifact_path
            )
    if de3_walkforward_gate_enabled and not de3_walkforward_gate_artifact:
        logging.warning(
            "DE3 walk-forward gate enabled but artifact missing/invalid: %s",
            de3_walkforward_gate_artifact_path or de3_walkforward_gate_artifact_raw or "<none>",
        )
        de3_walkforward_gate_enabled = False
    de3_walkforward_gate_periods = (
        list(de3_walkforward_gate_artifact.get("periods", []))
        if isinstance(de3_walkforward_gate_artifact.get("periods", []), list)
        else []
    )
    de3_walkforward_gate_compiled_periods: list[dict] = []
    for _period in de3_walkforward_gate_periods:
        if not isinstance(_period, dict):
            continue
        try:
            _start_ts = pd.Timestamp(_period.get("start")).to_pydatetime()
            _end_ts = pd.Timestamp(_period.get("end")).to_pydatetime()
        except Exception:
            continue
        if _start_ts.tzinfo is None:
            _start_ts = _start_ts.replace(tzinfo=NY_TZ)
        else:
            _start_ts = _start_ts.astimezone(NY_TZ)
        if _end_ts.tzinfo is None:
            _end_ts = _end_ts.replace(tzinfo=NY_TZ)
        else:
            _end_ts = _end_ts.astimezone(NY_TZ)
        _compiled = dict(_period)
        _compiled["_start_ts"] = _start_ts
        _compiled["_end_ts"] = _end_ts
        de3_walkforward_gate_compiled_periods.append(_compiled)
    de3_walkforward_gate_lane_history_window = max(
        1,
        _coerce_int(de3_walkforward_gate_artifact.get("lane_context_history_window", 20), 20),
    )
    de3_walkforward_gate_variant_history_window = max(
        1,
        _coerce_int(de3_walkforward_gate_artifact.get("variant_history_window", 12), 12),
    )
    de3_walkforward_gate_lane_history: defaultdict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=de3_walkforward_gate_lane_history_window)
    )
    de3_walkforward_gate_variant_history: defaultdict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=de3_walkforward_gate_variant_history_window)
    )
    de3_walkforward_gate_model_cache: dict[str, dict] = {}
    de3_walkforward_gate_checked = 0
    de3_walkforward_gate_applied = 0
    de3_walkforward_gate_blocked = 0
    de3_walkforward_gate_state_counts = Counter()
    de3_walkforward_gate_period_hits = Counter()
    de3_entry_margin_cfg = (
        de3_runtime_cfg.get("backtest_entry_model_margin_controller", {}) or {}
    )
    if not isinstance(de3_entry_margin_cfg, dict):
        de3_entry_margin_cfg = {}
    de3_entry_margin_enabled = bool(de3_entry_margin_cfg.get("enabled", False))
    de3_entry_margin_min_contracts = max(
        1,
        _coerce_int(de3_entry_margin_cfg.get("min_contracts", 1), 1),
    )
    de3_entry_margin_max_contracts = max(
        de3_entry_margin_min_contracts,
        _coerce_int(de3_entry_margin_cfg.get("max_contracts", CONTRACTS), CONTRACTS),
    )
    de3_entry_margin_reduce_only = bool(de3_entry_margin_cfg.get("reduce_only", False))
    de3_entry_margin_defensive_max = _coerce_float(
        de3_entry_margin_cfg.get("defensive_max_margin", 0.08),
        0.08,
    )
    de3_entry_margin_defensive_mult = max(
        0.0,
        _coerce_float(de3_entry_margin_cfg.get("defensive_size_multiplier", 0.60), 0.60),
    )
    de3_entry_margin_lane_scope_mult = max(
        0.0,
        _coerce_float(de3_entry_margin_cfg.get("lane_scope_size_multiplier", 0.80), 0.80),
    )
    de3_entry_margin_conservative_mult = max(
        0.0,
        _coerce_float(
            de3_entry_margin_cfg.get("conservative_tier_size_multiplier", 0.80),
            0.80,
        ),
    )
    de3_entry_margin_aggressive_min = _coerce_float(
        de3_entry_margin_cfg.get("aggressive_min_margin", 0.22),
        0.22,
    )
    de3_entry_margin_aggressive_mult = max(
        0.0,
        _coerce_float(de3_entry_margin_cfg.get("aggressive_size_multiplier", 1.25), 1.25),
    )
    de3_entry_margin_aggressive_variant_only = bool(
        de3_entry_margin_cfg.get("aggressive_variant_only", True)
    )
    de3_entry_margin_checked = 0
    de3_entry_margin_applied = 0
    de3_entry_margin_state_counts = Counter()
    de3_signal_size_cfg = de3_runtime_cfg.get("backtest_signal_size_rules", {}) or {}
    if not isinstance(de3_signal_size_cfg, dict):
        de3_signal_size_cfg = {}
    de3_signal_size_enabled = bool(de3_signal_size_cfg.get("enabled", False))
    de3_signal_size_log_applies = bool(de3_signal_size_cfg.get("log_applies", False))
    de3_signal_size_rules = [
        dict(rule)
        for rule in (
            de3_signal_size_cfg.get("rules", [])
            if isinstance(de3_signal_size_cfg.get("rules", []), (list, tuple))
            else []
        )
        if isinstance(rule, dict)
    ]
    de3_signal_size_applied = 0
    de3_signal_size_rule_hits = Counter()
    de3_policy_overlay_cfg = de3_runtime_cfg.get("backtest_policy_context_overlay", {}) or {}
    if not isinstance(de3_policy_overlay_cfg, dict):
        de3_policy_overlay_cfg = {}
    de3_policy_overlay_enabled = bool(de3_policy_overlay_cfg.get("enabled", False))
    de3_policy_overlay_reduce_only = bool(de3_policy_overlay_cfg.get("reduce_only", True))
    de3_policy_overlay_min_contracts = max(
        1,
        _coerce_int(de3_policy_overlay_cfg.get("min_contracts", 1), 1),
    )
    raw_de3_policy_overlay_min_conf = de3_policy_overlay_cfg.get("min_policy_confidence")
    de3_policy_overlay_min_confidence = (
        float("nan")
        if raw_de3_policy_overlay_min_conf in (None, "")
        else _coerce_float(raw_de3_policy_overlay_min_conf, float("nan"))
    )
    de3_policy_overlay_min_bucket_samples = max(
        0,
        _coerce_int(de3_policy_overlay_cfg.get("min_policy_bucket_samples", 120), 120),
    )
    de3_policy_overlay_require_signal_weakness = bool(
        de3_policy_overlay_cfg.get("require_signal_weakness", True)
    )
    raw_de3_policy_overlay_max_quality = de3_policy_overlay_cfg.get("max_execution_quality_score")
    de3_policy_overlay_max_execution_quality = (
        float("nan")
        if raw_de3_policy_overlay_max_quality in (None, "")
        else _coerce_float(raw_de3_policy_overlay_max_quality, float("nan"))
    )
    raw_de3_policy_overlay_max_margin = de3_policy_overlay_cfg.get("max_entry_model_margin")
    de3_policy_overlay_max_entry_margin = (
        float("nan")
        if raw_de3_policy_overlay_max_margin in (None, "")
        else _coerce_float(raw_de3_policy_overlay_max_margin, float("nan"))
    )
    raw_de3_policy_overlay_max_route = de3_policy_overlay_cfg.get("max_route_confidence")
    de3_policy_overlay_max_route_confidence = (
        float("nan")
        if raw_de3_policy_overlay_max_route in (None, "")
        else _coerce_float(raw_de3_policy_overlay_max_route, float("nan"))
    )
    raw_de3_policy_overlay_max_edge = de3_policy_overlay_cfg.get("max_edge_points")
    de3_policy_overlay_max_edge_points = (
        float("nan")
        if raw_de3_policy_overlay_max_edge in (None, "")
        else _coerce_float(raw_de3_policy_overlay_max_edge, float("nan"))
    )
    de3_policy_overlay_checked = 0
    de3_policy_overlay_applied = 0
    de3_policy_overlay_state_counts = Counter()

    ml_disabled_sessions = set(CONFIG.get("ML_PHYSICS_BACKTEST_DISABLED_SESSIONS", []) or [])
    ml_filter_bypass_cfg = CONFIG.get("BACKTEST_ML_PHYSICS_FILTER_BYPASS", {})
    if isinstance(ml_filter_bypass_cfg, dict):
        ml_filter_bypass_enabled = bool(ml_filter_bypass_cfg.get("enabled", False))
        ml_filter_bypass_sessions = {
            str(s).upper()
            for s in (ml_filter_bypass_cfg.get("sessions") or [])
            if s is not None and str(s).strip()
        }
    else:
        ml_filter_bypass_enabled = bool(ml_filter_bypass_cfg)
        ml_filter_bypass_sessions = set()
    if ml_filter_bypass_enabled:
        sessions_txt = ",".join(sorted(ml_filter_bypass_sessions)) if ml_filter_bypass_sessions else "ALL"
        logging.warning(
            "Backtest ML filter-bypass test mode ENABLED for sessions=%s (MLPhysics only).",
            sessions_txt,
        )
    ml_priority_boost_cfg = CONFIG.get("ML_PHYSICS_PRIORITY_BOOST", {}) or {}
    ml_priority_boost_enabled = bool(ml_priority_boost_cfg.get("enabled", False))
    try:
        ml_priority_boost_min_conf = float(ml_priority_boost_cfg.get("min_confidence", 0.0))
    except Exception:
        ml_priority_boost_min_conf = 0.0
    ml_priority_boost_sessions = tuple(ml_priority_boost_cfg.get("sessions") or [])
    try:
        ml_priority_boost_priority = int(ml_priority_boost_cfg.get("boost_priority", 1))
    except Exception:
        ml_priority_boost_priority = 1
    ml_runner_priority_boost_cfg = CONFIG.get("ML_PHYSICS_RUNNER_PRIORITY_BOOST", {}) or {}
    ml_runner_priority_boost_enabled = bool(ml_runner_priority_boost_cfg.get("enabled", False))
    try:
        ml_runner_priority_boost_priority = int(
            ml_runner_priority_boost_cfg.get("boost_priority", 1)
        )
    except Exception:
        ml_runner_priority_boost_priority = 1

    ml_soft_cfg = CONFIG.get("ML_PHYSICS_SOFT_GATING", {}) or {}
    ml_soft_enabled = bool(ml_soft_cfg.get("enabled", False))
    if bool(CONFIG.get("BACKTEST_ML_PHYSICS_GLOBAL_FILTERS_ONLY", False)):
        ml_soft_enabled = False
    ml_soft_sessions = tuple(ml_soft_cfg.get("sessions") or [])
    ml_soft_block_standard = bool(ml_soft_cfg.get("block_standard", True))
    ml_soft_block_fast = bool(ml_soft_cfg.get("block_fast", False))
    try:
        ml_soft_min_conf = float(ml_soft_cfg.get("min_confidence", 0.0))
    except Exception:
        ml_soft_min_conf = 0.0

    asia_soft_ext_cfg = CONFIG.get("ASIA_SOFT_EXTENSION_FILTER", {}) or {}
    asia_soft_ext_feature_enabled = bool(asia_soft_ext_cfg.get("enabled", False))
    try:
        asia_soft_ext_base = float(asia_soft_ext_cfg.get("base_score", 1.0))
    except Exception:
        asia_soft_ext_base = 1.0
    try:
        asia_soft_ext_penalty = float(asia_soft_ext_cfg.get("penalty", 0.35))
    except Exception:
        asia_soft_ext_penalty = 0.35
    try:
        asia_soft_ext_threshold = float(asia_soft_ext_cfg.get("score_threshold", 0.65))
    except Exception:
        asia_soft_ext_threshold = 0.65
    sltp_min_cfg = CONFIG.get("SLTP_MIN", {}) or {}
    sltp_exec_min_sl = round_points_to_tick(
        max(_coerce_float(sltp_min_cfg.get("sl", TICK_SIZE), TICK_SIZE), TICK_SIZE)
    )
    sltp_exec_min_tp = round_points_to_tick(
        max(_coerce_float(sltp_min_cfg.get("tp", TICK_SIZE), TICK_SIZE), TICK_SIZE)
    )

    regime_manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
    regime_manifold_bt_override = CONFIG.get("BACKTEST_REGIME_MANIFOLD", {}) or {}
    if isinstance(regime_manifold_bt_override, dict):
        regime_manifold_cfg.update(regime_manifold_bt_override)
    regime_manifold_mode = str(regime_manifold_cfg.get("mode", "enforce") or "enforce").lower()
    if regime_manifold_mode not in {"enforce", "shadow"}:
        regime_manifold_mode = "enforce"
    regime_manifold_enabled = bool(regime_manifold_cfg.get("enabled", False))
    if de3_manifold_adapt_enabled and not regime_manifold_enabled:
        regime_manifold_enabled = True
        logging.info(
            "Backtest RegimeManifold enabled for DE3 manifold adaptation (filter path remains optional)."
        )
    regime_manifold_engine = (
        RegimeManifoldEngine(regime_manifold_cfg) if regime_manifold_enabled else None
    )
    manifold_enforce_side_bias = bool(regime_manifold_cfg.get("enforce_side_bias", True))
    manifold_last_label = None
    manifold_checked = 0
    manifold_would_block = 0
    manifold_blocked = 0
    manifold_shadow_would_block = 0
    manifold_reasons = Counter()
    manifold_strategy_cfg = CONFIG.get("MANIFOLD_STRATEGY", {}) or {}
    manifold_bt_hard_filters_only = bool(
        manifold_strategy_cfg.get("backtest_hard_filters_only", False)
    )
    if regime_manifold_engine is not None:
        logging.info("Backtest RegimeManifold active (mode=%s)", regime_manifold_mode)

    regime_meta = None
    active_trade = None
    pending_entry = None
    pending_exit = False
    pending_exit_reason = None
    pending_loose_signals = {}
    opposite_signal_count = 0
    bar_count = 0
    processed_bars = 0
    cancelled = False
    session_flat_closes = 0
    session_entry_blocks = 0
    holiday_flat_closes = 0
    holiday_entry_blocks = 0
    holiday_signal_blocks = 0
    sl_cap_shadow_lock_until_index = -1
    sl_cap_shadow_lock_trigger_count = 0
    sl_cap_shadow_lock_entry_blocks = 0
    last_time = None
    last_close = None
    hostile_day_active = False
    hostile_day_reason = ""
    hostile_day_date = None
    hostile_engine_stats = {
        "DynamicEngine": {"trades": 0, "losses": 0},
        "Continuation": {"trades": 0, "losses": 0},
    }
    mom_rescue_date = None
    mom_rescue_scores = {"Long_Mom": 0, "Short_Mom": 0}

    trend_day_tier = 0
    trend_day_dir = None
    trend_day_max_sigma = 0.0
    impulse_day = None
    impulse_active = False
    impulse_dir = None
    impulse_start_price = None
    impulse_extreme = None
    pullback_extreme = None
    max_retracement = 0.0
    bars_since_impulse = 0
    last_trend_day_tier = 0
    last_trend_day_dir = None
    tier1_down_until = None
    tier1_up_until = None
    tier1_seen = False
    sticky_trend_dir = None
    sticky_reclaim_count = 0
    sticky_opposite_count = 0
    last_trend_session = None
    current_session = "OFF"
    trend_session = "OFF"
    vol_regime_current = None
    regime_meta = None
    is_choppy = False
    chop_reason = ""
    allowed_chop_side = None
    asia_viable = True
    asia_trend_bias_side = None

    def _deactivate_trend_day(reason: str, now: dt.datetime) -> None:
        nonlocal trend_day_tier, trend_day_dir
        nonlocal last_trend_day_tier, last_trend_day_dir
        nonlocal tier1_down_until, tier1_up_until, tier1_seen
        nonlocal sticky_trend_dir, sticky_opposite_count, sticky_reclaim_count
        nonlocal trend_day_max_sigma

        if (trend_day_tier > 0 or trend_day_dir) and BACKTEST_TRENDDAY_VERBOSE:
            print(
                f"\033[93m[TrendDay] Deactivated: {reason} @ "
                f"{now.strftime('%Y-%m-%d %H:%M')}\033[0m"
            )
        trend_day_tier = 0
        trend_day_dir = None
        last_trend_day_tier = 0
        last_trend_day_dir = None
        tier1_down_until = None
        tier1_up_until = None
        tier1_seen = False
        sticky_trend_dir = None
        sticky_opposite_count = 0
        sticky_reclaim_count = 0
        trend_day_max_sigma = 0.0

    def reset_mom_rescues(day: dt.date) -> None:
        nonlocal mom_rescue_date, mom_rescue_scores
        mom_rescue_date = day
        mom_rescue_scores = {"Long_Mom": 0, "Short_Mom": 0}

    def get_mom_rescue_key(origin_strategy: Optional[str], origin_sub: Optional[str]) -> Optional[str]:
        if not origin_strategy or not str(origin_strategy).startswith("DynamicEngine"):
            return None
        sub = str(origin_sub or "")
        if "_Long_Mom_" in sub:
            return "Long_Mom"
        if "_Short_Mom_" in sub:
            return "Short_Mom"
        return None

    def mom_rescue_banned(
        current_time: dt.datetime,
        origin_strategy: Optional[str],
        origin_sub: Optional[str],
    ) -> bool:
        key = get_mom_rescue_key(origin_strategy, origin_sub)
        if key is None:
            return False
        day = current_time.astimezone(NY_TZ).date()
        if mom_rescue_date != day:
            reset_mom_rescues(day)
        return mom_rescue_scores.get(key, 0) <= -1

    def update_mom_rescue_score(trade: dict, pnl_net: float, exit_time: dt.datetime) -> None:
        if trade.get("entry_mode") != "rescued":
            return
        if not str(trade.get("strategy", "")).startswith("Continuation_"):
            return
        key = get_mom_rescue_key(trade.get("rescue_from_strategy"), trade.get("rescue_from_sub_strategy"))
        if key is None:
            return
        day = exit_time.astimezone(NY_TZ).date()
        if mom_rescue_date != day:
            reset_mom_rescues(day)
        mom_rescue_scores[key] += 1 if pnl_net >= 0 else -1


    def reset_hostile_day(day: dt.date) -> None:
        nonlocal hostile_day_active, hostile_day_reason, hostile_day_date, hostile_engine_stats
        hostile_day_active = False
        hostile_day_reason = ""
        hostile_day_date = day
        hostile_engine_stats = {
            "DynamicEngine": {"trades": 0, "losses": 0},
            "Continuation": {"trades": 0, "losses": 0},
        }

    def update_hostile_day_on_close(strategy: Optional[str], pnl_points: float, exit_time: dt.datetime) -> None:
        nonlocal hostile_day_active, hostile_day_reason
        if not ENABLE_HOSTILE_DAY_GUARD or exit_time is None:
            return
        day = exit_time.astimezone(NY_TZ).date()
        if hostile_day_date != day:
            reset_hostile_day(day)
        engine_key = None
        if strategy == "DynamicEngine":
            engine_key = "DynamicEngine"
        elif strategy and str(strategy).startswith("Continuation_"):
            engine_key = "Continuation"
        if engine_key is None:
            return
        stats = hostile_engine_stats[engine_key]
        if stats["trades"] >= HOSTILE_DAY_MAX_TRADES:
            return
        stats["trades"] += 1
        if pnl_points < 0:
            stats["losses"] += 1
        dyn = hostile_engine_stats["DynamicEngine"]
        cont = hostile_engine_stats["Continuation"]
        if (
            dyn["trades"] >= HOSTILE_DAY_MIN_TRADES
            and cont["trades"] >= HOSTILE_DAY_MIN_TRADES
            and dyn["losses"] >= HOSTILE_DAY_LOSS_THRESHOLD
            and cont["losses"] >= HOSTILE_DAY_LOSS_THRESHOLD
        ):
            hostile_day_active = True
            hostile_day_reason = (
                f"DynamicEngine {dyn['losses']}/{dyn['trades']} losses "
                f"+ Continuation {cont['losses']}/{cont['trades']} losses"
            )

    def _safe_num(value) -> Optional[float]:
        try:
            num = float(value)
        except Exception:
            return None
        if not math.isfinite(num):
            return None
        return float(num)

    def _safe_bool(value) -> Optional[bool]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return bool(value)

    def _safe_idx(value) -> Optional[int]:
        try:
            idx = int(value)
        except Exception:
            return None
        if idx < 0 or idx >= len(full_df):
            return None
        return idx

    def _strategy_relevant_payload(signal_payload: Optional[dict]) -> dict:
        if not isinstance(signal_payload, dict):
            return {}
        key_prefixes = (
            "ml_",
            "vab_",
            "de3_",
            "regime_manifold_",
            "manifold_",
            "sltp_",
            "trend_day_",
            "rescue_",
            "consensus_",
        )
        key_allow = {
            "strategy",
            "sub_strategy",
            "side",
            "entry_mode",
            "vol_regime",
            "sl_dist",
            "tp_dist",
            "size",
            "confidence",
            "ml_confidence",
            "final_score",
            "score",
            "opt_wr",
            "wr",
            "win_rate",
            "combo_key",
            "reverted",
            "original_signal",
            "bypassed_filters",
            "sltp_bracket",
        }
        payload = {}
        for key, value in signal_payload.items():
            key_str = str(key)
            if key_str in key_allow or key_str.startswith(key_prefixes):
                payload[key_str] = _serialize_json_value(value)
        return payload

    def _resolve_runtime_vol_regime(signal_payload: Optional[dict], _bar_index: Optional[int]) -> Optional[str]:
        if isinstance(signal_payload, dict):
            signal_regime = signal_payload.get("vol_regime")
            if signal_regime not in (None, "", "UNKNOWN"):
                return str(signal_regime)
        if vol_regime_current not in (None, "", "UNKNOWN"):
            return str(vol_regime_current)
        return None

    def _snapshot_market_conditions(
        signal_payload: Optional[dict],
        ts: Optional[dt.datetime],
        phase: str,
        bar_index: Optional[int] = None,
        execution_price: Optional[float] = None,
    ) -> dict:
        if not market_snapshots_enabled:
            return {}
        idx = _safe_idx(bar_index)
        session_name = get_session_name(ts) if ts else None
        if (not session_name or session_name == "OFF") and idx is not None:
            session_name = str(full_session_arr[idx])
        if not session_name:
            session_name = str(current_session or "OFF")
        trend_session_name = "NY" if session_name in ("NY_AM", "NY_PM") else session_name
        if idx is not None:
            trend_session_name = str(full_trend_session_arr[idx])

        bar_open_val = _safe_num(open_arr[idx]) if idx is not None else None
        bar_high_val = _safe_num(high_arr[idx]) if idx is not None else None
        bar_low_val = _safe_num(low_arr[idx]) if idx is not None else None
        bar_close_val = _safe_num(close_arr[idx]) if idx is not None else None

        td_payload = {
            "tier": (
                signal_payload.get("trend_day_tier", trend_day_tier)
                if isinstance(signal_payload, dict)
                else trend_day_tier
            ),
            "dir": (
                signal_payload.get("trend_day_dir", trend_day_dir)
                if isinstance(signal_payload, dict)
                else trend_day_dir
            ),
            "sticky_dir": sticky_trend_dir,
            "impulse_active": bool(impulse_active),
            "impulse_dir": impulse_dir,
            "bars_since_impulse": int(bars_since_impulse),
            "max_retracement": _safe_num(max_retracement),
            "max_sigma": _safe_num(trend_day_max_sigma),
        }
        if idx is not None:
            td_payload.update(
                {
                    "ema50": _safe_num(td_ema50_arr[idx]),
                    "ema200": _safe_num(td_ema200_arr[idx]),
                    "atr_expansion": _safe_num(td_atr_exp_arr[idx]),
                    "vwap": _safe_num(td_vwap_arr[idx]),
                    "vwap_sigma": _safe_num(td_vwap_sigma_arr[idx]),
                    "reclaim_down": _safe_bool(td_reclaim_down_arr[idx]),
                    "reclaim_up": _safe_bool(td_reclaim_up_arr[idx]),
                    "no_reclaim_down_t1": _safe_bool(td_no_reclaim_down_t1_arr[idx]),
                    "no_reclaim_up_t1": _safe_bool(td_no_reclaim_up_t1_arr[idx]),
                    "no_reclaim_down_t2": _safe_bool(td_no_reclaim_down_t2_arr[idx]),
                    "no_reclaim_up_t2": _safe_bool(td_no_reclaim_up_t2_arr[idx]),
                    "trend_up_alt": _safe_bool(td_trend_up_alt_arr[idx]),
                    "trend_down_alt": _safe_bool(td_trend_down_alt_arr[idx]),
                    "adx_strong_up": _safe_bool(td_adx_strong_up_arr[idx]),
                    "adx_strong_down": _safe_bool(td_adx_strong_down_arr[idx]),
                }
            )

        signal_vol_regime = None
        if isinstance(signal_payload, dict):
            sig_vr = signal_payload.get("vol_regime")
            if sig_vr is not None:
                signal_vol_regime = str(sig_vr)
        runtime_vol_regime = _resolve_runtime_vol_regime(signal_payload, idx)

        manifold_payload = {}
        if isinstance(regime_meta, dict):
            for key in ("regime", "R", "stress", "risk_mult", "side_bias", "no_trade"):
                if key in regime_meta:
                    manifold_payload[key] = regime_meta.get(key)
        if isinstance(signal_payload, dict):
            for key, value in signal_payload.items():
                key_str = str(key)
                if key_str.startswith("regime_manifold_"):
                    manifold_payload[key_str] = value
            if "R" not in manifold_payload and signal_payload.get("manifold_R") is not None:
                manifold_payload["R"] = signal_payload.get("manifold_R")
            if manifold_payload.get("regime") in (None, "", "UNKNOWN", "unknown"):
                for fallback_key in ("regime_manifold_regime", "manifold_regime", "meta_regime"):
                    fallback_val = signal_payload.get(fallback_key)
                    if fallback_val in (None, "", "UNKNOWN", "unknown", "nan"):
                        continue
                    manifold_payload["regime"] = str(fallback_val)
                    break
        if manifold_payload.get("regime") in (None, "", "UNKNOWN", "unknown"):
            fallback_val = manifold_payload.get("regime_manifold_regime")
            if fallback_val not in (None, "", "UNKNOWN", "unknown", "nan"):
                manifold_payload["regime"] = str(fallback_val)

        regime_label = manifold_payload.get("regime", "UNKNOWN")
        vol_label = runtime_vol_regime or signal_vol_regime or "UNKNOWN"
        tier_label = td_payload.get("tier")
        if tier_label is None:
            tier_label = 0
        dir_label = td_payload.get("dir") or "none"
        context_tag = (
            f"{session_name}|{trend_session_name}|tier{tier_label}_{dir_label}|"
            f"vol={vol_label}|chop={1 if is_choppy else 0}|rm={regime_label}"
        )

        snapshot = {
            "phase": str(phase),
            "ts": ts,
            "bar_index": idx,
            "session": session_name,
            "trend_session": trend_session_name,
            "bar": {
                "open": bar_open_val,
                "high": bar_high_val,
                "low": bar_low_val,
                "close": bar_close_val,
                "execution_price": _safe_num(execution_price),
            },
            "volatility": {
                "runtime_vol_regime": runtime_vol_regime,
                "signal_vol_regime": signal_vol_regime,
            },
            "trend_day": td_payload,
            "chop": {
                "is_choppy": bool(is_choppy),
                "reason": str(chop_reason or ""),
                "allowed_side": allowed_chop_side,
            },
            "asia": {
                "viable": bool(asia_viable),
                "trend_bias_side": asia_trend_bias_side,
            },
            "regime_manifold": manifold_payload,
            "strategy_context": _strategy_relevant_payload(signal_payload),
            "context_tag": context_tag,
        }
        return _serialize_json_value(snapshot)

    def _find_uncapped_shadow_exit_index(
        *,
        start_index: int,
        end_index: int,
        side_name: str,
        stop_price: float,
        take_price: float,
    ) -> Optional[int]:
        """Find earliest bar index where the uncapped shadow trade would exit."""
        if end_index < start_index:
            return None
        window = slice(start_index, end_index + 1)
        forced_slice = shadow_forced_close_arr[window].astype(bool, copy=True)
        open_slice = open_arr[window]
        high_slice = high_arr[window]
        low_slice = low_arr[window]
        if side_name == "LONG":
            hit_stop = low_slice <= stop_price
            hit_take = high_slice >= take_price
            if BACKTEST_GAP_FILLS:
                hit_gap = (open_slice <= stop_price) | (open_slice >= take_price)
            else:
                hit_gap = np.zeros_like(hit_stop, dtype=bool)
        else:
            hit_stop = high_slice >= stop_price
            hit_take = low_slice <= take_price
            if BACKTEST_GAP_FILLS:
                hit_gap = (open_slice >= stop_price) | (open_slice <= take_price)
            else:
                hit_gap = np.zeros_like(hit_stop, dtype=bool)
        exit_mask = forced_slice | hit_gap | hit_stop | hit_take
        hit_positions = np.flatnonzero(exit_mask)
        if hit_positions.size == 0:
            return None
        return int(start_index + int(hit_positions[0]))

    def close_trade(
        exit_price: float,
        exit_time: dt.datetime,
        exit_reason: str = "unknown",
        bar_index: Optional[int] = None,
    ) -> None:
        nonlocal equity, peak, max_dd, trades, wins, losses, active_trade, opposite_signal_count
        nonlocal sl_cap_shadow_lock_until_index, sl_cap_shadow_lock_trigger_count
        nonlocal de3_early_exit_close_count
        nonlocal de3_entry_trade_day_extreme_early_exit_close_profile_hits
        if active_trade is None:
            return
        side = active_trade["side"]
        entry_price = active_trade["entry_price"]
        final_leg_size = _coerce_int(active_trade.get("size", CONTRACTS), CONTRACTS)
        if final_leg_size < 1:
            final_leg_size = 1
        original_size = max(
            final_leg_size,
            _coerce_int(active_trade.get("original_size", final_leg_size), final_leg_size),
        )
        pnl_points = compute_pnl_points(side, entry_price, exit_price)
        final_leg_pnl_dollars = pnl_points * POINT_VALUE * final_leg_size
        final_leg_fee_paid = FEE_PER_CONTRACT_RT * final_leg_size
        partial_pnl_dollars = _coerce_float(active_trade.get("de3_partial_realized_pnl_dollars"), 0.0)
        partial_fee_paid = _coerce_float(active_trade.get("de3_partial_realized_fee_paid"), 0.0)
        partial_pnl_net = _coerce_float(active_trade.get("de3_partial_realized_pnl_net"), 0.0)
        pnl_dollars = float(partial_pnl_dollars + final_leg_pnl_dollars)
        fee_paid = float(partial_fee_paid + final_leg_fee_paid)
        pnl_net = float(partial_pnl_net + (final_leg_pnl_dollars - final_leg_fee_paid))
        equity += pnl_net
        trades += 1
        if pnl_net >= 0:
            wins += 1
        else:
            losses += 1
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_dd:
            max_dd = drawdown
        if (
            _is_de3_v4_trade_management_payload(active_trade)
            and str(exit_reason) == "early_exit"
        ):
            de3_early_exit_close_count += 1
            if bool(
                active_trade.get(
                    "de3_entry_trade_day_extreme_early_exit_profile_active",
                    False,
                )
            ):
                profile_name = str(
                    active_trade.get("de3_early_exit_profile_name", "")
                    or "entry_trade_day_extreme_early_exit"
                )
                de3_entry_trade_day_extreme_early_exit_close_profile_hits[
                    profile_name
                ] += 1
        entry_time = active_trade.get("entry_time")
        exit_market_conditions = _snapshot_market_conditions(
            active_trade,
            exit_time,
            phase="exit",
            bar_index=bar_index,
            execution_price=exit_price,
        )
        trade_record = {
            "trade_id": int(trades),
            "strategy": active_trade.get("strategy", "Unknown"),
            "sub_strategy": active_trade.get("sub_strategy"),
            "side": side,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": original_size,
            "final_leg_size": final_leg_size,
            "pnl_points": pnl_points,
            "pnl_dollars": pnl_dollars,
            "pnl_net": pnl_net,
            "fee_paid": fee_paid,
            "final_leg_pnl_dollars": final_leg_pnl_dollars,
            "final_leg_fee_paid": final_leg_fee_paid,
            "de3_partial_realized_pnl_dollars": partial_pnl_dollars,
            "de3_partial_realized_fee_paid": partial_fee_paid,
            "de3_partial_realized_pnl_net": partial_pnl_net,
            "sl_dist": active_trade.get("sl_dist", MIN_SL),
            "tp_dist": active_trade.get("tp_dist", MIN_TP),
            "mfe_points": active_trade.get("mfe_points", 0.0),
            "mae_points": active_trade.get("mae_points", 0.0),
            "entry_mode": active_trade.get("entry_mode", "standard"),
            "vol_regime": active_trade.get("vol_regime", "UNKNOWN"),
            "exit_reason": exit_reason,
            "bars_held": active_trade.get("bars_held", 0),
            "horizon_bars": active_trade.get("horizon_bars", 0),
            "use_horizon_time_stop": bool(active_trade.get("use_horizon_time_stop", False)),
            "session": get_session_name(entry_time) if entry_time else "OFF",
            "rescue_from_strategy": active_trade.get("rescue_from_strategy"),
            "rescue_from_sub_strategy": active_trade.get("rescue_from_sub_strategy"),
            "rescue_trigger": active_trade.get("rescue_trigger"),
            "consensus_contributors": active_trade.get("consensus_contributors"),
            "bypassed_filters": active_trade.get("bypassed_filters"),
            "trend_day_tier": active_trade.get("trend_day_tier"),
            "trend_day_dir": active_trade.get("trend_day_dir"),
            "market_conditions": active_trade.get("market_conditions", {}),
            "decision_market_conditions": active_trade.get("decision_market_conditions", {}),
            "exit_market_conditions": exit_market_conditions,
            "effective_stop_price": active_trade.get("current_stop_price"),
        }
        initial_stop_price = (
            entry_price - _coerce_float(active_trade.get("sl_dist", MIN_SL), MIN_SL)
            if side == "LONG"
            else entry_price + _coerce_float(active_trade.get("sl_dist", MIN_SL), MIN_SL)
        )
        effective_stop_price = _coerce_float(active_trade.get("current_stop_price"), math.nan)
        if (
            str(exit_reason) in {"stop", "stop_gap"}
            and math.isfinite(effective_stop_price)
            and (
                effective_stop_price > initial_stop_price + 1e-12
                if side == "LONG"
                else effective_stop_price < initial_stop_price - 1e-12
            )
        ):
            trade_record["de3_break_even_stop_hit"] = True
        for key, value in active_trade.items():
            if str(key).startswith(("ml_", "vab_", "de3_", "regime_manifold_")):
                trade_record[key] = value
        for key in ("combo_key", "reverted", "original_signal"):
            if key in active_trade:
                trade_record[key] = active_trade.get(key)
        tracker.record_trade(trade_record)
        _record_de3_variant_adaptation_outcome(trade_record)
        _record_de3_backtest_walkforward_gate_outcome(trade_record)
        _record_de3_backtest_admission_outcome(trade_record)
        update_mom_rescue_score(trade_record, pnl_net, exit_time)
        update_hostile_day_on_close(trade_record.get("strategy"), pnl_points, exit_time)
        if enabled_filter_directional_loss:
            directional_loss_blocker.record_trade_result(side, pnl_points, exit_time)
        if (
            de3_runtime_is_v2
            and str(active_trade.get("strategy", "")).startswith("DynamicEngine3")
            and str(exit_reason) in {"stop", "stop_gap"}
            and bool(active_trade.get("sl_cap_applied", False))
            and _coerce_int(active_trade.get("size", CONTRACTS), CONTRACTS) >= CONTRACTS
            and bar_index is not None
        ):
            req_sl = round_points_to_tick(
                max(
                    _coerce_float(
                        active_trade.get("requested_sl_dist"),
                        _coerce_float(active_trade.get("sl_dist", MIN_SL), MIN_SL),
                    ),
                    MIN_SL,
                )
            )
            cap_sl = round_points_to_tick(max(_coerce_float(active_trade.get("sl_dist", MIN_SL), MIN_SL), MIN_SL))
            if req_sl > cap_sl:
                entry_index = _safe_idx(active_trade.get("entry_index"))
                exit_index = _safe_idx(bar_index)
                if entry_index is not None and exit_index is not None and entry_index <= exit_index:
                    entry_price = _coerce_float(active_trade.get("entry_price"), math.nan)
                    tp_dist = round_points_to_tick(
                        max(_coerce_float(active_trade.get("tp_dist", MIN_TP), MIN_TP), MIN_TP)
                    )
                    side_name = str(active_trade.get("side", "")).upper()
                    if math.isfinite(entry_price) and tp_dist > 0 and side_name in {"LONG", "SHORT"}:
                        stop_price = entry_price - req_sl if side_name == "LONG" else entry_price + req_sl
                        take_price = entry_price + tp_dist if side_name == "LONG" else entry_price - tp_dist
                        # We only need a future lock window (strictly after realized exit).
                        scan_start = exit_index + 1
                        if scan_start >= len(full_df):
                            scan_end = -1
                        else:
                            scan_end = len(full_df) - 1
                            next_forced_idx = int(next_shadow_forced_close_idx[scan_start])
                            if next_forced_idx >= 0:
                                scan_end = min(scan_end, next_forced_idx)
                        uncapped_exit_index: Optional[int] = None
                        if scan_start <= scan_end:
                            uncapped_exit_index = _find_uncapped_shadow_exit_index(
                                start_index=scan_start,
                                end_index=scan_end,
                                side_name=side_name,
                                stop_price=stop_price,
                                take_price=take_price,
                            )
                        if uncapped_exit_index is not None and uncapped_exit_index > exit_index:
                            sl_cap_shadow_lock_until_index = max(
                                sl_cap_shadow_lock_until_index,
                                int(uncapped_exit_index),
                            )
                            sl_cap_shadow_lock_trigger_count += 1
        active_trade = None
        opposite_signal_count = 0

    def _cached_recent_trades() -> list[str]:
        nonlocal progress_recent_trades_cache, progress_recent_trades_last_trade_count
        trade_count_now = len(tracker.trades)
        if trade_count_now != progress_recent_trades_last_trade_count:
            progress_recent_trades_cache = [format_recent_trade(trade) for trade in tracker.recent_trades]
            progress_recent_trades_last_trade_count = trade_count_now
        return progress_recent_trades_cache

    def _cached_live_trade_log() -> list[dict]:
        nonlocal live_trade_log_cache, live_trade_log_last_count
        trade_count_now = len(tracker.trades)
        if trade_count_now < live_trade_log_last_count:
            live_trade_log_cache = []
            live_trade_log_last_count = 0
        if trade_count_now > live_trade_log_last_count:
            live_trade_log_cache.extend(
                serialize_trade(trade)
                for trade in tracker.trades[live_trade_log_last_count:trade_count_now]
            )
            live_trade_log_last_count = trade_count_now
        return live_trade_log_cache

    def write_live_report(
        current_time: Optional[dt.datetime],
        current_price: Optional[float],
        *,
        done: bool = False,
        force: bool = False,
    ) -> None:
        nonlocal live_report_last_write, live_report_write_error
        if not live_report_enabled or live_report_path is None or live_report_write_error:
            return
        now_ts = time.time()
        if not force and not done and (now_ts - live_report_last_write) < live_report_every_sec:
            return
        try:
            unrealized = 0.0
            if active_trade is not None and current_price is not None:
                size = active_trade.get("size", CONTRACTS)
                unrealized_points = compute_pnl_points(
                    active_trade["side"],
                    active_trade["entry_price"],
                    float(current_price),
                )
                unrealized = float(unrealized_points * POINT_VALUE * size)
            winrate_val = (wins / trades * 100.0) if trades else 0.0
            payload = {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "status": "done" if done else "running",
                "range_start": start_time.isoformat(),
                "range_end": end_time.isoformat(),
                "progress": {
                    "bar_index": int(bar_count),
                    "total_bars": int(total_bars),
                    "percent": float((bar_count / total_bars) * 100.0) if total_bars else 100.0,
                    "current_time": current_time.isoformat() if current_time is not None else None,
                    "cancelled": bool(cancelled),
                },
                "summary": {
                    "equity": float(equity),
                    "unrealized": float(unrealized),
                    "total": float(equity + unrealized),
                    "trades": int(trades),
                    "wins": int(wins),
                    "losses": int(losses),
                    "winrate": float(winrate_val),
                    "max_drawdown": float(max_dd),
                },
                "assumptions": {
                    "contracts": CONTRACTS,
                    "point_value": POINT_VALUE,
                    "bar_signal": "close",
                    "entry": "next_open",
                    "bar_minutes": bar_minutes,
                    "selected_strategies": selected_strategy_names,
                    "selected_filters": selected_filter_names,
                },
                "recent_trades": _cached_recent_trades(),
                "de3_runtime_db_version": de3_runtime_db_version,
                "de3_runtime_family_mode_enabled": bool(de3_runtime_family_mode_enabled),
                "de3_runtime_family_artifact": de3_runtime_family_artifact,
                "de3_runtime_family_artifact_loaded": bool(de3_runtime_family_artifact_loaded),
            }
            if live_report_include_trade_log:
                payload["trade_log"] = _cached_live_trade_log()
            tmp_path = live_report_path.with_suffix(f"{live_report_path.suffix}.tmp")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                encoding="utf-8",
            )
            tmp_path.replace(live_report_path)
            live_report_last_write = now_ts
        except Exception as exc:
            live_report_write_error = True
            logging.warning("Backtest live report write failed (%s): %s", live_report_path, exc)

    def emit_progress(
        current_time: dt.datetime,
        current_price: float,
        force: bool = False,
        done: bool = False,
    ) -> None:
        nonlocal progress_report_cache, progress_report_last_build, progress_report_last_trade_count
        if not force and progress_every > 0 and (bar_count % progress_every) != 0:
            return
        if current_time < start_time and not done:
            return
        unrealized = 0.0
        if active_trade is not None:
            size = active_trade.get("size", CONTRACTS)
            unrealized_points = compute_pnl_points(
                active_trade["side"],
                active_trade["entry_price"],
                current_price,
            )
            unrealized = unrealized_points * POINT_VALUE * size
        winrate = (wins / trades * 100.0) if trades else 0.0
        payload = {
            "time": current_time,
            "equity": equity,
            "unrealized": unrealized,
            "total": equity + unrealized,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "winrate": winrate,
            "max_drawdown": max_dd,
            "bar_index": bar_count,
            "total_bars": total_bars,
            "active_side": active_trade["side"] if active_trade else None,
            "done": done,
            "cancelled": cancelled,
        }
        if progress_cb is not None:
            now_ts = time.time()
            trade_count_now = len(tracker.trades)
            should_refresh_report = bool(
                force
                or done
                or not progress_report_cache
                or (progress_report_refresh_on_trade and (trade_count_now != progress_report_last_trade_count))
                or ((now_ts - progress_report_last_build) >= progress_report_every_sec)
            )
            if should_refresh_report:
                progress_report_cache = tracker.build_report()
                progress_report_last_build = now_ts
                progress_report_last_trade_count = trade_count_now
            payload["report"] = progress_report_cache
            payload["recent_trades"] = _cached_recent_trades()
        if progress_cb is not None:
            try:
                progress_cb(payload)
            except Exception:
                pass
        write_live_report(current_time, current_price, done=done, force=force)

    def record_filter(name: str, kind: str = "block") -> None:
        tracker.record_filter(name, kind)
        if (
            kind == "block"
            and flip_tracker is not None
            and flip_tracker.enabled
            and flip_context is not None
        ):
            try:
                flip_tracker.record_block(
                    name,
                    flip_context.get("signal"),
                    flip_context.get("bar_index", 0),
                    flip_context.get("current_time"),
                    flip_context.get("history_df", pd.DataFrame()),
                    flip_context.get("session_name"),
                    flip_context.get("vol_regime"),
                )
            except Exception:
                pass
        if kind == "block" and name.startswith("TrendDayTier"):
            try:
                ts = current_time.strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts = "N/A"
            try:
                price = f"{bar_close:.2f}"
            except Exception:
                price = "N/A"
            if BACKTEST_TRENDDAY_VERBOSE:
                color = "\033[92m" if trend_day_dir == "up" else "\033[91m"
                print(f"{color}[TrendDay] {name} blocked counter-trend signal @ {ts} price={price}\033[0m")

    def continuation_core_trigger(trigger: Optional[str]) -> bool:
        if not trigger:
            return False
        if trigger.startswith("FilterStack"):
            return True
        return trigger in {"RegimeBlocker", "TrendFilter", "ChopFilter", "ExtensionFilter"}

    def _ny_gate_session_name(ts: dt.datetime) -> Optional[str]:
        if ts is None:
            return None
        try:
            ts = ts.astimezone(NY_TZ)
        except Exception:
            return None
        hour = ts.hour
        if 8 <= hour < 12:
            return "NY_AM"
        if 12 <= hour < 17:
            return "NY_PM"
        return None

    def _calc_atr20(history_df: pd.DataFrame, bar_index: Optional[int] = None) -> Optional[float]:
        if bar_index is None:
            if history_df.empty:
                return None
            idx = len(history_df) - 1
        else:
            idx = int(bar_index)
        # Preserve historical behavior: require at least 21 rows before ATR20 is considered valid.
        if idx < 20 or idx >= len(atr20_arr):
            return None
        atr20 = atr20_arr[idx]
        if not math.isfinite(atr20):
            return None
        return float(atr20)

    def _close_position_in_bar(high: float, low: float, close: float) -> float:
        rng = float(high) - float(low)
        if rng <= 0:
            return 0.5
        return (float(close) - float(low)) / rng

    def _de3_session_vwap(
        history_df: pd.DataFrame,
        current_time: dt.datetime,
        bar_index: Optional[int] = None,
    ) -> Optional[float]:
        if bar_index is None:
            if history_df.empty:
                return None
            idx = len(history_df) - 1
        else:
            idx = int(bar_index)
        if idx < 0 or idx >= len(de3_session_vwap_arr):
            return None
        val = de3_session_vwap_arr[idx]
        if not math.isfinite(val):
            return None
        return float(val)

    def _de3_strategy_type(signal: dict) -> str:
        sub = str(signal.get("sub_strategy") or "")
        if "_Long_Mom_" in sub:
            return "Long_Mom"
        if "_Long_Rev_" in sub:
            return "Long_Rev"
        if "_Short_Mom_" in sub:
            return "Short_Mom"
        if "_Short_Rev_" in sub:
            return "Short_Rev"
        return "Unknown"

    def _de3_meta_common_context(
        history_df: pd.DataFrame,
        current_time: dt.datetime,
        bar_index: Optional[int] = None,
    ) -> Optional[dict]:
        if history_df.empty:
            return None
        atr20 = _calc_atr20(history_df, bar_index=bar_index)
        if atr20 is None or atr20 <= 0:
            return None

        metrics = _balance_metrics(history_df, lookback=de3_meta_er_lookback, bar_index=bar_index)
        er = float(metrics[0]) if metrics else 0.0
        close_pos = _close_position_in_bar(bar_high, bar_low, bar_close)

        gap_atr = 0.0
        idx = bar_index if bar_index is not None else (len(history_df) - 1 if not history_df.empty else None)
        if idx is not None and int(idx) >= 1:
            try:
                prev_close = float(close_arr[int(idx) - 1])
                gap_atr = abs(float(bar_open) - prev_close) / atr20
            except Exception:
                gap_atr = 0.0
        range_atr = abs(float(bar_high) - float(bar_low)) / atr20 if atr20 > 0 else 0.0

        vwap_val = _de3_session_vwap(history_df, current_time, bar_index=bar_index)
        vwap_dist_atr = 0.0
        if vwap_val is not None:
            vwap_dist_atr = abs(float(bar_close) - float(vwap_val)) / atr20

        return {
            "atr20": float(atr20),
            "er": float(er),
            "gap_atr": float(gap_atr),
            "range_atr": float(range_atr),
            "vwap_dist_atr": float(vwap_dist_atr),
            "close_pos": float(close_pos),
        }

    def _de3_meta_policy(
        signal: dict,
        history_df: pd.DataFrame,
        current_time: dt.datetime,
        vol_regime: Optional[str],
        common_ctx: Optional[dict] = None,
        bar_index: Optional[int] = None,
    ) -> dict:
        result = {
            "allow": True,
            "would_block": False,
            "score": 100.0,
            "reasons": [],
        }
        if common_ctx is None:
            common_ctx = _de3_meta_common_context(
                history_df,
                current_time,
                bar_index=bar_index,
            )
        if not common_ctx:
            return result

        side = str(signal.get("side") or "").upper()
        strat_type = _de3_strategy_type(signal)
        atr20 = float(common_ctx.get("atr20", 0.0) or 0.0)
        er = float(common_ctx.get("er", 0.0) or 0.0)
        gap_atr = float(common_ctx.get("gap_atr", 0.0) or 0.0)
        range_atr = float(common_ctx.get("range_atr", 0.0) or 0.0)
        vwap_dist_atr = float(common_ctx.get("vwap_dist_atr", 0.0) or 0.0)
        close_pos = float(common_ctx.get("close_pos", 0.5) or 0.5)

        hard_block = False
        if str(strat_type).lower() in de3_meta_blocked_types:
            result["reasons"].append(f"type_blocked:{strat_type}")
            result["score"] = min(float(result["score"]), 0.0)
            hard_block = True

        if de3_meta_shock_gap_mult > 0 and gap_atr > de3_meta_shock_gap_mult:
            result["reasons"].append(f"shock_gap:{gap_atr:.2f}>{de3_meta_shock_gap_mult:.2f}")
            result["score"] -= 55.0
            hard_block = True
        if de3_meta_shock_range_mult > 0 and range_atr > de3_meta_shock_range_mult:
            result["reasons"].append(f"shock_range:{range_atr:.2f}>{de3_meta_shock_range_mult:.2f}")
            result["score"] -= 45.0
            hard_block = True

        if strat_type.endswith("_Mom"):
            if er < de3_meta_mom_min_er:
                result["reasons"].append(f"mom_low_er:{er:.2f}<{de3_meta_mom_min_er:.2f}")
                result["score"] -= 30.0
            if strat_type == "Long_Mom":
                if str(vol_regime or "").lower() == "high" and vwap_dist_atr > de3_meta_hv_max_vwap:
                    result["reasons"].append(
                        f"long_mom_stretched:{vwap_dist_atr:.2f}>{de3_meta_hv_max_vwap:.2f}"
                    )
                    result["score"] -= 20.0
                if close_pos > de3_meta_long_mom_max_close_pos:
                    result["reasons"].append(
                        f"long_mom_close_pos:{close_pos:.2f}>{de3_meta_long_mom_max_close_pos:.2f}"
                    )
                    result["score"] -= 15.0
            elif strat_type == "Short_Mom" and close_pos < de3_meta_short_mom_min_close_pos:
                result["reasons"].append(
                    f"short_mom_close_pos:{close_pos:.2f}<{de3_meta_short_mom_min_close_pos:.2f}"
                )
                result["score"] -= 15.0
        elif strat_type.endswith("_Rev"):
            if er > de3_meta_rev_max_er:
                result["reasons"].append(f"rev_high_er:{er:.2f}>{de3_meta_rev_max_er:.2f}")
                result["score"] -= 30.0

        result["would_block"] = hard_block or (result["score"] < de3_meta_min_score)
        result["allow"] = not result["would_block"] or de3_meta_mode == "shadow"
        result["context"] = {
            "type": strat_type,
            "side": side,
            "vol_regime": str(vol_regime or ""),
            "atr20": float(atr20),
            "er": float(er),
            "gap_atr": float(gap_atr),
            "range_atr": float(range_atr),
            "vwap_dist_atr": float(vwap_dist_atr),
            "close_pos": float(close_pos),
        }
        return result

    def _get_opening_range_levels(
        history_df: pd.DataFrame,
        current_time: dt.datetime,
        minutes: int = 15,
        bar_index: Optional[int] = None,
    ):
        if minutes == 15 and bar_index is not None:
            idx = int(bar_index)
            if 0 <= idx < len(ny_orh_arr):
                orh = ny_orh_arr[idx]
                orl = ny_orl_arr[idx]
                return (
                    float(orh) if math.isfinite(orh) else None,
                    float(orl) if math.isfinite(orl) else None,
                )
        if history_df.empty or current_time is None:
            return None, None
        try:
            ny_time = current_time.astimezone(NY_TZ)
        except Exception:
            return None, None
        session_start = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
        session_end = session_start + dt.timedelta(minutes=minutes)
        window = history_df.loc[(history_df.index >= session_start) & (history_df.index < session_end)]
        if window.empty:
            return None, None
        return float(window["high"].max()), float(window["low"].min())

    def _session_high_low(
        history_df: pd.DataFrame,
        current_time: dt.datetime,
        session_name: str,
        bar_index: Optional[int] = None,
    ):
        if bar_index is not None:
            idx = int(bar_index)
            if 0 <= idx < len(full_df):
                if session_name == "NY_AM":
                    hi = ny_am_high_arr[idx]
                    lo = ny_am_low_arr[idx]
                    return (
                        float(hi) if math.isfinite(hi) else None,
                        float(lo) if math.isfinite(lo) else None,
                    )
                if session_name == "NY_PM":
                    hi = ny_pm_high_arr[idx]
                    lo = ny_pm_low_arr[idx]
                    return (
                        float(hi) if math.isfinite(hi) else None,
                        float(lo) if math.isfinite(lo) else None,
                    )
        if history_df.empty or current_time is None:
            return None, None
        try:
            ny_time = current_time.astimezone(NY_TZ)
        except Exception:
            return None, None
        if session_name == "NY_AM":
            start = ny_time.replace(hour=8, minute=0, second=0, microsecond=0)
            end = ny_time.replace(hour=12, minute=0, second=0, microsecond=0)
        elif session_name == "NY_PM":
            start = ny_time.replace(hour=12, minute=0, second=0, microsecond=0)
            end = ny_time.replace(hour=17, minute=0, second=0, microsecond=0)
        else:
            return None, None
        window = history_df.loc[(history_df.index >= start) & (history_df.index <= min(ny_time, end))]
        if window.empty:
            return None, None
        return float(window["high"].max()), float(window["low"].min())

    def _select_interaction_level(
        levels: dict[str, float],
        price: float,
        history_df: pd.DataFrame,
        band: float,
        bar_index: Optional[int] = None,
    ):
        if not levels:
            return None
        if bar_index is not None and int(bar_index) >= 1:
            prev_close = float(close_arr[int(bar_index) - 1])
        else:
            if history_df.empty or len(history_df) < 2:
                return None
            prev_close = float(history_df["close"].iloc[-2])
        candidates: list[tuple[str, float]] = []
        for name, level in levels.items():
            if level is None or not np.isfinite(level):
                continue
            if abs(price - level) <= band:
                candidates.append((name, float(level)))
                continue
            crossed = (prev_close - level) * (price - level) < 0
            if crossed:
                candidates.append((name, float(level)))
        if not candidates:
            return None
        return min(candidates, key=lambda item: abs(price - item[1]))

    def _balance_metrics(
        history_df: pd.DataFrame,
        lookback: int = 10,
        bar_index: Optional[int] = None,
    ):
        if lookback <= 1:
            return None
        if bar_index is not None:
            idx = int(bar_index)
            start = idx - int(lookback) + 1
            if start < 0 or idx >= len(close_arr):
                return None
            closes = close_arr[start : idx + 1].astype(float, copy=False)
            highs = high_arr[start : idx + 1].astype(float, copy=False)
            lows = low_arr[start : idx + 1].astype(float, copy=False)
        elif history_df.empty or len(history_df) < lookback:
            return None
        else:
            window = history_df.iloc[-lookback:]
            closes = window["close"].to_numpy(dtype=float)
            highs = window["high"].to_numpy(dtype=float)
            lows = window["low"].to_numpy(dtype=float)
        deltas = np.diff(closes)
        sum_abs = float(np.sum(np.abs(deltas)))
        net_move = float(abs(closes[-1] - closes[0]))
        er = net_move / max(sum_abs, 1e-9)

        signs = np.sign(deltas)
        if len(signs) > 1:
            sign_prev = signs[:-1]
            sign_curr = signs[1:]
            flips = int(
                np.sum(
                    (sign_curr != 0)
                    & (sign_prev != 0)
                    & (sign_curr != sign_prev)
                )
            )
        else:
            flips = 0

        if len(highs) > 1:
            overlap = np.minimum(highs[1:], highs[:-1]) - np.maximum(lows[1:], lows[:-1])
            overlap_count = int(np.sum(overlap > 0))
        else:
            overlap_count = 0
        overlap_ratio = overlap_count / max(lookback - 1, 1)
        return er, overlap_ratio, flips

    def _ny_continuation_gates(
        signal: dict,
        side: str,
        current_time: dt.datetime,
        bar_close: float,
        history_df: pd.DataFrame,
        bar_index: Optional[int] = None,
    ) -> bool:
        nonlocal ny_gate_candidates
        nonlocal ny_gate_balance_blocked, ny_gate_acceptance_blocked, ny_gate_liquidity_blocked
        if not signal or history_df.empty:
            return True
        strat_name = str(signal.get("strategy") or "")
        if not strat_name.lower().startswith("continuation"):
            return True
        if bar_index is not None and 0 <= int(bar_index) < len(full_session_arr):
            session_name = str(full_session_arr[int(bar_index)])
        else:
            session_name = _ny_gate_session_name(current_time)
        if session_name not in ("NY_AM", "NY_PM"):
            return True

        ny_gate_candidates += 1

        # Gate B: Local balance detector
        metrics = _balance_metrics(history_df, lookback=10, bar_index=bar_index)
        if metrics:
            er, overlap_ratio, flips = metrics
            if er < 0.25 or overlap_ratio > 0.70 or flips >= 6:
                ny_gate_balance_blocked += 1
                ny_gate_blocked_by_strategy[strat_name] += 1
                ny_gate_blocked_by_session[session_name] += 1
                record_filter("NYGateBalance")
                logging.info(
                    "⛔ NYGateBalance blocked | session=%s | strategy=%s | side=%s | ER=%.2f | overlap=%.2f | flips=%s",
                    session_name,
                    strat_name,
                    side,
                    er,
                    overlap_ratio,
                    flips,
                )
                return False

        # Gate A: Acceptance vs rejection at key levels
        atr20 = _calc_atr20(history_df, bar_index=bar_index)
        band = max(0.2 * (atr20 or 0.0), 0.5)
        levels_long: dict[str, float] = {}
        levels_short: dict[str, float] = {}

        if bar_index is not None and 0 <= int(bar_index) < len(full_df):
            idx = int(bar_index)
            vwap = td_vwap_arr[idx]
            pdh = td_prior_session_high_arr[idx]
            pdl = td_prior_session_low_arr[idx]
        else:
            def series_at(key: str):
                series = trend_day_series.get(key) if trend_day_series else None
                if isinstance(series, pd.Series):
                    try:
                        return series.get(current_time, None)
                    except Exception:
                        return None
                return series

            vwap = series_at("vwap")
            pdh = series_at("prior_session_high")
            pdl = series_at("prior_session_low")
        if vwap is not None and not pd.isna(vwap):
            levels_long["VWAP"] = float(vwap)
            levels_short["VWAP"] = float(vwap)
        if pdh is not None and not pd.isna(pdh):
            levels_long["PDH"] = float(pdh)
        if pdl is not None and not pd.isna(pdl):
            levels_short["PDL"] = float(pdl)

        orh, orl = _get_opening_range_levels(
            history_df,
            current_time,
            bar_index=bar_index,
        )
        if orh is not None:
            levels_long["ORH"] = orh
        if orl is not None:
            levels_short["ORL"] = orl

        if bar_index is not None and 0 <= int(bar_index) < len(full_df):
            idx = int(bar_index)
            if idx in ny_gate_vp_cache:
                vp = ny_gate_vp_cache[idx]
            else:
                vp = build_volume_profile(history_df, lookback=120, tick_size=TICK_SIZE)
                ny_gate_vp_cache[idx] = vp
        else:
            vp = build_volume_profile(history_df, lookback=120, tick_size=TICK_SIZE)
        if vp:
            vah = vp.get("vah")
            val = vp.get("val")
            if vah is not None and np.isfinite(vah):
                levels_long["VAH"] = float(vah)
            if val is not None and np.isfinite(val):
                levels_short["VAL"] = float(val)

        if bar_index is not None and int(bar_index) >= 1:
            idx = int(bar_index)
            closes = np.array([float(close_arr[idx - 1]), float(close_arr[idx])], dtype=float)
            highs = np.array([float(high_arr[idx - 1]), float(high_arr[idx])], dtype=float)
            lows = np.array([float(low_arr[idx - 1]), float(low_arr[idx])], dtype=float)
        else:
            recent = history_df.iloc[-2:]
            if len(recent) >= 2:
                closes = recent["close"].to_numpy(dtype=float)
                highs = recent["high"].to_numpy(dtype=float)
                lows = recent["low"].to_numpy(dtype=float)
            else:
                closes = None
                close_pos = None
        if closes is not None:
            close_pos = [
                _close_position_in_bar(highs[i], lows[i], closes[i]) for i in range(2)
            ]
        else:
            closes = None
            close_pos = None

        if side == "LONG":
            chosen = _select_interaction_level(
                levels_long,
                bar_close,
                history_df,
                band,
                bar_index=bar_index,
            )
            if chosen is not None and closes is not None and close_pos is not None:
                level_name, level_val = chosen
                if not all(c >= level_val for c in closes) or not all(pos >= 0.60 for pos in close_pos):
                    ny_gate_acceptance_blocked += 1
                    ny_gate_blocked_by_strategy[strat_name] += 1
                    ny_gate_blocked_by_session[session_name] += 1
                    record_filter("NYGateAcceptance")
                    logging.info(
                        "⛔ NYGateAcceptance blocked | session=%s | strategy=%s | side=%s | "
                        "level=%s %.2f | closes=%s | close_pos=%s",
                        session_name,
                        strat_name,
                        side,
                        level_name,
                        level_val,
                        [round(c, 2) for c in closes],
                        [round(p, 2) for p in close_pos],
                    )
                    return False
        else:
            chosen = _select_interaction_level(
                levels_short,
                bar_close,
                history_df,
                band,
                bar_index=bar_index,
            )
            if chosen is not None and closes is not None and close_pos is not None:
                level_name, level_val = chosen
                if not all(c <= level_val for c in closes) or not all(pos <= 0.40 for pos in close_pos):
                    ny_gate_acceptance_blocked += 1
                    ny_gate_blocked_by_strategy[strat_name] += 1
                    ny_gate_blocked_by_session[session_name] += 1
                    record_filter("NYGateAcceptance")
                    logging.info(
                        "⛔ NYGateAcceptance blocked | session=%s | strategy=%s | side=%s | "
                        "level=%s %.2f | closes=%s | close_pos=%s",
                        session_name,
                        strat_name,
                        side,
                        level_name,
                        level_val,
                        [round(c, 2) for c in closes],
                        [round(p, 2) for p in close_pos],
                    )
                    return False

        # Gate C: Opposing liquidity distance
        try:
            tp_dist_val = float(signal.get("tp_dist") or 0.0)
        except Exception:
            tp_dist_val = 0.0
        if tp_dist_val > 0:
            session_high, session_low = _session_high_low(
                history_df,
                current_time,
                session_name,
                bar_index=bar_index,
            )
            if bar_index is not None and 0 <= int(bar_index) < len(full_df):
                idx = int(bar_index)
                sh = swing_high_30_arr[idx]
                sl = swing_low_30_arr[idx]
                swing_high = float(sh) if math.isfinite(sh) else None
                swing_low = float(sl) if math.isfinite(sl) else None
            else:
                swing_high = float(history_df["high"].iloc[-30:].max()) if len(history_df) >= 30 else None
                swing_low = float(history_df["low"].iloc[-30:].min()) if len(history_df) >= 30 else None
            opp_levels: dict[str, float] = {}
            if side == "LONG":
                opp_levels["SwingHigh"] = swing_high
                opp_levels["SessionHigh"] = session_high
                opp_levels["PDH"] = levels_long.get("PDH")
                opp_levels["ORH"] = levels_long.get("ORH")
                opp_levels["VAH"] = levels_long.get("VAH")
            else:
                opp_levels["SwingLow"] = swing_low
                opp_levels["SessionLow"] = session_low
                opp_levels["PDL"] = levels_short.get("PDL")
                opp_levels["ORL"] = levels_short.get("ORL")
                opp_levels["VAL"] = levels_short.get("VAL")

            distances: list[tuple[str, float]] = []
            for name, level in opp_levels.items():
                if level is None or not np.isfinite(level):
                    continue
                if side == "LONG" and level > bar_close:
                    distances.append((name, float(level - bar_close)))
                if side == "SHORT" and level < bar_close:
                    distances.append((name, float(bar_close - level)))
            if distances:
                nearest_name, nearest_dist = min(distances, key=lambda item: item[1])
                if nearest_dist < 0.60 * tp_dist_val:
                    ny_gate_liquidity_blocked += 1
                    ny_gate_blocked_by_strategy[strat_name] += 1
                    ny_gate_blocked_by_session[session_name] += 1
                    record_filter("NYGateLiquidity")
                    logging.info(
                        "⛔ NYGateLiquidity blocked | session=%s | strategy=%s | side=%s | "
                        "level=%s | dist=%.2f | tp=%.2f",
                        session_name,
                        strat_name,
                        side,
                        nearest_name,
                        nearest_dist,
                        tp_dist_val,
                    )
                    return False

        return True

    def continuation_rescue_allowed(
        signal: Optional[dict],
        side: str,
        current_time: dt.datetime,
        bar_close: float,
        history_df: pd.DataFrame,
        bar_index: Optional[int] = None,
        vol_regime_hint: Optional[str] = None,
    ) -> bool:
        if not signal:
            return False
        if continuation_signal_mode != "structure":
            raw_key = parse_continuation_key(signal.get("strategy"))
            key = continuation_allowlist_key(raw_key)
            if continuation_allowlist is not None:
                if not key or key not in continuation_allowlist:
                    record_filter("ContinuationAllowlist")
                    return False
        if continuation_allowed_regimes:
            if vol_regime_hint:
                regime = str(vol_regime_hint)
            else:
                regime, _, _ = volatility_filter.get_regime(history_df)
            if not regime or str(regime).lower() not in continuation_allowed_regimes:
                record_filter("ContinuationRegime")
                return False
        if not continuation_market_confirmed(
            side=side,
            current_time=current_time,
            bar_close=bar_close,
            trend_context=trend_day_series,
            cfg=continuation_confirm_cfg,
        ):
            record_filter("ContinuationConfirm")
            return False
        if not _ny_continuation_gates(
            signal,
            side,
            current_time,
            bar_close,
            history_df,
            bar_index=bar_index,
        ):
            return False
        return True

    de3_v4_cfg_local = CONFIG.get("DE3_V4", {}) if isinstance(CONFIG.get("DE3_V4", {}), dict) else {}
    de3_v4_runtime_cfg_local = (
        de3_v4_cfg_local.get("runtime", {})
        if isinstance(de3_v4_cfg_local.get("runtime", {}), dict)
        else {}
    )
    de3_v4_trade_management_cfg_local = (
        de3_v4_runtime_cfg_local.get("trade_management", {})
        if isinstance(de3_v4_runtime_cfg_local.get("trade_management", {}), dict)
        else {}
    )
    de3_v4_break_even_cfg_local = (
        de3_v4_trade_management_cfg_local.get("break_even", {})
        if isinstance(de3_v4_trade_management_cfg_local.get("break_even", {}), dict)
        else {}
    )
    de3_v4_early_exit_cfg_local = (
        de3_v4_trade_management_cfg_local.get("early_exit", {})
        if isinstance(de3_v4_trade_management_cfg_local.get("early_exit", {}), dict)
        else {}
    )
    de3_v4_tiered_take_cfg_local = (
        de3_v4_trade_management_cfg_local.get("tiered_take_profit", {})
        if isinstance(de3_v4_trade_management_cfg_local.get("tiered_take_profit", {}), dict)
        else {}
    )
    de3_v4_profit_milestone_cfg_local = (
        de3_v4_trade_management_cfg_local.get("profit_milestone_stop", {})
        if isinstance(de3_v4_trade_management_cfg_local.get("profit_milestone_stop", {}), dict)
        else {}
    )
    de3_v4_entry_trade_day_extreme_cfg_local = (
        de3_v4_trade_management_cfg_local.get("entry_trade_day_extreme_stop", {})
        if isinstance(de3_v4_trade_management_cfg_local.get("entry_trade_day_extreme_stop", {}), dict)
        else {}
    )
    de3_v4_entry_trade_day_extreme_admission_cfg_local = (
        de3_v4_trade_management_cfg_local.get("entry_trade_day_extreme_admission_block", {})
        if isinstance(
            de3_v4_trade_management_cfg_local.get("entry_trade_day_extreme_admission_block", {}),
            dict,
        )
        else {}
    )
    de3_v4_entry_trade_day_extreme_size_cfg_local = (
        de3_v4_trade_management_cfg_local.get("entry_trade_day_extreme_size_adjustment", {})
        if isinstance(
            de3_v4_trade_management_cfg_local.get("entry_trade_day_extreme_size_adjustment", {}),
            dict,
        )
        else {}
    )
    de3_v4_entry_trade_day_extreme_early_exit_cfg_local = (
        de3_v4_trade_management_cfg_local.get("entry_trade_day_extreme_early_exit", {})
        if isinstance(
            de3_v4_trade_management_cfg_local.get("entry_trade_day_extreme_early_exit", {}),
            dict,
        )
        else {}
    )
    de3_break_even_armed_trade_count = 0
    de3_break_even_stop_update_count = 0
    de3_early_exit_close_count = 0
    de3_tiered_take_fill_count = 0
    de3_tiered_take_closed_contract_count = 0
    de3_profit_milestone_profile_trade_count = 0
    de3_profit_milestone_reached_count = 0
    de3_entry_trade_day_extreme_profile_trade_count = 0
    de3_entry_trade_day_extreme_reached_count = 0
    de3_entry_trade_day_extreme_admission_checked = 0
    de3_entry_trade_day_extreme_admission_blocked = 0
    de3_entry_trade_day_extreme_admission_profile_hits: Counter[str] = Counter()
    de3_entry_trade_day_extreme_size_adjustment_checked = 0
    de3_entry_trade_day_extreme_size_adjustment_applied = 0
    de3_entry_trade_day_extreme_size_adjustment_profile_hits: Counter[str] = Counter()
    de3_entry_trade_day_extreme_early_exit_profile_trade_count = 0
    de3_entry_trade_day_extreme_early_exit_profile_hits: Counter[str] = Counter()
    de3_entry_trade_day_extreme_early_exit_close_profile_hits: Counter[str] = Counter()

    def _is_de3_v4_trade_management_payload(payload: Optional[dict]) -> bool:
        if not isinstance(payload, dict):
            return False
        strategy_name = str(payload.get("strategy", "") or "")
        if strategy_name and not strategy_name.startswith("DynamicEngine3"):
            return False
        if str(payload.get("de3_version", "") or "").strip().lower() == "v4":
            return True
        if payload.get("de3_v4_selected_variant_id"):
            return True
        if payload.get("de3_v4_selected_lane"):
            return True
        return False

    def _align_stop_price_to_tick(price: float, side: str) -> float:
        if not math.isfinite(price):
            return price
        if TICK_SIZE <= 0:
            return float(price)
        scaled = float(price) / float(TICK_SIZE)
        if str(side).upper() == "LONG":
            return round(math.floor(scaled + 1e-9) * float(TICK_SIZE), 10)
        return round(math.ceil(scaled - 1e-9) * float(TICK_SIZE), 10)

    def _resolve_de3_entry_trade_day_extreme_context(
        payload: Optional[dict],
        *,
        entry_price: float,
        tp_dist: float,
        entry_trade_day_high: float,
        entry_trade_day_low: float,
    ) -> dict:
        result = {
            "variant_id": "",
            "entry_trade_day_high": (
                float(entry_trade_day_high) if math.isfinite(entry_trade_day_high) else None
            ),
            "entry_trade_day_low": (
                float(entry_trade_day_low) if math.isfinite(entry_trade_day_low) else None
            ),
            "extreme_price": None,
            "target_beyond_trade_day_extreme": False,
            "progress_pct": None,
        }
        if not isinstance(payload, dict):
            return result
        side_name = str(payload.get("side", "") or "").upper()
        if (
            side_name not in {"LONG", "SHORT"}
            or not math.isfinite(entry_price)
            or not math.isfinite(tp_dist)
            or tp_dist <= 0.0
        ):
            return result
        variant_id = _de3_variant_id_from_payload(payload)
        result["variant_id"] = variant_id
        extreme_price = (
            float(entry_trade_day_high)
            if side_name == "LONG"
            else float(entry_trade_day_low)
        )
        if not math.isfinite(extreme_price):
            return result
        result["extreme_price"] = float(extreme_price)
        take_price = (
            float(entry_price + tp_dist)
            if side_name == "LONG"
            else float(entry_price - tp_dist)
        )
        target_beyond = (
            take_price > extreme_price + 1e-12
            if side_name == "LONG"
            else take_price < extreme_price - 1e-12
        )
        result["target_beyond_trade_day_extreme"] = bool(target_beyond)
        progress_points = (
            float(extreme_price - entry_price)
            if side_name == "LONG"
            else float(entry_price - extreme_price)
        )
        if math.isfinite(progress_points):
            result["progress_pct"] = float(
                min(1.0, max(0.0, progress_points / float(tp_dist)))
            )
        return result

    def _resolve_de3_entry_trade_day_extreme_profile(
        payload: Optional[dict],
        *,
        entry_price: float,
        tp_dist: float,
        entry_trade_day_high: float,
        entry_trade_day_low: float,
    ) -> dict:
        result = {
            "active": False,
            "profile_name": "",
            "variant_id": "",
            "entry_trade_day_high": (
                float(entry_trade_day_high) if math.isfinite(entry_trade_day_high) else None
            ),
            "entry_trade_day_low": (
                float(entry_trade_day_low) if math.isfinite(entry_trade_day_low) else None
            ),
            "extreme_price": None,
            "target_beyond_trade_day_extreme": False,
            "progress_pct": None,
            "force_break_even_on_reach": False,
            "post_reach_trail_pct": 0.0,
        }
        result.update(
            _resolve_de3_entry_trade_day_extreme_context(
                payload,
                entry_price=entry_price,
                tp_dist=tp_dist,
                entry_trade_day_high=entry_trade_day_high,
                entry_trade_day_low=entry_trade_day_low,
            )
        )
        if not bool(de3_v4_entry_trade_day_extreme_cfg_local.get("enabled", False)):
            return result
        if not _is_de3_v4_trade_management_payload(payload):
            return result
        profiles = de3_v4_entry_trade_day_extreme_cfg_local.get("profiles", [])
        if not isinstance(profiles, (list, tuple)):
            return result
        side_name = str(payload.get("side", "") or "").upper()
        variant_id = str(result.get("variant_id", "") or "")
        target_beyond = bool(result.get("target_beyond_trade_day_extreme", False))
        for index, profile in enumerate(profiles):
            if not isinstance(profile, dict) or not bool(profile.get("enabled", True)):
                continue
            if not _match_de3_rule_values(profile.get("apply_variants", []), variant_id):
                continue
            if not _match_de3_rule_values(profile.get("apply_sides", []), side_name):
                continue
            if (
                bool(profile.get("require_target_beyond_trade_day_extreme", False))
                and not target_beyond
            ):
                continue
            min_progress_pct = _coerce_float(
                profile.get("min_entry_trade_day_extreme_progress_pct"),
                float("nan"),
            )
            progress_pct = _coerce_float(result.get("progress_pct"), float("nan"))
            if math.isfinite(min_progress_pct) and (
                not math.isfinite(progress_pct)
                or progress_pct + 1e-12 < float(min_progress_pct)
            ):
                continue
            max_progress_pct = _coerce_float(
                profile.get("max_entry_trade_day_extreme_progress_pct"),
                float("nan"),
            )
            if math.isfinite(max_progress_pct) and (
                not math.isfinite(progress_pct)
                or progress_pct - 1e-12 > float(max_progress_pct)
            ):
                continue
            trail_default = _coerce_float(
                de3_v4_break_even_cfg_local.get("post_activation_trail_pct"),
                _coerce_float(de3_v4_break_even_cfg_local.get("trail_pct"), 0.0),
            )
            result["active"] = True
            result["profile_name"] = str(
                profile.get("name")
                or f"entry_trade_day_extreme_profile_{index + 1}"
            )
            result["force_break_even_on_reach"] = bool(
                profile.get("force_break_even_on_reach", False)
            )
            result["post_reach_trail_pct"] = float(
                max(
                    0.0,
                    _coerce_float(profile.get("post_reach_trail_pct"), trail_default),
                )
            )
            return result
        return result

    def _resolve_de3_entry_trade_day_extreme_admission_block(
        payload: Optional[dict],
        *,
        entry_price: float,
        tp_dist: float,
        entry_trade_day_high: float,
        entry_trade_day_low: float,
    ) -> dict:
        result = {
            "active": False,
            "profile_name": "",
            "variant_id": "",
            "entry_trade_day_high": (
                float(entry_trade_day_high) if math.isfinite(entry_trade_day_high) else None
            ),
            "entry_trade_day_low": (
                float(entry_trade_day_low) if math.isfinite(entry_trade_day_low) else None
            ),
            "extreme_price": None,
            "target_beyond_trade_day_extreme": False,
            "progress_pct": None,
        }
        if not bool(de3_v4_entry_trade_day_extreme_admission_cfg_local.get("enabled", False)):
            return result
        result.update(
            _resolve_de3_entry_trade_day_extreme_context(
                payload,
                entry_price=entry_price,
                tp_dist=tp_dist,
                entry_trade_day_high=entry_trade_day_high,
                entry_trade_day_low=entry_trade_day_low,
            )
        )
        if not _is_de3_v4_trade_management_payload(payload):
            return result
        profiles = de3_v4_entry_trade_day_extreme_admission_cfg_local.get("profiles", [])
        if not isinstance(profiles, (list, tuple)):
            return result
        side_name = str((payload or {}).get("side", "") or "").upper()
        variant_id = str(result.get("variant_id", "") or "")
        target_beyond = bool(result.get("target_beyond_trade_day_extreme", False))
        for index, profile in enumerate(profiles):
            if not isinstance(profile, dict) or not bool(profile.get("enabled", True)):
                continue
            if not _match_de3_rule_values(profile.get("apply_variants", []), variant_id):
                continue
            if not _match_de3_rule_values(profile.get("apply_sides", []), side_name):
                continue
            if (
                bool(profile.get("require_target_beyond_trade_day_extreme", False))
                and not target_beyond
            ):
                continue
            min_progress_pct = _coerce_float(
                profile.get("min_entry_trade_day_extreme_progress_pct"),
                float("nan"),
            )
            progress_pct = _coerce_float(result.get("progress_pct"), float("nan"))
            if math.isfinite(min_progress_pct) and (
                not math.isfinite(progress_pct)
                or progress_pct + 1e-12 < float(min_progress_pct)
            ):
                continue
            max_progress_pct = _coerce_float(
                profile.get("max_entry_trade_day_extreme_progress_pct"),
                float("nan"),
            )
            if math.isfinite(max_progress_pct) and (
                not math.isfinite(progress_pct)
                or progress_pct - 1e-12 > float(max_progress_pct)
            ):
                continue
            result["active"] = True
            result["profile_name"] = str(
                profile.get("name")
                or f"entry_trade_day_extreme_admission_block_{index + 1}"
            )
            return result
        return result

    def _resolve_de3_entry_trade_day_extreme_size_adjustment(
        payload: Optional[dict],
        *,
        entry_price: float,
        tp_dist: float,
        entry_trade_day_high: float,
        entry_trade_day_low: float,
    ) -> dict:
        result = {
            "active": False,
            "profile_name": "",
            "variant_id": "",
            "entry_trade_day_high": (
                float(entry_trade_day_high) if math.isfinite(entry_trade_day_high) else None
            ),
            "entry_trade_day_low": (
                float(entry_trade_day_low) if math.isfinite(entry_trade_day_low) else None
            ),
            "extreme_price": None,
            "target_beyond_trade_day_extreme": False,
            "progress_pct": None,
            "size_multiplier": 1.0,
            "min_contracts": 1,
        }
        if not bool(de3_v4_entry_trade_day_extreme_size_cfg_local.get("enabled", False)):
            return result
        result.update(
            _resolve_de3_entry_trade_day_extreme_context(
                payload,
                entry_price=entry_price,
                tp_dist=tp_dist,
                entry_trade_day_high=entry_trade_day_high,
                entry_trade_day_low=entry_trade_day_low,
            )
        )
        if not _is_de3_v4_trade_management_payload(payload):
            return result
        profiles = de3_v4_entry_trade_day_extreme_size_cfg_local.get("profiles", [])
        if not isinstance(profiles, (list, tuple)):
            return result
        side_name = str((payload or {}).get("side", "") or "").upper()
        variant_id = str(result.get("variant_id", "") or "")
        target_beyond = bool(result.get("target_beyond_trade_day_extreme", False))
        progress_pct = _coerce_float(result.get("progress_pct"), float("nan"))
        for index, profile in enumerate(profiles):
            if not isinstance(profile, dict) or not bool(profile.get("enabled", True)):
                continue
            if not _match_de3_rule_values(profile.get("apply_variants", []), variant_id):
                continue
            if not _match_de3_rule_values(profile.get("apply_sides", []), side_name):
                continue
            if (
                bool(profile.get("require_target_beyond_trade_day_extreme", False))
                and not target_beyond
            ):
                continue
            min_progress_pct = _coerce_float(
                profile.get("min_entry_trade_day_extreme_progress_pct"),
                float("nan"),
            )
            if math.isfinite(min_progress_pct) and (
                not math.isfinite(progress_pct)
                or progress_pct + 1e-12 < float(min_progress_pct)
            ):
                continue
            max_progress_pct = _coerce_float(
                profile.get("max_entry_trade_day_extreme_progress_pct"),
                float("nan"),
            )
            if math.isfinite(max_progress_pct) and (
                not math.isfinite(progress_pct)
                or progress_pct - 1e-12 > float(max_progress_pct)
            ):
                continue
            size_multiplier = _coerce_float(profile.get("size_multiplier"), math.nan)
            if not math.isfinite(size_multiplier) or size_multiplier <= 0.0:
                continue
            result["active"] = True
            result["profile_name"] = str(
                profile.get("name")
                or f"entry_trade_day_extreme_size_adjustment_{index + 1}"
            )
            result["size_multiplier"] = float(min(1.0, max(0.0, size_multiplier)))
            result["min_contracts"] = int(
                max(1, _coerce_int(profile.get("min_contracts"), 1))
            )
            return result
        return result

    def _match_de3_profile_tp_dist(profile: Optional[dict], tp_dist: float) -> bool:
        if not isinstance(profile, dict):
            return True
        tp_dist = round_points_to_tick(max(0.0, _coerce_float(tp_dist, 0.0)))
        apply_tp_dists = profile.get("apply_tp_dists", [])
        if isinstance(apply_tp_dists, (list, tuple, set)):
            normalized: list[float] = []
            for item in apply_tp_dists:
                item_value = _coerce_float(item, math.nan)
                if math.isfinite(item_value):
                    normalized.append(round_points_to_tick(max(0.0, item_value)))
            if normalized and not any(abs(tp_dist - value) <= 1e-9 for value in normalized):
                return False
        min_tp_dist = _coerce_float(profile.get("min_tp_dist"), math.nan)
        if math.isfinite(min_tp_dist) and tp_dist + 1e-9 < round_points_to_tick(max(0.0, min_tp_dist)):
            return False
        max_tp_dist = _coerce_float(profile.get("max_tp_dist"), math.nan)
        if math.isfinite(max_tp_dist) and tp_dist - 1e-9 > round_points_to_tick(max(0.0, max_tp_dist)):
            return False
        return True

    def _resolve_de3_entry_trade_day_extreme_early_exit_profile(
        payload: Optional[dict],
        *,
        entry_price: float,
        tp_dist: float,
        entry_trade_day_high: float,
        entry_trade_day_low: float,
    ) -> dict:
        result = {
            "active": False,
            "profile_name": "",
            "variant_id": "",
            "entry_trade_day_high": (
                float(entry_trade_day_high) if math.isfinite(entry_trade_day_high) else None
            ),
            "entry_trade_day_low": (
                float(entry_trade_day_low) if math.isfinite(entry_trade_day_low) else None
            ),
            "extreme_price": None,
            "target_beyond_trade_day_extreme": False,
            "progress_pct": None,
            "min_progress_by_bars": None,
            "min_progress_pct": None,
            "max_profit_crosses": None,
        }
        if not bool(de3_v4_entry_trade_day_extreme_early_exit_cfg_local.get("enabled", False)):
            return result
        result.update(
            _resolve_de3_entry_trade_day_extreme_context(
                payload,
                entry_price=entry_price,
                tp_dist=tp_dist,
                entry_trade_day_high=entry_trade_day_high,
                entry_trade_day_low=entry_trade_day_low,
            )
        )
        if not _is_de3_v4_trade_management_payload(payload):
            return result
        profiles = de3_v4_entry_trade_day_extreme_early_exit_cfg_local.get("profiles", [])
        if not isinstance(profiles, (list, tuple)):
            return result
        side_name = str((payload or {}).get("side", "") or "").upper()
        variant_id = str(result.get("variant_id", "") or "")
        target_beyond = bool(result.get("target_beyond_trade_day_extreme", False))
        progress_pct = _coerce_float(result.get("progress_pct"), float("nan"))
        for index, profile in enumerate(profiles):
            if not isinstance(profile, dict) or not bool(profile.get("enabled", True)):
                continue
            if not _match_de3_rule_values(profile.get("apply_variants", []), variant_id):
                continue
            if not _match_de3_rule_values(profile.get("apply_sides", []), side_name):
                continue
            if (
                bool(profile.get("require_target_beyond_trade_day_extreme", False))
                and not target_beyond
            ):
                continue
            min_entry_progress_pct = _coerce_float(
                profile.get("min_entry_trade_day_extreme_progress_pct"),
                float("nan"),
            )
            if math.isfinite(min_entry_progress_pct) and (
                not math.isfinite(progress_pct)
                or progress_pct + 1e-12 < float(min_entry_progress_pct)
            ):
                continue
            max_entry_progress_pct = _coerce_float(
                profile.get("max_entry_trade_day_extreme_progress_pct"),
                float("nan"),
            )
            if math.isfinite(max_entry_progress_pct) and (
                not math.isfinite(progress_pct)
                or progress_pct - 1e-12 > float(max_entry_progress_pct)
            ):
                continue
            min_progress_by_bars = int(
                max(0, _coerce_int(profile.get("min_progress_by_bars"), 0))
            )
            min_progress_pct = _coerce_float(profile.get("min_progress_pct"), float("nan"))
            max_profit_crosses_raw = profile.get("max_profit_crosses", None)
            max_profit_crosses = None
            if max_profit_crosses_raw is not None:
                max_profit_crosses = int(max(0, _coerce_int(max_profit_crosses_raw, 0)))
            has_progress_rule = (
                min_progress_by_bars > 0 and math.isfinite(min_progress_pct)
            )
            has_cross_rule = max_profit_crosses is not None
            if not has_progress_rule and not has_cross_rule:
                continue
            result["active"] = True
            result["profile_name"] = str(
                profile.get("name")
                or f"entry_trade_day_extreme_early_exit_{index + 1}"
            )
            if has_progress_rule:
                result["min_progress_by_bars"] = int(min_progress_by_bars)
                result["min_progress_pct"] = float(min_progress_pct)
            if has_cross_rule:
                result["max_profit_crosses"] = int(max_profit_crosses)
            return result
        return result

    def _resolve_de3_profit_milestone_profile(
        payload: Optional[dict],
        *,
        entry_price: float,
        tp_dist: float,
    ) -> dict:
        result = {
            "active": False,
            "profile_name": "",
            "milestone_price": None,
            "trigger_pct": 0.0,
            "force_break_even_on_reach": False,
            "post_reach_trail_pct": 0.0,
        }
        if not bool(de3_v4_profit_milestone_cfg_local.get("enabled", False)):
            return result
        if not _is_de3_v4_trade_management_payload(payload):
            return result
        side_name = str((payload or {}).get("side", "") or "").upper()
        entry_price = _coerce_float(entry_price, math.nan)
        tp_dist = round_points_to_tick(max(0.0, _coerce_float(tp_dist, 0.0)))
        if side_name not in {"LONG", "SHORT"} or not math.isfinite(entry_price) or tp_dist <= 0.0:
            return result
        profiles = de3_v4_profit_milestone_cfg_local.get("profiles", [])
        if not isinstance(profiles, (list, tuple)):
            return result
        variant_id = _de3_variant_id_from_payload(payload)
        trail_default = _coerce_float(
            de3_v4_break_even_cfg_local.get("post_activation_trail_pct"),
            _coerce_float(de3_v4_break_even_cfg_local.get("trail_pct"), 0.0),
        )
        for index, profile in enumerate(profiles):
            if not isinstance(profile, dict) or not bool(profile.get("enabled", True)):
                continue
            if not _match_de3_rule_values(profile.get("apply_variants", []), variant_id):
                continue
            if not _match_de3_rule_values(profile.get("apply_sides", []), side_name):
                continue
            if not _match_de3_profile_tp_dist(profile, tp_dist):
                continue
            trigger_pct = _coerce_float(profile.get("trigger_pct"), math.nan)
            if not math.isfinite(trigger_pct) or trigger_pct <= 0.0 or trigger_pct >= 1.0:
                continue
            trigger_points = round_points_to_tick(max(float(TICK_SIZE), tp_dist * float(trigger_pct)))
            milestone_price = (
                float(entry_price + trigger_points)
                if side_name == "LONG"
                else float(entry_price - trigger_points)
            )
            result["active"] = True
            result["profile_name"] = str(
                profile.get("name") or f"profit_milestone_profile_{index + 1}"
            )
            result["milestone_price"] = float(milestone_price)
            result["trigger_pct"] = float(trigger_pct)
            result["force_break_even_on_reach"] = bool(
                profile.get("force_break_even_on_reach", False)
            )
            result["post_reach_trail_pct"] = float(
                max(
                    0.0,
                    _coerce_float(profile.get("post_reach_trail_pct"), trail_default),
                )
            )
            return result
        return result

    def _update_de3_profit_milestone_state(
        trade: Optional[dict],
        *,
        bar_high: float,
        bar_low: float,
        bar_index: Optional[int] = None,
    ) -> None:
        nonlocal de3_profit_milestone_reached_count
        if not isinstance(trade, dict):
            return
        if not bool(trade.get("de3_profit_milestone_profile_active", False)):
            return
        if bool(trade.get("de3_profit_milestone_reached", False)):
            return
        side_name = str(trade.get("side", "") or "").upper()
        milestone_price = _coerce_float(trade.get("de3_profit_milestone_price"), math.nan)
        if side_name not in {"LONG", "SHORT"} or not math.isfinite(milestone_price):
            return
        bar_high_val = _coerce_float(bar_high, math.nan)
        bar_low_val = _coerce_float(bar_low, math.nan)
        reached = (
            math.isfinite(bar_high_val) and bar_high_val >= milestone_price - 1e-12
            if side_name == "LONG"
            else math.isfinite(bar_low_val) and bar_low_val <= milestone_price + 1e-12
        )
        if not reached:
            return
        trade["de3_profit_milestone_reached"] = True
        trade["de3_profit_milestone_reached_bar_index"] = _safe_idx(bar_index)
        trade["de3_profit_milestone_reached_mfe_points"] = float(
            max(0.0, _coerce_float(trade.get("mfe_points"), 0.0))
        )
        de3_profit_milestone_reached_count += 1

    def _update_de3_entry_trade_day_extreme_state(
        trade: Optional[dict],
        *,
        bar_high: float,
        bar_low: float,
        bar_index: Optional[int] = None,
    ) -> None:
        nonlocal de3_entry_trade_day_extreme_reached_count
        if not isinstance(trade, dict):
            return
        if not bool(trade.get("de3_entry_trade_day_extreme_profile_active", False)):
            return
        if bool(trade.get("de3_entry_trade_day_extreme_reached", False)):
            return
        side_name = str(trade.get("side", "") or "").upper()
        extreme_price = _coerce_float(
            trade.get("de3_entry_trade_day_extreme_price"),
            math.nan,
        )
        if side_name not in {"LONG", "SHORT"} or not math.isfinite(extreme_price):
            return
        bar_high_val = _coerce_float(bar_high, math.nan)
        bar_low_val = _coerce_float(bar_low, math.nan)
        reached = (
            math.isfinite(bar_high_val) and bar_high_val >= extreme_price - 1e-12
            if side_name == "LONG"
            else math.isfinite(bar_low_val) and bar_low_val <= extreme_price + 1e-12
        )
        if not reached:
            return
        trade["de3_entry_trade_day_extreme_reached"] = True
        trade["de3_entry_trade_day_extreme_reached_bar_index"] = _safe_idx(bar_index)
        trade["de3_entry_trade_day_extreme_reached_mfe_points"] = float(
            max(
                0.0,
                _coerce_float(trade.get("mfe_points"), 0.0),
            )
        )
        de3_entry_trade_day_extreme_reached_count += 1

    def _current_de3_break_even_trail_pct(trade: Optional[dict]) -> float:
        if not isinstance(trade, dict):
            return 0.0
        trail_pct = max(0.0, _coerce_float(trade.get("de3_break_even_trail_pct"), 0.0))
        post_activation_trail_pct = max(
            0.0,
            _coerce_float(trade.get("de3_break_even_post_activation_trail_pct"), 0.0),
        )
        post_partial_trail_pct = max(
            0.0,
            _coerce_float(trade.get("de3_break_even_post_partial_trail_pct"), 0.0),
        )
        post_entry_trade_day_extreme_trail_pct = max(
            0.0,
            _coerce_float(
                trade.get("de3_break_even_post_entry_trade_day_extreme_trail_pct"),
                0.0,
            ),
        )
        post_profit_milestone_trail_pct = max(
            0.0,
            _coerce_float(
                trade.get("de3_break_even_post_profit_milestone_trail_pct"),
                0.0,
            ),
        )
        if bool(trade.get("de3_break_even_applied", False)):
            trail_pct = max(trail_pct, post_activation_trail_pct)
        if bool(trade.get("de3_tiered_take_filled", False)):
            trail_pct = max(trail_pct, post_partial_trail_pct)
        if bool(trade.get("de3_profit_milestone_reached", False)):
            trail_pct = max(trail_pct, post_profit_milestone_trail_pct)
        if bool(trade.get("de3_entry_trade_day_extreme_reached", False)):
            trail_pct = max(trail_pct, post_entry_trade_day_extreme_trail_pct)
        return float(trail_pct)

    def _build_de3_break_even_stop_candidate(trade: Optional[dict]) -> Optional[dict]:
        if not isinstance(trade, dict):
            return None
        if not bool(trade.get("de3_break_even_enabled", False)):
            return None
        side_name = str(trade.get("side", "") or "").upper()
        entry_price = _coerce_float(trade.get("entry_price"), math.nan)
        current_stop_price = _coerce_float(trade.get("current_stop_price"), math.nan)
        tp_dist = _coerce_float(trade.get("tp_dist", MIN_TP), MIN_TP)
        if (
            side_name not in {"LONG", "SHORT"}
            or not math.isfinite(entry_price)
            or not math.isfinite(current_stop_price)
            or tp_dist <= 0.0
        ):
            return None
        trigger_pct = max(0.0, _coerce_float(trade.get("de3_break_even_trigger_pct"), 0.0))
        mfe_points = max(0.0, _coerce_float(trade.get("mfe_points"), 0.0))
        trigger_points = max(0.0, tp_dist * trigger_pct)
        allow_after_partial = bool(
            trade.get("de3_tiered_take_filled", False)
            and trade.get("de3_tiered_take_arm_break_even_after_fill", False)
        )
        allow_after_profit_milestone = bool(
            trade.get("de3_profit_milestone_reached", False)
            and trade.get("de3_profit_milestone_force_break_even", False)
        )
        allow_after_entry_trade_day_extreme = bool(
            trade.get("de3_entry_trade_day_extreme_reached", False)
            and trade.get("de3_entry_trade_day_extreme_force_break_even", False)
        )
        if (
            not allow_after_partial
            and not allow_after_profit_milestone
            and not allow_after_entry_trade_day_extreme
            and trigger_points > 0.0
            and mfe_points + 1e-9 < trigger_points
        ):
            return None
        trail_pct = _current_de3_break_even_trail_pct(trade)
        buffer_ticks = max(0, _coerce_int(trade.get("de3_break_even_buffer_ticks"), 0))
        buffer_points = float(buffer_ticks) * float(TICK_SIZE)
        locked_points = max(buffer_points, mfe_points * trail_pct if trail_pct > 0.0 else 0.0)
        candidate_stop_price = (
            entry_price + locked_points if side_name == "LONG" else entry_price - locked_points
        )
        candidate_stop_price = _align_stop_price_to_tick(candidate_stop_price, side_name)
        improved = (
            candidate_stop_price > current_stop_price + 1e-12
            if side_name == "LONG"
            else candidate_stop_price < current_stop_price - 1e-12
        )
        if not improved:
            return None
        return {
            "side_name": side_name,
            "entry_price": float(entry_price),
            "current_stop_price": float(current_stop_price),
            "candidate_stop_price": float(candidate_stop_price),
            "mfe_points": float(mfe_points),
            "trigger_points": float(trigger_points),
            "trail_pct": float(trail_pct),
            "locked_points": float(locked_points),
        }

    def _apply_de3_break_even_stop_update(
        trade: Optional[dict],
        new_stop_price: float,
        *,
        from_pending: bool,
        bar_index: Optional[int] = None,
    ) -> bool:
        nonlocal de3_break_even_armed_trade_count, de3_break_even_stop_update_count
        if not isinstance(trade, dict):
            return False
        side_name = str(trade.get("side", "") or "").upper()
        current_stop_price = _coerce_float(trade.get("current_stop_price"), math.nan)
        if side_name not in {"LONG", "SHORT"} or not math.isfinite(current_stop_price):
            return False
        target_stop_price = _align_stop_price_to_tick(
            _coerce_float(new_stop_price, math.nan),
            side_name,
        )
        if not math.isfinite(target_stop_price):
            return False
        entry_price = _coerce_float(trade.get("entry_price"), math.nan)
        tp_dist = _coerce_float(trade.get("tp_dist", MIN_TP), MIN_TP)
        if math.isfinite(entry_price) and tp_dist > 0.0:
            take_price = entry_price + tp_dist if side_name == "LONG" else entry_price - tp_dist
            if side_name == "LONG":
                max_stop_price = _align_stop_price_to_tick(take_price - TICK_SIZE, side_name)
                if math.isfinite(max_stop_price):
                    target_stop_price = min(target_stop_price, max_stop_price)
            else:
                min_stop_price = _align_stop_price_to_tick(take_price + TICK_SIZE, side_name)
                if math.isfinite(min_stop_price):
                    target_stop_price = max(target_stop_price, min_stop_price)
        improved = (
            target_stop_price > current_stop_price + 1e-12
            if side_name == "LONG"
            else target_stop_price < current_stop_price - 1e-12
        )
        if not improved:
            if from_pending:
                trade["de3_break_even_pending_stop_price"] = None
                trade["de3_break_even_pending_from_bar_index"] = None
            return False
        already_applied = bool(trade.get("de3_break_even_applied", False))
        trade["current_stop_price"] = float(target_stop_price)
        trade["de3_effective_stop_price"] = float(target_stop_price)
        trade["de3_break_even_last_stop_price"] = float(target_stop_price)
        trade["de3_break_even_armed"] = True
        trade["de3_break_even_applied"] = True
        trade["de3_break_even_move_count"] = int(
            _coerce_int(trade.get("de3_break_even_move_count"), 0)
        ) + 1
        trade["de3_break_even_last_update_bar_index"] = _safe_idx(bar_index)
        if trade.get("de3_break_even_first_applied_bar_index") is None:
            trade["de3_break_even_first_applied_bar_index"] = _safe_idx(bar_index)
        if from_pending:
            trade["de3_break_even_pending_stop_price"] = None
            trade["de3_break_even_pending_from_bar_index"] = None
        if not already_applied:
            de3_break_even_armed_trade_count += 1
        de3_break_even_stop_update_count += 1
        return True

    def _apply_pending_de3_break_even_stop_update(
        trade: Optional[dict],
        *,
        bar_index: Optional[int] = None,
    ) -> None:
        if not isinstance(trade, dict):
            return
        pending_stop_price = _coerce_float(
            trade.get("de3_break_even_pending_stop_price"),
            math.nan,
        )
        if not math.isfinite(pending_stop_price):
            return
        _apply_de3_break_even_stop_update(
            trade,
            pending_stop_price,
            from_pending=True,
            bar_index=bar_index,
        )

    def _stage_de3_break_even_stop_update(
        trade: Optional[dict],
        *,
        bar_index: Optional[int] = None,
    ) -> None:
        if not isinstance(trade, dict):
            return
        candidate = _build_de3_break_even_stop_candidate(trade)
        if not isinstance(candidate, dict):
            return
        side_name = str(candidate.get("side_name", "") or "").upper()
        entry_price = _coerce_float(candidate.get("entry_price"), math.nan)
        candidate_stop_price = _coerce_float(candidate.get("candidate_stop_price"), math.nan)
        trigger_points = max(0.0, _coerce_float(candidate.get("trigger_points"), 0.0))
        mfe_points = max(0.0, _coerce_float(candidate.get("mfe_points"), 0.0))
        locked_points = max(0.0, _coerce_float(candidate.get("locked_points"), 0.0))
        activate_on_next_bar = bool(trade.get("de3_break_even_activate_on_next_bar", True))
        trade["de3_break_even_armed"] = True
        trade["de3_break_even_trigger_price"] = (
            entry_price + trigger_points if side_name == "LONG" else entry_price - trigger_points
        )
        if trade.get("de3_break_even_trigger_bar_index") is None:
            trade["de3_break_even_trigger_bar_index"] = _safe_idx(bar_index)
        prev_trigger_mfe = _coerce_float(trade.get("de3_break_even_trigger_mfe_points"), 0.0)
        trade["de3_break_even_trigger_mfe_points"] = float(max(prev_trigger_mfe, mfe_points))
        trade["de3_break_even_locked_points"] = float(locked_points)
        if activate_on_next_bar:
            pending_stop_price = _coerce_float(
                trade.get("de3_break_even_pending_stop_price"),
                math.nan,
            )
            better_pending = (
                not math.isfinite(pending_stop_price)
                or (
                    candidate_stop_price > pending_stop_price + 1e-12
                    if side_name == "LONG"
                    else candidate_stop_price < pending_stop_price - 1e-12
                )
            )
            if better_pending:
                trade["de3_break_even_pending_stop_price"] = float(candidate_stop_price)
                trade["de3_break_even_pending_from_bar_index"] = _safe_idx(bar_index)
            return
        _apply_de3_break_even_stop_update(
            trade,
            candidate_stop_price,
            from_pending=False,
            bar_index=bar_index,
        )

    def _coerce_de3_tiered_take_close_size(trade: Optional[dict]) -> int:
        if not isinstance(trade, dict):
            return 0
        if not bool(trade.get("de3_tiered_take_enabled", False)):
            return 0
        if bool(trade.get("de3_tiered_take_filled", False)):
            return 0
        current_size = max(0, _coerce_int(trade.get("size"), 0))
        original_size = max(current_size, _coerce_int(trade.get("original_size"), current_size))
        min_entry_contracts = max(2, _coerce_int(trade.get("de3_tiered_take_min_entry_contracts"), 2))
        min_remaining_contracts = max(
            1,
            _coerce_int(trade.get("de3_tiered_take_min_remaining_contracts"), 1),
        )
        if original_size < min_entry_contracts or current_size <= min_remaining_contracts:
            return 0
        close_fraction = max(0.0, _coerce_float(trade.get("de3_tiered_take_close_fraction"), 0.0))
        close_size = int(math.floor(float(original_size) * float(close_fraction)))
        if close_size <= 0 and close_fraction > 0.0:
            close_size = 1
        max_close_size = max(0, current_size - min_remaining_contracts)
        if max_close_size <= 0:
            return 0
        close_size = min(max_close_size, close_size)
        return max(0, int(close_size))

    def _stage_de3_break_even_after_partial_fill(
        trade: Optional[dict],
        *,
        bar_index: Optional[int] = None,
    ) -> None:
        if not isinstance(trade, dict):
            return
        candidate = _build_de3_break_even_stop_candidate(trade)
        if not isinstance(candidate, dict):
            return
        side_name = str(candidate.get("side_name", "") or "").upper()
        candidate_stop_price = _coerce_float(candidate.get("candidate_stop_price"), math.nan)
        if not math.isfinite(candidate_stop_price):
            return
        activate_on_next_bar = bool(trade.get("de3_break_even_activate_on_next_bar", True))
        if activate_on_next_bar:
            pending_stop_price = _coerce_float(
                trade.get("de3_break_even_pending_stop_price"),
                math.nan,
            )
            better_pending = (
                not math.isfinite(pending_stop_price)
                or (
                    candidate_stop_price > pending_stop_price + 1e-12
                    if side_name == "LONG"
                    else candidate_stop_price < pending_stop_price - 1e-12
                )
            )
            if better_pending:
                trade["de3_break_even_pending_stop_price"] = float(candidate_stop_price)
                trade["de3_break_even_pending_from_bar_index"] = _safe_idx(bar_index)
            return
        _apply_de3_break_even_stop_update(
            trade,
            candidate_stop_price,
            from_pending=False,
            bar_index=bar_index,
        )

    def _apply_de3_tiered_take_fill(
        trade: Optional[dict],
        fill_price: float,
        fill_time,
        *,
        fill_reason: str,
        bar_index: Optional[int] = None,
    ) -> bool:
        nonlocal de3_tiered_take_fill_count, de3_tiered_take_closed_contract_count
        if not isinstance(trade, dict):
            return False
        close_size = _coerce_de3_tiered_take_close_size(trade)
        if close_size <= 0:
            return False
        side_name = str(trade.get("side", "") or "").upper()
        entry_price = _coerce_float(trade.get("entry_price"), math.nan)
        fill_price = _coerce_float(fill_price, math.nan)
        current_size = max(0, _coerce_int(trade.get("size"), 0))
        if side_name not in {"LONG", "SHORT"} or not math.isfinite(entry_price) or not math.isfinite(fill_price):
            return False
        if current_size <= close_size:
            return False
        pnl_points = compute_pnl_points(side_name, entry_price, fill_price)
        pnl_dollars = float(pnl_points) * float(POINT_VALUE) * float(close_size)
        fee_paid = float(FEE_PER_CONTRACT_RT) * float(close_size)
        pnl_net = float(pnl_dollars - fee_paid)
        remaining_size = max(0, current_size - close_size)
        if remaining_size <= 0:
            return False
        trade["size"] = int(remaining_size)
        trade["de3_tiered_take_filled"] = True
        trade["de3_tiered_take_fill_price"] = float(fill_price)
        trade["de3_tiered_take_fill_time"] = fill_time
        trade["de3_tiered_take_fill_bar_index"] = _safe_idx(bar_index)
        trade["de3_tiered_take_fill_reason"] = str(fill_reason or "tiered_take")
        trade["de3_tiered_take_close_size"] = int(close_size)
        trade["de3_tiered_take_remaining_size"] = int(remaining_size)
        trade["de3_tiered_take_realized_pnl_points"] = float(pnl_points)
        trade["de3_partial_realized_pnl_dollars"] = float(
            _coerce_float(trade.get("de3_partial_realized_pnl_dollars"), 0.0) + pnl_dollars
        )
        trade["de3_partial_realized_fee_paid"] = float(
            _coerce_float(trade.get("de3_partial_realized_fee_paid"), 0.0) + fee_paid
        )
        trade["de3_partial_realized_pnl_net"] = float(
            _coerce_float(trade.get("de3_partial_realized_pnl_net"), 0.0) + pnl_net
        )
        de3_tiered_take_fill_count += 1
        de3_tiered_take_closed_contract_count += int(close_size)
        if bool(trade.get("de3_tiered_take_arm_break_even_after_fill", False)):
            _stage_de3_break_even_after_partial_fill(trade, bar_index=bar_index)
        return True

    def open_trade(
        signal: dict,
        entry_price: float,
        entry_time: dt.datetime,
        bar_index: Optional[int] = None,
    ) -> None:
        nonlocal active_trade
        nonlocal de3_profit_milestone_profile_trade_count
        nonlocal de3_entry_trade_day_extreme_profile_trade_count
        nonlocal de3_entry_trade_day_extreme_admission_checked
        nonlocal de3_entry_trade_day_extreme_admission_blocked
        nonlocal de3_entry_trade_day_extreme_size_adjustment_checked
        nonlocal de3_entry_trade_day_extreme_size_adjustment_applied
        nonlocal de3_entry_trade_day_extreme_early_exit_profile_trade_count
        nonlocal de3_entry_trade_day_extreme_early_exit_profile_hits
        requested_sl = _coerce_float(signal.get("sl_dist", MIN_SL), MIN_SL)
        requested_tp = _coerce_float(signal.get("tp_dist", MIN_TP), MIN_TP)
        sl_dist = round_points_to_tick(max(requested_sl, MIN_SL))
        tp_dist = round_points_to_tick(max(requested_tp, MIN_TP))
        strategy_name = str(signal.get("strategy", "") or "")
        bypass_sl_cap_ml = bool(
            BACKTEST_DISABLE_MAX_STOPLOSS_FOR_MLPHYSICS
            and strategy_name.startswith("MLPhysics")
        )
        bypass_sl_cap_de3_v2 = bool(
            BACKTEST_DISABLE_MAX_STOPLOSS_FOR_DE3_V2
            and de3_runtime_is_v2
            and strategy_name.startswith("DynamicEngine3")
        )
        bypass_sl_cap = bool(bypass_sl_cap_ml or bypass_sl_cap_de3_v2)
        if BACKTEST_MAX_STOPLOSS_POINTS is not None:
            # Backtest execution cap: enforce a hard maximum stop distance,
            # except when explicitly bypassed.
            sl_cap = round_points_to_tick(max(float(BACKTEST_MAX_STOPLOSS_POINTS), MIN_SL))
            if bypass_sl_cap:
                signal["sl_cap_applied"] = False
                signal["sl_cap_bypassed"] = True
                if bypass_sl_cap_de3_v2:
                    signal["sl_cap_bypass_reason"] = "de3_v2"
                elif bypass_sl_cap_ml:
                    signal["sl_cap_bypass_reason"] = "mlphysics"
            else:
                if sl_dist > sl_cap:
                    sl_dist = sl_cap
                    signal["sl_cap_applied"] = True
                else:
                    signal["sl_cap_applied"] = False
                signal["sl_cap_bypassed"] = False
                signal["sl_cap_bypass_reason"] = ""
            signal["sl_cap_points"] = sl_cap
        if sl_dist <= 0 or tp_dist <= 0:
            # Safety fallback for invalid upstream signals (e.g., experimental bypass paths).
            if sl_dist <= 0:
                sl_dist = sltp_exec_min_sl
            if tp_dist <= 0:
                tp_dist = sltp_exec_min_tp
            signal["sl_dist"] = sl_dist
            signal["tp_dist"] = tp_dist
        else:
            signal["sl_dist"] = sl_dist
            signal["tp_dist"] = tp_dist
        if bool(de3_v4_entry_trade_day_extreme_admission_cfg_local.get("enabled", False)):
            de3_entry_trade_day_extreme_admission_checked += 1
            de3_entry_trade_day_extreme_admission_ctx = (
                _resolve_de3_entry_trade_day_extreme_admission_block(
                    signal,
                    entry_price=entry_price,
                    tp_dist=tp_dist,
                    entry_trade_day_high=_coerce_float(de3_trade_day_high_known, math.nan),
                    entry_trade_day_low=_coerce_float(de3_trade_day_low_known, math.nan),
                )
            )
            signal["de3_entry_trade_day_extreme_admission_block_active"] = bool(
                de3_entry_trade_day_extreme_admission_ctx.get("active", False)
            )
            signal["de3_entry_trade_day_extreme_admission_profile_name"] = str(
                de3_entry_trade_day_extreme_admission_ctx.get("profile_name", "") or ""
            )
            signal["de3_entry_trade_day_extreme_admission_target_beyond"] = bool(
                de3_entry_trade_day_extreme_admission_ctx.get(
                    "target_beyond_trade_day_extreme",
                    False,
                )
            )
            signal["de3_entry_trade_day_extreme_admission_progress_pct"] = (
                de3_entry_trade_day_extreme_admission_ctx.get("progress_pct")
            )
            if bool(de3_entry_trade_day_extreme_admission_ctx.get("active", False)):
                de3_entry_trade_day_extreme_admission_blocked += 1
                profile_name = str(
                    de3_entry_trade_day_extreme_admission_ctx.get("profile_name", "")
                    or "entry_trade_day_extreme_admission_block"
                )
                de3_entry_trade_day_extreme_admission_profile_hits[profile_name] += 1
                return
        side = signal["side"]
        size = _coerce_int(signal.get("size", CONTRACTS), CONTRACTS)
        size = _apply_de3_v4_confidence_tier_size(signal, size)
        size = _apply_de3_backtest_entry_model_margin_size(signal, size)
        size = _apply_de3_backtest_signal_size_rules(signal, size)
        size = _apply_de3_backtest_policy_context_overlay(signal, size)
        size = _apply_de3_backtest_variant_adaptation_size(signal, size)
        if BACKTEST_DRAWDOWN_SIZE_SCALING_ENABLED:
            current_realized_dd = max(0.0, float(peak - equity))
            if current_realized_dd <= BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD:
                dd_progress = 0.0
                drawdown_size_cap = int(BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS)
            elif current_realized_dd >= BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD:
                dd_progress = 1.0
                drawdown_size_cap = int(BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS)
            else:
                dd_above = (
                    current_realized_dd - BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD
                )
                dd_progress = min(1.0, dd_above * BACKTEST_DRAWDOWN_SIZE_SCALING_SPAN_INV)
                bucket = int(dd_above / BACKTEST_DRAWDOWN_SIZE_SCALING_STEP_USD)
                if bucket < 0:
                    bucket = 0
                elif bucket > BACKTEST_DRAWDOWN_SIZE_SCALING_CONTRACT_RANGE:
                    bucket = BACKTEST_DRAWDOWN_SIZE_SCALING_CONTRACT_RANGE
                drawdown_size_cap = int(
                    BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS - bucket
                )
                if drawdown_size_cap < BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS:
                    drawdown_size_cap = int(BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS)
            requested_size = int(size)
            if size > drawdown_size_cap:
                size = int(drawdown_size_cap)
            if size < BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS:
                size = int(BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS)
            if size > BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS:
                size = int(BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS)
            signal["drawdown_size_scaling_enabled"] = True
            signal["drawdown_size_realized_dd_usd"] = float(current_realized_dd)
            signal["drawdown_size_progress"] = float(dd_progress)
            signal["drawdown_size_cap"] = int(drawdown_size_cap)
            signal["drawdown_size_requested"] = int(requested_size)
            signal["drawdown_size_applied"] = bool(size < requested_size)
            signal["drawdown_size_step_usd"] = float(BACKTEST_DRAWDOWN_SIZE_SCALING_STEP_USD)
        else:
            signal["drawdown_size_scaling_enabled"] = False
            signal["drawdown_size_applied"] = False
            signal["drawdown_size_step_usd"] = 0.0
        signal["de3_entry_trade_day_extreme_size_adjustment_checked"] = False
        signal["de3_entry_trade_day_extreme_size_adjustment_active"] = False
        signal["de3_entry_trade_day_extreme_size_adjustment_applied"] = False
        signal["de3_entry_trade_day_extreme_size_adjustment_profile_name"] = ""
        signal["de3_entry_trade_day_extreme_size_adjustment_target_beyond"] = False
        signal["de3_entry_trade_day_extreme_size_adjustment_progress_pct"] = None
        signal["de3_entry_trade_day_extreme_size_adjustment_requested_size"] = int(size)
        signal["de3_entry_trade_day_extreme_size_adjustment_final_size"] = int(size)
        signal["de3_entry_trade_day_extreme_size_adjustment_multiplier"] = 1.0
        signal["de3_entry_trade_day_extreme_size_adjustment_min_contracts"] = 1
        if bool(de3_v4_entry_trade_day_extreme_size_cfg_local.get("enabled", False)):
            de3_entry_trade_day_extreme_size_adjustment_checked += 1
            signal["de3_entry_trade_day_extreme_size_adjustment_checked"] = True
            de3_entry_trade_day_extreme_size_ctx = (
                _resolve_de3_entry_trade_day_extreme_size_adjustment(
                    signal,
                    entry_price=entry_price,
                    tp_dist=tp_dist,
                    entry_trade_day_high=_coerce_float(de3_trade_day_high_known, math.nan),
                    entry_trade_day_low=_coerce_float(de3_trade_day_low_known, math.nan),
                )
            )
            signal["de3_entry_trade_day_extreme_size_adjustment_active"] = bool(
                de3_entry_trade_day_extreme_size_ctx.get("active", False)
            )
            signal["de3_entry_trade_day_extreme_size_adjustment_profile_name"] = str(
                de3_entry_trade_day_extreme_size_ctx.get("profile_name", "") or ""
            )
            signal["de3_entry_trade_day_extreme_size_adjustment_target_beyond"] = bool(
                de3_entry_trade_day_extreme_size_ctx.get(
                    "target_beyond_trade_day_extreme",
                    False,
                )
            )
            signal["de3_entry_trade_day_extreme_size_adjustment_progress_pct"] = (
                de3_entry_trade_day_extreme_size_ctx.get("progress_pct")
            )
            signal["de3_entry_trade_day_extreme_size_adjustment_multiplier"] = float(
                max(
                    0.0,
                    _coerce_float(
                        de3_entry_trade_day_extreme_size_ctx.get("size_multiplier"),
                        1.0,
                    ),
                )
            )
            signal["de3_entry_trade_day_extreme_size_adjustment_min_contracts"] = int(
                max(
                    1,
                    _coerce_int(
                        de3_entry_trade_day_extreme_size_ctx.get("min_contracts"),
                        1,
                    ),
                )
            )
            requested_size = int(size)
            signal["de3_entry_trade_day_extreme_size_adjustment_requested_size"] = int(
                requested_size
            )
            if bool(de3_entry_trade_day_extreme_size_ctx.get("active", False)):
                candidate_size = int(
                    math.floor(
                        float(requested_size)
                        * max(
                            0.0,
                            _coerce_float(
                                de3_entry_trade_day_extreme_size_ctx.get("size_multiplier"),
                                1.0,
                            ),
                        )
                    )
                )
                min_contracts = int(
                    max(
                        1,
                        _coerce_int(
                            de3_entry_trade_day_extreme_size_ctx.get("min_contracts"),
                            1,
                        ),
                    )
                )
                if candidate_size < min_contracts:
                    candidate_size = int(min_contracts)
                if candidate_size > requested_size:
                    candidate_size = int(requested_size)
                if candidate_size < requested_size:
                    size = int(candidate_size)
                    signal["de3_entry_trade_day_extreme_size_adjustment_applied"] = True
                    de3_entry_trade_day_extreme_size_adjustment_applied += 1
                    profile_name = str(
                        de3_entry_trade_day_extreme_size_ctx.get("profile_name", "")
                        or "entry_trade_day_extreme_size_adjustment"
                    )
                    de3_entry_trade_day_extreme_size_adjustment_profile_hits[
                        profile_name
                    ] += 1
            signal["de3_entry_trade_day_extreme_size_adjustment_final_size"] = int(size)
        signal["de3_variant_adapt_final_size"] = int(size)
        stop_price = entry_price - sl_dist if side == "LONG" else entry_price + sl_dist
        td_tier = signal.get("trend_day_tier")
        td_dir = signal.get("trend_day_dir")
        if td_tier is None:
            td_tier = trend_day_tier
            td_dir = trend_day_dir
        decision_market_conditions = signal.get("market_conditions", {})
        entry_market_conditions = _snapshot_market_conditions(
            signal,
            entry_time,
            phase="entry_execution",
            bar_index=bar_index,
            execution_price=entry_price,
        )
        de3_v4_trade_management_enabled = bool(
            de3_v4_trade_management_cfg_local.get("enabled", False)
            and _is_de3_v4_trade_management_payload(signal)
        )
        de3_break_even_enabled = bool(
            de3_v4_trade_management_enabled
            and de3_v4_break_even_cfg_local.get("enabled", False)
        )
        de3_tiered_take_enabled = bool(
            de3_v4_trade_management_enabled
            and de3_v4_tiered_take_cfg_local.get("enabled", False)
        )
        de3_global_early_exit_enabled = bool(
            de3_v4_trade_management_enabled
            and de3_v4_early_exit_cfg_local.get("enabled", False)
        )
        de3_profit_milestone_ctx = _resolve_de3_profit_milestone_profile(
            signal,
            entry_price=entry_price,
            tp_dist=tp_dist,
        )
        if bool(de3_profit_milestone_ctx.get("active", False)):
            de3_profit_milestone_profile_trade_count += 1
        de3_entry_trade_day_extreme_ctx = _resolve_de3_entry_trade_day_extreme_profile(
            signal,
            entry_price=entry_price,
            tp_dist=tp_dist,
            entry_trade_day_high=_coerce_float(de3_trade_day_high_known, math.nan),
            entry_trade_day_low=_coerce_float(de3_trade_day_low_known, math.nan),
        )
        if bool(de3_entry_trade_day_extreme_ctx.get("active", False)):
            de3_entry_trade_day_extreme_profile_trade_count += 1
        de3_entry_trade_day_extreme_early_exit_ctx = (
            _resolve_de3_entry_trade_day_extreme_early_exit_profile(
                signal,
                entry_price=entry_price,
                tp_dist=tp_dist,
                entry_trade_day_high=_coerce_float(de3_trade_day_high_known, math.nan),
                entry_trade_day_low=_coerce_float(de3_trade_day_low_known, math.nan),
            )
        )
        if bool(de3_entry_trade_day_extreme_early_exit_ctx.get("active", False)):
            de3_entry_trade_day_extreme_early_exit_profile_trade_count += 1
            profile_name = str(
                de3_entry_trade_day_extreme_early_exit_ctx.get("profile_name", "")
                or "entry_trade_day_extreme_early_exit"
            )
            de3_entry_trade_day_extreme_early_exit_profile_hits[profile_name] += 1
        de3_early_exit_enabled = bool(
            de3_global_early_exit_enabled
            or de3_entry_trade_day_extreme_early_exit_ctx.get("active", False)
        )
        active_trade = {
            "strategy": signal.get("strategy", "Unknown"),
            "sub_strategy": signal.get("sub_strategy"),
            "side": side,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "entry_index": _safe_idx(bar_index),
            "entry_bar": bar_count,
            "bars_held": 0,
            "horizon_bars": int(
                max(
                    0,
                    _coerce_int(
                        signal.get("horizon_bars", signal.get("aetherflow_horizon_bars", 0)),
                        0,
                    ),
                )
            ),
            "use_horizon_time_stop": bool(
                signal.get(
                    "use_horizon_time_stop",
                    signal.get("aetherflow_use_horizon_time_stop", False),
                )
            ),
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "requested_tp_dist": requested_tp,
            "requested_sl_dist": requested_sl,
            "sl_cap_points": signal.get("sl_cap_points"),
            "sl_cap_applied": bool(signal.get("sl_cap_applied", False)),
            "sl_cap_bypass_reason": signal.get("sl_cap_bypass_reason"),
            "drawdown_size_cap": signal.get("drawdown_size_cap"),
            "drawdown_size_requested": signal.get("drawdown_size_requested"),
            "drawdown_size_applied": bool(signal.get("drawdown_size_applied", False)),
            "drawdown_size_realized_dd_usd": signal.get("drawdown_size_realized_dd_usd"),
            "de3_signal_size_rules_requested_size": signal.get("de3_signal_size_rules_requested_size"),
            "de3_signal_size_rules_final_size": signal.get("de3_signal_size_rules_final_size"),
            "de3_signal_size_rules_applied": bool(signal.get("de3_signal_size_rules_applied", False)),
            "de3_signal_size_rule_names": signal.get("de3_signal_size_rule_names"),
            "de3_signal_size_rule_reasons": signal.get("de3_signal_size_rule_reasons"),
            "de3_entry_trade_day_extreme_size_adjustment_checked": bool(
                signal.get("de3_entry_trade_day_extreme_size_adjustment_checked", False)
            ),
            "de3_entry_trade_day_extreme_size_adjustment_active": bool(
                signal.get("de3_entry_trade_day_extreme_size_adjustment_active", False)
            ),
            "de3_entry_trade_day_extreme_size_adjustment_applied": bool(
                signal.get("de3_entry_trade_day_extreme_size_adjustment_applied", False)
            ),
            "de3_entry_trade_day_extreme_size_adjustment_profile_name": signal.get(
                "de3_entry_trade_day_extreme_size_adjustment_profile_name"
            ),
            "de3_entry_trade_day_extreme_size_adjustment_target_beyond": bool(
                signal.get("de3_entry_trade_day_extreme_size_adjustment_target_beyond", False)
            ),
            "de3_entry_trade_day_extreme_size_adjustment_progress_pct": signal.get(
                "de3_entry_trade_day_extreme_size_adjustment_progress_pct"
            ),
            "de3_entry_trade_day_extreme_size_adjustment_requested_size": signal.get(
                "de3_entry_trade_day_extreme_size_adjustment_requested_size"
            ),
            "de3_entry_trade_day_extreme_size_adjustment_final_size": signal.get(
                "de3_entry_trade_day_extreme_size_adjustment_final_size"
            ),
            "de3_entry_trade_day_extreme_size_adjustment_multiplier": signal.get(
                "de3_entry_trade_day_extreme_size_adjustment_multiplier"
            ),
            "de3_entry_trade_day_extreme_size_adjustment_min_contracts": signal.get(
                "de3_entry_trade_day_extreme_size_adjustment_min_contracts"
            ),
            "size": size,
            "original_size": size,
            "current_stop_price": stop_price,
            "profit_crosses": 0,
            "was_green": None,
            "entry_mode": signal.get("entry_mode", "standard"),
            "vol_regime": signal.get("vol_regime", "UNKNOWN"),
            "mfe_points": 0.0,
            "mae_points": 0.0,
            "rescue_from_strategy": signal.get("rescue_from_strategy"),
            "rescue_from_sub_strategy": signal.get("rescue_from_sub_strategy"),
            "rescue_trigger": signal.get("rescue_trigger"),
            "consensus_contributors": signal.get("consensus_contributors"),
            "bypassed_filters": signal.get("bypassed_filters"),
            "trend_day_tier": td_tier,
            "trend_day_dir": td_dir,
            "market_conditions": entry_market_conditions,
            "decision_market_conditions": decision_market_conditions,
            "de3_trade_management_enabled": de3_v4_trade_management_enabled,
            "de3_break_even_enabled": de3_break_even_enabled,
            "de3_break_even_activate_on_next_bar": bool(
                de3_v4_break_even_cfg_local.get("activate_on_next_bar", True)
            ),
            "de3_break_even_trigger_pct": float(
                _coerce_float(de3_v4_break_even_cfg_local.get("trigger_pct"), 0.0)
            ),
            "de3_break_even_buffer_ticks": int(
                max(0, _coerce_int(de3_v4_break_even_cfg_local.get("buffer_ticks"), 0))
            ),
            "de3_break_even_trail_pct": float(
                max(0.0, _coerce_float(de3_v4_break_even_cfg_local.get("trail_pct"), 0.0))
            ),
            "de3_break_even_post_activation_trail_pct": float(
                max(
                    0.0,
                    _coerce_float(
                        de3_v4_break_even_cfg_local.get("post_activation_trail_pct"),
                        _coerce_float(de3_v4_break_even_cfg_local.get("trail_pct"), 0.0),
                    ),
                )
            ),
            "de3_break_even_post_partial_trail_pct": float(
                max(
                    0.0,
                    _coerce_float(
                        de3_v4_break_even_cfg_local.get("post_partial_trail_pct"),
                        _coerce_float(
                            de3_v4_break_even_cfg_local.get("post_activation_trail_pct"),
                            _coerce_float(de3_v4_break_even_cfg_local.get("trail_pct"), 0.0),
                        ),
                    ),
                )
            ),
            "de3_break_even_armed": False,
            "de3_break_even_applied": False,
            "de3_break_even_move_count": 0,
            "de3_break_even_last_stop_price": None,
            "de3_break_even_pending_stop_price": None,
            "de3_break_even_pending_from_bar_index": None,
            "de3_break_even_trigger_bar_index": None,
            "de3_break_even_trigger_mfe_points": None,
            "de3_break_even_locked_points": 0.0,
            "de3_effective_stop_price": stop_price,
            "de3_profit_milestone_profile_active": bool(
                de3_profit_milestone_ctx.get("active", False)
            ),
            "de3_profit_milestone_profile_name": str(
                de3_profit_milestone_ctx.get("profile_name", "") or ""
            ),
            "de3_profit_milestone_price": de3_profit_milestone_ctx.get("milestone_price"),
            "de3_profit_milestone_trigger_pct": float(
                max(0.0, _coerce_float(de3_profit_milestone_ctx.get("trigger_pct"), 0.0))
            ),
            "de3_profit_milestone_force_break_even": bool(
                de3_profit_milestone_ctx.get("force_break_even_on_reach", False)
            ),
            "de3_profit_milestone_reached": False,
            "de3_profit_milestone_reached_bar_index": None,
            "de3_profit_milestone_reached_mfe_points": None,
            "de3_break_even_post_profit_milestone_trail_pct": float(
                max(
                    0.0,
                    _coerce_float(
                        de3_profit_milestone_ctx.get("post_reach_trail_pct"),
                        0.0,
                    ),
                )
            ),
            "de3_entry_trade_day_high": de3_entry_trade_day_extreme_ctx.get("entry_trade_day_high"),
            "de3_entry_trade_day_low": de3_entry_trade_day_extreme_ctx.get("entry_trade_day_low"),
            "de3_entry_trade_day_extreme_price": de3_entry_trade_day_extreme_ctx.get("extreme_price"),
            "de3_entry_trade_day_extreme_progress_pct": de3_entry_trade_day_extreme_ctx.get("progress_pct"),
            "de3_entry_trade_day_extreme_target_beyond": bool(
                de3_entry_trade_day_extreme_ctx.get("target_beyond_trade_day_extreme", False)
            ),
            "de3_entry_trade_day_extreme_profile_active": bool(
                de3_entry_trade_day_extreme_ctx.get("active", False)
            ),
            "de3_entry_trade_day_extreme_profile_name": str(
                de3_entry_trade_day_extreme_ctx.get("profile_name", "") or ""
            ),
            "de3_entry_trade_day_extreme_force_break_even": bool(
                de3_entry_trade_day_extreme_ctx.get("force_break_even_on_reach", False)
            ),
            "de3_entry_trade_day_extreme_reached": False,
            "de3_entry_trade_day_extreme_reached_bar_index": None,
            "de3_entry_trade_day_extreme_reached_mfe_points": None,
            "de3_break_even_post_entry_trade_day_extreme_trail_pct": float(
                max(
                    0.0,
                    _coerce_float(
                        de3_entry_trade_day_extreme_ctx.get("post_reach_trail_pct"),
                        0.0,
                    ),
                )
            ),
            "de3_tiered_take_enabled": de3_tiered_take_enabled,
            "de3_tiered_take_trigger_pct": float(
                max(0.0, _coerce_float(de3_v4_tiered_take_cfg_local.get("trigger_pct"), 0.0))
            ),
            "de3_tiered_take_close_fraction": float(
                max(0.0, _coerce_float(de3_v4_tiered_take_cfg_local.get("close_fraction"), 0.0))
            ),
            "de3_tiered_take_min_entry_contracts": int(
                max(2, _coerce_int(de3_v4_tiered_take_cfg_local.get("min_entry_contracts"), 2))
            ),
            "de3_tiered_take_min_remaining_contracts": int(
                max(1, _coerce_int(de3_v4_tiered_take_cfg_local.get("min_remaining_contracts"), 1))
            ),
            "de3_tiered_take_arm_break_even_after_fill": bool(
                de3_v4_tiered_take_cfg_local.get("arm_break_even_after_fill", True)
            ),
            "de3_tiered_take_filled": False,
            "de3_tiered_take_fill_price": None,
            "de3_tiered_take_fill_time": None,
            "de3_tiered_take_fill_bar_index": None,
            "de3_tiered_take_fill_reason": "",
            "de3_tiered_take_close_size": 0,
            "de3_tiered_take_remaining_size": int(size),
            "de3_tiered_take_realized_pnl_points": 0.0,
            "de3_partial_realized_pnl_dollars": 0.0,
            "de3_partial_realized_fee_paid": 0.0,
            "de3_partial_realized_pnl_net": 0.0,
            "de3_early_exit_enabled": de3_early_exit_enabled,
            "de3_early_exit_profile_name": str(
                de3_entry_trade_day_extreme_early_exit_ctx.get("profile_name", "") or ""
            ),
            "de3_entry_trade_day_extreme_early_exit_profile_active": bool(
                de3_entry_trade_day_extreme_early_exit_ctx.get("active", False)
            ),
            "de3_entry_trade_day_extreme_early_exit_target_beyond": bool(
                de3_entry_trade_day_extreme_early_exit_ctx.get(
                    "target_beyond_trade_day_extreme",
                    False,
                )
            ),
            "de3_early_exit_exit_if_not_green_by": (
                int(
                    max(
                        0,
                        _coerce_int(
                            de3_v4_early_exit_cfg_local.get("exit_if_not_green_by"),
                            0,
                        ),
                    )
                )
                if de3_global_early_exit_enabled
                else None
            ),
            "de3_early_exit_max_profit_crosses": (
                int(
                    max(
                        0,
                        _coerce_int(
                            de3_v4_early_exit_cfg_local.get("max_profit_crosses"),
                            0,
                        ),
                    )
                )
                if de3_global_early_exit_enabled
                else None
            ),
            "de3_early_exit_min_progress_by_bars": (
                int(
                    max(
                        0,
                        _coerce_int(
                            de3_entry_trade_day_extreme_early_exit_ctx.get(
                                "min_progress_by_bars"
                            ),
                            0,
                        ),
                    )
                )
                if de3_entry_trade_day_extreme_early_exit_ctx.get(
                    "min_progress_by_bars"
                )
                is not None
                else None
            ),
            "de3_early_exit_min_progress_pct": (
                float(
                    _coerce_float(
                        de3_entry_trade_day_extreme_early_exit_ctx.get(
                            "min_progress_pct"
                        ),
                        0.0,
                    )
                )
                if math.isfinite(
                    _coerce_float(
                        de3_entry_trade_day_extreme_early_exit_ctx.get(
                            "min_progress_pct"
                        ),
                        float("nan"),
                    )
                )
                else None
            ),
            "de3_early_exit_profile_max_profit_crosses": (
                int(
                    max(
                        0,
                        _coerce_int(
                            de3_entry_trade_day_extreme_early_exit_ctx.get(
                                "max_profit_crosses"
                            ),
                            0,
                        ),
                    )
                )
                if de3_entry_trade_day_extreme_early_exit_ctx.get(
                    "max_profit_crosses"
                )
                is not None
                else None
            ),
            "de3_early_exit_trigger_reason": "",
            "early_exit_enabled": signal.get("early_exit_enabled"),
            "early_exit_exit_if_not_green_by": int(
                max(0, _coerce_int(signal.get("early_exit_exit_if_not_green_by"), 0))
            ),
            "early_exit_max_profit_crosses": int(
                max(0, _coerce_int(signal.get("early_exit_max_profit_crosses"), 0))
            ),
        }
        for key, value in signal.items():
            if str(key).startswith("de3_variant_adapt_"):
                active_trade[key] = value
        for key, value in signal.items():
            # Persist strategy-specific instrumentation fields into the trade log.
            # (Used later for report analysis/tuning; keep this lightweight.)
            if str(key).startswith(("ml_", "vab_", "de3_", "regime_manifold_")):
                active_trade[key] = value
        for key in ("combo_key", "reverted", "original_signal"):
            if key in signal:
                active_trade[key] = signal.get(key)

    def check_stop_take(
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
    ) -> Optional[tuple[float, str]]:
        if active_trade is None:
            return None
        side = active_trade["side"]
        entry = active_trade["entry_price"]
        stop_price = active_trade.get("current_stop_price")
        if stop_price is None:
            sl_dist = _coerce_float(active_trade.get("sl_dist", MIN_SL), MIN_SL)
            stop_price = entry - sl_dist if side == "LONG" else entry + sl_dist
        tp_dist = _coerce_float(active_trade.get("tp_dist", MIN_TP), MIN_TP)
        take_price = entry + tp_dist if side == "LONG" else entry - tp_dist
        tiered_take_price = math.nan
        tiered_take_enabled = bool(
            active_trade.get("de3_tiered_take_enabled", False)
            and not active_trade.get("de3_tiered_take_filled", False)
            and _coerce_de3_tiered_take_close_size(active_trade) > 0
        )
        if tiered_take_enabled:
            tiered_take_trigger_pct = max(
                0.0,
                _coerce_float(active_trade.get("de3_tiered_take_trigger_pct"), 0.0),
            )
            if 0.0 < tiered_take_trigger_pct < 1.0:
                partial_tp_dist = round_points_to_tick(
                    max(float(TICK_SIZE), float(tp_dist) * float(tiered_take_trigger_pct))
                )
                tiered_take_price = (
                    entry + partial_tp_dist if side == "LONG" else entry - partial_tp_dist
                )
                if (
                    (side == "LONG" and tiered_take_price >= take_price - 1e-12)
                    or (side == "SHORT" and tiered_take_price <= take_price + 1e-12)
                ):
                    tiered_take_price = math.nan
        if BACKTEST_GAP_FILLS:
            if side == "LONG":
                if bar_open <= stop_price:
                    return stop_price, "stop_gap"
                if bar_open >= take_price:
                    return take_price, "take_gap"
                if math.isfinite(tiered_take_price) and bar_open >= tiered_take_price:
                    return tiered_take_price, "tiered_take_gap"
            else:
                if bar_open >= stop_price:
                    return stop_price, "stop_gap"
                if bar_open <= take_price:
                    return take_price, "take_gap"
                if math.isfinite(tiered_take_price) and bar_open <= tiered_take_price:
                    return tiered_take_price, "tiered_take_gap"
        hit_stop = bar_low <= stop_price if side == "LONG" else bar_high >= stop_price
        hit_take = bar_high >= take_price if side == "LONG" else bar_low <= take_price
        hit_tiered_take = False
        if math.isfinite(tiered_take_price):
            hit_tiered_take = (
                bar_high >= tiered_take_price if side == "LONG" else bar_low <= tiered_take_price
            )
        if hit_stop and hit_take:
            return _resolve_sl_tp_conflict(side, bar_open, bar_close, stop_price, take_price)
        if hit_take:
            return take_price, "take"
        if hit_stop and hit_tiered_take:
            partial_price, partial_reason = _resolve_sl_tp_conflict(
                side,
                bar_open,
                bar_close,
                stop_price,
                tiered_take_price,
            )
            if partial_reason == "take":
                return partial_price, "tiered_take"
            return stop_price, "stop"
        if hit_tiered_take:
            return tiered_take_price, "tiered_take"
        if hit_stop:
            return stop_price, "stop"
        return None

    early_exit_cfg_all = CONFIG.get("EARLY_EXIT", {}) or {}

    def advance_trade_management_state(current_price: float) -> None:
        if active_trade is None:
            return
        active_trade["bars_held"] = int(max(0, _coerce_int(active_trade.get("bars_held"), 0))) + 1
        if active_trade["side"] == "LONG":
            is_green = current_price > active_trade["entry_price"]
        else:
            is_green = current_price < active_trade["entry_price"]
        was_green = active_trade.get("was_green")
        if was_green is not None and is_green != was_green:
            active_trade["profit_crosses"] = active_trade.get("profit_crosses", 0) + 1
        active_trade["was_green"] = is_green

    def check_signal_horizon_exit() -> bool:
        if active_trade is None:
            return False
        if not bool(active_trade.get("use_horizon_time_stop", False)):
            return False
        horizon_bars = int(max(0, _coerce_int(active_trade.get("horizon_bars"), 0)))
        if horizon_bars <= 0:
            return False
        return int(max(0, _coerce_int(active_trade.get("bars_held"), 0))) >= horizon_bars

    def _current_trade_progress_pct(trade: Optional[dict], current_price: float) -> float:
        if not isinstance(trade, dict):
            return float("nan")
        side_name = str(trade.get("side", "") or "").upper()
        entry_price = _coerce_float(trade.get("entry_price"), math.nan)
        tp_dist = _coerce_float(trade.get("tp_dist"), math.nan)
        current_price = _coerce_float(current_price, math.nan)
        if (
            side_name not in {"LONG", "SHORT"}
            or not math.isfinite(entry_price)
            or not math.isfinite(tp_dist)
            or tp_dist <= 0.0
            or not math.isfinite(current_price)
        ):
            return float("nan")
        return float(compute_pnl_points(side_name, entry_price, current_price) / tp_dist)

    def check_early_exit(current_price: float) -> Optional[str]:
        if active_trade is None:
            return None
        early_exit_config = None
        if "early_exit_enabled" in active_trade and active_trade.get("early_exit_enabled") is not None:
            strategy_name = active_trade.get("strategy", "")
            base_cfg = early_exit_cfg_all.get(strategy_name, {})
            if not isinstance(base_cfg, dict):
                base_cfg = {}
            early_exit_config = {
                "enabled": bool(active_trade.get("early_exit_enabled")),
                "exit_if_not_green_by": (
                    int(
                        max(
                            0,
                            _coerce_int(
                                active_trade.get("early_exit_exit_if_not_green_by"),
                                base_cfg.get("exit_if_not_green_by", 0),
                            ),
                        )
                    )
                    if active_trade.get("early_exit_exit_if_not_green_by") is not None
                    or base_cfg.get("exit_if_not_green_by") is not None
                    else None
                ),
                "max_profit_crosses": (
                    int(
                        max(
                            0,
                            _coerce_int(
                                active_trade.get("early_exit_max_profit_crosses"),
                                base_cfg.get("max_profit_crosses", 0),
                            ),
                        )
                    )
                    if active_trade.get("early_exit_max_profit_crosses") is not None
                    or base_cfg.get("max_profit_crosses") is not None
                    else None
                ),
            }
        elif (
            bool(active_trade.get("de3_early_exit_enabled", False))
            and _is_de3_v4_trade_management_payload(active_trade)
        ):
            early_exit_config = {
                "enabled": True,
                "exit_if_not_green_by": (
                    int(
                        max(
                            0,
                            _coerce_int(
                                active_trade.get("de3_early_exit_exit_if_not_green_by"),
                                0,
                            ),
                        )
                    )
                    if active_trade.get("de3_early_exit_exit_if_not_green_by") is not None
                    else None
                ),
                "max_profit_crosses": (
                    int(
                        max(
                            0,
                            _coerce_int(
                                active_trade.get("de3_early_exit_max_profit_crosses"),
                                0,
                            ),
                        )
                    )
                    if active_trade.get("de3_early_exit_max_profit_crosses") is not None
                    else None
                ),
                "min_progress_by_bars": (
                    int(
                        max(
                            0,
                            _coerce_int(
                                active_trade.get("de3_early_exit_min_progress_by_bars"),
                                0,
                            ),
                        )
                    )
                    if active_trade.get("de3_early_exit_min_progress_by_bars") is not None
                    else None
                ),
                "min_progress_pct": (
                    float(
                        _coerce_float(
                            active_trade.get("de3_early_exit_min_progress_pct"),
                            0.0,
                        )
                    )
                    if math.isfinite(
                        _coerce_float(
                            active_trade.get("de3_early_exit_min_progress_pct"),
                            float("nan"),
                        )
                    )
                    else None
                ),
                "profile_max_profit_crosses": (
                    int(
                        max(
                            0,
                            _coerce_int(
                                active_trade.get("de3_early_exit_profile_max_profit_crosses"),
                                0,
                            ),
                        )
                    )
                    if active_trade.get("de3_early_exit_profile_max_profit_crosses")
                    is not None
                    else None
                ),
                "profile_name": str(active_trade.get("de3_early_exit_profile_name", "") or ""),
            }
        elif BACKTEST_EARLY_EXIT_ENABLED:
            strategy_name = active_trade.get("strategy", "")
            cfg = early_exit_cfg_all.get(strategy_name, {})
            if isinstance(cfg, dict):
                early_exit_config = cfg
        if not isinstance(early_exit_config, dict) or not early_exit_config.get("enabled", False):
            return None

        if active_trade["side"] == "LONG":
            is_green = current_price > active_trade["entry_price"]
        else:
            is_green = current_price < active_trade["entry_price"]

        progress_bars = early_exit_config.get("min_progress_by_bars", None)
        min_progress_pct = early_exit_config.get("min_progress_pct", None)
        if (
            progress_bars is not None
            and int(progress_bars) > 0
            and min_progress_pct is not None
            and active_trade["bars_held"] >= int(progress_bars)
        ):
            threshold = float(min_progress_pct)
            if threshold <= 1e-12:
                if not is_green:
                    return f"not_green_after_{int(progress_bars)}_bars"
            else:
                current_progress_pct = _current_trade_progress_pct(active_trade, current_price)
                if (
                    not math.isfinite(current_progress_pct)
                    or current_progress_pct + 1e-12 < threshold
                ):
                    return (
                        f"progress_{current_progress_pct:.4f}_lt_{threshold:.4f}"
                        f"_after_{int(progress_bars)}_bars"
                    )

        exit_time = early_exit_config.get("exit_if_not_green_by", None)
        exit_cross = early_exit_config.get("max_profit_crosses", None)
        profile_exit_cross = early_exit_config.get("profile_max_profit_crosses", None)

        if exit_time is not None and int(exit_time) > 0 and active_trade["bars_held"] >= int(exit_time) and not is_green:
            return f"not_green_after_{int(exit_time)}_bars"
        if profile_exit_cross is not None and active_trade.get("profit_crosses", 0) > int(profile_exit_cross):
            return f"choppy_{int(active_trade.get('profit_crosses', 0))}_crosses_gt_{int(profile_exit_cross)}"
        if exit_cross is not None and active_trade.get("profit_crosses", 0) > int(exit_cross):
            return f"choppy_{int(active_trade.get('profit_crosses', 0))}_crosses_gt_{int(exit_cross)}"
        return None

    def handle_signal(signal: dict, bar_index: Optional[int] = None) -> None:
        nonlocal pending_entry, pending_exit, pending_exit_reason, opposite_signal_count
        if isinstance(signal, dict) and "market_conditions" not in signal:
            signal["market_conditions"] = _snapshot_market_conditions(
                signal,
                current_time,
                phase="signal_decision",
                bar_index=bar_index,
                execution_price=bar_close,
            )
        if isinstance(signal, dict):
            if not _apply_de3_backtest_intraday_regime_control(
                signal,
                current_time,
                bar_index=bar_index,
            ):
                opposite_signal_count = 0
                return
            if not _apply_de3_backtest_walkforward_gate(signal, current_time):
                opposite_signal_count = 0
                return
            if not _apply_de3_backtest_admission_control(signal):
                opposite_signal_count = 0
                return

        if active_trade is None:
            if pending_entry is None:
                pending_entry = signal
            opposite_signal_count = 0
            return
        if active_trade["side"] == signal["side"]:
            opposite_signal_count = 0
            return
        opposite_signal_count += 1
        if opposite_signal_count >= OPPOSITE_SIGNAL_THRESHOLD:
            pending_exit = True
            pending_exit_reason = "reverse"
            if pending_entry is None:
                pending_entry = signal
            opposite_signal_count = 0

    mnq_pos = None
    vix_pos = None
    if not mnq_df.empty:
        mnq_pos = np.searchsorted(
            mnq_df.index.values.astype("datetime64[ns]"),
            full_index_ns,
            side="right",
        )
    if not vix_df.empty:
        vix_pos = np.searchsorted(
            vix_df.index.values.astype("datetime64[ns]"),
            full_index_ns,
            side="right",
        )
    empty_mnq_slice = mnq_df.iloc[:0]
    empty_vix_slice = vix_df.iloc[:0]
    asia_ema_fast_state: Optional[float] = None
    asia_ema_slow_state: Optional[float] = None
    asia_ema_obs = 0
    asia_ema_fast_window: deque[float] = deque(maxlen=max(asia_trend_ema_slope_bars, 1))
    asia_tf_high_window: deque[tuple[int, float]] = deque()
    asia_tf_low_window: deque[tuple[int, float]] = deque()
    asia_tf_box_range_current = math.nan

    emit_init_status("Initialization complete. Starting bar simulation...")
    sim_started_at = time.perf_counter()
    console_progress_last_ts = sim_started_at - float(console_progress_every_sec)

    def _tail_df_for_bar(
        lookback: int,
        history_df_local: Optional[pd.DataFrame],
        history_start_local: int,
        end_pos: int,
        cache: dict[int, pd.DataFrame],
    ) -> pd.DataFrame:
        lb = int(lookback) if lookback else 0
        if lb <= 0:
            if history_df_local is None:
                history_df_local = full_df.iloc[history_start_local:end_pos]
            return history_df_local
        cached = cache.get(lb)
        if cached is None:
            available = max(0, int(end_pos) - int(history_start_local))
            if lb >= available:
                cached = (
                    history_df_local
                    if history_df_local is not None
                    else full_df.iloc[history_start_local:end_pos]
                )
            else:
                slice_start = max(int(history_start_local), int(end_pos) - lb)
                cached = full_df.iloc[slice_start:end_pos]
            cache[lb] = cached
        return cached

    def _refresh_regime_meta(
        history_df_local: Optional[pd.DataFrame],
        ts,
        session_name_local: str,
    ) -> Optional[dict]:
        nonlocal manifold_last_label
        if regime_manifold_engine is None or history_df_local is None:
            return None
        try:
            current_regime_meta = regime_manifold_engine.update(
                history_df_local,
                ts=ts,
                session=session_name_local,
            )
            if isinstance(current_regime_meta, dict):
                manifold_label = str(current_regime_meta.get("regime", "UNKNOWN"))
                if manifold_label != manifold_last_label:
                    logging.info(
                        "Backtest RegimeManifold: %s | R=%.3f stress=%.3f risk=%.2f side=%s no_trade=%s",
                        manifold_label,
                        float(current_regime_meta.get("R", 0.0) or 0.0),
                        float(current_regime_meta.get("stress", 0.0) or 0.0),
                        float(current_regime_meta.get("risk_mult", 1.0) or 1.0),
                        int(current_regime_meta.get("side_bias", 0) or 0),
                        bool(current_regime_meta.get("no_trade", False)),
                    )
                    manifold_last_label = manifold_label
                return current_regime_meta
        except Exception as exc:
            logging.warning("Backtest RegimeManifold update failed: %s", exc)
        return None

    def _attach_regime_meta_for_de3_manifold(
        signal_payload: Optional[dict],
        fallback_name: Optional[str],
    ) -> None:
        if regime_meta is None or not isinstance(signal_payload, dict):
            return
        try:
            _, _, updates = apply_meta_policy(
                signal_payload,
                regime_meta,
                fallback_name=fallback_name,
                default_size=CONTRACTS,
                enforce_side_bias=manifold_enforce_side_bias,
            )
        except Exception:
            return
        if updates:
            updates = dict(updates)
            updates.pop("size", None)
            signal_payload.update(updates)

    def _de3_manifold_adaptation_allows_signal(
        signal_payload: Optional[dict],
        fallback_name: Optional[str],
    ) -> bool:
        nonlocal de3_manifold_adapt_checked
        nonlocal de3_manifold_adapt_would_block
        nonlocal de3_manifold_adapt_blocked
        nonlocal de3_manifold_adapt_shadow
        if not de3_manifold_adapt_enabled or regime_meta is None or not isinstance(signal_payload, dict):
            return True
        strat_label = str(signal_payload.get("strategy", fallback_name) or fallback_name)
        strat_label_norm = strat_label.strip().lower()
        de3_signal_version = str(
            signal_payload.get("de3_version", de3_runtime_db_version) or de3_runtime_db_version
        ).lower()
        if not strat_label_norm.startswith("dynamicengine3") or de3_signal_version != "v4":
            return True
        if "regime_manifold_regime" not in signal_payload:
            _attach_regime_meta_for_de3_manifold(signal_payload, fallback_name)
        de3_manifold_adapt_checked += 1
        reasons = []
        manifold_regime = str(signal_payload.get("regime_manifold_regime", "") or "").strip().upper()
        if manifold_regime and manifold_regime in de3_manifold_blocked_regimes:
            reasons.append(f"regime={manifold_regime}")
        if de3_manifold_block_no_trade and bool(signal_payload.get("regime_manifold_no_trade", False)):
            reasons.append("no_trade")
        if de3_manifold_require_allow_style and not bool(signal_payload.get("regime_manifold_allow_style", True)):
            reasons.append("allow_style=False")
        signal_payload["de3_manifold_adapt_mode"] = de3_manifold_adapt_mode
        signal_payload["de3_manifold_adapt_would_block"] = bool(reasons)
        signal_payload["de3_manifold_adapt_reasons"] = list(reasons)
        if not reasons:
            return True
        de3_manifold_adapt_would_block += 1
        for reason in reasons:
            de3_manifold_adapt_reasons[str(reason)] += 1
        if de3_manifold_adapt_mode == "block":
            de3_manifold_adapt_blocked += 1
            if de3_manifold_adapt_log_decisions:
                logging.info(
                    "DE3 manifold adapt block | sub=%s side=%s reasons=%s",
                    signal_payload.get("sub_strategy"),
                    signal_payload.get("side"),
                    ",".join(reasons),
                )
            return False
        de3_manifold_adapt_shadow += 1
        return True

    def _de3_admission_key_from_payload(payload: Optional[dict]) -> str:
        if not isinstance(payload, dict):
            return ""
        variant_id = _de3_variant_id_from_payload(payload)
        lane_name = str(payload.get("de3_v4_selected_lane", "") or "").strip()
        if de3_admission_key_granularity == "variant":
            return variant_id
        if de3_admission_key_granularity == "lane":
            return lane_name
        if variant_id:
            parts = [part.strip() for part in variant_id.split("_") if str(part).strip()]
            if len(parts) >= 3:
                base_parts = parts[:3]
                if de3_admission_key_granularity == "lane_margin_context":
                    score = _coerce_float(payload.get("de3_v4_entry_model_score"), float("nan"))
                    threshold = _coerce_float(payload.get("de3_v4_entry_model_threshold"), float("nan"))
                    if math.isfinite(score) and math.isfinite(threshold):
                        margin = float(score - threshold)
                        if margin <= 0.08:
                            margin_bucket = "margin_low"
                        elif margin <= 0.22:
                            margin_bucket = "margin_mid"
                        else:
                            margin_bucket = "margin_high"
                    else:
                        margin_bucket = "margin_na"
                    return "|".join([*base_parts, margin_bucket])
                return "|".join(base_parts)
        timeframe = str(payload.get("de3_timeframe", "") or "").strip()
        session_name = str(payload.get("session", "") or "").strip()
        parts = [part for part in (timeframe, session_name, lane_name) if part]
        if de3_admission_key_granularity == "lane_margin_context":
            score = _coerce_float(payload.get("de3_v4_entry_model_score"), float("nan"))
            threshold = _coerce_float(payload.get("de3_v4_entry_model_threshold"), float("nan"))
            if math.isfinite(score) and math.isfinite(threshold):
                margin = float(score - threshold)
                if margin <= 0.08:
                    parts.append("margin_low")
                elif margin <= 0.22:
                    parts.append("margin_mid")
                else:
                    parts.append("margin_high")
            else:
                parts.append("margin_na")
        return "|".join(parts)

    def _de3_signal_is_weak_from_thresholds(
        signal_payload: Optional[dict],
        *,
        require_signal_weakness: bool,
        max_execution_quality: float,
        max_entry_margin: float,
        max_route_confidence: float,
        max_edge_points: float,
    ) -> tuple[bool, dict]:
        metrics = {
            "execution_quality_score": None,
            "entry_model_margin": None,
            "route_confidence": None,
            "edge_points": None,
        }
        if not isinstance(signal_payload, dict):
            return (not require_signal_weakness), metrics

        weak = True
        weak_tests = 0

        quality_score = _coerce_float(
            signal_payload.get("de3_v4_execution_quality_score"),
            float("nan"),
        )
        if math.isfinite(quality_score):
            metrics["execution_quality_score"] = float(quality_score)
        if math.isfinite(max_execution_quality):
            weak_tests += 1
            weak = weak and math.isfinite(quality_score) and quality_score <= max_execution_quality

        entry_model_score = _coerce_float(
            signal_payload.get("de3_v4_entry_model_score"),
            float("nan"),
        )
        entry_model_threshold = _coerce_float(
            signal_payload.get("de3_v4_entry_model_threshold"),
            float("nan"),
        )
        entry_model_margin = float("nan")
        if math.isfinite(entry_model_score) and math.isfinite(entry_model_threshold):
            entry_model_margin = float(entry_model_score - entry_model_threshold)
            metrics["entry_model_margin"] = float(entry_model_margin)
        if math.isfinite(max_entry_margin):
            weak_tests += 1
            weak = weak and math.isfinite(entry_model_margin) and entry_model_margin <= max_entry_margin

        route_confidence = _coerce_float(
            signal_payload.get("de3_v4_route_confidence"),
            float("nan"),
        )
        if math.isfinite(route_confidence):
            metrics["route_confidence"] = float(route_confidence)
        if math.isfinite(max_route_confidence):
            weak_tests += 1
            weak = weak and math.isfinite(route_confidence) and route_confidence <= max_route_confidence

        edge_points = _coerce_float(
            signal_payload.get("de3_edge_points"),
            float("nan"),
        )
        if math.isfinite(edge_points):
            metrics["edge_points"] = float(edge_points)
        if math.isfinite(max_edge_points):
            weak_tests += 1
            weak = weak and math.isfinite(edge_points) and edge_points <= max_edge_points

        if not require_signal_weakness:
            return True, metrics
        if weak_tests <= 0:
            return False, metrics
        return bool(weak), metrics

    def _de3_admission_signal_is_weak(signal_payload: Optional[dict]) -> tuple[bool, dict]:
        return _de3_signal_is_weak_from_thresholds(
            signal_payload,
            require_signal_weakness=de3_admission_require_signal_weakness,
            max_execution_quality=de3_admission_max_execution_quality,
            max_entry_margin=de3_admission_max_entry_margin,
            max_route_confidence=de3_admission_max_route_confidence,
            max_edge_points=de3_admission_max_edge_points,
        )

    def _de3_intraday_regime_signal_is_weak(signal_payload: Optional[dict]) -> tuple[bool, dict]:
        return _de3_signal_is_weak_from_thresholds(
            signal_payload,
            require_signal_weakness=True,
            max_execution_quality=de3_intraday_regime_max_execution_quality,
            max_entry_margin=de3_intraday_regime_max_entry_margin,
            max_route_confidence=de3_intraday_regime_max_route_confidence,
            max_edge_points=de3_intraday_regime_max_edge_points,
        )

    def _de3_intraday_regime_component(value: float, threshold: float, weight: float) -> float:
        if (not math.isfinite(value)) or (not math.isfinite(threshold)) or threshold <= 0.0 or weight <= 0.0:
            return 0.0
        if value <= threshold:
            return 0.0
        return float(min(weight, weight * ((value - threshold) / threshold)))

    def _de3_intraday_route_bias(signal_payload: Optional[dict]) -> tuple[float, float, float]:
        route_scores = (
            signal_payload.get("de3_v4_route_scores", {})
            if isinstance(signal_payload, dict) and isinstance(signal_payload.get("de3_v4_route_scores", {}), dict)
            else {}
        )
        long_score = 0.0
        short_score = 0.0
        for raw_key, raw_value in route_scores.items():
            try:
                score_val = float(raw_value)
            except Exception:
                continue
            key = str(raw_key or "").strip().lower()
            if "long" in key:
                long_score += score_val
            elif "short" in key:
                short_score += score_val
        total = abs(long_score) + abs(short_score)
        if total <= 1e-9:
            return 0.0, float(long_score), float(short_score)
        return float((short_score - long_score) / total), float(long_score), float(short_score)

    def _apply_de3_backtest_intraday_regime_control(
        signal_payload: Optional[dict],
        signal_time: Optional[dt.datetime],
        *,
        bar_index: Optional[int] = None,
    ) -> bool:
        nonlocal de3_intraday_regime_checked
        nonlocal de3_intraday_regime_applied
        nonlocal de3_intraday_regime_blocked
        if not isinstance(signal_payload, dict):
            return True

        requested_size = max(1, _coerce_int(signal_payload.get("size", CONTRACTS), CONTRACTS))
        signal_payload["de3_intraday_regime_enabled"] = bool(de3_intraday_regime_enabled)
        signal_payload["de3_intraday_regime_requested_size"] = int(requested_size)
        signal_payload["de3_intraday_regime_final_size"] = int(requested_size)
        signal_payload["de3_intraday_regime_blocked"] = False
        signal_payload["de3_intraday_regime_applied"] = False
        signal_payload["de3_intraday_regime_action"] = "none"
        if not de3_intraday_regime_enabled:
            signal_payload["de3_intraday_regime_state"] = "disabled"
            return True
        if not _is_de3_v4_signal_for_sizing(signal_payload):
            signal_payload["de3_intraday_regime_state"] = "not_de3_v4"
            return True
        idx = _safe_idx(bar_index)
        if idx is None or signal_time is None or idx < 0 or idx >= len(close_arr):
            de3_intraday_regime_state_counts["missing_bar"] += 1
            signal_payload["de3_intraday_regime_state"] = "missing_bar"
            return True

        session_name = str(signal_payload.get("session") or full_session_arr[idx] or "").upper()
        signal_payload["de3_intraday_regime_session"] = session_name
        if de3_intraday_regime_apply_sessions and session_name not in de3_intraday_regime_apply_sessions:
            de3_intraday_regime_state_counts["session_skip"] += 1
            signal_payload["de3_intraday_regime_state"] = "session_skip"
            return True
        lane_name = str(signal_payload.get("de3_v4_selected_lane", "") or "").strip()
        signal_payload["de3_intraday_regime_lane"] = lane_name
        if de3_intraday_regime_apply_lanes and lane_name not in de3_intraday_regime_apply_lanes:
            de3_intraday_regime_state_counts["lane_skip"] += 1
            signal_payload["de3_intraday_regime_state"] = "lane_skip"
            return True

        atr20 = float(atr20_arr[idx]) if idx < len(atr20_arr) and math.isfinite(atr20_arr[idx]) else float("nan")
        if not math.isfinite(atr20) or atr20 <= 0.0:
            de3_intraday_regime_state_counts["missing_context"] += 1
            signal_payload["de3_intraday_regime_state"] = "missing_context"
            return True

        de3_intraday_regime_checked += 1
        close_val = float(close_arr[idx])
        minute_of_day = int(ny_minute_of_day_arr[idx])
        session_open = float(de3_session_open_arr[idx]) if math.isfinite(de3_session_open_arr[idx]) else float("nan")
        session_vwap = float(de3_session_vwap_arr[idx]) if math.isfinite(de3_session_vwap_arr[idx]) else float("nan")
        prior_high = float(de3_prior_session_high_arr[idx]) if math.isfinite(de3_prior_session_high_arr[idx]) else float("nan")
        prior_low = float(de3_prior_session_low_arr[idx]) if math.isfinite(de3_prior_session_low_arr[idx]) else float("nan")
        orh_val = float(ny_orh_arr[idx]) if math.isfinite(ny_orh_arr[idx]) else float("nan")
        orl_val = float(ny_orl_arr[idx]) if math.isfinite(ny_orl_arr[idx]) else float("nan")

        pressure_start = idx
        while (
            pressure_start > 0
            and (idx - pressure_start) < int(de3_intraday_regime_pressure_lookback)
            and de3_session_key_arr[pressure_start - 1] == de3_session_key_arr[idx]
        ):
            pressure_start -= 1
        pressure_window = close_arr[pressure_start : idx + 1]
        if pressure_window.size >= 2:
            deltas = np.diff(pressure_window)
            up_pressure = float(np.clip(deltas, 0.0, None).sum())
            down_pressure = float(np.clip(-deltas, 0.0, None).sum())
            net_return = float(pressure_window[-1] - pressure_window[0])
        else:
            up_pressure = 0.0
            down_pressure = 0.0
            net_return = 0.0
        pressure_total = up_pressure + down_pressure
        bear_pressure_balance = (
            float((down_pressure - up_pressure) / pressure_total) if pressure_total > 1e-9 else 0.0
        )
        bull_pressure_balance = -bear_pressure_balance
        net_return_atr = float(net_return / atr20) if atr20 > 0.0 else 0.0

        vwap_dist_atr = (
            float((close_val - session_vwap) / atr20)
            if math.isfinite(session_vwap) and atr20 > 0.0
            else float("nan")
        )
        slope_lookback = idx - int(de3_intraday_regime_vwap_slope_lookback)
        vwap_slope_atr = float("nan")
        if (
            slope_lookback >= 0
            and de3_session_key_arr[slope_lookback] == de3_session_key_arr[idx]
            and math.isfinite(de3_session_vwap_arr[slope_lookback])
            and math.isfinite(session_vwap)
            and atr20 > 0.0
        ):
            vwap_slope_atr = float((session_vwap - float(de3_session_vwap_arr[slope_lookback])) / atr20)
        session_move_atr = (
            float((close_val - session_open) / atr20)
            if math.isfinite(session_open) and atr20 > 0.0
            else float("nan")
        )

        gap_location = float("nan")
        gap_below_atr = 0.0
        gap_above_atr = 0.0
        if math.isfinite(session_open) and math.isfinite(prior_high) and math.isfinite(prior_low):
            prior_range = float(prior_high - prior_low)
            if prior_range > 1e-9:
                gap_location = float((session_open - prior_low) / prior_range)
            gap_below_atr = max(0.0, float((prior_low - session_open) / atr20))
            gap_above_atr = max(0.0, float((session_open - prior_high) / atr20))

        route_bias, route_long_score, route_short_score = _de3_intraday_route_bias(signal_payload)

        bearish_score = 0.0
        bullish_score = 0.0
        bearish_reasons: list[str] = []
        bullish_reasons: list[str] = []

        if math.isfinite(vwap_dist_atr):
            vwap_bear = _de3_intraday_regime_component(
                max(0.0, -vwap_dist_atr),
                de3_intraday_regime_vwap_dist_min_atr,
                de3_intraday_regime_vwap_dist_weight,
            )
            vwap_bull = _de3_intraday_regime_component(
                max(0.0, vwap_dist_atr),
                de3_intraday_regime_vwap_dist_min_atr,
                de3_intraday_regime_vwap_dist_weight,
            )
            bearish_score += vwap_bear
            bullish_score += vwap_bull
            if vwap_bear > 0.0:
                bearish_reasons.append(f"vwap_dist={vwap_dist_atr:.2f}")
            if vwap_bull > 0.0:
                bullish_reasons.append(f"vwap_dist={vwap_dist_atr:.2f}")

        if math.isfinite(vwap_slope_atr):
            slope_bear = _de3_intraday_regime_component(
                max(0.0, -vwap_slope_atr),
                de3_intraday_regime_vwap_slope_min_atr,
                de3_intraday_regime_vwap_slope_weight,
            )
            slope_bull = _de3_intraday_regime_component(
                max(0.0, vwap_slope_atr),
                de3_intraday_regime_vwap_slope_min_atr,
                de3_intraday_regime_vwap_slope_weight,
            )
            bearish_score += slope_bear
            bullish_score += slope_bull
            if slope_bear > 0.0:
                bearish_reasons.append(f"vwap_slope={vwap_slope_atr:.2f}")
            if slope_bull > 0.0:
                bullish_reasons.append(f"vwap_slope={vwap_slope_atr:.2f}")

        if math.isfinite(session_move_atr):
            move_bear = _de3_intraday_regime_component(
                max(0.0, -session_move_atr),
                de3_intraday_regime_session_move_min_atr,
                de3_intraday_regime_session_move_weight,
            )
            move_bull = _de3_intraday_regime_component(
                max(0.0, session_move_atr),
                de3_intraday_regime_session_move_min_atr,
                de3_intraday_regime_session_move_weight,
            )
            bearish_score += move_bear
            bullish_score += move_bull
            if move_bear > 0.0:
                bearish_reasons.append(f"session_move={session_move_atr:.2f}")
            if move_bull > 0.0:
                bullish_reasons.append(f"session_move={session_move_atr:.2f}")

        pressure_bear = _de3_intraday_regime_component(
            max(0.0, bear_pressure_balance),
            de3_intraday_regime_pressure_balance_min,
            de3_intraday_regime_pressure_weight,
        )
        pressure_bull = _de3_intraday_regime_component(
            max(0.0, bull_pressure_balance),
            de3_intraday_regime_pressure_balance_min,
            de3_intraday_regime_pressure_weight,
        )
        bearish_score += pressure_bear
        bullish_score += pressure_bull
        if pressure_bear > 0.0:
            bearish_reasons.append(f"pressure={bear_pressure_balance:.2f}")
        if pressure_bull > 0.0:
            bullish_reasons.append(f"pressure={bull_pressure_balance:.2f}")

        net_bear = _de3_intraday_regime_component(
            max(0.0, -net_return_atr),
            de3_intraday_regime_net_return_min_atr,
            de3_intraday_regime_net_return_weight,
        )
        net_bull = _de3_intraday_regime_component(
            max(0.0, net_return_atr),
            de3_intraday_regime_net_return_min_atr,
            de3_intraday_regime_net_return_weight,
        )
        bearish_score += net_bear
        bullish_score += net_bull
        if net_bear > 0.0:
            bearish_reasons.append(f"net_ret={net_return_atr:.2f}")
        if net_bull > 0.0:
            bullish_reasons.append(f"net_ret={net_return_atr:.2f}")

        if math.isfinite(gap_location):
            if gap_location < de3_intraday_regime_gap_location_low:
                contrib = float(
                    min(
                        de3_intraday_regime_gap_location_weight,
                        de3_intraday_regime_gap_location_weight
                        * ((de3_intraday_regime_gap_location_low - gap_location) / max(de3_intraday_regime_gap_location_low, 1e-6)),
                    )
                )
                bearish_score += contrib
                if contrib > 0.0:
                    bearish_reasons.append(f"gap_loc={gap_location:.2f}")
            elif gap_location > de3_intraday_regime_gap_location_high:
                contrib = float(
                    min(
                        de3_intraday_regime_gap_location_weight,
                        de3_intraday_regime_gap_location_weight
                        * ((gap_location - de3_intraday_regime_gap_location_high) / max(1.0 - de3_intraday_regime_gap_location_high, 1e-6)),
                    )
                )
                bullish_score += contrib
                if contrib > 0.0:
                    bullish_reasons.append(f"gap_loc={gap_location:.2f}")
        if gap_below_atr > 0.0:
            contrib = float(
                min(
                    de3_intraday_regime_gap_outside_weight,
                    de3_intraday_regime_gap_outside_weight
                    * (gap_below_atr / de3_intraday_regime_gap_outside_scale_atr),
                )
            )
            bearish_score += contrib
            bearish_reasons.append(f"gap_below={gap_below_atr:.2f}")
        if gap_above_atr > 0.0:
            contrib = float(
                min(
                    de3_intraday_regime_gap_outside_weight,
                    de3_intraday_regime_gap_outside_weight
                    * (gap_above_atr / de3_intraday_regime_gap_outside_scale_atr),
                )
            )
            bullish_score += contrib
            bullish_reasons.append(f"gap_above={gap_above_atr:.2f}")

        if route_bias > 0.0:
            contrib = _de3_intraday_regime_component(
                route_bias,
                de3_intraday_regime_route_bias_min,
                de3_intraday_regime_route_bias_weight,
            )
            bearish_score += contrib
            if contrib > 0.0:
                bearish_reasons.append(f"route_bias={route_bias:.2f}")
        elif route_bias < 0.0:
            contrib = _de3_intraday_regime_component(
                -route_bias,
                de3_intraday_regime_route_bias_min,
                de3_intraday_regime_route_bias_weight,
            )
            bullish_score += contrib
            if contrib > 0.0:
                bullish_reasons.append(f"route_bias={route_bias:.2f}")

        if session_name in {"NY_AM", "NY_PM"} and minute_of_day >= (570 + int(de3_intraday_regime_or_minutes)):
            if math.isfinite(orh_val) and math.isfinite(orl_val):
                or_mid = (orh_val + orl_val) / 2.0
                bear_break_atr = max(0.0, float((orl_val - close_val) / atr20))
                bull_break_atr = max(0.0, float((close_val - orh_val) / atr20))
                if close_val < or_mid and bear_break_atr > 0.0:
                    contrib = float(
                        min(
                            de3_intraday_regime_or_weight,
                            de3_intraday_regime_or_weight
                            * (bear_break_atr / de3_intraday_regime_or_scale_atr),
                        )
                    )
                    bearish_score += contrib
                    bearish_reasons.append(f"or_break={bear_break_atr:.2f}")
                if close_val > or_mid and bull_break_atr > 0.0:
                    contrib = float(
                        min(
                            de3_intraday_regime_or_weight,
                            de3_intraday_regime_or_weight
                            * (bull_break_atr / de3_intraday_regime_or_scale_atr),
                        )
                    )
                    bullish_score += contrib
                    bullish_reasons.append(f"or_break={bull_break_atr:.2f}")

        weak_signal, weak_metrics = _de3_intraday_regime_signal_is_weak(signal_payload)
        signal_payload["de3_intraday_regime_signal_weak"] = bool(weak_signal)
        for metric_name, metric_value in weak_metrics.items():
            signal_payload[f"de3_intraday_regime_{metric_name}"] = metric_value

        entry_model_score = _coerce_float(signal_payload.get("de3_v4_entry_model_score"), float("nan"))
        entry_model_threshold = _coerce_float(signal_payload.get("de3_v4_entry_model_threshold"), float("nan"))
        entry_margin = (
            float(entry_model_score - entry_model_threshold)
            if math.isfinite(entry_model_score) and math.isfinite(entry_model_threshold)
            else float("nan")
        )
        execution_quality = _coerce_float(signal_payload.get("de3_v4_execution_quality_score"), float("nan"))
        route_confidence = _coerce_float(signal_payload.get("de3_v4_route_confidence"), float("nan"))
        strong_relief = 0.0
        relief_hits = 0
        if math.isfinite(execution_quality) and execution_quality >= de3_intraday_regime_strong_quality:
            strong_relief += de3_intraday_regime_strong_relief / 3.0
            relief_hits += 1
        if math.isfinite(entry_margin) and entry_margin >= de3_intraday_regime_strong_margin:
            strong_relief += de3_intraday_regime_strong_relief / 3.0
            relief_hits += 1
        if math.isfinite(route_confidence) and route_confidence >= de3_intraday_regime_strong_route:
            strong_relief += de3_intraday_regime_strong_relief / 3.0
            relief_hits += 1
        if relief_hits > 0:
            strong_relief = min(de3_intraday_regime_strong_relief, strong_relief)

        signal_side = str(signal_payload.get("side", "") or "").upper()
        bias_gap = float(bearish_score - bullish_score)
        block = False
        defensive = False
        state = "neutral"
        action = "none"
        countertrend_pressure = 0.0
        reasons: list[str] = []

        if signal_side == "LONG":
            countertrend_pressure = max(0.0, float(bearish_score - strong_relief))
            if (
                countertrend_pressure >= de3_intraday_regime_block_score
                and bias_gap >= de3_intraday_regime_block_dominance
                and (weak_signal or not de3_intraday_regime_require_weak_block)
                and de3_intraday_regime_mode in {"block", "block_defensive"}
            ):
                block = True
                state = "bearish_blocked"
                action = "block"
                reasons = list(bearish_reasons)
            elif (
                countertrend_pressure >= de3_intraday_regime_defensive_score
                and bias_gap >= de3_intraday_regime_dominance
                and (weak_signal or not de3_intraday_regime_require_weak_defensive)
                and de3_intraday_regime_mode in {"defensive", "block_defensive"}
            ):
                defensive = True
                state = "bearish_defensive"
                action = "defensive"
                reasons = list(bearish_reasons)
            elif bearish_score >= bullish_score + de3_intraday_regime_dominance:
                state = "bearish_aligned"
        elif signal_side == "SHORT" and de3_intraday_regime_enable_bullish_mirror:
            countertrend_pressure = max(0.0, float(bullish_score - strong_relief))
            if (
                countertrend_pressure >= de3_intraday_regime_block_score
                and (-bias_gap) >= de3_intraday_regime_block_dominance
                and (weak_signal or not de3_intraday_regime_require_weak_block)
                and de3_intraday_regime_mode in {"block", "block_defensive"}
            ):
                block = True
                state = "bullish_blocked"
                action = "block"
                reasons = list(bullish_reasons)
            elif (
                countertrend_pressure >= de3_intraday_regime_defensive_score
                and (-bias_gap) >= de3_intraday_regime_dominance
                and (weak_signal or not de3_intraday_regime_require_weak_defensive)
                and de3_intraday_regime_mode in {"defensive", "block_defensive"}
            ):
                defensive = True
                state = "bullish_defensive"
                action = "defensive"
                reasons = list(bullish_reasons)
            elif bullish_score >= bearish_score + de3_intraday_regime_dominance:
                state = "bullish_aligned"

        signal_payload["de3_intraday_regime_bearish_score"] = float(bearish_score)
        signal_payload["de3_intraday_regime_bullish_score"] = float(bullish_score)
        signal_payload["de3_intraday_regime_bias_gap"] = float(bias_gap)
        signal_payload["de3_intraday_regime_countertrend_pressure"] = float(countertrend_pressure)
        signal_payload["de3_intraday_regime_strong_signal_relief"] = float(strong_relief)
        signal_payload["de3_intraday_regime_vwap_dist_atr"] = (
            float(vwap_dist_atr) if math.isfinite(vwap_dist_atr) else None
        )
        signal_payload["de3_intraday_regime_vwap_slope_atr"] = (
            float(vwap_slope_atr) if math.isfinite(vwap_slope_atr) else None
        )
        signal_payload["de3_intraday_regime_session_move_atr"] = (
            float(session_move_atr) if math.isfinite(session_move_atr) else None
        )
        signal_payload["de3_intraday_regime_gap_location"] = (
            float(gap_location) if math.isfinite(gap_location) else None
        )
        signal_payload["de3_intraday_regime_gap_below_atr"] = float(gap_below_atr)
        signal_payload["de3_intraday_regime_gap_above_atr"] = float(gap_above_atr)
        signal_payload["de3_intraday_regime_pressure_balance"] = float(bear_pressure_balance)
        signal_payload["de3_intraday_regime_net_return_atr"] = float(net_return_atr)
        signal_payload["de3_intraday_regime_route_bias"] = float(route_bias)
        signal_payload["de3_intraday_regime_route_long_score"] = float(route_long_score)
        signal_payload["de3_intraday_regime_route_short_score"] = float(route_short_score)
        signal_payload["de3_intraday_regime_bearish_reasons"] = list(bearish_reasons)
        signal_payload["de3_intraday_regime_bullish_reasons"] = list(bullish_reasons)
        signal_payload["de3_intraday_regime_state"] = str(state)
        signal_payload["de3_intraday_regime_action"] = str(action)

        if block:
            de3_intraday_regime_state_counts[str(state)] += 1
            de3_intraday_regime_action_counts["block"] += 1
            de3_intraday_regime_applied += 1
            de3_intraday_regime_blocked += 1
            signal_payload["de3_intraday_regime_applied"] = True
            signal_payload["de3_intraday_regime_blocked"] = True
            signal_payload["de3_intraday_regime_final_size"] = 0
            signal_payload["de3_intraday_regime_reasons"] = list(reasons)
            return False

        if defensive:
            scaled_size = int(round(float(requested_size) * max(0.0, de3_intraday_regime_defensive_mult)))
            scaled_size = max(de3_intraday_regime_min_contracts, scaled_size)
            if de3_intraday_regime_reduce_only:
                scaled_size = min(int(requested_size), int(scaled_size))
            applied = bool(int(scaled_size) != int(requested_size))
            de3_intraday_regime_state_counts[str(state)] += 1
            de3_intraday_regime_action_counts["defensive"] += 1
            signal_payload["de3_intraday_regime_final_size"] = int(scaled_size)
            signal_payload["de3_intraday_regime_applied"] = bool(applied)
            signal_payload["de3_intraday_regime_reasons"] = list(reasons)
            signal_payload["size"] = int(scaled_size)
            if applied:
                de3_intraday_regime_applied += 1
            return True

        de3_intraday_regime_state_counts[str(state)] += 1
        signal_payload["de3_intraday_regime_reasons"] = list(reasons)
        return True

    def _de3_walkforward_gate_period_for_time(
        signal_time: Optional[dt.datetime],
    ) -> Optional[dict]:
        if not de3_walkforward_gate_enabled or signal_time is None:
            return None
        for period in de3_walkforward_gate_compiled_periods:
            start_ts = period.get("_start_ts")
            end_ts = period.get("_end_ts")
            if isinstance(start_ts, dt.datetime) and isinstance(end_ts, dt.datetime) and start_ts <= signal_time <= end_ts:
                return period
        return None

    def _load_de3_walkforward_gate_model(period: Optional[dict]) -> dict:
        if not isinstance(period, dict):
            return {}
        model_path_raw = str(period.get("model_path", "") or "").strip()
        if not model_path_raw:
            return {}
        model_path = Path(model_path_raw).expanduser()
        if not model_path.is_absolute():
            if de3_walkforward_gate_artifact_path is not None:
                model_path = (de3_walkforward_gate_artifact_path.parent / model_path).resolve()
            else:
                model_path = (Path(__file__).resolve().parent / model_path).resolve()
        cache_key = str(model_path)
        if cache_key in de3_walkforward_gate_model_cache:
            return de3_walkforward_gate_model_cache.get(cache_key, {})
        bundle = load_de3_walkforward_model_bundle(model_path)
        if bundle and isinstance(bundle.get("model_columns"), list):
            bundle["model_column_index"] = {
                str(col): idx for idx, col in enumerate(bundle.get("model_columns", []) or [])
            }
        if bundle:
            de3_walkforward_gate_model_cache[cache_key] = bundle
        return bundle

    def _apply_de3_backtest_walkforward_gate(
        signal_payload: Optional[dict],
        signal_time: Optional[dt.datetime],
    ) -> bool:
        nonlocal de3_walkforward_gate_checked
        nonlocal de3_walkforward_gate_applied
        nonlocal de3_walkforward_gate_blocked
        if not isinstance(signal_payload, dict):
            return True

        requested_size = max(1, _coerce_int(signal_payload.get("size", CONTRACTS), CONTRACTS))
        signal_payload["de3_walkforward_gate_enabled"] = bool(de3_walkforward_gate_enabled)
        signal_payload["de3_walkforward_gate_requested_size"] = int(requested_size)
        signal_payload["de3_walkforward_gate_final_size"] = int(requested_size)
        signal_payload["de3_walkforward_gate_blocked"] = False
        if not de3_walkforward_gate_enabled:
            signal_payload["de3_walkforward_gate_state"] = "disabled"
            signal_payload["de3_walkforward_gate_action"] = "none"
            signal_payload["de3_walkforward_gate_applied"] = False
            return True
        if not _is_de3_v4_signal_for_sizing(signal_payload):
            signal_payload["de3_walkforward_gate_state"] = "not_de3_v4"
            signal_payload["de3_walkforward_gate_action"] = "none"
            signal_payload["de3_walkforward_gate_applied"] = False
            return True
        if signal_time is None:
            signal_payload["de3_walkforward_gate_state"] = "missing_time"
            signal_payload["de3_walkforward_gate_action"] = "none"
            signal_payload["de3_walkforward_gate_applied"] = False
            return True

        de3_walkforward_gate_checked += 1
        period = _de3_walkforward_gate_period_for_time(signal_time)
        if not isinstance(period, dict):
            de3_walkforward_gate_state_counts["no_period"] += 1
            signal_payload["de3_walkforward_gate_state"] = "no_period"
            signal_payload["de3_walkforward_gate_action"] = "none"
            signal_payload["de3_walkforward_gate_applied"] = False
            return True
        period_name = str(period.get("name", "") or "")
        signal_payload["de3_walkforward_gate_period"] = period_name
        if period_name:
            de3_walkforward_gate_period_hits[period_name] += 1

        bundle = _load_de3_walkforward_gate_model(period)
        model = bundle.get("model")
        model_columns = bundle.get("model_columns", []) if isinstance(bundle, dict) else []
        model_column_index = bundle.get("model_column_index", {}) if isinstance(bundle, dict) else {}
        if model is None:
            de3_walkforward_gate_state_counts["missing_model"] += 1
            signal_payload["de3_walkforward_gate_state"] = "missing_model"
            signal_payload["de3_walkforward_gate_action"] = "none"
            signal_payload["de3_walkforward_gate_applied"] = False
            return True

        feature_row = build_de3_walkforward_feature_row(
            signal_payload,
            timestamp=signal_time,
            lane_ctx_history=de3_walkforward_gate_lane_history,
            variant_history=de3_walkforward_gate_variant_history,
        )
        x_vec = build_de3_walkforward_model_vector(
            feature_row,
            model_columns=model_columns,
            model_column_index=model_column_index,
        )
        try:
            predicted_ev = float(model.predict(x_vec)[0])
        except Exception:
            predicted_ev = float("nan")
        signal_payload["de3_walkforward_gate_predicted_ev_per_contract"] = (
            float(predicted_ev) if math.isfinite(predicted_ev) else None
        )

        raw_block_threshold = period.get("block_threshold")
        raw_def_threshold = period.get("defensive_threshold")
        block_threshold = (
            float("nan")
            if raw_block_threshold in (None, "")
            else _coerce_float(raw_block_threshold, float("nan"))
        )
        def_threshold = (
            float("nan")
            if raw_def_threshold in (None, "")
            else _coerce_float(raw_def_threshold, float("nan"))
        )
        def_mult = max(
            0.0,
            _coerce_float(
                period.get("defensive_size_multiplier", de3_walkforward_gate_defensive_mult),
                de3_walkforward_gate_defensive_mult,
            ),
        )
        signal_payload["de3_walkforward_gate_block_threshold"] = (
            float(block_threshold) if math.isfinite(block_threshold) else None
        )
        signal_payload["de3_walkforward_gate_defensive_threshold"] = (
            float(def_threshold) if math.isfinite(def_threshold) else None
        )
        signal_payload["de3_walkforward_gate_defensive_size_multiplier"] = float(def_mult)

        if not math.isfinite(predicted_ev):
            de3_walkforward_gate_state_counts["nan_prediction"] += 1
            signal_payload["de3_walkforward_gate_state"] = "nan_prediction"
            signal_payload["de3_walkforward_gate_action"] = "none"
            signal_payload["de3_walkforward_gate_applied"] = False
            return True

        can_block = de3_walkforward_gate_mode in {"block", "block_defensive"}
        can_defend = de3_walkforward_gate_mode in {"defensive", "block_defensive"}
        if can_block and math.isfinite(block_threshold) and predicted_ev <= float(block_threshold):
            de3_walkforward_gate_state_counts["blocked"] += 1
            de3_walkforward_gate_applied += 1
            de3_walkforward_gate_blocked += 1
            signal_payload["de3_walkforward_gate_state"] = "blocked"
            signal_payload["de3_walkforward_gate_action"] = "block"
            signal_payload["de3_walkforward_gate_applied"] = True
            signal_payload["de3_walkforward_gate_blocked"] = True
            signal_payload["de3_walkforward_gate_final_size"] = 0
            return False

        if can_defend and math.isfinite(def_threshold) and predicted_ev <= float(def_threshold):
            scaled_size = int(round(float(requested_size) * float(def_mult)))
            scaled_size = max(de3_walkforward_gate_min_contracts, scaled_size)
            if de3_walkforward_gate_reduce_only:
                scaled_size = min(int(requested_size), int(scaled_size))
            applied = bool(int(scaled_size) != int(requested_size))
            de3_walkforward_gate_state_counts["defensive"] += 1
            signal_payload["de3_walkforward_gate_state"] = "defensive"
            signal_payload["de3_walkforward_gate_action"] = "defensive"
            signal_payload["de3_walkforward_gate_applied"] = bool(applied)
            signal_payload["de3_walkforward_gate_final_size"] = int(scaled_size)
            signal_payload["size"] = int(scaled_size)
            if applied:
                de3_walkforward_gate_applied += 1
            return True

        de3_walkforward_gate_state_counts["neutral"] += 1
        signal_payload["de3_walkforward_gate_state"] = "neutral"
        signal_payload["de3_walkforward_gate_action"] = "none"
        signal_payload["de3_walkforward_gate_applied"] = False
        return True

    def _de3_signal_size_rule_match(
        signal_payload: Optional[dict],
        rule: Optional[dict],
    ) -> list[str]:
        if not isinstance(signal_payload, dict) or not isinstance(rule, dict):
            return []
        if not bool(rule.get("enabled", True)):
            return []
        if bool(rule.get("live_only", False)):
            return []

        def _match_str_set(raw_values, value: str, *, lower: bool = False) -> bool:
            if not isinstance(raw_values, (list, tuple, set)):
                return True
            normalized = {
                (str(item).strip().lower() if lower else str(item).strip())
                for item in raw_values
                if str(item).strip()
            }
            if not normalized:
                return True
            key = str(value or "").strip().lower() if lower else str(value or "").strip()
            return key in normalized

        variant_id = str(
            signal_payload.get("de3_v4_selected_variant_id")
            or signal_payload.get("sub_strategy")
            or ""
        ).strip()
        lane_name = str(signal_payload.get("de3_v4_selected_lane", "") or "").strip()
        side_name = str(signal_payload.get("side", "") or "").strip().upper()
        session_name = str(signal_payload.get("session", "") or "").strip()
        timeframe_name = str(signal_payload.get("de3_timeframe", "") or "").strip()
        if not _match_str_set(rule.get("apply_lanes", []), lane_name, lower=True):
            return []
        if not _match_str_set(rule.get("apply_sessions", []), session_name):
            return []
        if not _match_str_set(rule.get("apply_variants", []), variant_id):
            return []
        if not _match_str_set(rule.get("apply_sides", []), side_name):
            return []
        if not _match_str_set(rule.get("apply_timeframes", []), timeframe_name, lower=True):
            return []

        upper_wick = _coerce_float(signal_payload.get("de3_entry_upper_wick_ratio"), float("nan"))
        lower_wick = _coerce_float(signal_payload.get("de3_entry_lower_wick_ratio"), float("nan"))
        close_pos1 = _coerce_float(signal_payload.get("de3_entry_close_pos1"), float("nan"))
        body1_ratio = _coerce_float(signal_payload.get("de3_entry_body1_ratio"), float("nan"))
        vol1_rel20 = _coerce_float(signal_payload.get("de3_entry_vol1_rel20"), float("nan"))
        ret1_atr = _coerce_float(signal_payload.get("de3_entry_ret1_atr"), float("nan"))
        range10_atr = _coerce_float(signal_payload.get("de3_entry_range10_atr"), float("nan"))
        dist_high5_atr = _coerce_float(signal_payload.get("de3_entry_dist_high5_atr"), float("nan"))
        dist_low5_atr = _coerce_float(signal_payload.get("de3_entry_dist_low5_atr"), float("nan"))
        selection_score = _coerce_float(signal_payload.get("de3_selection_score"), float("nan"))
        final_score = _coerce_float(signal_payload.get("de3_final_score"), float("nan"))
        route_conf = _coerce_float(signal_payload.get("de3_v4_route_confidence"), float("nan"))

        checks = [
            ("min_upper_wick_ratio", upper_wick, ">="),
            ("max_upper_wick_ratio", upper_wick, "<="),
            ("min_lower_wick_ratio", lower_wick, ">="),
            ("max_lower_wick_ratio", lower_wick, "<="),
            ("min_close_pos1", close_pos1, ">="),
            ("max_close_pos1", close_pos1, "<="),
            ("min_body1_ratio", body1_ratio, ">="),
            ("max_body1_ratio", body1_ratio, "<="),
            ("min_vol1_rel20", vol1_rel20, ">="),
            ("max_vol1_rel20", vol1_rel20, "<="),
            ("min_ret1_atr", ret1_atr, ">="),
            ("max_ret1_atr", ret1_atr, "<="),
            ("min_range10_atr", range10_atr, ">="),
            ("max_range10_atr", range10_atr, "<="),
            ("min_dist_high5_atr", dist_high5_atr, ">="),
            ("max_dist_high5_atr", dist_high5_atr, "<="),
            ("min_dist_low5_atr", dist_low5_atr, ">="),
            ("max_dist_low5_atr", dist_low5_atr, "<="),
            ("min_selection_score", selection_score, ">="),
            ("max_selection_score", selection_score, "<="),
            ("min_final_score", final_score, ">="),
            ("max_final_score", final_score, "<="),
            ("min_route_confidence", route_conf, ">="),
            ("max_route_confidence", route_conf, "<="),
        ]
        reasons: list[str] = []
        condition_count = 0
        for key, actual, direction in checks:
            raw_threshold = rule.get(key)
            if raw_threshold in (None, ""):
                continue
            threshold = _coerce_float(raw_threshold, float("nan"))
            if not math.isfinite(threshold) or not math.isfinite(actual):
                return []
            condition_count += 1
            if direction == ">=" and actual < threshold:
                return []
            if direction == "<=" and actual > threshold:
                return []
            reasons.append(f"{key}={actual:.4f}")
        if condition_count <= 0:
            return []
        return reasons

    def _apply_de3_backtest_entry_model_margin_size(
        signal_payload: Optional[dict],
        requested_size: int,
    ) -> int:
        nonlocal de3_entry_margin_checked
        nonlocal de3_entry_margin_applied
        if not isinstance(signal_payload, dict):
            return int(requested_size)
        requested_size = max(1, _coerce_int(requested_size, CONTRACTS))
        signal_payload["de3_entry_margin_enabled"] = bool(de3_entry_margin_enabled)
        signal_payload["de3_entry_margin_requested_size"] = int(requested_size)
        signal_payload["de3_entry_margin_final_size"] = int(requested_size)
        signal_payload["de3_entry_margin_applied"] = False
        if not de3_entry_margin_enabled:
            signal_payload["de3_entry_margin_state"] = "disabled"
            return int(requested_size)
        if not _is_de3_v4_signal_for_sizing(signal_payload):
            signal_payload["de3_entry_margin_state"] = "not_de3_v4"
            return int(requested_size)

        de3_entry_margin_checked += 1
        score = _coerce_float(signal_payload.get("de3_v4_entry_model_score"), float("nan"))
        threshold = _coerce_float(signal_payload.get("de3_v4_entry_model_threshold"), float("nan"))
        if (not math.isfinite(score)) or (not math.isfinite(threshold)):
            de3_entry_margin_state_counts["missing_margin"] += 1
            signal_payload["de3_entry_margin_state"] = "missing_margin"
            signal_payload["de3_entry_margin_score"] = (
                float(score) if math.isfinite(score) else None
            )
            signal_payload["de3_entry_margin_threshold"] = (
                float(threshold) if math.isfinite(threshold) else None
            )
            signal_payload["de3_entry_margin_margin"] = None
            return int(requested_size)

        margin = float(score - threshold)
        tier = str(signal_payload.get("de3_v4_entry_model_tier", "") or "").strip().lower()
        scope = str(signal_payload.get("de3_v4_entry_model_scope", "") or "").strip().lower()
        signal_payload["de3_entry_margin_score"] = float(score)
        signal_payload["de3_entry_margin_threshold"] = float(threshold)
        signal_payload["de3_entry_margin_margin"] = float(margin)
        signal_payload["de3_entry_margin_tier"] = str(tier)
        signal_payload["de3_entry_margin_scope"] = str(scope)

        multiplier = 1.0
        reasons: list[str] = []
        state = "neutral"

        if math.isfinite(de3_entry_margin_defensive_max) and margin <= float(de3_entry_margin_defensive_max):
            multiplier = min(multiplier, float(de3_entry_margin_defensive_mult))
            reasons.append("low_margin")
        if scope in {"lane", "global", "missing"}:
            multiplier = min(multiplier, float(de3_entry_margin_lane_scope_mult))
            reasons.append(f"{scope}_scope")
        if tier == "conservative":
            multiplier = min(multiplier, float(de3_entry_margin_conservative_mult))
            reasons.append("conservative_tier")

        aggressive_allowed = not reasons
        if de3_entry_margin_aggressive_variant_only and scope != "variant":
            aggressive_allowed = False
        if aggressive_allowed and math.isfinite(de3_entry_margin_aggressive_min) and margin >= float(de3_entry_margin_aggressive_min):
            multiplier = max(multiplier, float(de3_entry_margin_aggressive_mult))
            reasons.append("high_margin")

        scaled_size = int(round(float(requested_size) * max(0.0, float(multiplier))))
        scaled_size = max(de3_entry_margin_min_contracts, scaled_size)
        scaled_size = min(de3_entry_margin_max_contracts, scaled_size)
        if de3_entry_margin_reduce_only:
            scaled_size = min(int(requested_size), int(scaled_size))
        applied = bool(int(scaled_size) != int(requested_size))

        if "high_margin" in reasons:
            state = "aggressive"
        elif reasons:
            state = "defensive"
        signal_payload["de3_entry_margin_multiplier"] = float(multiplier)
        signal_payload["de3_entry_margin_reasons"] = list(reasons)
        signal_payload["de3_entry_margin_final_size"] = int(scaled_size)
        signal_payload["de3_entry_margin_applied"] = bool(applied)
        signal_payload["de3_entry_margin_state"] = str(state)
        if applied:
            de3_entry_margin_applied += 1
        de3_entry_margin_state_counts[str(state)] += 1
        return int(scaled_size)

    def _apply_de3_backtest_signal_size_rules(
        signal_payload: Optional[dict],
        size: int,
    ) -> int:
        nonlocal de3_signal_size_applied
        if not de3_signal_size_enabled or not isinstance(signal_payload, dict):
            return int(size)
        current_size = max(1, int(size))
        signal_payload["de3_signal_size_rules_requested_size"] = int(current_size)
        signal_payload["de3_signal_size_rules_applied"] = False
        signal_payload["de3_signal_size_rule_names"] = []
        signal_payload["de3_signal_size_rule_reasons"] = []
        for rule in de3_signal_size_rules:
            reasons = _de3_signal_size_rule_match(signal_payload, rule)
            if not reasons:
                continue
            rule_name = str(
                rule.get("name", "unnamed_signal_size_rule") or "unnamed_signal_size_rule"
            ).strip()
            size_multiplier = max(
                0.0,
                _coerce_float(rule.get("size_multiplier", 1.0), 1.0),
            )
            rule_min_contracts = max(
                1,
                _coerce_int(rule.get("min_contracts", 1), 1),
            )
            candidate_size = int(math.floor(float(current_size) * float(size_multiplier)))
            if candidate_size < rule_min_contracts:
                candidate_size = int(rule_min_contracts)
            if candidate_size < 1:
                candidate_size = 1
            if candidate_size >= current_size:
                continue
            signal_payload["de3_signal_size_rules_applied"] = True
            signal_payload.setdefault("de3_signal_size_rule_names", []).append(rule_name)
            signal_payload.setdefault("de3_signal_size_rule_reasons", []).append(
                {"name": rule_name, "reasons": list(reasons)}
            )
            signal_payload["de3_signal_size_rule_last_name"] = rule_name
            signal_payload["de3_signal_size_rule_last_multiplier"] = float(size_multiplier)
            current_size = int(candidate_size)
            de3_signal_size_applied += 1
            de3_signal_size_rule_hits[rule_name] += 1
            if de3_signal_size_log_applies:
                logging.info(
                    "DE3 signal-size rule | rule=%s size->%s sub=%s side=%s",
                    rule_name,
                    current_size,
                    signal_payload.get("sub_strategy"),
                    signal_payload.get("side"),
                )
        signal_payload["de3_signal_size_rules_final_size"] = int(current_size)
        return int(current_size)

    def _apply_de3_backtest_admission_control(signal_payload: Optional[dict]) -> bool:
        nonlocal de3_admission_checked
        nonlocal de3_admission_applied
        nonlocal de3_admission_blocked
        if not isinstance(signal_payload, dict):
            return True

        requested_size = max(1, _coerce_int(signal_payload.get("size", CONTRACTS), CONTRACTS))
        signal_payload["de3_admission_enabled"] = bool(de3_admission_enabled)
        signal_payload["de3_admission_requested_size"] = int(requested_size)
        signal_payload["de3_admission_final_size"] = int(requested_size)
        signal_payload["de3_admission_blocked"] = False
        if not de3_admission_enabled:
            signal_payload["de3_admission_state"] = "disabled"
            signal_payload["de3_admission_action"] = "none"
            signal_payload["de3_admission_applied"] = False
            return True
        if not _is_de3_v4_signal_for_sizing(signal_payload):
            signal_payload["de3_admission_state"] = "not_de3_v4"
            signal_payload["de3_admission_action"] = "none"
            signal_payload["de3_admission_applied"] = False
            return True

        de3_admission_checked += 1
        key = _de3_admission_key_from_payload(signal_payload)
        signal_payload["de3_admission_granularity"] = str(de3_admission_key_granularity)
        signal_payload["de3_admission_key"] = str(key)
        if not key:
            de3_admission_state_counts["missing_key"] += 1
            signal_payload["de3_admission_state"] = "missing_key"
            signal_payload["de3_admission_action"] = "none"
            signal_payload["de3_admission_applied"] = False
            return True

        history = de3_admission_recent_pnl_history[key]
        sample_count = len(history)
        signal_payload["de3_admission_history_trades"] = int(sample_count)
        signal_payload["de3_admission_history_window"] = int(de3_admission_history_window)
        if sample_count < de3_admission_warmup_trades:
            de3_admission_state_counts["warmup"] += 1
            signal_payload["de3_admission_state"] = "warmup"
            signal_payload["de3_admission_action"] = "none"
            signal_payload["de3_admission_applied"] = False
            return True

        recent_values = list(history)
        avg_net_per_contract = (
            float(sum(recent_values) / len(recent_values))
            if recent_values
            else 0.0
        )
        recent_winrate = (
            float(sum(1 for value in recent_values if value > 0.0) / len(recent_values))
            if recent_values
            else 0.0
        )
        signal_payload["de3_admission_avg_net_per_contract_usd"] = float(avg_net_per_contract)
        signal_payload["de3_admission_recent_winrate"] = float(recent_winrate)

        weak_signal, weak_metrics = _de3_admission_signal_is_weak(signal_payload)
        signal_payload["de3_admission_signal_weak"] = bool(weak_signal)
        for metric_name, metric_value in weak_metrics.items():
            signal_payload[f"de3_admission_{metric_name}"] = metric_value

        cold = (
            avg_net_per_contract <= de3_admission_cold_avg
            and recent_winrate <= de3_admission_cold_max_winrate
        )
        block = (
            math.isfinite(de3_admission_block_avg)
            and math.isfinite(de3_admission_block_max_winrate)
            and avg_net_per_contract <= de3_admission_block_avg
            and recent_winrate <= de3_admission_block_max_winrate
        )
        if de3_admission_require_signal_weakness and not weak_signal:
            cold = False
            block = False

        if block:
            de3_admission_state_counts["blocked"] += 1
            de3_admission_applied += 1
            de3_admission_blocked += 1
            de3_admission_key_actions[key] += 1
            signal_payload["de3_admission_state"] = "blocked"
            signal_payload["de3_admission_action"] = "block"
            signal_payload["de3_admission_applied"] = True
            signal_payload["de3_admission_blocked"] = True
            signal_payload["de3_admission_final_size"] = 0
            return False

        if cold:
            scaled_size = int(round(float(requested_size) * max(0.0, de3_admission_defensive_mult)))
            scaled_size = max(de3_admission_min_contracts, scaled_size)
            if de3_admission_reduce_only:
                scaled_size = min(int(requested_size), int(scaled_size))
            applied = bool(int(scaled_size) != int(requested_size))
            de3_admission_state_counts["cold_defensive"] += 1
            signal_payload["de3_admission_state"] = "cold_defensive"
            signal_payload["de3_admission_action"] = "defensive"
            signal_payload["de3_admission_applied"] = bool(applied)
            signal_payload["de3_admission_final_size"] = int(scaled_size)
            signal_payload["size"] = int(scaled_size)
            if applied:
                de3_admission_applied += 1
                de3_admission_key_actions[key] += 1
            return True

        de3_admission_state_counts["neutral"] += 1
        signal_payload["de3_admission_state"] = "neutral"
        signal_payload["de3_admission_action"] = "none"
        signal_payload["de3_admission_applied"] = False
        return True

    def _apply_de3_backtest_policy_context_overlay(
        signal_payload: Optional[dict],
        requested_size: int,
    ) -> int:
        nonlocal de3_policy_overlay_checked
        nonlocal de3_policy_overlay_applied
        if not isinstance(signal_payload, dict):
            return int(requested_size)
        requested_size = max(1, _coerce_int(requested_size, CONTRACTS))
        signal_payload["de3_policy_overlay_enabled"] = bool(de3_policy_overlay_enabled)
        signal_payload["de3_policy_overlay_requested_size"] = int(requested_size)
        signal_payload["de3_policy_overlay_final_size"] = int(requested_size)
        signal_payload["de3_policy_overlay_applied"] = False
        if not de3_policy_overlay_enabled:
            signal_payload["de3_policy_overlay_state"] = "disabled"
            return int(requested_size)
        if not _is_de3_v4_signal_for_sizing(signal_payload):
            signal_payload["de3_policy_overlay_state"] = "not_de3_v4"
            return int(requested_size)

        de3_policy_overlay_checked += 1
        policy_risk_mult = _coerce_float(signal_payload.get("de3_policy_risk_mult"), float("nan"))
        policy_confidence = _coerce_float(signal_payload.get("de3_policy_confidence"), float("nan"))
        policy_bucket_samples = max(
            0,
            _coerce_int(signal_payload.get("de3_policy_bucket_samples", 0), 0),
        )
        signal_payload["de3_policy_overlay_risk_mult"] = (
            float(policy_risk_mult) if math.isfinite(policy_risk_mult) else None
        )
        signal_payload["de3_policy_overlay_confidence"] = (
            float(policy_confidence) if math.isfinite(policy_confidence) else None
        )
        signal_payload["de3_policy_overlay_bucket_samples"] = int(policy_bucket_samples)

        if not math.isfinite(policy_risk_mult):
            de3_policy_overlay_state_counts["missing_policy"] += 1
            signal_payload["de3_policy_overlay_state"] = "missing_policy"
            return int(requested_size)
        if policy_risk_mult >= 1.0 - 1e-12:
            de3_policy_overlay_state_counts["neutral_mult"] += 1
            signal_payload["de3_policy_overlay_state"] = "neutral_mult"
            return int(requested_size)
        if (
            math.isfinite(de3_policy_overlay_min_confidence)
            and (
                (not math.isfinite(policy_confidence))
                or policy_confidence < float(de3_policy_overlay_min_confidence)
            )
        ):
            de3_policy_overlay_state_counts["low_confidence"] += 1
            signal_payload["de3_policy_overlay_state"] = "low_confidence"
            return int(requested_size)
        if policy_bucket_samples < de3_policy_overlay_min_bucket_samples:
            de3_policy_overlay_state_counts["low_samples"] += 1
            signal_payload["de3_policy_overlay_state"] = "low_samples"
            return int(requested_size)

        weak_signal, weak_metrics = _de3_signal_is_weak_from_thresholds(
            signal_payload,
            require_signal_weakness=de3_policy_overlay_require_signal_weakness,
            max_execution_quality=de3_policy_overlay_max_execution_quality,
            max_entry_margin=de3_policy_overlay_max_entry_margin,
            max_route_confidence=de3_policy_overlay_max_route_confidence,
            max_edge_points=de3_policy_overlay_max_edge_points,
        )
        signal_payload["de3_policy_overlay_signal_weak"] = bool(weak_signal)
        for metric_name, metric_value in weak_metrics.items():
            signal_payload[f"de3_policy_overlay_{metric_name}"] = metric_value
        if de3_policy_overlay_require_signal_weakness and not weak_signal:
            de3_policy_overlay_state_counts["strong_signal"] += 1
            signal_payload["de3_policy_overlay_state"] = "strong_signal"
            return int(requested_size)

        scaled_size = int(round(float(requested_size) * max(0.0, float(policy_risk_mult))))
        scaled_size = max(de3_policy_overlay_min_contracts, scaled_size)
        if de3_policy_overlay_reduce_only:
            scaled_size = min(int(requested_size), int(scaled_size))
        applied = bool(int(scaled_size) != int(requested_size))
        signal_payload["de3_policy_overlay_final_size"] = int(scaled_size)
        signal_payload["de3_policy_overlay_applied"] = bool(applied)
        if applied:
            de3_policy_overlay_applied += 1
            de3_policy_overlay_state_counts["applied"] += 1
            signal_payload["de3_policy_overlay_state"] = "applied"
        else:
            de3_policy_overlay_state_counts["no_change"] += 1
            signal_payload["de3_policy_overlay_state"] = "no_change"
        return int(scaled_size)

    def _apply_de3_backtest_variant_adaptation_size(
        signal_payload: Optional[dict],
        requested_size: int,
    ) -> int:
        nonlocal de3_variant_adapt_checked
        nonlocal de3_variant_adapt_applied
        if not isinstance(signal_payload, dict):
            return int(requested_size)
        requested_size = max(1, _coerce_int(requested_size, CONTRACTS))
        signal_payload["de3_variant_adapt_enabled"] = bool(de3_variant_adapt_enabled)
        signal_payload["de3_variant_adapt_requested_size"] = int(requested_size)
        signal_payload["de3_variant_adapt_final_size"] = int(requested_size)
        if not de3_variant_adapt_enabled:
            signal_payload["de3_variant_adapt_state"] = "disabled"
            signal_payload["de3_variant_adapt_applied"] = False
            return int(requested_size)
        if not _is_de3_v4_signal_for_sizing(signal_payload):
            signal_payload["de3_variant_adapt_state"] = "not_de3_v4"
            signal_payload["de3_variant_adapt_applied"] = False
            return int(requested_size)

        de3_variant_adapt_checked += 1
        variant_id = _de3_variant_id_from_payload(signal_payload)
        signal_payload["de3_variant_adapt_variant_id"] = variant_id
        if not variant_id:
            de3_variant_adapt_state_counts["missing_variant"] += 1
            signal_payload["de3_variant_adapt_state"] = "missing_variant"
            signal_payload["de3_variant_adapt_applied"] = False
            return int(requested_size)

        history = de3_variant_adapt_recent_pnl_history[variant_id]
        sample_count = len(history)
        signal_payload["de3_variant_adapt_history_trades"] = int(sample_count)
        signal_payload["de3_variant_adapt_history_window"] = int(de3_variant_adapt_history_window)
        if sample_count < de3_variant_adapt_warmup_trades:
            de3_variant_adapt_state_counts["warmup"] += 1
            signal_payload["de3_variant_adapt_state"] = "warmup"
            signal_payload["de3_variant_adapt_applied"] = False
            return int(requested_size)

        recent_values = list(history)
        avg_net_per_contract = float(sum(recent_values) / len(recent_values)) if recent_values else 0.0
        recent_winrate = (
            float(sum(1 for value in recent_values if value > 0.0) / len(recent_values))
            if recent_values
            else 0.0
        )
        lifetime_trade_count = int(de3_variant_adapt_lifetime_trade_count[variant_id])
        lifetime_avg_net_per_contract = (
            float(de3_variant_adapt_lifetime_pnl_sum[variant_id] / float(lifetime_trade_count))
            if lifetime_trade_count > 0
            else 0.0
        )
        lifetime_cap_active = math.isfinite(de3_variant_adapt_max_lifetime_avg)
        lifetime_cap_passed = (
            True
            if not lifetime_cap_active
            else lifetime_avg_net_per_contract <= float(de3_variant_adapt_max_lifetime_avg)
        )
        signal_payload["de3_variant_adapt_avg_net_per_contract_usd"] = float(avg_net_per_contract)
        signal_payload["de3_variant_adapt_recent_winrate"] = float(recent_winrate)
        signal_payload["de3_variant_adapt_lifetime_avg_net_per_contract_usd"] = float(
            lifetime_avg_net_per_contract
        )
        signal_payload["de3_variant_adapt_lifetime_trade_count"] = int(lifetime_trade_count)
        signal_payload["de3_variant_adapt_max_lifetime_avg_net_per_contract_usd"] = (
            float(de3_variant_adapt_max_lifetime_avg) if lifetime_cap_active else None
        )
        signal_payload["de3_variant_adapt_lifetime_cap_passed"] = bool(lifetime_cap_passed)

        state = "neutral"
        multiplier = 1.0
        if (
            math.isfinite(de3_variant_adapt_deep_cold_avg)
            and avg_net_per_contract <= de3_variant_adapt_deep_cold_avg
            and recent_winrate <= de3_variant_adapt_cold_max_winrate
            and lifetime_cap_passed
        ):
            state = "deep_cold"
            multiplier = float(de3_variant_adapt_deep_cold_mult)
        elif (
            avg_net_per_contract <= de3_variant_adapt_cold_avg
            and recent_winrate <= de3_variant_adapt_cold_max_winrate
            and lifetime_cap_passed
        ):
            state = "cold"
            multiplier = float(de3_variant_adapt_cold_mult)
        elif (
            lifetime_cap_active
            and avg_net_per_contract <= de3_variant_adapt_cold_avg
            and recent_winrate <= de3_variant_adapt_cold_max_winrate
            and not lifetime_cap_passed
        ):
            state = "lifetime_strong"

        scaled_size = int(round(float(requested_size) * max(0.0, float(multiplier))))
        scaled_size = max(de3_variant_adapt_min_contracts, scaled_size)
        if de3_variant_adapt_reduce_only:
            scaled_size = min(int(requested_size), int(scaled_size))

        applied = bool(int(scaled_size) != int(requested_size))
        de3_variant_adapt_state_counts[state] += 1
        if applied:
            de3_variant_adapt_applied += 1
            de3_variant_adapt_variant_reductions[variant_id] += 1
        signal_payload["de3_variant_adapt_state"] = str(state)
        signal_payload["de3_variant_adapt_multiplier"] = float(multiplier)
        signal_payload["de3_variant_adapt_applied"] = bool(applied)
        signal_payload["de3_variant_adapt_final_size"] = int(scaled_size)
        return int(scaled_size)

    def _record_de3_variant_adaptation_outcome(trade_payload: Optional[dict]) -> None:
        if not de3_variant_adapt_enabled or not isinstance(trade_payload, dict):
            return
        variant_id = _de3_variant_id_from_payload(trade_payload)
        size = _coerce_int(trade_payload.get("size", 0), 0)
        pnl_net = _coerce_float(trade_payload.get("pnl_net"), float("nan"))
        if not variant_id or size <= 0 or not math.isfinite(pnl_net):
            return
        per_contract_pnl = float(pnl_net) / float(size)
        de3_variant_adapt_recent_pnl_history[variant_id].append(per_contract_pnl)
        de3_variant_adapt_lifetime_pnl_sum[variant_id] += float(per_contract_pnl)
        de3_variant_adapt_lifetime_trade_count[variant_id] += 1

    def _record_de3_backtest_walkforward_gate_outcome(trade_payload: Optional[dict]) -> None:
        if not de3_walkforward_gate_enabled or not isinstance(trade_payload, dict):
            return
        size = _coerce_int(trade_payload.get("size", 0), 0)
        pnl_net = _coerce_float(trade_payload.get("pnl_net"), float("nan"))
        if size <= 0 or not math.isfinite(pnl_net):
            return
        per_contract_pnl = float(pnl_net) / float(size)
        lane_key = de3_walkforward_lane_context_key(trade_payload)
        variant_key = de3_walkforward_variant_key(trade_payload)
        if lane_key:
            de3_walkforward_gate_lane_history[lane_key].append(per_contract_pnl)
        if variant_key:
            de3_walkforward_gate_variant_history[variant_key].append(per_contract_pnl)

    def _record_de3_backtest_admission_outcome(trade_payload: Optional[dict]) -> None:
        if not de3_admission_enabled or not isinstance(trade_payload, dict):
            return
        key = _de3_admission_key_from_payload(trade_payload)
        size = _coerce_int(trade_payload.get("size", 0), 0)
        pnl_net = _coerce_float(trade_payload.get("pnl_net"), float("nan"))
        if not key or size <= 0 or not math.isfinite(pnl_net):
            return
        per_contract_pnl = float(pnl_net) / float(size)
        de3_admission_recent_pnl_history[key].append(per_contract_pnl)

    de3_trade_day_roll_hour_et = max(
        0,
        min(
            23,
            _coerce_int(
                de3_v4_entry_trade_day_extreme_cfg_local.get("trade_day_roll_hour_et"),
                18,
            ),
        ),
    )
    de3_trade_day_known_key = None
    de3_trade_day_high_known = float("nan")
    de3_trade_day_low_known = float("nan")

    for i in range(len(full_df)):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        history_end = i + 1
        history_start = 0
        if de3_only_fast_mode or de3_ml_fast_mode:
            history_start = max(0, history_end - de3_fast_history_bars)
        if manifold_only_fast_mode and manifold_fast_history_bars > 0:
            history_start = max(history_start, history_end - manifold_fast_history_bars)
        if aetherflow_only_fast_mode and aetherflow_fast_history_bars > 0:
            history_start = max(history_start, history_end - aetherflow_fast_history_bars)
        if backtest_max_history_bars > 0:
            history_start = max(history_start, history_end - backtest_max_history_bars)
        if dynamic_engine3_strat is not None and de3_history_cap_bars > 0:
            history_start = max(history_start, history_end - de3_history_cap_bars)
        history_df: Optional[pd.DataFrame] = None
        history_tail_cache: dict[int, pd.DataFrame] = {}

        current_time = full_index[i]
        current_hour_et = int(current_time.hour)
        current_minute_et = int(current_time.minute)
        entry_window_blocked = bool(
            BACKTEST_ENFORCE_NO_NEW_ENTRIES_WINDOW
            and _hour_in_window(
                current_hour_et,
                BACKTEST_NO_NEW_ENTRIES_START_HOUR_ET,
                BACKTEST_NO_NEW_ENTRIES_END_HOUR_ET,
            )
        )
        force_flat_now = bool(force_flat_arr[i])
        if need_mnq_slice:
            if mnq_pos is not None:
                mnq_end = int(mnq_pos[i])
                mnq_start = max(0, mnq_end - mnq_slice_tail_bars)
                mnq_slice = mnq_df.iloc[mnq_start:mnq_end]
            else:
                mnq_slice = slice_df_upto(mnq_df, current_time)
                if len(mnq_slice) > mnq_slice_tail_bars:
                    mnq_slice = mnq_slice.iloc[-mnq_slice_tail_bars:]
        else:
            mnq_slice = empty_mnq_slice
        if need_vix_slice:
            if vix_pos is not None:
                vix_end = int(vix_pos[i])
                vix_start = max(0, vix_end - vix_slice_tail_bars)
                vix_slice = vix_df.iloc[vix_start:vix_end]
            else:
                vix_slice = slice_df_upto(vix_df, current_time)
                if len(vix_slice) > vix_slice_tail_bars:
                    vix_slice = vix_slice.iloc[-vix_slice_tail_bars:]
        else:
            vix_slice = empty_vix_slice
        current_session = str(full_session_arr[i])
        session_name = current_session
        regime_meta = None
        if regime_manifold_engine is not None:
            if history_df is None:
                history_df = full_df.iloc[history_start:history_end]
            regime_meta = _refresh_regime_meta(history_df, current_time, session_name)
        holiday_closed_now = bool(holiday_flat_arr[i])
        bar_sl_multiplier = float(bar_sl_multiplier_arr[i])
        bar_tp_multiplier = float(bar_tp_multiplier_arr[i])
        bar_chop_multiplier = float(bar_chop_multiplier_arr[i])
        if abs(bar_sl_multiplier - active_sl_multiplier) > 1e-12:
            CONFIG["DYNAMIC_SL_MULTIPLIER"] = bar_sl_multiplier
            active_sl_multiplier = bar_sl_multiplier
            multiplier_updates += 1
        if abs(bar_tp_multiplier - active_tp_multiplier) > 1e-12:
            CONFIG["DYNAMIC_TP_MULTIPLIER"] = bar_tp_multiplier
            active_tp_multiplier = bar_tp_multiplier
            multiplier_updates += 1
        if abs(bar_chop_multiplier - active_chop_multiplier) > 1e-12:
            if chop_analyzer is not None:
                chop_analyzer.update_gemini_params(bar_chop_multiplier)
                multiplier_updates += 1
            active_chop_multiplier = bar_chop_multiplier
        ml_filter_bypass_bar_active = bool(
            ml_filter_bypass_enabled
            and (not ml_filter_bypass_sessions or current_session in ml_filter_bypass_sessions)
        )
        trend_session = str(full_trend_session_arr[i])
        bar_open = float(open_arr[i])
        bar_high = float(high_arr[i])
        bar_low = float(low_arr[i])
        bar_close = float(close_arr[i])
        current_de3_trade_day_key = _de3_trade_day_key(
            current_time,
            roll_hour_et=de3_trade_day_roll_hour_et,
        )
        if current_de3_trade_day_key != de3_trade_day_known_key:
            de3_trade_day_known_key = current_de3_trade_day_key
            de3_trade_day_high_known = float(bar_open)
            de3_trade_day_low_known = float(bar_open)
        else:
            if math.isfinite(bar_open):
                if math.isfinite(de3_trade_day_high_known):
                    de3_trade_day_high_known = max(float(de3_trade_day_high_known), float(bar_open))
                else:
                    de3_trade_day_high_known = float(bar_open)
                if math.isfinite(de3_trade_day_low_known):
                    de3_trade_day_low_known = min(float(de3_trade_day_low_known), float(bar_open))
                else:
                    de3_trade_day_low_known = float(bar_open)
        processed_bars += 1
        last_time = current_time
        last_close = bar_close
        if asia_ema_fast_state is None:
            asia_ema_fast_state = bar_close
        else:
            asia_ema_fast_state += asia_trend_alpha_fast * (bar_close - asia_ema_fast_state)
        if asia_ema_slow_state is None:
            asia_ema_slow_state = bar_close
        else:
            asia_ema_slow_state += asia_trend_alpha_slow * (bar_close - asia_ema_slow_state)
        asia_ema_obs += 1
        asia_ema_fast_window.append(float(asia_ema_fast_state))

        if asia_tf_enabled:
            while asia_tf_high_window and asia_tf_high_window[-1][1] <= bar_high:
                asia_tf_high_window.pop()
            asia_tf_high_window.append((i, bar_high))
            while asia_tf_low_window and asia_tf_low_window[-1][1] >= bar_low:
                asia_tf_low_window.pop()
            asia_tf_low_window.append((i, bar_low))
            _asia_cutoff = i - asia_tf_lookback
            while asia_tf_high_window and asia_tf_high_window[0][0] <= _asia_cutoff:
                asia_tf_high_window.popleft()
            while asia_tf_low_window and asia_tf_low_window[0][0] <= _asia_cutoff:
                asia_tf_low_window.popleft()
            if (i + 1) >= asia_tf_lookback and asia_tf_high_window and asia_tf_low_window:
                asia_tf_box_range_current = asia_tf_high_window[0][1] - asia_tf_low_window[0][1]
            else:
                asia_tf_box_range_current = math.nan

        if TREND_DAY_ENABLED and enabled_filter_trend_day_tier:
            if last_trend_session != trend_session:
                last_trend_session = trend_session
                trend_day_tier = 0
                trend_day_dir = None
                impulse_day = None
                impulse_active = False
                impulse_dir = None
                impulse_start_price = None
                impulse_extreme = None
                pullback_extreme = None
                max_retracement = 0.0
                bars_since_impulse = 0
                tier1_down_until = None
                tier1_up_until = None
                tier1_seen = False
                sticky_trend_dir = None
                sticky_reclaim_count = 0
                sticky_opposite_count = 0
                trend_day_max_sigma = 0.0
            day_key = td_day_index_arr[i]
            if impulse_day != day_key:
                impulse_day = day_key
                impulse_active = False
                impulse_dir = None
                impulse_start_price = None
                impulse_extreme = None
                pullback_extreme = None
                max_retracement = 0.0
                bars_since_impulse = 0
                tier1_down_until = None
                tier1_up_until = None
                tier1_seen = False
                sticky_trend_dir = None
                sticky_reclaim_count = 0
                sticky_opposite_count = 0
                trend_day_max_sigma = 0.0

            ema50_val = td_ema50_arr[i]
            ema200_val = td_ema200_arr[i]
            atr_expansion = td_atr_exp_arr[i]
            vwap_val = td_vwap_arr[i]
            vwap_sigma = td_vwap_sigma_arr[i]
            session_open = td_session_open_arr[i]
            prior_session_low = td_prior_session_low_arr[i]
            prior_session_high = td_prior_session_high_arr[i]
            trend_up_alt = bool(td_trend_up_alt_arr[i])
            trend_down_alt = bool(td_trend_down_alt_arr[i])
            adx_strong_up = bool(td_adx_strong_up_arr[i])
            adx_strong_down = bool(td_adx_strong_down_arr[i])

            trend_day_tier = 0
            trend_day_dir = None
            ema_down = False
            ema_up = False
            atr_ok_t1 = False
            atr_ok_t2 = False
            displaced_down = False
            displaced_up = False
            no_reclaim_down_t1 = False
            no_reclaim_up_t1 = False
            no_reclaim_down_t2 = False
            no_reclaim_up_t2 = False
            reclaim_down = False
            reclaim_up = False
            confirm_down = False
            confirm_up = False

            asia_structure_only = TREND_DAY_ASIA_STRUCTURE_ONLY and current_session == "ASIA"

            if pd.notna(vwap_sigma):
                displaced_down = vwap_sigma <= -VWAP_SIGMA_T1
                displaced_up = vwap_sigma >= VWAP_SIGMA_T1
                no_reclaim_down_t1 = bool(td_no_reclaim_down_t1_arr[i])
                no_reclaim_up_t1 = bool(td_no_reclaim_up_t1_arr[i])
                no_reclaim_down_t2 = bool(td_no_reclaim_down_t2_arr[i])
                no_reclaim_up_t2 = bool(td_no_reclaim_up_t2_arr[i])
                reclaim_down = bool(td_reclaim_down_arr[i])
                reclaim_up = bool(td_reclaim_up_arr[i])

            if pd.notna(atr_expansion):
                atr_ok_t1 = atr_expansion >= ATR_EXP_T1
                atr_ok_t2 = atr_expansion >= ATR_EXP_T2

            if asia_structure_only:
                atr_ok_t1 = True
                atr_ok_t2 = True
            if pd.notna(ema50_val):
                ema_down = bar_close < ema50_val
                ema_up = bar_close > ema50_val
                confirm_down = confirm_down or ema_down
                confirm_up = confirm_up or ema_up
            if pd.notna(session_open):
                confirm_down = confirm_down or (bar_close < session_open)
                confirm_up = confirm_up or (bar_close > session_open)
            if pd.notna(prior_session_low) or pd.notna(prior_session_high):
                prev_close = float(close_arr[i - 1]) if i > 0 else bar_close
                if pd.notna(prior_session_low):
                    confirm_down = confirm_down or (bar_close < prior_session_low and prev_close < prior_session_low)
                if pd.notna(prior_session_high):
                    confirm_up = confirm_up or (bar_close > prior_session_high and prev_close > prior_session_high)
            if impulse_active:
                if impulse_dir == "down" and reclaim_down:
                    impulse_active = False
                elif impulse_dir == "up" and reclaim_up:
                    impulse_active = False
                if not impulse_active:
                    impulse_dir = None
                    impulse_start_price = None
                    impulse_extreme = None
                    pullback_extreme = None
                    max_retracement = 0.0
                    bars_since_impulse = 0

            impulse_started = False
            if not impulse_active and pd.notna(vwap_val):
                start_down = displaced_down and bar_close < vwap_val
                start_up = displaced_up and bar_close > vwap_val
                if start_down and start_up:
                    if abs(vwap_sigma) >= VWAP_SIGMA_T1:
                        start_up = vwap_sigma > 0
                        start_down = not start_up
                if start_down:
                    impulse_active = True
                    impulse_dir = "down"
                    impulse_start_price = bar_close
                    impulse_extreme = bar_low
                    pullback_extreme = bar_high
                    max_retracement = 0.0
                    bars_since_impulse = 1
                    impulse_started = True
                elif start_up:
                    impulse_active = True
                    impulse_dir = "up"
                    impulse_start_price = bar_close
                    impulse_extreme = bar_high
                    pullback_extreme = bar_low
                    max_retracement = 0.0
                    bars_since_impulse = 1
                    impulse_started = True

            if impulse_active:
                if not impulse_started:
                    bars_since_impulse += 1
                if impulse_dir == "down":
                    if bar_low < impulse_extreme:
                        impulse_extreme = bar_low
                        pullback_extreme = bar_high
                    else:
                        pullback_extreme = max(pullback_extreme, bar_high)
                    impulse_range = (impulse_start_price or bar_close) - impulse_extreme
                    if impulse_range >= TICK_SIZE:
                        retracement = (pullback_extreme - impulse_extreme) / impulse_range
                        max_retracement = max(max_retracement, retracement)
                elif impulse_dir == "up":
                    if bar_high > impulse_extreme:
                        impulse_extreme = bar_high
                        pullback_extreme = bar_low
                    else:
                        pullback_extreme = min(pullback_extreme, bar_low)
                    impulse_range = impulse_extreme - (impulse_start_price or bar_close)
                    if impulse_range >= TICK_SIZE:
                        retracement = (impulse_extreme - pullback_extreme) / impulse_range
                        max_retracement = max(max_retracement, retracement)

            confirm_ok_down = (not TREND_DAY_T1_REQUIRE_CONFIRMATION) or confirm_down
            confirm_ok_up = (not TREND_DAY_T1_REQUIRE_CONFIRMATION) or confirm_up
            if TREND_DAY_T1_REQUIRE_STRUCTURAL_BIAS:
                confirm_ok_down = confirm_ok_down and (trend_down_alt or adx_strong_down)
                confirm_ok_up = confirm_ok_up and (trend_up_alt or adx_strong_up)

            tier1_down = atr_ok_t1 and displaced_down and no_reclaim_down_t1 and confirm_ok_down
            tier1_up = atr_ok_t1 and displaced_up and no_reclaim_up_t1 and confirm_ok_up

            if tier1_down:
                tier1_down_until = current_time + dt.timedelta(minutes=TREND_DAY_STICKY_RECLAIM_BARS)
            if tier1_up:
                tier1_up_until = current_time + dt.timedelta(minutes=TREND_DAY_STICKY_RECLAIM_BARS)

            tier1_down_active = bool(tier1_down_until and current_time <= tier1_down_until)
            tier1_up_active = bool(tier1_up_until and current_time <= tier1_up_until)
            if tier1_down_active or tier1_up_active:
                tier1_seen = True
            allow_alt = tier1_seen or (pd.notna(vwap_sigma) and abs(vwap_sigma) >= ALT_PRE_TIER1_VWAP_SIGMA)

            tier2_down = (
                atr_ok_t2
                and displaced_down
                and no_reclaim_down_t2
                and impulse_active
                and impulse_dir == "down"
                and bars_since_impulse >= IMPULSE_MIN_BARS
                and max_retracement <= IMPULSE_MAX_RETRACE
            )
            tier2_up = (
                atr_ok_t2
                and displaced_up
                and no_reclaim_up_t2
                and impulse_active
                and impulse_dir == "up"
                and bars_since_impulse >= IMPULSE_MIN_BARS
                and max_retracement <= IMPULSE_MAX_RETRACE
            )
            if TREND_DAY_ASIA_DISABLE_TIER2 and current_session == "ASIA":
                tier2_down = False
                tier2_up = False

            computed_tier = 0
            computed_dir = None
            if tier2_down and tier2_up:
                computed_dir = "down" if vwap_sigma < 0 else "up"
                computed_tier = 2
            elif tier2_down:
                computed_dir = "down"
                computed_tier = 2
            elif tier2_up:
                computed_dir = "up"
                computed_tier = 2
            elif tier1_down_active and tier1_up_active:
                computed_dir = "down" if vwap_sigma < 0 else "up"
                computed_tier = 1
            elif tier1_down_active:
                computed_dir = "down"
                computed_tier = 1
            elif tier1_up_active:
                computed_dir = "up"
                computed_tier = 1
            elif trend_up_alt and allow_alt:
                computed_dir = "up"
                computed_tier = 1
            elif trend_down_alt and allow_alt:
                computed_dir = "down"
                computed_tier = 1

            if sticky_trend_dir is None:
                if computed_dir:
                    sticky_trend_dir = computed_dir
                    sticky_opposite_count = 0
            else:
                if computed_dir and computed_dir != sticky_trend_dir:
                    adx_ok = (
                        (computed_dir == "up" and adx_strong_up)
                        or (computed_dir == "down" and adx_strong_down)
                    )
                    if adx_ok:
                        sticky_opposite_count += 1
                    else:
                        sticky_opposite_count = 0
                else:
                    sticky_opposite_count = 0
                if sticky_opposite_count >= ADX_FLIP_BARS:
                    sticky_trend_dir = computed_dir
                    sticky_opposite_count = 0

            if sticky_trend_dir:
                trend_day_dir = sticky_trend_dir
                if computed_dir == sticky_trend_dir and computed_tier == 2:
                    trend_day_tier = 2
                else:
                    trend_day_tier = 1
            else:
                trend_day_dir = computed_dir
                trend_day_tier = computed_tier

            if trend_day_tier > 0 and trend_day_dir and pd.notna(vwap_sigma):
                trend_day_max_sigma = max(trend_day_max_sigma, abs(float(vwap_sigma)))

            if trend_day_tier > 0 and trend_day_dir:
                loss_limit = directional_loss_blocker.consecutive_loss_limit
                if trend_day_dir == "up":
                    loss_count = directional_loss_blocker.long_consecutive_losses
                else:
                    loss_count = directional_loss_blocker.short_consecutive_losses
                if loss_count >= loss_limit:
                    msg = (
                        "[TrendDay] Deactivating tier/alt after "
                        f"{loss_count} consecutive {trend_day_dir.upper()} losses"
                    )
                    if BACKTEST_TRENDDAY_VERBOSE:
                        logging.warning(msg)
                    else:
                        logging.debug(msg)
                    _deactivate_trend_day(
                        f"DLB {loss_count} {trend_day_dir.upper()} losses",
                        current_time,
                    )

            # === TrendDay "rotation/mean-reversion" deactivation ===
            if trend_day_tier > 0 and trend_day_dir:
                # Gate A: VWAP reclaim + sigma decay hold
                if (
                    (i + 1) >= TREND_DAY_DEACTIVATE_RECLAIM_WINDOW
                    and (i + 1) >= TREND_DAY_DEACTIVATE_SIGMA_BARS
                ):
                    recent_close = close_arr[
                        i + 1 - TREND_DAY_DEACTIVATE_RECLAIM_WINDOW : i + 1
                    ].astype(float, copy=False)
                    recent_vwap = td_vwap_arr[
                        i + 1 - TREND_DAY_DEACTIVATE_RECLAIM_WINDOW : i + 1
                    ].astype(float, copy=False)
                    if trend_day_dir == "up":
                        reclaim_count = int((recent_close < recent_vwap).sum())
                    else:
                        reclaim_count = int((recent_close > recent_vwap).sum())

                    sigma_recent = td_vwap_sigma_arr[
                        i + 1 - TREND_DAY_DEACTIVATE_SIGMA_BARS : i + 1
                    ].astype(float, copy=False)
                    sigma_ok = bool(
                        (np.abs(sigma_recent) < TREND_DAY_DEACTIVATE_SIGMA_THRESHOLD).all()
                    )

                    if reclaim_count >= TREND_DAY_DEACTIVATE_RECLAIM_COUNT and sigma_ok:
                        _deactivate_trend_day(
                            "VWAP reclaim + sigma decay "
                            f"(reclaim={reclaim_count}/{TREND_DAY_DEACTIVATE_RECLAIM_WINDOW}, "
                            f"sigma<{TREND_DAY_DEACTIVATE_SIGMA_THRESHOLD})",
                            current_time,
                        )

                # Gate B: sigma decay kill switch
                if trend_day_tier > 0 and trend_day_dir:
                    if (i + 1) >= TREND_DAY_DEACTIVATE_SIGMA_DECAY_BARS:
                        sigma_decay_window = td_vwap_sigma_arr[
                            i + 1 - TREND_DAY_DEACTIVATE_SIGMA_DECAY_BARS : i + 1
                        ].astype(float, copy=False)
                        sigma_decay_ok = bool(
                            (np.abs(sigma_decay_window) < TREND_DAY_DEACTIVATE_SIGMA_DECAY_THRESHOLD).all()
                        )
                        current_sigma = td_vwap_sigma_arr[i]
                        guard_active = False
                        if pd.notna(current_sigma):
                            guard_active = (
                                trend_day_max_sigma >= TREND_DAY_DEACTIVATE_SIGMA_GUARD_MAX
                                and abs(float(current_sigma)) >= TREND_DAY_DEACTIVATE_SIGMA_GUARD_CURRENT
                            )
                        if sigma_decay_ok and not guard_active:
                            _deactivate_trend_day(
                                "VWAP sigma decay "
                                f"(<{TREND_DAY_DEACTIVATE_SIGMA_DECAY_THRESHOLD} for "
                                f"{TREND_DAY_DEACTIVATE_SIGMA_DECAY_BARS} bars)",
                                current_time,
                            )
            if trend_day_tier > 0 and (
                trend_day_tier != last_trend_day_tier or trend_day_dir != last_trend_day_dir
            ):
                if BACKTEST_TRENDDAY_VERBOSE:
                    color = "\033[92m" if trend_day_dir == "up" else "\033[91m"
                    print(
                        f"{color}[TrendDay] Tier {trend_day_tier} {trend_day_dir} activated @ "
                        f"{current_time.strftime('%Y-%m-%d %H:%M')}\033[0m"
                    )
                    atr_dbg = f"{atr_expansion:.3f}" if pd.notna(atr_expansion) else "na"
                    vwap_dbg = f"{vwap_sigma:.3f}" if pd.notna(vwap_sigma) else "na"
                    ema_dbg = f"{ema50_val:.2f}" if pd.notna(ema50_val) else "na"
                    vwap_val_dbg = f"{vwap_val:.2f}" if pd.notna(vwap_val) else "na"
                    print(
                        "[TrendDayDebug] "
                        f"atr_exp={atr_dbg} vwap_sigma={vwap_dbg} ema50={ema_dbg} vwap={vwap_val_dbg} "
                        f"disp_down={displaced_down} disp_up={displaced_up} "
                        f"nr_dn_t1={no_reclaim_down_t1} nr_up_t1={no_reclaim_up_t1} "
                        f"nr_dn_t2={no_reclaim_down_t2} nr_up_t2={no_reclaim_up_t2} "
                        f"tier1_dn_active={tier1_down_active} tier1_up_active={tier1_up_active} "
                        f"tier2_dn={tier2_down} tier2_up={tier2_up} "
                        f"trend_dn_alt={trend_down_alt} trend_up_alt={trend_up_alt} "
                        f"confirm_dn={confirm_down} confirm_up={confirm_up} "
                        f"adx_dn={adx_strong_down} adx_up={adx_strong_up} "
                        f"sticky_dir={sticky_trend_dir} computed_dir={computed_dir} computed_tier={computed_tier}"
                    )
            last_trend_day_tier = trend_day_tier
            last_trend_day_dir = trend_day_dir
        else:
            trend_day_tier = 0
            trend_day_dir = None
            last_trend_day_tier = 0
            last_trend_day_dir = None
            sticky_trend_dir = None
            sticky_reclaim_count = 0
            sticky_opposite_count = 0
            tier1_seen = False
            trend_day_max_sigma = 0.0

        in_test_range = current_time >= start_time

        if in_test_range:
            if holiday_closed_now and active_trade is not None:
                close_trade(
                    bar_open,
                    current_time,
                    "holiday_flat",
                    bar_index=i,
                )
                holiday_flat_closes += 1
                pending_exit = False
                pending_exit_reason = None
                opposite_signal_count = 0
            if holiday_closed_now and pending_loose_signals:
                pending_loose_signals.clear()
            if force_flat_now and active_trade is not None:
                close_trade(
                    bar_open,
                    current_time,
                    "session_flat",
                    bar_index=i,
                )
                session_flat_closes += 1
                pending_exit = False
                pending_exit_reason = None
                opposite_signal_count = 0
            if pending_exit and active_trade is not None:
                close_trade(
                    bar_open,
                    current_time,
                    pending_exit_reason or "reverse",
                    bar_index=i,
                )
                pending_exit = False
                pending_exit_reason = None
            if pending_entry is not None:
                if holiday_closed_now:
                    record_filter("HolidayClosed")
                    holiday_entry_blocks += 1
                    pending_entry = None
                if pending_entry is not None and entry_window_blocked:
                    record_filter("SessionTimeWindow")
                    session_entry_blocks += 1
                    pending_entry = None
                if pending_entry is not None and sl_cap_shadow_lock_until_index >= i:
                    record_filter("SLCapShadowLock")
                    sl_cap_shadow_lock_entry_blocks += 1
                    pending_entry = None
                if pending_entry is not None and trend_day_tier > 0 and trend_day_dir:
                    if (trend_day_dir == "down" and pending_entry["side"] == "LONG") or (
                        trend_day_dir == "up" and pending_entry["side"] == "SHORT"
                    ):
                        record_filter(f"TrendDayTier{trend_day_tier}")
                        pending_entry = None
                    else:
                        pending_entry["trend_day_tier"] = trend_day_tier
                        pending_entry["trend_day_dir"] = trend_day_dir
                if pending_entry is not None:
                    open_trade(pending_entry, bar_open, current_time, bar_index=i)
                    pending_entry = None

        if active_trade is not None:
            entry_price = active_trade["entry_price"]
            if active_trade["side"] == "LONG":
                mfe_points = bar_high - entry_price
                mae_points = entry_price - bar_low
            else:
                mfe_points = entry_price - bar_low
                mae_points = bar_high - entry_price
            if (not skip_mfe_mae) or bool(active_trade.get("de3_break_even_enabled", False)):
                active_trade["mfe_points"] = max(active_trade.get("mfe_points", 0.0), mfe_points)
            if not skip_mfe_mae:
                active_trade["mae_points"] = max(active_trade.get("mae_points", 0.0), mae_points)

            _update_de3_profit_milestone_state(
                active_trade,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_index=i,
            )
            _update_de3_entry_trade_day_extreme_state(
                active_trade,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_index=i,
            )
            _apply_pending_de3_break_even_stop_update(active_trade, bar_index=i)
            _stage_de3_break_even_stop_update(active_trade, bar_index=i)

            exit_hit = check_stop_take(bar_open, bar_high, bar_low, bar_close)
            if exit_hit is not None:
                exit_price, exit_reason = exit_hit
                if str(exit_reason) in {"tiered_take", "tiered_take_gap"}:
                    _apply_de3_tiered_take_fill(
                        active_trade,
                        exit_price,
                        current_time,
                        fill_reason=str(exit_reason),
                        bar_index=i,
                    )
                else:
                    close_trade(exit_price, current_time, exit_reason, bar_index=i)
                    pending_exit = False
                    pending_exit_reason = None
            elif active_trade is not None:
                advance_trade_management_state(bar_close)
                if check_signal_horizon_exit():
                    close_trade(bar_close, current_time, "horizon", bar_index=i)
                    pending_exit = False
                    pending_exit_reason = None
                elif (early_exit_reason := check_early_exit(bar_close)) is not None:
                    active_trade["de3_early_exit_trigger_reason"] = str(early_exit_reason)
                    close_trade(bar_close, current_time, "early_exit", bar_index=i)
                    pending_exit = False
                    pending_exit_reason = None

        if math.isfinite(bar_high):
            if math.isfinite(de3_trade_day_high_known):
                de3_trade_day_high_known = max(float(de3_trade_day_high_known), float(bar_high))
            else:
                de3_trade_day_high_known = float(bar_high)
        if math.isfinite(bar_low):
            if math.isfinite(de3_trade_day_low_known):
                de3_trade_day_low_known = min(float(de3_trade_day_low_known), float(bar_low))
            else:
                de3_trade_day_low_known = float(bar_low)

        if manifold_only_fast_mode and manifold_fast_strategy is not None:
            if not in_test_range:
                continue

            bar_count += 1
            if console_progress_enabled:
                now_ts = time.perf_counter()
                if (
                    (bar_count == 1)
                    or (bar_count == total_bars)
                    or ((now_ts - console_progress_last_ts) >= console_progress_every_sec)
                ):
                    elapsed_sim = max(0.0, now_ts - sim_started_at)
                    if bar_count > 0 and total_bars > 0:
                        eta_sec = (elapsed_sim / float(bar_count)) * float(max(0, total_bars - bar_count))
                        pct = (float(bar_count) / float(total_bars)) * 100.0
                    else:
                        eta_sec = float("nan")
                        pct = 0.0
                    print(
                        (
                            f"[backtest] {bar_count}/{total_bars} ({pct:.1f}%) "
                            f"elapsed={_format_hms(elapsed_sim)} eta={_format_hms(eta_sec)} "
                            f"trades={trades} equity=${equity:.2f} "
                            f"time={current_time.strftime('%Y-%m-%d %H:%M')}"
                        ),
                        flush=True,
                    )
                    console_progress_last_ts = now_ts

            if holiday_closed_now:
                record_filter("HolidayClosed")
                holiday_signal_blocks += 1
                emit_progress(current_time, bar_close)
                continue

            disabled_filter = execution_disabled_filter("ManifoldStrategy", current_session)
            if disabled_filter:
                record_filter(disabled_filter, kind="disabled")
                emit_progress(current_time, bar_close)
                continue

            news_globally_blocked = False
            if enabled_filter_news:
                news_blocked, _ = news_filter.should_block_trade(current_time)
                news_globally_blocked = bool(news_blocked)
            if news_globally_blocked:
                record_filter("NewsFilter")
                emit_progress(current_time, bar_close)
                continue

            if history_df is None:
                history_df = full_df.iloc[history_start:history_end]

            try:
                signal = manifold_fast_strategy.on_bar(history_df, current_time)
            except Exception:
                signal = None

            if signal:
                apply_multipliers(signal, strategy_hint="ManifoldStrategy")
                signal.setdefault("strategy", "ManifoldStrategy")

                if enabled_filter_fixed_sltp:
                    fixed_ok, fixed_details = apply_fixed_sltp(
                        signal,
                        history_df,
                        bar_close,
                        ts=current_time,
                        session=current_session,
                        sl_dist_override=signal.get("sl_dist"),
                    )
                else:
                    fixed_ok, fixed_details = True, None
                if fixed_ok:
                    if fixed_details:
                        signal["sl_dist"] = fixed_details.get(
                            "sl_dist",
                            signal.get("sl_dist", MIN_SL),
                        )
                        signal["tp_dist"] = fixed_details.get(
                            "tp_dist",
                            signal.get("tp_dist", MIN_TP),
                        )
                        signal["sltp_bracket"] = fixed_details.get("bracket")
                        signal["vol_regime"] = fixed_details.get(
                            "vol_regime",
                            signal.get("vol_regime"),
                        )
                        log_fixed_sltp(fixed_details, signal.get("strategy"))

                    signal_tp = signal.get("tp_dist", MIN_TP)
                    if not enabled_filter_target_feasibility or chop_analyzer is None:
                        is_feasible = True
                    else:
                        is_feasible, _ = chop_analyzer.check_target_feasibility(
                            entry_price=bar_close,
                            side=signal["side"],
                            tp_distance=signal_tp,
                            df_1m=history_df,
                        )

                    if is_feasible:
                        if not enabled_filter_vol_guard:
                            should_trade = True
                            vol_adj = {
                                "sl_dist": float(signal.get("sl_dist", MIN_SL)),
                                "tp_dist": float(signal.get("tp_dist", MIN_TP)),
                                "regime": "BYPASS",
                                "adjustment_applied": False,
                                "size": int(_signal_base_size(signal, CONTRACTS)),
                            }
                        else:
                            should_trade, vol_adj = check_volatility(
                                history_df,
                                signal.get("sl_dist", MIN_SL),
                                signal.get("tp_dist", MIN_TP),
                                base_size=_signal_base_size(signal, CONTRACTS),
                                ts=current_time,
                            )
                        if should_trade:
                            if isinstance(vol_adj, dict):
                                signal["sl_dist"] = vol_adj.get(
                                    "sl_dist",
                                    signal.get("sl_dist", MIN_SL),
                                )
                                signal["tp_dist"] = vol_adj.get(
                                    "tp_dist",
                                    signal.get("tp_dist", MIN_TP),
                                )
                                signal["vol_regime"] = vol_adj.get(
                                    "regime",
                                    signal.get("vol_regime", "UNKNOWN"),
                                )
                                if vol_adj.get("adjustment_applied", False):
                                    signal["size"] = vol_adj["size"]
                            signal["entry_mode"] = "manifold_hard_only_fast"
                            handle_signal(signal, bar_index=i)
                        else:
                            record_filter("VolatilityGuardrail")
                    else:
                        record_filter("TargetFeasibility")
                else:
                    record_filter("FixedSLTP")

            emit_progress(current_time, bar_close)
            continue

        if aetherflow_only_fast_mode and aetherflow_strategy is not None:
            if not in_test_range:
                continue

            bar_count += 1
            if console_progress_enabled:
                now_ts = time.perf_counter()
                if (
                    (bar_count == 1)
                    or (bar_count == total_bars)
                    or ((now_ts - console_progress_last_ts) >= console_progress_every_sec)
                ):
                    elapsed_sim = max(0.0, now_ts - sim_started_at)
                    if bar_count > 0 and total_bars > 0:
                        eta_sec = (elapsed_sim / float(bar_count)) * float(max(0, total_bars - bar_count))
                        pct = (float(bar_count) / float(total_bars)) * 100.0
                    else:
                        eta_sec = float("nan")
                        pct = 0.0
                    print(
                        (
                            f"[backtest] {bar_count}/{total_bars} ({pct:.1f}%) "
                            f"elapsed={_format_hms(elapsed_sim)} eta={_format_hms(eta_sec)} "
                            f"trades={trades} equity=${equity:.2f} "
                            f"time={current_time.strftime('%Y-%m-%d %H:%M')}"
                        ),
                        flush=True,
                    )
                    console_progress_last_ts = now_ts

            if holiday_closed_now:
                record_filter("HolidayClosed")
                holiday_signal_blocks += 1
                emit_progress(current_time, bar_close)
                continue

            disabled_filter = execution_disabled_filter("AetherFlowStrategy", current_session)
            if disabled_filter:
                record_filter(disabled_filter, kind="disabled")
                emit_progress(current_time, bar_close)
                continue

            news_globally_blocked = False
            if enabled_filter_news:
                news_blocked, _ = news_filter.should_block_trade(current_time)
                news_globally_blocked = bool(news_blocked)
            if news_globally_blocked:
                record_filter("NewsFilter")
                emit_progress(current_time, bar_close)
                continue

            if history_df is None:
                history_df = full_df.iloc[history_start:history_end]

            try:
                signal = aetherflow_strategy.on_bar(history_df, current_time)
            except Exception:
                signal = None

            if signal:
                apply_multipliers(signal, strategy_hint="AetherFlowStrategy")
                signal.setdefault("strategy", "AetherFlowStrategy")

                if enabled_filter_fixed_sltp:
                    fixed_ok, fixed_details = apply_fixed_sltp(
                        signal,
                        history_df,
                        bar_close,
                        ts=current_time,
                        session=current_session,
                        sl_dist_override=signal.get("sl_dist"),
                    )
                else:
                    fixed_ok, fixed_details = True, None
                if fixed_ok:
                    if fixed_details:
                        signal["sl_dist"] = fixed_details.get(
                            "sl_dist",
                            signal.get("sl_dist", MIN_SL),
                        )
                        signal["tp_dist"] = fixed_details.get(
                            "tp_dist",
                            signal.get("tp_dist", MIN_TP),
                        )
                        signal["sltp_bracket"] = fixed_details.get("bracket")
                        signal["vol_regime"] = fixed_details.get(
                            "vol_regime",
                            signal.get("vol_regime"),
                        )
                        log_fixed_sltp(fixed_details, signal.get("strategy"))

                    signal_tp = signal.get("tp_dist", MIN_TP)
                    if not enabled_filter_target_feasibility or chop_analyzer is None:
                        is_feasible = True
                    else:
                        is_feasible, _ = chop_analyzer.check_target_feasibility(
                            entry_price=bar_close,
                            side=signal["side"],
                            tp_distance=signal_tp,
                            df_1m=history_df,
                        )

                    if is_feasible:
                        if not enabled_filter_vol_guard:
                            should_trade = True
                            vol_adj = {
                                "sl_dist": float(signal.get("sl_dist", MIN_SL)),
                                "tp_dist": float(signal.get("tp_dist", MIN_TP)),
                                "regime": "BYPASS",
                                "adjustment_applied": False,
                                "size": int(_signal_base_size(signal, CONTRACTS)),
                            }
                        else:
                            should_trade, vol_adj = check_volatility(
                                history_df,
                                signal.get("sl_dist", MIN_SL),
                                signal.get("tp_dist", MIN_TP),
                                base_size=_signal_base_size(signal, CONTRACTS),
                                ts=current_time,
                            )
                        if should_trade:
                            if isinstance(vol_adj, dict):
                                signal["sl_dist"] = vol_adj.get(
                                    "sl_dist",
                                    signal.get("sl_dist", MIN_SL),
                                )
                                signal["tp_dist"] = vol_adj.get(
                                    "tp_dist",
                                    signal.get("tp_dist", MIN_TP),
                                )
                                signal["vol_regime"] = vol_adj.get(
                                    "regime",
                                    signal.get("vol_regime", "UNKNOWN"),
                                )
                                if vol_adj.get("adjustment_applied", False):
                                    signal["size"] = vol_adj["size"]
                            signal["entry_mode"] = "aetherflow_hard_only_fast"
                            handle_signal(signal, bar_index=i)
                        else:
                            record_filter("VolatilityGuardrail")
                    else:
                        record_filter("TargetFeasibility")
                else:
                    record_filter("FixedSLTP")

            emit_progress(current_time, bar_close)
            continue

        if enabled_filter_rejection:
            rejection_filter.update(current_time, bar_high, bar_low, bar_close)
        if enabled_filter_bank:
            bank_filter.update(current_time, bar_high, bar_low, bar_close)
        if enabled_filter_chop:
            chop_filter.update(bar_high, bar_low, bar_close, current_time)
        if enabled_filter_extension:
            extension_filter.update(bar_high, bar_low, bar_close, current_time)
        if enabled_filter_structure:
            structure_blocker.update(
                _tail_df_for_bar(
                    structure_lookback_const,
                    history_df,
                    history_start,
                    history_end,
                    history_tail_cache,
                )
            )
        if enabled_filter_regime_blocker:
            regime_blocker.update(
                _tail_df_for_bar(
                    regime_update_lookback_const,
                    history_df,
                    history_start,
                    history_end,
                    history_tail_cache,
                )
            )
        if penalty_blocker is not None and enabled_filter_penalty:
            penalty_blocker.update(
                _tail_df_for_bar(
                    penalty_lookback_const,
                    history_df,
                    history_start,
                    history_end,
                    history_tail_cache,
                )
            )
        if penalty_blocker_asia is not None and enabled_filter_penalty:
            penalty_blocker_asia.update(
                _tail_df_for_bar(
                    penalty_asia_lookback_const,
                    history_df,
                    history_start,
                    history_end,
                    history_tail_cache,
                )
            )
        if enabled_filter_memory_sr:
            memory_sr.update(
                _tail_df_for_bar(
                    memory_lookback_const,
                    history_df,
                    history_start,
                    history_end,
                    history_tail_cache,
                )
            )
        if enabled_filter_directional_loss:
            directional_loss_blocker.update_quarter(current_time)
        if enabled_filter_impulse:
            impulse_filter.update(
                _tail_df_for_bar(
                    impulse_update_lookback_const,
                    history_df,
                    history_start,
                    history_end,
                    history_tail_cache,
                )
            )

        if (
            enabled_filter_htf_fvg
            and htf_fvg_enabled_backtest
            and htf_fvg_filter is not None
            and resample_cache_60 is not None
            and resample_cache_240 is not None
            and processed_bars % 60 == 0
        ):
            df_60m = resample_cache_60.get_full(current_time)
            df_240m = resample_cache_240.get_full(current_time)
            htf_fvg_filter.update_structure_data(df_60m, df_240m)

        if not in_test_range:
            continue

        bar_count += 1
        if console_progress_enabled:
            now_ts = time.perf_counter()
            if (
                (bar_count == 1)
                or (bar_count == total_bars)
                or ((now_ts - console_progress_last_ts) >= console_progress_every_sec)
            ):
                elapsed_sim = max(0.0, now_ts - sim_started_at)
                if bar_count > 0 and total_bars > 0:
                    eta_sec = (elapsed_sim / float(bar_count)) * float(max(0, total_bars - bar_count))
                    pct = (float(bar_count) / float(total_bars)) * 100.0
                else:
                    eta_sec = float("nan")
                    pct = 0.0
                print(
                    (
                        f"[backtest] {bar_count}/{total_bars} ({pct:.1f}%) "
                        f"elapsed={_format_hms(elapsed_sim)} eta={_format_hms(eta_sec)} "
                        f"trades={trades} equity=${equity:.2f} "
                        f"time={current_time.strftime('%Y-%m-%d %H:%M')}"
                    ),
                    flush=True,
                )
                console_progress_last_ts = now_ts

        if holiday_closed_now:
            record_filter("HolidayClosed")
            holiday_signal_blocks += 1
            continue

        if de3_only_fast_mode and dynamic_engine3_strat is not None:
            # Fast path: DE3-only runs bypass the generic multi-strategy filter stack.
            # Core trade lifecycle controls (session window, force-flat, holiday, stop/take) remain active above.
            if history_df is None:
                history_df = full_df.iloc[history_start:history_end]
            is_choppy = False
            chop_reason = ""
            allowed_chop_side = None
            vol_regime_current = None
            asia_viable = True
            asia_trend_bias_side = None
            de3_signal = None
            try:
                de3_signal = dynamic_engine3_strat.on_bar(history_df)
            except Exception:
                if de3_debug_exceptions:
                    logging.exception(
                        "DynamicEngine3Strategy exception in de3_only_fast_mode at %s",
                        current_time,
                    )
                de3_signal = None
            if de3_signal:
                apply_multipliers(de3_signal, strategy_hint="DynamicEngine3Strategy")
                de3_signal.setdefault("strategy", "DynamicEngine3Strategy")
                if _de3_manifold_adaptation_allows_signal(
                    de3_signal,
                    "DynamicEngine3Strategy",
                ):
                    handle_signal(de3_signal, bar_index=i)
                else:
                    record_filter("DE3ManifoldAdapt")
            emit_progress(current_time, bar_close)
            continue


        news_globally_blocked = False
        if enabled_filter_news:
            news_blocked, _ = news_filter.should_block_trade(current_time)
            news_globally_blocked = bool(news_blocked)
        if news_globally_blocked and not ml_filter_bypass_bar_active:
            record_filter("NewsFilter")
            continue

        bar_filter_cache: dict[tuple, tuple] = {}
        cached_vol_regime: Optional[str] = None
        is_asia_session = False
        asia_trend_bias_side: Optional[str] = None

        def _cache_filter_result(cache_key: tuple, evaluator: Callable[[], object]):
            cached = bar_filter_cache.get(cache_key)
            if cached is None:
                cached = evaluator()
                bar_filter_cache[cache_key] = cached
            return cached

        def _get_vol_regime() -> str:
            nonlocal cached_vol_regime
            if cached_vol_regime is None:
                cached_vol_regime, _, _ = volatility_filter.get_regime(history_df)
            return cached_vol_regime

        def _trend_check(side: str, is_range_fade: bool = False):
            if not enabled_filter_trend:
                return False, "disabled"
            return _cache_filter_result(
                ("trend_filter", side, bool(is_range_fade)),
                lambda: trend_filter.should_block_trade(
                    _tail_df_for_bar(
                        trend_filter_lookback_const,
                        history_df,
                        history_start,
                        history_end,
                        history_tail_cache,
                    ),
                    side,
                    is_range_fade=is_range_fade,
                ),
            )

        def _regime_block_check(side: str):
            if not enabled_filter_regime_blocker:
                return False, "disabled"
            return _cache_filter_result(
                ("regime_blocker", side),
                lambda: regime_blocker.should_block_trade(side, bar_close),
            )

        def _directional_loss_check(side: str):
            if not enabled_filter_directional_loss:
                return False, "disabled"
            return _cache_filter_result(
                ("directional_loss_blocker", side),
                lambda: directional_loss_blocker.should_block_trade(side, current_time),
            )

        def _rejection_check(side: str):
            if not enabled_filter_rejection:
                return False, "disabled"
            return _cache_filter_result(
                ("rejection_filter", side),
                lambda: rejection_filter.should_block_trade(side),
            )

        def _impulse_check(side: str):
            if not enabled_filter_impulse:
                return False, "disabled"
            return _cache_filter_result(
                ("impulse_filter", side),
                lambda: impulse_filter.should_block_trade(side),
            )

        def _extension_check(side: str):
            if not enabled_filter_extension:
                return False, "disabled"
            return _cache_filter_result(
                ("extension_filter", side),
                lambda: extension_filter.should_block_trade(side),
            )

        def _structure_check(side: str):
            if not enabled_filter_structure:
                return False, "disabled"
            return _cache_filter_result(
                ("structure_blocker", side),
                lambda: structure_blocker.should_block_trade(side, bar_close),
            )

        def _bank_check(side: str):
            if not enabled_filter_bank:
                return False, "disabled"
            return _cache_filter_result(
                ("bank_filter", side),
                lambda: bank_filter.should_block_trade(side),
            )

        def _legacy_trend_check(side: str):
            if not enabled_filter_legacy_trend:
                return False, "disabled"
            return _cache_filter_result(
                ("legacy_trend", side),
                lambda: legacy_filters.check_trend(
                    _tail_df_for_bar(
                        legacy_trend_lookback_const,
                        history_df,
                        history_start,
                        history_end,
                        history_tail_cache,
                    ),
                    side,
                ),
            )

        def _penalty_check(source, side: str):
            if not enabled_filter_penalty:
                return False, "disabled"
            return _cache_filter_result(
                ("penalty_blocker", id(source), side),
                lambda: source.should_block_trade(side, bar_close),
            )

        def _memory_sr_check(side: str):
            if not enabled_filter_memory_sr:
                return False, "disabled"
            return _cache_filter_result(
                ("memory_sr", side),
                lambda: memory_sr.should_block_trade(side, bar_close),
            )

        def _chop_check(side: str, trend_state: str = "NEUTRAL"):
            if not enabled_filter_chop:
                return False, "disabled"
            vol_regime = _get_vol_regime()
            return _cache_filter_result(
                (
                    "chop_filter",
                    side,
                    trend_state,
                    vol_regime,
                    rejection_filter.prev_day_pm_bias,
                ),
                lambda: chop_filter.should_block_trade(
                    side,
                    rejection_filter.prev_day_pm_bias,
                    bar_close,
                    trend_state,
                    vol_regime,
                ),
            )

        def _target_feasibility_check(side: str, tp_distance: float):
            if not enabled_filter_target_feasibility:
                return True, "disabled"
            if chop_analyzer is None:
                return True, "disabled"
            try:
                tp_key = round(float(tp_distance), 6)
            except Exception:
                tp_key = round(float(MIN_TP), 6)
            return _cache_filter_result(
                ("target_feasibility", side, tp_key),
                lambda: chop_analyzer.check_target_feasibility(
                    entry_price=bar_close,
                    side=side,
                    tp_distance=tp_distance,
                    df_1m=history_df,
                ),
            )

        def _asia_target_feasibility_override_eval(
            side_key: str,
            tp_distance: float,
            trend_bias_side: Optional[str],
        ) -> bool:
            if not asia_tf_enabled:
                return False
            if not trend_bias_side or side_key != str(trend_bias_side).upper():
                return False
            tp_val = _coerce_float(tp_distance, 0.0)
            if tp_val <= 0:
                return False
            if (i + 1) < asia_tf_lookback:
                return False
            if not math.isfinite(asia_tf_box_range_current):
                return False
            box_range = float(asia_tf_box_range_current)
            if box_range <= 0 or box_range < asia_tf_min_box:
                return False
            if tp_val <= box_range * asia_tf_max_mult:
                return True
            return asia_tf_allow_trend_override

        def _asia_target_feasibility_override_check(side: str, tp_distance: float):
            if not (asia_calib_enabled and is_asia_session):
                return False
            side_key = str(side).upper()
            try:
                tp_key = round(float(tp_distance), 6)
            except Exception:
                tp_key = round(float(MIN_TP), 6)
            return bool(
                _cache_filter_result(
                    ("asia_target_feasibility", side_key, tp_key, asia_trend_bias_side),
                    lambda: _asia_target_feasibility_override_eval(
                        side_key,
                        tp_distance,
                        asia_trend_bias_side,
                    ),
                )
            )

        def _volatility_guard_check(sl_dist: float, tp_dist: float, base_size: int):
            if not enabled_filter_vol_guard:
                return (
                    True,
                    {
                        "sl_dist": _coerce_float(sl_dist, MIN_SL),
                        "tp_dist": _coerce_float(tp_dist, MIN_TP),
                        "regime": "BYPASS",
                        "adjustment_applied": False,
                        "size": _coerce_int(base_size, CONTRACTS),
                    },
                )
            sl_key = round(_coerce_float(sl_dist, MIN_SL), 6)
            tp_key = round(_coerce_float(tp_dist, MIN_TP), 6)
            size_key = _coerce_int(base_size, CONTRACTS)
            cached = _cache_filter_result(
                ("volatility_guard", sl_key, tp_key, size_key),
                lambda: check_volatility(
                    history_df,
                    sl_key,
                    tp_key,
                    base_size=size_key,
                    ts=current_time,
                ),
            )
            should_trade = bool(cached[0]) if isinstance(cached, tuple) and len(cached) >= 1 else False
            vol_adj = {}
            if isinstance(cached, tuple) and len(cached) >= 2 and isinstance(cached[1], dict):
                vol_adj = dict(cached[1])
            return should_trade, vol_adj

        def _apply_fixed_sltp_check(signal_payload: dict):
            if not enabled_filter_fixed_sltp:
                return True, None
            return apply_fixed_sltp(
                signal_payload,
                history_df,
                bar_close,
                ts=current_time,
                session=current_session,
                sl_dist_override=signal_payload.get("sl_dist"),
            )

        de3_v4_runtime_cfg = (
            (CONFIG.get("DE3_V4", {}) or {}).get("runtime", {})
            if isinstance((CONFIG.get("DE3_V4", {}) or {}).get("runtime", {}), dict)
            else {}
        )
        de3_v4_bracket_runtime_cfg = (
            de3_v4_runtime_cfg.get("bracket_module", {})
            if isinstance(de3_v4_runtime_cfg.get("bracket_module", {}), dict)
            else {}
        )
        de3_v4_preserve_bracket_fixed_sltp = bool(
            de3_v4_bracket_runtime_cfg.get("preserve_selected_bracket_through_fixed_sltp", False)
        )
        de3_v4_preserve_bracket_vol_guard = bool(
            de3_v4_bracket_runtime_cfg.get("preserve_selected_bracket_through_vol_guard", False)
        )

        def _is_de3_v4_signal_payload(signal_payload: dict) -> bool:
            if not isinstance(signal_payload, dict):
                return False
            if str(signal_payload.get("de3_version", "") or "").strip().lower() == "v4":
                return True
            if signal_payload.get("de3_v4_selected_variant_id"):
                return True
            if signal_payload.get("de3_v4_selected_lane"):
                return True
            return False

        def _apply_bracket_dist_update(
            signal_payload: dict,
            *,
            sl_value,
            tp_value,
            stage: str,
            preserve_selected: bool,
        ) -> None:
            base_sl = _coerce_float(signal_payload.get("sl_dist", MIN_SL), MIN_SL)
            base_tp = _coerce_float(signal_payload.get("tp_dist", MIN_TP), MIN_TP)
            requested_sl = _coerce_float(sl_value, base_sl)
            requested_tp = _coerce_float(tp_value, base_tp)
            applied_sl = requested_sl
            applied_tp = requested_tp
            preserved = False
            if preserve_selected and _is_de3_v4_signal_payload(signal_payload):
                selected_sl = _coerce_float(signal_payload.get("de3_v4_selected_sl", 0.0), 0.0)
                selected_tp = _coerce_float(signal_payload.get("de3_v4_selected_tp", 0.0), 0.0)
                if selected_sl > 0.0 and selected_tp > 0.0:
                    signal_payload[f"de3_v4_{stage}_suggested_sl"] = float(requested_sl)
                    signal_payload[f"de3_v4_{stage}_suggested_tp"] = float(requested_tp)
                    applied_sl = selected_sl
                    applied_tp = selected_tp
                    preserved = True
            signal_payload["sl_dist"] = applied_sl
            signal_payload["tp_dist"] = applied_tp
            if preserved:
                signal_payload["de3_v4_bracket_preserved"] = True
                preserved_stages = signal_payload.get("de3_v4_bracket_preserved_stages")
                if not isinstance(preserved_stages, list):
                    preserved_stages = []
                if stage not in preserved_stages:
                    preserved_stages.append(stage)
                signal_payload["de3_v4_bracket_preserved_stages"] = preserved_stages

        def _apply_fixed_details_to_signal(signal_payload: dict, fixed_details: dict) -> None:
            if not isinstance(fixed_details, dict):
                return
            _apply_bracket_dist_update(
                signal_payload,
                sl_value=fixed_details.get("sl_dist", signal_payload.get("sl_dist", MIN_SL)),
                tp_value=fixed_details.get("tp_dist", signal_payload.get("tp_dist", MIN_TP)),
                stage="fixed_sltp",
                preserve_selected=de3_v4_preserve_bracket_fixed_sltp,
            )
            signal_payload["sltp_bracket"] = fixed_details.get("bracket")
            signal_payload["vol_regime"] = fixed_details.get(
                "vol_regime",
                signal_payload.get("vol_regime"),
            )

        def _apply_vol_adj_to_signal(signal_payload: dict, vol_adj: dict) -> None:
            if not isinstance(vol_adj, dict):
                return
            _apply_bracket_dist_update(
                signal_payload,
                sl_value=vol_adj.get("sl_dist", signal_payload.get("sl_dist", MIN_SL)),
                tp_value=vol_adj.get("tp_dist", signal_payload.get("tp_dist", MIN_TP)),
                stage="vol_guard",
                preserve_selected=de3_v4_preserve_bracket_vol_guard,
            )
            signal_payload["vol_regime"] = vol_adj.get("regime", "UNKNOWN")
            if vol_adj.get("adjustment_applied", False):
                signal_payload["size"] = vol_adj["size"]

        if ENABLE_HOSTILE_DAY_GUARD:
            current_day = current_time.astimezone(NY_TZ).date()
            if hostile_day_date != current_day:
                reset_hostile_day(current_day)

        ml_signal = None
        ml_should_eval = (ml_eval_stride <= 1) or ((processed_bars % ml_eval_stride) == 0)
        if ml_strategy.model_loaded and ml_should_eval:
            try:
                ml_eval_df = None
                if not ml_strategy_history_free_eval:
                    if history_df is None:
                        history_df = full_df.iloc[history_start:history_end]
                    ml_eval_df = history_df
                ml_signal = ml_strategy.on_bar(ml_eval_df, current_time)
            except Exception:
                ml_signal = None
        ml_eval = getattr(ml_strategy, "last_eval", None)
        if ml_diag_enabled and ml_eval:
            decision = ml_eval.get("decision")
            if decision != "no_signal" or ml_diag_include_no_signal:
                tracker.record_ml_eval(serialize_trade(ml_eval), ml_diag_max_records)
        if ml_signal and current_session in ml_disabled_sessions:
            if ml_filter_bypass_bar_active:
                ml_signal["ml_filter_bypass_test"] = True
                bypassed = ml_signal.get("bypassed_filters")
                if not isinstance(bypassed, list):
                    bypassed = []
                if "MLSessionDisabled" not in bypassed:
                    bypassed.append("MLSessionDisabled")
                ml_signal["bypassed_filters"] = bypassed
                would_block = ml_signal.get("ml_would_blocked_filters")
                if not isinstance(would_block, list):
                    would_block = []
                if "MLSessionDisabled" not in would_block:
                    would_block.append("MLSessionDisabled")
                ml_signal["ml_would_blocked_filters"] = would_block
                record_filter("MLSessionDisabled", kind="bypass")
            else:
                ml_signal = None
        if ml_only_fast_mode and not ml_signal:
            emit_progress(current_time, bar_close)
            continue

        if history_df is None:
            history_df = full_df.iloc[history_start:history_end]

        candidate_signals = []

        if strategy_executor is not None:
            fast_tasks = [
                (strat, strat_name, history_df, vix_slice)
                for strat, strat_name in fast_strategy_defs
            ]
            fast_results = strategy_executor.map(_eval_fast_strategy_task, fast_tasks)
            for strat, strat_name, signal in fast_results:
                if not signal:
                    continue
                apply_multipliers(signal, strategy_hint=strat_name)
                signal.setdefault("strategy", strat_name)
                candidate_signals.append((1, strat, signal, strat_name))
        else:
            for strat, strat_name in fast_strategy_defs:
                try:
                    if strat_name == "VIXReversionStrategy":
                        signal = strat.on_bar(history_df, vix_slice)
                    else:
                        signal = strat.on_bar(history_df)
                except Exception:
                    signal = None
                if not signal:
                    continue
                apply_multipliers(signal, strategy_hint=strat_name)
                signal.setdefault("strategy", strat_name)
                candidate_signals.append((1, strat, signal, strat_name))

        if strategy_executor is not None:
            standard_tasks = [
                (strat, strat_name, history_df, mnq_slice)
                for strat, strat_name in standard_non_ml_defs
            ]
            standard_results = strategy_executor.map(_eval_standard_strategy_task, standard_tasks)
            for strat, strat_name, signal in standard_results:
                priority = 2
                if not signal:
                    continue
                if (
                    signal.get("strategy", strat_name) == "AuctionReversion"
                    and current_session == "NY_PM"
                    and trend_day_tier > 0
                ):
                    record_filter("AuctionReversionTrendDayTier")
                    continue
                apply_multipliers(signal, strategy_hint=strat_name)
                signal.setdefault("strategy", strat_name)
                candidate_signals.append((priority, strat, signal, strat_name))
        else:
            for strat, strat_name in standard_non_ml_defs:
                priority = 2
                try:
                    if strat_name == "SMTStrategy":
                        signal = strat.on_bar(history_df, mnq_slice)
                    else:
                        signal = strat.on_bar(history_df)
                except Exception:
                    signal = None
                if not signal:
                    continue
                if (
                    signal.get("strategy", strat_name) == "AuctionReversion"
                    and current_session == "NY_PM"
                    and trend_day_tier > 0
                ):
                    record_filter("AuctionReversionTrendDayTier")
                    continue
                apply_multipliers(signal, strategy_hint=strat_name)
                signal.setdefault("strategy", strat_name)
                candidate_signals.append((priority, strat, signal, strat_name))

        for strat, strat_name in standard_ml_defs:
            signal = ml_signal
            priority = 2
            if signal and ml_runner_priority_boost_enabled and bool(signal.get("ml_is_runner", False)):
                priority = min(priority, ml_runner_priority_boost_priority)
            if signal and ml_priority_boost_enabled:
                if not ml_priority_boost_sessions or current_session in ml_priority_boost_sessions:
                    try:
                        ml_conf = float(signal.get("ml_confidence", 0.0))
                    except Exception:
                        ml_conf = 0.0
                    if ml_conf >= ml_priority_boost_min_conf:
                        priority = ml_priority_boost_priority
            if not signal:
                continue
            apply_multipliers(signal, strategy_hint=strat_name)
            signal.setdefault("strategy", strat_name)
            candidate_signals.append((priority, strat, signal, strat_name))

        if (
            de3_ml_fast_mode
            and not de3_meta_enabled
            and not ml_soft_enabled
            and not ml_filter_bypass_enabled
        ):
            def _fast_signal_confidence(sig_payload: Optional[dict]) -> float:
                if not isinstance(sig_payload, dict):
                    return 0.0
                for key in (
                    "ml_confidence",
                    "manifold_confidence",
                    "confidence",
                    "final_score",
                    "score",
                    "opt_wr",
                    "wr",
                    "win_rate",
                ):
                    value = sig_payload.get(key)
                    if value is None:
                        continue
                    try:
                        return float(value)
                    except Exception:
                        continue
                return 0.0

            if de3_manifold_adapt_enabled and candidate_signals:
                filtered_candidate_signals = []
                for priority, strat, sig, s_name in candidate_signals:
                    if not _de3_manifold_adaptation_allows_signal(sig, s_name):
                        record_filter("DE3ManifoldAdapt")
                        continue
                    filtered_candidate_signals.append((priority, strat, sig, s_name))
                candidate_signals = filtered_candidate_signals

            if len(candidate_signals) > 1:
                candidate_signals.sort(key=lambda x: (x[0], -_fast_signal_confidence(x[2])))

            chosen_signal = None
            chosen_name = None
            consensus_supporters: list[str] = []
            consensus_supporters_seen = set()
            consensus_side = None
            consensus_tp_signal = None

            if ENABLE_CONSENSUS_BYPASS and len(candidate_signals) >= 2:
                direction_counts = {"LONG": 0, "SHORT": 0}
                eligible_consensus = {"LONG": [], "SHORT": []}
                for _, _, sig, s_name in candidate_signals:
                    side = sig.get("side")
                    if side not in direction_counts:
                        continue
                    direction_counts[side] += 1
                    eligible_consensus[side].append((sig, s_name))
                if direction_counts["LONG"] >= 2 and direction_counts["SHORT"] == 0:
                    consensus_side = "LONG"
                elif direction_counts["SHORT"] >= 2 and direction_counts["LONG"] == 0:
                    consensus_side = "SHORT"
                if consensus_side:
                    consensus_candidates = eligible_consensus.get(consensus_side, [])
                    tp_values = []
                    sl_values = []
                    for sig, s_name in consensus_candidates:
                        label = format_strategy_label(sig, s_name)
                        if label not in consensus_supporters_seen:
                            consensus_supporters_seen.add(label)
                            consensus_supporters.append(label)
                        try:
                            tp_val = float(sig.get("tp_dist"))
                            sl_val = float(sig.get("sl_dist"))
                        except Exception:
                            continue
                        if not math.isfinite(tp_val) or not math.isfinite(sl_val) or tp_val <= 0 or sl_val <= 0:
                            continue
                        tp_values.append(tp_val)
                        sl_values.append(sl_val)
                    if tp_values and sl_values:
                        if len(tp_values) == 1:
                            selected_tp = float(tp_values[0])
                            selected_sl = float(sl_values[0])
                        elif len(tp_values) == 2:
                            selected_tp = float(sum(tp_values) / 2.0)
                            selected_sl = float(sum(sl_values) / 2.0)
                        else:
                            selected_tp = float(statistics.median(tp_values))
                            selected_sl = float(statistics.median(sl_values))
                        consensus_tp_signal = {"tp_dist": selected_tp, "sl_dist": selected_sl}

            if candidate_signals:
                _, _, base_sig, chosen_name = candidate_signals[0]
                chosen_signal = dict(base_sig)
                chosen_signal.setdefault("strategy", chosen_name)
                chosen_signal.setdefault("sl_dist", MIN_SL)
                chosen_signal.setdefault("tp_dist", MIN_TP)
                if (
                    consensus_side
                    and chosen_signal.get("side") == consensus_side
                    and isinstance(consensus_tp_signal, dict)
                ):
                    chosen_signal["tp_dist"] = consensus_tp_signal.get(
                        "tp_dist",
                        chosen_signal.get("tp_dist", MIN_TP),
                    )
                    chosen_signal["sl_dist"] = consensus_tp_signal.get(
                        "sl_dist",
                        chosen_signal.get("sl_dist", MIN_SL),
                    )
                    chosen_signal["entry_mode"] = "consensus_fast"
                    primary_label = format_strategy_label(chosen_signal, chosen_name)
                    secondary = [
                        label for label in consensus_supporters if label != primary_label
                    ]
                    if secondary:
                        chosen_signal["consensus_contributors"] = secondary
                else:
                    chosen_signal["entry_mode"] = "standard_fast"
                handle_signal(chosen_signal, bar_index=i)
            emit_progress(current_time, bar_close)
            continue

        compute_chop_state = bool(chop_analyzer is not None)
        if compute_chop_state and resample_cache_60 is not None:
            df_60m = resample_cache_60.get_recent(current_time, chop_lookback_const)
        else:
            df_60m = pd.DataFrame()
        if compute_chop_state:
            is_choppy, chop_reason = chop_analyzer.check_market_state(
                _tail_df_for_bar(
                    chop_lookback_const,
                    history_df,
                    history_start,
                    history_end,
                    history_tail_cache,
                ),
                df_60m_current=df_60m,
            )
        else:
            is_choppy = False
            chop_reason = ""
        allowed_chop_side = None
        dynamic_chop_globally_blocked = False
        if is_choppy:
            if "ALLOW_LONG_ONLY" in chop_reason:
                allowed_chop_side = "LONG"
            elif "ALLOW_SHORT_ONLY" in chop_reason:
                allowed_chop_side = "SHORT"
            else:
                # Hard chop is now enforced per-strategy in pre-candidate gating.
                # Keep global flag false to avoid double-blocking later.
                dynamic_chop_globally_blocked = False

        if hostile_day_active:
            filtered_hostile = []
            for priority, strat, sig, s_name in candidate_signals:
                strat_label = str(sig.get("strategy", s_name) or s_name)
                if strat_label in ("DynamicEngine", "MLPhysics"):
                    if ml_filter_bypass_bar_active and strat_label.startswith("MLPhysics"):
                        sig["ml_filter_bypass_test"] = True
                        bypassed = sig.get("bypassed_filters")
                        if not isinstance(bypassed, list):
                            bypassed = []
                        if "HostileDayGate" not in bypassed:
                            bypassed.append("HostileDayGate")
                        sig["bypassed_filters"] = bypassed
                        would_block = sig.get("ml_would_blocked_filters")
                        if not isinstance(would_block, list):
                            would_block = []
                        if "HostileDayGate" not in would_block:
                            would_block.append("HostileDayGate")
                        sig["ml_would_blocked_filters"] = would_block
                        record_filter("HostileDayGate", kind="bypass")
                        filtered_hostile.append((priority, strat, sig, s_name))
                    continue
                filtered_hostile.append((priority, strat, sig, s_name))
            candidate_signals = filtered_hostile

        signal_conf_cache: dict[int, float] = {}

        def signal_confidence(sig):
            cache_key = id(sig)
            cached_val = signal_conf_cache.get(cache_key)
            if cached_val is not None:
                return cached_val
            for key in (
                "ml_confidence",
                "manifold_confidence",
                "confidence",
                "final_score",
                "score",
                "opt_wr",
                "wr",
                "win_rate",
            ):
                if key in sig and sig.get(key) is not None:
                    try:
                        value = float(sig.get(key))
                        signal_conf_cache[cache_key] = value
                        return value
                    except Exception:
                        continue
            signal_conf_cache[cache_key] = 0.0
            return 0.0

        def apply_regime_meta(signal_payload: Optional[dict], fallback_name: Optional[str]) -> tuple[bool, str]:
            nonlocal manifold_checked, manifold_would_block, manifold_shadow_would_block
            if not enabled_filter_regime_manifold:
                return True, ""
            if regime_meta is None or not isinstance(signal_payload, dict):
                return True, ""
            manifold_checked += 1
            allowed, reason, updates = apply_meta_policy(
                signal_payload,
                regime_meta,
                fallback_name=fallback_name,
                default_size=CONTRACTS,
                enforce_side_bias=manifold_enforce_side_bias,
            )
            if reason:
                manifold_would_block += 1
                manifold_reasons[str(reason)] += 1
            if regime_manifold_mode == "shadow":
                if reason:
                    manifold_shadow_would_block += 1
                if updates:
                    updates = dict(updates)
                    updates.pop("size", None)
                    signal_payload.update(updates)
                return True, reason
            if updates:
                signal_payload.update(updates)
            return allowed, reason

        def attach_regime_meta_only(signal_payload: Optional[dict], fallback_name: Optional[str]) -> None:
            if regime_meta is None or not isinstance(signal_payload, dict):
                return
            try:
                _, _, updates = apply_meta_policy(
                    signal_payload,
                    regime_meta,
                    fallback_name=fallback_name,
                    default_size=CONTRACTS,
                    enforce_side_bias=manifold_enforce_side_bias,
                )
            except Exception:
                return
            if updates:
                updates = dict(updates)
                updates.pop("size", None)
                signal_payload.update(updates)

        if candidate_signals:
            active_candidates = []
            for priority, strat, sig, s_name in candidate_signals:
                strat_label = str(sig.get("strategy", s_name) or s_name)
                ml_bypass_candidate = bool(
                    ml_filter_bypass_bar_active and strat_label.startswith("MLPhysics")
                )
                disabled_filter = execution_disabled_filter(strat_label, session_name)
                if disabled_filter:
                    if ml_bypass_candidate:
                        record_filter(disabled_filter, kind="bypass")
                        sig["ml_filter_bypass_test"] = True
                        bypassed = sig.get("bypassed_filters")
                        if not isinstance(bypassed, list):
                            bypassed = []
                        if disabled_filter not in bypassed:
                            bypassed.append(disabled_filter)
                        sig["bypassed_filters"] = bypassed
                        would_block = sig.get("ml_would_blocked_filters")
                        if not isinstance(would_block, list):
                            would_block = []
                        if disabled_filter not in would_block:
                            would_block.append(disabled_filter)
                        sig["ml_would_blocked_filters"] = would_block
                    else:
                        # Track the signal, but don't let it affect consensus or execution.
                        record_filter(disabled_filter, kind="disabled")
                        continue
                rm_ok, rm_reason = apply_regime_meta(sig, s_name)
                if not rm_ok and regime_manifold_mode == "enforce":
                    if ml_bypass_candidate:
                        record_filter("RegimeManifold", kind="bypass")
                        sig["ml_filter_bypass_test"] = True
                        bypassed = sig.get("bypassed_filters")
                        if not isinstance(bypassed, list):
                            bypassed = []
                        if "RegimeManifold" not in bypassed:
                            bypassed.append("RegimeManifold")
                        sig["bypassed_filters"] = bypassed
                        would_block = sig.get("ml_would_blocked_filters")
                        if not isinstance(would_block, list):
                            would_block = []
                        if "RegimeManifold" not in would_block:
                            would_block.append("RegimeManifold")
                        sig["ml_would_blocked_filters"] = would_block
                    else:
                        manifold_blocked += 1
                        record_filter("RegimeManifold")
                        continue
                active_candidates.append((priority, strat, sig, s_name))
            candidate_signals = active_candidates
        vol_regime_current = None
        try:
            vol_regime_current = _get_vol_regime()
        except Exception:
            vol_regime_current = None

        if de3_meta_enabled and candidate_signals:
            de3_meta_common_ctx = _de3_meta_common_context(
                history_df,
                current_time,
                bar_index=i,
            )
            filtered_signals = []
            for priority, strat, sig, s_name in candidate_signals:
                strat_label = str(sig.get("strategy", s_name) or s_name)
                if strat_label != "DynamicEngine3":
                    filtered_signals.append((priority, strat, sig, s_name))
                    continue
                de3_meta_checked += 1
                meta_eval = _de3_meta_policy(
                    sig,
                    history_df,
                    current_time,
                    vol_regime_current,
                    common_ctx=de3_meta_common_ctx,
                    bar_index=i,
                )
                sig["de3_meta_mode"] = de3_meta_mode
                sig["de3_meta_score"] = float(meta_eval.get("score", 100.0))
                sig["de3_meta_would_block"] = bool(meta_eval.get("would_block", False))
                sig["de3_meta_reasons"] = list(meta_eval.get("reasons", []))
                context = meta_eval.get("context", {}) or {}
                for key, value in context.items():
                    sig[f"de3_meta_{key}"] = value
                if bool(meta_eval.get("would_block", False)):
                    de3_meta_would_block += 1
                    reasons = meta_eval.get("reasons", []) or ["score"]
                    for reason in reasons:
                        de3_meta_reasons[str(reason)] += 1
                    if de3_meta_mode == "block":
                        de3_meta_blocked += 1
                        record_filter("DE3MetaPolicy")
                        if de3_meta_log_decisions:
                            logging.info(
                                "DE3 meta block | sub=%s side=%s score=%.2f reasons=%s",
                                sig.get("sub_strategy"),
                                sig.get("side"),
                                float(sig.get("de3_meta_score", 0.0)),
                                ",".join(reasons),
                            )
                        continue
                    de3_meta_shadow += 1
                filtered_signals.append((priority, strat, sig, s_name))
            candidate_signals = filtered_signals

        if de3_manifold_adapt_enabled and candidate_signals:
            filtered_signals = []
            for priority, strat, sig, s_name in candidate_signals:
                if not _de3_manifold_adaptation_allows_signal(sig, s_name):
                    record_filter("DE3ManifoldAdapt")
                    continue
                filtered_signals.append((priority, strat, sig, s_name))
            candidate_signals = filtered_signals

        is_asia_session = session_name == "ASIA"
        asia_trend_bias_side = None
        if (
            asia_calib_enabled
            and is_asia_session
            and asia_ema_fast_state is not None
            and asia_ema_slow_state is not None
            and asia_ema_obs >= asia_trend_required_len
            and len(asia_ema_fast_window) >= asia_trend_ema_slope_bars
        ):
            fast_val = float(asia_ema_fast_state)
            slow_val = float(asia_ema_slow_state)
            slope = fast_val - float(asia_ema_fast_window[0])
            if fast_val > slow_val and slope > 0 and (fast_val - slow_val) >= asia_trend_min_sep:
                asia_trend_bias_side = "LONG"
            elif fast_val < slow_val and slope < 0 and (slow_val - fast_val) >= asia_trend_min_sep:
                asia_trend_bias_side = "SHORT"

        asia_viable = True
        asia_viable_reason = None
        asia_viability_globally_blocked = False
        if is_asia_session:
            asia_viable, asia_viable_reason = asia_viability_gate(
                history_df,
                ts=current_time,
                session=session_name,
            )
            # ASIA viability is now enforced per-candidate. Keep global flag false
            # to avoid duplicate blocking in downstream execution loop.
            asia_viability_globally_blocked = False

        if candidate_signals:
            gated_candidates = []
            for priority, strat, sig, s_name in candidate_signals:
                strat_label = str(sig.get("strategy", s_name) or s_name)
                ml_bypass_candidate = bool(
                    ml_filter_bypass_bar_active and strat_label.startswith("MLPhysics")
                )
                if enabled_filter_pre_candidate:
                    gate_ok, gate_filter, gate_reason, gate_profile = evaluate_pre_signal_gate(
                        cfg=CONFIG,
                        session_name=session_name,
                        strategy_label=strat_label,
                        side=sig.get("side"),
                        asia_viable=asia_viable,
                        asia_reason=asia_viable_reason,
                        asia_trend_bias_side=asia_trend_bias_side,
                        is_choppy=bool(is_choppy),
                        chop_reason=chop_reason,
                        allowed_chop_side=allowed_chop_side,
                    )
                else:
                    gate_ok, gate_filter, gate_reason, gate_profile = True, None, "", {}
                sig["gate_profile"] = gate_profile
                if not gate_ok:
                    filter_name = gate_filter or "PreCandidateGate"
                    if ml_bypass_candidate:
                        record_filter(filter_name, kind="bypass")
                        sig["ml_filter_bypass_test"] = True
                        bypassed = sig.get("bypassed_filters")
                        if not isinstance(bypassed, list):
                            bypassed = []
                        if filter_name not in bypassed:
                            bypassed.append(filter_name)
                        sig["bypassed_filters"] = bypassed
                        would_block = sig.get("ml_would_blocked_filters")
                        if not isinstance(would_block, list):
                            would_block = []
                        if filter_name not in would_block:
                            would_block.append(filter_name)
                        sig["ml_would_blocked_filters"] = would_block
                    else:
                        record_filter(filter_name)
                        continue
                gated_candidates.append((priority, strat, sig, s_name))
            candidate_signals = gated_candidates

        def _ml_vol_regime_check(sig_payload: dict, session_label: str, vol_regime_value) -> bool:
            if not enabled_filter_ml_vol_regime:
                return True
            return ml_vol_regime_ok(
                sig_payload,
                session_label,
                vol_regime_value,
                asia_viable=asia_viable,
            )

        def ml_soft_gate_eligible(sig: dict) -> bool:
            if not consensus_ml_ok(sig):
                return False
            if not _ml_vol_regime_check(sig, session_name, vol_regime_current):
                return False
            sig_copy = dict(sig)
            fixed_ok, fixed_details = _apply_fixed_sltp_check(sig_copy)
            if not fixed_ok:
                return False
            if fixed_details:
                _apply_fixed_details_to_signal(sig_copy, fixed_details)
            sig_side = sig_copy.get("side")
            sig_tp = sig_copy.get("tp_dist", MIN_TP)
            is_feasible, _ = _target_feasibility_check(sig_side, sig_tp)
            if not is_feasible and _asia_target_feasibility_override_check(sig_side, sig_tp):
                return True
            return is_feasible

        if ml_soft_enabled and candidate_signals:
            if ml_soft_sessions and session_name not in ml_soft_sessions:
                pass
            else:
                ml_best = None
                ml_best_conf = -math.inf
                for priority, _, sig, s_name in candidate_signals:
                    if not str(sig.get("strategy", s_name)).startswith("MLPhysics"):
                        continue
                    conf = signal_confidence(sig)
                    if ml_best is None or conf > ml_best_conf:
                        ml_best = (priority, sig, s_name)
                        ml_best_conf = conf
                if ml_best is not None:
                    _, ml_sig, _ = ml_best
                    if ml_best_conf >= ml_soft_min_conf and ml_soft_gate_eligible(ml_sig):
                        blocked_priorities = set()
                        if ml_soft_block_standard:
                            blocked_priorities.add(2)
                        if ml_soft_block_fast:
                            blocked_priorities.add(1)
                        ml_side = ml_sig.get("side")
                        if ml_side in ("LONG", "SHORT") and blocked_priorities:
                            new_candidates = []
                            for priority, strat, sig, s_name in candidate_signals:
                                if str(sig.get("strategy", s_name)).startswith("MLPhysics"):
                                    new_candidates.append((priority, strat, sig, s_name))
                                    continue
                                if sig.get("side") != ml_side and priority in blocked_priorities:
                                    record_filter("MLSoftGate")
                                    continue
                                new_candidates.append((priority, strat, sig, s_name))
                            candidate_signals = new_candidates

        if len(candidate_signals) > 1:
            candidate_signals.sort(key=lambda x: (x[0], -signal_confidence(x[2])))

        asia_soft_ext_enabled = bool(
            asia_soft_ext_feature_enabled
            and is_asia_session
            and asia_viable
        )

        direction_counts = {"LONG": 0, "SHORT": 0}
        eligible_consensus = {"LONG": [], "SHORT": []}
        smt_side = None
        for _, _, sig, s_name in candidate_signals:
            side = sig.get("side")
            if s_name == "SMTStrategy":
                smt_side = side
            if side not in direction_counts:
                continue
            if str(sig.get("strategy", "")).startswith("MLPhysics"):
                if not consensus_ml_ok(sig):
                    continue
                if not _ml_vol_regime_check(sig, session_name, vol_regime_current):
                    continue
            weight = 2 if s_name == "SMTStrategy" else 1
            direction_counts[side] += weight
            eligible_consensus[side].append((sig, s_name))

        consensus_side = None
        max_count = max(direction_counts.values()) if direction_counts else 0
        if max_count >= 2:
            if direction_counts["LONG"] != direction_counts["SHORT"]:
                consensus_side = "LONG" if direction_counts["LONG"] > direction_counts["SHORT"] else "SHORT"
            elif smt_side:
                consensus_side = smt_side

        if not ENABLE_CONSENSUS_BYPASS:
            consensus_side = None

        consensus_tp_source = None
        consensus_tp_signal = None
        consensus_supporters: list[str] = []
        consensus_supporters_seen = set()
        if consensus_side:
            consensus_candidates = eligible_consensus.get(consensus_side, [])
            if consensus_candidates:
                tp_values = []
                sl_values = []
                candidate_brackets = []
                for sig, s_name in consensus_candidates:
                    label = format_strategy_label(sig, s_name)
                    if label not in consensus_supporters_seen:
                        consensus_supporters_seen.add(label)
                        consensus_supporters.append(label)
                    try:
                        tp_val = float(sig.get("tp_dist"))
                        sl_val = float(sig.get("sl_dist"))
                    except Exception:
                        continue
                    if not math.isfinite(tp_val) or not math.isfinite(sl_val) or tp_val <= 0 or sl_val <= 0:
                        continue
                    tp_values.append(tp_val)
                    sl_values.append(sl_val)
                    candidate_brackets.append((tp_val, sl_val, s_name))

                if tp_values and sl_values:
                    tp_values_sorted = sorted(tp_values)
                    sl_values_sorted = sorted(sl_values)
                    if len(tp_values_sorted) == 1:
                        desired_tp = tp_values_sorted[0]
                        desired_sl = sl_values_sorted[0]
                        consensus_tp_source = "single"
                    elif len(tp_values_sorted) == 2:
                        desired_tp = sum(tp_values_sorted) / 2.0
                        desired_sl = sum(sl_values_sorted) / 2.0
                        consensus_tp_source = "mean(2)"
                    else:
                        desired_tp = float(statistics.median(tp_values_sorted))
                        desired_sl = float(statistics.median(sl_values_sorted))
                        consensus_tp_source = f"median({len(tp_values_sorted)})"

                    selected_tp = desired_tp
                    selected_sl = desired_sl
                    feasibility_ok = True
                    if history_df is not None and not history_df.empty:
                        feasibility_ok, _ = _target_feasibility_check(consensus_side, selected_tp)
                    if not feasibility_ok and candidate_brackets:
                        candidate_brackets.sort(key=lambda item: item[0])
                        smallest_tp = candidate_brackets[0][0]
                        fallback = None
                        for tp_val, sl_val, s_name in candidate_brackets:
                            ok, _ = _target_feasibility_check(consensus_side, tp_val)
                            if ok:
                                fallback = (tp_val, sl_val, s_name)
                                break
                        if fallback:
                            selected_tp, selected_sl, fallback_source = fallback
                            consensus_tp_source = f"{consensus_tp_source}->feasible:{fallback_source}"
                        else:
                            selected_tp = smallest_tp
                            selected_sl = candidate_brackets[0][1]
                            consensus_tp_source = f"{consensus_tp_source}->smallest"

                    consensus_tp_signal = {"tp_dist": selected_tp, "sl_dist": selected_sl}

        signal_executed = False

        for _, _, sig, strat_name in candidate_signals:
            signal = sig
            strat_label = str(signal.get("strategy", strat_name) or strat_name)
            bypassed_filters: list[str] = []
            ml_filter_bypass_signal = bool(
                ml_filter_bypass_bar_active and strat_label.startswith("MLPhysics")
            )

            def _mark_ml_filter_bypass(filter_name: str, local_filters: Optional[list[str]] = None) -> bool:
                if not ml_filter_bypass_signal:
                    return False
                record_filter(filter_name, kind="bypass")
                if local_filters is not None and filter_name not in local_filters:
                    local_filters.append(filter_name)
                existing = signal.get("bypassed_filters")
                if not isinstance(existing, list):
                    existing = []
                if filter_name not in existing:
                    existing.append(filter_name)
                signal["bypassed_filters"] = existing
                ml_existing = signal.get("ml_would_blocked_filters")
                if not isinstance(ml_existing, list):
                    ml_existing = []
                if filter_name not in ml_existing:
                    ml_existing.append(filter_name)
                signal["ml_would_blocked_filters"] = ml_existing
                signal["ml_filter_bypass_test"] = True
                return True

            manifold_hard_only_signal = (
                manifold_bt_hard_filters_only and strat_label == "ManifoldStrategy"
            )
            if ml_filter_bypass_signal:
                cur_sl = _coerce_float(signal.get("sl_dist", 0.0), 0.0)
                cur_tp = _coerce_float(signal.get("tp_dist", 0.0), 0.0)
                if cur_sl <= 0 or cur_tp <= 0:
                    signal["sl_dist"] = sltp_exec_min_sl if cur_sl <= 0 else cur_sl
                    signal["tp_dist"] = sltp_exec_min_tp if cur_tp <= 0 else cur_tp
                    _mark_ml_filter_bypass("InvalidSLTP", bypassed_filters)

            if flip_tracker is not None and flip_tracker.enabled:
                flip_context = {
                    "signal": signal,
                    "bar_index": i,
                    "current_time": current_time,
                    "history_df": history_df,
                    "session_name": session_name,
                    "vol_regime": vol_regime_current,
                }
            else:
                flip_context = None
            if news_globally_blocked:
                if not _mark_ml_filter_bypass("NewsFilter", bypassed_filters):
                    record_filter("NewsFilter")
                    continue
            if dynamic_chop_globally_blocked:
                if not _mark_ml_filter_bypass("DynamicChop", bypassed_filters):
                    record_filter("DynamicChop")
                    continue
            if asia_viability_globally_blocked:
                if not _mark_ml_filter_bypass("AsiaViabilityGate", bypassed_filters):
                    record_filter("AsiaViabilityGate")
                    continue
            if (
                consensus_side
                and signal.get("side") != consensus_side
                and not manifold_hard_only_signal
                and not ml_filter_bypass_signal
            ):
                continue

            signal.setdefault("sl_dist", MIN_SL)
            signal.setdefault("tp_dist", MIN_TP)
            signal.setdefault("strategy", strat_name)
            trend_day_counter = False
            if trend_day_tier > 0 and trend_day_dir:
                trend_day_counter = (
                    (trend_day_dir == "down" and signal["side"] == "LONG")
                    or (trend_day_dir == "up" and signal["side"] == "SHORT")
                )
                signal["trend_day_tier"] = trend_day_tier
                signal["trend_day_dir"] = trend_day_dir

            if manifold_hard_only_signal:
                fixed_ok, fixed_details = _apply_fixed_sltp_check(signal)
                if not fixed_ok:
                    record_filter("FixedSLTP")
                    continue
                if fixed_details:
                    _apply_fixed_details_to_signal(signal, fixed_details)
                    log_fixed_sltp(fixed_details, signal.get("strategy"))

                signal_tp = signal.get("tp_dist", MIN_TP)
                is_feasible, _ = _target_feasibility_check(signal["side"], signal_tp)
                if not is_feasible:
                    if _asia_target_feasibility_override_check(signal["side"], signal_tp):
                        record_filter("TargetFeasibility", kind="bypass")
                        is_feasible = True
                    if not is_feasible:
                        record_filter("TargetFeasibility")
                        continue

                should_trade, vol_adj = _volatility_guard_check(
                    signal.get("sl_dist", MIN_SL),
                    signal.get("tp_dist", MIN_TP),
                    _signal_base_size(signal, CONTRACTS),
                )
                if not should_trade:
                    record_filter("VolatilityGuardrail")
                    continue

                _apply_vol_adj_to_signal(signal, vol_adj)
                signal["entry_mode"] = "manifold_hard_only"
                handle_signal(signal, bar_index=i)
                signal_executed = True
                break

            origin_strategy = signal.get("strategy", strat_name)
            origin_sub_strategy = signal.get("sub_strategy")
            allow_rescue = not str(signal.get("strategy", "")).startswith("MLPhysics")
            is_rescued = False
            consensus_rescued = False
            consensus_bypass_allowed = True
            rescue_bypass_allowed = True
            primary_label = format_strategy_label(signal, strat_name)
            consensus_secondary = [
                label for label in consensus_supporters if label != primary_label
            ]

            if consensus_side and signal.get("side") == consensus_side:
                rescue_side = "SHORT" if signal["side"] == "LONG" else "LONG"
                consensus_rescue_evaluated = False
                consensus_rescue_candidate = None

                def try_consensus_rescue(trigger: str) -> bool:
                    nonlocal signal, is_rescued, consensus_rescued, consensus_bypass_allowed, manifold_blocked
                    nonlocal consensus_rescue_evaluated, consensus_rescue_candidate
                    if not allow_rescue:
                        return False
                    if is_rescued:
                        return False
                    if trend_day_tier > 0 and trend_day_dir:
                        if (trend_day_dir == "down" and rescue_side == "LONG") or (
                            trend_day_dir == "up" and rescue_side == "SHORT"
                        ):
                            record_filter(f"TrendDayTier{trend_day_tier}", kind="rescue")
                            return False
                    if mom_rescue_banned(current_time, origin_strategy, origin_sub_strategy):
                        record_filter("MomRescueBan")
                        return False
                    if DISABLE_CONTINUATION_NY and session_name in ("NY_AM", "NY_PM"):
                        return False
                    if hostile_day_active:
                        return False
                    if not consensus_rescue_evaluated:
                        potential_rescue = continuation_manager.get_active_continuation_signal(
                            history_df,
                            current_time,
                            rescue_side,
                            current_price=bar_close,
                            trend_day_series=trend_day_series,
                            signal_mode=continuation_signal_mode,
                        )
                        if not continuation_rescue_allowed(
                            potential_rescue,
                            rescue_side,
                            current_time,
                            bar_close,
                            history_df,
                            bar_index=i,
                            vol_regime_hint=vol_regime_current,
                        ):
                            potential_rescue = None
                        consensus_rescue_candidate = potential_rescue
                        consensus_rescue_evaluated = True
                    potential_rescue = consensus_rescue_candidate
                    if not potential_rescue:
                        return False
                    if continuation_no_bypass and continuation_core_trigger(trigger):
                        return False
                    rescue_blocked, _ = _trend_check(potential_rescue["side"])
                    if rescue_blocked:
                        return False
                    rm_ok, _ = apply_regime_meta(
                        potential_rescue,
                        str(potential_rescue.get("strategy", strat_name)),
                    )
                    if not rm_ok and regime_manifold_mode == "enforce":
                        manifold_blocked += 1
                        record_filter("RegimeManifold")
                        return False
                    signal = potential_rescue
                    signal.setdefault("strategy", strat_name)
                    if flip_context is not None:
                        flip_context["signal"] = signal
                    signal["rescue_from_strategy"] = origin_strategy
                    if origin_sub_strategy:
                        signal["rescue_from_sub_strategy"] = origin_sub_strategy
                    signal["rescue_trigger"] = trigger
                    signal["entry_mode"] = "rescued"
                    if consensus_secondary:
                        signal["consensus_contributors"] = consensus_secondary
                    consensus_bypass_allowed = not continuation_no_bypass
                    if consensus_bypass_allowed:
                        add_bypass_filters_from_trigger(bypassed_filters, trigger)
                        if bypassed_filters:
                            signal["bypassed_filters"] = list(bypassed_filters)
                    is_rescued = True
                    consensus_rescued = consensus_bypass_allowed
                    return True

                if trend_day_counter:
                    if try_consensus_rescue(f"TrendDayTier{trend_day_tier}"):
                        record_filter(f"TrendDayTier{trend_day_tier}", kind="rescue")
                    else:
                        if not _mark_ml_filter_bypass(f"TrendDayTier{trend_day_tier}", bypassed_filters):
                            record_filter(f"TrendDayTier{trend_day_tier}")
                            continue

                if consensus_tp_signal is not None:
                    signal["tp_dist"] = consensus_tp_signal.get("tp_dist", signal.get("tp_dist", MIN_TP))
                    signal["sl_dist"] = consensus_tp_signal.get("sl_dist", signal.get("sl_dist", MIN_SL))
                fixed_ok, fixed_details = _apply_fixed_sltp_check(signal)
                if not fixed_ok:
                    if not _mark_ml_filter_bypass("FixedSLTP", bypassed_filters):
                        record_filter("FixedSLTP")
                        continue
                if fixed_details:
                    _apply_fixed_details_to_signal(signal, fixed_details)
                    log_fixed_sltp(fixed_details, signal.get("strategy"))
                signal_tp = signal.get("tp_dist", MIN_TP)
                is_feasible, _ = _target_feasibility_check(signal["side"], signal_tp)
                if not is_feasible:
                    if _asia_target_feasibility_override_check(signal["side"], signal_tp):
                        record_filter("TargetFeasibility", kind="bypass")
                        bypassed_filters.append("TargetFeasibility")
                        is_feasible = True
                    if not is_feasible:
                        if try_consensus_rescue("TargetFeasibility"):
                            record_filter("TargetFeasibility", kind="rescue")
                        else:
                            if not _mark_ml_filter_bypass("TargetFeasibility", bypassed_filters):
                                record_filter("TargetFeasibility")
                                continue
                if not consensus_rescued:
                    regime_blocked, _ = _regime_block_check(signal["side"])
                    if regime_blocked:
                        if try_consensus_rescue("RegimeBlocker"):
                            record_filter("RegimeBlocker", kind="rescue")
                        else:
                            if not _mark_ml_filter_bypass("RegimeBlocker", bypassed_filters):
                                record_filter("RegimeBlocker")
                                continue
                if not consensus_rescued:
                    dir_blocked, _ = _directional_loss_check(signal["side"])
                    if dir_blocked:
                        if try_consensus_rescue("DirectionalLossBlocker"):
                            record_filter("DirectionalLossBlocker", kind="rescue")
                        else:
                            if not _mark_ml_filter_bypass("DirectionalLossBlocker", bypassed_filters):
                                record_filter("DirectionalLossBlocker")
                                continue
                if not consensus_rescued:
                    trend_blocked, _ = _trend_check(signal["side"])
                    if trend_blocked:
                        if try_consensus_rescue("TrendFilter"):
                            record_filter("TrendFilter", kind="rescue")
                        else:
                            if not _mark_ml_filter_bypass("TrendFilter", bypassed_filters):
                                record_filter("TrendFilter")
                                continue
                if not consensus_rescued:
                    chop_blocked, chop_reason = _chop_check(signal["side"], trend_state="NEUTRAL")
                    if chop_blocked and asia_calib_enabled and is_asia_session:
                        if asia_chop_override(
                            chop_reason,
                            signal["side"],
                            asia_trend_bias_side,
                            asia_chop_cfg,
                        ):
                            record_filter("ChopFilter", kind="bypass")
                            bypassed_filters.append("ChopFilter")
                            chop_blocked = False
                    if chop_blocked:
                        if chop_reason:
                            reason_lc = str(chop_reason).lower()
                            if "wait for breakout" in reason_lc or "range too tight" in reason_lc:
                                if not _mark_ml_filter_bypass("ChopFilter", bypassed_filters):
                                    record_filter("ChopFilter")
                                    continue
                        if try_consensus_rescue("ChopFilter"):
                            record_filter("ChopFilter", kind="rescue")
                        else:
                            if not _mark_ml_filter_bypass("ChopFilter", bypassed_filters):
                                record_filter("ChopFilter")
                                continue
                if not consensus_rescued:
                    should_trade, vol_adj = _volatility_guard_check(
                        signal.get("sl_dist", MIN_SL),
                        signal.get("tp_dist", MIN_TP),
                        _signal_base_size(signal, CONTRACTS),
                    )
                    if not should_trade:
                        if try_consensus_rescue("VolatilityGuardrail"):
                            record_filter("VolatilityGuardrail", kind="rescue")
                        else:
                            if not _mark_ml_filter_bypass("VolatilityGuardrail", bypassed_filters):
                                record_filter("VolatilityGuardrail")
                                continue
                    _apply_vol_adj_to_signal(signal, vol_adj)
                    if not _ml_vol_regime_check(signal, session_name, signal["vol_regime"]):
                        if not _mark_ml_filter_bypass("MLVolRegimeGuard", bypassed_filters):
                            record_filter("MLVolRegimeGuard")
                            continue
                if not consensus_rescued:
                    if consensus_secondary:
                        signal["consensus_contributors"] = consensus_secondary
                    consensus_bypassed: list[str] = []
                    rej_blocked, _ = _rejection_check(signal["side"])
                    if rej_blocked:
                        consensus_bypassed.append("RejectionFilter")
                    range_bias_blocked = allowed_chop_side is not None and signal["side"] != allowed_chop_side
                    if range_bias_blocked:
                        consensus_bypassed.append("ChopRangeBias")
                    impulse_blocked, _ = _impulse_check(signal["side"])
                    if impulse_blocked:
                        consensus_bypassed.append("ImpulseFilter")
                    ext_blocked, _ = _extension_check(signal["side"])
                    ext_soft_passed = False
                    if ext_blocked and asia_soft_ext_enabled:
                        soft_score = asia_soft_ext_base - asia_soft_ext_penalty
                        if soft_score >= asia_soft_ext_threshold:
                            ext_blocked = False
                            ext_soft_passed = True
                    if ext_blocked or ext_soft_passed:
                        consensus_bypassed.append("ExtensionFilter")
                    bank_blocked, bank_reason = _bank_check(signal["side"])
                    upg_trend_blocked, upg_trend_reason = _trend_check(signal["side"])
                    legacy_blocked, _ = _legacy_trend_check(signal["side"])
                    upgraded_reasons = []
                    if bank_blocked:
                        upgraded_reasons.append(f"Bank: {bank_reason}")
                    if upg_trend_blocked:
                        upgraded_reasons.append(f"Trend: {upg_trend_reason}")
                    final_blocked = False
                    arb_blocked = False
                    if legacy_blocked and upgraded_reasons:
                        final_blocked = True
                    elif not legacy_blocked and upgraded_reasons:
                        if enabled_filter_filter_arbitrator:
                            arb = filter_arbitrator.arbitrate(
                                df=history_df,
                                side=signal["side"],
                                legacy_blocked=False,
                                legacy_reason="",
                                upgraded_blocked=True,
                                upgraded_reason="|".join(upgraded_reasons),
                                current_price=bar_close,
                                tp_dist=signal.get("tp_dist"),
                                sl_dist=signal.get("sl_dist"),
                            )
                            if not arb.allow_trade:
                                final_blocked = True
                                arb_blocked = True
                    if final_blocked:
                        if legacy_blocked:
                            consensus_bypassed.append("LegacyTrend")
                        if bank_blocked:
                            consensus_bypassed.append("BankLevelQuarterFilter")
                        if upg_trend_blocked:
                            consensus_bypassed.append("TrendFilter")
                        if arb_blocked:
                            consensus_bypassed.append("FilterArbitrator")
                    if consensus_bypassed:
                        existing_bypassed = signal.get("bypassed_filters")
                        if not isinstance(existing_bypassed, list):
                            existing_bypassed = []
                        for name in consensus_bypassed:
                            if name not in existing_bypassed:
                                existing_bypassed.append(name)
                        signal["bypassed_filters"] = existing_bypassed
                    if not is_rescued:
                        signal["entry_mode"] = "consensus"
                    if ml_filter_bypass_signal:
                        signal["entry_mode"] = "ml_filter_bypass_test"
                    handle_signal(signal, bar_index=i)
                    signal_executed = True
                    break

            rescue_side = "SHORT" if signal["side"] == "LONG" else "LONG"
            if not allow_rescue or is_rescued:
                potential_rescue = None
            elif DISABLE_CONTINUATION_NY and session_name in ("NY_AM", "NY_PM"):
                potential_rescue = None
            elif hostile_day_active:
                potential_rescue = None
            else:
                potential_rescue = continuation_manager.get_active_continuation_signal(
                    history_df,
                    current_time,
                    rescue_side,
                    current_price=bar_close,
                    trend_day_series=trend_day_series,
                    signal_mode=continuation_signal_mode,
                )
            if not continuation_rescue_allowed(
                potential_rescue,
                rescue_side,
                current_time,
                bar_close,
                history_df,
                bar_index=i,
                vol_regime_hint=vol_regime_current,
            ):
                potential_rescue = None

            def try_rescue(trigger: str) -> bool:
                nonlocal signal, is_rescued, potential_rescue, rescue_bypass_allowed, manifold_blocked
                if mom_rescue_banned(current_time, origin_strategy, origin_sub_strategy):
                    record_filter("MomRescueBan")
                    return False
                if continuation_no_bypass and continuation_core_trigger(trigger):
                    return False
                if potential_rescue and not is_rescued:
                    rescue_blocked, _ = _trend_check(potential_rescue["side"])
                    if rescue_blocked:
                        return False
                    rm_ok, _ = apply_regime_meta(
                        potential_rescue,
                        str(potential_rescue.get("strategy", strat_name)),
                    )
                    if not rm_ok and regime_manifold_mode == "enforce":
                        manifold_blocked += 1
                        record_filter("RegimeManifold")
                        return False
                    signal = potential_rescue
                    signal.setdefault("strategy", strat_name)
                    if flip_context is not None:
                        flip_context["signal"] = signal
                    signal["rescue_from_strategy"] = origin_strategy
                    if origin_sub_strategy:
                        signal["rescue_from_sub_strategy"] = origin_sub_strategy
                    signal["rescue_trigger"] = trigger
                    is_rescued = True
                    rescue_bypass_allowed = not continuation_no_bypass
                    if rescue_bypass_allowed:
                        add_bypass_filters_from_trigger(bypassed_filters, trigger)
                        if bypassed_filters:
                            signal["bypassed_filters"] = list(bypassed_filters)
                    potential_rescue = None
                    return True
                return False

            if trend_day_counter:
                if not try_rescue(f"TrendDayTier{trend_day_tier}"):
                    if not _mark_ml_filter_bypass(f"TrendDayTier{trend_day_tier}", bypassed_filters):
                        record_filter(f"TrendDayTier{trend_day_tier}")
                        continue
                record_filter(f"TrendDayTier{trend_day_tier}", kind="rescue")

            fixed_ok, fixed_details = _apply_fixed_sltp_check(signal)
            if not fixed_ok:
                if not _mark_ml_filter_bypass("FixedSLTP", bypassed_filters):
                    record_filter("FixedSLTP")
                    continue
            if fixed_details:
                _apply_fixed_details_to_signal(signal, fixed_details)
                log_fixed_sltp(fixed_details, signal.get("strategy"))

            signal_tp = signal.get("tp_dist", MIN_TP)
            is_feasible, _ = _target_feasibility_check(signal["side"], signal_tp)
            if not is_feasible:
                if _asia_target_feasibility_override_check(signal["side"], signal_tp):
                    record_filter("TargetFeasibility", kind="bypass")
                    bypassed_filters.append("TargetFeasibility")
                    is_feasible = True
                if not is_feasible:
                    if not _mark_ml_filter_bypass("TargetFeasibility", bypassed_filters):
                        record_filter("TargetFeasibility")
                        continue

            rej_blocked, _ = _rejection_check(signal["side"])
            range_bias_blocked = allowed_chop_side is not None and signal["side"] != allowed_chop_side
            if rej_blocked or range_bias_blocked:
                rescue_reasons = []
                if rej_blocked:
                    rescue_reasons.append("RejectionFilter")
                if range_bias_blocked:
                    rescue_reasons.append("ChopRangeBias")
                rescue_reason = "+".join(rescue_reasons) if rescue_reasons else "RejectionFilter"
                if not try_rescue(rescue_reason):
                    blocked_here = False
                    if rej_blocked and not _mark_ml_filter_bypass("RejectionFilter", bypassed_filters):
                        record_filter("RejectionFilter")
                        blocked_here = True
                    if range_bias_blocked and not _mark_ml_filter_bypass("ChopRangeBias", bypassed_filters):
                        record_filter("ChopRangeBias")
                        blocked_here = True
                    if blocked_here:
                        continue
                if rej_blocked:
                    record_filter("RejectionFilter", kind="rescue")
                if range_bias_blocked:
                    record_filter("ChopRangeBias", kind="rescue")

            dir_blocked, _ = _directional_loss_check(signal["side"])
            if dir_blocked:
                if not try_rescue("DirectionalLossBlocker"):
                    if not _mark_ml_filter_bypass("DirectionalLossBlocker", bypassed_filters):
                        record_filter("DirectionalLossBlocker")
                        continue
                record_filter("DirectionalLossBlocker", kind="rescue")

            impulse_blocked, _ = _impulse_check(signal["side"])
            if impulse_blocked:
                if not try_rescue("ImpulseFilter"):
                    if not _mark_ml_filter_bypass("ImpulseFilter", bypassed_filters):
                        record_filter("ImpulseFilter")
                        continue
                record_filter("ImpulseFilter", kind="rescue")

            regime_blocked, _ = _regime_block_check(signal["side"])
            if regime_blocked:
                if not _mark_ml_filter_bypass("RegimeBlocker", bypassed_filters):
                    record_filter("RegimeBlocker")
                    continue

            upgraded_reasons = []
            struct_blocked, struct_reason = _structure_check(signal["side"])
            if struct_blocked:
                upgraded_reasons.append(f"Structure: {struct_reason}")

            bank_blocked, bank_reason = _bank_check(signal["side"])
            if bank_blocked:
                upgraded_reasons.append(f"Bank: {bank_reason}")

            upg_trend_blocked, upg_trend_reason = _trend_check(signal["side"])
            if upg_trend_blocked:
                upgraded_reasons.append(f"Trend: {upg_trend_reason}")

            legacy_blocked, _ = _legacy_trend_check(signal["side"])

            final_blocked = False
            if legacy_blocked and upgraded_reasons:
                final_blocked = True
            elif not legacy_blocked and upgraded_reasons:
                if enabled_filter_filter_arbitrator:
                    arb = filter_arbitrator.arbitrate(
                        df=history_df,
                        side=signal["side"],
                        legacy_blocked=False,
                        legacy_reason="",
                        upgraded_blocked=True,
                        upgraded_reason="|".join(upgraded_reasons),
                        current_price=bar_close,
                        tp_dist=signal.get("tp_dist"),
                        sl_dist=signal.get("sl_dist"),
                    )
                    if not arb.allow_trade:
                        final_blocked = True

            blocked_filters = []
            if legacy_blocked:
                blocked_filters.append("LegacyTrend")
            if struct_blocked:
                blocked_filters.append("StructureBlocker")
            if bank_blocked:
                blocked_filters.append("BankLevelQuarterFilter")
            if upg_trend_blocked:
                blocked_filters.append("TrendFilter")

            if final_blocked:
                if ml_filter_bypass_signal:
                    for name in blocked_filters:
                        _mark_ml_filter_bypass(name, bypassed_filters)
                elif is_rescued:
                    if rescue_bypass_allowed:
                        for name in blocked_filters:
                            record_filter(name, kind="bypass")
                            bypassed_filters.append(name)
                    else:
                        for name in blocked_filters:
                            record_filter(name)
                        continue
                else:
                    rescue_reason = "FilterStack"
                    if blocked_filters:
                        rescue_reason = f"FilterStack:{'+'.join(blocked_filters)}"
                    if not try_rescue(rescue_reason):
                        for name in blocked_filters:
                            record_filter(name)
                        continue
                    for name in blocked_filters:
                        record_filter(name, kind="rescue")

            chop_blocked, chop_reason = _chop_check(signal["side"], trend_state="NEUTRAL")
            if chop_blocked and asia_calib_enabled and is_asia_session:
                if asia_chop_override(
                    chop_reason,
                    signal["side"],
                    asia_trend_bias_side,
                    asia_chop_cfg,
                ):
                    record_filter("ChopFilter", kind="bypass")
                    bypassed_filters.append("ChopFilter")
                    chop_blocked = False
            if chop_blocked:
                if chop_reason:
                    reason_lc = str(chop_reason).lower()
                    if "wait for breakout" in reason_lc or "range too tight" in reason_lc:
                        if not _mark_ml_filter_bypass("ChopFilter", bypassed_filters):
                            record_filter("ChopFilter")
                            continue
                if is_rescued:
                    if rescue_bypass_allowed:
                        record_filter("ChopFilter", kind="bypass")
                        bypassed_filters.append("ChopFilter")
                    else:
                        if not _mark_ml_filter_bypass("ChopFilter", bypassed_filters):
                            record_filter("ChopFilter")
                            continue
                else:
                    if not try_rescue("ChopFilter"):
                        if not _mark_ml_filter_bypass("ChopFilter", bypassed_filters):
                            record_filter("ChopFilter")
                            continue
                    record_filter("ChopFilter", kind="rescue")

            ext_blocked, _ = _extension_check(signal["side"])
            ext_soft_passed = False
            if ext_blocked and asia_soft_ext_enabled:
                soft_score = asia_soft_ext_base - asia_soft_ext_penalty
                if soft_score >= asia_soft_ext_threshold:
                    ext_blocked = False
                    ext_soft_passed = True
                    record_filter("ExtensionFilter", kind="bypass")
                    bypassed_filters.append("ExtensionFilter")
            if ext_blocked:
                if is_rescued:
                    if rescue_bypass_allowed:
                        record_filter("ExtensionFilter", kind="bypass")
                        bypassed_filters.append("ExtensionFilter")
                    else:
                        if not _mark_ml_filter_bypass("ExtensionFilter", bypassed_filters):
                            record_filter("ExtensionFilter")
                            continue
                else:
                    if not try_rescue("ExtensionFilter"):
                        if not _mark_ml_filter_bypass("ExtensionFilter", bypassed_filters):
                            record_filter("ExtensionFilter")
                            continue
                    record_filter("ExtensionFilter", kind="rescue")

            should_trade, vol_adj = _volatility_guard_check(
                signal.get("sl_dist", MIN_SL),
                signal.get("tp_dist", MIN_TP),
                _signal_base_size(signal, CONTRACTS),
            )
            if not should_trade:
                if not _mark_ml_filter_bypass("VolatilityGuardrail", bypassed_filters):
                    record_filter("VolatilityGuardrail")
                    continue

            _apply_vol_adj_to_signal(signal, vol_adj)
            if not _ml_vol_regime_check(signal, session_name, signal["vol_regime"]):
                if not _mark_ml_filter_bypass("MLVolRegimeGuard", bypassed_filters):
                    record_filter("MLVolRegimeGuard")
                    continue
            signal["entry_mode"] = "rescued" if is_rescued else "standard"
            if ml_filter_bypass_signal:
                signal["entry_mode"] = "ml_filter_bypass_test"

            if hostile_day_active and (
                signal.get("strategy") in ("DynamicEngine", "MLPhysics")
                or str(signal.get("strategy", "")).startswith("Continuation_")
            ):
                if not _mark_ml_filter_bypass("HostileDayGate", bypassed_filters):
                    record_filter("HostileDayGate")
                    continue
            if not ALLOW_DYNAMIC_ENGINE_SOLO and signal.get("strategy") == "DynamicEngine" and not is_rescued:
                record_filter("DynamicEngineSolo")
                continue

            if bypassed_filters:
                existing_bypassed = signal.get("bypassed_filters")
                if not isinstance(existing_bypassed, list):
                    existing_bypassed = []
                for name in bypassed_filters:
                    if name not in existing_bypassed:
                        existing_bypassed.append(name)
                signal["bypassed_filters"] = existing_bypassed
            handle_signal(signal, bar_index=i)
            signal_executed = True
            break

        flip_context = None
        if not signal_executed:
            for s_name in list(pending_loose_signals.keys()):
                pending = pending_loose_signals[s_name]
                pending["bar_count"] += 1
                if pending["bar_count"] < 1:
                    continue

                sig = pending["signal"]
                apply_multipliers(sig, strategy_hint=s_name)
                sig.setdefault("sl_dist", MIN_SL)
                sig.setdefault("tp_dist", MIN_TP)
                sig.setdefault("strategy", s_name)
                if enabled_filter_pre_candidate:
                    gate_ok, gate_filter, _, gate_profile = evaluate_pre_signal_gate(
                        cfg=CONFIG,
                        session_name=session_name,
                        strategy_label=str(sig.get("strategy", s_name) or s_name),
                        side=sig.get("side"),
                        asia_viable=asia_viable,
                        asia_reason=asia_viable_reason,
                        asia_trend_bias_side=asia_trend_bias_side,
                        is_choppy=bool(is_choppy),
                        chop_reason=chop_reason,
                        allowed_chop_side=allowed_chop_side,
                    )
                else:
                    gate_ok, gate_filter, gate_profile = True, None, {}
                sig["gate_profile"] = gate_profile
                if not gate_ok:
                    record_filter(gate_filter or "PreCandidateGate")
                    del pending_loose_signals[s_name]
                    continue
                rm_ok, _ = apply_regime_meta(sig, s_name)
                if not rm_ok and regime_manifold_mode == "enforce":
                    manifold_blocked += 1
                    record_filter("RegimeManifold")
                    del pending_loose_signals[s_name]
                    continue
                fixed_ok, fixed_details = _apply_fixed_sltp_check(sig)
                if not fixed_ok:
                    record_filter("FixedSLTP")
                    del pending_loose_signals[s_name]
                    continue
                if fixed_details:
                    _apply_fixed_details_to_signal(sig, fixed_details)
                    log_fixed_sltp(fixed_details, sig.get("strategy", s_name))
                if trend_day_tier > 0 and trend_day_dir:
                    if (trend_day_dir == "down" and sig["side"] == "LONG") or (
                        trend_day_dir == "up" and sig["side"] == "SHORT"
                    ):
                        record_filter(f"TrendDayTier{trend_day_tier}")
                        del pending_loose_signals[s_name]
                        continue
                    sig["trend_day_tier"] = trend_day_tier
                    sig["trend_day_dir"] = trend_day_dir

                if allowed_chop_side is not None and sig["side"] != allowed_chop_side:
                    record_filter("ChopRangeBias")
                    del pending_loose_signals[s_name]
                    continue

                sig_tp = sig.get("tp_dist", MIN_TP)
                is_feasible, _ = _target_feasibility_check(sig["side"], sig_tp)
                if not is_feasible:
                    if _asia_target_feasibility_override_check(sig["side"], sig_tp):
                        record_filter("TargetFeasibility", kind="bypass")
                        is_feasible = True
                    if not is_feasible:
                        record_filter("TargetFeasibility")
                        del pending_loose_signals[s_name]
                        continue

                rej_blocked, _ = _rejection_check(sig["side"])
                if rej_blocked:
                    record_filter("RejectionFilter")
                    del pending_loose_signals[s_name]
                    continue

                dir_blocked, _ = _directional_loss_check(sig["side"])
                if dir_blocked:
                    record_filter("DirectionalLossBlocker")
                    del pending_loose_signals[s_name]
                    continue

                impulse_blocked, _ = _impulse_check(sig["side"])
                if impulse_blocked:
                    record_filter("ImpulseFilter")
                    del pending_loose_signals[s_name]
                    continue

                tp_dist = sig.get("tp_dist", MIN_TP)
                effective_tp_dist = tp_dist
                if allowed_chop_side is not None and sig["side"] == allowed_chop_side:
                    effective_tp_dist = tp_dist * 0.5

                if enabled_filter_htf_fvg and htf_fvg_enabled_backtest and htf_fvg_filter is not None:
                    fvg_blocked, _ = htf_fvg_filter.check_signal_blocked(
                        sig["side"],
                        bar_close,
                        None,
                        None,
                        tp_dist=effective_tp_dist,
                        current_time=current_time,
                    )
                    if fvg_blocked:
                        record_filter("HTF_FVG")
                        del pending_loose_signals[s_name]
                        continue

                struct_blocked, _ = _structure_check(sig["side"])
                if struct_blocked:
                    record_filter("StructureBlocker")
                    del pending_loose_signals[s_name]
                    continue

                regime_blocked, _ = _regime_block_check(sig["side"])
                if regime_blocked:
                    record_filter("RegimeBlocker")
                    del pending_loose_signals[s_name]
                    continue

                penalty_source = None
                if is_asia_session and penalty_blocker_asia is not None:
                    penalty_source = penalty_blocker_asia
                elif penalty_blocker is not None:
                    penalty_source = penalty_blocker
                if penalty_source is not None:
                    penalty_blocked, _ = _penalty_check(penalty_source, sig["side"])
                    if penalty_blocked:
                        record_filter("PenaltyBoxBlocker")
                        del pending_loose_signals[s_name]
                        continue

                mem_blocked, _ = _memory_sr_check(sig["side"])
                if mem_blocked:
                    record_filter("MemorySRFilter")
                    del pending_loose_signals[s_name]
                    continue

                is_range_fade = allowed_chop_side is not None and sig["side"] == allowed_chop_side
                legacy_trend_blocked, legacy_trend_reason = _legacy_trend_check(sig["side"])
                upgraded_trend_blocked, upgraded_trend_reason = _trend_check(
                    sig["side"],
                    is_range_fade=is_range_fade,
                )

                if legacy_trend_blocked != upgraded_trend_blocked:
                    if enabled_filter_filter_arbitrator:
                        arb_result = filter_arbitrator.arbitrate(
                            df=history_df,
                            side=sig["side"],
                            legacy_blocked=legacy_trend_blocked,
                            legacy_reason=legacy_trend_reason or "",
                            upgraded_blocked=upgraded_trend_blocked,
                            upgraded_reason=upgraded_trend_reason or "",
                            current_price=bar_close,
                            tp_dist=sig.get("tp_dist"),
                            sl_dist=sig.get("sl_dist"),
                        )
                        trend_blocked = not arb_result.allow_trade
                        trend_reason = arb_result.reason
                    else:
                        trend_blocked = upgraded_trend_blocked
                        trend_reason = upgraded_trend_reason
                else:
                    trend_blocked = upgraded_trend_blocked
                    trend_reason = upgraded_trend_reason

                trend_state = trend_state_from_reason(trend_reason)
                chop_blocked, chop_reason = _chop_check(sig["side"], trend_state=trend_state)
                if chop_blocked and asia_calib_enabled and is_asia_session:
                    if asia_chop_override(
                        chop_reason,
                        sig["side"],
                        asia_trend_bias_side,
                        asia_chop_cfg,
                    ):
                        record_filter("ChopFilter", kind="bypass")
                        chop_blocked = False
                if chop_blocked:
                    record_filter("ChopFilter")
                    del pending_loose_signals[s_name]
                    continue

                ext_blocked, _ = _extension_check(sig["side"])
                if ext_blocked and asia_soft_ext_enabled:
                    soft_score = asia_soft_ext_base - asia_soft_ext_penalty
                    if soft_score >= asia_soft_ext_threshold:
                        ext_blocked = False
                        record_filter("ExtensionFilter", kind="bypass")
                if ext_blocked:
                    record_filter("ExtensionFilter")
                    del pending_loose_signals[s_name]
                    continue

                if trend_blocked:
                    record_filter("TrendFilter")
                    del pending_loose_signals[s_name]
                    continue

                should_trade, vol_adj = _volatility_guard_check(
                    sig.get("sl_dist", MIN_SL),
                    sig.get("tp_dist", MIN_TP),
                    _signal_base_size(sig, CONTRACTS),
                )
                if not should_trade:
                    record_filter("VolatilityGuardrail")
                    del pending_loose_signals[s_name]
                    continue

                _apply_vol_adj_to_signal(sig, vol_adj)
                if not _ml_vol_regime_check(sig, session_name, sig["vol_regime"]):
                    record_filter("MLVolRegimeGuard")
                    del pending_loose_signals[s_name]
                    continue
                sig["entry_mode"] = "loose"

                handle_signal(sig, bar_index=i)
                signal_executed = True
                del pending_loose_signals[s_name]
                break

        if not signal_executed:
            for strat in loose_strategies:
                s_name = strat.__class__.__name__
                try:
                    signal = strat.on_bar(history_df)
                except Exception:
                    signal = None
                if not signal:
                    continue
                apply_multipliers(signal, strategy_hint=s_name)
                signal.setdefault("strategy", s_name)
                if enabled_filter_pre_candidate:
                    gate_ok, gate_filter, _, gate_profile = evaluate_pre_signal_gate(
                        cfg=CONFIG,
                        session_name=session_name,
                        strategy_label=str(signal.get("strategy", s_name) or s_name),
                        side=signal.get("side"),
                        asia_viable=asia_viable,
                        asia_reason=asia_viable_reason,
                        asia_trend_bias_side=asia_trend_bias_side,
                        is_choppy=bool(is_choppy),
                        chop_reason=chop_reason,
                        allowed_chop_side=allowed_chop_side,
                    )
                else:
                    gate_ok, gate_filter, gate_profile = True, None, {}
                signal["gate_profile"] = gate_profile
                if not gate_ok:
                    record_filter(gate_filter or "PreCandidateGate")
                    continue
                rm_ok, _ = apply_regime_meta(signal, s_name)
                if not rm_ok and regime_manifold_mode == "enforce":
                    manifold_blocked += 1
                    record_filter("RegimeManifold")
                    continue
                if trend_day_tier > 0 and trend_day_dir:
                    if (trend_day_dir == "down" and signal["side"] == "LONG") or (
                        trend_day_dir == "up" and signal["side"] == "SHORT"
                    ):
                        record_filter(f"TrendDayTier{trend_day_tier}")
                        continue
                    signal["trend_day_tier"] = trend_day_tier
                    signal["trend_day_dir"] = trend_day_dir
                if allowed_chop_side is not None and signal["side"] != allowed_chop_side:
                    continue
                pending_loose_signals[s_name] = {"signal": signal, "bar_count": 0}

        if in_test_range:
            emit_progress(current_time, bar_close)

    if active_trade is not None and last_time is not None and last_close is not None:
        if cancelled:
            close_trade(float(last_close), last_time, "cancelled", bar_index=len(full_df) - 1)
        elif not test_df.empty:
            final_time = test_df.index[-1]
            final_close = float(test_df.iloc[-1]["close"])
            close_trade(final_close, final_time, "end_of_range", bar_index=len(full_df) - 1)

    if strategy_executor is not None:
        try:
            strategy_executor.shutdown(wait=True)
        except Exception:
            pass

    ny_gate_total_blocked = (
        ny_gate_balance_blocked + ny_gate_acceptance_blocked + ny_gate_liquidity_blocked
    )
    ny_gate_block_rate = (
        ny_gate_total_blocked / ny_gate_candidates if ny_gate_candidates else 0.0
    )
    ny_gate_summary = {
        "candidates": ny_gate_candidates,
        "blocked_total": ny_gate_total_blocked,
        "balance_blocked": ny_gate_balance_blocked,
        "acceptance_blocked": ny_gate_acceptance_blocked,
        "liquidity_blocked": ny_gate_liquidity_blocked,
        "blocked_by_session": dict(ny_gate_blocked_by_session),
        "blocked_by_strategy": dict(ny_gate_blocked_by_strategy),
        "block_rate": ny_gate_block_rate,
    }
    if ny_gate_candidates:
        print(
            "NY continuation gate summary: candidates="
            f"{ny_gate_candidates} blocked={ny_gate_total_blocked} "
            f"(balance={ny_gate_balance_blocked}, acceptance={ny_gate_acceptance_blocked}, "
            f"liquidity={ny_gate_liquidity_blocked}) rate={ny_gate_block_rate:.2%}"
        )

    winrate = (wins / trades * 100.0) if trades else 0.0
    tracker.finalize_streaks()
    log_mfe_tp_anomalies(tracker.trades)
    if progress_cb is not None and last_time is not None and last_close is not None:
        emit_progress(last_time, last_close, force=True, done=True)
    report_text = tracker.build_report(max_rows=50)
    def _calc_trade_metrics(trades: list) -> dict:
        if not trades:
            return {
                "trades": 0,
                "winrate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
            }
        pnl = [float(t.get("pnl_points") or 0.0) for t in trades]
        wins = sum(1 for p in pnl if p >= 0)
        total = float(sum(pnl))
        avg = total / float(len(pnl)) if pnl else 0.0
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnl:
            cum += p
            if cum > peak:
                peak = cum
            dd = cum - peak
            if dd < max_dd:
                max_dd = dd
        return {
            "trades": len(pnl),
            "winrate": wins / float(len(pnl)) if pnl else 0.0,
            "avg_pnl": avg,
            "total_pnl": total,
            "max_drawdown": max_dd,
        }

    def _de3_counterfactual(trades: list, veto_summary: Optional[dict]) -> Optional[dict]:
        if not trades:
            return None
        de3_trades = [t for t in trades if str(t.get("strategy", "")).lower() == "dynamicengine3"]
        if not de3_trades:
            return None
        if not any("de3_veto_p_loss" in t for t in de3_trades):
            return None
        kept = [t for t in de3_trades if not t.get("de3_veto_would_block")]
        blocked = [t for t in de3_trades if t.get("de3_veto_would_block")]
        base_metrics = _calc_trade_metrics(de3_trades)
        kept_metrics = _calc_trade_metrics(kept)
        blocked_metrics = _calc_trade_metrics(blocked)
        removed_pct = (len(blocked) / len(de3_trades)) if de3_trades else 0.0
        threshold = None
        for t in de3_trades:
            if "de3_veto_threshold" in t:
                try:
                    threshold = float(t.get("de3_veto_threshold"))
                except Exception:
                    threshold = None
                break
        return {
            "mode": (veto_summary or {}).get("mode") if veto_summary else None,
            "threshold": threshold,
            "removed_pct": removed_pct,
            "baseline": base_metrics,
            "kept": kept_metrics,
            "blocked": blocked_metrics,
        }

    def _de3_meta_counterfactual(trades: list, meta_summary: Optional[dict]) -> Optional[dict]:
        if not trades:
            return None
        de3_trades = [t for t in trades if str(t.get("strategy", "")).lower() == "dynamicengine3"]
        if not de3_trades:
            return None
        if not any("de3_meta_would_block" in t for t in de3_trades):
            return None
        kept = [t for t in de3_trades if not t.get("de3_meta_would_block")]
        blocked = [t for t in de3_trades if t.get("de3_meta_would_block")]
        base_metrics = _calc_trade_metrics(de3_trades)
        kept_metrics = _calc_trade_metrics(kept)
        blocked_metrics = _calc_trade_metrics(blocked)
        removed_pct = (len(blocked) / len(de3_trades)) if de3_trades else 0.0
        return {
            "mode": (meta_summary or {}).get("mode") if meta_summary else None,
            "removed_pct": removed_pct,
            "baseline": base_metrics,
            "kept": kept_metrics,
            "blocked": blocked_metrics,
        }

    flip_payload = {}
    if flip_tracker is not None and flip_tracker.enabled:
        flip_payload = flip_tracker.finalize()
    de3_veto_summary = None
    if dynamic_engine3_strat is not None:
        try:
            de3_veto_summary = dynamic_engine3_strat.get_veto_summary()
        except Exception:
            de3_veto_summary = None
    de3_veto_counterfactual = _de3_counterfactual(tracker.trades, de3_veto_summary)
    de3_meta_summary = {
        "enabled": de3_meta_enabled,
        "mode": de3_meta_mode,
        "checked": int(de3_meta_checked),
        "would_block": int(de3_meta_would_block),
        "blocked": int(de3_meta_blocked),
        "shadow_would_block": int(de3_meta_shadow),
        "reasons": dict(de3_meta_reasons),
    }
    de3_manifold_adaptation_summary = {
        "enabled": de3_manifold_adapt_enabled,
        "mode": de3_manifold_adapt_mode,
        "checked": int(de3_manifold_adapt_checked),
        "would_block": int(de3_manifold_adapt_would_block),
        "blocked": int(de3_manifold_adapt_blocked),
        "shadow_would_block": int(de3_manifold_adapt_shadow),
        "blocked_regimes": sorted(de3_manifold_blocked_regimes),
        "block_no_trade": bool(de3_manifold_block_no_trade),
        "require_allow_style": bool(de3_manifold_require_allow_style),
        "reasons": dict(de3_manifold_adapt_reasons),
    }
    regime_manifold_summary = {
        "enabled": regime_manifold_enabled,
        "mode": regime_manifold_mode,
        "checked": int(manifold_checked),
        "would_block": int(manifold_would_block),
        "blocked": int(manifold_blocked),
        "shadow_would_block": int(manifold_shadow_would_block),
        "reasons": dict(manifold_reasons),
    }
    de3_meta_counterfactual = _de3_meta_counterfactual(tracker.trades, de3_meta_summary)
    ml_diagnostics_summary = tracker.summarize_ml_diagnostics(max_rows=25)
    market_conditions_summary = (
        summarize_market_conditions(tracker.trades, top_n=50)
        if market_summary_enabled
        else {}
    )
    if int(ml_diagnostics_summary.get("total_evals", 0) or 0) > 0:
        top_block = "none"
        blocked_reason_counts = ml_diagnostics_summary.get("blocked_reason_counts", {}) or {}
        if blocked_reason_counts:
            top_block = sorted(
                blocked_reason_counts.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[0][0]
        print(
            "ML diagnostics summary: "
            f"evals={int(ml_diagnostics_summary.get('total_evals', 0))} "
            f"signals={int(ml_diagnostics_summary.get('signal_count', 0))} "
            f"signal_rate={float(ml_diagnostics_summary.get('signal_rate', 0.0)):.1%} "
            f"top_block={top_block}"
        )
    if live_report_enabled and live_report_path is not None:
        try:
            final_time_for_live = last_time if last_time is not None else end_time
            if last_close is not None:
                final_price_for_live = float(last_close)
            elif not test_df.empty:
                final_price_for_live = float(test_df.iloc[-1]["close"])
            else:
                final_price_for_live = 0.0
            write_live_report(final_time_for_live, final_price_for_live, done=True, force=True)
        except Exception:
            pass
    de3_decisions_export_meta = {
        "enabled": bool(de3_decision_export_enabled),
        "top_k": int(de3_decision_export_top_k),
        "rows": 0,
        "path": None,
        "trade_attribution_rows": 0,
        "trade_attribution_path": None,
        "summary_path": None,
        "family_rows": 0,
        "family_path": None,
        "family_summary_path": None,
        "activation_audit_path": None,
        "runtime_path_counters_path": None,
        "bundle_usage_audit_path": None,
        "config_usage_audit_path": None,
        "score_path_audit_path": None,
        "choice_path_audit_path": None,
        "refined_vs_raw_audit_path": None,
        "family_score_trace_path": None,
        "family_competition_health_path": None,
        "member_resolution_audit_path": None,
        "family_score_component_summary_path": None,
        "family_score_delta_ladder_path": None,
        "diff_readiness_path": None,
        "inert_change_summary_path": None,
    }
    if de3_decision_export_enabled and de3_decisions_out_path is not None:
        try:
            trade_rows: list[dict] = []
            for trade in tracker.trades:
                if str(trade.get("strategy", "")).lower() != "dynamicengine3":
                    continue
                decision_id = str(trade.get("de3_decision_id", "") or "").strip()
                if not decision_id:
                    continue
                entry_time_raw = trade.get("entry_time")
                exit_time_raw = trade.get("exit_time")
                trade_row = {
                    "decision_id": decision_id,
                    "trade_id": int(_coerce_int(trade.get("trade_id", 0), 0)),
                    "strategy": str(trade.get("strategy", "") or ""),
                    "sub_strategy": str(trade.get("sub_strategy", "") or ""),
                    "side": str(trade.get("side", "") or ""),
                    "entry_time": entry_time_raw.isoformat() if isinstance(entry_time_raw, dt.datetime) else entry_time_raw,
                    "exit_time": exit_time_raw.isoformat() if isinstance(exit_time_raw, dt.datetime) else exit_time_raw,
                    "realized_exit_type": str(trade.get("exit_reason", "") or ""),
                    "realized_pnl": float(_coerce_float(trade.get("pnl_net", 0.0), 0.0)),
                    "mfe": float(_coerce_float(trade.get("mfe_points", 0.0), 0.0)),
                    "mae": float(_coerce_float(trade.get("mae_points", 0.0), 0.0)),
                    "tp_dist": float(_coerce_float(trade.get("tp_dist", 0.0), 0.0)),
                    "sl_dist": float(_coerce_float(trade.get("sl_dist", 0.0), 0.0)),
                }
                trade_rows.append(trade_row)

            if de3_decisions_writer is not None:
                _flush_de3_decision_buffer()
                if de3_decisions_file_handle is not None:
                    try:
                        de3_decisions_file_handle.flush()
                    except Exception:
                        pass
                    try:
                        de3_decisions_file_handle.close()
                    except Exception:
                        pass
                    de3_decisions_file_handle = None
                decisions_row_count = int(de3_decision_rows_written)
                try:
                    decisions_df = pd.read_csv(
                        de3_decisions_out_path,
                        low_memory=False,
                    )
                except Exception:
                    decisions_df = pd.DataFrame(columns=de3_decision_fieldnames)
            else:
                decisions_df = pd.DataFrame(de3_decision_rows)
                if decisions_df.empty:
                    decisions_df = pd.DataFrame(columns=de3_decision_fieldnames)
                decisions_df.to_csv(de3_decisions_out_path, index=False)
                decisions_row_count = int(len(decisions_df))

            trade_out_path = de3_decisions_out_path.with_name(
                f"{de3_decisions_out_path.stem}_trade_attribution.csv"
            )
            trade_df = pd.DataFrame(trade_rows)
            trade_df.to_csv(trade_out_path, index=False)
            summary_out_path = de3_decisions_out_path.with_name(
                f"{de3_decisions_out_path.stem}_summary.json"
            )
            required_ctx_fields = [
                "ctx_volatility_regime",
                "ctx_compression_expansion_regime",
                "ctx_confidence_band",
            ]
            decisions_cols = set(str(col) for col in decisions_df.columns)
            ctx_present = [field for field in required_ctx_fields if field in decisions_cols]
            ctx_missing = [field for field in required_ctx_fields if field not in decisions_cols]
            summary_payload = {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "decision_rows": int(decisions_row_count),
                "trade_attribution_rows": int(len(trade_rows)),
                "top_k": int(de3_decision_export_top_k),
                "de3_runtime_version": str(de3_runtime_db_version),
                "de3_runtime_family_mode_enabled": bool(de3_runtime_family_mode_enabled),
                "de3_runtime_family_artifact": de3_runtime_family_artifact,
                "de3_runtime_context_profiles_loaded": bool(de3_runtime_context_profiles_loaded),
                "de3_runtime_enriched_export_required": bool(de3_runtime_enriched_export_required),
                "de3_runtime_context_profile_build": de3_runtime_context_profile_build,
                "de3_runtime_active_context_dimensions": list(de3_runtime_active_context_dimensions),
                "de3_runtime_context_trust": dict(de3_runtime_context_trust),
                "de3_runtime_local_bracket_freeze": dict(de3_runtime_local_bracket_freeze),
                "de3_runtime_state_loaded": bool(de3_runtime_state_loaded),
                "de3_runtime_state_build": dict(de3_runtime_state_build),
                "de3_runtime_export_raw_context_fields": bool(de3_runtime_export_raw_context_fields),
                "decision_export_context_fields_present": ctx_present,
                "decision_export_context_fields_missing": ctx_missing,
                "used_enriched_decision_export": bool(len(ctx_missing) == 0),
                "source_report_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "decision_csv": str(de3_decisions_out_path),
                "trade_attribution_csv": str(trade_out_path),
            }
            summary_out_path.write_text(
                json.dumps(summary_payload, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

            def _to_bool_col(series: pd.Series) -> pd.Series:
                return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})

            family_decisions_out_path = de3_decisions_out_path.with_name("de3_family_decisions.csv")
            family_summary_out_path = de3_decisions_out_path.with_name("de3_family_summary.json")
            family_decisions_df = pd.DataFrame()
            if not decisions_df.empty:
                version_col = decisions_df["de3_version"].fillna("").astype(str).str.lower() if "de3_version" in decisions_df.columns else pd.Series("", index=decisions_df.index)
                family_mode_col = _to_bool_col(decisions_df["family_mode"]) if "family_mode" in decisions_df.columns else pd.Series(False, index=decisions_df.index)
                family_decisions_df = decisions_df[version_col.str.startswith("v3") & family_mode_col].copy()
            if not family_decisions_df.empty:
                family_decisions_df.to_csv(family_decisions_out_path, index=False)

                chosen_mask = _to_bool_col(family_decisions_df["chosen"]) if "chosen" in family_decisions_df.columns else pd.Series(False, index=family_decisions_df.index)
                abstained_mask = _to_bool_col(family_decisions_df["abstained"]) if "abstained" in family_decisions_df.columns else pd.Series(False, index=family_decisions_df.index)
                chosen_df = family_decisions_df[chosen_mask & (~abstained_mask)].copy()
                if "chosen_family_id" in chosen_df.columns and "family_id" in chosen_df.columns:
                    chosen_df["family_id_effective"] = chosen_df["chosen_family_id"].fillna("").astype(str)
                    blank_mask = chosen_df["family_id_effective"].str.strip() == ""
                    chosen_df.loc[blank_mask, "family_id_effective"] = chosen_df.loc[blank_mask, "family_id"].fillna("").astype(str)
                else:
                    if "family_id" in chosen_df.columns:
                        chosen_df["family_id_effective"] = chosen_df["family_id"].fillna("").astype(str)
                    else:
                        chosen_df["family_id_effective"] = ""

                trade_agg = pd.DataFrame(columns=["decision_id", "realized_pnl", "avg_mae", "avg_mfe", "stop_hit", "stop_gap_hit"])
                if not trade_df.empty and "decision_id" in trade_df.columns:
                    trade_work = trade_df.copy()
                    trade_work["realized_exit_type"] = trade_work.get("realized_exit_type", "").fillna("").astype(str).str.lower()
                    trade_work["stop_hit"] = trade_work["realized_exit_type"].str.contains("stop", regex=False)
                    trade_work["stop_gap_hit"] = trade_work["realized_exit_type"].str.contains("stop", regex=False) & trade_work["realized_exit_type"].str.contains("gap", regex=False)
                    trade_agg = trade_work.groupby("decision_id", as_index=False).agg(
                        realized_pnl=("realized_pnl", "sum"),
                        avg_mae=("mae", "mean"),
                        avg_mfe=("mfe", "mean"),
                        stop_hit=("stop_hit", "max"),
                        stop_gap_hit=("stop_gap_hit", "max"),
                    )

                chosen_with_trade = chosen_df.merge(trade_agg, on="decision_id", how="left")
                trade_eval = chosen_with_trade[chosen_with_trade["realized_pnl"].notna()].copy()

                family_chosen_freq = {}
                if not chosen_df.empty:
                    family_chosen_freq = (
                        chosen_df["family_id_effective"].value_counts(dropna=False).to_dict()
                    )

                family_perf = {}
                if not trade_eval.empty:
                    for family_id, grp in trade_eval.groupby("family_id_effective"):
                        pnl_vals = grp["realized_pnl"].fillna(0.0)
                        gross_profit = float(pnl_vals[pnl_vals > 0].sum())
                        gross_loss_abs = float(abs(pnl_vals[pnl_vals < 0].sum()))
                        pf = (gross_profit / gross_loss_abs) if gross_loss_abs > 1e-9 else (999.0 if gross_profit > 0 else 0.0)
                        family_perf[str(family_id)] = {
                            "trades": int(len(grp)),
                            "pnl": float(pnl_vals.sum()),
                            "win_rate": float((pnl_vals > 0).mean()),
                            "profit_factor": float(pf),
                            "stop_rate": float(grp["stop_hit"].fillna(False).astype(bool).mean()),
                            "stop_gap_rate": float(grp["stop_gap_hit"].fillna(False).astype(bool).mean()),
                            "avg_mae": float(grp["avg_mae"].fillna(0.0).mean()),
                            "avg_mfe": float(grp["avg_mfe"].fillna(0.0).mean()),
                        }

                context_perf = {}
                context_cols = [
                    "ctx_volatility_regime",
                    "ctx_compression_expansion_regime",
                    "ctx_confidence_band",
                ]
                if not trade_eval.empty and all(col in trade_eval.columns for col in context_cols):
                    tmp = trade_eval.copy()
                    tmp["context_bucket_key"] = (
                        "vol=" + tmp["ctx_volatility_regime"].fillna("mid").astype(str)
                        + "|comp=" + tmp["ctx_compression_expansion_regime"].fillna("neutral").astype(str)
                        + "|conf=" + tmp["ctx_confidence_band"].fillna("mid").astype(str)
                    )
                    for (family_id, bucket_key), grp in tmp.groupby(["family_id_effective", "context_bucket_key"]):
                        pnl_vals = grp["realized_pnl"].fillna(0.0)
                        gross_profit = float(pnl_vals[pnl_vals > 0].sum())
                        gross_loss_abs = float(abs(pnl_vals[pnl_vals < 0].sum()))
                        pf = (gross_profit / gross_loss_abs) if gross_loss_abs > 1e-9 else (999.0 if gross_profit > 0 else 0.0)
                        fam_key = str(family_id)
                        context_perf.setdefault(fam_key, {})
                        context_perf[fam_key][str(bucket_key)] = {
                            "trades": int(len(grp)),
                            "pnl": float(pnl_vals.sum()),
                            "win_rate": float((pnl_vals > 0).mean()),
                            "profit_factor": float(pf),
                            "stop_rate": float(grp["stop_hit"].fillna(False).astype(bool).mean()),
                            "stop_gap_rate": float(grp["stop_gap_hit"].fillna(False).astype(bool).mean()),
                        }

                local_member_distribution = {}
                if not chosen_df.empty and "sub_strategy" in chosen_df.columns:
                    for family_id, grp in chosen_df.groupby("family_id_effective"):
                        local_member_distribution[str(family_id)] = (
                            grp["sub_strategy"].fillna("").astype(str).value_counts().to_dict()
                        )

                local_adaptation_mode_distribution = {}
                local_adaptation_rate_by_family = {}
                if not chosen_df.empty and "local_bracket_adaptation_mode" in chosen_df.columns:
                    local_adaptation_mode_distribution = (
                        chosen_df["local_bracket_adaptation_mode"].fillna("").astype(str).value_counts().to_dict()
                    )
                    for family_id, grp in chosen_df.groupby("family_id_effective"):
                        enabled = (
                            _to_bool_col(grp["local_bracket_adaptation_enabled"])
                            if "local_bracket_adaptation_enabled" in grp.columns
                            else pd.Series(False, index=grp.index)
                        )
                        overrides = (
                            _to_bool_col(grp["local_bracket_override_applied"])
                            if "local_bracket_override_applied" in grp.columns
                            else pd.Series(False, index=grp.index)
                        )
                        local_adaptation_rate_by_family[str(family_id)] = {
                            "adaptation_enabled_rate": float(enabled.mean()) if not enabled.empty else 0.0,
                            "override_applied_rate": float(overrides.mean()) if not overrides.empty else 0.0,
                        }

                evidence_support_tier_counts = {}
                if not chosen_df.empty and "family_evidence_support_tier" in chosen_df.columns:
                    evidence_support_tier_counts = (
                        chosen_df["family_evidence_support_tier"].fillna("none").astype(str).str.lower().value_counts().to_dict()
                    )
                competition_status_counts = {}
                if not chosen_df.empty and "family_competition_status" in chosen_df.columns:
                    competition_status_counts = (
                        chosen_df["family_competition_status"].fillna("competitive").astype(str).str.lower().value_counts().to_dict()
                    )

                canonical_usage_df = pd.DataFrame(index=chosen_df.index)
                if not chosen_df.empty:
                    canonical_usage_df = pd.DataFrame(
                        [canonical_context_usage_snapshot(rec) for rec in chosen_df.to_dict(orient="records")],
                        index=chosen_df.index,
                    )
                if canonical_usage_df.empty:
                    support_tier_series = pd.Series("low", index=chosen_df.index, dtype="object")
                    trusted_series = pd.Series(False, index=chosen_df.index, dtype="bool")
                    fallback_series = pd.Series(True, index=chosen_df.index, dtype="bool")
                    local_mode_series = pd.Series("none", index=chosen_df.index, dtype="object")
                    local_override_series = pd.Series(False, index=chosen_df.index, dtype="bool")
                else:
                    support_tier_series = canonical_usage_df["support_tier"].fillna("low").astype(str).str.lower()
                    trusted_series = canonical_usage_df["trusted_context_used"].fillna(False).astype(bool)
                    fallback_series = canonical_usage_df["fallback_to_priors"].fillna(True).astype(bool)
                    local_mode_series = canonical_usage_df["local_bracket_mode"].fillna("none").astype(str)
                    local_override_series = canonical_usage_df["local_bracket_override_applied"].fillna(False).astype(bool)
                profile_used_series = (
                    _to_bool_col(chosen_df["family_profile_used"])
                    if "family_profile_used" in chosen_df.columns
                    else pd.Series(False, index=chosen_df.index)
                )

                family_runtime_metrics = {}
                family_decision_groups = (
                    {str(fid): grp.copy() for fid, grp in family_decisions_df.groupby("family_id_effective")}
                    if not family_decisions_df.empty and "family_id_effective" in family_decisions_df.columns
                    else {}
                )
                for family_id, grp in chosen_df.groupby("family_id_effective"):
                    grp_index = grp.index
                    grp_usage = canonical_usage_df.loc[grp_index] if not canonical_usage_df.empty else pd.DataFrame(index=grp_index)
                    grp_all = family_decision_groups.get(str(family_id), grp)
                    tier_counts = {"strong": 0, "mid": 0, "low": 0}
                    if not grp_usage.empty and "support_tier" in grp_usage.columns:
                        raw_counts = grp_usage["support_tier"].fillna("low").astype(str).str.lower().value_counts().to_dict()
                        tier_counts["strong"] = int(raw_counts.get("strong", 0))
                        tier_counts["mid"] = int(raw_counts.get("mid", 0))
                        tier_counts["low"] = int(
                            sum(int(v) for k, v in raw_counts.items() if str(k) not in {"strong", "mid"})
                        )
                    grp_trade = trade_eval[trade_eval["family_id_effective"] == family_id].copy()
                    prior_series = (
                        _to_bool_col(grp["family_prior_eligible"])
                        if "family_prior_eligible" in grp.columns
                        else pd.Series(False, index=grp.index)
                    )
                    comp_series = (
                        _to_bool_col(grp["family_competition_eligible"])
                        if "family_competition_eligible" in grp.columns
                        else pd.Series(False, index=grp.index)
                    )
                    bootstrap_series = (
                        _to_bool_col(grp["family_bootstrap_competition_included"])
                        if "family_bootstrap_competition_included" in grp.columns
                        else (
                            _to_bool_col(grp["family_bootstrap_included"])
                            if "family_bootstrap_included" in grp.columns
                            else pd.Series(False, index=grp.index)
                        )
                    )
                    evidence_tier = (
                        grp["family_evidence_support_tier"].fillna("none").astype(str).str.lower().mode().iloc[0]
                        if "family_evidence_support_tier" in grp.columns and not grp.empty
                        else (
                            grp_usage["evidence_support_tier"].fillna("none").astype(str).str.lower().mode().iloc[0]
                            if ("evidence_support_tier" in grp_usage.columns and not grp_usage.empty)
                            else "none"
                        )
                    )
                    competition_status = (
                        grp["family_competition_status"].fillna("competitive").astype(str).str.lower().mode().iloc[0]
                        if "family_competition_status" in grp.columns and not grp.empty
                        else "competitive"
                    )
                    usability_adjustment_series = (
                        pd.to_numeric(grp["family_usability_adjustment"], errors="coerce")
                        if "family_usability_adjustment" in grp.columns
                        else pd.Series(0.0, index=grp.index)
                    )
                    suppression_reason = (
                        grp["family_suppression_reason"].fillna("").astype(str).mode().iloc[0]
                        if "family_suppression_reason" in grp.columns and not grp.empty
                        else ""
                    )
                    recent_share_series = (
                        pd.to_numeric(grp_all["recent_chosen_share"], errors="coerce")
                        if "recent_chosen_share" in grp_all.columns
                        else pd.Series(0.0, index=grp.index)
                    )
                    exploration_applied_series = (
                        _to_bool_col(grp_all["exploration_bonus_applied"])
                        if "exploration_bonus_applied" in grp_all.columns
                        else pd.Series(False, index=grp.index)
                    )
                    dominance_applied_series = (
                        _to_bool_col(grp_all["dominance_penalty_applied"])
                        if "dominance_penalty_applied" in grp_all.columns
                        else pd.Series(False, index=grp.index)
                    )
                    margin_qualified_series = (
                        _to_bool_col(grp_all["competition_margin_qualified"])
                        if "competition_margin_qualified" in grp_all.columns
                        else pd.Series(False, index=grp.index)
                    )
                    context_capped_series = (
                        _to_bool_col(grp_all["context_advantage_capped"])
                        if "context_advantage_capped" in grp_all.columns
                        else pd.Series(False, index=grp.index)
                    )
                    family_runtime_metrics[str(family_id)] = {
                        "prior_eligible": bool(prior_series.any()) if not prior_series.empty else False,
                        "evidence_support_tier": str(evidence_tier),
                        "competition_status": str(competition_status),
                        "competition_eligible": bool(comp_series.any()) if not comp_series.empty else False,
                        "bootstrap_competition_included": bool(bootstrap_series.any()) if not bootstrap_series.empty else False,
                        "chosen_count": int(len(grp)),
                        "executed_trade_count": int(len(grp_trade)),
                        "fallback_to_prior_rate": float(
                            grp_usage["fallback_to_priors"].fillna(True).astype(bool).mean()
                        ) if ("fallback_to_priors" in grp_usage.columns and not grp_usage.empty) else 1.0,
                        "context_supported_decision_rate": float(
                            grp_usage["support_tier"].fillna("low").astype(str).str.lower().isin(["strong", "mid"]).mean()
                        ) if ("support_tier" in grp_usage.columns and not grp_usage.empty) else 0.0,
                        "local_bracket_override_rate": float(
                            grp_usage["local_bracket_override_applied"].fillna(False).astype(bool).mean()
                        ) if ("local_bracket_override_applied" in grp_usage.columns and not grp_usage.empty) else 0.0,
                        "local_bracket_mode_counts": (
                            grp_usage["local_bracket_mode"].fillna("none").astype(str).value_counts().to_dict()
                            if ("local_bracket_mode" in grp_usage.columns and not grp_usage.empty)
                            else {}
                        ),
                        "context_support_tier_counts": dict(tier_counts),
                        "bootstrap_included_rate": float(bootstrap_series.mean()) if not bootstrap_series.empty else 0.0,
                        "usability_adjustment": float(usability_adjustment_series.fillna(0.0).mean()) if not usability_adjustment_series.empty else 0.0,
                        "suppression_reason": str(suppression_reason),
                        "remained_competitive_despite_low_support": bool(str(evidence_tier) in {"none", "low"} and (bool(comp_series.any()) if not comp_series.empty else False)),
                        "recent_chosen_share": float(recent_share_series.fillna(0.0).mean()) if not recent_share_series.empty else 0.0,
                        "exploration_bonus_applied_rate": float(exploration_applied_series.mean()) if not exploration_applied_series.empty else 0.0,
                        "dominance_penalty_applied_rate": float(dominance_applied_series.mean()) if not dominance_applied_series.empty else 0.0,
                        "competition_margin_qualified_rate": float(margin_qualified_series.mean()) if not margin_qualified_series.empty else 0.0,
                        "context_advantage_capped_rate": float(context_capped_series.mean()) if not context_capped_series.empty else 0.0,
                    }

                def _num(value, default=0.0):
                    try:
                        out = float(value)
                    except Exception:
                        return float(default)
                    return float(out) if np.isfinite(out) else float(default)

                runtime_family_states = (
                    (de3_runtime_state_build or {}).get("family_runtime_states", {})
                    if isinstance(de3_runtime_state_build, dict)
                    else {}
                )
                if isinstance(runtime_family_states, dict):
                    for family_id, state in runtime_family_states.items():
                        if not isinstance(state, dict):
                            continue
                        fam_key = str(family_id)
                        if fam_key in family_runtime_metrics:
                            continue
                        state_metrics = state.get("metrics", {}) if isinstance(state.get("metrics"), dict) else {}
                        family_runtime_metrics[fam_key] = {
                            "prior_eligible": bool(state.get("prior_eligible", False)),
                            "evidence_support_tier": str(state.get("evidence_support_tier", "none")),
                            "competition_status": str(state.get("competition_status", "suppressed")),
                            "competition_eligible": bool(state.get("competition_eligible", False)),
                            "bootstrap_competition_included": bool(state.get("bootstrap_competition_included", False)),
                            "chosen_count": int(_num(state_metrics.get("chosen_count", 0.0), 0.0)),
                            "executed_trade_count": int(_num(state_metrics.get("executed_trade_count", 0.0), 0.0)),
                            "fallback_to_prior_rate": float(_num(state_metrics.get("fallback_to_prior_rate", 1.0), 1.0)),
                            "context_supported_decision_rate": float(_num(state_metrics.get("context_supported_decision_rate", 0.0), 0.0)),
                            "local_bracket_override_rate": float(_num(state_metrics.get("local_bracket_override_rate", 0.0), 0.0)),
                            "local_bracket_mode_counts": dict(state_metrics.get("local_bracket_mode_counts", {}) if isinstance(state_metrics.get("local_bracket_mode_counts"), dict) else {}),
                            "context_support_tier_counts": {
                                "strong": int(_num(state_metrics.get("strong_context_support_count", 0), 0)),
                                "mid": int(_num(state_metrics.get("mid_context_support_count", 0), 0)),
                                "low": int(_num(state_metrics.get("low_context_support_count", 0), 0)),
                            },
                            "bootstrap_included_rate": 1.0 if bool(state.get("bootstrap_competition_included", False)) else 0.0,
                            "usability_adjustment": float(_num(state.get("usability_adjustment", 0.0), 0.0)),
                            "suppression_reason": str(state.get("suppression_reason", "") or ""),
                            "remained_competitive_despite_low_support": bool(
                                str(state.get("evidence_support_tier", "none")).lower() in {"none", "low"}
                                and bool(state.get("competition_eligible", False))
                            ),
                            "recent_chosen_share": 0.0,
                            "exploration_bonus_applied_rate": 0.0,
                            "dominance_penalty_applied_rate": 0.0,
                            "competition_margin_qualified_rate": 0.0,
                            "context_advantage_capped_rate": 0.0,
                        }

                family_metric_values = list(family_runtime_metrics.values())
                family_evidence_support_tier_counts = {
                    "none": int(sum(1 for m in family_metric_values if str(m.get("evidence_support_tier", "")) == "none")),
                    "low": int(sum(1 for m in family_metric_values if str(m.get("evidence_support_tier", "")) == "low")),
                    "mid": int(sum(1 for m in family_metric_values if str(m.get("evidence_support_tier", "")) == "mid")),
                    "strong": int(sum(1 for m in family_metric_values if str(m.get("evidence_support_tier", "")) == "strong")),
                }
                family_competition_status_counts = {
                    "competitive": int(sum(1 for m in family_metric_values if str(m.get("competition_status", "")) == "competitive")),
                    "competitive_bootstrap": int(sum(1 for m in family_metric_values if str(m.get("competition_status", "")) == "competitive_bootstrap")),
                    "fallback_only": int(sum(1 for m in family_metric_values if str(m.get("competition_status", "")) == "fallback_only")),
                    "suppressed": int(sum(1 for m in family_metric_values if str(m.get("competition_status", "")) == "suppressed")),
                }
                family_bootstrap_included_count = int(sum(1 for m in family_metric_values if bool(m.get("bootstrap_competition_included", False))))
                family_active_competition_count = int(
                    sum(
                        1
                        for m in family_metric_values
                        if bool(m.get("prior_eligible", False)) and str(m.get("competition_status", "")) in {"competitive", "competitive_bootstrap", "fallback_only"}
                    )
                )
                runtime_universe_summary = (
                    (de3_runtime_state_build or {}).get("usable_universe_summary", {})
                    if isinstance(de3_runtime_state_build, dict)
                    else {}
                )
                runtime_active_competition_count = int(
                    runtime_universe_summary.get("active_runtime_competition_family_count", family_active_competition_count)
                    or family_active_competition_count
                )
                chosen_total = int(len(chosen_df))
                chosen_family_unique_count = int(len(family_chosen_freq))
                top_family_chosen_share = 0.0
                top_2_family_share = 0.0
                if chosen_total > 0 and family_chosen_freq:
                    counts_desc = sorted([int(v) for v in family_chosen_freq.values()], reverse=True)
                    top_family_chosen_share = float(counts_desc[0] / float(chosen_total))
                    top_2_family_share = float(sum(counts_desc[:2]) / float(chosen_total))
                prior_eligible_but_never_chosen_count = int(
                    sum(
                        1
                        for metrics in family_runtime_metrics.values()
                        if bool(metrics.get("prior_eligible", False)) and int(metrics.get("chosen_count", 0) or 0) <= 0
                    )
                )
                prior_eligible_and_competitive_but_zero_realized_support_count = int(
                    sum(
                        1
                        for metrics in family_runtime_metrics.values()
                        if bool(metrics.get("prior_eligible", False))
                        and str(metrics.get("competition_status", "")).lower() in {"competitive", "competitive_bootstrap", "fallback_only"}
                        and int(metrics.get("executed_trade_count", 0) or 0) <= 0
                    )
                )
                bootstrap_competition_used_count = 0
                close_competition_decision_count = 0
                if not family_decisions_df.empty and "decision_id" in family_decisions_df.columns:
                    if "bootstrap_competition_used_decision" in family_decisions_df.columns:
                        boot_by_decision = (
                            family_decisions_df[["decision_id", "bootstrap_competition_used_decision"]]
                            .assign(
                                bootstrap_competition_used_decision=lambda df: _to_bool_col(
                                    df["bootstrap_competition_used_decision"]
                                )
                            )
                            .groupby("decision_id")["bootstrap_competition_used_decision"]
                            .max()
                        )
                        bootstrap_competition_used_count = int(boot_by_decision.sum())
                    if "close_competition_decision" in family_decisions_df.columns:
                        close_by_decision = (
                            family_decisions_df[["decision_id", "close_competition_decision"]]
                            .assign(close_competition_decision=lambda df: _to_bool_col(df["close_competition_decision"]))
                            .groupby("decision_id")["close_competition_decision"]
                            .max()
                        )
                        close_competition_decision_count = int(close_by_decision.sum())

                family_summary_payload = {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "de3_version": str(de3_runtime_db_version),
                    "family_mode_enabled": bool(de3_runtime_family_mode_enabled),
                    "decision_rows": int(len(family_decisions_df)),
                    "chosen_rows": int(len(chosen_df)),
                    "chosen_with_trade_rows": int(len(trade_eval)),
                    "family_chosen_frequency": family_chosen_freq,
                    "family_performance": family_perf,
                    "family_runtime_metrics": family_runtime_metrics,
                    "context_bucket_performance_by_family": context_perf,
                    "local_member_choice_distribution": local_member_distribution,
                    "local_bracket_adaptation_mode_distribution": local_adaptation_mode_distribution,
                    "local_bracket_adaptation_rate_by_family": local_adaptation_rate_by_family,
                    "evidence_support_tier_counts": evidence_support_tier_counts,
                    "competition_status_counts": competition_status_counts,
                    "family_evidence_support_tier_counts": family_evidence_support_tier_counts,
                    "family_competition_status_counts": family_competition_status_counts,
                    "bootstrap_included_family_count": int(family_bootstrap_included_count),
                    "active_runtime_competition_family_count": int(runtime_active_competition_count),
                    "chosen_family_competition_count": int(family_active_competition_count),
                    "inherited_family_universe_size": int(runtime_universe_summary.get("inherited_family_universe_size", 0) or 0),
                    "prior_eligible_family_count": int(runtime_universe_summary.get("prior_eligible_family_count", 0) or 0),
                    "suppressed_family_count": int(runtime_universe_summary.get("suppressed_family_count", 0) or 0),
                    "runtime_evidence_support_tier_counts": dict(runtime_universe_summary.get("evidence_support_tier_counts", {}) or {}),
                    "runtime_competition_status_counts": dict(runtime_universe_summary.get("competition_status_counts", {}) or {}),
                    "runtime_bootstrap_included_family_count": int(runtime_universe_summary.get("bootstrap_included_family_count", 0) or 0),
                    "family_competition_health": {
                        "chosen_family_unique_count": int(chosen_family_unique_count),
                        "top_family_chosen_share": float(top_family_chosen_share),
                        "top_2_family_share": float(top_2_family_share),
                        "prior_eligible_but_never_chosen_count": int(prior_eligible_but_never_chosen_count),
                        "prior_eligible_and_competitive_but_zero_realized_support_count": int(
                            prior_eligible_and_competitive_but_zero_realized_support_count
                        ),
                        "bootstrap_competition_used_count": int(bootstrap_competition_used_count),
                        "close_competition_decision_count": int(close_competition_decision_count),
                    },
                    "profile_support": {
                        "strong_support_count": int((support_tier_series == "strong").sum()) if not support_tier_series.empty else 0,
                        "mid_support_count": int((support_tier_series == "mid").sum()) if not support_tier_series.empty else 0,
                        "low_support_count": int((~support_tier_series.isin(["strong", "mid"])).sum()) if not support_tier_series.empty else 0,
                        "trusted_context_used_fraction": float(trusted_series.mean()) if not trusted_series.empty else 0.0,
                        "profile_used_fraction": float(profile_used_series.mean()) if not profile_used_series.empty else 0.0,
                        "fallback_to_priors_fraction": float(fallback_series.mean()) if not fallback_series.empty else 0.0,
                        "local_bracket_mode_counts": local_mode_series.fillna("none").astype(str).value_counts().to_dict(),
                        "local_bracket_override_fraction": float(local_override_series.mean()) if not local_override_series.empty else 0.0,
                    },
                    "family_universe_comparison": runtime_universe_summary,
                    "source_files": {
                        "family_decisions_csv": str(family_decisions_out_path),
                        "decision_csv": str(de3_decisions_out_path),
                        "trade_attribution_csv": str(trade_out_path),
                    },
                }
                family_summary_out_path.write_text(
                    json.dumps(family_summary_payload, indent=2, ensure_ascii=True),
                    encoding="utf-8",
                )

            de3_decisions_export_meta.update(
                {
                    "rows": int(decisions_row_count),
                    "path": str(de3_decisions_out_path),
                    "trade_attribution_rows": int(len(trade_rows)),
                    "trade_attribution_path": str(trade_out_path),
                    "summary_path": str(summary_out_path),
                    "family_rows": int(len(family_decisions_df)),
                    "family_path": str(family_decisions_out_path) if not family_decisions_df.empty else None,
                    "family_summary_path": str(family_summary_out_path) if not family_decisions_df.empty else None,
                }
            )
        except Exception as exc:
            logging.warning("DE3 decision export failed: %s", exc)
        finally:
            if de3_decisions_writer is not None:
                try:
                    _flush_de3_decision_buffer()
                except Exception:
                    pass
            if de3_decisions_file_handle is not None:
                try:
                    de3_decisions_file_handle.close()
                except Exception:
                    pass
                de3_decisions_file_handle = None
    if dynamic_engine3_strat is not None and de3_runtime_is_v4:
        runtime_meta_end_v4 = {}
        try:
            runtime_meta_end_v4 = (
                dynamic_engine3_strat.get_runtime_metadata()
                if hasattr(dynamic_engine3_strat, "get_runtime_metadata")
                else {}
            )
        except Exception:
            runtime_meta_end_v4 = {}
        if isinstance(runtime_meta_end_v4.get("v4_runtime_path_counters"), dict):
            de3_v4_runtime_path_counters = dict(runtime_meta_end_v4.get("v4_runtime_path_counters", {}))
        if isinstance(runtime_meta_end_v4.get("v4_activation_audit"), dict):
            de3_v4_activation_audit = dict(runtime_meta_end_v4.get("v4_activation_audit", {}))
        if isinstance(runtime_meta_end_v4.get("v4_router_summary"), dict):
            de3_v4_router_summary = dict(runtime_meta_end_v4.get("v4_router_summary", {}))
        if isinstance(runtime_meta_end_v4.get("v4_lane_selection_summary"), dict):
            de3_v4_lane_selection_summary = dict(runtime_meta_end_v4.get("v4_lane_selection_summary", {}))
        if isinstance(runtime_meta_end_v4.get("v4_bracket_summary"), dict):
            de3_v4_bracket_summary = dict(runtime_meta_end_v4.get("v4_bracket_summary", {}))
        if isinstance(runtime_meta_end_v4.get("v4_runtime_mode_summary"), dict):
            de3_v4_runtime_mode_summary = dict(runtime_meta_end_v4.get("v4_runtime_mode_summary", {}))
        if isinstance(runtime_meta_end_v4.get("v4_execution_policy_summary"), dict):
            de3_v4_execution_policy_summary = dict(runtime_meta_end_v4.get("v4_execution_policy_summary", {}))
        if isinstance(runtime_meta_end_v4.get("v4_decision_side_summary"), dict):
            de3_v4_decision_side_summary = dict(runtime_meta_end_v4.get("v4_decision_side_summary", {}))
        try:
            _write_json_report(
                de3_v4_activation_audit_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(de3_v4_activation_audit if isinstance(de3_v4_activation_audit, dict) else {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v4 activation audit export failed (%s): %s", de3_v4_activation_audit_path, exc)
        try:
            _write_json_report(
                de3_v4_runtime_path_counters_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    "counters": dict(de3_v4_runtime_path_counters or {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v4 runtime path counters export failed (%s): %s", de3_v4_runtime_path_counters_path, exc)
        try:
            _write_json_report(
                de3_v4_router_summary_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(de3_v4_router_summary if isinstance(de3_v4_router_summary, dict) else {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v4 router summary export failed (%s): %s", de3_v4_router_summary_path, exc)
        try:
            _write_json_report(
                de3_v4_lane_selection_summary_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(de3_v4_lane_selection_summary if isinstance(de3_v4_lane_selection_summary, dict) else {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v4 lane-selection summary export failed (%s): %s", de3_v4_lane_selection_summary_path, exc)
        try:
            _write_json_report(
                de3_v4_bracket_summary_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(de3_v4_bracket_summary if isinstance(de3_v4_bracket_summary, dict) else {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v4 bracket summary export failed (%s): %s", de3_v4_bracket_summary_path, exc)
        try:
            _write_json_report(
                de3_v4_runtime_mode_summary_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(de3_v4_runtime_mode_summary if isinstance(de3_v4_runtime_mode_summary, dict) else {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v4 runtime-mode summary export failed (%s): %s", de3_v4_runtime_mode_summary_path, exc)
        try:
            _write_json_report(
                de3_v4_execution_policy_summary_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(
                        de3_v4_execution_policy_summary
                        if isinstance(de3_v4_execution_policy_summary, dict)
                        else {}
                    ),
                },
            )
        except Exception as exc:
            logging.warning(
                "DE3v4 execution-policy summary export failed (%s): %s",
                de3_v4_execution_policy_summary_path,
                exc,
            )
        try:
            _write_json_report(
                de3_v4_decision_side_summary_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(
                        de3_v4_decision_side_summary
                        if isinstance(de3_v4_decision_side_summary, dict)
                        else {}
                    ),
                },
            )
        except Exception as exc:
            logging.warning(
                "DE3v4 decision-side summary export failed (%s): %s",
                de3_v4_decision_side_summary_path,
                exc,
            )
        try:
            de3_decisions_export_meta.update(
                {
                    "de3_v4_activation_audit_path": str(de3_v4_activation_audit_path),
                    "de3_v4_runtime_path_counters_path": str(de3_v4_runtime_path_counters_path),
                    "de3_v4_router_summary_path": str(de3_v4_router_summary_path),
                    "de3_v4_lane_selection_summary_path": str(de3_v4_lane_selection_summary_path),
                    "de3_v4_bracket_summary_path": str(de3_v4_bracket_summary_path),
                    "de3_v4_runtime_mode_summary_path": str(de3_v4_runtime_mode_summary_path),
                    "de3_v4_execution_policy_summary_path": str(
                        de3_v4_execution_policy_summary_path
                    ),
                    "de3_v4_decision_side_summary_path": str(
                        de3_v4_decision_side_summary_path
                    ),
                }
            )
        except Exception:
            pass

    if dynamic_engine3_strat is not None and de3_runtime_is_v3:
        runtime_meta_end = {}
        try:
            runtime_meta_end = (
                dynamic_engine3_strat.get_runtime_metadata()
                if hasattr(dynamic_engine3_strat, "get_runtime_metadata")
                else {}
            )
        except Exception:
            runtime_meta_end = {}

        if isinstance(runtime_meta_end.get("runtime_path_counters"), dict):
            de3_v3_runtime_path_counters = dict(runtime_meta_end.get("runtime_path_counters", {}))
        if isinstance(runtime_meta_end.get("bundle_usage_audit"), dict):
            de3_v3_bundle_usage_audit = dict(runtime_meta_end.get("bundle_usage_audit", {}))
        if isinstance(runtime_meta_end.get("config_usage_audit"), dict):
            de3_v3_config_usage_audit = dict(runtime_meta_end.get("config_usage_audit", {}))
        if isinstance(runtime_meta_end.get("score_path_audit"), dict):
            de3_v3_score_path_audit = dict(runtime_meta_end.get("score_path_audit", {}))
        if isinstance(runtime_meta_end.get("choice_path_audit"), dict):
            de3_v3_choice_path_audit = dict(runtime_meta_end.get("choice_path_audit", {}))
        if isinstance(runtime_meta_end.get("family_score_trace"), dict):
            de3_v3_family_score_trace = dict(runtime_meta_end.get("family_score_trace", {}))
        if isinstance(runtime_meta_end.get("member_resolution_audit"), dict):
            de3_v3_member_resolution_audit = dict(
                runtime_meta_end.get("member_resolution_audit", {})
            )
        if isinstance(runtime_meta_end.get("family_eligibility_trace"), dict):
            de3_v3_family_eligibility_trace = dict(
                runtime_meta_end.get("family_eligibility_trace", {})
            )
        if isinstance(runtime_meta_end.get("family_reachability_summary"), dict):
            de3_v3_family_reachability_summary = dict(
                runtime_meta_end.get("family_reachability_summary", {})
            )
        if isinstance(runtime_meta_end.get("family_compatibility_audit"), dict):
            de3_v3_family_compatibility_audit = dict(
                runtime_meta_end.get("family_compatibility_audit", {})
            )
        if isinstance(runtime_meta_end.get("pre_cap_candidate_audit"), dict):
            de3_v3_pre_cap_candidate_audit = dict(
                runtime_meta_end.get("pre_cap_candidate_audit", {})
            )
        if isinstance(runtime_meta_end.get("family_score_component_summary"), dict):
            de3_v3_family_score_component_summary = dict(
                runtime_meta_end.get("family_score_component_summary", {})
            )
        if isinstance(runtime_meta_end.get("family_score_delta_ladder"), dict):
            de3_v3_family_score_delta_ladder = dict(
                runtime_meta_end.get("family_score_delta_ladder", {})
            )
        if isinstance(runtime_meta_end.get("runtime_mode_summary"), dict):
            de3_v3_runtime_mode_summary = dict(
                runtime_meta_end.get("runtime_mode_summary", {})
            )
        if isinstance(runtime_meta_end.get("core_summary"), dict):
            de3_v3_core_summary = dict(runtime_meta_end.get("core_summary", {}))
        if isinstance(runtime_meta_end.get("t6_anchor_report"), dict):
            de3_v3_t6_anchor_report = dict(
                runtime_meta_end.get("t6_anchor_report", {})
            )
        if isinstance(runtime_meta_end.get("satellite_quality_report"), dict):
            de3_v3_satellite_quality_report = dict(
                runtime_meta_end.get("satellite_quality_report", {})
            )
        if isinstance(runtime_meta_end.get("portfolio_increment_report"), dict):
            de3_v3_portfolio_increment_report = dict(
                runtime_meta_end.get("portfolio_increment_report", {})
            )
        if isinstance(runtime_meta_end.get("activation_audit"), dict):
            # Keep start-of-backtest activation snapshot authoritative, but merge any runtime additions.
            merged_activation = dict(de3_v3_activation_audit)
            merged_activation.update(runtime_meta_end.get("activation_audit", {}))
            merged_activation["created_at"] = de3_v3_activation_audit.get("created_at", merged_activation.get("created_at"))
            de3_v3_activation_audit = merged_activation

        try:
            _write_json_report(
                de3_v3_runtime_path_counters_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    "family_artifact_path": str(de3_runtime_family_artifact or ""),
                    "counters": dict(de3_v3_runtime_path_counters),
                },
            )
        except Exception as exc:
            logging.warning("DE3v3 runtime path counters export failed (%s): %s", de3_v3_runtime_path_counters_path, exc)

        try:
            _write_json_report(
                de3_v3_bundle_usage_audit_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(de3_v3_bundle_usage_audit if isinstance(de3_v3_bundle_usage_audit, dict) else {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v3 bundle usage audit export failed (%s): %s", de3_v3_bundle_usage_audit_path, exc)

        try:
            _write_json_report(
                de3_v3_config_usage_audit_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(de3_v3_config_usage_audit if isinstance(de3_v3_config_usage_audit, dict) else {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v3 config usage audit export failed (%s): %s", de3_v3_config_usage_audit_path, exc)

        try:
            _write_json_report(
                de3_v3_score_path_audit_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(de3_v3_score_path_audit if isinstance(de3_v3_score_path_audit, dict) else {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v3 score-path audit export failed (%s): %s", de3_v3_score_path_audit_path, exc)

        try:
            _write_json_report(
                de3_v3_choice_path_audit_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(de3_v3_choice_path_audit if isinstance(de3_v3_choice_path_audit, dict) else {}),
                },
            )
        except Exception as exc:
            logging.warning("DE3v3 choice-path audit export failed (%s): %s", de3_v3_choice_path_audit_path, exc)

        family_decisions_df_audit = (
            family_decisions_df.copy()
            if "family_decisions_df" in locals() and isinstance(family_decisions_df, pd.DataFrame)
            else pd.DataFrame()
        )
        chosen_df_audit = (
            chosen_df.copy()
            if "chosen_df" in locals() and isinstance(chosen_df, pd.DataFrame)
            else pd.DataFrame()
        )
        family_summary_for_audits = (
            family_summary_payload
            if "family_summary_payload" in locals() and isinstance(family_summary_payload, dict)
            else {}
        )

        def _audit_to_bool_col(series: pd.Series) -> pd.Series:
            return (
                series.fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"1", "true", "yes", "y", "t"})
            )

        # Canonical family score trace export (one row per family candidate per decision).
        score_trace_df = pd.DataFrame()
        if not family_decisions_df_audit.empty:
            trace_cols = [
                "decision_id",
                "family_id",
                "family_candidate_count",
                "candidate_rank_before_adjustments",
                "family_candidate_source",
                "chosen_prior_component",
                "chosen_trusted_context_component",
                "chosen_evidence_adjustment",
                "chosen_adaptive_component",
                "chosen_competition_diversity_adjustment",
                "chosen_family_compatibility_component",
                "chosen_pre_adjustment_score",
                "chosen_final_family_score",
                "chosen_support_tier",
                "chosen_compatibility_tier",
                "chosen_session_compatibility_tier",
                "chosen_timeframe_compatibility_tier",
                "chosen_strategy_type_compatibility_tier",
                "chosen_context_trusted",
                "chosen_exploration_bonus_applied",
                "chosen_dominance_penalty_applied",
                "chosen_context_advantage_capped",
                "family_eligibility_tier",
                "family_preliminary_family_score",
                "family_preliminary_compatibility_penalty_component",
                "family_entered_pre_cap_pool",
                "family_survived_cap",
                "family_cap_drop_reason",
                "family_cap_tier_slot_used",
                "family_final_competition_pool_flag",
                "family_chosen_flag",
                "family_role",
            ]
            available_trace_cols = [
                col for col in trace_cols if col in family_decisions_df_audit.columns
            ]
            if available_trace_cols:
                score_trace_df = family_decisions_df_audit[available_trace_cols].copy()
                score_trace_df = score_trace_df.rename(
                    columns={
                        "chosen_prior_component": "prior_component",
                        "chosen_trusted_context_component": "trusted_context_component",
                        "chosen_evidence_adjustment": "evidence_adjustment",
                        "chosen_adaptive_component": "adaptive_component",
                        "chosen_competition_diversity_adjustment": "competition_diversity_adjustment",
                        "chosen_family_compatibility_component": "family_compatibility_component",
                        "chosen_pre_adjustment_score": "pre_adjustment_score",
                        "chosen_final_family_score": "final_family_score",
                        "chosen_support_tier": "support_tier",
                        "chosen_compatibility_tier": "compatibility_tier",
                        "chosen_session_compatibility_tier": "session_compatibility_tier",
                        "chosen_timeframe_compatibility_tier": "timeframe_compatibility_tier",
                        "chosen_strategy_type_compatibility_tier": "strategy_type_compatibility_tier",
                        "chosen_context_trusted": "context_trusted_flag",
                        "chosen_exploration_bonus_applied": "exploration_bonus_applied",
                        "chosen_dominance_penalty_applied": "dominance_penalty_applied",
                        "chosen_context_advantage_capped": "context_advantage_capped",
                        "family_eligibility_tier": "eligibility_tier",
                        "family_preliminary_family_score": "preliminary_family_score",
                        "family_preliminary_compatibility_penalty_component": "preliminary_compatibility_penalty_component",
                        "family_entered_pre_cap_pool": "entered_pre_cap_pool",
                        "family_survived_cap": "survived_cap",
                        "family_cap_drop_reason": "cap_drop_reason",
                        "family_cap_tier_slot_used": "cap_tier_slot_used",
                        "family_final_competition_pool_flag": "final_competition_pool_flag",
                    }
                )
                if "chosen_flag" not in score_trace_df.columns:
                    score_trace_df["chosen_flag"] = (
                        _audit_to_bool_col(score_trace_df["family_chosen_flag"])
                        if "family_chosen_flag" in score_trace_df.columns
                        else (
                            score_trace_df["family_role"].fillna("").astype(str).str.lower() == "chosen"
                            if "family_role" in score_trace_df.columns
                            else False
                        )
                    )
                if "runner_up_flag" not in score_trace_df.columns:
                    score_trace_df["runner_up_flag"] = (
                        score_trace_df["family_role"].fillna("").astype(str).str.lower()
                        == "runner_up"
                    ) if "family_role" in score_trace_df.columns else False
        if score_trace_df.empty:
            score_trace_rows_runtime = (
                list(de3_v3_family_score_trace.get("rows", []))
                if isinstance(de3_v3_family_score_trace, dict)
                and isinstance(de3_v3_family_score_trace.get("rows"), list)
                else []
            )
            score_trace_df = pd.DataFrame(score_trace_rows_runtime)
        try:
            if not score_trace_df.empty:
                score_trace_df.to_csv(de3_v3_family_score_trace_path, index=False)
            elif de3_v3_family_score_trace_path.exists():
                de3_v3_family_score_trace_path.unlink()
        except Exception as exc:
            logging.warning(
                "DE3v3 family score trace export failed (%s): %s",
                de3_v3_family_score_trace_path,
                exc,
            )

        # Member/bracket resolution audit export.
        member_resolution_rows = (
            list(de3_v3_member_resolution_audit.get("rows", []))
            if isinstance(de3_v3_member_resolution_audit, dict)
            and isinstance(de3_v3_member_resolution_audit.get("rows"), list)
            else []
        )
        member_resolution_df = pd.DataFrame(member_resolution_rows)
        if member_resolution_df.empty and not chosen_df_audit.empty:
            candidate_cols = [
                "decision_id",
                "chosen_family_id",
                "family_id_effective",
                "family_member_count",
                "local_member_count_within_family",
                "canonical_member_id",
                "sub_strategy",
                "local_bracket_adaptation_mode",
                "local_edge_component",
                "local_structural_component",
                "local_bracket_suitability_component",
                "local_final_member_score",
                "canonical_fallback_used",
                "why_non_anchor_beat_anchor",
                "why_anchor_forced",
                "no_local_alternative",
            ]
            present_cols = [col for col in candidate_cols if col in chosen_df_audit.columns]
            if present_cols:
                member_resolution_df = chosen_df_audit[present_cols].copy()
                if "chosen_family_id" not in member_resolution_df.columns:
                    member_resolution_df["chosen_family_id"] = member_resolution_df.get(
                        "family_id_effective", ""
                    )
                member_resolution_df["chosen_member_id"] = member_resolution_df.get(
                    "sub_strategy", ""
                )
                member_resolution_df["anchor_member_id"] = member_resolution_df.get(
                    "canonical_member_id", ""
                )
                member_resolution_df["anchor_selected"] = (
                    member_resolution_df["chosen_member_id"].fillna("").astype(str)
                    == member_resolution_df["anchor_member_id"].fillna("").astype(str)
                )
                member_resolution_df["local_member_selection_mode"] = member_resolution_df.get(
                    "local_bracket_adaptation_mode", "none"
                )
        if not member_resolution_df.empty and "anchor_selected" not in member_resolution_df.columns:
            member_resolution_df["anchor_selected"] = (
                member_resolution_df.get("chosen_member_id", pd.Series("", index=member_resolution_df.index))
                .fillna("")
                .astype(str)
                == member_resolution_df.get("anchor_member_id", pd.Series("", index=member_resolution_df.index))
                .fillna("")
                .astype(str)
            )
        mode_series = (
            member_resolution_df.get("local_member_selection_mode", pd.Series("none", index=member_resolution_df.index))
            .fillna("none")
            .astype(str)
            .str.lower()
            if not member_resolution_df.empty
            else pd.Series(dtype="object")
        )
        if not member_resolution_df.empty:
            anchor_member_series = (
                member_resolution_df.get(
                    "anchor_member_id", pd.Series("", index=member_resolution_df.index)
                )
                .fillna("")
                .astype(str)
                .str.strip()
            )
            chosen_member_series = (
                member_resolution_df.get(
                    "chosen_member_id", pd.Series("", index=member_resolution_df.index)
                )
                .fillna("")
                .astype(str)
                .str.strip()
            )
            anchor_selected_series = (
                member_resolution_df.get("anchor_selected", pd.Series(False, index=member_resolution_df.index))
                .fillna(False)
                .astype(bool)
            )
            anchor_alignment_known_series = (
                (anchor_member_series != "") & (chosen_member_series != "")
            )
        else:
            anchor_selected_series = pd.Series(dtype="bool")
            anchor_alignment_known_series = pd.Series(dtype="bool")
        member_resolution_payload = {
            "created_at": dt.datetime.now(NY_TZ).isoformat(),
            "active_de3_version": str(de3_runtime_db_version),
            "summary": {
                "row_count": int(len(member_resolution_df)),
                "anchor_selected_count": int(
                    anchor_selected_series.sum()
                )
                if not member_resolution_df.empty
                else 0,
                "non_anchor_selected_count": int(
                    ((~anchor_selected_series) & anchor_alignment_known_series).sum()
                )
                if not member_resolution_df.empty
                else 0,
                "unknown_anchor_alignment_count": int(
                    ((~anchor_selected_series) & (~anchor_alignment_known_series)).sum()
                )
                if not member_resolution_df.empty
                else 0,
                "frozen_mode_count": int((mode_series == "frozen").sum()) if not mode_series.empty else 0,
                "conservative_mode_count": int((mode_series == "conservative").sum()) if not mode_series.empty else 0,
                "full_mode_count": int((mode_series == "full").sum()) if not mode_series.empty else 0,
                "no_local_alternative_count": int(
                    _audit_to_bool_col(
                        member_resolution_df.get(
                            "no_local_alternative", pd.Series("", index=member_resolution_df.index)
                        )
                    ).sum()
                )
                if not member_resolution_df.empty
                else 0,
            },
            "decisions": member_resolution_df.to_dict(orient="records")
            if not member_resolution_df.empty
            else [],
        }
        try:
            _write_json_report(de3_v3_member_resolution_audit_path, member_resolution_payload)
        except Exception as exc:
            logging.warning(
                "DE3v3 member-resolution audit export failed (%s): %s",
                de3_v3_member_resolution_audit_path,
                exc,
            )

        # Family-eligibility trace export (one row per retained family scan per decision).
        family_eligibility_rows = (
            list(de3_v3_family_eligibility_trace.get("rows", []))
            if isinstance(de3_v3_family_eligibility_trace, dict)
            and isinstance(de3_v3_family_eligibility_trace.get("rows"), list)
            else []
        )
        family_eligibility_df = pd.DataFrame(family_eligibility_rows)
        if family_eligibility_df.empty and not family_decisions_df_audit.empty:
            eligibility_cols = [
                "decision_id",
                "family_id",
                "family_retained_runtime",
                "family_evaluated_for_eligibility",
                "family_coarse_eligible",
                "family_eligible_for_candidate_set",
                "family_eligibility_failure_reason",
                "family_coarse_eligibility_failure_reason",
                "family_excluded_by_session_mismatch",
                "family_excluded_by_side_mismatch",
                "family_excluded_by_timeframe_mismatch",
                "family_excluded_by_strategy_type_mismatch",
                "family_excluded_by_context_gate",
                "family_excluded_by_adaptive_policy_gate",
                "family_excluded_by_no_local_member_available",
                "family_excluded_by_temporary_exclusion",
                "family_excluded_by_candidate_cap",
                "family_compatibility_tier",
                "family_session_compatibility_tier",
                "family_timeframe_compatibility_tier",
                "family_strategy_type_compatibility_tier",
                "family_compatibility_component",
                "family_exact_match_eligible",
                "family_compatible_band_eligible",
                "family_incompatible_excluded",
                "family_entered_via_compatible_band",
                "family_eligibility_tier",
                "family_preliminary_family_score",
                "family_preliminary_compatibility_penalty_component",
                "family_entered_pre_cap_pool",
                "family_survived_cap",
                "family_cap_drop_reason",
                "family_cap_tier_slot_used",
                "family_final_competition_pool_flag",
                "pre_cap_candidate_count",
                "post_cap_candidate_count",
                "exact_match_survived_count",
                "compatible_band_survived_count",
                "compatible_band_dropped_by_cap_count",
                "family_coarse_compatibility_timeframe",
                "family_coarse_compatibility_session",
                "family_coarse_compatibility_side",
                "family_coarse_compatibility_strategy_type",
                "family_coarse_compatibility_threshold",
                "family_coarse_signature_sessions_seen",
                "family_coarse_signature_sides_seen",
                "family_coarse_signature_timeframes_seen",
                "family_coarse_signature_strategy_types_seen",
                "family_coarse_signature_decision_session",
                "family_coarse_signature_decision_hour_et",
                "family_member_candidates_seen_count",
                "feasible_member_count",
                "family_member_filtered_out_count",
                "family_chosen_flag",
                "retained_families_total",
                "retained_families_scanned",
                "retained_families_eligible",
                "retained_families_excluded",
                "retained_families_unscanned",
                "retained_family_scan_guarantee_pass",
            ]
            present_cols = [col for col in eligibility_cols if col in family_decisions_df_audit.columns]
            if present_cols:
                family_eligibility_df = family_decisions_df_audit[present_cols].copy()
                family_eligibility_df = family_eligibility_df.rename(
                    columns={
                        "family_retained_runtime": "retained_runtime",
                        "family_evaluated_for_eligibility": "evaluated_for_eligibility",
                        "family_coarse_eligible": "coarse_eligible",
                        "family_eligible_for_candidate_set": "eligible_for_candidate_set",
                        "family_eligibility_failure_reason": "failure_reason",
                        "family_coarse_eligibility_failure_reason": "coarse_failure_reason",
                        "family_excluded_by_session_mismatch": "excluded_by_session_mismatch",
                        "family_excluded_by_side_mismatch": "excluded_by_side_mismatch",
                        "family_excluded_by_timeframe_mismatch": "excluded_by_timeframe_mismatch",
                        "family_excluded_by_strategy_type_mismatch": "excluded_by_strategy_type_mismatch",
                        "family_excluded_by_context_gate": "excluded_by_context_gate",
                        "family_excluded_by_adaptive_policy_gate": "excluded_by_adaptive_policy_gate",
                        "family_excluded_by_no_local_member_available": "excluded_by_no_local_member_available",
                        "family_excluded_by_temporary_exclusion": "excluded_by_temporary_exclusion",
                        "family_excluded_by_candidate_cap": "excluded_by_candidate_cap",
                        "family_compatibility_tier": "compatibility_tier",
                        "family_session_compatibility_tier": "session_compatibility_tier",
                        "family_timeframe_compatibility_tier": "timeframe_compatibility_tier",
                        "family_strategy_type_compatibility_tier": "strategy_type_compatibility_tier",
                        "family_compatibility_component": "family_compatibility_component",
                        "family_exact_match_eligible": "exact_match_eligible",
                        "family_compatible_band_eligible": "compatible_band_eligible",
                        "family_incompatible_excluded": "incompatible_excluded",
                        "family_entered_via_compatible_band": "entered_via_compatible_band",
                        "family_eligibility_tier": "eligibility_tier",
                        "family_preliminary_family_score": "preliminary_family_score",
                        "family_preliminary_compatibility_penalty_component": "preliminary_compatibility_penalty_component",
                        "family_entered_pre_cap_pool": "entered_pre_cap_pool",
                        "family_survived_cap": "survived_cap",
                        "family_cap_drop_reason": "cap_drop_reason",
                        "family_cap_tier_slot_used": "cap_tier_slot_used",
                        "family_final_competition_pool_flag": "final_competition_pool_flag",
                        "family_coarse_compatibility_timeframe": "timeframe",
                        "family_coarse_compatibility_session": "session",
                        "family_coarse_compatibility_side": "side",
                        "family_coarse_compatibility_strategy_type": "de3_strategy_type",
                        "family_coarse_compatibility_threshold": "threshold",
                        "family_coarse_signature_sessions_seen": "sessions_seen",
                        "family_coarse_signature_sides_seen": "sides_seen",
                        "family_coarse_signature_timeframes_seen": "timeframes_seen",
                        "family_coarse_signature_strategy_types_seen": "strategy_types_seen",
                        "family_coarse_signature_decision_session": "decision_session",
                        "family_coarse_signature_decision_hour_et": "decision_hour_et",
                        "family_member_candidates_seen_count": "member_candidates_seen_count",
                        "family_member_filtered_out_count": "member_filtered_out_count",
                        "family_chosen_flag": "chosen",
                    }
                )
        try:
            if not family_eligibility_df.empty:
                family_eligibility_df.to_csv(de3_v3_family_eligibility_trace_path, index=False)
            elif de3_v3_family_eligibility_trace_path.exists():
                de3_v3_family_eligibility_trace_path.unlink()
        except Exception as exc:
            logging.warning(
                "DE3v3 family-eligibility trace export failed (%s): %s",
                de3_v3_family_eligibility_trace_path,
                exc,
            )

        # Family-reachability summary report.
        reachability_payload = (
            dict(de3_v3_family_reachability_summary)
            if isinstance(de3_v3_family_reachability_summary, dict)
            else {}
        )
        if not reachability_payload:
            retained_ids_fallback = set()
            if (
                not family_eligibility_df.empty
                and "family_id" in family_eligibility_df.columns
                and "retained_runtime" in family_eligibility_df.columns
            ):
                retained_mask = (
                    pd.Series(family_eligibility_df["retained_runtime"])
                    .fillna(False)
                    .astype(bool)
                )
                retained_ids_fallback = set(
                    family_eligibility_df.loc[retained_mask, "family_id"]
                    .fillna("")
                    .astype(str)
                    .tolist()
                )
            reachability_payload = {
                "retained_runtime_family_count": int(len(retained_ids_fallback)),
            }
        reachability_payload["created_at"] = dt.datetime.now(NY_TZ).isoformat()
        reachability_payload["active_de3_version"] = str(de3_runtime_db_version)
        reachability_payload["family_eligibility_trace_path"] = str(
            de3_v3_family_eligibility_trace_path
        )
        try:
            _write_json_report(
                de3_v3_family_reachability_summary_path,
                reachability_payload,
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 family-reachability summary export failed (%s): %s",
                de3_v3_family_reachability_summary_path,
                exc,
            )

        # Family-compatibility audit report.
        compatibility_payload = (
            dict(de3_v3_family_compatibility_audit)
            if isinstance(de3_v3_family_compatibility_audit, dict)
            else {}
        )
        if not compatibility_payload and isinstance(reachability_payload, dict):
            compatibility_payload = {
                "retained_runtime_family_count": int(
                    safe_float(
                        reachability_payload.get("retained_runtime_family_count", 0), 0
                    )
                ),
                "eligibility_outcome_counts": {
                    "exact_match_eligible": int(
                        safe_float(
                            reachability_payload.get(
                                "families_ever_exact_match_eligible_count", 0
                            ),
                            0,
                        )
                    ),
                    "compatible_band_eligible": int(
                        safe_float(
                            reachability_payload.get(
                                "families_ever_compatible_band_eligible_count", 0
                            ),
                            0,
                        )
                    ),
                    "incompatible_excluded": int(
                        safe_float(
                            reachability_payload.get(
                                "families_ever_incompatible_only_count", 0
                            ),
                            0,
                        )
                    ),
                },
                "exclusion_reasons_after_broadening": dict(
                    reachability_payload.get("top_compatibility_failure_reasons", {})
                    if isinstance(
                        reachability_payload.get("top_compatibility_failure_reasons", {}),
                        dict,
                    )
                    else {}
                ),
                "per_family": {},
                "candidate_set_size_histogram_current": dict(
                    (de3_v3_runtime_path_counters or {}).get(
                        "family_candidate_set_size_histogram", {}
                    )
                    if isinstance(
                        (de3_v3_runtime_path_counters or {}).get(
                            "family_candidate_set_size_histogram", {}
                        ),
                        dict,
                    )
                    else {}
                ),
                "candidate_set_size_histogram_before_broadening": None,
            }
        compatibility_payload["created_at"] = dt.datetime.now(NY_TZ).isoformat()
        compatibility_payload["active_de3_version"] = str(de3_runtime_db_version)
        compatibility_payload["family_reachability_summary_path"] = str(
            de3_v3_family_reachability_summary_path
        )
        try:
            _write_json_report(
                de3_v3_family_compatibility_audit_path,
                compatibility_payload,
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 family-compatibility audit export failed (%s): %s",
                de3_v3_family_compatibility_audit_path,
                exc,
            )

        # Pre-cap/post-cap candidate audit report.
        pre_cap_payload = (
            dict(de3_v3_pre_cap_candidate_audit)
            if isinstance(de3_v3_pre_cap_candidate_audit, dict)
            else {}
        )
        if not pre_cap_payload:
            invocations_fallback = int(
                safe_float((de3_v3_runtime_path_counters or {}).get("runtime_invocations", 0), 0)
            )
            pre_cap_payload = {
                "summary": {
                    "decision_count": int(invocations_fallback),
                    "pre_cap_candidate_count_avg": float(
                        safe_float((de3_v3_runtime_path_counters or {}).get("pre_cap_candidate_count_avg", 0.0), 0.0)
                    ),
                    "post_cap_candidate_count_avg": float(
                        safe_float((de3_v3_runtime_path_counters or {}).get("post_cap_candidate_count_avg", 0.0), 0.0)
                    ),
                    "exact_match_eligible_count_avg": float(
                        safe_float((de3_v3_runtime_path_counters or {}).get("pre_cap_exact_eligible_avg", 0.0), 0.0)
                    ),
                    "compatible_band_eligible_count_avg": float(
                        safe_float((de3_v3_runtime_path_counters or {}).get("pre_cap_compatible_eligible_avg", 0.0), 0.0)
                    ),
                    "exact_match_survived_count_avg": float(
                        safe_float((de3_v3_runtime_path_counters or {}).get("post_cap_exact_survived_avg", 0.0), 0.0)
                    ),
                    "compatible_band_survived_count_avg": float(
                        safe_float((de3_v3_runtime_path_counters or {}).get("post_cap_compatible_survived_avg", 0.0), 0.0)
                    ),
                    "compatible_band_dropped_by_cap_count_avg": float(
                        safe_float((de3_v3_runtime_path_counters or {}).get("compatible_dropped_by_cap_avg", 0.0), 0.0)
                    ),
                },
                "decisions": [],
                "per_family": {},
            }
        pre_cap_payload["created_at"] = dt.datetime.now(NY_TZ).isoformat()
        pre_cap_payload["active_de3_version"] = str(de3_runtime_db_version)
        pre_cap_payload["family_eligibility_trace_path"] = str(
            de3_v3_family_eligibility_trace_path
        )
        try:
            _write_json_report(
                de3_v3_pre_cap_candidate_audit_path,
                pre_cap_payload,
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 pre-cap candidate audit export failed (%s): %s",
                de3_v3_pre_cap_candidate_audit_path,
                exc,
            )

        artifact_path = Path(str(de3_runtime_family_artifact or "")).expanduser()
        if artifact_path and not artifact_path.is_absolute():
            artifact_path = Path(__file__).resolve().parent / artifact_path
        bundle_payload = {}
        if artifact_path and artifact_path.exists():
            try:
                bundle_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception:
                bundle_payload = {}

        raw_families = []
        refined_families = []
        retained_families = []
        suppressed_families = []
        suppressed_members = []
        if isinstance(bundle_payload, dict):
            raw_families = list(
                ((bundle_payload.get("raw_family_inventory", {}) if isinstance(bundle_payload.get("raw_family_inventory", {}), dict) else {}).get("families", []))
            )
            refined_families = list(
                ((bundle_payload.get("refined_family_inventory", {}) if isinstance(bundle_payload.get("refined_family_inventory", {}), dict) else {}).get("families", []))
            )
            retained_families = list(
                ((bundle_payload.get("retained_runtime_universe", {}) if isinstance(bundle_payload.get("retained_runtime_universe", {}), dict) else {}).get("families", []))
            )
            suppressed_families = list(
                ((bundle_payload.get("suppressed_families", {}) if isinstance(bundle_payload.get("suppressed_families", {}), dict) else {}).get("families", []))
            )
            suppressed_members = list(
                ((bundle_payload.get("suppressed_members", {}) if isinstance(bundle_payload.get("suppressed_members", {}), dict) else {}).get("members", []))
            )
        raw_member_count = int(
            sum(
                len((row.get("members", []) if isinstance(row, dict) and isinstance(row.get("members"), list) else []))
                for row in raw_families
            )
        )
        retained_member_count = int(
            sum(
                len((row.get("members", []) if isinstance(row, dict) and isinstance(row.get("members"), list) else []))
                for row in retained_families
            )
        )
        retained_family_ids = {
            str(row.get("family_id", "") or "")
            for row in retained_families
            if isinstance(row, dict) and str(row.get("family_id", "")).strip()
        }
        considered_family_ids = set()
        considered_family_ids_known = bool(not family_decisions_df_audit.empty)
        if not family_decisions_df_audit.empty and "family_id" in family_decisions_df_audit.columns:
            considered_family_ids = {
                str(v or "").strip()
                for v in family_decisions_df_audit["family_id"].fillna("").astype(str).tolist()
                if str(v or "").strip()
            }
        never_considered_retained = (
            sorted(fid for fid in retained_family_ids if fid not in considered_family_ids)
            if considered_family_ids_known
            else []
        )
        suppressed_family_ids = {
            str(row.get("family_id", "") or "")
            for row in suppressed_families
            if isinstance(row, dict) and str(row.get("family_id", "")).strip()
        }
        considered_suppressed_family_ids = sorted(fid for fid in considered_family_ids if fid in suppressed_family_ids)
        retained_member_ids = set()
        raw_family_ids = {
            str(row.get("family_id", "") or "")
            for row in raw_families
            if isinstance(row, dict) and str(row.get("family_id", "")).strip()
        }
        raw_member_ids = set()
        for row in raw_families:
            if not isinstance(row, dict):
                continue
            for member in (row.get("members", []) if isinstance(row.get("members"), list) else []):
                if not isinstance(member, dict):
                    continue
                member_id = str(member.get("member_id", member.get("strategy_id", "")) or "").strip()
                if member_id:
                    raw_member_ids.add(member_id)
        per_family_retained_member_counts = {}
        for row in retained_families:
            if not isinstance(row, dict):
                continue
            family_id = str(row.get("family_id", "") or "")
            members = row.get("members", []) if isinstance(row.get("members"), list) else []
            per_family_retained_member_counts[family_id] = int(len(members))
            for member in members:
                if not isinstance(member, dict):
                    continue
                member_id = str(member.get("member_id", member.get("strategy_id", "")) or "").strip()
                if member_id:
                    retained_member_ids.add(member_id)
        selected_member_ids = set()
        selected_member_ids_known = bool(not chosen_df_audit.empty)
        if not chosen_df_audit.empty and "sub_strategy" in chosen_df_audit.columns:
            selected_member_ids = {
                str(v or "").strip()
                for v in chosen_df_audit["sub_strategy"].fillna("").astype(str).tolist()
                if str(v or "").strip()
            }
        retained_members_never_selected = (
            sorted(mid for mid in retained_member_ids if mid not in selected_member_ids)
            if selected_member_ids_known
            else []
        )
        chosen_family_ids = {
            str(v or "").strip()
            for v in (
                chosen_df_audit["family_id_effective"].fillna("").astype(str).tolist()
                if (not chosen_df_audit.empty and "family_id_effective" in chosen_df_audit.columns)
                else []
            )
            if str(v or "").strip()
        }
        chosen_family_ids_not_in_raw = sorted(
            [fid for fid in chosen_family_ids if fid not in raw_family_ids]
        )
        chosen_member_ids_not_in_raw = sorted(
            [mid for mid in selected_member_ids if mid not in raw_member_ids]
        )
        raw_only_family_ids = sorted([fid for fid in raw_family_ids if fid not in retained_family_ids])
        raw_only_families_entering_runtime = sorted(
            [fid for fid in considered_family_ids if fid in raw_only_family_ids]
        )

        refined_vs_raw_payload = {
            "created_at": dt.datetime.now(NY_TZ).isoformat(),
            "active_de3_version": str(de3_runtime_db_version),
            "bundle_path": str(artifact_path),
            "bundle_fingerprint": _file_fingerprint(artifact_path) if artifact_path else {},
            "raw_family_count": int(len(raw_families)),
            "refined_family_count": int(len(refined_families)),
            "retained_runtime_family_count": int(len(retained_families)),
            "suppressed_family_count": int(len(suppressed_families)),
            "raw_member_count": int(raw_member_count),
            "refined_retained_member_count": int(retained_member_count),
            "suppressed_member_count": int(len(suppressed_members)),
            "per_family_retained_member_counts": per_family_retained_member_counts,
            "suppressed_family_ids": sorted(list(suppressed_family_ids)),
            "retained_refined_family_ids_never_considered_at_runtime": never_considered_retained,
            "retained_refined_family_ids_never_considered_count": int(len(never_considered_retained)),
            "retained_refined_family_ids_never_considered_known": bool(considered_family_ids_known),
            "retained_refined_member_ids_never_selected": retained_members_never_selected,
            "retained_refined_member_ids_never_selected_count": int(len(retained_members_never_selected)),
            "retained_refined_member_ids_never_selected_known": bool(selected_member_ids_known),
            "considered_family_ids_count": int(len(considered_family_ids)),
            "considered_suppressed_family_ids": considered_suppressed_family_ids,
            "runtime_candidate_construction_ignoring_refinement": bool(len(considered_suppressed_family_ids) > 0),
            "chosen_family_ids_not_in_raw": chosen_family_ids_not_in_raw,
            "chosen_member_ids_not_in_raw": chosen_member_ids_not_in_raw,
            "any_chosen_unavailable_in_raw_mode": bool(
                len(chosen_family_ids_not_in_raw) > 0 or len(chosen_member_ids_not_in_raw) > 0
            ),
            "raw_only_family_ids": raw_only_family_ids,
            "raw_only_families_entering_runtime_by_mistake": raw_only_families_entering_runtime,
            "raw_only_families_entering_runtime_by_mistake_flag": bool(
                len(raw_only_families_entering_runtime) > 0
            ),
        }
        try:
            _write_json_report(de3_v3_refined_vs_raw_audit_path, refined_vs_raw_payload)
        except Exception as exc:
            logging.warning("DE3v3 refined-vs-raw audit export failed (%s): %s", de3_v3_refined_vs_raw_audit_path, exc)

        # Family-competition health audit.
        comp_margin = float(
            (
                (
                    (de3_v3_activation_audit.get("family_competition_balancing_settings_used", {}))
                    if isinstance(de3_v3_activation_audit.get("family_competition_balancing_settings_used", {}), dict)
                    else {}
                ).get("competition_margin_points", 0.0)
            )
            or 0.0
        )
        if comp_margin <= 0.0:
            try:
                cfg_balance = (
                    ((CONFIG.get("DE3_V3", {}) if isinstance(CONFIG.get("DE3_V3", {}), dict) else {})
                     .get("family_competition", {})
                     .get("family_competition_balance", {}))
                )
                comp_margin = float(
                    cfg_balance.get("competition_margin_points", 0.22)
                    if isinstance(cfg_balance, dict)
                    else 0.22
                )
            except Exception:
                comp_margin = 0.22
        chosen_rows_for_health = (
            chosen_df_audit.copy() if isinstance(chosen_df_audit, pd.DataFrame) else pd.DataFrame()
        )
        runner_up_within_margin = 0
        balancing_changed_winner = 0
        if not chosen_rows_for_health.empty:
            if "chosen_vs_runner_up_score_delta" in chosen_rows_for_health.columns:
                delta_series = pd.to_numeric(
                    chosen_rows_for_health["chosen_vs_runner_up_score_delta"], errors="coerce"
                ).fillna(np.nan)
                runner_up_within_margin = int((delta_series.abs() <= float(comp_margin)).fillna(False).sum())
            if "competition_diversity_adjustment" in chosen_rows_for_health.columns:
                adj_series = pd.to_numeric(
                    chosen_rows_for_health["competition_diversity_adjustment"], errors="coerce"
                ).fillna(0.0)
                balancing_changed_winner = int((adj_series.abs() > 1e-12).sum())
        health_core = (
            family_summary_for_audits.get("family_competition_health", {})
            if isinstance(family_summary_for_audits.get("family_competition_health", {}), dict)
            else {}
        )
        candidate_histogram = (
            (de3_v3_runtime_path_counters.get("family_candidate_set_size_histogram", {}))
            if isinstance(de3_v3_runtime_path_counters, dict)
            else {}
        )
        top_inputs = (
            (de3_v3_runtime_path_counters.get("top_family_share_inputs", {}))
            if isinstance(de3_v3_runtime_path_counters, dict)
            else {}
        )
        family_competition_health_payload = {
            "created_at": dt.datetime.now(NY_TZ).isoformat(),
            "active_de3_version": str(de3_runtime_db_version),
            "chosen_family_unique_count": int(
                health_core.get("chosen_family_unique_count", top_inputs.get("window_size", 0))
                or 0
            ),
            "top_family_chosen_share": float(
                health_core.get("top_family_chosen_share", top_inputs.get("top_family_share", 0.0))
                or 0.0
            ),
            "top_2_family_share": float(
                health_core.get("top_2_family_share", top_inputs.get("top_2_family_share", 0.0))
                or 0.0
            ),
            "prior_eligible_but_never_chosen_count": int(
                health_core.get("prior_eligible_but_never_chosen_count", 0) or 0
            ),
            "retained_refined_but_never_chosen_count": int(
                len(never_considered_retained)
            ),
            "candidate_count_histogram": dict(candidate_histogram if isinstance(candidate_histogram, dict) else {}),
            "close_competition_decision_count": int(
                health_core.get("close_competition_decision_count", 0) or 0
            ),
            "decisions_where_runner_up_within_margin_count": int(runner_up_within_margin),
            "decisions_where_competition_balancing_changed_winner_count": int(
                balancing_changed_winner
            ),
            "average_pre_cap_family_candidate_count": float(
                safe_float(
                    (de3_v3_runtime_path_counters or {}).get("pre_cap_candidate_count_avg", 0.0),
                    0.0,
                )
            ),
            "average_post_cap_family_candidate_count": float(
                safe_float(
                    (de3_v3_runtime_path_counters or {}).get("post_cap_candidate_count_avg", 0.0),
                    0.0,
                )
            ),
            "average_compatible_band_families_pre_cap": float(
                safe_float(
                    (de3_v3_runtime_path_counters or {}).get("pre_cap_compatible_eligible_avg", 0.0),
                    0.0,
                )
            ),
            "average_compatible_band_families_post_cap": float(
                safe_float(
                    (de3_v3_runtime_path_counters or {}).get("post_cap_compatible_survived_avg", 0.0),
                    0.0,
                )
            ),
            "decisions_where_compatible_band_existed_but_all_dropped_by_cap": int(
                safe_float(
                    (de3_v3_runtime_path_counters or {}).get(
                        "decisions_with_compatible_pre_cap_all_dropped_count", 0
                    ),
                    0,
                )
            ),
            "family_monopoly_flags": {
                "monopoly_detected": bool(
                    float(health_core.get("top_family_chosen_share", top_inputs.get("top_family_share", 0.0)) or 0.0)
                    >= 0.90
                ),
                "top_family_id": str(top_inputs.get("top_family_id", "") or ""),
                "top_family_share": float(top_inputs.get("top_family_share", 0.0) or 0.0),
                "lookback_window": int(top_inputs.get("window_size", 0) or 0),
            },
        }
        try:
            _write_json_report(
                de3_v3_family_competition_health_path, family_competition_health_payload
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 family-competition health export failed (%s): %s",
                de3_v3_family_competition_health_path,
                exc,
            )

        # Family score component summary report.
        component_summary_payload = (
            dict(de3_v3_family_score_component_summary)
            if isinstance(de3_v3_family_score_component_summary, dict)
            else {}
        )
        if component_summary_payload:
            component_summary_payload = {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "active_de3_version": str(de3_runtime_db_version),
                **component_summary_payload,
            }
        else:
            component_summary_payload = {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "active_de3_version": str(de3_runtime_db_version),
                "row_count": 0,
                "decision_count_estimate": 0,
                "families": {},
                "overall": {},
            }
        try:
            _write_json_report(
                de3_v3_family_score_component_summary_path,
                component_summary_payload,
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 family-score component summary export failed (%s): %s",
                de3_v3_family_score_component_summary_path,
                exc,
            )

        # Family score delta ladder report.
        delta_ladder_payload = (
            dict(de3_v3_family_score_delta_ladder)
            if isinstance(de3_v3_family_score_delta_ladder, dict)
            else {}
        )
        if delta_ladder_payload:
            delta_ladder_payload = {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "active_de3_version": str(de3_runtime_db_version),
                **delta_ladder_payload,
            }
        else:
            delta_ladder_payload = {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "active_de3_version": str(de3_runtime_db_version),
                "row_count": 0,
                "decision_count_estimate": 0,
                "families": {},
                "frequently_eligible_rarely_chosen": [],
                "overall_component_gap_means": {},
                "dominant_gap_component_overall": "unknown",
            }
        try:
            _write_json_report(
                de3_v3_family_score_delta_ladder_path,
                delta_ladder_payload,
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 family-score delta ladder export failed (%s): %s",
                de3_v3_family_score_delta_ladder_path,
                exc,
            )
        try:
            _write_json_report(
                de3_v3_runtime_mode_summary_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(
                        de3_v3_runtime_mode_summary
                        if isinstance(de3_v3_runtime_mode_summary, dict)
                        else {}
                    ),
                },
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 runtime-mode summary export failed (%s): %s",
                de3_v3_runtime_mode_summary_path,
                exc,
            )
        try:
            _write_json_report(
                de3_v3_core_summary_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(
                        de3_v3_core_summary
                        if isinstance(de3_v3_core_summary, dict)
                        else {}
                    ),
                },
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 core summary export failed (%s): %s",
                de3_v3_core_summary_path,
                exc,
            )
        try:
            _write_json_report(
                de3_v3_t6_anchor_report_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(
                        de3_v3_t6_anchor_report
                        if isinstance(de3_v3_t6_anchor_report, dict)
                        else {}
                    ),
                },
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 T6 anchor report export failed (%s): %s",
                de3_v3_t6_anchor_report_path,
                exc,
            )
        try:
            _write_json_report(
                de3_v3_satellite_quality_report_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(
                        de3_v3_satellite_quality_report
                        if isinstance(de3_v3_satellite_quality_report, dict)
                        else {}
                    ),
                },
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 satellite quality report export failed (%s): %s",
                de3_v3_satellite_quality_report_path,
                exc,
            )
        try:
            _write_json_report(
                de3_v3_portfolio_increment_report_path,
                {
                    "created_at": dt.datetime.now(NY_TZ).isoformat(),
                    "active_de3_version": str(de3_runtime_db_version),
                    **(
                        de3_v3_portfolio_increment_report
                        if isinstance(de3_v3_portfolio_increment_report, dict)
                        else {}
                    ),
                },
            )
        except Exception as exc:
            logging.warning(
                "DE3v3 portfolio-increment report export failed (%s): %s",
                de3_v3_portfolio_increment_report_path,
                exc,
            )

        family_summary_for_inert = (
            family_summary_payload
            if "family_summary_payload" in locals() and isinstance(family_summary_payload, dict)
            else {}
        )
        health = family_summary_for_inert.get("family_competition_health", {}) if isinstance(family_summary_for_inert.get("family_competition_health"), dict) else {}
        top_share = float(health.get("top_family_chosen_share", 0.0) or 0.0)
        unique_count = int(health.get("chosen_family_unique_count", 0) or 0)
        invocations = int((de3_v3_runtime_path_counters or {}).get("runtime_invocations", 0) or 0)
        eq1_count = int((de3_v3_runtime_path_counters or {}).get("family_candidate_set_size_eq_1_count", 0) or 0)
        gt1_count = int((de3_v3_runtime_path_counters or {}).get("family_candidate_set_size_gt_1_count", 0) or 0)
        fallback_count = int((de3_v3_runtime_path_counters or {}).get("context_profile_fallback_to_priors_count", 0) or 0)
        trusted_count = int((de3_v3_runtime_path_counters or {}).get("context_profile_used_count", 0) or 0)
        dominant_gap_component = str(
            (delta_ladder_payload or {}).get("dominant_gap_component_overall", "unknown")
            or "unknown"
        ).strip().lower()
        component_gap_means = (
            dict((delta_ladder_payload or {}).get("overall_component_gap_means", {}))
            if isinstance((delta_ladder_payload or {}).get("overall_component_gap_means", {}), dict)
            else {}
        )
        concentration_category_map = {
            "prior": "prior_component_dominance",
            "context": "context_component_dominance",
            "evidence": "evidence_component_dominance",
            "adaptive": "adaptive_component_dominance",
            "compatibility": "compatibility_penalty_dominance",
            "competition": "competition_adjustment_too_weak",
        }
        concentration_reason_map = {
            "prior": "Winner advantage is mainly prior-component driven.",
            "context": "Winner advantage is mainly trusted-context-component driven.",
            "evidence": "Winner advantage is mainly evidence-adjustment driven.",
            "adaptive": "Winner advantage is mainly adaptive-component driven.",
            "compatibility": "Compatibility component is creating the largest winner gap.",
            "competition": "Competition adjustment component is not offsetting leader concentration.",
        }
        bottlenecks = []
        if top_share >= 0.95:
            concentration_category = concentration_category_map.get(
                dominant_gap_component,
                "runtime_scoring_concentration",
            )
            concentration_reason = concentration_reason_map.get(
                dominant_gap_component,
                "Top-family concentration remains high and score gaps favor the same family.",
            )
            bottlenecks.append(
                {
                    "rank": 1,
                    "category": str(concentration_category),
                    "reason": str(concentration_reason),
                    "evidence": {
                        "top_family_chosen_share": float(top_share),
                        "chosen_family_unique_count": int(unique_count),
                        "dominant_gap_component_overall": str(dominant_gap_component),
                        "overall_component_gap_means": component_gap_means,
                    },
                }
            )
        if invocations > 0 and (eq1_count / float(max(1, invocations))) >= 0.40:
            bottlenecks.append(
                {
                    "rank": len(bottlenecks) + 1,
                    "category": "feasibility_candidate_narrowing",
                    "reason": "Family candidate set is frequently singleton before final scoring.",
                    "evidence": {"runtime_invocations": int(invocations), "family_candidate_set_size_eq_1_count": int(eq1_count), "family_candidate_set_size_gt_1_count": int(gt1_count)},
                }
            )
        if invocations > 0 and (fallback_count / float(max(1, invocations))) >= 0.70:
            bottlenecks.append(
                {
                    "rank": len(bottlenecks) + 1,
                    "category": "context_support_issue",
                    "reason": "Most decisions fallback to priors; trusted context rarely used.",
                    "evidence": {"fallback_to_priors_decisions": int(fallback_count), "trusted_context_decisions": int(trusted_count), "runtime_invocations": int(invocations)},
                }
            )
        if considered_family_ids_known and len(never_considered_retained) > 0 and len(retained_family_ids) > 0:
            ratio = float(len(never_considered_retained) / float(max(1, len(retained_family_ids))))
            if ratio >= 0.50:
                bottlenecks.append(
                    {
                        "rank": len(bottlenecks) + 1,
                        "category": "refined_universe_reachability",
                        "reason": "Many retained refined families are never even considered at runtime.",
                        "evidence": {"retained_runtime_family_count": int(len(retained_family_ids)), "never_considered_retained_family_count": int(len(never_considered_retained)), "never_considered_ratio": float(ratio)},
                    }
                )
        bottlenecks = bottlenecks[:3]
        inert_summary_payload = {
            "created_at": dt.datetime.now(NY_TZ).isoformat(),
            "active_de3_version": str(de3_runtime_db_version),
            "are_recent_changes_reaching_runtime": bool(
                str((de3_v3_activation_audit or {}).get("loaded_universe", "")).lower() in {"retained_runtime", "refined", "refined_retained_filter"}
                and invocations > 0
            ),
            "top_family_dominance": {
                "chosen_family_unique_count": int(unique_count),
                "top_family_chosen_share": float(top_share),
            },
            "runtime_branch_activity": {
                "runtime_invocations": int(invocations),
                "family_candidate_set_size_eq_1_count": int(eq1_count),
                "family_candidate_set_size_gt_1_count": int(gt1_count),
                "context_profile_used_count": int(trusted_count),
                "context_profile_fallback_to_priors_count": int(fallback_count),
            },
            "bundle_usage_highlights": {
                "loaded_universe": str((de3_v3_bundle_usage_audit or {}).get("loaded_universe", "")),
                "unused_or_inert_sections": list((de3_v3_bundle_usage_audit or {}).get("unused_or_inert_sections", [])),
            },
            "refined_vs_raw_highlights": {
                "retained_runtime_family_count": int(len(retained_family_ids)),
                "never_considered_retained_family_count": int(len(never_considered_retained)),
                "runtime_candidate_construction_ignoring_refinement": bool(refined_vs_raw_payload.get("runtime_candidate_construction_ignoring_refinement", False)),
            },
            "scoring_concentration_driver": {
                "dominant_gap_component_overall": str(dominant_gap_component),
                "overall_component_gap_means": component_gap_means,
            },
            "top_3_likely_bottlenecks": bottlenecks,
            "recommended_next_work_category": (
                bottlenecks[0]["category"] if bottlenecks else "insufficient_diagnostic_data"
            ),
        }
        try:
            _write_json_report(de3_v3_inert_change_summary_path, inert_summary_payload)
        except Exception as exc:
            logging.warning("DE3v3 inert-change summary export failed (%s): %s", de3_v3_inert_change_summary_path, exc)
        try:
            latest_backtest_json_path = None
            try:
                backtest_report_dir = Path(__file__).resolve().parent / "backtest_reports"
                if backtest_report_dir.exists():
                    candidates = sorted(
                        backtest_report_dir.glob("backtest_*.json"),
                        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
                        reverse=True,
                    )
                    if candidates:
                        latest_backtest_json_path = candidates[0]
            except Exception:
                latest_backtest_json_path = None
            readiness_sections = {
                "backtest_json": {
                    "path": str(latest_backtest_json_path) if latest_backtest_json_path else None,
                    "available": bool(
                        isinstance(latest_backtest_json_path, Path)
                        and latest_backtest_json_path.exists()
                    ),
                    "note": "best-effort latest backtest report detection",
                },
                "family_summary": {
                    "path": str(family_summary_out_path) if "family_summary_out_path" in locals() else None,
                    "available": bool(
                        "family_summary_out_path" in locals()
                        and isinstance(family_summary_out_path, Path)
                        and family_summary_out_path.exists()
                    ),
                },
                "activation_audit": {
                    "path": str(de3_v3_activation_audit_path),
                    "available": bool(de3_v3_activation_audit_path.exists()),
                },
                "config_usage_audit": {
                    "path": str(de3_v3_config_usage_audit_path),
                    "available": bool(de3_v3_config_usage_audit_path.exists()),
                },
                "bundle_usage_audit": {
                    "path": str(de3_v3_bundle_usage_audit_path),
                    "available": bool(de3_v3_bundle_usage_audit_path.exists()),
                },
                "runtime_path_counters": {
                    "path": str(de3_v3_runtime_path_counters_path),
                    "available": bool(de3_v3_runtime_path_counters_path.exists()),
                },
                "score_trace_decomposition": {
                    "path": str(de3_v3_family_score_trace_path),
                    "available": bool(de3_v3_family_score_trace_path.exists()),
                },
                "choice_path_audit": {
                    "path": str(de3_v3_choice_path_audit_path),
                    "available": bool(de3_v3_choice_path_audit_path.exists()),
                },
                "member_resolution_audit": {
                    "path": str(de3_v3_member_resolution_audit_path),
                    "available": bool(de3_v3_member_resolution_audit_path.exists()),
                },
                "family_eligibility_trace": {
                    "path": str(de3_v3_family_eligibility_trace_path),
                    "available": bool(de3_v3_family_eligibility_trace_path.exists()),
                },
                "family_reachability_summary": {
                    "path": str(de3_v3_family_reachability_summary_path),
                    "available": bool(de3_v3_family_reachability_summary_path.exists()),
                },
                "family_compatibility_audit": {
                    "path": str(de3_v3_family_compatibility_audit_path),
                    "available": bool(de3_v3_family_compatibility_audit_path.exists()),
                },
                "pre_cap_candidate_audit": {
                    "path": str(de3_v3_pre_cap_candidate_audit_path),
                    "available": bool(de3_v3_pre_cap_candidate_audit_path.exists()),
                },
                "family_competition_health": {
                    "path": str(de3_v3_family_competition_health_path),
                    "available": bool(de3_v3_family_competition_health_path.exists()),
                },
                "family_score_component_summary": {
                    "path": str(de3_v3_family_score_component_summary_path),
                    "available": bool(
                        de3_v3_family_score_component_summary_path.exists()
                    ),
                },
                "family_score_delta_ladder": {
                    "path": str(de3_v3_family_score_delta_ladder_path),
                    "available": bool(de3_v3_family_score_delta_ladder_path.exists()),
                },
                "runtime_mode_summary": {
                    "path": str(de3_v3_runtime_mode_summary_path),
                    "available": bool(de3_v3_runtime_mode_summary_path.exists()),
                },
                "core_summary": {
                    "path": str(de3_v3_core_summary_path),
                    "available": bool(de3_v3_core_summary_path.exists()),
                },
                "t6_anchor_report": {
                    "path": str(de3_v3_t6_anchor_report_path),
                    "available": bool(de3_v3_t6_anchor_report_path.exists()),
                },
                "satellite_quality_report": {
                    "path": str(de3_v3_satellite_quality_report_path),
                    "available": bool(de3_v3_satellite_quality_report_path.exists()),
                },
                "portfolio_increment_report": {
                    "path": str(de3_v3_portfolio_increment_report_path),
                    "available": bool(de3_v3_portfolio_increment_report_path.exists()),
                },
            }
            missing_sections = [
                key for key, row in readiness_sections.items() if not bool(row.get("available", False))
            ]
            diff_readiness_payload = {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "active_de3_version": str(de3_runtime_db_version),
                "run_id": str(de3_v3_run_id),
                "sections": readiness_sections,
                "missing_sections": missing_sections,
                "diff_comparable": bool(len(missing_sections) == 0),
            }
            _write_json_report(de3_v3_diff_readiness_path, diff_readiness_payload)
        except Exception as exc:
            logging.warning("DE3v3 diff-readiness export failed (%s): %s", de3_v3_diff_readiness_path, exc)
        try:
            de3_decisions_export_meta.update(
                {
                    "activation_audit_path": str(de3_v3_activation_audit_path),
                    "runtime_path_counters_path": str(de3_v3_runtime_path_counters_path),
                    "bundle_usage_audit_path": str(de3_v3_bundle_usage_audit_path),
                    "config_usage_audit_path": str(de3_v3_config_usage_audit_path),
                    "score_path_audit_path": str(de3_v3_score_path_audit_path),
                    "choice_path_audit_path": str(de3_v3_choice_path_audit_path),
                    "refined_vs_raw_audit_path": str(de3_v3_refined_vs_raw_audit_path),
                    "family_score_trace_path": str(de3_v3_family_score_trace_path),
                    "family_competition_health_path": str(de3_v3_family_competition_health_path),
                    "member_resolution_audit_path": str(de3_v3_member_resolution_audit_path),
                    "family_eligibility_trace_path": str(de3_v3_family_eligibility_trace_path),
                    "family_reachability_summary_path": str(de3_v3_family_reachability_summary_path),
                    "family_compatibility_audit_path": str(de3_v3_family_compatibility_audit_path),
                    "pre_cap_candidate_audit_path": str(de3_v3_pre_cap_candidate_audit_path),
                    "family_score_component_summary_path": str(
                        de3_v3_family_score_component_summary_path
                    ),
                    "family_score_delta_ladder_path": str(
                        de3_v3_family_score_delta_ladder_path
                    ),
                    "runtime_mode_summary_path": str(de3_v3_runtime_mode_summary_path),
                    "core_summary_path": str(de3_v3_core_summary_path),
                    "t6_anchor_report_path": str(de3_v3_t6_anchor_report_path),
                    "satellite_quality_report_path": str(
                        de3_v3_satellite_quality_report_path
                    ),
                    "portfolio_increment_report_path": str(
                        de3_v3_portfolio_increment_report_path
                    ),
                    "diff_readiness_path": str(de3_v3_diff_readiness_path),
                    "inert_change_summary_path": str(de3_v3_inert_change_summary_path),
                }
            )
        except Exception:
            pass
    if speed_enabled and bool(speed_cfg.get("suppress_warnings", True)):
        root_logger.setLevel(prev_root_level)
    if console_progress_enabled:
        total_elapsed_sim = max(0.0, time.perf_counter() - sim_started_at)
        status_txt = "cancelled" if cancelled else "complete"
        print(
            (
                f"[backtest] simulation {status_txt} | elapsed={_format_hms(total_elapsed_sim)} "
                f"bars={bar_count}/{total_bars} trades={trades} net=${equity:.2f} dd=${max_dd:.2f}"
            ),
            flush=True,
        )
    risk_metrics = _compute_backtest_risk_metrics(tracker.trades)
    return {
        "equity": equity,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "max_drawdown": max_dd,
        **risk_metrics,
        "cancelled": cancelled,
        "max_stoploss_cap_points": BACKTEST_MAX_STOPLOSS_POINTS,
        "max_stoploss_cap_disabled_for_mlphysics": BACKTEST_DISABLE_MAX_STOPLOSS_FOR_MLPHYSICS,
        "max_stoploss_cap_disabled_for_de3_v2": BACKTEST_DISABLE_MAX_STOPLOSS_FOR_DE3_V2,
        "drawdown_size_scaling_enabled": BACKTEST_DRAWDOWN_SIZE_SCALING_ENABLED,
        "drawdown_size_scaling_start_usd": float(BACKTEST_DRAWDOWN_SIZE_SCALING_START_USD),
        "drawdown_size_scaling_max_usd": float(BACKTEST_DRAWDOWN_SIZE_SCALING_MAX_USD),
        "drawdown_size_scaling_base_contracts": int(BACKTEST_DRAWDOWN_SIZE_SCALING_BASE_CONTRACTS),
        "drawdown_size_scaling_min_contracts": int(BACKTEST_DRAWDOWN_SIZE_SCALING_MIN_CONTRACTS),
        "entry_window_block_enabled": BACKTEST_ENFORCE_NO_NEW_ENTRIES_WINDOW,
        "entry_window_block_hours_et": [
            BACKTEST_NO_NEW_ENTRIES_START_HOUR_ET,
            BACKTEST_NO_NEW_ENTRIES_END_HOUR_ET,
        ],
        "holiday_closure_enabled": BACKTEST_ENFORCE_US_HOLIDAY_CLOSURE,
        "holiday_closed_dates_count": int(len(holiday_closed_dates_et)),
        "holiday_closure_sessions_et": list(BACKTEST_HOLIDAY_CLOSURE_SESSIONS_ET),
        "holiday_extra_closed_dates_et": list(BACKTEST_EXTRA_CLOSED_DATES_ET),
        "force_flat_enabled": BACKTEST_FORCE_FLAT_AT_TIME,
        "force_flat_time_et": f"{BACKTEST_FORCE_FLAT_HOUR_ET:02d}:{BACKTEST_FORCE_FLAT_MINUTE_ET:02d}",
        "session_flat_closes": int(session_flat_closes),
        "session_entry_blocks": int(session_entry_blocks),
        "holiday_flat_closes": int(holiday_flat_closes),
        "holiday_entry_blocks": int(holiday_entry_blocks),
        "holiday_signal_blocks": int(holiday_signal_blocks),
        "sl_cap_shadow_lock_until_index": (
            int(sl_cap_shadow_lock_until_index) if sl_cap_shadow_lock_until_index >= 0 else None
        ),
        "sl_cap_shadow_lock_triggers": int(sl_cap_shadow_lock_trigger_count),
        "sl_cap_shadow_lock_entry_blocks": int(sl_cap_shadow_lock_entry_blocks),
        "bar_minutes": bar_minutes,
        "report": report_text,
        "trade_log": [serialize_trade(trade) for trade in tracker.trades],
        "market_conditions_summary": market_conditions_summary,
        "ml_diagnostics": tracker.ml_diagnostics,
        "ml_diagnostics_summary": ml_diagnostics_summary,
        "fast_mode": {"enabled": fast_enabled, "bar_stride": bar_stride, "skip_mfe_mae": skip_mfe_mae},
        "speed_profile": {
            "enabled": bool(speed_enabled),
            "ml_eval_stride": int(ml_eval_stride),
            "dist_input_bars": int(backtest_dist_bars),
            "ml_diagnostics_enabled": bool(ml_diag_enabled),
            "flip_confidence_enabled": bool((flip_cfg or {}).get("enabled", False)),
            "market_snapshots_enabled": bool(market_snapshots_enabled),
            "market_summary_enabled": bool(market_summary_enabled),
        },
        "ml_only_diagnostic_profile": {
            "active": bool(ml_only_diag_active),
            "filters": sorted(filter_selection) if filter_selection is not None else [],
            "speed_profile_forced_off": bool(
                ml_only_diag_active and bool(ml_only_diag_cfg.get("disable_speed_profile", True))
            ),
            "ml_diagnostics_forced_on": bool(
                ml_only_diag_active and bool(ml_only_diag_cfg.get("force_ml_diagnostics", True))
            ),
        },
        "selection": {
            "strategies": selected_strategy_names,
            "filters": selected_filter_names,
        },
        "backtest_multipliers": {
            **(multiplier_dataset_info or {}),
            "active_sl_multiplier": float(active_sl_multiplier),
            "active_tp_multiplier": float(active_tp_multiplier),
            "active_chop_multiplier": float(active_chop_multiplier),
            "updates_applied": int(multiplier_updates),
        },
        "ny_continuation_gate_summary": ny_gate_summary,
        "flip_confidence": flip_payload,
        "de3_veto_summary": de3_veto_summary,
        "de3_veto_counterfactual": de3_veto_counterfactual,
        "de3_meta_summary": de3_meta_summary,
        "de3_manifold_adaptation_summary": de3_manifold_adaptation_summary,
        "de3_walkforward_gate_summary": {
            "enabled": bool(de3_walkforward_gate_enabled),
            "artifact_path": str(de3_walkforward_gate_artifact_path)
            if de3_walkforward_gate_artifact_path is not None
            else None,
            "mode": str(de3_walkforward_gate_mode),
            "lane_context_history_window": int(de3_walkforward_gate_lane_history_window),
            "variant_history_window": int(de3_walkforward_gate_variant_history_window),
            "defensive_size_multiplier": float(de3_walkforward_gate_defensive_mult),
            "min_contracts": int(de3_walkforward_gate_min_contracts),
            "reduce_only": bool(de3_walkforward_gate_reduce_only),
            "periods_loaded": int(len(de3_walkforward_gate_periods)),
            "checked": int(de3_walkforward_gate_checked),
            "applied": int(de3_walkforward_gate_applied),
            "blocked": int(de3_walkforward_gate_blocked),
            "state_counts": {
                str(name): int(count)
                for name, count in de3_walkforward_gate_state_counts.items()
            },
            "period_hits": {
                str(name): int(count)
                for name, count in de3_walkforward_gate_period_hits.items()
            },
        },
        "de3_intraday_regime_summary": {
            "enabled": bool(de3_intraday_regime_enabled),
            "mode": str(de3_intraday_regime_mode),
            "apply_sessions": sorted(de3_intraday_regime_apply_sessions),
            "apply_lanes": sorted(de3_intraday_regime_apply_lanes),
            "enable_bullish_mirror": bool(de3_intraday_regime_enable_bullish_mirror),
            "defensive_size_multiplier": float(de3_intraday_regime_defensive_mult),
            "min_contracts": int(de3_intraday_regime_min_contracts),
            "reduce_only": bool(de3_intraday_regime_reduce_only),
            "defensive_score_threshold": float(de3_intraday_regime_defensive_score),
            "block_score_threshold": float(de3_intraday_regime_block_score),
            "dominance_threshold": float(de3_intraday_regime_dominance),
            "block_dominance_threshold": float(de3_intraday_regime_block_dominance),
            "require_signal_weakness_for_block": bool(de3_intraday_regime_require_weak_block),
            "require_signal_weakness_for_defensive": bool(de3_intraday_regime_require_weak_defensive),
            "checked": int(de3_intraday_regime_checked),
            "applied": int(de3_intraday_regime_applied),
            "blocked": int(de3_intraday_regime_blocked),
            "state_counts": {
                str(name): int(count)
                for name, count in de3_intraday_regime_state_counts.items()
            },
            "action_counts": {
                str(name): int(count)
                for name, count in de3_intraday_regime_action_counts.items()
            },
        },
        "de3_backtest_admission_summary": {
            "enabled": bool(de3_admission_enabled),
            "key_granularity": str(de3_admission_key_granularity),
            "history_window_trades": int(de3_admission_history_window),
            "warmup_trades": int(de3_admission_warmup_trades),
            "cold_avg_net_per_contract_usd": float(de3_admission_cold_avg),
            "cold_max_winrate": float(de3_admission_cold_max_winrate),
            "defensive_size_multiplier": float(de3_admission_defensive_mult),
            "block_avg_net_per_contract_usd": (
                float(de3_admission_block_avg)
                if math.isfinite(de3_admission_block_avg)
                else None
            ),
            "block_max_winrate": (
                float(de3_admission_block_max_winrate)
                if math.isfinite(de3_admission_block_max_winrate)
                else None
            ),
            "min_contracts": int(de3_admission_min_contracts),
            "reduce_only": bool(de3_admission_reduce_only),
            "require_signal_weakness": bool(de3_admission_require_signal_weakness),
            "max_execution_quality_score": (
                float(de3_admission_max_execution_quality)
                if math.isfinite(de3_admission_max_execution_quality)
                else None
            ),
            "max_entry_model_margin": (
                float(de3_admission_max_entry_margin)
                if math.isfinite(de3_admission_max_entry_margin)
                else None
            ),
            "max_route_confidence": (
                float(de3_admission_max_route_confidence)
                if math.isfinite(de3_admission_max_route_confidence)
                else None
            ),
            "max_edge_points": (
                float(de3_admission_max_edge_points)
                if math.isfinite(de3_admission_max_edge_points)
                else None
            ),
            "checked": int(de3_admission_checked),
            "applied": int(de3_admission_applied),
            "blocked": int(de3_admission_blocked),
            "state_counts": {str(k): int(v) for k, v in de3_admission_state_counts.items()},
            "top_action_keys": [
                {"key": str(name), "actions": int(count)}
                for name, count in de3_admission_key_actions.most_common(10)
            ],
        },
        "de3_entry_model_margin_summary": {
            "enabled": bool(de3_entry_margin_enabled),
            "min_contracts": int(de3_entry_margin_min_contracts),
            "max_contracts": int(de3_entry_margin_max_contracts),
            "reduce_only": bool(de3_entry_margin_reduce_only),
            "defensive_max_margin": float(de3_entry_margin_defensive_max),
            "defensive_size_multiplier": float(de3_entry_margin_defensive_mult),
            "lane_scope_size_multiplier": float(de3_entry_margin_lane_scope_mult),
            "conservative_tier_size_multiplier": float(de3_entry_margin_conservative_mult),
            "aggressive_min_margin": float(de3_entry_margin_aggressive_min),
            "aggressive_size_multiplier": float(de3_entry_margin_aggressive_mult),
            "aggressive_variant_only": bool(de3_entry_margin_aggressive_variant_only),
            "checked": int(de3_entry_margin_checked),
            "applied": int(de3_entry_margin_applied),
            "state_counts": {
                str(name): int(count)
                for name, count in de3_entry_margin_state_counts.items()
            },
        },
        "de3_backtest_signal_size_summary": {
            "enabled": bool(de3_signal_size_enabled),
            "applied": int(de3_signal_size_applied),
            "rule_hits": {
                str(name): int(count)
                for name, count in de3_signal_size_rule_hits.items()
            },
        },
        "de3_policy_overlay_summary": {
            "enabled": bool(de3_policy_overlay_enabled),
            "reduce_only": bool(de3_policy_overlay_reduce_only),
            "min_contracts": int(de3_policy_overlay_min_contracts),
            "min_policy_confidence": (
                float(de3_policy_overlay_min_confidence)
                if math.isfinite(de3_policy_overlay_min_confidence)
                else None
            ),
            "min_policy_bucket_samples": int(de3_policy_overlay_min_bucket_samples),
            "require_signal_weakness": bool(de3_policy_overlay_require_signal_weakness),
            "max_execution_quality_score": (
                float(de3_policy_overlay_max_execution_quality)
                if math.isfinite(de3_policy_overlay_max_execution_quality)
                else None
            ),
            "max_entry_model_margin": (
                float(de3_policy_overlay_max_entry_margin)
                if math.isfinite(de3_policy_overlay_max_entry_margin)
                else None
            ),
            "max_route_confidence": (
                float(de3_policy_overlay_max_route_confidence)
                if math.isfinite(de3_policy_overlay_max_route_confidence)
                else None
            ),
            "max_edge_points": (
                float(de3_policy_overlay_max_edge_points)
                if math.isfinite(de3_policy_overlay_max_edge_points)
                else None
            ),
            "checked": int(de3_policy_overlay_checked),
            "applied": int(de3_policy_overlay_applied),
            "state_counts": {
                str(name): int(count)
                for name, count in de3_policy_overlay_state_counts.items()
            },
        },
        "de3_variant_adaptation_summary": {
            "enabled": bool(de3_variant_adapt_enabled),
            "history_window_trades": int(de3_variant_adapt_history_window),
            "warmup_trades": int(de3_variant_adapt_warmup_trades),
            "cold_avg_net_per_contract_usd": float(de3_variant_adapt_cold_avg),
            "cold_max_winrate": float(de3_variant_adapt_cold_max_winrate),
            "cold_size_multiplier": float(de3_variant_adapt_cold_mult),
            "max_lifetime_avg_net_per_contract_usd": (
                float(de3_variant_adapt_max_lifetime_avg)
                if math.isfinite(de3_variant_adapt_max_lifetime_avg)
                else None
            ),
            "deep_cold_avg_net_per_contract_usd": (
                float(de3_variant_adapt_deep_cold_avg)
                if math.isfinite(de3_variant_adapt_deep_cold_avg)
                else None
            ),
            "deep_cold_size_multiplier": float(de3_variant_adapt_deep_cold_mult),
            "min_contracts": int(de3_variant_adapt_min_contracts),
            "reduce_only": bool(de3_variant_adapt_reduce_only),
            "checked": int(de3_variant_adapt_checked),
            "applied": int(de3_variant_adapt_applied),
            "state_counts": {str(k): int(v) for k, v in de3_variant_adapt_state_counts.items()},
            "top_reduced_variants": [
                {"variant": str(name), "reductions": int(count)}
                for name, count in de3_variant_adapt_variant_reductions.most_common(10)
            ],
        },
        "de3_meta_counterfactual": de3_meta_counterfactual,
        "de3_runtime": {
            "version": str(de3_runtime_db_version),
            "is_v2": bool(de3_runtime_is_v2),
            "is_v3": bool(de3_runtime_is_v3),
            "is_v4": bool(de3_runtime_is_v4),
            "family_mode_enabled": bool(de3_runtime_family_mode_enabled),
            "family_artifact_path": str(de3_runtime_family_artifact) if de3_runtime_family_artifact else None,
            "family_artifact_loaded": bool(de3_runtime_family_artifact_loaded),
            "context_profiles_loaded": bool(de3_runtime_context_profiles_loaded),
            "enriched_export_required": bool(de3_runtime_enriched_export_required),
            "context_profile_build": de3_runtime_context_profile_build,
            "active_context_dimensions": list(de3_runtime_active_context_dimensions),
            "context_trust": dict(de3_runtime_context_trust),
            "local_bracket_freeze": dict(de3_runtime_local_bracket_freeze),
            "runtime_state_loaded": bool(de3_runtime_state_loaded),
            "runtime_state_build": dict(de3_runtime_state_build),
            "export_raw_context_fields": bool(de3_runtime_export_raw_context_fields),
            "v4_reports": {
                "activation_audit_path": str(de3_v4_activation_audit_path),
                "runtime_path_counters_path": str(de3_v4_runtime_path_counters_path),
                "router_summary_path": str(de3_v4_router_summary_path),
                "lane_selection_summary_path": str(de3_v4_lane_selection_summary_path),
                "bracket_summary_path": str(de3_v4_bracket_summary_path),
                "runtime_mode_summary_path": str(de3_v4_runtime_mode_summary_path),
                "execution_policy_summary_path": str(de3_v4_execution_policy_summary_path),
                "decision_side_summary_path": str(de3_v4_decision_side_summary_path),
            },
        },
        "de3_trade_management": {
            "enabled": bool(de3_v4_trade_management_cfg_local.get("enabled", False)),
            "break_even_enabled": bool(de3_v4_break_even_cfg_local.get("enabled", False)),
            "break_even_activate_on_next_bar": bool(
                de3_v4_break_even_cfg_local.get("activate_on_next_bar", True)
            ),
            "break_even_trigger_pct": float(
                _coerce_float(de3_v4_break_even_cfg_local.get("trigger_pct"), 0.0)
            ),
            "break_even_buffer_ticks": int(
                max(0, _coerce_int(de3_v4_break_even_cfg_local.get("buffer_ticks"), 0))
            ),
            "break_even_trail_pct": float(
                max(0.0, _coerce_float(de3_v4_break_even_cfg_local.get("trail_pct"), 0.0))
            ),
            "break_even_post_activation_trail_pct": float(
                max(
                    0.0,
                    _coerce_float(
                        de3_v4_break_even_cfg_local.get("post_activation_trail_pct"),
                        _coerce_float(de3_v4_break_even_cfg_local.get("trail_pct"), 0.0),
                    ),
                )
            ),
            "break_even_post_partial_trail_pct": float(
                max(
                    0.0,
                    _coerce_float(
                        de3_v4_break_even_cfg_local.get("post_partial_trail_pct"),
                        _coerce_float(
                            de3_v4_break_even_cfg_local.get("post_activation_trail_pct"),
                            _coerce_float(de3_v4_break_even_cfg_local.get("trail_pct"), 0.0),
                        ),
                    ),
                )
            ),
            "break_even_armed_trades": int(de3_break_even_armed_trade_count),
            "break_even_stop_updates": int(de3_break_even_stop_update_count),
            "profit_milestone_stop_enabled": bool(
                de3_v4_profit_milestone_cfg_local.get("enabled", False)
            ),
            "profit_milestone_stop_profile_trade_count": int(
                de3_profit_milestone_profile_trade_count
            ),
            "profit_milestone_stop_reached_count": int(
                de3_profit_milestone_reached_count
            ),
            "entry_trade_day_extreme_stop_enabled": bool(
                de3_v4_entry_trade_day_extreme_cfg_local.get("enabled", False)
            ),
            "entry_trade_day_extreme_stop_profile_trade_count": int(
                de3_entry_trade_day_extreme_profile_trade_count
            ),
            "entry_trade_day_extreme_stop_reached_count": int(
                de3_entry_trade_day_extreme_reached_count
            ),
            "entry_trade_day_extreme_admission_block_enabled": bool(
                de3_v4_entry_trade_day_extreme_admission_cfg_local.get("enabled", False)
            ),
            "entry_trade_day_extreme_admission_block_checked": int(
                de3_entry_trade_day_extreme_admission_checked
            ),
            "entry_trade_day_extreme_admission_blocked": int(
                de3_entry_trade_day_extreme_admission_blocked
            ),
            "entry_trade_day_extreme_admission_top_profiles": [
                {"name": str(name), "count": int(count)}
                for name, count in de3_entry_trade_day_extreme_admission_profile_hits.most_common(10)
            ],
            "entry_trade_day_extreme_size_adjustment_enabled": bool(
                de3_v4_entry_trade_day_extreme_size_cfg_local.get("enabled", False)
            ),
            "entry_trade_day_extreme_size_adjustment_checked": int(
                de3_entry_trade_day_extreme_size_adjustment_checked
            ),
            "entry_trade_day_extreme_size_adjustment_applied": int(
                de3_entry_trade_day_extreme_size_adjustment_applied
            ),
            "entry_trade_day_extreme_size_adjustment_top_profiles": [
                {"name": str(name), "count": int(count)}
                for name, count in de3_entry_trade_day_extreme_size_adjustment_profile_hits.most_common(10)
            ],
            "entry_trade_day_extreme_early_exit_enabled": bool(
                de3_v4_entry_trade_day_extreme_early_exit_cfg_local.get("enabled", False)
            ),
            "entry_trade_day_extreme_early_exit_profile_trade_count": int(
                de3_entry_trade_day_extreme_early_exit_profile_trade_count
            ),
            "entry_trade_day_extreme_early_exit_top_profiles": [
                {"name": str(name), "count": int(count)}
                for name, count in de3_entry_trade_day_extreme_early_exit_profile_hits.most_common(10)
            ],
            "entry_trade_day_extreme_early_exit_close_top_profiles": [
                {"name": str(name), "count": int(count)}
                for name, count in de3_entry_trade_day_extreme_early_exit_close_profile_hits.most_common(10)
            ],
            "tiered_take_enabled": bool(de3_v4_tiered_take_cfg_local.get("enabled", False)),
            "tiered_take_trigger_pct": float(
                max(0.0, _coerce_float(de3_v4_tiered_take_cfg_local.get("trigger_pct"), 0.0))
            ),
            "tiered_take_close_fraction": float(
                max(0.0, _coerce_float(de3_v4_tiered_take_cfg_local.get("close_fraction"), 0.0))
            ),
            "tiered_take_min_entry_contracts": int(
                max(2, _coerce_int(de3_v4_tiered_take_cfg_local.get("min_entry_contracts"), 2))
            ),
            "tiered_take_min_remaining_contracts": int(
                max(1, _coerce_int(de3_v4_tiered_take_cfg_local.get("min_remaining_contracts"), 1))
            ),
            "tiered_take_fill_count": int(de3_tiered_take_fill_count),
            "tiered_take_closed_contract_count": int(de3_tiered_take_closed_contract_count),
            "early_exit_enabled": bool(de3_v4_early_exit_cfg_local.get("enabled", False)),
            "early_exit_exit_if_not_green_by": int(
                max(0, _coerce_int(de3_v4_early_exit_cfg_local.get("exit_if_not_green_by"), 0))
            ),
            "early_exit_max_profit_crosses": int(
                max(0, _coerce_int(de3_v4_early_exit_cfg_local.get("max_profit_crosses"), 0))
            ),
            "early_exit_closes": int(de3_early_exit_close_count),
        },
        "regime_manifold_summary": regime_manifold_summary,
        "live_report_path": str(live_report_path) if live_report_path is not None else None,
        "de3_decisions_export": de3_decisions_export_meta,
        "de3_v2_bucket_bracket_overrides": {
            "count": int(len(de3_bucket_bracket_override_map)),
            "map": dict(de3_bucket_bracket_override_map),
        },
    }


def _parse_cli_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Run MES backtest with optional DE3 decision export instrumentation.",
        add_help=True,
    )
    parser.add_argument(
        "--export_de3_decisions",
        action="store_true",
        help="Export DE3 decision journal rows for offline analysis (v2 candidate or v3 family mode).",
    )
    parser.add_argument(
        "--de3_decisions_top_k",
        type=int,
        default=5,
        help="Top-K feasible DE3 candidates to export per decision (v3 exports all feasible families).",
    )
    parser.add_argument(
        "--de3_decisions_out",
        type=str,
        default="./reports/de3_decisions.csv",
        help=(
            "Decision journal CSV output path. "
            "If left at default de3_decisions.csv, a unique run-stamped filename is auto-generated."
        ),
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        logging.warning("Ignoring unknown CLI args: %s", " ".join(unknown))
    return args


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    cli_args = _parse_cli_args()

    base_dir = Path(__file__).resolve().parent
    path_input = input(f"Data path [{DEFAULT_CSV_NAME}]: ").strip()
    path = Path(path_input) if path_input else Path(DEFAULT_CSV_NAME)
    if not path.is_file():
        path = base_dir / path
    if not path.is_file():
        raise SystemExit(f"Data file not found: {path}")

    df = load_csv_cached(path, cache_dir=base_dir / "cache", use_cache=True)
    if df.empty:
        raise SystemExit("No rows found in the CSV.")

    symbol_mode = str(CONFIG.get("BACKTEST_SYMBOL_MODE", "single") or "single").lower()
    symbol_method = CONFIG.get("BACKTEST_SYMBOL_AUTO_METHOD", "volume")

    symbol = None
    if "symbol" in df.columns and df["symbol"].nunique(dropna=True) > 1 and symbol_mode == "single":
        preferred_symbol = CONFIG.get("TARGET_SYMBOL")
        default_symbol = choose_symbol(df, preferred_symbol)
        symbol = input(f"Symbol [{default_symbol}]: ").strip() or default_symbol
        df = df[df["symbol"] == symbol]
        if df.empty:
            raise SystemExit("No rows found for selected symbol.")

    print(f"Available range: {df.index.min()} to {df.index.max()} (NY)")
    start_raw = input("Start datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM) [min]: ").strip()
    end_raw = input("End datetime (YYYY-MM-DD or YYYY-MM-DD HH:MM) [max]: ").strip()
    start_time = parse_user_datetime(start_raw, NY_TZ, is_end=False) if start_raw else df.index.min()
    end_time = parse_user_datetime(end_raw, NY_TZ, is_end=True) if end_raw else df.index.max()
    if start_time > end_time:
        raise SystemExit("Start must be before end.")

    source_df = df[df.index <= end_time]
    range_df = source_df[(source_df.index >= start_time) & (source_df.index <= end_time)]
    if range_df.empty:
        raise SystemExit("No rows found for selected range.")

    symbol_distribution = {}
    symbol_df = source_df
    if "symbol" in range_df.columns:
        if symbol_mode != "single":
            symbol_df, auto_label, _ = apply_symbol_mode(
                source_df, symbol_mode, symbol_method
            )
            if symbol_df.empty:
                raise SystemExit("No rows found after auto symbol selection.")
            selected_range_df = symbol_df[(symbol_df.index >= start_time) & (symbol_df.index <= end_time)]
            if selected_range_df.empty:
                raise SystemExit("No rows found in selected range after auto symbol selection.")
            symbol_distribution = selected_range_df["symbol"].value_counts().to_dict()
            symbol = auto_label
        else:
            symbol_distribution = range_df["symbol"].value_counts().to_dict()
            if symbol is None:
                symbol = choose_symbol(range_df, CONFIG.get("TARGET_SYMBOL"))
            symbol_df = source_df[source_df["symbol"] == symbol]
            if symbol_df.empty:
                raise SystemExit("No rows found for selected symbol.")
            selected_range_df = symbol_df[(symbol_df.index >= start_time) & (symbol_df.index <= end_time)]
            if selected_range_df.empty:
                raise SystemExit("No rows found in selected range for selected symbol.")
    if symbol is None:
        symbol = "AUTO"

    if "symbol" in symbol_df.columns:
        symbol_df = symbol_df.drop(columns=["symbol"], errors="ignore")
    source_attrs = getattr(source_df, "attrs", {}) or {}
    symbol_df = attach_backtest_symbol_context(
        symbol_df,
        symbol,
        symbol_mode,
        source_key=source_attrs.get("source_cache_key"),
        source_label=source_attrs.get("source_label"),
        source_path=source_attrs.get("source_path"),
    )

    stats = run_backtest(
        symbol_df,
        start_time,
        end_time,
        export_de3_decisions=bool(cli_args.export_de3_decisions),
        de3_decisions_top_k=int(cli_args.de3_decisions_top_k),
        de3_decisions_out=str(cli_args.de3_decisions_out),
    )
    stats["symbol_mode"] = symbol_mode
    if symbol_distribution:
        stats["symbol_distribution"] = symbol_distribution

    print("")
    print(f"Symbol: {symbol}")
    print(f"Trades: {stats['trades']}")
    print(f"Wins: {stats['wins']}  Losses: {stats['losses']}  Winrate: {stats['winrate']:.2f}%")
    print(f"Net PnL: ${stats['equity']:.2f}")
    print(f"Largest drawdown: ${stats['max_drawdown']:.2f}")
    print(
        f"Assumptions: {CONTRACTS} contracts, ${POINT_VALUE:.2f}/point, "
        f"${FEES_PER_20_CONTRACTS:.2f} per 20 contracts (round-trip), "
        "signals on bar close and entries on next bar open."
    )
    report_path = save_backtest_report(stats, symbol, start_time, end_time)
    print(f"Report saved: {report_path}")
    monte_carlo_report_path = report_path.with_name(f"{report_path.stem}_monte_carlo.json")
    if monte_carlo_report_path.exists():
        print(f"Monte Carlo report saved: {monte_carlo_report_path}")
    baseline_report_path = report_path.with_name(f"{report_path.stem}_baseline_comparison.json")
    if baseline_report_path.exists():
        print(f"Baseline comparison saved: {baseline_report_path}")
    gemini_report_path = report_path.with_name(f"{report_path.stem}_gemini_recommendation.json")
    if gemini_report_path.exists():
        print(f"Gemini recommendation saved: {gemini_report_path}")
    de3_export_meta = stats.get("de3_decisions_export", {}) or {}
    if bool(de3_export_meta.get("enabled")):
        if de3_export_meta.get("path"):
            print(f"DE3 decisions CSV: {de3_export_meta.get('path')}")
        if de3_export_meta.get("trade_attribution_path"):
            print(f"DE3 trade attribution CSV: {de3_export_meta.get('trade_attribution_path')}")
        if de3_export_meta.get("summary_path"):
            print(f"DE3 decision summary JSON: {de3_export_meta.get('summary_path')}")


if __name__ == "__main__":
    main()
