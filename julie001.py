import pandas as pd
import numpy as np
import datetime
import base64
import csv
import json
import os
import sys
from datetime import date
import time
import logging
import math
import statistics
from zoneinfo import ZoneInfo
from datetime import timezone as dt_timezone
from typing import Any, Dict, Optional, Tuple
import random
import asyncio
from pathlib import Path
from process_singleton import acquire_singleton_lock

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _maybe_acquire_direct_run_live_lock() -> Optional[object]:
    if __name__ != "__main__":
        return None
    enforce_singleton = str(
        os.environ.get("JULIE_ENFORCE_LIVE_SINGLETON", "1")
    ).strip().lower()
    if enforce_singleton not in TRUTHY_ENV_VALUES:
        return None
    lock_path = Path(__file__).resolve().parent / "logs" / str(
        os.environ.get("JULIE_LIVE_LOCK_FILENAME", "filterless_live.lock")
        or "filterless_live.lock"
    )
    lock_name = str(
        os.environ.get("JULIE_LIVE_LOCK_NAME", "filterless_live_bot")
        or "filterless_live_bot"
    )
    lock = acquire_singleton_lock(lock_path, name=lock_name)
    if lock is not None:
        return lock

    existing = ""
    try:
        existing = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    print(f"Another live JULIE bot instance is already running. Lock: {lock_path}")
    if existing:
        print(existing)
    raise SystemExit(0)


DIRECT_RUN_LIVE_LOCK = _maybe_acquire_direct_run_live_lock()

from config import CONFIG, refresh_target_symbol, determine_current_contract_symbol
from dynamic_sltp_params import dynamic_sltp_engine, get_sltp
from volatility_filter import volatility_filter, check_volatility, VolRegime
from fixed_sltp_framework import apply_fixed_sltp, log_fixed_sltp, asia_viability_gate
from regime_strategy import RegimeAdaptiveStrategy
from structural_level_tracker import StructuralLevelTracker
from level_fill_optimizer import (
    AT_LEVEL_THRESHOLD,
    LevelFillOptimizer,
    FILL_IMMEDIATE,
    FILL_AT_LEVEL,
    FILL_WAIT,
)
from dynamic_chop import DynamicChopAnalyzer
from ml_physics_strategy import MLPhysicsStrategy
from dynamic_engine3_strategy import DynamicEngine3Strategy
from dynamic_signal_engine3 import get_signal_engine
from volume_profile import build_volume_profile
from event_logger import event_logger
from circuit_breaker import CircuitBreaker
from news_filter import NewsFilter
from client import ProjectXClient
from kalshi_trade_overlay import (
    analyze_recent_price_action as analyze_kalshi_recent_price_action,
    build_trade_plan as build_kalshi_trade_plan,
    compute_tp_trail_stop as compute_kalshi_tp_trail_stop,
)
from pct_level_overlay import PctLevelOverlay, DEFAULT_OVERLAY_CONFIG as _PCT_LEVEL_OVERLAY_DEFAULTS
from pct_overlay_runtime import (
    get_pct_level_overlay as _get_pct_level_overlay,
    init_pct_level_overlay as _init_pct_level_overlay,
    update_pct_level_overlay as _update_pct_level_overlay,
)
from regime_classifier import (
    init_regime_classifier as _init_regime_classifier,
    update_regime_classifier as _update_regime_classifier,
    apply_regime_size_cap as _apply_regime_size_cap,
)
from loss_factor_guard import (
    init_guard as _init_loss_factor_guard,
    notify_trend_day as _lfg_notify_trend_day,
    notify_bar as _lfg_notify_bar,
    notify_trade_closed as _lfg_notify_trade_closed,
    should_veto_entry as _lfg_should_veto_entry,
)
from signal_gate_2025 import (
    init_gate as _init_signal_gate_2025,
    log_shadow_prediction as _signal_gate_shadow_log,
)


def _signal_birth_hook(signal):
    """Called at every signal-birth site. Does two things:
      1. Apply the regime size-cap (filter D), if enabled.
      2. Score the signal with filter G in shadow mode and log the prediction
         — runs on every signal regardless of what upstream filters (Kalshi,
         FILTER_CHECK) will do, so we accumulate calibration data for G even
         when Kalshi short-circuits first.
    Never raises — all failures silently no-op."""
    try:
        _apply_regime_size_cap(signal)
    except Exception:
        pass
    try:
        _signal_gate_shadow_log(signal)
    except Exception:
        pass


from pct_overlay_runtime import (
    attach_pct_overlay_snapshot as _attach_pct_overlay_snapshot,
    resolve_pct_overlay_snapshot as _resolve_pct_overlay_snapshot,
    apply_pct_level_overlay_to_signal as _apply_pct_level_overlay_to_signal,
)

import param_scaler
from bot_state import (
    STATE_PATH,
    STATE_VERSION,
    load_bot_state,
    normalize_sentiment_state,
    parse_dt,
    save_bot_state,
    trading_day_start,
)
from regime_manifold_engine import get_kalshi_gate_decision
from services.sentiment_service import (
    build_truth_social_sentiment_service,
    get_sentiment_state,
    set_sentiment_state,
)

# --- ASYNCIO IMPORTS ---
try:
    from async_market_stream import AsyncMarketDataManager
except Exception:
    AsyncMarketDataManager = None
from async_tasks import (
    heartbeat_task,
    htf_structure_task,
    kalshi_refresh_task,
    position_sync_task,
    sentiment_monitor_task,
)


# ---------------------------------------------------------------------------
# Kalshi provider — lazy singleton for trade-gating during settlement hours.
# Built on first access; returns None if credentials are missing/disabled.
# ---------------------------------------------------------------------------
_KALSHI_PROVIDER: Optional[Any] = None
_KALSHI_PROVIDER_INIT_DONE = False
# Settlement hours (ET) when trade gating is active with 3x sizing
# All settlement hours for data collection / dashboard display
_KALSHI_SETTLEMENT_HOURS_ET = [10, 11, 12, 13, 14, 15, 16]
# Hours where Kalshi crowd has directional edge (backtest: 70% at 60%+ conf)
# 10-11 AM excluded: crowd is contrarian (39.5% accuracy)
_KALSHI_GATING_HOURS_ET = [12, 13, 14, 15, 16]

# ---------------------------------------------------------------------------
# MLPhysics adaptive bank fill — US session only (Macros 13-19)
# Backtest: +0.508 pts/trade delta on reversal signals, consistent across
# all 60 quarters 2011-2026. Only active during US session macro trigger bars.
#
# US session macros = Macro_13..Macro_19 = PT hours 6-12 = ET hours 9-15
# ---------------------------------------------------------------------------
_BANK_FILL_US_SESSION_HOURS_ET: frozenset[int] = frozenset({9, 10, 11, 12, 13, 14, 15})
_BANK_FILL_MACRO_TRIGGER_MINUTE: int = 50   # macro windows fire at :50
_BANK_FILL_WINDOW_BARS: int = 30            # max bars to wait for bank fill
_BANK_FILL_STRONG_RANGE_ATR: float = 2.5   # bar range / ATR >= this → strong
_BANK_FILL_STRONG_SPAN_MIN: int = 3        # bank levels spanned >= this → strong
_BANK_FILL_STEP: float = 12.5              # ES bank level increment in points
_BANK_FILL_TREND_LOOKBACK: int = 30        # bars of lookback for reversal detection
# Strategies (besides MLPhysics) that participate in adaptive bank fill
_BANK_FILL_ELIGIBLE_STRATEGIES: frozenset[str] = frozenset({
    "DynamicEngine3Strategy", "AetherFlowStrategy", "RegimeAdaptiveStrategy",
})

# --- Pivot-level trailing stop ---
# When a confirmed swing pivot is detected during a US-session trade, the hard
# stop is ratcheted to just behind the highest/lowest bank level the pivot
# reached.  The ratchet is one-way (SL only moves in the favorable direction).
_PIVOT_TRAIL_LOOKBACK: int = 5        # bars in the swing-detection window
_PIVOT_TRAIL_BUFFER: float = 0.25     # 1 tick of clearance beyond the anchor level
_PIVOT_TRAIL_MIN_PROFIT_PTS: float = 12.5  # minimum unrealised PnL before trail arms
# Short-form strategy labels as stored on live trade dicts (vs the ``*Strategy``
# class names used by _BANK_FILL_ELIGIBLE_STRATEGIES above).
_PIVOT_TRAIL_ELIGIBLE_STRATEGIES: frozenset[str] = frozenset({
    "DynamicEngine3", "AetherFlow", "RegimeAdaptive",
})


def _active_kalshi_settlement_hour_et(kalshi: Optional[Any]) -> Optional[int]:
    if kalshi is None:
        return None
    try:
        return kalshi.active_settlement_hour_et()
    except Exception:
        return None


def _get_kalshi_provider():
    global _KALSHI_PROVIDER, _KALSHI_PROVIDER_INIT_DONE
    if _KALSHI_PROVIDER_INIT_DONE:
        return _KALSHI_PROVIDER
    _KALSHI_PROVIDER_INIT_DONE = True
    try:
        from config_secrets import SECRETS
        from services.kalshi_provider import KalshiProvider

        kalshi_cfg = CONFIG.get("KALSHI", {}) if isinstance(CONFIG, dict) else {}
        if not isinstance(kalshi_cfg, dict):
            return None
        provider_cfg = dict(kalshi_cfg)
        provider_cfg["key_id"] = str(SECRETS.get("KALSHI_KEY_ID", provider_cfg.get("key_id", "")) or "")
        provider_cfg["private_key_path"] = str(
            SECRETS.get("KALSHI_PRIVATE_KEY_PATH", provider_cfg.get("private_key_path", "")) or ""
        )
        _KALSHI_PROVIDER = KalshiProvider(provider_cfg)
    except Exception as exc:
        logging.warning("Kalshi provider not available for trade gating: %s", exc)
        _KALSHI_PROVIDER = None
    return _KALSHI_PROVIDER


def _mlphysics_is_us_macro_bar(current_time: datetime.datetime) -> bool:
    """True when current_time is a US session macro trigger bar (:50 minute, ET 9-15)."""
    try:
        ct_et = (
            current_time.astimezone(NY_TZ)
            if current_time.tzinfo is not None
            else current_time.replace(tzinfo=NY_TZ)
        )
        return ct_et.hour in _BANK_FILL_US_SESSION_HOURS_ET and ct_et.minute == _BANK_FILL_MACRO_TRIGGER_MINUTE
    except Exception:
        return False


def _mlphysics_bank_break_is_strong(df: "pd.DataFrame") -> bool:
    """True if the current macro bar is a STRONG break (fill immediately).
    Strong = bar spanned >= 3 bank levels OR bar range >= 2.5x ATR.
    Defaults to True on any error so we never incorrectly park a signal.
    """
    try:
        bar = df.iloc[-1]
        bar_range = float(bar["high"]) - float(bar["low"])
        # Bank span count
        if bar_range / _BANK_FILL_STEP >= _BANK_FILL_STRONG_SPAN_MIN:
            return True
        # ATR over last 14 bars
        if len(df) >= 15:
            trs = []
            for i in range(-14, 0):
                h = float(df.iloc[i]["high"])
                l = float(df.iloc[i]["low"])
                cp = float(df.iloc[i - 1]["close"])
                trs.append(max(h - l, abs(h - cp), abs(l - cp)))
            atr = sum(trs) / len(trs) if trs else bar_range
        else:
            atr = bar_range
        return (bar_range / atr) >= _BANK_FILL_STRONG_RANGE_ATR if atr > 0 else True
    except Exception:
        return True  # safe default — never wrongly park a signal


def _mlphysics_is_reversal(signal: dict, df: "pd.DataFrame") -> bool:
    """True if signal direction is fading intraday momentum (reversal trade).
    Uses a 30-bar price change as the trend proxy.
    Breakout = trading WITH momentum; Reversal = fading it.
    """
    try:
        side = str(signal.get("side", "") or "").upper()
        if side not in ("LONG", "SHORT"):
            return False
        lookback = min(_BANK_FILL_TREND_LOOKBACK, len(df) - 1)
        if lookback < 5:
            return False
        price_now = float(df.iloc[-1]["close"])
        price_then = float(df.iloc[-lookback]["close"])
        trend_up = price_now > price_then
        # Reversal: going against the prevailing trend
        return (side == "LONG" and not trend_up) or (side == "SHORT" and trend_up)
    except Exception:
        return False


def _mlphysics_bank_fill_target(side: str, price: float) -> float:
    """Nearest 12.5-pt bank level for a limit fill.
    LONG → bank at or below price (buy the pullback).
    SHORT → bank at or above price (sell the pump).
    """
    step = _BANK_FILL_STEP
    if side == "LONG":
        return math.floor(price / step) * step
    return math.ceil(price / step) * step


def _detect_pivot_high(df: "pd.DataFrame", lookback: int = 5) -> Optional[float]:
    """Return the confirmed swing-high price from ~lookback//2 bars ago, or None.

    A swing high is confirmed when the bar at position [-lookback//2] (within
    the last `lookback` bars) has the highest high of the entire window.  Using
    the middle bar means the pivot is confirmed by at least one subsequent lower
    bar, giving a reasonable lag vs. responsiveness trade-off on 5-min data.
    """
    half = lookback // 2
    if len(df) < lookback:
        return None
    try:
        highs = df["high"].iloc[-lookback:].values.astype(float)
        ph = highs[half]
        if all(ph >= highs[i] - 1e-9 for i in range(len(highs)) if i != half):
            return float(ph)
    except Exception:
        pass
    return None


def _detect_pivot_low(df: "pd.DataFrame", lookback: int = 5) -> Optional[float]:
    """Return the confirmed swing-low price from ~lookback//2 bars ago, or None."""
    half = lookback // 2
    if len(df) < lookback:
        return None
    try:
        lows = df["low"].iloc[-lookback:].values.astype(float)
        pl = lows[half]
        if all(pl <= lows[i] + 1e-9 for i in range(len(lows)) if i != half):
            return float(pl)
    except Exception:
        pass
    return None


def _compute_pivot_trail_sl(
    side: str,
    pivot_price: float,
    entry_price: float,
    current_sl: float,
    min_profit_pts: float = _PIVOT_TRAIL_MIN_PROFIT_PTS,
    step: float = _BANK_FILL_STEP,
    buffer: float = _PIVOT_TRAIL_BUFFER,
) -> Optional[float]:
    """Compute a trailing SL candidate triggered by a confirmed pivot.

    Two-tier anchor selection (LONG — symmetric for SHORT):

    Reading B (default — one level of room):
        anchor = highest bank level at-or-below pivot  MINUS one step
        SL     = anchor - buffer
        Gives the trade one full bank-width of breathing room after the pivot.
        Example: pivot 5248, anchor_C = 5237.5, anchor_B = 5225 → SL 5224.75

    Fallback to Reading C (when B would be at or below entry):
        anchor = highest bank level at-or-below pivot  (no step back)
        SL     = anchor - buffer
        Used when the "one step back" level doesn't lock any profit.
        Example: pivot 5212.5, entry 5200 → B would be 5199.75 (loss) → use C → SL 5212.25

    Ratchet: candidate only returned if it IMPROVES the current SL and locks
    genuine profit above (LONG) / below (SHORT) entry.
    """
    side = str(side).upper()
    if side == "LONG":
        profit_pts = pivot_price - entry_price
        if profit_pts < min_profit_pts - 1e-9:
            return None
        # Pivot-level anchor (Reading C)
        l_anchor_c = math.floor(pivot_price / step) * step
        # One level back toward entry (Reading B)
        l_anchor_b = l_anchor_c - step
        candidate_b = round(l_anchor_b - buffer, 4)
        if candidate_b > entry_price + 1e-9:
            candidate = candidate_b   # Reading B — one level of room
        else:
            candidate = round(l_anchor_c - buffer, 4)  # fallback to Reading C
        # Ratchet: only move SL up, and only if it locks real profit
        if candidate <= current_sl + 1e-9 or candidate <= entry_price + 1e-9:
            return None
        return candidate
    elif side == "SHORT":
        profit_pts = entry_price - pivot_price
        if profit_pts < min_profit_pts - 1e-9:
            return None
        # Pivot-level anchor (Reading C)
        l_anchor_c = math.ceil(pivot_price / step) * step
        # One level back toward entry (Reading B)
        l_anchor_b = l_anchor_c + step
        candidate_b = round(l_anchor_b + buffer, 4)
        if candidate_b < entry_price - 1e-9:
            candidate = candidate_b   # Reading B — one level of room
        else:
            candidate = round(l_anchor_c + buffer, 4)  # fallback to Reading C
        # Ratchet: only move SL down, and only if it locks real profit
        if candidate >= current_sl - 1e-9 or candidate >= entry_price - 1e-9:
            return None
        return candidate
    return None


def _live_pivot_trail_candidate(
    trade: Optional[dict],
    *,
    pivot_high: Optional[float],
    pivot_low: Optional[float],
    pivot_bar_index: Optional[int],
) -> Optional[float]:
    if not isinstance(trade, dict):
        return None
    side_name = str(trade.get("side", "") or "").upper()
    entry_price = _coerce_float(trade.get("entry_price"), math.nan)
    current_stop_price = _coerce_float(trade.get("current_stop_price"), math.nan)
    if (
        side_name not in {"LONG", "SHORT"}
        or not math.isfinite(entry_price)
        or not math.isfinite(current_stop_price)
    ):
        return None

    entry_bar_index = _coerce_int(trade.get("entry_bar"), None)
    if pivot_bar_index is None:
        return None
    if entry_bar_index is not None and int(pivot_bar_index) <= int(entry_bar_index):
        return None

    if side_name == "LONG" and pivot_high is not None:
        return _compute_pivot_trail_sl(
            "LONG",
            float(pivot_high),
            float(entry_price),
            float(current_stop_price),
        )
    if side_name == "SHORT" and pivot_low is not None:
        return _compute_pivot_trail_sl(
            "SHORT",
            float(pivot_low),
            float(entry_price),
            float(current_stop_price),
        )
    return None


def _level_fill_live_execution_allowed(
    entry: Optional[dict],
    current_price: float,
    *,
    tolerance: float = AT_LEVEL_THRESHOLD,
) -> Tuple[bool, Optional[float]]:
    if not isinstance(entry, dict):
        return False, None
    target_price = _coerce_float(entry.get("target_price"), math.nan)
    live_price = _coerce_float(current_price, math.nan)
    if not math.isfinite(target_price) or not math.isfinite(live_price):
        return True, None
    market_distance = abs(float(live_price) - float(target_price))
    return market_distance <= float(tolerance) + 1e-9, float(market_distance)


def _truth_social_cfg() -> Dict[str, Any]:
    cfg = CONFIG.get("TRUTH_SOCIAL_SENTIMENT", {}) if isinstance(CONFIG, dict) else {}
    return dict(cfg) if isinstance(cfg, dict) else {}


def _truth_social_signal_enabled() -> bool:
    return bool(_truth_social_cfg().get("enabled", False))


def _sentiment_snapshot_age_seconds(snapshot: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(snapshot, dict):
        return None
    analyzed_at = parse_dt(snapshot.get("last_analysis_at"))
    if not isinstance(analyzed_at, datetime.datetime):
        return None
    if analyzed_at.tzinfo is None:
        analyzed_at = analyzed_at.replace(tzinfo=dt_timezone.utc)
    now_utc = datetime.datetime.now(dt_timezone.utc)
    return max(0.0, (now_utc - analyzed_at.astimezone(dt_timezone.utc)).total_seconds())


def _evaluate_truth_social_emergency_exit(snapshot: Optional[Dict[str, Any]], side: Optional[str]) -> Optional[str]:
    normalized_side = str(side or "").strip().upper()
    if normalized_side not in {"LONG", "SHORT"}:
        return None
    if not isinstance(snapshot, dict):
        return None
    if not bool(snapshot.get("enabled")) or not bool(snapshot.get("healthy")):
        return None

    cfg = _truth_social_cfg()
    try:
        threshold = float(cfg.get("emergency_exit_threshold", -0.75) or -0.75)
    except Exception:
        threshold = -0.75
    max_age_seconds = max(
        60,
        int(cfg.get("emergency_exit_max_age_seconds", 3600) or 3600),
    )
    age_seconds = _sentiment_snapshot_age_seconds(snapshot)
    if age_seconds is None or age_seconds > float(max_age_seconds):
        return None

    score = _coerce_float(snapshot.get("sentiment_score"), math.nan)
    confidence = _coerce_float(snapshot.get("finbert_confidence"), math.nan)
    if not math.isfinite(score):
        return None

    if normalized_side == "LONG" and score <= threshold:
        return (
            f"SENTIMENT EMERGENCY EXIT: sentiment {score:.2f} <= {threshold:.2f}"
            + (f" (FinBERT {confidence:.2%})" if math.isfinite(confidence) else "")
        )
    short_threshold = abs(float(threshold))
    if normalized_side == "SHORT" and score >= short_threshold:
        return (
            f"SENTIMENT EMERGENCY EXIT: sentiment {score:.2f} >= {short_threshold:.2f}"
            + (f" (FinBERT {confidence:.2%})" if math.isfinite(confidence) else "")
        )
    return None


def _load_non_filterless_runtime() -> tuple[Any, ...]:
    from orb_strategy import OrbStrategy
    from intraday_dip_strategy import IntradayDipStrategy
    from confluence_strategy import ConfluenceStrategy
    from smt_strategy import SMTStrategy
    from ict_model_strategy import ICTModelStrategy
    from impulse_breakout_strategy import ImpulseBreakoutStrategy
    from auction_reversion_strategy import AuctionReversionStrategy
    from liquidity_sweep_strategy import LiquiditySweepStrategy
    from value_area_breakout_strategy import ValueAreaBreakoutStrategy
    from smooth_trend_asia_strategy import SmoothTrendAsiaStrategy
    from dynamic_engine_strategy import DynamicEngineStrategy
    from vixmeanreversion import VIXReversionStrategy
    from yahoo_vix_client import YahooVIXClient
    return (
        OrbStrategy,
        IntradayDipStrategy,
        ConfluenceStrategy,
        SMTStrategy,
        ICTModelStrategy,
        ImpulseBreakoutStrategy,
        AuctionReversionStrategy,
        LiquiditySweepStrategy,
        ValueAreaBreakoutStrategy,
        SmoothTrendAsiaStrategy,
        DynamicEngineStrategy,
        VIXReversionStrategy,
        YahooVIXClient,
    )


class _NoOpStatefulRuntime:
    def __init__(self, *args, **kwargs):
        pass

    def load_state(self, *_args, **_kwargs):
        return None

    def get_state(self):
        return {}

    def update(self, *_args, **_kwargs):
        return None

    def backfill(self, *_args, **_kwargs):
        return None

    def should_block_trade(self, *_args, **_kwargs):
        return False, None


class _NoOpRejectionFilter(_NoOpStatefulRuntime):
    prev_day_pm_bias = "NEUTRAL"


class _NoOpTrendFilter(_NoOpStatefulRuntime):
    def update_dynamic_params(self, *_args, **_kwargs):
        return None


class _NoOpDirectionalLossBlocker(_NoOpStatefulRuntime):
    def __init__(self, consecutive_loss_limit: int = 3, block_minutes: int = 15):
        self.consecutive_loss_limit = int(consecutive_loss_limit)
        self.block_minutes = int(block_minutes)
        self.long_consecutive_losses = 0
        self.short_consecutive_losses = 0
        self.long_blocked_until = None
        self.short_blocked_until = None

    def record_trade_result(self, *_args, **_kwargs):
        return None

    def update_quarter(self, *_args, **_kwargs):
        return None


class _NoOpImpulseFilter(_NoOpStatefulRuntime):
    def __init__(self, *args, **kwargs):
        self.wick_ratio_threshold = 0.0
        self.last_candle_high = 0.0
        self.last_candle_open = 0.0
        self.last_candle_close = 0.0
        self.last_candle_low = 0.0
        self.last_candle_body = 0.0
        self.last_candle_dir = ""

    def get_impulse_stats(self):
        return False, 0.0, {}


class _NoOpHTFFVGFilter(_NoOpStatefulRuntime):
    def __init__(self, *args, **kwargs):
        self.memory = []

    def check_signal_blocked(self, *_args, **_kwargs):
        return False, None


def _load_strategy_filter_runtime() -> tuple[Any, ...]:
    from rejection_filter import RejectionFilter
    from chop_filter import ChopFilter
    from extension_filter import ExtensionFilter
    from trend_filter import TrendFilter
    from dynamic_structure_blocker import (
        DynamicStructureBlocker,
        RegimeStructureBlocker,
        PenaltyBoxBlocker,
    )
    from directional_loss_blocker import DirectionalLossBlocker
    from impulse_filter import ImpulseFilter
    from htf_fvg_filter import HTFFVGFilter

    return (
        RejectionFilter,
        ChopFilter,
        ExtensionFilter,
        TrendFilter,
        DynamicStructureBlocker,
        RegimeStructureBlocker,
        PenaltyBoxBlocker,
        DirectionalLossBlocker,
        ImpulseFilter,
        HTFFVGFilter,
    )


def _load_continuation_strategy_runtime() -> tuple[Any, dict]:
    from continuation_strategy import FractalSweepStrategy, STRATEGY_CONFIGS

    return FractalSweepStrategy, STRATEGY_CONFIGS


def _load_filter_stack_runtime() -> tuple[Any, ...]:
    from bank_level_quarter_filter import BankLevelQuarterFilter
    from memory_sr_filter import MemorySRFilter
    from legacy_filters import LegacyFilterSystem
    from filter_arbitrator import FilterArbitrator
    from regime_manifold_engine import RegimeManifoldEngine, apply_meta_policy
    from strategy_gate_policy import evaluate_pre_signal_gate

    return (
        BankLevelQuarterFilter,
        MemorySRFilter,
        LegacyFilterSystem,
        FilterArbitrator,
        RegimeManifoldEngine,
        apply_meta_policy,
        evaluate_pre_signal_gate,
    )


def _load_gemini_runtime() -> Any:
    from gemini_optimizer import GeminiSessionOptimizer

    return GeminiSessionOptimizer


def _load_manifold_strategy_runtime() -> Any:
    from manifold_strategy import ManifoldStrategy

    return ManifoldStrategy


def _load_aetherflow_strategy_runtime() -> Any:
    from aetherflow_strategy import AetherFlowStrategy

    return AetherFlowStrategy


def _compute_session_quarter(hour: int, minute: int, base_session: str) -> int:
    session_key = str(base_session or "").upper()
    session_minutes = {
        "ASIA": 9 * 60,
        "LONDON": 5 * 60,
        "NY_AM": 4 * 60,
        "NY_PM": 5 * 60,
    }
    session_starts = {
        "ASIA": 18 * 60,
        "LONDON": 3 * 60,
        "NY_AM": 8 * 60,
        "NY_PM": 12 * 60,
    }
    total_minutes = session_minutes.get(session_key)
    start_minutes = session_starts.get(session_key)
    if total_minutes is None or start_minutes is None:
        return 1
    minute_of_day = (int(hour) * 60 + int(minute)) % (24 * 60)
    elapsed = (minute_of_day - start_minutes) % (24 * 60)
    quarter_span = max(1, int(math.ceil(total_minutes / 4.0)))
    return min(4, max(1, (elapsed // quarter_span) + 1))

# ==========================================
# RESAMPLER HELPER FUNCTION
# ==========================================
def resample_dataframe(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """
    Resamples 1-minute OHLCV data into higher timeframes (5m, 15m, 60m).
    """
    if df.empty:
        return pd.DataFrame()

    # Define aggregation rules
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    # Resample using the timeframe string (e.g., '5min' for 5 minutes)
    tf_code = f"{timeframe_minutes}min"
    resampled_df = df.resample(tf_code).agg(agg_dict).dropna()

    return resampled_df


def _validate_signal_sltp(signal: dict, strat_name: str) -> Optional[tuple[float, float]]:
    sl_raw = signal.get("sl_dist")
    tp_raw = signal.get("tp_dist")
    missing = []
    if sl_raw is None:
        missing.append("sl_dist")
    if tp_raw is None:
        missing.append("tp_dist")
    if missing:
        msg = f"⚠️ {strat_name} missing {', '.join(missing)}; skipping trade"
        logging.warning(msg)
        event_logger.log_error("MISSING_SLTP", msg)
        return None
    try:
        sl_val = float(sl_raw)
        tp_val = float(tp_raw)
    except (TypeError, ValueError) as exc:
        msg = f"⚠️ {strat_name} invalid sl/tp values (sl={sl_raw}, tp={tp_raw}); skipping trade"
        logging.warning(msg)
        event_logger.log_error("INVALID_SLTP", msg, exception=exc)
        return None
    if not math.isfinite(sl_val) or not math.isfinite(tp_val) or sl_val <= 0 or tp_val <= 0:
        msg = f"⚠️ {strat_name} non-positive sl/tp values (sl={sl_val}, tp={tp_val}); skipping trade"
        logging.warning(msg)
        event_logger.log_error("INVALID_SLTP", msg)
        return None
    signal["sl_dist"] = sl_val
    signal["tp_dist"] = tp_val
    return sl_val, tp_val


def _signal_base_size(signal: Optional[dict], fallback: int = 5) -> int:
    if not isinstance(signal, dict):
        return int(fallback)
    try:
        size = int(signal.get("size", fallback) or fallback)
    except Exception:
        size = int(fallback)
    return max(1, size)


def _coerce_float(value, fallback: Optional[float]) -> Optional[float]:
    if value is None:
        return None if fallback is None else float(fallback)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "nan"}:
            return None if fallback is None else float(fallback)
    try:
        out = float(value)
    except Exception:
        if fallback is None:
            return None
        return float(fallback)
    if not math.isfinite(out):
        if fallback is None:
            return None
        return float(fallback)
    return float(out)


def _coerce_int(value, fallback: Optional[int]) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        if fallback is None:
            return None
        return int(fallback)


def _is_de3_v4_signal_for_sizing(signal: Optional[dict]) -> bool:
    if not isinstance(signal, dict):
        return False
    de3_ver = str(signal.get("de3_version", "") or "").strip().lower()
    if de3_ver == "v4":
        return True
    if signal.get("de3_v4_selected_variant_id") or signal.get("de3_v4_selected_lane"):
        return True
    return False


def _is_regimeadaptive_signal_for_sizing(signal: Optional[dict]) -> bool:
    if not isinstance(signal, dict):
        return False
    strategy_name = str(signal.get("strategy", "") or "").strip().lower()
    return strategy_name.startswith("regimeadaptive")


def _apply_de3_v4_confidence_tier_size_live(signal: Optional[dict], base_size: int) -> int:
    if not isinstance(signal, dict):
        return int(base_size)
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
    fields = [
        str(v).strip()
        for v in raw_fields
        if str(v).strip()
    ] if isinstance(raw_fields, (list, tuple, set)) else []
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


def _normalize_live_drawdown_state(state: Optional[dict]) -> dict:
    normalized = {
        "account_id": None,
        "starting_balance": None,
        "balance": None,
        "peak_balance": None,
        "realized_pnl": 0.0,
        "peak_realized_pnl": 0.0,
        "current_drawdown_usd": 0.0,
        "source": "trade_pnl_fallback",
        "last_balance_update": None,
        "last_trade_close": None,
        "balance_pending_refresh": False,
    }
    if not isinstance(state, dict):
        return normalized
    raw_account_id = state.get("account_id")
    try:
        parsed_account_id = int(raw_account_id)
    except Exception:
        parsed_account_id = None
    normalized["account_id"] = parsed_account_id
    for key in ("starting_balance", "balance", "peak_balance", "realized_pnl", "peak_realized_pnl", "current_drawdown_usd"):
        raw_value = state.get(key)
        try:
            parsed = float(raw_value)
        except Exception:
            parsed = None
        if parsed is not None and math.isfinite(parsed):
            normalized[key] = float(parsed)
    for key in ("source", "last_balance_update", "last_trade_close"):
        value = state.get(key)
        if value is not None:
            normalized[key] = str(value)
    normalized["balance_pending_refresh"] = bool(state.get("balance_pending_refresh", False))
    return normalized


def _current_live_drawdown_metrics(state: Optional[dict]) -> dict:
    normalized = _normalize_live_drawdown_state(state)
    balance = normalized.get("balance")
    peak_balance = normalized.get("peak_balance")
    realized_pnl = float(normalized.get("realized_pnl", 0.0) or 0.0)
    peak_realized_pnl = float(normalized.get("peak_realized_pnl", realized_pnl) or realized_pnl)

    use_account_balance = (
        balance is not None
        and peak_balance is not None
        and not bool(normalized.get("balance_pending_refresh", False))
    )
    if use_account_balance:
        current_dd = max(0.0, float(peak_balance) - float(balance))
        source = "account_balance"
    else:
        current_dd = max(0.0, float(peak_realized_pnl) - float(realized_pnl))
        source = "trade_pnl_fallback"

    normalized["current_drawdown_usd"] = float(current_dd)
    normalized["source"] = source
    return normalized


def _sync_live_drawdown_from_account_state(
    state: Optional[dict],
    account_info: Optional[dict],
    *,
    event_time: Optional[datetime.datetime] = None,
) -> bool:
    if not isinstance(state, dict) or not isinstance(account_info, dict):
        return False
    account_id = _coerce_int(account_info.get("id"), 0)
    balance = _coerce_float(account_info.get("balance"), float("nan"))
    if not math.isfinite(balance):
        return False

    existing_account_id = state.get("account_id")
    if existing_account_id is not None and int(existing_account_id) != int(account_id):
        state["starting_balance"] = None
        state["balance"] = None
        state["peak_balance"] = None
        state["realized_pnl"] = 0.0
        state["peak_realized_pnl"] = 0.0
        state["current_drawdown_usd"] = 0.0
        state["last_balance_update"] = None
        state["last_trade_close"] = None
        state["balance_pending_refresh"] = False

    starting_balance = state.get("starting_balance")
    if starting_balance is None or not math.isfinite(float(starting_balance)):
        starting_balance = float(balance)
    peak_balance = state.get("peak_balance")
    if peak_balance is None or not math.isfinite(float(peak_balance)):
        peak_balance = float(balance)
    peak_balance = max(float(peak_balance), float(balance))

    realized_pnl = float(balance) - float(starting_balance)
    peak_realized = state.get("peak_realized_pnl")
    if peak_realized is None or not math.isfinite(float(peak_realized)):
        peak_realized = float(realized_pnl)
    peak_realized = max(float(peak_realized), float(realized_pnl))

    state["account_id"] = int(account_id)
    state["starting_balance"] = float(starting_balance)
    state["balance"] = float(balance)
    state["peak_balance"] = float(peak_balance)
    state["realized_pnl"] = float(realized_pnl)
    state["peak_realized_pnl"] = float(peak_realized)
    state["current_drawdown_usd"] = float(max(0.0, peak_balance - balance))
    state["source"] = "account_balance"
    state["balance_pending_refresh"] = False
    if isinstance(event_time, datetime.datetime):
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=datetime.timezone.utc)
        state["last_balance_update"] = event_time.astimezone(datetime.timezone.utc).isoformat()
    return True


def _record_live_realized_pnl(
    state: Optional[dict],
    pnl_dollars: float,
    *,
    close_time: Optional[datetime.datetime] = None,
) -> None:
    if not isinstance(state, dict):
        return
    realized_pnl = float(state.get("realized_pnl", 0.0) or 0.0) + float(pnl_dollars)
    peak_realized = float(state.get("peak_realized_pnl", realized_pnl) or realized_pnl)
    peak_realized = max(float(peak_realized), float(realized_pnl))
    state["realized_pnl"] = float(realized_pnl)
    state["peak_realized_pnl"] = float(peak_realized)
    state["current_drawdown_usd"] = float(max(0.0, peak_realized - realized_pnl))
    if state.get("balance") is not None:
        state["balance_pending_refresh"] = True
    else:
        state["source"] = "trade_pnl_fallback"
    if isinstance(close_time, datetime.datetime):
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=datetime.timezone.utc)
        state["last_trade_close"] = close_time.astimezone(datetime.timezone.utc).isoformat()


def _refresh_live_drawdown_from_client(
    client: Optional[ProjectXClient],
    state: Optional[dict],
    *,
    event_time: Optional[datetime.datetime] = None,
    force_refresh: bool = False,
) -> bool:
    if client is None or not isinstance(state, dict):
        return False
    account_info = client.get_account_info(force_refresh=force_refresh)
    return _sync_live_drawdown_from_account_state(state, account_info, event_time=event_time)


def _apply_live_drawdown_size(
    signal: Optional[dict],
    requested_size: int,
    live_drawdown_state: Optional[dict],
) -> int:
    if not isinstance(signal, dict):
        return int(requested_size)

    exec_cfg = CONFIG.get("BACKTEST_EXECUTION", {}) or {}
    enabled = bool(exec_cfg.get("drawdown_size_scaling_enabled", False))
    start_usd = max(0.0, _coerce_float(exec_cfg.get("drawdown_size_scaling_start_usd", 0.0), 0.0))
    max_usd = max(start_usd, _coerce_float(exec_cfg.get("drawdown_size_scaling_max_usd", 2000.0), 2000.0))
    configured_base_contracts = max(
        1,
        _coerce_int(exec_cfg.get("drawdown_size_scaling_base_contracts", requested_size), requested_size),
    )
    # Requested size can already include strategy-specific growth scaling.
    base_contracts = max(1, int(max(requested_size, configured_base_contracts)))
    min_contracts = max(1, _coerce_int(exec_cfg.get("drawdown_size_scaling_min_contracts", 1), 1))
    if min_contracts > base_contracts:
        min_contracts = int(base_contracts)

    metrics = _current_live_drawdown_metrics(live_drawdown_state)
    current_realized_dd = max(0.0, float(metrics.get("current_drawdown_usd", 0.0) or 0.0))
    signal["drawdown_size_source"] = str(metrics.get("source", "trade_pnl_fallback") or "trade_pnl_fallback")
    signal["drawdown_size_live_pending_balance_refresh"] = bool(
        metrics.get("balance_pending_refresh", False)
    )

    if not enabled or max_usd <= start_usd or base_contracts <= min_contracts:
        signal["drawdown_size_scaling_enabled"] = False
        signal["drawdown_size_applied"] = False
        signal["drawdown_size_step_usd"] = 0.0
        return int(requested_size)

    contract_range = max(1, int(base_contracts) - int(min_contracts))
    span_usd = max(1e-9, float(max_usd) - float(start_usd))
    step_usd = span_usd / float(contract_range)

    if current_realized_dd <= start_usd:
        dd_progress = 0.0
        drawdown_size_cap = int(base_contracts)
    elif current_realized_dd >= max_usd:
        dd_progress = 1.0
        drawdown_size_cap = int(min_contracts)
    else:
        dd_above = current_realized_dd - start_usd
        dd_progress = min(1.0, dd_above / span_usd)
        bucket = int(dd_above / step_usd)
        if bucket < 0:
            bucket = 0
        elif bucket > contract_range:
            bucket = contract_range
        drawdown_size_cap = int(base_contracts - bucket)
        if drawdown_size_cap < min_contracts:
            drawdown_size_cap = int(min_contracts)

    size = int(requested_size)
    requested = int(size)
    if size > drawdown_size_cap:
        size = int(drawdown_size_cap)
    if size < min_contracts:
        size = int(min_contracts)
    if size > base_contracts:
        size = int(base_contracts)

    signal["drawdown_size_scaling_enabled"] = True
    signal["drawdown_size_realized_dd_usd"] = float(current_realized_dd)
    signal["drawdown_size_progress"] = float(dd_progress)
    signal["drawdown_size_cap"] = int(drawdown_size_cap)
    signal["drawdown_size_requested"] = int(requested)
    signal["drawdown_size_applied"] = bool(size < requested)
    signal["drawdown_size_step_usd"] = float(step_usd)
    return int(size)


def _apply_regimeadaptive_live_growth_size(
    signal: Optional[dict],
    base_size: int,
    live_drawdown_state: Optional[dict],
) -> int:
    if not _is_regimeadaptive_signal_for_sizing(signal):
        return int(base_size)

    exec_cfg = CONFIG.get("BACKTEST_EXECUTION", {}) or {}
    enabled = bool(exec_cfg.get("regimeadaptive_growth_size_scaling_enabled", False))
    growth_step_usd = max(
        0.0,
        _coerce_float(exec_cfg.get("regimeadaptive_growth_profit_step_usd", 0.0), 0.0),
    )
    growth_cap = max(
        int(base_size),
        _coerce_int(exec_cfg.get("regimeadaptive_growth_size_scaling_max_contracts", base_size), base_size),
    )
    growth_anchor = str(exec_cfg.get("regimeadaptive_growth_anchor", "peak") or "peak").strip().lower()
    metrics = _current_live_drawdown_metrics(live_drawdown_state)

    if not enabled or growth_step_usd <= 0.0 or growth_cap <= int(base_size):
        if isinstance(signal, dict):
            signal["regimeadaptive_growth_sizing_enabled"] = bool(enabled)
            signal["regimeadaptive_growth_sizing_applied"] = False
        return int(base_size)

    anchor_profit = (
        float(metrics.get("peak_realized_pnl", 0.0) or 0.0)
        if growth_anchor == "peak"
        else float(metrics.get("realized_pnl", 0.0) or 0.0)
    )
    realized_profit = max(0.0, anchor_profit)
    growth_steps = int(realized_profit / growth_step_usd)
    target_size = min(growth_cap, int(base_size) + max(0, growth_steps))

    if isinstance(signal, dict):
        signal["regimeadaptive_growth_sizing_enabled"] = True
        signal["regimeadaptive_growth_sizing_anchor"] = str(growth_anchor)
        signal["regimeadaptive_growth_sizing_profit_usd"] = float(realized_profit)
        signal["regimeadaptive_growth_sizing_step_usd"] = float(growth_step_usd)
        signal["regimeadaptive_growth_sizing_steps"] = int(growth_steps)
        signal["regimeadaptive_growth_sizing_cap"] = int(growth_cap)
        signal["regimeadaptive_growth_sizing_applied"] = bool(int(target_size) != int(base_size))
        signal["regimeadaptive_growth_sizing_target"] = int(target_size)

    return int(target_size)


def _apply_aetherflow_live_conditional_size(
    signal: Optional[dict],
    requested_size: int,
    tracked_live_trades: Optional[list[dict]],
) -> int:
    requested = max(1, int(requested_size))
    if not isinstance(signal, dict):
        return int(requested)

    if _live_strategy_family_name(signal.get("strategy")) != "aetherflow":
        return int(requested)

    aetherflow_cfg = CONFIG.get("AETHERFLOW_STRATEGY", {}) or {}
    sizing_cfg = (
        aetherflow_cfg.get("conditional_live_sizing", {})
        if isinstance(aetherflow_cfg.get("conditional_live_sizing", {}), dict)
        else {}
    )
    enabled = bool(sizing_cfg.get("enabled", False))
    same_side_live_count = 0
    if isinstance(tracked_live_trades, (list, tuple)):
        same_side_live_count = sum(
            1
            for trade in tracked_live_trades
            if _same_side_active_trade(trade, signal)
        )

    signal["aetherflow_conditional_live_sizing_enabled"] = bool(enabled)
    signal["aetherflow_conditional_live_same_side_live_count"] = int(same_side_live_count)
    signal["aetherflow_conditional_live_requested_size"] = int(requested)

    if not enabled:
        signal["aetherflow_conditional_live_sizing_applied"] = False
        signal["aetherflow_conditional_live_reason"] = "disabled"
        signal["aetherflow_conditional_live_multiplier"] = 1.0
        signal["aetherflow_conditional_live_target_size"] = int(requested)
        return int(requested)

    solo_multiplier = max(
        0.0,
        _coerce_float(sizing_cfg.get("solo_multiplier", 1.0), 1.0),
    )
    stacked_multiplier = max(
        0.0,
        _coerce_float(sizing_cfg.get("stacked_multiplier", 1.0), 1.0),
    )
    max_contracts = max(
        1,
        _coerce_int(sizing_cfg.get("max_contracts", requested), requested),
    )

    same_side_stack = same_side_live_count > 0
    multiplier = stacked_multiplier if same_side_stack else solo_multiplier
    reason = "same_side_stack" if same_side_stack else "solo"
    target_size = int(round(float(requested) * float(multiplier)))
    target_size = max(1, min(int(max_contracts), int(target_size)))

    signal["aetherflow_conditional_live_sizing_applied"] = bool(target_size != int(requested))
    signal["aetherflow_conditional_live_reason"] = str(reason)
    signal["aetherflow_conditional_live_multiplier"] = float(multiplier)
    signal["aetherflow_conditional_live_target_size"] = int(target_size)
    return int(target_size)


def _apply_kalshi_gate_size(signal: Optional[dict], size: int) -> int:
    """Apply Kalshi crowd-sentiment trade gating during settlement hours.

    During 10 AM - 4 PM ET, if Kalshi probability aligns with the ML signal
    direction, the contract size is multiplied by up to 3x.  Outside those
    hours (or if Kalshi is unavailable) the size is returned unchanged.
    """
    if not isinstance(signal, dict):
        return size

    kalshi = _get_kalshi_provider()
    if kalshi is None or not getattr(kalshi, "enabled", False):
        if isinstance(signal, dict):
            signal["kalshi_gate_applied"] = False
            signal["kalshi_gate_reason"] = "Kalshi unavailable"
        return size

    settlement_hour = _active_kalshi_settlement_hour_et(kalshi)
    if settlement_hour not in _KALSHI_GATING_HOURS_ET:
        if isinstance(signal, dict):
            signal["kalshi_gate_applied"] = False
            if settlement_hour in (10, 11):
                signal["kalshi_gate_reason"] = "Morning settlement window — crowd unreliable, ML-only"
            else:
                signal["kalshi_gate_reason"] = "Outside settlement hours"
        return size

    side = str(signal.get("side", "") or "").strip().upper()
    direction = 1 if side == "LONG" else (-1 if side == "SHORT" else 0)
    if direction == 0:
        return size

    es_price = float(signal.get("entry_price", 0) or signal.get("price", 0) or 0)
    if es_price <= 0:
        return size

    allowed, reason, multiplier = get_kalshi_gate_decision(direction, es_price, kalshi, CONFIG)

    if isinstance(signal, dict):
        signal["kalshi_gate_applied"] = True
        signal["kalshi_gate_reason"] = reason
        signal["kalshi_gate_multiplier"] = float(multiplier)

    is_aetherflow = _live_strategy_family_name(signal.get("strategy")) == "aetherflow"

    if not allowed:
        if is_aetherflow:
            logging.info("Kalshi VETO skipped for AetherFlow (sizing-only mode): %s — size unchanged", reason)
            return size
        logging.info("Kalshi HARD VETO: %s — size set to 0", reason)
        return 0

    # For Aetherflow: only apply upward multipliers (boost), never reduce
    if is_aetherflow and multiplier < 1.0:
        logging.info("Kalshi soft veto skipped for AetherFlow (sizing-only mode): %s — size unchanged", reason)
        return size

    scaled_size = float(size) * float(multiplier)
    if multiplier < 1.0:
        gated_size = int(math.floor(scaled_size + 1e-9))
        if gated_size <= 0:
            logging.info(
                "Kalshi soft veto escalated to block for minimum-size trade: %s — size %d → 0 (%.1fx)",
                reason,
                size,
                multiplier,
            )
            return 0
    else:
        gated_size = max(1, int(round(scaled_size)))
    if multiplier != 1.0:
        logging.info("Kalshi gate: %s — size %d → %d (%.1fx)", reason, size, gated_size, multiplier)
    return gated_size


def _apply_kalshi_trade_overlay_to_signal(
    signal: Optional[dict],
    current_price: float,
    market_df: Optional[pd.DataFrame],
    *,
    price_action_profile: Optional[dict] = None,
) -> bool:
    if not isinstance(signal, dict):
        return True

    overlay_cfg = CONFIG.get("KALSHI_TRADE_OVERLAY", {}) if isinstance(CONFIG, dict) else {}
    if not bool((overlay_cfg or {}).get("enabled", False)):
        signal["kalshi_trade_overlay_applied"] = False
        signal["kalshi_trade_overlay_reason"] = "disabled"
        return True

    # v5.4 decision: keep Kalshi enabled for all strategies including DE3.
    # A/B tested three options on 5 2025 months + April 2026 two-week window:
    #   v5  (Kalshi on DE3 always):          +$6,469 combined
    #   v5.3 (Kalshi off DE3 always):         +$6,968 combined  (best 5-month but
    #                                                           -$502 on Apr 2026)
    #   v5.4a (Kalshi off DE3 in neutral):    +$6,064 combined  (dominated)
    # The regime-gated compromise underperforms because Apr 2026's Kalshi
    # wins on DE3 came from neutral-regime trades (would have been killed
    # by the regime gate). v5's "keep Kalshi on always" is preferred here —
    # the 2026 wins outweigh the Aug/Nov regressions, and v5 is what was
    # walk-forward-validated. Kill-switch available via JULIE_KALSHI_DE3_MODE=off
    # for experimentation.
    _strat = str(signal.get("strategy", "") or "").strip()
    _de3_mode = os.environ.get("JULIE_KALSHI_DE3_MODE", "on").strip().lower()
    if _strat.startswith("DynamicEngine3") and _de3_mode == "off":
        signal["kalshi_trade_overlay_applied"] = False
        signal["kalshi_trade_overlay_reason"] = "de3_skipped[env_override]"
        signal["kalshi_entry_blocked"] = False
        signal["kalshi_tp_trail_enabled"] = False
        return True

    signal.setdefault("entry_price", _coerce_float(signal.get("entry_price"), current_price))
    kalshi = _get_kalshi_provider()

    settlement_hour = _active_kalshi_settlement_hour_et(kalshi)
    if settlement_hour not in _KALSHI_GATING_HOURS_ET:
        signal["kalshi_trade_overlay_applied"] = False
        signal["kalshi_trade_overlay_reason"] = "outside_gating_hours"
        signal["kalshi_entry_blocked"] = False
        signal["kalshi_tp_trail_enabled"] = False
        return True

    plan = build_kalshi_trade_plan(
        signal,
        current_price,
        kalshi,
        price_action_profile=price_action_profile,
        overlay_cfg=overlay_cfg,
        tick_size=TICK_SIZE,
    )

    signal["kalshi_trade_overlay_applied"] = bool(plan.get("applied", False))
    signal["kalshi_trade_overlay_reason"] = str(plan.get("reason", "") or "")
    signal["kalshi_trade_overlay_role"] = str(plan.get("role", "") or "")
    signal["kalshi_trade_overlay_mode"] = str(plan.get("mode", "") or "")
    signal["kalshi_trade_overlay_forward_weight"] = float(
        _coerce_float(plan.get("forward_weight"), 0.0)
    )
    signal["kalshi_curve_informative"] = bool(plan.get("curve_informative", False))
    signal["kalshi_entry_probability"] = plan.get("entry_probability")
    signal["kalshi_probe_price"] = plan.get("probe_price")
    signal["kalshi_probe_probability"] = plan.get("probe_probability")
    signal["kalshi_momentum_delta"] = plan.get("momentum_delta")
    signal["kalshi_momentum_retention"] = plan.get("momentum_retention")
    signal["kalshi_entry_support_score"] = plan.get("entry_support_score")
    signal["kalshi_entry_threshold"] = plan.get("entry_threshold")
    signal["kalshi_entry_directional_distance_points"] = plan.get("directional_distance_points")
    signal["kalshi_sentiment_momentum"] = plan.get("sentiment_momentum")
    signal["kalshi_support_price"] = plan.get("support_price")
    signal["kalshi_fade_price"] = plan.get("fade_price")
    signal["kalshi_tp_anchor_price"] = plan.get("anchor_price")
    signal["kalshi_tp_anchor_probability"] = plan.get("anchor_probability")
    signal["kalshi_fade_reason"] = plan.get("fade_reason")
    signal["kalshi_support_span_points"] = plan.get("support_span_points")
    signal["kalshi_tp_trail_enabled"] = bool(plan.get("trail_enabled", False))
    signal["kalshi_tp_trigger_price"] = plan.get("trail_trigger_price")
    signal["kalshi_tp_trail_buffer_ticks"] = int(_coerce_int(plan.get("trail_buffer_ticks"), 0) or 0)

    if not bool(plan.get("applied", False)):
        return True

    if bool(plan.get("entry_blocked", False)):
        event_logger.log_kalshi_entry_view(
            str(signal.get("strategy", "Unknown") or "Unknown"),
            str(signal.get("side", "?") or "?"),
            float(_coerce_float(signal.get("entry_price"), current_price)),
            str(signal.get("kalshi_trade_overlay_role", "") or ""),
            "BLOCK",
            entry_probability=_coerce_float(signal.get("kalshi_entry_probability"), math.nan),
            probe_price=_coerce_float(signal.get("kalshi_probe_price"), math.nan),
            probe_probability=_coerce_float(signal.get("kalshi_probe_probability"), math.nan),
            momentum_delta=_coerce_float(signal.get("kalshi_momentum_delta"), math.nan),
            momentum_retention=_coerce_float(signal.get("kalshi_momentum_retention"), math.nan),
            support_score=_coerce_float(signal.get("kalshi_entry_support_score"), math.nan),
            threshold=_coerce_float(signal.get("kalshi_entry_threshold"), math.nan),
        )
        logging.info(
            "Kalshi overlay blocked entry: %s %s | role=%s | score=%s<thresh=%s | reason=%s",
            signal.get("strategy", "Unknown"),
            signal.get("side", "?"),
            signal.get("kalshi_trade_overlay_role", ""),
            signal.get("kalshi_entry_support_score"),
            signal.get("kalshi_entry_threshold"),
            signal.get("kalshi_trade_overlay_reason", ""),
        )
        signal["kalshi_entry_blocked"] = True
        return False

    signal["kalshi_entry_blocked"] = False
    event_logger.log_kalshi_entry_view(
        str(signal.get("strategy", "Unknown") or "Unknown"),
        str(signal.get("side", "?") or "?"),
        float(_coerce_float(signal.get("entry_price"), current_price)),
        str(signal.get("kalshi_trade_overlay_role", "") or ""),
        "PASS" if bool(plan.get("applied", False)) else "SKIP",
        entry_probability=_coerce_float(signal.get("kalshi_entry_probability"), math.nan),
        probe_price=_coerce_float(signal.get("kalshi_probe_price"), math.nan),
        probe_probability=_coerce_float(signal.get("kalshi_probe_probability"), math.nan),
        momentum_delta=_coerce_float(signal.get("kalshi_momentum_delta"), math.nan),
        momentum_retention=_coerce_float(signal.get("kalshi_momentum_retention"), math.nan),
        support_score=_coerce_float(signal.get("kalshi_entry_support_score"), math.nan),
        threshold=_coerce_float(signal.get("kalshi_entry_threshold"), math.nan),
    )
    # --- SHADOW: Kalshi ML gate ---
    # Shadow mode (default): log rule PASS + ML p_win side-by-side.
    # Live mode (JULIE_ML_KALSHI_ACTIVE=1): flip signal["kalshi_entry_blocked"]=True
    # when ML says p_win < threshold.
    try:
        import ml_overlay_shadow as _mls_k
        if _mls_k._KALSHI_PAYLOAD is not None and market_df is not None and len(market_df) >= 120:
            _k_closes = market_df["close"].astype(float).values
            _k_highs = market_df["high"].astype(float).values
            _k_lows = market_df["low"].astype(float).values
            _k_i = len(market_df) - 1
            _k_atr = float(sum(max(_k_highs[j] - _k_lows[j],
                                   abs(_k_highs[j] - _k_closes[j - 1]),
                                   abs(_k_lows[j] - _k_closes[j - 1]))
                               for j in range(_k_i - 13, _k_i + 1)) / 14.0)
            _k_r30 = float(_k_highs[_k_i - 29:_k_i + 1].max() - _k_lows[_k_i - 29:_k_i + 1].min())
            _k_t20 = (_k_closes[_k_i] - _k_closes[_k_i - 20]) / max(1.0, _k_closes[_k_i - 20]) * 100.0
            _k_hi20 = float(_k_highs[_k_i - 19:_k_i + 1].max())
            _k_lo20 = float(_k_lows[_k_i - 19:_k_i + 1].min())
            _k_dh = (_k_closes[_k_i] - _k_hi20) / max(1.0, _k_closes[_k_i]) * 100.0
            _k_dl = (_k_closes[_k_i] - _k_lo20) / max(1.0, _k_closes[_k_i]) * 100.0
            _k_v5 = (_k_closes[_k_i] - _k_closes[_k_i - 5]) / 5.0
            _k_px = _k_closes[_k_i]
            _k_dbank = min(_k_px - math.floor(_k_px / 12.5) * 12.5,
                           math.ceil(_k_px / 12.5) * 12.5 - _k_px)
            # Regime from 120-bar window (mirror regime_classifier.py)
            _k_win = _k_closes[_k_i - 119:_k_i + 1]
            _k_rets = (_k_win[1:] - _k_win[:-1]) / _k_win[:-1]
            _k_mean = float(_k_rets.mean())
            _k_var = float(((_k_rets - _k_mean) ** 2).sum() / max(1, len(_k_rets) - 1))
            _k_vol = (_k_var ** 0.5) * 10_000.0
            _k_abs = float(abs(_k_rets).sum())
            _k_eff = float(abs(_k_rets.sum()) / _k_abs) if _k_abs > 0 else 0.0
            if _k_vol > 3.5 and _k_eff < 0.05: _k_regime = "whipsaw"
            elif _k_eff > 0.12: _k_regime = "calm_trend"
            else: _k_regime = "neutral"
            _k_bar_feats = dict(
                atr14_pts=_k_atr, range_30bar_pts=_k_r30, trend_20bar_pct=_k_t20,
                dist_to_20bar_hi_pct=_k_dh, dist_to_20bar_lo_pct=_k_dl,
                vel_5bar_pts_per_min=_k_v5, dist_to_bank_pts=float(_k_dbank),
                regime_vol_bp=float(_k_vol), regime_eff=float(_k_eff),
            )
            # Get ET-hour fraction from the last bar timestamp (simulated time
            # in replays; wall-clock in live). Falls back to wall-clock if index
            # is missing tz info.
            try:
                _k_last_ts = market_df.index[-1]
                _k_now = _k_last_ts.to_pydatetime() if hasattr(_k_last_ts, "to_pydatetime") else _k_last_ts
                if _k_now.tzinfo is None:
                    _k_now = _k_now.replace(tzinfo=NY_TZ)
                else:
                    _k_now = _k_now.astimezone(NY_TZ)
            except Exception:
                _k_now = datetime.datetime.now(NY_TZ)
            _k_et_frac = _k_now.hour + _k_now.minute / 60.0
            _k_score = _mls_k.score_kalshi(
                signal, _k_bar_feats,
                regime=_k_regime,
                et_hour_frac=_k_et_frac,
                role=str(signal.get("kalshi_trade_overlay_role", "") or "unknown"),
            )
            if _k_score is not None:
                _k_p_win, _k_pred_pnl, _k_should_pass = _k_score
                logging.info(
                    "[SHADOW_KALSHI] rule=PASS ml_p_win=%.3f ml_pred_pnl=%.2f ml=%s "
                    "strat=%s side=%s regime=%s support=%.3f thr=%.3f",
                    _k_p_win, _k_pred_pnl,
                    "PASS" if _k_should_pass else "BLOCK",
                    signal.get("strategy", "?"),
                    signal.get("side", "?"),
                    _k_regime,
                    float(_coerce_float(signal.get("kalshi_entry_support_score"), 0.0)),
                    float(_coerce_float(signal.get("kalshi_entry_threshold"), 0.0)),
                )
                if _mls_k.is_kalshi_live_active() and not _k_should_pass:
                    signal["kalshi_entry_blocked"] = True
                    signal["kalshi_entry_block_reason"] = f"ml_kalshi p_win={_k_p_win:.3f} < thr"
                    return False
            # --- SHADOW: TP-aligned Kalshi gate ---
            # Uses the build_trade_plan output directly to find the strike
            # closest to tp_price and query aligned probability there.
            if _mls_k._KALSHI_TP_PAYLOAD is not None:
                try:
                    _tp_dist = float(_coerce_float(signal.get("tp_dist"), math.nan))
                    _entry_px = float(_coerce_float(signal.get("entry_price"), current_price))
                    _side = str(signal.get("side", "") or "").upper()
                    if math.isfinite(_tp_dist) and _tp_dist > 0 and _side in {"LONG", "SHORT"}:
                        _tp_px = _entry_px + _tp_dist if _side == "LONG" else _entry_px - _tp_dist
                        # Query Kalshi ladder at tp_price via the same interpolation
                        # the overlay uses. build_kalshi_trade_plan already walked
                        # the markets; re-query here via the provider.
                        from kalshi_trade_overlay import (
                            _extract_curve_markets as _mk_extract,
                            _interpolated_aligned_probability as _mk_interp,
                        )
                        _mk_cfg = CONFIG.get("KALSHI_TRADE_OVERLAY", {}) or {}
                        _markets = _mk_extract(kalshi, _entry_px, _mk_cfg)
                        if _markets:
                            _tp_aligned = _mk_interp(_markets, _tp_px, _side)
                            _entry_aligned_raw = float(
                                _coerce_float(signal.get("kalshi_entry_probability"), 0.5)
                            )
                            if _tp_aligned is not None and _entry_aligned_raw > 0:
                                # Nearest strike to TP
                                _nearest = min(
                                    _markets,
                                    key=lambda r: abs(float(r["strike_es"]) - _tp_px),
                                )
                                _nearest_dist = abs(float(_nearest["strike_es"]) - _tp_px)
                                _nearest_oi = float(_nearest.get("open_interest", 0) or 0)
                                _nearest_vol = float(_nearest.get("daily_volume", 0) or 0)
                                # Ladder slope near TP (prob delta / strike span in ±10pt window)
                                _up = [r for r in _markets
                                       if _tp_px < float(r["strike_es"]) <= _tp_px + 10]
                                _dn = [r for r in _markets
                                       if _tp_px - 10 <= float(r["strike_es"]) <= _tp_px]
                                if _up and _dn:
                                    _p_u = _mk_interp(
                                        _markets, float(_up[-1]["strike_es"]), _side
                                    )
                                    _p_d = _mk_interp(
                                        _markets, float(_dn[0]["strike_es"]), _side
                                    )
                                    _span = max(0.5,
                                                float(_up[-1]["strike_es"]) - float(_dn[0]["strike_es"]))
                                    _slope = ((_p_u or 0) - (_p_d or 0)) / _span
                                else:
                                    _slope = 0.0
                                # Minutes to settlement
                                _settle_h = _active_kalshi_settlement_hour_et(kalshi)
                                if _settle_h is not None:
                                    _now_et = current_time.astimezone(NY_TZ)
                                    _settle_dt = _now_et.replace(
                                        hour=int(_settle_h), minute=0, second=0, microsecond=0
                                    )
                                    _mins = max(
                                        0.0,
                                        (_settle_dt - _now_et).total_seconds() / 60.0,
                                    )
                                else:
                                    _mins = 0.0
                                _tp_score = _mls_k.score_kalshi_tp(
                                    signal,
                                    tp_aligned_prob=float(_tp_aligned),
                                    entry_aligned_prob=float(_entry_aligned_raw),
                                    nearest_strike_dist=float(_nearest_dist),
                                    nearest_strike_oi=_nearest_oi,
                                    nearest_strike_volume=_nearest_vol,
                                    ladder_slope_near_tp=float(_slope),
                                    minutes_to_settlement=float(_mins),
                                    atr14_pts=float(_k_atr),
                                    range_30bar_pts=float(_k_r30),
                                    trend_20bar_pct=float(_k_t20),
                                    vel_5bar_pts_per_min=float(_k_v5),
                                    regime=_k_regime,
                                    tp_dist_pts=float(_tp_dist),
                                )
                                if _tp_score is not None:
                                    _tp_p_hit, _tp_pred_pnl, _tp_should_pass = _tp_score
                                    logging.info(
                                        "[SHADOW_KALSHI_TP] rule=PASS ml_p_hit_tp=%.3f "
                                        "ml_pred_pnl=%+.2f ml=%s "
                                        "tp_px=%.2f tp_dist=%.1f tp_prob=%.3f "
                                        "entry_prob=%.3f strat=%s side=%s regime=%s",
                                        _tp_p_hit, _tp_pred_pnl,
                                        "PASS" if _tp_should_pass else "BLOCK",
                                        _tp_px, _tp_dist, float(_tp_aligned),
                                        _entry_aligned_raw,
                                        signal.get("strategy", "?"),
                                        _side, _k_regime,
                                    )
                                    if _mls_k.is_kalshi_tp_live_active() and not _tp_should_pass:
                                        signal["kalshi_entry_blocked"] = True
                                        signal["kalshi_entry_block_reason"] = (
                                            f"ml_kalshi_tp pred_pnl=${_tp_pred_pnl:+.2f} < thr"
                                        )
                                        return False
                except Exception as _tp_exc:
                    logging.debug("ml kalshi_tp shadow err: %s", _tp_exc)
    except Exception as _k_exc:
        logging.debug("ml kalshi shadow err: %s", _k_exc)
    size_multiplier = _coerce_float(plan.get("size_multiplier"), 1.0)
    base_size = max(1, _coerce_int(signal.get("size"), 1) or 1)
    if size_multiplier < 0.999:
        trimmed_size = max(1, int(math.floor((float(base_size) * float(size_multiplier)) + 1e-9)))
        if trimmed_size != base_size:
            signal["kalshi_entry_size_before"] = int(base_size)
            signal["kalshi_entry_size_multiplier"] = float(size_multiplier)
            signal["size"] = int(trimmed_size)
            logging.info(
                "Kalshi overlay size trim: %s %s | role=%s | size %s -> %s | score=%.3f thresh=%.3f",
                signal.get("strategy", "Unknown"),
                signal.get("side", "?"),
                signal.get("kalshi_trade_overlay_role", ""),
                int(base_size),
                int(trimmed_size),
                float(_coerce_float(signal.get("kalshi_entry_support_score"), 0.0)),
                float(_coerce_float(signal.get("kalshi_entry_threshold"), 0.0)),
            )

    adjusted_tp = _coerce_float(plan.get("tp_dist"), math.nan)
    old_tp = _coerce_float(signal.get("tp_dist"), math.nan)
    if math.isfinite(adjusted_tp) and adjusted_tp > 0.0 and math.isfinite(old_tp):
        signal["kalshi_tp_target_price"] = plan.get("target_price")
        signal["kalshi_tp_adjusted"] = bool(plan.get("tp_adjusted", False))
        if bool(plan.get("tp_adjusted", False)) and abs(old_tp - adjusted_tp) > 1e-9:
            signal["tp_dist"] = float(adjusted_tp)
            event_logger.log_kalshi_tp_adjust(
                str(signal.get("strategy", "Unknown") or "Unknown"),
                str(signal.get("side", "?") or "?"),
                float(old_tp),
                float(adjusted_tp),
                fade_strike=_coerce_float(signal.get("kalshi_tp_anchor_price"), math.nan),
                fade_probability=_coerce_float(signal.get("kalshi_tp_anchor_probability"), math.nan),
                role=str(signal.get("kalshi_trade_overlay_role", "") or ""),
                reason=str(signal.get("kalshi_fade_reason", "") or ""),
            )
            logging.info(
                "Kalshi TP overlay: %s %s | role=%s | tp %.2f -> %.2f | anchor=%s | entry_prob=%s | score=%s",
                signal.get("strategy", "Unknown"),
                signal.get("side", "?"),
                signal.get("kalshi_trade_overlay_role", ""),
                float(old_tp),
                float(adjusted_tp),
                signal.get("kalshi_tp_anchor_price"),
                signal.get("kalshi_entry_probability"),
                signal.get("kalshi_entry_support_score"),
            )

    # Pct overlay used to chain here; Option B handles it via snapshot at
    # signal birth + resolver before order placement.
    return True


def _apply_kalshi_tp_trail(
    client: Optional[ProjectXClient],
    trade: Optional[dict],
    *,
    current_time: Optional[datetime.datetime],
    market_price: float,
    bar_high: float,
    bar_low: float,
    bar_index: Optional[int] = None,
) -> Optional[dict]:
    if client is None or not isinstance(trade, dict):
        return None
    trail_plan = compute_kalshi_tp_trail_stop(
        trade,
        market_price=market_price,
        bar_high=bar_high,
        bar_low=bar_low,
        tick_size=TICK_SIZE,
    )
    if not bool(trail_plan.get("triggered", False)):
        return None

    trade["kalshi_tp_trail_seen"] = True
    target_stop_price = _coerce_float(trail_plan.get("stop_price"), math.nan)
    if not bool(trail_plan.get("should_update", False)) or not math.isfinite(target_stop_price):
        return {"status": "unchanged"}

    side_name = str(trade.get("side", "") or "").upper()
    current_stop_price = _coerce_float(trade.get("current_stop_price"), math.nan)
    stop_order_id = _coerce_int(trade.get("stop_order_id"), None)

    if _live_market_has_crossed_stop(target_stop_price, side_name, market_price):
        return _force_close_live_trade_for_crossed_stop(
            client,
            trade,
            current_time,
            market_price=market_price,
            target_stop_price=target_stop_price,
            failure_reason="kalshi_tp_trail_market_already_through_stop",
        )

    if not client.modify_stop_to_breakeven(
        stop_price=target_stop_price,
        side=side_name,
        known_size=max(1, _coerce_int(trade.get("size"), 1) or 1),
        stop_order_id=stop_order_id,
        current_stop_price=current_stop_price,
    ):
        if (
            _live_market_has_crossed_stop(target_stop_price, side_name, market_price)
            or _live_bar_has_crossed_stop(target_stop_price, side_name, bar_high, bar_low)
        ):
            return _force_close_live_trade_for_crossed_stop(
                client,
                trade,
                current_time,
                market_price=market_price,
                target_stop_price=target_stop_price,
                failure_reason="kalshi_tp_trail_stop_update_failed_after_cross",
            )
        logging.warning(
            "Kalshi TP trail stop update failed for %s %s -> %.2f",
            str(trade.get("strategy", "Unknown") or "Unknown"),
            side_name,
            float(target_stop_price),
        )
        return {"status": "failed"}

    trade["current_stop_price"] = float(target_stop_price)
    trade["kalshi_tp_trail_triggered"] = True
    trade["kalshi_tp_trail_trigger_count"] = int(
        _coerce_int(trade.get("kalshi_tp_trail_trigger_count"), 0) or 0
    ) + 1
    trade["kalshi_tp_trail_last_update_bar_index"] = bar_index
    updated_stop_order_id = getattr(client, "_active_stop_order_id", None)
    if updated_stop_order_id is not None:
        trade["stop_order_id"] = updated_stop_order_id

    event_logger.log_trade_modified(
        "KalshiTPTrailStop",
        float(current_stop_price),
        float(target_stop_price),
        f"lock fade strike {float(_coerce_float(trade.get('kalshi_tp_anchor_price'), target_stop_price)):.2f}",
    )
    logging.info(
        "Kalshi TP trail ratchet: %s %s | stop -> %.2f | anchor=%.2f | role=%s",
        str(trade.get("strategy", "Unknown") or "Unknown"),
        side_name,
        float(target_stop_price),
        float(_coerce_float(trade.get("kalshi_tp_anchor_price"), target_stop_price)),
        str(trade.get("kalshi_trade_overlay_role", "") or ""),
    )
    return {
        "status": "updated",
        "target_stop_price": float(target_stop_price),
    }


def _apply_live_execution_size(
    signal: Optional[dict],
    fallback_size: int,
    live_drawdown_state: Optional[dict],
    tracked_live_trades: Optional[list[dict]] = None,
) -> int:
    size = _apply_de3_v4_confidence_tier_size_live(
        signal,
        _signal_base_size(signal, fallback_size),
    )
    size = _apply_regimeadaptive_live_growth_size(signal, size, live_drawdown_state)
    size = _apply_aetherflow_live_conditional_size(signal, size, tracked_live_trades)
    size = _apply_live_drawdown_size(signal, size, live_drawdown_state)
    size = _apply_kalshi_gate_size(signal, size)
    if isinstance(signal, dict):
        signal["size"] = int(size)
    return int(size)


def _same_side_active_trade(active_trade: Optional[dict], signal: Optional[dict]) -> bool:
    if not isinstance(active_trade, dict) or not isinstance(signal, dict):
        return False
    active_side = str(active_trade.get("side", "") or "").strip().upper()
    signal_side = str(signal.get("side", "") or "").strip().upper()
    return bool(active_side and signal_side and active_side == signal_side)


def _live_strategy_family_name(value: Optional[str]) -> str:
    strategy_name = str(value or "").strip()
    if strategy_name.startswith("DynamicEngine3") or strategy_name in {"DynamicEngine", "DynamicEngineStrategy"}:
        return "de3"
    if strategy_name == "RegimeAdaptive":
        return "regimeadaptive"
    if strategy_name == "AetherFlowStrategy":
        return "aetherflow"
    return ""


def _live_strategy_identity_key(payload: Optional[dict]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    strategy_name = str(payload.get("strategy") or "").strip()
    if not strategy_name:
        return None
    sub_strategy = str(
        payload.get("sub_strategy")
        or payload.get("combo_key")
        or ""
    ).strip()
    if sub_strategy:
        return f"{strategy_name}:{sub_strategy}"
    return strategy_name


def _count_same_side_live_family_trades(
    tracked_live_trades: Optional[list[dict]],
    signal: Optional[dict],
    *,
    family_name: str,
) -> int:
    if not isinstance(tracked_live_trades, (list, tuple)):
        return 0
    count = 0
    for trade in tracked_live_trades:
        if not isinstance(trade, dict):
            continue
        if _live_strategy_family_name(trade.get("strategy")) != str(family_name):
            continue
        if not _same_side_active_trade(trade, signal):
            continue
        count += 1
    return int(count)


def _allow_same_side_parallel_entry(
    primary_trade: Optional[dict],
    signal: Optional[dict],
    tracked_live_trades: Optional[list[dict]] = None,
) -> bool:
    if not _same_side_active_trade(primary_trade, signal):
        return False
    if not isinstance(primary_trade, dict) or not isinstance(signal, dict):
        return False
    primary_family = _live_strategy_family_name(primary_trade.get("strategy"))
    signal_family = _live_strategy_family_name(signal.get("strategy"))
    if primary_family == "de3":
        return signal_family in {"regimeadaptive", "aetherflow"}
    if primary_family == "aetherflow" and signal_family == "aetherflow":
        max_legs = max(
            1,
            int(
                _coerce_int(
                    CONFIG.get("AETHERFLOW_STRATEGY", {}).get("live_same_side_parallel_max_legs", 1),
                    1,
                )
            ),
        )
        if max_legs <= 1:
            return False
        same_side_af_count = _count_same_side_live_family_trades(
            tracked_live_trades,
            signal,
            family_name="aetherflow",
        )
        return same_side_af_count < max_legs
    return False


def _normalize_live_side(value: Optional[str]) -> Optional[str]:
    side = str(value or "").strip().upper()
    if side in {"LONG", "SHORT"}:
        return side
    return None


def _live_signal_confidence(sig: Optional[dict]) -> float:
    if not isinstance(sig, dict):
        return 0.0
    for key in (
        "ml_confidence",
        "confidence",
        "gate_prob",
        "aetherflow_confidence",
        "de3_policy_confidence",
        "de3_edge_confidence",
        "aetherflow_selection_score",
        "de3_runtime_rank_score",
        "de3_v2_rank_score",
        "de3_final_score",
        "final_score",
        "score",
        "opt_wr",
        "wr",
        "win_rate",
    ):
        value = sig.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def _live_signal_sort_key(item: tuple[Any, Any, Any, Any]) -> tuple:
    priority, _strat, signal, strat_name = item
    sig = signal if isinstance(signal, dict) else {}
    strategy_label = str(sig.get("strategy", strat_name) or strat_name)
    sub_strategy = str(sig.get("sub_strategy", sig.get("combo_key", "")) or "")
    side = _normalize_live_side(sig.get("side")) or ""
    return (
        int(priority),
        -_live_signal_confidence(sig),
        strategy_label,
        sub_strategy,
        side,
    )


def _calculate_live_trade_close_metrics_from_price(
    trade: Optional[dict],
    exit_price: float,
    *,
    source: str,
    exit_time: Optional[datetime.datetime] = None,
    order_id: Optional[int] = None,
) -> Optional[dict]:
    if not isinstance(trade, dict):
        return None
    metrics = _calculate_live_trade_close_metrics(trade, exit_price)
    metrics["source"] = str(source or "price_snapshot")
    metrics["exit_time"] = exit_time
    if order_id is not None:
        metrics["order_id"] = int(order_id)
    entry_order_id = _coerce_int(trade.get("entry_order_id"), None)
    if entry_order_id is not None:
        metrics["entry_order_id"] = entry_order_id
    return metrics


def _hour_in_window(hour: int, start_hour: int, end_hour: int) -> bool:
    """Half-open [start_hour, end_hour) with wrap support across midnight."""
    if start_hour == end_hour:
        return False
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def _live_entry_window_block_reason(current_time: datetime.datetime) -> Optional[str]:
    risk_cfg = CONFIG.get("RISK", {}) or {}
    if not bool(risk_cfg.get("enforce_no_new_entries_window", False)):
        return None
    start_hour = min(23, max(0, _coerce_int(risk_cfg.get("no_new_entries_start_hour_et", 16), 16)))
    end_hour = min(23, max(0, _coerce_int(risk_cfg.get("no_new_entries_end_hour_et", 18), 18)))
    current_time_et = (
        current_time.replace(tzinfo=NY_TZ)
        if current_time.tzinfo is None
        else current_time.astimezone(NY_TZ)
    )
    if not _hour_in_window(int(current_time_et.hour), start_hour, end_hour):
        return None
    return (
        f"Live no-entry window active {start_hour:02d}:00-{end_hour:02d}:00 ET "
        f"(current {current_time_et:%H:%M:%S %Z})"
    )


def _log_live_entry_window_block(
    signal: Optional[dict],
    fallback_strategy: str,
    current_time: datetime.datetime,
) -> str:
    reason = _live_entry_window_block_reason(current_time) or "Live no-entry window active"
    strategy_name = (
        str((signal or {}).get("strategy", "") or "").strip()
        if isinstance(signal, dict)
        else ""
    ) or fallback_strategy
    side = (
        str((signal or {}).get("side", "") or "").strip().upper()
        if isinstance(signal, dict)
        else ""
    ) or "ALL"
    logging.info(
        "Skipping live entry attempt for %s (%s): %s",
        strategy_name,
        side,
        reason,
    )
    event_logger.log_filter_check(
        "LiveEntryWindow",
        side,
        False,
        reason,
        strategy=strategy_name,
        metrics={
            "current_time_et": (
                current_time.replace(tzinfo=NY_TZ)
                if current_time.tzinfo is None
                else current_time.astimezone(NY_TZ)
            ).isoformat()
        },
    )
    return reason


def _trade_point_value() -> float:
    risk_cfg = CONFIG.get("RISK", {}) or {}
    return _coerce_float(risk_cfg.get("POINT_VALUE", 5.0), 5.0)


def _trade_reporting_round_turn_fee_per_contract() -> float:
    risk_cfg = CONFIG.get("RISK", {}) or {}
    base_round_turn = float(_coerce_float(risk_cfg.get("FEES_PER_SIDE"), 0.37) or 0.37) * 2.0
    topstep_commission = float(
        _coerce_float(
            risk_cfg.get("TOPSTEP_COMMISSION_ROUND_TURN_PER_CONTRACT"),
            0.50,
        )
        or 0.50
    )
    return max(0.0, base_round_turn + topstep_commission)


def _calculate_live_trade_close_metrics(active_trade: dict, exit_price: float) -> dict:
    entry_price = _coerce_float(active_trade.get("entry_price", 0.0), 0.0)
    trade_size = max(1, _coerce_int(active_trade.get("size", 1), 1))
    point_value = _trade_point_value()
    trade_side = str(active_trade.get("side", "") or "").upper()
    if trade_side == "LONG":
        pnl_points = float(exit_price - entry_price)
    else:
        pnl_points = float(entry_price - exit_price)
    pnl_dollars_gross = float(pnl_points * point_value * trade_size)
    pnl_fee_dollars = float(_trade_reporting_round_turn_fee_per_contract() * float(trade_size))
    pnl_dollars_net = float(pnl_dollars_gross - pnl_fee_dollars)
    return {
        "exit_price": float(exit_price),
        "pnl_points": float(pnl_points),
        "pnl_dollars_gross": float(pnl_dollars_gross),
        "pnl_fee_dollars": float(pnl_fee_dollars),
        "pnl_dollars_net": float(pnl_dollars_net),
        "pnl_dollars": float(pnl_dollars_net),
        "source": "price_snapshot",
        "exit_time": None,
    }


def _reconcile_live_trade_close(
    client: ProjectXClient,
    active_trade: dict,
    current_time: datetime.datetime,
    *,
    fallback_exit_price: float,
    close_order_id: Optional[int] = None,
) -> dict:
    close_metrics = client.reconcile_trade_close(
        active_trade,
        exit_time=current_time,
        fallback_exit_price=fallback_exit_price,
        close_order_id=close_order_id,
        point_value=_trade_point_value(),
    )
    if isinstance(close_metrics, dict):
        return close_metrics
    return _calculate_live_trade_close_metrics(active_trade, fallback_exit_price)


def _infer_de3_lane_from_variant(variant_id: Optional[str]) -> str:
    variant_text = str(variant_id or "").strip()
    if "_Long_Mom_" in variant_text:
        return "Long_Mom"
    if "_Long_Rev_" in variant_text:
        return "Long_Rev"
    if "_Short_Mom_" in variant_text:
        return "Short_Mom"
    if "_Short_Rev_" in variant_text:
        return "Short_Rev"
    return ""


def _is_de3_v4_trade_management_payload(payload: Optional[dict]) -> bool:
    if not isinstance(payload, dict):
        return False
    strategy_name = str(payload.get("strategy", "") or "")
    variant_hint = str(
        payload.get("de3_v4_selected_variant_id")
        or payload.get("sub_strategy")
        or payload.get("combo_key")
        or ""
    ).strip()
    inferred_lane = _infer_de3_lane_from_variant(variant_hint)
    if strategy_name and not strategy_name.startswith("DynamicEngine3"):
        # Restored positions (synced from broker after a crash/restart) often
        # have no sub_strategy/combo_key, so the variant-based lane inferrer
        # returns "". Fall back to side-based default so the management layer
        # can still arm BE/trailing on an orphan trade rather than leaving it
        # to static broker brackets. The default lane applies default BE/trail
        # params — approximate-but-useful beats zero management.
        if strategy_name == "RestoredLivePosition":
            if inferred_lane:
                return True
            side = str(payload.get("side", "") or "").upper()
            return side in ("LONG", "SHORT")
        return False
    if str(payload.get("de3_version", "") or "").strip().lower() == "v4":
        return True
    if payload.get("de3_v4_selected_variant_id"):
        return True
    if payload.get("de3_v4_selected_lane"):
        return True
    if strategy_name.startswith("DynamicEngine3") and inferred_lane:
        return True
    return False


def _de3_trade_day_key(
    timestamp,
    *,
    roll_hour_et: int = 18,
) -> Optional[datetime.date]:
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


def _de3_variant_id_from_trade_payload(payload: Optional[dict]) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(
        payload.get("de3_v4_selected_variant_id")
        or payload.get("sub_strategy")
        or ""
    ).strip()


def _resolve_live_de3_entry_trade_day_extreme_context(
    payload: Optional[dict],
    *,
    entry_price: float,
    tp_dist: float,
    current_time: datetime.datetime,
    current_price: float,
    market_df: Optional[pd.DataFrame],
    trade_day_roll_hour_et: int = 18,
) -> dict:
    result = {
        "variant_id": "",
        "entry_trade_day_high": None,
        "entry_trade_day_low": None,
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
    variant_id = _de3_variant_id_from_trade_payload(payload)
    result["variant_id"] = variant_id

    roll_hour_et = max(0, min(23, int(trade_day_roll_hour_et)))
    day_key = _de3_trade_day_key(current_time, roll_hour_et=roll_hour_et)
    high_val = _coerce_float(current_price, math.nan)
    low_val = _coerce_float(current_price, math.nan)
    if (
        isinstance(market_df, pd.DataFrame)
        and not market_df.empty
        and isinstance(market_df.index, pd.DatetimeIndex)
        and "high" in market_df.columns
        and "low" in market_df.columns
        and day_key is not None
    ):
        try:
            current_ts = pd.Timestamp(current_time)
            index_tz = getattr(market_df.index, "tz", None)
            if index_tz is None:
                if current_ts.tzinfo is not None:
                    current_ts = current_ts.tz_localize(None)
            else:
                if current_ts.tzinfo is None:
                    current_ts = current_ts.tz_localize(index_tz)
                else:
                    current_ts = current_ts.tz_convert(index_tz)
            known_df = market_df.loc[market_df.index <= current_ts]
            if not known_df.empty:
                trade_day_mask = [
                    _de3_trade_day_key(ts, roll_hour_et=roll_hour_et) == day_key
                    for ts in known_df.index
                ]
                day_df = known_df.loc[trade_day_mask]
                if not day_df.empty:
                    day_high = _coerce_float(
                        pd.to_numeric(day_df["high"], errors="coerce").max(),
                        math.nan,
                    )
                    day_low = _coerce_float(
                        pd.to_numeric(day_df["low"], errors="coerce").min(),
                        math.nan,
                    )
                    if math.isfinite(day_high):
                        high_val = (
                            max(float(high_val), float(day_high))
                            if math.isfinite(high_val)
                            else float(day_high)
                        )
                    if math.isfinite(day_low):
                        low_val = (
                            min(float(low_val), float(day_low))
                            if math.isfinite(low_val)
                            else float(day_low)
                        )
        except Exception:
            pass

    result["entry_trade_day_high"] = float(high_val) if math.isfinite(high_val) else None
    result["entry_trade_day_low"] = float(low_val) if math.isfinite(low_val) else None
    extreme_price = high_val if side_name == "LONG" else low_val
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


def _resolve_live_de3_entry_trade_day_extreme_profile(
    payload: Optional[dict],
    *,
    entry_price: float,
    tp_dist: float,
    current_time: datetime.datetime,
    current_price: float,
    market_df: Optional[pd.DataFrame],
    break_even_cfg: Optional[dict],
    entry_trade_day_extreme_cfg: Optional[dict],
) -> dict:
    result = {
        "active": False,
        "profile_name": "",
        "variant_id": "",
        "entry_trade_day_high": None,
        "entry_trade_day_low": None,
        "extreme_price": None,
        "target_beyond_trade_day_extreme": False,
        "progress_pct": None,
        "force_break_even_on_reach": False,
        "post_reach_trail_pct": 0.0,
    }
    trade_day_cfg = entry_trade_day_extreme_cfg if isinstance(entry_trade_day_extreme_cfg, dict) else {}
    roll_hour_et = max(0, min(23, _coerce_int(trade_day_cfg.get("trade_day_roll_hour_et"), 18)))
    result.update(
        _resolve_live_de3_entry_trade_day_extreme_context(
            payload,
            entry_price=entry_price,
            tp_dist=tp_dist,
            current_time=current_time,
            current_price=current_price,
            market_df=market_df,
            trade_day_roll_hour_et=roll_hour_et,
        )
    )
    if not bool(trade_day_cfg.get("enabled", False)):
        return result
    if not _is_de3_v4_trade_management_payload(payload):
        return result
    side_name = str(payload.get("side", "") or "").upper()
    variant_id = str(result.get("variant_id", "") or "")
    target_beyond = bool(result.get("target_beyond_trade_day_extreme", False))
    profiles = trade_day_cfg.get("profiles", [])
    if not isinstance(profiles, (list, tuple)):
        return result
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
            ((break_even_cfg or {}).get("post_activation_trail_pct")),
            _coerce_float(((break_even_cfg or {}).get("trail_pct")), 0.0),
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


def _resolve_live_de3_entry_trade_day_extreme_admission_block(
    payload: Optional[dict],
    *,
    entry_price: float,
    tp_dist: float,
    current_time: datetime.datetime,
    current_price: float,
    market_df: Optional[pd.DataFrame],
) -> dict:
    runtime_cfg = (
        (CONFIG.get("DE3_V4", {}) or {}).get("runtime", {})
        if isinstance((CONFIG.get("DE3_V4", {}) or {}).get("runtime", {}), dict)
        else {}
    )
    trade_management_cfg = (
        runtime_cfg.get("trade_management", {})
        if isinstance(runtime_cfg.get("trade_management", {}), dict)
        else {}
    )
    block_cfg = (
        trade_management_cfg.get("entry_trade_day_extreme_admission_block", {})
        if isinstance(trade_management_cfg.get("entry_trade_day_extreme_admission_block", {}), dict)
        else {}
    )
    result = {
        "active": False,
        "profile_name": "",
        "variant_id": "",
        "entry_trade_day_high": None,
        "entry_trade_day_low": None,
        "extreme_price": None,
        "target_beyond_trade_day_extreme": False,
        "progress_pct": None,
    }
    if not isinstance(block_cfg, dict):
        return result
    if not bool(block_cfg.get("enabled", False)):
        return result
    roll_hour_et = max(0, min(23, _coerce_int(block_cfg.get("trade_day_roll_hour_et"), 18)))
    result.update(
        _resolve_live_de3_entry_trade_day_extreme_context(
            payload,
            entry_price=entry_price,
            tp_dist=tp_dist,
            current_time=current_time,
            current_price=current_price,
            market_df=market_df,
            trade_day_roll_hour_et=roll_hour_et,
        )
    )
    if not _is_de3_v4_trade_management_payload(payload):
        return result
    side_name = str((payload or {}).get("side", "") or "").upper()
    variant_id = str(result.get("variant_id", "") or "")
    target_beyond = bool(result.get("target_beyond_trade_day_extreme", False))
    profiles = block_cfg.get("profiles", [])
    if not isinstance(profiles, (list, tuple)):
        return result
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


def _resolve_live_de3_entry_trade_day_extreme_size_adjustment(
    payload: Optional[dict],
    *,
    entry_price: float,
    tp_dist: float,
    current_time: datetime.datetime,
    current_price: float,
    market_df: Optional[pd.DataFrame],
) -> dict:
    runtime_cfg = (
        (CONFIG.get("DE3_V4", {}) or {}).get("runtime", {})
        if isinstance((CONFIG.get("DE3_V4", {}) or {}).get("runtime", {}), dict)
        else {}
    )
    trade_management_cfg = (
        runtime_cfg.get("trade_management", {})
        if isinstance(runtime_cfg.get("trade_management", {}), dict)
        else {}
    )
    size_cfg = (
        trade_management_cfg.get("entry_trade_day_extreme_size_adjustment", {})
        if isinstance(trade_management_cfg.get("entry_trade_day_extreme_size_adjustment", {}), dict)
        else {}
    )
    result = {
        "active": False,
        "profile_name": "",
        "variant_id": "",
        "entry_trade_day_high": None,
        "entry_trade_day_low": None,
        "extreme_price": None,
        "target_beyond_trade_day_extreme": False,
        "progress_pct": None,
        "size_multiplier": 1.0,
        "min_contracts": 1,
    }
    if not isinstance(size_cfg, dict):
        return result
    if not bool(size_cfg.get("enabled", False)):
        return result
    roll_hour_et = max(0, min(23, _coerce_int(size_cfg.get("trade_day_roll_hour_et"), 18)))
    result.update(
        _resolve_live_de3_entry_trade_day_extreme_context(
            payload,
            entry_price=entry_price,
            tp_dist=tp_dist,
            current_time=current_time,
            current_price=current_price,
            market_df=market_df,
            trade_day_roll_hour_et=roll_hour_et,
        )
    )
    if not _is_de3_v4_trade_management_payload(payload):
        return result
    side_name = str((payload or {}).get("side", "") or "").upper()
    variant_id = str(result.get("variant_id", "") or "")
    target_beyond = bool(result.get("target_beyond_trade_day_extreme", False))
    progress_pct = _coerce_float(result.get("progress_pct"), float("nan"))
    profiles = size_cfg.get("profiles", [])
    if not isinstance(profiles, (list, tuple)):
        return result
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


def _apply_live_de3_entry_trade_day_extreme_size_adjustment(
    signal: Optional[dict],
    *,
    current_size: int,
    entry_price: float,
    tp_dist: float,
    current_time: datetime.datetime,
    current_price: float,
    market_df: Optional[pd.DataFrame],
) -> int:
    if not isinstance(signal, dict):
        return int(current_size)
    current_size = max(1, int(current_size))
    signal["de3_entry_trade_day_extreme_size_adjustment_active"] = False
    signal["de3_entry_trade_day_extreme_size_adjustment_applied"] = False
    signal["de3_entry_trade_day_extreme_size_adjustment_profile_name"] = ""
    signal["de3_entry_trade_day_extreme_size_adjustment_target_beyond"] = False
    signal["de3_entry_trade_day_extreme_size_adjustment_progress_pct"] = None
    signal["de3_entry_trade_day_extreme_size_adjustment_requested_size"] = int(current_size)
    signal["de3_entry_trade_day_extreme_size_adjustment_final_size"] = int(current_size)
    signal["de3_entry_trade_day_extreme_size_adjustment_multiplier"] = 1.0
    signal["de3_entry_trade_day_extreme_size_adjustment_min_contracts"] = 1
    ctx = _resolve_live_de3_entry_trade_day_extreme_size_adjustment(
        signal,
        entry_price=entry_price,
        tp_dist=tp_dist,
        current_time=current_time,
        current_price=current_price,
        market_df=market_df,
    )
    signal["de3_entry_trade_day_extreme_size_adjustment_active"] = bool(
        ctx.get("active", False)
    )
    signal["de3_entry_trade_day_extreme_size_adjustment_profile_name"] = str(
        ctx.get("profile_name", "") or ""
    )
    signal["de3_entry_trade_day_extreme_size_adjustment_target_beyond"] = bool(
        ctx.get("target_beyond_trade_day_extreme", False)
    )
    signal["de3_entry_trade_day_extreme_size_adjustment_progress_pct"] = ctx.get(
        "progress_pct"
    )
    signal["de3_entry_trade_day_extreme_size_adjustment_multiplier"] = float(
        max(0.0, _coerce_float(ctx.get("size_multiplier"), 1.0))
    )
    signal["de3_entry_trade_day_extreme_size_adjustment_min_contracts"] = int(
        max(1, _coerce_int(ctx.get("min_contracts"), 1))
    )
    if not bool(ctx.get("active", False)):
        signal["size"] = int(current_size)
        return int(current_size)
    candidate_size = int(
        math.floor(
            float(current_size)
            * max(0.0, _coerce_float(ctx.get("size_multiplier"), 1.0))
        )
    )
    min_contracts = int(max(1, _coerce_int(ctx.get("min_contracts"), 1)))
    if candidate_size < min_contracts:
        candidate_size = int(min_contracts)
    if candidate_size > current_size:
        candidate_size = int(current_size)
    if candidate_size < current_size:
        current_size = int(candidate_size)
        signal["de3_entry_trade_day_extreme_size_adjustment_applied"] = True
    signal["de3_entry_trade_day_extreme_size_adjustment_final_size"] = int(current_size)
    signal["size"] = int(current_size)
    return int(current_size)


def _resolve_live_de3_entry_trade_day_extreme_early_exit_profile(
    payload: Optional[dict],
    *,
    entry_price: float,
    tp_dist: float,
    current_time: datetime.datetime,
    current_price: float,
    market_df: Optional[pd.DataFrame],
) -> dict:
    runtime_cfg = (
        (CONFIG.get("DE3_V4", {}) or {}).get("runtime", {})
        if isinstance((CONFIG.get("DE3_V4", {}) or {}).get("runtime", {}), dict)
        else {}
    )
    trade_management_cfg = (
        runtime_cfg.get("trade_management", {})
        if isinstance(runtime_cfg.get("trade_management", {}), dict)
        else {}
    )
    early_exit_cfg = (
        trade_management_cfg.get("entry_trade_day_extreme_early_exit", {})
        if isinstance(trade_management_cfg.get("entry_trade_day_extreme_early_exit", {}), dict)
        else {}
    )
    result = {
        "active": False,
        "profile_name": "",
        "variant_id": "",
        "entry_trade_day_high": None,
        "entry_trade_day_low": None,
        "extreme_price": None,
        "target_beyond_trade_day_extreme": False,
        "progress_pct": None,
        "min_progress_by_bars": None,
        "min_progress_pct": None,
        "max_profit_crosses": None,
    }
    if not isinstance(early_exit_cfg, dict):
        return result
    if not bool(early_exit_cfg.get("enabled", False)):
        return result
    roll_hour_et = max(0, min(23, _coerce_int(early_exit_cfg.get("trade_day_roll_hour_et"), 18)))
    result.update(
        _resolve_live_de3_entry_trade_day_extreme_context(
            payload,
            entry_price=entry_price,
            tp_dist=tp_dist,
            current_time=current_time,
            current_price=current_price,
            market_df=market_df,
            trade_day_roll_hour_et=roll_hour_et,
        )
    )
    if not _is_de3_v4_trade_management_payload(payload):
        return result
    side_name = str((payload or {}).get("side", "") or "").upper()
    variant_id = str(result.get("variant_id", "") or "")
    target_beyond = bool(result.get("target_beyond_trade_day_extreme", False))
    progress_pct = _coerce_float(result.get("progress_pct"), float("nan"))
    profiles = early_exit_cfg.get("profiles", [])
    if not isinstance(profiles, (list, tuple)):
        return result
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
        has_progress_rule = min_progress_by_bars > 0 and math.isfinite(min_progress_pct)
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


def round_points_to_tick(value: float) -> float:
    value = _coerce_float(value, math.nan)
    if not math.isfinite(value):
        return float(value)
    tick_size = _coerce_float(globals().get("TICK_SIZE"), 0.0)
    if tick_size <= 0.0:
        return float(value)
    scaled = float(value) / float(tick_size)
    if scaled >= 0.0:
        rounded = math.floor(scaled + 0.5 + 1e-9)
    else:
        rounded = math.ceil(scaled - 0.5 - 1e-9)
    return round(float(rounded) * float(tick_size), 10)


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


def _resolve_live_de3_profit_milestone_profile(
    payload: Optional[dict],
    *,
    entry_price: float,
    tp_dist: float,
    break_even_cfg: Optional[dict],
    profit_milestone_cfg: Optional[dict],
) -> dict:
    result = {
        "active": False,
        "profile_name": "",
        "milestone_price": None,
        "trigger_pct": 0.0,
        "force_break_even_on_reach": False,
        "post_reach_trail_pct": 0.0,
    }
    milestone_cfg = profit_milestone_cfg if isinstance(profit_milestone_cfg, dict) else {}
    if not bool(milestone_cfg.get("enabled", False)):
        return result
    if not _is_de3_v4_trade_management_payload(payload):
        return result
    side_name = str((payload or {}).get("side", "") or "").upper()
    entry_price = _coerce_float(entry_price, math.nan)
    tp_dist = round_points_to_tick(max(0.0, _coerce_float(tp_dist, 0.0)))
    if side_name not in {"LONG", "SHORT"} or not math.isfinite(entry_price) or tp_dist <= 0.0:
        return result
    profiles = milestone_cfg.get("profiles", [])
    if not isinstance(profiles, (list, tuple)):
        return result
    variant_id = _de3_variant_id_from_trade_payload(payload)
    trail_default = _coerce_float(
        ((break_even_cfg or {}).get("post_activation_trail_pct")),
        _coerce_float(((break_even_cfg or {}).get("trail_pct")), 0.0),
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


def _align_stop_price_to_tick(price: float, side: str) -> float:
    if not math.isfinite(price):
        return float(price)
    if TICK_SIZE <= 0:
        return float(price)
    scaled = float(price) / float(TICK_SIZE)
    if str(side).upper() == "LONG":
        return round(math.floor(scaled + 1e-9) * float(TICK_SIZE), 10)
    return round(math.ceil(scaled - 1e-9) * float(TICK_SIZE), 10)


def _derive_live_stop_price(entry_price: float, sl_dist: float, side: str) -> float:
    if not math.isfinite(entry_price) or not math.isfinite(sl_dist) or sl_dist <= 0.0:
        return float("nan")
    if str(side).upper() == "LONG":
        return float(entry_price - sl_dist)
    if str(side).upper() == "SHORT":
        return float(entry_price + sl_dist)
    return float("nan")


def _derive_live_target_price(entry_price: float, tp_dist: float, side: str) -> float:
    if not math.isfinite(entry_price) or not math.isfinite(tp_dist) or tp_dist <= 0.0:
        return float("nan")
    if str(side).upper() == "LONG":
        return float(entry_price + tp_dist)
    if str(side).upper() == "SHORT":
        return float(entry_price - tp_dist)
    return float("nan")


DE3_LIVE_POSITION_SNAPSHOT_KEYS = (
    "de3_version",
    "de3_v4_selected_variant_id",
    "de3_v4_selected_lane",
    "de3_v4_selected_route_id",
    "mfe_points",
    "mae_points",
    "profit_crosses",
    "was_green",
    "de3_break_even_armed",
    "de3_break_even_applied",
    "de3_break_even_move_count",
    "de3_break_even_last_stop_price",
    "de3_break_even_pending_stop_price",
    "de3_break_even_pending_from_bar_index",
    "de3_break_even_trigger_bar_index",
    "de3_break_even_trigger_mfe_points",
    "de3_break_even_locked_points",
    "de3_effective_stop_price",
    "de3_profit_milestone_profile_active",
    "de3_profit_milestone_profile_name",
    "de3_profit_milestone_price",
    "de3_profit_milestone_trigger_pct",
    "de3_profit_milestone_force_break_even",
    "de3_profit_milestone_reached",
    "de3_profit_milestone_reached_bar_index",
    "de3_profit_milestone_reached_mfe_points",
    "de3_break_even_post_profit_milestone_trail_pct",
    "de3_entry_trade_day_high",
    "de3_entry_trade_day_low",
    "de3_entry_trade_day_extreme_price",
    "de3_entry_trade_day_extreme_progress_pct",
    "de3_entry_trade_day_extreme_target_beyond",
    "de3_entry_trade_day_extreme_profile_active",
    "de3_entry_trade_day_extreme_profile_name",
    "de3_entry_trade_day_extreme_force_break_even",
    "de3_entry_trade_day_extreme_reached",
    "de3_entry_trade_day_extreme_reached_bar_index",
    "de3_entry_trade_day_extreme_reached_mfe_points",
    "de3_break_even_post_entry_trade_day_extreme_trail_pct",
)

DE3_LIVE_POSITION_RUNTIME_RESTORE_KEYS = (
    "mfe_points",
    "mae_points",
    "profit_crosses",
    "was_green",
    "de3_break_even_armed",
    "de3_break_even_applied",
    "de3_break_even_move_count",
    "de3_break_even_last_stop_price",
    "de3_break_even_pending_stop_price",
    "de3_break_even_pending_from_bar_index",
    "de3_break_even_trigger_bar_index",
    "de3_break_even_trigger_mfe_points",
    "de3_break_even_locked_points",
    "de3_effective_stop_price",
    "de3_profit_milestone_reached",
    "de3_profit_milestone_reached_bar_index",
    "de3_profit_milestone_reached_mfe_points",
    "de3_entry_trade_day_high",
    "de3_entry_trade_day_low",
    "de3_entry_trade_day_extreme_price",
    "de3_entry_trade_day_extreme_progress_pct",
    "de3_entry_trade_day_extreme_target_beyond",
    "de3_entry_trade_day_extreme_reached",
    "de3_entry_trade_day_extreme_reached_bar_index",
    "de3_entry_trade_day_extreme_reached_mfe_points",
)

DE3_LIVE_POSITION_PROFIT_PROFILE_KEYS = (
    "de3_profit_milestone_profile_active",
    "de3_profit_milestone_profile_name",
    "de3_profit_milestone_price",
    "de3_profit_milestone_trigger_pct",
    "de3_profit_milestone_force_break_even",
    "de3_break_even_post_profit_milestone_trail_pct",
)

DE3_LIVE_POSITION_ENTRY_EXTREME_PROFILE_KEYS = (
    "de3_entry_trade_day_high",
    "de3_entry_trade_day_low",
    "de3_entry_trade_day_extreme_price",
    "de3_entry_trade_day_extreme_progress_pct",
    "de3_entry_trade_day_extreme_target_beyond",
    "de3_entry_trade_day_extreme_profile_active",
    "de3_entry_trade_day_extreme_profile_name",
    "de3_entry_trade_day_extreme_force_break_even",
    "de3_break_even_post_entry_trade_day_extreme_trail_pct",
)


def _copy_present_keys(
    source: Optional[dict],
    destination: Optional[dict],
    keys: tuple[str, ...],
) -> None:
    if not isinstance(source, dict) or not isinstance(destination, dict):
        return
    for key in keys:
        if key in source:
            destination[key] = source.get(key)


def _refresh_live_de3_management_profile_flags(
    trade: Optional[dict],
) -> None:
    if not isinstance(trade, dict):
        return
    runtime_cfg = (
        (CONFIG.get("DE3_V4", {}) or {}).get("runtime", {})
        if isinstance((CONFIG.get("DE3_V4", {}) or {}).get("runtime", {}), dict)
        else {}
    )
    trade_management_cfg = (
        runtime_cfg.get("trade_management", {})
        if isinstance(runtime_cfg.get("trade_management", {}), dict)
        else {}
    )
    break_even_cfg = (
        trade_management_cfg.get("break_even", {})
        if isinstance(trade_management_cfg.get("break_even", {}), dict)
        else {}
    )
    early_exit_cfg = (
        trade_management_cfg.get("early_exit", {})
        if isinstance(trade_management_cfg.get("early_exit", {}), dict)
        else {}
    )
    profit_milestone_cfg = (
        trade_management_cfg.get("profit_milestone_stop", {})
        if isinstance(trade_management_cfg.get("profit_milestone_stop", {}), dict)
        else {}
    )
    de3_trade_management_enabled = bool(
        trade_management_cfg.get("enabled", False)
        and _is_de3_v4_trade_management_payload(trade)
    )
    trade["de3_trade_management_enabled"] = de3_trade_management_enabled
    trade["de3_break_even_enabled"] = bool(
        de3_trade_management_enabled
        and break_even_cfg.get("enabled", False)
    )
    trade["de3_break_even_activate_on_next_bar"] = bool(
        break_even_cfg.get("activate_on_next_bar", True)
    )
    trade["de3_break_even_trigger_pct"] = float(
        _coerce_float(
            break_even_cfg.get("trigger_pct"),
            trade.get("de3_break_even_trigger_pct", 0.0),
        )
    )
    trade["de3_break_even_buffer_ticks"] = int(
        max(
            0,
            _coerce_int(
                break_even_cfg.get("buffer_ticks"),
                _coerce_int(trade.get("de3_break_even_buffer_ticks"), 0),
            ),
        )
    )
    trade["de3_break_even_trail_pct"] = float(
        max(
            0.0,
            _coerce_float(
                break_even_cfg.get("trail_pct"),
                trade.get("de3_break_even_trail_pct", 0.0),
            ),
        )
    )
    trade["de3_early_exit_enabled"] = bool(
        de3_trade_management_enabled
        and early_exit_cfg.get("enabled", False)
    ) or bool(trade.get("de3_entry_trade_day_extreme_early_exit_profile_active", False))
    entry_price = _coerce_float(trade.get("entry_price"), math.nan)
    tp_dist = _coerce_float(trade.get("tp_dist"), math.nan)
    if not math.isfinite(entry_price) or not math.isfinite(tp_dist) or tp_dist <= 0.0:
        return
    profit_ctx = _resolve_live_de3_profit_milestone_profile(
        trade,
        entry_price=entry_price,
        tp_dist=tp_dist,
        break_even_cfg=break_even_cfg,
        profit_milestone_cfg=profit_milestone_cfg,
    )
    trade["de3_profit_milestone_profile_active"] = bool(
        profit_ctx.get("active", False)
    )
    trade["de3_profit_milestone_profile_name"] = str(
        profit_ctx.get("profile_name", "") or ""
    )
    trade["de3_profit_milestone_price"] = profit_ctx.get("milestone_price")
    trade["de3_profit_milestone_trigger_pct"] = float(
        max(0.0, _coerce_float(profit_ctx.get("trigger_pct"), 0.0))
    )
    trade["de3_profit_milestone_force_break_even"] = bool(
        profit_ctx.get("force_break_even_on_reach", False)
    )
    trade["de3_break_even_post_profit_milestone_trail_pct"] = float(
        max(
            0.0,
            _coerce_float(profit_ctx.get("post_reach_trail_pct"), 0.0),
        )
    )


def _merge_restored_live_trade_runtime_state(
    trade: Optional[dict],
    position_snapshot: Optional[dict],
) -> None:
    if not isinstance(trade, dict) or not isinstance(position_snapshot, dict):
        return
    _copy_present_keys(
        position_snapshot,
        trade,
        DE3_LIVE_POSITION_RUNTIME_RESTORE_KEYS,
    )
    if bool(position_snapshot.get("de3_profit_milestone_profile_active", False)):
        _copy_present_keys(
            position_snapshot,
            trade,
            DE3_LIVE_POSITION_PROFIT_PROFILE_KEYS,
        )
    if (
        bool(position_snapshot.get("de3_entry_trade_day_extreme_profile_active", False))
        or position_snapshot.get("de3_entry_trade_day_extreme_price") is not None
        or position_snapshot.get("de3_entry_trade_day_high") is not None
        or position_snapshot.get("de3_entry_trade_day_low") is not None
    ):
        _copy_present_keys(
            position_snapshot,
            trade,
            DE3_LIVE_POSITION_ENTRY_EXTREME_PROFILE_KEYS,
        )


def _refresh_live_trade_brackets_from_projectx(
    client: Any,
    trade: Optional[dict],
    *,
    reference_entry_price: Optional[float] = None,
    max_cache_age_sec: float = 2.0,
    force_refresh: bool = False,
) -> dict:
    if client is None or not isinstance(trade, dict):
        return {}

    side_name = _normalize_live_side(trade.get("side"))
    size = max(0, _coerce_int(trade.get("size"), 0))
    if side_name not in {"LONG", "SHORT"} or size <= 0:
        return {}

    entry_price = _coerce_float(reference_entry_price, math.nan)
    if not math.isfinite(entry_price):
        entry_price = _coerce_float(trade.get("broker_entry_price"), math.nan)
    if not math.isfinite(entry_price):
        entry_price = _coerce_float(trade.get("entry_price"), math.nan)

    expected_stop_price = _coerce_float(trade.get("current_stop_price"), math.nan)
    if not math.isfinite(expected_stop_price):
        expected_stop_price = _derive_live_stop_price(
            entry_price,
            _coerce_float(trade.get("sl_dist"), math.nan),
            side_name,
        )

    expected_target_price = _coerce_float(trade.get("current_target_price"), math.nan)
    if not math.isfinite(expected_target_price):
        expected_target_price = _derive_live_target_price(
            entry_price,
            _coerce_float(trade.get("tp_dist"), math.nan),
            side_name,
        )

    try:
        bracket_state = client.get_live_bracket_state(
            side=side_name,
            size=size,
            reference_price=float(entry_price) if math.isfinite(entry_price) else None,
            expected_stop_price=(
                float(expected_stop_price) if math.isfinite(expected_stop_price) else None
            ),
            expected_target_price=(
                float(expected_target_price) if math.isfinite(expected_target_price) else None
            ),
            prefer_stop_order_id=_coerce_int(trade.get("stop_order_id"), None),
            prefer_target_order_id=_coerce_int(trade.get("target_order_id"), None),
            max_cache_age_sec=max(0.0, float(max_cache_age_sec)),
            force_refresh=force_refresh,
        )
    except Exception as exc:
        logging.debug("ProjectX live bracket refresh failed: %s", exc)
        return {}

    updates: dict[str, Any] = {}
    stop_order_id = _coerce_int(bracket_state.get("stop_order_id"), None)
    if stop_order_id is not None and stop_order_id != _coerce_int(trade.get("stop_order_id"), None):
        updates["stop_order_id"] = stop_order_id

    target_order_id = _coerce_int(bracket_state.get("target_order_id"), None)
    if target_order_id is not None and target_order_id != _coerce_int(trade.get("target_order_id"), None):
        updates["target_order_id"] = target_order_id

    stop_price = _coerce_float(bracket_state.get("stop_price", bracket_state.get("sl_price")), math.nan)
    if math.isfinite(stop_price):
        current_stop_price = _coerce_float(trade.get("current_stop_price"), math.nan)
        if not math.isfinite(current_stop_price) or abs(current_stop_price - stop_price) > 1e-9:
            updates["current_stop_price"] = float(stop_price)

    target_price = _coerce_float(
        bracket_state.get("target_price", bracket_state.get("tp_price")),
        math.nan,
    )
    if math.isfinite(target_price):
        current_target_price = _coerce_float(trade.get("current_target_price"), math.nan)
        if not math.isfinite(current_target_price) or abs(current_target_price - target_price) > 1e-9:
            updates["current_target_price"] = float(target_price)

    return updates


def _build_live_active_trade(
    signal: Optional[dict],
    order_details: Optional[dict],
    current_price: float,
    current_time: datetime.datetime,
    bar_count: int,
    *,
    market_df: Optional[pd.DataFrame] = None,
    stop_order_id: Optional[int] = None,
) -> dict:
    signal = signal if isinstance(signal, dict) else {}
    order_details = order_details if isinstance(order_details, dict) else {}

    entry_price = _coerce_float(order_details.get("entry_price", current_price), current_price)
    tp_dist_raw = order_details.get("tp_points")
    if tp_dist_raw is None:
        tp_dist_raw = signal.get("tp_dist")
    sl_dist_raw = order_details.get("sl_points")
    if sl_dist_raw is None:
        sl_dist_raw = signal.get("sl_dist")
    if tp_dist_raw is None or sl_dist_raw is None:
        logging.error("Order details missing sl/tp after execution; using 0.0 for tracking")
    tp_dist = _coerce_float(tp_dist_raw or 0.0, 0.0)
    sl_dist = _coerce_float(sl_dist_raw or 0.0, 0.0)
    raw_size = _coerce_int(order_details.get("size", signal.get("size", 5)), 5)
    size = raw_size if raw_size <= 0 else max(1, raw_size)
    side = str(signal.get("side", "") or "").upper()

    signal["tp_dist"] = tp_dist
    signal["sl_dist"] = sl_dist
    signal["size"] = size
    signal["entry_price"] = entry_price
    if stop_order_id is not None:
        signal["stop_order_id"] = stop_order_id

    stop_price = _coerce_float(order_details.get("sl_price"), math.nan)
    if not math.isfinite(stop_price):
        stop_price = _derive_live_stop_price(entry_price, sl_dist, side)
    target_price = _coerce_float(order_details.get("tp_price"), math.nan)
    if not math.isfinite(target_price):
        target_price = _derive_live_target_price(entry_price, tp_dist, side)

    runtime_cfg = (
        (CONFIG.get("DE3_V4", {}) or {}).get("runtime", {})
        if isinstance((CONFIG.get("DE3_V4", {}) or {}).get("runtime", {}), dict)
        else {}
    )
    trade_management_cfg = (
        runtime_cfg.get("trade_management", {})
        if isinstance(runtime_cfg.get("trade_management", {}), dict)
        else {}
    )
    break_even_cfg = (
        trade_management_cfg.get("break_even", {})
        if isinstance(trade_management_cfg.get("break_even", {}), dict)
        else {}
    )
    early_exit_cfg = (
        trade_management_cfg.get("early_exit", {})
        if isinstance(trade_management_cfg.get("early_exit", {}), dict)
        else {}
    )
    profit_milestone_cfg = (
        trade_management_cfg.get("profit_milestone_stop", {})
        if isinstance(trade_management_cfg.get("profit_milestone_stop", {}), dict)
        else {}
    )
    entry_trade_day_extreme_cfg = (
        trade_management_cfg.get("entry_trade_day_extreme_stop", {})
        if isinstance(trade_management_cfg.get("entry_trade_day_extreme_stop", {}), dict)
        else {}
    )
    de3_trade_management_enabled = bool(
        trade_management_cfg.get("enabled", False)
        and _is_de3_v4_trade_management_payload(signal)
    )
    de3_break_even_enabled = bool(
        de3_trade_management_enabled
        and break_even_cfg.get("enabled", False)
    )
    de3_global_early_exit_enabled = bool(
        de3_trade_management_enabled
        and early_exit_cfg.get("enabled", False)
    )
    de3_profit_milestone_ctx = _resolve_live_de3_profit_milestone_profile(
        signal,
        entry_price=entry_price,
        tp_dist=tp_dist,
        break_even_cfg=break_even_cfg,
        profit_milestone_cfg=profit_milestone_cfg,
    )
    de3_entry_trade_day_extreme_ctx = _resolve_live_de3_entry_trade_day_extreme_profile(
        signal,
        entry_price=entry_price,
        tp_dist=tp_dist,
        current_time=current_time,
        current_price=current_price,
        market_df=market_df,
        break_even_cfg=break_even_cfg,
        entry_trade_day_extreme_cfg=entry_trade_day_extreme_cfg,
    )
    de3_entry_trade_day_extreme_early_exit_ctx = (
        _resolve_live_de3_entry_trade_day_extreme_early_exit_profile(
            signal,
            entry_price=entry_price,
            tp_dist=tp_dist,
            current_time=current_time,
            current_price=current_price,
            market_df=market_df,
        )
    )
    de3_early_exit_enabled = bool(
        de3_global_early_exit_enabled
        or de3_entry_trade_day_extreme_early_exit_ctx.get("active", False)
    )

    active_trade = {
        "strategy": signal.get("strategy", "Unknown"),
        "sub_strategy": signal.get("sub_strategy"),
        "combo_key": signal.get("combo_key") or signal.get("sub_strategy"),
        "side": side,
        "entry_price": entry_price,
        "entry_time": current_time,
        "entry_order_id": (
            order_details.get("broker_order_id")
            or order_details.get("order_id")
        ),
        "target_order_id": order_details.get("target_order_id"),
        "entry_bar": bar_count,
        "bars_held": 0,
        "tp_dist": tp_dist,
        "sl_dist": sl_dist,
        "size": size,
        "stop_order_id": stop_order_id,
        "current_stop_price": stop_price,
        "current_target_price": float(target_price) if math.isfinite(target_price) else None,
        "entry_mode": signal.get("entry_mode", "standard"),
        "rule_id": signal.get("rule_id") or signal.get("de3_v4_selected_route_id"),
        "early_exit_enabled": signal.get("early_exit_enabled"),
        "vol_regime": signal.get("vol_regime"),
        "gate_prob": signal.get("gate_prob"),
        "gate_threshold": signal.get("gate_threshold"),
        "kalshi_trade_overlay_applied": bool(signal.get("kalshi_trade_overlay_applied", False)),
        "kalshi_trade_overlay_reason": signal.get("kalshi_trade_overlay_reason"),
        "kalshi_trade_overlay_role": signal.get("kalshi_trade_overlay_role"),
        "kalshi_trade_overlay_mode": signal.get("kalshi_trade_overlay_mode"),
        "kalshi_trade_overlay_forward_weight": signal.get("kalshi_trade_overlay_forward_weight"),
        "kalshi_curve_informative": bool(signal.get("kalshi_curve_informative", False)),
        "kalshi_entry_probability": signal.get("kalshi_entry_probability"),
        "kalshi_probe_price": signal.get("kalshi_probe_price"),
        "kalshi_probe_probability": signal.get("kalshi_probe_probability"),
        "kalshi_momentum_delta": signal.get("kalshi_momentum_delta"),
        "kalshi_momentum_retention": signal.get("kalshi_momentum_retention"),
        "kalshi_entry_support_score": signal.get("kalshi_entry_support_score"),
        "kalshi_entry_threshold": signal.get("kalshi_entry_threshold"),
        "kalshi_entry_directional_distance_points": signal.get("kalshi_entry_directional_distance_points"),
        "kalshi_sentiment_momentum": signal.get("kalshi_sentiment_momentum"),
        "kalshi_support_price": signal.get("kalshi_support_price"),
        "kalshi_fade_price": signal.get("kalshi_fade_price"),
        "kalshi_tp_anchor_price": signal.get("kalshi_tp_anchor_price"),
        "kalshi_tp_anchor_probability": signal.get("kalshi_tp_anchor_probability"),
        "kalshi_fade_reason": signal.get("kalshi_fade_reason"),
        "kalshi_support_span_points": signal.get("kalshi_support_span_points"),
        "kalshi_tp_trail_enabled": bool(signal.get("kalshi_tp_trail_enabled", False)),
        "kalshi_tp_trigger_price": signal.get("kalshi_tp_trigger_price"),
        "kalshi_tp_trail_buffer_ticks": int(
            _coerce_int(signal.get("kalshi_tp_trail_buffer_ticks"), 0) or 0
        ),
        "kalshi_tp_trail_triggered": False,
        "kalshi_tp_trail_trigger_count": 0,
        "profit_crosses": 0,
        "was_green": None,
        "mfe_points": 0.0,
        "mae_points": 0.0,
        "rescue_from_strategy": signal.get("rescue_from_strategy"),
        "rescue_from_sub_strategy": signal.get("rescue_from_sub_strategy"),
        "trend_day_tier": signal.get("trend_day_tier"),
        "trend_day_dir": signal.get("trend_day_dir"),
        "de3_trade_management_enabled": de3_trade_management_enabled,
        "de3_break_even_enabled": de3_break_even_enabled,
        "de3_break_even_activate_on_next_bar": bool(
            break_even_cfg.get("activate_on_next_bar", True)
        ),
        "de3_break_even_trigger_pct": float(
            _coerce_float(break_even_cfg.get("trigger_pct"), 0.0)
        ),
        "de3_break_even_buffer_ticks": int(
            max(0, _coerce_int(break_even_cfg.get("buffer_ticks"), 0))
        ),
        "de3_break_even_trail_pct": float(
            max(0.0, _coerce_float(break_even_cfg.get("trail_pct"), 0.0))
        ),
        "de3_break_even_post_profit_milestone_trail_pct": float(
            max(
                0.0,
                _coerce_float(
                    de3_profit_milestone_ctx.get("post_reach_trail_pct"),
                    0.0,
                ),
            )
        ),
        "de3_break_even_post_entry_trade_day_extreme_trail_pct": float(
            max(
                0.0,
                _coerce_float(
                    de3_entry_trade_day_extreme_ctx.get("post_reach_trail_pct"),
                    0.0,
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
        "de3_entry_trade_day_extreme_size_adjustment_active": bool(
            signal.get("de3_entry_trade_day_extreme_size_adjustment_active", False)
        ),
        "de3_entry_trade_day_extreme_size_adjustment_applied": bool(
            signal.get("de3_entry_trade_day_extreme_size_adjustment_applied", False)
        ),
        "de3_entry_trade_day_extreme_size_adjustment_profile_name": str(
            signal.get("de3_entry_trade_day_extreme_size_adjustment_profile_name", "") or ""
        ),
        "de3_entry_trade_day_extreme_size_adjustment_target_beyond": bool(
            signal.get("de3_entry_trade_day_extreme_size_adjustment_target_beyond", False)
        ),
        "de3_entry_trade_day_extreme_size_adjustment_progress_pct": signal.get(
            "de3_entry_trade_day_extreme_size_adjustment_progress_pct"
        ),
        "de3_entry_trade_day_extreme_size_adjustment_requested_size": int(
            max(
                1,
                _coerce_int(
                    signal.get(
                        "de3_entry_trade_day_extreme_size_adjustment_requested_size",
                        size,
                    ),
                    size,
                ),
            )
        ),
        "de3_entry_trade_day_extreme_size_adjustment_final_size": int(
            max(
                1,
                _coerce_int(
                    signal.get(
                        "de3_entry_trade_day_extreme_size_adjustment_final_size",
                        size,
                    ),
                    size,
                ),
            )
        ),
        "de3_entry_trade_day_extreme_size_adjustment_multiplier": float(
            max(
                0.0,
                _coerce_float(
                    signal.get("de3_entry_trade_day_extreme_size_adjustment_multiplier"),
                    1.0,
                ),
            )
        ),
        "de3_entry_trade_day_extreme_size_adjustment_min_contracts": int(
            max(
                1,
                _coerce_int(
                    signal.get("de3_entry_trade_day_extreme_size_adjustment_min_contracts"),
                    1,
                ),
            )
        ),
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
            int(max(0, _coerce_int(early_exit_cfg.get("exit_if_not_green_by"), 0)))
            if de3_global_early_exit_enabled
            else None
        ),
        "de3_early_exit_max_profit_crosses": (
            int(max(0, _coerce_int(early_exit_cfg.get("max_profit_crosses"), 0)))
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
            if de3_entry_trade_day_extreme_early_exit_ctx.get("min_progress_by_bars")
            is not None
            else None
        ),
        "de3_early_exit_min_progress_pct": (
            float(
                _coerce_float(
                    de3_entry_trade_day_extreme_early_exit_ctx.get("min_progress_pct"),
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
            if de3_entry_trade_day_extreme_early_exit_ctx.get("max_profit_crosses")
            is not None
            else None
        ),
        "de3_early_exit_trigger_reason": "",
    }
    for key in (
        "de3_version",
        "de3_v4_selected_variant_id",
        "de3_v4_selected_lane",
        "de3_v4_selected_route_id",
    ):
        if key in signal:
            active_trade[key] = signal.get(key)
    return active_trade


def _update_live_trade_mfe_mae(
    trade: Optional[dict],
    bar_high: float,
    bar_low: float,
) -> None:
    if not isinstance(trade, dict):
        return
    side_name = str(trade.get("side", "") or "").upper()
    entry_price = _coerce_float(trade.get("entry_price"), math.nan)
    bar_high = _coerce_float(bar_high, math.nan)
    bar_low = _coerce_float(bar_low, math.nan)
    if (
        side_name not in {"LONG", "SHORT"}
        or not math.isfinite(entry_price)
        or not math.isfinite(bar_high)
        or not math.isfinite(bar_low)
    ):
        return
    if side_name == "LONG":
        mfe_points = max(0.0, float(bar_high - entry_price))
        mae_points = max(0.0, float(entry_price - bar_low))
    else:
        mfe_points = max(0.0, float(entry_price - bar_low))
        mae_points = max(0.0, float(bar_high - entry_price))
    trade["mfe_points"] = max(_coerce_float(trade.get("mfe_points"), 0.0), mfe_points)
    trade["mae_points"] = max(_coerce_float(trade.get("mae_points"), 0.0), mae_points)


def _current_live_de3_break_even_trail_pct(trade: Optional[dict]) -> float:
    if not isinstance(trade, dict):
        return 0.0
    trail_pct = max(0.0, _coerce_float(trade.get("de3_break_even_trail_pct"), 0.0))
    post_profit_milestone_trail_pct = max(
        0.0,
        _coerce_float(
            trade.get("de3_break_even_post_profit_milestone_trail_pct"),
            0.0,
        ),
    )
    post_entry_trade_day_extreme_trail_pct = max(
        0.0,
        _coerce_float(
            trade.get("de3_break_even_post_entry_trade_day_extreme_trail_pct"),
            0.0,
        ),
    )
    if bool(trade.get("de3_profit_milestone_reached", False)):
        trail_pct = max(trail_pct, post_profit_milestone_trail_pct)
    if bool(trade.get("de3_entry_trade_day_extreme_reached", False)):
        trail_pct = max(trail_pct, post_entry_trade_day_extreme_trail_pct)
    return float(trail_pct)


def _update_live_de3_profit_milestone_state(
    trade: Optional[dict],
    *,
    bar_high: float,
    bar_low: float,
    bar_index: Optional[int] = None,
) -> None:
    if not isinstance(trade, dict):
        return
    if not bool(trade.get("de3_profit_milestone_profile_active", False)):
        return
    if bool(trade.get("de3_profit_milestone_reached", False)):
        return
    side_name = str(trade.get("side", "") or "").upper()
    milestone_price = _coerce_float(
        trade.get("de3_profit_milestone_price"),
        math.nan,
    )
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
    trade["de3_profit_milestone_reached_bar_index"] = bar_index
    trade["de3_profit_milestone_reached_mfe_points"] = float(
        max(
            0.0,
            _coerce_float(trade.get("mfe_points"), 0.0),
        )
    )


def _update_live_de3_entry_trade_day_extreme_state(
    trade: Optional[dict],
    *,
    bar_high: float,
    bar_low: float,
    bar_index: Optional[int] = None,
) -> None:
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
    trade["de3_entry_trade_day_extreme_reached_bar_index"] = bar_index
    trade["de3_entry_trade_day_extreme_reached_mfe_points"] = float(
        max(
            0.0,
            _coerce_float(trade.get("mfe_points"), 0.0),
        )
    )


def _live_market_has_crossed_stop(
    stop_price: float,
    side: str,
    market_price: float,
) -> bool:
    stop_price = _coerce_float(stop_price, math.nan)
    market_price = _coerce_float(market_price, math.nan)
    side_name = str(side or "").upper()
    if side_name not in {"LONG", "SHORT"}:
        return False
    if not math.isfinite(stop_price) or not math.isfinite(market_price):
        return False
    if side_name == "LONG":
        return bool(market_price <= stop_price + 1e-12)
    return bool(market_price >= stop_price - 1e-12)


def _live_bar_has_crossed_stop(
    stop_price: float,
    side: str,
    bar_high: float,
    bar_low: float,
) -> bool:
    stop_price = _coerce_float(stop_price, math.nan)
    bar_high = _coerce_float(bar_high, math.nan)
    bar_low = _coerce_float(bar_low, math.nan)
    side_name = str(side or "").upper()
    if side_name not in {"LONG", "SHORT"}:
        return False
    if not math.isfinite(stop_price):
        return False
    if side_name == "LONG" and math.isfinite(bar_low):
        return bool(bar_low <= stop_price + 1e-12)
    if side_name == "SHORT" and math.isfinite(bar_high):
        return bool(bar_high >= stop_price - 1e-12)
    return False


def _force_close_live_trade_for_crossed_stop(
    client: Optional[ProjectXClient],
    trade: Optional[dict],
    current_time: Optional[datetime.datetime],
    *,
    market_price: float,
    target_stop_price: float,
    failure_reason: str,
) -> dict:
    if client is None or not isinstance(trade, dict):
        return {"status": "failed"}

    side_name = str(trade.get("side", "") or "").upper()
    strategy_name = str(trade.get("strategy", "DynamicEngine3") or "DynamicEngine3")
    effective_time = (
        current_time
        if isinstance(current_time, datetime.datetime)
        else datetime.datetime.now(datetime.timezone.utc)
    )
    fallback_exit_price = _coerce_float(market_price, math.nan)
    if not math.isfinite(fallback_exit_price):
        fallback_exit_price = float(target_stop_price)

    position = client.get_position()
    if position.get("stale"):
        logging.warning(
            "DE3 break-even stop %.2f already crossed for %s %s, but broker position is stale; keeping trade live until sync recovers.",
            float(target_stop_price),
            strategy_name,
            side_name,
        )
        return {"status": "failed"}

    if position.get("side") is not None and position.get("size", 0):
        logging.warning(
            "DE3 break-even stop %.2f already crossed by market %.2f for %s %s; flattening instead of leaving the trade unprotected (%s).",
            float(target_stop_price),
            float(fallback_exit_price),
            strategy_name,
            side_name,
            failure_reason,
        )
        close_ok = client.close_trade_leg(trade)
        if not close_ok:
            close_ok = client.close_position(position)
        if not close_ok:
            logging.warning(
                "Emergency close failed after crossed DE3 stop update for %s %s.",
                strategy_name,
                side_name,
            )
            return {"status": "failed"}
        close_order_details = getattr(client, "_last_close_order_details", None) or {}
        close_metrics = _reconcile_live_trade_close(
            client,
            trade,
            effective_time,
            fallback_exit_price=fallback_exit_price,
            close_order_id=close_order_details.get("order_id"),
        )
        if not isinstance(close_metrics, dict):
            close_metrics = _calculate_live_trade_close_metrics_from_price(
                trade,
                _coerce_float(close_order_details.get("exit_price"), fallback_exit_price),
                source=str(close_order_details.get("method") or "crossed_stop_fallback"),
                exit_time=effective_time,
                order_id=_coerce_int(close_order_details.get("order_id"), None),
            )
        return {
            "status": "closed",
            "close_metrics": close_metrics,
            "log_prefix": "Trade closed (crossed stop fallback)",
        }

    logging.info(
        "Broker already flat while DE3 stop %.2f is crossed for %s %s; finalizing local trade state.",
        float(target_stop_price),
        strategy_name,
        side_name,
    )
    try:
        cancelled = client.cancel_open_exit_orders(
            side=None,
            reason="crossed-stop flat cleanup",
        )
        if cancelled:
            logging.info("Cancelled %s orphan exit order(s) after crossed-stop flat detection", cancelled)
    except Exception as exc:
        logging.warning("Exit-order cleanup after crossed-stop flat detection failed: %s", exc)
    client._local_position = {"side": None, "size": 0, "avg_price": 0.0}
    client._active_stop_order_id = None
    close_metrics = _reconcile_live_trade_close(
        client,
        trade,
        effective_time,
        fallback_exit_price=fallback_exit_price,
    )
    return {
        "status": "closed",
        "close_metrics": close_metrics,
        "log_prefix": "Trade closed (crossed stop fallback)",
    }


def _apply_live_de3_break_even_stop_update(
    client: Optional[ProjectXClient],
    trade: Optional[dict],
    new_stop_price: float,
    *,
    from_pending: bool,
    bar_index: Optional[int] = None,
) -> dict:
    if client is None or not isinstance(trade, dict):
        return {"status": "failed"}
    side_name = str(trade.get("side", "") or "").upper()
    if side_name not in {"LONG", "SHORT"}:
        return {"status": "failed"}
    entry_price = _coerce_float(trade.get("entry_price"), math.nan)
    current_stop_price = _coerce_float(trade.get("current_stop_price"), math.nan)
    if not math.isfinite(current_stop_price):
        current_stop_price = _derive_live_stop_price(
            entry_price,
            _coerce_float(trade.get("sl_dist"), math.nan),
            side_name,
        )
        if math.isfinite(current_stop_price):
            trade["current_stop_price"] = float(current_stop_price)
    if not math.isfinite(current_stop_price):
        return {"status": "failed"}

    target_stop_price = _align_stop_price_to_tick(
        _coerce_float(new_stop_price, math.nan),
        side_name,
    )
    if not math.isfinite(target_stop_price):
        return {"status": "failed"}

    take_price = _coerce_float(trade.get("current_target_price"), math.nan)
    if not math.isfinite(take_price):
        take_price = _derive_live_target_price(
            entry_price,
            _coerce_float(trade.get("tp_dist"), math.nan),
            side_name,
        )
    if math.isfinite(entry_price) and math.isfinite(take_price):
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
        return {"status": "unchanged"}

    raw_stop_order_id = trade.get("stop_order_id")
    try:
        stop_order_id = int(raw_stop_order_id) if raw_stop_order_id is not None else None
    except Exception:
        stop_order_id = None

    current_time = trade.get("_management_current_time")
    market_price = _coerce_float(
        trade.get("_management_market_price"),
        math.nan,
    )
    bar_high = _coerce_float(trade.get("_management_bar_high"), math.nan)
    bar_low = _coerce_float(trade.get("_management_bar_low"), math.nan)

    if _live_market_has_crossed_stop(target_stop_price, side_name, market_price):
        return _force_close_live_trade_for_crossed_stop(
            client,
            trade,
            current_time,
            market_price=market_price,
            target_stop_price=target_stop_price,
            failure_reason="market_already_through_new_stop",
        )

    if not client.modify_stop_to_breakeven(
        stop_price=target_stop_price,
        side=side_name,
        known_size=max(1, _coerce_int(trade.get("size"), 1)),
        stop_order_id=stop_order_id,
        current_stop_price=current_stop_price,
    ):
        if (
            _live_market_has_crossed_stop(target_stop_price, side_name, market_price)
            or _live_bar_has_crossed_stop(target_stop_price, side_name, bar_high, bar_low)
        ):
            return _force_close_live_trade_for_crossed_stop(
                client,
                trade,
                current_time,
                market_price=market_price,
                target_stop_price=target_stop_price,
                failure_reason="stop_update_failed_after_cross",
            )
        logging.warning(
            "DE3 v4 break-even stop update failed for %s @ %.2f",
            str(trade.get("strategy", "DynamicEngine3") or "DynamicEngine3"),
            float(target_stop_price),
        )
        return {"status": "failed"}

    already_applied = bool(trade.get("de3_break_even_applied", False))
    trade["current_stop_price"] = float(target_stop_price)
    trade["de3_effective_stop_price"] = float(target_stop_price)
    trade["de3_break_even_last_stop_price"] = float(target_stop_price)
    trade["de3_break_even_armed"] = True
    trade["de3_break_even_applied"] = True
    trade["de3_break_even_move_count"] = int(
        _coerce_int(trade.get("de3_break_even_move_count"), 0)
    ) + 1
    trade["de3_break_even_last_update_bar_index"] = bar_index
    if trade.get("de3_break_even_first_applied_bar_index") is None:
        trade["de3_break_even_first_applied_bar_index"] = bar_index
    if from_pending:
        trade["de3_break_even_pending_stop_price"] = None
        trade["de3_break_even_pending_from_bar_index"] = None
    updated_stop_order_id = getattr(client, "_active_stop_order_id", None)
    if updated_stop_order_id is not None:
        trade["stop_order_id"] = updated_stop_order_id
    if not already_applied:
        logging.info(
            "DE3 v4 break-even armed: %s %s -> %.2f",
            str(trade.get("strategy", "DynamicEngine3") or "DynamicEngine3"),
            side_name,
            float(target_stop_price),
        )
    return {
        "status": "updated",
        "target_stop_price": float(target_stop_price),
    }


def _apply_pending_live_de3_break_even_stop_update(
    client: Optional[ProjectXClient],
    trade: Optional[dict],
    *,
    bar_index: Optional[int] = None,
) -> Optional[dict]:
    if not isinstance(trade, dict):
        return None
    pending_stop_price = _coerce_float(
        trade.get("de3_break_even_pending_stop_price"),
        math.nan,
    )
    if not math.isfinite(pending_stop_price):
        return None
    return _apply_live_de3_break_even_stop_update(
        client,
        trade,
        pending_stop_price,
        from_pending=True,
        bar_index=bar_index,
    )


def _stage_live_de3_break_even_stop_update(
    client: Optional[ProjectXClient],
    trade: Optional[dict],
    *,
    bar_index: Optional[int] = None,
) -> Optional[dict]:
    if client is None or not isinstance(trade, dict):
        return None
    if not bool(trade.get("de3_break_even_enabled", False)):
        return None
    side_name = str(trade.get("side", "") or "").upper()
    entry_price = _coerce_float(trade.get("entry_price"), math.nan)
    current_stop_price = _coerce_float(trade.get("current_stop_price"), math.nan)
    if not math.isfinite(current_stop_price):
        current_stop_price = _derive_live_stop_price(
            entry_price,
            _coerce_float(trade.get("sl_dist"), math.nan),
            side_name,
        )
        if math.isfinite(current_stop_price):
            trade["current_stop_price"] = float(current_stop_price)
    tp_dist = _coerce_float(trade.get("tp_dist"), math.nan)
    if (
        side_name not in {"LONG", "SHORT"}
        or not math.isfinite(entry_price)
        or not math.isfinite(current_stop_price)
        or not math.isfinite(tp_dist)
        or tp_dist <= 0.0
    ):
        return None

    trigger_pct = max(0.0, _coerce_float(trade.get("de3_break_even_trigger_pct"), 0.0))
    trail_pct = _current_live_de3_break_even_trail_pct(trade)
    buffer_ticks = max(0, _coerce_int(trade.get("de3_break_even_buffer_ticks"), 0))
    activate_on_next_bar = bool(trade.get("de3_break_even_activate_on_next_bar", True))
    mfe_points = max(0.0, _coerce_float(trade.get("mfe_points"), 0.0))
    trigger_points = max(0.0, tp_dist * trigger_pct)
    allow_after_profit_milestone = bool(
        trade.get("de3_profit_milestone_reached", False)
        and trade.get("de3_profit_milestone_force_break_even", False)
    )
    allow_after_entry_trade_day_extreme = bool(
        trade.get("de3_entry_trade_day_extreme_reached", False)
        and trade.get("de3_entry_trade_day_extreme_force_break_even", False)
    )
    if (
        trigger_points > 0.0
        and not allow_after_profit_milestone
        and not allow_after_entry_trade_day_extreme
        and mfe_points + 1e-9 < trigger_points
    ):
        return None

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

    trade["de3_break_even_armed"] = True
    trade["de3_break_even_trigger_price"] = (
        entry_price + trigger_points if side_name == "LONG" else entry_price - trigger_points
    )
    if trade.get("de3_break_even_trigger_bar_index") is None:
        trade["de3_break_even_trigger_bar_index"] = bar_index
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
            trade["de3_break_even_pending_from_bar_index"] = bar_index
        return None

    return _apply_live_de3_break_even_stop_update(
        client,
        trade,
        candidate_stop_price,
        from_pending=False,
        bar_index=bar_index,
    )


def _apply_rl_management_action(
    client: Optional["ProjectXClient"],
    trade: Optional[dict],
    action: int,
    action_name: str,
    *,
    current_time: "datetime.datetime",
    market_price: float,
    bar_high: float,
    bar_low: float,
    bar_index: Optional[int] = None,
) -> Optional[dict]:
    """Execute an RL-policy action against the live trade.

    For safety we only execute the SL-modification actions here:
      MOVE_SL_TO_BE, TIGHTEN_SL_25PCT, TIGHTEN_SL_50PCT

    TAKE_PARTIAL_* and REVERSE are intentionally NOT wired yet — they
    require dedicated partial-close / flatten-and-reverse broker paths
    that need careful testing against the bot's own trade tracking. The
    shadow log still records the policy's preference for those actions
    so operators can observe how often they fire and make an informed
    decision before wiring them.

    Returns a dict of form:
      {"status": "applied", "new_sl": <price>}   on success
      {"status": "deferred", "reason": "..."}     when the action isn't
                                                    wired for live execution
      {"status": "skipped",  "reason": "..."}     when the action is wired
                                                    but current state makes
                                                    it a no-op (ratchet check
                                                    fails, already at BE, etc.)
      None                                         on error
    """
    if not isinstance(trade, dict):
        return None

    # Actions RL can fire (integers mirror rl/trade_env.py constants)
    ACT_HOLD = 0
    ACT_MOVE_SL_TO_BE = 1
    ACT_TIGHTEN_SL_25 = 2
    ACT_TIGHTEN_SL_50 = 3
    ACT_TAKE_PARTIAL_50 = 4
    ACT_TAKE_PARTIAL_FULL = 5
    ACT_REVERSE = 6

    if action == ACT_HOLD:
        return {"status": "applied", "reason": "hold"}

    side = str(trade.get("side", "")).upper()
    try:
        entry = float(trade.get("entry_price", 0.0) or 0.0)
        current_sl = float(trade.get("sl_price") or trade.get("effective_stop_price") or entry)
    except Exception:
        return None
    if entry <= 0 or side not in ("LONG", "SHORT"):
        return None

    target = None
    if action == ACT_MOVE_SL_TO_BE:
        target = entry + TICK_SIZE if side == "LONG" else entry - TICK_SIZE
    elif action == ACT_TIGHTEN_SL_25:
        if side == "LONG":
            dist = market_price - current_sl
            if dist > 0:
                target = current_sl + dist * 0.25
        else:
            dist = current_sl - market_price
            if dist > 0:
                target = current_sl - dist * 0.25
    elif action == ACT_TIGHTEN_SL_50:
        if side == "LONG":
            dist = market_price - current_sl
            if dist > 0:
                target = current_sl + dist * 0.50
        else:
            dist = current_sl - market_price
            if dist > 0:
                target = current_sl - dist * 0.50

    if target is not None:
        # Round to tick and enforce ratchet (don't loosen)
        target = round(target / TICK_SIZE) * TICK_SIZE
        if side == "LONG" and target <= current_sl + 1e-9:
            return {"status": "skipped", "reason": f"ratchet_would_not_improve {target}<={current_sl}"}
        if side == "SHORT" and target >= current_sl - 1e-9:
            return {"status": "skipped", "reason": f"ratchet_would_not_improve {target}>={current_sl}"}
        # Refuse to move SL to a price already breached by current market —
        # otherwise the broker force-closes at market immediately (seen live
        # on 2026-04-22: RL fired MOVE_SL_TO_BE on trades that were 0.25–1.25
        # pts underwater, the executor dutifully shipped the new SL, and the
        # broker instant-filled the exit as if the stop had hit. Trades that
        # would have recovered within a few bars got chopped at a small loss).
        # Required buffer: ≥1 tick between new SL and current market, on the
        # protective side.
        try:
            mp = float(market_price)
        except Exception:
            mp = 0.0
        if mp > 0:
            if side == "LONG" and target >= mp - TICK_SIZE:
                return {"status": "skipped",
                        "reason": f"would_breach_market SL={target} >= mkt-tick={mp - TICK_SIZE}"}
            if side == "SHORT" and target <= mp + TICK_SIZE:
                return {"status": "skipped",
                        "reason": f"would_breach_market SL={target} <= mkt+tick={mp + TICK_SIZE}"}
        result = _apply_pivot_trail_sl(
            client, trade, float(target),
            current_time=current_time, market_price=market_price,
            bar_high=bar_high, bar_low=bar_low, bar_index=bar_index,
        )
        if isinstance(result, dict) and result.get("status") == "updated":
            logging.info(
                "[RL_LIVE] %s %s SL → %.2f (action=%s)",
                str(trade.get("strategy", "")), side, float(target), action_name,
            )
            return {"status": "applied", "new_sl": float(target), "action": action_name,
                    "inner": result}
        return result

    # Partial close / reverse actions — deferred; not wired yet
    if action in (ACT_TAKE_PARTIAL_50, ACT_TAKE_PARTIAL_FULL, ACT_REVERSE):
        return {"status": "deferred",
                "reason": f"{action_name} needs dedicated broker path — shadow-only"}

    return {"status": "skipped", "reason": f"unknown_action_{action}"}


def _apply_pivot_trail_sl(
    client: Optional["ProjectXClient"],
    trade: Optional[dict],
    target_stop_price: float,
    *,
    current_time: "datetime.datetime",
    market_price: float,
    bar_high: float,
    bar_low: float,
    bar_index: Optional[int] = None,
) -> Optional[dict]:
    """Move the hard stop to *target_stop_price* via the broker API.

    Reuses _apply_live_de3_break_even_stop_update() after injecting the
    required management-context fields into the trade dict.  The ratchet
    check (target must improve current SL) is already done by the caller
    via _compute_pivot_trail_sl(), but the underlying function double-checks.
    """
    if not isinstance(trade, dict):
        return None
    trade["_management_current_time"] = current_time
    trade["_management_market_price"] = float(market_price)
    trade["_management_bar_high"] = float(bar_high)
    trade["_management_bar_low"] = float(bar_low)
    try:
        result = _apply_live_de3_break_even_stop_update(
            client,
            trade,
            target_stop_price,
            from_pending=False,
            bar_index=bar_index,
        )
        if isinstance(result, dict) and result.get("status") == "updated":
            logging.info(
                "[PivotTrail] %s %s SL → %.2f",
                str(trade.get("strategy", "")),
                str(trade.get("side", "")),
                float(target_stop_price),
            )
        return result
    finally:
        trade.pop("_management_current_time", None)
        trade.pop("_management_market_price", None)
        trade.pop("_management_bar_high", None)
        trade.pop("_management_bar_low", None)


def _process_live_trade_management_bar(
    client: Optional[ProjectXClient],
    trade: Optional[dict],
    *,
    current_time: datetime.datetime,
    market_price: float,
    bar_high: float,
    bar_low: float,
    bar_index: Optional[int] = None,
) -> Optional[dict]:
    if not isinstance(trade, dict):
        return None
    trade["_management_current_time"] = current_time
    trade["_management_market_price"] = float(market_price)
    trade["_management_bar_high"] = float(bar_high)
    trade["_management_bar_low"] = float(bar_low)
    try:
        _update_live_trade_mfe_mae(trade, bar_high, bar_low)
        _update_live_de3_profit_milestone_state(
            trade,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_index=bar_index,
        )
        _update_live_de3_entry_trade_day_extreme_state(
            trade,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_index=bar_index,
        )
        pending_result = _apply_pending_live_de3_break_even_stop_update(
            client,
            trade,
            bar_index=bar_index,
        )
        if isinstance(pending_result, dict) and pending_result.get("status") == "closed":
            return pending_result
        stage_result = _stage_live_de3_break_even_stop_update(
            client,
            trade,
            bar_index=bar_index,
        )
        if isinstance(stage_result, dict) and stage_result.get("status") == "closed":
            return stage_result
        return None
    finally:
        trade.pop("_management_current_time", None)
        trade.pop("_management_market_price", None)
        trade.pop("_management_bar_high", None)
        trade.pop("_management_bar_low", None)


def _current_live_trade_progress_pct(trade: Optional[dict], current_price: float) -> float:
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


def _evaluate_live_early_exit_reason(
    trade: Optional[dict],
    current_price: float,
    early_exit_config: Optional[dict],
) -> Optional[str]:
    if not isinstance(trade, dict):
        return None
    if not isinstance(early_exit_config, dict) or not bool(early_exit_config.get("enabled", False)):
        return None
    side_name = str(trade.get("side", "") or "").upper()
    entry_price = _coerce_float(trade.get("entry_price"), math.nan)
    current_price = _coerce_float(current_price, math.nan)
    if side_name == "LONG":
        is_green = current_price > entry_price
    elif side_name == "SHORT":
        is_green = current_price < entry_price
    else:
        return None
    progress_bars = early_exit_config.get("min_progress_by_bars", None)
    min_progress_pct = early_exit_config.get("min_progress_pct", None)
    if (
        progress_bars is not None
        and int(progress_bars) > 0
        and min_progress_pct is not None
        and int(_coerce_int(trade.get("bars_held"), 0)) >= int(progress_bars)
    ):
        threshold = float(min_progress_pct)
        if threshold <= 1e-12:
            if not is_green:
                return f"not green after {int(progress_bars)} bars"
        else:
            current_progress_pct = _current_live_trade_progress_pct(trade, current_price)
            if (
                not math.isfinite(current_progress_pct)
                or current_progress_pct + 1e-12 < threshold
            ):
                return (
                    f"progress {current_progress_pct:.2%} < {threshold:.2%}"
                    f" after {int(progress_bars)} bars"
                )
    exit_time = early_exit_config.get("exit_if_not_green_by", None)
    if (
        exit_time is not None
        and int(exit_time) > 0
        and int(_coerce_int(trade.get("bars_held"), 0)) >= int(exit_time)
        and not is_green
    ):
        return f"not green after {int(exit_time)} bars"
    profile_exit_cross = early_exit_config.get("profile_max_profit_crosses", None)
    profit_crosses = int(_coerce_int(trade.get("profit_crosses"), 0))
    if profile_exit_cross is not None and profit_crosses > int(profile_exit_cross):
        return f"choppy ({profit_crosses} crosses > {int(profile_exit_cross)})"
    exit_cross = early_exit_config.get("max_profit_crosses", None)
    if exit_cross is not None and profit_crosses > int(exit_cross):
        return f"choppy ({profit_crosses} crosses > {int(exit_cross)})"
    return None


def _check_kalshi_sentiment_exit(
    trade: Optional[dict],
    current_price: float,
    current_time: Optional[datetime.datetime],
) -> Optional[str]:
    """Hour-turn exit: close profitable positions when Kalshi crowd flips.

    Checks only at the top of each hour (first 2 minutes) during settlement
    hours (10 AM - 4 PM ET).  If the crowd sentiment flips against a
    profitable position, returns an exit reason string.
    """
    if not isinstance(trade, dict) or current_time is None:
        return None
    try:
        minute = int(getattr(current_time, "minute", 99))
    except (TypeError, ValueError):
        return None
    if minute > 2:
        return None

    kalshi = _get_kalshi_provider()
    if kalshi is None or not getattr(kalshi, "enabled", False) or not getattr(kalshi, "is_healthy", False):
        return None

    settlement_hour = _active_kalshi_settlement_hour_et(kalshi)
    if settlement_hour not in _KALSHI_GATING_HOURS_ET:
        return None

    side = str(trade.get("side", "") or "").upper()
    entry_price = _coerce_float(trade.get("entry_price"), math.nan)
    if not math.isfinite(entry_price):
        return None
    if side == "LONG" and current_price <= entry_price:
        return None
    if side == "SHORT" and current_price >= entry_price:
        return None

    sentiment = kalshi.get_sentiment(current_price)
    probability = sentiment.get("probability")
    if probability is None:
        return None

    # Only act on high-confidence crowd flips (60%+ = 70% accuracy)
    if side == "LONG" and probability < 0.40:
        return f"KALSHI HOUR-TURN: Crowd flipped bearish (prob={probability:.2f})"
    if side == "SHORT" and probability > 0.60:
        return f"KALSHI HOUR-TURN: Crowd flipped bullish (prob={probability:.2f})"
    return None


def _resolve_live_early_exit_config(trade: Optional[dict]) -> dict:
    if not isinstance(trade, dict):
        return {}
    if _is_de3_v4_trade_management_payload(trade):
        if not bool(trade.get("de3_early_exit_enabled", False)):
            return {}
        return {
            "enabled": True,
            "exit_if_not_green_by": (
                int(
                    max(
                        0,
                        _coerce_int(trade.get("de3_early_exit_exit_if_not_green_by"), 0),
                    )
                )
                if trade.get("de3_early_exit_exit_if_not_green_by") is not None
                else None
            ),
            "max_profit_crosses": (
                int(
                    max(
                        0,
                        _coerce_int(trade.get("de3_early_exit_max_profit_crosses"), 0),
                    )
                )
                if trade.get("de3_early_exit_max_profit_crosses") is not None
                else None
            ),
            "min_progress_by_bars": (
                int(
                    max(
                        0,
                        _coerce_int(trade.get("de3_early_exit_min_progress_by_bars"), 0),
                    )
                )
                if trade.get("de3_early_exit_min_progress_by_bars") is not None
                else None
            ),
            "min_progress_pct": (
                float(_coerce_float(trade.get("de3_early_exit_min_progress_pct"), 0.0))
                if math.isfinite(
                    _coerce_float(
                        trade.get("de3_early_exit_min_progress_pct"),
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
                            trade.get("de3_early_exit_profile_max_profit_crosses"),
                            0,
                        ),
                    )
                )
                if trade.get("de3_early_exit_profile_max_profit_crosses") is not None
                else None
            ),
            "profile_name": str(trade.get("de3_early_exit_profile_name", "") or ""),
        }
    strategy_name = str(trade.get("strategy", "") or "")
    early_exit_cfg = CONFIG.get("EARLY_EXIT", {}).get(strategy_name, {})
    resolved_cfg = dict(early_exit_cfg) if isinstance(early_exit_cfg, dict) else {}

    explicit_enabled = None
    explicit_enabled_raw = trade.get("early_exit_enabled")
    if explicit_enabled_raw is not None:
        if isinstance(explicit_enabled_raw, str):
            lowered = explicit_enabled_raw.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                explicit_enabled = True
            elif lowered in {"0", "false", "no", "off"}:
                explicit_enabled = False
        elif isinstance(explicit_enabled_raw, (bool, int, float)):
            explicit_enabled = bool(explicit_enabled_raw)

    if explicit_enabled is False:
        return {}
    if explicit_enabled is True:
        resolved_cfg["enabled"] = True

    per_trade_not_green = trade.get("early_exit_exit_if_not_green_by")
    if per_trade_not_green is not None:
        resolved_cfg["exit_if_not_green_by"] = int(
            max(0, _coerce_int(per_trade_not_green, 0))
        )

    per_trade_crosses = trade.get("early_exit_max_profit_crosses")
    if per_trade_crosses is not None:
        resolved_cfg["max_profit_crosses"] = int(
            max(0, _coerce_int(per_trade_crosses, 0))
        )

    return resolved_cfg if bool(resolved_cfg.get("enabled", False)) else {}


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


LOG_STRATEGY_ALIASES: Dict[str, Tuple[str, Optional[str]]] = {
    # RegimeAdaptive family
    "RegimeAdaptive": ("RegimeAdaptive", None),
    "AuctionReversion": ("RegimeAdaptive", "Auction Reversion"),
    "SmoothTrendAsia": ("RegimeAdaptive", "Smooth Trend Asia"),
    # SMT/liq sweep group
    "SMTAnalyzer": ("SMT Divergence", None),
    "SMTStrategy": ("SMT Divergence", None),
    "LiquiditySweep": ("SMT Divergence", "Liquidity Sweep"),
    # ORB/breakout group
    "ORB": ("Breakout Strategy", "Opening Range"),
    "ORBStrategy": ("Breakout Strategy", "Opening Range"),
    "ImpulseBreakout": ("Breakout Strategy", "Impulse Breakout"),
    "ValueAreaBreakout": ("Breakout Strategy", "Value Area Breakout"),
    # Dynamic Engine
    "DynamicEngine": ("DynamicEngine", None),
    "DynamicEngine2": ("DynamicEngine", "Dynamic Engine 2"),
    "DynamicEngine3": ("DynamicEngine3", None),
}


def get_log_strategy_info(raw_strategy: Optional[str], signal: Optional[Dict] = None) -> Tuple[str, Optional[str]]:
    """
    Map raw strategy names to UI-visible log labels, with optional sub-strategy
    for RegimeAdaptive family.
    """
    raw = str(raw_strategy) if raw_strategy else ""
    display, sub_strategy = LOG_STRATEGY_ALIASES.get(raw, (raw or "Unknown", None))
    if display == "RegimeAdaptive" and not sub_strategy and signal:
        signal_sub = signal.get("sub_strategy")
        if signal_sub:
            sub_strategy = signal_sub
    return display, sub_strategy


def format_ui_strategy_slot(
    raw_strategy: Optional[str],
    signal: Optional[Dict] = None,
    fallback: Optional[str] = None,
) -> str:
    """
    Render a strategy slot label for UI lists while keeping the same string format.
    """
    base = raw_strategy or fallback or "Unknown"
    display, sub_strategy = get_log_strategy_info(str(base), signal)
    if sub_strategy:
        return f"{display}:{sub_strategy}"
    return str(display)


def get_live_opposite_reversal_family_key(
    signal: Optional[Dict] = None,
    *,
    require_sub_strategy: bool = False,
) -> Optional[str]:
    if not isinstance(signal, dict):
        return None
    raw_strategy = signal.get("strategy")
    display, sub_strategy = get_log_strategy_info(raw_strategy, signal)
    family = str(display or raw_strategy or "").strip()
    if not family:
        return None
    if require_sub_strategy:
        sub_value = str(
            sub_strategy
            or signal.get("sub_strategy")
            or signal.get("combo_key")
            or ""
        ).strip()
        if sub_value:
            return f"{family}:{sub_value}"
    return family


def update_live_opposite_reversal_confirmation_state(
    state: Optional[Dict[str, Any]],
    signal_payload: Optional[Dict],
    current_bar_index: int,
    *,
    required_confirmations: int,
    window_bars: int,
    require_same_strategy_family: bool = True,
    require_same_sub_strategy: bool = False,
) -> Tuple[bool, int, Dict[str, Any]]:
    reset_state = {
        "count": 0,
        "side": None,
        "bar_index": None,
        "strategy_family": None,
    }
    if not isinstance(state, dict):
        state = dict(reset_state)

    side = None
    if isinstance(signal_payload, dict):
        side = _normalize_live_side(signal_payload.get("side"))
    if side is None:
        return False, 0, dict(reset_state)

    strategy_family = None
    if require_same_strategy_family:
        strategy_family = get_live_opposite_reversal_family_key(
            signal_payload,
            require_sub_strategy=require_same_sub_strategy,
        )

    prior_side = _normalize_live_side(state.get("side"))
    prior_count = int(state.get("count", 0) or 0)
    prior_bar_index = _coerce_int(state.get("bar_index"), None)
    prior_family = str(state.get("strategy_family") or "").strip() or None
    window_expired = (
        prior_bar_index is not None
        and (int(current_bar_index) - prior_bar_index) > int(window_bars)
    )
    family_changed = bool(
        require_same_strategy_family and prior_family != strategy_family
    )

    if prior_side != side or window_expired or family_changed:
        new_count = 1
    else:
        new_count = prior_count + 1

    next_state = {
        "count": int(new_count),
        "side": side,
        "bar_index": int(current_bar_index),
        "strategy_family": strategy_family,
    }
    confirmed = int(new_count) >= max(1, int(required_confirmations))
    return confirmed, int(new_count), next_state


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
        guard_cfg = CONFIG.get("CONTINUATION_GUARD", {}) or {}
        key_mode = guard_cfg.get("key_granularity", "full")
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


def _is_de3_signal(signal: Optional[Dict]) -> bool:
    if not isinstance(signal, dict):
        return False
    strategy_name = str(signal.get("strategy") or signal.get("strategy_name") or "").strip().lower()
    normalized = strategy_name.replace("_", "")
    if normalized in ("dynamicengine3", "de3"):
        return True
    return normalized.startswith("dynamicengine3")


def _de3_signal_timeframe(signal: Optional[Dict]) -> Optional[str]:
    if not isinstance(signal, dict):
        return None
    # Only trust fields from the final chosen DE3 signal payload.
    final_candidates = [
        signal.get("de3_timeframe"),
        signal.get("de3_strategy_id"),
        signal.get("sub_strategy"),
    ]
    for raw in final_candidates:
        text = str(raw or "").strip().lower().replace(" ", "")
        if not text:
            continue
        if "15min" in text or "15m" in text or text.startswith("m15"):
            return "15min"
        if "5min" in text or "5m" in text or text.startswith("m5"):
            return "5min"
    return None


def _last_bar_rejection_on_timeframe(df_tf: Optional[pd.DataFrame], side: str) -> Tuple[bool, str]:
    if df_tf is None or df_tf.empty:
        return False, "missing timeframe bars"
    if len(df_tf) < 1:
        return False, "insufficient timeframe bars"

    bar = df_tf.iloc[-1]
    try:
        open_ = float(bar.get("open"))
        high = float(bar.get("high"))
        low = float(bar.get("low"))
        close = float(bar.get("close"))
    except Exception:
        return False, "invalid OHLC on timeframe bar"

    bar_range = high - low
    if not np.isfinite(bar_range) or bar_range <= 0.0:
        return False, "non-positive bar range"

    body = abs(close - open_)
    upper_wick = high - max(open_, close)
    lower_wick = min(open_, close) - low
    wick_threshold = max(body * 0.5, bar_range * 0.25)
    side_u = str(side or "").upper()

    if side_u == "LONG":
        ok = lower_wick > wick_threshold
        return ok, f"lower_wick={lower_wick:.3f} thr={wick_threshold:.3f}"
    if side_u == "SHORT":
        ok = upper_wick > wick_threshold
        return ok, f"upper_wick={upper_wick:.3f} thr={wick_threshold:.3f}"
    return False, "unsupported side"


def _de3_timeframe_rejection_bypass(
    signal: Optional[Dict],
    side: str,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
) -> Tuple[bool, str]:
    if not _is_de3_signal(signal):
        return False, "not DE3"
    tf = _de3_signal_timeframe(signal)
    if tf == "5min":
        ok, detail = _last_bar_rejection_on_timeframe(df_5m, side)
        return ok, f"{tf} {detail}"
    if tf == "15min":
        ok, detail = _last_bar_rejection_on_timeframe(df_15m, side)
        return ok, f"{tf} {detail}"
    return False, "missing DE3 timeframe"


def load_continuation_allowlist(path_value: Optional[str]) -> Optional[set]:
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
    except Exception as e:
        logging.warning(f"⚠️ Continuation allowlist load failed: {e}")
        return None


def load_flip_confidence(path_value: Optional[str]) -> Optional[dict]:
    if not path_value:
        return None
    try:
        path = Path(path_value)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        if not path.exists():
            logging.warning(f"⚠️ Flip confidence file missing: {path}")
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        allowlist = payload.get("allowlist")
        if not isinstance(allowlist, list):
            allowlist = None
        return {
            "allowlist": set(allowlist) if allowlist is not None else None,
            "stats": payload.get("stats", {}) or {},
            "criteria": payload.get("criteria", {}) or {},
            "key_fields": payload.get("key_fields") or [],
            "allowed_filters": payload.get("allowed_filters") or [],
        }
    except Exception as e:
        logging.warning(f"⚠️ Flip confidence load failed: {e}")
        return None


def _flip_filter_candidates(filter_name: str, reason: Optional[str]) -> list[str]:
    name = str(filter_name or "")
    if not name:
        return []
    if name.startswith("FilterStack"):
        parts = name.split(":", 1)
        if len(parts) == 2 and parts[1]:
            return [item for item in parts[1].split("+") if item]
        return ["FilterStack"]
    if name == "Rejection/Bias":
        reason_lc = str(reason or "").lower()
        if "range bias" in reason_lc or "bias" in reason_lc:
            return ["ChopRangeBias"]
        return ["RejectionFilter"]
    if name == "DirectionalLoss":
        return ["DirectionalLossBlocker"]
    return [name]


def _flip_build_key(
    signal: dict,
    filter_name: str,
    session_name: Optional[str],
    key_fields: list[str],
    vol_regime: Optional[str] = None,
) -> str:
    side = str(signal.get("side") or "UNKNOWN")
    mapping = {
        "filter": filter_name,
        "session": session_name or "UNKNOWN",
        "regime": vol_regime or "UNKNOWN",
        "side": side,
        "strategy": str(signal.get("strategy") or "Unknown"),
        "sub_strategy": str(signal.get("sub_strategy") or "None"),
    }
    return "|".join(str(mapping.get(field, "UNKNOWN")) for field in key_fields)


def continuation_market_confirmed(
    side: str,
    bar_close: float,
    trend_day_series: Optional[dict],
    cfg: Optional[dict],
) -> bool:
    if not cfg or not cfg.get("enabled", True):
        return True
    if trend_day_series is None:
        logging.warning("⚠️ Continuation confirm missing trend_day_series")
        return False

    def last_val(key: str, default=None):
        series = trend_day_series.get(key) if isinstance(trend_day_series, dict) else None
        if isinstance(series, pd.Series):
            try:
                return series.iloc[-1]
            except Exception:
                return default
        return default

    use_adx = cfg.get("use_adx", True)
    use_trend_alt = cfg.get("use_trend_alt", True)
    use_vwap = cfg.get("use_vwap", True)
    use_structure = cfg.get("use_structure_break", True)
    vwap_sigma_min = float(cfg.get("vwap_sigma_min", 0.0) or 0.0)
    require_any = cfg.get("require_any", True)

    adx_up = bool(last_val("adx_strong_up", False))
    adx_down = bool(last_val("adx_strong_down", False))
    trend_up = bool(last_val("trend_up_alt", False))
    trend_down = bool(last_val("trend_down_alt", False))
    vwap_sigma = last_val("vwap_sigma_dist", 0.0)
    try:
        vwap_sigma = float(vwap_sigma)
    except Exception:
        vwap_sigma = 0.0

    prior_high = last_val("prior_session_high", None)
    prior_low = last_val("prior_session_low", None)
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


def continuation_core_trigger(filter_name: str) -> bool:
    if not filter_name:
        return False
    if filter_name.startswith("FilterStack"):
        return True
    return filter_name in {"RegimeBlocker", "TrendFilter", "ChopFilter", "ExtensionFilter"}


def _ny_base_session_from_ts(ts) -> Optional[str]:
    if ts is None:
        return None
    try:
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is not None:
                ts = ts.tz_convert(NY_TZ)
            ts = ts.to_pydatetime()
        hour = ts.hour
    except Exception:
        return None
    if hour >= 18 or hour < 3:
        return "ASIA"
    if 3 <= hour < 8:
        return "LONDON"
    if 8 <= hour < 12:
        return "NY_AM"
    if 12 <= hour < 17:
        return "NY_PM"
    return "OFF"


def _runtime_session_labels_from_ts(ts) -> tuple[Optional[str], Optional[str]]:
    if ts is None:
        return None, None
    try:
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is not None:
                ts = ts.tz_convert(NY_TZ)
            ts = ts.to_pydatetime()
        elif isinstance(ts, datetime.datetime) and ts.tzinfo is not None:
            ts = ts.astimezone(NY_TZ)
        hour = ts.hour
        minute = ts.minute
    except Exception:
        return None, None

    if 18 <= hour or hour < 3:
        base_session = "ASIA"
    elif 3 <= hour < 8:
        base_session = "LONDON"
    elif 8 <= hour < 12:
        base_session = "NY_AM"
    elif 12 <= hour < 17:
        base_session = "NY_PM"
    else:
        base_session = "POST_MARKET"

    current_session = base_session
    if base_session == "NY_AM":
        if (hour == 10 and minute >= 30) or hour == 11:
            current_session = "NY_LUNCH"
    elif base_session == "NY_PM":
        if hour == 12 and minute < 30:
            current_session = "NY_LUNCH"
        elif hour >= 15:
            current_session = "NY_CLOSE"

    return base_session, current_session


def _safe_last_value(series: Optional[pd.Series]) -> Optional[float]:
    if series is None:
        return None
    try:
        if isinstance(series, pd.Series):
            value = series.iloc[-1]
        else:
            value = series
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _calc_atr20(df: pd.DataFrame) -> Optional[float]:
    if df.empty or len(df) < 21:
        return None
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr20 = tr.rolling(20).mean().iloc[-1]
    if pd.isna(atr20):
        return None
    return float(atr20)


def _close_position_in_bar(high: float, low: float, close: float) -> float:
    rng = float(high) - float(low)
    if rng <= 0:
        return 0.5
    return (float(close) - float(low)) / rng


def _get_opening_range_levels(df: pd.DataFrame, current_time, minutes: int = 15) -> tuple[Optional[float], Optional[float]]:
    if df.empty or current_time is None:
        return None, None
    try:
        if isinstance(current_time, pd.Timestamp):
            if current_time.tzinfo is not None:
                current_time = current_time.tz_convert(NY_TZ)
            current_time = current_time.to_pydatetime()
        session_start = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        session_end = session_start + datetime.timedelta(minutes=minutes)
        window = df.loc[(df.index >= session_start) & (df.index < session_end)]
        if window.empty:
            return None, None
        return float(window["high"].max()), float(window["low"].min())
    except Exception:
        return None, None


def _session_high_low(df: pd.DataFrame, current_time, session_name: str) -> tuple[Optional[float], Optional[float]]:
    if df.empty or current_time is None:
        return None, None
    try:
        if isinstance(current_time, pd.Timestamp):
            if current_time.tzinfo is not None:
                current_time = current_time.tz_convert(NY_TZ)
            current_time = current_time.to_pydatetime()
        if session_name == "NY_AM":
            start = current_time.replace(hour=8, minute=0, second=0, microsecond=0)
            end = current_time.replace(hour=12, minute=0, second=0, microsecond=0)
        elif session_name == "NY_PM":
            start = current_time.replace(hour=12, minute=0, second=0, microsecond=0)
            end = current_time.replace(hour=17, minute=0, second=0, microsecond=0)
        else:
            return None, None
        window = df.loc[(df.index >= start) & (df.index <= min(current_time, end))]
        if window.empty:
            return None, None
        return float(window["high"].max()), float(window["low"].min())
    except Exception:
        return None, None


def _select_interaction_level(
    levels: Dict[str, float],
    price: float,
    df: pd.DataFrame,
    band: float,
) -> Optional[Tuple[str, float]]:
    if not levels:
        return None
    if df.empty or len(df) < 2:
        return None
    prev_close = float(df["close"].iloc[-2])
    candidates: list[Tuple[str, float]] = []
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


def _balance_metrics(df: pd.DataFrame, lookback: int = 10) -> Optional[Tuple[float, float, int]]:
    if df.empty or len(df) < lookback:
        return None
    window = df.iloc[-lookback:]
    closes = window["close"].to_numpy(dtype=float)
    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    deltas = np.diff(closes)
    sum_abs = float(np.sum(np.abs(deltas)))
    net_move = float(abs(closes[-1] - closes[0]))
    er = net_move / max(sum_abs, 1e-9)

    flips = 0
    signs = np.sign(deltas)
    for idx in range(1, len(signs)):
        if signs[idx] != 0 and signs[idx - 1] != 0 and signs[idx] != signs[idx - 1]:
            flips += 1

    overlap_count = 0
    for idx in range(1, lookback):
        overlap = min(highs[idx], highs[idx - 1]) - max(lows[idx], lows[idx - 1])
        if overlap > 0:
            overlap_count += 1
    overlap_ratio = overlap_count / max(lookback - 1, 1)
    return er, overlap_ratio, flips


def _ny_structure_gates(
    signal: dict,
    side: str,
    bar_close: float,
    df: pd.DataFrame,
    trend_day_series: Optional[dict],
    *,
    require_prefix: Optional[str] = None,
    log_prefix: str = "NYGate",
) -> bool:
    if not signal or df.empty:
        return True
    strat_name = str(signal.get("strategy") or "")
    if require_prefix:
        if not strat_name.lower().startswith(require_prefix.lower()):
            return True
    ts = df.index[-1]
    session_name = _ny_base_session_from_ts(ts)
    if session_name not in ("NY_AM", "NY_PM"):
        return True

    # Gate B: Local balance detector
    metrics = _balance_metrics(df, lookback=10)
    if metrics:
        er, overlap_ratio, flips = metrics
        if er < 0.25 or overlap_ratio > 0.70 or flips >= 6:
            logging.info(
                f"⛔ {log_prefix}Balance blocked | session={session_name} | strategy={strat_name} | "
                f"side={side} | ER={er:.2f} | overlap={overlap_ratio:.2f} | flips={flips}"
            )
            return False

    # Gate A: Acceptance vs rejection at key levels
    atr20 = _calc_atr20(df)
    band = max(0.2 * (atr20 or 0.0), 0.5)
    levels_long: Dict[str, float] = {}
    levels_short: Dict[str, float] = {}

    if trend_day_series:
        levels_long["VWAP"] = _safe_last_value(trend_day_series.get("vwap"))
        levels_short["VWAP"] = _safe_last_value(trend_day_series.get("vwap"))
        levels_long["PDH"] = _safe_last_value(trend_day_series.get("prior_session_high"))
        levels_short["PDL"] = _safe_last_value(trend_day_series.get("prior_session_low"))

    orh, orl = _get_opening_range_levels(df, ts)
    if orh is not None:
        levels_long["ORH"] = orh
    if orl is not None:
        levels_short["ORL"] = orl

    vp = build_volume_profile(df, lookback=120, tick_size=TICK_SIZE)
    if vp:
        levels_long["VAH"] = vp.get("vah")
        levels_short["VAL"] = vp.get("val")

    if side == "LONG":
        chosen = _select_interaction_level(levels_long, bar_close, df, band)
        if chosen is not None:
            level_name, level_val = chosen
            recent = df.iloc[-2:]
            if len(recent) < 2:
                return True
            closes = recent["close"].to_numpy(dtype=float)
            highs = recent["high"].to_numpy(dtype=float)
            lows = recent["low"].to_numpy(dtype=float)
            close_pos = [
                _close_position_in_bar(highs[i], lows[i], closes[i]) for i in range(2)
            ]
            if not all(c >= level_val for c in closes) or not all(pos >= 0.60 for pos in close_pos):
                logging.info(
                    f"⛔ {log_prefix}Acceptance blocked | session={session_name} | strategy={strat_name} | "
                    f"side={side} | level={level_name} {level_val:.2f} | "
                    f"closes={[round(c,2) for c in closes]} | close_pos={[round(p,2) for p in close_pos]}"
                )
                return False
    else:
        chosen = _select_interaction_level(levels_short, bar_close, df, band)
        if chosen is not None:
            level_name, level_val = chosen
            recent = df.iloc[-2:]
            if len(recent) < 2:
                return True
            closes = recent["close"].to_numpy(dtype=float)
            highs = recent["high"].to_numpy(dtype=float)
            lows = recent["low"].to_numpy(dtype=float)
            close_pos = [
                _close_position_in_bar(highs[i], lows[i], closes[i]) for i in range(2)
            ]
            if not all(c <= level_val for c in closes) or not all(pos <= 0.40 for pos in close_pos):
                logging.info(
                    f"⛔ {log_prefix}Acceptance blocked | session={session_name} | strategy={strat_name} | "
                    f"side={side} | level={level_name} {level_val:.2f} | "
                    f"closes={[round(c,2) for c in closes]} | close_pos={[round(p,2) for p in close_pos]}"
                )
                return False

    # Gate C: Opposing liquidity distance
    tp_dist = signal.get("tp_dist")
    try:
        tp_dist_val = float(tp_dist)
    except Exception:
        tp_dist_val = None
    if tp_dist_val and tp_dist_val > 0:
        session_high, session_low = _session_high_low(df, ts, session_name)
        swing_high = float(df["high"].iloc[-30:].max()) if len(df) >= 30 else None
        swing_low = float(df["low"].iloc[-30:].min()) if len(df) >= 30 else None

        opp_levels: Dict[str, float] = {}
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

        distances = []
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
                logging.info(
                    f"⛔ {log_prefix}Liquidity blocked | session={session_name} | strategy={strat_name} | "
                    f"side={side} | level={nearest_name} | dist={nearest_dist:.2f} | tp={tp_dist_val:.2f}"
                )
                return False

    return True


def _ny_continuation_gates(
    signal: dict,
    side: str,
    bar_close: float,
    df: pd.DataFrame,
    trend_day_series: Optional[dict],
) -> bool:
    return _ny_structure_gates(
        signal,
        side,
        bar_close,
        df,
        trend_day_series,
        require_prefix="continuation",
        log_prefix="NYGate",
    )


def continuation_rescue_allowed(
    signal: Optional[dict],
    side: str,
    bar_close: float,
    df: pd.DataFrame,
    trend_day_series: Optional[dict],
    allowlist: Optional[set],
    allowed_regimes: set,
    confirm_cfg: Optional[dict],
    guard_enabled: bool,
    signal_mode: Optional[str] = None,
) -> bool:
    if not guard_enabled:
        return True
    if not signal:
        return False
    mode = str(signal_mode or "calendar").lower()
    if mode != "structure":
        raw_key = parse_continuation_key(signal.get("strategy"))
        key = continuation_allowlist_key(raw_key)
        if allowlist is not None:
            if not key or key not in allowlist:
                logging.info(f"⛔ Continuation guard: key blocked ({key})")
                return False
    if allowed_regimes:
        try:
            regime, _, _ = volatility_filter.get_regime(df)
        except Exception:
            regime = None
        if not regime or str(regime).lower() not in allowed_regimes:
            logging.info(f"⛔ Continuation guard: regime blocked ({regime})")
            return False
    if not continuation_market_confirmed(side, bar_close, trend_day_series, confirm_cfg):
        logging.info("⛔ Continuation guard: confirmation failed")
        return False
    if not _ny_continuation_gates(signal, side, bar_close, df, trend_day_series):
        logging.info("⛔ Continuation guard: NY structure gate blocked")
        return False
    return True


def consensus_ml_ok(signal: Optional[dict], fallback_name: Optional[str] = None) -> bool:
    """Require stronger ML confidence before MLPhysics can support consensus."""
    if not signal:
        return False
    strat = str(signal.get("strategy") or fallback_name or "")
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
    min_conf = CONFIG.get("BACKTEST_CONSENSUS_ML_MIN_CONF")
    extra = CONFIG.get("BACKTEST_CONSENSUS_ML_EXTRA_MARGIN", 0.0)
    required = thr_val + float(extra or 0.0)
    if min_conf is not None:
        try:
            required = max(required, float(min_conf))
        except Exception:
            pass
    return conf_val >= required


def ml_vol_regime_ok(
    signal: Optional[dict],
    session_name: Optional[str],
    vol_regime: Optional[str],
    asia_viable: Optional[bool] = None,
) -> bool:
    """Require stronger ML confidence by volatility regime."""
    cfg = CONFIG.get("ML_PHYSICS_VOL_REGIME_GUARD", {}) or {}
    if not cfg.get("enabled", True):
        return True
    if not signal:
        return False
    strat = str(signal.get("strategy") or "")
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

    suppress_low_penalty = bool(
        asia_viable
        and session_name
        and str(session_name).upper() == "ASIA"
        and regime_key == "low"
    )

    delta = reg_cfg.get("min_conf_delta", 0.0)
    if isinstance(delta, dict):
        delta = delta.get(side, delta.get("default", 0.0))
    try:
        delta_val = float(delta or 0.0)
    except Exception:
        delta_val = 0.0
    if suppress_low_penalty:
        delta_val = 0.0
        required = thr_val
    else:
        required = thr_val + delta_val
        side_extra = reg_cfg.get("side_extra_delta")
        if isinstance(side_extra, dict) and side in side_extra:
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


def trim_incomplete_resample(df: pd.DataFrame, last_bar_time: datetime.datetime, timeframe_minutes: int) -> pd.DataFrame:
    """
    Drop the last resampled bar if the current 1m bar does not complete the window.
    """
    if df.empty:
        return df
    if last_bar_time.second != 0 or (last_bar_time.minute % timeframe_minutes) != (timeframe_minutes - 1):
        return df.iloc[:-1]
    return df


class CsvBarAppender:
    """
    Appends 1-minute bars to the existing history CSV without duplicates.
    """

    def __init__(self, csv_path: str, symbol: str, tz: ZoneInfo):
        self.csv_path = Path(csv_path)
        self.symbol = (symbol or "").replace(".", "")
        self.tz = tz
        self.mode = self._detect_mode()
        self._ensure_header()
        self.last_ts = self._read_last_timestamp()

    def _detect_mode(self) -> str:
        if not self.csv_path.exists():
            return "databento"
        try:
            with self.csv_path.open("r", errors="ignore") as f:
                first = f.readline().strip().lower()
            if first.startswith("ts_event"):
                return "databento"
            if first.startswith("time series"):
                return "legacy"
        except Exception:
            pass
        return "databento"

    def _ensure_header(self):
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        if self.mode == "legacy":
            header_symbol = self.symbol or "MES"
            lines = [
                f"Time Series,{header_symbol},,,,,",
                "Date,Symbol,Open,High,Low,Close,Volume",
            ]
            self.csv_path.write_text("\n".join(lines) + "\n")
        else:
            header = "ts_event,rtype,publisher_id,instrument_id,open,high,low,close,volume,symbol"
            self.csv_path.write_text(header + "\n")

    def _parse_date_from_line(self, line: str) -> Optional[datetime.datetime]:
        parts = line.split(",", 1)
        if not parts:
            return None
        try:
            if self.mode == "legacy":
                dt = datetime.datetime.strptime(parts[0], "%m/%d/%Y %I:%M %p")
                return dt.replace(tzinfo=self.tz)
            dt = pd.to_datetime(parts[0], utc=True, errors="coerce")
            if pd.isna(dt):
                return None
            return dt.tz_convert(self.tz)
        except Exception:
            return None

    def _read_last_timestamp(self) -> Optional[datetime.datetime]:
        if not self.csv_path.exists():
            return None
        try:
            with self.csv_path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size, 65536)
                f.seek(-read_size, 2)
                data = f.read().decode("utf-8", errors="ignore")
            lines = [line.strip() for line in data.splitlines() if line.strip()]
            for line in reversed(lines):
                if line.startswith("Time Series") or line.startswith("Date,"):
                    continue
                ts = self._parse_date_from_line(line)
                if ts:
                    return ts
        except Exception as e:
            logging.warning(f"CSV logger: failed reading last timestamp: {e}")
        return None

    def _format_row(self, ts: datetime.datetime, row: pd.Series) -> str:
        if self.mode == "legacy":
            ts_local = ts.astimezone(self.tz)
            ts_str = ts_local.strftime("%m/%d/%Y %I:%M %p")
            o = f"{float(row['open']):,.2f}"
            h = f"{float(row['high']):,.2f}"
            l = f"{float(row['low']):,.2f}"
            c = f"{float(row['close']):,.2f}"
            v = int(row.get("volume", 0))
            return f'{ts_str},{self.symbol},"{o}","{h}","{l}","{c}",{v}'

        ts_local = ts.astimezone(self.tz)
        ts_str = ts_local.replace(microsecond=0).isoformat(sep=" ")
        o = f"{float(row['open'])}"
        h = f"{float(row['high'])}"
        l = f"{float(row['low'])}"
        c = f"{float(row['close'])}"
        v = int(row.get("volume", 0))
        symbol = row.get("symbol") if "symbol" in row else None
        symbol = symbol or self.symbol or "MES"
        return f"{ts_str},33,1,0,{o},{h},{l},{c},{v},{symbol}"

    def append_from_df(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.sort_index()
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize(self.tz)
        last_ts = self.last_ts
        if last_ts is not None:
            if last_ts.tzinfo is None:
                if df.index.tz is not None:
                    last_ts = last_ts.replace(tzinfo=df.index.tz)
                else:
                    last_ts = last_ts.replace(tzinfo=self.tz)
            elif df.index.tz is not None:
                last_ts = last_ts.astimezone(df.index.tz)
            df = df[df.index > last_ts]
        if df.empty:
            return 0

        lines = []
        for ts, row in df.iterrows():
            lines.append(self._format_row(ts, row))

        with self.csv_path.open("a", newline="") as f:
            f.write("\n".join(lines) + "\n")

        self.last_ts = df.index[-1]
        return len(lines)


class TradeFactorCsvLogger:
    """
    Logs filled live trades to CSV with:
    - full signal payload (strategy-unique factors)
    - full latest-bar factor snapshot
    - decodable context_box (base64 JSON) for windowed diagnostics (e.g. ADX over ~4h)
    """

    _FIELDNAMES = [
        "event_time",
        "bar_time",
        "source",
        "strategy",
        "sub_strategy",
        "side",
        "entry_mode",
        "base_session",
        "current_session",
        "vol_regime",
        "trend_day_tier",
        "trend_day_dir",
        "current_price",
        "entry_price",
        "tp_dist",
        "sl_dist",
        "size",
        "entry_order_id",
        "stop_order_id",
        "target_order_id",
        "signal_factors_json",
        "bar_factors_json",
        "order_details_json",
        "strategy_results_json",
        "runtime_state_json",
        "context_box_b64",
        "context_box_json",
        "close_time",
        "close_source",
        "close_order_id",
        "close_result",
        "exit_price",
        "pnl_points",
        "pnl_dollars",
        "de3_management_close_reason",
        "close_details_json",
        "de3_management_json",
    ]

    def __init__(self, csv_path: str, tz: ZoneInfo, context_bars: int = 240, max_numeric_cols: int = 300):
        self.csv_path = Path(csv_path)
        self.tz = tz
        self.context_bars = max(10, int(context_bars or 240))
        self.max_numeric_cols = max(20, int(max_numeric_cols or 300))
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    @staticmethod
    def _to_native(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is None:
                return value.tz_localize("UTC").isoformat()
            return value.isoformat()
        if isinstance(value, datetime.datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=dt_timezone.utc).isoformat()
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, datetime.timedelta):
            return float(value.total_seconds())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): TradeFactorCsvLogger._to_native(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [TradeFactorCsvLogger._to_native(v) for v in value]
        if isinstance(value, (bool, int, float, str)):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return str(value)

    @classmethod
    def _json_text(cls, payload: Any) -> str:
        safe = cls._to_native(payload)
        return json.dumps(safe, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            try:
                with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_fields = list(reader.fieldnames or [])
                    rows = list(reader)
                if existing_fields == self._FIELDNAMES:
                    return
                needs_rewrite = any(field not in existing_fields for field in self._FIELDNAMES)
                if not needs_rewrite:
                    return
                with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow({field: row.get(field, "") for field in self._FIELDNAMES})
                return
            except Exception as exc:
                logging.warning("Trade-factor logger header migration failed (%s): %s", self.csv_path, exc)
        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writeheader()

    def _latest_bar_snapshot(self, df: pd.DataFrame) -> tuple[Optional[str], Dict[str, Any]]:
        if df is None or df.empty:
            return None, {}
        row = df.iloc[-1]
        out: Dict[str, Any] = {}
        for col, value in row.items():
            out[str(col)] = self._to_native(value)
        bar_ts = df.index[-1]
        if isinstance(bar_ts, pd.Timestamp):
            if bar_ts.tzinfo is None:
                bar_ts = bar_ts.tz_localize(self.tz)
            bar_ts_iso = bar_ts.astimezone(self.tz).isoformat()
        elif isinstance(bar_ts, datetime.datetime):
            if bar_ts.tzinfo is None:
                bar_ts = bar_ts.replace(tzinfo=self.tz)
            bar_ts_iso = bar_ts.astimezone(self.tz).isoformat()
        else:
            bar_ts_iso = None
        return bar_ts_iso, out

    def _numeric_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"window_bars": self.context_bars, "numeric_rollup": {}, "adx_4h_rollup": {}}

        lookback = min(len(df), self.context_bars)
        tail = df.tail(lookback)
        num_df = tail.select_dtypes(include=["number", "bool"])
        cols = list(num_df.columns)
        if len(cols) > self.max_numeric_cols:
            cols = cols[: self.max_numeric_cols]

        rollup: Dict[str, Any] = {}
        for col in cols:
            s = pd.to_numeric(num_df[col], errors="coerce").dropna()
            if s.empty:
                continue
            first = float(s.iloc[0])
            last = float(s.iloc[-1])
            n = int(len(s))
            rollup[str(col)] = {
                "n": n,
                "first": first,
                "last": last,
                "delta": float(last - first),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)) if n > 1 else 0.0,
                "min": float(s.min()),
                "max": float(s.max()),
            }

        adx_rollup: Dict[str, Any] = {}
        adx_cols = [c for c in num_df.columns if "adx" in str(c).lower()]
        if adx_cols:
            adx_tail = df.tail(min(len(df), 240))  # ~4h on 1-minute bars
            for col in adx_cols[:50]:
                s = pd.to_numeric(adx_tail[col], errors="coerce").dropna()
                if s.empty:
                    continue
                adx_rollup[str(col)] = {
                    "bars": int(len(s)),
                    "last": float(s.iloc[-1]),
                    "mean": float(s.mean()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "delta": float(s.iloc[-1] - s.iloc[0]) if len(s) > 1 else 0.0,
                }

        return {
            "window_bars": int(lookback),
            "numeric_rollup": rollup,
            "adx_4h_rollup": adx_rollup,
        }

    @classmethod
    def _json_dict(cls, raw: Any) -> Dict[str, Any]:
        if not raw:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        try:
            parsed = json.loads(str(raw))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _row_entry_order_ids(cls, row: Optional[Dict[str, Any]]) -> set[str]:
        if not isinstance(row, dict):
            return set()
        candidate_values = [
            row.get("entry_order_id"),
            cls._json_dict(row.get("order_details_json")).get("order_id"),
            cls._json_dict(row.get("signal_factors_json")).get("entry_order_id"),
        ]
        out: set[str] = set()
        for value in candidate_values:
            text = str(value or "").strip()
            if text:
                out.add(text)
        return out

    @staticmethod
    def _coerce_event_time(value: Any, fallback_tz: ZoneInfo) -> Optional[datetime.datetime]:
        if isinstance(value, str):
            try:
                value = datetime.datetime.fromisoformat(value)
            except Exception:
                value = None
        if isinstance(value, pd.Timestamp):
            value = value.to_pydatetime()
        if not isinstance(value, datetime.datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=fallback_tz)
        return value.astimezone(fallback_tz)

    def _build_synthetic_close_row(
        self,
        *,
        trade: Dict[str, Any],
        metrics: Dict[str, Any],
        close_time: datetime.datetime,
        close_details: Optional[Dict[str, Any]] = None,
        de3_management: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        close_time_value = self._coerce_event_time(close_time, self.tz) or datetime.datetime.now(self.tz)
        entry_time_value = self._coerce_event_time(trade.get("entry_time"), self.tz) or close_time_value
        entry_price = float(
            _coerce_float(
                metrics.get("entry_price", trade.get("broker_entry_price", trade.get("entry_price", 0.0))),
                0.0,
            )
        )
        exit_price = float(_coerce_float(metrics.get("exit_price"), entry_price))
        pnl_points = float(_coerce_float(metrics.get("pnl_points"), 0.0))
        pnl_dollars = float(_coerce_float(metrics.get("pnl_dollars"), 0.0))
        close_result = "win" if pnl_dollars > 0.0 else "loss" if pnl_dollars < 0.0 else "flat"
        order_details_payload = {
            "entry_order_id": trade.get("entry_order_id"),
            "stop_order_id": trade.get("stop_order_id"),
            "target_order_id": trade.get("target_order_id"),
            "broker_entry_price": trade.get("broker_entry_price"),
            "tracking_restored": trade.get("tracking_restored"),
        }
        runtime_state_payload = {
            "synthetic_close_only": True,
            "tracking_restored": bool(trade.get("tracking_restored", False)),
        }
        return {
            "event_time": entry_time_value.isoformat(),
            "bar_time": entry_time_value.isoformat(),
            "source": "synthetic_close_only",
            "strategy": str(trade.get("strategy") or ""),
            "sub_strategy": str(trade.get("sub_strategy") or ""),
            "side": str(trade.get("side") or ""),
            "entry_mode": str(trade.get("entry_mode") or ""),
            "base_session": "",
            "current_session": "",
            "vol_regime": str(trade.get("vol_regime") or ""),
            "trend_day_tier": str(trade.get("trend_day_tier") or ""),
            "trend_day_dir": str(trade.get("trend_day_dir") or ""),
            "current_price": str(exit_price),
            "entry_price": str(entry_price),
            "tp_dist": str(float(_coerce_float(trade.get("tp_dist"), 0.0))),
            "sl_dist": str(float(_coerce_float(trade.get("sl_dist"), 0.0))),
            "size": str(int(_coerce_int(trade.get("size"), 0))),
            "entry_order_id": str(metrics.get("entry_order_id") or trade.get("entry_order_id") or ""),
            "stop_order_id": str(trade.get("stop_order_id") or ""),
            "target_order_id": str(trade.get("target_order_id") or ""),
            "signal_factors_json": self._json_text(trade),
            "bar_factors_json": self._json_text({}),
            "order_details_json": self._json_text(order_details_payload),
            "strategy_results_json": self._json_text({}),
            "runtime_state_json": self._json_text(runtime_state_payload),
            "context_box_b64": "",
            "context_box_json": "",
            "close_time": close_time_value.isoformat(),
            "close_source": str(metrics.get("source") or ""),
            "close_order_id": str(metrics.get("order_id") or ""),
            "close_result": close_result,
            "exit_price": str(exit_price),
            "pnl_points": str(pnl_points),
            "pnl_dollars": str(pnl_dollars),
            "de3_management_close_reason": str((de3_management or {}).get("close_reason") or ""),
            "close_details_json": self._json_text(close_details or metrics or {}),
            "de3_management_json": self._json_text(de3_management or {}),
        }

    def annotate_trade_close(
        self,
        *,
        trade: Optional[Dict[str, Any]],
        close_metrics: Optional[Dict[str, Any]],
        close_time: datetime.datetime,
        close_details: Optional[Dict[str, Any]] = None,
        de3_management: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.csv_path.exists():
            return False
        if not isinstance(trade, dict):
            return False
        metrics = dict(close_metrics or {})
        close_time_value = self._coerce_event_time(close_time, self.tz) or datetime.datetime.now(self.tz)
        target_entry_ids = {
            text
            for text in [
                str(metrics.get("entry_order_id") or "").strip(),
                str(trade.get("entry_order_id") or "").strip(),
            ]
            if text
        }
        target_strategy = str(trade.get("strategy") or "").strip()
        target_sub_strategy = str(trade.get("sub_strategy") or "").strip()
        target_side = str(trade.get("side") or "").strip().upper()
        target_entry_price = round(
            float(
                _coerce_float(
                    metrics.get("entry_price", trade.get("broker_entry_price", trade.get("entry_price", 0.0))),
                    0.0,
                )
            ),
            4,
        )
        try:
            with self.csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as exc:
            logging.warning("Trade-factor logger close annotation read failed (%s): %s", self.csv_path, exc)
            return False

        match_index = None
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            row_entry_ids = self._row_entry_order_ids(row)
            if target_entry_ids and row_entry_ids.intersection(target_entry_ids):
                match_index = index
                break
        if match_index is None:
            for index in range(len(rows) - 1, -1, -1):
                row = rows[index]
                if str(row.get("strategy") or "").strip() != target_strategy:
                    continue
                if str(row.get("sub_strategy") or "").strip() != target_sub_strategy:
                    continue
                if str(row.get("side") or "").strip().upper() != target_side:
                    continue
                row_entry_price = round(float(_coerce_float(row.get("entry_price"), 0.0)), 4)
                if abs(row_entry_price - target_entry_price) > 1e-4:
                    continue
                if str(row.get("close_time") or "").strip():
                    continue
                match_index = index
                break
        if match_index is None:
            target_close_order_id = str(metrics.get("order_id") or "").strip()
            target_exit_price = round(float(_coerce_float(metrics.get("exit_price"), 0.0)), 4)
            close_time_iso = close_time_value.astimezone(self.tz).isoformat()
            repair_index = None
            target_size = int(_coerce_int(trade.get("size"), 0) or 0)
            for index in range(len(rows) - 1, -1, -1):
                row = rows[index]
                if str(row.get("strategy") or "").strip() != target_strategy:
                    continue
                if str(row.get("sub_strategy") or "").strip() != target_sub_strategy:
                    continue
                if str(row.get("side") or "").strip().upper() != target_side:
                    continue
                row_entry_price = round(float(_coerce_float(row.get("entry_price"), 0.0)), 4)
                if abs(row_entry_price - target_entry_price) > 1e-4:
                    continue
                row_size = int(_coerce_int(row.get("size"), 0) or 0)
                if target_size > 0 and row_size not in (0, target_size):
                    continue
                row_close_time = self._coerce_event_time(row.get("close_time"), self.tz)
                if row_close_time is None:
                    continue
                if abs((row_close_time - close_time_value).total_seconds()) > 120.0:
                    continue
                runtime_state = self._json_dict(row.get("runtime_state_json"))
                row_close_order_id = str(row.get("close_order_id") or "").strip()
                if target_close_order_id and row_close_order_id and row_close_order_id != target_close_order_id:
                    continue
                if not (
                    str(row.get("source") or "").strip() == "synthetic_close_only"
                    or bool(runtime_state.get("synthetic_close_only", False))
                    or str(row.get("close_source") or "").strip() in {"broker_flat_cleanup", "price_snapshot", "order_fill_fallback"}
                ):
                    continue
                repair_index = index
                break
            if repair_index is not None:
                match_index = repair_index

        if match_index is None:
            target_close_order_id = str(metrics.get("order_id") or "").strip()
            target_exit_price = round(float(_coerce_float(metrics.get("exit_price"), 0.0)), 4)
            close_time_iso = close_time_value.astimezone(self.tz).isoformat()
            duplicate_exists = False
            for row in rows:
                if target_close_order_id and str(row.get("close_order_id") or "").strip() == target_close_order_id:
                    duplicate_exists = True
                    break
                if (
                    str(row.get("strategy") or "").strip() == target_strategy
                    and str(row.get("sub_strategy") or "").strip() == target_sub_strategy
                    and str(row.get("side") or "").strip().upper() == target_side
                    and str(row.get("close_time") or "").strip() == close_time_iso
                    and abs(round(float(_coerce_float(row.get("entry_price"), 0.0)), 4) - target_entry_price) <= 1e-4
                    and abs(round(float(_coerce_float(row.get("exit_price"), 0.0)), 4) - target_exit_price) <= 1e-4
                ):
                    duplicate_exists = True
                    break
            if duplicate_exists:
                return True
            synthetic_row = self._build_synthetic_close_row(
                trade=trade,
                metrics=metrics,
                close_time=close_time_value,
                close_details=close_details,
                de3_management=de3_management,
            )
            rows.append(synthetic_row)
            try:
                with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                    writer.writeheader()
                    for current_row in rows:
                        writer.writerow({field: current_row.get(field, "") for field in self._FIELDNAMES})
            except Exception as exc:
                logging.warning("Trade-factor logger synthetic close write failed (%s): %s", self.csv_path, exc)
                return False
            logging.warning(
                "Trade-factor logger appended synthetic close-only row for strategy=%s entry_order_id=%s close_order_id=%s",
                target_strategy or "Unknown",
                str(metrics.get("entry_order_id") or trade.get("entry_order_id") or ""),
                target_close_order_id,
            )
            return True

        row = dict(rows[match_index])
        pnl_dollars = float(_coerce_float(metrics.get("pnl_dollars"), 0.0))
        row.update(
            {
                "entry_order_id": str(
                    metrics.get("entry_order_id")
                    or trade.get("entry_order_id")
                    or row.get("entry_order_id")
                    or ""
                ),
                "close_time": close_time_value.astimezone(self.tz).isoformat(),
                "close_source": str(metrics.get("source") or ""),
                "close_order_id": str(metrics.get("order_id") or ""),
                "close_result": "win" if pnl_dollars > 0.0 else "loss" if pnl_dollars < 0.0 else "flat",
                "exit_price": str(float(_coerce_float(metrics.get("exit_price"), 0.0))),
                "pnl_points": str(float(_coerce_float(metrics.get("pnl_points"), 0.0))),
                "pnl_dollars": str(pnl_dollars),
                "de3_management_close_reason": str(
                    (de3_management or {}).get("close_reason") or ""
                ),
                "close_details_json": self._json_text(close_details or metrics or {}),
                "de3_management_json": self._json_text(de3_management or {}),
            }
        )
        rows[match_index] = row
        try:
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
                writer.writeheader()
                for current_row in rows:
                    writer.writerow({field: current_row.get(field, "") for field in self._FIELDNAMES})
        except Exception as exc:
            logging.warning("Trade-factor logger close annotation write failed (%s): %s", self.csv_path, exc)
            return False
        return True

    @staticmethod
    def decode_context_box(context_box_b64: str) -> Dict[str, Any]:
        if not context_box_b64:
            return {}
        padded = context_box_b64 + "=" * ((4 - len(context_box_b64) % 4) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        return json.loads(raw)

    def log_trade_fill(
        self,
        *,
        current_time: datetime.datetime,
        current_price: float,
        entry_price: float,
        signal: Dict[str, Any],
        order_details: Optional[Dict[str, Any]],
        market_df: pd.DataFrame,
        source: str,
        base_session: Optional[str],
        current_session: Optional[str],
        vol_regime: Optional[str],
        trend_day_tier: Optional[int],
        trend_day_dir: Optional[str],
        strategy_results: Optional[Dict[str, Any]] = None,
        runtime_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        event_ts = current_time
        if event_ts.tzinfo is None:
            event_ts = event_ts.replace(tzinfo=self.tz)
        event_ts_iso = event_ts.astimezone(self.tz).isoformat()

        bar_ts_iso, bar_snapshot = self._latest_bar_snapshot(market_df)
        context_payload = {
            "schema": "trade_factors_v1",
            "decode_hint": "base64url -> utf8 json",
            "context": self._numeric_context(market_df),
        }
        context_box_json = self._json_text(context_payload)
        context_box_b64 = base64.urlsafe_b64encode(context_box_json.encode("utf-8")).decode("ascii")

        row = {
            "event_time": event_ts_iso,
            "bar_time": bar_ts_iso or "",
            "source": str(source or ""),
            "strategy": str(signal.get("strategy") or ""),
            "sub_strategy": str(signal.get("sub_strategy") or ""),
            "side": str(signal.get("side") or ""),
            "entry_mode": str(signal.get("entry_mode") or ""),
            "base_session": str(base_session or ""),
            "current_session": str(current_session or ""),
            "vol_regime": str(vol_regime or signal.get("vol_regime") or ""),
            "trend_day_tier": str(trend_day_tier if trend_day_tier is not None else ""),
            "trend_day_dir": str(trend_day_dir or ""),
            "current_price": str(float(current_price)),
            "entry_price": str(float(entry_price)),
            "tp_dist": str(float(signal.get("tp_dist", 0.0) or 0.0)),
            "sl_dist": str(float(signal.get("sl_dist", 0.0) or 0.0)),
            "size": str(int(signal.get("size", 0) or 0)),
            "entry_order_id": str(
                signal.get("entry_order_id")
                or (order_details or {}).get("broker_order_id")
                or (order_details or {}).get("order_id")
                or ""
            ),
            "stop_order_id": str(signal.get("stop_order_id") or ""),
            "target_order_id": str(signal.get("target_order_id") or (order_details or {}).get("target_order_id") or ""),
            "signal_factors_json": self._json_text(signal),
            "bar_factors_json": self._json_text(bar_snapshot),
            "order_details_json": self._json_text(order_details or {}),
            "strategy_results_json": self._json_text(strategy_results or {}),
            "runtime_state_json": self._json_text(runtime_state or {}),
            "context_box_b64": context_box_b64,
            "context_box_json": context_box_json,
            "close_time": "",
            "close_source": "",
            "close_order_id": "",
            "close_result": "",
            "exit_price": "",
            "pnl_points": "",
            "pnl_dollars": "",
            "de3_management_close_reason": "",
            "close_details_json": "",
            "de3_management_json": "",
        }

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writerow(row)


def _configure_utf8_stdio() -> None:
    """
    Make console/file logging resilient on Windows machines using cp1252 defaults.

    This workspace emits symbols like checkmarks, warning icons, and stop signs
    very early during import. After moving the repo between PCs, Python can end up
    with a different default stdio encoding than the original machine.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass


_configure_utf8_stdio()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("topstep_live_bot.log", encoding="utf-8", errors="backslashreplace"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True  # Override any pre-existing logging config
)

NY_TZ = ZoneInfo('America/New_York')
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
SIGMA_WINDOW = 30
IMPULSE_MIN_BARS = 30
IMPULSE_MAX_RETRACE = 0.25
TICK_SIZE = 0.25
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
TREND_DAY_T1_REQUIRE_CONFIRMATION = True
ALT_PRE_TIER1_VWAP_SIGMA = 2.25
TREND_DAY_REACTIVATION_COOLDOWN_MINUTES = 5
TREND_DAY_T1_REQUIRE_STRUCTURAL_BIAS = True
TREND_DAY_SMA9_REVERSAL_BARS = 4
TREND_DAY_SMA9_MIN_SLOPE = 0.2
TREND_DAY_ATR_CONTRACTION = 1.1
TREND_DAY_ASIA_STRUCTURE_ONLY = True
TREND_DAY_ASIA_DISABLE_TIER2 = True

# ==========================================
# TREND DAY DETECTOR (LIVE)
# ==========================================
def compute_trend_day_series(df: pd.DataFrame) -> dict:
    close = df["close"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma9 = close.rolling(9, min_periods=9).mean()
    sma9_slope = sma9.diff()

    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr20 = tr.ewm(alpha=1 / 20, adjust=False).mean()
    # Use prior session (NY day) ATR median as baseline to avoid shock contamination
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ)
    day_index = idx.date
    daily_atr_median = atr20.groupby(day_index).median()
    prior_day_median = daily_atr_median.shift(1)
    atr_baseline = pd.Series(day_index, index=atr20.index).map(prior_day_median)
    # Fallback to rolling median if prior session missing (e.g., first day)
    atr_baseline = atr_baseline.combine_first(
        atr20.rolling(ATR_BASELINE_WINDOW, min_periods=ATR_BASELINE_WINDOW).median()
    )
    atr_expansion = atr20 / atr_baseline.replace(0, np.nan)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
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

    session_open = df["open"].groupby(day_index).transform("first")
    daily_low = df["low"].groupby(day_index).min()
    daily_high = df["high"].groupby(day_index).max()
    prior_session_low = pd.Series(day_index, index=df.index).map(daily_low.shift(1))
    prior_session_high = pd.Series(day_index, index=df.index).map(daily_high.shift(1))

    sma50_slope_up = (sma50 - sma50.shift(TREND_UP_EMA_SLOPE_BARS)) > 0
    above_ema50 = close > ema50
    above_ema50_count = above_ema50.rolling(
        TREND_UP_ABOVE_EMA50_WINDOW, min_periods=TREND_UP_ABOVE_EMA50_WINDOW
    ).sum()
    above_ema50_ok = above_ema50_count >= TREND_UP_ABOVE_EMA50_COUNT
    seg = TREND_UP_HL_SEGMENT
    low_seg1 = df["low"].rolling(seg, min_periods=seg).min()
    low_seg2 = df["low"].shift(seg).rolling(seg, min_periods=seg).min()
    low_seg3 = df["low"].shift(seg * 2).rolling(seg, min_periods=seg).min()
    higher_lows = (low_seg1 > low_seg2) & (low_seg2 > low_seg3)
    trend_up_alt = sma50_slope_up & above_ema50_ok & higher_lows & (atr_expansion >= TREND_UP_ATR_EXP)

    sma50_slope_down = (sma50 - sma50.shift(TREND_DOWN_EMA_SLOPE_BARS)) < 0
    below_ema50 = close < ema50
    below_ema50_count = below_ema50.rolling(
        TREND_DOWN_BELOW_EMA50_WINDOW, min_periods=TREND_DOWN_BELOW_EMA50_WINDOW
    ).sum()
    below_ema50_ok = below_ema50_count >= TREND_DOWN_BELOW_EMA50_COUNT
    seg_down = TREND_DOWN_LH_SEGMENT
    high_seg1 = df["high"].rolling(seg_down, min_periods=seg_down).max()
    high_seg2 = df["high"].shift(seg_down).rolling(seg_down, min_periods=seg_down).max()
    high_seg3 = df["high"].shift(seg_down * 2).rolling(seg_down, min_periods=seg_down).max()
    lower_highs = (high_seg1 < high_seg2) & (high_seg2 < high_seg3)
    trend_down_alt = sma50_slope_down & below_ema50_ok & lower_highs & (atr_expansion >= TREND_DOWN_ATR_EXP)

    up_move = df["high"].diff()
    down_move = -df["low"].diff()
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

    return {
        "ema50": ema50,
        "ema200": ema200,
        "sma50": sma50,
        "sma9": sma9,
        "sma9_slope": sma9_slope,
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

# ==========================================
# 2a. REJECTION FILTER (Trade Direction Filters)
# ==========================================
# Implementation moved to rejection_filter.py to keep this entrypoint focused on
# bot orchestration.

# OptimizedTPEngine moved to risk_engine.py



# ProjectXClient moved to client.py

class ContinuationRescueManager:
    """
    Manages the FractalSweepStrategy (Continuation) lookups.
    Acts as a 'Second Opinion' when trades are blocked by filters.
    """
    def __init__(self):
        self.configs = None
        self.strategy_instances = {}
        self.strategy_cls = None
        # 1. TIMEZONE FIX: Strategies operate on NY Time
        self.ny_tz = ZoneInfo('America/New_York')

    def _ensure_loaded(self) -> None:
        if self.configs is not None and self.strategy_cls is not None:
            return
        strategy_cls, strategy_configs = _load_continuation_strategy_runtime()
        self.strategy_cls = strategy_cls
        self.configs = strategy_configs

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

        def last_val(key: str, default=None):
            series = trend_day_series.get(key) if isinstance(trend_day_series, dict) else None
            if isinstance(series, pd.Series):
                try:
                    return series.iloc[-1]
                except Exception:
                    return default
            return default

        prior_high = last_val("prior_session_high", None)
        prior_low = last_val("prior_session_low", None)

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
            current_time = current_time.replace(tzinfo=dt_timezone.utc)
        ny_time = current_time.astimezone(self.ny_tz)

        tp_dist = 6.0
        sl_dist = 4.0
        try:
            sltp = dynamic_sltp_engine.calculate_sltp("Continuation", df, ts=ny_time)
            tp_dist = float(sltp.get("tp_dist", tp_dist))
            sl_dist = float(sltp.get("sl_dist", sl_dist))
        except Exception as e:
            logging.warning(f"Continuation SL/TP ATR calc failed: {e}")

        return {
            "strategy": "Continuation_Structure",
            "side": required_side,
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "size": 5,
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
        """
        Checks if the current time matches a known Continuation Strategy window.
        Returns a rescue signal if valid for the REQUIRED_SIDE.
        """
        if not CONFIG.get("CONTINUATION_ENABLED", True):
            return None
        if df.empty:
            return None

        mode = str(
            signal_mode
            or (CONFIG.get("CONTINUATION_GUARD", {}) or {}).get("signal_mode", "calendar")
            or "calendar"
        ).lower()
        if mode == "structure":
            return self._structure_break_signal(
                df, current_time, required_side, current_price, trend_day_series
            )

        self._ensure_loaded()

        # 2. Convert Bot Time (UTC) to Strategy Time (NY)
        if current_time.tzinfo is None:
             current_time = current_time.replace(tzinfo=dt_timezone.utc)

        ny_time = current_time.astimezone(self.ny_tz)

        # 3. Construct Key using NY TIME (e.g. Q4_W45_D7_Asia)
        quarter = (ny_time.month - 1) // 3 + 1
        week = ny_time.isocalendar().week
        day = ny_time.weekday() + 1 # 1=Monday, 7=Sunday
        h = ny_time.hour

        if 18 <= h or h < 3: session = "Asia"
        elif 3 <= h < 8: session = "London"
        elif 8 <= h < 17: session = "NY"
        else: session = "Other"

        candidate_key = f"Q{quarter}_W{week}_D{day}_{session}"

        # 4. Check Config & Instantiate
        if candidate_key not in self.configs:
            return None

        if candidate_key not in self.strategy_instances:
            try:
                self.strategy_instances[candidate_key] = self.strategy_cls(candidate_key)
            except ValueError:
                return None

        strat = self.strategy_instances[candidate_key]

        # 5. Generate Signal
        try:
            signals_df = strat.generate_signals(df)

            if not signals_df.empty:
                # Verify freshness
                last_sig_time = signals_df.index[-1]
                if last_sig_time.tzinfo is None:
                    last_sig_time = last_sig_time.replace(tzinfo=dt_timezone.utc)
                else:
                    last_sig_time = last_sig_time.astimezone(dt_timezone.utc)

                check_time = current_time.astimezone(dt_timezone.utc)

                if last_sig_time == check_time:
                    tp_dist = strat.target if hasattr(strat, 'target') else 6.0
                    sl_dist = strat.stop if hasattr(strat, 'stop') else 4.0
                    try:
                        sltp = dynamic_sltp_engine.calculate_sltp("Continuation", df, ts=ny_time)
                        tp_dist = float(sltp.get("tp_dist", tp_dist))
                        sl_dist = float(sltp.get("sl_dist", sl_dist))
                    except Exception as e:
                        logging.warning(f"Continuation SL/TP ATR calc failed: {e}")

                    return {
                        'strategy': f"Continuation_{candidate_key}",
                        'side': required_side, # FORCE the direction we need (The Rescue Side)
                        'tp_dist': tp_dist,
                        'sl_dist': sl_dist,
                        'size': 5,
                        'rescued': True
                    }
        except Exception as e:
            logging.error(f"Continuation Strategy Error ({candidate_key}): {e}")
            return None

        return None

# ==========================================
# 12. MAIN EXECUTION LOOP (ASYNCIO UPGRADED)
# ==========================================
async def run_bot():
    """
    Main bot execution loop - now async with independent tasks.

    Benefits:
    - Independent Heartbeat task (validates session every 60s)
    - Independent Position Sync task (syncs broker position every 30s)
    - Non-blocking sleep for faster response times
    - Strategy calculations cannot block heartbeat or position sync
    """
    param_scaler.apply_scaling()  # Scale regime params to maintain R:R ratios

    refresh_target_symbol()
    print("=" * 60)
    print("PROJECTX GATEWAY - MES FUTURES BOT (LIVE)")
    print("--- Julie Pro (Session Specialized + AsyncIO) ---")
    print("--- DYNAMIC SL/TP ENGINE ENABLED ---")
    print(f"REST API: {CONFIG['REST_BASE_URL']}")
    print(f"Target Symbol: {CONFIG['TARGET_SYMBOL']}")
    print("=" * 60)

    client = ProjectXClient()
    
    # Step 1: Authenticate
    try:
        client.login()
    except Exception as e:
        print(f"CRITICAL: Failed to login. Check credentials. Error: {e}")
        return

    # Step 2: Fetch Account ID
    print("\n📋 Fetching account information...")
    account_id = client.fetch_accounts()
    if account_id is None:
        print("CRITICAL: Could not retrieve account ID")
        return
    
    # Step 3: Fetch Contract ID
    print("\n📋 Fetching available contracts...")
    contract_id = client.fetch_contracts()
    if contract_id is None:
        print("CRITICAL: Could not retrieve contract ID")
        return
    
    print(f"\n✅ Setup complete:")
    print(f"   Account ID: {client.account_id}")
    print(f"   Contract ID: {client.contract_id}")

    user_stream_enabled = bool(CONFIG.get("PROJECTX_USER_STREAM_ENABLED", True))
    if user_stream_enabled:
        try:
            started = await client.start_user_stream()
            if started:
                logging.info("📡 ProjectX user stream enabled for live account/position/trade updates")
            else:
                logging.warning("ProjectX user stream did not start; continuing with REST fallbacks")
        except Exception as exc:
            logging.warning("ProjectX user stream startup failed: %s", exc)

    filterless_only_mode = (
        str(os.environ.get("JULIE_FILTERLESS_ONLY", "")).strip().lower() in TRUTHY_ENV_VALUES
    )
    disable_strategy_filters = (
        str(os.environ.get("JULIE_DISABLE_STRATEGY_FILTERS", "")).strip().lower()
        in TRUTHY_ENV_VALUES
    )
    filterless_keep_gemini = (
        str(os.environ.get("JULIE_FILTERLESS_KEEP_GEMINI", "")).strip().lower()
        in TRUTHY_ENV_VALUES
    )
    filterless_disabled_raw = CONFIG.get("FILTERLESS_LIVE_DISABLED_STRATEGIES", []) or []
    filterless_disabled = {
        str(value).strip().lower()
        for value in filterless_disabled_raw
        if str(value).strip()
    }
    gemini_cfg = CONFIG.get("GEMINI", {}) or {}
    gemini_runtime_enabled = bool(gemini_cfg.get("enabled", False)) and (
        not filterless_only_mode or filterless_keep_gemini
    )
    filter_stack_runtime_enabled = not disable_strategy_filters

    mnq_client = None
    vix_client = None
    if filterless_only_mode:
        # Filterless mode previously skipped BOTH clients — but the
        # 2026-04-22 v2 LFO + v3 RL overlays need live MNQ and VIX
        # for the cross-market feature block. Initialize both even
        # in filterless mode; only skip when explicitly opted out.
        if os.environ.get("JULIE_SKIP_CROSS_MARKET_LIVE", "0").strip() != "1":
            try:
                from yahoo_vix_client import YahooVIXClient  # top-level importable
                mnq_target_symbol = determine_current_contract_symbol(
                    "MNQ", tz_name=CONFIG.get("TIMEZONE", "US/Eastern")
                )
                mnq_client = ProjectXClient(
                    contract_root="MNQ", target_symbol=mnq_target_symbol
                )
                logging.info(
                    "[cross-market] filterless mode: live MNQ (ProjectX) + "
                    "VIX (Yahoo) clients enabled for overlay cross-market features."
                )
                vix_client = YahooVIXClient(target_symbol="^VIX")
            except Exception as _cm_exc:
                logging.warning(
                    "[cross-market] failed to init live MNQ/VIX clients: %s; "
                    "v2 overlays will fall back to cached parquets", _cm_exc
                )
                mnq_client = None
                vix_client = None
        else:
            logging.info("[cross-market] JULIE_SKIP_CROSS_MARKET_LIVE=1 — "
                         "using cached MNQ/VIX parquets only")
    else:
        (
            OrbStrategy,
            IntradayDipStrategy,
            ConfluenceStrategy,
            SMTStrategy,
            ICTModelStrategy,
            ImpulseBreakoutStrategy,
            AuctionReversionStrategy,
            LiquiditySweepStrategy,
            ValueAreaBreakoutStrategy,
            SmoothTrendAsiaStrategy,
            DynamicEngineStrategy,
            VIXReversionStrategy,
            YahooVIXClient,
        ) = _load_non_filterless_runtime()
        mnq_target_symbol = determine_current_contract_symbol(
            "MNQ", tz_name=CONFIG.get("TIMEZONE", "US/Eastern")
        )
        mnq_client = ProjectXClient(contract_root="MNQ", target_symbol=mnq_target_symbol)
        # We use ^VIX (The Index) as it is the standard for mean reversion
        # and free via Yahoo, whereas Topstep Rithmic usually lacks CBOE data.
        logging.info("Initializing Virtual VIX Client (Yahoo Finance)...")
        vix_client = YahooVIXClient(target_symbol="^VIX")

    try:
        if mnq_client is not None:
            mnq_client.login()
            mnq_client.account_id = client.account_id or mnq_client.fetch_accounts()
            mnq_client.fetch_contracts()

        if vix_client is not None:
            # Login VIX client (Virtual)
            vix_client.login()
            # No account ID needed for Yahoo, but we call methods for consistency
            vix_client.fetch_contracts()
    except Exception as e:
        logging.error(f"❌ Failed to initialize secondary clients: {e}")
        return

    # Initialize all strategies
    truth_social_cfg = _truth_social_cfg()
    truth_social_enabled = bool(truth_social_cfg.get("enabled", False))
    sentiment_service = build_truth_social_sentiment_service(truth_social_cfg) if truth_social_enabled else None
    if filterless_only_mode:
        filterless_roster = ["DynamicEngine3", "RegimeAdaptive"]
        if "ml_physics" not in filterless_disabled:
            filterless_roster.append("MLPhysics")
        filterless_roster.append("AetherFlow")
        logging.info(
            "🧪 FILTERLESS ONLY MODE: execution roster limited to %s",
            ", ".join(filterless_roster),
        )
        event_logger.log_system_event(
            "MODE",
            "Filterless-only live mode enabled",
            {
                "mode": "FILTERLESS_ONLY",
                "roster": ",".join(filterless_roster),
            },
        )
        logging.info(
            "🧪 FILTERLESS ONLY MODE: unified candidate priority enabled across active strategies; "
            "confidence and strategy name break ties"
        )
        if "ml_physics" in filterless_disabled:
            logging.info("🧪 FILTERLESS ONLY MODE: MLPhysicsStrategy disabled by filterless live config")
            event_logger.log_system_event(
                "MODE",
                "Filterless live strategy disabled",
                {
                    "strategy": "MLPhysicsStrategy",
                    "status": "DISABLED",
                    "reason": "disabled in filterless live config",
                },
            )
    if disable_strategy_filters:
        logging.info(
            "🧪 FILTERLESS EXECUTION MODE: strategy filter stack disabled for live execution"
        )
        event_logger.log_system_event(
            "MODE",
            "Filterless execution mode enabled",
            {"mode": "FILTERLESS_EXECUTION"},
        )
        if not filterless_only_mode:
            logging.warning(
                "JULIE_DISABLE_STRATEGY_FILTERS is enabled without JULIE_FILTERLESS_ONLY; "
                "all live strategies will bypass the external filter stack."
            )
    if bool(gemini_cfg.get("enabled", False)) and not gemini_runtime_enabled:
        logging.info(
            "🧪 FILTERLESS ONLY MODE: Gemini session optimizer disabled; "
            "set JULIE_FILTERLESS_KEEP_GEMINI=1 to re-enable it."
        )
    if filterless_only_mode:
        CONFIG["DE3_VERSION"] = "v4"
        de3_v4_cfg = CONFIG.setdefault("DE3_V4", {})
        de3_v4_cfg["enabled"] = True
        try:
            get_signal_engine(force_reload=True)
        except Exception as exc:
            logging.warning("Failed to reload DynamicEngine3 engine for filterless v4 mode: %s", exc)
        else:
            logging.info("🧪 FILTERLESS ONLY MODE: forcing DynamicEngine3 to DE3 v4")
            event_logger.log_system_event(
                "MODE",
                "DynamicEngine3 forced to DE3 v4 for filterless live mode",
                {"de3_version": "v4"},
            )

    # Dynamic chop analyzer (tiered thresholds with LTF breakout override)
    chop_analyzer = DynamicChopAnalyzer(client)
    chop_analyzer.calibrate()  # Removed session_name argument
    last_chop_calibration = time.time()

    regimeadaptive_strategy = RegimeAdaptiveStrategy()

    # HIGH PRIORITY - Execute immediately on signal
    # CHANGED: Dynamic Engine stays here. VIX added. Intraday Dip removed.
    fast_strategies = [regimeadaptive_strategy]
    if not filterless_only_mode:
        vix_strategy = VIXReversionStrategy()
        fast_strategies.extend([
            vix_strategy,          # Promoted to Fast
            ImpulseBreakoutStrategy(),
        ])
    ENABLE_DYNAMIC_ENGINE_1 = False
    ENABLE_DYNAMIC_ENGINE_3 = True
    ALLOW_DYNAMIC_ENGINE_SOLO = False
    if ENABLE_DYNAMIC_ENGINE_1 and not filterless_only_mode:
        dynamic_engine_strat = DynamicEngineStrategy()
        fast_strategies.append(dynamic_engine_strat)  # Kept in Fast (Not Demoted)
    if ENABLE_DYNAMIC_ENGINE_3:
        dynamic_engine3_strat = DynamicEngine3Strategy()
        fast_strategies.append(dynamic_engine3_strat)

    # STANDARD PRIORITY - Normal execution
    ml_strategy = None
    standard_strategies = []
    if not filterless_only_mode:
        smt_strategy = SMTStrategy()
        standard_strategies = [
            IntradayDipStrategy(), # DEMOTED to Standard
            AuctionReversionStrategy(),
            LiquiditySweepStrategy(),
            ValueAreaBreakoutStrategy(),
            ConfluenceStrategy(),
            smt_strategy,
            SmoothTrendAsiaStrategy(),
        ]

    ml_filterless_disabled = filterless_only_mode and "ml_physics" in filterless_disabled
    if not ml_filterless_disabled:
        ml_strategy = MLPhysicsStrategy()
        # Only add ML strategy if at least one model loaded successfully
        if ml_strategy.model_loaded:
            standard_strategies.append(ml_strategy)
        else:
            print(f"⚠️ MLPhysicsStrategy disabled - no session model files found")
    else:
        logging.info("MLPhysicsStrategy skipped during live startup because it is disabled in filterless mode")

    manifold_cfg = CONFIG.get("MANIFOLD_STRATEGY", {}) or {}
    if bool(manifold_cfg.get("enabled_live", False)):
        manifold_strategy = _load_manifold_strategy_runtime()()
        if getattr(manifold_strategy, "model_loaded", False):
            standard_strategies.append(manifold_strategy)
        else:
            logging.warning("⚠️ ManifoldStrategy enabled_live but model artifact is missing.")
    aetherflow_cfg = CONFIG.get("AETHERFLOW_STRATEGY", {}) or {}
    if bool(aetherflow_cfg.get("enabled_live", False)):
        aetherflow_strategy = _load_aetherflow_strategy_runtime()()
        if getattr(aetherflow_strategy, "model_loaded", False):
            standard_strategies.append(aetherflow_strategy)
            logging.info("AetherFlowStrategy initialized for live execution")
        else:
            logging.warning("⚠️ AetherFlowStrategy enabled_live but model artifact is missing.")
    
    # LOW PRIORITY / LOOSE EXECUTION - Wait for next bar
    loose_strategies = [] if filterless_only_mode else [OrbStrategy()]
    ict_cfg = CONFIG.get("ICT_MODEL", {}) or {}
    if bool(ict_cfg.get("enabled_live", False)) and not filterless_only_mode:
        loose_strategies.append(ICTModelStrategy())
    elif filterless_only_mode and bool(ict_cfg.get("enabled_live", False)):
        logging.info("ICTModelStrategy disabled in filterless mode")
    else:
        logging.info("ICTModelStrategy disabled in live via config")

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

    disabled_signal_log_ts: dict[str, float] = {}

    def execution_disabled_filter(strategy_label: str, session_name: str) -> Optional[str]:
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

    def maybe_log_disabled_strategy(filter_label: str, side: str, strategy_label: str) -> None:
        # Throttle to avoid spam: log at most once every 5 minutes per label.
        if not filter_label:
            return
        now = time.time()
        last = disabled_signal_log_ts.get(filter_label)
        if last is None or (now - last) >= 300.0:
            disabled_signal_log_ts[filter_label] = now
            event_logger.log_filter_check(
                "StrategyDisabled",
                side or "ALL",
                False,
                filter_label,
                strategy=strategy_label,
            )
     
    # Initialize filters
    BankLevelQuarterFilter = None
    MemorySRFilter = None
    LegacyFilterSystem = None
    FilterArbitrator = None
    RegimeManifoldEngine = None
    apply_meta_policy_fn = None
    evaluate_pre_signal_gate_fn = None
    RejectionFilterCls = _NoOpRejectionFilter
    ChopFilterCls = _NoOpStatefulRuntime
    ExtensionFilterCls = _NoOpStatefulRuntime
    TrendFilterCls = _NoOpTrendFilter
    DynamicStructureBlockerCls = _NoOpStatefulRuntime
    RegimeStructureBlockerCls = _NoOpStatefulRuntime
    PenaltyBoxBlockerCls = _NoOpStatefulRuntime
    DirectionalLossBlockerCls = _NoOpDirectionalLossBlocker
    ImpulseFilterCls = _NoOpImpulseFilter
    HTFFVGFilterCls = _NoOpHTFFVGFilter
    if filter_stack_runtime_enabled or gemini_runtime_enabled:
        (
            BankLevelQuarterFilter,
            MemorySRFilter,
            LegacyFilterSystem,
            FilterArbitrator,
            RegimeManifoldEngine,
            apply_meta_policy_fn,
            evaluate_pre_signal_gate_fn,
        ) = _load_filter_stack_runtime()
    if filter_stack_runtime_enabled:
        (
            RejectionFilterCls,
            ChopFilterCls,
            ExtensionFilterCls,
            TrendFilterCls,
            DynamicStructureBlockerCls,
            RegimeStructureBlockerCls,
            PenaltyBoxBlockerCls,
            DirectionalLossBlockerCls,
            ImpulseFilterCls,
            HTFFVGFilterCls,
        ) = _load_strategy_filter_runtime()
    rejection_filter = RejectionFilterCls()
    bank_filter = (
        BankLevelQuarterFilter()
        if (filter_stack_runtime_enabled or gemini_runtime_enabled)
        else None
    )
    structural_tracker = StructuralLevelTracker()
    _init_pct_level_overlay()
    _init_regime_classifier()
    _init_loss_factor_guard()
    _init_signal_gate_2025()  # filter G (ML classifier) — toggle via JULIE_SIGNAL_GATE_2025
    # ML overlays: LFO (replacement for rule-based LevelFillOptimizer) and
    # PCT overlay classifier. Both run in SHADOW mode by default (log-only);
    # flip JULIE_ML_LFO_ACTIVE=1 / JULIE_ML_PCT_ACTIVE=1 to let them steer.
    try:
        from ml_overlay_shadow import init_ml_overlays as _init_ml_overlays
        _init_ml_overlays()
    except Exception as _exc:
        logging.warning("ml_overlay_shadow init failed: %s", _exc)
    _lfo_enabled = bool(CONFIG.get("LEVEL_FILL_OPTIMIZER_ENABLED", True))
    level_fill_optimizer = LevelFillOptimizer() if _lfo_enabled else None
    pending_level_fills: dict = {}   # uid → {signal, target_price, …}
    chop_filter = ChopFilterCls(lookback=20)
    extension_filter = ExtensionFilterCls()
    # 4-Tier Trend Filter (merged with Impulse logic)
    # Tier 1: Volume-supported impulse, Tier 2: Standard breakout, Tier 3: Extreme capitulation
    # Tier 4: Macro trend (50/200 EMA alignment) - bypassed by Range Fade logic
    trend_filter = TrendFilterCls()
    htf_fvg_cfg = CONFIG.get("HTF_FVG_FILTER", {}) or {}
    htf_fvg_enabled_live = bool(htf_fvg_cfg.get("enabled_live", True))
    htf_fvg_filter = HTFFVGFilterCls()
    if (not filter_stack_runtime_enabled) and htf_fvg_enabled_live:
        logging.info("🧪 FILTERLESS ONLY MODE: HTF_FVG filter runtime detached")
    elif not htf_fvg_enabled_live:
        logging.info("HTF_FVG filter disabled in live via config")
    structure_blocker = DynamicStructureBlockerCls(
        lookback=50,
        rejection_memory_bars=1,
    )  # Macro trend + fade detection
    regime_blocker = RegimeStructureBlockerCls(lookback=20)      # Regime-based EQH/EQL tolerance
    penalty_cfg = CONFIG.get("PENALTY_BOX", {}) or {}
    penalty_blocker = None
    if penalty_cfg.get("enabled", True):
        penalty_blocker = PenaltyBoxBlockerCls(
            lookback=int(penalty_cfg.get("lookback", 50) or 50),
            tolerance=float(penalty_cfg.get("tolerance", 5.0) or 5.0),
            penalty_bars=int(penalty_cfg.get("penalty_bars", 3) or 3),
        )
    asia_calib_cfg = CONFIG.get("ASIA_CALIBRATIONS", {}) or {}
    asia_calib_enabled = bool(asia_calib_cfg.get("enabled", False))
    penalty_blocker_asia = None
    if asia_calib_enabled:
        asia_penalty_cfg = asia_calib_cfg.get("penalty_box", {}) or {}
        if asia_penalty_cfg.get("enabled", True):
            penalty_blocker_asia = PenaltyBoxBlockerCls(
                lookback=int(asia_penalty_cfg.get("lookback", 50) or 50),
                tolerance=float(asia_penalty_cfg.get("tolerance", 1.5) or 1.5),
                penalty_bars=int(asia_penalty_cfg.get("penalty_bars", 3) or 3),
            )
    memory_sr = (
        MemorySRFilter(lookback_bars=300, zone_width=2.0, touch_threshold=2)
        if (filter_stack_runtime_enabled or gemini_runtime_enabled)
        else None
    )
    news_filter = NewsFilter()
    _cb_max_daily = float(os.environ.get("JULIE_CB_MAX_DAILY_LOSS", "600"))
    _cb_max_consec = int(os.environ.get("JULIE_CB_MAX_CONSEC_LOSSES", "7"))
    _cb_max_trailing = float(os.environ.get("JULIE_CB_MAX_TRAILING_DD", "0"))
    circuit_breaker = CircuitBreaker(
        max_daily_loss=_cb_max_daily,
        max_consecutive_losses=_cb_max_consec,
        max_trailing_dd=_cb_max_trailing,
    )
    import circuit_breaker as _cb_module
    _cb_module._GLOBAL_CB = circuit_breaker  # expose for regime-adaptive retuning
    logging.info(
        "CircuitBreaker init: max_daily_loss=$%.0f max_consec_losses=%d max_trailing_dd=$%.0f",
        _cb_max_daily, _cb_max_consec, _cb_max_trailing,
    )
    directional_loss_blocker = DirectionalLossBlockerCls(
        consecutive_loss_limit=3,
        block_minutes=15,
    )
    impulse_filter = ImpulseFilterCls(lookback=20, impulse_multiplier=2.5)

    # === DUAL-FILTER SYSTEM ===
    # Legacy (Dec 17th) filters for comparison + Arbitrator for override decisions
    legacy_filters = LegacyFilterSystem() if filter_stack_runtime_enabled else None
    filter_arbitrator = (
        FilterArbitrator(confidence_threshold=0.6) if filter_stack_runtime_enabled else None
    )

    # Initialize Gemini Session Optimizer
    optimizer = _load_gemini_runtime()() if gemini_runtime_enabled else None

    # Initialize Rescue Manager
    continuation_manager = ContinuationRescueManager()
    continuation_guard = CONFIG.get("CONTINUATION_GUARD", {}) or {}
    continuation_live_enabled = bool(CONFIG.get("CONTINUATION_ENABLED", True))
    continuation_guard_enabled = continuation_live_enabled and bool(
        continuation_guard.get("enabled", False)
    )
    continuation_signal_mode = str(continuation_guard.get("signal_mode", "calendar") or "calendar").lower()
    continuation_allowlist = None
    if continuation_guard_enabled:
        continuation_allowlist = load_continuation_allowlist(
            continuation_guard.get("allowlist_file")
        )
    continuation_allowed_regimes = {
        str(item).lower()
        for item in (continuation_guard.get("allowed_regimes") or [])
        if item is not None
    }
    continuation_confirm_cfg = continuation_guard.get("confirm", {}) or {}
    continuation_no_bypass = bool(continuation_guard.get("no_bypass", False))

    regime_manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
    regime_manifold_mode = str(regime_manifold_cfg.get("mode", "enforce") or "enforce").lower()
    if regime_manifold_mode not in {"enforce", "shadow"}:
        regime_manifold_mode = "enforce"
    regime_manifold_enabled = bool(
        filter_stack_runtime_enabled and regime_manifold_cfg.get("enabled", False)
    )
    regime_manifold_engine = RegimeManifoldEngine(regime_manifold_cfg) if regime_manifold_enabled else None
    manifold_enforce_side_bias = bool(regime_manifold_cfg.get("enforce_side_bias", True))
    manifold_persist_state = bool(regime_manifold_cfg.get("persist_state", True))
    last_regime_manifold_label = None
    kalshi_price_action_profile: dict = {}
    last_kalshi_overlay_role = None
    last_kalshi_overlay_mode = None
    if regime_manifold_engine is not None:
        logging.info("RegimeManifold active (mode=%s)", regime_manifold_mode)

    flip_conf_cfg = CONFIG.get("FLIP_CONFIDENCE", {}) or {}
    flip_conf_enabled = bool(flip_conf_cfg.get("enabled", False))
    flip_conf_data = None
    if flip_conf_enabled:
        flip_conf_data = load_flip_confidence(
            flip_conf_cfg.get("allowlist_file") or flip_conf_cfg.get("file")
        )
    flip_conf_allowed_filters = {
        str(item)
        for item in (flip_conf_cfg.get("allowed_filters") or [])
        if item is not None
    }

    last_processed_session = None
    last_processed_quarter = None  # Track quarter for quarterly optimization
    last_gemini_regime_key = None
    last_gemini_run_ts = 0.0

    print("\nActive Strategies:")
    print("  [FAST EXECUTION]")
    for strat in fast_strategies: print(f"    • {strat.__class__.__name__}")
    print("  [STANDARD EXECUTION]")
    for strat in standard_strategies: print(f"    • {strat.__class__.__name__}")
    print("  [LOOSE EXECUTION]")
    for strat in loose_strategies: print(f"    • {strat.__class__.__name__}")

    print("\n🚀 AsyncIO Upgrade Active - Launching Independent Tasks...")
    print("  ✓ Heartbeat Task (validates session every 60s)")
    print("  ✓ Position Sync Task (syncs broker position every 30s)")
    print("\nListening for market data (faster polling with async)...")

    mes_csv_appender_enabled = bool(CONFIG.get("LIVE_MES_CSV_APPENDER_ENABLED", False))
    mes_csv_path = str(CONFIG.get("LIVE_MES_CSV_PATH", "ml_mes_et.csv") or "ml_mes_et.csv")
    bar_logger = CsvBarAppender(mes_csv_path, CONFIG.get("TARGET_SYMBOL"), NY_TZ) if mes_csv_appender_enabled else None
    if mes_csv_appender_enabled:
        logging.info(f"MES bar CSV appender enabled: {mes_csv_path}")
    else:
        logging.info("MES bar CSV appender disabled by config (LIVE_MES_CSV_APPENDER_ENABLED=False)")

    trade_factor_logger_enabled = bool(CONFIG.get("LIVE_TRADE_FACTORS_LOGGER_ENABLED", True))
    trade_factor_csv_path = str(
        CONFIG.get("LIVE_TRADE_FACTORS_CSV_PATH", "live_trade_factors.csv") or "live_trade_factors.csv"
    )
    trade_factor_context_bars = int(CONFIG.get("LIVE_TRADE_FACTORS_CONTEXT_BARS", 240) or 240)
    trade_factor_max_numeric_cols = int(CONFIG.get("LIVE_TRADE_FACTORS_MAX_NUMERIC_COLS", 300) or 300)
    trade_factor_logger = (
        TradeFactorCsvLogger(
            trade_factor_csv_path,
            NY_TZ,
            context_bars=trade_factor_context_bars,
            max_numeric_cols=trade_factor_max_numeric_cols,
        )
        if trade_factor_logger_enabled
        else None
    )
    if trade_factor_logger_enabled:
        logging.info(
            f"Live trade-factor logger enabled: {trade_factor_csv_path} "
            f"(context_bars={trade_factor_context_bars}, max_numeric_cols={trade_factor_max_numeric_cols})"
        )
    else:
        logging.info("Live trade-factor logger disabled by config (LIVE_TRADE_FACTORS_LOGGER_ENABLED=False)")

    # === LAUNCH INDEPENDENT ASYNC TASKS ===
    # These tasks run independently and cannot be blocked by strategy calculations
    client._runtime_state_persist_ready = False
    heartbeat = asyncio.create_task(heartbeat_task(client, interval=60))
    position_sync = asyncio.create_task(position_sync_task(client, interval=30))
    sentiment_updater = None
    if sentiment_service is not None and getattr(sentiment_service, "enabled", False):
        def _persist_after_sentiment_poll():
            persist_runtime_state(reason="sentiment_poll")
        sentiment_updater = asyncio.create_task(
            sentiment_monitor_task(sentiment_service, on_poll_complete=_persist_after_sentiment_poll)
        )

    # NEW: Background HTF Updater
    # This keeps your FVG memory fresh without pausing the bot
    htf_updater = None
    if htf_fvg_enabled_live:
        htf_updater = asyncio.create_task(htf_structure_task(client, htf_fvg_filter, interval=60))
    kalshi_updater = None
    kalshi_provider = _get_kalshi_provider()
    if kalshi_provider is not None and getattr(kalshi_provider, "enabled", False):
        kalshi_interval = max(
            15,
            int(_coerce_int((CONFIG.get("KALSHI", {}) or {}).get("cache_ttl"), 120) or 120) // 2,
        )
        kalshi_updater = asyncio.create_task(kalshi_refresh_task(kalshi_provider, interval=kalshi_interval))

    # === TRACKING VARIABLES ===
    # Position sync now handled by independent async task - removed manual tracking
    
    # Track pending signals for delayed execution
    pending_loose_signals = {}
    last_processed_bar = None
    _cb_last_day: Optional[date] = None
    opposite_reversal_cfg = CONFIG.get("LIVE_OPPOSITE_REVERSAL", {}) or {}
    opposite_reversal_required = int(
        max(1, _coerce_int(opposite_reversal_cfg.get("required_confirmations"), 3))
    )
    opposite_reversal_window_bars = int(
        max(1, _coerce_int(opposite_reversal_cfg.get("window_bars"), 3))
    )
    opposite_reversal_require_same_strategy_family = bool(
        opposite_reversal_cfg.get("require_same_strategy_family", True)
    )
    opposite_reversal_require_same_active_trade_family = bool(
        opposite_reversal_cfg.get("require_same_active_trade_family", False)
    )
    opposite_reversal_require_same_sub_strategy = bool(
        opposite_reversal_cfg.get("require_same_sub_strategy", False)
    )
    opposite_reversal_state = {
        "count": 0,
        "side": None,
        "bar_index": None,
        "strategy_family": None,
    }
    pending_impulse_rescues = []

    # Adaptive bank fill: pending reversal signal waiting for bank level touch
    _pending_mlphysics_bank_fill: Optional[dict] = None
    _pending_mlphysics_bank_fill_bars: int = 0
    # Staged bank fill for non-MLPhysics strategies (injected into candidate_signals)
    _staged_bank_fill_candidate: Optional[tuple] = None

    # Early Exit Tracking
    active_trade = None
    parallel_active_trades = []
    bar_count = 0
    flat_position_streak = 0
    position_stale_streak = 0
    recent_closed_trades = []
    seen_closed_trade_keys: set[tuple[Any, ...]] = set()
    max_recent_closed_trades = 40
    last_projectx_trade_backfill_ts: Optional[datetime.datetime] = None

    def tracked_live_trades() -> list[dict]:
        trades: list[dict] = []
        if isinstance(active_trade, dict):
            trades.append(active_trade)
        trades.extend(
            trade for trade in parallel_active_trades
            if isinstance(trade, dict)
        )
        return trades

    def _serialize_live_trade_for_state(trade: Optional[dict]) -> Optional[dict]:
        if not isinstance(trade, dict):
            return None
        serialized: dict[str, Any] = {}
        for key, value in trade.items():
            if isinstance(value, datetime.datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=NY_TZ)
                else:
                    value = value.astimezone(NY_TZ)
                serialized[key] = value.isoformat()
            elif isinstance(value, date):
                serialized[key] = value.isoformat()
            elif isinstance(value, np.generic):
                serialized[key] = value.item()
            else:
                serialized[key] = value
        return serialized

    def _deserialize_live_trade_from_state(raw: Optional[dict]) -> Optional[dict]:
        if not isinstance(raw, dict):
            return None
        trade = dict(raw)
        entry_time = parse_dt(trade.get("entry_time"))
        if isinstance(entry_time, datetime.datetime):
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=NY_TZ)
            else:
                entry_time = entry_time.astimezone(NY_TZ)
            trade["entry_time"] = entry_time
        return trade

    def reset_opposite_reversal_state(reason: Optional[str] = None) -> None:
        previous_count = int(opposite_reversal_state.get("count", 0) or 0)
        previous_side = _normalize_live_side(opposite_reversal_state.get("side"))
        previous_family = str(
            opposite_reversal_state.get("strategy_family") or ""
        ).strip()
        if reason and (previous_count > 0 or previous_side):
            logging.info(
                "Reset opposite reversal confirmation state: %s (count=%s side=%s family=%s)",
                reason,
                previous_count,
                previous_side or "NONE",
                previous_family or "ANY",
            )
        opposite_reversal_state["count"] = 0
        opposite_reversal_state["side"] = None
        opposite_reversal_state["bar_index"] = None
        opposite_reversal_state["strategy_family"] = None

    def note_opposite_reversal_signal(
        signal_payload: Optional[dict],
        current_bar_index: int,
    ) -> Tuple[bool, int]:
        # Re-read required_confirmations from CONFIG each call so the regime
        # classifier (when enabled) can mutate it at runtime.
        _live_rev_cfg = CONFIG.get("LIVE_OPPOSITE_REVERSAL", {}) or {}
        _req_confirms = int(max(1, _coerce_int(_live_rev_cfg.get("required_confirmations"), opposite_reversal_required)))
        confirmed, new_count, next_state = update_live_opposite_reversal_confirmation_state(
            opposite_reversal_state,
            signal_payload,
            current_bar_index,
            required_confirmations=_req_confirms,
            window_bars=opposite_reversal_window_bars,
            require_same_strategy_family=opposite_reversal_require_same_strategy_family,
            require_same_sub_strategy=opposite_reversal_require_same_sub_strategy,
        )
        side = _normalize_live_side(next_state.get("side"))
        if side is None:
            reset_opposite_reversal_state("invalid opposite signal side")
            return False, 0
        opposite_reversal_state.update(next_state)
        strategy_family = str(next_state.get("strategy_family") or "").strip() or "ANY"
        logging.info(
            "Opposite reversal confirmation: %s %s/%s within %s bars | family=%s%s",
            side,
            new_count,
            _req_confirms,
            opposite_reversal_window_bars,
            strategy_family,
            " (confirmed)" if confirmed else "",
        )
        return confirmed, int(new_count)

    def opposite_reversal_matches_active_trade_family(
        signal_payload: Optional[dict],
        active_trades_payload: Optional[list[dict]],
    ) -> bool:
        if not opposite_reversal_require_same_active_trade_family:
            return True
        signal_family = get_live_opposite_reversal_family_key(
            signal_payload,
            require_sub_strategy=opposite_reversal_require_same_sub_strategy,
        )
        if not signal_family:
            return True
        active_families = {
            str(
                get_live_opposite_reversal_family_key(
                    trade,
                    require_sub_strategy=opposite_reversal_require_same_sub_strategy,
                )
                or ""
            ).strip()
            for trade in (active_trades_payload or [])
            if isinstance(trade, dict)
        }
        active_families = {family for family in active_families if family}
        if not active_families:
            return True
        return signal_family in active_families

    def same_strategy_opposite_reversal_blocked(
        signal_payload: Optional[dict],
        active_trades_payload: Optional[list[dict]],
    ) -> bool:
        signal_side = _normalize_live_side(
            signal_payload.get("side") if isinstance(signal_payload, dict) else None
        )
        signal_strategy_key = _live_strategy_identity_key(signal_payload)
        if signal_side is None or not signal_strategy_key:
            return False
        for trade in active_trades_payload or []:
            if not isinstance(trade, dict):
                continue
            trade_side = _normalize_live_side(trade.get("side"))
            if trade_side is None or trade_side == signal_side:
                continue
            if _live_strategy_identity_key(trade) == signal_strategy_key:
                return True
        return False

    def log_same_strategy_opposite_reversal_block(
        signal_payload: Optional[dict],
        active_trades_payload: Optional[list[dict]],
        *,
        prefix: str = "Holding position",
    ) -> None:
        signal_side = _normalize_live_side(
            signal_payload.get("side") if isinstance(signal_payload, dict) else None
        )
        signal_strategy_key = _live_strategy_identity_key(signal_payload) or "UNKNOWN"
        active_matches = sorted(
            {
                str(_live_strategy_identity_key(trade) or "").strip()
                for trade in (active_trades_payload or [])
                if isinstance(trade, dict)
                and _normalize_live_side(trade.get("side")) != signal_side
            }
            - {""}
        )
        active_label = ", ".join(active_matches) if active_matches else "UNKNOWN"
        logging.info(
            "%s: blocked same-strategy opposite reversal for %s against active %s",
            prefix,
            signal_strategy_key,
            active_label,
        )

    def log_opposite_reversal_active_trade_family_block(
        signal_payload: Optional[dict],
        active_trades_payload: Optional[list[dict]],
        *,
        prefix: str = "Holding position",
    ) -> None:
        signal_family = (
            get_live_opposite_reversal_family_key(
                signal_payload,
                require_sub_strategy=opposite_reversal_require_same_sub_strategy,
            )
            or "UNKNOWN"
        )
        active_families = sorted(
            {
                str(
                    get_live_opposite_reversal_family_key(
                        trade,
                        require_sub_strategy=opposite_reversal_require_same_sub_strategy,
                    )
                    or ""
                ).strip()
                for trade in (active_trades_payload or [])
                if isinstance(trade, dict)
            }
            - {""}
        )
        active_family_label = ", ".join(active_families) if active_families else "UNKNOWN"
        logging.info(
            "%s: opposite signal family %s cannot reverse active family %s",
            prefix,
            signal_family,
            active_family_label,
        )

    def _promote_parallel_trade_if_needed() -> None:
        nonlocal active_trade, parallel_active_trades
        if active_trade is None and parallel_active_trades:
            active_trade = parallel_active_trades.pop(0)

    def add_tracked_live_trade(trade: Optional[dict]) -> None:
        nonlocal active_trade, parallel_active_trades
        if not isinstance(trade, dict):
            return
        if active_trade is None:
            active_trade = trade
            return
        parallel_active_trades.append(trade)

    def remove_tracked_live_trade(trade: Optional[dict]) -> None:
        nonlocal active_trade, parallel_active_trades
        if not isinstance(trade, dict):
            return
        if active_trade is trade:
            active_trade = None
            _promote_parallel_trade_if_needed()
            return
        parallel_active_trades = [
            item for item in parallel_active_trades
            if item is not trade
        ]
        _promote_parallel_trade_if_needed()

    # Token refresh now handled by independent heartbeat task

    # Chop state tracking (only log when state changes)
    last_chop_reason = None

    # Hostile day guard (DynamicEngine + Continuation)
    hostile_guard = {
        "enabled": True,
        "max_trades": 3,
        "min_trades": 2,
        "loss_threshold": 2,
    }
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
    trend_day_lockout_until = None
    trend_day_max_sigma = 0.0
    was_news_blocked = False
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
    last_ui_trend_day_tier = None
    last_ui_trend_day_dir = None
    tier1_down_until = None
    tier1_up_until = None
    tier1_seen = False
    sticky_trend_dir = None
    sticky_reclaim_count = 0
    sticky_opposite_count = 0
    last_trend_session = None
    persisted_state = load_bot_state(STATE_PATH)
    set_sentiment_state(normalize_sentiment_state(persisted_state.get("sentiment")))
    live_drawdown_state = _normalize_live_drawdown_state(persisted_state.get("live_drawdown"))
    last_state_save = 0.0
    last_live_drawdown_refresh = 0.0
    state_restored = False

    def log_trade_factor_snapshot(
        *,
        source: str,
        signal_payload: Optional[dict],
        order_details: Optional[dict],
        entry_price: float,
        market_df: pd.DataFrame,
        event_time: datetime.datetime,
        market_price: float,
        base_session_name: Optional[str],
        current_session_name: Optional[str],
        vol_regime_name: Optional[str],
        trend_tier: Optional[int],
        trend_dir: Optional[str],
        strategy_eval: Optional[dict],
        regime_snapshot: Optional[dict],
        allowed_chop_bias: Optional[str],
        asia_viable_flag: Optional[bool],
        runtime_state_extra: Optional[dict] = None,
    ) -> None:
        if trade_factor_logger is None or not isinstance(signal_payload, dict):
            return
        try:
            live_dd_metrics = _current_live_drawdown_metrics(live_drawdown_state)
            runtime_state = {
                "news_blocked": bool(news_blocked),
                "hostile_day_active": bool(hostile_day_active),
                "hostile_day_reason": str(hostile_day_reason or ""),
                "allowed_chop_side": str(allowed_chop_bias or ""),
                "asia_viable": bool(asia_viable_flag) if asia_viable_flag is not None else None,
                "pending_loose_count": int(len(pending_loose_signals)),
                "pending_impulse_rescue_count": int(len(pending_impulse_rescues)),
                "impulse_active": bool(impulse_active),
                "impulse_dir": str(impulse_dir or ""),
                "max_retracement": float(max_retracement),
                "flat_position_streak": int(flat_position_streak),
                "regime_meta": regime_snapshot if isinstance(regime_snapshot, dict) else {},
                "live_drawdown_usd": float(live_dd_metrics.get("current_drawdown_usd", 0.0) or 0.0),
                "live_drawdown_source": str(live_dd_metrics.get("source", "trade_pnl_fallback") or "trade_pnl_fallback"),
            }
            if isinstance(runtime_state_extra, dict) and runtime_state_extra:
                runtime_state.update(runtime_state_extra)

            signal_for_log = dict(signal_payload)
            signal_for_log.setdefault("stop_order_id", getattr(client, "_active_stop_order_id", None))

            trade_factor_logger.log_trade_fill(
                current_time=event_time,
                current_price=float(market_price),
                entry_price=float(entry_price),
                signal=signal_for_log,
                order_details=order_details if isinstance(order_details, dict) else {},
                market_df=market_df,
                source=source,
                base_session=base_session_name,
                current_session=current_session_name,
                vol_regime=vol_regime_name,
                trend_day_tier=trend_tier,
                trend_day_dir=trend_dir,
                strategy_results=strategy_eval if isinstance(strategy_eval, dict) else {},
                runtime_state=runtime_state,
            )
        except Exception as exc:
            logging.warning(f"Trade-factor logger failed ({source}): {exc}")

    def closed_trade_keys(row: Optional[dict]) -> set[tuple[Any, ...]]:
        if not isinstance(row, dict):
            return set()
        strategy = str(row.get("strategy") or "").strip()
        side = str(row.get("side") or "").strip().upper()
        entry_price = round(float(_coerce_float(row.get("entry_price"), 0.0) or 0.0), 4)
        exit_price = round(float(_coerce_float(row.get("exit_price"), 0.0) or 0.0), 4)
        pnl_dollars = round(float(_coerce_float(row.get("pnl_dollars"), 0.0) or 0.0), 2)
        close_time = str(row.get("time") or "").strip()
        order_id = str(row.get("order_id") or "").strip()
        entry_order_id = str(row.get("entry_order_id") or "").strip()
        keys: set[tuple[Any, ...]] = {
            ("snapshot", strategy, side, entry_price, exit_price, pnl_dollars, close_time)
        }
        if order_id:
            keys.add(("order", order_id))
        if entry_order_id:
            keys.add(("entry_order", entry_order_id, strategy, side, exit_price, pnl_dollars))
        return keys

    def _find_projectx_backfill_match_index(closed_trade: Optional[dict]) -> Optional[int]:
        if not isinstance(closed_trade, dict):
            return None
        incoming_order_id = str(closed_trade.get("order_id") or "").strip()
        incoming_entry_order_id = str(closed_trade.get("entry_order_id") or "").strip()
        incoming_side = str(closed_trade.get("side") or "").strip().upper()
        incoming_size = max(0, _coerce_int(closed_trade.get("size"), 0))
        incoming_entry_price = round(float(_coerce_float(closed_trade.get("entry_price"), 0.0) or 0.0), 4)
        incoming_time = parse_dt(closed_trade.get("time"))
        for index in range(len(recent_closed_trades) - 1, -1, -1):
            row = recent_closed_trades[index]
            row_order_id = str(row.get("order_id") or "").strip()
            row_entry_order_id = str(row.get("entry_order_id") or "").strip()
            if incoming_order_id and row_order_id == incoming_order_id:
                return index
            if incoming_entry_order_id and row_entry_order_id == incoming_entry_order_id:
                return index
            row_time = parse_dt(row.get("time"))
            if not isinstance(incoming_time, datetime.datetime) or not isinstance(row_time, datetime.datetime):
                continue
            if abs((row_time - incoming_time).total_seconds()) > 120.0:
                continue
            if str(row.get("side") or "").strip().upper() != incoming_side:
                continue
            row_size = max(0, _coerce_int(row.get("size"), 0))
            if incoming_size > 0 and row_size not in (0, incoming_size):
                continue
            row_entry_price = round(float(_coerce_float(row.get("entry_price"), 0.0) or 0.0), 4)
            if abs(row_entry_price - incoming_entry_price) > 1e-4:
                continue
            row_source = str(row.get("source") or "").strip()
            if incoming_order_id and not row_order_id:
                return index
            if row_source in {"broker_flat_cleanup", "price_snapshot", "projectx_api"}:
                return index
        return None

    def _merge_recent_closed_trade_backfill(closed_trade: Optional[dict]) -> bool:
        nonlocal recent_closed_trades
        if not isinstance(closed_trade, dict):
            return False
        incoming_keys = closed_trade_keys(closed_trade)
        duplicate_index = next(
            (
                index
                for index, row in enumerate(recent_closed_trades)
                if not closed_trade_keys(row).isdisjoint(incoming_keys)
            ),
            None,
        )
        if duplicate_index is None:
            duplicate_index = _find_projectx_backfill_match_index(closed_trade)
        if duplicate_index is not None:
            existing_trade = dict(recent_closed_trades[duplicate_index])
            merged_trade = dict(existing_trade)
            merged_trade.update(
                {
                    key: value
                    for key, value in closed_trade.items()
                    if value not in (None, "")
                }
            )
            if merged_trade == existing_trade:
                return False
            recent_closed_trades[duplicate_index] = merged_trade
            recent_closed_trades.sort(key=lambda row: str(row.get("time") or ""))
            rebuild_seen_closed_trade_keys()
            return True
        recent_closed_trades.append(closed_trade)
        recent_closed_trades.sort(key=lambda row: str(row.get("time") or ""))
        if len(recent_closed_trades) > max_recent_closed_trades:
            recent_closed_trades = recent_closed_trades[-max_recent_closed_trades:]
        rebuild_seen_closed_trade_keys()
        return True

    def backfill_recent_closed_trades_from_projectx(
        current_time: datetime.datetime,
        *,
        force: bool = False,
    ) -> int:
        nonlocal last_projectx_trade_backfill_ts
        if client is None or not hasattr(client, "reconstruct_closed_trades"):
            return 0
        current_time = current_time.astimezone(NY_TZ)
        if (
            not force
            and isinstance(last_projectx_trade_backfill_ts, datetime.datetime)
            and (current_time - last_projectx_trade_backfill_ts) < datetime.timedelta(seconds=45)
        ):
            return 0
        session_start = trading_day_start(current_time)
        search_start = session_start - datetime.timedelta(minutes=5)
        search_end = current_time + datetime.timedelta(minutes=2)
        try:
            reconstructed = client.reconstruct_closed_trades(
                search_start,
                search_end,
                include_stream_trades=True,
            )
        except Exception as exc:
            logging.warning("ProjectX trade-history backfill failed: %s", exc)
            last_projectx_trade_backfill_ts = current_time
            return 0

        updates = 0
        for recovered in reconstructed[-max_recent_closed_trades:]:
            if not isinstance(recovered, dict):
                continue
            exit_time = recovered.get("exit_time")
            if not isinstance(exit_time, datetime.datetime):
                continue
            if exit_time.tzinfo is None:
                exit_time = exit_time.replace(tzinfo=NY_TZ)
            else:
                exit_time = exit_time.astimezone(NY_TZ)
            entry_time = recovered.get("entry_time")
            if isinstance(entry_time, datetime.datetime):
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=NY_TZ)
                else:
                    entry_time = entry_time.astimezone(NY_TZ)
            match_index = _find_projectx_backfill_match_index(
                {
                    "time": exit_time.isoformat(),
                    "side": recovered.get("side"),
                    "size": recovered.get("size"),
                    "entry_price": recovered.get("entry_price"),
                    "order_id": recovered.get("order_id"),
                    "entry_order_id": recovered.get("entry_order_id"),
                    "source": recovered.get("source"),
                }
            )
            existing_row = recent_closed_trades[match_index] if match_index is not None else {}
            strategy_name = str(
                existing_row.get("strategy")
                or recovered.get("strategy")
                or "ProjectXHistoryBackfill"
            )
            sub_strategy_name = existing_row.get("sub_strategy") or recovered.get("sub_strategy")
            combo_key = existing_row.get("combo_key") or recovered.get("combo_key") or sub_strategy_name
            pnl_dollars = float(_coerce_float(recovered.get("pnl_dollars"), 0.0) or 0.0)
            pnl_points = _coerce_float(recovered.get("pnl_points"), None)
            exit_price = _coerce_float(recovered.get("exit_price"), None)
            entry_price = _coerce_float(recovered.get("entry_price"), None)
            if pnl_points is None and entry_price is not None and exit_price is not None:
                if str(recovered.get("side") or "").upper() == "LONG":
                    pnl_points = float(exit_price - entry_price)
                else:
                    pnl_points = float(entry_price - exit_price)
            closed_trade = {
                "time": exit_time.isoformat(),
                "strategy": strategy_name,
                "strategy_label": str(existing_row.get("strategy_label") or strategy_name),
                "sub_strategy": sub_strategy_name,
                "combo_key": combo_key,
                "side": str(recovered.get("side") or ""),
                "size": max(1, _coerce_int(recovered.get("size"), 1)),
                "entry_price": float(entry_price) if entry_price is not None else None,
                "signal_entry_price": existing_row.get("signal_entry_price"),
                "exit_price": float(exit_price) if exit_price is not None else None,
                "pnl_points": float(pnl_points) if pnl_points is not None else None,
                "pnl_dollars": float(pnl_dollars),
                "result": "win" if pnl_dollars > 0.0 else "loss" if pnl_dollars < 0.0 else "flat",
                "source": "projectx_trade_history",
                "order_id": recovered.get("order_id"),
                "entry_order_id": recovered.get("entry_order_id"),
                "opened_at": entry_time.isoformat() if isinstance(entry_time, datetime.datetime) else existing_row.get("opened_at"),
                "de3_management_close_reason": str(existing_row.get("de3_management_close_reason") or ""),
                "de3_management": (
                    dict(existing_row.get("de3_management"))
                    if isinstance(existing_row.get("de3_management"), dict)
                    else {}
                ),
            }
            if not _merge_recent_closed_trade_backfill(closed_trade):
                continue
            updates += 1
            if trade_factor_logger is not None:
                try:
                    trade_factor_logger.annotate_trade_close(
                        trade={
                            "strategy": strategy_name,
                            "sub_strategy": sub_strategy_name,
                            "combo_key": combo_key,
                            "side": str(recovered.get("side") or ""),
                            "size": max(1, _coerce_int(recovered.get("size"), 1)),
                            "entry_price": float(entry_price) if entry_price is not None else None,
                            "entry_time": entry_time,
                            "entry_order_id": recovered.get("entry_order_id"),
                            "entry_mode": existing_row.get("entry_mode", "projectx_backfill"),
                            "vol_regime": existing_row.get("vol_regime"),
                            "tracking_restored": bool(existing_row.get("source") in {"broker_flat_cleanup", "price_snapshot"}),
                        },
                        close_metrics={
                            "source": "projectx_trade_history",
                            "entry_price": float(entry_price) if entry_price is not None else None,
                            "exit_price": float(exit_price) if exit_price is not None else None,
                            "pnl_points": float(pnl_points) if pnl_points is not None else None,
                            "pnl_dollars": float(pnl_dollars),
                            "exit_time": exit_time,
                            "order_id": recovered.get("order_id"),
                            "entry_order_id": recovered.get("entry_order_id"),
                        },
                        close_time=exit_time,
                        close_details={
                            "source": "projectx_trade_history",
                            "entry_order_ids": list(recovered.get("entry_order_ids") or []),
                            "raw_close_rows": list(recovered.get("raw_close_rows") or []),
                        },
                        de3_management=(
                            dict(existing_row.get("de3_management"))
                            if isinstance(existing_row.get("de3_management"), dict)
                            else {}
                        ),
                    )
                except Exception as exc:
                    logging.warning("ProjectX trade-factor backfill failed: %s", exc)

        last_projectx_trade_backfill_ts = current_time
        if updates:
            logging.info("ProjectX trade-history backfill reconciled %s closed trade(s)", updates)
            persist_runtime_state(current_time, reason="projectx_trade_backfill")
        return updates

    def _match_recent_closed_trade_to_tracked_trade(
        trade: Optional[dict],
        *,
        current_time: Optional[datetime.datetime] = None,
    ) -> Optional[dict]:
        if not isinstance(trade, dict):
            return None
        trade_side = str(trade.get("side") or "").strip().upper()
        if trade_side not in {"LONG", "SHORT"}:
            return None
        trade_entry_order_id = str(trade.get("entry_order_id") or "").strip()
        trade_size = max(0, _coerce_int(trade.get("size"), 0))
        trade_entry_price = _coerce_float(
            trade.get("broker_entry_price", trade.get("entry_price")),
            math.nan,
        )
        trade_entry_time = trade.get("entry_time")
        if isinstance(trade_entry_time, datetime.datetime):
            if trade_entry_time.tzinfo is None:
                trade_entry_time = trade_entry_time.replace(tzinfo=NY_TZ)
            else:
                trade_entry_time = trade_entry_time.astimezone(NY_TZ)
        search_now = current_time
        if isinstance(search_now, datetime.datetime):
            if search_now.tzinfo is None:
                search_now = search_now.replace(tzinfo=NY_TZ)
            else:
                search_now = search_now.astimezone(NY_TZ)

        for row in reversed(recent_closed_trades):
            if not isinstance(row, dict):
                continue
            row_side = str(row.get("side") or "").strip().upper()
            if row_side != trade_side:
                continue

            row_entry_order_id = str(row.get("entry_order_id") or "").strip()
            if trade_entry_order_id and row_entry_order_id and row_entry_order_id == trade_entry_order_id:
                return row

            row_size = max(0, _coerce_int(row.get("size"), 0))
            if trade_size > 0 and row_size not in (0, trade_size):
                continue

            row_entry_price = _coerce_float(row.get("entry_price"), math.nan)
            if math.isfinite(trade_entry_price) and math.isfinite(row_entry_price):
                if abs(float(row_entry_price) - float(trade_entry_price)) > 0.26:
                    continue
            elif trade_entry_order_id:
                continue

            row_opened_at = parse_dt(row.get("opened_at"))
            if isinstance(row_opened_at, datetime.datetime):
                if row_opened_at.tzinfo is None:
                    row_opened_at = row_opened_at.replace(tzinfo=NY_TZ)
                else:
                    row_opened_at = row_opened_at.astimezone(NY_TZ)
            row_closed_at = parse_dt(row.get("time"))
            if isinstance(row_closed_at, datetime.datetime):
                if row_closed_at.tzinfo is None:
                    row_closed_at = row_closed_at.replace(tzinfo=NY_TZ)
                else:
                    row_closed_at = row_closed_at.astimezone(NY_TZ)

            if isinstance(trade_entry_time, datetime.datetime) and isinstance(row_closed_at, datetime.datetime):
                if row_closed_at < (trade_entry_time - datetime.timedelta(minutes=2)):
                    continue
                if row_closed_at > (trade_entry_time + datetime.timedelta(hours=12)):
                    continue

            if isinstance(trade_entry_time, datetime.datetime) and isinstance(row_opened_at, datetime.datetime):
                if abs((row_opened_at - trade_entry_time).total_seconds()) > 1800.0:
                    continue
            elif trade_entry_order_id:
                continue

            if isinstance(search_now, datetime.datetime) and isinstance(row_closed_at, datetime.datetime):
                if abs((search_now - row_closed_at).total_seconds()) > 14400.0:
                    continue

            return row
        return None

    def _reconcile_stale_tracked_trades_from_recent_history(
        current_time: datetime.datetime,
        market_price: float,
    ) -> int:
        nonlocal active_trade, parallel_active_trades, flat_position_streak
        recovered = 0
        for trade in list(tracked_live_trades()):
            matched_close = _match_recent_closed_trade_to_tracked_trade(
                trade,
                current_time=current_time,
            )
            if not isinstance(matched_close, dict):
                continue

            close_time = parse_dt(matched_close.get("time"))
            if isinstance(close_time, datetime.datetime):
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=NY_TZ)
                else:
                    close_time = close_time.astimezone(NY_TZ)
            else:
                close_time = current_time

            entry_price = _coerce_float(
                matched_close.get(
                    "entry_price",
                    trade.get("broker_entry_price", trade.get("entry_price")),
                ),
                _coerce_float(trade.get("entry_price"), market_price),
            )
            exit_price = _coerce_float(
                matched_close.get("exit_price"),
                _coerce_float(trade.get("current_stop_price"), market_price),
            )
            pnl_points = _coerce_float(matched_close.get("pnl_points"), math.nan)
            if not math.isfinite(pnl_points):
                if str(trade.get("side") or "").strip().upper() == "LONG":
                    pnl_points = float(exit_price - entry_price)
                else:
                    pnl_points = float(entry_price - exit_price)

            close_metrics = {
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl_points": float(pnl_points),
                "pnl_dollars": float(
                    _coerce_float(matched_close.get("pnl_dollars"), 0.0) or 0.0
                ),
                "exit_time": close_time,
                "order_id": _coerce_int(matched_close.get("order_id"), None),
                "entry_order_id": _coerce_int(matched_close.get("entry_order_id"), None),
                "source": str(
                    matched_close.get("source") or "projectx_trade_history"
                ),
            }
            finalize_live_trade_close(
                trade,
                close_metrics,
                close_time,
                log_prefix="Trade closed (stale-history recovery)",
            )
            remove_tracked_live_trade(trade)
            recovered += 1

        if recovered:
            if not tracked_live_trades():
                reset_opposite_reversal_state("stale history reconciliation")
                client._local_position = {"side": None, "size": 0, "avg_price": 0.0}
                client._active_stop_order_id = None
                client._active_target_order_id = None
                flat_position_streak = 0
            persist_runtime_state(current_time, reason="stale_history_reconciliation")
        return recovered

    def rebuild_seen_closed_trade_keys() -> None:
        nonlocal seen_closed_trade_keys
        seen_closed_trade_keys = set()
        for row in recent_closed_trades:
            seen_closed_trade_keys.update(closed_trade_keys(row))

    def _build_restored_live_trade_from_snapshot(
        current_time: datetime.datetime,
        broker_position: Optional[dict],
        position_snapshot: Optional[dict],
    ) -> Optional[dict]:
        broker_position = broker_position if isinstance(broker_position, dict) else {}
        position_snapshot = position_snapshot if isinstance(position_snapshot, dict) else {}

        side = _normalize_live_side(
            broker_position.get("side", position_snapshot.get("side"))
        )
        size = max(
            0,
            _coerce_int(
                broker_position.get("size", position_snapshot.get("size", 0)),
                0,
            ),
        )
        if side is None or size <= 0:
            return None

        avg_price = _coerce_float(broker_position.get("avg_price"), math.nan)
        snapshot_entry_price = _coerce_float(
            position_snapshot.get("entry_price", position_snapshot.get("avg_price")),
            math.nan,
        )
        entry_price = avg_price if math.isfinite(avg_price) else snapshot_entry_price
        if not math.isfinite(entry_price):
            return None

        bracket_snapshot = client.get_live_bracket_state(
            side=side,
            size=size,
            reference_price=float(entry_price),
            expected_stop_price=_coerce_float(
                position_snapshot.get("stop_price", position_snapshot.get("sl_price")),
                None,
            ),
            expected_target_price=_coerce_float(
                position_snapshot.get("target_price", position_snapshot.get("tp_price")),
                None,
            ),
            prefer_stop_order_id=_coerce_int(position_snapshot.get("stop_order_id"), None),
            prefer_target_order_id=_coerce_int(position_snapshot.get("target_order_id"), None),
            max_cache_age_sec=0.0,
            force_refresh=True,
        )
        stop_price = _coerce_float(
            position_snapshot.get("stop_price", position_snapshot.get("sl_price")),
            math.nan,
        )
        if not math.isfinite(stop_price):
            stop_price = _coerce_float(
                bracket_snapshot.get("stop_price", bracket_snapshot.get("sl_price")),
                math.nan,
            )
        target_price = _coerce_float(
            position_snapshot.get("target_price", position_snapshot.get("tp_price")),
            math.nan,
        )
        if not math.isfinite(target_price):
            target_price = _coerce_float(
                bracket_snapshot.get("target_price", bracket_snapshot.get("tp_price")),
                math.nan,
            )

        if side == "LONG":
            sl_dist = max(0.0, float(entry_price - stop_price)) if math.isfinite(stop_price) else 0.0
            tp_dist = max(0.0, float(target_price - entry_price)) if math.isfinite(target_price) else 0.0
        else:
            sl_dist = max(0.0, float(stop_price - entry_price)) if math.isfinite(stop_price) else 0.0
            tp_dist = max(0.0, float(entry_price - target_price)) if math.isfinite(target_price) else 0.0

        detected_exit_ids: dict[str, Optional[int]] = {
            "stop_order_id": _coerce_int(bracket_snapshot.get("stop_order_id"), None),
            "target_order_id": _coerce_int(bracket_snapshot.get("target_order_id"), None),
        }
        if (
            math.isfinite(stop_price) or math.isfinite(target_price)
        ) and (
            detected_exit_ids.get("stop_order_id") is None
            or detected_exit_ids.get("target_order_id") is None
        ):
            try:
                recovered_exit_ids = client._identify_bracket_order_ids(
                    side=side,
                    size=size,
                    stop_price=float(stop_price) if math.isfinite(stop_price) else None,
                    target_price=float(target_price) if math.isfinite(target_price) else None,
                    prefer_stop_order_id=_coerce_int(position_snapshot.get("stop_order_id"), None),
                    prefer_target_order_id=_coerce_int(position_snapshot.get("target_order_id"), None),
                    max_attempts=1,
                )
                if isinstance(recovered_exit_ids, dict):
                    detected_exit_ids["stop_order_id"] = _coerce_int(
                        detected_exit_ids.get("stop_order_id"),
                        _coerce_int(recovered_exit_ids.get("stop_order_id"), None),
                    )
                    detected_exit_ids["target_order_id"] = _coerce_int(
                        detected_exit_ids.get("target_order_id"),
                        _coerce_int(recovered_exit_ids.get("target_order_id"), None),
                    )
            except Exception as exc:
                logging.warning("Live exit-order recovery failed during tracking restore: %s", exc)

        entry_time = (
            parse_dt(position_snapshot.get("opened_at"))
            or parse_dt(position_snapshot.get("updated_at"))
            or parse_dt(position_snapshot.get("timestamp"))
            or current_time
        )
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=NY_TZ)
        else:
            entry_time = entry_time.astimezone(NY_TZ)

        signal = {
            "strategy": position_snapshot.get("strategy", "RestoredLivePosition"),
            "sub_strategy": position_snapshot.get("sub_strategy"),
            "combo_key": position_snapshot.get("combo_key") or position_snapshot.get("sub_strategy"),
            "side": side,
            "entry_mode": position_snapshot.get("entry_mode", "restored"),
            "rule_id": position_snapshot.get("rule_id"),
            "early_exit_enabled": position_snapshot.get("early_exit_enabled"),
            "vol_regime": position_snapshot.get("vol_regime"),
            "tp_dist": float(tp_dist),
            "sl_dist": float(sl_dist),
            "size": int(size),
        }
        restored_variant_id = str(
            position_snapshot.get("de3_v4_selected_variant_id")
            or position_snapshot.get("sub_strategy")
            or position_snapshot.get("combo_key")
            or ""
        ).strip()
        restored_lane = str(
            position_snapshot.get("de3_v4_selected_lane")
            or _infer_de3_lane_from_variant(restored_variant_id)
            or ""
        ).strip()
        restored_strategy = str(signal.get("strategy", "") or "").strip()
        if restored_lane and not restored_strategy.startswith("DynamicEngine3"):
            signal["strategy"] = "DynamicEngine3"
        if restored_lane:
            signal["de3_version"] = str(position_snapshot.get("de3_version") or "v4")
            signal["de3_v4_selected_variant_id"] = restored_variant_id
            signal["de3_v4_selected_lane"] = restored_lane
            restored_route_id = str(
                position_snapshot.get("de3_v4_selected_route_id")
                or position_snapshot.get("rule_id")
                or ""
            ).strip()
            if restored_route_id:
                signal["de3_v4_selected_route_id"] = restored_route_id
        gate_prob = _coerce_float(position_snapshot.get("gate_prob"), math.nan)
        if math.isfinite(gate_prob):
            signal["gate_prob"] = float(gate_prob)
        gate_threshold = _coerce_float(position_snapshot.get("gate_threshold"), math.nan)
        if math.isfinite(gate_threshold):
            signal["gate_threshold"] = float(gate_threshold)

        target_order_id = _coerce_int(
            detected_exit_ids.get("target_order_id"),
            _coerce_int(position_snapshot.get("target_order_id"), None),
        )
        stop_order_id = _coerce_int(
            detected_exit_ids.get("stop_order_id"),
            _coerce_int(position_snapshot.get("stop_order_id"), None),
        )
        entry_order_id = _coerce_int(
            position_snapshot.get("entry_order_id", position_snapshot.get("order_id")),
            None,
        )
        order_details = {
            "entry_price": float(entry_price),
            "tp_points": float(tp_dist),
            "sl_points": float(sl_dist),
            "tp_price": float(target_price) if math.isfinite(target_price) else None,
            "sl_price": float(stop_price) if math.isfinite(stop_price) else None,
            "size": int(size),
            "broker_order_id": entry_order_id,
            "target_order_id": target_order_id,
        }
        restored_trade = _build_live_active_trade(
            signal,
            order_details,
            float(entry_price),
            entry_time,
            bar_count,
            market_df=None,
            stop_order_id=stop_order_id,
        )
        if not isinstance(restored_trade, dict):
            return None

        _refresh_live_de3_management_profile_flags(restored_trade)
        _merge_restored_live_trade_runtime_state(restored_trade, position_snapshot)
        restored_trade["entry_time"] = entry_time
        restored_trade["bars_held"] = max(
            0,
            _coerce_int(position_snapshot.get("bars_held"), 0),
        )
        restored_trade["entry_mode"] = signal.get("entry_mode", "restored")
        restored_trade["tracking_restored"] = True
        restored_trade["broker_entry_price"] = float(entry_price)
        if entry_order_id is not None:
            restored_trade["entry_order_id"] = entry_order_id
        if target_order_id is not None:
            restored_trade["target_order_id"] = target_order_id
        if stop_order_id is not None:
            restored_trade["stop_order_id"] = stop_order_id
        if math.isfinite(stop_price):
            restored_trade["current_stop_price"] = float(stop_price)
        if math.isfinite(target_price):
            restored_trade["current_target_price"] = float(target_price)
        return restored_trade

    def restore_live_trade_tracking_from_state(
        current_time: datetime.datetime,
        broker_position: Optional[dict] = None,
    ) -> bool:
        nonlocal active_trade, parallel_active_trades

        if tracked_live_trades():
            return True

        broker_position = broker_position if isinstance(broker_position, dict) else {}
        broker_side = _normalize_live_side(broker_position.get("side"))
        broker_size = max(0, _coerce_int(broker_position.get("size"), 0))
        if broker_side is None or broker_size <= 0:
            persisted_live_position = persisted_state.get("live_position")
            persisted_side = _normalize_live_side(
                persisted_live_position.get("side")
                if isinstance(persisted_live_position, dict)
                else None
            )
            persisted_size = max(
                0,
                _coerce_int(
                    persisted_live_position.get("size", 0)
                    if isinstance(persisted_live_position, dict)
                    else 0,
                    0,
                ),
            )
            if persisted_side in {"LONG", "SHORT"} and persisted_size > 0:
                broker_position = client.get_position()
                broker_side = _normalize_live_side(broker_position.get("side"))
                broker_size = max(0, _coerce_int(broker_position.get("size"), 0))
        if broker_side is None or broker_size <= 0:
            return False

        restored_rows = persisted_state.get("tracked_live_trades")
        if isinstance(restored_rows, list):
            restored_trades = [
                trade
                for trade in (
                    _deserialize_live_trade_from_state(row)
                    for row in restored_rows
                )
                if isinstance(trade, dict)
                and _normalize_live_side(trade.get("side")) == broker_side
                and max(0, _coerce_int(trade.get("size"), 0)) > 0
            ]
            restored_size = sum(
                max(0, _coerce_int(trade.get("size"), 0))
                for trade in restored_trades
            )
            if restored_trades and restored_size == broker_size:
                restored_trades.sort(
                    key=lambda trade: trade.get("entry_time") or datetime.datetime.min.replace(tzinfo=NY_TZ)
                )
                for trade in restored_trades:
                    _refresh_live_de3_management_profile_flags(trade)
                active_trade = restored_trades[0]
                parallel_active_trades = restored_trades[1:]
                client._active_stop_order_id = _coerce_int(active_trade.get("stop_order_id"), None)
                client._active_target_order_id = _coerce_int(active_trade.get("target_order_id"), None)
                logging.warning(
                    "Restored %s tracked live trade(s) from disk for broker position %s %s.",
                    len(restored_trades),
                    broker_side,
                    broker_size,
                )
                return True
            if restored_trades:
                logging.warning(
                    "Persisted tracked trades size (%s) does not match broker size (%s); rebuilding live tracker.",
                    restored_size,
                    broker_size,
                )

        restored_trade = _build_restored_live_trade_from_snapshot(
            current_time,
            broker_position,
            persisted_state.get("live_position"),
        )
        if not isinstance(restored_trade, dict):
            restored_trade = _build_restored_live_trade_from_snapshot(
                current_time,
                broker_position,
                {},
            )
        if not isinstance(restored_trade, dict):
            return False

        active_trade = restored_trade
        parallel_active_trades = []
        client._active_stop_order_id = _coerce_int(restored_trade.get("stop_order_id"), None)
        client._active_target_order_id = _coerce_int(restored_trade.get("target_order_id"), None)
        logging.warning(
            "Reconstructed tracked live trade from broker position: %s %s @ %.2f | strategy=%s",
            broker_side,
            broker_size,
            float(_coerce_float(restored_trade.get("entry_price"), 0.0) or 0.0),
            restored_trade.get("strategy", "RestoredLivePosition"),
        )
        return True

    def infer_live_de3_management_close_reason(
        trade: Optional[dict],
        close_source: str,
    ) -> str:
        if not isinstance(trade, dict):
            return ""
        strategy_name = str(trade.get("strategy", "") or "")
        if not strategy_name.startswith("DynamicEngine3"):
            return ""
        source_lower = str(close_source or "").strip().lower()
        if "target" in source_lower:
            return "target_fill"
        if "early_exit" in source_lower or "market_order" in source_lower:
            return "early_exit"
        stop_like = "stop" in source_lower
        if not stop_like:
            return ""
        if bool(trade.get("de3_profit_milestone_reached", False)):
            return "profit_milestone_stop"
        if bool(trade.get("de3_entry_trade_day_extreme_reached", False)):
            return "entry_trade_day_extreme_stop"
        if bool(trade.get("de3_break_even_applied", False)):
            return "break_even_stop"
        return "initial_stop_loss"

    def build_live_de3_management_log_payload(
        trade: Optional[dict],
        close_source: str,
    ) -> dict:
        if not isinstance(trade, dict):
            return {}
        strategy_name = str(trade.get("strategy", "") or "")
        if not strategy_name.startswith("DynamicEngine3"):
            return {}
        close_reason = infer_live_de3_management_close_reason(trade, close_source)
        def finite_or_none(value: Any) -> Optional[float]:
            out = _coerce_float(value, math.nan)
            return float(out) if math.isfinite(out) else None
        return {
            "close_reason": close_reason,
            "trade_management_enabled": bool(trade.get("de3_trade_management_enabled", False)),
            "break_even_enabled": bool(trade.get("de3_break_even_enabled", False)),
            "break_even_applied": bool(trade.get("de3_break_even_applied", False)),
            "break_even_armed": bool(trade.get("de3_break_even_armed", False)),
            "break_even_move_count": int(_coerce_int(trade.get("de3_break_even_move_count"), 0)),
            "break_even_last_stop_price": finite_or_none(trade.get("de3_break_even_last_stop_price")),
            "break_even_trigger_bar_index": _coerce_int(
                trade.get("de3_break_even_trigger_bar_index"),
                None,
            ),
            "break_even_trigger_mfe_points": finite_or_none(trade.get("de3_break_even_trigger_mfe_points")),
            "break_even_locked_points": finite_or_none(trade.get("de3_break_even_locked_points")),
            "profit_milestone_profile_active": bool(
                trade.get("de3_profit_milestone_profile_active", False)
            ),
            "profit_milestone_profile_name": str(
                trade.get("de3_profit_milestone_profile_name", "") or ""
            ),
            "profit_milestone_price": finite_or_none(trade.get("de3_profit_milestone_price")),
            "profit_milestone_trigger_pct": finite_or_none(trade.get("de3_profit_milestone_trigger_pct")),
            "profit_milestone_reached": bool(
                trade.get("de3_profit_milestone_reached", False)
            ),
            "profit_milestone_reached_bar_index": _coerce_int(
                trade.get("de3_profit_milestone_reached_bar_index"),
                None,
            ),
            "profit_milestone_reached_mfe_points": finite_or_none(
                trade.get("de3_profit_milestone_reached_mfe_points")
            ),
            "entry_trade_day_extreme_profile_active": bool(
                trade.get("de3_entry_trade_day_extreme_profile_active", False)
            ),
            "entry_trade_day_extreme_profile_name": str(
                trade.get("de3_entry_trade_day_extreme_profile_name", "") or ""
            ),
            "entry_trade_day_high": finite_or_none(trade.get("de3_entry_trade_day_high")),
            "entry_trade_day_low": finite_or_none(trade.get("de3_entry_trade_day_low")),
            "entry_trade_day_extreme_price": finite_or_none(
                trade.get("de3_entry_trade_day_extreme_price")
            ),
            "entry_trade_day_extreme_progress_pct": finite_or_none(
                trade.get("de3_entry_trade_day_extreme_progress_pct")
            ),
            "entry_trade_day_extreme_target_beyond": bool(
                trade.get("de3_entry_trade_day_extreme_target_beyond", False)
            ),
            "entry_trade_day_extreme_reached": bool(
                trade.get("de3_entry_trade_day_extreme_reached", False)
            ),
            "entry_trade_day_extreme_reached_bar_index": _coerce_int(
                trade.get("de3_entry_trade_day_extreme_reached_bar_index"),
                None,
            ),
            "entry_trade_day_extreme_reached_mfe_points": finite_or_none(
                trade.get("de3_entry_trade_day_extreme_reached_mfe_points")
            ),
            "entry_trade_day_extreme_size_adjustment_active": bool(
                trade.get("de3_entry_trade_day_extreme_size_adjustment_active", False)
            ),
            "entry_trade_day_extreme_size_adjustment_applied": bool(
                trade.get("de3_entry_trade_day_extreme_size_adjustment_applied", False)
            ),
            "entry_trade_day_extreme_size_adjustment_profile_name": str(
                trade.get("de3_entry_trade_day_extreme_size_adjustment_profile_name", "") or ""
            ),
            "entry_trade_day_extreme_size_adjustment_target_beyond": bool(
                trade.get("de3_entry_trade_day_extreme_size_adjustment_target_beyond", False)
            ),
            "entry_trade_day_extreme_size_adjustment_progress_pct": finite_or_none(
                trade.get("de3_entry_trade_day_extreme_size_adjustment_progress_pct")
            ),
            "entry_trade_day_extreme_size_adjustment_requested_size": _coerce_int(
                trade.get("de3_entry_trade_day_extreme_size_adjustment_requested_size"),
                None,
            ),
            "entry_trade_day_extreme_size_adjustment_final_size": _coerce_int(
                trade.get("de3_entry_trade_day_extreme_size_adjustment_final_size"),
                None,
            ),
            "entry_trade_day_extreme_size_adjustment_multiplier": finite_or_none(
                trade.get("de3_entry_trade_day_extreme_size_adjustment_multiplier")
            ),
            "entry_trade_day_extreme_early_exit_profile_active": bool(
                trade.get("de3_entry_trade_day_extreme_early_exit_profile_active", False)
            ),
            "entry_trade_day_extreme_early_exit_target_beyond": bool(
                trade.get("de3_entry_trade_day_extreme_early_exit_target_beyond", False)
            ),
            "early_exit_profile_name": str(
                trade.get("de3_early_exit_profile_name", "") or ""
            ),
            "early_exit_exit_if_not_green_by": _coerce_int(
                trade.get("de3_early_exit_exit_if_not_green_by"),
                None,
            ),
            "early_exit_min_progress_by_bars": _coerce_int(
                trade.get("de3_early_exit_min_progress_by_bars"),
                None,
            ),
            "early_exit_min_progress_pct": finite_or_none(
                trade.get("de3_early_exit_min_progress_pct")
            ),
            "early_exit_max_profit_crosses": _coerce_int(
                trade.get("de3_early_exit_max_profit_crosses"),
                None,
            ),
            "early_exit_profile_max_profit_crosses": _coerce_int(
                trade.get("de3_early_exit_profile_max_profit_crosses"),
                None,
            ),
            "early_exit_trigger_reason": str(
                trade.get("de3_early_exit_trigger_reason", "") or ""
            ),
        }

    def finalize_live_trade_close(
        trade: Optional[dict],
        close_metrics: Optional[dict],
        close_time: datetime.datetime,
        *,
        log_prefix: str,
    ) -> None:
        nonlocal recent_closed_trades, seen_closed_trade_keys
        if not isinstance(trade, dict):
            return
        metrics = dict(close_metrics or {})
        trade_side = str(trade.get("side", "") or "").upper()
        trade_size = max(1, _coerce_int(trade.get("size", 1), 1))
        entry_price = _coerce_float(
            metrics.get("entry_price", trade.get("broker_entry_price", trade.get("entry_price", 0.0))),
            0.0,
        )
        exit_price = _coerce_float(metrics.get("exit_price", entry_price), entry_price)
        pnl_points = _coerce_float(metrics.get("pnl_points", 0.0), 0.0)
        pnl_dollars = _coerce_float(metrics.get("pnl_dollars", 0.0), 0.0)
        pnl_dollars_gross = _coerce_float(metrics.get("pnl_dollars_gross"), None)
        pnl_fee_dollars = _coerce_float(metrics.get("pnl_fee_dollars"), None)
        pnl_dollars_net = _coerce_float(metrics.get("pnl_dollars_net"), pnl_dollars)
        if pnl_dollars_net is None:
            pnl_dollars_net = pnl_dollars
        if pnl_fee_dollars is None:
            if pnl_dollars_gross is not None:
                pnl_fee_dollars = max(0.0, float(pnl_dollars_gross) - float(pnl_dollars_net))
            else:
                pnl_fee_dollars = float(_trade_reporting_round_turn_fee_per_contract() * float(trade_size))
        if pnl_dollars_gross is None:
            pnl_dollars_gross = float(pnl_dollars_net) + float(pnl_fee_dollars)
        trade_label, trade_sub = get_log_strategy_info(trade.get("strategy"), trade)
        if trade_sub:
            trade_label = f"{trade_label}:{trade_sub}"
        close_source = str(metrics.get("source", "price_snapshot") or "price_snapshot")
        de3_management_log = build_live_de3_management_log_payload(trade, close_source)
        close_time_value = metrics.get("exit_time")
        if not isinstance(close_time_value, datetime.datetime):
            close_time_value = close_time
        if close_time_value.tzinfo is None:
            close_time_value = close_time_value.replace(tzinfo=NY_TZ)
        close_time_iso = close_time_value.astimezone(NY_TZ).isoformat()
        signal_entry_price = _coerce_float(trade.get("entry_price"), math.nan)
        closed_trade = {
            "time": close_time_iso,
            "strategy": trade.get("strategy"),
            "strategy_label": trade_label,
            "sub_strategy": trade.get("sub_strategy"),
            "combo_key": trade.get("combo_key") or trade.get("sub_strategy"),
            "side": trade_side,
            "size": trade_size,
            "entry_price": float(entry_price),
            "signal_entry_price": (
                float(signal_entry_price)
                if math.isfinite(signal_entry_price) and abs(signal_entry_price - entry_price) > 1e-9
                else None
            ),
            "exit_price": float(exit_price),
            "pnl_points": float(pnl_points),
            "pnl_dollars": float(pnl_dollars),
            "pnl_dollars_gross": float(pnl_dollars_gross) if pnl_dollars_gross is not None else None,
            "pnl_fee_dollars": float(pnl_fee_dollars) if pnl_fee_dollars is not None else None,
            "pnl_dollars_net": float(pnl_dollars_net) if pnl_dollars_net is not None else None,
            "result": "win" if pnl_dollars > 0.0 else "loss" if pnl_dollars < 0.0 else "flat",
            "source": close_source,
            "order_id": metrics.get("order_id"),
            "entry_order_id": metrics.get("entry_order_id") or trade.get("entry_order_id"),
            "de3_management_close_reason": str(
                de3_management_log.get("close_reason") or ""
            ),
            "de3_management": de3_management_log,
        }
        opened_at = metrics.get("entry_time")
        if not isinstance(opened_at, datetime.datetime):
            opened_at = trade.get("entry_time")
        if isinstance(opened_at, datetime.datetime):
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=NY_TZ)
            closed_trade["opened_at"] = opened_at.astimezone(NY_TZ).isoformat()
        incoming_keys = closed_trade_keys(closed_trade)
        duplicate_index = next(
            (
                index
                for index, row in enumerate(recent_closed_trades)
                if not closed_trade_keys(row).isdisjoint(incoming_keys)
            ),
            None,
        )
        if duplicate_index is not None or any(key in seen_closed_trade_keys for key in incoming_keys):
            if duplicate_index is not None:
                merged_trade = dict(recent_closed_trades[duplicate_index])
                merged_trade.update(
                    {
                        key: value
                        for key, value in closed_trade.items()
                        if value not in (None, "")
                    }
                )
                recent_closed_trades[duplicate_index] = merged_trade
                rebuild_seen_closed_trade_keys()
            logging.debug(
                "Skipping duplicate trade close for order_id=%s entry_order_id=%s",
                closed_trade.get("order_id"),
                closed_trade.get("entry_order_id"),
            )
            return

        update_mom_rescue_score(trade, pnl_points, close_time)
        update_hostile_day_on_close(trade.get("strategy"), pnl_points, close_time)
        directional_loss_blocker.record_trade_result(trade_side, pnl_points, close_time)
        circuit_breaker.update_trade_result(pnl_dollars)
        _lfg_notify_trade_closed(closed_trade)
        _record_live_realized_pnl(live_drawdown_state, pnl_dollars, close_time=close_time)
        _refresh_live_drawdown_from_client(
            client,
            live_drawdown_state,
            event_time=close_time,
            force_refresh=bool(live_drawdown_state.get("balance_pending_refresh", False)),
        )
        live_dd_metrics = _current_live_drawdown_metrics(live_drawdown_state)
        recent_closed_trades = [
            row for row in recent_closed_trades
            if closed_trade_keys(row).isdisjoint(incoming_keys)
        ]
        recent_closed_trades.append(closed_trade)
        recent_closed_trades.sort(key=lambda row: str(row.get("time") or ""))
        if len(recent_closed_trades) > max_recent_closed_trades:
            recent_closed_trades = recent_closed_trades[-max_recent_closed_trades:]
        rebuild_seen_closed_trade_keys()
        if trade_factor_logger is not None:
            try:
                trade_factor_logger.annotate_trade_close(
                    trade=trade,
                    close_metrics=metrics,
                    close_time=close_time_value,
                    close_details={
                        "log_prefix": log_prefix,
                        **metrics,
                    },
                    de3_management=de3_management_log,
                )
            except Exception as exc:
                logging.warning("Trade-factor logger close annotation failed: %s", exc)
        logging.info(
            f"{log_prefix}: {trade_label} {trade_side} | "
            f"Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | "
            f"PnL: {pnl_points:.2f} pts (${pnl_dollars:.2f}) | source={close_source} | "
            f"order_id={closed_trade.get('order_id') or ''} | "
            f"entry_order_id={closed_trade.get('entry_order_id') or ''} | "
            f"size={closed_trade.get('size') or ''} | "
            f"live_dd=${float(live_dd_metrics.get('current_drawdown_usd', 0.0) or 0.0):.2f} | "
            f"de3_reason={closed_trade.get('de3_management_close_reason') or ''}"
        )

    def sync_tracked_trades_with_broker_state(
        current_time: datetime.datetime,
        market_price: float,
    ) -> None:
        nonlocal active_trade, parallel_active_trades, flat_position_streak, position_stale_streak
        open_trades = tracked_live_trades()
        if not open_trades:
            broker_pos = client._local_position if isinstance(client._local_position, dict) else {}
            broker_side = _normalize_live_side(broker_pos.get("side"))
            broker_size = max(0, _coerce_int(broker_pos.get("size"), 0))
            if broker_side in {"LONG", "SHORT"} and broker_size > 0:
                if restore_live_trade_tracking_from_state(current_time, broker_pos):
                    persist_runtime_state(current_time, reason="tracked_trade_restore")
            flat_position_streak = 0
            position_stale_streak = 0
            return

        state_changed = False

        for trade in list(tracked_live_trades()):
            bracket_updates = _refresh_live_trade_brackets_from_projectx(
                client,
                trade,
                max_cache_age_sec=2.0,
            )
            if bracket_updates:
                trade.update(bracket_updates)
                if trade is active_trade:
                    client._active_stop_order_id = _coerce_int(trade.get("stop_order_id"), None)
                    client._active_target_order_id = _coerce_int(trade.get("target_order_id"), None)
                state_changed = True
            tracked_trade_size = max(1, _coerce_int(trade.get("size", 1), 1))
            exit_specs = [
                ("stop_order_id", "Trade closed (confirmed stop fill)"),
                ("target_order_id", "Trade closed (confirmed target fill)"),
            ]
            for order_key, log_prefix in exit_specs:
                exit_order_id = _coerce_int(trade.get(order_key), None)
                if exit_order_id is None or tracked_trade_size <= 0:
                    continue
                fill_summary = client.get_trade_fill_summary(
                    exit_order_id,
                    start_time=trade.get("entry_time"),
                    end_time=current_time + datetime.timedelta(minutes=2),
                    min_qty=tracked_trade_size,
                )
                if not (isinstance(fill_summary, dict) and bool(fill_summary.get("complete"))):
                    continue
                fill_price = _coerce_float(fill_summary.get("avg_price"), math.nan)
                if not math.isfinite(fill_price):
                    if order_key == "stop_order_id":
                        fill_price = _coerce_float(
                            trade.get("current_stop_price", trade.get("entry_price")),
                            market_price,
                        )
                    else:
                        side_name = str(trade.get("side", "") or "").upper()
                        entry_price = _coerce_float(trade.get("entry_price"), market_price)
                        fill_price = _coerce_float(trade.get("current_target_price"), math.nan)
                        if not math.isfinite(fill_price):
                            tp_dist = _coerce_float(trade.get("tp_dist"), 0.0)
                            fill_price = _derive_live_target_price(entry_price, tp_dist, side_name)
                sibling_order_key = "target_order_id" if order_key == "stop_order_id" else "stop_order_id"
                sibling_order_id = _coerce_int(trade.get(sibling_order_key), None)
                if sibling_order_id is not None and sibling_order_id != exit_order_id:
                    try:
                        client.cancel_order(sibling_order_id)
                    except Exception as exc:
                        logging.warning("Sibling exit-order cleanup failed: %s", exc)
                close_metrics = _reconcile_live_trade_close(
                    client,
                    trade,
                    current_time,
                    fallback_exit_price=fill_price,
                    close_order_id=exit_order_id,
                )
                if not isinstance(close_metrics, dict):
                    close_metrics = _calculate_live_trade_close_metrics_from_price(
                        trade,
                        fill_price,
                        source="order_fill_fallback",
                        exit_time=current_time,
                        order_id=exit_order_id,
                    )
                latest_fill_time = fill_summary.get("latest_fill_time")
                if (
                    isinstance(close_metrics, dict)
                    and isinstance(latest_fill_time, datetime.datetime)
                    and not isinstance(close_metrics.get("exit_time"), datetime.datetime)
                ):
                    close_metrics["exit_time"] = latest_fill_time
                finalize_live_trade_close(
                    trade,
                    close_metrics,
                    current_time,
                    log_prefix=log_prefix,
                )
                remove_tracked_live_trade(trade)
                state_changed = True
                break

        broker_pos = client.get_position()
        if broker_pos.get("stale"):
            position_stale_streak += 1
            recovered_from_history = 0
            if position_stale_streak % 10 == 0:
                auth_temporarily_unavailable = False
                try:
                    auth_temporarily_unavailable = bool(client._auth_temporarily_unavailable())
                except Exception:
                    auth_temporarily_unavailable = False
                if auth_temporarily_unavailable:
                    logging.warning(
                        "Position still stale after %s sync checks; waiting for ProjectX session recovery.",
                        position_stale_streak,
                    )
                else:
                    backfill_recent_closed_trades_from_projectx(current_time, force=True)
                    recovered_from_history = _reconcile_stale_tracked_trades_from_recent_history(
                        current_time,
                        market_price,
                    )
            if recovered_from_history:
                position_stale_streak = 0
                return
            logging.warning("Position stale during broker sync; skipping flat check.")
            if state_changed:
                persist_runtime_state(current_time, reason="tracked_trade_sync_update")
            return
        position_stale_streak = 0

        is_flat = broker_pos.get("side") is None or broker_pos.get("size", 0) == 0
        if is_flat:
            flat_position_streak += 1
        else:
            flat_position_streak = 0
            broker_avg_price = _coerce_float(broker_pos.get("avg_price"), math.nan)
            current_trades = tracked_live_trades()
            if math.isfinite(broker_avg_price) and len(current_trades) == 1:
                current_trades[0]["broker_entry_price"] = float(broker_avg_price)

        # Require two consecutive broker flat reads to avoid transient stream gaps.
        if not is_flat or flat_position_streak < 2:
            if state_changed:
                persist_runtime_state(current_time, reason="tracked_trade_sync_update")
            return

        logging.info("Broker reports flat while tracking live trades; clearing local state (confirmed).")
        close_order_details = getattr(client, "_last_close_order_details", None) or {}
        shared_exit_price = _coerce_float(close_order_details.get("exit_price"), market_price)
        try:
            cancelled = client.cancel_open_exit_orders(
                side=None,
                reason="flat-position cleanup",
            )
            if cancelled:
                logging.info("Cancelled %s orphan exit order(s) after broker-flat detection", cancelled)
        except Exception as exc:
            logging.warning("Exit-order cleanup after broker-flat detection failed: %s", exc)
        for trade in list(tracked_live_trades()):
            close_metrics = _calculate_live_trade_close_metrics_from_price(
                trade,
                shared_exit_price,
                source="broker_flat_cleanup",
                exit_time=current_time,
                order_id=_coerce_int(close_order_details.get("order_id"), None),
            )
            finalize_live_trade_close(
                trade,
                close_metrics,
                current_time,
                log_prefix="Trade closed",
            )
            remove_tracked_live_trade(trade)
        reset_opposite_reversal_state("broker_flat_cleanup")
        client._local_position = {"side": None, "size": 0, "avg_price": 0.0}
        client._active_stop_order_id = None
        client._active_target_order_id = None
        flat_position_streak = 0
        persist_runtime_state(current_time, reason="broker_flat_cleanup")

    def _state_is_fresh(current_time: datetime.datetime) -> bool:
        if not persisted_state or persisted_state.get("version") != STATE_VERSION:
            return False
        saved_start = parse_dt(persisted_state.get("trading_day_start"))
        if saved_start is None:
            return False
        current_start = trading_day_start(current_time.astimezone(NY_TZ))
        if saved_start != current_start:
            return False
        saved_last_bar = parse_dt(persisted_state.get("last_bar_ts"))
        if saved_last_bar is None:
            return False
        try:
            saved_last_bar = saved_last_bar.astimezone(NY_TZ)
        except Exception:
            pass
        current_bar = current_time.astimezone(NY_TZ)
        if saved_last_bar > current_bar:
            return False
        if (current_bar - saved_last_bar) > datetime.timedelta(minutes=5):
            return False
        return True

    def restore_persisted_state(current_time: datetime.datetime) -> None:
        nonlocal state_restored
        nonlocal active_trade, parallel_active_trades
        nonlocal mom_rescue_date, mom_rescue_scores
        nonlocal hostile_day_active, hostile_day_reason, hostile_day_date, hostile_engine_stats
        nonlocal trend_day_tier, trend_day_dir, impulse_day, impulse_active, impulse_dir
        nonlocal impulse_start_price, impulse_extreme, pullback_extreme, max_retracement, bars_since_impulse
        nonlocal last_trend_day_tier, last_trend_day_dir, tier1_down_until, tier1_up_until, tier1_seen
        nonlocal sticky_trend_dir, sticky_reclaim_count, sticky_opposite_count, last_trend_session
        nonlocal trend_day_max_sigma
        nonlocal recent_closed_trades
        nonlocal kalshi_price_action_profile, last_kalshi_overlay_role, last_kalshi_overlay_mode

        if state_restored or not _state_is_fresh(current_time):
            return

        extension_filter.load_state(persisted_state.get("extension_filter"))
        chop_filter.load_state(persisted_state.get("chop_filter"))
        directional_loss_blocker.load_state(persisted_state.get("directional_loss_blocker"))
        circuit_breaker.load_state(persisted_state.get("circuit_breaker"))
        if penalty_blocker is not None:
            penalty_blocker.load_state(persisted_state.get("penalty_box_blocker"))
        if penalty_blocker_asia is not None:
            penalty_blocker_asia.load_state(persisted_state.get("penalty_box_blocker_asia"))
        rejection_filter.load_state(persisted_state.get("rejection_filter"))
        if bank_filter is not None:
            bank_filter.load_state(persisted_state.get("bank_filter"))
        structural_tracker.load_state(persisted_state.get("structural_tracker", {}))
        restored_closed_trades = persisted_state.get("recent_closed_trades")
        if isinstance(restored_closed_trades, list):
            recent_closed_trades = [
                row for row in restored_closed_trades
                if isinstance(row, dict)
            ][-max_recent_closed_trades:]
            rebuild_seen_closed_trade_keys()

        trend_state = persisted_state.get("trend_day", {})
        trend_day_tier = int(trend_state.get("trend_day_tier", trend_day_tier))
        trend_day_dir = trend_state.get("trend_day_dir", trend_day_dir)
        impulse_day_val = trend_state.get("impulse_day")
        if impulse_day_val:
            try:
                impulse_day = date.fromisoformat(impulse_day_val)
            except Exception:
                pass
        impulse_active = bool(trend_state.get("impulse_active", impulse_active))
        impulse_dir = trend_state.get("impulse_dir", impulse_dir)
        impulse_start_price = trend_state.get("impulse_start_price", impulse_start_price)
        impulse_extreme = trend_state.get("impulse_extreme", impulse_extreme)
        pullback_extreme = trend_state.get("pullback_extreme", pullback_extreme)
        max_retracement = float(trend_state.get("max_retracement", max_retracement))
        bars_since_impulse = int(trend_state.get("bars_since_impulse", bars_since_impulse))
        last_trend_day_tier = int(trend_state.get("last_trend_day_tier", last_trend_day_tier))
        last_trend_day_dir = trend_state.get("last_trend_day_dir", last_trend_day_dir)
        try:
            trend_day_max_sigma = float(trend_state.get("trend_day_max_sigma", trend_day_max_sigma))
        except Exception:
            pass
        tier1_down_until = parse_dt(trend_state.get("tier1_down_until")) or tier1_down_until
        tier1_up_until = parse_dt(trend_state.get("tier1_up_until")) or tier1_up_until
        tier1_seen = bool(trend_state.get("tier1_seen", tier1_seen))
        sticky_trend_dir = trend_state.get("sticky_trend_dir", sticky_trend_dir)
        sticky_reclaim_count = int(trend_state.get("sticky_reclaim_count", sticky_reclaim_count))
        sticky_opposite_count = int(trend_state.get("sticky_opposite_count", sticky_opposite_count))
        last_trend_session = trend_state.get("last_trend_session", last_trend_session)

        mom_state = persisted_state.get("mom_rescue", {})
        mom_rescue_date_val = mom_state.get("mom_rescue_date")
        if mom_rescue_date_val:
            try:
                mom_rescue_date = date.fromisoformat(mom_rescue_date_val)
            except Exception:
                pass
        mom_rescue_scores = mom_state.get("mom_rescue_scores", mom_rescue_scores)

        hostile_state = persisted_state.get("hostile_day", {})
        hostile_day_active = bool(hostile_state.get("hostile_day_active", hostile_day_active))
        hostile_day_reason = hostile_state.get("hostile_day_reason", hostile_day_reason)
        hostile_day_date_val = hostile_state.get("hostile_day_date")
        if hostile_day_date_val:
            try:
                hostile_day_date = date.fromisoformat(hostile_day_date_val)
            except Exception:
                pass
        hostile_engine_stats = hostile_state.get("hostile_engine_stats", hostile_engine_stats)

        kalshi_overlay_state = persisted_state.get("kalshi_trade_overlay", {})
        if isinstance(kalshi_overlay_state, dict):
            restored_profile = kalshi_overlay_state.get("profile")
            if isinstance(restored_profile, dict):
                kalshi_price_action_profile = dict(restored_profile)
            restored_role = kalshi_overlay_state.get("last_role")
            if restored_role is not None:
                last_kalshi_overlay_role = str(restored_role or "")

        if regime_manifold_engine is not None and manifold_persist_state:
            try:
                regime_manifold_engine.load_state(persisted_state.get("regime_manifold"))
            except Exception as exc:
                logging.warning("RegimeManifold state restore failed: %s", exc)

        broker_position = client._local_position if isinstance(client._local_position, dict) else {}
        if not restore_live_trade_tracking_from_state(current_time, broker_position):
            active_trade = None
            parallel_active_trades = []

        state_restored = True
        logging.info("✅ Bot state restored from disk")

    def _deactivate_trend_day(reason: str, now: datetime.datetime) -> None:
        nonlocal trend_day_tier, trend_day_dir
        nonlocal last_trend_day_tier, last_trend_day_dir
        nonlocal tier1_down_until, tier1_up_until, tier1_seen
        nonlocal sticky_trend_dir, sticky_opposite_count, sticky_reclaim_count
        nonlocal trend_day_max_sigma

        if trend_day_tier > 0 or trend_day_dir:
            logging.warning(
                f"🛑 TrendDay deactivated: {reason} @ {now.strftime('%Y-%m-%d %H:%M')}"
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

    def build_live_position_snapshot(current_time: datetime.datetime) -> Optional[dict]:
        tracked_trades = tracked_live_trades()
        tracked_trade = tracked_trades[0] if tracked_trades else None
        broker_position = client._local_position if isinstance(client._local_position, dict) else {}
        broker_open_pnl = _coerce_float(broker_position.get("open_pnl"), math.nan)
        side = str(broker_position.get("side") or "").strip().upper()
        size = max(0, _coerce_int(broker_position.get("size", 0), 0))
        snapshot_source = "projectx_api"

        if (side not in {"LONG", "SHORT"} or size <= 0) and isinstance(tracked_trade, dict):
            tracked_side = str(tracked_trade.get("side") or "").strip().upper()
            tracked_size = max(0, _coerce_int(tracked_trade.get("size"), 0))
            tracked_entry = _coerce_float(
                tracked_trade.get("broker_entry_price", tracked_trade.get("entry_price")),
                math.nan,
            )
            if tracked_side in {"LONG", "SHORT"} and tracked_size > 0:
                side = tracked_side
                size = tracked_size
                broker_position = {
                    **broker_position,
                    "side": tracked_side,
                    "size": tracked_size,
                    "avg_price": float(tracked_entry) if math.isfinite(tracked_entry) else broker_position.get("avg_price"),
                    "stale": True,
                }
                snapshot_source = "tracked_live_state"

        if side not in {"LONG", "SHORT"} or size <= 0:
            return None

        current_time = current_time.astimezone(NY_TZ)
        base_session_name, current_session_name = _runtime_session_labels_from_ts(current_time)
        avg_price = _coerce_float(broker_position.get("avg_price"), math.nan)
        if not math.isfinite(avg_price) and isinstance(tracked_trade, dict):
            avg_price = _coerce_float(
                tracked_trade.get("broker_entry_price", tracked_trade.get("entry_price")),
                math.nan,
            )

        current_price = float("nan")
        if hasattr(client, "cached_df") and getattr(client, "cached_df", None) is not None:
            try:
                if not client.cached_df.empty:
                    current_price = _coerce_float(client.cached_df.iloc[-1]["close"], math.nan)
            except Exception:
                current_price = float("nan")

        open_pnl_dollars = broker_open_pnl
        if (
            not math.isfinite(open_pnl_dollars)
            and math.isfinite(avg_price)
            and math.isfinite(current_price)
            and side in {"LONG", "SHORT"}
            and size > 0
        ):
            pnl_points = (
                float(current_price - avg_price)
                if side == "LONG"
                else float(avg_price - current_price)
            )
            open_pnl_dollars = pnl_points * float(_trade_point_value()) * float(size)

        point_value = max(0.01, _trade_point_value())
        live_bracket_state: dict[str, Any] = {}

        snapshot = {
            "source": snapshot_source,
            "updated_at": current_time.isoformat(),
            "base_session": base_session_name,
            "current_session": current_session_name,
            "session": current_session_name or base_session_name,
            "side": side,
            "size": int(size),
            "avg_price": float(avg_price) if math.isfinite(avg_price) else None,
            "entry_price": float(avg_price) if math.isfinite(avg_price) else None,
            "current_price": float(current_price) if math.isfinite(current_price) else None,
            "open_pnl_dollars": float(open_pnl_dollars) if math.isfinite(open_pnl_dollars) else None,
            "open_pnl_points": (
                float(open_pnl_dollars) / float(point_value * size)
                if math.isfinite(open_pnl_dollars) and size > 0
                else None
            ),
            "point_value": float(point_value),
        }

        signal_entry_price = float("nan")
        tp_dist = float("nan")
        current_stop_price = float("nan")
        current_target_price = float("nan")
        sl_dist = float("nan")
        prefer_stop_order_id = None
        prefer_target_order_id = None
        if isinstance(tracked_trade, dict):
            signal_entry_price = _coerce_float(tracked_trade.get("entry_price"), math.nan)
            tp_dist = _coerce_float(tracked_trade.get("tp_dist"), math.nan)
            current_stop_price = _coerce_float(tracked_trade.get("current_stop_price"), math.nan)
            current_target_price = _coerce_float(tracked_trade.get("current_target_price"), math.nan)
            sl_dist = _coerce_float(tracked_trade.get("sl_dist"), math.nan)
            prefer_stop_order_id = _coerce_int(tracked_trade.get("stop_order_id"), None)
            prefer_target_order_id = _coerce_int(tracked_trade.get("target_order_id"), None)
            entry_time = tracked_trade.get("entry_time")
            if isinstance(entry_time, datetime.datetime):
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=NY_TZ)
                snapshot["opened_at"] = entry_time.astimezone(NY_TZ).isoformat()
            snapshot.update(
                {
                    "strategy": tracked_trade.get("strategy"),
                    "sub_strategy": tracked_trade.get("sub_strategy"),
                    "combo_key": tracked_trade.get("combo_key") or tracked_trade.get("sub_strategy"),
                    "entry_mode": tracked_trade.get("entry_mode"),
                    "order_id": tracked_trade.get("entry_order_id"),
                    "entry_order_id": tracked_trade.get("entry_order_id"),
                    "stop_order_id": tracked_trade.get("stop_order_id"),
                    "target_order_id": tracked_trade.get("target_order_id"),
                    "bars_held": int(_coerce_int(tracked_trade.get("bars_held"), 0)),
                    "tracking_restored": bool(tracked_trade.get("tracking_restored", False)),
                    "rule_id": tracked_trade.get("rule_id") or tracked_trade.get("de3_v4_selected_route_id"),
                    "early_exit_enabled": (
                        tracked_trade.get("early_exit_enabled")
                        if tracked_trade.get("early_exit_enabled") is not None
                        else tracked_trade.get("de3_early_exit_enabled")
                    ),
                    "vol_regime": tracked_trade.get("vol_regime"),
                    "gate_prob": _coerce_float(tracked_trade.get("gate_prob"), math.nan),
                    "gate_threshold": _coerce_float(tracked_trade.get("gate_threshold"), math.nan),
                    "signal_entry_price": float(signal_entry_price) if math.isfinite(signal_entry_price) else None,
                    "signal_side": tracked_trade.get("side"),
                    "kalshi_trade_overlay_role": tracked_trade.get("kalshi_trade_overlay_role"),
                    "kalshi_trade_overlay_mode": tracked_trade.get("kalshi_trade_overlay_mode"),
                    "kalshi_entry_probability": _coerce_float(tracked_trade.get("kalshi_entry_probability"), math.nan),
                    "kalshi_probe_probability": _coerce_float(tracked_trade.get("kalshi_probe_probability"), math.nan),
                    "kalshi_momentum_delta": _coerce_float(tracked_trade.get("kalshi_momentum_delta"), math.nan),
                    "kalshi_momentum_retention": _coerce_float(tracked_trade.get("kalshi_momentum_retention"), math.nan),
                    "kalshi_entry_support_score": _coerce_float(tracked_trade.get("kalshi_entry_support_score"), math.nan),
                    "kalshi_tp_anchor_price": _coerce_float(tracked_trade.get("kalshi_tp_anchor_price"), math.nan),
                }
            )
            if _is_de3_v4_trade_management_payload(tracked_trade):
                _copy_present_keys(
                    tracked_trade,
                    snapshot,
                    DE3_LIVE_POSITION_SNAPSHOT_KEYS,
                )
            if not math.isfinite(_coerce_float(snapshot.get("gate_prob"), math.nan)):
                snapshot.pop("gate_prob", None)
            if not math.isfinite(_coerce_float(snapshot.get("gate_threshold"), math.nan)):
                snapshot.pop("gate_threshold", None)
            if not math.isfinite(avg_price) and math.isfinite(signal_entry_price):
                snapshot["entry_price"] = float(signal_entry_price)
        bracket_reference_price = _coerce_float(snapshot.get("entry_price"), math.nan)
        expected_stop_price = current_stop_price
        if not math.isfinite(expected_stop_price):
            expected_stop_price = _derive_live_stop_price(bracket_reference_price, sl_dist, side)
        expected_target_price = current_target_price
        if not math.isfinite(expected_target_price):
            expected_target_price = _derive_live_target_price(bracket_reference_price, tp_dist, side)
        try:
            live_bracket_state = client.get_live_bracket_state(
                side=side,
                size=size,
                reference_price=float(bracket_reference_price) if math.isfinite(bracket_reference_price) else None,
                expected_stop_price=(
                    float(expected_stop_price) if math.isfinite(expected_stop_price) else None
                ),
                expected_target_price=(
                    float(expected_target_price) if math.isfinite(expected_target_price) else None
                ),
                prefer_stop_order_id=prefer_stop_order_id,
                prefer_target_order_id=prefer_target_order_id,
                max_cache_age_sec=2.0,
            )
        except Exception as exc:
            logging.debug("ProjectX live bracket snapshot failed: %s", exc)

        stop_order_id = _coerce_int(
            live_bracket_state.get("stop_order_id"),
            prefer_stop_order_id,
        )
        target_order_id = _coerce_int(
            live_bracket_state.get("target_order_id"),
            prefer_target_order_id,
        )
        if stop_order_id is not None:
            snapshot["stop_order_id"] = stop_order_id
        if target_order_id is not None:
            snapshot["target_order_id"] = target_order_id

        stop_price = _coerce_float(
            live_bracket_state.get("stop_price", live_bracket_state.get("sl_price")),
            current_stop_price,
        )
        if not math.isfinite(stop_price):
            stop_price = _derive_live_stop_price(bracket_reference_price, sl_dist, side)
        if math.isfinite(stop_price):
            snapshot["stop_price"] = float(stop_price)
            snapshot["sl_price"] = float(stop_price)

        target_price = _coerce_float(
            live_bracket_state.get("target_price", live_bracket_state.get("tp_price")),
            current_target_price,
        )
        if not math.isfinite(target_price):
            target_price = _derive_live_target_price(bracket_reference_price, tp_dist, side)
        if math.isfinite(target_price):
            snapshot["target_price"] = float(target_price)
            snapshot["tp_price"] = float(target_price)
        if parallel_active_trades:
            snapshot["parallel_trade_count"] = int(
                len([trade for trade in parallel_active_trades if isinstance(trade, dict)])
            )
            snapshot["parallel_strategies"] = [
                str(trade.get("strategy", "") or "")
                for trade in parallel_active_trades
                if isinstance(trade, dict)
            ]

        bracket_reference = avg_price if math.isfinite(avg_price) else _coerce_float(snapshot.get("entry_price"), math.nan)
        broker_brackets = client.get_live_bracket_snapshot(
            side=side,
            size=size,
            reference_price=bracket_reference if math.isfinite(bracket_reference) else None,
            max_cache_age_sec=15.0,
        )
        if broker_brackets:
            snapshot.update(broker_brackets)

        return snapshot

    def build_persisted_state(current_time: datetime.datetime) -> dict:
        current_time = current_time.astimezone(NY_TZ)
        base_session_name, current_session_name = _runtime_session_labels_from_ts(current_time)
        return {
            "version": STATE_VERSION,
            "timestamp": current_time.isoformat(),
            "base_session": base_session_name,
            "current_session": current_session_name,
            "session": current_session_name or base_session_name,
            "trading_day_start": trading_day_start(current_time).isoformat(),
            "last_bar_ts": current_time.isoformat(),
            "extension_filter": extension_filter.get_state(),
            "chop_filter": chop_filter.get_state(),
            "directional_loss_blocker": directional_loss_blocker.get_state(),
            "circuit_breaker": circuit_breaker.get_state(),
            "penalty_box_blocker": penalty_blocker.get_state() if penalty_blocker is not None else None,
            "penalty_box_blocker_asia": penalty_blocker_asia.get_state() if penalty_blocker_asia is not None else None,
            "rejection_filter": rejection_filter.get_state(),
            "bank_filter": bank_filter.get_state() if bank_filter is not None else None,
            "structural_tracker": structural_tracker.get_state(),
            "trend_day": {
                "trend_day_tier": trend_day_tier,
                "trend_day_dir": trend_day_dir,
                "impulse_day": impulse_day.isoformat() if impulse_day else None,
                "impulse_active": impulse_active,
                "impulse_dir": impulse_dir,
                "impulse_start_price": impulse_start_price,
                "impulse_extreme": impulse_extreme,
                "pullback_extreme": pullback_extreme,
                "max_retracement": max_retracement,
                "bars_since_impulse": bars_since_impulse,
                "last_trend_day_tier": last_trend_day_tier,
                "last_trend_day_dir": last_trend_day_dir,
                "trend_day_max_sigma": trend_day_max_sigma,
                "tier1_down_until": tier1_down_until.isoformat() if tier1_down_until else None,
                "tier1_up_until": tier1_up_until.isoformat() if tier1_up_until else None,
                "tier1_seen": tier1_seen,
                "sticky_trend_dir": sticky_trend_dir,
                "sticky_reclaim_count": sticky_reclaim_count,
                "sticky_opposite_count": sticky_opposite_count,
                "last_trend_session": last_trend_session,
            },
            "mom_rescue": {
                "mom_rescue_date": mom_rescue_date.isoformat() if mom_rescue_date else None,
                "mom_rescue_scores": mom_rescue_scores,
            },
            "hostile_day": {
                "hostile_day_active": hostile_day_active,
                "hostile_day_reason": hostile_day_reason,
                "hostile_day_date": hostile_day_date.isoformat() if hostile_day_date else None,
                "hostile_engine_stats": hostile_engine_stats,
            },
            "live_drawdown": dict(_current_live_drawdown_metrics(live_drawdown_state)),
            "sentiment": dict(get_sentiment_state()),
            "live_position": build_live_position_snapshot(current_time),
            "tracked_live_trades": [
                row
                for row in (
                    _serialize_live_trade_for_state(trade)
                    for trade in tracked_live_trades()
                )
                if isinstance(row, dict)
            ],
            "recent_closed_trades": list(recent_closed_trades),
            "kalshi_trade_overlay": {
                "profile": dict(kalshi_price_action_profile) if isinstance(kalshi_price_action_profile, dict) else {},
                "last_role": last_kalshi_overlay_role,
            },
            "regime_manifold": (
                regime_manifold_engine.get_state()
                if (regime_manifold_engine is not None and manifold_persist_state)
                else None
            ),
        }

    def persist_runtime_state(
        current_time: Optional[datetime.datetime] = None,
        *,
        reason: Optional[str] = None,
    ) -> None:
        nonlocal last_state_save
        snapshot_time = current_time or datetime.datetime.now(NY_TZ)
        save_bot_state(build_persisted_state(snapshot_time), STATE_PATH)
        last_state_save = time.time()
        if reason:
            logging.info("Persisted bot state (%s)", reason)

    def _persist_runtime_state_from_position_sync(_broker_pos=None) -> None:
        if not bool(getattr(client, "_runtime_state_persist_ready", False)):
            return
        persist_runtime_state(
            datetime.datetime.now(NY_TZ),
            reason="position_sync",
        )

    client._persist_runtime_state = _persist_runtime_state_from_position_sync

    async def process_truth_social_emergency_exit(
        current_time: datetime.datetime,
        current_price: float,
    ) -> bool:
        nonlocal active_trade, parallel_active_trades

        snapshot = get_sentiment_state()
        broker_position = (
            client._local_position.copy()
            if isinstance(getattr(client, "_local_position", None), dict)
            else {}
        )
        position_side = _normalize_live_side(broker_position.get("side"))
        if position_side is None:
            tracked = tracked_live_trades()
            if tracked:
                position_side = _normalize_live_side(tracked[0].get("side"))
                broker_position = {
                    "side": position_side,
                    "size": sum(max(0, _coerce_int(trade.get("size"), 0)) for trade in tracked),
                    "avg_price": _coerce_float(tracked[0].get("entry_price"), 0.0),
                }
        exit_reason = _evaluate_truth_social_emergency_exit(snapshot, position_side)
        if not exit_reason:
            return False

        tracked_before_exit = [dict(trade) for trade in tracked_live_trades()]
        entry_price_for_log = (
            _coerce_float(tracked_before_exit[0].get("entry_price"), current_price)
            if tracked_before_exit
            else _coerce_float(broker_position.get("avg_price"), current_price)
        )
        try:
            finbert_conf = _coerce_float(snapshot.get("finbert_confidence"), math.nan)
            event_logger.log_sentiment_event(
                "Truth Social emergency exit triggered",
                {
                    "strategy": "TruthSocialEngine",
                    "side": position_side,
                    "sentiment_score": snapshot.get("sentiment_score"),
                    "finbert_confidence": round(finbert_conf, 4) if math.isfinite(finbert_conf) else None,
                    "post_id": snapshot.get("latest_post_id"),
                    "reason": exit_reason,
                },
                level="WARNING",
            )
        except Exception:
            pass
        event_logger.log_early_exit(
            reason=exit_reason,
            bars_held=max(
                [_coerce_int(trade.get("bars_held"), 0) for trade in tracked_before_exit] or [0]
            ),
            current_price=current_price,
            entry_price=entry_price_for_log,
        )

        position = await client.async_get_position(prefer_stream=True, require_open_pnl=False)
        if not isinstance(position, dict):
            position = {}
        if _normalize_live_side(position.get("side")) is None:
            position = broker_position
        if _normalize_live_side(position.get("side")) is None:
            logging.warning("Truth Social emergency exit fired but no broker position was available to flatten.")
            return False

        if not await client.async_emergency_flatten_position(position, exit_reason):
            logging.warning("Truth Social emergency exit flatten failed; keeping tracked trades intact.")
            return False

        close_order_details = getattr(client, "_last_close_order_details", {}) or {}
        shared_exit_price = _coerce_float(close_order_details.get("exit_price"), current_price)
        for tracked_trade in tracked_before_exit:
            close_order_id = close_order_details.get("order_id")
            if len(tracked_before_exit) == 1:
                close_metrics = _reconcile_live_trade_close(
                    client,
                    tracked_trade,
                    current_time,
                    fallback_exit_price=shared_exit_price,
                    close_order_id=close_order_id,
                )
            else:
                close_metrics = _calculate_live_trade_close_metrics_from_price(
                    tracked_trade,
                    shared_exit_price,
                    source="truth_social_emergency_exit",
                    exit_time=current_time,
                    order_id=_coerce_int(close_order_id, None),
                )
            if not isinstance(close_metrics, dict):
                close_metrics = _calculate_live_trade_close_metrics_from_price(
                    tracked_trade,
                    shared_exit_price,
                    source="truth_social_emergency_exit_fallback",
                    exit_time=current_time,
                    order_id=_coerce_int(close_order_id, None),
                )
            finalize_live_trade_close(
                tracked_trade,
                close_metrics,
                current_time,
                log_prefix="📊 Truth Social emergency exit closed",
            )

        active_trade = None
        parallel_active_trades = []
        persist_runtime_state(current_time, reason="truth_social_emergency_exit")
        return True

    def reset_mom_rescues(day: date) -> None:
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
        current_time: datetime.datetime,
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

    def is_chop_hard_stop(reason: Optional[str]) -> bool:
        if not reason:
            return False
        text = str(reason).lower()
        hard_phrases = (
            "wait for breakout",
            "range too tight",
            "too tight to fade",
        )
        return any(phrase in text for phrase in hard_phrases)

    def update_mom_rescue_score(trade: dict, pnl_points: float, exit_time: datetime.datetime) -> None:
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
        mom_rescue_scores[key] += 1 if pnl_points >= 0 else -1

    def is_hostile_disabled_strategy(signal: dict, fallback_name: Optional[str] = None) -> bool:
        strategy_name = signal.get("strategy") or fallback_name or ""
        if strategy_name in ("DynamicEngine", "DynamicEngineStrategy", "DynamicEngine3"):
            return True
        if strategy_name == "MLPhysics" or strategy_name == "MLPhysicsStrategy":
            return True
        return str(strategy_name).startswith("Continuation_")

    def reset_hostile_day(day: date) -> None:
        nonlocal hostile_day_active, hostile_day_reason, hostile_day_date, hostile_engine_stats
        hostile_day_active = False
        hostile_day_reason = ""
        hostile_day_date = day
        hostile_engine_stats = {
            "DynamicEngine": {"trades": 0, "losses": 0},
            "Continuation": {"trades": 0, "losses": 0},
        }

    def update_hostile_day_on_close(strategy_name: Optional[str], pnl_points: float, exit_time: datetime.datetime) -> None:
        nonlocal hostile_day_active, hostile_day_reason
        if not hostile_guard["enabled"] or exit_time is None:
            return
        day = exit_time.astimezone(NY_TZ).date()
        if hostile_day_date != day:
            reset_hostile_day(day)
        engine_key = None
        if strategy_name in ("DynamicEngine", "DynamicEngineStrategy", "DynamicEngine3"):
            engine_key = "DynamicEngine"
        elif strategy_name and str(strategy_name).startswith("Continuation_"):
            engine_key = "Continuation"
        if engine_key is None:
            return
        stats = hostile_engine_stats[engine_key]
        if stats["trades"] >= hostile_guard["max_trades"]:
            return
        stats["trades"] += 1
        if pnl_points < 0:
            stats["losses"] += 1
        dyn = hostile_engine_stats["DynamicEngine"]
        cont = hostile_engine_stats["Continuation"]
        if (
            dyn["trades"] >= hostile_guard["min_trades"]
            and cont["trades"] >= hostile_guard["min_trades"]
            and dyn["losses"] >= hostile_guard["loss_threshold"]
            and cont["losses"] >= hostile_guard["loss_threshold"]
        ):
            hostile_day_active = True
            hostile_day_reason = (
                f"DynamicEngine {dyn['losses']}/{dyn['trades']} losses + "
                f"Continuation {cont['losses']}/{cont['trades']} losses"
            )
            logging.warning(f"🛑 HOSTILE DAY: {hostile_day_reason} (trading disabled)")

    # === STEP 1: INITIAL DATA LOAD (MAX HISTORY) ===
    event_logger.log_system_event("STARTUP", "⏳ Startup: Fetching 20,000 bar history (MES)...", {"status": "IN_PROGRESS"})
    logging.info("⏳ Startup: Fetching full 20,000 bar history (MES)...")
    # Fetch the maximum allowed history ONCE before the loop starts
    master_df = await client.async_get_market_data(lookback_minutes=20000, force_fetch=True)
    event_logger.log_system_event("STARTUP", f"✅ History Received: {len(master_df)} bars loaded (MES).", {"status": "COMPLETE"})

    master_mnq_df = pd.DataFrame()
    if mnq_client is not None:
        event_logger.log_system_event("STARTUP", "⏳ Startup: Fetching 20,000 bar history (MNQ)...", {"status": "IN_PROGRESS"})
        logging.info("⏳ Startup: Fetching full 20,000 bar history (MNQ)...")
        master_mnq_df = await mnq_client.async_get_market_data(lookback_minutes=20000, force_fetch=True)
        event_logger.log_system_event("STARTUP", f"✅ History Received: {len(master_mnq_df)} bars loaded (MNQ).", {"status": "COMPLETE"})

    if master_df.empty:
        logging.warning("⚠️ Startup fetch returned empty data (MES). Bot will attempt to build history in loop.")
        master_df = pd.DataFrame()

    if bar_logger is not None and not master_df.empty:
        appended = bar_logger.append_from_df(master_df)
        if appended:
            logging.info(f"CSV logger: appended {appended} bars to {bar_logger.csv_path}")

    # --- 10/10 UPGRADE: DYNAMIC VOLATILITY CALIBRATION ---
    # Use the 20,000 bars (approx 2 weeks) to recalibrate the Volatility Map
    # This ensures "High Volatility" means "High relative to TODAY", not 2024.
    if not master_df.empty:
        try:
            volatility_filter.calibrate(master_df)
        except Exception as e:
            logging.error(f"❌ Calibration Failed: {e} (Continuing with static thresholds)")

    # Backfill LFG's rolling bar cache with recent history so filters F + G
    # can score signals from bar 1 (avoids a ~45min cold-start warmup).
    if not master_df.empty:
        try:
            backfill_bars = master_df.tail(120)
            for ts, row in backfill_bars.iterrows():
                _lfg_notify_bar(
                    ts, row.get('close'),
                    open_price=row.get('open'),
                    high_price=row.get('high'),
                    low_price=row.get('low'),
                    volume=row.get('volume'),
                )
            logging.info(
                "LFG bar cache pre-warmed with %d historical bars (filters F+G ready)",
                len(backfill_bars),
            )
        except Exception as exc:
            logging.debug("LFG cache pre-warm failed: %s", exc, exc_info=True)
    # --- END CALIBRATION ---

    if mnq_client is not None and master_mnq_df.empty:
        logging.warning("⚠️ Startup fetch returned empty data (MNQ). Bot will attempt to build history in loop.")
        master_mnq_df = pd.DataFrame()

    # --- NEW: Initialize VIX master dataframe ---
    master_vix_df = pd.DataFrame()
    vix_fetch_toggle = True

    # One-time backfill flag
    data_backfilled = False

    try:
        if _refresh_live_drawdown_from_client(
            client,
            live_drawdown_state,
            event_time=datetime.datetime.now(datetime.timezone.utc),
            force_refresh=True,
        ):
            last_live_drawdown_refresh = time.time()
            dd_metrics = _current_live_drawdown_metrics(live_drawdown_state)
            logging.info(
                "Live drawdown state seeded: dd=$%.2f source=%s balance=%s peak=%s",
                float(dd_metrics.get("current_drawdown_usd", 0.0) or 0.0),
                str(dd_metrics.get("source", "trade_pnl_fallback") or "trade_pnl_fallback"),
                live_drawdown_state.get("balance"),
                live_drawdown_state.get("peak_balance"),
            )
    except Exception as exc:
        logging.warning("Initial live drawdown refresh failed: %s", exc)

    while True:
        try:
            # Token validation now handled by independent heartbeat task

            # === DATA FRESHNESS CHECK (Safety Circuit Breaker) ===
            if not master_df.empty:
                last_bar_time = master_df.index[-1]
                if last_bar_time.tzinfo is None:
                    last_bar_time = last_bar_time.replace(tzinfo=dt_timezone.utc)

                seconds_since_last_update = (datetime.datetime.now(dt_timezone.utc) - last_bar_time).total_seconds()

                if seconds_since_last_update > 300:  # Increased from 60 for low volume periods
                    event_logger.log_error("DATA_STALE", f"🚨 DATA LAG: Last update was {seconds_since_last_update:.0f}s ago. Moving to DEFENSIVE mode.")
                    logging.warning(f"🚨 DATA LAG: Last update was {seconds_since_last_update:.0f}s ago. Attempting to fetch fresh data...")
                    await asyncio.sleep(5)
                    # continue  # Removed: Allow bot to proceed to data fetch even when stale

            # Periodic chop threshold recalibration (default every 4 hours)
            if chop_analyzer.should_recalibrate(last_chop_calibration):
                chop_analyzer.calibrate() # Removed session_name argument
                last_chop_calibration = time.time()

            # === MARKET TIME (Use last CLOSED bar timestamp, not system clock) ===
            market_time_utc = datetime.datetime.now(dt_timezone.utc)
            if not master_df.empty:
                bar_time = master_df.index[-1]
                if bar_time.tzinfo is None:
                    bar_time = bar_time.replace(tzinfo=NY_TZ)
                now_utc = datetime.datetime.now(dt_timezone.utc)
                bar_time_utc = bar_time.astimezone(dt_timezone.utc)
                if bar_time_utc > now_utc and len(master_df) > 1:
                    bar_time = master_df.index[-2]
                    if bar_time.tzinfo is None:
                        bar_time = bar_time.replace(tzinfo=NY_TZ)
                    bar_time_utc = bar_time.astimezone(dt_timezone.utc)
                market_time_utc = bar_time_utc
            market_time_et = market_time_utc.astimezone(NY_TZ)

            # === GLOBAL RISK & NEWS FILTERS ===
            # Reset CB at day boundary so replay doesn't freeze forever after a daily-loss hit
            _cb_today = market_time_et.date()
            if _cb_last_day is not None and _cb_today != _cb_last_day:
                circuit_breaker.reset_daily()
                logging.info("🔄 Circuit Breaker: daily reset for new trading day %s", _cb_today)
            _cb_last_day = _cb_today

            cb_blocked, cb_reason = circuit_breaker.should_block_trade()
            if cb_blocked and not getattr(circuit_breaker, "_cb_logged_today", False):
                logging.info(f"🚫 Circuit Breaker Block: {cb_reason} — new entries paused until daily reset")
                circuit_breaker._cb_logged_today = True
            # NOTE: deliberately no `continue` / no sleep here. When CB trips, we
            # still want bar-state updates (ML features, regime tracking, exits)
            # to run. New entries are gated at the async_place_order sites via
            # circuit_breaker.is_tripped checks. Daily reset on next-day bar
            # clears _cb_logged_today.
            if not cb_blocked:
                circuit_breaker._cb_logged_today = False

            current_time = market_time_utc
            news_blocked, news_reason = news_filter.should_block_trade(current_time)
            if news_blocked:
                logging.info(f"🚫 NEWS WAIT: {news_reason}")
                # Enhanced logging with news filter details
                news_info = {
                    "Status": "BLACKOUT",
                    "Reason": "High-Impact Event"
                }
                # Extract time remaining from reason if available
                if "min" in news_reason:
                    # Try to extract the time remaining
                    import re
                    match = re.search(r'(\d+)\s*min', news_reason)
                    if match:
                        news_info["Wait"] = f"{match.group(1)}m"
                event_logger.log_filter_check("NewsFilter", "ALL", False, news_reason,
                                             additional_info=news_info, strategy="Global")
                if pending_impulse_rescues:
                    pending_impulse_rescues.clear()
                    logging.info("NEWS BLACKOUT: cleared pending impulse rescues")
                if trend_day_tier > 0 or trend_day_dir:
                    _deactivate_trend_day("news blackout", current_time)
                was_news_blocked = True
            elif was_news_blocked:
                trend_day_lockout_until = current_time + datetime.timedelta(minutes=10)
                logging.info(
                    f"🕒 TrendDay lockout until {trend_day_lockout_until.strftime('%Y-%m-%d %H:%M')} "
                    "after news blackout"
                )
                was_news_blocked = False

            # ==========================================
            # 🕒 UPDATED SESSION DETECTION (INTRADAY + MICRO-ZONES)
            # ==========================================
            current_time_et = market_time_et
            hour = current_time_et.hour
            minute = current_time_et.minute

            # 1. Determine Broad Parent Session (for data/config) + micro-session (for live diagnostics)
            base_session, current_session_name = _runtime_session_labels_from_ts(current_time_et)
            base_session = base_session or "POST_MARKET"
            current_session_name = current_session_name or base_session

            # --- OPTIMIZATION TRIGGER (Every Session Quarter) ---
            # Get current quarter (1-4) within the session
            current_quarter = (
                bank_filter.get_quarter(hour, minute, base_session)
                if bank_filter is not None
                else _compute_session_quarter(hour, minute, base_session)
            )

            # Track session/quarter changes for logging; Gemini runs only when regime changes
            session_changed = current_session_name != last_processed_session
            quarter_changed = current_quarter != last_processed_quarter

            # Gemini: Only re-prompt on regime changes (ADX/chop, vol regime, holiday/news context)
            gemini_enabled = bool(gemini_runtime_enabled and optimizer is not None)
            trend_status = "UNKNOWN"
            chop_status = "UNKNOWN"
            vol_regime = None
            holiday_context = "NORMAL_LIQUIDITY"
            seasonal_context = "NORMAL_SEASONAL"
            regime_key = None
            regime_changed = False

            if gemini_enabled:
                # ADX/Chop regime (15m)
                try:
                    if not master_df.empty:
                        df_15m = master_df.resample('15min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
                        if not df_15m.empty:
                            adx_score = optimizer._calculate_adx(df_15m)
                            chop_score = optimizer._calculate_choppiness_index(df_15m)
                            trend_status = "TRENDING" if adx_score > 25 else "CHOPPY/RANGING"
                            chop_status = "CHOPPY" if chop_score >= 61.8 else ("TRENDING" if chop_score <= 38.2 else "NEUTRAL")
                except Exception as e:
                    logging.debug(f"Gemini regime calc failed (ADX/Chop): {e}")

                # Volatility regime
                try:
                    if not master_df.empty:
                        vol_regime, _, _ = volatility_filter.get_regime(master_df, master_df.index[-1])
                except Exception:
                    vol_regime = None

                # Holiday/seasonal context
                try:
                    holiday_context = news_filter.get_holiday_context(current_time)
                except Exception as e:
                    logging.debug(f"Gemini holiday context failed: {e}")
                    holiday_context = "NORMAL_LIQUIDITY"

                try:
                    seasonal_context = news_filter.get_seasonal_context(current_time)
                except Exception as e:
                    logging.debug(f"Gemini seasonal context failed: {e}")
                    seasonal_context = "NORMAL_SEASONAL"

                regime_key = (
                    current_session_name,
                    trend_status,
                    chop_status,
                    vol_regime,
                    holiday_context,
                    seasonal_context,
                    bool(news_blocked),
                )
                regime_changed = regime_key != last_gemini_regime_key

            if regime_changed:
                if session_changed:
                    logging.info(f"🔄 SESSION HANDOVER: {last_processed_session} -> {current_session_name} Q{current_quarter} (Base: {base_session})")
                    if pending_level_fills:
                        if level_fill_optimizer is not None:
                            level_fill_optimizer.clear_all("session handover")
                        pending_level_fills.clear()
                elif quarter_changed:
                    logging.info(f"🔄 QUARTER CHANGE: {current_session_name} Q{last_processed_quarter} -> Q{current_quarter}")
                else:
                    logging.info(
                        f"🔄 REGIME CHANGE: {current_session_name} | trend={trend_status} | chop={chop_status} | "
                        f"vol={vol_regime} | holiday={holiday_context} | seasonal={seasonal_context} | "
                        f"news_blocked={news_blocked}"
                    )

                can_run_gemini = True
                now_ts = time.time()
                # Enforce a minimum cooldown of 30 minutes unless a session reset occurred.
                gemini_min_interval = float(CONFIG.get("GEMINI", {}).get("min_interval_minutes", 0))
                effective_min_interval = max(30.0, gemini_min_interval)
                if not session_changed and effective_min_interval > 0:
                    elapsed = now_ts - last_gemini_run_ts
                    min_interval_sec = effective_min_interval * 60.0
                    if elapsed < min_interval_sec:
                        can_run_gemini = False
                        remaining_min = (min_interval_sec - elapsed) / 60.0
                        logging.info(
                            f"⏳ Gemini cooldown: skipping optimization ({remaining_min:.1f} min remaining)"
                        )

                if gemini_enabled and can_run_gemini:
                    print("\n" + "=" * 60)
                    print(f"🧠 GEMINI OPTIMIZATION - {current_session_name} Q{current_quarter}")
                    print("=" * 60)

                    # 1. Fetch Events & Holiday Context
                    try:
                        raw_events = news_filter.fetch_news()
                        events_str = str(raw_events)
                    except Exception as e:
                        events_str = "Events data unavailable."

                    # Log holiday status
                    if holiday_context == "HOLIDAY_TODAY":
                        logging.info(f"🚨 HOLIDAY STATUS: {holiday_context} - Market closed/dead volume")
                    elif holiday_context.startswith("PRE_HOLIDAY"):
                        days = holiday_context.split("_")[-2]
                        logging.info(f"📅 HOLIDAY STATUS: Bank Holiday in {days} day(s) - Reducing targets")
                    elif holiday_context == "POST_HOLIDAY_RECOVERY":
                        logging.info(f"🔄 HOLIDAY STATUS: {holiday_context} - Volatility expanding")
                    else:
                        logging.info(f"✅ HOLIDAY STATUS: {holiday_context}")

                    # Get Seasonal Context
                    # Log seasonal phase with specific emoji indicators
                    if seasonal_context == "PHASE_1_LAST_GASP":
                        logging.info(f"⚡ SEASONAL PHASE: LAST GASP (Dec 20-23) - High volume, violent trends")
                    elif seasonal_context == "PHASE_2_DEAD_ZONE":
                        logging.info(f"☠️  SEASONAL PHASE: DEAD ZONE (Dec 24-31) - 60% volume drop, broken structure")
                    elif seasonal_context == "PHASE_3_JAN2_REENTRY":
                        logging.info(f"🐻 SEASONAL PHASE: JAN 2 RE-ENTRY - Bearish bias, funds returning")
                    # NORMAL_SEASONAL doesn't need logging

                    # Log Micro-Session Specifics
                    if current_session_name == "NY_LUNCH":
                        logging.info(f"🧟 MICRO-SESSION: ZOMBIE ZONE (10:30-12:30) - Liquidity drops to 58%")
                    elif current_session_name == "NY_CLOSE":
                        logging.info(f"⚠️  MICRO-SESSION: CLOSE TRAP (15:00-16:00) - High volume, mean-reversion")

                    # 2. Get Hardcoded Base Params for Session
                    # CRITICAL: Use 'base_session' to look up CONFIG, not the new Micro-Session name
                    # because your config.py likely only has ASIA, LONDON, NY_AM, NY_PM.
                    session_cfg = CONFIG['SESSIONS'].get(base_session, {})
                    base_sl = session_cfg.get('SL', 4.0)
                    base_tp = session_cfg.get('TP', 8.0)

                    # --- NEW: Generate Structure Context String ---
                    structure_price = master_df.iloc[-1]['close'] if not master_df.empty else 0

                    # 2a. Get Memory S/R (Nearest 2 levels)
                    nearest_supports = sorted(
                        [s for s in memory_sr.supports if s < structure_price],
                        key=lambda x: structure_price - x
                    )[:2]
                    nearest_resistances = sorted(
                        [r for r in memory_sr.resistances if r > structure_price],
                        key=lambda x: x - structure_price
                    )[:2]

                    sr_str = f"Current Price: {structure_price:.2f}\n"
                    sr_str += f"Nearest Support (Memory): {nearest_supports}\n"
                    sr_str += f"Nearest Resistance (Memory): {nearest_resistances}\n"

                    # 2b. Get HTF FVGs (Active Memories)
                    active_fvgs = htf_fvg_filter.memory
                    fvg_str = "Active HTF FVGs:\n"
                    if active_fvgs:
                        for fvg in active_fvgs:
                            dist = 0
                            status = "Away"
                            if fvg['type'] == 'bullish':
                                if fvg['bottom'] <= structure_price <= fvg['top']:
                                    status = "INSIDE ZONE"
                                elif structure_price > fvg['top']:
                                    dist = structure_price - fvg['top']
                                    status = f"{dist:.2f} pts above"
                            else:
                                if fvg['bottom'] <= structure_price <= fvg['top']:
                                    status = "INSIDE ZONE"
                                elif structure_price < fvg['bottom']:
                                    dist = fvg['bottom'] - structure_price
                                    status = f"{dist:.2f} pts below"

                            fvg_str += f" - {fvg['tf']} {fvg['type'].upper()} ({fvg['bottom']:.2f}-{fvg['top']:.2f}): {status}\n"
                    else:
                        fvg_str += " - None nearby\n"

                    full_structure_context = sr_str + "\n" + fvg_str
                    # -----------------------------------------------

                    # 3. Call Gemini with structure context (including seasonal & micro-session)
                    opt_result = optimizer.optimize_new_session(
                        master_df,
                        current_session_name,
                        events_str,
                        base_sl,
                        base_tp,
                        structure_context=full_structure_context,
                        active_fvgs=active_fvgs,
                        holiday_context=holiday_context,
                        seasonal_context=seasonal_context,
                        base_session_name=base_session  # Pass parent session for data slicing
                    )

                    if opt_result:
                        sl_mult = float(opt_result.get('sl_multiplier', 1.0))
                        tp_mult = float(opt_result.get('tp_multiplier', 1.0))
                        # NEW: Extract Chop Multiplier
                        chop_mult = float(opt_result.get('chop_multiplier', 1.0))

                        reason = opt_result.get('reasoning', '')
                        trend_params = opt_result.get('trend_params', {})

                        # 4. Update Global Config & Filters
                        CONFIG['DYNAMIC_SL_MULTIPLIER'] = sl_mult
                        CONFIG['DYNAMIC_TP_MULTIPLIER'] = tp_mult

                        # Update Fixed SL/TP viability overrides (session-level)
                        viability_params = opt_result.get("viability_params") or {}
                        fixed_cfg = CONFIG.setdefault("FIXED_SLTP_FRAMEWORK", {})
                        viability_cfg = fixed_cfg.setdefault("viability", {})
                        runtime_overrides = viability_cfg.setdefault("runtime_overrides", {})
                        if viability_params:
                            runtime_overrides[base_session] = viability_params
                            logging.info(
                                f"🧪 FixedSLTP viability override ({base_session}): {viability_params}"
                            )
                        elif base_session in runtime_overrides:
                            runtime_overrides.pop(base_session, None)
                            logging.info(
                                f"🧹 FixedSLTP viability override cleared ({base_session})"
                            )

                        # NEW: Update DynamicChop Analyzer
                        chop_analyzer.update_gemini_params(chop_mult)

                        # Update Trend Filter with dynamic parameters from Gemini
                        if trend_params:
                            trend_filter.update_dynamic_params(trend_params)

                        # Enhanced logging with holiday context
                        logging.info(f"🎯 NEW MULTIPLIERS | SL: {sl_mult}x | TP: {tp_mult}x | CHOP: {chop_mult}x")
                        logging.info(f"🌊 TREND REGIME: {trend_params.get('regime', 'DEFAULT')}")

                        # Show holiday-specific adjustments if applicable
                        if holiday_context != "NORMAL_LIQUIDITY":
                            if holiday_context == "HOLIDAY_TODAY":
                                logging.info(f"⚠️  HOLIDAY ADJUSTMENTS: Extreme risk reduction (Market closed)")
                            elif holiday_context.startswith("PRE_HOLIDAY"):
                                logging.info(f"⚠️  HOLIDAY ADJUSTMENTS: Targets reduced ~40% (Pre-holiday illiquidity)")
                            elif holiday_context == "POST_HOLIDAY_RECOVERY":
                                logging.info(f"⚠️  HOLIDAY ADJUSTMENTS: Stops widened +12% (Post-holiday volatility)")

                        logging.info(f"📝 REASONING: {reason}")
                        print("=" * 60 + "\n")
                    else:
                        CONFIG['DYNAMIC_SL_MULTIPLIER'] = 1.0
                        CONFIG['DYNAMIC_TP_MULTIPLIER'] = 1.0
                        chop_analyzer.update_gemini_params(1.0)  # Reset on failure
                        fixed_cfg = CONFIG.setdefault("FIXED_SLTP_FRAMEWORK", {})
                        viability_cfg = fixed_cfg.setdefault("viability", {})
                        runtime_overrides = viability_cfg.setdefault("runtime_overrides", {})
                        runtime_overrides.pop(base_session, None)
                        logging.warning("⚠️  Gemini optimization failed - using default multipliers")
                        print("=" * 60 + "\n")

                    last_gemini_run_ts = now_ts

                last_gemini_regime_key = regime_key

            # Always track latest session/quarter, even if Gemini doesn't re-run
            last_processed_session = current_session_name
            last_processed_quarter = current_quarter

            # === STEP 2: INCREMENTAL UPDATE (SEQUENTIAL FETCH) ===
            # Fetch MES first, then MNQ, then VIX immediately after to keep timestamps close
            recent_data = await client.async_get_market_data(lookback_minutes=15, force_fetch=True)
            recent_mnq_data = pd.DataFrame()
            if mnq_client is not None:
                recent_mnq_data = await mnq_client.async_get_market_data(lookback_minutes=15, force_fetch=True)
            # --- NEW: Fetch VIX Data ---
            fetch_vix = bool(vix_client is not None) and vix_fetch_toggle
            vix_fetch_toggle = not vix_fetch_toggle
            if fetch_vix and vix_client is not None:
                recent_vix_data = await vix_client.async_get_market_data(lookback_minutes=15, force_fetch=True)
            else:
                recent_vix_data = pd.DataFrame()

            if not recent_data.empty:
                if bar_logger is not None:
                    appended = bar_logger.append_from_df(recent_data)
                    if appended:
                        logging.debug(f"CSV logger: appended {appended} bars")
                # Append new data to our master history
                master_df = pd.concat([master_df, recent_data])

                # Remove duplicates based on timestamp (keep the newest version of the bar)
                master_df = master_df[~master_df.index.duplicated(keep='last')]

                # Optional: Keep memory safe (limit to 50k bars - deeper than API allows!)
                if len(master_df) > 50000:
                    master_df = master_df.iloc[-50000:]

            if not recent_mnq_data.empty:
                master_mnq_df = pd.concat([master_mnq_df, recent_mnq_data])
                master_mnq_df = master_mnq_df[~master_mnq_df.index.duplicated(keep='last')]
                if len(master_mnq_df) > 50000:
                    master_mnq_df = master_mnq_df.iloc[-50000:]

            # --- NEW: Handle VIX Data ---
            if not recent_vix_data.empty:
                master_vix_df = pd.concat([master_vix_df, recent_vix_data])
                master_vix_df = master_vix_df[~master_vix_df.index.duplicated(keep='last')]
                if len(master_vix_df) > 50000:
                    master_vix_df = master_vix_df.iloc[-50000:]

            # Make sure we have data before proceeding
            if master_df.empty or (mnq_client is not None and master_mnq_df.empty):
                # Early heartbeat - shows bot is alive even when no data available
                if not hasattr(client, '_empty_data_counter'):
                    client._empty_data_counter = 0
                client._empty_data_counter += 1
                if client._empty_data_counter % 30 == 0:
                    print(f"⏳ Waiting for data: {datetime.datetime.now().strftime('%H:%M:%S')} | No bars received (market may be closed or starting up)")
                    logging.info(f"No market data available - attempt #{client._empty_data_counter}")
                await asyncio.sleep(2)
                continue

            # Use master_df for all calculations now
            # This variable now holds 20k+ bars of history
            new_df = master_df
            last_bar_time = new_df.index[-1]
            last_bar_utc = last_bar_time
            if last_bar_utc.tzinfo is None:
                last_bar_utc = last_bar_utc.replace(tzinfo=dt_timezone.utc)
            else:
                last_bar_utc = last_bar_utc.astimezone(dt_timezone.utc)
            now_utc = datetime.datetime.now(dt_timezone.utc)
            if last_bar_utc > now_utc and len(new_df) > 1:
                new_df = new_df.iloc[:-1]
                last_bar_time = new_df.index[-1]

            # === LOCAL RESAMPLING ENGINE ===
            # Resample from our locally maintained deep history
            df_5m = resample_dataframe(new_df, 5)
            df_15m = resample_dataframe(new_df, 15)
            df_60m = resample_dataframe(new_df, 60)
            if not new_df.empty:
                last_bar_time = new_df.index[-1]
                df_5m = trim_incomplete_resample(df_5m, last_bar_time, 5)
                df_15m = trim_incomplete_resample(df_15m, last_bar_time, 15)
                df_60m = trim_incomplete_resample(df_60m, last_bar_time, 60)

            # === ONE-TIME BACKFILL ===
            if not data_backfilled:
                event_logger.log_system_event("STARTUP", "🔄 Restoring filter states from history...", {"type": "BACKFILL", "status": "IN_PROGRESS"})
                logging.info("🔄 Performing one-time backfill of filter state from history...")
                last_ts = new_df.index[-1]
                has_persisted = _state_is_fresh(last_ts)
                has_rejection = bool(persisted_state.get("rejection_filter"))
                has_bank = True if bank_filter is None else bool(persisted_state.get("bank_filter"))
                has_extension = bool(persisted_state.get("extension_filter"))

                if has_persisted and has_rejection and has_bank and has_extension:
                    logging.info("✅ Persisted state valid. Skipping full backfill.")
                else:
                    # Replay the history we just fetched
                    # This restores Midnight ORB, Prev Session, etc. instantly
                    rejection_filter.backfill(new_df)

                    # Backfill extension_filter (prevents Mid-Day Amnesia bug)
                    extension_filter.backfill(new_df)

                    # Also backfill bank_filter (has same update() signature)
                    if bank_filter is not None:
                        for ts, row in new_df.sort_index().iterrows():
                            bank_filter.update(ts, row['high'], row['low'], row['close'])

                restore_persisted_state(last_ts)
                client._runtime_state_persist_ready = True
                backfill_recent_closed_trades_from_projectx(last_ts, force=True)
                data_backfilled = True
                event_logger.log_system_event("STARTUP", "✅ State restored. Bot is ready.", {"status": "READY"})
                logging.info("✅ State restored from history.")

            # === UPDATE FILTERS (BEFORE CHOP CHECK - Prevents Stale Filters) ===
            # These must run before chop check so filters stay current even when choppy
            current_price = new_df.iloc[-1]['close']
            current_time = new_df.index[-1]
            currbar = new_df.iloc[-1]
            is_new_bar = (last_processed_bar is None or current_time > last_processed_bar)
            if is_new_bar and last_processed_bar is not None:
                bar_gap = current_time - last_processed_bar
                if bar_gap > datetime.timedelta(minutes=2):
                    logging.warning(f"BAR JUMP: {bar_gap}. Skipping signal processing for catch-up bar.")
                    last_processed_bar = current_time
                    continue

            if is_new_bar:
                try:
                    kalshi_price_action_profile = analyze_kalshi_recent_price_action(
                        new_df,
                        CONFIG.get("KALSHI_TRADE_OVERLAY", {}),
                    )
                except Exception as exc:
                    logging.warning("Kalshi recent-price-action profile failed: %s", exc)
                    kalshi_price_action_profile = {}
                kalshi_overlay_role = str(kalshi_price_action_profile.get("role", "") or "")
                kalshi_overlay_mode = str(kalshi_price_action_profile.get("mode", "level") or "level")
                if kalshi_overlay_role and (
                    kalshi_overlay_role != last_kalshi_overlay_role
                    or kalshi_overlay_mode != last_kalshi_overlay_mode
                ):
                    event_logger.log_kalshi_regime(
                        kalshi_overlay_mode,
                        kalshi_overlay_role,
                        forward_weight=_coerce_float(kalshi_price_action_profile.get("forward_weight"), math.nan),
                        mean_day_range=_coerce_float(kalshi_price_action_profile.get("mean_day_range_points"), math.nan),
                        max_day_range=_coerce_float(kalshi_price_action_profile.get("max_day_range_points"), math.nan),
                        mean_true_range=_coerce_float(kalshi_price_action_profile.get("mean_true_range_points"), math.nan),
                        mean_flip_rate=_coerce_float(kalshi_price_action_profile.get("mean_flip_rate"), math.nan),
                        trade_days=_coerce_int(kalshi_price_action_profile.get("trade_days_considered"), 0),
                        score=_coerce_int(kalshi_price_action_profile.get("score"), 0),
                    )
                    logging.info(
                        "Kalshi forward regime: %s | weight=%.2f | mean_range=%s | max_range=%s | flip=%s | large_share=%s | days=%s | score=%s"
                        " | today_range=%s | today_net_ratio=%s | today_signal=%s | breach_up=%s | breach_down=%s",
                        kalshi_overlay_mode,
                        float(_coerce_float(kalshi_price_action_profile.get("forward_weight"), 0.0)),
                        kalshi_price_action_profile.get("mean_day_range_points"),
                        kalshi_price_action_profile.get("max_day_range_points"),
                        kalshi_price_action_profile.get("mean_flip_rate"),
                        kalshi_price_action_profile.get("mean_large_bar_share"),
                        kalshi_price_action_profile.get("trade_days_considered"),
                        kalshi_price_action_profile.get("score"),
                        kalshi_price_action_profile.get("today_range_points"),
                        kalshi_price_action_profile.get("today_net_ratio"),
                        kalshi_price_action_profile.get("today_signal"),
                        kalshi_price_action_profile.get("today_breach_up"),
                        kalshi_price_action_profile.get("today_breach_down"),
                    )
                    last_kalshi_overlay_role = kalshi_overlay_role
                    last_kalshi_overlay_mode = kalshi_overlay_mode
                    if bool(getattr(client, "_runtime_state_persist_ready", False)):
                        persist_runtime_state(current_time, reason="kalshi_regime_shift")

            refresh_needed = (
                live_drawdown_state.get("balance") is None
                or bool(live_drawdown_state.get("balance_pending_refresh", False))
                or (time.time() - last_live_drawdown_refresh) >= 60.0
            )
            if refresh_needed:
                if _refresh_live_drawdown_from_client(
                    client,
                    live_drawdown_state,
                    event_time=current_time,
                    force_refresh=bool(live_drawdown_state.get("balance_pending_refresh", False)),
                ):
                    last_live_drawdown_refresh = time.time()

            rejection_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            if bank_filter is not None:
                bank_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            structural_tracker.update(
                current_time,
                float(currbar['open']), float(currbar['high']),
                float(currbar['low']),  float(currbar['close']),
            )
            _update_pct_level_overlay(
                current_time,
                currbar['open'], currbar['high'],
                currbar['low'], currbar['close'],
            )
            # SHADOW ML PCT-overlay — score on every fresh level touch and log
            # the ML bias next to the rule-based bias.
            try:
                import ml_overlay_shadow as _mls_p
                if _mls_p._PCT_PAYLOAD is not None:
                    _pct = _get_pct_level_overlay()
                    if _pct is not None and getattr(_pct.state, "at_level", False):
                        if str(getattr(_pct.state, "last_event", "")).startswith("fresh_touch_"):
                            _res = _mls_p.score_pct_overlay(_pct.state)
                            if _res is not None:
                                _p_bo, _ml_bias = _res
                                logging.info(
                                    "[SHADOW_PCT] rule=%s ml=%s p_breakout=%.3f lvl=%.2f%% dist=%.3f%%",
                                    _pct.state.bias, _ml_bias, _p_bo,
                                    _pct.state.nearest_level or 0.0,
                                    _pct.state.level_distance_pct or 0.0,
                                )
            except Exception as _mls_exc:
                logging.debug("shadow ML PCT scoring failed: %s", _mls_exc, exc_info=True)
            _update_regime_classifier(current_time, currbar['close'])
            # Feed trend-day state into LossFactorGuard so the counter-trend
            # reversal veto can act on it (filter C).
            _lfg_notify_trend_day(trend_day_tier, trend_day_dir)
            # Feed full OHLCV into LFG's rolling cache for filters F + G.
            # F only uses close; G (v2 ML gate) uses wicks/body/volume too.
            _lfg_notify_bar(
                current_time, currbar['close'],
                open_price=currbar.get('open'),
                high_price=currbar.get('high'),
                low_price=currbar.get('low'),
                volume=currbar.get('volume'),
            )
            chop_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            extension_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            structure_blocker.update(new_df)
            regime_blocker.update(new_df)
            active_penalty_blocker = (
                penalty_blocker_asia
                if base_session == "ASIA" and penalty_blocker_asia is not None
                else penalty_blocker
            )
            if active_penalty_blocker is not None:
                active_penalty_blocker.update(new_df)
            if memory_sr is not None:
                memory_sr.update(new_df)
            directional_loss_blocker.update_quarter(current_time)
            impulse_filter.update(new_df)

            # === DYNAMIC CHOP CHECK (Pass Local DFs) ===
            # We pass the locally generated df_60m so the analyzer can use it for breakout shift logic
            is_choppy, chop_reason = chop_analyzer.check_market_state(new_df, df_60m_current=df_60m)

            # Initialize allowed_chop_side for this iteration (Fixes NameError)
            allowed_chop_side = None

            if is_choppy:
                # Check if this is a "Range Fade" permission instead of a hard block
                if "ALLOW_LONG_ONLY" in chop_reason:
                    allowed_chop_side = "LONG"
                    # Do NOT continue; allow the loop to proceed but enforce LONG only
                    if last_chop_reason != chop_reason:
                        logging.info(f"🟣 CHOP RESTRICTION: {chop_reason}")
                        last_chop_reason = chop_reason

                elif "ALLOW_SHORT_ONLY" in chop_reason:
                    allowed_chop_side = "SHORT"
                    # Do NOT continue; allow the loop to proceed but enforce SHORT only
                    if last_chop_reason != chop_reason:
                        logging.info(f"🟣 CHOP RESTRICTION: {chop_reason}")
                        last_chop_reason = chop_reason

                else:
                    # Hard chop is now enforced per-strategy later in the pipeline.
                    # This avoids globally suppressing all candidates before consensus.
                    if last_chop_reason != chop_reason:
                        logging.info(f"🔴 CHOP HARD MODE: {chop_reason}")
                        last_chop_reason = chop_reason
            else:
                # Clear chop state if no restriction active
                if last_chop_reason is not None:
                    logging.info("✅ CHOP RESTRICTION CLEARED")
                    last_chop_reason = None

            if hostile_guard["enabled"]:
                current_day = current_time.astimezone(NY_TZ).date()
                if hostile_day_date != current_day:
                    reset_hostile_day(current_day)
                if hostile_day_active:
                    await asyncio.sleep(0.5)

            # ==========================================
            # HEARTBEAT & POSITION SYNC NOW HANDLED BY INDEPENDENT ASYNC TASKS
            # See: heartbeat_task() and position_sync_task() launched at startup
            # These tasks run independently and cannot be blocked by strategy logic
            # ==========================================
            now_ts = time.time()

            # === HTF FVG MEMORY NOW UPDATED BY BACKGROUND TASK ===
            # See: htf_structure_task() launched at startup
            # This task runs independently and cannot be blocked by strategy logic

            sync_tracked_trades_with_broker_state(current_time, current_price)

            # Only process signals on NEW bars
            # is_new_bar already computed above
            if is_new_bar:
                backfill_recent_closed_trades_from_projectx(current_time)
                bar_count += 1
                logging.info(f"Bar: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ET | Price: {current_price:.2f}")
                last_processed_bar = current_time

                trend_day_series = None
                # === TREND DAY DETECTOR (Tier 1/2 + sticky direction) ===
                if TREND_DAY_ENABLED:
                    try:
                        if trend_day_lockout_until and current_time <= trend_day_lockout_until:
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
                            raise StopIteration
                        trend_day_series = compute_trend_day_series(new_df)
                        trend_session = "NY" if base_session in ("NY_AM", "NY_PM") else base_session
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

                        day_key = trend_day_series["day_index"].iloc[-1]
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

                        bar_close = float(currbar["close"])
                        bar_high = float(currbar["high"])
                        bar_low = float(currbar["low"])

                        ema50_val = trend_day_series["ema50"].iloc[-1]
                        atr_expansion = trend_day_series["atr_expansion"].iloc[-1]
                        vwap_val = trend_day_series["vwap"].iloc[-1]
                        vwap_sigma = trend_day_series["vwap_sigma_dist"].iloc[-1]
                        session_open = trend_day_series["session_open"].iloc[-1]
                        prior_session_low = trend_day_series["prior_session_low"].iloc[-1]
                        prior_session_high = trend_day_series["prior_session_high"].iloc[-1]
                        trend_up_alt = bool(trend_day_series["trend_up_alt"].iloc[-1])
                        trend_down_alt = bool(trend_day_series["trend_down_alt"].iloc[-1])
                        adx_strong_up = bool(trend_day_series["adx_strong_up"].iloc[-1])
                        adx_strong_down = bool(trend_day_series["adx_strong_down"].iloc[-1])

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

                        asia_structure_only = TREND_DAY_ASIA_STRUCTURE_ONLY and base_session == "ASIA"

                        if pd.notna(vwap_sigma):
                            displaced_down = vwap_sigma <= -VWAP_SIGMA_T1
                            displaced_up = vwap_sigma >= VWAP_SIGMA_T1
                            no_reclaim_down_t1 = bool(trend_day_series["no_reclaim_down_t1"].iloc[-1])
                            no_reclaim_up_t1 = bool(trend_day_series["no_reclaim_up_t1"].iloc[-1])
                            no_reclaim_down_t2 = bool(trend_day_series["no_reclaim_down_t2"].iloc[-1])
                            no_reclaim_up_t2 = bool(trend_day_series["no_reclaim_up_t2"].iloc[-1])
                            reclaim_down = bool(trend_day_series["reclaim_down"].iloc[-1])
                            reclaim_up = bool(trend_day_series["reclaim_up"].iloc[-1])

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
                            prev_close = float(new_df.iloc[-2]["close"]) if len(new_df) > 1 else bar_close
                            if pd.notna(prior_session_low):
                                confirm_down = confirm_down or (
                                    bar_close < prior_session_low and prev_close < prior_session_low
                                )
                            if pd.notna(prior_session_high):
                                confirm_up = confirm_up or (
                                    bar_close > prior_session_high and prev_close > prior_session_high
                                )

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
                            tier1_down_until = current_time + datetime.timedelta(
                                minutes=TREND_DAY_STICKY_RECLAIM_BARS
                            )
                        if tier1_up:
                            tier1_up_until = current_time + datetime.timedelta(
                                minutes=TREND_DAY_STICKY_RECLAIM_BARS
                            )

                        tier1_down_active = bool(tier1_down_until and current_time <= tier1_down_until)
                        tier1_up_active = bool(tier1_up_until and current_time <= tier1_up_until)
                        if tier1_down_active or tier1_up_active:
                            tier1_seen = True
                        allow_alt = tier1_seen or (
                            pd.notna(vwap_sigma) and abs(vwap_sigma) >= ALT_PRE_TIER1_VWAP_SIGMA
                        )

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
                        if TREND_DAY_ASIA_DISABLE_TIER2 and base_session == "ASIA":
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
                                logging.warning(
                                    "[TrendDay] Deactivating tier/alt after "
                                    f"{loss_count} consecutive {trend_day_dir.upper()} losses"
                                )
                                dlb_blocked_until = (
                                    directional_loss_blocker.long_blocked_until
                                    if trend_day_dir == "up"
                                    else directional_loss_blocker.short_blocked_until
                                )
                                dlb_blocked = bool(dlb_blocked_until and current_time < dlb_blocked_until)
                                dlb_losses = (
                                    directional_loss_blocker.long_consecutive_losses
                                    if trend_day_dir == "up"
                                    else directional_loss_blocker.short_consecutive_losses
                                )
                                dlb_until_str = dlb_blocked_until.strftime("%H:%M:%S") if dlb_blocked_until else "n/a"
                                logging.warning(
                                    "[TrendDay] DLB status after deactivation: "
                                    f"blocked={dlb_blocked} losses={dlb_losses} until={dlb_until_str}"
                                )
                                _deactivate_trend_day(
                                    f"DLB {loss_count} {trend_day_dir.upper()} losses",
                                    current_time,
                                )
                        if trend_day_tier > 0 and trend_day_dir:
                            # Disable TrendDay if a large impulse prints in the OPPOSITE direction.
                            # Uses the same impulse criteria + wick override as ImpulseFilter.
                            opp_impulse = False
                            opp_reason = ""
                            is_impulse, impulse_threshold, _ = impulse_filter.get_impulse_stats()
                            if is_impulse:
                                upper_wick = impulse_filter.last_candle_high - max(
                                    impulse_filter.last_candle_open, impulse_filter.last_candle_close
                                )
                                lower_wick = min(
                                    impulse_filter.last_candle_open, impulse_filter.last_candle_close
                                ) - impulse_filter.last_candle_low
                                wick_threshold = impulse_filter.last_candle_body * impulse_filter.wick_ratio_threshold
                                if trend_day_dir == "up" and impulse_filter.last_candle_dir == "RED":
                                    if lower_wick <= wick_threshold:
                                        opp_impulse = True
                                        opp_reason = (
                                            f"Red impulse {impulse_filter.last_candle_body:.2f} > "
                                            f"{impulse_threshold:.2f}"
                                        )
                                elif trend_day_dir == "down" and impulse_filter.last_candle_dir == "GREEN":
                                    if upper_wick <= wick_threshold:
                                        opp_impulse = True
                                        opp_reason = (
                                            f"Green impulse {impulse_filter.last_candle_body:.2f} > "
                                            f"{impulse_threshold:.2f}"
                                        )
                            if opp_impulse:
                                logging.warning(
                                    "[TrendDay] Deactivating tier/alt due to opposite impulse: "
                                    f"{opp_reason}"
                                )
                                _deactivate_trend_day(f"opposite impulse ({opp_reason})", current_time)
                                if TREND_DAY_REACTIVATION_COOLDOWN_MINUTES > 0:
                                    cooldown_until = current_time + datetime.timedelta(
                                        minutes=TREND_DAY_REACTIVATION_COOLDOWN_MINUTES
                                    )
                                    if (
                                        trend_day_lockout_until is None
                                        or cooldown_until > trend_day_lockout_until
                                    ):
                                        trend_day_lockout_until = cooldown_until
                                    logging.info(
                                        "[TrendDay] reactivation cooldown until %s after opposite impulse",
                                        trend_day_lockout_until.strftime("%Y-%m-%d %H:%M"),
                                    )
                        if trend_day_tier > 0 and trend_day_dir:
                            sma9_slope_series = trend_day_series.get("sma9_slope")
                            sma9_series = trend_day_series.get("sma9")
                            atr_contract_ok = pd.notna(atr_expansion) and atr_expansion <= TREND_DAY_ATR_CONTRACTION
                            reversal_ok = False
                            if (
                                sma9_slope_series is not None
                                and len(sma9_slope_series) >= TREND_DAY_SMA9_REVERSAL_BARS
                            ):
                                recent_slopes = sma9_slope_series.iloc[-TREND_DAY_SMA9_REVERSAL_BARS:]
                                recent_closes = new_df["close"].iloc[-TREND_DAY_SMA9_REVERSAL_BARS:]
                                recent_sma9 = None
                                if sma9_series is not None and len(sma9_series) >= TREND_DAY_SMA9_REVERSAL_BARS:
                                    recent_sma9 = sma9_series.iloc[-TREND_DAY_SMA9_REVERSAL_BARS:]
                                if not recent_slopes.isna().any():
                                    if trend_day_dir == "up":
                                        slope_ok = (recent_slopes <= -TREND_DAY_SMA9_MIN_SLOPE).all()
                                        price_ok = (
                                            recent_sma9 is not None
                                            and (recent_closes < recent_sma9).all()
                                        )
                                        reversal_ok = slope_ok and price_ok
                                    elif trend_day_dir == "down":
                                        slope_ok = (recent_slopes >= TREND_DAY_SMA9_MIN_SLOPE).all()
                                        price_ok = (
                                            recent_sma9 is not None
                                            and (recent_closes > recent_sma9).all()
                                        )
                                        reversal_ok = slope_ok and price_ok
                            if reversal_ok and atr_contract_ok:
                                avg_slope = float(recent_slopes.mean())
                                logging.warning(
                                    "[TrendDay] Deactivating tier/alt due to SMA9 reversal + ATR contraction: "
                                    f"slope_avg={avg_slope:.3f} atr_exp={atr_expansion:.3f}"
                                )
                                _deactivate_trend_day(
                                    f"SMA9 reversal + ATR contraction (slope_avg={avg_slope:.3f})",
                                    current_time,
                                )

                        # === TrendDay "rotation/mean-reversion" deactivation ===
                        if trend_day_tier > 0 and trend_day_dir:
                            vwap_series = trend_day_series.get("vwap")
                            sigma_series = trend_day_series.get("vwap_sigma_dist")
                            close_series = new_df.get("close")

                            # Gate A: VWAP reclaim + sigma decay hold
                            if (
                                vwap_series is not None
                                and sigma_series is not None
                                and close_series is not None
                                and len(close_series) >= TREND_DAY_DEACTIVATE_RECLAIM_WINDOW
                                and len(vwap_series) >= TREND_DAY_DEACTIVATE_RECLAIM_WINDOW
                                and len(sigma_series) >= TREND_DAY_DEACTIVATE_SIGMA_BARS
                            ):
                                recent_close = close_series.iloc[-TREND_DAY_DEACTIVATE_RECLAIM_WINDOW:]
                                recent_vwap = vwap_series.iloc[-TREND_DAY_DEACTIVATE_RECLAIM_WINDOW:]
                                if trend_day_dir == "up":
                                    reclaim_count = int(
                                        (recent_close.to_numpy() < recent_vwap.to_numpy()).sum()
                                    )
                                else:
                                    reclaim_count = int(
                                        (recent_close.to_numpy() > recent_vwap.to_numpy()).sum()
                                    )

                                sigma_recent = sigma_series.iloc[-TREND_DAY_DEACTIVATE_SIGMA_BARS:]
                                sigma_ok = bool(
                                    (sigma_recent.abs() < TREND_DAY_DEACTIVATE_SIGMA_THRESHOLD).all()
                                )

                                if reclaim_count >= TREND_DAY_DEACTIVATE_RECLAIM_COUNT and sigma_ok:
                                    _deactivate_trend_day(
                                        "VWAP reclaim + sigma decay "
                                        f"(reclaim={reclaim_count}/{TREND_DAY_DEACTIVATE_RECLAIM_WINDOW}, "
                                        f"sigma<{TREND_DAY_DEACTIVATE_SIGMA_THRESHOLD})",
                                        current_time,
                                    )

                            # Gate B: sigma decay kill switch
                            if trend_day_tier > 0 and trend_day_dir and sigma_series is not None:
                                if len(sigma_series) >= TREND_DAY_DEACTIVATE_SIGMA_DECAY_BARS:
                                    sigma_decay_ok = bool(
                                        (
                                            sigma_series.iloc[-TREND_DAY_DEACTIVATE_SIGMA_DECAY_BARS:]
                                            .abs()
                                            < TREND_DAY_DEACTIVATE_SIGMA_DECAY_THRESHOLD
                                        ).all()
                                    )
                                    current_sigma = sigma_series.iloc[-1]
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
                            atr_dbg = float(atr_expansion) if pd.notna(atr_expansion) else float("nan")
                            sigma_dbg = float(vwap_sigma) if pd.notna(vwap_sigma) else float("nan")
                            trend_dir_label = str(trend_day_dir).lower()
                            trend_icon = "🟢" if trend_dir_label == "up" else "🛑"
                            trend_msg = (
                                f"{trend_icon} TrendDay Tier {trend_day_tier} {trend_day_dir} activated "
                                f"@ {current_time.strftime('%Y-%m-%d %H:%M')} "
                                f"| atr_exp={atr_dbg:.2f} sigma={sigma_dbg:.2f} "
                                f"trend_alt_up={int(trend_up_alt)} trend_alt_down={int(trend_down_alt)} "
                                f"adx_up={int(adx_strong_up)} adx_down={int(adx_strong_down)}"
                            )
                            if trend_dir_label == "up":
                                logging.info(trend_msg)
                            else:
                                logging.warning(trend_msg)
                        last_trend_day_tier = trend_day_tier
                        last_trend_day_dir = trend_day_dir
                    except StopIteration:
                        pass
                    except Exception as e:
                        logging.error(f"TrendDay calculation failed: {e}")
                        trend_day_tier = 0
                        trend_day_dir = None
                        last_trend_day_tier = 0
                        last_trend_day_dir = None
                        sticky_trend_dir = None
                        sticky_reclaim_count = 0
                        sticky_opposite_count = 0
                        tier1_seen = False
                        trend_day_max_sigma = 0.0
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

                if (
                    last_ui_trend_day_tier is None
                    or trend_day_tier != last_ui_trend_day_tier
                    or trend_day_dir != last_ui_trend_day_dir
                ):
                    status_dir = trend_day_dir if trend_day_dir else "none"
                    logging.info(
                        f"[TrendDay] status tier={trend_day_tier} dir={status_dir} "
                        f"@ {current_time.strftime('%Y-%m-%d %H:%M')}"
                    )
                    last_ui_trend_day_tier = trend_day_tier
                    last_ui_trend_day_dir = trend_day_dir

                # === RISK TELEMETRY (PERIODIC HEARTBEAT) ===
                # Calculate current risk metrics
                current_dd = abs(min(circuit_breaker.daily_pnl, 0))  # Current daily loss (positive value)
                max_dd = circuit_breaker.max_daily_loss
                daily_pnl = circuit_breaker.daily_pnl

                # Log telemetry every 15 minutes OR if drawdown > 50%
                minute = current_time.minute
                usage_pct = (current_dd / max_dd * 100) if max_dd > 0 else 0
                if minute % 15 == 0 or usage_pct > 50:
                    event_logger.log_risk_telemetry(
                        current_loss=current_dd,
                        limit=max_dd,
                        daily_pnl=daily_pnl
                    )

                # === LIVE TRADE MANAGEMENT / EARLY EXIT CHECK ===
                tracked_before_management = tracked_live_trades()
                if tracked_before_management:
                    trade_state_changed = False
                    bar_high = _coerce_float(currbar.get('high'), current_price)
                    bar_low = _coerce_float(currbar.get('low'), current_price)
                    # Pivot-trail: detect confirmed swing pivots once per bar
                    # (US session only — same gate as bank fill)
                    _pt_us_session = (
                        current_time.astimezone(NY_TZ).hour
                        in _BANK_FILL_US_SESSION_HOURS_ET
                    )
                    _pt_pivot_high = (
                        _detect_pivot_high(new_df, _PIVOT_TRAIL_LOOKBACK)
                        if _pt_us_session else None
                    )
                    _pt_pivot_low = (
                        _detect_pivot_low(new_df, _PIVOT_TRAIL_LOOKBACK)
                        if _pt_us_session else None
                    )
                    _pt_pivot_bar_index = (
                        int(bar_count - (_PIVOT_TRAIL_LOOKBACK // 2))
                        if _pt_us_session
                        else None
                    )
                    for trade in list(tracked_before_management):
                        management_result = _process_live_trade_management_bar(
                            client,
                            trade,
                            current_time=current_time,
                            market_price=current_price,
                            bar_high=bar_high,
                            bar_low=bar_low,
                            bar_index=bar_count,
                        )
                        if isinstance(management_result, dict) and management_result.get("status") == "closed":
                            finalize_live_trade_close(
                                trade,
                                management_result.get("close_metrics"),
                                current_time,
                                log_prefix=str(
                                    management_result.get("log_prefix")
                                    or "Trade closed"
                                ),
                            )
                            remove_tracked_live_trade(trade)
                            trade_state_changed = True
                            continue  # skip pivot trail on already-closed trade
                        # --- Pivot-trail SL ratchet (DE3 / AetherFlow / RegimeAdaptive only) ---
                        if _pt_us_session and str(trade.get("strategy", "")) in _PIVOT_TRAIL_ELIGIBLE_STRATEGIES:
                            _pt_new_sl = _live_pivot_trail_candidate(
                                trade,
                                pivot_high=_pt_pivot_high,
                                pivot_low=_pt_pivot_low,
                                pivot_bar_index=_pt_pivot_bar_index,
                            )
                            if _pt_new_sl is not None:
                                # --- SHADOW/LIVE: ML pivot-trail gate ---
                                # Shadow mode (default): score and log both decisions;
                                #   always trail (rule behavior).
                                # Live mode (JULIE_ML_PIVOT_TRAIL_ACTIVE=1): only trail
                                #   when ML says the pivot will hold.
                                _pt_ml_skip = False
                                try:
                                    import ml_overlay_shadow as _mls_pv
                                    if _mls_pv._PIVOT_PAYLOAD is not None and len(new_df) >= 21:
                                        _pt_type = "HIGH" if _pt_pivot_high is not None else "LOW"
                                        _pt_ppx = float(_pt_pivot_high if _pt_pivot_high is not None else _pt_pivot_low)
                                        _pt_anchor_c = (
                                            math.floor(_pt_ppx / _BANK_FILL_STEP) * _BANK_FILL_STEP
                                            if _pt_type == "HIGH"
                                            else math.ceil(_pt_ppx / _BANK_FILL_STEP) * _BANK_FILL_STEP
                                        )
                                        _pt_anchor_b = (
                                            _pt_anchor_c - _BANK_FILL_STEP
                                            if _pt_type == "HIGH"
                                            else _pt_anchor_c + _BANK_FILL_STEP
                                        )
                                        # Pivot bar = center of 5-bar lookback window
                                        _pt_center_i = max(0, len(new_df) - 1 - (_PIVOT_TRAIL_LOOKBACK // 2))
                                        _pt_bar = new_df.iloc[_pt_center_i]
                                        _pt_closes = new_df["close"].astype(float).values
                                        _pt_highs = new_df["high"].astype(float).values
                                        _pt_lows = new_df["low"].astype(float).values
                                        _pt_i = len(new_df) - 1
                                        _pt_atr14 = (
                                            float(sum(max(_pt_highs[k] - _pt_lows[k],
                                                          abs(_pt_highs[k] - _pt_closes[k - 1]),
                                                          abs(_pt_lows[k] - _pt_closes[k - 1]))
                                                      for k in range(_pt_i - 13, _pt_i + 1)) / 14)
                                            if _pt_i >= 14 else 0.0
                                        )
                                        _pt_r30 = float(_pt_highs[max(0, _pt_i - 29):_pt_i + 1].max()
                                                        - _pt_lows[max(0, _pt_i - 29):_pt_i + 1].min()) if _pt_i >= 5 else 0.0
                                        _pt_t20 = (
                                            (_pt_closes[_pt_i] - _pt_closes[_pt_i - 20]) / max(1.0, _pt_closes[_pt_i - 20]) * 100.0
                                            if _pt_i >= 20 else 0.0
                                        )
                                        _pt_hi20 = float(_pt_highs[_pt_i - 19:_pt_i + 1].max()) if _pt_i >= 20 else float(_pt_highs[_pt_i])
                                        _pt_lo20 = float(_pt_lows[_pt_i - 19:_pt_i + 1].min()) if _pt_i >= 20 else float(_pt_lows[_pt_i])
                                        _pt_dh = (_pt_closes[_pt_i] - _pt_hi20) / max(1.0, _pt_closes[_pt_i]) * 100.0
                                        _pt_dl = (_pt_closes[_pt_i] - _pt_lo20) / max(1.0, _pt_closes[_pt_i]) * 100.0
                                        _pt_v5 = (_pt_closes[_pt_i] - _pt_closes[_pt_i - 5]) / 5.0 if _pt_i >= 5 else 0.0
                                        _pt_v20 = (_pt_closes[_pt_i] - _pt_closes[_pt_i - 20]) / 20.0 if _pt_i >= 20 else 0.0
                                        # Tape classifier (same as trainer)
                                        _pt_span = (_pt_r30 / max(1.0, _pt_closes[_pt_i])) * 100.0
                                        if _pt_span < 0.15:
                                            _pt_tape = "chop"
                                        elif _pt_t20 >= 0.15:
                                            _pt_tape = "uptrend"
                                        elif _pt_t20 <= -0.15:
                                            _pt_tape = "downtrend"
                                        else:
                                            _pt_tape = "chop"
                                        _pt_et_hour = current_time.astimezone(NY_TZ).hour
                                        if 18 <= _pt_et_hour or _pt_et_hour < 3: _pt_sess = "ASIA"
                                        elif 3 <= _pt_et_hour < 7: _pt_sess = "LONDON"
                                        elif 7 <= _pt_et_hour < 9: _pt_sess = "NY_PRE"
                                        elif 9 <= _pt_et_hour < 12: _pt_sess = "NY_AM"
                                        elif 12 <= _pt_et_hour < 16: _pt_sess = "NY_PM"
                                        else: _pt_sess = "POST"
                                        _pt_score = _mls_pv.score_pivot_trail(
                                            _pt_type, _pt_ppx,
                                            float(_pt_bar["open"]), float(_pt_bar["high"]),
                                            float(_pt_bar["low"]), float(_pt_bar["close"]),
                                            atr14_pts=_pt_atr14,
                                            range_30bar_pts=_pt_r30,
                                            trend_20bar_pct=_pt_t20,
                                            dist_to_20bar_hi_pct=_pt_dh,
                                            dist_to_20bar_lo_pct=_pt_dl,
                                            vel_5bar_pts_per_min=_pt_v5,
                                            vel_20bar_pts_per_min=_pt_v20,
                                            anchor_c=_pt_anchor_c,
                                            anchor_b=_pt_anchor_b,
                                            session=_pt_sess,
                                            tape=_pt_tape,
                                            et_hour=_pt_et_hour,
                                        )
                                        if _pt_score is not None:
                                            _pt_p_hold, _pt_ml_ratchet = _pt_score
                                            logging.info(
                                                "[SHADOW_PIVOT] type=%s px=%.2f sl=%.2f rule=RATCHET "
                                                "ml_p_hold=%.3f ml=%s",
                                                _pt_type, _pt_ppx, _pt_new_sl,
                                                _pt_p_hold,
                                                "RATCHET" if _pt_ml_ratchet else "SKIP",
                                            )
                                            if _mls_pv.is_pivot_trail_live_active() and not _pt_ml_ratchet:
                                                _pt_ml_skip = True
                                except Exception as _pt_shadow_exc:
                                    logging.debug("ml pivot shadow err: %s", _pt_shadow_exc)
                                if _pt_ml_skip:
                                    continue  # ML says pivot won't hold — skip ratchet this bar
                                _pt_result = _apply_pivot_trail_sl(
                                    client,
                                    trade,
                                    _pt_new_sl,
                                    current_time=current_time,
                                    market_price=current_price,
                                    bar_high=bar_high,
                                    bar_low=bar_low,
                                    bar_index=bar_count,
                                )
                                if (
                                    isinstance(_pt_result, dict)
                                    and _pt_result.get("status") == "closed"
                                ):
                                    finalize_live_trade_close(
                                        trade,
                                        _pt_result.get("close_metrics"),
                                        current_time,
                                        log_prefix="[PivotTrail] closed — market through new SL",
                                    )
                                    remove_tracked_live_trade(trade)
                                    trade_state_changed = True
                                    continue
                        # --- SHADOW: PPO trade-management policy (Path 3) ---
                        # Per-bar: build the 172-dim observation the PPO agent
                        # was trained on, query the policy, log the action.
                        # ACTIVE mode would need per-action execution paths
                        # (MOVE_SL_TO_BE, TIGHTEN_SL_*, TAKE_PARTIAL_*, REVERSE)
                        # wired to the broker — for now shadow-only so we
                        # observe agreement/disagreement with rule behavior
                        # before committing to live steering.
                        try:
                            import ml_overlay_shadow as _mls_rl
                            _rl_ready = _mls_rl.init_rl_management()
                            if _rl_ready and len(new_df) >= 30 and str(
                                trade.get("strategy", "")
                            ) in _PIVOT_TRAIL_ELIGIBLE_STRATEGIES:
                                # Per-trade running state (init once on first bar)
                                trade.setdefault("_rl_bars_held", 0)
                                trade.setdefault("_rl_mfe_pts", 0.0)
                                trade.setdefault("_rl_mae_pts", 0.0)
                                trade.setdefault("_rl_peak_pnl_pts", 0.0)
                                trade.setdefault("_rl_trough_pnl_pts", 0.0)
                                trade["_rl_bars_held"] += 1
                                _rl_entry = float(_coerce_float(trade.get("entry_price"), 0.0))
                                _rl_side = str(trade.get("side", "")).upper()
                                if _rl_side in ("LONG", "SHORT") and _rl_entry > 0:
                                    _rl_cur = float(current_price)
                                    _rl_pts = _rl_cur - _rl_entry if _rl_side == "LONG" else _rl_entry - _rl_cur
                                    # MFE/MAE tracking off bar extremes
                                    _rl_high = float(bar_high)
                                    _rl_low = float(bar_low)
                                    if _rl_side == "LONG":
                                        _rl_fav = _rl_high - _rl_entry
                                        _rl_adv = _rl_entry - _rl_low
                                    else:
                                        _rl_fav = _rl_entry - _rl_low
                                        _rl_adv = _rl_high - _rl_entry
                                    if _rl_fav > trade["_rl_mfe_pts"]:
                                        trade["_rl_mfe_pts"] = _rl_fav
                                    if _rl_adv > trade["_rl_mae_pts"]:
                                        trade["_rl_mae_pts"] = _rl_adv
                                    if _rl_pts > trade["_rl_peak_pnl_pts"]:
                                        trade["_rl_peak_pnl_pts"] = _rl_pts
                                    if _rl_pts < trade["_rl_trough_pnl_pts"]:
                                        trade["_rl_trough_pnl_pts"] = _rl_pts
                                    # Regime + session labels (reuse earlier
                                    # SHADOW_KALSHI computation if in scope;
                                    # else quick local compute)
                                    _rl_et_hour = int(current_time.astimezone(NY_TZ).hour)
                                    if 18 <= _rl_et_hour or _rl_et_hour < 3:
                                        _rl_sess = "ASIA"
                                    elif 3 <= _rl_et_hour < 7:
                                        _rl_sess = "LONDON"
                                    elif 7 <= _rl_et_hour < 9:
                                        _rl_sess = "NY_PRE"
                                    elif 9 <= _rl_et_hour < 12:
                                        _rl_sess = "NY_AM"
                                    elif 12 <= _rl_et_hour < 16:
                                        _rl_sess = "NY_PM"
                                    else:
                                        _rl_sess = "POST"
                                    try:
                                        from regime_classifier import current_regime as _rl_regime
                                        _rl_reg = _rl_regime()
                                    except Exception:
                                        _rl_reg = "neutral"
                                    # SL/TP current + original
                                    _rl_cur_sl = float(_coerce_float(trade.get("sl_price"), _rl_entry))
                                    _rl_cur_tp = float(_coerce_float(trade.get("tp_price"), _rl_entry))
                                    _rl_orig_sl = float(_coerce_float(
                                        trade.get("original_sl_price", trade.get("sl_price")), _rl_cur_sl
                                    ))
                                    _rl_orig_tp = float(_coerce_float(
                                        trade.get("original_tp_price", trade.get("tp_price")), _rl_cur_tp
                                    ))
                                    # ATR14 from new_df
                                    try:
                                        _rl_c = new_df["close"].astype(float).values
                                        _rl_h = new_df["high"].astype(float).values
                                        _rl_l = new_df["low"].astype(float).values
                                        _rl_idx = len(new_df) - 1
                                        _rl_trs = [
                                            max(_rl_h[k] - _rl_l[k],
                                                abs(_rl_h[k] - _rl_c[k - 1]),
                                                abs(_rl_l[k] - _rl_c[k - 1]))
                                            for k in range(max(1, _rl_idx - 13), _rl_idx + 1)
                                        ]
                                        _rl_atr = float(sum(_rl_trs) / max(1, len(_rl_trs))) if _rl_trs else 1.0
                                    except Exception:
                                        _rl_atr = 1.0
                                    # v2 inputs — harmless when canonical is v1,
                                    # consumed when v2 policy is loaded (obs_dim=212).
                                    _rl_entry_time = trade.get("entry_time")
                                    _rl_entry_bar_idx = None
                                    if _rl_entry_time is not None:
                                        try:
                                            _rl_entry_bar_idx = int(new_df.index.get_loc(_rl_entry_time))
                                        except Exception:
                                            try:
                                                import numpy as _np_local
                                                _arr = new_df.index.to_numpy()
                                                _mask = _arr >= _np_local.datetime64(_rl_entry_time)
                                                _rl_entry_bar_idx = int(_np_local.argmax(_mask)) if _mask.any() else None
                                            except Exception:
                                                _rl_entry_bar_idx = None
                                    _rl_result = _mls_rl.score_rl_management(
                                        bars_df=new_df,
                                        entry_price=_rl_entry,
                                        side=_rl_side,
                                        atr14=max(0.25, _rl_atr),
                                        bars_held=int(trade["_rl_bars_held"]),
                                        mfe_pts=float(trade["_rl_mfe_pts"]),
                                        mae_pts=float(trade["_rl_mae_pts"]),
                                        current_sl_price=_rl_cur_sl,
                                        current_tp_price=_rl_cur_tp,
                                        original_sl_price=_rl_orig_sl,
                                        original_tp_price=_rl_orig_tp,
                                        running_peak_pnl_pts=float(trade["_rl_peak_pnl_pts"]),
                                        running_trough_pnl_pts=float(trade["_rl_trough_pnl_pts"]),
                                        regime_label=_rl_reg,
                                        session_label=_rl_sess,
                                        kalshi_probs=None,
                                        trade_id=id(trade),
                                        entry_time=_rl_entry_time,
                                        entry_bar_idx=_rl_entry_bar_idx,
                                        vix_override_df=master_vix_df,
                                        mnq_override_df=master_mnq_df,
                                    )
                                    if _rl_result is not None:
                                        _rl_action, _rl_name = _rl_result
                                        logging.info(
                                            "[SHADOW_RL] strat=%s side=%s bar=%d pnl_pts=%+.2f "
                                            "mfe=%.2f mae=%.2f regime=%s action=%s",
                                            trade.get("strategy", "?"), _rl_side,
                                            int(trade["_rl_bars_held"]), _rl_pts,
                                            float(trade["_rl_mfe_pts"]), float(trade["_rl_mae_pts"]),
                                            _rl_reg, _rl_name,
                                        )
                                        # Active mode: execute SAFE actions
                                        # (SL moves only). TAKE_PARTIAL_* /
                                        # REVERSE return "deferred" and are
                                        # only logged. Operators can observe
                                        # the deferred rate and decide when
                                        # to wire the missing paths.
                                        if _mls_rl.is_rl_management_live_active():
                                            try:
                                                _rl_apply = _apply_rl_management_action(
                                                    client, trade, int(_rl_action), _rl_name,
                                                    current_time=current_time,
                                                    market_price=current_price,
                                                    bar_high=bar_high, bar_low=bar_low,
                                                    bar_index=bar_count,
                                                )
                                                if isinstance(_rl_apply, dict):
                                                    logging.info(
                                                        "[RL_LIVE] status=%s action=%s %s",
                                                        _rl_apply.get("status"), _rl_name,
                                                        _rl_apply.get("reason", ""),
                                                    )
                                            except Exception as _rl_apply_exc:
                                                logging.debug(
                                                    "rl active-mode apply err: %s",
                                                    _rl_apply_exc,
                                                )
                        except Exception as _rl_exc:
                            logging.debug("rl management shadow err: %s", _rl_exc)
                        _kalshi_tp_result = _apply_kalshi_tp_trail(
                            client,
                            trade,
                            current_time=current_time,
                            market_price=current_price,
                            bar_high=bar_high,
                            bar_low=bar_low,
                            bar_index=bar_count,
                        )
                        if (
                            isinstance(_kalshi_tp_result, dict)
                            and _kalshi_tp_result.get("status") == "closed"
                        ):
                            finalize_live_trade_close(
                                trade,
                                _kalshi_tp_result.get("close_metrics"),
                                current_time,
                                log_prefix="[KalshiTrail] closed — market through TP-zone stop",
                            )
                            remove_tracked_live_trade(trade)
                            trade_state_changed = True
                        elif (
                            isinstance(_kalshi_tp_result, dict)
                            and _kalshi_tp_result.get("status") == "updated"
                        ):
                            trade_state_changed = True

                    for trade in list(tracked_live_trades()):
                        trade['bars_held'] += 1
                        strategy_name = trade['strategy']
                        early_exit_config = _resolve_live_early_exit_config(trade)

                        if trade['side'] == 'LONG':
                            is_green = current_price > trade['entry_price']
                        else:
                            is_green = current_price < trade['entry_price']

                        was_green = trade.get('was_green')
                        if was_green is not None and is_green != was_green:
                            trade['profit_crosses'] = trade.get('profit_crosses', 0) + 1
                        trade['was_green'] = is_green

                        # Kalshi hour-turn sentiment exit (independent of early exit config)
                        kalshi_exit = _check_kalshi_sentiment_exit(trade, current_price, current_time)
                        if kalshi_exit:
                            exit_reason = kalshi_exit
                        elif early_exit_config.get('enabled', False):
                            exit_reason = _evaluate_live_early_exit_reason(
                                trade,
                                current_price,
                                early_exit_config,
                            )
                        else:
                            exit_reason = None

                        if not exit_reason:
                            continue

                        trade["de3_early_exit_trigger_reason"] = str(exit_reason)
                        logging.info(f"⏰ EARLY EXIT: {strategy_name} - {exit_reason}")

                        event_logger.log_early_exit(
                            reason=exit_reason,
                            bars_held=trade['bars_held'],
                            current_price=current_price,
                            entry_price=trade['entry_price']
                        )
                        close_metrics = None
                        position = client.get_position()
                        if position.get("stale"):
                            logging.warning("Position stale on early-exit close; skipping close.")
                            continue

                        if len(tracked_live_trades()) > 1 and position['side'] is not None:
                            if not await client.async_close_trade_leg(trade):
                                logging.warning("Trade-leg early exit failed; keeping tracked trade intact.")
                                continue
                            close_order_details = getattr(client, "_last_close_order_details", {}) or {}
                            close_metrics = _reconcile_live_trade_close(
                                client,
                                trade,
                                current_time,
                                fallback_exit_price=current_price,
                                close_order_id=close_order_details.get("order_id"),
                            )
                            if not isinstance(close_metrics, dict):
                                close_metrics = _calculate_live_trade_close_metrics_from_price(
                                    trade,
                                    _coerce_float(close_order_details.get("exit_price"), current_price),
                                    source="partial_market_order",
                                    exit_time=current_time,
                                    order_id=_coerce_int(close_order_details.get("order_id"), None),
                                )
                        elif position['side'] is not None:
                            if not client.close_position(position):
                                logging.warning("Early-exit close request failed; keeping active trade intact.")
                                continue
                            close_order_details = getattr(client, "_last_close_order_details", {}) or {}
                            close_metrics = _reconcile_live_trade_close(
                                client,
                                trade,
                                current_time,
                                fallback_exit_price=current_price,
                                close_order_id=close_order_details.get("order_id"),
                            )
                        else:
                            close_metrics = _reconcile_live_trade_close(
                                client,
                                trade,
                                current_time,
                                fallback_exit_price=current_price,
                            )
                        if not isinstance(close_metrics, dict):
                            close_order_details = getattr(client, "_last_close_order_details", {}) or {}
                            close_metrics = _calculate_live_trade_close_metrics_from_price(
                                trade,
                                _coerce_float(close_order_details.get("exit_price"), current_price),
                                source=str(close_order_details.get("method") or "early_exit_fallback"),
                                exit_time=current_time,
                                order_id=_coerce_int(close_order_details.get("order_id"), None),
                            )
                        finalize_live_trade_close(
                            trade,
                            close_metrics,
                            current_time,
                            log_prefix="📊 Early exit closed",
                        )
                        remove_tracked_live_trade(trade)
                        trade_state_changed = True

                    if trade_state_changed:
                        persist_runtime_state(current_time, reason="trade_management_close")

                if await process_truth_social_emergency_exit(current_time, current_price):
                    await asyncio.sleep(2.0)
                    continue

                # === STRATEGY EXECUTION ===
                if news_blocked:
                    logging.info("📰 NEWS BLACKOUT: Skipping trade execution (data continues)")
                    await asyncio.sleep(10)
                    continue

                # === LEVEL-FILL PENDING QUEUE ===
                # Fire any deferred signals whose target level has been touched,
                # or abort them if price ran away. Runs before new signal eval.
                if is_new_bar and level_fill_optimizer is not None and pending_level_fills:
                    _lf_active = (
                        active_trade is not None or bool(tracked_live_trades())
                    )
                    if _lf_active:
                        level_fill_optimizer.clear_all("position already active")
                        pending_level_fills.clear()
                    else:
                        _lf_bar = {
                            "open":  float(currbar["open"]),
                            "high":  float(currbar["high"]),
                            "low":   float(currbar["low"]),
                            "close": float(currbar["close"]),
                        }
                        for _lf_uid in list(pending_level_fills.keys()):
                            _lf_result = level_fill_optimizer.check_pending(_lf_uid, _lf_bar)
                            if _lf_result["fire"]:
                                _lf_entry = pending_level_fills.get(_lf_uid)
                                if _lf_entry is None:
                                    pending_level_fills.pop(_lf_uid, None)
                                    continue
                                _lf_allowed, _lf_market_dist = _level_fill_live_execution_allowed(
                                    _lf_entry,
                                    current_price,
                                )
                                if not _lf_allowed:
                                    _lf_target = _coerce_float(_lf_entry.get("target_price"), math.nan)
                                    pending_level_fills.pop(_lf_uid, None)
                                    logging.info(
                                        "📌 LevelFill ABORT: touched previously but market is now %.2fpts from target %.2f",
                                        float(_lf_market_dist or 0.0),
                                        float(_lf_target),
                                    )
                                    continue
                                pending_level_fills.pop(_lf_uid, None)
                                _lf_sig = _lf_entry["signal"]
                                _lf_sig["level_fill_trigger"] = _lf_result["reason"]
                                if not _apply_kalshi_trade_overlay_to_signal(
                                    _lf_sig,
                                    current_price,
                                    new_df,
                                    price_action_profile=kalshi_price_action_profile,
                                ):
                                    logging.info(
                                        "📌 LevelFill ABORT: Kalshi overlay blocked %s %s",
                                        _lf_sig.get("strategy", "?"),
                                        _lf_sig.get("side", "?"),
                                    )
                                    continue
                                logging.info(
                                    "📌 LevelFill FIRE: %s %s — %s",
                                    _lf_sig.get("strategy", "?"),
                                    _lf_sig.get("side", "?"),
                                    _lf_result["reason"],
                                )
                                if not _resolve_pct_overlay_snapshot(_lf_sig):
                                    pending_level_fills.pop(_lf_uid, None)
                                    logging.info(
                                        "📌 LevelFill ABORT: pct overlay snapshot vetoed %s %s",
                                        _lf_sig.get("strategy", "?"),
                                        _lf_sig.get("side", "?"),
                                    )
                                    continue
                                if circuit_breaker.is_tripped:
                                    logging.info("🚫 CB-entry-gate: skipping level-fill entry (%s %s)",
                                                 _lf_sig.get("strategy","?"), _lf_sig.get("side","?"))
                                    continue
                                from regime_classifier import should_veto_entry as _regime_veto
                                _rv, _rr = _regime_veto()
                                if _rv:
                                    logging.info("🚫 regime-veto: skipping level-fill entry (%s %s) — %s",
                                                 _lf_sig.get("strategy","?"), _lf_sig.get("side","?"), _rr)
                                    continue
                                _lfgv, _lfgr = _lfg_should_veto_entry(_lf_sig, market_time_et)
                                if _lfgv:
                                    logging.info("🚫 lfg-veto: skipping level-fill entry (%s %s) — %s",
                                                 _lf_sig.get("strategy","?"), _lf_sig.get("side","?"), _lfgr)
                                    continue
                                _lf_resp = await client.async_place_order(_lf_sig, current_price)
                                if _lf_resp is not None:
                                    _lf_od = getattr(client, "_last_order_details", None) or {}
                                    _lf_ep = _lf_od.get("entry_price", current_price)
                                    _lf_sig["tp_dist"]    = _lf_od.get("tp_points", _lf_sig.get("tp_dist", 0.0))
                                    _lf_sig["sl_dist"]    = _lf_od.get("sl_points", _lf_sig.get("sl_dist", 0.0))
                                    _lf_sig["size"]       = _lf_od.get("size", _lf_sig.get("size", 1))
                                    _lf_sig["entry_price"] = _lf_ep
                                    add_tracked_live_trade(
                                        _build_live_active_trade(
                                            _lf_sig, _lf_od, current_price, current_time,
                                            bar_count, market_df=new_df,
                                            stop_order_id=_lf_od.get(
                                                "stop_order_id",
                                                getattr(client, "_active_stop_order_id", None),
                                            ),
                                        )
                                    )
                                    persist_runtime_state(current_time, reason="level_fill_execution")
                                break  # one fill per bar
                            elif _lf_result["abort"]:
                                pending_level_fills.pop(_lf_uid, None)
                                logging.info(
                                    "📌 LevelFill ABORT: %s",
                                    _lf_result["reason"],
                                )

                strategy_results = {'checked': [], 'rejected': [], 'executed': None}
                ui_slot_limit = max(1, int(CONFIG.get("UI_STRATEGY_SLOT_LIMIT", 8) or 8))

                def add_strategy_slot(slot_key: str, raw_strategy: Optional[str], signal_payload: Optional[dict], fallback: Optional[str] = None) -> None:
                    if slot_key not in strategy_results:
                        return
                    label = format_ui_strategy_slot(raw_strategy, signal_payload, fallback=fallback)
                    if not label:
                        return
                    bucket = strategy_results.get(slot_key)
                    if isinstance(bucket, list):
                        if label in bucket:
                            return
                        if len(bucket) >= ui_slot_limit:
                            return
                        bucket.append(label)
                    elif slot_key == "executed":
                        strategy_results[slot_key] = label

                regime_meta = None
                if regime_manifold_engine is not None:
                    try:
                        regime_meta = regime_manifold_engine.update(
                            new_df,
                            ts=current_time,
                            session=base_session,
                        )
                        if isinstance(regime_meta, dict):
                            manifold_label = str(regime_meta.get("regime", "UNKNOWN"))
                            if manifold_label != last_regime_manifold_label:
                                logging.info(
                                    "RegimeManifold: %s | R=%.3f stress=%.3f risk=%.2f side=%s no_trade=%s",
                                    manifold_label,
                                    float(regime_meta.get("R", 0.0) or 0.0),
                                    float(regime_meta.get("stress", 0.0) or 0.0),
                                    float(regime_meta.get("risk_mult", 1.0) or 1.0),
                                    int(regime_meta.get("side_bias", 0) or 0),
                                    bool(regime_meta.get("no_trade", False)),
                                )
                                last_regime_manifold_label = manifold_label
                    except Exception as exc:
                        logging.warning("RegimeManifold update failed: %s", exc)
                        regime_meta = None

                def apply_regime_meta(signal_payload: Optional[dict], fallback_name: Optional[str]) -> tuple[bool, str]:
                    if (
                        regime_meta is None
                        or apply_meta_policy_fn is None
                        or not isinstance(signal_payload, dict)
                    ):
                        return True, ""
                    allowed, reason, updates = apply_meta_policy_fn(
                        signal_payload,
                        regime_meta,
                        fallback_name=fallback_name,
                        default_size=5,
                        enforce_side_bias=manifold_enforce_side_bias,
                    )
                    if regime_manifold_mode == "shadow":
                        if updates:
                            updates = dict(updates)
                            updates.pop("size", None)
                            signal_payload.update(updates)
                        return True, reason
                    if updates:
                        signal_payload.update(updates)
                    return allowed, reason

                # === PENDING IMPULSE-RESCUE CONFIRMATION ===
                if pending_impulse_rescues:
                    executed_rescue = False
                    while pending_impulse_rescues and current_time > pending_impulse_rescues[0]["signal_time"]:
                        pending_impulse_rescue = pending_impulse_rescues.pop(0)
                        pending_signal = pending_impulse_rescue["signal"]
                        signal_price = pending_impulse_rescue["signal_price"]
                        signal_close = pending_impulse_rescue["signal_close"]

                        if pending_signal.get("side") == "SHORT":
                            retest_ok = currbar["high"] >= signal_price
                            close_ok = currbar["close"] <= signal_close
                        else:
                            retest_ok = currbar["low"] <= signal_price
                            close_ok = currbar["close"] >= signal_close

                        if retest_ok or close_ok:
                            pending_signal.setdefault("entry_mode", "rescued")
                            logging.info(
                                "RESCUE CONFIRMED: impulse-rescue passed retest or close confirmation"
                            )
                            rm_ok, rm_reason = apply_regime_meta(
                                pending_signal,
                                pending_signal.get("strategy"),
                            )
                            if (not rm_ok) and regime_manifold_mode == "enforce":
                                event_logger.log_filter_check(
                                    "RegimeManifold",
                                    pending_signal.get("side", "ALL"),
                                    False,
                                    rm_reason,
                                    strategy=pending_signal.get("strategy", "PendingRescue"),
                                    metrics={
                                        "regime": str(regime_meta.get("regime")) if isinstance(regime_meta, dict) else "UNKNOWN",
                                        "R": round(float(regime_meta.get("R", 0.0) or 0.0), 4) if isinstance(regime_meta, dict) else 0.0,
                                    },
                                )
                                continue

                            pending_signal["size"] = _apply_live_execution_size(
                                pending_signal,
                                5,
                                live_drawdown_state,
                                tracked_live_trades(),
                            )
                            if pending_signal["size"] <= 0:
                                logging.info("Kalshi HARD VETO — rescued trade blocked (size=0): %s", pending_signal.get("strategy", "PendingRescue"))
                                continue
                            if _same_side_active_trade(active_trade, pending_signal):
                                reset_opposite_reversal_state("same-side rescued signal")
                                logging.info(
                                    "Ignoring same-side rescued signal while %s position is already active: %s",
                                    active_trade.get("side"),
                                    pending_signal.get("strategy", "PendingRescue"),
                                )
                                executed_rescue = True
                                pending_impulse_rescues.clear()
                                break

                            current_trades = tracked_live_trades()
                            old_trades = [dict(trade) for trade in current_trades]

                            block_reason = _live_entry_window_block_reason(current_time)
                            if block_reason:
                                _log_live_entry_window_block(
                                    pending_signal,
                                    pending_signal.get("strategy", "PendingRescue"),
                                    current_time,
                                )
                                continue

                            if old_trades and not opposite_reversal_matches_active_trade_family(
                                pending_signal,
                                current_trades,
                            ):
                                reset_opposite_reversal_state(
                                    "opposite rescue active-trade family mismatch"
                                )
                                log_opposite_reversal_active_trade_family_block(
                                    pending_signal,
                                    current_trades,
                                    prefix="Holding position",
                                )
                                executed_rescue = True
                                pending_impulse_rescues.clear()
                                break

                            if old_trades and same_strategy_opposite_reversal_blocked(
                                pending_signal,
                                current_trades,
                            ):
                                reset_opposite_reversal_state(
                                    "same-strategy opposite rescue reversal blocked"
                                )
                                log_same_strategy_opposite_reversal_block(
                                    pending_signal,
                                    current_trades,
                                    prefix="Holding position",
                                )
                                executed_rescue = True
                                pending_impulse_rescues.clear()
                                break

                            reverse_state_count = 0
                            if old_trades:
                                reverse_confirmed, reverse_state_count = note_opposite_reversal_signal(
                                    pending_signal,
                                    bar_count,
                                )
                                if not reverse_confirmed:
                                    remaining = max(
                                        0,
                                        opposite_reversal_required - reverse_state_count,
                                    )
                                    logging.info(
                                        "Holding position: opposite rescue confirmation %s/%s for %s "
                                        "(need %s more within %s bars)",
                                        reverse_state_count,
                                        opposite_reversal_required,
                                        pending_signal.get("side"),
                                        remaining,
                                        opposite_reversal_window_bars,
                                    )
                                    executed_rescue = True
                                    pending_impulse_rescues.clear()
                                    break

                            if not _apply_kalshi_trade_overlay_to_signal(
                                pending_signal,
                                current_price,
                                new_df,
                                price_action_profile=kalshi_price_action_profile,
                            ):
                                pending_impulse_rescues.clear()
                                break

                            if not _resolve_pct_overlay_snapshot(pending_signal):
                                pending_impulse_rescues.clear()
                                break

                            if circuit_breaker.is_tripped:
                                logging.info("🚫 CB-entry-gate: skipping impulse-rescue reversal (%s %s)", pending_signal.get("strategy","?"), pending_signal.get("side","?"))
                                pending_impulse_rescues.clear()
                                break
                            from regime_classifier import should_veto_entry as _regime_veto
                            _rv, _rr = _regime_veto()
                            if _rv:
                                logging.info("🚫 regime-veto: skipping impulse-rescue reversal (%s %s) — %s", pending_signal.get("strategy","?"), pending_signal.get("side","?"), _rr)
                                pending_impulse_rescues.clear()
                                break
                            _lfgv, _lfgr = _lfg_should_veto_entry(pending_signal, market_time_et)
                            if _lfgv:
                                logging.info("🚫 lfg-veto: skipping impulse-rescue reversal (%s %s) — %s", pending_signal.get("strategy","?"), pending_signal.get("side","?"), _lfgr)
                                pending_impulse_rescues.clear()
                                break
                            success, reverse_state_count = await client.async_close_and_reverse(
                                pending_signal,
                                current_price,
                                reverse_state_count,
                            )
                            if success or reverse_state_count == 0:
                                reset_opposite_reversal_state("impulse rescue execution path completed")
                            if success:
                                if old_trades and any(old_trade.get("side") != pending_signal.get("side") for old_trade in old_trades):
                                    close_order_details = getattr(client, "_last_close_order_details", None) or {}
                                    shared_exit_price = _coerce_float(close_order_details.get("exit_price"), current_price)
                                    for old_trade in old_trades:
                                        if len(old_trades) == 1:
                                            close_metrics = _reconcile_live_trade_close(
                                                client,
                                                old_trade,
                                                current_time,
                                                fallback_exit_price=current_price,
                                                close_order_id=close_order_details.get("order_id"),
                                            )
                                        else:
                                            close_metrics = _calculate_live_trade_close_metrics_from_price(
                                                old_trade,
                                                shared_exit_price,
                                                source="shared_reverse_close",
                                                exit_time=current_time,
                                                order_id=_coerce_int(close_order_details.get("order_id"), None),
                                            )
                                        finalize_live_trade_close(
                                            old_trade,
                                            close_metrics,
                                            current_time,
                                            log_prefix="Trade closed (reverse)",
                                        )
                                    active_trade = None
                                    parallel_active_trades = []
                                order_details = getattr(client, "_last_order_details", None) or {}
                                entry_price = order_details.get("entry_price", current_price)
                                tp_dist = order_details.get("tp_points")
                                if tp_dist is None:
                                    tp_dist = pending_signal.get('tp_dist')
                                sl_dist = order_details.get("sl_points")
                                if sl_dist is None:
                                    sl_dist = pending_signal.get('sl_dist')
                                if tp_dist is None or sl_dist is None:
                                    logging.error("Order details missing sl/tp after execution; using 0.0 for tracking")
                                    tp_dist = tp_dist or 0.0
                                    sl_dist = sl_dist or 0.0
                                size = order_details.get("size", pending_signal.get('size', 5))
                                pending_signal['tp_dist'] = tp_dist
                                pending_signal['sl_dist'] = sl_dist
                                pending_signal['size'] = size
                                pending_signal['entry_price'] = entry_price
                                active_trade = None
                                parallel_active_trades = []
                                add_tracked_live_trade(
                                    _build_live_active_trade(
                                        pending_signal,
                                        order_details,
                                        current_price,
                                        current_time,
                                        bar_count,
                                        market_df=new_df,
                                        stop_order_id=order_details.get(
                                            "stop_order_id",
                                            getattr(client, "_active_stop_order_id", None),
                                        ),
                                    )
                                )
                                persist_runtime_state(current_time, reason="impulse_rescue_entry")
                                log_trade_factor_snapshot(
                                    source="impulse_rescue_confirmed",
                                    signal_payload=pending_signal,
                                    order_details=order_details,
                                    entry_price=float(entry_price),
                                    market_df=new_df,
                                    event_time=current_time,
                                    market_price=float(current_price),
                                    base_session_name=base_session,
                                    current_session_name=current_session_name,
                                    vol_regime_name=str(pending_signal.get("vol_regime", "")),
                                    trend_tier=trend_day_tier,
                                    trend_dir=trend_day_dir,
                                    strategy_eval=strategy_results,
                                    regime_snapshot=regime_meta,
                                    allowed_chop_bias=allowed_chop_side,
                                    asia_viable_flag=asia_viable,
                                )
                            executed_rescue = True
                            pending_impulse_rescues.clear()
                            break
                        else:
                            logging.info(
                                "RESCUE FAILED: impulse-rescue confirmation not met "
                                f"(retest={retest_ok}, close_confirm={close_ok})"
                            )
                    if executed_rescue:
                        signal_executed = True
                        continue

# ── Adaptive bank fill: check pending reversal signal ──
                _bank_fill_triggered = False
                if _pending_mlphysics_bank_fill is not None:
                    _pending_mlphysics_bank_fill_bars -= 1
                    _bfill_side = str(_pending_mlphysics_bank_fill.get("side", "")).upper()
                    _bfill_target = float(_pending_mlphysics_bank_fill.get("bank_target", 0.0))
                    _bfill_bar_low = float(new_df.iloc[-1]["low"])
                    _bfill_bar_high = float(new_df.iloc[-1]["high"])
                    _bfill_touched = (
                        (_bfill_side == "LONG" and _bfill_bar_low <= _bfill_target)
                        or (_bfill_side == "SHORT" and _bfill_bar_high >= _bfill_target)
                    )
                    if _bfill_touched:
                        logging.info(
                            "🏦 Bank fill triggered: %s target=%.2f (bar L=%.2f H=%.2f, bars_left=%d)",
                            _bfill_side, _bfill_target, _bfill_bar_low, _bfill_bar_high,
                            _pending_mlphysics_bank_fill_bars,
                        )
                        _bank_fill_triggered = True
                    elif _pending_mlphysics_bank_fill_bars <= 0:
                        logging.info(
                            "🏦 Bank fill expired (no touch): %s target=%.2f after %d bars",
                            _bfill_side, _bfill_target, _BANK_FILL_WINDOW_BARS,
                        )
                        _pending_mlphysics_bank_fill = None

# Run ML Analysis
                ml_signal = None
                if ml_strategy is not None and ml_strategy.model_loaded:
                    try:
                        ml_signal = ml_strategy.on_bar(new_df)
                        if ml_signal:
                            add_strategy_slot(
                                "checked",
                                ml_signal.get("strategy", "MLPhysics"),
                                ml_signal,
                                fallback="MLPhysics",
                            )
                    except Exception as e:
                        logging.error(f"ML Strategy Error: {e}")
                if ml_signal:
                    disabled_sessions = set(CONFIG.get("ML_PHYSICS_LIVE_DISABLED_SESSIONS", []))
                    if base_session in disabled_sessions:
                        logging.info(f"⚠️ MLPhysics disabled in live for session {base_session}")
                        ml_signal = None

                # =================================================================
                # 🎯 HARVEST ALL SIGNALS (Solves "Ghost Signal" Problem)
                # =================================================================
                # Collect ALL potential signals from ALL strategies BEFORE filtering
                # This enables opportunity cost analysis - see what was blocked
                candidate_signals = []  # List of (priority, strategy_instance, signal_dict, strat_name)

                # ── Adaptive bank fill: inject triggered pending signal ──
                _staged_bank_fill_candidate = None  # reset each bar
                if _bank_fill_triggered and _pending_mlphysics_bank_fill is not None:
                    _bfill_inj = dict(_pending_mlphysics_bank_fill)
                    _pending_mlphysics_bank_fill = None
                    _pending_mlphysics_bank_fill_bars = 0
                    if _bfill_inj.get("_bank_fill_source") == "harvest":
                        # DE3 / AetherFlow / RegimeAdaptive — stage for candidate_signals
                        _staged_bank_fill_candidate = (
                            int(_bfill_inj.get("_bank_fill_priority", 1)),
                            _bfill_inj.get("_bank_fill_strat_ref"),
                            _bfill_inj,
                            str(_bfill_inj.get("_bank_fill_strat_name",
                                               _bfill_inj.get("strategy", ""))),
                        )
                        logging.info(
                            "🏦 Bank fill triggered → staging candidate: %s %s @ %.2f",
                            _bfill_inj.get("_bank_fill_strat_name"),
                            _bfill_inj.get("side"),
                            _bfill_inj.get("bank_target", 0.0),
                        )
                    else:
                        # MLPhysics — inject as ml_signal (existing path)
                        ml_signal = _bfill_inj
                        logging.info(
                            "🏦 Injecting bank fill signal: %s @ bank_target=%.2f",
                            ml_signal.get("side"), ml_signal.get("bank_target", 0.0),
                        )
                # ── Adaptive bank fill: park weak US-session reversal signals ──
                elif ml_signal and _pending_mlphysics_bank_fill is None:
                    try:
                        if (
                            _mlphysics_is_us_macro_bar(current_time)
                            and not _mlphysics_bank_break_is_strong(new_df)
                            and _mlphysics_is_reversal(ml_signal, new_df)
                        ):
                            _bfill_park_target = _mlphysics_bank_fill_target(
                                str(ml_signal.get("side", "")), float(current_price)
                            )
                            _pending_mlphysics_bank_fill = dict(ml_signal)
                            _pending_mlphysics_bank_fill["bank_target"] = _bfill_park_target
                            _pending_mlphysics_bank_fill_bars = _BANK_FILL_WINDOW_BARS
                            logging.info(
                                "🏦 Parking weak reversal for bank fill: %s target=%.2f "
                                "(window=%d bars, macro_bar=%s)",
                                ml_signal.get("side"), _bfill_park_target,
                                _BANK_FILL_WINDOW_BARS, current_time,
                            )
                            ml_signal = None
                    except Exception as _bfe:
                        logging.debug("Bank fill park check error: %s", _bfe)

                # Inject a staged bank fill candidate from a prior bar's trigger
                if _staged_bank_fill_candidate is not None:
                    _signal_birth_hook(_staged_bank_fill_candidate[2])
                    _attach_pct_overlay_snapshot(_staged_bank_fill_candidate[2])
                    candidate_signals.append(_staged_bank_fill_candidate)
                    logging.info(
                        "🏦 Staged bank fill injected into candidates: %s %s",
                        _staged_bank_fill_candidate[3],
                        _staged_bank_fill_candidate[2].get("side"),
                    )
                    _staged_bank_fill_candidate = None

                # -----------------------------------------------------------------
                # HARVEST PHASE 1: FAST STRATEGIES (Priority 1)
                # -----------------------------------------------------------------
                current_fast = fast_strategies.copy()
                random.shuffle(current_fast)

                for strat in current_fast:
                    strat_name = strat.__class__.__name__
                    try:
                        # Handle specific arguments for VIX vs others
                        if strat_name == "VIXReversionStrategy":
                            if master_vix_df.empty or new_df.empty:
                                continue
                            vix_df = master_vix_df
                            vix_ts = vix_df.index[-1]
                            mes_ts = new_df.index[-1]
                            if vix_ts.tzinfo is None:
                                vix_ts = vix_ts.replace(tzinfo=dt_timezone.utc)
                            else:
                                vix_ts = vix_ts.astimezone(dt_timezone.utc)
                            if mes_ts.tzinfo is None:
                                mes_ts = mes_ts.replace(tzinfo=NY_TZ)
                            mes_ts = mes_ts.astimezone(dt_timezone.utc)
                            if vix_ts > mes_ts and len(vix_df) > 1:
                                vix_df = vix_df.iloc[:-1]
                                vix_ts = vix_df.index[-1]
                                if vix_ts.tzinfo is None:
                                    vix_ts = vix_ts.replace(tzinfo=dt_timezone.utc)
                                else:
                                    vix_ts = vix_ts.astimezone(dt_timezone.utc)
                            if abs((vix_ts - mes_ts).total_seconds()) > 950:
                                logging.info("VIX stale vs MES; skipping VIXReversionStrategy")
                                continue
                            signal = strat.on_bar(new_df, vix_df)
                        else:
                            signal = strat.on_bar(new_df)

                        if signal:
                            if (not disable_strategy_filters) and hostile_day_active and is_hostile_disabled_strategy(signal, strat_name):
                                event_logger.log_filter_check(
                                    "HostileDay",
                                    signal['side'],
                                    False,
                                    "hostile day active",
                                    strategy=signal.get('strategy', strat_name)
                                )
                                logging.info(f"🛑 HOSTILE DAY: Skipping {strat_name}")
                                continue
                            # ==========================================
                            # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                            # ==========================================
                            sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                            tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                            sltp = _validate_signal_sltp(signal, strat_name)
                            if sltp is None:
                                continue
                            old_sl, old_tp = sltp

                            signal['sl_dist'] = old_sl * sl_mult
                            signal['tp_dist'] = old_tp * tp_mult

                            if sl_mult != 1.0 or tp_mult != 1.0:
                                logging.info(f"🧠 GEMINI OPTIMIZED: {strat_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")

                            # Enforce HTF range fade directional restriction
                            # if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                            #    logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                            #    continue

                            strat_label = str(signal.get("strategy", strat_name) or strat_name)
                            disabled_filter = execution_disabled_filter(strat_label, base_session)
                            if disabled_filter:
                                maybe_log_disabled_strategy(
                                    disabled_filter,
                                    signal.get("side", "ALL"),
                                    strat_label,
                                )
                                continue

                            # ── Adaptive bank fill: park weak US-session reversal ──
                            if _pending_mlphysics_bank_fill is None and strat_name in _BANK_FILL_ELIGIBLE_STRATEGIES:
                                try:
                                    _bfill_hour = current_time.astimezone(NY_TZ).hour
                                    if (
                                        _bfill_hour in _BANK_FILL_US_SESSION_HOURS_ET
                                        and not _mlphysics_bank_break_is_strong(new_df)
                                        and _mlphysics_is_reversal(signal, new_df)
                                    ):
                                        _bfill_t = _mlphysics_bank_fill_target(
                                            str(signal.get("side", "")), float(current_price)
                                        )
                                        _pending_mlphysics_bank_fill = dict(signal)
                                        _pending_mlphysics_bank_fill["bank_target"] = _bfill_t
                                        _pending_mlphysics_bank_fill["_bank_fill_source"] = "harvest"
                                        _pending_mlphysics_bank_fill["_bank_fill_strat_ref"] = strat
                                        _pending_mlphysics_bank_fill["_bank_fill_priority"] = 1
                                        _pending_mlphysics_bank_fill["_bank_fill_strat_name"] = strat_name
                                        _pending_mlphysics_bank_fill_bars = _BANK_FILL_WINDOW_BARS
                                        logging.info(
                                            "🏦 Parking %s weak reversal: %s target=%.2f "
                                            "(window=%d bars)",
                                            strat_name, signal.get("side"), _bfill_t,
                                            _BANK_FILL_WINDOW_BARS,
                                        )
                                        continue  # skip appending to candidates this bar
                                except Exception as _bfe3:
                                    logging.debug("Bank fill park check (fast) error: %s", _bfe3)

                            # Add to candidate list (Priority 1 = FAST)
                            _signal_birth_hook(signal)
                            _attach_pct_overlay_snapshot(signal)
                            add_strategy_slot(
                                "checked",
                                signal.get("strategy", strat_name),
                                signal,
                                fallback=strat_name,
                            )
                            candidate_signals.append((1, strat, signal, strat_name))

                            # Log as candidate
                            log_strategy, log_sub = get_log_strategy_info(
                                signal.get('strategy', strat_name),
                                signal
                            )
                            log_info = {
                                "status": "CANDIDATE",
                                "priority": "FAST",
                            }
                            if log_sub:
                                log_info["sub_strategy"] = log_sub
                            for extra_key in (
                                "combo_key",
                                "rule_id",
                                "entry_mode",
                                "vol_regime",
                                "early_exit_enabled",
                                "gate_prob",
                                "gate_threshold",
                            ):
                                extra_value = signal.get(extra_key)
                                if extra_value is not None:
                                    log_info[extra_key] = extra_value
                            event_logger.log_strategy_signal(
                                strategy_name=log_strategy,
                                side=signal['side'],
                                tp_dist=signal.get('tp_dist'),
                                sl_dist=signal.get('sl_dist'),
                                price=current_price,
                                additional_info=log_info
                            )
                            logging.info(
                                f"📊 CANDIDATE (FAST): {strat_name} {signal['side']} @ {current_price:.2f}"
                            )

                    except Exception as e:
                        logging.exception("Error in %s", strat_name)

                # -----------------------------------------------------------------
                # HARVEST PHASE 2: STANDARD STRATEGIES (Priority 2)
                # -----------------------------------------------------------------
                # Shuffle standard strategies
                current_standard = standard_strategies.copy()
                random.shuffle(current_standard)

                for strat in current_standard:
                    strat_name = strat.__class__.__name__
                    signal = None
                    priority = 1 if filterless_only_mode else 2

                    # (SMT needs master_mnq_df, ML needs ml_signal, others use new_df)
                    if strat_name == "MLPhysicsStrategy":
                        signal = ml_signal
                        if signal and not filterless_only_mode:
                            boost_cfg = CONFIG.get("ML_PHYSICS_PRIORITY_BOOST", {}) or {}
                            if boost_cfg.get("enabled", False):
                                try:
                                    min_conf = float(boost_cfg.get("min_confidence", 0.0))
                                except Exception:
                                    min_conf = 0.0
                                sessions = boost_cfg.get("sessions") or []
                                if not sessions or base_session in sessions:
                                    try:
                                        ml_conf = float(signal.get("ml_confidence", 0.0))
                                    except Exception:
                                        ml_conf = 0.0
                                    if ml_conf >= min_conf:
                                        try:
                                            priority = int(boost_cfg.get("boost_priority", 1))
                                        except Exception:
                                            priority = 1
                    elif strat_name == "SMTStrategy":
                        try:
                            mnq_df = master_mnq_df
                            if not mnq_df.empty and not new_df.empty:
                                mnq_last = mnq_df.index[-1]
                                if mnq_last.tzinfo is None:
                                    mnq_last = mnq_last.replace(tzinfo=NY_TZ)
                                mes_last = new_df.index[-1]
                                if mes_last.tzinfo is None:
                                    mes_last = mes_last.replace(tzinfo=NY_TZ)
                                if mnq_last > mes_last and len(mnq_df) > 1:
                                    mnq_df = mnq_df.iloc[:-1]
                            signal = strat.on_bar(new_df, mnq_df)
                        except Exception as e:
                            logging.exception("Error in %s", strat_name)
                    else:
                        try:
                            signal = strat.on_bar(new_df)
                        except Exception as e:
                            logging.exception("Error in %s", strat_name)

                    if not signal and strat_name == "AetherFlowStrategy":
                        consume_runtime_event = getattr(strat, "consume_pending_runtime_event", None)
                        if callable(consume_runtime_event):
                            runtime_event = consume_runtime_event()
                            if isinstance(runtime_event, dict):
                                blocked_info = {
                                    "status": runtime_event.get("status", "BLOCKED"),
                                    "decision": runtime_event.get("decision", "blocked"),
                                    "reason": runtime_event.get("reason"),
                                }
                                for extra_key, event_key in (
                                    ("setup_family", "combo_key"),
                                    ("regime", "vol_regime"),
                                    ("confidence", "gate_prob"),
                                    ("threshold", "gate_threshold"),
                                    ("session_id", "session_id"),
                                ):
                                    extra_value = runtime_event.get(extra_key)
                                    if extra_value is not None:
                                        blocked_info[event_key] = extra_value
                                event_logger.log_strategy_signal(
                                    strategy_name="AetherFlow",
                                    side=runtime_event.get("side", "NONE"),
                                    tp_dist=0.0,
                                    sl_dist=0.0,
                                    price=current_price,
                                    additional_info=blocked_info,
                                )

                    if signal:
                        if (
                            signal.get("strategy", strat_name) == "AuctionReversion"
                            and base_session == "NY_PM"
                            and trend_day_tier > 0
                        ):
                            event_logger.log_filter_check(
                                "AuctionReversionTrendDayTier",
                                signal["side"],
                                False,
                                "NY_PM trend day tier > 1",
                                strategy=signal.get("strategy", strat_name),
                            )
                            logging.info(
                                "🛑 AuctionReversion blocked: NY_PM with TrendDayTier %s",
                                trend_day_tier,
                            )
                            continue
                        # ==========================================
                        # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                        # ==========================================
                        sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                        tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                        sltp = _validate_signal_sltp(signal, strat_name)
                        if sltp is None:
                            continue
                        old_sl, old_tp = sltp

                        signal['sl_dist'] = old_sl * sl_mult
                        signal['tp_dist'] = old_tp * tp_mult

                        if sl_mult != 1.0 or tp_mult != 1.0:
                            logging.info(f"🧠 GEMINI OPTIMIZED: {strat_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")

                        # Enforce HTF range fade directional restriction
                        # if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                        #    logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                        #    continue

                        strat_label = str(signal.get("strategy", strat_name) or strat_name)
                        disabled_filter = execution_disabled_filter(strat_label, base_session)
                        if disabled_filter:
                            maybe_log_disabled_strategy(
                                disabled_filter,
                                signal.get("side", "ALL"),
                                strat_label,
                            )
                            continue

                        # Add to candidate list (Priority 2 in normal mode; unified to 1 in filterless mode)
                        if (not disable_strategy_filters) and hostile_day_active and is_hostile_disabled_strategy(signal, strat_name):
                            event_logger.log_filter_check(
                                "HostileDay",
                                signal['side'],
                                False,
                                "hostile day active",
                                strategy=signal.get('strategy', strat_name)
                            )
                            logging.info(f"🛑 HOSTILE DAY: Skipping {strat_name}")
                            continue

                        # ── Adaptive bank fill: park weak US-session reversal ──
                        if _pending_mlphysics_bank_fill is None and strat_name in _BANK_FILL_ELIGIBLE_STRATEGIES:
                            try:
                                _bfill_hour = current_time.astimezone(NY_TZ).hour
                                if (
                                    _bfill_hour in _BANK_FILL_US_SESSION_HOURS_ET
                                    and not _mlphysics_bank_break_is_strong(new_df)
                                    and _mlphysics_is_reversal(signal, new_df)
                                ):
                                    _bfill_t = _mlphysics_bank_fill_target(
                                        str(signal.get("side", "")), float(current_price)
                                    )
                                    _pending_mlphysics_bank_fill = dict(signal)
                                    _pending_mlphysics_bank_fill["bank_target"] = _bfill_t
                                    _pending_mlphysics_bank_fill["_bank_fill_source"] = "harvest"
                                    _pending_mlphysics_bank_fill["_bank_fill_strat_ref"] = strat
                                    _pending_mlphysics_bank_fill["_bank_fill_priority"] = priority
                                    _pending_mlphysics_bank_fill["_bank_fill_strat_name"] = strat_name
                                    _pending_mlphysics_bank_fill_bars = _BANK_FILL_WINDOW_BARS
                                    logging.info(
                                        "🏦 Parking %s weak reversal: %s target=%.2f "
                                        "(window=%d bars)",
                                        strat_name, signal.get("side"), _bfill_t,
                                        _BANK_FILL_WINDOW_BARS,
                                    )
                                    continue  # skip appending to candidates this bar
                            except Exception as _bfe4:
                                logging.debug("Bank fill park check (standard) error: %s", _bfe4)

                        add_strategy_slot(
                            "checked",
                            signal.get("strategy", strat_name),
                            signal,
                            fallback=strat_name,
                        )
                        _signal_birth_hook(signal)
                        _attach_pct_overlay_snapshot(signal)
                        candidate_signals.append((priority, strat, signal, strat_name))

                        # Log as candidate
                        log_strategy, log_sub = get_log_strategy_info(
                            signal.get('strategy', strat_name),
                            signal
                        )
                        priority_label = "FAST" if priority == 1 else "STANDARD"
                        log_info = {"status": "CANDIDATE", "priority": priority_label}
                        if log_sub:
                            log_info["sub_strategy"] = log_sub
                        for extra_key in (
                            "combo_key",
                            "rule_id",
                            "entry_mode",
                            "vol_regime",
                            "early_exit_enabled",
                            "gate_prob",
                            "gate_threshold",
                        ):
                            extra_value = signal.get(extra_key)
                            if extra_value is not None:
                                log_info[extra_key] = extra_value
                        if strat_name == "MLPhysicsStrategy" and priority == 1:
                            log_info["confidence_boosted"] = True
                        event_logger.log_strategy_signal(
                            strategy_name=log_strategy,
                            side=signal['side'],
                            tp_dist=signal.get('tp_dist'),
                            sl_dist=signal.get('sl_dist'),
                            price=current_price,
                            additional_info=log_info
                        )
                        logging.info(
                            f"📊 CANDIDATE ({priority_label}): {strat_name} {signal['side']} @ {current_price:.2f}"
                        )
                # -----------------------------------------------------------------
                # SELECTION PHASE: Process candidates by priority until one passes
                # -----------------------------------------------------------------
                def signal_confidence(sig):
                    return _live_signal_confidence(sig)

                vol_regime_current = None
                try:
                    vol_regime_current, _, _ = volatility_filter.get_regime(new_df)
                except Exception:
                    vol_regime_current = None

                asia_trend_bias_side = None
                if asia_calib_enabled and base_session == "ASIA":
                    trend_cfg = asia_calib_cfg.get("trend_bias", {}) or {}
                    asia_trend_bias_side = asia_trend_bias(new_df, trend_cfg)

                asia_viable = True
                asia_viable_reason = None
                if base_session == "ASIA":
                    asia_viable, asia_viable_reason = asia_viability_gate(
                        new_df,
                        ts=current_time,
                        session=base_session,
                    )

                if candidate_signals:
                    active_candidates = []
                    for priority, strat, sig, s_name in candidate_signals:
                        strat_label = str(sig.get("strategy", s_name) or s_name)
                        disabled_filter = execution_disabled_filter(strat_label, base_session)
                        if disabled_filter:
                            # Keep the signal (for diagnostics), but don't let it affect
                            # consensus or execution.
                            maybe_log_disabled_strategy(
                                disabled_filter,
                                sig.get("side", "ALL"),
                                strat_label,
                            )
                            continue
                        if disable_strategy_filters:
                            sig["gate_profile"] = None
                            active_candidates.append((priority, strat, sig, s_name))
                            continue
                        gate_ok, gate_filter, gate_reason, gate_profile = evaluate_pre_signal_gate_fn(
                            cfg=CONFIG,
                            session_name=base_session,
                            strategy_label=strat_label,
                            side=sig.get("side"),
                            asia_viable=asia_viable,
                            asia_reason=asia_viable_reason,
                            asia_trend_bias_side=asia_trend_bias_side,
                            is_choppy=bool(is_choppy),
                            chop_reason=chop_reason,
                            allowed_chop_side=allowed_chop_side,
                        )
                        sig["gate_profile"] = gate_profile
                        if not gate_ok:
                            event_logger.log_filter_check(
                                gate_filter or "PreCandidateGate",
                                sig.get("side", "ALL"),
                                False,
                                gate_reason or "Pre-candidate gate blocked",
                                strategy=sig.get("strategy", s_name),
                            )
                            continue
                        rm_ok, rm_reason = apply_regime_meta(sig, s_name)
                        if (not rm_ok) and regime_manifold_mode == "enforce":
                            event_logger.log_filter_check(
                                "RegimeManifold",
                                sig.get("side", "ALL"),
                                False,
                                rm_reason,
                                strategy=sig.get("strategy", s_name),
                                metrics={
                                    "regime": str(regime_meta.get("regime")) if isinstance(regime_meta, dict) else "UNKNOWN",
                                    "R": round(float(regime_meta.get("R", 0.0) or 0.0), 4) if isinstance(regime_meta, dict) else 0.0,
                                },
                            )
                            continue
                        active_candidates.append((priority, strat, sig, s_name))
                    candidate_signals = active_candidates

                def ml_soft_gate_eligible(sig: dict, strat_name: Optional[str] = None) -> bool:
                    if not consensus_ml_ok(sig, strat_name):
                        return False
                    if not ml_vol_regime_ok(sig, base_session, vol_regime_current, asia_viable=asia_viable):
                        return False
                    sig_copy = dict(sig)
                    fixed_ok, fixed_details = apply_fixed_sltp(
                        sig_copy,
                        new_df,
                        current_price,
                        ts=current_time,
                        session=base_session,
                        sl_dist_override=sig_copy.get("sl_dist"),
                    )
                    if not fixed_ok:
                        return False
                    if fixed_details:
                        sig_copy["sl_dist"] = fixed_details["sl_dist"]
                        sig_copy["tp_dist"] = fixed_details["tp_dist"]
                        sig_copy["vol_regime"] = fixed_details.get("vol_regime", sig_copy.get("vol_regime"))
                    is_feasible, _ = chop_analyzer.check_target_feasibility(
                        entry_price=current_price,
                        side=sig_copy.get("side"),
                        tp_distance=sig_copy.get("tp_dist"),
                        df_1m=new_df,
                    )
                    if (not is_feasible) and asia_calib_enabled and base_session == "ASIA":
                        tf_cfg = asia_calib_cfg.get("target_feasibility", {}) or {}
                        lookback = int(getattr(chop_analyzer, "LOOKBACK", 20) or 20)
                        if asia_target_feasibility_override(
                            new_df,
                            sig_copy.get("side"),
                            sig_copy.get("tp_dist"),
                            asia_trend_bias_side,
                            tf_cfg,
                            lookback,
                        ):
                            return True
                    return is_feasible

                ml_soft_cfg = CONFIG.get("ML_PHYSICS_SOFT_GATING", {}) or {}
                if (not disable_strategy_filters) and ml_soft_cfg.get("enabled", False) and candidate_signals:
                    soft_sessions = ml_soft_cfg.get("sessions") or []
                    if soft_sessions and base_session not in soft_sessions:
                        pass
                    else:
                        ml_candidates = [
                            (priority, sig, s_name)
                            for priority, _, sig, s_name in candidate_signals
                            if str(sig.get("strategy", s_name)).startswith("MLPhysics")
                        ]
                        if ml_candidates:
                            ml_priority, ml_sig, ml_name = max(
                                ml_candidates, key=lambda item: signal_confidence(item[1])
                            )
                            ml_conf = signal_confidence(ml_sig)
                            try:
                                min_conf = float(ml_soft_cfg.get("min_confidence", 0.0))
                            except Exception:
                                min_conf = 0.0
                            if ml_conf >= min_conf and ml_soft_gate_eligible(ml_sig, ml_name):
                                block_standard = bool(ml_soft_cfg.get("block_standard", True))
                                block_fast = bool(ml_soft_cfg.get("block_fast", False))
                                blocked_priorities = set()
                                if block_standard:
                                    blocked_priorities.add(2)
                                if block_fast:
                                    blocked_priorities.add(1)
                                ml_side = ml_sig.get("side")
                                if ml_side in ("LONG", "SHORT") and blocked_priorities:
                                    new_candidates = []
                                    for priority, strat, sig, s_name in candidate_signals:
                                        if str(sig.get("strategy", s_name)).startswith("MLPhysics"):
                                            new_candidates.append((priority, strat, sig, s_name))
                                            continue
                                        if sig.get("side") != ml_side and priority in blocked_priorities:
                                            event_logger.log_filter_check(
                                                "MLSoftGate",
                                                sig.get("side", "UNKNOWN"),
                                                False,
                                                f"ml_conf {ml_conf:.1%} >= {min_conf:.1%}",
                                                strategy=sig.get("strategy", s_name),
                                            )
                                            logging.info(
                                                f"🧠 ML SOFT GATE: blocked {s_name} "
                                                f"{sig.get('side')} (conf {ml_conf:.1%})"
                                            )
                                            continue
                                        new_candidates.append((priority, strat, sig, s_name))
                                    candidate_signals = new_candidates

                candidate_signals.sort(key=_live_signal_sort_key)

                asia_soft_ext_cfg = CONFIG.get("ASIA_SOFT_EXTENSION_FILTER", {}) or {}
                asia_soft_ext_enabled = bool(
                    asia_soft_ext_cfg.get("enabled", False)
                    and base_session == "ASIA"
                    and asia_viable
                    and not disable_strategy_filters
                )
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

                # Multi-strategy consensus override (vote-based)
                direction_counts = {"LONG": 0, "SHORT": 0}
                smt_side = None
                for _, _, sig, s_name in candidate_signals:
                    if (not disable_strategy_filters) and hostile_day_active and is_hostile_disabled_strategy(sig, s_name):
                        continue
                    side = sig.get("side")
                    if side in direction_counts:
                        strat_label = sig.get("strategy", s_name)
                        if (not disable_strategy_filters) and str(strat_label).startswith("MLPhysics"):
                            if not consensus_ml_ok(sig, s_name):
                                continue
                            if not ml_vol_regime_ok(sig, base_session, vol_regime_current, asia_viable=asia_viable):
                                continue
                        weight = 2 if s_name == "SMTStrategy" else 1
                        direction_counts[side] += weight
                    if s_name == "SMTStrategy":
                        smt_side = side

                consensus_side = None
                max_count = max(direction_counts.values()) if direction_counts else 0
                if max_count >= 2:
                    if direction_counts["LONG"] != direction_counts["SHORT"]:
                        consensus_side = "LONG" if direction_counts["LONG"] > direction_counts["SHORT"] else "SHORT"
                    elif smt_side:
                        consensus_side = smt_side
                        logging.info(f"🧲 SMT TIEBREAK: {smt_side} ({direction_counts['LONG']}L/{direction_counts['SHORT']}S)")

                if consensus_side:
                    logging.info(f"🧠 CONSENSUS OVERRIDE: {consensus_side} ({direction_counts['LONG']}L/{direction_counts['SHORT']}S)")

                consensus_tp_source = None
                consensus_tp_signal = None
                if consensus_side:
                    consensus_candidates = [
                        (sig, s_name) for _, _, sig, s_name in candidate_signals
                        if sig.get("side") == consensus_side
                        and (
                            disable_strategy_filters
                            or not str(sig.get("strategy", s_name)).startswith("MLPhysics")
                            or (
                                consensus_ml_ok(sig, s_name)
                                and ml_vol_regime_ok(sig, base_session, vol_regime_current, asia_viable=asia_viable)
                            )
                        )
                    ]
                    if consensus_candidates:
                        tp_values = []
                        sl_values = []
                        candidate_brackets = []
                        for sig, s_name in consensus_candidates:
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
                            if not new_df.empty:
                                feasibility_ok, _ = chop_analyzer.check_target_feasibility(
                                    entry_price=current_price,
                                    side=consensus_side,
                                    tp_distance=selected_tp,
                                    df_1m=new_df,
                                )
                            if not feasibility_ok and candidate_brackets:
                                candidate_brackets.sort(key=lambda item: item[0])
                                smallest_tp = candidate_brackets[0][0]
                                fallback = None
                                for tp_val, sl_val, s_name in candidate_brackets:
                                    ok, _ = chop_analyzer.check_target_feasibility(
                                        entry_price=current_price,
                                        side=consensus_side,
                                        tp_distance=tp_val,
                                        df_1m=new_df,
                                    )
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
                            logging.info(
                                "🧮 CONSENSUS TP PICK: "
                                f"{consensus_tp_source} TP={selected_tp:.2f} SL={selected_sl:.2f}"
                            )

                signal_executed = False
                surviving_direction_counts = {"LONG": 0, "SHORT": 0}
                for _, _, sig, _ in candidate_signals:
                    side = _normalize_live_side(sig.get("side") if isinstance(sig, dict) else None)
                    if side in surviving_direction_counts:
                        surviving_direction_counts[side] += 1
                if surviving_direction_counts["LONG"] > 0 and surviving_direction_counts["SHORT"] > 0:
                    if consensus_side in {"LONG", "SHORT"}:
                        resolved_candidates = []
                        resolution_reason = (
                            "mixed same-bar directions resolved by consensus "
                            f"{consensus_side} ({surviving_direction_counts['LONG']} LONG / "
                            f"{surviving_direction_counts['SHORT']} SHORT)"
                        )
                        for priority, strat, sig, strat_name in candidate_signals:
                            side = _normalize_live_side(sig.get("side") if isinstance(sig, dict) else None)
                            if side == consensus_side:
                                resolved_candidates.append((priority, strat, sig, strat_name))
                                continue
                            add_strategy_slot(
                                "rejected",
                                sig.get("strategy", strat_name) if isinstance(sig, dict) else strat_name,
                                sig if isinstance(sig, dict) else None,
                                fallback=strat_name,
                            )
                            event_logger.log_filter_check(
                                "DirectionalConflict",
                                side or "UNKNOWN",
                                False,
                                resolution_reason,
                                strategy=(sig.get("strategy", strat_name) if isinstance(sig, dict) else strat_name),
                            )
                        candidate_signals = resolved_candidates
                        logging.info("🧭 DIRECTIONAL CONFLICT RESOLVED: %s", resolution_reason)
                    else:
                        conflict_reason = (
                            "mixed same-bar directions after filtering "
                            f"({surviving_direction_counts['LONG']} LONG / "
                            f"{surviving_direction_counts['SHORT']} SHORT)"
                        )
                        for _, _, sig, strat_name in candidate_signals:
                            side = _normalize_live_side(sig.get("side") if isinstance(sig, dict) else None)
                            add_strategy_slot(
                                "rejected",
                                sig.get("strategy", strat_name) if isinstance(sig, dict) else strat_name,
                                sig if isinstance(sig, dict) else None,
                                fallback=strat_name,
                            )
                            event_logger.log_filter_check(
                                "DirectionalConflict",
                                side or "UNKNOWN",
                                False,
                                conflict_reason,
                                strategy=(sig.get("strategy", strat_name) if isinstance(sig, dict) else strat_name),
                            )
                        logging.info("⛔ DIRECTIONAL CONFLICT: %s; skipping bar", conflict_reason)
                        reset_opposite_reversal_state("directional_conflict")
                        signal_executed = True
                        candidate_signals = []
                for priority, strat, sig, strat_name in candidate_signals:
                    signal = sig
                    priority_label = "SENTIMENT" if priority == 0 else ("FAST" if priority == 1 else "STANDARD")
                    do_execute = False
                    signal.setdefault("strategy", strat_name)
                    trend_day_counter = False
                    if (not disable_strategy_filters) and trend_day_tier > 0 and trend_day_dir:
                        trend_day_counter = (
                            (trend_day_dir == "down" and signal["side"] == "LONG")
                            or (trend_day_dir == "up" and signal["side"] == "SHORT")
                        )
                        signal["trend_day_tier"] = trend_day_tier
                        signal["trend_day_dir"] = trend_day_dir
                    origin_strategy = signal.get("strategy", strat_name)
                    origin_sub_strategy = signal.get("sub_strategy")
                    allow_rescue = not str(origin_strategy).startswith("MLPhysics")
                    is_rescued = False
                    consensus_rescued = False
                    consensus_bypass_allowed = True
                    rescue_bypass_allowed = True
                    rescue_context = None
                    rescue_logged = False

                    if consensus_side and signal['side'] != consensus_side:
                        event_logger.log_filter_check(
                            "Consensus",
                            signal['side'],
                            False,
                            f"consensus={consensus_side}",
                            strategy=signal.get('strategy', strat_name)
                        )
                        logging.info(f"⏭️ Skipping {strat_name} {signal['side']} due to consensus {consensus_side}")
                        continue
                    if (not disable_strategy_filters) and hostile_day_active and is_hostile_disabled_strategy(signal, strat_name):
                        event_logger.log_filter_check(
                            "HostileDay",
                            signal['side'],
                            False,
                            "hostile day active",
                            strategy=signal.get('strategy', strat_name)
                        )
                        logging.info(f"🛑 HOSTILE DAY: Skipping {strat_name}")
                        continue

                    def should_log_ui(current_signal, fallback_name):
                        return True

                    def flip_confidence_decision(
                        filter_name: str,
                        reason: Optional[str],
                        current_signal: dict,
                        session_name: Optional[str],
                        history_df: pd.DataFrame,
                    ) -> tuple[str, dict]:
                        if not flip_conf_enabled or not flip_conf_data:
                            return "skip", {}
                        candidates = _flip_filter_candidates(filter_name, reason)
                        allowed_filters = flip_conf_allowed_filters or set(
                            flip_conf_data.get("allowed_filters") or []
                        )
                        if allowed_filters:
                            candidates = [c for c in candidates if c in allowed_filters]
                        if not candidates:
                            return "skip", {}

                        key_fields = flip_conf_cfg.get("key_fields") or flip_conf_data.get("key_fields") or [
                            "filter",
                            "session",
                            "side",
                        ]
                        vol_regime = None
                        if "regime" in key_fields:
                            try:
                                vol_regime, _, _ = volatility_filter.get_regime(history_df)
                            except Exception:
                                vol_regime = None

                        allowlist = flip_conf_data.get("allowlist")
                        stats = flip_conf_data.get("stats", {})
                        criteria = flip_conf_data.get("criteria", {}) or {}

                        def meets_thresholds(stat: dict) -> bool:
                            min_trades = float(flip_conf_cfg.get("min_total_trades", criteria.get("min_total_trades", 0)))
                            min_win = float(flip_conf_cfg.get("min_win_rate", criteria.get("min_win_rate", 0.0)))
                            min_avg = float(
                                flip_conf_cfg.get(
                                    "min_avg_pnl_points", criteria.get("min_avg_pnl_points", 0.0)
                                )
                            )
                            return (
                                stat.get("total_trades", 0) >= min_trades
                                and stat.get("win_rate", 0.0) >= min_win
                                and stat.get("avg_pnl_points", 0.0) >= min_avg
                            )

                        for candidate in candidates:
                            key = _flip_build_key(current_signal, candidate, session_name, key_fields, vol_regime)
                            if allowlist is not None:
                                if key in allowlist:
                                    return "allow", {"filter": candidate, "key": key, "regime": vol_regime}
                                continue
                            stat = stats.get(key)
                            if stat and meets_thresholds(stat):
                                return "allow", {"filter": candidate, "key": key, "regime": vol_regime}

                        return "deny", {"candidates": candidates}

                    def log_filter_block(filter_name, reason, side_override=None):
                        add_strategy_slot(
                            "rejected",
                            signal.get("strategy", strat_name),
                            signal,
                            fallback=strat_name,
                        )
                        if should_log_ui(signal, strat_name):
                            event_logger.log_filter_check(
                                filter_name,
                                side_override or signal['side'],
                                False,
                                reason,
                                strategy=signal.get('strategy', strat_name)
                            )

                    def is_wick_rejection_block(reason: Optional[str]) -> bool:
                        return "wick rejection" in str(reason or "").lower()

                    def should_defer_impulse_rescue(filter_name: str, reason: Optional[str]) -> bool:
                        if filter_name == "ImpulseFilter":
                            return True
                        reason_text = str(reason or "")
                        if "Tier 1-3" in reason_text:
                            return True
                        return False

                    def log_rescue_success():
                        nonlocal rescue_logged, rescue_context
                        if rescue_context and not rescue_logged:
                            event_logger.log_continuation_rescue_success(
                                rescue_context["original_strategy"],
                                rescue_context["rescue_strategy"],
                                rescue_context["bias"]
                            )
                            rescue_logged = True

                    def log_rescue_failed(reason: str):
                        nonlocal rescue_logged, rescue_context
                        if rescue_context and not rescue_logged:
                            event_logger.log_continuation_rescue_blocked(
                                rescue_context["original_strategy"],
                                rescue_context["rescue_strategy"],
                                rescue_context["bias"],
                                reason
                            )
                            rescue_logged = True

                    if consensus_side and signal['side'] == consensus_side:
                        bypassed_filters = [
                            "Rejection/Bias",
                            "ImpulseFilter",
                            "HTF_FVG",
                            "StructureBlocker",
                            "BankFilter",
                            "LegacyTrend",
                            "FilterArbitrator",
                            "ChopFilter",
                            "ExtensionFilter",
                        ]
                        rescue_side = 'SHORT' if signal['side'] == 'LONG' else 'LONG'

                        def try_consensus_rescue(filter_name: str, reason: str) -> bool:
                            nonlocal signal, is_rescued, consensus_rescued, rescue_context, consensus_bypass_allowed
                            log_filter_block(filter_name, reason)
                            if filter_name == "ChopFilter" and is_chop_hard_stop(reason):
                                logging.info(f"CONSENSUS RESCUE BLOCKED: Chop hard-stop ({reason})")
                                return False
                            if is_wick_rejection_block(reason):
                                logging.info("⛔ CONSENSUS RESCUE BLOCKED: TrendFilter wick rejection cooldown")
                                return False
                            if not allow_rescue:
                                return False
                            if trend_day_tier > 0 and trend_day_dir:
                                if (trend_day_dir == "down" and rescue_side == "LONG") or (
                                    trend_day_dir == "up" and rescue_side == "SHORT"
                                ):
                                    log_filter_block(
                                        f"TrendDayTier{trend_day_tier}",
                                        "Rescue side counter-trend",
                                        side_override=rescue_side,
                                    )
                                    return False
                            if mom_rescue_banned(current_time, origin_strategy, origin_sub_strategy):
                                logging.info("⛔ CONSENSUS RESCUE BLOCKED: MomRescueBan")
                                return False
                            if hostile_day_active:
                                return False
                            if continuation_no_bypass and continuation_core_trigger(filter_name):
                                return False
                            flip_action, flip_meta = flip_confidence_decision(
                                filter_name,
                                reason,
                                signal,
                                base_session,
                                new_df,
                            )
                            if flip_action == "deny":
                                return False
                            potential_rescue = continuation_manager.get_active_continuation_signal(
                                new_df,
                                current_time,
                                rescue_side,
                                current_price=current_price,
                                trend_day_series=trend_day_series,
                                signal_mode=continuation_signal_mode,
                            )
                            if not continuation_rescue_allowed(
                                potential_rescue,
                                rescue_side,
                                current_price,
                                new_df,
                                trend_day_series,
                                continuation_allowlist,
                                continuation_allowed_regimes,
                                continuation_confirm_cfg,
                                continuation_guard_enabled,
                                continuation_signal_mode,
                            ):
                                potential_rescue = None
                            prefer_continuation = bool(flip_conf_cfg.get("prefer_continuation", True))
                            allow_direct_flip = bool(flip_conf_cfg.get("allow_direct_flip", True))
                            if flip_action == "allow" and (not potential_rescue or not prefer_continuation):
                                if not allow_direct_flip:
                                    return False
                                flip_signal = dict(signal)
                                flip_signal["side"] = rescue_side
                                flip_signal["entry_mode"] = "rescued"
                                flip_signal["rescue_from_strategy"] = origin_strategy
                                if origin_sub_strategy:
                                    flip_signal["rescue_from_sub_strategy"] = origin_sub_strategy
                                flip_signal["rescue_trigger"] = f"Flip:{filter_name}"
                                flip_signal["flip_filter"] = flip_meta.get("filter")
                                flip_signal["flip_key"] = flip_meta.get("key")
                                if trend_day_tier > 0 and trend_day_dir:
                                    flip_signal["trend_day_tier"] = trend_day_tier
                                    flip_signal["trend_day_dir"] = trend_day_dir
                                if _validate_signal_sltp(flip_signal, origin_strategy) is None:
                                    return False
                                rm_ok, rm_reason = apply_regime_meta(
                                    flip_signal,
                                    str(flip_signal.get("strategy", origin_strategy)),
                                )
                                if (not rm_ok) and regime_manifold_mode == "enforce":
                                    log_filter_block("RegimeManifold", rm_reason, side_override=flip_signal.get("side"))
                                    return False
                                logging.info(
                                    f"CONSENSUS FLIP: Blocked by {filter_name} ({reason}). "
                                    f"Direct flip to {flip_signal['side']}"
                                )
                                signal = flip_signal
                                rescue_context = {
                                    "original_strategy": origin_strategy,
                                    "rescue_strategy": "FlipConfidence",
                                    "bias": flip_signal["side"],
                                }
                                is_rescued = True
                                consensus_bypass_allowed = not continuation_no_bypass
                                consensus_rescued = consensus_bypass_allowed
                                return True
                            if not potential_rescue:
                                return False
                            rescue_blocked, rescue_reason = trend_filter.should_block_trade(new_df, potential_rescue['side'])
                            if rescue_blocked:
                                log_filter_block("TrendFilter", rescue_reason, side_override=potential_rescue['side'])
                                return False
                            rescue_label = potential_rescue.get("strategy", "ContinuationRescue")
                            if _validate_signal_sltp(potential_rescue, rescue_label) is None:
                                return False
                            rm_ok, rm_reason = apply_regime_meta(
                                potential_rescue,
                                str(potential_rescue.get("strategy", rescue_label)),
                            )
                            if (not rm_ok) and regime_manifold_mode == "enforce":
                                log_filter_block("RegimeManifold", rm_reason, side_override=potential_rescue.get("side"))
                                return False
                            signal = potential_rescue
                            signal['entry_mode'] = "rescued"
                            signal['rescue_from_strategy'] = origin_strategy
                            if origin_sub_strategy:
                                signal['rescue_from_sub_strategy'] = origin_sub_strategy
                            if trend_day_tier > 0 and trend_day_dir:
                                signal["trend_day_tier"] = trend_day_tier
                                signal["trend_day_dir"] = trend_day_dir
                            rescue_context = {
                                "original_strategy": origin_strategy,
                                "rescue_strategy": potential_rescue['strategy'],
                                "bias": potential_rescue['side'],
                            }
                            if should_defer_impulse_rescue(filter_name, reason):
                                signal["_defer_impulse_rescue"] = True
                                signal["_impulse_rescue_signal_time"] = current_time
                                signal["_impulse_rescue_signal_price"] = current_price
                                signal["_impulse_rescue_signal_close"] = float(currbar["close"])
                            is_rescued = True
                            consensus_bypass_allowed = not continuation_no_bypass
                            consensus_rescued = consensus_bypass_allowed
                            return True
                        if (not disable_strategy_filters) and trend_day_counter:
                            if not try_consensus_rescue(
                                f"TrendDayTier{trend_day_tier}",
                                "Counter-trend",
                            ):
                                logging.info(
                                    f"⛔ CONSENSUS BLOCKED by TrendDayTier{trend_day_tier}"
                                )
                                continue
                        if consensus_tp_signal is not None:
                            signal['tp_dist'] = consensus_tp_signal.get('tp_dist', signal.get('tp_dist'))
                            signal['sl_dist'] = consensus_tp_signal.get('sl_dist', signal.get('sl_dist'))
                            if consensus_tp_source:
                                logging.info(
                                    "🧮 CONSENSUS TP SOURCE: "
                                    f"{consensus_tp_source} TP={signal['tp_dist']:.2f} SL={signal['sl_dist']:.2f}"
                                )
                        if disable_strategy_filters:
                            fixed_ok, fixed_details = True, {}
                        else:
                            fixed_ok, fixed_details = apply_fixed_sltp(
                                signal,
                                new_df,
                                current_price,
                                ts=current_time,
                                session=base_session,
                                sl_dist_override=signal.get("sl_dist"),
                            )
                        if not fixed_ok:
                            reason = fixed_details.get("reason", "FixedSLTP blocked")
                            log_filter_block("FixedSLTP", reason)
                            if is_rescued:
                                log_rescue_failed(f"FixedSLTP: {reason}")
                            continue
                        if fixed_details:
                            signal["sl_dist"] = fixed_details["sl_dist"]
                            signal["tp_dist"] = fixed_details["tp_dist"]
                            signal["sltp_bracket"] = fixed_details.get("bracket")
                            signal["vol_regime"] = fixed_details.get("vol_regime", signal.get("vol_regime"))
                            log_fixed_sltp(fixed_details, signal.get("strategy"))
                        if disable_strategy_filters:
                            do_execute = True
                        else:
                            is_feasible, feasibility_reason = chop_analyzer.check_target_feasibility(
                                entry_price=current_price,
                                side=signal['side'],
                                tp_distance=signal.get('tp_dist'),
                                df_1m=new_df,
                            )
                            if (not is_feasible) and asia_calib_enabled and base_session == "ASIA":
                                tf_cfg = asia_calib_cfg.get("target_feasibility", {}) or {}
                                lookback = int(getattr(chop_analyzer, "LOOKBACK", 20) or 20)
                                if asia_target_feasibility_override(
                                    new_df,
                                    signal['side'],
                                    signal.get('tp_dist'),
                                    asia_trend_bias_side,
                                    tf_cfg,
                                    lookback,
                                ):
                                    is_feasible = True
                                    event_logger.log_filter_check(
                                        "TargetFeasibility",
                                        signal['side'],
                                        True,
                                        "ASIA trend override",
                                        strategy=signal.get('strategy', strat_name),
                                    )
                            if not is_feasible:
                                if try_consensus_rescue("TargetFeasibility", feasibility_reason):
                                    do_execute = True
                                else:
                                    logging.info(f"⛔ CONSENSUS BLOCKED by TargetFeasibility: {feasibility_reason}")
                                    continue
                            if not consensus_rescued:
                                regime_blocked, regime_reason = regime_blocker.should_block_trade(signal['side'], current_price)
                                if regime_blocked:
                                    if try_consensus_rescue("RegimeBlocker", regime_reason):
                                        do_execute = True
                                    else:
                                        logging.info(f"⛔ CONSENSUS BLOCKED by RegimeBlocker: {regime_reason}")
                                        continue
                            if not consensus_rescued:
                                dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(signal['side'], current_time)
                                if dir_blocked:
                                    if try_consensus_rescue("DirectionalLossBlocker", dir_reason):
                                        do_execute = True
                                    else:
                                        logging.info(f"⛔ CONSENSUS BLOCKED by DirectionalLossBlocker: {dir_reason}")
                                        continue
                            if not consensus_rescued:
                                trend_blocked, trend_reason = trend_filter.should_block_trade(new_df, signal['side'])
                                if trend_blocked:
                                    if try_consensus_rescue("TrendFilter", trend_reason):
                                        do_execute = True
                                    else:
                                        logging.info(f"⛔ CONSENSUS BLOCKED by TrendFilter: {trend_reason}")
                                        continue
                            if not consensus_rescued:
                                vol_regime, _, _ = volatility_filter.get_regime(new_df)
                                chop_blocked, chop_reason = chop_filter.should_block_trade(
                                    signal['side'],
                                    rejection_filter.prev_day_pm_bias,
                                    current_price,
                                    "NEUTRAL",
                                    vol_regime,
                                )
                                if chop_blocked and asia_calib_enabled and base_session == "ASIA":
                                    chop_cfg = asia_calib_cfg.get("chop_filter", {}) or {}
                                    if asia_chop_override(
                                        chop_reason,
                                        signal['side'],
                                        asia_trend_bias_side,
                                        chop_cfg,
                                    ):
                                        chop_blocked = False
                                        event_logger.log_filter_check(
                                            "ChopFilter",
                                            signal['side'],
                                            True,
                                            "ASIA trend override",
                                            strategy=signal.get('strategy', strat_name),
                                        )
                                if chop_blocked:
                                    if is_chop_hard_stop(chop_reason):
                                        log_filter_block("ChopFilter", chop_reason)
                                        logging.info("CHOP HARD-STOP: ChopFilter blocked (no rescue)")
                                        if is_rescued:
                                            log_rescue_failed(f"ChopFilter: {chop_reason}")
                                        continue
                                    if is_rescued:
                                        if consensus_bypass_allowed:
                                            logging.info(f"BYPASS Chop: Rescued by {signal['strategy']}")
                                        else:
                                            log_filter_block("ChopFilter", chop_reason)
                                            logging.info("CHOP BLOCKED: Rescue bypass disabled")
                                            log_rescue_failed(f"ChopFilter: {chop_reason}")
                                            continue
                                    elif not try_consensus_rescue("ChopFilter", chop_reason):
                                        continue

                            ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                            if ext_blocked and asia_soft_ext_enabled:
                                soft_score = asia_soft_ext_base - asia_soft_ext_penalty
                                if soft_score >= asia_soft_ext_threshold:
                                    ext_blocked = False
                                    ext_reason = (
                                        f"ASIA soft extension score {soft_score:.2f} >= "
                                        f"{asia_soft_ext_threshold:.2f}"
                                    )
                                    logging.info(f"✅ ExtensionFilter soft pass: {ext_reason}")
                            if ext_blocked:
                                if is_rescued:
                                    if consensus_bypass_allowed:
                                        logging.info(f"🛡️ BYPASS Extension: Rescued by {signal['strategy']}")
                                    else:
                                        log_filter_block("ExtensionFilter", ext_reason)
                                        logging.info("EXTENSION BLOCKED: Rescue bypass disabled")
                                        log_rescue_failed(f"ExtensionFilter: {ext_reason}")
                                        continue
                                elif not try_consensus_rescue("ExtensionFilter", ext_reason):
                                    continue

                            should_trade, vol_adj = check_volatility(
                                new_df,
                                signal.get('sl_dist'),
                                signal.get('tp_dist'),
                                base_size=_signal_base_size(signal, 5),
                            )
                            if not should_trade:
                                log_filter_block("VolatilityGuardrail", "Volatility check failed")
                                logging.info(f"⛔ BLOCKED by Volatility Guardrail")
                                if is_rescued:
                                    log_rescue_failed("VolatilityGuardrail: Volatility check failed")
                                continue

                            signal['sl_dist'] = vol_adj['sl_dist']
                            signal['tp_dist'] = vol_adj['tp_dist']
                            if vol_adj.get('adjustment_applied', False):
                                signal['size'] = vol_adj['size']
                            signal['vol_regime'] = vol_adj.get('regime', 'UNKNOWN')
                            if not ml_vol_regime_ok(signal, base_session, signal['vol_regime'], asia_viable=asia_viable):
                                log_filter_block("MLVolRegimeGuard", f"regime={signal['vol_regime']}")
                                if is_rescued:
                                    log_rescue_failed("MLVolRegimeGuard")
                                continue

                            do_execute = True

                        if not do_execute:
                            continue
                    else:
                        original_side = signal['side']
                        rescue_side = 'SHORT' if original_side == 'LONG' else 'LONG'
                        if hostile_day_active:
                            potential_rescue = None
                        else:
                            potential_rescue = continuation_manager.get_active_continuation_signal(
                                new_df,
                                current_time,
                                rescue_side,
                                current_price=current_price,
                                trend_day_series=trend_day_series,
                                signal_mode=continuation_signal_mode,
                            )
                        if not continuation_rescue_allowed(
                            potential_rescue,
                            rescue_side,
                            current_price,
                            new_df,
                            trend_day_series,
                            continuation_allowlist,
                            continuation_allowed_regimes,
                            continuation_confirm_cfg,
                            continuation_guard_enabled,
                            continuation_signal_mode,
                        ):
                            potential_rescue = None
                        logging.info(
                            f"EVALUATING {priority_label}: {strat_name} {original_side} | "
                            f"Rescue Available ({rescue_side}): {potential_rescue is not None}"
                        )

                        def try_rescue_trigger(block_reason, filter_name):
                            nonlocal signal, is_rescued, potential_rescue, rescue_context, rescue_bypass_allowed
                            log_filter_block(filter_name, block_reason)
                            if filter_name == 'ChopFilter' and is_chop_hard_stop(block_reason):
                                logging.info(f"CHOP HARD-STOP: {block_reason}")
                                return False
                            if is_wick_rejection_block(block_reason):
                                logging.info('RESCUE BLOCKED: TrendFilter wick rejection cooldown')
                                return False
                            if not allow_rescue:
                                return False
                            if trend_day_tier > 0 and trend_day_dir:
                                if (trend_day_dir == 'down' and rescue_side == 'LONG') or (trend_day_dir == 'up' and rescue_side == 'SHORT'):
                                    log_filter_block(
                                        f"TrendDayTier{trend_day_tier}",
                                        'Rescue side counter-trend',
                                        side_override=rescue_side,
                                    )
                                    return False
                            if mom_rescue_banned(current_time, origin_strategy, origin_sub_strategy):
                                logging.info('RESCUE BLOCKED: MomRescueBan')
                                return False
                            if continuation_no_bypass and continuation_core_trigger(filter_name):
                                return False
                            flip_action, flip_meta = flip_confidence_decision(
                                filter_name,
                                block_reason,
                                signal,
                                base_session,
                                new_df,
                            )
                            if flip_action == "deny":
                                return False
                            prefer_continuation = bool(flip_conf_cfg.get("prefer_continuation", True))
                            allow_direct_flip = bool(flip_conf_cfg.get("allow_direct_flip", True))
                            if flip_action == "allow" and (not potential_rescue or not prefer_continuation):
                                if not allow_direct_flip:
                                    return False
                                flip_signal = dict(signal)
                                flip_signal["side"] = rescue_side
                                flip_signal["entry_mode"] = "rescued"
                                flip_signal["rescue_from_strategy"] = origin_strategy
                                if origin_sub_strategy:
                                    flip_signal["rescue_from_sub_strategy"] = origin_sub_strategy
                                flip_signal["rescue_trigger"] = f"Flip:{filter_name}"
                                flip_signal["flip_filter"] = flip_meta.get("filter")
                                flip_signal["flip_key"] = flip_meta.get("key")
                                if trend_day_tier > 0 and trend_day_dir:
                                    flip_signal["trend_day_tier"] = trend_day_tier
                                    flip_signal["trend_day_dir"] = trend_day_dir
                                if _validate_signal_sltp(flip_signal, origin_strategy) is None:
                                    return False
                                rm_ok, rm_reason = apply_regime_meta(
                                    flip_signal,
                                    str(flip_signal.get("strategy", origin_strategy)),
                                )
                                if (not rm_ok) and regime_manifold_mode == "enforce":
                                    log_filter_block("RegimeManifold", rm_reason, side_override=flip_signal.get("side"))
                                    return False
                                logging.info(
                                    f"RESCUE FLIP: Blocked by {filter_name} ({block_reason}). "
                                    f"Direct flip to {flip_signal['side']}"
                                )
                                signal = flip_signal
                                rescue_context = {
                                    'original_strategy': origin_strategy,
                                    'rescue_strategy': 'FlipConfidence',
                                    'bias': flip_signal['side'],
                                }
                                is_rescued = True
                                rescue_bypass_allowed = not continuation_no_bypass
                                potential_rescue = None
                                return True
                            if potential_rescue and not is_rescued:
                                rescue_blocked, rescue_reason = trend_filter.should_block_trade(new_df, potential_rescue['side'])
                                if rescue_blocked:
                                    log_filter_block('TrendFilter', rescue_reason, side_override=potential_rescue['side'])
                                    return False
                                rm_ok, rm_reason = apply_regime_meta(
                                    potential_rescue,
                                    str(potential_rescue.get("strategy", strat_name)),
                                )
                                if (not rm_ok) and regime_manifold_mode == "enforce":
                                    log_filter_block("RegimeManifold", rm_reason, side_override=potential_rescue.get("side"))
                                    return False
                                logging.info(f"RESCUE FLIP: Blocked by {filter_name} ({block_reason}). "
                                             f"Flipping to {potential_rescue['strategy']} ({potential_rescue['side']})")
                                signal = potential_rescue
                                signal['entry_mode'] = 'rescued'
                                signal['rescue_from_strategy'] = origin_strategy
                                if origin_sub_strategy:
                                    signal['rescue_from_sub_strategy'] = origin_sub_strategy
                                if trend_day_tier > 0 and trend_day_dir:
                                    signal['trend_day_tier'] = trend_day_tier
                                    signal['trend_day_dir'] = trend_day_dir
                                rescue_context = {
                                    'original_strategy': origin_strategy,
                                    'rescue_strategy': potential_rescue['strategy'],
                                    'bias': potential_rescue['side'],
                                }
                                if should_defer_impulse_rescue(filter_name, block_reason):
                                    signal['_defer_impulse_rescue'] = True
                                    signal['_impulse_rescue_signal_time'] = current_time
                                    signal['_impulse_rescue_signal_price'] = current_price
                                    signal['_impulse_rescue_signal_close'] = float(currbar['close'])
                                is_rescued = True
                                rescue_bypass_allowed = not continuation_no_bypass
                                potential_rescue = None
                                return True
                            else:
                                logging.info(f"BLOCKED by {filter_name}: {block_reason}")
                                return False

                        # TrendDay counter-trend hard block/rescue attempt
                        if (not disable_strategy_filters) and trend_day_counter:
                            if not try_rescue_trigger('Counter-trend', f"TrendDayTier{trend_day_tier}"):
                                logging.info(f"BLOCKED by TrendDayTier{trend_day_tier}")
                                continue

                        if disable_strategy_filters:
                            fixed_ok, fixed_details = True, {}
                        else:
                            fixed_ok, fixed_details = apply_fixed_sltp(
                                signal,
                                new_df,
                                current_price,
                                ts=current_time,
                                session=base_session,
                                sl_dist_override=signal.get("sl_dist"),
                            )
                        if not fixed_ok:
                            reason = fixed_details.get("reason", "FixedSLTP blocked")
                            log_filter_block("FixedSLTP", reason)
                            if is_rescued:
                                log_rescue_failed(f"FixedSLTP: {reason}")
                            continue
                        if fixed_details:
                            signal["sl_dist"] = fixed_details["sl_dist"]
                            signal["tp_dist"] = fixed_details["tp_dist"]
                            signal["sltp_bracket"] = fixed_details.get("bracket")
                            signal["vol_regime"] = fixed_details.get("vol_regime", signal.get("vol_regime"))
                            log_fixed_sltp(fixed_details, signal.get("strategy"))

                        if disable_strategy_filters:
                            do_execute = True
                        else:
                            is_feasible, feasibility_reason = chop_analyzer.check_target_feasibility(
                                entry_price=current_price, side=signal['side'], tp_distance=signal.get('tp_dist'), df_1m=new_df
                            )
                            if (not is_feasible) and asia_calib_enabled and base_session == "ASIA":
                                tf_cfg = asia_calib_cfg.get("target_feasibility", {}) or {}
                                lookback = int(getattr(chop_analyzer, "LOOKBACK", 20) or 20)
                                if asia_target_feasibility_override(
                                    new_df,
                                    signal['side'],
                                    signal.get('tp_dist'),
                                    asia_trend_bias_side,
                                    tf_cfg,
                                    lookback,
                                ):
                                    is_feasible = True
                                    event_logger.log_filter_check(
                                        "TargetFeasibility",
                                        signal['side'],
                                        True,
                                        "ASIA trend override",
                                        strategy=signal.get('strategy', strat_name),
                                    )
                            if not is_feasible:
                                log_filter_block('TargetFeasibility', feasibility_reason)
                                logging.info(f"Signal ignored ({priority_label}): {feasibility_reason}")
                                continue

                            rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                            range_bias_blocked = (allowed_chop_side is not None and signal['side'] != allowed_chop_side)
                            if rej_blocked or range_bias_blocked:
                                reason = rej_reason if rej_blocked else f"Opposite HTF Range Bias ({allowed_chop_side})"
                                if not try_rescue_trigger(reason, 'Rejection/Bias'):
                                    continue

                            dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(signal['side'], current_time)
                            if dir_blocked:
                                if not try_rescue_trigger(dir_reason, 'DirectionalLoss'):
                                    continue

                            impulse_blocked, impulse_reason = impulse_filter.should_block_trade(signal['side'])
                            if impulse_blocked:
                                if not try_rescue_trigger(impulse_reason, 'ImpulseFilter'):
                                    continue

                            regime_blocked, regime_reason = regime_blocker.should_block_trade(signal['side'], current_price)
                            if regime_blocked:
                                log_filter_block('RegimeBlocker', regime_reason)
                                logging.info(f"HARD STOP by RegimeBlocker (EQH/EQL): {regime_reason}")
                                continue

                            upgraded_blocked = False
                            upgraded_reasons = []
                            tp_dist = signal.get('tp_dist', 15.0)
                            effective_tp_dist = tp_dist
                            if allowed_chop_side is not None and signal['side'] == allowed_chop_side:
                                effective_tp_dist = tp_dist * 0.5
                                logging.info(f"RELAXING FVG CHECK (Main): Fading Range {signal['side']} (Req Room: {effective_tp_dist*0.4:.2f} pts)")
                            fvg_blocked = False
                            fvg_reason = None
                            if htf_fvg_enabled_live:
                                fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                                    signal['side'], current_price, None, None, tp_dist=effective_tp_dist
                                )
                                if fvg_blocked:
                                    upgraded_reasons.append(f"HTF_FVG: {fvg_reason}")
                            de3_rej_ok, de3_rej_detail = _de3_timeframe_rejection_bypass(
                                signal,
                                signal.get("side"),
                                df_5m,
                                df_15m,
                            )
                            struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                            if (
                                struct_blocked
                                and de3_rej_ok
                                and "no volume/wick rejection" in str(struct_reason or "").lower()
                            ):
                                logging.info(
                                    "DE3 rejection bypass: StructureBlocker (%s)",
                                    de3_rej_detail,
                                )
                                struct_blocked = False
                                struct_reason = f"DE3 bypass ({de3_rej_detail})"
                            if struct_blocked:
                                upgraded_reasons.append(f"Structure: {struct_reason}")
                            bank_blocked, bank_reason = bank_filter.should_block_trade(signal['side'])
                            if bank_blocked:
                                upgraded_reasons.append(f"Bank: {bank_reason}")
                            upg_trend_blocked, upg_trend_reason = trend_filter.should_block_trade(new_df, signal['side'])
                            if upg_trend_blocked:
                                upgraded_reasons.append(f"Trend: {upg_trend_reason}")
                            if upgraded_reasons:
                                upgraded_blocked = True
                            legacy_blocked, legacy_reason = legacy_filters.check_trend(new_df, signal['side'])
                            if legacy_blocked and de3_rej_ok:
                                logging.info(
                                    "DE3 rejection bypass: LegacyTrend (%s)",
                                    de3_rej_detail,
                                )
                                legacy_blocked = False
                                legacy_reason = f"DE3 bypass ({de3_rej_detail})"
                            final_blocked = False
                            final_reason = ''
                            arb_blocked = False
                            if legacy_blocked and upgraded_blocked:
                                final_blocked = True
                                final_reason = f"Unanimous: {legacy_reason} & {upgraded_reasons}"
                            elif not legacy_blocked and upgraded_blocked:
                                arb = filter_arbitrator.arbitrate(
                                    new_df, signal['side'], False, '', True, '|'.join(upgraded_reasons), current_price, signal.get('tp_dist'), signal.get('sl_dist')
                                )
                                if not arb.allow_trade:
                                    final_blocked = True
                                    final_reason = arb.reason
                                    arb_blocked = True
                            blocked_filters = []
                            if legacy_blocked:
                                blocked_filters.append("LegacyTrend")
                            if struct_blocked:
                                blocked_filters.append("StructureBlocker")
                            if bank_blocked:
                                blocked_filters.append("BankLevelQuarterFilter")
                            if upg_trend_blocked:
                                blocked_filters.append("TrendFilter")
                            if arb_blocked:
                                blocked_filters.append("FilterArbitrator")
                            filter_stack_label = "FilterStack"
                            if blocked_filters:
                                filter_stack_label = f"FilterStack:{'+'.join(blocked_filters)}"
                            if final_blocked:
                                if is_rescued:
                                    if rescue_bypass_allowed:
                                        logging.info(f"BYPASS Filters ({final_reason}): Rescued by {signal['strategy']}")
                                    else:
                                        log_filter_block(filter_stack_label, final_reason)
                                        logging.info("FILTER STACK BLOCKED: Rescue bypass disabled")
                                        log_rescue_failed(f"FilterStack: {final_reason}")
                                        continue
                                else:
                                    if not try_rescue_trigger(final_reason, filter_stack_label):
                                        continue

                            vol_regime, _, _ = volatility_filter.get_regime(new_df)
                            chop_blocked, chop_reason = chop_filter.should_block_trade(
                                signal['side'], rejection_filter.prev_day_pm_bias, current_price, 'NEUTRAL', vol_regime
                            )
                            if chop_blocked and asia_calib_enabled and base_session == "ASIA":
                                chop_cfg = asia_calib_cfg.get("chop_filter", {}) or {}
                                if asia_chop_override(
                                    chop_reason,
                                    signal['side'],
                                    asia_trend_bias_side,
                                    chop_cfg,
                                ):
                                    chop_blocked = False
                                    event_logger.log_filter_check(
                                        "ChopFilter",
                                        signal['side'],
                                        True,
                                        "ASIA trend override",
                                        strategy=signal.get('strategy', strat_name),
                                    )
                            if chop_blocked:
                                if is_chop_hard_stop(chop_reason):
                                    log_filter_block('ChopFilter', chop_reason)
                                    logging.info('CHOP HARD-STOP: ChopFilter blocked (no rescue)')
                                    if is_rescued:
                                        log_rescue_failed(f"ChopFilter: {chop_reason}")
                                    continue
                                if is_rescued:
                                    if rescue_bypass_allowed:
                                        logging.info(f"BYPASS Chop: Rescued by {signal['strategy']}")
                                    else:
                                        log_filter_block('ChopFilter', chop_reason)
                                        logging.info('CHOP BLOCKED: Rescue bypass disabled')
                                        log_rescue_failed(f"ChopFilter: {chop_reason}")
                                        continue
                                elif not try_rescue_trigger(chop_reason, 'ChopFilter'):
                                    continue
                            ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                            if ext_blocked and asia_soft_ext_enabled:
                                soft_score = asia_soft_ext_base - asia_soft_ext_penalty
                                if soft_score >= asia_soft_ext_threshold:
                                    ext_blocked = False
                                    ext_reason = (
                                        f"ASIA soft extension score {soft_score:.2f} >= "
                                        f"{asia_soft_ext_threshold:.2f}"
                                    )
                                    logging.info(f"✅ ExtensionFilter soft pass: {ext_reason}")
                            if ext_blocked:
                                if is_rescued:
                                    if rescue_bypass_allowed:
                                        logging.info(f"BYPASS Extension: Rescued by {signal['strategy']}")
                                    else:
                                        log_filter_block('ExtensionFilter', ext_reason)
                                        logging.info('EXTENSION BLOCKED: Rescue bypass disabled')
                                        log_rescue_failed(f"ExtensionFilter: {ext_reason}")
                                        continue
                                elif not try_rescue_trigger(ext_reason, 'ExtensionFilter'):
                                    continue

                            should_trade, vol_adj = check_volatility(
                                new_df,
                                signal.get('sl_dist'),
                                signal.get('tp_dist'),
                                base_size=_signal_base_size(signal, 5),
                            )
                            if not should_trade:
                                log_filter_block('VolatilityGuardrail', 'Volatility check failed')
                                logging.info('BLOCKED by Volatility Guardrail')
                                if is_rescued:
                                    log_rescue_failed('VolatilityGuardrail: Volatility check failed')
                                continue
                            signal['sl_dist'] = vol_adj['sl_dist']
                            signal['tp_dist'] = vol_adj['tp_dist']
                            if vol_adj.get('adjustment_applied', False):
                                signal['size'] = vol_adj['size']
                            signal['vol_regime'] = vol_adj.get('regime', 'UNKNOWN')
                            if not ml_vol_regime_ok(signal, base_session, signal['vol_regime'], asia_viable=asia_viable):
                                log_filter_block("MLVolRegimeGuard", f"regime={signal['vol_regime']}")
                                if is_rescued:
                                    log_rescue_failed("MLVolRegimeGuard")
                                continue
                            do_execute = True
                    if not do_execute:
                        continue

                    # Decision summary for UI/logs
                    try:
                        decision_reason = (
                            f"priority={priority_label} entry_mode={signal.get('entry_mode','standard')} "
                            f"consensus={consensus_side or 'none'} "
                            f"vol_regime={signal.get('vol_regime', vol_regime_current) or 'UNKNOWN'} "
                            f"trend_day={trend_day_tier if trend_day_tier is not None else 0}/{trend_day_dir or 'none'}"
                        )
                        event_logger.log_filter_check(
                            "Decision",
                            signal['side'],
                            True,
                            decision_reason,
                            strategy=signal.get('strategy', strat_name)
                        )
                    except Exception:
                        pass
                    # === EXECUTION ===
                    sltp = _validate_signal_sltp(signal, signal.get("strategy", strat_name))
                    if sltp is None:
                        continue
                    signal.setdefault('entry_mode', "standard")
                    if signal.get("_defer_impulse_rescue"):
                        pending_entry = {
                            "signal": signal,
                            "signal_time": signal.pop("_impulse_rescue_signal_time", current_time),
                            "signal_price": signal.pop("_impulse_rescue_signal_price", current_price),
                            "signal_close": signal.pop("_impulse_rescue_signal_close", current_price),
                        }
                        signal.pop("_defer_impulse_rescue", None)
                        new_side = pending_entry["signal"].get("side")
                        if pending_impulse_rescues:
                            existing_side = pending_impulse_rescues[0]["signal"].get("side")
                            if existing_side != new_side:
                                pending_impulse_rescues.clear()
                                logging.info(
                                    "⏳ RESCUE DEFERRED: cleared pending opposite-direction rescue"
                                )
                        if not pending_impulse_rescues or pending_impulse_rescues[0]["signal"].get("side") == new_side:
                            _signal_birth_hook(pending_entry["signal"])
                            _attach_pct_overlay_snapshot(pending_entry["signal"])
                            pending_impulse_rescues.append(pending_entry)
                            logging.info("⏳ RESCUE DEFERRED: waiting for next bar confirmation")
                        signal_executed = True
                        break
                    allow_dyn_mom_solo = False
                    if (
                        signal.get('strategy') in ("DynamicEngine", "DynamicEngineStrategy")
                        and signal.get('entry_mode') not in ("consensus", "rescued")
                    ):
                        mom_key = get_mom_rescue_key(signal.get('strategy'), signal.get('sub_strategy'))
                        if mom_key:
                            mom_side_ok = (
                                (mom_key == "Long_Mom" and signal.get("side") == "LONG")
                                or (mom_key == "Short_Mom" and signal.get("side") == "SHORT")
                            )
                            trend_day_align = (
                                trend_day_tier > 0
                                and trend_day_dir is not None
                                and (
                                    (trend_day_dir == "up" and signal.get("side") == "LONG")
                                    or (trend_day_dir == "down" and signal.get("side") == "SHORT")
                                )
                            )
                            if mom_side_ok and trend_day_align:
                                allow_dyn_mom_solo = True
                                logging.info(
                                    "🟢 DynamicEngine solo allowed (TrendDay-aligned Mom): "
                                    f"{mom_key} {signal.get('side')} | tier={trend_day_tier} dir={trend_day_dir}"
                                )
                    if (
                        not ALLOW_DYNAMIC_ENGINE_SOLO
                        and signal.get('strategy') in ("DynamicEngine", "DynamicEngineStrategy")
                        and signal.get('entry_mode') not in ("consensus", "rescued")
                        and not allow_dyn_mom_solo
                    ):
                        log_filter_block("DynamicEngineSolo", "DynamicEngine solo blocked")
                        continue
                    signal["size"] = _apply_live_execution_size(
                        signal,
                        5,
                        live_drawdown_state,
                        tracked_live_trades(),
                    )
                    if signal["size"] <= 0:
                        logging.info("Kalshi HARD VETO — trade blocked (size=0): %s", signal.get("strategy", strat_name))
                        continue
                    if _same_side_active_trade(active_trade, signal):
                        if _allow_same_side_parallel_entry(active_trade, signal, tracked_live_trades()):
                            reset_opposite_reversal_state("same-side coexist signal")
                            block_reason = _live_entry_window_block_reason(current_time)
                            if block_reason:
                                add_strategy_slot(
                                    "rejected",
                                    signal.get("strategy", strat_name),
                                    signal,
                                    fallback=strat_name,
                                )
                                _log_live_entry_window_block(
                                    signal,
                                    signal.get("strategy", strat_name),
                                    current_time,
                                )
                                continue
                            live_entry_day_block = _resolve_live_de3_entry_trade_day_extreme_admission_block(
                                signal,
                                entry_price=float(current_price),
                                tp_dist=_coerce_float(signal.get("tp_dist"), 0.0),
                                current_time=current_time,
                                current_price=float(current_price),
                                market_df=new_df,
                            )
                            if bool(live_entry_day_block.get("active", False)):
                                signal["de3_entry_trade_day_extreme_admission_blocked"] = True
                                signal["de3_entry_trade_day_extreme_admission_profile"] = str(
                                    live_entry_day_block.get("profile_name", "") or ""
                                )
                                add_strategy_slot(
                                    "rejected",
                                    signal.get("strategy", strat_name),
                                    signal,
                                    fallback=strat_name,
                                )
                                logging.info(
                                    "🚫 DE3 entry-day-extreme admission block: %s | profile=%s | beyond=%s | progress=%.2f",
                                    signal.get("sub_strategy") or signal.get("strategy"),
                                    str(live_entry_day_block.get("profile_name", "") or ""),
                                    bool(
                                        live_entry_day_block.get(
                                            "target_beyond_trade_day_extreme",
                                            False,
                                        )
                                    ),
                                    float(
                                        _coerce_float(
                                            live_entry_day_block.get("progress_pct"),
                                            0.0,
                                        )
                                    ),
                                )
                                continue
                            adjusted_size = _apply_live_de3_entry_trade_day_extreme_size_adjustment(
                                signal,
                                current_size=_coerce_int(signal.get("size"), 1),
                                entry_price=float(current_price),
                                tp_dist=_coerce_float(signal.get("tp_dist"), 0.0),
                                current_time=current_time,
                                current_price=float(current_price),
                                market_df=new_df,
                            )
                            if bool(
                                signal.get(
                                    "de3_entry_trade_day_extreme_size_adjustment_applied",
                                    False,
                                )
                            ):
                                logging.info(
                                    "🪶 DE3 entry-day-extreme size trim: %s | profile=%s | size %s -> %s | beyond=%s | progress=%.2f",
                                    signal.get("sub_strategy") or signal.get("strategy"),
                                    str(
                                        signal.get(
                                            "de3_entry_trade_day_extreme_size_adjustment_profile_name",
                                            "",
                                        )
                                        or ""
                                    ),
                                    int(
                                        _coerce_int(
                                            signal.get(
                                                "de3_entry_trade_day_extreme_size_adjustment_requested_size"
                                            ),
                                            adjusted_size,
                                        )
                                    ),
                                    int(adjusted_size),
                                    bool(
                                        signal.get(
                                            "de3_entry_trade_day_extreme_size_adjustment_target_beyond",
                                            False,
                                        )
                                    ),
                                    float(
                                        _coerce_float(
                                            signal.get(
                                                "de3_entry_trade_day_extreme_size_adjustment_progress_pct"
                                            ),
                                            0.0,
                                        )
                                    ),
                                )
                            if not _apply_kalshi_trade_overlay_to_signal(
                                signal,
                                current_price,
                                new_df,
                                price_action_profile=kalshi_price_action_profile,
                            ):
                                add_strategy_slot(
                                    "rejected",
                                    signal.get("strategy", strat_name),
                                    signal,
                                    fallback=strat_name,
                                )
                                continue
                            log_rescue_success()
                            add_strategy_slot(
                                "executed",
                                signal.get("strategy", strat_name),
                                signal,
                                fallback=strat_name,
                            )
                            logging.info(
                                "✅ %s EXEC (same-side coexist): %s (%s) alongside %s",
                                priority_label,
                                signal['strategy'],
                                signal['side'],
                                active_trade.get("strategy"),
                            )
                            if not _resolve_pct_overlay_snapshot(signal):
                                add_strategy_slot(
                                    "rejected",
                                    signal.get("strategy", strat_name),
                                    signal,
                                    fallback=strat_name,
                                )
                                signal_executed = True
                                break
                            if circuit_breaker.is_tripped:
                                logging.info("🚫 CB-entry-gate: skipping entry (%s %s)",
                                             signal.get("strategy","?"), signal.get("side","?"))
                                signal_executed = True
                                break
                            from regime_classifier import should_veto_entry as _regime_veto
                            _rv, _rr = _regime_veto()
                            if _rv:
                                logging.info("🚫 regime-veto: skipping entry (%s %s) — %s",
                                             signal.get("strategy","?"), signal.get("side","?"), _rr)
                                signal_executed = True
                                break
                            _lfgv, _lfgr = _lfg_should_veto_entry(signal, market_time_et)
                            if _lfgv:
                                logging.info("🚫 lfg-veto: skipping entry (%s %s) — %s",
                                             signal.get("strategy","?"), signal.get("side","?"), _lfgr)
                                signal_executed = True
                                break
                            order_response = await client.async_place_order(signal, current_price)
                            if order_response is not None:
                                order_details = getattr(client, "_last_order_details", None) or {}
                                entry_price = order_details.get("entry_price", current_price)
                                tp_dist = order_details.get("tp_points")
                                if tp_dist is None:
                                    tp_dist = signal.get('tp_dist')
                                sl_dist = order_details.get("sl_points")
                                if sl_dist is None:
                                    sl_dist = signal.get('sl_dist')
                                if tp_dist is None or sl_dist is None:
                                    logging.error("Order details missing sl/tp after same-side execution; using 0.0 for tracking")
                                    tp_dist = tp_dist or 0.0
                                    sl_dist = sl_dist or 0.0
                                size = order_details.get("size", signal.get('size', 5))
                                signal['tp_dist'] = tp_dist
                                signal['sl_dist'] = sl_dist
                                signal['size'] = size
                                signal['entry_price'] = entry_price
                                add_tracked_live_trade(
                                    _build_live_active_trade(
                                        signal,
                                        order_details,
                                        current_price,
                                        current_time,
                                        bar_count,
                                        market_df=new_df,
                                        stop_order_id=order_details.get(
                                            "stop_order_id",
                                            getattr(client, "_active_stop_order_id", None),
                                        ),
                                    )
                                )
                                persist_runtime_state(current_time, reason="standard_parallel_entry")
                                log_trade_factor_snapshot(
                                    source="standard_parallel_execution",
                                    signal_payload=signal,
                                    order_details=order_details,
                                    entry_price=float(entry_price),
                                    market_df=new_df,
                                    event_time=current_time,
                                    market_price=float(current_price),
                                    base_session_name=base_session,
                                    current_session_name=current_session_name,
                                    vol_regime_name=vol_regime_current,
                                    trend_tier=trend_day_tier,
                                    trend_dir=trend_day_dir,
                                    strategy_eval=strategy_results,
                                    regime_snapshot=regime_meta,
                                    allowed_chop_bias=allowed_chop_side,
                                    asia_viable_flag=asia_viable,
                                )
                            signal_executed = True
                            break
                        reset_opposite_reversal_state("same-side signal ignored")
                        logging.info(
                            "Ignoring same-side signal while %s position is already active: %s",
                            active_trade.get("side"),
                            signal.get("strategy", strat_name),
                        )
                        signal_executed = True
                        break
                    block_reason = _live_entry_window_block_reason(current_time)
                    if block_reason:
                        add_strategy_slot(
                            "rejected",
                            signal.get("strategy", strat_name),
                            signal,
                            fallback=strat_name,
                        )
                        _log_live_entry_window_block(
                            signal,
                            signal.get("strategy", strat_name),
                            current_time,
                        )
                        continue
                    log_rescue_success()
                    current_trades = tracked_live_trades()
                    old_trades = [dict(trade) for trade in current_trades]
                    if old_trades and not opposite_reversal_matches_active_trade_family(
                        signal,
                        current_trades,
                    ):
                        reset_opposite_reversal_state(
                            "opposite active-trade family mismatch"
                        )
                        log_opposite_reversal_active_trade_family_block(
                            signal,
                            current_trades,
                            prefix="Holding position",
                        )
                        signal_executed = True
                        break

                    if old_trades and same_strategy_opposite_reversal_blocked(
                        signal,
                        current_trades,
                    ):
                        reset_opposite_reversal_state(
                            "same-strategy opposite reversal blocked"
                        )
                        log_same_strategy_opposite_reversal_block(
                            signal,
                            current_trades,
                            prefix="Holding position",
                        )
                        signal_executed = True
                        break

                    # === LEVEL-AWARE FILL CHECK ===
                    # Evaluate proximity to structural / bank levels.
                    # WAIT  → defer signal; fire on level touch (up to 3 bars)
                    # AT_LEVEL → execute immediately with a "best fill" log
                    # IMMEDIATE → proceed normally (default behaviour)
                    _lfo_decision = None
                    if (
                        level_fill_optimizer is not None
                        and not is_rescued
                        and not pending_level_fills
                        and signal.get("entry_mode") not in ("loose", "level_fill")
                    ):
                        try:
                            _lfo_bar = {
                                "open":  float(currbar["open"]),
                                "high":  float(currbar["high"]),
                                "low":   float(currbar["low"]),
                                "close": float(currbar["close"]),
                            }
                            _lfo_decision = level_fill_optimizer.evaluate(
                                signal,
                                float(current_price),
                                structural_tracker,
                                bank_filter,
                                bar_candle=_lfo_bar,
                            )
                        except Exception as _lfo_exc:
                            logging.warning("LevelFillOptimizer error: %s", _lfo_exc)
                            _lfo_decision = None

                        # SHADOW ML LFO scoring — logs a parallel ML prediction
                        # alongside the rule-based decision for A/B comparison.
                        try:
                            import ml_overlay_shadow as _mls
                            if _mls._LFO_PAYLOAD is not None:
                                import signal_gate_2025 as _sg
                                bank_base = (float(current_price) // 12.5) * 12.5
                                _dist_below = float(current_price) - bank_base
                                _dist_above = (bank_base + 12.5) - float(current_price)
                                _bar_range = float(_lfo_bar["high"]) - float(_lfo_bar["low"])
                                _body_pct = (float(_lfo_bar["close"]) - float(_lfo_bar["low"])) / max(0.01, _bar_range)
                                _bar_cache = getattr(loss_factor_guard, "_bar_cache", None) if loss_factor_guard else None
                                _feats = _sg.compute_bar_features_from_ohlcv(list(_bar_cache)) if _bar_cache and len(_bar_cache) >= 45 else {}
                                _sess = ("ASIA" if currbar.name.hour < 3 or currbar.name.hour >= 18 else
                                         "LONDON" if currbar.name.hour < 7 else
                                         "NY_PRE" if currbar.name.hour < 9 else
                                         "NY" if currbar.name.hour < 16 else "POST")
                                try:
                                    from regime_classifier import current_regime as _cr
                                    _mkt_regime = _cr() or ""
                                except Exception:
                                    _mkt_regime = ""
                                # Build bars_df from _bar_cache tuples for v2 LFO
                                # (encoder + cross-market features). Harmless when v1
                                # model is loaded — score_lfo skips augmentation.
                                _bars_df = None
                                if _bar_cache and len(_bar_cache) >= 10:
                                    try:
                                        import pandas as _pd_local
                                        _rows = list(_bar_cache)
                                        _bars_df = _pd_local.DataFrame(
                                            [(r[1], r[2], r[3], r[4], r[5]) for r in _rows],
                                            columns=["open", "high", "low", "close", "volume"],
                                            index=_pd_local.DatetimeIndex([r[0] for r in _rows]),
                                        )
                                    except Exception:
                                        _bars_df = None
                                # Live cross-market overrides — master_vix_df and
                                # master_mnq_df are accumulated elsewhere in the
                                # scan loop; passing them unconditionally is safe
                                # (v1 payloads + empty frames both ignore).
                                _ml_score = _mls.score_lfo(
                                    signal=signal,
                                    bar_features=_feats,
                                    dist_to_bank_below=_dist_below,
                                    dist_to_bank_above=_dist_above,
                                    bar_range_pts=_bar_range,
                                    bar_close_pct_body=_body_pct,
                                    sl_dist=float(signal.get("sl_dist") or 0),
                                    tp_dist=float(signal.get("tp_dist") or 0),
                                    session=_sess,
                                    mkt_regime=_mkt_regime,
                                    et_hour=int(currbar.name.hour),
                                    bars_df=_bars_df,
                                    current_time=currbar.name,
                                    vix_override_df=master_vix_df,
                                    mnq_override_df=master_mnq_df,
                                )
                                if _ml_score is not None:
                                    _p_wait, _thr = _ml_score
                                    _ml_choice = "WAIT" if _p_wait >= _thr else "IMMEDIATE"
                                    _rule_choice = _lfo_decision.get("mode", "IMMEDIATE") if _lfo_decision else "IMMEDIATE"
                                    logging.info(
                                        "[SHADOW_LFO] rule=%s ml=%s p_wait=%.3f thr=%.3f strat=%s side=%s",
                                        _rule_choice, _ml_choice, _p_wait, _thr,
                                        signal.get("strategy","?"), signal.get("side","?"),
                                    )
                                    if _mls.is_lfo_live_active():
                                        # Live mode: override _lfo_decision based on ML
                                        if _ml_choice == "IMMEDIATE" and _lfo_decision is not None:
                                            _lfo_decision["mode"] = "IMMEDIATE"
                                            _lfo_decision["reason"] = f"ml_lfo p_wait={_p_wait:.3f}<{_thr}"
                        except Exception as _ml_exc:
                            logging.debug("shadow ML LFO scoring failed: %s", _ml_exc, exc_info=True)

                    if _lfo_decision is not None and _lfo_decision.get("mode") == FILL_WAIT:
                        import uuid as _uuid_mod
                        _lfo_uid = str(_uuid_mod.uuid4())[:8]
                        signal["entry_mode"] = "level_fill_pending"
                        _signal_birth_hook(signal)
                        _attach_pct_overlay_snapshot(signal)
                        level_fill_optimizer.add_pending(_lfo_uid, signal, _lfo_decision, float(current_price))
                        pending_level_fills[_lfo_uid] = level_fill_optimizer.get_pending_signal(_lfo_uid)
                        add_strategy_slot("rejected", signal.get("strategy", strat_name), signal, fallback=strat_name)
                        signal_executed = True
                        break  # wait for level touch on next bar(s)

                    if _lfo_decision is not None and _lfo_decision.get("mode") == FILL_AT_LEVEL:
                        logging.info(
                            "📌 LevelFill AT LEVEL: %s %s @ %s (%.2f pts) — best fill",
                            signal.get("strategy", strat_name),
                            signal.get("side"),
                            _lfo_decision.get("target_name"),
                            _lfo_decision.get("dist") or 0,
                        )

                    if not _apply_kalshi_trade_overlay_to_signal(
                        signal,
                        current_price,
                        new_df,
                        price_action_profile=kalshi_price_action_profile,
                    ):
                        add_strategy_slot(
                            "rejected",
                            signal.get("strategy", strat_name),
                            signal,
                            fallback=strat_name,
                        )
                        continue

                    add_strategy_slot(
                        "executed",
                        signal.get("strategy", strat_name),
                        signal,
                        fallback=strat_name,
                    )
                    logging.info(f"✅ {priority_label} EXEC: {signal['strategy']} ({signal['side']})")

                    # ... [Remaining Execution Code same as before] ...
                    # Close and Reverse logic...
                    reverse_state_count = 0
                    if old_trades:
                        reverse_confirmed, reverse_state_count = note_opposite_reversal_signal(
                            signal,
                            bar_count,
                        )
                        if not reverse_confirmed:
                            remaining = max(
                                0,
                                opposite_reversal_required - reverse_state_count,
                            )
                            logging.info(
                                "Holding position: opposite confirmation %s/%s for %s "
                                "(need %s more within %s bars)",
                                reverse_state_count,
                                opposite_reversal_required,
                                signal.get("side"),
                                remaining,
                                opposite_reversal_window_bars,
                            )
                            signal_executed = True
                            break

                    if not _resolve_pct_overlay_snapshot(signal):
                        add_strategy_slot(
                            "rejected",
                            signal.get("strategy", strat_name),
                            signal,
                            fallback=strat_name,
                        )
                        signal_executed = True
                        break

                    if circuit_breaker.is_tripped:
                        logging.info("🚫 CB-entry-gate: skipping reversal (%s %s)", signal.get("strategy","?"), signal.get("side","?"))
                        signal_executed = True
                        break
                    from regime_classifier import should_veto_entry as _regime_veto
                    _rv, _rr = _regime_veto()
                    if _rv:
                        logging.info("🚫 regime-veto: skipping reversal (%s %s) — %s", signal.get("strategy","?"), signal.get("side","?"), _rr)
                        signal_executed = True
                        break
                    _lfgv, _lfgr = _lfg_should_veto_entry(signal, market_time_et)
                    if _lfgv:
                        logging.info("🚫 lfg-veto: skipping reversal (%s %s) — %s", signal.get("strategy","?"), signal.get("side","?"), _lfgr)
                        signal_executed = True
                        break
                    success, reverse_state_count = await client.async_close_and_reverse(
                        signal,
                        current_price,
                        reverse_state_count,
                    )
                    if success or reverse_state_count == 0:
                        reset_opposite_reversal_state("standard execution path completed")

                    if success:
                        if old_trades and any(old_trade.get("side") != signal.get("side") for old_trade in old_trades):
                            close_order_details = getattr(client, "_last_close_order_details", None) or {}
                            shared_exit_price = _coerce_float(close_order_details.get("exit_price"), current_price)
                            for old_trade in old_trades:
                                if len(old_trades) == 1:
                                    close_metrics = _reconcile_live_trade_close(
                                        client,
                                        old_trade,
                                        current_time,
                                        fallback_exit_price=current_price,
                                        close_order_id=close_order_details.get("order_id"),
                                    )
                                else:
                                    close_metrics = _calculate_live_trade_close_metrics_from_price(
                                        old_trade,
                                        shared_exit_price,
                                        source="shared_reverse_close",
                                        exit_time=current_time,
                                        order_id=_coerce_int(close_order_details.get("order_id"), None),
                                    )
                                finalize_live_trade_close(
                                    old_trade,
                                    close_metrics,
                                    current_time,
                                    log_prefix="📊 Trade closed (reverse)",
                                )
                            active_trade = None
                            parallel_active_trades = []
                        order_details = getattr(client, "_last_order_details", None) or {}
                        entry_price = order_details.get("entry_price", current_price)
                        tp_dist = order_details.get("tp_points")
                        if tp_dist is None:
                            tp_dist = signal.get('tp_dist')
                        sl_dist = order_details.get("sl_points")
                        if sl_dist is None:
                            sl_dist = signal.get('sl_dist')
                        if tp_dist is None or sl_dist is None:
                            logging.error("Order details missing sl/tp after execution; using 0.0 for tracking")
                            tp_dist = tp_dist or 0.0
                            sl_dist = sl_dist or 0.0
                        size = order_details.get("size", signal.get('size', 5))
                        signal['tp_dist'] = tp_dist
                        signal['sl_dist'] = sl_dist
                        signal['size'] = size
                        signal['entry_price'] = entry_price
                        active_trade = None
                        parallel_active_trades = []
                        add_tracked_live_trade(
                            _build_live_active_trade(
                                signal,
                                order_details,
                                current_price,
                                current_time,
                                bar_count,
                                market_df=new_df,
                                stop_order_id=order_details.get(
                                    "stop_order_id",
                                    getattr(client, "_active_stop_order_id", None),
                                ),
                            )
                        )
                        persist_runtime_state(current_time, reason="standard_entry")
                        log_trade_factor_snapshot(
                            source="standard_execution",
                            signal_payload=signal,
                            order_details=order_details,
                            entry_price=float(entry_price),
                            market_df=new_df,
                            event_time=current_time,
                            market_price=float(current_price),
                            base_session_name=base_session,
                            current_session_name=current_session_name,
                            vol_regime_name=vol_regime_current,
                            trend_tier=trend_day_tier,
                            trend_dir=trend_day_dir,
                            strategy_eval=strategy_results,
                            regime_snapshot=regime_meta,
                            allowed_chop_bias=allowed_chop_side,
                            asia_viable_flag=asia_viable,
                        )

                    signal_executed = True
                    break
# 2c. LOOSE STRATEGIES (Queued)
                if not signal_executed:
                    if is_new_bar:
                        # Process Pending
                        for s_name in list(pending_loose_signals.keys()):
                            pending = pending_loose_signals[s_name]
                            pending['bar_count'] += 1
                            if pending['bar_count'] >= 1:
                                sig = pending['signal']
                                sig.setdefault('entry_mode', "loose")
                                sig.setdefault('strategy', s_name)
                                rm_ok, rm_reason = apply_regime_meta(sig, s_name)
                                if (not rm_ok) and regime_manifold_mode == "enforce":
                                    event_logger.log_filter_check(
                                        "RegimeManifold",
                                        sig.get("side", "ALL"),
                                        False,
                                        rm_reason,
                                        strategy=sig.get("strategy", s_name),
                                        metrics={
                                            "regime": str(regime_meta.get("regime")) if isinstance(regime_meta, dict) else "UNKNOWN",
                                            "R": round(float(regime_meta.get("R", 0.0) or 0.0), 4) if isinstance(regime_meta, dict) else 0.0,
                                        },
                                    )
                                    del pending_loose_signals[s_name]
                                    continue

                                # ==========================================
                                # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                                # ==========================================
                                # Apply the active session multipliers from CONFIG
                                # If Gemini is disabled or failed, these default to 1.0
                                sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                                tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                                sltp = _validate_signal_sltp(sig, s_name)
                                if sltp is None:
                                    del pending_loose_signals[s_name]
                                    continue
                                old_sl, old_tp = sltp

                                # Apply Multipliers
                                sig['sl_dist'] = old_sl * sl_mult
                                sig['tp_dist'] = old_tp * tp_mult

                                if sl_mult != 1.0 or tp_mult != 1.0:
                                    logging.info(f"🧠 GEMINI OPTIMIZED: {s_name} | SL: {old_sl:.2f}->{sig['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{sig['tp_dist']:.2f} (x{tp_mult})")
                                # ==========================================

                                strat_label = str(sig.get("strategy", s_name) or s_name)
                                gate_ok, gate_filter, gate_reason, gate_profile = evaluate_pre_signal_gate_fn(
                                    cfg=CONFIG,
                                    session_name=base_session,
                                    strategy_label=strat_label,
                                    side=sig.get("side"),
                                    asia_viable=asia_viable,
                                    asia_reason=asia_viable_reason,
                                    asia_trend_bias_side=asia_trend_bias_side,
                                    is_choppy=bool(is_choppy),
                                    chop_reason=chop_reason,
                                    allowed_chop_side=allowed_chop_side,
                                )
                                sig["gate_profile"] = gate_profile
                                if not gate_ok:
                                    event_logger.log_filter_check(
                                        gate_filter or "PreCandidateGate",
                                        sig.get("side", "ALL"),
                                        False,
                                        gate_reason or "Pre-candidate gate blocked",
                                        strategy=sig.get("strategy", s_name),
                                    )
                                    del pending_loose_signals[s_name]
                                    continue

                                if disable_strategy_filters:
                                    fixed_ok, fixed_details = True, {}
                                else:
                                    fixed_ok, fixed_details = apply_fixed_sltp(
                                        sig,
                                        new_df,
                                        current_price,
                                        ts=current_time,
                                        session=base_session,
                                        sl_dist_override=sig.get("sl_dist") if isinstance(sig, dict) else None,
                                    )
                                if not fixed_ok:
                                    reason = fixed_details.get("reason", "FixedSLTP blocked")
                                    logging.info(f"⛔ Signal ignored (LOOSE): {reason}")
                                    event_logger.log_filter_check("FixedSLTP", sig["side"], False, reason, strategy=sig.get("strategy", s_name))
                                    del pending_loose_signals[s_name]
                                    continue
                                if fixed_details:
                                    sig["sl_dist"] = fixed_details["sl_dist"]
                                    sig["tp_dist"] = fixed_details["tp_dist"]
                                    sig["sltp_bracket"] = fixed_details.get("bracket")
                                    sig["vol_regime"] = fixed_details.get("vol_regime", sig.get("vol_regime"))
                                    log_fixed_sltp(fixed_details, sig.get("strategy", s_name))

                                # Enforce HTF range fade directional restriction
                                if trend_day_tier > 0 and trend_day_dir:
                                    if (trend_day_dir == "down" and sig["side"] == "LONG") or (
                                        trend_day_dir == "up" and sig["side"] == "SHORT"
                                    ):
                                        event_logger.log_filter_check(
                                            f"TrendDayTier{trend_day_tier}",
                                            sig["side"],
                                            False,
                                            "Counter-trend",
                                            strategy=sig.get('strategy', s_name),
                                        )
                                        del pending_loose_signals[s_name]
                                        continue
                                    sig["trend_day_tier"] = trend_day_tier
                                    sig["trend_day_dir"] = trend_day_dir
                                if allowed_chop_side is not None and sig['side'] != allowed_chop_side:
                                    logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {sig['side']} vs Allowed {allowed_chop_side}")
                                    del pending_loose_signals[s_name]
                                    continue

                                # ==========================================
                                # LAYER 1: TARGET FEASIBILITY CHECK (Master Gate)
                                # ==========================================
                                # The market condition check (chop) already happened globally.
                                # Now check if the TARGET is realistic before wasting filter cycles.
                                is_feasible, feasibility_reason = chop_analyzer.check_target_feasibility(
                                    entry_price=current_price,
                                    side=sig['side'],
                                    tp_distance=sig.get('tp_dist'),
                                    df_1m=new_df
                                )
                                if (not is_feasible) and asia_calib_enabled and base_session == "ASIA":
                                    tf_cfg = asia_calib_cfg.get("target_feasibility", {}) or {}
                                    lookback = int(getattr(chop_analyzer, "LOOKBACK", 20) or 20)
                                    if asia_target_feasibility_override(
                                        new_df,
                                        sig['side'],
                                        sig.get('tp_dist'),
                                        asia_trend_bias_side,
                                        tf_cfg,
                                        lookback,
                                    ):
                                        is_feasible = True
                                        event_logger.log_filter_check(
                                            "ChopFeasibility",
                                            sig['side'],
                                            True,
                                            "ASIA trend override",
                                            strategy=sig.get('strategy', s_name),
                                        )
                                if not is_feasible:
                                    logging.info(f"⛔ Signal ignored (LOOSE): {feasibility_reason}")
                                    event_logger.log_filter_check("ChopFeasibility", sig['side'], False, feasibility_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ChopFeasibility", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # ==========================================
                                # LAYER 2: SIGNAL QUALITY FILTERS
                                # ==========================================
                                # Re-check filters
                                rej_blocked, rej_reason = rejection_filter.should_block_trade(sig['side'])
                                if rej_blocked:
                                    event_logger.log_rejection_block("RejectionFilter", sig['side'], rej_reason or "Rejection bias")
                                    del pending_loose_signals[s_name]; continue

                                # Directional Loss Blocker (3 consecutive losses blocks direction for 15 min)
                                dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(sig['side'], current_time)
                                if dir_blocked:
                                    event_logger.log_filter_check("DirectionalLossBlocker", sig['side'], False, dir_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("DirectionalLossBlocker", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # Impulse Filter (Prevent catching falling knife / fading rocket ship)
                                impulse_blocked, impulse_reason = impulse_filter.should_block_trade(sig['side'])
                                if impulse_blocked:
                                    event_logger.log_filter_check("ImpulseFilter", sig['side'], False, impulse_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ImpulseFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # HTF FVG (Memory Based) - CONTEXT AWARE
                                # Pass the strategy's target profit so we know how much room we need
                                tp_dist = sig.get('tp_dist', 15.0)

                                # === FIX: Relax FVG check if we are trading WITH the Range Fade ===
                                # If Chop says "Long Only" and we are going Long, we expect to break resistance.
                                # We reduce the effective TP distance passed to the filter, making it less strict.
                                effective_tp_dist = tp_dist
                                if allowed_chop_side is not None and sig['side'] == allowed_chop_side:
                                    effective_tp_dist = tp_dist * 0.5  # Require 50% less room
                                    logging.info(f"🔓 RELAXING FVG CHECK (Loose): Fading Range {sig['side']} (Req Room: {effective_tp_dist*0.4:.2f} pts)")

                                if htf_fvg_enabled_live:
                                    fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                                        sig['side'], current_price, None, None, tp_dist=effective_tp_dist
                                    )

                                    if fvg_blocked:
                                        logging.info(f"🚫 BLOCKED (HTF FVG): {fvg_reason}")
                                        event_logger.log_filter_check("HTF_FVG", sig['side'], False, fvg_reason, strategy=sig.get('strategy', s_name))
                                        del pending_loose_signals[s_name]; continue
                                    else:
                                        event_logger.log_filter_check("HTF_FVG", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # === [FIX 1] UPDATED BLOCKER CHECK ===
                                de3_rej_ok, de3_rej_detail = _de3_timeframe_rejection_bypass(
                                    sig,
                                    sig.get("side"),
                                    df_5m,
                                    df_15m,
                                )
                                struct_blocked, struct_reason = structure_blocker.should_block_trade(sig['side'], current_price)
                                if (
                                    struct_blocked
                                    and de3_rej_ok
                                    and "no volume/wick rejection" in str(struct_reason or "").lower()
                                ):
                                    struct_blocked = False
                                    struct_reason = f"DE3 bypass ({de3_rej_detail})"
                                    logging.info(
                                        "DE3 rejection bypass (loose): StructureBlocker (%s)",
                                        de3_rej_detail,
                                    )
                                if struct_blocked:
                                    logging.info(f"🚫 {struct_reason}")
                                    event_logger.log_filter_check("StructureBlocker", sig['side'], False, struct_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("StructureBlocker", sig['side'], True, strategy=sig.get('strategy', s_name))
                                # Regime Structure Blocker (EQH/EQL with regime tolerance)
                                regime_blocked, regime_reason = regime_blocker.should_block_trade(sig['side'], current_price)
                                if regime_blocked:
                                    logging.info(f"🚫 {regime_reason}")
                                    event_logger.log_filter_check("RegimeBlocker", sig['side'], False, regime_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("RegimeBlocker", sig['side'], True, strategy=sig.get('strategy', s_name))
                                # Penalty Box Blocker (Fixed 5.0pt tolerance + 3-bar decay)
                                penalty_source = penalty_blocker_asia if base_session == "ASIA" and penalty_blocker_asia is not None else penalty_blocker
                                if penalty_source is not None:
                                    penalty_blocked, penalty_reason = penalty_source.should_block_trade(sig['side'], current_price)
                                    if penalty_blocked:
                                        logging.info(f"🚫 {penalty_reason}")
                                        event_logger.log_filter_check("PenaltyBoxBlocker", sig['side'], False, penalty_reason, strategy=sig.get('strategy', s_name))
                                        del pending_loose_signals[s_name]; continue
                                    else:
                                        event_logger.log_filter_check("PenaltyBoxBlocker", sig['side'], True, strategy=sig.get('strategy', s_name))
                                mem_blocked, mem_reason = memory_sr.should_block_trade(sig['side'], current_price)
                                if mem_blocked:
                                    logging.info(f"🚫 {mem_reason}")
                                    event_logger.log_filter_check("MemorySR", sig['side'], False, mem_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("MemorySR", sig['side'], True, strategy=sig.get('strategy', s_name))
                                # =====================================

                                # Determine if this is a Range Fade setup (used for filter bypasses)
                                is_range_fade = (allowed_chop_side is not None and sig['side'] == allowed_chop_side)

                                # === DUAL-FILTER TREND CHECK ===
                                legacy_trend_blocked, legacy_trend_reason = legacy_filters.check_trend(new_df, sig['side'])
                                if legacy_trend_blocked and de3_rej_ok:
                                    legacy_trend_blocked = False
                                    legacy_trend_reason = f"DE3 bypass ({de3_rej_detail})"
                                    logging.info(
                                        "DE3 rejection bypass (loose): LegacyTrend (%s)",
                                        de3_rej_detail,
                                    )
                                upgraded_trend_blocked, upgraded_trend_reason = trend_filter.should_block_trade(new_df, sig['side'], is_range_fade=is_range_fade)

                                if legacy_trend_blocked != upgraded_trend_blocked:
                                    arb_result = filter_arbitrator.arbitrate(
                                        df=new_df, side=sig['side'],
                                        legacy_blocked=legacy_trend_blocked, legacy_reason=legacy_trend_reason or "",
                                        upgraded_blocked=upgraded_trend_blocked, upgraded_reason=upgraded_trend_reason or "",
                                        current_price=current_price,
                                        tp_dist=sig.get('tp_dist'), sl_dist=sig.get('sl_dist')
                                    )
                                    trend_blocked = not arb_result.allow_trade
                                    trend_reason = arb_result.reason
                                else:
                                    trend_blocked = upgraded_trend_blocked
                                    trend_reason = upgraded_trend_reason
                                    # Log when both agree (so we know dual-filter is running)
                                    if trend_blocked:
                                        logging.info(f"🛡️ DUAL-FILTER: Both BLOCK {sig['side']} | reason: {trend_reason}")
                                    else:
                                        logging.info(f"✅ DUAL-FILTER: Both ALLOW {sig['side']} trend check")

                                trend_state = ("Strong Bearish" if (trend_reason and "Bearish" in str(trend_reason))
                                               else ("Strong Bullish" if (trend_reason and "Bullish" in str(trend_reason))
                                                     else "NEUTRAL"))
                                vol_regime, _, _ = volatility_filter.get_regime(new_df)

                                chop_blocked, chop_reason = chop_filter.should_block_trade(
                                    sig['side'],
                                    rejection_filter.prev_day_pm_bias,
                                    current_price,
                                    trend_state=trend_state,
                                    vol_regime=vol_regime
                                )
                                if chop_blocked and asia_calib_enabled and base_session == "ASIA":
                                    chop_cfg = asia_calib_cfg.get("chop_filter", {}) or {}
                                    if asia_chop_override(
                                        chop_reason,
                                        sig['side'],
                                        asia_trend_bias_side,
                                        chop_cfg,
                                    ):
                                        chop_blocked = False
                                        event_logger.log_filter_check(
                                            "ChopFilter",
                                            sig['side'],
                                            True,
                                            "ASIA trend override",
                                            strategy=sig.get('strategy', s_name),
                                        )
                                if chop_blocked:
                                    event_logger.log_filter_check("ChopFilter", sig['side'], False, chop_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ChopFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                ext_blocked, ext_reason = extension_filter.should_block_trade(sig['side'])
                                soft_ext_reason = None
                                if ext_blocked and asia_soft_ext_enabled:
                                    soft_score = asia_soft_ext_base - asia_soft_ext_penalty
                                    if soft_score >= asia_soft_ext_threshold:
                                        ext_blocked = False
                                        soft_ext_reason = (
                                            f"ASIA soft extension score {soft_score:.2f} >= "
                                            f"{asia_soft_ext_threshold:.2f}"
                                        )
                                if ext_blocked:
                                    event_logger.log_filter_check("ExtensionFilter", sig['side'], False, ext_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check(
                                        "ExtensionFilter",
                                        sig['side'],
                                        True,
                                        soft_ext_reason,
                                        strategy=sig.get('strategy', s_name),
                                    )

                                # Trend Filter (already checked above with is_range_fade)
                                if trend_blocked:
                                    event_logger.log_filter_check("TrendFilter", sig['side'], False, trend_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("TrendFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # Volatility & Guardrail Check
                                # We pass the Gemini-modified params (sig['sl_dist']) into the filter.
                                # The filter applies Guardrails + Rounding.
                                should_trade, vol_adj = check_volatility(
                                    new_df,
                                    sig.get('sl_dist'),
                                    sig.get('tp_dist'),
                                    base_size=_signal_base_size(sig, 5),
                                )

                                if not should_trade:
                                    event_logger.log_filter_check("VolatilityFilter", sig['side'], False, "Volatility check failed", strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("VolatilityFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # === APPLY SANITIZED VALUES ===
                                # Always update to the rounded version (e.g. 4.52 -> 4.50)
                                # regardless of whether a 'regime' change happened.
                                sig['sl_dist'] = vol_adj['sl_dist']
                                sig['tp_dist'] = vol_adj['tp_dist']
                                sig['vol_regime'] = vol_adj.get('regime', 'UNKNOWN')
                                if not ml_vol_regime_ok(sig, base_session, sig['vol_regime'], asia_viable=asia_viable):
                                    event_logger.log_filter_check(
                                        "MLVolRegimeGuard",
                                        sig['side'],
                                        False,
                                        f"regime={sig['vol_regime']}",
                                        strategy=sig.get('strategy', s_name)
                                    )
                                    del pending_loose_signals[s_name]; continue

                                # Only apply SIZE adjustment if the regime explicitly demands it (Low Vol)
                                if vol_adj.get('adjustment_applied', False):
                                    sig['size'] = vol_adj['size']
                                    event_logger.log_trade_modified(
                                        "VolatilityAdjustment",
                                        sig.get('tp_dist'),
                                        vol_adj['tp_dist'],
                                        f"Volatility/Guardrail adjustment (Regime: {vol_adj['regime']})"
                                    )

                                sig["size"] = _apply_live_execution_size(
                                    sig,
                                    5,
                                    live_drawdown_state,
                                    tracked_live_trades(),
                                )
                                if sig["size"] <= 0:
                                    logging.info("Kalshi HARD VETO — loose trade blocked (size=0): %s", sig.get("strategy", s_name))
                                    del pending_loose_signals[s_name]
                                    continue
                                if _same_side_active_trade(active_trade, sig):
                                    if _allow_same_side_parallel_entry(active_trade, sig, tracked_live_trades()):
                                        reset_opposite_reversal_state("same-side loose coexist signal")
                                        block_reason = _live_entry_window_block_reason(current_time)
                                        if block_reason:
                                            add_strategy_slot(
                                                "rejected",
                                                sig.get("strategy", s_name),
                                                sig,
                                                fallback=s_name,
                                            )
                                            _log_live_entry_window_block(
                                                sig,
                                                sig.get("strategy", s_name),
                                                current_time,
                                            )
                                            del pending_loose_signals[s_name]
                                            continue

                                        logging.info(
                                            "✅ LOOSE EXEC (same-side coexist): %s alongside %s",
                                            s_name,
                                            active_trade.get("strategy"),
                                        )
                                        live_entry_day_block = _resolve_live_de3_entry_trade_day_extreme_admission_block(
                                            sig,
                                            entry_price=float(current_price),
                                            tp_dist=_coerce_float(sig.get("tp_dist"), 0.0),
                                            current_time=current_time,
                                            current_price=float(current_price),
                                            market_df=new_df,
                                        )
                                        if bool(live_entry_day_block.get("active", False)):
                                            sig["de3_entry_trade_day_extreme_admission_blocked"] = True
                                            sig["de3_entry_trade_day_extreme_admission_profile"] = str(
                                                live_entry_day_block.get("profile_name", "") or ""
                                            )
                                            add_strategy_slot(
                                                "rejected",
                                                sig.get("strategy", s_name),
                                                sig,
                                                fallback=s_name,
                                            )
                                            logging.info(
                                                "🚫 DE3 entry-day-extreme admission block: %s | profile=%s | beyond=%s | progress=%.2f",
                                                sig.get("sub_strategy") or sig.get("strategy"),
                                                str(live_entry_day_block.get("profile_name", "") or ""),
                                                bool(
                                                    live_entry_day_block.get(
                                                        "target_beyond_trade_day_extreme",
                                                        False,
                                                    )
                                                ),
                                                float(
                                                    _coerce_float(
                                                        live_entry_day_block.get("progress_pct"),
                                                        0.0,
                                                    )
                                                ),
                                            )
                                            continue
                                        adjusted_size = _apply_live_de3_entry_trade_day_extreme_size_adjustment(
                                            sig,
                                            current_size=_coerce_int(sig.get("size"), 1),
                                            entry_price=float(current_price),
                                            tp_dist=_coerce_float(sig.get("tp_dist"), 0.0),
                                            current_time=current_time,
                                            current_price=float(current_price),
                                            market_df=new_df,
                                        )
                                        if bool(
                                            sig.get(
                                                "de3_entry_trade_day_extreme_size_adjustment_applied",
                                                False,
                                            )
                                        ):
                                            logging.info(
                                                "🪶 DE3 entry-day-extreme size trim: %s | profile=%s | size %s -> %s | beyond=%s | progress=%.2f",
                                                sig.get("sub_strategy") or sig.get("strategy"),
                                                str(
                                                    sig.get(
                                                        "de3_entry_trade_day_extreme_size_adjustment_profile_name",
                                                        "",
                                                    )
                                                    or ""
                                                ),
                                                int(
                                                    _coerce_int(
                                                        sig.get(
                                                            "de3_entry_trade_day_extreme_size_adjustment_requested_size"
                                                        ),
                                                        adjusted_size,
                                                    )
                                                ),
                                                int(adjusted_size),
                                                bool(
                                                    sig.get(
                                                        "de3_entry_trade_day_extreme_size_adjustment_target_beyond",
                                                        False,
                                                    )
                                                ),
                                                float(
                                                    _coerce_float(
                                                        sig.get(
                                                            "de3_entry_trade_day_extreme_size_adjustment_progress_pct"
                                                        ),
                                                        0.0,
                                                    )
                                                ),
                                            )
                                        if not _apply_kalshi_trade_overlay_to_signal(
                                            sig,
                                            current_price,
                                            new_df,
                                            price_action_profile=kalshi_price_action_profile,
                                        ):
                                            add_strategy_slot(
                                                "rejected",
                                                sig.get("strategy", s_name),
                                                sig,
                                                fallback=s_name,
                                            )
                                            del pending_loose_signals[s_name]
                                            continue
                                        exec_strategy, exec_sub = get_log_strategy_info(s_name, sig)
                                        if exec_sub:
                                            exec_name = f"{exec_strategy} ({exec_sub})"
                                        else:
                                            exec_name = exec_strategy
                                        event_logger.log_strategy_execution(
                                            exec_name,
                                            "LOOSE",
                                            side=sig.get("side"),
                                            price=current_price,
                                        )
                                        if not _resolve_pct_overlay_snapshot(sig):
                                            add_strategy_slot(
                                                "rejected",
                                                sig.get("strategy", s_name),
                                                sig,
                                                fallback=s_name,
                                            )
                                            del pending_loose_signals[s_name]
                                            signal_executed = True
                                            break
                                        add_strategy_slot(
                                            "executed",
                                            sig.get("strategy", s_name),
                                            sig,
                                            fallback=s_name,
                                        )
                                        if circuit_breaker.is_tripped:
                                            logging.info("🚫 CB-entry-gate: skipping rescue entry (%s %s)",
                                                         sig.get("strategy","?"), sig.get("side","?"))
                                            continue
                                        from regime_classifier import should_veto_entry as _regime_veto
                                        _rv, _rr = _regime_veto()
                                        if _rv:
                                            logging.info("🚫 regime-veto: skipping rescue entry (%s %s) — %s",
                                                         sig.get("strategy","?"), sig.get("side","?"), _rr)
                                            continue
                                        _lfgv, _lfgr = _lfg_should_veto_entry(sig, market_time_et)
                                        if _lfgv:
                                            logging.info("🚫 lfg-veto: skipping rescue entry (%s %s) — %s",
                                                         sig.get("strategy","?"), sig.get("side","?"), _lfgr)
                                            continue
                                        order_response = await client.async_place_order(sig, current_price)
                                        if order_response is not None:
                                            order_details = getattr(client, "_last_order_details", None) or {}
                                            entry_price = order_details.get("entry_price", current_price)
                                            tp_dist = order_details.get("tp_points")
                                            if tp_dist is None:
                                                tp_dist = sig.get('tp_dist')
                                            sl_dist = order_details.get("sl_points")
                                            if sl_dist is None:
                                                sl_dist = sig.get('sl_dist')
                                            if tp_dist is None or sl_dist is None:
                                                logging.error("Order details missing sl/tp after same-side loose execution; using 0.0 for tracking")
                                                tp_dist = tp_dist or 0.0
                                                sl_dist = sl_dist or 0.0
                                            size = order_details.get("size", sig.get('size', 5))
                                            sig['tp_dist'] = tp_dist
                                            sig['sl_dist'] = sl_dist
                                            sig['size'] = size
                                            sig['entry_price'] = entry_price
                                            add_tracked_live_trade(
                                                _build_live_active_trade(
                                                    sig,
                                                    order_details,
                                                    current_price,
                                                    current_time,
                                                    bar_count,
                                                    market_df=new_df,
                                                    stop_order_id=order_details.get(
                                                        "stop_order_id",
                                                        getattr(client, "_active_stop_order_id", None),
                                                    ),
                                                )
                                            )
                                            persist_runtime_state(current_time, reason="loose_parallel_entry")
                                            log_trade_factor_snapshot(
                                                source="loose_parallel_execution",
                                                signal_payload=sig,
                                                order_details=order_details,
                                                entry_price=float(entry_price),
                                                market_df=new_df,
                                                event_time=current_time,
                                                market_price=float(current_price),
                                                base_session_name=base_session,
                                                current_session_name=current_session_name,
                                                vol_regime_name=vol_regime_current,
                                                trend_tier=trend_day_tier,
                                                trend_dir=trend_day_dir,
                                                strategy_eval=strategy_results,
                                                regime_snapshot=regime_meta,
                                                allowed_chop_bias=allowed_chop_side,
                                                asia_viable_flag=asia_viable,
                                            )
                                        del pending_loose_signals[s_name]
                                        signal_executed = True
                                        break
                                    reset_opposite_reversal_state("same-side loose signal ignored")
                                    logging.info(
                                        "Ignoring same-side loose signal while %s position is already active: %s",
                                        active_trade.get("side"),
                                        sig.get("strategy", s_name),
                                    )
                                    del pending_loose_signals[s_name]
                                    signal_executed = True
                                    break

                                block_reason = _live_entry_window_block_reason(current_time)
                                if block_reason:
                                    add_strategy_slot(
                                        "rejected",
                                        sig.get("strategy", s_name),
                                        sig,
                                        fallback=s_name,
                                    )
                                    _log_live_entry_window_block(
                                        sig,
                                        sig.get("strategy", s_name),
                                        current_time,
                                    )
                                    del pending_loose_signals[s_name]
                                    continue

                                if not _apply_kalshi_trade_overlay_to_signal(
                                    sig,
                                    current_price,
                                    new_df,
                                    price_action_profile=kalshi_price_action_profile,
                                ):
                                    add_strategy_slot(
                                        "rejected",
                                        sig.get("strategy", s_name),
                                        sig,
                                        fallback=s_name,
                                    )
                                    del pending_loose_signals[s_name]
                                    continue

                                logging.info(f"✅ LOOSE EXEC: {s_name}")
                                exec_strategy, exec_sub = get_log_strategy_info(s_name, sig)
                                if exec_sub:
                                    exec_name = f"{exec_strategy} ({exec_sub})"
                                else:
                                    exec_name = exec_strategy
                                event_logger.log_strategy_execution(
                                    exec_name,
                                    "LOOSE",
                                    side=sig.get("side"),
                                    price=current_price,
                                )

                                current_trades = tracked_live_trades()
                                old_trades = [dict(trade) for trade in current_trades]
                                if old_trades and not opposite_reversal_matches_active_trade_family(
                                    sig,
                                    current_trades,
                                ):
                                    reset_opposite_reversal_state(
                                        "loose opposite active-trade family mismatch"
                                    )
                                    log_opposite_reversal_active_trade_family_block(
                                        sig,
                                        current_trades,
                                        prefix="Holding position",
                                    )
                                    del pending_loose_signals[s_name]
                                    signal_executed = True
                                    break

                                if old_trades and same_strategy_opposite_reversal_blocked(
                                    sig,
                                    current_trades,
                                ):
                                    reset_opposite_reversal_state(
                                        "same-strategy loose opposite reversal blocked"
                                    )
                                    log_same_strategy_opposite_reversal_block(
                                        sig,
                                        current_trades,
                                        prefix="Holding position",
                                    )
                                    del pending_loose_signals[s_name]
                                    signal_executed = True
                                    break

                                reverse_state_count = 0
                                if old_trades:
                                    reverse_confirmed, reverse_state_count = note_opposite_reversal_signal(
                                        sig,
                                        bar_count,
                                    )
                                    if not reverse_confirmed:
                                        remaining = max(
                                            0,
                                            opposite_reversal_required - reverse_state_count,
                                        )
                                        logging.info(
                                            "Holding position: loose opposite confirmation %s/%s for %s "
                                            "(need %s more within %s bars)",
                                            reverse_state_count,
                                            opposite_reversal_required,
                                            sig.get("side"),
                                            remaining,
                                            opposite_reversal_window_bars,
                                        )
                                        del pending_loose_signals[s_name]
                                        signal_executed = True
                                        break

                                if not _resolve_pct_overlay_snapshot(sig):
                                    add_strategy_slot(
                                        "rejected",
                                        sig.get("strategy", s_name),
                                        sig,
                                        fallback=s_name,
                                    )
                                    del pending_loose_signals[s_name]
                                    signal_executed = True
                                    break

                                if circuit_breaker.is_tripped:
                                    logging.info("🚫 CB-entry-gate: skipping loose reversal (%s %s)", sig.get("strategy","?"), sig.get("side","?"))
                                    signal_executed = True
                                    break
                                from regime_classifier import should_veto_entry as _regime_veto
                                _rv, _rr = _regime_veto()
                                if _rv:
                                    logging.info("🚫 regime-veto: skipping loose reversal (%s %s) — %s", sig.get("strategy","?"), sig.get("side","?"), _rr)
                                    signal_executed = True
                                    break
                                _lfgv, _lfgr = _lfg_should_veto_entry(sig, market_time_et)
                                if _lfgv:
                                    logging.info("🚫 lfg-veto: skipping loose reversal (%s %s) — %s", sig.get("strategy","?"), sig.get("side","?"), _lfgr)
                                    signal_executed = True
                                    break
                                success, reverse_state_count = await client.async_close_and_reverse(
                                    sig,
                                    current_price,
                                    reverse_state_count,
                                )
                                if success or reverse_state_count == 0:
                                    reset_opposite_reversal_state("loose execution path completed")
                                if success:
                                    if old_trades and any(old_trade.get("side") != sig.get("side") for old_trade in old_trades):
                                        close_order_details = getattr(client, "_last_close_order_details", None) or {}
                                        shared_exit_price = _coerce_float(close_order_details.get("exit_price"), current_price)
                                        for old_trade in old_trades:
                                            if len(old_trades) == 1:
                                                close_metrics = _reconcile_live_trade_close(
                                                    client,
                                                    old_trade,
                                                    current_time,
                                                    fallback_exit_price=current_price,
                                                    close_order_id=close_order_details.get("order_id"),
                                                )
                                            else:
                                                close_metrics = _calculate_live_trade_close_metrics_from_price(
                                                    old_trade,
                                                    shared_exit_price,
                                                    source="shared_reverse_close",
                                                    exit_time=current_time,
                                                    order_id=_coerce_int(close_order_details.get("order_id"), None),
                                                )
                                            finalize_live_trade_close(
                                                old_trade,
                                                close_metrics,
                                                current_time,
                                                log_prefix="📊 Trade closed (reverse)",
                                            )
                                        active_trade = None
                                        parallel_active_trades = []
                                    add_strategy_slot(
                                        "executed",
                                        sig.get("strategy", s_name),
                                        sig,
                                        fallback=s_name,
                                    )
                                    order_details = getattr(client, "_last_order_details", None) or {}
                                    entry_price = order_details.get("entry_price", current_price)
                                    tp_dist = order_details.get("tp_points")
                                    if tp_dist is None:
                                        tp_dist = sig.get('tp_dist')
                                    sl_dist = order_details.get("sl_points")
                                    if sl_dist is None:
                                        sl_dist = sig.get('sl_dist')
                                    if tp_dist is None or sl_dist is None:
                                        logging.error("Order details missing sl/tp after execution; using 0.0 for tracking")
                                        tp_dist = tp_dist or 0.0
                                        sl_dist = sl_dist or 0.0
                                    size = order_details.get("size", sig.get('size', 5))
                                    sig['tp_dist'] = tp_dist
                                    sig['sl_dist'] = sl_dist
                                    sig['size'] = size
                                    sig['entry_price'] = entry_price
                                    active_trade = None
                                    parallel_active_trades = []
                                    add_tracked_live_trade(
                                        _build_live_active_trade(
                                            sig,
                                            order_details,
                                            current_price,
                                            current_time,
                                            bar_count,
                                            market_df=new_df,
                                            stop_order_id=order_details.get(
                                                "stop_order_id",
                                                getattr(client, "_active_stop_order_id", None),
                                            ),
                                        )
                                    )
                                    persist_runtime_state(current_time, reason="loose_entry")
                                    log_trade_factor_snapshot(
                                        source="loose_execution",
                                        signal_payload=sig,
                                        order_details=order_details,
                                        entry_price=float(entry_price),
                                        market_df=new_df,
                                        event_time=current_time,
                                        market_price=float(current_price),
                                        base_session_name=base_session,
                                        current_session_name=current_session_name,
                                        vol_regime_name=vol_regime_current,
                                        trend_tier=trend_day_tier,
                                        trend_dir=trend_day_dir,
                                        strategy_eval=strategy_results,
                                        regime_snapshot=regime_meta,
                                        allowed_chop_bias=allowed_chop_side,
                                        asia_viable_flag=asia_viable,
                                    )

                                del pending_loose_signals[s_name]
                                signal_executed = True
                                break
                        
                        # Check New Loose Signals
                        if not signal_executed:
                            for strat in loose_strategies:
                                try:
                                    signal = strat.on_bar(new_df)
                                    s_name = strat.__class__.__name__
                                    if signal and s_name not in pending_loose_signals:
                                        # ==========================================
                                        # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                                        # ==========================================
                                        # Apply the active session multipliers from CONFIG
                                        # If Gemini is disabled or failed, these default to 1.0
                                        sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                                        tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                                        sltp = _validate_signal_sltp(signal, s_name)
                                        if sltp is None:
                                            continue
                                        old_sl, old_tp = sltp

                                        # Apply Multipliers
                                        signal['sl_dist'] = old_sl * sl_mult
                                        signal['tp_dist'] = old_tp * tp_mult

                                        if sl_mult != 1.0 or tp_mult != 1.0:
                                            logging.info(f"🧠 GEMINI OPTIMIZED: {s_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")
                                        # ==========================================

                                        signal.setdefault('strategy', s_name)
                                        signal['entry_mode'] = "loose"
                                        strat_label = str(signal.get("strategy", s_name) or s_name)
                                        gate_ok, gate_filter, gate_reason, gate_profile = evaluate_pre_signal_gate_fn(
                                            cfg=CONFIG,
                                            session_name=base_session,
                                            strategy_label=strat_label,
                                            side=signal.get("side"),
                                            asia_viable=asia_viable,
                                            asia_reason=asia_viable_reason,
                                            asia_trend_bias_side=asia_trend_bias_side,
                                            is_choppy=bool(is_choppy),
                                            chop_reason=chop_reason,
                                            allowed_chop_side=allowed_chop_side,
                                        )
                                        signal["gate_profile"] = gate_profile
                                        if not gate_ok:
                                            event_logger.log_filter_check(
                                                gate_filter or "PreCandidateGate",
                                                signal.get("side", "ALL"),
                                                False,
                                                gate_reason or "Pre-candidate gate blocked",
                                                strategy=signal.get("strategy", s_name),
                                            )
                                            continue
                                        rm_ok, rm_reason = apply_regime_meta(signal, s_name)
                                        if (not rm_ok) and regime_manifold_mode == "enforce":
                                            event_logger.log_filter_check(
                                                "RegimeManifold",
                                                signal.get("side", "ALL"),
                                                False,
                                                rm_reason,
                                                strategy=signal.get("strategy", s_name),
                                                metrics={
                                                    "regime": str(regime_meta.get("regime")) if isinstance(regime_meta, dict) else "UNKNOWN",
                                                    "R": round(float(regime_meta.get("R", 0.0) or 0.0), 4) if isinstance(regime_meta, dict) else 0.0,
                                                },
                                            )
                                            continue

                                        # Enforce HTF range fade directional restriction
                                        if trend_day_tier > 0 and trend_day_dir:
                                            if (trend_day_dir == "down" and signal["side"] == "LONG") or (
                                                trend_day_dir == "up" and signal["side"] == "SHORT"
                                            ):
                                                event_logger.log_filter_check(
                                                    f"TrendDayTier{trend_day_tier}",
                                                    signal["side"],
                                                    False,
                                                    "Counter-trend",
                                                    strategy=signal.get('strategy', s_name),
                                                )
                                                continue
                                            signal["trend_day_tier"] = trend_day_tier
                                            signal["trend_day_dir"] = trend_day_dir
                                        if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                                            logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                                            continue

                                        # Enhanced event logging: Strategy signal generated
                                        log_strategy, log_sub = get_log_strategy_info(
                                            signal.get('strategy', s_name),
                                            signal
                                        )
                                        log_info = {"execution_type": "LOOSE"}
                                        if log_sub:
                                            log_info["sub_strategy"] = log_sub
                                        for extra_key in (
                                            "combo_key",
                                            "rule_id",
                                            "entry_mode",
                                            "vol_regime",
                                            "early_exit_enabled",
                                            "gate_prob",
                                            "gate_threshold",
                                        ):
                                            extra_value = signal.get(extra_key)
                                            if extra_value is not None:
                                                log_info[extra_key] = extra_value
                                        event_logger.log_strategy_signal(
                                            strategy_name=log_strategy,
                                            side=signal['side'],
                                            tp_dist=signal.get('tp_dist'),
                                            sl_dist=signal.get('sl_dist'),
                                            price=current_price,
                                            additional_info=log_info
                                        )

                                        rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                                        if rej_blocked:
                                            event_logger.log_rejection_block("RejectionFilter", signal['side'], rej_reason or "Rejection bias")
                                            continue

                                        # Directional Loss Blocker (3 consecutive losses blocks direction for 15 min)
                                        dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(signal['side'], current_time)
                                        if dir_blocked:
                                            event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], False, dir_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("DirectionalLossBlocker", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        tp_dist = signal.get('tp_dist', 15.0)

                                        effective_tp_dist = tp_dist
                                        if allowed_chop_side is not None and signal['side'] == allowed_chop_side:
                                            effective_tp_dist = tp_dist * 0.5
                                            logging.info(f"🔓 RELAXING FVG CHECK (Loose): Fading Range {signal['side']} (Req Room: {effective_tp_dist*0.4:.2f} pts)")

                                        if htf_fvg_enabled_live:
                                            fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                                                signal['side'], current_price, None, None, tp_dist=effective_tp_dist
                                            )
                                            if fvg_blocked:
                                                event_logger.log_filter_check("HTF_FVG", signal['side'], False, fvg_reason, strategy=signal.get('strategy', s_name))
                                                continue
                                            else:
                                                event_logger.log_filter_check("HTF_FVG", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # === [FIX 2] UPDATED BLOCKER CHECK ===
                                        de3_rej_ok, de3_rej_detail = _de3_timeframe_rejection_bypass(
                                            signal,
                                            signal.get("side"),
                                            df_5m,
                                            df_15m,
                                        )
                                        struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
                                        if (
                                            struct_blocked
                                            and de3_rej_ok
                                            and "no volume/wick rejection" in str(struct_reason or "").lower()
                                        ):
                                            struct_blocked = False
                                            struct_reason = f"DE3 bypass ({de3_rej_detail})"
                                            logging.info(
                                                "DE3 rejection bypass (loose): StructureBlocker (%s)",
                                                de3_rej_detail,
                                            )
                                        if struct_blocked:
                                            event_logger.log_filter_check("StructureBlocker", signal['side'], False, struct_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("StructureBlocker", signal['side'], True, strategy=signal.get('strategy', s_name))
                                        # Regime Structure Blocker (EQH/EQL with regime tolerance)
                                        regime_blocked, regime_reason = regime_blocker.should_block_trade(signal['side'], current_price)
                                        if regime_blocked:
                                            event_logger.log_filter_check("RegimeBlocker", signal['side'], False, regime_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("RegimeBlocker", signal['side'], True, strategy=signal.get('strategy', s_name))
                                        # Penalty Box Blocker (Fixed 5.0pt tolerance + 3-bar decay)
                                        penalty_source = penalty_blocker_asia if base_session == "ASIA" and penalty_blocker_asia is not None else penalty_blocker
                                        if penalty_source is not None:
                                            penalty_blocked, penalty_reason = penalty_source.should_block_trade(signal['side'], current_price)
                                            if penalty_blocked:
                                                event_logger.log_filter_check("PenaltyBoxBlocker", signal['side'], False, penalty_reason, strategy=signal.get('strategy', s_name))
                                                continue
                                            else:
                                                event_logger.log_filter_check("PenaltyBoxBlocker", signal['side'], True, strategy=signal.get('strategy', s_name))
                                        mem_blocked, mem_reason = memory_sr.should_block_trade(signal['side'], current_price)
                                        if mem_blocked:
                                            event_logger.log_filter_check("MemorySR", signal['side'], False, mem_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("MemorySR", signal['side'], True, strategy=signal.get('strategy', s_name))
                                        # =====================================

                                        # Determine if this is a Range Fade setup (used for filter bypasses)
                                        is_range_fade = (allowed_chop_side is not None and signal['side'] == allowed_chop_side)

                                        # === DUAL-FILTER TREND CHECK ===
                                        legacy_trend_blocked, legacy_trend_reason = legacy_filters.check_trend(new_df, signal['side'])
                                        if legacy_trend_blocked and de3_rej_ok:
                                            legacy_trend_blocked = False
                                            legacy_trend_reason = f"DE3 bypass ({de3_rej_detail})"
                                            logging.info(
                                                "DE3 rejection bypass (loose): LegacyTrend (%s)",
                                                de3_rej_detail,
                                            )
                                        upgraded_trend_blocked, upgraded_trend_reason = trend_filter.should_block_trade(new_df, signal['side'], is_range_fade=is_range_fade)

                                        if legacy_trend_blocked != upgraded_trend_blocked:
                                            arb_result = filter_arbitrator.arbitrate(
                                                df=new_df, side=signal['side'],
                                                legacy_blocked=legacy_trend_blocked, legacy_reason=legacy_trend_reason or "",
                                                upgraded_blocked=upgraded_trend_blocked, upgraded_reason=upgraded_trend_reason or "",
                                                current_price=current_price,
                                                tp_dist=signal.get('tp_dist'), sl_dist=signal.get('sl_dist')
                                            )
                                            trend_blocked = not arb_result.allow_trade
                                            trend_reason = arb_result.reason
                                        else:
                                            trend_blocked = upgraded_trend_blocked
                                            trend_reason = upgraded_trend_reason
                                            # Log when both agree (so we know dual-filter is running)
                                            if trend_blocked:
                                                logging.info(f"🛡️ DUAL-FILTER: Both BLOCK {signal['side']} | reason: {trend_reason}")
                                            else:
                                                logging.info(f"✅ DUAL-FILTER: Both ALLOW {signal['side']} trend check")

                                        trend_state = ("Strong Bearish" if (trend_reason and "Bearish" in str(trend_reason))
                                                       else ("Strong Bullish" if (trend_reason and "Bullish" in str(trend_reason))
                                                             else "NEUTRAL"))
                                        vol_regime, _, _ = volatility_filter.get_regime(new_df)

                                        chop_blocked, chop_reason = chop_filter.should_block_trade(
                                            signal['side'],
                                            rejection_filter.prev_day_pm_bias,
                                            current_price,
                                            trend_state=trend_state,
                                            vol_regime=vol_regime
                                        )
                                        if chop_blocked and asia_calib_enabled and base_session == "ASIA":
                                            chop_cfg = asia_calib_cfg.get("chop_filter", {}) or {}
                                            if asia_chop_override(
                                                chop_reason,
                                                signal['side'],
                                                asia_trend_bias_side,
                                                chop_cfg,
                                            ):
                                                chop_blocked = False
                                                event_logger.log_filter_check(
                                                    "ChopFilter",
                                                    signal['side'],
                                                    True,
                                                    "ASIA trend override",
                                                    strategy=signal.get('strategy', s_name),
                                                )
                                        if chop_blocked:
                                            event_logger.log_filter_check("ChopFilter", signal['side'], False, chop_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("ChopFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                                        soft_ext_reason = None
                                        if ext_blocked and asia_soft_ext_enabled:
                                            soft_score = asia_soft_ext_base - asia_soft_ext_penalty
                                            if soft_score >= asia_soft_ext_threshold:
                                                ext_blocked = False
                                                soft_ext_reason = (
                                                    f"ASIA soft extension score {soft_score:.2f} >= "
                                                    f"{asia_soft_ext_threshold:.2f}"
                                                )
                                        if ext_blocked:
                                            event_logger.log_filter_check("ExtensionFilter", signal['side'], False, ext_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check(
                                                "ExtensionFilter",
                                                signal['side'],
                                                True,
                                                soft_ext_reason,
                                                strategy=signal.get('strategy', s_name),
                                            )

                                        # Trend Filter (already checked above with is_range_fade)
                                        if trend_blocked:
                                            event_logger.log_filter_check("TrendFilter", signal['side'], False, trend_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("TrendFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # Volatility & Guardrail Check
                                        # We pass the Gemini-modified params (signal['sl_dist']) into the filter.
                                        # The filter applies Guardrails + Rounding.
                                        should_trade, vol_adj = check_volatility(
                                            new_df,
                                            signal.get('sl_dist'),
                                            signal.get('tp_dist'),
                                            base_size=_signal_base_size(signal, 5),
                                        )

                                        if not should_trade:
                                            event_logger.log_filter_check("VolatilityFilter", signal['side'], False, "Volatility check failed", strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("VolatilityFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # === APPLY SANITIZED VALUES ===
                                        # Always update to the rounded version (e.g. 4.52 -> 4.50)
                                        # regardless of whether a 'regime' change happened.
                                        signal['sl_dist'] = vol_adj['sl_dist']
                                        signal['tp_dist'] = vol_adj['tp_dist']
                                        signal['vol_regime'] = vol_adj.get('regime', 'UNKNOWN')
                                        if not ml_vol_regime_ok(signal, base_session, signal['vol_regime'], asia_viable=asia_viable):
                                            event_logger.log_filter_check(
                                                "MLVolRegimeGuard",
                                                signal['side'],
                                                False,
                                                f"regime={signal['vol_regime']}",
                                                strategy=signal.get('strategy', s_name)
                                            )
                                            continue

                                        # Only apply SIZE adjustment if the regime explicitly demands it (Low Vol)
                                        if vol_adj.get('adjustment_applied', False):
                                            signal['size'] = vol_adj['size']
                                            event_logger.log_trade_modified(
                                                "VolatilityAdjustment",
                                                signal.get('tp_dist'),
                                                vol_adj['tp_dist'],
                                                f"Volatility/Guardrail adjustment (Regime: {vol_adj['regime']})"
                                            )

                                        # Log as QUEUED for UI visibility
                                        log_strategy, log_sub = get_log_strategy_info(
                                            signal.get('strategy', s_name),
                                            signal
                                        )
                                        log_info = {"status": "QUEUED", "priority": "LOOSE"}
                                        if log_sub:
                                            log_info["sub_strategy"] = log_sub
                                        event_logger.log_strategy_signal(
                                            strategy_name=log_strategy,
                                            side=signal['side'],
                                            tp_dist=signal.get('tp_dist', 0),
                                            sl_dist=signal.get('sl_dist', 0),
                                            price=current_price,
                                            additional_info=log_info
                                        )
                                        add_strategy_slot(
                                            "checked",
                                            signal.get("strategy", s_name),
                                            signal,
                                            fallback=s_name,
                                        )
                                        logging.info(f"🕐 Queuing {s_name} signal")
                                        _signal_birth_hook(signal)
                                        _attach_pct_overlay_snapshot(signal)
                                        pending_loose_signals[s_name] = {'signal': signal, 'bar_count': 0}
                                except Exception as e:
                                    logging.exception("Error in %s", s_name)

            if is_new_bar:
                now_save = time.time()
                if now_save - last_state_save >= 30:
                    try:
                        save_bot_state(build_persisted_state(current_time), STATE_PATH)
                        last_state_save = now_save
                    except Exception as e:
                        logging.warning(f"State save failed: {e}")

            await asyncio.sleep(2.0)  # Slower polling to avoid Topstep rate limits

        except KeyboardInterrupt:
            print("\nBot Stopped by User.")
            try:
                now_et = datetime.datetime.now(NY_TZ)
                save_bot_state(build_persisted_state(now_et), STATE_PATH)
            except Exception as e:
                logging.warning(f"State save on shutdown failed: {e}")
            try:
                if sentiment_service is not None:
                    sentiment_service.stop()
            except Exception:
                pass
            try:
                await client.stop_user_stream()
            except Exception:
                pass
            break
        except Exception as e:
            logging.error(f"Main Loop Error: {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    # Run the async bot with asyncio
    asyncio.run(run_bot())
