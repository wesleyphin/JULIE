import requests
import pandas as pd
import numpy as np
import datetime
import json
from datetime import date
import time
import logging
from zoneinfo import ZoneInfo
from datetime import timezone as dt_timezone
import uuid
from typing import Dict, Optional, List
import random
import asyncio
from pathlib import Path

from config import CONFIG, refresh_target_symbol, determine_current_contract_symbol
from dynamic_sltp_params import dynamic_sltp_engine, get_sltp
from volatility_filter import volatility_filter, check_volatility, VolRegime
from regime_strategy import RegimeAdaptiveStrategy
from htf_fvg_filter import HTFFVGFilter
from rejection_filter import RejectionFilter
from chop_filter import ChopFilter
from extension_filter import ExtensionFilter
from trend_filter import TrendFilter
from dynamic_structure_blocker import DynamicStructureBlocker, RegimeStructureBlocker, PenaltyBoxBlocker
from bank_level_quarter_filter import BankLevelQuarterFilter
from memory_sr_filter import MemorySRFilter
from orb_strategy import OrbStrategy
from intraday_dip_strategy import IntradayDipStrategy
from confluence_strategy import ConfluenceStrategy
from smt_strategy import SMTStrategy
from dynamic_chop import DynamicChopAnalyzer
from ict_model_strategy import ICTModelStrategy
from ml_physics_strategy import MLPhysicsStrategy
from dynamic_engine_strategy import DynamicEngineStrategy
from event_logger import event_logger
from circuit_breaker import CircuitBreaker
from news_filter import NewsFilter
from directional_loss_blocker import DirectionalLossBlocker
from impulse_filter import ImpulseFilter
from client import ProjectXClient
from risk_engine import OptimizedTPEngine
from gemini_optimizer import GeminiSessionOptimizer
import param_scaler
from bot_state import STATE_PATH, STATE_VERSION, load_bot_state, save_bot_state, trading_day_start, parse_dt
# --- NEW IMPORTS ---
from vixmeanreversion import VIXReversionStrategy
from yahoo_vix_client import YahooVIXClient
from legacy_filters import LegacyFilterSystem
from filter_arbitrator import FilterArbitrator
# --- NEW IMPORTS ---
from continuation_strategy import FractalSweepStrategy, STRATEGY_CONFIGS

# --- ASYNCIO IMPORTS ---
from async_market_stream import AsyncMarketDataManager
from async_tasks import heartbeat_task, position_sync_task, htf_structure_task

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


def parse_continuation_key(strategy_name: Optional[str]) -> Optional[str]:
    if not strategy_name:
        return None
    name = str(strategy_name)
    if name.startswith("Continuation_"):
        return name.split("Continuation_", 1)[1]
    return None


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
        key = parse_continuation_key(signal.get("strategy"))
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("topstep_live_bot.log"), logging.StreamHandler()],
    force=True  # Override any pre-existing logging config
)

NY_TZ = ZoneInfo('America/New_York')
TREND_DAY_ENABLED = True
ATR_BASELINE_WINDOW = 390
ATR_EXP_T1 = 1.4
ATR_EXP_T2 = 1.6
VWAP_SIGMA_T1 = 1.5
VWAP_NO_RECLAIM_BARS_T1 = 20
VWAP_NO_RECLAIM_BARS_T2 = 20
VWAP_RECLAIM_SIGMA = 0.5
VWAP_RECLAIM_CONSECUTIVE_BARS = 15
TREND_DAY_STICKY_RECLAIM_BARS = 30
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
TREND_DAY_T1_REQUIRE_CONFIRMATION = False
ALT_PRE_TIER1_VWAP_SIGMA = 2.0
TREND_DAY_SMA9_REVERSAL_BARS = 4
TREND_DAY_SMA9_MIN_SLOPE = 0.2
TREND_DAY_ATR_CONTRACTION = 1.1

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

# ==========================================
# 2d. HTF FVG REJECTION
# ==========================================
try:
    from htf_fvg_filter import HTFFVGFilter
    logging.info("✅ HTFFVGFilter module loaded")
except ImportError as e:
    logging.error(f"❌ Failed to import htf_fvg_filter.py: {e}")
    # Dummy class to prevent crash if file missing
    class HTFFVGFilter:
        def check_signal_blocked(self, *args): return False, None



# OptimizedTPEngine moved to risk_engine.py



# ProjectXClient moved to client.py

class ContinuationRescueManager:
    """
    Manages the FractalSweepStrategy (Continuation) lookups.
    Acts as a 'Second Opinion' when trades are blocked by filters.
    """
    def __init__(self):
        self.configs = STRATEGY_CONFIGS
        self.strategy_instances = {}
        # 1. TIMEZONE FIX: Strategies operate on NY Time
        self.ny_tz = ZoneInfo('America/New_York')

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
                self.strategy_instances[candidate_key] = FractalSweepStrategy(candidate_key)
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

    # Secondary client for MNQ data (SMT divergence inputs)
    mnq_target_symbol = determine_current_contract_symbol(
        "MNQ", tz_name=CONFIG.get("TIMEZONE", "US/Eastern")
    )
    mnq_client = ProjectXClient(contract_root="MNQ", target_symbol=mnq_target_symbol)

    # --- UPDATED: VIX Client (Using Yahoo Finance) ---
    # We use ^VIX (The Index) as it is the standard for mean reversion
    # and free via Yahoo, whereas Topstep Rithmic usually lacks CBOE data.
    logging.info("Initializing Virtual VIX Client (Yahoo Finance)...")
    vix_client = YahooVIXClient(target_symbol="^VIX")

    try:
        mnq_client.login()
        mnq_client.account_id = client.account_id or mnq_client.fetch_accounts()
        mnq_client.fetch_contracts()

        # Login VIX client (Virtual)
        vix_client.login()
        # No account ID needed for Yahoo, but we call methods for consistency
        vix_client.fetch_contracts()
    except Exception as e:
        logging.error(f"❌ Failed to initialize secondary clients: {e}")
        return

    # Initialize all strategies

    # Dynamic chop analyzer (tiered thresholds with LTF breakout override)
    chop_analyzer = DynamicChopAnalyzer(client)
    chop_analyzer.calibrate()  # Removed session_name argument
    last_chop_calibration = time.time()

    # --- NEW: Initialize VIX Strategy ---
    vix_strategy = VIXReversionStrategy()

    # HIGH PRIORITY - Execute immediately on signal
    # CHANGED: Dynamic Engine stays here. VIX added. Intraday Dip removed.
    fast_strategies = [
        RegimeAdaptiveStrategy(),
        vix_strategy,          # Promoted to Fast
    ]
    ENABLE_DYNAMIC_ENGINE_1 = True
    ALLOW_DYNAMIC_ENGINE_SOLO = False
    if ENABLE_DYNAMIC_ENGINE_1:
        dynamic_engine_strat = DynamicEngineStrategy()
        fast_strategies.append(dynamic_engine_strat)  # Kept in Fast (Not Demoted)

    # STANDARD PRIORITY - Normal execution
    ml_strategy = MLPhysicsStrategy()
    smt_strategy = SMTStrategy()

    standard_strategies = [
        IntradayDipStrategy(), # DEMOTED to Standard
        ConfluenceStrategy(),
        smt_strategy,
    ]
    
    # Only add ML strategy if at least one model loaded successfully
    if ml_strategy.model_loaded:
        standard_strategies.append(ml_strategy)
    else:
        print(f"⚠️ MLPhysicsStrategy disabled - no session model files found")
    
    # LOW PRIORITY / LOOSE EXECUTION - Wait for next bar
    loose_strategies = [
        OrbStrategy(),
        ICTModelStrategy(),
    ]
    
    # Initialize filters
    rejection_filter = RejectionFilter()
    bank_filter = BankLevelQuarterFilter()
    chop_filter = ChopFilter(lookback=20)
    extension_filter = ExtensionFilter()
    # 4-Tier Trend Filter (merged with Impulse logic)
    # Tier 1: Volume-supported impulse, Tier 2: Standard breakout, Tier 3: Extreme capitulation
    # Tier 4: Macro trend (50/200 EMA alignment) - bypassed by Range Fade logic
    trend_filter = TrendFilter()
    htf_fvg_filter = HTFFVGFilter() # Now uses Memory-Based Class
    structure_blocker = DynamicStructureBlocker(lookback=50)  # Macro trend + fade detection
    regime_blocker = RegimeStructureBlocker(lookback=20)      # Regime-based EQH/EQL tolerance
    penalty_blocker = PenaltyBoxBlocker(lookback=50, tolerance=5.0, penalty_bars=3)  # Fixed 5pt + 3-bar decay
    memory_sr = MemorySRFilter(lookback_bars=300, zone_width=2.0, touch_threshold=2)
    news_filter = NewsFilter()
    circuit_breaker = CircuitBreaker(max_daily_loss=600, max_consecutive_losses=7)
    directional_loss_blocker = DirectionalLossBlocker(consecutive_loss_limit=3, block_minutes=15)
    impulse_filter = ImpulseFilter(lookback=20, impulse_multiplier=2.5)

    # === DUAL-FILTER SYSTEM ===
    # Legacy (Dec 17th) filters for comparison + Arbitrator for override decisions
    legacy_filters = LegacyFilterSystem()
    filter_arbitrator = FilterArbitrator(confidence_threshold=0.6)

    # Initialize Gemini Session Optimizer
    optimizer = GeminiSessionOptimizer()

    # Initialize Rescue Manager
    continuation_manager = ContinuationRescueManager()
    continuation_guard = CONFIG.get("CONTINUATION_GUARD", {}) or {}
    continuation_guard_enabled = bool(continuation_guard.get("enabled", False))
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

    last_processed_session = None
    last_processed_quarter = None  # Track quarter for quarterly optimization

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

    bar_logger = CsvBarAppender("ml_mes_et.csv", CONFIG.get("TARGET_SYMBOL"), NY_TZ)

    # === LAUNCH INDEPENDENT ASYNC TASKS ===
    # These tasks run independently and cannot be blocked by strategy calculations
    heartbeat = asyncio.create_task(heartbeat_task(client, interval=60))
    position_sync = asyncio.create_task(position_sync_task(client, interval=30))

    # NEW: Background HTF Updater
    # This keeps your FVG memory fresh without pausing the bot
    htf_updater = asyncio.create_task(htf_structure_task(client, htf_fvg_filter, interval=60))

    # === TRACKING VARIABLES ===
    # Position sync now handled by independent async task - removed manual tracking
    
    # Track pending signals for delayed execution
    pending_loose_signals = {}
    last_processed_bar = None
    opposite_signal_count = 0
    pending_impulse_rescues = []

    # Early Exit Tracking
    active_trade = None
    bar_count = 0
    flat_position_streak = 0

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
    tier1_down_until = None
    tier1_up_until = None
    tier1_seen = False
    sticky_trend_dir = None
    sticky_reclaim_count = 0
    sticky_opposite_count = 0
    last_trend_session = None
    persisted_state = load_bot_state(STATE_PATH)
    last_state_save = 0.0
    state_restored = False

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
        nonlocal mom_rescue_date, mom_rescue_scores
        nonlocal hostile_day_active, hostile_day_reason, hostile_day_date, hostile_engine_stats
        nonlocal trend_day_tier, trend_day_dir, impulse_day, impulse_active, impulse_dir
        nonlocal impulse_start_price, impulse_extreme, pullback_extreme, max_retracement, bars_since_impulse
        nonlocal last_trend_day_tier, last_trend_day_dir, tier1_down_until, tier1_up_until, tier1_seen
        nonlocal sticky_trend_dir, sticky_reclaim_count, sticky_opposite_count, last_trend_session

        if state_restored or not _state_is_fresh(current_time):
            return

        extension_filter.load_state(persisted_state.get("extension_filter"))
        chop_filter.load_state(persisted_state.get("chop_filter"))
        directional_loss_blocker.load_state(persisted_state.get("directional_loss_blocker"))
        circuit_breaker.load_state(persisted_state.get("circuit_breaker"))
        penalty_blocker.load_state(persisted_state.get("penalty_box_blocker"))
        rejection_filter.load_state(persisted_state.get("rejection_filter"))
        bank_filter.load_state(persisted_state.get("bank_filter"))

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

        state_restored = True
        logging.info("✅ Bot state restored from disk")

    def build_persisted_state(current_time: datetime.datetime) -> dict:
        current_time = current_time.astimezone(NY_TZ)
        return {
            "version": STATE_VERSION,
            "timestamp": current_time.isoformat(),
            "trading_day_start": trading_day_start(current_time).isoformat(),
            "last_bar_ts": current_time.isoformat(),
            "extension_filter": extension_filter.get_state(),
            "chop_filter": chop_filter.get_state(),
            "directional_loss_blocker": directional_loss_blocker.get_state(),
            "circuit_breaker": circuit_breaker.get_state(),
            "penalty_box_blocker": penalty_blocker.get_state(),
            "rejection_filter": rejection_filter.get_state(),
            "bank_filter": bank_filter.get_state(),
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
        }

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
        if strategy_name == "DynamicEngine" or strategy_name == "DynamicEngineStrategy":
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
        if strategy_name == "DynamicEngine":
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

    event_logger.log_system_event("STARTUP", "⏳ Startup: Fetching 20,000 bar history (MNQ)...", {"status": "IN_PROGRESS"})
    logging.info("⏳ Startup: Fetching full 20,000 bar history (MNQ)...")
    master_mnq_df = await mnq_client.async_get_market_data(lookback_minutes=20000, force_fetch=True)
    event_logger.log_system_event("STARTUP", f"✅ History Received: {len(master_mnq_df)} bars loaded (MNQ).", {"status": "COMPLETE"})

    if master_df.empty:
        logging.warning("⚠️ Startup fetch returned empty data (MES). Bot will attempt to build history in loop.")
        master_df = pd.DataFrame()

    if not master_df.empty:
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
    # --- END CALIBRATION ---

    if master_mnq_df.empty:
        logging.warning("⚠️ Startup fetch returned empty data (MNQ). Bot will attempt to build history in loop.")
        master_mnq_df = pd.DataFrame()

    # --- NEW: Initialize VIX master dataframe ---
    master_vix_df = pd.DataFrame()
    vix_fetch_toggle = True

    # One-time backfill flag
    data_backfilled = False

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
            cb_blocked, cb_reason = circuit_breaker.should_block_trade()
            if cb_blocked:
                logging.info(f"🚫 Circuit Breaker Block: {cb_reason}")
                await asyncio.sleep(60)
                continue

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
                    logging.warning("🛑 TrendDay reset due to news blackout")
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

            # 1. Determine Broad Parent Session (For Data Slicing & Config Lookup)
            # (Keeps your original logic intact for data fetching)
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

            # 2. Determine Micro-Session (The "Trump Era" Logic)
            # (Adds granularity for the Optimizer, defaults to base_session)
            current_session_name = base_session

            if base_session == "NY_AM":
                # "Safe Window" is 09:30-10:15 (Standard NY_AM)
                # "Lunchtime Death" starts 10:30
                if hour == 10 and minute >= 30:
                    current_session_name = "NY_LUNCH"
                elif hour == 11:
                    current_session_name = "NY_LUNCH"

            elif base_session == "NY_PM":
                # "Lunchtime Death" ends 12:30
                if hour == 12 and minute < 30:
                    current_session_name = "NY_LUNCH"
                # "Close Trap" starts 15:00
                elif hour >= 15:
                    current_session_name = "NY_CLOSE"

            # --- OPTIMIZATION TRIGGER (Every Session Quarter) ---
            # Get current quarter (1-4) within the session
            current_quarter = bank_filter.get_quarter(hour, minute, base_session)

            # Trigger optimization on session change OR quarter change (4 sessions × 4 quarters = 16 per day)
            session_changed = current_session_name != last_processed_session
            quarter_changed = current_quarter != last_processed_quarter

            if session_changed or quarter_changed:
                if session_changed:
                    logging.info(f"🔄 SESSION HANDOVER: {last_processed_session} -> {current_session_name} Q{current_quarter} (Base: {base_session})")
                else:
                    logging.info(f"🔄 QUARTER CHANGE: {current_session_name} Q{last_processed_quarter} -> Q{current_quarter}")

                if CONFIG.get('GEMINI', {}).get('enabled', False):
                    print("\n" + "=" * 60)
                    print(f"🧠 GEMINI OPTIMIZATION - {current_session_name} Q{current_quarter}")
                    print("=" * 60)

                    # 1. Fetch Events & Holiday Context
                    try:
                        raw_events = news_filter.fetch_news()
                        events_str = str(raw_events)
                    except Exception as e:
                        events_str = "Events data unavailable."

                    try:
                        holiday_context = news_filter.get_holiday_context(current_time)
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
                    except Exception as e:
                        logging.warning(f"Failed to get holiday context: {e}")
                        holiday_context = "NORMAL_LIQUIDITY"

                    # Get Seasonal Context
                    try:
                        seasonal_context = news_filter.get_seasonal_context(current_time)
                        # Log seasonal phase with specific emoji indicators
                        if seasonal_context == "PHASE_1_LAST_GASP":
                            logging.info(f"⚡ SEASONAL PHASE: LAST GASP (Dec 20-23) - High volume, violent trends")
                        elif seasonal_context == "PHASE_2_DEAD_ZONE":
                            logging.info(f"☠️  SEASONAL PHASE: DEAD ZONE (Dec 24-31) - 60% volume drop, broken structure")
                        elif seasonal_context == "PHASE_3_JAN2_REENTRY":
                            logging.info(f"🐻 SEASONAL PHASE: JAN 2 RE-ENTRY - Bearish bias, funds returning")
                        # NORMAL_SEASONAL doesn't need logging
                    except Exception as e:
                        logging.warning(f"Failed to get seasonal context: {e}")
                        seasonal_context = "NORMAL_SEASONAL"

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
                        logging.warning("⚠️  Gemini optimization failed - using default multipliers")
                        print("=" * 60 + "\n")

                last_processed_session = current_session_name
                last_processed_quarter = current_quarter

            # === STEP 2: INCREMENTAL UPDATE (SEQUENTIAL FETCH) ===
            # Fetch MES first, then MNQ, then VIX immediately after to keep timestamps close
            recent_data = await client.async_get_market_data(lookback_minutes=15, force_fetch=True)
            recent_mnq_data = await mnq_client.async_get_market_data(lookback_minutes=15, force_fetch=True)
            # --- NEW: Fetch VIX Data ---
            fetch_vix = vix_fetch_toggle
            vix_fetch_toggle = not vix_fetch_toggle
            if fetch_vix:
                recent_vix_data = await vix_client.async_get_market_data(lookback_minutes=15, force_fetch=True)
            else:
                recent_vix_data = pd.DataFrame()

            if not recent_data.empty:
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
            if master_df.empty or master_mnq_df.empty:
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
                has_bank = bool(persisted_state.get("bank_filter"))
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
                    for ts, row in new_df.sort_index().iterrows():
                        bank_filter.update(ts, row['high'], row['low'], row['close'])

                data_backfilled = True
                restore_persisted_state(last_ts)
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

            rejection_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            bank_filter.update(current_time, currbar['high'], currbar['low'], currbar['close'])
            chop_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            extension_filter.update(currbar['high'], currbar['low'], currbar['close'], current_time)
            structure_blocker.update(new_df)
            regime_blocker.update(new_df)
            penalty_blocker.update(new_df)
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
                        logging.info(f"⚠️ CHOP RESTRICTION: {chop_reason}")
                        last_chop_reason = chop_reason

                elif "ALLOW_SHORT_ONLY" in chop_reason:
                    allowed_chop_side = "SHORT"
                    # Do NOT continue; allow the loop to proceed but enforce SHORT only
                    if last_chop_reason != chop_reason:
                        logging.info(f"⚠️ CHOP RESTRICTION: {chop_reason}")
                        last_chop_reason = chop_reason

                else:
                    # Hard Block (Standard Chop)
                    # Log every single time or throttle it
                    # logging.info(f"⛔ TRADE BLOCKED: {chop_reason}")
                    await asyncio.sleep(0.5)  # Faster check when choppy
                    continue
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

            # Only process signals on NEW bars
            # is_new_bar already computed above
            if is_new_bar:

                # Sync local active trade with broker state to avoid getting stuck
                if active_trade is not None:
                    broker_pos = client.get_position()
                    if broker_pos.get("stale"):
                        logging.warning("Position stale during broker sync; skipping flat check.")
                    else:
                        is_flat = broker_pos.get('side') is None or broker_pos.get('size', 0) == 0
                        if is_flat:
                            flat_position_streak += 1
                        else:
                            flat_position_streak = 0

                        # SAFETY CHECK: Only clear after two consecutive flat reads
                        if is_flat and flat_position_streak >= 2:
                            logging.info("Broker reports flat while tracking active_trade; clearing local state (confirmed).")
                            # Calculate PnL for directional loss tracking
                            trade_side = active_trade['side']
                            entry_price = active_trade['entry_price']
                            trade_size = active_trade.get('size', 5)
                            exit_price = current_price
                            tp_dist = active_trade.get('tp_dist')
                            sl_dist = active_trade.get('sl_dist')
                            if tp_dist is not None and sl_dist is not None:
                                if trade_side == 'LONG':
                                    tp_price = entry_price + tp_dist
                                    sl_price = entry_price - sl_dist
                                if current_price >= tp_price:
                                    exit_price = tp_price
                                elif current_price <= sl_price:
                                    exit_price = sl_price
                            else:
                                tp_price = entry_price - tp_dist
                                sl_price = entry_price + sl_dist
                                if current_price <= tp_price:
                                    exit_price = tp_price
                                elif current_price >= sl_price:
                                    exit_price = sl_price
                        if trade_side == 'LONG':
                            pnl_points = exit_price - entry_price
                        else:
                            pnl_points = entry_price - exit_price
                        # Convert points to dollars: MES = $5 per point per contract
                        pnl_dollars = pnl_points * 5.0 * trade_size
                        update_mom_rescue_score(active_trade, pnl_points, current_time)
                        update_hostile_day_on_close(active_trade.get('strategy'), pnl_points, current_time)
                        directional_loss_blocker.record_trade_result(trade_side, pnl_points, current_time)
                        circuit_breaker.update_trade_result(pnl_dollars)
                        logging.info(f"Trade closed: {trade_side} | Entry: {entry_price:.2f} | Exit: {exit_price:.2f} | PnL: {pnl_points:.2f} pts (${pnl_dollars:.2f})")
                        active_trade = None
                        opposite_signal_count = 0
                        client._local_position = {'side': None, 'size': 0, 'avg_price': 0.0}
                        flat_position_streak = 0
                else:
                    flat_position_streak = 0


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

                        if pd.notna(atr_expansion) and pd.notna(vwap_sigma):
                            atr_ok_t1 = atr_expansion >= ATR_EXP_T1
                            atr_ok_t2 = atr_expansion >= ATR_EXP_T2
                            displaced_down = vwap_sigma <= -VWAP_SIGMA_T1
                            displaced_up = vwap_sigma >= VWAP_SIGMA_T1
                            no_reclaim_down_t1 = bool(trend_day_series["no_reclaim_down_t1"].iloc[-1])
                            no_reclaim_up_t1 = bool(trend_day_series["no_reclaim_up_t1"].iloc[-1])
                            no_reclaim_down_t2 = bool(trend_day_series["no_reclaim_down_t2"].iloc[-1])
                            no_reclaim_up_t2 = bool(trend_day_series["no_reclaim_up_t2"].iloc[-1])
                            reclaim_down = bool(trend_day_series["reclaim_down"].iloc[-1])
                            reclaim_up = bool(trend_day_series["reclaim_up"].iloc[-1])

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
                        if trend_day_tier > 0 and trend_day_dir:
                            # Disable TrendDay if a large impulse prints in the OPPOSITE direction.
                            # Uses the same impulse criteria + wick override as ImpulseFilter.
                            opp_impulse = False
                            opp_reason = ""
                            if impulse_filter.avg_body_size > 0:
                                impulse_threshold = impulse_filter.avg_body_size * impulse_filter.impulse_multiplier
                                if impulse_filter.last_candle_body > impulse_threshold:
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

                        if trend_day_tier > 0 and (
                            trend_day_tier != last_trend_day_tier or trend_day_dir != last_trend_day_dir
                        ):
                            logging.warning(
                                f"🛑 TrendDay Tier {trend_day_tier} {trend_day_dir} activated "
                                f"@ {current_time.strftime('%Y-%m-%d %H:%M')}"
                            )
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
                else:
                    trend_day_tier = 0
                    trend_day_dir = None
                    last_trend_day_tier = 0
                    last_trend_day_dir = None
                    sticky_trend_dir = None
                    sticky_reclaim_count = 0
                    sticky_opposite_count = 0
                    tier1_seen = False

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

                # === EARLY EXIT CHECK ===
                if active_trade is not None:
                    active_trade['bars_held'] += 1
                    strategy_name = active_trade['strategy']
                    early_exit_config = CONFIG.get('EARLY_EXIT', {}).get(strategy_name, {})

                    if active_trade['side'] == 'LONG':
                        is_green = current_price > active_trade['entry_price']
                    else:
                        is_green = current_price < active_trade['entry_price']

                    was_green = active_trade.get('was_green')
                    if was_green is not None and is_green != was_green:
                        active_trade['profit_crosses'] = active_trade.get('profit_crosses', 0) + 1
                    active_trade['was_green'] = is_green

                    if early_exit_config.get('enabled', False):
                        exit_time = early_exit_config.get('exit_if_not_green_by', 50)
                        exit_cross = early_exit_config.get('max_profit_crosses', 100)
                        should_exit = False
                        
                        if active_trade['bars_held'] >= exit_time and not is_green:
                            should_exit = True
                            exit_reason = f"not green after {active_trade['bars_held']} bars"
                        if active_trade.get('profit_crosses', 0) > exit_cross:
                            should_exit = True
                            exit_reason = f"choppy ({active_trade['profit_crosses']} crosses)"

                        if should_exit:
                            logging.info(f"⏰ EARLY EXIT: {strategy_name} - {exit_reason}")

                            event_logger.log_early_exit(
                                reason=exit_reason,
                                bars_held=active_trade['bars_held'],
                                current_price=current_price,
                                entry_price=active_trade['entry_price']
                            )

                            trade_side = active_trade['side']
                            entry_price = active_trade['entry_price']
                            trade_size = active_trade.get('size', 5)
                            if trade_side == 'LONG':
                                pnl_points = current_price - entry_price
                            else:
                                pnl_points = entry_price - current_price
                            pnl_dollars = pnl_points * 5.0 * trade_size
                            update_mom_rescue_score(active_trade, pnl_points, current_time)
                            update_hostile_day_on_close(strategy_name, pnl_points, current_time)
                            directional_loss_blocker.record_trade_result(trade_side, pnl_points, current_time)
                            circuit_breaker.update_trade_result(pnl_dollars)
                            logging.info(
                                f"📊 Early exit closed: {trade_side} | Entry: {entry_price:.2f} | "
                                f"Exit: {current_price:.2f} | PnL: {pnl_points:.2f} pts (${pnl_dollars:.2f})"
                            )

                            position = client.get_position()
                            if position.get("stale"):
                                logging.warning("Position stale on early-exit close; skipping close.")
                            elif position['side'] is not None:
                                client.close_position(position)
                            active_trade = None

                # === STRATEGY EXECUTION ===
                if news_blocked:
                    logging.info("📰 NEWS BLACKOUT: Skipping trade execution (data continues)")
                    await asyncio.sleep(10)
                    continue
                strategy_results = {'checked': [], 'rejected': [], 'executed': None}

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

                            if active_trade is not None:
                                old_side = active_trade['side']
                                old_entry = active_trade['entry_price']
                                old_size = active_trade.get('size', 5)
                                if old_side == 'LONG':
                                    old_pnl_points = current_price - old_entry
                                else:
                                    old_pnl_points = old_entry - current_price
                                old_pnl_dollars = old_pnl_points * 5.0 * old_size
                                update_mom_rescue_score(active_trade, old_pnl_points, current_time)
                                update_hostile_day_on_close(active_trade.get('strategy'), old_pnl_points, current_time)
                                directional_loss_blocker.record_trade_result(old_side, old_pnl_points, current_time)
                                circuit_breaker.update_trade_result(old_pnl_dollars)
                                logging.info(
                                    f"Trade closed (reverse): {old_side} | Entry: {old_entry:.2f} | "
                                    f"Exit: {current_price:.2f} | PnL: {old_pnl_points:.2f} pts (${old_pnl_dollars:.2f})"
                                )

                            success, opposite_signal_count = await client.async_close_and_reverse(
                                pending_signal, current_price, opposite_signal_count
                            )
                            if success:
                                order_details = getattr(client, "_last_order_details", None) or {}
                                entry_price = order_details.get("entry_price", current_price)
                                tp_dist = order_details.get("tp_points", pending_signal.get('tp_dist', 6.0))
                                sl_dist = order_details.get("sl_points", pending_signal.get('sl_dist', 4.0))
                                size = order_details.get("size", pending_signal.get('size', 5))
                                pending_signal['tp_dist'] = tp_dist
                                pending_signal['sl_dist'] = sl_dist
                                pending_signal['size'] = size
                                pending_signal['entry_price'] = entry_price
                                active_trade = {
                                    'strategy': pending_signal['strategy'],
                                    'sub_strategy': pending_signal.get('sub_strategy'),
                                    'side': pending_signal['side'],
                                    'entry_price': entry_price,
                                    'entry_bar': bar_count,
                                    'bars_held': 0,
                                    'tp_dist': tp_dist,
                                    'sl_dist': sl_dist,
                                    'size': size,
                                    'stop_order_id': client._active_stop_order_id,
                                    'entry_mode': pending_signal.get('entry_mode', "rescued"),
                                    'profit_crosses': 0,
                                    'was_green': None,
                                    'rescue_from_strategy': pending_signal.get('rescue_from_strategy'),
                                    'rescue_from_sub_strategy': pending_signal.get('rescue_from_sub_strategy'),
                                    'trend_day_tier': pending_signal.get('trend_day_tier'),
                                    'trend_day_dir': pending_signal.get('trend_day_dir'),
                                }
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

# Run ML Analysis
                ml_signal = None
                if ml_strategy.model_loaded:
                    try:
                        ml_signal = ml_strategy.on_bar(new_df)
                        if ml_signal: strategy_results['checked'].append('MLPhysics')
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
                            if abs((vix_ts - mes_ts).total_seconds()) > 120:
                                logging.info("VIX stale vs MES; skipping VIXReversionStrategy")
                                continue
                            signal = strat.on_bar(new_df, vix_df)
                        else:
                            signal = strat.on_bar(new_df)

                        if signal:
                            if hostile_day_active and is_hostile_disabled_strategy(signal, strat_name):
                                logging.info(f"🛑 HOSTILE DAY: Skipping {strat_name}")
                                continue
                            # ==========================================
                            # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                            # ==========================================
                            sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                            tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                            old_sl = signal.get('sl_dist', 4.0)
                            old_tp = signal.get('tp_dist', 6.0)

                            if 'sl_dist' not in signal or 'tp_dist' not in signal:
                                logging.warning(f"⚠️ {strat_name} missing sl_dist/tp_dist, using defaults")

                            signal['sl_dist'] = old_sl * sl_mult
                            signal['tp_dist'] = old_tp * tp_mult

                            # Enforce minimums
                            MIN_SL = 4.0
                            MIN_TP = 6.0
                            if signal['sl_dist'] < MIN_SL:
                                logging.warning(f"⚠️ SL too tight ({signal['sl_dist']:.2f}), enforcing minimum {MIN_SL}")
                                signal['sl_dist'] = MIN_SL
                            if signal['tp_dist'] < MIN_TP:
                                logging.warning(f"⚠️ TP too tight ({signal['tp_dist']:.2f}), enforcing minimum {MIN_TP}")
                                signal['tp_dist'] = MIN_TP

                            if sl_mult != 1.0 or tp_mult != 1.0:
                                logging.info(f"🧠 GEMINI OPTIMIZED: {strat_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")

                            # Enforce HTF range fade directional restriction
                            # if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                            #    logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                            #    continue

                            # Add to candidate list (Priority 1 = FAST)
                            candidate_signals.append((1, strat, signal, strat_name))

                            # Log as candidate
                            event_logger.log_strategy_signal(
                                strategy_name=signal.get('strategy', strat_name),
                                side=signal['side'],
                                tp_dist=signal.get('tp_dist', 6.0),
                                sl_dist=signal.get('sl_dist', 4.0),
                                price=current_price,
                                additional_info={"status": "CANDIDATE", "priority": "FAST"}
                            )
                            logging.info(f"📊 CANDIDATE (FAST): {strat_name} {signal['side']} @ {current_price:.2f}")

                    except Exception as e:
                        logging.error(f"Error in {strat_name}: {e}")

                # -----------------------------------------------------------------
                # HARVEST PHASE 2: STANDARD STRATEGIES (Priority 2)
                # -----------------------------------------------------------------
                # Shuffle standard strategies
                current_standard = standard_strategies.copy()
                random.shuffle(current_standard)

                for strat in current_standard:
                    strat_name = strat.__class__.__name__
                    signal = None

                    # (SMT needs master_mnq_df, ML needs ml_signal, others use new_df)
                    if strat_name == "MLPhysicsStrategy":
                        signal = ml_signal
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
                            logging.error(f"Error in {strat_name}: {e}")
                    else:
                        try:
                            signal = strat.on_bar(new_df)
                        except Exception as e:
                            logging.error(f"Error in {strat_name}: {e}")

                    if signal:
                        # ==========================================
                        # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                        # ==========================================
                        sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                        tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                        old_sl = signal.get('sl_dist', 4.0)
                        old_tp = signal.get('tp_dist', 6.0)

                        if 'sl_dist' not in signal or 'tp_dist' not in signal:
                            logging.warning(f"⚠️ {strat_name} missing sl_dist/tp_dist, using defaults")

                        signal['sl_dist'] = old_sl * sl_mult
                        signal['tp_dist'] = old_tp * tp_mult

                        # Enforce minimums
                        MIN_SL = 4.0
                        MIN_TP = 6.0
                        if signal['sl_dist'] < MIN_SL:
                            logging.warning(f"⚠️ SL too tight ({signal['sl_dist']:.2f}), enforcing minimum {MIN_SL}")
                            signal['sl_dist'] = MIN_SL
                        if signal['tp_dist'] < MIN_TP:
                            logging.warning(f"⚠️ TP too tight ({signal['tp_dist']:.2f}), enforcing minimum {MIN_TP}")
                            signal['tp_dist'] = MIN_TP

                        if sl_mult != 1.0 or tp_mult != 1.0:
                            logging.info(f"🧠 GEMINI OPTIMIZED: {strat_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")

                        # Enforce HTF range fade directional restriction
                        # if allowed_chop_side is not None and signal['side'] != allowed_chop_side:
                        #    logging.info(f"⛔ BLOCKED by HTF Range Rule: Signal {signal['side']} vs Allowed {allowed_chop_side}")
                        #    continue

                        # Add to candidate list (Priority 2 = STANDARD)
                        if hostile_day_active and is_hostile_disabled_strategy(signal, strat_name):
                            logging.info(f"🛑 HOSTILE DAY: Skipping {strat_name}")
                            continue
                        candidate_signals.append((2, strat, signal, strat_name))

                        # Log as candidate
                        event_logger.log_strategy_signal(
                            strategy_name=signal.get('strategy', strat_name),
                            side=signal['side'],
                            tp_dist=signal.get('tp_dist', 6.0),
                            sl_dist=signal.get('sl_dist', 4.0),
                            price=current_price,
                            additional_info={"status": "CANDIDATE", "priority": "STANDARD"}
                        )
                        logging.info(f"📊 CANDIDATE (STANDARD): {strat_name} {signal['side']} @ {current_price:.2f}")

                # -----------------------------------------------------------------
                # SELECTION PHASE: Process candidates by priority until one passes
                # -----------------------------------------------------------------
                candidate_signals.sort(key=lambda x: x[0])

                # Multi-strategy consensus override (vote-based)
                direction_counts = {"LONG": 0, "SHORT": 0}
                smt_side = None
                for _, _, sig, s_name in candidate_signals:
                    if hostile_day_active and is_hostile_disabled_strategy(sig, s_name):
                        continue
                    side = sig.get("side")
                    if side in direction_counts:
                        strat_label = sig.get("strategy", s_name)
                        if str(strat_label).startswith("MLPhysics") and not consensus_ml_ok(sig, s_name):
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
                if consensus_side:
                    consensus_candidates = [
                        (sig, s_name) for _, _, sig, s_name in candidate_signals
                        if sig.get("side") == consensus_side
                        and (
                            not str(sig.get("strategy", s_name)).startswith("MLPhysics")
                            or consensus_ml_ok(sig, s_name)
                        )
                    ]
                    if consensus_candidates:
                        consensus_tp_signal, consensus_tp_source = min(
                            consensus_candidates,
                            key=lambda item: item[0].get('tp_dist', float('inf'))
                        )
                        logging.info(
                            "🧮 CONSENSUS TP PICK: "
                            f"{consensus_tp_source} TP={consensus_tp_signal.get('tp_dist', 0):.2f} "
                            f"SL={consensus_tp_signal.get('sl_dist', 0):.2f}"
                        )

                signal_executed = False
                for priority, strat, sig, strat_name in candidate_signals:
                    signal = sig
                    priority_label = "FAST" if priority == 1 else "STANDARD"
                    do_execute = False
                    signal.setdefault("strategy", strat_name)
                    trend_day_counter = False
                    if trend_day_tier > 0 and trend_day_dir:
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
                        logging.info(f"⏭️ Skipping {strat_name} {signal['side']} due to consensus {consensus_side}")
                        continue
                    if hostile_day_active and is_hostile_disabled_strategy(signal, strat_name):
                        logging.info(f"🛑 HOSTILE DAY: Skipping {strat_name}")
                        continue

                    def should_log_ui(current_signal, fallback_name):
                        return True

                    def log_filter_block(filter_name, reason, side_override=None):
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
                            if not potential_rescue:
                                return False
                            if continuation_no_bypass and continuation_core_trigger(filter_name):
                                return False
                            rescue_blocked, rescue_reason = trend_filter.should_block_trade(new_df, potential_rescue['side'])
                            if rescue_blocked:
                                log_filter_block("TrendFilter", rescue_reason, side_override=potential_rescue['side'])
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
                        if trend_day_counter:
                            if not try_consensus_rescue(
                                f"TrendDayTier{trend_day_tier}",
                                "Counter-trend",
                            ):
                                logging.info(
                                    f"⛔ CONSENSUS BLOCKED by TrendDayTier{trend_day_tier}"
                                )
                                continue
                        if consensus_tp_source:
                            signal['tp_dist'] = consensus_tp_signal.get('tp_dist', signal.get('tp_dist', 6.0))
                            signal['sl_dist'] = consensus_tp_signal.get('sl_dist', signal.get('sl_dist', 4.0))
                            logging.info(
                                "🧮 CONSENSUS TP SOURCE: "
                                f"{consensus_tp_source} TP={signal['tp_dist']:.2f} SL={signal['sl_dist']:.2f}"
                            )
                        is_feasible, feasibility_reason = chop_analyzer.check_target_feasibility(
                            entry_price=current_price,
                            side=signal['side'],
                            tp_distance=signal.get('tp_dist', 6.0),
                            df_1m=new_df,
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

                        # 7. Volatility Guardrail (Physics - Apply to Rescued too)
                        should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0), base_size=5)
                        if not should_trade:
                            # If physics fail, we can't trade.
                            log_filter_block("VolatilityGuardrail", "Volatility check failed")
                            logging.info(f"⛔ BLOCKED by Volatility Guardrail")
                            if is_rescued:
                                log_rescue_failed("VolatilityGuardrail: Volatility check failed")
                            continue

                        signal['sl_dist'] = vol_adj['sl_dist']
                        signal['tp_dist'] = vol_adj['tp_dist']
                        if vol_adj.get('adjustment_applied', False): signal['size'] = vol_adj['size']

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
                            if potential_rescue and not is_rescued:
                                rescue_blocked, rescue_reason = trend_filter.should_block_trade(new_df, potential_rescue['side'])
                                if rescue_blocked:
                                    log_filter_block('TrendFilter', rescue_reason, side_override=potential_rescue['side'])
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
                        if trend_day_counter:
                            if not try_rescue_trigger('Counter-trend', f"TrendDayTier{trend_day_tier}"):
                                logging.info(f"BLOCKED by TrendDayTier{trend_day_tier}")
                                continue

                        # 1. Target Feasibility
                        is_feasible, feasibility_reason = chop_analyzer.check_target_feasibility(
                            entry_price=current_price, side=signal['side'], tp_distance=signal.get('tp_dist', 6.0), df_1m=new_df
                        )
                        if not is_feasible:
                            log_filter_block('TargetFeasibility', feasibility_reason)
                            logging.info(f"Signal ignored ({priority_label}): {feasibility_reason}")
                            continue

                        # 2. Rejection/Bias
                        rej_blocked, rej_reason = rejection_filter.should_block_trade(signal['side'])
                        range_bias_blocked = (allowed_chop_side is not None and signal['side'] != allowed_chop_side)
                        if rej_blocked or range_bias_blocked:
                            reason = rej_reason if rej_blocked else f"Opposite HTF Range Bias ({allowed_chop_side})"
                            if not try_rescue_trigger(reason, 'Rejection/Bias'):
                                continue

                        # 3. Directional Loss Blocker
                        dir_blocked, dir_reason = directional_loss_blocker.should_block_trade(signal['side'], current_time)
                        if dir_blocked:
                            if not try_rescue_trigger(dir_reason, 'DirectionalLoss'):
                                continue

                        # 4. Impulse Filter
                        impulse_blocked, impulse_reason = impulse_filter.should_block_trade(signal['side'])
                        if impulse_blocked:
                            if not try_rescue_trigger(impulse_reason, 'ImpulseFilter'):
                                continue

                        # 5. Regime Structure Blocker (hard stop)
                        regime_blocked, regime_reason = regime_blocker.should_block_trade(signal['side'], current_price)
                        if regime_blocked:
                            log_filter_block('RegimeBlocker', regime_reason)
                            logging.info(f"HARD STOP by RegimeBlocker (EQH/EQL): {regime_reason}")
                            continue

                        # 6. Independent System Checks (HTF/Structure/Bank/Trend)
                        upgraded_blocked = False
                        upgraded_reasons = []
                        tp_dist = signal.get('tp_dist', 15.0)
                        effective_tp_dist = tp_dist
                        if allowed_chop_side is not None and signal['side'] == allowed_chop_side:
                            effective_tp_dist = tp_dist * 0.5
                            logging.info(f"RELAXING FVG CHECK (Main): Fading Range {signal['side']} (Req Room: {effective_tp_dist*0.4:.2f} pts)")
                        fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                            signal['side'], current_price, None, None, tp_dist=effective_tp_dist
                        )
                        if fvg_blocked:
                            upgraded_reasons.append(f"HTF_FVG: {fvg_reason}")
                        struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
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
                        final_blocked = False
                        final_reason = ''
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
                        if final_blocked:
                            if is_rescued:
                                if rescue_bypass_allowed:
                                    logging.info(f"BYPASS Filters ({final_reason}): Rescued by {signal['strategy']}")
                                else:
                                    log_filter_block("FilterStack", final_reason)
                                    logging.info("FILTER STACK BLOCKED: Rescue bypass disabled")
                                    log_rescue_failed(f"FilterStack: {final_reason}")
                                    continue
                            else:
                                if not try_rescue_trigger(final_reason, 'FilterStack'):
                                    continue

                        # 7. Chop & Extension
                        vol_regime, _, _ = volatility_filter.get_regime(new_df)
                        chop_blocked, chop_reason = chop_filter.should_block_trade(
                            signal['side'], rejection_filter.prev_day_pm_bias, current_price, 'NEUTRAL', vol_regime
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

                        # 8. Volatility Guardrail
                        should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0), base_size=5)
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
                        do_execute = True
                    if not do_execute:
                        continue

                    # === EXECUTION ===
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
                    log_rescue_success()
                    strategy_results['executed'] = strat_name
                    logging.info(f"✅ {priority_label} EXEC: {signal['strategy']} ({signal['side']})")

                    # ... [Remaining Execution Code same as before] ...
                    # Close and Reverse logic...
                    if active_trade is not None:
                        old_side = active_trade['side']
                        old_entry = active_trade['entry_price']
                        old_size = active_trade.get('size', 5)
                        if old_side == 'LONG':
                            old_pnl_points = current_price - old_entry
                        else:
                            old_pnl_points = old_entry - current_price
                        # Convert points to dollars: MES = $5 per point per contract
                        old_pnl_dollars = old_pnl_points * 5.0 * old_size
                        update_mom_rescue_score(active_trade, old_pnl_points, current_time)
                        update_hostile_day_on_close(active_trade.get('strategy'), old_pnl_points, current_time)
                        directional_loss_blocker.record_trade_result(old_side, old_pnl_points, current_time)
                        circuit_breaker.update_trade_result(old_pnl_dollars)
                        logging.info(f"📊 Trade closed (reverse): {old_side} | Entry: {old_entry:.2f} | Exit: {current_price:.2f} | PnL: {old_pnl_points:.2f} pts (${old_pnl_dollars:.2f})")

                    success, opposite_signal_count = await client.async_close_and_reverse(signal, current_price, opposite_signal_count)

                    if success:
                        order_details = getattr(client, "_last_order_details", None) or {}
                        entry_price = order_details.get("entry_price", current_price)
                        tp_dist = order_details.get("tp_points", signal.get('tp_dist', 6.0))
                        sl_dist = order_details.get("sl_points", signal.get('sl_dist', 4.0))
                        size = order_details.get("size", signal.get('size', 5))
                        signal['tp_dist'] = tp_dist
                        signal['sl_dist'] = sl_dist
                        signal['size'] = size
                        signal['entry_price'] = entry_price
                        active_trade = {
                            'strategy': signal['strategy'],
                            'sub_strategy': signal.get('sub_strategy'),
                            'side': signal['side'],
                            'entry_price': entry_price,
                            'entry_bar': bar_count,
                            'bars_held': 0,
                            'tp_dist': tp_dist,
                            'sl_dist': sl_dist,
                            'size': size,
                            'stop_order_id': client._active_stop_order_id,
                            'entry_mode': signal.get('entry_mode', "standard"),
                            'profit_crosses': 0,
                            'was_green': None,
                            'rescue_from_strategy': signal.get('rescue_from_strategy'),
                            'rescue_from_sub_strategy': signal.get('rescue_from_sub_strategy'),
                            'trend_day_tier': signal.get('trend_day_tier'),
                            'trend_day_dir': signal.get('trend_day_dir'),
                        }

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

                                # ==========================================
                                # 🧠 GEMINI 3.0: APPLY OPTIMIZATION
                                # ==========================================
                                # Apply the active session multipliers from CONFIG
                                # If Gemini is disabled or failed, these default to 1.0
                                sl_mult = CONFIG.get('DYNAMIC_SL_MULTIPLIER', 1.0)
                                tp_mult = CONFIG.get('DYNAMIC_TP_MULTIPLIER', 1.0)

                                # ALWAYS ensure sl_dist/tp_dist are set (fix for missing values)
                                old_sl = sig.get('sl_dist', 4.0)
                                old_tp = sig.get('tp_dist', 6.0)

                                # Warn if strategy didn't set these
                                if 'sl_dist' not in sig or 'tp_dist' not in sig:
                                    logging.warning(f"⚠️ {s_name} missing sl_dist/tp_dist, using defaults")

                                # Apply Multipliers
                                sig['sl_dist'] = old_sl * sl_mult
                                sig['tp_dist'] = old_tp * tp_mult

                                # Enforce minimums to prevent dangerously tight stops
                                MIN_SL = 4.0  # 16 ticks minimum
                                MIN_TP = 6.0  # 24 ticks minimum (1.5:1 RR)
                                if sig['sl_dist'] < MIN_SL:
                                    logging.warning(f"⚠️ SL too tight ({sig['sl_dist']:.2f}), enforcing minimum {MIN_SL}")
                                    sig['sl_dist'] = MIN_SL
                                if sig['tp_dist'] < MIN_TP:
                                    logging.warning(f"⚠️ TP too tight ({sig['tp_dist']:.2f}), enforcing minimum {MIN_TP}")
                                    sig['tp_dist'] = MIN_TP

                                if sl_mult != 1.0 or tp_mult != 1.0:
                                    logging.info(f"🧠 GEMINI OPTIMIZED: {s_name} | SL: {old_sl:.2f}->{sig['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{sig['tp_dist']:.2f} (x{tp_mult})")
                                # ==========================================

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
                                    tp_distance=sig.get('tp_dist', 6.0),
                                    df_1m=new_df
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
                                struct_blocked, struct_reason = structure_blocker.should_block_trade(sig['side'], current_price)
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
                                penalty_blocked, penalty_reason = penalty_blocker.should_block_trade(sig['side'], current_price)
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
                                if chop_blocked:
                                    event_logger.log_filter_check("ChopFilter", sig['side'], False, chop_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ChopFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                ext_blocked, ext_reason = extension_filter.should_block_trade(sig['side'])
                                if ext_blocked:
                                    event_logger.log_filter_check("ExtensionFilter", sig['side'], False, ext_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("ExtensionFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # Trend Filter (already checked above with is_range_fade)
                                if trend_blocked:
                                    event_logger.log_filter_check("TrendFilter", sig['side'], False, trend_reason, strategy=sig.get('strategy', s_name))
                                    del pending_loose_signals[s_name]; continue
                                else:
                                    event_logger.log_filter_check("TrendFilter", sig['side'], True, strategy=sig.get('strategy', s_name))

                                # Volatility & Guardrail Check
                                # We pass the Gemini-modified params (sig['sl_dist']) into the filter.
                                # The filter applies Guardrails + Rounding.
                                should_trade, vol_adj = check_volatility(new_df, sig.get('sl_dist', 4.0), sig.get('tp_dist', 6.0), base_size=5)

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

                                # Only apply SIZE adjustment if the regime explicitly demands it (Low Vol)
                                if vol_adj.get('adjustment_applied', False):
                                    sig['size'] = vol_adj['size']
                                    event_logger.log_trade_modified(
                                        "VolatilityAdjustment",
                                        sig.get('tp_dist', 6.0),
                                        vol_adj['tp_dist'],
                                        f"Volatility/Guardrail adjustment (Regime: {vol_adj['regime']})"
                                    )

                                logging.info(f"✅ LOOSE EXEC: {s_name}")
                                event_logger.log_strategy_execution(s_name, "LOOSE")

                                # Track old trade result BEFORE close_and_reverse overwrites active_trade
                                if active_trade is not None:
                                    old_side = active_trade['side']
                                    old_entry = active_trade['entry_price']
                                    old_size = active_trade.get('size', 5)
                                    if old_side == 'LONG':
                                        old_pnl_points = current_price - old_entry
                                    else:
                                        old_pnl_points = old_entry - current_price
                                    # Convert points to dollars: MES = $5 per point per contract
                                    old_pnl_dollars = old_pnl_points * 5.0 * old_size
                                    update_mom_rescue_score(active_trade, old_pnl_points, current_time)
                                    update_hostile_day_on_close(active_trade.get('strategy'), old_pnl_points, current_time)
                                    directional_loss_blocker.record_trade_result(old_side, old_pnl_points, current_time)
                                    circuit_breaker.update_trade_result(old_pnl_dollars)
                                    logging.info(f"📊 Trade closed (reverse): {old_side} | Entry: {old_entry:.2f} | Exit: {current_price:.2f} | PnL: {old_pnl_points:.2f} pts (${old_pnl_dollars:.2f})")

                                success, opposite_signal_count = await client.async_close_and_reverse(sig, current_price, opposite_signal_count)
                                if success:
                                    order_details = getattr(client, "_last_order_details", None) or {}
                                    entry_price = order_details.get("entry_price", current_price)
                                    tp_dist = order_details.get("tp_points", sig.get('tp_dist', 6.0))
                                    sl_dist = order_details.get("sl_points", sig.get('sl_dist', 4.0))
                                    size = order_details.get("size", sig.get('size', 5))
                                    sig['tp_dist'] = tp_dist
                                    sig['sl_dist'] = sl_dist
                                    sig['size'] = size
                                    sig['entry_price'] = entry_price
                                    active_trade = {
                                        'strategy': s_name,
                                        'sub_strategy': sig.get('sub_strategy'),
                                        'side': sig['side'],
                                        'entry_price': entry_price,
                                        'entry_bar': bar_count,
                                        'bars_held': 0,
                                        'tp_dist': tp_dist,
                                        'sl_dist': sl_dist,  # Store SL for consistency
                                        'size': size,  # Use order size when available
                                        'stop_order_id': client._active_stop_order_id,
                                        'entry_mode': sig.get('entry_mode', "loose"),
                                        'profit_crosses': 0,
                                        'was_green': None,
                                        'rescue_from_strategy': sig.get('rescue_from_strategy'),
                                        'rescue_from_sub_strategy': sig.get('rescue_from_sub_strategy'),
                                        'trend_day_tier': sig.get('trend_day_tier'),
                                        'trend_day_dir': sig.get('trend_day_dir'),
                                    }

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

                                        # ALWAYS ensure sl_dist/tp_dist are set (fix for missing values)
                                        old_sl = signal.get('sl_dist', 4.0)
                                        old_tp = signal.get('tp_dist', 6.0)

                                        # Warn if strategy didn't set these
                                        if 'sl_dist' not in signal or 'tp_dist' not in signal:
                                            logging.warning(f"⚠️ {s_name} missing sl_dist/tp_dist, using defaults")

                                        # Apply Multipliers
                                        signal['sl_dist'] = old_sl * sl_mult
                                        signal['tp_dist'] = old_tp * tp_mult

                                        # Enforce minimums to prevent dangerously tight stops
                                        MIN_SL = 4.0  # 16 ticks minimum
                                        MIN_TP = 6.0  # 24 ticks minimum (1.5:1 RR)
                                        if signal['sl_dist'] < MIN_SL:
                                            logging.warning(f"⚠️ SL too tight ({signal['sl_dist']:.2f}), enforcing minimum {MIN_SL}")
                                            signal['sl_dist'] = MIN_SL
                                        if signal['tp_dist'] < MIN_TP:
                                            logging.warning(f"⚠️ TP too tight ({signal['tp_dist']:.2f}), enforcing minimum {MIN_TP}")
                                            signal['tp_dist'] = MIN_TP

                                        if sl_mult != 1.0 or tp_mult != 1.0:
                                            logging.info(f"🧠 GEMINI OPTIMIZED: {s_name} | SL: {old_sl:.2f}->{signal['sl_dist']:.2f} (x{sl_mult}) | TP: {old_tp:.2f}->{signal['tp_dist']:.2f} (x{tp_mult})")
                                        # ==========================================

                                        signal['entry_mode'] = "loose"

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
                                        event_logger.log_strategy_signal(
                                            strategy_name=signal.get('strategy', s_name),
                                            side=signal['side'],
                                            tp_dist=signal.get('tp_dist', 6.0),
                                            sl_dist=signal.get('sl_dist', 4.0),
                                            price=current_price,
                                            additional_info={"execution_type": "LOOSE"}
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

                                        fvg_blocked, fvg_reason = htf_fvg_filter.check_signal_blocked(
                                            signal['side'], current_price, None, None, tp_dist=effective_tp_dist
                                        )
                                        if fvg_blocked:
                                            event_logger.log_filter_check("HTF_FVG", signal['side'], False, fvg_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("HTF_FVG", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # === [FIX 2] UPDATED BLOCKER CHECK ===
                                        struct_blocked, struct_reason = structure_blocker.should_block_trade(signal['side'], current_price)
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
                                        penalty_blocked, penalty_reason = penalty_blocker.should_block_trade(signal['side'], current_price)
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
                                        if chop_blocked:
                                            event_logger.log_filter_check("ChopFilter", signal['side'], False, chop_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("ChopFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        ext_blocked, ext_reason = extension_filter.should_block_trade(signal['side'])
                                        if ext_blocked:
                                            event_logger.log_filter_check("ExtensionFilter", signal['side'], False, ext_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("ExtensionFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # Trend Filter (already checked above with is_range_fade)
                                        if trend_blocked:
                                            event_logger.log_filter_check("TrendFilter", signal['side'], False, trend_reason, strategy=signal.get('strategy', s_name))
                                            continue
                                        else:
                                            event_logger.log_filter_check("TrendFilter", signal['side'], True, strategy=signal.get('strategy', s_name))

                                        # Volatility & Guardrail Check
                                        # We pass the Gemini-modified params (signal['sl_dist']) into the filter.
                                        # The filter applies Guardrails + Rounding.
                                        should_trade, vol_adj = check_volatility(new_df, signal.get('sl_dist', 4.0), signal.get('tp_dist', 6.0), base_size=5)

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

                                        # Only apply SIZE adjustment if the regime explicitly demands it (Low Vol)
                                        if vol_adj.get('adjustment_applied', False):
                                            signal['size'] = vol_adj['size']
                                            event_logger.log_trade_modified(
                                                "VolatilityAdjustment",
                                                signal.get('tp_dist', 6.0),
                                                vol_adj['tp_dist'],
                                                f"Volatility/Guardrail adjustment (Regime: {vol_adj['regime']})"
                                            )

                                        # Log as QUEUED for UI visibility
                                        event_logger.log_strategy_signal(
                                            strategy_name=signal.get('strategy', s_name),
                                            side=signal['side'],
                                            tp_dist=signal.get('tp_dist', 0),
                                            sl_dist=signal.get('sl_dist', 0),
                                            price=current_price,
                                            additional_info={"status": "QUEUED", "priority": "LOOSE"}
                                        )
                                        logging.info(f"🕐 Queuing {s_name} signal")
                                        pending_loose_signals[s_name] = {'signal': signal, 'bar_count': 0}
                                except Exception as e:
                                    logging.error(f"Error in {s_name}: {e}")

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
            break
        except Exception as e:
            logging.error(f"Main Loop Error: {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    # Run the async bot with asyncio
    asyncio.run(run_bot())
