import builtins
import glob
import datetime as dt
import json
import logging
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# Provide fallbacks for modules that assume these globals without imports.
builtins.logging = logging
builtins.datetime = dt

from config import CONFIG, refresh_target_symbol
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
from orb_strategy import OrbStrategy
from intraday_dip_strategy import IntradayDipStrategy
from confluence_strategy import ConfluenceStrategy
from smt_strategy import SMTStrategy
from ict_model_strategy import ICTModelStrategy
from ml_physics_strategy import MLPhysicsStrategy
from dynamic_engine_strategy import DynamicEngineStrategy
from volatility_filter import volatility_filter, check_volatility
from regime_strategy import RegimeAdaptiveStrategy
from dynamic_sltp_params import dynamic_sltp_engine
from vixmeanreversion import VIXReversionStrategy
from directional_loss_blocker import DirectionalLossBlocker
from impulse_filter import ImpulseFilter
from legacy_filters import LegacyFilterSystem
from filter_arbitrator import FilterArbitrator
from continuation_strategy import FractalSweepStrategy, STRATEGY_CONFIGS
from htf_fvg_filter import HTFFVGFilter
from news_filter import NewsFilter


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_CSV_NAME = "ml_mes_et.csv"
CONTRACTS = 5
POINT_VALUE = 5.0
FEES_PER_20_CONTRACTS = 7.50
FEE_PER_CONTRACT_RT = FEES_PER_20_CONTRACTS / 20.0
FEE_PER_TRADE = FEE_PER_CONTRACT_RT * CONTRACTS
TICK_SIZE = 0.25
WARMUP_BARS = 20000
OPPOSITE_SIGNAL_THRESHOLD = 3
MIN_SL = 4.0
MIN_TP = 6.0
ENABLE_CONSENSUS_BYPASS = True
DISABLE_CONTINUATION_NY = False
ENABLE_DYNAMIC_ENGINE_1 = True
ALLOW_DYNAMIC_ENGINE_SOLO = False
ENABLE_HOSTILE_DAY_GUARD = True
HOSTILE_DAY_MAX_TRADES = 3
HOSTILE_DAY_MIN_TRADES = 2
HOSTILE_DAY_LOSS_THRESHOLD = 2
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
TREND_DAY_T1_REQUIRE_CONFIRMATION = False
TREND_DAY_TIMEFRAME_MINUTES = 1
ALT_PRE_TIER1_VWAP_SIGMA = 2.0

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


def bucket_label(value: float, edges: list[float]) -> str:
    for edge in edges:
        if value <= edge:
            return f"<= {edge:.2f}"
    return f"> {edges[-1]:.2f}"


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


def build_continuation_allowlist(cfg: Optional[dict], base_dir: Path) -> tuple[Optional[set], dict]:
    if not cfg or not cfg.get("enabled", True):
        return None, {}
    pattern = cfg.get("reports_glob", "backtest_reports/backtest_*.json")
    report_paths = [Path(p) for p in glob.glob(str(base_dir / pattern))]

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
            key = parse_continuation_key(trade.get("strategy"))
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

        if allowed_regimes:
            try:
                volatility_filter.calibrate(symbol_df)
            except Exception:
                pass

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
            key = f"Q{quarters[pos]}_W{weeks[pos]}_D{days[pos]}_{session[pos]}"
            if key not in STRATEGY_CONFIGS:
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


class AttributionTracker:
    def __init__(self, recent_limit: int = 25):
        self.trades = []
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

    def record_filter(self, name: str, kind: str = "block") -> None:
        if kind == "rescue":
            self.filter_rescues[name] += 1
        elif kind == "bypass":
            self.filter_bypasses[name] += 1
        else:
            self.filter_blocks[name] += 1

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
        self.recent_trades.append(trade)

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

        avg_mfe = self.mfe_sum / len(self.trades) if self.trades else 0.0
        avg_mae = self.mae_sum / len(self.trades) if self.trades else 0.0
        avg_mfe_win = self.mfe_win_sum / max(1, sum(1 for t in self.trades if t["pnl_net"] >= 0))
        avg_mae_win = self.mae_win_sum / max(1, sum(1 for t in self.trades if t["pnl_net"] >= 0))
        avg_mfe_loss = self.mfe_loss_sum / max(1, sum(1 for t in self.trades if t["pnl_net"] < 0))
        avg_mae_loss = self.mae_loss_sum / max(1, sum(1 for t in self.trades if t["pnl_net"] < 0))

        loss_avg_len = (self.loss_streak_len_total / self.loss_streak_count) if self.loss_streak_count else 0.0
        loss_avg_pnl = (self.loss_streak_pnl_total / self.loss_streak_count) if self.loss_streak_count else 0.0

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

    def _partial_bar(self, current_time: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
        bin_start = pd.Timestamp(current_time).floor(self.freq)
        partial_df = self.df.loc[bin_start:current_time]
        if partial_df.empty:
            return pd.DataFrame(), bin_start
        partial_row = pd.DataFrame(
            {
                "open": [float(partial_df["open"].iloc[0])],
                "high": [float(partial_df["high"].max())],
                "low": [float(partial_df["low"].min())],
                "close": [float(partial_df["close"].iloc[-1])],
                "volume": [float(partial_df["volume"].sum())],
            },
            index=[bin_start],
        )
        return partial_row, bin_start

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
    if preferred:
        symbols = set(df["symbol"].dropna().unique())
        if preferred in symbols:
            return preferred
    counts = df["symbol"].value_counts()
    return counts.index[0]


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


def apply_multipliers(signal: dict) -> None:
    sl_mult = CONFIG.get("DYNAMIC_SL_MULTIPLIER", 1.0)
    tp_mult = CONFIG.get("DYNAMIC_TP_MULTIPLIER", 1.0)
    old_sl = float(signal.get("sl_dist", MIN_SL))
    old_tp = float(signal.get("tp_dist", MIN_TP))
    signal["sl_dist"] = max(old_sl * sl_mult, MIN_SL)
    signal["tp_dist"] = max(old_tp * tp_mult, MIN_TP)


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
    trend_context: dict,
    cfg: Optional[dict],
) -> bool:
    if not cfg or not cfg.get("enabled", True):
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
) -> bool:
    """Require stronger ML confidence by volatility regime."""
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

    delta = reg_cfg.get("min_conf_delta", 0.0)
    if isinstance(delta, dict):
        delta = delta.get(side, delta.get("default", 0.0))
    try:
        delta_val = float(delta or 0.0)
    except Exception:
        delta_val = 0.0
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
    exit_str = exit_time.strftime("%H:%M") if exit_time else "-"
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
    return f"{entry_str} {exit_str} {side} {strategy} {mode} pnl={pnl:.2f} exit={reason}{suffix}"


def serialize_trade(trade: dict) -> dict:
    serialized = {}
    for key, value in trade.items():
        if isinstance(value, dt.datetime):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized


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
            "cancelled": stats.get("cancelled"),
        },
        "assumptions": {
            "contracts": CONTRACTS,
            "point_value": POINT_VALUE,
            "fees_per_20_contracts": FEES_PER_20_CONTRACTS,
            "bar_signal": "close",
            "entry": "next_open",
            "fast_mode": stats.get("fast_mode"),
        },
        "report": stats.get("report", ""),
        "trade_log": stats.get("trade_log", []),
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return report_path


def run_backtest(
    df: pd.DataFrame,
    start_time: dt.datetime,
    end_time: dt.datetime,
    mnq_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
    cancel_event: Optional["threading.Event"] = None,
    progress_every: int = 25,
) -> dict:
    configure_risk()
    CONFIG.setdefault("GEMINI", {})["enabled"] = False
    CONFIG["DYNAMIC_SL_MULTIPLIER"] = 1.0
    CONFIG["DYNAMIC_TP_MULTIPLIER"] = 1.0

    fast_cfg = CONFIG.get("BACKTEST_FAST_MODE", {}) or {}
    fast_enabled = bool(fast_cfg.get("enabled"))
    bar_stride = int(fast_cfg.get("bar_stride", 1) or 1)
    if bar_stride < 1:
        bar_stride = 1
    skip_mfe_mae = bool(fast_cfg.get("skip_mfe_mae", False))

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

    fast_strategies = [
        RegimeAdaptiveStrategy(),
        VIXReversionStrategy(),
    ]
    if ENABLE_DYNAMIC_ENGINE_1:
        dynamic_engine_strat = DynamicEngineStrategy()
        fast_strategies.append(dynamic_engine_strat)

    ml_strategy = MLPhysicsStrategy()
    smt_strategy = SMTStrategy()
    standard_strategies = [
        IntradayDipStrategy(),
        ConfluenceStrategy(),
        smt_strategy,
    ]
    if ml_strategy.model_loaded:
        standard_strategies.append(ml_strategy)

    loose_strategies = [OrbStrategy(), ICTModelStrategy()]

    rejection_filter = RejectionFilter()
    bank_filter = BankLevelQuarterFilter()
    chop_filter = ChopFilter(lookback=20)
    extension_filter = ExtensionFilter()
    trend_filter = TrendFilter()
    htf_fvg_filter = BacktestHTFFVGFilter()
    structure_blocker = DynamicStructureBlocker(lookback=50)
    regime_blocker = RegimeStructureBlocker(lookback=20)
    penalty_blocker = PenaltyBoxBlocker(lookback=50, tolerance=5.0, penalty_bars=3)
    memory_sr = MemorySRFilter(lookback_bars=300, zone_width=2.0, touch_threshold=2)
    directional_loss_blocker = DirectionalLossBlocker(consecutive_loss_limit=3, block_minutes=15)
    impulse_filter = ImpulseFilter(lookback=20, impulse_multiplier=2.5)
    legacy_filters = LegacyFilterSystem()
    filter_arbitrator = FilterArbitrator(confidence_threshold=0.6)

    news_filter = BacktestNewsFilter()

    continuation_manager = ContinuationRescueManager()
    continuation_allow_cfg = CONFIG.get("BACKTEST_CONTINUATION_ALLOWLIST", {})
    allowlist_mode = str(continuation_allow_cfg.get("mode", "reports") or "reports").lower()
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

    if mnq_df is None:
        mnq_df = pd.DataFrame()
    if vix_df is None:
        vix_df = pd.DataFrame()

    df = normalize_index(df, NY_TZ)
    mnq_df = normalize_index(mnq_df, NY_TZ)
    vix_df = normalize_index(vix_df, NY_TZ)
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=NY_TZ)
    else:
        start_time = start_time.astimezone(NY_TZ)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=NY_TZ)
    else:
        end_time = end_time.astimezone(NY_TZ)

    warmup_df = df[df.index < start_time].tail(WARMUP_BARS)
    test_df = df[(df.index >= start_time) & (df.index <= end_time)]
    if test_df.empty:
        raise ValueError("No bars in range to backtest.")
    if fast_enabled and bar_stride > 1:
        warmup_df = warmup_df.iloc[::bar_stride]
        test_df = test_df.iloc[::bar_stride]
    total_bars = len(test_df)

    full_df = pd.concat([warmup_df, test_df])
    vol_base = warmup_df if not warmup_df.empty else test_df
    if not vol_base.empty:
        try:
            volatility_filter.calibrate(vol_base)
        except Exception:
            pass

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

    if allowlist_mode == "csv_fast":
        continuation_allowlist, continuation_allow_stats = build_continuation_allowlist_from_df(
            full_df,
            trend_day_series,
            continuation_allow_cfg,
            continuation_allowed_regimes,
            continuation_confirm_cfg,
        )

    if continuation_allow_cfg.get("enabled", True):
        summary = (continuation_allow_stats or {}).get("summary", {})
        logging.info(
            "Continuation allowlist (%s): %s keys (reports used: %s)",
            allowlist_mode,
            len(continuation_allowlist or []),
            summary.get("reports_used", 0),
        )

    resample_cache_60 = ResampleCache(full_df, 60)
    resample_cache_240 = ResampleCache(full_df, 240)

    chop_client = BacktestClient(full_df)
    chop_analyzer = DynamicChopAnalyzer(chop_client)
    try:
        chop_analyzer.calibrate()
    except Exception:
        pass

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    trades = 0
    wins = 0
    losses = 0
    tracker = AttributionTracker()

    active_trade = None
    pending_entry = None
    pending_exit = False
    pending_exit_reason = None
    pending_loose_signals = {}
    opposite_signal_count = 0
    bar_count = 0
    processed_bars = 0
    cancelled = False
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

    def close_trade(exit_price: float, exit_time: dt.datetime, exit_reason: str = "unknown") -> None:
        nonlocal equity, peak, max_dd, trades, wins, losses, active_trade, opposite_signal_count
        if active_trade is None:
            return
        side = active_trade["side"]
        entry_price = active_trade["entry_price"]
        size = active_trade.get("size", CONTRACTS)
        pnl_points = compute_pnl_points(side, entry_price, exit_price)
        pnl_dollars = pnl_points * POINT_VALUE * size
        pnl_net = pnl_dollars - FEE_PER_TRADE
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
        entry_time = active_trade.get("entry_time")
        trade_record = {
            "strategy": active_trade.get("strategy", "Unknown"),
            "sub_strategy": active_trade.get("sub_strategy"),
            "side": side,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl_points": pnl_points,
            "pnl_dollars": pnl_dollars,
            "pnl_net": pnl_net,
            "sl_dist": active_trade.get("sl_dist", MIN_SL),
            "tp_dist": active_trade.get("tp_dist", MIN_TP),
            "mfe_points": active_trade.get("mfe_points", 0.0),
            "mae_points": active_trade.get("mae_points", 0.0),
            "entry_mode": active_trade.get("entry_mode", "standard"),
            "vol_regime": active_trade.get("vol_regime", "UNKNOWN"),
            "exit_reason": exit_reason,
            "bars_held": active_trade.get("bars_held", 0),
            "session": get_session_name(entry_time) if entry_time else "OFF",
            "rescue_from_strategy": active_trade.get("rescue_from_strategy"),
            "rescue_from_sub_strategy": active_trade.get("rescue_from_sub_strategy"),
            "rescue_trigger": active_trade.get("rescue_trigger"),
            "consensus_contributors": active_trade.get("consensus_contributors"),
            "bypassed_filters": active_trade.get("bypassed_filters"),
            "trend_day_tier": active_trade.get("trend_day_tier"),
            "trend_day_dir": active_trade.get("trend_day_dir"),
        }
        tracker.record_trade(trade_record)
        update_mom_rescue_score(trade_record, pnl_net, exit_time)
        update_hostile_day_on_close(trade_record.get("strategy"), pnl_points, exit_time)
        directional_loss_blocker.record_trade_result(side, pnl_points, exit_time)
        active_trade = None
        opposite_signal_count = 0

    def emit_progress(
        current_time: dt.datetime,
        current_price: float,
        force: bool = False,
        done: bool = False,
    ) -> None:
        if progress_cb is None:
            return
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
        payload["report"] = tracker.build_report()
        payload["recent_trades"] = [format_recent_trade(trade) for trade in tracker.recent_trades]
        try:
            progress_cb(payload)
        except Exception:
            pass

    def record_filter(name: str, kind: str = "block") -> None:
        tracker.record_filter(name, kind)
        if kind == "block" and name.startswith("TrendDayTier"):
            try:
                ts = current_time.strftime("%Y-%m-%d %H:%M")
            except Exception:
                ts = "N/A"
            try:
                price = f"{bar_close:.2f}"
            except Exception:
                price = "N/A"
            color = "\033[92m" if trend_day_dir == "up" else "\033[91m"
            print(f"{color}[TrendDay] {name} blocked counter-trend signal @ {ts} price={price}\033[0m")

    def continuation_core_trigger(trigger: Optional[str]) -> bool:
        if not trigger:
            return False
        if trigger.startswith("FilterStack"):
            return True
        return trigger in {"RegimeBlocker", "TrendFilter", "ChopFilter", "ExtensionFilter"}

    def continuation_rescue_allowed(
        signal: Optional[dict],
        side: str,
        current_time: dt.datetime,
        bar_close: float,
        history_df: pd.DataFrame,
    ) -> bool:
        if not signal:
            return False
        if continuation_signal_mode != "structure":
            key = parse_continuation_key(signal.get("strategy"))
            if continuation_allowlist is not None:
                if not key or key not in continuation_allowlist:
                    record_filter("ContinuationAllowlist")
                    return False
        if continuation_allowed_regimes:
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
        return True

    def open_trade(signal: dict, entry_price: float, entry_time: dt.datetime) -> None:
        nonlocal active_trade
        raw_sl = float(signal.get("sl_dist", MIN_SL))
        raw_tp = float(signal.get("tp_dist", MIN_TP))
        sl_dist = round_points_to_tick(max(raw_sl, MIN_SL))
        tp_dist = round_points_to_tick(max(raw_tp, MIN_TP))
        side = signal["side"]
        size = int(signal.get("size", CONTRACTS))
        stop_price = entry_price - sl_dist if side == "LONG" else entry_price + sl_dist
        active_trade = {
            "strategy": signal.get("strategy", "Unknown"),
            "sub_strategy": signal.get("sub_strategy"),
            "side": side,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "entry_bar": bar_count,
            "bars_held": 0,
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "requested_tp_dist": raw_tp,
            "requested_sl_dist": raw_sl,
            "size": size,
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
            "trend_day_tier": signal.get("trend_day_tier"),
            "trend_day_dir": signal.get("trend_day_dir"),
        }

    def check_stop_take(bar_high: float, bar_low: float) -> Optional[tuple[float, str]]:
        if active_trade is None:
            return None
        side = active_trade["side"]
        entry = active_trade["entry_price"]
        stop_price = active_trade.get("current_stop_price")
        tp_dist = active_trade.get("tp_dist", MIN_TP)
        take_price = entry + tp_dist if side == "LONG" else entry - tp_dist
        hit_stop = bar_low <= stop_price if side == "LONG" else bar_high >= stop_price
        hit_take = bar_high >= take_price if side == "LONG" else bar_low <= take_price
        if hit_stop and hit_take:
            return stop_price, "stop"
        if hit_stop:
            return stop_price, "stop"
        if hit_take:
            return take_price, "take"
        return None

    def check_early_exit(current_price: float) -> bool:
        if active_trade is None:
            return False
        active_trade["bars_held"] += 1
        strategy_name = active_trade.get("strategy", "")
        early_exit_config = CONFIG.get("EARLY_EXIT", {}).get(strategy_name, {})

        if active_trade["side"] == "LONG":
            is_green = current_price > active_trade["entry_price"]
        else:
            is_green = current_price < active_trade["entry_price"]

        was_green = active_trade.get("was_green")
        if was_green is not None and is_green != was_green:
            active_trade["profit_crosses"] = active_trade.get("profit_crosses", 0) + 1
        active_trade["was_green"] = is_green

        if not early_exit_config.get("enabled", False):
            return False

        exit_time = early_exit_config.get("exit_if_not_green_by", 50)
        exit_cross = early_exit_config.get("max_profit_crosses", 100)

        if active_trade["bars_held"] >= exit_time and not is_green:
            return True
        if active_trade.get("profit_crosses", 0) > exit_cross:
            return True
        return False

    def handle_signal(signal: dict) -> None:
        nonlocal pending_entry, pending_exit, pending_exit_reason, opposite_signal_count
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

    for i in range(len(full_df)):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        history_df = full_df.iloc[: i + 1]
        current_time = history_df.index[-1]
        mnq_slice = slice_df_upto(mnq_df, current_time)
        vix_slice = slice_df_upto(vix_df, current_time)
        current_session = get_session_name(current_time)
        trend_session = "NY" if current_session in ("NY_AM", "NY_PM") else current_session
        currbar = history_df.iloc[-1]
        bar_open = float(currbar["open"])
        bar_high = float(currbar["high"])
        bar_low = float(currbar["low"])
        bar_close = float(currbar["close"])
        processed_bars += 1
        last_time = current_time
        last_close = bar_close

        if TREND_DAY_ENABLED:
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
            day_key = td_day_index.iloc[i]
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

            ema50_val = td_ema50.iloc[i]
            ema200_val = td_ema200.iloc[i]
            atr_expansion = td_atr_exp.iloc[i]
            vwap_val = td_vwap.iloc[i]
            vwap_sigma = td_vwap_sigma.iloc[i]
            session_open = td_session_open.iloc[i]
            prior_session_low = td_prior_session_low.iloc[i]
            prior_session_high = td_prior_session_high.iloc[i]
            trend_up_alt = bool(td_trend_up_alt.iloc[i])
            trend_down_alt = bool(td_trend_down_alt.iloc[i])
            adx_strong_up = bool(td_adx_strong_up.iloc[i])
            adx_strong_down = bool(td_adx_strong_down.iloc[i])

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
                no_reclaim_down_t1 = bool(td_no_reclaim_down_t1.iloc[i])
                no_reclaim_up_t1 = bool(td_no_reclaim_up_t1.iloc[i])
                no_reclaim_down_t2 = bool(td_no_reclaim_down_t2.iloc[i])
                no_reclaim_up_t2 = bool(td_no_reclaim_up_t2.iloc[i])

                reclaim_down = bool(td_reclaim_down.iloc[i])
                reclaim_up = bool(td_reclaim_up.iloc[i])
            if pd.notna(ema50_val):
                ema_down = bar_close < ema50_val
                ema_up = bar_close > ema50_val
                confirm_down = confirm_down or ema_down
                confirm_up = confirm_up or ema_up
            if pd.notna(session_open):
                confirm_down = confirm_down or (bar_close < session_open)
                confirm_up = confirm_up or (bar_close > session_open)
            if pd.notna(prior_session_low) or pd.notna(prior_session_high):
                prev_close = float(history_df.iloc[-2]["close"]) if len(history_df) > 1 else bar_close
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

            tier1_down = atr_ok_t1 and displaced_down and no_reclaim_down_t1 and confirm_ok_down
            tier1_up = atr_ok_t1 and displaced_up and no_reclaim_up_t1 and confirm_ok_up

            if tier1_down:
                tier1_down_until = current_time + dt.timedelta(minutes=30)
            if tier1_up:
                tier1_up_until = current_time + dt.timedelta(minutes=30)

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

        in_test_range = current_time >= start_time

        if in_test_range:
            if pending_exit and active_trade is not None:
                close_trade(bar_open, current_time, pending_exit_reason or "reverse")
                pending_exit = False
                pending_exit_reason = None
            if pending_entry is not None:
                if trend_day_tier > 0 and trend_day_dir:
                    if (trend_day_dir == "down" and pending_entry["side"] == "LONG") or (
                        trend_day_dir == "up" and pending_entry["side"] == "SHORT"
                    ):
                        record_filter(f"TrendDayTier{trend_day_tier}")
                        pending_entry = None
                    else:
                        pending_entry["trend_day_tier"] = trend_day_tier
                        pending_entry["trend_day_dir"] = trend_day_dir
                if pending_entry is not None:
                    open_trade(pending_entry, bar_open, current_time)
                    pending_entry = None

        if active_trade is not None:
            entry_price = active_trade["entry_price"]
            if not skip_mfe_mae:
                if active_trade["side"] == "LONG":
                    mfe_points = bar_high - entry_price
                    mae_points = entry_price - bar_low
                else:
                    mfe_points = entry_price - bar_low
                    mae_points = bar_high - entry_price
                active_trade["mfe_points"] = max(active_trade.get("mfe_points", 0.0), mfe_points)
                active_trade["mae_points"] = max(active_trade.get("mae_points", 0.0), mae_points)

            exit_hit = check_stop_take(bar_high, bar_low)
            if exit_hit is not None:
                exit_price, exit_reason = exit_hit
                close_trade(exit_price, current_time, exit_reason)
                pending_exit = False
                pending_exit_reason = None

        rejection_filter.update(current_time, bar_high, bar_low, bar_close)
        bank_filter.update(current_time, bar_high, bar_low, bar_close)
        chop_filter.update(bar_high, bar_low, bar_close, current_time)
        extension_filter.update(bar_high, bar_low, bar_close, current_time)
        structure_blocker.update(history_df)
        regime_blocker.update(history_df)
        penalty_blocker.update(history_df)
        memory_sr.update(history_df)
        directional_loss_blocker.update_quarter(current_time)
        impulse_filter.update(history_df)

        if processed_bars % 60 == 0:
            df_60m = resample_cache_60.get_full(current_time)
            df_240m = resample_cache_240.get_full(current_time)
            htf_fvg_filter.update_structure_data(df_60m, df_240m)

        if not in_test_range:
            continue

        bar_count += 1

        if active_trade is not None and check_early_exit(bar_close):
            pending_exit = True
            pending_exit_reason = "early_exit"


        news_blocked, _ = news_filter.should_block_trade(current_time)
        if news_blocked:
            record_filter("NewsFilter")
            continue

        if ENABLE_HOSTILE_DAY_GUARD:
            current_day = current_time.astimezone(NY_TZ).date()
            if hostile_day_date != current_day:
                reset_hostile_day(current_day)

        df_60m = resample_cache_60.get_recent(current_time, chop_analyzer.LOOKBACK)
        is_choppy, chop_reason = chop_analyzer.check_market_state(history_df, df_60m_current=df_60m)
        allowed_chop_side = None
        if is_choppy:
            if "ALLOW_LONG_ONLY" in chop_reason:
                allowed_chop_side = "LONG"
            elif "ALLOW_SHORT_ONLY" in chop_reason:
                allowed_chop_side = "SHORT"
            else:
                record_filter("DynamicChop")
                continue

        ml_signal = None
        if ml_strategy.model_loaded:
            try:
                ml_signal = ml_strategy.on_bar(history_df, current_time)
            except Exception:
                ml_signal = None
        if ml_signal:
            disabled_sessions = set(CONFIG.get("ML_PHYSICS_BACKTEST_DISABLED_SESSIONS", []))
            if get_session_name(current_time) in disabled_sessions:
                ml_signal = None

        candidate_signals = []

        for strat in fast_strategies:
            strat_name = strat.__class__.__name__
            try:
                if strat_name == "VIXReversionStrategy":
                    signal = strat.on_bar(history_df, vix_slice)
                else:
                    signal = strat.on_bar(history_df)
            except Exception:
                signal = None

            if not signal:
                continue
            apply_multipliers(signal)
            signal.setdefault("strategy", strat_name)
            candidate_signals.append((1, strat, signal, strat_name))

        for strat in standard_strategies:
            strat_name = strat.__class__.__name__
            signal = None
            try:
                if strat_name == "MLPhysicsStrategy":
                    signal = ml_signal
                elif strat_name == "SMTStrategy":
                    signal = strat.on_bar(history_df, mnq_slice)
                else:
                    signal = strat.on_bar(history_df)
            except Exception:
                signal = None

            if not signal:
                continue
            apply_multipliers(signal)
            signal.setdefault("strategy", strat_name)
            candidate_signals.append((2, strat, signal, strat_name))

        candidate_signals.sort(key=lambda x: x[0])
        if hostile_day_active:
            candidate_signals = [
                (priority, strat, sig, s_name)
                for priority, strat, sig, s_name in candidate_signals
                if sig.get("strategy") not in ("DynamicEngine", "MLPhysics")
            ]

        session_name = get_session_name(current_time)
        vol_regime_current = None
        try:
            vol_regime_current, _, _ = volatility_filter.get_regime(history_df)
        except Exception:
            vol_regime_current = None

        direction_counts = {"LONG": 0, "SHORT": 0}
        smt_side = None
        for _, _, sig, s_name in candidate_signals:
            side = sig.get("side")
            if side in direction_counts:
                if str(sig.get("strategy", "")).startswith("MLPhysics"):
                    if not consensus_ml_ok(sig):
                        continue
                    if not ml_vol_regime_ok(sig, session_name, vol_regime_current):
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

        if not ENABLE_CONSENSUS_BYPASS:
            consensus_side = None

        consensus_tp_source = None
        consensus_tp_signal = None
        consensus_supporters: list[str] = []
        if consensus_side:
            consensus_candidates = [
                (sig, s_name)
                for _, _, sig, s_name in candidate_signals
                if sig.get("side") == consensus_side
                and (
                    not str(sig.get("strategy", "")).startswith("MLPhysics")
                    or (
                        consensus_ml_ok(sig)
                        and ml_vol_regime_ok(sig, session_name, vol_regime_current)
                    )
                )
            ]
            if consensus_candidates:
                consensus_tp_signal, consensus_tp_source = min(
                    consensus_candidates,
                    key=lambda item: item[0].get("tp_dist", float("inf")),
                )
                for sig, s_name in consensus_candidates:
                    label = format_strategy_label(sig, s_name)
                    if label not in consensus_supporters:
                        consensus_supporters.append(label)

        signal_executed = False

        for _, _, sig, strat_name in candidate_signals:
            signal = sig
            if consensus_side and signal.get("side") != consensus_side:
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
            origin_strategy = signal.get("strategy", strat_name)
            origin_sub_strategy = signal.get("sub_strategy")
            allow_rescue = not str(signal.get("strategy", "")).startswith("MLPhysics")
            is_rescued = False
            consensus_rescued = False
            consensus_bypass_allowed = True
            rescue_bypass_allowed = True
            bypassed_filters: list[str] = []
            primary_label = format_strategy_label(signal, strat_name)
            consensus_secondary = [
                label for label in consensus_supporters if label != primary_label
            ]

            if consensus_side and signal.get("side") == consensus_side:
                rescue_side = "SHORT" if signal["side"] == "LONG" else "LONG"

                def try_consensus_rescue(trigger: str) -> bool:
                    nonlocal signal, is_rescued, consensus_rescued, consensus_bypass_allowed
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
                    session_name = get_session_name(current_time)
                    if DISABLE_CONTINUATION_NY and session_name in ("NY_AM", "NY_PM"):
                        return False
                    if hostile_day_active:
                        return False
                    potential_rescue = continuation_manager.get_active_continuation_signal(
                        history_df,
                        current_time,
                        rescue_side,
                        current_price=bar_close,
                        trend_day_series=trend_day_series,
                        signal_mode=continuation_signal_mode,
                    )
                    if not continuation_rescue_allowed(
                        potential_rescue, rescue_side, current_time, bar_close, history_df
                    ):
                        potential_rescue = None
                    if not potential_rescue:
                        return False
                    if continuation_no_bypass and continuation_core_trigger(trigger):
                        return False
                    rescue_blocked, _ = trend_filter.should_block_trade(history_df, potential_rescue["side"])
                    if rescue_blocked:
                        return False
                    signal = potential_rescue
                    signal.setdefault("strategy", strat_name)
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
                        record_filter(f"TrendDayTier{trend_day_tier}")
                        continue

                if consensus_tp_signal is not None:
                    signal["tp_dist"] = consensus_tp_signal.get("tp_dist", signal.get("tp_dist", MIN_TP))
                    signal["sl_dist"] = consensus_tp_signal.get("sl_dist", signal.get("sl_dist", MIN_SL))
                is_feasible, _ = chop_analyzer.check_target_feasibility(
                    entry_price=bar_close,
                    side=signal["side"],
                    tp_distance=signal.get("tp_dist", MIN_TP),
                    df_1m=history_df,
                )
                if not is_feasible:
                    if try_consensus_rescue("TargetFeasibility"):
                        record_filter("TargetFeasibility", kind="rescue")
                    else:
                        record_filter("TargetFeasibility")
                        continue
                if not consensus_rescued:
                    regime_blocked, _ = regime_blocker.should_block_trade(signal["side"], bar_close)
                    if regime_blocked:
                        if try_consensus_rescue("RegimeBlocker"):
                            record_filter("RegimeBlocker", kind="rescue")
                        else:
                            record_filter("RegimeBlocker")
                            continue
                if not consensus_rescued:
                    dir_blocked, _ = directional_loss_blocker.should_block_trade(signal["side"], current_time)
                    if dir_blocked:
                        if try_consensus_rescue("DirectionalLossBlocker"):
                            record_filter("DirectionalLossBlocker", kind="rescue")
                        else:
                            record_filter("DirectionalLossBlocker")
                            continue
                if not consensus_rescued:
                    trend_blocked, _ = trend_filter.should_block_trade(history_df, signal["side"])
                    if trend_blocked:
                        if try_consensus_rescue("TrendFilter"):
                            record_filter("TrendFilter", kind="rescue")
                        else:
                            record_filter("TrendFilter")
                            continue
                if not consensus_rescued:
                    vol_regime, _, _ = volatility_filter.get_regime(history_df)
                    chop_blocked, chop_reason = chop_filter.should_block_trade(
                        signal["side"],
                        rejection_filter.prev_day_pm_bias,
                        bar_close,
                        "NEUTRAL",
                        vol_regime,
                    )
                    if chop_blocked:
                        if chop_reason:
                            reason_lc = str(chop_reason).lower()
                            if "wait for breakout" in reason_lc or "range too tight" in reason_lc:
                                record_filter("ChopFilter")
                                continue
                        if try_consensus_rescue("ChopFilter"):
                            record_filter("ChopFilter", kind="rescue")
                        else:
                            record_filter("ChopFilter")
                            continue
                if not consensus_rescued:
                    should_trade, vol_adj = check_volatility(
                        history_df,
                        signal.get("sl_dist", MIN_SL),
                        signal.get("tp_dist", MIN_TP),
                        base_size=CONTRACTS,
                        ts=current_time,
                    )
                    if not should_trade:
                        if try_consensus_rescue("VolatilityGuardrail"):
                            record_filter("VolatilityGuardrail", kind="rescue")
                        else:
                            record_filter("VolatilityGuardrail")
                            continue
                    signal["sl_dist"] = vol_adj.get("sl_dist", signal["sl_dist"])
                    signal["tp_dist"] = vol_adj.get("tp_dist", signal["tp_dist"])
                    if vol_adj.get("adjustment_applied", False):
                        signal["size"] = vol_adj["size"]
                    signal["vol_regime"] = vol_adj.get("regime", "UNKNOWN")
                    if not ml_vol_regime_ok(signal, session_name, signal["vol_regime"]):
                        record_filter("MLVolRegimeGuard")
                        continue
                if not consensus_rescued:
                    if consensus_secondary:
                        signal["consensus_contributors"] = consensus_secondary
                    consensus_bypassed: list[str] = []
                    rej_blocked, _ = rejection_filter.should_block_trade(signal["side"])
                    if rej_blocked:
                        consensus_bypassed.append("RejectionFilter")
                    range_bias_blocked = allowed_chop_side is not None and signal["side"] != allowed_chop_side
                    if range_bias_blocked:
                        consensus_bypassed.append("ChopRangeBias")
                    impulse_blocked, _ = impulse_filter.should_block_trade(signal["side"])
                    if impulse_blocked:
                        consensus_bypassed.append("ImpulseFilter")
                    ext_blocked, _ = extension_filter.should_block_trade(signal["side"])
                    if ext_blocked:
                        consensus_bypassed.append("ExtensionFilter")
                    bank_blocked, bank_reason = bank_filter.should_block_trade(signal["side"])
                    upg_trend_blocked, upg_trend_reason = trend_filter.should_block_trade(history_df, signal["side"])
                    legacy_blocked, _ = legacy_filters.check_trend(history_df, signal["side"])
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
                        signal["bypassed_filters"] = consensus_bypassed
                    if not is_rescued:
                        signal["entry_mode"] = "consensus"
                    handle_signal(signal)
                    signal_executed = True
                    break

            rescue_side = "SHORT" if signal["side"] == "LONG" else "LONG"
            session_name = get_session_name(current_time)
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
                potential_rescue, rescue_side, current_time, bar_close, history_df
            ):
                potential_rescue = None

            def try_rescue(trigger: str) -> bool:
                nonlocal signal, is_rescued, potential_rescue, rescue_bypass_allowed
                if mom_rescue_banned(current_time, origin_strategy, origin_sub_strategy):
                    record_filter("MomRescueBan")
                    return False
                if continuation_no_bypass and continuation_core_trigger(trigger):
                    return False
                if potential_rescue and not is_rescued:
                    rescue_blocked, _ = trend_filter.should_block_trade(history_df, potential_rescue["side"])
                    if rescue_blocked:
                        return False
                    signal = potential_rescue
                    signal.setdefault("strategy", strat_name)
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
                    record_filter(f"TrendDayTier{trend_day_tier}")
                    continue
                record_filter(f"TrendDayTier{trend_day_tier}", kind="rescue")

            is_feasible, _ = chop_analyzer.check_target_feasibility(
                entry_price=bar_close,
                side=signal["side"],
                tp_distance=signal.get("tp_dist", MIN_TP),
                df_1m=history_df,
            )
            if not is_feasible:
                record_filter("TargetFeasibility")
                continue

            rej_blocked, _ = rejection_filter.should_block_trade(signal["side"])
            range_bias_blocked = allowed_chop_side is not None and signal["side"] != allowed_chop_side
            if rej_blocked or range_bias_blocked:
                rescue_reasons = []
                if rej_blocked:
                    rescue_reasons.append("RejectionFilter")
                if range_bias_blocked:
                    rescue_reasons.append("ChopRangeBias")
                rescue_reason = "+".join(rescue_reasons) if rescue_reasons else "RejectionFilter"
                if not try_rescue(rescue_reason):
                    if rej_blocked:
                        record_filter("RejectionFilter")
                    if range_bias_blocked:
                        record_filter("ChopRangeBias")
                    continue
                if rej_blocked:
                    record_filter("RejectionFilter", kind="rescue")
                if range_bias_blocked:
                    record_filter("ChopRangeBias", kind="rescue")

            dir_blocked, _ = directional_loss_blocker.should_block_trade(signal["side"], current_time)
            if dir_blocked:
                if not try_rescue("DirectionalLossBlocker"):
                    record_filter("DirectionalLossBlocker")
                    continue
                record_filter("DirectionalLossBlocker", kind="rescue")

            impulse_blocked, _ = impulse_filter.should_block_trade(signal["side"])
            if impulse_blocked:
                if not try_rescue("ImpulseFilter"):
                    record_filter("ImpulseFilter")
                    continue
                record_filter("ImpulseFilter", kind="rescue")

            regime_blocked, _ = regime_blocker.should_block_trade(signal["side"], bar_close)
            if regime_blocked:
                record_filter("RegimeBlocker")
                continue

            upgraded_reasons = []
            struct_blocked, struct_reason = structure_blocker.should_block_trade(signal["side"], bar_close)
            if struct_blocked:
                upgraded_reasons.append(f"Structure: {struct_reason}")

            bank_blocked, bank_reason = bank_filter.should_block_trade(signal["side"])
            if bank_blocked:
                upgraded_reasons.append(f"Bank: {bank_reason}")

            upg_trend_blocked, upg_trend_reason = trend_filter.should_block_trade(history_df, signal["side"])
            if upg_trend_blocked:
                upgraded_reasons.append(f"Trend: {upg_trend_reason}")

            legacy_blocked, _ = legacy_filters.check_trend(history_df, signal["side"])

            final_blocked = False
            if legacy_blocked and upgraded_reasons:
                final_blocked = True
            elif not legacy_blocked and upgraded_reasons:
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
                if is_rescued:
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

            vol_regime, _, _ = volatility_filter.get_regime(history_df)
            chop_blocked, chop_reason = chop_filter.should_block_trade(
                signal["side"],
                rejection_filter.prev_day_pm_bias,
                bar_close,
                "NEUTRAL",
                vol_regime,
            )
            if chop_blocked:
                if chop_reason:
                    reason_lc = str(chop_reason).lower()
                    if "wait for breakout" in reason_lc or "range too tight" in reason_lc:
                        record_filter("ChopFilter")
                        continue
                if is_rescued:
                    if rescue_bypass_allowed:
                        record_filter("ChopFilter", kind="bypass")
                        bypassed_filters.append("ChopFilter")
                    else:
                        record_filter("ChopFilter")
                        continue
                else:
                    if not try_rescue("ChopFilter"):
                        record_filter("ChopFilter")
                        continue
                    record_filter("ChopFilter", kind="rescue")

            ext_blocked, _ = extension_filter.should_block_trade(signal["side"])
            if ext_blocked:
                if is_rescued:
                    if rescue_bypass_allowed:
                        record_filter("ExtensionFilter", kind="bypass")
                        bypassed_filters.append("ExtensionFilter")
                    else:
                        record_filter("ExtensionFilter")
                        continue
                else:
                    if not try_rescue("ExtensionFilter"):
                        record_filter("ExtensionFilter")
                        continue
                    record_filter("ExtensionFilter", kind="rescue")

            should_trade, vol_adj = check_volatility(
                history_df,
                signal.get("sl_dist", MIN_SL),
                signal.get("tp_dist", MIN_TP),
                base_size=CONTRACTS,
                ts=current_time,
            )
            if not should_trade:
                record_filter("VolatilityGuardrail")
                continue

            signal["sl_dist"] = vol_adj["sl_dist"]
            signal["tp_dist"] = vol_adj["tp_dist"]
            signal["vol_regime"] = vol_adj.get("regime", "UNKNOWN")
            if not ml_vol_regime_ok(signal, session_name, signal["vol_regime"]):
                record_filter("MLVolRegimeGuard")
                continue
            signal["entry_mode"] = "rescued" if is_rescued else "standard"

            if hostile_day_active and (
                signal.get("strategy") in ("DynamicEngine", "MLPhysics")
                or str(signal.get("strategy", "")).startswith("Continuation_")
            ):
                record_filter("HostileDayGate")
                continue
            if not ALLOW_DYNAMIC_ENGINE_SOLO and signal.get("strategy") == "DynamicEngine" and not is_rescued:
                record_filter("DynamicEngineSolo")
                continue

            if bypassed_filters:
                signal["bypassed_filters"] = bypassed_filters
            handle_signal(signal)
            signal_executed = True
            break

        if not signal_executed:
            for s_name in list(pending_loose_signals.keys()):
                pending = pending_loose_signals[s_name]
                pending["bar_count"] += 1
                if pending["bar_count"] < 1:
                    continue

                sig = pending["signal"]
                apply_multipliers(sig)
                sig.setdefault("sl_dist", MIN_SL)
                sig.setdefault("tp_dist", MIN_TP)
                sig.setdefault("strategy", s_name)
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

                is_feasible, _ = chop_analyzer.check_target_feasibility(
                    entry_price=bar_close,
                    side=sig["side"],
                    tp_distance=sig.get("tp_dist", MIN_TP),
                    df_1m=history_df,
                )
                if not is_feasible:
                    record_filter("TargetFeasibility")
                    del pending_loose_signals[s_name]
                    continue

                rej_blocked, _ = rejection_filter.should_block_trade(sig["side"])
                if rej_blocked:
                    record_filter("RejectionFilter")
                    del pending_loose_signals[s_name]
                    continue

                dir_blocked, _ = directional_loss_blocker.should_block_trade(sig["side"], current_time)
                if dir_blocked:
                    record_filter("DirectionalLossBlocker")
                    del pending_loose_signals[s_name]
                    continue

                impulse_blocked, _ = impulse_filter.should_block_trade(sig["side"])
                if impulse_blocked:
                    record_filter("ImpulseFilter")
                    del pending_loose_signals[s_name]
                    continue

                tp_dist = sig.get("tp_dist", MIN_TP)
                effective_tp_dist = tp_dist
                if allowed_chop_side is not None and sig["side"] == allowed_chop_side:
                    effective_tp_dist = tp_dist * 0.5

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

                struct_blocked, _ = structure_blocker.should_block_trade(sig["side"], bar_close)
                if struct_blocked:
                    record_filter("StructureBlocker")
                    del pending_loose_signals[s_name]
                    continue

                regime_blocked, _ = regime_blocker.should_block_trade(sig["side"], bar_close)
                if regime_blocked:
                    record_filter("RegimeBlocker")
                    del pending_loose_signals[s_name]
                    continue

                penalty_blocked, _ = penalty_blocker.should_block_trade(sig["side"], bar_close)
                if penalty_blocked:
                    record_filter("PenaltyBoxBlocker")
                    del pending_loose_signals[s_name]
                    continue

                mem_blocked, _ = memory_sr.should_block_trade(sig["side"], bar_close)
                if mem_blocked:
                    record_filter("MemorySRFilter")
                    del pending_loose_signals[s_name]
                    continue

                is_range_fade = allowed_chop_side is not None and sig["side"] == allowed_chop_side
                legacy_trend_blocked, legacy_trend_reason = legacy_filters.check_trend(history_df, sig["side"])
                upgraded_trend_blocked, upgraded_trend_reason = trend_filter.should_block_trade(
                    history_df,
                    sig["side"],
                    is_range_fade=is_range_fade,
                )

                if legacy_trend_blocked != upgraded_trend_blocked:
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

                trend_state = trend_state_from_reason(trend_reason)
                vol_regime, _, _ = volatility_filter.get_regime(history_df)
                chop_blocked, _ = chop_filter.should_block_trade(
                    sig["side"],
                    rejection_filter.prev_day_pm_bias,
                    bar_close,
                    trend_state=trend_state,
                    vol_regime=vol_regime,
                )
                if chop_blocked:
                    record_filter("ChopFilter")
                    del pending_loose_signals[s_name]
                    continue

                ext_blocked, _ = extension_filter.should_block_trade(sig["side"])
                if ext_blocked:
                    record_filter("ExtensionFilter")
                    del pending_loose_signals[s_name]
                    continue

                if trend_blocked:
                    record_filter("TrendFilter")
                    del pending_loose_signals[s_name]
                    continue

                should_trade, vol_adj = check_volatility(
                    history_df,
                    sig.get("sl_dist", MIN_SL),
                    sig.get("tp_dist", MIN_TP),
                    base_size=CONTRACTS,
                    ts=current_time,
                )
                if not should_trade:
                    record_filter("VolatilityGuardrail")
                    del pending_loose_signals[s_name]
                    continue

                sig["sl_dist"] = vol_adj["sl_dist"]
                sig["tp_dist"] = vol_adj["tp_dist"]
                sig["vol_regime"] = vol_adj.get("regime", "UNKNOWN")
                if not ml_vol_regime_ok(sig, session_name, sig["vol_regime"]):
                    record_filter("MLVolRegimeGuard")
                    del pending_loose_signals[s_name]
                    continue
                sig["entry_mode"] = "loose"

                handle_signal(sig)
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
                apply_multipliers(signal)
                signal.setdefault("strategy", s_name)
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
            close_trade(float(last_close), last_time, "cancelled")
        elif not test_df.empty:
            final_time = test_df.index[-1]
            final_close = float(test_df.iloc[-1]["close"])
            close_trade(final_close, final_time, "end_of_range")

    winrate = (wins / trades * 100.0) if trades else 0.0
    tracker.finalize_streaks()
    if progress_cb is not None and last_time is not None and last_close is not None:
        emit_progress(last_time, last_close, force=True, done=True)
    report_text = tracker.build_report(max_rows=50)
    return {
        "equity": equity,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "max_drawdown": max_dd,
        "cancelled": cancelled,
        "report": report_text,
        "trade_log": [serialize_trade(trade) for trade in tracker.trades],
        "fast_mode": {"enabled": fast_enabled, "bar_stride": bar_stride, "skip_mfe_mae": skip_mfe_mae},
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

    base_dir = Path(__file__).resolve().parent
    path_input = input(f"CSV path [{DEFAULT_CSV_NAME}]: ").strip()
    path = Path(path_input) if path_input else Path(DEFAULT_CSV_NAME)
    if not path.is_file():
        path = base_dir / path
    if not path.is_file():
        raise SystemExit(f"CSV not found: {path}")

    df = load_csv(path)
    if df.empty:
        raise SystemExit("No rows found in the CSV.")

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

    stats = run_backtest(df, start_time, end_time)

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


if __name__ == "__main__":
    main()
