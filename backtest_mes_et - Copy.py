import builtins
import datetime as dt
import json
import logging
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Callable, Optional

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
from vixmeanreversion import VIXReversionStrategy
from circuit_breaker import CircuitBreaker
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

SL_BUCKETS = [4.0, 6.0, 8.0, 10.0, 15.0]
TP_BUCKETS = [6.0, 8.0, 10.0, 15.0, 20.0, 30.0]
RR_BUCKETS = [1.0, 1.5, 2.0, 3.0]


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


def format_rows(title: str, rows: list[tuple], headers: list[str], max_rows: int = 10) -> str:
    lines = [title]
    if not rows:
        lines.append("  (none)")
        return "\n".join(lines)
    lines.append("  " + " | ".join(headers))
    for row in rows[:max_rows]:
        lines.append("  " + " | ".join(str(item) for item in row))
    return "\n".join(lines)


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

    def get_active_continuation_signal(self, df: pd.DataFrame, current_time, required_side: str):
        if df.empty:
            return None

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
                return {
                    "strategy": f"Continuation_{candidate_key}",
                    "side": required_side,
                    "tp_dist": strat.target if hasattr(strat, "target") else 6.0,
                    "sl_dist": strat.stop if hasattr(strat, "stop") else 4.0,
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


def compute_pnl_points(side: str, entry_price: float, exit_price: float) -> float:
    return exit_price - entry_price if side == "LONG" else entry_price - exit_price


def convert_points_to_live(points: float, is_tp: bool) -> float:
    abs_points = abs(points)
    threshold = 10.0 if is_tp else 6.0
    if abs_points >= threshold:
        abs_ticks = int(abs_points)
    else:
        abs_ticks = int(abs_points / 0.5)
    return abs_ticks * TICK_SIZE


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
    circuit_breaker = CircuitBreaker(max_daily_loss=600, max_consecutive_losses=7)
    directional_loss_blocker = DirectionalLossBlocker(consecutive_loss_limit=3, block_minutes=15)
    impulse_filter = ImpulseFilter(lookback=20, impulse_multiplier=2.5)
    legacy_filters = LegacyFilterSystem()
    filter_arbitrator = FilterArbitrator(confidence_threshold=0.6)

    news_filter = BacktestNewsFilter()

    continuation_manager = ContinuationRescueManager()

    if mnq_df is None:
        mnq_df = pd.DataFrame()
    if vix_df is None:
        vix_df = pd.DataFrame()

    warmup_df = df[df.index < start_time].tail(WARMUP_BARS)
    test_df = df[(df.index >= start_time) & (df.index <= end_time)]
    if test_df.empty:
        raise ValueError("No bars in range to backtest.")
    total_bars = len(test_df)

    full_df = pd.concat([warmup_df, test_df])
    vol_base = warmup_df if not warmup_df.empty else test_df
    if not vol_base.empty:
        try:
            volatility_filter.calibrate(vol_base)
        except Exception:
            pass

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
        }
        tracker.record_trade(trade_record)
        update_mom_rescue_score(trade_record, pnl_net, exit_time)
        update_hostile_day_on_close(trade_record.get("strategy"), pnl_points, exit_time)
        directional_loss_blocker.record_trade_result(side, pnl_points, exit_time)
        circuit_breaker.update_trade_result(pnl_dollars)
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

    def open_trade(signal: dict, entry_price: float, entry_time: dt.datetime) -> None:
        nonlocal active_trade
        sl_dist = float(signal.get("sl_dist", MIN_SL))
        tp_dist = float(signal.get("tp_dist", MIN_TP))
        sl_dist = convert_points_to_live(sl_dist, is_tp=False)
        tp_dist = convert_points_to_live(tp_dist, is_tp=True)
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
        currbar = history_df.iloc[-1]
        bar_open = float(currbar["open"])
        bar_high = float(currbar["high"])
        bar_low = float(currbar["low"])
        bar_close = float(currbar["close"])
        processed_bars += 1
        last_time = current_time
        last_close = bar_close

        in_test_range = current_time >= start_time

        if in_test_range:
            if pending_exit and active_trade is not None:
                close_trade(bar_open, current_time, pending_exit_reason or "reverse")
                pending_exit = False
                pending_exit_reason = None
            if pending_entry is not None:
                open_trade(pending_entry, bar_open, current_time)
                pending_entry = None

        if active_trade is not None:
            entry_price = active_trade["entry_price"]
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

        cb_blocked, _ = circuit_breaker.should_block_trade()
        if cb_blocked:
            record_filter("CircuitBreaker")
            continue

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
        if ml_signal and get_session_name(current_time) == "ASIA":
            ml_signal = None

        candidate_signals = []

        for strat in fast_strategies:
            strat_name = strat.__class__.__name__
            try:
                if strat_name == "VIXReversionStrategy":
                    signal = strat.on_bar(history_df, vix_df)
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
                    signal = strat.on_bar(history_df, mnq_df)
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

        direction_counts = {"LONG": 0, "SHORT": 0}
        smt_side = None
        for _, _, sig, s_name in candidate_signals:
            side = sig.get("side")
            if side in direction_counts:
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
        if consensus_side:
            consensus_candidates = [
                (sig, s_name)
                for _, _, sig, s_name in candidate_signals
                if sig.get("side") == consensus_side
            ]
            if consensus_candidates:
                consensus_tp_signal, consensus_tp_source = min(
                    consensus_candidates,
                    key=lambda item: item[0].get("tp_dist", float("inf")),
                )

        signal_executed = False

        for _, _, sig, strat_name in candidate_signals:
            signal = sig
            if consensus_side and signal.get("side") != consensus_side:
                continue

            signal.setdefault("sl_dist", MIN_SL)
            signal.setdefault("tp_dist", MIN_TP)
            signal.setdefault("strategy", strat_name)
            origin_strategy = signal.get("strategy", strat_name)
            origin_sub_strategy = signal.get("sub_strategy")
            allow_rescue = not str(signal.get("strategy", "")).startswith("MLPhysics")
            is_rescued = False
            consensus_rescued = False

            if consensus_side and signal.get("side") == consensus_side:
                rescue_side = "SHORT" if signal["side"] == "LONG" else "LONG"

                def try_consensus_rescue(trigger: str) -> bool:
                    nonlocal signal, is_rescued, consensus_rescued
                    if not allow_rescue:
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
                        history_df, current_time, rescue_side
                    )
                    if not potential_rescue:
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
                    is_rescued = True
                    consensus_rescued = True
                    return True

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
                    chop_blocked, _ = chop_filter.should_block_trade(
                        signal["side"],
                        rejection_filter.prev_day_pm_bias,
                        bar_close,
                        "NEUTRAL",
                        vol_regime,
                    )
                    if chop_blocked:
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
                if not consensus_rescued:
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
                    history_df, current_time, rescue_side
                )

            def try_rescue(trigger: str) -> bool:
                nonlocal signal, is_rescued, potential_rescue
                if mom_rescue_banned(current_time, origin_strategy, origin_sub_strategy):
                    record_filter("MomRescueBan")
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
                    potential_rescue = None
                    return True
                return False

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
                    for name in blocked_filters:
                        record_filter(name, kind="bypass")
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
            chop_blocked, _ = chop_filter.should_block_trade(
                signal["side"],
                rejection_filter.prev_day_pm_bias,
                bar_close,
                "NEUTRAL",
                vol_regime,
            )
            if chop_blocked:
                if is_rescued:
                    record_filter("ChopFilter", kind="bypass")
                else:
                    if not try_rescue("ChopFilter"):
                        record_filter("ChopFilter")
                        continue
                    record_filter("ChopFilter", kind="rescue")

            ext_blocked, _ = extension_filter.should_block_trade(signal["side"])
            if ext_blocked:
                if is_rescued:
                    record_filter("ExtensionFilter", kind="bypass")
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
