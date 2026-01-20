import builtins
import datetime as dt
import logging
from pathlib import Path
from typing import Optional

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


def run_backtest(
    df: pd.DataFrame,
    start_time: dt.datetime,
    end_time: dt.datetime,
    mnq_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
) -> dict:
    configure_risk()
    CONFIG.setdefault("GEMINI", {})["enabled"] = False
    CONFIG["DYNAMIC_SL_MULTIPLIER"] = 1.0
    CONFIG["DYNAMIC_TP_MULTIPLIER"] = 1.0

    param_scaler.apply_scaling()
    refresh_target_symbol()

    dynamic_engine_strat = DynamicEngineStrategy()
    fast_strategies = [
        RegimeAdaptiveStrategy(),
        VIXReversionStrategy(),
        dynamic_engine_strat,
    ]

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

    NewsFilter.refresh_calendar = lambda self: None
    news_filter = NewsFilter()
    news_filter.calendar_blackouts = []
    news_filter.recent_events = []

    continuation_manager = ContinuationRescueManager()

    if mnq_df is None:
        mnq_df = pd.DataFrame()
    if vix_df is None:
        vix_df = pd.DataFrame()

    warmup_df = df[df.index < start_time].tail(WARMUP_BARS)
    test_df = df[(df.index >= start_time) & (df.index <= end_time)]
    if test_df.empty:
        raise ValueError("No bars in range to backtest.")

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

    active_trade = None
    pending_entry = None
    pending_exit = False
    pending_loose_signals = {}
    opposite_signal_count = 0
    bar_count = 0
    processed_bars = 0

    def close_trade(exit_price: float, exit_time: dt.datetime) -> None:
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
        directional_loss_blocker.record_trade_result(side, pnl_points, exit_time)
        circuit_breaker.update_trade_result(pnl_dollars)
        active_trade = None
        opposite_signal_count = 0

    def open_trade(signal: dict, entry_price: float) -> None:
        nonlocal active_trade
        sl_dist = float(signal.get("sl_dist", MIN_SL))
        tp_dist = float(signal.get("tp_dist", MIN_TP))
        side = signal["side"]
        stop_price = entry_price - sl_dist if side == "LONG" else entry_price + sl_dist
        active_trade = {
            "strategy": signal.get("strategy", "Unknown"),
            "side": side,
            "entry_price": entry_price,
            "entry_bar": bar_count,
            "bars_held": 0,
            "tp_dist": tp_dist,
            "sl_dist": sl_dist,
            "size": CONTRACTS,
            "current_stop_price": stop_price,
            "break_even_triggered": False,
            "profit_crosses": 0,
            "was_green": None,
        }

    def check_stop_take(bar_high: float, bar_low: float) -> Optional[float]:
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
            return stop_price
        if hit_stop:
            return stop_price
        if hit_take:
            return take_price
        return None

    def update_break_even(current_price: float) -> None:
        if active_trade is None:
            return
        be_config = CONFIG.get("BREAK_EVEN", {})
        if not be_config.get("enabled", False):
            return

        tp_dist = active_trade.get("tp_dist", MIN_TP)
        entry_price = active_trade["entry_price"]
        trigger_pct = be_config.get("trigger_pct", 0.40)
        trail_pct = be_config.get("trail_pct", 0.25)

        if active_trade["side"] == "LONG":
            current_profit = current_price - entry_price
        else:
            current_profit = entry_price - current_price

        profit_threshold = tp_dist * trigger_pct
        if current_profit < profit_threshold:
            return

        buffer = be_config.get("buffer_ticks", 1) * TICK_SIZE
        if not active_trade.get("break_even_triggered", False):
            new_stop_price = entry_price + buffer if active_trade["side"] == "LONG" else entry_price - buffer
        else:
            trail_amount = current_profit * trail_pct
            if active_trade["side"] == "LONG":
                new_stop_price = entry_price + trail_amount
            else:
                new_stop_price = entry_price - trail_amount
            new_stop_price = round(new_stop_price * 4) / 4

        current_stop = active_trade.get("current_stop_price", entry_price)
        should_modify = False
        if active_trade["side"] == "LONG":
            if new_stop_price > current_stop + TICK_SIZE:
                should_modify = True
        else:
            if new_stop_price < current_stop - TICK_SIZE:
                should_modify = True

        if should_modify:
            active_trade["break_even_triggered"] = True
            active_trade["current_stop_price"] = new_stop_price

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
        nonlocal pending_entry, pending_exit, opposite_signal_count
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
            if pending_entry is None:
                pending_entry = signal
            opposite_signal_count = 0

    for i in range(len(full_df)):
        history_df = full_df.iloc[: i + 1]
        current_time = history_df.index[-1]
        currbar = history_df.iloc[-1]
        bar_open = float(currbar["open"])
        bar_high = float(currbar["high"])
        bar_low = float(currbar["low"])
        bar_close = float(currbar["close"])
        processed_bars += 1

        in_test_range = current_time >= start_time

        if in_test_range:
            if pending_exit and active_trade is not None:
                close_trade(bar_open, current_time)
                pending_exit = False
            if pending_entry is not None:
                open_trade(pending_entry, bar_open)
                pending_entry = None

        if active_trade is not None:
            exit_price = check_stop_take(bar_high, bar_low)
            if exit_price is not None:
                close_trade(exit_price, current_time)
                pending_exit = False

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

        if active_trade is not None:
            update_break_even(bar_close)

        if not in_test_range:
            continue

        bar_count += 1

        if active_trade is not None and check_early_exit(bar_close):
            pending_exit = True

        cb_blocked, _ = circuit_breaker.should_block_trade()
        if cb_blocked:
            continue

        news_blocked, _ = news_filter.should_block_trade(current_time)
        if news_blocked:
            continue

        df_60m = resample_cache_60.get_recent(current_time, chop_analyzer.LOOKBACK)
        is_choppy, chop_reason = chop_analyzer.check_market_state(history_df, df_60m_current=df_60m)
        allowed_chop_side = None
        if is_choppy:
            if "ALLOW_LONG_ONLY" in chop_reason:
                allowed_chop_side = "LONG"
            elif "ALLOW_SHORT_ONLY" in chop_reason:
                allowed_chop_side = "SHORT"
            else:
                continue

        ml_signal = None
        if ml_strategy.model_loaded:
            try:
                ml_signal = ml_strategy.on_bar(history_df)
            except Exception:
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

        direction_counts = {"LONG": 0, "SHORT": 0}
        smt_side = None
        for _, _, sig, s_name in candidate_signals:
            side = sig.get("side")
            if side in direction_counts:
                direction_counts[side] += 1
            if s_name == "SMTStrategy":
                smt_side = side

        consensus_side = None
        max_count = max(direction_counts.values()) if direction_counts else 0
        if max_count >= 2:
            if direction_counts["LONG"] != direction_counts["SHORT"]:
                consensus_side = "LONG" if direction_counts["LONG"] > direction_counts["SHORT"] else "SHORT"
            elif smt_side:
                consensus_side = smt_side

        signal_executed = False

        for _, _, sig, strat_name in candidate_signals:
            signal = sig
            if consensus_side and signal.get("side") != consensus_side:
                continue

            signal.setdefault("sl_dist", MIN_SL)
            signal.setdefault("tp_dist", MIN_TP)
            signal.setdefault("strategy", strat_name)

            if consensus_side and signal.get("side") == consensus_side:
                regime_blocked, _ = regime_blocker.should_block_trade(signal["side"], bar_close)
                if regime_blocked:
                    continue
                vol_adj = volatility_filter.get_adjustments(
                    history_df,
                    signal.get("sl_dist", MIN_SL),
                    signal.get("tp_dist", MIN_TP),
                    base_size=CONTRACTS,
                    ts=current_time,
                )
                signal["sl_dist"] = vol_adj.get("sl_dist", signal["sl_dist"])
                signal["tp_dist"] = vol_adj.get("tp_dist", signal["tp_dist"])
                handle_signal(signal)
                signal_executed = True
                break

            is_rescued = False
            rescue_side = "SHORT" if signal["side"] == "LONG" else "LONG"
            potential_rescue = continuation_manager.get_active_continuation_signal(
                history_df, current_time, rescue_side
            )

            def try_rescue() -> bool:
                nonlocal signal, is_rescued, potential_rescue
                if potential_rescue and not is_rescued:
                    rescue_blocked, _ = trend_filter.should_block_trade(history_df, potential_rescue["side"])
                    if rescue_blocked:
                        return False
                    signal = potential_rescue
                    signal.setdefault("strategy", strat_name)
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
                continue

            rej_blocked, _ = rejection_filter.should_block_trade(signal["side"])
            range_bias_blocked = allowed_chop_side is not None and signal["side"] != allowed_chop_side
            if rej_blocked or range_bias_blocked:
                if not try_rescue():
                    continue

            dir_blocked, _ = directional_loss_blocker.should_block_trade(signal["side"], current_time)
            if dir_blocked:
                if not try_rescue():
                    continue

            impulse_blocked, _ = impulse_filter.should_block_trade(signal["side"])
            if impulse_blocked:
                if not try_rescue():
                    continue

            regime_blocked, _ = regime_blocker.should_block_trade(signal["side"], bar_close)
            if regime_blocked:
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

            if final_blocked and not is_rescued:
                if not try_rescue():
                    continue

            vol_regime, _, _ = volatility_filter.get_regime(history_df)
            chop_blocked, _ = chop_filter.should_block_trade(
                signal["side"],
                rejection_filter.prev_day_pm_bias,
                bar_close,
                "NEUTRAL",
                vol_regime,
            )
            if chop_blocked and not is_rescued:
                if not try_rescue():
                    continue

            ext_blocked, _ = extension_filter.should_block_trade(signal["side"])
            if ext_blocked and not is_rescued:
                if not try_rescue():
                    continue

            should_trade, vol_adj = check_volatility(
                history_df,
                signal.get("sl_dist", MIN_SL),
                signal.get("tp_dist", MIN_TP),
                base_size=CONTRACTS,
                ts=current_time,
            )
            if not should_trade:
                continue

            signal["sl_dist"] = vol_adj["sl_dist"]
            signal["tp_dist"] = vol_adj["tp_dist"]

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
                    del pending_loose_signals[s_name]
                    continue

                is_feasible, _ = chop_analyzer.check_target_feasibility(
                    entry_price=bar_close,
                    side=sig["side"],
                    tp_distance=sig.get("tp_dist", MIN_TP),
                    df_1m=history_df,
                )
                if not is_feasible:
                    del pending_loose_signals[s_name]
                    continue

                rej_blocked, _ = rejection_filter.should_block_trade(sig["side"])
                if rej_blocked:
                    del pending_loose_signals[s_name]
                    continue

                dir_blocked, _ = directional_loss_blocker.should_block_trade(sig["side"], current_time)
                if dir_blocked:
                    del pending_loose_signals[s_name]
                    continue

                impulse_blocked, _ = impulse_filter.should_block_trade(sig["side"])
                if impulse_blocked:
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
                    del pending_loose_signals[s_name]
                    continue

                struct_blocked, _ = structure_blocker.should_block_trade(sig["side"], bar_close)
                if struct_blocked:
                    del pending_loose_signals[s_name]
                    continue

                regime_blocked, _ = regime_blocker.should_block_trade(sig["side"], bar_close)
                if regime_blocked:
                    del pending_loose_signals[s_name]
                    continue

                penalty_blocked, _ = penalty_blocker.should_block_trade(sig["side"], bar_close)
                if penalty_blocked:
                    del pending_loose_signals[s_name]
                    continue

                mem_blocked, _ = memory_sr.should_block_trade(sig["side"], bar_close)
                if mem_blocked:
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
                    del pending_loose_signals[s_name]
                    continue

                ext_blocked, _ = extension_filter.should_block_trade(sig["side"])
                if ext_blocked:
                    del pending_loose_signals[s_name]
                    continue

                if trend_blocked:
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
                    del pending_loose_signals[s_name]
                    continue

                sig["sl_dist"] = vol_adj["sl_dist"]
                sig["tp_dist"] = vol_adj["tp_dist"]

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

    if active_trade is not None and not test_df.empty:
        final_time = test_df.index[-1]
        final_close = float(test_df.iloc[-1]["close"])
        close_trade(final_close, final_time)

    winrate = (wins / trades * 100.0) if trades else 0.0
    return {
        "equity": equity,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "max_drawdown": max_dd,
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


if __name__ == "__main__":
    main()
