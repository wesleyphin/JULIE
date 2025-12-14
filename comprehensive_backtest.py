"""
JULIE001 Comprehensive Backtest Suite
======================================
Tests ALL features of the trading bot EXCEPT Machine Learning strategies.

Tested Components:
- Non-ML Strategies: Confluence, RegimeAdaptive, ICT Model, Intraday Dip, ORB
- All 7 Filters: Volatility, Rejection, Chop, Extension, BankLevel, HTF FVG, StructureBlocker
- Dynamic SLTP Engine (2880 combinations)
- Trade Parameters: Break-even, Early Exit rules
- Session Management and Time Hierarchy (320 combinations)

Data Format:
    Default data file: es_2023_2024_2025.csv
    Expected columns: Datetime,Open,High,Low,Close,Volume
    Datetime format: YYYY-MM-DD HH:MM:SS-05:00 (timezone-aware)

Usage:
    python comprehensive_backtest.py --start 2023-01-01 --end 2024-12-31
    python comprehensive_backtest.py --start 2024-01-01 --end 2024-06-30 --strategy Confluence
    python comprehensive_backtest.py --start 2024-01-01 --end 2024-12-31 --filter-test
    python comprehensive_backtest.py --synthetic --start 2024-01-01 --end 2024-12-31
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
import pytz
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import bot components
try:
    from confluence_strategy import ConfluenceStrategy
    from regime_strategy import RegimeAdaptiveStrategy
    from ict_model_strategy import ICTModelStrategy
    from intraday_dip_strategy import IntradayDipStrategy
    from orb_strategy import OrbStrategy
    from volatility_filter import HierarchicalVolatilityFilter, VOLATILITY_HIERARCHY
    from rejection_filter import RejectionFilter
    from chop_filter import ChopFilter, CHOP_THRESHOLDS
    from extension_filter import ExtensionFilter, EXTENSION_THRESHOLDS
    from bank_level_quarter_filter import BankLevelQuarterFilter
    from htf_fvg_filter import HTFFVGFilter
    from dynamic_structure_blocker import DynamicStructureBlocker
    from dynamic_sltp_params import DynamicSLTPEngine
    from config import CONFIG
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    logger.info("Make sure you're running from the JULIE directory")
    sys.exit(1)

# Constants
NY_TZ = pytz.timezone('US/Eastern')
TICK_SIZE = 0.25
TICK_VALUE = 12.50  # $12.50 per tick for ES


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    strategy: str = ""
    direction: str = ""  # LONG or SHORT
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    initial_sl: float = 0.0  # For break-even tracking
    initial_tp: float = 0.0
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    exit_reason: str = ""  # TP, SL, BREAK_EVEN, EARLY_EXIT, CLOSE_REVERSE
    bars_held: int = 0
    max_favorable: float = 0.0  # Max favorable excursion
    max_adverse: float = 0.0   # Max adverse excursion
    filters_passed: List[str] = field(default_factory=list)
    filters_blocked: List[str] = field(default_factory=list)
    hierarchy_key: str = ""  # Q1_W2_MON_NY_AM
    session: str = ""
    be_triggered: bool = False  # Break-even was triggered


@dataclass
class BacktestMetrics:
    """Aggregated backtest metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    total_pnl_points: float = 0.0
    total_pnl_dollars: float = 0.0

    gross_profit: float = 0.0
    gross_loss: float = 0.0

    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0

    max_drawdown_points: float = 0.0
    max_drawdown_dollars: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    avg_bars_held: float = 0.0
    avg_mae: float = 0.0  # Average max adverse excursion
    avg_mfe: float = 0.0  # Average max favorable excursion

    be_triggered_count: int = 0
    early_exit_count: int = 0

    # By exit reason
    exits_by_tp: int = 0
    exits_by_sl: int = 0
    exits_by_be: int = 0
    exits_by_early: int = 0
    exits_by_reverse: int = 0


@dataclass
class FilterStats:
    """Statistics for a single filter."""
    name: str
    signals_checked: int = 0
    signals_blocked: int = 0
    block_rate: float = 0.0
    blocked_would_win: int = 0
    blocked_would_lose: int = 0
    passed_won: int = 0
    passed_lost: int = 0
    value_added_points: float = 0.0  # Blocked losses - Blocked wins


# ============================================================
# SYNTHETIC DATA GENERATOR (for testing without API)
# ============================================================
class SyntheticDataGenerator:
    """Generates realistic ES futures data for backtesting."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate(self, start_date: str, end_date: str,
                 base_price: float = 5000.0) -> pd.DataFrame:
        """
        Generate 1-minute OHLCV data for ES futures.

        Models:
        - Session volatility patterns (Asia < London < NY)
        - Overnight gaps
        - Mean reversion at extremes
        - Trending and ranging regimes
        """
        start = pd.Timestamp(start_date, tz=NY_TZ)
        end = pd.Timestamp(end_date, tz=NY_TZ)

        # Generate all trading minutes
        dates = pd.date_range(start=start, end=end, freq='1min', tz=NY_TZ)

        # Filter to trading hours only (6pm - 5pm next day, Sun-Fri)
        trading_dates = []
        for dt in dates:
            dow = dt.weekday()
            hour = dt.hour

            # Skip Saturday and Sunday daytime
            if dow == 5:  # Saturday
                continue
            if dow == 6 and hour < 18:  # Sunday before 6pm
                continue

            # Skip 5pm-6pm daily maintenance
            if hour == 17:
                continue

            trading_dates.append(dt)

        if not trading_dates:
            logger.warning("No trading dates generated")
            return pd.DataFrame()

        n_bars = len(trading_dates)
        logger.info(f"Generating {n_bars} 1-minute bars from {start_date} to {end_date}")

        # Initialize price series
        prices = np.zeros(n_bars)
        prices[0] = base_price

        # Generate returns with session-specific volatility
        for i in range(1, n_bars):
            dt = trading_dates[i]
            hour = dt.hour

            # Session-based volatility
            if 18 <= hour or hour < 3:  # ASIA
                vol = 0.00008
            elif 3 <= hour < 8:  # LONDON
                vol = 0.00012
            elif 8 <= hour < 12:  # NY_AM
                vol = 0.00020
            elif 12 <= hour < 17:  # NY_PM
                vol = 0.00015
            else:
                vol = 0.00005

            # Add some autocorrelation (trending)
            if i > 1:
                prev_return = (prices[i-1] - prices[i-2]) / prices[i-2]
                momentum = prev_return * 0.1
            else:
                momentum = 0

            # Mean reversion at extremes
            deviation = (prices[i-1] - base_price) / base_price
            mean_rev = -deviation * 0.001

            # Generate return
            noise = np.random.randn() * vol
            ret = momentum + mean_rev + noise

            prices[i] = prices[i-1] * (1 + ret)

        # Generate OHLC from prices
        data = []
        for i in range(n_bars):
            price = prices[i]
            dt = trading_dates[i]
            hour = dt.hour

            # Session-based range
            if 18 <= hour or hour < 3:
                range_pct = 0.0003
            elif 3 <= hour < 8:
                range_pct = 0.0004
            elif 8 <= hour < 12:
                range_pct = 0.0006
            else:
                range_pct = 0.0005

            bar_range = price * range_pct

            # Random OHLC within range
            open_price = price + np.random.uniform(-bar_range/4, bar_range/4)
            close_price = price + np.random.uniform(-bar_range/4, bar_range/4)

            high = max(open_price, close_price) + np.random.uniform(0, bar_range/2)
            low = min(open_price, close_price) - np.random.uniform(0, bar_range/2)

            # Round to tick size
            open_price = round(open_price / TICK_SIZE) * TICK_SIZE
            high = round(high / TICK_SIZE) * TICK_SIZE
            low = round(low / TICK_SIZE) * TICK_SIZE
            close_price = round(close_price / TICK_SIZE) * TICK_SIZE

            # Ensure OHLC consistency
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Volume based on session
            if 8 <= hour < 16:
                volume = np.random.randint(5000, 20000)
            else:
                volume = np.random.randint(1000, 5000)

            data.append({
                'timestamp': dt,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        logger.info(f"Generated {len(df)} bars. Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
        return df


# ============================================================
# BACKTEST ENGINE
# ============================================================
class ComprehensiveBacktester:
    """
    Full-featured backtesting engine for JULIE trading bot.
    Tests all non-ML strategies and filters.
    """

    def __init__(self,
                 enable_filters: bool = True,
                 enable_break_even: bool = True,
                 enable_early_exit: bool = True):

        self.enable_filters = enable_filters
        self.enable_break_even = enable_break_even
        self.enable_early_exit = enable_early_exit

        # Initialize strategies
        self.strategies = {
            'Confluence': ConfluenceStrategy(),
            'RegimeAdaptive': RegimeAdaptiveStrategy(),
            'ICTModel': ICTModelStrategy(),
            'IntradayDip': IntradayDipStrategy(),
            'ORB': OrbStrategy()
        }

        # Initialize filters
        self.filters = {
            'Volatility': HierarchicalVolatilityFilter(),
            'Rejection': RejectionFilter(),
            'Chop': ChopFilter(),
            'Extension': ExtensionFilter(),
            'BankLevel': BankLevelQuarterFilter(),
            'HTFFVG': HTFFVGFilter(),
            'StructureBlocker': DynamicStructureBlocker()
        }

        # Dynamic SLTP engine
        self.sltp_engine = DynamicSLTPEngine()

        # Trade tracking
        self.trades: List[Trade] = []
        self.current_trade: Optional[Trade] = None
        self.equity_curve: List[float] = []

        # Filter statistics
        self.filter_stats: Dict[str, FilterStats] = {
            name: FilterStats(name=name) for name in self.filters.keys()
        }

        # Strategy statistics
        self.strategy_metrics: Dict[str, BacktestMetrics] = {
            name: BacktestMetrics() for name in self.strategies.keys()
        }

        # Early exit config from CONFIG
        self.early_exit_config = CONFIG.get('EARLY_EXIT', {})
        self.break_even_config = CONFIG.get('BREAK_EVEN', {
            'enabled': True,
            'trigger_pct': 0.4,
            'buffer_ticks': 1
        })

        # Session tracking
        self.session_stats: Dict[str, BacktestMetrics] = {
            session: BacktestMetrics() for session in ['ASIA', 'LONDON', 'NY_AM', 'NY_PM']
        }

        # Hierarchy stats (Q_W_DOW_Session)
        self.hierarchy_stats: Dict[str, BacktestMetrics] = defaultdict(BacktestMetrics)

    def get_session(self, hour: int) -> str:
        """Determine trading session from hour."""
        if 18 <= hour or hour < 3:
            return 'ASIA'
        elif 3 <= hour < 8:
            return 'LONDON'
        elif 8 <= hour < 12:
            return 'NY_AM'
        elif 12 <= hour < 17:
            return 'NY_PM'
        return 'CLOSED'

    def get_hierarchy_key(self, ts: datetime) -> str:
        """Get full hierarchy key: Q1_W2_MON_NY_AM."""
        return self.sltp_engine.get_hierarchy_key(ts)

    def check_filters(self, signal: str, df: pd.DataFrame,
                      current_price: float, dt: datetime) -> Tuple[bool, List[str], List[str]]:
        """
        Run signal through all filters.

        Returns:
            (passed, filters_passed, filters_blocked)
        """
        if not self.enable_filters:
            return True, ['ALL_DISABLED'], []

        passed_filters = []
        blocked_filters = []

        # 1. Volatility Filter
        try:
            vol_filter = self.filters['Volatility']
            blocked, reason = vol_filter.should_block_trade(signal, dt)
            self.filter_stats['Volatility'].signals_checked += 1
            if blocked:
                blocked_filters.append(f"Volatility: {reason}")
                self.filter_stats['Volatility'].signals_blocked += 1
            else:
                passed_filters.append('Volatility')
        except Exception as e:
            logger.debug(f"Volatility filter error: {e}")
            passed_filters.append('Volatility')

        # 2. Rejection Filter
        try:
            rej_filter = self.filters['Rejection']
            if len(df) > 0:
                bar = df.iloc[-1]
                rej_filter.update(dt, bar['high'], bar['low'], bar['close'])
            # Convert signal from LONG/SHORT to BUY/SELL for rejection filter
            direction = 'BUY' if signal == 'LONG' else 'SELL'
            blocked, reason = rej_filter.should_block_trade(direction)
            self.filter_stats['Rejection'].signals_checked += 1
            if blocked:
                blocked_filters.append(f"Rejection: {reason}")
                self.filter_stats['Rejection'].signals_blocked += 1
            else:
                passed_filters.append('Rejection')
        except Exception as e:
            logger.debug(f"Rejection filter error: {e}")
            passed_filters.append('Rejection')

        # 3. Chop Filter
        try:
            chop_filter = self.filters['Chop']
            if len(df) > 0:
                bar = df.iloc[-1]
                chop_filter.update(bar['high'], bar['low'], bar['close'], dt)
            blocked, reason = chop_filter.should_block_trade(signal)
            self.filter_stats['Chop'].signals_checked += 1
            if blocked:
                blocked_filters.append(f"Chop: {reason}")
                self.filter_stats['Chop'].signals_blocked += 1
            else:
                passed_filters.append('Chop')
        except Exception as e:
            logger.debug(f"Chop filter error: {e}")
            passed_filters.append('Chop')

        # 4. Extension Filter
        try:
            ext_filter = self.filters['Extension']
            if len(df) > 0:
                bar = df.iloc[-1]
                ext_filter.update(bar['high'], bar['low'], bar['close'], dt)
            blocked, reason = ext_filter.should_block_trade(signal)
            self.filter_stats['Extension'].signals_checked += 1
            if blocked:
                blocked_filters.append(f"Extension: {reason}")
                self.filter_stats['Extension'].signals_blocked += 1
            else:
                passed_filters.append('Extension')
        except Exception as e:
            logger.debug(f"Extension filter error: {e}")
            passed_filters.append('Extension')

        # 5. Bank Level Filter
        try:
            bank_filter = self.filters['BankLevel']
            if len(df) > 0:
                bar = df.iloc[-1]
                bank_filter.update(dt, bar['high'], bar['low'], bar['close'])
            blocked, reason = bank_filter.should_block_trade(signal)
            self.filter_stats['BankLevel'].signals_checked += 1
            if blocked:
                blocked_filters.append(f"BankLevel: {reason}")
                self.filter_stats['BankLevel'].signals_blocked += 1
            else:
                passed_filters.append('BankLevel')
        except Exception as e:
            logger.debug(f"BankLevel filter error: {e}")
            passed_filters.append('BankLevel')

        # 6. HTF FVG Filter
        try:
            fvg_filter = self.filters['HTFFVG']
            # Resample to 1H for FVG detection
            df_1h = df.resample('1H').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna() if len(df) > 60 else None

            blocked, reason = fvg_filter.check_signal_blocked(signal, current_price, df_1h)
            self.filter_stats['HTFFVG'].signals_checked += 1
            if blocked:
                blocked_filters.append(f"HTFFVG: {reason}")
                self.filter_stats['HTFFVG'].signals_blocked += 1
            else:
                passed_filters.append('HTFFVG')
        except Exception as e:
            logger.debug(f"HTFFVG filter error: {e}")
            passed_filters.append('HTFFVG')

        # 7. Structure Blocker
        try:
            struct_filter = self.filters['StructureBlocker']
            if len(df) > 50:
                struct_filter.update(df.tail(50))
            blocked, reason = struct_filter.should_block_trade(signal, current_price)
            self.filter_stats['StructureBlocker'].signals_checked += 1
            if blocked:
                blocked_filters.append(f"Structure: {reason}")
                self.filter_stats['StructureBlocker'].signals_blocked += 1
            else:
                passed_filters.append('StructureBlocker')
        except Exception as e:
            logger.debug(f"StructureBlocker filter error: {e}")
            passed_filters.append('StructureBlocker')

        # Signal passes if no filters blocked it
        passed = len(blocked_filters) == 0
        return passed, passed_filters, blocked_filters

    def calculate_sltp(self, strategy_name: str, direction: str,
                       entry_price: float, df: pd.DataFrame,
                       dt: datetime) -> Tuple[float, float]:
        """
        Calculate dynamic SL/TP using the SLTP engine.

        Returns (stop_loss_price, take_profit_price)
        """
        full_strategy_name = f"{strategy_name}_{direction}"

        try:
            result = self.sltp_engine.calculate_sltp(full_strategy_name, df, dt)

            sl_distance = result.get('sl_points', 5.0)
            tp_distance = result.get('tp_points', 10.0)

        except Exception as e:
            logger.debug(f"SLTP calculation error for {full_strategy_name}: {e}")
            # Default values
            sl_distance = 5.0
            tp_distance = 10.0

        # Calculate price levels
        if direction == 'LONG':
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SHORT
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Round to tick size
        stop_loss = round(stop_loss / TICK_SIZE) * TICK_SIZE
        take_profit = round(take_profit / TICK_SIZE) * TICK_SIZE

        return stop_loss, take_profit

    def check_break_even(self, trade: Trade, current_high: float,
                         current_low: float) -> bool:
        """
        Check and apply break-even logic.

        Returns True if break-even was triggered.
        """
        if not self.enable_break_even:
            return False

        if trade.be_triggered:
            return False  # Already triggered

        be_config = self.break_even_config
        if not be_config.get('enabled', True):
            return False

        trigger_pct = be_config.get('trigger_pct', 0.4)
        buffer_ticks = be_config.get('buffer_ticks', 1)
        buffer = buffer_ticks * TICK_SIZE

        if trade.direction == 'LONG':
            tp_distance = trade.initial_tp - trade.entry_price
            trigger_price = trade.entry_price + (tp_distance * trigger_pct)

            if current_high >= trigger_price:
                # Move stop to entry + buffer
                trade.stop_loss = trade.entry_price + buffer
                trade.be_triggered = True
                return True
        else:  # SHORT
            tp_distance = trade.entry_price - trade.initial_tp
            trigger_price = trade.entry_price - (tp_distance * trigger_pct)

            if current_low <= trigger_price:
                trade.stop_loss = trade.entry_price - buffer
                trade.be_triggered = True
                return True

        return False

    def check_early_exit(self, trade: Trade, bars_since_entry: int,
                         green_red_crosses: int) -> Tuple[bool, str]:
        """
        Check early exit conditions.

        Returns (should_exit, reason)
        """
        if not self.enable_early_exit:
            return False, ""

        strategy = trade.strategy
        config = self.early_exit_config.get(strategy, {})

        # Check bars without profit
        exit_if_not_green = config.get('exit_if_not_green_by', 30)
        if bars_since_entry >= exit_if_not_green:
            if trade.pnl_points <= 0:
                return True, f"EARLY_EXIT: Not green by bar {exit_if_not_green}"

        # Check profit crosses
        max_crosses = config.get('max_profit_crosses', 10)
        if green_red_crosses >= max_crosses:
            return True, f"EARLY_EXIT: {green_red_crosses} green/red crosses (max: {max_crosses})"

        return False, ""

    def process_bar(self, bar_idx: int, df: pd.DataFrame,
                    active_strategies: List[str] = None) -> Optional[Trade]:
        """
        Process a single bar for signal generation and trade management.
        """
        if bar_idx < 100:  # Need history
            return None

        current_bar = df.iloc[bar_idx]
        dt = df.index[bar_idx]

        session = self.get_session(dt.hour)
        if session == 'CLOSED':
            return None

        # Get lookback data
        lookback_df = df.iloc[max(0, bar_idx-500):bar_idx+1].copy()
        current_price = current_bar['close']

        # Manage existing trade
        if self.current_trade is not None:
            trade = self.current_trade
            trade.bars_held += 1

            # Update MAE/MFE
            if trade.direction == 'LONG':
                current_pnl = current_bar['close'] - trade.entry_price
                trade.max_favorable = max(trade.max_favorable, current_bar['high'] - trade.entry_price)
                trade.max_adverse = max(trade.max_adverse, trade.entry_price - current_bar['low'])
            else:
                current_pnl = trade.entry_price - current_bar['close']
                trade.max_favorable = max(trade.max_favorable, trade.entry_price - current_bar['low'])
                trade.max_adverse = max(trade.max_adverse, current_bar['high'] - trade.entry_price)

            trade.pnl_points = current_pnl

            # Check break-even
            self.check_break_even(trade, current_bar['high'], current_bar['low'])

            # Check stop loss
            hit_sl = False
            if trade.direction == 'LONG':
                if current_bar['low'] <= trade.stop_loss:
                    hit_sl = True
                    trade.exit_price = trade.stop_loss
            else:
                if current_bar['high'] >= trade.stop_loss:
                    hit_sl = True
                    trade.exit_price = trade.stop_loss

            if hit_sl:
                trade.exit_time = dt
                if trade.be_triggered:
                    trade.exit_reason = 'BREAK_EVEN'
                else:
                    trade.exit_reason = 'STOP_LOSS'
                self._close_trade(trade)
                return None

            # Check take profit
            hit_tp = False
            if trade.direction == 'LONG':
                if current_bar['high'] >= trade.take_profit:
                    hit_tp = True
                    trade.exit_price = trade.take_profit
            else:
                if current_bar['low'] <= trade.take_profit:
                    hit_tp = True
                    trade.exit_price = trade.take_profit

            if hit_tp:
                trade.exit_time = dt
                trade.exit_reason = 'TAKE_PROFIT'
                self._close_trade(trade)
                return None

            # Check early exit (simplified - would need to track crosses properly)
            should_exit, reason = self.check_early_exit(
                trade, trade.bars_held,
                green_red_crosses=0  # Simplified
            )
            if should_exit:
                trade.exit_time = dt
                trade.exit_price = current_price
                trade.exit_reason = 'EARLY_EXIT'
                self._close_trade(trade)
                return None

            return None  # Trade still open

        # No position - look for signals
        strategies_to_check = active_strategies or list(self.strategies.keys())

        for strategy_name in strategies_to_check:
            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                continue

            try:
                # Generate signal
                signal_dict = strategy.on_bar(lookback_df)

                if signal_dict is None:
                    continue

                # Extract direction from signal dict
                direction = signal_dict.get('side', 'NONE').upper()
                if direction not in ['LONG', 'SHORT']:
                    logger.debug(f"{dt}: {strategy_name} returned invalid direction: {direction}")
                    continue

                logger.info(f"{dt}: {strategy_name} generated {direction} signal")

                # Run through filters
                passed, passed_filters, blocked_filters = self.check_filters(
                    direction, lookback_df, current_price, dt
                )

                if not passed:
                    logger.info(f"{dt}: {strategy_name} {direction} blocked by {blocked_filters}")
                    continue

                # Calculate SL/TP
                stop_loss, take_profit = self.calculate_sltp(
                    strategy_name, direction, current_price, lookback_df, dt
                )

                # Create trade
                hierarchy_key = self.get_hierarchy_key(dt)

                trade = Trade(
                    entry_time=dt,
                    strategy=strategy_name,
                    direction=direction,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    initial_sl=stop_loss,
                    initial_tp=take_profit,
                    filters_passed=passed_filters,
                    filters_blocked=blocked_filters,
                    hierarchy_key=hierarchy_key,
                    session=session
                )

                self.current_trade = trade
                logger.info(f"{dt}: ENTRY {strategy_name} {direction} @ {current_price:.2f} "
                           f"SL: {stop_loss:.2f} TP: {take_profit:.2f}")
                return trade

            except Exception as e:
                logger.warning(f"Error in {strategy_name}: {e}", exc_info=True)
                continue

        return None

    def _close_trade(self, trade: Trade):
        """Finalize and record a closed trade."""
        # Calculate final PnL
        if trade.direction == 'LONG':
            trade.pnl_points = trade.exit_price - trade.entry_price
        else:
            trade.pnl_points = trade.entry_price - trade.exit_price

        trade.pnl_dollars = trade.pnl_points * (TICK_VALUE / TICK_SIZE)

        self.trades.append(trade)
        self.current_trade = None

        # Update equity curve
        if self.equity_curve:
            self.equity_curve.append(self.equity_curve[-1] + trade.pnl_dollars)
        else:
            self.equity_curve.append(trade.pnl_dollars)

        logger.debug(f"{trade.exit_time}: EXIT {trade.strategy} {trade.direction} @ {trade.exit_price:.2f} "
                   f"PnL: {trade.pnl_points:.2f}pts ({trade.exit_reason})")

    def run(self, df: pd.DataFrame,
            strategies: List[str] = None,
            progress_interval: int = 10000) -> BacktestMetrics:
        """
        Run full backtest on provided data.

        Args:
            df: OHLCV DataFrame with datetime index
            strategies: List of strategy names to test (None = all)
            progress_interval: Log progress every N bars

        Returns:
            BacktestMetrics object with results
        """
        logger.info(f"Starting backtest on {len(df)} bars...")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Strategies: {strategies or 'ALL'}")
        logger.info(f"Filters enabled: {self.enable_filters}")
        logger.info(f"Break-even enabled: {self.enable_break_even}")
        logger.info(f"Early exit enabled: {self.enable_early_exit}")

        # Reset state
        self.trades = []
        self.current_trade = None
        self.equity_curve = []

        # Reset filter stats
        for stats in self.filter_stats.values():
            stats.signals_checked = 0
            stats.signals_blocked = 0

        # Track signal generation
        signal_count = 0
        bars_processed = 0

        # Process each bar
        for i in range(len(df)):
            if i > 0 and i % progress_interval == 0:
                logger.info(f"Processed {i}/{len(df)} bars ({i*100/len(df):.1f}%) - {len(self.trades)} trades so far")

            bars_processed += 1
            self.process_bar(i, df, strategies)

        # Close any open trade at end
        if self.current_trade is not None:
            self.current_trade.exit_time = df.index[-1]
            self.current_trade.exit_price = df.iloc[-1]['close']
            self.current_trade.exit_reason = 'END_OF_DATA'
            self._close_trade(self.current_trade)

        # Calculate metrics
        metrics = self._calculate_metrics()

        logger.info(f"\n{'='*60}")
        logger.info(f"Backtest complete: {bars_processed} bars processed, {len(self.trades)} trades found")
        logger.info(f"{'='*60}\n")

        return metrics

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics."""
        metrics = BacktestMetrics()

        if not self.trades:
            logger.warning("No trades to analyze")
            return metrics

        metrics.total_trades = len(self.trades)

        # Win/Loss tracking
        wins = []
        losses = []
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        is_winning_streak = None

        for trade in self.trades:
            metrics.total_pnl_points += trade.pnl_points
            metrics.total_pnl_dollars += trade.pnl_dollars

            if trade.pnl_points > 0:
                metrics.winning_trades += 1
                metrics.gross_profit += trade.pnl_dollars
                wins.append(trade.pnl_dollars)

                if is_winning_streak is True:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning_streak = True
                max_win_streak = max(max_win_streak, current_streak)

            elif trade.pnl_points < 0:
                metrics.losing_trades += 1
                metrics.gross_loss += abs(trade.pnl_dollars)
                losses.append(trade.pnl_dollars)

                if is_winning_streak is False:
                    current_streak += 1
                else:
                    current_streak = 1
                    is_winning_streak = False
                max_loss_streak = max(max_loss_streak, current_streak)
            else:
                metrics.breakeven_trades += 1

            # Exit reasons
            if trade.exit_reason == 'TAKE_PROFIT':
                metrics.exits_by_tp += 1
            elif trade.exit_reason == 'STOP_LOSS':
                metrics.exits_by_sl += 1
            elif trade.exit_reason == 'BREAK_EVEN':
                metrics.exits_by_be += 1
            elif trade.exit_reason == 'EARLY_EXIT':
                metrics.exits_by_early += 1
            elif trade.exit_reason == 'CLOSE_REVERSE':
                metrics.exits_by_reverse += 1

            if trade.be_triggered:
                metrics.be_triggered_count += 1

        # Calculate derived metrics
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
            metrics.avg_bars_held = sum(t.bars_held for t in self.trades) / len(self.trades)
            metrics.avg_mae = sum(t.max_adverse for t in self.trades) / len(self.trades)
            metrics.avg_mfe = sum(t.max_favorable for t in self.trades) / len(self.trades)

        if wins:
            metrics.avg_win = np.mean(wins)
        if losses:
            metrics.avg_loss = np.mean([abs(l) for l in losses])

        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss

        # Expectancy: (Win% * AvgWin) - (Loss% * AvgLoss)
        if metrics.total_trades > 0:
            win_pct = metrics.win_rate
            loss_pct = 1 - win_pct
            metrics.expectancy = (win_pct * metrics.avg_win) - (loss_pct * metrics.avg_loss)

        metrics.max_consecutive_wins = max_win_streak
        metrics.max_consecutive_losses = max_loss_streak

        # Calculate max drawdown
        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0
            for equity in self.equity_curve:
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd
            metrics.max_drawdown_dollars = max_dd
            metrics.max_drawdown_points = max_dd / (TICK_VALUE / TICK_SIZE)

        # Calculate filter stats
        for name, stats in self.filter_stats.items():
            if stats.signals_checked > 0:
                stats.block_rate = stats.signals_blocked / stats.signals_checked

        return metrics

    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as a DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()

        data = []
        for t in self.trades:
            data.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'strategy': t.strategy,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'pnl_points': t.pnl_points,
                'pnl_dollars': t.pnl_dollars,
                'exit_reason': t.exit_reason,
                'bars_held': t.bars_held,
                'max_favorable': t.max_favorable,
                'max_adverse': t.max_adverse,
                'hierarchy_key': t.hierarchy_key,
                'session': t.session,
                'be_triggered': t.be_triggered
            })

        return pd.DataFrame(data)

    def print_report(self, metrics: BacktestMetrics):
        """Print comprehensive backtest report."""
        print("\n" + "="*80)
        print("JULIE001 COMPREHENSIVE BACKTEST REPORT")
        print("="*80)

        print("\n--- OVERALL PERFORMANCE ---")
        print(f"Total Trades:       {metrics.total_trades}")
        print(f"Winning Trades:     {metrics.winning_trades}")
        print(f"Losing Trades:      {metrics.losing_trades}")
        print(f"Breakeven Trades:   {metrics.breakeven_trades}")
        print(f"Win Rate:           {metrics.win_rate*100:.2f}%")
        print(f"\nTotal PnL:          {metrics.total_pnl_points:.2f} pts (${metrics.total_pnl_dollars:,.2f})")
        print(f"Gross Profit:       ${metrics.gross_profit:,.2f}")
        print(f"Gross Loss:         ${metrics.gross_loss:,.2f}")
        print(f"Profit Factor:      {metrics.profit_factor:.2f}")
        print(f"Expectancy:         ${metrics.expectancy:,.2f}")

        print("\n--- TRADE STATISTICS ---")
        print(f"Average Win:        ${metrics.avg_win:,.2f}")
        print(f"Average Loss:       ${metrics.avg_loss:,.2f}")
        print(f"Max Win Streak:     {metrics.max_consecutive_wins}")
        print(f"Max Loss Streak:    {metrics.max_consecutive_losses}")
        print(f"Avg Bars Held:      {metrics.avg_bars_held:.1f}")
        print(f"Avg MAE:            {metrics.avg_mae:.2f} pts")
        print(f"Avg MFE:            {metrics.avg_mfe:.2f} pts")

        print("\n--- RISK METRICS ---")
        print(f"Max Drawdown:       ${metrics.max_drawdown_dollars:,.2f} ({metrics.max_drawdown_points:.2f} pts)")

        print("\n--- EXIT REASONS ---")
        print(f"Take Profit:        {metrics.exits_by_tp} ({metrics.exits_by_tp/max(1,metrics.total_trades)*100:.1f}%)")
        print(f"Stop Loss:          {metrics.exits_by_sl} ({metrics.exits_by_sl/max(1,metrics.total_trades)*100:.1f}%)")
        print(f"Break-Even:         {metrics.exits_by_be} ({metrics.exits_by_be/max(1,metrics.total_trades)*100:.1f}%)")
        print(f"Early Exit:         {metrics.exits_by_early} ({metrics.exits_by_early/max(1,metrics.total_trades)*100:.1f}%)")

        print("\n--- FEATURE USAGE ---")
        print(f"Break-Even Triggered: {metrics.be_triggered_count} times")

        print("\n--- FILTER STATISTICS ---")
        for name, stats in self.filter_stats.items():
            if stats.signals_checked > 0:
                print(f"{name:20s}: {stats.signals_blocked:4d}/{stats.signals_checked:4d} blocked ({stats.block_rate*100:.1f}%)")

        # Strategy breakdown
        print("\n--- STRATEGY BREAKDOWN ---")
        strategy_trades = defaultdict(list)
        for trade in self.trades:
            strategy_trades[trade.strategy].append(trade)

        for strat_name, trades in strategy_trades.items():
            wins = sum(1 for t in trades if t.pnl_points > 0)
            total_pnl = sum(t.pnl_dollars for t in trades)
            wr = wins / len(trades) * 100 if trades else 0
            print(f"{strat_name:20s}: {len(trades):3d} trades, WR: {wr:5.1f}%, PnL: ${total_pnl:,.2f}")

        # Session breakdown
        print("\n--- SESSION BREAKDOWN ---")
        session_trades = defaultdict(list)
        for trade in self.trades:
            session_trades[trade.session].append(trade)

        for session_name in ['ASIA', 'LONDON', 'NY_AM', 'NY_PM']:
            trades = session_trades[session_name]
            if trades:
                wins = sum(1 for t in trades if t.pnl_points > 0)
                total_pnl = sum(t.pnl_dollars for t in trades)
                wr = wins / len(trades) * 100 if trades else 0
                print(f"{session_name:10s}: {len(trades):3d} trades, WR: {wr:5.1f}%, PnL: ${total_pnl:,.2f}")

        print("\n" + "="*80)


# ============================================================
# FILTER ISOLATION TESTER
# ============================================================
class FilterIsolationTester:
    """
    Tests each filter in isolation to measure its impact.
    """

    def __init__(self):
        self.results = {}

    def test_filter(self, filter_name: str, df: pd.DataFrame,
                    strategies: List[str] = None) -> Dict:
        """
        Test a single filter by running backtests with/without it.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"TESTING FILTER: {filter_name}")
        logger.info(f"{'='*60}")

        # Baseline: All filters enabled
        bt_all = ComprehensiveBacktester(
            enable_filters=True,
            enable_break_even=True,
            enable_early_exit=True
        )
        metrics_all = bt_all.run(df, strategies)

        # Test: Only this filter disabled
        bt_without = ComprehensiveBacktester(
            enable_filters=True,
            enable_break_even=True,
            enable_early_exit=True
        )
        # Disable specific filter
        if filter_name in bt_without.filters:
            del bt_without.filters[filter_name]
        metrics_without = bt_without.run(df, strategies)

        # Calculate filter impact
        pnl_diff = metrics_all.total_pnl_dollars - metrics_without.total_pnl_dollars
        trade_diff = metrics_without.total_trades - metrics_all.total_trades
        wr_diff = (metrics_all.win_rate - metrics_without.win_rate) * 100

        result = {
            'filter': filter_name,
            'pnl_with': metrics_all.total_pnl_dollars,
            'pnl_without': metrics_without.total_pnl_dollars,
            'pnl_impact': pnl_diff,
            'trades_with': metrics_all.total_trades,
            'trades_without': metrics_without.total_trades,
            'trades_blocked': trade_diff,
            'wr_with': metrics_all.win_rate * 100,
            'wr_without': metrics_without.win_rate * 100,
            'wr_impact': wr_diff
        }

        self.results[filter_name] = result

        logger.info(f"\n{filter_name} IMPACT:")
        logger.info(f"  PnL: ${pnl_diff:,.2f} ({'BETTER' if pnl_diff > 0 else 'WORSE'} with filter)")
        logger.info(f"  Trades blocked: {trade_diff}")
        logger.info(f"  Win Rate: {wr_diff:+.2f}%")

        return result

    def test_all_filters(self, df: pd.DataFrame,
                         strategies: List[str] = None) -> pd.DataFrame:
        """Test all filters and return comparison DataFrame."""
        filter_names = [
            'Volatility', 'Rejection', 'Chop', 'Extension',
            'BankLevel', 'HTFFVG', 'StructureBlocker'
        ]

        for name in filter_names:
            self.test_filter(name, df, strategies)

        return pd.DataFrame(list(self.results.values()))


# ============================================================
# HIERARCHY ANALYZER
# ============================================================
class HierarchyAnalyzer:
    """
    Analyzes performance across the 320 time hierarchy combinations.
    """

    def __init__(self, trades: List[Trade]):
        self.trades = trades

    def analyze(self) -> pd.DataFrame:
        """
        Analyze performance by hierarchy key.
        Returns DataFrame with metrics for each Q_W_DOW_Session combination.
        """
        hierarchy_data = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'pnl': 0, 'avg_pnl': 0
        })

        for trade in self.trades:
            key = trade.hierarchy_key
            hierarchy_data[key]['trades'] += 1
            hierarchy_data[key]['pnl'] += trade.pnl_dollars
            if trade.pnl_points > 0:
                hierarchy_data[key]['wins'] += 1

        # Calculate averages
        rows = []
        for key, data in hierarchy_data.items():
            if data['trades'] > 0:
                data['win_rate'] = data['wins'] / data['trades']
                data['avg_pnl'] = data['pnl'] / data['trades']
            rows.append({'hierarchy': key, **data})

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('pnl', ascending=False)

        return df

    def get_best_hierarchies(self, n: int = 10) -> pd.DataFrame:
        """Get top N performing hierarchies."""
        df = self.analyze()
        return df.head(n) if len(df) >= n else df

    def get_worst_hierarchies(self, n: int = 10) -> pd.DataFrame:
        """Get bottom N performing hierarchies."""
        df = self.analyze()
        return df.tail(n) if len(df) >= n else df


# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='JULIE Comprehensive Backtest')
    parser.add_argument('--start', type=str, default='2023-01-01',
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-31',
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategy', type=str, default=None,
                        help='Specific strategy to test (or all)')
    parser.add_argument('--filter-test', action='store_true',
                        help='Run filter isolation tests')
    parser.add_argument('--no-filters', action='store_true',
                        help='Disable all filters')
    parser.add_argument('--no-be', action='store_true',
                        help='Disable break-even')
    parser.add_argument('--no-early-exit', action='store_true',
                        help='Disable early exit')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (for testing)')
    parser.add_argument('--data-file', type=str, default='es_2023_2024_2025.csv',
                        help='Path to CSV data file (default: es_2023_2024_2025.csv)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for trades CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load or generate data
    if args.synthetic:
        logger.info("Generating synthetic data...")
        generator = SyntheticDataGenerator()
        df = generator.generate(args.start, args.end)
    elif args.data_file and os.path.exists(args.data_file):
        logger.info(f"Loading data from {args.data_file}")

        # Read CSV - handle both 'Datetime' and 'timestamp' column names
        df = pd.read_csv(args.data_file, low_memory=False)

        # Find the datetime column (handle different naming conventions)
        datetime_col = None
        for col in ['Datetime', 'datetime', 'timestamp', 'Timestamp', 'Date', 'date', 'Time Series', 'time series']:
            if col in df.columns:
                datetime_col = col
                break

        if datetime_col is None:
            logger.error("Could not find datetime column in data file")
            logger.error(f"Available columns: {list(df.columns)}")
            logger.info("\nFirst few rows:")
            logger.info(df.head())
            return

        # Parse datetime and set as index
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)

        # Handle timezone
        if df.index.tz is None:
            # If no timezone info, assume US/Eastern
            df.index = df.index.tz_localize(NY_TZ)
        else:
            # Convert to US/Eastern if different timezone
            df.index = df.index.tz_convert(NY_TZ)

        # Handle unnamed columns (common in some data formats)
        # If we have Unnamed columns, they're likely OHLCV in order
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col) or str(col).startswith('ES')]

        # If we have 5+ unnamed columns or contract columns, try to map them to OHLCV
        if len(df.columns) >= 5:
            # Common patterns:
            # Pattern 1: [contract_name, open, high, low, close, volume]
            # Pattern 2: [open, high, low, close, volume]

            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

            if len(numeric_cols) >= 5:
                # Assume first 5 numeric columns are O, H, L, C, V
                logger.info(f"Auto-detecting OHLCV columns from {len(numeric_cols)} numeric columns")
                df = df[numeric_cols]  # Keep only numeric columns
                df.columns = ['open', 'high', 'low', 'close', 'volume'] + list(df.columns[5:])
                logger.info(f"Mapped columns to: {list(df.columns[:5])}")
            elif len(df.columns) == 6:
                # Likely: [contract, O, H, L, C, V]
                logger.info("Detected 6-column format, assuming [contract, O, H, L, C, V]")
                df = df.iloc[:, 1:]  # Skip first column (contract name)
                df.columns = ['open', 'high', 'low', 'close', 'volume']

        # Normalize column names to lowercase (strategies expect lowercase)
        df.columns = df.columns.str.lower()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            logger.error(f"Available columns: {list(df.columns)}")
            logger.info("\nData shape: {}".format(df.shape))
            logger.info("First few rows:")
            logger.info(df.head())
            return

        # Filter by date range if specified
        start_dt = pd.Timestamp(args.start, tz=NY_TZ)
        end_dt = pd.Timestamp(args.end, tz=NY_TZ) + pd.Timedelta(days=1)  # Include end date
        df = df[(df.index >= start_dt) & (df.index < end_dt)]

        if len(df) == 0:
            logger.error(f"No data found in date range {args.start} to {args.end}")
            return

        logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    else:
        logger.info("Data file not found, using synthetic data")
        logger.info(f"Tried: {args.data_file}")
        logger.info("Use --data-file to provide real ES futures data, or --synthetic for generated data")
        generator = SyntheticDataGenerator()
        df = generator.generate(args.start, args.end)

    if df.empty:
        logger.error("No data available for backtest")
        return

    # Parse strategy
    strategies = [args.strategy] if args.strategy else None

    # Run filter isolation tests if requested
    if args.filter_test:
        logger.info("\nRunning Filter Isolation Tests...")
        tester = FilterIsolationTester()
        filter_results = tester.test_all_filters(df, strategies)

        print("\n" + "="*80)
        print("FILTER ISOLATION TEST RESULTS")
        print("="*80)
        print(filter_results.to_string(index=False))
        return

    # Run main backtest
    backtester = ComprehensiveBacktester(
        enable_filters=not args.no_filters,
        enable_break_even=not args.no_be,
        enable_early_exit=not args.no_early_exit
    )

    metrics = backtester.run(df, strategies)

    # Print report
    backtester.print_report(metrics)

    # Hierarchy analysis
    if backtester.trades:
        analyzer = HierarchyAnalyzer(backtester.trades)

        print("\n--- TOP 10 PERFORMING HIERARCHIES ---")
        best = analyzer.get_best_hierarchies(10)
        print(best.to_string(index=False) if len(best) > 0 else "No data")

        print("\n--- BOTTOM 10 PERFORMING HIERARCHIES ---")
        worst = analyzer.get_worst_hierarchies(10)
        print(worst.to_string(index=False) if len(worst) > 0 else "No data")

    # Save trades to CSV if requested
    if args.output:
        trades_df = backtester.get_trades_df()
        trades_df.to_csv(args.output, index=False)
        logger.info(f"Trades saved to {args.output}")

    # Return metrics for programmatic use
    return metrics


if __name__ == '__main__':
    main()
