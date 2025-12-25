"""
RegimeAdaptive Strategy Module
==============================
Extracted from julie001.py with signal reversion for underperforming time contexts.

Based on backtest results, combos with WR < 35% get signals REVERTED (LONG<->SHORT).
This turns losing patterns into winning ones by fading the original signal.

Import this in julie001.py:
    from regime_strategy import RegimeAdaptiveStrategy
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

# ============================================================
# Try to import optimized SLTP params
# ============================================================
try:
    from regime_sltp_params import (
        get_regime_sltp, 
        is_reverted_combo as _is_reverted_from_params,
        REVERTED_COMBOS as PARAMS_REVERTED_COMBOS,
        PARAMS as SLTP_PARAMS
    )
    SLTP_PARAMS_AVAILABLE = True
    logging.info(f"Loaded regime_sltp_params with {len(SLTP_PARAMS)} combos")
except ImportError:
    SLTP_PARAMS_AVAILABLE = False
    PARAMS_REVERTED_COMBOS = set()
    SLTP_PARAMS = {}
    logging.warning("regime_sltp_params not found, using defaults")

# ============================================================
# REVERTED COMBOS - from regime_sltp_params.py if available
# These combos performed poorly with normal signals, so we
# trade the OPPOSITE direction (fade the signal)
# ============================================================
if SLTP_PARAMS_AVAILABLE and PARAMS_REVERTED_COMBOS:
    REVERTED_COMBOS = PARAMS_REVERTED_COMBOS
    logging.info(f"Using {len(REVERTED_COMBOS)} reverted combos from regime_sltp_params")
else:
    # Fallback defaults if params file not available
    REVERTED_COMBOS = {
        'Q1_W1_TUE_ASIA', 'Q1_W1_WED_ASIA', 'Q1_W2_FRI_NY_PM',
        'Q1_W4_FRI_NY_PM', 'Q1_W4_MON_ASIA', 'Q1_W4_THU_ASIA',
        'Q1_W4_TUE_NY_PM', 'Q2_W2_TUE_NY_AM', 'Q2_W2_WED_NY_PM',
        'Q2_W3_WED_LONDON', 'Q2_W4_MON_NY_AM', 'Q2_W4_MON_NY_PM',
        'Q3_W1_TUE_NY_PM', 'Q3_W1_WED_ASIA', 'Q3_W1_WED_LONDON',
        'Q3_W1_WED_NY_PM', 'Q3_W2_WED_ASIA', 'Q3_W3_THU_NY_AM',
        'Q3_W4_FRI_NY_PM', 'Q3_W4_MON_ASIA', 'Q3_W4_MON_NY_PM',
        'Q3_W4_THU_LONDON', 'Q3_W4_THU_NY_AM',
    }


def get_session(hour: int) -> str:
    """Determine trading session from hour."""
    if 18 <= hour <= 23 or 0 <= hour < 3:
        return 'ASIA'
    elif 3 <= hour < 8:
        return 'LONDON'
    elif 8 <= hour < 12:
        return 'NY_AM'
    elif 12 <= hour < 17:
        return 'NY_PM'
    return 'CLOSED'


def get_yearly_quarter(month: int) -> str:
    """Get yearly quarter from month."""
    if month <= 3: return 'Q1'
    elif month <= 6: return 'Q2'
    elif month <= 9: return 'Q3'
    return 'Q4'


def get_monthly_quarter(day: int) -> str:
    """Get monthly quarter (week of month) from day."""
    if day <= 7: return 'W1'
    elif day <= 14: return 'W2'
    elif day <= 21: return 'W3'
    return 'W4'


def get_day_name(dow: int) -> str:
    """Get day name from day of week integer."""
    return ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][dow]


def get_time_context(ts) -> Dict[str, str]:
    """Extract full time context from timestamp."""
    return {
        'yearly_q': get_yearly_quarter(ts.month),
        'monthly_q': get_monthly_quarter(ts.day),
        'day_of_week': get_day_name(ts.dayofweek),
        'session': get_session(ts.hour)
    }


def get_combo_key(ts) -> str:
    """Get hierarchy combo key for timestamp."""
    ctx = get_time_context(ts)
    return f"{ctx['yearly_q']}_{ctx['monthly_q']}_{ctx['day_of_week']}_{ctx['session']}"


def should_revert_signal(ts) -> bool:
    """Check if current time context should have signals reverted."""
    combo = get_combo_key(ts)
    return combo in REVERTED_COMBOS


def get_optimized_sltp(side: str, ts) -> Dict[str, float]:
    """
    Get optimized SL/TP for the given side and timestamp.

    Returns dict with 'sl_dist' and 'tp_dist'
    """
    # Minimum enforcement for positive RR
    MIN_SL = 4.0  # 16 ticks minimum
    MIN_TP = 6.0  # 24 ticks minimum (1.5:1 RR)

    ctx = get_time_context(ts)

    if SLTP_PARAMS_AVAILABLE:
        params = get_regime_sltp(
            side,
            ctx['yearly_q'],
            ctx['monthly_q'],
            ctx['day_of_week'],
            ctx['session']
        )
        sl_dist = max(params['sl'], MIN_SL)
        tp_dist = max(params['tp'], MIN_TP)
        return {
            'sl_dist': sl_dist,
            'tp_dist': tp_dist
        }

    # Defaults if no params available
    return {'sl_dist': MIN_SL, 'tp_dist': MIN_TP}


class RegimeAdaptiveStrategy:
    """
    RegimeAdaptive Strategy with Time-Context Signal Reversion

    Logic:
    - NORMAL: LONG on uptrend dip, SHORT on downtrend rally
    - REVERTED: For poor-performing time contexts, flip LONG<->SHORT

    Signal conditions:
    - Uptrend: SMA20 > SMA200 + low volatility
    - Downtrend: SMA20 < SMA200 + low volatility
    - Entry: Price crosses fast SMA with range spike

    Equal low/high filter prevents continuation pattern entries.

    Optimized SL/TP params loaded from regime_sltp_params.py if available.
    """

    def __init__(self, dynamic_sltp_engine=None):
        self.strategy_name = "RegimeAdaptive"
        self.use_eq_filter = True
        self.eq_tolerance = 0.5
        self.lookback_bars = 10
        self.dynamic_sltp_engine = dynamic_sltp_engine
        self.last_combo_key = None
        self.bars_since_log = 0

        # SMA periods
        self.sma_fast = 20
        self.sma_slow = 200

        # Use optimized SLTP from regime_sltp_params if available
        self.use_optimized_sltp = SLTP_PARAMS_AVAILABLE

        logging.info(f"RegimeAdaptiveStrategy initialized | Reverted combos: {len(REVERTED_COMBOS)} | SLTP params: {SLTP_PARAMS_AVAILABLE}")
    
    def _is_equal_low(self, df: pd.DataFrame) -> bool:
        """Check if current low matches recent lows (continuation pattern)."""
        if len(df) < self.lookback_bars + 1:
            return False
        curr_low = df.iloc[-1]['low']
        recent_lows = df['low'].iloc[-(self.lookback_bars + 1):-1].values
        return any(abs(curr_low - l) <= self.eq_tolerance for l in recent_lows)
    
    def _is_equal_high(self, df: pd.DataFrame) -> bool:
        """Check if current high matches recent highs (continuation pattern)."""
        if len(df) < self.lookback_bars + 1:
            return False
        curr_high = df.iloc[-1]['high']
        recent_highs = df['high'].iloc[-(self.lookback_bars + 1):-1].values
        return any(abs(curr_high - h) <= self.eq_tolerance for h in recent_highs)
    
    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate trading signal on new bar.
        
        Returns:
            Dict with signal info or None
        """
        if len(df) < 200:
            return None
        
        curr = df.iloc[-1]
        ts = df.index[-1]
        
        # Get time context
        combo_key = get_combo_key(ts)
        ctx = get_time_context(ts)

        # Log context changes
        if self.last_combo_key != combo_key:
            self.last_combo_key = combo_key
            is_reverted = combo_key in REVERTED_COMBOS
            logging.info(f"RegimeAdaptive: Context changed to {combo_key} | Reverted: {is_reverted}")

        # Skip if CLOSED session
        if ctx['session'] == 'CLOSED':
            return None
        
        # Calculate indicators
        sma_fast = df['close'].rolling(self.sma_fast).mean().iloc[-1]
        sma_slow = df['close'].rolling(self.sma_slow).mean().iloc[-1]
        
        # Volatility
        vol_series = df['close'].pct_change().rolling(20).std()
        curr_vol = vol_series.iloc[-1]
        vol_median = vol_series.rolling(100).median().iloc[-1]
        
        # Range spike
        rng = df['high'] - df['low']
        range_sma = rng.rolling(20).mean().iloc[-1]
        curr_range = rng.iloc[-1]
        
        # Regime detection
        trending_up = (sma_fast > sma_slow) and (curr_vol < vol_median)
        trending_down = (sma_fast < sma_slow) and (curr_vol < vol_median)
        range_spike = curr_range > range_sma
        
        # Check if we should revert signals for this time context
        revert = should_revert_signal(ts)
        
        signal = None
        
        # LONG signal: Uptrend + dip below fast SMA + range spike
        if trending_up and (curr['close'] < sma_fast) and range_spike:
            if self.use_eq_filter and self._is_equal_low(df):
                logging.debug(f"RegimeAdaptive: LONG skipped - equal low at {combo_key}")
                return None
            signal = 'LONG'
        
        # SHORT signal: Downtrend + rally above fast SMA + range spike
        elif trending_down and (curr['close'] > sma_fast) and range_spike:
            if self.use_eq_filter and self._is_equal_high(df):
                logging.debug(f"RegimeAdaptive: SHORT skipped - equal high at {combo_key}")
                return None
            signal = 'SHORT'
        
        if signal is None:
            return None
        
        # Apply reversion if needed
        original_signal = signal
        if revert:
            signal = 'SHORT' if signal == 'LONG' else 'LONG'
            logging.info(f"RegimeAdaptive: Signal REVERTED {original_signal}->{signal} for {combo_key}")
        else:
            logging.info(f"RegimeAdaptive: {signal} signal generated | Combo: {combo_key}")
            logging.info(f"   SMA20: {sma_fast:.2f} | SMA200: {sma_slow:.2f} | Vol: {curr_vol:.4f} (median: {vol_median:.4f})")
            logging.info(f"   Range: {curr_range:.2f} (SMA: {range_sma:.2f}) | Price: {curr['close']:.2f}")

        # Get SL/TP - use optimized params if available
        if self.use_optimized_sltp:
            sltp = get_optimized_sltp(signal, ts)
        elif self.dynamic_sltp_engine:
            sltp = self.dynamic_sltp_engine.calculate_dynamic_sltp(df)
        else:
            sltp = {'sl_dist': 4.0, 'tp_dist': 6.0}
        
        # Log params if engine available
        if self.dynamic_sltp_engine:
            self.dynamic_sltp_engine.log_params(sltp)
        
        return {
            "strategy": "RegimeAdaptive",
            "side": signal,
            "tp_dist": sltp['tp_dist'],
            "sl_dist": sltp['sl_dist'],
            "combo_key": combo_key,
            "reverted": revert,
            "original_signal": original_signal if revert else signal
        }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def load_reverted_combos_from_csv(csv_path: str, wr_threshold: float = 35.0) -> set:
    """
    Load hierarchy CSV and return combos with WR below threshold.
    
    Usage:
        combos = load_reverted_combos_from_csv('regime_backtest_hierarchy.csv', 35.0)
        print(combos)  # Copy these into REVERTED_COMBOS
    """
    df = pd.read_csv(csv_path)
    poor = df[df['wr'] < wr_threshold]
    return set(poor['combo'].tolist())


def get_strategy_stats() -> Dict:
    """Get current strategy configuration stats."""
    return {
        'reverted_combos': len(REVERTED_COMBOS),
        'sltp_params_available': SLTP_PARAMS_AVAILABLE,
        'sltp_combos': len(SLTP_PARAMS) if SLTP_PARAMS_AVAILABLE else 0
    }


if __name__ == "__main__":
    # Test: Show current configuration
    from datetime import datetime
    from zoneinfo import ZoneInfo

    et = ZoneInfo('America/New_York')
    now = datetime.now(et)
    
    combo = get_combo_key(now)
    revert = should_revert_signal(now)
    
    print("=" * 60)
    print("REGIME STRATEGY CONFIGURATION")
    print("=" * 60)
    print(f"Current time: {now}")
    print(f"Combo key: {combo}")
    print(f"Should revert: {revert}")
    print(f"\nTotal reverted combos: {len(REVERTED_COMBOS)}")
    print(f"SLTP params available: {SLTP_PARAMS_AVAILABLE}")
    
    if SLTP_PARAMS_AVAILABLE:
        print(f"SLTP combos loaded: {len(SLTP_PARAMS)}")
        
        # Show current SLTP
        ctx = get_time_context(now)
        for side in ['LONG', 'SHORT']:
            sltp = get_optimized_sltp(side, now)
            print(f"\n{side} SLTP for {combo}:")
            print(f"  SL: {sltp['sl_dist']:.2f} pts")
            print(f"  TP: {sltp['tp_dist']:.2f} pts")
