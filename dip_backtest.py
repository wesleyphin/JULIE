

import pandas as pd
import numpy as np
import datetime
from datetime import time as dt_time
from typing import Dict, List, Tuple, Optional
from itertools import product
import logging
from collections import defaultdict
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ============================================================
# TIME CONTEXT HELPERS
# ============================================================

def get_yearly_quarter(month: int) -> str:
    if month <= 3: return 'Q1'
    elif month <= 6: return 'Q2'
    elif month <= 9: return 'Q3'
    return 'Q4'

def get_monthly_quarter(day: int) -> str:
    if day <= 7: return 'W1'
    elif day <= 14: return 'W2'
    elif day <= 21: return 'W3'
    return 'W4'

def get_day_name(dow: int) -> str:
    return ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'][dow]

def get_session(hour: int) -> str:
    if 18 <= hour <= 23 or 0 <= hour < 3:
        return 'ASIA'
    elif 3 <= hour < 8:
        return 'LONDON'
    elif 8 <= hour < 12:
        return 'NY_AM'
    elif 12 <= hour < 17:
        return 'NY_PM'
    return 'CLOSED'

def get_time_context(ts) -> Dict[str, str]:
    return {
        'yearly_q': get_yearly_quarter(ts.month),
        'monthly_q': get_monthly_quarter(ts.day),
        'day_of_week': get_day_name(ts.dayofweek),
        'session': get_session(ts.hour)
    }

def get_combo_key(ts) -> str:
    ctx = get_time_context(ts)
    return f"{ctx['yearly_q']}_{ctx['monthly_q']}_{ctx['day_of_week']}_{ctx['session']}"


# ============================================================
# INTRADAY DIP STRATEGY - Imported Logic from julie001.py
# ============================================================

class IntradayDipBacktestStrategy:
    """
    Exact replication of IntradayDipStrategy from julie001.py
    Modified for backtesting (no dynamic_sltp_engine dependency)
    
    Signal Logic:
    - LONG: Price down >= 1.0% from session open, z-score < -0.5, volatility spike
    - SHORT: Price up >= 1.25% from session open, z-score > 1.0, volatility spike
    """
    
    def __init__(self):
        self.session_open = None
        self.current_date = None
    
    def check_signal_at_index(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        """
        Check for signal at specific index in dataframe.
        This mirrors on_bar() but works with historical data.
        """
        if idx < 20:
            return None
        
        # Get data window up to current index
        window_df = df.iloc[:idx+1]
        
        ts = df.index[idx]
        curr = df.iloc[idx]
        
        # Skip weekends
        if ts.dayofweek >= 5:
            return None
        
        # Skip CLOSED session
        if get_session(ts.hour) == 'CLOSED':
            return None
        
        # Reset on new day
        if self.current_date != ts.date():
            self.session_open = None
            self.current_date = ts.date()
        
        # Capture session open at 9:30 ET
        # Check if this bar is 9:30 or find the 9:30 open for today
        if ts.hour == 9 and ts.minute == 30:
            self.session_open = curr['open']
        elif self.session_open is None:
            # Try to find today's 9:30 open in the data
            today = ts.date()
            # Handle timezone-aware index
            today_mask = pd.Series([t.date() == today for t in window_df.index], index=window_df.index)
            today_data = window_df[today_mask]
            for t, row in today_data.iterrows():
                if t.hour == 9 and t.minute == 30:
                    self.session_open = row['open']
                    break
        
        if self.session_open is None:
            return None
        
        # Calculate intraday % change from session open
        pct_change = (curr['close'] - self.session_open) / self.session_open * 100
        
        # Z-score calculation (using rolling window)
        if len(window_df) < 20:
            return None
        
        recent = window_df['close'].iloc[-20:]
        sma20 = recent.mean()
        std20 = recent.std()
        
        if std20 == 0 or pd.isna(std20):
            return None
        
        z_score = (curr['close'] - sma20) / std20
        
        # Volatility spike detection
        range_series = window_df['high'].iloc[-20:] - window_df['low'].iloc[-20:]
        range_sma = range_series.mean()
        curr_range = curr['high'] - curr['low']
        is_vol_spike = curr_range > range_sma
        
        # LONG: Down 1%+, oversold (z < -0.5), volatility spike
        if (pct_change <= -1.0) and (z_score < -0.5) and is_vol_spike:
            return {
                'side': 'LONG',
                'entry_price': curr['close'],
                'entry_time': ts,
                'combo_key': get_combo_key(ts),
                'context': get_time_context(ts),
                'pct_change': pct_change,
                'z_score': z_score
            }
        
        # SHORT: Up 1.25%+, overbought (z > 1.0), volatility spike
        if (pct_change >= 1.25) and (z_score > 1.0) and is_vol_spike:
            return {
                'side': 'SHORT',
                'entry_price': curr['close'],
                'entry_time': ts,
                'combo_key': get_combo_key(ts),
                'context': get_time_context(ts),
                'pct_change': pct_change,
                'z_score': z_score
            }
        
        return None


# ============================================================
# BACKTEST ENGINE
# ============================================================

def simulate_trade(df: pd.DataFrame, entry_idx: int, side: str, 
                   entry_price: float, sl_pts: float, tp_pts: float,
                   max_bars: int = 500) -> Dict:
    """Simulate a single trade from entry point."""
    if side == 'LONG':
        sl_price = entry_price - sl_pts
        tp_price = entry_price + tp_pts
    else:
        sl_price = entry_price + sl_pts
        tp_price = entry_price - tp_pts
    
    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[i]
        high, low = bar['high'], bar['low']
        
        if side == 'LONG':
            if low <= sl_price:
                return {'outcome': 'LOSS', 'pnl': -sl_pts, 'bars_held': i - entry_idx, 'exit_price': sl_price}
            if high >= tp_price:
                return {'outcome': 'WIN', 'pnl': tp_pts, 'bars_held': i - entry_idx, 'exit_price': tp_price}
        else:
            if high >= sl_price:
                return {'outcome': 'LOSS', 'pnl': -sl_pts, 'bars_held': i - entry_idx, 'exit_price': sl_price}
            if low <= tp_price:
                return {'outcome': 'WIN', 'pnl': tp_pts, 'bars_held': i - entry_idx, 'exit_price': tp_price}
    
    # Timeout
    final_price = df.iloc[min(entry_idx + max_bars - 1, len(df) - 1)]['close']
    pnl = (final_price - entry_price) if side == 'LONG' else (entry_price - final_price)
    return {'outcome': 'TIMEOUT', 'pnl': pnl, 'bars_held': max_bars, 'exit_price': final_price}


def generate_sltp_combinations() -> List[Tuple[float, float]]:
    """Generate 725 SL/TP combinations (29 SL × 25 TP)"""
    sl_values = [x * 0.5 for x in range(2, 31)]  # 1.0 to 15.0 by 0.5
    tp_values = [x * 1.0 for x in range(1, 26)]  # 1.0 to 25.0 by 1.0
    return list(product(sl_values, tp_values))


def run_backtest(df: pd.DataFrame, progress_interval: int = 5000) -> Tuple[pd.DataFrame, List[Dict]]:
    """Run full backtest across all time contexts and SL/TP combinations."""
    strategy = IntradayDipBacktestStrategy()
    sltp_combos = generate_sltp_combinations()
    
    logging.info(f"Generated {len(sltp_combos)} SL/TP combinations")
    
    # Find all signals
    logging.info("Scanning for signals...")
    signals = []
    
    for i in range(20, len(df)):
        if i % 50000 == 0:
            logging.info(f"Scanning bar {i:,}/{len(df):,} ({i/len(df)*100:.1f}%)")
        
        signal = strategy.check_signal_at_index(df, i)
        if signal:
            signal['entry_idx'] = i
            signals.append(signal)
    
    logging.info(f"Found {len(signals)} signals")
    
    if not signals:
        logging.warning("No signals found!")
        return pd.DataFrame(), signals
    
    # Results storage
    results = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'timeouts': 0, 'pnl': 0.0, 'trades': 0}))
    results_yearly = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': 0}))
    results_monthly = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': 0}))
    results_day = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': 0}))
    results_session = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': 0}))
    
    # Also track by side
    results_side = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': 0}))
    
    total_iterations = len(signals) * len(sltp_combos)
    logging.info(f"Running {total_iterations:,} trade simulations...")
    
    iteration = 0
    for sig in signals:
        entry_idx = sig['entry_idx']
        side = sig['side']
        entry_price = sig['entry_price']
        combo_key = sig['combo_key']
        ctx = sig['context']
        
        for sl, tp in sltp_combos:
            result = simulate_trade(df, entry_idx, side, entry_price, sl, tp)
            
            # Update full combo results
            r = results[combo_key][(sl, tp)]
            r['trades'] += 1
            r['pnl'] += result['pnl']
            if result['outcome'] == 'WIN': r['wins'] += 1
            elif result['outcome'] == 'LOSS': r['losses'] += 1
            else: r['timeouts'] += 1
            
            # Update aggregated results
            for ctx_key, ctx_results in [
                (ctx['yearly_q'], results_yearly),
                (ctx['monthly_q'], results_monthly),
                (ctx['day_of_week'], results_day),
                (ctx['session'], results_session),
                (side, results_side)
            ]:
                cr = ctx_results[ctx_key][(sl, tp)]
                cr['trades'] += 1
                cr['pnl'] += result['pnl']
                if result['outcome'] == 'WIN': cr['wins'] += 1
                elif result['outcome'] == 'LOSS': cr['losses'] += 1
            
            iteration += 1
            if iteration % progress_interval == 0:
                logging.info(f"Progress: {iteration:,}/{total_iterations:,} ({iteration/total_iterations*100:.1f}%)")
    
    logging.info("Building results DataFrame...")
    
    rows = []
    
    # Full hierarchy combos
    for combo_key, sltp_results in results.items():
        parts = combo_key.split('_')
        if len(parts) != 4: continue
        yearly_q, monthly_q, day, session = parts
        
        for (sl, tp), stats in sltp_results.items():
            if stats['trades'] > 0:
                wr = stats['wins'] / stats['trades'] * 100
                rows.append({
                    'level': 'FULL_COMBO',
                    'combo_key': combo_key,
                    'yearly_q': yearly_q,
                    'monthly_q': monthly_q,
                    'day_of_week': day,
                    'session': session,
                    'side': 'ALL',
                    'sl': sl, 'tp': tp,
                    'trades': stats['trades'],
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'wr': round(wr, 2),
                    'pnl': round(stats['pnl'], 2),
                    'avg_pnl': round(stats['pnl'] / stats['trades'], 2)
                })
    
    # Aggregated by level
    for level_name, level_results, fill_cols in [
        ('YEARLY_Q', results_yearly, {'monthly_q': 'ALL', 'day_of_week': 'ALL', 'session': 'ALL'}),
        ('MONTHLY_Q', results_monthly, {'yearly_q': 'ALL', 'day_of_week': 'ALL', 'session': 'ALL'}),
        ('DAY', results_day, {'yearly_q': 'ALL', 'monthly_q': 'ALL', 'session': 'ALL'}),
        ('SESSION', results_session, {'yearly_q': 'ALL', 'monthly_q': 'ALL', 'day_of_week': 'ALL'}),
        ('SIDE', results_side, {'yearly_q': 'ALL', 'monthly_q': 'ALL', 'day_of_week': 'ALL', 'session': 'ALL'})
    ]:
        for key, sltp_results in level_results.items():
            for (sl, tp), stats in sltp_results.items():
                if stats['trades'] > 0:
                    wr = stats['wins'] / stats['trades'] * 100
                    row = {
                        'level': level_name,
                        'combo_key': key,
                        'sl': sl, 'tp': tp,
                        'trades': stats['trades'],
                        'wins': stats['wins'],
                        'losses': stats['losses'],
                        'wr': round(wr, 2),
                        'pnl': round(stats['pnl'], 2),
                        'avg_pnl': round(stats['pnl'] / stats['trades'], 2),
                        'side': 'ALL'
                    }
                    row.update(fill_cols)
                    if level_name == 'YEARLY_Q': row['yearly_q'] = key
                    elif level_name == 'MONTHLY_Q': row['monthly_q'] = key
                    elif level_name == 'DAY': row['day_of_week'] = key
                    elif level_name == 'SESSION': row['session'] = key
                    elif level_name == 'SIDE': row['side'] = key
                    rows.append(row)
    
    df_results = pd.DataFrame(rows)
    logging.info(f"Generated {len(df_results)} result rows")
    
    return df_results, signals


def find_best_params(df_results: pd.DataFrame) -> pd.DataFrame:
    """Find best SL/TP for each context by PnL and WR"""
    best_rows = []
    
    for level in df_results['level'].unique():
        level_df = df_results[df_results['level'] == level]
        
        for combo_key in level_df['combo_key'].unique():
            combo_df = level_df[level_df['combo_key'] == combo_key]
            if len(combo_df) == 0: continue
            
            # Best by PnL
            best_pnl = combo_df.loc[combo_df['pnl'].idxmax()]
            
            # Best by WR (min 5 trades)
            valid_wr = combo_df[combo_df['trades'] >= 5]
            best_wr = valid_wr.loc[valid_wr['wr'].idxmax()] if len(valid_wr) > 0 else best_pnl
            
            best_rows.append({
                'level': level,
                'combo_key': combo_key,
                'yearly_q': best_pnl.get('yearly_q', 'ALL'),
                'monthly_q': best_pnl.get('monthly_q', 'ALL'),
                'day_of_week': best_pnl.get('day_of_week', 'ALL'),
                'session': best_pnl.get('session', 'ALL'),
                'best_pnl_sl': best_pnl['sl'],
                'best_pnl_tp': best_pnl['tp'],
                'best_pnl': best_pnl['pnl'],
                'best_pnl_wr': best_pnl['wr'],
                'best_pnl_trades': best_pnl['trades'],
                'best_wr_sl': best_wr['sl'],
                'best_wr_tp': best_wr['tp'],
                'best_wr': best_wr['wr'],
                'best_wr_pnl': best_wr['pnl'],
                'best_wr_trades': best_wr['trades']
            })
    
    return pd.DataFrame(best_rows)


def load_es_data(filepath: str) -> pd.DataFrame:
    """Load ES 2023-2025 data"""
    logging.info(f"Loading {filepath}...")
    
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')
    
    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    
    df = df.sort_index()
    logging.info(f"Loaded {len(df):,} bars: {df.index[0]} to {df.index[-1]}")
    
    return df


def main():
    # Paths
    data_file = 'es_2023_2025.csv'
    output_file = 'intraday_dip_results.csv'
    best_file = 'intraday_dip_best_params.csv'
    signals_file = 'intraday_dip_signals.csv'
    
    # Load data
    df = load_es_data(data_file)
    
    # Run backtest
    results, signals = run_backtest(df)
    
    if results.empty:
        logging.error("No results!")
        return
    
    # Save results
    results.to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")
    
    # Save best params
    best_params = find_best_params(results)
    best_params.to_csv(best_file, index=False)
    logging.info(f"Best params saved to {best_file}")
    
    # Save signal log
    signals_df = pd.DataFrame(signals)
    signals_df.to_csv(signals_file, index=False)
    logging.info(f"Signals saved to {signals_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("INTRADAY DIP BACKTEST SUMMARY")
    print("="*70)
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Total signals: {len(signals)}")
    print(f"LONG signals: {sum(1 for s in signals if s['side'] == 'LONG')}")
    print(f"SHORT signals: {sum(1 for s in signals if s['side'] == 'SHORT')}")
    print(f"SL/TP combinations: {len(generate_sltp_combinations())}")
    print(f"Total result rows: {len(results)}")
    
    # Top results by level
    for level in ['YEARLY_Q', 'SESSION', 'DAY', 'SIDE']:
        level_best = best_params[best_params['level'] == level].sort_values('best_pnl', ascending=False)
        if not level_best.empty:
            print(f"\n--- Best by PnL ({level}) ---")
            for _, row in level_best.head(5).iterrows():
                print(f"  {row['combo_key']}: SL={row['best_pnl_sl']:.1f} TP={row['best_pnl_tp']:.1f} | "
                      f"WR={row['best_pnl_wr']:.1f}% PnL={row['best_pnl']:.2f} ({int(row['best_pnl_trades'])} trades)")


if __name__ == "__main__":
    main()
