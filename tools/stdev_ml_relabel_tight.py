"""Re-label TRAIN/VAL/OOS at multiple bracket geometries to find one
where σ-feature signal is strong enough to meet hard constraints.

Tries: TP/SL = (4,4), (3,3), (2,2), (5,5), (3,4), (2,3) and aggregates.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/Users/wes/Downloads/JULIE001')
BAR = ROOT / 'es_master_outrights-2.parquet'
SRC = ROOT / 'artifacts' / 'stdev_ml_hr9_11'
OUT = ROOT / 'artifacts' / 'stdev_ml_hr9_11' / 'tight_brackets'
OUT.mkdir(parents=True, exist_ok=True)

ET = 'US/Eastern'
H_MIN = 120


def label_dataset(period_name, start, end, tp_sl_pairs):
    """Walk-forward label each bar at multiple bracket configs."""
    print(f'\n[{period_name}] {start} → {end}')
    t0 = time.time()
    bars = pd.read_parquet(BAR)
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert(ET)
    bars = bars.loc[start:end]
    if bars.empty: return None

    # Pick dominant contract per day
    bars = bars.reset_index().rename(columns={'index': 'ts'})
    if 'ts' not in bars.columns:
        bars = bars.rename(columns={bars.columns[0]: 'ts'})
    bars['_date'] = bars['ts'].dt.date
    daily_vol = bars.groupby(['_date', 'symbol'])['volume'].sum().reset_index()
    dominant = daily_vol.loc[daily_vol.groupby('_date')['volume'].idxmax(), ['_date', 'symbol']]
    dominant.columns = ['_date', 'dominant_symbol']
    bars = bars.merge(dominant, on='_date', how='left')
    bars = bars[bars['symbol'] == bars['dominant_symbol']].copy()
    bars = bars.drop(columns=['_date', 'dominant_symbol', 'symbol'])
    bars = bars.set_index('ts').sort_index()
    print(f'  loaded {len(bars):,} bars, dt={time.time()-t0:.1f}s')

    # Filter to ET 9-11 weekdays
    h = bars.index.hour
    dow = bars.index.dayofweek
    win_mask = (h >= 9) & (h <= 11) & (dow < 5)

    high = bars['high'].values
    low = bars['low'].values
    open_ = bars['open'].values
    close = bars['close'].values
    n = len(bars)

    # For each (bar, side, tp/sl), determine TP-first vs SL-first
    out_data = []
    win_idx = np.where(win_mask)[0]
    print(f'  in-window bars: {len(win_idx):,}')

    for tp, sl in tp_sl_pairs:
        labels_long = np.full(len(win_idx), -1, dtype=np.int8)
        labels_short = np.full(len(win_idx), -1, dtype=np.int8)
        for ii, i in enumerate(win_idx):
            if i + 1 + H_MIN >= n: continue
            entry = open_[i + 1]  # next-bar open
            tp_long = entry + tp; sl_long = entry - sl
            tp_short = entry - tp; sl_short = entry + sl
            l_long = -1; l_short = -1
            for j in range(i + 1, min(i + 1 + H_MIN, n)):
                hh = high[j]; ll = low[j]
                # LONG check
                if l_long == -1:
                    if ll <= sl_long: l_long = 0
                    elif hh >= tp_long: l_long = 1
                # SHORT check
                if l_short == -1:
                    if hh >= sl_short: l_short = 0
                    elif ll <= tp_short: l_short = 1
                if l_long != -1 and l_short != -1: break
            # Timeout fallback
            if l_long == -1:
                last = close[min(i + H_MIN, n - 1)]
                l_long = 1 if last > entry else 0
            if l_short == -1:
                last = close[min(i + H_MIN, n - 1)]
                l_short = 1 if last < entry else 0
            labels_long[ii] = l_long
            labels_short[ii] = l_short

        out_data.append({'tp': tp, 'sl': sl, 'long': labels_long, 'short': labels_short})

    timestamps = bars.index[win_mask][:len(win_idx)]
    return {'timestamps': timestamps, 'tp_sl': out_data, 'n': len(win_idx)}


def main():
    PAIRS = [(4, 4), (3, 3), (2, 2), (5, 5), (3, 4), (2, 3)]
    PERIODS = {
        'TRAIN_t1':   ('2017-01-20', '2021-01-20'),
        'TRAIN_2025': ('2025-01-20', '2025-12-31'),
        'VAL_biden':  ('2021-01-21', '2025-01-19'),
        'OOS_2026':   ('2026-01-01', '2026-04-30'),
    }
    all_periods = {}
    for name, (s, e) in PERIODS.items():
        all_periods[name] = label_dataset(name, s, e, PAIRS)

    # Aggregate per (tp, sl) — compute baseline WR for LONG/SHORT in each period
    print('\n=== Baseline WR per period × bracket × side ===')
    print(f'{"period":<12s} {"tp/sl":>6s} {"side":>6s} {"n":>7s} {"WR%":>6s}')
    rows = []
    for name, d in all_periods.items():
        if d is None: continue
        for tp_sl in d['tp_sl']:
            tp, sl = tp_sl['tp'], tp_sl['sl']
            for side, labels in [('LONG', tp_sl['long']), ('SHORT', tp_sl['short'])]:
                valid = labels >= 0
                n = int(valid.sum())
                if n == 0: continue
                wr = float(labels[valid].mean())
                breakeven = sl / (tp + sl)
                edge = wr - breakeven
                rows.append({
                    'period': name, 'tp': tp, 'sl': sl, 'side': side,
                    'n': n, 'wr': wr, 'breakeven': breakeven, 'edge': edge
                })
                print(f'{name:<12s} {tp}/{sl:<3} {side:>6s} {n:>7d} {wr*100:>5.1f}%   (breakeven {breakeven*100:.1f}% → edge {edge*100:+.1f}%)')

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'baseline_wr_per_bracket.csv', index=False)
    print(f'\nSaved baseline_wr_per_bracket.csv')

    # Best edge bucket: which TP/SL × side has the highest WR-vs-breakeven edge?
    print('\n=== Best baseline edge by (tp, sl, side) — averaged across periods ===')
    agg = df.groupby(['tp', 'sl', 'side']).agg(
        avg_wr=('wr', 'mean'), avg_edge=('edge', 'mean'), total_n=('n', 'sum')
    ).reset_index().sort_values('avg_edge', ascending=False)
    print(agg.to_string(index=False, float_format='%.4f'))

    # Save labels for the best (tp, sl, side) candidates
    best_combos = agg.head(6).to_dict('records')
    print(f'\nTop 6 (tp, sl, side) combos to explore further:')
    for c in best_combos:
        print(f'  TP={c["tp"]}/SL={c["sl"]} {c["side"]}: avg WR={c["avg_wr"]*100:.1f}%, edge={c["avg_edge"]*100:+.2f}%, n={int(c["total_n"]):,}')


if __name__ == '__main__':
    main()
