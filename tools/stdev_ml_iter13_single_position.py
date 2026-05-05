"""Iter 13 — re-evaluate iter 12 best config with SINGLE-POSITION constraint.

Iter 11/12 evaluators fired on every bar where p ≥ threshold and counted each
fire as a separate trade. That implies parallel positions, which the live bot
(Julie001) cannot take — it's single-position-at-a-time.

Real-world replay:
  - Iterate bars in chronological order
  - Track state: (in_position, entry_ts, entry_price, side, tp_price, sl_price, horizon_end)
  - At each bar:
      a. If in_position: check this bar's high/low for TP/SL hit, or close at horizon_end
      b. If not in_position: check for fire signal (p ≥ thr); if yes, open
  - Daily DD circuit breaker still applies: after intraday cum PnL ≤ -cap, no more entries today

Stats: same per-month + per-year decomposition.
"""
from __future__ import annotations
import time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, '/Users/wes/Downloads/JULIE001/tools')
from stdev_ml_iter11_atr import (
    compute_features, load_period, PERIODS, WIN_HOURS, H_MIN_NEW, PT_USD,
)
from stdev_ml_iter12_both_sides import label_walk_both_sides

warnings.filterwarnings("ignore")

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter12_both_sides'

t0 = time.time()
TP_M, SL_M = 1.5, 1.0
THR, CAP = 0.75, 200

bL = joblib.load(OUT / f'iter12_{TP_M}_{SL_M}_L.pkl')
bS = joblib.load(OUT / f'iter12_{TP_M}_{SL_M}_S.pkl')
hgb_L = bL['hgb']; top60_L = bL['features']
hgb_S = bS['hgb']; top60_S = bS['features']
print(f'Best config: TP={TP_M}×ATR / SL={SL_M}×ATR, thr={THR}, daily_dd_cap=${CAP}, horizon={H_MIN_NEW}min')
print(f'SINGLE-POSITION constraint enforced — only 1 open trade at a time.\n')


def predict_partition(name):
    """Build features+labels, predict both sides, return rows in chronological per-bar form.
    Returns:
      bars_df: indexed by ts, with high/low/open/close + features per BAR (not per side-row)
      fire_df: list of fire candidates with ts, side, p, tp_pts, sl_pts in chronological order
    """
    s, e = PERIODS[name]
    print(f'[{name}] building features {s}→{e}...')
    bars = load_period(s, e)
    df = compute_features(bars)
    in_win = (df.index.hour >= WIN_HOURS[0]) & (df.index.hour < WIN_HOURS[1])
    win_idx = np.where(in_win)[0]
    o = df['open'].values; h = df['high'].values; l_ = df['low'].values; c = df['close'].values
    atr = df['atr_60'].values
    rows = label_walk_both_sides(o, h, l_, c, atr, win_idx, H_MIN_NEW, TP_M, SL_M)
    if not rows:
        return None, None
    feat_cols = [col for col in df.columns
                 if col not in ('open','high','low','close','volume','symbol','side','label','tp_pts','sl_pts')
                 and pd.api.types.is_numeric_dtype(df[col])]
    feat_df = df[feat_cols].copy()
    idxs = [r[0] for r in rows]
    sides = [r[1] for r in rows]
    tps = [r[3] for r in rows]
    sls = [r[4] for r in rows]
    out_features = feat_df.iloc[idxs].copy()
    out_features['_orig_side'] = sides
    out_features['tp_pts'] = tps
    out_features['sl_pts'] = sls
    out_features.index = df.index[idxs]

    valid = (out_features[feat_cols].isna().sum(axis=1) < len(feat_cols) * 0.5)
    out_features = out_features[valid].copy()

    is_long = (out_features['_orig_side'] == 0).values
    is_short = (out_features['_orig_side'] == 1).values
    X_L = out_features.loc[is_long, top60_L].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    X_S = out_features.loc[is_short, top60_S].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    p_L = hgb_L.predict_proba(X_L)[:, 1] if len(X_L) else np.array([])
    p_S = hgb_S.predict_proba(X_S)[:, 1] if len(X_S) else np.array([])

    # Build chronological fire candidates: at each bar, both LONG and SHORT predictions.
    # Then pick the higher-confidence side per bar; that's THE candidate at that bar.
    long_part = out_features[is_long].copy()
    long_part['p'] = p_L
    long_part['side'] = 0
    short_part = out_features[is_short].copy()
    short_part['p'] = p_S
    short_part['side'] = 1
    combined = pd.concat([long_part, short_part], axis=0).sort_values(['_orig_side'])
    # Within same ts, keep the row with higher p
    combined = combined.sort_values(['p'], ascending=False)
    combined = combined[~combined.index.duplicated(keep='first')].sort_index()

    return df, combined


def replay_single_position(bars_df, fire_df, thr, daily_dd_cap, horizon_min):
    """Replay chronologically, single-position only. Returns list of completed trades."""
    open_arr = bars_df['open'].values
    high_arr = bars_df['high'].values
    low_arr = bars_df['low'].values
    close_arr = bars_df['close'].values
    bars_idx = bars_df.index
    bars_idx_lookup = pd.Index(bars_idx)

    # Build fire signal lookup: ts -> (side, p, tp_pts, sl_pts)
    fire_lookup = fire_df[['side', 'p', 'tp_pts', 'sl_pts']].to_dict(orient='index')

    trades = []
    in_pos = False
    pos_state = None
    last_date = None
    daily_cum_pnl = 0.0
    daily_muted = False

    for i, ts in enumerate(bars_idx):
        date = ts.normalize()
        if date != last_date:
            daily_cum_pnl = 0.0
            daily_muted = False
            last_date = date

        if in_pos:
            # Check if position closes this bar
            tp_hit = False; sl_hit = False
            if pos_state['side'] == 0:  # LONG
                if low_arr[i] <= pos_state['sl_price']:
                    sl_hit = True
                if high_arr[i] >= pos_state['tp_price']:
                    tp_hit = True
            else:  # SHORT
                if high_arr[i] >= pos_state['sl_price']:
                    sl_hit = True
                if low_arr[i] <= pos_state['tp_price']:
                    tp_hit = True
            # If both hit in same bar, treat as SL (conservative)
            closed = False; pnl_pts = 0
            if sl_hit:
                closed = True
                pnl_pts = -pos_state['sl_pts']
            elif tp_hit:
                closed = True
                pnl_pts = pos_state['tp_pts']
            elif i >= pos_state['horizon_end_idx']:
                # Close at this bar's open (timeout)
                closed = True
                if pos_state['side'] == 0:
                    pnl_pts = open_arr[i] - pos_state['entry_price']
                else:
                    pnl_pts = pos_state['entry_price'] - open_arr[i]
            if closed:
                pnl = pnl_pts * PT_USD
                daily_cum_pnl += pnl
                trades.append({
                    'entry_ts': pos_state['entry_ts'],
                    'exit_ts': ts,
                    'side': pos_state['side'],
                    'pnl': pnl,
                    'pnl_pts': pnl_pts,
                    'win': pnl > 0,
                })
                in_pos = False
                pos_state = None
                if daily_cum_pnl <= -daily_dd_cap:
                    daily_muted = True

        # Check for fire (only if NOT in position and not muted today)
        if not in_pos and not daily_muted and ts in fire_lookup:
            sig = fire_lookup[ts]
            if sig['p'] >= thr:
                # Only fire if window allows (already filtered in fire_df, but double-check)
                if WIN_HOURS[0] <= ts.hour < WIN_HOURS[1]:
                    # Use NEXT bar's open as entry
                    if i + 1 < len(bars_idx):
                        entry_price = open_arr[i + 1]
                        side = sig['side']
                        tp_pts = sig['tp_pts']; sl_pts = sig['sl_pts']
                        if side == 0:  # LONG
                            tp_price = entry_price + tp_pts
                            sl_price = entry_price - sl_pts
                        else:  # SHORT
                            tp_price = entry_price - tp_pts
                            sl_price = entry_price + sl_pts
                        horizon_end_idx = min(i + 1 + horizon_min, len(bars_idx) - 1)
                        in_pos = True
                        pos_state = {
                            'entry_ts': bars_idx[i + 1],
                            'entry_price': entry_price,
                            'side': side,
                            'tp_pts': tp_pts,
                            'sl_pts': sl_pts,
                            'tp_price': tp_price,
                            'sl_price': sl_price,
                            'horizon_end_idx': horizon_end_idx,
                        }
    return pd.DataFrame(trades)


def summarize(trades_df, partition_name, days, baseline_days_per_month=None):
    if trades_df is None or len(trades_df) == 0:
        print(f'\n=== {partition_name} ===  no trades')
        return
    print(f'\n=== {partition_name} ===')
    n = len(trades_df)
    wins = int(trades_df['win'].sum())
    wr = wins / n
    pnl = float(trades_df['pnl'].sum())
    avg = pnl / n
    per_day = n / max(days, 1)
    print(f'  rate: {per_day:.2f}/d  WR: {wr*100:.2f}%  avg/trade: ${avg:.0f}  total PnL: ${pnl:+,.0f}  ({n} trades over {days} sessions)')

    trades_df['date'] = pd.to_datetime(trades_df['exit_ts']).dt.normalize()
    trades_df['month'] = pd.to_datetime(trades_df['exit_ts']).dt.to_period('M').astype(str)
    trades_df['year'] = pd.to_datetime(trades_df['exit_ts']).dt.year
    print(f'\n  per-month:')
    print(f'  {"month":>9s} {"days":>5s} {"n":>5s} {"/d":>5s} {"WR%":>6s} {"PnL":>10s}')
    print(f'  {"-"*9} {"-"*5} {"-"*5} {"-"*5} {"-"*6} {"-"*10}')
    monthly = trades_df.groupby('month').agg(
        n=('pnl','count'), w=('win','sum'), pnl=('pnl','sum'),
        days=('date','nunique'),
    ).reset_index()
    monthly['wr'] = monthly['w'] / monthly['n']
    monthly['per_day'] = monthly['n'] / monthly['days']
    for _, m in monthly.iterrows():
        print(f'  {m["month"]:>9s} {int(m["days"]):>5d} {int(m["n"]):>5d} {m["per_day"]:>4.2f} {m["wr"]*100:>5.1f}% ${m["pnl"]:>+9,.0f}')

    print(f'\n  per-year:')
    print(f'  {"year":>5s} {"n":>5s} {"WR%":>6s} {"PnL":>10s} {"DD":>10s}')
    for yr, grp in trades_df.groupby('year'):
        grp_sorted = grp.sort_values('exit_ts')
        cum = grp_sorted['pnl'].cumsum()
        peak = cum.cummax()
        dd = float((cum - peak).min())
        yr_n = len(grp_sorted)
        yr_wr = grp_sorted['win'].sum() / yr_n
        yr_pnl = float(grp_sorted['pnl'].sum())
        flag = '✅' if (yr_pnl > 0 and yr_wr > 0.5 and dd >= -900) else '❌'
        print(f'  {int(yr):>5d} {yr_n:>5d} {yr_wr*100:>5.1f}% ${yr_pnl:>+9,.0f} ${dd:>+9,.0f}  {flag}')


# Build + replay each partition
for name in ['VAL_biden', 'TRAIN_2025', 'OOS_2026']:
    bars_df, fire_df = predict_partition(name)
    if bars_df is None: continue
    print(f'  fire candidates: {len(fire_df):,}')
    trades = replay_single_position(bars_df, fire_df, THR, CAP, H_MIN_NEW)
    days = bars_df.index.normalize().nunique()
    note = '⚠️ in-sample' if name == 'TRAIN_2025' else '✓ true OOS' if name == 'OOS_2026' else ''
    summarize(trades, f'{name}  {note}', days)

print(f'\nWall: {time.time()-t0:.0f}s')
