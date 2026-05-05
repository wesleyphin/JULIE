"""Iter 12 — emit 2025 + 2026 per-month + per-year stats for the best config.

Best config: TP=1.5×ATR / SL=1.0×ATR / thr=0.75 / cap=$200 / 60-min horizon
LONG model: iter12_1.5_1.0_L.pkl
SHORT model: iter12_1.5_1.0_S.pkl
Combine: at each bar, fire on whichever side has higher predicted-prob, if ≥thr.
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
from stdev_ml_iter12_both_sides import label_walk_both_sides, evaluate_combined

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
print(f'LONG AUC:  VAL={bL["auc_val"]:.4f}, OOS={bL["auc_oos"]:.4f}')
print(f'SHORT AUC: VAL={bS["auc_val"]:.4f}, OOS={bS["auc_oos"]:.4f}\n')


def predict_partition(name):
    s, e = PERIODS[name]
    print(f'[{name}] building features {s}→{e}...')
    bars = load_period(s, e)
    df = compute_features(bars)
    in_win = (df.index.hour >= WIN_HOURS[0]) & (df.index.hour < WIN_HOURS[1])
    win_idx = np.where(in_win)[0]
    o = df['open'].values; h = df['high'].values; l = df['low'].values; c = df['close'].values
    atr = df['atr_60'].values
    rows = label_walk_both_sides(o, h, l, c, atr, win_idx, H_MIN_NEW, TP_M, SL_M)
    if not rows:
        return None
    feat_cols = [col for col in df.columns
                 if col not in ('open','high','low','close','volume','symbol','side','label','tp_pts','sl_pts')
                 and pd.api.types.is_numeric_dtype(df[col])]
    feat_df = df[feat_cols].copy()
    idxs = [r[0] for r in rows]
    sides = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    tps = [r[3] for r in rows]
    sls = [r[4] for r in rows]
    out = feat_df.iloc[idxs].copy()
    out['_orig_side'] = sides
    out['label'] = labels
    out['tp_pts'] = tps
    out['sl_pts'] = sls
    out.index = df.index[idxs]

    valid = (out[feat_cols].isna().sum(axis=1) < len(feat_cols) * 0.5)
    out = out[valid].copy()

    is_long = (out['_orig_side'] == 0).values
    is_short = (out['_orig_side'] == 1).values
    X_L = out.loc[is_long, top60_L].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    X_S = out.loc[is_short, top60_S].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    p_L = hgb_L.predict_proba(X_L)[:, 1] if len(X_L) else np.array([])
    p_S = hgb_S.predict_proba(X_S)[:, 1] if len(X_S) else np.array([])
    return out, is_long, is_short, p_L, p_S


def show(label, out, is_long, is_short, p_L, p_S, days, note):
    if out is None:
        print(f'\n=== {label} ===  no data')
        return
    long_rows = out[is_long]
    short_rows = out[is_short]
    stats = evaluate_combined(
        p_L, p_S,
        long_rows['label'].values, short_rows['label'].values,
        long_rows.index.values, short_rows.index.values,
        long_rows['tp_pts'].values, short_rows['tp_pts'].values,
        long_rows['sl_pts'].values, short_rows['sl_pts'].values,
        THR, days, CAP,
    )
    if stats is None:
        print(f'\n=== {label} ===  no fires')
        return
    print(f'\n=== {label} ===  {note}')
    print(f'  rate: {stats["per_day"]:.2f}/d  WR: {stats["wr"]*100:.2f}%  avg/trade: ${stats["avg"]:.0f}  total PnL: ${stats["pnl"]:+,.0f}')
    print(f'\n  per-month:')
    print(f'  {"month":>9s} {"days":>5s} {"n":>5s} {"/d":>5s} {"WR%":>6s} {"PnL":>10s}')
    print(f'  {"-"*9} {"-"*5} {"-"*5} {"-"*5} {"-"*6} {"-"*10}')
    for m in stats['monthly']:
        wr_str = f'{m["wr"]*100:>5.1f}%' if m['n'] else '   --'
        print(f'  {m["month"]:>9s} {m["days"]:>5d} {m["n"]:>5d} {m["per_day"]:>4.2f} {wr_str} ${m["pnl"]:>+9,.0f}')
    print(f'\n  per-year:')
    print(f'  {"year":>5s} {"n":>5s} {"WR%":>6s} {"PnL":>10s} {"DD":>10s}')
    for yy in stats['yearly']:
        flag = '✅' if (yy['pnl']>0 and yy['wr']>0.5 and yy['dd']>=-900) else '❌'
        print(f'  {yy["year"]:>5d} {yy["n"]:>5d} {yy["wr"]*100:>5.1f}% ${yy["pnl"]:>+9,.0f} ${yy["dd"]:>+9,.0f}  {flag}')


# Predict three relevant partitions
val_data = predict_partition('VAL_biden')
t25_data = predict_partition('TRAIN_2025')
oos_data = predict_partition('OOS_2026')


def slice_to_year(data, year):
    if data is None: return None
    out, is_long, is_short, p_L, p_S = data
    long_rows = out[is_long]; short_rows = out[is_short]
    long_yr_mask = (pd.DatetimeIndex(long_rows.index).year == year)
    short_yr_mask = (pd.DatetimeIndex(short_rows.index).year == year)
    new_long_rows = long_rows[long_yr_mask]
    new_short_rows = short_rows[short_yr_mask]
    new_p_L = p_L[long_yr_mask]
    new_p_S = p_S[short_yr_mask]
    # Rebuild the row-wise is_long/is_short masks for combined eval doesn't matter — eval_combined uses long/short separately.
    return new_long_rows, new_short_rows, new_p_L, new_p_S


# 2025 from VAL slice (Jan 1-19)
if val_data is not None:
    out, is_long, is_short, p_L, p_S = val_data
    long_rows = out[is_long]; short_rows = out[is_short]
    long_2025 = pd.DatetimeIndex(long_rows.index).year == 2025
    short_2025 = pd.DatetimeIndex(short_rows.index).year == 2025
    long_2025_rows = long_rows[long_2025]; short_2025_rows = short_rows[short_2025]
    p_L_2025 = p_L[long_2025]; p_S_2025 = p_S[short_2025]
    days = pd.DatetimeIndex(long_2025_rows.index.union(short_2025_rows.index)).normalize().nunique() if len(long_2025_rows) or len(short_2025_rows) else 0
    if days > 0:
        stats = evaluate_combined(p_L_2025, p_S_2025,
                                   long_2025_rows['label'].values, short_2025_rows['label'].values,
                                   long_2025_rows.index.values, short_2025_rows.index.values,
                                   long_2025_rows['tp_pts'].values, short_2025_rows['tp_pts'].values,
                                   long_2025_rows['sl_pts'].values, short_2025_rows['sl_pts'].values,
                                   THR, days, CAP)
        if stats:
            print(f'\n=== 2025 — Jan 1–19 (true OOS, VAL slice) === (only 13 trading days)')
            print(f'  rate: {stats["per_day"]:.2f}/d  WR: {stats["wr"]*100:.2f}%  avg/trade: ${stats["avg"]:.0f}  total PnL: ${stats["pnl"]:+,.0f}')
            print(f'\n  per-month:')
            print(f'  {"month":>9s} {"days":>5s} {"n":>5s} {"/d":>5s} {"WR%":>6s} {"PnL":>10s}')
            for m in stats['monthly']:
                wr_str = f'{m["wr"]*100:>5.1f}%' if m['n'] else '   --'
                print(f'  {m["month"]:>9s} {m["days"]:>5d} {m["n"]:>5d} {m["per_day"]:>4.2f} {wr_str} ${m["pnl"]:>+9,.0f}')
            for yy in stats['yearly']:
                flag = '✅' if (yy['pnl']>0 and yy['wr']>0.5 and yy['dd']>=-900) else '❌'
                print(f'  per-year: {yy["year"]} n={yy["n"]} WR={yy["wr"]*100:.1f}% PnL=${yy["pnl"]:+,.0f} DD=${yy["dd"]:+,.0f} {flag}')

# TRAIN_2025 (in-sample, Jan 20-Dec 31)
if t25_data is not None:
    out, is_long, is_short, p_L, p_S = t25_data
    days = pd.DatetimeIndex(out.index).normalize().nunique()
    show('2025 — Jan 20–Dec 31 (IN-SAMPLE — model trained on this)',
         out, is_long, is_short, p_L, p_S, days,
         '⚠️ in-sample, predictions inflated')

# OOS_2026
if oos_data is not None:
    out, is_long, is_short, p_L, p_S = oos_data
    days = pd.DatetimeIndex(out.index).normalize().nunique()
    show('2026 — Jan 1–Apr 30 (OOS, true holdout)',
         out, is_long, is_short, p_L, p_S, days,
         '✓ true OOS — no peeking')

print(f'\nWall: {time.time()-t0:.0f}s')
