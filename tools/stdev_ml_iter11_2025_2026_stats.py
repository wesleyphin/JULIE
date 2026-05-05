"""Iter 11 final probe — emit 2025 (VAL Jan slice + TRAIN_2025 in-sample) and 2026 OOS
per-month + per-year stats for the ship candidate.

Ship candidate: TP=1.5×ATR / SL=1.0×ATR / thr=0.79 / cap=$200, 60-min horizon.

Notes:
- TRAIN_2025 (Jan 20–Dec 31) was in the model fit set; flagged.
- VAL_biden's 2025 slice is Jan 1–19 only (true OOS but tiny sample).
- OOS_2026 is the only meaningful unseen partition.
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
    compute_features, load_period, label_walk_atr, evaluate_with_circuit_atr,
    PERIODS, WIN_HOURS, H_MIN_NEW, PT_USD,
)

warnings.filterwarnings("ignore")

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter11_atr'

t0 = time.time()
b = joblib.load(OUT / 'iter11_atr_1.5_1.0_model.pkl')
hgb = b['hgb']; top60 = b['features']; TP_M = b['tp_mult']; SL_M = b['sl_mult']
THR = 0.79; CAP = 200  # ship candidate, NOT b['threshold']
print(f'Ship candidate: TP={TP_M}×ATR / SL={SL_M}×ATR, threshold={THR}, daily_dd_cap=${CAP}, horizon={H_MIN_NEW}min')
print(f'Model AUCs: VAL={b["val_auc"]:.4f}, OOS={b["oos_auc"]:.4f}\n')


def predict_partition(name):
    s, e = PERIODS[name]
    print(f'[{name}] building features {s}→{e}...')
    bars = load_period(s, e)
    df = compute_features(bars)
    in_win = (df.index.hour >= WIN_HOURS[0]) & (df.index.hour < WIN_HOURS[1])
    win_idx = np.where(in_win)[0]
    o = df['open'].values; h = df['high'].values; l = df['low'].values; c = df['close'].values
    atr = df['atr_60'].values
    labels, sides, tp_arr, sl_arr = label_walk_atr(o, h, l, c, atr, win_idx, H_MIN_NEW, TP_M, SL_M)
    valid = labels != -1
    feat_cols = [col for col in df.columns
                 if col not in ('open','high','low','close','volume','symbol')
                 and pd.api.types.is_numeric_dtype(df[col])]
    out = df.iloc[valid][feat_cols].copy()
    out['label'] = labels[valid]; out['tp_pts'] = tp_arr[valid]; out['sl_pts'] = sl_arr[valid]
    out['side'] = sides[valid]

    # Filter rows with too many NaN
    Xfull = out.drop(columns=['label','tp_pts','sl_pts','side']).astype(np.float32)
    valid_mask = (Xfull.isna().sum(axis=1) < len(Xfull.columns) * 0.5)
    out = out[valid_mask.values].copy()

    # Build top60 feature matrix in correct order
    X = out[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    p = hgb.predict_proba(X)[:, 1]
    return out, p


def show(label, out, p, note):
    fired = p >= THR
    if not fired.any():
        print(f'\n=== {label} ===  no fires')
        return
    days = out.index.normalize().nunique()
    stats = evaluate_with_circuit_atr(p, out['label'].values, out.index.values,
                                      out['tp_pts'].values, out['sl_pts'].values,
                                      fired, days, CAP)
    if stats is None:
        print(f'\n=== {label} ===  evaluation failed')
        return
    print(f'\n=== {label} ===  {note}')
    print(f'  rate: {stats["per_day"]:.2f}/d  WR: {stats["wr"]*100:.2f}%  avg/trade: ${stats["avg"]:.0f}  total PnL: ${stats["pnl"]:+,.0f}')
    print(f'\n  per-month:')
    print(f'  {"month":>9s} {"n":>5s} {"WR%":>6s} {"PnL":>10s}')
    print(f'  {"-"*9} {"-"*5} {"-"*6} {"-"*10}')
    for m in stats['monthly']:
        print(f'  {str(m["month"]):>9s} {m["n"]:>5d} {m["wr"]*100:>5.1f}% ${m["pnl"]:>+9,.0f}')
    print(f'\n  per-year:')
    print(f'  {"year":>5s} {"n":>5s} {"WR%":>6s} {"PnL":>10s} {"DD":>10s}')
    for yy in stats['yearly']:
        flag = '✅' if (yy['pnl']>0 and yy['wr']>0.5 and yy['dd']>=-900) else '❌'
        print(f'  {yy["year"]:>5d} {yy["n"]:>5d} {yy["wr"]*100:>5.1f}% ${yy["pnl"]:>+9,.0f} ${yy["dd"]:>+9,.0f}  {flag}')


# Predict all three relevant partitions
val_out, val_p = predict_partition('VAL_biden')
t25_out, t25_p = predict_partition('TRAIN_2025')
oos_out, oos_p = predict_partition('OOS_2026')

# 2025 (VAL Jan slice — held-out, but tiny)
val_dt = pd.DatetimeIndex(val_out.index)
val_2025_mask = (val_dt.year == 2025)
val_2025_out = val_out[val_2025_mask]
val_2025_p = val_p[val_2025_mask]
show('2025 — Jan 1–19 (true OOS, VAL slice)', val_2025_out, val_2025_p,
     '(only 13 trading days — sample is small)')

# 2025 (TRAIN_2025 — IN SAMPLE, flagged)
show('2025 — Jan 20–Dec 31 (IN-SAMPLE — model trained on this)', t25_out, t25_p,
     '⚠️ in-sample, predictions inflated')

# 2026 — OOS
show('2026 — Jan 1–Apr 30 (OOS, true holdout)', oos_out, oos_p,
     '✓ true OOS — no peeking')

print(f'\nWall: {time.time()-t0:.0f}s')
