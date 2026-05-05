"""Diagnose why iter11 OOS fires 676 in Jan-2026 then 4 in Feb-2026.

Hypotheses:
  H1. Feature distribution shift between Jan and Feb (model sees Jan as similar
      to TRAIN, Feb as alien)
  H2. A single feature (likely vwap_60 — price-level) is driving prediction;
      Jan price level happens to match TRAIN distribution, Feb doesn't
  H3. The circuit breaker is muting Feb after one bad day
  H4. Labeling failure — Feb has fewer labeled bars (TP/SL never hit within 60min
      due to compressed range), so fewer fire candidates

Output:
  - per-month: count of labeled rows, count fired pre-circuit, count fired post,
    mean/std of predictions, top feature distributions
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings

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
THR = 0.79

# Build OOS_2026
s, e = PERIODS['OOS_2026']
print(f'Building OOS_2026 features...')
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
Xfull = out.drop(columns=['label','tp_pts','sl_pts']).astype(np.float32)
valid_mask = (Xfull.isna().sum(axis=1) < len(Xfull.columns) * 0.5)
out = out[valid_mask.values].copy()

X = out[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
p = hgb.predict_proba(X)[:, 1]
out['p'] = p
out['fired_pre'] = p >= THR
out['month'] = pd.DatetimeIndex(out.index).to_period('M').astype(str)

# Also evaluate post-circuit (for comparison)
days_per_month = pd.DatetimeIndex(out.index).normalize().to_series().reset_index(drop=True).groupby(out['month'].values).nunique().to_dict()

print(f'\n=== OOS 2026 — per-month diagnostics ===\n')
print(f'  {"month":>9s} {"days":>5s} {"labeled":>8s} {"fired":>7s} {"fire%":>6s} {"p mean":>8s} {"p std":>7s} {"p p50":>8s} {"p p90":>8s} {"p p99":>8s} {"WR(fired)":>10s} {"baseWR":>7s}')
print(f'  {"-"*9} {"-"*5} {"-"*8} {"-"*7} {"-"*6} {"-"*8} {"-"*7} {"-"*8} {"-"*8} {"-"*8} {"-"*10} {"-"*7}')
for mo, grp in out.groupby('month'):
    n_labeled = len(grp)
    n_fired = int(grp['fired_pre'].sum())
    fire_pct = 100 * n_fired / max(n_labeled, 1)
    p_mean = grp['p'].mean(); p_std = grp['p'].std()
    p50, p90, p99 = grp['p'].quantile([0.5, 0.9, 0.99])
    wr_fired = (grp[grp['fired_pre']]['label'].mean() * 100) if n_fired else 0
    base_wr = grp['label'].mean() * 100
    days = days_per_month.get(mo, 0)
    print(f'  {mo:>9s} {days:>5d} {n_labeled:>8d} {n_fired:>7d} {fire_pct:>5.1f}% {p_mean:>7.3f} {p_std:>6.3f} {p50:>7.3f} {p90:>7.3f} {p99:>7.3f} {wr_fired:>8.1f}% {base_wr:>5.1f}%')

# Also: what's the distribution of vwap_60 / atr_60 / sigma_60 by month?
print(f'\n=== Feature distributions by month (suspicion: regime drift) ===\n')
for col in ['vwap_60', 'atr_60', 'sigma_60', 'sigma_15_tod_z', 'sigma_of_sigma_15_60']:
    if col not in out.columns:
        # may have been filtered. check df
        if col in df.columns:
            # backfill from df
            mask_idx = out.index
            out[col] = df.loc[mask_idx, col].values
        else:
            print(f'  {col}: NOT FOUND'); continue
    print(f'  {col}:')
    for mo, grp in out.groupby('month'):
        vals = grp[col].dropna()
        if len(vals) == 0: continue
        print(f'    {mo}: n={len(vals):>5d}  mean={vals.mean():>10.4f}  std={vals.std():>10.4f}  p50={vals.quantile(0.5):>10.4f}')
    print()

print(f'Wall: {time.time()-t0:.0f}s')
