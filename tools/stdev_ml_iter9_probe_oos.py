"""Iter 9 step A.5 — evaluate OOS_2026 using the saved iter9 probe model.

Question: does the 2025 partition's tiny n in VAL_biden indicate a problem,
or is it just a 13-day sliver? Also: does 2026 OOS pass the constraints?
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, '/Users/wes/Downloads/JULIE001/tools')
from stdev_ml_iter7_macro import fetch_macros, PERIODS
from stdev_ml_iter7_macro import build_period as build_iter7
from stdev_ml_iter8_circuit import evaluate_with_circuit, check_relaxed

warnings.filterwarnings("ignore")

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter9_probe'

t0 = time.time()
bundle = joblib.load(OUT / 'iter9_probe_model.pkl')
hgb = bundle['hgb']
top60 = bundle['features']
TP, SL, THR, CAP = bundle['tp'], bundle['sl'], bundle['threshold'], bundle['daily_dd_cap']
sc = bundle['scaler']; pca = bundle['pca']; km = bundle['kmeans']; good = bundle['good_clusters']
cluster_features = bundle['cluster_features']

print(f'Loaded probe bundle: TP={TP}/SL={SL} thr={THR} cap=${CAP}, AUC={bundle["val_auc"]:.4f}')

macro = fetch_macros()

# Build OOS_2026
s, e = PERIODS['OOS_2026']
oos = build_iter7('OOS_2026', s, e, TP, SL, macro)
print(f'OOS_2026 rows: {len(oos):,}, period {s} → {e}')

feat_cols = [c for c in oos.columns if c != 'label']
X = oos[feat_cols].astype(np.float32)
y = oos['label'].astype(np.int8).values
nan_x = X.isna().sum(axis=1)
valid = (nan_x < len(feat_cols) * 0.5)
X = X[valid].reset_index(drop=True); y = y[valid.values]
ts = oos.index[valid.values]
days = pd.to_datetime(ts).normalize().nunique()

long_mask = (X['side'] == 0).values
X_L = X[long_mask].reset_index(drop=True); y_L = y[long_mask]; ts_L = ts[long_mask]
print(f'OOS LONG: {len(X_L):,}, days={days}, baseline WR={y_L.mean()*100:.1f}%')

# Cluster mask using saved transformers
X_c = X_L[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
X_s = sc.transform(X_c); X_p = pca.transform(X_s)
clusters = km.predict(X_p)
cluster_mask = np.isin(clusters, good) if good is not None else np.ones(len(X_L), dtype=bool)
print(f'cluster keep: {int(cluster_mask.sum()):,}/{len(X_L):,}')

# HGB predict
X_hgb = X_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
p = hgb.predict_proba(X_hgb)[:, 1]
fired = (p >= THR) & cluster_mask
print(f'fired pre-circuit: {int(fired.sum()):,}')

stats = evaluate_with_circuit(p, y_L, ts_L, fired, TP, SL, days, CAP)
passed, fails, pct_pos = check_relaxed(stats)
print(f'\n  per-day: {stats["per_day"]:.2f}, WR: {stats["wr"]*100:.1f}%, avg: ${stats["avg"]:.0f}, '
      f'PnL: ${stats["pnl"]:+,.0f}, mos+: {pct_pos*100:.0f}%')
print(f'  passed: {passed}; fails: {fails}\n')

print(f'  {"month":>8s} {"n":>5s} {"WR%":>6s} {"PnL":>10s}')
print(f'  {"-"*8} {"-"*5} {"-"*6} {"-"*10}')
for m in stats['monthly']:
    print(f'  {str(m["month"]):>8s} {m["n"]:>5d} {m["wr"]*100:>5.1f}% ${m["pnl"]:>+9,.0f}')

print(f'\n  {"year":>5s} {"n":>5s} {"WR%":>6s} {"PnL":>10s} {"DD":>10s}')
print(f'  {"-"*5} {"-"*5} {"-"*6} {"-"*10} {"-"*10}')
for yy in stats['yearly']:
    print(f'  {yy["year"]:>5d} {yy["n"]:>5d} {yy["wr"]*100:>5.1f}% ${yy["pnl"]:>+9,.0f} ${yy["dd"]:>+9,.0f}')

with open(OUT / 'iter9_oos_summary.json', 'w') as f:
    json.dump({
        'tp': TP, 'sl': SL, 'threshold': THR, 'daily_dd_cap': CAP,
        'wr': stats['wr'], 'avg': stats['avg'], 'pnl': stats['pnl'],
        'per_day': stats['per_day'], 'mos_pos_pct': pct_pos,
        'passed': passed, 'fails': fails,
        'yearly': stats['yearly'], 'monthly': stats['monthly'],
    }, f, indent=2, default=str)
print(f'\nWall: {time.time()-t0:.0f}s')
