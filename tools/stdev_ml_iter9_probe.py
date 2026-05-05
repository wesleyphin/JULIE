"""Iter 9 step A — probe.

Run ONLY TP=40/SL=15 thr=0.83 cap=$200 and dump per-year stats so we can
identify which year fails the all-positive constraint. Iter 8 ran the same
config but only printed yearly stats for *passing* configs (none passed).

Saves both the trained HGB model + per-year stats so step B (Fed-pivot kill
switch) can re-use the prediction stream without retraining.
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, '/Users/wes/Downloads/JULIE001/tools')
from stdev_ml_iter7_macro import (
    fetch_macros, PERIODS, PT_USD,
)
from stdev_ml_iter7_macro import build_period as build_iter7
from stdev_ml_iter8_circuit import evaluate_with_circuit, check_relaxed

warnings.filterwarnings("ignore")

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter9_probe'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    TP, SL, THR, CAP = 40, 15, 0.83, 200
    breakeven = SL / (TP + SL)

    macro = fetch_macros()

    print(f'\n========== Probe: TP={TP}/SL={SL} thr={THR} cap=${CAP} ==========')
    all_data = {}
    for name, (s, e) in PERIODS.items():
        df = build_iter7(name, s, e, TP, SL, macro)
        if df is None or df.empty: continue
        all_data[name] = df

    train = pd.concat([all_data.get('TRAIN_t1', pd.DataFrame()),
                       all_data.get('TRAIN_2025', pd.DataFrame())], axis=0)
    val = all_data['VAL_biden']
    feat_cols = [c for c in train.columns if c != 'label']
    Xt = train[feat_cols].astype(np.float32); yt = train['label'].astype(np.int8).values
    Xv = val[feat_cols].astype(np.float32); yv = val['label'].astype(np.int8).values
    nan_t = Xt.isna().sum(axis=1); nan_v = Xv.isna().sum(axis=1)
    valid_t = (nan_t < len(feat_cols) * 0.5); valid_v = (nan_v < len(feat_cols) * 0.5)
    Xt = Xt[valid_t].reset_index(drop=True); yt = yt[valid_t.values]
    Xv = Xv[valid_v].reset_index(drop=True); yv = yv[valid_v.values]
    val_ts = val.index[valid_v.values]
    val_days = pd.to_datetime(val_ts).normalize().nunique()

    long_t = (Xt['side'] == 0).values
    long_v = (Xv['side'] == 0).values
    Xt_L = Xt[long_t].reset_index(drop=True); yt_L = yt[long_t]
    Xv_L = Xv[long_v].reset_index(drop=True); yv_L = yv[long_v]
    val_ts_L = val_ts[long_v]
    print(f'  LONG TRAIN={len(Xt_L):,}, VAL={len(Xv_L):,}, days={val_days}, baseline WR={yv_L.mean()*100:.1f}%')

    Xt_clean = Xt_L.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    F, _ = f_classif(Xt_clean.values, yt_L)
    fa = pd.DataFrame({'f': feat_cols, 'F': F}).sort_values('F', ascending=False)
    top60 = fa.head(60)['f'].tolist()

    cluster_features = [f for f in top60 if any(k in f for k in ['sigma','z_','slope','momentum','hurst','vwap_dist','range_pos','rsi','macro'])][:20]
    Xt_c = Xt_L[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    Xv_c = Xv_L[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    sc = StandardScaler(); Xt_s = sc.fit_transform(Xt_c); Xv_s = sc.transform(Xv_c)
    pca = PCA(n_components=10, random_state=42); Xt_p = pca.fit_transform(Xt_s); Xv_p = pca.transform(Xv_s)
    best_K = None
    for K in [4, 6, 8, 10, 12]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        ct = km.fit_predict(Xt_p); cv = km.predict(Xv_p)
        df_t = pd.DataFrame({'c': ct, 'y': yt_L}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        df_v = pd.DataFrame({'c': cv, 'y': yv_L}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        good = [c for c in df_t.index if df_t.loc[c,'wr'] > breakeven + 0.03 and df_t.loc[c,'n'] > 1000
                and c in df_v.index and df_v.loc[c,'wr'] > breakeven + 0.02]
        if good and (not best_K or len(good) > len(best_K['good'])):
            best_K = {'K': K, 'good': good, 'cv': cv, 'sc': sc, 'pca': pca, 'km': km}
    cluster_mask_v = np.isin(best_K['cv'], best_K['good']) if best_K else np.ones(len(Xv_L), dtype=bool)

    Xt_hgb = Xt_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    Xv_hgb = Xv_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400,
                                         early_stopping=True, n_iter_no_change=15, random_state=42)
    hgb.fit(Xt_hgb, yt_L)
    pv = hgb.predict_proba(Xv_hgb)[:, 1]
    auc = roc_auc_score(yv_L, pv)
    print(f'  HGB Val AUC: {auc:.4f}')

    fired = (pv >= THR) & cluster_mask_v
    print(f'  fired pre-circuit: {int(fired.sum()):,}')
    stats = evaluate_with_circuit(pv, yv_L, val_ts_L, fired, TP, SL, val_days, CAP)
    passed, fails, pct_pos = check_relaxed(stats)
    print(f'\n  per-day: {stats["per_day"]:.2f}, WR: {stats["wr"]*100:.1f}%, avg: ${stats["avg"]:.0f}, PnL: ${stats["pnl"]:+,.0f}, mos+: {pct_pos*100:.0f}%')
    print(f'  passed: {passed}; fails: {fails}\n')

    print(f'  {"year":>5s} {"n":>6s} {"WR%":>6s} {"PnL":>10s} {"DD":>10s}  status')
    print(f'  {"-"*5} {"-"*6} {"-"*6} {"-"*10} {"-"*10}  {"-"*30}')
    for y in stats['yearly']:
        ok_pnl = y['pnl'] > 0
        ok_wr = y['wr'] > 0.5
        ok_dd = y['dd'] >= -900
        marks = []
        if not ok_pnl: marks.append('PnL≤0')
        if not ok_wr: marks.append('WR≤50')
        if not ok_dd: marks.append('DD<-900')
        status = ' '.join(marks) if marks else 'OK'
        flag = '❌' if marks else '✅'
        print(f'  {y["year"]:>5d} {y["n"]:>6d} {y["wr"]*100:>5.1f}% ${y["pnl"]:>+9,.0f} ${y["dd"]:>+9,.0f}  {flag} {status}')

    # Save artifacts
    joblib.dump({
        'hgb': hgb, 'features': top60, 'tp': TP, 'sl': SL, 'threshold': THR,
        'daily_dd_cap': CAP, 'val_auc': float(auc),
        'cluster_features': cluster_features,
        'scaler': best_K['sc'] if best_K else None,
        'pca': best_K['pca'] if best_K else None,
        'kmeans': best_K['km'] if best_K else None,
        'good_clusters': best_K['good'] if best_K else None,
    }, OUT / 'iter9_probe_model.pkl')

    # Save the per-bar prediction stream for step B (Fed-pivot kill switch development)
    val_meta = pd.DataFrame({
        'ts': pd.to_datetime(val_ts_L),
        'p': pv,
        'fired_pre_circuit': fired,
        'y': yv_L,
    })
    val_meta.to_parquet(OUT / 'iter9_val_predictions.parquet', index=False)

    # Save the macro-augmented Xv so step B can develop the kill switch on the same exact rows
    Xv_L.to_parquet(OUT / 'iter9_val_features.parquet', index=False)

    with open(OUT / 'iter9_probe_summary.json', 'w') as f:
        json.dump({
            'tp': TP, 'sl': SL, 'threshold': THR, 'daily_dd_cap': CAP,
            'val_auc': float(auc),
            'wr': stats['wr'], 'avg': stats['avg'], 'pnl': stats['pnl'],
            'per_day': stats['per_day'], 'mos_pos_pct': pct_pos,
            'yearly': stats['yearly'],
            'monthly': stats['monthly'],
            'failing_years': [y['year'] for y in stats['yearly']
                              if not (y['pnl'] > 0 and y['wr'] > 0.5 and y['dd'] >= -900)],
            'fails': fails,
            'passed': passed,
        }, f, indent=2, default=str)

    print(f'\n  Saved: {OUT}/iter9_probe_model.pkl, iter9_val_predictions.parquet, iter9_val_features.parquet, iter9_probe_summary.json')
    print(f'\nWall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
