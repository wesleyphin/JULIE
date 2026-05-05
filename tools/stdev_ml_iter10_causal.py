"""Iter 10 — fix the macro look-ahead leak in iter 7 and re-run the probe end-to-end.

Bug: iter7's `add_macro_features` reindexes daily macro values onto 1-min bars via
ffill. The yfinance daily index is the trading date at midnight ET, but the value
is the day's CLOSE (~4pm). So a bar at 11:30 ET on 2022-03-01 reads VIX's 4pm
close from the same day → look-ahead leakage of ~5 hours.

Fix: shift the macro DataFrame down by 1 row (one trading day) before reindexing.
Each bar then reads the PRIOR trading day's close, which IS causally available.

This script:
  1. Monkey-patches `stdev_ml_iter7_macro.add_macro_features` with a causal version
  2. Re-runs the probe (TP=40/SL=15 thr=0.83 cap=$200) on VAL_biden + OOS_2026
  3. Dumps per-year and per-month stats for both partitions
  4. Saves the trained HGB model so we can do follow-on threshold sweeps cheaply

Output: artifacts/stdev_ml_hr11_12/iter10_causal/
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
import stdev_ml_iter7_macro as it7
from stdev_ml_iter7_macro import fetch_macros, PERIODS

warnings.filterwarnings("ignore")

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter10_causal'
OUT.mkdir(parents=True, exist_ok=True)


# === CAUSAL macro feature fn — replaces the leaky version ===
def add_macro_features_causal(df, macro):
    """Causal version. Each bar reads the PRIOR trading day's close, not today's."""
    if macro is None: return df
    mc = macro.copy().sort_index()
    # Compute derived features on the ORIGINAL daily series (each row = that day's close)
    for col in list(mc.columns):
        s = mc[col]
        mc[f'{col}_z_20d'] = (s - s.rolling(20).mean()) / s.rolling(20).std().replace(0, np.nan)
        mc[f'{col}_chg_1d'] = s.pct_change(1)
        mc[f'{col}_chg_5d'] = s.pct_change(5)
    keep_cols = list(mc.columns)

    # CAUSAL SHIFT: row at index 2022-03-02 now holds 2022-03-01's close.
    # A bar at 2022-03-02 11:30 ffills to the 2022-03-02 anchor → reads 2022-03-01's close.
    mc = mc.shift(1)

    bar_idx = df.index
    mc_reindexed = mc.reindex(bar_idx, method='ffill')
    for col in keep_cols:
        df[f'macro_{col}'] = mc_reindexed[col].values
    return df


# Monkey-patch BEFORE importing anything that calls compute_features
it7.add_macro_features = add_macro_features_causal


def main():
    t0 = time.time()
    TP, SL, THR, CAP = 40, 15, 0.83, 200
    breakeven = SL / (TP + SL)

    macro = fetch_macros()

    # Smoke test the patch — print VIX value at 2022-03-01 11:30 ET via the patched function
    test_idx = pd.DatetimeIndex([pd.Timestamp('2022-03-01 11:30', tz='America/New_York')])
    test_df = pd.DataFrame(index=test_idx)
    test_df = add_macro_features_causal(test_df, macro)
    print(f'\n[CAUSALITY CHECK] At 2022-03-01 11:30 ET:')
    print(f'  macro_VIX = {test_df["macro_VIX"].iloc[0]:.2f}')
    print(f'  Pre-fix value was 33.32 (today\'s close — leaky).')
    print(f'  Post-fix should be 30.15 (2022-02-28 Mon close).\n')

    print(f'========== Iter 10 (causal) probe: TP={TP}/SL={SL} thr={THR} cap=${CAP} ==========\n')
    all_data = {}
    for name, (s, e) in PERIODS.items():
        df = it7.build_period(name, s, e, TP, SL, macro)
        if df is None or df.empty: continue
        all_data[name] = df

    train = pd.concat([all_data.get('TRAIN_t1', pd.DataFrame()),
                       all_data.get('TRAIN_2025', pd.DataFrame())], axis=0)
    val = all_data['VAL_biden']
    oos = all_data['OOS_2026']

    feat_cols = [c for c in train.columns if c != 'label']
    Xt = train[feat_cols].astype(np.float32); yt = train['label'].astype(np.int8).values
    Xv = val[feat_cols].astype(np.float32); yv = val['label'].astype(np.int8).values
    Xo = oos[feat_cols].astype(np.float32); yo = oos['label'].astype(np.int8).values
    valid_t = (Xt.isna().sum(axis=1) < len(feat_cols)*0.5)
    valid_v = (Xv.isna().sum(axis=1) < len(feat_cols)*0.5)
    valid_o = (Xo.isna().sum(axis=1) < len(feat_cols)*0.5)
    Xt = Xt[valid_t].reset_index(drop=True); yt = yt[valid_t.values]
    Xv = Xv[valid_v].reset_index(drop=True); yv = yv[valid_v.values]
    Xo = Xo[valid_o].reset_index(drop=True); yo = yo[valid_o.values]
    val_ts = val.index[valid_v.values]; oos_ts = oos.index[valid_o.values]
    val_days = pd.to_datetime(val_ts).normalize().nunique()
    oos_days = pd.to_datetime(oos_ts).normalize().nunique()

    long_t = (Xt['side']==0).values
    long_v = (Xv['side']==0).values
    long_o = (Xo['side']==0).values
    Xt_L = Xt[long_t].reset_index(drop=True); yt_L = yt[long_t]
    Xv_L = Xv[long_v].reset_index(drop=True); yv_L = yv[long_v]
    Xo_L = Xo[long_o].reset_index(drop=True); yo_L = yo[long_o]
    val_ts_L = val_ts[long_v]; oos_ts_L = oos_ts[long_o]
    print(f'  LONG TRAIN={len(Xt_L):,}, VAL={len(Xv_L):,} ({val_days}d), OOS={len(Xo_L):,} ({oos_days}d)')
    print(f'  baseline WR: VAL={yv_L.mean()*100:.1f}%, OOS={yo_L.mean()*100:.1f}%')

    Xt_clean = Xt_L.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    F, _ = f_classif(Xt_clean.values, yt_L)
    fa = pd.DataFrame({'f': feat_cols, 'F': F}).sort_values('F', ascending=False)
    print('\n  Top 10 F-stat features (post-fix):')
    for _, r in fa.head(10).iterrows():
        print(f'    {r["f"]:<35s} F={r["F"]:.1f}')
    top60 = fa.head(60)['f'].tolist()

    cluster_features = [f for f in top60 if any(k in f for k in ['sigma','z_','slope','momentum','hurst','vwap_dist','range_pos','rsi','macro'])][:20]
    Xt_c = Xt_L[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    Xv_c = Xv_L[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    Xo_c = Xo_L[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    sc = StandardScaler(); Xt_s = sc.fit_transform(Xt_c); Xv_s = sc.transform(Xv_c); Xo_s = sc.transform(Xo_c)
    pca = PCA(n_components=10, random_state=42); Xt_p = pca.fit_transform(Xt_s); Xv_p = pca.transform(Xv_s); Xo_p = pca.transform(Xo_s)
    best_K = None
    for K in [4, 6, 8, 10, 12]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        ct = km.fit_predict(Xt_p); cv = km.predict(Xv_p); co = km.predict(Xo_p)
        df_t = pd.DataFrame({'c': ct, 'y': yt_L}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        df_v = pd.DataFrame({'c': cv, 'y': yv_L}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        good = [c for c in df_t.index if df_t.loc[c,'wr'] > breakeven + 0.03 and df_t.loc[c,'n'] > 1000
                and c in df_v.index and df_v.loc[c,'wr'] > breakeven + 0.02]
        if good and (not best_K or len(good) > len(best_K['good'])):
            best_K = {'K': K, 'good': good, 'cv': cv, 'co': co, 'sc': sc, 'pca': pca, 'km': km}
    if best_K is None:
        cv_mask = np.ones(len(Xv_L), dtype=bool); co_mask = np.ones(len(Xo_L), dtype=bool)
        print('  No good clusters — disabling cluster filter')
    else:
        cv_mask = np.isin(best_K['cv'], best_K['good'])
        co_mask = np.isin(best_K['co'], best_K['good'])
        print(f'  K={best_K["K"]}, good clusters={best_K["good"]}')

    Xt_hgb = Xt_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    Xv_hgb = Xv_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    Xo_hgb = Xo_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400,
                                         early_stopping=True, n_iter_no_change=15, random_state=42)
    hgb.fit(Xt_hgb, yt_L)
    pv = hgb.predict_proba(Xv_hgb)[:, 1]
    po = hgb.predict_proba(Xo_hgb)[:, 1]
    auc_v = roc_auc_score(yv_L, pv)
    auc_o = roc_auc_score(yo_L, po)
    print(f'\n  HGB AUC: VAL={auc_v:.4f}, OOS={auc_o:.4f}')
    print(f'  (pre-fix VAL AUC was 0.6744 — lower post-fix means leak was contributing)\n')

    # Evaluate at thr=0.83 cap=$200 on both VAL and OOS
    from stdev_ml_iter8_circuit import evaluate_with_circuit, check_relaxed
    for label, p, y, ts, mask, days in [('VAL', pv, yv_L, val_ts_L, cv_mask, val_days),
                                         ('OOS', po, yo_L, oos_ts_L, co_mask, oos_days)]:
        fired = (p >= THR) & mask
        stats = evaluate_with_circuit(p, y, ts, fired, TP, SL, days, CAP)
        if stats is None:
            print(f'\n  [{label}] no fires at thr={THR}')
            continue
        passed, fails, pct_pos = check_relaxed(stats)
        print(f'\n  [{label}] thr={THR} cap=${CAP}: {stats["per_day"]:.2f}/d, WR={stats["wr"]*100:.1f}%, avg=${stats["avg"]:.0f}, PnL=${stats["pnl"]:+,.0f}, mos+={pct_pos*100:.0f}%')
        print(f'    passed={passed}, fails={fails}')
        for yy in stats['yearly']:
            ok_pnl = yy['pnl'] > 0; ok_wr = yy['wr'] > 0.5; ok_dd = yy['dd'] >= -900
            flag = '✅' if (ok_pnl and ok_wr and ok_dd) else '❌'
            print(f'    {yy["year"]}: n={yy["n"]} WR={yy["wr"]*100:.1f}% PnL=${yy["pnl"]:+,.0f} DD=${yy["dd"]:+,.0f} {flag}')

    # Also do a threshold sweep on OOS with rate-cap relaxed (user said >4/d is OK)
    print(f'\n  ===== OOS threshold sweep (rate cap relaxed; DD ≤ $900 firm) =====')
    print(f'  {"thr":>5s} {"cap":>5s} {"n":>5s} {"/d":>5s} {"WR%":>5s} {"avg":>7s} {"PnL":>10s} {"yrDD":>10s}')
    sweep_results = []
    for thr in [0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81, 0.79, 0.77, 0.75]:
        for cap in [100, 150, 200, 300]:
            fired = (po >= thr) & co_mask
            if not fired.any(): continue
            stats = evaluate_with_circuit(po, yo_L, oos_ts_L, fired, TP, SL, oos_days, cap)
            if stats is None: continue
            yr_dd = stats['yearly'][0]['dd'] if stats['yearly'] else 0
            yr_pnl = stats['yearly'][0]['pnl'] if stats['yearly'] else 0
            yr_wr = stats['yearly'][0]['wr'] if stats['yearly'] else 0
            print(f'  {thr:.2f} ${cap:>3d} {stats["n"]:>5d} {stats["per_day"]:>4.2f} {stats["wr"]*100:>4.1f}% ${stats["avg"]:>+5.0f} ${stats["pnl"]:>+8.0f} ${yr_dd:>+8.0f}')
            sweep_results.append({'thr': thr, 'cap': cap, **stats, 'yr_dd': yr_dd, 'yr_pnl': yr_pnl, 'yr_wr': yr_wr})

    # Save artifacts
    joblib.dump({
        'hgb': hgb, 'features': top60, 'tp': TP, 'sl': SL, 'threshold': THR, 'daily_dd_cap': CAP,
        'val_auc': float(auc_v), 'oos_auc': float(auc_o),
        'cluster_features': cluster_features,
        'scaler': best_K['sc'] if best_K else None,
        'pca': best_K['pca'] if best_K else None,
        'kmeans': best_K['km'] if best_K else None,
        'good_clusters': best_K['good'] if best_K else None,
        'top10_F': fa.head(10).to_dict('records'),
    }, OUT / 'iter10_causal_model.pkl')

    with open(OUT / 'iter10_causal_summary.json', 'w') as f:
        json.dump({
            'tp': TP, 'sl': SL, 'threshold': THR, 'daily_dd_cap': CAP,
            'val_auc': float(auc_v), 'oos_auc': float(auc_o),
            'top10_F': fa.head(10).to_dict('records'),
            'oos_sweep': [{k: v for k, v in r.items() if k != 'monthly'} for r in sweep_results],
        }, f, indent=2, default=str)

    print(f'\n  Saved: {OUT}/iter10_causal_model.pkl, iter10_causal_summary.json')
    print(f'\nWall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
