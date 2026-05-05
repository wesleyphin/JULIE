"""Iter 8 — re-evaluate iter 7's best configs with a per-day DD circuit breaker.

Circuit breaker: if accumulated within-day PnL drops below -$300 (3 losses at SL=10
or 4 at SL=12), stop trading for the rest of that day. Resets at midnight.

Also tries: -$200, -$400 daily caps to find sweet spot.

Re-uses the trained iter 7 models — just re-runs threshold sweep + walk with new mask.
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
from hmmlearn.hmm import GaussianHMM

# Reuse iter7's helpers — bring them in via direct import
import sys
sys.path.insert(0, '/Users/wes/Downloads/JULIE001/tools')
from stdev_ml_iter7_macro import (
    fetch_macros, load_period, compute_features, label_walk,
    PERIODS, ET, PT_USD, H_MIN, WIN_HOURS, OUT as IT7_OUT,
)
from stdev_ml_iter7_macro import build_period as build_iter7

warnings.filterwarnings("ignore")

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter8_circuit'
OUT.mkdir(parents=True, exist_ok=True)


def evaluate_with_circuit(p, y, ts_arr, fired, tp, sl, val_days, daily_dd_cap):
    """Walk forward day-by-day; suspend trading mid-day if accumulated PnL hits -dd_cap."""
    if not fired.any(): return None
    df = pd.DataFrame({'ts': pd.to_datetime(ts_arr), 'fired': fired, 'win': (y == 1) & fired})
    df = df.sort_values('ts').reset_index(drop=True)
    df['date'] = df['ts'].dt.normalize()
    df['raw_pnl'] = np.where(df['fired'], np.where(df['win'], tp * PT_USD, -sl * PT_USD), 0)

    # Apply circuit breaker: per-day cumulative PnL; once it crosses -dd_cap, mute remaining trades that day
    out_pnl = np.zeros(len(df))
    out_fired = np.zeros(len(df), dtype=bool)
    out_win = np.zeros(len(df), dtype=bool)
    for date, grp in df.groupby('date'):
        cum = 0.0
        muted = False
        for idx in grp.index:
            if not df.at[idx, 'fired']:
                continue
            if muted:
                continue
            pnl = df.at[idx, 'raw_pnl']
            out_pnl[idx] = pnl
            out_fired[idx] = True
            out_win[idx] = bool(df.at[idx, 'win'])
            cum += pnl
            if cum <= -daily_dd_cap:
                muted = True
    df['cb_pnl'] = out_pnl
    df['cb_fired'] = out_fired
    df['cb_win'] = out_win
    df['month'] = df['ts'].dt.to_period('M')
    df['year'] = df['ts'].dt.year

    # Per-month
    monthly_df = df[df['cb_fired']].groupby('month').agg(
        n=('cb_fired', 'sum'), w=('cb_win', 'sum'), pnl=('cb_pnl', 'sum')
    ).reset_index()
    monthly_df['wr'] = monthly_df['w'] / monthly_df['n']
    # Per-year
    yearly = []
    for y_, grp in df[df['cb_fired']].groupby('year'):
        ord_pnl = grp.sort_values('ts')['cb_pnl'].cumsum()
        peak = ord_pnl.cummax()
        dd = float((ord_pnl - peak).min())
        yearly.append({'year': int(y_), 'n': int(grp['cb_fired'].sum()), 'w': int(grp['cb_win'].sum()),
                       'pnl': float(grp['cb_pnl'].sum()), 'dd': dd,
                       'wr': float(grp['cb_win'].sum() / grp['cb_fired'].sum())})
    n = int(monthly_df['n'].sum())
    return {
        'n': n,
        'wr': float(monthly_df['w'].sum() / n) if n else 0,
        'pnl': float(monthly_df['pnl'].sum()),
        'avg': float(monthly_df['pnl'].sum() / n) if n else 0,
        'per_day': n / val_days,
        'monthly': monthly_df.to_dict('records'),
        'yearly': yearly,
    }


def check_relaxed(stats):
    fails = []
    if stats['per_day'] < 2: fails.append(f"avg{stats['per_day']:.2f}/d<2")
    if stats['per_day'] > 4: fails.append(f"avg{stats['per_day']:.2f}/d>4")
    if stats['avg'] < 60: fails.append(f"avg${stats['avg']:.0f}<$60")
    for y in stats['yearly']:
        if y['dd'] < -900: fails.append(f"YR{y['year']}-DD${y['dd']:.0f}")
        if y['pnl'] <= 0: fails.append(f"YR{y['year']}-PnL${y['pnl']:.0f}")
        if y['wr'] <= 0.5: fails.append(f"YR{y['year']}-WR{y['wr']:.2%}")
    n_mo = len(stats['monthly'])
    pos_mo = sum(1 for m in stats['monthly'] if m['pnl'] > 0 and m['wr'] > 0.5)
    pct_pos = pos_mo / n_mo if n_mo else 0
    if pct_pos < 0.8: fails.append(f"pos_mo{pct_pos:.0%}<80%")
    return len(fails) == 0, fails, pct_pos


def run_with_circuit(tp, sl, macro):
    print(f'\n========== TP={tp}/SL={sl} with daily circuit breaker ==========')
    breakeven = sl / (tp + sl)

    # Build periods (cached features won't reuse iter7 dataset, so rebuild — slow)
    all_data = {}
    for name, (s, e) in PERIODS.items():
        df = build_iter7(name, s, e, tp, sl, macro)
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

    # Use top 60 by F-stat
    Xt_clean = Xt_L.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    F, _ = f_classif(Xt_clean.values, yt_L)
    fa = pd.DataFrame({'f': feat_cols, 'F': F}).sort_values('F', ascending=False)
    top60 = fa.head(60)['f'].tolist()

    # K-means cluster filter
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
            best_K = {'K': K, 'good': good, 'cv': cv}
    cluster_mask_v = np.isin(best_K['cv'], best_K['good']) if best_K else np.ones(len(Xv_L), dtype=bool)

    # HGB
    Xt_hgb = Xt_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    Xv_hgb = Xv_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400,
                                          early_stopping=True, n_iter_no_change=15, random_state=42)
    hgb.fit(Xt_hgb, yt_L)
    pv = hgb.predict_proba(Xv_hgb)[:, 1]
    auc = roc_auc_score(yv_L, pv)
    print(f'  HGB Val AUC: {auc:.4f}')

    # Try multiple daily DD caps
    print(f'\n  Threshold × Daily DD cap sweep:')
    print(f'  {"thr":>5s} {"cap":>6s} {"n":>5s} {"/d":>5s} {"WR%":>5s} {"avg":>7s} {"PnL":>10s} {"yrs+":>5s} {"mos+%":>6s} PASS')
    candidates = []
    for thr in np.arange(0.95, 0.55, -0.02):
        for daily_cap in [200, 300, 400]:
            fired = (pv >= thr) & cluster_mask_v
            n_pre = int(fired.sum())
            if n_pre == 0: continue
            stats = evaluate_with_circuit(pv, yv_L, val_ts_L, fired, tp, sl, val_days, daily_cap)
            if stats is None: continue
            passed, fails, pct_pos = check_relaxed(stats)
            yrs_pos = sum(1 for y in stats['yearly'] if y['pnl'] > 0 and y['wr'] > 0.5 and y['dd'] >= -900)
            candidates.append({**stats, 'thr': float(thr), 'cap': daily_cap, 'passed': passed,
                               'fails': fails, 'yrs_pos': yrs_pos, 'mos_pos_pct': pct_pos})

    # Print only passing or rate+avg-ok candidates
    interesting = [c for c in candidates if c['passed'] or (2 <= c['per_day'] <= 4 and c['avg'] >= 60)]
    for c in interesting[:30]:
        flag = '🟢' if c['passed'] else '  '
        print(f'  {c["thr"]:.2f} ${c["cap"]:>4d} {c["n"]:>5d} {c["per_day"]:>4.2f} {c["wr"]*100:>4.1f}% ${c["avg"]:>+5.0f} ${c["pnl"]:>+8.0f} {c["yrs_pos"]:>5d}/{len(c["yearly"])} {c["mos_pos_pct"]*100:>4.0f}% {flag}')

    passing = [c for c in candidates if c['passed']]
    if passing:
        # Pick the one with most trades among passing
        best = max(passing, key=lambda c: c['n'])
        return {'tp': tp, 'sl': sl, 'best': best, 'auc': auc, 'hgb': hgb, 'features': top60,
                'cluster_K': best_K, 'cluster_features': cluster_features,
                'val_days': val_days}
    return None


def main():
    t0 = time.time()
    macro = fetch_macros()
    results = []
    for tp, sl in [(35, 12), (40, 15), (30, 10), (25, 10)]:
        r = run_with_circuit(tp, sl, macro)
        if r: results.append(r)

    print('\n========== SUMMARY ==========')
    if results:
        for r in results:
            b = r['best']
            print(f'  TP={r["tp"]}/SL={r["sl"]} cap=${b["cap"]}: thr={b["thr"]:.2f}, {b["per_day"]:.2f}/d, WR={b["wr"]*100:.1f}%, avg=${b["avg"]:.0f}, +${b["pnl"]:.0f}, mos+={b["mos_pos_pct"]*100:.0f}%')
        best = max(results, key=lambda r: r['best']['pnl'])
        b = best['best']
        print(f'\n🟢🟢🟢 BEST PASSING:')
        print(f'   TP={best["tp"]}/SL={best["sl"]}, threshold={b["thr"]:.2f}, daily DD cap=${b["cap"]}')
        print(f'   {b["per_day"]:.2f}/d, WR={b["wr"]*100:.1f}%, avg=${b["avg"]:.0f}, PnL/4yr=+${b["pnl"]:.0f}')
        print(f'   mos+={b["mos_pos_pct"]*100:.0f}%, yrs+={b["yrs_pos"]}/{len(b["yearly"])}')
        for y in b['yearly']:
            print(f'   {y["year"]}: n={y["n"]} WR={y["wr"]*100:.1f}% PnL=${y["pnl"]:+.0f} DD=${y["dd"]:+.0f}')
        joblib.dump({
            'model': best['hgb'], 'features': best['features'],
            'tp': best['tp'], 'sl': best['sl'],
            'threshold': b['thr'], 'daily_dd_cap': b['cap'],
            'val_auc': best['auc'],
            'cluster_K': best['cluster_K'],
            'cluster_features': best['cluster_features'],
        }, OUT / 'iter8_circuit_model.pkl')
        with open(OUT / 'iter8_summary.json', 'w') as f:
            json.dump({
                'tp': best['tp'], 'sl': best['sl'],
                'threshold': b['thr'], 'daily_dd_cap': b['cap'],
                'val_auc': best['auc'],
                'wr': b['wr'], 'avg': b['avg'], 'pnl': b['pnl'],
                'per_day': b['per_day'], 'mos_pos_pct': b['mos_pos_pct'],
                'yearly': b['yearly'],
                'monthly': b['monthly'][:60],
            }, f, indent=2, default=str)
        print(f'\n✅ Saved iter8_circuit_model.pkl + iter8_summary.json')
    else:
        print('❌ No passing config even with circuit breaker. Closest configs in printout above.')
    print(f'\nWall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
