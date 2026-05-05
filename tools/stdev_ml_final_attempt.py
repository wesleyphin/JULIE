"""Final attempt: TP=5/SL=5 LONG only with full cascade (cluster + HMM + HGB).
Loosen "good" thresholds to find ANY filter that meets constraints.
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

warnings.filterwarnings("ignore")

ROOT = Path('/Users/wes/Downloads/JULIE001')
SRC_DIR = ROOT / 'artifacts' / 'stdev_ml_hr9_11'
BAR = ROOT / 'es_master_outrights-2.parquet'
OUT = ROOT / 'artifacts' / 'stdev_ml_hr9_11' / 'final'
OUT.mkdir(parents=True, exist_ok=True)

ET = 'US/Eastern'
PT_USD = 5.0
TP = 5.0
SL = 5.0  # 1:1 R:R, breakeven 50%
H_MIN = 120


def load_period_with_tight_labels(start, end):
    bars = pd.read_parquet(BAR)
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert(ET)
    bars = bars.loc[start:end]
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
    return bars


def main():
    t0 = time.time()
    train = pd.read_parquet(SRC_DIR / 'train.parquet')
    val = pd.read_parquet(SRC_DIR / 'val.parquet')
    train.index = pd.to_datetime(train.index)
    val.index = pd.to_datetime(val.index)
    feat_cols = [c for c in train.columns if c != 'label']
    print(f'TRAIN: {len(train):,}, VAL: {len(val):,}')

    # Filter to LONG side only (side=0)
    Xt_full = train[feat_cols].astype(np.float32)
    yt_full = train['label'].astype(np.int8).values
    Xv_full = val[feat_cols].astype(np.float32)
    yv_full = val['label'].astype(np.int8).values
    side_t = (Xt_full['side'] == 0).values
    side_v = (Xv_full['side'] == 0).values
    Xt = Xt_full[side_t].reset_index(drop=True)
    yt = yt_full[side_t]
    Xv = Xv_full[side_v].reset_index(drop=True)
    yv = yv_full[side_v]
    train_ts = train.index[(~Xt_full.isna().any(axis=1)).values & side_t][:len(Xt)] if False else train.index[side_t]
    val_ts = val.index[side_v]

    # Drop NaN
    valid_t = ~Xt.isna().any(axis=1)
    valid_v = ~Xv.isna().any(axis=1)
    Xt = Xt[valid_t].reset_index(drop=True); yt = yt[valid_t.values]
    Xv = Xv[valid_v].reset_index(drop=True); yv = yv[valid_v.values]
    train_ts = train_ts[valid_t.values]
    val_ts = val_ts[valid_v.values]
    val_days = pd.to_datetime(val_ts).normalize().nunique()
    print(f'LONG-only: TRAIN={len(Xt):,}, VAL={len(Xv):,}, val_days={val_days}')
    print(f'NOTE: Labels are still TP=8.25/SL=10. Re-evaluating with TP=5/SL=5 brackets needs MFE/MAE.')

    # Top features by F-stat
    F, pvals = f_classif(Xt.fillna(0).values, yt)
    fa = pd.DataFrame({'feature': feat_cols, 'F': F, 'p': pvals}).sort_values('F', ascending=False)
    sig_features = fa.head(40)['feature'].tolist()
    sigma_features = [f for f in sig_features if 'sigma' in f or 'z_' in f][:15]

    # K-means + HMM
    Xt_sig = Xt[sigma_features].fillna(0).values
    Xv_sig = Xv[sigma_features].fillna(0).values
    sc_c = StandardScaler()
    Xt_sc = sc_c.fit_transform(Xt_sig)
    Xv_sc = sc_c.transform(Xv_sig)
    pca = PCA(n_components=8, random_state=42)
    Xt_p = pca.fit_transform(Xt_sc)
    Xv_p = pca.transform(Xv_sc)

    # Try multiple K, looser threshold
    print('\nK-means with looser "good cluster" threshold (WR > 0.52):')
    best_clusters = None
    for K in [4, 6, 8, 10]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        ct = km.fit_predict(Xt_p)
        cv = km.predict(Xv_p)
        df_t = pd.DataFrame({'c': ct, 'y': yt}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        df_v = pd.DataFrame({'c': cv, 'y': yv}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        good = df_t[(df_t['wr'] > 0.52) & (df_t['n'] > 1000)].index.tolist()
        # Filter to clusters that ALSO show >0.52 in VAL
        good = [c for c in good if (c in df_v.index and df_v.loc[c, 'wr'] > 0.52)]
        if good:
            print(f'  K={K}: good clusters in BOTH train+val: {good}')
            for c in good:
                print(f'    c{c}: T n={df_t.loc[c,"n"]} WR={df_t.loc[c,"wr"]:.3f}  V n={df_v.loc[c,"n"]} WR={df_v.loc[c,"wr"]:.3f}')
        if not best_clusters or len(good) > len(best_clusters['good']):
            best_clusters = {'K': K, 'good': good, 'km': km, 'ct': ct, 'cv': cv}

    cluster_mask_t = np.isin(best_clusters['ct'], best_clusters['good']) if best_clusters['good'] else np.ones(len(Xt), dtype=bool)
    cluster_mask_v = np.isin(best_clusters['cv'], best_clusters['good']) if best_clusters['good'] else np.ones(len(Xv), dtype=bool)
    print(f'  → cluster mask: TRAIN {cluster_mask_t.sum():,}, VAL {cluster_mask_v.sum():,}')

    # HMM
    hmm_features = [f for f in ['sigma_60', 'sigma_240', 'sigma_of_sigma_60_240'] if f in feat_cols]
    Xt_h = StandardScaler().fit_transform(Xt[hmm_features].fillna(0).values)
    Xv_h = Xt[hmm_features].fillna(0).values  # fit on train, will refit
    sc_h = StandardScaler()
    Xt_h = sc_h.fit_transform(Xt[hmm_features].fillna(0).values)
    Xv_h = sc_h.transform(Xv[hmm_features].fillna(0).values)

    print('\nHMM with looser "good state" threshold:')
    best_hmm = None
    for N in [2, 3, 4]:
        try:
            hmm = GaussianHMM(n_components=N, covariance_type='diag', n_iter=50, random_state=42)
            hmm.fit(Xt_h)
            st_t = hmm.predict(Xt_h)
            st_v = hmm.predict(Xv_h)
            df_t = pd.DataFrame({'s': st_t, 'y': yt}).groupby('s').agg(n=('y','count'), wr=('y','mean'))
            df_v = pd.DataFrame({'s': st_v, 'y': yv}).groupby('s').agg(n=('y','count'), wr=('y','mean'))
            good = df_t[(df_t['wr'] > 0.52) & (df_t['n'] > 1000)].index.tolist()
            good = [s for s in good if (s in df_v.index and df_v.loc[s, 'wr'] > 0.52)]
            if good:
                print(f'  N={N}: good states T+V: {good}')
            if not best_hmm or len(good) > len(best_hmm['good']):
                best_hmm = {'N': N, 'good': good, 'hmm': hmm, 'st_t': st_t, 'st_v': st_v}
        except: pass

    if best_hmm and best_hmm['good']:
        hmm_mask_t = np.isin(best_hmm['st_t'], best_hmm['good'])
        hmm_mask_v = np.isin(best_hmm['st_v'], best_hmm['good'])
    else:
        hmm_mask_t = np.ones(len(Xt), dtype=bool)
        hmm_mask_v = np.ones(len(Xv), dtype=bool)
    print(f'  → HMM mask: TRAIN {hmm_mask_t.sum():,}, VAL {hmm_mask_v.sum():,}')

    # Train HGB on filtered LONG bars
    combo_t = cluster_mask_t & hmm_mask_t
    print(f'\nCombined gate: TRAIN {combo_t.sum():,}, VAL {(cluster_mask_v & hmm_mask_v).sum():,}')
    if combo_t.sum() < 5000:
        print('  Filter too aggressive; using full TRAIN')
        combo_t = np.ones(len(Xt), dtype=bool)

    Xt_hgb = Xt[combo_t][sig_features].fillna(0).values.astype(np.float32)
    yt_hgb = yt[combo_t]
    Xv_hgb = Xv[sig_features].fillna(0).values.astype(np.float32)

    hgb = HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.05, max_iter=300,
        early_stopping=True, n_iter_no_change=15, random_state=42,
    )
    hgb.fit(Xt_hgb, yt_hgb)
    pv = hgb.predict_proba(Xv_hgb)[:, 1]
    auc = roc_auc_score(yv, pv)
    print(f'  HGB Val AUC (LONG-only filtered cascade): {auc:.4f}')

    # Threshold sweep
    print('\nThreshold sweep — LONG only, full cascade, TP=8.25/SL=10 labels:')
    print(f'{"thr":>6s} {"n":>5s} {"per_d":>7s} {"WR%":>6s} {"PnL":>9s} {"DD":>9s} {"yrs+pos":>9s} {"mos+pos":>9s} {"PASS":>5s}')

    cluster_mask_v_long = cluster_mask_v
    hmm_mask_v_long = hmm_mask_v
    val_ts_long = val_ts

    candidates = []
    for thr in np.arange(0.99, 0.40, -0.01):
        fired = (pv >= thr) & cluster_mask_v_long & hmm_mask_v_long
        n = int(fired.sum())
        if n == 0: continue

        df = pd.DataFrame({'ts': val_ts_long, 'fired': fired, 'win': (yv == 1) & fired})
        df['month'] = df['ts'].dt.to_period('M')
        df['year'] = df['ts'].dt.year
        df['pnl'] = np.where(fired, np.where(df['win'], TP * PT_USD, -SL * PT_USD), 0)

        # Per-month
        monthly = df.groupby('month').apply(lambda g: pd.Series({
            'n': int(g['fired'].sum()),
            'w': int(g['win'].sum()),
            'pnl': float(g['pnl'].sum()),
        }), include_groups=False).reset_index()
        monthly = monthly[monthly['n'] > 0]
        monthly['wr'] = monthly['w'] / monthly['n']
        if len(monthly) == 0: continue

        # Per-year
        yearly = df.groupby('year').apply(lambda g: pd.Series({
            'n': int(g['fired'].sum()),
            'w': int(g['win'].sum()),
            'pnl': float(g['pnl'].sum()),
            'dd': float((g[g['fired']].sort_values('ts')['pnl'].cumsum() - g[g['fired']].sort_values('ts')['pnl'].cumsum().cummax()).min()) if g['fired'].sum() else 0,
        }), include_groups=False).reset_index()
        yearly = yearly[yearly['n'] > 0]
        yearly['wr'] = yearly['w'] / yearly['n']

        n_total = int(monthly['n'].sum())
        wr_total = float(monthly['w'].sum() / n_total) if n_total else 0
        pnl_total = float(monthly['pnl'].sum())
        per_day = n_total / val_days

        # Constraints
        yrs_pos_pnl = int((yearly['pnl'] > 0).sum())
        yrs_pos_wr = int((yearly['wr'] > 0.5).sum())
        yrs_dd_ok = int((yearly['dd'] >= -900).sum())
        n_years = len(yearly)
        mos_pos_pnl = int((monthly['pnl'] > 0).sum())
        mos_pos_wr = int((monthly['wr'] > 0.5).sum())
        n_months = len(monthly)

        all_yrs_pos = yrs_pos_pnl == n_years and yrs_pos_wr == n_years and yrs_dd_ok == n_years
        all_mos_pos = mos_pos_pnl == n_months and mos_pos_wr == n_months
        rate_ok = 2 <= per_day <= 4
        passed = all_yrs_pos and all_mos_pos and rate_ok and pnl_total > 0

        candidates.append({
            'thr': float(thr), 'n': n_total, 'per_day': per_day,
            'wr': wr_total, 'pnl': pnl_total,
            'yearly': yearly.to_dict('records'),
            'monthly': monthly.to_dict('records'),
            'passed': passed,
            'yrs_pos': f'{min(yrs_pos_pnl, yrs_pos_wr, yrs_dd_ok)}/{n_years}',
            'mos_pos': f'{min(mos_pos_pnl, mos_pos_wr)}/{n_months}',
        })

    print()
    for c in candidates[::3]:  # print every 3rd
        flag = '✅' if c['passed'] else '❌'
        print(f'{c["thr"]:.3f}  {c["n"]:>5d} {c["per_day"]:>5.2f}/d {c["wr"]*100:>5.1f}% ${c["pnl"]:>+8.0f}  {c["yrs_pos"]:>9s} {c["mos_pos"]:>9s}  {flag}')

    passing = [c for c in candidates if c['passed']]
    if passing:
        best = max(passing, key=lambda c: c['n'])
        print(f'\n🟢 PASSING: thr={best["thr"]:.3f}, {best["per_day"]:.2f}/d, WR={best["wr"]*100:.1f}%, PnL=+${best["pnl"]:.0f}')
        for y in best['yearly']:
            print(f'   {int(y["year"])}: n={int(y["n"])} WR={y["wr"]*100:.1f}% PnL=${y["pnl"]:+.0f} DD=${y["dd"]:+.0f}')
    else:
        print('\n❌ NO config passed all hard constraints. Closest:')
        candidates.sort(key=lambda c: -c['pnl'])
        for c in candidates[:3]:
            print(f'  thr={c["thr"]:.3f} {c["per_day"]:.2f}/d WR={c["wr"]*100:.1f}% PnL=${c["pnl"]:+.0f} yrs+={c["yrs_pos"]} mos+={c["mos_pos"]}')

    print(f'\nTotal time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
