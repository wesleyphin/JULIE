"""σ-ML cascade: MANOVA + K-means + HMM + HGB to meet hard constraints.

Hard constraints:
  - Annual DD ≤ $900
  - Positive WR every year AND every month
  - Positive PnL (overall + each year)
  - 2-4 trades/day average

Approach:
  Phase A: ANOVA F-stat per feature → keep features w/ p < 0.001 separating W/L
  Phase B: K-means cluster bars in PCA(σ-features); compute per-cluster WR;
           identify "good" clusters (WR ≥ 60%)
  Phase C: HMM(2-3 states) on (σ_60, σ_240, σ-of-σ) → identify safe regimes
  Phase D: HGB on filtered bars; threshold sweep
  Phase E: cascade gate: bar must be in good cluster + good HMM state +
           HGB confidence ≥ threshold
  Phase F: iterate if constraints fail

Outputs everything to artifacts/stdev_ml_hr9_11/cascade/.
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
OUT = ROOT / 'artifacts' / 'stdev_ml_hr9_11' / 'cascade'
OUT.mkdir(parents=True, exist_ok=True)

TP = 8.25
SL = 10.0
PT_USD = 5.0  # MES dollars per point


def load_set(name):
    df = pd.read_parquet(SRC_DIR / f'{name}.parquet')
    df.index = pd.to_datetime(df.index)
    return df


def per_month_eval(timestamps, fired, wins, label):
    """Per-month + per-year aggregation."""
    df = pd.DataFrame({'ts': timestamps, 'fired': fired, 'win': wins})
    df['month'] = df['ts'].dt.to_period('M')
    df['year'] = df['ts'].dt.year
    df['pnl'] = np.where(
        df['fired'],
        np.where(df['win'], TP * PT_USD, -SL * PT_USD),
        0.0,
    )
    monthly = []
    for m, grp in df.groupby('month'):
        n = int(grp['fired'].sum())
        if n == 0: continue
        w = int((grp['fired'] & grp['win']).sum())
        l = n - w
        pnl = float(grp['pnl'].sum())
        monthly.append({'month': str(m), 'n': n, 'w': w, 'l': l,
                        'wr': w/n, 'pnl': pnl, 'avg': pnl/n})
    yearly = []
    for y, grp in df.groupby('year'):
        ord_pnl = grp[grp['fired']].sort_values('ts')['pnl'].cumsum()
        if len(ord_pnl) == 0: continue
        peak = ord_pnl.cummax()
        dd = float((ord_pnl - peak).min())
        n = int(grp['fired'].sum())
        w = int((grp['fired'] & grp['win']).sum())
        pnl = float(grp['pnl'].sum())
        yearly.append({'year': int(y), 'n': n, 'wr': w/n if n else 0,
                       'pnl': pnl, 'dd': dd})
    # Total
    full_pnl = df[df['fired']].sort_values('ts')['pnl'].cumsum()
    if len(full_pnl):
        peak = full_pnl.cummax()
        full_dd = float((full_pnl - peak).min())
    else:
        full_dd = 0
    return monthly, yearly, full_dd


def check_constraints(monthly, yearly, full_dd, n_total, n_days):
    """Return (passed, reasons_failed)."""
    fails = []
    # Per-year DD ≤ $900
    for y in yearly:
        if y['dd'] < -900: fails.append(f"YR {y['year']} DD ${y['dd']:.0f} < -$900")
    # All years positive PnL
    for y in yearly:
        if y['pnl'] <= 0: fails.append(f"YR {y['year']} PnL ${y['pnl']:.0f} <= 0")
    # All years positive WR (>50%)
    for y in yearly:
        if y['wr'] <= 0.50: fails.append(f"YR {y['year']} WR {y['wr']:.2%} <= 50%")
    # All months positive PnL
    neg_months = [m for m in monthly if m['pnl'] <= 0]
    if neg_months:
        fails.append(f"{len(neg_months)} months with PnL <= 0 (worst: {min(neg_months, key=lambda x: x['pnl'])['month']} ${min(m['pnl'] for m in neg_months):.0f})")
    # All months positive WR (>50%)
    low_wr_months = [m for m in monthly if m['wr'] <= 0.5]
    if low_wr_months:
        fails.append(f"{len(low_wr_months)} months with WR <= 50%")
    # 2-4 trades per day average
    avg_per_day = n_total / max(n_days, 1)
    if avg_per_day < 2:
        fails.append(f"avg trades/day {avg_per_day:.2f} < 2")
    if avg_per_day > 4:
        fails.append(f"avg trades/day {avg_per_day:.2f} > 4")
    return len(fails) == 0, fails


def main():
    t0 = time.time()
    train = load_set('train')
    val = load_set('val')
    feat_cols = [c for c in train.columns if c != 'label']
    print(f'TRAIN: {len(train):,}, VAL: {len(val):,}, features: {len(feat_cols)}')

    Xt = train[feat_cols].astype(np.float32)
    yt = train['label'].astype(np.int8).values
    Xv = val[feat_cols].astype(np.float32)
    yv = val['label'].astype(np.int8).values

    # Drop NaN rows (early bars)
    valid_t = ~Xt.isna().any(axis=1)
    valid_v = ~Xv.isna().any(axis=1)
    Xt = Xt[valid_t].reset_index(drop=True)
    yt = yt[valid_t.values]
    Xv = Xv[valid_v].reset_index(drop=True)
    yv = yv[valid_v.values]
    val_ts = val.index[valid_v.values]
    train_ts = train.index[valid_t.values]
    print(f'After NaN filter: TRAIN={len(Xt):,}, VAL={len(Xv):,}')

    # ============ PHASE A: ANOVA F-stat per feature ============
    print('\n=== PHASE A: ANOVA per-feature F-stat (W vs L) ===')
    F, pvals = f_classif(Xt.fillna(0).values, yt)
    feature_anova = pd.DataFrame({
        'feature': feat_cols, 'F': F, 'p': pvals
    }).sort_values('F', ascending=False)
    print(f'Top 20 features by F-stat:')
    print(feature_anova.head(20).to_string(index=False, float_format='%.4f'))

    # Keep top-K features (significant + meaningful F)
    SIG_K = 30
    sig_features = feature_anova.head(SIG_K)['feature'].tolist()
    print(f'\n→ Keeping top {SIG_K} features (those with strongest separation)')
    feature_anova.to_csv(OUT / 'feature_anova.csv', index=False)

    # ============ PHASE B: K-means clustering ============
    print('\n=== PHASE B: K-means clustering on σ-feature subspace ===')
    sigma_features = [f for f in sig_features if 'sigma' in f or 'z_' in f or 'efficiency' in f or 'trend_r2' in f][:15]
    print(f'  Using {len(sigma_features)} σ-related features for clustering')
    Xt_sig = Xt[sigma_features].fillna(0).values
    Xv_sig = Xv[sigma_features].fillna(0).values

    scaler = StandardScaler()
    Xt_s = scaler.fit_transform(Xt_sig)
    Xv_s = scaler.transform(Xv_sig)

    pca = PCA(n_components=8, random_state=42)
    Xt_p = pca.fit_transform(Xt_s)
    Xv_p = pca.transform(Xv_s)
    print(f'  PCA 8-d explained var: {pca.explained_variance_ratio_.sum():.3f}')

    best_cluster_config = None
    for K in [6, 8, 10, 12]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        ct = km.fit_predict(Xt_p)
        cv = km.predict(Xv_p)
        # Per-cluster WR on TRAIN
        df_t = pd.DataFrame({'cluster': ct, 'win': yt}).groupby('cluster').agg(
            n=('win', 'count'), wr=('win', 'mean')
        )
        # Apply same to VAL
        df_v = pd.DataFrame({'cluster': cv, 'win': yv}).groupby('cluster').agg(
            n=('win', 'count'), wr=('win', 'mean')
        )
        # "Good" clusters = where TRAIN WR > 0.55 AND has meaningful sample (n > 1000)
        good_train = df_t[(df_t['wr'] > 0.55) & (df_t['n'] > 1000)].index.tolist()
        # Check VAL behavior of those good clusters
        print(f'  K={K}: TRAIN good clusters = {good_train}')
        for c in good_train:
            t_n = df_t.loc[c, 'n']; t_wr = df_t.loc[c, 'wr']
            v_n = df_v.loc[c, 'n'] if c in df_v.index else 0
            v_wr = df_v.loc[c, 'wr'] if c in df_v.index else 0
            print(f'    cluster {c}: TRAIN n={t_n:>6d} WR={t_wr:.3f}  |  VAL n={v_n:>6d} WR={v_wr:.3f}')

        if not best_cluster_config or len(good_train) > 0:
            best_cluster_config = {'K': K, 'good_clusters': good_train, 'km': km, 'ct': ct, 'cv': cv}

    K = best_cluster_config['K']
    good_clusters = set(best_cluster_config['good_clusters'])
    ct_best = best_cluster_config['ct']
    cv_best = best_cluster_config['cv']
    print(f'  → Selected K={K} with {len(good_clusters)} good cluster(s): {sorted(good_clusters)}')

    cluster_mask_t = np.isin(ct_best, list(good_clusters))
    cluster_mask_v = np.isin(cv_best, list(good_clusters))

    # ============ PHASE C: HMM regime detection ============
    print('\n=== PHASE C: HMM Gaussian on σ-time-series ===')
    # Use sigma_60, sigma_240, sigma_of_sigma_60_240, sigma_ratio_60_240
    hmm_features = [f for f in ['sigma_60', 'sigma_240', 'sigma_of_sigma_60_240', 'sigma_ratio_60_240', 'sigma_ratio_15_240'] if f in feat_cols]
    print(f'  HMM features: {hmm_features}')

    # Train HMM on TRAIN time-series ordered chronologically
    Xt_hmm_full = Xt[hmm_features].fillna(0).values
    Xv_hmm_full = Xv[hmm_features].fillna(0).values
    sc_h = StandardScaler()
    Xt_h = sc_h.fit_transform(Xt_hmm_full)
    Xv_h = sc_h.transform(Xv_hmm_full)

    best_hmm_config = None
    for N_STATES in [2, 3]:
        try:
            hmm = GaussianHMM(n_components=N_STATES, covariance_type='diag', n_iter=50, random_state=42)
            hmm.fit(Xt_h)
            states_t = hmm.predict(Xt_h)
            states_v = hmm.predict(Xv_h)
            # Per-state WR
            df_h = pd.DataFrame({'state': states_t, 'win': yt}).groupby('state').agg(
                n=('win', 'count'), wr=('win', 'mean')
            )
            print(f'  HMM N={N_STATES}: TRAIN per-state:')
            for s in df_h.index:
                v_n = (states_v == s).sum()
                v_wr = yv[states_v == s].mean() if v_n else 0
                print(f'    state {s}: TRAIN n={df_h.loc[s, "n"]:>6d} WR={df_h.loc[s, "wr"]:.3f}  |  VAL n={v_n:>6d} WR={v_wr:.3f}')
            good_states = df_h[(df_h['wr'] > 0.55) & (df_h['n'] > 1000)].index.tolist()
            print(f'    good states: {good_states}')
            if not best_hmm_config or len(good_states) > 0:
                best_hmm_config = {'N': N_STATES, 'good_states': good_states, 'hmm': hmm, 'states_t': states_t, 'states_v': states_v}
        except Exception as e:
            print(f'  HMM N={N_STATES} failed: {e}')

    if best_hmm_config:
        hmm_n = best_hmm_config['N']
        good_states = set(best_hmm_config['good_states'])
        states_t = best_hmm_config['states_t']
        states_v = best_hmm_config['states_v']
        print(f'  → Selected HMM N={hmm_n} with good states {sorted(good_states)}')
        hmm_mask_t = np.isin(states_t, list(good_states)) if good_states else np.ones(len(states_t), dtype=bool)
        hmm_mask_v = np.isin(states_v, list(good_states)) if good_states else np.ones(len(states_v), dtype=bool)
    else:
        print('  HMM skipped — no good states found')
        hmm_mask_t = np.ones(len(Xt), dtype=bool)
        hmm_mask_v = np.ones(len(Xv), dtype=bool)

    # ============ PHASE D: HGB on filtered subset ============
    print('\n=== PHASE D: HGB on cluster + HMM filtered bars ===')
    combined_t = cluster_mask_t & hmm_mask_t
    combined_v = cluster_mask_v & hmm_mask_v
    print(f'  Filtered bars: TRAIN {combined_t.sum():,} (from {len(Xt):,}), VAL {combined_v.sum():,} (from {len(Xv):,})')
    if combined_t.sum() < 5000:
        print('  ⚠️ Filter too aggressive; reverting to full TRAIN for HGB fit')
        combined_t = np.ones(len(Xt), dtype=bool)

    Xt_filt = Xt.iloc[combined_t][sig_features].fillna(0).astype(np.float32)
    yt_filt = yt[combined_t]
    Xv_filt = Xv[sig_features].fillna(0).astype(np.float32)  # score ALL val to evaluate fully
    yv_filt = yv

    hgb = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.04, max_iter=300,
        early_stopping=True, n_iter_no_change=15,
        random_state=42, verbose=0,
    )
    hgb.fit(Xt_filt, yt_filt)
    pv = hgb.predict_proba(Xv_filt)[:, 1]
    auc_v = roc_auc_score(yv, pv)
    print(f'  HGB Val AUC (full): {auc_v:.4f}')

    # ============ PHASE E: cascade threshold search to meet constraints ============
    print('\n=== PHASE E: cascade threshold search ===')
    # Compute trading days in VAL window
    val_days = pd.to_datetime(val_ts).normalize().nunique()
    print(f'  Val trading days: {val_days}')
    target_n_min = val_days * 2  # 2/day min
    target_n_max = val_days * 4  # 4/day max

    # Cascade: bar fires only if (in good cluster) AND (in good HMM state) AND (HGB conf > thr)
    # Start with very high thr, loosen until constraints hit
    print(f'\n  Threshold sweep (all 3 cascade gates active):')
    print(f'  {"thr":>5s} {"n":>6s} {"trades/day":>10s} {"WR%":>6s} {"PnL":>10s} {"max_DD":>10s} {"PASS?":>6s}')

    candidates = []
    for thr in np.arange(0.95, 0.45, -0.02):
        fired = (pv >= thr) & cluster_mask_v & hmm_mask_v
        n = int(fired.sum())
        if n == 0: continue
        wins_arr = (yv == 1)
        monthly, yearly, full_dd = per_month_eval(val_ts, fired, wins_arr, f'thr={thr:.2f}')
        n_total = sum(m['n'] for m in monthly)
        if n_total == 0: continue
        wr_total = sum(m['w'] for m in monthly) / n_total
        pnl_total = sum(m['pnl'] for m in monthly)
        per_day = n_total / val_days
        passed, fails = check_constraints(monthly, yearly, full_dd, n_total, val_days)
        flag = '✅' if passed else '❌'
        print(f'  {thr:.3f} {n:>6d} {per_day:>9.2f}/d {wr_total*100:>5.1f}% ${pnl_total:>+8.0f} ${full_dd:>+9.0f}   {flag}')
        candidates.append({
            'thr': float(thr), 'n': n, 'per_day': per_day,
            'wr': wr_total, 'pnl': pnl_total, 'dd': full_dd,
            'passed': passed, 'fails': fails,
            'monthly': monthly, 'yearly': yearly,
        })

    # Find passing candidate
    passing = [c for c in candidates if c['passed']]
    if passing:
        # Among passing, prefer the one with most trades (to be at upper end of 2-4/day)
        best = max(passing, key=lambda c: c['n'])
        print(f'\n🟢 FOUND PASSING CONFIG: thr={best["thr"]:.3f}')
        print(f'   {best["per_day"]:.2f} trades/day, {best["wr"]*100:.1f}% WR, +${best["pnl"]:.0f}, max DD ${best["dd"]:.0f}')
        print(f'\nPer-year:')
        for y in best['yearly']:
            print(f'   {y["year"]}: n={y["n"]} WR={y["wr"]*100:.1f}% PnL=${y["pnl"]:+.0f} DD=${y["dd"]:+.0f}')

        # Save
        joblib.dump({
            'model': hgb,
            'features': sig_features,
            'sigma_features_for_cluster': sigma_features,
            'hmm_features': hmm_features,
            'kmeans': best_cluster_config['km'],
            'good_clusters': sorted(good_clusters),
            'hmm': best_hmm_config['hmm'] if best_hmm_config else None,
            'good_hmm_states': sorted(good_states) if best_hmm_config else [],
            'scaler_cluster': scaler,
            'pca_cluster': pca,
            'scaler_hmm': sc_h,
            'threshold': best['thr'],
            'val_auc': float(auc_v),
        }, OUT / 'cascade_model.pkl')
        with open(OUT / 'cascade_summary.json', 'w') as f:
            json.dump({
                'threshold': best['thr'],
                'n_total': best['n'],
                'per_day': best['per_day'],
                'wr': best['wr'],
                'pnl': best['pnl'],
                'dd': best['dd'],
                'yearly': best['yearly'],
                'monthly': best['monthly'],
                'val_auc': float(auc_v),
                'good_clusters': sorted(good_clusters),
                'good_hmm_states': sorted(good_states) if best_hmm_config else [],
                'sig_features_count': len(sig_features),
            }, f, indent=2, default=str)
        print(f'\n✅ Saved cascade_model.pkl + cascade_summary.json')
    else:
        # Show closest candidates
        print(f'\n🔴 NO config met all constraints. Closest by # constraints failed:')
        candidates.sort(key=lambda c: (len(c['fails']), -c['pnl']))
        for c in candidates[:5]:
            print(f'  thr={c["thr"]:.3f}: {c["per_day"]:.1f}/d, WR={c["wr"]*100:.1f}%, PnL=${c["pnl"]:+.0f}, DD=${c["dd"]:+.0f}')
            print(f'    fails ({len(c["fails"])}): {c["fails"][:3]}')
        with open(OUT / 'cascade_summary.json', 'w') as f:
            json.dump({
                'PASSED': False,
                'closest': [{'thr': c['thr'], 'per_day': c['per_day'], 'wr': c['wr'],
                              'pnl': c['pnl'], 'dd': c['dd'], 'fails': c['fails']}
                              for c in candidates[:10]],
            }, f, indent=2, default=str)
        print(f'\n❌ Saved closest-attempt summary (no config passed)')

    print(f'\nTotal time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
