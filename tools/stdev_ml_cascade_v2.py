"""σ-ML cascade v2: tighter brackets + side-aware + multi-criteria gate iteration.

Iteration 1 failed because TP=8.25/SL=10 has 54.8% breakeven WR which is
above any single regime's WR ceiling on σ-features alone (max ~55%).

Iteration 2 strategy:
  - Try multiple bracket geometries (1:1, 2:1, asymmetric)
  - Side-stratified: train separate model per side OR use side-conditional features
  - Loosen "good cluster" threshold to WR > breakeven_wr + 2%
  - Iterate over: brackets × thresholds × cluster K × HMM states
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
OUT = ROOT / 'artifacts' / 'stdev_ml_hr9_11' / 'cascade_v2'
OUT.mkdir(parents=True, exist_ok=True)

PT_USD = 5.0


def per_month_eval_brackets(timestamps, fired, win_rate, side_arr, tp, sl, val_idx):
    """Per-month + per-year aggregation with custom brackets.

    Note: 'fired' bool array, 'win_rate' here is actually win_label (1/0) at
    THE ORIGINAL TP/SL labels — we re-use them with new brackets by
    re-deriving from MFE/MAE walk-forward... but we don't have that data.

    Instead, this version requires re-labeling at the new TP/SL. Done
    upstream.
    """
    df = pd.DataFrame({
        'ts': timestamps, 'fired': fired, 'win': win_rate, 'side': side_arr
    })
    df['month'] = df['ts'].dt.to_period('M')
    df['year'] = df['ts'].dt.year
    df['pnl'] = np.where(
        df['fired'],
        np.where(df['win'] == 1, tp * PT_USD, -sl * PT_USD),
        0.0,
    )
    monthly, yearly = [], []
    for m, grp in df.groupby('month'):
        n = int(grp['fired'].sum())
        if n == 0: continue
        w = int((grp['fired'] & (grp['win'] == 1)).sum())
        pnl = float(grp['pnl'].sum())
        monthly.append({'month': str(m), 'n': n, 'w': w, 'l': n - w,
                        'wr': w/n, 'pnl': pnl, 'avg': pnl/n})
    for y, grp in df.groupby('year'):
        ord_pnl = grp[grp['fired']].sort_values('ts')['pnl'].cumsum()
        if len(ord_pnl) == 0: continue
        peak = ord_pnl.cummax()
        dd = float((ord_pnl - peak).min())
        n = int(grp['fired'].sum())
        w = int((grp['fired'] & (grp['win'] == 1)).sum())
        pnl = float(grp['pnl'].sum())
        yearly.append({'year': int(y), 'n': n, 'wr': w/n if n else 0,
                       'pnl': pnl, 'dd': dd})
    full_pnl = df[df['fired']].sort_values('ts')['pnl'].cumsum()
    full_dd = float((full_pnl - full_pnl.cummax()).min()) if len(full_pnl) else 0
    return monthly, yearly, full_dd


def check_constraints(monthly, yearly, full_dd, n_total, n_days):
    fails = []
    for y in yearly:
        if y['dd'] < -900: fails.append(f"YR{y['year']}-DD${y['dd']:.0f}")
        if y['pnl'] <= 0: fails.append(f"YR{y['year']}-PnL${y['pnl']:.0f}")
        if y['wr'] <= 0.50: fails.append(f"YR{y['year']}-WR{y['wr']:.2%}")
    neg_m = [m for m in monthly if m['pnl'] <= 0]
    if neg_m: fails.append(f"{len(neg_m)}m-NEG-PnL")
    low_wr = [m for m in monthly if m['wr'] <= 0.50]
    if low_wr: fails.append(f"{len(low_wr)}m-WR<50")
    avg_pd = n_total / max(n_days, 1)
    if avg_pd < 2: fails.append(f"avg{avg_pd:.2f}/d<2")
    if avg_pd > 4: fails.append(f"avg{avg_pd:.2f}/d>4")
    return len(fails) == 0, fails


def relabel_via_walk_forward(bars_full_idx, val_idx_set, tp, sl, h_min):
    """We don't have this — we'd need full bar high/low data for every val timestamp.
    For iteration 2 we work with the existing TP=8.25/SL=10 labels and only
    explore alternative thresholds + side filtering. Bracket-relabeling
    is a separate run that requires loading bars again."""
    return None  # placeholder


def find_best_for_brackets(train_p, train_y, val_p, val_y, val_ts, val_side,
                           tp, sl, val_days, label_method='current',
                           cluster_mask_v=None, hmm_mask_v=None):
    """Sweep thresholds at given bracket geometry. Return list of candidates."""
    candidates = []
    breakeven_wr = sl / (tp + sl)
    for thr in np.arange(0.99, 0.40, -0.01):
        # Cascade gates
        fired = (val_p >= thr)
        if cluster_mask_v is not None: fired = fired & cluster_mask_v
        if hmm_mask_v is not None: fired = fired & hmm_mask_v
        n = int(fired.sum())
        if n == 0: continue
        if n < val_days * 1: continue  # need at least 1/day to be meaningful
        wins = (val_y == 1)
        monthly, yearly, full_dd = per_month_eval_brackets(
            val_ts, fired, val_y, val_side, tp, sl, None
        )
        n_total = sum(m['n'] for m in monthly)
        if n_total == 0: continue
        wr_total = sum(m['w'] for m in monthly) / n_total
        pnl_total = sum(m['pnl'] for m in monthly)
        per_day = n_total / val_days
        passed, fails = check_constraints(monthly, yearly, full_dd, n_total, val_days)
        candidates.append({
            'thr': float(thr), 'tp': tp, 'sl': sl,
            'n': n_total, 'per_day': per_day,
            'wr': wr_total, 'pnl': pnl_total, 'dd': full_dd,
            'passed': passed, 'fails': fails,
            'monthly': monthly, 'yearly': yearly,
            'breakeven_wr': breakeven_wr,
        })
    return candidates


def main():
    t0 = time.time()
    train = pd.read_parquet(SRC_DIR / 'train.parquet')
    val = pd.read_parquet(SRC_DIR / 'val.parquet')
    train.index = pd.to_datetime(train.index)
    val.index = pd.to_datetime(val.index)
    feat_cols = [c for c in train.columns if c != 'label']
    print(f'TRAIN: {len(train):,}, VAL: {len(val):,}, features: {len(feat_cols)}')

    Xt = train[feat_cols].astype(np.float32)
    yt = train['label'].astype(np.int8).values
    Xv = val[feat_cols].astype(np.float32)
    yv = val['label'].astype(np.int8).values

    valid_t = ~Xt.isna().any(axis=1)
    valid_v = ~Xv.isna().any(axis=1)
    Xt = Xt[valid_t].reset_index(drop=True); yt = yt[valid_t.values]
    Xv = Xv[valid_v].reset_index(drop=True); yv = yv[valid_v.values]
    val_ts = val.index[valid_v.values]
    train_ts = train.index[valid_t.values]
    val_days = pd.to_datetime(val_ts).normalize().nunique()
    print(f'After NaN: TRAIN={len(Xt):,}, VAL={len(Xv):,}, val_days={val_days}')

    val_side = Xv['side'].values  # 0=LONG, 1=SHORT

    # ============ ITERATION over (side filter, K, HMM-states, threshold) ============
    print('\n=== ITERATION 2: side-stratified search ===')

    # Top features by F-stat
    F, pvals = f_classif(Xt.fillna(0).values, yt)
    feature_anova = pd.DataFrame({'feature': feat_cols, 'F': F, 'p': pvals}).sort_values('F', ascending=False)
    sig_features = feature_anova.head(40)['feature'].tolist()
    sigma_features = [f for f in sig_features if 'sigma' in f or 'z_' in f][:15]

    # Train one HGB per side (LONG-only and SHORT-only)
    best_overall = None

    for side_filter, side_label in [(0, 'LONG'), (1, 'SHORT'), (None, 'BOTH')]:
        print(f'\n--- Side filter: {side_label} ---')
        if side_filter is None:
            t_mask = np.ones(len(Xt), dtype=bool)
            v_mask = np.ones(len(Xv), dtype=bool)
        else:
            t_mask = (Xt['side'].values == side_filter)
            v_mask = (Xv['side'].values == side_filter)

        Xt_s = Xt[t_mask][sig_features].fillna(0).astype(np.float32)
        yt_s = yt[t_mask]
        Xv_s = Xv[v_mask][sig_features].fillna(0).astype(np.float32)
        yv_s = yv[v_mask]
        val_ts_s = val_ts[v_mask]
        val_side_s = val_side[v_mask]

        if len(Xt_s) < 5000 or len(Xv_s) < 1000:
            continue

        print(f'  TRAIN n={len(Xt_s):,}, VAL n={len(Xv_s):,}, baseline VAL WR={yv_s.mean():.3f}')

        hgb = HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.05, max_iter=400,
            early_stopping=True, n_iter_no_change=15,
            random_state=42, verbose=0,
        )
        hgb.fit(Xt_s, yt_s)
        pv = hgb.predict_proba(Xv_s)[:, 1]
        try: auc = roc_auc_score(yv_s, pv)
        except: auc = 0.5
        print(f'  Val AUC: {auc:.4f}')

        # Try multiple bracket geometries
        for tp, sl in [(8.25, 10), (4, 4), (3, 3), (4, 6), (6, 6), (2, 2), (5, 5)]:
            # WARNING: yv_s was labeled at TP=8.25/SL=10 — different brackets need re-label
            # For TP=8.25/SL=10 we use yv_s directly. For others, we'd need MFE/MAE.
            # PRAGMATIC: only sweep at the labeled bracket for now.
            if (tp, sl) != (8.25, 10):
                continue  # skip non-default brackets — would need bar-level re-label

            cands = find_best_for_brackets(
                None, None, pv, yv_s, val_ts_s, val_side_s,
                tp, sl, val_days,
            )
            passing = [c for c in cands if c['passed']]
            print(f'    TP={tp}/SL={sl}: {len(cands)} thresholds explored, {len(passing)} pass')
            if passing:
                best = max(passing, key=lambda c: c['n'])
                print(f'      🟢 PASSING: thr={best["thr"]:.3f}, n={best["n"]} ({best["per_day"]:.2f}/d)')
                print(f'         WR={best["wr"]*100:.1f}%, PnL=+${best["pnl"]:.0f}, DD=${best["dd"]:.0f}')
                if not best_overall or best['n'] > best_overall['n']:
                    best_overall = {**best, 'side_filter': side_filter, 'side_label': side_label, 'auc': auc}

            # Show closest if no pass
            if not passing and cands:
                cands.sort(key=lambda c: (len(c['fails']), -c['pnl']))
                c = cands[0]
                print(f'      closest (fails={len(c["fails"])}): thr={c["thr"]:.3f}, {c["per_day"]:.2f}/d, WR={c["wr"]*100:.1f}%, PnL=${c["pnl"]:+.0f}, DD=${c["dd"]:.0f}')

    if best_overall:
        print(f'\n🟢 BEST PASSING: side={best_overall["side_label"]}, thr={best_overall["thr"]:.3f}')
        print(f'   {best_overall["per_day"]:.2f}/d, WR={best_overall["wr"]*100:.1f}%, PnL=+${best_overall["pnl"]:.0f}, DD=${best_overall["dd"]:.0f}')
        for y in best_overall['yearly']:
            print(f'   {y["year"]}: n={y["n"]} WR={y["wr"]*100:.1f}% PnL=${y["pnl"]:+.0f} DD=${y["dd"]:+.0f}')
        with open(OUT / 'iter2_passing.json', 'w') as f:
            json.dump(best_overall, f, indent=2, default=str)
    else:
        print(f'\n❌ Iteration 2 — no passing config under TP=8.25/SL=10 for any side filter.')
        print('   Need: bar-level re-labeling at tighter brackets (TP=4/SL=4 etc) to test 1:1 R:R geometry.')

    print(f'\nTotal time: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
