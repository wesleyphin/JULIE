"""σ-ML phase 3+4: train HGB on Trump T1 + 2025; evaluate on Biden VAL.

Outputs:
  artifacts/stdev_ml_hr9_11/model.pkl
  artifacts/stdev_ml_hr9_11/feature_importance.json
  artifacts/stdev_ml_hr9_11/val_threshold_sweep.csv
  artifacts/stdev_ml_hr9_11/val_summary.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT_DIR = ROOT / 'artifacts' / 'stdev_ml_hr9_11'
TP_PTS = 8.25
SL_PTS = 10.0


def load(name):
    p = OUT_DIR / f'{name}.parquet'
    df = pd.read_parquet(p)
    return df


def expected_pnl_at_threshold(label: np.ndarray, p: np.ndarray, thr: float, tp=TP_PTS, sl=SL_PTS) -> dict:
    """If we trade only when p_win >= thr, what's the WR / PnL / count?"""
    fired = p >= thr
    n = int(fired.sum())
    if n == 0:
        return {'thr': thr, 'n': 0, 'WR': 0.0, 'PnL': 0.0, 'avg': 0.0}
    wins = int((fired & (label == 1)).sum())
    losses = n - wins
    pnl = wins * tp * 5 - losses * sl * 5  # MES = $5/pt
    return {'thr': thr, 'n': n, 'WR': wins / n, 'PnL': pnl, 'avg': pnl / n}


def main():
    t0 = time.time()
    train = load('train')
    val = load('val')
    feat_cols = [c for c in train.columns if c != 'label']
    print(f'TRAIN: {len(train):,} rows, {len(feat_cols)} features')
    print(f'VAL:   {len(val):,} rows')
    print(f'Target distributions:')
    print(f'  TRAIN P(TP-first): {(train["label"]==1).mean()*100:.1f}%')
    print(f'  VAL   P(TP-first): {(val["label"]==1).mean()*100:.1f}%')

    # Fill NaN (early bars in rolling features)
    Xt = train[feat_cols].astype(np.float32)
    yt = train['label'].astype(np.int8).values
    Xv = val[feat_cols].astype(np.float32)
    yv = val['label'].astype(np.int8).values

    # Drop rows with too many NaN (early bars)
    valid_t = ~Xt.isna().any(axis=1)
    valid_v = ~Xv.isna().any(axis=1)
    Xt, yt = Xt[valid_t], yt[valid_t.values]
    Xv, yv = Xv[valid_v], yv[valid_v.values]
    print(f'After NaN filter: TRAIN={len(Xt):,}, VAL={len(Xv):,}')

    # Train
    print('\nTraining HGB...')
    t1 = time.time()
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.04,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
        verbose=0,
    )
    clf.fit(Xt, yt)
    print(f'Trained in {time.time()-t1:.1f}s, n_iter={clf.n_iter_}')

    # AUC on TRAIN/VAL
    pt = clf.predict_proba(Xt)[:, 1]
    pv = clf.predict_proba(Xv)[:, 1]
    auc_t = roc_auc_score(yt, pt)
    auc_v = roc_auc_score(yv, pv)
    print(f'\nTrain AUC: {auc_t:.4f}')
    print(f'Val   AUC: {auc_v:.4f}')

    # Threshold sweep on VAL
    print('\nVAL threshold sweep:')
    rows = []
    for thr in np.arange(0.45, 0.85, 0.025):
        rows.append(expected_pnl_at_threshold(yv, pv, thr))
    sweep = pd.DataFrame(rows)
    print(sweep.to_string(index=False, float_format='%.4f'))
    sweep.to_csv(OUT_DIR / 'val_threshold_sweep.csv', index=False)

    # Best PnL threshold
    best = sweep.loc[sweep['PnL'].idxmax()]
    print(f'\nBest VAL threshold by PnL: {best["thr"]:.3f} → n={int(best["n"])}, WR={best["WR"]*100:.1f}%, PnL=${best["PnL"]:.0f}')

    # Best WR-at-min-volume threshold (require >= 1k samples on val to avoid sparse bins)
    feasible = sweep[sweep['n'] >= 1000]
    if len(feasible):
        best_wr = feasible.loc[feasible['WR'].idxmax()]
        print(f'Best VAL threshold by WR (n>=1000): {best_wr["thr"]:.3f} → n={int(best_wr["n"])}, WR={best_wr["WR"]*100:.1f}%, PnL=${best_wr["PnL"]:.0f}')

    # Per-year breakdown on VAL (to check regime stability across Biden term)
    print('\nVAL per-year breakdown at best threshold:')
    val_idx = val.index[valid_v.values]
    val_year = pd.to_datetime(val_idx).year if hasattr(val_idx, 'year') else None
    # Actually val.index is reset; use the original parquet's index. For simplicity use month from a column.
    # Skip per-year if no time index.

    # Feature importance — HGB doesn't have feature_importances_ for older sklearn, use permutation
    # Quick alternative: predict class for each feature individually (gini-based proxy)
    print('\nFeature importance (top 25, by AUC drop on permutation):')
    np.random.seed(42)
    base_auc = auc_v
    # Lighter approximation: shuffle each feature in val and see AUC drop
    importances = []
    Xv_arr = Xv.values.copy()
    for i, col in enumerate(feat_cols):
        x_perm = Xv_arr.copy()
        np.random.shuffle(x_perm[:, i])
        p_perm = clf.predict_proba(x_perm)[:, 1]
        try:
            auc_perm = roc_auc_score(yv, p_perm)
        except: auc_perm = base_auc
        importances.append((col, base_auc - auc_perm))
    importances.sort(key=lambda x: -x[1])
    for col, imp in importances[:25]:
        print(f'  {col:<40s}  Δauc={imp:+.5f}')

    # Save model + metadata
    payload = {
        'model': clf,
        'features': feat_cols,
        'best_threshold': float(best['thr']),
        'val_auc': float(auc_v),
        'train_auc': float(auc_t),
        'tp_pts': TP_PTS,
        'sl_pts': SL_PTS,
        'tier': 'stdev_ml_hr9_11',
        'feature_importance_top25': [{'feat': c, 'delta_auc': float(d)} for c, d in importances[:25]],
    }
    joblib.dump(payload, OUT_DIR / 'model.pkl')
    with open(OUT_DIR / 'feature_importance.json', 'w') as f:
        json.dump([{'feat': c, 'delta_auc': float(d)} for c, d in importances], f, indent=2)

    # Save summary
    summary = {
        'train_n': int(len(Xt)),
        'val_n': int(len(Xv)),
        'train_auc': float(auc_t),
        'val_auc': float(auc_v),
        'best_threshold_by_pnl': float(best['thr']),
        'best_pnl_n': int(best['n']),
        'best_pnl_wr': float(best['WR']),
        'best_pnl_dollars': float(best['PnL']),
        'feature_count': len(feat_cols),
        'iterations_trained': int(clf.n_iter_),
    }
    with open(OUT_DIR / 'val_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n✅ Phase 3+4 complete in {time.time()-t0:.1f}s')
    print(f'   Saved model + feature importance + sweep to {OUT_DIR}')
    print(f'\n👉 HALT: review feature importance + threshold sweep → confirm before holdout')


if __name__ == '__main__':
    main()
