"""σ-ML phase 5: ONE-SHOT 2026 holdout evaluation.

Loads the locked model from phase 3+4 and evaluates on the 2026 OOS slice.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT_DIR = ROOT / 'artifacts' / 'stdev_ml_hr9_11'
TP_PTS = 8.25
SL_PTS = 10.0


def main():
    payload = joblib.load(OUT_DIR / 'model.pkl')
    clf = payload['model']
    feat_cols = payload['features']
    threshold = payload['best_threshold']
    print(f'Loaded model. Locked threshold: {threshold:.3f}')

    oos = pd.read_parquet(OUT_DIR / 'oos.parquet')
    print(f'OOS: {len(oos):,} rows')
    Xo = oos[feat_cols].astype(np.float32)
    yo = oos['label'].astype(np.int8).values
    valid = ~Xo.isna().any(axis=1)
    Xo, yo = Xo[valid], yo[valid.values]
    print(f'After NaN filter: {len(Xo):,}')

    p = clf.predict_proba(Xo)[:, 1]
    auc = roc_auc_score(yo, p)
    print(f'\nHoldout AUC: {auc:.4f}')
    print(f'Holdout baseline P(TP-first): {(yo==1).mean()*100:.1f}%')

    # Apply locked threshold
    fired = p >= threshold
    n = int(fired.sum())
    if n == 0:
        print('NO TRADES FIRED at locked threshold — gate is too strict')
        return
    wins = int((fired & (yo == 1)).sum())
    losses = n - wins
    pnl = wins * TP_PTS * 5 - losses * SL_PTS * 5

    print(f'\n=== HOLDOUT @ threshold {threshold:.3f} ===')
    print(f'Fires:    {n:>8,}  ({n/len(yo)*100:.2f}% of OOS bars-sides)')
    print(f'WR:       {wins/n*100:>7.2f}%  ({wins} W / {losses} L)')
    print(f'PnL @ size 1:  ${pnl:>10,.2f}')
    print(f'Avg/trade:     ${pnl/n:>10.2f}')

    # Per-month breakdown
    oos_idx = oos.index[valid.values]
    months = pd.to_datetime(oos_idx).to_period('M')
    print(f'\nPer-month:')
    print(f'{"month":<10s} {"fires":>7s} {"WR%":>6s} {"PnL":>10s} {"avg":>8s}')
    df_per = pd.DataFrame({'month': months, 'p': p, 'y': yo})
    df_per['fired'] = df_per['p'] >= threshold
    for m, grp in df_per.groupby('month'):
        n_m = int(grp['fired'].sum())
        if n_m == 0:
            print(f'{str(m):<10s} {0:>7d}  --   $   --   $   --')
            continue
        w_m = int((grp['fired'] & (grp['y'] == 1)).sum())
        l_m = n_m - w_m
        pnl_m = w_m * TP_PTS * 5 - l_m * SL_PTS * 5
        print(f'{str(m):<10s} {n_m:>7d} {w_m/n_m*100:>5.1f}% ${pnl_m:>8.2f} ${pnl_m/n_m:>7.2f}')

    # Per-hour breakdown (9, 10, 11 ET)
    hours = pd.to_datetime(oos_idx).hour
    print(f'\nPer-hour (ET):')
    print(f'{"hour":<5s} {"fires":>7s} {"WR%":>6s} {"PnL":>10s}')
    df_h = pd.DataFrame({'hour': hours, 'p': p, 'y': yo, 'fired': p >= threshold})
    for h, grp in df_h.groupby('hour'):
        n_h = int(grp['fired'].sum())
        if n_h == 0: continue
        w_h = int((grp['fired'] & (grp['y'] == 1)).sum())
        l_h = n_h - w_h
        pnl_h = w_h * TP_PTS * 5 - l_h * SL_PTS * 5
        print(f'ET {h:<2d} {n_h:>7d} {w_h/n_h*100:>5.1f}% ${pnl_h:>8.2f}')

    # Per-side
    if 'side' in feat_cols:
        sides = Xo['side'].values
        print('\nPer-side:')
        for s_val, label in [(0, 'LONG'), (1, 'SHORT')]:
            mask = (sides == s_val) & (p >= threshold)
            n_s = int(mask.sum())
            if n_s == 0: continue
            w_s = int((mask & (yo == 1)).sum())
            l_s = n_s - w_s
            pnl_s = w_s * TP_PTS * 5 - l_s * SL_PTS * 5
            print(f'  {label:>5s}: n={n_s:>5d} WR={w_s/n_s*100:>5.1f}% PnL=${pnl_s:>+9.2f}')

    # Save holdout summary
    summary = {
        'auc': float(auc),
        'threshold': float(threshold),
        'fires': int(n),
        'wins': int(wins),
        'losses': int(losses),
        'wr': float(wins / n) if n else 0.0,
        'pnl_size1': float(pnl),
        'avg_pnl_per_trade': float(pnl / n) if n else 0.0,
        'oos_rows_total': int(len(yo)),
        'fire_rate': float(n / len(yo)) if len(yo) else 0.0,
    }
    with open(OUT_DIR / 'holdout_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved holdout_summary.json')

    # Ship gates
    print('\n=== SHIP GATES ===')
    gates = {
        'PnL > 0': pnl > 0,
        'WR > 50%': (wins / n if n else 0) > 0.5,
        'AUC > 0.55': auc > 0.55,
        'Fires >= 100': n >= 100,
        'Avg/trade > $5': (pnl / n if n else 0) > 5,
    }
    passed = sum(gates.values())
    for g, p_ in gates.items():
        print(f'  {"✅" if p_ else "❌"}  {g}')
    print(f'\nGates passed: {passed}/{len(gates)}')
    if passed >= 4:
        print('🟢 SHIP CANDIDATE — ≥4/5 gates passed')
    else:
        print('🔴 KILL — insufficient gates passed')


if __name__ == '__main__':
    main()
