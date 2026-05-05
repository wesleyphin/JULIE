"""Iter 12 — add SHORT side + combine both sides for more uniform fire rate.

Iter 11 nailed the LONG side but went silent in regimes where the LONG model's
confidence distribution sat below the DD-safe threshold (e.g. Feb 2026: 4 fires
vs Jan 2026: 676 fires). To meet the user's "every month positive AND ≥2/d"
constraint, we need fires in regime-silent months too.

Approach:
  - Generate both LONG and SHORT labels for every bar in the window
  - Train separate HGB models for each side
  - At inference, pick the higher-confidence side; fire if it crosses threshold
  - Sweep thresholds × caps × bracket geometries
  - Check stricter constraint: every month PnL>0 AND ≥2/d
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, '/Users/wes/Downloads/JULIE001/tools')
from stdev_ml_iter11_atr import (
    compute_features, load_period, PERIODS, WIN_HOURS, H_MIN_NEW, PT_USD,
)

warnings.filterwarnings("ignore")

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter12_both_sides'
OUT.mkdir(parents=True, exist_ok=True)


def label_walk_both_sides(opens, highs, lows, closes, atrs, win_idx, h_min, tp_mult, sl_mult):
    """Label both LONG and SHORT for each window bar; emit two rows per bar."""
    n = len(opens)
    rows = []
    for i in win_idx:
        if i + h_min >= n: continue
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0: continue
        entry = opens[i+1] if i+1 < n else closes[i]
        tp_pts = atr * tp_mult
        sl_pts = atr * sl_mult
        end = min(i+1+h_min, n)
        seg_lows = lows[i+1:end]
        seg_highs = highs[i+1:end]

        # LONG: TP above, SL below
        sl_long_hits = np.where(seg_lows <= entry - sl_pts)[0]
        tp_long_hits = np.where(seg_highs >= entry + tp_pts)[0]
        sl_long_first = sl_long_hits[0] if len(sl_long_hits) else 10**9
        tp_long_first = tp_long_hits[0] if len(tp_long_hits) else 10**9
        if not (sl_long_first == 10**9 and tp_long_first == 10**9):
            label_long = 1 if tp_long_first < sl_long_first else 0
            rows.append((i, 0, label_long, tp_pts, sl_pts))  # side=0 LONG

        # SHORT: TP below, SL above
        sl_short_hits = np.where(seg_highs >= entry + sl_pts)[0]
        tp_short_hits = np.where(seg_lows <= entry - tp_pts)[0]
        sl_short_first = sl_short_hits[0] if len(sl_short_hits) else 10**9
        tp_short_first = tp_short_hits[0] if len(tp_short_hits) else 10**9
        if not (sl_short_first == 10**9 and tp_short_first == 10**9):
            label_short = 1 if tp_short_first < sl_short_first else 0
            rows.append((i, 1, label_short, tp_pts, sl_pts))  # side=1 SHORT
    return rows


def build_period(name, start, end, tp_mult, sl_mult):
    print(f'\n[{name}] {start}→{end}, TP_mult={tp_mult} SL_mult={sl_mult} (LONG+SHORT)')
    t0 = time.time()
    bars = load_period(start, end)
    if bars.empty: return None
    df = compute_features(bars)
    print(f'  features computed: {len(df.columns)} cols, dt={int(time.time()-t0)}s')

    in_win = (df.index.hour >= WIN_HOURS[0]) & (df.index.hour < WIN_HOURS[1])
    win_idx = np.where(in_win)[0]
    o = df['open'].values; h = df['high'].values; l = df['low'].values; c = df['close'].values
    atr = df['atr_60'].values
    rows = label_walk_both_sides(o, h, l, c, atr, win_idx, H_MIN_NEW, tp_mult, sl_mult)
    if not rows: return None
    print(f'  labeled rows: {len(rows):,}')

    feat_cols = [col for col in df.columns
                 if col not in ('open','high','low','close','volume','symbol','side','label','tp_pts','sl_pts')
                 and pd.api.types.is_numeric_dtype(df[col])]
    feat_df = df[feat_cols].copy()

    # Build output: two rows per bar (one LONG, one SHORT)
    idxs = [r[0] for r in rows]
    sides = [r[1] for r in rows]
    labels = [r[2] for r in rows]
    tps = [r[3] for r in rows]
    sls = [r[4] for r in rows]
    out_index = df.index[idxs]
    out_features = feat_df.iloc[idxs].copy()
    out_features['_orig_side'] = sides
    out_features['label'] = labels
    out_features['tp_pts'] = tps
    out_features['sl_pts'] = sls
    out_features.index = out_index
    print(f'  → {len(out_features):,} samples, dt={int(time.time()-t0)}s')
    return out_features


def evaluate_combined(p_long, p_short, y_long, y_short, ts_long, ts_short,
                     tp_long, tp_short, sl_long, sl_short, thr, val_days, daily_dd_cap):
    """Combine LONG and SHORT predictions: at each timestamp, pick the higher-confidence side."""
    # Build per-bar dataframes
    df_l = pd.DataFrame({'ts': pd.to_datetime(ts_long), 'p': p_long, 'y': y_long,
                         'tp': tp_long, 'sl': sl_long, 'side': 0})
    df_s = pd.DataFrame({'ts': pd.to_datetime(ts_short), 'p': p_short, 'y': y_short,
                         'tp': tp_short, 'sl': sl_short, 'side': 1})
    # For each timestamp, pick max-p side
    df_all = pd.concat([df_l, df_s], axis=0).sort_values(['ts', 'p'], ascending=[True, False])
    df_max = df_all.drop_duplicates(subset='ts', keep='first').reset_index(drop=True)

    fired = (df_max['p'] >= thr).values
    if not fired.any(): return None
    df_max['fired'] = fired
    df_max['win'] = (df_max['y'] == 1) & fired
    df_max['date'] = df_max['ts'].dt.normalize()
    df_max['raw_pnl'] = np.where(df_max['fired'],
                                 np.where(df_max['win'], df_max['tp'] * PT_USD, -df_max['sl'] * PT_USD), 0)

    out_pnl = np.zeros(len(df_max))
    out_fired = np.zeros(len(df_max), dtype=bool)
    out_win = np.zeros(len(df_max), dtype=bool)
    for date, grp in df_max.groupby('date'):
        cum = 0.0; muted = False
        for idx in grp.index:
            if not df_max.at[idx, 'fired']: continue
            if muted: continue
            pnl = df_max.at[idx, 'raw_pnl']
            out_pnl[idx] = pnl
            out_fired[idx] = True
            out_win[idx] = bool(df_max.at[idx, 'win'])
            cum += pnl
            if cum <= -daily_dd_cap: muted = True
    df_max['cb_pnl'] = out_pnl
    df_max['cb_fired'] = out_fired
    df_max['cb_win'] = out_win
    df_max['month'] = df_max['ts'].dt.to_period('M')
    df_max['year'] = df_max['ts'].dt.year

    monthly_df = df_max[df_max['cb_fired']].groupby('month').agg(
        n=('cb_fired', 'sum'), w=('cb_win', 'sum'), pnl=('cb_pnl', 'sum')
    ).reset_index()
    monthly_df['wr'] = monthly_df['w'] / monthly_df['n']

    # All months covered (whether fired or not)
    all_months = df_max['month'].unique()
    monthly_full = []
    days_per_month = df_max.groupby('month')['date'].nunique().to_dict()
    for mo in sorted(all_months):
        sub = monthly_df[monthly_df['month'] == mo]
        if len(sub):
            r = sub.iloc[0]
            monthly_full.append({'month': str(mo), 'n': int(r['n']), 'pnl': float(r['pnl']),
                                'wr': float(r['wr']), 'days': int(days_per_month[mo]),
                                'per_day': r['n'] / max(days_per_month[mo], 1)})
        else:
            monthly_full.append({'month': str(mo), 'n': 0, 'pnl': 0.0, 'wr': 0.0,
                                'days': int(days_per_month[mo]), 'per_day': 0.0})

    yearly = []
    for y_, grp in df_max[df_max['cb_fired']].groupby('year'):
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
        'monthly': monthly_full,
        'yearly': yearly,
    }


def check_strict(stats):
    """User constraints: ≥2/d (avg), DD≤$900/yr, every month PnL>0 AND ≥2/d."""
    fails = []
    if stats['per_day'] < 2: fails.append(f"avg{stats['per_day']:.2f}/d<2")
    for y in stats['yearly']:
        if y['dd'] < -900: fails.append(f"YR{y['year']}-DD${y['dd']:.0f}")
    n_mo = len(stats['monthly'])
    bad_pnl_mos = [m['month'] for m in stats['monthly'] if m['pnl'] <= 0]
    bad_rate_mos = [m['month'] for m in stats['monthly'] if m['per_day'] < 2]
    if bad_pnl_mos: fails.append(f"mo_PnL≤0:{','.join(bad_pnl_mos)}")
    if bad_rate_mos: fails.append(f"mo<2/d:{','.join(bad_rate_mos)}")
    return len(fails) == 0, fails


def run_bracket(tp_mult, sl_mult):
    print(f'\n========== TP_mult={tp_mult}×ATR / SL_mult={sl_mult}×ATR (LONG+SHORT) ==========')

    all_data = {}
    for name, (s, e) in PERIODS.items():
        df = build_period(name, s, e, tp_mult, sl_mult)
        if df is None or df.empty: continue
        all_data[name] = df

    train = pd.concat([all_data.get('TRAIN_t1', pd.DataFrame()),
                       all_data.get('TRAIN_2025', pd.DataFrame())], axis=0)
    val = all_data['VAL_biden']
    oos = all_data['OOS_2026']

    # Feature columns: everything that's numeric and not a label/side helper
    feat_cols = [c for c in train.columns if c not in ('label','tp_pts','sl_pts','_orig_side')]

    def prep(df_):
        X = df_[feat_cols].astype(np.float32)
        y = df_['label'].astype(np.int8).values
        side = df_['_orig_side'].astype(np.int8).values
        valid = (X.isna().sum(axis=1) < len(feat_cols)*0.5)
        X = X[valid].reset_index(drop=True); y = y[valid.values]; side = side[valid.values]
        ts = df_.index[valid.values]
        tp = df_['tp_pts'].values[valid.values]
        sl = df_['sl_pts'].values[valid.values]
        return X, y, side, ts, tp, sl

    Xt, yt, st, _, _, _ = prep(train)
    Xv, yv, sv, ts_v, tp_v, sl_v = prep(val)
    Xo, yo, so, ts_o, tp_o, sl_o = prep(oos)
    val_days = pd.to_datetime(ts_v).normalize().nunique()
    oos_days = pd.to_datetime(ts_o).normalize().nunique()

    print(f'  TRAIN={len(Xt):,}, VAL={len(Xv):,} ({val_days}d), OOS={len(Xo):,} ({oos_days}d)')
    print(f'  baseline WR — TRAIN: L={yt[st==0].mean()*100:.1f}% S={yt[st==1].mean()*100:.1f}%')
    print(f'  baseline WR — VAL:   L={yv[sv==0].mean()*100:.1f}% S={yv[sv==1].mean()*100:.1f}%')
    print(f'  baseline WR — OOS:   L={yo[so==0].mean()*100:.1f}% S={yo[so==1].mean()*100:.1f}%')

    # Train separate models per side
    models = {}
    for side_label, side_id in [('LONG', 0), ('SHORT', 1)]:
        mask_t = st == side_id; mask_v = sv == side_id; mask_o = so == side_id
        Xt_s = Xt[mask_t].reset_index(drop=True); yt_s = yt[mask_t]
        Xv_s = Xv[mask_v].reset_index(drop=True); yv_s = yv[mask_v]
        Xo_s = Xo[mask_o].reset_index(drop=True); yo_s = yo[mask_o]

        Xt_clean = Xt_s.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
        F, _ = f_classif(Xt_clean.values, yt_s)
        fa = pd.DataFrame({'f': feat_cols, 'F': F}).sort_values('F', ascending=False)
        top60 = fa.head(60)['f'].tolist()

        Xt_hgb = Xt_s[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
        Xv_hgb = Xv_s[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
        Xo_hgb = Xo_s[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
        hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400,
                                             early_stopping=True, n_iter_no_change=15, random_state=42)
        hgb.fit(Xt_hgb, yt_s)
        pv = hgb.predict_proba(Xv_hgb)[:, 1]
        po = hgb.predict_proba(Xo_hgb)[:, 1]
        auc_v = roc_auc_score(yv_s, pv)
        auc_o = roc_auc_score(yo_s, po)
        print(f'  {side_label} HGB AUC: VAL={auc_v:.4f}, OOS={auc_o:.4f} (n_train={len(Xt_s):,})')

        models[side_id] = {'hgb': hgb, 'top60': top60, 'auc_v': float(auc_v), 'auc_o': float(auc_o),
                          'pv': pv, 'po': po, 'mask_v': mask_v, 'mask_o': mask_o}

    # Combine: rebuild aligned per-bar predictions for both sides
    # NOTE: VAL/OOS rows are 2 per bar (LONG, SHORT). We need ts × side aligned.
    # ts_v[mask_v_long] gives LONG timestamps; ts_v[mask_v_short] gives SHORT timestamps.
    # They should match.
    p_v_long = models[0]['pv']; ts_v_long = ts_v[models[0]['mask_v']]
    y_v_long = yv[models[0]['mask_v']]; tp_v_long = tp_v[models[0]['mask_v']]; sl_v_long = sl_v[models[0]['mask_v']]
    p_v_short = models[1]['pv']; ts_v_short = ts_v[models[1]['mask_v']]
    y_v_short = yv[models[1]['mask_v']]; tp_v_short = tp_v[models[1]['mask_v']]; sl_v_short = sl_v[models[1]['mask_v']]

    p_o_long = models[0]['po']; ts_o_long = ts_o[models[0]['mask_o']]
    y_o_long = yo[models[0]['mask_o']]; tp_o_long = tp_o[models[0]['mask_o']]; sl_o_long = sl_o[models[0]['mask_o']]
    p_o_short = models[1]['po']; ts_o_short = ts_o[models[1]['mask_o']]
    y_o_short = yo[models[1]['mask_o']]; tp_o_short = tp_o[models[1]['mask_o']]; sl_o_short = sl_o[models[1]['mask_o']]

    # Threshold sweep on VAL with strict constraint
    print(f'\n  ===== VAL sweep (combined LONG+SHORT) — strict every-month-2/d =====')
    print(f'  {"thr":>5s} {"cap":>5s} {"n":>5s} {"/d":>5s} {"WR%":>5s} {"avg":>7s} {"PnL":>10s} {"yrs+":>5s} {"maxDD":>10s} {"mos<2/d":>9s} {"mos≤0":>7s} {"PASS":>5s}')
    val_passing = []
    for thr in [0.95,0.91,0.87,0.83,0.79,0.75,0.71,0.67,0.63,0.59,0.55,0.51]:
        for cap in [100, 150, 200, 300]:
            stats = evaluate_combined(p_v_long, p_v_short, y_v_long, y_v_short, ts_v_long, ts_v_short,
                                       tp_v_long, tp_v_short, sl_v_long, sl_v_short, thr, val_days, cap)
            if stats is None: continue
            passed, fails = check_strict(stats)
            yrs_pos = sum(1 for yy in stats['yearly'] if yy['pnl'] > 0 and yy['wr'] > 0.5 and yy['dd'] >= -900)
            max_dd = min((yy['dd'] for yy in stats['yearly']), default=0)
            mos_low = sum(1 for m in stats['monthly'] if m['per_day'] < 2)
            mos_neg = sum(1 for m in stats['monthly'] if m['pnl'] <= 0)
            if stats['per_day'] >= 1 and max_dd >= -900:
                flag = '🟢' if passed else '  '
                print(f'  {thr:.2f} ${cap:>3d} {stats["n"]:>5d} {stats["per_day"]:>4.2f} {stats["wr"]*100:>4.1f}% ${stats["avg"]:>+5.0f} ${stats["pnl"]:>+8.0f} {yrs_pos:>2d}/{len(stats["yearly"])} ${max_dd:>+8.0f} {mos_low:>5d}/{len(stats["monthly"])} {mos_neg:>5d}/{len(stats["monthly"])} {flag}')
                if passed:
                    val_passing.append({**stats, 'thr': float(thr), 'cap': cap, 'tp_mult': tp_mult, 'sl_mult': sl_mult})

    # OOS at the top VAL-passing configs
    print(f'\n  ===== OOS sweep (combined LONG+SHORT) =====')
    print(f'  {"thr":>5s} {"cap":>5s} {"n":>5s} {"/d":>5s} {"WR%":>5s} {"avg":>7s} {"PnL":>10s} {"DD":>10s} {"mos<2/d":>9s} {"mos≤0":>7s}')
    oos_results = []
    for thr in [0.95,0.91,0.87,0.83,0.79,0.75,0.71,0.67,0.63,0.59,0.55,0.51]:
        for cap in [100, 150, 200, 300]:
            stats = evaluate_combined(p_o_long, p_o_short, y_o_long, y_o_short, ts_o_long, ts_o_short,
                                       tp_o_long, tp_o_short, sl_o_long, sl_o_short, thr, oos_days, cap)
            if stats is None: continue
            yr_dd = stats['yearly'][0]['dd'] if stats['yearly'] else 0
            mos_low = sum(1 for m in stats['monthly'] if m['per_day'] < 2)
            mos_neg = sum(1 for m in stats['monthly'] if m['pnl'] <= 0)
            if stats['per_day'] >= 1 and yr_dd >= -900:
                flag = '🟢' if (mos_low == 0 and mos_neg == 0) else '  '
                print(f'  {thr:.2f} ${cap:>3d} {stats["n"]:>5d} {stats["per_day"]:>4.2f} {stats["wr"]*100:>4.1f}% ${stats["avg"]:>+5.0f} ${stats["pnl"]:>+8.0f} ${yr_dd:>+8.0f} {mos_low:>5d}/{len(stats["monthly"])} {mos_neg:>5d}/{len(stats["monthly"])} {flag}')
                oos_results.append({**stats, 'thr': float(thr), 'cap': cap})

    return models, val_passing, oos_results


def main():
    t0 = time.time()
    all_results = {}
    for tp_mult, sl_mult in [(1.5, 1.0), (2.0, 1.0), (1.2, 1.0), (1.8, 1.2)]:
        models, val_pass, oos_res = run_bracket(tp_mult, sl_mult)
        all_results[f'{tp_mult}_{sl_mult}'] = {
            'val_passing': val_pass,
            'oos_results': oos_res,
        }
        # Save models for the bracket
        for side_id, m in models.items():
            joblib.dump({
                'hgb': m['hgb'], 'features': m['top60'],
                'auc_val': m['auc_v'], 'auc_oos': m['auc_o'],
                'tp_mult': tp_mult, 'sl_mult': sl_mult,
                'side': 'LONG' if side_id == 0 else 'SHORT',
            }, OUT / f'iter12_{tp_mult}_{sl_mult}_{"L" if side_id==0 else "S"}.pkl')

    print(f'\n========== SUMMARY ==========')
    for bracket, r in all_results.items():
        print(f'  TP/SL={bracket}: {len(r["val_passing"])} VAL passing configs, {len(r["oos_results"])} OOS w/ DD≤$900')
    print(f'\nWall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
