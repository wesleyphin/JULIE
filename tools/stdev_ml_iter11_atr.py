"""Iter 11 — different thesis: σ-ML on σ's natural horizon.

Hypothesis: σ predicts near-term volatility, not 2-hour direction. Iters 1-10
forced σ to predict 120-min TP-first classification → AUC 0.55 ceiling because
the prediction horizon was too far. Try:

  1. Shorter label horizon: 60-min TP/SL instead of 120-min
  2. ATR-scaled bracket: TP and SL set as multiples of realized 60-min volatility,
     so calm bars get tight brackets and volatile bars get wide ones — adapts to
     regime instead of forcing a fixed point bracket
  3. Cross-sectional time-of-day baseline features: "is THIS 11:30 unusual vs
     last 20 days' 11:30" — surfaces intraday seasonality the rolling-window
     features couldn't capture
  4. Drop macro features entirely (post-fix they added zero AUC)

Train: Trump T1 (2017-2021) + 2025-01-20→2025-12-31
Val:   Biden (2021-01-21→2025-01-19)
OOS:   2026-01-01→2026-04-30

If this doesn't beat AUC 0.55 with profitable per-year DD ≤ $900, the σ-ML
strategy at this window is a real dead end and we stop.
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

warnings.filterwarnings("ignore")

ROOT = Path('/Users/wes/Downloads/JULIE001')
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter11_atr'
OUT.mkdir(parents=True, exist_ok=True)
BAR = ROOT / 'es_master_outrights-2.parquet'

import pytz
ET = pytz.timezone('America/New_York')
PT_USD = 5.0  # MES point value
WIN_HOURS = (11, 12)  # ET 11-12 = PT 8-9
H_MIN_NEW = 60  # SHORTER horizon — was 120

PERIODS = {
    'TRAIN_t1': ('2017-01-20', '2021-01-20'),
    'TRAIN_2025': ('2025-01-20', '2025-12-31'),
    'VAL_biden': ('2021-01-21', '2025-01-19'),
    'OOS_2026':  ('2026-01-01', '2026-04-30'),
}


def load_period(start, end):
    bars = pd.read_parquet(BAR)
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert(ET)
    bars = bars.loc[start:end].copy()
    return bars


def rolling_slope(series, w):
    def slope_calc(y):
        if len(y) < 2: return 0
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0] if not np.isnan(y).any() else 0
    return series.rolling(w).apply(slope_calc, raw=True)


def rsi(close, w):
    delta = close.diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(w).mean(); avg_loss = loss.rolling(w).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_features(bars):
    df = bars.copy()
    df['ret_1'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_5'] = np.log(df['close'] / df['close'].shift(5))
    df['ret_15'] = np.log(df['close'] / df['close'].shift(15))
    df['ret_60'] = np.log(df['close'] / df['close'].shift(60))
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()

    # σ family
    sigmas = {}
    for w in (5, 10, 15, 30, 60, 120, 240):
        s = df['ret_1'].rolling(w).std() * np.sqrt(w)
        df[f'sigma_{w}'] = s
        df[f'sigma_{w}_log'] = np.log(s.replace(0, np.nan))
        sigmas[w] = s
    df['sigma_intraday'] = df['ret_1'].rolling(60).std()

    # σ ratios + percentiles
    df['sigma_5_15'] = sigmas[5] / sigmas[15].replace(0, np.nan)
    df['sigma_15_60'] = sigmas[15] / sigmas[60].replace(0, np.nan)
    df['sigma_60_240'] = sigmas[60] / sigmas[240].replace(0, np.nan)
    df['sigma_15_pctile_240'] = sigmas[15].rolling(240).rank(pct=True)
    df['sigma_60_pctile_390'] = sigmas[60].rolling(390).rank(pct=True)
    df['sigma_60_pctile_hourly_20d'] = sigmas[60].rolling(240*20).rank(pct=True)

    # σ-of-σ
    df['sigma_of_sigma_15_60'] = sigmas[15].rolling(60).std()
    df['sigma_of_sigma_60_240'] = sigmas[60].rolling(240).std()

    # ATR (will be used for bracket scaling)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['atr_60'] = tr.rolling(60).mean()
    df['atr_15'] = tr.rolling(15).mean()
    df['atr_ratio'] = df['atr_15'] / df['atr_60'].replace(0, np.nan)

    # Z-scores
    rng = df['range']
    df['z_range_60'] = (rng - rng.rolling(60).mean()) / rng.rolling(60).std().replace(0, np.nan)
    df['z_volume_60'] = (df['volume'] - df['volume'].rolling(60).mean()) / df['volume'].rolling(60).std().replace(0, np.nan)
    df['z_return_5'] = (df['ret_5'] - df['ret_5'].rolling(60).mean()) / df['ret_5'].rolling(60).std().replace(0, np.nan)
    df['z_return_15'] = (df['ret_15'] - df['ret_15'].rolling(60).mean()) / df['ret_15'].rolling(60).std().replace(0, np.nan)
    df['skew_60'] = df['ret_1'].rolling(60).skew()
    df['kurt_60'] = df['ret_1'].rolling(60).kurt()

    # Direction
    df['slope_15'] = rolling_slope(df['close'], 15)
    df['slope_60'] = rolling_slope(df['close'], 60)
    df['momentum_z_15'] = df['ret_1'].rolling(15).mean() / df['ret_1'].rolling(15).std().replace(0, np.nan)
    df['momentum_z_60'] = df['ret_1'].rolling(60).mean() / df['ret_1'].rolling(60).std().replace(0, np.nan)
    df['rsi_14'] = rsi(df['close'], 14)
    df['rsi_60'] = rsi(df['close'], 60)

    # VWAP / range pos
    pv = df['close'] * df['volume']
    df['vwap_60'] = pv.rolling(60).sum() / df['volume'].rolling(60).sum().replace(0, np.nan)
    df['vwap_dist_60'] = (df['close'] - df['vwap_60']) / df['close'].rolling(60).std().replace(0, np.nan)
    df['range_position_60'] = (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min()).replace(0, np.nan)

    # Time-of-day cross-sectional features (NEW for iter 11)
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['date'] = df.index.normalize()

    # For each minute-of-day, compute the rolling 20-day average σ_15 at that exact minute
    # This surfaces intraday seasonality that rolling window features can't see
    print(f'  computing time-of-day cross-sectional features...')
    # Build a working frame with ts as a column (regardless of index name)
    df_xs = df[['minute_of_day', 'date', 'sigma_15', 'sigma_60', 'range', 'volume', 'ret_1']].copy()
    df_xs['_ts'] = df.index
    # Pivot: rows=date, cols=minute, values=sigma_15. Then rolling 20-day mean per minute.
    for col in ['sigma_15', 'sigma_60', 'range', 'volume']:
        pv = df_xs.pivot_table(index='date', columns='minute_of_day', values=col, aggfunc='last')
        # CAUSAL: shift(1) so today's row uses only data up to and including yesterday
        pv_20d = pv.shift(1).rolling(20, min_periods=10).mean()
        pv_20d_std = pv.shift(1).rolling(20, min_periods=10).std()
        # Melt back and merge
        long_mean = pv_20d.stack().rename(f'{col}_tod_mean_20d').reset_index()
        long_std = pv_20d_std.stack().rename(f'{col}_tod_std_20d').reset_index()
        df_xs = df_xs.merge(long_mean, on=['date', 'minute_of_day'], how='left')
        df_xs = df_xs.merge(long_std, on=['date', 'minute_of_day'], how='left')
        df_xs[f'{col}_tod_z'] = (df_xs[col] - df_xs[f'{col}_tod_mean_20d']) / df_xs[f'{col}_tod_std_20d'].replace(0, np.nan)
    df_xs = df_xs.set_index('_ts').sort_index()
    df_xs = df_xs[~df_xs.index.duplicated(keep='last')]
    for c in df_xs.columns:
        if c.endswith('_tod_z') or c.endswith('_tod_mean_20d') or c.endswith('_tod_std_20d'):
            df[c] = df_xs[c].reindex(df.index).values

    # Time
    df['hour_et'] = df.index.hour
    df['minute_of_hour'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day

    # Drop date / minute_of_day from features
    df = df.drop(columns=['date'])
    return df


def label_walk_atr(opens, highs, lows, closes, atrs, win_idx, h_min, tp_mult, sl_mult):
    """Walk forward h_min bars, label win/loss based on ATR-scaled TP/SL.
    Vectorized-ish: loops over win_idx in pure Python (LONG side only)."""
    n = len(opens)
    out_label = np.full(n, -1, dtype=np.int8)
    out_side = np.full(n, -1, dtype=np.int8)
    out_tp = np.full(n, np.nan, dtype=np.float32)
    out_sl = np.full(n, np.nan, dtype=np.float32)

    for i in win_idx:
        if i + h_min >= n: continue
        atr = atrs[i]
        if not np.isfinite(atr) or atr <= 0: continue
        entry = opens[i+1] if i+1 < n else closes[i]
        tp_pts = atr * tp_mult
        sl_pts = atr * sl_mult

        # LONG side — vectorized inner loop using numpy slices
        end = min(i+1+h_min, n)
        sl_long = entry - sl_pts
        tp_long = entry + tp_pts
        seg_lows = lows[i+1:end]
        seg_highs = highs[i+1:end]
        # Find first SL or TP hit
        sl_hits = np.where(seg_lows <= sl_long)[0]
        tp_hits = np.where(seg_highs >= tp_long)[0]
        sl_first = sl_hits[0] if len(sl_hits) else 10**9
        tp_first = tp_hits[0] if len(tp_hits) else 10**9
        if sl_first == 10**9 and tp_first == 10**9:
            continue
        if tp_first < sl_first:
            out_label[i] = 1
        else:
            out_label[i] = 0
        out_side[i] = 0
        out_tp[i] = tp_pts
        out_sl[i] = sl_pts
    return out_label, out_side, out_tp, out_sl


def build_period(name, start, end, tp_mult, sl_mult):
    print(f'\n[{name}] {start}→{end}, TP_mult={tp_mult} SL_mult={sl_mult}')
    t0 = time.time()
    bars = load_period(start, end)
    if bars.empty: return None
    df = compute_features(bars)
    print(f'  features computed: {len(df.columns)} cols, dt={int(time.time()-t0)}s')

    # Window: ET 11-12
    in_win = (df.index.hour >= WIN_HOURS[0]) & (df.index.hour < WIN_HOURS[1])
    win_idx = np.where(in_win)[0]
    print(f'  window bars: {len(win_idx):,}')

    o = df['open'].values; h = df['high'].values; l = df['low'].values; c = df['close'].values
    atr = df['atr_60'].values
    labels, sides, tp_arr, sl_arr = label_walk_atr(o, h, l, c, atr, win_idx, H_MIN_NEW, tp_mult, sl_mult)
    valid = labels != -1
    feat_cols = [col for col in df.columns
                 if col not in ('open','high','low','close','volume','symbol')
                 and pd.api.types.is_numeric_dtype(df[col])]
    out = df.iloc[valid][feat_cols].copy()
    out['label'] = labels[valid]
    out['side'] = sides[valid]
    out['tp_pts'] = tp_arr[valid]
    out['sl_pts'] = sl_arr[valid]
    print(f'  → {len(out):,} samples (LONG only here), labeled. dt={int(time.time()-t0)}s')
    return out


def evaluate_with_circuit_atr(p, y, ts_arr, tp_pts_arr, sl_pts_arr, fired, val_days, daily_dd_cap):
    """Walk forward day-by-day; ATR-scaled per-trade PnL; suspend trading mid-day if cumulative PnL hits -dd_cap."""
    if not fired.any(): return None
    df = pd.DataFrame({
        'ts': pd.to_datetime(ts_arr),
        'fired': fired,
        'win': (y == 1) & fired,
        'tp_pts': tp_pts_arr,
        'sl_pts': sl_pts_arr,
    })
    df = df.sort_values('ts').reset_index(drop=True)
    df['date'] = df['ts'].dt.normalize()
    df['raw_pnl'] = np.where(df['fired'], np.where(df['win'], df['tp_pts'] * PT_USD, -df['sl_pts'] * PT_USD), 0)

    out_pnl = np.zeros(len(df))
    out_fired = np.zeros(len(df), dtype=bool)
    out_win = np.zeros(len(df), dtype=bool)
    for date, grp in df.groupby('date'):
        cum = 0.0; muted = False
        for idx in grp.index:
            if not df.at[idx, 'fired']: continue
            if muted: continue
            pnl = df.at[idx, 'raw_pnl']
            out_pnl[idx] = pnl
            out_fired[idx] = True
            out_win[idx] = bool(df.at[idx, 'win'])
            cum += pnl
            if cum <= -daily_dd_cap:
                muted = True
    df['cb_pnl'] = out_pnl; df['cb_fired'] = out_fired; df['cb_win'] = out_win
    df['month'] = df['ts'].dt.to_period('M')
    df['year'] = df['ts'].dt.year

    monthly_df = df[df['cb_fired']].groupby('month').agg(
        n=('cb_fired', 'sum'), w=('cb_win', 'sum'), pnl=('cb_pnl', 'sum')
    ).reset_index()
    monthly_df['wr'] = monthly_df['w'] / monthly_df['n']

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


def main():
    t0 = time.time()
    # Try a few ATR multiplier brackets
    for tp_mult, sl_mult in [(2.0, 1.0), (2.5, 1.2), (3.0, 1.5), (1.5, 1.0)]:
        breakeven = sl_mult / (tp_mult + sl_mult)
        print(f'\n========== ATR brackets TP×{tp_mult} SL×{sl_mult} (breakeven={breakeven:.1%}, {H_MIN_NEW}-min horizon) ==========')

        all_data = {}
        for name, (s, e) in PERIODS.items():
            df = build_period(name, s, e, tp_mult, sl_mult)
            if df is None or df.empty: continue
            all_data[name] = df

        train = pd.concat([all_data.get('TRAIN_t1', pd.DataFrame()),
                           all_data.get('TRAIN_2025', pd.DataFrame())], axis=0)
        val = all_data['VAL_biden']
        oos = all_data['OOS_2026']

        feat_cols = [c for c in train.columns if c not in ('label','side','tp_pts','sl_pts')]
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
        val_tp = val['tp_pts'].values[valid_v.values]; val_sl = val['sl_pts'].values[valid_v.values]
        oos_tp = oos['tp_pts'].values[valid_o.values]; oos_sl = oos['sl_pts'].values[valid_o.values]
        val_days = pd.to_datetime(val_ts).normalize().nunique()
        oos_days = pd.to_datetime(oos_ts).normalize().nunique()
        print(f'  TRAIN={len(Xt):,}, VAL={len(Xv):,} ({val_days}d), OOS={len(Xo):,} ({oos_days}d)')
        print(f'  baseline WR: TRAIN={yt.mean()*100:.1f}%, VAL={yv.mean()*100:.1f}%, OOS={yo.mean()*100:.1f}%')

        # F-stat top features
        Xt_clean = Xt.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
        F, _ = f_classif(Xt_clean.values, yt)
        fa = pd.DataFrame({'f': feat_cols, 'F': F}).sort_values('F', ascending=False)
        print('\n  Top 10 F-stat features:')
        for _, r in fa.head(10).iterrows():
            print(f'    {r["f"]:<35s} F={r["F"]:.1f}')
        top60 = fa.head(60)['f'].tolist()

        Xt_hgb = Xt[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
        Xv_hgb = Xv[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
        Xo_hgb = Xo[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
        hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400,
                                             early_stopping=True, n_iter_no_change=15, random_state=42)
        hgb.fit(Xt_hgb, yt)
        pv = hgb.predict_proba(Xv_hgb)[:, 1]
        po = hgb.predict_proba(Xo_hgb)[:, 1]
        auc_v = roc_auc_score(yv, pv)
        auc_o = roc_auc_score(yo, po)
        print(f'\n  HGB AUC: VAL={auc_v:.4f}, OOS={auc_o:.4f}')

        # Threshold sweep on VAL + OOS at multiple DD caps
        print(f'\n  ===== VAL sweep =====')
        print(f'  {"thr":>5s} {"cap":>5s} {"n":>5s} {"/d":>5s} {"WR%":>5s} {"avg":>7s} {"PnL":>10s} {"yrs+":>5s} {"maxDD":>10s}')
        best_v = None
        for thr in np.arange(0.95, 0.55, -0.04):
            for cap in [100, 150, 200]:
                fired = pv >= thr
                if not fired.any(): continue
                stats = evaluate_with_circuit_atr(pv, yv, val_ts, val_tp, val_sl, fired, val_days, cap)
                if stats is None: continue
                yrs_pos = sum(1 for yy in stats['yearly'] if yy['pnl'] > 0 and yy['wr'] > 0.5 and yy['dd'] >= -900)
                max_dd = min((yy['dd'] for yy in stats['yearly']), default=0)
                if stats['per_day'] >= 0.5 and stats['avg'] >= 30:
                    print(f'  {thr:.2f} ${cap:>3d} {stats["n"]:>5d} {stats["per_day"]:>4.2f} {stats["wr"]*100:>4.1f}% ${stats["avg"]:>+5.0f} ${stats["pnl"]:>+8.0f} {yrs_pos:>2d}/{len(stats["yearly"])} ${max_dd:>+8.0f}')
                    if (best_v is None or stats['pnl'] > best_v['pnl']) and yrs_pos == len(stats['yearly']) and stats['per_day'] >= 1:
                        best_v = {**stats, 'thr': float(thr), 'cap': cap, 'yrs_pos': yrs_pos, 'max_dd': max_dd}

        print(f'\n  ===== OOS sweep =====')
        print(f'  {"thr":>5s} {"cap":>5s} {"n":>5s} {"/d":>5s} {"WR%":>5s} {"avg":>7s} {"PnL":>10s} {"DD":>10s}')
        for thr in np.arange(0.95, 0.55, -0.04):
            for cap in [100, 150, 200]:
                fired = po >= thr
                if not fired.any(): continue
                stats = evaluate_with_circuit_atr(po, yo, oos_ts, oos_tp, oos_sl, fired, oos_days, cap)
                if stats is None: continue
                yr_dd = stats['yearly'][0]['dd'] if stats['yearly'] else 0
                if stats['per_day'] >= 0.5 and stats['avg'] >= 30:
                    print(f'  {thr:.2f} ${cap:>3d} {stats["n"]:>5d} {stats["per_day"]:>4.2f} {stats["wr"]*100:>4.1f}% ${stats["avg"]:>+5.0f} ${stats["pnl"]:>+8.0f} ${yr_dd:>+8.0f}')

        if best_v:
            print(f'\n  🟢 VAL passing config: thr={best_v["thr"]:.2f} cap=${best_v["cap"]}, '
                  f'{best_v["per_day"]:.2f}/d, WR={best_v["wr"]*100:.1f}%, avg=${best_v["avg"]:.0f}, PnL=${best_v["pnl"]:+,.0f}, '
                  f'yrs+={best_v["yrs_pos"]}/{len(best_v["yearly"])}, maxDD=${best_v["max_dd"]:+.0f}')
            joblib.dump({
                'tp_mult': tp_mult, 'sl_mult': sl_mult, 'h_min': H_MIN_NEW,
                'threshold': best_v['thr'], 'daily_dd_cap': best_v['cap'],
                'val_auc': float(auc_v), 'oos_auc': float(auc_o),
                'hgb': hgb, 'features': top60,
            }, OUT / f'iter11_atr_{tp_mult}_{sl_mult}_model.pkl')

    print(f'\nWall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
