"""Iter 5: ET 11-12 window (PT 8-9) with wider R:R brackets to hit $60 avg/trade.

Hard constraints:
  - Annual DD ≤ $900
  - Positive WR + PnL every year AND every month
  - 2-4 trades/day average
  - **NEW: avg PnL/trade ≥ $60**

Window: ET 11-12 only (drops ET 9-10, which gave us flat AUC in iter 1-4).
Brackets: try TP/SL = (15,10), (25,10), (30,10), (12,8), (20,12), (25,15) — all R>1.
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
BAR = ROOT / 'es_master_outrights-2.parquet'
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter5'
OUT.mkdir(parents=True, exist_ok=True)

ET = 'US/Eastern'
PT_USD = 5.0
H_MIN = 120  # 2-hour walk-forward
WIN_HOURS = (11, 12)  # ET hours 11 and 12 only

PERIODS = {
    'TRAIN_t1':   ('2017-01-20', '2021-01-20'),
    'TRAIN_2025': ('2025-01-20', '2025-12-31'),
    'VAL_biden':  ('2021-01-21', '2025-01-19'),
    'OOS_2026':   ('2026-01-01', '2026-04-30'),
}

# Bracket geometries to try (TP > SL for R:R > 1)
BRACKETS = [(15, 10), (25, 10), (30, 10), (12, 8), (20, 12), (25, 15), (40, 15)]


def load_bars_period(start, end):
    bars = pd.read_parquet(BAR)
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert(ET)
    bars = bars.loc[start:end]
    bars = bars.reset_index().rename(columns={'index': 'ts'})
    if 'ts' not in bars.columns:
        bars = bars.rename(columns={bars.columns[0]: 'ts'})
    bars['_date'] = bars['ts'].dt.date
    daily_vol = bars.groupby(['_date', 'symbol'])['volume'].sum().reset_index()
    dom = daily_vol.loc[daily_vol.groupby('_date')['volume'].idxmax(), ['_date', 'symbol']]
    dom.columns = ['_date', 'dom']
    bars = bars.merge(dom, on='_date', how='left')
    bars = bars[bars['symbol'] == bars['dom']].copy()
    bars = bars.drop(columns=['_date', 'dom', 'symbol']).set_index('ts').sort_index()
    return bars


def compute_sigma_features(bars):
    """Same 82 features as before but operating on ET 11-12 window's surrounding bars."""
    df = bars.copy()
    df['ret_1'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_5'] = np.log(df['close'] / df['close'].shift(5))
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    for w in [5, 10, 15, 30, 60, 120, 240]:
        df[f'sigma_{w}'] = df['ret_1'].rolling(w).std() * np.sqrt(w)
    df['sigma_5_log'] = np.log1p(df['sigma_5'].clip(lower=1e-9))
    df['sigma_15_log'] = np.log1p(df['sigma_15'].clip(lower=1e-9))
    df['sigma_60_log'] = np.log1p(df['sigma_60'].clip(lower=1e-9))
    df['sigma_intraday_running'] = df['ret_1'].rolling(60).std()
    df['sigma_open_to_now'] = df['close'].rolling(60).std() / df['close'].shift(60)
    s5,s15,s30,s60,s120,s240 = df['sigma_5'], df['sigma_15'], df['sigma_30'], df['sigma_60'], df['sigma_120'], df['sigma_240']
    df['sigma_ratio_5_60'] = s5 / s60.replace(0, np.nan)
    df['sigma_ratio_15_60'] = s15 / s60.replace(0, np.nan)
    df['sigma_ratio_15_240'] = s15 / s240.replace(0, np.nan)
    df['sigma_ratio_60_240'] = s60 / s240.replace(0, np.nan)
    df['sigma_ratio_5_15'] = s5 / s15.replace(0, np.nan)
    df['sigma_ratio_30_120'] = s30 / s120.replace(0, np.nan)
    df['sigma_ratio_intraday_240'] = df['sigma_intraday_running'] / s240.replace(0, np.nan)
    df['sigma_ratio_15_60_log'] = df['sigma_15_log'] - df['sigma_60_log']
    df['sigma_ratio_5_60_log'] = df['sigma_5_log'] - df['sigma_60_log']
    df['sigma_5_chg_5'] = s5.pct_change(5)
    df['sigma_15_pctile_60'] = s15.rolling(60).rank(pct=True)
    df['sigma_15_pctile_240'] = s15.rolling(240).rank(pct=True)
    df['sigma_15_pctile_390'] = s15.rolling(390).rank(pct=True)
    df['sigma_60_pctile_240'] = s60.rolling(240).rank(pct=True)
    df['sigma_60_pctile_390'] = s60.rolling(390).rank(pct=True)
    df['sigma_15_pctile_hourly_20d'] = s15.rolling(240*20).rank(pct=True)
    df['sigma_60_pctile_hourly_20d'] = s60.rolling(240*20).rank(pct=True)
    df['sigma_intraday_pctile_20d'] = df['sigma_intraday_running'].rolling(240*20).rank(pct=True)
    df['sigma_of_sigma_15_60'] = s15.rolling(60).std()
    df['sigma_of_sigma_15_240'] = s15.rolling(240).std()
    df['sigma_of_sigma_60_240'] = s60.rolling(240).std()
    df['delta_sigma_15_5'] = s15 - s15.shift(5)
    df['delta_sigma_60_15'] = s60 - s60.shift(15)
    df['sigma_acceleration'] = (s5 - s5.shift(5)) / s5.shift(5).replace(0, np.nan)
    df['sigma_skew_30'] = df['ret_1'].rolling(30).skew()
    df['sigma_skew_120'] = df['ret_1'].rolling(120).skew()
    for w in [15, 60, 240]:
        m = df['close'].rolling(w).mean()
        s = df['close'].rolling(w).std()
        df[f'z_close_{w}'] = (df['close'] - m) / s.replace(0, np.nan)
    df['z_high_15']  = (df['high'] - df['close'].rolling(15).mean()) / df['close'].rolling(15).std().replace(0, np.nan)
    df['z_low_15']   = (df['low']  - df['close'].rolling(15).mean()) / df['close'].rolling(15).std().replace(0, np.nan)
    rng = df['range']
    df['z_range_15'] = (rng - rng.rolling(15).mean()) / rng.rolling(15).std().replace(0, np.nan)
    df['z_range_60'] = (rng - rng.rolling(60).mean()) / rng.rolling(60).std().replace(0, np.nan)
    df['z_body_15']  = (df['body'] - df['body'].rolling(15).mean()) / df['body'].rolling(15).std().replace(0, np.nan)
    df['z_upper_wick_15'] = (df['upper_wick'] - df['upper_wick'].rolling(15).mean()) / df['upper_wick'].rolling(15).std().replace(0, np.nan)
    df['z_lower_wick_15'] = (df['lower_wick'] - df['lower_wick'].rolling(15).mean()) / df['lower_wick'].rolling(15).std().replace(0, np.nan)
    vol = df['volume']
    df['z_volume_15'] = (vol - vol.rolling(15).mean()) / vol.rolling(15).std().replace(0, np.nan)
    df['z_volume_60'] = (vol - vol.rolling(60).mean()) / vol.rolling(60).std().replace(0, np.nan)
    df['z_return_1'] = (df['ret_1'] - df['ret_1'].rolling(60).mean()) / df['ret_1'].rolling(60).std().replace(0, np.nan)
    df['z_return_5'] = (df['ret_5'] - df['ret_5'].rolling(60).mean()) / df['ret_5'].rolling(60).std().replace(0, np.nan)
    df['skew_return_60'] = df['ret_1'].rolling(60).skew()
    df['skew_return_240'] = df['ret_1'].rolling(240).skew()
    df['kurt_return_60'] = df['ret_1'].rolling(60).kurt()
    df['kurt_return_240'] = df['ret_1'].rolling(240).kurt()
    abs_ret = df['ret_1'].abs()
    df['efficiency_60'] = (df['close'] - df['close'].shift(60)).abs() / (abs_ret.rolling(60).sum() * df['close'].shift(60)).replace(0, np.nan)
    df['efficiency_240'] = (df['close'] - df['close'].shift(240)).abs() / (abs_ret.rolling(240).sum() * df['close'].shift(240)).replace(0, np.nan)
    df['hour_et'] = df.index.hour
    df['minute_of_hour'] = df.index.minute
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_et'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_et'] / 24)
    df['day_of_week'] = df.index.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    df['week_of_year'] = df.index.isocalendar().week.values
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['prev_bar_color'] = np.sign(df['close'].shift(1) - df['open'].shift(1))
    df['prev_2_bar_color'] = np.sign(df['close'].shift(2) - df['open'].shift(2))
    df['ret_15'] = np.log(df['close'] / df['close'].shift(15))
    df['ret_60'] = np.log(df['close'] / df['close'].shift(60))
    df['body_share'] = df['body'] / df['range'].replace(0, np.nan)
    df['upper_wick_share'] = df['upper_wick'] / df['range'].replace(0, np.nan)
    df['lower_wick_share'] = df['lower_wick'] / df['range'].replace(0, np.nan)
    return df


def label_walk_forward(df_full, win_idx, tp, sl, h_min):
    """For each bar in win_idx, walk forward H bars and label LONG/SHORT outcomes."""
    high = df_full['high'].values
    low = df_full['low'].values
    open_ = df_full['open'].values
    close = df_full['close'].values
    n = len(df_full)
    long_lbl = np.full(len(win_idx), -1, dtype=np.int8)
    short_lbl = np.full(len(win_idx), -1, dtype=np.int8)
    for ii, i in enumerate(win_idx):
        if i + 1 + h_min >= n: continue
        entry = open_[i + 1]
        tp_long = entry + tp; sl_long = entry - sl
        tp_short = entry - tp; sl_short = entry + sl
        l_long = -1; l_short = -1
        for j in range(i + 1, min(i + 1 + h_min, n)):
            hh = high[j]; ll = low[j]
            if l_long == -1:
                if ll <= sl_long: l_long = 0
                elif hh >= tp_long: l_long = 1
            if l_short == -1:
                if hh >= sl_short: l_short = 0
                elif ll <= tp_short: l_short = 1
            if l_long != -1 and l_short != -1: break
        if l_long == -1:
            last = close[min(i + h_min, n - 1)]
            l_long = 1 if last > entry else 0
        if l_short == -1:
            last = close[min(i + h_min, n - 1)]
            l_short = 1 if last < entry else 0
        long_lbl[ii] = l_long
        short_lbl[ii] = l_short
    return long_lbl, short_lbl


def build_period(name, start, end, tp, sl):
    print(f'\n[{name}] {start} → {end}, TP={tp}/SL={sl}')
    t0 = time.time()
    bars = load_bars_period(start, end)
    if bars.empty: return None
    feats = compute_sigma_features(bars)
    h = feats.index.hour
    dow = feats.index.dayofweek
    win_mask = np.isin(h, WIN_HOURS) & (dow < 5)
    win_idx = np.where(win_mask)[0]
    print(f'  bars={len(feats):,}, in-window={len(win_idx):,}')
    long_lbl, short_lbl = label_walk_forward(feats, win_idx, tp, sl, H_MIN)
    feature_cols = [c for c in feats.columns if c not in ['open','high','low','close','volume','range','body','upper_wick','lower_wick']]
    feats_win = feats.iloc[win_idx][feature_cols].copy()
    feats_win['label_long'] = long_lbl
    feats_win['label_short'] = short_lbl
    long_df = feats_win.copy(); long_df['side'] = 0; long_df['label'] = long_df['label_long']
    short_df = feats_win.copy(); short_df['side'] = 1; short_df['label'] = short_df['label_short']
    out = pd.concat([long_df, short_df], axis=0)
    out = out[out['label'] >= 0].copy()
    out = out.drop(columns=['label_long', 'label_short'])
    print(f'  → {len(out):,} samples (long+short), dt={time.time()-t0:.1f}s')
    return out


def evaluate(p_win, y, ts_arr, fired, tp, sl, val_days):
    """Compute monthly + yearly stats given fired mask + outcomes."""
    if not fired.any():
        return None
    df = pd.DataFrame({'ts': ts_arr, 'fired': fired, 'win': (y == 1) & fired})
    ts_dt = pd.to_datetime(df['ts'])
    df['month'] = ts_dt.dt.to_period('M')
    df['year'] = ts_dt.dt.year
    df['pnl'] = np.where(fired, np.where(df['win'], tp * PT_USD, -sl * PT_USD), 0)

    monthly = df[df['fired']].groupby('month').agg(
        n=('fired', 'sum'), w=('win', 'sum'), pnl=('pnl', 'sum')
    ).reset_index()
    monthly['wr'] = monthly['w'] / monthly['n']
    monthly['avg'] = monthly['pnl'] / monthly['n']

    yearly_records = []
    for y_, grp in df[df['fired']].groupby('year'):
        ord_pnl = grp.sort_values('ts')['pnl'].cumsum()
        peak = ord_pnl.cummax()
        dd = float((ord_pnl - peak).min())
        yearly_records.append({'year': y_, 'n': int(grp['fired'].sum()),
                               'w': int(grp['win'].sum()), 'pnl': float(grp['pnl'].sum()),
                               'dd': dd})
    yearly = pd.DataFrame(yearly_records)
    if len(yearly): yearly['wr'] = yearly['w'] / yearly['n']

    n_total = int(monthly['n'].sum())
    pnl_total = float(monthly['pnl'].sum())
    wr_total = float(monthly['w'].sum() / n_total) if n_total else 0
    avg_total = pnl_total / n_total if n_total else 0
    per_day = n_total / val_days

    return {
        'n': n_total, 'wr': wr_total, 'pnl': pnl_total, 'avg': avg_total,
        'per_day': per_day, 'monthly': monthly.to_dict('records'),
        'yearly': yearly.to_dict('records'),
    }


def check(stats, tp, sl):
    fails = []
    if not stats: return False, ['no fires']
    if stats['per_day'] < 2: fails.append(f"avg{stats['per_day']:.2f}/d<2")
    if stats['per_day'] > 4: fails.append(f"avg{stats['per_day']:.2f}/d>4")
    if stats['avg'] < 60: fails.append(f"avg${stats['avg']:.0f}<$60")
    for y in stats['yearly']:
        if y['dd'] < -900: fails.append(f"YR{y['year']}-DD${y['dd']:.0f}")
        if y['pnl'] <= 0: fails.append(f"YR{y['year']}-PnL${y['pnl']:.0f}")
        if y['wr'] <= 0.5: fails.append(f"YR{y['year']}-WR{y['wr']:.2%}")
    neg = [m for m in stats['monthly'] if m['pnl'] <= 0]
    if neg: fails.append(f"{len(neg)}m-NEGPnL")
    lwr = [m for m in stats['monthly'] if m['wr'] <= 0.5]
    if lwr: fails.append(f"{len(lwr)}m-WR<50")
    return len(fails) == 0, fails


def run_for_brackets(tp, sl):
    print(f'\n========== BRACKETS: TP={tp} / SL={sl} (R={tp/sl:.2f}) ==========')
    breakeven = sl / (tp + sl)
    print(f'breakeven WR = {breakeven*100:.1f}%')

    # Build all 4 periods
    all_data = {}
    for name, (s, e) in PERIODS.items():
        df = build_period(name, s, e, tp, sl)
        if df is None or df.empty: continue
        all_data[name] = df
    if 'TRAIN_t1' not in all_data or 'VAL_biden' not in all_data:
        return None

    train = pd.concat([all_data.get('TRAIN_t1', pd.DataFrame()),
                       all_data.get('TRAIN_2025', pd.DataFrame())], axis=0)
    val = all_data['VAL_biden']
    feat_cols = [c for c in train.columns if c != 'label']

    Xt = train[feat_cols].astype(np.float32)
    yt = train['label'].astype(np.int8).values
    Xv = val[feat_cols].astype(np.float32)
    yv = val['label'].astype(np.int8).values
    valid_t = ~Xt.isna().any(axis=1)
    valid_v = ~Xv.isna().any(axis=1)
    Xt = Xt[valid_t].reset_index(drop=True); yt = yt[valid_t.values]
    Xv = Xv[valid_v].reset_index(drop=True); yv = yv[valid_v.values]
    val_ts = val.index[valid_v.values]
    val_days = pd.to_datetime(val_ts).normalize().nunique()

    # Show baseline WR per side
    long_mask_v = (Xv['side'] == 0).values
    short_mask_v = (Xv['side'] == 1).values
    print(f'  Baseline VAL WR: LONG={yv[long_mask_v].mean()*100:.1f}%  SHORT={yv[short_mask_v].mean()*100:.1f}%  breakeven={breakeven*100:.1f}%')

    # LONG-only HGB
    side_t = (Xt['side'] == 0).values
    Xt_L = Xt[side_t].reset_index(drop=True); yt_L = yt[side_t]
    Xv_L = Xv[long_mask_v].reset_index(drop=True); yv_L = yv[long_mask_v]
    val_ts_L = val_ts[long_mask_v]

    if len(Xt_L) < 5000:
        return None

    F, pvals = f_classif(Xt_L.fillna(0).values, yt_L)
    fa = pd.DataFrame({'f': feat_cols, 'F': F, 'p': pvals}).sort_values('F', ascending=False)
    sig = fa.head(40)['f'].tolist()
    sigma_feats = [f for f in sig if 'sigma' in f or 'z_' in f][:15]

    Xt_sig = Xt_L[sigma_feats].fillna(0).values
    Xv_sig = Xv_L[sigma_feats].fillna(0).values
    sc = StandardScaler(); Xt_s = sc.fit_transform(Xt_sig); Xv_s = sc.transform(Xv_sig)
    pca = PCA(n_components=8, random_state=42); Xt_p = pca.fit_transform(Xt_s); Xv_p = pca.transform(Xv_s)

    best_K = None
    for K in [4, 6, 8]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        ct = km.fit_predict(Xt_p); cv = km.predict(Xv_p)
        df_t = pd.DataFrame({'c': ct, 'y': yt_L}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        df_v = pd.DataFrame({'c': cv, 'y': yv_L}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        good = [c for c in df_t.index if df_t.loc[c, 'wr'] > breakeven + 0.02 and df_t.loc[c, 'n'] > 1000
                and c in df_v.index and df_v.loc[c, 'wr'] > breakeven + 0.01]
        if good and (not best_K or len(good) > len(best_K['good'])):
            best_K = {'K': K, 'good': good, 'cv': cv}

    cluster_mask_v = np.isin(best_K['cv'], best_K['good']) if best_K else np.ones(len(Xv_L), dtype=bool)
    print(f'  cluster mask: {cluster_mask_v.sum():,} of {len(Xv_L):,}')

    Xt_full = Xt_L[sig].fillna(0).values.astype(np.float32)
    Xv_full = Xv_L[sig].fillna(0).values.astype(np.float32)
    hgb = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=300,
                                          early_stopping=True, n_iter_no_change=15, random_state=42)
    hgb.fit(Xt_full, yt_L)
    pv = hgb.predict_proba(Xv_full)[:, 1]
    try: auc = roc_auc_score(yv_L, pv)
    except: auc = 0.5

    print(f'  Val AUC: {auc:.4f}')

    # Threshold sweep
    print(f'\n  Threshold sweep (LONG, K-means cluster gate active):')
    print(f'  {"thr":>5s} {"n":>5s} {"/d":>5s} {"WR%":>5s} {"avg":>7s} {"PnL":>9s} {"yrs+":>5s} {"mos+":>5s} {"PASS":>5s}')

    candidates = []
    for thr in np.arange(0.99, 0.30, -0.01):
        fired = (pv >= thr) & cluster_mask_v
        n = int(fired.sum())
        if n == 0: continue
        stats = evaluate(pv, yv_L, val_ts_L, fired, tp, sl, val_days)
        if stats is None: continue
        passed, fails = check(stats, tp, sl)
        yrs_pos = sum(1 for y in stats['yearly'] if y['pnl'] > 0 and y['wr'] > 0.5 and y['dd'] >= -900)
        mos_pos = sum(1 for m in stats['monthly'] if m['pnl'] > 0 and m['wr'] > 0.5)
        candidates.append({**stats, 'thr': thr, 'passed': passed, 'fails': fails,
                           'yrs_pos': f'{yrs_pos}/{len(stats["yearly"])}',
                           'mos_pos': f'{mos_pos}/{len(stats["monthly"])}'})

    # Print sparse subset (every 5th)
    for c in candidates[::5]:
        flag = '✅' if c['passed'] else ' '
        print(f'  {c["thr"]:.3f} {c["n"]:>5d} {c["per_day"]:>4.2f} {c["wr"]*100:>4.1f}% ${c["avg"]:>+5.0f} ${c["pnl"]:>+8.0f} {c["yrs_pos"]:>5s} {c["mos_pos"]:>5s} {flag}')

    passing = [c for c in candidates if c['passed']]
    if passing:
        best = max(passing, key=lambda c: c['n'])
        print(f'\n🟢 PASSING: thr={best["thr"]:.3f}')
        print(f'   {best["per_day"]:.2f}/d, WR={best["wr"]*100:.1f}%, avg=${best["avg"]:.2f}, PnL=+${best["pnl"]:.0f}')
        for y in best['yearly']:
            print(f'   {int(y["year"])}: n={int(y["n"])} WR={y["wr"]*100:.1f}% PnL=${y["pnl"]:+.0f} DD=${y["dd"]:+.0f}')
        return {'tp': tp, 'sl': sl, 'best': best, 'auc': auc}
    else:
        # Closest by PnL among rate-ok
        rate_ok = [c for c in candidates if 2 <= c['per_day'] <= 4]
        if rate_ok:
            avg_ok = [c for c in rate_ok if c['avg'] >= 60]
            if avg_ok:
                best_close = max(avg_ok, key=lambda c: c['pnl'])
            else:
                best_close = max(rate_ok, key=lambda c: c['avg'])
            print(f'\n  closest(rate-ok): thr={best_close["thr"]:.3f}, {best_close["per_day"]:.2f}/d, WR={best_close["wr"]*100:.1f}%, avg=${best_close["avg"]:.0f}, PnL=${best_close["pnl"]:+.0f}, fails={len(best_close["fails"])}')
        return None


def main():
    t0 = time.time()
    results = []
    for tp, sl in BRACKETS:
        r = run_for_brackets(tp, sl)
        if r:
            results.append(r)
    print(f'\n\n========== SUMMARY ==========')
    if results:
        print(f'PASSING configurations:')
        for r in results:
            b = r['best']
            print(f'  TP={r["tp"]}/SL={r["sl"]}: thr={b["thr"]:.3f}, {b["per_day"]:.2f}/d, WR={b["wr"]*100:.1f}%, avg=${b["avg"]:.0f}, PnL=${b["pnl"]:+.0f}')
        # Save best
        best = max(results, key=lambda r: r['best']['pnl'])
        with open(OUT / 'best_config.json', 'w') as f:
            json.dump({'tp': best['tp'], 'sl': best['sl'], 'best': best['best'], 'auc': best['auc']}, f, indent=2, default=str)
        print(f'\n→ best by PnL saved')
    else:
        print('NO PASSING CONFIG across any bracket geometry — surface dead-end')
    print(f'\nWall: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
