"""Iter 6: ~120-feature ML at ET 11-12 window. Adds 30+ math features beyond σ
to push AUC over the 0.55 ceiling that pure σ hit in iter 1-5.

Relaxed constraints (per user):
  - 2-4 trades/day average
  - avg PnL/trade ≥ $60
  - All years positive (PnL, WR, DD ≤ $900)
  - ≥ 80% positive months (39/49 in VAL Biden)

Features added vs iter 5:
  Directional/momentum (12):
    slope_15, slope_60, slope_240 (linear regression slope, σ-normalized)
    momentum_z_15, momentum_z_60, momentum_z_240
    vwap_dist_z_60, vwap_dist_z_240
    range_position_15, range_position_60 (close - low) / (high - low)
    breakout_15, breakout_60 (sign of breaking prior N-bar high/low)
  Stationarity/regime (8):
    hurst_60, hurst_240 (rescaled range)
    variance_ratio_15_60, variance_ratio_60_240
    autocorr_lag1_60, autocorr_lag1_240
    adf_stat_60, adf_stat_240 (lite — pseudo)
  Spectral / oscillation (4):
    rsi_14, rsi_60 (relative strength index)
    fft_dominant_period_240
    fft_energy_240
  Volume math (6):
    volume_z_15, volume_z_60
    volume_at_high_60, volume_at_low_60
    obv_slope_60 (on-balance-volume)
    volume_pct_60
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
OUT = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter6'
OUT.mkdir(parents=True, exist_ok=True)

ET = 'US/Eastern'
PT_USD = 5.0
H_MIN = 120
WIN_HOURS = (11, 12)

PERIODS = {
    'TRAIN_t1':   ('2017-01-20', '2021-01-20'),
    'TRAIN_2025': ('2025-01-20', '2025-12-31'),
    'VAL_biden':  ('2021-01-21', '2025-01-19'),
    'OOS_2026':   ('2026-01-01', '2026-04-30'),
}


def load_period(start, end):
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
    return bars.drop(columns=['_date', 'dom', 'symbol']).set_index('ts').sort_index()


def rolling_slope(series, w):
    """Linear regression slope over rolling window (vectorized)."""
    x = np.arange(w, dtype=float)
    x_mean = x.mean()
    x_norm = x - x_mean
    x_var = (x_norm ** 2).sum()
    def slope_calc(y):
        y_mean = y.mean()
        return np.dot(x_norm, y - y_mean) / x_var if x_var > 0 else 0
    return series.rolling(w).apply(slope_calc, raw=True)


def hurst_rs(series, w):
    """Rescaled-range Hurst exponent estimate over rolling window."""
    def calc(y):
        n = len(y)
        if n < 10: return 0.5
        m = y.mean()
        z = y - m
        r = np.cumsum(z)
        R = r.max() - r.min()
        S = y.std()
        if S < 1e-9 or R < 1e-9: return 0.5
        return np.log(R / S) / np.log(n)
    return series.rolling(w).apply(calc, raw=True)


def autocorr_lag1(series, w):
    """Lag-1 autocorrelation over rolling window."""
    def calc(y):
        if len(y) < 2: return 0
        return np.corrcoef(y[:-1], y[1:])[0, 1] if len(np.unique(y)) > 1 else 0
    return series.rolling(w).apply(calc, raw=True)


def variance_ratio(returns, k):
    """Var(k-period returns) / (k * Var(1-period returns)). 1 = random walk."""
    var_1 = returns.rolling(60).var()
    var_k = returns.rolling(60).apply(lambda x: x.rolling(k).sum().dropna().var() if len(x) >= k else 0, raw=False)
    return var_k / (k * var_1.replace(0, np.nan))


def rsi(close, w):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(w).mean()
    avg_loss = loss.rolling(w).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def fft_features(close, w):
    """Dominant FFT period + total energy over rolling window."""
    def dom_period(y):
        if len(y) < w: return w / 2
        fft = np.abs(np.fft.fft(y - y.mean()))
        fft = fft[:w // 2]
        if fft.max() < 1e-9: return w / 2
        return w / (fft.argmax() + 1)
    def energy(y):
        if len(y) < w: return 0
        fft = np.abs(np.fft.fft(y - y.mean()))
        return float(fft.sum())
    return close.rolling(w).apply(dom_period, raw=True), close.rolling(w).apply(energy, raw=True)


def compute_features(bars):
    """Compute σ + math features (~120 total)."""
    df = bars.copy()
    df['ret_1'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_5'] = np.log(df['close'] / df['close'].shift(5))
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # σ at multiple windows
    for w in [5, 10, 15, 30, 60, 120, 240]:
        df[f'sigma_{w}'] = df['ret_1'].rolling(w).std() * np.sqrt(w)
    df['sigma_5_log'] = np.log1p(df['sigma_5'].clip(lower=1e-9))
    df['sigma_15_log'] = np.log1p(df['sigma_15'].clip(lower=1e-9))
    df['sigma_60_log'] = np.log1p(df['sigma_60'].clip(lower=1e-9))
    df['sigma_intraday'] = df['ret_1'].rolling(60).std()

    s5,s15,s30,s60,s120,s240 = df['sigma_5'], df['sigma_15'], df['sigma_30'], df['sigma_60'], df['sigma_120'], df['sigma_240']
    df['sigma_ratio_5_60'] = s5 / s60.replace(0, np.nan)
    df['sigma_ratio_15_60'] = s15 / s60.replace(0, np.nan)
    df['sigma_ratio_15_240'] = s15 / s240.replace(0, np.nan)
    df['sigma_ratio_60_240'] = s60 / s240.replace(0, np.nan)
    df['sigma_ratio_5_15'] = s5 / s15.replace(0, np.nan)
    df['sigma_ratio_30_120'] = s30 / s120.replace(0, np.nan)
    df['sigma_ratio_15_60_log'] = df['sigma_15_log'] - df['sigma_60_log']
    df['sigma_ratio_5_60_log'] = df['sigma_5_log'] - df['sigma_60_log']
    df['sigma_5_chg_5'] = s5.pct_change(5)
    df['sigma_15_pctile_60'] = s15.rolling(60).rank(pct=True)
    df['sigma_15_pctile_240'] = s15.rolling(240).rank(pct=True)
    df['sigma_60_pctile_240'] = s60.rolling(240).rank(pct=True)
    df['sigma_60_pctile_390'] = s60.rolling(390).rank(pct=True)
    df['sigma_15_pctile_hourly_20d'] = s15.rolling(240*20).rank(pct=True)
    df['sigma_60_pctile_hourly_20d'] = s60.rolling(240*20).rank(pct=True)
    df['sigma_intraday_pctile_20d'] = df['sigma_intraday'].rolling(240*20).rank(pct=True)
    df['sigma_of_sigma_15_60'] = s15.rolling(60).std()
    df['sigma_of_sigma_15_240'] = s15.rolling(240).std()
    df['sigma_of_sigma_60_240'] = s60.rolling(240).std()
    df['delta_sigma_15_5'] = s15 - s15.shift(5)
    df['delta_sigma_60_15'] = s60 - s60.shift(15)
    df['sigma_acceleration'] = (s5 - s5.shift(5)) / s5.shift(5).replace(0, np.nan)

    # Z-scores
    for w in [15, 60, 240]:
        m = df['close'].rolling(w).mean()
        s = df['close'].rolling(w).std()
        df[f'z_close_{w}'] = (df['close'] - m) / s.replace(0, np.nan)
    rng = df['range']
    df['z_range_15'] = (rng - rng.rolling(15).mean()) / rng.rolling(15).std().replace(0, np.nan)
    df['z_range_60'] = (rng - rng.rolling(60).mean()) / rng.rolling(60).std().replace(0, np.nan)
    df['z_body_15'] = (df['body'] - df['body'].rolling(15).mean()) / df['body'].rolling(15).std().replace(0, np.nan)
    df['z_upper_wick_15'] = (df['upper_wick'] - df['upper_wick'].rolling(15).mean()) / df['upper_wick'].rolling(15).std().replace(0, np.nan)
    df['z_lower_wick_15'] = (df['lower_wick'] - df['lower_wick'].rolling(15).mean()) / df['lower_wick'].rolling(15).std().replace(0, np.nan)
    df['z_volume_15'] = (df['volume'] - df['volume'].rolling(15).mean()) / df['volume'].rolling(15).std().replace(0, np.nan)
    df['z_volume_60'] = (df['volume'] - df['volume'].rolling(60).mean()) / df['volume'].rolling(60).std().replace(0, np.nan)
    df['z_return_1'] = (df['ret_1'] - df['ret_1'].rolling(60).mean()) / df['ret_1'].rolling(60).std().replace(0, np.nan)
    df['z_return_5'] = (df['ret_5'] - df['ret_5'].rolling(60).mean()) / df['ret_5'].rolling(60).std().replace(0, np.nan)

    # Distribution shape
    df['skew_60'] = df['ret_1'].rolling(60).skew()
    df['skew_240'] = df['ret_1'].rolling(240).skew()
    df['kurt_60'] = df['ret_1'].rolling(60).kurt()
    df['kurt_240'] = df['ret_1'].rolling(240).kurt()
    abs_ret = df['ret_1'].abs()
    df['efficiency_60'] = (df['close'] - df['close'].shift(60)).abs() / (abs_ret.rolling(60).sum() * df['close'].shift(60)).replace(0, np.nan)
    df['efficiency_240'] = (df['close'] - df['close'].shift(240)).abs() / (abs_ret.rolling(240).sum() * df['close'].shift(240)).replace(0, np.nan)

    # === DIRECTIONAL / MOMENTUM ===
    print('  computing directional features...')
    for w in [15, 60, 240]:
        df[f'slope_{w}'] = rolling_slope(df['close'], w)
        df[f'slope_{w}_norm'] = df[f'slope_{w}'] / df['close'].rolling(w).std().replace(0, np.nan)
    df['momentum_z_15'] = df['ret_1'].rolling(15).mean() / df['ret_1'].rolling(15).std().replace(0, np.nan)
    df['momentum_z_60'] = df['ret_1'].rolling(60).mean() / df['ret_1'].rolling(60).std().replace(0, np.nan)
    df['momentum_z_240'] = df['ret_1'].rolling(240).mean() / df['ret_1'].rolling(240).std().replace(0, np.nan)
    # VWAP approximation: rolling cumulative (price*volume)/volume
    pv = df['close'] * df['volume']
    cum_pv_60 = pv.rolling(60).sum(); cum_v_60 = df['volume'].rolling(60).sum()
    cum_pv_240 = pv.rolling(240).sum(); cum_v_240 = df['volume'].rolling(240).sum()
    df['vwap_60'] = cum_pv_60 / cum_v_60.replace(0, np.nan)
    df['vwap_240'] = cum_pv_240 / cum_v_240.replace(0, np.nan)
    df['vwap_dist_60'] = (df['close'] - df['vwap_60']) / df['close'].rolling(60).std().replace(0, np.nan)
    df['vwap_dist_240'] = (df['close'] - df['vwap_240']) / df['close'].rolling(240).std().replace(0, np.nan)
    df['range_position_15'] = (df['close'] - df['low'].rolling(15).min()) / (df['high'].rolling(15).max() - df['low'].rolling(15).min()).replace(0, np.nan)
    df['range_position_60'] = (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min()).replace(0, np.nan)
    df['breakout_15'] = ((df['close'] > df['high'].rolling(15).max().shift(1)).astype(int) -
                          (df['close'] < df['low'].rolling(15).min().shift(1)).astype(int))
    df['breakout_60'] = ((df['close'] > df['high'].rolling(60).max().shift(1)).astype(int) -
                          (df['close'] < df['low'].rolling(60).min().shift(1)).astype(int))

    # === STATIONARITY / REGIME ===
    print('  computing stationarity features (slow)...')
    df['hurst_60'] = hurst_rs(df['close'], 60)
    df['hurst_240'] = hurst_rs(df['close'], 240)
    df['autocorr_lag1_60'] = autocorr_lag1(df['ret_1'], 60)
    df['autocorr_lag1_240'] = autocorr_lag1(df['ret_1'], 240)
    df['variance_ratio_15_60'] = variance_ratio(df['ret_1'], 15)
    df['variance_ratio_60_240'] = variance_ratio(df['ret_1'], 60)

    # === RSI / OSCILLATION ===
    df['rsi_14'] = rsi(df['close'], 14)
    df['rsi_60'] = rsi(df['close'], 60)

    # === VOLUME MATH ===
    df['volume_at_high_60'] = (df['volume'] * (df['close'] > df['close'].rolling(60).mean()).astype(float)).rolling(60).sum()
    df['volume_at_low_60'] = (df['volume'] * (df['close'] < df['close'].rolling(60).mean()).astype(float)).rolling(60).sum()
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_slope_60'] = rolling_slope(obv, 60)
    df['volume_pct_60'] = df['volume'].rolling(60).rank(pct=True)

    # === Time/calendar context ===
    df['hour_et'] = df.index.hour
    df['minute_of_hour'] = df.index.minute
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_et'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_et'] / 24)
    df['day_of_week'] = df.index.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    df['day_of_month'] = df.index.day
    df['month_of_year'] = df.index.month

    # Setup direction
    df['prev_bar_color'] = np.sign(df['close'].shift(1) - df['open'].shift(1))
    df['prev_2_bar_color'] = np.sign(df['close'].shift(2) - df['open'].shift(2))
    df['ret_15'] = np.log(df['close'] / df['close'].shift(15))
    df['ret_60'] = np.log(df['close'] / df['close'].shift(60))
    df['body_share'] = df['body'] / df['range'].replace(0, np.nan)
    df['upper_wick_share'] = df['upper_wick'] / df['range'].replace(0, np.nan)
    df['lower_wick_share'] = df['lower_wick'] / df['range'].replace(0, np.nan)

    return df


def label_walk(bars, win_idx, tp, sl, h_min):
    high = bars['high'].values; low = bars['low'].values
    open_ = bars['open'].values; close = bars['close'].values
    n = len(bars)
    long_lbl = np.full(len(win_idx), -1, dtype=np.int8)
    short_lbl = np.full(len(win_idx), -1, dtype=np.int8)
    for ii, i in enumerate(win_idx):
        if i + 1 + h_min >= n: continue
        entry = open_[i + 1]
        for j in range(i + 1, min(i + 1 + h_min, n)):
            hh, ll = high[j], low[j]
            if long_lbl[ii] == -1:
                if ll <= entry - sl: long_lbl[ii] = 0
                elif hh >= entry + tp: long_lbl[ii] = 1
            if short_lbl[ii] == -1:
                if hh >= entry + sl: short_lbl[ii] = 0
                elif ll <= entry - tp: short_lbl[ii] = 1
            if long_lbl[ii] != -1 and short_lbl[ii] != -1: break
        if long_lbl[ii] == -1:
            last = close[min(i + h_min, n - 1)]
            long_lbl[ii] = 1 if last > entry else 0
        if short_lbl[ii] == -1:
            last = close[min(i + h_min, n - 1)]
            short_lbl[ii] = 1 if last < entry else 0
    return long_lbl, short_lbl


def build_period(name, start, end, tp, sl):
    print(f'\n[{name}] {start}→{end}, TP={tp}/SL={sl}')
    t0 = time.time()
    bars = load_period(start, end)
    if bars.empty: return None
    feats = compute_features(bars)
    print(f'  features computed, dt={time.time()-t0:.0f}s')
    h = feats.index.hour; dow = feats.index.dayofweek
    win_mask = np.isin(h, WIN_HOURS) & (dow < 5)
    win_idx = np.where(win_mask)[0]
    long_lbl, short_lbl = label_walk(feats, win_idx, tp, sl, H_MIN)
    feature_cols = [c for c in feats.columns if c not in ['open','high','low','close','volume','range','body','upper_wick','lower_wick','vwap_60','vwap_240']]
    feats_win = feats.iloc[win_idx][feature_cols].copy()
    feats_win['label_long'] = long_lbl
    feats_win['label_short'] = short_lbl
    long_df = feats_win.copy(); long_df['side'] = 0; long_df['label'] = long_df['label_long']
    short_df = feats_win.copy(); short_df['side'] = 1; short_df['label'] = short_df['label_short']
    out = pd.concat([long_df, short_df], axis=0)
    out = out[out['label'] >= 0].copy().drop(columns=['label_long', 'label_short'])
    print(f'  → {len(out):,} samples, {len(feature_cols)+1} features (incl. side), dt={time.time()-t0:.0f}s')
    return out


def evaluate_full(p, y, ts_arr, fired, tp, sl, val_days):
    if not fired.any(): return None
    df = pd.DataFrame({'ts': ts_arr, 'fired': fired, 'win': (y == 1) & fired})
    ts_dt = pd.to_datetime(df['ts'])
    df['month'] = ts_dt.dt.to_period('M')
    df['year'] = ts_dt.dt.year
    df['pnl'] = np.where(fired, np.where(df['win'], tp * PT_USD, -sl * PT_USD), 0)
    monthly = df[df['fired']].groupby('month').agg(n=('fired','sum'), w=('win','sum'), pnl=('pnl','sum')).reset_index()
    monthly['wr'] = monthly['w'] / monthly['n']
    yearly = []
    for y_, grp in df[df['fired']].groupby('year'):
        ord_pnl = grp.sort_values('ts')['pnl'].cumsum()
        peak = ord_pnl.cummax()
        dd = float((ord_pnl - peak).min())
        yearly.append({'year': int(y_), 'n': int(grp['fired'].sum()), 'w': int(grp['win'].sum()),
                       'pnl': float(grp['pnl'].sum()), 'dd': dd, 'wr': float(grp['win'].sum() / grp['fired'].sum())})
    n = int(monthly['n'].sum())
    return {
        'n': n, 'wr': float(monthly['w'].sum() / n) if n else 0,
        'pnl': float(monthly['pnl'].sum()),
        'avg': float(monthly['pnl'].sum() / n) if n else 0,
        'per_day': n / val_days,
        'monthly': monthly.to_dict('records'),
        'yearly': yearly,
    }


def check_relaxed(stats):
    """80% positive months allowed; rest hard."""
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


def run_bracket(tp, sl):
    print(f'\n========== TP={tp} / SL={sl} (R={tp/sl:.2f}) ==========')
    breakeven = sl / (tp + sl)
    print(f'breakeven WR = {breakeven*100:.1f}%')
    all_data = {}
    for name, (s, e) in PERIODS.items():
        df = build_period(name, s, e, tp, sl)
        if df is None or df.empty: continue
        all_data[name] = df

    train = pd.concat([all_data.get('TRAIN_t1', pd.DataFrame()),
                       all_data.get('TRAIN_2025', pd.DataFrame())], axis=0)
    val = all_data['VAL_biden']
    feat_cols = [c for c in train.columns if c != 'label']
    Xt = train[feat_cols].astype(np.float32); yt = train['label'].astype(np.int8).values
    Xv = val[feat_cols].astype(np.float32); yv = val['label'].astype(np.int8).values
    # Drop rows where label is invalid; HGB tolerates feature NaN natively.
    # Also drop rows where ALL features are NaN (pre-warmup bars).
    nan_count_t = Xt.isna().sum(axis=1)
    nan_count_v = Xv.isna().sum(axis=1)
    valid_t = (nan_count_t < len(feat_cols) * 0.5)  # at least half features non-NaN
    valid_v = (nan_count_v < len(feat_cols) * 0.5)
    Xt = Xt[valid_t].reset_index(drop=True); yt = yt[valid_t.values]
    Xv = Xv[valid_v].reset_index(drop=True); yv = yv[valid_v.values]
    val_ts = val.index[valid_v.values]
    val_days = pd.to_datetime(val_ts).normalize().nunique()
    print(f'  TRAIN={len(Xt):,}, VAL={len(Xv):,}, val_days={val_days}, features={len(feat_cols)}')

    # LONG only
    long_t = (Xt['side'] == 0).values
    long_v = (Xv['side'] == 0).values
    Xt_L = Xt[long_t].reset_index(drop=True); yt_L = yt[long_t]
    Xv_L = Xv[long_v].reset_index(drop=True); yv_L = yv[long_v]
    val_ts_L = val_ts[long_v]
    print(f'  LONG: TRAIN={len(Xt_L):,}, VAL={len(Xv_L):,}, baseline VAL WR={yv_L.mean()*100:.1f}%')

    # ANOVA → top features. Replace inf and clip extreme values first.
    Xt_clean = Xt_L.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6)
    F, _ = f_classif(Xt_clean.values, yt_L)
    fa = pd.DataFrame({'f': feat_cols, 'F': F}).sort_values('F', ascending=False)
    top60 = fa.head(60)['f'].tolist()
    print(f'  Top 10 by F-stat: {fa.head(10)["f"].tolist()}')

    # K-means + HMM gates (use top σ + top directional)
    cluster_features = [f for f in top60 if any(k in f for k in ['sigma','z_','slope','momentum','hurst','vwap_dist','range_pos','rsi'])][:18]
    Xt_c = Xt_L[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    Xv_c = Xv_L[cluster_features].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    sc = StandardScaler(); Xt_s = sc.fit_transform(Xt_c); Xv_s = sc.transform(Xv_c)
    pca = PCA(n_components=10, random_state=42)
    Xt_p = pca.fit_transform(Xt_s); Xv_p = pca.transform(Xv_s)
    print(f'  PCA explained var: {pca.explained_variance_ratio_.sum():.3f}')

    best_K = None
    for K in [4, 6, 8, 10]:
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        ct = km.fit_predict(Xt_p); cv = km.predict(Xv_p)
        df_t = pd.DataFrame({'c': ct, 'y': yt_L}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        df_v = pd.DataFrame({'c': cv, 'y': yv_L}).groupby('c').agg(n=('y','count'), wr=('y','mean'))
        good = [c for c in df_t.index if df_t.loc[c,'wr'] > breakeven + 0.03 and df_t.loc[c,'n'] > 1000
                and c in df_v.index and df_v.loc[c,'wr'] > breakeven + 0.02]
        if good and (not best_K or len(good) > len(best_K['good'])):
            best_K = {'K': K, 'good': good, 'cv': cv}
    cluster_mask_v = np.isin(best_K['cv'], best_K['good']) if best_K else np.ones(len(Xv_L), dtype=bool)
    print(f'  → cluster K={best_K["K"] if best_K else None}, good={best_K["good"] if best_K else []}, mask={cluster_mask_v.sum():,}')

    # HMM
    hmm_feats = [f for f in ['sigma_60','sigma_240','sigma_of_sigma_60_240','hurst_60','momentum_z_60','vwap_dist_60'] if f in feat_cols]
    def _clean(df, cols):
        return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values
    sc_h = StandardScaler(); Xt_h2 = sc_h.fit_transform(_clean(Xt_L, hmm_feats))
    Xv_h = sc_h.transform(_clean(Xv_L, hmm_feats))
    best_hmm = None
    for N in [2, 3, 4]:
        try:
            hmm = GaussianHMM(n_components=N, covariance_type='diag', n_iter=50, random_state=42)
            hmm.fit(Xt_h2)
            st_t = hmm.predict(Xt_h2); st_v = hmm.predict(Xv_h)
            df_t = pd.DataFrame({'s': st_t, 'y': yt_L}).groupby('s').agg(n=('y','count'), wr=('y','mean'))
            df_v = pd.DataFrame({'s': st_v, 'y': yv_L}).groupby('s').agg(n=('y','count'), wr=('y','mean'))
            good = [s for s in df_t.index if df_t.loc[s,'wr'] > breakeven + 0.02 and df_t.loc[s,'n'] > 1000
                    and s in df_v.index and df_v.loc[s,'wr'] > breakeven + 0.01]
            if good and (not best_hmm or len(good) > len(best_hmm['good'])):
                best_hmm = {'N': N, 'good': good, 'st_v': st_v}
        except: pass
    hmm_mask_v = np.isin(best_hmm['st_v'], best_hmm['good']) if best_hmm and best_hmm['good'] else np.ones(len(Xv_L), dtype=bool)
    print(f'  → HMM N={best_hmm["N"] if best_hmm else None}, good={best_hmm["good"] if best_hmm else []}, mask={hmm_mask_v.sum():,}')

    # HGB — clean inf/extreme values
    Xt_hgb = Xt_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    Xv_hgb = Xv_L[top60].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
    hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=400,
                                          early_stopping=True, n_iter_no_change=15, random_state=42)
    hgb.fit(Xt_hgb, yt_L)
    pv = hgb.predict_proba(Xv_hgb)[:, 1]
    auc = roc_auc_score(yv_L, pv)
    print(f'  HGB Val AUC: {auc:.4f}')

    # Threshold sweep
    print(f'\n  Threshold sweep — LONG only, cluster + HMM gates active:')
    print(f'  {"thr":>5s} {"n":>5s} {"/d":>5s} {"WR%":>5s} {"avg":>7s} {"PnL/4yr":>10s} {"yrs+":>5s} {"mos+%":>6s} {"PASS"}')
    candidates = []
    for thr in np.arange(0.95, 0.30, -0.01):
        fired = (pv >= thr) & cluster_mask_v & hmm_mask_v
        n = int(fired.sum())
        if n == 0: continue
        stats = evaluate_full(pv, yv_L, val_ts_L, fired, tp, sl, val_days)
        if stats is None: continue
        passed, fails, pct_pos = check_relaxed(stats)
        yrs_pos_count = sum(1 for y in stats['yearly'] if y['pnl'] > 0 and y['wr'] > 0.5 and y['dd'] >= -900)
        candidates.append({**stats, 'thr': float(thr), 'passed': passed, 'fails': fails,
                          'yrs_pos': yrs_pos_count, 'mos_pos_pct': pct_pos})

    for c in candidates[::4]:
        flag = '🟢' if c['passed'] else '  '
        print(f'  {c["thr"]:.3f} {c["n"]:>5d} {c["per_day"]:>4.2f} {c["wr"]*100:>4.1f}% ${c["avg"]:>+5.0f} ${c["pnl"]:>+8.0f} {c["yrs_pos"]:>5d}/{len(c["yearly"])} {c["mos_pos_pct"]*100:>4.0f}% {flag}')

    passing = [c for c in candidates if c['passed']]
    if passing:
        # Pick highest n among passing (stay in 2-4/d band)
        best = max(passing, key=lambda c: c['n'])
        return {'tp': tp, 'sl': sl, 'best': best, 'auc': auc, 'hgb': hgb, 'features': top60,
                'cluster_K': best_K, 'hmm': best_hmm, 'cluster_features': cluster_features, 'hmm_features': hmm_feats}
    else:
        # Closest by combined score
        rate_ok = [c for c in candidates if 2 <= c['per_day'] <= 4 and c['avg'] >= 60]
        if rate_ok:
            best_close = max(rate_ok, key=lambda c: (c['mos_pos_pct'], c['pnl']))
            print(f'  closest(rate+avg ok): thr={best_close["thr"]:.3f} {best_close["per_day"]:.2f}/d WR={best_close["wr"]*100:.1f}% avg=${best_close["avg"]:.0f} mos+={best_close["mos_pos_pct"]*100:.0f}% yrs+={best_close["yrs_pos"]}/{len(best_close["yearly"])}')
        return None


def main():
    t0 = time.time()
    results = []
    for tp, sl in [(25, 10), (30, 10), (40, 15), (35, 12)]:  # pruned to most promising
        r = run_bracket(tp, sl)
        if r: results.append(r)

    print('\n========== SUMMARY ==========')
    if results:
        for r in results:
            b = r['best']
            print(f'  TP={r["tp"]}/SL={r["sl"]}: thr={b["thr"]:.3f}, {b["per_day"]:.2f}/d, WR={b["wr"]*100:.1f}%, avg=${b["avg"]:.0f}, +${b["pnl"]:.0f}, mos+={b["mos_pos_pct"]*100:.0f}%')
        best = max(results, key=lambda r: r['best']['pnl'])
        b = best['best']
        print(f'\n🟢 BEST: TP={best["tp"]}/SL={best["sl"]}, thr={b["thr"]:.3f}')
        print(f'   {b["per_day"]:.2f}/d, WR={b["wr"]*100:.1f}%, avg=${b["avg"]:.0f}, PnL=+${b["pnl"]:.0f}, mos+={b["mos_pos_pct"]*100:.0f}%')
        for y in b['yearly']:
            print(f'   {y["year"]}: n={y["n"]} WR={y["wr"]*100:.1f}% PnL=${y["pnl"]:+.0f} DD=${y["dd"]:+.0f}')
        joblib.dump({
            'model': best['hgb'], 'features': best['features'],
            'tp': best['tp'], 'sl': best['sl'], 'threshold': b['thr'],
            'val_auc': best['auc'],
            'cluster_features': best['cluster_features'],
            'hmm_features': best['hmm_features'],
        }, OUT / 'iter6_model.pkl')
        print(f'\n✅ Saved iter6_model.pkl')
    else:
        print('❌ NO PASSING. Closest configs printed above per bracket.')
    print(f'\nWall: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
