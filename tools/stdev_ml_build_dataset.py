"""σ-ML training dataset builder for ET 9-11 window.

Produces a labeled feature matrix per (bar, side) for:
  TRAIN    Trump T1 (2017-01-20 → 2021-01-20) + 2025 (2025-01-20 → 2025-12-31)
  VAL      Biden term (2021-01-21 → 2025-01-19)
  OOS      Trump T2 2026 (2026-01-01 → 2026-04-30)

Labels: 1 if TP=8.25pts hits before SL=10pts in next 120 min, else 0.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/Users/wes/Downloads/JULIE001')
BAR_FILE = ROOT / 'es_master_outrights-2.parquet'
OUT_DIR = ROOT / 'artifacts' / 'stdev_ml_hr9_11'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TP_PTS = 8.25
SL_PTS = 10.0
HORIZON_MIN = 120
ET = 'US/Eastern'

PERIODS = {
    'TRAIN_trump_t1':   ('2017-01-20', '2021-01-20'),
    'TRAIN_trump_2025': ('2025-01-20', '2025-12-31'),
    'VAL_biden':        ('2021-01-21', '2025-01-19'),
    'OOS_2026':         ('2026-01-01', '2026-04-30'),
}


def load_bars_for_period(start: str, end: str) -> pd.DataFrame:
    """Load bars filtered to date range, pick dominant ES contract per day by volume."""
    print(f'  loading {start} → {end}')
    bars = pd.read_parquet(BAR_FILE)
    bars.index = pd.to_datetime(bars.index, utc=True).tz_convert(ET)
    bars = bars.loc[start:end]
    if bars.empty:
        return bars

    # Pick dominant contract per day by volume (handles roll periods)
    # Preserve timestamp index across merge by using reset_index pattern
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


def filter_window(bars: pd.DataFrame) -> pd.DataFrame:
    """Filter to ET 9-11 weekdays (Mon-Fri)."""
    if bars.empty: return bars
    h = bars.index.hour
    dow = bars.index.dayofweek
    mask = (h >= 9) & (h <= 11) & (dow < 5)
    return bars[mask].copy()


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute 86 σ-derived features. Operates on FULL bar series (not just window)
    so rolling stats have proper context, then we filter at the end."""
    if bars.empty: return pd.DataFrame()
    print('    computing features...')
    df = bars.copy()
    df['ret_1'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_5'] = np.log(df['close'] / df['close'].shift(5))
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # ============ Pure σ at multiple windows (12 features) ============
    for w in [5, 10, 15, 30, 60, 120, 240]:
        df[f'sigma_{w}'] = df['ret_1'].rolling(w).std() * np.sqrt(w)
    df['sigma_5_log'] = np.log1p(df['sigma_5'].clip(lower=1e-9))
    df['sigma_15_log'] = np.log1p(df['sigma_15'].clip(lower=1e-9))
    df['sigma_60_log'] = np.log1p(df['sigma_60'].clip(lower=1e-9))
    df['sigma_intraday_running'] = df['ret_1'].groupby(df.index.normalize()).transform(
        lambda x: x.expanding().std()
    )
    df['sigma_open_to_now'] = df.groupby(df.index.normalize())['close'].transform(
        lambda x: x.expanding().std() / x.iloc[0] if len(x) > 0 else 0
    )

    # ============ σ ratios (10 features) ============
    df['sigma_ratio_5_60']    = df['sigma_5'] / df['sigma_60'].replace(0, np.nan)
    df['sigma_ratio_15_60']   = df['sigma_15'] / df['sigma_60'].replace(0, np.nan)
    df['sigma_ratio_15_240']  = df['sigma_15'] / df['sigma_240'].replace(0, np.nan)
    df['sigma_ratio_60_240']  = df['sigma_60'] / df['sigma_240'].replace(0, np.nan)
    df['sigma_ratio_5_15']    = df['sigma_5'] / df['sigma_15'].replace(0, np.nan)
    df['sigma_ratio_30_120']  = df['sigma_30'] / df['sigma_120'].replace(0, np.nan)
    df['sigma_ratio_intraday_240'] = df['sigma_intraday_running'] / df['sigma_240'].replace(0, np.nan)
    df['sigma_ratio_15_60_log']  = df['sigma_15_log'] - df['sigma_60_log']
    df['sigma_ratio_5_60_log']   = df['sigma_5_log'] - df['sigma_60_log']
    df['sigma_5_chg_5'] = df['sigma_5'].pct_change(5)

    # ============ σ percentile rankings (8 features) ============
    df['sigma_15_pctile_60']  = df['sigma_15'].rolling(60).rank(pct=True)
    df['sigma_15_pctile_240'] = df['sigma_15'].rolling(240).rank(pct=True)
    df['sigma_15_pctile_390'] = df['sigma_15'].rolling(390).rank(pct=True)
    df['sigma_60_pctile_240'] = df['sigma_60'].rolling(240).rank(pct=True)
    df['sigma_60_pctile_390'] = df['sigma_60'].rolling(390).rank(pct=True)
    # Same-hour-of-day rolling percentile (last 20 days)
    df['hour'] = df.index.hour
    df['sigma_15_pctile_hourly_20d'] = df.groupby('hour')['sigma_15'].transform(
        lambda x: x.rolling(20 * 12).rank(pct=True)
    )
    df['sigma_60_pctile_hourly_20d'] = df.groupby('hour')['sigma_60'].transform(
        lambda x: x.rolling(20 * 12).rank(pct=True)
    )
    df['sigma_intraday_pctile_20d'] = df.groupby('hour')['sigma_intraday_running'].transform(
        lambda x: x.rolling(20 * 12).rank(pct=True)
    )

    # ============ σ-of-σ (vol of vol) (8 features) ============
    df['sigma_of_sigma_15_60'] = df['sigma_15'].rolling(60).std()
    df['sigma_of_sigma_15_240'] = df['sigma_15'].rolling(240).std()
    df['sigma_of_sigma_60_240'] = df['sigma_60'].rolling(240).std()
    df['delta_sigma_15_5'] = df['sigma_15'] - df['sigma_15'].shift(5)
    df['delta_sigma_60_15'] = df['sigma_60'] - df['sigma_60'].shift(15)
    df['sigma_acceleration'] = (df['sigma_5'] - df['sigma_5'].shift(5)) / df['sigma_5'].shift(5).replace(0, np.nan)
    df['sigma_skew_30']  = df['ret_1'].rolling(30).skew()
    df['sigma_skew_120'] = df['ret_1'].rolling(120).skew()

    # ============ Z-scores (16 features) ============
    for w in [15, 60, 240]:
        m = df['close'].rolling(w).mean()
        s = df['close'].rolling(w).std()
        df[f'z_close_{w}'] = (df['close'] - m) / s.replace(0, np.nan)
    rng = df['range']
    df['z_high_15']  = (df['high']  - df['close'].rolling(15).mean()) / df['close'].rolling(15).std().replace(0, np.nan)
    df['z_low_15']   = (df['low']   - df['close'].rolling(15).mean()) / df['close'].rolling(15).std().replace(0, np.nan)
    df['z_range_15']  = (rng - rng.rolling(15).mean()) / rng.rolling(15).std().replace(0, np.nan)
    df['z_range_60']  = (rng - rng.rolling(60).mean()) / rng.rolling(60).std().replace(0, np.nan)
    df['z_body_15']   = (df['body'] - df['body'].rolling(15).mean()) / df['body'].rolling(15).std().replace(0, np.nan)
    df['z_upper_wick_15'] = (df['upper_wick'] - df['upper_wick'].rolling(15).mean()) / df['upper_wick'].rolling(15).std().replace(0, np.nan)
    df['z_lower_wick_15'] = (df['lower_wick'] - df['lower_wick'].rolling(15).mean()) / df['lower_wick'].rolling(15).std().replace(0, np.nan)
    vol = df['volume']
    df['z_volume_15']  = (vol - vol.rolling(15).mean()) / vol.rolling(15).std().replace(0, np.nan)
    df['z_volume_60']  = (vol - vol.rolling(60).mean()) / vol.rolling(60).std().replace(0, np.nan)
    df['z_return_1']   = (df['ret_1'] - df['ret_1'].rolling(60).mean()) / df['ret_1'].rolling(60).std().replace(0, np.nan)
    df['z_return_5']   = (df['ret_5'] - df['ret_5'].rolling(60).mean()) / df['ret_5'].rolling(60).std().replace(0, np.nan)
    # Open of day for z_close_to_open
    df['day_open'] = df.groupby(df.index.normalize())['open'].transform('first')
    df['z_close_to_open'] = (df['close'] - df['day_open']) / df['close'].rolling(60).std().replace(0, np.nan)

    # ============ Distribution shape (8 features) ============
    df['skew_return_60']  = df['ret_1'].rolling(60).skew()
    df['skew_return_240'] = df['ret_1'].rolling(240).skew()
    df['kurt_return_60']  = df['ret_1'].rolling(60).kurt()
    df['kurt_return_240'] = df['ret_1'].rolling(240).kurt()
    # Efficiency ratio: net move / sum of absolute moves
    abs_ret = df['ret_1'].abs()
    df['efficiency_60']  = (df['close'] - df['close'].shift(60)).abs() / (abs_ret.rolling(60).sum() * df['close'].shift(60)).replace(0, np.nan)
    df['efficiency_240'] = (df['close'] - df['close'].shift(240)).abs() / (abs_ret.rolling(240).sum() * df['close'].shift(240)).replace(0, np.nan)
    # Trend strength: R² of close vs time over rolling window (approximation)
    def rolling_r2(s, w):
        x = np.arange(w)
        x_mean = x.mean(); x_std = x.std()
        def r2(y):
            y_mean = y.mean(); y_std = y.std()
            if y_std < 1e-12 or x_std < 1e-12: return 0
            cov = ((x - x_mean) * (y - y_mean)).mean()
            return (cov / (x_std * y_std)) ** 2
        return s.rolling(w).apply(r2, raw=True)
    # Sample lighter version: correlation² over window using close prices
    df['trend_r2_60']  = df['close'].rolling(60).corr(pd.Series(np.arange(len(df)), index=df.index)) ** 2
    df['trend_r2_240'] = df['close'].rolling(240).corr(pd.Series(np.arange(len(df)), index=df.index)) ** 2

    # ============ Time/calendar context (10 features) ============
    df['hour_et']  = df.index.hour
    df['minute_of_hour'] = df.index.minute
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_et'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_et'] / 24)
    df['day_of_week']     = df.index.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    df['week_of_year']    = df.index.isocalendar().week.values
    df['month']           = df.index.month
    df['day_of_month']    = df.index.day

    # ============ Setup direction & momentum context (8 features) ============
    df['prev_bar_color']  = np.sign(df['close'].shift(1) - df['open'].shift(1))
    df['prev_2_bar_color']= np.sign(df['close'].shift(2) - df['open'].shift(2))
    df['ret_15'] = np.log(df['close'] / df['close'].shift(15))
    df['ret_60'] = np.log(df['close'] / df['close'].shift(60))
    df['cum_session_ret'] = (df['close'] - df['day_open']) / df['day_open']
    # Bar-pattern: body share of range
    df['body_share'] = df['body'] / df['range'].replace(0, np.nan)
    df['upper_wick_share'] = df['upper_wick'] / df['range'].replace(0, np.nan)
    df['lower_wick_share'] = df['lower_wick'] / df['range'].replace(0, np.nan)

    # Drop helpers, keep only feature columns + OHLC for label generation
    drop_cols = ['day_open', 'date', 'hour']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def vectorized_walk_forward(bars: pd.DataFrame, side: str, tp: float, sl: float, h: int) -> np.ndarray:
    """Vectorized TP/SL walk-forward. Returns array of {1=TP-first, 0=SL-first or timeout-loss, -1=skip}."""
    n = len(bars)
    out = np.zeros(n, dtype=np.int8)
    high = bars['high'].values
    low = bars['low'].values
    close = bars['close'].values
    open_ = bars['open'].values

    for i in range(n - h - 1):
        entry = open_[i + 1] if i + 1 < n else close[i]  # enter at next bar's open
        if side == 'LONG':
            tp_lvl = entry + tp
            sl_lvl = entry - sl
            # Iterate forward bars to find first hit
            for j in range(i + 1, min(i + 1 + h, n)):
                if low[j] <= sl_lvl:
                    out[i] = 0  # SL first
                    break
                if high[j] >= tp_lvl:
                    out[i] = 1  # TP first
                    break
            else:
                # Timeout: label by sign of final close vs entry
                last = close[min(i + h, n - 1)]
                out[i] = 1 if last > entry else 0
        else:  # SHORT
            tp_lvl = entry - tp
            sl_lvl = entry + sl
            for j in range(i + 1, min(i + 1 + h, n)):
                if high[j] >= sl_lvl:
                    out[i] = 0
                    break
                if low[j] <= tp_lvl:
                    out[i] = 1
                    break
            else:
                last = close[min(i + h, n - 1)]
                out[i] = 1 if last < entry else 0
    # Last h bars: can't label
    out[max(0, n - h - 1):] = -1
    return out


def build_period(name: str, start: str, end: str) -> pd.DataFrame:
    print(f'\n[{name}] {start} → {end}')
    t0 = time.time()
    bars = load_bars_for_period(start, end)
    if bars.empty:
        print(f'  ❌ no bars'); return pd.DataFrame()
    print(f'  loaded {len(bars):,} bars, dt={time.time()-t0:.1f}s')
    feats = compute_features(bars)
    if feats.empty: return pd.DataFrame()
    # Filter to ET 9-11 weekday window
    h = feats.index.hour; dow = feats.index.dayofweek
    win = (h >= 9) & (h <= 11) & (dow < 5)
    feats_win = feats[win].copy()
    print(f'  windowed: {len(feats_win):,} bars in ET 9-11')

    # Walk forward labels for both sides — use full bars (not windowed) for label horizon
    rows = []
    for side in ['LONG', 'SHORT']:
        labels = vectorized_walk_forward(feats, side, TP_PTS, SL_PTS, HORIZON_MIN)
        feats['label_' + side.lower()] = labels
    # Now sub-select windowed rows
    feats_win_l = feats[win].copy()
    feats_win_s = feats[win].copy()
    feats_win_l['side'] = 0  # LONG=0
    feats_win_l['label'] = feats_win_l['label_long']
    feats_win_s['side'] = 1  # SHORT=1
    feats_win_s['label'] = feats_win_s['label_short']
    out = pd.concat([feats_win_l, feats_win_s], axis=0)
    # Drop bars with no valid label
    out = out[out['label'] >= 0].copy()
    # Keep only feature columns + label
    feature_cols = [c for c in out.columns if c not in ['label', 'label_long', 'label_short', 'open', 'high', 'low', 'close', 'volume', 'range', 'body', 'upper_wick', 'lower_wick']]
    keep = feature_cols + ['label']
    out = out[keep]
    print(f'  final: {len(out):,} (LONG+SHORT) labeled samples, {len(feature_cols)} features, dt={time.time()-t0:.1f}s')
    return out


def main():
    all_data = {}
    for name, (start, end) in PERIODS.items():
        df = build_period(name, start, end)
        if df.empty: continue
        all_data[name] = df

    # Combine TRAIN periods
    train = pd.concat([all_data['TRAIN_trump_t1'], all_data['TRAIN_trump_2025']], axis=0)
    val = all_data['VAL_biden']
    oos = all_data['OOS_2026']

    feat_cols = [c for c in train.columns if c != 'label']
    print(f'\n=== FINAL DATASET ===')
    print(f'TRAIN:  {len(train):,} samples,  {(train["label"]==1).mean()*100:.1f}% TP-first WR')
    print(f'VAL:    {len(val):,} samples,  {(val["label"]==1).mean()*100:.1f}% TP-first WR')
    print(f'OOS:    {len(oos):,} samples,  {(oos["label"]==1).mean()*100:.1f}% TP-first WR')
    print(f'Features: {len(feat_cols)}')

    train.to_parquet(OUT_DIR / 'train.parquet')
    val.to_parquet(OUT_DIR / 'val.parquet')
    oos.to_parquet(OUT_DIR / 'oos.parquet')
    print(f'\nSaved to {OUT_DIR}')

if __name__ == '__main__':
    main()
