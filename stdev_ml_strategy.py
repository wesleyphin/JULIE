"""StdevMlStrategy — σ-based ML strategy, ATR-scaled brackets, ET 11-12 window.

Final config from iter 12/13 (single-position evaluator):
  - Bracket: TP = 1.5 × ATR_60, SL = 1.0 × ATR_60 (in points)
  - Threshold: 0.75 (combined LONG/SHORT max-prob)
  - Daily DD cap: $200 (wire via circuit_breaker max_daily_loss)
  - Horizon: 60 minutes
  - Window: ET 11:00–12:00 (= PT 8-9)

OOS_2026 single-position results: 225 trades, 77.3% WR, $94 avg, +$21,125,
DD -$724. All months positive PnL. Worst-WR month: 2026-04 at 57.4%.

Behavior is regime-dependent: heavy fire in trending bull regimes (2024 = 6/d,
+$90k), near-silent in 2022-2023 chop (~0.5/d, ~$5k each year). DD stays low
even in silent periods.

Caveats:
  1. AUC drops VAL→OOS: 0.72 → 0.62. Calibration drift across years.
  2. Top features include vwap_60 (a price-LEVEL feature) which may track year/
     regime. Probably partly responsible for cross-year edge stability.
  3. Cross-sectional time-of-day features (sigma_15_tod_z etc.) require ≥20 days
     of intraday history at the same minute-of-day. Bot must keep ≥4800 bars
     of warm-up (~5 trading days × 960 minutes) for compute_features to produce
     all features. Strategy returns None until warmed up.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

from config import CONFIG
from strategy_base import Strategy

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts' / 'stdev_ml_hr11_12' / 'iter12_both_sides'
SEED_PARQUET = ROOT / 'es_master_outrights-2.parquet'

# ATR multipliers + threshold + window — frozen from iter 13 ship config
TP_ATR_MULT = float(CONFIG.get('STDEV_ML_TP_ATR_MULT', 1.5))
SL_ATR_MULT = float(CONFIG.get('STDEV_ML_SL_ATR_MULT', 1.0))
THRESHOLD = float(CONFIG.get('STDEV_ML_THRESHOLD', 0.75))
WIN_HOUR_START = int(CONFIG.get('STDEV_ML_WINDOW_START_ET', 11))  # ET 11:00
WIN_HOUR_END = int(CONFIG.get('STDEV_ML_WINDOW_END_ET', 12))      # ET 12:00 (exclusive)
WARMUP_BARS = int(CONFIG.get('STDEV_ML_WARMUP_BARS', 4800))
# Tail of historical OHLCV parquet to seed the feature buffer at __init__.
# Without this, ~5 days of bot uptime gives only ~2,880 bars in the live buffer
# and the warmup gate never clears. With seed, first on_bar already has 8k+ bars.
SEED_BARS = int(CONFIG.get('STDEV_ML_SEED_BARS', 8000))


def _rolling_slope(series, w):
    def slope_calc(y):
        if len(y) < 2 or np.isnan(y).any(): return 0
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]
    return series.rolling(w).apply(slope_calc, raw=True)


def _rsi(close, w):
    delta = close.diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(w).mean(); avg_loss = loss.rolling(w).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _compute_features_live(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror tools/stdev_ml_iter11_atr.compute_features. Same columns, causal."""
    df = df.copy()
    df['ret_1'] = np.log(df['close'] / df['close'].shift(1))
    df['ret_5'] = np.log(df['close'] / df['close'].shift(5))
    df['ret_15'] = np.log(df['close'] / df['close'].shift(15))
    df['ret_60'] = np.log(df['close'] / df['close'].shift(60))
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()

    sigmas = {}
    for w in (5, 10, 15, 30, 60, 120, 240):
        s = df['ret_1'].rolling(w).std() * np.sqrt(w)
        df[f'sigma_{w}'] = s
        df[f'sigma_{w}_log'] = np.log(s.replace(0, np.nan))
        sigmas[w] = s
    df['sigma_intraday'] = df['ret_1'].rolling(60).std()
    df['sigma_5_15'] = sigmas[5] / sigmas[15].replace(0, np.nan)
    df['sigma_15_60'] = sigmas[15] / sigmas[60].replace(0, np.nan)
    df['sigma_60_240'] = sigmas[60] / sigmas[240].replace(0, np.nan)
    df['sigma_15_pctile_240'] = sigmas[15].rolling(240).rank(pct=True)
    df['sigma_60_pctile_390'] = sigmas[60].rolling(390).rank(pct=True)
    df['sigma_60_pctile_hourly_20d'] = sigmas[60].rolling(240*20).rank(pct=True)
    df['sigma_of_sigma_15_60'] = sigmas[15].rolling(60).std()
    df['sigma_of_sigma_60_240'] = sigmas[60].rolling(240).std()

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['atr_60'] = tr.rolling(60).mean()
    df['atr_15'] = tr.rolling(15).mean()
    df['atr_ratio'] = df['atr_15'] / df['atr_60'].replace(0, np.nan)

    rng = df['range']
    df['z_range_60'] = (rng - rng.rolling(60).mean()) / rng.rolling(60).std().replace(0, np.nan)
    df['z_volume_60'] = (df['volume'] - df['volume'].rolling(60).mean()) / df['volume'].rolling(60).std().replace(0, np.nan)
    df['z_return_5'] = (df['ret_5'] - df['ret_5'].rolling(60).mean()) / df['ret_5'].rolling(60).std().replace(0, np.nan)
    df['z_return_15'] = (df['ret_15'] - df['ret_15'].rolling(60).mean()) / df['ret_15'].rolling(60).std().replace(0, np.nan)
    df['skew_60'] = df['ret_1'].rolling(60).skew()
    df['kurt_60'] = df['ret_1'].rolling(60).kurt()

    df['slope_15'] = _rolling_slope(df['close'], 15)
    df['slope_60'] = _rolling_slope(df['close'], 60)
    df['momentum_z_15'] = df['ret_1'].rolling(15).mean() / df['ret_1'].rolling(15).std().replace(0, np.nan)
    df['momentum_z_60'] = df['ret_1'].rolling(60).mean() / df['ret_1'].rolling(60).std().replace(0, np.nan)
    df['rsi_14'] = _rsi(df['close'], 14)
    df['rsi_60'] = _rsi(df['close'], 60)

    pv = df['close'] * df['volume']
    df['vwap_60'] = pv.rolling(60).sum() / df['volume'].rolling(60).sum().replace(0, np.nan)
    df['vwap_dist_60'] = (df['close'] - df['vwap_60']) / df['close'].rolling(60).std().replace(0, np.nan)
    df['range_position_60'] = (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min()).replace(0, np.nan)

    # Time-of-day cross-sectional features (causal: shift(1) before rolling)
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['date'] = df.index.normalize()
    df_xs = df[['minute_of_day', 'date', 'sigma_15', 'sigma_60', 'range', 'volume']].copy()
    df_xs['_ts'] = df.index
    for col in ['sigma_15', 'sigma_60', 'range', 'volume']:
        pv2 = df_xs.pivot_table(index='date', columns='minute_of_day', values=col, aggfunc='last')
        pv_20d = pv2.shift(1).rolling(20, min_periods=10).mean()
        pv_20d_std = pv2.shift(1).rolling(20, min_periods=10).std()
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

    df['hour_et'] = df.index.hour
    df['minute_of_hour'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day

    df = df.drop(columns=['date'])
    return df


class StdevMlStrategy(Strategy):
    """Live wiring for the σ-ML strategy. Single-position handled by bot framework."""

    def __init__(self):
        self.long_model = None
        self.short_model = None
        self.long_features = None
        self.short_features = None
        self.last_eval = None
        self._seed_bars = None
        self._load_models()
        self._load_seed_bars()

    def _load_models(self):
        try:
            bL = joblib.load(ART / 'iter12_1.5_1.0_L.pkl')
            bS = joblib.load(ART / 'iter12_1.5_1.0_S.pkl')
            self.long_model = bL['hgb']
            self.long_features = bL['features']
            self.short_model = bS['hgb']
            self.short_features = bS['features']
            logging.info(
                f"StdevMlStrategy: loaded models — LONG AUC val={bL['auc_val']:.3f} oos={bL['auc_oos']:.3f}, "
                f"SHORT AUC val={bS['auc_val']:.3f} oos={bS['auc_oos']:.3f}"
            )
        except Exception as exc:
            logging.error(f"StdevMlStrategy: model load failed — {exc}")

    def _load_seed_bars(self):
        """Read the tail of es_master_outrights-2.parquet once at startup so the
        warmup gate clears on the first on_bar even when the live buffer is short.

        Why: the warmup threshold is 4,800 1-min bars but the bot persists at most
        ~2,880 bars in price_history_ohlc. Without a seed, StdevMl waits ~5 trading
        days before firing. With it, the first on_bar sees seed (~8k) + live (~2.9k)
        and clears the gate immediately.

        Failure mode: if the parquet is missing or malformed, log a warning and set
        seed=None — the strategy gracefully falls back to the original
        wait-for-buffer behavior.
        """
        try:
            if not SEED_PARQUET.exists():
                logging.warning(
                    f"StdevMlStrategy: seed parquet not found at {SEED_PARQUET} — "
                    f"strategy will wait {WARMUP_BARS} bars to fill before firing"
                )
                return
            df = pd.read_parquet(SEED_PARQUET, columns=['open', 'high', 'low', 'close', 'volume'])
            df = df.iloc[-SEED_BARS:].copy()
            self._seed_bars = df
            logging.info(
                f"StdevMlStrategy: seed bars loaded — {len(df):,} rows "
                f"({df.index.min()} → {df.index.max()})"
            )
        except Exception as exc:
            logging.warning(f"StdevMlStrategy: seed-bar load failed — {exc}; "
                            f"strategy will wait {WARMUP_BARS} bars before firing")
            self._seed_bars = None

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        self.last_eval = None
        if self.long_model is None or self.short_model is None:
            return None
        # Splice seed bars in front of the live buffer when live is short.
        # Only path that exits with `warmup<N` is when neither seed nor live can
        # produce ≥ WARMUP_BARS combined.
        if df is not None and self._seed_bars is not None and len(df) < WARMUP_BARS:
            try:
                seed = self._seed_bars
                # Align timezones — pd.concat degrades DatetimeIndex to object
                # Index when the tz strings differ even if they're aliases
                # (e.g. 'US/Eastern' vs 'America/New_York'). Without this,
                # df.index.hour blows up downstream in feature-build.
                if df.index.tz is not None and seed.index.tz != df.index.tz:
                    seed = seed.tz_convert(df.index.tz)
                if df.empty:
                    df = seed.copy()
                else:
                    seed = seed[seed.index < df.index[0]]
                    df = pd.concat([seed, df], axis=0).sort_index()
                    df = df[~df.index.duplicated(keep='last')]
            except Exception as exc:
                logging.warning(f"StdevMlStrategy: seed splice failed — {exc}; "
                                f"falling back to live-only buffer")
        if df is None or len(df) < WARMUP_BARS:
            self.last_eval = {'decision': 'no_signal', 'reason': f'warmup<{WARMUP_BARS}'}
            return None
        ts = df.index[-1]
        # Window check (ET hour-of-day)
        if not (WIN_HOUR_START <= ts.hour < WIN_HOUR_END):
            self.last_eval = {'decision': 'no_signal', 'reason': 'outside_window'}
            return None
        try:
            tail = df.tail(WARMUP_BARS).copy()
            feat_df = _compute_features_live(tail)
        except Exception as exc:
            logging.error(f"StdevMlStrategy: feature build failed — {exc}")
            self.last_eval = {'decision': 'error', 'error': str(exc)}
            return None
        last = feat_df.iloc[-1]
        atr_60 = float(last.get('atr_60', np.nan))
        if not np.isfinite(atr_60) or atr_60 <= 0:
            self.last_eval = {'decision': 'no_signal', 'reason': 'atr_unavailable'}
            return None

        try:
            X_L = feat_df[self.long_features].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
            X_S = feat_df[self.short_features].iloc[[-1]].replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e6, 1e6).values.astype(np.float32)
            p_long = float(self.long_model.predict_proba(X_L)[0, 1])
            p_short = float(self.short_model.predict_proba(X_S)[0, 1])
        except Exception as exc:
            logging.error(f"StdevMlStrategy: predict failed — {exc}")
            self.last_eval = {'decision': 'error', 'error': str(exc)}
            return None

        side = 'LONG' if p_long >= p_short else 'SHORT'
        conf = max(p_long, p_short)

        self.last_eval = {
            'decision': 'evaluated',
            'p_long': p_long,
            'p_short': p_short,
            'side': side,
            'confidence': conf,
            'atr_60': atr_60,
            'threshold': THRESHOLD,
            'ts': ts.isoformat(),
        }

        if conf < THRESHOLD:
            self.last_eval['decision'] = 'no_signal'
            self.last_eval['reason'] = f'conf<{THRESHOLD}'
            return None

        tp_dist = float(round(TP_ATR_MULT * atr_60, 2))
        sl_dist = float(round(SL_ATR_MULT * atr_60, 2))
        # Floor: avoid sub-fee brackets
        tp_dist = max(tp_dist, 8.0)
        sl_dist = max(sl_dist, 5.0)

        size = int(CONFIG.get('STDEV_ML_SIZE', 1))
        self.last_eval['decision'] = 'signal'
        self.last_eval['tp_dist'] = tp_dist
        self.last_eval['sl_dist'] = sl_dist

        return {
            'strategy': 'StdevMlStrategy',
            'side': side,
            'tp_dist': tp_dist,
            'sl_dist': sl_dist,
            'size': size,
            'confidence': conf,
            'stdev_ml_p_long': p_long,
            'stdev_ml_p_short': p_short,
            'stdev_ml_atr_60': atr_60,
            'stdev_ml_threshold': THRESHOLD,
        }
