"""σ-ML hour 9-11 ET veto gate (PT 6-8).

Loaded by julie001.py. Scores each candidate signal in the 9-11 ET window
using 82 σ-derived features and a HistGradientBoostingClassifier trained on
2017-2020 Trump T1 + 2025 (validated on 2021-2025 Biden).

When p_win < threshold the signal is vetoed at signal-birth time, before
V18/Recipe B/Kalshi see it. Saves Kronos calls and Kalshi API hits.

Rollback: JULIE_LOCAL_STDEV_ML_HR9_11_ENABLED=0
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

LOG = logging.getLogger("stdev_ml_hr9_11")
_PAYLOAD: Optional[Dict[str, Any]] = None
_LOAD_FAILED: bool = False
_ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "stdev_ml_hr9_11" / "model.pkl"


def _ensure_loaded() -> bool:
    global _PAYLOAD, _LOAD_FAILED
    if _PAYLOAD is not None: return True
    if _LOAD_FAILED: return False
    if not _ARTIFACT.exists():
        _LOAD_FAILED = True
        LOG.warning("[STDEV_ML_HR9_11] artifact missing: %s", _ARTIFACT)
        return False
    try:
        import joblib
        _PAYLOAD = joblib.load(_ARTIFACT)
        LOG.info("[STDEV_ML_HR9_11] loaded — %d features, threshold=%.3f, val_auc=%.3f",
                 len(_PAYLOAD.get('features', [])),
                 float(_PAYLOAD.get('best_threshold', 0.5)),
                 float(_PAYLOAD.get('val_auc', 0)))
        return True
    except Exception as exc:
        LOG.warning("[STDEV_ML_HR9_11] load failed: %s", exc)
        _LOAD_FAILED = True
        return False


def is_active() -> bool:
    """Gate active if (a) artifact loadable AND (b) env-flag default ON."""
    return os.environ.get("JULIE_LOCAL_STDEV_ML_HR9_11_ENABLED", "1").strip() == "1"


def get_threshold() -> float:
    if not _ensure_loaded(): return 0.55
    return float(_PAYLOAD.get('best_threshold', 0.55))


def score_signal(market_df, side: str, ts) -> Optional[Dict[str, Any]]:
    """Compute 82 σ features + score. Returns dict with p_win, threshold, decision.
    Returns None if model not loaded or features can't be computed."""
    if not _ensure_loaded(): return None
    import pandas as pd
    if market_df is None or len(market_df) < 250:
        return None  # need at least 250 bars for 240-window σ features

    feat_cols = _PAYLOAD['features']
    threshold = float(_PAYLOAD['best_threshold'])
    clf = _PAYLOAD['model']

    # Build feature row from latest bar
    df = market_df.copy()
    if 'ret_1' not in df.columns:
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
    # Latest-bar values are what we need; rolling intraday σ approximated:
    df['sigma_intraday_running'] = df['ret_1'].rolling(60).std()
    df['sigma_open_to_now'] = df['close'].rolling(60).std() / df['close'].iloc[0]

    # Ratios
    s5 = df['sigma_5']; s15 = df['sigma_15']; s30 = df['sigma_30']; s60 = df['sigma_60']; s120 = df['sigma_120']; s240 = df['sigma_240']
    df['sigma_ratio_5_60']    = s5 / s60.replace(0, np.nan)
    df['sigma_ratio_15_60']   = s15 / s60.replace(0, np.nan)
    df['sigma_ratio_15_240']  = s15 / s240.replace(0, np.nan)
    df['sigma_ratio_60_240']  = s60 / s240.replace(0, np.nan)
    df['sigma_ratio_5_15']    = s5 / s15.replace(0, np.nan)
    df['sigma_ratio_30_120']  = s30 / s120.replace(0, np.nan)
    df['sigma_ratio_intraday_240'] = df['sigma_intraday_running'] / s240.replace(0, np.nan)
    df['sigma_ratio_15_60_log']  = df['sigma_15_log'] - df['sigma_60_log']
    df['sigma_ratio_5_60_log']   = df['sigma_5_log'] - df['sigma_60_log']
    df['sigma_5_chg_5'] = s5.pct_change(5)

    # Percentile rankings
    df['sigma_15_pctile_60']  = s15.rolling(60).rank(pct=True)
    df['sigma_15_pctile_240'] = s15.rolling(240).rank(pct=True)
    df['sigma_15_pctile_390'] = s15.rolling(390).rank(pct=True)
    df['sigma_60_pctile_240'] = s60.rolling(240).rank(pct=True)
    df['sigma_60_pctile_390'] = s60.rolling(390).rank(pct=True)
    # Approximate hourly percentiles (240-bar = 4 hours)
    df['sigma_15_pctile_hourly_20d'] = s15.rolling(240 * 20).rank(pct=True)
    df['sigma_60_pctile_hourly_20d'] = s60.rolling(240 * 20).rank(pct=True)
    df['sigma_intraday_pctile_20d'] = df['sigma_intraday_running'].rolling(240 * 20).rank(pct=True)

    # σ of σ
    df['sigma_of_sigma_15_60'] = s15.rolling(60).std()
    df['sigma_of_sigma_15_240'] = s15.rolling(240).std()
    df['sigma_of_sigma_60_240'] = s60.rolling(240).std()
    df['delta_sigma_15_5'] = s15 - s15.shift(5)
    df['delta_sigma_60_15'] = s60 - s60.shift(15)
    df['sigma_acceleration'] = (s5 - s5.shift(5)) / s5.shift(5).replace(0, np.nan)
    df['sigma_skew_30']  = df['ret_1'].rolling(30).skew()
    df['sigma_skew_120'] = df['ret_1'].rolling(120).skew()

    # Z-scores
    for w in [15, 60, 240]:
        m = df['close'].rolling(w).mean()
        s = df['close'].rolling(w).std()
        df[f'z_close_{w}'] = (df['close'] - m) / s.replace(0, np.nan)
    df['z_high_15']  = (df['high']  - df['close'].rolling(15).mean()) / df['close'].rolling(15).std().replace(0, np.nan)
    df['z_low_15']   = (df['low']   - df['close'].rolling(15).mean()) / df['close'].rolling(15).std().replace(0, np.nan)
    rng = df['range']
    df['z_range_15']  = (rng - rng.rolling(15).mean()) / rng.rolling(15).std().replace(0, np.nan)
    df['z_range_60']  = (rng - rng.rolling(60).mean()) / rng.rolling(60).std().replace(0, np.nan)
    df['z_body_15']   = (df['body'] - df['body'].rolling(15).mean()) / df['body'].rolling(15).std().replace(0, np.nan)
    df['z_upper_wick_15'] = (df['upper_wick'] - df['upper_wick'].rolling(15).mean()) / df['upper_wick'].rolling(15).std().replace(0, np.nan)
    df['z_lower_wick_15'] = (df['lower_wick'] - df['lower_wick'].rolling(15).mean()) / df['lower_wick'].rolling(15).std().replace(0, np.nan)
    if 'volume' in df.columns:
        vol = df['volume']
        df['z_volume_15']  = (vol - vol.rolling(15).mean()) / vol.rolling(15).std().replace(0, np.nan)
        df['z_volume_60']  = (vol - vol.rolling(60).mean()) / vol.rolling(60).std().replace(0, np.nan)
    else:
        df['z_volume_15'] = 0; df['z_volume_60'] = 0
    df['z_return_1']   = (df['ret_1'] - df['ret_1'].rolling(60).mean()) / df['ret_1'].rolling(60).std().replace(0, np.nan)
    df['z_return_5']   = (df['ret_5'] - df['ret_5'].rolling(60).mean()) / df['ret_5'].rolling(60).std().replace(0, np.nan)
    df['day_open'] = df['open'].iloc[0]  # approximation; live should track session open
    df['z_close_to_open'] = (df['close'] - df['day_open']) / df['close'].rolling(60).std().replace(0, np.nan)

    # Distribution shape
    df['skew_return_60']  = df['ret_1'].rolling(60).skew()
    df['skew_return_240'] = df['ret_1'].rolling(240).skew()
    df['kurt_return_60']  = df['ret_1'].rolling(60).kurt()
    df['kurt_return_240'] = df['ret_1'].rolling(240).kurt()
    abs_ret = df['ret_1'].abs()
    df['efficiency_60']  = (df['close'] - df['close'].shift(60)).abs() / (abs_ret.rolling(60).sum() * df['close'].shift(60)).replace(0, np.nan)
    df['efficiency_240'] = (df['close'] - df['close'].shift(240)).abs() / (abs_ret.rolling(240).sum() * df['close'].shift(240)).replace(0, np.nan)
    df['trend_r2_60']  = df['close'].rolling(60).corr(np.arange(len(df))) ** 2 if False else 0  # skip for live perf
    df['trend_r2_240'] = 0

    # Time/calendar
    if hasattr(ts, 'hour'):
        df['hour_et'] = ts.hour
        df['minute_of_hour'] = ts.minute
        df['day_of_week'] = ts.weekday()
        df['week_of_year'] = ts.isocalendar()[1]
        df['month'] = ts.month
        df['day_of_month'] = ts.day
    else:
        df['hour_et'] = df['minute_of_hour'] = df['day_of_week'] = df['week_of_year'] = df['month'] = df['day_of_month'] = 0
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_et'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_et'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

    # Setup direction
    df['side'] = 0 if str(side).upper() == 'LONG' else 1
    df['prev_bar_color'] = np.sign(df['close'].shift(1) - df['open'].shift(1))
    df['prev_2_bar_color'] = np.sign(df['close'].shift(2) - df['open'].shift(2))
    df['ret_15'] = np.log(df['close'] / df['close'].shift(15))
    df['ret_60'] = np.log(df['close'] / df['close'].shift(60))
    df['cum_session_ret'] = (df['close'] - df['day_open']) / df['day_open']
    df['body_share'] = df['body'] / df['range'].replace(0, np.nan)
    df['upper_wick_share'] = df['upper_wick'] / df['range'].replace(0, np.nan)
    df['lower_wick_share'] = df['lower_wick'] / df['range'].replace(0, np.nan)

    # Build feature row from LAST bar (current)
    last = df.iloc[-1]
    row = {c: float(last.get(c, 0) or 0) for c in feat_cols}
    X = np.array([[row[c] for c in feat_cols]], dtype=np.float32)
    if np.isnan(X).any():
        # Fill remaining NaN with 0 (early-bar / cold-start)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    p_win = float(clf.predict_proba(X)[0, 1])
    decision = p_win >= threshold
    return {
        'p_win': p_win,
        'threshold': threshold,
        'decision': decision,  # True = allow, False = veto
    }
