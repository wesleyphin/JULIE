import math
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from regime_manifold_engine import DEFAULT_MANIFOLD_CONFIG, RegimeManifoldEngine


REGIME_TO_ID = {
    "TREND_GEODESIC": 0,
    "CHOP_SPIRAL": 1,
    "DISPERSED": 2,
    "ROTATIONAL_TURBULENCE": 3,
}

MODEL_MANIFOLD_FEATURE_COLUMNS = [
    "manifold_R",
    "manifold_alignment",
    "manifold_smoothness",
    "manifold_stress",
    "manifold_dispersion",
    "manifold_risk_mult",
    "manifold_R_pct",
    "manifold_alignment_pct",
    "manifold_smoothness_pct",
    "manifold_stress_pct",
    "manifold_dispersion_pct",
]

POLICY_MANIFOLD_FEATURE_COLUMNS = [
    "manifold_side_bias",
    "manifold_no_trade",
    "manifold_regime_id",
    "manifold_allow_trend",
    "manifold_allow_mean_reversion",
    "manifold_allow_breakout",
    "manifold_allow_fade",
]

AUX_FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_15",
    "atr14",
    "atr14_z",
    "range_z",
    "vol_z",
    "vwap_dist_atr",
    "ema_spread",
    "ema_slope_20",
    "hour_sin",
    "hour_cos",
    "session_id",
]

FEATURE_COLUMNS = MODEL_MANIFOLD_FEATURE_COLUMNS + AUX_FEATURE_COLUMNS
EXPORT_COLUMNS = MODEL_MANIFOLD_FEATURE_COLUMNS + POLICY_MANIFOLD_FEATURE_COLUMNS + AUX_FEATURE_COLUMNS


def get_session_name(ts: pd.Timestamp) -> str:
    hour = int(ts.hour)
    if hour >= 18 or hour < 3:
        return "ASIA"
    if 3 <= hour < 8:
        return "LONDON"
    if 8 <= hour < 12:
        return "NY_AM"
    if 12 <= hour < 17:
        return "NY_PM"
    return "OFF"


def _session_id(ts: pd.Timestamp) -> int:
    sess = get_session_name(ts)
    if sess == "ASIA":
        return 0
    if sess == "LONDON":
        return 1
    if sess == "NY_AM":
        return 2
    if sess == "NY_PM":
        return 3
    return -1


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(10, window // 5)).mean()
    std = series.rolling(window, min_periods=max(10, window // 5)).std()
    out = (series - mean) / std.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    work = df.copy()
    work.columns = [str(c).lower() for c in work.columns]
    for col in ("open", "high", "low", "close"):
        if col not in work.columns:
            raise ValueError(f"Missing required column: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "volume" not in work.columns:
        work["volume"] = 0.0
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["open", "high", "low", "close"])
    work = work.sort_index()
    return work[["open", "high", "low", "close", "volume"]]


def build_aux_features(df: pd.DataFrame) -> pd.DataFrame:
    work = _normalize_ohlcv(df)
    if work.empty:
        return pd.DataFrame(columns=AUX_FEATURE_COLUMNS)

    close = work["close"]
    high = work["high"]
    low = work["low"]
    volume = work["volume"]

    ret_1 = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_5 = close.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_15 = close.pct_change(15).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean().ffill().fillna(0.0)
    atr14_z = _zscore(atr14, 200)

    bar_range = (high - low).abs()
    range_z = _zscore(bar_range, 200)
    vol_z = _zscore(volume, 200)

    idx = pd.DatetimeIndex(work.index)
    trading_day = (idx - pd.Timedelta(hours=18)).normalize()
    pv = close * volume
    cum_pv = pv.groupby(trading_day).cumsum()
    cum_vol = volume.groupby(trading_day).cumsum().replace(0.0, np.nan)
    vwap = (cum_pv / cum_vol).replace([np.inf, -np.inf], np.nan)
    vwap_dist_atr = ((close - vwap) / atr14.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema_spread = ((ema20 - ema50) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ema_slope_20 = ema20.pct_change(5).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    minutes = idx.hour.to_numpy(dtype=float) * 60.0 + idx.minute.to_numpy(dtype=float)
    angle = (2.0 * math.pi * minutes) / 1440.0
    hour_sin = np.sin(angle)
    hour_cos = np.cos(angle)
    session_id = np.array([_session_id(ts) for ts in idx], dtype=float)

    out = pd.DataFrame(
        {
            "ret_1": ret_1.to_numpy(dtype=float),
            "ret_5": ret_5.to_numpy(dtype=float),
            "ret_15": ret_15.to_numpy(dtype=float),
            "atr14": atr14.to_numpy(dtype=float),
            "atr14_z": atr14_z.to_numpy(dtype=float),
            "range_z": range_z.to_numpy(dtype=float),
            "vol_z": vol_z.to_numpy(dtype=float),
            "vwap_dist_atr": vwap_dist_atr.to_numpy(dtype=float),
            "ema_spread": ema_spread.to_numpy(dtype=float),
            "ema_slope_20": ema_slope_20.to_numpy(dtype=float),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "session_id": session_id,
        },
        index=work.index,
    )
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def meta_to_feature_dict(meta: Dict) -> Dict[str, float]:
    allow = meta.get("allow") or {}
    regime = str(meta.get("regime", "DISPERSED"))
    return {
        "manifold_R": float(meta.get("R", 0.0) or 0.0),
        "manifold_alignment": float(meta.get("alignment", 0.0) or 0.0),
        "manifold_smoothness": float(meta.get("smoothness", 0.0) or 0.0),
        "manifold_stress": float(meta.get("stress", 0.0) or 0.0),
        "manifold_dispersion": float(meta.get("dispersion", 0.0) or 0.0),
        "manifold_R_pct": float(meta.get("R_pct", 0.0) or 0.0),
        "manifold_alignment_pct": float(meta.get("alignment_pct", 0.0) or 0.0),
        "manifold_smoothness_pct": float(meta.get("smoothness_pct", 0.0) or 0.0),
        "manifold_stress_pct": float(meta.get("stress_pct", 0.0) or 0.0),
        "manifold_dispersion_pct": float(meta.get("dispersion_pct", 0.0) or 0.0),
        "manifold_side_bias": float(meta.get("side_bias", 0) or 0.0),
        "manifold_risk_mult": float(meta.get("risk_mult", 1.0) or 1.0),
        "manifold_no_trade": float(1.0 if bool(meta.get("no_trade", False)) else 0.0),
        "manifold_regime_id": float(REGIME_TO_ID.get(regime, -1)),
        "manifold_allow_trend": float(1.0 if bool(allow.get("trend", False)) else 0.0),
        "manifold_allow_mean_reversion": float(1.0 if bool(allow.get("mean_reversion", False)) else 0.0),
        "manifold_allow_breakout": float(1.0 if bool(allow.get("breakout", False)) else 0.0),
        "manifold_allow_fade": float(1.0 if bool(allow.get("fade", False)) else 0.0),
    }


def _manifold_lookback(cfg: Dict) -> int:
    vol_z_window = int(cfg.get("vol_z_window", 390) or 390)
    mom_window = int(cfg.get("mom_window", 60) or 60)
    min_bars = int(cfg.get("min_bars", 80) or 80)
    return max(vol_z_window + 20, mom_window + 20, min_bars, 120)


def build_training_feature_frame(
    df: pd.DataFrame,
    manifold_cfg: Optional[Dict] = None,
    log_every: int = 0,
) -> pd.DataFrame:
    features, _, _ = build_training_feature_frame_with_state(
        df,
        manifold_cfg=manifold_cfg,
        log_every=log_every,
    )
    return features


def build_training_feature_frame_with_state(
    df: pd.DataFrame,
    manifold_cfg: Optional[Dict] = None,
    log_every: int = 0,
    initial_state: Optional[Dict] = None,
    start_after: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict, int]:
    work = _normalize_ohlcv(df)
    if work.empty:
        engine = RegimeManifoldEngine(manifold_cfg)
        if isinstance(initial_state, dict):
            engine.load_state(initial_state)
        return pd.DataFrame(columns=EXPORT_COLUMNS), engine.get_state(), _manifold_lookback(engine.cfg)

    merged_cfg = dict(DEFAULT_MANIFOLD_CONFIG)
    if isinstance(manifold_cfg, dict):
        merged_cfg.update(manifold_cfg)
    engine = RegimeManifoldEngine(merged_cfg)
    if isinstance(initial_state, dict):
        engine.load_state(initial_state)
    lookback = _manifold_lookback(merged_cfg)
    aux = build_aux_features(work)
    progress_every = int(log_every) if int(log_every) > 0 else max(1000, len(work) // 20)
    start_after_ts = pd.Timestamp(start_after) if start_after is not None else None
    logging.info(
        "Manifold feature build start: bars=%d lookback=%d progress_every=%d",
        len(work),
        lookback,
        progress_every,
    )

    rows = []
    row_index = []
    for i in range(len(work)):
        start = 0 if i < lookback else (i - lookback + 1)
        hist = work.iloc[start : i + 1]
        ts = work.index[i]
        if start_after_ts is not None and pd.Timestamp(ts) <= start_after_ts:
            continue
        meta = engine.update(hist, ts=ts, session=get_session_name(pd.Timestamp(ts)))
        reason = None
        debug = meta.get("debug")
        if isinstance(debug, dict):
            reason = debug.get("reason")
        if reason == "warmup":
            continue

        row = meta_to_feature_dict(meta)
        aux_row = aux.iloc[i]
        for col in AUX_FEATURE_COLUMNS:
            row[col] = float(aux_row.get(col, 0.0) or 0.0)
        rows.append(row)
        row_index.append(ts)

        if i > 0 and (i % progress_every == 0):
            pct = (100.0 * float(i)) / float(len(work))
            logging.info("Manifold feature build progress: %d/%d (%.1f%%)", i, len(work), pct)

    if not rows:
        return pd.DataFrame(columns=EXPORT_COLUMNS), engine.get_state(), int(lookback)

    out = pd.DataFrame(rows, index=pd.DatetimeIndex(row_index))
    out = out.reindex(columns=EXPORT_COLUMNS).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, engine.get_state(), int(lookback)


def build_live_feature_row(
    df: pd.DataFrame,
    engine: RegimeManifoldEngine,
    ts: Optional[pd.Timestamp] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    work = _normalize_ohlcv(df)
    if work.empty:
        return None, None
    if ts is None:
        ts = pd.Timestamp(work.index[-1])
    else:
        ts = pd.Timestamp(ts)
    meta = engine.update(work, ts=ts, session=get_session_name(ts))
    aux = build_aux_features(work)
    if aux.empty:
        return None, meta
    row = meta_to_feature_dict(meta)
    aux_last = aux.iloc[-1]
    for col in AUX_FEATURE_COLUMNS:
        row[col] = float(aux_last.get(col, 0.0) or 0.0)
    x = pd.DataFrame([row], index=[work.index[-1]])
    x = x.reindex(columns=FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, meta
