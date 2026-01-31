import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import CONFIG
from dynamic_sltp_params import SESSION_DEFAULTS, STRATEGY_PARAMS
from ml_physics_strategy import (
    ML_FEATURE_COLUMNS,
    OPTIONAL_FEATURE_COLUMNS,
    ml_calculate_adx,
    ml_calculate_atr,
    ml_calculate_rsi,
    ml_calculate_rvol,
    ml_calculate_slope,
    ml_encode_cyclical_time,
    ml_zscore_normalize,
    RSI_PERIOD,
    ATR_PERIOD,
    ADX_PERIOD,
    ZSCORE_WINDOW,
    RVOL_LOOKBACK_DAYS,
)
from volatility_filter import (
    VOLATILITY_HIERARCHY,
    COARSE_VOLATILITY_HIERARCHY,
    COARSE_VOLATILITY_HIERARCHY_NO_Q,
    SESSION_FALLBACKS,
    VolRegime,
    volatility_filter,
)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HistGradientBoostingClassifier = None

FALLBACK_PARAMS = {"sl_mult": 2.0, "tp_mult": 2.5, "atr_med": 1.5}

# Silence sklearn calibration FutureWarning about cv='prefit'
warnings.filterwarnings(
    "ignore",
    message=r".*cv='prefit'.*deprecated.*",
    category=FutureWarning,
)

_VOL_THRESHOLDS_OVERRIDE = None
_VOL_SESSION_FALLBACKS_OVERRIDE = None
_VOL_THRESHOLDS_META = None
_VOL_THRESHOLDS_LOADED = False


def _load_csv(csv_path: Path) -> pd.DataFrame:
    with csv_path.open("r", errors="ignore") as f:
        first = f.readline()
        second = f.readline()
        needs_skip = "Time Series" in first and "Date" in second

    df = pd.read_csv(csv_path, skiprows=1 if needs_skip else 0)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = None
    if "ts_event" in df.columns:
        date_col = "ts_event"
    elif "timestamp" in df.columns:
        date_col = "timestamp"
    elif "date" in df.columns:
        date_col = "date"
    if date_col is None:
        raise ValueError("CSV missing timestamp column")

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', "").str.replace(",", "")
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df.dropna(subset=["timestamp"], inplace=True)
    df.set_index("timestamp", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def _resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty or minutes <= 1:
        return df
    rule = f"{int(minutes)}min"
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def _drop_gap_days(df: pd.DataFrame, max_gap_minutes: float) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    if df.empty or max_gap_minutes <= 0:
        return df, []

    idx = df.index
    day_index = pd.Series(idx.normalize(), index=idx)
    idx_series = pd.Series(idx, index=idx)
    gaps = idx_series.diff()
    gaps[day_index != day_index.shift()] = pd.Timedelta(0)

    maint_cfg = CONFIG.get("TRAINING_MAINTENANCE_WINDOW", {}) or {}
    start_str = str(maint_cfg.get("start", "17:00"))
    end_str = str(maint_cfg.get("end", "18:00"))
    try:
        tol_minutes = float(maint_cfg.get("tolerance_minutes", 0.0) or 0.0)
    except Exception:
        tol_minutes = 0.0

    def _parse_time(value: str) -> pd.Timedelta:
        try:
            parts = value.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
        except Exception:
            hour, minute = 17, 0
        return pd.Timedelta(hours=hour, minutes=minute)

    mw_start = _parse_time(start_str) - pd.Timedelta(minutes=tol_minutes)
    mw_end = _parse_time(end_str) + pd.Timedelta(minutes=tol_minutes)

    prev_ts = idx_series.shift(1)
    curr_ts = idx_series
    same_day = day_index == day_index.shift()
    adjusted_gap = gaps.copy()
    if mw_start < mw_end:
        overlaps = []
        for start, end, same in zip(prev_ts, curr_ts, same_day):
            if not same or pd.isna(start) or pd.isna(end):
                overlaps.append(pd.Timedelta(0))
                continue
            day = end.normalize()
            window_start = day + mw_start
            window_end = day + mw_end
            if end <= window_start or start >= window_end:
                overlaps.append(pd.Timedelta(0))
                continue
            overlap = min(end, window_end) - max(start, window_start)
            if overlap < pd.Timedelta(0):
                overlap = pd.Timedelta(0)
            overlaps.append(overlap)
        overlap_series = pd.Series(overlaps, index=idx_series.index)
        adjusted_gap = gaps - overlap_series
        adjusted_gap = adjusted_gap.clip(lower=pd.Timedelta(0))

    gap_minutes = adjusted_gap.dt.total_seconds().fillna(0) / 60.0
    max_gap_by_day = gap_minutes.groupby(day_index).max()
    drop_days = max_gap_by_day[max_gap_by_day > max_gap_minutes].index
    if drop_days.empty:
        return df, []
    filtered = df[~day_index.isin(drop_days)]
    return filtered, list(drop_days)


def _resolve_session_settings(args, session_name: str) -> Dict:
    """Build per-session training settings with CONFIG presets overriding CLI defaults."""
    settings = {
        "timeframe_minutes": args.timeframe_minutes,
        "horizon_bars": args.horizon_bars,
        "label_mode": args.label_mode,
        "drop_neutral": args.drop_neutral,
        "sltp_strategy": args.sltp_strategy,
        "thr_min": args.thr_min,
        "thr_max": args.thr_max,
        "thr_step": args.thr_step,
        "drop_gap_minutes": args.drop_gap_minutes,
        "max_rows": args.max_rows,
    }
    presets = CONFIG.get("ML_PHYSICS_TRAINING_PRESETS", {}) or {}
    preset = presets.get(session_name, {}) or {}
    for key, value in preset.items():
        settings[key] = value
    return settings


def _session_from_hours(hours: np.ndarray) -> np.ndarray:
    session = np.full(len(hours), "CLOSED", dtype=object)
    session[(hours >= 18) | (hours < 3)] = "ASIA"
    session[(hours >= 3) & (hours < 8)] = "LONDON"
    session[(hours >= 8) & (hours < 12)] = "NY_AM"
    session[(hours >= 12) & (hours < 17)] = "NY_PM"
    return session


def _yearly_quarter(months: np.ndarray) -> np.ndarray:
    return np.select(
        [months <= 3, months <= 6, months <= 9, months <= 12],
        ["Q1", "Q2", "Q3", "Q4"],
        default="Q4",
    )


def _monthly_quarter(days: np.ndarray) -> np.ndarray:
    return np.select(
        [days <= 7, days <= 14, days <= 21, days <= 31],
        ["W1", "W2", "W3", "W4"],
        default="W4",
    )


def _dow_name(dows: np.ndarray) -> np.ndarray:
    names = np.array(["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"], dtype=object)
    return names[dows]


def _build_hierarchy_keys(index: pd.DatetimeIndex, sessions: np.ndarray) -> np.ndarray:
    months = index.month.to_numpy()
    days = index.day.to_numpy()
    dows = index.weekday.to_numpy()
    yq = _yearly_quarter(months)
    mq = _monthly_quarter(days)
    dow = _dow_name(dows)
    keys = np.char.add(np.char.add(np.char.add(np.char.add(yq, "_"), mq), "_"), dow)
    keys = np.char.add(np.char.add(keys, "_"), sessions)
    return keys


def _build_coarse_keys(index: pd.DatetimeIndex, sessions: np.ndarray, include_quarter: bool) -> np.ndarray:
    months = index.month.to_numpy()
    dows = index.weekday.to_numpy()
    dow = _dow_name(dows)
    if include_quarter:
        yq = _yearly_quarter(months)
        keys = np.char.add(np.char.add(yq, "_"), dow)
        keys = np.char.add(np.char.add(keys, "_"), sessions)
        return keys
    keys = np.char.add(np.char.add(dow, "_"), sessions)
    return keys


def _normalize_threshold_entry(entry: Dict) -> Dict:
    if not isinstance(entry, dict):
        return {}
    norm = {}
    for key in ("p10", "p25", "p50", "median", "p75"):
        if key in entry:
            try:
                norm[key] = float(entry[key])
            except Exception:
                continue
    if "median" not in norm and "p50" in norm:
        norm["median"] = norm["p50"]
    if "p50" not in norm and "median" in norm:
        norm["p50"] = norm["median"]
    if "p10" in norm and "p25" in norm and "p75" in norm:
        return {
            "p10": norm["p10"],
            "p25": norm["p25"],
            "median": norm.get("median"),
            "p75": norm["p75"],
        }
    return {}


def _load_volatility_thresholds_from_file() -> None:
    global _VOL_THRESHOLDS_OVERRIDE
    global _VOL_SESSION_FALLBACKS_OVERRIDE
    global _VOL_THRESHOLDS_META
    global _VOL_THRESHOLDS_LOADED

    if _VOL_THRESHOLDS_LOADED:
        return
    _VOL_THRESHOLDS_LOADED = True

    file_name = str(CONFIG.get("VOLATILITY_THRESHOLDS_FILE", "") or "").strip()
    if not file_name:
        return
    path = Path(file_name)
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            path = cwd_path
        else:
            alt_path = Path(__file__).resolve().parent / path
            if alt_path.exists():
                path = alt_path
    if not path.exists():
        return

    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        logging.warning("Failed to read volatility thresholds from %s: %s", path, exc)
        return

    if not isinstance(payload, dict):
        logging.warning("Failed to read volatility thresholds from %s: invalid payload", path)
        return

    thresholds_raw = payload.get("thresholds", {}) or {}
    session_raw = payload.get("session_fallbacks", payload.get("session_fallback", {})) or {}
    meta = payload.get("meta", {}) or {}

    thresholds = {}
    for key, entry in thresholds_raw.items():
        normalized = _normalize_threshold_entry(entry)
        if normalized:
            thresholds[key] = normalized

    session_fallbacks = {}
    for key, entry in session_raw.items():
        normalized = _normalize_threshold_entry(entry)
        if normalized:
            session_fallbacks[key] = normalized

    if thresholds:
        _VOL_THRESHOLDS_OVERRIDE = thresholds
    if session_fallbacks:
        _VOL_SESSION_FALLBACKS_OVERRIDE = session_fallbacks
    if meta:
        _VOL_THRESHOLDS_META = meta

    logging.info(
        "Loaded volatility thresholds from %s (keys=%d, session_fallbacks=%d)",
        path,
        len(thresholds),
        len(session_fallbacks),
    )


def _compute_session_std_values(
    df: pd.DataFrame,
    sessions: np.ndarray,
    default_window: int,
    session_windows: Dict,
) -> np.ndarray:
    returns = df["close"].pct_change()
    std_default = returns.rolling(default_window).std().fillna(0.0).to_numpy()
    std_values = std_default.copy()

    for session_name, window in (session_windows or {}).items():
        try:
            window = int(window)
        except Exception:
            continue
        if window <= 0 or window == default_window:
            continue
        mask = sessions == session_name
        if not mask.any():
            continue
        std_session = returns.rolling(window).std().fillna(0.0).to_numpy()
        std_values[mask] = std_session[mask]

    return std_values


def _build_volatility_thresholds(
    df: pd.DataFrame,
    min_samples: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    if df.empty:
        return {}, {}

    index = df.index
    sessions = _session_from_hours(index.hour.to_numpy())

    mode_cfg = CONFIG.get("VOLATILITY_HIERARCHY_MODE", {}) or {}
    mode = str(mode_cfg.get("mode", "full") or "full").lower()
    include_quarter = bool(mode_cfg.get("include_quarter", True))

    if mode == "coarse":
        keys = _build_coarse_keys(index, sessions, include_quarter)
    else:
        keys = _build_hierarchy_keys(index, sessions)

    std_cfg = CONFIG.get("VOLATILITY_STD_WINDOWS", {}) or {}
    default_window = int(std_cfg.get("default", getattr(volatility_filter, "std_window", 20)) or 20)
    session_windows = std_cfg.get("sessions", {}) or {}

    std_values = _compute_session_std_values(df, sessions, default_window, session_windows)
    std_mask = std_values > 0
    if not np.any(std_mask):
        return {}, {}

    vol_df = pd.DataFrame({
        "key": keys,
        "session": sessions,
        "std": std_values,
    })
    vol_df = vol_df[std_mask]

    thresholds = {}
    for key, group in vol_df.groupby("key")["std"]:
        if len(group) < min_samples:
            continue
        thresholds[key] = {
            "p10": float(group.quantile(0.10)),
            "p25": float(group.quantile(0.25)),
            "median": float(group.quantile(0.50)),
            "p75": float(group.quantile(0.75)),
        }

    session_fallbacks = {}
    for session_name, group in vol_df.groupby("session")["std"]:
        if len(group) < min_samples:
            continue
        session_fallbacks[session_name] = {
            "p10": float(group.quantile(0.10)),
            "p25": float(group.quantile(0.25)),
            "median": float(group.quantile(0.50)),
            "p75": float(group.quantile(0.75)),
        }

    return thresholds, session_fallbacks


def _compute_vol_regimes(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    index = df.index
    sessions = _session_from_hours(index.hour.to_numpy())
    _load_volatility_thresholds_from_file()

    mode_cfg = CONFIG.get("VOLATILITY_HIERARCHY_MODE", {}) or {}
    mode = str(mode_cfg.get("mode", "full") or "full").lower()
    include_quarter = bool(mode_cfg.get("include_quarter", True))

    if isinstance(_VOL_THRESHOLDS_META, dict):
        meta_mode = _VOL_THRESHOLDS_META.get("mode")
        meta_include = _VOL_THRESHOLDS_META.get("include_quarter")
        if meta_mode:
            mode = str(meta_mode).lower()
        if isinstance(meta_include, bool):
            include_quarter = meta_include

    if mode == "coarse":
        keys = _build_coarse_keys(index, sessions, include_quarter)
        thresholds_map = _VOL_THRESHOLDS_OVERRIDE or (
            COARSE_VOLATILITY_HIERARCHY if include_quarter else COARSE_VOLATILITY_HIERARCHY_NO_Q
        )
    else:
        keys = _build_hierarchy_keys(index, sessions)
        thresholds_map = _VOL_THRESHOLDS_OVERRIDE or VOLATILITY_HIERARCHY

    std_cfg = CONFIG.get("VOLATILITY_STD_WINDOWS", {}) or {}
    if isinstance(_VOL_THRESHOLDS_META, dict):
        meta_std = _VOL_THRESHOLDS_META.get("std_windows")
        if isinstance(meta_std, dict):
            std_cfg = meta_std
    default_window = int(std_cfg.get("default", getattr(volatility_filter, "std_window", 20)) or 20)
    session_windows = std_cfg.get("sessions", {}) or {}

    std_values = _compute_session_std_values(df, sessions, default_window, session_windows)

    default_thresholds = {"p10": 0.00008, "p25": 0.00012, "p75": 0.00028}
    session_fallbacks = _VOL_SESSION_FALLBACKS_OVERRIDE or SESSION_FALLBACKS

    regimes = np.empty(len(df), dtype=object)
    for i, (key, sess) in enumerate(zip(keys, sessions)):
        current_std = std_values[i]
        thresholds = thresholds_map.get(key) or session_fallbacks.get(sess) or default_thresholds
        if current_std < thresholds["p10"]:
            regimes[i] = VolRegime.ULTRA_LOW
        elif current_std < thresholds["p25"]:
            regimes[i] = VolRegime.LOW
        elif current_std > thresholds["p75"]:
            regimes[i] = VolRegime.HIGH
        else:
            regimes[i] = VolRegime.NORMAL
    return pd.Series(regimes, index=index)


def _build_sltp_arrays(df: pd.DataFrame, strategy_name: str) -> Tuple[np.ndarray, np.ndarray]:
    index = df.index
    sessions = _session_from_hours(index.hour.to_numpy())
    strat_params = STRATEGY_PARAMS.get(strategy_name)
    if strat_params:
        keys = _build_hierarchy_keys(index, sessions)
        params = [
            strat_params.get(key)
            or SESSION_DEFAULTS.get(sess)
            or FALLBACK_PARAMS
            for key, sess in zip(keys, sessions)
        ]
    else:
        params = [
            SESSION_DEFAULTS.get(sess) or FALLBACK_PARAMS
            for sess in sessions
        ]

    sl_mult = np.array([p["sl_mult"] for p in params], dtype=float)
    tp_mult = np.array([p["tp_mult"] for p in params], dtype=float)
    atr_med = np.array([p["atr_med"] for p in params], dtype=float)

    atr = (df["high"] - df["low"]).rolling(14).mean().to_numpy()
    atr = np.where(np.isnan(atr), atr_med, atr)

    sl_dist = np.round(atr * sl_mult * 4.0) / 4.0
    tp_dist = np.round(atr * tp_mult * 4.0) / 4.0
    sl_dist = np.maximum(sl_dist, 4.0)
    tp_dist = np.maximum(tp_dist, 6.0)
    return sl_dist, tp_dist


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    w_df = df.copy()
    w_df["datetime"] = pd.to_datetime(w_df.index)
    try:
        if w_df["datetime"].dt.tz is None:
            w_df["datetime"] = w_df["datetime"].dt.tz_localize("US/Eastern")
        else:
            w_df["datetime"] = w_df["datetime"].dt.tz_convert("US/Eastern")
    except Exception:
        pass

    w_df["RSI"] = ml_calculate_rsi(w_df["close"], period=RSI_PERIOD)
    w_df["ATR"] = ml_calculate_atr(w_df["high"], w_df["low"], w_df["close"], period=ATR_PERIOD)
    adx, plus_di, minus_di = ml_calculate_adx(
        w_df["high"], w_df["low"], w_df["close"], period=ADX_PERIOD
    )
    w_df["ADX"] = adx
    w_df["PLUS_DI"] = plus_di
    w_df["MINUS_DI"] = minus_di

    w_df["RSI_Vel"] = w_df["RSI"].diff(3)
    w_df["Adx_Vel"] = w_df["ADX"].diff(3)
    w_df["Slope"] = ml_calculate_slope(w_df["close"], window=15)
    w_df["Volatility"] = w_df["close"].rolling(window=15).std()
    w_df["Range"] = w_df["high"] - w_df["low"]
    w_df["Return_1"] = w_df["close"].pct_change(1)
    w_df["Return_5"] = w_df["close"].pct_change(5)
    w_df["Return_15"] = w_df["close"].pct_change(15)

    w_df["Close_ZScore"] = ml_zscore_normalize(w_df["close"], window=ZSCORE_WINDOW)
    w_df["High_ZScore"] = ml_zscore_normalize(w_df["high"], window=ZSCORE_WINDOW)
    w_df["Low_ZScore"] = ml_zscore_normalize(w_df["low"], window=ZSCORE_WINDOW)
    w_df["ATR_ZScore"] = ml_zscore_normalize(w_df["ATR"], window=ZSCORE_WINDOW)
    w_df["Volatility_ZScore"] = ml_zscore_normalize(w_df["Volatility"], window=ZSCORE_WINDOW)
    w_df["Range_ZScore"] = ml_zscore_normalize(w_df["Range"], window=ZSCORE_WINDOW)
    w_df["Volume_ZScore"] = ml_zscore_normalize(w_df["volume"], window=ZSCORE_WINDOW)
    w_df["RSI_Norm"] = (w_df["RSI"] - 50.0) / 50.0
    w_df["ADX_Norm"] = (w_df["ADX"] / 50.0) - 1.0
    w_df["Slope_ZScore"] = ml_zscore_normalize(w_df["Slope"], window=ZSCORE_WINDOW)
    w_df["RVol"] = ml_calculate_rvol(w_df["volume"], w_df["datetime"], lookback_days=RVOL_LOOKBACK_DAYS)
    w_df["RVol_ZScore"] = ml_zscore_normalize(w_df["RVol"], window=ZSCORE_WINDOW)

    (w_df["Hour_Sin"], w_df["Hour_Cos"],
     w_df["Minute_Sin"], w_df["Minute_Cos"],
     w_df["DOW_Sin"], w_df["DOW_Cos"]) = ml_encode_cyclical_time(w_df["datetime"])

    w_df["Is_Trending"] = (w_df["ADX"] > 25).astype(int)
    w_df["Trend_Direction"] = np.sign(w_df["PLUS_DI"] - w_df["MINUS_DI"])
    w_df["ATR_MA"] = w_df["ATR"].rolling(window=100).mean()
    w_df["High_Volatility"] = (w_df["ATR"] > w_df["ATR_MA"] * 1.5).astype(int)

    features = w_df[ML_FEATURE_COLUMNS].copy()
    for col in OPTIONAL_FEATURE_COLUMNS:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)
    features = features.dropna()
    return features


def _label_barrier(df: pd.DataFrame, sl_dist: np.ndarray, tp_dist: np.ndarray, horizon: int, drop_neutral: bool) -> pd.Series:
    n = len(df)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    labels = np.full(n, np.nan)

    max_i = n - horizon - 1
    for i in range(max_i):
        up = closes[i] + tp_dist[i]
        dn = closes[i] - sl_dist[i]
        hit = None
        for j in range(1, horizon + 1):
            hi = highs[i + j]
            lo = lows[i + j]
            if hi >= up and lo <= dn:
                hit = "both"
                break
            if hi >= up:
                labels[i] = 1
                hit = "tp"
                break
            if lo <= dn:
                labels[i] = 0
                hit = "sl"
                break
        if hit is None or hit == "both":
            if not drop_neutral:
                labels[i] = 1 if closes[i + horizon] > closes[i] else 0

    return pd.Series(labels, index=df.index)


def _build_labels(df: pd.DataFrame, horizon: int, label_mode: str, drop_neutral: bool,
                  sl_dist: np.ndarray, tp_dist: np.ndarray) -> pd.Series:
    if label_mode == "barrier":
        return _label_barrier(df, sl_dist, tp_dist, horizon, drop_neutral)

    future_close = df["close"].shift(-horizon)
    future_return = (future_close / df["close"]) - 1.0

    if label_mode == "atr":
        atr = ml_calculate_atr(df["high"], df["low"], df["close"], period=ATR_PERIOD)
        thresh = (atr * 0.5) / df["close"]
        if drop_neutral:
            mask = (future_return > thresh) | (future_return < -thresh)
            labels = (future_return > thresh).astype(int)
            labels = labels.where(mask)
        else:
            labels = (future_return > thresh).astype(int)
        return labels

    return (future_return > 0).astype(int)


def _split_index(index: pd.DatetimeIndex, horizon: int, val_frac: float, test_frac: float) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    n = len(index)
    test_start = int(n * (1.0 - test_frac))
    val_start = int(n * (1.0 - test_frac - val_frac))
    train_end = max(0, val_start - horizon)
    val_end = max(val_start, test_start - horizon)

    train_idx = index[:train_end]
    val_idx = index[val_start:val_end]
    test_idx = index[test_start:]
    return train_idx, val_idx, test_idx


def _walk_forward_splits(index: pd.DatetimeIndex, horizon: int,
                         train_frac: float, val_frac: float, test_frac: float, step_frac: float) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Generate walk-forward splits with horizon-aware boundaries."""
    n = len(index)
    train_size = int(n * train_frac)
    val_size = int(n * val_frac)
    test_size = int(n * test_frac)
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        return []

    step_size = max(1, int(n * step_frac))
    splits = []
    start = 0
    while start + train_size + val_size + test_size <= n:
        train_start = start
        val_start = train_start + train_size
        test_start = val_start + val_size

        train_end = max(train_start, val_start - horizon)
        val_end = max(val_start, test_start - horizon)

        train_idx = index[train_start:train_end]
        val_idx = index[val_start:val_end]
        test_idx = index[test_start:test_start + test_size]

        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            break

        splits.append((train_idx, val_idx, test_idx))
        start += step_size

    return splits


def _aggregate_wf_metrics(metrics_list: List[Dict]) -> Dict:
    if not metrics_list:
        return {}

    folds = len(metrics_list)
    positive_folds = 0
    for m in metrics_list:
        avg_pnl = m.get("avg_pnl", 0.0)
        trade_count = m.get("trade_count", 0)
        try:
            avg_pnl = float(avg_pnl)
        except Exception:
            avg_pnl = 0.0
        try:
            trade_count = int(trade_count)
        except Exception:
            trade_count = 0
        if trade_count > 0 and avg_pnl > 0:
            positive_folds += 1
    positive_ratio = float(positive_folds) / float(folds) if folds else 0.0

    total_trades = sum(m.get("trade_count", 0) for m in metrics_list)
    total_pnl = sum(m.get("total_pnl", 0.0) for m in metrics_list)

    def weighted_avg(key: str) -> float:
        if total_trades <= 0:
            return float(np.mean([m.get(key, 0.0) for m in metrics_list]))
        return float(np.sum([m.get(key, 0.0) * m.get("trade_count", 0) for m in metrics_list]) / total_trades)

    return {
        "folds": folds,
        "trade_count": int(total_trades),
        "avg_pnl": weighted_avg("avg_pnl"),
        "total_pnl": float(total_pnl),
        "win_rate": weighted_avg("win_rate"),
        "accuracy": float(np.mean([m.get("accuracy", 0.0) for m in metrics_list])),
        "precision": float(np.mean([m.get("precision", 0.0) for m in metrics_list])),
        "recall": float(np.mean([m.get("recall", 0.0) for m in metrics_list])),
        "f1": float(np.mean([m.get("f1", 0.0) for m in metrics_list])),
        "auc": float(np.mean([m.get("auc", 0.0) for m in metrics_list])),
        "positive_folds": int(positive_folds),
        "positive_ratio": float(positive_ratio),
    }


def _build_model(model_name: str, seed: int):
    if model_name == "hgb" and HistGradientBoostingClassifier is not None:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=seed,
        )
    if model_name == "hgb":
        logging.warning("HistGradientBoostingClassifier unavailable; falling back to RandomForest")

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def _sample_weights(labels: pd.Series) -> np.ndarray:
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    if pos == 0 or neg == 0:
        return np.ones(len(labels))
    ratio = float(neg) / float(pos)
    return np.where(labels.to_numpy() == 1, ratio, 1.0)


def _optimize_threshold(proba: np.ndarray, labels: np.ndarray, sl_dist: np.ndarray, tp_dist: np.ndarray,
                        fees_pts: float, thr_min: float, thr_max: float, thr_step: float) -> Dict:
    best = {
        "threshold": 0.55,
        "avg_pnl": -1e9,
        "total_pnl": -1e9,
        "trades": 0,
        "win_rate": 0.0,
    }
    thresholds = np.arange(thr_min, thr_max + 1e-9, thr_step)
    for thr in thresholds:
        long_mask = proba >= thr
        short_mask = proba <= (1.0 - thr)
        take = long_mask | short_mask
        if not np.any(take):
            continue
        pnl = np.zeros_like(proba)
        pnl[long_mask & (labels == 1)] = tp_dist[long_mask & (labels == 1)]
        pnl[long_mask & (labels == 0)] = -sl_dist[long_mask & (labels == 0)]
        pnl[short_mask & (labels == 0)] = tp_dist[short_mask & (labels == 0)]
        pnl[short_mask & (labels == 1)] = -sl_dist[short_mask & (labels == 1)]
        pnl[take] = pnl[take] - fees_pts

        avg_pnl = float(np.mean(pnl[take]))
        total_pnl = float(np.sum(pnl[take]))
        wins = np.sum((pnl[take]) > 0)
        win_rate = float(wins) / float(np.sum(take))

        if avg_pnl > best["avg_pnl"] or (avg_pnl == best["avg_pnl"] and total_pnl > best["total_pnl"]):
            best.update({
                "threshold": float(thr),
                "avg_pnl": avg_pnl,
                "total_pnl": total_pnl,
                "trades": int(np.sum(take)),
                "win_rate": win_rate,
            })
    return best


def _evaluate(model, X: pd.DataFrame, y: pd.Series, sl_dist: np.ndarray, tp_dist: np.ndarray,
              threshold: float, fees_pts: float) -> Dict:
    proba = model.predict_proba(X)[:, 1]
    labels = y.to_numpy()

    long_mask = proba >= threshold
    short_mask = proba <= (1.0 - threshold)
    take = long_mask | short_mask
    pnl = np.zeros_like(proba)
    pnl[long_mask & (labels == 1)] = tp_dist[long_mask & (labels == 1)]
    pnl[long_mask & (labels == 0)] = -sl_dist[long_mask & (labels == 0)]
    pnl[short_mask & (labels == 0)] = tp_dist[short_mask & (labels == 0)]
    pnl[short_mask & (labels == 1)] = -sl_dist[short_mask & (labels == 1)]
    pnl[take] = pnl[take] - fees_pts

    preds = (proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(labels, proba)
    except ValueError:
        auc = float("nan")

    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(auc),
        "trade_count": int(np.sum(take)),
        "avg_pnl": float(np.mean(pnl[take])) if np.any(take) else 0.0,
        "total_pnl": float(np.sum(pnl[take])) if np.any(take) else 0.0,
        "win_rate": float(np.sum(pnl[take] > 0) / np.sum(take)) if np.any(take) else 0.0,
    }


def _walk_forward_evaluate(
    data: pd.DataFrame,
    session_sl: np.ndarray,
    session_tp: np.ndarray,
    settings: Dict,
    args,
    fees_pts: float,
) -> Dict:
    """Run optional walk-forward evaluation and aggregate metrics."""
    splits = _walk_forward_splits(
        data.index,
        settings["horizon_bars"],
        args.wf_train_frac,
        args.wf_val_frac,
        args.wf_test_frac,
        args.wf_step_frac,
    )
    if not splits:
        return {}

    metrics_list = []
    for fold, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        train = data.loc[train_idx]
        val = data.loc[val_idx]
        test = data.loc[test_idx]

        if train.empty or val.empty or test.empty:
            continue

        X_train = train[ML_FEATURE_COLUMNS]
        y_train = train["label"].astype(int)
        X_val = val[ML_FEATURE_COLUMNS]
        y_val = val["label"].astype(int)
        X_test = test[ML_FEATURE_COLUMNS]
        y_test = test["label"].astype(int)

        model = _build_model(args.model, args.seed)
        sample_weight = _sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
        calibrated.fit(X_val, y_val)

        val_sl = pd.Series(session_sl, index=data.index).loc[val_idx].to_numpy()
        val_tp = pd.Series(session_tp, index=data.index).loc[val_idx].to_numpy()
        val_result = _optimize_threshold(
            calibrated.predict_proba(X_val)[:, 1],
            y_val.to_numpy(),
            val_sl,
            val_tp,
            fees_pts,
            settings["thr_min"],
            settings["thr_max"],
            settings["thr_step"],
        )

        test_sl = pd.Series(session_sl, index=data.index).loc[test_idx].to_numpy()
        test_tp = pd.Series(session_tp, index=data.index).loc[test_idx].to_numpy()
        test_metrics = _evaluate(calibrated, X_test, y_test, test_sl, test_tp, val_result["threshold"], fees_pts)
        test_metrics["threshold"] = val_result["threshold"]
        test_metrics["fold"] = fold
        metrics_list.append(test_metrics)

    aggregate = _aggregate_wf_metrics(metrics_list)
    if not aggregate:
        return {}
    aggregate["folds_detail"] = metrics_list
    return aggregate


def _train_single_model(
    session_name: str,
    data: pd.DataFrame,
    session_sl: pd.Series,
    session_tp: pd.Series,
    train_settings: Dict,
    args,
    fees_pts: float,
    model_path: Path,
    tag: Optional[str] = None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    idx = data.index
    train_idx, val_idx, test_idx = _split_index(
        idx,
        train_settings["horizon_bars"],
        args.val_frac,
        args.test_frac,
    )

    train = data.loc[train_idx]
    val = data.loc[val_idx]
    test = data.loc[test_idx]

    if train.empty or val.empty or test.empty:
        logging.warning(f"{session_name}{f' {tag}' if tag else ''}: not enough samples for train/val/test split")
        return None, None

    X_train = train[ML_FEATURE_COLUMNS]
    y_train = train["label"].astype(int)
    X_val = val[ML_FEATURE_COLUMNS]
    y_val = val["label"].astype(int)
    X_test = test[ML_FEATURE_COLUMNS]
    y_test = test["label"].astype(int)

    model = _build_model(args.model, args.seed)
    sample_weight = _sample_weights(y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)

    val_sl = session_sl.loc[val_idx].to_numpy()
    val_tp = session_tp.loc[val_idx].to_numpy()
    val_result = _optimize_threshold(
        calibrated.predict_proba(X_val)[:, 1],
        y_val.to_numpy(),
        val_sl,
        val_tp,
        fees_pts,
        train_settings["thr_min"],
        train_settings["thr_max"],
        train_settings["thr_step"],
    )

    test_sl = session_sl.loc[test_idx].to_numpy()
    test_tp = session_tp.loc[test_idx].to_numpy()
    test_metrics = _evaluate(calibrated, X_test, y_test, test_sl, test_tp, val_result["threshold"], fees_pts)

    joblib.dump(calibrated, model_path)
    logging.info(f"{session_name}{f' {tag}' if tag else ''}: saved {model_path}")

    if args.walk_forward:
        wf_metrics = _walk_forward_evaluate(
            data,
            session_sl,
            session_tp,
            train_settings,
            args,
            fees_pts,
        )
        if wf_metrics:
            test_metrics["walk_forward"] = wf_metrics

    return val_result, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train MLPhysics session models (optimized).")
    parser.add_argument("--csv", default="ml_mes_et.csv")
    parser.add_argument("--timeframe-minutes", type=int, default=CONFIG.get("ML_PHYSICS_TIMEFRAME_MINUTES", 1))
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--label-mode", choices=["barrier", "return", "atr"], default="barrier")
    parser.add_argument("--drop-neutral", action="store_true")
    parser.add_argument("--sltp-strategy", default="Generic")
    parser.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--thr-min", type=float, default=0.52)
    parser.add_argument("--thr-max", type=float, default=0.75)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--drop-gap-minutes", type=float, default=75.0)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--wf-train-frac", type=float, default=0.60)
    parser.add_argument("--wf-val-frac", type=float, default=0.10)
    parser.add_argument("--wf-test-frac", type=float, default=0.20)
    parser.add_argument("--wf-step-frac", type=float, default=0.10)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    base_df = _load_csv(Path(args.csv))
    base_df = base_df.sort_index()
    if args.max_rows and args.max_rows > 0:
        base_df = base_df.iloc[-args.max_rows:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol_min_samples = int(CONFIG.get("VOLATILITY_THRESHOLDS_MIN_SAMPLES", 50) or 50)
    thresholds_map, session_fallbacks = _build_volatility_thresholds(base_df, vol_min_samples)
    mode_cfg = CONFIG.get("VOLATILITY_HIERARCHY_MODE", {}) or {}
    mode = str(mode_cfg.get("mode", "full") or "full").lower()
    include_quarter = bool(mode_cfg.get("include_quarter", True))
    std_cfg = CONFIG.get("VOLATILITY_STD_WINDOWS", {}) or {}
    default_window = int(std_cfg.get("default", getattr(volatility_filter, "std_window", 20)) or 20)
    std_windows = {
        "default": default_window,
        "sessions": std_cfg.get("sessions", {}) or {},
    }

    thresholds_file = str(CONFIG.get("VOLATILITY_THRESHOLDS_FILE", "") or "").strip()
    if thresholds_file:
        thresholds_payload = {
            "meta": {
                "mode": mode,
                "include_quarter": include_quarter,
                "std_windows": std_windows,
                "min_samples": vol_min_samples,
                "row_count": int(len(base_df)),
                "generated_at": pd.Timestamp.utcnow().isoformat(),
            },
            "thresholds": thresholds_map,
            "session_fallbacks": session_fallbacks,
        }
        thresholds_path = out_dir / thresholds_file
        thresholds_path.write_text(json.dumps(thresholds_payload, indent=2))
        logging.info("Saved volatility thresholds: %s", thresholds_path)

    global _VOL_THRESHOLDS_OVERRIDE
    global _VOL_SESSION_FALLBACKS_OVERRIDE
    global _VOL_THRESHOLDS_META
    global _VOL_THRESHOLDS_LOADED

    _VOL_THRESHOLDS_OVERRIDE = thresholds_map or None
    _VOL_SESSION_FALLBACKS_OVERRIDE = session_fallbacks or None
    _VOL_THRESHOLDS_META = {
        "mode": mode,
        "include_quarter": include_quarter,
        "std_windows": std_windows,
    }
    _VOL_THRESHOLDS_LOADED = True

    risk_cfg = CONFIG.get("RISK", {})
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0))
    fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.5))
    fees_pts = (fees_per_side * 2.0) / point_value

    thresholds_report = {}
    metrics_report = {}

    for session_name, settings in CONFIG["SESSIONS"].items():
        train_settings = _resolve_session_settings(args, session_name)

        df = base_df
        if train_settings["max_rows"] and train_settings["max_rows"] > 0:
            df = df.iloc[-train_settings["max_rows"]:]

        df = _resample_ohlcv(df, train_settings["timeframe_minutes"])
        df, dropped_days = _drop_gap_days(df, train_settings["drop_gap_minutes"])
        if dropped_days:
            sample = ", ".join([d.strftime("%Y-%m-%d") for d in dropped_days[:10]])
            more = "" if len(dropped_days) <= 10 else f" (+{len(dropped_days) - 10} more)"
            logging.info(
                "%s: Dropping %d day(s) with gaps > %.1f minutes: %s%s",
                session_name,
                len(dropped_days),
                train_settings["drop_gap_minutes"],
                sample,
                more,
            )

        features = _build_features(df)
        if features.empty:
            logging.warning(f"{session_name}: no features after preprocessing")
            continue

        sl_dist, tp_dist = _build_sltp_arrays(df, train_settings["sltp_strategy"])
        labels = _build_labels(
            df,
            train_settings["horizon_bars"],
            train_settings["label_mode"],
            train_settings["drop_neutral"],
            sl_dist,
            tp_dist,
        )

        labels = labels.reindex(features.index)
        sl_dist = pd.Series(sl_dist, index=df.index).reindex(features.index).to_numpy()
        tp_dist = pd.Series(tp_dist, index=df.index).reindex(features.index).to_numpy()

        hours = settings["HOURS"]
        session_mask = features.index.hour.isin(hours)
        session_features = features.loc[session_mask]
        session_labels = labels.loc[session_features.index]
        session_sl = pd.Series(sl_dist, index=features.index).loc[session_features.index].to_numpy()
        session_tp = pd.Series(tp_dist, index=features.index).loc[session_features.index].to_numpy()

        data = session_features.join(session_labels.rename("label"), how="inner")
        data = data.dropna(subset=["label"])
        if data.empty:
            logging.warning(f"{session_name}: no labeled samples after session filter")
            continue

        session_sl_series = pd.Series(session_sl, index=session_features.index)
        session_tp_series = pd.Series(session_tp, index=session_features.index)

        vol_split = CONFIG.get("ML_PHYSICS_VOL_SPLIT", {}) or {}
        vol_split_3way = CONFIG.get("ML_PHYSICS_VOL_SPLIT_3WAY", {}) or {}
        split_enabled = bool(vol_split.get("enabled")) and session_name in set(vol_split.get("sessions", []))
        split_3way = bool(vol_split_3way.get("enabled")) and session_name in set(vol_split_3way.get("sessions", []))
        if split_3way:
            split_enabled = True
        if split_enabled and not split_3way:
            feature_name = vol_split.get("feature", "High_Volatility")
            if feature_name not in data.columns:
                logging.warning(f"{session_name}: split enabled but missing feature '{feature_name}'. Falling back to single model.")
                split_enabled = False

        if split_enabled:
            if split_3way:
                regimes = _compute_vol_regimes(df)
                regime_series = regimes.reindex(session_features.index).reindex(data.index)
                low_mask = regime_series.isin([VolRegime.ULTRA_LOW, VolRegime.LOW])
                normal_mask = regime_series == VolRegime.NORMAL
                high_mask = regime_series == VolRegime.HIGH

                base_model_path = out_dir / settings["MODEL_FILE"]
                low_model_path = out_dir / settings.get("MODEL_FILE_LOW", settings["MODEL_FILE"])
                high_model_path = out_dir / settings.get("MODEL_FILE_HIGH", settings["MODEL_FILE"])
                normal_setting = settings.get("MODEL_FILE_NORMAL")
                if normal_setting:
                    normal_model_path = out_dir / normal_setting
                else:
                    if base_model_path.suffix:
                        normal_model_path = base_model_path.with_name(
                            f"{base_model_path.stem}_normal{base_model_path.suffix}"
                        )
                    else:
                        normal_model_path = Path(f"{base_model_path}_normal")

                low_data = data.loc[low_mask]
                normal_data = data.loc[normal_mask]
                high_data = data.loc[high_mask]

                low_val, low_metrics = _train_single_model(
                    session_name,
                    low_data,
                    session_sl_series.loc[low_data.index],
                    session_tp_series.loc[low_data.index],
                    train_settings,
                    args,
                    fees_pts,
                    low_model_path,
                    tag="LOW_VOL",
                )
                normal_val, normal_metrics = _train_single_model(
                    session_name,
                    normal_data,
                    session_sl_series.loc[normal_data.index],
                    session_tp_series.loc[normal_data.index],
                    train_settings,
                    args,
                    fees_pts,
                    normal_model_path,
                    tag="NORMAL_VOL",
                )
                high_val, high_metrics = _train_single_model(
                    session_name,
                    high_data,
                    session_sl_series.loc[high_data.index],
                    session_tp_series.loc[high_data.index],
                    train_settings,
                    args,
                    fees_pts,
                    high_model_path,
                    tag="HIGH_VOL",
                )

                if low_val or normal_val or high_val:
                    thresholds_report[session_name] = {
                        "low": {**(low_val or {}), "settings": train_settings},
                        "normal": {**(normal_val or {}), "settings": train_settings},
                        "high": {**(high_val or {}), "settings": train_settings},
                        "split": True,
                        "split_mode": "3way",
                    }
                    metrics_report[session_name] = {
                        "low": low_metrics or {},
                        "normal": normal_metrics or {},
                        "high": high_metrics or {},
                        "split_mode": "3way",
                    }
                    if low_val and low_metrics:
                        logging.info(
                            f"{session_name} LOW: threshold={low_val['threshold']:.2f} "
                            f"trades={low_metrics['trade_count']} win_rate={low_metrics['win_rate']:.2%} "
                            f"avg_pnl={low_metrics['avg_pnl']:.2f} total_pnl={low_metrics['total_pnl']:.2f}"
                        )
                    if normal_val and normal_metrics:
                        logging.info(
                            f"{session_name} NORMAL: threshold={normal_val['threshold']:.2f} "
                            f"trades={normal_metrics['trade_count']} win_rate={normal_metrics['win_rate']:.2%} "
                            f"avg_pnl={normal_metrics['avg_pnl']:.2f} total_pnl={normal_metrics['total_pnl']:.2f}"
                        )
                    if high_val and high_metrics:
                        logging.info(
                            f"{session_name} HIGH: threshold={high_val['threshold']:.2f} "
                            f"trades={high_metrics['trade_count']} win_rate={high_metrics['win_rate']:.2%} "
                            f"avg_pnl={high_metrics['avg_pnl']:.2f} total_pnl={high_metrics['total_pnl']:.2f}"
                        )
                else:
                    logging.warning(f"{session_name}: split training failed; no models saved.")
                continue

            low_mask = data[feature_name] < 0.5
            high_mask = ~low_mask
            low_data = data.loc[low_mask]
            high_data = data.loc[high_mask]

            low_model_path = out_dir / settings.get("MODEL_FILE_LOW", settings["MODEL_FILE"])
            high_model_path = out_dir / settings.get("MODEL_FILE_HIGH", settings["MODEL_FILE"])

            low_val, low_metrics = _train_single_model(
                session_name,
                low_data,
                session_sl_series.loc[low_data.index],
                session_tp_series.loc[low_data.index],
                train_settings,
                args,
                fees_pts,
                low_model_path,
                tag="LOW_VOL",
            )
            high_val, high_metrics = _train_single_model(
                session_name,
                high_data,
                session_sl_series.loc[high_data.index],
                session_tp_series.loc[high_data.index],
                train_settings,
                args,
                fees_pts,
                high_model_path,
                tag="HIGH_VOL",
            )

            if low_val or high_val:
                thresholds_report[session_name] = {
                    "low": {**(low_val or {}), "settings": train_settings},
                    "high": {**(high_val or {}), "settings": train_settings},
                    "split": True,
                }
                metrics_report[session_name] = {
                    "low": low_metrics or {},
                    "high": high_metrics or {},
                }
                if low_val and low_metrics:
                    logging.info(
                        f"{session_name} LOW: threshold={low_val['threshold']:.2f} "
                        f"trades={low_metrics['trade_count']} win_rate={low_metrics['win_rate']:.2%} "
                        f"avg_pnl={low_metrics['avg_pnl']:.2f} total_pnl={low_metrics['total_pnl']:.2f}"
                    )
                if high_val and high_metrics:
                    logging.info(
                        f"{session_name} HIGH: threshold={high_val['threshold']:.2f} "
                        f"trades={high_metrics['trade_count']} win_rate={high_metrics['win_rate']:.2%} "
                        f"avg_pnl={high_metrics['avg_pnl']:.2f} total_pnl={high_metrics['total_pnl']:.2f}"
                    )
            else:
                logging.warning(f"{session_name}: split training failed; no models saved.")
            continue

        out_path = out_dir / settings["MODEL_FILE"]
        val_result, test_metrics = _train_single_model(
            session_name,
            data,
            session_sl_series,
            session_tp_series,
            train_settings,
            args,
            fees_pts,
            out_path,
        )
        if not val_result or not test_metrics:
            continue

        thresholds_report[session_name] = {
            **val_result,
            "settings": train_settings,
        }
        metrics_report[session_name] = test_metrics
        logging.info(
            f"{session_name}: threshold={val_result['threshold']:.2f} "
            f"trades={test_metrics['trade_count']} win_rate={test_metrics['win_rate']:.2%} "
            f"avg_pnl={test_metrics['avg_pnl']:.2f} total_pnl={test_metrics['total_pnl']:.2f}"
        )

    thresholds_path = out_dir / "ml_physics_thresholds.json"
    metrics_path = out_dir / "ml_physics_metrics.json"
    thresholds_path.write_text(json.dumps(thresholds_report, indent=2))
    metrics_path.write_text(json.dumps(metrics_report, indent=2))
    logging.info(f"Saved thresholds: {thresholds_path}")
    logging.info(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
