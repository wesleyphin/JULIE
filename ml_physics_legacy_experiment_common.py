from __future__ import annotations

from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import CONFIG
from dynamic_sltp_params import SESSION_DEFAULTS, STRATEGY_PARAMS


NY_TZ = ZoneInfo("America/New_York")

RSI_PERIOD = 9
ADX_PERIOD = 14
ATR_PERIOD = 14
ZSCORE_WINDOW = 50
RVOL_LOOKBACK_DAYS = 20
SLOPE_WINDOW = 15
VOLATILITY_WINDOW = 15
RSI_VEL_LAG = 3
ADX_VEL_LAG = 3

ML_FEATURE_COLUMNS = [
    "Close_ZScore",
    "High_ZScore",
    "Low_ZScore",
    "ATR_ZScore",
    "Volatility_ZScore",
    "Range_ZScore",
    "Volume_ZScore",
    "Slope_ZScore",
    "RVol_ZScore",
    "RSI_Vel",
    "Adx_Vel",
    "RSI_Norm",
    "ADX_Norm",
    "Return_1",
    "Return_5",
    "Return_15",
    "Hour_Sin",
    "Hour_Cos",
    "Minute_Sin",
    "Minute_Cos",
    "DOW_Sin",
    "DOW_Cos",
    "Is_Trending",
    "Trend_Direction",
    "High_Volatility",
]

OPTIONAL_FEATURE_COLUMNS = {"RVol_ZScore"}


def _normalize_datetime_like(value) -> pd.DatetimeIndex:
    dt_index = pd.DatetimeIndex(pd.to_datetime(value, errors="coerce"))
    if dt_index.tz is None:
        return dt_index.tz_localize(NY_TZ)
    return dt_index.tz_convert(NY_TZ)


def _legacy_sltp_floors() -> tuple[float, float]:
    cfg = CONFIG.get("ML_PHYSICS_LEGACY_EXPERIMENT", {}) or {}
    min_sl = float(cfg.get("min_sl_points", 4.0) or 4.0)
    min_tp = float(cfg.get("min_tp_points", 6.0) or 6.0)
    if not np.isfinite(min_sl) or min_sl <= 0.0:
        min_sl = 4.0
    if not np.isfinite(min_tp) or min_tp <= 0.0:
        min_tp = 6.0
    return float(min_sl), float(min_tp)


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    work = df.copy()
    lower_map = {str(col).lower(): col for col in work.columns}
    out = pd.DataFrame(index=work.index)
    for src in ("open", "high", "low", "close"):
        col = lower_map.get(src)
        if col is None:
            raise KeyError(f"Missing required column: {src}")
        out[src] = pd.to_numeric(work[col], errors="coerce")
    vol_col = lower_map.get("volume")
    out["volume"] = pd.to_numeric(work[vol_col], errors="coerce") if vol_col is not None else 0.0

    dt_col = lower_map.get("datetime")
    if dt_col is not None:
        dt_index = _normalize_datetime_like(work[dt_col])
    else:
        dt_index = _normalize_datetime_like(work.index)
    out.index = pd.DatetimeIndex(dt_index)
    out = out[~out.index.isna()]
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    normalized = normalize_ohlcv_frame(df)
    if normalized.empty or int(minutes or 1) <= 1:
        return normalized
    rule = f"{int(minutes)}min"
    out = normalized.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return out.dropna()


def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = tr.rolling(window=period).mean()
    plus_series = pd.Series(plus_dm, index=high.index)
    minus_series = pd.Series(minus_dm, index=high.index)
    plus_di = 100 * plus_series.rolling(window=period).mean() / atr
    minus_di = 100 * minus_series.rolling(window=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()
    return adx, plus_di, minus_di


def calculate_slope(series: pd.Series, window: int = SLOPE_WINDOW) -> pd.Series:
    def _slope_calc(values) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        y = np.asarray(values, dtype=float)
        covariance = np.cov(x, y)[0, 1]
        variance = np.var(x)
        return float(covariance / variance) if variance != 0 else 0.0

    return series.rolling(window=window).apply(_slope_calc, raw=False)


def zscore_normalize(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / (rolling_std + 1e-10)


def calculate_rvol(volume: pd.Series, datetime_index: pd.Series, lookback_days: int = RVOL_LOOKBACK_DAYS) -> pd.Series:
    dt_index = _normalize_datetime_like(datetime_index)
    dt_series = pd.Series(dt_index, index=volume.index, name="datetime")
    tmp = pd.DataFrame({"volume": volume.to_numpy(), "datetime": dt_series.to_numpy()}, index=volume.index)
    tmp["time_bucket"] = (tmp["datetime"].dt.hour * 60) + tmp["datetime"].dt.minute
    avg_vol_by_time = tmp.groupby("time_bucket")["volume"].transform(
        lambda x: x.rolling(window=max(int(lookback_days), 1) * 5, min_periods=10).mean()
    )
    return volume / (avg_vol_by_time + 1.0)


def encode_cyclical_time(datetime_series: pd.Series):
    hour = datetime_series.dt.hour
    minute = datetime_series.dt.minute
    day_of_week = datetime_series.dt.dayofweek
    return (
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * minute / 60),
        np.cos(2 * np.pi * minute / 60),
        np.sin(2 * np.pi * day_of_week / 5),
        np.cos(2 * np.pi * day_of_week / 5),
    )


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    w_df = normalize_ohlcv_frame(df)
    if w_df.empty:
        return pd.DataFrame(columns=ML_FEATURE_COLUMNS)

    dt_series = pd.Series(w_df.index, index=w_df.index, name="datetime")
    w_df["RSI"] = calculate_rsi(w_df["close"], period=RSI_PERIOD)
    w_df["ATR"] = calculate_atr(w_df["high"], w_df["low"], w_df["close"], period=ATR_PERIOD)
    adx, plus_di, minus_di = calculate_adx(w_df["high"], w_df["low"], w_df["close"], period=ADX_PERIOD)
    w_df["ADX"] = adx
    w_df["PLUS_DI"] = plus_di
    w_df["MINUS_DI"] = minus_di

    w_df["RSI_Vel"] = w_df["RSI"].diff(RSI_VEL_LAG)
    w_df["Adx_Vel"] = w_df["ADX"].diff(ADX_VEL_LAG)
    w_df["Slope"] = calculate_slope(w_df["close"], window=SLOPE_WINDOW)
    w_df["Volatility"] = w_df["close"].rolling(window=VOLATILITY_WINDOW).std()
    w_df["Range"] = w_df["high"] - w_df["low"]

    w_df["Return_1"] = w_df["close"].pct_change(1)
    w_df["Return_5"] = w_df["close"].pct_change(5)
    w_df["Return_15"] = w_df["close"].pct_change(15)

    w_df["Close_ZScore"] = zscore_normalize(w_df["close"], window=ZSCORE_WINDOW)
    w_df["High_ZScore"] = zscore_normalize(w_df["high"], window=ZSCORE_WINDOW)
    w_df["Low_ZScore"] = zscore_normalize(w_df["low"], window=ZSCORE_WINDOW)
    w_df["ATR_ZScore"] = zscore_normalize(w_df["ATR"], window=ZSCORE_WINDOW)
    w_df["Volatility_ZScore"] = zscore_normalize(w_df["Volatility"], window=ZSCORE_WINDOW)
    w_df["Range_ZScore"] = zscore_normalize(w_df["Range"], window=ZSCORE_WINDOW)
    w_df["Volume_ZScore"] = zscore_normalize(w_df["volume"], window=ZSCORE_WINDOW)
    w_df["RSI_Norm"] = (w_df["RSI"] - 50.0) / 50.0
    w_df["ADX_Norm"] = (w_df["ADX"] / 50.0) - 1.0
    w_df["Slope_ZScore"] = zscore_normalize(w_df["Slope"], window=ZSCORE_WINDOW)

    w_df["RVol"] = calculate_rvol(w_df["volume"], dt_series, lookback_days=RVOL_LOOKBACK_DAYS)
    w_df["RVol_ZScore"] = zscore_normalize(w_df["RVol"], window=ZSCORE_WINDOW)

    (
        w_df["Hour_Sin"],
        w_df["Hour_Cos"],
        w_df["Minute_Sin"],
        w_df["Minute_Cos"],
        w_df["DOW_Sin"],
        w_df["DOW_Cos"],
    ) = encode_cyclical_time(dt_series)

    w_df["Is_Trending"] = (w_df["ADX"] > 25).astype(int)
    w_df["Trend_Direction"] = np.sign(w_df["PLUS_DI"] - w_df["MINUS_DI"])
    w_df["ATR_MA"] = w_df["ATR"].rolling(window=100).mean()
    w_df["High_Volatility"] = (w_df["ATR"] > w_df["ATR_MA"] * 1.5).astype(int)

    features = w_df[ML_FEATURE_COLUMNS].copy()
    for col in OPTIONAL_FEATURE_COLUMNS:
        if col in features.columns:
            features[col] = features[col].fillna(0.0)
    critical_cols = [col for col in ML_FEATURE_COLUMNS if col not in OPTIONAL_FEATURE_COLUMNS]
    features = features.dropna(subset=critical_cols)
    return features


def extract_latest_feature_row(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    features = build_feature_frame(df)
    if features.empty:
        return None
    last = features.iloc[-1]
    if any(pd.isna(last[col]) for col in ML_FEATURE_COLUMNS if col not in OPTIONAL_FEATURE_COLUMNS):
        return None
    return pd.DataFrame([last.to_dict()], columns=ML_FEATURE_COLUMNS, index=[features.index[-1]])


def session_from_hour(hour: int) -> str:
    if hour >= 18 or hour < 3:
        return "ASIA"
    if 3 <= hour < 8:
        return "LONDON"
    if 8 <= hour < 12:
        return "NY_AM"
    if 12 <= hour < 17:
        return "NY_PM"
    return "CLOSED"


def yearly_quarter(months: np.ndarray) -> np.ndarray:
    return np.select(
        [months <= 3, months <= 6, months <= 9, months <= 12],
        ["Q1", "Q2", "Q3", "Q4"],
        default="Q4",
    )


def monthly_quarter(days: np.ndarray) -> np.ndarray:
    return np.select(
        [days <= 7, days <= 14, days <= 21, days <= 31],
        ["W1", "W2", "W3", "W4"],
        default="W4",
    )


def dow_name(dows: np.ndarray) -> np.ndarray:
    names = np.array(["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"], dtype=object)
    return names[dows]


def build_hierarchy_keys(index: pd.DatetimeIndex, sessions: np.ndarray) -> np.ndarray:
    months = index.month.to_numpy()
    days = index.day.to_numpy()
    dows = index.weekday.to_numpy()
    yq = yearly_quarter(months)
    mq = monthly_quarter(days)
    dow = dow_name(dows)
    keys = np.char.add(np.char.add(np.char.add(np.char.add(yq, "_"), mq), "_"), dow)
    keys = np.char.add(np.char.add(keys, "_"), sessions)
    return keys


def build_sltp_arrays(df: pd.DataFrame, strategy_name: str = "Generic") -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_ohlcv_frame(df)
    index = normalized.index
    sessions = np.array([session_from_hour(int(ts.hour)) for ts in index], dtype=object)
    strategy_params = STRATEGY_PARAMS.get(strategy_name) if strategy_name in STRATEGY_PARAMS else None
    if isinstance(strategy_params, dict):
        keys = build_hierarchy_keys(index, sessions)
        params = []
        session_defaults = strategy_params.get("SESSION_DEFAULTS") if isinstance(strategy_params.get("SESSION_DEFAULTS"), dict) else {}
        default_values = strategy_params.get("DEFAULT") if isinstance(strategy_params.get("DEFAULT"), dict) else None
        for key, session in zip(keys, sessions):
            row = strategy_params.get(key)
            if not isinstance(row, dict):
                row = session_defaults.get(session)
            if not isinstance(row, dict):
                row = default_values
            if not isinstance(row, dict):
                row = SESSION_DEFAULTS.get(session, {"sl_mult": 2.0, "tp_mult": 2.5, "atr_med": 1.5})
            params.append(row)
    else:
        params = [
            SESSION_DEFAULTS.get(session, {"sl_mult": 2.0, "tp_mult": 2.5, "atr_med": 1.5})
            for session in sessions
        ]

    sl_mult = np.array([float(p.get("sl_mult", 2.0) or 2.0) for p in params], dtype=float)
    tp_mult = np.array([float(p.get("tp_mult", 2.5) or 2.5) for p in params], dtype=float)
    atr_med = np.array([float(p.get("atr_med", 1.5) or 1.5) for p in params], dtype=float)

    atr = (normalized["high"] - normalized["low"]).rolling(14).mean().to_numpy(dtype=float)
    atr = np.where(np.isfinite(atr), atr, atr_med)

    sl_dist = np.round((atr * sl_mult) * 4.0) / 4.0
    tp_dist = np.round((atr * tp_mult) * 4.0) / 4.0
    min_sl, min_tp = _legacy_sltp_floors()
    sl_dist = np.maximum(sl_dist, min_sl)
    tp_dist = np.maximum(tp_dist, min_tp)
    return sl_dist, tp_dist


def latest_sltp(df: pd.DataFrame, strategy_name: str = "Generic") -> dict:
    normalized = normalize_ohlcv_frame(df)
    if normalized.empty:
        return {"sl_dist": 0.0, "tp_dist": 0.0}
    sl_dist, tp_dist = build_sltp_arrays(normalized, strategy_name=strategy_name)
    return {
        "sl_dist": float(sl_dist[-1]) if len(sl_dist) else 0.0,
        "tp_dist": float(tp_dist[-1]) if len(tp_dist) else 0.0,
    }


def resolve_artifact_path(base_dir: str | Path, raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip())
    if path.is_absolute():
        return path
    return Path(base_dir) / path
