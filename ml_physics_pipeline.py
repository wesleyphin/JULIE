import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from config import CONFIG


NY_TZ = ZoneInfo("America/New_York")

# Keep defaults aligned with ml_physics_strategy.py constants.
RSI_PERIOD = 14
ADX_PERIOD = 20
ATR_PERIOD = 20
ZSCORE_WINDOW = 100
RVOL_LOOKBACK_DAYS = 20
SLOPE_WINDOW = 30
VOLATILITY_WINDOW = 30
RSI_VEL_LAG = 5
ADX_VEL_LAG = 5
RETURN_FAST = 1
RETURN_MED = 10
RETURN_SLOW = 30
VWAP_CROSS_WINDOW = 30
VWAP_PERSIST_CAP = 300
TREND_EMA_FAST = 20
TREND_EMA_SLOW = 50
TREND_PERSIST_CAP = 300
ATR_SLOPE_LAG = 10
SESSION_OPEN_MINUTES = 60

ML_FEATURE_COLUMNS = [
    "Close_ZScore",
    "High_ZScore",
    "Low_ZScore",
    "ATR_ZScore",
    "Volatility_ZScore",
    "Range_ZScore",
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
    "VWAP_Dist_ATR",
    "VWAP_Crosses_30",
    "VWAP_Bars_Since_Cross",
    "VWAP_Behavior",
    "Range_Overlap_Ratio",
    "Gap_Size_ATR",
    "Gap_Filled",
    "ATR_Slope",
    "Trend_Persistence",
    "Session_Open_Range_ATR",
    "Session_Open_Drive",
    "Session_ID",
    "ATR_State",
    "Trend_State",
    "Liquidity_State",
    "Macro_Regime_ID",
]

OPTIONAL_FEATURE_COLUMNS = {"RVol_ZScore"}
_MACRO_REGIMES_CACHE = None


def _opt_cfg() -> Dict:
    cfg = CONFIG.get("ML_PHYSICS_OPT", {}) or {}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "mode": str(cfg.get("mode", "backtest") or "backtest").lower(),
        "feature_cache_dir": str(cfg.get("feature_cache_dir", "cache/ml_physics/") or "cache/ml_physics/"),
        "prediction_cache": bool(cfg.get("prediction_cache", True)),
        "overwrite_cache": bool(cfg.get("overwrite_cache", False)),
    }


def _sanitize_token(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("_") or "unknown"


def _guess_symbol(df: pd.DataFrame) -> str:
    if "symbol" in df.columns:
        try:
            vals = pd.Series(df["symbol"]).dropna().astype(str).unique().tolist()
            if len(vals) == 1:
                return vals[0]
            if len(vals) > 1:
                return "MULTI"
        except Exception:
            pass
    return str(CONFIG.get("TARGET_SYMBOL") or CONFIG.get("CONTRACT_ROOT") or "ES")


def _cache_suffix(df: pd.DataFrame, tf_minutes: int, align_mode: str) -> str:
    idx = df.index
    start = str(idx.min()) if len(idx) else "NA"
    end = str(idx.max()) if len(idx) else "NA"
    token = f"{len(df)}|{start}|{end}|tf{tf_minutes}|{align_mode}"
    return hashlib.sha1(token.encode("utf-8")).hexdigest()[:10]


def get_feature_cache_path(config_dict: Dict, symbol: str, start: str, end: str, tf_minutes: int, align_mode: str) -> Path:
    cache_dir = Path(str(config_dict.get("feature_cache_dir", "cache/ml_physics/") or "cache/ml_physics/"))
    if not cache_dir.is_absolute():
        cache_dir = Path(__file__).resolve().parent / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{_sanitize_token(symbol)}_{_sanitize_token(start)}_{_sanitize_token(end)}"
        f"_tf{int(tf_minutes)}_{_sanitize_token(align_mode)}_features.parquet"
    )
    return cache_dir / name


def get_prediction_cache_path(config_dict: Dict, symbol: str, start: str, end: str, tf_minutes: int, align_mode: str) -> Path:
    cache_dir = Path(str(config_dict.get("feature_cache_dir", "cache/ml_physics/") or "cache/ml_physics/"))
    if not cache_dir.is_absolute():
        cache_dir = Path(__file__).resolve().parent / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{_sanitize_token(symbol)}_{_sanitize_token(start)}_{_sanitize_token(end)}"
        f"_tf{int(tf_minutes)}_{_sanitize_token(align_mode)}_predictions.parquet"
    )
    return cache_dir / name


def _rolling_slope_fast(series: pd.Series, window: int = SLOPE_WINDOW) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if window <= 1 or n < window:
        return pd.Series(out, index=series.index)

    x = np.arange(window, dtype=float)
    sum_x = float(np.sum(x))
    sum_x2 = float(np.sum(x * x))
    denom = (window * sum_x2) - (sum_x * sum_x)
    if denom == 0:
        return pd.Series(out, index=series.index)

    vals_no_nan = np.where(np.isfinite(values), values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(vals_no_nan)))
    sum_y = csum[window:] - csum[:-window]
    sum_xy = np.convolve(vals_no_nan, x, mode="valid")

    valid = np.isfinite(values).astype(np.int64)
    ccount = np.concatenate(([0], np.cumsum(valid)))
    counts = ccount[window:] - ccount[:-window]

    slope = ((window * sum_xy) - (sum_x * sum_y)) / denom
    slope[counts < window] = np.nan
    out[window - 1 :] = slope
    return pd.Series(out, index=series.index)


def _zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    return (series - rolling_mean) / (rolling_std + 1e-10)


def _calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD):
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


def _calculate_rvol_fast(volume: pd.Series, datetime_series: pd.Series, lookback_days: int = RVOL_LOOKBACK_DAYS) -> pd.Series:
    vol = pd.to_numeric(volume, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    dt_ser = pd.to_datetime(datetime_series)
    buckets = (dt_ser.dt.hour.to_numpy(dtype=int) * 60) + dt_ser.dt.minute.to_numpy(dtype=int)
    out = np.full(len(vol), np.nan, dtype=float)

    window = max(int(lookback_days), 1)
    min_periods = min(10, window)

    for bucket in np.unique(buckets):
        idx = np.where(buckets == bucket)[0]
        if len(idx) == 0:
            continue
        vals = vol[idx]
        csum = np.concatenate(([0.0], np.cumsum(vals)))
        pos = np.arange(len(vals), dtype=int)
        starts = np.maximum(0, pos - window + 1)
        counts = pos - starts + 1
        sums = csum[pos + 1] - csum[starts]
        means = np.where(counts >= min_periods, sums / counts, np.nan)
        out[idx] = vals / (means + 1.0)
    return pd.Series(out, index=volume.index)


def _encode_time(dt_series: pd.Series):
    hour = dt_series.dt.hour
    minute = dt_series.dt.minute
    dow = dt_series.dt.dayofweek
    return (
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * minute / 60),
        np.cos(2 * np.pi * minute / 60),
        np.sin(2 * np.pi * dow / 5),
        np.cos(2 * np.pi * dow / 5),
    )


def _load_macro_regimes():
    global _MACRO_REGIMES_CACHE
    if _MACRO_REGIMES_CACHE is not None:
        return _MACRO_REGIMES_CACHE
    if not CONFIG.get("ML_PHYSICS_USE_MACRO_REGIME", False):
        _MACRO_REGIMES_CACHE = []
        return _MACRO_REGIMES_CACHE
    path = CONFIG.get("ML_PHYSICS_REGIMES_FILE") or "regimes.json"
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        _MACRO_REGIMES_CACHE = []
        return _MACRO_REGIMES_CACHE
    parsed = []
    try:
        regimes = payload.get("regimes") if isinstance(payload, dict) else []
        for idx, reg in enumerate(regimes or []):
            name = str(reg.get("name", f"regime_{idx}"))
            start = pd.to_datetime(reg.get("start"))
            end = pd.to_datetime(reg.get("end"))
            if pd.isna(start) or pd.isna(end):
                continue
            parsed.append((name, start.normalize(), end.normalize(), idx + 1))
    except Exception:
        parsed = []
    _MACRO_REGIMES_CACHE = parsed
    return _MACRO_REGIMES_CACHE


def _map_macro_regime_id(dt_series: pd.Series) -> np.ndarray:
    if not CONFIG.get("ML_PHYSICS_USE_MACRO_REGIME", False):
        return np.zeros(len(dt_series), dtype=np.int8)
    regimes = _load_macro_regimes()
    if not regimes:
        return np.zeros(len(dt_series), dtype=np.int8)
    dates = pd.to_datetime(dt_series).dt.normalize()
    out = np.zeros(len(dates), dtype=np.int8)
    for _, start, end, rid in regimes:
        mask = (dates >= start) & (dates <= end)
        out[mask.to_numpy()] = np.int8(rid)
    return out


def _extract_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    lower = {str(c).lower(): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for src, dst in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")):
        col = lower.get(src)
        if col is None:
            if src == "volume":
                out[dst] = 0.0
                continue
            raise KeyError(f"Missing required column: {src}")
        out[dst] = pd.to_numeric(df[col], errors="coerce")
    if "datetime" in lower:
        dt_series = pd.to_datetime(df[lower["datetime"]], errors="coerce")
    else:
        dt_series = pd.to_datetime(df.index, errors="coerce")
    if getattr(dt_series.dt, "tz", None) is None:
        dt_series = dt_series.dt.tz_localize(NY_TZ)
    else:
        dt_series = dt_series.dt.tz_convert(NY_TZ)
    out["datetime"] = dt_series
    return out


def compute_features(df: pd.DataFrame, *, cast_float32: bool = False) -> pd.DataFrame:
    """
    Vectorized feature engineering for MLPhysics.
    Expects sorted OHLCV bars; returns full feature dataframe (not single-row).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    w_df = _extract_ohlcv(df)
    w_df["RSI"] = _calculate_rsi(w_df["Close"], period=RSI_PERIOD)
    w_df["ATR"] = _calculate_atr(w_df["High"], w_df["Low"], w_df["Close"], period=ATR_PERIOD)
    adx, plus_di, minus_di = _calculate_adx(w_df["High"], w_df["Low"], w_df["Close"], period=ADX_PERIOD)
    w_df["ADX"] = adx
    w_df["PLUS_DI"] = plus_di
    w_df["MINUS_DI"] = minus_di

    w_df["RSI_Vel"] = w_df["RSI"].diff(RSI_VEL_LAG)
    w_df["Adx_Vel"] = w_df["ADX"].diff(ADX_VEL_LAG)
    w_df["Slope"] = _rolling_slope_fast(w_df["Close"], window=SLOPE_WINDOW)
    w_df["Volatility"] = w_df["Close"].rolling(window=VOLATILITY_WINDOW).std()
    w_df["Range"] = w_df["High"] - w_df["Low"]

    w_df["Return_1"] = w_df["Close"].pct_change(RETURN_FAST)
    w_df["Return_5"] = w_df["Close"].pct_change(RETURN_MED)
    w_df["Return_15"] = w_df["Close"].pct_change(RETURN_SLOW)

    w_df["Close_ZScore"] = _zscore(w_df["Close"], window=ZSCORE_WINDOW)
    w_df["High_ZScore"] = _zscore(w_df["High"], window=ZSCORE_WINDOW)
    w_df["Low_ZScore"] = _zscore(w_df["Low"], window=ZSCORE_WINDOW)
    w_df["ATR_ZScore"] = _zscore(w_df["ATR"], window=ZSCORE_WINDOW)
    w_df["Volatility_ZScore"] = _zscore(w_df["Volatility"], window=ZSCORE_WINDOW)
    w_df["Range_ZScore"] = _zscore(w_df["Range"], window=ZSCORE_WINDOW)
    w_df["Volume_ZScore"] = _zscore(w_df["Volume"], window=ZSCORE_WINDOW)
    w_df["RSI_Norm"] = (w_df["RSI"] - 50.0) / 50.0
    w_df["ADX_Norm"] = (w_df["ADX"] / 50.0) - 1.0
    w_df["Slope_ZScore"] = _zscore(w_df["Slope"], window=ZSCORE_WINDOW)

    w_df["RVol"] = _calculate_rvol_fast(w_df["Volume"], w_df["datetime"], lookback_days=RVOL_LOOKBACK_DAYS)
    w_df["RVol_ZScore"] = _zscore(w_df["RVol"], window=ZSCORE_WINDOW)

    hour_sin, hour_cos, minute_sin, minute_cos, dow_sin, dow_cos = _encode_time(w_df["datetime"])
    w_df["Hour_Sin"] = hour_sin
    w_df["Hour_Cos"] = hour_cos
    w_df["Minute_Sin"] = minute_sin
    w_df["Minute_Cos"] = minute_cos
    w_df["DOW_Sin"] = dow_sin
    w_df["DOW_Cos"] = dow_cos

    w_df["Is_Trending"] = (w_df["ADX"] > 25).astype(int)
    w_df["Trend_Direction"] = np.sign(w_df["PLUS_DI"] - w_df["MINUS_DI"])
    w_df["ATR_MA"] = w_df["ATR"].rolling(window=100).mean()
    w_df["High_Volatility"] = (w_df["ATR"] > w_df["ATR_MA"] * 1.5).astype(int)

    dt = w_df["datetime"]
    trading_day = (dt - pd.Timedelta(hours=18)).dt.normalize()
    pv = w_df["Close"] * w_df["Volume"]
    cum_pv = pv.groupby(trading_day).cumsum()
    cum_vol = w_df["Volume"].groupby(trading_day).cumsum()
    vwap = cum_pv / cum_vol.replace(0, np.nan)
    vwap_dist = (w_df["Close"] - vwap) / w_df["ATR"]
    w_df["VWAP_Dist_ATR"] = vwap_dist.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vwap_sign = np.sign(w_df["Close"] - vwap).replace(0, np.nan).ffill().fillna(0)
    vwap_cross = vwap_sign.ne(vwap_sign.shift(1)).fillna(False)
    day_pos = w_df.groupby(trading_day).cumcount().to_numpy()
    last_cross = (
        pd.Series(np.where(vwap_cross.to_numpy(), day_pos, -1), index=w_df.index)
        .groupby(trading_day)
        .cummax()
        .to_numpy()
    )
    bars_since_cross = day_pos - last_cross
    bars_since_cross[last_cross < 0] = day_pos[last_cross < 0]
    w_df["VWAP_Bars_Since_Cross"] = np.clip(bars_since_cross, 0, VWAP_PERSIST_CAP) / float(VWAP_PERSIST_CAP)
    w_df["VWAP_Crosses_30"] = vwap_cross.rolling(VWAP_CROSS_WINDOW, min_periods=1).sum() / float(VWAP_CROSS_WINDOW)
    w_df["VWAP_Behavior"] = np.where(
        bars_since_cross <= 5,
        0,
        np.where((bars_since_cross >= 30) & (np.abs(w_df["VWAP_Dist_ATR"]) >= 1.0), 2, 1),
    )

    daily = w_df.groupby(trading_day).agg(
        day_open=("Open", "first"),
        day_high=("High", "max"),
        day_low=("Low", "min"),
        day_close=("Close", "last"),
    )
    prev_high = daily["day_high"].shift(1).reindex(trading_day).to_numpy()
    prev_low = daily["day_low"].shift(1).reindex(trading_day).to_numpy()
    prev_close = daily["day_close"].shift(1).reindex(trading_day).to_numpy()
    day_open = daily["day_open"].reindex(trading_day).to_numpy()
    cum_high = w_df["High"].groupby(trading_day).cummax().to_numpy()
    cum_low = w_df["Low"].groupby(trading_day).cummin().to_numpy()

    overlap = np.maximum(0.0, np.minimum(prev_high, cum_high) - np.maximum(prev_low, cum_low))
    curr_range = np.maximum(cum_high - cum_low, 1e-9)
    overlap_ratio = overlap / curr_range
    overlap_ratio = np.where(np.isnan(prev_high), 0.0, overlap_ratio)
    w_df["Range_Overlap_Ratio"] = overlap_ratio

    gap = day_open - prev_close
    atr_vals = w_df["ATR"].to_numpy()
    gap_size_atr = np.divide(
        np.abs(gap),
        atr_vals,
        out=np.full_like(atr_vals, np.nan, dtype=float),
        where=atr_vals != 0,
    )
    gap_size_atr = np.where(np.isfinite(gap_size_atr), gap_size_atr, 0.0)
    w_df["Gap_Size_ATR"] = gap_size_atr
    gap_filled = (cum_low <= prev_close) & (cum_high >= prev_close)
    w_df["Gap_Filled"] = np.where(np.isnan(prev_close), 0, gap_filled.astype(int))

    atr = w_df["ATR"].to_numpy()
    atr_prev = np.roll(atr, ATR_SLOPE_LAG)
    atr_prev[:ATR_SLOPE_LAG] = np.nan
    atr_slope = np.divide(
        (atr - atr_prev),
        atr_prev,
        out=np.full_like(atr, np.nan, dtype=float),
        where=atr_prev != 0,
    )
    w_df["ATR_Slope"] = np.where(np.isfinite(atr_slope), atr_slope, 0.0)

    ema_fast = w_df["Close"].ewm(span=TREND_EMA_FAST, adjust=False).mean()
    ema_slow = w_df["Close"].ewm(span=TREND_EMA_SLOW, adjust=False).mean()
    trend_sign = np.sign(ema_fast - ema_slow).replace(0, np.nan).ffill().fillna(0)
    trend_change = trend_sign.ne(trend_sign.shift(1)).fillna(False)
    idx = np.arange(len(trend_sign))
    last_change = pd.Series(np.where(trend_change.to_numpy(), idx, -1)).cummax().to_numpy()
    bars_since_trend = idx - last_change
    bars_since_trend[last_change < 0] = idx[last_change < 0]
    w_df["Trend_Persistence"] = np.clip(bars_since_trend, 0, TREND_PERSIST_CAP) / float(TREND_PERSIST_CAP)

    hours = dt.dt.hour.to_numpy(dtype=int)
    session = np.full(len(hours), "CLOSED", dtype=object)
    session[(hours >= 18) | (hours < 3)] = "ASIA"
    session[(hours >= 3) & (hours < 8)] = "LONDON"
    session[(hours >= 8) & (hours < 12)] = "NY_AM"
    session[(hours >= 12) & (hours < 17)] = "NY_PM"
    session_start_hour = np.select(
        [session == "ASIA", session == "LONDON", session == "NY_AM", session == "NY_PM"],
        [18, 3, 8, 12],
        default=np.nan,
    )
    session_start = dt.dt.normalize() + pd.to_timedelta(session_start_hour, unit="h")
    mask_asia_early = (session == "ASIA") & (hours < 3)
    session_start = session_start.where(~mask_asia_early, session_start - pd.Timedelta(days=1))
    minutes_since = (dt - session_start).dt.total_seconds() / 60.0
    first_hour = (minutes_since >= 0) & (minutes_since < SESSION_OPEN_MINUTES)

    first_high = w_df["High"].where(first_hour).groupby(session_start).cummax()
    first_low = w_df["Low"].where(first_hour).groupby(session_start).cummin()
    first_close = w_df["Close"].where(first_hour).groupby(session_start).ffill()
    first_high = first_high.groupby(session_start).ffill()
    first_low = first_low.groupby(session_start).ffill()
    session_open = w_df["Open"].groupby(session_start).transform("first")

    open_range = first_high - first_low
    open_range_atr = open_range / w_df["ATR"]
    w_df["Session_Open_Range_ATR"] = open_range_atr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w_df["Session_Open_Drive"] = np.sign(first_close - session_open).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    session_id = np.full(len(session), -1, dtype=np.int8)
    session_id[session == "ASIA"] = 0
    session_id[session == "LONDON"] = 1
    session_id[session == "NY_AM"] = 2
    session_id[session == "NY_PM"] = 3
    w_df["Session_ID"] = session_id

    atr_z = w_df["ATR_ZScore"].to_numpy()
    atr_state = np.select([atr_z <= -0.5, atr_z <= 0.5, atr_z <= 1.5], [0, 1, 2], default=3).astype(np.int8)
    w_df["ATR_State"] = atr_state

    trend_p = w_df["Trend_Persistence"].to_numpy()
    is_trending = w_df["Is_Trending"].to_numpy()
    trend_state = np.where(is_trending <= 0, 0, np.where(trend_p >= 0.7, 2, 1)).astype(np.int8)
    w_df["Trend_State"] = trend_state

    liq_src = w_df["RVol_ZScore"] if "RVol_ZScore" in w_df.columns else w_df["Volume_ZScore"]
    liq_vals = liq_src.to_numpy()
    liq_state = np.select([liq_vals <= -0.5, liq_vals <= 0.5], [0, 1], default=2).astype(np.int8)
    w_df["Liquidity_State"] = liq_state
    w_df["Macro_Regime_ID"] = _map_macro_regime_id(w_df["datetime"])

    if cast_float32:
        float_cols = [c for c in w_df.columns if c != "datetime"]
        for col in float_cols:
            if pd.api.types.is_float_dtype(w_df[col]) or pd.api.types.is_integer_dtype(w_df[col]):
                w_df[col] = pd.to_numeric(w_df[col], errors="coerce").astype(np.float32)

    return w_df


def _session_for_hours(hours: np.ndarray) -> np.ndarray:
    session = np.full(len(hours), "OFF", dtype=object)
    for name, settings in (CONFIG.get("SESSIONS", {}) or {}).items():
        allowed = np.array(settings.get("HOURS", []), dtype=int)
        if allowed.size <= 0:
            continue
        session[np.isin(hours, allowed)] = str(name)
    return session


def _predict_proba_batch(model, x_df: pd.DataFrame) -> np.ndarray:
    if x_df.empty:
        return np.zeros(0, dtype=float)
    x_in = x_df
    cols = getattr(model, "feature_names_in_", None)
    if cols is not None:
        x_in = x_df.reindex(columns=list(cols), fill_value=0.0)
    probs = model.predict_proba(x_in)
    return np.asarray(probs[:, 1], dtype=float)


def compute_predictions(df_features: pd.DataFrame, session_manager) -> pd.DataFrame:
    """
    Batch-compute ML score/probabilities from precomputed features.
    """
    out = df_features.copy()
    if out.empty:
        out["ml_prob_up"] = np.nan
        out["ml_prob_down"] = np.nan
        out["ml_score"] = np.nan
        out["ml_session"] = "OFF"
        out["ml_regime_key"] = "unknown"
        return out

    dt_series = pd.to_datetime(out["datetime"], errors="coerce")
    hours = dt_series.dt.hour.to_numpy(dtype=int)
    sessions = _session_for_hours(hours)
    out["ml_session"] = sessions
    out["ml_prob_up"] = np.nan
    out["ml_prob_down"] = np.nan
    out["ml_score"] = np.nan
    out["ml_regime_key"] = "unknown"

    if session_manager is None:
        return out

    for session_name in np.unique(sessions):
        if session_name in {"OFF", "UNKNOWN", "CLOSED"}:
            continue
        mask = out["ml_session"] == session_name
        if not mask.any():
            continue
        setup_ts = pd.Timestamp(dt_series.loc[mask].iloc[0])
        setup = session_manager.get_current_setup(setup_ts.to_pydatetime() if hasattr(setup_ts, "to_pydatetime") else None)
        if not isinstance(setup, dict):
            continue
        model = setup.get("model")
        if model is None:
            continue

        if isinstance(model, dict):
            # Approximate split routing for batch inference.
            if "normal" in model:
                atr_state = pd.to_numeric(out.loc[mask, "ATR_State"], errors="coerce").fillna(1.0)
                regime_series = np.where(atr_state >= 3, "high", np.where(atr_state <= 0, "low", "normal"))
            else:
                hv = pd.to_numeric(out.loc[mask, "High_Volatility"], errors="coerce").fillna(0.0)
                regime_series = np.where(hv >= 0.5, "high", "low")
            out.loc[mask, "ml_regime_key"] = regime_series
            idx_mask = out.index[mask]
            for regime in ("low", "normal", "high"):
                reg_model = model.get(regime)
                if reg_model is None:
                    continue
                reg_idx = idx_mask[np.where(regime_series == regime)[0]]
                if len(reg_idx) == 0:
                    continue
                x_df = out.loc[reg_idx, ML_FEATURE_COLUMNS]
                prob_up = _predict_proba_batch(reg_model, x_df)
                out.loc[reg_idx, "ml_prob_up"] = prob_up
        else:
            x_df = out.loc[mask, ML_FEATURE_COLUMNS]
            prob_up = _predict_proba_batch(model, x_df)
            out.loc[mask, "ml_prob_up"] = prob_up
            out.loc[mask, "ml_regime_key"] = "single"

    out["ml_prob_down"] = 1.0 - pd.to_numeric(out["ml_prob_up"], errors="coerce")
    out["ml_score"] = out["ml_prob_up"]
    return out


def _resample_ohlcv(df: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    rule = f"{int(tf_minutes)}min"
    lower = {str(c).lower(): c for c in df.columns}
    open_col = lower.get("open")
    high_col = lower.get("high")
    low_col = lower.get("low")
    close_col = lower.get("close")
    volume_col = lower.get("volume")
    if not all((open_col, high_col, low_col, close_col)):
        raise KeyError("Missing OHLC columns for resample")
    agg_map = {
        open_col: "first",
        high_col: "max",
        low_col: "min",
        close_col: "last",
    }
    if volume_col is not None:
        agg_map[volume_col] = "sum"
    out = df.resample(rule, closed="left", label="left").agg(agg_map).dropna(subset=[open_col, high_col, low_col, close_col])
    rename_map = {
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close",
    }
    if volume_col is not None:
        rename_map[volume_col] = "volume"
    out = out.rename(columns=rename_map)
    if "volume" not in out.columns:
        out["volume"] = 0.0
    return out


def prepare_full_dataset(df_full: pd.DataFrame, *, session_manager=None) -> pd.DataFrame:
    """
    Backtest optimization pipeline:
    - compute full feature table once
    - batch inference once
    - cache to parquet
    - return index-aligned precomputed rows for each 1m bar timestamp
    """
    if df_full is None or df_full.empty:
        return pd.DataFrame()
    cfg = _opt_cfg()
    tf_minutes = int(CONFIG.get("ML_PHYSICS_TIMEFRAME_MINUTES", 1) or 1)
    align_mode = str(CONFIG.get("ML_PHYSICS_BAR_ALIGNMENT", "open") or "open").lower()
    if align_mode not in {"open", "close"}:
        align_mode = "open"

    symbol = _guess_symbol(df_full)
    start = str(pd.Timestamp(df_full.index.min()).date())
    end = str(pd.Timestamp(df_full.index.max()).date())
    suffix = _cache_suffix(df_full, tf_minutes=tf_minutes, align_mode=align_mode)

    fcache = get_feature_cache_path(cfg, symbol, start, f"{end}_{suffix}", tf_minutes, align_mode)
    pcache = get_prediction_cache_path(cfg, symbol, start, f"{end}_{suffix}", tf_minutes, align_mode)
    overwrite = bool(cfg.get("overwrite_cache", False))
    use_pred_cache = bool(cfg.get("prediction_cache", True))

    aligned_index = pd.DatetimeIndex(df_full.index)
    if aligned_index.tz is None:
        aligned_index = aligned_index.tz_localize(NY_TZ)
    else:
        aligned_index = aligned_index.tz_convert(NY_TZ)

    if use_pred_cache and pcache.exists() and not overwrite:
        try:
            cached = pd.read_parquet(pcache)
            if "datetime" in cached.columns:
                cached["datetime"] = pd.to_datetime(cached["datetime"], errors="coerce")
            cached.index = aligned_index
            logging.info("MLPhysics OPT: loaded prediction cache %s", pcache)
            return cached
        except Exception as exc:
            logging.warning("MLPhysics OPT: prediction cache load failed (%s): %s", pcache, exc)

    base_df = df_full
    if tf_minutes > 1:
        base_df = _resample_ohlcv(df_full, tf_minutes=tf_minutes)

    if fcache.exists() and not overwrite:
        try:
            feat_df = pd.read_parquet(fcache)
            if "datetime" in feat_df.columns:
                feat_df["datetime"] = pd.to_datetime(feat_df["datetime"], errors="coerce")
            feat_df.index = pd.DatetimeIndex(feat_df.index)
            if feat_df.index.tz is None:
                feat_df.index = feat_df.index.tz_localize(NY_TZ)
            else:
                feat_df.index = feat_df.index.tz_convert(NY_TZ)
            logging.info("MLPhysics OPT: loaded feature cache %s", fcache)
        except Exception as exc:
            logging.warning("MLPhysics OPT: feature cache load failed (%s): %s", fcache, exc)
            feat_df = compute_features(base_df, cast_float32=False)
    else:
        feat_df = compute_features(base_df, cast_float32=False)
        try:
            feat_df.to_parquet(fcache, engine="pyarrow", compression="zstd")
        except Exception:
            feat_df.to_parquet(fcache)
        logging.info("MLPhysics OPT: saved feature cache %s", fcache)

    pred_df = compute_predictions(feat_df, session_manager=session_manager)

    if tf_minutes <= 1:
        mapped_idx = aligned_index
        can_eval = np.ones(len(aligned_index), dtype=bool)
    else:
        mins = aligned_index.minute.to_numpy(dtype=int)
        if align_mode == "close":
            can_eval = (mins % tf_minutes) == (tf_minutes - 1)
            mapped_idx = aligned_index.floor(f"{tf_minutes}min")
        else:
            can_eval = (mins % tf_minutes) == 0
            mapped_idx = (aligned_index - pd.Timedelta(minutes=1)).floor(f"{tf_minutes}min")

    aligned = pred_df.reindex(mapped_idx)
    aligned.index = aligned_index
    aligned["ml_can_eval"] = can_eval.astype(np.int8)
    if len(aligned):
        off_mask = aligned["ml_can_eval"] <= 0
        if off_mask.any():
            cols = [c for c in ML_FEATURE_COLUMNS if c in aligned.columns]
            cols += [c for c in ("ml_prob_up", "ml_prob_down", "ml_score", "ml_session", "ml_regime_key") if c in aligned.columns]
            for col in cols:
                aligned.loc[off_mask, col] = np.nan

    try:
        if use_pred_cache:
            aligned.to_parquet(pcache, engine="pyarrow", compression="zstd")
            logging.info("MLPhysics OPT: saved prediction cache %s", pcache)
    except Exception as exc:
        logging.warning("MLPhysics OPT: prediction cache save failed (%s): %s", pcache, exc)

    return aligned
