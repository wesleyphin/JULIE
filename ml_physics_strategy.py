import logging
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import CONFIG
from dynamic_sltp_params import dynamic_sltp_engine
from session_manager import SessionManager
from strategy_base import Strategy
from volatility_filter import volatility_filter, VolRegime

# ==========================================
# ML FEATURE PIPELINE (Synced with ml_train_v11.py)
# ==========================================
# These constants & helpers mirror the institutional ML training script.
# They are used to build the exact same feature vector at runtime.

RSI_PERIOD = 9       # Faster for V-bottoms in high-vol regimes
ADX_PERIOD = 14      # Standard, robust
ATR_PERIOD = 14      # For dynamic volatility context
ZSCORE_WINDOW = 50   # Lookback for Z-score normalization
RVOL_LOOKBACK_DAYS = 20

ML_FEATURE_COLUMNS = [
    'Close_ZScore',
    'High_ZScore',
    'Low_ZScore',
    'ATR_ZScore',
    'Volatility_ZScore',
    'Range_ZScore',
    'Volume_ZScore',
    'Slope_ZScore',
    'RVol_ZScore',
    'RSI_Vel',
    'Adx_Vel',
    'RSI_Norm',
    'ADX_Norm',
    'Return_1',
    'Return_5',
    'Return_15',
    'Hour_Sin',
    'Hour_Cos',
    'Minute_Sin',
    'Minute_Cos',
    'DOW_Sin',
    'DOW_Cos',
    'Is_Trending',
    'Trend_Direction',
    'High_Volatility',
]

OPTIONAL_FEATURE_COLUMNS = {
    'RVol_ZScore',
}


def ml_calculate_rsi(series, period=RSI_PERIOD):
    """RSI calculation (same as ml_train_v11.calculate_rsi)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def ml_calculate_atr(high, low, close, period=ATR_PERIOD):
    """ATR for volatility-adjusted features (same as ml_train_v11.calculate_atr)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def ml_calculate_adx(high, low, close, period=ADX_PERIOD):
    """ADX + DI lines for regime detection (same as ml_train_v11.calculate_adx)."""
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed values
    atr = tr.rolling(window=period).mean()
    plus_series = pd.Series(plus_dm, index=high.index)
    minus_series = pd.Series(minus_dm, index=high.index)
    plus_di = 100 * plus_series.rolling(window=period).mean() / atr
    minus_di = 100 * minus_series.rolling(window=period).mean() / atr

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di


def ml_calculate_slope(series, window=15):
    """Rolling linear regression slope (same as ml_train_v11.calculate_slope)."""
    def slope_calc(y):
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y))
        covariance = np.cov(x, y)[0, 1]
        variance = np.var(x)
        return float(covariance / variance) if variance != 0 else 0.0
    return series.rolling(window=window).apply(slope_calc, raw=False)


def ml_zscore_normalize(series, window=ZSCORE_WINDOW):
    """Z-score normalization (same as ml_train_v11.zscore_normalize)."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    zscore = (series - rolling_mean) / (rolling_std + 1e-10)
    return zscore


def ml_calculate_rvol(volume, datetime_index, lookback_days=RVOL_LOOKBACK_DAYS):
    """Relative Volume: current vol / avg vol at this time of day (ml_train_v11.calculate_rvol)."""
    df_temp = pd.DataFrame({'volume': volume, 'datetime': datetime_index})
    df_temp['hour'] = df_temp['datetime'].dt.hour
    df_temp['minute'] = df_temp['datetime'].dt.minute
    df_temp['time_bucket'] = df_temp['hour'] * 60 + df_temp['minute']

    avg_vol_by_time = df_temp.groupby('time_bucket')['volume'].transform(
        lambda x: x.rolling(window=lookback_days * 5, min_periods=10).mean()
    )
    rvol = volume / (avg_vol_by_time + 1)
    return rvol


def ml_encode_cyclical_time(datetime_series):
    """Cyclical time encoding using sin/cos (same as ml_train_v11.encode_cyclical_time)."""
    hour = datetime_series.dt.hour
    minute = datetime_series.dt.minute
    day_of_week = datetime_series.dt.dayofweek

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    dow_sin = np.sin(2 * np.pi * day_of_week / 5)
    dow_cos = np.cos(2 * np.pi * day_of_week / 5)

    return hour_sin, hour_cos, minute_sin, minute_cos, dow_sin, dow_cos


class MLPhysicsStrategy(Strategy):
    """
    Session-specialized ML strategy using four neural networks.
    Mirrors juliemlsession.py behavior with identical feature pipeline.
    """

    def __init__(self):
        self.sm = SessionManager()
        self.window_size = CONFIG.get("WINDOW_SIZE", 15)
        self.model_loaded = any(self.sm.brains.values())  # True if at least one model loaded
        self._last_feature_log_ts = 0.0
        self._feature_log_interval = 300.0
        self._logged_resample = False

    def _log_feature_issue(self, cols, bar_count, optional=False):
        now = time.time()
        if now - self._last_feature_log_ts < self._feature_log_interval:
            return
        status = "optional" if optional else "critical"
        level = logging.info if optional else logging.warning
        level(f"MLPhysics: {status} feature NaNs: {', '.join(cols)} | bars={bar_count}")
        self._last_feature_log_ts = now

    def _resample_ohlcv(self, df: pd.DataFrame, minutes: int) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset({c.lower() for c in df.columns}):
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            except Exception:
                return df
        rule = f"{int(minutes)}min"
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return resampled

    def calculate_slope(self, values):
        """Exact copy from juliemlsession.py"""
        y = np.array(values)
        x = np.arange(len(y))
        if len(y) < 2:
            return 0
        cov = np.cov(x, y)[0, 1]
        var = np.var(x)
        return cov / var if var != 0 else 0

    def prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Build ML feature vector for the latest bar using the same pipeline
        as ml_train_v11.prepare_data + get_feature_columns().

        Expects:
            df index or 'datetime' column: timezone-aware or naive timestamps
            price/volume columns: open, high, low, close, volume (any case)
        Returns:
            Single-row DataFrame with ML_FEATURE_COLUMNS, or None if not enough history.
        """
        if df is None or len(df) < 200:
            # Need sufficient history for ATR / ADX / Z-score / RVol windows
            return None

        w_df = df.copy()

        # Ensure we have a datetime column
        if 'datetime' in w_df.columns:
            w_df['datetime'] = pd.to_datetime(w_df['datetime'])
        else:
            w_df['datetime'] = pd.to_datetime(w_df.index)

        # Timezone handling – align to US/Eastern like training script
        try:
            if w_df['datetime'].dt.tz is None:
                # Assume timestamps are already ET but naive
                w_df['datetime'] = w_df['datetime'].dt.tz_localize('US/Eastern')
            else:
                w_df['datetime'] = w_df['datetime'].dt.tz_convert('US/Eastern')
        except Exception:
            # Fallback: best-effort localization
            try:
                w_df['datetime'] = w_df['datetime'].dt.tz_localize('US/Eastern')
            except Exception:
                pass

        w_df.sort_values('datetime', inplace=True)

        # Normalize column names to the format used during training
        col_map = {}
        for col in w_df.columns:
            cl = col.lower()
            if cl == 'open':
                col_map[col] = 'Open'
            elif cl == 'high':
                col_map[col] = 'High'
            elif cl == 'low':
                col_map[col] = 'Low'
            elif cl == 'close':
                col_map[col] = 'Close'
            elif cl == 'volume':
                col_map[col] = 'Volume'
        w_df.rename(columns=col_map, inplace=True)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for rc in required_cols:
            if rc not in w_df.columns:
                logging.warning(f"MLPhysics: missing column '{rc}' in feature builder.")
                return None

        # ==========================
        # RAW TECHNICAL INDICATORS
        # ==========================
        w_df['RSI'] = ml_calculate_rsi(w_df['Close'], period=RSI_PERIOD)
        w_df['ATR'] = ml_calculate_atr(w_df['High'], w_df['Low'], w_df['Close'], period=ATR_PERIOD)
        adx, plus_di, minus_di = ml_calculate_adx(w_df['High'], w_df['Low'], w_df['Close'], period=ADX_PERIOD)
        w_df['ADX'] = adx
        w_df['PLUS_DI'] = plus_di
        w_df['MINUS_DI'] = minus_di

        # Velocity-style features
        w_df['RSI_Vel'] = w_df['RSI'].diff(3)
        w_df['Adx_Vel'] = w_df['ADX'].diff(3)
        w_df['Slope'] = ml_calculate_slope(w_df['Close'], window=15)

        # Volatility and range
        w_df['Volatility'] = w_df['Close'].rolling(window=15).std()
        w_df['Range'] = w_df['High'] - w_df['Low']

        # Returns (already stationary)
        w_df['Return_1'] = w_df['Close'].pct_change(1)
        w_df['Return_5'] = w_df['Close'].pct_change(5)
        w_df['Return_15'] = w_df['Close'].pct_change(15)

        # ==========================
        # Z-SCORE NORMALIZATION
        # ==========================
        w_df['Close_ZScore'] = ml_zscore_normalize(w_df['Close'], window=ZSCORE_WINDOW)
        w_df['High_ZScore'] = ml_zscore_normalize(w_df['High'], window=ZSCORE_WINDOW)
        w_df['Low_ZScore'] = ml_zscore_normalize(w_df['Low'], window=ZSCORE_WINDOW)
        w_df['ATR_ZScore'] = ml_zscore_normalize(w_df['ATR'], window=ZSCORE_WINDOW)
        w_df['Volatility_ZScore'] = ml_zscore_normalize(w_df['Volatility'], window=ZSCORE_WINDOW)
        w_df['Range_ZScore'] = ml_zscore_normalize(w_df['Range'], window=ZSCORE_WINDOW)
        w_df['Volume_ZScore'] = ml_zscore_normalize(w_df['Volume'], window=ZSCORE_WINDOW)

        # RSI / ADX bounded normalization
        w_df['RSI_Norm'] = (w_df['RSI'] - 50.0) / 50.0
        w_df['ADX_Norm'] = (w_df['ADX'] / 50.0) - 1.0

        # Slope Z-score
        w_df['Slope_ZScore'] = ml_zscore_normalize(w_df['Slope'], window=ZSCORE_WINDOW)

        # ==========================
        # RELATIVE VOLUME
        # ==========================
        w_df['RVol'] = ml_calculate_rvol(w_df['Volume'], w_df['datetime'], lookback_days=RVOL_LOOKBACK_DAYS)
        w_df['RVol_ZScore'] = ml_zscore_normalize(w_df['RVol'], window=ZSCORE_WINDOW)

        # ==========================
        # CYCLICAL TIME ENCODING
        # ==========================
        (w_df['Hour_Sin'], w_df['Hour_Cos'],
         w_df['Minute_Sin'], w_df['Minute_Cos'],
         w_df['DOW_Sin'], w_df['DOW_Cos']) = ml_encode_cyclical_time(w_df['datetime'])

        # ==========================
        # REGIME FEATURES
        # ==========================
        w_df['Is_Trending'] = (w_df['ADX'] > 25).astype(int)
        w_df['Trend_Direction'] = np.sign(w_df['PLUS_DI'] - w_df['MINUS_DI'])
        w_df['ATR_MA'] = w_df['ATR'].rolling(window=100).mean()
        w_df['High_Volatility'] = (w_df['ATR'] > w_df['ATR_MA'] * 1.5).astype(int)

        # Final row must have a complete feature vector (no NaNs)
        last = w_df.iloc[-1]
        feature_vals = []
        critical_missing = []
        optional_filled = []
        for col in ML_FEATURE_COLUMNS:
            val = last.get(col, np.nan)
            if pd.isna(val):
                if col in OPTIONAL_FEATURE_COLUMNS:
                    val = 0.0
                    optional_filled.append(col)
                else:
                    critical_missing.append(col)
            feature_vals.append(val)

        if critical_missing:
            self._log_feature_issue(sorted(critical_missing), len(w_df), optional=False)
            return None
        if optional_filled:
            self._log_feature_issue(sorted(optional_filled), len(w_df), optional=True)

        X = pd.DataFrame([feature_vals], columns=ML_FEATURE_COLUMNS)
        return X

    @staticmethod
    def _efficiency_ratio(close: pd.Series, window: int) -> Optional[float]:
        if close is None or len(close) < window + 1:
            return None
        recent = close.iloc[-(window + 1):]
        net = float(abs(recent.iloc[-1] - recent.iloc[0]))
        denom = float(recent.diff().abs().sum())
        if denom <= 0:
            return 0.0
        return net / denom

    @staticmethod
    def _vwap_crosses(df: pd.DataFrame, window: int) -> Optional[int]:
        if df is None or len(df) < window:
            return None
        if "close" not in df.columns or "volume" not in df.columns:
            return None
        price = df["close"]
        volume = df["volume"]
        pv = (price * volume).rolling(window).sum()
        vv = volume.rolling(window).sum()
        vwap = pv / vv.replace(0, np.nan)
        delta = (price - vwap).iloc[-window:]
        sign = np.sign(delta.to_numpy())
        for i in range(1, len(sign)):
            if sign[i] == 0:
                sign[i] = sign[i - 1]
        if len(sign) < 2:
            return 0
        crosses = np.sum(sign[1:] * sign[:-1] < 0)
        return int(crosses)

    def on_bar(self, df: pd.DataFrame, current_time=None) -> Optional[Dict]:
        """
        Adapted from juliemlsession.py on_bar method.
        Uses SessionManager to get the correct model and parameters.
        Uses the full dataframe passed in (which already has 500 bars of history).
        """
        # 1. Get current session setup
        setup = self.sm.get_current_setup(current_time)

        if setup is None:
            logging.info("💤 Market Closed (No active session strategy)")
            return None

        if setup.get("disabled"):
            logging.info(f"⚠️ MLPhysics {setup['name']} disabled by guardrails")
            return None

        if setup['model'] is None:
            logging.info(f"⚠️ Session {setup['name']} active, but brain file is missing!")
            return None

        # 2. Convert df to the format expected by prepare_features
        # Need columns: datetime, Open, High, Low, Close, Volume
        hist_df = df.copy()
        tf_minutes = setup.get("timeframe_minutes") or CONFIG.get("ML_PHYSICS_TIMEFRAME_MINUTES", 1)
        if isinstance(tf_minutes, (int, float)) and tf_minutes > 1:
            hist_df = self._resample_ohlcv(hist_df, int(tf_minutes))
            if not self._logged_resample:
                logging.info(f"MLPhysics: resampling to {int(tf_minutes)}min for feature build")
                self._logged_resample = True
        hist_df['datetime'] = hist_df.index

        # Rename columns to match expected format (capitalize)
        col_map = {}
        for col in hist_df.columns:
            if col.lower() == 'open':
                col_map[col] = 'open'
            elif col.lower() == 'high':
                col_map[col] = 'high'
            elif col.lower() == 'low':
                col_map[col] = 'low'
            elif col.lower() == 'close':
                col_map[col] = 'close'
            elif col.lower() == 'volume':
                col_map[col] = 'volume'
        hist_df = hist_df.rename(columns=col_map)

        # Ensure we have enough data
        if len(hist_df) < 20:
            logging.info(f"📊 {setup['name']}: Building Physics Data ({len(hist_df)}/20)...")
            return None

        # 3. Run Prediction
        X = self.prepare_features(hist_df)

        if X is not None:
            model = setup.get("model")
            threshold = setup.get("threshold")
            vol_split_cfg = CONFIG.get("ML_PHYSICS_VOL_SPLIT", {}) or {}
            vol_split_3way = CONFIG.get("ML_PHYSICS_VOL_SPLIT_3WAY", {}) or {}
            vol_guard = CONFIG.get("ML_PHYSICS_VOL_GUARD", {}) or {}
            gate_cfg = CONFIG.get("ML_PHYSICS_HIGH_VOL_DIRECTIONAL_GATE", {}) or {}
            bump_cfg = CONFIG.get("ML_PHYSICS_HIGH_VOL_THRESHOLD_BUMP", {}) or {}

            split_regimes = set(setup.get("split_regimes") or [])
            split_3way = bool(vol_split_3way.get("enabled")) and setup.get("name") in set(vol_split_3way.get("sessions", []))
            if "normal" in split_regimes:
                split_3way = True

            regime_key = None
            high_vol = False
            if split_3way:
                try:
                    vol_regime, _, _ = volatility_filter.get_regime(hist_df)
                except Exception:
                    vol_regime = VolRegime.LOW
                if vol_regime == VolRegime.ULTRA_LOW:
                    vol_regime = VolRegime.LOW
                if vol_regime == VolRegime.HIGH:
                    regime_key = "high"
                elif vol_regime == VolRegime.NORMAL:
                    regime_key = "normal"
                else:
                    regime_key = "low"
                high_vol = regime_key == "high"
            else:
                feature_name = (
                    gate_cfg.get("feature")
                    or vol_guard.get("feature")
                    or vol_split_cfg.get("feature")
                    or "High_Volatility"
                )
                if feature_name in X.columns:
                    try:
                        high_vol = float(X.iloc[0].get(feature_name, 0.0)) >= 0.5
                    except Exception:
                        high_vol = False
                regime_key = "high" if high_vol else "low"

            # Volatility split: choose model/threshold by regime
            if setup.get("split") and isinstance(model, dict):
                regime = regime_key or ("high" if high_vol else "low")
                if regime in (setup.get("disabled_regimes") or set()):
                    logging.info(f"⚠️ MLPhysics {setup['name']} {regime} disabled by guardrails")
                    return None
                model_candidate = model.get(regime)
                if model_candidate is None and regime == "normal":
                    model_candidate = model.get("low") or model.get("high")
                    if model_candidate is not None:
                        logging.info(
                            f"⚠️ MLPhysics {setup['name']} normal model missing; "
                            "falling back to low/high"
                        )
                model = model_candidate
                if model is None:
                    logging.info(f"⚠️ MLPhysics {setup['name']} missing {regime} model")
                    return None
            else:
                # Optional volatility guard (skip ML trades during high-vol regimes)
                if vol_guard.get("enabled"):
                    guard_sessions = set(vol_guard.get("sessions", []))
                    if setup.get("name") in guard_sessions and high_vol:
                        logging.info(f"⚠️ MLPhysics {setup['name']} blocked (high volatility)")
                        return None

            if isinstance(threshold, dict):
                key = regime_key or ("high" if high_vol else "low")
                threshold = threshold.get(key, threshold.get("low", threshold.get("high", setup.get("threshold"))))

            ny_normal_cfg = CONFIG.get("ML_PHYSICS_NY_NORMAL_FILTER", {}) or {}
            normal_margin = None
            if ny_normal_cfg.get("enabled") and setup.get("name") in ("NY_AM", "NY_PM") and regime_key == "normal":
                er_window = int(ny_normal_cfg.get("er_window", 30) or 30)
                vwap_window = int(ny_normal_cfg.get("vwap_cross_window", 60) or 60)
                er_min = float(ny_normal_cfg.get("er_min", 0.25) or 0.25)
                vwap_cross_max = int(ny_normal_cfg.get("vwap_cross_max", 3) or 3)
                er_value = self._efficiency_ratio(hist_df["close"], er_window)
                vwap_crosses = self._vwap_crosses(hist_df, vwap_window)
                if er_value is None or vwap_crosses is None:
                    logging.info(f"⚠️ MLPhysics {setup['name']} normal: insufficient data for ER/VWAP filter")
                    return None
                is_trend = (er_value >= er_min) and (vwap_crosses <= vwap_cross_max)
                if not is_trend and ny_normal_cfg.get("block_chop", True):
                    logging.info(
                        f"⚠️ MLPhysics {setup['name']} normal chop blocked "
                        f"(ER={er_value:.2f} VWAPx={vwap_crosses})"
                    )
                    return None
                try:
                    normal_margin = float(ny_normal_cfg.get("margin", 0.0) or 0.0)
                except Exception:
                    normal_margin = None

            # High-vol threshold bump
            if high_vol and bump_cfg.get("enabled"):
                bump_sessions = set(bump_cfg.get("sessions", []))
                if setup.get("name") in bump_sessions:
                    try:
                        bump = float(bump_cfg.get("bump", 0.0) or 0.0)
                    except Exception:
                        bump = 0.0
                    if bump > 0:
                        try:
                            max_thr = float(bump_cfg.get("max_threshold", 0.95))
                        except Exception:
                            max_thr = 0.95
                        new_threshold = min(max_thr, float(threshold) + bump)
                        if new_threshold != threshold:
                            logging.info(
                                f"⚠️ MLPhysics {setup['name']} high-vol threshold bump: "
                                f"{threshold:.2f} -> {new_threshold:.2f}"
                            )
                        threshold = new_threshold

            # High-vol directional gate config
            gate_block_sides = set()
            gate_min_conf = None
            if high_vol and gate_cfg.get("enabled"):
                overrides = gate_cfg.get("overrides", {}) or {}
                sess_cfg = overrides.get(setup.get("name"), {}) if isinstance(overrides, dict) else {}
                gate_block_sides = {str(s).upper() for s in sess_cfg.get("block", [])}
                if gate_block_sides:
                    delta = sess_cfg.get("min_conf_delta", gate_cfg.get("min_conf_delta", 0.0))
                    min_conf = sess_cfg.get("min_conf", gate_cfg.get("min_conf"))
                    if min_conf is None:
                        min_conf = float(threshold) + float(delta or 0.0)
                    else:
                        try:
                            min_conf = float(min_conf)
                        except Exception:
                            min_conf = float(threshold)
                        if delta:
                            min_conf = max(min_conf, float(threshold) + float(delta))
                    try:
                        max_conf = float(sess_cfg.get("max_conf", gate_cfg.get("max_conf", 0.95)))
                    except Exception:
                        max_conf = 0.95
                    gate_min_conf = min(max_conf, min_conf)

            # Align columns to what the specific brain expects
            if hasattr(model, "feature_names_in_"):
                X = X.reindex(columns=model.feature_names_in_, fill_value=0)

            try:
                # Ask the Specialist Brain
                prob_up = model.predict_proba(X)[0][1]
                prob_down = 1.0 - prob_up

                status = "💚" if prob_up > 0.5 else "🔴"
                req = threshold if isinstance(threshold, (int, float)) else setup['threshold']
                short_req = 1.0 - req

                if regime_key == "normal" and normal_margin is not None and normal_margin > 0:
                    margin_val = abs(prob_up - 0.5)
                    if margin_val < normal_margin:
                        logging.info(
                            f"⚠️ MLPhysics {setup['name']} normal margin block "
                            f"(margin {margin_val:.3f} < {normal_margin:.3f})"
                        )
                        return None

                logging.info(
                    f"{setup['name']} Analysis {status} | "
                    f"ConfUp: {prob_up:.1%} (Req>= {req:.1%}) | "
                    f"ConfDn: {prob_down:.1%} (Req>= {short_req:.1%})"
                )

                # LONG signal: high probability of up move
                if prob_up >= req:
                    if high_vol and "LONG" in gate_block_sides and gate_min_conf is not None:
                        if prob_up < gate_min_conf:
                            logging.info(
                                f"⚠️ MLPhysics {setup['name']} LONG blocked in high vol "
                                f"(conf {prob_up:.1%} < gate {gate_min_conf:.1%})"
                            )
                            return None
                    # Get dynamic SL/TP from engine
                    sltp = dynamic_sltp_engine.calculate_dynamic_sltp(hist_df)
                    logging.info(
                        f"🎯 {setup['name']} LONG SIGNAL CONFIRMED "
                        f"(ConfUp={prob_up:.1%} >= {req:.1%})"
                    )
                    dynamic_sltp_engine.log_params(sltp)
                    return {
                        "strategy": f"MLPhysics_{setup['name']}",
                        "side": "LONG",
                        "tp_dist": sltp['tp_dist'],
                        "sl_dist": sltp['sl_dist'],
                        "ml_confidence": float(prob_up),
                        "ml_threshold": float(req),
                    }

                # SHORT signal: high probability of down move (low prob_up)
                elif prob_up <= (1.0 - req):
                    if high_vol and "SHORT" in gate_block_sides and gate_min_conf is not None:
                        if prob_down < gate_min_conf:
                            logging.info(
                                f"⚠️ MLPhysics {setup['name']} SHORT blocked in high vol "
                                f"(conf {prob_down:.1%} < gate {gate_min_conf:.1%})"
                            )
                            return None
                    sltp = dynamic_sltp_engine.calculate_dynamic_sltp(hist_df)
                    logging.info(
                        f"🎯 {setup['name']} SHORT SIGNAL CONFIRMED "
                        f"(ConfDn={prob_down:.1%} >= {short_req:.1%})"
                    )
                    dynamic_sltp_engine.log_params(sltp)
                    return {
                        "strategy": f"MLPhysics_{setup['name']}",
                        "side": "SHORT",
                        "tp_dist": sltp['tp_dist'],
                        "sl_dist": sltp['sl_dist'],
                        "ml_confidence": float(prob_down),
                        "ml_threshold": float(req),
                    }

            except Exception as e:
                logging.error(f"Prediction Error: {e}")

        return None
