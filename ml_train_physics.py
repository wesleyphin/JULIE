import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HistGradientBoostingClassifier = None

FALLBACK_PARAMS = {"sl_mult": 2.0, "tp_mult": 2.5, "atr_med": 1.5}


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
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    df = _load_csv(Path(args.csv))
    if args.max_rows and args.max_rows > 0:
        df = df.iloc[-args.max_rows:]

    df = _resample_ohlcv(df, args.timeframe_minutes)
    features = _build_features(df)

    sl_dist, tp_dist = _build_sltp_arrays(df, args.sltp_strategy)
    labels = _build_labels(df, args.horizon_bars, args.label_mode, args.drop_neutral, sl_dist, tp_dist)

    labels = labels.reindex(features.index)
    sl_dist = pd.Series(sl_dist, index=df.index).reindex(features.index).to_numpy()
    tp_dist = pd.Series(tp_dist, index=df.index).reindex(features.index).to_numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    risk_cfg = CONFIG.get("RISK", {})
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0))
    fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.5))
    fees_pts = (fees_per_side * 2.0) / point_value

    thresholds_report = {}
    metrics_report = {}

    for session_name, settings in CONFIG["SESSIONS"].items():
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

        idx = data.index
        train_idx, val_idx, test_idx = _split_index(idx, args.horizon_bars, args.val_frac, args.test_frac)

        train = data.loc[train_idx]
        val = data.loc[val_idx]
        test = data.loc[test_idx]

        if train.empty or val.empty or test.empty:
            logging.warning(f"{session_name}: not enough samples for train/val/test split")
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

        val_sl = pd.Series(session_sl, index=session_features.index).loc[val_idx].to_numpy()
        val_tp = pd.Series(session_tp, index=session_features.index).loc[val_idx].to_numpy()
        val_result = _optimize_threshold(
            calibrated.predict_proba(X_val)[:, 1],
            y_val.to_numpy(),
            val_sl,
            val_tp,
            fees_pts,
            args.thr_min,
            args.thr_max,
            args.thr_step,
        )

        test_sl = pd.Series(session_sl, index=session_features.index).loc[test_idx].to_numpy()
        test_tp = pd.Series(session_tp, index=session_features.index).loc[test_idx].to_numpy()
        test_metrics = _evaluate(calibrated, X_test, y_test, test_sl, test_tp, val_result["threshold"], fees_pts)

        out_path = out_dir / settings["MODEL_FILE"]
        joblib.dump(calibrated, out_path)
        logging.info(f"{session_name}: saved {out_path}")

        thresholds_report[session_name] = val_result
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
