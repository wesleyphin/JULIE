from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HistGradientBoostingClassifier = None

try:
    from sklearn.frozen import FrozenEstimator
except Exception:
    FrozenEstimator = None

from backtest_symbol_context import apply_symbol_mode, choose_symbol
from config import CONFIG
from ml_physics_legacy_experiment_common import (
    ML_FEATURE_COLUMNS,
    build_feature_frame,
    build_sltp_arrays,
    normalize_ohlcv_frame,
    resample_ohlcv,
)


NY_TZ = ZoneInfo("America/New_York")


def _resolve_source(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = Path(__file__).resolve().parent / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"Data file not found: {path_arg}")


def _parse_datetime(value: str, *, is_end: bool) -> pd.Timestamp:
    text = str(value or "").strip()
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        if len(text) <= 10 and is_end:
            ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        ts = ts.tz_localize(NY_TZ)
    else:
        ts = ts.tz_convert(NY_TZ)
    return ts


def _load_csv(path: Path) -> pd.DataFrame:
    with path.open("r", errors="ignore") as handle:
        first = handle.readline()
        second = handle.readline()
        needs_skip = "Time Series" in first and "Date" in second

    df = pd.read_csv(path, skiprows=1 if needs_skip else 0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    date_col = None
    for name in ("ts_event", "timestamp", "datetime", "date"):
        if name in df.columns:
            date_col = name
            break
    if date_col is None:
        raise ValueError("CSV missing timestamp column")

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', "").str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dt_index = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.loc[~dt_index.isna()].copy()
    dt_index = pd.DatetimeIndex(dt_index.loc[~dt_index.isna()]).tz_convert(NY_TZ)
    df.index = dt_index
    df = df.loc[~df.index.duplicated(keep="last")]
    return df


def _load_source_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = _load_csv(path)
    lower_map = {str(col).lower(): col for col in df.columns}
    if not isinstance(df.index, pd.DatetimeIndex):
        dt_col = None
        for name in ("datetime", "timestamp", "ts_event", "date"):
            if name in lower_map:
                dt_col = lower_map[name]
                break
        if dt_col is None:
            raise ValueError(f"Source missing datetime information: {path}")
        dt_index = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.loc[~dt_index.isna()].copy()
        dt_index = pd.DatetimeIndex(dt_index.loc[~dt_index.isna()])
        if dt_index.tz is None:
            dt_index = dt_index.tz_localize(NY_TZ)
        else:
            dt_index = dt_index.tz_convert(NY_TZ)
        df.index = dt_index
    else:
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize(NY_TZ)
        else:
            idx = idx.tz_convert(NY_TZ)
        df.index = idx
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="last")]
    return df


def _prepare_source_frame(
    source_path: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    symbol_mode: str,
    symbol_method: str,
) -> pd.DataFrame:
    df = _load_source_frame(source_path)
    df = df[(df.index >= start_time) & (df.index <= end_time)]
    if df.empty:
        raise SystemExit("No rows found in requested range.")

    if "symbol" in {str(col).lower() for col in df.columns}:
        symbol_col = next(col for col in df.columns if str(col).lower() == "symbol")
        if symbol_mode != "single":
            filtered, _, _ = apply_symbol_mode(df.rename(columns={symbol_col: "symbol"}), symbol_mode, symbol_method)
            df = filtered
        else:
            selected = choose_symbol(df.rename(columns={symbol_col: "symbol"}), None)
            df = df[df[symbol_col].astype(str) == str(selected)]
        df = df.drop(columns=[symbol_col], errors="ignore")

    return normalize_ohlcv_frame(df)


def _build_labels(
    df: pd.DataFrame,
    *,
    horizon_bars: int,
    label_mode: str,
    drop_neutral: bool,
    sl_dist: np.ndarray,
    tp_dist: np.ndarray,
) -> pd.Series:
    n = len(df)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    labels = np.full(n, np.nan, dtype=float)

    if label_mode == "return":
        future_close = pd.Series(closes, index=df.index).shift(-horizon_bars)
        future_return = (future_close / df["close"]) - 1.0
        return (future_return > 0).astype(float)

    max_i = n - horizon_bars - 1
    for i in range(max_i):
        up = closes[i] + tp_dist[i]
        dn = closes[i] - sl_dist[i]
        hit = None
        for j in range(1, horizon_bars + 1):
            hi = highs[i + j]
            lo = lows[i + j]
            if hi >= up and lo <= dn:
                hit = "both"
                break
            if hi >= up:
                labels[i] = 1.0
                hit = "tp"
                break
            if lo <= dn:
                labels[i] = 0.0
                hit = "sl"
                break
        if hit is None or hit == "both":
            if not drop_neutral:
                labels[i] = 1.0 if closes[i + horizon_bars] > closes[i] else 0.0

    return pd.Series(labels, index=df.index)


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
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return np.ones(len(labels), dtype=float)
    ratio = float(neg) / float(pos)
    return np.where(labels.to_numpy() == 1, ratio, 1.0).astype(float)


def _threshold_score(
    *,
    avg_pnl: float,
    trades: int,
    coverage: float,
    coverage_target: Optional[float],
    coverage_penalty: float,
    thr_score_penalty: float,
) -> float:
    if trades <= 0:
        return -1e12
    score = float(avg_pnl)
    score -= float(thr_score_penalty) / float(np.sqrt(max(trades, 1)))
    if coverage_target is not None and np.isfinite(coverage_target):
        score -= float(coverage_penalty) * abs(float(coverage) - float(coverage_target))
    return score


def _optimize_threshold(
    *,
    proba: np.ndarray,
    labels: np.ndarray,
    sl_dist: np.ndarray,
    tp_dist: np.ndarray,
    fees_pts: float,
    preset: dict,
) -> dict:
    best = {
        "threshold": None,
        "score": -1e12,
        "avg_pnl": -1e12,
        "total_pnl": -1e12,
        "trades": 0,
        "win_rate": 0.0,
        "coverage": 0.0,
        "long_trades": 0,
        "short_trades": 0,
        "side_share": 0.0,
    }

    total_rows = len(labels)
    thresholds = np.arange(
        float(preset.get("thr_min", 0.52) or 0.52),
        float(preset.get("thr_max", 0.75) or 0.75) + 1e-9,
        float(preset.get("thr_step", 0.01) or 0.01),
    )
    require_both_sides = bool(preset.get("require_both_sides", True))
    min_val_trades = int(preset.get("min_val_trades", 0) or 0)
    min_long_trades = int(preset.get("min_long_trades", 0) or 0)
    min_short_trades = int(preset.get("min_short_trades", 0) or 0)
    max_side_share = float(preset.get("max_side_share", 1.0) or 1.0)
    coverage_min_raw = preset.get("coverage_min")
    coverage_min = None if coverage_min_raw in (None, "") else float(coverage_min_raw)
    coverage_max_raw = preset.get("coverage_max")
    coverage_max = None if coverage_max_raw in (None, "") else float(coverage_max_raw)
    coverage_target_raw = preset.get("coverage_target")
    coverage_target = None if coverage_target_raw in (None, "") else float(coverage_target_raw)
    coverage_penalty = float(preset.get("coverage_penalty", 0.0) or 0.0)
    thr_score_penalty = float(preset.get("thr_score_penalty", 0.0) or 0.0)

    for thr in thresholds:
        long_mask = proba >= thr
        short_mask = proba <= (1.0 - thr)
        take = long_mask | short_mask
        trades = int(np.sum(take))
        if trades < min_val_trades:
            continue

        long_trades = int(np.sum(long_mask))
        short_trades = int(np.sum(short_mask))
        if require_both_sides and (long_trades < min_long_trades or short_trades < min_short_trades):
            continue
        side_share = float(max(long_trades, short_trades)) / float(trades) if trades else 0.0
        if side_share > max_side_share:
            continue

        coverage = float(trades) / float(total_rows) if total_rows else 0.0
        if coverage_min is not None and coverage < coverage_min:
            continue
        if coverage_max is not None and coverage > coverage_max:
            continue

        pnl = np.zeros_like(proba, dtype=float)
        pnl[long_mask & (labels == 1)] = tp_dist[long_mask & (labels == 1)]
        pnl[long_mask & (labels == 0)] = -sl_dist[long_mask & (labels == 0)]
        pnl[short_mask & (labels == 0)] = tp_dist[short_mask & (labels == 0)]
        pnl[short_mask & (labels == 1)] = -sl_dist[short_mask & (labels == 1)]
        pnl[take] -= fees_pts

        avg_pnl = float(np.mean(pnl[take])) if trades else -1e12
        total_pnl = float(np.sum(pnl[take])) if trades else -1e12
        wins = int(np.sum(pnl[take] > 0)) if trades else 0
        win_rate = float(wins) / float(trades) if trades else 0.0
        score = _threshold_score(
            avg_pnl=avg_pnl,
            trades=trades,
            coverage=coverage,
            coverage_target=coverage_target,
            coverage_penalty=coverage_penalty,
            thr_score_penalty=thr_score_penalty,
        )
        if score > best["score"] or (score == best["score"] and total_pnl > best["total_pnl"]):
            best.update(
                {
                    "threshold": float(thr),
                    "score": float(score),
                    "avg_pnl": avg_pnl,
                    "total_pnl": total_pnl,
                    "trades": trades,
                    "win_rate": win_rate,
                    "coverage": coverage,
                    "long_trades": long_trades,
                    "short_trades": short_trades,
                    "side_share": side_share,
                }
            )
    return best


def _evaluate(model, X: pd.DataFrame, y: pd.Series, sl_dist: np.ndarray, tp_dist: np.ndarray, threshold: float, fees_pts: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    labels = y.to_numpy(dtype=int)
    long_mask = proba >= threshold
    short_mask = proba <= (1.0 - threshold)
    take = long_mask | short_mask
    pnl = np.zeros_like(proba, dtype=float)
    pnl[long_mask & (labels == 1)] = tp_dist[long_mask & (labels == 1)]
    pnl[long_mask & (labels == 0)] = -sl_dist[long_mask & (labels == 0)]
    pnl[short_mask & (labels == 0)] = tp_dist[short_mask & (labels == 0)]
    pnl[short_mask & (labels == 1)] = -sl_dist[short_mask & (labels == 1)]
    pnl[take] -= fees_pts

    preds = (proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(labels, proba)
    except ValueError:
        auc = float("nan")

    trade_count = int(np.sum(take))
    return {
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(auc),
        "trade_count": trade_count,
        "long_trades": int(np.sum(long_mask)),
        "short_trades": int(np.sum(short_mask)),
        "avg_pnl": float(np.mean(pnl[take])) if trade_count else 0.0,
        "total_pnl": float(np.sum(pnl[take])) if trade_count else 0.0,
        "win_rate": float(np.sum(pnl[take] > 0) / trade_count) if trade_count else 0.0,
    }


def _fit_calibrated_model(base_model, X_val: pd.DataFrame, y_val: pd.Series):
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid", cv=None)
        calibrated.fit(X_val, y_val)
        return calibrated
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def _relaxed_legacy_preset(preset: dict) -> dict:
    relaxed = dict(preset or {})
    relaxed["min_val_trades"] = min(int(relaxed.get("min_val_trades", 20) or 20), 20)
    relaxed["require_both_sides"] = False
    relaxed["min_long_trades"] = 0
    relaxed["min_short_trades"] = 0
    relaxed["max_side_share"] = 1.0
    relaxed["coverage_min"] = None
    relaxed["coverage_max"] = None
    relaxed["coverage_target"] = None
    relaxed["coverage_penalty"] = 0.0
    relaxed["thr_score_penalty"] = min(float(relaxed.get("thr_score_penalty", 0.0) or 0.0), 0.1)
    relaxed["thr_min"] = min(float(relaxed.get("thr_min", 0.50) or 0.50), 0.50)
    relaxed["thr_max"] = max(float(relaxed.get("thr_max", 0.75) or 0.75), 0.75)
    relaxed["thr_step"] = float(relaxed.get("thr_step", 0.01) or 0.01)
    return relaxed


def _train_session(
    session_name: str,
    source_df: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    validate_start: pd.Timestamp,
    validate_end: pd.Timestamp,
    test_start: Optional[pd.Timestamp],
    test_end: Optional[pd.Timestamp],
    model_name: str,
    seed: int,
    fees_pts: float,
    output_dir: Path,
    strategy_name: str,
) -> tuple[Optional[dict], Optional[dict]]:
    session_cfg = (CONFIG.get("SESSIONS", {}) or {}).get(session_name, {}) or {}
    preset = dict(((CONFIG.get("ML_PHYSICS_TRAINING_PRESETS", {}) or {}).get(session_name, {}) or {}))
    timeframe_minutes = int(preset.get("timeframe_minutes", 5) or 5)
    horizon_bars = int(preset.get("horizon_bars", 10) or 10)
    label_mode = str(preset.get("label_mode", "barrier") or "barrier")
    drop_neutral = bool(preset.get("drop_neutral", True))
    session_hours = set(int(h) for h in session_cfg.get("HOURS", []) if h is not None)
    model_file = str(session_cfg.get("MODEL_FILE", f"model_{session_name.lower()}.joblib") or f"model_{session_name.lower()}.joblib")

    resampled = resample_ohlcv(source_df, timeframe_minutes)
    if resampled.empty:
        logging.warning("%s: no rows after resample", session_name)
        return None, None

    features = build_feature_frame(resampled)
    sl_dist, tp_dist = build_sltp_arrays(resampled, strategy_name=strategy_name)
    labels = _build_labels(
        resampled,
        horizon_bars=horizon_bars,
        label_mode=label_mode,
        drop_neutral=drop_neutral,
        sl_dist=sl_dist,
        tp_dist=tp_dist,
    )

    frame = features.copy()
    frame["label"] = labels.reindex(features.index)
    frame["sl_dist"] = pd.Series(sl_dist, index=resampled.index).reindex(features.index)
    frame["tp_dist"] = pd.Series(tp_dist, index=resampled.index).reindex(features.index)
    frame = frame[frame.index.hour.isin(session_hours)]
    frame = frame.dropna(subset=["label"])
    if frame.empty:
        logging.warning("%s: no labeled session rows", session_name)
        return None, None

    leakage_buffer = pd.Timedelta(minutes=timeframe_minutes * horizon_bars)
    train_mask = (frame.index >= train_start) & (frame.index <= (train_end - leakage_buffer))
    val_mask = (frame.index >= validate_start) & (frame.index <= (validate_end - leakage_buffer))
    train = frame.loc[train_mask]
    val = frame.loc[val_mask]
    test = pd.DataFrame()
    if test_start is not None and test_end is not None:
        test_mask = (frame.index >= test_start) & (frame.index <= (test_end - leakage_buffer))
        test = frame.loc[test_mask]

    if train.empty or val.empty:
        logging.warning("%s: insufficient train/validation rows after split", session_name)
        return None, None

    X_train = train[ML_FEATURE_COLUMNS]
    y_train = train["label"].astype(int)
    X_val = val[ML_FEATURE_COLUMNS]
    y_val = val["label"].astype(int)
    base_model = _build_model(model_name, seed)
    base_model.fit(X_train, y_train, sample_weight=_sample_weights(y_train))

    calibrated = _fit_calibrated_model(base_model, X_val, y_val)

    val_result = _optimize_threshold(
        proba=calibrated.predict_proba(X_val)[:, 1],
        labels=y_val.to_numpy(dtype=int),
        sl_dist=val["sl_dist"].to_numpy(dtype=float),
        tp_dist=val["tp_dist"].to_numpy(dtype=float),
        fees_pts=fees_pts,
        preset=preset,
    )
    threshold_selection_mode = "strict"
    if val_result.get("threshold") is None:
        relaxed_preset = _relaxed_legacy_preset(preset)
        val_result = _optimize_threshold(
            proba=calibrated.predict_proba(X_val)[:, 1],
            labels=y_val.to_numpy(dtype=int),
            sl_dist=val["sl_dist"].to_numpy(dtype=float),
            tp_dist=val["tp_dist"].to_numpy(dtype=float),
            fees_pts=fees_pts,
            preset=relaxed_preset,
        )
        threshold_selection_mode = "relaxed_fallback"
    if val_result.get("threshold") is None:
        logging.warning("%s: no threshold candidate passed validation constraints, even after relaxed fallback", session_name)
        return None, None
    if threshold_selection_mode != "strict":
        logging.warning(
            "%s: using relaxed legacy threshold fallback at %.2f (%d trades, avg_pnl=%.4f)",
            session_name,
            float(val_result["threshold"]),
            int(val_result["trades"]),
            float(val_result["avg_pnl"]),
        )

    model_path = output_dir / model_file
    joblib.dump(calibrated, model_path)
    logging.info("%s: saved %s", session_name, model_path)

    threshold_entry = {
        "threshold": float(val_result["threshold"]),
        "avg_pnl": float(val_result["avg_pnl"]),
        "total_pnl": float(val_result["total_pnl"]),
        "trade_count": int(val_result["trades"]),
        "win_rate": float(val_result["win_rate"]),
        "coverage": float(val_result["coverage"]),
        "long_trades": int(val_result["long_trades"]),
        "short_trades": int(val_result["short_trades"]),
        "side_share": float(val_result["side_share"]),
        "timeframe_minutes": int(timeframe_minutes),
        "horizon_bars": int(horizon_bars),
        "label_mode": str(label_mode),
        "drop_neutral": bool(drop_neutral),
        "model_file": model_file,
        "session_hours": sorted(session_hours),
        "strategy_name": strategy_name,
        "selection_mode": threshold_selection_mode,
    }

    metrics_entry = {
        "validation": threshold_entry,
        "saved_model": model_file,
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
    }
    if not test.empty:
        X_test = test[ML_FEATURE_COLUMNS]
        y_test = test["label"].astype(int)
        metrics_entry["test"] = _evaluate(
            calibrated,
            X_test,
            y_test,
            test["sl_dist"].to_numpy(dtype=float),
            test["tp_dist"].to_numpy(dtype=float),
            float(val_result["threshold"]),
            fees_pts,
        )
        metrics_entry["test_rows"] = int(len(test))
    return threshold_entry, metrics_entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the isolated MLPhysics legacy experiment stack.")
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--train-start", default="2011-01-01")
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--validate-start", default="2024-01-01")
    parser.add_argument("--validate-end", default="2024-12-31")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2025-12-31")
    parser.add_argument("--model", choices=("hgb", "rf"), default="hgb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts/ml_physics_legacy_experiment")
    parser.add_argument("--strategy-name", default="Generic")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    source_path = _resolve_source(args.source)
    train_start = _parse_datetime(str(args.train_start), is_end=False)
    train_end = _parse_datetime(str(args.train_end), is_end=True)
    validate_start = _parse_datetime(str(args.validate_start), is_end=False)
    validate_end = _parse_datetime(str(args.validate_end), is_end=True)
    test_start = _parse_datetime(str(args.test_start), is_end=False) if str(args.test_start or "").strip() else None
    test_end = _parse_datetime(str(args.test_end), is_end=True) if str(args.test_end or "").strip() else None

    source_df = _prepare_source_frame(
        source_path,
        start_time=train_start,
        end_time=test_end or validate_end,
        symbol_mode=str(args.symbol_mode or "auto_by_day").strip().lower(),
        symbol_method=str(args.symbol_method or "volume").strip().lower(),
    )

    output_dir = Path(args.out_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    risk_cfg = CONFIG.get("RISK", {}) or {}
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
    fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.5) or 2.5)
    fees_pts = (fees_per_side * 2.0) / point_value if point_value > 0 else 0.0

    thresholds_report = {
        "_meta": {
            "source": str(source_path),
            "symbol_mode": str(args.symbol_mode),
            "symbol_method": str(args.symbol_method),
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "validate_start": validate_start.isoformat(),
            "validate_end": validate_end.isoformat(),
            "test_start": test_start.isoformat() if test_start is not None else "",
            "test_end": test_end.isoformat() if test_end is not None else "",
            "model": str(args.model),
            "strategy_name": str(args.strategy_name),
        }
    }
    metrics_report = {"_meta": dict(thresholds_report["_meta"])}

    for session_name in ("ASIA", "LONDON", "NY_AM", "NY_PM"):
        threshold_entry, metrics_entry = _train_session(
            session_name,
            source_df,
            train_start=train_start,
            train_end=train_end,
            validate_start=validate_start,
            validate_end=validate_end,
            test_start=test_start,
            test_end=test_end,
            model_name=str(args.model),
            seed=int(args.seed),
            fees_pts=fees_pts,
            output_dir=output_dir,
            strategy_name=str(args.strategy_name),
        )
        if threshold_entry is not None:
            thresholds_report[session_name] = threshold_entry
        if metrics_entry is not None:
            metrics_report[session_name] = metrics_entry

    thresholds_path = output_dir / "ml_physics_thresholds.json"
    metrics_path = output_dir / "ml_physics_metrics.json"
    thresholds_path.write_text(json.dumps(thresholds_report, indent=2, ensure_ascii=True), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics_report, indent=2, ensure_ascii=True), encoding="utf-8")
    logging.info("Saved thresholds: %s", thresholds_path)
    logging.info("Saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
