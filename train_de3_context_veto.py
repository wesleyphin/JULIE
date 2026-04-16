import argparse
import datetime as dt
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from config import (
    append_artifact_suffix,
    get_experimental_training_window,
    resolve_artifact_suffix,
)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for train_de3_context_veto.py") from exc

try:
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyarrow is required for train_de3_context_veto.py") from exc


FEATURE_ORDER = ["atr_ratio", "price_location", "vwap_dist_atr", "sl_atr"]
BUCKET_SCHEMA = ["session_bucket", "timeframe", "strategy_type", "vol_regime", "thresh_bucket"]
VOL_REGIMES = {"ultra_low", "low", "normal", "high"}
THRESH_ALL = "ALL"
VOL_ALL = "ALL"
DEFAULT_THRESHOLDS = "0.50,0.55,0.60,0.65,0.70,0.75,0.80"
QUICK_THRESHOLDS = "0.55,0.65,0.75"


def _parse_thresholds(text: str) -> List[float]:
    values: List[float] = []
    for part in str(text or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except Exception as exc:
            raise ValueError(f"Invalid threshold value '{token}' in --thresholds={text}") from exc
    if not values:
        raise ValueError("No threshold values parsed from --thresholds")
    return sorted(set(values))


class _TimedProgressLogger:
    def __init__(self, label: str, total: int, interval_seconds: float = 60.0):
        self.label = str(label)
        self.total = int(max(0, total))
        self.interval_seconds = float(max(1.0, interval_seconds))
        self._last_value = -1
        self._start_ts = time.monotonic()
        self._last_log_ts = 0.0
        logging.info(
            "%s start: total=%d log_interval=%ss",
            self.label,
            self.total,
            int(self.interval_seconds),
        )
        self._flush_handlers()
        self._log_progress(0, force=True)

    @staticmethod
    def _flush_handlers() -> None:
        root = logging.getLogger()
        for handler in root.handlers:
            try:
                handler.flush()
            except Exception:
                continue

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        sec = int(max(0.0, seconds))
        mins, rem = divmod(sec, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours:02d}:{mins:02d}:{rem:02d}"
        return f"{mins:02d}:{rem:02d}"

    def _log_progress(self, current: int, force: bool = False) -> None:
        now = time.monotonic()
        if not force and current < self.total and (now - self._last_log_ts) < self.interval_seconds:
            return
        if self.total <= 0:
            pct = 100.0
        else:
            pct = (100.0 * float(current)) / float(self.total)
        elapsed = max(0.0, now - self._start_ts)
        rate = float(current) / elapsed if elapsed > 0 else 0.0
        eta = (float(self.total - current) / rate) if rate > 0 and current < self.total else 0.0
        logging.info(
            "%s progress: %.1f%% (%d/%d) elapsed=%s eta=%s",
            self.label,
            pct,
            int(current),
            int(self.total),
            self._fmt_duration(elapsed),
            self._fmt_duration(eta),
        )
        self._flush_handlers()
        self._last_log_ts = now

    def update(self, current: int) -> None:
        if self.total <= 0:
            self._log_progress(0, force=True)
            return
        cur = int(min(max(0, current), self.total))
        if cur <= self._last_value and cur < self.total:
            return
        self._last_value = cur
        self._log_progress(cur, force=(cur == 0 or cur >= self.total))


@dataclass
class BucketModel:
    bucket_key: Tuple[str, ...]
    model: CalibratedClassifierCV
    n_samples: int
    n_loss: int
    n_win: int
    calibration: str
    level: str


def _safe_float_array(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float64", copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _norm_text(val, default: str = "unknown") -> str:
    out = str(val if val is not None else "").strip()
    if not out:
        return default
    return out


def _norm_vol_regime(val) -> str:
    regime = _norm_text(val, default="unknown").lower()
    if regime == VOL_ALL.lower():
        return VOL_ALL
    if regime in VOL_REGIMES:
        return regime
    return "unknown"


def _norm_thresh_bucket(val) -> str:
    text = _norm_text(val, default=THRESH_ALL).upper()
    if text == THRESH_ALL:
        return THRESH_ALL
    if text.startswith("T") and len(text) > 1:
        suffix = text[1:]
        try:
            return f"T{int(float(suffix))}"
        except Exception:
            return THRESH_ALL
    try:
        return f"T{int(round(float(text)))}"
    except Exception:
        return THRESH_ALL


def _make_bucket_key(
    session_bucket,
    timeframe,
    strategy_type,
    vol_regime,
    thresh_bucket,
) -> Tuple[str, str, str, str, str]:
    return (
        _norm_text(session_bucket, default="UNKNOWN"),
        _norm_text(timeframe, default="UNKNOWN"),
        _norm_text(strategy_type, default="UNKNOWN"),
        _norm_vol_regime(vol_regime),
        _norm_thresh_bucket(thresh_bucket),
    )


def _dataset_filters(start: Optional[str], end: Optional[str]) -> Optional[ds.Expression]:
    if not start and not end:
        return None
    start_ts = pd.to_datetime(start) if start else None
    end_ts = None
    if end:
        end_raw = str(end).strip()
        end_ts = pd.to_datetime(end_raw)
        has_time = ("T" in end_raw) or (":" in end_raw)
        if not has_time:
            end_ts = end_ts + pd.Timedelta(days=1)
    filters = None
    if start_ts is not None:
        filters = ds.field("entry_time") >= start_ts
    if end_ts is not None:
        end_filter = ds.field("entry_time") < end_ts
        filters = end_filter if filters is None else (filters & end_filter)
    return filters


def load_dataset(path: Path, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    available = set(dataset.schema.names)
    base_columns = [
        "entry_time",
        "trade_date",
        "session_bucket",
        "timeframe",
        "strategy_type",
        "atr_5m",
        "atr_5m_median",
        "price_location",
        "vwap_dist_atr",
        "sl_points",
        "outcome",
    ]
    optional = ["vol_regime", "thresh_bucket"]
    columns = base_columns + [c for c in optional if c in available]

    filt = _dataset_filters(start, end)
    try:
        table = dataset.to_table(columns=columns, filter=filt)
        df = table.to_pandas()
    except Exception as exc:
        if filt is None:
            raise
        logging.warning(
            "Dataset filter pushdown failed (%s). Falling back to pandas-side date filter.",
            exc,
        )
        table = dataset.to_table(columns=columns)
        df = table.to_pandas()
    if df.empty:
        raise ValueError("Context dataset is empty for the given window.")
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    entry_tz = getattr(df["entry_time"].dt, "tz", None)

    def _coerce_bound(text: Optional[str], *, is_end: bool) -> Optional[pd.Timestamp]:
        if not text:
            return None
        raw = str(text).strip()
        if not raw:
            return None
        has_time = ("T" in raw) or (":" in raw)
        ts = pd.to_datetime(raw, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Invalid date bound: {text}")
        if is_end and not has_time:
            ts = ts + pd.Timedelta(days=1)
        if entry_tz is not None:
            if ts.tzinfo is None:
                ts = ts.tz_localize(entry_tz)
            else:
                ts = ts.tz_convert(entry_tz)
        else:
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
        return ts

    start_ts = _coerce_bound(start, is_end=False)
    end_ts = _coerce_bound(end, is_end=True)
    if start_ts is not None:
        df = df[df["entry_time"] >= start_ts]
    if end_ts is not None:
        df = df[df["entry_time"] < end_ts]
    if df.empty:
        raise ValueError("Context dataset is empty after applying pandas date filter.")

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    if "vol_regime" not in df.columns:
        df["vol_regime"] = "unknown"
    if "thresh_bucket" not in df.columns:
        df["thresh_bucket"] = THRESH_ALL
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    atr = df["atr_5m"].replace(0, np.nan)
    atr_med = df["atr_5m_median"].replace(0, np.nan)
    df["atr_ratio"] = atr / atr_med
    df["sl_atr"] = df["sl_points"] / atr
    df["loss"] = (df["outcome"].astype(int) == 0).astype(int)
    df["vol_regime"] = df["vol_regime"].map(_norm_vol_regime)
    df["thresh_bucket"] = df["thresh_bucket"].map(_norm_thresh_bucket)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_ORDER + ["loss"])
    return df


def _train_bucket_model(
    X: np.ndarray,
    y: np.ndarray,
    calibration: str,
    min_splits: int = 3,
    max_iter: int = 500,
) -> CalibratedClassifierCV:
    min_splits = int(max(2, min_splits))
    base = LogisticRegression(
        solver="lbfgs",
        max_iter=int(max(50, max_iter)),
        class_weight="balanced",
        penalty="l2",
    )
    tscv = TimeSeriesSplit(n_splits=min_splits)
    try:
        model = CalibratedClassifierCV(estimator=base, method=calibration, cv=tscv)
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, method=calibration, cv=tscv)
    model.fit(X, y)
    return model


def _fit_group_model(
    key: Tuple[str, ...],
    group: pd.DataFrame,
    level: str,
    min_samples: int,
    iso_min_samples: int,
    iso_min_class: int,
    min_splits: int,
    max_iter: int,
) -> Optional[BucketModel]:
    if len(group) < min_samples:
        return None
    group = group.sort_values("entry_time")
    y = group["loss"].to_numpy()
    if len(np.unique(y)) < 2:
        return None
    n_loss = int(y.sum())
    n_win = int(len(y) - n_loss)
    if min(n_loss, n_win) < 5:
        return None
    calibration = "isotonic"
    if len(group) < iso_min_samples or min(n_loss, n_win) < iso_min_class:
        calibration = "sigmoid"
    X = _safe_float_array(group[FEATURE_ORDER].to_numpy())
    try:
        model = _train_bucket_model(X, y, calibration=calibration, min_splits=min_splits, max_iter=max_iter)
    except Exception as exc:
        logging.warning("Bucket %s (%s) training failed: %s", key, level, exc)
        return None
    return BucketModel(
        bucket_key=key,
        model=model,
        n_samples=len(group),
        n_loss=n_loss,
        n_win=n_win,
        calibration=calibration,
        level=level,
    )


def train_bucket_models(
    df: pd.DataFrame,
    min_samples: int,
    iso_min_samples: int,
    iso_min_class: int,
    min_splits: int,
    max_iter: int,
) -> Dict[Tuple[str, ...], BucketModel]:
    models: Dict[Tuple[str, ...], BucketModel] = {}
    levels = [
        {
            "name": "full",
            "group_cols": ["session_bucket", "timeframe", "strategy_type", "vol_regime", "thresh_bucket"],
            "key_fn": lambda vals: _make_bucket_key(vals[0], vals[1], vals[2], vals[3], vals[4]),
            "min_samples": int(min_samples),
        },
        {
            "name": "no_thresh",
            "group_cols": ["session_bucket", "timeframe", "strategy_type", "vol_regime"],
            "key_fn": lambda vals: _make_bucket_key(vals[0], vals[1], vals[2], vals[3], THRESH_ALL),
            "min_samples": int(max(min_samples, round(min_samples * 1.5))),
        },
        {
            "name": "no_regime",
            "group_cols": ["session_bucket", "timeframe", "strategy_type", "thresh_bucket"],
            "key_fn": lambda vals: _make_bucket_key(vals[0], vals[1], vals[2], VOL_ALL, vals[3]),
            "min_samples": int(max(min_samples, round(min_samples * 1.5))),
        },
        {
            "name": "coarse",
            "group_cols": ["session_bucket", "timeframe", "strategy_type"],
            "key_fn": lambda vals: _make_bucket_key(vals[0], vals[1], vals[2], VOL_ALL, THRESH_ALL),
            "min_samples": int(max(min_samples, round(min_samples * 2.0))),
        },
    ]
    grouped_levels = []
    total_groups = 0
    for level in levels:
        grouped = df.groupby(level["group_cols"], sort=False)
        total_groups += int(grouped.ngroups)
        grouped_levels.append((level, grouped))
    progress = _TimedProgressLogger("Bucket model training", total_groups)
    processed = 0
    for level, grouped in grouped_levels:
        for bucket, group in grouped:
            processed += 1
            vals = bucket if isinstance(bucket, tuple) else (bucket,)
            key = level["key_fn"](vals)
            if key in models:
                progress.update(processed)
                continue
            fit = _fit_group_model(
                key=key,
                group=group,
                level=level["name"],
                min_samples=level["min_samples"],
                iso_min_samples=iso_min_samples,
                iso_min_class=iso_min_class,
                min_splits=min_splits,
                max_iter=max_iter,
            )
            if fit is not None:
                models[key] = fit
            progress.update(processed)
    return models


def _predict_loss_proba(
    df: pd.DataFrame,
    models: Dict[Tuple[str, ...], BucketModel],
) -> np.ndarray:
    if df.empty:
        return np.full(0, np.nan)
    local_df = df.reset_index(drop=True)
    probs = np.full(len(local_df), np.nan)
    unresolved = np.ones(len(local_df), dtype=bool)
    level_defs = [
        {
            "group_cols": ["session_bucket", "timeframe", "strategy_type", "vol_regime", "thresh_bucket"],
            "key_fn": lambda vals: _make_bucket_key(vals[0], vals[1], vals[2], vals[3], vals[4]),
        },
        {
            "group_cols": ["session_bucket", "timeframe", "strategy_type", "vol_regime"],
            "key_fn": lambda vals: _make_bucket_key(vals[0], vals[1], vals[2], vals[3], THRESH_ALL),
        },
        {
            "group_cols": ["session_bucket", "timeframe", "strategy_type", "thresh_bucket"],
            "key_fn": lambda vals: _make_bucket_key(vals[0], vals[1], vals[2], VOL_ALL, vals[3]),
        },
        {
            "group_cols": ["session_bucket", "timeframe", "strategy_type"],
            "key_fn": lambda vals: _make_bucket_key(vals[0], vals[1], vals[2], VOL_ALL, THRESH_ALL),
        },
    ]
    for level in level_defs:
        if not np.any(unresolved):
            break
        sub = local_df.loc[unresolved]
        if sub.empty:
            break
        grouped = sub.groupby(level["group_cols"], sort=False).groups
        for bucket_vals, idx_labels in grouped.items():
            vals = bucket_vals if isinstance(bucket_vals, tuple) else (bucket_vals,)
            key = level["key_fn"](vals)
            model = models.get(key)
            if model is None and len(key) >= 3:
                model = models.get(key[:3])
            if model is None:
                continue
            idx_arr = np.asarray(list(idx_labels), dtype=int)
            X = _safe_float_array(local_df.loc[idx_arr, FEATURE_ORDER].to_numpy())
            try:
                probs[idx_arr] = model.model.predict_proba(X)[:, 1]
                unresolved[idx_arr] = False
            except Exception as exc:
                logging.warning("Predict failed for bucket %s: %s", model.bucket_key, exc)
    return probs


def _evaluate_threshold(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> dict:
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "removed_pct": 0.0,
            "baseline_loss_rate": 0.0,
            "veto_loss_rate": 0.0,
            "kept_winrate": 0.0,
            "avg_outcome": 0.0,
        }
    loss = df["loss"].to_numpy()
    veto_mask = (probs > threshold) & np.isfinite(probs)
    kept_mask = ~veto_mask
    kept_total = int(kept_mask.sum())
    removed_pct = float(veto_mask.mean())
    baseline_loss = float(loss.mean())
    if kept_total > 0:
        kept_loss = float(loss[kept_mask].mean())
        kept_winrate = float(1.0 - kept_loss)
        avg_outcome = float(1.0 - kept_loss)
    else:
        kept_loss = 0.0
        kept_winrate = 0.0
        avg_outcome = 0.0
    return {
        "total": total,
        "removed_pct": removed_pct,
        "baseline_loss_rate": baseline_loss,
        "veto_loss_rate": kept_loss,
        "kept_winrate": kept_winrate,
        "avg_outcome": avg_outcome,
    }


def walk_forward_splits(
    df: pd.DataFrame, train_months: int, test_months: int, step_months: int
) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    start = df["entry_time"].min()
    end = df["entry_time"].max()
    start_month = pd.Timestamp(year=start.year, month=start.month, day=1, tz=start.tzinfo)
    cursor = start_month
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if train_end >= end:
            break
        if train_end >= test_end:
            break
        yield train_start, train_end, train_end, test_end
        cursor = cursor + pd.DateOffset(months=step_months)


def evaluate_walk_forward(
    df: pd.DataFrame,
    train_months: int,
    test_months: int,
    step_months: int,
    min_samples: int,
    iso_min_samples: int,
    iso_min_class: int,
    thresholds: Sequence[float],
    min_splits: int,
    max_iter: int,
) -> dict:
    split_results = []
    threshold_scores = {t: {"total": 0, "kept_wins": 0, "kept_trades": 0, "removed": 0} for t in thresholds}
    split_windows = list(walk_forward_splits(df, train_months, test_months, step_months))
    split_progress = _TimedProgressLogger("Walk-forward evaluation", len(split_windows))

    for split_idx, (train_start, train_end, test_start, test_end) in enumerate(split_windows, start=1):
        train_df = df[(df["entry_time"] >= train_start) & (df["entry_time"] < train_end)]
        test_df = df[(df["entry_time"] >= test_start) & (df["entry_time"] < test_end)]
        if train_df.empty or test_df.empty:
            split_progress.update(split_idx)
            continue
        models = train_bucket_models(
            train_df,
            min_samples=min_samples,
            iso_min_samples=iso_min_samples,
            iso_min_class=iso_min_class,
            min_splits=min_splits,
            max_iter=max_iter,
        )
        probs = _predict_loss_proba(test_df, models)

        split_payload = {
            "split": split_idx,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "thresholds": {},
        }

        for threshold in thresholds:
            metrics = _evaluate_threshold(test_df, probs, float(threshold))
            split_payload["thresholds"][str(threshold)] = metrics

            removed = int(metrics["removed_pct"] * metrics["total"])
            kept = metrics["total"] - removed
            kept_wins = int(metrics["kept_winrate"] * kept)
            threshold_scores[float(threshold)]["total"] += metrics["total"]
            threshold_scores[float(threshold)]["removed"] += removed
            threshold_scores[float(threshold)]["kept_trades"] += kept
            threshold_scores[float(threshold)]["kept_wins"] += kept_wins

        split_results.append(split_payload)
        split_progress.update(split_idx)

    chosen = None
    best_score = -1.0
    fallback_threshold = None
    fallback_removed = None
    for threshold in thresholds:
        totals = threshold_scores[float(threshold)]
        total = totals["total"]
        if total == 0:
            continue
        removed_pct = totals["removed"] / float(total)
        if fallback_removed is None or removed_pct < fallback_removed:
            fallback_removed = removed_pct
            fallback_threshold = float(threshold)
        if removed_pct > 0.40:
            continue
        kept_trades = totals["kept_trades"]
        kept_winrate = totals["kept_wins"] / float(kept_trades) if kept_trades else 0.0
        if kept_winrate > best_score:
            best_score = kept_winrate
            chosen = float(threshold)
    if chosen is None:
        chosen = fallback_threshold if fallback_threshold is not None else float(thresholds[0])
        if fallback_removed is not None:
            logging.warning(
                "No threshold met removal cap (<=40%%). Using least-removal threshold %.2f (removed=%.2f%%).",
                float(chosen),
                float(fallback_removed) * 100.0,
            )

    return {
        "splits": split_results,
        "threshold_scores": threshold_scores,
        "chosen_threshold": chosen,
    }


def evaluate_lodo(
    df: pd.DataFrame,
    start: str,
    end: str,
    threshold: float,
    min_samples: int,
    iso_min_samples: int,
    iso_min_class: int,
    min_splits: int,
    max_iter: int,
    day_step: int = 1,
    max_days: int = 0,
) -> dict:
    start_ts = pd.to_datetime(start).date()
    end_ts = pd.to_datetime(end).date()
    window_df = df[(df["trade_date"] >= start_ts) & (df["trade_date"] < end_ts)]
    if window_df.empty:
        return {}

    bucket_data = {}
    grouped_buckets = window_df.groupby(BUCKET_SCHEMA, sort=False)
    bucket_progress = _TimedProgressLogger("LODO bucket cache", int(grouped_buckets.ngroups))
    for bucket_idx, (bucket, group) in enumerate(grouped_buckets, start=1):
        key = _make_bucket_key(bucket[0], bucket[1], bucket[2], bucket[3], bucket[4])
        group = group.sort_values("entry_time")
        bucket_data[key] = {
            "X": _safe_float_array(group[FEATURE_ORDER].to_numpy()),
            "y": group["loss"].to_numpy(),
            "dates": np.array(group["trade_date"].to_list()),
        }
        bucket_progress.update(bucket_idx)

    all_days = sorted(window_df["trade_date"].unique())
    day_step = int(max(1, day_step))
    days = all_days[::day_step]
    max_days = int(max(0, max_days))
    if max_days and len(days) > max_days:
        idx = np.linspace(0, len(days) - 1, num=max_days, dtype=int)
        unique_idx = []
        seen = set()
        for i in idx.tolist():
            if i in seen:
                continue
            seen.add(i)
            unique_idx.append(i)
        days = [days[i] for i in unique_idx]

    daily_stats = []
    day_progress = _TimedProgressLogger("LODO daily evaluation", len(days))
    for day_idx, day in enumerate(days, start=1):
        day_df = window_df[window_df["trade_date"] == day]
        if day_df.empty:
            day_progress.update(day_idx)
            continue
        day_buckets = day_df.groupby(BUCKET_SCHEMA).groups
        models = {}
        for bucket in day_buckets:
            key = _make_bucket_key(bucket[0], bucket[1], bucket[2], bucket[3], bucket[4])
            data = bucket_data.get(key)
            if data is None:
                continue
            mask = data["dates"] != day
            if int(mask.sum()) < int(min_samples):
                continue
            y_train = data["y"][mask]
            if len(np.unique(y_train)) < 2:
                continue
            n_loss = int(y_train.sum())
            n_win = int(len(y_train) - n_loss)
            if min(n_loss, n_win) < 5:
                continue
            calibration = "isotonic"
            if int(mask.sum()) < int(iso_min_samples) or min(n_loss, n_win) < int(iso_min_class):
                calibration = "sigmoid"
            try:
                model = _train_bucket_model(
                    data["X"][mask],
                    y_train,
                    calibration=calibration,
                    min_splits=min_splits,
                    max_iter=max_iter,
                )
            except Exception:
                continue
            models[key] = BucketModel(
                bucket_key=key,
                model=model,
                n_samples=int(mask.sum()),
                n_loss=n_loss,
                n_win=n_win,
                calibration=calibration,
                level="full",
            )

        probs = _predict_loss_proba(day_df, models)
        metrics = _evaluate_threshold(day_df, probs, float(threshold))
        daily_stats.append(metrics)
        day_progress.update(day_idx)

    if not daily_stats:
        return {}

    avg_removed = float(np.mean([d["removed_pct"] for d in daily_stats]))
    avg_winrate = float(np.mean([d["kept_winrate"] for d in daily_stats]))
    avg_outcome = float(np.mean([d["avg_outcome"] for d in daily_stats]))
    baseline_loss = float(np.mean([d["baseline_loss_rate"] for d in daily_stats]))
    veto_loss = float(np.mean([d["veto_loss_rate"] for d in daily_stats]))
    return {
        "window_start": str(start_ts),
        "window_end": str(end_ts),
        "available_days": int(len(all_days)),
        "evaluated_days": int(len(days)),
        "day_step": int(day_step),
        "max_days": int(max_days),
        "mean_daily_removed_pct": avg_removed,
        "mean_daily_kept_winrate": avg_winrate,
        "mean_daily_avg_outcome": avg_outcome,
        "mean_daily_baseline_loss": baseline_loss,
        "mean_daily_veto_loss": veto_loss,
    }


def _serialize_bucket_model(model: BucketModel) -> dict:
    payload = {
        "n_samples": model.n_samples,
        "n_loss": model.n_loss,
        "n_win": model.n_win,
        "calibration": model.calibration,
        "level": model.level,
        "models": [],
    }
    for calibrated in model.model.calibrated_classifiers_:
        estimator = getattr(calibrated, "estimator", None) or getattr(calibrated, "base_estimator", None)
        calibrators = getattr(calibrated, "calibrators", None)
        if calibrators is None:
            calibrators = getattr(calibrated, "calibrators_", None)
        calibrator = calibrators[0] if isinstance(calibrators, (list, tuple)) and calibrators else calibrators

        calib_payload = {"method": model.calibration}
        if calibrator is not None:
            if hasattr(calibrator, "X_thresholds_") and hasattr(calibrator, "y_thresholds_"):
                calib_payload["x_thresholds"] = calibrator.X_thresholds_.tolist()
                calib_payload["y_thresholds"] = calibrator.y_thresholds_.tolist()
            elif hasattr(calibrator, "a_") and hasattr(calibrator, "b_"):
                calib_payload["a"] = float(calibrator.a_)
                calib_payload["b"] = float(calibrator.b_)

        if estimator is not None:
            coef = estimator.coef_.ravel().tolist()
            intercept = float(estimator.intercept_[0])
        else:
            coef = []
            intercept = 0.0

        payload["models"].append(
            {
                "coef": coef,
                "intercept": intercept,
                "calibration": calib_payload,
            }
        )
    return payload


def write_outputs(
    models: Dict[Tuple[str, ...], BucketModel],
    report: dict,
    threshold: float,
    out_dir: Path,
    artifact_suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "feature_order": FEATURE_ORDER,
        "bucket_schema": BUCKET_SCHEMA,
        "threshold": threshold,
        "bucket_models": {},
    }
    for bucket, model in models.items():
        bucket_key = "|".join(bucket)
        model_payload["bucket_models"][bucket_key] = _serialize_bucket_model(model)

    model_name = (
        append_artifact_suffix("de3_context_veto_models.json", artifact_suffix)
        if artifact_suffix
        else "de3_context_veto_models.json"
    )
    report_name = (
        append_artifact_suffix("de3_context_veto_report.json", artifact_suffix)
        if artifact_suffix
        else "de3_context_veto_report.json"
    )

    (out_dir / model_name).write_text(
        json.dumps(model_payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    (out_dir / report_name).write_text(
        json.dumps(report, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DE3 context loss veto models.")
    parser.add_argument("--dataset", default="cache/de3_context_dataset", help="Dataset root directory.")
    parser.add_argument("--out-dir", default=".", help="Output directory for JSON artifacts.")
    parser.add_argument("--start", "--train-start", dest="start", default=None, help="Train start date (YYYY-MM-DD).")
    parser.add_argument("--end", "--train-end", dest="end", default=None, help="Train end date (YYYY-MM-DD).")
    parser.add_argument(
        "--experimental-window",
        action="store_true",
        help="Train only on configured experimental window (2011-01-01 .. 2017-12-31).",
    )
    parser.add_argument(
        "--artifact-suffix",
        default=None,
        help="Suffix appended to output artifacts (e.g. _exp2011_2017).",
    )
    parser.add_argument("--quick", action="store_true", help="Faster training preset with lighter validation.")
    parser.add_argument("--train-months", type=int, default=24)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS, help="Comma-separated veto threshold grid.")
    parser.add_argument("--lodo-start", default="2024-01-01")
    parser.add_argument("--lodo-end", default="2026-01-01")
    parser.add_argument("--skip-lodo", action="store_true", help="Skip leave-one-day-out diagnostics.")
    parser.add_argument("--lodo-day-step", type=int, default=1, help="Evaluate every Nth day in LODO.")
    parser.add_argument(
        "--lodo-max-days",
        type=int,
        default=0,
        help="Cap LODO evaluated days (0 = all sampled days).",
    )
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--iso-min-samples", type=int, default=800)
    parser.add_argument("--iso-min-class", type=int, default=50)
    parser.add_argument("--min-splits", type=int, default=3, help="TimeSeriesSplit folds for calibration CV.")
    parser.add_argument("--max-iter", type=int, default=500, help="LogisticRegression max_iter.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )

    exp_enabled = bool(args.experimental_window)
    train_start = args.start
    train_end = args.end
    lodo_start = args.lodo_start
    lodo_end = args.lodo_end
    if exp_enabled:
        exp_start, exp_end = get_experimental_training_window()
        train_start = exp_start
        train_end = exp_end
        lodo_start = exp_start or lodo_start
        lodo_end = exp_end or lodo_end
        logging.info("Experimental window enabled: %s -> %s", train_start, train_end)
    if not train_start or not train_end:
        logging.warning(
            "Training window is not fully bounded (start=%s end=%s). "
            "For strict OOS workflows, pass both --train-start and --train-end.",
            train_start,
            train_end,
        )
    artifact_suffix = resolve_artifact_suffix(args.artifact_suffix, exp_enabled)

    df = load_dataset(Path(args.dataset), start=train_start, end=train_end)
    df = build_features(df)

    if args.quick:
        if args.thresholds == DEFAULT_THRESHOLDS:
            args.thresholds = QUICK_THRESHOLDS
        if args.min_splits == 3:
            args.min_splits = 2
        if args.max_iter == 500:
            args.max_iter = 300
        if args.lodo_day_step == 1:
            args.lodo_day_step = 3
        if args.lodo_max_days == 0:
            args.lodo_max_days = 90

    thresholds = _parse_thresholds(args.thresholds)
    logging.info(
        "Training config: thresholds=%s min_splits=%d max_iter=%d skip_lodo=%s lodo_day_step=%d lodo_max_days=%d",
        ",".join(f"{v:.2f}" for v in thresholds),
        int(args.min_splits),
        int(args.max_iter),
        bool(args.skip_lodo),
        int(max(1, args.lodo_day_step)),
        int(max(0, args.lodo_max_days)),
    )
    wf_report = evaluate_walk_forward(
        df,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        min_samples=args.min_samples,
        iso_min_samples=args.iso_min_samples,
        iso_min_class=args.iso_min_class,
        thresholds=thresholds,
        min_splits=args.min_splits,
        max_iter=args.max_iter,
    )
    chosen_threshold = wf_report["chosen_threshold"]

    models = train_bucket_models(
        df,
        min_samples=args.min_samples,
        iso_min_samples=args.iso_min_samples,
        iso_min_class=args.iso_min_class,
        min_splits=args.min_splits,
        max_iter=args.max_iter,
    )

    if args.skip_lodo:
        lodo_report = {"skipped": True}
    else:
        lodo_report = evaluate_lodo(
            df,
            start=lodo_start,
            end=lodo_end,
            threshold=chosen_threshold,
            min_samples=args.min_samples,
            iso_min_samples=args.iso_min_samples,
            iso_min_class=args.iso_min_class,
            min_splits=args.min_splits,
            max_iter=args.max_iter,
            day_step=args.lodo_day_step,
            max_days=args.lodo_max_days,
        )

    level_counts: Dict[str, int] = {}
    for model in models.values():
        level_counts[model.level] = int(level_counts.get(model.level, 0)) + 1

    report = {
        "train_window": {
            "start": str(train_start) if train_start else None,
            "end": str(train_end) if train_end else None,
        },
        "lodo_window": {
            "start": str(lodo_start) if lodo_start else None,
            "end": str(lodo_end) if lodo_end else None,
        },
        "walk_forward": wf_report,
        "lodo": lodo_report,
        "bucket_coverage": len(models),
        "bucket_coverage_by_level": level_counts,
        "bucket_schema": BUCKET_SCHEMA,
        "threshold": chosen_threshold,
    }

    write_outputs(
        models,
        report,
        chosen_threshold,
        Path(args.out_dir),
        artifact_suffix=artifact_suffix,
    )
    model_name = (
        append_artifact_suffix("de3_context_veto_models.json", artifact_suffix)
        if artifact_suffix
        else "de3_context_veto_models.json"
    )
    report_name = (
        append_artifact_suffix("de3_context_veto_report.json", artifact_suffix)
        if artifact_suffix
        else "de3_context_veto_report.json"
    )
    logging.info("Wrote %s and %s", model_name, report_name)


if __name__ == "__main__":
    main()
