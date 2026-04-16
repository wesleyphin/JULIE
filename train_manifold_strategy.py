import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from config import (
    CONFIG,
    append_artifact_suffix,
    get_experimental_training_window,
    resolve_artifact_suffix,
)
from manifold_strategy_features import FEATURE_COLUMNS, build_training_feature_frame
from manifold_confluence import apply_confluence_formula, calibrate_confluence

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HistGradientBoostingClassifier = None

try:
    # sklearn>=1.6 preferred path for prefit calibration
    from sklearn.frozen import FrozenEstimator
except Exception:
    FrozenEstimator = None


DEFAULT_OUTRIGHT_SYMBOL_REGEX = r"^ES[HMUZ]\d{1,2}$"


def _fit_prefit_calibrator(model, X_val, y_val) -> CalibratedClassifierCV:
    """
    Fit a sigmoid calibrator on a prefit model with sklearn-version compatibility.

    sklearn<1.8 supports cv='prefit'; sklearn>=1.8 expects FrozenEstimator + cv=None.
    """
    if FrozenEstimator is not None:
        frozen = FrozenEstimator(model)
        try:
            calibrated = CalibratedClassifierCV(estimator=frozen, method="sigmoid", cv=None)
        except TypeError:
            calibrated = CalibratedClassifierCV(frozen, method="sigmoid", cv=None)
        calibrated.fit(X_val, y_val)
        return calibrated

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated


def _auto_select_symbol_by_day(df: pd.DataFrame, method: str = "volume") -> tuple[pd.DataFrame, dict]:
    if df.empty or "symbol" not in df.columns:
        return df, {}

    work = df.copy()
    work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
    score_col = "volume" if (method == "volume" and "volume" in work.columns) else "_rows"
    if score_col == "_rows":
        work[score_col] = 1.0

    date_key = work.index.date
    stats = (
        work.groupby([date_key, work["symbol"]], sort=False)[score_col]
        .sum(min_count=1)
        .fillna(0.0)
        .rename("score")
        .reset_index()
    )
    stats.columns = ["date", "symbol", "score"]
    stats = stats.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
    best = stats.drop_duplicates("date")
    day_to_symbol = dict(zip(best["date"], best["symbol"]))

    chosen = pd.Series(date_key, index=work.index).map(day_to_symbol)
    mask = work["symbol"].astype(str) == chosen.astype(str)
    filtered = work.loc[mask].copy()
    if score_col == "_rows":
        filtered.drop(columns=["_rows"], inplace=True, errors="ignore")
    return filtered, day_to_symbol


def _normalize_loaded_bars(
    df: pd.DataFrame,
    source_name: str,
    *,
    allow_spreads: bool,
    symbol_regex: Optional[str],
    min_valid_price: float,
) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{source_name} is empty.")

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]
    lc_map = {c.lower(): c for c in work.columns}

    time_col = None
    for key in ("datetime", "timestamp", "ts_event", "time", "date"):
        if key in lc_map:
            time_col = lc_map[key]
            break
    if time_col is None:
        if isinstance(work.index, pd.DatetimeIndex):
            work = work.sort_index()
        else:
            raise ValueError(f"{source_name} missing datetime/timestamp column.")
    else:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col]).set_index(time_col).sort_index()

    rename_map = {}
    for target in ("open", "high", "low", "close", "volume", "symbol"):
        for col in work.columns:
            if str(col).lower() == target:
                rename_map[col] = target
                break
    work = work.rename(columns=rename_map)

    for col in ("open", "high", "low", "close"):
        if col not in work.columns:
            raise ValueError(f"{source_name} missing required column: {col}")
        work[col] = pd.to_numeric(work[col], errors="coerce")
    if "volume" not in work.columns:
        work["volume"] = 0.0
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)

    if "symbol" in work.columns:
        work["symbol"] = work["symbol"].astype(str).str.strip().str.upper()
        work = work.loc[work["symbol"] != ""]

        if not allow_spreads:
            spread_mask = work["symbol"].str.contains("-", regex=False)
            spread_rows = int(spread_mask.sum())
            if spread_rows > 0:
                logging.info("Dropped spread-symbol rows: %d", spread_rows)
                work = work.loc[~spread_mask]

        if symbol_regex:
            try:
                pattern = re.compile(str(symbol_regex))
            except re.error as exc:
                raise ValueError(f"Invalid --symbol-regex pattern: {symbol_regex}") from exc
            symbol_ok = work["symbol"].str.match(pattern, na=False)
            dropped = int((~symbol_ok).sum())
            if dropped > 0:
                logging.info("Dropped rows failing symbol regex '%s': %d", symbol_regex, dropped)
            work = work.loc[symbol_ok]

        try:
            n_symbols = int(work["symbol"].nunique(dropna=True))
        except Exception:
            n_symbols = 0
        if n_symbols > 1:
            logging.info(
                "Detected %s symbols; selecting dominant symbol by day.",
                n_symbols,
            )
            work, selected = _auto_select_symbol_by_day(work, method="volume")
            logging.info("Symbol selection complete (days=%s).", len(selected))

    work = work.replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["open", "high", "low", "close"])

    ohlc_cols = ["open", "high", "low", "close"]
    min_price = float(min_valid_price)
    if min_price > 0.0:
        positive_mask = (work[ohlc_cols] >= min_price).all(axis=1)
        dropped = int((~positive_mask).sum())
        if dropped > 0:
            logging.info("Dropped bars below min_valid_price %.3f: %d", min_price, dropped)
        work = work.loc[positive_mask]

    high_ok = work["high"] >= work[["open", "close", "low"]].max(axis=1)
    low_ok = work["low"] <= work[["open", "close", "high"]].min(axis=1)
    range_ok = work["high"] >= work["low"]
    ohlc_ok = high_ok & low_ok & range_ok
    dropped_ohlc = int((~ohlc_ok).sum())
    if dropped_ohlc > 0:
        logging.info("Dropped OHLC-inconsistent bars: %d", dropped_ohlc)
        work = work.loc[ohlc_ok]

    if work.index.has_duplicates:
        work = work[~work.index.duplicated(keep="last")]
    return work[["open", "high", "low", "close", "volume"]]


def _load_bars(
    path: Path,
    *,
    allow_spreads: bool,
    symbol_regex: Optional[str],
    min_valid_price: float,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        logging.info("Loading parquet input: %s", path)
        df = pd.read_parquet(path)
        return _normalize_loaded_bars(
            df,
            source_name="Parquet input",
            allow_spreads=allow_spreads,
            symbol_regex=symbol_regex,
            min_valid_price=min_valid_price,
        )

    logging.info("Loading CSV input: %s", path)
    df = pd.read_csv(path)
    return _normalize_loaded_bars(
        df,
        source_name="CSV input",
        allow_spreads=allow_spreads,
        symbol_regex=symbol_regex,
        min_valid_price=min_valid_price,
    )


def _parse_date(value: Optional[str], *, is_end: bool = False) -> Optional[pd.Timestamp]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    has_time = ("T" in raw) or (":" in raw)
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date: {value}")
    if ts.tzinfo is None:
        ts = ts.tz_localize("US/Eastern")
    else:
        ts = ts.tz_convert("US/Eastern")
    if is_end and not has_time:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _filter_range(
    bars: pd.DataFrame,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if bars.empty:
        return bars
    out = bars
    if start is not None:
        out = out.loc[out.index >= start]
    if end is not None:
        out = out.loc[out.index <= end]
    return out


def _build_labels(close: pd.Series, horizon_bars: int, min_move_points: float) -> Tuple[pd.Series, pd.Series]:
    fwd_points = close.shift(-int(horizon_bars)) - close
    label = pd.Series(np.nan, index=close.index, dtype="float64")
    label.loc[fwd_points >= float(min_move_points)] = 1.0
    label.loc[fwd_points <= -float(min_move_points)] = 0.0
    return label, fwd_points


def _coerce_session_ids(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(-999.0).to_numpy(dtype=float)
    return np.rint(arr).astype(np.int16)


def _parse_allowed_sessions(raw: Optional[str]) -> Optional[set[int]]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    parts = [p.strip() for p in txt.split(",")]
    out: set[int] = set()
    for p in parts:
        if not p:
            continue
        try:
            out.add(int(float(p)))
        except Exception:
            raise ValueError(f"Invalid session id in --allowed-session-ids: {p}")
    return out if out else None


def _session_name_from_id(session_id: int) -> str:
    mapping = {
        -1: "OFF",
        0: "ASIA",
        1: "LONDON",
        2: "NY_AM",
        3: "NY_PM",
    }
    return mapping.get(int(session_id), f"SESSION_{int(session_id)}")


def _build_model(model_name: str, seed: int, workers: int = 1):
    if model_name == "hgb" and HistGradientBoostingClassifier is not None:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=5,
            max_iter=300,
            min_samples_leaf=64,
            random_state=int(seed),
        )
    if model_name == "logreg":
        return LogisticRegression(
            C=0.5,
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            random_state=int(seed),
        )
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=int(seed),
        n_jobs=max(1, int(workers)),
    )


def _load_cached_training_data(features_path: Path) -> pd.DataFrame:
    data = pd.read_parquet(features_path)
    required = set(FEATURE_COLUMNS) | {"label", "future_points"}
    missing = sorted(col for col in required if col not in data.columns)
    if missing:
        raise RuntimeError(
            f"Cached feature parquet is missing required columns: {', '.join(missing)}"
        )
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "future_points"])
    data["label"] = data["label"].astype(int)
    return data


def _sample_weights(y: pd.Series) -> np.ndarray:
    yv = y.astype(int).to_numpy()
    n = len(yv)
    if n == 0:
        return np.array([], dtype=float)
    pos = float(np.sum(yv == 1))
    neg = float(np.sum(yv == 0))
    if pos <= 0 or neg <= 0:
        return np.ones(n, dtype=float)
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(yv == 1, w_pos, w_neg).astype(float)


def _evaluate_threshold(
    prob_up: np.ndarray,
    future_points: np.ndarray,
    long_thr: float,
    short_thr: float,
    fees_points: float,
    session_ids: Optional[np.ndarray] = None,
    allowed_session_ids: Optional[set[int]] = None,
) -> Dict:
    side = np.where(prob_up >= long_thr, 1, np.where(prob_up <= short_thr, -1, 0))
    trade_mask = side != 0
    if session_ids is not None and allowed_session_ids:
        sess = np.asarray(session_ids, dtype=np.int16).reshape(-1)
        if len(sess) != len(side):
            raise ValueError("session_ids length mismatch in _evaluate_threshold")
        allow = np.isin(sess, np.asarray(sorted(allowed_session_ids), dtype=np.int16))
        trade_mask = trade_mask & allow
    trades = int(np.sum(trade_mask))
    if trades <= 0:
        return {
            "trade_count": 0,
            "avg_pnl": 0.0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "long_share": 0.0,
            "short_share": 0.0,
        }

    pnl = (future_points[trade_mask] * side[trade_mask]) - float(fees_points)
    total_pnl = float(np.sum(pnl))
    avg_pnl = float(np.mean(pnl))
    win_rate = float(np.mean(pnl > 0.0))
    long_share = float(np.mean(side[trade_mask] == 1))
    short_share = float(np.mean(side[trade_mask] == -1))
    return {
        "trade_count": trades,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "long_share": long_share,
        "short_share": short_share,
    }


def _evaluate_threshold_folds(
    prob_up: np.ndarray,
    future_points: np.ndarray,
    long_thr: float,
    short_thr: float,
    fees_points: float,
    folds: int,
    session_ids: Optional[np.ndarray] = None,
    allowed_session_ids: Optional[set[int]] = None,
) -> list[Dict]:
    n = int(len(prob_up))
    folds = max(1, int(folds))
    if n <= 0 or folds <= 1:
        return []
    edges = np.linspace(0, n, folds + 1, dtype=int)
    out: list[Dict] = []
    for idx in range(folds):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        if end - start <= 0:
            continue
        sess_slice = None
        if session_ids is not None:
            sess_slice = np.asarray(session_ids[start:end], dtype=np.int16)
        stats = _evaluate_threshold(
            prob_up[start:end],
            future_points[start:end],
            long_thr,
            short_thr,
            fees_points,
            session_ids=sess_slice,
            allowed_session_ids=allowed_session_ids,
        )
        out.append(stats)
    return out


def _search_threshold(
    prob_up: np.ndarray,
    future_points: np.ndarray,
    thr_min: float,
    thr_max: float,
    thr_step: float,
    min_trades: int,
    fees_points: float,
    score_penalty: float,
    require_both_sides: bool,
    session_ids: Optional[np.ndarray] = None,
    allowed_session_ids: Optional[set[int]] = None,
    min_coverage: float = 0.0,
    coverage_target: float = 0.0,
    coverage_penalty: float = 0.0,
    val_folds: int = 1,
    min_fold_trades: int = 0,
    min_positive_folds: int = 0,
    objective_mean_weight: float = 0.60,
    objective_worst_weight: float = 0.25,
    objective_total_weight: float = 0.15,
    objective_std_penalty: float = 0.10,
) -> Dict:
    best: Optional[Dict] = None
    thr = float(thr_min)
    total_rows = max(1, int(len(prob_up)))
    while thr <= float(thr_max) + 1e-12:
        long_thr = float(thr)
        short_thr = float(1.0 - thr)
        stats = _evaluate_threshold(
            prob_up,
            future_points,
            long_thr,
            short_thr,
            fees_points,
            session_ids=session_ids,
            allowed_session_ids=allowed_session_ids,
        )
        trades = int(stats["trade_count"])
        coverage = float(trades) / float(total_rows)
        if trades >= int(min_trades):
            if require_both_sides and (stats["long_share"] <= 0.0 or stats["short_share"] <= 0.0):
                thr += float(thr_step)
                continue
            if coverage < float(min_coverage):
                thr += float(thr_step)
                continue

            fold_stats = _evaluate_threshold_folds(
                prob_up,
                future_points,
                long_thr,
                short_thr,
                fees_points,
                folds=int(val_folds),
                session_ids=session_ids,
                allowed_session_ids=allowed_session_ids,
            )
            valid_folds = [row for row in fold_stats if int(row.get("trade_count", 0)) >= int(min_fold_trades)]
            positive_folds = int(sum(1 for row in valid_folds if float(row.get("total_pnl", 0.0)) > 0.0))
            if valid_folds and positive_folds < int(min_positive_folds):
                thr += float(thr_step)
                continue

            fold_avg_pnls = np.asarray([float(row.get("avg_pnl", 0.0)) for row in valid_folds], dtype=float)
            mean_fold_avg = float(np.mean(fold_avg_pnls)) if fold_avg_pnls.size else float(stats["avg_pnl"])
            worst_fold_avg = float(np.min(fold_avg_pnls)) if fold_avg_pnls.size else float(stats["avg_pnl"])
            std_fold_avg = float(np.std(fold_avg_pnls)) if fold_avg_pnls.size else 0.0
            total_component = float(stats["avg_pnl"]) * coverage * 100.0
            coverage_gap = abs(coverage - float(coverage_target)) if float(coverage_target) > 0.0 else 0.0
            score = (
                float(objective_mean_weight) * mean_fold_avg
                + float(objective_worst_weight) * worst_fold_avg
                + float(objective_total_weight) * total_component
                - float(objective_std_penalty) * std_fold_avg
                - float(coverage_penalty) * coverage_gap
                - float(score_penalty) / max(1.0, math.sqrt(float(trades)))
            )
            candidate = {
                "threshold": long_thr,
                "short_threshold": short_thr,
                "score": score,
                "coverage": coverage,
                "mean_fold_avg_pnl": mean_fold_avg,
                "worst_fold_avg_pnl": worst_fold_avg,
                "fold_std_avg_pnl": std_fold_avg,
                "positive_folds": positive_folds,
                "evaluated_folds": int(len(valid_folds)),
                **stats,
            }
            if best is None:
                best = candidate
            else:
                if candidate["score"] > best["score"] + 1e-12:
                    best = candidate
                elif abs(candidate["score"] - best["score"]) <= 1e-12:
                    if candidate["total_pnl"] > best["total_pnl"] + 1e-12:
                        best = candidate
        thr += float(thr_step)

    if best is None:
        raise RuntimeError("No valid threshold found (check min_trades/range settings).")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ManifoldStrategy directional model.")
    parser.add_argument("--input", default="es_master.parquet", help="Input bars (.parquet or .csv).")
    parser.add_argument("--start", "--train-start", dest="start", default=None, help="Train start date (YYYY-MM-DD).")
    parser.add_argument("--end", "--train-end", dest="end", default=None, help="Train end date (YYYY-MM-DD).")
    parser.add_argument("--out-dir", default=".", help="Output directory.")
    parser.add_argument("--model-file", default="model_manifold_strategy.joblib")
    parser.add_argument("--thresholds-file", default="manifold_strategy_thresholds.json")
    parser.add_argument("--confluence-file", default="manifold_strategy_confluence.json")
    parser.add_argument("--metrics-file", default="manifold_strategy_metrics.json")
    parser.add_argument("--features-parquet", default="manifold_strategy_features.parquet")
    parser.add_argument("--horizon-bars", type=int, default=10)
    parser.add_argument("--min-move-points", type=float, default=0.75)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--thr-min", type=float, default=0.52)
    parser.add_argument("--thr-max", type=float, default=0.80)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--min-val-trades", type=int, default=100)
    parser.add_argument("--min-val-coverage", type=float, default=0.003)
    parser.add_argument("--coverage-target", type=float, default=0.0075)
    parser.add_argument("--coverage-penalty", type=float, default=0.20)
    parser.add_argument("--score-penalty", type=float, default=0.0)
    parser.add_argument("--val-folds", type=int, default=4)
    parser.add_argument("--min-fold-trades", type=int, default=20)
    parser.add_argument("--min-positive-folds", type=int, default=2)
    parser.add_argument("--objective-mean-weight", type=float, default=0.60)
    parser.add_argument("--objective-worst-weight", type=float, default=0.25)
    parser.add_argument("--objective-total-weight", type=float, default=0.15)
    parser.add_argument("--objective-std-penalty", type=float, default=0.10)
    parser.add_argument("--confluence-trials", type=int, default=320)
    parser.add_argument("--skip-confluence", action="store_true")
    parser.add_argument("--allow-one-sided", action="store_true")
    parser.add_argument("--model", choices=["hgb", "rf"], default="hgb")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--fees-points", type=float, default=None)
    parser.add_argument(
        "--allowed-session-ids",
        default="",
        help="Optional comma-separated session_id allowlist (0=ASIA,1=LONDON,2=NY_AM,3=NY_PM).",
    )
    parser.add_argument(
        "--allow-spreads",
        action="store_true",
        help="Allow spread symbols (contains '-') instead of filtering them out.",
    )
    parser.add_argument(
        "--symbol-regex",
        default=DEFAULT_OUTRIGHT_SYMBOL_REGEX,
        help="Regex for allowed symbols before daily selection.",
    )
    parser.add_argument(
        "--min-valid-price",
        type=float,
        default=100.0,
        help="Minimum valid OHLC value; bars below are dropped.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Feature-loop progress interval in bars (0 = auto heartbeat).",
    )
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
    parser.add_argument(
        "--reuse-features-parquet",
        action="store_true",
        help="Reuse an existing labeled features parquet and skip the feature-build stage.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info("Logging configured (force=True).")
    exp_enabled = bool(args.experimental_window)
    start_arg = args.start
    end_arg = args.end
    if exp_enabled:
        exp_start, exp_end = get_experimental_training_window()
        start_arg = exp_start
        end_arg = exp_end
        logging.info("Experimental window enabled: %s -> %s", start_arg, end_arg)
    if not start_arg or not end_arg:
        logging.warning(
            "Training window is not fully bounded (start=%s end=%s). "
            "For strict OOS workflows, pass both --train-start and --train-end.",
            start_arg,
            end_arg,
        )
    artifact_suffix = resolve_artifact_suffix(args.artifact_suffix, exp_enabled)
    workers = max(1, int(args.workers))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_file = append_artifact_suffix(str(args.model_file), artifact_suffix) if artifact_suffix else str(args.model_file)
    thresholds_file = (
        append_artifact_suffix(str(args.thresholds_file), artifact_suffix)
        if artifact_suffix
        else str(args.thresholds_file)
    )
    confluence_file = (
        append_artifact_suffix(str(args.confluence_file), artifact_suffix)
        if artifact_suffix
        else str(args.confluence_file)
    )
    metrics_file = append_artifact_suffix(str(args.metrics_file), artifact_suffix) if artifact_suffix else str(args.metrics_file)
    features_file = (
        append_artifact_suffix(str(args.features_parquet), artifact_suffix)
        if artifact_suffix
        else str(args.features_parquet)
    )

    model_path = out_dir / model_file
    thresholds_path = out_dir / thresholds_file
    confluence_path = out_dir / confluence_file
    metrics_path = out_dir / metrics_file
    features_path = out_dir / features_file

    data: Optional[pd.DataFrame] = None
    if args.reuse_features_parquet:
        if not features_path.exists():
            raise RuntimeError(
                f"Requested --reuse-features-parquet but file does not exist: {features_path}"
            )
        if start_arg or end_arg:
            logging.warning(
                "Ignoring --start/--end because --reuse-features-parquet loads cached labeled rows as-is."
            )
        logging.info("Loading cached training data: %s", features_path)
        data = _load_cached_training_data(features_path)
        logging.info(
            "Cached training data loaded: rows=%d range=%s -> %s",
            len(data),
            data.index.min() if len(data) else None,
            data.index.max() if len(data) else None,
        )

    if data is None:
        input_path = Path(args.input)
        if input_path.suffix.lower() in (".parquet", ".pq") and not input_path.exists():
            csv_fallback = input_path.with_suffix(".csv")
            if csv_fallback.exists():
                logging.info(
                    "Parquet input missing; building parquet cache from CSV: %s -> %s",
                    csv_fallback,
                    input_path,
                )
                csv_df = _load_bars(
                    csv_fallback,
                    allow_spreads=bool(args.allow_spreads),
                    symbol_regex=str(args.symbol_regex or "").strip() or None,
                    min_valid_price=float(args.min_valid_price),
                ).sort_index()
                input_path.parent.mkdir(parents=True, exist_ok=True)
                csv_df.to_parquet(input_path, index=True)
                logging.info("Wrote parquet cache: %s", input_path)
        bars = _load_bars(
            input_path,
            allow_spreads=bool(args.allow_spreads),
            symbol_regex=str(args.symbol_regex or "").strip() or None,
            min_valid_price=float(args.min_valid_price),
        ).sort_index()
        start_ts = _parse_date(start_arg, is_end=False)
        end_ts = _parse_date(end_arg, is_end=True)
        bars = _filter_range(bars, start_ts, end_ts)
        if args.max_rows and int(args.max_rows) > 0:
            bars = bars.iloc[-int(args.max_rows) :]
        if bars.empty:
            raise RuntimeError("No rows available after applying date filters.")
        logging.info(
            "Loaded bars: rows=%d range=%s -> %s",
            len(bars),
            bars.index.min(),
            bars.index.max(),
        )

        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg["enabled"] = True
        feat = build_training_feature_frame(
            bars,
            manifold_cfg=manifold_cfg,
            log_every=int(args.log_every or 0),
        )
        if feat.empty:
            raise RuntimeError("Feature build returned empty frame.")
        logging.info("Feature frame built: rows=%d cols=%d", len(feat), len(feat.columns))

        label, fwd_points = _build_labels(
            bars["close"],
            horizon_bars=int(args.horizon_bars),
            min_move_points=float(args.min_move_points),
        )
        data = feat.join(label.rename("label"), how="inner")
        data = data.join(fwd_points.rename("future_points"), how="inner")
        data = data.dropna(subset=["label", "future_points"])
        data["label"] = data["label"].astype(int)
        if data.empty:
            raise RuntimeError("No labeled training rows after joins.")

    label_counts = data["label"].value_counts().to_dict()
    logging.info(
        "Labeled rows=%d (up=%d down=%d)",
        len(data),
        int(label_counts.get(1, 0)),
        int(label_counts.get(0, 0)),
    )

    n = len(data)
    train_end = int(n * float(args.train_frac))
    val_end = train_end + int(n * float(args.val_frac))
    train_end = max(100, min(train_end, n - 50))
    val_end = max(train_end + 25, min(val_end, n - 25))

    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]
    if train.empty or val.empty or test.empty:
        raise RuntimeError("Train/val/test split is empty; adjust fractions or input size.")
    if train["label"].nunique() < 2:
        raise RuntimeError("Train split has only one class; adjust min_move_points/horizon.")
    if val["label"].nunique() < 2:
        raise RuntimeError("Validation split has only one class; adjust settings.")

    logging.info("Split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    X_train = train[FEATURE_COLUMNS]
    y_train = train["label"]
    X_val = val[FEATURE_COLUMNS]
    y_val = val["label"]
    X_test = test[FEATURE_COLUMNS]
    y_test = test["label"]
    val_session_ids = _coerce_session_ids(X_val["session_id"])
    test_session_ids = _coerce_session_ids(X_test["session_id"])
    allowed_session_ids = _parse_allowed_sessions(args.allowed_session_ids)
    if allowed_session_ids:
        session_names = [_session_name_from_id(s) for s in sorted(allowed_session_ids)]
        logging.info("Session allowlist enabled: ids=%s names=%s", sorted(allowed_session_ids), session_names)

    model_choice = str(args.model)
    fit_workers = workers
    model = _build_model(model_choice, args.seed, workers=fit_workers)
    sample_weight = _sample_weights(y_train)
    try:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    except (PermissionError, OSError) as exc:
        if model_choice == "hgb":
            logging.warning(
                "HGB fit failed in this environment (%s). Falling back to RandomForest.",
                exc,
            )
            model_choice = "rf"
            model = _build_model(model_choice, args.seed, workers=fit_workers)
            try:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            except (PermissionError, OSError) as rf_exc:
                logging.warning(
                    "RandomForest fit failed with workers=%d (%s). Retrying with workers=1.",
                    fit_workers,
                    rf_exc,
                )
                fit_workers = 1
                model = _build_model(model_choice, args.seed, workers=fit_workers)
                model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            logging.warning(
                "RandomForest fit failed with workers=%d (%s). Retrying with workers=1.",
                fit_workers,
                exc,
            )
            fit_workers = 1
            model = _build_model(model_choice, args.seed, workers=fit_workers)
            model.fit(X_train, y_train, sample_weight=sample_weight)
    calibrated = _fit_prefit_calibrator(model, X_val, y_val)
    logging.info("Model fitted and calibrated (%s, workers=%d).", model_choice, fit_workers)

    prob_val = calibrated.predict_proba(X_val)[:, 1]
    if args.fees_points is not None:
        fees_points = float(args.fees_points)
    else:
        risk_cfg = CONFIG.get("RISK", {}) or {}
        point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
        fees_per_side = float(risk_cfg.get("FEES_PER_SIDE", 2.5) or 2.5)
        fees_points = (fees_per_side * 2.0) / max(1e-9, point_value)

    best_thr = _search_threshold(
        prob_val,
        val["future_points"].to_numpy(dtype=float),
        thr_min=float(args.thr_min),
        thr_max=float(args.thr_max),
        thr_step=float(args.thr_step),
        min_trades=int(args.min_val_trades),
        fees_points=float(fees_points),
        score_penalty=float(args.score_penalty),
        require_both_sides=not bool(args.allow_one_sided),
        session_ids=val_session_ids,
        allowed_session_ids=allowed_session_ids,
        min_coverage=float(args.min_val_coverage),
        coverage_target=float(args.coverage_target),
        coverage_penalty=float(args.coverage_penalty),
        val_folds=int(args.val_folds),
        min_fold_trades=int(args.min_fold_trades),
        min_positive_folds=int(args.min_positive_folds),
        objective_mean_weight=float(args.objective_mean_weight),
        objective_worst_weight=float(args.objective_worst_weight),
        objective_total_weight=float(args.objective_total_weight),
        objective_std_penalty=float(args.objective_std_penalty),
    )
    logging.info(
        "Best threshold: long=%.3f short=%.3f trades=%d coverage=%.4f win=%.2f%% avg_pnl=%.3f total_pnl=%.1f",
        float(best_thr["threshold"]),
        float(best_thr["short_threshold"]),
        int(best_thr["trade_count"]),
        float(best_thr.get("coverage", 0.0)),
        100.0 * float(best_thr["win_rate"]),
        float(best_thr["avg_pnl"]),
        float(best_thr["total_pnl"]),
    )

    confluence_enabled = not bool(args.skip_confluence)
    confluence_payload: Dict = {
        "enabled": False,
        "reason": "skip_confluence_flag" if not confluence_enabled else "not_calibrated",
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }

    final_threshold = float(best_thr["threshold"])
    final_short_threshold = float(best_thr["short_threshold"])
    prob_test = calibrated.predict_proba(X_test)[:, 1]
    prob_test_effective = prob_test

    if confluence_enabled:
        fallback_stats = _evaluate_threshold(
            prob_val,
            val["future_points"].to_numpy(dtype=float),
            long_thr=float(best_thr["threshold"]),
            short_thr=float(best_thr["short_threshold"]),
            fees_points=float(fees_points),
            session_ids=val_session_ids,
            allowed_session_ids=allowed_session_ids,
        )
        fallback_eval = {
            "threshold": float(best_thr["threshold"]),
            "short_threshold": float(best_thr["short_threshold"]),
            "score": float(best_thr.get("score", 0.0)),
            **fallback_stats,
        }

        def _eval_with_search(prob_arr: np.ndarray) -> Dict:
            try:
                return _search_threshold(
                    prob_arr,
                    val["future_points"].to_numpy(dtype=float),
                    thr_min=float(args.thr_min),
                    thr_max=float(args.thr_max),
                    thr_step=float(args.thr_step),
                    min_trades=int(args.min_val_trades),
                    fees_points=float(fees_points),
                    score_penalty=float(args.score_penalty),
                    require_both_sides=not bool(args.allow_one_sided),
                    session_ids=val_session_ids,
                    allowed_session_ids=allowed_session_ids,
                    min_coverage=float(args.min_val_coverage),
                    coverage_target=float(args.coverage_target),
                    coverage_penalty=float(args.coverage_penalty),
                    val_folds=int(args.val_folds),
                    min_fold_trades=int(args.min_fold_trades),
                    min_positive_folds=int(args.min_positive_folds),
                    objective_mean_weight=float(args.objective_mean_weight),
                    objective_worst_weight=float(args.objective_worst_weight),
                    objective_total_weight=float(args.objective_total_weight),
                    objective_std_penalty=float(args.objective_std_penalty),
                )
            except Exception:
                return dict(fallback_eval)

        try:
            conf_best = calibrate_confluence(
                feature_frame=X_val.reset_index(drop=True),
                prob_up=prob_val,
                evaluate_fn=_eval_with_search,
                trials=int(args.confluence_trials),
                seed=int(args.seed) + 991,
            )
            confluence_params = dict(conf_best.get("params", {}))
            stats = dict(conf_best.get("stats", {}))
            candidate_score = float(stats.get("score", conf_best.get("score", -math.inf)))
            baseline_score = float(fallback_eval.get("score", -math.inf))
            candidate_total = float(stats.get("total_pnl", 0.0))
            baseline_total = float(fallback_eval.get("total_pnl", 0.0))

            use_confluence = False
            if candidate_score > baseline_score + 1e-12:
                use_confluence = True
            elif abs(candidate_score - baseline_score) <= 1e-12 and candidate_total > baseline_total + 1e-12:
                use_confluence = True

            if use_confluence:
                test_adj = apply_confluence_formula(
                    X_test.reset_index(drop=True),
                    prob_test,
                    confluence_params,
                )
                prob_test_effective = test_adj["prob_up_adj"]
                final_threshold = float(stats.get("threshold", final_threshold))
                final_short_threshold = float(stats.get("short_threshold", final_short_threshold))

                confluence_payload = {
                    "enabled": True,
                    "trained_at": pd.Timestamp.utcnow().isoformat(),
                    "trials": int(args.confluence_trials),
                    "params": confluence_params,
                    "validation": {
                        "threshold": final_threshold,
                        "short_threshold": final_short_threshold,
                        "score": float(stats.get("score", conf_best.get("score", 0.0))),
                        "trade_count": int(stats.get("trade_count", 0)),
                        "win_rate": float(stats.get("win_rate", 0.0)),
                        "avg_pnl": float(stats.get("avg_pnl", 0.0)),
                        "total_pnl": float(stats.get("total_pnl", 0.0)),
                        "long_share": float(stats.get("long_share", 0.0)),
                        "short_share": float(stats.get("short_share", 0.0)),
                        "alpha_mean": float(conf_best.get("alpha_mean", 0.0)),
                        "alpha_std": float(conf_best.get("alpha_std", 0.0)),
                    },
                }
                logging.info(
                    "Confluence accepted: long=%.3f short=%.3f trades=%d avg_pnl=%.3f",
                    final_threshold,
                    final_short_threshold,
                    int(stats.get("trade_count", 0)),
                    float(stats.get("avg_pnl", 0.0)),
                )
            else:
                confluence_enabled = False
                confluence_payload = {
                    "enabled": False,
                    "reason": "not_better_than_baseline",
                    "trained_at": pd.Timestamp.utcnow().isoformat(),
                    "trials": int(args.confluence_trials),
                    "baseline_validation": fallback_eval,
                    "candidate_validation": {
                        "threshold": float(stats.get("threshold", final_threshold)),
                        "short_threshold": float(stats.get("short_threshold", final_short_threshold)),
                        "score": candidate_score,
                        "trade_count": int(stats.get("trade_count", 0)),
                        "win_rate": float(stats.get("win_rate", 0.0)),
                        "avg_pnl": float(stats.get("avg_pnl", 0.0)),
                        "total_pnl": float(stats.get("total_pnl", 0.0)),
                        "long_share": float(stats.get("long_share", 0.0)),
                        "short_share": float(stats.get("short_share", 0.0)),
                        "alpha_mean": float(conf_best.get("alpha_mean", 0.0)),
                        "alpha_std": float(conf_best.get("alpha_std", 0.0)),
                    },
                }
                logging.info(
                    "Confluence rejected: baseline_score=%.3f candidate_score=%.3f",
                    baseline_score,
                    candidate_score,
                )
        except Exception as exc:
            confluence_enabled = False
            confluence_payload = {
                "enabled": False,
                "reason": f"calibration_failed: {exc}",
                "trained_at": pd.Timestamp.utcnow().isoformat(),
            }
            logging.warning("Confluence calibration skipped after failure: %s", exc)

    test_trade = _evaluate_threshold(
        prob_test_effective,
        test["future_points"].to_numpy(dtype=float),
        long_thr=float(final_threshold),
        short_thr=float(final_short_threshold),
        fees_points=float(fees_points),
        session_ids=test_session_ids,
        allowed_session_ids=allowed_session_ids,
    )
    pred_test = (prob_test_effective >= 0.5).astype(int)
    test_metrics = {
        "row_count": int(len(test)),
        "accuracy": float(accuracy_score(y_test, pred_test)),
        "precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "recall": float(recall_score(y_test, pred_test, zero_division=0)),
        "f1": float(f1_score(y_test, pred_test, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob_test_effective)),
        "trade_count": int(test_trade["trade_count"]),
        "win_rate": float(test_trade["win_rate"]),
        "avg_pnl": float(test_trade["avg_pnl"]),
        "total_pnl": float(test_trade["total_pnl"]),
        "long_share": float(test_trade["long_share"]),
        "short_share": float(test_trade["short_share"]),
        "confluence_enabled": bool(confluence_enabled),
        "threshold": float(final_threshold),
        "short_threshold": float(final_short_threshold),
        "train_window": {
            "start": str(start_arg) if start_arg else None,
            "end": str(end_arg) if end_arg else None,
        },
    }
    logging.info(
        "Test: acc=%.3f f1=%.3f auc=%.3f trades=%d win=%.2f%% avg_pnl=%.3f total_pnl=%.1f",
        test_metrics["accuracy"],
        test_metrics["f1"],
        test_metrics["roc_auc"],
        test_metrics["trade_count"],
        100.0 * test_metrics["win_rate"],
        test_metrics["avg_pnl"],
        test_metrics["total_pnl"],
    )

    joblib.dump(calibrated, model_path)
    validation_block = dict(best_thr)
    if confluence_enabled and isinstance(confluence_payload.get("validation"), dict):
        validation_block = dict(confluence_payload["validation"])
    threshold_payload = {
        "threshold": float(final_threshold),
        "short_threshold": float(final_short_threshold),
        "horizon_bars": int(args.horizon_bars),
        "min_move_points": float(args.min_move_points),
        "fees_points": float(fees_points),
        "feature_columns": list(FEATURE_COLUMNS),
        "model_type": model_choice,
        "workers": int(fit_workers),
        "confluence_enabled": bool(confluence_enabled),
        "confluence_file": str(confluence_file),
        "allowed_session_ids": sorted(int(s) for s in allowed_session_ids) if allowed_session_ids else [],
        "allowed_session_names": (
            [_session_name_from_id(s) for s in sorted(allowed_session_ids)]
            if allowed_session_ids
            else []
        ),
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "train_window": {
            "start": str(start_arg) if start_arg else None,
            "end": str(end_arg) if end_arg else None,
        },
        "validation": {
            "trade_count": int(validation_block.get("trade_count", 0)),
            "win_rate": float(validation_block.get("win_rate", 0.0)),
            "avg_pnl": float(validation_block.get("avg_pnl", 0.0)),
            "total_pnl": float(validation_block.get("total_pnl", 0.0)),
            "long_share": float(validation_block.get("long_share", 0.0)),
            "short_share": float(validation_block.get("short_share", 0.0)),
            "score": float(validation_block.get("score", 0.0)),
            "coverage": float(validation_block.get("coverage", 0.0)),
            "mean_fold_avg_pnl": float(validation_block.get("mean_fold_avg_pnl", 0.0)),
            "worst_fold_avg_pnl": float(validation_block.get("worst_fold_avg_pnl", 0.0)),
            "fold_std_avg_pnl": float(validation_block.get("fold_std_avg_pnl", 0.0)),
            "positive_folds": int(validation_block.get("positive_folds", 0)),
            "evaluated_folds": int(validation_block.get("evaluated_folds", 0)),
        },
    }
    thresholds_path.write_text(json.dumps(threshold_payload, indent=2))
    confluence_path.write_text(json.dumps(confluence_payload, indent=2))
    metrics_path.write_text(json.dumps(test_metrics, indent=2))
    data.to_parquet(features_path, index=True)

    logging.info("Saved model: %s", model_path)
    logging.info("Saved thresholds: %s", thresholds_path)
    logging.info("Saved confluence: %s", confluence_path)
    logging.info("Saved metrics: %s", metrics_path)
    logging.info("Saved training features parquet: %s", features_path)


if __name__ == "__main__":
    main()
