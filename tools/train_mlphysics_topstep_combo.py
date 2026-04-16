from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_physics_topstep_combo_common import (
    MODEL_CATEGORICAL_COLUMNS,
    build_topstep_combo_bank_event_dataset,
    build_topstep_combo_dataset,
    feature_column_sets,
    load_combo_catalog,
    load_market_data,
    simulate_bracket_trade,
)


MES_FEE_POINTS = 0.15
CURATED_NUMERIC_FEATURE_CANDIDATES = [
    "macro_index",
    "atr_pts",
    "macro_window_minute_offset",
    "dist_nearest_abs_bank_pts",
    "dist_nearest_abs_bank_atr",
    "abs_bank_range_touch",
    "abs_bank_span_count",
    "abs_bank_open_close_cross",
    "dist_nearest_rel_bank_pts",
    "dist_nearest_rel_bank_atr",
    "rel_bank_range_touch",
    "rel_bank_span_count",
    "rel_bank_open_close_cross",
    "abs_rel_bank_gap_pts",
    "abs_rel_bank_confluence",
    "abs_bank_vs_open_ref_steps",
    "rel_bank_vs_open_ref_steps",
    "abs_rel_bank_gap_steps",
    "abs_bank_touched_offset_steps",
    "rel_bank_touched_offset_steps",
    "abs_bank_level_nearest_dist_pts",
    "abs_bank_level_nearest_dist_atr",
    "abs_bank_level_count_025atr",
    "abs_bank_level_count_050atr",
    "abs_bank_level_count_100atr",
    "rel_bank_level_nearest_dist_pts",
    "rel_bank_level_nearest_dist_atr",
    "rel_bank_level_count_025atr",
    "rel_bank_level_count_050atr",
    "rel_bank_level_count_100atr",
    "best_bank_level_nearest_dist_atr",
    "combined_bank_level_count_050atr",
    "shared_bank_nearest_level",
    "price_vs_open_ref_atr",
    "window_range_atr",
    "window_extension_up_atr",
    "window_extension_dn_atr",
    "high_breach_count",
    "low_breach_count",
    "high_selected_count",
    "low_selected_count",
    "upper_breached_prev_sess",
    "lower_breached_prev_sess",
    "dist_upper_prev_sess_atr",
    "dist_lower_prev_sess_atr",
    "upper_breached_q1",
    "lower_breached_q1",
    "dist_upper_q1_atr",
    "dist_lower_q1_atr",
    "upper_breached_mid_orb",
    "lower_breached_mid_orb",
    "dist_upper_mid_orb_atr",
    "dist_lower_mid_orb_atr",
    "upper_breached_morn_orb",
    "lower_breached_morn_orb",
    "dist_upper_morn_orb_atr",
    "dist_lower_morn_orb_atr",
    "upper_breached_prev_day",
    "lower_breached_prev_day",
    "dist_upper_prev_day_atr",
    "dist_lower_prev_day_atr",
    "upper_breached_sess_max",
    "lower_breached_sess_max",
    "dist_upper_sess_max_atr",
    "dist_lower_sess_max_atr",
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


def _resolve_file(path_arg: str) -> Path:
    candidate = Path(path_arg).expanduser()
    if candidate.exists():
        return candidate
    fallback = ROOT / path_arg
    if fallback.exists():
        return fallback
    raise SystemExit(f"Path not found: {path_arg}")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def resolve_feature_columns(
    dataset: pd.DataFrame,
    *,
    feature_mode: str,
) -> tuple[list[str], list[str], list[str]]:
    feature_columns, categorical_columns, numeric_columns = feature_column_sets(dataset)
    mode = str(feature_mode or "curated").strip().lower()
    if mode == "full":
        return feature_columns, categorical_columns, numeric_columns
    if mode == "curated":
        curated_numeric = [column for column in CURATED_NUMERIC_FEATURE_CANDIDATES if column in numeric_columns]
        return list(categorical_columns) + list(curated_numeric), categorical_columns, curated_numeric
    raise SystemExit(f"Unsupported feature mode: {feature_mode}")


def build_dataset(
    market_df: pd.DataFrame,
    combo_catalog: pd.DataFrame,
    *,
    dataset_mode: str,
) -> pd.DataFrame:
    mode = str(dataset_mode or "snapshot").strip().lower()
    if mode == "snapshot":
        return build_topstep_combo_dataset(market_df, combo_catalog)
    if mode == "bank_event":
        return build_topstep_combo_bank_event_dataset(market_df, combo_catalog)
    raise SystemExit(f"Unsupported dataset mode: {dataset_mode}")


def _build_preprocessor(categorical_columns: list[str], numeric_columns: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ("numeric", "passthrough", numeric_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def _build_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=0.05,
        max_iter=250,
        random_state=int(seed),
    )


def _build_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=0.05,
        max_iter=250,
        random_state=int(seed),
    )


def _class_sample_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos <= 0 or neg <= 0:
        return np.ones(len(labels), dtype=float)
    pos_weight = float(neg) / float(pos)
    return np.where(labels == 1, pos_weight, 1.0).astype(float)


def fit_side_bundle(
    frame: pd.DataFrame,
    *,
    side: str,
    categorical_columns: list[str],
    numeric_columns: list[str],
    seed: int,
) -> dict[str, Any]:
    features = frame[categorical_columns + numeric_columns].copy()
    label_win = frame[f"{side}_win"].to_numpy(dtype=int)
    weights = _class_sample_weights(label_win)

    prep = _build_preprocessor(categorical_columns, numeric_columns)
    clf = Pipeline([("prep", prep), ("model", _build_classifier(seed))])
    clf.fit(features, label_win, model__sample_weight=weights)

    pnl_model = Pipeline([("prep", _build_preprocessor(categorical_columns, numeric_columns)), ("model", _build_regressor(seed + 11))])
    pnl_model.fit(features, frame[f"{side}_exit_pnl_pts"].to_numpy(dtype=float))

    mfe_model = Pipeline([("prep", _build_preprocessor(categorical_columns, numeric_columns)), ("model", _build_regressor(seed + 23))])
    mfe_model.fit(features, frame[f"{side}_mfe_pts"].to_numpy(dtype=float))

    mae_model = Pipeline([("prep", _build_preprocessor(categorical_columns, numeric_columns)), ("model", _build_regressor(seed + 37))])
    mae_model.fit(features, frame[f"{side}_mae_pts"].to_numpy(dtype=float))

    return {
        "classifier": clf,
        "pnl_model": pnl_model,
        "mfe_model": mfe_model,
        "mae_model": mae_model,
    }


def predict_side_bundle(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    *,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> pd.DataFrame:
    features = frame[categorical_columns + numeric_columns].copy()
    out = pd.DataFrame(index=frame.index)
    out["prob"] = bundle["classifier"].predict_proba(features)[:, 1]
    out["pred_pnl"] = bundle["pnl_model"].predict(features)
    out["pred_mfe"] = bundle["mfe_model"].predict(features)
    out["pred_mae"] = bundle["mae_model"].predict(features)
    return out


def trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "trade_days": 0,
            "trades_per_day": 0.0,
            "win_rate": 0.0,
            "net_points": 0.0,
            "avg_points": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_points": 0.0,
        }

    work = trades.sort_values("entry_time").copy()
    pnl = work["pnl_points_net"].to_numpy(dtype=float)
    equity = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    drawdowns = running_peak[1:] - equity
    gross_profit = float(np.sum(pnl[pnl > 0.0]))
    gross_loss = float(-np.sum(pnl[pnl < 0.0]))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (999.0 if gross_profit > 0.0 else 0.0)
    trade_days = int(work["trade_day"].astype(str).nunique()) if "trade_day" in work.columns else 0
    trades_per_day = float(len(work) / trade_days) if trade_days > 0 else 0.0

    return {
        "trades": int(len(work)),
        "trade_days": int(trade_days),
        "trades_per_day": float(trades_per_day),
        "win_rate": float(np.mean(pnl > 0.0)),
        "net_points": float(np.sum(pnl)),
        "avg_points": float(np.mean(pnl)),
        "profit_factor": float(profit_factor),
        "max_drawdown_points": float(np.max(drawdowns)) if drawdowns.size else 0.0,
    }


def bootstrap_monte_carlo(trades: pd.DataFrame, *, simulations: int, seed: int) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {"enabled": True, "status": "empty", "simulations": int(simulations)}

    daily = (
        trades.groupby("trade_day", sort=True)["pnl_points_net"]
        .sum()
        .to_numpy(dtype=float)
    )
    if daily.size <= 0:
        return {"enabled": True, "status": "empty", "simulations": int(simulations)}

    rng = np.random.default_rng(int(seed))
    net = np.empty(int(simulations), dtype=float)
    max_dd = np.empty(int(simulations), dtype=float)
    wins = np.empty(int(simulations), dtype=float)

    for idx in range(int(simulations)):
        sampled = daily[rng.integers(0, len(daily), size=len(daily))]
        net[idx] = float(np.sum(sampled))
        equity = np.cumsum(sampled)
        peaks = np.maximum.accumulate(np.concatenate(([0.0], equity)))
        drawdowns = peaks[1:] - equity
        max_dd[idx] = float(np.max(drawdowns)) if drawdowns.size else 0.0
        wins[idx] = float(np.mean(sampled > 0.0))

    def _summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }

    return {
        "enabled": True,
        "status": "ok",
        "simulations": int(simulations),
        "seed": int(seed),
        "trade_days": int(len(daily)),
        "net_points": _summary(net),
        "max_drawdown_points": _summary(max_dd),
        "positive_day_share": _summary(wins),
    }


def make_walkforward_splits(
    trade_days: list[str],
    *,
    min_train_days: int,
    test_days: int,
    max_folds: int,
) -> list[dict[str, Any]]:
    ordered_days = sorted(pd.Index(trade_days).astype(str).unique().tolist())
    if len(ordered_days) < 2:
        return []

    folds: list[dict[str, Any]] = []
    cursor = max(int(min_train_days), max(60, len(ordered_days) // 2))
    while cursor < len(ordered_days):
        train_days = ordered_days[:cursor]
        test_slice = ordered_days[cursor : cursor + int(test_days)]
        if len(test_slice) < max(20, int(test_days // 2)):
            break
        folds.append(
            {
                "train_days": train_days,
                "test_days": test_slice,
            }
        )
        cursor += int(test_days)

    if not folds:
        split = max(min_train_days, int(len(ordered_days) * 0.7))
        if split < len(ordered_days):
            folds.append(
                {
                    "train_days": ordered_days[:split],
                    "test_days": ordered_days[split:],
                }
            )

    return folds[-int(max_folds) :]


def split_fit_tune_days(train_days: list[str], *, tune_fraction: float = 0.2, min_tune_days: int = 45) -> tuple[list[str], list[str]]:
    tune_days = max(int(round(len(train_days) * float(tune_fraction))), int(min_tune_days))
    tune_days = min(tune_days, max(20, len(train_days) // 2))
    fit_days = train_days[:-tune_days]
    tune = train_days[-tune_days:]
    if not fit_days:
        midpoint = max(1, len(train_days) // 2)
        fit_days = train_days[:midpoint]
        tune = train_days[midpoint:]
    return fit_days, tune


def build_prediction_frame(
    frame: pd.DataFrame,
    *,
    long_bundle: dict[str, Any],
    short_bundle: dict[str, Any],
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> pd.DataFrame:
    long_pred = predict_side_bundle(long_bundle, frame, categorical_columns=categorical_columns, numeric_columns=numeric_columns)
    short_pred = predict_side_bundle(short_bundle, frame, categorical_columns=categorical_columns, numeric_columns=numeric_columns)
    work = frame.reset_index(drop=True).copy()
    for column in long_pred.columns:
        work[f"long_{column}"] = long_pred[column].to_numpy()
        work[f"short_{column}"] = short_pred[column].to_numpy()
    work["long_score"] = work["long_pred_pnl"] * work["long_prob"]
    work["short_score"] = work["short_pred_pnl"] * work["short_prob"]
    return work


def _selection_metric_score(
    metrics: dict[str, Any],
    *,
    target_trades_per_day: float = 0.0,
) -> float:
    trades = max(float(metrics.get("trades", 0) or 0.0), 0.0)
    net_points = float(metrics.get("net_points", 0.0) or 0.0)
    avg_points = float(metrics.get("avg_points", 0.0) or 0.0)
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    max_drawdown = float(metrics.get("max_drawdown_points", 0.0) or 0.0)
    trades_per_day = float(metrics.get("trades_per_day", 0.0) or 0.0)
    expectancy_component = avg_points * min(trades, 150.0)
    profit_factor_component = max(profit_factor - 1.0, 0.0) * 40.0
    flow_component = 0.0
    if float(target_trades_per_day or 0.0) > 0.0:
        target = float(target_trades_per_day)
        achieved = min(trades_per_day, target)
        flow_component += achieved * 70.0
        if trades_per_day < target:
            flow_component -= (target - trades_per_day) * 45.0
    return float(net_points + expectancy_component + profit_factor_component + flow_component - max_drawdown * 0.35)


def _area_key(macro_name: str, session_window: str, open_ref_name: str, side: str) -> str:
    return "|".join(
        [
            str(macro_name or "").strip(),
            str(session_window or "").strip(),
            str(open_ref_name or "").strip(),
            str(side or "").strip().upper(),
        ]
    )


def _combo_side_key(combo_key: str, side: str) -> str:
    return "|".join([str(combo_key or "").strip(), str(side or "").strip().upper()])


def _profit_factor(values: pd.Series) -> float:
    pnl = values.to_numpy(dtype=float)
    gross_profit = float(np.sum(pnl[pnl > 0.0]))
    gross_loss = float(-np.sum(pnl[pnl < 0.0]))
    if gross_loss <= 1e-12:
        return 999.0 if gross_profit > 0.0 else 0.0
    return float(gross_profit / gross_loss)


def _summarize_viability_groups(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    grouped = (
        trades.groupby(group_columns, dropna=False)["pnl_points_net"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "trades", "sum": "net_points", "mean": "avg_points"})
    )
    profit_factor = (
        trades.groupby(group_columns, dropna=False)["pnl_points_net"]
        .apply(_profit_factor)
        .reset_index(name="profit_factor")
    )
    daily_share = (
        trades.groupby(group_columns + ["trade_day"], dropna=False)["pnl_points_net"]
        .sum()
        .groupby(group_columns, dropna=False)
        .apply(lambda values: float(np.mean(values.to_numpy(dtype=float) > 0.0)))
        .reset_index(name="positive_day_share")
    )
    grouped = grouped.merge(profit_factor, on=group_columns, how="left")
    grouped = grouped.merge(daily_share, on=group_columns, how="left")
    grouped["expectancy_score"] = grouped["avg_points"] * np.sqrt(grouped["trades"].clip(lower=1))
    grouped = grouped.sort_values(
        ["net_points", "expectancy_score", "avg_points", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return grouped


def build_viability_allowlists(
    trades: pd.DataFrame,
    *,
    gate_params: dict[str, Any],
) -> dict[str, Any]:
    area_min_trades = int(gate_params.get("area_min_trades", 0) or 0)
    combo_min_trades = int(gate_params.get("combo_min_trades", 0) or 0)
    area_min_avg = float(gate_params.get("area_min_avg", 0.0) or 0.0)
    combo_min_avg = float(gate_params.get("combo_min_avg", 0.0) or 0.0)
    area_min_pf = float(gate_params.get("area_min_pf", 0.0) or 0.0)
    combo_min_pf = float(gate_params.get("combo_min_pf", 0.0) or 0.0)
    area_min_positive_day_share = float(gate_params.get("area_min_positive_day_share", 0.0) or 0.0)
    combo_min_positive_day_share = float(gate_params.get("combo_min_positive_day_share", 0.0) or 0.0)

    area_stats = _summarize_viability_groups(
        trades,
        ["macro_name", "session_window", "open_ref_name", "side"],
    )
    combo_stats = _summarize_viability_groups(
        trades,
        ["combo_key", "macro_name", "session_window", "open_ref_name", "side"],
    )

    area_enabled = area_min_trades > 0
    combo_enabled = combo_min_trades > 0
    allowed_area_keys: set[str] = set()
    allowed_combo_side_keys: set[str] = set()

    if area_enabled and not area_stats.empty:
        area_allowed = area_stats.loc[
            (area_stats["trades"] >= area_min_trades)
            & (area_stats["avg_points"] > area_min_avg)
            & (area_stats["profit_factor"] >= area_min_pf)
            & (area_stats["positive_day_share"] >= area_min_positive_day_share)
        ].copy()
        allowed_area_keys = {
            _area_key(row.macro_name, row.session_window, row.open_ref_name, row.side)
            for row in area_allowed.itertuples(index=False)
        }
    else:
        area_allowed = area_stats.head(0).copy()

    if combo_enabled and not combo_stats.empty:
        combo_source = combo_stats
        if area_enabled and allowed_area_keys:
            combo_source = combo_source.loc[
                combo_source.apply(
                    lambda row: _area_key(row["macro_name"], row["session_window"], row["open_ref_name"], row["side"])
                    in allowed_area_keys,
                    axis=1,
                )
            ].copy()
        combo_allowed = combo_source.loc[
            (combo_source["trades"] >= combo_min_trades)
            & (combo_source["avg_points"] > combo_min_avg)
            & (combo_source["profit_factor"] >= combo_min_pf)
            & (combo_source["positive_day_share"] >= combo_min_positive_day_share)
        ].copy()
        allowed_combo_side_keys = {
            _combo_side_key(row.combo_key, row.side)
            for row in combo_allowed.itertuples(index=False)
        }
    else:
        combo_allowed = combo_stats.head(0).copy()

    return {
        "area_enabled": bool(area_enabled),
        "combo_enabled": bool(combo_enabled),
        "allowed_area_keys": allowed_area_keys,
        "allowed_combo_side_keys": allowed_combo_side_keys,
        "area_preview": area_allowed.head(25).to_dict(orient="records"),
        "combo_preview": combo_allowed.head(25).to_dict(orient="records"),
        "area_stats": area_stats,
        "combo_stats": combo_stats,
    }


def filter_candidates_by_allowlists(
    candidates: pd.DataFrame,
    *,
    allowlists: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if candidates is None or candidates.empty or not allowlists:
        return candidates.copy() if candidates is not None else pd.DataFrame()
    work = candidates.copy()
    if bool(allowlists.get("area_enabled", False)):
        allowed_area_keys = set(allowlists.get("allowed_area_keys", set()) or set())
        if not allowed_area_keys:
            return work.iloc[0:0].copy()
        work = work.loc[work["area_key"].isin(allowed_area_keys)].copy()
    if bool(allowlists.get("combo_enabled", False)):
        allowed_combo_keys = set(allowlists.get("allowed_combo_side_keys", set()) or set())
        if not allowed_combo_keys:
            return work.iloc[0:0].copy()
        work = work.loc[work["combo_side_key"].isin(allowed_combo_keys)].copy()
    return work.reset_index(drop=True)


def select_live_compatible_trades(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    work = candidates.sort_values(
        ["entry_pos", "pred_score", "pred_prob"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    occupied_until_pos: int | None = None
    for row in work.itertuples(index=False):
        entry_pos = int(row.entry_pos)
        exit_pos = int(row.exit_pos)
        if occupied_until_pos is not None and entry_pos <= occupied_until_pos:
            continue
        records.append(row._asdict())
        occupied_until_pos = exit_pos
    return pd.DataFrame.from_records(records)


def collect_trade_candidates(
    scored: pd.DataFrame,
    *,
    params: dict[str, float],
    market_index: pd.DatetimeIndex,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    prob_threshold = float(params["prob_threshold"])
    score_threshold = float(params["score_threshold"])
    tp_mult = float(params["tp_mult"])
    sl_mult = float(params["sl_mult"])
    min_rr = float(params["min_rr"])
    min_tp = float(params["min_tp"])
    max_tp = float(params["max_tp"])
    min_sl = float(params["min_sl"])
    max_sl = float(params["max_sl"])
    raw_max_hold_minutes = params.get("max_hold_minutes")
    max_hold_minutes = int(raw_max_hold_minutes) if raw_max_hold_minutes is not None else None

    for row in scored.itertuples(index=False):
        candidates: list[tuple[str, float, float, float, float, float]] = []
        if row.long_prob >= prob_threshold and row.long_pred_pnl > 0.0:
            candidates.append(("LONG", float(row.long_score), float(row.long_prob), float(row.long_pred_pnl), float(row.long_pred_mfe), float(row.long_pred_mae)))
        if row.short_prob >= prob_threshold and row.short_pred_pnl > 0.0:
            candidates.append(("SHORT", float(row.short_score), float(row.short_prob), float(row.short_pred_pnl), float(row.short_pred_mfe), float(row.short_pred_mae)))
        if not candidates:
            continue

        side, score, prob, pred_pnl, pred_mfe, pred_mae = max(candidates, key=lambda item: item[1])
        if score < score_threshold:
            continue

        tp_dist = min(max(tp_mult * max(pred_mfe, 0.25), min_tp), max_tp)
        sl_dist = min(max(sl_mult * max(pred_mae, 0.25), min_sl), max_sl)
        rr = tp_dist / sl_dist if sl_dist > 1e-12 else 0.0
        if rr < min_rr:
            continue

        effective_exit_pos = int(row.exit_pos)
        if max_hold_minutes is not None and max_hold_minutes > 0:
            effective_exit_pos = min(effective_exit_pos, int(row.entry_pos) + int(max_hold_minutes))
        if effective_exit_pos <= int(row.entry_pos):
            continue

        simulation = simulate_bracket_trade(
            side=side,
            entry_price=float(row.entry_price),
            entry_pos=int(row.entry_pos),
            exit_pos=int(effective_exit_pos),
            tp_dist=float(tp_dist),
            sl_dist=float(sl_dist),
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
            fee_points=MES_FEE_POINTS,
        )
        records.append(
            {
                "entry_time": row.entry_time,
                "exit_time": market_index[int(simulation["exit_pos"])].isoformat(),
                "entry_pos": int(row.entry_pos),
                "exit_pos": int(simulation["exit_pos"]),
                "trade_day": row.trade_day,
                "macro_name": row.macro_name,
                "session_name": row.session_name,
                "session_window": row.session_window,
                "open_ref_name": row.open_ref_name,
                "high_breach_combo": row.high_breach_combo,
                "low_breach_combo": row.low_breach_combo,
                "combo_key": row.combo_key,
                "side": side,
                "area_key": _area_key(row.macro_name, row.session_window, row.open_ref_name, side),
                "combo_side_key": _combo_side_key(row.combo_key, side),
                "pred_prob": prob,
                "pred_pnl": pred_pnl,
                "pred_score": score,
                "pred_mfe": pred_mfe,
                "pred_mae": pred_mae,
                "tp_dist": float(tp_dist),
                "sl_dist": float(sl_dist),
                "rr": float(rr),
                "entry_price": float(row.entry_price),
                "max_hold_minutes": max_hold_minutes,
                "exit_reason": simulation["exit_reason"],
                "pnl_points_gross": float(simulation["pnl_points_gross"]),
                "pnl_points_net": float(simulation["pnl_points_net"]),
            }
        )

    return pd.DataFrame.from_records(records)


def simulate_selection(
    scored: pd.DataFrame,
    *,
    params: dict[str, float],
    market_index: pd.DatetimeIndex,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    allowlists: dict[str, Any] | None = None,
) -> pd.DataFrame:
    candidates = collect_trade_candidates(
        scored,
        params=params,
        market_index=market_index,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
    )
    candidates = filter_candidates_by_allowlists(candidates, allowlists=allowlists)
    selected = select_live_compatible_trades(candidates)
    drop_columns = [column for column in ["area_key", "combo_side_key"] if column in selected.columns]
    if drop_columns:
        selected = selected.drop(columns=drop_columns)
    return selected.reset_index(drop=True)


def tune_execution_params(
    scored: pd.DataFrame,
    *,
    market_index: pd.DatetimeIndex,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    target_trades_per_day: float = 0.0,
) -> dict[str, Any]:
    selected_scores = np.maximum(
        scored["long_score"].to_numpy(dtype=float),
        scored["short_score"].to_numpy(dtype=float),
    )
    positive_scores = selected_scores[np.isfinite(selected_scores) & (selected_scores > 0.0)]
    score_thresholds = [0.0]
    if positive_scores.size:
        score_thresholds.extend(
            [
                float(np.quantile(positive_scores, quantile))
                for quantile in (0.25, 0.5, 0.75)
            ]
        )
    score_thresholds = sorted(set(round(value, 6) for value in score_thresholds))

    best = {
        "score": -1e18,
        "params": {
            "prob_threshold": 0.55,
            "score_threshold": 0.0,
            "tp_mult": 0.60,
            "sl_mult": 0.80,
            "min_rr": 1.00,
            "min_tp": 2.0,
            "max_tp": 18.0,
            "min_sl": 1.5,
            "max_sl": 12.0,
            "max_hold_minutes": None,
        },
        "metrics": trade_metrics(pd.DataFrame()),
    }

    min_required_trades = max(12, min(40, int(len(scored) * 0.02)))
    grid = product(
        [0.45, 0.50, 0.55],
        score_thresholds,
        [0.30, 0.45, 0.60],
        [0.45, 0.60, 0.80],
        [0.75, 1.00, 1.25],
        [30, 60, 90, 180, None],
    )
    for prob_threshold, score_threshold, tp_mult, sl_mult, min_rr, max_hold_minutes in grid:
        params = {
            "prob_threshold": float(prob_threshold),
            "score_threshold": float(score_threshold),
            "tp_mult": float(tp_mult),
            "sl_mult": float(sl_mult),
            "min_rr": float(min_rr),
            "min_tp": 1.0,
            "max_tp": 18.0,
            "min_sl": 1.0,
            "max_sl": 12.0,
            "max_hold_minutes": None if max_hold_minutes is None else int(max_hold_minutes),
        }
        trades = simulate_selection(
            scored,
            params=params,
            market_index=market_index,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
        )
        metrics = trade_metrics(trades)
        if metrics["trades"] < min_required_trades:
            continue
        metric_score = _selection_metric_score(metrics, target_trades_per_day=target_trades_per_day)
        if metric_score > best["score"]:
            best = {
                "score": float(metric_score),
                "params": params,
                "metrics": metrics,
            }

    return best


def tune_viability_gate(
    fit_candidates: pd.DataFrame,
    tune_candidates: pd.DataFrame,
    *,
    target_trades_per_day: float = 0.0,
) -> dict[str, Any]:
    default_gate = {
        "area_min_trades": 0,
        "combo_min_trades": 0,
        "area_min_avg": 0.0,
        "combo_min_avg": 0.0,
        "area_min_pf": 0.0,
        "combo_min_pf": 0.0,
        "area_min_positive_day_share": 0.0,
        "combo_min_positive_day_share": 0.0,
    }
    default_allowlists = build_viability_allowlists(fit_candidates, gate_params=default_gate)
    default_metrics = trade_metrics(select_live_compatible_trades(tune_candidates))
    best = {
        "score": _selection_metric_score(default_metrics, target_trades_per_day=target_trades_per_day),
        "gate_params": default_gate,
        "metrics": default_metrics,
        "fit_allowlists": default_allowlists,
        "tune_trades": select_live_compatible_trades(tune_candidates),
    }

    min_required_trades = max(8, min(30, int(len(tune_candidates) * 0.05)))
    gate_grid: list[dict[str, Any]] = [default_gate]
    for area_min_trades in [0, 8, 12, 20]:
        for combo_min_trades in [0, 3, 5, 8]:
            if area_min_trades <= 0 and combo_min_trades <= 0:
                continue
            area_pf_options = [0.0] if area_min_trades <= 0 else [0.0, 1.05]
            combo_pf_options = [0.0] if combo_min_trades <= 0 else [0.0, 1.05]
            for area_min_pf in area_pf_options:
                for combo_min_pf in combo_pf_options:
                    gate_grid.append(
                        {
                            "area_min_trades": int(area_min_trades),
                            "combo_min_trades": int(combo_min_trades),
                            "area_min_avg": 0.0,
                            "combo_min_avg": 0.0,
                            "area_min_pf": float(area_min_pf),
                            "combo_min_pf": float(combo_min_pf),
                            "area_min_positive_day_share": 0.0,
                            "combo_min_positive_day_share": 0.0,
                        }
                    )

    seen_configs: set[str] = set()
    for gate_params in gate_grid:
        gate_key = json.dumps(gate_params, sort_keys=True)
        if gate_key in seen_configs:
            continue
        seen_configs.add(gate_key)
        allowlists = build_viability_allowlists(fit_candidates, gate_params=gate_params)
        tune_trades = select_live_compatible_trades(
            filter_candidates_by_allowlists(tune_candidates, allowlists=allowlists)
        )
        metrics = trade_metrics(tune_trades)
        if gate_params["area_min_trades"] > 0 and not allowlists["allowed_area_keys"]:
            continue
        if gate_params["combo_min_trades"] > 0 and not allowlists["allowed_combo_side_keys"]:
            continue
        if metrics["trades"] < min_required_trades:
            continue
        metric_score = _selection_metric_score(metrics, target_trades_per_day=target_trades_per_day)
        if metric_score > best["score"]:
            best = {
                "score": float(metric_score),
                "gate_params": gate_params,
                "metrics": metrics,
                "fit_allowlists": allowlists,
                "tune_trades": tune_trades,
            }

    return best


def _rank_trade_groups(
    grouped: pd.DataFrame,
    *,
    min_trades: int,
) -> pd.DataFrame:
    if grouped.empty:
        return grouped
    ranked = grouped.loc[grouped["trades"] >= int(min_trades)].copy()
    if ranked.empty:
        return ranked
    ranked["expectancy_score"] = ranked["avg_points"] * np.sqrt(ranked["trades"].clip(lower=1))
    ranked = ranked.sort_values(
        ["net_points", "expectancy_score", "avg_points", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return ranked


def summarize_best_combos(trades: pd.DataFrame, *, min_trades: int = 1) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    grouped = (
        trades.groupby(
            ["combo_key", "macro_name", "session_window", "open_ref_name", "side"],
            dropna=False,
        )["pnl_points_net"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "trades", "sum": "net_points", "mean": "avg_points"})
    )
    win_rate = (
        trades.assign(win=trades["pnl_points_net"] > 0.0)
        .groupby(["combo_key", "macro_name", "session_window", "open_ref_name", "side"], dropna=False)["win"]
        .mean()
        .reset_index(name="win_rate")
    )
    grouped = grouped.merge(win_rate, on=["combo_key", "macro_name", "session_window", "open_ref_name", "side"], how="left")
    return _rank_trade_groups(grouped, min_trades=min_trades)


def summarize_best_areas(trades: pd.DataFrame, *, min_trades: int = 1) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    grouped = (
        trades.groupby(
            ["macro_name", "session_window", "open_ref_name", "side"],
            dropna=False,
        )["pnl_points_net"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "trades", "sum": "net_points", "mean": "avg_points"})
    )
    win_rate = (
        trades.assign(win=trades["pnl_points_net"] > 0.0)
        .groupby(["macro_name", "session_window", "open_ref_name", "side"], dropna=False)["win"]
        .mean()
        .reset_index(name="win_rate")
    )
    grouped = grouped.merge(
        win_rate,
        on=["macro_name", "session_window", "open_ref_name", "side"],
        how="left",
    )
    return _rank_trade_groups(grouped, min_trades=min_trades)


def build_stability_pruned_recommendation(
    trades: pd.DataFrame,
    *,
    area_min_trades: int = 5,
    combo_min_trades: int = 3,
) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "config": {
                "area_min_trades": int(area_min_trades),
                "combo_min_trades": int(combo_min_trades),
            },
            "selection_stage": "empty",
            "area_only_summary": trade_metrics(pd.DataFrame()),
            "combo_filtered_summary": trade_metrics(pd.DataFrame()),
            "summary": trade_metrics(pd.DataFrame()),
            "recommended_areas": pd.DataFrame(),
            "recommended_combos": pd.DataFrame(),
            "recommended_trades": pd.DataFrame(),
        }

    recommended_areas = summarize_best_areas(trades, min_trades=area_min_trades)
    recommended_areas = recommended_areas.loc[
        (recommended_areas["net_points"] > 0.0) & (recommended_areas["avg_points"] > 0.0)
    ].reset_index(drop=True)

    if recommended_areas.empty:
        return {
            "config": {
                "area_min_trades": int(area_min_trades),
                "combo_min_trades": int(combo_min_trades),
            },
            "selection_stage": "empty",
            "area_only_summary": trade_metrics(pd.DataFrame()),
            "combo_filtered_summary": trade_metrics(pd.DataFrame()),
            "summary": trade_metrics(pd.DataFrame()),
            "recommended_areas": recommended_areas,
            "recommended_combos": pd.DataFrame(),
            "recommended_trades": pd.DataFrame(),
        }

    area_filtered = trades.merge(
        recommended_areas[["macro_name", "session_window", "open_ref_name", "side"]],
        on=["macro_name", "session_window", "open_ref_name", "side"],
        how="inner",
    )
    area_only_summary = trade_metrics(area_filtered)

    recommended_combos = summarize_best_combos(area_filtered, min_trades=combo_min_trades)
    recommended_combos = recommended_combos.loc[
        (recommended_combos["net_points"] > 0.0) & (recommended_combos["avg_points"] > 0.0)
    ].reset_index(drop=True)

    combo_filtered = area_filtered
    if not recommended_combos.empty:
        combo_filtered = area_filtered.merge(
            recommended_combos[["combo_key", "side"]],
            on=["combo_key", "side"],
            how="inner",
        )
    combo_filtered_summary = trade_metrics(combo_filtered)
    min_combo_survival_trades = max(
        10,
        int(math.ceil(float(area_only_summary.get("trades", 0) or 0.0) * 0.20)),
    )
    combo_has_enough_volume = int(combo_filtered_summary.get("trades", 0) or 0) >= min_combo_survival_trades

    if combo_has_enough_volume and _selection_metric_score(combo_filtered_summary) > _selection_metric_score(area_only_summary):
        selection_stage = "area_plus_combo"
        selected_trades = combo_filtered
        selected_summary = combo_filtered_summary
    else:
        selection_stage = "area_only"
        selected_trades = area_filtered
        selected_summary = area_only_summary

    return {
        "config": {
            "area_min_trades": int(area_min_trades),
            "combo_min_trades": int(combo_min_trades),
        },
        "selection_stage": selection_stage,
        "area_only_summary": area_only_summary,
        "combo_filtered_summary": combo_filtered_summary,
        "summary": selected_summary,
        "recommended_areas": recommended_areas,
        "recommended_combos": recommended_combos,
        "recommended_trades": selected_trades.reset_index(drop=True),
    }


def build_run_metadata(
    *,
    run_dir: Path,
    source_path: Path,
    combos_path: Path,
    indicator_path: Path,
    dataset_mode: str,
    feature_mode: str,
    dataset: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    fold_summaries: list[dict[str, Any]],
    oos_summary: dict[str, Any],
    monte_carlo: dict[str, Any],
    final_params: dict[str, Any],
    final_gate_params: dict[str, Any],
    final_area_preview: list[dict[str, Any]],
    final_combo_preview: list[dict[str, Any]],
    best_combo_preview: list[dict[str, Any]],
    best_area_preview: list[dict[str, Any]],
    stability_config: dict[str, Any],
    stability_selection_stage: str,
    stability_pruned_summary: dict[str, Any],
    stability_area_only_summary: dict[str, Any],
    stability_combo_filtered_summary: dict[str, Any],
    stability_area_preview: list[dict[str, Any]],
    stability_combo_preview: list[dict[str, Any]],
) -> dict[str, Any]:
    indicator_text = indicator_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "artifact_dir": str(run_dir),
        "source_file": str(source_path),
        "combos_file": str(combos_path),
        "indicator_reference_file": str(indicator_path),
        "indicator_reference_sha1": hashlib.sha1(indicator_text.encode("utf-8")).hexdigest(),
        "dataset_mode": str(dataset_mode or "snapshot"),
        "feature_mode": str(feature_mode or "curated"),
        "dataset_rows": int(len(dataset)),
        "dataset_range": {
            "start": str(dataset["entry_time"].min()) if not dataset.empty else None,
            "end": str(dataset["entry_time"].max()) if not dataset.empty else None,
        },
        "feature_columns": {
            "categorical": list(categorical_columns),
            "numeric": list(numeric_columns),
        },
        "fold_summaries": fold_summaries,
        "oos_summary": oos_summary,
        "monte_carlo": monte_carlo,
        "final_params": final_params,
        "final_gate_params": final_gate_params,
        "final_area_preview": final_area_preview,
        "final_combo_preview": final_combo_preview,
        "best_combo_preview": best_combo_preview,
        "best_area_preview": best_area_preview,
        "stability_config": stability_config,
        "stability_selection_stage": stability_selection_stage,
        "stability_pruned_summary": stability_pruned_summary,
        "stability_area_only_summary": stability_area_only_summary,
        "stability_combo_filtered_summary": stability_combo_filtered_summary,
        "stability_area_preview": stability_area_preview,
        "stability_combo_preview": stability_combo_preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Topstep combo / indicator reference MLPhysics experiment with OOS and Monte Carlo validation."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--combos-file", default="topstep_combos_master.csv")
    parser.add_argument("--indicator-file", default="weswesindicator.txt")
    parser.add_argument("--artifact-dir", default="artifacts/ml_physics_topstep_combo")
    parser.add_argument("--dataset-mode", default="snapshot")
    parser.add_argument("--feature-mode", default="curated")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--symbol-mode", default="auto")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--min-train-days", type=int, default=220)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--max-folds", type=int, default=3)
    parser.add_argument("--monte-carlo-sims", type=int, default=2000)
    parser.add_argument("--stability-area-min-trades", type=int, default=5)
    parser.add_argument("--stability-combo-min-trades", type=int, default=3)
    parser.add_argument("--target-trades-per-day", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_path = _resolve_file(args.source)
    combos_path = _resolve_file(args.combos_file)
    indicator_path = _resolve_file(args.indicator_file)
    artifact_root = _resolve_file(args.artifact_dir) if Path(args.artifact_dir).exists() else (ROOT / args.artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_dir = artifact_root / f"run_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    market_df = load_market_data(
        source_path,
        start=str(args.start or "").strip() or None,
        end=str(args.end or "").strip() or None,
        symbol_mode=str(args.symbol_mode or "auto"),
        symbol_method=str(args.symbol_method or "volume"),
    )
    combo_catalog = load_combo_catalog(combos_path)
    dataset = build_dataset(
        market_df,
        combo_catalog,
        dataset_mode=str(args.dataset_mode or "snapshot"),
    )
    if dataset.empty:
        raise SystemExit("Topstep combo dataset is empty. No trainable snapshots were produced.")

    feature_columns, categorical_columns, numeric_columns = resolve_feature_columns(
        dataset,
        feature_mode=str(args.feature_mode or "curated"),
    )
    model_feature_columns = list(categorical_columns) + list(numeric_columns)
    if not model_feature_columns:
        raise SystemExit("No model feature columns were produced.")

    market_index = pd.DatetimeIndex(market_df.index)
    high_arr = market_df["high"].to_numpy(dtype=float)
    low_arr = market_df["low"].to_numpy(dtype=float)
    close_arr = market_df["close"].to_numpy(dtype=float)

    ordered_days = sorted(dataset["trade_day"].astype(str).unique().tolist())
    folds = make_walkforward_splits(
        ordered_days,
        min_train_days=int(args.min_train_days),
        test_days=int(args.test_days),
        max_folds=int(args.max_folds),
    )
    if not folds:
        raise SystemExit("Unable to create walkforward folds from the dataset.")

    all_oos_trades: list[pd.DataFrame] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold_idx, fold in enumerate(folds, start=1):
        train_days = fold["train_days"]
        test_days = fold["test_days"]
        fit_days, tune_days = split_fit_tune_days(train_days)

        fit_df = dataset.loc[dataset["trade_day"].isin(fit_days)].reset_index(drop=True)
        tune_df = dataset.loc[dataset["trade_day"].isin(tune_days)].reset_index(drop=True)
        train_df = dataset.loc[dataset["trade_day"].isin(train_days)].reset_index(drop=True)
        test_df = dataset.loc[dataset["trade_day"].isin(test_days)].reset_index(drop=True)
        if fit_df.empty or tune_df.empty or test_df.empty:
            continue

        provisional_long = fit_side_bundle(
            fit_df,
            side="long",
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            seed=int(args.seed) + fold_idx * 100,
        )
        provisional_short = fit_side_bundle(
            fit_df,
            side="short",
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            seed=int(args.seed) + fold_idx * 1000,
        )
        tune_scored = build_prediction_frame(
            tune_df,
            long_bundle=provisional_long,
            short_bundle=provisional_short,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )
        fit_scored = build_prediction_frame(
            fit_df,
            long_bundle=provisional_long,
            short_bundle=provisional_short,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )
        tuned = tune_execution_params(
            tune_scored,
            market_index=market_index,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
            target_trades_per_day=float(args.target_trades_per_day or 0.0),
        )
        fit_candidates = collect_trade_candidates(
            fit_scored,
            params=tuned["params"],
            market_index=market_index,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
        )
        tune_candidates = collect_trade_candidates(
            tune_scored,
            params=tuned["params"],
            market_index=market_index,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
        )
        gate_tuned = tune_viability_gate(
            fit_candidates,
            tune_candidates,
            target_trades_per_day=float(args.target_trades_per_day or 0.0),
        )

        final_long = fit_side_bundle(
            train_df,
            side="long",
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            seed=int(args.seed) + fold_idx * 100,
        )
        final_short = fit_side_bundle(
            train_df,
            side="short",
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            seed=int(args.seed) + fold_idx * 1000,
        )
        test_scored = build_prediction_frame(
            test_df,
            long_bundle=final_long,
            short_bundle=final_short,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )
        train_scored = build_prediction_frame(
            train_df,
            long_bundle=final_long,
            short_bundle=final_short,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )
        train_candidates = collect_trade_candidates(
            train_scored,
            params=tuned["params"],
            market_index=market_index,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
        )
        final_allowlists = build_viability_allowlists(
            train_candidates,
            gate_params=gate_tuned["gate_params"],
        )
        test_candidates = collect_trade_candidates(
            test_scored,
            params=tuned["params"],
            market_index=market_index,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
        )
        fold_trades = select_live_compatible_trades(
            filter_candidates_by_allowlists(test_candidates, allowlists=final_allowlists)
        )
        if not fold_trades.empty:
            fold_trades["fold_id"] = int(fold_idx)
            all_oos_trades.append(fold_trades)
        fold_summaries.append(
            {
                "fold_id": int(fold_idx),
                "fit_start": fit_days[0],
                "fit_end": fit_days[-1],
                "tune_start": tune_days[0],
                "tune_end": tune_days[-1],
                "test_start": test_days[0],
                "test_end": test_days[-1],
                "tuned_params": tuned["params"],
                "tune_metrics": tuned["metrics"],
                "tuned_gate_params": gate_tuned["gate_params"],
                "tune_gate_metrics": gate_tuned["metrics"],
                "allowed_area_count": int(len(final_allowlists["allowed_area_keys"])),
                "allowed_combo_count": int(len(final_allowlists["allowed_combo_side_keys"])),
                "allowed_area_preview": final_allowlists["area_preview"][:10],
                "allowed_combo_preview": final_allowlists["combo_preview"][:10],
                "test_metrics": trade_metrics(fold_trades),
            }
        )

    oos_trades = pd.concat(all_oos_trades, ignore_index=True) if all_oos_trades else pd.DataFrame()
    oos_summary = trade_metrics(oos_trades)
    monte_carlo = bootstrap_monte_carlo(
        oos_trades,
        simulations=int(args.monte_carlo_sims),
        seed=int(args.seed) + 90_000,
    )
    best_combos = summarize_best_combos(oos_trades)
    best_combos_min10 = summarize_best_combos(oos_trades, min_trades=10)
    best_areas = summarize_best_areas(oos_trades, min_trades=20)
    stability_recommendation = build_stability_pruned_recommendation(
        oos_trades,
        area_min_trades=int(args.stability_area_min_trades),
        combo_min_trades=int(args.stability_combo_min_trades),
    )

    final_fit_days, final_tune_days = split_fit_tune_days(ordered_days)
    final_fit_df = dataset.loc[dataset["trade_day"].isin(final_fit_days)].reset_index(drop=True)
    final_tune_df = dataset.loc[dataset["trade_day"].isin(final_tune_days)].reset_index(drop=True)
    final_provisional_long = fit_side_bundle(
        final_fit_df,
        side="long",
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        seed=int(args.seed) + 501,
    )
    final_provisional_short = fit_side_bundle(
        final_fit_df,
        side="short",
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        seed=int(args.seed) + 1501,
    )
    final_tune_scored = build_prediction_frame(
        final_tune_df,
        long_bundle=final_provisional_long,
        short_bundle=final_provisional_short,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    final_fit_scored = build_prediction_frame(
        final_fit_df,
        long_bundle=final_provisional_long,
        short_bundle=final_provisional_short,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    final_tuned = tune_execution_params(
        final_tune_scored,
        market_index=market_index,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
        target_trades_per_day=float(args.target_trades_per_day or 0.0),
    )
    final_fit_candidates = collect_trade_candidates(
        final_fit_scored,
        params=final_tuned["params"],
        market_index=market_index,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
    )
    final_tune_candidates = collect_trade_candidates(
        final_tune_scored,
        params=final_tuned["params"],
        market_index=market_index,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
    )
    final_gate_tuned = tune_viability_gate(
        final_fit_candidates,
        final_tune_candidates,
        target_trades_per_day=float(args.target_trades_per_day or 0.0),
    )

    final_long_bundle = fit_side_bundle(
        dataset,
        side="long",
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        seed=int(args.seed) + 777,
    )
    final_short_bundle = fit_side_bundle(
        dataset,
        side="short",
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        seed=int(args.seed) + 1777,
    )

    model_bundle = {
        "long": final_long_bundle,
        "short": final_short_bundle,
        "dataset_mode": str(args.dataset_mode or "snapshot"),
        "feature_mode": str(args.feature_mode or "curated"),
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "model_feature_columns": model_feature_columns,
        "tuned_params": final_tuned["params"],
        "gate_params": final_gate_tuned["gate_params"],
    }
    joblib.dump(model_bundle, run_dir / "topstep_combo_mlphysics_bundle.joblib")

    final_dataset_scored = build_prediction_frame(
        dataset,
        long_bundle=final_long_bundle,
        short_bundle=final_short_bundle,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
    )
    final_dataset_candidates = collect_trade_candidates(
        final_dataset_scored,
        params=final_tuned["params"],
        market_index=market_index,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
    )
    final_live_allowlists = build_viability_allowlists(
        final_dataset_candidates,
        gate_params=final_gate_tuned["gate_params"],
    )

    dataset_summary = {
        "rows": int(len(dataset)),
        "trade_days": int(dataset["trade_day"].nunique()),
        "combo_keys": int(dataset["combo_key"].nunique()),
        "macro_names": int(dataset["macro_name"].nunique()),
        "dataset_mode": str(args.dataset_mode or "snapshot"),
        "feature_mode": str(args.feature_mode or "curated"),
        "open_refs": sorted(dataset["open_ref_name"].astype(str).unique().tolist()),
        "session_windows": sorted(dataset["session_window"].astype(str).unique().tolist()),
    }
    metadata = build_run_metadata(
        run_dir=run_dir,
        source_path=source_path,
        combos_path=combos_path,
        indicator_path=indicator_path,
        dataset_mode=str(args.dataset_mode or "snapshot"),
        feature_mode=str(args.feature_mode or "curated"),
        dataset=dataset,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        fold_summaries=fold_summaries,
        oos_summary=oos_summary,
        monte_carlo=monte_carlo,
        final_params=final_tuned["params"],
        final_gate_params=final_gate_tuned["gate_params"],
        final_area_preview=final_live_allowlists["area_preview"][:15],
        final_combo_preview=final_live_allowlists["combo_preview"][:15],
        best_combo_preview=best_combos_min10.head(15).to_dict(orient="records"),
        best_area_preview=best_areas.head(15).to_dict(orient="records"),
        stability_config=stability_recommendation["config"],
        stability_selection_stage=stability_recommendation["selection_stage"],
        stability_pruned_summary=stability_recommendation["summary"],
        stability_area_only_summary=stability_recommendation["area_only_summary"],
        stability_combo_filtered_summary=stability_recommendation["combo_filtered_summary"],
        stability_area_preview=stability_recommendation["recommended_areas"].head(15).to_dict(orient="records"),
        stability_combo_preview=stability_recommendation["recommended_combos"].head(15).to_dict(orient="records"),
    )
    metadata["dataset_summary"] = dataset_summary

    (run_dir / "dataset_summary.json").write_text(json.dumps(dataset_summary, indent=2, default=_json_default), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    (run_dir / "oos_summary.json").write_text(
        json.dumps(
            {
                "folds": fold_summaries,
                "oos_summary": oos_summary,
                "monte_carlo": monte_carlo,
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    if not oos_trades.empty:
        oos_trades.to_csv(run_dir / "oos_trades.csv", index=False)
    if not best_combos.empty:
        best_combos.to_csv(run_dir / "best_combos.csv", index=False)
    if not best_combos_min10.empty:
        best_combos_min10.to_csv(run_dir / "best_combos_min10.csv", index=False)
    if not best_areas.empty:
        best_areas.to_csv(run_dir / "best_areas.csv", index=False)
    if not stability_recommendation["recommended_areas"].empty:
        stability_recommendation["recommended_areas"].to_csv(run_dir / "stability_recommended_areas.csv", index=False)
    if not stability_recommendation["recommended_combos"].empty:
        stability_recommendation["recommended_combos"].to_csv(run_dir / "stability_recommended_combos.csv", index=False)
    if not stability_recommendation["recommended_trades"].empty:
        stability_recommendation["recommended_trades"].to_csv(run_dir / "stability_pruned_oos_trades.csv", index=False)
    (run_dir / "stability_pruned_oos_summary.json").write_text(
        json.dumps(
            {
                "config": stability_recommendation["config"],
                "selection_stage": stability_recommendation["selection_stage"],
                "area_only_summary": stability_recommendation["area_only_summary"],
                "combo_filtered_summary": stability_recommendation["combo_filtered_summary"],
                "summary": stability_recommendation["summary"],
            },
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    latest_path = artifact_root / "latest.json"
    latest_path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")

    print(f"Saved Topstep combo MLPhysics artifact to {run_dir}")
    print(f"Dataset rows: {len(dataset)}")
    print(f"OOS trades: {oos_summary['trades']}")
    print(f"OOS net points: {oos_summary['net_points']:.2f}")
    print(f"OOS profit factor: {oos_summary['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
