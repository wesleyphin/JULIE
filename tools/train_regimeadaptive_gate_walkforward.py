import argparse
import copy
import datetime as dt
import json
import math
import platform
import shutil
import sys
from pathlib import Path

if sys.platform.startswith("win"):
    import os

    _platform_machine = str(os.environ.get("PROCESSOR_ARCHITECTURE", "") or "").strip()
    if _platform_machine:
        platform.machine = lambda: _platform_machine  # type: ignore[assignment]

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from regimeadaptive_artifact import load_regimeadaptive_artifact
from regimeadaptive_gate import GATE_FEATURE_COLUMNS, build_gate_feature_frame_for_positions
from tools.backtest_regimeadaptive_robust import (
    _artifact_rule_order,
    _build_artifact_lookups,
    _build_artifact_rule_lookup,
    _build_multirule_signal_arrays,
    _build_rule_strength_arrays,
    _build_signal_arrays,
    _parse_datetime,
    _robustness_metrics,
    _rolling_extrema_cache,
    _simulate,
    _write_converted_csv,
)
from tools.regimeadaptive_filterless_runner import (
    NY_TZ,
    _atr_array,
    _build_combo_arrays,
    _build_holiday_mask,
    _load_bars,
    _rolling_cache,
)
from tools.train_regimeadaptive_v2 import _json_safe


def _resolve_path(path_text: str, default_relative: str) -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _build_classifier(model_name: str, random_state: int):
    name = str(model_name or "hgb").strip().lower()
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=30,
            random_state=random_state,
            n_jobs=-1,
        )
    return HistGradientBoostingClassifier(
        max_depth=5,
        max_iter=300,
        learning_rate=0.05,
        min_samples_leaf=60,
        l2_regularization=0.5,
        random_state=random_state,
    )


def _probability_threshold_grid(probs: np.ndarray) -> list[float]:
    if probs.size == 0:
        return []
    quantiles = np.linspace(0.15, 0.90, 9)
    values = [float(np.quantile(probs, q)) for q in quantiles]
    values.extend([0.50, 0.55, 0.60, 0.65, 0.70])
    return sorted({round(float(value), 6) for value in values if np.isfinite(value)})


def _parse_threshold_candidates(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw or "").split(","):
        value = str(item or "").strip()
        if not value:
            continue
        parsed = float(value)
        if math.isfinite(parsed):
            out.append(round(float(parsed), 6))
    return sorted({float(value) for value in out})


def _unique_years(index: pd.DatetimeIndex) -> list[int]:
    return sorted({int(ts.year) for ts in index})


def _period_bounds(index: pd.DatetimeIndex, years: list[int]) -> tuple[pd.Timestamp, pd.Timestamp]:
    mask = np.isin(index.year.to_numpy(dtype=np.int16), np.asarray(years, dtype=np.int16))
    if not np.any(mask):
        raise ValueError(f"No timestamps found for years: {years}")
    period_index = index[mask]
    return period_index[0], period_index[-1]


def _build_walkforward_folds(
    index: pd.DatetimeIndex,
    min_train_years: int,
    valid_years: int,
    test_years: int,
    first_test_year: int | None,
) -> list[dict]:
    years = _unique_years(index)
    if len(years) < (min_train_years + valid_years + test_years):
        raise ValueError("Not enough years for the requested walk-forward configuration.")

    folds: list[dict] = []
    start_idx = min_train_years + valid_years
    for test_start_idx in range(start_idx, len(years) - test_years + 1):
        train_years = years[: test_start_idx - valid_years]
        valid_years_list = years[test_start_idx - valid_years : test_start_idx]
        test_years_list = years[test_start_idx : test_start_idx + test_years]
        if first_test_year is not None and test_years_list[0] < int(first_test_year):
            continue
        valid_start, valid_end = _period_bounds(index, valid_years_list)
        test_start, test_end = _period_bounds(index, test_years_list)
        folds.append(
            {
                "name": f"test_{test_years_list[0]}_{test_years_list[-1]}",
                "train_years": list(train_years),
                "valid_years": list(valid_years_list),
                "test_years": list(test_years_list),
                "valid_start": valid_start,
                "valid_end": valid_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return folds


def _summarize_period_trade_log(
    trade_log: list[dict],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    args,
) -> dict:
    if not trade_log:
        return {
            "score": float("-inf"),
            "equity": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "avg_trade_net": 0.0,
            "profit_factor": 0.0,
            "daily_sharpe": 0.0,
            "negative_years": 0,
            "worst_3y_pnl": 0.0,
            "worst_5y_pnl": 0.0,
            "yearly_pnl_std": 0.0,
            "max_drawdown": 0.0,
        }

    trades = pd.DataFrame(trade_log)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(NY_TZ)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert(NY_TZ)
    trades["pnl_net"] = trades["pnl_net"].astype(float)

    total_net = float(trades["pnl_net"].sum())
    wins = int((trades["pnl_net"] > 0).sum())
    losses = int((trades["pnl_net"] <= 0).sum())
    trade_count = int(len(trades))
    gross_profit = float(trades.loc[trades["pnl_net"] > 0, "pnl_net"].sum())
    gross_loss = float(trades.loc[trades["pnl_net"] < 0, "pnl_net"].sum())
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0
    robustness = _robustness_metrics(trade_log, start_time, end_time)
    equity_curve = trades.sort_values("exit_time")["pnl_net"].cumsum()
    running_peak = equity_curve.cummax()
    max_drawdown = float((running_peak - equity_curve).max()) if not equity_curve.empty else 0.0

    score = float(total_net)
    score += float(args.robust_sharpe_weight) * float(robustness["daily_sharpe"])
    score += float(args.trade_count_weight) * float(trade_count)
    score -= float(args.negative_year_penalty) * float(robustness["negative_years"])
    score += float(args.worst_3y_weight) * float(robustness["worst_3y_pnl"])
    score += float(args.worst_5y_weight) * float(robustness["worst_5y_pnl"])
    score -= float(args.max_drawdown_penalty) * float(max_drawdown)
    score -= float(args.yearly_std_penalty) * float(robustness["yearly_pnl_std"])

    return {
        "score": float(score),
        "equity": total_net,
        "trades": trade_count,
        "wins": wins,
        "losses": losses,
        "winrate": float((wins / trade_count) * 100.0) if trade_count else 0.0,
        "avg_trade_net": float(total_net / trade_count) if trade_count else 0.0,
        "profit_factor": None if not math.isfinite(profit_factor) else float(profit_factor),
        "daily_sharpe": float(robustness["daily_sharpe"]),
        "negative_years": int(robustness["negative_years"]),
        "worst_3y_pnl": float(robustness["worst_3y_pnl"]),
        "worst_5y_pnl": float(robustness["worst_5y_pnl"]),
        "yearly_pnl_std": float(robustness["yearly_pnl_std"]),
        "max_drawdown": float(max_drawdown),
    }


def _run_period_simulation(
    df: pd.DataFrame,
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    holiday_mask: np.ndarray,
    signal_side: np.ndarray,
    signal_early_exit: np.ndarray,
    signal_sl: np.ndarray,
    signal_tp: np.ndarray,
    original_side: np.ndarray,
    contracts: int,
    point_value: float,
    fee_per_contract_rt: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    *,
    selected_rule_index: np.ndarray | None,
    rule_order: list[str] | None,
    gate_prob: np.ndarray | None,
    gate_threshold: float | None,
    hours: np.ndarray,
    minutes: np.ndarray,
    test_positions: np.ndarray,
) -> dict:
    period_mask = (df.index >= start_time) & (df.index <= end_time)
    if not bool(np.any(period_mask)):
        return {
            "equity": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "avg_trade_net": 0.0,
            "avg_contracts": 0.0,
            "min_contracts_used": 0,
            "max_contracts_used": 0,
            "contract_usage": {},
            "max_drawdown": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "daily_sharpe": 0.0,
            "negative_years": 0,
            "worst_year_pnl": 0.0,
            "best_year_pnl": 0.0,
            "yearly_pnl_std": 0.0,
            "worst_3y_pnl": 0.0,
            "worst_5y_pnl": 0.0,
            "yearly_pnl": {},
            "exit_reasons": {},
            "sessions": {},
            "trade_log": [],
        }

    df_period = df.loc[period_mask].copy()
    combo_ids_period = combo_ids[period_mask]
    session_codes_period = session_codes[period_mask]
    holiday_mask_period = holiday_mask[period_mask]
    signal_side_period = signal_side[period_mask]
    signal_early_exit_period = signal_early_exit[period_mask]
    signal_sl_period = signal_sl[period_mask]
    signal_tp_period = signal_tp[period_mask]
    original_side_period = original_side[period_mask]
    selected_rule_index_period = selected_rule_index[period_mask] if isinstance(selected_rule_index, np.ndarray) else None
    gate_prob_period = gate_prob[period_mask] if isinstance(gate_prob, np.ndarray) else None
    hours_period = hours[period_mask]
    minutes_period = minutes[period_mask]
    period_positions = np.arange(len(df_period), dtype=np.int32)

    return _simulate(
        df_period,
        pd.Timestamp(df_period.index[0]),
        pd.Timestamp(df_period.index[-1]),
        combo_ids_period,
        session_codes_period,
        holiday_mask_period,
        signal_side_period.copy(),
        signal_early_exit_period.copy(),
        signal_sl_period.copy(),
        signal_tp_period.copy(),
        original_side_period.copy(),
        int(contracts),
        point_value,
        fee_per_contract_rt,
        selected_rule_index=selected_rule_index_period,
        rule_order=rule_order,
        gate_prob=gate_prob_period,
        gate_threshold=gate_threshold,
        hours=hours_period,
        minutes=minutes_period,
        test_positions=period_positions,
    )


def _train_weights(y_train: np.ndarray) -> np.ndarray:
    positive = max(1, int((y_train == 1).sum()))
    negative = max(1, int((y_train == 0).sum()))
    return np.where(
        y_train == 1,
        len(y_train) / (2.0 * positive),
        len(y_train) / (2.0 * negative),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research-only walk-forward RegimeAdaptive gate trainer with validation-only threshold selection."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--artifact", default="artifacts/regimeadaptive_v14_dense_balanced/latest.json")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--model", default="hgb")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--valid-years", type=int, default=1)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--first-test-year", type=int, default=2018)
    parser.add_argument("--final-holdout-years", type=int, default=0)
    parser.add_argument("--threshold-candidates", default="")
    parser.add_argument("--min-valid-trades", type=int, default=180)
    parser.add_argument("--min-valid-trade-ratio", type=float, default=0.50)
    parser.add_argument("--require-positive-valid", action="store_true")
    parser.add_argument("--require-valid-pf-above-one", action="store_true")
    parser.add_argument("--robust-sharpe-weight", type=float, default=1000.0)
    parser.add_argument("--negative-year-penalty", type=float, default=300.0)
    parser.add_argument("--worst-3y-weight", type=float, default=0.15)
    parser.add_argument("--worst-5y-weight", type=float, default=0.35)
    parser.add_argument("--max-drawdown-penalty", type=float, default=0.05)
    parser.add_argument("--yearly-std-penalty", type=float, default=0.02)
    parser.add_argument("--trade-count-weight", type=float, default=0.5)
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--write-latest", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Source parquet not found: {source}")

    artifact = load_regimeadaptive_artifact(str(args.artifact))
    if artifact is None:
        raise SystemExit(f"Artifact could not be loaded: {args.artifact}")

    start_time = _parse_datetime(args.start, is_end=False)
    end_time = _parse_datetime(args.end, is_end=True)
    df, symbol_label = _load_bars(source, str(args.symbol_mode), str(args.symbol_method))
    df = df[(df.index >= start_time) & (df.index <= end_time)].copy()
    if df.empty:
        raise SystemExit("No data in requested range.")

    combo_ids, session_codes = _build_combo_arrays(df.index)
    holiday_mask = _build_holiday_mask(df.index, session_codes)
    all_years = _unique_years(df.index)
    holdout_years: list[int] = []
    if int(args.final_holdout_years) > 0:
        if int(args.final_holdout_years) >= len(all_years):
            raise SystemExit("final-holdout-years leaves no data for walk-forward selection.")
        holdout_years = list(all_years[-int(args.final_holdout_years) :])
    selection_index = df.index[
        ~np.isin(
            df.index.year.to_numpy(dtype=np.int16),
            np.asarray(holdout_years, dtype=np.int16),
        )
    ]
    folds = _build_walkforward_folds(
        selection_index,
        int(args.min_train_years),
        int(args.valid_years),
        int(args.test_years),
        int(args.first_test_year) if args.first_test_year else None,
    )
    if not folds:
        raise SystemExit("No walk-forward folds were created.")
    explicit_thresholds = _parse_threshold_candidates(str(args.threshold_candidates))

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    open_arr = df["open"].to_numpy(dtype=np.float64)
    hours = np.fromiter((ts.hour for ts in df.index), dtype=np.int8, count=len(df.index))
    minutes = np.fromiter((ts.minute for ts in df.index), dtype=np.int8, count=len(df.index))
    test_positions = np.arange(len(df.index), dtype=np.int32)

    rule_order = _artifact_rule_order(artifact)
    multirule_enabled = bool(getattr(artifact, "rule_catalog", {}) or {})
    if multirule_enabled:
        rule_payloads = {str(rule_id): dict(artifact.rule_catalog[str(rule_id)]) for rule_id in rule_order}
    else:
        base_rule = artifact.base_rule or {}
        rule_order = ["__base__"]
        rule_payloads = {"__base__": dict(base_rule)}

    sma_windows = sorted(
        {int(rule_payload.get("sma_fast", 20) or 20) for rule_payload in rule_payloads.values()}
        | {int(rule_payload.get("sma_slow", 200) or 200) for rule_payload in rule_payloads.values()}
    )
    atr_periods = sorted({int(rule_payload.get("atr_period", 20) or 20) for rule_payload in rule_payloads.values()})
    pattern_lookbacks = sorted(
        {
            max(1, int(rule_payload.get("pattern_lookback", 8) or 8))
            for rule_payload in rule_payloads.values()
            if str(rule_payload.get("rule_type", "pullback") or "pullback").strip().lower() in {"continuation", "breakout"}
        }
    )
    rolling = _rolling_cache(close, sma_windows)
    atr_cache = {period: _atr_array(high, low, close, int(period)) for period in atr_periods}
    rolling_high_cache = _rolling_extrema_cache(high, pattern_lookbacks, "max") if pattern_lookbacks else {}
    rolling_low_cache = _rolling_extrema_cache(low, pattern_lookbacks, "min") if pattern_lookbacks else {}

    policy_lookup, early_exit_lookup, long_sl_lookup, long_tp_lookup, short_sl_lookup, short_tp_lookup = _build_artifact_lookups(artifact)
    if multirule_enabled:
        rule_lookup = _build_artifact_rule_lookup(artifact, rule_order)
        long_strength_matrix = np.zeros((len(rule_order), len(close)), dtype=np.float32)
        short_strength_matrix = np.zeros((len(rule_order), len(close)), dtype=np.float32)
        for idx, rule_id in enumerate(rule_order):
            long_strength_matrix[idx], short_strength_matrix[idx] = _build_rule_strength_arrays(
                session_codes,
                close,
                high,
                low,
                rolling,
                atr_cache,
                rolling_high_cache,
                rolling_low_cache,
                rule_payloads[str(rule_id)],
            )
        signal_side, signal_early_exit, signal_sl, signal_tp, original_side, selected_rule_index = _build_multirule_signal_arrays(
            combo_ids,
            session_codes,
            rule_lookup,
            policy_lookup,
            early_exit_lookup,
            long_sl_lookup,
            long_tp_lookup,
            short_sl_lookup,
            short_tp_lookup,
            long_strength_matrix,
            short_strength_matrix,
        )
    else:
        base_rule = artifact.base_rule or {}
        base_long_strength, base_short_strength = _build_rule_strength_arrays(
            session_codes,
            close,
            high,
            low,
            rolling,
            atr_cache,
            rolling_high_cache,
            rolling_low_cache,
            base_rule,
        )
        signal_side, signal_early_exit, signal_sl, signal_tp, original_side = _build_signal_arrays(
            combo_ids,
            base_long_strength,
            base_short_strength,
            policy_lookup,
            early_exit_lookup,
            long_sl_lookup,
            long_tp_lookup,
            short_sl_lookup,
            short_tp_lookup,
        )
        selected_rule_index = np.zeros(len(close), dtype=np.int16)
        long_strength_matrix = np.expand_dims(base_long_strength, axis=0)
        short_strength_matrix = np.expand_dims(base_short_strength, axis=0)

    risk_cfg = CONFIG.get("RISK", {}) or {}
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
    fee_per_side = float(risk_cfg.get("FEES_PER_SIDE", 0.37) or 0.37)
    fee_per_contract_rt = fee_per_side * 2.0

    full_baseline = _run_period_simulation(
        df,
        combo_ids,
        session_codes,
        holiday_mask,
        signal_side,
        signal_early_exit,
        signal_sl,
        signal_tp,
        original_side,
        int(args.contracts),
        point_value,
        fee_per_contract_rt,
        start_time,
        end_time,
        selected_rule_index=selected_rule_index,
        rule_order=rule_order if multirule_enabled else None,
        gate_prob=None,
        gate_threshold=None,
        hours=hours,
        minutes=minutes,
        test_positions=test_positions,
    )
    baseline_trade_log = full_baseline.get("trade_log", []) or []
    if not baseline_trade_log:
        raise SystemExit("Baseline artifact produced no trades.")

    signal_positions = np.flatnonzero(signal_side != 0)
    signal_features = build_gate_feature_frame_for_positions(
        pd.DatetimeIndex(df.index),
        open_arr,
        high,
        low,
        close,
        combo_ids,
        signal_side,
        original_side,
        selected_rule_index,
        signal_positions,
        rule_order,
        rule_payloads,
        rolling,
        atr_cache,
        long_strength_matrix,
        short_strength_matrix,
    ).reindex(columns=GATE_FEATURE_COLUMNS, fill_value=0.0)

    signal_pos_to_feature_idx = {int(pos): idx for idx, pos in enumerate(signal_positions.tolist())}
    index_lookup = {pd.Timestamp(ts).isoformat(): pos for pos, ts in enumerate(df.index)}
    executed_positions: list[int] = []
    executed_labels: list[int] = []
    executed_times: list[str] = []
    seen_trade_ids: set[tuple[str, str, str]] = set()
    for trade in baseline_trade_log:
        signal_time_text = str(trade.get("signal_time", "") or trade.get("entry_time", "") or "")
        side_text = str(trade.get("side", "") or "")
        combo_key = str(trade.get("combo_key", "") or "")
        unique_key = (signal_time_text, side_text, combo_key)
        if unique_key in seen_trade_ids:
            continue
        seen_trade_ids.add(unique_key)
        ts = pd.Timestamp(signal_time_text)
        pos = index_lookup.get(ts.isoformat())
        if pos is None or int(pos) not in signal_pos_to_feature_idx:
            continue
        executed_positions.append(int(pos))
        executed_labels.append(1 if float(trade.get("pnl_net", 0.0) or 0.0) > 0.0 else 0)
        executed_times.append(signal_time_text)

    if not executed_positions:
        raise SystemExit("Could not align executed trades back to signal positions.")

    executed_feature_indices = np.asarray([signal_pos_to_feature_idx[int(pos)] for pos in executed_positions], dtype=np.int32)
    executed_features = signal_features.iloc[executed_feature_indices].reset_index(drop=True)
    executed_years = pd.to_datetime(pd.Series(executed_times), utc=True).dt.tz_convert(NY_TZ).dt.year.to_numpy(dtype=np.int16)
    signal_years = df.index.year.to_numpy(dtype=np.int16)[signal_positions]
    y = np.asarray(executed_labels, dtype=np.int8)

    fold_rows: list[dict] = []
    oos_trade_log: list[dict] = []
    baseline_oos_trade_log: list[dict] = []
    chosen_thresholds: list[float] = []

    for fold in folds:
        train_years = np.asarray(fold["train_years"], dtype=np.int16)
        valid_years_arr = np.asarray(fold["valid_years"], dtype=np.int16)
        test_years_arr = np.asarray(fold["test_years"], dtype=np.int16)

        train_mask = np.isin(executed_years, train_years)
        if int(train_mask.sum()) < 100:
            continue

        model = _build_classifier(str(args.model), int(args.random_state))
        X_train = executed_features.loc[train_mask].to_numpy(dtype=np.float32)
        y_train = y[train_mask]
        model.fit(X_train, y_train, sample_weight=_train_weights(y_train))

        signal_probs = np.asarray(model.predict_proba(signal_features.to_numpy(dtype=np.float32))[:, 1], dtype=np.float64)
        valid_mask = np.isin(signal_years, valid_years_arr)
        valid_signal_positions = signal_positions[valid_mask]
        valid_signal_probs = signal_probs[valid_mask]
        threshold_grid = explicit_thresholds or _probability_threshold_grid(valid_signal_probs)
        if not threshold_grid:
            continue

        valid_baseline = _run_period_simulation(
            df,
            combo_ids,
            session_codes,
            holiday_mask,
            signal_side,
            signal_early_exit,
            signal_sl,
            signal_tp,
            original_side,
            int(args.contracts),
            point_value,
            fee_per_contract_rt,
            fold["valid_start"],
            fold["valid_end"],
            selected_rule_index=selected_rule_index,
            rule_order=rule_order if multirule_enabled else None,
            gate_prob=None,
            gate_threshold=None,
            hours=hours,
            minutes=minutes,
            test_positions=test_positions,
        )
        valid_baseline_summary = _summarize_period_trade_log(
            valid_baseline.get("trade_log", []) or [],
            fold["valid_start"],
            fold["valid_end"],
            args,
        )
        min_required_valid_trades = max(
            int(args.min_valid_trades),
            int(math.floor(float(valid_baseline_summary["trades"]) * float(args.min_valid_trade_ratio))),
        )

        best_valid_summary = None
        best_threshold = None
        for threshold in threshold_grid:
            gated_signal_side = signal_side.copy()
            gated_signal_early_exit = signal_early_exit.copy()
            blocked_positions = valid_signal_positions[valid_signal_probs < float(threshold)]
            if blocked_positions.size:
                gated_signal_side[blocked_positions] = 0
                gated_signal_early_exit[blocked_positions] = -1
            gate_prob_full = np.full(len(close), np.nan, dtype=np.float64)
            gate_prob_full[signal_positions] = signal_probs
            valid_result = _run_period_simulation(
                df,
                combo_ids,
                session_codes,
                holiday_mask,
                gated_signal_side,
                gated_signal_early_exit,
                signal_sl,
                signal_tp,
                original_side,
                int(args.contracts),
                point_value,
                fee_per_contract_rt,
                fold["valid_start"],
                fold["valid_end"],
                selected_rule_index=selected_rule_index,
                rule_order=rule_order if multirule_enabled else None,
                gate_prob=gate_prob_full,
                gate_threshold=float(threshold),
                hours=hours,
                minutes=minutes,
                test_positions=test_positions,
            )
            valid_summary = _summarize_period_trade_log(
                valid_result.get("trade_log", []) or [],
                fold["valid_start"],
                fold["valid_end"],
                args,
            )
            if int(valid_summary["trades"]) < min_required_valid_trades:
                continue
            if bool(args.require_positive_valid) and float(valid_summary["equity"]) <= 0.0:
                continue
            if bool(args.require_valid_pf_above_one) and float(valid_summary.get("profit_factor") or 0.0) <= 1.0:
                continue
            if best_valid_summary is None or float(valid_summary["score"]) > float(best_valid_summary["score"]):
                best_valid_summary = dict(valid_summary)
                best_threshold = float(threshold)

        if best_valid_summary is None or best_threshold is None:
            continue

        chosen_thresholds.append(float(best_threshold))
        gate_prob_full = np.full(len(close), np.nan, dtype=np.float64)
        gate_prob_full[signal_positions] = signal_probs
        test_mask = np.isin(signal_years, test_years_arr)
        test_signal_positions = signal_positions[test_mask]
        test_signal_probs = signal_probs[test_mask]
        gated_signal_side = signal_side.copy()
        gated_signal_early_exit = signal_early_exit.copy()
        blocked_positions = test_signal_positions[test_signal_probs < float(best_threshold)]
        if blocked_positions.size:
            gated_signal_side[blocked_positions] = 0
            gated_signal_early_exit[blocked_positions] = -1
        test_result = _run_period_simulation(
            df,
            combo_ids,
            session_codes,
            holiday_mask,
            gated_signal_side,
            gated_signal_early_exit,
            signal_sl,
            signal_tp,
            original_side,
            int(args.contracts),
            point_value,
            fee_per_contract_rt,
            fold["test_start"],
            fold["test_end"],
            selected_rule_index=selected_rule_index,
            rule_order=rule_order if multirule_enabled else None,
            gate_prob=gate_prob_full,
            gate_threshold=float(best_threshold),
            hours=hours,
            minutes=minutes,
            test_positions=test_positions,
        )
        test_summary = _summarize_period_trade_log(
            test_result.get("trade_log", []) or [],
            fold["test_start"],
            fold["test_end"],
            args,
        )
        test_baseline = _run_period_simulation(
            df,
            combo_ids,
            session_codes,
            holiday_mask,
            signal_side,
            signal_early_exit,
            signal_sl,
            signal_tp,
            original_side,
            int(args.contracts),
            point_value,
            fee_per_contract_rt,
            fold["test_start"],
            fold["test_end"],
            selected_rule_index=selected_rule_index,
            rule_order=rule_order if multirule_enabled else None,
            gate_prob=None,
            gate_threshold=None,
            hours=hours,
            minutes=minutes,
            test_positions=test_positions,
        )
        test_baseline_summary = _summarize_period_trade_log(
            test_baseline.get("trade_log", []) or [],
            fold["test_start"],
            fold["test_end"],
            args,
        )
        for trade in test_baseline.get("trade_log", []) or []:
            tagged = dict(trade)
            tagged["walkforward_fold"] = str(fold["name"])
            baseline_oos_trade_log.append(tagged)

        for trade in test_result.get("trade_log", []) or []:
            tagged = dict(trade)
            tagged["walkforward_fold"] = str(fold["name"])
            oos_trade_log.append(tagged)

        fold_rows.append(
            {
                "fold": str(fold["name"]),
                "train_years": list(fold["train_years"]),
                "valid_years": list(fold["valid_years"]),
                "test_years": list(fold["test_years"]),
                "selected_threshold": float(best_threshold),
                "min_required_valid_trades": int(min_required_valid_trades),
                "valid_baseline": {key: _json_safe(value) for key, value in valid_baseline_summary.items()},
                "valid_selected": {key: _json_safe(value) for key, value in best_valid_summary.items()},
                "test_baseline": {key: _json_safe(value) for key, value in test_baseline_summary.items()},
                "test_selected": {key: _json_safe(value) for key, value in test_summary.items()},
            }
        )
        print(
            f"{fold['name']} threshold={float(best_threshold):.6f} "
            f"valid_score={float(best_valid_summary['score']):.2f} test_score={float(test_summary['score']):.2f} "
            f"test_trades={int(test_summary['trades'])}"
        )

    if not fold_rows:
        raise SystemExit("No walk-forward folds produced a valid threshold.")

    oos_trade_log = sorted(
        oos_trade_log,
        key=lambda trade: (
            str(trade.get("entry_time", "") or ""),
            str(trade.get("exit_time", "") or ""),
            str(trade.get("combo_key", "") or ""),
        ),
    )
    selected_fold_names = {str(row["fold"]) for row in fold_rows}
    selected_folds = [fold for fold in folds if str(fold["name"]) in selected_fold_names]
    oos_start = pd.Timestamp(selected_folds[0]["test_start"])
    oos_end = pd.Timestamp(selected_folds[-1]["test_end"])
    oos_summary = _summarize_period_trade_log(oos_trade_log, oos_start, oos_end, args)
    baseline_oos_trade_log = sorted(
        baseline_oos_trade_log,
        key=lambda trade: (
            str(trade.get("entry_time", "") or ""),
            str(trade.get("exit_time", "") or ""),
            str(trade.get("combo_key", "") or ""),
        ),
    )
    baseline_oos_summary = _summarize_period_trade_log(baseline_oos_trade_log, oos_start, oos_end, args)

    stable_threshold = float(np.median(np.asarray(chosen_thresholds, dtype=np.float64)))
    final_model = _build_classifier(str(args.model), int(args.random_state))
    final_train_mask = np.ones(len(executed_years), dtype=bool)
    if holdout_years:
        final_train_mask = ~np.isin(executed_years, np.asarray(holdout_years, dtype=np.int16))
    if int(final_train_mask.sum()) < 100:
        raise SystemExit("Not enough pre-holdout executed trades to fit final model.")
    final_model.fit(
        executed_features.loc[final_train_mask].to_numpy(dtype=np.float32),
        y[final_train_mask],
        sample_weight=_train_weights(y[final_train_mask]),
    )

    holdout_summary = None
    baseline_holdout_summary = None
    holdout_trade_log: list[dict] = []
    baseline_holdout_trade_log: list[dict] = []
    holdout_start = None
    holdout_end = None
    if holdout_years:
        holdout_years_arr = np.asarray(holdout_years, dtype=np.int16)
        final_signal_probs = np.asarray(
            final_model.predict_proba(signal_features.to_numpy(dtype=np.float32))[:, 1],
            dtype=np.float64,
        )
        holdout_signal_mask = np.isin(signal_years, holdout_years_arr)
        holdout_signal_positions = signal_positions[holdout_signal_mask]
        holdout_signal_probs = final_signal_probs[holdout_signal_mask]
        holdout_start, holdout_end = _period_bounds(df.index, holdout_years)
        gate_prob_full = np.full(len(close), np.nan, dtype=np.float64)
        gate_prob_full[signal_positions] = final_signal_probs
        gated_signal_side = signal_side.copy()
        gated_signal_early_exit = signal_early_exit.copy()
        blocked_positions = holdout_signal_positions[holdout_signal_probs < float(stable_threshold)]
        if blocked_positions.size:
            gated_signal_side[blocked_positions] = 0
            gated_signal_early_exit[blocked_positions] = -1
        holdout_result = _run_period_simulation(
            df,
            combo_ids,
            session_codes,
            holiday_mask,
            gated_signal_side,
            gated_signal_early_exit,
            signal_sl,
            signal_tp,
            original_side,
            int(args.contracts),
            point_value,
            fee_per_contract_rt,
            holdout_start,
            holdout_end,
            selected_rule_index=selected_rule_index,
            rule_order=rule_order if multirule_enabled else None,
            gate_prob=gate_prob_full,
            gate_threshold=float(stable_threshold),
            hours=hours,
            minutes=minutes,
            test_positions=test_positions,
        )
        baseline_holdout = _run_period_simulation(
            df,
            combo_ids,
            session_codes,
            holiday_mask,
            signal_side,
            signal_early_exit,
            signal_sl,
            signal_tp,
            original_side,
            int(args.contracts),
            point_value,
            fee_per_contract_rt,
            holdout_start,
            holdout_end,
            selected_rule_index=selected_rule_index,
            rule_order=rule_order if multirule_enabled else None,
            gate_prob=None,
            gate_threshold=None,
            hours=hours,
            minutes=minutes,
            test_positions=test_positions,
        )
        holdout_trade_log = holdout_result.get("trade_log", []) or []
        baseline_holdout_trade_log = baseline_holdout.get("trade_log", []) or []
        holdout_summary = _summarize_period_trade_log(holdout_trade_log, holdout_start, holdout_end, args)
        baseline_holdout_summary = _summarize_period_trade_log(
            baseline_holdout_trade_log,
            holdout_start,
            holdout_end,
            args,
        )

    artifact_root = _resolve_path(str(args.artifact_root), "artifacts/regimeadaptive_v16_walkforward_research")
    artifact_root.mkdir(parents=True, exist_ok=True)
    model_path = artifact_root / "regimeadaptive_gate_model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": list(GATE_FEATURE_COLUMNS),
            "threshold": float(stable_threshold),
            "artifact_source": str(getattr(artifact, "path", "")),
            "selection_method": "walkforward_validation_median_threshold",
            "final_holdout_years": list(holdout_years),
        },
        model_path,
    )

    candidate_payload = copy.deepcopy(artifact.payload)
    candidate_payload["created_at"] = dt.datetime.now(NY_TZ).isoformat()
    candidate_payload["version"] = "regimeadaptive_dense_ml_gate_walkforward_research"
    candidate_payload["signal_gate"] = {
        "enabled": True,
        "model_path": str(model_path.name),
        "threshold": float(stable_threshold),
        "feature_columns": list(GATE_FEATURE_COLUMNS),
    }
    candidate_payload["metadata"] = {
        **(candidate_payload.get("metadata", {}) if isinstance(candidate_payload.get("metadata", {}), dict) else {}),
        "gate_training": {
            "source_artifact_path": str(getattr(artifact, "path", "")),
            "model_name": str(args.model),
            "random_state": int(args.random_state),
            "feature_columns": list(GATE_FEATURE_COLUMNS),
            "selection_method": "walkforward_validation_only",
            "final_holdout_years": list(holdout_years),
            "final_fit_years": [int(year) for year in all_years if int(year) not in set(holdout_years)],
            "stable_threshold": float(stable_threshold),
            "chosen_thresholds": [float(value) for value in chosen_thresholds],
            "fold_rows": fold_rows,
            "oos_summary": {key: _json_safe(value) for key, value in oos_summary.items()},
            "baseline_oos_summary": {key: _json_safe(value) for key, value in baseline_oos_summary.items()},
            "holdout_summary": {key: _json_safe(value) for key, value in (holdout_summary or {}).items()},
            "baseline_holdout_summary": {
                key: _json_safe(value) for key, value in (baseline_holdout_summary or {}).items()
            },
        },
    }

    artifact_path = artifact_root / "regimeadaptive_gated_walkforward_research.json"
    artifact_path.write_text(json.dumps(_json_safe(candidate_payload), indent=2, ensure_ascii=True), encoding="utf-8")
    if bool(args.write_latest):
        shutil.copyfile(artifact_path, artifact_root / "latest.json")

    walkforward_report_path = artifact_root / "walkforward_report.json"
    walkforward_report_path.write_text(
        json.dumps(
            {
                "created_at": dt.datetime.now(NY_TZ).isoformat(),
                "source_artifact_path": str(getattr(artifact, "path", "")),
                "source_data_path": str(source),
                "symbol": symbol_label,
                "range_start": start_time.isoformat(),
                "range_end": end_time.isoformat(),
                "walkforward_oos_start": oos_start.isoformat(),
                "walkforward_oos_end": oos_end.isoformat(),
                "final_holdout_years": list(holdout_years),
                "final_holdout_start": holdout_start.isoformat() if holdout_start is not None else None,
                "final_holdout_end": holdout_end.isoformat() if holdout_end is not None else None,
                "fold_rows": fold_rows,
                "oos_summary": {key: _json_safe(value) for key, value in oos_summary.items()},
                "baseline_oos_summary": {key: _json_safe(value) for key, value in baseline_oos_summary.items()},
                "holdout_summary": {key: _json_safe(value) for key, value in (holdout_summary or {}).items()},
                "baseline_holdout_summary": {
                    key: _json_safe(value) for key, value in (baseline_holdout_summary or {}).items()
                },
                "stable_threshold": float(stable_threshold),
                "candidate_artifact_path": str(artifact_path),
                "candidate_model_path": str(model_path),
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    oos_csv_path = artifact_root / "walkforward_oos_trades.csv"
    _write_converted_csv(oos_csv_path, oos_trade_log)
    if holdout_trade_log:
        holdout_csv_path = artifact_root / "final_holdout_trades.csv"
        _write_converted_csv(holdout_csv_path, holdout_trade_log)
    else:
        holdout_csv_path = None

    print(f"artifact={artifact_path}")
    print(f"model={model_path}")
    print(f"walkforward_report={walkforward_report_path}")
    print(f"walkforward_csv={oos_csv_path}")
    if holdout_csv_path is not None:
        print(f"holdout_csv={holdout_csv_path}")
    print(
        f"oos trades={int(oos_summary['trades'])} equity={float(oos_summary['equity']):.2f} "
        f"daily_sharpe={float(oos_summary['daily_sharpe']):.4f} stable_threshold={float(stable_threshold):.6f}"
    )
    if holdout_summary is not None:
        print(
            f"holdout years={holdout_years} trades={int(holdout_summary['trades'])} "
            f"equity={float(holdout_summary['equity']):.2f} "
            f"daily_sharpe={float(holdout_summary['daily_sharpe']):.4f}"
        )


if __name__ == "__main__":
    main()
