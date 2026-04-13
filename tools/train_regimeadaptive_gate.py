import argparse
import copy
import datetime as dt
import json
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
from regimeadaptive_gate import (
    GATE_FEATURE_COLUMNS,
    build_gate_feature_frame_for_positions,
)
from tools.backtest_regimeadaptive_robust import (
    _artifact_rule_order,
    _build_artifact_lookups,
    _build_artifact_rule_lookup,
    _build_multirule_signal_arrays,
    _build_rule_strength_arrays,
    _build_signal_arrays,
    _parse_datetime,
    _rolling_extrema_cache,
    _simulate,
)
from tools.regimeadaptive_filterless_runner import (
    NY_TZ,
    _atr_array,
    _build_combo_arrays,
    _build_holiday_mask,
    _load_bars,
    _rolling_cache,
)
from tools.train_regimeadaptive_robust import _build_split_map
from tools.train_regimeadaptive_v2 import _json_safe
from tools.train_regimeadaptive_v5 import _evaluate_trade_log_v5


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


def _label_years(times: pd.Series, split_meta: dict) -> pd.Series:
    years = pd.to_datetime(times, utc=True).dt.tz_convert(NY_TZ).dt.year.astype(int)
    labels = pd.Series("train", index=times.index, dtype=object)
    valid_set = set(split_meta.get("valid_years", []))
    test_set = set(split_meta.get("test_years", []))
    labels.loc[years.isin(valid_set)] = "valid"
    labels.loc[years.isin(test_set)] = "test"
    return labels


def _probability_threshold_grid(probs: np.ndarray) -> list[float]:
    quantiles = np.linspace(0.0, 0.92, 24)
    values = [float(np.quantile(probs, q)) for q in quantiles]
    values.extend([0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    out = sorted({round(float(value), 6) for value in values if np.isfinite(value)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a dense ML gate on top of a RegimeAdaptive artifact and search gating thresholds."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--artifact", default="artifacts/regimeadaptive_v14_dense_balanced/latest.json")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--valid-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--min-target-trades", type=int, default=4674)
    parser.add_argument("--model", default="hgb")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-score-weight", type=float, default=0.25)
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
    _, _, split_meta = _build_split_map(df.index, int(args.valid_years), int(args.test_years))

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
    selected_rule_index = None
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

    baseline_result = _simulate(
        df,
        start_time,
        end_time,
        combo_ids,
        session_codes,
        holiday_mask,
        signal_side.copy(),
        signal_early_exit.copy(),
        signal_sl.copy(),
        signal_tp.copy(),
        original_side.copy(),
        int(args.contracts),
        point_value,
        fee_per_contract_rt,
        selected_rule_index=selected_rule_index,
        rule_order=rule_order if multirule_enabled else None,
        hours=hours,
        minutes=minutes,
        test_positions=test_positions,
    )
    baseline_trade_log = baseline_result.get("trade_log", []) or []
    baseline_eval = _evaluate_trade_log_v5(
        baseline_trade_log,
        split_meta,
        float(args.train_score_weight),
        start_time,
        end_time,
        args,
    )
    if not baseline_trade_log:
        raise SystemExit("Baseline artifact produced no trades.")

    index_lookup = {pd.Timestamp(ts).isoformat(): pos for pos, ts in enumerate(df.index)}
    executed_positions: list[int] = []
    executed_labels: list[int] = []
    executed_pnl: list[float] = []
    executed_times: list[str] = []
    seen_trade_ids: set[tuple[str, str, str]] = set()
    for trade in baseline_trade_log:
        signal_time_text = str(trade.get("signal_time", "") or trade.get("entry_time", "") or "")
        entry_time_text = str(trade.get("entry_time", "") or "")
        side_text = str(trade.get("side", "") or "")
        combo_key = str(trade.get("combo_key", "") or "")
        unique_key = (signal_time_text, side_text, combo_key)
        if unique_key in seen_trade_ids:
            continue
        seen_trade_ids.add(unique_key)
        ts = pd.Timestamp(signal_time_text)
        pos = index_lookup.get(ts.isoformat())
        if pos is None:
            continue
        executed_positions.append(int(pos))
        executed_labels.append(1 if float(trade.get("pnl_net", 0.0) or 0.0) > 0.0 else 0)
        executed_pnl.append(float(trade.get("pnl_net", 0.0) or 0.0))
        executed_times.append(signal_time_text)

    if not executed_positions:
        raise SystemExit("Could not align executed trades back to bar positions.")

    executed_positions_arr = np.asarray(executed_positions, dtype=np.int32)
    executed_features = build_gate_feature_frame_for_positions(
        pd.DatetimeIndex(df.index),
        open_arr,
        high,
        low,
        close,
        combo_ids,
        signal_side,
        original_side,
        selected_rule_index,
        executed_positions_arr,
        rule_order,
        rule_payloads,
        rolling,
        atr_cache,
        long_strength_matrix,
        short_strength_matrix,
    )
    executed_features = executed_features.reindex(columns=GATE_FEATURE_COLUMNS, fill_value=0.0)
    executed_times_ser = pd.Series(executed_times, index=executed_features.index)
    split_labels = _label_years(executed_times_ser, split_meta)
    y = np.asarray(executed_labels, dtype=np.int8)
    X = executed_features.reset_index(drop=True)
    split_labels = split_labels.reset_index(drop=True)

    train_mask = split_labels == "train"
    valid_mask = split_labels == "valid"
    test_mask = split_labels == "test"
    if int(train_mask.sum()) == 0 or int(valid_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        raise SystemExit("Not enough train/valid/test executed trades for gate training.")

    y_train = y[train_mask.to_numpy()]
    positive = max(1, int((y_train == 1).sum()))
    negative = max(1, int((y_train == 0).sum()))
    sample_weight = np.where(
        y_train == 1,
        len(y_train) / (2.0 * positive),
        len(y_train) / (2.0 * negative),
    )

    model = _build_classifier(str(args.model), int(args.random_state))
    model.fit(X.loc[train_mask].to_numpy(dtype=np.float32), y_train, sample_weight=sample_weight)

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
    signal_probs = np.asarray(
        model.predict_proba(signal_features.to_numpy(dtype=np.float32))[:, 1],
        dtype=np.float64,
    )
    probability_grid = _probability_threshold_grid(signal_probs)

    best_eval = None
    best_threshold = None
    threshold_rows: list[dict] = []
    for threshold in probability_grid:
        gated_signal_side = signal_side.copy()
        gated_signal_early_exit = signal_early_exit.copy()
        blocked_positions = signal_positions[signal_probs < float(threshold)]
        if blocked_positions.size:
            gated_signal_side[blocked_positions] = 0
            gated_signal_early_exit[blocked_positions] = -1
        result = _simulate(
            df,
            start_time,
            end_time,
            combo_ids,
            session_codes,
            holiday_mask,
            gated_signal_side,
            gated_signal_early_exit,
            signal_sl.copy(),
            signal_tp.copy(),
            original_side.copy(),
            int(args.contracts),
            point_value,
            fee_per_contract_rt,
            selected_rule_index=selected_rule_index,
            rule_order=rule_order if multirule_enabled else None,
            gate_prob=np.where(signal_side != 0, np.nan, np.nan),  # placeholder for signature compatibility
            gate_threshold=float(threshold),
            hours=hours,
            minutes=minutes,
            test_positions=test_positions,
        )
        eval_metrics = _evaluate_trade_log_v5(
            result.get("trade_log", []) or [],
            split_meta,
            float(args.train_score_weight),
            start_time,
            end_time,
            args,
        )
        threshold_row = {
            "threshold": float(threshold),
            "trades": int(eval_metrics["trades"]),
            "score": float(eval_metrics["score"]),
            "daily_sharpe": float(eval_metrics["daily_sharpe"]),
            "profit_factor": _json_safe(eval_metrics.get("profit_factor")),
            "negative_years": int(eval_metrics["negative_years"]),
            "worst_5y_pnl": float(eval_metrics["worst_5y_pnl"]),
        }
        threshold_rows.append(threshold_row)
        print(
            f"threshold={float(threshold):.6f} trades={int(eval_metrics['trades'])} "
            f"score={float(eval_metrics['score']):.2f} sharpe={float(eval_metrics['daily_sharpe']):.4f}"
        )
        if int(eval_metrics["trades"]) < int(args.min_target_trades):
            continue
        if best_eval is None or float(eval_metrics["score"]) > float(best_eval["score"]):
            best_eval = dict(eval_metrics)
            best_threshold = float(threshold)

    if best_eval is None or best_threshold is None:
        raise SystemExit(f"No threshold met the minimum trade target of {int(args.min_target_trades)}.")

    artifact_root = _resolve_path(str(args.artifact_root), "artifacts/regimeadaptive_v15_gated")
    artifact_root.mkdir(parents=True, exist_ok=True)

    model_path = artifact_root / "regimeadaptive_gate_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(GATE_FEATURE_COLUMNS),
            "threshold": float(best_threshold),
            "artifact_source": str(getattr(artifact, "path", "")),
        },
        model_path,
    )

    gated_payload = copy.deepcopy(artifact.payload)
    gated_payload["created_at"] = dt.datetime.now(NY_TZ).isoformat()
    gated_payload["version"] = "regimeadaptive_dense_ml_gate"
    gated_payload["signal_gate"] = {
        "enabled": True,
        "model_path": str(model_path.name),
        "threshold": float(best_threshold),
        "feature_columns": list(GATE_FEATURE_COLUMNS),
    }
    gated_payload["metadata"] = {
        **(gated_payload.get("metadata", {}) if isinstance(gated_payload.get("metadata", {}), dict) else {}),
        "gate_training": {
            "source_artifact_path": str(getattr(artifact, "path", "")),
            "model_name": str(args.model),
            "random_state": int(args.random_state),
            "feature_columns": list(GATE_FEATURE_COLUMNS),
            "baseline_eval": {key: _json_safe(value) for key, value in baseline_eval.items() if key != "trade_log"},
            "best_eval": {key: _json_safe(value) for key, value in best_eval.items() if key != "trade_log"},
            "best_threshold": float(best_threshold),
            "threshold_rows": threshold_rows,
            "executed_train_samples": int(train_mask.sum()),
            "executed_valid_samples": int(valid_mask.sum()),
            "executed_test_samples": int(test_mask.sum()),
            "signal_rows": int(len(signal_positions)),
            "min_target_trades": int(args.min_target_trades),
        },
    }

    artifact_path = artifact_root / "regimeadaptive_gated_artifact.json"
    artifact_path.write_text(json.dumps(_json_safe(gated_payload), indent=2, ensure_ascii=True), encoding="utf-8")
    report_path = artifact_root / "gate_report.json"
    report_path.write_text(
        json.dumps(
            {
                "baseline_eval": {key: _json_safe(value) for key, value in baseline_eval.items() if key != "trade_log"},
                "best_eval": {key: _json_safe(value) for key, value in best_eval.items() if key != "trade_log"},
                "best_threshold": float(best_threshold),
                "threshold_rows": threshold_rows,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    if bool(args.write_latest):
        shutil.copyfile(artifact_path, artifact_root / "latest.json")

    print(f"artifact={artifact_path}")
    print(f"model={model_path}")
    print(f"best_threshold={float(best_threshold):.6f}")
    print(
        f"baseline trades={int(baseline_eval['trades'])} sharpe={float(baseline_eval['daily_sharpe']):.4f} "
        f"best trades={int(best_eval['trades'])} sharpe={float(best_eval['daily_sharpe']):.4f}"
    )


if __name__ == "__main__":
    main()
