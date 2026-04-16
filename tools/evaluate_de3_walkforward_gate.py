from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest_mes_et as bt
from de3_walkforward_gate import (
    build_feature_row,
    build_model_frame,
    de3_lane_context_key,
    de3_variant_key,
)
import tools.evaluate_de3_backtest_admission as eva


DEFAULT_PRE2022_REPORT = (
    "backtest_reports/backtest_AUTO_BY_DAY_20110101_0000_20241231_2359_20260318_230819.json"
)
DEFAULT_YEAR_REPORTS = {
    2022: "backtest_reports/de3_backtest_admission_eval_candidate_tuned/backtest_AUTO_BY_DAY_20220101_0000_20221231_2359_20260322_121929.json",
    2023: "backtest_reports/de3_backtest_admission_eval_candidate_tuned/backtest_AUTO_BY_DAY_20230101_0000_20231231_2359_20260322_124834.json",
    2024: "backtest_reports/de3_backtest_admission_eval_candidate_tuned/backtest_AUTO_BY_DAY_20240101_0000_20241231_2359_20260322_132056.json",
}


def _resolve_path(path_text: str) -> Path:
    path = Path(str(path_text or "").strip()).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_trade_log(report_path: Path) -> list[dict]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    trade_log = payload.get("trade_log", []) or []
    return [dict(trade) for trade in trade_log if isinstance(trade, dict)]


def _trade_time(trade: dict) -> pd.Timestamp:
    raw = trade.get("entry_time") or trade.get("signal_time") or trade.get("exit_time")
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        return ts.tz_localize(bt.NY_TZ)
    return ts.tz_convert(bt.NY_TZ)


def _per_contract_pnl(trade: dict) -> float:
    size = max(1, int(trade.get("size", 1) or 1))
    pnl_net = float(trade.get("pnl_net", trade.get("pnl_dollars", 0.0)) or 0.0)
    return float(pnl_net / float(size))


def _rows_from_trades(
    trades: list[dict],
    *,
    lane_window: int,
    variant_window: int,
) -> pd.DataFrame:
    lane_history: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=int(lane_window)))
    variant_history: defaultdict[str, deque[float]] = defaultdict(lambda: deque(maxlen=int(variant_window)))
    rows: list[dict[str, Any]] = []
    ordered = sorted(trades, key=_trade_time)
    for trade in ordered:
        ts = _trade_time(trade)
        row = build_feature_row(
            trade,
            timestamp=ts.to_pydatetime(),
            lane_ctx_history=lane_history,
            variant_history=variant_history,
        )
        size = max(1, int(trade.get("size", 1) or 1))
        pnl_net = float(trade.get("pnl_net", trade.get("pnl_dollars", 0.0)) or 0.0)
        row["__entry_time"] = ts.isoformat()
        row["__year"] = int(ts.year)
        row["__size"] = int(size)
        row["__pnl_net"] = float(pnl_net)
        row["__pnl_per_contract"] = float(pnl_net / float(size))
        rows.append(row)

        per_contract = float(pnl_net / float(size))
        lane_key = de3_lane_context_key(trade)
        variant_key = de3_variant_key(trade)
        if lane_key:
            lane_history[lane_key].append(per_contract)
        if variant_key:
            variant_history[variant_key].append(per_contract)
    return pd.DataFrame(rows)


def _build_training_dataset(
    pre2022_report: Path,
    year_reports: dict[int, Path],
    *,
    lane_window: int,
    variant_window: int,
) -> pd.DataFrame:
    pre2022_trades = [
        trade
        for trade in _load_trade_log(pre2022_report)
        if _trade_time(trade).year < 2022
    ]
    all_frames = [
        _rows_from_trades(
            pre2022_trades,
            lane_window=lane_window,
            variant_window=variant_window,
        )
    ]
    for year in sorted(year_reports):
        trade_log = _load_trade_log(year_reports[year])
        if not trade_log:
            continue
        all_frames.append(
            _rows_from_trades(
                trade_log,
                lane_window=lane_window,
                variant_window=variant_window,
            )
        )
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def _simulate_policy(
    rows_df: pd.DataFrame,
    predicted_ev: np.ndarray,
    *,
    block_threshold: float | None,
    defensive_threshold: float | None,
    defensive_mult: float,
    min_contracts: int,
    reduce_only: bool,
) -> dict[str, Any]:
    trades_out: list[dict] = []
    for idx, row in rows_df.iterrows():
        requested_size = max(1, int(row["__size"]))
        eff_size = requested_size
        pred = float(predicted_ev[idx])
        if block_threshold is not None and math.isfinite(float(block_threshold)) and pred <= float(block_threshold):
            eff_size = 0
        elif (
            defensive_threshold is not None
            and math.isfinite(float(defensive_threshold))
            and pred <= float(defensive_threshold)
        ):
            eff_size = int(round(float(requested_size) * float(defensive_mult)))
            eff_size = max(int(min_contracts), eff_size)
            if reduce_only:
                eff_size = min(requested_size, eff_size)
        if eff_size <= 0:
            continue
        per_contract = float(row["__pnl_per_contract"])
        adjusted_pnl = float(per_contract * float(eff_size))
        trades_out.append(
            {
                "entry_time": str(row["__entry_time"]),
                "exit_time": str(row["__entry_time"]),
                "pnl_net": float(adjusted_pnl),
                "pnl_dollars": float(adjusted_pnl),
                "size": int(eff_size),
            }
        )

    equity_curve = np.cumsum(
        np.asarray([float(trade["pnl_net"]) for trade in trades_out], dtype=np.float64)
    )
    running_peak = np.maximum.accumulate(equity_curve) if equity_curve.size else np.asarray([], dtype=np.float64)
    max_dd = float(np.max(running_peak - equity_curve)) if equity_curve.size else 0.0
    risk = bt._compute_backtest_risk_metrics(trades_out)
    return {
        "trade_log": trades_out,
        "equity": float(sum(float(trade["pnl_net"]) for trade in trades_out)),
        "trades": int(len(trades_out)),
        "max_drawdown": float(max_dd),
        "profit_factor": float(risk.get("profit_factor", 0.0) or 0.0),
        "daily_sharpe": float(risk.get("daily_sharpe", 0.0) or 0.0),
        "ui_sharpe": float(eva._ui_style_sharpe(trades_out)),
    }


def _choose_policy(
    valid_df: pd.DataFrame,
    valid_pred: np.ndarray,
    *,
    min_contracts: int,
    reduce_only: bool,
    min_trade_ratio: float,
    min_validation_lift: float,
    dd_weight: float,
    sharpe_weight: float,
) -> dict[str, Any]:
    baseline = _simulate_policy(
        valid_df,
        valid_pred,
        block_threshold=None,
        defensive_threshold=None,
        defensive_mult=1.0,
        min_contracts=min_contracts,
        reduce_only=reduce_only,
    )
    if valid_df.empty or valid_pred.size == 0:
        return {
            "block_threshold": None,
            "defensive_threshold": None,
            "defensive_size_multiplier": 1.0,
            "validation_baseline": baseline,
            "validation_selected": baseline,
            "validation_lift": 0.0,
        }

    quantile_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.55, 0.65]
    q_map = {q: float(np.quantile(valid_pred, q)) for q in quantile_levels}
    best = {
        "score": 0.0,
        "block_threshold": None,
        "defensive_threshold": None,
        "defensive_size_multiplier": 1.0,
        "summary": baseline,
    }
    base_trade_floor = max(1, int(math.floor(float(baseline["trades"]) * float(min_trade_ratio))))

    block_candidates = [None, q_map[0.05], q_map[0.10], q_map[0.15], q_map[0.20]]
    defensive_candidates = [None, q_map[0.20], q_map[0.25], q_map[0.35], q_map[0.45], q_map[0.55]]
    for block_threshold in block_candidates:
        for defensive_threshold in defensive_candidates:
            if (
                block_threshold is not None
                and defensive_threshold is not None
                and float(block_threshold) > float(defensive_threshold)
            ):
                continue
            for defensive_mult in [0.40, 0.50, 0.60, 0.75]:
                summary = _simulate_policy(
                    valid_df,
                    valid_pred,
                    block_threshold=block_threshold,
                    defensive_threshold=defensive_threshold,
                    defensive_mult=float(defensive_mult),
                    min_contracts=min_contracts,
                    reduce_only=reduce_only,
                )
                if summary["trades"] < base_trade_floor:
                    continue
                score = (
                    float(summary["equity"] - baseline["equity"])
                    + float(dd_weight) * float(baseline["max_drawdown"] - summary["max_drawdown"])
                    + float(sharpe_weight) * float(summary["daily_sharpe"] - baseline["daily_sharpe"])
                )
                if score > float(best["score"]) + 1e-9:
                    best = {
                        "score": float(score),
                        "block_threshold": block_threshold,
                        "defensive_threshold": defensive_threshold,
                        "defensive_size_multiplier": float(defensive_mult),
                        "summary": summary,
                    }

    if float(best["score"]) < float(min_validation_lift):
        return {
            "block_threshold": None,
            "defensive_threshold": None,
            "defensive_size_multiplier": 1.0,
            "validation_baseline": baseline,
            "validation_selected": baseline,
            "validation_lift": 0.0,
        }
    return {
        "block_threshold": best["block_threshold"],
        "defensive_threshold": best["defensive_threshold"],
        "defensive_size_multiplier": float(best["defensive_size_multiplier"]),
        "validation_baseline": baseline,
        "validation_selected": best["summary"],
        "validation_lift": float(best["score"]),
    }


def _train_artifact(
    dataset: pd.DataFrame,
    *,
    test_years: list[int],
    out_dir: Path,
    lane_window: int,
    variant_window: int,
    random_state: int,
    min_trade_ratio: float,
    min_validation_lift: float,
    dd_weight: float,
    sharpe_weight: float,
) -> tuple[Path, dict[str, Any]]:
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    folds: list[dict[str, Any]] = []
    periods: list[dict[str, Any]] = []
    for test_year in test_years:
        valid_year = int(test_year) - 1
        train_df = dataset.loc[dataset["__year"] < valid_year].reset_index(drop=True)
        valid_df = dataset.loc[dataset["__year"] == valid_year].reset_index(drop=True)
        if len(train_df) < 200 or len(valid_df) < 50:
            folds.append(
                {
                    "test_year": int(test_year),
                    "status": "insufficient_data",
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                }
            )
            continue
        x_train = build_model_frame(train_df)
        y_train = np.clip(
            train_df["__pnl_per_contract"].to_numpy(dtype=np.float64),
            -25.0,
            25.0,
        )
        sample_weight = 1.0 + np.clip(np.abs(y_train), 0.0, 20.0) / 10.0
        model = HistGradientBoostingRegressor(
            max_depth=5,
            max_iter=350,
            learning_rate=0.05,
            min_samples_leaf=40,
            l2_regularization=0.2,
            random_state=int(random_state + test_year),
        )
        model.fit(
            x_train.to_numpy(dtype=np.float32, copy=False),
            y_train.astype(np.float32, copy=False),
            sample_weight=sample_weight.astype(np.float32, copy=False),
        )
        model_columns = list(x_train.columns)
        x_valid = build_model_frame(valid_df, model_columns=model_columns)
        valid_pred = np.asarray(
            model.predict(x_valid.to_numpy(dtype=np.float32, copy=False)),
            dtype=np.float64,
        )
        policy = _choose_policy(
            valid_df,
            valid_pred,
            min_contracts=1,
            reduce_only=True,
            min_trade_ratio=min_trade_ratio,
            min_validation_lift=min_validation_lift,
            dd_weight=dd_weight,
            sharpe_weight=sharpe_weight,
        )
        model_path = model_dir / f"de3_walkforward_gate_{test_year}.joblib"
        joblib.dump(
            {
                "model": model,
                "model_columns": model_columns,
            },
            model_path,
        )
        periods.append(
            {
                "name": str(test_year),
                "start": f"{test_year}-01-01T00:00:00-05:00",
                "end": f"{test_year}-12-31T23:59:59-05:00",
                "model_path": str(model_path),
                "block_threshold": (
                    float(policy["block_threshold"])
                    if policy["block_threshold"] is not None
                    else None
                ),
                "defensive_threshold": (
                    float(policy["defensive_threshold"])
                    if policy["defensive_threshold"] is not None
                    else None
                ),
                "defensive_size_multiplier": float(policy["defensive_size_multiplier"]),
            }
        )
        folds.append(
            {
                "test_year": int(test_year),
                "status": "ok",
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "block_threshold": periods[-1]["block_threshold"],
                "defensive_threshold": periods[-1]["defensive_threshold"],
                "defensive_size_multiplier": float(policy["defensive_size_multiplier"]),
                "validation_lift": float(policy["validation_lift"]),
                "validation_baseline": policy["validation_baseline"],
                "validation_selected": policy["validation_selected"],
            }
        )

    artifact = {
        "version": "de3_walkforward_gate_v1",
        "created_at": datetime.now(bt.NY_TZ).isoformat(),
        "lane_context_history_window": int(lane_window),
        "variant_history_window": int(variant_window),
        "periods": periods,
        "folds": folds,
    }
    artifact_path = out_dir / "de3_walkforward_gate.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path, artifact


def _run_exact_mode(
    symbol_df,
    *,
    start_raw: str,
    end_raw: str,
    out_dir: Path,
    mode_label: str,
    gate_enabled: bool,
    artifact_path: Path | None,
    gate_mode: str,
    admission_enabled: bool | None,
) -> dict[str, Any]:
    cfg_backup = copy.deepcopy(bt.CONFIG)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        de3_runtime_cfg = (
            bt.CONFIG.setdefault("DE3_V4", {}).setdefault("runtime", {})
            if isinstance(bt.CONFIG.setdefault("DE3_V4", {}).setdefault("runtime", {}), dict)
            else {}
        )
        if admission_enabled is not None:
            admission_cfg = de3_runtime_cfg.get("backtest_admission_controller", {}) or {}
            if not isinstance(admission_cfg, dict):
                admission_cfg = {}
            admission_cfg["enabled"] = bool(admission_enabled)
            de3_runtime_cfg["backtest_admission_controller"] = admission_cfg
        gate_cfg = de3_runtime_cfg.get("backtest_walkforward_gate", {}) or {}
        if not isinstance(gate_cfg, dict):
            gate_cfg = {}
        gate_cfg["enabled"] = bool(gate_enabled)
        gate_cfg["mode"] = str(gate_mode)
        if artifact_path is not None:
            gate_cfg["artifact_path"] = str(artifact_path)
        de3_runtime_cfg["backtest_walkforward_gate"] = gate_cfg

        start_time = bt.parse_user_datetime(start_raw, bt.NY_TZ, is_end=False)
        end_time = bt.parse_user_datetime(end_raw, bt.NY_TZ, is_end=True)
        stats = bt.run_backtest(
            symbol_df,
            start_time,
            end_time,
            enabled_strategies={"DynamicEngine3Strategy"},
            enabled_filters=set(),
        )
        trade_log = stats.get("trade_log", []) or []
        timestamp = datetime.now(bt.NY_TZ).strftime("%Y%m%d_%H%M%S")
        start_tag = start_time.strftime("%Y%m%d_%H%M")
        end_tag = end_time.strftime("%Y%m%d_%H%M")
        stem = f"backtest_AUTO_BY_DAY_{start_tag}_{end_tag}_{timestamp}_{mode_label}"
        report_path = out_dir / f"{stem}.json"
        csv_path = out_dir / f"converted_{stem}.csv"
        eva._write_converted_csv(trade_log, csv_path)
        mc = (
            bt._build_monte_carlo_summary(
                trade_log,
                stats,
                simulations=bt.BACKTEST_MONTE_CARLO_SIMULATIONS,
                seed=bt.BACKTEST_MONTE_CARLO_SEED,
                starting_balance=bt.BACKTEST_MONTE_CARLO_START_BALANCE,
            )
            if bt.BACKTEST_MONTE_CARLO_ENABLED
            else {"enabled": False, "status": "disabled"}
        )
        payload = {
            "created_at": datetime.now(bt.NY_TZ).isoformat(),
            "range_start": start_time.isoformat(),
            "range_end": end_time.isoformat(),
            "mode": mode_label,
            "summary": {
                "equity": float(stats.get("equity", 0.0) or 0.0),
                "trades": int(stats.get("trades", 0) or 0),
                "wins": int(stats.get("wins", 0) or 0),
                "losses": int(stats.get("losses", 0) or 0),
                "winrate": float(stats.get("winrate", 0.0) or 0.0),
                "max_drawdown": float(stats.get("max_drawdown", 0.0) or 0.0),
                "profit_factor": float(stats.get("profit_factor", 0.0) or 0.0),
                "avg_trade_net": float(stats.get("avg_trade_net", 0.0) or 0.0),
                "daily_sharpe": float(stats.get("daily_sharpe", 0.0) or 0.0),
                "ui_sharpe": float(eva._ui_style_sharpe(trade_log)),
                "trading_days": int(stats.get("trading_days", 0) or 0),
            },
            "monte_carlo": mc,
            "de3_walkforward_gate_summary": copy.deepcopy(
                stats.get("de3_walkforward_gate_summary", {}) or {}
            ),
            "de3_backtest_admission_summary": copy.deepcopy(
                stats.get("de3_backtest_admission_summary", {}) or {}
            ),
            "csv_path": str(csv_path),
        }
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "report_path": str(report_path),
            "csv_path": str(csv_path),
            "summary": payload["summary"],
            "monte_carlo": mc,
            "de3_walkforward_gate_summary": payload["de3_walkforward_gate_summary"],
            "de3_backtest_admission_summary": payload["de3_backtest_admission_summary"],
        }
    finally:
        bt.CONFIG.clear()
        bt.CONFIG.update(cfg_backup)


def _delta(base: dict[str, Any], adapted: dict[str, Any]) -> dict[str, float]:
    b = base.get("summary", {}) or {}
    a = adapted.get("summary", {}) or {}
    bmc = base.get("monte_carlo", {}) or {}
    amc = adapted.get("monte_carlo", {}) or {}
    return {
        "equity": float(a.get("equity", 0.0) or 0.0) - float(b.get("equity", 0.0) or 0.0),
        "trades": int(a.get("trades", 0) or 0) - int(b.get("trades", 0) or 0),
        "winrate": float(a.get("winrate", 0.0) or 0.0) - float(b.get("winrate", 0.0) or 0.0),
        "max_drawdown": float(a.get("max_drawdown", 0.0) or 0.0) - float(b.get("max_drawdown", 0.0) or 0.0),
        "daily_sharpe": float(a.get("daily_sharpe", 0.0) or 0.0) - float(b.get("daily_sharpe", 0.0) or 0.0),
        "ui_sharpe": float(a.get("ui_sharpe", 0.0) or 0.0) - float(b.get("ui_sharpe", 0.0) or 0.0),
        "profit_factor": float(a.get("profit_factor", 0.0) or 0.0) - float(b.get("profit_factor", 0.0) or 0.0),
        "mc_net_pnl_mean": float(amc.get("net_pnl_mean", 0.0) or 0.0) - float(bmc.get("net_pnl_mean", 0.0) or 0.0),
        "mc_net_pnl_p05": float(amc.get("net_pnl_p05", 0.0) or 0.0) - float(bmc.get("net_pnl_p05", 0.0) or 0.0),
        "mc_max_drawdown_mean": float(amc.get("max_drawdown_mean", 0.0) or 0.0) - float(bmc.get("max_drawdown_mean", 0.0) or 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a backtest-only DE3 walk-forward gate and run exact baseline vs gated comparisons."
    )
    parser.add_argument("--pre2022-report", default=DEFAULT_PRE2022_REPORT)
    parser.add_argument("--report-2022", default=DEFAULT_YEAR_REPORTS[2022])
    parser.add_argument("--report-2023", default=DEFAULT_YEAR_REPORTS[2023])
    parser.add_argument("--report-2024", default=DEFAULT_YEAR_REPORTS[2024])
    parser.add_argument("--output-dir", default="backtest_reports/de3_walkforward_gate_eval")
    parser.add_argument("--test-years", default="2022,2023,2024,2025")
    parser.add_argument("--lane-history-window", type=int, default=20)
    parser.add_argument("--variant-history-window", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=1337)
    parser.add_argument("--min-trade-ratio", type=float, default=0.60)
    parser.add_argument("--min-validation-lift", type=float, default=75.0)
    parser.add_argument("--dd-weight", type=float, default=0.30)
    parser.add_argument("--sharpe-weight", type=float, default=250.0)
    parser.add_argument("--gate-mode", choices=["block", "defensive", "block_defensive"], default="block_defensive")
    parser.add_argument("--baseline-admission", choices=["default", "on", "off"], default="default")
    parser.add_argument("--adapted-admission", choices=["default", "on", "off"], default="default")
    parser.add_argument("--run-continuous-recent", action="store_true")
    args = parser.parse_args()

    out_dir = _resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    year_reports = {
        2022: _resolve_path(args.report_2022),
        2023: _resolve_path(args.report_2023),
        2024: _resolve_path(args.report_2024),
    }
    dataset = _build_training_dataset(
        _resolve_path(args.pre2022_report),
        year_reports,
        lane_window=int(args.lane_history_window),
        variant_window=int(args.variant_history_window),
    )
    test_years = [int(part.strip()) for part in str(args.test_years).split(",") if part.strip()]
    artifact_path, artifact = _train_artifact(
        dataset,
        test_years=test_years,
        out_dir=out_dir,
        lane_window=int(args.lane_history_window),
        variant_window=int(args.variant_history_window),
        random_state=int(args.random_state),
        min_trade_ratio=float(args.min_trade_ratio),
        min_validation_lift=float(args.min_validation_lift),
        dd_weight=float(args.dd_weight),
        sharpe_weight=float(args.sharpe_weight),
    )

    _, symbol_df = eva._load_symbol_df()
    exact_results: dict[str, Any] = {
        "artifact_path": str(artifact_path),
        "folds": artifact.get("folds", []),
        "years": {},
    }
    baseline_admission_enabled = None
    if args.baseline_admission == "on":
        baseline_admission_enabled = True
    elif args.baseline_admission == "off":
        baseline_admission_enabled = False
    adapted_admission_enabled = None
    if args.adapted_admission == "on":
        adapted_admission_enabled = True
    elif args.adapted_admission == "off":
        adapted_admission_enabled = False

    for year in test_years:
        start_raw = f"{year}-01-01"
        end_raw = f"{year}-12-31 23:59"
        baseline = _run_exact_mode(
            symbol_df,
            start_raw=start_raw,
            end_raw=end_raw,
            out_dir=out_dir,
            mode_label=f"{year}_baseline",
            gate_enabled=False,
            artifact_path=None,
            gate_mode=str(args.gate_mode),
            admission_enabled=baseline_admission_enabled,
        )
        adapted = _run_exact_mode(
            symbol_df,
            start_raw=start_raw,
            end_raw=end_raw,
            out_dir=out_dir,
            mode_label=f"{year}_adapted",
            gate_enabled=True,
            artifact_path=artifact_path,
            gate_mode=str(args.gate_mode),
            admission_enabled=adapted_admission_enabled,
        )
        exact_results["years"][str(year)] = {
            "baseline": baseline,
            "adapted": adapted,
            "delta": _delta(baseline, adapted),
        }

    if args.run_continuous_recent:
        baseline = _run_exact_mode(
            symbol_df,
            start_raw="2024-01-01",
            end_raw="2025-12-31 23:59",
            out_dir=out_dir,
            mode_label="2024_2025_baseline",
            gate_enabled=False,
            artifact_path=None,
            gate_mode=str(args.gate_mode),
            admission_enabled=baseline_admission_enabled,
        )
        adapted = _run_exact_mode(
            symbol_df,
            start_raw="2024-01-01",
            end_raw="2025-12-31 23:59",
            out_dir=out_dir,
            mode_label="2024_2025_adapted",
            gate_enabled=True,
            artifact_path=artifact_path,
            gate_mode=str(args.gate_mode),
            admission_enabled=adapted_admission_enabled,
        )
        exact_results["continuous_2024_2025"] = {
            "baseline": baseline,
            "adapted": adapted,
            "delta": _delta(baseline, adapted),
        }

    summary_path = out_dir / "de3_walkforward_gate_exact_summary.json"
    summary_path.write_text(json.dumps(exact_results, indent=2), encoding="utf-8")
    print(f"artifact={artifact_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
