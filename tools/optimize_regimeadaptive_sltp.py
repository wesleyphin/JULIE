import argparse
import copy
import datetime as dt
import json
import os
import platform
import shutil
import sys
from pathlib import Path

if sys.platform.startswith("win"):
    _platform_machine = str(os.environ.get("PROCESSOR_ARCHITECTURE", "") or "").strip()
    if _platform_machine:
        platform.machine = lambda: _platform_machine  # type: ignore[assignment]

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from regimeadaptive_artifact import RegimeAdaptiveArtifact, load_regimeadaptive_artifact
from tools.backtest_regimeadaptive_robust import _parse_datetime
from tools.regimeadaptive_filterless_runner import (
    NY_TZ,
    SESSION_NAMES,
    _atr_array,
    _build_combo_arrays,
    _build_holiday_mask,
    _load_bars,
    _rolling_cache,
    _round_points_to_tick,
)
from tools.train_regimeadaptive_robust import _build_split_map, _fit_bracket_defaults, _simulate_event_bracket
from tools.train_regimeadaptive_v3 import _simulate_payload_v3
from tools.train_regimeadaptive_v5 import _evaluate_trade_log_v5
from tools.backtest_regimeadaptive_robust import _build_rule_strength_arrays


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw or "").split(","):
        text = str(item or "").strip()
        if text:
            out.append(float(text))
    return out


def _ensure_session_side(payload: dict, session_name: str, side: str, current_support: int) -> None:
    session_text = str(session_name).strip().upper()
    side_text = str(side).strip().upper()
    default_side = dict((payload.get("global_default", {}) or {}).get(side_text, {}))
    if not default_side:
        default_side = {"sl": 2.0, "tp": 3.0, "support": int(current_support)}
    payload.setdefault("session_defaults", {}).setdefault(session_text, {}).setdefault(side_text, default_side)


def _set_session_side_sltp(payload: dict, session_name: str, side: str, sl: float, tp: float, support: int) -> dict:
    next_payload = copy.deepcopy(payload)
    _ensure_session_side(next_payload, session_name, side, support)
    next_payload.setdefault("session_defaults", {}).setdefault(str(session_name).upper(), {})[str(side).upper()] = {
        "sl": float(sl),
        "tp": float(tp),
        "support": int(support),
    }
    return next_payload


def _active_session_side_buckets(trade_log: list[dict]) -> list[dict]:
    if not trade_log:
        return []
    trades = pd.DataFrame(trade_log)
    trades["session"] = trades["session"].astype(str)
    trades["side"] = trades["side"].astype(str)
    trades["pnl_net"] = trades["pnl_net"].astype(float)
    rows: list[dict] = []
    for (session_name, side_name), group in trades.groupby(["session", "side"], sort=False):
        rows.append(
            {
                "session": str(session_name),
                "side": str(side_name),
                "support": int(len(group)),
                "net": float(group["pnl_net"].sum()),
            }
        )
    rows.sort(key=lambda item: (item["net"], item["support"]), reverse=True)
    return rows


def _candidate_pairs(sl_candidates: list[float], tp_candidates: list[float]) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for sl in sl_candidates:
        for tp in tp_candidates:
            sl_val = _round_points_to_tick(float(sl))
            tp_val = _round_points_to_tick(float(tp))
            if tp_val < max(sl_val, 1.5):
                continue
            pairs.append((float(sl_val), float(tp_val)))
    deduped = sorted({(sl, tp) for sl, tp in pairs})
    return deduped


def _build_selected_events_from_trade_log(trade_log: list[dict], df: pd.DataFrame, split_meta: dict) -> pd.DataFrame:
    if not trade_log:
        return pd.DataFrame(columns=["signal_idx", "selected_side", "session", "split", "combo_key"])
    trades = pd.DataFrame(trade_log)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(NY_TZ)
    trades["session"] = trades["session"].astype(str)
    trades["combo_key"] = trades["combo_key"].astype(str)
    trades["selected_side"] = np.where(trades["side"].astype(str).str.upper().eq("LONG"), 1, -1).astype(np.int8)
    trades["year"] = trades["entry_time"].dt.year.astype(int)
    valid_set = set(split_meta.get("valid_years", []))
    test_set = set(split_meta.get("test_years", []))
    trades["split"] = "train"
    trades.loc[trades["year"].isin(valid_set), "split"] = "valid"
    trades.loc[trades["year"].isin(test_set), "split"] = "test"

    index_positions = {ts: idx for idx, ts in enumerate(df.index)}
    signal_indices: list[int] = []
    keep_rows: list[bool] = []
    for row in trades.itertuples(index=False):
        entry_ts = pd.Timestamp(row.entry_time)
        pos = index_positions.get(entry_ts)
        if pos is None or int(pos) <= 0:
            keep_rows.append(False)
            signal_indices.append(-1)
            continue
        keep_rows.append(True)
        signal_indices.append(int(pos) - 1)
    trades = trades.loc[keep_rows].copy()
    trades["signal_idx"] = np.asarray([idx for idx, keep in zip(signal_indices, keep_rows) if keep], dtype=np.int32)
    return trades[["signal_idx", "selected_side", "session", "split", "combo_key"]].reset_index(drop=True)


def _optimization_score_args() -> argparse.Namespace:
    return argparse.Namespace(
        robust_sharpe_weight=1000.0,
        trade_count_weight=0.5,
        negative_year_penalty=300.0,
        worst_3y_weight=0.15,
        worst_5y_weight=0.35,
        max_drawdown_penalty=0.05,
        yearly_std_penalty=0.02,
    )


def _sltp_change_records(current_payload: dict, candidate_payload: dict) -> list[dict]:
    changes: list[dict] = []

    current_global = current_payload.get("global_default", {}) if isinstance(current_payload.get("global_default", {}), dict) else {}
    candidate_global = candidate_payload.get("global_default", {}) if isinstance(candidate_payload.get("global_default", {}), dict) else {}
    for side_name in ("LONG", "SHORT"):
        current_side = current_global.get(side_name, {}) if isinstance(current_global.get(side_name, {}), dict) else {}
        candidate_side = candidate_global.get(side_name, {}) if isinstance(candidate_global.get(side_name, {}), dict) else {}
        current_sl = float(current_side.get("sl", 0.0) or 0.0)
        current_tp = float(current_side.get("tp", 0.0) or 0.0)
        candidate_sl = float(candidate_side.get("sl", 0.0) or 0.0)
        candidate_tp = float(candidate_side.get("tp", 0.0) or 0.0)
        if abs(candidate_sl - current_sl) > 1e-9 or abs(candidate_tp - current_tp) > 1e-9:
            changes.append(
                {
                    "scope": "global",
                    "side": side_name,
                    "from_sl": current_sl,
                    "from_tp": current_tp,
                    "to_sl": candidate_sl,
                    "to_tp": candidate_tp,
                }
            )

    current_sessions = current_payload.get("session_defaults", {}) if isinstance(current_payload.get("session_defaults", {}), dict) else {}
    candidate_sessions = candidate_payload.get("session_defaults", {}) if isinstance(candidate_payload.get("session_defaults", {}), dict) else {}
    for session_name in sorted(set(current_sessions) | set(candidate_sessions)):
        current_side_map = current_sessions.get(session_name, {}) if isinstance(current_sessions.get(session_name, {}), dict) else {}
        candidate_side_map = candidate_sessions.get(session_name, {}) if isinstance(candidate_sessions.get(session_name, {}), dict) else {}
        for side_name in ("LONG", "SHORT"):
            current_side = current_side_map.get(side_name, {}) if isinstance(current_side_map.get(side_name, {}), dict) else {}
            candidate_side = candidate_side_map.get(side_name, {}) if isinstance(candidate_side_map.get(side_name, {}), dict) else {}
            current_sl = float(current_side.get("sl", 0.0) or 0.0)
            current_tp = float(current_side.get("tp", 0.0) or 0.0)
            candidate_sl = float(candidate_side.get("sl", 0.0) or 0.0)
            candidate_tp = float(candidate_side.get("tp", 0.0) or 0.0)
            if abs(candidate_sl - current_sl) > 1e-9 or abs(candidate_tp - current_tp) > 1e-9:
                changes.append(
                    {
                        "scope": "session",
                        "session": str(session_name),
                        "side": side_name,
                        "from_sl": current_sl,
                        "from_tp": current_tp,
                        "to_sl": candidate_sl,
                        "to_tp": candidate_tp,
                    }
                )
    return changes


def _evaluate_bracket_group(
    group: pd.DataFrame,
    sl_dist: float,
    tp_dist: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    max_hold_bars: int,
    point_value: float,
    fee_per_trade: float,
    gap_fills: bool,
    train_score_weight: float,
) -> dict:
    pnl = [
        _simulate_event_bracket(
            int(row.signal_idx),
            int(row.selected_side),
            float(sl_dist),
            float(tp_dist),
            opens,
            highs,
            lows,
            closes,
            int(max_hold_bars),
            float(point_value),
            float(fee_per_trade),
            bool(gap_fills),
        )
        for row in group.itertuples(index=False)
    ]
    trial = group[["split"]].copy()
    trial["pnl"] = pnl
    split_totals = trial.groupby("split")["pnl"].sum().to_dict()
    split_means = trial.groupby("split")["pnl"].mean().to_dict()
    oos_values = [float(split_means.get(name, 0.0)) for name in ("valid", "test") if name in split_means]
    robust_edge = min(oos_values) if oos_values else float(split_means.get("train", 0.0))
    return {
        "sl": float(sl_dist),
        "tp": float(tp_dist),
        "score": float(
            float(split_totals.get("test", 0.0))
            + (0.5 * float(split_totals.get("valid", 0.0)))
            + (float(train_score_weight) * float(split_totals.get("train", 0.0)))
        ),
        "train_total": float(split_totals.get("train", 0.0)),
        "valid_total": float(split_totals.get("valid", 0.0)),
        "test_total": float(split_totals.get("test", 0.0)),
        "robust_edge": float(robust_edge),
        "support": int(len(group)),
    }


def _fit_exact_overrides(
    selected_events: pd.DataFrame,
    payload: dict,
    max_hold_bars: int,
    point_value: float,
    fee_per_trade: float,
    train_score_weight: float,
    sl_candidates: list[float],
    tp_candidates: list[float],
    min_exact_trades: int,
    max_exact_overrides: int,
    min_exact_score_gain: float,
    min_exact_robust_edge: float,
    require_oos_improvement: bool,
    df: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    if selected_events.empty:
        return copy.deepcopy(payload), []

    artifact = RegimeAdaptiveArtifact(payload, Path("in_memory_regimeadaptive_sltp.json"))
    gap_fills = bool((CONFIG.get("BACKTEST_EXECUTION", {}) or {}).get("gap_fills", True))
    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    pairs = _candidate_pairs(sl_candidates, tp_candidates)
    if not pairs:
        return copy.deepcopy(payload), []

    candidates: list[dict] = []
    for (combo_key, selected_side), group in selected_events.groupby(["combo_key", "selected_side"], sort=False):
        if len(group) < int(min_exact_trades):
            continue
        combo_text = str(combo_key)
        side_value = int(selected_side)
        side_name = "LONG" if side_value > 0 else "SHORT"
        session_name = str(group["session"].iloc[0])
        current_sltp = artifact.get_sltp(side_name, combo_text, session_name)
        current_eval = _evaluate_bracket_group(
            group,
            float(current_sltp.get("sl_dist", 2.0) or 2.0),
            float(current_sltp.get("tp_dist", 3.0) or 3.0),
            opens,
            highs,
            lows,
            closes,
            max_hold_bars,
            point_value,
            fee_per_trade,
            gap_fills,
            train_score_weight,
        )
        best_eval = dict(current_eval)
        for sl_dist, tp_dist in pairs:
            if abs(float(sl_dist) - float(current_eval["sl"])) < 1e-9 and abs(float(tp_dist) - float(current_eval["tp"])) < 1e-9:
                continue
            trial_eval = _evaluate_bracket_group(
                group,
                float(sl_dist),
                float(tp_dist),
                opens,
                highs,
                lows,
                closes,
                max_hold_bars,
                point_value,
                fee_per_trade,
                gap_fills,
                train_score_weight,
            )
            if float(trial_eval["score"]) > float(best_eval["score"]) or (
                abs(float(trial_eval["score"]) - float(best_eval["score"])) <= 1e-9
                and float(trial_eval["robust_edge"]) > float(best_eval["robust_edge"])
            ):
                best_eval = dict(trial_eval)
        score_gain = float(best_eval["score"]) - float(current_eval["score"])
        oos_gain = float(best_eval["valid_total"] + best_eval["test_total"]) - float(current_eval["valid_total"] + current_eval["test_total"])
        if score_gain <= float(min_exact_score_gain):
            continue
        if float(best_eval["robust_edge"]) < float(min_exact_robust_edge):
            continue
        if bool(require_oos_improvement) and oos_gain <= 0.0:
            continue
        candidates.append(
            {
                "scope": "exact",
                "combo_key": combo_text,
                "session": session_name,
                "side": side_name,
                "support": int(len(group)),
                "from_sl": float(current_eval["sl"]),
                "from_tp": float(current_eval["tp"]),
                "to_sl": float(best_eval["sl"]),
                "to_tp": float(best_eval["tp"]),
                "score_gain": float(score_gain),
                "oos_gain": float(oos_gain),
                "robust_edge_before": float(current_eval["robust_edge"]),
                "robust_edge_after": float(best_eval["robust_edge"]),
                "valid_total_before": float(current_eval["valid_total"]),
                "valid_total_after": float(best_eval["valid_total"]),
                "test_total_before": float(current_eval["test_total"]),
                "test_total_after": float(best_eval["test_total"]),
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item["score_gain"]),
            float(item["oos_gain"]),
            float(item["robust_edge_after"]),
            int(item["support"]),
        ),
        reverse=True,
    )
    selected = candidates[: max(0, int(max_exact_overrides))]
    if not selected:
        return copy.deepcopy(payload), []

    next_payload = copy.deepcopy(payload)
    combo_policies = next_payload.setdefault("combo_policies", {})
    for item in selected:
        combo_entry = combo_policies.get(str(item["combo_key"]), {})
        combo_entry = dict(combo_entry) if isinstance(combo_entry, dict) else {}
        combo_entry[str(item["side"])] = {
            "sl": float(item["to_sl"]),
            "tp": float(item["to_tp"]),
            "support": int(item["support"]),
        }
        combo_policies[str(item["combo_key"])] = combo_entry
    return next_payload, selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize RegimeAdaptive SL/TP defaults from executed trades, then validate with the direct simulator."
    )
    parser.add_argument("--artifact", default="artifacts/regimeadaptive_v5/latest.json")
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--opt-start", default="2015-01-01")
    parser.add_argument("--opt-end", default="2024-12-31")
    parser.add_argument("--valid-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--sl-candidates", default="1.5,2.0,3.0,4.0,5.0")
    parser.add_argument("--tp-candidates", default="3.0,4.0,5.0,6.0,8.0")
    parser.add_argument("--min-bracket-trades", type=int, default=25)
    parser.add_argument("--exact-min-trades", type=int, default=40)
    parser.add_argument("--max-exact-overrides", type=int, default=4)
    parser.add_argument("--min-exact-score-gain", type=float, default=25.0)
    parser.add_argument("--min-exact-robust-edge", type=float, default=0.0)
    parser.add_argument("--disable-oos-improvement-filter", action="store_true")
    parser.add_argument("--max-passes", type=int, default=2)
    parser.add_argument("--max-hold-bars", type=int, default=0)
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--write-latest", action="store_true")
    args = parser.parse_args()

    artifact = load_regimeadaptive_artifact(str(args.artifact))
    if artifact is None:
        raise SystemExit(f"Artifact could not be loaded: {args.artifact}")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Source parquet not found: {source}")

    start_time = _parse_datetime(args.opt_start, is_end=False)
    end_time = _parse_datetime(args.opt_end, is_end=True)
    df, symbol_label = _load_bars(source, str(args.symbol_mode), str(args.symbol_method))
    df = df[(df.index >= start_time) & (df.index <= end_time)].copy()
    if df.empty:
        raise SystemExit("No data in optimization range.")

    combo_ids, session_codes = _build_combo_arrays(df.index)
    holiday_mask = _build_holiday_mask(df.index, session_codes)
    _, _, split_meta = _build_split_map(df.index, int(args.valid_years), int(args.test_years))

    rule_catalog = getattr(artifact, "rule_catalog", {}) or {}
    if not rule_catalog:
        base_rule = getattr(artifact, "base_rule", {}) or {}
        rule_catalog = {"__base__": dict(base_rule)}
    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    sma_windows = sorted(
        {
            int(rule_payload.get("sma_fast", 20) or 20)
            for rule_payload in rule_catalog.values()
        }
        | {
            int(rule_payload.get("sma_slow", 200) or 200)
            for rule_payload in rule_catalog.values()
        }
    )
    rolling_cache = _rolling_cache(close, sma_windows)
    atr_periods = sorted({int(rule_payload.get("atr_period", 20) or 20) for rule_payload in rule_catalog.values()})
    atr_cache = {period: _atr_array(high, low, close, period) for period in atr_periods}

    rule_long_strength: dict[str, np.ndarray] = {}
    rule_short_strength: dict[str, np.ndarray] = {}
    for rule_id, rule_payload in rule_catalog.items():
        long_strength, short_strength = _build_rule_strength_arrays(
            session_codes,
            close,
            rolling_cache[int(rule_payload.get("sma_fast", 20) or 20)],
            rolling_cache[int(rule_payload.get("sma_slow", 200) or 200)],
            atr_cache[int(rule_payload.get("atr_period", 20) or 20)],
            float(rule_payload.get("cross_atr_mult", 0.1) or 0.1),
        )
        rule_long_strength[str(rule_id)] = long_strength
        rule_short_strength[str(rule_id)] = short_strength

    risk_cfg = CONFIG.get("RISK", {}) or {}
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
    fee_per_side = float(risk_cfg.get("FEES_PER_SIDE", 0.37) or 0.37)
    fee_per_contract_rt = fee_per_side * 2.0
    fee_per_trade = fee_per_contract_rt
    score_args = _optimization_score_args()
    train_score_weight = float(((artifact.payload.get("training_config", {}) or {}).get("train_score_weight", 0.25) or 0.25))
    sl_candidates = _parse_float_list(args.sl_candidates)
    tp_candidates = _parse_float_list(args.tp_candidates)
    inferred_max_hold_bars = max(
        int((rule_payload or {}).get("max_hold_bars", 30) or 30)
        for rule_payload in rule_catalog.values()
    )
    max_hold_bars = int(args.max_hold_bars) if int(args.max_hold_bars) > 0 else inferred_max_hold_bars

    current_payload = copy.deepcopy(artifact.payload)
    current_result = _simulate_payload_v3(
        df,
        combo_ids,
        session_codes,
        holiday_mask,
        current_payload,
        rule_long_strength,
        rule_short_strength,
        start_time,
        end_time,
        int(args.contracts),
        point_value,
        fee_per_contract_rt,
    )
    current_eval = _evaluate_trade_log_v5(
        current_result.get("trade_log", []) or [],
        split_meta,
        train_score_weight,
        start_time,
        end_time,
        score_args,
    )
    baseline_eval = dict(current_eval)

    accepted_actions: list[dict] = []

    for pass_idx in range(int(args.max_passes)):
        selected_events = _build_selected_events_from_trade_log(current_result.get("trade_log", []) or [], df, split_meta)
        if selected_events.empty:
            break
        working_payload = copy.deepcopy(current_payload)
        fitted_global_default, fitted_session_defaults = _fit_bracket_defaults(
            selected_events,
            df,
            max_hold_bars,
            point_value,
            fee_per_trade,
            int(args.min_bracket_trades),
            sl_candidates,
            tp_candidates,
        )
        working_payload["global_default"] = copy.deepcopy(fitted_global_default)
        working_payload["session_defaults"] = copy.deepcopy(fitted_session_defaults)
        change_records = _sltp_change_records(current_payload, working_payload)
        candidate_payload, exact_override_records = _fit_exact_overrides(
            selected_events,
            working_payload,
            max_hold_bars,
            point_value,
            fee_per_trade,
            train_score_weight,
            sl_candidates,
            tp_candidates,
            int(args.exact_min_trades),
            int(args.max_exact_overrides),
            float(args.min_exact_score_gain),
            float(args.min_exact_robust_edge),
            not bool(args.disable_oos_improvement_filter),
            df,
        )
        change_records.extend(exact_override_records)
        if not change_records:
            break
        candidate_result = _simulate_payload_v3(
            df,
            combo_ids,
            session_codes,
            holiday_mask,
            candidate_payload,
            rule_long_strength,
            rule_short_strength,
            start_time,
            end_time,
            int(args.contracts),
            point_value,
            fee_per_contract_rt,
        )
        candidate_eval = _evaluate_trade_log_v5(
            candidate_result.get("trade_log", []) or [],
            split_meta,
            train_score_weight,
            start_time,
            end_time,
            score_args,
        )
        if float(candidate_eval["score"]) <= float(current_eval["score"]):
            break
        accepted_actions.append(
            {
                "pass": int(pass_idx + 1),
                "changes": change_records,
                "score_before": float(current_eval["score"]),
                "score_after": float(candidate_eval["score"]),
                "daily_sharpe_before": float(current_eval["daily_sharpe"]),
                "daily_sharpe_after": float(candidate_eval["daily_sharpe"]),
                "worst_5y_pnl_before": float(current_eval["worst_5y_pnl"]),
                "worst_5y_pnl_after": float(candidate_eval["worst_5y_pnl"]),
                "negative_years_before": int(current_eval["negative_years"]),
                "negative_years_after": int(candidate_eval["negative_years"]),
                "trades_before": int(current_eval["trades"]),
                "trades_after": int(candidate_eval["trades"]),
            }
        )
        current_payload = candidate_payload
        current_result = candidate_result
        current_eval = candidate_eval

    if str(args.artifact_root or "").strip():
        artifact_root = Path(args.artifact_root).expanduser().resolve()
    else:
        artifact_root = ROOT / "artifacts" / f"regimeadaptive_sltp_{dt.datetime.now(NY_TZ).strftime('%Y%m%d_%H%M%S')}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / "regimeadaptive_sltp_artifact.json"
    current_payload["created_at"] = dt.datetime.now(NY_TZ).isoformat()
    current_payload.setdefault("metadata", {})
    current_payload["metadata"]["sltp_optimization"] = {
        "base_artifact": str(getattr(artifact, "path", args.artifact)),
        "opt_range_start": start_time.isoformat(),
        "opt_range_end": end_time.isoformat(),
        "min_bracket_trades": int(args.min_bracket_trades),
        "max_hold_bars": int(max_hold_bars),
        "exact_min_trades": int(args.exact_min_trades),
        "max_exact_overrides": int(args.max_exact_overrides),
        "min_exact_score_gain": float(args.min_exact_score_gain),
        "min_exact_robust_edge": float(args.min_exact_robust_edge),
        "require_oos_improvement": bool(not args.disable_oos_improvement_filter),
        "sl_candidates": [float(value) for value in sl_candidates],
        "tp_candidates": [float(value) for value in tp_candidates],
        "accepted_actions": accepted_actions,
        "baseline_candidate": {
            "score": float(baseline_eval["score"]),
            "daily_sharpe": float(baseline_eval["daily_sharpe"]),
            "negative_years": int(baseline_eval["negative_years"]),
            "worst_3y_pnl": float(baseline_eval["worst_3y_pnl"]),
            "worst_5y_pnl": float(baseline_eval["worst_5y_pnl"]),
            "max_drawdown": float(baseline_eval["max_drawdown"]),
            "trades": int(baseline_eval["trades"]),
        },
        "best_candidate": {
            "score": float(current_eval["score"]),
            "daily_sharpe": float(current_eval["daily_sharpe"]),
            "negative_years": int(current_eval["negative_years"]),
            "worst_3y_pnl": float(current_eval["worst_3y_pnl"]),
            "worst_5y_pnl": float(current_eval["worst_5y_pnl"]),
            "max_drawdown": float(current_eval["max_drawdown"]),
            "trades": int(current_eval["trades"]),
        },
    }
    artifact_path.write_text(json.dumps(current_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        "baseline_score={:.4f} optimized_score={:.4f} baseline_sharpe={:.4f} optimized_sharpe={:.4f} baseline_trades={} optimized_trades={}".format(
            float(baseline_eval["score"]),
            float(current_eval["score"]),
            float(baseline_eval["daily_sharpe"]),
            float(current_eval["daily_sharpe"]),
            int(baseline_eval["trades"]),
            int(current_eval["trades"]),
        )
    )
    print(f"artifact={artifact_path}")
    if bool(args.write_latest):
        latest_dir = ROOT / "artifacts" / "regimeadaptive_sltp"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_path = latest_dir / "latest.json"
        shutil.copyfile(artifact_path, latest_path)
        print(f"latest={latest_path}")


if __name__ == "__main__":
    main()
