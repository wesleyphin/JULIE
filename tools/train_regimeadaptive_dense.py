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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from regimeadaptive_artifact import load_regimeadaptive_artifact
from tools.backtest_regimeadaptive_robust import (
    _build_rule_strength_arrays,
    _parse_datetime,
    _rolling_extrema_cache,
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
from tools.train_regimeadaptive_v2 import _json_safe, _summarize_side_groups
from tools.train_regimeadaptive_v3 import (
    _merge_combo_policies,
    _parse_rule_specs,
    _simulate_payload_v3,
)
from tools.train_regimeadaptive_v5 import _evaluate_trade_log_v5


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw or "").split(","):
        text = str(item or "").strip()
        if text:
            out.append(int(text))
    return out


def _parse_text_list(raw: str) -> list[str]:
    out: list[str] = []
    for item in str(raw or "").split(","):
        text = str(item or "").strip().lower()
        if text:
            out.append(text)
    return out


def _resolve_path(path_text: str, default_relative: str) -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _combo_session_name(combo_key: str) -> str:
    parts = [str(part or "").strip().upper() for part in str(combo_key or "").split("_")]
    if len(parts) < 4:
        return ""
    return "_".join(parts[3:])


def _parse_session_top_n(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in str(raw or "").split(","):
        text = str(item or "").strip()
        if not text or "=" not in text:
            continue
        name_text, count_text = text.split("=", 1)
        session_name = str(name_text or "").strip().upper()
        if not session_name:
            continue
        out[session_name] = max(0, int(count_text))
    return out


def _parse_session_side_top_n(raw: str) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for item in str(raw or "").split(","):
        text = str(item or "").strip()
        if not text or "=" not in text or ":" not in text:
            continue
        left_text, count_text = text.split("=", 1)
        session_text, side_text = left_text.split(":", 1)
        session_name = str(session_text or "").strip().upper()
        side_name = str(side_text or "").strip().upper()
        if not session_name or side_name not in {"LONG", "SHORT"}:
            continue
        out[(session_name, side_name)] = max(0, int(count_text))
    return out


def _seed_signal_policies_from_artifact(baseline_artifact) -> dict[str, dict[str, dict]]:
    seeded: dict[str, dict[str, dict]] = {}
    signal_policies = baseline_artifact.payload.get("signal_policies", {}) if isinstance(baseline_artifact.payload.get("signal_policies", {}), dict) else {}
    for combo_key, side_map in signal_policies.items():
        if not isinstance(side_map, dict):
            continue
        for original_signal, record in side_map.items():
            if not isinstance(record, dict):
                continue
            seeded.setdefault(str(combo_key), {})[str(original_signal).upper()] = copy.deepcopy(record)
    return seeded


def _merged_rule_catalog(baseline_artifact, rule_catalog: dict[str, dict]) -> tuple[dict[str, dict], str]:
    merged_catalog = copy.deepcopy(baseline_artifact.payload.get("rule_catalog", {}) or {})
    baseline_default_rule_id = str(baseline_artifact.payload.get("default_rule_id", "") or "").strip()
    if baseline_default_rule_id and baseline_default_rule_id not in merged_catalog:
        base_rule = baseline_artifact.payload.get("base_rule", {}) if isinstance(baseline_artifact.payload.get("base_rule", {}), dict) else {}
        if base_rule:
            merged_catalog[baseline_default_rule_id] = copy.deepcopy(base_rule)
    for rule_id, rule_payload in rule_catalog.items():
        merged_catalog[str(rule_id)] = copy.deepcopy(rule_payload)
    default_rule_id = baseline_default_rule_id if baseline_default_rule_id in merged_catalog else next(iter(rule_catalog.keys()))
    return merged_catalog, str(default_rule_id)


def _trade_day_coverage_stats(
    trade_log: list[dict],
    *,
    trading_days: int,
) -> dict[str, float | int]:
    if not trade_log or int(trading_days) <= 0:
        return {
            "trade_days_covered": 0,
            "trade_day_coverage": 0.0,
            "avg_trades_per_trade_day": 0.0,
        }

    entry_times = pd.to_datetime(
        [trade.get("entry_time") for trade in trade_log],
        utc=True,
        errors="coerce",
    )
    if len(entry_times) == 0:
        return {
            "trade_days_covered": 0,
            "trade_day_coverage": 0.0,
            "avg_trades_per_trade_day": 0.0,
        }
    entry_days = pd.Index(entry_times.tz_convert(NY_TZ).normalize()).nunique()
    coverage = float(entry_days) / float(trading_days) if int(trading_days) > 0 else 0.0
    avg_trades_per_trade_day = float(len(trade_log)) / float(entry_days) if int(entry_days) > 0 else 0.0
    return {
        "trade_days_covered": int(entry_days),
        "trade_day_coverage": float(coverage),
        "avg_trades_per_trade_day": float(avg_trades_per_trade_day),
    }


def _dense_selection_score(eval_metrics: dict, coverage_stats: dict, args) -> float:
    score = float(eval_metrics.get("score", float("-inf")) or float("-inf"))
    score += float(args.trade_day_coverage_weight) * float(coverage_stats.get("trade_day_coverage", 0.0) or 0.0)
    return float(score)


def _universal_probe_payload(
    rule_id: str,
    rule_payload: dict,
    rule_catalog: dict[str, dict],
    policy: str,
    baseline_artifact,
    early_exit_enabled: bool,
) -> dict:
    return {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "version": "regimeadaptive_dense_universal_probe",
        "policy_mode": "side_specific_execution_multirule_hierarchical",
        "base_rule": dict(rule_payload),
        "rule_catalog": copy.deepcopy(rule_catalog),
        "default_rule_id": str(rule_id),
        "combo_policies": {},
        "signal_policies": {},
        "group_signal_policies": {
            "ALL_ALL_ALL_ALL": {
                "LONG": {
                    "policy": str(policy),
                    "early_exit_enabled": bool(early_exit_enabled),
                    "rule_id": str(rule_id),
                },
                "SHORT": {
                    "policy": str(policy),
                    "early_exit_enabled": bool(early_exit_enabled),
                    "rule_id": str(rule_id),
                },
            }
        },
        "session_defaults": copy.deepcopy(baseline_artifact.payload.get("session_defaults", {}) or {}),
        "global_default": copy.deepcopy(baseline_artifact.payload.get("global_default", {}) or {}),
        "metadata": {
            "stage": "dense_universal_probe",
            "rule_id": str(rule_id),
            "policy": str(policy),
        },
    }


def _candidate_rank_columns(metric: str) -> tuple[str, list[str]]:
    primary = str(metric or "score").strip().lower()
    if primary not in {"score", "robust_edge", "test_total", "valid_total"}:
        primary = "score"
    return primary, [primary, "score", "test_total", "valid_total", "support_total"]


def _candidate_pool(rows: list[dict], rank_metric: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    primary, sort_cols = _candidate_rank_columns(rank_metric)
    candidate_df = pd.DataFrame.from_records(rows)
    if candidate_df.empty:
        return candidate_df
    candidate_df = candidate_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    candidate_df = candidate_df.drop_duplicates(["combo_key", "original_signal"], keep="first")
    candidate_df = candidate_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
    candidate_df["_rank_metric"] = primary
    return candidate_df


def _seeded_side_keys(seed_signal_policies: dict[str, dict[str, dict]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for combo_key, side_map in seed_signal_policies.items():
        if not isinstance(side_map, dict):
            continue
        for original_signal in side_map.keys():
            out.add((str(combo_key), str(original_signal).upper()))
    return out


def _select_candidate_frame(
    candidate_df: pd.DataFrame,
    *,
    top_n: int,
    session_top_n: dict[str, int],
    session_side_top_n: dict[tuple[str, str], int],
) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df
    if not session_top_n and not session_side_top_n:
        return candidate_df.head(int(top_n)).copy()

    selected_parts: list[pd.DataFrame] = []
    if session_side_top_n:
        for (session_name, side_name), limit in session_side_top_n.items():
            if int(limit) <= 0:
                continue
            side_df = candidate_df.loc[
                (candidate_df["session_name"].astype(str).str.upper() == str(session_name).upper())
                & (candidate_df["original_signal"].astype(str).str.upper() == str(side_name).upper())
            ]
            if side_df.empty:
                continue
            selected_parts.append(side_df.head(int(limit)).copy())
    elif session_top_n:
        for session_name, limit in session_top_n.items():
            if int(limit) <= 0:
                continue
            session_df = candidate_df.loc[candidate_df["session_name"].astype(str).str.upper() == str(session_name).upper()]
            if session_df.empty:
                continue
            selected_parts.append(session_df.head(int(limit)).copy())
    if not selected_parts:
        return candidate_df.iloc[0:0].copy()
    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df = selected_df.drop_duplicates(["combo_key", "original_signal"], keep="first").reset_index(drop=True)
    return selected_df


def _build_selected_payload(
    baseline_artifact,
    rule_catalog: dict[str, dict],
    default_rule_id: str,
    selected_df: pd.DataFrame,
    early_exit_enabled: bool,
    metadata: dict,
    seed_signal_policies: dict[str, dict[str, dict]] | None = None,
) -> dict:
    signal_policies: dict[str, dict[str, dict]] = copy.deepcopy(seed_signal_policies or {})
    for row in selected_df.itertuples(index=False):
        signal_policies.setdefault(str(row.combo_key), {})[str(row.original_signal)] = {
            "policy": str(row.policy),
            "early_exit_enabled": bool(early_exit_enabled),
            "rule_id": str(row.rule_id),
        }
    combo_policies = _merge_combo_policies({}, signal_policies)
    return {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "version": "regimeadaptive_dense_exact",
        "policy_mode": "side_specific_execution_multirule_hierarchical",
        "base_rule": dict(rule_catalog[str(default_rule_id)]),
        "rule_catalog": copy.deepcopy(rule_catalog),
        "default_rule_id": str(default_rule_id),
        "combo_policies": combo_policies,
        "signal_policies": signal_policies,
        "group_signal_policies": {},
        "session_defaults": copy.deepcopy(baseline_artifact.payload.get("session_defaults", {}) or {}),
        "global_default": copy.deepcopy(baseline_artifact.payload.get("global_default", {}) or {}),
        "seed_artifact_path": str(getattr(baseline_artifact, "path", "")),
        "metadata": dict(metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a dense RegimeAdaptive artifact from universal exact bucket probes."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--baseline-artifact", default="artifacts/regimeadaptive_v13/latest.json")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--rule-specs", default="8:21:0.0,13:34:0.0")
    parser.add_argument("--valid-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--disable-early-exit", action="store_true")
    parser.add_argument("--universal-policies", default="normal,reversed")
    parser.add_argument("--min-total-support", type=int, default=150)
    parser.add_argument("--min-recent-support", type=int, default=15)
    parser.add_argument("--min-positive-oos-splits", type=int, default=1)
    parser.add_argument("--min-robust-edge", type=float, default=0.0)
    parser.add_argument("--rank-metrics", default="score,robust_edge")
    parser.add_argument("--top-n-grid", default="24,36,48,60,72,84")
    parser.add_argument("--min-target-trades", type=int, default=4674)
    parser.add_argument("--min-trade-days-covered", type=int, default=0)
    parser.add_argument("--min-trade-day-coverage", type=float, default=0.0)
    parser.add_argument("--trade-day-coverage-weight", type=float, default=0.0)
    parser.add_argument("--seed-baseline-exact", action="store_true")
    parser.add_argument("--session-top-n", default="")
    parser.add_argument("--session-side-top-n", default="")
    parser.add_argument("--candidate-pool-output", default="")
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
    baseline_artifact = load_regimeadaptive_artifact(str(args.baseline_artifact))
    if baseline_artifact is None:
        raise SystemExit(f"Baseline artifact could not be loaded: {args.baseline_artifact}")

    start_time = _parse_datetime(args.start, is_end=False)
    end_time = _parse_datetime(args.end, is_end=True)
    df, symbol_label = _load_bars(source, str(args.symbol_mode), str(args.symbol_method))
    df = df[(df.index >= start_time) & (df.index <= end_time)].copy()
    if df.empty:
        raise SystemExit("No data in requested range.")

    combo_ids, session_codes = _build_combo_arrays(df.index)
    holiday_mask = _build_holiday_mask(df.index, session_codes)
    _, _, split_meta = _build_split_map(df.index, int(args.valid_years), int(args.test_years))
    trading_days = int(pd.Index(df.index.normalize()).nunique())
    hours = np.fromiter((ts.hour for ts in df.index), dtype=np.int8, count=len(df.index))
    minutes = np.fromiter((ts.minute for ts in df.index), dtype=np.int8, count=len(df.index))
    test_positions = np.arange(len(df.index), dtype=np.int32)

    base_rule = baseline_artifact.base_rule or {}
    atr_period = int(base_rule.get("atr_period", 20) or 20)
    rule_catalog = _parse_rule_specs(
        str(args.rule_specs),
        atr_period,
        int(base_rule.get("max_hold_bars", 30) or 30),
    )
    if not rule_catalog:
        raise SystemExit("No rule specs were provided.")
    seed_signal_policies = _seed_signal_policies_from_artifact(baseline_artifact) if bool(args.seed_baseline_exact) else {}
    seeded_side_keys = _seeded_side_keys(seed_signal_policies)
    rule_catalog, default_rule_id = _merged_rule_catalog(baseline_artifact, rule_catalog)
    session_top_n = _parse_session_top_n(str(args.session_top_n))
    session_side_top_n = _parse_session_side_top_n(str(args.session_side_top_n))

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
    atr_cache = {int(atr_period): _atr_array(high, low, close, int(atr_period))}
    pattern_lookbacks = sorted(
        {
            max(1, int(rule_payload.get("pattern_lookback", 8) or 8))
            for rule_payload in rule_catalog.values()
            if str(rule_payload.get("rule_type", "pullback") or "pullback").strip().lower() in {"continuation", "breakout"}
        }
    )
    rolling_high_cache = _rolling_extrema_cache(high, pattern_lookbacks, "max") if pattern_lookbacks else {}
    rolling_low_cache = _rolling_extrema_cache(low, pattern_lookbacks, "min") if pattern_lookbacks else {}

    risk_cfg = CONFIG.get("RISK", {}) or {}
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
    fee_per_side = float(risk_cfg.get("FEES_PER_SIDE", 0.37) or 0.37)
    fee_per_contract_rt = fee_per_side * 2.0

    rule_long_strength: dict[str, np.ndarray] = {}
    rule_short_strength: dict[str, np.ndarray] = {}
    for rule_id, rule_payload in rule_catalog.items():
        long_strength, short_strength = _build_rule_strength_arrays(
            session_codes,
            close,
            high,
            low,
            rolling_cache,
            atr_cache,
            rolling_high_cache,
            rolling_low_cache,
            rule_payload,
        )
        rule_long_strength[str(rule_id)] = long_strength
        rule_short_strength[str(rule_id)] = short_strength
    rule_order = list(rule_catalog.keys())
    prebuilt_long_strength_matrix = np.vstack(
        [rule_long_strength[str(rule_id)] for rule_id in rule_order]
    ).astype(np.float32)
    prebuilt_short_strength_matrix = np.vstack(
        [rule_short_strength[str(rule_id)] for rule_id in rule_order]
    ).astype(np.float32)

    candidate_rows: list[dict] = []
    probe_summaries: list[dict] = []
    for rule_id, rule_payload in rule_catalog.items():
        for policy in _parse_text_list(str(args.universal_policies)):
            if policy not in {"normal", "reversed"}:
                continue
            payload = _universal_probe_payload(
                str(rule_id),
                rule_payload,
                rule_catalog,
                policy,
                baseline_artifact,
                not bool(args.disable_early_exit),
            )
            result = _simulate_payload_v3(
                df,
                combo_ids,
                session_codes,
                holiday_mask,
                payload,
                rule_long_strength,
                rule_short_strength,
                start_time,
                end_time,
                int(args.contracts),
                point_value,
                fee_per_contract_rt,
                hours=hours,
                minutes=minutes,
                test_positions=test_positions,
                prebuilt_rule_order=rule_order,
                prebuilt_long_strength_matrix=prebuilt_long_strength_matrix,
                prebuilt_short_strength_matrix=prebuilt_short_strength_matrix,
            )
            trade_log = result.get("trade_log", []) or []
            probe_eval = _evaluate_trade_log_v5(
                trade_log,
                split_meta,
                float(args.train_score_weight),
                start_time,
                end_time,
                args,
            )
            side_stats = _summarize_side_groups(trade_log, split_meta)
            probe_summaries.append(
                {
                    "rule_id": str(rule_id),
                    "policy": str(policy),
                    "trades": int(probe_eval["trades"]),
                    "score": float(probe_eval["score"]),
                    "daily_sharpe": float(probe_eval["daily_sharpe"]),
                    "negative_years": int(probe_eval["negative_years"]),
                }
            )
            if side_stats.empty:
                continue
            for row in side_stats.itertuples(index=False):
                if int(row.support_total) < int(args.min_total_support):
                    continue
                if int(row.support_recent) < int(args.min_recent_support):
                    continue
                if int(row.positive_oos_splits) < int(args.min_positive_oos_splits):
                    continue
                if float(row.robust_edge) <= float(args.min_robust_edge):
                    continue
                candidate_rows.append(
                    {
                        "combo_key": str(row.combo_key),
                        "session_name": _combo_session_name(str(row.combo_key)),
                        "original_signal": str(row.original_signal),
                        "policy": str(policy),
                        "rule_id": str(rule_id),
                        "score": float(row.test_total) + (0.5 * float(row.valid_total)) + (float(args.train_score_weight) * float(row.train_total)),
                        "train_total": float(row.train_total),
                        "valid_total": float(row.valid_total),
                        "test_total": float(row.test_total),
                        "train_mean": float(row.train_mean),
                        "valid_mean": float(row.valid_mean),
                        "test_mean": float(row.test_mean),
                        "robust_edge": float(row.robust_edge),
                        "support_total": int(row.support_total),
                        "support_recent": int(row.support_recent),
                        "positive_oos_splits": int(row.positive_oos_splits),
                    }
                )
            print(
                f"probe rule_id={rule_id} policy={policy} trades={int(probe_eval['trades'])} "
                f"score={float(probe_eval['score']):.2f}"
            )

    if not candidate_rows:
        raise SystemExit("No dense candidates survived the minimum-support filters.")

    candidate_pool_output = _resolve_path(str(args.candidate_pool_output), "") if str(args.candidate_pool_output).strip() else None
    if candidate_pool_output is not None:
        candidate_pool_output.parent.mkdir(parents=True, exist_ok=True)
        candidate_pool_output.write_text(
            json.dumps(_json_safe({"rows": candidate_rows}), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    rank_metrics = _parse_text_list(str(args.rank_metrics))
    top_n_values = _parse_int_list(str(args.top_n_grid))
    best_payload = None
    best_eval = None
    best_selection_meta = None
    selected_trade_log: list[dict] = []

    for rank_metric in rank_metrics:
        candidate_df = _candidate_pool(candidate_rows, rank_metric)
        if candidate_df.empty:
            continue
        if seeded_side_keys:
            candidate_df = candidate_df.loc[
                ~candidate_df.apply(
                    lambda row: (str(row["combo_key"]), str(row["original_signal"]).upper()) in seeded_side_keys,
                    axis=1,
                )
            ].reset_index(drop=True)
        if candidate_df.empty:
            continue
        if session_side_top_n:
            selection_top_n_values = [sum(session_side_top_n.values())]
        elif session_top_n:
            selection_top_n_values = [sum(session_top_n.values())]
        else:
            selection_top_n_values = top_n_values
        for top_n in selection_top_n_values:
            selected_df = _select_candidate_frame(
                candidate_df,
                top_n=int(top_n),
                session_top_n=session_top_n,
                session_side_top_n=session_side_top_n,
            )
            if selected_df.empty:
                continue
            payload = _build_selected_payload(
                baseline_artifact,
                rule_catalog,
                default_rule_id,
                selected_df,
                not bool(args.disable_early_exit),
                metadata={
                    "stage": "dense_selection",
                    "rank_metric": str(rank_metric),
                    "top_n": int(top_n),
                    "candidate_count": int(len(selected_df)),
                    "session_top_n": {key: int(value) for key, value in session_top_n.items()},
                    "session_side_top_n": {
                        f"{session_name}:{side_name}": int(value)
                        for (session_name, side_name), value in session_side_top_n.items()
                    },
                    "seed_baseline_exact": bool(args.seed_baseline_exact),
                    "seeded_side_count": int(len(seeded_side_keys)),
                    "trading_days_in_range": int(trading_days),
                },
                seed_signal_policies=seed_signal_policies,
            )
            result = _simulate_payload_v3(
                df,
                combo_ids,
                session_codes,
                holiday_mask,
                payload,
                rule_long_strength,
                rule_short_strength,
                start_time,
                end_time,
                int(args.contracts),
                point_value,
                fee_per_contract_rt,
                hours=hours,
                minutes=minutes,
                test_positions=test_positions,
                prebuilt_rule_order=rule_order,
                prebuilt_long_strength_matrix=prebuilt_long_strength_matrix,
                prebuilt_short_strength_matrix=prebuilt_short_strength_matrix,
            )
            trade_log = result.get("trade_log", []) or []
            eval_metrics = _evaluate_trade_log_v5(
                trade_log,
                split_meta,
                float(args.train_score_weight),
                start_time,
                end_time,
                args,
            )
            coverage_stats = _trade_day_coverage_stats(
                trade_log,
                trading_days=int(trading_days),
            )
            if int(eval_metrics["trades"]) < int(args.min_target_trades):
                print(
                    f"skip rank_metric={rank_metric} top_n={top_n} trades={int(eval_metrics['trades'])} "
                    f"< target={int(args.min_target_trades)}"
                )
                continue
            if int(coverage_stats["trade_days_covered"]) < int(args.min_trade_days_covered):
                print(
                    f"skip rank_metric={rank_metric} top_n={top_n} trade_days={int(coverage_stats['trade_days_covered'])} "
                    f"< min_trade_days={int(args.min_trade_days_covered)}"
                )
                continue
            if float(coverage_stats["trade_day_coverage"]) + 1e-12 < float(args.min_trade_day_coverage):
                print(
                    f"skip rank_metric={rank_metric} top_n={top_n} trade_day_coverage={float(coverage_stats['trade_day_coverage']):.4f} "
                    f"< min_coverage={float(args.min_trade_day_coverage):.4f}"
                )
                continue
            selection_score = _dense_selection_score(eval_metrics, coverage_stats, args)
            current_best_score = (
                float(best_selection_meta.get("selection_score"))
                if isinstance(best_selection_meta, dict) and best_selection_meta.get("selection_score") is not None
                else float("-inf")
            )
            if best_eval is None or float(selection_score) > float(current_best_score):
                best_payload = payload
                best_eval = eval_metrics
                best_selection_meta = {
                    "rank_metric": str(rank_metric),
                    "top_n": int(top_n),
                    "candidate_count": int(len(selected_df)),
                    "session_top_n": {key: int(value) for key, value in session_top_n.items()},
                    "session_side_top_n": {
                        f"{session_name}:{side_name}": int(value)
                        for (session_name, side_name), value in session_side_top_n.items()
                    },
                    "seed_baseline_exact": bool(args.seed_baseline_exact),
                    "seeded_side_count": int(len(seeded_side_keys)),
                    "selection_score": float(selection_score),
                    "trade_day_coverage": float(coverage_stats["trade_day_coverage"]),
                    "trade_days_covered": int(coverage_stats["trade_days_covered"]),
                    "avg_trades_per_trade_day": float(coverage_stats["avg_trades_per_trade_day"]),
                    "selected_by_session": {
                        str(key): int(value)
                        for key, value in selected_df["session_name"].astype(str).value_counts().to_dict().items()
                    },
                    "selected_candidates": selected_df.to_dict(orient="records"),
                }
                selected_trade_log = trade_log
            print(
                f"candidate rank_metric={rank_metric} top_n={top_n} trades={int(eval_metrics['trades'])} "
                f"score={float(eval_metrics['score']):.2f} selection_score={float(selection_score):.2f} "
                f"coverage={float(coverage_stats['trade_day_coverage']):.4f} sharpe={float(eval_metrics['daily_sharpe']):.4f} "
                f"seeded={int(len(seeded_side_keys))}"
            )

    if best_payload is None or best_eval is None or best_selection_meta is None:
        raise SystemExit(
            f"No dense configuration met the minimum trade target of {int(args.min_target_trades)}."
        )

    coverage_stats = _trade_day_coverage_stats(
        selected_trade_log,
        trading_days=int(trading_days),
    )
    best_payload["metadata"] = {
        "stage": "dense_final",
        "baseline_artifact_path": str(getattr(baseline_artifact, "path", "")),
        "source": str(source),
        "symbol_mode": str(args.symbol_mode),
        "symbol_method": str(args.symbol_method),
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "trading_days_in_range": int(trading_days),
        "trade_days_covered": int(coverage_stats["trade_days_covered"]),
        "trade_day_coverage": float(coverage_stats["trade_day_coverage"]),
        "avg_trades_per_trading_day": float(best_eval["trades"]) / float(trading_days) if trading_days else 0.0,
        "avg_trades_per_trade_day": float(coverage_stats["avg_trades_per_trade_day"]),
        "probe_summaries": probe_summaries,
        "selection": best_selection_meta,
        "validation": {
            key: _json_safe(value)
            for key, value in best_eval.items()
            if key != "trade_log"
        },
        "config": {
            "rule_specs": str(args.rule_specs),
            "universal_policies": _parse_text_list(str(args.universal_policies)),
            "min_total_support": int(args.min_total_support),
            "min_recent_support": int(args.min_recent_support),
            "min_positive_oos_splits": int(args.min_positive_oos_splits),
            "min_robust_edge": float(args.min_robust_edge),
            "rank_metrics": rank_metrics,
            "top_n_grid": top_n_values,
            "min_target_trades": int(args.min_target_trades),
            "min_trade_days_covered": int(args.min_trade_days_covered),
            "min_trade_day_coverage": float(args.min_trade_day_coverage),
            "trade_day_coverage_weight": float(args.trade_day_coverage_weight),
            "seed_baseline_exact": bool(args.seed_baseline_exact),
            "session_top_n": {key: int(value) for key, value in session_top_n.items()},
            "session_side_top_n": {
                f"{session_name}:{side_name}": int(value)
                for (session_name, side_name), value in session_side_top_n.items()
            },
            "contracts": int(args.contracts),
            "early_exit_enabled": not bool(args.disable_early_exit),
        },
    }

    artifact_root = _resolve_path(str(args.artifact_root), "artifacts/regimeadaptive_dense")
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / "regimeadaptive_dense_artifact.json"
    artifact_path.write_text(
        json.dumps(_json_safe(best_payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    if bool(args.write_latest):
        latest_path = artifact_root / "latest.json"
        shutil.copyfile(artifact_path, latest_path)

    print(f"artifact={artifact_path}")
    print(
        f"best trades={int(best_eval['trades'])} sharpe={float(best_eval['daily_sharpe']):.4f} "
        f"trade_day_coverage={float(coverage_stats['trade_day_coverage']):.4f} "
        f"avg_trades_per_trading_day={float(best_eval['trades']) / float(trading_days):.4f}"
    )


if __name__ == "__main__":
    main()
