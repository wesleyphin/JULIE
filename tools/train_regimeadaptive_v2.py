import argparse
import copy
import datetime as dt
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CONFIG
from regimeadaptive_artifact import RegimeAdaptiveArtifact, load_regimeadaptive_artifact
from tools.backtest_regimeadaptive_robust import (
    _build_artifact_lookups,
    _build_signal_arrays,
    _parse_datetime,
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


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(sub) for key, sub in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for item in str(raw or "").split(","):
        text = item.strip()
        if text:
            out.append(int(text))
    return out


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw or "").split(","):
        text = item.strip()
        if text:
            out.append(float(text))
    return out


def _default_regimeadaptive_early_exit_enabled() -> bool:
    cfg = (CONFIG.get("EARLY_EXIT", {}) or {}).get("RegimeAdaptive", {}) or {}
    return bool(cfg.get("enabled", False))


def _seed_signal_policies(
    seed_artifact: RegimeAdaptiveArtifact,
    include_top_skipped: int = 0,
    min_best_score: float = 0.0,
) -> dict[str, dict[str, dict]]:
    if getattr(seed_artifact, "signal_policies", None):
        seeded: dict[str, dict[str, dict]] = {}
        for combo_key, side_map in seed_artifact.signal_policies.items():
            for original_side, record in side_map.items():
                seeded.setdefault(str(combo_key), {})[str(original_side)] = {
                    "policy": str(record.get("policy", "skip")),
                    "early_exit_enabled": (
                        bool(record.get("early_exit_enabled"))
                        if record.get("early_exit_enabled") is not None
                        else _default_regimeadaptive_early_exit_enabled()
                    ),
                }
        return seeded

    seeded = {}
    default_ee = _default_regimeadaptive_early_exit_enabled()
    combo_records = list((seed_artifact.combo_policies or {}).items())
    for combo_key, record in combo_records:
        policy = str(record.get("policy", "skip")).strip().lower()
        if policy not in {"normal", "reversed"}:
            continue
        side_map = seeded.setdefault(str(combo_key), {})
        for original_side in ("LONG", "SHORT"):
            side_map[original_side] = {
                "policy": policy,
                "early_exit_enabled": default_ee,
            }
    if include_top_skipped > 0:
        skipped_candidates: list[tuple[float, str, str]] = []
        for combo_key, record in combo_records:
            if not isinstance(record, dict):
                continue
            if str(record.get("policy", "skip")).strip().lower() != "skip":
                continue
            score_normal = float(record.get("score_normal", 0.0) or 0.0)
            score_reversed = float(record.get("score_reversed", 0.0) or 0.0)
            best_score = max(score_normal, score_reversed)
            if best_score < float(min_best_score):
                continue
            best_policy = "normal" if score_normal >= score_reversed else "reversed"
            skipped_candidates.append((best_score, str(combo_key), best_policy))
        skipped_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, combo_key, best_policy in skipped_candidates[: int(include_top_skipped)]:
            side_map = seeded.setdefault(str(combo_key), {})
            for original_side in ("LONG", "SHORT"):
                side_map.setdefault(
                    original_side,
                    {
                        "policy": best_policy,
                        "early_exit_enabled": default_ee,
                    },
                )
    return seeded


def _coarse_combo_policies(signal_policies: dict[str, dict[str, dict]]) -> dict[str, dict]:
    combo_policies: dict[str, dict] = {}
    for combo_key, side_map in signal_policies.items():
        long_rec = side_map.get("LONG") if isinstance(side_map, dict) else None
        short_rec = side_map.get("SHORT") if isinstance(side_map, dict) else None
        if not isinstance(long_rec, dict) or not isinstance(short_rec, dict):
            continue
        long_policy = str(long_rec.get("policy", "skip")).strip().lower()
        short_policy = str(short_rec.get("policy", "skip")).strip().lower()
        if long_policy == short_policy and long_policy in {"normal", "reversed", "skip"}:
            combo_policies[str(combo_key)] = {"policy": long_policy}
    return combo_policies


def _evaluate_trade_log(trade_log: list[dict], split_meta: dict, train_score_weight: float) -> dict:
    if not trade_log:
        return {
            "score": float("-inf"),
            "train_total": 0.0,
            "valid_total": 0.0,
            "test_total": 0.0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "avg_trade_net": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    trades = pd.DataFrame(trade_log)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(NY_TZ)
    trades["year"] = trades["entry_time"].dt.year.astype(int)
    trades["pnl_net"] = trades["pnl_net"].astype(float)

    train_set = set(split_meta.get("train_years", []))
    valid_set = set(split_meta.get("valid_years", []))
    test_set = set(split_meta.get("test_years", []))
    trades["split"] = "train"
    trades.loc[trades["year"].isin(valid_set), "split"] = "valid"
    trades.loc[trades["year"].isin(test_set), "split"] = "test"

    split_totals = trades.groupby("split")["pnl_net"].sum().to_dict()
    train_total = float(split_totals.get("train", 0.0))
    valid_total = float(split_totals.get("valid", 0.0))
    test_total = float(split_totals.get("test", 0.0))
    total_net = float(trades["pnl_net"].sum())
    wins = int((trades["pnl_net"] > 0).sum())
    losses = int((trades["pnl_net"] <= 0).sum())
    trade_count = int(len(trades))
    gross_profit = float(trades.loc[trades["pnl_net"] > 0, "pnl_net"].sum())
    gross_loss = float(trades.loc[trades["pnl_net"] < 0, "pnl_net"].sum())
    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else float("inf") if gross_profit > 0 else 0.0
    score = test_total + (0.5 * valid_total) + (float(train_score_weight) * train_total)
    return {
        "score": float(score),
        "train_total": train_total,
        "valid_total": valid_total,
        "test_total": test_total,
        "trades": trade_count,
        "wins": wins,
        "losses": losses,
        "winrate": float((wins / trade_count) * 100.0) if trade_count else 0.0,
        "avg_trade_net": float(total_net / trade_count) if trade_count else 0.0,
        "profit_factor": None if not math.isfinite(profit_factor) else float(profit_factor),
        "max_drawdown": 0.0,
    }


def _summarize_side_groups(trade_log: list[dict], split_meta: dict) -> pd.DataFrame:
    if not trade_log:
        return pd.DataFrame()

    trades = pd.DataFrame(trade_log)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(NY_TZ)
    trades["year"] = trades["entry_time"].dt.year.astype(int)
    trades["pnl_net"] = trades["pnl_net"].astype(float)
    trades["combo_key"] = trades["combo_key"].astype(str)
    trades["original_signal"] = trades["original_signal"].astype(str).str.upper()

    train_set = set(split_meta.get("train_years", []))
    valid_set = set(split_meta.get("valid_years", []))
    test_set = set(split_meta.get("test_years", []))
    trades["split"] = "train"
    trades.loc[trades["year"].isin(valid_set), "split"] = "valid"
    trades.loc[trades["year"].isin(test_set), "split"] = "test"

    records: list[dict] = []
    for (combo_key, original_signal), group in trades.groupby(["combo_key", "original_signal"], sort=False):
        split_means = group.groupby("split")["pnl_net"].mean().to_dict()
        split_totals = group.groupby("split")["pnl_net"].sum().to_dict()
        split_counts = group.groupby("split")["pnl_net"].size().to_dict()
        positive_oos = sum(1 for name in ("valid", "test") if float(split_means.get(name, 0.0)) > 0.0)
        records.append(
            {
                "combo_key": str(combo_key),
                "original_signal": str(original_signal),
                "support_total": int(len(group)),
                "support_recent": int(split_counts.get("test", 0)),
                "total_net": float(group["pnl_net"].sum()),
                "train_total": float(split_totals.get("train", 0.0)),
                "valid_total": float(split_totals.get("valid", 0.0)),
                "test_total": float(split_totals.get("test", 0.0)),
                "train_mean": float(split_means.get("train", 0.0)),
                "valid_mean": float(split_means.get("valid", 0.0)),
                "test_mean": float(split_means.get("test", 0.0)),
                "robust_edge": float(min(split_means.get("valid", 0.0), split_means.get("test", 0.0))),
                "positive_oos_splits": int(positive_oos),
                "winrate": float((group["pnl_net"] > 0).mean() * 100.0),
            }
        )
    return pd.DataFrame.from_records(records)


def _simulate_payload(
    df: pd.DataFrame,
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    holiday_mask: np.ndarray,
    rolling_cache: dict[int, np.ndarray],
    atr_cache: dict[int, np.ndarray],
    payload: dict,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    contracts: int,
    point_value: float,
    fee_per_contract_rt: float,
) -> dict:
    artifact = RegimeAdaptiveArtifact(payload, Path("in_memory_regimeadaptive_v2.json"))
    base_rule = payload.get("base_rule", {}) if isinstance(payload.get("base_rule", {}), dict) else {}
    sma_fast = int(base_rule.get("sma_fast", 20) or 20)
    sma_slow = int(base_rule.get("sma_slow", 200) or 200)
    atr_period = int(base_rule.get("atr_period", 20) or 20)
    cross_atr_mult = float(base_rule.get("cross_atr_mult", 0.1) or 0.1)

    close = df["close"].to_numpy(dtype=np.float64)
    sma_fast_arr = rolling_cache[sma_fast]
    sma_slow_arr = rolling_cache[sma_slow]
    atr = atr_cache[atr_period]

    policy_lookup, early_exit_lookup, long_sl_lookup, long_tp_lookup, short_sl_lookup, short_tp_lookup = _build_artifact_lookups(artifact)
    signal_side, signal_early_exit, signal_sl, signal_tp, original_side = _build_signal_arrays(
        combo_ids,
        session_codes,
        close,
        sma_fast_arr,
        sma_slow_arr,
        atr,
        cross_atr_mult,
        policy_lookup,
        early_exit_lookup,
        long_sl_lookup,
        long_tp_lookup,
        short_sl_lookup,
        short_tp_lookup,
    )
    result = _simulate(
        df,
        start_time,
        end_time,
        combo_ids,
        session_codes,
        holiday_mask,
        signal_side,
        signal_early_exit,
        signal_sl,
        signal_tp,
        original_side,
        contracts,
        point_value,
        fee_per_contract_rt,
    )
    return result


def _build_candidate_payload(seed_artifact: RegimeAdaptiveArtifact, base_rule: dict, signal_policies: dict, metadata: dict | None = None) -> dict:
    return {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "version": "regimeadaptive_v2_exec_sideprune",
        "policy_mode": "side_specific_execution",
        "base_rule": dict(base_rule),
        "combo_policies": _coarse_combo_policies(signal_policies),
        "signal_policies": copy.deepcopy(signal_policies),
        "session_defaults": copy.deepcopy(seed_artifact.payload.get("session_defaults", {}) or {}),
        "global_default": copy.deepcopy(seed_artifact.payload.get("global_default", {}) or {}),
        "seed_artifact_path": str(getattr(seed_artifact, "path", "")),
        "metadata": dict(metadata or {}),
    }


def _apply_prune_rules(signal_policies: dict, side_stats: pd.DataFrame, args) -> tuple[dict, dict]:
    next_policies = copy.deepcopy(signal_policies)
    reasons: dict[str, str] = {}
    if side_stats.empty:
        return next_policies, reasons

    for row in side_stats.itertuples(index=False):
        combo_key = str(row.combo_key)
        original_signal = str(row.original_signal)
        keep = True
        if int(row.support_total) < int(args.min_total_trades):
            keep = False
            reason = "low_total_support"
        elif int(row.support_recent) < int(args.min_recent_trades):
            keep = False
            reason = "low_recent_support"
        elif float(row.total_net) <= float(args.min_total_net):
            keep = False
            reason = "non_positive_total_net"
        elif float(row.train_mean) < float(args.min_train_avg):
            keep = False
            reason = "train_mean_below_floor"
        elif float(row.robust_edge) < float(args.min_split_edge):
            keep = False
            reason = "oos_edge_below_floor"
        elif int(row.positive_oos_splits) < int(args.min_positive_oos_splits):
            keep = False
            reason = "insufficient_positive_oos_splits"
        else:
            reason = "kept"
        side_map = next_policies.setdefault(combo_key, {})
        record = side_map.setdefault(original_signal, {"policy": "skip", "early_exit_enabled": _default_regimeadaptive_early_exit_enabled()})
        if keep:
            continue
        record["policy"] = "skip"
        reasons[f"{combo_key}:{original_signal}"] = reason
    return next_policies, reasons


def _retained_side_keys(signal_policies: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for combo_key, side_map in signal_policies.items():
        if not isinstance(side_map, dict):
            continue
        for original_signal, record in side_map.items():
            if isinstance(record, dict) and str(record.get("policy", "skip")).strip().lower() in {"normal", "reversed"}:
                out.append((str(combo_key), str(original_signal)))
    return out


def _side_record(signal_policies: dict, combo_key: str, original_signal: str) -> dict:
    side_map = signal_policies.get(str(combo_key), {})
    if isinstance(side_map, dict):
        record = side_map.get(str(original_signal))
        if isinstance(record, dict):
            return record
    return {}


def _activate_side_policy(payload: dict, combo_key: str, original_signal: str, policy: str, early_exit_enabled: bool) -> dict:
    next_payload = copy.deepcopy(payload)
    side_map = next_payload.setdefault("signal_policies", {}).setdefault(str(combo_key), {})
    side_map[str(original_signal)] = {
        "policy": str(policy),
        "early_exit_enabled": bool(early_exit_enabled),
    }
    next_payload["combo_policies"] = _coarse_combo_policies(next_payload.get("signal_policies", {}) or {})
    return next_payload


def _forward_add_search(
    df: pd.DataFrame,
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    holiday_mask: np.ndarray,
    rolling_cache: dict[int, np.ndarray],
    atr_cache: dict[int, np.ndarray],
    payload: dict,
    eval_summary: dict,
    seed_signal_policies: dict,
    split_meta: dict,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    contracts: int,
    point_value: float,
    fee_per_contract_rt: float,
    train_score_weight: float,
    max_additions: int,
) -> tuple[dict, dict, dict, list[dict]]:
    tuned_payload = copy.deepcopy(payload)
    tuned_result = _simulate_payload(
        df,
        combo_ids,
        session_codes,
        holiday_mask,
        rolling_cache,
        atr_cache,
        tuned_payload,
        start_time,
        end_time,
        contracts,
        point_value,
        fee_per_contract_rt,
    )
    tuned_eval = dict(eval_summary)
    additions: list[dict] = []
    if int(max_additions) <= 0:
        return tuned_payload, tuned_result, tuned_eval, additions

    for _ in range(int(max_additions)):
        best_payload = None
        best_result = None
        best_eval = None
        best_addition = None

        for combo_key, side_map in seed_signal_policies.items():
            if not isinstance(side_map, dict):
                continue
            for original_signal, seed_record in side_map.items():
                if not isinstance(seed_record, dict):
                    continue
                seed_policy = str(seed_record.get("policy", "skip")).strip().lower()
                if seed_policy not in {"normal", "reversed"}:
                    continue
                current_record = _side_record(tuned_payload.get("signal_policies", {}) or {}, combo_key, original_signal)
                current_policy = str(current_record.get("policy", "skip")).strip().lower()
                if current_policy in {"normal", "reversed"}:
                    continue

                seed_ee = bool(seed_record.get("early_exit_enabled", _default_regimeadaptive_early_exit_enabled()))
                for candidate_ee in (seed_ee, not seed_ee):
                    candidate_payload = _activate_side_policy(
                        tuned_payload,
                        combo_key,
                        original_signal,
                        seed_policy,
                        candidate_ee,
                    )
                    candidate_result = _simulate_payload(
                        df,
                        combo_ids,
                        session_codes,
                        holiday_mask,
                        rolling_cache,
                        atr_cache,
                        candidate_payload,
                        start_time,
                        end_time,
                        contracts,
                        point_value,
                        fee_per_contract_rt,
                    )
                    candidate_eval = _evaluate_trade_log(
                        candidate_result.get("trade_log", []) or [],
                        split_meta,
                        float(train_score_weight),
                    )
                    if best_eval is None or candidate_eval["score"] > best_eval["score"]:
                        best_payload = candidate_payload
                        best_result = candidate_result
                        best_eval = candidate_eval
                        best_addition = {
                            "combo_key": str(combo_key),
                            "original_signal": str(original_signal),
                            "policy": seed_policy,
                            "early_exit_enabled": bool(candidate_ee),
                            "score": float(candidate_eval["score"]),
                            "valid_total": float(candidate_eval["valid_total"]),
                            "test_total": float(candidate_eval["test_total"]),
                            "trades": int(candidate_eval["trades"]),
                        }

        if best_eval is None or best_payload is None or best_result is None or best_addition is None:
            break
        if float(best_eval["score"]) <= float(tuned_eval["score"]):
            break
        tuned_payload = best_payload
        tuned_result = best_result
        tuned_eval = best_eval
        additions.append(best_addition)

    if additions:
        tuned_payload.setdefault("metadata", {})["forward_additions"] = additions
    return tuned_payload, tuned_result, tuned_eval, additions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RegimeAdaptive v2 with execution-aware side pruning and per-side early-exit search."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--seed-artifact", default="artifacts/regimeadaptive_robust/latest.json")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--sma-fast-values", default="")
    parser.add_argument("--sma-slow-values", default="")
    parser.add_argument("--cross-atr-mults", default="")
    parser.add_argument("--valid-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--contracts", type=int, default=5)
    parser.add_argument("--min-total-trades", type=int, default=25)
    parser.add_argument("--min-recent-trades", type=int, default=5)
    parser.add_argument("--min-total-net", type=float, default=0.0)
    parser.add_argument("--min-train-avg", type=float, default=-1.0)
    parser.add_argument("--min-split-edge", type=float, default=0.0)
    parser.add_argument("--min-positive-oos-splits", type=int, default=1)
    parser.add_argument("--train-score-weight", type=float, default=0.25)
    parser.add_argument("--seed-top-skipped", type=int, default=0)
    parser.add_argument("--seed-min-best-score", type=float, default=0.0)
    parser.add_argument("--max-forward-additions", type=int, default=3)
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--write-latest", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Source parquet not found: {source}")
    seed_artifact = load_regimeadaptive_artifact(str(args.seed_artifact))
    if seed_artifact is None:
        raise SystemExit(f"Seed artifact could not be loaded: {args.seed_artifact}")

    seed_base_rule = seed_artifact.base_rule or {}
    sma_fast_values = _parse_int_list(args.sma_fast_values) or [int(seed_base_rule.get("sma_fast", 50) or 50)]
    sma_slow_values = _parse_int_list(args.sma_slow_values) or [int(seed_base_rule.get("sma_slow", 300) or 300)]
    cross_atr_mults = _parse_float_list(args.cross_atr_mults) or [float(seed_base_rule.get("cross_atr_mult", 2.0) or 2.0)]
    atr_period = int(seed_base_rule.get("atr_period", 20) or 20)
    max_hold_bars = int(seed_base_rule.get("max_hold_bars", 30) or 30)

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
    rolling_cache = _rolling_cache(close, list(sma_fast_values) + list(sma_slow_values))
    atr_cache = {int(atr_period): _atr_array(high, low, close, int(atr_period))}

    risk_cfg = CONFIG.get("RISK", {}) or {}
    point_value = float(risk_cfg.get("POINT_VALUE", 5.0) or 5.0)
    fee_per_side = float(risk_cfg.get("FEES_PER_SIDE", 0.37) or 0.37)
    fee_per_contract_rt = fee_per_side * 2.0

    seed_signal_policies = _seed_signal_policies(
        seed_artifact,
        include_top_skipped=int(args.seed_top_skipped),
        min_best_score=float(args.seed_min_best_score),
    )
    candidate_summaries: list[dict] = []
    best_payload = None
    best_result = None
    best_eval = None
    best_side_stats = None

    for sma_fast in sma_fast_values:
        for sma_slow in sma_slow_values:
            if sma_fast <= 0 or sma_slow <= 0 or sma_fast >= sma_slow:
                continue
            for cross_atr_mult in cross_atr_mults:
                base_rule = {
                    "sma_fast": int(sma_fast),
                    "sma_slow": int(sma_slow),
                    "cross_atr_mult": float(cross_atr_mult),
                    "atr_period": int(atr_period),
                    "max_hold_bars": int(max_hold_bars),
                }
                initial_payload = _build_candidate_payload(
                    seed_artifact,
                    base_rule,
                    seed_signal_policies,
                    metadata={"stage": "seed_side_expansion"},
                )
                initial_result = _simulate_payload(
                    df,
                    combo_ids,
                    session_codes,
                    holiday_mask,
                    rolling_cache,
                    atr_cache,
                    initial_payload,
                    start_time,
                    end_time,
                    int(args.contracts),
                    point_value,
                    fee_per_contract_rt,
                )
                side_stats = _summarize_side_groups(initial_result.get("trade_log", []) or [], split_meta)
                pruned_signal_policies, prune_reasons = _apply_prune_rules(seed_signal_policies, side_stats, args)
                pruned_payload = _build_candidate_payload(
                    seed_artifact,
                    base_rule,
                    pruned_signal_policies,
                    metadata={
                        "stage": "post_prune",
                        "prune_reasons": prune_reasons,
                    },
                )
                pruned_result = _simulate_payload(
                    df,
                    combo_ids,
                    session_codes,
                    holiday_mask,
                    rolling_cache,
                    atr_cache,
                    pruned_payload,
                    start_time,
                    end_time,
                    int(args.contracts),
                    point_value,
                    fee_per_contract_rt,
                )
                pruned_eval = _evaluate_trade_log(pruned_result.get("trade_log", []) or [], split_meta, float(args.train_score_weight))

                # Greedy single-pass early-exit toggle search on retained side policies.
                tuned_payload = copy.deepcopy(pruned_payload)
                tuned_result = pruned_result
                tuned_eval = dict(pruned_eval)
                retained_keys = _retained_side_keys(tuned_payload.get("signal_policies", {}) or {})
                for combo_key, original_signal in retained_keys:
                    side_map = tuned_payload.setdefault("signal_policies", {}).setdefault(combo_key, {})
                    record = side_map.setdefault(
                        original_signal,
                        {"policy": "skip", "early_exit_enabled": _default_regimeadaptive_early_exit_enabled()},
                    )
                    current_ee = bool(record.get("early_exit_enabled", _default_regimeadaptive_early_exit_enabled()))
                    candidate_payload = copy.deepcopy(tuned_payload)
                    candidate_payload["signal_policies"][combo_key][original_signal]["early_exit_enabled"] = not current_ee
                    candidate_result = _simulate_payload(
                        df,
                        combo_ids,
                        session_codes,
                        holiday_mask,
                        rolling_cache,
                        atr_cache,
                        candidate_payload,
                        start_time,
                        end_time,
                        int(args.contracts),
                        point_value,
                        fee_per_contract_rt,
                    )
                    candidate_eval = _evaluate_trade_log(candidate_result.get("trade_log", []) or [], split_meta, float(args.train_score_weight))
                    if candidate_eval["score"] > tuned_eval["score"]:
                        tuned_payload = candidate_payload
                        tuned_result = candidate_result
                        tuned_eval = candidate_eval

                tuned_payload, tuned_result, tuned_eval, forward_additions = _forward_add_search(
                    df,
                    combo_ids,
                    session_codes,
                    holiday_mask,
                    rolling_cache,
                    atr_cache,
                    tuned_payload,
                    tuned_eval,
                    seed_signal_policies,
                    split_meta,
                    start_time,
                    end_time,
                    int(args.contracts),
                    point_value,
                    fee_per_contract_rt,
                    float(args.train_score_weight),
                    int(args.max_forward_additions),
                )

                final_side_stats = _summarize_side_groups(tuned_result.get("trade_log", []) or [], split_meta)
                summary = {
                    "sma_fast": int(sma_fast),
                    "sma_slow": int(sma_slow),
                    "cross_atr_mult": float(cross_atr_mult),
                    "score": float(tuned_eval["score"]),
                    "train_total": float(tuned_eval["train_total"]),
                    "valid_total": float(tuned_eval["valid_total"]),
                    "test_total": float(tuned_eval["test_total"]),
                    "trades": int(tuned_eval["trades"]),
                    "wins": int(tuned_eval["wins"]),
                    "losses": int(tuned_eval["losses"]),
                    "winrate": float(tuned_eval["winrate"]),
                    "avg_trade_net": float(tuned_eval["avg_trade_net"]),
                    "profit_factor": None if tuned_eval["profit_factor"] is None else float(tuned_eval["profit_factor"]),
                    "retained_side_policy_count": int(len(_retained_side_keys(tuned_payload.get("signal_policies", {}) or {}))),
                    "forward_additions": copy.deepcopy(forward_additions),
                }
                candidate_summaries.append(summary)
                print(
                    f"sma_fast={sma_fast} sma_slow={sma_slow} cross_atr={cross_atr_mult:.2f} "
                    f"score={summary['score']:.2f} valid={summary['valid_total']:.2f} "
                    f"test={summary['test_total']:.2f} trades={summary['trades']} retained_side_policies={summary['retained_side_policy_count']}"
                )
                if best_eval is None or summary["score"] > best_eval["score"]:
                    best_payload = tuned_payload
                    best_result = tuned_result
                    best_eval = summary
                    best_side_stats = final_side_stats

    if best_payload is None or best_result is None or best_eval is None:
        raise SystemExit("No v2 candidates were produced.")

    if str(args.artifact_root or "").strip():
        artifact_root = Path(args.artifact_root).expanduser().resolve()
    else:
        artifact_root = ROOT / "artifacts" / f"regimeadaptive_v2_{dt.datetime.now(NY_TZ).strftime('%Y%m%d_%H%M%S')}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / "regimeadaptive_v2_artifact.json"

    best_payload.update(
        {
            "created_at": dt.datetime.now(NY_TZ).isoformat(),
            "source_data_path": str(source),
            "symbol": symbol_label,
            "range_start": start_time.isoformat(),
            "range_end": end_time.isoformat(),
            "split_meta": split_meta,
            "training_config": {
                "seed_artifact": str(getattr(seed_artifact, "path", args.seed_artifact)),
                "contracts": int(args.contracts),
                "min_total_trades": int(args.min_total_trades),
                "min_recent_trades": int(args.min_recent_trades),
                "min_total_net": float(args.min_total_net),
                "min_train_avg": float(args.min_train_avg),
                "min_split_edge": float(args.min_split_edge),
                "min_positive_oos_splits": int(args.min_positive_oos_splits),
                "train_score_weight": float(args.train_score_weight),
                "seed_top_skipped": int(args.seed_top_skipped),
                "seed_min_best_score": float(args.seed_min_best_score),
                "max_forward_additions": int(args.max_forward_additions),
            },
            "candidate_summaries": candidate_summaries,
            "best_candidate": best_eval,
            "execution_result": {
                key: value
                for key, value in best_result.items()
                if key != "trade_log"
            },
            "summary": {
                "retained_combo_count": int(len({combo for combo, _ in _retained_side_keys(best_payload.get("signal_policies", {}) or {})})),
                "retained_side_policy_count": int(len(_retained_side_keys(best_payload.get("signal_policies", {}) or {}))),
                "trade_count": int(best_result.get("trades", 0)),
            },
            "side_policy_stats": best_side_stats.to_dict("records") if isinstance(best_side_stats, pd.DataFrame) and not best_side_stats.empty else [],
        }
    )
    artifact_path.write_text(json.dumps(_json_safe(best_payload), indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"artifact={artifact_path}")
    if bool(args.write_latest):
        latest_dir = ROOT / "artifacts" / "regimeadaptive_v2"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_path = latest_dir / "latest.json"
        shutil.copyfile(artifact_path, latest_path)
        print(f"latest={latest_path}")


if __name__ == "__main__":
    main()
