import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import math
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
from regimeadaptive_artifact import load_regimeadaptive_artifact
from tools.backtest_regimeadaptive_robust import _build_rule_strength_arrays, _parse_datetime, _rolling_extrema_cache
from tools.regimeadaptive_filterless_runner import (
    NY_TZ,
    _atr_array,
    _build_combo_arrays,
    _build_holiday_mask,
    _load_bars,
    _rolling_cache,
)
from tools.train_regimeadaptive_v2 import (
    _apply_prune_rules,
    _coarse_combo_policies,
    _default_regimeadaptive_early_exit_enabled,
    _evaluate_trade_log,
    _json_safe,
    _retained_side_keys,
    _seed_signal_policies,
    _summarize_side_groups,
)
from tools.train_regimeadaptive_v3 import (
    _baseline_payload,
    _merge_combo_policies,
    _parse_rule_specs,
    _rule_seed_payload,
    _simulate_payload_v3,
)


GROUP_TEMPLATE_MAP: dict[str, tuple[str, str, str, str]] = {
    "week_day_session": ("ALL", "{week}", "{day}", "{session}"),
    "quarter_day_session": ("{quarter}", "ALL", "{day}", "{session}"),
    "quarter_week_session": ("{quarter}", "{week}", "ALL", "{session}"),
    "day_session": ("ALL", "ALL", "{day}", "{session}"),
    "quarter_session": ("{quarter}", "ALL", "ALL", "{session}"),
    "week_session": ("ALL", "{week}", "ALL", "{session}"),
    "session_only": ("ALL", "ALL", "ALL", "{session}"),
    "day_only": ("ALL", "ALL", "{day}", "ALL"),
    "quarter_day": ("{quarter}", "ALL", "{day}", "ALL"),
    "week_day": ("ALL", "{week}", "{day}", "ALL"),
    "quarter_only": ("{quarter}", "ALL", "ALL", "ALL"),
    "week_only": ("ALL", "{week}", "ALL", "ALL"),
}

SIM_CACHE_VERSION = "regimeadaptive_v5_sim_cache_v1"
CHECKPOINT_VERSION = "regimeadaptive_v5_checkpoint_v1"


def _parse_group_templates(raw: str) -> list[str]:
    out: list[str] = []
    for item in str(raw or "").split(","):
        name = str(item or "").strip().lower()
        if not name:
            continue
        if name not in GROUP_TEMPLATE_MAP:
            raise ValueError(f"Unsupported group template: {item}")
        out.append(name)
    return out


def _group_pattern(combo_key: str, template_name: str) -> str | None:
    parts = [str(part or "").strip().upper() for part in str(combo_key or "").split("_")]
    if len(parts) != 4:
        return None
    quarter, week, day, session = parts
    template = GROUP_TEMPLATE_MAP.get(str(template_name))
    if template is None:
        return None
    values = {
        "quarter": quarter,
        "week": week,
        "day": day,
        "session": session,
    }
    return "_".join(segment.format(**values) for segment in template)


def _candidate_rank_score(train_total: float, valid_total: float, test_total: float, train_score_weight: float) -> float:
    return float(test_total) + (0.5 * float(valid_total)) + (float(train_score_weight) * float(train_total))


def _trade_frame(trade_log: list[dict], split_meta: dict) -> pd.DataFrame:
    if not trade_log:
        return pd.DataFrame()
    trades = pd.DataFrame(trade_log)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(NY_TZ)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert(NY_TZ)
    trades["year"] = trades["entry_time"].dt.year.astype(int)
    trades["pnl_net"] = trades["pnl_net"].astype(float)
    trades["combo_key"] = trades["combo_key"].astype(str)
    trades["original_signal"] = trades["original_signal"].astype(str).str.upper()
    trades["policy"] = np.where(trades.get("reverted", False).astype(bool), "reversed", "normal")
    valid_set = set(split_meta.get("valid_years", []))
    test_set = set(split_meta.get("test_years", []))
    trades["split"] = "train"
    trades.loc[trades["year"].isin(valid_set), "split"] = "valid"
    trades.loc[trades["year"].isin(test_set), "split"] = "test"
    return trades


def _summarize_group_side_groups(trade_log: list[dict], split_meta: dict, template_names: list[str]) -> pd.DataFrame:
    trades = _trade_frame(trade_log, split_meta)
    if trades.empty or not template_names:
        return pd.DataFrame()

    expanded_rows: list[dict] = []
    for row in trades.itertuples(index=False):
        for template_name in template_names:
            group_key = _group_pattern(str(row.combo_key), template_name)
            if not group_key:
                continue
            expanded_rows.append(
                {
                    "group_key": str(group_key),
                    "template_name": str(template_name),
                    "combo_key": str(row.combo_key),
                    "original_signal": str(row.original_signal),
                    "policy": str(row.policy),
                    "split": str(row.split),
                    "pnl_net": float(row.pnl_net),
                    "is_win": bool(float(row.pnl_net) > 0.0),
                }
            )
    if not expanded_rows:
        return pd.DataFrame()

    grouped = pd.DataFrame.from_records(expanded_rows)
    records: list[dict] = []
    for (group_key, template_name, original_signal, policy), group in grouped.groupby(
        ["group_key", "template_name", "original_signal", "policy"], sort=False
    ):
        split_means = group.groupby("split")["pnl_net"].mean().to_dict()
        split_totals = group.groupby("split")["pnl_net"].sum().to_dict()
        split_counts = group.groupby("split")["pnl_net"].size().to_dict()
        positive_oos = sum(1 for name in ("valid", "test") if float(split_means.get(name, 0.0)) > 0.0)
        records.append(
            {
                "group_key": str(group_key),
                "template_name": str(template_name),
                "original_signal": str(original_signal),
                "policy": str(policy),
                "support_total": int(len(group)),
                "support_recent": int(split_counts.get("test", 0)),
                "distinct_combo_count": int(group["combo_key"].nunique()),
                "total_net": float(group["pnl_net"].sum()),
                "train_total": float(split_totals.get("train", 0.0)),
                "valid_total": float(split_totals.get("valid", 0.0)),
                "test_total": float(split_totals.get("test", 0.0)),
                "train_mean": float(split_means.get("train", 0.0)),
                "valid_mean": float(split_means.get("valid", 0.0)),
                "test_mean": float(split_means.get("test", 0.0)),
                "robust_edge": float(min(split_means.get("valid", 0.0), split_means.get("test", 0.0))),
                "positive_oos_splits": int(positive_oos),
                "winrate": float(group["is_win"].mean() * 100.0),
            }
        )
    return pd.DataFrame.from_records(records)


def _group_row_passes(row, args) -> bool:
    if int(row.distinct_combo_count) < int(args.group_min_combo_count):
        return False
    if int(row.support_total) < int(args.group_min_total_trades):
        return False
    if int(row.support_recent) < int(args.group_min_recent_trades):
        return False
    if float(row.total_net) <= float(args.min_total_net):
        return False
    if float(row.train_mean) <= float(args.min_train_avg):
        return False
    if float(row.robust_edge) <= float(args.min_split_edge):
        return False
    if int(row.positive_oos_splits) < int(args.min_positive_oos_splits):
        return False
    return True


def _build_split_map(index: pd.DatetimeIndex, valid_years: int, test_years: int) -> tuple[np.ndarray, list[str], dict]:
    years = sorted({int(ts.year) for ts in index})
    if len(years) <= max(valid_years + test_years, 2):
        raise ValueError("Not enough years in range for the requested validation/test split.")
    test_set = set(years[-test_years:])
    valid_set = set(years[-(valid_years + test_years):-test_years])
    split = np.empty(len(index), dtype=object)
    split[:] = "train"
    years_arr = index.year.to_numpy(dtype=np.int16)
    split[np.isin(years_arr, list(valid_set))] = "valid"
    split[np.isin(years_arr, list(test_set))] = "test"
    return split, ["train", "valid", "test"], {
        "train_years": [year for year in years if year not in valid_set and year not in test_set],
        "valid_years": sorted(valid_set),
        "test_years": sorted(test_set),
    }


def _evaluate_trade_log_v5(
    trade_log: list[dict],
    split_meta: dict,
    train_score_weight: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    args,
) -> dict:
    base = _evaluate_trade_log(trade_log, split_meta, train_score_weight)
    if not trade_log:
        base.update(
            {
                "daily_sharpe": 0.0,
                "negative_years": 0,
                "worst_3y_pnl": 0.0,
                "worst_5y_pnl": 0.0,
                "yearly_pnl_std": 0.0,
                "max_drawdown": 0.0,
            }
        )
        return base

    trades = _trade_frame(trade_log, split_meta)
    yearly = trades.groupby("year")["pnl_net"].sum().sort_index()
    day_index = pd.date_range(start_time.tz_convert(NY_TZ).normalize(), end_time.tz_convert(NY_TZ).normalize(), freq="D")
    daily = (
        trades.groupby(trades["exit_time"].dt.normalize())["pnl_net"]
        .sum()
        .sort_index()
        .reindex(day_index, fill_value=0.0)
    )
    daily_mean = float(daily.mean())
    daily_std = float(daily.std(ddof=0))
    daily_sharpe = (daily_mean / daily_std) * math.sqrt(252.0) if daily_std > 0.0 else 0.0
    roll3 = yearly.rolling(3).sum().dropna()
    roll5 = yearly.rolling(5).sum().dropna()
    equity_curve = trades.sort_values("exit_time")["pnl_net"].cumsum()
    running_peak = equity_curve.cummax()
    max_drawdown = float((running_peak - equity_curve).max()) if not equity_curve.empty else 0.0
    negative_years = int((yearly < 0).sum())
    worst_3y_pnl = float(roll3.min()) if not roll3.empty else float(yearly.sum())
    worst_5y_pnl = float(roll5.min()) if not roll5.empty else float(yearly.sum())
    yearly_pnl_std = float(yearly.std(ddof=0)) if len(yearly) > 1 else 0.0

    score = float(base["score"])
    score += float(args.robust_sharpe_weight) * float(daily_sharpe)
    score += float(args.trade_count_weight) * float(base["trades"])
    score -= float(args.negative_year_penalty) * float(negative_years)
    score += float(args.worst_3y_weight) * float(worst_3y_pnl)
    score += float(args.worst_5y_weight) * float(worst_5y_pnl)
    score -= float(args.max_drawdown_penalty) * float(max_drawdown)
    score -= float(args.yearly_std_penalty) * float(yearly_pnl_std)

    base.update(
        {
            "score": float(score),
            "daily_sharpe": float(daily_sharpe),
            "negative_years": int(negative_years),
            "worst_3y_pnl": float(worst_3y_pnl),
            "worst_5y_pnl": float(worst_5y_pnl),
            "yearly_pnl_std": float(yearly_pnl_std),
            "max_drawdown": float(max_drawdown),
        }
    )
    return base


def _policy_signature(payload: dict) -> set[tuple[str, str, str, str, str, bool]]:
    out: set[tuple[str, str, str, str, str, bool]] = set()
    for candidate_type, collection_name in (("exact", "signal_policies"), ("group", "group_signal_policies")):
        mapping = payload.get(collection_name, {}) if isinstance(payload.get(collection_name, {}), dict) else {}
        for target_key, side_map in mapping.items():
            if not isinstance(side_map, dict):
                continue
            for original_signal, record in side_map.items():
                if not isinstance(record, dict):
                    continue
                policy = str(record.get("policy", "skip")).strip().lower()
                if policy not in {"normal", "reversed"}:
                    continue
                out.add(
                    (
                        str(candidate_type),
                        str(target_key),
                        str(original_signal),
                        policy,
                        str(record.get("rule_id", "") or ""),
                        bool(record.get("early_exit_enabled", _default_regimeadaptive_early_exit_enabled())),
                    )
                )
    return out


def _apply_candidate(payload: dict, candidate: dict, early_exit_enabled: bool) -> dict:
    next_payload = copy.deepcopy(payload)
    record = {
        "policy": str(candidate["policy"]),
        "early_exit_enabled": bool(early_exit_enabled),
        "rule_id": str(candidate["rule_id"]),
    }
    if str(candidate["candidate_type"]) == "group":
        next_payload.setdefault("group_signal_policies", {}).setdefault(str(candidate["target_key"]), {})[str(candidate["original_signal"])] = record
    else:
        next_payload.setdefault("signal_policies", {}).setdefault(str(candidate["target_key"]), {})[str(candidate["original_signal"])] = record
        next_payload["combo_policies"] = _merge_combo_policies(
            next_payload.get("combo_policies", {}) or {},
            next_payload.get("signal_policies", {}) or {},
        )
    return next_payload


def _active_policy_targets(payload: dict) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for candidate_type, collection_name in (("exact", "signal_policies"), ("group", "group_signal_policies")):
        mapping = payload.get(collection_name, {}) if isinstance(payload.get(collection_name, {}), dict) else {}
        for target_key, side_map in mapping.items():
            if not isinstance(side_map, dict):
                continue
            for original_signal, record in side_map.items():
                if not isinstance(record, dict):
                    continue
                policy = str(record.get("policy", "skip")).strip().lower()
                if policy in {"normal", "reversed"}:
                    out.append((str(candidate_type), str(target_key), str(original_signal)))
    return out


def _resolve_path(path_text: str, default_relative: str) -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _canonical_json(value) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _config_signature(args) -> str:
    ignored = {"artifact_root", "write_latest", "cache_dir", "checkpoint_root", "resume", "run_tag"}
    config_payload = {
        str(key): value
        for key, value in vars(args).items()
        if str(key) not in ignored
    }
    return hashlib.sha256(_canonical_json(config_payload).encode("utf-8")).hexdigest()


def _simulation_cache_key(
    payload: dict,
    source: Path,
    symbol_mode: str,
    symbol_method: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    contracts: int,
) -> str:
    meta = {
        "cache_version": SIM_CACHE_VERSION,
        "payload_hash": _payload_hash(payload),
        "source": str(source.resolve()),
        "source_size": int(source.stat().st_size),
        "source_mtime_ns": int(source.stat().st_mtime_ns),
        "symbol_mode": str(symbol_mode),
        "symbol_method": str(symbol_method),
        "range_start": start_time.isoformat(),
        "range_end": end_time.isoformat(),
        "contracts": int(contracts),
    }
    return hashlib.sha256(_canonical_json(meta).encode("utf-8")).hexdigest()


def _simulation_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / str(cache_key[:2]) / f"{cache_key}.json"


def _load_checkpoint(checkpoint_path: Path) -> dict | None:
    if not checkpoint_path.is_file():
        return None
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_checkpoint(checkpoint_path: Path, state: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(_json_safe(state), indent=2, ensure_ascii=True), encoding="utf-8")


def _dedupe_candidates(candidate_rows: list[dict], baseline_payload: dict, max_candidates_total: int) -> list[dict]:
    rows = sorted(
        candidate_rows,
        key=lambda item: (
            item["candidate_score"],
            item["test_total"],
            item["valid_total"],
            item["distinct_combo_count"],
            item["support_total"],
        ),
        reverse=True,
    )
    deduped_candidates: list[dict] = []
    seen_signatures: set[tuple[str, str, str, str, str]] = set()
    baseline_signature = _policy_signature(baseline_payload)
    for row in rows:
        sig = (
            str(row["candidate_type"]),
            str(row["target_key"]),
            str(row["original_signal"]),
            str(row["policy"]),
            str(row["rule_id"]),
        )
        if sig in seen_signatures:
            continue
        if (
            str(row["candidate_type"]),
            str(row["target_key"]),
            str(row["original_signal"]),
            str(row["policy"]),
            str(row["rule_id"]),
            bool(row["early_exit_enabled"]),
        ) in baseline_signature:
            continue
        seen_signatures.add(sig)
        deduped_candidates.append(dict(row))
        if len(deduped_candidates) >= int(max_candidates_total):
            break
    return deduped_candidates


def _evaluate_payload_cached(
    payload: dict,
    *,
    cache_dir: Path,
    source: Path,
    symbol_mode: str,
    symbol_method: str,
    df: pd.DataFrame,
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    holiday_mask: np.ndarray,
    rule_long_strength: dict[str, np.ndarray],
    rule_short_strength: dict[str, np.ndarray],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    contracts: int,
    point_value: float,
    fee_per_contract_rt: float,
    split_meta: dict,
    template_names: list[str],
    args,
    hours: np.ndarray | None = None,
    minutes: np.ndarray | None = None,
    test_positions: np.ndarray | None = None,
    prebuilt_rule_order: list[str] | None = None,
    prebuilt_long_strength_matrix: np.ndarray | None = None,
    prebuilt_short_strength_matrix: np.ndarray | None = None,
    include_side_stats: bool = False,
    include_group_stats: bool = False,
) -> dict:
    cache_key = _simulation_cache_key(payload, source, symbol_mode, symbol_method, start_time, end_time, contracts)
    cache_path = _simulation_cache_path(cache_dir, cache_key)
    cache_hit = False
    if cache_path.is_file():
        cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        result = dict(cached_payload.get("result", {}) or {})
        cache_hit = True
    else:
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
            int(contracts),
            point_value,
            fee_per_contract_rt,
            hours=hours,
            minutes=minutes,
            test_positions=test_positions,
            prebuilt_rule_order=prebuilt_rule_order,
            prebuilt_long_strength_matrix=prebuilt_long_strength_matrix,
            prebuilt_short_strength_matrix=prebuilt_short_strength_matrix,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "cache_version": SIM_CACHE_VERSION,
                    "cache_key": cache_key,
                    "payload_hash": _payload_hash(payload),
                    "result": _json_safe(result),
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
    trade_log = result.get("trade_log", []) or []
    evaluation = _evaluate_trade_log_v5(
        trade_log,
        split_meta,
        float(args.train_score_weight),
        start_time,
        end_time,
        args,
    )
    side_stats = _summarize_side_groups(trade_log, split_meta) if include_side_stats else pd.DataFrame()
    group_stats = (
        _summarize_group_side_groups(trade_log, split_meta, template_names)
        if include_group_stats and template_names
        else pd.DataFrame()
    )
    return {
        "cache_key": cache_key,
        "cache_hit": bool(cache_hit),
        "result": result,
        "eval": evaluation,
        "side_stats": side_stats,
        "group_stats": group_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RegimeAdaptive v5 with hierarchical fallback policies and a robustness-weighted objective."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--baseline-artifact", default="artifacts/regimeadaptive_v4/latest.json")
    parser.add_argument("--candidate-seed-artifact", default="artifacts/regimeadaptive_robust/latest.json")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--rule-specs", default="50:200:2.0,50:200:1.5,50:200:1.0,34:100:1.0")
    parser.add_argument("--valid-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--contracts", type=int, default=10)
    parser.add_argument("--min-total-trades", type=int, default=25)
    parser.add_argument("--min-recent-trades", type=int, default=5)
    parser.add_argument("--min-total-net", type=float, default=0.0)
    parser.add_argument("--min-train-avg", type=float, default=-1.0)
    parser.add_argument("--min-split-edge", type=float, default=0.0)
    parser.add_argument("--min-positive-oos-splits", type=int, default=1)
    parser.add_argument("--train-score-weight", type=float, default=0.25)
    parser.add_argument("--seed-top-skipped", type=int, default=20)
    parser.add_argument("--seed-min-best-score", type=float, default=1.5)
    parser.add_argument("--max-candidates-per-rule", type=int, default=4)
    parser.add_argument("--max-group-candidates-per-rule", type=int, default=3)
    parser.add_argument("--max-candidates-total", type=int, default=8)
    parser.add_argument("--max-selection-steps", type=int, default=2)
    parser.add_argument("--group-templates", default="week_day_session,quarter_day_session,day_session,quarter_session")
    parser.add_argument("--group-min-combo-count", type=int, default=2)
    parser.add_argument("--group-min-total-trades", type=int, default=40)
    parser.add_argument("--group-min-recent-trades", type=int, default=8)
    parser.add_argument("--group-diversity-bonus", type=float, default=100.0)
    parser.add_argument("--robust-sharpe-weight", type=float, default=1000.0)
    parser.add_argument("--negative-year-penalty", type=float, default=300.0)
    parser.add_argument("--worst-3y-weight", type=float, default=0.15)
    parser.add_argument("--worst-5y-weight", type=float, default=0.35)
    parser.add_argument("--max-drawdown-penalty", type=float, default=0.05)
    parser.add_argument("--yearly-std-penalty", type=float, default=0.02)
    parser.add_argument("--trade-count-weight", type=float, default=0.5)
    parser.add_argument("--cache-dir", default="cache/regimeadaptive_v5_sim")
    parser.add_argument("--checkpoint-root", default="cache/regimeadaptive_v5_runs")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--write-latest", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"Source parquet not found: {source}")
    baseline_artifact = load_regimeadaptive_artifact(str(args.baseline_artifact))
    if baseline_artifact is None:
        raise SystemExit(f"Baseline artifact could not be loaded: {args.baseline_artifact}")
    candidate_seed_artifact = load_regimeadaptive_artifact(str(args.candidate_seed_artifact))
    if candidate_seed_artifact is None:
        raise SystemExit(f"Candidate seed artifact could not be loaded: {args.candidate_seed_artifact}")
    template_names = _parse_group_templates(str(args.group_templates))
    config_signature = _config_signature(args)
    cache_dir = _resolve_path(str(args.cache_dir), "cache/regimeadaptive_v5_sim")
    checkpoint_root = _resolve_path(str(args.checkpoint_root), "cache/regimeadaptive_v5_runs")
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    run_tag = str(args.run_tag or config_signature[:16]).strip()
    run_dir = checkpoint_root / run_tag
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint is None and bool(args.resume):
        raise SystemExit(f"No checkpoint found to resume: {checkpoint_path}")
    if checkpoint is not None:
        if str(checkpoint.get("checkpoint_version", "")) != CHECKPOINT_VERSION:
            raise SystemExit(f"Checkpoint version mismatch: {checkpoint_path}")
        if str(checkpoint.get("config_signature", "")) != config_signature:
            raise SystemExit(f"Checkpoint config mismatch: {checkpoint_path}")
        print(f"resume_checkpoint={checkpoint_path}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    baseline_rule = baseline_artifact.base_rule or {}
    atr_period = int(baseline_rule.get("atr_period", 20) or 20)
    max_hold_bars = int(baseline_rule.get("max_hold_bars", 30) or 30)
    rule_catalog = _parse_rule_specs(str(args.rule_specs), atr_period, max_hold_bars)
    if not rule_catalog:
        raise SystemExit("No rule specs were provided.")
    baseline_rule_id = None
    for rule_id, payload in rule_catalog.items():
        if dict(payload) == dict(baseline_rule):
            baseline_rule_id = str(rule_id)
            break
    if baseline_rule_id is None:
        baseline_rule_id = next(iter(rule_catalog.keys()))

    start_time = _parse_datetime(args.start, is_end=False)
    end_time = _parse_datetime(args.end, is_end=True)
    df, symbol_label = _load_bars(source, str(args.symbol_mode), str(args.symbol_method))
    df = df[(df.index >= start_time) & (df.index <= end_time)].copy()
    if df.empty:
        raise SystemExit("No data in requested range.")

    combo_ids, session_codes = _build_combo_arrays(df.index)
    holiday_mask = _build_holiday_mask(df.index, session_codes)
    _, _, split_meta = _build_split_map(df.index, int(args.valid_years), int(args.test_years))
    hours = np.fromiter((ts.hour for ts in df.index), dtype=np.int8, count=len(df.index))
    minutes = np.fromiter((ts.minute for ts in df.index), dtype=np.int8, count=len(df.index))
    test_positions = np.arange(len(df.index), dtype=np.int32)

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
    pattern_lookbacks = sorted(
        {
            max(1, int(rule_payload.get("pattern_lookback", 8) or 8))
            for rule_payload in rule_catalog.values()
            if str(rule_payload.get("rule_type", "pullback") or "pullback").strip().lower() in {"continuation", "breakout"}
        }
    )
    rolling_cache = _rolling_cache(close, sma_windows)
    atr_cache = {int(atr_period): _atr_array(high, low, close, int(atr_period))}
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
    static_rule_order = [str(baseline_rule_id)] + [
        str(rule_id)
        for rule_id in sorted(rule_catalog.keys())
        if str(rule_id) != str(baseline_rule_id)
    ]
    prebuilt_long_strength_matrix = np.vstack(
        [rule_long_strength[str(rule_id)] for rule_id in static_rule_order]
    ).astype(np.float32)
    prebuilt_short_strength_matrix = np.vstack(
        [rule_short_strength[str(rule_id)] for rule_id in static_rule_order]
    ).astype(np.float32)

    baseline_payload = _baseline_payload(baseline_artifact, rule_catalog, baseline_rule_id)
    state = checkpoint if isinstance(checkpoint, dict) else None
    if state is None:
        state = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "config_signature": config_signature,
            "run_tag": run_tag,
            "stage": "baseline",
            "rule_scan_index": 0,
            "rule_summaries": [],
            "candidate_rows": [],
            "deduped_candidates": [],
            "baseline_eval": {},
            "current_payload": {},
            "current_eval": {},
            "accepted_actions": [],
            "selection_step": 0,
            "toggle_index": 0,
            "toggle_targets": [],
        }
        _write_checkpoint(checkpoint_path, state)
    if str(state.get("stage", "")) == "final":
        artifact_done = Path(str(state.get("artifact_path", "") or "")).expanduser()
        if artifact_done.is_file():
            print(f"artifact={artifact_done.resolve()}")
            return
    if isinstance(state.get("baseline_eval", {}), dict) and state.get("baseline_eval"):
        baseline_eval = dict(state["baseline_eval"])
    else:
        baseline_state = _evaluate_payload_cached(
            baseline_payload,
            cache_dir=cache_dir,
            source=source,
            symbol_mode=str(args.symbol_mode),
            symbol_method=str(args.symbol_method),
            df=df,
            combo_ids=combo_ids,
            session_codes=session_codes,
            holiday_mask=holiday_mask,
            rule_long_strength=rule_long_strength,
            rule_short_strength=rule_short_strength,
            start_time=start_time,
            end_time=end_time,
            contracts=int(args.contracts),
            point_value=point_value,
            fee_per_contract_rt=fee_per_contract_rt,
            split_meta=split_meta,
            template_names=template_names,
            args=args,
            hours=hours,
            minutes=minutes,
            test_positions=test_positions,
            prebuilt_rule_order=static_rule_order,
            prebuilt_long_strength_matrix=prebuilt_long_strength_matrix,
            prebuilt_short_strength_matrix=prebuilt_short_strength_matrix,
            include_side_stats=False,
            include_group_stats=False,
        )
        baseline_eval = dict(baseline_state["eval"])
        state["baseline_eval"] = dict(baseline_eval)
        _write_checkpoint(checkpoint_path, state)
    if not isinstance(state.get("current_payload", {}), dict) or not state.get("current_payload"):
        state["current_payload"] = copy.deepcopy(baseline_payload)
    if not isinstance(state.get("current_eval", {}), dict) or not state.get("current_eval"):
        state["current_eval"] = dict(baseline_eval)
    if str(state.get("stage", "")) == "baseline":
        state["stage"] = "rule_scan"
        _write_checkpoint(checkpoint_path, state)

    seed_signal_policies = _seed_signal_policies(
        candidate_seed_artifact,
        include_top_skipped=int(args.seed_top_skipped),
        min_best_score=float(args.seed_min_best_score),
    )

    candidate_rows: list[dict] = list(state.get("candidate_rows", []))
    rule_summaries: list[dict] = list(state.get("rule_summaries", []))
    rule_items = list(rule_catalog.items())
    if str(state.get("stage", "rule_scan")) == "rule_scan":
        start_rule_index = int(state.get("rule_scan_index", 0))
        for rule_index in range(start_rule_index, len(rule_items)):
            rule_id, rule_payload = rule_items[rule_index]
            payload = _rule_seed_payload(candidate_seed_artifact, str(rule_id), rule_payload, seed_signal_policies)
            rule_state = _evaluate_payload_cached(
                payload,
                cache_dir=cache_dir,
                source=source,
                symbol_mode=str(args.symbol_mode),
                symbol_method=str(args.symbol_method),
                df=df,
                combo_ids=combo_ids,
                session_codes=session_codes,
                holiday_mask=holiday_mask,
                rule_long_strength=rule_long_strength,
                rule_short_strength=rule_short_strength,
                start_time=start_time,
                end_time=end_time,
                contracts=int(args.contracts),
                point_value=point_value,
                fee_per_contract_rt=fee_per_contract_rt,
                split_meta=split_meta,
                template_names=template_names,
                args=args,
                hours=hours,
                minutes=minutes,
                test_positions=test_positions,
                prebuilt_rule_order=static_rule_order,
                prebuilt_long_strength_matrix=prebuilt_long_strength_matrix,
                prebuilt_short_strength_matrix=prebuilt_short_strength_matrix,
                include_side_stats=True,
                include_group_stats=True,
            )
            side_stats = rule_state["side_stats"]
            group_stats = rule_state["group_stats"]
            rule_eval = rule_state["eval"]
            pruned_signal_policies, prune_reasons = _apply_prune_rules(seed_signal_policies, side_stats, args)
            rule_summaries.append(
                {
                    "rule_id": str(rule_id),
                    "rule": dict(rule_payload),
                    "score": float(rule_eval["score"]),
                    "daily_sharpe": float(rule_eval["daily_sharpe"]),
                    "negative_years": int(rule_eval["negative_years"]),
                    "worst_5y_pnl": float(rule_eval["worst_5y_pnl"]),
                    "valid_total": float(rule_eval["valid_total"]),
                    "test_total": float(rule_eval["test_total"]),
                    "trades": int(rule_eval["trades"]),
                    "cache_hit": bool(rule_state["cache_hit"]),
                    "cache_key": str(rule_state["cache_key"]),
                }
            )

            if not side_stats.empty:
                side_stats = side_stats.copy()
                side_stats["candidate_score"] = side_stats.apply(
                    lambda row: _candidate_rank_score(row.train_total, row.valid_total, row.test_total, float(args.train_score_weight)),
                    axis=1,
                )
                retained_keys = set(_retained_side_keys(pruned_signal_policies))
                retained_rows: list[dict] = []
                for row in side_stats.itertuples(index=False):
                    key = (str(row.combo_key), str(row.original_signal))
                    if key not in retained_keys:
                        continue
                    record = (pruned_signal_policies.get(str(row.combo_key), {}) or {}).get(str(row.original_signal), {})
                    if not isinstance(record, dict):
                        continue
                    retained_rows.append(
                        {
                            "candidate_type": "exact",
                            "target_key": str(row.combo_key),
                            "original_signal": str(row.original_signal),
                            "policy": str(record.get("policy", "skip")),
                            "early_exit_enabled": bool(record.get("early_exit_enabled", _default_regimeadaptive_early_exit_enabled())),
                            "rule_id": str(rule_id),
                            "candidate_score": float(row.candidate_score),
                            "train_total": float(row.train_total),
                            "valid_total": float(row.valid_total),
                            "test_total": float(row.test_total),
                            "support_total": int(row.support_total),
                            "support_recent": int(row.support_recent),
                            "distinct_combo_count": 1,
                            "template_name": "exact",
                            "prune_reason_count": int(len(prune_reasons)),
                        }
                    )
                retained_rows.sort(
                    key=lambda item: (item["candidate_score"], item["test_total"], item["valid_total"], item["support_total"]),
                    reverse=True,
                )
                candidate_rows.extend(retained_rows[: int(args.max_candidates_per_rule)])

            if not group_stats.empty:
                group_stats = group_stats.copy()
                group_stats["candidate_score"] = group_stats.apply(
                    lambda row: (
                        _candidate_rank_score(row.train_total, row.valid_total, row.test_total, float(args.train_score_weight))
                        + (float(args.group_diversity_bonus) * max(0, int(row.distinct_combo_count) - 1))
                    ),
                    axis=1,
                )
                group_rows: list[dict] = []
                for row in group_stats.itertuples(index=False):
                    if not _group_row_passes(row, args):
                        continue
                    group_rows.append(
                        {
                            "candidate_type": "group",
                            "target_key": str(row.group_key),
                            "original_signal": str(row.original_signal),
                            "policy": str(row.policy),
                            "early_exit_enabled": _default_regimeadaptive_early_exit_enabled(),
                            "rule_id": str(rule_id),
                            "candidate_score": float(row.candidate_score),
                            "train_total": float(row.train_total),
                            "valid_total": float(row.valid_total),
                            "test_total": float(row.test_total),
                            "support_total": int(row.support_total),
                            "support_recent": int(row.support_recent),
                            "distinct_combo_count": int(row.distinct_combo_count),
                            "template_name": str(row.template_name),
                        }
                    )
                group_rows.sort(
                    key=lambda item: (
                        item["candidate_score"],
                        item["test_total"],
                        item["valid_total"],
                        item["distinct_combo_count"],
                        item["support_total"],
                    ),
                    reverse=True,
                )
                candidate_rows.extend(group_rows[: int(args.max_group_candidates_per_rule)])

            state.update(
                {
                    "stage": "rule_scan",
                    "rule_scan_index": int(rule_index + 1),
                    "rule_summaries": rule_summaries,
                    "candidate_rows": candidate_rows,
                }
            )
            _write_checkpoint(checkpoint_path, state)
            print(
                f"rule_progress={rule_index + 1}/{len(rule_items)} rule_id={rule_id} cache_hit={rule_state['cache_hit']} checkpoint={checkpoint_path}"
            )

        state.update(
            {
                "stage": "selection",
                "deduped_candidates": _dedupe_candidates(candidate_rows, baseline_payload, int(args.max_candidates_total)),
                "current_payload": copy.deepcopy(baseline_payload),
                "current_eval": dict(baseline_eval),
                "accepted_actions": [],
                "selection_step": 0,
                "toggle_index": 0,
                "toggle_targets": [],
            }
        )
        _write_checkpoint(checkpoint_path, state)

    deduped_candidates: list[dict] = list(state.get("deduped_candidates", []))
    if not deduped_candidates:
        deduped_candidates = _dedupe_candidates(candidate_rows, baseline_payload, int(args.max_candidates_total))
        state["deduped_candidates"] = deduped_candidates
        _write_checkpoint(checkpoint_path, state)

    current_payload = copy.deepcopy(state.get("current_payload", baseline_payload))
    current_eval = dict(state.get("current_eval", baseline_eval))
    accepted_actions: list[dict] = list(state.get("accepted_actions", []))
    toggle_targets: list[list[str]] = list(state.get("toggle_targets", []))

    if str(state.get("stage", "selection")) == "selection":
        start_selection_step = int(state.get("selection_step", 0))
        for selection_step in range(start_selection_step, int(args.max_selection_steps)):
            best_payload = None
            best_eval = None
            best_action = None
            best_cache_hit = False
            for candidate in deduped_candidates:
                candidate_payload = _apply_candidate(current_payload, candidate, bool(candidate["early_exit_enabled"]))
                candidate_state = _evaluate_payload_cached(
                    candidate_payload,
                    cache_dir=cache_dir,
                    source=source,
                    symbol_mode=str(args.symbol_mode),
                    symbol_method=str(args.symbol_method),
                    df=df,
                    combo_ids=combo_ids,
                    session_codes=session_codes,
                    holiday_mask=holiday_mask,
                    rule_long_strength=rule_long_strength,
                    rule_short_strength=rule_short_strength,
                    start_time=start_time,
                    end_time=end_time,
                    contracts=int(args.contracts),
                    point_value=point_value,
                    fee_per_contract_rt=fee_per_contract_rt,
                    split_meta=split_meta,
                    template_names=template_names,
                    args=args,
                    hours=hours,
                    minutes=minutes,
                    test_positions=test_positions,
                    prebuilt_rule_order=static_rule_order,
                    prebuilt_long_strength_matrix=prebuilt_long_strength_matrix,
                    prebuilt_short_strength_matrix=prebuilt_short_strength_matrix,
                    include_side_stats=False,
                    include_group_stats=False,
                )
                candidate_eval = candidate_state["eval"]
                if best_eval is None or float(candidate_eval["score"]) > float(best_eval["score"]):
                    best_payload = candidate_payload
                    best_eval = candidate_eval
                    best_cache_hit = bool(candidate_state["cache_hit"])
                    best_action = {
                        "candidate_type": str(candidate["candidate_type"]),
                        "target_key": str(candidate["target_key"]),
                        "original_signal": str(candidate["original_signal"]),
                        "policy": str(candidate["policy"]),
                        "rule_id": str(candidate["rule_id"]),
                        "template_name": str(candidate.get("template_name", "")),
                        "distinct_combo_count": int(candidate.get("distinct_combo_count", 1)),
                        "early_exit_enabled": bool(candidate["early_exit_enabled"]),
                        "score": float(candidate_eval["score"]),
                        "daily_sharpe": float(candidate_eval["daily_sharpe"]),
                        "negative_years": int(candidate_eval["negative_years"]),
                        "worst_5y_pnl": float(candidate_eval["worst_5y_pnl"]),
                        "valid_total": float(candidate_eval["valid_total"]),
                        "test_total": float(candidate_eval["test_total"]),
                        "trades": int(candidate_eval["trades"]),
                    }
                del candidate_state
                gc.collect()
            if best_eval is None or best_payload is None or best_action is None:
                break
            if float(best_eval["score"]) <= float(current_eval["score"]):
                break
            current_payload = best_payload
            current_eval = dict(best_eval)
            accepted_actions.append(best_action)
            state.update(
                {
                    "stage": "selection",
                    "current_payload": copy.deepcopy(current_payload),
                    "current_eval": dict(current_eval),
                    "accepted_actions": accepted_actions,
                    "selection_step": int(selection_step + 1),
                }
            )
            _write_checkpoint(checkpoint_path, state)
            print(
                f"selection_step={selection_step + 1}/{int(args.max_selection_steps)} "
                f"score={float(current_eval['score']):.2f} cache_hit={best_cache_hit} checkpoint={checkpoint_path}"
            )

        toggle_targets = [list(target) for target in _active_policy_targets(current_payload)]
        state.update(
            {
                "stage": "toggle",
                "current_payload": copy.deepcopy(current_payload),
                "current_eval": dict(current_eval),
                "accepted_actions": accepted_actions,
                "toggle_index": int(state.get("toggle_index", 0)),
                "toggle_targets": toggle_targets,
            }
        )
        _write_checkpoint(checkpoint_path, state)

    if str(state.get("stage", "")) == "toggle":
        if not toggle_targets:
            toggle_targets = [list(target) for target in _active_policy_targets(current_payload)]
            state["toggle_targets"] = toggle_targets
            _write_checkpoint(checkpoint_path, state)
        start_toggle_index = int(state.get("toggle_index", 0))
        for toggle_index in range(start_toggle_index, len(toggle_targets)):
            candidate_type, target_key, original_signal = toggle_targets[toggle_index]
            candidate_payload = copy.deepcopy(current_payload)
            collection_name = "group_signal_policies" if str(candidate_type) == "group" else "signal_policies"
            side_map = candidate_payload.setdefault(collection_name, {}).setdefault(str(target_key), {})
            record = side_map.setdefault(
                str(original_signal),
                {"policy": "skip", "early_exit_enabled": _default_regimeadaptive_early_exit_enabled()},
            )
            current_ee = bool(record.get("early_exit_enabled", _default_regimeadaptive_early_exit_enabled()))
            candidate_payload[collection_name][str(target_key)][str(original_signal)]["early_exit_enabled"] = not current_ee
            if collection_name == "signal_policies":
                candidate_payload["combo_policies"] = _merge_combo_policies(
                    candidate_payload.get("combo_policies", {}) or {},
                    candidate_payload.get("signal_policies", {}) or {},
                )
            candidate_state = _evaluate_payload_cached(
                candidate_payload,
                cache_dir=cache_dir,
                source=source,
                symbol_mode=str(args.symbol_mode),
                symbol_method=str(args.symbol_method),
                df=df,
                combo_ids=combo_ids,
                session_codes=session_codes,
                holiday_mask=holiday_mask,
                rule_long_strength=rule_long_strength,
                rule_short_strength=rule_short_strength,
                start_time=start_time,
                end_time=end_time,
                contracts=int(args.contracts),
                point_value=point_value,
                fee_per_contract_rt=fee_per_contract_rt,
                split_meta=split_meta,
                template_names=template_names,
                args=args,
                hours=hours,
                minutes=minutes,
                test_positions=test_positions,
                prebuilt_rule_order=static_rule_order,
                prebuilt_long_strength_matrix=prebuilt_long_strength_matrix,
                prebuilt_short_strength_matrix=prebuilt_short_strength_matrix,
                include_side_stats=False,
                include_group_stats=False,
            )
            candidate_eval = candidate_state["eval"]
            if float(candidate_eval["score"]) > float(current_eval["score"]):
                current_payload = candidate_payload
                current_eval = dict(candidate_eval)
            state.update(
                {
                    "stage": "toggle",
                    "current_payload": copy.deepcopy(current_payload),
                    "current_eval": dict(current_eval),
                    "toggle_index": int(toggle_index + 1),
                    "toggle_targets": toggle_targets,
                }
            )
            _write_checkpoint(checkpoint_path, state)
            print(
                f"toggle_progress={toggle_index + 1}/{len(toggle_targets)} "
                f"score={float(current_eval['score']):.2f} cache_hit={bool(candidate_state['cache_hit'])} checkpoint={checkpoint_path}"
            )
            del candidate_state
            gc.collect()
        state.update(
            {
                "stage": "final",
                "current_payload": copy.deepcopy(current_payload),
                "current_eval": dict(current_eval),
                "accepted_actions": accepted_actions,
                "toggle_index": len(toggle_targets),
                "toggle_targets": toggle_targets,
            }
        )
        _write_checkpoint(checkpoint_path, state)

    final_state = _evaluate_payload_cached(
        current_payload,
        cache_dir=cache_dir,
        source=source,
        symbol_mode=str(args.symbol_mode),
        symbol_method=str(args.symbol_method),
        df=df,
        combo_ids=combo_ids,
        session_codes=session_codes,
        holiday_mask=holiday_mask,
        rule_long_strength=rule_long_strength,
        rule_short_strength=rule_short_strength,
        start_time=start_time,
        end_time=end_time,
        contracts=int(args.contracts),
        point_value=point_value,
        fee_per_contract_rt=fee_per_contract_rt,
        split_meta=split_meta,
        template_names=template_names,
        args=args,
        hours=hours,
        minutes=minutes,
        test_positions=test_positions,
        prebuilt_rule_order=static_rule_order,
        prebuilt_long_strength_matrix=prebuilt_long_strength_matrix,
        prebuilt_short_strength_matrix=prebuilt_short_strength_matrix,
        include_side_stats=True,
        include_group_stats=True,
    )
    current_result = final_state["result"]
    current_eval = final_state["eval"]
    final_side_stats = final_state["side_stats"]
    final_group_stats = final_state["group_stats"]
    artifact_root = _resolve_path(str(args.artifact_root), f"artifacts/regimeadaptive_v5_{run_tag}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / "regimeadaptive_v5_artifact.json"

    current_payload["version"] = "regimeadaptive_v5_multirule_hierarchical"
    current_payload["policy_mode"] = "side_specific_execution_multirule_hierarchical"
    current_payload.update(
        {
            "created_at": dt.datetime.now(NY_TZ).isoformat(),
            "source_data_path": str(source),
            "symbol": symbol_label,
            "range_start": start_time.isoformat(),
            "range_end": end_time.isoformat(),
            "split_meta": split_meta,
            "training_config": {
                "baseline_artifact": str(getattr(baseline_artifact, "path", args.baseline_artifact)),
                "candidate_seed_artifact": str(getattr(candidate_seed_artifact, "path", args.candidate_seed_artifact)),
                "run_tag": str(run_tag),
                "config_signature": str(config_signature),
                "cache_dir": str(cache_dir),
                "checkpoint_path": str(checkpoint_path),
                "contracts": int(args.contracts),
                "rule_specs": str(args.rule_specs),
                "group_templates": template_names,
                "group_min_combo_count": int(args.group_min_combo_count),
                "group_min_total_trades": int(args.group_min_total_trades),
                "group_min_recent_trades": int(args.group_min_recent_trades),
                "group_diversity_bonus": float(args.group_diversity_bonus),
                "min_total_trades": int(args.min_total_trades),
                "min_recent_trades": int(args.min_recent_trades),
                "min_total_net": float(args.min_total_net),
                "min_train_avg": float(args.min_train_avg),
                "min_split_edge": float(args.min_split_edge),
                "min_positive_oos_splits": int(args.min_positive_oos_splits),
                "train_score_weight": float(args.train_score_weight),
                "seed_top_skipped": int(args.seed_top_skipped),
                "seed_min_best_score": float(args.seed_min_best_score),
                "max_candidates_per_rule": int(args.max_candidates_per_rule),
                "max_group_candidates_per_rule": int(args.max_group_candidates_per_rule),
                "max_candidates_total": int(args.max_candidates_total),
                "max_selection_steps": int(args.max_selection_steps),
                "robust_sharpe_weight": float(args.robust_sharpe_weight),
                "negative_year_penalty": float(args.negative_year_penalty),
                "worst_3y_weight": float(args.worst_3y_weight),
                "worst_5y_weight": float(args.worst_5y_weight),
                "max_drawdown_penalty": float(args.max_drawdown_penalty),
                "yearly_std_penalty": float(args.yearly_std_penalty),
                "trade_count_weight": float(args.trade_count_weight),
            },
            "candidate_rule_summaries": rule_summaries,
            "accepted_actions": accepted_actions,
            "best_candidate": {
                "score": float(current_eval["score"]),
                "train_total": float(current_eval["train_total"]),
                "valid_total": float(current_eval["valid_total"]),
                "test_total": float(current_eval["test_total"]),
                "trades": int(current_eval["trades"]),
                "wins": int(current_eval["wins"]),
                "losses": int(current_eval["losses"]),
                "winrate": float(current_eval["winrate"]),
                "avg_trade_net": float(current_eval["avg_trade_net"]),
                "profit_factor": None if current_eval["profit_factor"] is None else float(current_eval["profit_factor"]),
                "daily_sharpe": float(current_eval["daily_sharpe"]),
                "negative_years": int(current_eval["negative_years"]),
                "worst_3y_pnl": float(current_eval["worst_3y_pnl"]),
                "worst_5y_pnl": float(current_eval["worst_5y_pnl"]),
                "yearly_pnl_std": float(current_eval["yearly_pnl_std"]),
                "max_drawdown": float(current_eval["max_drawdown"]),
            },
            "baseline_candidate": {
                "score": float(baseline_eval["score"]),
                "train_total": float(baseline_eval["train_total"]),
                "valid_total": float(baseline_eval["valid_total"]),
                "test_total": float(baseline_eval["test_total"]),
                "trades": int(baseline_eval["trades"]),
                "wins": int(baseline_eval["wins"]),
                "losses": int(baseline_eval["losses"]),
                "winrate": float(baseline_eval["winrate"]),
                "avg_trade_net": float(baseline_eval["avg_trade_net"]),
                "profit_factor": None if baseline_eval["profit_factor"] is None else float(baseline_eval["profit_factor"]),
                "daily_sharpe": float(baseline_eval["daily_sharpe"]),
                "negative_years": int(baseline_eval["negative_years"]),
                "worst_3y_pnl": float(baseline_eval["worst_3y_pnl"]),
                "worst_5y_pnl": float(baseline_eval["worst_5y_pnl"]),
                "yearly_pnl_std": float(baseline_eval["yearly_pnl_std"]),
                "max_drawdown": float(baseline_eval["max_drawdown"]),
            },
            "execution_result": {key: value for key, value in current_result.items() if key != "trade_log"},
            "summary": {
                "retained_combo_count": int(len({combo for combo, _ in _retained_side_keys(current_payload.get("signal_policies", {}) or {})})),
                "retained_side_policy_count": int(len(_retained_side_keys(current_payload.get("signal_policies", {}) or {}))),
                "retained_group_policy_count": int(
                    sum(
                        1
                        for side_map in (current_payload.get("group_signal_policies", {}) or {}).values()
                        if isinstance(side_map, dict)
                        for record in side_map.values()
                        if isinstance(record, dict) and str(record.get("policy", "skip")).strip().lower() in {"normal", "reversed"}
                    )
                ),
                "trade_count": int(current_result.get("trades", 0)),
                "selected_rule_count": int(
                    len(
                        {
                            str(trade.get("rule_id", "") or "")
                            for trade in current_result.get("trade_log", []) or []
                            if str(trade.get("rule_id", "") or "").strip()
                        }
                    )
                ),
            },
            "side_policy_stats": final_side_stats.to_dict("records") if isinstance(final_side_stats, pd.DataFrame) and not final_side_stats.empty else [],
            "group_policy_stats": final_group_stats.to_dict("records") if isinstance(final_group_stats, pd.DataFrame) and not final_group_stats.empty else [],
        }
    )

    artifact_path.write_text(json.dumps(_json_safe(current_payload), indent=2, ensure_ascii=True), encoding="utf-8")
    state.update(
        {
            "stage": "final",
            "current_payload": copy.deepcopy(current_payload),
            "current_eval": dict(current_eval),
            "accepted_actions": accepted_actions,
            "toggle_targets": toggle_targets,
            "artifact_path": str(artifact_path),
        }
    )
    _write_checkpoint(checkpoint_path, state)
    print(f"artifact={artifact_path}")
    if bool(args.write_latest):
        latest_dir = ROOT / "artifacts" / "regimeadaptive_v5"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_path = latest_dir / "latest.json"
        shutil.copyfile(artifact_path, latest_path)
        print(f"latest={latest_path}")


if __name__ == "__main__":
    main()
