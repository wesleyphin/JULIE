import argparse
import copy
import datetime as dt
import json
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
    _artifact_rule_order,
    _build_artifact_lookups,
    _build_artifact_rule_lookup,
    _build_multirule_signal_arrays,
    _build_rule_strength_arrays,
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


def _rule_id_for_payload(rule_payload: dict) -> str:
    rule_type = str(rule_payload.get("rule_type", "pullback") or "pullback").strip().lower()
    fast = int(rule_payload.get("sma_fast", 0) or 0)
    slow = int(rule_payload.get("sma_slow", 0) or 0)
    cross = float(rule_payload.get("cross_atr_mult", 0.0) or 0.0)
    cross_text = str(cross).replace("-", "m").replace(".", "p")
    if rule_type == "pullback":
        return f"f{fast}_s{slow}_x{cross_text}"
    lookback = max(1, int(rule_payload.get("pattern_lookback", 1) or 1))
    prefix = "cont" if rule_type == "continuation" else "brk"
    return f"{prefix}_f{fast}_s{slow}_l{lookback}_x{cross_text}"


def _parse_rule_specs(raw: str, atr_period: int, max_hold_bars: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in str(raw or "").split(","):
        text = item.strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(":")]
        if len(parts) == 3:
            payload = {
                "rule_type": "pullback",
                "sma_fast": int(parts[0]),
                "sma_slow": int(parts[1]),
                "cross_atr_mult": float(parts[2]),
                "atr_period": int(atr_period),
                "max_hold_bars": int(max_hold_bars),
            }
        elif len(parts) == 4 and parts[0].strip().lower() in {"pullback", "pb"}:
            payload = {
                "rule_type": "pullback",
                "sma_fast": int(parts[1]),
                "sma_slow": int(parts[2]),
                "cross_atr_mult": float(parts[3]),
                "atr_period": int(atr_period),
                "max_hold_bars": int(max_hold_bars),
            }
        elif len(parts) == 5 and parts[0].strip().lower() in {"continuation", "cont", "breakout", "brk"}:
            raw_type = parts[0].strip().lower()
            payload = {
                "rule_type": "continuation" if raw_type in {"continuation", "cont"} else "breakout",
                "sma_fast": int(parts[1]),
                "sma_slow": int(parts[2]),
                "pattern_lookback": int(parts[3]),
                "cross_atr_mult": float(parts[4]),
                "atr_period": int(atr_period),
                "max_hold_bars": int(max_hold_bars),
            }
            if payload["rule_type"] == "continuation":
                payload["touch_atr_mult"] = 0.25
        else:
            raise ValueError(f"Unsupported rule spec: {text}")
        rule_id = _rule_id_for_payload(payload)
        out[rule_id] = payload
    return out


def _merge_combo_policies(existing_combo_policies: dict, signal_policies: dict[str, dict[str, dict]]) -> dict[str, dict]:
    coarse_combo_policies = _coarse_combo_policies(signal_policies)
    existing_mapping = existing_combo_policies if isinstance(existing_combo_policies, dict) else {}
    merged: dict[str, dict] = {}
    for combo_key in sorted(set(existing_mapping) | set(coarse_combo_policies)):
        existing_record = existing_mapping.get(combo_key, {})
        next_record = dict(existing_record) if isinstance(existing_record, dict) else {}
        coarse_record = coarse_combo_policies.get(combo_key, {})
        if isinstance(coarse_record, dict) and str(coarse_record.get("policy", "") or "").strip():
            next_record["policy"] = str(coarse_record.get("policy"))
        elif "policy" in next_record and combo_key not in coarse_combo_policies:
            next_record.pop("policy", None)
        for side_name in ("LONG", "SHORT"):
            side_payload = next_record.get(side_name)
            if not isinstance(side_payload, dict):
                next_record.pop(side_name, None)
        if next_record:
            merged[str(combo_key)] = next_record
    return merged


def _attach_rule_catalog(payload: dict, rule_catalog: dict[str, dict], default_rule_id: str) -> dict:
    next_payload = copy.deepcopy(payload)
    next_payload["version"] = "regimeadaptive_v3_multirule"
    next_payload["policy_mode"] = "side_specific_execution_multirule"
    next_payload["rule_catalog"] = copy.deepcopy(rule_catalog)
    next_payload["default_rule_id"] = str(default_rule_id)
    if str(default_rule_id) in rule_catalog:
        next_payload["base_rule"] = dict(rule_catalog[str(default_rule_id)])
    return next_payload


def _baseline_payload(
    baseline_artifact: RegimeAdaptiveArtifact,
    rule_catalog: dict[str, dict],
    default_rule_id: str,
) -> dict:
    payload = _attach_rule_catalog(baseline_artifact.payload, rule_catalog, default_rule_id)
    signal_policies = payload.get("signal_policies", {}) if isinstance(payload.get("signal_policies", {}), dict) else {}
    for combo_key, side_map in signal_policies.items():
        if not isinstance(side_map, dict):
            continue
        for original_signal, record in side_map.items():
            if not isinstance(record, dict):
                continue
            policy = str(record.get("policy", "skip")).strip().lower()
            if policy in {"normal", "reversed"} and not str(record.get("rule_id", "") or "").strip():
                record["rule_id"] = str(default_rule_id)
    payload["combo_policies"] = _merge_combo_policies(payload.get("combo_policies", {}), signal_policies)
    return payload


def _rule_seed_payload(
    candidate_seed_artifact: RegimeAdaptiveArtifact,
    rule_id: str,
    rule_payload: dict,
    seed_signal_policies: dict[str, dict[str, dict]],
) -> dict:
    seeded = copy.deepcopy(seed_signal_policies)
    for side_map in seeded.values():
        if not isinstance(side_map, dict):
            continue
        for record in side_map.values():
            if not isinstance(record, dict):
                continue
            policy = str(record.get("policy", "skip")).strip().lower()
            if policy in {"normal", "reversed"}:
                record["rule_id"] = str(rule_id)
    payload = {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "version": "regimeadaptive_v3_multirule",
        "policy_mode": "side_specific_execution_multirule",
        "base_rule": dict(rule_payload),
        "combo_policies": _merge_combo_policies(candidate_seed_artifact.payload.get("combo_policies", {}) or {}, seeded),
        "signal_policies": seeded,
        "session_defaults": copy.deepcopy(candidate_seed_artifact.payload.get("session_defaults", {}) or {}),
        "global_default": copy.deepcopy(candidate_seed_artifact.payload.get("global_default", {}) or {}),
        "seed_artifact_path": str(getattr(candidate_seed_artifact, "path", "")),
        "rule_catalog": {str(rule_id): dict(rule_payload)},
        "default_rule_id": str(rule_id),
        "metadata": {"stage": "v3_rule_seed", "rule_id": str(rule_id)},
    }
    return payload


def _simulate_payload_v3(
    df: pd.DataFrame,
    combo_ids: np.ndarray,
    session_codes: np.ndarray,
    holiday_mask: np.ndarray,
    payload: dict,
    rule_long_strength: dict[str, np.ndarray],
    rule_short_strength: dict[str, np.ndarray],
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    contracts: int,
    point_value: float,
    fee_per_contract_rt: float,
    hours: np.ndarray | None = None,
    minutes: np.ndarray | None = None,
    test_positions: np.ndarray | None = None,
    prebuilt_rule_order: list[str] | None = None,
    prebuilt_long_strength_matrix: np.ndarray | None = None,
    prebuilt_short_strength_matrix: np.ndarray | None = None,
) -> dict:
    artifact = RegimeAdaptiveArtifact(payload, Path("in_memory_regimeadaptive_v3.json"))
    policy_lookup, early_exit_lookup, long_sl_lookup, long_tp_lookup, short_sl_lookup, short_tp_lookup = _build_artifact_lookups(artifact)
    if (
        prebuilt_rule_order is not None
        and prebuilt_long_strength_matrix is not None
        and prebuilt_short_strength_matrix is not None
        and set(str(rule_id) for rule_id in prebuilt_rule_order) == set((artifact.rule_catalog or {}).keys())
    ):
        rule_order = list(prebuilt_rule_order)
        long_strength_matrix = prebuilt_long_strength_matrix
        short_strength_matrix = prebuilt_short_strength_matrix
    else:
        rule_order = _artifact_rule_order(artifact)
        long_strength_matrix = np.vstack([rule_long_strength[str(rule_id)] for rule_id in rule_order]).astype(np.float32)
        short_strength_matrix = np.vstack([rule_short_strength[str(rule_id)] for rule_id in rule_order]).astype(np.float32)
    rule_lookup = _build_artifact_rule_lookup(artifact, rule_order)
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
    return _simulate(
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
        selected_rule_index=selected_rule_index,
        rule_order=rule_order,
        hours=hours,
        minutes=minutes,
        test_positions=test_positions,
    )


def _candidate_score(row, train_score_weight: float) -> float:
    return float(row.test_total) + (0.5 * float(row.valid_total)) + (float(train_score_weight) * float(row.train_total))


def _apply_side_candidate(payload: dict, candidate: dict, early_exit_enabled: bool) -> dict:
    next_payload = copy.deepcopy(payload)
    combo_key = str(candidate["combo_key"])
    original_signal = str(candidate["original_signal"])
    next_payload.setdefault("signal_policies", {}).setdefault(combo_key, {})[original_signal] = {
        "policy": str(candidate["policy"]),
        "early_exit_enabled": bool(early_exit_enabled),
        "rule_id": str(candidate["rule_id"]),
    }
    next_payload["combo_policies"] = _merge_combo_policies(
        next_payload.get("combo_policies", {}) or {},
        next_payload.get("signal_policies", {}) or {},
    )
    return next_payload


def _side_signature(payload: dict) -> set[tuple[str, str, str, str, bool]]:
    out: set[tuple[str, str, str, str, bool]] = set()
    signal_policies = payload.get("signal_policies", {}) if isinstance(payload.get("signal_policies", {}), dict) else {}
    for combo_key, side_map in signal_policies.items():
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
                    str(combo_key),
                    str(original_signal),
                    policy,
                    str(record.get("rule_id", "") or ""),
                    bool(record.get("early_exit_enabled", _default_regimeadaptive_early_exit_enabled())),
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RegimeAdaptive v3 with a shared multi-base-rule catalog and greedy side selection."
    )
    parser.add_argument("--source", default="es_master_outrights.parquet")
    parser.add_argument("--baseline-artifact", default="artifacts/regimeadaptive_v2/latest.json")
    parser.add_argument("--candidate-seed-artifact", default="artifacts/regimeadaptive_robust/latest.json")
    parser.add_argument("--symbol-mode", default="auto_by_day")
    parser.add_argument("--symbol-method", default="volume")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--rule-specs", default="50:200:2.0,50:200:1.5,34:200:1.5,50:300:2.0")
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
    parser.add_argument("--max-candidates-total", type=int, default=12)
    parser.add_argument("--max-selection-steps", type=int, default=4)
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

    baseline_rule = baseline_artifact.base_rule or {}
    atr_period = int(baseline_rule.get("atr_period", 20) or 20)
    max_hold_bars = int(baseline_rule.get("max_hold_bars", 30) or 30)
    rule_catalog = _parse_rule_specs(str(args.rule_specs), atr_period, max_hold_bars)
    if not rule_catalog:
        raise SystemExit("No rule specs were provided.")

    baseline_rule_id = _rule_id_for_payload(baseline_rule or next(iter(rule_catalog.values())))
    if baseline_rule_id not in rule_catalog:
        rule_catalog[baseline_rule_id] = {
            "sma_fast": int(baseline_rule.get("sma_fast", 50) or 50),
            "sma_slow": int(baseline_rule.get("sma_slow", 200) or 200),
            "cross_atr_mult": float(baseline_rule.get("cross_atr_mult", 2.0) or 2.0),
            "atr_period": int(atr_period),
            "max_hold_bars": int(max_hold_bars),
        }

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

    baseline_payload = _baseline_payload(baseline_artifact, rule_catalog, baseline_rule_id)
    baseline_result = _simulate_payload_v3(
        df,
        combo_ids,
        session_codes,
        holiday_mask,
        baseline_payload,
        rule_long_strength,
        rule_short_strength,
        start_time,
        end_time,
        int(args.contracts),
        point_value,
        fee_per_contract_rt,
    )
    baseline_eval = _evaluate_trade_log(
        baseline_result.get("trade_log", []) or [],
        split_meta,
        float(args.train_score_weight),
    )

    seed_signal_policies = _seed_signal_policies(
        candidate_seed_artifact,
        include_top_skipped=int(args.seed_top_skipped),
        min_best_score=float(args.seed_min_best_score),
    )

    candidate_rows: list[dict] = []
    rule_summaries: list[dict] = []
    for rule_id, rule_payload in rule_catalog.items():
        payload = _rule_seed_payload(candidate_seed_artifact, str(rule_id), rule_payload, seed_signal_policies)
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
        )
        side_stats = _summarize_side_groups(result.get("trade_log", []) or [], split_meta)
        pruned_signal_policies, prune_reasons = _apply_prune_rules(seed_signal_policies, side_stats, args)
        rule_eval = _evaluate_trade_log(
            result.get("trade_log", []) or [],
            split_meta,
            float(args.train_score_weight),
        )
        rule_summaries.append(
            {
                "rule_id": str(rule_id),
                "rule": dict(rule_payload),
                "score": float(rule_eval["score"]),
                "valid_total": float(rule_eval["valid_total"]),
                "test_total": float(rule_eval["test_total"]),
                "trades": int(rule_eval["trades"]),
            }
        )
        if side_stats.empty:
            continue
        side_stats = side_stats.copy()
        side_stats["candidate_score"] = side_stats.apply(
            lambda row: _candidate_score(row, float(args.train_score_weight)),
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
                    "combo_key": str(row.combo_key),
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
                    "prune_reason_count": int(len(prune_reasons)),
                }
            )
        retained_rows.sort(
            key=lambda item: (item["candidate_score"], item["test_total"], item["valid_total"], item["support_total"]),
            reverse=True,
        )
        candidate_rows.extend(retained_rows[: int(args.max_candidates_per_rule)])

    candidate_rows.sort(
        key=lambda item: (item["candidate_score"], item["test_total"], item["valid_total"], item["support_total"]),
        reverse=True,
    )
    deduped_candidates: list[dict] = []
    seen_signatures: set[tuple[str, str, str, str]] = set()
    baseline_signature = _side_signature(baseline_payload)
    for row in candidate_rows:
        sig = (row["combo_key"], row["original_signal"], row["policy"], row["rule_id"])
        if sig in seen_signatures:
            continue
        if (
            row["combo_key"],
            row["original_signal"],
            row["policy"],
            row["rule_id"],
            bool(row["early_exit_enabled"]),
        ) in baseline_signature:
            continue
        seen_signatures.add(sig)
        deduped_candidates.append(row)
        if len(deduped_candidates) >= int(args.max_candidates_total):
            break

    current_payload = copy.deepcopy(baseline_payload)
    current_result = baseline_result
    current_eval = dict(baseline_eval)
    accepted_actions: list[dict] = []

    for _ in range(int(args.max_selection_steps)):
        best_payload = None
        best_result = None
        best_eval = None
        best_action = None
        for candidate in deduped_candidates:
            for candidate_ee in (bool(candidate["early_exit_enabled"]), not bool(candidate["early_exit_enabled"])):
                candidate_payload = _apply_side_candidate(current_payload, candidate, candidate_ee)
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
                candidate_eval = _evaluate_trade_log(
                    candidate_result.get("trade_log", []) or [],
                    split_meta,
                    float(args.train_score_weight),
                )
                if best_eval is None or float(candidate_eval["score"]) > float(best_eval["score"]):
                    best_payload = candidate_payload
                    best_result = candidate_result
                    best_eval = candidate_eval
                    best_action = {
                        "combo_key": str(candidate["combo_key"]),
                        "original_signal": str(candidate["original_signal"]),
                        "policy": str(candidate["policy"]),
                        "rule_id": str(candidate["rule_id"]),
                        "early_exit_enabled": bool(candidate_ee),
                        "score": float(candidate_eval["score"]),
                        "valid_total": float(candidate_eval["valid_total"]),
                        "test_total": float(candidate_eval["test_total"]),
                        "trades": int(candidate_eval["trades"]),
                    }
        if best_eval is None or best_payload is None or best_result is None or best_action is None:
            break
        if float(best_eval["score"]) <= float(current_eval["score"]):
            break
        current_payload = best_payload
        current_result = best_result
        current_eval = best_eval
        accepted_actions.append(best_action)

    retained_keys = _retained_side_keys(current_payload.get("signal_policies", {}) or {})
    for combo_key, original_signal in retained_keys:
        side_map = current_payload.setdefault("signal_policies", {}).setdefault(combo_key, {})
        record = side_map.setdefault(
            original_signal,
            {"policy": "skip", "early_exit_enabled": _default_regimeadaptive_early_exit_enabled()},
        )
        current_ee = bool(record.get("early_exit_enabled", _default_regimeadaptive_early_exit_enabled()))
        candidate_payload = copy.deepcopy(current_payload)
        candidate_payload["signal_policies"][combo_key][original_signal]["early_exit_enabled"] = not current_ee
        candidate_payload["combo_policies"] = _merge_combo_policies(
            candidate_payload.get("combo_policies", {}) or {},
            candidate_payload.get("signal_policies", {}) or {},
        )
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
        candidate_eval = _evaluate_trade_log(
            candidate_result.get("trade_log", []) or [],
            split_meta,
            float(args.train_score_weight),
        )
        if float(candidate_eval["score"]) > float(current_eval["score"]):
            current_payload = candidate_payload
            current_result = candidate_result
            current_eval = candidate_eval

    final_side_stats = _summarize_side_groups(current_result.get("trade_log", []) or [], split_meta)
    if str(args.artifact_root or "").strip():
        artifact_root = Path(args.artifact_root).expanduser().resolve()
    else:
        artifact_root = ROOT / "artifacts" / f"regimeadaptive_v3_{dt.datetime.now(NY_TZ).strftime('%Y%m%d_%H%M%S')}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / "regimeadaptive_v3_artifact.json"

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
                "contracts": int(args.contracts),
                "rule_specs": str(args.rule_specs),
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
                "max_candidates_total": int(args.max_candidates_total),
                "max_selection_steps": int(args.max_selection_steps),
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
            },
            "execution_result": {key: value for key, value in current_result.items() if key != "trade_log"},
            "summary": {
                "retained_combo_count": int(len({combo for combo, _ in _retained_side_keys(current_payload.get("signal_policies", {}) or {})})),
                "retained_side_policy_count": int(len(_retained_side_keys(current_payload.get("signal_policies", {}) or {}))),
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
        }
    )
    artifact_path.write_text(json.dumps(_json_safe(current_payload), indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"artifact={artifact_path}")
    if bool(args.write_latest):
        latest_dir = ROOT / "artifacts" / "regimeadaptive_v3"
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_path = latest_dir / "latest.json"
        shutil.copyfile(artifact_path, latest_path)
        print(f"latest={latest_path}")


if __name__ == "__main__":
    main()
