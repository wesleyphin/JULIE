import argparse
import datetime as dt
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from config import CONFIG
except Exception:
    CONFIG = {}

from de3_v3_family_schema import (
    ACTIVE_FAMILY_CONTEXT_DIMENSIONS,
    INACTIVE_FAMILY_CONTEXT_DIMENSIONS,
    FAMILY_CONTEXT_BUCKETS,
    OPTIONAL_CONTEXT_FIELDS,
    REQUIRED_CONTEXT_FIELDS,
    build_active_context_joint_key,
    build_family_key_from_candidate,
    canonical_competition_status,
    canonical_context_usage_snapshot,
    canonical_member_score,
    compact_member_definition,
    competition_status_is_eligible,
    evidence_support_tier_from_sample_count,
    empty_family_context_profiles,
    family_id_from_key,
    max_value,
    mean,
    min_value,
    normalize_context_buckets,
    parse_context_inputs_payload,
    percentile,
    safe_float,
    safe_int,
    strategy_member_id,
)


def _resolve_path(raw_path: Any) -> Path:
    out = Path(str(raw_path or "").strip())
    if not out.is_absolute():
        out = Path(__file__).resolve().parent / out
    return out


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    raw = str(value or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "t"}


def _load_v2_candidates(source_v2_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    payload = _load_json(source_v2_path)
    payload_version = "v2"
    if isinstance(payload, dict):
        payload_version = str(payload.get("version", "v2") or "v2")
        strategies = payload.get("strategies") if isinstance(payload.get("strategies"), list) else []
    elif isinstance(payload, list):
        strategies = payload
    else:
        strategies = []
    out: List[Dict[str, Any]] = []
    for row in strategies:
        if not isinstance(row, dict):
            continue
        cand = dict(row)
        cand["timeframe"] = cand.get("timeframe", cand.get("TF"))
        cand["session"] = cand.get("session", cand.get("Session"))
        cand["strategy_type"] = cand.get("strategy_type", cand.get("Type"))
        cand["thresh"] = cand.get("thresh", cand.get("Thresh"))
        cand["sl"] = cand.get("sl", cand.get("Best_SL"))
        cand["tp"] = cand.get("tp", cand.get("Best_TP"))
        if "signal" not in cand:
            stype = str(cand.get("strategy_type", "") or "").lower()
            if "long" in stype:
                cand["signal"] = "LONG"
            elif "short" in stype:
                cand["signal"] = "SHORT"
        if not cand.get("strategy_id"):
            cand["strategy_id"] = strategy_member_id(cand, default_session=str(cand.get("session") or ""))
        out.append(cand)
    return out, payload_version


def _family_member_sort_key(member: Dict[str, Any]) -> Tuple[float, str]:
    score = canonical_member_score(member)
    sid = str(member.get("strategy_id", member.get("member_id", "")) or "")
    return (float(score), sid)


def _build_family_priors(members: List[Dict[str, Any]]) -> Dict[str, float]:
    metric_rows = [m.get("metrics", {}) if isinstance(m.get("metrics"), dict) else {} for m in members]
    support = [safe_float(row.get("support_trades", 0.0), 0.0) for row in metric_rows]
    structural = [safe_float(row.get("structural_score", 0.0), 0.0) for row in metric_rows]
    avg_pnl = [safe_float(row.get("avg_pnl", 0.0), 0.0) for row in metric_rows]
    pf = [safe_float(row.get("profit_factor", 0.0), 0.0) for row in metric_rows]
    pbr = [safe_float(row.get("profitable_block_ratio", 0.0), 0.0) for row in metric_rows]
    worst_avg = [safe_float(row.get("worst_block_avg_pnl", 0.0), 0.0) for row in metric_rows]
    worst_pf = [safe_float(row.get("worst_block_pf", 0.0), 0.0) for row in metric_rows]
    dd = [safe_float(row.get("drawdown_norm", 0.0), 0.0) for row in metric_rows]
    stop_share = [safe_float(row.get("stop_like_share", 0.0), 0.0) for row in metric_rows]
    loss_share = [safe_float(row.get("loss_share", 0.0), 0.0) for row in metric_rows]
    return {
        "family_member_count": int(len(members)),
        "total_support_trades": float(sum(support)),
        "best_member_structural_score": max_value(structural, 0.0),
        "median_member_structural_score": percentile(structural, 50.0, 0.0),
        "mean_member_structural_score": mean(structural, 0.0),
        "best_member_avg_pnl": max_value(avg_pnl, 0.0),
        "median_member_avg_pnl": percentile(avg_pnl, 50.0, 0.0),
        "mean_member_avg_pnl": mean(avg_pnl, 0.0),
        "best_member_profit_factor": max_value(pf, 0.0),
        "median_member_profit_factor": percentile(pf, 50.0, 0.0),
        "mean_member_profit_factor": mean(pf, 0.0),
        "best_member_profitable_block_ratio": max_value(pbr, 0.0),
        "median_member_profitable_block_ratio": percentile(pbr, 50.0, 0.0),
        "best_member_worst_block_avg_pnl": max_value(worst_avg, 0.0),
        "median_member_worst_block_avg_pnl": percentile(worst_avg, 50.0, 0.0),
        "best_member_worst_block_pf": max_value(worst_pf, 0.0),
        "median_member_worst_block_pf": percentile(worst_pf, 50.0, 0.0),
        "min_drawdown_norm_across_members": min_value(dd, 0.0),
        "median_drawdown_norm": percentile(dd, 50.0, 0.0),
        "median_stop_like_share": percentile(stop_share, 50.0, 0.0),
        "median_loss_share": percentile(loss_share, 50.0, 0.0),
    }


def _schema_audit(decisions_df: pd.DataFrame, allow_parse_legacy_context_inputs: bool) -> Dict[str, Any]:
    columns = set(str(col) for col in decisions_df.columns)
    present_required = [field for field in REQUIRED_CONTEXT_FIELDS if field in columns]
    missing_required = [field for field in REQUIRED_CONTEXT_FIELDS if field not in columns]
    present_optional = [field for field in OPTIONAL_CONTEXT_FIELDS if field in columns]
    legacy_present = "family_context_inputs" in columns

    legacy_keys_seen: set[str] = set()
    if allow_parse_legacy_context_inputs and legacy_present and not decisions_df.empty:
        sample = decisions_df["family_context_inputs"].dropna().head(300)
        for value in sample.tolist():
            parsed = parse_context_inputs_payload(value)
            for key in parsed.keys():
                legacy_keys_seen.add(str(key))

    dim_sources = {
        "volatility_regime": ("ctx_volatility_regime" in columns) or ("volatility_regime" in legacy_keys_seen),
        "chop_trend_regime": ("ctx_chop_trend_regime" in columns) or ("chop_trend_regime" in legacy_keys_seen),
        "compression_expansion_regime": ("ctx_compression_expansion_regime" in columns)
        or ("compression_expansion_regime" in legacy_keys_seen),
        "confidence_band": ("ctx_confidence_band" in columns) or ("confidence_band" in legacy_keys_seen),
        "rvol_liquidity_state": ("ctx_rvol_liquidity_state" in columns) or ("rvol_liquidity_state" in legacy_keys_seen),
        "session_substate": ("ctx_session_substate" in columns) or ("session_substate" in legacy_keys_seen),
    }
    active_dims = tuple(str(dim) for dim in ACTIVE_FAMILY_CONTEXT_DIMENSIONS)
    inactive_dims = tuple(str(dim) for dim in INACTIVE_FAMILY_CONTEXT_DIMENSIONS)
    enriched_required = not all(bool(dim_sources.get(dim, False)) for dim in active_dims)
    return {
        "decision_columns": sorted(columns),
        "required_fields_present": present_required,
        "required_fields_missing": missing_required,
        "optional_fields_present": present_optional,
        "legacy_context_inputs_present": bool(legacy_present),
        "legacy_context_keys_seen": sorted(legacy_keys_seen),
        "dimension_sources": {k: bool(v) for k, v in dim_sources.items()},
        "active_dimensions": list(active_dims),
        "inactive_dimensions": list(inactive_dims),
        "enriched_export_required_for_full_bucketing": bool(enriched_required),
    }


def _parse_context_from_decision_row(row: pd.Series) -> Dict[str, Any]:
    context = {}
    for field in REQUIRED_CONTEXT_FIELDS + OPTIONAL_CONTEXT_FIELDS:
        value = row.get(field)
        if pd.isna(value):
            continue
        context[field.replace("ctx_", "")] = value
    legacy = parse_context_inputs_payload(row.get("family_context_inputs"))
    for key, value in legacy.items():
        if key not in context:
            context[str(key)] = value
    normalized = normalize_context_buckets(context)
    context.update(normalized)
    return context


def _make_bucket_acc() -> Dict[str, float]:
    return {
        "sample_count": 0.0,
        "wins": 0.0,
        "pnl_sum": 0.0,
        "gross_profit": 0.0,
        "gross_loss_abs": 0.0,
        "stop_hits": 0.0,
        "stop_gap_hits": 0.0,
        "mae_sum": 0.0,
        "mfe_sum": 0.0,
    }


def _accumulate_bucket(acc: Dict[str, float], outcome: Dict[str, float]) -> None:
    pnl = safe_float(outcome.get("realized_pnl", 0.0), 0.0)
    stop_hit = 1.0 if _parse_bool(outcome.get("stop_hit", False)) else 0.0
    stop_gap_hit = 1.0 if _parse_bool(outcome.get("stop_gap_hit", False)) else 0.0
    mae = safe_float(outcome.get("avg_mae", 0.0), 0.0)
    mfe = safe_float(outcome.get("avg_mfe", 0.0), 0.0)
    acc["sample_count"] += 1.0
    if pnl > 0.0:
        acc["wins"] += 1.0
        acc["gross_profit"] += pnl
    elif pnl < 0.0:
        acc["gross_loss_abs"] += abs(pnl)
    acc["pnl_sum"] += pnl
    acc["stop_hits"] += stop_hit
    acc["stop_gap_hits"] += stop_gap_hit
    acc["mae_sum"] += mae
    acc["mfe_sum"] += mfe


def _finalize_bucket(acc: Dict[str, float], min_samples: int, strong_samples: int) -> Dict[str, Any]:
    n = int(safe_int(acc.get("sample_count", 0), 0))
    if n <= 0:
        out = {
            "sample_count": 0,
            "win_rate": 0.0,
            "avg_pnl": 0.0,
            "profit_factor": 0.0,
            "stop_rate": 0.0,
            "stop_gap_rate": 0.0,
            "avg_mae": 0.0,
            "avg_mfe": 0.0,
        }
    else:
        gross_profit = safe_float(acc.get("gross_profit", 0.0), 0.0)
        gross_loss_abs = safe_float(acc.get("gross_loss_abs", 0.0), 0.0)
        if gross_loss_abs > 1e-9:
            pf = gross_profit / gross_loss_abs
        elif gross_profit > 0.0:
            pf = float("inf")
        else:
            pf = 0.0
        out = {
            "sample_count": n,
            "win_rate": float(safe_float(acc.get("wins", 0.0), 0.0) / float(max(1, n))),
            "avg_pnl": float(safe_float(acc.get("pnl_sum", 0.0), 0.0) / float(max(1, n))),
            "profit_factor": float(pf if math.isfinite(pf) else 999.0),
            "stop_rate": float(safe_float(acc.get("stop_hits", 0.0), 0.0) / float(max(1, n))),
            "stop_gap_rate": float(safe_float(acc.get("stop_gap_hits", 0.0), 0.0) / float(max(1, n))),
            "avg_mae": float(safe_float(acc.get("mae_sum", 0.0), 0.0) / float(max(1, n))),
            "avg_mfe": float(safe_float(acc.get("mfe_sum", 0.0), 0.0) / float(max(1, n))),
        }
    out["low_support"] = bool(n < int(max(1, min_samples)))
    out["strong_support"] = bool(n >= int(max(1, strong_samples)))
    return out


def _family_id_from_decision_row(row: pd.Series) -> str:
    chosen_family = str(row.get("chosen_family_id", "") or "").strip()
    if chosen_family:
        return chosen_family
    family = str(row.get("family_id", "") or "").strip()
    if family:
        return family
    key = {
        "timeframe": str(row.get("timeframe", "") or "").strip().lower(),
        "session": str(row.get("session", "") or "").strip().upper(),
        "side": str(row.get("side_considered", "") or "").strip().lower(),
        "de3_strategy_type": str(row.get("strategy_type", "") or "").strip(),
        "threshold": "TNA",
    }
    thresh = safe_float(row.get("thresh", float("nan")), float("nan"))
    if math.isfinite(thresh):
        if abs(thresh - round(thresh)) < 1e-9:
            key["threshold"] = f"T{int(round(thresh))}"
        else:
            key["threshold"] = f"T{str(thresh).rstrip('0').rstrip('.')}"
    return family_id_from_key(key)


def _trade_outcome_from_rows(rows: pd.DataFrame) -> Dict[str, Any]:
    pnl = safe_float(rows["realized_pnl"].fillna(0.0).sum(), 0.0) if "realized_pnl" in rows else 0.0
    if "realized_exit_type" in rows:
        exit_types = rows["realized_exit_type"].fillna("").astype(str).str.lower().tolist()
    else:
        exit_types = []
    stop_hit = any("stop" in txt for txt in exit_types)
    stop_gap_hit = any(("stop" in txt) and ("gap" in txt) for txt in exit_types)
    avg_mae = safe_float(rows["mae"].fillna(0.0).mean(), 0.0) if "mae" in rows else 0.0
    avg_mfe = safe_float(rows["mfe"].fillna(0.0).mean(), 0.0) if "mfe" in rows else 0.0
    return {
        "realized_pnl": float(pnl),
        "stop_hit": bool(stop_hit),
        "stop_gap_hit": bool(stop_gap_hit),
        "avg_mae": float(avg_mae),
        "avg_mfe": float(avg_mfe),
    }


def _to_bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index)
    return df[column].map(_parse_bool)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        logging.warning("DE3v3 builder CSV read failed (%s): %s", path, exc)
        return pd.DataFrame()


def _filter_v3_family_mode_rows(decisions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if decisions.empty:
        return decisions.copy(), {"total_rows": 0, "v3_family_mode_rows": 0}
    version_mask = (
        decisions["de3_version"].fillna("").astype(str).str.lower().str.startswith("v3")
        if "de3_version" in decisions.columns
        else pd.Series(True, index=decisions.index)
    )
    family_mode_mask = _to_bool_series(decisions, "family_mode", default=True)
    out = decisions[version_mask & family_mode_mask].copy()
    return out, {
        "total_rows": int(len(decisions)),
        "v3_family_mode_rows": int(len(out)),
    }


def _load_v3_chosen_decisions_with_outcomes(
    *,
    decision_csv_path: Optional[Path],
    trade_attribution_csv_path: Optional[Path],
    allow_parse_legacy_context_inputs: bool,
) -> Dict[str, Any]:
    missing_audit = {
        "status": "missing_inputs",
        "decision_csv_found": False,
        "trade_attribution_csv_found": False,
        "enriched_export_required_for_full_bucketing": True,
    }
    if decision_csv_path is None or trade_attribution_csv_path is None:
        return {
            "audit": dict(missing_audit),
            "decisions_v3": pd.DataFrame(),
            "chosen_v3": pd.DataFrame(),
            "chosen_with_outcomes": pd.DataFrame(),
            "outcomes_by_decision": {},
            "decision_source_counts": {"total_rows": 0, "v3_family_mode_rows": 0, "chosen_rows": 0, "chosen_with_outcomes_rows": 0},
        }
    if not decision_csv_path.exists() or not trade_attribution_csv_path.exists():
        missing_audit["decision_csv_found"] = bool(decision_csv_path.exists())
        missing_audit["trade_attribution_csv_found"] = bool(trade_attribution_csv_path.exists())
        return {
            "audit": dict(missing_audit),
            "decisions_v3": pd.DataFrame(),
            "chosen_v3": pd.DataFrame(),
            "chosen_with_outcomes": pd.DataFrame(),
            "outcomes_by_decision": {},
            "decision_source_counts": {"total_rows": 0, "v3_family_mode_rows": 0, "chosen_rows": 0, "chosen_with_outcomes_rows": 0},
        }

    decisions = _read_csv_or_empty(decision_csv_path)
    trade_attr = _read_csv_or_empty(trade_attribution_csv_path)
    audit = _schema_audit(decisions, allow_parse_legacy_context_inputs)

    if decisions.empty or trade_attr.empty:
        audit["status"] = "empty_inputs"
        return {
            "audit": audit,
            "decisions_v3": pd.DataFrame(),
            "chosen_v3": pd.DataFrame(),
            "chosen_with_outcomes": pd.DataFrame(),
            "outcomes_by_decision": {},
            "decision_source_counts": {"total_rows": int(len(decisions)), "v3_family_mode_rows": 0, "chosen_rows": 0, "chosen_with_outcomes_rows": 0},
        }

    decisions_v3, source_counts = _filter_v3_family_mode_rows(decisions)
    if decisions_v3.empty:
        audit["status"] = "no_v3_rows"
        return {
            "audit": audit,
            "decisions_v3": decisions_v3,
            "chosen_v3": pd.DataFrame(),
            "chosen_with_outcomes": pd.DataFrame(),
            "outcomes_by_decision": {},
            "decision_source_counts": {
                **source_counts,
                "chosen_rows": 0,
                "chosen_with_outcomes_rows": 0,
            },
        }

    chosen_mask = _to_bool_series(decisions_v3, "chosen", default=False)
    abstain_mask = _to_bool_series(decisions_v3, "abstained", default=False)
    chosen_v3 = decisions_v3[chosen_mask & (~abstain_mask)].copy()
    if chosen_v3.empty:
        audit["status"] = "no_chosen_rows"
        return {
            "audit": audit,
            "decisions_v3": decisions_v3,
            "chosen_v3": chosen_v3,
            "chosen_with_outcomes": pd.DataFrame(),
            "outcomes_by_decision": {},
            "decision_source_counts": {
                **source_counts,
                "chosen_rows": 0,
                "chosen_with_outcomes_rows": 0,
            },
        }

    if "decision_id" not in chosen_v3.columns or "decision_id" not in trade_attr.columns:
        audit["status"] = "missing_decision_id"
        return {
            "audit": audit,
            "decisions_v3": decisions_v3,
            "chosen_v3": chosen_v3,
            "chosen_with_outcomes": pd.DataFrame(),
            "outcomes_by_decision": {},
            "decision_source_counts": {
                **source_counts,
                "chosen_rows": int(len(chosen_v3)),
                "chosen_with_outcomes_rows": 0,
            },
        }

    chosen_v3["decision_id"] = chosen_v3["decision_id"].fillna("").astype(str).str.strip()
    chosen_v3 = chosen_v3[chosen_v3["decision_id"] != ""].copy()
    trade_attr = trade_attr.copy()
    trade_attr["decision_id"] = trade_attr["decision_id"].fillna("").astype(str).str.strip()
    trade_attr = trade_attr[trade_attr["decision_id"] != ""].copy()

    outcomes_by_decision: Dict[str, Dict[str, Any]] = {}
    for decision_id, group in trade_attr.groupby("decision_id", sort=False):
        did = str(decision_id or "").strip()
        if not did:
            continue
        outcomes_by_decision[did] = _trade_outcome_from_rows(group)

    chosen_with_outcomes = chosen_v3.copy()
    chosen_with_outcomes["family_id_effective"] = chosen_with_outcomes.apply(_family_id_from_decision_row, axis=1)
    chosen_with_outcomes["realized_pnl"] = chosen_with_outcomes["decision_id"].map(
        lambda did: (outcomes_by_decision.get(str(did), {}) or {}).get("realized_pnl")
    )
    chosen_with_outcomes["stop_hit"] = chosen_with_outcomes["decision_id"].map(
        lambda did: (outcomes_by_decision.get(str(did), {}) or {}).get("stop_hit")
    )
    chosen_with_outcomes["stop_gap_hit"] = chosen_with_outcomes["decision_id"].map(
        lambda did: (outcomes_by_decision.get(str(did), {}) or {}).get("stop_gap_hit")
    )
    chosen_with_outcomes["avg_mae"] = chosen_with_outcomes["decision_id"].map(
        lambda did: (outcomes_by_decision.get(str(did), {}) or {}).get("avg_mae")
    )
    chosen_with_outcomes["avg_mfe"] = chosen_with_outcomes["decision_id"].map(
        lambda did: (outcomes_by_decision.get(str(did), {}) or {}).get("avg_mfe")
    )
    chosen_with_outcomes["has_outcome"] = chosen_with_outcomes["realized_pnl"].notna()
    audit["status"] = "ok"
    return {
        "audit": audit,
        "decisions_v3": decisions_v3,
        "chosen_v3": chosen_v3,
        "chosen_with_outcomes": chosen_with_outcomes,
        "outcomes_by_decision": outcomes_by_decision,
        "decision_source_counts": {
            **source_counts,
            "chosen_rows": int(len(chosen_v3)),
            "chosen_with_outcomes_rows": int(chosen_with_outcomes["has_outcome"].sum()),
        },
    }


def _build_family_context_profiles(
    *,
    families_by_id: Dict[str, Dict[str, Any]],
    chosen_with_outcomes: pd.DataFrame,
    audit: Dict[str, Any],
    min_bucket_samples: int,
    strong_bucket_samples: int,
) -> Dict[str, Any]:
    audit_out = dict(audit or {})
    if chosen_with_outcomes is None or chosen_with_outcomes.empty:
        audit_out["status"] = audit_out.get("status", "no_chosen_rows")
        return {"audit": audit_out, "family_profiles": {}, "decisions_with_outcomes_used": 0}

    work = chosen_with_outcomes.copy()
    if "has_outcome" in work.columns:
        work = work[work["has_outcome"].map(_parse_bool)].copy()
    if work.empty:
        audit_out["status"] = "no_outcome_rows"
        return {"audit": audit_out, "family_profiles": {}, "decisions_with_outcomes_used": 0}

    family_dim_acc: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(_make_bucket_acc))
    )
    family_joint_acc: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(_make_bucket_acc))
    decisions_used = 0

    for _, row in work.iterrows():
        family_id = str(row.get("family_id_effective", "") or "").strip()
        if not family_id:
            family_id = _family_id_from_decision_row(row)
        if family_id not in families_by_id:
            continue
        outcome = {
            "realized_pnl": safe_float(row.get("realized_pnl", 0.0), 0.0),
            "stop_hit": bool(_parse_bool(row.get("stop_hit", False))),
            "stop_gap_hit": bool(_parse_bool(row.get("stop_gap_hit", False))),
            "avg_mae": safe_float(row.get("avg_mae", 0.0), 0.0),
            "avg_mfe": safe_float(row.get("avg_mfe", 0.0), 0.0),
        }
        context = _parse_context_from_decision_row(row)
        buckets = normalize_context_buckets(context)
        decisions_used += 1
        for dim in ACTIVE_FAMILY_CONTEXT_DIMENSIONS:
            bucket = str(buckets.get(dim, "") or "")
            if not bucket:
                continue
            _accumulate_bucket(family_dim_acc[family_id][dim][bucket], outcome)
        joint_key = build_active_context_joint_key(buckets)
        _accumulate_bucket(family_joint_acc[family_id][joint_key], outcome)

    family_profiles: Dict[str, Dict[str, Any]] = {}
    for family_id in families_by_id.keys():
        profiles = empty_family_context_profiles()
        for dim in ACTIVE_FAMILY_CONTEXT_DIMENSIONS:
            dim_acc = family_dim_acc.get(family_id, {}).get(dim, {})
            profiles[dim] = {}
            for bucket in FAMILY_CONTEXT_BUCKETS.get(dim, []):
                acc = dim_acc.get(bucket)
                profiles[dim][bucket] = _finalize_bucket(
                    acc if acc is not None else _make_bucket_acc(),
                    min_bucket_samples,
                    strong_bucket_samples,
                )

        for dim in INACTIVE_FAMILY_CONTEXT_DIMENSIONS:
            profiles[dim] = {}

        joint_out: Dict[str, Dict[str, Any]] = {}
        for joint_key, acc in sorted(family_joint_acc.get(family_id, {}).items()):
            joint_out[joint_key] = _finalize_bucket(acc, min_bucket_samples, strong_bucket_samples)
        profiles["joint_profiles"] = joint_out
        profiles["_meta"] = {
            "schema_version": "v1",
            "has_profile_data": bool(len(joint_out) > 0),
            "active_dimensions": list(ACTIVE_FAMILY_CONTEXT_DIMENSIONS),
            "inactive_dimensions": list(INACTIVE_FAMILY_CONTEXT_DIMENSIONS),
            "min_bucket_samples": int(max(1, min_bucket_samples)),
            "strong_bucket_samples": int(max(1, strong_bucket_samples)),
            "decisions_with_outcomes_used": int(decisions_used),
            "context_schema_audit": dict(audit_out),
        }
        family_profiles[family_id] = profiles

    audit_out["status"] = "ok"
    return {"audit": audit_out, "family_profiles": family_profiles, "decisions_with_outcomes_used": int(decisions_used)}


def _default_usable_family_cfg() -> Dict[str, Any]:
    de3_v3_cfg = CONFIG.get("DE3_V3", {}) if isinstance(CONFIG.get("DE3_V3", {}), dict) else {}
    usable_cfg = (
        de3_v3_cfg.get("usable_family_universe", {})
        if isinstance(de3_v3_cfg.get("usable_family_universe", {}), dict)
        else {}
    )
    return dict(usable_cfg)


def _default_prior_eligibility_cfg() -> Dict[str, Any]:
    de3_v3_cfg = CONFIG.get("DE3_V3", {}) if isinstance(CONFIG.get("DE3_V3", {}), dict) else {}
    prior_cfg = (
        de3_v3_cfg.get("prior_eligibility", {})
        if isinstance(de3_v3_cfg.get("prior_eligibility", {}), dict)
        else (
            de3_v3_cfg.get("family_eligibility", {})
            if isinstance(de3_v3_cfg.get("family_eligibility", {}), dict)
            else {}
        )
    )
    return dict(prior_cfg)


def _default_family_competition_cfg() -> Dict[str, Any]:
    de3_v3_cfg = CONFIG.get("DE3_V3", {}) if isinstance(CONFIG.get("DE3_V3", {}), dict) else {}
    comp_cfg = (
        de3_v3_cfg.get("family_competition", {})
        if isinstance(de3_v3_cfg.get("family_competition", {}), dict)
        else {}
    )
    return dict(comp_cfg)


def _evaluate_prior_eligibility_from_priors(priors: Dict[str, Any], prior_cfg: Dict[str, Any]) -> Dict[str, Any]:
    p = priors if isinstance(priors, dict) else {}
    if not p:
        return {"prior_eligible": False, "prior_reason": "missing_family_priors", "catastrophic_prior": True}
    best_pf = safe_float(p.get("best_member_profit_factor", 0.0), 0.0)
    best_pbr = safe_float(p.get("best_member_profitable_block_ratio", 0.0), 0.0)
    best_worst_pf = safe_float(p.get("best_member_worst_block_pf", 0.0), 0.0)
    best_worst_avg = safe_float(p.get("best_member_worst_block_avg_pnl", 0.0), 0.0)
    support = safe_float(p.get("total_support_trades", 0.0), 0.0)
    median_dd = safe_float(p.get("median_drawdown_norm", 0.0), 0.0)
    median_loss = safe_float(p.get("median_loss_share", 0.0), 0.0)
    median_struct = safe_float(p.get("median_member_structural_score", 0.0), 0.0)

    cat_min_best_pf = safe_float(prior_cfg.get("catastrophic_min_best_member_profit_factor", 0.50), 0.50)
    cat_min_best_worst_pf = safe_float(prior_cfg.get("catastrophic_min_best_member_worst_block_pf", 0.25), 0.25)
    cat_max_dd = safe_float(prior_cfg.get("catastrophic_max_median_drawdown_norm", 2.30), 2.30)
    cat_max_loss = safe_float(prior_cfg.get("catastrophic_max_median_loss_share", 0.95), 0.95)
    if (
        best_pf < cat_min_best_pf
        or best_worst_pf < cat_min_best_worst_pf
        or median_dd > cat_max_dd
        or median_loss > cat_max_loss
    ):
        return {"prior_eligible": False, "prior_reason": "catastrophic_prior_failure", "catastrophic_prior": True}

    if not bool(prior_cfg.get("enabled", True)):
        return {"prior_eligible": True, "prior_reason": "", "catastrophic_prior": False}

    reasons: List[str] = []
    if support < safe_float(prior_cfg.get("min_total_support_trades", 25), 25):
        reasons.append("support_below_min")
    if best_pf < safe_float(prior_cfg.get("min_best_member_profit_factor", 0.90), 0.90):
        reasons.append("best_pf_below_min")
    if best_pbr < safe_float(prior_cfg.get("min_best_member_profitable_block_ratio", 0.45), 0.45):
        reasons.append("best_pbr_below_min")
    if best_worst_pf < safe_float(prior_cfg.get("min_best_member_worst_block_pf", 0.70), 0.70):
        reasons.append("best_worst_pf_below_min")
    if best_worst_avg < safe_float(prior_cfg.get("min_best_member_worst_block_avg_pnl", -0.85), -0.85):
        reasons.append("worst_block_avg_too_low")
    if median_dd > safe_float(prior_cfg.get("max_median_drawdown_norm", 1.60), 1.60):
        reasons.append("median_drawdown_too_high")
    if median_loss > safe_float(prior_cfg.get("max_median_loss_share", 0.85), 0.85):
        reasons.append("median_loss_share_too_high")
    if median_struct < safe_float(prior_cfg.get("min_median_member_structural_score", -2.0), -2.0):
        reasons.append("median_structural_too_low")
    return {
        "prior_eligible": bool(len(reasons) == 0),
        "prior_reason": ",".join(reasons),
        "catastrophic_prior": False,
    }


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _classify_family_evidence_state(
    metrics: Dict[str, float],
    cfg: Dict[str, Any],
    *,
    prior_eligible: bool,
    catastrophic_prior: bool,
    prior_reason: str,
) -> Dict[str, Any]:
    chosen_count = int(max(0, safe_float(metrics.get("chosen_count", 0.0), 0.0)))
    executed_count = int(max(0, safe_float(metrics.get("executed_trade_count", 0.0), 0.0)))
    realized_pf = safe_float(metrics.get("realized_profit_factor", 0.0), 0.0)
    realized_avg_pnl = safe_float(metrics.get("realized_avg_pnl", 0.0), 0.0)
    stop_rate = safe_float(metrics.get("realized_stop_rate", 0.0), 0.0)
    stop_gap_rate = safe_float(metrics.get("realized_stop_gap_rate", 0.0), 0.0)
    fallback_rate = safe_float(metrics.get("fallback_to_prior_rate", 1.0), 1.0)

    support_cfg = cfg.get("evidence_support", {}) if isinstance(cfg.get("evidence_support"), dict) else {}
    min_mid_samples = int(
        max(
            1,
            safe_float(
                support_cfg.get("min_mid_samples", cfg.get("min_trades_for_state", 8)),
                8,
            ),
        )
    )
    strong_samples = int(
        max(
            min_mid_samples,
            safe_float(
                support_cfg.get("strong_samples", cfg.get("min_trades_for_active", 20)),
                20,
            ),
        )
    )
    evidence_support_tier = evidence_support_tier_from_sample_count(
        executed_count,
        min_samples=min_mid_samples,
        strong_samples=strong_samples,
    )
    if executed_count <= 0:
        evidence_reason = "no_realized_history"
    elif evidence_support_tier == "low":
        evidence_reason = "insufficient_v3_support"
    elif evidence_support_tier == "mid":
        evidence_reason = "moderate_v3_support"
    else:
        evidence_reason = "strong_v3_support"

    suppress_cfg = cfg.get("suppression", {}) if isinstance(cfg.get("suppression"), dict) else {}
    suppression_min_trades = int(
        max(
            min_mid_samples,
            safe_float(suppress_cfg.get("min_trades", cfg.get("suppression_min_trades", 20)), 20),
        )
    )
    suppress_if_pf_below = safe_float(
        suppress_cfg.get("if_profit_factor_below", cfg.get("suppress_if_pf_below", 0.70)),
        0.70,
    )
    suppress_if_stop_rate_above = safe_float(
        suppress_cfg.get("if_stop_rate_above", cfg.get("suppress_if_stop_rate_above", 0.78)),
        0.78,
    )
    suppress_if_stop_gap_rate_above = safe_float(
        suppress_cfg.get("if_stop_gap_rate_above", cfg.get("suppress_if_stop_gap_rate_above", 0.35)),
        0.35,
    )
    suppress_if_avg_pnl_below = safe_float(
        suppress_cfg.get("if_avg_pnl_below", cfg.get("suppress_if_avg_pnl_below", -0.80)),
        -0.80,
    )

    fallback_cfg = cfg.get("fallback_only", {}) if isinstance(cfg.get("fallback_only"), dict) else {}
    fallback_min_trades = int(
        max(
            min_mid_samples,
            safe_float(fallback_cfg.get("min_trades", min_mid_samples), min_mid_samples),
        )
    )
    fallback_pf_below = safe_float(
        fallback_cfg.get("if_profit_factor_below", max(0.0, safe_float(cfg.get("min_pf_for_fallback", 0.90), 0.90))),
        0.90,
    )
    fallback_avg_pnl_below = safe_float(
        fallback_cfg.get("if_avg_pnl_below", safe_float(cfg.get("min_avg_pnl_for_active", -0.10), -0.10)),
        -0.10,
    )
    fallback_stop_rate_above = safe_float(fallback_cfg.get("if_stop_rate_above", 0.70), 0.70)
    fallback_stop_gap_rate_above = safe_float(fallback_cfg.get("if_stop_gap_rate_above", 0.30), 0.30)
    fallback_to_prior_rate_above = safe_float(fallback_cfg.get("if_fallback_to_prior_rate_above", 0.90), 0.90)

    competition_status = "competitive"
    suppression_reason = ""
    competition_reason = ""
    if (not prior_eligible) or catastrophic_prior:
        competition_status = "suppressed"
        suppression_reason = "catastrophic_prior_failure" if catastrophic_prior else (
            f"prior_ineligible:{prior_reason}" if str(prior_reason or "").strip() else "prior_ineligible"
        )
        competition_reason = suppression_reason
    else:
        catastrophic_realized = bool(
            executed_count >= suppression_min_trades
            and (
                realized_pf < suppress_if_pf_below
                or stop_rate > suppress_if_stop_rate_above
                or stop_gap_rate > suppress_if_stop_gap_rate_above
                or realized_avg_pnl < suppress_if_avg_pnl_below
            )
        )
        weak_realized = bool(
            executed_count >= fallback_min_trades
            and (
                realized_pf < fallback_pf_below
                or realized_avg_pnl < fallback_avg_pnl_below
                or stop_rate > fallback_stop_rate_above
                or stop_gap_rate > fallback_stop_gap_rate_above
                or fallback_rate > fallback_to_prior_rate_above
            )
        )
        if catastrophic_realized:
            competition_status = "suppressed"
            suppression_reason = "catastrophic_realized_behavior"
            competition_reason = suppression_reason
        elif weak_realized:
            competition_status = "fallback_only"
            competition_reason = "weak_realized_behavior"
        else:
            competition_status = "competitive"
            competition_reason = "prior_eligible_competitive"

    adj_cfg = cfg.get("evidence_adjustment", {}) if isinstance(cfg.get("evidence_adjustment"), dict) else {}
    base_map = adj_cfg.get("support_tier_base", {}) if isinstance(adj_cfg.get("support_tier_base"), dict) else {}
    quality_map = adj_cfg.get("support_tier_quality_scale", {}) if isinstance(adj_cfg.get("support_tier_quality_scale"), dict) else {}
    base_by_tier = {
        "none": safe_float(base_map.get("none", 0.0), 0.0),
        "low": safe_float(base_map.get("low", -0.005), -0.005),
        "mid": safe_float(base_map.get("mid", 0.03), 0.03),
        "strong": safe_float(base_map.get("strong", 0.08), 0.08),
    }
    quality_scale_by_tier = {
        "none": safe_float(quality_map.get("none", 0.0), 0.0),
        "low": safe_float(quality_map.get("low", 0.01), 0.01),
        "mid": safe_float(quality_map.get("mid", 0.08), 0.08),
        "strong": safe_float(quality_map.get("strong", 0.15), 0.15),
    }
    fallback_only_penalty = safe_float(adj_cfg.get("fallback_only_penalty", -0.05), -0.05)
    suppressed_adjustment = safe_float(adj_cfg.get("suppressed_adjustment", -0.20), -0.20)
    low_tier_min_adjustment = safe_float(adj_cfg.get("low_tier_min_adjustment", -0.02), -0.02)
    max_abs_adjustment = abs(safe_float(adj_cfg.get("max_abs_adjustment", 0.20), 0.20))
    quality_confidence_trades = int(
        max(
            1,
            safe_float(
                adj_cfg.get("quality_confidence_trades", cfg.get("quality_confidence_trades", 40)),
                40,
            ),
        )
    )

    pf_term = _clip((realized_pf - 1.0) / 0.35, -2.0, 2.0)
    avg_pnl_term = _clip(realized_avg_pnl / 1.25, -2.0, 2.0)
    stop_penalty = _clip((stop_rate - 0.40) / 0.30, -2.0, 2.0)
    fallback_penalty = _clip((fallback_rate - 0.50) / 0.30, -2.0, 2.0)
    quality = (0.55 * pf_term) + (0.30 * avg_pnl_term) - (0.10 * stop_penalty) - (0.05 * fallback_penalty)
    confidence = _clip(
        math.log1p(float(max(0, executed_count))) / math.log1p(float(max(1, quality_confidence_trades))),
        0.0,
        1.0,
    )
    tier_base = base_by_tier.get(evidence_support_tier, 0.0)
    tier_quality_scale = quality_scale_by_tier.get(evidence_support_tier, 0.0)
    usability_adjustment = float(tier_base + (confidence * tier_quality_scale * quality))
    if competition_status == "fallback_only":
        usability_adjustment += float(fallback_only_penalty)
    if competition_status == "suppressed":
        usability_adjustment = min(float(usability_adjustment), float(suppressed_adjustment))
    if competition_status != "suppressed" and evidence_support_tier in {"none", "low"}:
        usability_adjustment = max(float(usability_adjustment), float(low_tier_min_adjustment))
    usability_adjustment = _clip(float(usability_adjustment), -max_abs_adjustment, max_abs_adjustment)

    return {
        "evidence_support_tier": str(evidence_support_tier),
        "evidence_reason": str(evidence_reason),
        "competition_status": canonical_competition_status(competition_status),
        "competition_reason": str(competition_reason),
        "suppression_reason": str(suppression_reason),
        "usability_adjustment": float(usability_adjustment),
        "chosen_count": int(chosen_count),
        "executed_count": int(executed_count),
    }


def _build_family_runtime_state(
    *,
    families_by_id: Dict[str, Dict[str, Any]],
    chosen_with_outcomes: pd.DataFrame,
    usable_cfg: Dict[str, Any],
    prior_cfg: Dict[str, Any],
    competition_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    family_states: Dict[str, Dict[str, Any]] = {}
    exclude_only_suppressed = bool(usable_cfg.get("exclude_only_suppressed_families", True))
    bootstrap_min = int(max(1, safe_float(competition_cfg.get("bootstrap_min_competing_families", 3), 3)))
    support_cfg = usable_cfg.get("evidence_support", {}) if isinstance(usable_cfg.get("evidence_support"), dict) else {}
    evidence_mid_samples = int(max(1, safe_float(support_cfg.get("min_mid_samples", usable_cfg.get("min_trades_for_state", 8)), 8)))
    evidence_strong_samples = int(max(evidence_mid_samples, safe_float(support_cfg.get("strong_samples", usable_cfg.get("min_trades_for_active", 20)), 20)))

    if chosen_with_outcomes is None or chosen_with_outcomes.empty:
        chosen_with_outcomes = pd.DataFrame()

    work = chosen_with_outcomes.copy()
    if not work.empty:
        work["family_id_effective"] = work["family_id_effective"].fillna("").astype(str)
        work = work[work["family_id_effective"].str.strip() != ""].copy()
        if "has_outcome" in work.columns:
            work["has_outcome_bool"] = work["has_outcome"].map(_parse_bool)
        else:
            work["has_outcome_bool"] = work["realized_pnl"].notna()
    grouped_by_family = (
        {str(family_id): grp.copy() for family_id, grp in work.groupby("family_id_effective", sort=False)}
        if not work.empty
        else {}
    )
    empty_rows = work.iloc[0:0].copy() if not work.empty else pd.DataFrame()

    evidence_support_tier_counts = {"none": 0, "low": 0, "mid": 0, "strong": 0}
    competition_status_counts = {"competitive": 0, "competitive_bootstrap": 0, "fallback_only": 0, "suppressed": 0}
    prior_eligible_count = 0
    prior_eligible_low_support_count = 0
    suppressed_count = 0
    active_competition_count = 0
    bootstrap_included_count = 0

    for family_id, family_row in families_by_id.items():
        fam_key = str(family_id)
        fam_rows = grouped_by_family.get(fam_key, empty_rows)
        chosen_count = int(len(fam_rows))
        with_outcome = fam_rows[fam_rows["has_outcome_bool"]].copy() if (not fam_rows.empty and "has_outcome_bool" in fam_rows.columns) else fam_rows.copy()
        executed_count = int(len(with_outcome))

        realized_net_pnl = float(with_outcome["realized_pnl"].fillna(0.0).sum()) if ("realized_pnl" in with_outcome.columns and executed_count > 0) else 0.0
        realized_avg_pnl = float(realized_net_pnl / float(max(1, executed_count))) if executed_count > 0 else 0.0
        if executed_count > 0 and "realized_pnl" in with_outcome.columns:
            pnl_vals = with_outcome["realized_pnl"].fillna(0.0)
            gross_profit = float(pnl_vals[pnl_vals > 0].sum())
            gross_loss_abs = float(abs(pnl_vals[pnl_vals < 0].sum()))
            realized_pf = (gross_profit / gross_loss_abs) if gross_loss_abs > 1e-9 else (999.0 if gross_profit > 0 else 0.0)
            realized_stop_rate = float(with_outcome["stop_hit"].fillna(False).map(_parse_bool).mean()) if "stop_hit" in with_outcome.columns else 0.0
            realized_stop_gap_rate = float(with_outcome["stop_gap_hit"].fillna(False).map(_parse_bool).mean()) if "stop_gap_hit" in with_outcome.columns else 0.0
            realized_avg_mae = float(with_outcome["avg_mae"].fillna(0.0).mean()) if "avg_mae" in with_outcome.columns else 0.0
            realized_avg_mfe = float(with_outcome["avg_mfe"].fillna(0.0).mean()) if "avg_mfe" in with_outcome.columns else 0.0
        else:
            realized_pf = 0.0
            realized_stop_rate = 0.0
            realized_stop_gap_rate = 0.0
            realized_avg_mae = 0.0
            realized_avg_mfe = 0.0

        strong_count = 0
        mid_count = 0
        low_count = 0
        trusted_count = 0
        fallback_count = 0
        mode_counts = {"full": 0, "conservative": 0, "frozen": 0, "none": 0}
        override_count = 0
        for _, decision_row in fam_rows.iterrows():
            snap = canonical_context_usage_snapshot(decision_row.to_dict())
            tier = str(snap.get("support_tier", "low"))
            if tier == "strong":
                strong_count += 1
            elif tier == "mid":
                mid_count += 1
            else:
                low_count += 1
            if bool(snap.get("trusted_context_used", False)):
                trusted_count += 1
            if bool(snap.get("fallback_to_priors", True)):
                fallback_count += 1
            mode = str(snap.get("local_bracket_mode", "none"))
            mode_counts[mode if mode in mode_counts else "none"] = int(mode_counts.get(mode if mode in mode_counts else "none", 0) + 1)
            if bool(snap.get("local_bracket_override_applied", False)):
                override_count += 1

        context_supported_rate = float((strong_count + mid_count) / float(max(1, chosen_count)))
        fallback_rate = float(fallback_count / float(max(1, chosen_count)))
        local_override_rate = float(override_count / float(max(1, chosen_count)))

        metrics = {
            "chosen_count": int(chosen_count),
            "executed_trade_count": int(executed_count),
            "realized_net_pnl": float(realized_net_pnl),
            "realized_avg_pnl": float(realized_avg_pnl),
            "realized_profit_factor": float(realized_pf),
            "realized_stop_rate": float(realized_stop_rate),
            "realized_stop_gap_rate": float(realized_stop_gap_rate),
            "realized_avg_mae": float(realized_avg_mae),
            "realized_avg_mfe": float(realized_avg_mfe),
            "fallback_to_prior_rate": float(fallback_rate),
            "trusted_context_used_rate": float(trusted_count / float(max(1, chosen_count))),
            "context_supported_decision_rate": float(context_supported_rate),
            "local_bracket_override_rate": float(local_override_rate),
            "strong_context_support_count": int(strong_count),
            "mid_context_support_count": int(mid_count),
            "low_context_support_count": int(low_count),
            "local_bracket_mode_counts": dict(mode_counts),
        }

        prior_eval = _evaluate_prior_eligibility_from_priors(
            family_row.get("family_priors", {}) if isinstance(family_row, dict) else {},
            prior_cfg,
        )
        prior_eligible = bool(prior_eval.get("prior_eligible", False))
        catastrophic_prior = bool(prior_eval.get("catastrophic_prior", False))
        state_eval = _classify_family_evidence_state(
            metrics,
            usable_cfg,
            prior_eligible=prior_eligible,
            catastrophic_prior=catastrophic_prior,
            prior_reason=str(prior_eval.get("prior_reason", "") or ""),
        )
        evidence_support_tier = str(state_eval.get("evidence_support_tier", "none") or "none")
        if evidence_support_tier not in evidence_support_tier_counts:
            evidence_support_tier = "none"
        competition_status = canonical_competition_status(state_eval.get("competition_status", "competitive"))
        usability_adjustment = safe_float(state_eval.get("usability_adjustment", 0.0), 0.0)
        suppression_reason = str(state_eval.get("suppression_reason", "") or "")
        competition_reason = str(state_eval.get("competition_reason", "") or "")
        bootstrap_included = False
        if exclude_only_suppressed:
            competition_eligible = bool(prior_eligible and competition_status_is_eligible(competition_status))
        else:
            competition_eligible = bool(prior_eligible and competition_status in {"competitive", "competitive_bootstrap"})
        if not competition_eligible and not competition_reason:
            if competition_status == "suppressed":
                competition_reason = suppression_reason or "suppressed"
            elif not prior_eligible:
                reason_txt = str(prior_eval.get("prior_reason", "") or "").strip()
                competition_reason = f"prior_ineligible:{reason_txt}" if reason_txt else "prior_ineligible"

        evidence_support_tier_counts[evidence_support_tier] = int(evidence_support_tier_counts.get(evidence_support_tier, 0) + 1)
        competition_status_counts[competition_status] = int(competition_status_counts.get(competition_status, 0) + 1)
        if prior_eligible:
            prior_eligible_count += 1
        if prior_eligible and evidence_support_tier in {"none", "low"}:
            prior_eligible_low_support_count += 1
        if competition_status == "suppressed" or catastrophic_prior:
            suppressed_count += 1
        if competition_eligible:
            active_competition_count += 1
        if bootstrap_included:
            bootstrap_included_count += 1

        family_states[fam_key] = {
            "prior_eligible": bool(prior_eligible),
            "prior_eligibility_reason": str(prior_eval.get("prior_reason", "") or ""),
            "catastrophic_prior": bool(catastrophic_prior),
            "evidence_support_tier": str(evidence_support_tier),
            "evidence_reason": str(state_eval.get("evidence_reason", "") or ""),
            "competition_status": str(competition_status),
            "usability_adjustment": float(usability_adjustment),
            "suppression_reason": str(suppression_reason),
            "bootstrap_competition_included": bool(bootstrap_included),
            # Backward-compatible aliases.
            "usability_state": (
                "suppressed"
                if competition_status == "suppressed"
                else ("fallback_only" if competition_status == "fallback_only" else ("low_support" if evidence_support_tier in {"none", "low"} else "active"))
            ),
            "usability_reason": str(competition_reason or state_eval.get("evidence_reason", "") or ""),
            "usability_component_hint": float(usability_adjustment),
            "competition_eligible": bool(competition_eligible),
            "competition_eligibility_reason": str(competition_reason),
            "source": "v3_decision_exports",
            "metrics": metrics,
        }

    inherited_count = int(len(families_by_id))
    runtime_status = "ok" if not work.empty else "no_v3_decision_history"
    chosen_counts = [
        int(((state.get("metrics", {}) or {}).get("chosen_count", 0) or 0))
        for state in family_states.values()
    ]
    total_chosen = int(sum(chosen_counts))
    chosen_nonzero = sorted([cnt for cnt in chosen_counts if cnt > 0], reverse=True)
    top_family_share = float(chosen_nonzero[0] / float(total_chosen)) if (total_chosen > 0 and chosen_nonzero) else 0.0
    top2_share = float(sum(chosen_nonzero[:2]) / float(total_chosen)) if (total_chosen > 0 and chosen_nonzero) else 0.0
    prior_eligible_but_never_chosen = int(
        sum(
            1
            for state in family_states.values()
            if bool((state or {}).get("prior_eligible", False))
            and int((((state or {}).get("metrics", {}) or {}).get("chosen_count", 0) or 0)) <= 0
        )
    )
    prior_eligible_competitive_zero_support = int(
        sum(
            1
            for state in family_states.values()
            if bool((state or {}).get("prior_eligible", False))
            and str((state or {}).get("competition_status", "suppressed")).lower()
            in {"competitive", "competitive_bootstrap", "fallback_only"}
            and int((((state or {}).get("metrics", {}) or {}).get("executed_trade_count", 0) or 0)) <= 0
        )
    )
    return {
        "family_runtime_states": family_states,
        "runtime_state_meta": {
            "status": runtime_status,
            "chosen_rows": int(len(work)),
            "chosen_with_outcomes_rows": int(work["has_outcome_bool"].sum()) if "has_outcome_bool" in work.columns else 0,
            "bootstrap_min_competing_families": int(bootstrap_min),
            "exclude_only_suppressed_families": bool(exclude_only_suppressed),
            "evidence_mid_samples": int(evidence_mid_samples),
            "evidence_strong_samples": int(evidence_strong_samples),
        },
        "usable_universe_summary": {
            "inherited_family_universe_size": inherited_count,
            "prior_eligible_family_count": int(prior_eligible_count),
            "v3_native_usable_family_universe_size": int(
                sum(
                    1
                    for x in family_states.values()
                    if bool((x or {}).get("prior_eligible", False))
                    and competition_status_is_eligible((x or {}).get("competition_status", "suppressed"))
                )
            ),
            "active_runtime_competition_family_count": int(active_competition_count),
            "prior_eligible_low_support_family_count": int(prior_eligible_low_support_count),
            "suppressed_family_count": int(suppressed_count),
            "evidence_support_tier_counts": dict(evidence_support_tier_counts),
            "competition_status_counts": dict(competition_status_counts),
            "bootstrap_included_family_count": int(bootstrap_included_count),
            "bootstrap_min_competing_families": int(bootstrap_min),
            "family_competition_health": {
                "chosen_family_unique_count": int(len(chosen_nonzero)),
                "top_family_chosen_share": float(top_family_share),
                "top_2_family_share": float(top2_share),
                "prior_eligible_but_never_chosen_count": int(prior_eligible_but_never_chosen),
                "prior_eligible_and_competitive_but_zero_realized_support_count": int(
                    prior_eligible_competitive_zero_support
                ),
            },
            # Backward-compatible alias.
            "usability_state_counts": {
                "active": int(
                    sum(
                        1
                        for x in family_states.values()
                        if str((x or {}).get("usability_state", "")) == "active"
                    )
                ),
                "fallback_only": int(
                    sum(
                        1
                        for x in family_states.values()
                        if str((x or {}).get("usability_state", "")) == "fallback_only"
                    )
                ),
                "low_support": int(
                    sum(
                        1
                        for x in family_states.values()
                        if str((x or {}).get("usability_state", "")) == "low_support"
                    )
                ),
                "suppressed": int(
                    sum(
                        1
                        for x in family_states.values()
                        if str((x or {}).get("usability_state", "")) == "suppressed"
                    )
                ),
            },
        },
    }


def _default_context_profile_cfg() -> Dict[str, Any]:
    de3_v3_cfg = CONFIG.get("DE3_V3", {}) if isinstance(CONFIG.get("DE3_V3", {}), dict) else {}
    cp_cfg = de3_v3_cfg.get("context_profiles", {}) if isinstance(de3_v3_cfg.get("context_profiles", {}), dict) else {}
    return dict(cp_cfg)


def build_de3_v3_family_inventory(
    *,
    source_v2_path: Path,
    decision_csv_path: Optional[Path] = None,
    trade_attribution_csv_path: Optional[Path] = None,
    min_bucket_samples: Optional[int] = None,
    strong_bucket_samples: Optional[int] = None,
    context_profiles_enabled: Optional[bool] = None,
    allow_parse_legacy_context_inputs: Optional[bool] = None,
) -> Dict[str, Any]:
    source_v2_path = _resolve_path(source_v2_path)
    candidates, source_v2_version = _load_v2_candidates(source_v2_path)
    grouped: Dict[str, Dict[str, Any]] = {}

    for cand in candidates:
        family_key = build_family_key_from_candidate(cand, default_session=str(cand.get("session") or ""))
        family_id = family_id_from_key(family_key)
        member = compact_member_definition(cand, default_session=str(cand.get("session") or ""))
        block = grouped.setdefault(
            family_id,
            {
                "family_id": family_id,
                "family_key": family_key,
                "members": [],
            },
        )
        block["members"].append(member)

    families: List[Dict[str, Any]] = []
    for family_id, block in grouped.items():
        members = list(block.get("members", []))
        members.sort(key=_family_member_sort_key, reverse=True)
        canonical = members[0] if members else {}
        priors = _build_family_priors(members)
        family_row = {
            "family_id": family_id,
            "family_key": dict(block.get("family_key") or {}),
            "member_count": int(len(members)),
            "member_ids": [str(m.get("member_id", "")) for m in members],
            "members": members,
            "canonical_representative_member": canonical,
            "family_priors": priors,
            "family_context_profiles": empty_family_context_profiles(),
        }
        families.append(family_row)

    families.sort(key=lambda row: str(row.get("family_id", "")))
    families_by_id = {str(row.get("family_id", "")): row for row in families}

    cp_defaults = _default_context_profile_cfg()
    usable_defaults = _default_usable_family_cfg()
    if min_bucket_samples is None:
        min_bucket_samples = safe_int(cp_defaults.get("min_bucket_samples", 12), 12)
    if strong_bucket_samples is None:
        strong_bucket_samples = safe_int(cp_defaults.get("strong_bucket_samples", 40), 40)
    if context_profiles_enabled is None:
        context_profiles_enabled = bool(cp_defaults.get("enabled", True))
    if allow_parse_legacy_context_inputs is None:
        allow_parse_legacy_context_inputs = bool(cp_defaults.get("allow_parse_legacy_context_inputs", True))

    decision_csv_path = decision_csv_path or _resolve_path(cp_defaults.get("decision_csv_path", "reports/de3_decisions.csv"))
    trade_attribution_csv_path = trade_attribution_csv_path or _resolve_path(
        cp_defaults.get("trade_attribution_csv_path", "reports/de3_decisions_trade_attribution.csv")
    )
    decision_summary_json_path = _resolve_path(
        cp_defaults.get("decision_summary_json_path", "reports/de3_decisions_summary.json")
    )
    threshold_sensitivity_json_path = _resolve_path(
        cp_defaults.get("threshold_sensitivity_json_path", "reports/de3_threshold_sensitivity.json")
    )
    bucket_attribution_csv_path = _resolve_path(
        cp_defaults.get("bucket_attribution_csv_path", "reports/de3_bucket_attribution.csv")
    )
    runtime_state_json_path = _resolve_path(
        usable_defaults.get("runtime_state_json_path", "reports/de3_family_runtime_state.json")
    )

    cp_build_meta: Dict[str, Any] = {
        "enabled": bool(context_profiles_enabled),
        "active_dimensions": list(ACTIVE_FAMILY_CONTEXT_DIMENSIONS),
        "inactive_dimensions": list(INACTIVE_FAMILY_CONTEXT_DIMENSIONS),
        "decision_csv_path": str(decision_csv_path),
        "trade_attribution_csv_path": str(trade_attribution_csv_path),
        "decision_summary_json_path": str(decision_summary_json_path),
        "threshold_sensitivity_json_path": str(threshold_sensitivity_json_path),
        "bucket_attribution_csv_path": str(bucket_attribution_csv_path),
        "decision_summary_json_found": bool(decision_summary_json_path.exists()),
        "threshold_sensitivity_json_found": bool(threshold_sensitivity_json_path.exists()),
        "bucket_attribution_csv_found": bool(bucket_attribution_csv_path.exists()),
        "min_bucket_samples": int(max(1, min_bucket_samples)),
        "strong_bucket_samples": int(max(1, strong_bucket_samples)),
    }
    if decision_summary_json_path.exists():
        try:
            decision_summary_payload = _load_json(decision_summary_json_path)
            if isinstance(decision_summary_payload, dict):
                cp_build_meta["decision_summary_meta"] = {
                    "decision_rows": decision_summary_payload.get("decision_rows"),
                    "trade_attribution_rows": decision_summary_payload.get("trade_attribution_rows"),
                    "used_enriched_decision_export": decision_summary_payload.get("used_enriched_decision_export"),
                }
        except Exception:
            pass
    if threshold_sensitivity_json_path.exists():
        try:
            threshold_payload = _load_json(threshold_sensitivity_json_path)
            if isinstance(threshold_payload, dict):
                cp_build_meta["threshold_sensitivity_meta"] = {
                    "executed_trades": threshold_payload.get("executed_trades"),
                }
        except Exception:
            pass
    if bucket_attribution_csv_path.exists():
        try:
            cp_build_meta["bucket_attribution_rows"] = int(len(pd.read_csv(bucket_attribution_csv_path)))
        except Exception:
            pass

    decision_source = _load_v3_chosen_decisions_with_outcomes(
        decision_csv_path=decision_csv_path,
        trade_attribution_csv_path=trade_attribution_csv_path,
        allow_parse_legacy_context_inputs=bool(allow_parse_legacy_context_inputs),
    )
    decision_source_counts = (
        decision_source.get("decision_source_counts", {})
        if isinstance(decision_source.get("decision_source_counts"), dict)
        else {}
    )
    source_audit = decision_source.get("audit", {}) if isinstance(decision_source.get("audit"), dict) else {}
    chosen_with_outcomes = (
        decision_source.get("chosen_with_outcomes")
        if isinstance(decision_source.get("chosen_with_outcomes"), pd.DataFrame)
        else pd.DataFrame()
    )
    cp_build_meta["source_counts"] = dict(decision_source_counts)
    cp_build_meta["source_audit"] = dict(source_audit)
    cp_build_meta["decision_source"] = "v3_family_mode_decisions_only"

    if context_profiles_enabled:
        cp_result = _build_family_context_profiles(
            families_by_id=families_by_id,
            chosen_with_outcomes=chosen_with_outcomes,
            audit=source_audit,
            min_bucket_samples=int(max(1, min_bucket_samples)),
            strong_bucket_samples=int(max(1, strong_bucket_samples)),
        )
        cp_build_meta["audit"] = cp_result.get("audit", {})
        cp_build_meta["decisions_with_outcomes_used"] = int(cp_result.get("decisions_with_outcomes_used", 0) or 0)
        family_profiles = cp_result.get("family_profiles", {}) if isinstance(cp_result.get("family_profiles"), dict) else {}
        for family_id, family in families_by_id.items():
            profiles = family_profiles.get(family_id)
            if isinstance(profiles, dict):
                family["family_context_profiles"] = profiles
                meta = family["family_context_profiles"].get("_meta") if isinstance(family["family_context_profiles"].get("_meta"), dict) else {}
                family["family_context_profiles"]["_meta"] = {
                    **meta,
                    "min_bucket_samples": int(max(1, min_bucket_samples)),
                    "strong_bucket_samples": int(max(1, strong_bucket_samples)),
                }
    else:
        cp_build_meta["audit"] = {"status": "disabled", **dict(source_audit)}
        cp_build_meta["decisions_with_outcomes_used"] = 0

    prior_defaults = _default_prior_eligibility_cfg()
    competition_defaults = _default_family_competition_cfg()
    runtime_state_result = _build_family_runtime_state(
        families_by_id=families_by_id,
        chosen_with_outcomes=chosen_with_outcomes,
        usable_cfg=usable_defaults,
        prior_cfg=prior_defaults,
        competition_cfg=competition_defaults,
    )
    family_runtime_states = (
        runtime_state_result.get("family_runtime_states", {})
        if isinstance(runtime_state_result.get("family_runtime_states"), dict)
        else {}
    )
    for family_id, family in families_by_id.items():
        state = family_runtime_states.get(family_id)
        if isinstance(state, dict):
            family["family_runtime_state"] = state
        else:
            family["family_runtime_state"] = {
                "prior_eligible": False,
                "prior_eligibility_reason": "missing_runtime_state_row",
                "catastrophic_prior": False,
                "evidence_support_tier": "none",
                "evidence_reason": "missing_runtime_state_row",
                "competition_status": "suppressed",
                "usability_adjustment": safe_float(
                    ((usable_defaults.get("evidence_adjustment", {}) or {}).get("suppressed_adjustment", -0.20)),
                    -0.20,
                ),
                "suppression_reason": "missing_runtime_state_row",
                "bootstrap_competition_included": False,
                # Backward-compatible aliases.
                "usability_state": "suppressed",
                "usability_reason": "missing_runtime_state_row",
                "usability_component_hint": safe_float(
                    ((usable_defaults.get("evidence_adjustment", {}) or {}).get("suppressed_adjustment", -0.20)),
                    -0.20,
                ),
                "competition_eligible": False,
                "competition_eligibility_reason": "missing_runtime_state_row",
                "source": "v3_decision_exports",
                "metrics": {
                    "chosen_count": 0,
                    "executed_trade_count": 0,
                    "realized_net_pnl": 0.0,
                    "realized_avg_pnl": 0.0,
                    "realized_profit_factor": 0.0,
                    "realized_stop_rate": 0.0,
                    "realized_stop_gap_rate": 0.0,
                    "realized_avg_mae": 0.0,
                    "realized_avg_mfe": 0.0,
                    "fallback_to_prior_rate": 0.0,
                    "context_supported_decision_rate": 0.0,
                    "local_bracket_override_rate": 0.0,
                    "strong_context_support_count": 0,
                    "mid_context_support_count": 0,
                    "low_context_support_count": 0,
                },
            }
        family["family_usability_state"] = str((family.get("family_runtime_state", {}) or {}).get("usability_state", "low_support"))
        family["family_evidence_support_tier"] = str((family.get("family_runtime_state", {}) or {}).get("evidence_support_tier", "none"))
        family["family_competition_status"] = str((family.get("family_runtime_state", {}) or {}).get("competition_status", "suppressed"))
        family["family_usability_adjustment"] = float(safe_float((family.get("family_runtime_state", {}) or {}).get("usability_adjustment", 0.0), 0.0))
        family["family_bootstrap_competition_included"] = bool((family.get("family_runtime_state", {}) or {}).get("bootstrap_competition_included", False))
        family["family_prior_eligible"] = bool((family.get("family_runtime_state", {}) or {}).get("prior_eligible", False))
        family["family_prior_eligibility_reason"] = str((family.get("family_runtime_state", {}) or {}).get("prior_eligibility_reason", "") or "")
        family["family_competition_eligible"] = bool((family.get("family_runtime_state", {}) or {}).get("competition_eligible", False))
        family["family_competition_eligibility_reason"] = str((family.get("family_runtime_state", {}) or {}).get("competition_eligibility_reason", "") or "")

    runtime_state_meta = (
        runtime_state_result.get("runtime_state_meta", {})
        if isinstance(runtime_state_result.get("runtime_state_meta"), dict)
        else {}
    )
    usable_universe_summary = (
        runtime_state_result.get("usable_universe_summary", {})
        if isinstance(runtime_state_result.get("usable_universe_summary"), dict)
        else {}
    )
    runtime_state_build_meta = {
        "enabled": bool(usable_defaults.get("enabled", True)),
        "decision_csv_path": str(decision_csv_path),
        "trade_attribution_csv_path": str(trade_attribution_csv_path),
        "runtime_state_json_path": str(runtime_state_json_path),
        "source_counts": dict(decision_source_counts),
        "state_config": dict(usable_defaults),
        "prior_eligibility_config": dict(prior_defaults),
        "family_competition_config": dict(competition_defaults),
        "runtime_state_meta": dict(runtime_state_meta),
        "usable_universe_summary": dict(usable_universe_summary),
    }
    try:
        runtime_state_json_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_state_json_path.write_text(
            json.dumps(
                {
                    "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
                    "source_v2_path": str(source_v2_path),
                    "source_v2_version": str(source_v2_version),
                    "runtime_state_meta": runtime_state_meta,
                    "usable_universe_summary": usable_universe_summary,
                    "family_runtime_states": family_runtime_states,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        runtime_state_build_meta["runtime_state_json_written"] = True
    except Exception as exc:
        runtime_state_build_meta["runtime_state_json_written"] = False
        runtime_state_build_meta["runtime_state_json_error"] = str(exc)

    payload = {
        "version": "v3_family_inventory",
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "source_v2_path": str(source_v2_path),
        "source_v2_version": str(source_v2_version),
        "family_identity_fields": ["timeframe", "session", "side", "de3_strategy_type", "threshold"],
        "member_identity_fields": ["timeframe", "session", "side", "de3_strategy_type", "threshold", "sl", "tp"],
        "family_context_profiles_enabled": bool(context_profiles_enabled),
        "family_context_profile_build": cp_build_meta,
        "family_runtime_state_build": runtime_state_build_meta,
        "v3_native_usable_family_universe": usable_universe_summary,
        "summary": {
            "family_count": int(len(families)),
            "member_count": int(sum(int(row.get("member_count", 0) or 0) for row in families)),
            "v3_native_usable_family_count": int(usable_universe_summary.get("v3_native_usable_family_universe_size", 0) or 0),
            "v3_active_runtime_family_count": int(usable_universe_summary.get("active_runtime_competition_family_count", 0) or 0),
        },
        "families": families,
    }
    return payload


def load_de3_v3_family_inventory(path: Path, *, prefer_refined: bool = False) -> Dict[str, Any]:
    path = _resolve_path(path)
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid family payload at {path}")
    if isinstance(payload.get("families"), list):
        return payload

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    source_inputs = metadata.get("source_inputs", {}) if isinstance(metadata.get("source_inputs"), dict) else {}
    runtime_defaults = payload.get("runtime_state_defaults", {}) if isinstance(payload.get("runtime_state_defaults"), dict) else {}
    context_profiles = payload.get("context_profiles", {}) if isinstance(payload.get("context_profiles"), dict) else {}
    refinement_summary = payload.get("refinement_summary", {}) if isinstance(payload.get("refinement_summary"), dict) else {}
    runtime_core_sat_state = (
        payload.get("runtime_core_satellite_state", {})
        if isinstance(payload.get("runtime_core_satellite_state"), dict)
        else {}
    )
    core_families = payload.get("core_families", {}) if isinstance(payload.get("core_families"), dict) else {}
    core_members = payload.get("core_members", {}) if isinstance(payload.get("core_members"), dict) else {}
    satellite_quality_summary = (
        payload.get("satellite_quality_summary", {})
        if isinstance(payload.get("satellite_quality_summary"), dict)
        else {}
    )
    portfolio_incremental_tests = (
        payload.get("portfolio_incremental_tests", {})
        if isinstance(payload.get("portfolio_incremental_tests"), dict)
        else {}
    )
    orthogonality_summary = (
        payload.get("orthogonality_summary", {})
        if isinstance(payload.get("orthogonality_summary"), dict)
        else {}
    )
    core_summary = (
        payload.get("core_summary", {})
        if isinstance(payload.get("core_summary"), dict)
        else {}
    )
    t6_anchor_report = (
        payload.get("t6_anchor_report", {})
        if isinstance(payload.get("t6_anchor_report"), dict)
        else {}
    )
    runtime_mode_summary = (
        payload.get("runtime_mode_summary", {})
        if isinstance(payload.get("runtime_mode_summary"), dict)
        else {}
    )

    # Consolidated bundle compatibility.
    legacy = payload.get("legacy_family_inventory")
    refined_section = payload.get("refined_family_inventory") if isinstance(payload.get("refined_family_inventory"), dict) else {}
    retained_runtime_section = (
        payload.get("retained_runtime_universe")
        if isinstance(payload.get("retained_runtime_universe"), dict)
        else {}
    )
    raw_section = payload.get("raw_family_inventory") if isinstance(payload.get("raw_family_inventory"), dict) else {}
    refined_rows = list(refined_section.get("families", []) if isinstance(refined_section.get("families"), list) else [])
    raw_rows = list(raw_section.get("families", []) if isinstance(raw_section.get("families"), list) else [])

    selected_rows: List[Dict[str, Any]] = []
    selected_universe = "legacy"
    retained_runtime_rows = list(
        retained_runtime_section.get("families", [])
        if isinstance(retained_runtime_section.get("families"), list)
        else []
    )
    if bool(prefer_refined) and retained_runtime_rows:
        selected_rows = list(retained_runtime_rows)
        selected_universe = "retained_runtime"
        # Ensure configured core families remain reachable even if refined retention omitted them.
        core_ids_from_bundle = (
            runtime_core_sat_state.get("core_family_ids", [])
            if isinstance(runtime_core_sat_state.get("core_family_ids"), (list, tuple, set))
            else []
        )
        core_ids = {
            str(v).strip()
            for v in core_ids_from_bundle
            if str(v).strip()
        }
        if core_ids:
            selected_ids = {
                str((row or {}).get("family_id", "")).strip()
                for row in selected_rows
                if isinstance(row, dict)
            }
            by_id: Dict[str, Dict[str, Any]] = {}
            for row in refined_rows:
                if isinstance(row, dict):
                    fid = str(row.get("family_id", "")).strip()
                    if fid:
                        by_id[fid] = row
            for row in raw_rows:
                if isinstance(row, dict):
                    fid = str(row.get("family_id", "")).strip()
                    if fid and fid not in by_id:
                        by_id[fid] = row
            for fid in sorted(core_ids):
                if fid and fid not in selected_ids and fid in by_id:
                    selected_rows.append(by_id[fid])
                    selected_ids.add(fid)
    elif bool(prefer_refined) and refined_rows:
        filtered_refined_rows = [
            row for row in refined_rows
            if not isinstance(row, dict) or bool(row.get("family_retained", True))
        ]
        if filtered_refined_rows:
            selected_rows = filtered_refined_rows
            selected_universe = "refined_retained_filter"
        else:
            selected_rows = refined_rows
            selected_universe = "refined"
    elif isinstance(legacy, dict) and isinstance(legacy.get("families"), list):
        out = dict(legacy)
        out["_artifact_kind"] = "bundle"
        out["_bundle_selected_universe"] = "legacy"
        out["_bundle_metadata"] = dict(metadata)
        out["_bundle_version"] = str(payload.get("bundle_version", "") or "")
        out["_bundle_refinement_summary"] = dict(refinement_summary)
        out["_bundle_runtime_state_defaults"] = dict(runtime_defaults)
        out["_bundle_raw_family_count"] = int(len(raw_rows))
        out["_bundle_refined_family_count"] = int(len(refined_rows))
        out["_bundle_retained_runtime_family_count"] = int(len(retained_runtime_rows))
        out["_bundle_runtime_core_satellite_state"] = dict(runtime_core_sat_state)
        out["_bundle_core_families"] = dict(core_families)
        out["_bundle_core_members"] = dict(core_members)
        out["_bundle_satellite_quality_summary"] = dict(satellite_quality_summary)
        out["_bundle_portfolio_incremental_tests"] = dict(portfolio_incremental_tests)
        out["_bundle_orthogonality_summary"] = dict(orthogonality_summary)
        out["_bundle_core_summary"] = dict(core_summary)
        out["_bundle_t6_anchor_report"] = dict(t6_anchor_report)
        out["_bundle_runtime_mode_summary"] = dict(runtime_mode_summary)
        return out
    elif raw_rows:
        selected_rows = raw_rows
        selected_universe = "raw"
    elif refined_rows:
        selected_rows = refined_rows
        selected_universe = "refined"

    if selected_rows:
        out = {
            "version": "v3_family_inventory_from_bundle",
            "generated_at": metadata.get("build_timestamp", payload.get("generated_at")),
            "source_v2_path": source_inputs.get("source_v2_path", payload.get("source_v2_path", "")),
            "source_v2_version": source_inputs.get("source_v2_version", payload.get("source_v2_version", "")),
            "family_identity_fields": payload.get("family_identity_fields", ["timeframe", "session", "side", "de3_strategy_type", "threshold"]),
            "member_identity_fields": payload.get("member_identity_fields", ["timeframe", "session", "side", "de3_strategy_type", "threshold", "sl", "tp"]),
            "family_context_profiles_enabled": bool(context_profiles.get("enabled", True)),
            "family_context_profile_build": dict(context_profiles.get("build_meta", payload.get("family_context_profile_build", {})) or {}),
            "family_runtime_state_build": {
                "runtime_state_meta": dict(runtime_defaults.get("runtime_state_meta", {}) or {}),
                "usable_universe_summary": dict(runtime_defaults.get("usable_universe_summary", {}) or {}),
                "runtime_state_json_path": runtime_defaults.get("runtime_state_json_path"),
                "source_counts": dict(runtime_defaults.get("source_counts", {}) or {}),
            },
            "v3_native_usable_family_universe": dict(runtime_defaults.get("usable_universe_summary", {}) or {}),
            "summary": {
                "family_count": int(len(selected_rows)),
                "member_count": int(sum(int((row or {}).get("member_count", 0) or 0) for row in selected_rows)),
            },
            "families": selected_rows,
        }
        out["_artifact_kind"] = "bundle"
        out["_bundle_selected_universe"] = str(selected_universe)
        out["_bundle_metadata"] = dict(metadata)
        out["_bundle_version"] = str(payload.get("bundle_version", "") or "")
        out["_bundle_refinement_summary"] = dict(refinement_summary)
        out["_bundle_runtime_state_defaults"] = dict(runtime_defaults)
        out["_bundle_raw_family_count"] = int(len(raw_rows))
        out["_bundle_refined_family_count"] = int(len(refined_rows))
        out["_bundle_retained_runtime_family_count"] = int(len(retained_runtime_rows))
        out["_bundle_runtime_core_satellite_state"] = dict(runtime_core_sat_state)
        out["_bundle_core_families"] = dict(core_families)
        out["_bundle_core_members"] = dict(core_members)
        out["_bundle_satellite_quality_summary"] = dict(satellite_quality_summary)
        out["_bundle_portfolio_incremental_tests"] = dict(portfolio_incremental_tests)
        out["_bundle_orthogonality_summary"] = dict(orthogonality_summary)
        out["_bundle_core_summary"] = dict(core_summary)
        out["_bundle_t6_anchor_report"] = dict(t6_anchor_report)
        out["_bundle_runtime_mode_summary"] = dict(runtime_mode_summary)
        return out

    raise ValueError(f"Family payload missing families list: {path}")


def build_and_write_de3_v3_family_inventory(
    *,
    source_v2_path: Path,
    out_path: Path,
    decision_csv_path: Optional[Path] = None,
    trade_attribution_csv_path: Optional[Path] = None,
    min_bucket_samples: Optional[int] = None,
    strong_bucket_samples: Optional[int] = None,
    context_profiles_enabled: Optional[bool] = None,
    allow_parse_legacy_context_inputs: Optional[bool] = None,
) -> Dict[str, Any]:
    payload = build_de3_v3_family_inventory(
        source_v2_path=_resolve_path(source_v2_path),
        decision_csv_path=decision_csv_path,
        trade_attribution_csv_path=trade_attribution_csv_path,
        min_bucket_samples=min_bucket_samples,
        strong_bucket_samples=strong_bucket_samples,
        context_profiles_enabled=context_profiles_enabled,
        allow_parse_legacy_context_inputs=allow_parse_legacy_context_inputs,
    )
    out_path = _resolve_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    cp_defaults = _default_context_profile_cfg()
    parser = argparse.ArgumentParser(
        description="Build dynamic_engine3_families_v3.json from DE3 v2 members and historical decision exports."
    )
    parser.add_argument("--source-v2", default="dynamic_engine3_strategies_v2.json", help="Path to DE3 v2 member DB JSON.")
    parser.add_argument("--out", default="dynamic_engine3_families_v3.json", help="Output family inventory JSON.")
    parser.add_argument(
        "--decisions-csv",
        default=cp_defaults.get("decision_csv_path", "reports/de3_decisions.csv"),
        help="Decision export CSV path.",
    )
    parser.add_argument(
        "--trade-attribution-csv",
        default=cp_defaults.get("trade_attribution_csv_path", "reports/de3_decisions_trade_attribution.csv"),
        help="Decision trade attribution CSV path.",
    )
    parser.add_argument(
        "--min-bucket-samples",
        type=int,
        default=safe_int(cp_defaults.get("min_bucket_samples", 12), 12),
        help="Minimum sample count before a profile bucket is not low-support.",
    )
    parser.add_argument(
        "--strong-bucket-samples",
        type=int,
        default=safe_int(cp_defaults.get("strong_bucket_samples", 40), 40),
        help="Sample count for strong context support.",
    )
    parser.add_argument("--disable-context-profiles", action="store_true", help="Build family inventory without context profiles.")
    parser.add_argument(
        "--disable-legacy-context-parse",
        action="store_true",
        help="Disable parsing legacy family_context_inputs string payloads.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args(argv)
    payload = build_and_write_de3_v3_family_inventory(
        source_v2_path=_resolve_path(args.source_v2),
        out_path=_resolve_path(args.out),
        decision_csv_path=_resolve_path(args.decisions_csv),
        trade_attribution_csv_path=_resolve_path(args.trade_attribution_csv),
        min_bucket_samples=int(max(1, args.min_bucket_samples)),
        strong_bucket_samples=int(max(1, args.strong_bucket_samples)),
        context_profiles_enabled=not bool(args.disable_context_profiles),
        allow_parse_legacy_context_inputs=not bool(args.disable_legacy_context_parse),
    )
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    cp_meta = payload.get("family_context_profile_build", {}) if isinstance(payload.get("family_context_profile_build"), dict) else {}
    audit = cp_meta.get("audit", {}) if isinstance(cp_meta.get("audit"), dict) else {}
    runtime_meta = payload.get("family_runtime_state_build", {}) if isinstance(payload.get("family_runtime_state_build"), dict) else {}
    usable_summary = payload.get("v3_native_usable_family_universe", {}) if isinstance(payload.get("v3_native_usable_family_universe"), dict) else {}
    logging.info(
        "DE3v3 family inventory built: families=%s members=%s out=%s",
        summary.get("family_count", 0),
        summary.get("member_count", 0),
        _resolve_path(args.out),
    )
    logging.info(
        "Context profile build: enabled=%s decisions_used=%s audit_status=%s enriched_required=%s",
        cp_meta.get("enabled", False),
        cp_meta.get("decisions_with_outcomes_used", 0),
        audit.get("status"),
        audit.get("enriched_export_required_for_full_bucketing"),
    )
    logging.info(
        "V3-native family universe: inherited=%s prior_eligible=%s usable=%s active=%s suppressed=%s tiers=%s competition=%s bootstrap=%s runtime_state_status=%s",
        usable_summary.get("inherited_family_universe_size"),
        usable_summary.get("prior_eligible_family_count"),
        usable_summary.get("v3_native_usable_family_universe_size"),
        usable_summary.get("active_runtime_competition_family_count"),
        usable_summary.get("suppressed_family_count"),
        usable_summary.get("evidence_support_tier_counts"),
        usable_summary.get("competition_status_counts"),
        usable_summary.get("bootstrap_included_family_count"),
        (runtime_meta.get("runtime_state_meta", {}) or {}).get("status"),
    )


if __name__ == "__main__":
    main()
