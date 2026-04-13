import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_json_optional(path_raw: str) -> Dict[str, Any]:
    text = str(path_raw or "").strip()
    if not text:
        return {}
    return _load_json(Path(text))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _is_missing_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return True
    if not payload:
        return True
    return False


def _section_status(prev_payload: Dict[str, Any], curr_payload: Dict[str, Any]) -> str:
    prev_missing = _is_missing_payload(prev_payload)
    curr_missing = _is_missing_payload(curr_payload)
    if prev_missing and curr_missing:
        return "unavailable"
    if prev_missing or curr_missing:
        return "partial_unavailable"
    return "available"


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            key_text = str(key)
            next_prefix = f"{prefix}.{key_text}" if prefix else key_text
            out.update(_flatten(sub, next_prefix))
    else:
        out[str(prefix)] = value
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _load_csv_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _extract_backtest_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    stats = payload.get("stats", {}) if isinstance(payload.get("stats"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    source = {}
    for candidate in (stats, summary, payload):
        if isinstance(candidate, dict):
            source.update(candidate)
    return {
        "total_trades": int(_safe_float(source.get("total_trades", source.get("trades", 0)), 0)),
        "net_pnl": float(
            _safe_float(
                source.get(
                    "net_pnl",
                    source.get("total_pnl", source.get("net_profit", source.get("pnl_points", 0.0))),
                ),
                0.0,
            )
        ),
        "max_drawdown": float(
            _safe_float(
                source.get("max_drawdown", source.get("max_drawdown_points", source.get("drawdown", 0.0))),
                0.0,
            )
        ),
    }


def _distribution_distance(prev_dist: Dict[str, Any], curr_dist: Dict[str, Any]) -> float:
    prev_total = float(sum(_safe_float(v, 0.0) for v in prev_dist.values()))
    curr_total = float(sum(_safe_float(v, 0.0) for v in curr_dist.values()))
    if prev_total <= 0.0 and curr_total <= 0.0:
        return 0.0
    keys = set(str(k) for k in prev_dist.keys()) | set(str(k) for k in curr_dist.keys())
    l1 = 0.0
    for key in keys:
        p = _safe_float(prev_dist.get(key, 0.0), 0.0) / (prev_total if prev_total > 0 else 1.0)
        c = _safe_float(curr_dist.get(key, 0.0), 0.0) / (curr_total if curr_total > 0 else 1.0)
        l1 += abs(p - c)
    return float(0.5 * l1)


def _chosen_decomposition_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    chosen_rows = []
    for row in rows:
        chosen = str(row.get("chosen", "")).strip().lower() in {"1", "true", "yes", "y", "t"}
        abstained = str(row.get("abstained", "")).strip().lower() in {"1", "true", "yes", "y", "t"}
        if chosen and not abstained:
            chosen_rows.append(row)
    if not chosen_rows:
        return {"row_count": 0, "means": {}}
    fields = [
        "chosen_prior_component",
        "chosen_trusted_context_component",
        "chosen_evidence_adjustment",
        "chosen_adaptive_component",
        "chosen_competition_diversity_adjustment",
        "chosen_pre_adjustment_score",
        "chosen_final_family_score",
        "chosen_vs_runner_up_score_delta",
    ]
    means = {}
    for field in fields:
        vals = [_safe_float(row.get(field, 0.0), 0.0) for row in chosen_rows]
        means[field] = float(sum(vals) / float(max(1, len(vals))))
    return {"row_count": int(len(chosen_rows)), "means": means}


def _score_branches_fired(counters: Dict[str, Any]) -> bool:
    branch_keys = [
        "runtime_invocations",
        "family_candidate_set_size_gt_1_count",
        "context_profile_used_count",
        "exploration_bonus_applied_count",
        "dominance_penalty_applied_count",
        "context_advantage_capped_count",
    ]
    return any(_safe_float(counters.get(key, 0), 0.0) > 0 for key in branch_keys)


def _decomposition_status(decomp: Dict[str, Any], counters: Dict[str, Any]) -> Dict[str, Any]:
    row_count = int(decomp.get("row_count", 0) or 0)
    means = decomp.get("means", {}) if isinstance(decomp.get("means"), dict) else {}
    if row_count <= 0:
        return {
            "status": "unavailable",
            "all_zero_means": False,
            "scoring_branches_fired": bool(_score_branches_fired(counters)),
            "likely_score_export_bug": False,
        }
    all_zero_means = bool(not means or all(abs(_safe_float(v, 0.0)) <= 1e-12 for v in means.values()))
    branches_fired = bool(_score_branches_fired(counters))
    suspicious = bool(all_zero_means and branches_fired)
    return {
        "status": "suspicious_zeroed" if suspicious else "valid",
        "all_zero_means": bool(all_zero_means),
        "scoring_branches_fired": bool(branches_fired),
        "likely_score_export_bug": bool(suspicious),
    }


def _counter_diff(prev_counters: Dict[str, Any], curr_counters: Dict[str, Any]) -> Dict[str, Any]:
    keys = sorted(set(prev_counters.keys()) | set(curr_counters.keys()))
    out = {}
    for key in keys:
        prev_val = _safe_float(prev_counters.get(key, 0), 0.0)
        curr_val = _safe_float(curr_counters.get(key, 0), 0.0)
        delta = curr_val - prev_val
        rel = (delta / abs(prev_val)) if abs(prev_val) > 1e-9 else (1.0 if abs(curr_val) > 1e-9 else 0.0)
        out[key] = {
            "prev": prev_val,
            "curr": curr_val,
            "delta": delta,
            "relative_change": rel,
        }
    return out


def _dict_change_summary(prev_payload: Dict[str, Any], curr_payload: Dict[str, Any]) -> Dict[str, Any]:
    status = _section_status(prev_payload, curr_payload)
    prev_flat = _flatten(prev_payload)
    curr_flat = _flatten(curr_payload)
    changed = []
    for key in sorted(set(prev_flat.keys()) | set(curr_flat.keys())):
        pv = prev_flat.get(key)
        cv = curr_flat.get(key)
        if pv != cv:
            changed.append({"key": key, "prev": pv, "curr": cv})
    return {
        "status": str(status),
        "materially_changed": bool(status == "available" and len(changed) > 0),
        "changed_count": int(len(changed)),
        "changed_examples": changed[:200],
    }


def _share_triplet(counters: Dict[str, Any]) -> Dict[str, float]:
    strong = _safe_float(counters.get("strong_support_decision_count", 0), 0.0)
    mid = _safe_float(counters.get("mid_support_decision_count", 0), 0.0)
    low = _safe_float(counters.get("low_or_none_support_decision_count", 0), 0.0)
    total = max(1.0, strong + mid + low)
    return {
        "strong": float(strong / total),
        "mid": float(mid / total),
        "low_or_none": float(low / total),
    }


def _mode_share(counters: Dict[str, Any]) -> Dict[str, float]:
    full = _safe_float(counters.get("local_bracket_full_mode_count", 0), 0.0)
    conservative = _safe_float(counters.get("local_bracket_conservative_mode_count", 0), 0.0)
    frozen = _safe_float(counters.get("local_bracket_frozen_mode_count", 0), 0.0)
    total = max(1.0, full + conservative + frozen)
    return {
        "full": float(full / total),
        "conservative": float(conservative / total),
        "frozen": float(frozen / total),
    }


def _auto_family_decisions_path(family_summary_path: Path) -> Optional[Path]:
    candidate = family_summary_path.with_name("de3_family_decisions.csv")
    return candidate if candidate.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two DE3v3 runs and flag likely inert changes.")
    parser.add_argument("--prev-backtest-json", required=True)
    parser.add_argument("--curr-backtest-json", required=True)
    parser.add_argument("--prev-family-summary", required=True)
    parser.add_argument("--curr-family-summary", required=True)
    parser.add_argument("--prev-activation-audit", required=True)
    parser.add_argument("--curr-activation-audit", required=True)
    parser.add_argument("--prev-runtime-counters", required=True)
    parser.add_argument("--curr-runtime-counters", required=True)
    parser.add_argument("--prev-family-decisions-csv", default="")
    parser.add_argument("--curr-family-decisions-csv", default="")
    parser.add_argument("--prev-config-usage-audit", default="")
    parser.add_argument("--curr-config-usage-audit", default="")
    parser.add_argument("--prev-bundle-usage-audit", default="")
    parser.add_argument("--curr-bundle-usage-audit", default="")
    parser.add_argument("--prev-score-path-audit", default="")
    parser.add_argument("--curr-score-path-audit", default="")
    parser.add_argument("--prev-choice-path-audit", default="")
    parser.add_argument("--curr-choice-path-audit", default="")
    parser.add_argument("--prev-refined-vs-raw-audit", default="")
    parser.add_argument("--curr-refined-vs-raw-audit", default="")
    parser.add_argument("--prev-family-competition-health", default="")
    parser.add_argument("--curr-family-competition-health", default="")
    parser.add_argument("--prev-member-resolution-audit", default="")
    parser.add_argument("--curr-member-resolution-audit", default="")
    parser.add_argument("--prev-diff-readiness", default="")
    parser.add_argument("--curr-diff-readiness", default="")
    parser.add_argument("--out", default="reports/de3_v3_diff_summary.json")
    parser.add_argument("--out-inert", default="reports/de3_v3_inert_change_summary.json")
    args = parser.parse_args()

    prev_backtest = _load_json(Path(args.prev_backtest_json))
    curr_backtest = _load_json(Path(args.curr_backtest_json))
    prev_family_summary_path = Path(args.prev_family_summary)
    curr_family_summary_path = Path(args.curr_family_summary)
    prev_family_summary = _load_json(prev_family_summary_path)
    curr_family_summary = _load_json(curr_family_summary_path)
    prev_activation = _load_json(Path(args.prev_activation_audit))
    curr_activation = _load_json(Path(args.curr_activation_audit))
    prev_counters_payload = _load_json(Path(args.prev_runtime_counters))
    curr_counters_payload = _load_json(Path(args.curr_runtime_counters))
    prev_config_usage = _load_json_optional(args.prev_config_usage_audit)
    curr_config_usage = _load_json_optional(args.curr_config_usage_audit)
    prev_bundle_usage = _load_json_optional(args.prev_bundle_usage_audit)
    curr_bundle_usage = _load_json_optional(args.curr_bundle_usage_audit)
    prev_score_path_audit = _load_json_optional(args.prev_score_path_audit)
    curr_score_path_audit = _load_json_optional(args.curr_score_path_audit)
    prev_choice_path_audit = _load_json_optional(args.prev_choice_path_audit)
    curr_choice_path_audit = _load_json_optional(args.curr_choice_path_audit)
    prev_refined_vs_raw = _load_json_optional(args.prev_refined_vs_raw_audit)
    curr_refined_vs_raw = _load_json_optional(args.curr_refined_vs_raw_audit)
    prev_competition_health = _load_json_optional(args.prev_family_competition_health)
    curr_competition_health = _load_json_optional(args.curr_family_competition_health)
    prev_member_resolution = _load_json_optional(args.prev_member_resolution_audit)
    curr_member_resolution = _load_json_optional(args.curr_member_resolution_audit)
    prev_diff_readiness = _load_json_optional(args.prev_diff_readiness)
    curr_diff_readiness = _load_json_optional(args.curr_diff_readiness)

    prev_counters = dict(prev_counters_payload.get("counters", {}) if isinstance(prev_counters_payload.get("counters"), dict) else prev_counters_payload)
    curr_counters = dict(curr_counters_payload.get("counters", {}) if isinstance(curr_counters_payload.get("counters"), dict) else curr_counters_payload)

    prev_decisions_path = Path(args.prev_family_decisions_csv) if str(args.prev_family_decisions_csv).strip() else _auto_family_decisions_path(prev_family_summary_path)
    curr_decisions_path = Path(args.curr_family_decisions_csv) if str(args.curr_family_decisions_csv).strip() else _auto_family_decisions_path(curr_family_summary_path)
    prev_decisions = _load_csv_rows(prev_decisions_path)
    curr_decisions = _load_csv_rows(curr_decisions_path)

    prev_dist = prev_family_summary.get("family_chosen_frequency", {}) if isinstance(prev_family_summary.get("family_chosen_frequency"), dict) else {}
    curr_dist = curr_family_summary.get("family_chosen_frequency", {}) if isinstance(curr_family_summary.get("family_chosen_frequency"), dict) else {}
    dist_delta = _distribution_distance(prev_dist, curr_dist)
    prev_health = prev_family_summary.get("family_competition_health", {}) if isinstance(prev_family_summary.get("family_competition_health"), dict) else {}
    curr_health = curr_family_summary.get("family_competition_health", {}) if isinstance(curr_family_summary.get("family_competition_health"), dict) else {}
    top_share_delta = abs(
        _safe_float(curr_health.get("top_family_chosen_share", 0.0), 0.0)
        - _safe_float(prev_health.get("top_family_chosen_share", 0.0), 0.0)
    )
    unique_delta = int(curr_health.get("chosen_family_unique_count", 0) or 0) - int(prev_health.get("chosen_family_unique_count", 0) or 0)

    prev_bt = _extract_backtest_metrics(prev_backtest)
    curr_bt = _extract_backtest_metrics(curr_backtest)
    prev_decomp = _chosen_decomposition_metrics(prev_decisions)
    curr_decomp = _chosen_decomposition_metrics(curr_decisions)
    prev_decomp_status = _decomposition_status(prev_decomp, prev_counters)
    curr_decomp_status = _decomposition_status(curr_decomp, curr_counters)
    decomp_deltas = {}
    for key in set(prev_decomp.get("means", {}).keys()) | set(curr_decomp.get("means", {}).keys()):
        decomp_deltas[key] = float(
            _safe_float(curr_decomp.get("means", {}).get(key, 0.0), 0.0)
            - _safe_float(prev_decomp.get("means", {}).get(key, 0.0), 0.0)
        )

    prev_support_share = _share_triplet(prev_counters)
    curr_support_share = _share_triplet(curr_counters)
    support_delta = {
        key: float(curr_support_share.get(key, 0.0) - prev_support_share.get(key, 0.0))
        for key in {"strong", "mid", "low_or_none"}
    }
    prev_mode_share = _mode_share(prev_counters)
    curr_mode_share = _mode_share(curr_counters)
    mode_delta = {
        key: float(curr_mode_share.get(key, 0.0) - prev_mode_share.get(key, 0.0))
        for key in {"full", "conservative", "frozen"}
    }

    section_diffs = {
        "activation_audit": _dict_change_summary(prev_activation, curr_activation),
        "config_usage_audit": _dict_change_summary(prev_config_usage, curr_config_usage),
        "bundle_usage_audit": _dict_change_summary(prev_bundle_usage, curr_bundle_usage),
        "score_path_audit": _dict_change_summary(prev_score_path_audit, curr_score_path_audit),
        "choice_path_audit": _dict_change_summary(prev_choice_path_audit, curr_choice_path_audit),
        "refined_vs_raw_audit": _dict_change_summary(prev_refined_vs_raw, curr_refined_vs_raw),
        "family_competition_health": _dict_change_summary(
            prev_competition_health, curr_competition_health
        ),
        "member_resolution_audit": _dict_change_summary(
            prev_member_resolution, curr_member_resolution
        ),
        "diff_readiness": _dict_change_summary(prev_diff_readiness, curr_diff_readiness),
    }

    counters_delta = _counter_diff(prev_counters, curr_counters)

    prev_cfg_flat = _flatten(prev_activation.get("exact_relevant_config_snapshot", {}), "DE3_V3")
    curr_cfg_flat = _flatten(curr_activation.get("exact_relevant_config_snapshot", {}), "DE3_V3")
    cfg_keys = sorted(set(prev_cfg_flat.keys()) | set(curr_cfg_flat.keys()))
    changed_config = []
    for key in cfg_keys:
        prev_val = prev_cfg_flat.get(key)
        curr_val = curr_cfg_flat.get(key)
        if prev_val != curr_val:
            changed_config.append({"key": key, "prev": prev_val, "curr": curr_val})

    materially_changed_distribution = bool(dist_delta >= 0.10 or top_share_delta >= 0.05 or abs(unique_delta) >= 1)
    materially_changed_candidates = bool(
        abs(_safe_float(counters_delta.get("family_candidate_set_size_eq_1_count", {}).get("delta", 0.0), 0.0)) >= 5
        or abs(_safe_float(counters_delta.get("family_candidate_set_size_gt_1_count", {}).get("delta", 0.0), 0.0)) >= 5
    )
    materially_changed_decomposition = bool(
        any(abs(v) >= 0.05 for v in decomp_deltas.values())
    ) if decomp_deltas else False
    decomposition_suspicious_zeroed = bool(
        prev_decomp_status.get("status") == "suspicious_zeroed"
        or curr_decomp_status.get("status") == "suspicious_zeroed"
    )
    materially_changed_support = bool(any(abs(v) >= 0.10 for v in support_delta.values()))
    materially_changed_modes = bool(any(abs(v) >= 0.10 for v in mode_delta.values()))
    materially_changed_branches = bool(
        any(
            abs(_safe_float(item.get("delta", 0.0), 0.0)) >= 5
            and abs(_safe_float(item.get("relative_change", 0.0), 0.0)) >= 0.20
            for item in counters_delta.values()
        )
    )

    inert_changed_config = []
    for row in changed_config:
        key = str(row.get("key", ""))
        if "family_competition" in key and not materially_changed_branches:
            inert_changed_config.append(dict(row, reason="competition_config_changed_without_branch_delta"))
        elif "local_member_selection" in key and not materially_changed_modes:
            inert_changed_config.append(dict(row, reason="local_member_config_changed_without_mode_delta"))
        elif ("family_scoring" in key or "context_profiles" in key) and not (
            materially_changed_decomposition or materially_changed_support or decomposition_suspicious_zeroed
        ):
            inert_changed_config.append(dict(row, reason="scoring_or_context_config_changed_without_score_shift"))
        elif "refined_universe" in key and not materially_changed_candidates:
            inert_changed_config.append(dict(row, reason="refined_universe_config_changed_without_candidate_shift"))

    out_payload = {
        "created_at": curr_activation.get("created_at"),
        "inputs": {
            "prev_backtest_json": str(args.prev_backtest_json),
            "curr_backtest_json": str(args.curr_backtest_json),
            "prev_family_summary": str(args.prev_family_summary),
            "curr_family_summary": str(args.curr_family_summary),
            "prev_activation_audit": str(args.prev_activation_audit),
            "curr_activation_audit": str(args.curr_activation_audit),
            "prev_runtime_counters": str(args.prev_runtime_counters),
            "curr_runtime_counters": str(args.curr_runtime_counters),
            "prev_family_decisions_csv": str(prev_decisions_path) if prev_decisions_path else None,
            "curr_family_decisions_csv": str(curr_decisions_path) if curr_decisions_path else None,
            "prev_config_usage_audit": str(args.prev_config_usage_audit or ""),
            "curr_config_usage_audit": str(args.curr_config_usage_audit or ""),
            "prev_bundle_usage_audit": str(args.prev_bundle_usage_audit or ""),
            "curr_bundle_usage_audit": str(args.curr_bundle_usage_audit or ""),
            "prev_score_path_audit": str(args.prev_score_path_audit or ""),
            "curr_score_path_audit": str(args.curr_score_path_audit or ""),
            "prev_choice_path_audit": str(args.prev_choice_path_audit or ""),
            "curr_choice_path_audit": str(args.curr_choice_path_audit or ""),
            "prev_refined_vs_raw_audit": str(args.prev_refined_vs_raw_audit or ""),
            "curr_refined_vs_raw_audit": str(args.curr_refined_vs_raw_audit or ""),
            "prev_family_competition_health": str(args.prev_family_competition_health or ""),
            "curr_family_competition_health": str(args.curr_family_competition_health or ""),
            "prev_member_resolution_audit": str(args.prev_member_resolution_audit or ""),
            "curr_member_resolution_audit": str(args.curr_member_resolution_audit or ""),
            "prev_diff_readiness": str(args.prev_diff_readiness or ""),
            "curr_diff_readiness": str(args.curr_diff_readiness or ""),
        },
        "backtest_metrics": {
            "prev": prev_bt,
            "curr": curr_bt,
            "delta": {
                "total_trades": int(curr_bt.get("total_trades", 0) - prev_bt.get("total_trades", 0)),
                "net_pnl": float(curr_bt.get("net_pnl", 0.0) - prev_bt.get("net_pnl", 0.0)),
                "max_drawdown": float(curr_bt.get("max_drawdown", 0.0) - prev_bt.get("max_drawdown", 0.0)),
            },
        },
        "chosen_family_distribution": {
            "prev": prev_dist,
            "curr": curr_dist,
            "distance_l1_half": float(dist_delta),
            "top_family_share_delta": float(top_share_delta),
            "chosen_family_unique_count_delta": int(unique_delta),
            "materially_changed": bool(materially_changed_distribution),
        },
        "family_candidate_counts": {
            "prev_eq_1": int(_safe_float(prev_counters.get("family_candidate_set_size_eq_1_count", 0), 0)),
            "curr_eq_1": int(_safe_float(curr_counters.get("family_candidate_set_size_eq_1_count", 0), 0)),
            "prev_gt_1": int(_safe_float(prev_counters.get("family_candidate_set_size_gt_1_count", 0), 0)),
            "curr_gt_1": int(_safe_float(curr_counters.get("family_candidate_set_size_gt_1_count", 0), 0)),
            "materially_changed": bool(materially_changed_candidates),
        },
        "score_decomposition": {
            "prev": prev_decomp,
            "curr": curr_decomp,
            "prev_status": prev_decomp_status,
            "curr_status": curr_decomp_status,
            "status": (
                "suspicious_zeroed"
                if decomposition_suspicious_zeroed
                else ("unavailable" if curr_decomp_status.get("status") == "unavailable" else "valid")
            ),
            "likely_score_export_bug": bool(decomposition_suspicious_zeroed),
            "deltas": decomp_deltas,
            "materially_changed": bool(materially_changed_decomposition),
        },
        "support_tier_usage": {
            "prev_share": prev_support_share,
            "curr_share": curr_support_share,
            "delta": support_delta,
            "materially_changed": bool(materially_changed_support),
        },
        "local_bracket_mode_usage": {
            "prev_share": prev_mode_share,
            "curr_share": curr_mode_share,
            "delta": mode_delta,
            "materially_changed": bool(materially_changed_modes),
        },
        "runtime_branch_counts": {
            "deltas": counters_delta,
            "materially_changed": bool(materially_changed_branches),
        },
        "audit_section_diffs": section_diffs,
        "config_changes": {
            "changed_values": changed_config,
            "changed_count": int(len(changed_config)),
            "likely_inert_changes": inert_changed_config,
        },
        "likely_inert_change_flags": {
            "distribution_status": (
                "unchanged" if (not materially_changed_distribution) else "changed"
            ),
            "candidate_counts_status": (
                "unchanged" if (not materially_changed_candidates) else "changed"
            ),
            "decomposition_status": (
                "suspicious_zeroed"
                if decomposition_suspicious_zeroed
                else ("unchanged" if (not materially_changed_decomposition) else "changed")
            ),
            "support_usage_status": (
                "unchanged" if (not materially_changed_support) else "changed"
            ),
            "local_mode_status": (
                "unchanged" if (not materially_changed_modes) else "changed"
            ),
            "branch_counts_status": (
                "unchanged" if (not materially_changed_branches) else "changed"
            ),
            "config_changed_but_no_downstream_effect": bool(len(inert_changed_config) > 0),
            "likely_score_export_bug": bool(decomposition_suspicious_zeroed),
        },
        "availability": {
            "activation_audit": section_diffs["activation_audit"]["status"],
            "config_usage_audit": section_diffs["config_usage_audit"]["status"],
            "bundle_usage_audit": section_diffs["bundle_usage_audit"]["status"],
            "score_path_audit": section_diffs["score_path_audit"]["status"],
            "choice_path_audit": section_diffs["choice_path_audit"]["status"],
            "refined_vs_raw_audit": section_diffs["refined_vs_raw_audit"]["status"],
            "family_competition_health": section_diffs["family_competition_health"]["status"],
            "member_resolution_audit": section_diffs["member_resolution_audit"]["status"],
            "diff_readiness": section_diffs["diff_readiness"]["status"],
        },
    }
    _write_json(Path(args.out), out_payload)

    inert_top = []
    if decomposition_suspicious_zeroed:
        inert_top.append(
            {
                "rank": 1,
                "category": "score_export_bug",
                "reason": "Score decomposition is suspicious_zeroed while scoring branches fired.",
            }
        )
    if float(curr_health.get("top_family_chosen_share", 0.0) or 0.0) >= 0.95:
        inert_top.append(
            {
                "rank": len(inert_top) + 1,
                "category": "family_allocation_concentration",
                "reason": "Top family still dominates chosen distribution.",
            }
        )
    unavailable_sections = [
        section
        for section, section_row in section_diffs.items()
        if str(section_row.get("status", "")) != "available"
    ]
    if unavailable_sections:
        inert_top.append(
            {
                "rank": len(inert_top) + 1,
                "category": "missing_audit_sections",
                "reason": "Some audit sections are unavailable or partially unavailable.",
                "sections": unavailable_sections,
            }
        )
    inert_top = inert_top[:3]
    inert_payload = {
        "created_at": curr_activation.get("created_at"),
        "active_de3_version": str(curr_activation.get("active_de3_version", "v3") or "v3"),
        "did_changes_reach_runtime": bool(
            _safe_float(curr_counters.get("runtime_invocations", 0), 0.0) > 0.0
        ),
        "score_decomposition_status": out_payload["score_decomposition"]["status"],
        "audit_section_availability": out_payload["availability"],
        "likely_inert_changes": inert_changed_config,
        "top_3_likely_bottlenecks": inert_top,
        "source_diff_summary": str(args.out),
    }
    _write_json(Path(args.out_inert), inert_payload)


if __name__ == "__main__":
    main()
