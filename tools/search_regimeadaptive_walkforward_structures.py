import argparse
import copy
import datetime as dt
import json
import math
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

if sys.platform.startswith("win"):
    _platform_machine = str(os.environ.get("PROCESSOR_ARCHITECTURE", "") or "").strip()
    if _platform_machine:
        platform.machine = lambda: _platform_machine  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_regimeadaptive_generalized_artifact import (  # noqa: E402
    GROUP_TEMPLATE_MAP,
    _build_pattern,
    _resolve_gate_model_path,
)
from tools.regimeadaptive_filterless_runner import COMBO_SPACE, NY_TZ, _combo_key_from_id  # noqa: E402
from tools.train_regimeadaptive_gate_walkforward import (  # noqa: E402
    build_arg_parser as build_walkforward_arg_parser,
    run_walkforward_gate_training,
)
from tools.train_regimeadaptive_v2 import _json_safe  # noqa: E402


VALID_POLICIES = {"normal", "reversed"}
VALID_SIDES = {"LONG", "SHORT"}
DEFAULT_TEMPLATE_NAMES = [
    "quarter_day_session",
    "quarter_week_session",
    "week_day_session",
    "day_session",
    "quarter_session",
    "week_session",
]


def _search_root_path(path_text: str) -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / "artifacts/regimeadaptive_walkforward_structure_search")
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
    return slug or "candidate"


def _copy_json(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _parse_templates(raw_items: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw_item in raw_items or []:
        for item in str(raw_item or "").split(","):
            name = str(item or "").strip().lower()
            if not name:
                continue
            if name not in GROUP_TEMPLATE_MAP:
                raise SystemExit(f"Unsupported --template value: {name}")
            if name not in out:
                out.append(name)
    return out or list(DEFAULT_TEMPLATE_NAMES)


def _parse_sessions(raw_items: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw_item in raw_items or []:
        for item in str(raw_item or "").split(","):
            session_name = str(item or "").strip().upper()
            if session_name and session_name not in out:
                out.append(session_name)
    return out


def _normalize_side(value) -> str:
    side = str(value or "").strip().upper()
    return side if side in VALID_SIDES else ""


def _normalize_policy(value) -> str:
    policy = str(value or "").strip().lower()
    return policy if policy in VALID_POLICIES else ""


def _combo_session(combo_key: str) -> str:
    parts = [str(part or "").strip().upper() for part in str(combo_key or "").split("_")]
    if len(parts) < 4:
        return ""
    return "_".join(parts[3:])


def _pattern_matches_combo(pattern: str, combo_key: str) -> bool:
    pattern_parts = [str(part or "").strip().upper() for part in str(pattern or "").split("_")]
    combo_parts = [str(part or "").strip().upper() for part in str(combo_key or "").split("_")]
    if len(pattern_parts) < 4 or len(combo_parts) < 4:
        return False
    combo_session = "_".join(combo_parts[3:])
    pattern_session = "_".join(pattern_parts[3:])
    return (
        (pattern_parts[0] == "ALL" or pattern_parts[0] == combo_parts[0])
        and (pattern_parts[1] == "ALL" or pattern_parts[1] == combo_parts[1])
        and (pattern_parts[2] == "ALL" or pattern_parts[2] == combo_parts[2])
        and (pattern_session == "ALL" or pattern_session == combo_session)
    )


def _pattern_specificity(pattern: str) -> int:
    parts = [str(part or "").strip().upper() for part in str(pattern or "").split("_")]
    if len(parts) < 4:
        return 0
    session_name = "_".join(parts[3:])
    return int(parts[0] != "ALL") + int(parts[1] != "ALL") + int(parts[2] != "ALL") + int(session_name != "ALL")


def _all_combo_keys() -> list[str]:
    return [_combo_key_from_id(combo_idx) for combo_idx in range(COMBO_SPACE)]


def _candidate_record(
    *,
    template_name: str,
    pattern: str,
    side: str,
    policy: str,
    rule_id: str,
    early_exit_enabled,
    support: int,
    selected_count: int,
    selected_share: float,
    rule_share: float,
    policy_support: int,
    coverage_gain: int,
    matched_combo_count: int,
    specificity: int,
    session_name: str,
    support_combos: list[str],
) -> dict:
    heuristic = (
        float(selected_count)
        * float(selected_share)
        * max(1.0, float(specificity))
        * math.log1p(float(max(1, coverage_gain)))
    )
    candidate_id = f"{template_name}:{pattern}:{side}:{policy}:{rule_id or 'default'}"
    return {
        "candidate_id": candidate_id,
        "template_name": str(template_name),
        "pattern": str(pattern),
        "side": str(side),
        "policy": str(policy),
        "rule_id": str(rule_id),
        "early_exit_enabled": early_exit_enabled,
        "support": int(support),
        "selected_count": int(selected_count),
        "selected_share": float(selected_share),
        "rule_share": float(rule_share),
        "policy_support": int(policy_support),
        "coverage_gain": int(coverage_gain),
        "matched_combo_count": int(matched_combo_count),
        "specificity": int(specificity),
        "session_name": str(session_name),
        "support_combos": list(sorted({str(combo) for combo in support_combos})),
        "heuristic": float(heuristic),
    }


def generate_group_candidates(
    payload: dict,
    *,
    coverage_payload: dict | None,
    template_names: list[str],
    include_sessions: list[str],
    min_support: int,
    min_policy_share: float,
    min_rule_share: float,
    min_coverage_gain: int,
    max_matched_combos: int,
) -> list[dict]:
    raw_signal_policies = payload.get("signal_policies", {})
    if not isinstance(raw_signal_policies, dict):
        raise SystemExit("Artifact does not contain a signal_policies mapping.")
    coverage_signal_policies = (
        coverage_payload.get("signal_policies", {})
        if isinstance((coverage_payload or {}).get("signal_policies", {}), dict)
        else {}
    )

    include_session_set = {str(session).upper() for session in include_sessions if str(session).strip()}
    all_combo_keys = _all_combo_keys()
    exact_policy_lookup: dict[tuple[str, str], str] = {}
    for combo_key in all_combo_keys:
        side_map = coverage_signal_policies.get(str(combo_key), {})
        if not isinstance(side_map, dict):
            side_map = {}
        for side in VALID_SIDES:
            record = side_map.get(str(side), {})
            policy = _normalize_policy(record.get("policy")) if isinstance(record, dict) else ""
            exact_policy_lookup[(str(combo_key), str(side))] = policy or "skip"

    grouped_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for combo_key, side_map in raw_signal_policies.items():
        if not isinstance(side_map, dict):
            continue
        combo_session = _combo_session(str(combo_key))
        if include_session_set and combo_session not in include_session_set:
            continue
        for template_name in template_names:
            pattern = _build_pattern(str(combo_key), template_name)
            for side, record in side_map.items():
                side_key = _normalize_side(side)
                if not side_key or not isinstance(record, dict):
                    continue
                policy = _normalize_policy(record.get("policy"))
                if not policy:
                    continue
                grouped_records[(pattern, side_key)].append(
                    {
                        "combo_key": str(combo_key),
                        "policy": str(policy),
                        "rule_id": str(record.get("rule_id", "") or "").strip(),
                        "early_exit_enabled": record.get("early_exit_enabled"),
                        "template_name": str(template_name),
                    }
                )

    candidates: list[dict] = []
    for (pattern, side), records in grouped_records.items():
        policy_counts = Counter(str(record["policy"]) for record in records)
        if not policy_counts:
            continue
        selected_policy, selected_policy_count = policy_counts.most_common(1)[0]
        total_support = int(len(records))
        selected_share = float(selected_policy_count / total_support) if total_support else 0.0
        if total_support < int(min_support) or selected_share + 1e-12 < float(min_policy_share):
            continue

        selected_records = [record for record in records if str(record["policy"]) == str(selected_policy)]
        matched_combos = [combo_key for combo_key in all_combo_keys if _pattern_matches_combo(pattern, combo_key)]
        matched_combo_count = int(len(matched_combos))
        if matched_combo_count <= 0:
            continue
        if int(max_matched_combos) > 0 and matched_combo_count > int(max_matched_combos):
            continue
        coverage_gain = int(
            sum(1 for combo_key in matched_combos if exact_policy_lookup.get((str(combo_key), str(side)), "skip") == "skip")
        )
        if coverage_gain < int(min_coverage_gain):
            continue

        specificity = _pattern_specificity(pattern)
        session_name = _combo_session(pattern)
        rule_counts = Counter(str(record["rule_id"]) for record in selected_records if str(record["rule_id"]).strip())
        if not rule_counts:
            continue
        for rule_id, rule_count in rule_counts.items():
            rule_share = float(rule_count / len(selected_records)) if selected_records else 0.0
            if int(rule_count) < int(min_support) or rule_share + 1e-12 < float(min_rule_share):
                continue
            rule_records = [record for record in selected_records if str(record["rule_id"]) == str(rule_id)]
            early_exit_counts = Counter(
                bool(record["early_exit_enabled"])
                for record in rule_records
                if record.get("early_exit_enabled") is not None
            )
            early_exit_enabled = None
            if early_exit_counts:
                early_exit_enabled = bool(early_exit_counts.most_common(1)[0][0])
            candidates.append(
                _candidate_record(
                    template_name=str(rule_records[0].get("template_name", "")),
                    pattern=str(pattern),
                    side=str(side),
                    policy=str(selected_policy),
                    rule_id=str(rule_id),
                    early_exit_enabled=early_exit_enabled,
                    support=int(rule_count),
                    selected_count=int(rule_count),
                    selected_share=selected_share,
                    rule_share=rule_share,
                    policy_support=int(selected_policy_count),
                    coverage_gain=coverage_gain,
                    matched_combo_count=matched_combo_count,
                    specificity=specificity,
                    session_name=session_name,
                    support_combos=[str(record["combo_key"]) for record in rule_records],
                )
            )

    candidates.sort(
        key=lambda row: (
            -float(row["heuristic"]),
            -int(row["support"]),
            -float(row["selected_share"]),
            -int(row["coverage_gain"]),
            str(row["candidate_id"]),
        )
    )
    return candidates


def _prune_candidates(candidates: list[dict], *, max_candidates: int, max_per_session: int) -> list[dict]:
    chosen: list[dict] = []
    seen_ids: set[str] = set()
    session_counts: Counter[str] = Counter()
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen_ids:
            continue
        session_name = str(candidate.get("session_name", "") or "ALL").upper()
        if int(max_per_session) > 0 and session_counts[session_name] >= int(max_per_session):
            continue
        chosen.append(candidate)
        seen_ids.add(candidate_id)
        session_counts[session_name] += 1
        if int(max_candidates) > 0 and len(chosen) >= int(max_candidates):
            break
    if int(max_candidates) > 0:
        return chosen[: int(max_candidates)]
    return chosen


def _merge_group_candidates(base_payload: dict, candidates: list[dict], *, search_metadata: dict) -> dict:
    payload = _copy_json(base_payload)
    raw_group_signal_policies = payload.get("group_signal_policies", {})
    group_signal_policies = raw_group_signal_policies if isinstance(raw_group_signal_policies, dict) else {}
    payload["group_signal_policies"] = _copy_json(group_signal_policies)
    for candidate in candidates:
        side_map = payload["group_signal_policies"].setdefault(str(candidate["pattern"]), {})
        if not isinstance(side_map, dict):
            side_map = {}
            payload["group_signal_policies"][str(candidate["pattern"])] = side_map
        record = {
            "policy": str(candidate["policy"]),
            "rule_id": str(candidate["rule_id"]),
        }
        if candidate.get("early_exit_enabled") is not None:
            record["early_exit_enabled"] = bool(candidate["early_exit_enabled"])
        side_map[str(candidate["side"])] = record
    payload["group_policy_priority"] = "override_skip"
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["walkforward_structure_search"] = _json_safe(search_metadata)
    payload["metadata"] = metadata
    return payload


def _candidate_root(search_root: Path, label: str) -> Path:
    return search_root / _safe_slug(label)


def _build_eval_args(base_args, *, artifact_path: Path, artifact_root: Path):
    eval_args = copy.deepcopy(base_args)
    eval_args.artifact = str(artifact_path)
    eval_args.artifact_root = str(artifact_root)
    eval_args.write_latest = True
    return eval_args


def _oos_score(result: dict) -> float:
    summary = result.get("oos_summary", {})
    try:
        score = float(summary.get("score"))
    except Exception:
        score = float("-inf")
    return score if math.isfinite(score) else float("-inf")


def _holdout_equity(result: dict) -> float:
    summary = result.get("holdout_summary", {})
    try:
        equity = float(summary.get("equity"))
    except Exception:
        equity = float("nan")
    return equity if math.isfinite(equity) else float("nan")


def _state_summary(state: dict) -> dict:
    result = state.get("result", {})
    oos_summary = result.get("oos_summary", {}) if isinstance(result.get("oos_summary", {}), dict) else {}
    holdout_summary = result.get("holdout_summary", {}) if isinstance(result.get("holdout_summary", {}), dict) else {}
    return {
        "label": str(state.get("label", "")),
        "candidate_ids": list(state.get("candidate_ids", [])),
        "candidate_count": int(len(state.get("candidate_ids", []))),
        "artifact_path": result.get("artifact_path"),
        "walkforward_report_path": result.get("walkforward_report_path"),
        "stable_threshold": result.get("stable_threshold"),
        "stable_session_thresholds": result.get("stable_session_thresholds", {}),
        "oos_summary": {key: _json_safe(value) for key, value in oos_summary.items()},
        "holdout_summary": {key: _json_safe(value) for key, value in holdout_summary.items()},
    }


def evaluate_candidate_state(
    *,
    base_args,
    base_payload: dict,
    source_artifact_path: Path,
    search_root: Path,
    label: str,
    candidates: list[dict],
) -> dict:
    state_root = _candidate_root(search_root, label)
    state_root.mkdir(parents=True, exist_ok=True)
    artifact_path = state_root / "candidate_input.json"
    payload = _merge_group_candidates(
        base_payload,
        candidates,
        search_metadata={
            "source_artifact_path": str(source_artifact_path),
            "candidate_ids": [str(candidate["candidate_id"]) for candidate in candidates],
            "candidate_details": [
                {
                    key: _json_safe(value)
                    for key, value in candidate.items()
                    if key != "support_combos"
                }
                for candidate in candidates
            ],
        },
    )
    artifact_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    _resolve_gate_model_path(source_artifact_path, artifact_path, payload)
    artifact_path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=True), encoding="utf-8")
    result = run_walkforward_gate_training(
        _build_eval_args(
            base_args,
            artifact_path=artifact_path,
            artifact_root=state_root / "walkforward",
        )
    )
    return {
        "label": str(label),
        "candidate_ids": [str(candidate["candidate_id"]) for candidate in candidates],
        "candidates": [
            {key: _json_safe(value) for key, value in candidate.items()}
            for candidate in candidates
        ],
        "artifact_input_path": str(artifact_path),
        "result": result,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        parents=[build_walkforward_arg_parser(add_help=False)],
        description="Search RegimeAdaptive group fallbacks with session-aware walk-forward gate validation.",
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Repeatable or comma-separated candidate template(s). Defaults to a session-aware search set.",
    )
    parser.add_argument(
        "--session-include",
        action="append",
        default=[],
        help="Optional repeatable or comma-separated session filter, e.g. ASIA,LONDON,NY_AM,NY_PM.",
    )
    parser.add_argument(
        "--candidate-source-artifact",
        default="",
        help="Optional artifact used only to mine fallback candidates. Defaults to --artifact.",
    )
    parser.add_argument("--search-root", default="")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-policy-share", type=float, default=0.6)
    parser.add_argument("--min-rule-share", type=float, default=0.5)
    parser.add_argument("--min-coverage-gain", type=int, default=1)
    parser.add_argument("--max-matched-combos", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-per-session", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-oos-score-improvement", type=float, default=100.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    search_root = _search_root_path(str(args.search_root))
    search_root.mkdir(parents=True, exist_ok=True)

    source_artifact_path = Path(str(args.artifact)).expanduser()
    if not source_artifact_path.is_absolute():
        source_artifact_path = (ROOT / source_artifact_path).resolve()
    if not source_artifact_path.is_file():
        raise SystemExit(f"Artifact not found: {source_artifact_path}")

    base_payload = json.loads(source_artifact_path.read_text(encoding="utf-8"))
    candidate_source_path = Path(str(args.candidate_source_artifact or "")).expanduser() if str(args.candidate_source_artifact or "").strip() else source_artifact_path
    if not candidate_source_path.is_absolute():
        candidate_source_path = (ROOT / candidate_source_path).resolve()
    if not candidate_source_path.is_file():
        raise SystemExit(f"Candidate source artifact not found: {candidate_source_path}")
    candidate_source_payload = json.loads(candidate_source_path.read_text(encoding="utf-8"))
    template_names = _parse_templates(list(args.template or []))
    include_sessions = _parse_sessions(list(args.session_include or []))

    candidate_pool_all = generate_group_candidates(
        candidate_source_payload,
        coverage_payload=base_payload,
        template_names=template_names,
        include_sessions=include_sessions,
        min_support=int(args.min_support),
        min_policy_share=float(args.min_policy_share),
        min_rule_share=float(args.min_rule_share),
        min_coverage_gain=int(args.min_coverage_gain),
        max_matched_combos=int(args.max_matched_combos),
    )
    candidate_pool = _prune_candidates(
        candidate_pool_all,
        max_candidates=int(args.max_candidates),
        max_per_session=int(args.max_per_session),
    )
    if not candidate_pool:
        raise SystemExit("No structural candidates met the search filters.")

    print(
        f"candidate_pool total={len(candidate_pool_all)} selected={len(candidate_pool)} "
        f"templates={template_names} sessions={include_sessions or 'ALL'}"
    )

    baseline_state = evaluate_candidate_state(
        base_args=args,
        base_payload=base_payload,
        source_artifact_path=source_artifact_path,
        search_root=search_root,
        label="baseline",
        candidates=[],
    )
    baseline_score = _oos_score(baseline_state["result"])
    print(
        f"baseline oos_score={baseline_score:.2f} "
        f"holdout_equity={_holdout_equity(baseline_state['result']):.2f}"
    )

    evaluation_cache: dict[tuple[str, ...], dict] = {tuple(): baseline_state}
    single_states: list[dict] = []
    for idx, candidate in enumerate(candidate_pool, start=1):
        label = f"single_{idx:02d}_{candidate['pattern']}_{candidate['side']}"
        state = evaluate_candidate_state(
            base_args=args,
            base_payload=base_payload,
            source_artifact_path=source_artifact_path,
            search_root=search_root,
            label=label,
            candidates=[candidate],
        )
        state["oos_score"] = _oos_score(state["result"])
        state["holdout_equity"] = _holdout_equity(state["result"])
        single_states.append(state)
        evaluation_cache[(str(candidate["candidate_id"]),)] = state
        print(
            f"{label} oos_score={state['oos_score']:.2f} "
            f"holdout_equity={state['holdout_equity']:.2f}"
        )

    single_states.sort(
        key=lambda state: (
            -float(state.get("oos_score", float("-inf"))),
            -float(state.get("holdout_equity", float("-inf"))),
            str(state.get("label", "")),
        )
    )
    selection_path = [_state_summary(baseline_state)]
    current_state = baseline_state
    used_candidate_ids: set[str] = set()

    for depth in range(1, max(1, int(args.max_depth)) + 1):
        best_next_state = None
        best_next_score = _oos_score(current_state["result"])
        base_candidate_ids = list(current_state.get("candidate_ids", []))
        for candidate in candidate_pool:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id in used_candidate_ids:
                continue
            candidate_ids = tuple(sorted(base_candidate_ids + [candidate_id]))
            cached_state = evaluation_cache.get(candidate_ids)
            if cached_state is None:
                selected_candidates = [
                    next(item for item in candidate_pool if str(item["candidate_id"]) == selected_id)
                    for selected_id in candidate_ids
                ]
                label = f"depth_{depth}_{'_'.join(_safe_slug(selected_id) for selected_id in candidate_ids)}"
                cached_state = evaluate_candidate_state(
                    base_args=args,
                    base_payload=base_payload,
                    source_artifact_path=source_artifact_path,
                    search_root=search_root,
                    label=label,
                    candidates=selected_candidates,
                )
                evaluation_cache[candidate_ids] = cached_state
            candidate_score = _oos_score(cached_state["result"])
            if candidate_score > best_next_score + float(args.min_oos_score_improvement):
                best_next_state = cached_state
                best_next_score = candidate_score
        if best_next_state is None:
            break
        current_state = best_next_state
        used_candidate_ids = set(current_state.get("candidate_ids", []))
        selection_path.append(_state_summary(current_state))
        print(
            f"selected depth={depth} label={current_state['label']} "
            f"oos_score={_oos_score(current_state['result']):.2f} "
            f"holdout_equity={_holdout_equity(current_state['result']):.2f}"
        )

    leaderboard = [
        _state_summary(state)
        for state in sorted(
            [baseline_state, *single_states, *[state for key, state in evaluation_cache.items() if key]],
            key=lambda state: (
                -_oos_score(state["result"]),
                -_holdout_equity(state["result"]),
                str(state.get("label", "")),
            ),
        )
    ]

    output_payload = {
        "created_at": dt.datetime.now(NY_TZ).isoformat(),
        "source_artifact_path": str(source_artifact_path),
        "candidate_source_artifact_path": str(candidate_source_path),
        "search_root": str(search_root),
        "search_args": {
            "template_names": template_names,
            "include_sessions": include_sessions,
            "min_support": int(args.min_support),
            "min_policy_share": float(args.min_policy_share),
            "min_rule_share": float(args.min_rule_share),
            "min_coverage_gain": int(args.min_coverage_gain),
            "max_matched_combos": int(args.max_matched_combos),
            "max_candidates": int(args.max_candidates),
            "max_per_session": int(args.max_per_session),
            "max_depth": int(args.max_depth),
            "min_oos_score_improvement": float(args.min_oos_score_improvement),
        },
        "candidate_pool_all": [
            {key: _json_safe(value) for key, value in candidate.items()}
            for candidate in candidate_pool_all
        ],
        "candidate_pool_selected": [
            {key: _json_safe(value) for key, value in candidate.items()}
            for candidate in candidate_pool
        ],
        "baseline": _state_summary(baseline_state),
        "selection_path": selection_path,
        "final_selected": _state_summary(current_state),
        "leaderboard": leaderboard,
    }
    output_path = search_root / "search_summary.json"
    output_path.write_text(json.dumps(_json_safe(output_payload), indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"search_summary={output_path}")


if __name__ == "__main__":
    main()
