import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


VALID_POLICIES = {"normal", "reversed", "skip"}
VALID_SIDES = {"LONG", "SHORT"}
GROUP_TEMPLATE_MAP = {
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
    # Backward-compatible aliases from the first version of this tool.
    "qds": ("{quarter}", "ALL", "{day}", "{session}"),
    "ds": ("ALL", "ALL", "{day}", "{session}"),
}


def _normalize_policy(value) -> str:
    policy = str(value or "skip").strip().lower()
    return policy if policy in VALID_POLICIES else "skip"


def _normalize_side(value) -> str:
    side = str(value or "").strip().upper()
    return side if side in VALID_SIDES else ""


def _parse_sessions(raw_items: list[str] | None) -> set[str]:
    out: set[str] = set()
    for raw_item in raw_items or []:
        for item in str(raw_item or "").split(","):
            session = str(item or "").strip().upper()
            if session:
                out.add(session)
    return out


def _combo_session(combo_key: str) -> str:
    parts = [str(part or "").strip().upper() for part in str(combo_key or "").split("_")]
    if len(parts) < 4:
        return ""
    return "_".join(parts[3:])


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
    if not out:
        raise SystemExit("At least one --template must be provided.")
    return out


def _build_pattern(combo_key: str, template_name: str) -> str:
    parts = [str(part or "").strip().upper() for part in str(combo_key or "").split("_")]
    if len(parts) < 4 or any(not part for part in parts):
        raise ValueError(f"Expected Qx_Wy_DAY_SESSION combo key, got {combo_key!r}")
    quarter, week, day = parts[:3]
    session = "_".join(parts[3:])
    if not session:
        raise ValueError(f"Expected Qx_Wy_DAY_SESSION combo key, got {combo_key!r}")
    template = GROUP_TEMPLATE_MAP.get(str(template_name))
    if template is None:
        raise ValueError(f"Unsupported template {template_name!r}")
    values = {"quarter": quarter, "week": week, "day": day, "session": session}
    return "_".join(segment.format(**values) for segment in template)


def _resolve_gate_model_path(source_artifact: Path, output_artifact: Path, payload: dict) -> None:
    signal_gate = payload.get("signal_gate", {})
    if not isinstance(signal_gate, dict):
        return
    model_path_text = str(signal_gate.get("model_path", "") or "").strip()
    if not model_path_text:
        return
    model_path = Path(model_path_text)
    if not model_path.is_absolute():
        model_path = (source_artifact.parent / model_path).resolve()
    if not model_path.is_file():
        return
    relative_path = os.path.relpath(model_path, output_artifact.parent)
    signal_gate["model_path"] = relative_path


def _copy_json(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def build_generalized_artifact(
    source_artifact: Path,
    output_artifact: Path,
    *,
    template_names: list[str],
    min_count: int,
    min_share: float,
    active_only: bool,
    include_sessions: set[str],
    merge_existing_group_policies: bool,
    group_policy_priority: str,
) -> dict:
    payload = json.loads(source_artifact.read_text(encoding="utf-8"))
    raw_signal_policies = payload.get("signal_policies", {})
    if not isinstance(raw_signal_policies, dict):
        raise SystemExit("Artifact does not contain a signal_policies mapping.")

    grouped_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for combo_key, side_map in raw_signal_policies.items():
        if not isinstance(side_map, dict):
            continue
        session_name = _combo_session(combo_key)
        if include_sessions and session_name not in include_sessions:
            continue
        for template_name in template_names:
            pattern = _build_pattern(combo_key, template_name)
            for side, record in side_map.items():
                side_key = _normalize_side(side)
                if not side_key or not isinstance(record, dict):
                    continue
                grouped_records[(pattern, side_key)].append(
                    {
                        "policy": _normalize_policy(record.get("policy")),
                        "rule_id": str(record.get("rule_id", "") or "").strip(),
                        "early_exit_enabled": record.get("early_exit_enabled"),
                        "template_name": str(template_name),
                    }
                )

    generated_entries = []
    group_signal_policies: dict[str, dict[str, dict]] = {}
    for (pattern, side), records in sorted(grouped_records.items()):
        policy_counts = Counter(record["policy"] for record in records)
        total = int(sum(policy_counts.values()))
        if total < int(min_count):
            continue
        selected_policy, selected_count = policy_counts.most_common(1)[0]
        share = float(selected_count / total) if total else 0.0
        if share + 1e-12 < float(min_share):
            continue
        if bool(active_only) and selected_policy == "skip":
            continue

        selected_records = [record for record in records if record["policy"] == selected_policy]
        output_record = {"policy": selected_policy}

        rule_id_counts = Counter(record["rule_id"] for record in selected_records if record["rule_id"])
        if rule_id_counts:
            output_record["rule_id"] = rule_id_counts.most_common(1)[0][0]

        early_exit_counts = Counter(
            bool(record["early_exit_enabled"])
            for record in selected_records
            if record["early_exit_enabled"] is not None
        )
        if early_exit_counts:
            output_record["early_exit_enabled"] = bool(early_exit_counts.most_common(1)[0][0])

        group_signal_policies.setdefault(pattern, {})[side] = output_record
        generated_entries.append(
            {
                "pattern": pattern,
                "template_names": sorted({str(record.get("template_name", "")) for record in records if record.get("template_name")}),
                "side": side,
                "policy": selected_policy,
                "support": total,
                "selected_count": int(selected_count),
                "selected_share": round(share, 4),
                "policy_counts": dict(sorted(policy_counts.items())),
                "rule_id": output_record.get("rule_id"),
                "early_exit_enabled": output_record.get("early_exit_enabled"),
            }
        )

    output_payload = _copy_json(payload)
    existing_group_signal_policies = {}
    if bool(merge_existing_group_policies) and output_artifact.is_file():
        try:
            existing_payload = json.loads(output_artifact.read_text(encoding="utf-8"))
            existing_group_signal_policies = (
                existing_payload.get("group_signal_policies", {})
                if isinstance(existing_payload.get("group_signal_policies", {}), dict)
                else {}
            )
        except Exception:
            existing_group_signal_policies = {}
    merged_group_signal_policies = _copy_json(existing_group_signal_policies)
    for pattern, side_map in group_signal_policies.items():
        target = merged_group_signal_policies.setdefault(pattern, {})
        if not isinstance(target, dict):
            target = {}
            merged_group_signal_policies[pattern] = target
        for side, record in side_map.items():
            target[str(side)] = dict(record)
    output_payload["group_signal_policies"] = merged_group_signal_policies
    _resolve_gate_model_path(source_artifact, output_artifact, output_payload)

    metadata = output_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["generated_group_policy_overlay"] = {
        "template_names": list(template_names),
        "min_count": int(min_count),
        "min_share": float(min_share),
        "active_only": bool(active_only),
        "include_sessions": sorted(include_sessions),
        "merge_existing_group_policies": bool(merge_existing_group_policies),
        "source_artifact": str(source_artifact.resolve()),
        "generated_group_pattern_count": int(len(generated_entries)),
        "total_group_pattern_count": int(len(merged_group_signal_policies)),
    }
    output_payload["metadata"] = metadata
    output_payload["group_policy_priority"] = str(group_policy_priority)

    output_artifact.parent.mkdir(parents=True, exist_ok=True)
    output_artifact.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return {
        "output_artifact": str(output_artifact.resolve()),
        "generated_group_pattern_count": int(len(generated_entries)),
        "total_group_pattern_count": int(len(merged_group_signal_policies)),
        "generated_entries": generated_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a RegimeAdaptive artifact with wildcard group_signal_policies derived from exact signal_policies."
    )
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Repeatable or comma-separated group template(s), e.g. quarter_day_session,day_session.",
    )
    parser.add_argument("--min-count", type=int, required=True)
    parser.add_argument("--min-share", type=float, required=True)
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Only emit group fallbacks whose selected majority policy is normal or reversed.",
    )
    parser.add_argument(
        "--session-include",
        action="append",
        default=[],
        help="Optional repeatable or comma-separated session filter, e.g. LONDON or NY_AM,NY_PM.",
    )
    parser.add_argument(
        "--merge-existing-group-policies",
        action="store_true",
        help="Merge newly generated group policies into an existing output artifact if it already exists.",
    )
    parser.add_argument(
        "--group-policy-priority",
        choices=("fill_only", "override_skip"),
        default="fill_only",
        help="How group policies interact with exact side records.",
    )
    args = parser.parse_args()

    if int(args.min_count) <= 0:
        raise SystemExit("--min-count must be positive.")
    if not math.isfinite(float(args.min_share)) or not (0.0 < float(args.min_share) <= 1.0):
        raise SystemExit("--min-share must be in (0, 1].")
    template_names = _parse_templates(list(args.template or []))
    include_sessions = _parse_sessions(list(args.session_include or []))

    result = build_generalized_artifact(
        Path(str(args.source_artifact)).expanduser().resolve(),
        Path(str(args.output_artifact)).expanduser().resolve(),
        template_names=template_names,
        min_count=int(args.min_count),
        min_share=float(args.min_share),
        active_only=bool(args.active_only),
        include_sessions=include_sessions,
        merge_existing_group_policies=bool(args.merge_existing_group_policies),
        group_policy_priority=str(args.group_policy_priority),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
