import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


ROOT = Path(__file__).resolve().parents[1]
LANE_KEYS = (
    "long_rev_variants",
    "short_rev_variants",
    "long_mom_variants",
    "short_mom_variants",
)


def _resolve_path(path_arg: str, *, must_exist: bool) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if must_exist and not path.is_file():
        raise SystemExit(f"File not found: {path_arg}")
    return path


def _load_json_obj(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object payload: {path}")
    return payload


def _load_db_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("strategies") or payload.get("rows") or []
    else:
        rows = []
    if not isinstance(rows, list):
        raise SystemExit(f"Expected list-like strategy rows: {path}")
    return [dict(row) for row in rows if isinstance(row, dict)]


def _bundle_overlay_for_variant_ids(
    *,
    overlay_bundle: Dict[str, Any],
    strategy_ids: Set[str],
) -> Dict[str, Any]:
    selected_variant_ids: Set[str] = set()
    overlay: Dict[str, Any] = {
        "lane_variant_quality": {},
        "bracket_defaults": {},
        "bracket_modes": {},
    }
    for lane_key in LANE_KEYS:
        rows = []
        for row in overlay_bundle.get(lane_key, []):
            if not isinstance(row, dict):
                continue
            variant_id = str(row.get("variant_id", "") or "").strip()
            if variant_id in strategy_ids:
                rows.append(dict(row))
                selected_variant_ids.add(variant_id)
        overlay[lane_key] = rows

    lane_variant_quality = (
        overlay_bundle.get("lane_variant_quality", {})
        if isinstance(overlay_bundle.get("lane_variant_quality"), dict)
        else {}
    )
    bracket_defaults = (
        overlay_bundle.get("bracket_defaults", {})
        if isinstance(overlay_bundle.get("bracket_defaults"), dict)
        else {}
    )
    bracket_modes = (
        overlay_bundle.get("bracket_modes", {})
        if isinstance(overlay_bundle.get("bracket_modes"), dict)
        else {}
    )
    overlay["lane_variant_quality"] = {
        key: dict(value)
        for key, value in lane_variant_quality.items()
        if key in selected_variant_ids and isinstance(value, dict)
    }
    overlay["bracket_defaults"] = {
        key: dict(value)
        for key, value in bracket_defaults.items()
        if key in selected_variant_ids and isinstance(value, dict)
    }
    overlay["bracket_modes"] = {
        key: dict(value)
        for key, value in bracket_modes.items()
        if key in selected_variant_ids and isinstance(value, dict)
    }
    return overlay


def _merge_bundle(
    *,
    baseline_bundle: Dict[str, Any],
    overlay_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    merged = json.loads(json.dumps(baseline_bundle))
    lane_inventory = merged.setdefault("lane_inventory", {})
    if not isinstance(lane_inventory, dict):
        lane_inventory = {}
        merged["lane_inventory"] = lane_inventory
    lane_variant_quality = merged.setdefault("lane_variant_quality", {})
    if not isinstance(lane_variant_quality, dict):
        lane_variant_quality = {}
        merged["lane_variant_quality"] = lane_variant_quality
    bracket_defaults = merged.setdefault("bracket_defaults", {})
    if not isinstance(bracket_defaults, dict):
        bracket_defaults = {}
        merged["bracket_defaults"] = bracket_defaults
    bracket_modes = merged.setdefault("bracket_modes", {})
    if not isinstance(bracket_modes, dict):
        bracket_modes = {}
        merged["bracket_modes"] = bracket_modes

    for lane_key in LANE_KEYS:
        existing = [
            dict(row)
            for row in merged.get(lane_key, [])
            if isinstance(row, dict)
        ]
        by_variant = {
            str(row.get("variant_id", "") or "").strip(): row
            for row in existing
            if str(row.get("variant_id", "") or "").strip()
        }
        for row in overlay_bundle.get(lane_key, []):
            if not isinstance(row, dict):
                continue
            variant_id = str(row.get("variant_id", "") or "").strip()
            lane = str(row.get("lane", "") or "").strip()
            if not variant_id:
                continue
            by_variant[variant_id] = dict(row)
            inv = lane_inventory.get(lane, [])
            if not isinstance(inv, list):
                inv = list(inv) if inv else []
            if variant_id not in inv:
                inv.append(variant_id)
            lane_inventory[lane] = inv
        merged[lane_key] = sorted(
            by_variant.values(),
            key=lambda item: (
                str(item.get("session", "") or ""),
                str(item.get("timeframe", "") or ""),
                str(item.get("strategy_type", "") or ""),
                str(item.get("variant_id", "") or ""),
            ),
        )

    for key, value in overlay_bundle.get("lane_variant_quality", {}).items():
        if isinstance(value, dict):
            lane_variant_quality[str(key)] = dict(value)
    for key, value in overlay_bundle.get("bracket_defaults", {}).items():
        if isinstance(value, dict):
            bracket_defaults[str(key)] = dict(value)
    for key, value in overlay_bundle.get("bracket_modes", {}).items():
        if isinstance(value, dict):
            bracket_modes[str(key)] = dict(value)
    return merged


def _merge_db_rows(
    *,
    baseline_rows: List[Dict[str, Any]],
    overlay_rows: List[Dict[str, Any]],
    strategy_ids: Set[str],
) -> List[Dict[str, Any]]:
    merged = [dict(row) for row in baseline_rows]
    seen = {
        str(row.get("strategy_id", "") or "").strip()
        for row in merged
        if str(row.get("strategy_id", "") or "").strip()
    }
    for row in overlay_rows:
        if not isinstance(row, dict):
            continue
        strategy_id = str(row.get("strategy_id", "") or "").strip()
        if strategy_id not in strategy_ids:
            continue
        if strategy_id in seen:
            continue
        merged.append(dict(row))
        seen.add(strategy_id)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DE3 research hybrid bundle/member DB from selected family strategy IDs."
    )
    parser.add_argument("--baseline-bundle", required=True, help="Baseline DE3 v4 bundle.")
    parser.add_argument("--baseline-db", required=True, help="Baseline DE3 member DB.")
    parser.add_argument("--overlay-bundle", required=True, help="Overlay DE3 v4 bundle.")
    parser.add_argument("--overlay-db", required=True, help="Overlay DE3 member DB.")
    parser.add_argument(
        "--strategy-id",
        action="append",
        default=[],
        help="Variant/strategy ID to graft in. Repeatable.",
    )
    parser.add_argument("--output-bundle", required=True, help="Output hybrid bundle path.")
    parser.add_argument("--output-db", required=True, help="Output hybrid member DB path.")
    args = parser.parse_args()

    strategy_ids = {
        str(item or "").strip()
        for item in (args.strategy_id or [])
        if str(item or "").strip()
    }
    if not strategy_ids:
        raise SystemExit("Provide at least one --strategy-id")

    baseline_bundle = _load_json_obj(_resolve_path(str(args.baseline_bundle), must_exist=True))
    baseline_rows = _load_db_rows(_resolve_path(str(args.baseline_db), must_exist=True))
    overlay_bundle_src = _load_json_obj(_resolve_path(str(args.overlay_bundle), must_exist=True))
    overlay_rows = _load_db_rows(_resolve_path(str(args.overlay_db), must_exist=True))

    overlay_bundle = _bundle_overlay_for_variant_ids(
        overlay_bundle=overlay_bundle_src,
        strategy_ids=strategy_ids,
    )
    matched_bundle_ids = {
        str(row.get("variant_id", "") or "").strip()
        for lane_key in LANE_KEYS
        for row in overlay_bundle.get(lane_key, [])
        if isinstance(row, dict)
    }
    matched_db_ids = {
        str(row.get("strategy_id", "") or "").strip()
        for row in overlay_rows
        if isinstance(row, dict) and str(row.get("strategy_id", "") or "").strip() in strategy_ids
    }
    if not matched_bundle_ids:
        raise SystemExit("No matching variant IDs found in overlay bundle.")
    if not matched_db_ids:
        raise SystemExit("No matching strategy IDs found in overlay DB.")

    merged_bundle = _merge_bundle(
        baseline_bundle=baseline_bundle,
        overlay_bundle=overlay_bundle,
    )
    merged_rows = _merge_db_rows(
        baseline_rows=baseline_rows,
        overlay_rows=overlay_rows,
        strategy_ids=strategy_ids,
    )

    metadata = merged_bundle.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        merged_bundle["metadata"] = metadata
    metadata["research_hybrid_strategy_ids"] = sorted(list(strategy_ids))

    output_bundle = _resolve_path(str(args.output_bundle), must_exist=False)
    output_db = _resolve_path(str(args.output_db), must_exist=False)
    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    output_bundle.write_text(json.dumps(merged_bundle, indent=2, ensure_ascii=True), encoding="utf-8")
    output_db.write_text(json.dumps(merged_rows, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"output_bundle={output_bundle}")
    print(f"output_db={output_db}")
    print(f"matched_bundle_ids={sorted(list(matched_bundle_ids))}")
    print(f"matched_db_ids={sorted(list(matched_db_ids))}")
    print(f"total_db_rows={len(merged_rows)}")


if __name__ == "__main__":
    main()
