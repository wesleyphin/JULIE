import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


LANE_KEYS = (
    "long_rev_variants",
    "short_rev_variants",
    "long_mom_variants",
    "short_mom_variants",
)


def _resolve_path(path_arg: str, *, must_exist: bool) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_absolute():
        resolved = path
    else:
        resolved = Path(__file__).resolve().parents[1] / path
    if must_exist and not resolved.is_file():
        raise SystemExit(f"File not found: {path_arg}")
    return resolved


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object at {path}")
    return payload


def _family_overlay_variants(bundle: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for lane_key in LANE_KEYS:
        variants = bundle.get(lane_key, [])
        if not isinstance(variants, list):
            continue
        for raw_row in variants:
            if not isinstance(raw_row, dict):
                continue
            family_tag = str(raw_row.get("family_tag", "") or "").strip()
            family_id = str(raw_row.get("family_id", "") or "").strip()
            if family_tag or "|F" in family_id:
                rows.append((lane_key, dict(raw_row)))
    return rows


def _ensure_variant_list(rows: Any) -> List[Dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _merge_lane_variants(
    *,
    baseline: Dict[str, Any],
    overlay_variants: Iterable[Tuple[str, Dict[str, Any]]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    lane_inventory = baseline.setdefault("lane_inventory", {})
    lane_variant_quality = baseline.setdefault("lane_variant_quality", {})
    bracket_defaults = baseline.setdefault("bracket_defaults", {})
    bracket_modes = baseline.setdefault("bracket_modes", {})

    if not isinstance(lane_inventory, dict):
        lane_inventory = {}
        baseline["lane_inventory"] = lane_inventory
    if not isinstance(lane_variant_quality, dict):
        lane_variant_quality = {}
        baseline["lane_variant_quality"] = lane_variant_quality
    if not isinstance(bracket_defaults, dict):
        bracket_defaults = {}
        baseline["bracket_defaults"] = bracket_defaults
    if not isinstance(bracket_modes, dict):
        bracket_modes = {}
        baseline["bracket_modes"] = bracket_modes

    for lane_key in LANE_KEYS:
        merged_rows = _ensure_variant_list(baseline.get(lane_key, []))
        by_variant_id = {
            str(row.get("variant_id", "") or "").strip(): row
            for row in merged_rows
            if str(row.get("variant_id", "") or "").strip()
        }
        for row_lane_key, row in overlay_variants:
            if row_lane_key != lane_key:
                continue
            variant_id = str(row.get("variant_id", "") or "").strip()
            lane = str(row.get("lane", "") or "").strip()
            if not variant_id:
                continue
            by_variant_id[variant_id] = dict(row)
            inv = lane_inventory.get(lane, [])
            if not isinstance(inv, list):
                inv = list(inv) if inv else []
            if variant_id not in inv:
                inv.append(variant_id)
            lane_inventory[lane] = inv
            quality_row = lane_variant_quality.get(variant_id, {})
            if not isinstance(quality_row, dict) or not quality_row:
                quality_row = {}
            if variant_id not in lane_variant_quality:
                lane_variant_quality[variant_id] = {
                    "variant_id": variant_id,
                    "family_id": str(row.get("family_id", "") or ""),
                    "lane": lane,
                    "standalone_viability_component": float(row.get("standalone_viability_component", 0.0) or 0.0),
                    "incremental_component": float(row.get("incremental_component", 0.0) or 0.0),
                    "orthogonality_component": float(row.get("orthogonality_component", 0.0) or 0.0),
                    "redundancy_penalty": float(row.get("redundancy_penalty", 0.0) or 0.0),
                    "satellite_quality_score": float(
                        row.get("satellite_quality_score", row.get("quality_proxy", 0.0)) or 0.0
                    ),
                    "quality_proxy": float(row.get("quality_proxy", 0.0) or 0.0),
                    "performance_metrics_source": str(row.get("performance_metrics_source", "") or ""),
                }
            if variant_id not in bracket_defaults:
                bracket_defaults[variant_id] = {
                    "sl": float(row.get("best_sl", 0.0) or 0.0),
                    "tp": float(row.get("best_tp", 0.0) or 0.0),
                    "support_trades": int(float(row.get("support_trades", 0) or 0)),
                }
            if variant_id not in bracket_modes:
                bracket_modes[variant_id] = {}
        baseline[lane_key] = sorted(
            by_variant_id.values(),
            key=lambda item: (
                str(item.get("session", "") or ""),
                str(item.get("timeframe", "") or ""),
                str(item.get("strategy_type", "") or ""),
                str(item.get("variant_id", "") or ""),
            ),
        )
        counts[lane_key] = int(len(by_variant_id))
    return counts


def _inject_variant_quality_priors(
    *,
    baseline: Dict[str, Any],
    overlay_variants: Iterable[Tuple[str, Dict[str, Any]]],
) -> int:
    decision_model = baseline.get("decision_policy_model", {})
    if not isinstance(decision_model, dict):
        return 0
    priors = decision_model.get("variant_quality_priors", {})
    if not isinstance(priors, dict):
        priors = {}
        decision_model["variant_quality_priors"] = priors
    injected = 0
    for _lane_key, row in overlay_variants:
        variant_id = str(row.get("variant_id", "") or "").strip()
        if not variant_id:
            continue
        if variant_id in priors:
            continue
        priors[variant_id] = float(
            row.get("satellite_quality_score", row.get("quality_proxy", 0.0)) or 0.0
        )
        injected += 1
    return injected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DE3 hybrid bundle by grafting family-tagged variants onto a baseline bundle."
    )
    parser.add_argument("--baseline", required=True, help="Baseline DE3 v4 bundle JSON.")
    parser.add_argument("--overlay", required=True, help="Overlay DE3 v4 bundle JSON.")
    parser.add_argument("--output", required=True, help="Output hybrid bundle JSON.")
    parser.add_argument(
        "--inject-variant-priors",
        action="store_true",
        help="Add decision_policy_model.variant_quality_priors for grafted family variants.",
    )
    args = parser.parse_args()

    baseline_path = _resolve_path(str(args.baseline), must_exist=True)
    overlay_path = _resolve_path(str(args.overlay), must_exist=True)
    output_path = _resolve_path(str(args.output), must_exist=False)

    baseline = _load_json(baseline_path)
    overlay = _load_json(overlay_path)
    overlay_variants = _family_overlay_variants(overlay)
    if not overlay_variants:
        raise SystemExit(f"No family-tagged overlay variants found in {overlay_path}")

    merged_bundle = dict(baseline)
    lane_counts = _merge_lane_variants(
        baseline=merged_bundle,
        overlay_variants=overlay_variants,
    )
    injected_priors = 0
    if bool(args.inject_variant_priors):
        injected_priors = _inject_variant_quality_priors(
            baseline=merged_bundle,
            overlay_variants=overlay_variants,
        )

    metadata = merged_bundle.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        merged_bundle["metadata"] = metadata
    metadata["hybrid_overlay_baseline_path"] = str(baseline_path)
    metadata["hybrid_overlay_source_path"] = str(overlay_path)
    metadata["hybrid_overlay_family_variant_count"] = int(len(overlay_variants))
    metadata["hybrid_overlay_injected_variant_priors"] = int(injected_priors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_bundle, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"output={output_path}")
    print(f"overlay_family_variants={len(overlay_variants)}")
    print(f"injected_variant_priors={injected_priors}")
    print(f"lane_counts={json.dumps(lane_counts, sort_keys=True)}")


if __name__ == "__main__":
    main()
