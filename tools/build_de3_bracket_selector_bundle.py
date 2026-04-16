import argparse
import json
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]


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
        raise SystemExit(f"Expected JSON object at {path}")
    return payload


def _ensure_dict(value: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"Expected object for {label}")
    return value


def _copy_family(
    *,
    selector_families: Dict[str, Any],
    overlay_families: Dict[str, Any],
    family_key: str,
) -> None:
    overlay_family = overlay_families.get(family_key)
    if not isinstance(overlay_family, dict):
        raise SystemExit(f"Overlay family missing: {family_key}")
    selector_families[family_key] = deepcopy(overlay_family)


def _merge_family(
    *,
    selector_families: Dict[str, Any],
    overlay_families: Dict[str, Any],
    family_key: str,
    merge_spec: Dict[str, Any],
) -> None:
    overlay_family = overlay_families.get(family_key)
    if not isinstance(overlay_family, dict):
        raise SystemExit(f"Overlay family missing: {family_key}")

    base_family = selector_families.get(family_key)
    if not isinstance(base_family, dict):
        base_family = {}
    base_family = deepcopy(base_family)
    overlay_family = deepcopy(overlay_family)

    if merge_spec.get("copy_options_from_overlay"):
        base_family["options"] = deepcopy(overlay_family.get("options", []))
    if merge_spec.get("copy_global_default_from_overlay"):
        base_family["global_default"] = deepcopy(overlay_family.get("global_default", {}))

    base_context = base_family.setdefault("context_overrides", {})
    overlay_context = _ensure_dict(overlay_family.get("context_overrides", {}), label=f"{family_key}.context_overrides")
    clear_scopes = {
        str(scope)
        for scope in merge_spec.get("clear_scopes", [])
        if str(scope).strip()
    }
    context_spec = _ensure_dict(merge_spec.get("context_overrides", {}), label=f"{family_key}.merge_spec.context_overrides")
    for scope_name, context_keys in context_spec.items():
        scope = str(scope_name).strip()
        if not scope:
            continue
        overlay_scope = overlay_context.get(scope)
        if not isinstance(overlay_scope, dict):
            raise SystemExit(f"Overlay scope missing for {family_key}: {scope}")
        if scope in clear_scopes:
            base_context[scope] = {}
        base_scope = base_context.setdefault(scope, {})
        if not isinstance(base_scope, dict):
            base_scope = {}
            base_context[scope] = base_scope
        if not isinstance(context_keys, Iterable) or isinstance(context_keys, (str, bytes)):
            raise SystemExit(f"Expected list of context keys for {family_key}.{scope}")
        for raw_context_key in context_keys:
            context_key = str(raw_context_key).strip()
            if not context_key:
                continue
            overlay_entry = overlay_scope.get(context_key)
            if not isinstance(overlay_entry, dict):
                raise SystemExit(f"Overlay context key missing for {family_key}.{scope}: {context_key}")
            base_scope[context_key] = deepcopy(overlay_entry)

    selector_families[family_key] = base_family


def _apply_spec(
    *,
    base_bundle: Dict[str, Any],
    overlay_bundle: Dict[str, Any],
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    merged = deepcopy(base_bundle)
    selector = _ensure_dict(merged.get("family_bracket_selector", {}), label="base.family_bracket_selector")
    overlay_selector = _ensure_dict(overlay_bundle.get("family_bracket_selector", {}), label="overlay.family_bracket_selector")
    selector_families = _ensure_dict(selector.get("families", {}), label="base.family_bracket_selector.families")
    overlay_families = _ensure_dict(overlay_selector.get("families", {}), label="overlay.family_bracket_selector.families")

    for raw_family_key in spec.get("copy_families", []):
        family_key = str(raw_family_key).strip()
        if family_key:
            _copy_family(
                selector_families=selector_families,
                overlay_families=overlay_families,
                family_key=family_key,
            )

    merge_families = _ensure_dict(spec.get("merge_families", {}), label="spec.merge_families")
    for raw_family_key, raw_merge_spec in merge_families.items():
        family_key = str(raw_family_key).strip()
        if not family_key:
            continue
        merge_spec = _ensure_dict(raw_merge_spec, label=f"spec.merge_families[{family_key}]")
        _merge_family(
            selector_families=selector_families,
            overlay_families=overlay_families,
            family_key=family_key,
            merge_spec=merge_spec,
        )

    selector["families"] = selector_families
    merged["family_bracket_selector"] = selector

    metadata_updates = _ensure_dict(spec.get("metadata_updates", {}), label="spec.metadata_updates")
    if metadata_updates:
        metadata = merged.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(deepcopy(metadata_updates))
        merged["metadata"] = metadata

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DE3 bundle by copying or merging family bracket selector overrides.")
    parser.add_argument("--base-bundle", required=True)
    parser.add_argument("--overlay-bundle", required=True)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--output-bundle", required=True)
    args = parser.parse_args()

    base_path = _resolve_path(args.base_bundle, must_exist=True)
    overlay_path = _resolve_path(args.overlay_bundle, must_exist=True)
    spec_path = _resolve_path(args.spec_json, must_exist=True)
    output_path = _resolve_path(args.output_bundle, must_exist=False)

    base_bundle = _load_json_obj(base_path)
    overlay_bundle = _load_json_obj(overlay_path)
    spec = _load_json_obj(spec_path)
    merged = _apply_spec(base_bundle=base_bundle, overlay_bundle=overlay_bundle, spec=spec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
