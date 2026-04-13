import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_model_bundle import (  # noqa: E402
    bundle_feature_columns,
    bundle_feature_columns_by_family,
    curated_family_feature_columns,
    make_family_head_bundle,
    normalize_model_bundle,
)


def _resolve_path(path_text: str, default_relative: str = "") -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _load_bundle(path: Path) -> dict:
    with path.open("rb") as fh:
        return normalize_model_bundle(pickle.load(fh))


def _load_threshold_payload(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose a hybrid AetherFlow bundle from a shared live model and one family-native model.")
    parser.add_argument("--shared-model-file", required=True)
    parser.add_argument("--shared-thresholds-file", default="aetherflow_thresholds_deploy_2026oos.json")
    parser.add_argument("--family-model-file", required=True)
    parser.add_argument("--family-thresholds-file", required=True)
    parser.add_argument("--family-name", required=True)
    parser.add_argument("--family-head-weight", type=float, default=1.0)
    parser.add_argument("--conditional-match-session-ids", default=None)
    parser.add_argument("--conditional-match-regimes", default=None)
    parser.add_argument("--family-feature-mode", choices=["all", "curated"], default="curated")
    parser.add_argument("--output-model-file", required=True)
    parser.add_argument("--output-thresholds-file", required=True)
    args = parser.parse_args()

    shared_model_path = _resolve_path(str(args.shared_model_file))
    shared_thresholds_path = _resolve_path(str(args.shared_thresholds_file), str(args.shared_thresholds_file))
    family_model_path = _resolve_path(str(args.family_model_file))
    family_thresholds_path = _resolve_path(str(args.family_thresholds_file))
    output_model_path = _resolve_path(str(args.output_model_file))
    output_thresholds_path = _resolve_path(str(args.output_thresholds_file))
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_thresholds_path.parent.mkdir(parents=True, exist_ok=True)

    family_name = str(args.family_name or "").strip()
    if not family_name:
        raise RuntimeError("family-name is required")

    shared_bundle = _load_bundle(shared_model_path)
    family_bundle = _load_bundle(family_model_path)
    shared_thresholds = _load_threshold_payload(shared_thresholds_path)
    family_thresholds = _load_threshold_payload(family_thresholds_path)

    family_model = family_bundle.get("shared_model", family_bundle.get("model"))
    if family_model is None:
        raise RuntimeError(f"Family bundle has no model: {family_model_path}")

    existing_family_models = dict(shared_bundle.get("family_models", {}) or {})
    existing_family_columns = dict(shared_bundle.get("feature_columns_by_family", {}) or {})
    existing_family_calibrators = dict(shared_bundle.get("family_calibrators", {}) or {})

    family_feature_columns = bundle_feature_columns_by_family(family_bundle).get(family_name)
    if not family_feature_columns:
        family_feature_columns = list(family_bundle.get("shared_feature_columns", []) or family_bundle.get("feature_columns", []) or bundle_feature_columns(family_bundle))
    if not family_feature_columns:
        family_feature_columns = curated_family_feature_columns(family_name)
    existing_family_models[family_name] = family_model
    existing_family_columns[family_name] = list(family_feature_columns)
    family_calibrator = family_bundle.get("shared_calibrator")
    if isinstance(family_calibrator, dict):
        existing_family_calibrators[family_name] = dict(family_calibrator)

    conditional_session_ids = None
    if str(args.conditional_match_session_ids or "").strip():
        conditional_session_ids = {
            int(float(item.strip()))
            for item in str(args.conditional_match_session_ids).split(",")
            if str(item).strip()
        }
    conditional_regimes = None
    if str(args.conditional_match_regimes or "").strip():
        conditional_regimes = {
            str(item).strip().upper()
            for item in str(args.conditional_match_regimes).split(",")
            if str(item).strip()
        }

    if conditional_session_ids or conditional_regimes:
        hybrid_bundle = dict(shared_bundle)
        hybrid_bundle["bundle_design"] = str(shared_bundle.get("bundle_design", "single") or "single")
        hybrid_bundle["shared_model"] = shared_bundle.get("shared_model", shared_bundle.get("model"))
        hybrid_bundle["model"] = shared_bundle.get("shared_model", shared_bundle.get("model"))
        hybrid_bundle["shared_feature_columns"] = list(shared_bundle.get("shared_feature_columns", shared_bundle.get("feature_columns", [])) or bundle_feature_columns(shared_bundle))
        hybrid_bundle["conditional_models"] = list(shared_bundle.get("conditional_models", []) or [])
        hybrid_bundle["conditional_models"].append(
            {
                "family_name": str(family_name),
                "model": family_model,
                "feature_columns": list(family_feature_columns),
                "match_session_ids": sorted(conditional_session_ids) if conditional_session_ids else None,
                "match_regimes": sorted(conditional_regimes) if conditional_regimes else None,
                "weight": float(args.family_head_weight),
                "calibrator": dict(family_calibrator) if isinstance(family_calibrator, dict) else None,
            }
        )
        hybrid_bundle["feature_columns"] = bundle_feature_columns(hybrid_bundle)
        hybrid_bundle["threshold"] = float(shared_bundle.get("threshold", 0.58) or 0.58)
    else:
        hybrid_bundle = make_family_head_bundle(
            shared_model=shared_bundle.get("shared_model", shared_bundle.get("model")),
            family_models=existing_family_models,
            family_feature_columns=existing_family_columns,
            family_head_weight=float(args.family_head_weight),
            threshold=float(shared_bundle.get("threshold", 0.58) or 0.58),
            trained_at=shared_bundle.get("trained_at"),
            walkforward_fold=family_bundle.get("walkforward_fold"),
            family_feature_mode=str(args.family_feature_mode or "curated"),
            shared_calibrator=shared_bundle.get("shared_calibrator"),
            family_calibrators=existing_family_calibrators,
        )
        hybrid_bundle["shared_feature_columns"] = list(shared_bundle.get("shared_feature_columns", shared_bundle.get("feature_columns", [])) or bundle_feature_columns(shared_bundle))
        hybrid_bundle["feature_columns"] = bundle_feature_columns(hybrid_bundle)
        hybrid_bundle["threshold"] = float(shared_bundle.get("threshold", 0.58) or 0.58)

    with output_model_path.open("wb") as fh:
        pickle.dump(hybrid_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    output_thresholds = {
        "threshold": float(shared_thresholds.get("threshold", hybrid_bundle["threshold"])),
        "feature_columns": bundle_feature_columns(hybrid_bundle),
        "feature_columns_by_family": bundle_feature_columns_by_family(hybrid_bundle),
        "bundle_design": str(hybrid_bundle.get("bundle_design", "family_heads")),
        "family_feature_mode": str(args.family_feature_mode or "curated"),
        "family_head_weight": float(args.family_head_weight),
        "family_heads": {
            str(family_name): {
                "model_source": str(family_model_path),
                "thresholds_source": str(family_thresholds_path),
                "conditional_match_session_ids": sorted(conditional_session_ids) if conditional_session_ids else [],
                "conditional_match_regimes": sorted(conditional_regimes) if conditional_regimes else [],
            }
        },
        "shared_model_source": str(shared_model_path),
        "shared_thresholds_source": str(shared_thresholds_path),
        "family_thresholds_payload": family_thresholds,
    }
    output_thresholds_path.write_text(json.dumps(output_thresholds, indent=2), encoding="utf-8")
    print(f"output_model={output_model_path}")
    print(f"output_thresholds={output_thresholds_path}")


if __name__ == "__main__":
    main()
