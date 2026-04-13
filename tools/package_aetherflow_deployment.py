import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_path(raw: str, default_name: str) -> Path:
    text = str(raw or "").strip()
    path = Path(text).expanduser() if text else (ROOT / default_name)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Package an OOS-validated AetherFlow artifact for live deployment.")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-thresholds", required=True)
    parser.add_argument("--dest-model", default="model_aetherflow_deploy_2026oos.pkl")
    parser.add_argument("--dest-thresholds", default="aetherflow_thresholds_deploy_2026oos.json")
    parser.add_argument("--dest-metrics", default="aetherflow_metrics_deploy_2026oos.json")
    parser.add_argument("--threshold", type=float, default=0.59)
    parser.add_argument("--allow-setup", action="append", default=None)
    parser.add_argument("--block-regime", action="append", default=None)
    parser.add_argument("--policy-name", default="burst_cr")
    parser.add_argument("--source-fold", default="test_2026_2026")
    parser.add_argument("--validation-window-start", default="2026-01-01")
    parser.add_argument("--validation-window-end", default="2026-01-26")
    parser.add_argument("--exact-oos-equity", type=float, default=None)
    parser.add_argument("--exact-oos-trades", type=int, default=None)
    parser.add_argument("--exact-oos-max-dd", type=float, default=None)
    parser.add_argument("--exact-oos-profit-factor", type=float, default=None)
    args = parser.parse_args()

    source_model = _resolve_path(str(args.source_model), "")
    source_thresholds = _resolve_path(str(args.source_thresholds), "")
    dest_model = _resolve_path(str(args.dest_model), "model_aetherflow_deploy_2026oos.pkl")
    dest_thresholds = _resolve_path(str(args.dest_thresholds), "aetherflow_thresholds_deploy_2026oos.json")
    dest_metrics = _resolve_path(str(args.dest_metrics), "aetherflow_metrics_deploy_2026oos.json")

    if not source_model.exists():
        raise SystemExit(f"Source model not found: {source_model}")
    if not source_thresholds.exists():
        raise SystemExit(f"Source thresholds not found: {source_thresholds}")

    payload = json.loads(source_thresholds.read_text())
    feature_columns = payload.get("feature_columns", [])
    if not isinstance(feature_columns, list) or not feature_columns:
        raise SystemExit(f"Feature columns missing in {source_thresholds}")
    feature_columns_by_family = payload.get("feature_columns_by_family", {})
    if not isinstance(feature_columns_by_family, dict):
        feature_columns_by_family = {}
    bundle_design = str(payload.get("bundle_design", "single") or "single")
    family_feature_mode = str(payload.get("family_feature_mode", "all") or "all")
    family_head_weight = float(payload.get("family_head_weight", 1.0) or 1.0)
    family_heads = payload.get("family_heads", {})
    if not isinstance(family_heads, dict):
        family_heads = {}

    allowed_setup_families = [
        str(item).strip()
        for item in (args.allow_setup or ["compression_release", "transition_burst"])
        if str(item).strip()
    ]
    hazard_block_regimes = [
        str(item).strip().upper()
        for item in (args.block_regime or ["ROTATIONAL_TURBULENCE"])
        if str(item).strip()
    ]

    dest_model.parent.mkdir(parents=True, exist_ok=True)
    dest_thresholds.parent.mkdir(parents=True, exist_ok=True)
    dest_metrics.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_model, dest_model)

    threshold_payload = {
        "threshold": float(args.threshold),
        "feature_columns": feature_columns,
        "feature_columns_by_family": feature_columns_by_family,
        "bundle_design": bundle_design,
        "family_feature_mode": family_feature_mode,
        "family_head_weight": family_head_weight,
        "family_heads": family_heads,
        "policy_name": str(args.policy_name),
        "allowed_setup_families": allowed_setup_families,
        "hazard_block_regimes": hazard_block_regimes,
        "source_fold": str(args.source_fold),
        "source_model_file": str(source_model),
        "source_thresholds_file": str(source_thresholds),
        "validation_window": {
            "start": str(args.validation_window_start),
            "end": str(args.validation_window_end),
        },
        "packaged_at": pd.Timestamp.now("UTC").isoformat(),
    }
    dest_thresholds.write_text(json.dumps(threshold_payload, indent=2))

    metrics_payload = {
        "deployment_candidate": True,
        "strategy": "AetherFlowStrategy",
        "policy_name": str(args.policy_name),
        "threshold": float(args.threshold),
        "bundle_design": bundle_design,
        "family_feature_mode": family_feature_mode,
        "family_head_weight": family_head_weight,
        "allowed_setup_families": allowed_setup_families,
        "hazard_block_regimes": hazard_block_regimes,
        "source_fold": str(args.source_fold),
        "source_model_file": str(source_model),
        "source_thresholds_file": str(source_thresholds),
        "source_model_sha256": _sha256(source_model),
        "packaged_model_file": str(dest_model),
        "packaged_thresholds_file": str(dest_thresholds),
        "packaged_at": pd.Timestamp.now("UTC").isoformat(),
        "validation_window": {
            "start": str(args.validation_window_start),
            "end": str(args.validation_window_end),
        },
        "exact_oos": {
            "equity": args.exact_oos_equity,
            "trades": args.exact_oos_trades,
            "max_drawdown": args.exact_oos_max_dd,
            "profit_factor": args.exact_oos_profit_factor,
        },
    }
    dest_metrics.write_text(json.dumps(metrics_payload, indent=2))

    print(f"dest_model={dest_model}")
    print(f"dest_thresholds={dest_thresholds}")
    print(f"dest_metrics={dest_metrics}")


if __name__ == "__main__":
    main()
