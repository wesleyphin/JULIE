#!/usr/bin/env python3
"""Splice a freshly-trained shared model (and optionally a freshly-trained
family-head model) into the deploy bundle's hybrid structure.

The deployed `model_aetherflow_deploy_2026oos.pkl` is a 'single'-design
bundle whose runtime uses `shared_model` for ~97% of predictions and one
`conditional_models[0]` entry (`transition_burst` + session_id=2 +
CHOP_SPIRAL regime, weight 0.03) for the remaining ~3% slice.

This script copies the deployed bundle's structure verbatim but replaces
`shared_model` / `shared_feature_columns` / `threshold` with the
retrained values, and optionally replaces the conditional-model entry
too. Outputs a drop-in replacement pkl + thresholds JSON that the live
strategy loads without config changes.

Usage (Path A — shared only, conditional preserved):
    python3 scripts/splice_aetherflow_retrained_bundle.py \
        --new-shared artifacts/aetherflow_retrain_b/model.pkl \
        --new-thresholds artifacts/aetherflow_retrain_b/thresholds.json

Usage (Path B — replace conditional head too):
    python3 scripts/splice_aetherflow_retrained_bundle.py \
        --new-shared artifacts/aetherflow_retrain_b/model.pkl \
        --new-thresholds artifacts/aetherflow_retrain_b/thresholds.json \
        --new-conditional artifacts/aetherflow_retrain_b/transition_burst_model.pkl

Output files (next to repo root):
    model_aetherflow_deploy_2026full.pkl
    aetherflow_thresholds_deploy_2026full.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEPLOY_PKL = ROOT / "model_aetherflow_deploy_2026oos.pkl"
DEPLOY_THR = ROOT / "aetherflow_thresholds_deploy_2026oos.json"
OUT_PKL = ROOT / "model_aetherflow_deploy_2026full.pkl"
OUT_THR = ROOT / "aetherflow_thresholds_deploy_2026full.json"


def _load_pkl(p: Path):
    with p.open("rb") as fh:
        return pickle.load(fh)


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-shared", required=True,
                    help="Path to the freshly-trained shared model pkl")
    ap.add_argument("--new-thresholds", required=True,
                    help="Path to the freshly-trained thresholds JSON")
    ap.add_argument("--new-conditional", default=None,
                    help="Optional: path to retrained conditional-head pkl "
                         "(same narrow slice as deploy: transition_burst/"
                         "session=2/CHOP_SPIRAL)")
    ap.add_argument("--out-pkl", default=str(OUT_PKL))
    ap.add_argument("--out-thresholds", default=str(OUT_THR))
    args = ap.parse_args()

    deploy = _load_pkl(DEPLOY_PKL)
    new_shared = _load_pkl(Path(args.new_shared))
    new_thr = _load_json(Path(args.new_thresholds))

    # Pull the newly-trained shared model + its feature list
    new_shared_model = (new_shared.get("shared_model")
                        or new_shared.get("model"))
    if new_shared_model is None:
        print(f"ERROR: {args.new_shared} has no 'shared_model' or 'model'")
        return 2
    new_shared_cols = list(new_shared.get("shared_feature_columns")
                           or new_shared.get("feature_columns")
                           or [])
    if not new_shared_cols:
        print(f"ERROR: {args.new_shared} has no feature_columns")
        return 2

    # Start from deploy bundle, replace the shared components
    out = dict(deploy)
    out["shared_model"] = new_shared_model
    out["model"] = new_shared_model
    out["shared_feature_columns"] = new_shared_cols

    # Preserve full 84-feature superset at top-level for downstream reindex —
    # deploy's `feature_columns` was a union of shared + family cols
    deploy_full = list(deploy.get("feature_columns", []))
    out["feature_columns"] = deploy_full if len(deploy_full) >= len(new_shared_cols) else new_shared_cols

    # Threshold: use the newly-trained threshold
    new_threshold = float(new_thr.get("threshold", 0.55))
    out["threshold"] = new_threshold

    # Optionally replace the conditional head (path B)
    if args.new_conditional:
        cond = _load_pkl(Path(args.new_conditional))
        cond_model = cond.get("shared_model") or cond.get("model")
        cond_feature_cols = list(cond.get("shared_feature_columns")
                                 or cond.get("feature_columns")
                                 or [])
        if cond_model is None:
            print(f"ERROR: {args.new_conditional} has no model")
            return 2
        # Preserve deploy's match filter (session=2, CHOP_SPIRAL, weight=0.03)
        original_entry = dict((deploy.get("conditional_models") or [{}])[0])
        original_entry["model"] = cond_model
        original_entry["feature_columns"] = cond_feature_cols
        out["conditional_models"] = [original_entry]
        print(f"  [B] conditional head replaced (match_session_ids="
              f"{original_entry.get('match_session_ids')}, "
              f"match_regimes={original_entry.get('match_regimes')}, "
              f"weight={original_entry.get('weight')})")
    else:
        # Path A — conditional head inherited from deploy unchanged
        print(f"  [A] conditional head preserved from deploy bundle")

    out["trained_at"] = datetime.now(timezone.utc).isoformat()
    out["walkforward_fold"] = f"retrained_full_2024-07_to_2026-04"

    # Write pkl
    Path(args.out_pkl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_pkl, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[write] {args.out_pkl}")

    # Write thresholds JSON mirroring deploy's format
    deploy_thr = _load_json(DEPLOY_THR)
    out_thr = dict(deploy_thr)
    out_thr["threshold"] = new_threshold
    out_thr["packaged_at"] = datetime.now(timezone.utc).isoformat()
    out_thr["source_model_file"] = str(args.new_shared)
    out_thr["source_thresholds_file"] = str(args.new_thresholds)
    out_thr["source_conditional_file"] = str(args.new_conditional or "")
    out_thr["notes"] = ("Retrained on 2024-07 → 2026-04-20 full window "
                        "(Path B). Shared+threshold updated; conditional "
                        "head {}.".format("replaced" if args.new_conditional
                                           else "preserved from deploy"))
    Path(args.out_thresholds).write_text(
        json.dumps(out_thr, indent=2, default=str), encoding="utf-8",
    )
    print(f"[write] {args.out_thresholds}")
    print(f"\nThreshold: {new_threshold}")
    print(f"Feature columns (shared): {len(new_shared_cols)}")
    print(f"Feature columns (union):  {len(out['feature_columns'])}")
    print(f"\nTo activate in live:")
    print(f"  edit config.py AETHERFLOW_STRATEGY:")
    print(f"    thresholds_file: \"{Path(args.out_thresholds).name}\"")
    print(f"    model_file:      \"{Path(args.out_pkl).name}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
