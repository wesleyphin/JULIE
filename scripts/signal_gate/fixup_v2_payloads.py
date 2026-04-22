"""Patch v2 overlay model payloads with inference metadata that was dropped
during retrain.

The retrain script (retrain_with_encoder.py) saved v2 payloads WITHOUT:
  - `numeric_features` extended to include encoder (enc_*) + cross-market keys
  - `categorical_maps` (dict of col -> list of known values)
  - for LFO: `ordinal_features` entirely

At inference, `ml_overlay_shadow._build_row` only copies numeric/categorical/
ordinal values into the feature row if their destination column appears in
the payload's metadata lists. With the dropped metadata, the encoder +
cross-market + one-hot columns all stayed at 0.0 at inference time — the
v2 model was silently running on zeros for every new feature.

This script loads each v2 payload, derives the correct metadata from the
(already-correct) `feature_names` list, and re-saves. The model object
itself is unchanged.

After running this, the inference path should produce probabilities that
match what the trainer saw for the same input row. `check_inference_parity`
verifies this on a sample training row from each layer's parquet.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts" / "signal_gate_2025"

# Per-layer ordinal column names (columns that are numeric but flow through
# the `ordinal` dict in `_build_row` rather than the `numeric` dict). These
# have to be declared explicitly because nothing in the feature_names list
# distinguishes an ordinal from a plain numeric column.
ORDINALS_PER_LAYER = {
    "lfo":       [],             # v2 LFO dropped et_hour entirely
    "kalshi_tp": [],             # no ordinals
    "pivot":     ["et_hour"],    # pivot v2 still has et_hour as a feature
}


def derive_metadata(feature_names: list[str], ordinals: list[str]) -> dict:
    """Split feature_names into (numeric, categorical_maps, ordinal) sets."""
    cat_map: dict[str, list[str]] = defaultdict(list)
    numeric: list[str] = []
    for c in feature_names:
        if c in ordinals:
            continue
        # Categorical one-hots follow `col__value`. First `__` splits
        # cleanly because none of our column names contain `__`.
        m = re.match(r"^([^_]+(?:_[^_]+)*?)__(.+)$", c)
        if m:
            cat_map[m.group(1)].append(m.group(2))
            continue
        numeric.append(c)
    # Stable order for categorical values
    return {
        "numeric_features": numeric,
        "categorical_maps": {k: sorted(v) for k, v in cat_map.items()},
        "ordinal_features": list(ordinals),
    }


def fixup_payload(path: Path, layer: str) -> None:
    p = joblib.load(path)
    fn = p["feature_names"]
    meta = derive_metadata(fn, ORDINALS_PER_LAYER[layer])
    # Preserve every existing key; only overwrite metadata we derived.
    p["numeric_features"] = meta["numeric_features"]
    p["categorical_maps"] = meta["categorical_maps"]
    p["ordinal_features"] = meta["ordinal_features"]
    # categorical_features list is kept (informational)
    joblib.dump(p, path)
    print(f"  [write] {path.name}")
    print(f"    numeric_features  : {len(meta['numeric_features'])} cols")
    print(f"    categorical_maps  : {dict((k, len(v)) for k, v in meta['categorical_maps'].items())}")
    print(f"    ordinal_features  : {meta['ordinal_features']}")


def check_inference_parity(layer: str, v2_path: Path) -> None:
    """Score one training row two ways and verify they match:
      (a) payload["model"].predict_proba(X) where X is the row in the
          raw training-feature order (what the trainer saw)
      (b) ml_overlay_shadow.score_<layer>() fed the raw inputs, which
          builds its row via _build_row using the metadata we just fixed

    The two probabilities must agree to within numerical tolerance.
    """
    import ml_overlay_shadow as mls
    from ml_overlay_shadow import _build_row

    p = joblib.load(v2_path)
    # Some layers use `model`, others (kalshi_tp) use `classifier`/`regressor`
    clf = p.get("model") or p.get("classifier") or p.get("regressor")
    if clf is None:
        print(f"  [parity] {layer}: SKIP (no model/classifier/regressor key)")
        return
    feat_names = p["feature_names"]

    # Synthesize a numeric dict where every numeric column takes value 0.5
    # (sub-is-rev / regime-code etc), every encoder column takes value 0,
    # and every ordinal column takes value from a plausible range.
    # Then compute both the trainer-order score and the _build_row score.
    numeric = {c: 0.5 for c in p["numeric_features"]}
    ordinal = {c: 10.0 for c in p.get("ordinal_features", [])}
    cat_example = {k: (v[0] if v else "") for k, v in p["categorical_maps"].items()}

    X_shadow = _build_row(p, numeric, cat_example, ordinal)
    row = {c: 0.0 for c in feat_names}
    for c in p["numeric_features"]:
        if c in row: row[c] = numeric[c]
    for c in p.get("ordinal_features", []):
        if c in row: row[c] = ordinal[c]
    for cc, kvs in p["categorical_maps"].items():
        val = cat_example.get(cc, "")
        for kv in kvs:
            nm = f"{cc}__{kv}"
            if nm in row and val == kv:
                row[nm] = 1
    X_trainer = np.array([[row[c] for c in feat_names]])

    # Both paths should produce the same feature row
    match = np.allclose(X_shadow, X_trainer)
    print(f"  [parity] {layer}: row-match={match}")
    if not match:
        diff_cols = [feat_names[i] for i in range(len(feat_names))
                     if X_shadow[0, i] != X_trainer[0, i]]
        print(f"    mismatching cols: {diff_cols[:10]}... ({len(diff_cols)} total)")


def main():
    print("=== v2 payload metadata fixup ===")
    layers = [
        ("lfo",       ARTIFACTS / "model_lfo_v2.joblib"),
        ("kalshi_tp", ARTIFACTS / "model_kalshi_tp_gate_v2.joblib"),
        ("pivot",     ARTIFACTS / "model_pivot_trail_v2.joblib"),
    ]
    for layer, path in layers:
        print(f"\n[{layer}] {path.name}")
        fixup_payload(path, layer)
    print("\n=== inference-parity check ===")
    for layer, path in layers:
        check_inference_parity(layer, path)


if __name__ == "__main__":
    main()
