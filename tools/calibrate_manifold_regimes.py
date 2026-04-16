import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manifold_regime_calibration import (
    apply_calibration_frame,
    build_calibration_payload,
    save_calibration_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Manifold regime calibration and transformed feature parquet.")
    parser.add_argument(
        "--features",
        default="manifold_strategy_features_clean_full.parquet",
        help="Input labeled feature parquet.",
    )
    parser.add_argument(
        "--calibration-out",
        default="manifold_regime_calibration_clean_full.json",
        help="Output calibration JSON.",
    )
    parser.add_argument(
        "--features-out",
        default="manifold_strategy_features_clean_full_robust.parquet",
        help="Output transformed feature parquet.",
    )
    args = parser.parse_args()

    in_path = Path(args.features).expanduser()
    if not in_path.is_absolute():
        in_path = Path.cwd() / in_path
    if not in_path.exists():
        raise SystemExit(f"Missing input parquet: {in_path}")

    out_cal = Path(args.calibration_out).expanduser()
    if not out_cal.is_absolute():
        out_cal = Path.cwd() / out_cal
    out_features = Path(args.features_out).expanduser()
    if not out_features.is_absolute():
        out_features = Path.cwd() / out_features

    df = pd.read_parquet(in_path)
    payload = build_calibration_payload(df, source_path=str(in_path))
    transformed = apply_calibration_frame(df, payload, preserve_existing_side_bias=True)

    save_calibration_payload(payload, out_cal)
    out_features.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_parquet(out_features, index=True)

    counts = (
        pd.to_numeric(transformed.get("manifold_regime_id"), errors="coerce")
        .round()
        .astype("Int64")
        .value_counts(normalize=True)
        .sort_index()
    )
    print(f"saved_calibration={out_cal}")
    print(f"saved_features={out_features}")
    print(f"rows={len(transformed)}")
    print(f"regime_mix={counts.to_dict()}")


if __name__ == "__main__":
    main()
