import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_features import ensure_feature_columns  # noqa: E402
from aetherflow_model_bundle import normalize_model_bundle, predict_bundle_probabilities  # noqa: E402


def _resolve_path(path_text: str, default_relative: str = "") -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _fit_logit_linear_calibrator(probabilities: np.ndarray, labels: np.ndarray, *, min_rows: int) -> dict | None:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=np.int8)
    valid_mask = np.isfinite(probs)
    probs = probs[valid_mask]
    y = y[valid_mask]
    if probs.size < int(min_rows) or len(np.unique(y)) < 2:
        return None
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=250,
        random_state=42,
    )
    calibrator.fit(logits, y)
    return {
        "kind": "logit_linear",
        "coef": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y.astype(float))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a shared logit-linear calibrator onto an existing AetherFlow bundle.")
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--thresholds-file", required=True)
    parser.add_argument("--features-parquet", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--mode", choices=["shared", "per_family"], default="shared")
    parser.add_argument("--min-rows", type=int, default=800)
    parser.add_argument("--family-min-rows", type=int, default=None)
    parser.add_argument("--output-model-file", required=True)
    parser.add_argument("--output-thresholds-file", required=True)
    args = parser.parse_args()

    model_path = _resolve_path(args.model_file)
    thresholds_path = _resolve_path(args.thresholds_file)
    features_path = _resolve_path(args.features_parquet)
    output_model_path = _resolve_path(args.output_model_file)
    output_thresholds_path = _resolve_path(args.output_thresholds_file)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_thresholds_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("rb") as fh:
        raw_bundle = pickle.load(fh)
    bundle = normalize_model_bundle(raw_bundle)

    df = pd.read_parquet(features_path)
    df = ensure_feature_columns(df)
    df.index = pd.to_datetime(df.index)
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    if getattr(df.index, "tz", None) is not None:
        tz = df.index.tz
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(tz)
        else:
            start_ts = start_ts.tz_convert(tz)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(tz)
        else:
            end_ts = end_ts.tz_convert(tz)
    frame = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
    frame = frame.loc[pd.to_numeric(frame.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["label"])
    if frame.empty:
        raise RuntimeError("No calibration rows found in requested window.")

    probs = predict_bundle_probabilities(bundle, frame)
    calibrator = _fit_logit_linear_calibrator(
        probs,
        frame["label"].astype(int).to_numpy(dtype=np.int8),
        min_rows=int(args.min_rows),
    )
    if calibrator is None:
        raise RuntimeError("Calibration fit failed or not enough rows.")
    family_calibrators: dict[str, dict] = {}
    if str(args.mode or "shared").strip().lower() == "per_family":
        family_min_rows = int(args.family_min_rows if args.family_min_rows is not None else args.min_rows)
        family_series = frame["setup_family"].astype(str)
        for family_name in sorted(family_series.dropna().unique().tolist()):
            family_mask = family_series.eq(str(family_name)).to_numpy(dtype=bool)
            family_calibrator = _fit_logit_linear_calibrator(
                probs[family_mask],
                frame.loc[family_mask, "label"].astype(int).to_numpy(dtype=np.int8),
                min_rows=family_min_rows,
            )
            if family_calibrator is not None:
                family_calibrators[str(family_name)] = family_calibrator

    if isinstance(raw_bundle, dict):
        raw_bundle = dict(raw_bundle)
        raw_bundle["shared_calibrator"] = dict(calibrator)
        raw_bundle["family_calibrators"] = dict(family_calibrators)
    else:
        raise RuntimeError("Unsupported model bundle format; expected dict-like bundle.")

    with output_model_path.open("wb") as fh:
        pickle.dump(raw_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    thresholds_payload = {}
    if thresholds_path.exists():
        try:
            thresholds_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
            if not isinstance(thresholds_payload, dict):
                thresholds_payload = {}
        except Exception:
            thresholds_payload = {}
    thresholds_payload["shared_calibrator"] = dict(calibrator)
    thresholds_payload["family_calibrators"] = dict(family_calibrators)
    thresholds_payload["calibration_window"] = {
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "mode": str(args.mode),
    }
    output_thresholds_path.write_text(json.dumps(thresholds_payload, indent=2), encoding="utf-8")

    print(f"output_model={output_model_path}")
    print(f"output_thresholds={output_thresholds_path}")
    print(
        json.dumps(
            {
                "shared_calibrator": calibrator,
                "family_calibrators": family_calibrators,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
