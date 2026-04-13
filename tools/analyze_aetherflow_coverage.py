import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_model_bundle import predict_bundle_probabilities  # noqa: E402
from aetherflow_strategy import (  # noqa: E402
    AetherFlowStrategy,
    _coerce_session_allowlist,
    _coerce_string_allowlist,
    _coerce_upper_string_allowlist,
    _normalize_family_policies,
)


SESSION_ID_TO_NAME = {
    0: "ASIA",
    1: "LONDON",
    2: "NY_AM",
    3: "NY_PM",
    -1: "OFF",
    -999: "UNKNOWN",
}


def _safe_session_id(value: Any) -> int:
    try:
        out = float(value)
    except Exception:
        return -999
    return int(round(out)) if np.isfinite(out) else -999


def _session_name_from_id(value: Any) -> str:
    session_id = _safe_session_id(value)
    return SESSION_ID_TO_NAME.get(session_id, f"ID_{session_id}")


def _load_policy_override(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Policy override must be a JSON object: {path}")
    return payload


def _apply_policy_override(strategy: AetherFlowStrategy, payload: dict) -> None:
    if "threshold" in payload:
        strategy.threshold = float(payload["threshold"])
    if "min_confidence" in payload:
        strategy.min_confidence = float(payload["min_confidence"])
    if "allowed_session_ids" in payload:
        strategy.allowed_session_ids = _coerce_session_allowlist(payload.get("allowed_session_ids"))
    if "allowed_setup_families" in payload:
        strategy.allowed_setup_families = _coerce_string_allowlist(payload.get("allowed_setup_families"))
    if "hazard_block_regimes" in payload:
        strategy.hazard_block_regimes = _coerce_upper_string_allowlist(payload.get("hazard_block_regimes")) or set()
    if "family_policies" in payload:
        strategy.family_policies = _normalize_family_policies(payload.get("family_policies"))


def _coerce_ts(text: str | None, *, is_end: bool) -> pd.Timestamp | None:
    if not text:
        return None
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    if is_end:
        return ts
    return ts


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _json_safe(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (set, tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _summarize_group(df: pd.DataFrame, group_cols: list[str]) -> list[dict]:
    if df.empty:
        return []
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            rows=("passed", "size"),
            passed=("passed", "sum"),
            total_net_points=("net_points", "sum"),
            avg_net_points=("net_points", "mean"),
            win_rate=("net_points", lambda s: float(np.mean(pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.0))),
            avg_confidence=("aetherflow_confidence", "mean"),
        )
        .reset_index()
    )
    grouped["coverage"] = grouped["passed"] / grouped["rows"].clip(lower=1)
    grouped = grouped.sort_values(["passed", "total_net_points"], ascending=[False, False], kind="mergesort")
    rows = grouped.to_dict("records")
    out: list[dict] = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
                clean[key] = float(value)
            else:
                clean[key] = value
        out.append(clean)
    return out


def _top_reason_counts(df: pd.DataFrame, group_cols: list[str]) -> list[dict]:
    blocked = df.loc[~df["passed"]].copy()
    if blocked.empty:
        return []
    grouped = (
        blocked.groupby(group_cols + ["block_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False, kind="mergesort")
    )
    rows = grouped.to_dict("records")
    out: list[dict] = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            clean[key] = int(value) if isinstance(value, (np.integer,)) else value
        out.append(clean)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AetherFlow coverage, block reasons, and pass-rate by session/family.")
    parser.add_argument("--features-parquet", default="aetherflow_features_fullrange_v2.parquet")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--policy-file", default=None, help="Optional JSON policy override.")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    strategy = AetherFlowStrategy()
    if not strategy.model_loaded or strategy.model is None:
        raise RuntimeError("AetherFlowStrategy model failed to load.")

    if args.policy_file:
        _apply_policy_override(strategy, _load_policy_override(Path(args.policy_file)))

    features_path = ROOT / str(args.features_parquet)
    if not features_path.exists():
        raise RuntimeError(f"Features parquet not found: {features_path}")
    df = pd.read_parquet(features_path)
    if df.empty:
        raise RuntimeError(f"No rows in features parquet: {features_path}")

    start_ts = _coerce_ts(args.start, is_end=False)
    end_ts = _coerce_ts(args.end, is_end=True)
    if start_ts is not None:
        df = df.loc[df.index >= start_ts]
    if end_ts is not None:
        df = df.loc[df.index <= end_ts]
    if df.empty:
        raise RuntimeError("No rows remain after applying the requested window.")

    work = df.copy()
    work["aetherflow_confidence"] = predict_bundle_probabilities(strategy.model_bundle, work)
    work["session_id_int"] = work["session_id"].map(_safe_session_id)
    work["session_name"] = work["session_id_int"].map(_session_name_from_id)
    work["setup_family"] = work["setup_family"].astype(str)

    records = work.to_dict("records")
    reasons: list[str] = []
    for row in records:
        reason = strategy._row_block_reason(row)
        reasons.append(str(reason or ""))
    work["block_reason"] = reasons
    work["passed"] = work["block_reason"].eq("")
    work["net_points"] = pd.to_numeric(work.get("net_points"), errors="coerce").fillna(0.0)

    summary = {
        "window": {
            "start": str(work.index.min()),
            "end": str(work.index.max()),
        },
        "policy": {
            "threshold": float(strategy.threshold),
            "min_confidence": float(strategy.min_confidence),
            "allowed_session_ids": sorted(int(v) for v in strategy.allowed_session_ids) if strategy.allowed_session_ids else [],
            "allowed_setup_families": sorted(str(v) for v in strategy.allowed_setup_families) if strategy.allowed_setup_families else [],
            "hazard_block_regimes": sorted(str(v) for v in strategy.hazard_block_regimes) if strategy.hazard_block_regimes else [],
            "family_policies": _json_safe(strategy.family_policies),
        },
        "totals": {
            "rows": int(len(work)),
            "passed": int(work["passed"].sum()),
            "coverage": float(work["passed"].mean()),
            "passed_total_net_points": float(work.loc[work["passed"], "net_points"].sum()),
            "passed_avg_net_points": float(work.loc[work["passed"], "net_points"].mean()) if work["passed"].any() else 0.0,
            "passed_win_rate": float((work.loc[work["passed"], "net_points"] > 0.0).mean()) if work["passed"].any() else 0.0,
        },
        "by_session": _summarize_group(work, ["session_name"]),
        "by_family": _summarize_group(work, ["setup_family"]),
        "by_session_family": _summarize_group(work, ["session_name", "setup_family"]),
        "block_reasons_overall": _top_reason_counts(work, []),
        "block_reasons_by_session": _top_reason_counts(work, ["session_name"]),
        "block_reasons_by_family": _top_reason_counts(work, ["setup_family"]),
    }

    topn = max(1, int(args.topn))
    print(json.dumps(
        {
            "window": summary["window"],
            "totals": summary["totals"],
            "by_session": summary["by_session"][:topn],
            "by_family": summary["by_family"][:topn],
            "by_session_family": summary["by_session_family"][:topn],
            "block_reasons_overall": summary["block_reasons_overall"][:topn],
            "block_reasons_by_session": summary["block_reasons_by_session"][:topn],
            "block_reasons_by_family": summary["block_reasons_by_family"][:topn],
        },
        indent=2,
    ))

    if args.output_json:
        out_path = ROOT / str(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
        print(f"\nSaved full summary to {out_path}")


if __name__ == "__main__":
    main()
