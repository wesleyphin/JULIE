import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_de3_decision_side_model import (
    _apply_cfg_metadata_to_model,
    _apply_model,
    _build_model,
    _default_profiles,
    _points_for_action,
    _prepare_frame,
)


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_file():
        return path
    candidate = ROOT / path
    if candidate.is_file():
        return candidate
    raise SystemExit(f"File not found: {path_arg}")


def _resolve_output_dir(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _baseline_action(row: Dict[str, Any]) -> str:
    baseline = str(row.get("chosen_side", "") or "").strip().lower()
    return baseline if baseline in {"long", "short"} else "no_trade"


def _evaluate_stack_action(
    row: Dict[str, Any],
    trained_models: List[Dict[str, Any]],
) -> Tuple[str, bool, float]:
    baseline_action = _baseline_action(row)
    applied_action = str(baseline_action)
    override_applied = False
    best_side_score = float("-inf")
    for item in trained_models:
        model = item.get("model", {})
        params = item.get("params", {})
        predicted, eval_row = _apply_model(row, model, params)
        side_scores: List[float] = []
        if bool(row.get("long_available", False)):
            side_scores.append(float(eval_row.get("long_score", 0.0) or 0.0))
        if bool(row.get("short_available", False)):
            side_scores.append(float(eval_row.get("short_score", 0.0) or 0.0))
        if side_scores:
            best_side_score = max(best_side_score, max(side_scores))
        if str(model.get("application_mode", "soft_prior") or "soft_prior").strip().lower() != "hard_override":
            continue
        if predicted in {"long", "short", "no_trade"}:
            if predicted != applied_action:
                override_applied = True
            applied_action = str(predicted)
    return str(applied_action), bool(override_applied), float(best_side_score)


def _evaluate_stack(
    tune_df: pd.DataFrame,
    trained_models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for row in tune_df.to_dict("records"):
        baseline_action = _baseline_action(row)
        applied_action, override_applied, best_side_score = _evaluate_stack_action(
            row=row,
            trained_models=trained_models,
        )
        rows.append(
            {
                "timestamp": row.get("timestamp"),
                "baseline_points": float(_points_for_action(row, baseline_action)),
                "applied_points": float(_points_for_action(row, applied_action)),
                "baseline_action": str(baseline_action),
                "applied_action": str(applied_action),
                "override": bool(override_applied),
                "best_side_score": float(best_side_score),
            }
        )
    eval_df = pd.DataFrame(rows)
    if eval_df.empty:
        return {
            "override_count": 0,
            "override_rate": 0.0,
            "baseline_points": 0.0,
            "total_points": 0.0,
            "points_uplift": 0.0,
            "max_drawdown_points": 0.0,
            "objective": 0.0,
        }
    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"], errors="coerce", utc=True)
    eval_df = eval_df.sort_values("timestamp", kind="mergesort")
    equity = eval_df["applied_points"].cumsum()
    max_dd = float((equity.cummax() - equity).max()) if not equity.empty else 0.0
    baseline_points = float(eval_df["baseline_points"].sum())
    total_points = float(eval_df["applied_points"].sum())
    override_count = int(eval_df["override"].fillna(False).astype(bool).sum())
    override_rate = float(override_count / max(1, len(eval_df)))
    objective = float(total_points - (0.12 * max_dd))
    score_mean = (
        float(eval_df.loc[eval_df["override"], "best_side_score"].mean())
        if override_count > 0
        else 0.0
    )
    return {
        "override_count": int(override_count),
        "override_rate": float(override_rate),
        "baseline_points": float(baseline_points),
        "total_points": float(total_points),
        "points_uplift": float(total_points - baseline_points),
        "max_drawdown_points": float(max_dd),
        "objective": float(objective),
        "override_best_side_score_mean": float(score_mean),
    }


def _train_profile(
    *,
    name: str,
    cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    tune_df: pd.DataFrame,
) -> Dict[str, Any]:
    model, model_summary = _build_model(train_df=train_df, cfg=cfg)
    _apply_cfg_metadata_to_model(model=model, cfg=cfg)
    from tools.train_de3_decision_side_model import _evaluate_thresholds

    threshold_eval = _evaluate_thresholds(tune_df=tune_df, model=model, cfg=cfg)
    selected = dict(threshold_eval["selected"])
    model["selected_trade_threshold"] = float(selected.get("trade_threshold", float("inf")) or float("inf"))
    model["selected_side_margin"] = float(selected.get("side_margin", 0.0) or 0.0)
    model["selected_no_trade_threshold"] = float(
        selected.get("no_trade_threshold", float("inf")) or float("inf")
    )
    model["selected_no_trade_margin"] = float(selected.get("no_trade_margin", 0.0) or 0.0)
    model["selected_threshold_source"] = (
        "tune_2024_optimized" if bool(selected.get("override_count", 0)) else "baseline_no_override"
    )
    model["model_name"] = str(name)
    report = {
        "status": "ok",
        "train_rows": int(len(train_df)),
        "tune_rows": int(len(tune_df)),
        "model_summary": dict(model_summary),
        "baseline_tune_metrics": dict(threshold_eval["baseline"]),
        "selected_tune_metrics": dict(selected),
        "threshold_trials": list(threshold_eval["trials"]),
    }
    return {
        "name": str(name),
        "cfg": dict(cfg),
        "model": dict(model),
        "report": dict(report),
        "selected": dict(selected),
        "params": {
            "trade_threshold": float(selected.get("trade_threshold", float("inf")) or float("inf")),
            "side_margin": float(selected.get("side_margin", 0.0) or 0.0),
            "no_trade_threshold": float(
                selected.get("no_trade_threshold", float("inf")) or float("inf")
            ),
            "no_trade_margin": float(selected.get("no_trade_margin", 0.0) or 0.0),
            "min_match_count": 1,
        },
    }


def _default_stack_candidates() -> Dict[str, List[str]]:
    return {
        "decision_action_stack_exact_hard_v1": [
            "decision_side_both_exact_pair_consistent_hard_ex_hour_12",
            "decision_side_long_only_exact_hard_ex_18_21",
            "decision_side_short_only_exact_hard",
        ],
        "decision_action_stack_guarded_short_v1": [
            "decision_side_both_exact_pair_consistent_hard_ex_hour_12",
            "decision_side_long_only_exact_hard_ex_18_21",
            "decision_side_short_only_guarded_hard",
        ],
        "decision_action_stack_guarded_both_v1": [
            "decision_side_both_exact_pair_consistent_hard_ex_hour_12",
            "decision_side_long_only_guarded_hard_ex_18_21",
            "decision_side_short_only_guarded_hard",
        ],
    }


def _parse_stack_specs(raw_specs: List[str]) -> Dict[str, List[str]]:
    if not raw_specs:
        return _default_stack_candidates()
    stacks: Dict[str, List[str]] = {}
    for raw in raw_specs:
        text = str(raw or "").strip()
        if "=" not in text:
            raise SystemExit(f"Invalid --stack-candidate value: {raw}")
        name, members = text.split("=", 1)
        profile_names = [part.strip() for part in members.split(",") if part.strip()]
        if not name.strip() or not profile_names:
            raise SystemExit(f"Invalid --stack-candidate value: {raw}")
        stacks[str(name).strip()] = profile_names
    return stacks


def train_stack_candidates(
    *,
    dataset_path: Path,
    base_bundle_path: Path,
    output_dir: Path,
    stack_candidates: Dict[str, List[str]],
) -> None:
    data = pd.read_csv(dataset_path)
    data = _prepare_frame(data)
    data = data[data["year"].notna()].copy()
    train_df = data[data["year"] <= 2023].copy()
    tune_df = data[data["year"] == 2024].copy()
    if train_df.empty or tune_df.empty:
        raise SystemExit("Decision-side dataset missing train/tune rows.")
    base_bundle = json.loads(base_bundle_path.read_text(encoding="utf-8"))
    profiles = _default_profiles()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "base_bundle_path": str(base_bundle_path),
        "stack_candidates": {},
    }
    trained_profile_cache: Dict[str, Dict[str, Any]] = {}

    def get_trained_profile(profile_name: str) -> Dict[str, Any]:
        if profile_name in trained_profile_cache:
            return trained_profile_cache[profile_name]
        cfg = profiles.get(profile_name)
        if not isinstance(cfg, dict):
            raise SystemExit(f"Unknown decision-side profile: {profile_name}")
        trained = _train_profile(name=profile_name, cfg=cfg, train_df=train_df, tune_df=tune_df)
        trained_profile_cache[profile_name] = trained
        return trained

    for stack_name, profile_names in stack_candidates.items():
        trained_models = [get_trained_profile(name) for name in profile_names]
        stack_eval = _evaluate_stack(tune_df=tune_df, trained_models=trained_models)
        bundle = copy.deepcopy(base_bundle)
        bundle["decision_side_models"] = [copy.deepcopy(item["model"]) for item in trained_models]
        bundle["decision_side_model"] = copy.deepcopy(trained_models[0]["model"])
        bundle["decision_side_training_report"] = {
            "status": "ok",
            "stack_name": str(stack_name),
            "stack_members": [str(item["name"]) for item in trained_models],
            "train_rows": int(len(train_df)),
            "tune_rows": int(len(tune_df)),
            "stack_tune_metrics": dict(stack_eval),
            "member_reports": {str(item["name"]): dict(item["report"]) for item in trained_models},
        }
        meta = bundle.get("metadata", {}) if isinstance(bundle.get("metadata"), dict) else {}
        meta["decision_side_model_retrained_at_utc"] = datetime.now(timezone.utc).isoformat()
        meta["decision_side_model_dataset_path"] = str(dataset_path)
        meta["decision_side_stack_name"] = str(stack_name)
        meta["decision_side_stack_members"] = [str(item["name"]) for item in trained_models]
        bundle["metadata"] = meta
        bundle_path = output_dir / f"dynamic_engine3_v4_bundle.{stack_name}.json"
        report_path = output_dir / f"{stack_name}.decision_side_stack_training_report.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")
        report_path.write_text(
            json.dumps(bundle["decision_side_training_report"], indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        summary["stack_candidates"][str(stack_name)] = {
            "bundle_path": str(bundle_path),
            "report_path": str(report_path),
            "stack_members": [str(item["name"]) for item in trained_models],
            "tune_objective_score": float(stack_eval.get("objective", 0.0)),
            "tune_points_uplift": float(stack_eval.get("points_uplift", 0.0)),
            "tune_override_count": int(stack_eval.get("override_count", 0)),
            "tune_override_rate": float(stack_eval.get("override_rate", 0.0)),
            "tune_max_drawdown_points": float(stack_eval.get("max_drawdown_points", 0.0)),
        }
        print(
            f"{stack_name}: members={profile_names} uplift={stack_eval.get('points_uplift')} "
            f"dd={stack_eval.get('max_drawdown_points')} overrides={stack_eval.get('override_count')}"
        )
    summary_path = output_dir / "candidate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"candidate_summary={summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train stacked DE3 decision-action bundles from the decision-side dataset."
    )
    parser.add_argument(
        "--dataset",
        default="reports/de3_decision_side_dataset_fresh_current_live_2011_2024.csv",
    )
    parser.add_argument("--base-bundle", default="artifacts/de3_v4_live/latest.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--stack-candidate",
        action="append",
        default=[],
        help="Repeatable: NAME=profile_a,profile_b,profile_c",
    )
    args = parser.parse_args()

    train_stack_candidates(
        dataset_path=_resolve_path(str(args.dataset)),
        base_bundle_path=_resolve_path(str(args.base_bundle)),
        output_dir=_resolve_output_dir(str(args.output_dir)),
        stack_candidates=_parse_stack_specs(list(args.stack_candidate or [])),
    )


if __name__ == "__main__":
    main()
