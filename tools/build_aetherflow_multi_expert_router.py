import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aetherflow_features import ensure_feature_columns  # noqa: E402
from aetherflow_model_bundle import (  # noqa: E402
    ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS,
    _router_activation_mask,
    build_routed_ensemble_router_frame,
    bundle_feature_columns,
    make_routed_ensemble_bundle,
    normalize_model_bundle,
    predict_bundle_probabilities,
)
from aetherflow_strategy import augment_aetherflow_phase_features  # noqa: E402


def _resolve_path(path_text: str, default_relative: str = "") -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _clip_probs(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def _logloss_per_row(y_true: np.ndarray, prob: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=float)
    p = _clip_probs(prob)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _logloss(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean(_logloss_per_row(y_true, prob))) if len(y_true) else float("nan")


def _brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    return float(np.mean(np.square(p - y))) if len(y_true) else float("nan")


def _auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(prob, dtype=float)))


def _predict_in_batches(bundle: dict[str, Any], features: pd.DataFrame, batch_size: int) -> np.ndarray:
    if features.empty:
        return np.asarray([], dtype=float)
    outputs: list[np.ndarray] = []
    for start in range(0, len(features), max(1, int(batch_size))):
        batch = features.iloc[start : start + int(batch_size)]
        outputs.append(np.asarray(predict_bundle_probabilities(bundle, batch), dtype=float))
    return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=float)


def _split_mask(index: pd.DatetimeIndex, start: str | None, end: str | None) -> np.ndarray:
    mask = np.ones(len(index), dtype=bool)
    if start:
        mask &= index >= pd.Timestamp(start, tz=index.tz)
    if end:
        mask &= index <= pd.Timestamp(end, tz=index.tz)
    return mask


def _metrics_payload(y_true: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(np.asarray(y_true, dtype=float))) if len(y_true) else float("nan"),
        "logloss": _logloss(y_true, prob),
        "brier": _brier(y_true, prob),
        "auc": _auc(y_true, prob),
        "prob_mean": float(np.mean(np.asarray(prob, dtype=float))) if len(y_true) else float("nan"),
    }


def _compute_router_targets(
    *,
    labels: np.ndarray,
    net_points: np.ndarray,
    expert_probabilities: dict[str, np.ndarray],
    expert_names: list[str],
    active_matrix: np.ndarray,
    target_mode: str,
    utility_center_prob: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    loss_matrix = np.full((len(labels), len(expert_names)), np.inf, dtype=float)
    for idx, expert_name in enumerate(expert_names):
        probs = np.asarray(expert_probabilities[expert_name], dtype=float)
        row_loss = _logloss_per_row(labels, probs)
        loss_matrix[:, idx] = np.where(active_matrix[:, idx], row_loss, np.inf)

    mode_key = str(target_mode or "best_expert_logloss").strip().lower()
    diagnostics: dict[str, Any] = {"target_mode": mode_key}
    if mode_key == "best_expert_trade_utility":
        center_prob = float(np.clip(float(utility_center_prob), 0.05, 0.95))
        pnl_scale = np.sqrt(np.maximum(np.abs(np.asarray(net_points, dtype=float)), 0.0))
        pnl_sign = np.sign(np.asarray(net_points, dtype=float))
        utility_matrix = np.full((len(labels), len(expert_names)), -np.inf, dtype=float)
        for idx, expert_name in enumerate(expert_names):
            probs = np.asarray(expert_probabilities[expert_name], dtype=float)
            centered = probs - center_prob
            utility = (pnl_sign * pnl_scale * centered) + (-1e-4 * loss_matrix[:, idx])
            utility_matrix[:, idx] = np.where(active_matrix[:, idx], utility, -np.inf)
        target_index = np.argmax(utility_matrix, axis=1).astype(int)
        sorted_utilities = np.sort(utility_matrix, axis=1)
        best_utility = sorted_utilities[:, -1]
        if len(expert_names) > 1:
            second_utility = sorted_utilities[:, -2]
            second_utility = np.where(np.isfinite(second_utility), second_utility, best_utility)
        else:
            second_utility = best_utility
        sample_weight = np.maximum(best_utility - second_utility, 0.01)
        diagnostics["utility_center_prob"] = center_prob
        diagnostics["target_score_summary"] = {
            "mean_best_utility": float(np.mean(best_utility)) if len(best_utility) else float("nan"),
            "mean_gap": float(np.mean(best_utility - second_utility)) if len(best_utility) else float("nan"),
        }
        return target_index, sample_weight, diagnostics

    target_index = np.argmin(loss_matrix, axis=1).astype(int)
    sorted_losses = np.sort(loss_matrix, axis=1)
    best_loss = sorted_losses[:, 0]
    if len(expert_names) > 1:
        second_loss = sorted_losses[:, 1]
        second_loss = np.where(np.isfinite(second_loss), second_loss, best_loss)
    else:
        second_loss = best_loss
    sample_weight = np.maximum(second_loss - best_loss, 0.01)
    diagnostics["target_score_summary"] = {
        "mean_best_logloss": float(np.mean(best_loss)) if len(best_loss) else float("nan"),
        "mean_gap": float(np.mean(second_loss - best_loss)) if len(best_loss) else float("nan"),
    }
    return target_index, sample_weight, diagnostics


def _fit_router(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    sample_weight: np.ndarray,
    max_depth: int,
    max_iter: int,
    learning_rate: float,
    min_samples_leaf: int,
    random_state: int,
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        l2_regularization=0.05,
        random_state=int(random_state),
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def _load_json_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to load JSON payload: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return payload


def _load_expert_specs(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = _load_json_payload(path)
    fallback_expert = str(payload.get("fallback_expert", "") or "").strip()
    raw_experts = payload.get("experts", []) or []
    if not isinstance(raw_experts, list) or not raw_experts:
        raise RuntimeError(f"No experts found in {path}")
    experts: list[dict[str, Any]] = []
    for item in raw_experts:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        model_file = str(item.get("model_file", "") or "").strip()
        thresholds_file = str(item.get("thresholds_file", "") or "").strip()
        if not name or not model_file or not thresholds_file:
            continue
        activation_rules = item.get("activation_rules", []) or []
        if not isinstance(activation_rules, list):
            activation_rules = []
        override = item.get("override")
        if not isinstance(override, dict):
            override = None
        experts.append(
            {
                "name": name,
                "model_path": _resolve_path(model_file),
                "thresholds_path": _resolve_path(thresholds_file),
                "activation_rules": [dict(rule) for rule in activation_rules if isinstance(rule, dict)],
                "override": dict(override or {}) if isinstance(override, dict) else None,
            }
        )
    if not experts:
        raise RuntimeError(f"No usable experts in {path}")
    if not fallback_expert:
        fallback_expert = str(experts[0]["name"])
    return experts, fallback_expert


def _frame_to_jsonable(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        payload: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                payload[str(key)] = None
            elif isinstance(value, (np.integer,)):
                payload[str(key)] = int(value)
            elif isinstance(value, (np.floating,)):
                payload[str(key)] = float(value)
            else:
                payload[str(key)] = value
        rows.append(payload)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multi-expert AetherFlow routed ensemble artifact.")
    parser.add_argument("--experts-file", required=True)
    parser.add_argument("--features-parquet", required=True)
    parser.add_argument("--output-model-file", required=True)
    parser.add_argument("--output-thresholds-file", required=True)
    parser.add_argument("--output-metrics-file", required=True)
    parser.add_argument("--train-end", default="2024-12-31 23:59")
    parser.add_argument("--validation-end", default="2025-12-31 23:59")
    parser.add_argument("--refit-through-validation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-samples-leaf", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=1337)
    parser.add_argument("--router-mode", choices=["soft_blend", "hard_route"], default="soft_blend")
    parser.add_argument("--router-min-top-prob", type=float, default=0.0)
    parser.add_argument("--router-min-top-gap", type=float, default=0.0)
    parser.add_argument(
        "--router-target-mode",
        choices=["best_expert_logloss", "best_expert_trade_utility"],
        default="best_expert_logloss",
    )
    parser.add_argument("--utility-center-prob", type=float, default=0.55)
    args = parser.parse_args()

    experts_file = _resolve_path(str(args.experts_file))
    features_path = _resolve_path(str(args.features_parquet))
    output_model_path = _resolve_path(str(args.output_model_file))
    output_thresholds_path = _resolve_path(str(args.output_thresholds_file))
    output_metrics_path = _resolve_path(str(args.output_metrics_file))

    expert_specs, fallback_expert = _load_expert_specs(experts_file)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    experts: list[dict[str, Any]] = []
    for spec in expert_specs:
        with spec["model_path"].open("rb") as fh:
            bundle = normalize_model_bundle(pickle.load(fh))
        thresholds = _load_json_payload(spec["thresholds_path"])
        experts.append(
            {
                "name": spec["name"],
                "bundle": bundle,
                "thresholds": thresholds,
                "activation_rules": list(spec.get("activation_rules", []) or []),
                "override": dict(spec.get("override") or {}) if isinstance(spec.get("override"), dict) else None,
                "model_path": spec["model_path"],
                "thresholds_path": spec["thresholds_path"],
            }
        )

    stable_thresholds = dict(experts[0]["thresholds"] or {})
    live_families = [
        str(item).strip()
        for item in (stable_thresholds.get("allowed_setup_families", []) or [])
        if str(item).strip()
    ]
    blocked_regimes = {
        str(item).strip().upper()
        for item in (stable_thresholds.get("hazard_block_regimes", []) or [])
        if str(item).strip()
    }

    data = pd.read_parquet(features_path)
    data = ensure_feature_columns(data)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["label", "net_points"]).copy()
    data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(0).astype(int)
    data = data.sort_index()
    data = augment_aetherflow_phase_features(data)
    if live_families:
        data = data.loc[data["setup_family"].astype(str).isin(live_families)].copy()
    if blocked_regimes and "manifold_regime_name" in data.columns:
        data = data.loc[~data["manifold_regime_name"].astype(str).str.upper().isin(sorted(blocked_regimes))].copy()
    if data.empty:
        raise RuntimeError("No multi-expert training rows remain after live-family filtering.")

    labels = data["label"].to_numpy(dtype=int)
    net_points = pd.to_numeric(data["net_points"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    expert_probabilities: dict[str, np.ndarray] = {}
    active_matrix = np.ones((len(data), len(experts)), dtype=bool)
    for idx, expert in enumerate(experts):
        expert_probabilities[expert["name"]] = _predict_in_batches(expert["bundle"], data, int(args.batch_size))
        rules = list(expert.get("activation_rules", []) or [])
        if rules:
            active_matrix[:, idx] = _router_activation_mask(data, rules)

    expert_names = [expert["name"] for expert in experts]
    target_index, sample_weight, target_diagnostics = _compute_router_targets(
        labels=labels,
        net_points=net_points,
        expert_probabilities=expert_probabilities,
        expert_names=expert_names,
        active_matrix=active_matrix,
        target_mode=str(args.router_target_mode),
        utility_center_prob=float(args.utility_center_prob),
    )

    router_frame = build_routed_ensemble_router_frame(data, expert_probabilities)
    router_feature_columns = [
        col for col in ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS if col in router_frame.columns
    ] or list(router_frame.columns)
    router_frame = router_frame.reindex(columns=router_feature_columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    index = pd.DatetimeIndex(data.index)
    train_mask = _split_mask(index, None, str(args.train_end))
    val_mask = _split_mask(index, str(pd.Timestamp(args.train_end) + pd.Timedelta(minutes=1)), str(args.validation_end))
    post_val_mask = _split_mask(index, str(pd.Timestamp(args.validation_end) + pd.Timedelta(minutes=1)), None)
    jan_2026_mask = _split_mask(index, "2026-01-01", "2026-01-26 23:59")
    fresh_2026_mask = _split_mask(index, "2026-01-27", None)
    april_2026_mask = _split_mask(index, "2026-04-01", None)

    if not bool(train_mask.any()):
        raise RuntimeError("Router train split is empty.")
    if not bool(val_mask.any()):
        raise RuntimeError("Router validation split is empty.")

    router_model = _fit_router(
        router_frame.loc[train_mask],
        target_index[train_mask],
        sample_weight=sample_weight[train_mask],
        max_depth=int(args.max_depth),
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.random_state),
    )

    final_router_model = router_model
    effective_fit_end = str(args.train_end)
    if bool(args.refit_through_validation):
        refit_mask = train_mask | val_mask
        final_router_model = _fit_router(
            router_frame.loc[refit_mask],
            target_index[refit_mask],
            sample_weight=sample_weight[refit_mask],
            max_depth=int(args.max_depth),
            max_iter=int(args.max_iter),
            learning_rate=float(args.learning_rate),
            min_samples_leaf=int(args.min_samples_leaf),
            random_state=int(args.random_state),
        )
        effective_fit_end = str(args.validation_end)

    bundle = make_routed_ensemble_bundle(
        experts=[
            {
                "name": expert["name"],
                "bundle": expert["bundle"],
                "activation_rules": expert.get("activation_rules", []),
                "override": expert.get("override"),
            }
            for expert in experts
        ],
        router_model=final_router_model,
        router_feature_columns=router_feature_columns,
        threshold=float(stable_thresholds.get("threshold", experts[0]["bundle"].get("threshold", 0.54)) or 0.54),
        trained_at=pd.Timestamp.now("UTC").isoformat(),
        walkforward_fold="multi_expert_router_2011_2025",
        router_mode=str(args.router_mode),
        router_weight_floor=0.0,
        router_weight_ceiling=1.0,
        router_training_report={
            "train_end": str(args.train_end),
            "validation_end": str(args.validation_end),
            "effective_router_fit_end": effective_fit_end,
            "target_mode": str(args.router_target_mode),
            "utility_center_prob": float(args.utility_center_prob),
            "target_diagnostics": target_diagnostics,
            "experts_file": str(experts_file),
        },
        router_fallback_expert=fallback_expert,
        router_min_top_prob=float(args.router_min_top_prob),
        router_min_top_gap=float(args.router_min_top_gap),
    )
    normalized_bundle = normalize_model_bundle(bundle)
    routed_prob = _predict_in_batches(normalized_bundle, data, int(args.batch_size))

    with output_model_path.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    thresholds_payload = {
        "threshold": float(stable_thresholds.get("threshold", normalized_bundle.get("threshold", 0.54)) or 0.54),
        "bundle_design": str(normalized_bundle.get("bundle_design", "routed_ensemble") or "routed_ensemble"),
        "feature_columns": bundle_feature_columns(normalized_bundle),
        "router_mode": str(args.router_mode),
        "router_min_top_prob": float(args.router_min_top_prob),
        "router_min_top_gap": float(args.router_min_top_gap),
        "router_target_mode": str(args.router_target_mode),
        "utility_center_prob": float(args.utility_center_prob),
        "router_feature_columns": list(router_feature_columns),
        "router_fallback_expert": str(fallback_expert),
        "allowed_setup_families": live_families,
        "hazard_block_regimes": sorted(blocked_regimes),
        "experts": [
            {
                "name": expert["name"],
                "model_file": str(expert["model_path"]),
                "thresholds_file": str(expert["thresholds_path"]),
                "activation_rules": list(expert.get("activation_rules", []) or []),
                "override": dict(expert.get("override") or {}) if isinstance(expert.get("override"), dict) else None,
                "bundle_design": str(expert["bundle"].get("bundle_design", "single") or "single"),
                "conditional_models": len(expert["bundle"].get("conditional_models", []) or []),
            }
            for expert in experts
        ],
        "packaged_at": pd.Timestamp.now("UTC").isoformat(),
    }
    output_thresholds_path.write_text(json.dumps(thresholds_payload, indent=2), encoding="utf-8")

    split_masks = {
        "train_2011_2024": train_mask,
        "validation_2025": val_mask,
        "holdout_post_validation": post_val_mask,
        "jan_2026": jan_2026_mask,
        "fresh_2026_post_jan26": fresh_2026_mask,
        "april_2026": april_2026_mask,
    }
    split_metrics: dict[str, Any] = {}
    for split_name, mask in split_masks.items():
        if not bool(mask.any()):
            continue
        split_metrics[split_name] = {
            expert["name"]: _metrics_payload(labels[mask], expert_probabilities[expert["name"]][mask])
            for expert in experts
        }
        split_metrics[split_name]["routed"] = _metrics_payload(labels[mask], routed_prob[mask])

    target_shares = {}
    for idx, expert in enumerate(experts):
        target_shares[expert["name"]] = {
            "all": float(np.mean(target_index == idx)),
            "train": float(np.mean(target_index[train_mask] == idx)) if bool(train_mask.any()) else None,
            "validation": float(np.mean(target_index[val_mask] == idx)) if bool(val_mask.any()) else None,
        }

    metrics_payload = {
        "artifact": {
            "model_file": str(output_model_path),
            "thresholds_file": str(output_thresholds_path),
            "features_parquet": str(features_path),
        },
        "training_rows": int(len(data)),
        "training_families": live_families,
        "blocked_regimes": sorted(blocked_regimes),
        "router": {
            "feature_columns": list(router_feature_columns),
            "fit_end": effective_fit_end,
            "fallback_expert": str(fallback_expert),
            "router_mode": str(args.router_mode),
            "router_min_top_prob": float(args.router_min_top_prob),
            "router_min_top_gap": float(args.router_min_top_gap),
            "router_target_mode": str(args.router_target_mode),
            "utility_center_prob": float(args.utility_center_prob),
            "target_shares": target_shares,
            "target_diagnostics": target_diagnostics,
            "params": {
                "max_depth": int(args.max_depth),
                "max_iter": int(args.max_iter),
                "learning_rate": float(args.learning_rate),
                "min_samples_leaf": int(args.min_samples_leaf),
                "random_state": int(args.random_state),
            },
        },
        "split_metrics": split_metrics,
    }
    output_metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    preview = pd.DataFrame(
        [
            {
                "expert": expert["name"],
                "target_share_all": target_shares[expert["name"]]["all"],
                "target_share_train": target_shares[expert["name"]]["train"],
                "target_share_validation": target_shares[expert["name"]]["validation"],
            }
            for expert in experts
        ]
    )
    print(f"output_model={output_model_path}")
    print(f"output_thresholds={output_thresholds_path}")
    print(f"output_metrics={output_metrics_path}")
    print("target_share_summary=" + json.dumps(_frame_to_jsonable(preview), indent=2))


if __name__ == "__main__":
    main()
