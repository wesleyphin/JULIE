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


def _load_bundle(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return normalize_model_bundle(pickle.load(fh))


def _load_threshold_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_activation_rules(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw_rules = payload.get("rules", payload if isinstance(payload, list) else [])
    if not isinstance(raw_rules, list):
        return []
    return [dict(item) for item in raw_rules if isinstance(item, dict)]


def _allowed_setup_families(payload: dict[str, Any]) -> list[str]:
    return [
        str(item).strip()
        for item in (payload.get("allowed_setup_families", []) or [])
        if str(item).strip()
    ]


def _blocked_regimes(payload: dict[str, Any]) -> list[str]:
    return [
        str(item).strip().upper()
        for item in (payload.get("hazard_block_regimes", []) or [])
        if str(item).strip()
    ]


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


def _blend_with_adaptive_weight(stable_prob: np.ndarray, adaptive_prob: np.ndarray, adaptive_weight: np.ndarray) -> np.ndarray:
    w = np.asarray(adaptive_weight, dtype=float)
    stable = np.asarray(stable_prob, dtype=float)
    adaptive = np.asarray(adaptive_prob, dtype=float)
    return ((1.0 - w) * stable) + (w * adaptive)


def _candidate_weight_grid() -> list[tuple[float, float]]:
    floors = [0.05, 0.10, 0.15, 0.20]
    ceilings = [0.80, 0.85, 0.90, 0.95]
    out: list[tuple[float, float]] = []
    for floor in floors:
        for ceiling in ceilings:
            if ceiling > floor:
                out.append((float(floor), float(ceiling)))
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a routed two-expert AetherFlow ensemble artifact.")
    parser.add_argument("--stable-model-file", required=True)
    parser.add_argument("--stable-thresholds-file", required=True)
    parser.add_argument("--adaptive-model-file", required=True)
    parser.add_argument("--adaptive-thresholds-file", required=True)
    parser.add_argument("--features-parquet", required=True)
    parser.add_argument("--output-model-file", required=True)
    parser.add_argument("--output-thresholds-file", required=True)
    parser.add_argument("--output-metrics-file", required=True)
    parser.add_argument("--activation-rules-file", default=None)
    parser.add_argument("--train-end", default="2024-12-31 23:59")
    parser.add_argument("--validation-end", default="2025-12-31 23:59")
    parser.add_argument("--refit-through-validation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-samples-leaf", type=int, default=256)
    parser.add_argument("--random-state", type=int, default=1337)
    args = parser.parse_args()

    stable_model_path = _resolve_path(str(args.stable_model_file))
    stable_thresholds_path = _resolve_path(str(args.stable_thresholds_file))
    adaptive_model_path = _resolve_path(str(args.adaptive_model_file))
    adaptive_thresholds_path = _resolve_path(str(args.adaptive_thresholds_file))
    features_path = _resolve_path(str(args.features_parquet))
    output_model_path = _resolve_path(str(args.output_model_file))
    output_thresholds_path = _resolve_path(str(args.output_thresholds_file))
    output_metrics_path = _resolve_path(str(args.output_metrics_file))
    activation_rules_path = _resolve_path(str(args.activation_rules_file)) if args.activation_rules_file else None

    stable_bundle = _load_bundle(stable_model_path)
    adaptive_bundle = _load_bundle(adaptive_model_path)
    stable_thresholds = _load_threshold_payload(stable_thresholds_path)
    adaptive_thresholds = _load_threshold_payload(adaptive_thresholds_path)
    activation_rules = _load_activation_rules(activation_rules_path)

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(features_path)
    data = ensure_feature_columns(data)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["label", "net_points"]).copy()
    data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(0).astype(int)
    data = data.sort_index()
    data = augment_aetherflow_phase_features(data)

    live_families = _allowed_setup_families(stable_thresholds)
    if live_families:
        data = data.loc[data["setup_family"].astype(str).isin(live_families)].copy()
    blocked_regimes = set(_blocked_regimes(stable_thresholds))
    if blocked_regimes and "manifold_regime_name" in data.columns:
        data = data.loc[~data["manifold_regime_name"].astype(str).str.upper().isin(sorted(blocked_regimes))].copy()
    if data.empty:
        raise RuntimeError("No routed-ensemble training rows remain after live-family filtering.")

    stable_prob = _predict_in_batches(stable_bundle, data, int(args.batch_size))
    adaptive_prob = _predict_in_batches(adaptive_bundle, data, int(args.batch_size))
    labels = data["label"].to_numpy(dtype=int)

    stable_loss = _logloss_per_row(labels, stable_prob)
    adaptive_loss = _logloss_per_row(labels, adaptive_prob)
    loss_gap = stable_loss - adaptive_loss
    router_target = (loss_gap > 0.0).astype(int)
    router_weight = np.maximum(np.abs(loss_gap), 0.01)

    router_frame = build_routed_ensemble_router_frame(
        data,
        {
            "stable": stable_prob,
            "adaptive": adaptive_prob,
        },
    )
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

    if not bool(train_mask.any()):
        raise RuntimeError("Router train split is empty.")
    if not bool(val_mask.any()):
        raise RuntimeError("Router validation split is empty.")

    router_model = _fit_router(
        router_frame.loc[train_mask],
        router_target[train_mask],
        sample_weight=router_weight[train_mask],
        max_depth=int(args.max_depth),
        max_iter=int(args.max_iter),
        learning_rate=float(args.learning_rate),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.random_state),
    )
    router_prob_val = np.asarray(router_model.predict_proba(router_frame.loc[val_mask])[:, 1], dtype=float)

    weight_rows: list[dict[str, Any]] = []
    best_floor = 0.05
    best_ceiling = 0.95
    best_logloss = math.inf
    for floor, ceiling in _candidate_weight_grid():
        adaptive_weight = floor + ((ceiling - floor) * router_prob_val)
        blended_val = _blend_with_adaptive_weight(
            stable_prob[val_mask],
            adaptive_prob[val_mask],
            adaptive_weight,
        )
        score = _logloss(labels[val_mask], blended_val)
        weight_rows.append(
            {
                "floor": float(floor),
                "ceiling": float(ceiling),
                "validation_logloss": float(score),
                "validation_brier": _brier(labels[val_mask], blended_val),
                "validation_auc": _auc(labels[val_mask], blended_val),
                "validation_prob_mean": float(np.mean(blended_val)),
            }
        )
        if score < best_logloss:
            best_logloss = float(score)
            best_floor = float(floor)
            best_ceiling = float(ceiling)

    final_router_model = router_model
    final_train_end = str(args.train_end)
    if bool(args.refit_through_validation):
        refit_mask = train_mask | val_mask
        final_router_model = _fit_router(
            router_frame.loc[refit_mask],
            router_target[refit_mask],
            sample_weight=router_weight[refit_mask],
            max_depth=int(args.max_depth),
            max_iter=int(args.max_iter),
            learning_rate=float(args.learning_rate),
            min_samples_leaf=int(args.min_samples_leaf),
            random_state=int(args.random_state),
        )
        final_train_end = str(args.validation_end)

    bundle = make_routed_ensemble_bundle(
        experts=[
            {"name": "stable", "bundle": stable_bundle},
            {"name": "adaptive", "bundle": adaptive_bundle},
        ],
        router_model=final_router_model,
        router_feature_columns=router_feature_columns,
        threshold=float(stable_thresholds.get("threshold", stable_bundle.get("threshold", 0.54)) or 0.54),
        trained_at=pd.Timestamp.now("UTC").isoformat(),
        walkforward_fold="routed_ensemble_2011_2025",
        router_mode="soft_blend",
        router_weight_floor=float(best_floor),
        router_weight_ceiling=float(best_ceiling),
        router_training_report={
            "train_end": str(args.train_end),
            "validation_end": str(args.validation_end),
            "effective_router_fit_end": final_train_end,
            "router_target": "adaptive_better_logloss",
            "router_positive_rate_train": float(np.mean(router_target[train_mask])) if bool(train_mask.any()) else None,
            "router_positive_rate_validation": float(np.mean(router_target[val_mask])) if bool(val_mask.any()) else None,
            "weight_grid_results": weight_rows,
            "selected_floor": float(best_floor),
            "selected_ceiling": float(best_ceiling),
            "stable_model_file": str(stable_model_path),
            "adaptive_model_file": str(adaptive_model_path),
        },
        router_activation_rules=activation_rules,
        router_fallback_expert="stable",
    )
    normalized_bundle = normalize_model_bundle(bundle)
    full_router_prob = np.asarray(final_router_model.predict_proba(router_frame)[:, 1], dtype=float)
    routed_prob = _predict_in_batches(normalized_bundle, data, int(args.batch_size))
    adaptive_weight = np.zeros(len(data), dtype=float)
    denom = np.asarray(adaptive_prob, dtype=float) - np.asarray(stable_prob, dtype=float)
    valid = np.abs(denom) > 1e-9
    adaptive_weight[valid] = (np.asarray(routed_prob, dtype=float)[valid] - np.asarray(stable_prob, dtype=float)[valid]) / denom[valid]
    adaptive_weight = np.clip(adaptive_weight, 0.0, 1.0)

    with output_model_path.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    thresholds_payload = {
        "threshold": float(stable_thresholds.get("threshold", normalized_bundle.get("threshold", 0.54)) or 0.54),
        "bundle_design": str(normalized_bundle.get("bundle_design", "routed_ensemble") or "routed_ensemble"),
        "feature_columns": bundle_feature_columns(normalized_bundle),
        "router_mode": "soft_blend",
        "router_feature_columns": list(router_feature_columns),
        "router_weight_floor": float(best_floor),
        "router_weight_ceiling": float(best_ceiling),
        "router_activation_rules": activation_rules,
        "router_fallback_expert": "stable",
        "allowed_setup_families": live_families,
        "hazard_block_regimes": sorted(blocked_regimes),
        "experts": [
            {
                "name": "stable",
                "model_file": str(stable_model_path),
                "thresholds_file": str(stable_thresholds_path),
                "bundle_design": str(stable_bundle.get("bundle_design", "single") or "single"),
                "conditional_models": len(stable_bundle.get("conditional_models", []) or []),
            },
            {
                "name": "adaptive",
                "model_file": str(adaptive_model_path),
                "thresholds_file": str(adaptive_thresholds_path),
                "bundle_design": str(adaptive_bundle.get("bundle_design", "single") or "single"),
                "conditional_models": len(adaptive_bundle.get("conditional_models", []) or []),
            },
        ],
        "source_thresholds_payload": {
            "stable": stable_thresholds,
            "adaptive": adaptive_thresholds,
        },
        "packaged_at": pd.Timestamp.now("UTC").isoformat(),
    }
    output_thresholds_path.write_text(json.dumps(thresholds_payload, indent=2), encoding="utf-8")

    split_masks = {
        "train_2011_2024": train_mask,
        "validation_2025": val_mask,
        "holdout_post_validation": post_val_mask,
        "jan_2026": jan_2026_mask,
        "fresh_2026_post_jan26": fresh_2026_mask,
    }
    split_metrics: dict[str, Any] = {}
    for name, mask in split_masks.items():
        if not bool(mask.any()):
            continue
        split_metrics[name] = {
            "stable": _metrics_payload(labels[mask], stable_prob[mask]),
            "adaptive": _metrics_payload(labels[mask], adaptive_prob[mask]),
            "routed": _metrics_payload(labels[mask], routed_prob[mask]),
            "adaptive_weight_mean": float(np.mean(adaptive_weight[mask])),
            "adaptive_weight_median": float(np.median(adaptive_weight[mask])),
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
            "selected_floor": float(best_floor),
            "selected_ceiling": float(best_ceiling),
            "fit_end": final_train_end,
            "activation_rules_file": str(activation_rules_path) if activation_rules_path is not None else None,
            "activation_rules": activation_rules,
            "params": {
                "max_depth": int(args.max_depth),
                "max_iter": int(args.max_iter),
                "learning_rate": float(args.learning_rate),
                "min_samples_leaf": int(args.min_samples_leaf),
                "random_state": int(args.random_state),
            },
            "router_train_positive_rate": float(np.mean(router_target[train_mask])) if bool(train_mask.any()) else None,
            "router_validation_positive_rate": float(np.mean(router_target[val_mask])) if bool(val_mask.any()) else None,
            "weight_grid_results": weight_rows,
        },
        "split_metrics": split_metrics,
        "score_distribution": {
            "stable_mean": float(np.mean(stable_prob)),
            "adaptive_mean": float(np.mean(adaptive_prob)),
            "routed_mean": float(np.mean(routed_prob)),
            "adaptive_weight_mean": float(np.mean(adaptive_weight)),
            "adaptive_weight_q10": float(np.quantile(adaptive_weight, 0.10)),
            "adaptive_weight_q50": float(np.quantile(adaptive_weight, 0.50)),
            "adaptive_weight_q90": float(np.quantile(adaptive_weight, 0.90)),
        },
        "expert_gap_summary": {
            "adaptive_better_share_all": float(np.mean(router_target)),
            "adaptive_better_share_abs_gap_ge_005": float(np.mean(np.abs(loss_gap) >= 0.05)),
            "stable_logloss_all": _logloss(labels, stable_prob),
            "adaptive_logloss_all": _logloss(labels, adaptive_prob),
            "routed_logloss_all": _logloss(labels, routed_prob),
        },
    }
    output_metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    preview = pd.DataFrame(weight_rows).sort_values(["validation_logloss", "validation_brier"], ascending=[True, True]).head(8)
    print(f"output_model={output_model_path}")
    print(f"output_thresholds={output_thresholds_path}")
    print(f"output_metrics={output_metrics_path}")
    print(f"selected_floor={best_floor:.3f}")
    print(f"selected_ceiling={best_ceiling:.3f}")
    print("top_weight_grid=" + json.dumps(_frame_to_jsonable(preview), indent=2))


if __name__ == "__main__":
    main()
