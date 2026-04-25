import argparse
import gc
import json
import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_aetherflow import (  # noqa: E402
    FEATURE_COLUMNS,
    _evaluate_threshold,
    _search_threshold,
    _setup_mix,
)
from aetherflow_features import ensure_feature_columns  # noqa: E402
from aetherflow_model_bundle import (  # noqa: E402
    META_CONTEXT_FEATURE_COLUMNS,
    build_meta_context_frame,
    bundle_feature_columns,
    bundle_feature_columns_by_family,
    family_feature_columns_map,
    make_family_head_bundle,
    predict_bundle_probabilities,
)
from train_manifold_strategy import _build_model, _sample_weights  # noqa: E402


def _resolve_path(path_text: str, default_relative: str) -> Path:
    raw = str(path_text or "").strip()
    path = Path(raw).expanduser() if raw else (ROOT / default_relative)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


_REGIME_NAME_TO_ID = {
    "TREND_GEODESIC": 0,
    "CHOP_SPIRAL": 1,
    "DISPERSED": 2,
    "ROTATIONAL_TURBULENCE": 3,
}


def _load_walkforward_data(
    path: Path,
    *,
    allowed_setup_families: Optional[set[str]] = None,
    allowed_session_ids: Optional[set[int]] = None,
    allowed_regimes: Optional[set[str]] = None,
) -> pd.DataFrame:
    required = set(FEATURE_COLUMNS) | {"label", "net_points", "candidate_side", "setup_family"}
    filters = []
    if allowed_setup_families:
        filters.append(("setup_family", "in", sorted(str(item) for item in allowed_setup_families)))
    if allowed_session_ids:
        filters.append(("session_id", "in", sorted(int(item) for item in allowed_session_ids)))
    if allowed_regimes:
        normalized_regimes = [
            str(regime_name).strip().upper()
            for regime_name in allowed_regimes
            if str(regime_name).strip()
        ]
        allowed_regime_ids = [
            int(_REGIME_NAME_TO_ID[regime_name])
            for regime_name in sorted(normalized_regimes)
            if regime_name in _REGIME_NAME_TO_ID
        ]
        if allowed_regime_ids:
            filters.append(("manifold_regime_id", "in", allowed_regime_ids))
    read_kwargs = {"filters": filters} if filters else {}
    data = pd.read_parquet(path, **read_kwargs)
    data = ensure_feature_columns(data)
    missing = sorted(col for col in required if col not in data.columns)
    if missing:
        raise RuntimeError(f"Cached AetherFlow training parquet missing columns: {', '.join(missing)}")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["label", "net_points"])
    data["label"] = data["label"].astype(int)
    return data


def _training_sample_weights(
    train: pd.DataFrame,
    *,
    mode: str,
    edge_power: float,
    edge_cap: float,
) -> np.ndarray:
    base = _sample_weights(train["label"])
    mode_key = str(mode or "balanced").strip().lower()
    if mode_key == "balanced":
        return base

    net_points = pd.to_numeric(train.get("net_points"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    abs_net = np.abs(net_points)
    nonzero_abs = abs_net[abs_net > 0.0]
    scale = float(np.quantile(nonzero_abs, 0.75)) if nonzero_abs.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    power = max(0.1, float(edge_power))
    cap = max(1.0, float(edge_cap))
    edge = 1.0 + np.power(np.clip(abs_net / scale, 0.0, cap), power)

    if mode_key == "balanced_abs_net":
        return (base * edge).astype(float)

    if mode_key == "balanced_positive_edge":
        positive_edge = 1.0 + np.power(np.clip(np.maximum(net_points, 0.0) / scale, 0.0, cap), power)
        y_arr = train["label"].astype(int).to_numpy(dtype=np.int8)
        return (base * np.where(y_arr == 1, positive_edge, 1.0)).astype(float)

    if mode_key == "balanced_positive_edge_family":
        positive_edge = 1.0 + np.power(np.clip(np.maximum(net_points, 0.0) / scale, 0.0, cap), power)
        y_arr = train["label"].astype(int).to_numpy(dtype=np.int8)
        family = train["setup_family"].astype(str).to_numpy()
        family_bonus = np.where(
            family == "aligned_flow",
            1.20,
            np.where(family == "transition_burst", 1.10, 1.0),
        )
        return (base * np.where(y_arr == 1, positive_edge * family_bonus, 1.0)).astype(float)

    raise ValueError(f"Unsupported sample-weight mode: {mode}")


def _filter_training_rows_by_edge(
    train: pd.DataFrame,
    *,
    positive_net_quantile: float,
    positive_net_threshold: Optional[float],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quantile = float(positive_net_quantile)
    explicit_threshold = (
        float(positive_net_threshold)
        if positive_net_threshold is not None and np.isfinite(float(positive_net_threshold))
        else None
    )
    if explicit_threshold is None and quantile <= 0.0:
        return train, {
            "enabled": False,
            "rows_before": int(len(train)),
            "rows_after": int(len(train)),
            "positive_rows_before": int(np.sum(train["label"].astype(int).to_numpy(dtype=np.int8) == 1)),
            "positive_rows_after": int(np.sum(train["label"].astype(int).to_numpy(dtype=np.int8) == 1)),
            "mode": "none",
            "thresholds_by_family": {},
        }

    work = train.copy()
    labels = work["label"].astype(int)
    net_points = pd.to_numeric(work.get("net_points"), errors="coerce").fillna(0.0)
    setup_family = work["setup_family"].astype(str)
    positive_mask = labels.eq(1)
    keep_mask = np.ones(len(work), dtype=bool)
    thresholds_by_family: dict[str, float] = {}

    if explicit_threshold is not None:
        keep_mask = (~positive_mask.to_numpy(dtype=bool)) | (net_points.to_numpy(dtype=float) >= float(explicit_threshold))
        mode = "fixed_positive_threshold"
    else:
        mode = "family_positive_quantile"
        keep_mask = ~positive_mask.to_numpy(dtype=bool)
        for family_name in sorted(setup_family.dropna().unique().tolist()):
            family_positive = positive_mask & setup_family.eq(str(family_name))
            if not bool(family_positive.any()):
                continue
            threshold_value = float(net_points.loc[family_positive].quantile(quantile))
            thresholds_by_family[str(family_name)] = threshold_value
            family_keep = family_positive.to_numpy(dtype=bool) & (
                net_points.to_numpy(dtype=float) >= float(threshold_value)
            )
            keep_mask = keep_mask | family_keep

    filtered = work.loc[keep_mask].copy()
    if filtered.empty or filtered["label"].nunique() < 2:
        return train, {
            "enabled": False,
            "fallback_used": True,
            "rows_before": int(len(train)),
            "rows_after": int(len(train)),
            "positive_rows_before": int(np.sum(labels.to_numpy(dtype=np.int8) == 1)),
            "positive_rows_after": int(np.sum(labels.to_numpy(dtype=np.int8) == 1)),
            "mode": mode,
            "thresholds_by_family": thresholds_by_family,
        }
    return filtered, {
        "enabled": True,
        "fallback_used": False,
        "rows_before": int(len(train)),
        "rows_after": int(len(filtered)),
        "positive_rows_before": int(np.sum(labels.to_numpy(dtype=np.int8) == 1)),
        "positive_rows_after": int(np.sum(filtered["label"].astype(int).to_numpy(dtype=np.int8) == 1)),
        "mode": mode,
        "positive_net_quantile": (float(quantile) if explicit_threshold is None else None),
        "positive_net_threshold": explicit_threshold,
        "thresholds_by_family": thresholds_by_family,
    }


def _parse_allowed_setups(raw_values: Optional[list[str]]) -> Optional[set[str]]:
    if not raw_values:
        return None
    values: set[str] = set()
    for raw in raw_values:
        for item in str(raw or "").split(","):
            text = str(item or "").strip()
            if text:
                values.add(text)
    return values or None


def _parse_allowed_sessions(raw_value: Optional[str]) -> Optional[set[int]]:
    text = str(raw_value or "").strip()
    if not text:
        return None
    values: set[int] = set()
    for item in text.split(","):
        piece = str(item or "").strip()
        if not piece:
            continue
        try:
            values.add(int(piece))
        except Exception as exc:
            raise ValueError(f"Invalid session id: {piece}") from exc
    return values or None


def _coerce_session_allowlist(value) -> Optional[set[int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: set[int] = set()
    for item in items:
        try:
            out.add(int(float(item)))
        except Exception:
            continue
    return out if out else None


def _coerce_upper_string_allowlist(value) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out = {str(item).strip().upper() for item in items if str(item).strip()}
    return out if out else None


def _coerce_side_allowlist(value) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: set[str] = set()
    for item in items:
        text = str(item).strip().upper()
        if text in {"1", "+1", "LONG", "BUY"}:
            out.add("LONG")
        elif text in {"-1", "SHORT", "SELL"}:
            out.add("SHORT")
    return out if out else None


def _coerce_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return float(out) if np.isfinite(out) else None


def _normalize_train_policy_mapping(raw_policy, *, allow_match_fields: bool, allow_rules: bool) -> dict:
    if not isinstance(raw_policy, dict):
        return {}
    policy = dict(raw_policy or {})
    out: dict = {}
    for key in (
        "threshold",
        "max_abs_vwap_dist_atr",
        "max_directional_vwap_dist_atr",
        "min_d_alignment_3",
        "min_signed_d_alignment_3",
        "min_d_coherence_3",
        "min_setup_strength",
        "min_alignment_pct",
        "min_smoothness_pct",
        "max_stress_pct",
        "min_flow_agreement",
        "min_flow_mag_slow",
        "max_flow_mag_slow",
        "min_pressure_imbalance_30",
        "min_signed_pressure_30",
        "min_coherence_pct",
        "min_phase_regime_run_bars",
        "min_phase_d_alignment_mean_5",
    ):
        if key in policy:
            out[key] = _coerce_optional_float(policy.get(key))
    if "allowed_session_ids" in policy:
        out["allowed_session_ids"] = _coerce_session_allowlist(policy.get("allowed_session_ids"))
    if "allowed_regimes" in policy:
        out["allowed_regimes"] = _coerce_upper_string_allowlist(policy.get("allowed_regimes"))
    if "blocked_regimes" in policy:
        out["blocked_regimes"] = _coerce_upper_string_allowlist(policy.get("blocked_regimes"))
    if "allowed_sides" in policy:
        out["allowed_sides"] = _coerce_side_allowlist(policy.get("allowed_sides"))
    if allow_match_fields:
        if "match_session_ids" in policy:
            out["match_session_ids"] = _coerce_session_allowlist(policy.get("match_session_ids"))
        if "match_regimes" in policy:
            out["match_regimes"] = _coerce_upper_string_allowlist(policy.get("match_regimes"))
        if "match_sides" in policy:
            out["match_sides"] = _coerce_side_allowlist(policy.get("match_sides"))
    if allow_rules:
        raw_rules = policy.get("policy_rules", policy.get("rules"))
        if isinstance(raw_rules, list):
            out["policy_rules"] = [
                normalized_rule
                for item in raw_rules
                if (normalized_rule := _normalize_train_policy_mapping(item, allow_match_fields=True, allow_rules=False))
            ]
    return out


def _merge_train_policy_layers(*layers: dict) -> dict:
    merged: dict = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for key, value in layer.items():
            if key in {"policy_rules", "rules", "match_session_ids", "match_regimes", "match_sides", "name"}:
                continue
            if isinstance(value, set):
                merged[key] = set(value)
            else:
                merged[key] = value
    return merged


def _regime_name_series(frame: pd.DataFrame) -> pd.Series:
    regime_id = pd.to_numeric(frame.get("manifold_regime_id"), errors="coerce").fillna(-1).round().astype(int)
    mapping = {
        0: "TREND_GEODESIC",
        1: "CHOP_SPIRAL",
        2: "DISPERSED",
        3: "ROTATIONAL_TURBULENCE",
    }
    return regime_id.map(mapping).fillna("")


def _side_label_series(frame: pd.DataFrame) -> pd.Series:
    side = pd.to_numeric(frame.get("candidate_side"), errors="coerce").fillna(0.0)
    values = np.where(side.to_numpy(dtype=float) > 0.0, "LONG", np.where(side.to_numpy(dtype=float) < 0.0, "SHORT", ""))
    return pd.Series(values, index=frame.index)


def _apply_train_policy_constraints(frame: pd.DataFrame, policy: dict) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    mask = pd.Series(True, index=frame.index)
    session_id = pd.to_numeric(frame.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
    regime_name = _regime_name_series(frame)
    side_label = _side_label_series(frame)

    allowed_sessions = policy.get("allowed_session_ids")
    if allowed_sessions:
        mask &= session_id.isin(sorted(allowed_sessions))
    allowed_regimes = policy.get("allowed_regimes")
    if allowed_regimes:
        mask &= regime_name.isin(sorted(allowed_regimes))
    blocked_regimes = policy.get("blocked_regimes")
    if blocked_regimes:
        mask &= ~regime_name.isin(sorted(blocked_regimes))
    allowed_sides = policy.get("allowed_sides")
    if allowed_sides:
        mask &= side_label.isin(sorted(allowed_sides))

    max_abs_vwap_dist_atr = policy.get("max_abs_vwap_dist_atr")
    if max_abs_vwap_dist_atr is not None:
        vwap_abs = pd.to_numeric(frame.get("vwap_dist_atr"), errors="coerce").fillna(0.0).abs()
        mask &= vwap_abs <= float(max_abs_vwap_dist_atr)

    max_directional_vwap_dist_atr = policy.get("max_directional_vwap_dist_atr")
    if max_directional_vwap_dist_atr is not None:
        directional_vwap = (
            pd.to_numeric(frame.get("candidate_side"), errors="coerce").fillna(0.0)
            * pd.to_numeric(frame.get("vwap_dist_atr"), errors="coerce").fillna(0.0)
        )
        mask &= directional_vwap <= float(max_directional_vwap_dist_atr)

    min_checks = {
        "min_d_alignment_3": "d_alignment_3",
        "min_d_coherence_3": "d_coherence_3",
        "min_setup_strength": "setup_strength",
        "min_alignment_pct": "manifold_alignment_pct",
        "min_smoothness_pct": "manifold_smoothness_pct",
        "min_flow_agreement": "flow_agreement",
        "min_flow_mag_slow": "flow_mag_slow",
        "min_pressure_imbalance_30": "pressure_imbalance_30",
        "min_coherence_pct": "coherence",
        "min_phase_regime_run_bars": "phase_regime_run_bars",
        "min_phase_d_alignment_mean_5": "phase_d_alignment_3_mean_5",
    }
    for policy_key, column_name in min_checks.items():
        min_value = policy.get(policy_key)
        if min_value is not None:
            values = pd.to_numeric(frame.get(column_name), errors="coerce").fillna(0.0)
            mask &= values >= float(min_value)

    min_signed_d_alignment_3 = policy.get("min_signed_d_alignment_3")
    if min_signed_d_alignment_3 is not None:
        signed_alignment = (
            pd.to_numeric(frame.get("candidate_side"), errors="coerce").fillna(0.0)
            * pd.to_numeric(frame.get("d_alignment_3"), errors="coerce").fillna(0.0)
        )
        mask &= signed_alignment >= float(min_signed_d_alignment_3)

    min_signed_pressure_30 = policy.get("min_signed_pressure_30")
    if min_signed_pressure_30 is not None:
        signed_pressure = (
            pd.to_numeric(frame.get("candidate_side"), errors="coerce").fillna(0.0)
            * pd.to_numeric(frame.get("pressure_imbalance_30"), errors="coerce").fillna(0.0)
        )
        mask &= signed_pressure >= float(min_signed_pressure_30)

    max_stress_pct = policy.get("max_stress_pct")
    if max_stress_pct is not None:
        stress = pd.to_numeric(frame.get("manifold_stress_pct"), errors="coerce").fillna(0.0)
        mask &= stress <= float(max_stress_pct)

    max_flow_mag_slow = policy.get("max_flow_mag_slow")
    if max_flow_mag_slow is not None:
        flow_mag_slow = pd.to_numeric(frame.get("flow_mag_slow"), errors="coerce").fillna(0.0)
        mask &= flow_mag_slow <= float(max_flow_mag_slow)

    return mask


def _row_match_mask(
    frame: pd.DataFrame,
    *,
    match_session_ids: Optional[set[int]],
    match_regimes: Optional[set[str]],
    match_sides: Optional[set[str]] = None,
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if match_session_ids:
        session_id = pd.to_numeric(frame.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
        mask &= session_id.isin(sorted(match_session_ids))
    if match_regimes:
        mask &= _regime_name_series(frame).isin(sorted(match_regimes))
    if match_sides:
        mask &= _side_label_series(frame).isin(sorted(match_sides))
    return mask


def _apply_family_train_rule_filter(
    frame: pd.DataFrame,
    *,
    policy: Optional[dict],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty or not isinstance(policy, dict) or not policy:
        return frame, {
            "enabled": False,
            "rows_before": int(len(frame)),
            "rows_after": int(len(frame)),
            "rules": [],
        }
    base_policy = _normalize_train_policy_mapping(policy, allow_match_fields=False, allow_rules=True)
    base_core = _merge_train_policy_layers(base_policy)
    selected_mask = _apply_train_policy_constraints(frame, base_core)
    rule_hits: list[dict[str, Any]] = []
    for idx, rule in enumerate(base_policy.get("policy_rules", []) or [], start=1):
        match_mask = _row_match_mask(
            frame,
            match_session_ids=rule.get("match_session_ids"),
            match_regimes=rule.get("match_regimes"),
            match_sides=rule.get("match_sides"),
        )
        if not bool(match_mask.any()):
            rule_hits.append({"index": int(idx), "matched_rows": 0, "kept_rows": 0})
            continue
        merged_policy = _merge_train_policy_layers(base_core, rule)
        rule_keep = _apply_train_policy_constraints(frame.loc[match_mask], merged_policy)
        selected_mask.loc[match_mask] = rule_keep
        rule_hits.append(
            {
                "index": int(idx),
                "matched_rows": int(match_mask.sum()),
                "kept_rows": int(rule_keep.sum()),
            }
        )
    filtered = frame.loc[selected_mask].copy()
    return filtered, {
        "enabled": True,
        "rows_before": int(len(frame)),
        "rows_after": int(len(filtered)),
        "rules": rule_hits,
    }


def _load_family_train_rules(path_text: Optional[str]) -> dict[str, dict]:
    raw = str(path_text or "").strip()
    if not raw:
        return {}
    path = _resolve_path(raw, raw)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "family_policies" in payload:
        payload = payload.get("family_policies", {})
    if not isinstance(payload, dict):
        raise RuntimeError(f"Family train rules file must contain a mapping: {path}")
    out: dict[str, dict] = {}
    for family_name, raw_policy in payload.items():
        family_key = str(family_name or "").strip()
        if not family_key:
            continue
        normalized = _normalize_train_policy_mapping(raw_policy, allow_match_fields=False, allow_rules=True)
        if normalized:
            out[family_key] = normalized
    return out


def _unique_years(index: pd.DatetimeIndex) -> list[int]:
    return sorted({int(ts.year) for ts in index})


def _build_walkforward_folds(
    index: pd.DatetimeIndex,
    *,
    min_train_years: int,
    max_train_years: Optional[int],
    valid_years: int,
    test_years: int,
    first_test_year: Optional[int],
) -> list[dict]:
    years = _unique_years(index)
    required = int(min_train_years) + int(valid_years) + int(test_years)
    if len(years) < required:
        raise ValueError(
            f"Not enough years for walkforward: have={len(years)} required={required}"
        )

    folds: list[dict] = []
    start_idx = int(min_train_years) + int(valid_years)
    max_start = len(years) - int(test_years)
    for test_start_idx in range(start_idx, max_start + 1):
        train_years = years[: test_start_idx - int(valid_years)]
        if max_train_years is not None and int(max_train_years) > 0 and len(train_years) > int(max_train_years):
            train_years = train_years[-int(max_train_years) :]
        valid_years_list = years[test_start_idx - int(valid_years) : test_start_idx]
        test_years_list = years[test_start_idx : test_start_idx + int(test_years)]
        if not train_years or not valid_years_list or not test_years_list:
            continue
        if first_test_year is not None and int(test_years_list[0]) < int(first_test_year):
            continue
        folds.append(
            {
                "name": f"test_{test_years_list[0]}_{test_years_list[-1]}",
                "train_years": list(train_years),
                "valid_years": list(valid_years_list),
                "test_years": list(test_years_list),
            }
        )
    return folds


def _fit_model_with_fallback(
    *,
    model_type: str,
    seed: int,
    workers: int,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray,
):
    fit_workers = max(1, int(workers))
    model_choice = str(model_type or "hgb").strip().lower()
    if model_choice == "hgb_reg":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=5,
            max_iter=300,
            min_samples_leaf=64,
            random_state=int(seed),
        )
    elif model_choice == "rf_reg":
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=20,
            random_state=int(seed),
            n_jobs=max(1, int(fit_workers)),
        )
    else:
        model = _build_model(model_choice, int(seed), workers=fit_workers)
    try:
        model.fit(x_train, y_train, sample_weight=sample_weight)
    except (PermissionError, OSError) as exc:
        if model_choice == "hgb":
            logging.warning("HGB fit failed (%s); falling back to RandomForest.", exc)
            model_choice = "rf"
            model = _build_model(model_choice, int(seed), workers=fit_workers)
            model.fit(x_train, y_train, sample_weight=sample_weight)
        elif model_choice == "hgb_reg":
            logging.warning("HGB reg fit failed (%s); falling back to RandomForest reg.", exc)
            model_choice = "rf_reg"
            model = RandomForestRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=20,
                random_state=int(seed),
                n_jobs=max(1, int(fit_workers)),
            )
            model.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            logging.warning("RF fit failed with workers=%d (%s); retrying with workers=1.", fit_workers, exc)
            fit_workers = 1
            if model_choice == "rf_reg":
                model = RandomForestRegressor(
                    n_estimators=400,
                    max_depth=12,
                    min_samples_leaf=20,
                    random_state=int(seed),
                    n_jobs=1,
                )
            else:
                model = _build_model(model_choice, int(seed), workers=fit_workers)
            model.fit(x_train, y_train, sample_weight=sample_weight)
    return model, model_choice, fit_workers


def _is_regression_model_type(model_type: str) -> bool:
    return str(model_type or "").strip().lower() in {"hgb_reg", "rf_reg"}


def _training_target(train: pd.DataFrame, *, model_type: str) -> tuple[pd.Series, str, float]:
    model_key = str(model_type or "").strip().lower()
    if _is_regression_model_type(model_key):
        target = pd.to_numeric(train.get("net_points"), errors="coerce").fillna(0.0).clip(-6.0, 10.0)
        return target.astype(float), "edge_sigmoid", 1.5
    return train["label"], "proba", 1.0


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _fold_seed(base_seed: int, fold: dict) -> int:
    test_years = [int(y) for y in (fold.get("test_years") or [])]
    valid_years = [int(y) for y in (fold.get("valid_years") or [])]
    anchor = 0
    if test_years:
        anchor += int(min(test_years)) * 101
    if valid_years:
        anchor += int(min(valid_years)) * 17
    return int(base_seed) + int(anchor)


def _train_single_bundle(
    *,
    train: pd.DataFrame,
    model_type: str,
    seed: int,
    workers: int,
    sample_weight_mode: str,
    sample_weight_power: float,
    sample_weight_cap: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train, predict_mode, edge_scale = _training_target(train, model_type=str(model_type))
    sample_weight = _training_sample_weights(
        train,
        mode=str(sample_weight_mode),
        edge_power=float(sample_weight_power),
        edge_cap=float(sample_weight_cap),
    )
    model, fitted_model_type, fit_workers = _fit_model_with_fallback(
        model_type=str(model_type),
        seed=int(seed),
        workers=int(workers),
        x_train=train[FEATURE_COLUMNS],
        y_train=y_train,
        sample_weight=sample_weight,
    )
    bundle = {
        "bundle_design": "single",
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "predict_mode": str(predict_mode),
        "edge_scale": float(edge_scale),
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "threshold": 0.58,
    }
    meta = {
        "bundle_design": "single",
        "shared_model_type": str(fitted_model_type),
        "shared_workers": int(fit_workers),
        "family_heads": {},
        "family_feature_mode": "all",
        "family_head_weight": 1.0,
        "sample_weight_mode": str(sample_weight_mode),
        "predict_mode": str(predict_mode),
        "edge_scale": float(edge_scale),
    }
    return bundle, meta


def _train_family_head_bundle(
    *,
    train: pd.DataFrame,
    model_type: str,
    seed: int,
    workers: int,
    family_feature_mode: str,
    family_head_weight: float,
    head_min_train_rows: int,
    head_family_names: Optional[set[str]] = None,
    family_head_model_type: Optional[str] = None,
    sample_weight_mode: str = "balanced",
    sample_weight_power: float = 0.5,
    sample_weight_cap: float = 4.0,
    family_train_rules: Optional[dict[str, dict]] = None,
    filter_shared_train_by_family_rules: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    family_train_rules = dict(family_train_rules or {})
    shared_train = train
    shared_filter_meta = {
        "enabled": False,
        "rows_before": int(len(train)),
        "rows_after": int(len(train)),
        "families": {},
    }
    if bool(filter_shared_train_by_family_rules) and family_train_rules:
        family_series_all = train["setup_family"].astype(str)
        shared_parts: list[pd.DataFrame] = []
        shared_family_meta: dict[str, Any] = {}
        for family_name in sorted(family_series_all.dropna().unique().tolist()):
            family_rows = train.loc[family_series_all.eq(str(family_name))].copy()
            filtered_rows, filter_meta = _apply_family_train_rule_filter(
                family_rows,
                policy=family_train_rules.get(str(family_name)),
            )
            shared_family_meta[str(family_name)] = filter_meta
            if not filtered_rows.empty:
                shared_parts.append(filtered_rows)
        if shared_parts:
            shared_train = pd.concat(shared_parts).sort_index()
            shared_filter_meta = {
                "enabled": True,
                "rows_before": int(len(train)),
                "rows_after": int(len(shared_train)),
                "families": shared_family_meta,
            }
    if shared_train.empty or shared_train["label"].nunique() < 2:
        shared_train = train
        shared_filter_meta = {
            "enabled": False,
            "fallback_used": True,
            "rows_before": int(len(train)),
            "rows_after": int(len(train)),
            "families": shared_filter_meta.get("families", {}),
        }

    y_train_shared, predict_mode, edge_scale = _training_target(shared_train, model_type=str(model_type))
    shared_sample_weight = _training_sample_weights(
        shared_train,
        mode=str(sample_weight_mode),
        edge_power=float(sample_weight_power),
        edge_cap=float(sample_weight_cap),
    )
    shared_model, shared_model_type, shared_workers = _fit_model_with_fallback(
        model_type=str(model_type),
        seed=int(seed),
        workers=int(workers),
        x_train=shared_train[FEATURE_COLUMNS],
        y_train=y_train_shared,
        sample_weight=shared_sample_weight,
    )
    family_names = sorted(str(name) for name in train["setup_family"].dropna().astype(str).unique().tolist() if str(name).strip())
    feature_columns_by_family = family_feature_columns_map(
        families=family_names,
        mode=str(family_feature_mode or "curated"),
    )
    family_models: dict[str, Any] = {}
    family_heads: dict[str, Any] = {}
    family_series = train["setup_family"].astype(str)
    eligible_head_families = (
        {str(name).strip() for name in (head_family_names or set()) if str(name).strip()}
        if head_family_names
        else set(family_names)
    )

    for family_idx, family_name in enumerate(family_names, start=1):
        family_train = train.loc[family_series.eq(str(family_name))].copy()
        head_info = {
            "train_rows": int(len(family_train)),
            "train_positive_rate": float(np.mean(family_train["label"].to_numpy(dtype=float))) if not family_train.empty else 0.0,
            "trained": False,
            "eligible": bool(str(family_name) in eligible_head_families),
            "feature_columns": list(feature_columns_by_family.get(str(family_name), [])),
        }
        family_train_filtered, family_filter_meta = _apply_family_train_rule_filter(
            family_train,
            policy=family_train_rules.get(str(family_name)),
        )
        head_info["train_filter"] = family_filter_meta
        if (
            str(family_name) not in eligible_head_families
            or family_train_filtered.empty
            or len(family_train_filtered) < int(head_min_train_rows)
            or family_train_filtered["label"].nunique() < 2
        ):
            family_heads[str(family_name)] = head_info
            continue
        family_model, family_model_type, family_workers = _fit_model_with_fallback(
            model_type=str(family_head_model_type or model_type),
            seed=int(seed) + (int(family_idx) * 997),
            workers=int(workers),
            x_train=family_train_filtered[feature_columns_by_family[str(family_name)]],
            y_train=_training_target(
                family_train_filtered,
                model_type=str(family_head_model_type or model_type),
            )[0],
            sample_weight=_training_sample_weights(
                family_train_filtered,
                mode=str(sample_weight_mode),
                edge_power=float(sample_weight_power),
                edge_cap=float(sample_weight_cap),
            ),
        )
        family_models[str(family_name)] = family_model
        head_info.update(
            {
                "trained": True,
                "model_type": str(family_model_type),
                "workers": int(family_workers),
                "fit_rows": int(len(family_train_filtered)),
                "fit_positive_rate": float(np.mean(family_train_filtered["label"].to_numpy(dtype=float))),
            }
        )
        family_heads[str(family_name)] = head_info

    bundle = make_family_head_bundle(
        shared_model=shared_model,
        family_models=family_models,
        family_feature_columns=feature_columns_by_family,
        family_head_weight=float(family_head_weight),
        threshold=0.58,
        trained_at=pd.Timestamp.now("UTC").isoformat(),
        family_feature_mode=str(family_feature_mode or "curated"),
        shared_calibrator=None,
        family_calibrators=None,
    )
    bundle["predict_mode"] = str(predict_mode)
    bundle["edge_scale"] = float(edge_scale)
    meta = {
        "bundle_design": "family_heads",
        "shared_model_type": str(shared_model_type),
        "shared_workers": int(shared_workers),
        "family_heads": family_heads,
        "family_feature_mode": str(family_feature_mode or "curated"),
        "family_head_weight": float(family_head_weight),
        "sample_weight_mode": str(sample_weight_mode),
        "predict_mode": str(predict_mode),
        "edge_scale": float(edge_scale),
        "shared_train_filter": shared_filter_meta,
    }
    return bundle, meta


def _fit_logit_linear_calibrator(
    probabilities: np.ndarray,
    labels: pd.Series,
    *,
    min_rows: int,
) -> Optional[dict]:
    probs = np.asarray(probabilities, dtype=float)
    y = pd.Series(labels).astype(int).to_numpy(dtype=np.int8)
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


def _attach_bundle_calibrators(
    *,
    bundle: dict[str, Any],
    val: pd.DataFrame,
    calibration_mode: str,
    calibration_min_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mode = str(calibration_mode or "none").strip().lower()
    if mode == "none":
        bundle["shared_calibrator"] = None
        bundle["family_calibrators"] = {}
        return bundle, {
            "mode": "none",
            "shared_calibrator": None,
            "family_calibrators": {},
        }

    raw_probs = predict_bundle_probabilities(bundle, val)
    if str(bundle.get("bundle_design", "") or "").strip().lower() == "meta_context":
        meta_calibrator = _fit_logit_linear_calibrator(
            raw_probs,
            val["label"],
            min_rows=int(calibration_min_rows),
        )
        bundle["meta_calibrator"] = meta_calibrator
        return bundle, {
            "mode": mode,
            "shared_calibrator": None,
            "family_calibrators": {},
            "meta_calibrator": _json_safe(meta_calibrator) if meta_calibrator is not None else None,
        }
    shared_calibrator = _fit_logit_linear_calibrator(
        raw_probs,
        val["label"],
        min_rows=int(calibration_min_rows),
    )
    family_calibrators: dict[str, dict] = {}
    if mode == "per_family":
        family_series = val["setup_family"].astype(str)
        for family_name in sorted(family_series.dropna().unique().tolist()):
            family_mask = family_series.eq(str(family_name)).to_numpy(dtype=bool)
            calibrator = _fit_logit_linear_calibrator(
                raw_probs[family_mask],
                val.loc[family_mask, "label"],
                min_rows=int(calibration_min_rows),
            )
            if calibrator is not None:
                family_calibrators[str(family_name)] = calibrator

    bundle["shared_calibrator"] = shared_calibrator
    bundle["family_calibrators"] = family_calibrators
    return bundle, {
        "mode": mode,
        "shared_calibrator": _json_safe(shared_calibrator) if shared_calibrator is not None else None,
        "family_calibrators": _json_safe(family_calibrators),
        "meta_calibrator": None,
    }


def _train_meta_context_bundle(
    *,
    train: pd.DataFrame,
    model_type: str,
    meta_model_type: str,
    seed: int,
    workers: int,
    meta_weight: float,
    sample_weight_mode: str,
    sample_weight_power: float,
    sample_weight_cap: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    y_train_base, predict_mode, edge_scale = _training_target(train, model_type=str(model_type))
    base_sample_weight = _training_sample_weights(
        train,
        mode=str(sample_weight_mode),
        edge_power=float(sample_weight_power),
        edge_cap=float(sample_weight_cap),
    )
    base_model, base_model_type, base_workers = _fit_model_with_fallback(
        model_type=str(model_type),
        seed=int(seed),
        workers=int(workers),
        x_train=train[FEATURE_COLUMNS],
        y_train=y_train_base,
        sample_weight=base_sample_weight,
    )
    base_bundle = {
        "bundle_design": "single",
        "model": base_model,
        "feature_columns": list(FEATURE_COLUMNS),
        "predict_mode": str(predict_mode),
        "edge_scale": float(edge_scale),
        "threshold": 0.58,
    }
    base_prob_train = predict_bundle_probabilities(base_bundle, train)
    meta_train = build_meta_context_frame(train, base_prob_train)
    meta_model, meta_model_type_fitted, meta_workers = _fit_model_with_fallback(
        model_type=str(meta_model_type or "logreg"),
        seed=int(seed) + 313,
        workers=int(workers),
        x_train=meta_train[META_CONTEXT_FEATURE_COLUMNS],
        y_train=train["label"],
        sample_weight=base_sample_weight,
    )
    bundle = {
        "bundle_design": "meta_context",
        "base_model": base_model,
        "base_feature_columns": list(FEATURE_COLUMNS),
        "meta_model": meta_model,
        "meta_feature_columns": list(META_CONTEXT_FEATURE_COLUMNS),
        "meta_weight": float(meta_weight),
        "predict_mode": str(predict_mode),
        "edge_scale": float(edge_scale),
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "threshold": 0.58,
    }
    meta = {
        "bundle_design": "meta_context",
        "shared_model_type": str(base_model_type),
        "shared_workers": int(base_workers),
        "meta_model_type": str(meta_model_type_fitted),
        "meta_workers": int(meta_workers),
        "meta_weight": float(meta_weight),
        "family_heads": {},
        "family_feature_mode": "meta_context",
        "family_head_weight": 1.0,
        "sample_weight_mode": str(sample_weight_mode),
        "predict_mode": str(predict_mode),
        "edge_scale": float(edge_scale),
    }
    return bundle, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Walkforward OOS validation for AetherFlow.")
    parser.add_argument("--features-parquet", default="aetherflow_features_fullrange_v2.parquet")
    parser.add_argument("--output-dir", default="artifacts/aetherflow_walkforward_fullrange_v2")
    parser.add_argument("--report-name", default="walkforward_report.json")
    parser.add_argument("--csv-name", default="walkforward_folds.csv")
    parser.add_argument("--save-fold-artifacts", action="store_true")
    parser.add_argument("--model-type", default="hgb")
    parser.add_argument("--bundle-design", choices=["single", "family_heads", "meta_context"], default="single")
    parser.add_argument("--family-feature-mode", choices=["all", "curated"], default="curated")
    parser.add_argument("--family-head-weight", type=float, default=0.8)
    parser.add_argument("--head-min-train-rows", type=int, default=1500)
    parser.add_argument("--family-head-setup-family", action="append", default=None)
    parser.add_argument("--family-head-model-type", default=None)
    parser.add_argument("--family-head-train-rules-file", default=None)
    parser.add_argument("--filter-shared-train-by-family-rules", action="store_true")
    parser.add_argument("--meta-model-type", default="logreg")
    parser.add_argument("--meta-weight", type=float, default=0.25)
    parser.add_argument("--calibration-mode", choices=["none", "global", "per_family"], default="none")
    parser.add_argument("--calibration-min-rows", type=int, default=800)
    parser.add_argument(
        "--sample-weight-mode",
        choices=["balanced", "balanced_abs_net", "balanced_positive_edge", "balanced_positive_edge_family"],
        default="balanced",
    )
    parser.add_argument("--sample-weight-power", type=float, default=0.5)
    parser.add_argument("--sample-weight-cap", type=float, default=4.0)
    parser.add_argument("--positive-net-quantile-filter", type=float, default=0.0)
    parser.add_argument("--positive-net-threshold", type=float, default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-years", type=int, default=6)
    parser.add_argument("--max-train-years", type=int, default=0)
    parser.add_argument("--valid-years", type=int, default=1)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--first-test-year", type=int, default=2018)
    parser.add_argument("--thr-min", type=float, default=0.53)
    parser.add_argument("--thr-max", type=float, default=0.82)
    parser.add_argument("--thr-step", type=float, default=0.01)
    parser.add_argument("--fallback-threshold-on-search-fail", type=float, default=None)
    parser.add_argument("--min-val-trades", type=int, default=120)
    parser.add_argument("--min-val-coverage", type=float, default=0.02)
    parser.add_argument("--coverage-target", type=float, default=0.08)
    parser.add_argument("--coverage-penalty", type=float, default=0.20)
    parser.add_argument("--val-folds", type=int, default=4)
    parser.add_argument("--min-fold-trades", type=int, default=25)
    parser.add_argument("--min-positive-folds", type=int, default=2)
    parser.add_argument("--objective-mean-weight", type=float, default=0.55)
    parser.add_argument("--objective-worst-weight", type=float, default=0.25)
    parser.add_argument("--objective-total-weight", type=float, default=0.20)
    parser.add_argument("--objective-std-penalty", type=float, default=0.15)
    parser.add_argument("--allowed-session-ids", default=None)
    parser.add_argument("--allowed-setup-family", action="append", default=None)
    parser.add_argument("--allowed-regimes", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features_path = _resolve_path(str(args.features_parquet), "aetherflow_features_fullrange_v2.parquet")
    output_dir = _resolve_path(str(args.output_dir), "artifacts/aetherflow_walkforward_fullrange_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / str(args.report_name)
    csv_path = output_dir / str(args.csv_name)
    fold_artifacts_dir = output_dir / "fold_artifacts"
    if bool(args.save_fold_artifacts):
        fold_artifacts_dir.mkdir(parents=True, exist_ok=True)

    allowed_setups = _parse_allowed_setups(args.allowed_setup_family)
    head_setups = _parse_allowed_setups(args.family_head_setup_family)
    family_train_rules = _load_family_train_rules(args.family_head_train_rules_file)
    allowed_session_ids = _parse_allowed_sessions(args.allowed_session_ids)
    allowed_regimes = _coerce_upper_string_allowlist(args.allowed_regimes)
    data = _load_walkforward_data(
        features_path,
        allowed_setup_families=allowed_setups,
        allowed_session_ids=allowed_session_ids,
        allowed_regimes=allowed_regimes,
    ).sort_index()
    if allowed_setups:
        data = data.loc[data["setup_family"].astype(str).isin(sorted(allowed_setups))].copy()
        logging.info("Setup filter enabled: %s rows=%d", sorted(allowed_setups), len(data))
    if head_setups:
        logging.info("Family-head override enabled: %s", sorted(head_setups))
    if family_train_rules:
        logging.info("Family head train rules enabled for: %s", sorted(family_train_rules.keys()))
    if allowed_session_ids:
        session_series = pd.to_numeric(data.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
        data = data.loc[session_series.isin(sorted(allowed_session_ids))].copy()
        logging.info("Session filter enabled: %s rows=%d", sorted(allowed_session_ids), len(data))
    if allowed_regimes:
        regime_series = _regime_name_series(data)
        data = data.loc[regime_series.isin(sorted(allowed_regimes))].copy()
        logging.info("Regime filter enabled: %s rows=%d", sorted(allowed_regimes), len(data))
    if data.empty:
        raise RuntimeError(f"No rows found in {features_path}")

    folds = _build_walkforward_folds(
        pd.DatetimeIndex(data.index),
        min_train_years=int(args.min_train_years),
        max_train_years=(None if int(args.max_train_years) <= 0 else int(args.max_train_years)),
        valid_years=int(args.valid_years),
        test_years=int(args.test_years),
        first_test_year=int(args.first_test_year) if args.first_test_year is not None else None,
    )
    if not folds:
        raise RuntimeError("No walkforward folds were generated.")

    year_arr = pd.DatetimeIndex(data.index).year.to_numpy(dtype=np.int16)
    fold_rows: list[dict] = []

    for fold_idx, fold in enumerate(folds, start=1):
        train_idx = np.flatnonzero(np.isin(year_arr, np.asarray(fold["train_years"], dtype=np.int16)))
        val_idx = np.flatnonzero(np.isin(year_arr, np.asarray(fold["valid_years"], dtype=np.int16)))
        test_idx = np.flatnonzero(np.isin(year_arr, np.asarray(fold["test_years"], dtype=np.int16)))
        train = data.iloc[train_idx]
        val = data.iloc[val_idx]
        test = data.iloc[test_idx]
        if train.empty or val.empty or test.empty:
            logging.warning("Skipping %s because one split is empty.", fold["name"])
            continue
        if train["label"].nunique() < 2 or val["label"].nunique() < 2 or test["label"].nunique() < 2:
            logging.warning("Skipping %s because one split lacks both classes.", fold["name"])
            continue

        y_val = val["label"]
        y_test = test["label"]

        logging.info(
            "[%d/%d] %s train=%s valid=%s test=%s rows=%d/%d/%d",
            fold_idx,
            len(folds),
            fold["name"],
            fold["train_years"],
            fold["valid_years"],
            fold["test_years"],
            len(train),
            len(val),
            len(test),
        )
        train_fit, train_filter_meta = _filter_training_rows_by_edge(
            train,
            positive_net_quantile=float(args.positive_net_quantile_filter),
            positive_net_threshold=args.positive_net_threshold,
        )
        logging.info(
            "%s train_filter: enabled=%s mode=%s rows=%d->%d positives=%d->%d",
            fold["name"],
            bool(train_filter_meta.get("enabled", False)),
            str(train_filter_meta.get("mode", "none")),
            int(train_filter_meta.get("rows_before", len(train))),
            int(train_filter_meta.get("rows_after", len(train_fit))),
            int(train_filter_meta.get("positive_rows_before", 0)),
            int(train_filter_meta.get("positive_rows_after", 0)),
        )

        bundle_design = str(args.bundle_design or "single").strip().lower()
        if bundle_design == "family_heads":
            bundle, bundle_meta = _train_family_head_bundle(
                train=train_fit,
                model_type=str(args.model_type),
                seed=_fold_seed(int(args.seed), fold),
                workers=int(args.workers),
                family_feature_mode=str(args.family_feature_mode),
                family_head_weight=float(args.family_head_weight),
                head_min_train_rows=int(args.head_min_train_rows),
                head_family_names=head_setups,
                family_head_model_type=(str(args.family_head_model_type).strip().lower() if args.family_head_model_type else None),
                sample_weight_mode=str(args.sample_weight_mode),
                sample_weight_power=float(args.sample_weight_power),
                sample_weight_cap=float(args.sample_weight_cap),
                family_train_rules=family_train_rules,
                filter_shared_train_by_family_rules=bool(args.filter_shared_train_by_family_rules),
            )
        elif bundle_design == "meta_context":
            bundle, bundle_meta = _train_meta_context_bundle(
                train=train_fit,
                model_type=str(args.model_type),
                meta_model_type=str(args.meta_model_type),
                seed=_fold_seed(int(args.seed), fold),
                workers=int(args.workers),
                meta_weight=float(args.meta_weight),
                sample_weight_mode=str(args.sample_weight_mode),
                sample_weight_power=float(args.sample_weight_power),
                sample_weight_cap=float(args.sample_weight_cap),
            )
        else:
            bundle, bundle_meta = _train_single_bundle(
                train=train_fit,
                model_type=str(args.model_type),
                seed=_fold_seed(int(args.seed), fold),
                workers=int(args.workers),
                sample_weight_mode=str(args.sample_weight_mode),
                sample_weight_power=float(args.sample_weight_power),
                sample_weight_cap=float(args.sample_weight_cap),
            )
        logging.info(
            "%s bundle fitted: design=%s shared_model=%s family_heads=%s",
            fold["name"],
            str(bundle_meta.get("bundle_design", "single")),
            str(bundle_meta.get("shared_model_type", args.model_type)),
            ",".join(
                family_name
                for family_name, head in (bundle_meta.get("family_heads", {}) or {}).items()
                if bool((head or {}).get("trained", False))
            ) or "none",
        )
        bundle, calibration_meta = _attach_bundle_calibrators(
            bundle=bundle,
            val=val,
            calibration_mode=str(args.calibration_mode),
            calibration_min_rows=int(args.calibration_min_rows),
        )
        logging.info(
            "%s calibration: mode=%s shared=%s family=%s meta=%s",
            fold["name"],
            str(calibration_meta.get("mode", "none")),
            "yes" if calibration_meta.get("shared_calibrator") else "no",
            ",".join(sorted((calibration_meta.get("family_calibrators", {}) or {}).keys())) or "none",
            "yes" if calibration_meta.get("meta_calibrator") else "no",
        )

        prob_val = predict_bundle_probabilities(bundle, val)
        try:
            best_thr = _search_threshold(
                prob_val,
                val["net_points"].to_numpy(dtype=float),
                thr_min=float(args.thr_min),
                thr_max=float(args.thr_max),
                thr_step=float(args.thr_step),
                min_trades=int(args.min_val_trades),
                min_coverage=float(args.min_val_coverage),
                coverage_target=float(args.coverage_target),
                coverage_penalty=float(args.coverage_penalty),
                val_folds=int(args.val_folds),
                min_fold_trades=int(args.min_fold_trades),
                min_positive_folds=int(args.min_positive_folds),
                objective_mean_weight=float(args.objective_mean_weight),
                objective_worst_weight=float(args.objective_worst_weight),
                objective_total_weight=float(args.objective_total_weight),
                objective_std_penalty=float(args.objective_std_penalty),
                sides=val["candidate_side"].to_numpy(dtype=float),
                session_ids=None,
                allowed_session_ids=None,
            )
        except RuntimeError as exc:
            fallback_threshold = args.fallback_threshold_on_search_fail
            if fallback_threshold is None:
                logging.warning("Skipping %s because threshold search failed: %s", fold["name"], exc)
                continue
            logging.warning(
                "%s threshold search failed (%s); using fallback threshold %.3f for artifact export.",
                fold["name"],
                exc,
                float(fallback_threshold),
            )
            fallback_eval = _evaluate_threshold(
                prob_val,
                val["net_points"].to_numpy(dtype=float),
                threshold=float(fallback_threshold),
                sides=val["candidate_side"].to_numpy(dtype=float),
                session_ids=None,
                allowed_session_ids=None,
            )
            best_thr = {
                "threshold": float(fallback_threshold),
                "trade_count": int(fallback_eval.get("trade_count", 0)),
                "coverage": float(fallback_eval.get("trade_count", 0)) / float(max(1, len(val))),
                "avg_pnl": float(fallback_eval.get("avg_pnl", 0.0)),
                "total_pnl": float(fallback_eval.get("total_pnl", 0.0)),
                "win_rate": float(fallback_eval.get("win_rate", 0.0)),
                "long_share": float(fallback_eval.get("long_share", 0.0)),
                "short_share": float(fallback_eval.get("short_share", 0.0)),
                "objective": None,
                "fallback_used": True,
            }

        bundle["threshold"] = float(best_thr["threshold"])
        prob_test = predict_bundle_probabilities(bundle, test)
        test_eval = _evaluate_threshold(
            prob_test,
            test["net_points"].to_numpy(dtype=float),
            threshold=float(best_thr["threshold"]),
            sides=test["candidate_side"].to_numpy(dtype=float),
            session_ids=None,
            allowed_session_ids=None,
        )
        pred_test = (prob_test >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_test, prob_test))
        except Exception:
            auc = 0.0

        row = {
            "fold": str(fold["name"]),
            "train_years": list(fold["train_years"]),
            "valid_years": list(fold["valid_years"]),
            "test_years": list(fold["test_years"]),
            "model_type": str(bundle_meta.get("shared_model_type", args.model_type)),
            "workers": int(bundle_meta.get("shared_workers", args.workers)),
            "bundle_design": str(bundle_meta.get("bundle_design", "single")),
            "family_feature_mode": str(bundle_meta.get("family_feature_mode", "all")),
            "family_head_weight": float(bundle_meta.get("family_head_weight", 1.0)),
            "sample_weight_mode": str(bundle_meta.get("sample_weight_mode", args.sample_weight_mode)),
            "predict_mode": str(bundle_meta.get("predict_mode", "proba")),
            "edge_scale": float(bundle_meta.get("edge_scale", 1.0)),
            "train_filter": _json_safe(train_filter_meta),
            "family_heads": _json_safe(bundle_meta.get("family_heads", {})),
            "shared_train_filter": _json_safe(bundle_meta.get("shared_train_filter", {})),
            "calibration": _json_safe(calibration_meta),
            "train_rows": int(len(train)),
            "fit_rows": int(len(train_fit)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
            "train_positive_rate": float(np.mean(train["label"].to_numpy(dtype=float))),
            "fit_positive_rate": float(np.mean(train_fit["label"].to_numpy(dtype=float))),
            "val_positive_rate": float(np.mean(y_val.to_numpy(dtype=float))),
            "test_positive_rate": float(np.mean(y_test.to_numpy(dtype=float))),
            "threshold": float(best_thr["threshold"]),
            "validation": {str(k): _json_safe(v) for k, v in best_thr.items()},
            "test": {
                **{str(k): _json_safe(v) for k, v in test_eval.items()},
                "accuracy": float(accuracy_score(y_test, pred_test)),
                "precision": float(precision_score(y_test, pred_test, zero_division=0)),
                "recall": float(recall_score(y_test, pred_test, zero_division=0)),
                "f1": float(f1_score(y_test, pred_test, zero_division=0)),
                "roc_auc": float(auc),
            },
            "setup_mix_train": _setup_mix(train),
            "setup_mix_val": _setup_mix(val),
            "setup_mix_test": _setup_mix(test),
        }
        if bool(args.save_fold_artifacts):
            artifact_stub = str(fold["name"])
            model_path = fold_artifacts_dir / f"{artifact_stub}_model.pkl"
            thresholds_path = fold_artifacts_dir / f"{artifact_stub}_thresholds.json"
            with model_path.open("wb") as fh:
                artifact_bundle = dict(bundle)
                artifact_bundle["trained_at"] = pd.Timestamp.now("UTC").isoformat()
                artifact_bundle["threshold"] = float(best_thr["threshold"])
                artifact_bundle["walkforward_fold"] = str(fold["name"])
                pickle.dump(artifact_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
            thresholds_path.write_text(
                json.dumps(
                    _json_safe(
                        {
                            "threshold": float(best_thr["threshold"]),
                            "validation": {str(k): _json_safe(v) for k, v in best_thr.items()},
                            "feature_columns": bundle_feature_columns(bundle),
                            "feature_columns_by_family": bundle_feature_columns_by_family(bundle),
                            "bundle_design": str(bundle_meta.get("bundle_design", "single")),
                            "family_feature_mode": str(bundle_meta.get("family_feature_mode", "all")),
                            "family_head_weight": float(bundle_meta.get("family_head_weight", 1.0)),
                            "sample_weight_mode": str(bundle_meta.get("sample_weight_mode", args.sample_weight_mode)),
                            "predict_mode": str(bundle_meta.get("predict_mode", "proba")),
                            "edge_scale": float(bundle_meta.get("edge_scale", 1.0)),
                            "train_filter": _json_safe(train_filter_meta),
                            "family_heads": _json_safe(bundle_meta.get("family_heads", {})),
                            "shared_train_filter": _json_safe(bundle_meta.get("shared_train_filter", {})),
                            "calibration": _json_safe(calibration_meta),
                            "trained_at": pd.Timestamp.now("UTC").isoformat(),
                            "walkforward_fold": str(fold["name"]),
                        }
                    ),
                    indent=2,
                )
            )
            row["artifacts"] = {
                "model_file": str(model_path),
                "thresholds_file": str(thresholds_path),
            }
        fold_rows.append(row)
        logging.info(
            "%s threshold=%.3f test_trades=%d test_total_pnl=%.2f test_avg_pnl=%.4f auc=%.4f",
            fold["name"],
            float(best_thr["threshold"]),
            int(test_eval["trade_count"]),
            float(test_eval["total_pnl"]),
            float(test_eval["avg_pnl"]),
            float(auc),
        )
        del train, val, test, y_val, y_test, prob_val, prob_test, pred_test, bundle
        gc.collect()

    if not fold_rows:
        raise RuntimeError("All walkforward folds were skipped.")

    pnl_arr = np.asarray([float(row["test"]["total_pnl"]) for row in fold_rows], dtype=float)
    trades_arr = np.asarray([int(row["test"]["trade_count"]) for row in fold_rows], dtype=float)
    avg_arr = np.asarray([float(row["test"]["avg_pnl"]) for row in fold_rows], dtype=float)
    auc_arr = np.asarray([float(row["test"]["roc_auc"]) for row in fold_rows], dtype=float)
    thr_arr = np.asarray([float(row["threshold"]) for row in fold_rows], dtype=float)
    recent_rows = fold_rows[-min(3, len(fold_rows)) :]
    recent_pnl = float(sum(float(row["test"]["total_pnl"]) for row in recent_rows))
    recent_trades = int(sum(int(row["test"]["trade_count"]) for row in recent_rows))

    summary = {
        "fold_count": int(len(fold_rows)),
        "total_test_pnl": float(np.sum(pnl_arr)),
        "mean_test_pnl": float(np.mean(pnl_arr)),
        "median_test_pnl": float(np.median(pnl_arr)),
        "positive_test_folds": int(np.sum(pnl_arr > 0.0)),
        "negative_test_folds": int(np.sum(pnl_arr <= 0.0)),
        "total_test_trades": int(np.sum(trades_arr)),
        "weighted_test_avg_pnl": float(np.sum(avg_arr * trades_arr) / max(1.0, np.sum(trades_arr))),
        "mean_test_avg_pnl": float(np.mean(avg_arr)),
        "mean_test_auc": float(np.mean(auc_arr)),
        "median_test_auc": float(np.median(auc_arr)),
        "threshold_mean": float(np.mean(thr_arr)),
        "threshold_std": float(np.std(thr_arr)),
        "recent_test_pnl_last_3_folds": recent_pnl,
        "recent_test_trades_last_3_folds": recent_trades,
        "recent_test_avg_pnl_last_3_folds": float(recent_pnl / max(1, recent_trades)),
    }

    report = {
        "version": "aetherflow_walkforward_v1",
        "features_parquet": str(features_path),
        "config": {
            "model_type": str(args.model_type),
            "bundle_design": str(args.bundle_design),
            "family_feature_mode": str(args.family_feature_mode),
            "family_head_weight": float(args.family_head_weight),
            "head_min_train_rows": int(args.head_min_train_rows),
            "family_head_setups": (sorted(str(v) for v in head_setups) if head_setups else []),
            "family_head_model_type": (str(args.family_head_model_type).strip().lower() if args.family_head_model_type else None),
            "family_head_train_rules_file": (str(args.family_head_train_rules_file) if args.family_head_train_rules_file else None),
            "filter_shared_train_by_family_rules": bool(args.filter_shared_train_by_family_rules),
            "meta_model_type": str(args.meta_model_type),
            "meta_weight": float(args.meta_weight),
            "calibration_mode": str(args.calibration_mode),
            "calibration_min_rows": int(args.calibration_min_rows),
            "sample_weight_mode": str(args.sample_weight_mode),
            "sample_weight_power": float(args.sample_weight_power),
            "sample_weight_cap": float(args.sample_weight_cap),
            "positive_net_quantile_filter": float(args.positive_net_quantile_filter),
            "positive_net_threshold": (
                float(args.positive_net_threshold)
                if args.positive_net_threshold is not None
                else None
            ),
            "workers": int(args.workers),
            "seed": int(args.seed),
            "min_train_years": int(args.min_train_years),
            "max_train_years": (None if int(args.max_train_years) <= 0 else int(args.max_train_years)),
            "valid_years": int(args.valid_years),
            "test_years": int(args.test_years),
            "first_test_year": int(args.first_test_year) if args.first_test_year is not None else None,
            "thr_min": float(args.thr_min),
            "thr_max": float(args.thr_max),
            "thr_step": float(args.thr_step),
            "fallback_threshold_on_search_fail": (
                float(args.fallback_threshold_on_search_fail)
                if args.fallback_threshold_on_search_fail is not None
                else None
            ),
            "min_val_trades": int(args.min_val_trades),
            "min_val_coverage": float(args.min_val_coverage),
            "coverage_target": float(args.coverage_target),
            "coverage_penalty": float(args.coverage_penalty),
            "val_folds": int(args.val_folds),
            "min_fold_trades": int(args.min_fold_trades),
            "min_positive_folds": int(args.min_positive_folds),
            "objective_mean_weight": float(args.objective_mean_weight),
            "objective_worst_weight": float(args.objective_worst_weight),
            "objective_total_weight": float(args.objective_total_weight),
            "objective_std_penalty": float(args.objective_std_penalty),
            "allowed_session_ids": (sorted(int(v) for v in allowed_session_ids) if allowed_session_ids else []),
            "allowed_setup_families": (sorted(str(v) for v in allowed_setups) if allowed_setups else []),
        },
        "summary": summary,
        "folds": fold_rows,
    }

    flat_rows = []
    for row in fold_rows:
        flat_rows.append(
            {
                "fold": row["fold"],
                "train_years": ",".join(str(y) for y in row["train_years"]),
                "valid_years": ",".join(str(y) for y in row["valid_years"]),
                "test_years": ",".join(str(y) for y in row["test_years"]),
                "threshold": float(row["threshold"]),
                "train_rows": int(row["train_rows"]),
                "val_rows": int(row["val_rows"]),
                "test_rows": int(row["test_rows"]),
                "validation_trade_count": int(row["validation"].get("trade_count", 0)),
                "validation_total_pnl": float(row["validation"].get("total_pnl", 0.0)),
                "validation_avg_pnl": float(row["validation"].get("avg_pnl", 0.0)),
                "test_trade_count": int(row["test"].get("trade_count", 0)),
                "test_total_pnl": float(row["test"].get("total_pnl", 0.0)),
                "test_avg_pnl": float(row["test"].get("avg_pnl", 0.0)),
                "test_win_rate": float(row["test"].get("win_rate", 0.0)),
                "test_roc_auc": float(row["test"].get("roc_auc", 0.0)),
            }
        )

    report_path.write_text(json.dumps(_json_safe(report), indent=2))
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(f"report={report_path}")
    print(f"csv={csv_path}")
    print(f"folds={len(fold_rows)}")
    print(f"total_test_pnl={summary['total_test_pnl']}")
    print(f"positive_test_folds={summary['positive_test_folds']}")
    print(f"mean_test_auc={summary['mean_test_auc']}")


if __name__ == "__main__":
    main()
