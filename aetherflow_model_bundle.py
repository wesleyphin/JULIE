from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from aetherflow_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS


BUNDLE_DESIGN_SINGLE = "single"
BUNDLE_DESIGN_FAMILY_HEADS = "family_heads"
BUNDLE_DESIGN_META_CONTEXT = "meta_context"
BUNDLE_DESIGN_ROUTED_ENSEMBLE = "routed_ensemble"


ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS = [
    "expert_0_prob",
    "expert_1_prob",
    "expert_prob_gap_01",
    "expert_prob_mean",
    "expert_prob_max",
    "setup_strength",
    "candidate_side",
    "session_id",
    "manifold_regime_id",
    "manifold_alignment_pct",
    "manifold_smoothness_pct",
    "manifold_stress_pct",
    "manifold_dispersion_pct",
    "vwap_dist_atr",
    "flow_mag_fast",
    "flow_mag_slow",
    "flow_agreement",
    "compression_score",
    "transition_energy",
    "coherence",
    "d_alignment_3",
    "d_coherence_3",
    "pressure_imbalance_30",
    "phase_regime_run_bars",
    "phase_regime_flip_count_10",
    "phase_manifold_alignment_pct_mean_5",
    "phase_manifold_stress_pct_mean_5",
    "phase_d_alignment_3_mean_5",
    "family_is_compression_release",
    "family_is_aligned_flow",
    "family_is_transition_burst",
    "family_is_exhaustion_reversal",
    "session_is_asia",
    "session_is_london",
    "session_is_nyam",
    "session_is_nypm",
    "regime_is_trend_geodesic",
    "regime_is_chop_spiral",
    "regime_is_dispersed",
    "regime_is_rotational_turbulence",
]


_CURATED_DERIVED_BY_FAMILY = {
    "compression_release": [
        "flow_fast",
        "flow_slow",
        "flow_mag_fast",
        "flow_agreement",
        "coherence",
        "compression_score",
        "expansion_score",
        "transition_energy",
        "d_alignment_3",
        "d_alignment_10",
        "d_stress_3",
        "d_dispersion_3",
        "d_r_3",
        "directional_vwap_dist",
        "alignment_minus_stress",
        "release_energy",
        "compression_release_edge",
        "compression_release_session_edge",
        "session_is_london",
        "session_is_nypm",
        "setup_strength",
        "candidate_side",
    ],
    "aligned_flow": [
        "flow_fast",
        "flow_slow",
        "flow_mag_fast",
        "flow_mag_slow",
        "flow_agreement",
        "coherence",
        "d_alignment_3",
        "d_alignment_10",
        "d_coherence_3",
        "directional_vwap_dist",
        "trend_persistence",
        "coherence_recovery",
        "aligned_flow_edge",
        "aligned_flow_ny_dispersed_edge",
        "aligned_flow_nypm_trend_edge",
        "session_is_nyam",
        "session_is_nypm",
        "regime_is_dispersed",
        "regime_is_trend_geodesic",
        "setup_strength",
        "candidate_side",
    ],
    "exhaustion_reversal": [
        "flow_fast",
        "flow_curvature",
        "extension_score",
        "d_stress_3",
        "d_coherence_3",
        "skew_20",
        "skew_60",
        "kurt_20",
        "setup_strength",
        "candidate_side",
    ],
    "transition_burst": [
        "flow_fast",
        "flow_slow",
        "flow_mag_fast",
        "flow_curvature",
        "pressure_imbalance_10",
        "pressure_imbalance_30",
        "coherence",
        "transition_energy",
        "novelty_score",
        "regime_change",
        "d_alignment_3",
        "d_stress_3",
        "d_dispersion_3",
        "d_r_3",
        "d_coherence_3",
        "burst_pressure",
        "burst_regime_shift",
        "transition_burst_edge",
        "transition_burst_nyam_chop_edge",
        "session_is_nyam",
        "regime_is_chop_spiral",
        "setup_strength",
        "candidate_side",
    ],
}


META_CONTEXT_FEATURE_COLUMNS = [
    "base_prob",
    "setup_strength",
    "candidate_side",
    "session_id",
    "manifold_regime_id",
    "manifold_alignment_pct",
    "manifold_smoothness_pct",
    "manifold_stress_pct",
    "manifold_dispersion_pct",
    "vwap_dist_atr",
    "flow_mag_fast",
    "flow_mag_slow",
    "compression_score",
    "transition_energy",
    "coherence",
    "d_alignment_3",
    "d_coherence_3",
    "family_is_compression_release",
    "family_is_aligned_flow",
    "family_is_transition_burst",
    "family_is_exhaustion_reversal",
    "session_is_asia",
    "session_is_london",
    "session_is_nyam",
    "session_is_nypm",
    "regime_is_trend_geodesic",
    "regime_is_chop_spiral",
    "regime_is_dispersed",
    "regime_is_rotational_turbulence",
]


def _coerce_ordered_columns(raw: Any, fallback: list[str], allowed_columns: list[str]) -> list[str]:
    if not isinstance(raw, list) or not raw:
        return list(fallback)
    requested = [str(col) for col in raw if str(col).strip()]
    if not requested:
        return list(fallback)
    out: list[str] = []
    seen: set[str] = set()
    for col in requested:
        if col in seen:
            continue
        out.append(col)
        seen.add(col)
    return out or list(fallback)


def _coerce_feature_columns(raw: Any, fallback: list[str]) -> list[str]:
    return _coerce_ordered_columns(raw, fallback, list(FEATURE_COLUMNS))


def _clamp_weight(value: Any, default: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    return float(min(1.0, max(0.0, out)))


def _apply_calibrator(probabilities: np.ndarray, calibrator: Any) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if not isinstance(calibrator, dict):
        return probs
    kind = str(calibrator.get("kind", "") or "").strip().lower()
    if kind != "logit_linear":
        return probs
    try:
        coef = float(calibrator.get("coef", 1.0))
        intercept = float(calibrator.get("intercept", 0.0))
    except Exception:
        return probs
    if not np.isfinite(coef) or not np.isfinite(intercept):
        return probs
    clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    calibrated_logits = (logits * float(coef)) + float(intercept)
    return 1.0 / (1.0 + np.exp(-calibrated_logits))


def _coerce_session_allowlist(value: Any) -> Optional[set[int]]:
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


def _coerce_upper_string_allowlist(value: Any) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out = {str(item).strip().upper() for item in items if str(item).strip()}
    return out if out else None


def _coerce_side_allowlist(value: Any) -> Optional[set[str]]:
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


def curated_family_feature_columns(family_name: str) -> list[str]:
    family_key = str(family_name or "").strip()
    derived = list(_CURATED_DERIVED_BY_FAMILY.get(family_key, []))
    requested = list(BASE_FEATURE_COLUMNS) + derived
    requested_set = set(requested)
    return [col for col in FEATURE_COLUMNS if col in requested_set]


def build_meta_context_frame(features: pd.DataFrame, base_probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(index=features.index.copy())
    setup_family = features.get("setup_family", pd.Series("", index=features.index)).astype(str)
    session_id = pd.to_numeric(features.get("session_id"), errors="coerce").fillna(-999.0)
    regime_id = pd.to_numeric(features.get("manifold_regime_id"), errors="coerce").fillna(-1.0)

    frame["base_prob"] = np.asarray(base_probabilities, dtype=float)
    frame["setup_strength"] = pd.to_numeric(features.get("setup_strength"), errors="coerce").fillna(0.0)
    frame["candidate_side"] = pd.to_numeric(features.get("candidate_side"), errors="coerce").fillna(0.0)
    frame["session_id"] = session_id
    frame["manifold_regime_id"] = regime_id
    frame["manifold_alignment_pct"] = pd.to_numeric(features.get("manifold_alignment_pct"), errors="coerce").fillna(0.0)
    frame["manifold_smoothness_pct"] = pd.to_numeric(features.get("manifold_smoothness_pct"), errors="coerce").fillna(0.0)
    frame["manifold_stress_pct"] = pd.to_numeric(features.get("manifold_stress_pct"), errors="coerce").fillna(0.0)
    frame["manifold_dispersion_pct"] = pd.to_numeric(features.get("manifold_dispersion_pct"), errors="coerce").fillna(0.0)
    frame["vwap_dist_atr"] = pd.to_numeric(features.get("vwap_dist_atr"), errors="coerce").fillna(0.0)
    frame["flow_mag_fast"] = pd.to_numeric(features.get("flow_mag_fast"), errors="coerce").fillna(0.0)
    frame["flow_mag_slow"] = pd.to_numeric(features.get("flow_mag_slow"), errors="coerce").fillna(0.0)
    frame["compression_score"] = pd.to_numeric(features.get("compression_score"), errors="coerce").fillna(0.0)
    frame["transition_energy"] = pd.to_numeric(features.get("transition_energy"), errors="coerce").fillna(0.0)
    frame["coherence"] = pd.to_numeric(features.get("coherence"), errors="coerce").fillna(0.0)
    frame["d_alignment_3"] = pd.to_numeric(features.get("d_alignment_3"), errors="coerce").fillna(0.0)
    frame["d_coherence_3"] = pd.to_numeric(features.get("d_coherence_3"), errors="coerce").fillna(0.0)
    frame["family_is_compression_release"] = setup_family.eq("compression_release").astype(float)
    frame["family_is_aligned_flow"] = setup_family.eq("aligned_flow").astype(float)
    frame["family_is_transition_burst"] = setup_family.eq("transition_burst").astype(float)
    frame["family_is_exhaustion_reversal"] = setup_family.eq("exhaustion_reversal").astype(float)
    frame["session_is_asia"] = session_id.eq(0.0).astype(float)
    frame["session_is_london"] = session_id.eq(1.0).astype(float)
    frame["session_is_nyam"] = session_id.eq(2.0).astype(float)
    frame["session_is_nypm"] = session_id.eq(3.0).astype(float)
    frame["regime_is_trend_geodesic"] = regime_id.eq(0.0).astype(float)
    frame["regime_is_chop_spiral"] = regime_id.eq(1.0).astype(float)
    frame["regime_is_dispersed"] = regime_id.eq(2.0).astype(float)
    frame["regime_is_rotational_turbulence"] = regime_id.eq(3.0).astype(float)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def family_feature_columns_map(*, families: Optional[list[str]] = None, mode: str = "curated") -> dict[str, list[str]]:
    mode_key = str(mode or "curated").strip().lower()
    family_names = [str(name or "").strip() for name in (families or []) if str(name or "").strip()]
    if not family_names:
        family_names = sorted(_CURATED_DERIVED_BY_FAMILY.keys())
    out: dict[str, list[str]] = {}
    for family_name in family_names:
        if mode_key == "all":
            out[family_name] = list(FEATURE_COLUMNS)
        else:
            out[family_name] = curated_family_feature_columns(family_name)
    return out


def _normalize_conditional_models(bundle: dict[str, Any], raw_feature_columns: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "family_name": str(item.get("family_name", "") or "").strip(),
            "model": item.get("model"),
            "feature_columns": _coerce_feature_columns(
                item.get("feature_columns"),
                raw_feature_columns,
            ),
            "match_session_ids": _coerce_session_allowlist(item.get("match_session_ids")),
            "match_regimes": _coerce_upper_string_allowlist(item.get("match_regimes")),
            "match_sides": _coerce_side_allowlist(item.get("match_sides")),
            "weight": _clamp_weight(item.get("weight", 1.0), 1.0),
            "calibrator": item.get("calibrator"),
        }
        for item in (bundle.get("conditional_models", []) or [])
        if isinstance(item, dict) and item.get("model") is not None
    ]


def _normalize_routed_experts(raw_experts: Any) -> list[dict[str, Any]]:
    experts: list[dict[str, Any]] = []
    if not isinstance(raw_experts, list):
        return experts
    for idx, item in enumerate(raw_experts):
        if not isinstance(item, dict):
            continue
        raw_bundle = item.get("bundle")
        if raw_bundle is None:
            raw_bundle = item.get("model_bundle", item.get("model"))
        if raw_bundle is None:
            continue
        name = str(item.get("name", "") or "").strip() or f"expert_{idx}"
        rules = [
            dict(rule)
            for rule in (item.get("activation_rules", []) or [])
            if isinstance(rule, dict)
        ]
        experts.append(
            {
                "name": name,
                "bundle": normalize_model_bundle(raw_bundle),
                "activation_rules": rules,
                "override": dict(item.get("override") or {}) if isinstance(item.get("override"), dict) else None,
            }
        )
    return experts


def normalize_model_bundle(bundle: Any) -> dict[str, Any]:
    if isinstance(bundle, dict):
        raw_feature_columns = _coerce_feature_columns(bundle.get("feature_columns"), list(FEATURE_COLUMNS))
        bundle_design = str(bundle.get("bundle_design", "") or "").strip().lower()
        if bundle_design == BUNDLE_DESIGN_ROUTED_ENSEMBLE or bundle.get("experts") is not None:
            experts = _normalize_routed_experts(bundle.get("experts", []) or [])
            router_feature_columns = _coerce_ordered_columns(
                bundle.get("router_feature_columns"),
                list(ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS),
                list(ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS),
            )
            router_context_source_columns = _coerce_ordered_columns(
                bundle.get("router_context_source_columns"),
                raw_feature_columns,
                list(FEATURE_COLUMNS),
            )
            return {
                "bundle_design": BUNDLE_DESIGN_ROUTED_ENSEMBLE,
                "model": None,
                "shared_model": None,
                "shared_feature_columns": raw_feature_columns,
                "feature_columns": raw_feature_columns,
                "family_models": {},
                "feature_columns_by_family": {},
                "conditional_models": [],
                "shared_calibrator": None,
                "family_calibrators": {},
                "family_feature_mode": str(bundle.get("family_feature_mode", "all") or "all"),
                "family_head_weight": 1.0,
                "predict_mode": str(bundle.get("predict_mode", "proba") or "proba"),
                "edge_scale": float(bundle.get("edge_scale", 1.0) or 1.0),
                "threshold": float(bundle.get("threshold", 0.58) or 0.58),
                "trained_at": bundle.get("trained_at"),
                "walkforward_fold": bundle.get("walkforward_fold"),
                "experts": experts,
                "router_model": bundle.get("router_model"),
                "router_feature_columns": router_feature_columns,
                "router_calibrator": bundle.get("router_calibrator"),
                "router_mode": str(bundle.get("router_mode", "soft_blend") or "soft_blend").strip().lower(),
                "router_weight_floor": _clamp_weight(bundle.get("router_weight_floor", 0.0), 0.0),
                "router_weight_ceiling": _clamp_weight(bundle.get("router_weight_ceiling", 1.0), 1.0),
                "router_min_top_prob": _clamp_weight(bundle.get("router_min_top_prob", 0.0), 0.0),
                "router_min_top_gap": _clamp_weight(bundle.get("router_min_top_gap", 0.0), 0.0),
                "router_context_source_columns": router_context_source_columns,
                "router_activation_rules": [
                    dict(rule)
                    for rule in (bundle.get("router_activation_rules", []) or [])
                    if isinstance(rule, dict)
                ],
                "router_fallback_expert": str(bundle.get("router_fallback_expert", "") or "").strip(),
                "router_training_report": dict(bundle.get("router_training_report") or {})
                if isinstance(bundle.get("router_training_report"), dict)
                else {},
            }
        if bundle.get("meta_model") is not None:
            meta_feature_columns = _coerce_ordered_columns(
                bundle.get("meta_feature_columns"),
                list(META_CONTEXT_FEATURE_COLUMNS),
                list(META_CONTEXT_FEATURE_COLUMNS),
            )
            return {
                "bundle_design": str(bundle.get("bundle_design", BUNDLE_DESIGN_META_CONTEXT) or BUNDLE_DESIGN_META_CONTEXT),
                "model": bundle.get("model", bundle.get("base_model")),
                "shared_model": bundle.get("model", bundle.get("base_model")),
                "shared_feature_columns": _coerce_feature_columns(
                    bundle.get("base_feature_columns", bundle.get("shared_feature_columns")),
                    raw_feature_columns,
                ),
                "base_model": bundle.get("model", bundle.get("base_model")),
                "base_feature_columns": _coerce_feature_columns(
                    bundle.get("base_feature_columns", bundle.get("shared_feature_columns")),
                    raw_feature_columns,
                ),
                "meta_model": bundle.get("meta_model"),
                "meta_feature_columns": meta_feature_columns,
                "meta_calibrator": bundle.get("meta_calibrator"),
                "meta_weight": _clamp_weight(bundle.get("meta_weight", 1.0), 1.0),
                "feature_columns": raw_feature_columns,
                "family_models": {},
                "feature_columns_by_family": {},
                "conditional_models": _normalize_conditional_models(bundle, raw_feature_columns),
                "shared_calibrator": bundle.get("shared_calibrator"),
                "family_calibrators": {},
                "family_feature_mode": str(bundle.get("family_feature_mode", "all") or "all"),
                "family_head_weight": 1.0,
                "predict_mode": str(bundle.get("predict_mode", "proba") or "proba"),
                "edge_scale": float(bundle.get("edge_scale", 1.0) or 1.0),
                "threshold": float(bundle.get("threshold", 0.58) or 0.58),
                "trained_at": bundle.get("trained_at"),
                "walkforward_fold": bundle.get("walkforward_fold"),
            }
        raw_family_models = bundle.get("family_models")
        if isinstance(raw_family_models, dict) and raw_family_models:
            family_models = {
                str(name or "").strip(): model
                for name, model in raw_family_models.items()
                if str(name or "").strip() and model is not None
            }
            raw_family_cols = bundle.get("feature_columns_by_family", {}) or {}
            feature_columns_by_family = {
                family_name: _coerce_feature_columns(
                    raw_family_cols.get(family_name),
                    curated_family_feature_columns(family_name),
                )
                for family_name in sorted(family_models.keys())
            }
            shared_model = bundle.get("model", bundle.get("shared_model"))
            shared_feature_columns = _coerce_feature_columns(
                bundle.get("shared_feature_columns"),
                raw_feature_columns,
            )
            return {
                "bundle_design": str(bundle.get("bundle_design", BUNDLE_DESIGN_FAMILY_HEADS) or BUNDLE_DESIGN_FAMILY_HEADS),
                "model": shared_model,
                "shared_model": shared_model,
                "shared_feature_columns": shared_feature_columns,
                "feature_columns": raw_feature_columns,
                "family_models": family_models,
                "feature_columns_by_family": feature_columns_by_family,
                "conditional_models": _normalize_conditional_models(bundle, raw_feature_columns),
                "shared_calibrator": bundle.get("shared_calibrator"),
                "family_calibrators": dict(bundle.get("family_calibrators", {}) or {}),
                "family_feature_mode": str(bundle.get("family_feature_mode", "curated") or "curated"),
                "family_head_weight": _clamp_weight(bundle.get("family_head_weight", 1.0), 1.0),
                "predict_mode": str(bundle.get("predict_mode", "proba") or "proba"),
                "edge_scale": float(bundle.get("edge_scale", 1.0) or 1.0),
                "threshold": float(bundle.get("threshold", 0.58) or 0.58),
                "trained_at": bundle.get("trained_at"),
                "walkforward_fold": bundle.get("walkforward_fold"),
            }
        model = bundle.get("model", bundle.get("shared_model"))
        shared_feature_columns = _coerce_feature_columns(
            bundle.get("shared_feature_columns"),
            raw_feature_columns,
        )
        return {
            "bundle_design": str(bundle.get("bundle_design", BUNDLE_DESIGN_SINGLE) or BUNDLE_DESIGN_SINGLE),
            "model": model,
            "shared_model": model,
            "shared_feature_columns": shared_feature_columns,
            "feature_columns": raw_feature_columns,
            "family_models": {},
            "feature_columns_by_family": {},
            "conditional_models": _normalize_conditional_models(bundle, raw_feature_columns),
            "shared_calibrator": bundle.get("shared_calibrator"),
            "family_calibrators": dict(bundle.get("family_calibrators", {}) or {}),
            "family_feature_mode": str(bundle.get("family_feature_mode", "all") or "all"),
            "family_head_weight": 1.0,
            "predict_mode": str(bundle.get("predict_mode", "proba") or "proba"),
            "edge_scale": float(bundle.get("edge_scale", 1.0) or 1.0),
            "threshold": float(bundle.get("threshold", 0.58) or 0.58),
            "trained_at": bundle.get("trained_at"),
            "walkforward_fold": bundle.get("walkforward_fold"),
        }

    return {
        "bundle_design": BUNDLE_DESIGN_SINGLE,
        "model": bundle,
        "shared_model": bundle,
        "shared_feature_columns": list(FEATURE_COLUMNS),
        "base_model": bundle,
        "base_feature_columns": list(FEATURE_COLUMNS),
        "meta_model": None,
        "meta_feature_columns": list(META_CONTEXT_FEATURE_COLUMNS),
        "meta_calibrator": None,
        "meta_weight": 1.0,
        "feature_columns": list(FEATURE_COLUMNS),
        "family_models": {},
        "feature_columns_by_family": {},
        "conditional_models": [],
        "shared_calibrator": None,
        "family_calibrators": {},
        "family_feature_mode": "all",
        "family_head_weight": 1.0,
        "predict_mode": "proba",
        "edge_scale": 1.0,
        "threshold": 0.58,
        "trained_at": None,
        "walkforward_fold": None,
    }


def bundle_feature_columns(bundle: Any) -> list[str]:
    normalized = normalize_model_bundle(bundle)
    if str(normalized.get("bundle_design", "") or "").strip().lower() == BUNDLE_DESIGN_ROUTED_ENSEMBLE:
        requested: list[str] = []
        requested.extend(normalized.get("feature_columns", []) or [])
        requested.extend(normalized.get("router_context_source_columns", []) or [])
        requested.extend(normalized.get("router_feature_columns", []) or [])
        for expert in (normalized.get("experts", []) or []):
            requested.extend(bundle_feature_columns(expert.get("bundle")) if isinstance(expert, dict) else [])
        requested_set = set(requested)
        ordered = [col for col in FEATURE_COLUMNS if col in requested_set]
        extras = [col for col in requested if col not in set(FEATURE_COLUMNS)]
        out: list[str] = []
        seen: set[str] = set()
        for col in ordered + extras:
            if col in seen:
                continue
            out.append(col)
            seen.add(col)
        return out or list(FEATURE_COLUMNS)
    if str(normalized.get("bundle_design", "") or "").strip().lower() == BUNDLE_DESIGN_META_CONTEXT:
        requested = list(normalized.get("feature_columns", []) or [])
        requested.extend(normalized.get("base_feature_columns", []) or [])
        requested_set = set(requested)
        ordered = [col for col in FEATURE_COLUMNS if col in requested_set]
        extras = [col for col in requested if col not in set(FEATURE_COLUMNS)]
        out: list[str] = []
        seen: set[str] = set()
        for col in ordered + extras:
            if col in seen:
                continue
            out.append(col)
            seen.add(col)
        return out or list(FEATURE_COLUMNS)
    requested: list[str] = []
    requested.extend(normalized.get("feature_columns", []) or [])
    requested.extend(normalized.get("shared_feature_columns", []) or [])
    for family_columns in (normalized.get("feature_columns_by_family", {}) or {}).values():
        requested.extend(family_columns or [])
    for conditional in (normalized.get("conditional_models", []) or []):
        requested.extend(conditional.get("feature_columns", []) or [])
    requested_set = set(requested)
    ordered = [col for col in FEATURE_COLUMNS if col in requested_set]
    extras = [col for col in requested if col not in set(FEATURE_COLUMNS)]
    out: list[str] = []
    seen: set[str] = set()
    for col in ordered + extras:
        if col in seen:
            continue
        out.append(col)
        seen.add(col)
    return out or list(FEATURE_COLUMNS)


def bundle_feature_columns_by_family(bundle: Any) -> dict[str, list[str]]:
    normalized = normalize_model_bundle(bundle)
    family_cols = dict(normalized.get("feature_columns_by_family", {}) or {})
    if family_cols:
        return family_cols
    family_names = sorted((normalized.get("family_models", {}) or {}).keys())
    if not family_names:
        return {}
    mode = str(normalized.get("family_feature_mode", "curated") or "curated")
    return family_feature_columns_map(families=family_names, mode=mode)


def bundle_family_names(bundle: Any) -> list[str]:
    normalized = normalize_model_bundle(bundle)
    family_models = normalized.get("family_models", {}) or {}
    return sorted(str(name) for name in family_models.keys() if str(name).strip())


def bundle_has_predictor(bundle: Any) -> bool:
    normalized = normalize_model_bundle(bundle)
    if str(normalized.get("bundle_design", "") or "").strip().lower() == BUNDLE_DESIGN_ROUTED_ENSEMBLE:
        experts = normalized.get("experts", []) or []
        return bool(normalized.get("router_model") is not None and experts)
    if normalized.get("shared_model") is not None:
        return True
    family_models = normalized.get("family_models", {}) or {}
    if any(model is not None for model in family_models.values()):
        return True
    conditional_models = normalized.get("conditional_models", []) or []
    return any(item.get("model") is not None for item in conditional_models if isinstance(item, dict))


def make_family_head_bundle(
    *,
    shared_model: Any,
    family_models: dict[str, Any],
    family_feature_columns: dict[str, list[str]],
    family_head_weight: float,
    threshold: float,
    trained_at: Optional[str] = None,
    walkforward_fold: Optional[str] = None,
    family_feature_mode: str = "curated",
    shared_calibrator: Optional[dict] = None,
    family_calibrators: Optional[dict[str, dict]] = None,
) -> dict[str, Any]:
    normalized_family_cols = {
        str(family_name): _coerce_feature_columns(columns, curated_family_feature_columns(str(family_name)))
        for family_name, columns in (family_feature_columns or {}).items()
        if str(family_name or "").strip()
    }
    all_features = bundle_feature_columns(
        {
            "model": shared_model,
            "feature_columns": list(FEATURE_COLUMNS),
            "family_models": dict(family_models or {}),
            "feature_columns_by_family": normalized_family_cols,
        }
    )
    return {
        "bundle_design": BUNDLE_DESIGN_FAMILY_HEADS,
        "model": shared_model,
        "shared_model": shared_model,
        "shared_feature_columns": list(FEATURE_COLUMNS),
        "feature_columns": all_features,
        "family_models": dict(family_models or {}),
        "feature_columns_by_family": normalized_family_cols,
        "shared_calibrator": dict(shared_calibrator or {}) if isinstance(shared_calibrator, dict) else None,
        "family_calibrators": {
            str(name): dict(payload or {})
            for name, payload in (family_calibrators or {}).items()
            if str(name or "").strip() and isinstance(payload, dict)
        },
        "family_feature_mode": str(family_feature_mode or "curated"),
        "family_head_weight": _clamp_weight(family_head_weight, 1.0),
        "threshold": float(threshold),
        "trained_at": trained_at,
        "walkforward_fold": walkforward_fold,
    }


def make_routed_ensemble_bundle(
    *,
    experts: list[dict[str, Any]],
    router_model: Any,
    router_feature_columns: list[str],
    threshold: float,
    trained_at: Optional[str] = None,
    walkforward_fold: Optional[str] = None,
    router_mode: str = "soft_blend",
    router_weight_floor: float = 0.0,
    router_weight_ceiling: float = 1.0,
    router_training_report: Optional[dict[str, Any]] = None,
    router_activation_rules: Optional[list[dict[str, Any]]] = None,
    router_fallback_expert: Optional[str] = None,
    router_min_top_prob: float = 0.0,
    router_min_top_gap: float = 0.0,
) -> dict[str, Any]:
    normalized_experts: list[dict[str, Any]] = []
    requested: list[str] = []
    for idx, expert in enumerate(experts or []):
        if not isinstance(expert, dict):
            continue
        raw_bundle = expert.get("bundle", expert.get("model_bundle", expert.get("model")))
        if raw_bundle is None:
            continue
        name = str(expert.get("name", "") or "").strip() or f"expert_{idx}"
        normalized_bundle = normalize_model_bundle(raw_bundle)
        requested.extend(bundle_feature_columns(normalized_bundle))
        normalized_experts.append(
            {
                "name": name,
                "bundle": normalized_bundle,
                "activation_rules": [
                    dict(rule)
                    for rule in (expert.get("activation_rules", []) or [])
                    if isinstance(rule, dict)
                ],
                "override": dict(expert.get("override") or {}) if isinstance(expert.get("override"), dict) else None,
            }
        )
    requested.extend(list(FEATURE_COLUMNS))
    feature_columns = _coerce_ordered_columns(requested, list(FEATURE_COLUMNS), list(FEATURE_COLUMNS))
    router_columns = _coerce_ordered_columns(
        list(router_feature_columns or []),
        list(ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS),
        list(ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS),
    )
    return {
        "bundle_design": BUNDLE_DESIGN_ROUTED_ENSEMBLE,
        "experts": normalized_experts,
        "router_model": router_model,
        "router_feature_columns": router_columns,
        "router_calibrator": None,
        "router_mode": str(router_mode or "soft_blend").strip().lower(),
        "router_weight_floor": _clamp_weight(router_weight_floor, 0.0),
        "router_weight_ceiling": _clamp_weight(router_weight_ceiling, 1.0),
        "router_min_top_prob": _clamp_weight(router_min_top_prob, 0.0),
        "router_min_top_gap": _clamp_weight(router_min_top_gap, 0.0),
        "router_context_source_columns": feature_columns,
        "router_activation_rules": [
            dict(rule)
            for rule in (router_activation_rules or [])
            if isinstance(rule, dict)
        ],
        "router_fallback_expert": str(router_fallback_expert or "").strip(),
        "router_training_report": dict(router_training_report or {}) if isinstance(router_training_report, dict) else {},
        "feature_columns": feature_columns,
        "threshold": float(threshold),
        "trained_at": trained_at,
        "walkforward_fold": walkforward_fold,
    }


def _predict_with_model(model: Any, features: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    x_all = (
        features.reindex(columns=feature_columns, fill_value=0.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return np.asarray(model.predict_proba(x_all)[:, 1], dtype=float)


def _predict_scores_with_mode(
    model: Any,
    features: pd.DataFrame,
    feature_columns: list[str],
    *,
    predict_mode: str,
    edge_scale: float,
) -> np.ndarray:
    mode = str(predict_mode or "proba").strip().lower()
    if hasattr(model, "predict_proba"):
        return _predict_with_model(model, features, feature_columns)
    x_all = (
        features.reindex(columns=feature_columns, fill_value=0.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    raw = np.asarray(model.predict(x_all), dtype=float)
    if mode == "edge_sigmoid":
        scale = max(1e-6, float(edge_scale or 1.0))
        clipped = np.clip(raw / scale, -8.0, 8.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    return raw


def _regime_name_series(features: pd.DataFrame) -> pd.Series:
    raw_name = features.get("manifold_regime_name")
    if raw_name is not None:
        text = pd.Series(raw_name, index=features.index).astype(str).str.strip().str.upper()
        if bool((text != "").any()):
            return text
    regime_id = pd.to_numeric(features.get("manifold_regime_id"), errors="coerce").fillna(-1).round().astype(int)
    mapping = {
        0: "TREND_GEODESIC",
        1: "CHOP_SPIRAL",
        2: "DISPERSED",
        3: "ROTATIONAL_TURBULENCE",
    }
    return regime_id.map(mapping).fillna("")


def _side_label_series(features: pd.DataFrame) -> pd.Series:
    side = pd.to_numeric(features.get("candidate_side"), errors="coerce").fillna(0.0)
    values = np.where(side.to_numpy(dtype=float) > 0.0, "LONG", np.where(side.to_numpy(dtype=float) < 0.0, "SHORT", ""))
    return pd.Series(values, index=features.index)


def _router_feature_column_from_suffix(suffix: str) -> str:
    aliases = {
        "phase_d_alignment_mean_5": "phase_d_alignment_3_mean_5",
        "phase_alignment_mean_5": "phase_manifold_alignment_pct_mean_5",
        "phase_stress_mean_5": "phase_manifold_stress_pct_mean_5",
    }
    return aliases.get(str(suffix), str(suffix))


def _router_activation_mask(features: pd.DataFrame, rules: list[dict[str, Any]]) -> np.ndarray:
    if features is None or features.empty:
        return np.zeros(0, dtype=bool)
    if not rules:
        return np.ones(len(features), dtype=bool)
    combined = pd.Series(False, index=features.index)
    family = features.get("setup_family", pd.Series("", index=features.index)).astype(str)
    session_id = pd.to_numeric(features.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
    regime_name = _regime_name_series(features)
    side_label = _side_label_series(features)
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        mask = pd.Series(True, index=features.index)
        match_families = _coerce_upper_string_allowlist(rule.get("match_families"))
        if match_families:
            mask &= family.str.upper().isin(sorted(match_families))
        match_session_ids = _coerce_session_allowlist(rule.get("match_session_ids"))
        if match_session_ids:
            mask &= session_id.isin(sorted(match_session_ids))
        match_regimes = _coerce_upper_string_allowlist(rule.get("match_regimes"))
        if match_regimes:
            mask &= regime_name.isin(sorted(match_regimes))
        match_sides = _coerce_side_allowlist(rule.get("match_sides"))
        if match_sides:
            mask &= side_label.isin(sorted(match_sides))
        for key, raw_value in rule.items():
            key_text = str(key)
            if key_text.startswith("match_min_"):
                try:
                    value = float(raw_value)
                except Exception:
                    continue
                column = _router_feature_column_from_suffix(key_text[len("match_min_") :])
                series = pd.to_numeric(features.get(column), errors="coerce").fillna(0.0)
                mask &= series >= float(value)
            elif key_text.startswith("match_max_"):
                try:
                    value = float(raw_value)
                except Exception:
                    continue
                column = _router_feature_column_from_suffix(key_text[len("match_max_") :])
                series = pd.to_numeric(features.get(column), errors="coerce").fillna(0.0)
                mask &= series <= float(value)
        combined |= mask
    return combined.to_numpy(dtype=bool)


def build_routed_ensemble_router_frame(
    features: pd.DataFrame,
    expert_probabilities: dict[str, np.ndarray],
) -> pd.DataFrame:
    if features is None:
        return pd.DataFrame()
    frame = pd.DataFrame(index=features.index.copy())
    names = list(expert_probabilities.keys())
    probs: list[np.ndarray] = []
    for idx, name in enumerate(names):
        values = np.asarray(expert_probabilities.get(name), dtype=float)
        if len(values) != len(features):
            values = np.resize(values, len(features)) if len(values) else np.zeros(len(features), dtype=float)
        values = np.clip(values, 0.0, 1.0)
        probs.append(values)
        frame[f"expert_{idx}_prob"] = values
        safe_name = str(name or f"expert_{idx}").strip().lower().replace(" ", "_")
        if safe_name:
            frame[f"expert_{safe_name}_prob"] = values
    if probs:
        matrix = np.vstack(probs).T
        frame["expert_prob_mean"] = np.mean(matrix, axis=1)
        frame["expert_prob_max"] = np.max(matrix, axis=1)
        frame["expert_prob_min"] = np.min(matrix, axis=1)
        frame["expert_prob_std"] = np.std(matrix, axis=1)
        if len(probs) >= 2:
            frame["expert_prob_gap_01"] = probs[1] - probs[0]
            frame["expert_prob_abs_gap_01"] = np.abs(probs[1] - probs[0])
    else:
        frame["expert_prob_mean"] = 0.0
        frame["expert_prob_max"] = 0.0
        frame["expert_prob_min"] = 0.0
        frame["expert_prob_std"] = 0.0

    for col in ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS:
        if col in frame.columns:
            continue
        if col in features.columns:
            frame[col] = features[col]
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _routed_expert_override_config(expert: dict[str, Any]) -> dict[str, Any]:
    override = expert.get("override") if isinstance(expert, dict) else None
    if not isinstance(override, dict) or not bool(override.get("enabled", False)):
        return {}
    out: dict[str, Any] = {"enabled": True}
    reference = str(override.get("reference_expert", "") or "").strip()
    if reference:
        out["reference_expert"] = reference
    try:
        min_prob = float(override.get("min_prob"))
    except Exception:
        min_prob = float("nan")
    if np.isfinite(min_prob):
        out["min_prob"] = float(min_prob)
    return out


def predict_bundle_probabilities(bundle: Any, features: pd.DataFrame) -> np.ndarray:
    if features is None:
        return np.asarray([], dtype=float)
    normalized = normalize_model_bundle(bundle)
    shared_model = normalized.get("shared_model")
    family_models = normalized.get("family_models", {}) or {}
    conditional_models = normalized.get("conditional_models", []) or []
    if str(normalized.get("bundle_design", "") or "").strip().lower() == BUNDLE_DESIGN_ROUTED_ENSEMBLE:
        experts = [item for item in (normalized.get("experts", []) or []) if isinstance(item, dict)]
        router_model = normalized.get("router_model")
        if router_model is None or not experts:
            return np.zeros(len(features), dtype=float)
        expert_names: list[str] = []
        expert_probs: dict[str, np.ndarray] = {}
        active_masks: list[np.ndarray] = []
        override_fallback_refs: list[str] = []
        for idx, expert in enumerate(experts):
            name = str(expert.get("name", "") or "").strip() or f"expert_{idx}"
            expert_names.append(name)
            expert_probs[name] = np.clip(
                np.asarray(predict_bundle_probabilities(expert.get("bundle"), features), dtype=float),
                0.0,
                1.0,
            )
            rules = list(expert.get("activation_rules", []) or [])
            active_mask = (
                np.asarray(_router_activation_mask(features, rules), dtype=bool).copy()
                if rules
                else np.ones(len(features), dtype=bool)
            )
            override = _routed_expert_override_config(expert)
            override_fallback_refs.append(str(override.get("reference_expert", "") or "").strip())
            if "min_prob" in override:
                active_mask &= expert_probs[name] >= float(override["min_prob"])
            active_masks.append(active_mask)
        active_matrix = np.vstack(active_masks).T if active_masks else np.ones((len(features), 0), dtype=bool)
        router_frame = build_routed_ensemble_router_frame(features, expert_probs)
        router_columns = _coerce_ordered_columns(
            normalized.get("router_feature_columns"),
            list(ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS),
            list(ROUTED_ENSEMBLE_DEFAULT_ROUTER_FEATURE_COLUMNS),
        )
        x_router = (
            router_frame.reindex(columns=router_columns, fill_value=0.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        raw_router = np.asarray(router_model.predict_proba(x_router), dtype=float)
        mode = str(normalized.get("router_mode", "soft_blend") or "soft_blend").strip().lower()
        prob_matrix = np.vstack([expert_probs[name] for name in expert_names]).T
        if mode == "hard_route":
            class_labels = getattr(router_model, "classes_", np.arange(raw_router.shape[1]))
            class_labels = np.asarray(class_labels)
            top_pos = np.argmax(raw_router, axis=1)
            top_prob = raw_router[np.arange(len(raw_router)), top_pos]
            if raw_router.shape[1] > 1:
                sorted_probs = np.sort(raw_router, axis=1)
                top_gap = sorted_probs[:, -1] - sorted_probs[:, -2]
            else:
                top_gap = np.ones(len(raw_router), dtype=float)
            selected = []
            for pos in top_pos:
                label = class_labels[int(pos)] if int(pos) < len(class_labels) else int(pos)
                try:
                    selected.append(int(label))
                except Exception:
                    selected.append(int(pos))
            selected = np.asarray(selected, dtype=int)
            fallback_name = str(normalized.get("router_fallback_expert", "") or "").strip()
            fallback_idx = expert_names.index(fallback_name) if fallback_name in expert_names else 0
            invalid = (
                (selected < 0)
                | (selected >= len(expert_names))
                | (top_prob < float(normalized.get("router_min_top_prob", 0.0) or 0.0))
                | (top_gap < float(normalized.get("router_min_top_gap", 0.0) or 0.0))
            )
            valid_selected = np.clip(selected, 0, max(0, len(expert_names) - 1))
            invalid |= ~active_matrix[np.arange(len(features)), valid_selected]
            fallback_selected = np.full(len(features), int(fallback_idx), dtype=int)
            for expert_idx, reference_name in enumerate(override_fallback_refs):
                if not reference_name or reference_name not in expert_names:
                    continue
                fallback_selected[(selected == int(expert_idx)) & invalid] = int(expert_names.index(reference_name))
            selected = np.where(invalid, fallback_selected, selected)
            return prob_matrix[np.arange(len(features)), selected]
        if raw_router.ndim == 2 and raw_router.shape[1] == len(expert_names) and len(expert_names) > 2:
            weights = raw_router * active_matrix.astype(float)
            row_sum = weights.sum(axis=1, keepdims=True)
            fallback_name = str(normalized.get("router_fallback_expert", "") or "").strip()
            fallback_idx = expert_names.index(fallback_name) if fallback_name in expert_names else 0
            empty_rows = row_sum[:, 0] <= 1e-12
            if bool(np.any(empty_rows)):
                weights[empty_rows, :] = 0.0
                weights[empty_rows, fallback_idx] = 1.0
                row_sum = weights.sum(axis=1, keepdims=True)
            weights = weights / np.maximum(row_sum, 1e-12)
            return np.sum(prob_matrix * weights, axis=1)
        if len(expert_names) >= 2:
            router_prob = raw_router[:, -1] if raw_router.ndim == 2 else np.asarray(raw_router, dtype=float)
            router_prob = _apply_calibrator(router_prob, normalized.get("router_calibrator"))
            floor = float(normalized.get("router_weight_floor", 0.0) or 0.0)
            ceiling = float(normalized.get("router_weight_ceiling", 1.0) or 1.0)
            weight = floor + ((ceiling - floor) * np.clip(router_prob, 0.0, 1.0))
            weight = np.where(active_matrix[:, 1], weight, 0.0)
            weight = np.where(active_matrix[:, 0], weight, 1.0)
            return ((1.0 - weight) * prob_matrix[:, 0]) + (weight * prob_matrix[:, 1])
        return prob_matrix[:, 0]
    if str(normalized.get("bundle_design", "") or "").strip().lower() == BUNDLE_DESIGN_META_CONTEXT:
        base_model = normalized.get("base_model")
        meta_model = normalized.get("meta_model")
        if base_model is None or meta_model is None:
            return np.zeros(len(features), dtype=float)
        base_feature_columns = _coerce_feature_columns(
            normalized.get("base_feature_columns"),
            normalized.get("feature_columns", list(FEATURE_COLUMNS)),
        )
        base_probs = _predict_scores_with_mode(
            base_model,
            features,
            base_feature_columns,
            predict_mode=normalized.get("predict_mode", "proba"),
            edge_scale=float(normalized.get("edge_scale", 1.0) or 1.0),
        )
        meta_frame = build_meta_context_frame(features, base_probs)
        meta_feature_columns = _coerce_ordered_columns(
            normalized.get("meta_feature_columns"),
            list(META_CONTEXT_FEATURE_COLUMNS),
            list(META_CONTEXT_FEATURE_COLUMNS),
        )
        meta_probs = _predict_scores_with_mode(
            meta_model,
            meta_frame,
            meta_feature_columns,
            predict_mode="proba",
            edge_scale=1.0,
        )
        meta_probs = _apply_calibrator(meta_probs, normalized.get("meta_calibrator"))
        meta_weight = _clamp_weight(normalized.get("meta_weight", 1.0), 1.0)
        return ((1.0 - meta_weight) * base_probs) + (meta_weight * meta_probs)
    if not family_models and not conditional_models:
        if shared_model is None:
            return np.zeros(len(features), dtype=float)
        shared_feature_columns = _coerce_feature_columns(
            normalized.get("shared_feature_columns"),
            normalized.get("feature_columns", list(FEATURE_COLUMNS)),
        )
        return _predict_scores_with_mode(
            shared_model,
            features,
            shared_feature_columns,
            predict_mode=normalized.get("predict_mode", "proba"),
            edge_scale=float(normalized.get("edge_scale", 1.0) or 1.0),
        )

    family_series = features.get("setup_family")
    if family_series is None:
        if shared_model is None:
            return np.zeros(len(features), dtype=float)
        shared_feature_columns = _coerce_feature_columns(
            normalized.get("shared_feature_columns"),
            normalized.get("feature_columns", list(FEATURE_COLUMNS)),
        )
        return _predict_scores_with_mode(
            shared_model,
            features,
            shared_feature_columns,
            predict_mode=normalized.get("predict_mode", "proba"),
            edge_scale=float(normalized.get("edge_scale", 1.0) or 1.0),
        )

    family_series = family_series.astype(str)
    output = np.full(len(features), np.nan, dtype=float)
    family_feature_columns = bundle_feature_columns_by_family(normalized)
    family_head_weight = _clamp_weight(normalized.get("family_head_weight", 1.0), 1.0)
    shared_calibrator = normalized.get("shared_calibrator")
    family_calibrators = dict(normalized.get("family_calibrators", {}) or {})
    shared_feature_columns = _coerce_feature_columns(
        normalized.get("shared_feature_columns"),
        normalized.get("feature_columns", list(FEATURE_COLUMNS)),
    )

    if shared_model is not None and (family_head_weight < 1.0 or conditional_models):
        shared_probs_full = _predict_scores_with_mode(
            shared_model,
            features,
            shared_feature_columns,
            predict_mode=normalized.get("predict_mode", "proba"),
            edge_scale=float(normalized.get("edge_scale", 1.0) or 1.0),
        )
    else:
        shared_probs_full = None

    if conditional_models:
        session_id = pd.to_numeric(features.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
        regime_name = _regime_name_series(features)
        side_label = _side_label_series(features)
        for item in conditional_models:
            model = item.get("model")
            if model is None:
                continue
            mask = pd.Series(True, index=features.index)
            family_name = str(item.get("family_name", "") or "").strip()
            if family_name:
                mask &= family_series.eq(family_name)
            match_session_ids = item.get("match_session_ids")
            if match_session_ids:
                mask &= session_id.isin(sorted(match_session_ids))
            match_regimes = item.get("match_regimes")
            if match_regimes:
                mask &= regime_name.isin(sorted(match_regimes))
            match_sides = item.get("match_sides")
            if match_sides:
                mask &= side_label.isin(sorted(match_sides))
            if not bool(mask.any()):
                continue
            positions = np.flatnonzero(mask.to_numpy(dtype=bool))
            frame = features.iloc[positions]
            head_probs = _predict_scores_with_mode(
                model,
                frame,
                _coerce_feature_columns(item.get("feature_columns"), normalized.get("feature_columns", list(FEATURE_COLUMNS))),
                predict_mode=normalized.get("predict_mode", "proba"),
                edge_scale=float(normalized.get("edge_scale", 1.0) or 1.0),
            )
            weight = _clamp_weight(item.get("weight", 1.0), 1.0)
            if shared_probs_full is not None and weight < 1.0:
                blended = (weight * head_probs) + ((1.0 - weight) * shared_probs_full[positions])
            else:
                blended = head_probs
            output[positions] = _apply_calibrator(blended, item.get("calibrator") or shared_calibrator)

    for family_name, family_model in family_models.items():
        family_mask = family_series.eq(str(family_name))
        family_mask &= ~pd.Series(np.isfinite(output), index=features.index)
        if not bool(family_mask.any()):
            continue
        family_positions = np.flatnonzero(family_mask.to_numpy(dtype=bool))
        family_frame = features.iloc[family_positions]
        head_columns = _coerce_feature_columns(
            family_feature_columns.get(str(family_name)),
            curated_family_feature_columns(str(family_name)),
        )
        head_probs = _predict_scores_with_mode(
            family_model,
            family_frame,
            head_columns,
            predict_mode=normalized.get("predict_mode", "proba"),
            edge_scale=float(normalized.get("edge_scale", 1.0) or 1.0),
        )
        if shared_probs_full is not None:
            blended = (family_head_weight * head_probs) + ((1.0 - family_head_weight) * shared_probs_full[family_positions])
            output[family_positions] = _apply_calibrator(
                blended,
                family_calibrators.get(str(family_name)) or shared_calibrator,
            )
        else:
            output[family_positions] = _apply_calibrator(
                head_probs,
                family_calibrators.get(str(family_name)) or shared_calibrator,
            )

    missing_mask = ~np.isfinite(output)
    if bool(np.any(missing_mask)) and shared_model is not None:
        if shared_probs_full is None:
            shared_probs_full = _predict_scores_with_mode(
                shared_model,
                features,
                shared_feature_columns,
                predict_mode=normalized.get("predict_mode", "proba"),
                edge_scale=float(normalized.get("edge_scale", 1.0) or 1.0),
            )
        output[missing_mask] = _apply_calibrator(shared_probs_full[missing_mask], shared_calibrator)

    if shared_model is not None and shared_calibrator is not None:
        unresolved_mask = ~np.isfinite(output)
        if not bool(np.any(unresolved_mask)):
            family_series = features.get("setup_family")
            if family_series is not None:
                unresolved_mask = np.asarray(
                    [str(family or "").strip() not in family_models for family in family_series.astype(str).tolist()],
                    dtype=bool,
                )
        if bool(np.any(unresolved_mask)) and shared_probs_full is not None:
            output[unresolved_mask] = _apply_calibrator(shared_probs_full[unresolved_mask], shared_calibrator)

    output[~np.isfinite(output)] = 0.0
    return output


def bundle_conditional_models_metadata(bundle: Any) -> list[dict[str, Any]]:
    normalized = normalize_model_bundle(bundle)
    out: list[dict[str, Any]] = []
    for item in (normalized.get("conditional_models", []) or []):
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "family_name": str(item.get("family_name", "") or ""),
                "feature_columns": list(item.get("feature_columns", []) or []),
                "match_session_ids": sorted(int(x) for x in (item.get("match_session_ids") or [])),
                "match_regimes": sorted(str(x).upper() for x in (item.get("match_regimes") or [])),
                "match_sides": sorted(str(x).upper() for x in (item.get("match_sides") or [])),
                "weight": float(item.get("weight", 1.0) or 1.0),
                "has_calibrator": isinstance(item.get("calibrator"), dict),
            }
        )
    return out
