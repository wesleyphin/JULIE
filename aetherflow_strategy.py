import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from aetherflow_base_cache import (
    resolve_full_manifold_base_features_path,
    validate_full_manifold_base_features_path,
)
from aetherflow_features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, build_feature_frame, resolve_setup_params
from aetherflow_model_bundle import (
    bundle_feature_columns,
    bundle_has_predictor,
    normalize_model_bundle,
    predict_bundle_probabilities,
)
from config import CONFIG
from manifold_strategy_features import build_training_feature_frame as build_manifold_base_frame
from strategy_base import Strategy

REGIME_ID_TO_NAME = {
    0: "TREND_GEODESIC",
    1: "CHOP_SPIRAL",
    2: "DISPERSED",
    3: "ROTATIONAL_TURBULENCE",
}
REGIME_NAME_TO_ID = {str(name).upper(): int(regime_id) for regime_id, name in REGIME_ID_TO_NAME.items()}
DEFAULT_LIVE_BASE_OVERLAY_FILE = "artifacts/aetherflow_live_runtime/aetherflow_manifold_runtime_tail.parquet"


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


def _coerce_string_allowlist(value) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out = {str(item).strip() for item in items if str(item).strip()}
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
    raw = _coerce_upper_string_allowlist(value)
    if not raw:
        return None
    out = {item for item in raw if item in {"LONG", "SHORT"}}
    return out if out else None


def _coerce_optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return float(out) if np.isfinite(out) else None


def _coerce_optional_int(value) -> Optional[int]:
    coerced = _coerce_optional_float(value)
    if coerced is None:
        return None
    return int(round(float(coerced)))


def _row_int(row: Dict, key: str, default: int = 0) -> int:
    coerced = _coerce_optional_int(row.get(key))
    return int(coerced) if coerced is not None else int(default)


def _row_optional_float(row: Dict, key: str) -> Optional[float]:
    if key not in row:
        return None
    return _coerce_optional_float(row.get(key))


def _normalize_early_exit_policy(value) -> Optional[dict]:
    if value is None or not isinstance(value, dict):
        return None
    return {
        "enabled": bool(value.get("enabled", False)),
        "exit_if_not_green_by": max(0, int(_coerce_optional_int(value.get("exit_if_not_green_by")) or 0)),
        "max_profit_crosses": max(0, int(_coerce_optional_int(value.get("max_profit_crosses")) or 0)),
    }


def _normalize_policy_mapping(raw_policy, *, allow_match_fields: bool, allow_rules: bool) -> dict:
    if not isinstance(raw_policy, dict):
        return {}
    policy = dict(raw_policy or {})
    out: dict = {}

    if "threshold" in policy:
        out["threshold"] = _coerce_optional_float(policy.get("threshold"))
    if "allowed_session_ids" in policy:
        out["allowed_session_ids"] = _coerce_session_allowlist(policy.get("allowed_session_ids"))
    if "allowed_regimes" in policy:
        out["allowed_regimes"] = _coerce_upper_string_allowlist(policy.get("allowed_regimes"))
    if "allowed_sides" in policy:
        out["allowed_sides"] = _coerce_side_allowlist(policy.get("allowed_sides"))
    if "blocked_regimes" in policy:
        out["blocked_regimes"] = _coerce_upper_string_allowlist(policy.get("blocked_regimes"))
    if "max_abs_vwap_dist_atr" in policy:
        out["max_abs_vwap_dist_atr"] = _coerce_optional_float(policy.get("max_abs_vwap_dist_atr"))
    if "max_directional_vwap_dist_atr" in policy:
        out["max_directional_vwap_dist_atr"] = _coerce_optional_float(policy.get("max_directional_vwap_dist_atr"))
    if "min_d_alignment_3" in policy:
        out["min_d_alignment_3"] = _coerce_optional_float(policy.get("min_d_alignment_3"))
    if "min_signed_d_alignment_3" in policy:
        out["min_signed_d_alignment_3"] = _coerce_optional_float(policy.get("min_signed_d_alignment_3"))
    if "min_d_coherence_3" in policy:
        out["min_d_coherence_3"] = _coerce_optional_float(policy.get("min_d_coherence_3"))
    if "min_signed_pressure_30" in policy:
        out["min_signed_pressure_30"] = _coerce_optional_float(policy.get("min_signed_pressure_30"))
    if "min_pressure_imbalance_30" in policy:
        out["min_pressure_imbalance_30"] = _coerce_optional_float(policy.get("min_pressure_imbalance_30"))
    if "max_pressure_imbalance_30" in policy:
        out["max_pressure_imbalance_30"] = _coerce_optional_float(policy.get("max_pressure_imbalance_30"))
    if "min_flow_agreement" in policy:
        out["min_flow_agreement"] = _coerce_optional_float(policy.get("min_flow_agreement"))
    if "min_flow_mag_slow" in policy:
        out["min_flow_mag_slow"] = _coerce_optional_float(policy.get("min_flow_mag_slow"))
    if "min_setup_strength" in policy:
        out["min_setup_strength"] = _coerce_optional_float(policy.get("min_setup_strength"))
    if "min_alignment_pct" in policy:
        out["min_alignment_pct"] = _coerce_optional_float(policy.get("min_alignment_pct"))
    if "min_smoothness_pct" in policy:
        out["min_smoothness_pct"] = _coerce_optional_float(policy.get("min_smoothness_pct"))
    if "max_stress_pct" in policy:
        out["max_stress_pct"] = _coerce_optional_float(policy.get("max_stress_pct"))
    if "max_flow_mag_slow" in policy:
        out["max_flow_mag_slow"] = _coerce_optional_float(policy.get("max_flow_mag_slow"))
    if "min_phase_regime_run_bars" in policy:
        out["min_phase_regime_run_bars"] = _coerce_optional_int(policy.get("min_phase_regime_run_bars"))
    if "max_phase_regime_flip_count_10" in policy:
        out["max_phase_regime_flip_count_10"] = _coerce_optional_int(policy.get("max_phase_regime_flip_count_10"))
    if "min_phase_alignment_mean_5" in policy:
        out["min_phase_alignment_mean_5"] = _coerce_optional_float(policy.get("min_phase_alignment_mean_5"))
    if "max_phase_stress_mean_5" in policy:
        out["max_phase_stress_mean_5"] = _coerce_optional_float(policy.get("max_phase_stress_mean_5"))
    if "min_phase_alignment_trend_5" in policy:
        out["min_phase_alignment_trend_5"] = _coerce_optional_float(policy.get("min_phase_alignment_trend_5"))
    if "max_phase_stress_trend_5" in policy:
        out["max_phase_stress_trend_5"] = _coerce_optional_float(policy.get("max_phase_stress_trend_5"))
    if "min_phase_d_alignment_mean_5" in policy:
        out["min_phase_d_alignment_mean_5"] = _coerce_optional_float(policy.get("min_phase_d_alignment_mean_5"))
    if "max_sl_dist" in policy:
        out["max_sl_dist"] = _coerce_optional_float(policy.get("max_sl_dist"))
    if "max_tp_dist" in policy:
        out["max_tp_dist"] = _coerce_optional_float(policy.get("max_tp_dist"))
    if "selection_score_bias" in policy:
        out["selection_score_bias"] = _coerce_optional_float(policy.get("selection_score_bias"))
    if "selection_score_scale" in policy:
        out["selection_score_scale"] = _coerce_optional_float(policy.get("selection_score_scale"))
    if "size_multiplier" in policy:
        out["size_multiplier"] = _coerce_optional_float(policy.get("size_multiplier"))
    if "entry_mode" in policy:
        out["entry_mode"] = str(policy.get("entry_mode", "market_next_bar") or "market_next_bar").strip().lower()
    if "sl_mult_override" in policy:
        out["sl_mult_override"] = _coerce_optional_float(policy.get("sl_mult_override"))
    if "tp_mult_override" in policy:
        out["tp_mult_override"] = _coerce_optional_float(policy.get("tp_mult_override"))
    if "horizon_bars_override" in policy:
        out["horizon_bars_override"] = _coerce_optional_int(policy.get("horizon_bars_override"))
    if "use_horizon_time_stop" in policy:
        out["use_horizon_time_stop"] = bool(policy.get("use_horizon_time_stop", False))
    if "early_exit" in policy:
        out["early_exit"] = _normalize_early_exit_policy(policy.get("early_exit")) or {}

    if allow_match_fields:
        if "name" in policy and str(policy.get("name", "") or "").strip():
            out["name"] = str(policy.get("name", "") or "").strip()
        if "match_session_ids" in policy:
            out["match_session_ids"] = _coerce_session_allowlist(policy.get("match_session_ids"))
        if "match_regimes" in policy:
            out["match_regimes"] = _coerce_upper_string_allowlist(policy.get("match_regimes"))
        if "match_sides" in policy:
            out["match_sides"] = _coerce_side_allowlist(policy.get("match_sides"))
        if "match_max_phase_regime_run_bars" in policy:
            out["match_max_phase_regime_run_bars"] = _coerce_optional_int(policy.get("match_max_phase_regime_run_bars"))
        if "match_min_phase_regime_flip_count_10" in policy:
            out["match_min_phase_regime_flip_count_10"] = _coerce_optional_int(policy.get("match_min_phase_regime_flip_count_10"))
        if "match_min_phase_regime_run_bars" in policy:
            out["match_min_phase_regime_run_bars"] = _coerce_optional_int(policy.get("match_min_phase_regime_run_bars"))
        if "match_max_phase_regime_flip_count_10" in policy:
            out["match_max_phase_regime_flip_count_10"] = _coerce_optional_int(policy.get("match_max_phase_regime_flip_count_10"))
        if "match_max_phase_alignment_mean_5" in policy:
            out["match_max_phase_alignment_mean_5"] = _coerce_optional_float(policy.get("match_max_phase_alignment_mean_5"))
        if "match_min_phase_stress_mean_5" in policy:
            out["match_min_phase_stress_mean_5"] = _coerce_optional_float(policy.get("match_min_phase_stress_mean_5"))
        if "match_min_phase_alignment_mean_5" in policy:
            out["match_min_phase_alignment_mean_5"] = _coerce_optional_float(policy.get("match_min_phase_alignment_mean_5"))
        if "match_max_phase_stress_mean_5" in policy:
            out["match_max_phase_stress_mean_5"] = _coerce_optional_float(policy.get("match_max_phase_stress_mean_5"))
        if "match_max_phase_d_alignment_mean_5" in policy:
            out["match_max_phase_d_alignment_mean_5"] = _coerce_optional_float(policy.get("match_max_phase_d_alignment_mean_5"))
        if "match_min_phase_d_alignment_mean_5" in policy:
            out["match_min_phase_d_alignment_mean_5"] = _coerce_optional_float(policy.get("match_min_phase_d_alignment_mean_5"))
        if "match_min_flow_agreement" in policy:
            out["match_min_flow_agreement"] = _coerce_optional_float(policy.get("match_min_flow_agreement"))
        if "match_max_flow_agreement" in policy:
            out["match_max_flow_agreement"] = _coerce_optional_float(policy.get("match_max_flow_agreement"))

    if allow_rules:
        raw_rules = policy.get("policy_rules", policy.get("rules"))
        if isinstance(raw_rules, list):
            out["policy_rules"] = [
                normalized_rule
                for item in raw_rules
                if (normalized_rule := _normalize_policy_mapping(item, allow_match_fields=True, allow_rules=False))
            ]

    return out


def _merge_policy_layers(*layers: dict) -> dict:
    merged: dict = {}
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        for key, value in layer.items():
            if key in {"policy_rules", "rules", "match_session_ids", "match_regimes", "match_sides", "name"}:
                continue
            if key == "early_exit":
                merged[key] = dict(value or {})
            elif isinstance(value, set):
                merged[key] = set(value)
            else:
                merged[key] = value
    return merged


def _policy_rule_matches_row(rule: Dict, row: Dict) -> bool:
    if not isinstance(rule, dict):
        return False
    session_id = _row_int(row, "session_id", default=-999)
    if "match_session_ids" in rule:
        match_session_ids = rule.get("match_session_ids")
        if match_session_ids and session_id not in match_session_ids:
            return False
    regime_name = _regime_name_from_row(row)
    if "match_regimes" in rule:
        match_regimes = rule.get("match_regimes")
        if match_regimes and regime_name not in match_regimes:
            return False
    side_num = _row_int(row, "candidate_side", default=0)
    side_label = "LONG" if side_num > 0 else "SHORT" if side_num < 0 else None
    if "match_sides" in rule:
        match_sides = rule.get("match_sides")
        if match_sides and side_label not in match_sides:
            return False
    match_min_phase_run = rule.get("match_min_phase_regime_run_bars")
    if match_min_phase_run is not None:
        phase_run = _row_optional_float(row, "phase_regime_run_bars")
        if phase_run is None or phase_run < float(match_min_phase_run):
            return False
    match_max_phase_run = rule.get("match_max_phase_regime_run_bars")
    if match_max_phase_run is not None:
        phase_run = _row_optional_float(row, "phase_regime_run_bars")
        if phase_run is None or phase_run > float(match_max_phase_run):
            return False
    match_min_phase_flips = rule.get("match_min_phase_regime_flip_count_10")
    if match_min_phase_flips is not None:
        phase_flips = _row_optional_float(row, "phase_regime_flip_count_10")
        if phase_flips is None or phase_flips < float(match_min_phase_flips):
            return False
    match_max_phase_flips = rule.get("match_max_phase_regime_flip_count_10")
    if match_max_phase_flips is not None:
        phase_flips = _row_optional_float(row, "phase_regime_flip_count_10")
        if phase_flips is None or phase_flips > float(match_max_phase_flips):
            return False
    match_max_phase_alignment = rule.get("match_max_phase_alignment_mean_5")
    if match_max_phase_alignment is not None:
        phase_alignment = _row_optional_float(row, "phase_manifold_alignment_pct_mean_5")
        if phase_alignment is None or phase_alignment > float(match_max_phase_alignment):
            return False
    match_min_phase_alignment = rule.get("match_min_phase_alignment_mean_5")
    if match_min_phase_alignment is not None:
        phase_alignment = _row_optional_float(row, "phase_manifold_alignment_pct_mean_5")
        if phase_alignment is None or phase_alignment < float(match_min_phase_alignment):
            return False
    match_min_phase_stress = rule.get("match_min_phase_stress_mean_5")
    if match_min_phase_stress is not None:
        phase_stress = _row_optional_float(row, "phase_manifold_stress_pct_mean_5")
        if phase_stress is None or phase_stress < float(match_min_phase_stress):
            return False
    match_max_phase_stress = rule.get("match_max_phase_stress_mean_5")
    if match_max_phase_stress is not None:
        phase_stress = _row_optional_float(row, "phase_manifold_stress_pct_mean_5")
        if phase_stress is None or phase_stress > float(match_max_phase_stress):
            return False
    match_max_phase_d_alignment = rule.get("match_max_phase_d_alignment_mean_5")
    if match_max_phase_d_alignment is not None:
        phase_d_alignment = _row_optional_float(row, "phase_d_alignment_3_mean_5")
        if phase_d_alignment is None or phase_d_alignment > float(match_max_phase_d_alignment):
            return False
    match_min_phase_d_alignment = rule.get("match_min_phase_d_alignment_mean_5")
    if match_min_phase_d_alignment is not None:
        phase_d_alignment = _row_optional_float(row, "phase_d_alignment_3_mean_5")
        if phase_d_alignment is None or phase_d_alignment < float(match_min_phase_d_alignment):
            return False
    match_min_flow_agreement = rule.get("match_min_flow_agreement")
    if match_min_flow_agreement is not None:
        flow_agreement = _row_optional_float(row, "flow_agreement")
        if flow_agreement is None or flow_agreement < float(match_min_flow_agreement):
            return False
    match_max_flow_agreement = rule.get("match_max_flow_agreement")
    if match_max_flow_agreement is not None:
        flow_agreement = _row_optional_float(row, "flow_agreement")
        if flow_agreement is None or flow_agreement > float(match_max_flow_agreement):
            return False
    return True


def _normalize_family_policies(value) -> dict[str, dict]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, dict] = {}
    for family_name, raw_policy in value.items():
        family_key = str(family_name or "").strip()
        if not family_key:
            continue
        out[family_key] = _normalize_policy_mapping(raw_policy, allow_match_fields=False, allow_rules=True)
    return out


def _regime_name_from_row(row: Dict) -> str:
    raw_name = str(row.get("manifold_regime_name", "") or row.get("manifold_regime", "") or "").strip().upper()
    if raw_name:
        return raw_name
    regime_id = _row_int(row, "manifold_regime_id", default=-1)
    return str(REGIME_ID_TO_NAME.get(regime_id, "") or "")


def augment_aetherflow_phase_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    work = frame.sort_index(kind="mergesort").copy()
    if "manifold_regime_id" in work.columns:
        regime_id = pd.to_numeric(work.get("manifold_regime_id"), errors="coerce")
    else:
        regime_name = work.get("manifold_regime_name")
        if regime_name is None:
            regime_name = work.get("manifold_regime")
        regime_id = pd.Series(regime_name, index=work.index).astype(str).str.upper().map(REGIME_NAME_TO_ID)
    regime_id = regime_id.fillna(-1).round().astype(int)
    regime_change = regime_id.ne(regime_id.shift(1)).fillna(True)
    group_id = regime_change.cumsum()
    work["phase_regime_run_bars"] = group_id.groupby(group_id).cumcount() + 1
    work["phase_regime_flip_count_10"] = regime_change.astype(int).rolling(10, min_periods=1).sum()

    def _series(name: str) -> pd.Series:
        if name in work.columns:
            raw = work[name]
        else:
            raw = pd.Series(0.0, index=work.index)
        return pd.to_numeric(raw, errors="coerce").fillna(0.0)

    for col in [
        "manifold_alignment_pct",
        "manifold_smoothness_pct",
        "manifold_stress_pct",
        "manifold_dispersion_pct",
        "d_alignment_3",
        "d_stress_3",
        "d_dispersion_3",
        "flow_agreement",
        "flow_mag_slow",
        "pressure_imbalance_30",
    ]:
        series = _series(col)
        work[f"phase_{col}_mean_5"] = series.rolling(5, min_periods=1).mean()
        work[f"phase_{col}_trend_5"] = series - series.shift(5).fillna(series.iloc[0] if len(series) else 0.0)
    return work


def _row_float(row: Dict, key: str, default: float = 0.0) -> float:
    try:
        out = float(row.get(key, default))
    except Exception:
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _selection_score(confidence: float, policy: Optional[Dict]) -> float:
    policy = dict(policy or {})
    raw_scale = _coerce_optional_float(policy.get("selection_score_scale"))
    raw_bias = _coerce_optional_float(policy.get("selection_score_bias"))
    scale = float(raw_scale) if raw_scale is not None and raw_scale > 0.0 else 1.0
    bias = float(raw_bias) if raw_bias is not None else 0.0
    return float(float(confidence) * scale + bias)


def _context_block_reason(row: Dict, policy: Dict) -> str:
    max_abs_vwap_dist_atr = policy.get("max_abs_vwap_dist_atr")
    if max_abs_vwap_dist_atr is not None:
        if abs(_row_float(row, "vwap_dist_atr", 0.0)) > float(max_abs_vwap_dist_atr):
            return "abs_vwap_too_far"

    max_directional_vwap_dist_atr = policy.get("max_directional_vwap_dist_atr")
    if max_directional_vwap_dist_atr is not None:
        directional_vwap = _row_float(row, "candidate_side", 0.0) * _row_float(row, "vwap_dist_atr", 0.0)
        if directional_vwap > float(max_directional_vwap_dist_atr):
            return "directional_vwap_too_far"

    min_d_alignment_3 = policy.get("min_d_alignment_3")
    if min_d_alignment_3 is not None and _row_float(row, "d_alignment_3", 0.0) < float(min_d_alignment_3):
        return "d_alignment_too_low"

    min_signed_d_alignment_3 = policy.get("min_signed_d_alignment_3")
    if min_signed_d_alignment_3 is not None:
        signed_d_alignment = _row_float(row, "candidate_side", 0.0) * _row_float(row, "d_alignment_3", 0.0)
        if signed_d_alignment < float(min_signed_d_alignment_3):
            return "signed_d_alignment_too_low"

    min_d_coherence_3 = policy.get("min_d_coherence_3")
    if min_d_coherence_3 is not None and _row_float(row, "d_coherence_3", 0.0) < float(min_d_coherence_3):
        return "d_coherence_too_low"

    min_signed_pressure_30 = policy.get("min_signed_pressure_30")
    if min_signed_pressure_30 is not None:
        signed_pressure = _row_float(row, "candidate_side", 0.0) * _row_float(row, "pressure_imbalance_30", 0.0)
        if signed_pressure < float(min_signed_pressure_30):
            return "signed_pressure_too_low"

    min_pressure_imbalance_30 = policy.get("min_pressure_imbalance_30")
    if min_pressure_imbalance_30 is not None and _row_float(row, "pressure_imbalance_30", 0.0) < float(min_pressure_imbalance_30):
        return "pressure_imbalance_too_low"

    max_pressure_imbalance_30 = policy.get("max_pressure_imbalance_30")
    if max_pressure_imbalance_30 is not None and _row_float(row, "pressure_imbalance_30", 0.0) > float(max_pressure_imbalance_30):
        return "pressure_imbalance_too_high"

    min_flow_agreement = policy.get("min_flow_agreement")
    if min_flow_agreement is not None and _row_float(row, "flow_agreement", 0.0) < float(min_flow_agreement):
        return "flow_agreement_too_low"

    min_flow_mag_slow = policy.get("min_flow_mag_slow")
    if min_flow_mag_slow is not None and _row_float(row, "flow_mag_slow", 0.0) < float(min_flow_mag_slow):
        return "flow_mag_slow_too_low"

    min_setup_strength = policy.get("min_setup_strength")
    if min_setup_strength is not None and _row_float(row, "setup_strength", 0.0) < float(min_setup_strength):
        return "setup_strength_too_low"

    min_alignment_pct = policy.get("min_alignment_pct")
    if min_alignment_pct is not None and _row_float(row, "manifold_alignment_pct", 0.0) < float(min_alignment_pct):
        return "alignment_pct_too_low"

    min_smoothness_pct = policy.get("min_smoothness_pct")
    if min_smoothness_pct is not None and _row_float(row, "manifold_smoothness_pct", 0.0) < float(min_smoothness_pct):
        return "smoothness_pct_too_low"

    max_stress_pct = policy.get("max_stress_pct")
    if max_stress_pct is not None and _row_float(row, "manifold_stress_pct", 0.0) > float(max_stress_pct):
        return "stress_pct_too_high"

    max_flow_mag_slow = policy.get("max_flow_mag_slow")
    if max_flow_mag_slow is not None and _row_float(row, "flow_mag_slow", 0.0) > float(max_flow_mag_slow):
        return "flow_mag_slow_too_high"

    min_phase_regime_run_bars = policy.get("min_phase_regime_run_bars")
    if min_phase_regime_run_bars is not None:
        phase_run = _row_optional_float(row, "phase_regime_run_bars")
        if phase_run is not None and phase_run < float(min_phase_regime_run_bars):
            return "phase_regime_run_too_short"

    max_phase_regime_flip_count_10 = policy.get("max_phase_regime_flip_count_10")
    if max_phase_regime_flip_count_10 is not None:
        phase_flips = _row_optional_float(row, "phase_regime_flip_count_10")
        if phase_flips is not None and phase_flips > float(max_phase_regime_flip_count_10):
            return "phase_regime_flip_too_high"

    min_phase_alignment_mean_5 = policy.get("min_phase_alignment_mean_5")
    if min_phase_alignment_mean_5 is not None:
        phase_alignment = _row_optional_float(row, "phase_manifold_alignment_pct_mean_5")
        if phase_alignment is not None and phase_alignment < float(min_phase_alignment_mean_5):
            return "phase_alignment_mean_too_low"

    max_phase_stress_mean_5 = policy.get("max_phase_stress_mean_5")
    if max_phase_stress_mean_5 is not None:
        phase_stress = _row_optional_float(row, "phase_manifold_stress_pct_mean_5")
        if phase_stress is not None and phase_stress > float(max_phase_stress_mean_5):
            return "phase_stress_mean_too_high"

    min_phase_alignment_trend_5 = policy.get("min_phase_alignment_trend_5")
    if min_phase_alignment_trend_5 is not None:
        phase_alignment_trend = _row_optional_float(row, "phase_manifold_alignment_pct_trend_5")
        if phase_alignment_trend is not None and phase_alignment_trend < float(min_phase_alignment_trend_5):
            return "phase_alignment_trend_too_low"

    max_phase_stress_trend_5 = policy.get("max_phase_stress_trend_5")
    if max_phase_stress_trend_5 is not None:
        phase_stress_trend = _row_optional_float(row, "phase_manifold_stress_pct_trend_5")
        if phase_stress_trend is not None and phase_stress_trend > float(max_phase_stress_trend_5):
            return "phase_stress_trend_too_high"

    min_phase_d_alignment_mean_5 = policy.get("min_phase_d_alignment_mean_5")
    if min_phase_d_alignment_mean_5 is not None:
        phase_d_alignment = _row_optional_float(row, "phase_d_alignment_3_mean_5")
        if phase_d_alignment is not None and phase_d_alignment < float(min_phase_d_alignment_mean_5):
            return "phase_d_alignment_mean_too_low"

    return ""


def _params_block_reason(row: Dict, policy: Dict, *, params: Optional[Dict] = None) -> str:
    max_sl_dist = policy.get("max_sl_dist")
    max_tp_dist = policy.get("max_tp_dist")
    if max_sl_dist is None and max_tp_dist is None:
        return ""
    if params is None:
        params_row = pd.Series(dict(row))
        sl_mult_override = policy.get("sl_mult_override")
        tp_mult_override = policy.get("tp_mult_override")
        horizon_bars_override = policy.get("horizon_bars_override")
        if sl_mult_override is not None:
            params_row["setup_sl_mult"] = float(sl_mult_override)
        if tp_mult_override is not None:
            params_row["setup_tp_mult"] = float(tp_mult_override)
        if horizon_bars_override is not None:
            params_row["setup_horizon_bars"] = int(horizon_bars_override)
        params = resolve_setup_params(params_row)
    sl_points = _coerce_optional_float((params or {}).get("sl_points"))
    if max_sl_dist is not None and sl_points is not None and sl_points > float(max_sl_dist):
        return "sl_dist_too_wide"
    tp_points = _coerce_optional_float((params or {}).get("tp_points"))
    if max_tp_dist is not None and tp_points is not None and tp_points > float(max_tp_dist):
        return "tp_dist_too_wide"
    return ""


class AetherFlowStrategy(Strategy):
    def __init__(self):
        self.cfg = dict(CONFIG.get("AETHERFLOW_STRATEGY", {}) or {})
        self.model_path = Path(self.cfg.get("model_file", "model_aetherflow_v1.pkl"))
        self.thresholds_path = Path(self.cfg.get("thresholds_file", "aetherflow_thresholds_v1.json"))
        self.metrics_path = Path(self.cfg.get("metrics_file", "aetherflow_metrics_v1.json"))
        self.min_bars = int(self.cfg.get("min_bars", 320) or 320)
        self.min_confidence = float(self.cfg.get("min_confidence", 0.0) or 0.0)
        threshold_override = self.cfg.get("threshold_override", None)
        try:
            self.threshold_override = float(threshold_override) if threshold_override is not None else None
        except Exception:
            self.threshold_override = None
        self.size = int(self.cfg.get("size", 5) or 5)
        self.log_evals = bool(self.cfg.get("log_evals", False))
        self.max_feature_bars = max(self.min_bars + 64, int(self.cfg.get("max_feature_bars", 900) or 900))
        self.backtest_base_features_path = resolve_full_manifold_base_features_path(
            self.cfg.get("backtest_base_features_file", "")
        )
        self.backtest_precomputed_features_path = self._resolve_local_path(
            self.cfg.get("backtest_precomputed_features_file", "")
        )
        try:
            overlap_bars = int(
                self.cfg.get("backtest_precompute_overlap_bars", self.max_feature_bars + 128)
                or (self.max_feature_bars + 128)
            )
        except Exception:
            overlap_bars = self.max_feature_bars + 128
        self.backtest_precompute_overlap_bars = max(self.max_feature_bars, overlap_bars)
        self.live_base_history_bars = max(
            self.max_feature_bars,
            int(self.cfg.get("live_base_history_bars", self.max_feature_bars + 128) or (self.max_feature_bars + 128)),
        )
        self.live_base_overlap_bars = max(
            self.live_base_history_bars,
            int(self.cfg.get("live_base_overlap_bars", self.backtest_precompute_overlap_bars) or self.backtest_precompute_overlap_bars),
        )
        self.allowed_session_ids: Optional[set[int]] = _coerce_session_allowlist(self.cfg.get("allowed_session_ids"))
        self.allowed_setup_families: Optional[set[str]] = _coerce_string_allowlist(self.cfg.get("allowed_setup_families"))
        self.hazard_block_regimes = {
            str(item).strip().upper() for item in (self.cfg.get("hazard_block_regimes", []) or []) if str(item).strip()
        }
        self.family_policies = _normalize_family_policies(self.cfg.get("family_policies"))
        self.model = None
        self.model_bundle = None
        self.model_loaded = False
        self.threshold = 0.58
        self.feature_columns = list(FEATURE_COLUMNS)
        self.last_eval: Optional[Dict] = None
        self._pending_runtime_event: Optional[Dict] = None
        self._last_runtime_event_signature: Optional[tuple] = None
        self._precomputed_backtest_df: Optional[pd.DataFrame] = None
        self._precomputed_lookup: dict[int, dict] = {}
        self._precomputed_known_timestamps: set[int] = set()
        self._live_base_features_path: Optional[Path] = self._discover_live_base_features_path()
        self._live_base_overlay_path: Optional[Path] = self._resolve_local_path(
            self.cfg.get("live_base_overlay_file", DEFAULT_LIVE_BASE_OVERLAY_FILE)
        )
        self._live_base_features_validated = False
        self._live_base_overlay_validated = False
        self._live_base_window: Optional[pd.DataFrame] = None
        self._load_artifacts()

    def _resolve_local_path(self, value) -> Optional[Path]:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        return path.resolve()

    def _load_artifacts(self) -> None:
        threshold_loaded = False
        threshold_payload: dict = {}
        if self.thresholds_path.exists():
            try:
                payload = json.loads(self.thresholds_path.read_text())
                threshold_payload = payload if isinstance(payload, dict) else {}
                self.threshold = float(threshold_payload.get("threshold", self.threshold) or self.threshold)
                threshold_loaded = True
                feat_cols = threshold_payload.get("feature_columns")
                if isinstance(feat_cols, list) and feat_cols:
                    self.feature_columns = [str(col) for col in feat_cols]
                if not self.family_policies:
                    payload_family_policies = _normalize_family_policies(threshold_payload.get("family_policies"))
                    if payload_family_policies:
                        self.family_policies = payload_family_policies
                if self.allowed_setup_families is None:
                    payload_allow = _coerce_string_allowlist(threshold_payload.get("allowed_setup_families"))
                    if payload_allow:
                        self.allowed_setup_families = payload_allow
                if not self.hazard_block_regimes:
                    payload_block = threshold_payload.get("hazard_block_regimes", []) or []
                    self.hazard_block_regimes = {
                        str(item).strip().upper() for item in payload_block if str(item).strip()
                    }
                logging.info(
                    "AetherFlowStrategy thresholds loaded: %s (threshold=%.3f)",
                    self.thresholds_path,
                    self.threshold,
                )
            except Exception as exc:
                logging.warning("AetherFlowStrategy threshold load failed: %s", exc)
        if self.threshold_override is not None and np.isfinite(self.threshold_override):
            self.threshold = float(self.threshold_override)
            logging.info(
                "AetherFlowStrategy threshold override active: %.3f",
                self.threshold,
            )
        if not self.model_path.exists():
            logging.warning("AetherFlowStrategy model artifact missing: %s", self.model_path)
            return
        try:
            with self.model_path.open("rb") as fh:
                raw_bundle = pickle.load(fh)
            self.model_bundle = normalize_model_bundle(raw_bundle)
            self.model = self.model_bundle.get("shared_model")
            self.feature_columns = bundle_feature_columns(self.model_bundle)
            if not threshold_loaded:
                self.threshold = float(self.model_bundle.get("threshold", self.threshold) or self.threshold)
            self.model_loaded = bundle_has_predictor(self.model_bundle)
            if threshold_payload:
                mismatch_parts: list[str] = []
                expected_bundle_design = str(threshold_payload.get("bundle_design", "") or "").strip().lower()
                actual_bundle_design = str(self.model_bundle.get("bundle_design", "") or "").strip().lower()
                if expected_bundle_design and expected_bundle_design != actual_bundle_design:
                    mismatch_parts.append(
                        f"bundle_design thresholds={expected_bundle_design} model={actual_bundle_design}"
                    )
                expected_feature_mode = str(threshold_payload.get("family_feature_mode", "") or "").strip().lower()
                actual_feature_mode = str(self.model_bundle.get("family_feature_mode", "") or "").strip().lower()
                if expected_feature_mode and expected_feature_mode != actual_feature_mode:
                    mismatch_parts.append(
                        f"family_feature_mode thresholds={expected_feature_mode} model={actual_feature_mode}"
                    )
                expected_head_weight = threshold_payload.get("family_head_weight")
                try:
                    expected_head_weight = float(expected_head_weight) if expected_head_weight is not None else None
                except Exception:
                    expected_head_weight = None
                actual_head_weight = float(self.model_bundle.get("family_head_weight", 1.0) or 1.0)
                if expected_head_weight is not None and not np.isclose(expected_head_weight, actual_head_weight):
                    mismatch_parts.append(
                        f"family_head_weight thresholds={expected_head_weight:.3f} model={actual_head_weight:.3f}"
                    )
                payload_family_heads = threshold_payload.get("family_heads")
                if isinstance(payload_family_heads, dict):
                    payload_head_names = sorted(
                        str(name).strip()
                        for name in payload_family_heads.keys()
                        if str(name).strip()
                    )
                    model_head_names = sorted(
                        str(name).strip()
                        for name in (self.model_bundle.get("family_models", {}) or {}).keys()
                        if str(name).strip()
                    )
                    if payload_head_names != model_head_names:
                        mismatch_parts.append(
                            f"family_heads thresholds={payload_head_names or ['none']} model={model_head_names or ['none']}"
                        )
                payload_conditional = threshold_payload.get("conditional_models")
                model_conditional_count = len(self.model_bundle.get("conditional_models", []) or [])
                if isinstance(payload_conditional, list):
                    if len(payload_conditional) != model_conditional_count:
                        mismatch_parts.append(
                            f"conditional_models thresholds={len(payload_conditional)} model={model_conditional_count}"
                        )
                elif model_conditional_count:
                    mismatch_parts.append(
                        f"conditional_models thresholds=missing model={model_conditional_count}"
                    )
                if mismatch_parts:
                    logging.warning(
                        "AetherFlowStrategy artifact metadata mismatch: %s",
                        "; ".join(mismatch_parts),
                    )
            if self.model_loaded:
                logging.info("AetherFlowStrategy model loaded: %s", self.model_path)
        except Exception as exc:
            logging.error("AetherFlowStrategy artifact load failed: %s", exc)
            self.model = None
            self.model_bundle = None
            self.model_loaded = False

    def set_precomputed_backtest_df(
        self,
        df: Optional[pd.DataFrame],
        *,
        known_index: Optional[pd.Index] = None,
    ) -> None:
        self._precomputed_backtest_df = None if df is None else df.copy()
        self._precomputed_lookup = {}
        self._precomputed_known_timestamps = set()
        if known_index is not None:
            try:
                known_dt_index = pd.DatetimeIndex(known_index)
                self._precomputed_known_timestamps = {
                    int(ts.value) for ts in known_dt_index
                }
            except Exception:
                self._precomputed_known_timestamps = set()
        if not isinstance(self._precomputed_backtest_df, pd.DataFrame) or self._precomputed_backtest_df.empty:
            return
        rows = self._precomputed_backtest_df.to_dict("records")
        for ts, row in zip(pd.DatetimeIndex(self._precomputed_backtest_df.index), rows):
            self._precomputed_lookup[int(ts.value)] = row

    def _load_seed_features(
        self,
        *,
        start_time: Optional[pd.Timestamp],
        end_time: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        seed_path = self.backtest_precomputed_features_path
        if seed_path is None or not seed_path.exists():
            return pd.DataFrame()
        try:
            seed = pd.read_parquet(seed_path)
        except Exception as exc:
            logging.warning("AetherFlowStrategy seed feature load failed: %s", exc)
            return pd.DataFrame()
        if seed.empty:
            return pd.DataFrame()
        try:
            seed.index = pd.DatetimeIndex(seed.index)
        except Exception:
            logging.warning(
                "AetherFlowStrategy seed feature index is not datetime-like: %s",
                seed_path,
            )
            return pd.DataFrame()
        seed = seed.sort_index()
        if start_time is not None:
            seed = seed.loc[seed.index >= pd.Timestamp(start_time)]
        if end_time is not None:
            seed = seed.loc[seed.index <= pd.Timestamp(end_time)]
        if seed.empty:
            return pd.DataFrame()
        return seed.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _seeded_feature_window(
        self,
        df: pd.DataFrame,
        *,
        start_time: Optional[pd.Timestamp],
        end_time: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        source_df = df.sort_index()
        target_end = pd.Timestamp(end_time) if end_time is not None else pd.Timestamp(source_df.index.max())
        source_df = source_df.loc[source_df.index <= target_end]
        if source_df.empty:
            return pd.DataFrame()

        if start_time is not None:
            start_pos = max(
                0,
                int(source_df.index.searchsorted(pd.Timestamp(start_time), side="left"))
                - int(self.backtest_precompute_overlap_bars),
            )
        else:
            start_pos = max(0, len(source_df) - int(self.backtest_precompute_overlap_bars))
        feature_start = pd.Timestamp(source_df.index[start_pos])

        seed_df = self._load_seed_features(start_time=feature_start, end_time=target_end)
        seed_last_ts = (
            pd.Timestamp(seed_df.index.max())
            if isinstance(seed_df, pd.DataFrame) and not seed_df.empty
            else None
        )
        if seed_last_ts is not None and seed_last_ts >= target_end:
            logging.info(
                "AetherFlowStrategy backtest seed fully covers requested window: rows=%s range=%s->%s",
                len(seed_df),
                seed_df.index.min(),
                seed_df.index.max(),
            )
            return seed_df

        if seed_last_ts is not None:
            build_start_pos = max(
                0,
                int(source_df.index.searchsorted(seed_last_ts, side="left"))
                - int(self.backtest_precompute_overlap_bars),
            )
        else:
            build_start_pos = start_pos
        build_source_df = source_df.iloc[build_start_pos:]
        if build_source_df.empty:
            return seed_df

        logging.info(
            "AetherFlowStrategy backtest seed extend: seed_rows=%s seed_end=%s build_rows=%s build_range=%s->%s",
            int(len(seed_df)) if isinstance(seed_df, pd.DataFrame) else 0,
            seed_last_ts,
            int(len(build_source_df)),
            build_source_df.index.min(),
            build_source_df.index.max(),
        )
        computed_df = build_feature_frame(
            build_source_df,
            preferred_setup_families=self.allowed_setup_families,
        )
        if computed_df.empty:
            return seed_df
        if seed_last_ts is not None:
            computed_df = computed_df.loc[computed_df.index > seed_last_ts]
        if computed_df.empty:
            return seed_df

        parts = []
        if isinstance(seed_df, pd.DataFrame) and not seed_df.empty:
            parts.append(seed_df)
        parts.append(computed_df)
        combined = pd.concat(parts, axis=0).sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="last")]
        combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        logging.info(
            "AetherFlowStrategy backtest seed extend complete: combined_rows=%s appended_rows=%s combined_range=%s->%s",
            int(len(combined)),
            int(len(computed_df)),
            combined.index.min(),
            combined.index.max(),
        )
        return combined

    def _discover_live_base_features_path(self) -> Optional[Path]:
        try:
            resolved = resolve_full_manifold_base_features_path(
                self.cfg.get("live_base_features_file", "")
            )
        except Exception:
            resolved = None
        return resolved if resolved is not None and resolved.exists() else None

    def _load_live_base_cache_slice(
        self,
        *,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        path = self._live_base_features_path
        if path is None or not path.exists():
            return pd.DataFrame()
        if not self._live_base_features_validated:
            validate_full_manifold_base_features_path(path, BASE_FEATURE_COLUMNS)
            self._live_base_features_validated = True
        try:
            seed = pd.read_parquet(
                path,
                columns=sorted(set(BASE_FEATURE_COLUMNS)),
                filters=[
                    ("__index_level_0__", ">=", pd.Timestamp(start_time)),
                    ("__index_level_0__", "<=", pd.Timestamp(end_time)),
                ],
            )
        except Exception as exc:
            logging.warning("AetherFlowStrategy live base cache slice load failed: %s", exc)
            return pd.DataFrame()
        if seed.empty:
            return pd.DataFrame()
        seed.index = pd.DatetimeIndex(seed.index)
        seed = seed.sort_index()
        return seed.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _load_live_base_overlay_slice(
        self,
        *,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        path = self._live_base_overlay_path
        if path is None or not path.exists():
            return pd.DataFrame()
        if not self._live_base_overlay_validated:
            validate_full_manifold_base_features_path(path, BASE_FEATURE_COLUMNS)
            self._live_base_overlay_validated = True
        try:
            overlay = pd.read_parquet(
                path,
                columns=sorted(set(BASE_FEATURE_COLUMNS)),
                filters=[
                    ("__index_level_0__", ">=", pd.Timestamp(start_time)),
                    ("__index_level_0__", "<=", pd.Timestamp(end_time)),
                ],
            )
        except Exception as exc:
            logging.warning("AetherFlowStrategy live base overlay slice load failed: %s", exc)
            return pd.DataFrame()
        if overlay.empty:
            return pd.DataFrame()
        overlay.index = pd.DatetimeIndex(overlay.index)
        overlay = overlay.sort_index()
        return overlay.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _load_live_base_seed_slice(
        self,
        *,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> pd.DataFrame:
        parts = []
        seed = self._load_live_base_cache_slice(start_time=start_time, end_time=end_time)
        seed_end = pd.Timestamp(seed.index.max()) if not seed.empty else None
        if not seed.empty:
            parts.append(seed)
        overlay = self._load_live_base_overlay_slice(start_time=start_time, end_time=end_time)
        if seed_end is not None and not overlay.empty:
            overlay = overlay.loc[overlay.index > seed_end]
        if not overlay.empty:
            parts.append(overlay)
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts, axis=0).sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="last")]
        return combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _seeded_live_base_window(
        self,
        df: pd.DataFrame,
        *,
        end_time: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if self._live_base_features_path is None or not self._live_base_features_path.exists():
            return pd.DataFrame()

        source_df = df.sort_index()
        target_end = pd.Timestamp(end_time) if end_time is not None else pd.Timestamp(source_df.index.max())
        source_df = source_df.loc[source_df.index <= target_end]
        if source_df.empty:
            return pd.DataFrame()

        window = self._live_base_window
        if isinstance(window, pd.DataFrame) and not window.empty:
            window = window.sort_index()
            if pd.Timestamp(window.index.max()) >= target_end:
                return window.loc[window.index <= target_end].tail(self.live_base_history_bars)

        start_pos = max(0, len(source_df) - int(self.live_base_history_bars))
        feature_start = pd.Timestamp(source_df.index[start_pos])

        if not isinstance(window, pd.DataFrame) or window.empty:
            seed_df = self._load_live_base_seed_slice(start_time=feature_start, end_time=target_end)
        else:
            seed_df = window.loc[window.index >= feature_start].copy()
            if seed_df.empty:
                seed_df = self._load_live_base_seed_slice(start_time=feature_start, end_time=target_end)

        seed_last_ts = pd.Timestamp(seed_df.index.max()) if isinstance(seed_df, pd.DataFrame) and not seed_df.empty else None
        if seed_last_ts is not None and seed_last_ts >= target_end:
            self._live_base_window = seed_df.tail(self.live_base_history_bars).copy()
            return self._live_base_window

        if seed_last_ts is not None:
            build_start_pos = max(
                0,
                int(source_df.index.searchsorted(seed_last_ts, side="left")) - int(self.live_base_overlap_bars),
            )
        else:
            build_start_pos = max(0, len(source_df) - int(self.live_base_overlap_bars))
        build_source_df = source_df.iloc[build_start_pos:]
        if build_source_df.empty:
            return seed_df.tail(self.live_base_history_bars) if isinstance(seed_df, pd.DataFrame) else pd.DataFrame()

        manifold_cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        manifold_cfg["enabled"] = True
        rebuilt = build_manifold_base_frame(
            build_source_df,
            manifold_cfg=manifold_cfg,
            log_every=0,
        )
        if rebuilt.empty:
            return seed_df.tail(self.live_base_history_bars) if isinstance(seed_df, pd.DataFrame) else pd.DataFrame()
        if seed_last_ts is not None:
            rebuilt = rebuilt.loc[rebuilt.index > seed_last_ts]
        if rebuilt.empty:
            self._live_base_window = seed_df.tail(self.live_base_history_bars).copy()
            return self._live_base_window

        combined_parts = []
        if isinstance(seed_df, pd.DataFrame) and not seed_df.empty:
            combined_parts.append(seed_df)
        combined_parts.append(rebuilt)
        combined = pd.concat(combined_parts, axis=0).sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="last")]
        combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._live_base_window = combined.tail(self.live_base_history_bars).copy()
        return self._live_base_window

    def _queue_runtime_event(
        self,
        *,
        status: str,
        side: Optional[str],
        reason: Optional[str],
        setup_family: Optional[str] = None,
        regime: Optional[str] = None,
        confidence: Optional[float] = None,
        threshold: Optional[float] = None,
        session_id: Optional[int] = None,
    ) -> None:
        signature = (
            str(status or "").upper(),
            str(side or "").upper(),
            str(reason or ""),
            str(setup_family or ""),
            str(regime or ""),
            round(float(confidence or 0.0), 4),
            round(float(threshold or 0.0), 4),
            int(session_id) if session_id is not None else None,
        )
        if signature == self._last_runtime_event_signature:
            return
        self._last_runtime_event_signature = signature
        self._pending_runtime_event = {
            "status": str(status or "BLOCKED").upper(),
            "side": str(side or "NONE").upper(),
            "reason": reason,
            "decision": "blocked",
            "setup_family": setup_family,
            "regime": regime,
            "confidence": confidence,
            "threshold": threshold,
            "session_id": session_id,
        }

    def consume_pending_runtime_event(self) -> Optional[Dict]:
        event = self._pending_runtime_event
        self._pending_runtime_event = None
        return dict(event) if isinstance(event, dict) else None

    def _candidate_family_names(self) -> list[str]:
        if self.family_policies:
            return sorted(str(name) for name in self.family_policies.keys() if str(name).strip())
        if self.allowed_setup_families:
            return sorted(str(name) for name in self.allowed_setup_families if str(name).strip())
        return []

    def _policy_for_family(self, family_name: str, row: Optional[Dict] = None) -> Optional[dict]:
        family_key = str(family_name or "").strip()
        if not family_key:
            return None
        default_policy = {
            "threshold": float(self.threshold),
            "allowed_session_ids": self.allowed_session_ids,
            "allowed_regimes": None,
            "allowed_sides": None,
            "blocked_regimes": self.hazard_block_regimes,
            "max_abs_vwap_dist_atr": None,
            "max_directional_vwap_dist_atr": None,
            "min_d_alignment_3": None,
            "min_d_coherence_3": None,
            "min_pressure_imbalance_30": None,
            "max_pressure_imbalance_30": None,
            "min_setup_strength": None,
            "min_alignment_pct": None,
            "min_smoothness_pct": None,
            "max_stress_pct": None,
            "max_flow_mag_slow": None,
            "min_phase_regime_run_bars": None,
            "max_phase_regime_flip_count_10": None,
            "min_phase_alignment_mean_5": None,
            "max_phase_stress_mean_5": None,
            "min_phase_alignment_trend_5": None,
            "max_phase_stress_trend_5": None,
            "min_phase_d_alignment_mean_5": None,
            "selection_score_bias": 0.0,
            "selection_score_scale": 1.0,
            "size_multiplier": 1.0,
            "entry_mode": "market_next_bar",
            "sl_mult_override": None,
            "tp_mult_override": None,
            "horizon_bars_override": None,
            "use_horizon_time_stop": False,
            "early_exit": {},
        }
        if self.family_policies:
            raw = self.family_policies.get(family_key)
            if not isinstance(raw, dict):
                return None
            base_policy = _merge_policy_layers(default_policy, raw)
            if row is not None:
                for rule in raw.get("policy_rules", []) or []:
                    if _policy_rule_matches_row(rule, row):
                        return _merge_policy_layers(base_policy, rule)
            return base_policy
        if self.allowed_setup_families and family_key not in self.allowed_setup_families:
            return None
        return default_policy

    def _row_block_reason(self, row: Dict) -> str:
        side_num = int(round(float(row.get("candidate_side", 0.0) or 0.0)))
        if side_num == 0:
            return "no_setup"
        side_label = "LONG" if side_num > 0 else "SHORT"
        setup_family = str(row.get("setup_family", "") or "").strip()
        policy = self._policy_for_family(setup_family, row)
        if policy is None:
            return "setup_family_blocked"
        session_id = _row_int(row, "session_id", default=-999)
        allowed_session_ids = policy.get("allowed_session_ids")
        if allowed_session_ids and session_id not in allowed_session_ids:
            return "session_not_allowed"
        regime_name = _regime_name_from_row(row)
        allowed_regimes = policy.get("allowed_regimes")
        if allowed_regimes and regime_name not in allowed_regimes:
            return "regime_not_allowed"
        allowed_sides = policy.get("allowed_sides")
        if allowed_sides and side_label not in allowed_sides:
            return "side_not_allowed"
        blocked_regimes = policy.get("blocked_regimes") or set()
        if regime_name and regime_name in blocked_regimes:
            return "hazard_blocked"
        prob = float(row.get("aetherflow_confidence", 0.0) or 0.0)
        threshold = float(policy.get("threshold", self.threshold) or self.threshold)
        if prob < max(threshold, self.min_confidence):
            return "below_threshold"
        context_reason = _context_block_reason(row, policy)
        if context_reason:
            return context_reason
        params_reason = _params_block_reason(row, policy)
        if params_reason:
            return params_reason
        return ""

    def _compute_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        return predict_bundle_probabilities(self.model_bundle, features)

    def _build_family_candidate_frame(self, source_df: pd.DataFrame, family_name: str) -> pd.DataFrame:
        features = build_feature_frame(
            source_df,
            preferred_setup_families={family_name},
        )
        if features.empty:
            return pd.DataFrame()
        features = augment_aetherflow_phase_features(features)
        features = features.loc[
            (features["setup_family"].astype(str) == str(family_name))
            & (pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0)
        ].copy()
        if features.empty:
            return pd.DataFrame()
        features["aetherflow_confidence"] = self._compute_probabilities(features)
        features["manifold_regime_name"] = features.apply(lambda row: _regime_name_from_row(row.to_dict()), axis=1)
        features["selection_score"] = [
            _selection_score(
                float(confidence),
                self._policy_for_family(family_name, row_dict) or {},
            )
            for row_dict, confidence in zip(features.to_dict("records"), features["aetherflow_confidence"].tolist())
        ]
        return features

    def _select_signal_rows(self, features: pd.DataFrame) -> pd.DataFrame:
        if features is None or features.empty:
            return pd.DataFrame()
        ordered = features.copy()
        if "selection_score" not in ordered.columns:
            ordered["selection_score"] = pd.to_numeric(ordered.get("aetherflow_confidence"), errors="coerce").fillna(0.0)
        ordered = ordered.sort_values(
            by=["selection_score", "aetherflow_confidence", "setup_strength"],
            ascending=[False, False, False],
            kind="mergesort",
        )
        rows = []
        index = []
        for ts, grp in ordered.groupby(level=0, sort=True):
            for _, candidate_row in grp.iterrows():
                signal = self._signal_from_row(candidate_row.to_dict())
                if signal is None:
                    continue
                rows.append(signal)
                index.append(ts)
                break
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows, index=pd.DatetimeIndex(index))

    def _signal_from_row(self, row: Dict) -> Optional[Dict]:
        side_num = int(round(float(row.get("candidate_side", 0.0) or 0.0)))
        if side_num == 0:
            return None
        side_label = "LONG" if side_num > 0 else "SHORT"
        setup_family = str(row.get("setup_family", "") or "").strip()
        policy = self._policy_for_family(setup_family, row)
        if policy is None:
            return None
        prob = float(row.get("aetherflow_confidence", 0.0) or 0.0)
        threshold = float(policy.get("threshold", self.threshold) or self.threshold)
        if prob < max(threshold, self.min_confidence):
            return None
        session_id = _row_int(row, "session_id", default=-999)
        allowed_session_ids = policy.get("allowed_session_ids")
        if allowed_session_ids and session_id not in allowed_session_ids:
            return None
        regime_name = _regime_name_from_row(row)
        allowed_regimes = policy.get("allowed_regimes")
        if allowed_regimes and regime_name not in allowed_regimes:
            return None
        allowed_sides = policy.get("allowed_sides")
        if allowed_sides and side_label not in allowed_sides:
            return None
        blocked_regimes = policy.get("blocked_regimes") or set()
        if regime_name and regime_name in blocked_regimes:
            return None
        if _context_block_reason(row, policy):
            return None
        params_row = pd.Series(dict(row))
        sl_mult_override = policy.get("sl_mult_override")
        tp_mult_override = policy.get("tp_mult_override")
        horizon_bars_override = policy.get("horizon_bars_override")
        if sl_mult_override is not None:
            params_row["setup_sl_mult"] = float(sl_mult_override)
        if tp_mult_override is not None:
            params_row["setup_tp_mult"] = float(tp_mult_override)
        if horizon_bars_override is not None:
            params_row["setup_horizon_bars"] = int(horizon_bars_override)
        params = resolve_setup_params(params_row)
        if _params_block_reason(row, policy, params=params):
            return None
        signal = {
            "strategy": "AetherFlowStrategy",
            "side": side_label,
            "tp_dist": float(params["tp_points"]),
            "sl_dist": float(params["sl_points"]),
            "size": int(self.size),
            "size_multiplier": float(policy.get("size_multiplier", 1.0) or 1.0),
            "entry_mode": str(policy.get("entry_mode", "market_next_bar") or "market_next_bar"),
            "horizon_bars": int(params["horizon_bars"]),
            "use_horizon_time_stop": bool(policy.get("use_horizon_time_stop", False)),
            "confidence": prob,
            "aetherflow_confidence": prob,
            "aetherflow_selection_score": float(row.get("selection_score", _selection_score(prob, policy)) or _selection_score(prob, policy)),
            "aetherflow_threshold": float(threshold),
            "aetherflow_setup_family": setup_family,
            "aetherflow_setup_strength": float(row.get("setup_strength", 0.0) or 0.0),
            "aetherflow_horizon_bars": int(params["horizon_bars"]),
            "aetherflow_use_horizon_time_stop": bool(policy.get("use_horizon_time_stop", False)),
            "aetherflow_regime": regime_name,
            "aetherflow_size_multiplier": float(policy.get("size_multiplier", 1.0) or 1.0),
        }
        early_exit_cfg = dict(policy.get("early_exit", {}) or {})
        if early_exit_cfg:
            signal["early_exit_enabled"] = bool(early_exit_cfg.get("enabled", False))
            signal["early_exit_exit_if_not_green_by"] = int(
                max(0, _coerce_optional_int(early_exit_cfg.get("exit_if_not_green_by")) or 0)
            )
            signal["early_exit_max_profit_crosses"] = int(
                max(0, _coerce_optional_int(early_exit_cfg.get("max_profit_crosses")) or 0)
            )
        return signal

    def _backtest_signals_from_features(
        self,
        features: pd.DataFrame,
        *,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        features = augment_aetherflow_phase_features(features)
        if features.empty:
            return pd.DataFrame()
        if start_time is not None:
            features = features.loc[features.index >= pd.Timestamp(start_time)]
        if end_time is not None:
            features = features.loc[features.index <= pd.Timestamp(end_time)]
        if features.empty:
            return pd.DataFrame()
        if self.family_policies:
            allow_families = set(self._candidate_family_names())
            if allow_families:
                features = features.loc[
                    features["setup_family"].astype(str).isin(sorted(allow_families))
                ]
        elif self.allowed_setup_families:
            features = features.loc[
                features["setup_family"].astype(str).isin(sorted(self.allowed_setup_families))
            ]
        if features.empty:
            return pd.DataFrame()
        features = features.copy()
        if "aetherflow_confidence" not in features.columns:
            features["aetherflow_confidence"] = self._compute_probabilities(features)
        else:
            features["aetherflow_confidence"] = pd.to_numeric(
                features.get("aetherflow_confidence"),
                errors="coerce",
            ).fillna(0.0)
        if self.family_policies:
            row_dicts = features.to_dict("records")
            features["manifold_regime_name"] = [
                _regime_name_from_row(row_dict)
                for row_dict in row_dicts
            ]
            features["selection_score"] = [
                _selection_score(
                    float(row_dict.get("aetherflow_confidence", 0.0) or 0.0),
                    self._policy_for_family(
                        str(row_dict.get("setup_family", "") or ""),
                        row_dict,
                    )
                    or {},
                )
                for row_dict in row_dicts
            ]
        if self.allowed_session_ids and not self.family_policies:
            sess = pd.to_numeric(features.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
            features = features.loc[sess.isin(sorted(self.allowed_session_ids))]
        features = features.loc[
            pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0
        ]
        if features.empty:
            return pd.DataFrame()
        return self._select_signal_rows(features)

    def build_backtest_df_from_base_features(
        self,
        base_features: pd.DataFrame,
        *,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if not self.model_loaded or self.model_bundle is None or base_features is None or base_features.empty:
            return pd.DataFrame()
        features = build_feature_frame(
            base_features=base_features,
            preferred_setup_families=self.allowed_setup_families,
        )
        if features.empty:
            return pd.DataFrame()
        return self._backtest_signals_from_features(
            features,
            start_time=start_time,
            end_time=end_time,
        )

    def build_precomputed_backtest_df(
        self,
        df: pd.DataFrame,
        *,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        if not self.model_loaded or self.model_bundle is None or df is None or df.empty:
            return pd.DataFrame()
        features = self._seeded_feature_window(
            df,
            start_time=start_time,
            end_time=end_time,
        )
        if features.empty:
            features = build_feature_frame(
                df,
                preferred_setup_families=self.allowed_setup_families,
            )
        return self._backtest_signals_from_features(
            features,
            start_time=start_time,
            end_time=end_time,
        )

    def on_bar(self, df, current_time=None) -> Optional[Dict]:
        ts = current_time
        if ts is None and df is not None and not df.empty:
            try:
                ts = pd.Timestamp(df.index[-1])
            except Exception:
                ts = None
        if ts is not None:
            ts_key = int(pd.Timestamp(ts).value)
            cached = self._precomputed_lookup.get(ts_key)
            if cached is not None:
                self.last_eval = dict(cached)
                self._pending_runtime_event = None
                return dict(cached)
            if ts_key in self._precomputed_known_timestamps:
                self.last_eval = None
                self._pending_runtime_event = None
                return None
        if not self.model_loaded or self.model_bundle is None:
            return None
        if df is None or df.empty or len(df) < self.min_bars:
            return None

        history = df.tail(self.max_feature_bars)
        try:
            base_history = self._seeded_live_base_window(
                df,
                end_time=pd.Timestamp(ts) if ts is not None else None,
            )
            if self.family_policies:
                candidate_payloads: list[Dict] = []
                for family_name in self._candidate_family_names():
                    family_features = (
                        build_feature_frame(
                            base_features=base_history,
                            preferred_setup_families={family_name},
                        )
                        if isinstance(base_history, pd.DataFrame) and not base_history.empty
                        else build_feature_frame(
                            history,
                            preferred_setup_families={family_name},
                        )
                    )
                    if family_features.empty:
                        continue
                    family_features = augment_aetherflow_phase_features(family_features)
                    row = family_features.iloc[-1]
                    if str(row.get("setup_family", "") or "") != str(family_name):
                        continue
                    side_num = int(round(float(row.get("candidate_side", 0.0) or 0.0)))
                    if side_num == 0:
                        continue
                    x_row = pd.DataFrame([row], index=[family_features.index[-1]])
                    prob = float(self._compute_probabilities(x_row)[0])
                    eval_payload = row.to_dict()
                    eval_payload["aetherflow_confidence"] = prob
                    policy = self._policy_for_family(family_name, eval_payload) or {}
                    eval_payload["selection_score"] = _selection_score(prob, policy)
                    eval_payload["manifold_regime_name"] = _regime_name_from_row(eval_payload)
                    candidate_payloads.append(eval_payload)

                if not candidate_payloads:
                    self.last_eval = {"decision": "no_signal", "reason": "no_setup"}
                    self._pending_runtime_event = None
                    return None

                candidate_payloads.sort(
                    key=lambda item: (
                        float(item.get("selection_score", 0.0) or 0.0),
                        float(item.get("aetherflow_confidence", 0.0) or 0.0),
                        float(item.get("setup_strength", 0.0) or 0.0),
                    ),
                    reverse=True,
                )
                for eval_payload in candidate_payloads:
                    signal = self._signal_from_row(eval_payload)
                    setup_family = str(eval_payload.get("setup_family", "") or "")
                    regime_name = str(eval_payload.get("manifold_regime_name", "") or "")
                    prob = float(eval_payload.get("aetherflow_confidence", 0.0) or 0.0)
                    policy = self._policy_for_family(setup_family, eval_payload) or {}
                    threshold = float(policy.get("threshold", self.threshold) or self.threshold)
                    session_id = _row_int(eval_payload, "session_id", default=-999)
                    side_num = int(round(float(eval_payload.get("candidate_side", 0.0) or 0.0)))
                    side_label = "LONG" if side_num > 0 else "SHORT" if side_num < 0 else None
                    if signal is not None:
                        self.last_eval = {
                            "decision": "signal",
                            "prob_success": prob,
                            "selection_score": float(eval_payload.get("selection_score", 0.0) or 0.0),
                            "threshold": float(threshold),
                            "setup_family": setup_family,
                            "setup_strength": float(eval_payload.get("setup_strength", 0.0) or 0.0),
                            "regime": regime_name,
                            **signal,
                        }
                        if self.log_evals:
                            logging.info(
                                "AetherFlow signal: %s conf=%.3f thr=%.3f setup=%s strength=%.3f",
                                signal["side"],
                                prob,
                                threshold,
                                signal.get("aetherflow_setup_family"),
                                float(signal.get("aetherflow_setup_strength", 0.0) or 0.0),
                            )
                        self._pending_runtime_event = None
                        self._last_runtime_event_signature = None
                        return signal

                best_eval = candidate_payloads[0]
                reason = self._row_block_reason(best_eval) or "blocked"
                setup_family = str(best_eval.get("setup_family", "") or "")
                regime_name = str(best_eval.get("manifold_regime_name", "") or "")
                prob = float(best_eval.get("aetherflow_confidence", 0.0) or 0.0)
                policy = self._policy_for_family(setup_family, best_eval) or {}
                threshold = float(policy.get("threshold", self.threshold) or self.threshold)
                session_id = _row_int(best_eval, "session_id", default=-999)
                side_num = int(round(float(best_eval.get("candidate_side", 0.0) or 0.0)))
                side_label = "LONG" if side_num > 0 else "SHORT" if side_num < 0 else None
                self.last_eval = {
                    "decision": "no_signal",
                    "prob_success": prob,
                    "selection_score": float(best_eval.get("selection_score", 0.0) or 0.0),
                    "threshold": float(threshold),
                    "setup_family": setup_family,
                    "setup_strength": float(best_eval.get("setup_strength", 0.0) or 0.0),
                    "regime": regime_name,
                    "reason": reason,
                }
                self._queue_runtime_event(
                    status="BLOCKED",
                    side=side_label,
                    reason=reason,
                    setup_family=setup_family,
                    regime=regime_name,
                    confidence=prob,
                    threshold=float(threshold),
                    session_id=session_id,
                )
                return None

            features = (
                build_feature_frame(
                    base_features=base_history,
                    preferred_setup_families=self.allowed_setup_families,
                )
                if isinstance(base_history, pd.DataFrame) and not base_history.empty
                else build_feature_frame(
                    history,
                    preferred_setup_families=self.allowed_setup_families,
                )
            )
            features = augment_aetherflow_phase_features(features)
            if features.empty:
                return None
            row = features.iloc[-1]
            session_id = _row_int(row, "session_id", default=-999)
            side_num = int(round(float(row.get("candidate_side", 0.0) or 0.0)))
            side_label = "LONG" if side_num > 0 else "SHORT" if side_num < 0 else None
            if self.allowed_session_ids and not self.family_policies and session_id not in self.allowed_session_ids:
                self.last_eval = {
                    "decision": "blocked",
                    "reason": "session_not_allowed",
                    "session_id": session_id,
                }
                self._queue_runtime_event(
                    status="BLOCKED",
                    side=side_label,
                    reason="session_not_allowed",
                    session_id=session_id,
                    threshold=float(self.threshold),
                )
                return None
            if side_num == 0:
                self.last_eval = {"decision": "no_signal", "reason": "no_setup"}
                self._pending_runtime_event = None
                return None
            x_row = pd.DataFrame([row], index=[features.index[-1]])
            prob = float(self._compute_probabilities(x_row)[0])
            eval_payload = row.to_dict()
            eval_payload["aetherflow_confidence"] = prob
            eval_payload["manifold_regime_name"] = _regime_name_from_row(eval_payload)
            setup_family = str(row.get("setup_family", "") or "")
            regime_name = str(eval_payload.get("manifold_regime_name", "") or "")
            self.last_eval = {
                "decision": "candidate",
                "prob_success": prob,
                "threshold": float(self.threshold),
                "setup_family": setup_family,
                "setup_strength": float(row.get("setup_strength", 0.0) or 0.0),
                "regime": regime_name,
            }
            signal = self._signal_from_row(eval_payload)
            if signal is None:
                self.last_eval["decision"] = "no_signal"
                reason = "blocked"
                if self.allowed_setup_families and setup_family not in self.allowed_setup_families:
                    reason = "setup_family_blocked"
                elif regime_name and regime_name in self.hazard_block_regimes:
                    reason = "hazard_blocked"
                elif prob < max(self.threshold, self.min_confidence):
                    reason = "below_threshold"
                self.last_eval["reason"] = reason
                self._queue_runtime_event(
                    status="BLOCKED",
                    side=side_label,
                    reason=reason,
                    setup_family=setup_family,
                    regime=regime_name,
                    confidence=prob,
                    threshold=float(self.threshold),
                    session_id=session_id,
                )
                return None
            if self.log_evals:
                logging.info(
                    "AetherFlow signal: %s conf=%.3f thr=%.3f setup=%s strength=%.3f",
                    signal["side"],
                    prob,
                    self.threshold,
                    signal.get("aetherflow_setup_family"),
                    float(signal.get("aetherflow_setup_strength", 0.0) or 0.0),
                )
            self._pending_runtime_event = None
            self._last_runtime_event_signature = None
            self.last_eval.update({"decision": "signal", **signal})
            return signal
        except Exception as exc:
            logging.error("AetherFlowStrategy prediction error: %s", exc)
            self.last_eval = {"decision": "error", "error": str(exc)}
            self._queue_runtime_event(status="BLOCKED", side=None, reason="error", threshold=float(self.threshold))
            return None
