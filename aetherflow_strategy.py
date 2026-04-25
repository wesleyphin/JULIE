import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from aetherflow_base_cache import resolve_full_manifold_base_features_path
from aetherflow_features import (
    BASE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_feature_frame,
    build_feature_frames_by_family,
    resolve_setup_params,
)
from aetherflow_model_bundle import (
    bundle_feature_columns,
    bundle_has_predictor,
    normalize_model_bundle,
    predict_bundle_probabilities,
)
from config import CONFIG
from manifold_strategy_features import (
    build_training_feature_frame as build_manifold_feature_frame,
    build_training_feature_frame_with_state,
)
from strategy_base import Strategy


ROOT = Path(__file__).resolve().parent

REGIME_ID_TO_NAME = {
    0: "TREND_GEODESIC",
    1: "CHOP_SPIRAL",
    2: "DISPERSED",
    3: "ROTATIONAL_TURBULENCE",
}


def _resolve_repo_path(value) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


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


def _coerce_optional_int(value) -> Optional[int]:
    coerced = _coerce_optional_float(value)
    if coerced is None:
        return None
    return int(round(float(coerced)))


def _row_int(row: Dict, key: str, default: int = 0) -> int:
    coerced = _coerce_optional_int(row.get(key))
    return int(coerced) if coerced is not None else int(default)


def _side_label_from_row(row: Dict) -> str:
    try:
        side = float(row.get("candidate_side", 0.0) or 0.0)
    except Exception:
        return ""
    if side > 0.0:
        return "LONG"
    if side < 0.0:
        return "SHORT"
    return ""


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
    if "blocked_regimes" in policy:
        out["blocked_regimes"] = _coerce_upper_string_allowlist(policy.get("blocked_regimes"))
    if "allowed_sides" in policy:
        out["allowed_sides"] = _coerce_side_allowlist(policy.get("allowed_sides"))
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
    if "min_setup_strength" in policy:
        out["min_setup_strength"] = _coerce_optional_float(policy.get("min_setup_strength"))
    if "min_alignment_pct" in policy:
        out["min_alignment_pct"] = _coerce_optional_float(policy.get("min_alignment_pct"))
    if "min_smoothness_pct" in policy:
        out["min_smoothness_pct"] = _coerce_optional_float(policy.get("min_smoothness_pct"))
    if "max_stress_pct" in policy:
        out["max_stress_pct"] = _coerce_optional_float(policy.get("max_stress_pct"))
    if "min_flow_agreement" in policy:
        out["min_flow_agreement"] = _coerce_optional_float(policy.get("min_flow_agreement"))
    if "min_flow_mag_slow" in policy:
        out["min_flow_mag_slow"] = _coerce_optional_float(policy.get("min_flow_mag_slow"))
    if "max_flow_mag_slow" in policy:
        out["max_flow_mag_slow"] = _coerce_optional_float(policy.get("max_flow_mag_slow"))
    if "min_pressure_imbalance_30" in policy:
        out["min_pressure_imbalance_30"] = _coerce_optional_float(policy.get("min_pressure_imbalance_30"))
    if "min_signed_pressure_30" in policy:
        out["min_signed_pressure_30"] = _coerce_optional_float(policy.get("min_signed_pressure_30"))
    if "min_coherence_pct" in policy:
        out["min_coherence_pct"] = _coerce_optional_float(policy.get("min_coherence_pct"))
    if "min_phase_regime_run_bars" in policy:
        out["min_phase_regime_run_bars"] = _coerce_optional_float(policy.get("min_phase_regime_run_bars"))
    if "min_phase_d_alignment_mean_5" in policy:
        out["min_phase_d_alignment_mean_5"] = _coerce_optional_float(policy.get("min_phase_d_alignment_mean_5"))
    if "size_multiplier" in policy:
        out["size_multiplier"] = _coerce_optional_float(policy.get("size_multiplier"))
    if "selection_score_bias" in policy:
        out["selection_score_bias"] = _coerce_optional_float(policy.get("selection_score_bias"))
    if "score_bias" in policy and "selection_score_bias" not in out:
        out["selection_score_bias"] = _coerce_optional_float(policy.get("score_bias"))
    if "selection_score_scale" in policy:
        out["selection_score_scale"] = _coerce_optional_float(policy.get("selection_score_scale"))
    if "score_scale" in policy and "selection_score_scale" not in out:
        out["selection_score_scale"] = _coerce_optional_float(policy.get("score_scale"))
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
        if "match_setup_families" in policy:
            out["match_setup_families"] = _coerce_string_allowlist(policy.get("match_setup_families"))
        if "match_session_ids" in policy:
            out["match_session_ids"] = _coerce_session_allowlist(policy.get("match_session_ids"))
        if "match_regimes" in policy:
            out["match_regimes"] = _coerce_upper_string_allowlist(policy.get("match_regimes"))
        if "match_sides" in policy:
            out["match_sides"] = _coerce_side_allowlist(policy.get("match_sides"))
        for key, value in policy.items():
            key_text = str(key)
            if key_text.startswith("match_min_") or key_text.startswith("match_max_"):
                out[key_text] = _coerce_optional_float(value)

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
            if key in {
                "policy_rules",
                "rules",
                "match_setup_families",
                "match_session_ids",
                "match_regimes",
                "match_sides",
                "name",
            }:
                continue
            if str(key).startswith("match_min_") or str(key).startswith("match_max_"):
                continue
            if key == "early_exit":
                merged[key] = dict(value or {})
            elif isinstance(value, set):
                merged[key] = set(value)
            else:
                merged[key] = value
    return merged


def _policy_feature_column_from_suffix(suffix: str) -> str:
    aliases = {
        "phase_d_alignment_mean_5": "phase_d_alignment_3_mean_5",
        "phase_alignment_mean_5": "phase_manifold_alignment_pct_mean_5",
        "phase_stress_mean_5": "phase_manifold_stress_pct_mean_5",
        "phase_regime_run_bars": "phase_regime_run_bars",
        "phase_regime_flip_count_10": "phase_regime_flip_count_10",
        "flow_agreement": "flow_agreement",
    }
    return aliases.get(str(suffix), str(suffix))


def _policy_rule_matches_row(rule: Dict, row: Dict) -> bool:
    if not isinstance(rule, dict):
        return False
    setup_families = rule.get("match_setup_families")
    if setup_families:
        setup_family = str(row.get("setup_family", "") or "").strip()
        if setup_family not in setup_families:
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
    if "match_sides" in rule:
        match_sides = rule.get("match_sides")
        if match_sides and _side_label_from_row(row) not in match_sides:
            return False
    for key, raw_value in rule.items():
        key_text = str(key)
        if key_text.startswith("match_min_"):
            min_value = _coerce_optional_float(raw_value)
            if min_value is None:
                continue
            column = _policy_feature_column_from_suffix(key_text[len("match_min_") :])
            if _row_float(row, column, 0.0) < float(min_value):
                return False
        elif key_text.startswith("match_max_"):
            max_value = _coerce_optional_float(raw_value)
            if max_value is None:
                continue
            column = _policy_feature_column_from_suffix(key_text[len("match_max_") :])
            if _row_float(row, column, 0.0) > float(max_value):
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


def _normalize_post_policy_size_rules(value) -> list[dict]:
    if not isinstance(value, dict) or not bool(value.get("enabled", False)):
        return []
    raw_rules = value.get("rules", [])
    if not isinstance(raw_rules, list):
        return []
    rules: list[dict] = []
    for item in raw_rules:
        rule = _normalize_policy_mapping(item, allow_match_fields=True, allow_rules=False)
        multiplier = _coerce_optional_float((item or {}).get("size_multiplier")) if isinstance(item, dict) else None
        if multiplier is None or multiplier <= 0.0:
            continue
        rule["size_multiplier"] = float(multiplier)
        if rule:
            rules.append(rule)
    return rules


def _regime_name_from_row(row: Dict) -> str:
    raw_name = str(row.get("manifold_regime_name", "") or row.get("manifold_regime", "") or "").strip().upper()
    if raw_name:
        return raw_name
    regime_id = _row_int(row, "manifold_regime_id", default=-1)
    return str(REGIME_ID_TO_NAME.get(regime_id, "") or "")


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
    allowed_sides = policy.get("allowed_sides")
    if allowed_sides and _side_label_from_row(row) not in allowed_sides:
        return "side_not_allowed"

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
        signed_alignment = _row_float(row, "candidate_side", 0.0) * _row_float(row, "d_alignment_3", 0.0)
        if signed_alignment < float(min_signed_d_alignment_3):
            return "signed_d_alignment_too_low"

    min_d_coherence_3 = policy.get("min_d_coherence_3")
    if min_d_coherence_3 is not None and _row_float(row, "d_coherence_3", 0.0) < float(min_d_coherence_3):
        return "d_coherence_too_low"

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

    min_flow_agreement = policy.get("min_flow_agreement")
    if min_flow_agreement is not None and _row_float(row, "flow_agreement", 0.0) < float(min_flow_agreement):
        return "flow_agreement_too_low"

    min_flow_mag_slow = policy.get("min_flow_mag_slow")
    if min_flow_mag_slow is not None and _row_float(row, "flow_mag_slow", 0.0) < float(min_flow_mag_slow):
        return "flow_mag_slow_too_low"

    max_flow_mag_slow = policy.get("max_flow_mag_slow")
    if max_flow_mag_slow is not None and _row_float(row, "flow_mag_slow", 0.0) > float(max_flow_mag_slow):
        return "flow_mag_slow_too_high"

    min_pressure_imbalance_30 = policy.get("min_pressure_imbalance_30")
    if min_pressure_imbalance_30 is not None and _row_float(row, "pressure_imbalance_30", 0.0) < float(min_pressure_imbalance_30):
        return "pressure_imbalance_30_too_low"

    min_signed_pressure_30 = policy.get("min_signed_pressure_30")
    if min_signed_pressure_30 is not None:
        signed_pressure = _row_float(row, "candidate_side", 0.0) * _row_float(row, "pressure_imbalance_30", 0.0)
        if signed_pressure < float(min_signed_pressure_30):
            return "signed_pressure_30_too_low"

    min_coherence_pct = policy.get("min_coherence_pct")
    if min_coherence_pct is not None and _row_float(row, "coherence", 0.0) < float(min_coherence_pct):
        return "coherence_too_low"

    min_phase_regime_run_bars = policy.get("min_phase_regime_run_bars")
    if min_phase_regime_run_bars is not None and _row_float(row, "phase_regime_run_bars", 0.0) < float(min_phase_regime_run_bars):
        return "phase_regime_run_too_short"

    min_phase_d_alignment_mean_5 = policy.get("min_phase_d_alignment_mean_5")
    if min_phase_d_alignment_mean_5 is not None:
        value = _row_float(row, "phase_d_alignment_3_mean_5", _row_float(row, "phase_d_alignment_mean_5", 0.0))
        if value < float(min_phase_d_alignment_mean_5):
            return "phase_d_alignment_too_low"

    return ""


def augment_aetherflow_phase_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame
    work = frame.sort_index(kind="mergesort").copy()
    regime_id = pd.to_numeric(work.get("manifold_regime_id"), errors="coerce").fillna(-1).round().astype(int)
    regime_change = regime_id.ne(regime_id.shift(1)).fillna(True)
    group_id = regime_change.cumsum()
    work["phase_regime_run_bars"] = group_id.groupby(group_id).cumcount() + 1
    work["phase_regime_flip_count_10"] = regime_change.astype(int).rolling(10, min_periods=1).sum()

    def _numeric_series(name: str) -> pd.Series:
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
        series = _numeric_series(col)
        work[f"phase_{col}_mean_5"] = series.rolling(5, min_periods=1).mean()
        first = series.iloc[0] if len(series) else 0.0
        work[f"phase_{col}_trend_5"] = series - series.shift(5).fillna(first)
    return work.replace([np.inf, -np.inf], np.nan).fillna(0.0)


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
        self.live_incremental_manifold = bool(self.cfg.get("live_incremental_manifold", True))
        self.live_base_history_bars = max(
            self.max_feature_bars,
            int(self.cfg.get("live_base_history_bars", self.max_feature_bars) or self.max_feature_bars),
        )
        self.live_base_overlap_bars = max(
            self.min_bars,
            int(self.cfg.get("live_base_overlap_bars", min(self.live_base_history_bars, 720)) or self.min_bars),
        )
        self.backtest_base_features_path: Optional[Path] = None
        self.live_base_features_path: Optional[Path] = None
        try:
            self.backtest_base_features_path = resolve_full_manifold_base_features_path(
                self.cfg.get("backtest_base_features_file")
            )
        except Exception:
            self.backtest_base_features_path = None
        try:
            self.live_base_features_path = resolve_full_manifold_base_features_path(
                self.cfg.get("live_base_features_file") or self.cfg.get("backtest_base_features_file")
            )
            self._live_base_features_path: Optional[Path] = self.live_base_features_path
        except Exception:
            self.live_base_features_path = None
            self._live_base_features_path = None
        self._live_base_overlay_path: Optional[Path] = _resolve_repo_path(self.cfg.get("live_base_overlay_file"))
        self._live_base_features_validated = False
        self._live_base_overlay_validated = False
        self._live_base_window: Optional[pd.DataFrame] = None
        self._live_manifold_state: Optional[Dict] = None
        self._live_manifold_lookback_bars = 0
        self._live_last_base_ts: Optional[pd.Timestamp] = None
        self.allowed_session_ids: Optional[set[int]] = _coerce_session_allowlist(self.cfg.get("allowed_session_ids"))
        self.allowed_setup_families: Optional[set[str]] = _coerce_string_allowlist(self.cfg.get("allowed_setup_families"))
        self.hazard_block_regimes = {
            str(item).strip().upper() for item in (self.cfg.get("hazard_block_regimes", []) or []) if str(item).strip()
        }
        self.family_policies = _normalize_family_policies(self.cfg.get("family_policies"))
        self.post_policy_size_rules = _normalize_post_policy_size_rules(self.cfg.get("post_policy_size_rules"))
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
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        threshold_loaded = False
        if self.thresholds_path.exists():
            try:
                payload = json.loads(self.thresholds_path.read_text())
                self.threshold = float(payload.get("threshold", self.threshold) or self.threshold)
                threshold_loaded = True
                feat_cols = payload.get("feature_columns")
                if isinstance(feat_cols, list) and feat_cols:
                    self.feature_columns = [str(col) for col in feat_cols]
                if not self.family_policies:
                    payload_family_policies = _normalize_family_policies(payload.get("family_policies"))
                    if payload_family_policies:
                        self.family_policies = payload_family_policies
                if self.allowed_setup_families is None:
                    payload_allow = _coerce_string_allowlist(payload.get("allowed_setup_families"))
                    if payload_allow:
                        self.allowed_setup_families = payload_allow
                if not self.hazard_block_regimes:
                    payload_block = payload.get("hazard_block_regimes", []) or []
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
            if self.model_loaded:
                logging.info("AetherFlowStrategy model loaded: %s", self.model_path)
                self._prewarm_model_predictions()
        except Exception as exc:
            logging.error("AetherFlowStrategy artifact load failed: %s", exc)
            self.model = None
            self.model_bundle = None
            self.model_loaded = False

    def _prewarm_model_predictions(self) -> None:
        if not self.model_loaded or self.model_bundle is None:
            return
        try:
            row = {str(col): 0.0 for col in self.feature_columns}
            row.update(
                {
                    "setup_family": "transition_burst",
                    "candidate_side": 1.0,
                    "session_id": 2.0,
                    "manifold_regime_id": 1.0,
                    "manifold_regime_name": "CHOP_SPIRAL",
                    "setup_strength": 1.0,
                    "setup_transition_burst": 1.0,
                }
            )
            _ = predict_bundle_probabilities(self.model_bundle, pd.DataFrame([row]))
        except Exception as exc:
            logging.debug("AetherFlowStrategy prediction prewarm skipped: %s", exc)

    def set_precomputed_backtest_df(self, df: Optional[pd.DataFrame]) -> None:
        self._precomputed_backtest_df = None if df is None else df.copy()
        self._precomputed_lookup = {}
        if not isinstance(self._precomputed_backtest_df, pd.DataFrame) or self._precomputed_backtest_df.empty:
            return
        rows = self._precomputed_backtest_df.to_dict("records")
        for ts, row in zip(pd.DatetimeIndex(self._precomputed_backtest_df.index), rows):
            self._precomputed_lookup[int(ts.value)] = row

    def _manifold_cfg(self) -> dict:
        cfg = dict(CONFIG.get("REGIME_MANIFOLD", {}) or {})
        cfg["enabled"] = True
        params = self.cfg.get("manifold_params")
        if isinstance(params, dict):
            cfg.update(params)
        return cfg

    @staticmethod
    def _normalize_base_frame(frame: Optional[pd.DataFrame]) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=BASE_FEATURE_COLUMNS)
        out = frame.copy()
        out.index = pd.DatetimeIndex(out.index)
        out = out.sort_index()
        out = out.loc[~out.index.duplicated(keep="last")]
        out = out.reindex(columns=BASE_FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    def _read_live_base_parquet(self, path: Optional[Path]) -> pd.DataFrame:
        if path is None or not Path(path).exists():
            return pd.DataFrame(columns=BASE_FEATURE_COLUMNS)
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            logging.warning("AetherFlow live base parquet load failed: %s (%s)", path, exc)
            return pd.DataFrame(columns=BASE_FEATURE_COLUMNS)
        return self._normalize_base_frame(frame)

    def _load_live_base_seed_slice(
        self,
        *,
        start_time: Optional[pd.Timestamp] = None,
        end_time: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        base = self._read_live_base_parquet(self._live_base_features_path)
        overlay = self._read_live_base_parquet(self._live_base_overlay_path)
        if not overlay.empty:
            if not base.empty:
                overlay = overlay.loc[~overlay.index.isin(base.index)]
            combined = pd.concat([base, overlay], axis=0)
        else:
            combined = base
        combined = self._normalize_base_frame(combined)
        if combined.empty:
            return combined
        if start_time is not None:
            combined = combined.loc[pd.DatetimeIndex(combined.index) >= pd.Timestamp(start_time)]
        if end_time is not None:
            combined = combined.loc[pd.DatetimeIndex(combined.index) <= pd.Timestamp(end_time)]
        combined = self._normalize_base_frame(combined)
        return combined.reindex(columns=sorted(set(BASE_FEATURE_COLUMNS)))

    def _seeded_live_base_window(self, raw: pd.DataFrame, *, end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        if raw is None or raw.empty:
            return pd.DataFrame(columns=BASE_FEATURE_COLUMNS)
        work = raw.copy()
        work.index = pd.DatetimeIndex(work.index)
        work = work.sort_index()
        end_ts = pd.Timestamp(end_time) if end_time is not None else pd.Timestamp(work.index[-1])
        work = work.loc[pd.DatetimeIndex(work.index) <= end_ts]
        if work.empty:
            return pd.DataFrame(columns=BASE_FEATURE_COLUMNS)

        seed = self._load_live_base_seed_slice(end_time=end_ts)
        if not seed.empty:
            seed = seed.tail(max(1, int(self.live_base_history_bars)))
            seed_end = pd.Timestamp(seed.index.max())
            rebuilt = build_manifold_feature_frame(work, manifold_cfg=self._manifold_cfg(), log_every=0)
            rebuilt = self._normalize_base_frame(rebuilt)
            rebuilt = rebuilt.loc[pd.DatetimeIndex(rebuilt.index) > seed_end]
            if not rebuilt.empty:
                rebuilt = rebuilt.loc[~rebuilt.index.isin(seed.index)]
                combined = pd.concat([seed, rebuilt], axis=0)
            else:
                combined = seed
        else:
            combined = build_manifold_feature_frame(work, manifold_cfg=self._manifold_cfg(), log_every=0)
        combined = self._normalize_base_frame(combined)
        return combined.tail(max(1, int(self.live_base_history_bars)))

    def _reset_live_base_window(self, history: pd.DataFrame) -> pd.DataFrame:
        base, state, lookback_bars = build_training_feature_frame_with_state(
            history,
            manifold_cfg=self._manifold_cfg(),
            log_every=0,
        )
        base = self._normalize_base_frame(base)
        self._live_manifold_state = state
        self._live_manifold_lookback_bars = int(lookback_bars)
        self._live_base_window = base.tail(max(1, int(self.live_base_history_bars)))
        self._live_last_base_ts = (
            pd.Timestamp(self._live_base_window.index.max()) if not self._live_base_window.empty else None
        )
        return self._live_base_window

    def _live_base_frame(self, history: pd.DataFrame) -> pd.DataFrame:
        if history is None or history.empty:
            return pd.DataFrame(columns=BASE_FEATURE_COLUMNS)
        work = history.copy()
        work.index = pd.DatetimeIndex(work.index)
        work = work.sort_index()
        if not self.live_incremental_manifold:
            return self._normalize_base_frame(
                build_manifold_feature_frame(work, manifold_cfg=self._manifold_cfg(), log_every=0)
            )

        end_ts = pd.Timestamp(work.index[-1])
        last_ts = self._live_last_base_ts
        needs_reset = (
            self._live_base_window is None
            or self._live_manifold_state is None
            or last_ts is None
            or end_ts <= last_ts
            or last_ts not in set(pd.DatetimeIndex(work.index))
        )
        if needs_reset:
            return self._reset_live_base_window(work)

        new_base, state, lookback_bars = build_training_feature_frame_with_state(
            work,
            manifold_cfg=self._manifold_cfg(),
            log_every=0,
            initial_state=self._live_manifold_state,
            start_after=last_ts,
        )
        self._live_manifold_state = state
        self._live_manifold_lookback_bars = int(lookback_bars)
        new_base = self._normalize_base_frame(new_base)
        if not new_base.empty:
            self._live_base_window = self._normalize_base_frame(
                pd.concat([self._live_base_window, new_base], axis=0)
            ).tail(max(1, int(self.live_base_history_bars)))
            self._live_last_base_ts = pd.Timestamp(self._live_base_window.index.max())
        return self._live_base_window if self._live_base_window is not None else pd.DataFrame(columns=BASE_FEATURE_COLUMNS)

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
            "min_signed_d_alignment_3": None,
            "min_d_coherence_3": None,
            "min_setup_strength": None,
            "min_alignment_pct": None,
            "min_smoothness_pct": None,
            "max_stress_pct": None,
            "min_flow_agreement": None,
            "min_flow_mag_slow": None,
            "max_flow_mag_slow": None,
            "min_pressure_imbalance_30": None,
            "min_signed_pressure_30": None,
            "min_coherence_pct": None,
            "min_phase_regime_run_bars": None,
            "min_phase_d_alignment_mean_5": None,
            "selection_score_bias": 0.0,
            "selection_score_scale": 1.0,
            "size_multiplier": None,
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

    def _post_policy_size_multiplier(self, setup_family: str, row: Dict) -> float:
        multiplier = 1.0
        if not self.post_policy_size_rules:
            return multiplier
        rule_row = dict(row or {})
        rule_row["setup_family"] = str(setup_family or rule_row.get("setup_family", "") or "").strip()
        for rule in self.post_policy_size_rules:
            if not _policy_rule_matches_row(rule, rule_row):
                continue
            rule_multiplier = _coerce_optional_float(rule.get("size_multiplier"))
            if rule_multiplier is None or rule_multiplier <= 0.0:
                continue
            multiplier *= float(rule_multiplier)
        return float(multiplier)

    def _row_block_reason(self, row: Dict) -> str:
        side_num = int(round(float(row.get("candidate_side", 0.0) or 0.0)))
        if side_num == 0:
            return "no_setup"
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
        return ""

    def _compute_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        return predict_bundle_probabilities(self.model_bundle, features)

    def _score_family_candidate_frame(self, features: pd.DataFrame, family_name: str) -> pd.DataFrame:
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

    def _build_family_candidate_frame(
        self,
        source_df: pd.DataFrame,
        family_name: str,
        *,
        base_features: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        build_kwargs = {"preferred_setup_families": {family_name}}
        if base_features is None:
            features = build_feature_frame(source_df, **build_kwargs)
        else:
            features = build_feature_frame(base_features=base_features, **build_kwargs)
        return self._score_family_candidate_frame(features, family_name)

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
        size_multiplier = policy.get("size_multiplier")
        policy_size_multiplier = float(size_multiplier) if size_multiplier is not None else 1.0
        post_policy_size_multiplier = self._post_policy_size_multiplier(setup_family, row)
        combined_size_multiplier = float(policy_size_multiplier) * float(post_policy_size_multiplier)
        effective_size = int(self.size)
        if abs(combined_size_multiplier - 1.0) > 1e-12:
            effective_size = max(1, int(round(float(self.size) * float(combined_size_multiplier))))
        signal = {
            "strategy": "AetherFlowStrategy",
            "side": "LONG" if side_num > 0 else "SHORT",
            "tp_dist": float(params["tp_points"]),
            "sl_dist": float(params["sl_points"]),
            "size": int(effective_size),
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
        }
        if abs(combined_size_multiplier - 1.0) > 1e-12:
            signal["aetherflow_size_multiplier"] = float(combined_size_multiplier)
            signal["aetherflow_policy_size_multiplier"] = float(policy_size_multiplier)
            signal["aetherflow_post_policy_size_multiplier"] = float(post_policy_size_multiplier)
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

    def build_precomputed_backtest_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.model_loaded or self.model_bundle is None or df is None or df.empty:
            return pd.DataFrame()
        if self.family_policies:
            base_features = build_manifold_feature_frame(df)
            if base_features.empty:
                return pd.DataFrame()
            family_names = self._candidate_family_names()
            family_frames = build_feature_frames_by_family(
                base_features=base_features,
                preferred_setup_families=set(family_names),
            )
            candidate_frames = []
            for family_name in family_names:
                family_frame = self._score_family_candidate_frame(
                    family_frames.get(family_name, pd.DataFrame()),
                    family_name,
                )
                if not family_frame.empty:
                    candidate_frames.append(family_frame)
            if not candidate_frames:
                return pd.DataFrame()
            merged = pd.concat(candidate_frames, axis=0).sort_index()
            return self._select_signal_rows(merged)
        features = build_feature_frame(
            df,
            preferred_setup_families=self.allowed_setup_families,
        )
        if features.empty:
            return pd.DataFrame()
        features = augment_aetherflow_phase_features(features)
        if self.allowed_session_ids:
            sess = pd.to_numeric(features.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
            features = features.loc[sess.isin(sorted(self.allowed_session_ids))]
        features = features.loc[pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0]
        if self.allowed_setup_families:
            features = features.loc[features["setup_family"].astype(str).isin(sorted(self.allowed_setup_families))]
        if features.empty:
            return pd.DataFrame()
        features = features.copy()
        features["aetherflow_confidence"] = self._compute_probabilities(features)
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
        base = self._normalize_base_frame(base_features)
        if base.empty:
            return pd.DataFrame()
        if start_time is not None:
            base = base.loc[pd.DatetimeIndex(base.index) >= pd.Timestamp(start_time)]
        if end_time is not None:
            base = base.loc[pd.DatetimeIndex(base.index) <= pd.Timestamp(end_time)]
        base = self._normalize_base_frame(base)
        if base.empty:
            return pd.DataFrame()
        if self.family_policies:
            family_names = self._candidate_family_names()
            family_frames = build_feature_frames_by_family(
                base_features=base,
                preferred_setup_families=set(family_names),
            )
            candidate_frames = []
            for family_name in family_names:
                family_frame = self._score_family_candidate_frame(
                    family_frames.get(family_name, pd.DataFrame()),
                    family_name,
                )
                if not family_frame.empty:
                    candidate_frames.append(family_frame)
            if not candidate_frames:
                return pd.DataFrame()
            merged = pd.concat(candidate_frames, axis=0).sort_index()
            return self._select_signal_rows(merged)

        features = build_feature_frame(
            base_features=base,
            preferred_setup_families=self.allowed_setup_families,
        )
        if features.empty:
            return pd.DataFrame()
        features = augment_aetherflow_phase_features(features)
        if self.allowed_session_ids:
            sess = pd.to_numeric(features.get("session_id"), errors="coerce").fillna(-999).round().astype(int)
            features = features.loc[sess.isin(sorted(self.allowed_session_ids))]
        features = features.loc[pd.to_numeric(features.get("candidate_side", 0.0), errors="coerce").fillna(0.0) != 0.0]
        if self.allowed_setup_families:
            features = features.loc[features["setup_family"].astype(str).isin(sorted(self.allowed_setup_families))]
        if features.empty:
            return pd.DataFrame()
        features = features.copy()
        features["aetherflow_confidence"] = self._compute_probabilities(features)
        features["manifold_regime_name"] = features.apply(lambda row: _regime_name_from_row(row.to_dict()), axis=1)
        return self._select_signal_rows(features)

    def on_bar(self, df, current_time=None) -> Optional[Dict]:
        ts = current_time
        if ts is None and df is not None and not df.empty:
            try:
                ts = pd.Timestamp(df.index[-1])
            except Exception:
                ts = None
        if ts is not None:
            cached = self._precomputed_lookup.get(int(pd.Timestamp(ts).value))
            if cached is not None:
                self.last_eval = dict(cached)
                self._pending_runtime_event = None
                return dict(cached)
        if not self.model_loaded or self.model_bundle is None:
            return None
        if df is None or df.empty or len(df) < self.min_bars:
            return None

        history = df.tail(self.max_feature_bars)
        try:
            if self.family_policies:
                base_features = self._live_base_frame(history)
                if base_features.empty:
                    self.last_eval = {"decision": "no_signal", "reason": "no_setup"}
                    self._pending_runtime_event = None
                    return None
                candidate_payloads: list[Dict] = []
                family_names = self._candidate_family_names()
                family_frames = build_feature_frames_by_family(
                    base_features=base_features,
                    preferred_setup_families=set(family_names),
                )
                for family_name in family_names:
                    family_features = family_frames.get(family_name, pd.DataFrame())
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

            base_features = self._live_base_frame(history)
            features = build_feature_frame(
                base_features=base_features,
                preferred_setup_families=self.allowed_setup_families,
            )
            if features.empty:
                return None
            features = augment_aetherflow_phase_features(features)
            row = features.iloc[-1]
            session_id = _row_int(row, "session_id", default=-999)
            side_num = int(round(float(row.get("candidate_side", 0.0) or 0.0)))
            side_label = "LONG" if side_num > 0 else "SHORT" if side_num < 0 else None
            if self.allowed_session_ids and session_id not in self.allowed_session_ids:
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
                else:
                    policy = self._policy_for_family(setup_family, eval_payload) or {}
                    reason = _context_block_reason(eval_payload, policy) or reason
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
