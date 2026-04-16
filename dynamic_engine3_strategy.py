import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dynamic_signal_engine3 import get_signal_engine
from de3_v4_runtime import DE3V4Runtime
from fixed_sltp_framework import apply_fixed_sltp
from volatility_filter import volatility_filter
from strategy_base import Strategy
from config import CONFIG
from incremental_ohlcv_resampler import IncrementalOHLCVResampler


_DE3_V3_FAMILY_RUNTIME_CLS = None
_DE3_CONTEXT_VETO_CLS = None
_DE3_V3_CANONICAL_CONTEXT_SNAPSHOT_FN = None


def _get_de3_v3_family_runtime_cls():
    global _DE3_V3_FAMILY_RUNTIME_CLS
    if _DE3_V3_FAMILY_RUNTIME_CLS is None:
        from de3_v3_family_runtime import DE3V3FamilyRuntime

        _DE3_V3_FAMILY_RUNTIME_CLS = DE3V3FamilyRuntime
    return _DE3_V3_FAMILY_RUNTIME_CLS


def _get_de3_context_veto_cls():
    global _DE3_CONTEXT_VETO_CLS
    if _DE3_CONTEXT_VETO_CLS is None:
        from de3_context_veto import DE3ContextVeto

        _DE3_CONTEXT_VETO_CLS = DE3ContextVeto
    return _DE3_CONTEXT_VETO_CLS


def _canonical_context_usage_snapshot(out_row: Dict[str, Any]) -> Dict[str, Any]:
    global _DE3_V3_CANONICAL_CONTEXT_SNAPSHOT_FN
    if _DE3_V3_CANONICAL_CONTEXT_SNAPSHOT_FN is None:
        from de3_v3_family_schema import canonical_context_usage_snapshot

        _DE3_V3_CANONICAL_CONTEXT_SNAPSHOT_FN = canonical_context_usage_snapshot
    return _DE3_V3_CANONICAL_CONTEXT_SNAPSHOT_FN(out_row)


DE3_V4_EXPORT_FIELDS = (
    "de3_v4_route_decision",
    "de3_v4_route_confidence",
    "de3_v4_selected_lane",
    "de3_v4_selected_variant_id",
    "de3_v4_lane_candidate_count",
    "de3_v4_lane_selection_reason",
    "de3_v4_bracket_mode",
    "de3_v4_selected_sl",
    "de3_v4_selected_tp",
    "de3_v4_canonical_default_used",
    "de3_v4_runtime_mode",
    "de3_v4_route_scores",
    "de3_v4_execution_policy_tier",
    "de3_v4_execution_quality_score",
    "de3_v4_execution_policy_reason",
    "de3_v4_execution_policy_source",
    "de3_v4_execution_policy_enforce_veto",
    "de3_v4_execution_policy_soft_pass",
    "de3_v4_execution_policy_hard_limit_triggered",
    "de3_v4_execution_policy_hard_limit_reason",
    "de3_v4_execution_policy_components",
    "de3_v4_entry_model_enabled",
    "de3_v4_entry_model_allow",
    "de3_v4_entry_model_tier",
    "de3_v4_entry_model_reason",
    "de3_v4_entry_model_score",
    "de3_v4_entry_model_threshold",
    "de3_v4_entry_model_threshold_base",
    "de3_v4_entry_model_threshold_scope_offset",
    "de3_v4_entry_model_scope",
    "de3_v4_entry_model_stats",
    "de3_v4_entry_model_components",
    "de3_v4_profit_gate_lane",
    "de3_v4_profit_gate_session",
    "de3_v4_profit_gate_min_samples_eff",
    "de3_v4_profit_gate_max_p_loss_std_eff",
    "de3_v4_profit_gate_min_ev_lcb_points_eff",
    "de3_v4_profit_gate_min_ev_mean_points_eff",
    "de3_v4_profit_gate_soft_pass",
    "de3_v4_profit_gate_catastrophic_block",
    "de3_v4_profit_gate_catastrophic_reason",
    "decision_side_model_enabled",
    "decision_side_model_name",
    "decision_side_model_application_mode",
    "decision_side_model_predicted_action",
    "decision_side_model_side_pattern",
    "decision_side_model_baseline_side_guess",
    "decision_side_model_match_count",
    "decision_side_model_long_score",
    "decision_side_model_short_score",
    "decision_side_model_no_trade_score",
    "decision_side_model_prior_score",
    "decision_side_model_prior_component_total",
    "book_gate_enabled",
    "book_gate_selected_book",
    "book_gate_scope",
    "book_gate_bucket",
    "book_gate_reason",
    "book_gate_fallback_to_default",
    "candidate_variant_filter_book",
)


class DynamicEngine3Strategy(Strategy):
    """
    Wrapper for DynamicSignalEngine3 using the trained JSON strategy database.
    """

    def __init__(self):
        self.strategy_name = "DynamicEngine3"
        self.engine = get_signal_engine()
        self.db_version = str(getattr(self.engine, "db_version", "v1") or "v1").strip().lower()
        self._de3_v2_runtime = self.db_version.startswith("v2")
        self._de3_v3_runtime = self.db_version.startswith("v3")
        self._de3_v4_runtime = self.db_version.startswith("v4")
        # DE3-specific overlays (veto/policy/meta guards, custom candle filters, drift gate)
        # are currently intended for v1 only.
        self._v1_specific_filters_enabled = not (
            self._de3_v2_runtime or self._de3_v3_runtime or self._de3_v4_runtime
        )
        self.last_processed_time = None
        self._last_5m_close = None
        self._last_15m_close = None
        self._cached_5m = None
        self._cached_15m = None
        self._candidate_cache_key = None
        self._candidate_cache_value: List[Dict] = []
        self._resampler_5m = IncrementalOHLCVResampler(5)
        self._resampler_15m = IncrementalOHLCVResampler(15)
        self._veto = None
        self._veto_cfg = CONFIG.get("DE3_CONTEXT_VETO", {}) or {}
        self._policy_cfg = CONFIG.get("DE3_ADAPTIVE_POLICY", {}) or {}
        self._de3_v3_cfg = CONFIG.get("DE3_V3", {}) or {}
        de3_v3_context_cfg = (
            self._de3_v3_cfg.get("context_profiles", {})
            if isinstance(self._de3_v3_cfg.get("context_profiles", {}), dict)
            else {}
        )
        self._de3_v3_export_raw_context_fields = bool(
            de3_v3_context_cfg.get("export_raw_context_fields_in_decision_journal", True)
        )
        self._verbose_warnings = bool(CONFIG.get("DE3_VERBOSE_WARNINGS", True))
        self._log_selection_details = bool(CONFIG.get("DE3_LOG_SELECTION_DETAILS", False))
        self._log_signal_emits = bool(CONFIG.get("DE3_LOG_SIGNAL_EMITS", False))
        self._de3_v2_entry_bar_block_cfg = (
            CONFIG.get("DE3_V2_ENTRY_BAR_HARD_BLOCKS", {}) or {}
        )
        self._de3_v2_entry_bar_block_enabled = bool(
            self._de3_v2_runtime and self._de3_v2_entry_bar_block_cfg.get("enabled", False)
        )
        self._de3_v2_constraints_cfg = (
            CONFIG.get("DE3_V2_RUNTIME_CONSTRAINTS", {}) or {}
        )
        self._de3_v2_constraints_enabled = bool(
            self._de3_v2_runtime and self._de3_v2_constraints_cfg.get("enabled", False)
        )
        try:
            self._de3_v2_min_trades = int(self._de3_v2_constraints_cfg.get("min_trades", 0) or 0)
        except Exception:
            self._de3_v2_min_trades = 0
        max_thresh_raw = self._de3_v2_constraints_cfg.get("max_thresh", None)
        try:
            self._de3_v2_max_thresh = (
                float(max_thresh_raw) if max_thresh_raw is not None else None
            )
        except Exception:
            self._de3_v2_max_thresh = None
        min_final_score_raw = self._de3_v2_constraints_cfg.get("min_final_score", None)
        try:
            self._de3_v2_min_final_score = (
                float(min_final_score_raw) if min_final_score_raw is not None else None
            )
        except Exception:
            self._de3_v2_min_final_score = None
        self._de3_v2_constraints_log = bool(
            self._de3_v2_constraints_cfg.get("log_decisions", False)
        )
        de3_v2_cfg = CONFIG.get("DE3_V2", {}) or {}
        robust_cfg = de3_v2_cfg.get("robust_ranking", {}) or {}
        self._de3_v2_robust_cfg = robust_cfg if isinstance(robust_cfg, dict) else {}
        self._de3_v2_robust_enabled = bool(
            self._de3_v2_runtime and self._de3_v2_robust_cfg.get("enabled", False)
        )
        runtime_weights = self._de3_v2_robust_cfg.get("runtime_weights", {}) or {}
        self._de3_v2_runtime_w_edge = float(runtime_weights.get("edge_points", 0.35) or 0.35)
        self._de3_v2_runtime_w_gap = float(runtime_weights.get("edge_gap", 0.20) or 0.20)
        self._de3_v2_runtime_w_struct = float(runtime_weights.get("structural_score", 0.30) or 0.30)
        self._de3_v2_runtime_w_bucket = float(runtime_weights.get("bucket_score", 0.10) or 0.10)
        self._de3_v2_runtime_w_conf = float(runtime_weights.get("confidence", 0.05) or 0.05)
        self._de3_v2_runtime_w_ambiguity = float(runtime_weights.get("ambiguity_penalty", -0.15) or -0.15)
        self._de3_v2_runtime_w_concentration = float(
            runtime_weights.get("concentration_penalty", -0.10) or -0.10
        )
        runtime_abstain_cfg = self._de3_v2_robust_cfg.get("runtime_abstain", {}) or {}
        self._de3_v2_runtime_abstain_enabled = bool(
            self._de3_v2_robust_enabled and runtime_abstain_cfg.get("enabled", True)
        )
        self._de3_v2_runtime_min_edge = float(runtime_abstain_cfg.get("min_edge_points", 0.16) or 0.16)
        self._de3_v2_runtime_min_gap = float(runtime_abstain_cfg.get("min_edge_gap_points", 0.12) or 0.12)
        self._de3_v2_runtime_min_struct = float(
            runtime_abstain_cfg.get("min_structural_score", -1.00) or -1.00
        )
        self._de3_v2_runtime_min_rank = float(
            runtime_abstain_cfg.get("min_runtime_rank_score", 0.08) or 0.08
        )
        self._de3_v2_runtime_log_decisions = bool(self._de3_v2_robust_cfg.get("log_decisions", False))
        try:
            self._de3_v2_runtime_log_top_k = max(
                1,
                int(self._de3_v2_robust_cfg.get("log_top_k", 3) or 3),
            )
        except Exception:
            self._de3_v2_runtime_log_top_k = 3
        drift_cfg = CONFIG.get("DYNAMIC_ENGINE3_DRIFT", {}) or {}
        self._drift_enabled = bool(drift_cfg.get("enabled", True))
        try:
            self._drift_max_atr = float(drift_cfg.get("max_atr", 1.0) or 1.0)
        except Exception:
            self._drift_max_atr = 1.0
        try:
            self._drift_atr_period = int(drift_cfg.get("atr_period", 14) or 14)
        except Exception:
            self._drift_atr_period = 14
        try:
            self._drift_fallback_points = float(drift_cfg.get("fallback_points", 0.0) or 0.0)
        except Exception:
            self._drift_fallback_points = 0.0
        if not np.isfinite(self._drift_max_atr):
            self._drift_max_atr = 1.0
        if not np.isfinite(self._drift_fallback_points):
            self._drift_fallback_points = 0.0
        self._drift_max_atr = max(0.0, float(self._drift_max_atr))
        self._drift_fallback_points = max(0.0, float(self._drift_fallback_points))
        if not self._v1_specific_filters_enabled:
            self._drift_enabled = False
        self._drift_anchors: Dict[Tuple[str, str, pd.Timestamp], float] = {}
        self._de3_decision_export_enabled = False
        self._de3_decision_export_top_k = 5
        self._de3_decision_export_sink: Optional[Callable[[dict], None]] = None
        self._de3_decision_export_seq = 0
        self._de3_v2_bucket_bracket_overrides: Dict[str, Tuple[float, float]] = {}
        self._de3_v3_family_runtime: Optional[Any] = None
        self._de3_v3_family_status: Dict[str, Any] = {}
        self._de3_v4_cfg = CONFIG.get("DE3_V4", {}) or {}
        de3_v4_runtime_cfg = (
            self._de3_v4_cfg.get("runtime", {})
            if isinstance(self._de3_v4_cfg.get("runtime", {}), dict)
            else {}
        )
        self._de3_v4_prune_cfg = (
            de3_v4_runtime_cfg.get("prune_rules", {})
            if isinstance(de3_v4_runtime_cfg.get("prune_rules", {}), dict)
            else {}
        )
        self._de3_v4_signal_size_cfg = (
            de3_v4_runtime_cfg.get("signal_size_rules", {})
            if isinstance(de3_v4_runtime_cfg.get("signal_size_rules", {}), dict)
            else {}
        )
        self._de3_v4_profit_gate_cfg = (
            de3_v4_runtime_cfg.get("pre_router_profit_gate_v2", {})
            if isinstance(de3_v4_runtime_cfg.get("pre_router_profit_gate_v2", {}), dict)
            else {}
        )
        def _to_float(raw: Any, default: Optional[float] = None) -> Optional[float]:
            try:
                value = float(raw)
            except Exception:
                return default
            if not np.isfinite(value):
                return default
            return float(value)

        def _to_int(raw: Any, default: int = 0) -> int:
            try:
                return int(raw)
            except Exception:
                return int(default)
        # v4 cleanup mode: disable legacy pre-router gates so router/lane logic sees
        # candidates directly unless explicitly re-enabled.
        self._de3_v4_disable_context_policy_gate = bool(
            self._de3_v4_runtime and de3_v4_runtime_cfg.get("disable_context_policy_gate", False)
        )
        self._de3_v4_disable_context_veto_gate = bool(
            self._de3_v4_runtime and de3_v4_runtime_cfg.get("disable_context_veto_gate", False)
        )
        self._de3_v4_disable_ny_conf_gate = bool(
            self._de3_v4_runtime and de3_v4_runtime_cfg.get("disable_ny_conf_gate", False)
        )
        self._de3_v4_prune_enabled = bool(
            self._de3_v4_runtime and self._de3_v4_prune_cfg.get("enabled", False)
        )
        self._de3_v4_prune_log_blocks = bool(self._de3_v4_prune_cfg.get("log_blocks", False))
        self._de3_v4_prune_backtest_active = bool(CONFIG.get("_BACKTEST_ACTIVE", False))
        self._de3_v4_signal_size_enabled = bool(
            self._de3_v4_runtime and self._de3_v4_signal_size_cfg.get("enabled", False)
        )
        self._de3_v4_signal_size_log_applies = bool(
            self._de3_v4_signal_size_cfg.get("log_applies", False)
        )
        prune_rules_raw = (
            self._de3_v4_prune_cfg.get("rules", [])
            if isinstance(self._de3_v4_prune_cfg.get("rules", []), (list, tuple))
            else []
        )
        self._de3_v4_prune_rules = [dict(r) for r in prune_rules_raw if isinstance(r, dict)]
        signal_size_rules_raw = (
            self._de3_v4_signal_size_cfg.get("rules", [])
            if isinstance(self._de3_v4_signal_size_cfg.get("rules", []), (list, tuple))
            else []
        )
        self._de3_v4_signal_size_rules = [
            dict(r) for r in signal_size_rules_raw if isinstance(r, dict)
        ]
        self._de3_v4_profit_gate_enabled = bool(
            self._de3_v4_runtime and self._de3_v4_profit_gate_cfg.get("enabled", False)
        )
        self._de3_v4_profit_gate_policy_mode = str(
            self._de3_v4_profit_gate_cfg.get("policy_mode", "block") or "block"
        ).strip().lower()
        if self._de3_v4_profit_gate_policy_mode not in {"block", "shadow"}:
            self._de3_v4_profit_gate_policy_mode = "block"
        self._de3_v4_profit_gate_disable_block_all_on_top = bool(
            self._de3_v4_profit_gate_cfg.get("disable_block_all_on_top", True)
        )
        self._de3_v4_profit_gate_soft_pass_non_catastrophic = bool(
            self._de3_v4_profit_gate_cfg.get("soft_pass_non_catastrophic_blocks", True)
        )
        self._de3_v4_profit_gate_soft_pass_risk_mult_cap = _to_float(
            self._de3_v4_profit_gate_cfg.get("soft_pass_risk_mult_cap", 0.85)
        )
        if (
            self._de3_v4_profit_gate_soft_pass_risk_mult_cap is not None
            and self._de3_v4_profit_gate_soft_pass_risk_mult_cap <= 0.0
        ):
            self._de3_v4_profit_gate_soft_pass_risk_mult_cap = None
        self._de3_v4_profit_gate_min_samples = _to_int(
            self._de3_v4_profit_gate_cfg.get("min_samples", 100)
        )
        self._de3_v4_profit_gate_max_p_loss_std = _to_float(
            self._de3_v4_profit_gate_cfg.get("max_p_loss_std", 0.30)
        )
        self._de3_v4_profit_gate_min_ev_lcb_points = _to_float(
            self._de3_v4_profit_gate_cfg.get("min_ev_lcb_points", -0.10)
        )
        self._de3_v4_profit_gate_min_ev_mean_points = _to_float(
            self._de3_v4_profit_gate_cfg.get("min_ev_mean_points", None)
        )
        cat_cfg = (
            self._de3_v4_profit_gate_cfg.get("catastrophic", {})
            if isinstance(self._de3_v4_profit_gate_cfg.get("catastrophic", {}), dict)
            else {}
        )
        self._de3_v4_profit_gate_cat_enabled = bool(cat_cfg.get("enabled", True))
        self._de3_v4_profit_gate_cat_min_samples = _to_int(cat_cfg.get("min_samples", 200))
        self._de3_v4_profit_gate_cat_max_ev_lcb_points = _to_float(
            cat_cfg.get("max_ev_lcb_points", -0.40)
        )
        self._de3_v4_profit_gate_cat_min_p_loss = _to_float(cat_cfg.get("min_p_loss", 0.62))
        self._de3_v4_profit_gate_cat_max_p_loss_std = _to_float(
            cat_cfg.get("max_p_loss_std", 0.30)
        )
        self._de3_v4_profit_gate_cat_min_p_loss_lcb = _to_float(
            cat_cfg.get("min_p_loss_lcb", None)
        )
        self._de3_v4_profit_gate_cat_max_edge_points = _to_float(
            cat_cfg.get("max_edge_points", None)
        )
        self._de3_v4_profit_gate_cat_max_final_score = _to_float(
            cat_cfg.get("max_final_score", None)
        )
        self._de3_v4_profit_gate_cat_min_trigger_count = _to_int(
            cat_cfg.get("min_trigger_count", 2), 2
        )
        if self._de3_v4_profit_gate_cat_min_trigger_count <= 0:
            self._de3_v4_profit_gate_cat_min_trigger_count = 1
        lane_overrides_raw = (
            self._de3_v4_profit_gate_cfg.get("lane_overrides", {})
            if isinstance(self._de3_v4_profit_gate_cfg.get("lane_overrides", {}), dict)
            else {}
        )
        session_overrides_raw = (
            self._de3_v4_profit_gate_cfg.get("session_overrides", {})
            if isinstance(self._de3_v4_profit_gate_cfg.get("session_overrides", {}), dict)
            else {}
        )
        lane_session_overrides_raw = (
            self._de3_v4_profit_gate_cfg.get("lane_session_overrides", {})
            if isinstance(self._de3_v4_profit_gate_cfg.get("lane_session_overrides", {}), dict)
            else {}
        )
        self._de3_v4_profit_gate_lane_overrides = {
            str(k).strip().lower(): dict(v)
            for k, v in lane_overrides_raw.items()
            if str(k).strip() and isinstance(v, dict)
        }
        self._de3_v4_profit_gate_session_overrides = {
            str(k).strip().lower(): dict(v)
            for k, v in session_overrides_raw.items()
            if str(k).strip() and isinstance(v, dict)
        }
        self._de3_v4_profit_gate_lane_session_overrides = {
            str(k).strip().lower(): dict(v)
            for k, v in lane_session_overrides_raw.items()
            if str(k).strip() and isinstance(v, dict)
        }
        self._de3_v4_runtime_module: Optional[DE3V4Runtime] = None
        self._de3_v4_status: Dict[str, Any] = {}
        if self._de3_v3_runtime:
            self._de3_v3_family_runtime = _get_de3_v3_family_runtime_cls()(self._de3_v3_cfg)
            self._de3_v3_family_status = self._de3_v3_family_runtime.get_runtime_status()
        if self._de3_v4_runtime:
            self._de3_v4_runtime_module = DE3V4Runtime(self._de3_v4_cfg)
            self._de3_v4_status = self._de3_v4_runtime_module.get_runtime_status()
        self._de3_context_eval_enabled = bool(
            self._v1_specific_filters_enabled or self._de3_v3_runtime or self._de3_v4_runtime
        )
        veto_enabled = bool(self._de3_context_eval_enabled and self._veto_cfg.get("enabled", False))
        policy_enabled = bool(self._de3_context_eval_enabled and self._policy_cfg.get("enabled", False))
        if self._de3_v4_disable_context_veto_gate:
            veto_enabled = False
        if self._de3_v4_disable_context_policy_gate:
            policy_enabled = False
        policy_mode = str(self._policy_cfg.get("mode", "block") or "block").lower()
        if policy_mode not in {"block", "shadow"}:
            policy_mode = "block"
        self.veto_stats = {
            "enabled": bool(veto_enabled or policy_enabled),
            "policy_enabled": bool(policy_enabled),
            "model_ready": False,
            "mode": str(self._veto_cfg.get("mode", "block") or "block").lower(),
            "policy_mode": policy_mode,
            "decisions": 0,
            "checked": 0,
            "blocked": 0,
            "skipped": 0,
            "policy_checked": 0,
            "policy_blocked": 0,
            "policy_shadow_would_block": 0,
            "policy_skipped": 0,
        }
        if veto_enabled or policy_enabled:
            model_path = self._policy_cfg.get("model_path") or self._veto_cfg.get(
                "model_path",
                "de3_context_veto_models.json",
            )
            self._veto = _get_de3_context_veto_cls()(model_path)
            self.veto_stats["model_ready"] = bool(self._veto and self._veto.ready)
        num_strategies = len(self.engine.strategies) if hasattr(self.engine, "strategies") else 0
        logging.info(
            (
                "DynamicEngine3Strategy initialized | %s sub-strategies loaded | db_version=%s | "
                "v1_filters=%s | v2_entry_bar_blocks=%s | v2_constraints=%s | v2_robust_rank=%s | "
                "v3_family_mode=%s | v3_family_artifact=%s | v3_family_loaded=%s | v3_family_count=%s | "
                "v3_runtime_use_refined=%s | v3_loaded_universe=%s | v3_raw_family_count=%s | v3_retained_family_count=%s | "
                "v3_raw_member_count=%s | v3_retained_member_count=%s | "
                "v3_artifact_kind=%s | v3_bundle_loaded=%s | v3_context_profiles_loaded=%s | v3_enriched_export_required=%s | "
                "v3_active_context_dims=%s | v3_runtime_state_loaded=%s"
            ),
            num_strategies,
            self.db_version,
            self._v1_specific_filters_enabled,
            self._de3_v2_entry_bar_block_enabled,
            self._de3_v2_constraints_enabled,
            self._de3_v2_robust_enabled,
            bool(self._de3_v3_runtime and self._de3_v3_family_runtime is not None),
            self._de3_v3_family_status.get("family_artifact_path"),
            self._de3_v3_family_status.get("family_artifact_loaded"),
            self._de3_v3_family_status.get("family_count"),
            self._de3_v3_family_status.get("runtime_use_refined"),
            self._de3_v3_family_status.get("loaded_universe"),
            self._de3_v3_family_status.get("raw_family_count"),
            self._de3_v3_family_status.get("retained_family_count"),
            self._de3_v3_family_status.get("raw_member_count"),
            self._de3_v3_family_status.get("retained_member_count"),
            self._de3_v3_family_status.get("artifact_kind"),
            self._de3_v3_family_status.get("bundle_loaded"),
            self._de3_v3_family_status.get("context_profiles_loaded"),
            self._de3_v3_family_status.get("enriched_export_required"),
            self._de3_v3_family_status.get("active_context_dimensions"),
            self._de3_v3_family_status.get("runtime_state_loaded"),
        )
        if self._de3_v4_runtime:
            logging.info(
                (
                    "DynamicEngine3Strategy DE3v4 mode | bundle=%s loaded=%s runtime_mode=%s "
                    "core_anchors=%s router=%s lane_selector=%s bracket_module=%s lane_count=%s lane_variants=%s "
                    "prune_enabled=%s prune_rules=%s signal_size_enabled=%s signal_size_rules=%s "
                    "policy_gate_disabled=%s veto_gate_disabled=%s ny_gate_disabled=%s "
                    "profit_gate_v2=%s profit_gate_mode=%s"
                ),
                self._de3_v4_status.get("bundle_path"),
                self._de3_v4_status.get("bundle_loaded"),
                self._de3_v4_status.get("runtime_mode"),
                self._de3_v4_status.get("core_anchor_family_ids"),
                self._de3_v4_status.get("router_enabled"),
                self._de3_v4_status.get("lane_selector_enabled"),
                self._de3_v4_status.get("bracket_module_enabled"),
                self._de3_v4_status.get("lane_count"),
                self._de3_v4_status.get("lane_variant_count"),
                bool(self._de3_v4_prune_enabled),
                int(len(self._de3_v4_prune_rules)),
                bool(self._de3_v4_signal_size_enabled),
                int(len(self._de3_v4_signal_size_rules)),
                bool(self._de3_v4_disable_context_policy_gate),
                bool(self._de3_v4_disable_context_veto_gate),
                bool(self._de3_v4_disable_ny_conf_gate),
                bool(self._de3_v4_profit_gate_enabled),
                str(self._de3_v4_profit_gate_policy_mode),
            )

    def _warn(self, msg: str, *args) -> None:
        if self._verbose_warnings:
            logging.warning(msg, *args)
        else:
            logging.debug(msg, *args)

    @staticmethod
    def _de3_v4_lane_from_cand_type(cand_type_lower: str) -> str:
        raw = str(cand_type_lower or "").strip().lower().replace("-", "_")
        if raw in {"long_rev", "short_rev", "long_mom", "short_mom"}:
            return raw
        return "unknown"

    def _de3_v4_effective_profit_gate_cfg(
        self,
        *,
        lane: str,
        session_name: str,
    ) -> Dict[str, Any]:
        lane_key = str(lane or "").strip().lower()
        session_key = str(session_name or "").strip().lower()
        merged: Dict[str, Any] = {}
        if isinstance(self._de3_v4_profit_gate_lane_overrides.get(lane_key), dict):
            merged.update(self._de3_v4_profit_gate_lane_overrides.get(lane_key, {}))
        if isinstance(self._de3_v4_profit_gate_session_overrides.get(session_key), dict):
            merged.update(self._de3_v4_profit_gate_session_overrides.get(session_key, {}))
        if lane_key and session_key:
            lane_session_key = f"{lane_key}|{session_key}"
            if isinstance(self._de3_v4_profit_gate_lane_session_overrides.get(lane_session_key), dict):
                merged.update(self._de3_v4_profit_gate_lane_session_overrides.get(lane_session_key, {}))
        return merged

    @staticmethod
    def _de3_v4_rule_match_sources(
        chosen_entry: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        primary = chosen_entry if isinstance(chosen_entry, dict) else {}
        nested = primary.get("cand") if isinstance(primary.get("cand"), dict) else None
        if isinstance(nested, dict) and nested:
            return primary, nested
        return primary, primary

    @staticmethod
    def _de3_v4_rule_pick(
        primary: Dict[str, Any],
        nested: Dict[str, Any],
        *keys: str,
    ) -> Any:
        for key in keys:
            if not key:
                continue
            for source in (primary, nested):
                if not isinstance(source, dict):
                    continue
                value = source.get(key, None)
                if value not in (None, ""):
                    return value
        return None

    def _de3_v4_pre_entry_prune_reason(
        self,
        chosen_entry: Dict[str, Any],
        *,
        current_time: Any = None,
        engine_session: str = "",
    ) -> Optional[str]:
        if not self._de3_v4_prune_enabled:
            return None
        if not isinstance(chosen_entry, dict):
            return None
        if not self._de3_v4_prune_rules:
            return None
        primary, cand = self._de3_v4_rule_match_sources(chosen_entry)
        lane_name = str(
            self._de3_v4_rule_pick(primary, cand, "de3_v4_selected_lane", "strategy_type") or ""
        ).strip()
        side_name = str(self._de3_v4_rule_pick(primary, cand, "signal", "side") or "").strip().upper()
        variant_id = str(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "de3_v4_selected_variant_id",
                "sub_strategy",
                "strategy_id",
            )
            or ""
        ).strip()
        timeframe_name = str(
            self._de3_v4_rule_pick(primary, cand, "de3_timeframe", "timeframe")
            or ""
        ).strip()
        session_name = str(
            self._de3_v4_rule_pick(primary, cand, "session")
            or engine_session
            or ""
        ).strip()
        if not session_name and variant_id:
            try:
                parts = variant_id.split("_", 3)
                if len(parts) >= 2:
                    session_name = str(parts[1] or "").strip()
            except Exception:
                session_name = ""

        def _to_float(value: Any) -> Optional[float]:
            try:
                out = float(value)
            except Exception:
                return None
            if not np.isfinite(out):
                return None
            return float(out)

        route_conf = _to_float(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "de3_v4_route_confidence",
                "de3_policy_confidence",
                "de3_edge_confidence",
            )
        )
        if route_conf is None:
            route_conf = _to_float(
                cand.get(
                    "de3_v4_route_confidence",
                    cand.get("de3_policy_confidence", cand.get("de3_edge_confidence", None)),
                )
            )
        upper_wick = _to_float(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "de3_entry_upper_wick_ratio",
                "de3_entry_upper1_ratio",
                "upper_wick_ratio",
                "upper1_ratio",
            )
        )
        lower_wick = _to_float(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "de3_entry_lower_wick_ratio",
                "lower_wick_ratio",
            )
        )
        close_pos1 = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_close_pos1", "close_pos1"))
        body1_ratio = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_body1_ratio", "body1_ratio"))
        vol1_rel20 = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_vol1_rel20", "vol1_rel20"))
        ret1_atr = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_ret1_atr", "ret1_atr"))
        range10_atr = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_range10_atr", "range10_atr"))
        dist_high5_atr = _to_float(
            self._de3_v4_rule_pick(primary, cand, "de3_entry_dist_high5_atr", "dist_high5_atr")
        )
        dist_low5_atr = _to_float(
            self._de3_v4_rule_pick(primary, cand, "de3_entry_dist_low5_atr", "dist_low5_atr")
        )
        flips5 = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_flips5", "flips5"))
        down3 = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_down3", "down3"))
        selection_score = _to_float(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "selection_score",
                "de3_selection_score",
            )
        )
        final_score = _to_float(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "final_score",
                "de3_final_score",
            )
        )

        entry_ts = None
        for ts_candidate in (
            current_time,
            primary.get("timestamp", None),
            primary.get("datetime", None),
            primary.get("bar_time", None),
            primary.get("time", None),
            cand.get("timestamp", None),
            cand.get("datetime", None),
            cand.get("bar_time", None),
            cand.get("time", None),
        ):
            if ts_candidate is None:
                continue
            try:
                parsed_ts = pd.Timestamp(ts_candidate)
            except Exception:
                continue
            if pd.isna(parsed_ts):
                continue
            entry_ts = parsed_ts
            break

        weekday_norm = ""
        quarter_norm = ""
        wom_norm = ""
        if entry_ts is not None:
            weekday_norm = str(entry_ts.strftime("%a")).strip().upper()
            try:
                quarter_norm = f"Q{int(((int(entry_ts.month) - 1) // 3) + 1)}"
            except Exception:
                quarter_norm = ""
            try:
                wom_norm = f"W{int(((int(entry_ts.day) - 1) // 7) + 1)}"
            except Exception:
                wom_norm = ""

        hour_bucket_norm = str(
            self._de3_v4_rule_pick(primary, cand, "hour_bucket")
            or ""
        ).strip()
        if not hour_bucket_norm and variant_id:
            try:
                variant_parts = variant_id.split("_", 3)
                if len(variant_parts) >= 2:
                    hour_bucket_norm = str(variant_parts[1] or "").strip()
            except Exception:
                hour_bucket_norm = ""

        lane_norm = lane_name.lower()
        timeframe_norm = timeframe_name.lower()
        session_norm = session_name.strip()
        variant_norm = variant_id.strip()
        for rule in self._de3_v4_prune_rules:
            if not bool(rule.get("enabled", True)):
                continue
            if bool(rule.get("backtest_only", False)) and not self._de3_v4_prune_backtest_active:
                continue
            if bool(rule.get("live_only", False)) and self._de3_v4_prune_backtest_active:
                continue
            rule_name = str(rule.get("name", "unnamed_prune_rule") or "unnamed_prune_rule").strip()
            apply_lanes = {
                str(v).strip().lower()
                for v in (
                    rule.get("apply_lanes", [])
                    if isinstance(rule.get("apply_lanes", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_lanes and lane_norm not in apply_lanes:
                continue
            apply_timeframes = {
                str(v).strip().lower()
                for v in (
                    rule.get("apply_timeframes", [])
                    if isinstance(rule.get("apply_timeframes", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_timeframes and timeframe_norm not in apply_timeframes:
                continue
            apply_sessions = {
                str(v).strip()
                for v in (
                    rule.get("apply_sessions", [])
                    if isinstance(rule.get("apply_sessions", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_sessions and session_norm not in apply_sessions:
                continue
            apply_weekdays = {
                str(v).strip().upper()
                for v in (
                    rule.get("apply_weekdays", [])
                    if isinstance(rule.get("apply_weekdays", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_weekdays and weekday_norm not in apply_weekdays:
                continue
            apply_quarters = {
                str(v).strip().upper()
                for v in (
                    rule.get("apply_quarters", [])
                    if isinstance(rule.get("apply_quarters", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_quarters and quarter_norm not in apply_quarters:
                continue
            apply_weeks_of_month = {
                str(v).strip().upper()
                for v in (
                    rule.get("apply_weeks_of_month", [])
                    if isinstance(rule.get("apply_weeks_of_month", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_weeks_of_month and wom_norm not in apply_weeks_of_month:
                continue
            apply_hour_buckets = {
                str(v).strip()
                for v in (
                    rule.get("apply_hour_buckets", [])
                    if isinstance(rule.get("apply_hour_buckets", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_hour_buckets and hour_bucket_norm not in apply_hour_buckets:
                continue
            apply_variants = {
                str(v).strip()
                for v in (
                    rule.get("apply_variants", [])
                    if isinstance(rule.get("apply_variants", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_variants and variant_norm not in apply_variants:
                continue
            apply_sides = {
                str(v).strip().upper()
                for v in (
                    rule.get("apply_sides", [])
                    if isinstance(rule.get("apply_sides", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_sides and side_name not in apply_sides:
                continue

            reasons: list[str] = []
            condition_count = 0

            min_route_conf = _to_float(rule.get("min_route_confidence", None))
            max_route_conf = _to_float(rule.get("max_route_confidence", None))
            if min_route_conf is not None or max_route_conf is not None:
                condition_count += 1
                if route_conf is None:
                    continue
                if min_route_conf is not None and route_conf < min_route_conf:
                    continue
                if max_route_conf is not None and route_conf > max_route_conf:
                    continue
                if min_route_conf is not None:
                    reasons.append(f"route_conf={route_conf:.4f}>={min_route_conf:.4f}")
                if max_route_conf is not None:
                    reasons.append(f"route_conf={route_conf:.4f}<={max_route_conf:.4f}")

            min_upper_wick = _to_float(rule.get("min_upper_wick_ratio", None))
            max_upper_wick = _to_float(rule.get("max_upper_wick_ratio", None))
            if min_upper_wick is not None or max_upper_wick is not None:
                condition_count += 1
                if upper_wick is None:
                    continue
                if min_upper_wick is not None and upper_wick < min_upper_wick:
                    continue
                if max_upper_wick is not None and upper_wick > max_upper_wick:
                    continue
                if min_upper_wick is not None:
                    reasons.append(f"upper_wick_ratio={upper_wick:.4f}>={min_upper_wick:.4f}")
                if max_upper_wick is not None:
                    reasons.append(f"upper_wick_ratio={upper_wick:.4f}<={max_upper_wick:.4f}")

            min_lower_wick = _to_float(rule.get("min_lower_wick_ratio", None))
            max_lower_wick = _to_float(rule.get("max_lower_wick_ratio", None))
            if min_lower_wick is not None or max_lower_wick is not None:
                condition_count += 1
                if lower_wick is None:
                    continue
                if min_lower_wick is not None and lower_wick < min_lower_wick:
                    continue
                if max_lower_wick is not None and lower_wick > max_lower_wick:
                    continue
                if min_lower_wick is not None:
                    reasons.append(f"lower_wick_ratio={lower_wick:.4f}>={min_lower_wick:.4f}")
                if max_lower_wick is not None:
                    reasons.append(f"lower_wick_ratio={lower_wick:.4f}<={max_lower_wick:.4f}")

            min_close_pos1 = _to_float(rule.get("min_close_pos1", None))
            max_close_pos1 = _to_float(rule.get("max_close_pos1", None))
            if min_close_pos1 is not None or max_close_pos1 is not None:
                condition_count += 1
                if close_pos1 is None:
                    continue
                if min_close_pos1 is not None and close_pos1 < min_close_pos1:
                    continue
                if max_close_pos1 is not None and close_pos1 > max_close_pos1:
                    continue
                if min_close_pos1 is not None:
                    reasons.append(f"close_pos1={close_pos1:.4f}>={min_close_pos1:.4f}")
                if max_close_pos1 is not None:
                    reasons.append(f"close_pos1={close_pos1:.4f}<={max_close_pos1:.4f}")

            min_body1_ratio = _to_float(rule.get("min_body1_ratio", None))
            max_body1_ratio = _to_float(rule.get("max_body1_ratio", None))
            if min_body1_ratio is not None or max_body1_ratio is not None:
                condition_count += 1
                if body1_ratio is None:
                    continue
                if min_body1_ratio is not None and body1_ratio < min_body1_ratio:
                    continue
                if max_body1_ratio is not None and body1_ratio > max_body1_ratio:
                    continue
                if min_body1_ratio is not None:
                    reasons.append(f"body1_ratio={body1_ratio:.4f}>={min_body1_ratio:.4f}")
                if max_body1_ratio is not None:
                    reasons.append(f"body1_ratio={body1_ratio:.4f}<={max_body1_ratio:.4f}")

            min_vol1_rel20 = _to_float(rule.get("min_vol1_rel20", None))
            max_vol1_rel20 = _to_float(rule.get("max_vol1_rel20", None))
            if min_vol1_rel20 is not None or max_vol1_rel20 is not None:
                condition_count += 1
                if vol1_rel20 is None:
                    continue
                if min_vol1_rel20 is not None and vol1_rel20 < min_vol1_rel20:
                    continue
                if max_vol1_rel20 is not None and vol1_rel20 > max_vol1_rel20:
                    continue
                if min_vol1_rel20 is not None:
                    reasons.append(f"vol1_rel20={vol1_rel20:.4f}>={min_vol1_rel20:.4f}")
                if max_vol1_rel20 is not None:
                    reasons.append(f"vol1_rel20={vol1_rel20:.4f}<={max_vol1_rel20:.4f}")

            min_down3 = _to_float(rule.get("min_down3", None))
            max_down3 = _to_float(rule.get("max_down3", None))
            if min_down3 is not None or max_down3 is not None:
                condition_count += 1
                if down3 is None:
                    continue
                if min_down3 is not None and down3 < min_down3:
                    continue
                if max_down3 is not None and down3 > max_down3:
                    continue
                if min_down3 is not None:
                    reasons.append(f"down3={down3:.4f}>={min_down3:.4f}")
                if max_down3 is not None:
                    reasons.append(f"down3={down3:.4f}<={max_down3:.4f}")

            min_flips5 = _to_float(rule.get("min_flips5", None))
            max_flips5 = _to_float(rule.get("max_flips5", None))
            if min_flips5 is not None or max_flips5 is not None:
                condition_count += 1
                if flips5 is None:
                    continue
                if min_flips5 is not None and flips5 < min_flips5:
                    continue
                if max_flips5 is not None and flips5 > max_flips5:
                    continue
                if min_flips5 is not None:
                    reasons.append(f"flips5={flips5:.4f}>={min_flips5:.4f}")
                if max_flips5 is not None:
                    reasons.append(f"flips5={flips5:.4f}<={max_flips5:.4f}")

            min_ret1_atr = _to_float(rule.get("min_ret1_atr", None))
            max_ret1_atr = _to_float(rule.get("max_ret1_atr", None))
            if min_ret1_atr is not None or max_ret1_atr is not None:
                condition_count += 1
                if ret1_atr is None:
                    continue
                if min_ret1_atr is not None and ret1_atr < min_ret1_atr:
                    continue
                if max_ret1_atr is not None and ret1_atr > max_ret1_atr:
                    continue
                if min_ret1_atr is not None:
                    reasons.append(f"ret1_atr={ret1_atr:.4f}>={min_ret1_atr:.4f}")
                if max_ret1_atr is not None:
                    reasons.append(f"ret1_atr={ret1_atr:.4f}<={max_ret1_atr:.4f}")

            min_range10_atr = _to_float(rule.get("min_range10_atr", None))
            max_range10_atr = _to_float(rule.get("max_range10_atr", None))
            if min_range10_atr is not None or max_range10_atr is not None:
                condition_count += 1
                if range10_atr is None:
                    continue
                if min_range10_atr is not None and range10_atr < min_range10_atr:
                    continue
                if max_range10_atr is not None and range10_atr > max_range10_atr:
                    continue
                if min_range10_atr is not None:
                    reasons.append(f"range10_atr={range10_atr:.4f}>={min_range10_atr:.4f}")
                if max_range10_atr is not None:
                    reasons.append(f"range10_atr={range10_atr:.4f}<={max_range10_atr:.4f}")

            min_dist_high5_atr = _to_float(rule.get("min_dist_high5_atr", None))
            max_dist_high5_atr = _to_float(rule.get("max_dist_high5_atr", None))
            if min_dist_high5_atr is not None or max_dist_high5_atr is not None:
                condition_count += 1
                if dist_high5_atr is None:
                    continue
                if min_dist_high5_atr is not None and dist_high5_atr < min_dist_high5_atr:
                    continue
                if max_dist_high5_atr is not None and dist_high5_atr > max_dist_high5_atr:
                    continue
                if min_dist_high5_atr is not None:
                    reasons.append(f"dist_high5_atr={dist_high5_atr:.4f}>={min_dist_high5_atr:.4f}")
                if max_dist_high5_atr is not None:
                    reasons.append(f"dist_high5_atr={dist_high5_atr:.4f}<={max_dist_high5_atr:.4f}")

            min_dist_low5_atr = _to_float(rule.get("min_dist_low5_atr", None))
            max_dist_low5_atr = _to_float(rule.get("max_dist_low5_atr", None))
            if min_dist_low5_atr is not None or max_dist_low5_atr is not None:
                condition_count += 1
                if dist_low5_atr is None:
                    continue
                if min_dist_low5_atr is not None and dist_low5_atr < min_dist_low5_atr:
                    continue
                if max_dist_low5_atr is not None and dist_low5_atr > max_dist_low5_atr:
                    continue
                if min_dist_low5_atr is not None:
                    reasons.append(f"dist_low5_atr={dist_low5_atr:.4f}>={min_dist_low5_atr:.4f}")
                if max_dist_low5_atr is not None:
                    reasons.append(f"dist_low5_atr={dist_low5_atr:.4f}<={max_dist_low5_atr:.4f}")

            min_selection_score = _to_float(rule.get("min_selection_score", None))
            max_selection_score = _to_float(rule.get("max_selection_score", None))
            if min_selection_score is not None or max_selection_score is not None:
                condition_count += 1
                if selection_score is None:
                    continue
                if min_selection_score is not None and selection_score < min_selection_score:
                    continue
                if max_selection_score is not None and selection_score > max_selection_score:
                    continue
                if min_selection_score is not None:
                    reasons.append(f"selection_score={selection_score:.4f}>={min_selection_score:.4f}")
                if max_selection_score is not None:
                    reasons.append(f"selection_score={selection_score:.4f}<={max_selection_score:.4f}")

            min_final_score = _to_float(rule.get("min_final_score", None))
            max_final_score = _to_float(rule.get("max_final_score", None))
            if min_final_score is not None or max_final_score is not None:
                condition_count += 1
                if final_score is None:
                    continue
                if min_final_score is not None and final_score < min_final_score:
                    continue
                if max_final_score is not None and final_score >= max_final_score:
                    continue
                if min_final_score is not None:
                    reasons.append(f"final_score={final_score:.4f}>={min_final_score:.4f}")
                if max_final_score is not None:
                    reasons.append(f"final_score={final_score:.4f}<{max_final_score:.4f}")

            if condition_count <= 0 and bool(rule.get("match_scope_only", False)):
                return f"v4_prune:{rule_name}:scope_only"
            if condition_count <= 0:
                continue
            if reasons:
                return f"v4_prune:{rule_name}:{','.join(reasons)}"
        return None

    def _de3_v4_apply_signal_size_rules(
        self,
        chosen_entry: Dict[str, Any],
        *,
        current_time: Any = None,
        engine_session: str = "",
        requested_size: int,
    ) -> Tuple[int, List[Dict[str, Any]]]:
        if not self._de3_v4_signal_size_enabled:
            return int(max(1, requested_size)), []
        if not isinstance(chosen_entry, dict):
            return int(max(1, requested_size)), []
        if not self._de3_v4_signal_size_rules:
            return int(max(1, requested_size)), []

        primary, cand = self._de3_v4_rule_match_sources(chosen_entry)
        lane_name = str(
            self._de3_v4_rule_pick(primary, cand, "de3_v4_selected_lane", "strategy_type") or ""
        ).strip()
        side_name = str(self._de3_v4_rule_pick(primary, cand, "signal", "side") or "").strip().upper()
        variant_id = str(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "de3_v4_selected_variant_id",
                "sub_strategy",
                "strategy_id",
            )
            or ""
        ).strip()
        timeframe_name = str(
            self._de3_v4_rule_pick(primary, cand, "de3_timeframe", "timeframe")
            or ""
        ).strip()
        session_name = str(
            self._de3_v4_rule_pick(primary, cand, "session")
            or engine_session
            or ""
        ).strip()
        if not session_name and variant_id:
            try:
                parts = variant_id.split("_", 3)
                if len(parts) >= 2:
                    session_name = str(parts[1] or "").strip()
            except Exception:
                session_name = ""

        def _to_float(value: Any) -> Optional[float]:
            try:
                out = float(value)
            except Exception:
                return None
            if not np.isfinite(out):
                return None
            return float(out)

        route_conf = _to_float(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "de3_v4_route_confidence",
                "de3_policy_confidence",
                "de3_edge_confidence",
            )
        )
        if route_conf is None:
            route_conf = _to_float(
                cand.get(
                    "de3_v4_route_confidence",
                    cand.get("de3_policy_confidence", cand.get("de3_edge_confidence", None)),
                )
            )
        upper_wick = _to_float(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "de3_entry_upper_wick_ratio",
                "de3_entry_upper1_ratio",
                "upper_wick_ratio",
                "upper1_ratio",
            )
        )
        lower_wick = _to_float(
            self._de3_v4_rule_pick(
                primary,
                cand,
                "de3_entry_lower_wick_ratio",
                "lower_wick_ratio",
            )
        )
        close_pos1 = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_close_pos1", "close_pos1"))
        body1_ratio = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_body1_ratio", "body1_ratio"))
        vol1_rel20 = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_vol1_rel20", "vol1_rel20"))
        range10_atr = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_range10_atr", "range10_atr"))
        dist_high5_atr = _to_float(
            self._de3_v4_rule_pick(primary, cand, "de3_entry_dist_high5_atr", "dist_high5_atr")
        )
        dist_low5_atr = _to_float(
            self._de3_v4_rule_pick(primary, cand, "de3_entry_dist_low5_atr", "dist_low5_atr")
        )
        flips5 = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_flips5", "flips5"))
        down3 = _to_float(self._de3_v4_rule_pick(primary, cand, "de3_entry_down3", "down3"))

        entry_ts = None
        for ts_candidate in (
            current_time,
            primary.get("timestamp", None),
            primary.get("datetime", None),
            primary.get("bar_time", None),
            primary.get("time", None),
            cand.get("timestamp", None),
            cand.get("datetime", None),
            cand.get("bar_time", None),
            cand.get("time", None),
        ):
            if ts_candidate is None:
                continue
            try:
                parsed_ts = pd.Timestamp(ts_candidate)
            except Exception:
                continue
            if pd.isna(parsed_ts):
                continue
            entry_ts = parsed_ts
            break

        weekday_norm = ""
        quarter_norm = ""
        wom_norm = ""
        if entry_ts is not None:
            weekday_norm = str(entry_ts.strftime("%a")).strip().upper()
            try:
                quarter_norm = f"Q{int(((int(entry_ts.month) - 1) // 3) + 1)}"
            except Exception:
                quarter_norm = ""
            try:
                wom_norm = f"W{int(((int(entry_ts.day) - 1) // 7) + 1)}"
            except Exception:
                wom_norm = ""

        hour_bucket_norm = str(self._de3_v4_rule_pick(primary, cand, "hour_bucket") or "").strip()
        if not hour_bucket_norm and variant_id:
            try:
                variant_parts = variant_id.split("_", 3)
                if len(variant_parts) >= 2:
                    hour_bucket_norm = str(variant_parts[1] or "").strip()
            except Exception:
                hour_bucket_norm = ""

        lane_norm = lane_name.lower()
        timeframe_norm = timeframe_name.lower()
        session_norm = session_name.strip()
        variant_norm = variant_id.strip()
        current_size = max(1, int(requested_size))
        applied: List[Dict[str, Any]] = []

        for rule in self._de3_v4_signal_size_rules:
            if not bool(rule.get("enabled", True)):
                continue
            if bool(rule.get("backtest_only", False)) and not self._de3_v4_prune_backtest_active:
                continue
            if bool(rule.get("live_only", False)) and self._de3_v4_prune_backtest_active:
                continue

            apply_lanes = {
                str(v).strip().lower()
                for v in (
                    rule.get("apply_lanes", [])
                    if isinstance(rule.get("apply_lanes", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_lanes and lane_norm not in apply_lanes:
                continue
            apply_timeframes = {
                str(v).strip().lower()
                for v in (
                    rule.get("apply_timeframes", [])
                    if isinstance(rule.get("apply_timeframes", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_timeframes and timeframe_norm not in apply_timeframes:
                continue
            apply_sessions = {
                str(v).strip()
                for v in (
                    rule.get("apply_sessions", [])
                    if isinstance(rule.get("apply_sessions", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_sessions and session_norm not in apply_sessions:
                continue
            apply_weekdays = {
                str(v).strip().upper()
                for v in (
                    rule.get("apply_weekdays", [])
                    if isinstance(rule.get("apply_weekdays", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_weekdays and weekday_norm not in apply_weekdays:
                continue
            apply_quarters = {
                str(v).strip().upper()
                for v in (
                    rule.get("apply_quarters", [])
                    if isinstance(rule.get("apply_quarters", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_quarters and quarter_norm not in apply_quarters:
                continue
            apply_weeks_of_month = {
                str(v).strip().upper()
                for v in (
                    rule.get("apply_weeks_of_month", [])
                    if isinstance(rule.get("apply_weeks_of_month", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_weeks_of_month and wom_norm not in apply_weeks_of_month:
                continue
            apply_hour_buckets = {
                str(v).strip()
                for v in (
                    rule.get("apply_hour_buckets", [])
                    if isinstance(rule.get("apply_hour_buckets", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_hour_buckets and hour_bucket_norm not in apply_hour_buckets:
                continue
            apply_variants = {
                str(v).strip()
                for v in (
                    rule.get("apply_variants", [])
                    if isinstance(rule.get("apply_variants", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_variants and variant_norm not in apply_variants:
                continue
            apply_sides = {
                str(v).strip().upper()
                for v in (
                    rule.get("apply_sides", [])
                    if isinstance(rule.get("apply_sides", []), (list, tuple, set))
                    else []
                )
                if str(v).strip()
            }
            if apply_sides and side_name not in apply_sides:
                continue

            reasons: List[str] = []
            condition_count = 0

            checks = (
                ("min_route_confidence", route_conf, ">="),
                ("max_route_confidence", route_conf, "<="),
                ("min_upper_wick_ratio", upper_wick, ">="),
                ("max_upper_wick_ratio", upper_wick, "<="),
                ("min_lower_wick_ratio", lower_wick, ">="),
                ("max_lower_wick_ratio", lower_wick, "<="),
                ("min_close_pos1", close_pos1, ">="),
                ("max_close_pos1", close_pos1, "<="),
                ("min_body1_ratio", body1_ratio, ">="),
                ("max_body1_ratio", body1_ratio, "<="),
                ("min_vol1_rel20", vol1_rel20, ">="),
                ("max_vol1_rel20", vol1_rel20, "<="),
                ("min_flips5", flips5, ">="),
                ("max_flips5", flips5, "<="),
                ("min_down3", down3, ">="),
                ("max_down3", down3, "<="),
                ("min_range10_atr", range10_atr, ">="),
                ("max_range10_atr", range10_atr, "<="),
                ("min_dist_high5_atr", dist_high5_atr, ">="),
                ("max_dist_high5_atr", dist_high5_atr, "<="),
                ("min_dist_low5_atr", dist_low5_atr, ">="),
                ("max_dist_low5_atr", dist_low5_atr, "<="),
            )
            matched = True
            for key, observed, op in checks:
                bound = _to_float(rule.get(key, None))
                if bound is None:
                    continue
                condition_count += 1
                if observed is None:
                    matched = False
                    break
                if op == ">=" and observed < bound:
                    matched = False
                    break
                if op == "<=" and observed > bound:
                    matched = False
                    break
                reasons.append(f"{key}({observed:.4f},{bound:.4f})")
            if not matched or condition_count <= 0:
                continue

            rule_name = str(rule.get("name", "unnamed_signal_size_rule") or "unnamed_signal_size_rule").strip()
            size_multiplier = max(0.0, _to_float(rule.get("size_multiplier", 1.0)) or 1.0)
            rule_min_contracts = max(1, int(rule.get("min_contracts", 1) or 1))
            candidate_size = int(math.floor(float(current_size) * float(size_multiplier)))
            if candidate_size < rule_min_contracts:
                candidate_size = int(rule_min_contracts)
            if candidate_size < 1:
                candidate_size = 1
            if candidate_size >= current_size:
                continue

            applied.append(
                {
                    "name": rule_name,
                    "reasons": list(reasons),
                    "size_multiplier": float(size_multiplier),
                    "from_size": int(current_size),
                    "to_size": int(candidate_size),
                }
            )
            current_size = int(candidate_size)
            if self._de3_v4_signal_size_log_applies:
                logging.info(
                    "DE3 v4 signal-size rule | rule=%s size %s->%s sub=%s side=%s",
                    rule_name,
                    applied[-1]["from_size"],
                    applied[-1]["to_size"],
                    variant_norm,
                    side_name,
                )

        return int(current_size), applied

    def configure_decision_journal_export(
        self,
        *,
        enabled: bool,
        top_k: int = 5,
        sink: Optional[Callable[[dict], None]] = None,
    ) -> None:
        if enabled and callable(sink):
            self._de3_decision_export_enabled = True
            self._de3_decision_export_sink = sink
        else:
            self._de3_decision_export_enabled = False
            self._de3_decision_export_sink = None
        try:
            self._de3_decision_export_top_k = max(1, int(top_k))
        except Exception:
            self._de3_decision_export_top_k = 5

    def configure_de3_v2_bucket_bracket_overrides(
        self,
        overrides: Optional[Dict[str, dict]],
    ) -> None:
        clean: Dict[str, Tuple[float, float]] = {}
        if isinstance(overrides, dict):
            for raw_key, raw_payload in overrides.items():
                key = str(raw_key or "").strip()
                if not key:
                    continue
                payload = raw_payload if isinstance(raw_payload, dict) else {}
                try:
                    sl_val = float(payload.get("sl", np.nan))
                    tp_val = float(payload.get("tp", np.nan))
                except Exception:
                    continue
                if not (np.isfinite(sl_val) and np.isfinite(tp_val)):
                    continue
                if sl_val <= 0.0 or tp_val <= 0.0:
                    continue
                clean[key] = (float(sl_val), float(tp_val))
        self._de3_v2_bucket_bracket_overrides = clean

    def _next_de3_decision_id(self, ts: pd.Timestamp) -> str:
        self._de3_decision_export_seq += 1
        ts_ns = int(pd.Timestamp(ts).value)
        return f"de3d_{ts_ns}_{self._de3_decision_export_seq}"

    @staticmethod
    def _flatten_de3_context_fields(context_inputs: Any) -> Dict[str, Any]:
        ctx = context_inputs if isinstance(context_inputs, dict) else {}
        return {
            "ctx_volatility_regime": ctx.get("volatility_regime"),
            "ctx_chop_trend_regime": ctx.get("chop_trend_regime"),
            "ctx_compression_expansion_regime": ctx.get("compression_expansion_regime"),
            "ctx_confidence_band": ctx.get("confidence_band"),
            "ctx_rvol_liquidity_state": ctx.get("rvol_liquidity_state"),
            "ctx_session_substate": ctx.get("session_substate"),
            "ctx_atr_ratio": ctx.get("atr_ratio"),
            "ctx_vwap_dist_atr": ctx.get("vwap_dist_atr"),
            "ctx_price_location": ctx.get("price_location"),
            "ctx_rvol_ratio": ctx.get("rvol_ratio"),
            "ctx_hour_et": ctx.get("hour_et"),
        }

    def _emit_de3_decision_journal(
        self,
        *,
        current_time: pd.Timestamp,
        session_name: str,
        feasible_candidates: List[Dict[str, object]],
        export_candidates: Optional[List[Dict[str, object]]] = None,
        chosen_cand_id: Optional[str],
        abstained: bool,
        abstain_reason: str = "",
    ) -> Optional[str]:
        candidate_rows = (
            [row for row in export_candidates if isinstance(row, dict)]
            if isinstance(export_candidates, list)
            else [row for row in feasible_candidates if isinstance(row, dict)]
        )
        if (
            (not self._de3_decision_export_enabled)
            or ((not self._de3_v2_runtime) and (not self._de3_v3_runtime) and (not self._de3_v4_runtime))
            or (self._de3_decision_export_sink is None)
            or (not candidate_rows)
        ):
            return None

        decision_id = self._next_de3_decision_id(current_time)
        if self._de3_v3_runtime:
            top_k = int(len(candidate_rows))
        else:
            top_k = min(max(1, self._de3_decision_export_top_k), len(candidate_rows))
        ts_iso = pd.Timestamp(current_time).isoformat()
        chosen_norm = str(chosen_cand_id or "").strip()
        abstain_text = str(abstain_reason or "").strip()
        session_out = str(session_name or "")
        sink = self._de3_decision_export_sink
        de3_version = str(self.db_version or "")
        family_mode = bool(self._de3_v3_runtime)
        family_artifact = (
            self._de3_v3_family_status.get("family_artifact_path")
            if isinstance(self._de3_v3_family_status, dict)
            else None
        )
        try:
            for rank, entry in enumerate(candidate_rows[:top_k], start=1):
                cand = entry.get("cand") or {}
                cand_id = str(entry.get("cand_id", cand.get("strategy_id", "")) or "")
                side_val = str(cand.get("signal", "") or "").strip().lower()
                if side_val not in {"long", "short"}:
                    side_val = ""
                context_inputs = entry.get("family_context_inputs", {})
                context_flat = (
                    self._flatten_de3_context_fields(context_inputs)
                    if self._de3_v3_export_raw_context_fields
                    else {}
                )
                out_row = {
                    "decision_id": decision_id,
                    "timestamp": ts_iso,
                    "de3_version": de3_version,
                    "family_mode": bool(family_mode),
                    "session": session_out,
                    "side_considered": side_val,
                    "chosen": bool((not abstained) and chosen_norm and cand_id == chosen_norm),
                    "abstained": bool(abstained),
                    "abstain_reason": abstain_text,
                    "rank": int(rank),
                    "family_rank": entry.get("family_rank"),
                    "family_id": entry.get("family_id", cand.get("de3_family_id")),
                    "chosen_family_id": entry.get("chosen_family_id"),
                    "family_score": entry.get("family_score"),
                    "family_context_ev": entry.get("family_context_ev"),
                    "family_confidence": entry.get("family_confidence"),
                    "family_prior": entry.get("family_prior"),
                    "family_profile": entry.get("family_profile"),
                    "family_member_count": entry.get("family_member_count"),
                    "feasible_family_count": entry.get("feasible_family_count"),
                    "feasible_family_ids": entry.get("feasible_family_ids"),
                    "family_context_inputs": entry.get("family_context_inputs"),
                    "family_artifact": entry.get("family_artifact", family_artifact),
                    "canonical_member_id": entry.get("canonical_member_id"),
                    "member_local_score": entry.get("member_local_score"),
                    "family_context_support_ratio": entry.get("family_context_support_ratio"),
                    "family_context_support_tier": entry.get("family_context_support_tier"),
                    "family_local_support_tier": entry.get("family_local_support_tier"),
                    "family_context_sample_count": entry.get("family_context_sample_count"),
                    "family_context_weight": entry.get("family_context_weight"),
                    "family_context_trusted": entry.get("family_context_trusted"),
                    "family_context_fallback_priors": entry.get("family_context_fallback_priors"),
                    "family_active_context_buckets": entry.get("family_active_context_buckets"),
                    "family_profile_used": entry.get("family_profile_used"),
                    "family_profile_fallback": entry.get("family_profile_fallback"),
                    "family_usability_state": entry.get("family_usability_state"),
                    "family_usability_component": entry.get("family_usability_component"),
                    "family_evidence_support_tier": entry.get("family_evidence_support_tier"),
                    "family_competition_status": entry.get("family_competition_status"),
                    "family_usability_adjustment": entry.get("family_usability_adjustment"),
                    "family_suppression_reason": entry.get("family_suppression_reason"),
                    "base_family_score": entry.get("base_family_score"),
                    "diversity_adjustment": entry.get("diversity_adjustment"),
                    "competition_diversity_adjustment": entry.get("competition_diversity_adjustment", entry.get("diversity_adjustment")),
                    "final_family_score": entry.get("final_family_score", entry.get("family_score")),
                    "recent_chosen_share": entry.get("recent_chosen_share"),
                    "exploration_bonus": entry.get("exploration_bonus"),
                    "dominance_penalty": entry.get("dominance_penalty"),
                    "exploration_bonus_applied": entry.get("exploration_bonus_applied"),
                    "dominance_penalty_applied": entry.get("dominance_penalty_applied"),
                    "competition_margin_qualified": entry.get("competition_margin_qualified"),
                    "context_advantage_capped": entry.get("context_advantage_capped"),
                    "context_advantage_cap_delta": entry.get("context_advantage_cap_delta"),
                    "close_competition_decision": entry.get("close_competition_decision"),
                    "bootstrap_competition_used_decision": entry.get("bootstrap_competition_used_decision"),
                    "family_monopoly_active": entry.get("family_monopoly_active"),
                    "family_monopoly_top_share": entry.get("family_monopoly_top_share"),
                    "family_monopoly_top_family_id": entry.get("family_monopoly_top_family_id"),
                    "family_monopoly_unique_count": entry.get("family_monopoly_unique_count"),
                    "monopoly_canonical_force_applied": entry.get("monopoly_canonical_force_applied"),
                    "family_chosen_flag": entry.get("family_chosen_flag"),
                    "local_bracket_adaptation_mode": entry.get("local_bracket_adaptation_mode"),
                    "local_bracket_adaptation_enabled": entry.get("local_bracket_adaptation_enabled"),
                    "local_bracket_override_applied": entry.get("local_bracket_override_applied"),
                    "family_prior_eligible": entry.get("family_prior_eligible"),
                    "family_prior_eligibility_reason": entry.get("family_prior_eligibility_reason"),
                    "family_competition_eligible": entry.get("family_competition_eligible"),
                    "family_competition_eligibility_reason": entry.get("family_competition_eligibility_reason"),
                    "family_bootstrap_competition_included": entry.get("family_bootstrap_competition_included"),
                    "family_bootstrap_included": entry.get("family_bootstrap_included"),
                    "family_catastrophic_prior": entry.get("family_catastrophic_prior"),
                    # Backward-compatible aliases.
                    "family_eligible": entry.get("family_eligible"),
                    "family_eligibility_reason": entry.get("family_eligibility_reason"),
                    "sub_strategy": cand_id,
                    "timeframe": str(cand.get("timeframe", "") or ""),
                    "strategy_type": str(cand.get("strategy_type", "") or ""),
                    "thresh": cand.get("thresh"),
                    "sl": cand.get("sl"),
                    "tp": cand.get("tp"),
                    "edge_points": entry.get("edge_points"),
                    "edge_gap_points": entry.get("edge_gap"),
                    "runtime_rank_score": entry.get("runtime_rank_score"),
                    "structural_score": entry.get("structural_score"),
                    "final_score": cand.get("final_score"),
                    "bucket_score": cand.get("score_raw", cand.get("final_score")),
                    "stop_like_share": entry.get(
                        "stop_like_share", cand.get("oos_stop_like_share")
                    ),
                    "loss_share": entry.get("loss_share", cand.get("oos_loss_share")),
                    "profitable_block_ratio": entry.get(
                        "profitable_block_ratio",
                        cand.get("ProfitableBlockRatio", cand.get("profitable_block_ratio")),
                    ),
                    "worst_block_avg_pnl": entry.get(
                        "worst_block_avg_pnl",
                        cand.get("WorstBlockAvgPnL", cand.get("worst_block_avg_pnl")),
                    ),
                    "worst_block_pf": entry.get(
                        "worst_block_pf",
                        cand.get("WorstBlockPF", cand.get("worst_block_pf")),
                    ),
                    **context_flat,
                }
                # Preserve DE3v3 score-path decomposition fields exactly as emitted by
                # the family-first runtime journal builder.
                de3_v3_passthrough_fields = [
                    "family_candidate_count",
                    "family_candidate_set_gt_1",
                    "family_candidate_source",
                    "family_runtime_role",
                    "is_core_family",
                    "is_satellite_family",
                    "candidate_rank_before_adjustments",
                    "choice_path_mode",
                    "score_path_inconsistency_flag",
                    "score_post_processing_applied",
                    "any_post_processing_adjustment_applied",
                    "runner_up_available",
                    "chosen_vs_runner_up_score_delta",
                    "chosen_prior_component",
                    "chosen_trusted_context_component",
                    "chosen_evidence_adjustment",
                    "chosen_adaptive_component",
                    "chosen_competition_diversity_adjustment",
                    "chosen_family_compatibility_component",
                    "chosen_pre_adjustment_score",
                    "chosen_final_family_score",
                    "chosen_context_trusted",
                    "chosen_support_tier",
                    "chosen_compatibility_tier",
                    "chosen_session_compatibility_tier",
                    "chosen_timeframe_compatibility_tier",
                    "chosen_strategy_type_compatibility_tier",
                    "chosen_exploration_bonus_applied",
                    "chosen_dominance_penalty_applied",
                    "chosen_context_advantage_capped",
                    "runner_up_family_id",
                    "runner_up_prior_component",
                    "runner_up_trusted_context_component",
                    "runner_up_evidence_adjustment",
                    "runner_up_adaptive_component",
                    "runner_up_competition_diversity_adjustment",
                    "runner_up_family_compatibility_component",
                    "runner_up_pre_adjustment_score",
                    "runner_up_final_family_score",
                    "runner_up_context_trusted",
                    "runner_up_support_tier",
                    "runner_up_compatibility_tier",
                    "runner_up_session_compatibility_tier",
                    "runner_up_timeframe_compatibility_tier",
                    "runner_up_strategy_type_compatibility_tier",
                    "runner_up_exploration_bonus_applied",
                    "runner_up_dominance_penalty_applied",
                    "runner_up_context_advantage_capped",
                    "family_compatibility_component",
                    "family_compatibility_tier",
                    "family_session_compatibility_tier",
                    "family_timeframe_compatibility_tier",
                    "family_strategy_type_compatibility_tier",
                    "family_entered_via_compatible_band",
                    "family_exact_match_eligible",
                    "family_compatible_band_eligible",
                    "family_incompatible_excluded",
                    "family_excluded_by_temporary_exclusion",
                    "family_excluded_by_candidate_cap",
                    "family_eligibility_tier",
                    "family_preliminary_family_score",
                    "family_preliminary_compatibility_penalty_component",
                    "family_entered_pre_cap_pool",
                    "family_survived_cap",
                    "family_cap_drop_reason",
                    "family_cap_tier_slot_used",
                    "family_final_competition_pool_flag",
                    "pre_cap_candidate_count",
                    "post_cap_candidate_count",
                    "exact_match_survived_count",
                    "compatible_band_survived_count",
                    "compatible_band_dropped_by_cap_count",
                    "local_member_count_within_family",
                    "local_edge_component",
                    "local_structural_component",
                    "local_bracket_suitability_component",
                    "local_confidence_component",
                    "local_payoff_component",
                    "local_final_member_score",
                    "canonical_fallback_used",
                    "why_non_anchor_beat_anchor",
                    "why_anchor_forced",
                    "no_local_alternative",
                ]
                for key in de3_v3_passthrough_fields:
                    if key in entry:
                        out_row[key] = entry.get(key)
                for key in DE3_V4_EXPORT_FIELDS:
                    if key in entry:
                        out_row[key] = entry.get(key)
                    elif key in cand:
                        out_row[key] = cand.get(key)
                for key, value in cand.items():
                    if str(key).startswith("de3_entry_"):
                        out_row[key] = value
                if bool(out_row.get("family_mode", False)):
                    snap = _canonical_context_usage_snapshot(out_row)
                    out_row["family_context_support_tier"] = str(snap.get("support_tier", "low"))
                    out_row["family_context_trusted"] = bool(snap.get("trusted_context_used", False))
                    out_row["family_context_fallback_priors"] = bool(snap.get("fallback_to_priors", True))
                    out_row["local_bracket_adaptation_mode"] = str(snap.get("local_bracket_mode", "none"))
                    out_row["local_bracket_adaptation_enabled"] = bool(snap.get("local_bracket_mode_enabled", False))
                    out_row["local_bracket_override_applied"] = bool(snap.get("local_bracket_override_applied", False))
                sink(out_row)
        except Exception:
            pass
        return decision_id

    def _emit_de3_v3_family_decision_journal(
        self,
        *,
        current_time: pd.Timestamp,
        session_name: str,
        family_rows: List[Dict[str, Any]],
        chosen_family_id: str,
        chosen_entry: Optional[Dict[str, Any]],
        chosen_member_local_score: Optional[float],
        context_inputs: Dict[str, Any],
        abstained: bool,
        abstain_reason: str,
    ) -> Optional[str]:
        if not family_rows:
            return None
        family_ids = [str(row.get("family_id", "")) for row in family_rows if str(row.get("family_id", "")).strip()]
        feasible_ids_joined = ",".join(family_ids)
        journal_rows: List[Dict[str, Any]] = []
        chosen_cand_id = ""
        if not abstained and isinstance(chosen_entry, dict):
            chosen_cand_id = str(chosen_entry.get("cand_id", "") or "")
        competitive_rows = [row for row in family_rows if bool(row.get("competition_eligible", False))]

        def _score_decomp(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not isinstance(row, dict):
                return {
                    "family_id": "",
                    "prior_component": 0.0,
                    "trusted_context_component": 0.0,
                    "evidence_adjustment": 0.0,
                    "adaptive_component": 0.0,
                    "competition_diversity_adjustment": 0.0,
                    "family_compatibility_component": 0.0,
                    "pre_adjustment_score": 0.0,
                    "final_family_score": 0.0,
                    "context_trusted": False,
                    "support_tier": "low",
                    "exploration_bonus_applied": False,
                    "dominance_penalty_applied": False,
                    "context_advantage_capped": False,
                    "compatibility_tier": "incompatible",
                    "session_compatibility_tier": "incompatible",
                    "timeframe_compatibility_tier": "incompatible",
                    "strategy_type_compatibility_tier": "incompatible",
                    "entered_via_compatible_band": False,
                    "any_post_processing_applied": False,
                    "chosen_flag": False,
                    "runner_up_flag": False,
                }
            score_obj = (
                row.get("family_score_object")
                if isinstance(row.get("family_score_object"), dict)
                else {}
            )
            comps = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
            if not score_obj:
                score_obj = {
                    "family_id": str(row.get("family_id", "") or ""),
                    "prior_component": float(comps.get("family_prior_component_weighted", comps.get("family_prior_component", 0.0)) or 0.0),
                    "trusted_context_component": float(comps.get("context_total_component", 0.0) or 0.0),
                    "evidence_adjustment": float(comps.get("v3_realized_usability_component", 0.0) or 0.0),
                    "adaptive_component": float(comps.get("adaptive_policy_component_weighted", comps.get("adaptive_policy_component", 0.0)) or 0.0),
                    "competition_diversity_adjustment": float(
                        comps.get("competition_diversity_adjustment", row.get("diversity_adjustment", 0.0)) or 0.0
                    ),
                    "family_compatibility_component": float(
                        comps.get("family_compatibility_component", row.get("family_compatibility_component", 0.0)) or 0.0
                    ),
                    "pre_adjustment_score": float(row.get("base_family_score", comps.get("base_family_score", 0.0)) or 0.0),
                    "final_family_score": float(row.get("family_score", comps.get("final_family_score", 0.0)) or 0.0),
                    "context_trusted_flag": bool(comps.get("trusted_context_used", row.get("profile_trusted", False))),
                    "support_tier": str(
                        row.get(
                            "family_local_support_tier",
                            comps.get("context_support_tier", row.get("context_support_tier", "low")),
                        )
                        or "low"
                    ),
                    "exploration_bonus_applied": bool(comps.get("exploration_bonus_applied", row.get("exploration_bonus_applied", False))),
                    "dominance_penalty_applied": bool(comps.get("dominance_penalty_applied", row.get("dominance_penalty_applied", False))),
                    "context_advantage_capped": bool(comps.get("context_advantage_capped", row.get("context_advantage_capped", False))),
                    "compatibility_tier": str(
                        row.get("compatibility_tier", comps.get("compatibility_tier", "incompatible"))
                        or "incompatible"
                    ),
                    "session_compatibility_tier": str(
                        row.get(
                            "session_compatibility_tier",
                            comps.get("session_compatibility_tier", "incompatible"),
                        )
                        or "incompatible"
                    ),
                    "timeframe_compatibility_tier": str(
                        row.get(
                            "timeframe_compatibility_tier",
                            comps.get("timeframe_compatibility_tier", "incompatible"),
                        )
                        or "incompatible"
                    ),
                    "strategy_type_compatibility_tier": str(
                        row.get(
                            "strategy_type_compatibility_tier",
                            comps.get("strategy_type_compatibility_tier", "incompatible"),
                        )
                        or "incompatible"
                    ),
                    "entered_via_compatible_band": bool(
                        row.get(
                            "entered_via_compatible_band",
                            comps.get("entered_via_compatible_band", False),
                        )
                    ),
                    "any_post_processing_applied": bool(row.get("score_post_processing_applied", False)),
                    "chosen_flag": bool(row.get("family_chosen_flag", False)),
                    "runner_up_flag": bool(row.get("family_runner_up_flag", False)),
                }
            return {
                "family_id": str(score_obj.get("family_id", row.get("family_id", "")) or ""),
                "prior_component": float(score_obj.get("prior_component", 0.0) or 0.0),
                "trusted_context_component": float(score_obj.get("trusted_context_component", 0.0) or 0.0),
                "evidence_adjustment": float(score_obj.get("evidence_adjustment", 0.0) or 0.0),
                "adaptive_component": float(score_obj.get("adaptive_component", 0.0) or 0.0),
                "competition_diversity_adjustment": float(score_obj.get("competition_diversity_adjustment", 0.0) or 0.0),
                "family_compatibility_component": float(score_obj.get("family_compatibility_component", 0.0) or 0.0),
                "pre_adjustment_score": float(score_obj.get("pre_adjustment_score", 0.0) or 0.0),
                "final_family_score": float(score_obj.get("final_family_score", 0.0) or 0.0),
                "context_trusted": bool(score_obj.get("context_trusted_flag", score_obj.get("context_trusted", False))),
                "support_tier": str(score_obj.get("support_tier", "low") or "low"),
                "exploration_bonus_applied": bool(score_obj.get("exploration_bonus_applied", False)),
                "dominance_penalty_applied": bool(score_obj.get("dominance_penalty_applied", False)),
                "context_advantage_capped": bool(score_obj.get("context_advantage_capped", False)),
                "compatibility_tier": str(score_obj.get("compatibility_tier", "incompatible") or "incompatible"),
                "session_compatibility_tier": str(
                    score_obj.get("session_compatibility_tier", "incompatible") or "incompatible"
                ),
                "timeframe_compatibility_tier": str(
                    score_obj.get("timeframe_compatibility_tier", "incompatible") or "incompatible"
                ),
                "strategy_type_compatibility_tier": str(
                    score_obj.get("strategy_type_compatibility_tier", "incompatible") or "incompatible"
                ),
                "entered_via_compatible_band": bool(score_obj.get("entered_via_compatible_band", False)),
                "any_post_processing_applied": bool(score_obj.get("any_post_processing_applied", False)),
                "chosen_flag": bool(score_obj.get("chosen_flag", False)),
                "runner_up_flag": bool(score_obj.get("runner_up_flag", False)),
            }

        score_decomp_by_family: Dict[str, Dict[str, Any]] = {}
        for row in family_rows:
            if not isinstance(row, dict):
                continue
            family_id = str(row.get("family_id", "") or "")
            if family_id and family_id not in score_decomp_by_family:
                score_decomp_by_family[family_id] = _score_decomp(row)

        def _row_score(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            if not isinstance(row, dict):
                return _score_decomp(None)
            family_id = str(row.get("family_id", "") or "")
            if family_id and family_id in score_decomp_by_family:
                return score_decomp_by_family[family_id]
            score = _score_decomp(row)
            if family_id:
                score_decomp_by_family[family_id] = score
            return score

        competitive_rows.sort(
            key=lambda row: float(_row_score(row).get("final_family_score", float("-inf"))),
            reverse=True,
        )
        chosen_row = None
        if str(chosen_family_id or "").strip():
            for row in competitive_rows:
                if str(row.get("family_id", "") or "") == str(chosen_family_id or ""):
                    chosen_row = row
                    break
        if chosen_row is None:
            chosen_row = competitive_rows[0] if competitive_rows else None
        runner_up_row = None
        if isinstance(chosen_row, dict):
            for row in competitive_rows:
                if str(row.get("family_id", "") or "") != str(chosen_row.get("family_id", "") or ""):
                    runner_up_row = row
                    break

        chosen_decomp = _row_score(chosen_row)
        runner_up_decomp = _row_score(runner_up_row)
        for rank, row in enumerate(family_rows, start=1):
            preview_entry = row.get("preview_entry") if isinstance(row.get("preview_entry"), dict) else None
            is_chosen_family = str(row.get("family_id", "")) == str(chosen_family_id or "")
            is_runner_up_family = bool(
                isinstance(runner_up_row, dict)
                and str(row.get("family_id", "")) == str(runner_up_row.get("family_id", ""))
            )
            row_entry = chosen_entry if (is_chosen_family and isinstance(chosen_entry, dict)) else preview_entry
            cand = row_entry.get("cand") if isinstance(row_entry, dict) and isinstance(row_entry.get("cand"), dict) else {}
            cand_id = (
                str(row_entry.get("cand_id", "") or "")
                if isinstance(row_entry, dict)
                else str(cand.get("strategy_id", "") or "")
            )
            components = row.get("family_score_components") if isinstance(row.get("family_score_components"), dict) else {}
            inventory_row = row.get("inventory_row") if isinstance(row.get("inventory_row"), dict) else {}
            canonical = (
                inventory_row.get("canonical_representative_member")
                if isinstance(inventory_row.get("canonical_representative_member"), dict)
                else {}
            )
            journal_rows.append(
                {
                    "cand": cand,
                    "cand_id": cand_id,
                    "edge_points": row_entry.get("edge_points") if isinstance(row_entry, dict) else None,
                    "structural_score": row_entry.get("structural_score") if isinstance(row_entry, dict) else None,
                    "family_rank": int(rank),
                    "family_id": str(row.get("family_id", "")),
                    "chosen_family_id": str(chosen_family_id or ""),
                    "family_score": row.get("family_score"),
                    "family_context_ev": components.get("context_profile_expectancy_component", components.get("context_ev_component")),
                    "family_confidence": components.get("context_profile_confidence_component", components.get("confidence_component")),
                    "family_prior": components.get("family_prior_component"),
                    "family_profile": components.get("context_profile_expectancy_component", components.get("context_profile_component")),
                    "family_member_count": row.get("inventory_member_count"),
                    "feasible_family_count": int(len(family_rows)),
                    "feasible_family_ids": feasible_ids_joined,
                    "family_context_inputs": dict(context_inputs or {}),
                    "family_artifact": self._de3_v3_family_status.get("family_artifact_path")
                    if isinstance(self._de3_v3_family_status, dict)
                    else None,
                    "family_role": "chosen" if is_chosen_family else ("runner_up" if is_runner_up_family else "other"),
                    "family_candidate_count": int(len(competitive_rows)),
                    "family_candidate_set_gt_1": bool(len(competitive_rows) > 1),
                    "family_candidate_source": str(
                        row.get(
                            "family_candidate_source",
                            (
                                chosen_decomp.get("family_candidate_source", "")
                                if is_chosen_family
                                else row.get("family_candidate_source", "")
                            ),
                        )
                        or ""
                    ),
                    "candidate_rank_before_adjustments": int(
                        row.get("candidate_rank_before_adjustments", 0) or 0
                    ),
                    "choice_path_mode": str(row.get("choice_path_mode", "score_ranking") or "score_ranking"),
                    "score_path_inconsistency_flag": bool(row.get("score_path_inconsistency_flag", False)),
                    "any_post_processing_adjustment_applied": bool(
                        row.get(
                            "score_post_processing_applied",
                            row.get("context_advantage_capped", False)
                            or abs(float(row.get("diversity_adjustment", 0.0) or 0.0)) > 1e-9,
                        )
                    ),
                    "runner_up_available": bool(isinstance(runner_up_row, dict)),
                    "chosen_vs_runner_up_score_delta": (
                        float(chosen_decomp.get("final_family_score", 0.0) - runner_up_decomp.get("final_family_score", 0.0))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "chosen_prior_component": float(chosen_decomp.get("prior_component", 0.0)),
                    "chosen_trusted_context_component": float(chosen_decomp.get("trusted_context_component", 0.0)),
                    "chosen_evidence_adjustment": float(chosen_decomp.get("evidence_adjustment", 0.0)),
                    "chosen_adaptive_component": float(chosen_decomp.get("adaptive_component", 0.0)),
                    "chosen_competition_diversity_adjustment": float(chosen_decomp.get("competition_diversity_adjustment", 0.0)),
                    "chosen_family_compatibility_component": float(
                        chosen_decomp.get("family_compatibility_component", 0.0)
                    ),
                    "chosen_pre_adjustment_score": float(chosen_decomp.get("pre_adjustment_score", 0.0)),
                    "chosen_final_family_score": float(chosen_decomp.get("final_family_score", 0.0)),
                    "chosen_context_trusted": bool(chosen_decomp.get("context_trusted", False)),
                    "chosen_support_tier": str(chosen_decomp.get("support_tier", "low")),
                    "chosen_compatibility_tier": str(
                        chosen_decomp.get("compatibility_tier", "incompatible")
                    ),
                    "chosen_session_compatibility_tier": str(
                        chosen_decomp.get("session_compatibility_tier", "incompatible")
                    ),
                    "chosen_timeframe_compatibility_tier": str(
                        chosen_decomp.get("timeframe_compatibility_tier", "incompatible")
                    ),
                    "chosen_strategy_type_compatibility_tier": str(
                        chosen_decomp.get("strategy_type_compatibility_tier", "incompatible")
                    ),
                    "chosen_exploration_bonus_applied": bool(chosen_decomp.get("exploration_bonus_applied", False)),
                    "chosen_dominance_penalty_applied": bool(chosen_decomp.get("dominance_penalty_applied", False)),
                    "chosen_context_advantage_capped": bool(chosen_decomp.get("context_advantage_capped", False)),
                    "runner_up_family_id": runner_up_decomp.get("family_id"),
                    "runner_up_prior_component": (
                        float(runner_up_decomp.get("prior_component", 0.0))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_trusted_context_component": (
                        float(runner_up_decomp.get("trusted_context_component", 0.0))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_evidence_adjustment": (
                        float(runner_up_decomp.get("evidence_adjustment", 0.0))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_adaptive_component": (
                        float(runner_up_decomp.get("adaptive_component", 0.0))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_competition_diversity_adjustment": (
                        float(runner_up_decomp.get("competition_diversity_adjustment", 0.0))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_family_compatibility_component": (
                        float(runner_up_decomp.get("family_compatibility_component", 0.0))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_pre_adjustment_score": (
                        float(runner_up_decomp.get("pre_adjustment_score", 0.0))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_final_family_score": (
                        float(runner_up_decomp.get("final_family_score", 0.0))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_context_trusted": (
                        bool(runner_up_decomp.get("context_trusted", False))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_support_tier": (
                        str(runner_up_decomp.get("support_tier", "low"))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_compatibility_tier": (
                        str(runner_up_decomp.get("compatibility_tier", "incompatible"))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_session_compatibility_tier": (
                        str(runner_up_decomp.get("session_compatibility_tier", "incompatible"))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_timeframe_compatibility_tier": (
                        str(runner_up_decomp.get("timeframe_compatibility_tier", "incompatible"))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_strategy_type_compatibility_tier": (
                        str(runner_up_decomp.get("strategy_type_compatibility_tier", "incompatible"))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_exploration_bonus_applied": (
                        bool(runner_up_decomp.get("exploration_bonus_applied", False))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_dominance_penalty_applied": (
                        bool(runner_up_decomp.get("dominance_penalty_applied", False))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "runner_up_context_advantage_capped": (
                        bool(runner_up_decomp.get("context_advantage_capped", False))
                        if isinstance(runner_up_row, dict)
                        else None
                    ),
                    "canonical_member_id": row.get("canonical_member_id", canonical.get("member_id")),
                    "member_local_score": (
                        float(chosen_member_local_score)
                        if is_chosen_family and chosen_member_local_score is not None
                        else (
                            row.get("chosen_member_local_score")
                            if row.get("chosen_member_local_score") is not None
                            else (row_entry.get("de3_member_local_score") if isinstance(row_entry, dict) else None)
                        )
                    ),
                    "family_context_support_ratio": components.get("context_support_ratio"),
                    "family_context_support_tier": components.get("context_support_tier", row.get("context_support_tier")),
                    "family_local_support_tier": row.get("family_local_support_tier"),
                    "family_context_sample_count": components.get("context_sample_count", row.get("context_sample_count")),
                    "family_context_weight": components.get("context_profile_weight", row.get("context_profile_weight")),
                    "family_context_trusted": components.get("trusted_context_used", row.get("profile_trusted")),
                    "family_context_fallback_priors": components.get("fallback_to_priors", row.get("profile_fallback")),
                    "family_active_context_buckets": row.get("active_context_buckets"),
                    "family_profile_used": row.get("profile_used"),
                    "family_profile_fallback": row.get("profile_fallback"),
                    "family_usability_state": row.get("family_usability_state"),
                    "family_usability_component": row.get("family_usability_component"),
                    "family_evidence_support_tier": row.get("family_evidence_support_tier"),
                    "family_competition_status": row.get("family_competition_status"),
                    "family_usability_adjustment": row.get("family_usability_adjustment"),
                    "family_suppression_reason": row.get("family_suppression_reason"),
                    "base_family_score": row.get("base_family_score", components.get("base_family_score")),
                    "diversity_adjustment": row.get("diversity_adjustment", components.get("diversity_adjustment")),
                    "competition_diversity_adjustment": row.get(
                        "competition_diversity_adjustment",
                        components.get("competition_diversity_adjustment", row.get("diversity_adjustment", components.get("diversity_adjustment"))),
                    ),
                    "final_family_score": row.get("family_score", components.get("final_family_score")),
                    "recent_chosen_share": row.get("recent_chosen_share", components.get("recent_chosen_share")),
                    "exploration_bonus": row.get("exploration_bonus", components.get("exploration_bonus")),
                    "dominance_penalty": row.get("dominance_penalty", components.get("dominance_penalty")),
                    "exploration_bonus_applied": row.get("exploration_bonus_applied", components.get("exploration_bonus_applied")),
                    "dominance_penalty_applied": row.get("dominance_penalty_applied", components.get("dominance_penalty_applied")),
                    "competition_margin_qualified": row.get("competition_margin_qualified", components.get("competition_margin_qualified")),
                    "context_advantage_capped": row.get("context_advantage_capped", components.get("context_advantage_capped")),
                    "context_advantage_cap_delta": row.get("context_advantage_cap_delta", components.get("context_advantage_cap_delta")),
                    "close_competition_decision": row.get("close_competition_decision", components.get("close_competition_decision")),
                    "bootstrap_competition_used_decision": row.get("bootstrap_competition_used_decision", components.get("bootstrap_competition_used_decision")),
                    "family_monopoly_active": row.get("family_monopoly_active", components.get("family_monopoly_active")),
                    "family_monopoly_top_share": row.get("family_monopoly_top_share", components.get("family_monopoly_top_share")),
                    "family_monopoly_top_family_id": row.get("family_monopoly_top_family_id", components.get("family_monopoly_top_family_id")),
                    "family_monopoly_unique_count": row.get("family_monopoly_unique_count", components.get("family_monopoly_unique_count")),
                    "monopoly_canonical_force_applied": row.get("monopoly_canonical_force_applied", components.get("monopoly_canonical_force_applied")),
                    "family_chosen_flag": bool(is_chosen_family),
                    "local_bracket_adaptation_mode": row.get("local_bracket_adaptation_mode"),
                    "local_bracket_adaptation_enabled": row.get("local_bracket_adaptation_enabled"),
                    "local_bracket_override_applied": row.get("local_bracket_override_applied"),
                    "local_member_count_within_family": row.get("local_member_count_within_family"),
                    "local_edge_component": row.get("local_edge_component"),
                    "local_structural_component": row.get("local_structural_component"),
                    "local_bracket_suitability_component": row.get("local_bracket_suitability_component"),
                    "local_confidence_component": row.get("local_confidence_component"),
                    "local_payoff_component": row.get("local_payoff_component"),
                    "local_final_member_score": row.get("local_final_member_score"),
                    "canonical_fallback_used": row.get("canonical_fallback_used"),
                    "anchor_selected": row.get("anchor_selected"),
                    "why_non_anchor_beat_anchor": row.get("why_non_anchor_beat_anchor"),
                    "why_anchor_forced": row.get("why_anchor_forced"),
                    "no_local_alternative": row.get("no_local_alternative"),
                    "family_prior_eligible": row.get("prior_eligible"),
                    "family_prior_eligibility_reason": row.get("prior_eligibility_reason"),
                    "family_competition_eligible": row.get("competition_eligible"),
                    "family_competition_eligibility_reason": row.get("competition_eligibility_reason"),
                    "family_retained_runtime": row.get("retained_runtime"),
                    "family_evaluated_for_eligibility": row.get("evaluated_for_eligibility"),
                    "family_coarse_eligible": row.get("coarse_eligible"),
                    "family_eligible_for_candidate_set": row.get("eligible_for_candidate_set", row.get("competition_eligible")),
                    "family_eligibility_failure_reason": row.get("eligibility_failure_reason"),
                    "family_coarse_eligibility_failure_reason": row.get("coarse_eligibility_failure_reason"),
                    "family_excluded_by_session_mismatch": row.get("excluded_by_session_mismatch"),
                    "family_excluded_by_side_mismatch": row.get("excluded_by_side_mismatch"),
                    "family_excluded_by_timeframe_mismatch": row.get("excluded_by_timeframe_mismatch"),
                    "family_excluded_by_strategy_type_mismatch": row.get("excluded_by_strategy_type_mismatch"),
                    "family_excluded_by_context_gate": row.get("excluded_by_context_gate"),
                    "family_excluded_by_adaptive_policy_gate": row.get("excluded_by_adaptive_policy_gate"),
                    "family_excluded_by_no_local_member_available": row.get("excluded_by_no_local_member_available"),
                    "family_excluded_by_temporary_exclusion": row.get("excluded_by_temporary_exclusion"),
                    "family_excluded_by_candidate_cap": row.get("excluded_by_candidate_cap"),
                    "family_compatibility_tier": row.get("compatibility_tier"),
                    "family_session_compatibility_tier": row.get("session_compatibility_tier"),
                    "family_timeframe_compatibility_tier": row.get("timeframe_compatibility_tier"),
                    "family_strategy_type_compatibility_tier": row.get("strategy_type_compatibility_tier"),
                    "family_compatibility_component": row.get("family_compatibility_component"),
                    "family_entered_via_compatible_band": row.get("entered_via_compatible_band"),
                    "family_exact_match_eligible": row.get("exact_match_eligible"),
                    "family_compatible_band_eligible": row.get("compatible_band_eligible"),
                    "family_incompatible_excluded": row.get("incompatible_excluded"),
                    "family_eligibility_tier": row.get("eligibility_tier"),
                    "family_preliminary_family_score": row.get("preliminary_family_score"),
                    "family_preliminary_compatibility_penalty_component": row.get(
                        "preliminary_compatibility_penalty_component"
                    ),
                    "family_entered_pre_cap_pool": row.get("entered_pre_cap_pool"),
                    "family_survived_cap": row.get("survived_cap"),
                    "family_cap_drop_reason": row.get("cap_drop_reason"),
                    "family_cap_tier_slot_used": row.get("cap_tier_slot_used"),
                    "family_final_competition_pool_flag": row.get("final_competition_pool_flag"),
                    "pre_cap_candidate_count": row.get("pre_cap_candidate_count"),
                    "post_cap_candidate_count": row.get("post_cap_candidate_count"),
                    "exact_match_survived_count": row.get("exact_match_survived_count"),
                    "compatible_band_survived_count": row.get("compatible_band_survived_count"),
                    "compatible_band_dropped_by_cap_count": row.get(
                        "compatible_band_dropped_by_cap_count"
                    ),
                    "family_coarse_compatibility_timeframe": row.get("coarse_compatibility_timeframe"),
                    "family_coarse_compatibility_session": row.get("coarse_compatibility_session"),
                    "family_coarse_compatibility_side": row.get("coarse_compatibility_side"),
                    "family_coarse_compatibility_strategy_type": row.get("coarse_compatibility_strategy_type"),
                    "family_coarse_compatibility_threshold": row.get("coarse_compatibility_threshold"),
                    "family_coarse_signature_sessions_seen": row.get("coarse_signature_sessions_seen"),
                    "family_coarse_signature_sides_seen": row.get("coarse_signature_sides_seen"),
                    "family_coarse_signature_timeframes_seen": row.get("coarse_signature_timeframes_seen"),
                    "family_coarse_signature_strategy_types_seen": row.get("coarse_signature_strategy_types_seen"),
                    "family_coarse_signature_decision_session": row.get("coarse_signature_decision_session"),
                    "family_coarse_signature_decision_hour_et": row.get("coarse_signature_decision_hour_et"),
                    "family_member_candidates_seen_count": row.get("member_candidates_seen_count"),
                    "family_member_filtered_out_count": row.get("member_filtered_out_count"),
                    "retained_families_total": row.get("retained_families_total"),
                    "retained_families_scanned": row.get("retained_families_scanned"),
                    "retained_families_eligible": row.get("retained_families_eligible"),
                    "retained_families_excluded": row.get("retained_families_excluded"),
                    "retained_families_unscanned": row.get("retained_families_unscanned"),
                    "retained_family_scan_guarantee_pass": row.get("retained_family_scan_guarantee_pass"),
                    "family_bootstrap_competition_included": row.get("bootstrap_competition_included"),
                    "family_bootstrap_included": row.get("bootstrap_included"),
                    "family_catastrophic_prior": row.get("catastrophic_prior"),
                    # Backward-compatible aliases.
                    "family_eligible": row.get("competition_eligible"),
                    "family_eligibility_reason": row.get("competition_eligibility_reason"),
                }
            )
        return self._emit_de3_decision_journal(
            current_time=current_time,
            session_name=session_name,
            feasible_candidates=journal_rows,
            chosen_cand_id=chosen_cand_id if not abstained else None,
            abstained=bool(abstained),
            abstain_reason=abstain_reason,
        )

    @staticmethod
    def _is_bar_close(ts: pd.Timestamp, minutes: int) -> bool:
        return ts.minute % minutes == minutes - 1 and ts.second == 0

    @staticmethod
    def _signal_bucket(ts, tf: str, strategy_id: str = "") -> pd.Timestamp:
        ts_val = pd.Timestamp(ts)
        tf_norm = str(tf or "").lower()
        sid = str(strategy_id or "").lower()
        if "15" in tf_norm or "_15" in sid:
            return ts_val.floor("15min")
        if "5" in tf_norm or "_5" in sid:
            return ts_val.floor("5min")
        return ts_val.floor("1min")

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
        agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if df is not None and "volume" in df.columns:
            agg["volume"] = "sum"
        out = df.resample(rule, closed="left", label="left").agg(agg)
        out = out.dropna(subset=["open", "high", "low", "close"])
        if "volume" in out.columns:
            out["volume"] = out["volume"].fillna(0.0)
        return out

    @staticmethod
    def _augment_5m_features(df_5m: pd.DataFrame, atr_period: int, atr_median_window: int, price_loc_window: int) -> pd.DataFrame:
        if df_5m is None or df_5m.empty:
            return df_5m
        high = df_5m["high"]
        low = df_5m["low"]
        close = df_5m["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / float(atr_period), adjust=False).mean()
        atr_median = atr.rolling(atr_median_window, min_periods=atr_median_window).median()
        high_roll = high.rolling(price_loc_window, min_periods=price_loc_window).max()
        low_roll = low.rolling(price_loc_window, min_periods=price_loc_window).min()
        denom = (high_roll - low_roll).replace(0, np.nan)
        price_loc = (close - low_roll) / denom
        df_5m = df_5m.copy()
        df_5m["atr_5m"] = atr
        df_5m["atr_5m_median"] = atr_median
        df_5m["price_location"] = price_loc.clip(lower=0.0, upper=1.0)
        return df_5m

    @staticmethod
    def _fast_5m_context_snapshot(
        df_5m: pd.DataFrame,
        *,
        atr_period: int,
        atr_median_window: int,
        price_loc_window: int,
        rvol_window: int = 20,
    ) -> Optional[Dict[str, float]]:
        """Compute only the latest 5m context values from a bounded tail window."""
        if df_5m is None or df_5m.empty:
            return None
        try:
            atr_period = max(2, int(atr_period))
        except Exception:
            atr_period = 20
        try:
            atr_median_window = max(2, int(atr_median_window))
        except Exception:
            atr_median_window = 390
        try:
            price_loc_window = max(2, int(price_loc_window))
        except Exception:
            price_loc_window = 20
        try:
            rvol_window = max(2, int(rvol_window))
        except Exception:
            rvol_window = 20

        need = max(atr_period + 2, price_loc_window + 2, rvol_window + 2)
        if len(df_5m) < need:
            return None
        tail_n = max(need, atr_median_window + atr_period + 8)
        tail_n = min(len(df_5m), max(240, tail_n))
        tail = df_5m.iloc[-tail_n:]

        high = tail["high"]
        low = tail["low"]
        close = tail["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / float(atr_period), adjust=False).mean()
        if atr.empty:
            return None
        try:
            atr_5m = float(atr.iloc[-1])
        except Exception:
            return None
        if not np.isfinite(atr_5m) or atr_5m <= 0:
            return None

        if len(atr) >= atr_median_window:
            atr_med = float(atr.iloc[-atr_median_window:].median())
        else:
            atr_med = float(atr.median())
        if not np.isfinite(atr_med) or atr_med <= 0:
            return None

        loc_tail = tail.iloc[-price_loc_window:]
        recent_high = float(loc_tail["high"].max())
        recent_low = float(loc_tail["low"].min())
        close_last = float(close.iloc[-1])
        denom = recent_high - recent_low
        if np.isfinite(denom) and denom > 0:
            price_loc = float((close_last - recent_low) / denom)
            price_loc = float(max(0.0, min(1.0, price_loc)))
        else:
            price_loc = 0.5

        rvol_ratio = None
        if "volume" in tail.columns and len(tail) >= rvol_window:
            try:
                vol_now = float(tail["volume"].iloc[-1])
                vol_base = float(tail["volume"].iloc[-rvol_window:].mean())
                if np.isfinite(vol_now) and np.isfinite(vol_base) and vol_base > 0:
                    rvol_ratio = float(vol_now / vol_base)
            except Exception:
                rvol_ratio = None

        return {
            "atr_5m": float(atr_5m),
            "atr_5m_median": float(atr_med),
            "price_location": float(price_loc),
            "rvol_ratio": rvol_ratio,
        }

    @staticmethod
    def _compute_atr_simple(df: pd.DataFrame, period: int) -> Optional[float]:
        if df is None or df.empty:
            return None
        try:
            period = int(period)
        except Exception:
            period = 14
        if period <= 0 or len(df) < period:
            return None
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        try:
            value = float(atr.iloc[-1])
        except Exception:
            return None
        if not np.isfinite(value) or value <= 0:
            return None
        return value

    @staticmethod
    def _compute_entry_structure_features(
        df_1m: pd.DataFrame,
        atr_period: int = 14,
        flip_window: int = 5,
        range_window: int = 10,
        dist_window: int = 5,
        min_bars: int = 40,
    ) -> Optional[Dict[str, float]]:
        """
        Compute short-horizon candle-structure features from 1m bars.
        Uses only bars strictly before the current signal bar to avoid lookahead.
        """
        if df_1m is None or df_1m.empty:
            return None
        required = max(int(min_bars), int(atr_period) + 2, int(range_window) + 1, int(flip_window) + 2)
        if len(df_1m) < required + 1:
            return None
        pre = df_1m.iloc[:-1]
        if pre is None or pre.empty or len(pre) < required:
            return None
        for col in ("open", "high", "low", "close"):
            if col not in pre.columns:
                return None

        atr_val = DynamicEngine3Strategy._compute_atr_simple(pre, int(atr_period))
        if atr_val is None or not np.isfinite(atr_val) or atr_val <= 0:
            return None

        last = pre.iloc[-1]
        try:
            bar_open = float(last["open"])
            bar_high = float(last["high"])
            bar_low = float(last["low"])
            bar_close = float(last["close"])
        except Exception:
            return None
        bar_range = max(1e-9, bar_high - bar_low)
        body = float(bar_close - bar_open)
        body1_ratio = float(abs(body) / bar_range)
        close_pos1 = float((bar_close - bar_low) / bar_range)
        upper_wick_ratio = float((bar_high - max(bar_open, bar_close)) / bar_range)
        lower_wick_ratio = float((min(bar_open, bar_close) - bar_low) / bar_range)
        ret1_atr = float(body / atr_val)

        closes = pre["close"].iloc[-(int(flip_window) + 1) :]
        flips = 0
        if len(closes) >= 3:
            diffs = np.diff(closes.to_numpy(dtype=float))
            signs = np.sign(diffs)
            flips = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) >= 2 else 0

        closes4 = pre["close"].iloc[-4:]
        down3 = 0
        if len(closes4) >= 4:
            down3 = int(np.sum(np.diff(closes4.to_numpy(dtype=float)) < 0))

        high_n = pre["high"].iloc[-int(range_window) :].max()
        low_n = pre["low"].iloc[-int(range_window) :].min()
        range_n_atr = float((float(high_n) - float(low_n)) / atr_val)

        low_d = pre["low"].iloc[-int(dist_window) :].min()
        high_d = pre["high"].iloc[-int(dist_window) :].max()
        dist_low_atr = float((bar_close - float(low_d)) / atr_val)
        dist_high_atr = float((float(high_d) - bar_close) / atr_val)

        vol1_rel20 = np.nan
        if "volume" in pre.columns and len(pre) >= 20:
            try:
                v1 = float(pre["volume"].iloc[-1])
                v20 = float(pre["volume"].iloc[-20:].mean())
                if np.isfinite(v20) and v20 > 0:
                    vol1_rel20 = float(v1 / v20)
            except Exception:
                vol1_rel20 = np.nan

        return {
            "ret1_atr": float(ret1_atr),
            "body_pos1": float(body / bar_range),
            "lower_wick_ratio": float(lower_wick_ratio),
            "upper_wick_ratio": float(upper_wick_ratio),
            "upper1_ratio": float(upper_wick_ratio),
            "body1_ratio": float(body1_ratio),
            "close_pos1": float(close_pos1),
            "flips5": float(flips),
            "down3": float(down3),
            "range10_atr": float(range_n_atr),
            "dist_low5_atr": float(dist_low_atr),
            "dist_high5_atr": float(dist_high_atr),
            "vol1_rel20": float(vol1_rel20) if np.isfinite(vol1_rel20) else np.nan,
            "atr14": float(atr_val),
        }

    @staticmethod
    def _compute_entry_bar_shape_features(
        df_1m: pd.DataFrame,
        min_bars: int = 2,
    ) -> Optional[Dict[str, float]]:
        """
        Lightweight 1m entry-bar shape features for DE3 v2 runtime hard blocks.
        Uses the latest fully closed bar only (df_1m.iloc[-2] relative to current in-progress row).
        """
        if df_1m is None or df_1m.empty:
            return None
        try:
            need = max(2, int(min_bars))
        except Exception:
            need = 2
        if len(df_1m) < need:
            return None
        pre = df_1m.iloc[:-1]
        if pre is None or pre.empty:
            return None
        last = pre.iloc[-1]
        try:
            bar_open = float(last["open"])
            bar_high = float(last["high"])
            bar_low = float(last["low"])
            bar_close = float(last["close"])
        except Exception:
            return None
        bar_range = float(bar_high - bar_low)
        if not np.isfinite(bar_range) or bar_range <= 0:
            return None
        body_pos1 = float((bar_close - bar_open) / bar_range)
        close_pos1 = float((bar_close - bar_low) / bar_range)
        lower_wick_ratio = float((min(bar_open, bar_close) - bar_low) / bar_range)
        return {
            "body_pos1": body_pos1,
            "close_pos1": close_pos1,
            "lower_wick_ratio": lower_wick_ratio,
        }

    @staticmethod
    def _de3_v2_entry_bar_block_reason(
        features: Optional[Dict[str, float]],
        cfg: Dict,
        side: str,
        timeframe: Optional[str] = None,
    ) -> Optional[str]:
        if not cfg or not bool(cfg.get("enabled", False)):
            return None
        if not isinstance(features, dict):
            return None

        sides = {str(x).upper() for x in (cfg.get("apply_sides", ["LONG"]) or ["LONG"])}
        if sides and str(side or "").upper() not in sides:
            return None
        tf_allowed = {str(x).lower() for x in (cfg.get("apply_timeframes", ["5min", "15min"]) or ["5min", "15min"])}
        tf_norm = str(timeframe or "").strip().lower()
        if tf_allowed and tf_norm and tf_norm not in tf_allowed:
            return None

        body_pos = float(features.get("body_pos1", np.nan))
        close_pos = float(features.get("close_pos1", np.nan))
        lower_wick_ratio = float(features.get("lower_wick_ratio", np.nan))
        rules = cfg.get("rules", {}) or {}

        def _rule_cfg(name: str, defaults: Dict[str, float]) -> Dict:
            raw = rules.get(name, {}) or {}
            out = dict(defaults)
            out.update(raw)
            return out

        r_long_lower = _rule_cfg("long_lower_wick", {"enabled": True, "lower_wick_min": 0.47})
        if bool(r_long_lower.get("enabled", True)) and np.isfinite(lower_wick_ratio):
            try:
                lower_min = float(r_long_lower.get("lower_wick_min", 0.47) or 0.47)
            except Exception:
                lower_min = 0.47
            if lower_wick_ratio >= lower_min:
                return f"V2EntryBar long_lower_wick lower_wick={lower_wick_ratio:.3f} >= {lower_min:.3f}"

        r_strong_bear = _rule_cfg("body_strong_bear", {"enabled": True, "body_pos_max": -0.40})
        if bool(r_strong_bear.get("enabled", True)) and np.isfinite(body_pos):
            try:
                body_max = float(r_strong_bear.get("body_pos_max", -0.40) or -0.40)
            except Exception:
                body_max = -0.40
            if body_pos <= body_max:
                return f"V2EntryBar strong_bear body_pos={body_pos:.3f} <= {body_max:.3f}"

        r_bear_nowick = _rule_cfg(
            "bear_no_lower_wick",
            {"enabled": True, "body_pos_max": -0.30, "lower_wick_max": 0.10},
        )
        if bool(r_bear_nowick.get("enabled", True)) and np.isfinite(body_pos) and np.isfinite(lower_wick_ratio):
            try:
                body_max = float(r_bear_nowick.get("body_pos_max", -0.30) or -0.30)
            except Exception:
                body_max = -0.30
            try:
                wick_max = float(r_bear_nowick.get("lower_wick_max", 0.10) or 0.10)
            except Exception:
                wick_max = 0.10
            if body_pos <= body_max and lower_wick_ratio <= wick_max:
                return (
                    f"V2EntryBar bear_no_lower_wick body_pos={body_pos:.3f}<= {body_max:.3f}, "
                    f"lower_wick={lower_wick_ratio:.3f}<= {wick_max:.3f}"
                )

        r_close_low = _rule_cfg("close_near_low", {"enabled": True, "close_pos_max": 0.20})
        if bool(r_close_low.get("enabled", True)) and np.isfinite(close_pos):
            try:
                close_max = float(r_close_low.get("close_pos_max", 0.20) or 0.20)
            except Exception:
                close_max = 0.20
            if close_pos <= close_max:
                return f"V2EntryBar close_near_low close_pos={close_pos:.3f} <= {close_max:.3f}"

        r_body_non_pos = _rule_cfg("body_non_positive", {"enabled": True, "body_pos_max": 0.0})
        if bool(r_body_non_pos.get("enabled", True)) and np.isfinite(body_pos):
            try:
                body_max = float(r_body_non_pos.get("body_pos_max", 0.0) or 0.0)
            except Exception:
                body_max = 0.0
            if body_pos <= body_max:
                return f"V2EntryBar body_non_positive body_pos={body_pos:.3f} <= {body_max:.3f}"
        return None

    @staticmethod
    def _entry_candle_block_reason(
        features: Optional[Dict[str, float]],
        cfg: Dict,
        side: str,
        timeframe: Optional[str] = None,
    ) -> Optional[str]:
        if not cfg or not bool(cfg.get("enabled", False)):
            return None
        if not isinstance(features, dict):
            return None
        sides = {str(x).upper() for x in (cfg.get("apply_sides", ["SHORT"]) or ["SHORT"])}
        if str(side or "").upper() not in sides:
            return None

        ret1_atr = float(features.get("ret1_atr", np.nan))
        range10_atr = float(features.get("range10_atr", np.nan))
        vol1_rel20 = float(features.get("vol1_rel20", np.nan))
        body1_ratio = float(features.get("body1_ratio", np.nan))
        close_pos1 = float(features.get("close_pos1", np.nan))
        upper1_ratio = float(
            features.get("upper1_ratio", features.get("upper_wick_ratio", np.nan))
        )
        dist_high5_atr = float(features.get("dist_high5_atr", np.nan))

        # N1: weak prior impulse in stretched + active context.
        n1 = cfg.get("n1", {}) or {}
        # N2: weak prior impulse with poor candle shape.
        n2 = cfg.get("n2", {}) or {}
        # N3: 5m structure trap (near local high with visible upper wick).
        n3 = cfg.get("n3", {}) or {}
        try:
            n1_ret1_atr_max = float(n1.get("ret1_atr_max", 0.017) or 0.017)
        except Exception:
            n1_ret1_atr_max = 0.017
        try:
            n1_range10_atr_min = float(n1.get("range10_atr_min", 2.612) or 2.612)
        except Exception:
            n1_range10_atr_min = 2.612
        try:
            n1_vol1_rel20_min = float(n1.get("vol1_rel20_min", 0.014) or 0.014)
        except Exception:
            n1_vol1_rel20_min = 0.014
        try:
            n2_ret1_atr_max = float(n2.get("ret1_atr_max", 0.017) or 0.017)
        except Exception:
            n2_ret1_atr_max = 0.017
        try:
            n2_body1_ratio_min = float(n2.get("body1_ratio_min", 0.55) or 0.55)
        except Exception:
            n2_body1_ratio_min = 0.55
        try:
            n2_close_pos1_max = float(n2.get("close_pos1_max", 0.35) or 0.35)
        except Exception:
            n2_close_pos1_max = 0.35
        try:
            n3_enabled = bool(n3.get("enabled", False))
        except Exception:
            n3_enabled = False
        try:
            n3_dist_high5_atr_max = float(n3.get("dist_high5_atr_max", 1.53343) or 1.53343)
        except Exception:
            n3_dist_high5_atr_max = 1.53343
        try:
            n3_upper1_ratio_min = float(n3.get("upper1_ratio_min", 0.0125) or 0.0125)
        except Exception:
            n3_upper1_ratio_min = 0.0125
        n3_timeframes = {str(x).lower() for x in (n3.get("timeframes", ["5min"]) or ["5min"])}

        hit_n1 = (
            np.isfinite(ret1_atr)
            and np.isfinite(range10_atr)
            and np.isfinite(vol1_rel20)
            and ret1_atr <= n1_ret1_atr_max
            and range10_atr >= n1_range10_atr_min
            and vol1_rel20 >= n1_vol1_rel20_min
        )
        hit_n2 = (
            np.isfinite(ret1_atr)
            and np.isfinite(body1_ratio)
            and np.isfinite(close_pos1)
            and ret1_atr <= n2_ret1_atr_max
            and body1_ratio >= n2_body1_ratio_min
            and close_pos1 <= n2_close_pos1_max
        )
        tf_norm = str(timeframe or "").strip().lower()
        hit_n3 = (
            n3_enabled
            and tf_norm in n3_timeframes
            and np.isfinite(dist_high5_atr)
            and np.isfinite(upper1_ratio)
            and dist_high5_atr <= n3_dist_high5_atr_max
            and upper1_ratio >= n3_upper1_ratio_min
        )

        if not (hit_n1 or hit_n2 or hit_n3):
            return None
        if hit_n1 and hit_n2 and hit_n3:
            return (
                f"EntryStruct N1+N2+N3 ret1_atr={ret1_atr:.3f} range10_atr={range10_atr:.2f} "
                f"vol1_rel20={vol1_rel20:.2f} body1_ratio={body1_ratio:.2f} close_pos1={close_pos1:.2f} "
                f"dist_high5_atr={dist_high5_atr:.2f} upper1_ratio={upper1_ratio:.3f}"
            )
        if hit_n1 and hit_n2:
            return (
                f"EntryStruct N1+N2 ret1_atr={ret1_atr:.3f} range10_atr={range10_atr:.2f} "
                f"vol1_rel20={vol1_rel20:.2f} body1_ratio={body1_ratio:.2f} close_pos1={close_pos1:.2f}"
            )
        if hit_n1 and hit_n3:
            return (
                f"EntryStruct N1+N3 ret1_atr={ret1_atr:.3f} range10_atr={range10_atr:.2f} "
                f"vol1_rel20={vol1_rel20:.2f} dist_high5_atr={dist_high5_atr:.2f} upper1_ratio={upper1_ratio:.3f}"
            )
        if hit_n2 and hit_n3:
            return (
                f"EntryStruct N2+N3 ret1_atr={ret1_atr:.3f} body1_ratio={body1_ratio:.2f} "
                f"close_pos1={close_pos1:.2f} dist_high5_atr={dist_high5_atr:.2f} upper1_ratio={upper1_ratio:.3f}"
            )
        if hit_n1:
            return (
                f"EntryStruct N1 ret1_atr={ret1_atr:.3f} range10_atr={range10_atr:.2f} "
                f"vol1_rel20={vol1_rel20:.2f}"
            )
        if hit_n3:
            return (
                f"EntryStruct N3 dist_high5_atr={dist_high5_atr:.2f} "
                f"upper1_ratio={upper1_ratio:.3f} tf={tf_norm or 'na'}"
            )
        return (
            f"EntryStruct N2 ret1_atr={ret1_atr:.3f} body1_ratio={body1_ratio:.2f} "
            f"close_pos1={close_pos1:.2f}"
        )

    @staticmethod
    def _check_long_mom_filters(
        df_tf: pd.DataFrame,
        cfg: Dict,
        timeframe: str,
        cand_thresh: Optional[float] = None,
        regime_norm: str = "",
    ) -> Tuple[bool, Optional[str]]:
        if df_tf is None or df_tf.empty:
            return True, None

        atr_period = int(cfg.get("atr_period", 14) or 14)
        compression_lookback = int(cfg.get("compression_lookback", 4) or 4)
        compression_atr_mult = float(cfg.get("compression_atr_mult", 0.7) or 0.7)
        late_breakout_atr_mult = float(cfg.get("late_breakout_atr_mult", 1.0) or 1.0)
        impulse_body_atr_min = float(cfg.get("impulse_body_atr_min", 0.6) or 0.6)
        impulse_close_pos_min = float(cfg.get("impulse_close_pos_min", 0.8) or 0.8)
        require_bull_close = bool(cfg.get("require_bull_close", True))
        shock_gap_atr_mult = float(cfg.get("shock_gap_atr_mult", 0.7) or 0.7)
        shock_range_atr_mult = float(cfg.get("shock_range_atr_mult", 2.0) or 2.0)
        block_thresh_high_vol = cfg.get("block_thresh_values_in_high_vol", [2.0]) or []

        atr_val = None
        if str(timeframe).startswith("5") and "atr_5m" in df_tf.columns:
            try:
                atr_val = float(df_tf["atr_5m"].iloc[-1])
            except Exception:
                atr_val = None
        if atr_val is None or not np.isfinite(atr_val) or atr_val <= 0:
            atr_val = DynamicEngine3Strategy._compute_atr_simple(df_tf, atr_period)
        if atr_val is None:
            return True, None

        pre = None
        if compression_lookback > 0 and len(df_tf) >= compression_lookback + 1:
            pre = df_tf.iloc[-(compression_lookback + 1):-1]
            if not pre.empty:
                rng = float(pre["high"].max() - pre["low"].min())
                if np.isfinite(rng) and rng > (compression_atr_mult * atr_val):
                    return False, f"Long_Mom freshness: range {rng:.2f} > {compression_atr_mult:.2f}xATR"

        last = df_tf.iloc[-1]
        try:
            bar_high = float(last["high"])
            bar_low = float(last["low"])
            bar_open = float(last["open"])
            bar_close = float(last["close"])
        except Exception:
            return True, None

        bar_range = bar_high - bar_low
        if bar_range <= 0:
            return False, "Long_Mom impulse: zero range"

        if require_bull_close and bar_close <= bar_open:
            return False, "Long_Mom impulse: not bullish close"

        body = abs(bar_close - bar_open)
        body_atr = body / atr_val if atr_val > 0 else 0.0
        close_pos = (bar_close - bar_low) / bar_range

        if body_atr < impulse_body_atr_min:
            return False, f"Long_Mom impulse: body/ATR {body_atr:.2f} < {impulse_body_atr_min:.2f}"
        if close_pos < impulse_close_pos_min:
            return False, f"Long_Mom impulse: close_pos {close_pos:.2f} < {impulse_close_pos_min:.2f}"

        # High-vol threshold toxicity guard: some threshold bands are persistent loss drivers.
        if regime_norm == "high" and cand_thresh is not None:
            blocked = {round(float(x), 4) for x in block_thresh_high_vol}
            if round(float(cand_thresh), 4) in blocked:
                return False, f"Long_Mom high-vol thresh blocked: {cand_thresh:g}"

        # Shock-gap guard: skip chasing momentum entries after a discontinuous bar open.
        if len(df_tf) >= 2 and shock_gap_atr_mult > 0:
            try:
                prev_close = float(df_tf["close"].iloc[-2])
                gap_atr = abs(bar_open - prev_close) / atr_val
            except Exception:
                gap_atr = 0.0
            if gap_atr > shock_gap_atr_mult:
                return False, f"Long_Mom shock gap: gap/ATR {gap_atr:.2f} > {shock_gap_atr_mult:.2f}"

        # Shock-range guard: skip oversized expansion bars that are more likely exhaustion.
        range_atr = bar_range / atr_val if atr_val > 0 else 0.0
        if shock_range_atr_mult > 0 and range_atr > shock_range_atr_mult:
            return False, f"Long_Mom shock range: range/ATR {range_atr:.2f} > {shock_range_atr_mult:.2f}"

        # Late breakout check: if current close is already too far beyond the prior range,
        # it's likely an exhausted move rather than fresh acceptance.
        if pre is not None and not pre.empty and late_breakout_atr_mult > 0:
            try:
                pre_high = float(pre["high"].max())
            except Exception:
                pre_high = None
            if pre_high is not None and np.isfinite(pre_high):
                dist = bar_close - pre_high
                if dist > (late_breakout_atr_mult * atr_val):
                    return False, f"Long_Mom late breakout: {dist:.2f} > {late_breakout_atr_mult:.2f}xATR"

        return True, None

    @staticmethod
    def _compute_vwap_value(df_1m: pd.DataFrame, ts: pd.Timestamp) -> Optional[float]:
        if df_1m is None or df_1m.empty or ts is None:
            return None
        idx = df_1m.index
        try:
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            else:
                ts = ts.tz_convert("America/New_York")
        except Exception:
            pass
        if isinstance(idx, pd.DatetimeIndex):
            try:
                if idx.tz is None:
                    idx = idx.tz_localize(ts.tzinfo)
                else:
                    idx = idx.tz_convert(ts.tzinfo)
            except Exception:
                idx = df_1m.index
        day_start = pd.Timestamp(ts.date(), tz=ts.tzinfo)
        day_end = day_start + pd.Timedelta(days=1)
        if not isinstance(idx, pd.DatetimeIndex):
            return None
        try:
            start_pos = int(idx.searchsorted(day_start, side="left"))
            end_pos = int(idx.searchsorted(day_end, side="left"))
        except Exception:
            return None
        if end_pos <= start_pos:
            return None
        df_day = df_1m.iloc[start_pos:end_pos]
        if df_day.empty:
            return None
        typical = (df_day["high"] + df_day["low"] + df_day["close"]) / 3.0
        volume = df_day["volume"] if "volume" in df_day.columns else pd.Series(1.0, index=df_day.index)
        volume = volume.fillna(0.0)
        cum_pv = (typical * volume).cumsum()
        cum_v = volume.cumsum()
        vwap = cum_pv / cum_v.replace(0, np.nan)
        try:
            return float(vwap.iloc[-1])
        except Exception:
            return None

    @staticmethod
    def _get_safety_guard_reason(
        cand_tf: str,
        cand_type: str,
        cand_thresh: Optional[float],
        regime_norm: str,
        safety_cfg: Optional[dict] = None,
    ) -> Optional[str]:
        cfg = safety_cfg or {}
        # Configurable high-vol reversal clamp.
        if bool(cfg.get("block_high_vol_reversals", False)) and regime_norm == "high":
            long_rev_tfs = {str(x).lower() for x in (cfg.get("block_long_rev_timeframes") or ["5min", "15min"])}
            short_rev_tfs = {str(x).lower() for x in (cfg.get("block_short_rev_timeframes") or ["5min", "15min"])}
            if cand_type == "long_rev" and cand_tf in long_rev_tfs:
                return f"blocked {cand_tf} Long_Rev in high vol"
            if cand_type == "short_rev" and cand_tf in short_rev_tfs:
                return f"blocked {cand_tf} Short_Rev in high vol"

        # 1) 15m Long_Mom is consistently toxic -> disable entirely.
        if cand_tf == "15min" and cand_type == "long_mom":
            return "blocked 15m Long_Mom"
        # 2) 5m Long_Mom performs only in high-vol -> allow only when vol_regime=high.
        if cand_tf == "5min" and cand_type == "long_mom" and regime_norm != "high":
            return "blocked 5m Long_Mom outside high vol"
        # 2b) Thresh=2 momentum signals are a major drawdown driver.
        if cand_type == "long_mom" and cand_thresh is not None and abs(cand_thresh - 2.0) < 1e-6:
            return "blocked Long_Mom thresh=2"
        # 2c) Thresh=15 short momentum is also consistently toxic.
        if cand_type == "short_mom" and cand_thresh is not None and abs(cand_thresh - 15.0) < 1e-6:
            return "blocked Short_Mom thresh=15"
        # 3) 15m Short_Rev underperforms in high-vol.
        if cand_tf == "15min" and cand_type == "short_rev" and regime_norm == "high":
            return "blocked 15m Short_Rev in high vol"
        return None

    @staticmethod
    def _veto_thresh_bucket(cand_thresh: Optional[float]) -> str:
        if cand_thresh is None:
            return "ALL"
        try:
            value = float(cand_thresh)
        except Exception:
            return "ALL"
        if not np.isfinite(value):
            return "ALL"
        return f"T{int(round(value))}"

    @classmethod
    def _build_veto_bucket_keys(
        cls,
        engine_session: Optional[str],
        cand_tf_raw: str,
        cand_type_raw: str,
        regime_norm: str,
        cand_thresh: Optional[float],
    ) -> List[str]:
        sess = str(engine_session or "UNKNOWN")
        tf = str(cand_tf_raw or "UNKNOWN")
        stype = str(cand_type_raw or "UNKNOWN")
        regime = str(regime_norm or "unknown")
        thresh_bucket = cls._veto_thresh_bucket(cand_thresh)
        return [
            f"{sess}|{tf}|{stype}|{regime}|{thresh_bucket}",
            f"{sess}|{tf}|{stype}|{regime}|ALL",
            f"{sess}|{tf}|{stype}|ALL|{thresh_bucket}",
            f"{sess}|{tf}|{stype}|ALL|ALL",
            f"{sess}|{tf}|{stype}",
        ]

    @staticmethod
    def _clip01(value: float) -> float:
        try:
            return float(max(0.0, min(1.0, value)))
        except Exception:
            return 0.0

    @staticmethod
    def _clip_range(value: float, lo: float, hi: float) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(lo)
        return float(max(float(lo), min(float(hi), out)))

    @staticmethod
    def _de3_v2_same_family_key(
        cand: Dict,
        default_session: Optional[str] = None,
    ) -> Tuple[str, str, str, str, str, str]:
        if not isinstance(cand, dict):
            cand = {}

        timeframe = str(cand.get("timeframe", "") or "").strip().lower()
        session = str(cand.get("session", default_session or "") or "").strip().upper()
        side = str(cand.get("signal", cand.get("side", "")) or "").strip().upper()
        strategy_type = str(cand.get("strategy_type", "") or "").strip().lower()

        try:
            sl = float(cand.get("sl", np.nan))
            sl_key = f"{sl:.8f}" if np.isfinite(sl) else ""
        except Exception:
            sl_key = ""
        try:
            tp = float(cand.get("tp", np.nan))
            tp_key = f"{tp:.8f}" if np.isfinite(tp) else ""
        except Exception:
            tp_key = ""

        return (timeframe, session, side, strategy_type, sl_key, tp_key)

    def _de3_v2_apply_same_family_near_tie_override(
        self,
        feasible_candidates: List[Dict[str, object]],
        *,
        engine_session: Optional[str],
        current_time: pd.Timestamp,
    ) -> Dict[str, object]:
        if len(feasible_candidates) <= 1:
            return feasible_candidates[0]

        chosen_entry = feasible_candidates[0]
        chosen_cand = chosen_entry.get("cand") or {}
        chosen_family = chosen_entry.get("family_key")
        if not chosen_family:
            chosen_family = self._de3_v2_same_family_key(chosen_cand, default_session=engine_session)

        try:
            chosen_edge = float(chosen_entry.get("edge_points", 0.0) or 0.0)
        except Exception:
            chosen_edge = 0.0
        try:
            chosen_struct = float(chosen_entry.get("structural_score", 0.0) or 0.0)
        except Exception:
            chosen_struct = 0.0

        best_alt = None
        best_alt_struct = chosen_struct
        best_alt_rank = float("-inf")
        for alt in feasible_candidates[1:]:
            alt_cand = alt.get("cand") or {}
            alt_family = alt.get("family_key")
            if not alt_family:
                alt_family = self._de3_v2_same_family_key(alt_cand, default_session=engine_session)
            if alt_family != chosen_family:
                continue

            try:
                alt_edge = float(alt.get("edge_points", 0.0) or 0.0)
            except Exception:
                alt_edge = 0.0
            if abs(alt_edge - chosen_edge) > 0.15:
                continue

            try:
                alt_struct = float(alt.get("structural_score", 0.0) or 0.0)
            except Exception:
                alt_struct = 0.0
            if alt_struct <= (chosen_struct + 0.25):
                continue

            try:
                alt_rank = float(alt.get("runtime_rank_score", float("-inf")) or float("-inf"))
            except Exception:
                alt_rank = float("-inf")

            if best_alt is None or alt_struct > best_alt_struct or (
                alt_struct == best_alt_struct and alt_rank > best_alt_rank
            ):
                best_alt = alt
                best_alt_struct = alt_struct
                best_alt_rank = alt_rank

        if best_alt is None:
            return chosen_entry

        try:
            best_alt_edge = float(best_alt.get("edge_points", 0.0) or 0.0)
        except Exception:
            best_alt_edge = 0.0
        try:
            best_alt_struct = float(best_alt.get("structural_score", 0.0) or 0.0)
        except Exception:
            best_alt_struct = 0.0
        logging.info(
            (
                "DE3 v2 same-family near-tie structural override ts=%s chosen=%s overridden=%s "
                "edge_chosen=%.3f edge_override=%.3f struct_chosen=%.3f struct_override=%.3f "
                "reason=same_family_near_tie_structural_override"
            ),
            pd.Timestamp(current_time).isoformat(),
            str(chosen_entry.get("cand_id", "")),
            str(best_alt.get("cand_id", "")),
            float(chosen_edge),
            float(best_alt_edge),
            float(chosen_struct),
            float(best_alt_struct),
        )
        return best_alt

    @classmethod
    def _compute_policy_risk_mult(cls, policy_eval: dict, policy_cfg: Optional[dict]) -> float:
        cfg = policy_cfg or {}
        risk_cfg = cfg.get("risk", {}) or {}
        try:
            min_mult = float(risk_cfg.get("min_mult", 0.60))
        except Exception:
            min_mult = 0.60
        try:
            max_mult = float(risk_cfg.get("max_mult", 1.40))
        except Exception:
            max_mult = 1.40
        if not np.isfinite(min_mult):
            min_mult = 0.60
        if not np.isfinite(max_mult):
            max_mult = 1.40
        if max_mult < min_mult:
            max_mult = min_mult

        try:
            confidence_weight = float(risk_cfg.get("confidence_weight", 0.60))
        except Exception:
            confidence_weight = 0.60
        confidence_weight = cls._clip01(confidence_weight)

        try:
            ev_scale = float(risk_cfg.get("ev_lcb_scale_points", 6.0))
        except Exception:
            ev_scale = 6.0
        ev_scale = max(1e-6, ev_scale)

        confidence = cls._clip01(float(policy_eval.get("confidence", 0.5) or 0.5))
        try:
            ev_lcb = float(policy_eval.get("ev_lcb_points", 0.0) or 0.0)
        except Exception:
            ev_lcb = 0.0
        ev_score = cls._clip01(0.5 + (0.5 * np.tanh(ev_lcb / ev_scale)))

        score = ((1.0 - confidence_weight) * ev_score) + (confidence_weight * confidence)
        risk_mult = float(min_mult + (max_mult - min_mult) * score)
        if not np.isfinite(risk_mult):
            risk_mult = 1.0
        return float(max(min_mult, min(max_mult, risk_mult)))

    @staticmethod
    def _size_from_risk_mult(base_size: int, risk_mult: float, policy_cfg: Optional[dict]) -> int:
        cfg = policy_cfg or {}
        risk_cfg = cfg.get("risk", {}) or {}
        try:
            min_contracts = int(risk_cfg.get("min_contracts", 1) or 1)
        except Exception:
            min_contracts = 1
        if min_contracts < 1:
            min_contracts = 1
        try:
            max_contracts = int(risk_cfg.get("max_contracts", max(min_contracts, base_size * 3)) or max(min_contracts, base_size * 3))
        except Exception:
            max_contracts = max(min_contracts, base_size * 3)
        if max_contracts < min_contracts:
            max_contracts = min_contracts
        try:
            raw_size = int(round(float(base_size) * float(risk_mult)))
        except Exception:
            raw_size = base_size
        return int(max(min_contracts, min(max_contracts, max(1, raw_size))))

    @classmethod
    def _compute_candidate_selection_metrics(
        cls,
        cand: dict,
        final_tp: float,
        final_sl: float,
        fees_per_side: float,
        point_value: float,
        prefer_policy_ev_lcb: bool = True,
        use_policy_edge: bool = True,
    ) -> Tuple[float, float, float, float]:
        """Return (selection_score, edge_points, edge_proxy_points, confidence)."""
        try:
            opt_wr = float(cand.get("opt_wr", 0.5) or 0.5)
        except Exception:
            opt_wr = 0.5
        opt_wr = cls._clip01(opt_wr)

        fee_points = 0.0
        if np.isfinite(point_value) and point_value > 0:
            fee_points = float((2.0 * float(fees_per_side)) / float(point_value))

        edge_proxy_points = float((opt_wr * float(final_tp)) - ((1.0 - opt_wr) * float(final_sl)) - fee_points)

        try:
            policy_ev_lcb = float(cand.get("de3_policy_ev_lcb_points", np.nan))
        except Exception:
            policy_ev_lcb = np.nan
        try:
            policy_ev_mean = float(cand.get("de3_policy_ev_points", np.nan))
        except Exception:
            policy_ev_mean = np.nan
        if use_policy_edge:
            try:
                confidence = cls._clip01(float(cand.get("de3_policy_confidence", 0.5) or 0.5))
            except Exception:
                confidence = 0.5
        else:
            # Keep "risk-only" experiments isolated: when policy edge is disabled,
            # do not let policy confidence leak into later execution-policy stages.
            confidence = 0.5

        edge_points = edge_proxy_points
        if use_policy_edge and prefer_policy_ev_lcb and np.isfinite(policy_ev_lcb):
            edge_points = float(policy_ev_lcb - fee_points)
        elif use_policy_edge and np.isfinite(policy_ev_mean):
            edge_points = float((policy_ev_mean * (0.75 + (0.25 * confidence))) - fee_points)

        # Keep selection_score for backward compatibility, but keep it neutral and
        # local-edge-based so final ranking is controlled by runtime_rank_score.
        selection_score = float(edge_points)

        if not np.isfinite(selection_score):
            selection_score = edge_points
        if not np.isfinite(edge_points):
            edge_points = edge_proxy_points
        if not np.isfinite(edge_proxy_points):
            edge_proxy_points = 0.0
        return float(selection_score), float(edge_points), float(edge_proxy_points), float(confidence)

    @staticmethod
    def _extract_candidate_quality_metrics(cand: dict) -> Dict[str, Any]:
        try:
            structural_score = float(
                cand.get(
                    "StructuralScore",
                    cand.get("structural_score", cand.get("de3_v2_rank_score", 0.0)),
                )
                or 0.0
            )
        except Exception:
            structural_score = 0.0
        structural_pass = bool(cand.get("StructuralPass", cand.get("structural_pass", True)))
        try:
            worst_block_avg_pnl = float(cand.get("WorstBlockAvgPnL", cand.get("worst_block_avg_pnl", 0.0)) or 0.0)
        except Exception:
            worst_block_avg_pnl = 0.0
        try:
            worst_block_pf = float(cand.get("WorstBlockPF", cand.get("worst_block_pf", 0.0)) or 0.0)
        except Exception:
            worst_block_pf = 0.0
        try:
            profitable_block_ratio = float(
                cand.get("ProfitableBlockRatio", cand.get("profitable_block_ratio", 0.0)) or 0.0
            )
        except Exception:
            profitable_block_ratio = 0.0
        try:
            stop_like_share = float(cand.get("oos_stop_like_share", 0.0) or 0.0)
        except Exception:
            stop_like_share = 0.0
        try:
            loss_share = float(cand.get("oos_loss_share", 0.0) or 0.0)
        except Exception:
            loss_share = 0.0
        return {
            "structural_score": float(structural_score),
            "structural_pass": bool(structural_pass),
            "worst_block_avg_pnl": float(worst_block_avg_pnl),
            "worst_block_pf": float(worst_block_pf),
            "profitable_block_ratio": float(profitable_block_ratio),
            "stop_like_share": float(stop_like_share),
            "loss_share": float(loss_share),
        }

    def _compute_v2_runtime_rank(
        self,
        cand: dict,
        *,
        edge_points: float,
        second_best_edge_points: Optional[float] = None,
        confidence: float = 0.5,
        bucket_concentration: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """Return (runtime_rank_score, components)."""
        def _to_float(value, fallback: float = 0.0) -> float:
            try:
                out = float(value)
                if np.isfinite(out):
                    return out
            except Exception:
                pass
            return float(fallback)

        structural_score = _to_float(
            cand.get("StructuralScore", cand.get("structural_score", cand.get("de3_v2_rank_score", 0.0))),
            0.0,
        )
        structural_pass = bool(cand.get("StructuralPass", cand.get("structural_pass", True)))
        bucket_score = _to_float(cand.get("final_score", cand.get("score_raw", 0.0)), 0.0)

        edge_term = self._clip_range(float(edge_points) / 0.75, -2.0, 2.0)
        if second_best_edge_points is None:
            edge_gap = 0.0
        else:
            edge_gap = float(edge_points - float(second_best_edge_points))
        edge_gap_term = self._clip_range(edge_gap / 0.30, -2.0, 2.0)
        struct_term = self._clip_range(float(structural_score) / 2.0, -2.0, 2.0)
        bucket_term = self._clip_range(float(bucket_score) / 8.0, -2.0, 2.0)
        conf_term = self._clip_range((self._clip01(confidence) - 0.5) / 0.25, -2.0, 2.0)

        ambiguity_penalty = 0.0
        if edge_gap < 0.20:
            ambiguity_penalty = self._clip_range((0.20 - edge_gap) / 0.20, 0.0, 1.0)
        concentration_penalty = self._clip_range(float(bucket_concentration), 0.0, 1.0)

        rank_score = (
            (self._de3_v2_runtime_w_edge * edge_term)
            + (self._de3_v2_runtime_w_gap * edge_gap_term)
            + (self._de3_v2_runtime_w_struct * struct_term)
            + (self._de3_v2_runtime_w_bucket * bucket_term)
            + (self._de3_v2_runtime_w_conf * conf_term)
            + (self._de3_v2_runtime_w_ambiguity * ambiguity_penalty)
            + (self._de3_v2_runtime_w_concentration * concentration_penalty)
        )

        return float(rank_score), {
            "structural_score": float(structural_score),
            "structural_pass": 1.0 if structural_pass else 0.0,
            "edge_points": float(edge_points),
            "edge_gap": float(edge_gap),
            "edge_term": float(edge_term),
            "edge_gap_term": float(edge_gap_term),
            "bucket_score": float(bucket_score),
            "bucket_term": float(bucket_term),
            "confidence": float(self._clip01(confidence)),
            "confidence_term": float(conf_term),
            "ambiguity_penalty": float(ambiguity_penalty),
            "concentration_penalty": float(concentration_penalty),
            "runtime_rank_score": float(rank_score),
        }

    def _prune_drift_anchors(self, current_time: pd.Timestamp) -> None:
        if not self._drift_anchors:
            return
        cutoff = pd.Timestamp(current_time) - pd.Timedelta(hours=8)
        self._drift_anchors = {k: v for k, v in self._drift_anchors.items() if pd.Timestamp(k[2]) >= cutoff}

    def _passes_drift_gate(
        self,
        *,
        strategy_id: str,
        side: str,
        signal_tf: str,
        current_time: pd.Timestamp,
        current_price: float,
        df_1m: pd.DataFrame,
        df_tf: Optional[pd.DataFrame],
    ) -> Tuple[bool, Dict]:
        if not self._drift_enabled or self._drift_max_atr <= 0:
            return True, {}
        self._prune_drift_anchors(current_time)

        bucket = self._signal_bucket(current_time, signal_tf, strategy_id)
        key = (str(strategy_id or ""), str(side or "").upper(), bucket)
        if key not in self._drift_anchors:
            self._drift_anchors[key] = float(current_price)

        anchor = float(self._drift_anchors[key])
        dist_points = abs(float(current_price) - anchor)

        atr_value = None
        df_base = None
        if df_tf is not None and not df_tf.empty:
            df_base = df_tf.iloc[:-1] if len(df_tf) > 1 else df_tf
        atr_value = self._compute_atr_simple(df_base, self._drift_atr_period) if df_base is not None else None
        if atr_value is None:
            atr_value = self._compute_atr_simple(df_1m, self._drift_atr_period)
        if atr_value is None or not np.isfinite(atr_value) or atr_value <= 0:
            atr_value = self._drift_fallback_points if self._drift_fallback_points > 0 else None
        if atr_value is None or atr_value <= 0:
            return True, {
                "de3_drift_anchor": float(anchor),
                "de3_drift_dist_points": float(dist_points),
                "de3_drift_dist_atr": None,
                "de3_drift_limit_atr": float(self._drift_max_atr),
            }

        dist_atr = float(dist_points / float(atr_value))
        ok = dist_atr <= float(self._drift_max_atr)
        return ok, {
            "de3_drift_anchor": float(anchor),
            "de3_drift_dist_points": float(dist_points),
            "de3_drift_dist_atr": float(dist_atr),
            "de3_drift_limit_atr": float(self._drift_max_atr),
            "de3_drift_atr": float(atr_value),
        }

    def get_veto_summary(self) -> dict:
        return dict(self.veto_stats)

    def get_db_version(self) -> str:
        return str(getattr(self.engine, "db_version", self.db_version) or self.db_version)

    def get_runtime_metadata(self) -> dict:
        status = dict(self._de3_v3_family_status) if isinstance(self._de3_v3_family_status, dict) else {}
        v4_status = dict(self._de3_v4_status) if isinstance(self._de3_v4_status, dict) else {}
        runtime_counters = {}
        activation_audit = {}
        bundle_usage_audit = {}
        config_usage_audit = {}
        score_path_audit = {}
        choice_path_audit = {}
        family_score_trace = {}
        member_resolution_audit = {}
        family_eligibility_trace = {}
        family_reachability_summary = {}
        family_compatibility_audit = {}
        pre_cap_candidate_audit = {}
        family_score_component_summary = {}
        family_score_delta_ladder = {}
        runtime_mode_summary = {}
        core_summary = {}
        t6_anchor_report = {}
        satellite_quality_report = {}
        portfolio_increment_report = {}
        v4_runtime_counters = {}
        v4_activation_audit = {}
        v4_router_summary = {}
        v4_lane_selection_summary = {}
        v4_bracket_summary = {}
        v4_runtime_mode_summary = {}
        v4_execution_policy_summary = {}
        v4_decision_side_summary = {}
        if self._de3_v3_family_runtime is not None:
            try:
                runtime_counters = self._de3_v3_family_runtime.get_runtime_path_counters()
            except Exception:
                runtime_counters = {}
            try:
                activation_audit = self._de3_v3_family_runtime.get_activation_audit()
            except Exception:
                activation_audit = {}
            try:
                bundle_usage_audit = self._de3_v3_family_runtime.get_bundle_usage_audit()
            except Exception:
                bundle_usage_audit = {}
            try:
                config_usage_audit = self._de3_v3_family_runtime.get_config_usage_audit()
            except Exception:
                config_usage_audit = {}
            try:
                score_path_audit = self._de3_v3_family_runtime.get_score_path_audit()
            except Exception:
                score_path_audit = {}
            try:
                choice_path_audit = self._de3_v3_family_runtime.get_choice_path_audit()
            except Exception:
                choice_path_audit = {}
            try:
                family_score_trace = self._de3_v3_family_runtime.get_family_score_trace()
            except Exception:
                family_score_trace = {}
            try:
                member_resolution_audit = self._de3_v3_family_runtime.get_member_resolution_audit()
            except Exception:
                member_resolution_audit = {}
            try:
                family_eligibility_trace = (
                    self._de3_v3_family_runtime.get_family_eligibility_trace()
                )
            except Exception:
                family_eligibility_trace = {}
            try:
                family_reachability_summary = (
                    self._de3_v3_family_runtime.get_family_reachability_summary()
                )
            except Exception:
                family_reachability_summary = {}
            try:
                family_compatibility_audit = (
                    self._de3_v3_family_runtime.get_family_compatibility_audit()
                )
            except Exception:
                family_compatibility_audit = {}
            try:
                pre_cap_candidate_audit = (
                    self._de3_v3_family_runtime.get_pre_cap_candidate_audit()
                )
            except Exception:
                pre_cap_candidate_audit = {}
            try:
                family_score_component_summary = (
                    self._de3_v3_family_runtime.get_family_score_component_summary()
                )
            except Exception:
                family_score_component_summary = {}
            try:
                family_score_delta_ladder = (
                    self._de3_v3_family_runtime.get_family_score_delta_ladder()
                )
            except Exception:
                family_score_delta_ladder = {}
            try:
                runtime_mode_summary = (
                    self._de3_v3_family_runtime.get_runtime_mode_summary()
                )
            except Exception:
                runtime_mode_summary = {}
            try:
                core_summary = self._de3_v3_family_runtime.get_core_summary()
            except Exception:
                core_summary = {}
            try:
                t6_anchor_report = self._de3_v3_family_runtime.get_t6_anchor_report()
            except Exception:
                t6_anchor_report = {}
            try:
                satellite_quality_report = (
                    self._de3_v3_family_runtime.get_satellite_quality_report()
                )
            except Exception:
                satellite_quality_report = {}
            try:
                portfolio_increment_report = (
                    self._de3_v3_family_runtime.get_portfolio_increment_report()
                )
            except Exception:
                portfolio_increment_report = {}
        if self._de3_v4_runtime_module is not None:
            try:
                v4_runtime_counters = self._de3_v4_runtime_module.get_runtime_path_counters()
            except Exception:
                v4_runtime_counters = {}
            try:
                v4_activation_audit = self._de3_v4_runtime_module.get_activation_audit()
            except Exception:
                v4_activation_audit = {}
            try:
                v4_router_summary = self._de3_v4_runtime_module.get_router_summary()
            except Exception:
                v4_router_summary = {}
            try:
                v4_lane_selection_summary = self._de3_v4_runtime_module.get_lane_selection_summary()
            except Exception:
                v4_lane_selection_summary = {}
            try:
                v4_bracket_summary = self._de3_v4_runtime_module.get_bracket_summary()
            except Exception:
                v4_bracket_summary = {}
            try:
                v4_runtime_mode_summary = self._de3_v4_runtime_module.get_runtime_mode_summary()
            except Exception:
                v4_runtime_mode_summary = {}
            try:
                v4_execution_policy_summary = self._de3_v4_runtime_module.get_execution_policy_summary()
            except Exception:
                v4_execution_policy_summary = {}
            try:
                v4_decision_side_summary = self._de3_v4_runtime_module.get_decision_side_summary()
            except Exception:
                v4_decision_side_summary = {}
        return {
            "db_version": str(self.get_db_version()),
            "family_mode_enabled": bool(self._de3_v3_runtime and self._de3_v3_family_runtime is not None),
            "family_artifact_path": status.get("family_artifact_path"),
            "family_artifact_loaded": bool(status.get("family_artifact_loaded", False)),
            "family_count": int(status.get("family_count", 0) or 0),
            "context_profiles_loaded": bool(status.get("context_profiles_loaded", False)),
            "context_profile_build": status.get("context_profile_build"),
            "enriched_export_required": bool(status.get("enriched_export_required", False)),
            "export_raw_context_fields_in_decision_journal": bool(self._de3_v3_export_raw_context_fields),
            "active_context_dimensions": status.get("active_context_dimensions"),
            "context_trust": status.get("context_trust"),
            "local_bracket_freeze": status.get("local_bracket_freeze"),
            "runtime_state_loaded": bool(status.get("runtime_state_loaded", False)),
            "runtime_state_build": status.get("runtime_state_build"),
            "runtime_path_counters": dict(runtime_counters),
            "activation_audit": dict(activation_audit),
            "bundle_usage_audit": dict(bundle_usage_audit),
            "config_usage_audit": dict(config_usage_audit),
            "score_path_audit": dict(score_path_audit),
            "choice_path_audit": dict(choice_path_audit),
            "family_score_trace": dict(family_score_trace),
            "member_resolution_audit": dict(member_resolution_audit),
            "family_eligibility_trace": dict(family_eligibility_trace),
            "family_reachability_summary": dict(family_reachability_summary),
            "family_compatibility_audit": dict(family_compatibility_audit),
            "pre_cap_candidate_audit": dict(pre_cap_candidate_audit),
            "family_score_component_summary": dict(family_score_component_summary),
            "family_score_delta_ladder": dict(family_score_delta_ladder),
            "runtime_mode_summary": dict(runtime_mode_summary),
            "core_summary": dict(core_summary),
            "t6_anchor_report": dict(t6_anchor_report),
            "satellite_quality_report": dict(satellite_quality_report),
            "portfolio_increment_report": dict(portfolio_increment_report),
            "family_status": status,
            "v4_runtime_enabled": bool(self._de3_v4_runtime and self._de3_v4_runtime_module is not None),
            "v4_status": dict(v4_status),
            "v4_runtime_path_counters": dict(v4_runtime_counters),
            "v4_activation_audit": dict(v4_activation_audit),
            "v4_router_summary": dict(v4_router_summary),
            "v4_lane_selection_summary": dict(v4_lane_selection_summary),
            "v4_bracket_summary": dict(v4_bracket_summary),
            "v4_runtime_mode_summary": dict(v4_runtime_mode_summary),
            "v4_execution_policy_summary": dict(v4_execution_policy_summary),
            "v4_decision_side_summary": dict(v4_decision_side_summary),
        }

    @staticmethod
    def _de3_v3_context_inputs(
        *,
        current_time: pd.Timestamp,
        engine_session: Optional[str],
        vol_regime: Optional[str],
        atr_5m: Optional[float],
        atr_med: Optional[float],
        atr_ratio: Optional[float],
        vwap_dist_atr: Optional[float],
        price_loc: Optional[float],
        rvol_ratio: Optional[float],
    ) -> Dict[str, Any]:
        hour_et = int(pd.Timestamp(current_time).hour)
        vol_raw = str(vol_regime or "").strip().lower()
        if vol_raw in {"low", "ultra_low"}:
            vol_bucket = "low"
        elif vol_raw in {"high", "ultra_high"}:
            vol_bucket = "high"
        else:
            vol_bucket = "mid"

        if atr_ratio is not None and np.isfinite(float(atr_ratio)):
            atr_ratio_val = float(atr_ratio)
            if atr_ratio_val >= 1.10:
                comp_bucket = "expanding"
            elif atr_ratio_val <= 0.90:
                comp_bucket = "compressed"
            else:
                comp_bucket = "neutral"
        else:
            atr_ratio_val = None
            comp_bucket = "neutral"

        if vwap_dist_atr is not None and np.isfinite(float(vwap_dist_atr)):
            vwap_val = float(vwap_dist_atr)
            if vwap_val >= 1.10:
                chop_bucket = "trend"
            elif vwap_val <= 0.55:
                chop_bucket = "chop"
            else:
                chop_bucket = "neutral"
        else:
            vwap_val = None
            chop_bucket = "neutral"

        if price_loc is not None and np.isfinite(float(price_loc)):
            price_loc_val = float(price_loc)
            if price_loc_val >= 0.70:
                conf_bucket = "high"
            elif price_loc_val <= 0.30:
                conf_bucket = "low"
            else:
                conf_bucket = "mid"
        else:
            price_loc_val = None
            conf_bucket = "mid"

        if rvol_ratio is not None and np.isfinite(float(rvol_ratio)):
            rvol_val = float(rvol_ratio)
            if rvol_val >= 1.15:
                rvol_bucket = "high"
            elif rvol_val <= 0.85:
                rvol_bucket = "low"
            else:
                rvol_bucket = "normal"
        else:
            rvol_val = None
            rvol_bucket = "normal"

        session_substate = ""
        session_text = str(engine_session or "")
        try:
            start_hour = int(session_text.split("-")[0])
            rel_hour = (hour_et - start_hour) % 24
            if rel_hour < 1:
                session_substate = "open"
            elif rel_hour < 2:
                session_substate = "mid"
            else:
                session_substate = "late"
        except Exception:
            session_substate = ""

        out: Dict[str, Any] = {
            "session": str(engine_session or ""),
            "hour_et": int(hour_et),
            "volatility_regime": vol_bucket,
            "chop_trend_regime": chop_bucket,
            "compression_expansion_regime": comp_bucket,
            "confidence_band": conf_bucket,
            "rvol_liquidity_state": rvol_bucket,
            "session_substate": session_substate,
        }
        if atr_ratio_val is not None:
            out["atr_ratio"] = atr_ratio_val
        if vwap_val is not None:
            out["vwap_dist_atr"] = vwap_val
        if price_loc_val is not None:
            out["price_location"] = price_loc_val
        if rvol_val is not None:
            out["rvol_ratio"] = rvol_val
        if atr_5m is not None and np.isfinite(float(atr_5m)):
            out["atr_5m"] = float(atr_5m)
        if atr_med is not None and np.isfinite(float(atr_med)):
            out["atr_5m_median"] = float(atr_med)
        return out

    @staticmethod
    def _apply_v3_family_metrics_to_candidate(
        candidate: Dict[str, Any],
        *,
        family_id: str,
        family_score: Optional[float],
        member_local_score: Optional[float],
        family_components: Dict[str, float],
    ) -> None:
        if not isinstance(candidate, dict):
            return
        candidate["de3_family_id"] = str(family_id or candidate.get("de3_family_id", ""))
        if family_score is not None:
            candidate["de3_family_score"] = float(family_score)
        candidate["de3_family_context_ev"] = float(
            family_components.get("context_profile_expectancy_component", family_components.get("context_ev_component", 0.0))
            or 0.0
        )
        candidate["de3_family_confidence"] = float(
            family_components.get("context_profile_confidence_component", family_components.get("confidence_component", 0.0))
            or 0.0
        )
        candidate["de3_family_prior"] = float(family_components.get("family_prior_component", 0.0) or 0.0)
        candidate["de3_family_profile"] = float(
            family_components.get("context_profile_expectancy_component", family_components.get("context_profile_component", 0.0))
            or 0.0
        )
        candidate["de3_family_context_support_ratio"] = float(family_components.get("context_support_ratio", 0.0) or 0.0)
        candidate["de3_family_context_support_tier"] = str(
            family_components.get("context_support_tier", "low") or "low"
        )
        candidate["de3_family_context_sample_count"] = float(
            family_components.get("context_sample_count", 0.0) or 0.0
        )
        candidate["de3_family_context_weight"] = float(
            family_components.get("context_profile_weight", 0.0) or 0.0
        )
        candidate["de3_family_evidence_support_tier"] = str(
            family_components.get("evidence_support_tier", "none") or "none"
        )
        candidate["de3_family_competition_status"] = str(
            family_components.get("competition_status", "competitive") or "competitive"
        )
        candidate["de3_family_usability_adjustment"] = float(
            family_components.get("usability_adjustment", 0.0) or 0.0
        )
        candidate["de3_family_adaptive_component"] = float(family_components.get("adaptive_policy_component", 0.0) or 0.0)
        candidate["de3_family_usability_component"] = float(
            family_components.get("v3_realized_usability_component", 0.0) or 0.0
        )
        candidate["de3_family_base_score"] = float(
            family_components.get("base_family_score", family_score if family_score is not None else 0.0) or 0.0
        )
        candidate["de3_family_diversity_adjustment"] = float(
            family_components.get("diversity_adjustment", 0.0) or 0.0
        )
        candidate["de3_family_recent_chosen_share"] = float(
            family_components.get("recent_chosen_share", 0.0) or 0.0
        )
        if member_local_score is not None:
            candidate["de3_member_local_score"] = float(member_local_score)

    def on_bar(self, df: pd.DataFrame) -> Optional[Dict]:
        if df is None or len(df) < 60:
            return None

        current_time = df.index[-1]
        if self.last_processed_time == current_time:
            return None

        df_5m = self._resampler_5m.update(df)
        if df_5m.empty:
            return None
        if self._v1_specific_filters_enabled:
            veto_cfg = self._veto_cfg or {}
            df_5m = self._augment_5m_features(
                df_5m,
                atr_period=int(veto_cfg.get("atr_period", 20) or 20),
                atr_median_window=int(veto_cfg.get("atr_median_window", 390) or 390),
                price_loc_window=int(veto_cfg.get("price_location_window", 20) or 20),
            )
        self._cached_5m = df_5m
        self._last_5m_close = df_5m.index[-1]

        df_15m = self._resampler_15m.update(df)
        if df_15m.empty:
            return None
        self._cached_15m = df_15m
        self._last_15m_close = df_15m.index[-1]

        if df_5m is None or df_15m is None:
            return None

        self.last_processed_time = current_time
        cache_session = None
        try:
            cache_session = self.engine.get_session_from_time(current_time)
        except Exception:
            cache_session = None
        prev_5m_close = df_5m.index[-2] if len(df_5m) >= 2 else None
        prev_15m_close = df_15m.index[-2] if len(df_15m) >= 2 else None
        candidate_cache_key = (cache_session, prev_5m_close, prev_15m_close)
        if self._candidate_cache_key == candidate_cache_key:
            candidates = [dict(c) for c in self._candidate_cache_value]
        else:
            candidates = []
            if hasattr(self.engine, "check_signals"):
                try:
                    candidates = self.engine.check_signals(current_time, df_5m, df_15m) or []
                except Exception:
                    candidates = []
            if not candidates:
                signal_data = self.engine.check_signal(current_time, df_5m, df_15m)
                candidates = [signal_data] if signal_data else []
            self._candidate_cache_key = candidate_cache_key
            self._candidate_cache_value = [dict(c) for c in candidates]

        if candidates:
            engine_session = None
            try:
                engine_session = self.engine.get_session_from_time(current_time)
            except Exception:
                engine_session = None

            ny_conf = CONFIG.get("DYNAMIC_ENGINE3_NY_CONF")
            if ny_conf is None:
                ny_conf = CONFIG.get("DYNAMIC_ENGINE_NY_CONF", {}) or {}

            # If a specific sub-strategy is blocked (guards), we fall back to the next-best candidate.
            # This prevents "disable X" from accidentally zeroing out DynamicEngine3's vote in consensus.
            ny_gate_active = bool(ny_conf.get("enabled")) and bool(engine_session) and engine_session in set(
                ny_conf.get("sessions", []) or []
            )
            if self._de3_v4_disable_ny_conf_gate:
                ny_gate_active = False
            min_opt_wr = 0.0
            min_final_score = None
            if ny_gate_active:
                try:
                    min_opt_wr = float(ny_conf.get("min_opt_wr", 0.0))
                except Exception:
                    min_opt_wr = 0.0
                min_final_score = ny_conf.get("min_final_score")
                try:
                    min_final_score = float(min_final_score) if min_final_score is not None else None
                except Exception:
                    min_final_score = None

            sltp_session = volatility_filter.get_session(current_time.hour)
            min_cfg = CONFIG.get("SLTP_MIN", {}) or {}
            min_sl = float(min_cfg.get("sl", 1.25))
            min_tp = float(min_cfg.get("tp", 1.5))
            long_mom_cfg = CONFIG.get("DE3_LONG_MOM_BRACKETS", {}) or {}

            risk_cfg = CONFIG.get("RISK", {})
            point_value = risk_cfg.get("POINT_VALUE", 5.0)
            fees_per_side = risk_cfg.get("FEES_PER_SIDE", 2.50)
            min_net_profit = risk_cfg.get("MIN_NET_PROFIT", 10.0)
            enforce_min_net = bool(risk_cfg.get("ENFORCE_MIN_NET_PROFIT", True))

            vol_regime = None
            if self._v1_specific_filters_enabled:
                try:
                    vol_regime, _, _ = volatility_filter.get_regime(df, current_time)
                except Exception:
                    vol_regime = None

            regime_norm = str(vol_regime or "").lower()

            parsed_candidates = []
            for cand in candidates:
                cand_tf_raw = str(cand.get("timeframe") or "")
                cand_type_raw = str(cand.get("strategy_type") or "")
                try:
                    cand_thresh = float(cand.get("thresh"))
                except Exception:
                    cand_thresh = None
                parsed_candidates.append(
                    (
                        cand,
                        cand.get("strategy_id"),
                        cand_tf_raw,
                        cand_type_raw,
                        cand_tf_raw.lower(),
                        cand_type_raw.lower(),
                        cand_thresh,
                    )
                )

            safety_cfg = CONFIG.get("DE3_SAFETY_GUARDS", {}) or {}
            safety_enabled = bool(self._v1_specific_filters_enabled and safety_cfg.get("enabled", True))
            if safety_enabled:
                # User-requested behavior: a safety-guard match in candidacy blocks the entire DE3 trade.
                for cand, cand_id, _, _, cand_tf_scan, cand_type_scan, cand_thresh_scan in parsed_candidates:
                    safety_reason = self._get_safety_guard_reason(
                        cand_tf_scan,
                        cand_type_scan,
                        cand_thresh_scan,
                        regime_norm,
                        safety_cfg=safety_cfg,
                    )
                    if safety_reason:
                        self._warn(
                            "🚫 DynamicEngine3 safety block: %s (candidate=%s)",
                            safety_reason,
                            cand_id,
                        )
                        return None

            veto_cfg = self._veto_cfg or {}
            policy_cfg = self._policy_cfg or {}
            policy_enabled = bool(self._de3_context_eval_enabled and policy_cfg.get("enabled", False))
            policy_mode = str(policy_cfg.get("mode", "block") or "block").lower()
            if policy_mode not in {"block", "shadow"}:
                policy_mode = "block"
            policy_shadow = policy_mode == "shadow"
            policy_log_decisions = bool(policy_cfg.get("log_decisions", False))

            veto_enabled = bool(self._de3_context_eval_enabled and veto_cfg.get("enabled", False))
            veto_ready = bool(self._veto and self._veto.ready and veto_enabled)
            veto_mode = str(veto_cfg.get("mode", "block") or "block").lower()
            shadow_mode = veto_mode == "shadow"
            if self._de3_v4_disable_context_veto_gate:
                veto_enabled = False
                veto_ready = False

            if self._de3_v4_disable_context_policy_gate:
                policy_enabled = False
            use_policy = bool(policy_enabled and self._veto and self._veto.ready)
            use_legacy_veto = bool((not use_policy) and veto_ready)
            context_enabled = bool(policy_enabled or veto_enabled)

            self.veto_stats["enabled"] = bool(context_enabled)
            self.veto_stats["policy_enabled"] = bool(policy_enabled)
            self.veto_stats["mode"] = str(policy_mode if policy_enabled else veto_mode)
            self.veto_stats["policy_mode"] = str(policy_mode)
            self.veto_stats["model_ready"] = bool(self._veto and self._veto.ready)
            if context_enabled:
                self.veto_stats["decisions"] += 1
            if policy_enabled and not use_policy:
                self.veto_stats["skipped"] += 1
                self.veto_stats["policy_skipped"] += 1
            elif (not policy_enabled) and veto_enabled and not veto_ready:
                self.veto_stats["skipped"] += 1

            veto_threshold = veto_cfg.get("threshold", None)
            try:
                veto_uncertainty_z = float(veto_cfg.get("uncertainty_z", 1.0) or 1.0)
            except Exception:
                veto_uncertainty_z = 1.0
            veto_max_std_raw = veto_cfg.get("max_std", None)
            try:
                veto_max_std = float(veto_max_std_raw) if veto_max_std_raw is not None else None
            except Exception:
                veto_max_std = None
            try:
                veto_min_samples = int(veto_cfg.get("min_bucket_samples", 0) or 0)
            except Exception:
                veto_min_samples = 0
            veto_block_all_on_top = bool(veto_cfg.get("block_all_on_top_veto", False))

            try:
                policy_min_samples = int(policy_cfg.get("min_samples", 120) or 120)
            except Exception:
                policy_min_samples = 120
            try:
                policy_uncertainty_z = float(policy_cfg.get("uncertainty_z", 1.0) or 1.0)
            except Exception:
                policy_uncertainty_z = 1.0
            policy_max_std_raw = policy_cfg.get("max_p_loss_std", 0.20)
            try:
                policy_max_std = float(policy_max_std_raw) if policy_max_std_raw is not None else None
            except Exception:
                policy_max_std = None
            policy_blend_empirical = bool(policy_cfg.get("blend_empirical", True))
            try:
                policy_prior_strength = float(policy_cfg.get("prior_strength", 300.0) or 300.0)
            except Exception:
                policy_prior_strength = 300.0
            try:
                policy_min_ev_lcb = float(policy_cfg.get("min_ev_lcb_points", 0.0) or 0.0)
            except Exception:
                policy_min_ev_lcb = 0.0
            policy_min_ev_mean_raw = policy_cfg.get("min_ev_mean_points", None)
            try:
                policy_min_ev_mean = (
                    float(policy_min_ev_mean_raw) if policy_min_ev_mean_raw is not None else None
                )
            except Exception:
                policy_min_ev_mean = None
            policy_block_all_on_top = bool(policy_cfg.get("block_all_on_top", True))
            if self._de3_v4_profit_gate_enabled:
                policy_mode = str(self._de3_v4_profit_gate_policy_mode)
                policy_shadow = policy_mode == "shadow"
                if self._de3_v4_profit_gate_min_samples is not None:
                    policy_min_samples = int(self._de3_v4_profit_gate_min_samples)
                if self._de3_v4_profit_gate_max_p_loss_std is not None:
                    policy_max_std = float(self._de3_v4_profit_gate_max_p_loss_std)
                if self._de3_v4_profit_gate_min_ev_lcb_points is not None:
                    policy_min_ev_lcb = float(self._de3_v4_profit_gate_min_ev_lcb_points)
                if self._de3_v4_profit_gate_min_ev_mean_points is not None:
                    policy_min_ev_mean = float(self._de3_v4_profit_gate_min_ev_mean_points)
                if self._de3_v4_profit_gate_disable_block_all_on_top:
                    policy_block_all_on_top = False
            policy_risk_cfg = (policy_cfg.get("risk", {}) or {})
            policy_risk_enabled = bool(policy_risk_cfg.get("enabled", True))
            policy_apply_size = bool(policy_risk_cfg.get("apply_to_size", True))
            try:
                base_contracts = int(policy_risk_cfg.get("base_size", 5) or 5)
            except Exception:
                base_contracts = 5
            if base_contracts < 1:
                base_contracts = 1

            atr_5m = None
            atr_med = None
            price_loc = None
            vwap_dist_atr = None
            atr_ratio = None
            rvol_ratio = None
            context_eval_ready = bool(
                use_policy or use_legacy_veto or self._de3_v3_runtime or self._de3_v4_runtime
            )
            if context_eval_ready:
                veto_cfg = self._veto_cfg or {}
                atr_period_ctx = int(veto_cfg.get("atr_period", 20) or 20)
                atr_median_window_ctx = int(veto_cfg.get("atr_median_window", 390) or 390)
                price_loc_window_ctx = int(veto_cfg.get("price_location_window", 20) or 20)
                snapshot = None
                try:
                    if {"atr_5m", "atr_5m_median", "price_location"}.issubset(df_5m.columns):
                        atr_5m = float(df_5m["atr_5m"].iloc[-1])
                        atr_med = float(df_5m["atr_5m_median"].iloc[-1])
                        price_loc = float(df_5m["price_location"].iloc[-1])
                    else:
                        snapshot = self._fast_5m_context_snapshot(
                            df_5m,
                            atr_period=atr_period_ctx,
                            atr_median_window=atr_median_window_ctx,
                            price_loc_window=price_loc_window_ctx,
                            rvol_window=20,
                        )
                        if isinstance(snapshot, dict):
                            atr_5m = float(snapshot.get("atr_5m", np.nan))
                            atr_med = float(snapshot.get("atr_5m_median", np.nan))
                            price_loc = float(snapshot.get("price_location", 0.5))
                except Exception:
                    atr_5m = None
                    atr_med = None
                    price_loc = None
                if atr_5m is None or atr_med is None or not np.isfinite(atr_5m) or atr_5m <= 0:
                    context_eval_ready = False
                elif not np.isfinite(atr_med) or atr_med <= 0:
                    context_eval_ready = False
                else:
                    if price_loc is None or not np.isfinite(price_loc):
                        price_loc = 0.5
                    entry_price = float(df["close"].iloc[-1])
                    vwap_val = self._compute_vwap_value(df, current_time)
                    if vwap_val is None or not np.isfinite(vwap_val):
                        vwap_dist_atr = 0.0
                    else:
                        vwap_dist_atr = float(abs(entry_price - float(vwap_val)) / atr_5m)
                    atr_ratio = float(atr_5m / atr_med)
                    if isinstance(snapshot, dict) and snapshot.get("rvol_ratio") is not None:
                        try:
                            rvol_ratio = float(snapshot.get("rvol_ratio"))
                        except Exception:
                            rvol_ratio = None
                    elif "volume" in df_5m.columns and len(df_5m) >= 20:
                        vol_now = float(df_5m["volume"].iloc[-1])
                        vol_base = float(df_5m["volume"].iloc[-20:].mean())
                        if np.isfinite(vol_now) and np.isfinite(vol_base) and vol_base > 0:
                            rvol_ratio = float(vol_now / vol_base)

                    # v3 fast-path: avoid expensive volatility_filter regime calls by
                    # deriving a coarse regime directly from atr_ratio.
                    if (vol_regime is None or str(vol_regime).strip() == "") and np.isfinite(atr_ratio):
                        if atr_ratio >= 1.10:
                            vol_regime = "high"
                            regime_norm = "high"
                        elif atr_ratio <= 0.90:
                            vol_regime = "low"
                            regime_norm = "low"
                        else:
                            vol_regime = "normal"
                            regime_norm = "normal"
            if (use_policy or use_legacy_veto) and not context_eval_ready:
                self.veto_stats["skipped"] += 1
                if use_policy:
                    self.veto_stats["policy_skipped"] += 1
            chosen = None
            chosen_exec = None
            chosen_idx = None
            chosen_decision_id: Optional[str] = None
            chosen_family_id = ""
            chosen_family_score: Optional[float] = None
            chosen_member_local_score: Optional[float] = None
            chosen_family_components: Dict[str, float] = {}
            feasible_candidates: List[Dict[str, object]] = []
            top_blocked_id = None
            top_blocked_reason = None
            veto_block_all = False
            top_candidate_id = parsed_candidates[0][1] if parsed_candidates else None

            selection_cfg = CONFIG.get("DE3_CANDIDATE_SELECTION", {}) or {}
            selection_enabled = bool(selection_cfg.get("enabled", True))
            if self._de3_v3_runtime:
                selection_enabled = True
            prefer_policy_ev_lcb = bool(selection_cfg.get("prefer_policy_ev_lcb", True))
            use_policy_edge_in_ranking = bool(selection_cfg.get("use_policy_edge_in_ranking", True))
            log_rerank = bool(selection_cfg.get("log_rerank", True))
            if self._de3_v2_runtime and self._de3_v2_robust_enabled:
                min_edge_points = float(self._de3_v2_runtime_min_edge)
                min_score_gap_points = float(self._de3_v2_runtime_min_gap)
            else:
                try:
                    min_edge_points = float(selection_cfg.get("min_edge_points", 0.0) or 0.0)
                except Exception:
                    min_edge_points = 0.0
                try:
                    min_score_gap_points = float(selection_cfg.get("min_score_gap_points", 0.05) or 0.05)
                except Exception:
                    min_score_gap_points = 0.05

            entry_struct_cfg = CONFIG.get("DE3_ENTRY_CANDLE_FILTERS", {}) or {}
            # For v4 prune rules we still need entry-structure features even when v1-only
            # candle filters are disabled, otherwise prune inputs remain missing.
            entry_struct_needed_for_v4_prune = bool(self._de3_v4_runtime and self._de3_v4_prune_enabled)
            entry_struct_enabled = bool(
                (self._v1_specific_filters_enabled and entry_struct_cfg.get("enabled", False))
                or entry_struct_needed_for_v4_prune
            )
            entry_struct_log = bool(entry_struct_cfg.get("log_decisions", False))
            entry_struct_features = None
            if entry_struct_enabled:
                try:
                    entry_struct_features = self._compute_entry_structure_features(
                        df,
                        atr_period=int(entry_struct_cfg.get("atr_period", 14) or 14),
                        flip_window=int(entry_struct_cfg.get("flip_window", 5) or 5),
                        range_window=int(entry_struct_cfg.get("range_window", 10) or 10),
                        dist_window=int(entry_struct_cfg.get("dist_window", 5) or 5),
                        min_bars=int(entry_struct_cfg.get("min_bars", 40) or 40),
                    )
                except Exception:
                    entry_struct_features = None
            v2_entry_bar_cfg = self._de3_v2_entry_bar_block_cfg or {}
            v2_entry_bar_enabled = bool(
                self._de3_v2_entry_bar_block_enabled and v2_entry_bar_cfg.get("enabled", False)
            )
            v2_entry_bar_log = bool(v2_entry_bar_cfg.get("log_decisions", False))
            v2_entry_bar_features = None
            if v2_entry_bar_enabled:
                try:
                    v2_entry_bar_features = self._compute_entry_bar_shape_features(
                        df,
                        min_bars=int(v2_entry_bar_cfg.get("min_bars", 2) or 2),
                    )
                except Exception:
                    v2_entry_bar_features = None

            for idx, (cand, cand_id, cand_tf_raw, cand_type_raw, cand_tf, cand_type, cand_thresh) in enumerate(parsed_candidates):
                block_reason = None
                if self._de3_v2_constraints_enabled:
                    cand["de3_v2_constraints_hit"] = False
                    try:
                        cand_trades = int(cand.get("trades", 0) or 0)
                    except Exception:
                        cand_trades = 0
                    try:
                        cand_final_score = float(cand.get("final_score", 0.0) or 0.0)
                    except Exception:
                        cand_final_score = 0.0
                    if self._de3_v2_min_trades > 0 and cand_trades < self._de3_v2_min_trades:
                        block_reason = (
                            f"V2Constraints trades {cand_trades} < {self._de3_v2_min_trades}"
                        )
                    elif (
                        self._de3_v2_max_thresh is not None
                        and cand_thresh is not None
                        and float(cand_thresh) > float(self._de3_v2_max_thresh)
                    ):
                        block_reason = (
                            f"V2Constraints thresh {float(cand_thresh):g} > {float(self._de3_v2_max_thresh):g}"
                        )
                    elif (
                        self._de3_v2_min_final_score is not None
                        and cand_final_score < float(self._de3_v2_min_final_score)
                    ):
                        block_reason = (
                            f"V2Constraints final_score {cand_final_score:.3f} < "
                            f"{float(self._de3_v2_min_final_score):.3f}"
                        )
                    if block_reason is not None:
                        cand["de3_v2_constraints_hit"] = True
                        cand["de3_v2_constraints_reason"] = block_reason
                        if self._de3_v2_constraints_log:
                            logging.info(
                                "DynamicEngine3 v2 constraints block: %s | %s",
                                cand_id,
                                block_reason,
                            )
                if isinstance(entry_struct_features, dict):
                    for k, v in entry_struct_features.items():
                        cand[f"de3_entry_{k}"] = float(v) if np.isfinite(v) else np.nan
                if isinstance(v2_entry_bar_features, dict):
                    for k, v in v2_entry_bar_features.items():
                        if np.isfinite(v):
                            cand[f"de3_v2_entry_{k}"] = float(v)
                if context_eval_ready and atr_ratio is not None and vwap_dist_atr is not None and atr_5m is not None:
                    sl_points = max(min_sl, float(cand.get("sl", min_sl)))
                    tp_points = max(min_tp, float(cand.get("tp", min_tp)))
                    sl_atr = float(sl_points / atr_5m)
                    bucket_keys = self._build_veto_bucket_keys(
                        engine_session,
                        cand_tf_raw,
                        cand_type_raw,
                        regime_norm,
                        cand_thresh,
                    )
                    feat_vec = [atr_ratio, price_loc, vwap_dist_atr, sl_atr]
                    if use_policy:
                        policy_min_samples_eff = int(policy_min_samples)
                        policy_max_std_eff = policy_max_std
                        policy_min_ev_lcb_eff = float(policy_min_ev_lcb)
                        policy_min_ev_mean_eff = policy_min_ev_mean
                        soft_pass_risk_mult_cap_eff = self._de3_v4_profit_gate_soft_pass_risk_mult_cap
                        cat_enabled_eff = bool(self._de3_v4_profit_gate_cat_enabled)
                        cat_min_samples_eff = int(self._de3_v4_profit_gate_cat_min_samples or 0)
                        cat_max_ev_lcb_eff = self._de3_v4_profit_gate_cat_max_ev_lcb_points
                        cat_min_p_loss_eff = self._de3_v4_profit_gate_cat_min_p_loss
                        cat_max_p_std_eff = self._de3_v4_profit_gate_cat_max_p_loss_std
                        cat_min_p_loss_lcb_eff = self._de3_v4_profit_gate_cat_min_p_loss_lcb
                        cat_max_edge_points_eff = self._de3_v4_profit_gate_cat_max_edge_points
                        cat_max_final_score_eff = self._de3_v4_profit_gate_cat_max_final_score
                        cat_min_trigger_count_eff = int(self._de3_v4_profit_gate_cat_min_trigger_count or 1)
                        gate_lane = self._de3_v4_lane_from_cand_type(cand_type)
                        gate_session = str(engine_session or "").strip().lower()
                        if self._de3_v4_runtime and self._de3_v4_profit_gate_enabled:
                            gate_overrides = self._de3_v4_effective_profit_gate_cfg(
                                lane=gate_lane,
                                session_name=gate_session,
                            )
                            if isinstance(gate_overrides, dict) and gate_overrides:
                                try:
                                    if gate_overrides.get("min_samples") is not None:
                                        policy_min_samples_eff = int(gate_overrides.get("min_samples"))
                                except Exception:
                                    pass
                                try:
                                    if gate_overrides.get("max_p_loss_std") is not None:
                                        policy_max_std_eff = float(gate_overrides.get("max_p_loss_std"))
                                except Exception:
                                    pass
                                try:
                                    if gate_overrides.get("min_ev_lcb_points") is not None:
                                        policy_min_ev_lcb_eff = float(gate_overrides.get("min_ev_lcb_points"))
                                except Exception:
                                    pass
                                try:
                                    if gate_overrides.get("min_ev_mean_points") is not None:
                                        policy_min_ev_mean_eff = float(gate_overrides.get("min_ev_mean_points"))
                                except Exception:
                                    pass
                                try:
                                    if gate_overrides.get("soft_pass_risk_mult_cap") is not None:
                                        soft_pass_risk_mult_cap_eff = float(gate_overrides.get("soft_pass_risk_mult_cap"))
                                except Exception:
                                    pass
                                cat_override = (
                                    gate_overrides.get("catastrophic", {})
                                    if isinstance(gate_overrides.get("catastrophic", {}), dict)
                                    else {}
                                )
                                if isinstance(cat_override, dict) and cat_override:
                                    if cat_override.get("enabled") is not None:
                                        cat_enabled_eff = bool(cat_override.get("enabled"))
                                    try:
                                        if cat_override.get("min_samples") is not None:
                                            cat_min_samples_eff = int(cat_override.get("min_samples"))
                                    except Exception:
                                        pass
                                    try:
                                        if cat_override.get("max_ev_lcb_points") is not None:
                                            cat_max_ev_lcb_eff = float(cat_override.get("max_ev_lcb_points"))
                                    except Exception:
                                        pass
                                    try:
                                        if cat_override.get("min_p_loss") is not None:
                                            cat_min_p_loss_eff = float(cat_override.get("min_p_loss"))
                                    except Exception:
                                        pass
                                    try:
                                        if cat_override.get("max_p_loss_std") is not None:
                                            cat_max_p_std_eff = float(cat_override.get("max_p_loss_std"))
                                    except Exception:
                                        pass
                                    try:
                                        if cat_override.get("min_p_loss_lcb") is not None:
                                            cat_min_p_loss_lcb_eff = float(cat_override.get("min_p_loss_lcb"))
                                    except Exception:
                                        pass
                                    try:
                                        if cat_override.get("max_edge_points") is not None:
                                            cat_max_edge_points_eff = float(cat_override.get("max_edge_points"))
                                    except Exception:
                                        pass
                                    try:
                                        if cat_override.get("max_final_score") is not None:
                                            cat_max_final_score_eff = float(cat_override.get("max_final_score"))
                                    except Exception:
                                        pass
                                    try:
                                        if cat_override.get("min_trigger_count") is not None:
                                            cat_min_trigger_count_eff = int(cat_override.get("min_trigger_count"))
                                    except Exception:
                                        pass
                                    if cat_min_trigger_count_eff <= 0:
                                        cat_min_trigger_count_eff = 1

                        self.veto_stats["checked"] += 1
                        self.veto_stats["policy_checked"] += 1
                        policy_eval = self._veto.evaluate_candidate_ev(
                            bucket_keys,
                            feat_vec,
                            tp_points=tp_points,
                            sl_points=sl_points,
                            uncertainty_z=policy_uncertainty_z,
                            min_samples=policy_min_samples_eff,
                            max_p_loss_std=policy_max_std_eff,
                            blend_empirical=policy_blend_empirical,
                            prior_strength=policy_prior_strength,
                            min_ev_lcb_points=policy_min_ev_lcb_eff,
                            min_ev_mean_points=policy_min_ev_mean_eff,
                        )
                        if policy_eval is not None:
                            try:
                                ev_scale = float((policy_cfg.get("risk", {}) or {}).get("ev_lcb_scale_points", 6.0) or 6.0)
                            except Exception:
                                ev_scale = 6.0
                            ev_scale = max(1e-6, ev_scale)
                            try:
                                sample_target = int(policy_cfg.get("confidence_sample_target", 400) or 400)
                            except Exception:
                                sample_target = 400
                            sample_target = max(2, sample_target)
                            try:
                                p_std_ref = float(policy_cfg.get("confidence_std_ref", 0.20) or 0.20)
                            except Exception:
                                p_std_ref = 0.20
                            p_std_ref = max(1e-6, p_std_ref)

                            ev_lcb = float(policy_eval.get("ev_lcb_points", 0.0) or 0.0)
                            p_std = max(0.0, float(policy_eval.get("p_loss_std", 0.0) or 0.0))
                            n_samp = int(policy_eval.get("n_samples", 0) or 0)
                            ev_conf = self._clip01(0.5 + (0.5 * np.tanh(ev_lcb / ev_scale)))
                            sample_conf = self._clip01(np.log1p(max(0, n_samp)) / np.log1p(sample_target))
                            std_conf = 1.0 - self._clip01(p_std / p_std_ref)
                            confidence = self._clip01((0.55 * ev_conf) + (0.25 * sample_conf) + (0.20 * std_conf))
                            policy_eval["confidence"] = float(confidence)

                            risk_mult = self._compute_policy_risk_mult(policy_eval, policy_cfg)
                            policy_eval["risk_mult"] = float(risk_mult)

                            cand["de3_policy_allow"] = bool(policy_eval.get("allow", False))
                            cand["de3_policy_reason"] = str(policy_eval.get("reason", "") or "")
                            cand["de3_policy_ev_points"] = float(policy_eval.get("ev_points", 0.0) or 0.0)
                            cand["de3_policy_ev_lcb_points"] = float(policy_eval.get("ev_lcb_points", 0.0) or 0.0)
                            cand["de3_policy_ev_ucb_points"] = float(policy_eval.get("ev_ucb_points", 0.0) or 0.0)
                            cand["de3_policy_p_loss"] = float(policy_eval.get("p_loss", 0.5) or 0.5)
                            cand["de3_policy_p_loss_std"] = float(policy_eval.get("p_loss_std", 0.0) or 0.0)
                            cand["de3_policy_p_loss_lcb"] = float(policy_eval.get("p_loss_lcb", 0.0) or 0.0)
                            cand["de3_policy_p_loss_ucb"] = float(policy_eval.get("p_loss_ucb", 0.0) or 0.0)
                            cand["de3_policy_confidence"] = float(policy_eval.get("confidence", 0.0) or 0.0)
                            cand["de3_policy_risk_mult"] = float(policy_eval.get("risk_mult", 1.0) or 1.0)
                            cand["de3_policy_bucket"] = str(policy_eval.get("bucket_key", "") or "")
                            cand["de3_policy_bucket_samples"] = int(policy_eval.get("n_samples", 0) or 0)
                            cand["de3_policy_level"] = str(policy_eval.get("level", "") or "")
                            cand["de3_policy_would_block"] = bool(not policy_eval.get("allow", False))
                            if self._de3_v4_runtime and self._de3_v4_profit_gate_enabled:
                                cand["de3_v4_profit_gate_lane"] = str(gate_lane)
                                cand["de3_v4_profit_gate_session"] = str(gate_session)
                                cand["de3_v4_profit_gate_min_samples_eff"] = int(policy_min_samples_eff)
                                cand["de3_v4_profit_gate_max_p_loss_std_eff"] = (
                                    float(policy_max_std_eff) if policy_max_std_eff is not None else np.nan
                                )
                                cand["de3_v4_profit_gate_min_ev_lcb_points_eff"] = float(policy_min_ev_lcb_eff)
                                cand["de3_v4_profit_gate_min_ev_mean_points_eff"] = (
                                    float(policy_min_ev_mean_eff) if policy_min_ev_mean_eff is not None else np.nan
                                )

                            # Compatibility with existing DE3 veto counterfactual wiring.
                            cand["de3_veto_p_loss"] = float(policy_eval.get("p_loss", 0.5) or 0.5)
                            cand["de3_veto_p_loss_std"] = float(policy_eval.get("p_loss_std", 0.0) or 0.0)
                            cand["de3_veto_p_loss_lcb"] = float(policy_eval.get("p_loss_lcb", 0.0) or 0.0)
                            cand["de3_veto_threshold"] = float(policy_min_ev_lcb_eff)
                            cand["de3_veto_bucket"] = str(policy_eval.get("bucket_key", "") or "")
                            cand["de3_veto_bucket_samples"] = int(policy_eval.get("n_samples", 0) or 0)
                            cand["de3_veto_level"] = str(policy_eval.get("level", "") or "")
                            cand["de3_veto_would_block"] = bool(not policy_eval.get("allow", False))
                            if self._de3_v4_runtime:
                                cand["de3_v4_profit_gate_soft_pass"] = False
                                cand["de3_v4_profit_gate_catastrophic_block"] = False
                                cand["de3_v4_profit_gate_catastrophic_reason"] = ""

                            if not bool(policy_eval.get("allow", False)):
                                catastrophic_block = False
                                if self._de3_v4_runtime and self._de3_v4_profit_gate_enabled and cat_enabled_eff:
                                    ev_lcb_val = float(policy_eval.get("ev_lcb_points", 0.0) or 0.0)
                                    p_loss_val = float(policy_eval.get("p_loss", 0.5) or 0.5)
                                    p_std_val = float(policy_eval.get("p_loss_std", 0.0) or 0.0)
                                    p_loss_lcb_val = float(policy_eval.get("p_loss_lcb", 0.0) or 0.0)
                                    n_samp_val = int(policy_eval.get("n_samples", 0) or 0)
                                    try:
                                        edge_points_val_raw = cand.get(
                                            "de3_edge_points",
                                            cand.get("edge_points", np.nan),
                                        )
                                        edge_points_val = float(edge_points_val_raw)
                                        if not np.isfinite(edge_points_val):
                                            edge_points_val = np.nan
                                    except Exception:
                                        edge_points_val = np.nan
                                    try:
                                        final_score_val_raw = cand.get("final_score", cand_final_score)
                                        final_score_val = float(final_score_val_raw)
                                        if not np.isfinite(final_score_val):
                                            final_score_val = np.nan
                                    except Exception:
                                        final_score_val = np.nan
                                    catastrophic_reasons: List[str] = []
                                    trigger_count = 0
                                    if (
                                        cat_max_ev_lcb_eff is not None
                                        and ev_lcb_val <= float(cat_max_ev_lcb_eff)
                                    ):
                                        trigger_count += 1
                                        catastrophic_reasons.append("ev_lcb")
                                    if (
                                        cat_min_p_loss_eff is not None
                                        and p_loss_val >= float(cat_min_p_loss_eff)
                                    ):
                                        trigger_count += 1
                                        catastrophic_reasons.append("p_loss")
                                    if (
                                        cat_max_p_std_eff is not None
                                        and p_std_val <= float(cat_max_p_std_eff)
                                    ):
                                        trigger_count += 1
                                        catastrophic_reasons.append("p_loss_std")
                                    if (
                                        cat_min_p_loss_lcb_eff is not None
                                        and p_loss_lcb_val >= float(cat_min_p_loss_lcb_eff)
                                    ):
                                        trigger_count += 1
                                        catastrophic_reasons.append("p_loss_lcb")
                                    if (
                                        cat_max_edge_points_eff is not None
                                        and np.isfinite(edge_points_val)
                                        and edge_points_val <= float(cat_max_edge_points_eff)
                                    ):
                                        trigger_count += 1
                                        catastrophic_reasons.append("edge_points")
                                    if (
                                        cat_max_final_score_eff is not None
                                        and np.isfinite(final_score_val)
                                        and final_score_val <= float(cat_max_final_score_eff)
                                    ):
                                        trigger_count += 1
                                        catastrophic_reasons.append("final_score")
                                    catastrophic_block = bool(
                                        n_samp_val >= int(cat_min_samples_eff or 0)
                                        and trigger_count >= int(cat_min_trigger_count_eff or 1)
                                    )
                                    cand["de3_v4_profit_gate_catastrophic_block"] = bool(catastrophic_block)
                                    cand["de3_v4_profit_gate_catastrophic_reason"] = (
                                        "|".join(catastrophic_reasons) if catastrophic_block else ""
                                    )
                                if policy_shadow:
                                    self.veto_stats["policy_shadow_would_block"] += 1
                                elif (
                                    self._de3_v4_runtime
                                    and self._de3_v4_profit_gate_enabled
                                    and self._de3_v4_profit_gate_soft_pass_non_catastrophic
                                    and (not catastrophic_block)
                                ):
                                    # Profit-gate v2 soft pass: keep candidate reachable while
                                    # tagging would-block reason and capping risk.
                                    self.veto_stats["policy_shadow_would_block"] += 1
                                    cand["de3_v4_profit_gate_soft_pass"] = True
                                    cand["de3_policy_allow"] = True
                                    cand["de3_policy_reason"] = str(
                                        f"{policy_eval.get('reason', 'blocked')}:v4_soft_pass"
                                    )
                                    if soft_pass_risk_mult_cap_eff is not None:
                                        capped_mult = float(
                                            min(
                                                float(policy_eval.get("risk_mult", 1.0) or 1.0),
                                                float(soft_pass_risk_mult_cap_eff),
                                            )
                                        )
                                        policy_eval["risk_mult"] = float(capped_mult)
                                        cand["de3_policy_risk_mult"] = float(capped_mult)
                                else:
                                    self.veto_stats["blocked"] += 1
                                    self.veto_stats["policy_blocked"] += 1
                                    block_reason = (
                                        f"adaptive {policy_eval.get('reason', 'blocked')}"
                                    )
                                if policy_log_decisions:
                                    logging.info(
                                        "DE3 adaptive policy: %s | allow=%s | ev=%.2f ev_lcb=%.2f p_loss=%.3f std=%.3f conf=%.2f bucket=%s",
                                        cand_id,
                                        bool(policy_eval.get("allow", False)),
                                        float(policy_eval.get("ev_points", 0.0) or 0.0),
                                        float(policy_eval.get("ev_lcb_points", 0.0) or 0.0),
                                        float(policy_eval.get("p_loss", 0.5) or 0.5),
                                        float(policy_eval.get("p_loss_std", 0.0) or 0.0),
                                        float(policy_eval.get("confidence", 0.0) or 0.0),
                                        str(policy_eval.get("bucket_key", "") or ""),
                                    )
                                if (block_reason is not None) and idx == 0 and policy_block_all_on_top:
                                    veto_block_all = True
                                    top_blocked_id = cand_id
                                    top_blocked_reason = block_reason
                                    break
                        else:
                            cand["de3_policy_allow"] = True
                            cand["de3_policy_reason"] = "model_unavailable"
                            cand["de3_policy_would_block"] = False
                    elif use_legacy_veto:
                        veto_limit = None if veto_threshold is None else float(veto_threshold)
                        self.veto_stats["checked"] += 1
                        veto_hit, veto_pred = self._veto.should_veto(
                            bucket_keys,
                            feat_vec,
                            threshold=veto_limit,
                            uncertainty_z=veto_uncertainty_z,
                            max_std=veto_max_std,
                            min_samples=veto_min_samples,
                        )
                        if veto_pred is not None:
                            cand["de3_veto_p_loss"] = float(veto_pred.get("p_loss", 0.0))
                            cand["de3_veto_p_loss_std"] = float(veto_pred.get("p_loss_std", 0.0))
                            cand["de3_veto_p_loss_lcb"] = float(veto_pred.get("p_loss_lcb", 0.0))
                            cand["de3_veto_threshold"] = float(veto_pred.get("threshold", self._veto.threshold))
                            cand["de3_veto_bucket"] = str(veto_pred.get("bucket_key") or "")
                            cand["de3_veto_bucket_samples"] = int(veto_pred.get("n_samples", 0) or 0)
                            cand["de3_veto_level"] = str(veto_pred.get("level", "") or "")
                            cand["de3_veto_would_block"] = bool(veto_hit)
                        if veto_hit and not shadow_mode:
                            self.veto_stats["blocked"] += 1
                            p_loss = None if veto_pred is None else float(veto_pred.get("p_loss", np.nan))
                            p_lcb = None if veto_pred is None else float(veto_pred.get("p_loss_lcb", np.nan))
                            thr = float(self._veto.threshold if veto_threshold is None else veto_threshold)
                            if p_lcb is not None and np.isfinite(p_lcb):
                                block_reason = f"veto p_loss_lcb={p_lcb:.2f}>{thr:.2f}"
                            elif p_loss is not None and np.isfinite(p_loss):
                                block_reason = f"veto p_loss={p_loss:.2f}>{thr:.2f}"
                            else:
                                block_reason = "veto"
                            if veto_cfg.get("log_decisions", False):
                                bucket_used = None if veto_pred is None else veto_pred.get("bucket_key")
                                logging.info("DE3 veto: %s | %s | bucket=%s", cand_id, block_reason, bucket_used)
                            if idx == 0 and veto_block_all_on_top:
                                veto_block_all = True
                                top_blocked_id = cand_id
                                top_blocked_reason = block_reason
                                break
                if ny_gate_active and block_reason is None:
                    try:
                        opt_wr = float(cand.get("opt_wr", 0.0) or 0.0)
                    except Exception:
                        opt_wr = 0.0
                    try:
                        final_score = float(cand.get("final_score", 0.0) or 0.0)
                    except Exception:
                        final_score = 0.0
                    if opt_wr < min_opt_wr:
                        block_reason = f"NY gate opt_wr {opt_wr:.3f} < {min_opt_wr:.3f}"
                    elif min_final_score is not None and final_score < min_final_score:
                        block_reason = f"NY gate score {final_score:.2f} < {min_final_score:.2f}"

                # Long_Mom quality filters (root-cause fixes for late breakouts / weak impulse).
                lm_cfg = CONFIG.get("DE3_LONG_MOM_FILTERS", {}) or {}
                if (
                    block_reason is None
                    and self._v1_specific_filters_enabled
                    and cand_type == "long_mom"
                    and lm_cfg.get("enabled", True)
                ):
                    df_tf = df
                    if cand_tf.startswith("5") and df_5m is not None:
                        df_tf = df_5m
                    elif cand_tf.startswith("15") and df_15m is not None:
                        df_tf = df_15m
                    ok, reason = self._check_long_mom_filters(
                        df_tf,
                        lm_cfg,
                        cand_tf,
                        cand_thresh=cand_thresh,
                        regime_norm=regime_norm,
                    )
                    if not ok and reason:
                        block_reason = reason

                if block_reason is None and entry_struct_enabled:
                    side_for_filter = str(cand.get("signal", "") or "").upper()
                    entry_struct_reason = self._entry_candle_block_reason(
                        entry_struct_features,
                        entry_struct_cfg,
                        side_for_filter,
                        cand_tf,
                    )
                    if entry_struct_reason:
                        block_reason = entry_struct_reason
                        cand["de3_entry_filter_hit"] = True
                        cand["de3_entry_filter_reason"] = entry_struct_reason
                        if entry_struct_log:
                            logging.info(
                                "DynamicEngine3 entry-structure block: %s | %s",
                                cand_id,
                                entry_struct_reason,
                            )
                    else:
                        cand["de3_entry_filter_hit"] = False

                if block_reason is None and v2_entry_bar_enabled:
                    side_for_filter = str(cand.get("signal", "") or "").upper()
                    v2_bar_reason = self._de3_v2_entry_bar_block_reason(
                        v2_entry_bar_features,
                        v2_entry_bar_cfg,
                        side_for_filter,
                        cand_tf,
                    )
                    if v2_bar_reason:
                        block_reason = v2_bar_reason
                        cand["de3_v2_entry_block_hit"] = True
                        cand["de3_v2_entry_block_reason"] = v2_bar_reason
                        if v2_entry_bar_log:
                            logging.info(
                                "DynamicEngine3 v2 entry-bar block: %s | %s",
                                cand_id,
                                v2_bar_reason,
                            )
                    else:
                        cand["de3_v2_entry_block_hit"] = False

                if block_reason is None:
                    # Evaluate runtime execution viability/economics for each candidate so we can
                    # fall back to the next ranked candidate instead of aborting DE3 on first miss.
                    df_for_viability = df
                    if cand_tf.startswith("5") and df_5m is not None and not df_5m.empty:
                        df_for_viability = df_5m
                    elif cand_tf.startswith("15") and df_15m is not None and not df_15m.empty:
                        df_for_viability = df_15m

                    fixed_ok = True
                    fixed_details = {}
                    if self._v1_specific_filters_enabled:
                        fixed_ok, fixed_details = apply_fixed_sltp(
                            {
                                "side": cand["signal"],
                                "strategy": "DynamicEngine3",
                                "sl_dist": max(float(cand.get("sl", min_sl)), min_sl),
                                "tp_dist": max(float(cand.get("tp", min_tp)), min_tp),
                            },
                            df_for_viability,
                            float(df["close"].iloc[-1]),
                            ts=current_time,
                            session=sltp_session,
                            vol_regime=vol_regime,
                        )
                    if not fixed_ok:
                        fixed_reason = str((fixed_details or {}).get("reason", "FixedSLTP blocked"))
                        block_reason = f"FixedSLTP {fixed_reason}"
                    else:
                        if fixed_details:
                            final_sl = float(fixed_details.get("sl_dist", max(float(cand["sl"]), min_sl)))
                            final_tp = float(fixed_details.get("tp_dist", max(float(cand["tp"]), min_tp)))
                        else:
                            final_sl = max(float(cand["sl"]), min_sl)
                            final_tp = max(float(cand["tp"]), min_tp)

                        # Long_Mom bracket override in high-vol: widen SL and tighten TP using ATR.
                        if (
                            self._v1_specific_filters_enabled
                            and
                            long_mom_cfg.get("enabled", False)
                            and cand_type == "long_mom"
                            and str(vol_regime or "").lower() == "high"
                        ):
                            atr_period = int(long_mom_cfg.get("atr_period", 14) or 14)
                            sl_atr = float(long_mom_cfg.get("sl_atr", 1.2) or 1.2)
                            tp_atr = float(long_mom_cfg.get("tp_atr", 0.9) or 0.9)
                            atr_val = self._compute_atr_simple(df_for_viability, atr_period)
                            if atr_val is not None:
                                final_sl = max(atr_val * sl_atr, min_sl)
                                final_tp = max(atr_val * tp_atr, min_tp)

                        if block_reason is None:
                            policy_risk_mult = 1.0
                            try:
                                policy_risk_mult = float(cand.get("de3_policy_risk_mult", 1.0) or 1.0)
                            except Exception:
                                policy_risk_mult = 1.0

                            num_contracts = base_contracts
                            if use_policy and policy_risk_enabled and policy_apply_size:
                                num_contracts = self._size_from_risk_mult(base_contracts, policy_risk_mult, policy_cfg)

                            gross_profit = final_tp * point_value * num_contracts
                            total_fees = fees_per_side * 2 * num_contracts
                            net_profit = gross_profit - total_fees
                            if enforce_min_net and net_profit < min_net_profit:
                                block_reason = (
                                    f"fees net ${net_profit:.2f} < ${float(min_net_profit):.2f}"
                                )
                            else:
                                chosen_exec = {
                                    "final_sl": final_sl,
                                    "final_tp": final_tp,
                                    "num_contracts": int(num_contracts),
                                    "policy_risk_mult": float(policy_risk_mult),
                                    "gross_profit": float(gross_profit),
                                    "total_fees": float(total_fees),
                                    "net_profit": float(net_profit),
                                }
                                sel_score, edge_points, edge_proxy_points, edge_conf = self._compute_candidate_selection_metrics(
                                    cand,
                                    final_tp=final_tp,
                                    final_sl=final_sl,
                                    fees_per_side=fees_per_side,
                                    point_value=point_value,
                                    prefer_policy_ev_lcb=prefer_policy_ev_lcb,
                                    use_policy_edge=use_policy_edge_in_ranking,
                                )
                                cand["de3_selection_score"] = float(sel_score)
                                cand["de3_edge_points"] = float(edge_points)
                                cand["de3_edge_proxy_points"] = float(edge_proxy_points)
                                cand["de3_edge_confidence"] = float(edge_conf)
                                quality_metrics = self._extract_candidate_quality_metrics(cand)
                                feasible_candidates.append(
                                    {
                                        "idx": int(idx),
                                        "cand": cand,
                                        "cand_id": cand_id,
                                        "exec": chosen_exec,
                                        "selection_score": float(sel_score),
                                        "edge_points": float(edge_points),
                                        "edge_proxy_points": float(edge_proxy_points),
                                        "edge_confidence": float(edge_conf),
                                        "structural_score": quality_metrics["structural_score"],
                                        "structural_pass": quality_metrics["structural_pass"],
                                        "worst_block_avg_pnl": quality_metrics["worst_block_avg_pnl"],
                                        "worst_block_pf": quality_metrics["worst_block_pf"],
                                        "profitable_block_ratio": quality_metrics["profitable_block_ratio"],
                                        "stop_like_share": quality_metrics["stop_like_share"],
                                        "loss_share": quality_metrics["loss_share"],
                                    }
                                )

                if block_reason:
                    if idx == 0:
                        top_blocked_id = cand_id
                        top_blocked_reason = block_reason
                    continue

            if veto_block_all:
                self._warn(
                    "🚫 DynamicEngine3: context policy blocked trade (%s)",
                    top_blocked_reason or "context",
                )
                return None

            if feasible_candidates:
                if selection_enabled:
                    if self._de3_v4_runtime and self._de3_v4_runtime_module is not None:
                        context_inputs = self._de3_v3_context_inputs(
                            current_time=current_time,
                            engine_session=engine_session,
                            vol_regime=vol_regime,
                            atr_5m=atr_5m,
                            atr_med=atr_med,
                            atr_ratio=atr_ratio,
                            vwap_dist_atr=vwap_dist_atr,
                            price_loc=price_loc,
                            rvol_ratio=rvol_ratio,
                        )
                        v4_result = self._de3_v4_runtime_module.select_route_and_variant(
                            feasible_candidates=feasible_candidates,
                            default_session=str(engine_session or ""),
                            context_inputs=context_inputs,
                            current_time=current_time,
                        )
                        if bool(v4_result.get("abstained", False)):
                            chosen_decision_id = self._emit_de3_decision_journal(
                                current_time=current_time,
                                session_name=str(engine_session or ""),
                                feasible_candidates=feasible_candidates,
                                export_candidates=(
                                    v4_result.get("decision_export_rows")
                                    if isinstance(v4_result.get("decision_export_rows"), list)
                                    else (
                                        v4_result.get("feasible_rows")
                                        if isinstance(v4_result.get("feasible_rows"), list)
                                        else feasible_candidates
                                    )
                                ),
                                chosen_cand_id=None,
                                abstained=True,
                                abstain_reason=str(v4_result.get("abstain_reason", "v4_abstain") or "v4_abstain"),
                            )
                            if self._log_selection_details:
                                logging.info(
                                    "DE3 v4 abstain: %s (route=%s conf=%.4f)",
                                    str(v4_result.get("abstain_reason", "")),
                                    str(v4_result.get("route_decision", "")),
                                    float(v4_result.get("route_confidence", 0.0) or 0.0),
                                )
                            return None
                        chosen_entry = (
                            v4_result.get("chosen_entry")
                            if isinstance(v4_result.get("chosen_entry"), dict)
                            else None
                        )
                        if chosen_entry is None:
                            chosen_decision_id = self._emit_de3_decision_journal(
                                current_time=current_time,
                                session_name=str(engine_session or ""),
                                feasible_candidates=feasible_candidates,
                                export_candidates=(
                                    v4_result.get("decision_export_rows")
                                    if isinstance(v4_result.get("decision_export_rows"), list)
                                    else (
                                        v4_result.get("feasible_rows")
                                        if isinstance(v4_result.get("feasible_rows"), list)
                                        else feasible_candidates
                                    )
                                ),
                                chosen_cand_id=None,
                                abstained=True,
                                abstain_reason="v4_missing_chosen_entry",
                            )
                            if self._log_selection_details:
                                logging.info("DE3 v4 abstain: missing chosen entry")
                            return None
                        chosen_family_id = str(v4_result.get("chosen_family_id", "") or "")
                        bracket_result = (
                            v4_result.get("bracket_result")
                            if isinstance(v4_result.get("bracket_result"), dict)
                            else {}
                        )
                        # Apply bracket expression chosen by the v4 bracket module.
                        chosen_exec = (
                            chosen_entry.get("exec")
                            if isinstance(chosen_entry.get("exec"), dict)
                            else {}
                        )
                        if isinstance(chosen_exec, dict):
                            selected_sl = float(bracket_result.get("selected_sl", chosen_exec.get("final_sl", 0.0)) or 0.0)
                            selected_tp = float(bracket_result.get("selected_tp", chosen_exec.get("final_tp", 0.0)) or 0.0)
                            if selected_sl > 0.0 and selected_tp > 0.0:
                                chosen_exec["final_sl"] = float(selected_sl)
                                chosen_exec["final_tp"] = float(selected_tp)
                                num_contracts_eff = int(chosen_exec.get("num_contracts", 1) or 1)
                                gross_profit_eff = float(selected_tp * point_value * num_contracts_eff)
                                total_fees_eff = float(fees_per_side * 2 * num_contracts_eff)
                                chosen_exec["gross_profit"] = float(gross_profit_eff)
                                chosen_exec["total_fees"] = float(total_fees_eff)
                                chosen_exec["net_profit"] = float(gross_profit_eff - total_fees_eff)
                            chosen_entry["exec"] = chosen_exec

                        chosen_entry["de3_v4_route_decision"] = str(v4_result.get("route_decision", ""))
                        chosen_entry["de3_v4_route_confidence"] = float(v4_result.get("route_confidence", 0.0) or 0.0)
                        chosen_entry["de3_v4_selected_lane"] = str(v4_result.get("selected_lane", ""))
                        chosen_entry["de3_v4_selected_variant_id"] = str(v4_result.get("selected_variant_id", ""))
                        chosen_entry["de3_v4_lane_candidate_count"] = int(v4_result.get("lane_candidate_count", 0) or 0)
                        chosen_entry["de3_v4_lane_selection_reason"] = str(v4_result.get("lane_selection_reason", ""))
                        chosen_entry["de3_v4_bracket_mode"] = str(bracket_result.get("bracket_mode", "canonical"))
                        chosen_entry["de3_v4_selected_sl"] = float(bracket_result.get("selected_sl", 0.0) or 0.0)
                        chosen_entry["de3_v4_selected_tp"] = float(bracket_result.get("selected_tp", 0.0) or 0.0)
                        chosen_entry["de3_v4_canonical_default_used"] = bool(
                            bracket_result.get("canonical_default_used", True)
                        )
                        chosen_entry["de3_v4_runtime_mode"] = str(
                            (self._de3_v4_status or {}).get("runtime_mode", "")
                        )
                        chosen_entry["de3_v4_core_anchor_family_ids"] = list(
                            (self._de3_v4_status or {}).get("core_anchor_family_ids", [])
                        )
                        chosen_entry["de3_v4_route_scores"] = dict(
                            v4_result.get("route_scores", {})
                            if isinstance(v4_result.get("route_scores", {}), dict)
                            else {}
                        )
                        chosen_cand = chosen_entry.get("cand") if isinstance(chosen_entry.get("cand"), dict) else {}
                        if isinstance(chosen_cand, dict):
                            chosen_cand["de3_v4_route_decision"] = chosen_entry.get("de3_v4_route_decision")
                            chosen_cand["de3_v4_route_confidence"] = chosen_entry.get("de3_v4_route_confidence")
                            chosen_cand["de3_v4_selected_lane"] = chosen_entry.get("de3_v4_selected_lane")
                            chosen_cand["de3_v4_selected_variant_id"] = chosen_entry.get("de3_v4_selected_variant_id")
                            chosen_cand["de3_v4_lane_candidate_count"] = chosen_entry.get("de3_v4_lane_candidate_count")
                            chosen_cand["de3_v4_lane_selection_reason"] = chosen_entry.get("de3_v4_lane_selection_reason")
                            chosen_cand["de3_v4_bracket_mode"] = chosen_entry.get("de3_v4_bracket_mode")
                            chosen_cand["de3_v4_selected_sl"] = chosen_entry.get("de3_v4_selected_sl")
                            chosen_cand["de3_v4_selected_tp"] = chosen_entry.get("de3_v4_selected_tp")
                            chosen_cand["de3_v4_canonical_default_used"] = chosen_entry.get("de3_v4_canonical_default_used")
                            chosen_cand["de3_v4_runtime_mode"] = chosen_entry.get("de3_v4_runtime_mode")
                            chosen_cand["de3_v4_route_scores"] = chosen_entry.get("de3_v4_route_scores")
                            chosen_cand["de3_v4_execution_policy_tier"] = chosen_entry.get("de3_v4_execution_policy_tier")
                            chosen_cand["de3_v4_execution_quality_score"] = chosen_entry.get("de3_v4_execution_quality_score")
                            chosen_cand["de3_v4_execution_policy_reason"] = chosen_entry.get("de3_v4_execution_policy_reason")
                            chosen_cand["de3_v4_execution_policy_source"] = chosen_entry.get("de3_v4_execution_policy_source")
                            chosen_cand["de3_v4_execution_policy_enforce_veto"] = chosen_entry.get("de3_v4_execution_policy_enforce_veto")
                            chosen_cand["de3_v4_execution_policy_soft_pass"] = chosen_entry.get("de3_v4_execution_policy_soft_pass")
                            chosen_cand["de3_v4_execution_policy_hard_limit_triggered"] = chosen_entry.get("de3_v4_execution_policy_hard_limit_triggered")
                            chosen_cand["de3_v4_execution_policy_hard_limit_reason"] = chosen_entry.get("de3_v4_execution_policy_hard_limit_reason")
                            chosen_cand["de3_v4_execution_policy_components"] = chosen_entry.get("de3_v4_execution_policy_components")
                            chosen_cand["de3_v4_entry_model_enabled"] = chosen_entry.get("de3_v4_entry_model_enabled")
                            chosen_cand["de3_v4_entry_model_allow"] = chosen_entry.get("de3_v4_entry_model_allow")
                            chosen_cand["de3_v4_entry_model_tier"] = chosen_entry.get("de3_v4_entry_model_tier")
                            chosen_cand["de3_v4_entry_model_reason"] = chosen_entry.get("de3_v4_entry_model_reason")
                            chosen_cand["de3_v4_entry_model_score"] = chosen_entry.get("de3_v4_entry_model_score")
                            chosen_cand["de3_v4_entry_model_threshold"] = chosen_entry.get("de3_v4_entry_model_threshold")
                            chosen_cand["de3_v4_entry_model_threshold_base"] = chosen_entry.get("de3_v4_entry_model_threshold_base")
                            chosen_cand["de3_v4_entry_model_threshold_scope_offset"] = chosen_entry.get("de3_v4_entry_model_threshold_scope_offset")
                            chosen_cand["de3_v4_entry_model_scope"] = chosen_entry.get("de3_v4_entry_model_scope")
                            chosen_cand["de3_v4_entry_model_stats"] = chosen_entry.get("de3_v4_entry_model_stats")
                            chosen_cand["de3_v4_entry_model_components"] = chosen_entry.get("de3_v4_entry_model_components")
                            chosen_entry["cand"] = chosen_cand
                        prune_reason = self._de3_v4_pre_entry_prune_reason(
                            chosen_entry,
                            current_time=current_time,
                            engine_session=str(engine_session or ""),
                        )
                        if prune_reason:
                            chosen_decision_id = self._emit_de3_decision_journal(
                                current_time=current_time,
                                session_name=str(engine_session or ""),
                                feasible_candidates=feasible_candidates,
                                export_candidates=(
                                    v4_result.get("decision_export_rows")
                                    if isinstance(v4_result.get("decision_export_rows"), list)
                                    else (
                                        v4_result.get("feasible_rows")
                                        if isinstance(v4_result.get("feasible_rows"), list)
                                        else feasible_candidates
                                    )
                                ),
                                chosen_cand_id=None,
                                abstained=True,
                                abstain_reason=prune_reason,
                            )
                            if self._de3_v4_prune_log_blocks or self._log_selection_details:
                                logging.info(
                                    "DE3 v4 prune abstain: %s | candidate=%s",
                                    prune_reason,
                                    str(chosen_entry.get("cand_id", "") or ""),
                                )
                            return None
                        chosen_decision_id = self._emit_de3_decision_journal(
                            current_time=current_time,
                            session_name=str(engine_session or ""),
                            feasible_candidates=feasible_candidates,
                            export_candidates=(
                                v4_result.get("decision_export_rows")
                                if isinstance(v4_result.get("decision_export_rows"), list)
                                else (
                                    v4_result.get("feasible_rows")
                                    if isinstance(v4_result.get("feasible_rows"), list)
                                    else feasible_candidates
                                )
                            ),
                            chosen_cand_id=str(chosen_entry.get("cand_id", "") or ""),
                            abstained=False,
                            abstain_reason="",
                        )
                    elif self._de3_v3_runtime and self._de3_v3_family_runtime is not None:
                        context_inputs = self._de3_v3_context_inputs(
                            current_time=current_time,
                            engine_session=engine_session,
                            vol_regime=vol_regime,
                            atr_5m=atr_5m,
                            atr_med=atr_med,
                            atr_ratio=atr_ratio,
                            vwap_dist_atr=vwap_dist_atr,
                            price_loc=price_loc,
                            rvol_ratio=rvol_ratio,
                        )
                        family_result = self._de3_v3_family_runtime.select_family_and_member(
                            feasible_candidates=feasible_candidates,
                            default_session=engine_session,
                            context_inputs=context_inputs,
                        )
                        family_rows = (
                            family_result.get("feasible_family_rows")
                            if isinstance(family_result.get("feasible_family_rows"), list)
                            else []
                        )
                        chosen_family_id = str(family_result.get("chosen_family_id", "") or "")
                        chosen_member_local_score = family_result.get("chosen_member_local_score")
                        if self._de3_v3_family_runtime.log_decisions and family_rows:
                            top_k = min(self._de3_v3_family_runtime.log_top_k, len(family_rows))
                            for i, row in enumerate(family_rows[:top_k], 1):
                                components = (
                                    row.get("family_score_components")
                                    if isinstance(row.get("family_score_components"), dict)
                                    else {}
                                )
                                logging.info(
                                    (
                                        "DE3 v3 family rank #%s %s | score=%.3f base=%.3f div=%.3f recent=%.3f ctx_exp=%.3f "
                                        "ctx_conf=%.3f prior=%.3f usability=%.3f adaptive=%.3f "
                                        "support=%.3f ctx_tier=%s ev_tier=%s trusted=%s fallback=%s status=%s margin=%s cap=%s "
                                        "eligible=%s feasible_members=%s chosen=%s"
                                    ),
                                    i,
                                    str(row.get("family_id", "")),
                                    float(row.get("family_score", 0.0) or 0.0),
                                    float(row.get("base_family_score", 0.0) or 0.0),
                                    float(row.get("diversity_adjustment", 0.0) or 0.0),
                                    float(row.get("recent_chosen_share", 0.0) or 0.0),
                                    float(
                                        components.get(
                                            "context_profile_expectancy_component",
                                            components.get("context_ev_component", 0.0),
                                        )
                                        or 0.0
                                    ),
                                    float(
                                        components.get(
                                        "context_profile_confidence_component",
                                        components.get("confidence_component", 0.0),
                                    )
                                    or 0.0
                                ),
                                float(components.get("family_prior_component", 0.0) or 0.0),
                                float(components.get("v3_realized_usability_component", 0.0) or 0.0),
                                float(components.get("adaptive_policy_component", 0.0) or 0.0),
                                float(components.get("context_support_ratio", 0.0) or 0.0),
                                str(components.get("context_support_tier", row.get("context_support_tier", "low"))),
                                str(components.get("evidence_support_tier", row.get("family_evidence_support_tier", "none"))),
                                bool(components.get("trusted_context_used", row.get("profile_trusted", False))),
                                bool(components.get("fallback_to_priors", row.get("profile_fallback", True))),
                                str(row.get("family_competition_status", "competitive")),
                                bool(row.get("competition_margin_qualified", False)),
                                bool(row.get("context_advantage_capped", False)),
                                bool(row.get("competition_eligible", False)),
                                int(row.get("feasible_member_count", 0) or 0),
                                bool(i == 1),
                            )
                        chosen_entry = (
                            family_result.get("chosen_entry")
                            if isinstance(family_result.get("chosen_entry"), dict)
                            else None
                        )
                        if chosen_entry is None:
                            abstain_reason = str(
                                family_result.get("abstain_reason", "no_family_selected")
                                or "no_family_selected"
                            )
                            chosen_decision_id = self._emit_de3_v3_family_decision_journal(
                                current_time=current_time,
                                session_name=str(engine_session or ""),
                                family_rows=family_rows,
                                chosen_family_id=chosen_family_id,
                                chosen_entry=None,
                                chosen_member_local_score=chosen_member_local_score,
                                context_inputs=context_inputs,
                                abstained=True,
                                abstain_reason=abstain_reason,
                            )
                            if self._de3_v3_family_runtime.log_decisions or self._log_selection_details:
                                logging.info("DE3 v3 abstain: %s", abstain_reason)
                            return None

                        chosen_family_row = (
                            family_result.get("chosen_family_row")
                            if isinstance(family_result.get("chosen_family_row"), dict)
                            else None
                        )
                        if chosen_family_row is None:
                            for row in family_rows:
                                if str(row.get("family_id", "")) == chosen_family_id:
                                    chosen_family_row = row
                                    break
                        chosen_family_score = 0.0
                        chosen_family_components: Dict[str, float] = {}
                        if isinstance(chosen_family_row, dict):
                            chosen_family_score = float(chosen_family_row.get("family_score", 0.0) or 0.0)
                            comps = (
                                chosen_family_row.get("family_score_components")
                                if isinstance(chosen_family_row.get("family_score_components"), dict)
                                else {}
                            )
                            score_obj = (
                                chosen_family_row.get("family_score_object")
                                if isinstance(chosen_family_row.get("family_score_object"), dict)
                                else {}
                            )
                            chosen_family_components = {
                                "context_profile_expectancy_component": float(
                                    comps.get("context_profile_expectancy_component", comps.get("context_ev_component", 0.0))
                                    or 0.0
                                ),
                                "context_profile_confidence_component": float(
                                    comps.get("context_profile_confidence_component", comps.get("confidence_component", 0.0))
                                    or 0.0
                                ),
                                "family_prior_component": float(comps.get("family_prior_component", 0.0) or 0.0),
                                "adaptive_policy_component": float(comps.get("adaptive_policy_component", 0.0) or 0.0),
                                "v3_realized_usability_component": float(
                                    comps.get("v3_realized_usability_component", 0.0) or 0.0
                                ),
                                "context_support_ratio": float(comps.get("context_support_ratio", 0.0) or 0.0),
                                "base_family_score": float(
                                    score_obj.get("pre_adjustment_score", comps.get("base_family_score", chosen_family_score))
                                    or chosen_family_score
                                ),
                                "diversity_adjustment": float(
                                    score_obj.get(
                                        "competition_diversity_adjustment",
                                        comps.get("competition_diversity_adjustment", comps.get("diversity_adjustment", 0.0)),
                                    )
                                    or 0.0
                                ),
                                "final_family_score": float(
                                    score_obj.get("final_family_score", comps.get("final_family_score", chosen_family_score))
                                    or chosen_family_score
                                ),
                                "recent_chosen_share": float(comps.get("recent_chosen_share", 0.0) or 0.0),
                            }
                            chosen_entry["family_rank"] = 1
                            chosen_entry["family_id"] = chosen_family_id
                            chosen_entry["chosen_family_id"] = chosen_family_id
                            chosen_entry["family_score"] = chosen_family_score
                            chosen_entry["family_context_ev"] = chosen_family_components.get(
                                "context_profile_expectancy_component"
                            )
                            chosen_entry["family_confidence"] = chosen_family_components.get(
                                "context_profile_confidence_component"
                            )
                            chosen_entry["family_prior"] = chosen_family_components.get("family_prior_component")
                            chosen_entry["family_profile"] = chosen_family_components.get(
                                "context_profile_expectancy_component"
                            )
                            chosen_entry["family_context_support_ratio"] = chosen_family_components.get(
                                "context_support_ratio"
                            )
                            chosen_entry["family_context_support_tier"] = comps.get(
                                "context_support_tier", chosen_family_row.get("context_support_tier")
                            )
                            chosen_entry["family_local_support_tier"] = chosen_family_row.get(
                                "family_local_support_tier",
                                chosen_family_row.get("context_support_tier"),
                            )
                            chosen_entry["family_context_sample_count"] = comps.get(
                                "context_sample_count", chosen_family_row.get("context_sample_count")
                            )
                            chosen_entry["family_context_weight"] = comps.get(
                                "context_profile_weight", chosen_family_row.get("context_profile_weight")
                            )
                            chosen_entry["family_context_trusted"] = comps.get(
                                "trusted_context_used", chosen_family_row.get("profile_trusted")
                            )
                            chosen_entry["family_context_fallback_priors"] = comps.get(
                                "fallback_to_priors", chosen_family_row.get("profile_fallback")
                            )
                            chosen_entry["family_active_context_buckets"] = dict(
                                chosen_family_row.get("active_context_buckets") or {}
                            )
                            chosen_entry["family_profile_used"] = bool(
                                chosen_family_row.get("profile_used", False)
                            )
                            chosen_entry["family_profile_fallback"] = bool(chosen_family_row.get("profile_fallback", True))
                            chosen_entry["family_usability_state"] = str(
                                chosen_family_row.get("family_usability_state", "low_support")
                            )
                            chosen_entry["family_usability_component"] = float(
                                chosen_family_row.get("family_usability_component", 0.0) or 0.0
                            )
                            chosen_entry["family_evidence_support_tier"] = str(
                                chosen_family_row.get("family_evidence_support_tier", "none")
                            )
                            chosen_entry["family_competition_status"] = str(
                                chosen_family_row.get("family_competition_status", "competitive")
                            )
                            chosen_entry["family_usability_adjustment"] = float(
                                chosen_family_row.get("family_usability_adjustment", 0.0) or 0.0
                            )
                            chosen_entry["family_suppression_reason"] = str(
                                chosen_family_row.get("family_suppression_reason", "") or ""
                            )
                            chosen_entry["base_family_score"] = float(
                                chosen_family_row.get("base_family_score", chosen_family_components.get("base_family_score", chosen_family_score))
                                or chosen_family_score
                            )
                            chosen_entry["diversity_adjustment"] = float(
                                chosen_family_row.get("diversity_adjustment", chosen_family_components.get("diversity_adjustment", 0.0))
                                or 0.0
                            )
                            chosen_entry["competition_diversity_adjustment"] = float(chosen_entry["diversity_adjustment"])
                            if isinstance(score_obj, dict):
                                chosen_entry["competition_diversity_adjustment"] = float(
                                    score_obj.get(
                                        "competition_diversity_adjustment",
                                        chosen_entry.get("competition_diversity_adjustment", 0.0),
                                    )
                                    or 0.0
                                )
                            chosen_entry["final_family_score"] = float(
                                chosen_family_row.get("family_score", chosen_family_components.get("final_family_score", chosen_family_score))
                                or chosen_family_score
                            )
                            chosen_entry["recent_chosen_share"] = float(
                                chosen_family_row.get("recent_chosen_share", chosen_family_components.get("recent_chosen_share", 0.0))
                                or 0.0
                            )
                            chosen_entry["exploration_bonus"] = float(chosen_family_row.get("exploration_bonus", 0.0) or 0.0)
                            chosen_entry["dominance_penalty"] = float(chosen_family_row.get("dominance_penalty", 0.0) or 0.0)
                            chosen_entry["exploration_bonus_applied"] = bool(chosen_family_row.get("exploration_bonus_applied", False))
                            chosen_entry["dominance_penalty_applied"] = bool(chosen_family_row.get("dominance_penalty_applied", False))
                            chosen_entry["competition_margin_qualified"] = bool(chosen_family_row.get("competition_margin_qualified", False))
                            chosen_entry["context_advantage_capped"] = bool(chosen_family_row.get("context_advantage_capped", False))
                            chosen_entry["context_advantage_cap_delta"] = float(chosen_family_row.get("context_advantage_cap_delta", 0.0) or 0.0)
                            chosen_entry["close_competition_decision"] = bool(chosen_family_row.get("close_competition_decision", False))
                            chosen_entry["bootstrap_competition_used_decision"] = bool(
                                chosen_family_row.get("bootstrap_competition_used_decision", False)
                            )
                            chosen_entry["family_monopoly_active"] = bool(chosen_family_row.get("family_monopoly_active", False))
                            chosen_entry["family_monopoly_top_share"] = float(chosen_family_row.get("family_monopoly_top_share", 0.0) or 0.0)
                            chosen_entry["family_monopoly_top_family_id"] = str(chosen_family_row.get("family_monopoly_top_family_id", "") or "")
                            chosen_entry["family_monopoly_unique_count"] = int(chosen_family_row.get("family_monopoly_unique_count", 0) or 0)
                            chosen_entry["monopoly_canonical_force_applied"] = bool(
                                chosen_family_row.get("monopoly_canonical_force_applied", False)
                            )
                            chosen_entry["family_prior_eligible"] = bool(
                                chosen_family_row.get("prior_eligible", False)
                            )
                            chosen_entry["family_prior_eligibility_reason"] = str(
                                chosen_family_row.get("prior_eligibility_reason", "") or ""
                            )
                            chosen_entry["family_competition_eligible"] = bool(
                                chosen_family_row.get("competition_eligible", False)
                            )
                            chosen_entry["family_competition_eligibility_reason"] = str(
                                chosen_family_row.get("competition_eligibility_reason", "") or ""
                            )
                            chosen_entry["family_bootstrap_included"] = bool(
                                chosen_family_row.get("bootstrap_included", False)
                            )
                            chosen_entry["family_bootstrap_competition_included"] = bool(
                                chosen_family_row.get("bootstrap_competition_included", chosen_family_row.get("bootstrap_included", False))
                            )
                            chosen_entry["family_catastrophic_prior"] = bool(
                                chosen_family_row.get("catastrophic_prior", False)
                            )
                            # Backward-compatible aliases.
                            chosen_entry["family_eligible"] = bool(
                                chosen_family_row.get("competition_eligible", False)
                            )
                            chosen_entry["family_eligibility_reason"] = str(
                                chosen_family_row.get("competition_eligibility_reason", "") or ""
                            )
                            chosen_entry["local_bracket_adaptation_mode"] = family_result.get(
                                "local_bracket_adaptation_mode",
                                chosen_family_row.get("local_bracket_adaptation_mode"),
                            )
                            chosen_entry["local_bracket_adaptation_enabled"] = family_result.get(
                                "local_bracket_adaptation_enabled",
                                chosen_family_row.get("local_bracket_adaptation_enabled"),
                            )
                            chosen_entry["local_bracket_override_applied"] = family_result.get(
                                "local_bracket_override_applied",
                                chosen_family_row.get("local_bracket_override_applied"),
                            )
                            chosen_entry["local_member_count_within_family"] = chosen_family_row.get(
                                "local_member_count_within_family"
                            )
                            chosen_entry["local_edge_component"] = chosen_family_row.get(
                                "local_edge_component"
                            )
                            chosen_entry["local_structural_component"] = chosen_family_row.get(
                                "local_structural_component"
                            )
                            chosen_entry[
                                "local_bracket_suitability_component"
                            ] = chosen_family_row.get("local_bracket_suitability_component")
                            chosen_entry["local_confidence_component"] = chosen_family_row.get(
                                "local_confidence_component"
                            )
                            chosen_entry["local_payoff_component"] = chosen_family_row.get(
                                "local_payoff_component"
                            )
                            chosen_entry["local_final_member_score"] = chosen_family_row.get(
                                "local_final_member_score"
                            )
                            chosen_entry["canonical_fallback_used"] = chosen_family_row.get(
                                "canonical_fallback_used"
                            )
                            chosen_entry["why_non_anchor_beat_anchor"] = chosen_family_row.get(
                                "why_non_anchor_beat_anchor"
                            )
                            chosen_entry["why_anchor_forced"] = chosen_family_row.get(
                                "why_anchor_forced"
                            )
                            chosen_entry["no_local_alternative"] = chosen_family_row.get(
                                "no_local_alternative"
                            )
                            chosen_entry["family_member_count"] = int(
                                chosen_family_row.get("inventory_member_count", 0) or 0
                            )
                            chosen_entry["feasible_family_count"] = int(len(family_rows))
                            chosen_entry["feasible_family_ids"] = ",".join(
                                [str(row.get("family_id", "")) for row in family_rows]
                            )
                            chosen_entry["family_context_inputs"] = dict(context_inputs or {})
                            chosen_entry["family_artifact"] = self._de3_v3_family_status.get("family_artifact_path")
                            chosen_entry["member_local_score"] = (
                                float(chosen_member_local_score)
                                if chosen_member_local_score is not None
                                else chosen_entry.get("de3_member_local_score")
                            )
                        chosen_cand = chosen_entry.get("cand") if isinstance(chosen_entry.get("cand"), dict) else {}
                        if isinstance(chosen_cand, dict):
                            self._apply_v3_family_metrics_to_candidate(
                                chosen_cand,
                                family_id=chosen_family_id,
                                family_score=chosen_family_score,
                                member_local_score=chosen_member_local_score,
                                family_components=chosen_family_components,
                            )
                            chosen_cand["de3_family_context_inputs"] = dict(context_inputs or {})
                            chosen_cand["de3_feasible_family_count"] = int(len(family_rows))
                            chosen_cand["de3_feasible_family_ids"] = [
                                str(row.get("family_id", "")) for row in family_rows
                            ]
                            chosen_cand["de3_family_usability_state"] = chosen_entry.get("family_usability_state")
                            chosen_cand["de3_family_competition_status"] = chosen_entry.get("family_competition_status")
                            chosen_cand["de3_family_evidence_support_tier"] = chosen_entry.get("family_evidence_support_tier")
                            chosen_cand["de3_family_usability_adjustment"] = chosen_entry.get("family_usability_adjustment")
                            chosen_cand["de3_family_prior_eligible"] = chosen_entry.get("family_prior_eligible")
                            chosen_cand["de3_family_competition_eligible"] = chosen_entry.get("family_competition_eligible")
                            chosen_cand["de3_family_bootstrap_included"] = chosen_entry.get("family_bootstrap_included")
                            chosen_cand["de3_family_base_score"] = chosen_entry.get("base_family_score")
                            chosen_cand["de3_family_diversity_adjustment"] = chosen_entry.get("diversity_adjustment")
                            chosen_cand["de3_family_recent_chosen_share"] = chosen_entry.get("recent_chosen_share")
                            chosen_cand["de3_family_monopoly_active"] = chosen_entry.get("family_monopoly_active")
                            chosen_cand["de3_family_monopoly_top_share"] = chosen_entry.get("family_monopoly_top_share")
                            chosen_cand["de3_family_context_support_tier"] = chosen_entry.get(
                                "family_context_support_tier"
                            )
                            chosen_cand["de3_local_bracket_adaptation_mode"] = chosen_entry.get(
                                "local_bracket_adaptation_mode"
                            )
                            chosen_cand["de3_local_bracket_override_applied"] = chosen_entry.get(
                                "local_bracket_override_applied"
                            )
                        chosen_decision_id = self._emit_de3_v3_family_decision_journal(
                            current_time=current_time,
                            session_name=str(engine_session or ""),
                            family_rows=family_rows,
                            chosen_family_id=chosen_family_id,
                            chosen_entry=chosen_entry,
                            chosen_member_local_score=chosen_member_local_score,
                            context_inputs=context_inputs,
                            abstained=False,
                            abstain_reason="",
                        )
                    elif self._de3_v2_runtime:
                        # Stage 1: edge-only ordering to compute local ambiguity (top-vs-next gap).
                        edge_sorted = sorted(
                            feasible_candidates,
                            key=lambda x: float(x.get("edge_points", float("-inf"))),
                            reverse=True,
                        )
                        edge_gap_map: Dict[int, Tuple[Optional[float], float]] = {}
                        for i, entry in enumerate(edge_sorted):
                            next_edge = (
                                float(edge_sorted[i + 1].get("edge_points", 0.0) or 0.0)
                                if i + 1 < len(edge_sorted)
                                else None
                            )
                            edge_val = float(entry.get("edge_points", 0.0) or 0.0)
                            edge_gap = float(edge_val - next_edge) if next_edge is not None else 0.0
                            edge_gap_map[id(entry)] = (next_edge, edge_gap)

                        # Stage 2: compute concentration term once (no nested loops).
                        bucket_counts: Dict[Tuple[str, str], int] = {}
                        for entry in feasible_candidates:
                            cand = entry.get("cand") or {}
                            bucket = (
                                str(cand.get("timeframe", "")),
                                str(cand.get("strategy_type", "")),
                            )
                            bucket_counts[bucket] = int(bucket_counts.get(bucket, 0) + 1)
                        denom = max(1, len(feasible_candidates) - 1)

                        # Stage 3: compute robust runtime rank per candidate.
                        for entry in feasible_candidates:
                            cand = entry.get("cand") or {}
                            next_edge, edge_gap = edge_gap_map.get(id(entry), (None, 0.0))
                            bucket = (
                                str(cand.get("timeframe", "")),
                                str(cand.get("strategy_type", "")),
                            )
                            entry["family_key"] = self._de3_v2_same_family_key(
                                cand,
                                default_session=engine_session,
                            )
                            bucket_concentration = max(
                                0.0,
                                float(bucket_counts.get(bucket, 1) - 1) / float(denom),
                            )
                            runtime_rank, rank_components = self._compute_v2_runtime_rank(
                                cand,
                                edge_points=float(entry.get("edge_points", 0.0) or 0.0),
                                second_best_edge_points=next_edge,
                                confidence=float(entry.get("edge_confidence", 0.5) or 0.5),
                                bucket_concentration=bucket_concentration,
                            )
                            entry["edge_gap"] = float(edge_gap)
                            entry["bucket_concentration"] = float(bucket_concentration)
                            entry["runtime_rank_score"] = float(runtime_rank)
                            entry["runtime_rank_components"] = rank_components
                            entry["structural_score"] = float(rank_components.get("structural_score", entry.get("structural_score", 0.0)))
                            entry["structural_pass"] = bool(entry.get("structural_pass", True))

                            cand["de3_runtime_rank_score"] = float(runtime_rank)
                            cand["de3_v2_rank_score"] = float(runtime_rank)
                            cand["de3_edge_gap_points"] = float(edge_gap)
                            cand["de3_bucket_concentration"] = float(bucket_concentration)
                            cand["de3_structural_score"] = float(entry.get("structural_score", 0.0) or 0.0)
                            cand["de3_structural_pass"] = bool(entry.get("structural_pass", True))

                        # Stage 4: final sort by runtime rank (soft trust via score).
                        feasible_candidates.sort(
                            key=lambda x: (
                                float(x.get("runtime_rank_score", float("-inf"))),
                                float(x.get("edge_points", float("-inf"))),
                                float(x.get("structural_score", float("-inf"))),
                                -int(x.get("idx", 0)),
                            ),
                            reverse=True,
                        )

                        if self._de3_v2_runtime_log_decisions:
                            top_k = min(self._de3_v2_runtime_log_top_k, len(feasible_candidates))
                            for i, entry in enumerate(feasible_candidates[:top_k], 1):
                                logging.info(
                                    (
                                        "DE3 v2 rank #%s %s | edge=%.3f gap=%.3f struct=%.3f pass=%s "
                                        "final=%.3f rank=%.3f wb_avg=%.3f pbr=%.3f stop=%.3f loss=%.3f chosen=%s"
                                    ),
                                    i,
                                    str(entry.get("cand_id", "")),
                                    float(entry.get("edge_points", 0.0) or 0.0),
                                    float(entry.get("edge_gap", 0.0) or 0.0),
                                    float(entry.get("structural_score", 0.0) or 0.0),
                                    bool(entry.get("structural_pass", False)),
                                    float((entry.get("cand") or {}).get("final_score", 0.0) or 0.0),
                                    float(entry.get("runtime_rank_score", 0.0) or 0.0),
                                    float(entry.get("worst_block_avg_pnl", 0.0) or 0.0),
                                    float(entry.get("profitable_block_ratio", 0.0) or 0.0),
                                    float(entry.get("stop_like_share", 0.0) or 0.0),
                                    float(entry.get("loss_share", 0.0) or 0.0),
                                    bool(i == 1),
                                )
                    else:
                        feasible_candidates.sort(
                            key=lambda x: (
                                float(x.get("selection_score", float("-inf"))),
                                float(x.get("edge_points", float("-inf"))),
                                -int(x.get("idx", 0)),
                            ),
                            reverse=True,
                        )
                    # v2/v1 fallback ranking path only. Keep DE3v4 chosen_entry intact.
                    if (not self._de3_v3_runtime) and (not self._de3_v4_runtime):
                        chosen_entry = feasible_candidates[0]
                        if self._de3_v2_runtime:
                            chosen_entry = self._de3_v2_apply_same_family_near_tie_override(
                                feasible_candidates,
                                engine_session=engine_session,
                                current_time=current_time,
                            )
                        best_edge = float(chosen_entry.get("edge_points", 0.0) or 0.0)
                        if self._de3_v2_runtime and self._de3_v2_runtime_abstain_enabled:
                            best_edge_gap = float(chosen_entry.get("edge_gap", 0.0) or 0.0)
                            best_struct = float(chosen_entry.get("structural_score", 0.0) or 0.0)
                            best_runtime_rank = float(chosen_entry.get("runtime_rank_score", 0.0) or 0.0)
                            abstain_reason = None
                            if best_edge < float(self._de3_v2_runtime_min_edge):
                                abstain_reason = (
                                    f"edge {best_edge:.3f} < {float(self._de3_v2_runtime_min_edge):.3f}"
                                )
                            elif best_edge_gap < float(self._de3_v2_runtime_min_gap):
                                abstain_reason = (
                                    f"edge_gap {best_edge_gap:.3f} < {float(self._de3_v2_runtime_min_gap):.3f}"
                                )
                            elif best_struct < float(self._de3_v2_runtime_min_struct):
                                abstain_reason = (
                                    f"structural {best_struct:.3f} < {float(self._de3_v2_runtime_min_struct):.3f}"
                                )
                            elif best_runtime_rank < float(self._de3_v2_runtime_min_rank):
                                abstain_reason = (
                                    f"runtime_rank {best_runtime_rank:.3f} < {float(self._de3_v2_runtime_min_rank):.3f}"
                                )
                            if abstain_reason:
                                chosen_decision_id = self._emit_de3_decision_journal(
                                    current_time=current_time,
                                    session_name=str(engine_session or ""),
                                    feasible_candidates=feasible_candidates,
                                    chosen_cand_id=None,
                                    abstained=True,
                                    abstain_reason=abstain_reason,
                                )
                                if self._de3_v2_runtime_log_decisions or self._log_selection_details:
                                    logging.info("DE3 v2 abstain: %s", abstain_reason)
                                return None
                        else:
                            best_score_key = "selection_score"
                            best_score = float(chosen_entry.get(best_score_key, 0.0) or 0.0)
                            second_score = None
                            if len(feasible_candidates) > 1:
                                second_score = float(feasible_candidates[1].get(best_score_key, 0.0) or 0.0)
                            if best_edge < float(min_edge_points):
                                if self._v1_specific_filters_enabled:
                                    self._warn(
                                        "🚫 DynamicEngine3: best feasible edge %.3f < min_edge_points %.3f",
                                        best_edge,
                                        float(min_edge_points),
                                    )
                                    return None
                            if second_score is not None and (best_score - second_score) < float(min_score_gap_points):
                                if self._v1_specific_filters_enabled:
                                    self._warn(
                                        "🚫 DynamicEngine3: best-vs-second edge gap %.3f < %.3f",
                                        float(best_score - second_score),
                                        float(min_score_gap_points),
                                    )
                                    return None
                else:
                    chosen_entry = min(feasible_candidates, key=lambda x: int(x.get("idx", 0)))

                if self._de3_v2_runtime:
                    chosen_decision_id = self._emit_de3_decision_journal(
                        current_time=current_time,
                        session_name=str(engine_session or ""),
                        feasible_candidates=feasible_candidates,
                        chosen_cand_id=str(chosen_entry.get("cand_id", "") or ""),
                        abstained=False,
                        abstain_reason="",
                    )
                elif self._de3_v3_runtime and chosen_decision_id is None:
                    chosen_decision_id = self._emit_de3_decision_journal(
                        current_time=current_time,
                        session_name=str(engine_session or ""),
                        feasible_candidates=feasible_candidates,
                        chosen_cand_id=str(chosen_entry.get("cand_id", "") or ""),
                        abstained=False,
                        abstain_reason="",
                    )

                chosen = chosen_entry.get("cand")
                chosen_exec = chosen_entry.get("exec")
                try:
                    chosen_idx = int(chosen_entry.get("idx", 0))
                except Exception:
                    chosen_idx = 0

                if (
                    chosen_idx > 0
                    and top_blocked_id
                    and top_blocked_reason
                    and self._log_selection_details
                ):
                    logging.info(
                        "DynamicEngine3 fallback: top=%s blocked (%s) -> using %s",
                        top_blocked_id,
                        top_blocked_reason,
                        str(chosen_entry.get("cand_id", "")),
                    )
                elif (
                    bool(selection_enabled)
                    and bool(log_rerank)
                    and bool(self._log_selection_details)
                    and top_candidate_id is not None
                    and str(chosen_entry.get("cand_id", "")) != str(top_candidate_id)
                ):
                    logging.info(
                        "DynamicEngine3 re-rank: top=%s -> selected=%s (edge=%.3f score=%.3f)",
                        str(top_candidate_id),
                        str(chosen_entry.get("cand_id", "")),
                        float(chosen_entry.get("edge_points", 0.0) or 0.0),
                        float(
                            chosen_entry.get(
                                "family_score",
                                chosen_entry.get(
                                    "runtime_rank_score",
                                    chosen_entry.get("selection_score", 0.0),
                                ),
                            )
                            or 0.0
                        ),
                    )
            if chosen is None:
                if top_blocked_id and top_blocked_reason:
                    self._warn(
                        "🚫 DynamicEngine3: all %s candidates blocked (top=%s: %s)",
                        len(candidates),
                        top_blocked_id,
                        top_blocked_reason,
                    )
                return None

            signal_data = chosen
            if chosen_exec is None:
                # Defensive fallback: selection loop should always set this for chosen candidate.
                self._warn("🚫 DynamicEngine3 internal selection error: missing execution payload")
                return None

            final_sl = float(chosen_exec["final_sl"])
            final_tp = float(chosen_exec["final_tp"])
            num_contracts = int(chosen_exec["num_contracts"])
            policy_risk_mult = float(chosen_exec["policy_risk_mult"])
            gross_profit = float(chosen_exec["gross_profit"])
            total_fees = float(chosen_exec["total_fees"])
            net_profit = float(chosen_exec["net_profit"])
            override_applied = False
            override_sl = final_sl
            override_tp = final_tp
            if self._de3_v2_runtime and self._de3_v2_bucket_bracket_overrides:
                strategy_id = str(signal_data.get("strategy_id", "") or "").strip()
                if strategy_id:
                    override_vals = self._de3_v2_bucket_bracket_overrides.get(strategy_id)
                    if override_vals is not None:
                        try:
                            ov_sl, ov_tp = override_vals
                            ov_sl = float(ov_sl)
                            ov_tp = float(ov_tp)
                            if np.isfinite(ov_sl) and np.isfinite(ov_tp) and ov_sl > 0.0 and ov_tp > 0.0:
                                override_applied = True
                                override_sl = float(ov_sl)
                                override_tp = float(ov_tp)
                                final_sl = override_sl
                                final_tp = override_tp
                                base_tp = float(chosen_exec.get("final_tp", 0.0) or 0.0)
                                base_size = max(1.0, float(num_contracts))
                                if base_tp > 0.0:
                                    point_value_eff = float(chosen_exec.get("gross_profit", 0.0) or 0.0) / (
                                        base_tp * base_size
                                    )
                                else:
                                    point_value_eff = 0.0
                                fees_eff = float(chosen_exec.get("total_fees", 0.0) or 0.0)
                                gross_profit = float(final_tp * point_value_eff * base_size)
                                total_fees = fees_eff
                                net_profit = float(gross_profit - total_fees)
                        except Exception:
                            pass
            signal_size_requested = int(num_contracts)
            signal_size_final = int(num_contracts)
            signal_size_applied = False
            signal_size_rule_names: List[str] = []
            signal_size_rule_reasons: List[Dict[str, Any]] = []
            if self._de3_v4_runtime:
                signal_size_final, signal_size_rule_reasons = self._de3_v4_apply_signal_size_rules(
                    signal_data,
                    current_time=current_time,
                    engine_session=str(engine_session or ""),
                    requested_size=int(num_contracts),
                )
                signal_size_applied = bool(int(signal_size_final) != int(signal_size_requested))
                signal_size_rule_names = [
                    str(item.get("name", "") or "")
                    for item in signal_size_rule_reasons
                    if isinstance(item, dict) and str(item.get("name", "") or "").strip()
                ]
                if signal_size_applied:
                    num_contracts = int(signal_size_final)
                    gross_profit = float(final_tp * point_value * num_contracts)
                    total_fees = float(fees_per_side * 2 * num_contracts)
                    net_profit = float(gross_profit - total_fees)
            signal_data["de3_signal_size_rules_requested_size"] = int(signal_size_requested)
            signal_data["de3_signal_size_rules_final_size"] = int(signal_size_final)
            signal_data["de3_signal_size_rules_applied"] = bool(signal_size_applied)
            signal_data["de3_signal_size_rule_names"] = list(signal_size_rule_names)
            signal_data["de3_signal_size_rule_reasons"] = list(signal_size_rule_reasons)
            signal_tf = str(signal_data.get("timeframe", "5min") or "5min").lower()
            signal_id = str(signal_data.get("strategy_id", "") or "")
            signal_side = str(signal_data.get("signal", "") or "").upper()
            source_df = df_15m if signal_tf.startswith("15") else df_5m
            drift_ok, drift_ctx = self._passes_drift_gate(
                strategy_id=signal_id,
                side=signal_side,
                signal_tf=signal_tf,
                current_time=current_time,
                current_price=float(df["close"].iloc[-1]),
                df_1m=df,
                df_tf=source_df,
            )
            if not drift_ok:
                if self._log_selection_details:
                    logging.info(
                        "DynamicEngine3 drift block: %s %s dist_atr=%.2f > %.2f",
                        signal_id,
                        signal_side,
                        float(drift_ctx.get("de3_drift_dist_atr", 0.0) or 0.0),
                        float(self._drift_max_atr),
                    )
                return None

            if self._log_signal_emits:
                logging.info("DynamicEngine3: %s signal from %s", signal_data["signal"], signal_data["strategy_id"])
                logging.info(
                    "   TP: %.2f | SL: %.2f | size=%s | risk_mult=%.2f",
                    final_tp,
                    final_sl,
                    num_contracts,
                    policy_risk_mult,
                )
                logging.info(
                    "   Net profit: $%.2f (gross $%.2f - fees $%.2f)",
                    net_profit,
                    gross_profit,
                    total_fees,
                )

            signal_out = {
                "strategy": "DynamicEngine3",
                "sub_strategy": signal_data["strategy_id"],
                "side": signal_data["signal"],
                "tp_dist": final_tp,
                "sl_dist": final_sl,
                "size": num_contracts,
            }
            # Preserve core DE3 candidate metadata for post-backtest diagnostics.
            for key in ("strategy_type", "timeframe", "thresh", "opt_wr", "score_raw", "final_score", "trades", "avg_pnl"):
                if key in signal_data:
                    signal_out[f"de3_{key}"] = signal_data.get(key)
            for key in ("de3_selection_score", "de3_edge_points", "de3_edge_proxy_points", "de3_edge_confidence"):
                if key in signal_data:
                    signal_out[key] = signal_data.get(key)
            if "de3_v2_rank_score" in signal_data:
                signal_out["de3_v2_rank_score"] = signal_data.get("de3_v2_rank_score")
            for key in (
                "de3_runtime_rank_score",
                "de3_edge_gap_points",
                "de3_bucket_concentration",
                    "de3_structural_score",
                    "de3_structural_pass",
                ):
                if key in signal_data:
                    signal_out[key] = signal_data.get(key)
            for key in (
                "de3_signal_size_rules_requested_size",
                "de3_signal_size_rules_final_size",
                "de3_signal_size_rules_applied",
                "de3_signal_size_rule_names",
                "de3_signal_size_rule_reasons",
            ):
                if key in signal_data:
                    signal_out[key] = signal_data.get(key)
            for key in (
                "de3_family_id",
                "de3_family_score",
                "de3_family_context_ev",
                "de3_family_confidence",
                "de3_family_prior",
                "de3_family_profile",
                "de3_family_context_support_ratio",
                "de3_family_adaptive_component",
                "de3_member_local_score",
                "de3_feasible_family_count",
                "de3_feasible_family_ids",
            ):
                if key in signal_data:
                    signal_out[key] = signal_data.get(key)
            if isinstance(signal_data.get("de3_family_context_inputs"), dict):
                signal_out["de3_family_context_inputs"] = dict(signal_data.get("de3_family_context_inputs"))
            if self._de3_v2_runtime:
                signal_out["de3_version"] = "v2"
            elif self._de3_v3_runtime:
                signal_out["de3_version"] = "v3"
                signal_out["de3_family_mode"] = True
            elif self._de3_v4_runtime:
                signal_out["de3_version"] = "v4"
                signal_out["de3_v4_mode"] = str(signal_data.get("de3_v4_runtime_mode", ""))
            for key in DE3_V4_EXPORT_FIELDS:
                if key in signal_data:
                    signal_out[key] = signal_data.get(key)
            if chosen_decision_id:
                signal_out["de3_decision_id"] = str(chosen_decision_id)
            chosen_cand = signal_data.get("cand", {}) if isinstance(signal_data.get("cand"), dict) else {}
            for source in (signal_data, chosen_cand):
                if not isinstance(source, dict):
                    continue
                for key, value in source.items():
                    if str(key).startswith("de3_entry_") and key not in signal_out:
                        signal_out[key] = value
            if override_applied:
                signal_out["de3_bracket_override_applied"] = True
                signal_out["de3_bracket_override_sl"] = float(override_sl)
                signal_out["de3_bracket_override_tp"] = float(override_tp)
                signal_out["de3_bracket_original_sl"] = float(chosen_exec.get("final_sl", override_sl))
                signal_out["de3_bracket_original_tp"] = float(chosen_exec.get("final_tp", override_tp))
            for key in (
                "de3_policy_allow",
                "de3_policy_reason",
                "de3_policy_ev_points",
                "de3_policy_ev_lcb_points",
                "de3_policy_ev_ucb_points",
                "de3_policy_p_loss",
                "de3_policy_p_loss_std",
                "de3_policy_p_loss_lcb",
                "de3_policy_p_loss_ucb",
                "de3_policy_confidence",
                "de3_policy_risk_mult",
                "de3_policy_bucket",
                "de3_policy_bucket_samples",
                "de3_policy_level",
                "de3_policy_would_block",
                "de3_v4_profit_gate_soft_pass",
                "de3_v4_profit_gate_catastrophic_block",
            ):
                if key in signal_data:
                    signal_out[key] = signal_data.get(key)
            for key, value in signal_data.items():
                if str(key).startswith("de3_entry_"):
                    signal_out[key] = value
            for key, value in signal_data.items():
                if str(key).startswith("de3_v2_entry_"):
                    signal_out[key] = value
            # Preserve veto diagnostics for downstream trade-log counterfactual analysis.
            for key, value in signal_data.items():
                if str(key).startswith("de3_veto_"):
                    signal_out[key] = value
            if drift_ctx:
                signal_out.update(drift_ctx)
            return signal_out

        return None
